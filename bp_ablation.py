#!/usr/bin/env python3
"""Paired replay: does removing the team bullpen term change any decision?

The v9/v10 matchup value is a two-phase blend, weighted by each phase's share
of plate appearances (`sequential_xwoba_phases` in build_site.py):

    mx = q*mx_sp + (1-q)*mx_bp        edge = mx - L        xw_net = edge_away - edge_home

Ablating the bullpen term means q = 1, i.e. the model becomes the starter
matchup alone and `edge` collapses to `edge_xwOBA_sp`. Both quantities are
already persisted per row, so both variants are read out of the ledger rather
than recomputed:

    with-BP (shipped) : edge_xwoba_away    - edge_xwoba_home
    no-BP   (ablated) : edge_xwoba_sp_away - edge_xwoba_sp_home

No Savant re-pull, no lookahead. Every input is a value that was written at
build time for that slate. The with-BP reconstruction is asserted against the
shipped `xw_lean` on every row, so a silent formula drift fails the run rather
than quietly reporting against the wrong baseline.

Design notes:

  * This is a PAIRED test. Same games, two variants. Unpaired comparison is
    swamped by game-level noise.

  * The informative subsample is the DISAGREEMENT SET -- games where the two
    variants lean opposite sides, or where one abstains and the other does not.
    Where both variants agree, CLV is identical by construction and those rows
    only shrink the paired mean toward zero. Both are reported.

  * POWER IS REPORTED BEFORE SIGNIFICANCE, and a null is not a result on its
    own. It is only evidence of "no effect" once the minimum detectable effect
    is smaller than the effect you would act on. Read `n needed` first.

  * CLV uses the price actually available at lean time: the de-vigged opening
    moneyline as entry, the de-vigged close as exit. Both markets are scored --
    the bullpen term should matter more full-game than F5, and an ablation that
    moves F5 as much as it moves the full game is suspect.

Usage:
  python bp_ablation.py                       # defaults to data/mlb_lean_ledger.csv
  python bp_ablation.py --ledger PATH
  python bp_ablation.py --care-about 0.005    # effect size you would act on
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

BOOT_SEED = 20260801
BOOT_DRAWS = 10000
DEFAULT_LEDGER = "data/mlb_lean_ledger.csv"

# Phase decomposition, written by build_site.py from v9 onward. Rows predating
# it cannot be ablated -- the bullpen phase was not a separate term to remove.
PHASE_COLS = (
    "edge_xwoba_away", "edge_xwoba_home",
    "edge_xwoba_sp_away", "edge_xwoba_sp_home",
)

MARKETS = {
    "full": ("open_home_ml", "open_away_ml", "close_p_home", "full_away", "full_home", "xw_full"),
    "F5": ("f5_open_home_ml", "f5_open_away_ml", "f5_close_p_home", "f5_away", "f5_home", "xw_f5"),
}


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def ml_to_prob(ml):
    """American moneyline -> raw implied probability (vig still in)."""
    ml = ml.astype(float)
    return pd.Series(
        np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0)),
        index=ml.index,
    )


def devig(p_home, p_away):
    """Proportional (multiplicative) de-vig -- the convention close_p_home uses."""
    total = p_home + p_away
    return p_home.where(total.isna() | total.eq(0), p_home / total)


# ------------------------------------------------------------------ variants ---

def variants(d):
    """Both model variants, read from persisted phase values.

    Returns (with_bp_net, no_bp_net, eligible_mask). Aborts if the with-BP
    reconstruction disagrees with the shipped lean on any eligible row.
    """
    missing = [c for c in PHASE_COLS if c not in d.columns]
    if missing:
        sys.exit(f"ledger lacks the phase decomposition: {missing}\n"
                 f"only v9+ rows carry it; there is nothing to ablate without it")
    for col in ("xw_net", "xw_lean", "home", "away"):
        if col not in d.columns:
            sys.exit(f"ledger lacks '{col}' -- cannot verify the with-BP baseline")

    with_bp = _num(d, "edge_xwoba_away") - _num(d, "edge_xwoba_home")
    no_bp = _num(d, "edge_xwoba_sp_away") - _num(d, "edge_xwoba_sp_home")
    eligible = with_bp.notna() & no_bp.notna()

    # The shipped xw_net must equal the reconstruction; otherwise the phase
    # columns and the lean have drifted apart and the baseline is not the model.
    shipped = _num(d, "xw_net")
    drift = (with_bp - shipped).abs()
    bad = eligible & drift.gt(1e-9)
    if bad.any():
        sys.exit(f"with-BP reconstruction != shipped xw_net on {int(bad.sum())} row(s); "
                 f"max |diff| {drift[eligible].max():.3e} -- refusing to report")

    lean = pd.Series(
        np.where(with_bp > 0, d.get("home"), np.where(with_bp < 0, d.get("away"), None)),
        index=d.index,
    )
    mismatch = eligible & d.get("xw_lean").notna() & lean.ne(d.get("xw_lean"))
    if mismatch.any():
        sys.exit(f"reconstructed lean != shipped xw_lean on {int(mismatch.sum())} row(s) "
                 f"-- refusing to report")

    return with_bp, no_bp, eligible


def side_of(net):
    """Sign -> side. Exactly zero is abstention, per the v7 convention."""
    return pd.Series(
        np.where(net > 0, "HOME", np.where(net < 0, "AWAY", "")),
        index=net.index,
    )


# ------------------------------------------------------------------- metrics ---

def clv(side, entry_home, close_home):
    """Signed closing-line value in probability points on the leaned side.

    Positive => the market moved toward the side taken. Abstentions contribute
    0: no position, no CLV.
    """
    val = pd.Series(0.0, index=side.index)
    home, away = side.eq("HOME"), side.eq("AWAY")
    val[home] = (close_home - entry_home)[home]
    # The away price is the complement, so its move is the negated home move.
    val[away] = (entry_home - close_home)[away]
    return val


def grade(side, away_score, home_score):
    """W/L/T for the leaned side, scored from the linescore directly."""
    out = pd.Series(pd.NA, index=side.index, dtype="object")
    have = away_score.notna() & home_score.notna() & side.ne("")
    margin = (home_score - away_score).where(have)
    signed = margin.where(side.eq("HOME"), -margin)
    out[have & signed.gt(0)] = "W"
    out[have & signed.lt(0)] = "L"
    out[have & signed.eq(0)] = "T"
    return out


def record(grades):
    g = grades.dropna()
    w, l, t = (g == "W").sum(), (g == "L").sum(), (g == "T").sum()
    pct = w / (w + l) if (w + l) else float("nan")
    return w, l, t, pct


def paired_report(delta, name, care_about):
    """Paired test with power stated before significance."""
    d = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(d)
    print(f"  {name}")
    if n < 2:
        print(f"    n paired          : {n}  -- too few to test")
        return
    if np.allclose(d, 0):
        print(f"    n paired          : {n}  -- variants identical on every row")
        return

    mean, sd = d.mean(), d.std(ddof=1)
    # Two-sided alpha=.05 at 80% power: (1.96 + 0.84) = 2.8 standard errors.
    mde = 2.8 * sd / np.sqrt(n)
    needed = int(np.ceil((2.8 * sd / care_about) ** 2)) if care_about > 0 else 0

    print(f"    n paired          : {n}")
    print(f"    mean delta        : {mean:+.5f} prob pts  (no-BP minus with-BP)")

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.choice(d, size=(BOOT_DRAWS, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"    95% bootstrap CI  : [{lo:+.5f}, {hi:+.5f}]")

    print(f"    MDE @80% power    : {mde:.5f}   <-- READ THIS FIRST")
    print(f"    to detect {care_about:.5f} : need n ~ {needed} (have {n})")

    if stats is not None:
        _, p_t = stats.ttest_1samp(d, 0.0)
        try:
            # zeros carry no information about direction; drop before ranking
            nz = d[d != 0]
            p_w = stats.wilcoxon(nz)[1] if len(nz) >= 2 else float("nan")
        except ValueError:
            p_w = float("nan")
        print(f"    paired t p        : {p_t:.4f}   wilcoxon p: {p_w:.4f} (n={len(d[d != 0])} nonzero)")
    else:
        print("    (scipy absent -- p-values skipped; CI and MDE stand on their own)")

    if mde > care_about:
        print(f"    verdict           : UNDERPOWERED. Cannot see a {care_about:.5f} effect "
              f"at this n.")
        print(f"                        A null here is not evidence the term is useless.")
    elif abs(mean) < mde:
        print(f"    verdict           : powered, and no effect of actionable size detected")
    else:
        winner = "no-BP" if mean > 0 else "with-BP"
        print(f"    verdict           : {winner} variant captured more CLV")


def absorption(no_bp, bp_part, close_home, mask):
    """Does the closing line move with the bullpen term?

    Regress the market's de-vigged home probability on the ablated model and on
    the bullpen contribution that the ablation removes.

    Reading the coefficient on bp_part:
      significantly positive -> the close moves WITH the bullpen term, so the
                                market already reflects it; removing the term
                                costs no market-recognised signal.
      indistinguishable from 0 -> the close does NOT move with it. That is NOT
                                evidence the market priced it -- it is the
                                opposite. The term is either noise or signal the
                                market has not found, and this regression alone
                                cannot tell those apart; that needs outcomes.
    """
    x1 = no_bp[mask].to_numpy(dtype=float)
    x2 = bp_part[mask].to_numpy(dtype=float)
    y = close_home[mask].to_numpy(dtype=float)
    if len(y) < 4:
        print("  too few rows to regress")
        return
    if np.allclose(x2, 0):
        print("  bullpen contribution is identically zero -- nothing to regress")
        return

    X = np.column_stack([np.ones_like(x1), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * sigma2))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan")

    print(f"  close_p_home ~ nobp_net + bp_contribution     (n={len(y)}, R^2={r2:.3f})")
    print("  coefficients are probability points per xwOBA point")
    for nm, b, s in zip(("intercept", "nobp_net", "bp_contrib"), beta, se):
        t = b / s if s > 0 else float("nan")
        print(f"    {nm:<11} {b:+9.4f}  (se {s:.4f}, t {t:+.2f})")

    t_bp = beta[2] / se[2] if se[2] > 0 else float("nan")
    if abs(t_bp) >= 2:
        print("  -> the close moves with the bullpen term: the market reflects it.")
        print("     Caveat: bullpen quality correlates with team quality, which the")
        print("     market also prices. This cannot separate 'the market prices the")
        print("     bullpen matchup' from 'the bullpen term proxies team strength'.")
    else:
        print("  -> the close does NOT move with the bullpen term. This is not evidence")
        print("     the market priced it -- it means the term is noise OR unpriced alpha,")
        print("     and this regression cannot separate those. Outcomes can.")


# -------------------------------------------------------------------- driver ---

def run(ledger_path, care_about):
    try:
        d = pd.read_csv(ledger_path)
    except FileNotFoundError:
        sys.exit(f"no ledger at {ledger_path}")

    with_bp, no_bp, eligible = variants(d)
    side_w, side_n = side_of(with_bp), side_of(no_bp)
    disagree = eligible & side_w.ne(side_n)
    bp_part = with_bp - no_bp

    print("=" * 70)
    print("BULLPEN-TERM ABLATION -- PAIRED REPLAY")
    print("=" * 70)
    print(f"ledger              : {ledger_path}  ({len(d)} rows)")
    print(f"ablatable (v9+)     : {int(eligible.sum())}  "
          f"({int((~eligible).sum())} rows predate the phase split)")
    if eligible.any():
        tags = d.loc[eligible, "model_tag"].value_counts()
        print(f"  model_tag         : {', '.join(f'{k} n={v}' for k, v in tags.items())}")
    print(f"with-BP reconstruction verified against shipped xw_lean on all "
          f"{int(eligible.sum())} rows")

    print(f"\ndecision impact")
    print(f"  leans fired with-BP : {int((eligible & side_w.ne('')).sum())}")
    print(f"  leans fired no-BP   : {int((eligible & side_n.ne('')).sum())}")
    print(f"  DISAGREEMENT SET    : {int(disagree.sum())} "
          f"({disagree.sum() / max(int(eligible.sum()), 1):.1%} of ablatable games)")
    print(f"  median |bp contrib| : {bp_part[eligible].abs().median():.5f} xwOBA pts "
          f"(vs median |xw_net| {with_bp[eligible].abs().median():.5f})")
    if 0 < disagree.sum() < 30:
        print(f"  !! only {int(disagree.sum())} games can differ at all -- every CLV number")
        print(f"     below is drawn from that set and will be underpowered")

    for label, (o_h, o_a, c_col, a_sc, h_sc, shipped_grade) in MARKETS.items():
        entry = devig(ml_to_prob(_num(d, o_h)), ml_to_prob(_num(d, o_a)))
        close = _num(d, c_col)
        priced = eligible & entry.notna() & close.notna()
        delta = clv(side_n, entry, close) - clv(side_w, entry, close)

        print(f"\n{'-' * 70}")
        print(f"CLV -- {label} market   (entry = de-vigged open, exit = de-vigged close)")
        print(f"  priced rows: {int(priced.sum())} of {int(eligible.sum())} ablatable")
        # Two different estimands. The full sample answers "what does ablating
        # cost per slate game?" and is mostly structural zeros, which shrink the
        # spread and so flatter the power. The disagreement subset answers "what
        # does it cost per decision actually changed?" -- the larger effect, and
        # the one with almost no rows behind it.
        paired_report(delta[priced], "all priced games  [per-slate-game average]",
                      care_about)
        paired_report(delta[priced & disagree], "disagreement subset  [per changed decision]",
                      care_about)

        aw, hm = _num(d, a_sc), _num(d, h_sc)
        g_w, g_n = grade(side_w, aw, hm), grade(side_n, aw, hm)
        scored = eligible & g_w.notna()
        if shipped_grade in d.columns:
            clash = scored & d[shipped_grade].notna() & g_w.ne(d[shipped_grade])
            if clash.any():
                print(f"  !! scoring disagrees with shipped {shipped_grade} on "
                      f"{int(clash.sum())} row(s) -- grades below are suspect")
        if scored.any():
            w1, l1, t1, p1 = record(g_w[scored])
            w2, l2, t2, p2 = record(g_n[scored])
            print(f"  record on the same {int(scored.sum())} graded games")
            print(f"    with-BP : {w1}-{l1}" + (f"-{t1}" if t1 else "") + f"  ({p1:.3f})")
            print(f"    no-BP   : {w2}-{l2}" + (f"-{t2}" if t2 else "") + f"  ({p2:.3f})")
            n_diff = int((scored & disagree).sum())
            print(f"    the two records can only differ on the {n_diff} disagreement "
                  f"game(s); at that n a record gap is noise")

    print(f"\n{'-' * 70}")
    print("market absorption")
    close_ok = eligible & _num(d, "close_p_home").notna()
    absorption(no_bp, bp_part, _num(d, "close_p_home"), close_ok)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=DEFAULT_LEDGER,
                    help=f"ledger CSV (default {DEFAULT_LEDGER})")
    ap.add_argument("--care-about", type=float, default=0.005,
                    help="CLV effect size in probability points you would act on "
                         "(default 0.005 = half a point)")
    args = ap.parse_args()
    run(args.ledger, args.care_about)


if __name__ == "__main__":
    main()
