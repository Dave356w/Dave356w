"""Phase-matched peer benchmarks: what cancels, what changes, what the rows say.

The shipped v12 matchup value is one ratio against one denominator:

    M_SP = B_vsSP * SP / L        M_BP = B_neutral * BP / L
    mx   = q*M_SP + (1-q)*M_BP    edge = mx - L

where `L` is the PA-weighted league batter xwOBA -- the SAME constant for the
hitter ratio, the pitcher ratio and the baseline, in both phases. The natural
refinement is to give every ratio the benchmark of its own population:

    M_SP = E_SP * (B_vsSP / H_SP) * (SP / P_SP)
    M_BP = E_BP * (B_neutral / H_BP) * (BP / P_BP)

This module measures that change instead of arguing it. Read the closure
first: most of the variants anyone would preregister are the same predictor.

WHAT ARITHMETIC CLOSES

  *Peer-reporting only* -- computing separate league averages and displaying
  them, without touching the formula -- is a NO-OP on every lean, delta and
  grade by construction. It is not scored here because there is nothing to
  score; a diagnostic that does not enter `matchup_value` cannot move one.

  *The pitcher benchmark cancels out of M whenever E = P.* With the phase's
  expected level set to the level that phase's pitchers allow -- which is what
  it means for both to describe the same plate appearances -- `E_SP * SP/P_SP`
  is `SP` again, and the peer benchmark survives only in the BASELINE
  `E = q*E_SP + (1-q)*E_BP` that `edge` subtracts. So "phase-matched
  benchmarks" is not a third input; it is a re-centring, and its whole content
  is that the two phases are centred in DIFFERENT places. Give both phases one
  common centre and `net` is multiplied by a positive constant: the delta
  stretches, every sign survives and every correlation is untouched. A test
  pins that, and it is a SCALE statement rather than an identity -- so a
  decision-level null here would still leave a `_SCALE_FAMILIES` question if
  anything were ever shipped, which is why the panel prints median |net|
  beside every variant.

  *The hitter half cannot matter much and is measured to confirm it.* `H_SP`
  and `H_BP` are the same nine bats with and without a platoon offset, so they
  differ by the offset's league mean -- 0.0016 on the current family -- and
  enter both sides of every game. The pitcher centres differ by 0.0156, ten
  times as much. Expect the hitter knob to be arithmetically inert; the panel
  prints it rather than asserting it.

WHAT IS LEFT, AND THE ONE MEASUREMENT THAT DECIDES IT

  What remains is a single question with a directly observable answer: the
  shipped form predicts a phase gap `M_SP - M_BP` of about +0.017, because the
  bullpen composite is a usage-weighted aggregate of the arms a club actually
  uses and sits well below the league batter centre it is divided by. Phase
  matching sets that gap to ~0. So EITHER relievers really do suppress offense
  by about that much and the shipped form is right, OR they do not and the gap
  is a benchmark artifact.

  The ledger already carries the answer. `act_sp_*` is the starter's allowed
  line and the team batting line minus it is the bullpen's, so the realised
  SP-minus-BP wOBA gap is computable from committed rows with no API call and
  no lookahead. That number, not a correlation, is what this question turns
  on -- a correlation can prefer a change that removes a real effect, and on
  this sample it does.

WHAT THIS CANNOT ANSWER

  Whether `K` should differ between starters and relievers. `K = sigma^2/tau^2`
  needs per-player raw rates and sample sizes; the ledger stores shrunk
  aggregates only. `reliever_shrink_probe` fits K from StatsAPI box lines and
  is therefore wOBA-denominated, which CLAUDE.md records as unusable against an
  xwOBA build. Nothing here bears on it either way.

  Whether the SHRINKAGE TARGETS are right. That is measured, live, by
  `build_site.prior_population_centres`, which prints each pool's centre and
  the displacement the shared target imposes on a member, every build. Read
  the build log, not this probe. What this probe can add is the multi-slate
  version of the same gap for the one pool the ledger records -- the slate
  probables -- since `starter_xwoba_*` is that pool's published value.

READ-ONLY. Committed artifacts only; runs anywhere. Nothing here reaches a
lean, a dump or a ledger row, so it carries no MODEL_TAG implication.

Usage:
  python phase_benchmark_probe.py
  python phase_benchmark_probe.py --all-tags
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import actuals_backfill as ab
import build_site
import market_backfill

LEDGER = "data/mlb_lean_ledger.csv"
BOOT = 4000
SEED = 0

# The four pools the construction touches, and the ledger column each one's
# published member value lives in. Keys are the denominators in the formula
# above: H_* divide a lineup composite, P_* divide a pitching composite.
POOL_COL = {
    "H_SP": "opp_xwoba_vs_sp_{side}",
    "H_BP": "opp_xwoba_neutral_{side}",
    "P_SP": "starter_xwoba_{side}",
    "P_BP": "bullpen_xwoba_{side}",
}


def _num(df, col):
    return pd.to_numeric(df.get(col), errors="coerce")


def load(ledger=LEDGER, tags=None):
    """Current-family ledger rows with the build's own `L` recovered per row.

    `L` is not stored, but every phase value is `B * P / L`, so it inverts
    exactly from any one of the four (B, P, mx) triples the row carries. All
    four are computed and the spread between them is reported by `report` as a
    gate: they must agree to float precision or the recovery is wrong and
    every number below it is too.

    The family is derived from `build_site.RECORD_TAGS` rather than named, so
    a MODEL_TAG bump carries this probe forward. `tags=()` scores every family,
    which is a MIXTURE of scales and is offered for inspection only.
    """
    d = pd.read_csv(ledger, low_memory=False)
    fam = tuple(build_site.RECORD_TAGS if tags is None else tags)
    if fam and "model_tag" in d.columns:
        d = d[d["model_tag"].isin(fam)]
    d = d.reset_index(drop=True).copy()
    if d.empty:
        return d
    cands = []
    for side in ("away", "home"):
        for b, p, m in (("opp_xwoba_vs_sp", "starter_xwoba", "mx_xwoba_sp"),
                        ("opp_xwoba_neutral", "bullpen_xwoba", "mx_xwoba_bp")):
            cands.append(_num(d, f"{b}_{side}") * _num(d, f"{p}_{side}")
                         / _num(d, f"{m}_{side}"))
    c = pd.concat(cands, axis=1)
    extra = pd.DataFrame({"L": c.mean(axis=1),
                          "L_spread": c.max(axis=1) - c.min(axis=1)},
                         index=d.index)
    return pd.concat([d, extra], axis=1)


def load_all(ledger=LEDGER):
    """Every ledger row, unfiltered by family.

    The realised phase gap is a BOX-SCORE quantity -- `phase_lines` reads
    `act_sp_*` and the team batting line and nothing else -- so it does not
    depend on which model wrote the row, and scoping it to the current family
    throws away half the games for no reason. The first version of this module
    did exactly that, and on 429 games the interval was wide enough to contain
    the shipped construction; on 966 it is not.

    The PREDICTED gap is not model-independent and stays family-scoped: it is
    in the family's own rate units, and pooling wOBA and xwOBA rows into one
    predicted gap is the `_SCALE_FAMILIES` error. So the two halves of the
    comparison legitimately read different row sets, and `report` prints the
    licence for the pooled half rather than assuming it.
    """
    return pd.read_csv(ledger, low_memory=False)


def pooling_licence(d, fam, boot=BOOT, seed=SEED):
    """Is the realised gap the same inside and outside the current family?

    Pooling is only licensed if this is null. It is a seasonal/compositional
    check, not a model one -- the two row sets are different months of
    baseball, and if relievers' advantage moved between them the pooled
    estimate describes neither window.
    """
    inside = d[d["model_tag"].isin(fam)] if fam else d.iloc[0:0]
    outside = d[~d["model_tag"].isin(fam)] if fam else d
    a = realised_phase_gap(inside, boot=boot, seed=seed) if len(inside) else None
    b = realised_phase_gap(outside, boot=boot, seed=seed) if len(outside) else None
    if a is None or b is None:
        return None
    diff = a["gap"] - b["gap"]
    se = float(np.hypot(a["se"], b["se"]))
    return {"in": a, "out": b, "diff": diff, "se": se,
            "z": diff / se if se > 0 else float("nan")}


def peer_centres(d):
    """Unweighted centre of each pool, pooled over the frame's sides.

    Unweighted because empirical Bayes and a peer benchmark both want the
    centre of the population a member is drawn from, not the centre of the
    innings -- the argument `relief_pool_prior` already makes one level down.
    Pooled rather than per-slate on purpose: a slate supplies 30 members, and
    the panel measures the per-slate version too so the estimation noise that
    buys is visible rather than assumed.
    """
    out = {}
    for key, tmpl in POOL_COL.items():
        v = pd.concat([_num(d, tmpl.format(side=s)) for s in ("away", "home")])
        out[key] = float(v.mean()) if v.notna().any() else float("nan")
    return out


def denominators(centres, L, lam_hit=0.0, lam_pit=0.0):
    """The four denominators at a partial correction, as Series aligned to `L`.

    `lam = 0` is the shipped single denominator and `lam = 1` is the pool's own
    centre; between them the move is geometric, because the denominators enter
    multiplicatively and a geometric path keeps each phase's value a smooth
    power of the correction rather than kinking at either end. Nothing is
    fitted on `lam` -- the panel sweeps it to show the shape and reports the
    one value the realised phase gap implies.
    """
    lam = {"H_SP": lam_hit, "H_BP": lam_hit, "P_SP": lam_pit, "P_BP": lam_pit}
    return {k: L * (centres[k] / L) ** lam[k] for k in POOL_COL}


def variant_net(d, centres, lam_hit=0.0, lam_pit=0.0):
    """`net` (home offense minus away offense) under a partial correction.

    The away-PITCHING row carries the HOME offense, which is why the sides are
    subtracted in that order -- the same cross `build_site.game_net` applies
    and the one join here that a silent sign error could hide. A test pins it
    against the stored `xw_net` at lam = 0 rather than against a correlation.
    """
    den = denominators(centres, d["L"], lam_hit, lam_pit)
    edge = {}
    for side in ("away", "home"):
        q = _num(d, f"sp_share_{side}")
        m_sp = (_num(d, f"opp_xwoba_vs_sp_{side}")
                * _num(d, f"starter_xwoba_{side}") / den["P_SP"]
                * (d["L"] / den["H_SP"]))
        m_bp = (_num(d, f"opp_xwoba_neutral_{side}")
                * _num(d, f"bullpen_xwoba_{side}") / den["P_BP"]
                * (d["L"] / den["H_BP"]))
        base = q * den["P_SP"] + (1.0 - q) * den["P_BP"]
        edge[side] = q * m_sp + (1.0 - q) * m_bp - base
    return edge["away"] - edge["home"]


def phase_gap(d, centres, lam_hit=0.0, lam_pit=0.0):
    """Mean predicted `M_SP - M_BP` over sides, in the same units as the
    realised gap below, so the two are directly comparable."""
    den = denominators(centres, d["L"], lam_hit, lam_pit)
    gaps = []
    for side in ("away", "home"):
        m_sp = (_num(d, f"opp_xwoba_vs_sp_{side}")
                * _num(d, f"starter_xwoba_{side}") / den["P_SP"]
                * (d["L"] / den["H_SP"]))
        m_bp = (_num(d, f"opp_xwoba_neutral_{side}")
                * _num(d, f"bullpen_xwoba_{side}") / den["P_BP"]
                * (d["L"] / den["H_BP"]))
        gaps.append(m_sp - m_bp)
    return float(pd.concat(gaps).mean())


def realised_phase_gap(d, boot=BOOT, seed=SEED):
    """PA-weighted realised starter-allowed minus bullpen-allowed wOBA.

    PA-weighted, not per-side-game: a bullpen residual over four batters is a
    rate with no information in it, and weighting the side-games equally hands
    those rows the same vote as a nine-inning line. The unweighted figure is
    returned beside it because the two differ by a lot here and a reader should
    see which one the verdict rests on.

    The interval is a GAME-clustered bootstrap -- the two sides of a game share
    a park, a date and each other's batting line, so resampling side-games
    independently would understate it.
    """
    rows = []
    for _, r in d.iterrows():
        for side in ("away", "home"):        # the PITCHING side
            sp, bp = ab.phase_lines(r, side)
            if sp is None or bp is None:
                continue
            w_sp, w_bp = (ab.woba_from_components(sp),
                          ab.woba_from_components(bp))
            if w_sp is None or w_bp is None:
                continue

            def _den(c):
                return c["ab"] + (c["bb"] - c["ibb"]) + c["sf"] + c["hbp"]

            pa_sp, pa_bp = _den(sp), _den(bp)
            if pa_sp <= 0 or pa_bp <= 0:
                continue
            rows.append({"game_pk": r.get("game_pk"), "w_sp": w_sp,
                         "pa_sp": pa_sp, "w_bp": w_bp, "pa_bp": pa_bp})
    p = pd.DataFrame(rows)
    if p.empty:
        return None
    lvl = lambda f, w: float(np.average(p[f], weights=p[w]))  # noqa: E731
    gap = lvl("w_sp", "pa_sp") - lvl("w_bp", "pa_bp")
    games = p["game_pk"].to_numpy()
    uniq = np.unique(games)
    where = {g: np.flatnonzero(games == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(boot)
    for i in range(boot):
        idx = np.concatenate([where[g]
                              for g in rng.choice(uniq, uniq.size, replace=True)])
        s = p.iloc[idx]
        draws[i] = (np.average(s["w_sp"], weights=s["pa_sp"])
                    - np.average(s["w_bp"], weights=s["pa_bp"]))
    return {
        "n_sides": int(len(p)), "n_games": int(uniq.size),
        "sp": lvl("w_sp", "pa_sp"), "bp": lvl("w_bp", "pa_bp"),
        "sp_unw": float(p["w_sp"].mean()), "bp_unw": float(p["w_bp"].mean()),
        "pa_sp": float(p["pa_sp"].sum()), "pa_bp": float(p["pa_bp"].sum()),
        "gap": gap, "se": float(draws.std(ddof=1)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


def paired_dcorr(base, cand, act, boot=BOOT, seed=SEED):
    """Bootstrap the DIFFERENCE of two correlations on the same games.

    Paired, because the variants here agree above 0.997 by construction: the
    standalone correlations share almost all of their noise and their
    difference is estimated an order of magnitude better than either of them.
    Resampling both arms on one index is what cancels it.
    """
    m = base.notna() & cand.notna() & act.notna()
    n = int(m.sum())
    if n < 8:
        return {"n": n, "corr": float("nan"), "d": float("nan"),
                "se": float("nan")}
    b, c, a = (base[m].to_numpy(float), cand[m].to_numpy(float),
               act[m].to_numpy(float))
    d = float(np.corrcoef(c, a)[0, 1] - np.corrcoef(b, a)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (boot, n))
    draws = np.array([np.corrcoef(c[i], a[i])[0, 1]
                      - np.corrcoef(b[i], a[i])[0, 1] for i in idx])
    return {"n": n, "corr": float(np.corrcoef(c, a)[0, 1]), "d": d,
            "se": float(draws.std(ddof=1))}


def implied_lambda(d, centres, target, lam_hit=0.0):
    """The `lam_pit` whose predicted phase gap equals `target`, or None.

    Bisection rather than a fit: `target` comes from the realised phase lines,
    which are a different measurement on different rows, so `lam` is DERIVED
    from it and never chosen to make anything else look better.
    """
    lo, hi = 0.0, 1.0
    f = lambda x: phase_gap(d, centres, lam_hit, x) - target  # noqa: E731
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def report(ledger=LEDGER, tags=None):
    out = []

    def say(s=""):
        out.append(s)

    d = load(ledger, tags)
    fam = tuple(build_site.RECORD_TAGS if tags is None else tags)
    say("PHASE-MATCHED PEER BENCHMARKS")
    say(f"  family {', '.join(fam) if fam else 'ALL TAGS (mixed scales)'}"
        f"   rows {len(d)}"
        f"   with a phase split {int(d['L'].notna().sum()) if len(d) else 0}")
    if d.empty:
        say("  no rows; nothing to measure.")
        return out

    act = _num(d, "act_woba_home") - _num(d, "act_woba_away")
    base = variant_net(d, peer_centres(d), 0.0, 0.0)
    stored = _num(d, "xw_net")
    m = base.notna() & stored.notna()
    say()
    say("  GATE. `L` is recovered by inverting the row's own phase values, and")
    say("  the shipped net is then rebuilt from stored inputs. Both must agree")
    say("  to float precision or nothing below this line means anything.")
    say(f"    L recovered on {int(d['L'].notna().sum())} rows, "
        f"worst disagreement between the four inversions "
        f"{float(d['L_spread'].max()):.2e}")
    say(f"    shipped net rebuilt on {int(m.sum())} rows, "
        f"worst |rebuilt - stored| {float((base[m] - stored[m]).abs().max()):.2e}")

    c = peer_centres(d)
    L = float(d["L"].mean())
    say()
    say("  PEER CENTRES, recovered from the ledger's own published values.")
    metric = market_backfill.metric_label(d, mixed="MIXED METRICS")
    say(f"    shared denominator L (league batter {metric})"
        f"{'':>{max(0, 21 - len(metric))}}{L:.5f}")
    for k, label in (("H_SP", "lineup composite vs the starter"),
                     ("H_BP", "lineup composite, neutral"),
                     ("P_SP", "slate starters (shrunk)"),
                     ("P_BP", "bullpen composites (shrunk, usage-wtd)")):
        say(f"    {k}  {label:<38} {c[k]:.5f}   vs L {c[k] - L:+.5f}")
    say(f"    pitcher centres differ by {c['P_SP'] - c['P_BP']:+.5f}; "
        f"hitter centres by {c['H_SP'] - c['H_BP']:+.5f}.")
    say("    That ratio is the closure: the hitter knob has an order of")
    say("    magnitude less room to move anything than the pitcher knob.")

    allrows = load_all(ledger)
    rg = realised_phase_gap(allrows)
    lic = pooling_licence(allrows, fam)
    say()
    say("  THE REALISED PHASE GAP — the measurement this question turns on.")
    say("    Scored on EVERY ledger row, not just the family: `phase_lines`")
    say("    reads box scores, so this quantity does not know which model")
    say("    wrote the row. The predicted gap below it stays family-scoped,")
    say("    because that one is in the family's own rate units.")
    if lic is not None:
        say(f"    pooling licence: in-family {lic['in']['gap']:+.5f} vs "
            f"out-of-family {lic['out']['gap']:+.5f}, "
            f"difference {lic['diff']:+.5f} +- {lic['se']:.5f} "
            f"(z {lic['z']:+.2f}) — "
            f"{'null, so pooling holds' if abs(lic['z']) < 2 else 'NOT null; do not pool'}")
    if rg is None:
        say("    no row carries a usable starter-allowed line; not computable.")
    else:
        say(f"    starter-allowed  {rg['sp']:.5f} PA-weighted "
            f"({rg['sp_unw']:.5f} unweighted, {rg['pa_sp']:.0f} PA)")
        say(f"    bullpen-allowed  {rg['bp']:.5f} PA-weighted "
            f"({rg['bp_unw']:.5f} unweighted, {rg['pa_bp']:.0f} PA)")
        say(f"    realised gap     {rg['gap']:+.5f} +- {rg['se']:.5f}  "
            f"CI [{rg['lo']:+.5f}, {rg['hi']:+.5f}]  "
            f"({rg['n_sides']} sides / {rg['n_games']} games)")
        g0, g1 = phase_gap(d, c, 0.0, 0.0), phase_gap(d, c, 1.0, 1.0)
        say(f"    shipped model predicts {g0:+.5f}; "
            f"full phase matching predicts {g1:+.5f}.")
        if not rg["se"] > 0:
            # A zero-width interval is the estimator saying it has nothing to
            # say: every resample returned the same number. Rendering
            # containment against it would publish a verdict with no sampling
            # distribution behind it -- the `an SE of zero is never a result`
            # entry. Refuse rather than print a confident-looking word.
            say(f"    ZERO-WIDTH interval over {rg['n_games']} game(s): no")
            say("    sampling distribution, so no containment verdict.")
        else:
            inside = lambda g: rg["lo"] <= g <= rg["hi"]  # noqa: E731
            for lab, g in (("shipped model", g0), ("full phase matching", g1)):
                say(f"      {lab} is "
                    f"{'inside' if inside(g) else 'OUTSIDE'} the interval")
            # Asserted in prose on the first version of this module, and the
            # first run on a wider row set falsified it the same day: at 966
            # games the interval tightens and the SHIPPED gap falls outside
            # it, where at the family's own 429 it did not. Every reading is
            # computed now, for the reason the rest of this file exists -- a
            # sentence is not a measurement, and the sentence that got
            # published was the flattering one about what already ships.
            if inside(g0) and inside(g1):
                say("    Neither is rejected, so the actuals do not separate")
                say("    them on these rows.")
            elif inside(g1) and not inside(g0):
                say("    The SHIPPED gap is REJECTED and the matched one is")
                say("    not: the construction overstates how much relievers")
                if abs(rg["gap"]) > 1e-9:
                    say(f"    suppress offense, by {g0 / rg['gap']:.1f}x on the")
                    say("    point estimate. That does NOT make zero the")
                else:
                    say("    suppress offense. That does NOT make zero the")
                say("    answer — the interval's upper end is a real gap this")
                say("    sample cannot rule out either.")
            elif inside(g0) and not inside(g1):
                say("    The MATCHED gap is rejected and the shipped one is")
                say("    not: on these rows the actuals favour what ships.")
            else:
                say("    BOTH are rejected; the construction is mis-levelled")
                say("    in a way neither candidate fixes.")
            if rg["lo"] > 0:
                say("    Zero is excluded, so relievers measurably do suppress")
                say("    offense and a correction that sets the gap to zero")
                say("    removes a real effect rather than an artifact.")
            else:
                say("    Zero is inside the interval, so 'relievers suppress")
                say("    offense' is the point estimate's direction and not an")
                say("    established fact on these rows.")

    say()
    say("  THE KNOBS, scored against the realised run differential.")
    say("    Paired against the shipped net on the same games; the standalone")
    say("    correlation is not the statistic — these variants correlate above")
    say("    0.997 with each other and their difference is what is estimable.")
    cands = [
        ("hitter knob only", 1.0, 0.0,
         "phase-matched lineup benchmark; the user-facing refinement"),
        ("pitcher knob only", 0.0, 1.0,
         "phase-matched pitching benchmark; carries the whole effect"),
        ("both (full model)", 1.0, 1.0,
         "every ratio against its own peer group"),
    ]
    say(f"    {'variant':<20}{'flips':>7}{'mean|dnet|':>12}{'med|net|':>10}"
        f"{'corr':>9}{'d_corr':>9}{'se':>8}{'z':>7}")
    say(f"    {'shipped':<20}{0:>7}{0.0:>12.5f}"
        f"{float(base.abs().median()):>10.5f}"
        f"{paired_dcorr(base, base, act)['corr']:>+9.4f}{'':>9}{'':>8}{'':>7}")
    zs = []
    for label, lh, lp, note in cands:
        x = variant_net(d, c, lh, lp)
        ok = x.notna() & base.notna()
        flips = int(((np.sign(x[ok]) != np.sign(base[ok]))
                     & (base[ok] != 0)).sum())
        r = paired_dcorr(base, x, act)
        z = r["d"] / r["se"] if r["se"] and np.isfinite(r["se"]) else float("nan")
        zs.append(z)
        say(f"    {label:<20}{flips:>7}{float((x[ok] - base[ok]).abs().mean()):>12.5f}"
            f"{float(x[ok].abs().median()):>10.5f}"
            f"{r['corr']:>+9.4f}{r['d']:>+9.4f}{r['se']:>8.4f}{z:>+7.2f}")
        say(f"      {note}")
    say("    med|net| is there because a correction that decided nothing could")
    say("    still move the delta SCALE, and `_SCALE_FAMILIES` is a separate")
    say("    question from `_RECORD_FAMILIES` in this repo.")
    bar = float(np.sqrt(2.0 * np.log(len(cands))))
    say(f"    BAR  {len(cands)} comparisons, so the largest |z| among them")
    say(f"         averages {bar:.2f} under pure noise. Clearing zero is not")
    say("         clearing this panel.")

    say()
    say("  THE CORRECTION IS A DIAL, NOT A SWITCH, and the dial is monotone.")
    say("    lam = 0 is what ships; lam = 1 is full phase matching.")
    say(f"    {'lam':>6}{'pred gap':>11}{'flips':>7}{'d_corr':>9}{'se':>8}{'z':>7}")
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = variant_net(d, c, 0.0, lam)
        ok = x.notna() & base.notna()
        flips = int(((np.sign(x[ok]) != np.sign(base[ok]))
                     & (base[ok] != 0)).sum())
        r = paired_dcorr(base, x, act)
        z = (r["d"] / r["se"]) if (r["se"] and r["se"] > 0) else 0.0
        say(f"    {lam:>6.2f}{phase_gap(d, c, 0.0, lam):>+11.5f}{flips:>7}"
            f"{r['d']:>+9.4f}{r['se']:>8.4f}{z:>+7.2f}")
    if rg is not None:
        li = implied_lambda(d, c, rg["gap"])
        say(f"    lam implied by the realised gap: "
            f"{'none in [0,1]' if li is None else f'{li:.3f}'}")
        say("    Note what the two columns disagree about. d_corr rises")
        say("    monotonically to lam = 1 and never separates; the actuals")
        say("    point at the middle of the range. A monotone unseparated")
        say("    curve is what a null looks like when the change is small, and")
        say("    the correlation has no way to tell 'removes a bias' from")
        say("    'removes a real effect' — which is why the gap decides.")

    say()
    say("  WHAT DOES NOT FOLLOW FROM ANY OF THIS.")
    say("    Nothing here bears on K. sigma^2/tau^2 needs per-player raw rates")
    say("    and sample sizes; the ledger stores shrunk aggregates only.")
    say("    Nothing here re-derives a shrinkage TARGET either: the build's")
    say("    own `prior_population_centres` prints every pool's centre and its")
    say("    displacement each run, and the build log is where that is read.")
    say("    The one pool this frame can follow across slates is the slate")
    say(f"    probables, at {c['P_SP'] - L:+.5f} against the shared target on")
    say("    published (already shrunk) values.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default=LEDGER)
    p.add_argument("--all-tags", action="store_true",
                   help="score every model tag (a MIXTURE of delta scales)")
    a = p.parse_args()
    print("\n".join(report(a.ledger, tags=() if a.all_tags else None)))


if __name__ == "__main__":
    main()
