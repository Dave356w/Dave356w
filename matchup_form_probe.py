"""Is B*P/L the right combiner? Mostly an unanswerable question -- so ask a better one.

Run this where StatsAPI is reachable if `--backfill` is used; otherwise it runs
anywhere the ledger already carries phase actuals.
See `.github/workflows/matchup-form-probe.yml`.

The question as usually posed
----------------------------
`matchup_value` combines a lineup rate B and a pitching rate P against the
league rate L multiplicatively:

    mult = B*P/L        vs the obvious alternative        add = B + P - L

THE DATA CANNOT SEPARATE THESE, and that is a finding rather than a limitation
to work around. The two forms differ by exactly the interaction term

    mult - add = (B - L)*(P - L)/L

and on this ledger's own inputs that term has mean |.| of 0.00039 wOBA
(SP sd 0.00073, BP sd 0.00043). The realised phase wOBA it would have to be
detected against has sd ~0.11, because a starter phase is ~25 batters faced and
a bullpen phase ~15. Detecting a mean difference that small at 80% power needs
on the order of 600,000 phase-observations -- roughly 1,600 seasons of the
sample this repo accumulates in a month.

So no held-out MSE comparison between mult and add is reported. One would run,
print a number, and that number would be noise wearing a decision's clothing.
The separation arithmetic is printed instead, because a decision NOT to spend
more effort here should rest on a number too.

The question that is answerable
-------------------------------
Both forms imply, to first order around L, that B and P enter with EQUAL
weight: d(mult)/dB = P/L ~ 1 and d(mult)/dP = B/L ~ 1. That is a real,
testable restriction, and it is the one that matters for tuning -- if a
starter's rate deserves more weight than the lineup composite it faces, the
combiner is mis-weighted regardless of which functional form wraps it.

So this fits the free form per phase

    act ~ a + b1*B + b2*P

against the realised phase outcome, weighted by that phase's plate
appearances, and reports b1 and b2 with standard errors. The shipped model
asserts b1 = b2 = 1. Whether that survives is measurable at this sample size,
unlike the interaction term.

This is the phase-level version of the SP-vs-lineup weight fit grade_leans
already contemplates (`net_w = d_lineup + w*d_sp`, gate 120), which has never
fired. It differs in target: that one fits the win, this one fits the rate the
model actually predicts, so it is not swamped by the ~50% per-game noise that
sits between a rate and a result.

Controls, because a fit without one is not a result
---------------------------------------------------
Every form is scored against a CONSTANT predictor -- the pool's own weighted
mean rate. If predicting one number for every phase beats the model's combiner,
the combiner has no signal to tune and the coefficients above are decoration.
Held-out by game, never by row: the two sides of one game share park, weather
and umpire, so splitting between them leaks.

The reconstruction guard
------------------------
B, P and L are recovered from the ledger (L as mx - edge, never a literal), and
`mult` is re-derived and checked against the stored `mx_xwoba_*`. A drift there
means the terms have been mis-identified and every number below would be
plausible and wrong, so it aborts rather than reports. This mirrors
bp_ablation, which asserts its own reconstruction against the shipped lean.

Usage:
  python matchup_form_probe.py
  python matchup_form_probe.py --backfill        # fetch phase actuals in memory
  python matchup_form_probe.py --out report.txt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import actuals_backfill as ab

LEDGER = "data/mlb_lean_ledger.csv"
RECON_TOL = 5e-3          # stored ledger values are rounded; see report header
FAMILIES = (
    ("v9/v10", ["xw+plat_consol_v9", "xw+plat_consol_v10"]),
    ("v7", ["xw+plat_consol_v7"]),
    ("wOBA v2", ["woba+plat_consol_v2"]),
)
# (phase, lineup column, pitching column, stored mx, stored edge)
PHASES = (
    ("SP", "opp_xwoba_vs_sp", "starter_xwoba", "mx_xwoba_sp", "edge_xwoba_sp"),
    ("BP", "opp_xwoba_neutral", "bullpen_xwoba", "mx_xwoba_bp", "edge_xwoba_bp"),
)


def _n(df, name):
    if name not in getattr(df, "columns", []):
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def observations(df):
    """Long frame of one row per (game, side, phase) with B, P, L and the
    realised phase rate. The actual comes from actuals_backfill.phase_lines,
    which owns the starter/bullpen split and its residual validation."""
    rows = []
    for _, r in df.iterrows():
        for side in ("away", "home"):
            sp_line, bp_line = ab.phase_lines(r, side)
            for phase, Bc, Pc, mxc, edc in PHASES:
                line = sp_line if phase == "SP" else bp_line
                if line is None:
                    continue
                act = ab.woba_from_components(line)
                if act is None:
                    continue
                B, P = ab._f(r.get(f"{Bc}_{side}")), ab._f(r.get(f"{Pc}_{side}"))
                mx, ed = ab._f(r.get(f"{mxc}_{side}")), ab._f(r.get(f"{edc}_{side}"))
                if None in (B, P, mx, ed):
                    continue
                L = mx - ed
                if not L or not np.isfinite(L):
                    continue
                den = (line["ab"] + line["bb"] - line["ibb"]
                       + line["hbp"] + line["sf"])
                if den <= 0:
                    continue
                rows.append({"game_pk": r.get("game_pk"),
                             "model_tag": r.get("model_tag"), "phase": phase,
                             "side": side, "B": B, "P": P, "L": L,
                             "mx": mx, "act": act, "w": den})
    return (pd.DataFrame(rows) if rows else
            pd.DataFrame(columns=["game_pk", "model_tag", "phase", "side",
                                  "B", "P", "L", "mx", "act", "w"]))


def check_reconstruction(o):
    """Abort unless B*P/L reproduces the stored mx. Mis-identified terms would
    make every number downstream plausible and wrong."""
    if o.empty:
        return
    err = (o.B * o.P / o.L - o.mx).abs()
    if err.max() > RECON_TOL:
        bad = o.loc[err.idxmax()]
        raise SystemExit(
            f"FAIL: B*P/L does not reproduce the stored mx.\n"
            f"  worst row: phase={bad.phase} side={bad.side} "
            f"game_pk={bad.game_pk}\n"
            f"  B={bad.B:.5f} P={bad.P:.5f} L={bad.L:.5f} "
            f"-> {bad.B * bad.P / bad.L:.5f} vs stored {bad.mx:.5f}\n"
            f"  max error {err.max():.2e} over {len(o)} rows "
            f"(tolerance {RECON_TOL:.0e})\n"
            "The term identification is wrong, or the combiner changed. "
            "Refusing to report a fit on mis-identified inputs.")


def wls(y, X, w):
    """Weighted least squares -> (coefs, standard errors). X includes no
    intercept column; one is added here."""
    X = np.column_stack([np.ones(len(y)), X])
    W = np.asarray(w, dtype=float)
    XtW = X.T * W
    XtWX = XtW @ X
    try:
        inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None, None
    beta = inv @ (XtW @ np.asarray(y, dtype=float))
    resid = np.asarray(y, dtype=float) - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return beta, None
    s2 = float((W * resid ** 2).sum() / dof)
    se = np.sqrt(np.diag(inv) * s2)
    return beta, se


def wmse(y, pred, w):
    y, pred, w = (np.asarray(v, dtype=float) for v in (y, pred, w))
    return float((w * (y - pred) ** 2).sum() / w.sum())


def separation(o, say):
    """Print why mult vs add is not a question this data can answer."""
    say("Why no mult-vs-add comparison is reported")
    say("  " + "-" * 68)
    say(f"  {'phase':<6s}{'n':>6s}{'mean|mult-add|':>16s}{'sd':>12s}"
        f"{'max|.|':>12s}{'n needed (80%)':>16s}")
    for phase, s in o.groupby("phase"):
        d = (s.B * s.P / s.L) - (s.B + s.P - s.L)
        noise = float(np.sqrt(np.average((s.act - np.average(s.act, weights=s.w)) ** 2,
                                         weights=s.w)))
        eff = float(d.abs().mean())
        need = int((2.8 * noise / eff) ** 2) if eff > 0 else -1
        say(f"  {phase:<6s}{len(s):>6d}{eff:>16.6f}{float(d.std()):>12.6f}"
            f"{float(d.abs().max()):>12.6f}{need:>16,d}")
    say("  The two forms differ only by (B-L)(P-L)/L. Against a realised phase")
    say("  rate with sd ~0.11 that difference is unresolvable -- not merely")
    say("  under-powered at this n, but out of reach by ~3 orders of magnitude.")
    say("  Spending effort choosing between them is the thing to stop doing.")
    say("")


def fit_arm(o, label, say, seed=20260805):
    """Free fit vs the constrained forms, held out by game."""
    say(f"Free fit  act ~ a + b1*B + b2*P   [{label}]")
    say("  the shipped combiner asserts b1 = b2 = 1")
    say("  " + "-" * 68)
    say(f"  {'phase':<6s}{'n':>5s}{'b1 (lineup)':>18s}{'b2 (pitching)':>18s}"
        f"{'intercept':>12s}")
    for phase, s in o.groupby("phase"):
        if len(s) < 30:
            say(f"  {phase:<6s}{len(s):>5d}   too few observations to fit")
            continue
        beta, se = wls(s.act, np.column_stack([s.B, s.P]), s.w)
        if beta is None:
            say(f"  {phase:<6s}{len(s):>5d}   singular")
            continue
        b = f"{beta[1]:+.2f}±{se[1]:.2f}" if se is not None else f"{beta[1]:+.2f}"
        c = f"{beta[2]:+.2f}±{se[2]:.2f}" if se is not None else f"{beta[2]:+.2f}"
        say(f"  {phase:<6s}{len(s):>5d}{b:>18s}{c:>18s}{beta[0]:>12.3f}")
        if se is not None:
            for nm, i in (("b1", 1), ("b2", 2)):
                z = (beta[i] - 1.0) / se[i] if se[i] > 0 else float("nan")
                if abs(z) > 2:
                    say(f"         {nm} is {z:+.1f} se from 1.0 -- the equal-weight "
                        f"restriction is rejected for {phase}")
    say("")

    # Held-out, split by GAME: the two sides of one game share park, weather and
    # umpire, so splitting between them leaks the very thing that makes a phase
    # rate deviate.
    say(f"Held-out weighted MSE, split by game  [{label}]")
    say("  a form that cannot beat the CONSTANT has no signal to tune")
    say("  " + "-" * 68)
    say(f"  {'phase':<6s}{'n_test':>8s}{'constant':>12s}{'mult':>12s}"
        f"{'add':>12s}{'free':>12s}")
    rng = np.random.default_rng(seed)
    for phase, s in o.groupby("phase"):
        games = np.array(sorted(set(s.game_pk.dropna())))
        if len(games) < 20:
            say(f"  {phase:<6s}{'-':>8s}  too few games to hold out")
            continue
        rng.shuffle(games)
        tr_games = set(games[: len(games) // 2])
        tr = s[s.game_pk.isin(tr_games)]
        te = s[~s.game_pk.isin(tr_games)]
        if len(tr) < 20 or len(te) < 20:
            continue
        const = float(np.average(tr.act, weights=tr.w))
        beta, _ = wls(tr.act, np.column_stack([tr.B, tr.P]), tr.w)
        free = (beta[0] + beta[1] * te.B + beta[2] * te.P if beta is not None
                else np.full(len(te), const))
        say(f"  {phase:<6s}{len(te):>8d}"
            f"{wmse(te.act, np.full(len(te), const), te.w):>12.5f}"
            f"{wmse(te.act, te.B * te.P / te.L, te.w):>12.5f}"
            f"{wmse(te.act, te.B + te.P - te.L, te.w):>12.5f}"
            f"{wmse(te.act, free, te.w):>12.5f}")
    say("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--backfill", action="store_true",
                    help="fetch phase actuals into memory first (needs StatsAPI)")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    led = pd.read_csv(a.ledger)
    if a.backfill:
        ab.BUDGET_S = a.budget
        led = ab.attach_actuals(led.copy(), verbose=True)

    L = []

    def say(s=""):
        print(s)
        L.append(s)

    o = observations(led)
    check_reconstruction(o)

    say("Matchup combiner probe -- what the data can and cannot decide")
    say("=" * 74)
    say(f"observations: {len(o)}"
        + (f"  ({o.phase.value_counts().to_dict()})" if not o.empty else ""))
    if o.empty:
        say("")
        say("No phase actuals in this ledger. Either the Actions bot has not yet")
        say("run the schema-2 backfill, or this was invoked without --backfill.")
        if a.out:
            open(a.out, "w").write("\n".join(L) + "\n")
        return 0
    err = (o.B * o.P / o.L - o.mx).abs()
    say(f"reconstruction of stored mx from B*P/L: mean {err.mean():.2e} "
        f"max {err.max():.2e} (ledger values are rounded)")
    say("")

    separation(o, say)
    fit_arm(o, "all families pooled -- see per-family below", say)
    for label, tags in FAMILIES:
        fam = o[o.model_tag.astype(str).isin(tags)]
        if len(fam) < 30:
            continue
        fit_arm(fam, label, say)

    say("Pooling caveat: families predict B and P from different inputs, so the")
    say("pooled fit above is a mixture. The per-family arms are the readable")
    say("ones; the pooled arm is there to show the direction is not a single")
    say("family's accident.")

    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
