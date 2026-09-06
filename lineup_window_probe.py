"""Is the negative lineup slope an artifact of the SP/BP scoring window?

`ledger_report.txt`'s component block scores each model term "against its own
realised phase". The lineup term reads slope -0.83 +/- 0.51 with corr -0.066
over 598 side-games, while the weight fit on the same family gives
b_lineup = +0.158 +/- 0.130 and a symmetry test (z = +0.37) that cannot reject
equal first-order weight with SP. Two measurements of one term, disagreeing in
sign.

REGISTERED HYPOTHESIS (docs/lineup_window_registration.md, fixed before any
output here was computed). The SP/BP boundary is endogenous to lineup quality:
a strong lineup chases the starter early and is scored over a short window; a
weak one gets a long window that includes third-time-through PAs. If that
boundary drives the result, the negative slope measures window length rather
than the absence of signal.

  R1  lineup residual on starter batters faced -> POSITIVE coefficient.
  R2  refitting the lineup slope with BF as a covariate -> moves UPWARD.
  FALSIFIER  a conditioned slope within ~0.1 of -0.83 kills the artifact story.
  CONTROL  the same two fits for SP, whose window is endogenous the same way.
           Lineup moving while SP does not WEAKENS the mechanism.

WHAT THIS PROBE CANNOT ANSWER. A FIXED-window rescore -- score the lineup over
the first N batters faced regardless of when the starter left -- is not
computable from committed artifacts, and the two halves fail for different
reasons. The PREDICTED side is blocked by aggregation: 18 batters faced is
exactly two turns through the order, so slot-PA weights go uniform and the
fixed-window prediction is the unweighted nine-hitter mean -- easy in a build
that still holds the per-hitter frame, unrecoverable from a dump that persists
only the weighted composite and its sd. The ACTUAL side is blocked by the data
source and is the binding one: `actuals_backfill.parse_boxscore` reads
`teams.<side>.teamStats.batting` and one pitcher object, so what is stored is
a whole-game total and a starter-allowed total, neither of which is a PREFIX
of the game. Only play-by-play supplies that.

A batting-order index is NOT the missing piece, and it is not missing: the
build assigns one (`build_site.hitter_rows`, `enumerate(lu, start=1)`) and
weights by it, then aggregates it away before writing. Persisting it would
serve the predicted side and leave the actual side exactly as blocked.

`act_sp_bf_<side>` is a per-start COUNT -- enough to CONDITION on window length
and not enough to REDEFINE it. So this probe can say whether the slope is
sensitive to the window; it cannot produce the window-free slope that would
settle it.

Nor is the conditioning causal. Starter BF is a post-treatment outcome of the
same game it is conditioning on -- a lineup that hits well shortens the start
-- so `act ~ pred + BF` is a descriptive decomposition, not an unbiased
estimate of a lineup effect purged of window length. It answers "does the
slope depend on the window", which is the registered question.

Diagnostic only. Nothing here feeds back into a lean, a delta or a grade, and
no lean, delta or ledger row is written.

Usage:
    python lineup_window_probe.py
    python lineup_window_probe.py --tags xw+plat_consol_v12
"""
import argparse

import numpy as np
import pandas as pd

import actuals_backfill as ab
from build_site import RECORD_TAGS

LEDGER = "data/mlb_lean_ledger.csv"


def ols(y, X, names):
    """OLS with standard errors. `X` carries no intercept; one is added.

    Returns {name: (coef, se)} plus n. Uses lstsq rather than a normal-equation
    inverse so a near-collinear design degrades rather than blowing up, and
    reports the se from the residual variance and (X'X)^-1 the same way
    `actuals_backfill.calibration` does for the one-regressor case. The two must
    agree EXACTLY at a single column, or "the slope moved" could be two
    estimators disagreeing rather than the covariate doing anything;
    tests/test_lineup_window_probe.py pins that, and pins the unconditioned fit
    against `actuals_backfill.paired_components` so the comparison is
    like-for-like on whatever the ledger holds.
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - k)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    out = {"n": n, "const": (float(beta[0]), float(se[0]))}
    for j, nm in enumerate(names, start=1):
        out[nm] = (float(beta[j]), float(se[j]))
    return out


def frame(df, tags):
    """Per-component (pred, act, window) rows for the SP and lineup terms.

    The pairings are `actuals_backfill.paired_components`' and are NOT re-derived
    here by hand -- getting the lineup cross backwards yields a plausible
    near-zero slope rather than an error. What is added is `bf`, the window:

      lineup  `opp_xwoba_neutral_<side>` predicts the offense that side's
              pitching faces, i.e. the OTHER side's bats. The share of that
              offense's game spent against a starter is that side's starter's
              batters faced -> `act_sp_bf_<side>`, SAME side as the pred column.
      SP      `starter_xwoba_<side>` is that side's starter and the realised
              allowed line is keyed the same way -> `act_sp_bf_<side>` again.

    Both therefore key on the pitching side, which is why one column serves
    both and why neither takes the cross that the lineup's ACTUAL does.
    """
    if tags is not None and "model_tag" in df.columns:
        df = df[df["model_tag"].astype(str).isin(set(tags))]
    rows = []
    for _, r in df.iterrows():
        for side in ("away", "home"):
            other = "home" if side == "away" else "away"
            sp, _bp = ab.phase_lines(r, side)
            bf = ab._f(r.get(f"act_sp_bf_{side}"))
            for comp, pred_col, act in (
                ("SP", f"starter_xwoba_{side}",
                 ab.woba_from_components(sp) if sp else None),
                ("lineup", f"opp_xwoba_neutral_{side}",
                 ab._f(r.get(f"act_woba_{other}"))),
            ):
                pred = ab._f(r.get(pred_col))
                if pred is None or act is None:
                    continue
                rows.append({"component": comp, "side": side, "pred": pred,
                             "act": act, "bf": bf})
    return pd.DataFrame(rows)


def report(tags=RECORD_TAGS):
    led = pd.read_csv(LEDGER)
    p = frame(led, tags)
    out = []

    def say(s=""):
        out.append(s)

    say(f"LINEUP WINDOW PROBE — registered {', '.join(sorted(set(tags)))}")
    say("registration: docs/lineup_window_registration.md (fixed before any output)")
    say()

    for comp in ("lineup", "SP"):
        s = p[p.component == comp]
        base = ab.calibration(s["pred"], s["act"])
        have = s[s["bf"].notna()]
        say(f"{comp}")
        say(f"  n={len(s)}  with a starter-BF value: {len(have)}")
        if base:
            r = float(np.corrcoef(s["pred"], s["act"])[0, 1])
            mae = float((s["act"] - s["pred"]).abs().mean())
            say(f"  UNCONDITIONED   slope {base['slope']:+.3f} +/- "
                f"{base['se_slope']:.3f}  corr {r:+.4f}  MAE {mae:.4f}")
        if have.empty:
            say("  no BF values; nothing conditioned to report.")
            say()
            continue

        # Descriptive, not a test: the mechanism's first link. If a stronger
        # predicted lineup does NOT get a shorter window, the story fails
        # before either registered prediction is reached.
        rpb = float(np.corrcoef(have["pred"], have["bf"])[0, 1])
        say(f"  descriptive: corr(pred, starter BF) {rpb:+.4f}   "
            f"BF mean {have['bf'].mean():.2f} sd {have['bf'].std():.2f} "
            f"range {have['bf'].min():.0f}-{have['bf'].max():.0f}")

        # R1 -- residual on window length.
        resid = have["act"] - have["pred"]
        f1 = ols(resid, [have["bf"]], ["bf"])
        if f1:
            b, se = f1["bf"]
            say(f"  R1  residual ~ BF        coef {b:+.6f} +/- {se:.6f}  "
                f"z={b / se:+.2f}   (registered direction: POSITIVE)")

        # R2 -- the slope refitted with the window as a covariate.
        f2 = ols(have["act"], [have["pred"], have["bf"]], ["pred", "bf"])
        b0 = ab.calibration(have["pred"], have["act"])
        if f2 and b0:
            b, se = f2["pred"]
            say(f"  R2  act ~ pred + BF      slope {b:+.3f} +/- {se:.3f}   "
                f"(unconditioned on the same {len(have)} rows: "
                f"{b0['slope']:+.3f} +/- {b0['se_slope']:.3f}; "
                f"move {b - b0['slope']:+.3f})")
            bb, bse = f2["bf"]
            say(f"      BF coefficient in that fit {bb:+.6f} +/- {bse:.6f}")
        say()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None,
                    help="comma-separated model tags (default: RECORD_TAGS)")
    a = ap.parse_args()
    tags = tuple(a.tags.split(",")) if a.tags else RECORD_TAGS
    print("\n".join(report(tags)))


if __name__ == "__main__":
    main()
