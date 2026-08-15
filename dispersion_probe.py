"""Does a concentrated lineup beat the mean the model averages it into?

`B_0` is a slot-PA-weighted MEAN of nine shrunk rates, so a lineup of nine .310
bats and a lineup of one .400 and eight .299 bats publish the same composite and
produce the same lean. `opp_xwOBA_sd` records how concentrated the talent behind
that mean was. This scores whether that concentration carries information the
mean throws away.

WRITTEN BEFORE THE FIRST v12 ROW GRADED, ON PURPOSE
---------------------------------------------------
The column shipped 2026-08-15 and no row carrying it had an outcome attached
when this file was written. That is the point: the tests below are fixed in
advance, so this is a stated hypothesis meeting data rather than a search for
whichever cut of the data happens to look significant. If you add a test later,
say so in the output -- an after-the-fact cut is a different kind of evidence
and must not be reported as if it were this one.

THE HYPOTHESIS
--------------
Run scoring is convex in on-base events: rallies compound, and a great hitter
drives in the men already on. If that convexity matters, a lineup whose talent
is concentrated should OUT-PRODUCE the flat lineup with the same mean.

The prior runs the other way and is worth stating before any number appears.
The model's proximate target is team wOBA PER PA, and for that a weighted mean
is not an approximation -- it is the definition. An elite hitter genuinely takes
about a ninth of the plate appearances. Convexity bites on RUNS, which is why
H2 exists alongside H1. Expect a small effect at best.

THE TESTS, in the order they are reported
-----------------------------------------
H1  side level.  residual = act_woba - mx_xwoba, regressed on `sd`.
    A positive slope means concentrated lineups beat their predicted rate.
    Reported three ways: raw, with the backfill count as a covariate, and on
    the zero-backfill subset. A Savant-missing hitter carries the team
    aggregate and sits at the mean by construction, so he DEFLATES sd -- if
    the raw and filtered reads disagree, the filtered one is the answer.

H2  game level.  (act_woba_home - act_woba_away) and run margin, each
    regressed on `d_sd` (home offense's dispersion minus away's, oriented to
    match xw_net). H2 is the one the record actually scores: the model can
    predict each offense well and still order the difference badly.

H3  control.  Is `sd` just a proxy for the composite level? If good lineups are
    also concentrated ones, H1 could be picking up mean quality rather than
    concentration. Reported as corr(sd, pred) so it can be read alongside.

POWER, so a null is readable as a null
--------------------------------------
Printed from the actual n every run rather than assumed. At the ~28 side-games
a full slate contributes, detecting a correlation of 0.15 at 80% power needs
roughly 350 side-games -- about 13 slates. Below that the honest output is an
interval, not a verdict, and the report says so rather than going quiet: a
suppressed number invites someone to recompute it without the caveat.

Usage:
    python dispersion_probe.py
    python dispersion_probe.py --tags xw+plat_consol_v12
    python dispersion_probe.py --boot 20000
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import actuals_backfill as ab

LEDGER = os.path.join(os.environ.get("DATA_DIR", "data"), "mlb_lean_ledger.csv")

# Correlation this is sized to detect, and the power to detect it with. Both
# are stated rather than implied so "underpowered" is a measured claim.
TARGET_R = 0.15
TARGET_POWER = 0.80


def n_for_r(r=TARGET_R, power=TARGET_POWER, alpha=0.05):
    """Side-games needed to detect correlation `r` via Fisher's z."""
    if not 0 < abs(r) < 1:
        return float("inf")
    z_a, z_b = 1.959963985, 0.8416212336   # two-sided 5%, 80%
    if power != 0.80:
        z_b = 0.8416212336
    return math.ceil(((z_a + z_b) / (0.5 * math.log((1 + r) / (1 - r)))) ** 2 + 3)


def ols(x, y):
    """(slope, se, n, r) for y ~ x, or None below three usable points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or x.std() == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    dof = n - 2
    se = math.sqrt((resid @ resid) / dof / ((x - x.mean()) @ (x - x.mean())))
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(b), "se": float(se), "n": int(n), "r": r}


def ols2(x1, x2, y):
    """Slope on x1 controlling for x2. Same return shape, se from the full
    covariance so the control's own uncertainty is carried, not dropped."""
    x1, x2, y = (np.asarray(v, float) for v in (x1, x2, y))
    m = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1, x2, y = x1[m], x2[m], y[m]
    n = x1.size
    if n < 4 or x1.std() == 0:
        return None
    X = np.column_stack([np.ones(n), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid @ resid) / (n - 3)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return {"slope": float(beta[1]), "se": float(math.sqrt(max(cov[1, 1], 0.0))),
            "n": int(n), "r": float("nan")}


def _line(label, fit, need, unit=""):
    if fit is None:
        return f"  {label:<34} — (fewer than 3 usable rows)"
    z = fit["slope"] / fit["se"] if fit["se"] else float("nan")
    lo, hi = fit["slope"] - 1.96 * fit["se"], fit["slope"] + 1.96 * fit["se"]
    tail = ""
    if fit["n"] < need:
        tail = f"   UNDERPOWERED (n={fit['n']}, need ~{need})"
    elif abs(z) < 2:
        tail = "   no effect at this size"
    r = "" if math.isnan(fit["r"]) else f"  corr {fit['r']:+.3f}"
    return (f"  {label:<34} slope {fit['slope']:+.4f} ± {fit['se']:.4f}{unit}"
            f"  z={z:+.2f}  95% [{lo:+.4f}, {hi:+.4f}]{r}"
            f"  n={fit['n']}{tail}")


def run(tags=None):
    led = pd.read_csv(LEDGER, low_memory=False)
    led = led[led["status"] == "graded"]
    if tags:
        led = led[led["model_tag"].isin(tags)]
    out = []
    say = out.append

    say("=" * 78)
    say("LINEUP DISPERSION PROBE — does concentration beat the mean?")
    say(f"  hypothesis and tests fixed before the first row carrying "
        f"opp_xwOBA_sd graded")
    say(f"  scored on {' + '.join(tags) if tags else 'every graded family'}")
    say("=" * 78)

    rates = ab.paired_rates(led)
    have = rates[rates["sd"].notna()] if "sd" in rates else rates.iloc[0:0]
    need = n_for_r()

    say(f"\nrows with a dispersion value AND an outcome: {len(have)}")
    if have.empty:
        say("  none yet. The column ships on new rows only -- every row graded")
        say("  before 2026-08-15 predates it and can never be backfilled,")
        say("  because the nine individual rates were not persisted.")
        say(f"  Expect a first read at ~{need} side-games (~{need // 28} slates).")
        say("=" * 78)
        return out

    resid = have["act"] - have["pred"]
    say(f"\nH1  side level — does a concentrated lineup beat its predicted rate?")
    say(_line("residual ~ sd", ols(have["sd"], resid), need))
    if have["backfill"].notna().any():
        say(_line("  ...controlling for backfill",
                  ols2(have["sd"], have["backfill"].fillna(0), resid), need))
        clean = have[have["backfill"].fillna(0) == 0]
        say(_line("  ...zero-backfill subset only",
                  ols(clean["sd"], clean["act"] - clean["pred"]), need))

    nets = ab.paired_net(led)
    nets = nets[nets["d_sd"].notna()] if "d_sd" in nets else nets.iloc[0:0]
    say(f"\nH2  game level — does the LEAN err with the dispersion gap? "
        f"(n={len(nets)})")
    if len(nets) >= 3:
        say(_line("wOBA differential ~ d_sd", ols(nets["d_sd"], nets["act"]),
                  need // 2))
        say(_line("run margin ~ d_sd", ols(nets["d_sd"], nets["margin"]),
                  need // 2, unit=" runs"))
        lean_err = nets["act"] - nets["pred"]
        say(_line("lean error ~ d_sd", ols(nets["d_sd"], lean_err), need // 2))
    else:
        say("  no game has both offenses' dispersion and an outcome yet.")

    say(f"\nH3  control — is sd just a proxy for lineup quality?")
    c = ols(have["sd"], have["pred"])
    if c:
        say(f"  corr(sd, predicted rate) = {c['r']:+.3f}  n={c['n']}")
        say("  A large value means H1 may be reading mean quality rather than")
        say("  concentration; read the controlled line in H1 first if so.")

    say(f"\ndistribution of sd over these rows:")
    s = have["sd"]
    say(f"  min {s.min():.4f}  p25 {s.quantile(.25):.4f}  median {s.median():.4f}"
        f"  p75 {s.quantile(.75):.4f}  max {s.max():.4f}")
    say("=" * 78)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", help="comma-separated model tags to scope to")
    a = ap.parse_args(argv)
    tags = [t.strip() for t in a.tags.split(",")] if a.tags else None
    print("\n".join(run(tags)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
