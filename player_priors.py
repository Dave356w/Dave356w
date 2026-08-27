"""Reading side of the frozen player priors. No repo imports, so anyone can use it.

`priors_snapshot.py` writes `data/woba_priors_<season>.csv` and
`data/woba_prior_centres.csv`; this module reads them and turns them into the
shrinkage target `build_site` regresses toward.

It imports nothing from this repo on purpose. The writer needs `build_site`
(for the season and the metric name) and `build_site` needs the reader, so a
reader living in the writer would be a cycle. One home for the reading side,
imported by the build, the probe and the snapshot alike.

The whole model is one expression:

    pi = (H * theta_hist + C * mu) / (H + C)

  * `mu` is the population centre the build already shrinks that pool toward
    (unchanged by this module: league PA-weighted batter rate for batters and
    starters, the relief pool's own unweighted centre for relievers).
  * `theta_hist` is the player's recency-weighted history, expressed as a
    DEVIATION from the centre he earned it against and carried onto `mu`.
  * `H` is his discounted history sample.
  * `C` is how much population-average history to mix in.

H = 0 returns `mu` exactly, so a rookie or a call-up needs no branch, no
imputed line and no minimum-PA gate: he is the n = 0 limit of the same
formula. That is deliberate -- a hard `>= N` here would relabel a player on his
71st plate appearance, which is the threshold-cliff mistake CLAUDE.md records
twice.

Constants
---------
Both were measured by `player_prior_probe.py` on the runner, not chosen.

RECENCY is the proposal as stated and is not fitted; the probe's `3c` arm
(previous season only) is the parsimony control and loses to the three-rung
ladder in the early-season case on every population (BAT +22.69 vs +25.35,
SP +13.98 vs +16.38, RP +15.73 vs +22.28).

C = 200 is ONE constant for all three populations, not a role split. The
per-population fits were BAT 275 / SP 320 / RP 133 on the early-season split
and BAT 203 / SP 213 / RP 144 on the balanced split; their geometric mean is
204. Weighted MSE is flat near its minimum here exactly as it is for
XWOBA_SHRINK_K, so do not read the third digit -- what is distinguishable is
200 from 0 (the raw-career arm, which loses on every population) and from
infinity (the league centre, which is what this replaces).
"""
from __future__ import annotations

import math
import os

# Most recent season first. Not fitted -- see the module docstring.
RECENCY = (0.50, 0.30, 0.20)

# Pseudo-sample of population-average history mixed into every personal prior.
PRIOR_HISTORY_C = 200.0

PRIORS_PREFIX = "woba_priors_"
CENTRES_NAME = "woba_prior_centres.csv"


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_priors(data_dir="data"):
    """({pid: {season: (woba, pa, role)}}, {(season, pool): centre}).

    Empty maps when nothing is committed, so every caller degrades to the
    population centre -- the pre-v4 behaviour -- instead of failing. Same
    posture `relief_pool_prior` takes when too few arms resolve.

    Reads with the csv module rather than pandas: `build_site` imports this at
    module scope and a reader with no heavy dependency is one less way for the
    daily build to fail before it writes a page.
    """
    import csv

    hist, centres = {}, {}
    if not os.path.isdir(data_dir):
        return hist, centres
    cpath = os.path.join(data_dir, CENTRES_NAME)
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                season, pool = _f(r.get("season")), (r.get("pool") or "").strip()
                wtd, unw = _f(r.get("woba_wtd")), _f(r.get("woba_unw"))
                if season is None or not pool or wtd is None or unw is None:
                    continue
                centres[(int(season), pool)] = {
                    "wtd": wtd, "unw": unw,
                    "n": int(_f(r.get("n")) or 0), "pa": _f(r.get("pa")) or 0.0,
                }
    for name in sorted(os.listdir(data_dir)):
        if not (name.startswith(PRIORS_PREFIX) and name.endswith(".csv")):
            continue
        with open(os.path.join(data_dir, name), "r", encoding="utf-8-sig",
                  newline="") as fh:
            for r in csv.DictReader(fh):
                pid, season = _f(r.get("player_id")), _f(r.get("season"))
                pa, woba = _f(r.get("pa")), _f(r.get("woba"))
                if pid is None or season is None or pa is None or woba is None:
                    continue
                if pa <= 0:
                    continue
                hist.setdefault(int(pid), {})[int(season)] = (
                    woba, pa, (r.get("role") or "").strip())
    return hist, centres


def centred_history(hist, centres, target_season, mu_now, weights=RECENCY,
                    role=None, lookback=None):
    """(theta_hist on today's scale, H), or (nan, 0.0) when no season qualifies.

    Each season's rate is read against ITS OWN pool centre and the deviation is
    carried onto `mu_now`. A level is a property of a run environment; a
    deviation is a property of the player. Two things this protects against,
    and only the second is measurable in the probe:

      * League drift. 2023's offence is not 2026's, and an uncentred rate
        imports the gap. The probe measures this as a wash on adjacent seasons
        (its `3d` arm), because adjacent seasons barely drift -- which is
        exactly why it is not evidence about a three-year-old prior.

      * Pool baselines, which do NOT cancel. The build shrinks relievers toward
        their pool's unweighted centre and everyone else toward the league
        PA-weighted rate, and those sit ~0.010 apart -- the same gap that
        motivated moving the relief target in v3. A reliever's stored rate has
        to be read against the relief pool he earned it in, not against the
        league. That is what makes this the deviation form rather than a tidier
        way to write the same number.

    `role` filters history to seasons spent in that role. The probe measured
    that as worth nothing (SP +16.38 vs +16.85 role-blind, RP +22.28 vs
    +20.87), so the build passes None and the parameter exists for the probe.
    """
    num = den = 0.0
    weights = tuple(weights)
    lookback = len(weights) if lookback is None else lookback
    for lag in range(1, lookback + 1):
        got = (hist or {}).get(target_season - lag)
        if got is None:
            continue
        rate, n, r = got
        if role is not None and r != role:
            continue
        v = weights[lag - 1] if lag - 1 < len(weights) else 0.0
        if v <= 0 or n <= 0:
            continue
        c = (centres or {}).get((target_season - lag, r))
        if c is None:
            # No centre beside the rate means it cannot be de-levelled, and
            # using its raw level would import that season's environment
            # silently. Skip the season instead.
            continue
        base = c["unw"] if r == "RP" else c["wtd"]
        num += v * n * (rate - base)
        den += v * n
    if den <= 0:
        return float("nan"), 0.0
    return mu_now + num / den, den


def personal_prior(theta_hist, h, mu, c=PRIOR_HISTORY_C):
    """pi = (H*theta_hist + C*mu) / (H + C). H = 0 -> mu, with no branch."""
    mu_f = _f(mu)
    if mu_f is None:
        return None
    h = _f(h) or 0.0
    th = _f(theta_hist)
    if h <= 0 or th is None:
        return mu_f
    c = _f(c)
    if c is None or c <= 0:
        return th
    return (h * th + c * mu_f) / (h + c)


def prior_for(pid, hist, centres, season, mu, c=PRIOR_HISTORY_C,
              weights=RECENCY, role=None):
    """One player's shrinkage target. `mu` whenever he has no usable history."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return _f(mu)
    th, h = centred_history((hist or {}).get(pid), centres, season, mu,
                            weights=weights, role=role)
    return personal_prior(th, h, mu, c)
