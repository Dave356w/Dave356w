"""Should the shrinkage target be the player's own history instead of the pool?

Run this where StatsAPI is reachable -- for this repo that means a GitHub
runner, via `.github/workflows/player-prior-probe.yml`.

Why this exists
---------------
wOBA v3 ships

    theta* = (n*theta + K*mu) / (n + K),  K = 400

where `mu` is a POPULATION centre: the league PA-weighted batter rate for
batters and probable starters, the relief pool's own unweighted centre for
relievers (`relief_pool_prior`). K was fitted (`reliever_shrink_probe.py`);
the target was measured; neither asked whether `mu` should be a player.

At K=400 the prior supplies roughly 70% of a median reliever's published rate
and about half of a 400-PA batter's, so the target is not a detail -- it is
most of the number. A career .360 hitter and a career .290 hitter with equal
current-season samples are published within a hair of each other today. The
proposal under test is to replace `mu` with a per-player preseason estimate
`pi_i`, keeping K at 400.

The proposal has three levels and only the third is interesting:

  1. population centre (shipped)
  2. raw career rate -- unadjusted, whatever history exists
  3. recency-weighted career rate, itself regressed toward the population
     centre by how much history exists

(2) is in the report because it is the version that sounds right and is
expected to lose: a player with 70 career PA would receive a fully
individualised prior off almost nothing, and that prior then supplies most of
his published rate. (3) is the hypothesis. Reporting (2) is what makes (3) a
measurement rather than an assertion.

What is actually fitted
-----------------------
Level (3) has its own constant. With recency weights v = (0.50, 0.30, 0.20)
on the previous three seasons,

    theta_hist = sum_s v_s n_s theta_s / sum_s v_s n_s
    H          = sum_s v_s n_s                       (discounted history sample)
    pi         = (H*theta_hist + C*mu) / (H + C)

`C` is the pseudo-sample of population-average history mixed into every
personal prior, and it is fitted here the same way K was: by out-of-sample
weighted MSE on a period the estimate did not see.

All three levels are the SAME expression, not three branches:

    level 1 = H -> 0        (no history: the prior is the population centre)
    level 2 = C -> 0        (all history: the prior is the career rate)
    level 3 = C fitted

That matters beyond tidiness. CLAUDE.md's threshold-cliff entry is exactly
this shape -- the fix for "use career history when the player has enough of
it" is not a better minimum-PA gate, it is a weight, so nothing switches when
a rookie's 71st plate appearance lands.

The objective
-------------
For each player-season in a target season, split the season at a month
boundary. Period 1 is what the model would know; period 2 is what it is
predicting. Every arm is scored by

    loss = sum_i n2_i * (w2_i - (n1_i*w1_i + K*pi_i)/(n1_i + K))^2 / sum_i n2_i

which is `reliever_shrink_probe.loss` with `mu` promoted to a vector. The BF/PA
weighting is the model's own: an estimate that will face 200 hitters matters
ten times more than one that will face 20.

Arms are compared PAIRED, on identical rows, and the reported interval is a
bootstrap over the *difference*. Two arms fitted independently and reported
with overlapping CIs would say nothing about which is better on the same
player.

`C` is refitted inside every bootstrap draw, and a cross-season section fits
`C` on one target season and scores it on another. A constant fitted and
scored on the same rows is optimistically biased, and the size of that bias is
a finding rather than something to argue away.

Why this is not lookahead
-------------------------
Same argument as `reliever_shrink_probe.py`, one step stronger. A prior built
from COMPLETED prior seasons is knowable before the target season starts --
that is what makes it a preseason estimate. No ledger row is touched, no
historical lean is re-derived, and `.savant_cache/` is not consulted: the rate
is rebuilt from StatsAPI counting stats, which read the same whenever they are
pulled.

The one hazard is scoring an arm on the period its constant was fitted on, and
it is measured rather than assumed (the cross-season section).

Reading the output honestly
---------------------------
  * `dMSE vs league` is the decision column. A personal prior that improves
    weighted MSE by 0.3% is not worth a MODEL_TAG bump, however clean the
    theory. Read the paired CI beside it: if it spans zero, this probe has not
    made a case.

  * The n1 buckets are where the claim lives. A personal prior can only matter
    while the current-season sample is small; by September every arm converges
    on the same published rate by construction. If the pooled number is driven
    entirely by the low-n1 bucket, say so -- the shipped model would then be
    buying an April/May improvement and nothing else.

  * Coverage is not incidental. Players with three seasons of history are the
    survivors; rookies and call-ups have none and every arm scores them
    identically. The share of rows and of BF that HAVE usable history bounds
    how much any of this can move a slate.

  * Aging is not modelled. A 38-year-old's three-season history is a biased
    estimate of his talent today and this probe does not correct it; the
    recency ladder is the only aging adjustment present. If level (3) wins by
    a hair, part of that hair is being spent on aging.

What this probe cannot answer
-----------------------------
How many leans a personal prior would flip. That needs a Savant leaderboard as
it read on each past slate, which `.savant_cache/` being gitignored makes
unrecoverable on purpose. The flip count has to come from a forward comparison
on live slates, and it is the gate before any MODEL_TAG bump.

Usage:
  python player_prior_probe.py
  python player_prior_probe.py --targets 2024,2025 --history 3
  python player_prior_probe.py --out player_prior_probe_report.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

import numpy as np

import build_site
from priors_snapshot import centred_history, load_priors
from reliever_shrink_probe import (
    _add,
    _centre,
    fetch_month_periods,
    fetch_season_totals,
    woba,
)

# Recency ladder on the previous three seasons, most recent first. Not fitted:
# it is the proposal as stated, and fitting three weights on top of C against
# the same loss would be three more degrees of freedom for a report this size
# to support. `--recency` varies it so the report can show what the ladder is
# worth, and the `prev-season only` arm is the degenerate (1, 0, 0) case,
# printed every run as the parsimony control.
RECENCY_WEIGHTS = (0.50, 0.30, 0.20)

# Read, never copied. A probe that hardcodes the shipped K reports against
# whatever it was when the probe was written -- which is exactly the
# workflow-pin instance in CLAUDE.md, and the reason `--k` exists only to
# override this for a sensitivity run rather than to restate it.
SHIPPED_K = float(build_site.XWOBA_SHRINK_K)

C_LO, C_HI = 1.0, 50000.0        # search bounds for the history constant
K_FIT_LO, K_FIT_HI = 1.0, 20000.0  # search bounds for the reported K refit
BOOT_SEED = 20260805
BOOT_DRAWS = 1500

# Populations, in the shipped model's own terms. Batters and probable starters
# are regressed toward the league PA-weighted batter centre; relievers toward
# the relief pool's own unweighted centre (build_site.relief_pool_prior). The
# probe mirrors that split so arm 1 is the shipped behaviour and not a
# simplified stand-in for it.
POPULATIONS = ("BAT", "SP", "RP")


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------
def season_role(rec, role_cut):
    """'BAT', 'SP', 'RP' or None for one player-season counting line.

    Role is a property of the season being classified, never of the season
    being predicted. A reliever who moved into a rotation has relief history,
    and the role-separated arm is the one that refuses to average the two.
    """
    g, gs = rec.get("G", 0.0), rec.get("GS", 0.0)
    if g is None or g <= 0:
        return None
    return "RP" if (gs / g) <= role_cut else "SP"


def history_index(seasons, group, role_cut, verbose=True):
    """({pid: {season: (rate, denominator, role)}}, {(season, pool): centre}).

    One call per season-group -- the `stats=season` shape `load_league_era`
    already runs daily, so this is the fetch path least likely to surprise.

    The per-season pool centres come out of the same pull because a rate is
    only comparable across seasons against the centre it was earned under.
    They are what the centred arms read; computing them here costs nothing and
    keeps them from being fetched twice out of step.
    """
    out = defaultdict(dict)
    pools = defaultdict(list)
    for season in sorted(seasons):
        totals = fetch_season_totals(season, group, verbose=verbose)
        for (pid, _), rec in totals.items():
            rate, den = woba(rec)
            if den <= 0 or not math.isfinite(rate):
                continue
            role = "BAT" if group == "hitting" else season_role(rec, role_cut)
            if role is None:
                continue
            out[pid][season] = (float(rate), float(den), role)
            pools[(season, role)].append((rate, den))
            if role != "BAT":
                pools[(season, "P")].append((rate, den))
    centres = {}
    for key, members in pools.items():
        c = _centre(members)
        if c:
            centres[key] = c
    return out, centres


def weighted_history(hist, target_season, weights, role=None, lookback=None):
    """(theta_hist, H) over prior seasons, or (nan, 0.0) when none qualify.

    `weights` is indexed by lag: weights[0] multiplies season-1, weights[1]
    season-2, and so on. A season beyond the ladder contributes nothing, which
    is how the raw-career arm is expressed as a flat ladder rather than as a
    separate code path.

    `role` filters history to seasons the player spent in that role. That is
    the proposal's "starter and reliever history should be separated", and it
    is reported both ways because separating costs sample: an arm with two
    relief seasons and one rotation season keeps two thirds of his history.
    """
    num = den = 0.0
    lookback = len(weights) if lookback is None else lookback
    for lag in range(1, lookback + 1):
        s = target_season - lag
        got = hist.get(s)
        if got is None:
            continue
        rate, n, r = got
        if role is not None and r != role:
            continue
        v = weights[lag - 1] if lag - 1 < len(weights) else 0.0
        if v <= 0 or n <= 0:
            continue
        num += v * n * rate
        den += v * n
    if den <= 0:
        return float("nan"), 0.0
    return num / den, den


def personal_prior(theta_hist, h, mu, c):
    """pi = (H*theta_hist + C*mu) / (H + C). The whole proposal, in one line.

    H = 0 (no usable history) returns `mu` exactly, so a rookie is the n=0
    limit of the same expression rather than a branch. C = 0 returns the raw
    history, which is the arm expected to lose.
    """
    if h is None or h <= 0 or theta_hist is None or not math.isfinite(theta_hist):
        return mu
    if c is None or c <= 0:
        return float(theta_hist)
    return float((h * theta_hist + c * mu) / (h + c))


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------
# A row is one player-season in a target season: what the model would know at
# the split (n1, w1), what it is predicting (n2, w2), the population centre it
# ships against (mu), and every history aggregation an arm might use. History
# is precomputed per row because only C varies during a fit -- refetching or
# re-aggregating inside the golden section would make the search the slow part.
HIST_SPECS = ("recency_role", "recency_blind", "career_role", "prev_role",
              "recency_centred", "savant_centred")


def build_rows(target, periods, hist, split_after, population, role_cut,
               mu, weights, career_lookback, centres=None,
               savant_hist=None, savant_centres=None):
    """Rows for one target season and one population. Returns a list of dicts.

    Role is classified on PERIOD 1 only, for the same reason
    `reliever_shrink_probe.build_pairs` does it: letting the period being
    predicted decide pool membership is picking a threshold with the answer in
    hand.
    """
    by_pid = defaultdict(lambda: [dict(), dict()])
    for (pid, period), rec in periods.items():
        by_pid[pid][0 if period <= split_after else 1] = _add(
            by_pid[pid][0 if period <= split_after else 1], rec)
    flat = [1.0] * max(career_lookback, len(weights))
    rows = []
    for pid, (p1, p2) in by_pid.items():
        if not p1 or not p2:
            continue
        w1, n1 = woba(p1)
        w2, n2 = woba(p2)
        if n1 <= 0 or n2 <= 0 or not (math.isfinite(w1) and math.isfinite(w2)):
            continue
        if population == "BAT":
            role = "BAT"
        else:
            role = season_role(p1, role_cut)
            if role != population:
                continue
        h = hist.get(pid) or {}
        rows.append({
            "pid": int(pid), "season": int(target),
            "n1": float(n1), "w1": float(w1),
            "n2": float(n2), "w2": float(w2), "mu": float(mu),
            "hist": {
                # The proposal as stated: recency ladder over three seasons,
                # history filtered to the role being predicted.
                "recency_role": weighted_history(h, target, weights, role=role),
                # Same ladder, role ignored. The control for whether splitting
                # starter from relief history earns the sample it costs.
                "recency_blind": weighted_history(h, target, weights),
                # Level (2): unadjusted career rate, flat weights over as many
                # prior seasons as were pulled.
                "career_role": weighted_history(h, target, flat, role=role,
                                                lookback=career_lookback),
                # The parsimony control: one season of history, regressed.
                "prev_role": weighted_history(h, target, (1.0,), role=role),
                # Same ladder and the same rows as `recency_role`, but each
                # season read against ITS OWN pool centre and the deviation
                # carried onto today's. Isolates league drift from talent:
                # `recency_role` imports 2023's run environment into a 2025
                # prediction and this one does not.
                "recency_centred": centred_history(
                    h, centres, target, weights, mu, role=role),
                # The shipped metric. History from Savant's completed-season
                # leaderboards (`priors_snapshot.py`) rather than rebuilt from
                # StatsAPI counting stats -- centred, because it is on neither
                # the same scale nor the same season as the target period.
                "savant_centred": centred_history(
                    (savant_hist or {}).get(pid), savant_centres, target,
                    weights, mu, role=role),
            },
        })
    return rows


def to_arrays(rows, spec):
    """(n1, w1, n2, w2, mu, theta_hist, H) as float arrays for one history spec."""
    if not rows:
        z = np.empty(0)
        return (z, z, z, z, z, z, z)
    n1 = np.array([r["n1"] for r in rows], dtype=float)
    w1 = np.array([r["w1"] for r in rows], dtype=float)
    n2 = np.array([r["n2"] for r in rows], dtype=float)
    w2 = np.array([r["w2"] for r in rows], dtype=float)
    mu = np.array([r["mu"] for r in rows], dtype=float)
    th = np.array([r["hist"][spec][0] for r in rows], dtype=float)
    h = np.array([r["hist"][spec][1] for r in rows], dtype=float)
    th = np.where(np.isfinite(th), th, mu)
    h = np.where(np.isfinite(h) & (h > 0), h, 0.0)
    return n1, w1, n2, w2, mu, th, h


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def priors(mu, theta_hist, h, c):
    """Vectorised `personal_prior`. c=None means raw history (C -> 0)."""
    if c is None:
        return np.where(h > 0, theta_hist, mu)
    return (h * theta_hist + c * mu) / (h + c)


def arm_loss(c, n1, w1, n2, w2, mu, theta_hist, h, k=SHIPPED_K):
    """Weighted out-of-sample MSE for one arm. `reliever_shrink_probe.loss`
    with the scalar prior promoted to a vector."""
    pi = priors(mu, theta_hist, h, c)
    pred = (n1 * w1 + k * pi) / (n1 + k)
    tot = float(np.sum(n2))
    if tot <= 0:
        return float("nan")
    return float(np.sum(n2 * (w2 - pred) ** 2) / tot)


def _golden(fn, lo, hi, iters=60):
    """Minimise fn over log x in [lo, hi]. Grid bracket first, then golden.

    Same construction and the same refusals as `reliever_shrink_probe.fit_k`: a
    surface that is flat to float noise gets `nan` rather than whichever bit
    pattern won, and a minimum pinned to a bound is returned as the bound for
    the formatter to flag.
    """
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 60))
    vals = [fn(x) for x in grid]
    hi_v, lo_v = max(vals), min(vals)
    if not math.isfinite(hi_v) or hi_v <= 1e-12 or (hi_v - lo_v) / hi_v < 1e-9:
        return float("nan")
    i = int(np.argmin(vals))
    a = math.log(grid[max(i - 1, 0)])
    b = math.log(grid[min(i + 1, len(grid) - 1)])
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = fn(math.exp(c)), fn(math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(math.exp(d))
        if b - a < 1e-6:
            break
    return float(math.exp((a + b) / 2))


def fit_c(n1, w1, n2, w2, mu, theta_hist, h, k=SHIPPED_K):
    """C minimising the same out-of-sample loss the arms are scored on."""
    if len(n1) < 10 or not np.any(h > 0):
        return float("nan")
    return _golden(lambda c: arm_loss(c, n1, w1, n2, w2, mu, theta_hist, h, k),
                   C_LO, C_HI)


def fit_k_vector(n1, w1, n2, w2, mu, theta_hist, h, c):
    """K minimising the loss at a fixed prior. Reported, not shipped.

    A better prior should be worth MORE weight, so a personal-prior arm whose
    fitted K comes back near 400 is evidence the prior is not carrying new
    information -- and one whose fitted K is far above it is a second
    MODEL_TAG decision, not a free improvement.
    """
    if len(n1) < 10:
        return float("nan")
    return _golden(lambda k: arm_loss(c, n1, w1, n2, w2, mu, theta_hist, h, k),
                   K_FIT_LO, K_FIT_HI)


ARMS = (
    # label, history spec, fit C?
    #
    # Order is load-bearing for the first three: arm 1 is the baseline every
    # dMSE is measured against, and the verdict reads indices 1 and 2.
    ("1 league/role centre", None, False),
    ("2 raw career", "career_role", False),
    ("3 regressed recency", "recency_role", True),
    ("3b regressed, role-blind", "recency_blind", True),
    ("3c regressed, prev season", "prev_role", True),
    # Two separable questions about arm 3, one variable each: does centring
    # each season on its own league help, and is Savant's published rate a
    # better history than the StatsAPI reconstruction? The second arm is empty
    # (H=0 everywhere, so it ties arm 1) until `priors_snapshot.py` has been
    # run and its files committed -- which is the honest way for it to report
    # "no data" rather than a number.
    ("3d regressed, centred", "recency_centred", True),
    ("3e Savant history, centred", "savant_centred", True),
)


def score_arms(rows, k=SHIPPED_K, c_override=None):
    """Every arm on identical rows. Returns [dict] in ARMS order."""
    out = []
    base = None
    for label, spec, do_fit in ARMS:
        if spec is None:
            n1, w1, n2, w2, mu, th, h = to_arrays(rows, HIST_SPECS[0])
            th, h = mu.copy(), np.zeros_like(mu)
            c = None
        else:
            n1, w1, n2, w2, mu, th, h = to_arrays(rows, spec)
            if do_fit:
                c = (c_override.get(label) if c_override else None)
                if c is None:
                    c = fit_c(n1, w1, n2, w2, mu, th, h, k)
                # An arm with no history behind it (nothing frozen yet, or a
                # source that resolved no player) is arm 1 exactly: every prior
                # is mu. Report it as the tie it is rather than as `nan`.
                # Deliberately NOT applied when history exists and the fit
                # still failed -- C = None there would silently score the
                # raw-career arm under a regressed arm's label.
                if not math.isfinite(c or float("nan")) and not np.any(h > 0):
                    c = None
            else:
                c = None
        mse = arm_loss(c, n1, w1, n2, w2, mu, th, h, k)
        if base is None:
            base = mse
        out.append({
            "label": label, "spec": spec, "c": c, "mse": mse,
            "n": len(rows),
            "dmse": ((base - mse) / base * 100.0
                     if base and math.isfinite(mse) and base > 0 else float("nan")),
            "k_fit": fit_k_vector(n1, w1, n2, w2, mu, th, h, c),
            "cover": float(np.mean(h > 0)) if len(h) else float("nan"),
            "cover_bf": (float(np.sum(n2[h > 0]) / np.sum(n2))
                         if len(h) and np.sum(n2) > 0 else float("nan")),
        })
    return out


def paired_bootstrap(rows, spec, do_fit, k=SHIPPED_K,
                     draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Percentile CI for dMSE against the league arm, resampling players.

    C is refitted inside each draw. Holding it at the full-sample value would
    leak every row into every draw and hand the arm a free interval -- the same
    reason `bootstrap_k` recomputes its prior per draw.
    """
    n = len(rows)
    if n < 30:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        sub = [rows[i] for i in idx]
        n1, w1, n2, w2, mu, th, h = to_arrays(sub, spec)
        base = arm_loss(None, n1, w1, n2, w2, mu, mu, np.zeros_like(mu), k)
        c = fit_c(n1, w1, n2, w2, mu, th, h, k) if do_fit else None
        if do_fit and not math.isfinite(c):
            continue
        mse = arm_loss(c, n1, w1, n2, w2, mu, th, h, k)
        if base > 0 and math.isfinite(mse):
            diffs.append((base - mse) / base * 100.0)
    if len(diffs) < draws // 2:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt_c(s):
    """C for one arm. The league arm has no C at all and must not print 0 --
    C = 0 is the raw-career arm, which is a different thing entirely."""
    if s["spec"] is None:
        return "  (H=0)"
    # No history resolved for this source at all. Printing `0` here would read
    # as C = 0 -- the raw-career arm -- which is the opposite of what happened.
    if not s.get("cover"):
        return " (empty)"
    c = s["c"]
    if c is None:
        return "      0"
    if not math.isfinite(c):
        return "     --"
    # A minimum pinned to a search bound is not a fit. Flagged, never printed
    # as one -- the same refusal `reliever_shrink_probe._fmt_k` makes.
    if c >= C_HI * 0.999:
        return f"{'>' + format(C_HI, '.0f'):>7s}"
    if c <= C_LO * 1.001:
        return f"{'<' + format(C_LO, '.0f'):>7s}"
    return f"{c:7.0f}"


def _fmt_k(k):
    if k is None or not math.isfinite(k):
        return "     --"
    if k >= K_FIT_HI * 0.999:
        return f"{'>' + format(K_FIT_HI, '.0f'):>7s}"
    if k <= K_FIT_LO * 1.001:
        return f"{'<' + format(K_FIT_LO, '.0f'):>7s}"
    return f"{k:7.0f}"


HEAD = ("  arm                             n       C       K*   MSE*1e4"
        "   dMSE       95% CI\n  " + "-" * 80)


def _row(s, ci=None):
    lo, hi = ci or (float("nan"), float("nan"))
    ci_s = ("        --" if not (math.isfinite(lo) and math.isfinite(hi))
            else f"[{lo:+6.2f},{hi:+6.2f}]")
    return (f"  {s['label']:<26s} {s['n']:>6d} {_fmt_c(s)} "
            f"{_fmt_k(s['k_fit'])} {s['mse'] * 1e4:>9.3f} "
            f"{s['dmse']:>+7.2f}% {ci_s}")


BUCKETS = ((0, 50), (50, 150), (150, 350), (350, 10 ** 9))


def bucket_label(lo, hi):
    return f"n1 {lo}-{'inf' if hi > 10 ** 8 else hi}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", default="2024,2025",
                    help="seasons to score (comma-separated)")
    ap.add_argument("--history", type=int, default=3,
                    help="prior seasons pulled for the recency ladder")
    ap.add_argument("--career-lookback", type=int, default=0,
                    help="prior seasons in the raw-career arm (0 = all pulled)")
    ap.add_argument("--role-cut", type=float, default=0.20,
                    help="max share of appearances started to count as relief")
    ap.add_argument("--k", type=float, default=SHIPPED_K,
                    help="the shipped XWOBA_SHRINK_K the arms are scored at")
    ap.add_argument("--recency", default=",".join(str(v) for v in RECENCY_WEIGHTS),
                    help="recency ladder, most recent first")
    ap.add_argument("--draws", type=int, default=BOOT_DRAWS)
    ap.add_argument("--priors", default="data",
                    help="directory holding the frozen Savant prior files")
    ap.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args(argv)

    targets = [int(s) for s in a.targets.split(",") if s.strip()]
    weights = tuple(float(v) for v in a.recency.split(",") if v.strip())
    lookback = a.career_lookback or a.history
    hist_seasons = sorted({t - lag for t in targets
                           for lag in range(1, max(a.history, lookback) + 1)})

    L = []

    def say(line=""):
        print(line)
        L.append(line)

    # ---- pull -------------------------------------------------------------
    print("Pulling StatsAPI season totals (history) ...", file=sys.stderr)
    hist, hist_centres = {}, {}
    for group in ("hitting", "pitching"):
        hist[group], centres_g = history_index(hist_seasons, group, a.role_cut)
        hist_centres.update(centres_g)
    # Frozen Savant history, when it has been committed. Absent is not an
    # error: `load_priors` returns empty maps and the Savant arm degrades to
    # H=0 everywhere, which ties arm 1 and prints as such.
    savant_hist, savant_centres = load_priors(a.priors)
    if savant_hist:
        print(f"  Savant priors: {len(savant_hist)} players, "
              f"{len(savant_centres)} season-pool centres", file=sys.stderr)
    else:
        print(f"  Savant priors: none committed under {a.priors}/ -- arm 3e "
              "will report as empty", file=sys.stderr)
    print("Pulling StatsAPI monthly splits (targets) ...", file=sys.stderr)
    monthly, centres = {}, {}
    for season in targets:
        for group in ("pitching", "hitting"):
            try:
                monthly[(season, group)] = fetch_month_periods(season, group)
            except Exception as e:  # noqa: BLE001
                print(f"  {season} {group} byMonth unavailable: {e!r}",
                      file=sys.stderr)
        centres[season] = target_centres(season, a.role_cut)

    say("Player-specific shrinkage priors -- measured against the shipped centre")
    say("=" * 78)
    say(f"targets {targets} | history {a.history} prior seasons | "
        f"recency {weights}")
    say(f"scored at K = {a.k:.0f} (the shipped XWOBA_SHRINK_K), role cut "
        f"started <= {a.role_cut:.0%}")
    say("")
    say("Every arm is the same expression: pi = (H*theta_hist + C*mu)/(H + C).")
    say("Arm 1 is H=0, arm 2 is C=0, arm 3 fits C. dMSE is measured against")
    say("arm 1 on IDENTICAL rows; the CI is a paired bootstrap over players")
    say("with C refitted inside each draw. Read dMSE and its CI first: a")
    say("personal prior that cannot clear zero has not earned a tag bump.")
    say("")

    if not any(monthly.values()):
        say("!! No within-season split available for any target season. Nothing")
        say("!! below can be fitted; see the shape log in reliever_shrink_probe.")
        _finish(L, a.out)
        return 1

    # ---- rows -------------------------------------------------------------
    rows_by_pop = {p: [] for p in POPULATIONS}
    rows_by_pop_season = defaultdict(list)
    for season in targets:
        for pop in POPULATIONS:
            group = "hitting" if pop == "BAT" else "pitching"
            per = monthly.get((season, group))
            if not per:
                continue
            split = balanced_split(per)
            if split is None:
                continue
            mu = shipped_mu(pop, centres.get(season))
            if mu is None:
                continue
            rs = build_rows(season, per, hist[group], split, pop, a.role_cut,
                            mu, weights, lookback, centres=hist_centres,
                            savant_hist=savant_hist,
                            savant_centres=savant_centres)
            rows_by_pop[pop].extend(rs)
            rows_by_pop_season[(pop, season)].extend(rs)

    # ---- coverage ---------------------------------------------------------
    say("Coverage -- how many rows a personal prior can even touch")
    say("  population        rows   with history   share of period-2 BF   mu"
        "        Savant")
    say("  " + "-" * 80)
    for pop in POPULATIONS:
        rs = rows_by_pop[pop]
        if not rs:
            say(f"  {pop:<12s}  no rows")
            continue
        _, _, n2, _, mu, _, h = to_arrays(rs, "recency_role")
        _, _, _, _, _, _, hs = to_arrays(rs, "savant_centred")
        say(f"  {pop:<12s} {len(rs):>7d} {np.mean(h > 0):>13.1%} "
            f"{np.sum(n2[h > 0]) / np.sum(n2):>21.1%}   {np.mean(mu):.4f}"
            f"  {np.mean(hs > 0):>8.1%}")
    say("")
    say("  Rows without usable history score identically under every arm, so")
    say("  the BF share bounds what any of this can buy on a real slate.")
    say("")

    # ---- headline ---------------------------------------------------------
    fitted_c = {}
    for pop in POPULATIONS:
        rs = rows_by_pop[pop]
        if len(rs) < 30:
            continue
        say(f"{pop} -- pooled over {sorted({r['season'] for r in rs})}")
        say(HEAD)
        scored = score_arms(rs, a.k)
        fitted_c[pop] = {s["label"]: s["c"] for s in scored}
        for s, (_label, spec, do_fit) in zip(scored, ARMS):
            ci = (None if spec is None
                  else paired_bootstrap(rs, spec, do_fit, a.k, a.draws))
            say(_row(s, ci))
        say("")

    # ---- the early-season claim ------------------------------------------
    say("By current-season sample -- the claim is that priors matter early")
    say("  " + "-" * 74)
    for pop in POPULATIONS:
        rs = rows_by_pop[pop]
        if len(rs) < 30:
            continue
        say(f"  {pop}")
        for lo, hi in BUCKETS:
            sub = [r for r in rs if lo <= r["n1"] < hi]
            if len(sub) < 30:
                say(f"    {bucket_label(lo, hi):<14s} {len(sub):>6d} rows "
                    f"-- too few to fit")
                continue
            scored = score_arms(sub, a.k, c_override=fitted_c.get(pop))
            best = scored[2]
            raw = scored[1]
            say(f"    {bucket_label(lo, hi):<14s} {len(sub):>6d} rows  "
                f"arm2 {raw['dmse']:>+6.2f}%   arm3 {best['dmse']:>+6.2f}%")
        say("")
    say("  C is held at the pooled fit inside the buckets -- refitting it per")
    say("  bucket would fit four constants and report the best of them.")
    say("")

    # ---- fit here, score there -------------------------------------------
    if len(targets) >= 2:
        say("Cross-season transfer -- C fitted on one season, scored on another")
        say("  " + "-" * 74)
        for pop in POPULATIONS:
            for fit_season in targets:
                src = rows_by_pop_season.get((pop, fit_season)) or []
                dst = [r for s in targets if s != fit_season
                       for r in rows_by_pop_season.get((pop, s)) or []]
                if len(src) < 30 or len(dst) < 30:
                    continue
                n1, w1, n2, w2, mu, th, h = to_arrays(src, "recency_role")
                c = fit_c(n1, w1, n2, w2, mu, th, h, a.k)
                if not math.isfinite(c):
                    continue
                in_sample = score_arms(src, a.k)[2]["dmse"]
                out_sample = score_arms(dst, a.k,
                                        c_override={ARMS[2][0]: c})[2]["dmse"]
                say(f"  {pop:<4s} C={c:>7.0f} fitted on {fit_season}: "
                    f"in-sample {in_sample:>+6.2f}%  "
                    f"applied elsewhere {out_sample:>+6.2f}%")
        say("")
        say("  The gap between those two columns is the optimism in every")
        say("  in-sample dMSE above. If it is most of the improvement, the")
        say("  personal prior is fitting these seasons, not talent.")
        say("")

    _verdict(say, rows_by_pop, a)
    _finish(L, a.out)
    return 0


def target_centres(season, role_cut):
    """Population centres for one season, in the shipped model's own terms."""
    pools = defaultdict(list)
    pit = fetch_season_totals(season, "pitching", verbose=False)
    bat = fetch_season_totals(season, "hitting", verbose=False)
    for (_, _), rec in pit.items():
        rate, den = woba(rec)
        role = season_role(rec, role_cut)
        if den <= 0 or not math.isfinite(rate) or role is None:
            continue
        pools[role].append((rate, den))
    for (_, _), rec in bat.items():
        rate, den = woba(rec)
        if den <= 0 or not math.isfinite(rate):
            continue
        pools["BAT"].append((rate, den))
    return {k: c for k, c in ((k, _centre(v)) for k, v in pools.items()) if c}


def shipped_mu(population, centres):
    """The centre build_site actually shrinks this population toward.

    Batters AND probable starters regress toward the league PA-weighted batter
    rate -- `build_pitching_plans` passes `league_baseline['xwOBA']` for the
    starter too. Only relievers get their own pool, and unweighted, per
    `relief_pool_prior`. Reproducing that split is what makes arm 1 the shipped
    behaviour rather than a tidier stand-in for it.
    """
    if not centres:
        return None
    if population == "RP":
        rp = centres.get("RP")
        return float(rp["unw"]) if rp else None
    bat = centres.get("BAT")
    return float(bat["wtd"]) if bat else None


def balanced_split(periods):
    """Month boundary dividing the season's BF as evenly as months allow."""
    months = sorted({p for _, p in periods})
    if len(months) < 2:
        return None
    totals = defaultdict(float)
    for (_, p), rec in periods.items():
        totals[p] += rec.get("BF", 0.0)
    total = sum(totals.values())
    if total <= 0:
        return None
    best, best_bal = None, -1.0
    run = 0.0
    for m in months[:-1]:
        run += totals[m]
        bal = min(run, total - run)
        if bal > best_bal:
            best, best_bal = m, bal
    return best


def _verdict(say, rows_by_pop, a):
    say("Verdict")
    say("  " + "-" * 74)
    any_pop = False
    for pop in POPULATIONS:
        rs = rows_by_pop[pop]
        if len(rs) < 30:
            continue
        any_pop = True
        scored = score_arms(rs, a.k)
        lo, hi = paired_bootstrap(rs, ARMS[2][1], True, a.k, a.draws)
        arm3, arm2 = scored[2], scored[1]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            call = "CI unavailable -- too many draws failed to fit"
        elif lo <= 0.0 <= hi:
            call = "CI spans zero -- no case for a personal prior here"
        elif lo > 0:
            call = f"beats the shipped centre by {arm3['dmse']:+.2f}% (CI excludes 0)"
        else:
            call = f"LOSES to the shipped centre ({arm3['dmse']:+.2f}%)"
        say(f"  {pop:<4s} arm 3: {call}")
        say(f"       arm 2 (raw career): {arm2['dmse']:+.2f}% -- "
            + ("as predicted, worse than arm 3"
               if arm2["dmse"] < arm3["dmse"] else
               "BETTER than arm 3; the regression step is not earning its C"))
    if not any_pop:
        say("  No population had enough rows to fit.")
    say("")
    say("  None of this licenses a MODEL_TAG bump on its own. A lean-flip count")
    say("  needs a forward comparison on live slates -- see the header.")


def _finish(lines, out):
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
