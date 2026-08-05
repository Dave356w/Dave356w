"""What K should regress a reliever's wOBA-allowed? Fit it, don't inherit it.

Run this where StatsAPI is reachable -- for this repo that means a GitHub
runner, via `.github/workflows/reliever-shrink-probe.yml`.

Why this exists
---------------
`XWOBA_SHRINK_K = 100.0` regresses three different populations toward one
target: batters by PA, probable starters by BF, and every member of the
bullpen pool by BF (`bullpen_xwoba_aggregate` -> `_shrink_one`). v8 chose the
single fixed value for reproducibility, retiring v7's per-pool estimator, and
nothing since has asked whether one number fits all three.

Marcel is the precedent for expecting it not to: it carries separate baselines
by role -- league mean over 200 PA for a hitter, 60 IP for a starter, 25 IP for
a reliever -- so a reliever is regressed toward league roughly 2.4x harder than
a starter per inning. That is IP-denominated and this model is BF-denominated,
and no published BF-denominated reliever constant exists to convert it against.
So the constant has to be fitted here, off games this repo can pull.

Nothing in this probe ships. It moves no lean, writes no ledger row, and
carries no MODEL_TAG implication. It prints a number and a confidence interval
so a decision about `XWOBA_SHRINK_K` can be argued against measurements.

What is being fitted
--------------------
For a reliever with `n` batters faced and observed rate `w`, the build
publishes

    w* = (n*w + K*mu) / (n + K)

so K is the pseudo-sample of league-average performance mixed into every
estimate: at n = K the published rate is half the pitcher and half the prior.
The K that belongs there is the one that best predicts the innings the model
is actually forecasting -- the pitcher's *next* batters faced, not the ones
already in his line.

So each pitcher-season is split into two disjoint periods, and K is chosen to
minimise out-of-sample error on the second:

    loss(K) = sum_i n2_i * (w2_i - (n1_i*w1_i + K*mu1)/(n1_i + K))^2 / sum_i n2_i

Weighting by second-period BF is what makes this the model's own objective: an
estimate that will face 200 hitters matters ten times more than one that will
face 20. The weighting also makes the fit unbiased in K -- the irreducible
sampling noise in `w2` contributes `sum_i n2_i * Var(w2_i) = n * sigma^2`,
which is the same constant for every K and so cannot move the minimum.

A second, independent estimate is reported beside it: the variance-components
form `K = sigma^2 / tau^2`, where `tau^2` is the between-reliever talent
variance recovered from the covariance of the two periods (independent samples,
so their covariance is talent alone) and `sigma^2` is the per-BF outcome
variance backed out of the total spread. It optimises nothing. When a fitted
minimum and a moment estimate that share no machinery land in the same place,
the number is a property of the data rather than of the search.

Why this is not lookahead
-------------------------
The `.savant_cache/` rule exists because a Savant leaderboard is a season
aggregate: today's pull cannot tell you what it read on 3 July, so historical
rows cannot be re-derived from it. This probe never re-derives a row. It fits a
prior -- a population constant, from completed pitcher-seasons, for use on
future slates -- which is the same status `LEAN_STRENGTH_FALLBACK`'s prior
holds. No ledger row is touched, so no row's inputs can be contaminated.

The one real hazard is fitting on periods the shipped constant would then be
applied to, and it is handled by construction: every arm predicts a period it
did not see.

Reading the output honestly
---------------------------
  * The headline is the pooled within-season fit with a bootstrap CI over
    pitchers. If that CI contains 100, this probe has not made a case for
    changing anything, however far the point estimate sits from it.

  * `dMSE vs K=100` is the decision-relevant column, not the distance between
    K values. Weighted MSE is flat near its minimum -- K can be off by a factor
    of two for a fraction of a percent of error -- so a large move in K that
    buys nothing measurable is not a finding.

  * Starters and batters are fitted by the identical code path. They are the
    control: if all three arms land together, the single shipped K is right and
    role-splitting is unjustified. The claim under test is a *difference*.

  * The season-pair arm (year N -> year N+1) is an upper bound on K, not an
    alternative estimate of it. A year of aging, injury and role change sits
    inside that residual, and the fit charges all of it to sampling noise. It
    is reported because Marcel's constants are year-to-year ones, so it is the
    arm comparable to the 200/60/25 precedent; the within-season arm is the one
    that matches what the build does.

  * Every arm is fitted under both plausible priors -- the reliever pool's own
    centre and the league-wide centre the build actually uses. These are not
    the same number, and a K fitted against the wrong centre absorbs the offset
    by shrinking less. That interaction is a live question in this codebase
    (`_pool_moments` in build_site.py logs exactly this gap), so it is reported
    rather than assumed away.

What this probe cannot answer
-----------------------------
How many leans a new K would flip. That needs a Savant pull per historical
slate, which is the one thing the no-lookahead rule forbids. The flip count has
to come from a forward comparison on live slates, and it is the gate before any
MODEL_TAG bump -- not something to infer from the numbers below.

Usage:
  python reliever_shrink_probe.py
  python reliever_shrink_probe.py --seasons 2024,2025,2026
  python reliever_shrink_probe.py --out reliever_shrink_probe_report.txt
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

from fetch_headers import FETCH_HEADERS

STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
SPORT_ID = 1

# Linear weights for reconstructing wOBA from StatsAPI counting stats. Savant
# publishes the rate directly; StatsAPI does not, and StatsAPI is the host that
# serves per-month splits, so the rate is rebuilt here.
#
# The scale constant does not matter. Multiplying every weight by c scales both
# tau^2 and sigma^2 by c^2 and leaves K = sigma^2/tau^2 exactly unchanged, so
# no year-specific wOBA scaling is needed and none is applied. Only the weights
# *relative to each other* can move the answer, and those drift by ~1% a year;
# `--weight-jitter` measures what that drift is worth instead of arguing it.
WOBA_W = {"BB": 0.690, "HBP": 0.720, "1B": 0.880,
          "2B": 1.250, "3B": 1.590, "HR": 2.030}

K_LO, K_HI = 0.5, 5000.0          # search bounds, log-spaced
BOOT_SEED = 20260805
BOOT_DRAWS = 2000


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def _get(url, params, tries=4):
    q = urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{url}?{q}", headers=FETCH_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if k == tries - 1:
                raise
            time.sleep(1.5 * (2 ** k))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _outs(ip):
    """StatsAPI innings ('62.1') -> outs. Third digit is outs, not a fraction."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac[0]) if frac else 0)
    except (TypeError, ValueError):
        return 0


def _splits(payload):
    for blk in payload.get("stats", []) or []:
        for sk in blk.get("splits", []) or []:
            yield sk


def _bail(what, season, group, stat_type, payload, extra=""):
    """Die naming the field that failed, not the guess that follows.

    A probe that prints a report over an empty pull is worse than one that
    crashes: the numbers look like measurements. This repo has paid for that
    twice (see pythag_control_probe._bail), so every fetch path lands here.
    """
    n_blk = len(payload.get("stats", []) or [])
    n_spl = sum(1 for _ in _splits(payload))
    sample = []
    for sk in _splits(payload):
        sample.append({k: sk.get(k) for k in ("season", "month", "numTeams")
                       if k in sk})
        if len(sample) >= 3:
            break
    raise SystemExit(
        f"FAIL: {what} for season={season} group={group} stats={stat_type}.\n"
        f"  stat blocks returned: {n_blk}\n"
        f"  splits returned: {n_spl}\n"
        f"  split keys sampled: {json.dumps(sample)[:400]}\n"
        f"  {extra}\n"
        "Refusing to print a report built on an empty pull.")


def _collect(payload, group, period, into, dropped):
    """Accumulate one payload's splits into `into`, keyed (pid, period).

    Accumulates rather than assigns. A player traded mid-season appears under
    each club, and the per-team shapes below issue one call per club, so the
    same (pid, period) is written more than once by construction. Assigning
    would keep the last club's line and silently discard the rest of his
    season.
    """
    n_raw = 0
    for sk in _splits(payload):
        n_raw += 1
        who = sk.get("player") or sk.get("person") or {}
        pid = who.get("id")
        if pid is None:
            dropped["no player id"] += 1
            continue
        # Position players mopping up in blowouts are not the population the
        # bullpen pool draws from, and their rates are extreme enough to move a
        # variance estimate on their own.
        pos = (sk.get("position") or {}) or (who.get("primaryPosition") or {})
        if group == "pitching" and pos and str(pos.get("code")) not in ("1", "None", ""):
            dropped["non-pitcher"] += 1
            continue
        p = sk.get("month") if period is None else period
        if p is None:
            dropped["no month on split"] += 1
            continue
        st = sk.get("stat") or {}
        rec = {
            "AB": _num(st.get("atBats")),
            "H": _num(st.get("hits")),
            "2B": _num(st.get("doubles")),
            "3B": _num(st.get("triples")),
            "HR": _num(st.get("homeRuns")),
            "BB": _num(st.get("baseOnBalls")),
            "IBB": _num(st.get("intentionalWalks")),
            "HBP": _num(st.get("hitByPitch")),
            "SF": _num(st.get("sacFlies")),
            "BF": _num(st.get("battersFaced") or st.get("plateAppearances")),
            "G": _num(st.get("gamesPitched") or st.get("gamesPlayed")),
            "GS": _num(st.get("gamesStarted")),
            "outs": float(_outs(st.get("inningsPitched"))),
        }
        key = (int(pid), int(p))
        into[key] = _add(into[key], rec) if key in into else rec
    return n_raw


_team_ids = {}


def team_ids(season):
    """Club ids for a season. Needed only by the per-team shapes."""
    if season not in _team_ids:
        js = _get("https://statsapi.mlb.com/api/v1/teams",
                  {"sportId": SPORT_ID, "season": season})
        _team_ids[season] = [int(t["id"]) for t in js.get("teams", [])
                             if t.get("id") is not None]
    return _team_ids[season]


def _month_ranges(season):
    for m in range(3, 12):
        last = calendar.monthrange(season, m)[1]
        yield m, f"{season}-{m:02d}-01", f"{season}-{m:02d}-{last:02d}"


# Candidate call shapes for splitting a season into periods, tried in order.
#
# The first version of this probe assumed one shape and hard-failed on it. CI
# measured that assumption wrong: `stats=byMonth` with `sportIds=1` and
# `playerPool=ALL` returned zero stat blocks for every season. That is the same
# mistake this repo already has on record for the schedule endpoint's
# startDate/endDate form -- guessing at parameters from a sandbox that cannot
# reach StatsAPI.
#
# So the shape is discovered rather than assumed, and which one answered is
# printed in the report. `sportId` singular is listed before `sportIds` plural
# on the theory that the split endpoints want the singular; the plural is kept
# in the list precisely because CI has now measured it, and a shape known not
# to work is worth keeping as a recorded measurement rather than deleting.
#
# The per-team shapes are last because they cost 30 calls per season-group, and
# first among the fallbacks in likelihood: `load_team_pitcher_roles` already
# proves `teamId` + `season` answers on this endpoint, so they vary one
# parameter from a call this repo runs on every build.
def _shape_calls(name, season, group):
    """(params, period) pairs for one candidate shape. period None = read month."""
    base = {"group": group, "season": season, "gameType": "R", "limit": 5000}
    if name == "byMonth/sportId":
        return [({**base, "stats": "byMonth", "sportId": SPORT_ID,
                  "playerPool": "ALL"}, None)]
    if name == "byMonth/sportIds":
        return [({**base, "stats": "byMonth", "sportIds": SPORT_ID,
                  "playerPool": "ALL"}, None)]
    if name == "byMonth/no-pool":
        return [({**base, "stats": "byMonth", "sportId": SPORT_ID}, None)]
    if name == "byDateRange/month":
        return [({**base, "stats": "byDateRange", "sportId": SPORT_ID,
                  "playerPool": "ALL", "startDate": s, "endDate": e}, m)
                for m, s, e in _month_ranges(season)]
    if name == "byMonth/team":
        return [({**base, "stats": "byMonth", "sportId": SPORT_ID,
                  "teamId": t, "limit": 1000}, None)
                for t in team_ids(season)]
    raise ValueError(name)


PERIOD_SHAPES = ["byMonth/sportId", "byMonth/sportIds", "byMonth/no-pool",
                 "byDateRange/month", "byMonth/team"]

_shape_log = []          # (shape, season, group, outcome) -- printed in the report
_working_shape = {}      # group -> the shape that answered, reused across seasons


def fetch_month_periods(season, group, verbose=True):
    """(pid, month) -> counting stats, via whichever candidate shape answers.

    Returns {} when no shape yields two periods. Deliberately not a hard exit:
    the season-pair arm runs on `stats=season`, which IS a proven shape, so a
    within-season split that cannot be built must not take down an arm that
    can. The caller reports the gap instead.
    """
    order = PERIOD_SHAPES
    if group in _working_shape:      # don't re-probe once one has answered
        order = [_working_shape[group]] + [s for s in PERIOD_SHAPES
                                           if s != _working_shape[group]]
    for name in order:
        out, dropped, n_raw = {}, defaultdict(int), 0
        try:
            for params, period in _shape_calls(name, season, group):
                n_raw += _collect(_get(STATS_URL, params), group, period,
                                  out, dropped)
        except Exception as e:  # noqa: BLE001
            _shape_log.append((name, season, group, f"error {e!r}"[:60]))
            continue
        periods = {p for _, p in out}
        total_bf = sum(r["BF"] for r in out.values())
        if len(periods) >= 2 and total_bf > 0:
            _shape_log.append((name, season, group,
                               f"OK: {len(out)} player-periods, "
                               f"{len(periods)} periods, {total_bf:.0f} BF"))
            _working_shape[group] = name
            if verbose:
                print(f"  {season} {group}: {name} -> {len(out)} player-periods, "
                      f"{len(periods)} periods, dropped {dict(dropped)}",
                      file=sys.stderr)
            return out
        _shape_log.append((name, season, group,
                           f"empty: {n_raw} raw splits, {len(periods)} periods"))
    if verbose:
        print(f"  {season} {group}: no candidate shape returned two periods",
              file=sys.stderr)
    return {}


def fetch_season_totals(season, group, verbose=True):
    """(pid, 0) -> season counting stats. The proven shape, unchanged.

    This is `load_league_era`'s call with `group` varied -- endpoint, sportIds,
    playerPool and limit all identical to what the build runs daily. It is the
    one fetch here allowed to hard-fail, because if it returns nothing the
    problem is StatsAPI or this repo's access to it, not a parameter guess.
    """
    out, dropped = {}, defaultdict(int)
    payload = _get(STATS_URL, {
        "stats": "season", "group": group, "season": season,
        "sportIds": SPORT_ID, "gameType": "R", "playerPool": "ALL",
        "limit": 5000,
    })
    n_raw = _collect(payload, group, 0, out, dropped)
    if not out:
        _bail("no usable splits", season, group, "season", payload,
              extra=f"raw splits {n_raw}, dropped {dict(dropped)}")
    if verbose:
        print(f"  {season} {group} season: {len(out)} players, "
              f"dropped {dict(dropped)}", file=sys.stderr)
    return out


def fetch_periods(season, group, stat_type, verbose=True):
    """Dispatch to the season-totals or month-split fetcher."""
    if stat_type == "season":
        return fetch_season_totals(season, group, verbose)
    return fetch_month_periods(season, group, verbose)


# --- league prior by role and hand -----------------------------------------
# A separate question from K, and the one that binds if K stays at 100.
#
# The build shrinks batters, probable starters and every reliever toward ONE
# target: the PA-weighted league batter wOBA from compute_league_baseline. At
# K = 100 and a median reliever's ~92 BF that target supplies 52% of every
# published reliever rate, so if it is the centre of no actual subpool, half
# of each estimate is a number that describes nobody. `_pool_moments` in
# build_site.py logs this gap; this measures it by role and by throwing hand.
#
# TWO THINGS THIS CANNOT SAY, both because of how the rate is rebuilt:
#
#   * Absolute level. wOBA here is reconstructed from StatsAPI counting stats
#     with the fixed linear weights above, while the build's target comes from
#     Savant's own wOBA column. Those differ by the year's wOBA scale, so a
#     centre printed here is NOT a value to paste into build_site.py. K was
#     immune to this -- it is a variance ratio, so the scale cancels -- and a
#     centre is not. Only the GAPS BETWEEN POOLS below are scale-robust,
#     because the same weights are applied to every pool.
#
#   * Therefore the fix is a per-build derivation off the Savant leaderboard
#     the build already fetches, never a literal copied from this report. That
#     is the constants-frozen-from-data entry in CLAUDE.md, and a role-split
#     target would be a fresh instance of it.
#
# Weighted and unweighted centres are both reported, and the difference is not
# cosmetic. Empirical Bayes wants the centre of the population each estimate is
# drawn from -- the player-level, unweighted centre. The build's target is
# PA-weighted, which flatters both pools relative to their own members, since
# good hitters take more PA and good relievers get more usage. That is the
# first of the two gaps _pool_moments separates, and it survives here.
def fetch_hands(pids, verbose=True):
    """pid -> {'throws', 'bats'}. Mirrors build_site.load_people's call shape."""
    out = {}
    pids = [int(p) for p in pids]
    for i in range(0, len(pids), 100):
        chunk = pids[i:i + 100]
        data = _get("https://statsapi.mlb.com/api/v1/people",
                    {"personIds": ",".join(map(str, chunk))})
        for p in data.get("people", []) or []:
            out[int(p["id"])] = {
                "throws": (p.get("pitchHand") or {}).get("code"),
                "bats": (p.get("batSide") or {}).get("code"),
            }
    if verbose:
        print(f"  hands: {len(out)} of {len(pids)} players resolved",
              file=sys.stderr)
    return out


def _centre(members):
    """(n, weighted centre, unweighted centre, total weight, median weight)."""
    if not members:
        return None
    r = np.array([m[0] for m in members], dtype=float)
    w = np.array([m[1] for m in members], dtype=float)
    ok = np.isfinite(r) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return None
    r, w = r[ok], w[ok]
    return {"n": int(len(r)), "wtd": float(np.sum(r * w) / np.sum(w)),
            "unw": float(np.mean(r)), "total": float(np.sum(w)),
            "med": float(np.median(w))}


def role_centres(season, role_cut, verbose=True):
    """{label: centre dict} for batters and for SP/RP split by throwing hand.

    Role is classified on season starts/appearances, and handedness comes from
    the people endpoint rather than being inferred from splits -- a pitcher's
    hand is a fact about him, not a property of the sample.
    """
    pools = defaultdict(list)
    pit = fetch_season_totals(season, "pitching", verbose=False)
    bat = fetch_season_totals(season, "hitting", verbose=False)
    hands = fetch_hands(sorted({p for p, _ in pit} | {p for p, _ in bat}),
                        verbose=verbose)
    for (pid, _), rec in pit.items():
        rate, den = woba(rec)
        if den <= 0 or not math.isfinite(rate):
            continue
        g, gs = rec.get("G", 0.0), rec.get("GS", 0.0)
        if g <= 0:
            continue
        role = "RP" if (gs / g) <= role_cut else "SP"
        hand = (hands.get(pid) or {}).get("throws") or "?"
        pools[f"{role}-{hand}"].append((rate, den))
        pools[role].append((rate, den))
    for (pid, _), rec in bat.items():
        rate, den = woba(rec)
        if den <= 0 or not math.isfinite(rate):
            continue
        side = (hands.get(pid) or {}).get("bats") or "?"
        pools["BAT"].append((rate, den))
        pools[f"BAT-{side}"].append((rate, den))
    return {k: c for k, c in ((k, _centre(v)) for k, v in pools.items()) if c}


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------
def _add(a, b):
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in set(a) | set(b)}


def woba(rec, w=None):
    """(rate, denominator). Denominator is AB + BB - IBB + SF + HBP.

    Returns (nan, 0) when the denominator is empty, which is how a pitcher with
    an appearance but no completed plate appearance leaves the pool.
    """
    w = w or WOBA_W
    den = rec["AB"] + rec["BB"] - rec["IBB"] + rec["SF"] + rec["HBP"]
    if den <= 0:
        return float("nan"), 0.0
    singles = rec["H"] - rec["2B"] - rec["3B"] - rec["HR"]
    num = (w["BB"] * (rec["BB"] - rec["IBB"]) + w["HBP"] * rec["HBP"]
           + w["1B"] * singles + w["2B"] * rec["2B"]
           + w["3B"] * rec["3B"] + w["HR"] * rec["HR"])
    return num / den, den


def build_pairs(periods, split_after, role, role_cut, weights=None):
    """Pitcher-seasons as (n1, w1, n2, w2, bf_per_ip), split at a period boundary.

    Role is classified on the FIRST period only. Using the whole season would
    let the second period -- the one being predicted -- decide who is in the
    pool, which is the same error as picking a threshold with the answer in
    hand. It costs some precision on arms that convert mid-season, and that is
    the correct trade.
    """
    by_pid = defaultdict(lambda: [dict(), dict()])
    for (pid, period), rec in periods.items():
        half = 0 if period <= split_after else 1
        by_pid[pid][half] = _add(by_pid[pid][half], rec)
    rows = []
    for pid, (p1, p2) in by_pid.items():
        if not p1 or not p2:
            continue
        w1, n1 = woba(p1, weights)
        w2, n2 = woba(p2, weights)
        if n1 <= 0 or n2 <= 0 or not (math.isfinite(w1) and math.isfinite(w2)):
            continue
        if role is not None:
            g, gs = p1.get("G", 0.0), p1.get("GS", 0.0)
            if g <= 0:
                continue
            share = gs / g
            if role == "RP" and share > role_cut:
                continue
            if role == "SP" and share <= role_cut:
                continue
        ip = p1.get("outs", 0.0) / 3.0
        rows.append((n1, w1, n2, w2, (p1["BF"] / ip) if ip > 0 else float("nan")))
    return np.array(rows, dtype=float) if rows else np.empty((0, 5))


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def loss(k, n1, w1, n2, w2, mu):
    pred = (n1 * w1 + k * mu) / (n1 + k)
    return float(np.sum(n2 * (w2 - pred) ** 2) / np.sum(n2))


def fit_k(n1, w1, n2, w2, mu):
    """K minimising weighted out-of-sample MSE, by golden section on log K.

    Coarse grid first: golden section needs a unimodal bracket, and bracketing
    from the grid's minimum is cheaper than trusting the loss to be unimodal
    over four orders of magnitude. A minimum that lands on a bound is returned
    as the bound and flagged by the caller, never silently reported as a fit.
    """
    if len(n1) < 3:
        return float("nan")
    grid = np.exp(np.linspace(math.log(K_LO), math.log(K_HI), 60))
    vals = [loss(k, n1, w1, n2, w2, mu) for k in grid]
    # A pool with no talent spread has the same loss at every K, and the
    # argmin of a flat surface is whichever float noise won. Returning it would
    # print a confident four-digit K off nothing. Caught here rather than left
    # to the reader: the first dry run of this probe reported K = 3016 and a
    # -10% dMSE from exactly this, and both looked like measurements.
    #
    # Both tests are needed. The relative one catches a surface that is flat
    # but real; it does NOT catch an all-identical pool, where the loss is
    # float rounding around 1e-34 and the *relative* spread across that noise
    # is enormous. Hence the absolute floor: a weighted MSE below 1e-12 in
    # wOBA^2 means residuals under a millionth of a point of wOBA, which is not
    # baseball, it is the last bits of a double.
    hi_v, lo_v = max(vals), min(vals)
    if not math.isfinite(hi_v) or hi_v <= 1e-12 or (hi_v - lo_v) / hi_v < 1e-9:
        return float("nan")
    i = int(np.argmin(vals))
    lo = math.log(grid[max(i - 1, 0)])
    hi = math.log(grid[min(i + 1, len(grid) - 1)])
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc = loss(math.exp(c), n1, w1, n2, w2, mu)
    fd = loss(math.exp(d), n1, w1, n2, w2, mu)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(math.exp(c), n1, w1, n2, w2, mu)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(math.exp(d), n1, w1, n2, w2, mu)
        if b - a < 1e-6:
            break
    return float(math.exp((a + b) / 2))


def moments_k(n1, w1, n2, w2):
    """Variance-components K = sigma^2 / tau^2. Optimises nothing.

    The two periods are disjoint, so their sampling errors are independent and
    cov(w1, w2) estimates the between-pitcher talent variance directly. The
    per-BF outcome variance then falls out of the first period's total spread:

        var(w1) = tau^2 + sigma^2 * E[1/n1]

    Pitchers are weighted equally here, not by BF. That is the estimator's own
    requirement -- E[1/n1] has to be taken over the same population the variance
    is measured on -- and it is why this is a cross-check rather than the
    headline: equal weighting lets a 12-BF arm carry a September closer's vote.
    """
    if len(n1) < 10:
        return float("nan"), float("nan"), float("nan")
    m1, m2 = w1.mean(), w2.mean()
    tau2 = float(np.mean((w1 - m1) * (w2 - m2)))
    var1 = float(np.mean((w1 - m1) ** 2))
    e_inv = float(np.mean(1.0 / n1))
    sigma2 = (var1 - tau2) / e_inv if e_inv > 0 else float("nan")
    if not (math.isfinite(tau2) and math.isfinite(sigma2)) or tau2 <= 0 or sigma2 <= 0:
        return float("nan"), tau2, sigma2
    return sigma2 / tau2, tau2, sigma2


def bootstrap_k(rows, mu_mode, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Percentile CI for K, resampling pitchers.

    The prior is recomputed inside each draw when it is derived from the pool.
    Holding it fixed at the full-sample value would leak the whole sample into
    every draw and narrow the interval for free.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n < 10:
        return float("nan"), float("nan")
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        r = rows[idx]
        mu = mu_mode if np.isscalar(mu_mode) else _pool_mu(r)
        k = fit_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3], mu)
        if math.isfinite(k):
            out.append(k)
    if len(out) < draws // 2:
        return float("nan"), float("nan")
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def _pool_mu(rows):
    """BF-weighted first-period centre of whatever pool is passed in."""
    return float(np.sum(rows[:, 0] * rows[:, 1]) / np.sum(rows[:, 0]))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt_k(k):
    if not math.isfinite(k):
        return "     --"
    # A minimum pinned to a search bound is not a fit, and printing it as one
    # invites reading `5000` as an estimate. Flagged, never silently returned.
    if k >= K_HI * 0.999:
        return f"{'>' + format(K_HI, '.0f'):>7s}"
    if k <= K_LO * 1.001:
        return f"{'<' + format(K_LO, '.1f'):>7s}"
    return f"{k:7.1f}"


def summarise(rows, label, mu, shipped=100.0):
    """One fitted arm as a printable dict. Never returns a bare number."""
    if len(rows) < 3:
        return {"label": label, "n": len(rows), "k": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "dmse": float("nan"),
                "k_mom": float("nan"), "bf": 0.0, "ip_equiv": float("nan")}
    n1, w1, n2, w2 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    k = fit_k(n1, w1, n2, w2, mu)
    l_ship = loss(shipped, n1, w1, n2, w2, mu)
    l_fit = loss(k, n1, w1, n2, w2, mu) if math.isfinite(k) else float("nan")
    k_mom, _, _ = moments_k(n1, w1, n2, w2)
    bfip = rows[:, 4]
    bfip = bfip[np.isfinite(bfip)]
    rate = float(np.median(bfip)) if len(bfip) else float("nan")
    return {
        "label": label, "n": int(len(rows)), "k": k,
        "k_mom": k_mom,
        "dmse": ((l_ship - l_fit) / l_ship * 100.0
                 if l_ship > 0 and math.isfinite(l_fit) else float("nan")),
        "bf": float(np.median(n1)),
        "bf_per_ip": rate,
        "ip_equiv": k / rate if rate and math.isfinite(rate) else float("nan"),
        "mu": mu,
    }


def _row(s):
    return (f"  {s['label']:<26s} {s['n']:>5d} {_fmt_k(s['k'])} "
            f"{_fmt_k(s.get('k_mom', float('nan')))} "
            f"{s['dmse']:>8.2f}%  {s.get('ip_equiv', float('nan')):>7.1f} "
            f"{s.get('bf', 0):>7.0f}")


HEAD = ("  arm                            n       K   K_mom  dMSE vs100"
        "   IP-eq  med BF\n  " + "-" * 74)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", default="2023,2024,2025,2026",
                    help="seasons to fit (comma-separated)")
    ap.add_argument("--role-cut", type=float, default=0.20,
                    help="max share of appearances started to count as relief")
    ap.add_argument("--shipped", type=float, default=100.0,
                    help="the K currently in build_site.py, for the dMSE column")
    ap.add_argument("--weight-jitter", type=float, default=0.05,
                    help="relative wOBA-weight perturbation for the sensitivity arm")
    ap.add_argument("--draws", type=int, default=BOOT_DRAWS)
    ap.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args(argv)

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    L = []

    def say(line=""):
        print(line)
        L.append(line)

    # ---- pull -------------------------------------------------------------
    print("Pulling StatsAPI monthly splits ...", file=sys.stderr)
    monthly = {}
    for season in seasons:
        for group in ("pitching", "hitting"):
            try:
                monthly[(season, group)] = fetch_periods(season, group, "byMonth")
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  {season} {group} byMonth unavailable: {e!r}", file=sys.stderr)

    say("Reliever regression constant -- fitted, not inherited")
    say("=" * 78)
    say(f"seasons {seasons} | role cut: started <= {a.role_cut:.0%} of appearances")
    say(f"shipped XWOBA_SHRINK_K = {a.shipped:.0f} (batters, starters and relievers alike)")
    say("")
    say("K is a pseudo-sample of league-average BF. Higher K = regress harder.")
    say("IP-eq converts K at the pool's own median BF/IP, for comparison with")
    say("Marcel's IP-denominated 60 (SP) / 25 (RP). dMSE is the out-of-sample")
    say("weighted MSE the fitted K buys over the shipped one -- read it first.")
    say("")

    # ---- which call shape answered ----------------------------------------
    # Printed, not assumed. The first version of this probe hard-coded one
    # shape and CI measured it returning zero stat blocks, so the shape is now
    # a finding in its own right: the next person changing this fetch should be
    # reading measurements rather than re-deriving them from the docs.
    if _shape_log:
        say("StatsAPI call shapes tried (the within-season split depends on these)")
        say("  " + "-" * 74)
        seen = set()
        for name, season, group, outcome in _shape_log:
            if (name, group) in seen and not outcome.startswith("OK"):
                continue
            seen.add((name, group))
            say(f"  {name:<20s} {season} {group:<9s} {outcome}")
        say("")

    if not any(monthly.values()):
        say("!! No candidate shape produced a within-season split. Every arm")
        say("!! below that needs two periods inside one season is absent; the")
        say("!! season-pair arm still runs, on the proven `stats=season` shape.")
        say("")

    # ---- per-season within-season fits ------------------------------------
    # The split point is chosen to divide each season's BF as evenly as the
    # month boundaries allow, rather than fixed at a date: seasons differ in
    # length and 2026 is partial. Balance is a property of the data, so it is
    # measured per season instead of asserted once.
    say("Within-season split (period 1 -> period 2), relievers")
    say(HEAD)
    canonical = {}
    for season in seasons:
        per = monthly.get((season, "pitching"))
        if not per:
            continue
        months = sorted({p for _, p in per})
        best, best_bal = None, -1.0
        for s in months[:-1]:
            rows = build_pairs(per, s, "RP", a.role_cut)
            if len(rows) < 20:
                continue
            bal = min(rows[:, 0].sum(), rows[:, 2].sum())
            if bal > best_bal:
                best, best_bal = s, bal
        if best is None:
            say(f"  {season}: no split with enough relievers on both sides")
            continue
        for s in months[:-1]:
            rows = build_pairs(per, s, "RP", a.role_cut)
            if len(rows) < 20:
                continue
            tag = "*" if s == best else " "
            say(_row(summarise(rows, f"{season} split after M{s}{tag}",
                               _pool_mu(rows), a.shipped)))
        canonical[season] = best
    say("  (* = the balanced split, pooled into the headline below)")
    say("")

    # ---- headline ---------------------------------------------------------
    pooled = {}
    for role, name in (("RP", "relievers"), ("SP", "starters")):
        chunks = [build_pairs(monthly[(s, "pitching")], canonical[s], role, a.role_cut)
                  for s in canonical if (s, "pitching") in monthly]
        chunks = [c for c in chunks if len(c)]
        pooled[role] = np.vstack(chunks) if chunks else np.empty((0, 5))
    bat = []
    for s in canonical:
        if (s, "hitting") in monthly:
            bat.append(build_pairs(monthly[(s, "hitting")], canonical[s], None, a.role_cut))
    pooled["BAT"] = np.vstack([c for c in bat if len(c)]) if bat else np.empty((0, 5))

    say("Pooled across seasons, by population -- the control arms are the point")
    say(HEAD)
    summaries = {}
    for key, name in (("RP", "relievers (BF)"), ("SP", "starters (BF)"),
                      ("BAT", "batters (PA)")):
        rows = pooled[key]
        if not len(rows):
            continue
        s = summarise(rows, name, _pool_mu(rows), a.shipped)
        summaries[key] = s
        say(_row(s))
    say("")

    for key, name in (("RP", "relievers"), ("SP", "starters"), ("BAT", "batters")):
        if key not in summaries or not len(pooled[key]):
            continue
        lo, hi = bootstrap_k(pooled[key], None, draws=a.draws)
        summaries[key]["lo"], summaries[key]["hi"] = lo, hi
        if not (math.isfinite(lo) and math.isfinite(hi)):
            verdict = "CI unavailable -- too many draws failed to fit"
        elif lo <= a.shipped <= hi:
            verdict = f"contains {a.shipped:.0f} -- no case for moving it"
        else:
            verdict = f"excludes {a.shipped:.0f}"
        say(f"  {name:<12s} K = {_fmt_k(summaries[key]['k']).strip():>8s}"
            f"  95% CI [{lo:7.1f}, {hi:7.1f}]  {verdict}")
    say("")

    # ---- prior sensitivity ------------------------------------------------
    # The build shrinks relievers toward the league-wide PA-weighted centre,
    # which is a batter-population number and sits above the relief pool's own
    # centre. A K fitted against a displaced target absorbs the offset by
    # shrinking less, so the two fits are reported side by side rather than one
    # being chosen here.
    if len(pooled.get("RP", [])) and len(pooled.get("BAT", [])):
        mu_rp = _pool_mu(pooled["RP"])
        mu_lg = _pool_mu(pooled["BAT"])
        say("Prior sensitivity -- which centre relievers are shrunk toward")
        say(HEAD)
        say(_row(summarise(pooled["RP"], "toward relief centre", mu_rp, a.shipped)))
        say(_row(summarise(pooled["RP"], "toward league centre", mu_lg, a.shipped)))
        say(f"  relief centre {mu_rp:.4f} vs league centre {mu_lg:.4f} "
            f"(gap {mu_lg - mu_rp:+.4f} wOBA); the build uses the league centre")
        say("")

    # ---- robustness -------------------------------------------------------
    say("Robustness -- a K that moves with these is a fit to one choice")
    say(HEAD)
    for cut in (0.0, 0.10, 0.20, 0.35):
        chunks = [build_pairs(monthly[(s, "pitching")], canonical[s], "RP", cut)
                  for s in canonical if (s, "pitching") in monthly]
        chunks = [c for c in chunks if len(c)]
        if not chunks:
            continue
        r = np.vstack(chunks)
        say(_row(summarise(r, f"role cut <= {cut:.0%}", _pool_mu(r), a.shipped)))
    jw = {k: v * (1.0 + a.weight_jitter * (1 if i % 2 else -1))
          for i, (k, v) in enumerate(sorted(WOBA_W.items()))}
    chunks = [build_pairs(monthly[(s, "pitching")], canonical[s], "RP",
                          a.role_cut, weights=jw)
              for s in canonical if (s, "pitching") in monthly]
    chunks = [c for c in chunks if len(c)]
    if chunks:
        r = np.vstack(chunks)
        say(_row(summarise(r, f"wOBA weights +/-{a.weight_jitter:.0%}",
                           _pool_mu(r), a.shipped)))
    # A minimum-BF filter is a threshold, and this repo's standing lesson is
    # that the fix for a threshold is a weight. BF weighting already does that
    # job here; these rows exist to show the fit does not depend on it.
    if len(pooled.get("RP", [])):
        for floor in (0, 25, 75):
            r = pooled["RP"][pooled["RP"][:, 0] >= floor]
            if len(r) < 20:
                continue
            say(_row(summarise(r, f"first-period BF >= {floor}", _pool_mu(r), a.shipped)))
    say("")

    # ---- league prior by role and hand ------------------------------------
    # Printed with the displacement each pool suffers, not just the centre.
    # A gap in wOBA points is not actionable on its own -- what matters is
    # gap * K/(n + K), the share of it that actually lands in a published rate
    # at the shipped K and that pool's own median sample.
    say("League prior by role and hand -- the target, not the constant")
    say("  NOTE: levels here are NOT comparable to build_site's Savant-derived")
    say("  target (different wOBA weights). Only gaps BETWEEN pools are, since")
    say("  the same weights apply to every pool. Do not paste a centre below")
    say("  into the build -- derive it per-build from Savant.")
    say("")
    say(f"  {'pool':<10s} {'n':>5s} {'wtd':>8s} {'unwtd':>8s} "
        f"{'vs BAT':>8s} {'medPA':>6s} {'displ@K=100':>12s}")
    say("  " + "-" * 68)
    for season in seasons:
        try:
            cent = role_centres(season, a.role_cut)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            say(f"  {season}: unavailable ({e!r})")
            continue
        ref = (cent.get("BAT") or {}).get("wtd")
        say(f"  -- {season} --")
        for key in ("BAT", "BAT-L", "BAT-R", "SP", "SP-L", "SP-R",
                    "RP", "RP-L", "RP-R"):
            c = cent.get(key)
            if not c:
                continue
            gap = (c["wtd"] - ref) if ref is not None else float("nan")
            # What the single shared target actually costs a member of this
            # pool: the gap times the prior's weight at the shipped K.
            displ = gap * (a.shipped / (c["med"] + a.shipped))
            say(f"  {key:<10s} {c['n']:>5d} {c['wtd']:>8.4f} {c['unw']:>8.4f} "
                f"{gap:>+8.4f} {c['med']:>6.0f} {displ:>+12.4f}")
        for role in ("SP", "RP"):
            l, r = cent.get(f"{role}-L"), cent.get(f"{role}-R")
            if l and r:
                say(f"  {role} L-R hand gap: {l['wtd'] - r['wtd']:+.4f} wOBA "
                    f"(L n={l['n']}, R n={r['n']})")
    say("")
    say("  Read the hand rows against the platoon term already in the model.")
    say("  wOBA v2 applies an exposure-centred 0.021 starter platoon gap. That")
    say("  models batter-vs-hand matchup; an L/R TARGET models the mean level")
    say("  of each hand's population. They are different quantities, but they")
    say("  overlap to the extent a hand's centre is itself set by the batters")
    say("  it faces -- so shipping both without checking is a double count.")
    say("")

    # ---- season-pair arm --------------------------------------------------
    say("Season pair (year N -> year N+1) -- Marcel's comparison, upper bound")
    say(HEAD)
    for lo_s, hi_s in zip(seasons, seasons[1:]):
        try:
            p_lo = fetch_periods(lo_s, "pitching", "season")
            p_hi = fetch_periods(hi_s, "pitching", "season")
        except Exception as e:  # noqa: BLE001
            say(f"  {lo_s}->{hi_s}: unavailable ({e!r})")
            continue
        merged = {(pid, 0): rec for (pid, _), rec in p_lo.items()}
        merged.update({(pid, 1): rec for (pid, _), rec in p_hi.items()})
        rows = build_pairs(merged, 0, "RP", a.role_cut)
        if len(rows) < 20:
            say(f"  {lo_s}->{hi_s}: too few paired relievers ({len(rows)})")
            continue
        say(_row(summarise(rows, f"{lo_s} -> {hi_s} RP", _pool_mu(rows), a.shipped)))
    say("")

    say("Marcel's published baselines, for scale: 200 PA (hitters), 60 IP (SP),")
    say("25 IP (RP) -- IP-denominated, and stated as a no-projection filler. The")
    say("IP-eq column is the only place these are commensurable, and the")
    say("within-season arm is not measuring the same thing Marcel's year-to-year")
    say("constants are. Compare the season-pair arm to Marcel; ship from the")
    say("within-season arm.")
    say("")
    say("Before any of this moves XWOBA_SHRINK_K: a role-split K is lean-path,")
    say("so it bumps MODEL_TAG, and the flip count that decides whether it also")
    say("splits the record family cannot be computed here -- historical slates")
    say("would need a Savant re-pull, which is lookahead. Run the new K forward")
    say("on live slates against the shipped one and count flips there.")

    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
