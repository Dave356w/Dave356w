"""Is a walk-forward Pythagorean arm worth a control tile? Measure, don't assume.

Run this where StatsAPI is reachable -- for this repo that means a GitHub
runner, via `.github/workflows/pythag-probe.yml`.

Why this exists
---------------
`_baseline_controls()` scores always-home and always-chalk on the graded rows,
because a .568 headline is only a result next to something. A third arm --
team strength, combined into an implied win% -- was in the repo once and was
removed in a UI declutter (CLAUDE.md, "Deleting controls as clutter").

A first prototype rebuilt it from the ledger's own final scores and scored
**188-188 (.500)** over the full parameter sweep. That was not a tuning
failure: unshrunk, the median |p_home - .5| was 0.130, so the arm made
confident picks -- they just carried no information (top-half-edge .489 vs
bottom-half .503, i.e. a bigger edge did not hit more often). The cause was
the input, not the method: the ledger only holds games the model published a
lean on, so a club had a median of 12 prior games at pick time, and those 12
are a biased subsample of its season.

This probe replaces that input with the real one: every regular-season game
from StatsAPI, which by August is ~110 games per club instead of 12.

Why pulling today's schedule is NOT lookahead
---------------------------------------------
The `.savant_cache/` rule exists because a Savant leaderboard is a *season
aggregate*: today's pull cannot tell you what it read on 3 July, so historical
rows cannot be re-derived from it. A game result is not like that. "SEA beat
TEX 5-3 on 12 July" is a timestamped event that reads the same whenever it is
fetched. Filtering to `game_date < D` is therefore exact, not approximate, and
every number below is computed from games that had already finished when the
lean was published.

The sweep
---------
Rolling windows are game-indexed, not calendar-indexed, which is why this pulls
the schedule rather than differencing standings snapshots: standings give
cumulative totals per *date*, so differencing them yields a window in days, and
a club's last 30 *games* is the quantity that actually means something.

  window   last 30 / 60 / 90 games, or the whole season to date
  expo     Pythagorean exponent (1.83 is the usual MLB fit; 2.0 is Bill James')
  shrink   pseudo-games toward .500 -- a weight, not a gate, so no config
           switches discontinuously (CLAUDE.md threshold-cliff lesson)

Reading the output honestly
---------------------------
The grid is ~32 configs against ~389 rows. Reporting its maximum would be
selection on noise, and the max of 32 draws beats the mean by construction. So
the report leads with the *spread* across configs, and the number to actually
believe is the held-out one: the config chosen on the first half of the rows,
scored on the second half it never saw. A config that wins in-sample and
reverts out-of-sample is the expected outcome, not a surprise.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

from fetch_headers import FETCH_HEADERS

LEDGER = "data/mlb_lean_ledger.csv"
LEDGER2SA = {"ARI": "AZ"}          # ledger -> StatsAPI abbr (mirrors market_backfill)
NON_CLUBS = {"AME", "NAT"}         # All-Star sides; not MLB clubs


def _get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=FETCH_HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if k == tries - 1:
                raise
            time.sleep(1.5 * (2 ** k))


def _is_final(gm):
    """Finished games report 'F' or 'O' in codedGameState; abstractGameState
    is the authority. Requiring 'F' alone silently drops completed games."""
    st = gm.get("status") or {}
    return (st.get("abstractGameState") == "Final"
            or st.get("codedGameState") in ("F", "O"))


def _days(start, end):
    d = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while d <= stop:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def fetch_game_log(season, end_date, verbose=None):
    """(abbr) -> sorted list of (date, runs_scored, runs_allowed), regular season.

    Queries one day at a time with `?sportId=1&date=YYYY-MM-DD`. That is more
    requests than a date range, and it is deliberate: this exact call shape is
    what market_backfill._statsapi_day runs successfully on every build, so it
    is the one shape in this repo known to work. Two attempts at the range form
    (`startDate`/`endDate`, with and without month chunking) both came back with
    zero dates, and guessing at range parameters from a sandbox that cannot
    reach StatsAPI is how that time got spent. A proven call beats a clever one.

    Diagnostics print as they happen rather than accumulating into the report:
    the first version collected them into the report string, which is only
    emitted at the end, so the fetch failure discarded exactly the lines needed
    to diagnose it.
    """
    per = defaultdict(list)
    seen = set()
    days = list(_days(f"{season}-03-01", end_date))
    n_days_with_games = 0
    n_raw = 0                      # games seen at all, before any filtering
    for i, d in enumerate(days):
        js = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}")
        games = [gm for day in js.get("dates", []) for gm in day.get("games", [])]
        if games:
            n_days_with_games += 1
        n_raw += len(games)
        for gm in games:
            pk = gm.get("gamePk")
            if pk in seen or gm.get("gameType") != "R" or not _is_final(gm):
                continue
            t = gm.get("teams") or {}
            a, h = t.get("away") or {}, t.get("home") or {}
            sa, sh = a.get("score"), h.get("score")
            if sa is None or sh is None:
                continue
            ab_a = ((a.get("team") or {}).get("abbreviation"))
            ab_h = ((h.get("team") or {}).get("abbreviation"))
            if not ab_a or not ab_h:
                continue
            seen.add(pk)
            per[ab_a].append((d, float(sa), float(sh)))
            per[ab_h].append((d, float(sh), float(sa)))
        if i % 30 == 0 or d == days[-1]:
            line = (f"    {d}: {len(seen):4d} final regular-season games so far "
                    f"({n_raw} seen, {n_days_with_games} days with any)")
            print(line, flush=True)
            if verbose is not None:
                verbose.append(line)
        time.sleep(0.12)
    if not seen:
        # Separate "the query returned nothing" from "the filter ate everything";
        # they have completely different fixes and the message should say which.
        raise SystemExit(
            f"FAIL: no finished regular-season games for {season} up to "
            f"{end_date} after querying {len(days)} days.\n"
            f"  days returning any game: {n_days_with_games}\n"
            f"  games seen before filtering: {n_raw}\n"
            + ("  -> the query itself came back empty; the date range or season "
               "is wrong.\n" if n_raw == 0 else
               "  -> games WERE returned and the gameType/final filter removed "
               "all of them; inspect gameType and status values.\n")
            + "Refusing to print a report built on an empty log.")
    return {k: sorted(v) for k, v in per.items()}


class Log:
    """Prefix-summed game log; O(log n) window queries strictly before a date."""

    def __init__(self, games):
        self.dates = [g[0] for g in games]
        rs = np.cumsum([0.0] + [g[1] for g in games])
        ra = np.cumsum([0.0] + [g[2] for g in games])
        self.rs, self.ra = rs, ra

    def window(self, before, w):
        """(runs_scored, runs_allowed, n_games) over the last `w` games strictly
        before `before`. w <= 0 means the whole season to date."""
        i = bisect.bisect_left(self.dates, before)   # games with date < before
        if i == 0:
            return 0.0, 0.0, 0
        j = 0 if w <= 0 else max(0, i - w)
        return float(self.rs[i] - self.rs[j]), float(self.ra[i] - self.ra[j]), i - j


def pyth(rs, ra, n, expo, K):
    base = 0.5 if (rs + ra) <= 0 else (rs ** expo) / (rs ** expo + ra ** expo)
    if n + K == 0:
        return 0.5
    return (n * base + K * 0.5) / (n + K)


def log5(ph, pa):
    d = ph + pa - 2 * ph * pa
    return 0.5 if d == 0 else (ph - ph * pa) / d


def load_rows():
    led = pd.read_csv(LEDGER)
    fa = pd.to_numeric(led["full_away"], errors="coerce")
    fh = pd.to_numeric(led["full_home"], errors="coerce")
    keep = (fa.notna() & fh.notna() & (fa != fh)
            & ~led["away"].isin(NON_CLUBS) & ~led["home"].isin(NON_CLUBS))
    g = led[keep].copy()
    g["home_won"] = (fh[keep] > fa[keep]).values
    g["p_close"] = pd.to_numeric(g.get("close_p_home"), errors="coerce") \
        if "close_p_home" in g.columns else np.nan
    return g.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def run_config(rows, logs, w, expo, K, eps=1e-9):
    """Returns a bool array `picked_home` and a mask of non-abstentions."""
    n = len(rows)
    pick = np.zeros(n, dtype=bool)
    live = np.zeros(n, dtype=bool)
    for i, r in rows.iterrows():
        la = logs.get(LEDGER2SA.get(r["away"], r["away"]))
        lh = logs.get(LEDGER2SA.get(r["home"], r["home"]))
        if la is None or lh is None:
            continue
        d = r["game_date"]
        rsa, raa, na = la.window(d, w)
        rsh, rah, nh = lh.window(d, w)
        p = log5(pyth(rsh, rah, nh, expo, K), pyth(rsa, raa, na, expo, K))
        if abs(p - 0.5) < eps:
            continue                      # exact .500 abstains; never defaults home
        live[i] = True
        pick[i] = p > 0.5
    return pick, live


def rec(pick, live, won, sub=None):
    m = live if sub is None else (live & sub)
    hit = (pick == won) & m
    w = int(hit.sum())
    l = int(m.sum()) - w
    return w, l, (w / (w + l) if (w + l) else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--windows", default="30,60,90,0",
                    help="game-count windows; 0 = whole season to date")
    ap.add_argument("--expos", default="1.83,2.0")
    ap.add_argument("--shrink", default="0,6,12,25")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",")]
    expos = [float(x) for x in args.expos.split(",")]
    shrinks = [float(x) for x in args.shrink.split(",")]

    rows = load_rows()
    end = rows["game_date"].max()
    rep = ["=" * 74, "WALK-FORWARD PYTHAGOREAN CONTROL — probe", "=" * 74,
           f"  ledger rows (graded, decided, MLB clubs): {len(rows)}",
           f"  date range: {rows['game_date'].min()} -> {end}"]

    rep.append("  StatsAPI fetch:")
    raw = fetch_game_log(args.season, end, verbose=rep)
    logs = {k: Log(v) for k, v in raw.items()}
    depth = [len(v) for v in raw.values()]
    rep.append(f"  StatsAPI clubs: {len(logs)}  |  final games per club: "
               f"median {np.median(depth):.0f}, min {min(depth)}, max {max(depth)}")
    rep.append("  (the ledger-only prototype had a median of 12 prior games per pick)")

    # Provenance, not an assumption: a club whose abbreviation does not match
    # StatsAPI's would be silently skipped by run_config, quietly shrinking the
    # scored set. Count it and name the offenders instead.
    miss = sorted({t for r in (rows["away"].tolist() + rows["home"].tolist())
                   for t in [LEDGER2SA.get(r, r)] if t not in logs})
    n_miss = int(sum(1 for _, r in rows.iterrows()
                     if LEDGER2SA.get(r["away"], r["away"]) not in logs
                     or LEDGER2SA.get(r["home"], r["home"]) not in logs))
    rep.append(f"  ledger clubs absent from StatsAPI log: "
               f"{miss if miss else 'none'}  ({n_miss} rows unscoreable)")

    won = rows["home_won"].to_numpy()
    all_true = np.ones(len(rows), dtype=bool)
    hw, hl, hr = rec(np.ones(len(rows), dtype=bool), all_true, won)
    rep += ["", f"  control · always-home : {hw}-{hl} ({hr:.3f})"]
    pc = rows["p_close"].to_numpy(dtype=float)
    mk = ~np.isnan(pc)
    if mk.any():
        mw, ml, mr = rec(pc >= 0.5, mk, won)
        rep.append(f"  control · always-chalk: {mw}-{ml} ({mr:.3f})  n={int(mk.sum())}")

    # ---------------------------------------------------------------- grid --
    rep += ["", "IN-SAMPLE GRID (all rows) — read the spread, not the maximum",
            f"  {'window':>7} {'expo':>5} {'K':>5} {'W':>4} {'L':>4} {'rate':>7} {'abst':>5}"]
    results = {}
    for w, e, K in itertools.product(windows, expos, shrinks):
        pick, live = run_config(rows, logs, w, e, K)
        W, L, R = rec(pick, live, won)
        results[(w, e, K)] = (pick, live, R)
        rep.append(f"  {('season' if w <= 0 else w):>7} {e:5.2f} {K:5.0f} "
                   f"{W:4d} {L:4d} {R:7.3f} {int((~live).sum()):5d}")
    rates = np.array([v[2] for v in results.values()])
    rep += ["", f"  spread across {len(rates)} configs: min {rates.min():.3f} "
                f"max {rates.max():.3f} mean {rates.mean():.3f} sd {rates.std():.3f}"]

    # ------------------------------------------------------- held-out split --
    mid = len(rows) // 2
    early = np.zeros(len(rows), dtype=bool); early[:mid] = True
    late = ~early
    best, best_r = None, -1.0
    for key, (pick, live, _) in results.items():
        _, _, r = rec(pick, live, won, early)
        if r == r and r > best_r:
            best_r, best = r, key
    pick, live, _ = results[best]
    ew, el, er = rec(pick, live, won, early)
    lw, ll, lr = rec(pick, live, won, late)
    oos = [rec(p, lv, won, late)[2] for (p, lv, _) in results.values()]
    oos = np.array([x for x in oos if x == x])
    rep += ["", "HELD-OUT (the number to believe)",
            f"  split at row {mid} ({rows['game_date'].iloc[mid]})",
            f"  best config on the FIRST half : window={'season' if best[0] <= 0 else best[0]}"
            f" expo={best[1]} K={best[2]:.0f}  -> {ew}-{el} ({er:.3f})",
            f"  same config on the SECOND half: {lw}-{ll} ({lr:.3f})",
            f"  all configs on the second half: mean {oos.mean():.3f} sd {oos.std():.3f}",
            f"  always-home on the second half: "
            f"{rec(np.ones(len(rows), dtype=bool), late, won)[2]:.3f}"]

    # ------------------------------------------------------------- verdict --
    rep += ["", "=" * 74, "READING"]
    if lr <= 0.52:
        rep += ["  Held-out rate is at or near a coin flip. Team strength via",
                "  Pythagorean+log5 does not earn a control tile: a yardstick that",
                "  measures nothing is worse than the two already on the page.",
                "  Do NOT wire this into _baseline_controls()."]
    elif lr < er - 0.03:
        rep += ["  In-sample rate did not survive the split -- the grid maximum was",
                "  selection on noise. Do not ship the winning config; if anything",
                "  ship a fixed, pre-registered config and re-measure next month."]
    else:
        rep += ["  Held-out rate holds up. A control tile is defensible -- wire the",
                "  HELD-OUT config (not the grid maximum) into _baseline_controls()",
                "  and state its n, as always-chalk already does."]
    rep.append("=" * 74)

    text = "\n".join(rep)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
