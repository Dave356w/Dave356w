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


def _bail(season, end_date, days, n_days_with_games, n_raw,
          rej, seen_types, seen_states, sample, early):
    """Die with everything needed to name the culprit in one read.

    The point is that the reader should not have to guess which test rejected
    what -- the first three failures of this probe were each 'fixed' by a guess
    because the message named no field. Rejection counts, the gameType and
    status values actually observed, and two raw samples between them make the
    answer unambiguous."""
    when = (f"after {n_days_with_games} days with games (stopping early)"
            if early else f"after querying {len(days)} days")
    raise SystemExit(
        f"FAIL: no finished regular-season games for {season} up to "
        f"{end_date} {when}.\n"
        f"  days returning any game: {n_days_with_games}\n"
        f"  games seen before filtering: {n_raw}\n"
        f"  rejected by reason: {sorted(rej.items(), key=lambda kv: -kv[1])[:6]}\n"
        f"  gameType values seen: "
        f"{dict(sorted(seen_types.items(), key=lambda kv: -kv[1])[:8])}\n"
        f"  abstract/coded states seen: "
        f"{dict(sorted(seen_states.items(), key=lambda kv: -kv[1])[:8])}\n"
        f"  sample games: {json.dumps(sample, default=str)[:900]}\n"
        "Refusing to print a report built on an empty log.")


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
    # Rejections counted by reason. The previous run proved the query works
    # (2076 games over 154 days) and that the filter dropped all of them, but
    # not WHICH test did it -- so the fix was still a guess. These counters
    # name the culprit instead.
    rej = defaultdict(int)
    seen_types, seen_states = defaultdict(int), defaultdict(int)
    sample = []
    for i, d in enumerate(days):
        # `hydrate=team` is load-bearing, not decoration: without it the team
        # object carries only id/link/name, `abbreviation` is absent, and every
        # game is rejected for a missing abbreviation. That is exactly what
        # happened, and it is why market_backfill._statsapi_day -- the call this
        # one is modelled on -- hydrates `team`. Dropping it while "simplifying"
        # the call is what cost three CI rounds.
        js = _get("https://statsapi.mlb.com/api/v1/schedule"
                  f"?sportId=1&date={d}&hydrate=team")
        games = [gm for day in js.get("dates", []) for gm in day.get("games", [])]
        if games:
            n_days_with_games += 1
        n_raw += len(games)
        for gm in games:
            st = gm.get("status") or {}
            seen_types[str(gm.get("gameType"))] += 1
            seen_states[f"{st.get('abstractGameState')}/{st.get('codedGameState')}"] += 1
            if len(sample) < 2:
                t0 = gm.get("teams") or {}
                sample.append({
                    "top_level_keys": sorted(gm.keys())[:14],
                    "gameType": gm.get("gameType"),
                    "status": {k: st.get(k) for k in
                               ("abstractGameState", "codedGameState", "detailedState")},
                    "away": {k: (t0.get("away") or {}).get(k) for k in ("score",)},
                    "home": {k: (t0.get("home") or {}).get(k) for k in ("score",)},
                    "away_team_keys": sorted(((t0.get("away") or {}).get("team") or {}).keys())[:10],
                })
            pk = gm.get("gamePk")
            if pk in seen:
                rej["duplicate"] += 1
                continue
            if gm.get("gameType") != "R":
                rej[f"gameType={gm.get('gameType')!r}"] += 1
                continue
            if not _is_final(gm):
                rej["not final"] += 1
                continue
            t = gm.get("teams") or {}
            a, h = t.get("away") or {}, t.get("home") or {}
            sa, sh = a.get("score"), h.get("score")
            if sa is None or sh is None:
                rej["missing score"] += 1
                continue
            ab_a = ((a.get("team") or {}).get("abbreviation"))
            ab_h = ((h.get("team") or {}).get("abbreviation"))
            if not ab_a or not ab_h:
                rej["missing team abbreviation"] += 1
                continue
            seen.add(pk)
            per[ab_a].append((d, float(sa), float(sh)))
            per[ab_h].append((d, float(sh), float(sa)))
        # Fail fast on a structural mismatch -- but the condition has to be
        # "regular-season games arrived and none survived", not "nothing
        # survived". The first version tripped on 2026-03-01..05, which is
        # entirely spring training ('S' and 'E'): keeping none of those is
        # correct behaviour, not a fault, and the season simply had not started
        # yet. Gate on `R` games actually seen so the guard cannot fire before
        # there is anything it could be right about.
        if not seen and seen_types.get("R", 0) >= 20:
            _bail(season, end_date, days, n_days_with_games, n_raw,
                  rej, seen_types, seen_states, sample, early=True)
        if i % 30 == 0 or d == days[-1]:
            line = (f"    {d}: {len(seen):4d} final regular-season games so far "
                    f"({n_raw} seen, {n_days_with_games} days with any)")
            print(line, flush=True)
            if verbose is not None:
                verbose.append(line)
        time.sleep(0.12)
    if not seen:
        _bail(season, end_date, days, n_days_with_games, n_raw,
              rej, seen_types, seen_states, sample, early=False)
    # Provenance on the success path too: a filter that quietly drops a third
    # of the season would otherwise look identical to one that drops nothing.
    kept_line = (f"    kept {len(seen)} of {n_raw} games; "
                 f"rejected: {dict(sorted(rej.items(), key=lambda kv: -kv[1])[:5])}")
    print(kept_line, flush=True)
    if verbose is not None:
        verbose.append(kept_line)
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
    prob = np.full(n, np.nan)
    for i, r in rows.iterrows():
        la = logs.get(LEDGER2SA.get(r["away"], r["away"]))
        lh = logs.get(LEDGER2SA.get(r["home"], r["home"]))
        if la is None or lh is None:
            continue
        d = r["game_date"]
        rsa, raa, na = la.window(d, w)
        rsh, rah, nh = lh.window(d, w)
        p = log5(pyth(rsh, rah, nh, expo, K), pyth(rsa, raa, na, expo, K))
        prob[i] = p
        if abs(p - 0.5) < eps:
            continue                      # exact .500 abstains; never defaults home
        live[i] = True
        pick[i] = p > 0.5
    return pick, live, prob


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _logistic_fit(X, y, iters=50):
    """Plain IRLS. Returns (beta, se). No sklearn/statsmodels dependency --
    requirements.txt carries neither, and the build job has no reason to."""
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            b_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    eta = X @ b
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    try:
        cov = np.linalg.inv((X.T * w) @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full_like(b, np.nan)
    return b, se


def _dec(ml):
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / (-ml))


def rec(pick, live, won, sub=None):
    m = live if sub is None else (live & sub)
    hit = (pick == won) & m
    w = int(hit.sum())
    l = int(m.sum()) - w
    return w, l, (w / (w + l) if (w + l) else float("nan"))


def vs_market(rows, prob, won, rep, label):
    """Does the Pythagorean probability say anything the closing line doesn't?

    Accuracy is the wrong test here. A signal can pick the right side often and
    still be worthless if the market already knew -- and run differential is
    public, so the prior is that it did. The test that answers the question is
    conditional: regress the outcome on the market's log-odds AND the
    Pythagorean log-odds together. The market coefficient absorbs everything
    the line already prices; whatever is left on the Pythagorean coefficient is
    its marginal contribution. A coefficient within ~2 se of zero means no
    information beyond the close, however good its raw hit rate looked.

    Brier is reported alongside because it is the honest scoring rule for a
    probability, and because "worse Brier than the market" and "no marginal
    information" are different failures worth telling apart.
    """
    m = rows["p_close"].to_numpy(dtype=float)
    ok = ~np.isnan(m) & ~np.isnan(prob)
    n = int(ok.sum())
    if n < 40:
        rep.append(f"  {label}: only {n} rows carry a close; skipping")
        return
    pm, pp, y = m[ok], prob[ok], won[ok].astype(float)
    rep.append(f"  {label}  (n={n})")
    rep.append(f"    corr(pythag, market)      {np.corrcoef(pp, pm)[0, 1]:+.3f}")
    rep.append(f"    mean |pythag - market|    {np.mean(np.abs(pp - pm)):.4f}")
    rep.append(f"    Brier  pythag {np.mean((pp - y) ** 2):.4f}   "
               f"market {np.mean((pm - y) ** 2):.4f}   "
               f"(lower is better; .25 = always .500)")
    b, se = _logistic_fit(np.column_stack([_logit(pm), _logit(pp)]), y)
    rep.append(f"    outcome ~ market_logit + pythag_logit")
    rep.append(f"      market coef {b[1]:+.3f} +/- {se[1]:.3f}"
               f"   (t {b[1] / se[1] if se[1] else float('nan'):+.2f})")
    rep.append(f"      pythag coef {b[2]:+.3f} +/- {se[2]:.3f}"
               f"   (t {b[2] / se[2] if se[2] else float('nan'):+.2f})"
               "   <- marginal information")
    verdict = ("no marginal information beyond the close"
               if abs(b[2]) < 2 * (se[2] or np.inf)
               else "carries information the close does not")
    rep.append(f"      => {verdict}")

    # Practical form of the same question: bet pythag's side when it disagrees
    # with the market by more than a threshold, priced at the close.
    hml = pd.to_numeric(rows.get("close_home_ml"), errors="coerce").to_numpy(dtype=float)[ok]
    aml = pd.to_numeric(rows.get("close_away_ml"), errors="coerce").to_numpy(dtype=float)[ok]
    if np.isnan(hml).all():
        return
    rep.append("    disagreement betting at the close (flat 1u):")
    for thr in (0.00, 0.03, 0.05, 0.10):
        sel = np.abs(pp - pm) >= thr
        sel &= ~np.isnan(hml) & ~np.isnan(aml)
        if sel.sum() < 15:
            continue
        take_home = pp[sel] > pm[sel]
        w = np.where(take_home, y[sel] == 1, y[sel] == 0)
        ml = np.where(take_home, hml[sel], aml[sel])
        profit = np.where(w, np.array([_dec(x) - 1 for x in ml]), -1.0)
        roi = profit.sum()
        rep.append(f"      |diff| >= {thr:.2f}: {int(sel.sum()):3d} bets  "
                   f"{int(w.sum()):3d}-{int(sel.sum() - w.sum()):3d}  "
                   f"{roi:+.2f}u  ({100 * roi / sel.sum():+.1f}% ROI)")


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
        pick, live, _ = run_config(rows, logs, w, e, K)
        W, L, R = rec(pick, live, won)
        results[(w, e, K)] = (pick, live, R)
        rep.append(f"  {('season' if w <= 0 else w):>7} {e:5.2f} {K:5.0f} "
                   f"{W:4d} {L:4d} {R:7.3f} {int((~live).sum()):5d}")
    # Report the spread over DISTINCT outcomes, not over grid cells. The first
    # run printed "spread across 32 configs ... sd 0.018" when the 32 cells
    # collapse to 4 distinct records: at ~113 games per club, neither the
    # exponent (1.83 vs 2.0) nor the shrink K (0..25) ever changes a pick, so
    # only the window does anything. Quoting 32 there would overstate how much
    # was actually varied -- the same sin the held-out split exists to avoid.
    rates = np.array([v[2] for v in results.values()])
    distinct = sorted({(W, L) for (W, L, _) in
                       (rec(p, lv, won) for (p, lv, _) in results.values())})
    urates = np.array([w / (w + l) for w, l in distinct])
    rep += ["", f"  {len(rates)} grid cells collapse to {len(distinct)} distinct "
                f"records: expo and K never flip a pick at ~113 games/club,",
            "  so the window is the only live knob.",
            f"  spread over those {len(urates)}: min {urates.min():.3f} "
            f"max {urates.max():.3f} mean {urates.mean():.3f} sd {urates.std():.3f}"]

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

    # --------------------------------------------------- pythag vs market --
    rep += ["", "PYTHAG vs ODDS-IMPLIED (does it add anything to the close?)"]
    for w in windows:
        _, _, pr = run_config(rows, logs, w, expos[0], shrinks[0])
        vs_market(rows, pr, won, rep,
                  f"window={'season' if w <= 0 else w}")

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
