#!/usr/bin/env python3
"""Would postseason games in the pitching inputs change any lean?

Companion to postseason_inputs_shadow.py, which covered the Savant rate
boards (now folded in production). Three pitching inputs still read regular-
season games only:

  velo     starter velocity trend: Savant search hfGT=R, and
           starter_velocity.per_start keeps game_type R
  recent   recent-start ERA / role profile and the expected-IP inputs:
           StatsAPI gameLog gameType=R
  roles    bullpen workload roles (starter vs relief pool, IP and BF per
           appearance): StatsAPI team season stats gameType=R

Each postseason slate is replayed through the production path once as
shipped (baseline) and once per arm with that input widened to postseason
games dated strictly BEFORE the slate, plus an `all` arm with all three.
Every arm runs in its own cache directory (velocity is cached per pitcher per
day, keyed without game type). Widening is done by overriding the provider's
loaders for the duration of the call; no production code is changed.

The team season endpoint ignores a gameType list (checked 2026-10-08: CHC
gamesPlayed was 680 under "R" and "R,F,D,L,W"), so `roles` fetches each
postseason type separately by date range ending the day before the slate and
adds its counting stats (appearances, starts, outs, BF) to the season line.

Research only: nothing here reaches a lean, a delta, a grade or data/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POST = ("F", "D", "L", "W")
ARMS = ("baseline", "velo", "recent", "roles", "all")


def _day_before(day):
    return (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _postseason_roles(bs, team_id, day):
    """{pid: {apps, starts, outs, bf}} over postseason games before `day`."""
    out = {}
    for gt in POST:
        data = bs._get_json("https://statsapi.mlb.com/api/v1/stats", {
            "stats": "byDateRange", "group": "pitching", "season": bs.SEASON,
            "sportIds": bs.SPORT_ID, "teamId": int(team_id), "gameType": gt,
            "playerPool": "ALL", "limit": 1000,
            "startDate": f"{bs.SEASON}-01-01", "endDate": _day_before(day)})
        for blk in data.get("stats", []):
            for sk in blk.get("splits", []):
                pid = (sk.get("player") or sk.get("person") or {}).get("id")
                if pid is None:
                    continue
                st = sk.get("stat", {}) or {}
                o = out.setdefault(int(pid), dict(apps=0, starts=0, outs=0, bf=0.0))
                o["apps"] += int(st.get("gamesPitched") or st.get("gamesPlayed") or 0)
                o["starts"] += int(st.get("gamesStarted") or 0)
                o["outs"] += bs._innings_to_outs(st.get("inningsPitched"))
                o["bf"] += float(st.get("battersFaced") or 0)
    return out


def make_provider(bs, day, arm):
    widen = {"velo", "recent", "roles"} if arm == "all" else {arm}
    role_memo = {}

    class Widened(bs.LiveDataProvider):

        def load_starter_velocity(self, ids):
            if "velo" not in widen:
                return super().load_starter_velocity(ids)
            url = bs._SAVANT_PITCHER_URL.replace(
                "hfGT=R%7C", "hfGT=R%7C" + "".join(f"{t}%7C" for t in POST))
            orig = bs._sv.per_start

            def per_start(pitches):
                if pitches is not None and "game_type" in getattr(pitches, "columns", ()):
                    gt = pitches["game_type"].astype(str)
                    pitches = pitches.assign(
                        game_type=np.where(gt.isin(POST), "R", gt))
                return orig(pitches)

            with mock.patch.object(bs, "_SAVANT_PITCHER_URL", url), \
                    mock.patch.object(bs._sv, "per_start", per_start):
                return super().load_starter_velocity(ids)

        def load_recent_start_era(self, ids):
            if "recent" not in widen:
                return super().load_recent_start_era(ids)
            orig = bs._get_json

            def get_json(url, params=None, **kw):
                if ("/people/" in url and params
                        and params.get("stats") == "gameLog"
                        and params.get("gameType") == "R"):
                    params = {**params, "gameType": "R," + ",".join(POST)}
                return orig(url, params, **kw)

            with mock.patch.object(bs, "_get_json", get_json):
                return super().load_recent_start_era(ids)

        def load_team_pitcher_roles(self, team_id):
            base = super().load_team_pitcher_roles(team_id)
            if "roles" not in widen:
                return base
            if team_id in role_memo:
                return role_memo[team_id]
            post = _postseason_roles(bs, team_id, day)
            merged = {}
            for pid in set(base) | set(post):
                b = base.get(pid) or {}
                p = post.get(pid) or dict(apps=0, starts=0, outs=0, bf=0.0)
                apps0 = int(b.get("appearances") or 0)
                outs0 = (b.get("avg_ip_per_appearance") or 0) * 3.0 * apps0
                if pd.isna(outs0):
                    outs0 = 0.0
                apps = apps0 + p["apps"]
                starts = int(b.get("starts") or 0) + p["starts"]
                outs = outs0 + p["outs"]
                merged[pid] = {
                    "appearances": apps, "starts": starts,
                    "start_share": starts / apps if apps > 0 else np.nan,
                    "avg_ip_per_appearance": outs / 3.0 / apps if apps > 0 else np.nan,
                    "batters_faced": float(b.get("batters_faced") or 0) + p["bf"],
                }
            role_memo[team_id] = merged
            return merged

    return Widened(day)


def replay(bs, gl, day, arm, cache_root):
    bs.SLATE_DATE = day
    bs.CACHE_DIR = os.path.join(cache_root, arm)
    bs._team_pitcher_role_cache.clear()
    data = bs.fetch_all(day, provider=make_provider(bs, day, arm),
                        write_audit=False)
    if data.get("empty"):
        return pd.DataFrame()
    m, _, _ = bs.build_xwoba_matchup(data["pitchers_df"], data["league_baseline"])
    m = bs.apply_pitching_plans(m, data.get("pitching_plans"), data["league_baseline"])
    if m is None or m.empty:
        return pd.DataFrame()
    rows = pd.DataFrame(gl.rows_from_dump(m, None))
    return rows[["game_pk", "game_date", "away", "home", "xw_net", "xw_lean"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=str(ROOT / "data" / "mlb_postseason_ledger.csv"))
    ap.add_argument("--cache", required=True, help="scratch dir; one subdir per arm")
    ap.add_argument("--out", required=True, help="per-game CSV, one column pair per arm")
    a = ap.parse_args(argv)

    os.environ["CACHE_DIR"] = a.cache
    import build_site as bs
    import grade_leans as gl

    led = pd.read_csv(a.ledger)
    days = sorted(led["game_date"].astype(str).str[:10].unique())
    frames = []
    for day in days:
        per_arm = {}
        for arm in ARMS:
            r = replay(bs, gl, day, arm, a.cache)
            if not r.empty:
                per_arm[arm] = r.set_index("game_pk")
        if "baseline" not in per_arm:
            print(f"{day}: no replayable matchups")
            continue
        f = per_arm["baseline"][["game_date", "away", "home"]].copy()
        for arm, r in per_arm.items():
            f[f"xw_net_{arm}"] = r["xw_net"]
            f[f"xw_lean_{arm}"] = r["xw_lean"]
        frames.append(f.reset_index())
        print(f"{day}: {len(f)} games, arms {sorted(per_arm)}")

    res = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    base_abs = res["xw_net_baseline"].abs()
    print(f"\n== {len(res)} games; median |xw_net| baseline {base_abs.median():.4f} ==")
    for arm in ARMS[1:]:
        col = f"xw_net_{arm}"
        if col not in res:
            continue
        ok = res[col].notna()
        shift = (res[col] - res["xw_net_baseline"]).abs()[ok]
        flips = res.loc[ok, f"xw_lean_{arm}"].ne(res.loc[ok, "xw_lean_baseline"])
        print(f"{arm:7s} n={int(ok.sum()):2d}  flips={int(flips.sum())}  "
              f"|shift| median {shift.median():.4f} max {shift.max():.4f}")
        if flips.any():
            print(res.loc[ok][flips.values][["game_date", "away", "home",
                                             "xw_net_baseline", col]]
                  .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
