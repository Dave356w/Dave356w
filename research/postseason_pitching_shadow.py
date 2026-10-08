#!/usr/bin/env python3
"""Did postseason games in the pitching inputs change any lean?

Companion to postseason_inputs_shadow.py (the Savant rate boards). Since
2026-10-08 a postseason slate's pitching loaders also read prior postseason
games (build_site.season_game_types):

  velo     starter velocity trend (Savant search + starter_velocity.per_start)
  recent   recent-start ERA / role profile and the expected-IP inputs
           (StatsAPI gameLog; accepts a gameType list)
  roles    bullpen workload roles (StatsAPI team stats; ignores a gameType
           list -- CHC gamesPlayed was 680 under "R" and "R,F,D,L,W" on
           2026-10-08 -- so each postseason type is its own dated call)

Each postseason slate is replayed through the production path with each
loader pinned to regular season (`baseline`, the rate fold still on), one
loader widened at a time, all three widened (`all`), and the provider
untouched (`production`), which must equal `all`. Postseason games are read
only when dated strictly BEFORE the slate. Every arm has its own cache
directory, since velocity is cached per pitcher per day without game type.

RESULT, 2026-10-08 (data commit e8f7fb0; 23 postseason games 09-29..10-07):
0 flips in every arm. Games moved / max |xw_net| shift: velo 7 / .0044,
recent 7 / .0014, roles 19 / .0001, all 19 / .0052, against a median
|xw_net| of .0217 and a closest call of .0018. velo and recent move only
once a probable has a postseason start (10-03 on). Not a forward test.

Research only: nothing here reaches a lean, a delta, a grade or data/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = ("baseline", "velo", "recent", "roles", "all", "production")


def make_provider(bs, day, arm):
    """Provider whose pitching loaders read the season only where `arm` says.

    `production` is LiveDataProvider untouched (fetch_all picks the slate's
    game types); every other arm pins each loader to regular season or to
    the full season through the same production loaders.
    """
    if arm == "production":
        return bs.LiveDataProvider(day)
    widen = {"velo", "recent", "roles"} if arm == "all" else {arm}
    full = bs.postseason_rates.SEASON_GAME_TYPES
    reg = bs.postseason_rates.REGULAR_GAME_TYPES

    def types(which):
        return full if which in widen else reg

    class Pinned(bs.LiveDataProvider):
        def load_starter_velocity(self, ids):
            return bs.load_starter_velocity(ids, before_date=self.slate_date,
                                            game_types=types("velo"))

        def load_recent_start_era(self, ids):
            return bs.load_recent_start_era(ids, before_date=self.slate_date,
                                            game_types=types("recent"))

        def load_team_pitcher_roles(self, team_id):
            return bs.load_team_pitcher_roles(team_id, game_types=types("roles"))

    return Pinned(day)


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
