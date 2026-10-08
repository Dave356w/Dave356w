#!/usr/bin/env python3
"""Did folding postseason PAs into the season rates change any lean?

BACKGROUND. Savant's custom leaderboard serves regular-season PAs only
(checked 2026-10-08: all 101 hitters with a Division Series PA showed exactly
their regular-season PA count). Since 2026-10-08 the production build folds
prior postseason PAs back in on postseason slates (postseason_rates.py, wired
in build_site.load_stat_lookups); before that, postseason slates were built
on rates frozen at the last regular-season game.

THE COMPARISON, paired. Each postseason slate is replayed twice through the
production `build_site.fetch_all` -> `build_xwoba_matchup` ->
`apply_pitching_plans` -> `grade_leans.rows_from_dump` path:

  frozen     production with `load_postseason_pas` switched off: the board
             exactly as Savant serves it
  augmented  production as shipped: postseason PAs from games dated strictly
             BEFORE the slate folded into `pa`, `xwoba` and `woba`

Everything else (lineups, rosters, velocity, workload, splits) is fetched the
same way for both arms, so a flip is caused by the rate change alone. The
regular-season board has not moved since the season ended, so the frozen arm
should also reproduce the native pregame lean of slates built before the
change; how often it does is printed as a replay check. Mismatches there come
from inputs the replay cannot freeze (actual vs projected lineups, current
rosters).

RESULT, 2026-10-08 (data commit e8f7fb0; 23 postseason games 09-29..10-07,
19 with prior postseason PAs; 1,375 postseason PAs): 0 of 23 leans flipped.
|xw_net| moved by median .0009, max .0026, against a median |xw_net| of
.0223; the closest call (09-30 CWS@HOU, .0034) moved .0015 and held. The
frozen replay matched 22 of 23 native leans, and its own drift from native
(median .0023) was larger than the augmentation effect. Not a forward test.

Research only: nothing here reaches a lean, a delta, a grade or data/.
Writes go to --cache (Savant/velocity caches) and --out (the paired CSV).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frozen_provider(bs, day):
    """LiveDataProvider with the postseason fold switched off."""
    class Frozen(bs.LiveDataProvider):
        def load_postseason_pas(self):
            return None
    return Frozen(day)


def replay(bs, gl, day, augmented):
    """Leans for one slate through the production path."""
    bs.SLATE_DATE = day
    bs._team_pitcher_role_cache.clear()
    provider = (bs.LiveDataProvider(day) if augmented
                else frozen_provider(bs, day))
    data = bs.fetch_all(day, provider=provider, write_audit=False)
    if data.get("empty"):
        return pd.DataFrame()
    m, _, _ = bs.build_xwoba_matchup(data["pitchers_df"],
                                     data["league_baseline"])
    m = bs.apply_pitching_plans(m, data.get("pitching_plans"),
                                data["league_baseline"])
    if m is None or m.empty:
        return pd.DataFrame()
    rows = pd.DataFrame(gl.rows_from_dump(m, None))
    return rows[["game_pk", "game_date", "away", "home", "xw_net", "xw_lean"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=str(ROOT / "data" /
                                            "mlb_postseason_ledger.csv"))
    ap.add_argument("--cache", required=True,
                    help="scratch dir for Savant/velocity caches")
    ap.add_argument("--out", required=True, help="paired per-game CSV")
    a = ap.parse_args(argv)

    os.environ["CACHE_DIR"] = a.cache
    import build_site as bs
    import grade_leans as gl
    bs.CACHE_DIR = a.cache

    led = pd.read_csv(a.ledger)
    days = sorted(led["game_date"].astype(str).str[:10].unique())

    paired = []
    for day in days:
        frozen = replay(bs, gl, day, augmented=False)
        aug = replay(bs, gl, day, augmented=True)
        if frozen.empty or aug.empty:
            print(f"{day}: no replayable matchups")
            continue
        j = frozen.merge(aug[["game_pk", "xw_net", "xw_lean"]], on="game_pk",
                         suffixes=("_frozen", "_aug"))
        paired.append(j)
        print(f"{day}: {len(j)} games")

    res = pd.concat(paired, ignore_index=True) if paired else pd.DataFrame()
    if res.empty:
        print("nothing replayed")
        return 1
    native = led[["game_pk", "xw_net", "xw_lean"]].rename(
        columns={"xw_net": "xw_net_native", "xw_lean": "xw_lean_native"})
    res = res.merge(native, on="game_pk", how="left")
    res["shift"] = res["xw_net_aug"] - res["xw_net_frozen"]
    res["flip"] = res["xw_lean_aug"].ne(res["xw_lean_frozen"])
    res["replay_matches_native"] = res["xw_lean_frozen"].eq(res["xw_lean_native"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    with_post = res[res["game_date"] > days[0]]
    print("\n== paired result (frozen vs augmented, same replay) ==")
    print(f"games replayed: {len(res)}  (with prior postseason PAs: "
          f"{len(with_post)})")
    print(f"replay check, frozen lean == native pregame lean: "
          f"{int(res['replay_matches_native'].sum())}/"
          f"{int(res['xw_lean_native'].notna().sum())}")
    print(f"lean flips: {int(res['flip'].sum())}/{len(res)}")
    s = res["shift"].abs()
    print(f"|xw_net shift|: median {s.median():.4f}  max {s.max():.4f}  "
          f"(|xw_net frozen| median {res['xw_net_frozen'].abs().median():.4f})")
    if res["flip"].any():
        print(res.loc[res["flip"], ["game_date", "away", "home",
                                    "xw_net_frozen", "xw_net_aug",
                                    "xw_lean_frozen", "xw_lean_aug"]]
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
