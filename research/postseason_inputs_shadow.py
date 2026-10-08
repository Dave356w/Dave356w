#!/usr/bin/env python3
"""Would folding postseason PAs into the season rates change any lean?

THE QUESTION. The live build reads Savant's custom leaderboard, which serves
regular-season PAs only (checked 2026-10-08: all 101 hitters with a Division
Series PA showed exactly their regular-season PA count). So from the last
regular-season day on, every postseason slate is built on frozen season
rates. This script measures what keeping them live would have done.

THE COMPARISON, paired. Each postseason slate is replayed twice through the
production `build_site.fetch_all` -> `build_xwoba_matchup` ->
`apply_pitching_plans` -> `grade_leans.rows_from_dump` path:

  frozen     the custom leaderboard exactly as Savant serves it
  augmented  the same board with every postseason PA from games dated
             strictly BEFORE the slate folded into `pa`, `xwoba` and `woba`

Everything else (lineups, rosters, velocity, workload, splits) comes from
one live fetch shared in shape by both arms, so a flip is caused by the rate
change alone. The regular-season board has not moved since the season ended,
so the frozen arm should also reproduce the native pregame lean; how often it
does is printed as a replay check. Mismatches there come from inputs the
replay cannot freeze (actual vs projected lineups, current rosters).

Postseason rates are built from pitch-level Statcast (hfGT=F|D|L|W): xwOBA
numerator is `estimated_woba_using_speedangle` on batted balls and
`woba_value` otherwise, over `woba_denom`; wOBA is `woba_value` over
`woba_denom`. They are combined with the season rate by PA weight. Only the
two model rates and `pa` are augmented; K%, BB% and the other display columns
stay regular season. Players absent from the season board are not added.

RESULT, 2026-10-08 (data commit e8f7fb0; 23 postseason games 09-29..10-07,
19 with prior postseason PAs; 1,375 postseason PAs): 0 of 23 leans flipped.
|xw_net| moved by median .0009, max .0026, against a median |xw_net| of
.0223; the closest call (09-30 CWS@HOU, .0034) moved .0015 and held. The
frozen replay matched 22 of 23 native leans, and its own drift from native
(median .0023) was larger than the augmentation effect. Not a forward test:
it says freezing the rates cost no published decision this October.

Research only: nothing here reaches a lean, a delta, a grade or data/.
Writes go to --cache (Savant/velocity caches) and --out (the paired CSV).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POSTSEASON_TYPES = "F%7CD%7CL%7CW%7C"
_STATCAST_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB="
    "&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT={gt}&hfSea={season}%7C"
    "&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws="
    "&batter_stands=&hfSA=&game_date_gt={day}&game_date_lt={day}&team="
    "&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
    "&min_results=0&group_by=name&sort_col=pitches"
    "&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details&")


def fetch_postseason_pas(bs, days, season):
    """One row per postseason PA (game_date, batter, pitcher, xw_num, w_num, den)."""
    frames = []
    for day in days:
        r = bs.session.get(_STATCAST_URL.format(gt=POSTSEASON_TYPES,
                                                season=season, day=day),
                           timeout=120)
        r.raise_for_status()
        if not r.text.strip():
            continue
        d = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if d.empty or "events" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["game_date", "batter", "pitcher",
                                     "xw_num", "w_num", "den"])
    d = pd.concat(frames, ignore_index=True)
    d = d[d["events"].notna() & d["game_type"].astype(str).ne("R")].copy()
    den = pd.to_numeric(d["woba_denom"], errors="coerce").fillna(0.0)
    wv = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0.0)
    est = pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce")
    d["den"] = den
    d["w_num"] = wv
    d["xw_num"] = np.where(est.notna(), est, wv)
    d["game_date"] = d["game_date"].astype(str).str[:10]
    return d[["game_date", "game_pk", "batter", "pitcher",
              "xw_num", "w_num", "den"]]


def augment_board(cust, pas, player_type, before):
    """Season custom board with postseason PAs dated < `before` folded in."""
    key = "batter" if player_type == "batter" else "pitcher"
    p = pas[pas["game_date"] < before]
    if p.empty:
        return cust.copy(), 0
    g = p.groupby(key).agg(n=("den", "size"), den=("den", "sum"),
                          xw=("xw_num", "sum"), w=("w_num", "sum"))
    out = cust.copy()
    ids = pd.to_numeric(out["player_id"], errors="coerce")
    hit = ids.isin(g.index)
    pa = pd.to_numeric(out["pa"], errors="coerce")
    for col, num in (("xwoba", "xw"), ("woba", "w")):
        if col not in out.columns:
            continue
        season_rate = pd.to_numeric(out[col], errors="coerce")
        n_add = ids.map(g["n"]).fillna(0.0)
        den = ids.map(g["den"]).replace(0.0, np.nan)
        post_rate = ids.map(g[num]) / den
        mixed = (season_rate * pa + post_rate * n_add) / (pa + n_add)
        out[col] = np.where(hit & post_rate.notna() & season_rate.notna(),
                            mixed, season_rate)
    out["pa"] = np.where(hit, pa + ids.map(g["n"]).fillna(0.0), pa)
    return out, int(hit.sum())


def replay(bs, gl, day, pas, augmented):
    """Leans for one slate; `augmented` swaps in the postseason-folded board."""
    bs.SLATE_DATE = day
    bs._team_pitcher_role_cache.clear()
    real_cached = bs.cached_csv
    touched = {}

    def cached_csv(url, cache_name, tries=4):
        df = real_cached(url, cache_name, tries)
        if augmented and cache_name.startswith(bs.STATCAST_CACHE_NS):
            ptype = cache_name.rsplit("_", 1)[-1]
            df, touched[ptype] = augment_board(df, pas, ptype, day)
        return df

    bs.cached_csv = cached_csv
    try:
        data = bs.fetch_all(day, provider=bs.LiveDataProvider(day),
                            write_audit=False)
    finally:
        bs.cached_csv = real_cached
    if data.get("empty"):
        return pd.DataFrame(), touched
    m, _, _ = bs.build_xwoba_matchup(data["pitchers_df"],
                                     data["league_baseline"])
    m = bs.apply_pitching_plans(m, data.get("pitching_plans"),
                                data["league_baseline"])
    if m is None or m.empty:
        return pd.DataFrame(), touched
    rows = pd.DataFrame(gl.rows_from_dump(m, None))
    keep = ["game_pk", "game_date", "away", "home", "xw_net", "xw_lean"]
    return rows[keep], touched


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
    season = int(days[0][:4])
    pas = fetch_postseason_pas(bs, days, season)
    print(f"postseason PAs fetched: {len(pas)} over "
          f"{pas['game_pk'].nunique() if len(pas) else 0} games")

    paired = []
    for day in days:
        frozen, _ = replay(bs, gl, day, pas, augmented=False)
        aug, touched = replay(bs, gl, day, pas, augmented=True)
        if frozen.empty or aug.empty:
            print(f"{day}: no replayable matchups")
            continue
        j = frozen.merge(aug[["game_pk", "xw_net", "xw_lean"]], on="game_pk",
                         suffixes=("_frozen", "_aug"))
        j["boards_touched"] = str(touched)
        paired.append(j)
        print(f"{day}: {len(j)} games, players augmented {touched}")

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
