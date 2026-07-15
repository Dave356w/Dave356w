#!/usr/bin/env python3
# ============================================================
# MLB Probable-Pitcher vs Opponent-Lineup matchup -- STATIC SITE GENERATOR
#
# Render-free build of the Colab notebook
# (Shrunk_mlb_matchup_render_consolidated.ipynb), turned into a one-shot
# static-site generator: it pulls the slate via keyless APIs, builds the
# xwOBA + platoon-OPS matchup tables, and writes a fully self-contained
# index.html (inline CSS, no external assets, dark-mode via
# prefers-color-scheme) for GitHub Pages.
#
# Data sources (all keyless):
#   - MLB StatsAPI          : slate, probables, rosters, bio, vL/vR splits
#   - Savant gf?game_pk=     : posted lineups (JSON)
#   - Savant CSV leaderboards (cached once/day): custom + batted-ball
#
# Pipeline mirrors notebook cells 1 (fetch) -> 4 (xwOBA matchup) ->
# 5 (platoon split) -> 6 (consolidated card render). The notebook's
# clean/validate cells (2, 3) are skipped: they produced *_clean / *_strict
# frames that the matchup/render cells never consume (the API build_tables
# output is already clean).
#
# Behaviour for unattended runs:
#   - SLATE_DATE is computed in America/New_York with a ~3am ET rollover
#     (override with the SLATE_DATE env var, YYYY-MM-DD).
#   - A top-level guard means a failed/empty core fetch raises a non-zero
#     exit WITHOUT writing index.html, so CI skips the deploy and the last
#     good page stays live. A legitimate empty slate (off-day) writes a
#     friendly "no games" page instead (a successful build).
#   - index.html carries a "built HH:MM ET" line beside the slate date.
# ============================================================

import io
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ET = ZoneInfo("America/New_York")
ROLLOVER_HOUR = 3          # hold "today" until ~3am ET so night games don't roll early
SPORT_ID = 1
LINEUP_SIZE = 9
REQUEST_DELAY = 0.25
CACHE_DIR = os.environ.get("CACHE_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "public")
DATA_DIR = os.environ.get("DATA_DIR", "data")            # grading ledger home
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
MODEL_TAG = os.environ.get("MODEL_TAG", "xw+plat_consol_v1")  # keep in sync with grade_leans.py

STATCAST_SELECTIONS = ["pa", "k_percent", "bb_percent", "xwoba", "xba", "xslg",
                       "exit_velocity_avg", "launch_angle_avg", "hard_hit_percent"]

# Batted-ball direction/tendency rates with true league-wide anchors.
BATTED_RATE_COLS_FOR_BASELINE = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]

# Use all Savant-listed hitters for platoon priors instead of today's loaded lineups.
# Costs ~10-15 batched StatsAPI calls (~+5s) but removes slate-dependent shrinkage baselines.
FULL_LEAGUE_PLATOON_BASELINES = True
MIN_LEAGUE_BASELINE_PA = 1


def slate_date_now():
    """Slate date in ET with a ~3am rollover. Env override wins."""
    override = os.environ.get("SLATE_DATE", "").strip()
    if override:
        return override
    now_et = datetime.now(ET)
    day = now_et.date() if now_et.hour >= ROLLOVER_HOUR else (now_et - timedelta(days=1)).date()
    return day.isoformat()


SLATE_DATE = slate_date_now()
SEASON = int(SLATE_DATE[:4])

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/csv,*/*",
})


def log(*a):
    print(*a, flush=True)


# ============================================================
# CELL 1 -- FETCH (render-free / no browser)
# ============================================================
def _get_json(url, params=None, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.6 * (2 ** k))


def cached_csv(url, cache_name, tries=4):
    """Fetch a Savant CSV leaderboard, caching the raw text once per ET day."""
    path = os.path.join(CACHE_DIR, f"savant_cache_{cache_name}_{SLATE_DATE}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            text = r.text
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
            except Exception:
                pass
            return pd.read_csv(io.StringIO(text))
        except Exception as e:  # noqa: BLE001
            last = e
            if k == tries - 1:
                break
            time.sleep(0.8 * (2 ** k))
    raise last


def get_slate(slate_date, sport_id=1):
    data = _get_json("https://statsapi.mlb.com/api/v1/schedule",
                     {"sportId": sport_id, "date": slate_date,
                      "hydrate": "probablePitcher,team,venue,linescore"})
    rows = []
    for db in data.get("dates", []):
        od = db.get("date", slate_date)
        for g in db.get("games", []):
            a, h = g["teams"]["away"], g["teams"]["home"]
            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": od,
                "game_datetime_utc": g.get("gameDate"),
                "matchup": f'{a["team"]["name"]} @ {h["team"]["name"]}',
                "away_team": a["team"]["name"], "home_team": h["team"]["name"],
                "away_team_id": a["team"]["id"], "home_team_id": h["team"]["id"],
                "away_abbrev": a["team"].get("abbreviation"),
                "home_abbrev": h["team"].get("abbreviation"),
                "away_probable_pitcher": a.get("probablePitcher", {}).get("fullName"),
                "home_probable_pitcher": h.get("probablePitcher", {}).get("fullName"),
                "away_probable_pitcher_id": a.get("probablePitcher", {}).get("id"),
                "home_probable_pitcher_id": h.get("probablePitcher", {}).get("id"),
                "status": g.get("status", {}).get("detailedState"),
                "venue": g.get("venue", {}).get("name"),
                "savant_preview_url": f'https://baseballsavant.mlb.com/preview?game_pk={g.get("gamePk")}&game_date={od}',
            })
    return pd.DataFrame(rows)


def load_stat_lookups(player_type):
    """player_type in {'batter','pitcher'} -> (stat dict, bbprofile dict, custom df)."""
    sel = ",".join(STATCAST_SELECTIONS)
    cust = cached_csv(
        f"https://baseballsavant.mlb.com/leaderboard/custom?year={SEASON}"
        f"&type={player_type}&min=1&selections={sel}&csv=true",
        f"custom_{player_type}")
    bb = cached_csv(
        f"https://baseballsavant.mlb.com/leaderboard/batted-ball?type={player_type}"
        f"&year={SEASON}&min=1&csv=true",
        f"battedball_{player_type}")

    REN_STAT = {"xwoba": "xwOBA", "xba": "xBA", "xslg": "xSLG", "exit_velocity_avg": "EV",
                "launch_angle_avg": "LA°", "hard_hit_percent": "Hard Hit%",
                "k_percent": "K%", "bb_percent": "BB%", "pa": "PA"}
    stat = {}
    for _, r in cust.iterrows():
        pid = int(r["player_id"])
        stat[pid] = {REN_STAT[k]: r.get(k) for k in REN_STAT if k in cust.columns}

    BB_REN = {"gb_rate": "GB%", "fb_rate": "FB%", "ld_rate": "LD%", "pu_rate": "PU%",
              "pull_rate": "Pull%", "straight_rate": "Straight%", "oppo_rate": "Oppo%"}
    bbprofile = {}
    for _, r in bb.iterrows():
        pid = int(r["id"])
        prof = {v: (r[k] * 100 if pd.notna(r[k]) else np.nan) for k, v in BB_REN.items() if k in bb.columns}
        bbprofile[pid] = prof
        stat.setdefault(pid, {})
        stat[pid]["BBE"] = r.get("bbe")
    return stat, bbprofile, cust


def load_people(ids):
    ids = [int(i) for i in ids if pd.notna(i)]
    info = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        data = _get_json("https://statsapi.mlb.com/api/v1/people",
                         {"personIds": ",".join(map(str, chunk))})
        for p in data.get("people", []):
            info[p["id"]] = {
                "name": p.get("fullName"),
                "pos": (p.get("primaryPosition", {}) or {}).get("abbreviation"),
                "bats": (p.get("batSide", {}) or {}).get("code"),
                "throws": (p.get("pitchHand", {}) or {}).get("code"),
            }
    return info


def _parse_rate(x):
    try:
        return float(str(x))
    except Exception:
        return np.nan


def load_splits(ids, group):
    """vL/vR splits + season overall via batched hydrate. group in {'hitting','pitching'}."""
    ids = [int(i) for i in ids if pd.notna(i)]
    pa_field = "plateAppearances" if group == "hitting" else "battersFaced"
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        hyd = f"stats(group=[{group}],type=[statSplits,season],sitCodes=[vl,vr],season={SEASON})"
        data = _get_json("https://statsapi.mlb.com/api/v1/people",
                         {"personIds": ",".join(map(str, chunk)), "hydrate": hyd})
        for p in data.get("people", []):
            rec = {"L": {}, "R": {}, "overall": {}}
            for blk in p.get("stats", []):
                tname = (blk.get("type", {}) or {}).get("displayName")
                for sk in blk.get("splits", []):
                    code = sk.get("split", {}).get("code")
                    st = sk.get("stat", {})
                    val = {"ops": _parse_rate(st.get("ops")),
                           "obp": _parse_rate(st.get("obp")),
                           "slg": _parse_rate(st.get("slg")),
                           "pa": int(st.get(pa_field) or st.get("plateAppearances") or 0),
                           "k": st.get("strikeOuts"), "bb": st.get("baseOnBalls")}
                    if tname == "statSplits" and code in ("vl", "vr"):
                        rec["L" if code == "vl" else "R"] = val
                    elif tname == "season" and code is None:
                        rec["overall"] = {"ops": val["ops"], "pa": val["pa"]}
            out[p["id"]] = rec
        time.sleep(REQUEST_DELAY)
    return out


_gf_cache = {}


def gf_lineups(game_pk):
    """Raw Savant posted lineup ids. Cached per game_pk."""
    if game_pk in _gf_cache:
        return _gf_cache[game_pk]
    try:
        gf = _get_json(f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}", tries=2)
        out = ([int(x) for x in gf.get("away_lineup", []) or []],
               [int(x) for x in gf.get("home_lineup", []) or []])
    except Exception:
        out = ([], [])
    _gf_cache[game_pk] = out
    return out


_roster_cache = {}


def roster_lineup(team_id, batter_stat):
    """Projected lineup pool = active-roster position players, ranked by season PA."""
    if team_id in _roster_cache:
        ids = _roster_cache[team_id]
    else:
        data = _get_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                         {"rosterType": "active"})
        ids = [r["person"]["id"] for r in data.get("roster", [])
               if (r.get("position", {}) or {}).get("abbreviation") != "P"]
        _roster_cache[team_id] = ids
    return sorted([int(i) for i in ids if int(i) in batter_stat],
                  key=lambda i: (batter_stat[i].get("PA") or 0), reverse=True)


def fill_lineup_from_roster(team_id, posted_ids, batter_stat, target_size=LINEUP_SIZE):
    """Keep valid posted hitters in order; fill only missing slots by roster PA."""
    resolved, seen = [], set()
    for pid in posted_ids or []:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid in batter_stat and pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            return resolved[:target_size]
    for pid in roster_lineup(team_id, batter_stat):
        if pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            break
    return resolved[:target_size]


def resolve_lineup(game_pk, side, team_id, batter_stat, return_meta=False):
    """posted (>=9 valid) / partial_filled (1-8 kept, rest filled) / projected (0)."""
    away_ids, home_ids = gf_lineups(game_pk)
    raw = away_ids if side == "away" else home_ids
    valid_posted, seen = [], set()
    for pid in raw:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid in batter_stat and pid not in seen:
            valid_posted.append(pid); seen.add(pid)
    resolved = fill_lineup_from_roster(team_id, valid_posted, batter_stat, LINEUP_SIZE)
    posted_count = min(len(valid_posted), LINEUP_SIZE)
    filled_count = max(0, len(resolved) - posted_count)
    status = ("posted" if posted_count >= LINEUP_SIZE
              else "partial_filled" if posted_count > 0 else "projected")
    meta = {"status": status, "projected": status != "posted",
            "posted_count": int(posted_count), "filled_count": int(filled_count),
            "resolved_count": int(len(resolved))}
    return (resolved, meta) if return_meta else (resolved, meta["projected"])


STAT_COLS = ["BBE", "LA°", "EV", "Hard Hit%", "xwOBA", "xBA", "xSLG", "K%", "BB%"]
BB_COLS = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]


def _meta(row):
    return {k: row[k] for k in ["game_pk", "game_date", "matchup", "away_team", "home_team",
                                "away_probable_pitcher", "home_probable_pitcher", "savant_preview_url"]}


def build_tables(slate, lineups, batter_stat, pitcher_stat, batter_bb, pitcher_bb, people):
    pit_rows, bb_rows = [], []

    for _, g in slate.iterrows():
        meta = _meta(g)
        asp, hsp = g["away_probable_pitcher_id"], g["home_probable_pitcher_id"]
        away_lu, home_lu = lineups[g["game_pk"]]

        def pitcher_row(pid, tidx, table):
            if pd.isna(pid):
                return
            pid = int(pid)
            bio = people.get(pid, {})
            nm = bio.get("name") or f"pitcher_{pid}"
            src = (pitcher_stat if table == "stat" else pitcher_bb).get(pid, {})
            base = {**meta, "table_index": tidx, "Name": nm}
            if table == "stat":
                pit_rows.append({**base, "table_type": "pitchers", "Pos.": "P",
                                 **{c: src.get(c) for c in STAT_COLS},
                                 "player_id": pid, "bats": bio.get("bats"),
                                 "throws": bio.get("throws")})
            else:
                bbe = (pitcher_stat.get(pid, {}) or {}).get("BBE")
                bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": "P",
                                **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                "player_id": pid, "bats": bio.get("bats"),
                                "throws": bio.get("throws")})

        def hitter_rows(lu, tidx, table):
            for pid in lu:
                pid = int(pid)
                bio = people.get(pid, {})
                nm = bio.get("name") or f"batter_{pid}"
                pos = bio.get("pos") or "DH"
                if pos == "P":
                    pos = "DH"
                src = (batter_stat if table == "stat" else batter_bb).get(pid, {})
                base = {**meta, "table_index": tidx, "Name": nm}
                if table == "stat":
                    pit_rows.append({**base, "table_type": "pitchers", "Pos.": pos,
                                     **{c: src.get(c) for c in STAT_COLS},
                                     "player_id": pid, "bats": bio.get("bats"),
                                     "throws": bio.get("throws")})
                else:
                    bbe = (batter_stat.get(pid, {}) or {}).get("BBE")
                    bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": pos,
                                    **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                    "player_id": pid, "bats": bio.get("bats"),
                                    "throws": bio.get("throws")})

        pitcher_row(asp, 1, "stat"); hitter_rows(home_lu, 1, "stat")
        pitcher_row(hsp, 2, "stat"); hitter_rows(away_lu, 2, "stat")
        pitcher_row(asp, 1, "bb"); hitter_rows(home_lu, 1, "bb")
        pitcher_row(hsp, 2, "bb"); hitter_rows(away_lu, 2, "bb")

    META = ["game_pk", "game_date", "matchup", "away_team", "home_team",
            "away_probable_pitcher", "home_probable_pitcher", "savant_preview_url",
            "table_type", "table_index", "Name"]
    pdf = pd.DataFrame(pit_rows)
    if not pdf.empty:
        pdf = pdf[META + ["Pos."] + STAT_COLS + ["player_id", "bats", "throws"]]
    bdf = pd.DataFrame(bb_rows)
    if not bdf.empty:
        bdf = bdf[META + ["Pos."] + BB_COLS + ["BBE", "player_id", "bats", "throws"]]
    return pdf, bdf


def compute_league_baseline(batter_cust):
    _LB_MAP = {"xwoba": "xwOBA", "xba": "xBA", "xslg": "xSLG", "k_percent": "K%", "bb_percent": "BB%",
               "exit_velocity_avg": "EV", "launch_angle_avg": "LA°", "hard_hit_percent": "Hard Hit%"}
    league_baseline = {}
    _w = pd.to_numeric(batter_cust.get("pa"), errors="coerce")
    for raw, disp in _LB_MAP.items():
        if raw in batter_cust.columns:
            v = pd.to_numeric(batter_cust[raw], errors="coerce")
            m = v.notna() & _w.notna() & (_w > 0)
            league_baseline[disp] = round(float(np.average(v[m], weights=_w[m])), 3) if m.any() else np.nan
    return league_baseline


def fetch_all(slate_date):
    """Run the full cell-1 fetch. Returns a dict of everything downstream needs."""
    log(f"Pulling slate for {slate_date} ...")
    slate_df = get_slate(slate_date, SPORT_ID)
    log(f"Games: {len(slate_df)}")
    if slate_df.empty:
        return {"slate_df": slate_df, "empty": True}

    log("Loading Savant leaderboards (cached once/day) ...")
    batter_stat, batter_bb, batter_cust = load_stat_lookups("batter")
    pitcher_stat, pitcher_bb, _ = load_stat_lookups("pitcher")
    log(f"  batters: {len(batter_stat)} | pitchers: {len(pitcher_stat)}")

    league_baseline = compute_league_baseline(batter_cust)
    # Full-population batted-ball anchors so GB/FB/LD/PU/Pull/Straight/Oppo matchup
    # edges are not NaN or slate-dependent. Rates are already stored as percentages.
    for c in BATTED_RATE_COLS_FOR_BASELINE:
        vals, wts = [], []
        for pid, prof in batter_bb.items():
            v = prof.get(c)
            w = (batter_stat.get(pid, {}) or {}).get("BBE")
            if pd.notna(v) and pd.notna(w) and float(w) > 0:
                vals.append(float(v)); wts.append(float(w))
        league_baseline[c] = round(float(np.average(vals, weights=wts)), 3) if vals else np.nan
    log("  league baselines:", {k: league_baseline.get(k) for k in
        ["xwOBA", "Hard Hit%", "K%", "EV", "GB%", "FB%", "Pull%"]})

    log("Resolving lineups (gf -> posted/partial-fill/projected) ...")
    lineups, proj_flags, lineup_ids, prob_ids = {}, [], set(), set()
    for _, g in slate_df.iterrows():
        al, ai = resolve_lineup(g["game_pk"], "away", g["away_team_id"], batter_stat, return_meta=True)
        hl, hi = resolve_lineup(g["game_pk"], "home", g["home_team_id"], batter_stat, return_meta=True)
        lineups[g["game_pk"]] = (al, hl)
        proj_flags.append({
            "game_pk": g["game_pk"],
            "away_lineup_projected": ai["projected"],
            "home_lineup_projected": hi["projected"],
            "away_lineup_status": ai["status"],
            "home_lineup_status": hi["status"],
            "away_posted_count": ai["posted_count"],
            "home_posted_count": hi["posted_count"],
            "away_filled_count": ai["filled_count"],
            "home_filled_count": hi["filled_count"],
            "away_resolved_count": ai["resolved_count"],
            "home_resolved_count": hi["resolved_count"],
        })
        lineup_ids.update(al); lineup_ids.update(hl)
        for c in ("away_probable_pitcher_id", "home_probable_pitcher_id"):
            if pd.notna(g[c]):
                prob_ids.add(int(g[c]))
        time.sleep(REQUEST_DELAY)
    lineup_projection_df = pd.DataFrame(proj_flags)
    os.makedirs(DATA_DIR, exist_ok=True)
    lineup_projection_df.to_csv(os.path.join(DATA_DIR, "lineup_resolution_audit.csv"), index=False)

    log("Loading player bio + vL/vR platoon splits ...")
    league_hitter_ids = {int(pid) for pid, st in batter_stat.items()
                         if pd.notna((st or {}).get("PA"))
                         and float((st or {}).get("PA") or 0) >= MIN_LEAGUE_BASELINE_PA}
    if FULL_LEAGUE_PLATOON_BASELINES:
        log(f"  league hitter split population: {len(league_hitter_ids)}")
        people_league_hitters = load_people(league_hitter_ids)
        player_splits_hit_league = load_splits(league_hitter_ids, "hitting")
    else:
        people_league_hitters, player_splits_hit_league = {}, {}
    people = dict(people_league_hitters)
    people.update(load_people(lineup_ids | prob_ids))
    player_splits_hit = load_splits(lineup_ids, "hitting")
    player_splits_pit = load_splits(prob_ids, "pitching")

    log("Assembling tables ...")
    pitchers_df, batted_ball_profile_df = build_tables(
        slate_df, lineups, batter_stat, pitcher_stat, batter_bb, pitcher_bb, people)

    side_status = pd.concat([lineup_projection_df["away_lineup_status"].rename("status"),
                             lineup_projection_df["home_lineup_status"].rename("status")],
                            ignore_index=True) if not lineup_projection_df.empty else pd.Series(dtype=str)
    n_proj = int(lineup_projection_df[["away_lineup_projected", "home_lineup_projected"]].sum().sum())
    log(f"lineup sources: {side_status.value_counts().to_dict()} of {2 * len(slate_df)} sides; "
        f"projected_or_partial={n_proj}")

    return {
        "empty": False,
        "slate_df": slate_df,
        "pitchers_df": pitchers_df,
        "batted_ball_profile_df": batted_ball_profile_df,
        "league_baseline": league_baseline,
        "people": people,
        "player_splits_hit": player_splits_hit,
        "player_splits_pit": player_splits_pit,
        "player_splits_hit_league": player_splits_hit_league,
        "people_league_hitters": people_league_hitters,
        "lineup_projection_df": lineup_projection_df,
    }


# ============================================================
# CELL 4 -- STATCAST (xwOBA) MATCHUP
# ============================================================
STATCAST_RATE_COLS = ["xwOBA", "xBA", "xSLG", "Hard Hit%", "EV", "LA°", "K%", "BB%"]
WEIGHT_COL = "BBE"
USE_WEIGHTED = True
ADD_STATS = {"EV", "LA°"}


def matchup_value(B, P, stat, L):
    if pd.isna(B) or pd.isna(P) or pd.isna(L):
        return np.nan
    if stat in ADD_STATS:
        return B + P - L
    return (B * P / L) if L else np.nan


def norm_name(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    s = s.lower().replace("\xa0", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def coerce_numeric(df, cols):
    df = df.copy()
    for c in set(cols) | {WEIGHT_COL}:
        if c in df.columns:
            s = (df[c].astype(str)
                 .str.replace("%", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.strip())
            s = s.where(~s.isin(["", "nan", "None", "--", "—"]), other=np.nan)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def wmean(vals, wts):
    vals = pd.to_numeric(pd.Series(vals).reset_index(drop=True), errors="coerce")
    if wts is None:
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
    wts = pd.to_numeric(pd.Series(wts).reset_index(drop=True), errors="coerce")
    m = vals.notna() & wts.notna() & (wts > 0)
    if not m.any():
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
    return float(np.average(vals[m], weights=wts[m]))


def segment_pitcher_blocks(df, rate_cols):
    df = coerce_numeric(df, rate_cols)
    has_pos = "Pos." in df.columns
    pos = df["Pos."].astype(str).str.upper().str.strip() if has_pos else None

    pitcher_rows, hitter_rows = [], []
    for _, g in df.groupby(["game_pk", "table_index"], sort=False):
        cur_p = cur_side = None
        away_key = norm_name(g["away_probable_pitcher"].iloc[0]) if "away_probable_pitcher" in g else ""
        home_key = norm_name(g["home_probable_pitcher"].iloc[0]) if "home_probable_pitcher" in g else ""
        for idx, r in g.iterrows():
            name = r.get("Name")
            nkey = norm_name(name)
            is_pitcher = (pos.loc[idx] == "P") if has_pos else (nkey in {away_key, home_key} and nkey != "")
            if is_pitcher:
                cur_p = name
                cur_side = "away" if nkey == away_key else "home" if nkey == home_key else None
                pitcher_rows.append({**r.to_dict(), "pitcher_side": cur_side})
                continue
            if cur_p is None:
                continue
            bat_side = {"away": "home", "home": "away"}.get(cur_side)
            hitter_rows.append({**r.to_dict(), "faced_pitcher": cur_p,
                                "pitcher_side": cur_side, "batting_side": bat_side})

    P = pd.DataFrame(pitcher_rows)
    H = pd.DataFrame(hitter_rows)
    if not P.empty:
        P = P.drop_duplicates(subset=["game_pk", "Name"]).reset_index(drop=True)
    if not H.empty:
        H = H.drop_duplicates(subset=["game_pk", "faced_pitcher", "Name"]).reset_index(drop=True)
    return P, H


def aggregate_lineup(H, rate_cols, weighted=True):
    if H is None or H.empty:
        return pd.DataFrame()
    out = []
    for (gpk, fp), g in H.groupby(["game_pk", "faced_pitcher"], sort=False):
        rec = {"game_pk": gpk, "faced_pitcher": fp,
               "pitcher_side": g["pitcher_side"].iloc[0],
               "batting_side": g["batting_side"].iloc[0],
               "n_opp_hitters": len(g)}
        for c in rate_cols:
            if c not in g.columns:
                continue
            rec[f"opp_{c}_mean"] = round(float(g[c].mean(skipna=True)), 3) if g[c].notna().any() else np.nan
            rec[f"opp_{c}_wmean"] = round(wmean(g[c], g.get(WEIGHT_COL)), 3)
            rec[f"opp_{c}"] = rec[f"opp_{c}_wmean"] if weighted else rec[f"opp_{c}_mean"]
        out.append(rec)
    return pd.DataFrame(out)


def build_matchup(P, agg, rate_cols, league_baseline):
    if P.empty or agg.empty:
        return pd.DataFrame()
    Pk = P.set_index(["game_pk", "Name"])
    rows = []
    for _, a in agg.iterrows():
        key = (a["game_pk"], a["faced_pitcher"])
        if key not in Pk.index:
            continue
        pr = Pk.loc[key]
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        side = a["pitcher_side"]
        opp_team = pr.get("home_team") if side == "away" else pr.get("away_team") if side == "home" else None
        rec = {"game_pk": a["game_pk"], "game_date": pr.get("game_date"),
               "matchup": pr.get("matchup"), "side": side, "pitcher": a["faced_pitcher"],
               "opp_team": opp_team, "n_opp": int(a["n_opp_hitters"])}
        for c in rate_cols:
            pv = pd.to_numeric(pr.get(c), errors="coerce")
            ov = a.get(f"opp_{c}")
            L = league_baseline.get(c, np.nan)
            rec[f"pit_{c}"] = round(float(pv), 3) if pd.notna(pv) else np.nan
            rec[f"opp_{c}"] = ov
            rec[f"lg_{c}"] = L
            M = matchup_value(float(pv) if pd.notna(pv) else np.nan,
                              float(ov) if pd.notna(ov) else np.nan, c, L)
            rec[f"mx_{c}"] = round(M, 3) if pd.notna(M) else np.nan
            rec[f"edge_{c}"] = round(M - L, 3) if pd.notna(M) and pd.notna(L) else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["game_pk", "side"]).reset_index(drop=True)


def build_xwoba_matchup(pitchers_df, league_baseline):
    pitcher_rows_df, opp_hitters_df = segment_pitcher_blocks(pitchers_df, STATCAST_RATE_COLS)
    opp_lineup_agg_df = aggregate_lineup(opp_hitters_df, STATCAST_RATE_COLS, weighted=USE_WEIGHTED)
    matchup_df = build_matchup(pitcher_rows_df, opp_lineup_agg_df, STATCAST_RATE_COLS, league_baseline)
    return matchup_df, pitcher_rows_df, opp_hitters_df


# ============================================================
# CELL 5 -- PLATOON / HANDEDNESS SPLIT MATCHUP
# ============================================================
MIN_SPLIT_PA = 30
MIN_PITCHER_SPLIT_BF = 50
K_BAT = 100
K_PIT = 200
K0 = 200
OPP = {"L": "R", "R": "L"}


def build_platoon_matchup(pitcher_rows_df, opp_hitters_df, people,
                          player_splits_hit, player_splits_pit,
                          league_splits=None, league_people=None):
    # League OPS cells prefer the full hitter split population (fetch_all with
    # FULL_LEAGUE_PLATOON_BASELINES); fall back to today's lineups if absent.
    _league_split_source = league_splits if league_splits else player_splits_hit
    _league_people_source = league_people if league_people else people
    _pmeta = pitcher_rows_df.set_index(["game_pk", "Name"])

    def _pitcher_attr(gpk, name, attr):
        try:
            v = _pmeta.loc[(gpk, name), attr]
            return v.iloc[0] if isinstance(v, pd.Series) else v
        except KeyError:
            return None

    def _hit_split(pid, hand):
        return (player_splits_hit.get(int(pid), {}) or {}).get(hand, {}) if pd.notna(pid) else {}

    def _pit_split(pid, hand):
        return (player_splits_pit.get(int(pid), {}) or {}).get(hand, {}) if pd.notna(pid) else {}

    def _overall(splits, pid):
        o = (splits.get(int(pid), {}) or {}).get("overall", {}) or {} if pd.notna(pid) else {}
        return o.get("ops"), (o.get("pa") or 0)

    def _shrink(obs, n, overall, overall_pa, Lcell, Loverall, K):
        if overall is not None and not pd.isna(overall) and Loverall:
            op = overall_pa or 0
            overall_reg = (op * overall + K0 * Loverall) / (op + K0)
            prior = overall_reg * (Lcell / Loverall) if Lcell else overall_reg
        else:
            prior = Lcell if Lcell else Loverall
        if obs is None or pd.isna(obs):
            return prior
        n = n or 0
        return (n * obs + K * prior) / (n + K) if (n + K) > 0 else prior

    def _bats_of(pid):
        return (_league_people_source.get(int(pid), {}) or {}).get("bats") if pd.notna(pid) else None

    def _compute_league_ops_cells():
        buckets = {("L", "L"): [], ("L", "R"): [], ("R", "L"): [], ("R", "R"): []}
        allv = []
        for pid, sp in _league_split_source.items():
            b = (_bats_of(pid) or "")[:1].upper()
            for T in ("L", "R"):
                s = sp.get(T, {}) or {}
                ops, pa = s.get("ops"), s.get("pa") or 0
                if ops is None or (isinstance(ops, float) and np.isnan(ops)) or pa <= 0:
                    continue
                eff = b if b in ("L", "R") else OPP[T]
                buckets[(eff, T)].append((ops, pa)); allv.append((ops, pa))
        Lc = {}
        for k, v in buckets.items():
            if v:
                o = np.array([x[0] for x in v]); w = np.array([x[1] for x in v], float)
                Lc[k] = round(float(np.average(o, weights=w)), 3)
        overall = round(float(np.average([x[0] for x in allv],
                        weights=[x[1] for x in allv])), 3) if allv else np.nan
        return Lc, overall

    league_ops_cell, league_ops_overall = _compute_league_ops_cells()

    rows = []
    for _, h in opp_hitters_df.iterrows():
        gpk, fp = h["game_pk"], h["faced_pitcher"]
        T = _pitcher_attr(gpk, fp, "throws")
        if T not in ("L", "R"):
            continue
        bats = (h.get("bats") or "")[:1].upper()
        pid = h.get("player_id")
        pid_p = _pitcher_attr(gpk, fp, "player_id")
        eff_stand = bats if bats in ("L", "R") else OPP[T]
        L = league_ops_cell.get((eff_stand, T), league_ops_overall)
        Lo = league_ops_overall
        B_raw = _hit_split(pid, T).get("ops"); pa_b = _hit_split(pid, T).get("pa") or 0
        P_raw = (_pit_split(pid_p, eff_stand) or {}).get("ops")
        bf_p = (_pit_split(pid_p, eff_stand) or {}).get("pa") or 0
        ov_b, ov_b_pa = _overall(player_splits_hit, pid)
        ov_p, ov_p_pa = _overall(player_splits_pit, pid_p)
        B = _shrink(B_raw, pa_b, ov_b, ov_b_pa, L, Lo, K_BAT)
        P = _shrink(P_raw, bf_p, ov_p, ov_p_pa, L, Lo, K_PIT)
        Mi = (B * P / L) if (B is not None and P is not None and L) and not (
            pd.isna(B) or pd.isna(P)) else np.nan
        rows.append({
            "game_pk": gpk, "matchup": h.get("matchup"), "faced_pitcher": fp,
            "pitcher_throws": T, "batter": h["Name"], "bats": bats or "?",
            "eff_stand": eff_stand, "platoon_adv": eff_stand != T,
            "ops_vs_hand_raw": B_raw, "ops_vs_hand": round(B, 3) if pd.notna(B) else np.nan,
            "pit_ops_allowed_raw": P_raw, "pit_ops_allowed": round(P, 3) if pd.notna(P) else np.nan,
            "lg_cell": L, "mx_ops": round(Mi, 3) if pd.notna(Mi) else np.nan,
            "split_pa": pa_b, "pit_split_bf": bf_p,
            "low_sample": pa_b < MIN_SPLIT_PA,
            "pit_low_sample": bf_p < MIN_PITCHER_SPLIT_BF,
        })
    opp_platoon_detail_df = pd.DataFrame(rows)

    def _wmean(v, w):
        v = pd.to_numeric(pd.Series(v).reset_index(drop=True), errors="coerce")
        w = pd.to_numeric(pd.Series(w).reset_index(drop=True), errors="coerce")
        m = v.notna() & w.notna() & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    if opp_platoon_detail_df.empty:
        return pd.DataFrame(), opp_platoon_detail_df, league_ops_overall

    plat_rows = []
    for (gpk, fp), g in opp_platoon_detail_df.groupby(["game_pk", "faced_pitcher"], sort=False):
        T = g["pitcher_throws"].iloc[0]
        # All aggregates are lineup-composition weighted (clip 0-PA splits to 1 so
        # prior-driven bats still count as lineup exposure, incl. SP OPS display).
        lineup_w = pd.to_numeric(g["split_pa"], errors="coerce").fillna(0).clip(lower=1)
        opp_ops_raw = _wmean(g["ops_vs_hand_raw"], lineup_w)
        opp_ops = _wmean(g["ops_vs_hand"], lineup_w)
        pit_ops_raw = _wmean(g["pit_ops_allowed_raw"], lineup_w)
        pit_ops = _wmean(g["pit_ops_allowed"], lineup_w)
        mx_ops = _wmean(g["mx_ops"], lineup_w)
        edge = mx_ops - league_ops_overall if pd.notna(mx_ops) else np.nan
        bf_present = [v for v in g.loc[g["pit_split_bf"] > 0, "pit_split_bf"].unique()]
        pit_min_bf = int(min(bf_present)) if bf_present else 0
        pit_low = bool(g["pit_low_sample"].any())
        meta_row = opp_hitters_df[(opp_hitters_df.game_pk == gpk) &
                                  (opp_hitters_df.faced_pitcher == fp)].iloc[0]
        opp_team = meta_row["home_team"] if meta_row["pitcher_side"] == "away" else meta_row["away_team"]
        plat_rows.append({
            "game_pk": gpk, "game_date": meta_row.get("game_date"), "matchup": meta_row["matchup"],
            "side": meta_row["pitcher_side"], "pitcher": fp, "throws": T, "opp_team": opp_team,
            "n_opp": len(g),
            "n_LHB": int((g["bats"] == "L").sum()), "n_RHB": int((g["bats"] == "R").sum()),
            "n_SW": int((g["bats"] == "S").sum()),
            "n_platoon_adv": int(g["platoon_adv"].sum()),
            "n_low_sample": int(g["low_sample"].sum()),
            "pit_min_split_bf": pit_min_bf, "pit_low_sample": pit_low,
            "reliable": (not pit_low) and (int(g["low_sample"].sum()) <= 4),
            "opp_OPS_raw": round(opp_ops_raw, 3) if pd.notna(opp_ops_raw) else np.nan,
            "opp_OPS_vs_hand": round(opp_ops, 3) if pd.notna(opp_ops) else np.nan,
            "pit_OPS_raw": round(pit_ops_raw, 3) if pd.notna(pit_ops_raw) else np.nan,
            "pit_OPS_allowed": round(pit_ops, 3) if pd.notna(pit_ops) else np.nan,
            "mx_OPS": round(mx_ops, 3) if pd.notna(mx_ops) else np.nan,
            "edge_OPS": round(edge, 3) if pd.notna(edge) else np.nan,
        })
    matchup_platoon_df = (pd.DataFrame(plat_rows)
                          .sort_values(["game_pk", "side"]).reset_index(drop=True))
    return matchup_platoon_df, opp_platoon_detail_df, league_ops_overall


# ============================================================
# CELL 6 -- CONSOLIDATED CARD RENDER (emits HTML string)
# ============================================================
ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def tint(edge, domain, ksign=1):
    if edge is None:
        return "transparent"
    s = edge * ksign
    t = clamp(abs(s) / domain, 0, 1) ** 0.85
    rgb = "var(--warm)" if s >= 0 else "var(--cool)"
    a = round(0.08 + 0.60 * t, 3)
    return f"rgba({rgb},{a})"


def edge_color(edge, ksign=1):
    if edge is None:
        return "var(--faint)"
    return "rgba(var(--warm),1)" if edge * ksign >= 0 else "rgba(var(--cool),1)"


def f3(v):
    return "—" if v is None else f"{v:.3f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.3f}"


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def sgn3(v):
    return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v):.3f}".rstrip()


def sgn1(v):
    return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v):.1f}"


def _abbr(name):
    return ABBR.get(name, str(name or "")[:3].upper())


def meter(away_abbr, home_abbr, away_off, home_off, domain):
    if away_off is None or home_off is None:
        return ""
    net = home_off - away_off
    pos = 50 + clamp(net / domain, -1, 1) * 44
    if pos >= 50:
        fl, fw = 50, pos - 50
    else:
        fl, fw = pos, 50 - pos
    fav_home = home_off >= away_off
    lcl = "" if fav_home else " on"
    rcl = " on" if fav_home else ""
    return (
        f"<div class='mx-meter'>"
        f"<span class='pole left{lcl}'>{away_abbr}</span>"
        f"<div class='track'>"
        f"<div class='fill' style='left:{fl:.1f}%;width:{fw:.1f}%'></div>"
        f"<div class='cen'></div>"
        f"<div class='mark' style='left:{pos:.1f}%'></div>"
        f"</div>"
        f"<span class='pole right{rcl}'>{home_abbr}</span>"
        f"</div>")


XW_METRICS = [
    ("xw", "xwOBA", f3, 0.060, 1),
    ("hh", "HardHit", f1, 12.0, 1),
    ("ev", "EV", f1, 3.0, 1),
    ("xs", "xSLG", f3, 0.120, 1),
    ("k", "K%", f1, 9.0, -1),
]


def _xw_strip(r):
    chips = ""
    for key, lab, fmt, dom, ks in XW_METRICS:
        pit, opp, mx, edge = r[key]
        bg = tint(edge, dom, ks)
        title = f"{lab}  pit {fmt(pit)} / opp {fmt(opp)} -> mx {fmt(mx)}  (edge {sgn3(edge) if dom < 1 else sgn1(edge)})"
        chips += (f"<div class='chip' style='background:{bg}' title='{title}'>"
                  f"<div class='lab'>{lab}</div>"
                  f"<div class='val'>{fmt(mx)}</div>"
                  f"<div class='sub'>{fmt(pit)}<span>/</span>{fmt(opp)}</div></div>")
    return chips


def _pl_chip(r):
    if not r.get("has_pl"):
        return ("<div class='chip wide muted'><div class='lab'>OPS matchup</div>"
                "<div class='val'>—</div><div class='sub'>no vs-hand split</div></div>", None, "var(--faint)")
    rel = r["pl_reliable"]; edge = r["pl_edge"]
    bg = tint(edge, 0.20, 1) if rel else "transparent"
    ecol = edge_color(edge) if rel else "var(--faint)"
    flags = ""
    fl = r.get("pl_fl", {})
    if "thin" in fl:
        flags += f"<span class='flag warn'>thin SP {fl['thin']}bf</span>"
    if "lowpa" in fl:
        flags += f"<span class='flag'>{fl['lowpa']} low-PA</span>"
    if not rel:
        flags += "<span class='flag mute'>prior-driven</span>"
    chip = (f"<div class='chip wide{'' if rel else ' unrel'}' style='background:{bg}'>"
            f"<div class='lab'>OPS matchup{flags}</div>"
            f"<div class='val'>{f3(r['pl_mx'])}</div>"
            f"<div class='sub'>opp {f3(r['pl_opp_raw'])}<span>/</span>SP {f3(r['pl_sp_raw'])}</div></div>")
    return chip, edge, ecol


def cmb_row(side, r, opp_abbr):
    badge = f" <em>{r['t']}</em>" if r.get('t') in ('L', 'R') else ""
    comp = f"{r['R']}R/{r['L']}L" + (f"/{r['S']}S" if r['S'] else "") if r.get("has_pl") else "—"
    padv = f" · {r['padv']} plt-adv" if r.get("has_pl") else ""
    xw_oe = r["xw"][3]
    pl_chip, pl_edge, pl_ecol = _pl_chip(r)
    return (
        f"<div class='prow'>"
        f"<div class='pmeta'>"
        f"<div class='pname'>{r['p']}{badge}</div>"
        f"<div class='prole'>{side} SP · vs {opp_abbr} · {comp}{padv}</div>"
        f"</div>"
        f"<div class='lensline'>"
        f"<div class='strip'>{_xw_strip(r)}</div>"
        f"<div class='offedge' style='color:{edge_color(xw_oe)}' "
        f"title='opponent-offense xwOBA edge vs league -- drives the lean'>"
        f"<span>xw edge</span>{sgn3(xw_oe)}</div>"
        f"</div>"
        f"<div class='lensline pl'>"
        f"{pl_chip}"
        f"<div class='offedge' style='color:{pl_ecol}' title='opponent-offense platoon-OPS edge vs league'>"
        f"<span>ops edge</span>{sgn3(pl_edge)}</div>"
        f"</div>"
        f"</div>")


def _consensus(away_team, home_team, a, h):
    xw_home, xw_away = a["xw"][3], h["xw"][3]
    xw_fav = home_team if (xw_home is not None and xw_away is not None and xw_home >= xw_away) else away_team
    xw_d = abs((xw_home or 0) - (xw_away or 0)) if (xw_home is not None and xw_away is not None) else None
    pl_home = a["pl_edge"] if (a.get("has_pl") and a.get("pl_reliable")) else None
    pl_away = h["pl_edge"] if (h.get("has_pl") and h.get("pl_reliable")) else None
    if pl_home is not None and pl_away is not None:
        pl_fav = home_team if pl_home >= pl_away else away_team
        pl_d = abs(pl_home - pl_away)
        if pl_fav == xw_fav:
            tag, tcl = "AGREE", "agree"
        else:
            tag, tcl = "DIVERGE", "diverge"
        pl_txt = f"OPS → <b>{pl_fav}</b> Δ{pl_d:.3f}"
    else:
        tag, tcl = "n/a", "na"
        pl_txt = "OPS → <span class='muted'>unreliable / no split</span>"
    xw_txt = (f"xwOBA → <b>{xw_fav}</b> Δ{xw_d:.3f}" if xw_d is not None
              else "xwOBA → <span class='muted'>—</span>")
    return (f"<div class='consensus'>{xw_txt}<span class='dot'>·</span>{pl_txt}"
            f"<span class='ctag {tcl}'>{tag}</span></div>")


def cmb_card(g):
    a, h = g["away"], g["home"]
    home_team = _abbr(a["opp"])
    away_team = _abbr(h["opp"])
    home_off, away_off = a["xw"][3], h["xw"][3]
    ho = home_off if home_off is not None else 0.0
    ao = away_off if away_off is not None else 0.0
    delta = abs(ho - ao)
    fav = home_team if ho >= ao else away_team
    pill_a = round(clamp(delta / 0.06, 0, 1) * 0.16 + 0.04, 3)
    return (
        f"<article class='card'>"
        f"<header class='head'>"
        f"<div class='eyebrow'>{away_team} <span>@</span> {home_team}</div>"
        f"<div class='lean' style='background:rgba(var(--lean),{pill_a})'>"
        f"<span class='lk'>lean</span><span class='lt'>{fav}</span>"
        f"<span class='ld'>Δxw {delta:.3f}</span></div>"
        f"</header>"
        f"{meter(away_team, home_team, away_off, home_off, 0.08)}"
        f"{_consensus(away_team, home_team, a, h)}"
        f"<div class='rows'>{cmb_row('AWAY', a, home_team)}{cmb_row('HOME', h, away_team)}</div>"
        f"</article>")


def build_combined(games):
    cards = sorted(
        games,
        key=lambda g: abs((g['away']['xw'][3] or 0) - (g['home']['xw'][3] or 0)),
        reverse=True)
    return "<div class='grid'>" + "".join(cmb_card(g) for g in cards) + "</div>"


def _f(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows_by_side(gg):
    a = gg[gg["side"] == "away"]; h = gg[gg["side"] == "home"]
    return (a.iloc[0] if len(a) else None), (h.iloc[0] if len(h) else None)


def _tup(r, col):
    return (_f(r.get(f"pit_{col}")), _f(r.get(f"opp_{col}")),
            _f(r.get(f"mx_{col}")), _f(r.get(f"edge_{col}")))


def _pl_lookup(pl_df):
    out = {}
    if pl_df is None or getattr(pl_df, "empty", True):
        return out
    for _, r in pl_df.iterrows():
        out[(r["game_pk"], r["side"])] = r
    return out


def _df_to_combined_games(xw_df, pl_df, throws):
    pl_map = _pl_lookup(pl_df)
    games = []
    for gpk, gg in xw_df.groupby("game_pk", sort=False):
        a, h = _rows_by_side(gg)
        if a is None or h is None:
            continue

        def mk(r):
            side = r["side"]
            t = throws.get((r["game_pk"], r["pitcher"]), "")
            d = dict(p=r["pitcher"], t=t if t in ("L", "R") else "", opp=r["opp_team"],
                     xw=_tup(r, "xwOBA"), hh=_tup(r, "Hard Hit%"), ev=_tup(r, "EV"),
                     xs=_tup(r, "xSLG"), k=_tup(r, "K%"),
                     has_pl=False, R=0, L=0, S=0, padv=0, pl_fl={})
            pr = pl_map.get((r["game_pk"], side))
            if pr is not None:
                fl = {}
                if bool(pr.get("pit_low_sample")) and int(pr.get("pit_min_split_bf") or 0) > 0:
                    fl["thin"] = int(pr["pit_min_split_bf"])
                if int(pr.get("n_low_sample") or 0) > 0:
                    fl["lowpa"] = int(pr["n_low_sample"])
                if not d["t"]:
                    pt = pr.get("throws"); d["t"] = pt if pt in ("L", "R") else ""
                d.update(has_pl=True,
                         R=int(pr.get("n_RHB") or 0), L=int(pr.get("n_LHB") or 0),
                         S=int(pr.get("n_SW") or 0), padv=int(pr.get("n_platoon_adv") or 0),
                         pl_opp_raw=_f(pr.get("opp_OPS_raw")), pl_sp_raw=_f(pr.get("pit_OPS_raw")),
                         pl_mx=_f(pr.get("mx_OPS")), pl_edge=_f(pr.get("edge_OPS")),
                         pl_reliable=bool(pr.get("reliable")), pl_fl=fl)
            return d
        games.append(dict(away=mk(a), home=mk(h)))
    return games


def _legend(model_label, built_txt):
    date = SLATE_DATE
    dtxt = f" · {date}" if date else ""
    btxt = f" · built {built_txt}" if built_txt else ""
    return (
        "<div class='legend'>"
        f"<div class='lg-title'>{model_label}{dtxt}{btxt} · "
        "<em>offense-vs-starter contact lean, not a win projection</em></div>"
        "<div class='lg-keys'>"
        "<span class='k'><i class='sw warm'></i>offense-favorable</span>"
        "<span class='k'><i class='sw cool'></i>pitcher-favorable</span>"
        "<span class='k'><i class='sw lean'></i>lean / net tilt (xwOBA)</span>"
        "<span class='k'><i class='sw grey'></i>unreliable (prior-driven)</span>"
        "<span class='k note'>top row = Statcast xwOBA · bottom = platoon OPS · cards sorted by xwOBA Δedge</span>"
        "</div></div>")


CSS = r"""/* ============================================================
   MLB matchup leans -- design tokens
   ============================================================ */
:root{
  --bg:#eef0f4; --surface:#ffffff; --surface-2:#f7f8fb; --ink:#15171c;
  --muted:#646b78; --faint:#9aa1ad; --line:#e6e8ee; --line-2:#eef0f5;
  --warm:206,74,46;
  --cool:18,138,140;
  --lean:88,90,196;
  --chip-fg:#1b1e25;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Arial,sans-serif;
  --shadow:0 1px 2px rgba(16,18,29,.05),0 14px 34px -22px rgba(16,18,29,.30);
  --r:14px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c0e13; --surface:#161a22; --surface-2:#11141b; --ink:#e9ebf0;
  --muted:#98a0ac; --faint:#646b78; --line:#242a35; --line-2:#1b202a;
  --warm:232,108,76; --cool:52,176,176; --lean:124,126,228; --chip-fg:#e9ebf0;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 18px 40px -24px rgba(0,0,0,.8);
}}
html[data-theme="dark"]{
  --bg:#0c0e13; --surface:#161a22; --surface-2:#11141b; --ink:#e9ebf0;
  --muted:#98a0ac; --faint:#646b78; --line:#242a35; --line-2:#1b202a;
  --warm:232,108,76; --cool:52,176,176; --lean:124,126,228; --chip-fg:#e9ebf0;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 18px 40px -24px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;padding:22px 16px 60px}
.mx-wrap{max-width:760px;margin:0 auto}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
.brand{font:750 15px/1 var(--mono);letter-spacing:.02em}
.theme{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--muted);
  font:600 12px/1 var(--sans);padding:8px 11px;border-radius:9px;cursor:pointer}
.theme:hover{color:var(--ink)}

.legend{margin:2px 2px 16px}
.lg-title{font-size:13px;font-weight:650;letter-spacing:.01em;margin-bottom:8px}
.lg-title em{font-style:normal;color:var(--muted);font-weight:500}
.lg-keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:11.5px;color:var(--muted)}
.lg-keys .k{display:inline-flex;align-items:center;gap:6px}
.lg-keys .note{color:var(--faint)}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.sw.warm{background:rgba(var(--warm),.85)} .sw.cool{background:rgba(var(--cool),.85)}
.sw.lean{background:rgba(var(--lean),.9)}  .sw.grey{background:var(--line);border:1px solid var(--faint)}

.grid{display:flex;flex-direction:column;gap:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow);overflow:hidden}

.head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:13px 16px 9px}
.eyebrow{font:700 15px/1 var(--mono);letter-spacing:.04em}
.eyebrow span{color:var(--faint);font-weight:500;margin:0 3px}
.lean{display:inline-flex;align-items:baseline;gap:8px;padding:5px 11px;border-radius:999px;
  border:1px solid var(--line-2)}
.lean .lk{font:600 10px/1 var(--sans);text-transform:uppercase;letter-spacing:.13em;color:var(--muted)}
.lean .lt{font:750 15px/1 var(--mono);letter-spacing:.03em;color:var(--ink)}
.lean .ld{font:500 11.5px/1 var(--mono);color:var(--muted)}

.mx-meter{display:flex;align-items:center;gap:10px;padding:2px 16px 14px}
.mx-meter .pole{font:600 11px/1 var(--mono);letter-spacing:.04em;color:var(--faint);width:34px;
  text-align:center;transition:.15s}
.mx-meter .pole.left{text-align:right} .mx-meter .pole.right{text-align:left}
.mx-meter .pole.on{color:var(--ink)}
.track{position:relative;flex:1;height:7px;background:var(--line-2);border-radius:999px}
.track .cen{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--faint);opacity:.5}
.track .fill{position:absolute;top:0;bottom:0;background:rgba(var(--lean),.30);border-radius:999px}
.track .mark{position:absolute;top:50%;width:13px;height:13px;border-radius:50%;
  background:rgba(var(--lean),1);transform:translate(-50%,-50%);
  box-shadow:0 0 0 3px var(--surface),0 1px 4px rgba(0,0,0,.28)}

.rows{border-top:1px solid var(--line-2)}
.prow{padding:11px 16px;display:flex;flex-direction:column;gap:9px}
.prow + .prow{border-top:1px solid var(--line-2)}
.pmeta{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
.pname{font:650 14px/1.1 var(--sans)}
.pname em{font-style:normal;font-family:var(--mono);font-size:11px;color:#fff;
  background:var(--faint);padding:1px 5px;border-radius:5px;margin-left:5px;vertical-align:1px}
.prole{font-size:11.5px;color:var(--muted);letter-spacing:.01em}

.strip{display:flex;gap:6px;flex-wrap:wrap}
.chip{flex:1 1 64px;min-width:60px;border:1px solid var(--line-2);border-radius:9px;
  padding:6px 7px 5px;text-align:center;background:transparent}
.chip .lab{font:600 9px/1 var(--sans);text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.chip .val{font:600 16px/1.15 var(--mono);color:var(--chip-fg);font-variant-numeric:tabular-nums;margin-top:2px}
.chip .sub{font:400 10px/1 var(--mono);color:var(--muted);margin-top:2px}
.chip .sub span{color:var(--faint);margin:0 1px}
.chip.wide{flex:0 0 150px;text-align:left}
.chip.wide .val{font-size:18px}

.offedge{font:700 15px/1 var(--mono);font-variant-numeric:tabular-nums;text-align:right;
  display:flex;align-items:baseline;justify-content:flex-end;gap:8px}
.offedge span{font:600 9.5px/1 var(--sans);text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}

.flag{font:600 9.5px/1 var(--mono);letter-spacing:.02em;color:var(--muted);
  background:var(--surface-2);border:1px solid var(--line);border-radius:5px;
  padding:2px 5px;margin-left:4px;white-space:nowrap}
.flag.warn{color:rgba(var(--warm),1);border-color:rgba(var(--warm),.35);background:rgba(var(--warm),.07)}
.flag.mute{color:var(--faint)}

@media (max-width:540px){
  .head{flex-direction:column;align-items:flex-start;gap:7px}
  .chip.wide{flex:1}
  .offedge{justify-content:flex-start}
}
"""

CSS_COMBINED = r"""
.lensline{display:flex;align-items:center;gap:10px}
.lensline + .lensline{margin-top:8px}
.lensline > .strip{flex:1 1 auto}
.lensline.pl{padding-top:7px;border-top:1px dashed var(--line-2)}
.lensline.pl .chip.wide{flex:0 0 168px}
.chip.wide.muted{opacity:.55} .chip.wide.muted .val{color:var(--faint)}
.chip.wide.unrel{border-style:dashed}
.chip.wide .lab{display:flex;align-items:center;gap:5px;flex-wrap:wrap}

.consensus{display:flex;align-items:center;flex-wrap:wrap;gap:7px;
  padding:0 16px 12px;font:600 12px/1.3 var(--mono);color:var(--ink);
  font-variant-numeric:tabular-nums}
.consensus .muted{color:var(--muted);font-weight:500}
.consensus .dot{color:var(--faint)}
.consensus .ctag{font:700 9.5px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;
  padding:3px 7px;border-radius:6px;margin-left:auto}
.consensus .ctag.agree{color:rgba(var(--warm),1);background:rgba(var(--warm),.12);
  border:1px solid rgba(var(--warm),.3)}
.consensus .ctag.diverge{color:rgba(var(--cool),1);background:rgba(var(--cool),.12);
  border:1px solid rgba(var(--cool),.3)}
.consensus .ctag.na{color:var(--faint);background:var(--surface-2);border:1px solid var(--line)}

@media (max-width:540px){
  .lensline{flex-wrap:wrap}
  .lensline.pl .chip.wide{flex:1 1 auto}
  .consensus .ctag{margin-left:0}
}
"""

CSS_GRADES = r"""
.gradestrip{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:0 2px 16px;
  padding:10px 14px;background:var(--surface);border:1px solid var(--line);border-radius:11px;
  font:600 12px/1.4 var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}
.gradestrip .lab{font:600 9.5px/1 var(--sans);text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.gradestrip .muted{color:var(--muted);font-weight:500}
.gradestrip a{color:rgba(var(--lean),1);text-decoration:none;margin-left:auto;font:600 12px/1 var(--sans);white-space:nowrap}
.gradestrip a:hover{text-decoration:underline}

.backlink{font:600 12px/1 var(--sans);margin:0 2px 12px}
.backlink a{color:rgba(var(--lean),1);text-decoration:none}
.backlink a:hover{text-decoration:underline}
.gr-summary{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}
.gr-summary .chip{flex:1 1 110px;background:var(--surface);border-color:var(--line)}
.gr-note{font:500 11.5px/1.5 var(--mono);color:var(--muted);margin:0 2px 14px}
.gr-tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r);box-shadow:var(--shadow)}
table.gr{border-collapse:collapse;width:100%;min-width:760px;
  font:500 12px/1.35 var(--mono);font-variant-numeric:tabular-nums}
table.gr th{font:600 9.5px/1 var(--sans);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap;
  position:sticky;top:0;background:var(--surface)}
table.gr td{padding:8px 10px;border-bottom:1px solid var(--line-2);white-space:nowrap;vertical-align:top}
table.gr tr:last-child td{border-bottom:none}
table.gr tr.void{opacity:.5}
table.gr .sp{font:400 11px/1.3 var(--mono);color:var(--muted)}
.wlt{display:inline-block;min-width:16px;text-align:center;font:700 11px/1 var(--mono);
  padding:3px 6px;border-radius:6px}
.wlt.W{color:rgba(var(--cool),1);background:rgba(var(--cool),.12);border:1px solid rgba(var(--cool),.3)}
.wlt.L{color:rgba(var(--warm),1);background:rgba(var(--warm),.12);border:1px solid rgba(var(--warm),.3)}
.wlt.T{color:var(--muted);background:var(--surface-2);border:1px solid var(--line)}
.wlt.none{color:var(--faint);background:transparent;border:1px dashed var(--line)}
.st{font:600 9.5px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.st.void{color:rgba(var(--warm),.9)}
"""

THEME_JS = r"""
(function(){
  var KEY='mlb-mx-theme';
  var btn=document.getElementById('themeBtn');
  function cur(){return document.documentElement.getAttribute('data-theme')||'auto';}
  function apply(m){
    if(m==='auto'){document.documentElement.removeAttribute('data-theme');}
    else{document.documentElement.setAttribute('data-theme',m);}
    if(btn){btn.textContent='Theme: '+m;}
  }
  try{apply(localStorage.getItem(KEY)||'auto');}catch(e){apply('auto');}
  if(btn){btn.addEventListener('click',function(){
    var order=['auto','light','dark'];var n=order[(order.indexOf(cur())+1)%3];
    apply(n);try{localStorage.setItem(KEY,n);}catch(e){}
  });}
})();
"""


def html_document(body, built_txt, title=None):
    title = title or f"MLB matchup leans — {SLATE_DATE}"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        "<meta name='robots' content='noindex'>"
        f"<style>{CSS}{CSS_COMBINED}{CSS_GRADES}</style></head><body>"
        f"<div class='mx-wrap'>"
        "<div class='topbar'><div class='brand'>MLB matchup leans</div>"
        "<button id='themeBtn' class='theme' type='button'>Theme: auto</button></div>"
        f"{body}</div>"
        f"<script>{THEME_JS}</script>"
        "</body></html>")


# ============================================================
# GRADES -- records strip (index) + full ledger page (grades.html)
# Renders purely from data/mlb_lean_ledger.csv, which grade_leans.py
# maintains and CI commits back to the repo; the grading pass runs
# before this build so records are current as of the run.
# ============================================================
def load_ledger_df():
    if not os.path.exists(LEDGER_PATH):
        return None
    try:
        led = pd.read_csv(LEDGER_PATH)
    except Exception as e:  # noqa: BLE001
        log(f"Ledger unreadable, grades render degraded: {e!r}")
        return None
    return None if led.empty else led


def _esc(x):
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _rec_parts(s):
    """W/L/T series -> ('W-L' or 'W-L-T', 'pct' or None)."""
    s = s.dropna()
    w, l, t = int((s == "W").sum()), int((s == "L").sum()), int((s == "T").sum())
    base = f"{w}-{l}" + (f"-{t}" if t else "")
    return base, (f"{w / (w + l):.3f}" if (w + l) else None)


def _rec_txt(s):
    base, pct = _rec_parts(s)
    return f"{base} ({pct})" if pct else base


def records_strip_html():
    led = load_ledger_df()
    if led is None:
        return ""
    g = led[(led["status"] == "graded") & (led["model_tag"] == MODEL_TAG)]
    if g.empty:
        inner = "<span class='muted'>no graded games yet</span>"
    else:
        bits = [f"xwOBA full {_rec_txt(g['xw_full'])}",
                f"F5 {_rec_txt(g['xw_f5'])}"]
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            bits.append(f"platoon full {_rec_txt(ov['ops_full'])}")
            bits.append(f"F5 {_rec_txt(ov['ops_f5'])}")
        inner = " <span class='muted'>·</span> ".join(bits)
    return ("<div class='gradestrip'><span class='lab'>Record</span>"
            f"<span>{inner}</span><a href='grades.html'>full ledger →</a></div>")


def _wlt_badge(v):
    if isinstance(v, str) and v in ("W", "L", "T"):
        return f"<span class='wlt {v}'>{v}</span>"
    return "<span class='wlt none'>–</span>"


def _fmt_ml(v):
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return "<span class='muted'>—</span>"
    return f"+{int(v)}" if v > 0 else f"{int(v)}"


def _lean_ml_cell(r, lean_col):
    """Closing ML of the lean's side; '—' when no market data."""
    lean = r.get(lean_col)
    if not isinstance(lean, str) or not lean:
        return "<span class='muted'>—</span>"
    return _fmt_ml(r.get("close_home_ml") if lean == r.get("home") else r.get("close_away_ml"))


def _lean_cell(lean, delta, muted=False):
    if not isinstance(lean, str) or not lean:
        return "<span class='muted'>—</span>"
    d = pd.to_numeric(delta, errors="coerce")
    txt = _esc(lean) + (f" <span class='muted'>Δ{d:.3f}</span>" if pd.notna(d) else "")
    return f"<span class='muted'>{txt}</span>" if muted else txt


def _grades_row(r, show_tag, show_ml=False):
    status = str(r["status"])
    fa, fh = pd.to_numeric(r["full_away"], errors="coerce"), pd.to_numeric(r["full_home"], errors="coerce")
    f5a, f5h = pd.to_numeric(r["f5_away"], errors="coerce"), pd.to_numeric(r["f5_home"], errors="coerce")
    if status != "graded":
        final = f"<span class='st {status}'>{_esc(status)}</span>"
        f5 = "<span class='muted'>—</span>"
    else:
        final = f"{int(fa)}–{int(fh)}" if pd.notna(fa) and pd.notna(fh) else "<span class='muted'>—</span>"
        f5 = f"{int(f5a)}–{int(f5h)}" if pd.notna(f5a) and pd.notna(f5h) else "<span class='muted'>—</span>"
    ops_valid = bool(r["ops_valid"]) if pd.notna(r["ops_valid"]) else False
    cells = [
        _esc(r["game_date"]),
        (f"{_esc(r['away'])} <span class='muted'>@</span> {_esc(r['home'])}"
         f"<br><span class='sp'>{_esc(r.get('away_sp') or '—')} v {_esc(r.get('home_sp') or '—')}</span>"),
        _lean_cell(r["xw_lean"], r["xw_delta"]),
        _lean_cell(r["ops_lean"] if ops_valid else None, r["ops_delta"]) if ops_valid
        else _lean_cell(r["ops_lean"], r["ops_delta"], muted=True),
    ]
    if show_ml:
        cells += [_lean_ml_cell(r, "xw_lean"), _lean_ml_cell(r, "ops_lean")]
    cells += [
        final, f5,
        _wlt_badge(r["xw_full"]), _wlt_badge(r["xw_f5"]),
        _wlt_badge(r["ops_full"]), _wlt_badge(r["ops_f5"]),
    ]
    if show_tag:
        cells.append(f"<span class='muted'>{_esc(r['model_tag'])}</span>")
    cls = " class='void'" if status == "void" else ""
    return f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def render_grades_html(built_txt):
    back = "<div class='backlink'><a href='index.html'>← today's leans</a></div>"
    led = load_ledger_df()
    if led is None:
        body = back + ("<div class='legend'><div class='lg-title'>Grading ledger · "
                       "no graded data yet — the ledger appears after the first CI run."
                       "</div></div>")
        return html_document(body, built_txt, title="MLB lean grades")

    n_graded = int((led["status"] == "graded").sum())
    n_pend = int((led["status"] == "pending").sum())
    n_void = int((led["status"] == "void").sum())
    head = ("<div class='legend'><div class='lg-title'>"
            f"Grading ledger · {n_graded} graded · {n_pend} pending"
            + (f" · {n_void} void" if n_void else "")
            + f" · built {built_txt}<br>"
            f"<em>records below are for model {_esc(MODEL_TAG)} · graded rows are immutable · "
            "platoon records count reliable-split games only</em></div></div>")

    g = led[(led["status"] == "graded") & (led["model_tag"] == MODEL_TAG)]
    chips, notes = [], []

    def chip(lab, val, sub=None):
        s = f"<div class='sub'>{sub}</div>" if sub else ""
        chips.append(f"<div class='chip'><div class='lab'>{lab}</div>"
                     f"<div class='val'>{val}</div>{s}</div>")

    if g.empty:
        summary = "<div class='gr-note'>No graded games under this model tag yet.</div>"
    else:
        chip("Graded", str(len(g)), f"{n_pend} pending")
        b, p = _rec_parts(g["xw_full"]); chip("xwOBA · full", b, p)
        b, p = _rec_parts(g["xw_f5"]);   chip("xwOBA · F5", b, p)
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            b, p = _rec_parts(ov["ops_full"]); chip("Platoon · full", b, p)
            b, p = _rec_parts(ov["ops_f5"]);   chip("Platoon · F5", b, p)
            notes.append(f"reliable-platoon subset n={len(ov)}: xwOBA on those games "
                         f"full {_rec_txt(ov['xw_full'])}, F5 {_rec_txt(ov['xw_f5'])}")
        dv = g[g["consensus"] == "DIVERGE"]
        if len(dv):
            notes.append(f"DIVERGE head-to-head (F5): xwOBA {int((dv['xw_f5'] == 'W').sum())} — "
                         f"platoon {int((dv['ops_f5'] == 'W').sum())}  (n={len(dv)})")
        # vs-market scoreboard (closing DK MLs attached by grade_leans.py via
        # market_backfill; columns absent until the first market run).
        if "close_p_home" in g.columns and g["close_p_home"].notna().any():
            try:
                from market_backfill import vs_market_summary
                mkt = vs_market_summary(g)
            except Exception as e:  # noqa: BLE001
                log(f"vs-market summary degraded: {e!r}")
                mkt = {}
            for lab, key in (("xwOBA · vs mkt", "xwOBA"), ("Platoon · vs mkt", "platoon")):
                m = mkt.get(key)
                if m:
                    chip(lab, f"{m['w']}-{m['n'] - m['w']}",
                         f"z {m['z']:+.2f} · {m['roi_units']:+.2f}u flat")
            if mkt:
                notes.append("market = closing DK moneylines via ESPN, devigged two-way; "
                             "vs-market z and flat ROI are the primary metrics")
        summary = ("<div class='gr-summary'>" + "".join(chips) + "</div>"
                   + (f"<div class='gr-note'>{' · '.join(notes)}</div>" if notes else ""))

    show_tag = led["model_tag"].nunique() > 1
    show_ml = "close_home_ml" in led.columns and led["close_home_ml"].notna().any()
    heads = (["Date", "Game", "xwOBA lean", "Platoon lean"]
             + (["xw ML", "pl ML"] if show_ml else [])
             + ["Final", "F5", "xw F", "xw F5", "pl F", "pl F5"]
             + (["Model"] if show_tag else []))
    led = led.sort_values(["game_date", "game_pk"], ascending=[False, True])
    rows = "".join(_grades_row(r, show_tag, show_ml) for _, r in led.iterrows())
    table = ("<div class='gr-tablewrap'><table class='gr'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in heads)
             + f"</tr></thead><tbody>{rows}</tbody></table></div>")
    return html_document(back + head + summary + table, built_txt, title="MLB lean grades")


def render_combined_html(xw_df, pl_df, pitcher_rows_df, built_txt):
    throws = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, pr in pitcher_rows_df.iterrows():
            throws[(pr["game_pk"], pr["Name"])] = pr.get("throws")
    games = _df_to_combined_games(xw_df, pl_df, throws)
    legend = _legend("MLB matchup leans — xwOBA + platoon OPS", built_txt)
    strip = records_strip_html()
    if not games:
        inner = legend + strip + "<div class='legend'><div class='lg-title'>No paired probables yet — " \
                                 "probables/lineups not posted. Check back closer to first pitch.</div></div>"
        return html_document(inner, built_txt)
    body = legend + strip + build_combined(games)
    return html_document(body, built_txt)


def empty_slate_html(built_txt):
    body = (
        "<div class='legend'>"
        f"<div class='lg-title'>MLB matchup leans · {SLATE_DATE} · built {built_txt}</div>"
        "<div class='lg-keys'><span class='k'>No MLB games scheduled for this date.</span></div>"
        "</div>") + records_strip_html()
    return html_document(body, built_txt)


# ============================================================
# MAIN
# ============================================================
def main():
    built_txt = datetime.now(ET).strftime("%H:%M ET · %Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")

    # Grades page renders purely from the committed ledger, so it's written
    # up front and ships on every path that deploys (full, empty-slate,
    # check-back). If the build aborts on a fetch failure nothing deploys,
    # same as before.
    grades_path = os.path.join(OUT_DIR, "grades.html")
    with open(grades_path, "w") as f:
        f.write(render_grades_html(built_txt))
    log(f"Wrote {grades_path}")

    # Top-level guard: a failed core fetch must NOT write index.html, so CI
    # skips the deploy and the last good page stays live.
    data = fetch_all(SLATE_DATE)

    if data.get("empty"):
        log("No games for this date -> writing friendly empty-slate page.")
        with open(out_path, "w") as f:
            f.write(empty_slate_html(built_txt))
        log(f"Wrote {out_path}")
        return 0

    log("Building xwOBA matchup ...")
    matchup_df, pitcher_rows_df, opp_hitters_df = build_xwoba_matchup(
        data["pitchers_df"], data["league_baseline"])

    if matchup_df is None or matchup_df.empty:
        # Probables/lineups not posted yet. Render a valid "check back" page
        # rather than failing -- this is a real pre-slate state, not a fetch error.
        log("No xwOBA matchups built (probables not posted) -> check-back page.")
        html = render_combined_html(pd.DataFrame(columns=["game_pk", "side"]),
                                    pd.DataFrame(), pitcher_rows_df, built_txt)
        with open(out_path, "w") as f:
            f.write(html)
        log(f"Wrote {out_path}")
        return 0

    log("Building platoon-OPS matchup ...")
    try:
        matchup_platoon_df, _detail, _lg = build_platoon_matchup(
            pitcher_rows_df, opp_hitters_df, data["people"],
            data["player_splits_hit"], data["player_splits_pit"],
            league_splits=data.get("player_splits_hit_league"),
            league_people=data.get("people_league_hitters"))
    except Exception as e:  # noqa: BLE001
        # Platoon lens is optional; degrade to xwOBA-only rather than fail.
        log(f"Platoon lens skipped: {e!r}")
        matchup_platoon_df = pd.DataFrame()

    # Dump the day's model outputs for the grading ledger (grade_leans.py
    # ingests these on the next CI step; data/ is committed back to the repo).
    os.makedirs("data", exist_ok=True)
    matchup_df.to_csv(f"data/leans_{SLATE_DATE}_xw.csv", index=False)
    if matchup_platoon_df is not None and not matchup_platoon_df.empty:
        matchup_platoon_df.to_csv(f"data/leans_{SLATE_DATE}_pl.csv", index=False)

    log("Rendering index.html ...")
    html = render_combined_html(matchup_df, matchup_platoon_df, pitcher_rows_df, built_txt)
    with open(out_path, "w") as f:
        f.write(html)
    log(f"Wrote {out_path} ({len(html):,} bytes, {len(matchup_df)} matchup rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
