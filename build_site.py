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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")   # display timezone (build time, game times)
UTC = timezone.utc
ROLLOVER_HOUR = 3          # hold "today" until ~3am ET so night games don't roll early
SPORT_ID = 1
LINEUP_SIZE = 9
REQUEST_DELAY = 0.25

# Column carrying a hitter's 1-9 batting-order slot through the pipeline.
BATTING_ORDER_COL = "batting_order"
# Expected plate appearances per game by batting-order slot. A lineup turns over
# from the top, so the leadoff man bats ~4.6 times while the 9-hole bats ~3.8 --
# roughly a 22% spread in in-game exposure across the order. We weight lineup
# composites by this slot expectation instead of by each hitter's season volume
# (BBE / split PA), which otherwise over-weights whoever has simply logged more
# playing time rather than who will actually see more pitches tonight. Values are
# league PA/game-by-order-position anchors (modern 9-inning norms).
LINEUP_SLOT_PA = {1: 4.61, 2: 4.50, 3: 4.39, 4: 4.29, 5: 4.18,
                  6: 4.08, 7: 3.97, 8: 3.87, 9: 3.76}
USE_SLOT_PA_WEIGHTS = True
CACHE_DIR = os.environ.get("CACHE_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "public")
DATA_DIR = os.environ.get("DATA_DIR", "data")            # grading ledger home
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
MODEL_TAG = os.environ.get("MODEL_TAG", "xw+plat_consol_v5")  # keep in sync with grade_leans.py
_RECORD_FAMILIES = {
    # v3 changed only ledger locking/identity; its prediction math is v2.
    "xw+plat_consol_v3": ("xw+plat_consol_v2", "xw+plat_consol_v3"),
    # v4 re-weights lineup composites by expected PA per batting-order slot
    # (was season BBE / split PA); that changes prediction math, so it starts a
    # fresh record family and never mixes with v2/v3 in the ledger or weight fit.
    # v5 adds empirical-Bayes xwOBA shrinkage (batters + starter) on top of v4;
    # another prediction-math change, so it starts its own family again.
}
RECORD_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "RECORD_TAGS", ",".join(_RECORD_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
    ).split(",") if t.strip()
)

STATCAST_SELECTIONS = ["pa", "k_percent", "bb_percent", "xwoba", "xba", "xslg",
                       "exit_velocity_avg", "launch_angle_avg", "hard_hit_percent"]

# Batted-ball direction/tendency rates with true league-wide anchors.
BATTED_RATE_COLS_FOR_BASELINE = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]

# Use all Savant-listed hitters for platoon priors instead of today's loaded lineups.
# Costs ~10-15 batched StatsAPI calls (~+5s) but removes slate-dependent shrinkage baselines.
FULL_LEAGUE_PLATOON_BASELINES = True
MIN_LEAGUE_BASELINE_PA = 1
RECENT_STARTS = 5


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
                "game_number": g.get("gameNumber"),
                "double_header": g.get("doubleHeader"),
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
                           "era": _parse_rate(st.get("era")),
                           "pa": int(st.get(pa_field) or st.get("plateAppearances") or 0),
                           "k": st.get("strikeOuts"), "bb": st.get("baseOnBalls")}
                    if tname == "statSplits" and code in ("vl", "vr"):
                        rec["L" if code == "vl" else "R"] = val
                    elif tname == "season" and code is None:
                        rec["overall"] = {"ops": val["ops"], "pa": val["pa"],
                                          "era": val["era"]}
            out[p["id"]] = rec
        time.sleep(REQUEST_DELAY)
    return out


def _innings_to_outs(value):
    """Convert MLB's baseball-notation innings (for example, 5.2) to outs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = str(value).strip()
    if not s:
        return 0
    try:
        whole, dot, frac = s.partition(".")
        extra = int(frac[:1] or 0) if dot else 0
        return max(0, int(whole) * 3 + min(max(extra, 0), 2))
    except (TypeError, ValueError):
        return 0


def load_recent_start_era(ids, limit=RECENT_STARTS):
    """Aggregate each probable pitcher's ERA over starts before this slate."""
    out = {}
    for pid in sorted({int(i) for i in ids if pd.notna(i)}):
        try:
            data = _get_json(
                f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
                {"stats": "gameLog", "group": "pitching", "season": SEASON,
                 "gameType": "R"})
        except Exception as e:  # noqa: BLE001
            log(f"  recent-start ERA unavailable for {pid}: {e!r}")
            continue

        starts = []
        for blk in data.get("stats", []):
            for sk in blk.get("splits", []):
                st = sk.get("stat", {}) or {}
                if int(st.get("gamesStarted") or 0) < 1:
                    continue
                # Never let the current slate's start leak into a pregame metric.
                game_date = str(sk.get("date") or
                                (sk.get("game", {}) or {}).get("gameDate") or "")[:10]
                if not game_date or game_date >= SLATE_DATE:
                    continue
                try:
                    er = float(st.get("earnedRuns") or 0)
                except (TypeError, ValueError):
                    continue
                game_pk = (sk.get("game", {}) or {}).get("gamePk") or 0
                starts.append((game_date, int(game_pk),
                               _innings_to_outs(st.get("inningsPitched")), er))

        recent = sorted(starts, key=lambda x: (x[0], x[1]), reverse=True)[:limit]
        outs = sum(x[2] for x in recent)
        earned_runs = sum(x[3] for x in recent)
        out[pid] = {
            "era": round(earned_runs * 27.0 / outs, 2) if outs > 0 else np.nan,
            "starts": len(recent),
        }
        time.sleep(REQUEST_DELAY)
    return out


def load_league_era():
    """Current-season MLB pitching ERA for the recent-start comparison."""
    endpoint = "https://statsapi.mlb.com/api/v1/stats"
    data = _get_json(endpoint, {
        "stats": "season", "group": "pitching", "season": SEASON,
        "sportIds": SPORT_ID, "gameType": "R", "playerPool": "ALL",
        "limit": 5000,
    })
    outs = earned_runs = 0
    for blk in data.get("stats", []):
        for sk in blk.get("splits", []):
            st = sk.get("stat", {}) or {}
            outs += _innings_to_outs(st.get("inningsPitched"))
            try:
                earned_runs += int(st.get("earnedRuns") or 0)
            except (TypeError, ValueError):
                pass
    return round(earned_runs * 27.0 / outs, 2) if outs > 0 else np.nan


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
    return {k: row[k] for k in ["game_pk", "game_date", "game_datetime_utc",
                                "matchup", "away_team", "home_team",
                                "away_probable_pitcher", "home_probable_pitcher",
                                "savant_preview_url"]}


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
                                 **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
                                 "player_id": pid, "bats": bio.get("bats"),
                                 "throws": bio.get("throws")})
            else:
                bbe = (pitcher_stat.get(pid, {}) or {}).get("BBE")
                bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": "P",
                                **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                "player_id": pid, "bats": bio.get("bats"),
                                "throws": bio.get("throws")})

        def hitter_rows(lu, tidx, table):
            for slot, pid in enumerate(lu, start=1):
                pid = int(pid)
                bio = people.get(pid, {})
                nm = bio.get("name") or f"batter_{pid}"
                pos = bio.get("pos") or "DH"
                if pos == "P":
                    pos = "DH"
                src = (batter_stat if table == "stat" else batter_bb).get(pid, {})
                base = {**meta, "table_index": tidx, "Name": nm,
                        BATTING_ORDER_COL: slot}
                if table == "stat":
                    pit_rows.append({**base, "table_type": "pitchers", "Pos.": pos,
                                     **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
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

    META = ["game_pk", "game_date", "game_datetime_utc", "matchup", "away_team", "home_team",
            "away_probable_pitcher", "home_probable_pitcher", "savant_preview_url",
            "table_type", "table_index", "Name"]
    pdf = pd.DataFrame(pit_rows)
    if not pdf.empty:
        pdf = pdf[META + ["Pos.", BATTING_ORDER_COL] + STAT_COLS + ["PA", "player_id", "bats", "throws"]]
    bdf = pd.DataFrame(bb_rows)
    if not bdf.empty:
        bdf = bdf[META + ["Pos.", BATTING_ORDER_COL] + BB_COLS + ["BBE", "player_id", "bats", "throws"]]
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
    pitcher_stat, pitcher_bb, pitcher_cust = load_stat_lookups("pitcher")
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

    # Method-of-moments xwOBA shrinkage constants, estimated once per build over
    # the full custom leaderboards (xwoba + pa columns). Stashed on
    # league_baseline so build_xwoba_matchup can read them without a new arg.
    if USE_XWOBA_SHRINK:
        prior = league_baseline.get("xwOBA")

        def _xw_pa_pairs(cust):
            if cust is None or "xwoba" not in cust.columns or "pa" not in cust.columns:
                return []
            xw = pd.to_numeric(cust["xwoba"], errors="coerce")
            pa = pd.to_numeric(cust["pa"], errors="coerce")
            return list(zip(xw.tolist(), pa.tolist()))

        k_bat, note_b = estimate_shrink_k(_xw_pa_pairs(batter_cust), prior, K_BAT_DEFAULT, K_BAT_BAND)
        k_pit, note_p = estimate_shrink_k(_xw_pa_pairs(pitcher_cust), prior, K_PIT_DEFAULT, K_PIT_BAND)
        league_baseline["_xwOBA_K_bat"] = k_bat
        league_baseline["_xwOBA_K_pit"] = k_pit
        log(f"  xwOBA shrink: prior={prior} | K_bat={k_bat:.0f} ({note_b}) "
            f"| K_pit={k_pit:.0f} ({note_p})")

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

    log(f"Loading probable-pitcher ERA over the last {RECENT_STARTS} starts ...")
    recent_start_era = load_recent_start_era(prob_ids)
    try:
        league_baseline["ERA"] = load_league_era()
    except Exception as e:  # noqa: BLE001
        log(f"  league ERA unavailable: {e!r}")
        league_baseline["ERA"] = np.nan
    log(f"  recent ERA: {len(recent_start_era)} pitchers | league {league_baseline['ERA']}")

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
    if not pitchers_df.empty:
        def _recent_value(r, key, default=np.nan):
            if r.get("Pos.") != "P" or pd.isna(r.get("player_id")):
                return default
            return recent_start_era.get(int(r["player_id"]), {}).get(key, default)

        pitchers_df["ERA_L5"] = pitchers_df.apply(lambda r: _recent_value(r, "era"), axis=1)
        pitchers_df["ERA_L5_GS"] = pitchers_df.apply(
            lambda r: _recent_value(r, "starts", 0), axis=1)
        pitchers_df["ERA_SEASON"] = pitchers_df.apply(
            lambda r: ((player_splits_pit.get(int(r["player_id"]), {}).get("overall", {})
                        or {}).get("era", np.nan))
            if r.get("Pos.") == "P" and pd.notna(r.get("player_id")) else np.nan,
            axis=1)

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

# --- xwOBA empirical-Bayes shrinkage (v5) ----------------------------------
# Season xwOBA (each hitter) and xwOBA-allowed (the starter) are regressed
# toward the league xwOBA baseline by sample size before they drive the lean:
#     x* = (n*x + K*prior) / (n + K)
# so a small-sample bat or a starter with few batters faced is pulled toward
# league-average rather than taken at face value. Both sides share one prior
# (the league xwOBA baseline). K is estimated by method of moments per player
# pool each build: sampling noise scales as 1/n, so the gap between the
# unweighted and the PA-weighted dispersion around the mean identifies the
# within-PA (sigma^2) and between-player (tau^2) variance components, and
# K = sigma^2 / tau^2 -- no fixed per-PA variance constant. The estimate is
# clamped to a plausible PA band with a fixed fallback and logged each run, so
# a degenerate pool cannot silently distort leans. Shrinkage touches only
# xwOBA (the lean stat); the other columns and the raw per-hitter card values
# are untouched.
USE_XWOBA_SHRINK = True
XWOBA_SHRINK_COL = "xwOBA"
K_BAT_DEFAULT, K_BAT_BAND = 130.0, (40.0, 500.0)
K_PIT_DEFAULT, K_PIT_BAND = 600.0, (120.0, 1500.0)
MIN_K_POOL_PA = 10        # drop 1-PA noise from the K-estimation pool
MIN_K_POOL_N = 20         # need a real population to split the variance


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


def slot_pa_weights(order):
    """Map a batting-order (1-9) series to expected PA/game weights.

    Returns a positionally-reindexed Series (aligns with the equally
    reset-indexed value vectors used by wmean/_wmean); slots outside 1-9 map to
    NaN so wmean falls back to an equal mean for them.
    """
    o = pd.to_numeric(pd.Series(order).reset_index(drop=True), errors="coerce")
    return o.map(LINEUP_SLOT_PA)


def lineup_weight(g, fallback_col):
    """Slot-PA weights for a lineup group, falling back to season volume.

    Uses expected PA-per-order-slot when enabled and a usable batting_order is
    present; otherwise returns the group's season-volume weight column so
    behavior is unchanged wherever the order is unavailable.
    """
    if USE_SLOT_PA_WEIGHTS and BATTING_ORDER_COL in getattr(g, "columns", []):
        w = slot_pa_weights(g[BATTING_ORDER_COL])
        if w.notna().any():
            return w
    return g.get(fallback_col) if hasattr(g, "get") else None


def shrink_xwoba(x, n, prior, k):
    """Regress observed rate(s) toward prior by sample size: (n*x + k*prior)/(n+k).

    Series in / positionally-reindexed Series out (aligns with wmean's reset
    index). A missing rate or n<=0 collapses to the prior. Pass-through when
    shrinkage is disabled or k/prior are unusable.
    """
    xs = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
    if not USE_XWOBA_SHRINK or k is None or not np.isfinite(k) or prior is None or not np.isfinite(prior):
        return xs
    ns = pd.to_numeric(pd.Series(n).reset_index(drop=True), errors="coerce").fillna(0.0).clip(lower=0.0)
    return (ns * xs.fillna(prior) + k * prior) / (ns + k)


def _shrink_one(x, n, prior, k):
    """Scalar form of shrink_xwoba for a single pitcher row."""
    if not USE_XWOBA_SHRINK or k is None or not np.isfinite(k) or prior is None or pd.isna(prior):
        return x
    n = float(n) if pd.notna(n) else 0.0
    x = float(prior) if pd.isna(x) else float(x)
    return (n * x + k * prior) / (n + k)


def estimate_shrink_k(pairs, prior, default, band):
    """Method-of-moments shrinkage constant K = sigma^2 / tau^2 over (rate, n) pairs.

    Sampling variance scales as 1/n, so the unweighted dispersion (which lets
    low-n players count fully) exceeds the n-weighted dispersion by exactly the
    sampling component; that gap identifies sigma^2 (within-PA) and leaves
    tau^2 (between-player talent). Returns (k, note); falls back to `default`
    and clamps to `band` when the pool is thin or a component goes non-positive.
    """
    lo, hi = band
    if prior is None or not np.isfinite(prior):
        return default, "fallback:no_prior"
    x = np.array([p[0] for p in pairs], dtype=float)
    n = np.array([p[1] for p in pairs], dtype=float)
    m = np.isfinite(x) & np.isfinite(n) & (n >= MIN_K_POOL_PA)
    x, n = x[m], n[m]
    if len(x) < MIN_K_POOL_N or n.sum() <= 0:
        return default, "fallback:thin_pool"
    d2 = (x - prior) ** 2
    Vw = float(np.average(d2, weights=n))                 # PA-weighted dispersion
    Vu = float(d2.mean())                                 # unweighted dispersion
    coef = float((1.0 / n).mean() - 1.0 / n.mean())       # >= 0 by AM-HM; 0 iff all n equal
    if coef <= 0:
        return default, "fallback:equal_n"
    sig2 = (Vu - Vw) / coef
    tau2 = Vw - sig2 / n.mean()
    if sig2 <= 0 or tau2 <= 0:
        return default, "fallback:neg_var_component"
    k = sig2 / tau2
    if not np.isfinite(k) or k <= 0:
        return default, "fallback:bad_k"
    return float(min(max(k, lo), hi)), ("ok" if lo <= k <= hi else f"clamped_from_{k:.0f}")


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


def aggregate_lineup(H, rate_cols, weighted=True, shrink_prior=None, shrink_k=None):
    if H is None or H.empty:
        return pd.DataFrame()
    out = []
    for (gpk, fp), g in H.groupby(["game_pk", "faced_pitcher"], sort=False):
        rec = {"game_pk": gpk, "faced_pitcher": fp,
               "pitcher_side": g["pitcher_side"].iloc[0],
               "batting_side": g["batting_side"].iloc[0],
               "n_opp_hitters": len(g)}
        # Weight the lineup composite by expected PA per batting-order slot
        # (top of the order sees more pitches) rather than by season BBE volume.
        w = lineup_weight(g, WEIGHT_COL)
        for c in rate_cols:
            if c not in g.columns:
                continue
            # Regress each hitter's xwOBA toward the league prior by their PA
            # BEFORE compositing, so a small-sample bat can't swing the lineup
            # number; other columns pass through raw.
            if c == XWOBA_SHRINK_COL and shrink_prior is not None and "PA" in g.columns:
                vals = shrink_xwoba(g[c], g["PA"], shrink_prior, shrink_k)
            else:
                vals = pd.to_numeric(pd.Series(g[c]).reset_index(drop=True), errors="coerce")
            rec[f"opp_{c}_mean"] = round(float(vals.mean(skipna=True)), 3) if vals.notna().any() else np.nan
            rec[f"opp_{c}_wmean"] = round(wmean(vals, w), 3)
            rec[f"opp_{c}"] = rec[f"opp_{c}_wmean"] if weighted else rec[f"opp_{c}_mean"]
        out.append(rec)
    return pd.DataFrame(out)


def build_matchup(P, agg, rate_cols, league_baseline, shrink_prior=None, shrink_k=None):
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
               "game_datetime_utc": pr.get("game_datetime_utc"),
               "matchup": pr.get("matchup"), "side": side, "pitcher": a["faced_pitcher"],
               "opp_team": opp_team, "n_opp": int(a["n_opp_hitters"])}
        for c in rate_cols:
            pv = pd.to_numeric(pr.get(c), errors="coerce")
            # Regress the starter's xwOBA-allowed toward the league prior by
            # batters faced (PA), so a short-sample starter isn't taken at face
            # value. This shrunk value drives the lean and the SP card display.
            if c == XWOBA_SHRINK_COL and shrink_prior is not None:
                pv = _shrink_one(pv, pd.to_numeric(pr.get("PA"), errors="coerce"),
                                 shrink_prior, shrink_k)
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
    prior = league_baseline.get(XWOBA_SHRINK_COL) if USE_XWOBA_SHRINK else None
    k_bat = league_baseline.get("_xwOBA_K_bat")
    k_pit = league_baseline.get("_xwOBA_K_pit")
    pitcher_rows_df, opp_hitters_df = segment_pitcher_blocks(pitchers_df, STATCAST_RATE_COLS)
    opp_lineup_agg_df = aggregate_lineup(opp_hitters_df, STATCAST_RATE_COLS, weighted=USE_WEIGHTED,
                                         shrink_prior=prior, shrink_k=k_bat)
    matchup_df = build_matchup(pitcher_rows_df, opp_lineup_agg_df, STATCAST_RATE_COLS, league_baseline,
                               shrink_prior=prior, shrink_k=k_pit)
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
            BATTING_ORDER_COL: h.get(BATTING_ORDER_COL),
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
        # All aggregates are lineup-composition weighted by expected PA per
        # batting-order slot (top of the order carries more in-game exposure).
        # Fall back to season split PA (clip 0-PA splits to 1 so prior-driven
        # bats still count as lineup exposure, incl. SP OPS display) wherever the
        # batting order is unavailable.
        lineup_w = lineup_weight(g, None)
        if lineup_w is None or not pd.Series(lineup_w).notna().any():
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
# PREGAME MARKET (display-only) -- best-effort DK odds via ESPN
# Same endpoints/join as market_backfill.py (scoreboard -> core
# /odds, provider 100), but *pregame*: current + open moneylines
# and the total, devigged home implied %. Strictly best-effort:
# any failure logs and omits -- an odds outage never fails the
# build, and missing values render as em-dashes, never defaults.
# ============================================================
_ESPN2SA = {"CHW": "CWS", "ARI": "AZ", "OAK": "ATH"}  # ESPN -> StatsAPI abbrev


def _amer_ml(x):
    if isinstance(x, dict):
        x = x.get("american")
    try:
        return int(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _imp_ml(ml):
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def _parse_espn_dt(s):
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except (TypeError, ValueError):
            continue
    return None


def fetch_pregame_odds(slate_df):
    """game_pk -> {away_ml, home_ml, open_away_ml, open_home_ml, total, p_home}."""
    out = {}
    try:
        ds = SLATE_DATE.replace("-", "")
        sb = _get_json("https://site.api.espn.com/apis/site/v2/sports/baseball/"
                       f"mlb/scoreboard?dates={ds}", tries=2)
    except Exception as e:  # noqa: BLE001
        log(f"pregame odds: scoreboard fetch failed ({e!r}) -> strip renders em-dashes")
        return out
    evs = {}
    for ev in sb.get("events", []):
        try:
            comp = ev["competitions"][0]
            t = {c["homeAway"]: _ESPN2SA.get(c["team"]["abbreviation"],
                                             c["team"]["abbreviation"])
                 for c in comp["competitors"]}
            tid = {str(c["team"]["id"]): c["homeAway"] for c in comp["competitors"]}
            evs.setdefault((t["away"], t["home"]), []).append(
                dict(eid=ev["id"], start=ev.get("date"), tid=tid))
        except Exception:  # noqa: BLE001
            continue

    def _team_odds(dk, which):
        td = dk.get(which) or {}
        cur = _amer_ml(((td.get("current") or {}).get("moneyLine")))
        if cur is None:
            cur = _amer_ml(td.get("moneyLine"))
        opn = _amer_ml(((td.get("open") or {}).get("moneyLine")))
        return cur, opn

    for _, g in slate_df.iterrows():
        cands = evs.get((g.get("away_abbrev"), g.get("home_abbrev")), [])
        if not cands:
            continue
        pick = cands[0]
        if len(cands) > 1:  # doubleheader: nearest scheduled start
            gt = _parse_espn_dt(str(g.get("game_datetime_utc") or ""))
            if gt is not None:
                def _gap(e):
                    et_ = _parse_espn_dt(str(e.get("start") or ""))
                    return abs((et_ - gt).total_seconds()) if et_ else 1e12
                pick = min(cands, key=_gap)
        try:
            odds = _get_json("https://sports.core.api.espn.com/v2/sports/baseball/"
                             f"leagues/mlb/events/{pick['eid']}/competitions/"
                             f"{pick['eid']}/odds", tries=2)
        except Exception as e:  # noqa: BLE001
            log(f"pregame odds: game_pk={g['game_pk']} odds fetch failed ({e!r})")
            continue
        dk = next((i for i in odds.get("items", [])
                   if str((i.get("provider") or {}).get("id")) == "100"), None)
        if dk is None:
            continue
        h_cur, h_opn = _team_odds(dk, "homeTeamOdds")
        a_cur, a_opn = _team_odds(dk, "awayTeamOdds")
        try:
            total = float(dk.get("overUnder")) if dk.get("overUnder") is not None else None
        except (TypeError, ValueError):
            total = None
        p_home = None
        if h_cur is not None and a_cur is not None:
            ih, ia = _imp_ml(h_cur), _imp_ml(a_cur)
            p_home = ih / (ih + ia) if (ih + ia) > 0 else None
        # F5 (1st 5 innings) DK ML from the same event's propBets child, using
        # the validated parser from market_backfill; pregame `value` is the
        # current F5 price, `open` its opener. Best-effort: any failure omits
        # F5 and the strip renders em-dashes. Devig is CONDITIONAL on a decided
        # half (DK F5 ties push).
        f5a = f5h = f5a_opn = f5h_opn = f5_p_home = None
        try:
            from market_backfill import _espn_f5
            f5 = _espn_f5(pick["eid"], pick.get("tid") or {})
            if f5:
                f5a, f5h = f5["f5_close_away_ml"], f5["f5_close_home_ml"]
                f5a_opn, f5h_opn = f5["f5_open_away_ml"], f5["f5_open_home_ml"]
                if f5a is not None and f5h is not None:
                    ih5, ia5 = _imp_ml(f5h), _imp_ml(f5a)
                    f5_p_home = ih5 / (ih5 + ia5) if (ih5 + ia5) > 0 else None
        except Exception as e:  # noqa: BLE001
            log(f"pregame odds: game_pk={g['game_pk']} F5 fetch failed ({e!r})")
        out[g["game_pk"]] = dict(away_ml=a_cur, home_ml=h_cur,
                                 open_away_ml=a_opn, open_home_ml=h_opn,
                                 total=total, p_home=p_home,
                                 f5_away_ml=f5a, f5_home_ml=f5h,
                                 f5_open_away_ml=f5a_opn, f5_open_home_ml=f5h_opn,
                                 f5_p_home=f5_p_home)
        time.sleep(0.15)
    log(f"pregame odds attached: {len(out)}/{len(slate_df)} games")
    return out


def fetch_last10_records(slate_df=None):
    """team_id -> 'W-L' over each team's last 10 games (StatsAPI standings).

    Display-only and strictly best-effort: any failure logs and returns an
    empty map so the card header simply omits the record rather than failing
    the build. `slate_df` is accepted for signature symmetry with the odds
    fetch; standings cover all teams regardless of the slate.
    """
    out = {}
    try:
        js = _get_json("https://statsapi.mlb.com/api/v1/standings",
                       {"leagueId": "103,104", "season": SEASON,
                        "date": SLATE_DATE, "standingsTypes": "regularSeason"},
                       tries=2)
    except Exception as e:  # noqa: BLE001
        log(f"last-10 records: standings fetch failed ({e!r}) -> header omits")
        return out
    for grp in js.get("records", []):
        for tr in grp.get("teamRecords", []):
            tid = (tr.get("team") or {}).get("id")
            if tid is None:
                continue
            l10 = next((s for s in (tr.get("records") or {}).get("splitRecords", [])
                        if s.get("type") == "lastTen"), None)
            if l10 and l10.get("wins") is not None and l10.get("losses") is not None:
                try:
                    out[int(tid)] = f"{int(l10['wins'])}-{int(l10['losses'])}"
                except (TypeError, ValueError):
                    continue
    log(f"last-10 records attached: {len(out)} teams")
    return out


# ============================================================
# CELL 6 -- CARD RENDER (per-hitter lineup layout, emits HTML)
# ============================================================
ABBR = {
    "Arizona Diamondbacks": "AZ", "Athletics": "ATH", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

HITTER_EDGE_DOMAIN = 0.100  # shared per-hitter edge-bar axis (|mx_ops - lg|)

# ---------- heatmap tints (casual-UI layer; numbers unchanged) ----------
HEAT_ALPHA_MAX = 0.30
HEAT_DOMAINS = {"xwOBA_sp": 0.035, "K-BB%": 7.0,
                "OPS": 0.080, "ERA": 1.50, "xwOBA_bat": 0.045}   # |dev| at full saturation


def heat_style(val, lg, domain, hi="warm"):
    """Cell tint vs league. hi = token when value is ABOVE league:
    'warm' (offense-favorable) or 'cool' (pitcher-favorable)."""
    v, L = _f(val), _f(lg)
    if v is None or L is None:
        return ""
    a = clamp(abs(v - L) / domain, 0.0, 1.0) * HEAT_ALPHA_MAX
    if a < 0.04:          # dead zone: ~league-average stays untinted
        return ""
    tok = hi if v > L else ("cool" if hi == "warm" else "warm")
    return f"background:rgba(var(--{tok}),{a:.2f})"


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def edge_color(edge, ksign=1):
    if edge is None:
        return "var(--faint)"
    return "rgba(var(--warm),1)" if edge * ksign >= 0 else "rgba(var(--cool),1)"


def f3(v):
    return "—" if v is None else f"{v:.3f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.3f}"


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def sgn3(v):
    return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v):.3f}"


def _abbr(name):
    return ABBR.get(name, str(name or "")[:3].upper())


def _fmt_ml(v):
    if v is None:
        return "—"
    return f"+{v}" if v > 0 else str(v)


def _fmt_pt_clock(dt):
    return dt.strftime("%I:%M %p").lstrip("0") + " PT"


def _game_time_pt(iso_utc):
    dt = _parse_espn_dt(str(iso_utc or ""))
    if dt is None:
        return ""
    return _fmt_pt_clock(dt.replace(tzinfo=UTC).astimezone(PT))


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


def _pl_lookup(pl_df):
    out = {}
    if pl_df is None or getattr(pl_df, "empty", True):
        return out
    for _, r in pl_df.iterrows():
        out[(r["game_pk"], r["side"])] = r
    return out


def _hitters_for(opp_hitters_df, detail_df, gpk, fp, lg_ops):
    """Batting-order rows for the lineup this SP faces. Row order in
    opp_hitters_df IS lineup order -- never sort. Detail (platoon) join is by
    (game_pk, faced_pitcher, batter name); a hitter with no vs-hand data keeps
    the xwOBA cell and renders em-dashes for the platoon columns."""
    rows = []
    if opp_hitters_df is None or getattr(opp_hitters_df, "empty", True):
        return rows
    H = opp_hitters_df[(opp_hitters_df["game_pk"] == gpk)
                       & (opp_hitters_df["faced_pitcher"] == fp)]
    D = {}
    if detail_df is not None and not getattr(detail_df, "empty", True):
        dd = detail_df[(detail_df["game_pk"] == gpk)
                       & (detail_df["faced_pitcher"] == fp)]
        for _, r in dd.iterrows():
            D[r["batter"]] = r
    for _, r in H.iterrows():
        d = D.get(r["Name"])
        mx = _f(d["mx_ops"]) if d is not None else None
        edge = (mx - lg_ops) if (mx is not None and lg_ops is not None
                                 and pd.notna(lg_ops)) else None
        rows.append(dict(
            name=r["Name"], pos=str(r.get("Pos.") or ""),
            bats=(str(r.get("bats") or ""))[:1].upper(),
            xw=_f(r.get("xwOBA")),
            adv=bool(d["platoon_adv"]) if d is not None else False,
            ops=_f(d["ops_vs_hand"]) if d is not None else None,
            pa=int(d["split_pa"] or 0) if d is not None else 0,
            low=bool(d["low_sample"]) if d is not None else False,
            mx=mx, edge=edge,
        ))
    return rows


def _hitter_row_html(i, hr, lg_xw=None):
    nm = _esc(hr["name"])
    b = f"<span class='b'>{hr['bats']}</span>" if hr["bats"] else ""
    adv = "<span class='adv' title='platoon advantage vs this SP'>◆</span>" if hr["adv"] else ""
    if hr["ops"] is None:
        ops_cell = "<td class='na'>—</td>"
    else:
        if hr["pa"] > 0:
            src = f"<span class='pa'>({hr['pa']})</span>"
            low = " <span class='flag mute'>low PA</span>" if hr["low"] else ""
        else:
            src, low = "<span class='pa'>(prior)</span>", ""
        ops_cell = f"<td>{f3(hr['ops'])} {src}{low}</td>"
    if hr["edge"] is None:
        bar = "<div class='eb'></div>"
    else:
        w = clamp(abs(hr["edge"]) / HITTER_EDGE_DOMAIN, 0, 1) * 50
        cls = "w" if hr["edge"] >= 0 else "c"
        bar = (f"<div class='eb' title='xOPS edge vs league {sgn3(hr['edge'])}'>"
               f"<i class='{cls}' style='width:{w:.0f}%'></i></div>")
    lowcls = " class='low'" if (hr["ops"] is not None and (hr["pa"] == 0 or hr["low"])) else ""
    xw_st = heat_style(hr["xw"], lg_xw, HEAT_DOMAINS["xwOBA_bat"])
    xw_td = f"<td style='{xw_st}'>{f3(hr['xw'])}</td>" if xw_st else f"<td>{f3(hr['xw'])}</td>"
    return (f"<tr{lowcls}><td class='ord'>{i}</td>"
            f"<td class='n'>{nm}{b}{adv}</td><td class='pos'>{_esc(hr['pos'])}</td>"
            f"{xw_td}{ops_cell}"
            f"<td class='bar'>{bar}</td></tr>")


def _lineup_details(side_d, lg_xw=None):
    st = side_d["lu_status"]
    st_lab = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, st or "—")
    st_cls = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, "projected")
    parts = []
    if side_d["opp_xw"] is not None:
        parts.append(f"xwOBA {f3(side_d['opp_xw'])}")
    if side_d["pl_mx"] is not None:
        parts.append(f"xOPS {f3(side_d['pl_mx'])}")
    summ = " · ".join(parts) if parts else ""
    hand = side_d["t"] if side_d["t"] in ("L", "R") else "?"
    head = (f"<tr><th></th><th class='n'>Hitter</th><th></th><th>xwOBA</th>"
            f"<th>vs-{hand} OPS</th><th>edge</th></tr>")
    body = "".join(_hitter_row_html(i + 1, hr, lg_xw) for i, hr in enumerate(side_d["hitters"]))
    if not body:
        body = "<tr><td class='na' colspan='6'>lineup unavailable</td></tr>"
    return (
        "<details class='lineup' open>"
        "<summary><span class='chev'>▶</span>"
        f"<span class='tl'>{_esc(side_d['opp_abbr'])} lineup</span>"
        f"<span class='st {st_cls}'>{st_lab}</span>"
        f"<span class='lw'>{summ}</span></summary>"
        f"<div class='lu-scroll'><table class='lu'>{head}{body}</table></div></details>")


def _sp_stat_cell(lab, val, fmt, sub=None, heat=""):
    s = f"<div class='s'>{sub}</div>" if sub else ""
    st = f" style='{heat}'" if heat else ""
    return (f"<div class='stat'{st}><div class='l'>{lab}</div>"
            f"<div class='v'>{fmt(val)}</div>{s}</div>")


def _side_html(label, d, league_baseline):
    badge = f"<span class='hand'>{d['t']}HP</span>" if d["t"] in ("L", "R") else ""
    thin = (f"<span class='flag warn'>thin hand split {d['pl_fl']['thin']} BF</span>"
            if "thin" in d["pl_fl"] else "")
    comp = (f"{d['R']}R/{d['L']}L" + (f"/{d['S']}S" if d["S"] else "")) if d["has_pl"] else "—"
    padv = f" · {d['padv']} plt-adv" if d["has_pl"] else ""
    lb = league_baseline or {}
    # lg K%/BB% come from the PA-weighted *batter* leaderboard; league K% and
    # BB% are symmetric (every batter K/BB is a pitcher K/BB), so they are
    # valid pitcher references and lg K-BB% = lg K% - lg BB%.
    lg = {k: _f(lb.get(k)) for k in ("xwOBA", "K%", "BB%", "OPS", "ERA")}
    kbb = (d["pit_k"] - d["pit_bb"]) if (d["pit_k"] is not None
                                         and d["pit_bb"] is not None) else None
    lg_kbb = (lg["K%"] - lg["BB%"]) if (lg["K%"] is not None
                                        and lg["BB%"] is not None) else None
    era_sub = []
    if d["era_season"] is not None:
        era_sub.append(f"season {f2(d['era_season'])}")
    gs = int(d.get("era_l5_gs") or 0)
    era_lab = f"ERA · L{gs}" if 0 < gs < RECENT_STARTS else f"ERA · L{RECENT_STARTS}"
    stats = (
        _sp_stat_cell("xwOBA agn", d["pit_xw"], f3,
                      f"lg {f3(lg['xwOBA'])}" if lg["xwOBA"] is not None else None,
                      heat=heat_style(d["pit_xw"], lg["xwOBA"], HEAT_DOMAINS["xwOBA_sp"]))
        + _sp_stat_cell("K-BB%", kbb, f1,
                        f"lg {f1(lg_kbb)}" if lg_kbb is not None else None,
                        heat=heat_style(kbb, lg_kbb, HEAT_DOMAINS["K-BB%"], hi="cool"))
        + _sp_stat_cell("OPS alwd*", d["pl_sp"], f3,
                        f"raw {f3(d['pl_sp_raw'])}" if d["pl_sp_raw"] is not None else None,
                        heat=heat_style(d["pl_sp"], lg["OPS"], HEAT_DOMAINS["OPS"]))
        + _sp_stat_cell(era_lab, d["era_l5"], f2,
                        " · ".join(era_sub) or None,
                        heat=heat_style(d["era_l5"], lg["ERA"], HEAT_DOMAINS["ERA"])))
    pl_bits = f3(None) if not d["has_pl"] else sgn3(d["pl_edge"])
    pl_col = edge_color(d["pl_edge"]) if (d["has_pl"] and d["pl_reliable"]) else "var(--faint)"
    pl_flag = "" if (not d["has_pl"] or d["pl_reliable"]) else " <span class='flag mute'>prior-driven</span>"
    return (
        f"<section class='side'>"
        f"<div class='sp'><div class='who'><span class='nm'>{_esc(d['p'])}</span>{badge}{thin}</div>"
        f"<div class='role'>{label} SP · faces {_esc(d['opp_abbr'])} lineup · {comp}{padv}</div>"
        f"<div class='spstats'>{stats}</div></div>"
        f"<div class='agg'>"
        f"<div class='e' style='color:{edge_color(d['xw_edge'])}'>"
        f"<span>xw edge (drives lean)</span>{sgn3(d['xw_edge'])}</div>"
        f"<div class='e' style='color:{pl_col}'><span>xOPS edge</span>{pl_bits}{pl_flag}</div>"
        f"</div>"
        f"{_lineup_details(d, lg['xwOBA'])}"
        f"</section>")


def _consensus_html(away_abbr, home_abbr, a, h):
    xw_home, xw_away = h["xw_edge"], a["xw_edge"]
    have_xw = xw_home is not None and xw_away is not None
    xw_fav = home_abbr if (have_xw and a["xw_edge"] >= h["xw_edge"]) else away_abbr
    # NOTE: a = away SP row; a's xw_edge is the HOME offense's edge, and vice
    # versa -- identical convention to the previous _consensus().
    xw_d = abs(a["xw_edge"] - h["xw_edge"]) if have_xw else None
    pl_home = a["pl_edge"] if (a["has_pl"] and a["pl_reliable"]) else None
    pl_away = h["pl_edge"] if (h["has_pl"] and h["pl_reliable"]) else None
    if pl_home is not None and pl_away is not None:
        pl_fav = home_abbr if pl_home >= pl_away else away_abbr
        pl_d = abs(pl_home - pl_away)
        tag, tcl = ("AGREE", "agree") if pl_fav == xw_fav else ("DIVERGE", "diverge")
        pl_txt = f"OPS → <b>{pl_fav}</b> Δ{pl_d:.3f}"
    else:
        tag, tcl = "n/a", "na"
        pl_txt = "OPS → <span class='muted'>unreliable / no split</span>"
    xw_txt = (f"xwOBA → <b>{xw_fav}</b> Δ{xw_d:.3f}" if xw_d is not None
              else "xwOBA → <span class='muted'>—</span>")
    return (f"<div class='consensus'>{xw_txt}<span class='dot'>·</span>{pl_txt}"
            f"<span class='ctag {tcl}'>{tag}</span></div>")


def _market_html(o, away_abbr, home_abbr, built_short):
    o = o or {}
    def _mlcell(prefix, lab, cur, opn):
        sub = f" <span class='mv'>← {_fmt_ml(opn)} open</span>" if opn is not None else ""
        return (f"<div class='mcell'><div class='l'>{prefix} · {lab}</div>"
                f"<div class='v'>{_fmt_ml(cur)}{sub}</div></div>")
    tot = f"o/u {o['total']:g}" if o.get("total") is not None else "—"
    ph = f"{o['p_home'] * 100:.1f}%" if o.get("p_home") is not None else "—"
    # F5 devig is conditional on a decided half (DK F5 ties push).
    ph5 = f"{o['f5_p_home'] * 100:.1f}%" if o.get("f5_p_home") is not None else "—"
    return (
        "<div class='market'>"
        + _mlcell("DK ML", away_abbr, o.get("away_ml"), o.get("open_away_ml"))
        + _mlcell("DK ML", home_abbr, o.get("home_ml"), o.get("open_home_ml"))
        + _mlcell("DK F5", away_abbr, o.get("f5_away_ml"), o.get("f5_open_away_ml"))
        + _mlcell("DK F5", home_abbr, o.get("f5_home_ml"), o.get("f5_open_home_ml"))
        + f"<div class='mcell'><div class='l'>Total</div><div class='v'>{tot}</div></div>"
        + f"<div class='mcell'><div class='l'>Implied {home_abbr} (devig)</div><div class='v'>{ph}</div></div>"
        + f"<div class='mcell'><div class='l'>F5 impl {home_abbr} (devig · cond.)</div><div class='v'>{ph5}</div></div>"
        + f"<div class='mcell note'><div class='l'>Market</div><div class='v'>as of build {built_short}</div></div>"
        + "</div>")


def _l10_span(rec):
    """Small 'W-L' chip (last-10 record) appended to a team abbr; '' if unknown."""
    return f"<span class='l10' title='last 10 games'>{_esc(rec)}</span>" if rec else ""


def cmb_card(g, built_short):
    if g.get("unavailable"):
        game_no = f" <span class='game-no'>{_esc(g['game_label'])}</span>" if g.get("game_label") else ""
        when = " · ".join(x for x in (g.get("time_pt"), g.get("venue")) if x)
        probables = " vs ".join(
            "TBD" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
            for x in (g.get("away_probable"), g.get("home_probable")))
        return (
            "<article class='card unavailable'>"
            "<div class='gamehead'>"
            f"<span class='teams'>{g['away_abbr']}{_l10_span(g.get('away_l10'))} "
            f"<span class='at'>@</span> {g['home_abbr']}{_l10_span(g.get('home_l10'))}{game_no}</span>"
            + (f"<span class='when'>{_esc(when)}</span>" if when else "")
            + "</div>"
            "<div class='card-note'><b>Awaiting paired probable pitchers</b>"
            f"<span>{_esc(probables)} · matchup lean will appear after both starters are listed.</span></div>"
            "</article>")

    a, h = g["away"], g["home"]
    away_abbr, home_abbr = g["away_abbr"], g["home_abbr"]
    # a = away SP -> his xw_edge is the HOME offense edge (same as before).
    home_off = a["xw_edge"] if a["xw_edge"] is not None else 0.0
    away_off = h["xw_edge"] if h["xw_edge"] is not None else 0.0
    delta = abs(home_off - away_off)
    fav = home_abbr if home_off >= away_off else away_abbr
    when = " · ".join(x for x in (g.get("time_pt"), g.get("venue")) if x)
    game_no = f" <span class='game-no'>{_esc(g['game_label'])}</span>" if g.get("game_label") else ""
    return (
        "<article class='card'>"
        "<div class='gamehead'>"
        f"<span class='teams'>{away_abbr}{_l10_span(g.get('away_l10'))} "
        f"<span class='at'>@</span> {home_abbr}{_l10_span(g.get('home_l10'))}{game_no}</span>"
        + (f"<span class='when'>{_esc(when)}</span>" if when else "")
        + f"<span class='lean'><span class='lk'>lean</span><span class='lt'>{fav}</span>"
        f"<span class='ld'>Δxw {delta:.3f}</span></span>"
        f"{_consensus_html(away_abbr, home_abbr, a, h)}"
        "</div>"
        f"{_market_html(g.get('odds'), away_abbr, home_abbr, built_short)}"
        f"<div class='sides'>{_side_html('AWAY', a, g['league_baseline'])}"
        f"{_side_html('HOME', h, g['league_baseline'])}</div>"
        "</article>")


def _game_order_key(game):
    """Chronological slate order with deterministic doubleheader tie-breaks."""
    raw_start = game.get("game_datetime_utc")
    try:
        start = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        start = start.astimezone(UTC)
    except (TypeError, ValueError):
        start = datetime.max.replace(tzinfo=UTC)
    game_number = pd.to_numeric(game.get("game_number"), errors="coerce")
    game_pk = pd.to_numeric(game.get("game_pk"), errors="coerce")
    return (
        start,
        int(game_number) if pd.notna(game_number) else 99,
        int(game_pk) if pd.notna(game_pk) else 2**63 - 1,
    )


def build_combined(games, built_short):
    cards = sorted(games, key=_game_order_key)
    return "<div class='grid'>" + "".join(cmb_card(g, built_short) for g in cards) + "</div>"


def _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                          opp_hitters_df=None, detail_df=None, lg_ops=None,
                          slate_df=None, lineup_df=None,
                          league_baseline=None, odds=None, last10=None):
    last10 = last10 or {}

    def _l10(team_id):
        try:
            return last10.get(int(team_id))
        except (TypeError, ValueError):
            return None

    throws = {}
    recent_era = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, pr in pitcher_rows_df.iterrows():
            throws[(pr["game_pk"], pr["Name"])] = pr.get("throws")
            recent_era[(pr["game_pk"], pr["Name"])] = (
                _f(pr.get("ERA_L5")), int(pr.get("ERA_L5_GS") or 0),
                _f(pr.get("ERA_SEASON")))
    pl_map = _pl_lookup(pl_df)
    slate_map = {}
    if slate_df is not None and not getattr(slate_df, "empty", True):
        for _, s in slate_df.iterrows():
            slate_map[s["game_pk"]] = s
    lu_map = {}
    if lineup_df is not None and not getattr(lineup_df, "empty", True):
        for _, s in lineup_df.iterrows():
            lu_map[s["game_pk"]] = {"away": s.get("away_lineup_status"),
                                    "home": s.get("home_lineup_status")}
    lg_ops_f = _f(lg_ops)

    games = []
    for gpk, gg in xw_df.groupby("game_pk", sort=False):
        a, h = _rows_by_side(gg)
        if a is None or h is None:
            continue
        srow = slate_map.get(gpk)

        def mk(r):
            side = r["side"]
            t = throws.get((gpk, r["pitcher"]), "")
            opp_lu_side = "home" if side == "away" else "away"
            d = dict(p=r["pitcher"], t=t if t in ("L", "R") else "",
                     opp=r["opp_team"], opp_abbr=_abbr(r["opp_team"]),
                     pit_xw=_f(r.get("pit_xwOBA")), pit_k=_f(r.get("pit_K%")),
                     pit_bb=_f(r.get("pit_BB%")),
                     era_l5=recent_era.get((gpk, r["pitcher"]), (None, 0, None))[0],
                     era_l5_gs=recent_era.get((gpk, r["pitcher"]), (None, 0, None))[1],
                     era_season=recent_era.get((gpk, r["pitcher"]), (None, 0, None))[2],
                     opp_xw=_f(r.get("opp_xwOBA")),
                     xw_edge=_f(r.get("edge_xwOBA")),
                     lu_status=(lu_map.get(gpk) or {}).get(opp_lu_side),
                     has_pl=False, R=0, L=0, S=0, padv=0, pl_fl={},
                     pl_sp=None, pl_sp_raw=None, pl_mx=None, pl_edge=None,
                     pl_reliable=False,
                     hitters=_hitters_for(opp_hitters_df, detail_df,
                                          gpk, r["pitcher"], lg_ops_f))
            pr = pl_map.get((gpk, side))
            if pr is not None:
                fl = {}
                if bool(pr.get("pit_low_sample")) and int(pr.get("pit_min_split_bf") or 0) > 0:
                    fl["thin"] = int(pr["pit_min_split_bf"])
                if int(pr.get("n_low_sample") or 0) > 0:
                    fl["lowpa"] = int(pr["n_low_sample"])
                if not d["t"]:
                    pt_ = pr.get("throws"); d["t"] = pt_ if pt_ in ("L", "R") else ""
                d.update(has_pl=True,
                         R=int(pr.get("n_RHB") or 0), L=int(pr.get("n_LHB") or 0),
                         S=int(pr.get("n_SW") or 0), padv=int(pr.get("n_platoon_adv") or 0),
                         pl_sp=_f(pr.get("pit_OPS_allowed")), pl_sp_raw=_f(pr.get("pit_OPS_raw")),
                         pl_mx=_f(pr.get("mx_OPS")), pl_edge=_f(pr.get("edge_OPS")),
                         pl_reliable=bool(pr.get("reliable")), pl_fl=fl)
            return d

        away_abbr = (srow.get("away_abbrev") if srow is not None else None) or _abbr(h["opp_team"])
        home_abbr = (srow.get("home_abbrev") if srow is not None else None) or _abbr(a["opp_team"])
        game_number = pd.to_numeric(srow.get("game_number"), errors="coerce") if srow is not None else np.nan
        is_dh = (str(srow.get("double_header") or "N") != "N") if srow is not None else False
        game_label = f"G{int(game_number)}" if pd.notna(game_number) and (is_dh or game_number > 1) else ""
        games.append(dict(
            away=mk(a), home=mk(h),
            away_abbr=away_abbr, home_abbr=home_abbr,
            away_l10=_l10(srow.get("away_team_id")) if srow is not None else None,
            home_l10=_l10(srow.get("home_team_id")) if srow is not None else None,
            game_pk=gpk, game_number=game_number, game_label=game_label,
            game_datetime_utc=(srow.get("game_datetime_utc") if srow is not None else None),
            time_pt=_game_time_pt(srow.get("game_datetime_utc")) if srow is not None else "",
            venue=str(srow.get("venue") or "") if srow is not None else "",
            odds=(odds or {}).get(gpk),
            league_baseline={**(league_baseline or {}), "OPS": lg_ops_f},
        ))

    # Keep the slate complete even when one or both probable pitchers are not
    # posted yet. Previously these games silently disappeared because no paired
    # xwOBA rows could be built.
    built_game_pks = {int(g["game_pk"]) for g in games if g.get("game_pk") is not None}
    if slate_df is not None and not getattr(slate_df, "empty", True):
        for _, srow in slate_df.iterrows():
            gpk = int(srow["game_pk"])
            if gpk in built_game_pks:
                continue
            game_number = pd.to_numeric(srow.get("game_number"), errors="coerce")
            is_dh = str(srow.get("double_header") or "N") != "N"
            game_label = f"G{int(game_number)}" if pd.notna(game_number) and (is_dh or game_number > 1) else ""
            games.append(dict(
                unavailable=True, game_pk=gpk, game_number=game_number, game_label=game_label,
                game_datetime_utc=srow.get("game_datetime_utc"),
                away_abbr=srow.get("away_abbrev") or _abbr(srow.get("away_team")),
                home_abbr=srow.get("home_abbrev") or _abbr(srow.get("home_team")),
                away_l10=_l10(srow.get("away_team_id")),
                home_l10=_l10(srow.get("home_team_id")),
                time_pt=_game_time_pt(srow.get("game_datetime_utc")),
                venue=str(srow.get("venue") or ""),
                away_probable=srow.get("away_probable_pitcher"),
                home_probable=srow.get("home_probable_pitcher"),
            ))
    return games


def _legend(model_label, built_txt):
    date = SLATE_DATE
    head = model_label
    if built_txt:
        head += f" · built {built_txt}"
    elif date:
        head += f" · {date}"
    return (
        "<div class='legend'>"
        f"<div class='lg-title'>{head}</div>"
        "<div class='lg-lead'><b>How to read a card:</b> each one sets a team's "
        "projected lineup against the other team's starting pitcher, both ways. The "
        "<b>lean</b> points to the side the models favor — a bigger tilt means a "
        "stronger lean. Everything here is an estimate of matchup strength from "
        "season data, not a prediction of the final score.</div>"
        "<div class='lg-keys'>"
        "<span class='k'><i class='sw warm'></i>warmer = better for hitters</span>"
        "<span class='k'><i class='sw cool'></i>cooler = better for the pitcher</span>"
        "<span class='k'><i class='sw lean'></i>lean / net tilt (xwOBA)</span>"
        "<span class='k'>◆ lineup has the lefty/righty edge vs the starter</span>"
        "</div>"
        "<div class='lg-notes'>"
        "<span><b>xwOBA</b> This lineup's overall hitting quality this season (expected "
        "on-base value per plate appearance). Hitters we've seen little of are pulled "
        "toward the league average, then each is weighted by the plate appearances his "
        "spot in the order usually gets. Not adjusted for today's starter.</span>"
        "<span><b>xOPS</b> The matchup number: how this lineup projects to hit against "
        "<i>today's starter specifically</i>, built from each hitter's and the pitcher's "
        "lefty/righty splits and weighted by expected same-hand at-bats.</span>"
        "<span><b>SP OPS alwd*</b> How much hitting today's starter has typically given up, "
        "measured against the exact lefty/righty mix he'll face tonight. Lower is better "
        "for the pitcher.</span>"
        "<span><b>K-BB%</b> Strikeout rate minus walk rate — the most reliable single read "
        "on a starter's skill (it ignores defense and luck). Higher favors the pitcher.</span>"
        f"<span><b>ERA · L{RECENT_STARTS}</b> Earned runs allowed per 9 innings over the "
        f"starter's last few outings (up to {RECENT_STARTS}; the label shows the real count "
        "when fewer), with his season ERA alongside for trend. Context only — a handful of "
        "starts is mostly noise.</span>"
        "<span><b>Edge bars</b> For each hitter, how far his matchup projection (xOPS) sits "
        "above or below a league-average bat.</span>"
        "<span><b>Shading</b> Cells tint ember when the number favors hitters and steel when "
        "it favors the pitcher; the deeper the tint, the further from the league average.</span>"
        "<span class='wide'>Moneylines are DraftKings prices pulled from ESPN at build time. "
        "Cards are ordered by the biggest gap between the two lineups' xwOBA edges — the "
        "games where the models see the clearest side come first.</span>"
        "</div></div>")


CSS = r"""/* ============================================================
   MLB matchup leans -- box-score / scout-sheet tokens
   ============================================================ */
:root{
  --bg:#f3f5f7; --surface:#ffffff; --surface-2:#eef1f4; --ink:#161b20;
  --muted:#5d6a76; --faint:#98a4af; --line:#dde3e8; --line-2:#eceff2;
  --warm:198,84,44;             /* ember  -- offense-favorable  */
  --cool:52,116,168;            /* steel  -- pitcher-favorable  */
  --lean:176,124,16;            /* amber  -- lean pill / links  */
  --amberbg:246,196,86;
  --chip-fg:#1b1e25;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  --shadow:0 1px 2px rgba(16,18,29,.05),0 10px 26px -20px rgba(16,18,29,.28);
  --r:6px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#5f6c78; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}}
html[data-theme="dark"]{
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#5f6c78; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 var(--sans);
  -webkit-font-smoothing:antialiased;padding:20px 16px 60px}
.mx-wrap{max-width:1060px;margin:0 auto}

.topbar{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:12px}
.brand{font:800 16px/1 var(--sans);letter-spacing:.13em;text-transform:uppercase}
.theme{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--muted);
  font:600 12px/1 var(--sans);padding:7px 11px;border-radius:6px;cursor:pointer}
.theme:hover{color:var(--ink)}
.theme:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}

.legend{margin:2px 2px 14px}
.lg-title{font-size:13px;font-weight:650;letter-spacing:.01em;margin-bottom:7px}
.lg-title em{font-style:normal;color:var(--muted);font-weight:500}
.lg-lead{font-size:11.5px;line-height:1.5;color:var(--muted);max-width:82ch;margin-bottom:9px}
.lg-lead b{color:var(--ink);font-weight:700}
.lg-keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:11.5px;color:var(--muted)}
.lg-keys .k{display:inline-flex;align-items:center;gap:6px}
.lg-notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 18px;
  margin-top:8px;font-size:11px;line-height:1.4;color:var(--faint)}
.lg-notes b{color:var(--muted);font-weight:700}
.lg-notes i{font-style:normal;color:var(--muted)}
.lg-notes .wide{grid-column:1/-1}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
.sw.warm{background:rgba(var(--warm),.85)} .sw.cool{background:rgba(var(--cool),.85)}
.sw.lean{background:rgba(var(--amberbg),.6)} .sw.grey{background:var(--line);border:1px solid var(--faint)}

/* chips kept for grades.html summary */
.chip{flex:1 1 64px;min-width:60px;border:1px solid var(--line-2);border-radius:6px;
  padding:6px 7px 5px;text-align:center;background:transparent}
.chip .lab{font:600 9px/1 var(--sans);text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.chip .val{font:600 16px/1.15 var(--mono);color:var(--chip-fg);font-variant-numeric:tabular-nums;margin-top:2px}
.chip .sub{font:400 10px/1 var(--mono);color:var(--muted);margin-top:2px}

.grid{display:flex;flex-direction:column;gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow);overflow:hidden}

/* ---------- game header ---------- */
.gamehead{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;
  padding:12px 16px 10px;border-bottom:1px solid var(--line-2)}
.teams{font:800 22px/1 var(--sans);letter-spacing:.02em}
.teams .at{color:var(--faint);font-weight:400;font-size:16px;margin:0 4px}
.teams .l10{font:600 11px/1 var(--mono);color:var(--muted);letter-spacing:0;
  margin-left:3px;vertical-align:middle}
.game-no{font:700 11px/1 var(--mono);color:var(--muted);vertical-align:middle}
.when{font:400 12px/1.3 var(--sans);color:var(--muted)}
.lean{margin-left:auto;display:flex;align-items:baseline;gap:7px;
  border:1px solid rgba(var(--amberbg),.55);background:rgba(var(--amberbg),.14);
  border-radius:4px;padding:4px 10px}
.lean .lk{font:600 10px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:rgba(var(--lean),1)}
.lean .lt{font:800 15px/1 var(--sans)}
.card-note{display:flex;flex-direction:column;gap:4px;padding:14px 16px 16px;color:var(--muted)}
.card-note b{color:var(--ink);font-size:13px}.card-note span{font-size:12px}
.lean .ld{font:500 12px/1 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.consensus{width:100%;font:600 12px/1.4 var(--mono);color:var(--muted);
  display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;font-variant-numeric:tabular-nums}
.consensus b{color:var(--ink);font-weight:600}
.consensus .muted{color:var(--faint);font-weight:500}
.consensus .dot{color:var(--faint)}
.ctag{font:700 10px/1 var(--sans);letter-spacing:.12em;border-radius:3px;padding:2px 7px}
.ctag.agree{color:rgba(var(--cool),1);border:1px solid rgba(var(--cool),.5)}
.ctag.diverge{color:rgba(var(--warm),1);border:1px solid rgba(var(--warm),.5)}
.ctag.na{color:var(--faint);border:1px solid var(--line)}

/* ---------- market strip ---------- */
.market{display:flex;flex-wrap:wrap;border-bottom:1px solid var(--line-2);background:var(--surface-2)}
.mcell{padding:7px 16px;border-right:1px solid var(--line-2);min-width:110px}
.mcell:last-child{border-right:0;margin-left:auto;min-width:0}
.mcell .l{font:600 9px/1.4 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.mcell .v{font:500 13px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.mcell .v .mv{color:var(--faint);font-size:11px}
.mcell.note .v{color:var(--faint);font-size:11px;padding-top:3px}

/* ---------- two sides ---------- */
.sides{display:grid;grid-template-columns:1fr 1fr}
.side{padding:12px 16px 14px;min-width:0}
.side + .side{border-left:1px solid var(--line-2)}
@media (max-width:760px){
  .sides{grid-template-columns:1fr}
  .side + .side{border-left:0;border-top:1px solid var(--line-2)}
}

/* SP block */
.sp .who{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.sp .nm{font:700 16px/1.2 var(--sans)}
.hand{font:500 10px/1.4 var(--mono);color:var(--muted);border:1px solid var(--line);border-radius:3px;padding:0 4px}
.sp .role{font-size:11px;color:var(--faint);margin-top:1px}
.spstats{display:flex;margin-top:8px;border:1px solid var(--line-2);border-radius:4px;overflow:hidden}
.stat{flex:1;padding:6px 8px 5px;border-right:1px solid var(--line-2);min-width:0}
.stat:last-child{border-right:0}
.stat .l{font:600 9px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
.stat .v{font:500 14px/1.3 var(--mono);font-variant-numeric:tabular-nums}
.stat .s{font:400 10px/1.3 var(--mono);color:var(--faint)}
.flag{font:600 9px/1.4 var(--sans);letter-spacing:.06em;color:var(--muted);
  border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:5px;white-space:nowrap}
.flag.warn{color:rgba(var(--warm),1);border-color:rgba(var(--warm),.45)}
.flag.mute{color:var(--faint)}

/* side aggregate edges */
.agg{display:flex;gap:14px;margin-top:9px;align-items:baseline;flex-wrap:wrap}
.agg .e{font:600 13px/1.3 var(--mono);font-variant-numeric:tabular-nums}
.agg .e span{font:600 9px/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-right:6px}

/* ---------- lineup: order rail + shared-axis edge bars ---------- */
details.lineup{margin-top:11px;border-top:1px solid var(--line-2)}
details.lineup summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:8px;
  padding:8px 0 6px;user-select:none}
details.lineup summary::-webkit-details-marker{display:none}
details.lineup summary:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}
summary .tl{font:700 11px/1.4 var(--sans);letter-spacing:.12em;text-transform:uppercase}
summary .st{font:600 10px/1.4 var(--sans);letter-spacing:.08em;border-radius:3px;padding:1px 6px}
summary .st.posted{color:rgba(var(--cool),1);border:1px solid rgba(var(--cool),.45)}
summary .st.partial{color:rgba(var(--lean),1);border:1px solid rgba(var(--amberbg),.5)}
summary .st.projected{color:var(--faint);border:1px solid var(--line)}
summary .lw{margin-left:auto;font:500 11px/1.4 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
summary .chev{font-size:10px;color:var(--faint);transition:transform .12s}
details[open] summary .chev{transform:rotate(90deg)}
@media (prefers-reduced-motion:reduce){summary .chev{transition:none}}

.lu-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.lu{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
table.lu th{font:600 9px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  text-align:right;padding:2px 6px 4px;border-bottom:1px solid var(--line-2);white-space:nowrap}
table.lu th.n,table.lu td.n{text-align:left}
table.lu td{font:400 12px/1.4 var(--mono);text-align:right;padding:4px 6px;
  border-bottom:1px solid var(--line-2);white-space:nowrap}
table.lu tr:last-child td{border-bottom:0}
td.ord{font-size:11px;color:var(--faint);width:26px;text-align:center;border-right:1px solid var(--line-2)}
td.n{font:400 12.5px/1.4 var(--sans);max-width:150px;overflow:hidden;text-overflow:ellipsis}
td.n .b{font:400 9px/1 var(--mono);color:var(--muted);margin-left:4px}
td.n .adv{color:rgba(var(--warm),1);font-size:9px;margin-left:2px}
td.pos{color:var(--muted);font-size:10px}
td .pa{color:var(--faint);font-size:10px}
td.na{color:var(--faint)}
tr.low td{color:var(--muted)}

td.bar{width:86px;padding:4px 8px 4px 2px}
.eb{position:relative;height:9px;background:linear-gradient(var(--line-2),var(--line-2)) no-repeat center/1px 100%}
.eb i{position:absolute;top:1px;bottom:1px;border-radius:1px}
.eb i.w{background:rgba(var(--warm),.85);left:50%}
.eb i.c{background:rgba(var(--cool),.85);right:50%}

@media (max-width:540px){
  .gamehead{gap:6px 10px}
  .teams{font-size:19px}
  .lean{margin-left:0}
  .mcell{min-width:0;flex:1 1 45%}
  .lg-notes{grid-template-columns:1fr}
  .lg-notes .wide{grid-column:auto}
  td.n{max-width:none}
  table.lu{min-width:460px}
  .spstats{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .stat{min-width:88px}
}
"""

CSS_GRADES = r"""
.gradestrip{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:0 2px 16px;
  padding:10px 14px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
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
        f"<style>{CSS}{CSS_GRADES}</style></head><body>"
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


def _record_grades(led):
    """Graded rows whose tags share the current prediction methodology."""
    return led[(led["status"] == "graded") & (led["model_tag"].isin(RECORD_TAGS))]


def records_strip_html():
    led = load_ledger_df()
    if led is None:
        return ""
    g = _record_grades(led)
    if g.empty:
        inner = "<span class='muted'>no graded games yet</span>"
    else:
        bits = [f"xwOBA full {_rec_txt(g['xw_full'])}"]
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            bits.append(f"platoon F5 {_rec_txt(ov['ops_f5'])}")
        # xwOBA vs-market (z / flat ROI) — mirrors the grades-page core card;
        # market columns are absent until the first market run.
        if "close_p_home" in g.columns and g["close_p_home"].notna().any():
            try:
                from market_backfill import vs_market_summary
                m = vs_market_summary(g).get("xwOBA")
            except Exception as e:  # noqa: BLE001
                log(f"vs-market strip degraded: {e!r}")
                m = None
            if m:
                bits.append(f"xwOBA vs mkt z {m['z']:+.2f} ({m['roi_units']:+.2f}u)")
        inner = " <span class='muted'>·</span> ".join(bits)
    return ("<div class='gradestrip'><span class='lab'>Record</span>"
            f"<span>{inner}</span><a href='grades.html'>full ledger →</a></div>")


def _wlt_badge(v):
    if isinstance(v, str) and v in ("W", "L", "T"):
        return f"<span class='wlt {v}'>{v}</span>"
    return "<span class='wlt none'>–</span>"


def _fmt_ml_cell(v):
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return "<span class='muted'>—</span>"
    return f"+{int(v)}" if v > 0 else f"{int(v)}"


def _lean_ml_cell(r, lean_col):
    """Closing ML of the lean's side; '—' when no market data."""
    lean = r.get(lean_col)
    if not isinstance(lean, str) or not lean:
        return "<span class='muted'>—</span>"
    return _fmt_ml_cell(r.get("close_home_ml") if lean == r.get("home") else r.get("close_away_ml"))


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
        cells += [_lean_ml_cell(r, "xw_lean")]
    cells += [
        final, f5,
        _wlt_badge(r["xw_full"]),
        _wlt_badge(r["ops_f5"]),
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
            f"<em>summary records combine {_esc(' + '.join(RECORD_TAGS))} because the prediction math is unchanged · "
            f"new {_esc(MODEL_TAG)} rows are hard-locked pregame · row tags preserve the audit regime · "
            "platoon records count reliable-split games only</em></div></div>")

    g = _record_grades(led)
    chips, notes = [], []

    def chip(lab, val, sub=None):
        s = f"<div class='sub'>{sub}</div>" if sub else ""
        chips.append(f"<div class='chip'><div class='lab'>{lab}</div>"
                     f"<div class='val'>{val}</div>{s}</div>")

    if g.empty:
        summary = "<div class='gr-note'>No graded games in this model family yet.</div>"
    else:
        chip("Graded", str(len(g)), f"{n_pend} pending")
        b, p = _rec_parts(g["xw_full"]); chip("xwOBA · full", b, p)
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            b, p = _rec_parts(ov["ops_f5"]); chip("Platoon · F5", b, p)
        # vs-market scoreboard (closing DK MLs attached by grade_leans.py via
        # market_backfill; columns absent until the first market run).
        if "close_p_home" in g.columns and g["close_p_home"].notna().any():
            try:
                from market_backfill import vs_market_summary
                mkt = vs_market_summary(g)
            except Exception as e:  # noqa: BLE001
                log(f"vs-market summary degraded: {e!r}")
                mkt = {}
            m = mkt.get("xwOBA")
            if m:
                chip("xwOBA · vs mkt", f"{m['w']}-{m['n'] - m['w']}",
                     f"z {m['z']:+.2f} · {m['roi_units']:+.2f}u flat")
            m5 = mkt.get("platoon_f5")
            if m5:
                chip("Platoon · vs F5 mkt", f"{m5['w']}-{m5['n'] - m5['w']}",
                     f"z {m5['z']:+.2f} · {m5['roi_units']:+.2f}u flat")
        notes.append("xwOBA graded full-game vs devigged DK closing ML (ESPN capture); "
                     "z and flat ROI are the primary metrics. Platoon graded vs the "
                     "devigged DK F5 closing ML from the same capture (ties push), "
                     "reliable-split games only.")
        summary = ("<div class='gr-summary'>" + "".join(chips) + "</div>"
                   + (f"<div class='gr-note'>{' · '.join(notes)}</div>" if notes else ""))

    show_tag = led["model_tag"].nunique() > 1
    show_ml = "close_home_ml" in led.columns and led["close_home_ml"].notna().any()
    heads = (["Date", "Game", "xwOBA lean", "Platoon lean"]
             + (["xw ML"] if show_ml else [])
             + ["Final", "F5", "xw F", "pl F5"]
             + (["Model"] if show_tag else []))
    led = led.sort_values(["game_date", "game_pk"], ascending=[False, True])
    rows = "".join(_grades_row(r, show_tag, show_ml) for _, r in led.iterrows())
    table = ("<div class='gr-tablewrap'><table class='gr'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in heads)
             + f"</tr></thead><tbody>{rows}</tbody></table></div>")
    return html_document(back + head + summary + table, built_txt, title="MLB lean grades")


def render_combined_html(xw_df, pl_df, pitcher_rows_df, built_txt,
                         opp_hitters_df=None, detail_df=None, lg_ops=None,
                         slate_df=None, lineup_df=None,
                         league_baseline=None, odds=None, last10=None):
    games = _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                                  opp_hitters_df=opp_hitters_df, detail_df=detail_df,
                                  lg_ops=lg_ops, slate_df=slate_df, lineup_df=lineup_df,
                                  league_baseline=league_baseline, odds=odds, last10=last10)
    legend = _legend("MLB matchup leans — xwOBA + platoon OPS", built_txt)
    strip = records_strip_html()
    if not games:
        inner = legend + strip + "<div class='legend'><div class='lg-title'>No paired probables yet — " \
                                 "probables/lineups not posted. Check back closer to first pitch.</div></div>"
        return html_document(inner, built_txt)
    built_short = built_txt.split("·")[0].strip()
    body = legend + strip + build_combined(games, built_short)
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
def _built_text_now():
    now_pt = datetime.now(PT)
    return (now_pt.strftime("%I:%M %p").lstrip("0")
            + now_pt.strftime(" PT · %Y-%m-%d"))


def write_grades_page(built_txt=None):
    """Render grades.html from the latest on-disk ledger without fetching a slate."""
    built_txt = built_txt or _built_text_now()
    os.makedirs(OUT_DIR, exist_ok=True)
    grades_path = os.path.join(OUT_DIR, "grades.html")
    with open(grades_path, "w") as f:
        f.write(render_grades_html(built_txt))
    log(f"Wrote {grades_path}")
    return grades_path


def main():
    built_txt = _built_text_now()
    if "--grades-only" in sys.argv:
        write_grades_page(built_txt)
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")

    # Render once here for local/partial paths. CI renders it again after the
    # post-build grading pass so the deployed ledger includes newly ingested
    # matchups from this same build.
    write_grades_page(built_txt)

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
        matchup_platoon_df, platoon_detail_df, league_ops_overall = build_platoon_matchup(
            pitcher_rows_df, opp_hitters_df, data["people"],
            data["player_splits_hit"], data["player_splits_pit"],
            league_splits=data.get("player_splits_hit_league"),
            league_people=data.get("people_league_hitters"))
    except Exception as e:  # noqa: BLE001
        # Platoon lens is optional; degrade to xwOBA-only rather than fail.
        log(f"Platoon lens skipped: {e!r}")
        matchup_platoon_df = pd.DataFrame()
        platoon_detail_df, league_ops_overall = pd.DataFrame(), np.nan

    # Dump an auditable pregame snapshot for the grading ledger. The grader
    # rejects rows captured at/after their scheduled start, so in-progress
    # refreshes cannot alter the published pregame record.
    snapshot_utc = datetime.now(UTC).isoformat()
    for frame in (matchup_df, matchup_platoon_df):
        if frame is not None and not frame.empty:
            frame["model_tag"] = MODEL_TAG
            frame["snapshot_utc"] = snapshot_utc
            if "game_datetime_utc" in frame.columns:
                frame["scheduled_start_utc"] = frame["game_datetime_utc"]
    os.makedirs(DATA_DIR, exist_ok=True)
    matchup_df.to_csv(os.path.join(DATA_DIR, f"leans_{SLATE_DATE}_xw.csv"), index=False)
    if matchup_platoon_df is not None and not matchup_platoon_df.empty:
        matchup_platoon_df.to_csv(os.path.join(DATA_DIR, f"leans_{SLATE_DATE}_pl.csv"), index=False)

    log("Fetching pregame odds (best-effort, display-only) ...")
    try:
        odds = fetch_pregame_odds(data["slate_df"])
    except Exception as e:  # noqa: BLE001
        log(f"pregame odds skipped: {e!r}")
        odds = {}

    log("Fetching last-10 records (best-effort, display-only) ...")
    try:
        last10 = fetch_last10_records(data["slate_df"])
    except Exception as e:  # noqa: BLE001
        log(f"last-10 records skipped: {e!r}")
        last10 = {}

    log("Rendering index.html ...")
    html = render_combined_html(
        matchup_df, matchup_platoon_df, pitcher_rows_df, built_txt,
        opp_hitters_df=opp_hitters_df, detail_df=platoon_detail_df,
        lg_ops=league_ops_overall, slate_df=data["slate_df"],
        lineup_df=data["lineup_projection_df"],
        league_baseline=data["league_baseline"], odds=odds, last10=last10)
    with open(out_path, "w") as f:
        f.write(html)
    log(f"Wrote {out_path} ({len(html):,} bytes, {len(matchup_df)} matchup rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
