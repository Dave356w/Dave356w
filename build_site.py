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
#   - index.html carries a "built HH:MM" line beside the slate date; every
#     clock on the site is Pacific and the legend is where that is stated.
# ============================================================

import base64
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

import pitch_arsenal

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
# True only for a posted hitter who is absent from the season Savant
# leaderboard and therefore carries the active team's PA-weighted xwOBA.
XWOBA_TEAM_BACKFILL_COL = "xwOBA_team_backfill"
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
# Team cap logos are keyed by the same StatsAPI team id the slate already
# carries. They live on MLB's static CDN (not in the API payload), so the build
# fetches the on-light / on-dark cap SVGs once per run and inlines them as data
# URIs -- keeping the page fully self-contained. Any fetch miss degrades to the
# text abbreviation, so the CDN being unreachable never breaks the build.
USE_TEAM_LOGOS = os.environ.get("USE_TEAM_LOGOS", "1") != "0"
LOGO_CDN = "https://www.mlbstatic.com/team-logos"
DATA_DIR = os.environ.get("DATA_DIR", "data")            # grading ledger home
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
MODEL_TAG = os.environ.get("MODEL_TAG", "xw+plat_consol_v10")  # keep in sync with grade_leans.py
_RECORD_FAMILIES = {
    # v3 changed only ledger locking/identity; its prediction math is v2.
    "xw+plat_consol_v3": ("xw+plat_consol_v2", "xw+plat_consol_v3"),
    # v4 re-weights lineup composites by expected PA per batting-order slot
    # (was season BBE / split PA); that changes prediction math, so it starts a
    # fresh record family and never mixes with v2/v3 in the ledger or weight fit.
    # v5 adds empirical-Bayes xwOBA shrinkage (batters + starter) on top of v4;
    # another prediction-math change, so it starts its own family again.
    # v6 replaces the starter-only pitching input with a nine-inning blend of
    # expected starter workload and a role-filtered bullpen aggregate. It is a
    # new prediction family; older tags remain in the ledger history.
    # v7 centre-matches the weighted and unweighted moments used to estimate
    # xwOBA shrinkage K. It changes prediction math and starts another family.
    # v8 replaces the per-build estimates with fixed K=100 shrinkage for both
    # batters and pitchers. It changes prediction math and starts another family.
    # v9 applies the starter-handedness lineup adjustment only to expected
    # starter innings, then uses the neutral lineup against the bullpen. It is
    # isolated from v8 for both records and delta-scale ranking.
    #
    # v10 blends the two phases on their share of plate appearances rather than
    # of innings, using measured BF/IP. Prediction math changed, so the tag is
    # bumped -- but the record question is answered by measurement, not reflex.
    # Sweeping the starter/bullpen BF/IP ratio over 0.95-1.10 on the 2026-07-28
    # slate moves xw_net with sd 0.0004-0.0009 against a median |xw_net| of
    # 0.0268 (1.6-3.4%) and flips 0 of 12 leans; only an implausible 1.20 ratio
    # flips one, on a game whose |xw_net| is 0.0013. v9 and v10 rows are
    # therefore produced by models that agree on the decision and share a
    # win-loss line. Isolating v10 would reset the graded sample to zero for
    # the eighth time in a month and buy nothing.
    "xw+plat_consol_v9": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
    "xw+plat_consol_v10": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
}
RECORD_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "RECORD_TAGS", ",".join(_RECORD_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
    ).split(",") if t.strip()
)
# Scale-compatibility: which tags measure |xw_net| on the SAME units. Separate
# from RECORD_TAGS (win-loss compatibility) -- the two equivalence relations
# disagree. v5 introduced empirical-Bayes shrinkage that halved the delta scale
# (median |xw_net| .036 -> .018); v6 inherits v5's shrinkage, so v5 and v6 share
# units even though v6 is a fresh RECORD family. v7 changes the estimated K and
# therefore starts its own scale family. v8 fixes K at 100 for both pools,
# widening the shrunk distribution and starting another scale family. v10
# only re-weights a convex combination of the two existing phases, so the
# delta distribution is unchanged in units. Pre-v5
# tags are on the older, unshrunk scale. The default isolates any unlisted tag
# (never mixes scales); list a tag here only to pool it with others that share
# its units.
#
# v8 and v9 share units. v9 differs from v8 by exactly one term --
# v9 - v8 == -(1 - q) * platoon_delta_sp * P_BP / L -- because the two forms
# expand to the same starter phase and differ only in which lineup composite
# meets the bullpen. Measured on the 2026-07-28 slate that term has sd 0.00103
# against a matchup dispersion of 0.01662 (6.2%), and compare_v8_v9.py reports
# 0 lean flips over 24 eligible games. That is a units-preserving change, so
# the two pool for delta-scale ranking. RECORD_TAGS is a separate question and
# is deliberately left isolated.
_SCALE_FAMILIES = {
    "xw+plat_consol_v5": ("xw+plat_consol_v5", "xw+plat_consol_v6"),
    "xw+plat_consol_v6": ("xw+plat_consol_v5", "xw+plat_consol_v6"),
    "xw+plat_consol_v7": ("xw+plat_consol_v7",),
    "xw+plat_consol_v8": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                          "xw+plat_consol_v10"),
    "xw+plat_consol_v9": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                          "xw+plat_consol_v10"),
    "xw+plat_consol_v10": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                           "xw+plat_consol_v10"),
}
SCALE_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "SCALE_TAGS", ",".join(_SCALE_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
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

# Sequential starter/bullpen model. A probable pitcher is treated as an opener when either his
# recent starts are repeatedly short or his recent usage is overwhelmingly
# short relief work. The latter catches relievers named for a one-off start
# before they have accumulated two official starts. v6 estimates the probable
# pitcher's innings, then fills the balance of nine innings with a filtered
# bullpen aggregate. Long/bulk relievers remain in the bullpen pool, but the
# model deliberately does not guess which individual bulk arm will follow.
#
# The bullpen pool uses season role stats to exclude rotation starters:
# <=35% of appearances started and <=3.0 IP per appearance. Each included
# reliever's xwOBA is shrunk independently before the pool is weighted by his
# estimated relief batters faced. If the role-filtered aggregate is unavailable,
# the starter phase is still displayed but the full-game lean is suppressed.
# There is no whole-staff opener fallback: it could include rotation arms and
# double-count the probable pitcher.
# The platoon lens is left alone -- an opener's tiny vL/vR split already fails
# the reliability gate, so that lens abstains on its own. Each affected side is
# flagged with its detection reason and pitching basis so it stays auditable.
OPENER_MAX_AVG_IP = 3.0     # avg IP/start below this => treat probable as an opener
OPENER_MIN_STARTS = 2       # need a repeated pattern, not one rain-shortened start
BULLPEN_MIN_PITCHERS = 3    # minimum role-filtered relievers for an aggregate
OPENER_ROLE_APPEARANCES = 10
OPENER_ROLE_MIN_APPEARANCES = 3
OPENER_ROLE_MAX_STARTS = 1
OPENER_ROLE_MIN_RELIEF_SHARE = 0.70
OPENER_ROLE_MAX_MEDIAN_IP = 2.0
OPENER_ROLE_MAX_P80_PITCHES = 55.0
OPENER_STRETCHED_MIN_IP = 4.0
OPENER_STRETCHED_MIN_PITCHES = 65
GAME_INNINGS = 9.0
SP_IP_PRIOR = 5.2
SP_IP_PRIOR_STARTS = 3.0
SP_IP_MIN = 3.0
SP_IP_MAX = 7.0
OPENER_IP_DEFAULT = 1.5
OPENER_IP_MIN = 1.0 / 3.0
RP_MAX_START_SHARE = 0.35
RP_MAX_IP_PER_APPEARANCE = 3.0


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


_team_logo_cache = {}
_logo_cdn_down = False   # tripped once a fetch exhausts retries; skips further
                         # network so an unreachable CDN can't stall the build


def _fetch_logo_svg(url, cache_name, tries=2):
    """Fetch one SVG logo, cached to CACHE_DIR (cap marks are effectively
    static). Returns the raw SVG text, or None on any failure / non-SVG body so
    the caller can fall back to the text abbreviation. A cached file is always
    served; once the CDN is known-unreachable this run, uncached teams skip the
    network entirely rather than each re-paying the retry budget."""
    global _logo_cdn_down
    path = os.path.join(CACHE_DIR, f"logo_{cache_name}.svg")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read() or None
        except Exception:  # noqa: BLE001
            pass
    if _logo_cdn_down:
        return None
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            text = r.text
            if "<svg" not in text.lower():
                return None
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:  # noqa: BLE001
                pass
            return text
        except Exception as e:  # noqa: BLE001
            last = e
            if k == tries - 1:
                break
            time.sleep(0.4 * (2 ** k))
    _logo_cdn_down = True
    log(f"  team logo fetch failed ({cache_name}); disabling logo fetch this run: {last!r}")
    return None


def _svg_data_uri(svg_text):
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def team_logo_uris(team_id):
    """(<light-bg data-uri>, <dark-bg data-uri>) cap logos for a team, or
    (None, None) when unavailable. Both cap variants share the StatsAPI team id;
    each is fetched once per build and memoized, then embedded so the page stays
    self-contained. If only one variant resolves it is used for both; if neither
    does the card keeps its text abbreviation."""
    if not USE_TEAM_LOGOS or team_id is None:
        return None, None
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return None, None
    if tid in _team_logo_cache:
        return _team_logo_cache[tid]
    light = _fetch_logo_svg(f"{LOGO_CDN}/team-cap-on-light/{tid}.svg", f"cap_light_{tid}")
    dark = _fetch_logo_svg(f"{LOGO_CDN}/team-cap-on-dark/{tid}.svg", f"cap_dark_{tid}")
    if not light and not dark:
        _team_logo_cache[tid] = (None, None)
        return None, None
    light, dark = (light or dark), (dark or light)
    uris = (_svg_data_uri(light), _svg_data_uri(dark))
    _team_logo_cache[tid] = uris
    return uris


def get_slate(slate_date, sport_id=1):
    data = _get_json("https://statsapi.mlb.com/api/v1/schedule",
                     {"sportId": sport_id, "date": slate_date,
                      "hydrate": "probablePitcher,team,venue,linescore"})
    rows = []
    for db in data.get("dates", []):
        od = db.get("date", slate_date)
        for g in db.get("games", []):
            a, h = g["teams"]["away"], g["teams"]["home"]
            linescore = g.get("linescore") or {}
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
                "abstract_state": g.get("status", {}).get("abstractGameState"),
                "status": g.get("status", {}).get("detailedState"),
                "away_score": a.get("score"),
                "home_score": h.get("score"),
                "current_inning": linescore.get("currentInning"),
                "current_inning_ordinal": linescore.get("currentInningOrdinal"),
                "inning_half": linescore.get("inningHalf"),
                "is_top_inning": linescore.get("isTopInning"),
                # Base-out state. `offense` carries a player object per occupied
                # base and omits the key entirely when the base is empty, so
                # presence is the signal. Absent on unstarted games -> all False
                # / no outs, and the card hides the indicator.
                "on_first": bool((linescore.get("offense") or {}).get("first")),
                "on_second": bool((linescore.get("offense") or {}).get("second")),
                "on_third": bool((linescore.get("offense") or {}).get("third")),
                "outs": linescore.get("outs"),
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


def load_pitcher_xera():
    """player_id -> Statcast xERA (expected ERA), cached once per ET day.

    xERA lives on Savant's *expected_statistics* leaderboard, not the custom
    board, so it needs its own CSV. `min=1` (not the default `min=q`) keeps
    non-qualified probables in. Strictly best-effort and display-only: any fetch
    failure or a missing/renamed column yields an empty map and the card's xERA
    cell renders an em-dash — a Savant outage never fails the build here."""
    try:
        df = cached_csv(
            "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
            f"?type=pitcher&year={SEASON}&position=&team=&filterType=bip&min=1&csv=true",
            "xstats_pitcher")
    except Exception as e:  # noqa: BLE001
        log(f"  xERA leaderboard unavailable ({e!r}) -> em-dashes")
        return {}
    # Export labels the expected-ERA column `xera`; tolerate the `est_era`
    # variant used elsewhere on the expected-stats board.
    col = next((c for c in ("xera", "est_era") if c in df.columns), None)
    if col is None:
        log(f"  xERA column absent from expected-stats board (cols={list(df.columns)[:12]}...) -> em-dashes")
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            pid = int(r["player_id"])
        except (TypeError, ValueError):
            continue
        v = _f(r.get(col))
        if v is not None:
            out[pid] = v
    log(f"  xERA leaderboard: {len(out)} pitchers (col '{col}')")
    return out


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
    """Build pre-slate recent-start and recent-role profiles for probables."""
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

        appearances = []
        for blk in data.get("stats", []):
            for sk in blk.get("splits", []):
                st = sk.get("stat", {}) or {}
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
                try:
                    pitches = float(st.get("numberOfPitches"))
                except (TypeError, ValueError):
                    pitches = np.nan
                appearances.append({
                    "date": game_date,
                    "game_pk": int(game_pk),
                    "outs": _innings_to_outs(st.get("inningsPitched")),
                    "earned_runs": er,
                    "is_start": int(st.get("gamesStarted") or 0) >= 1,
                    "pitches": pitches,
                })

        appearances.sort(key=lambda x: (x["date"], x["game_pk"]), reverse=True)
        all_starts = [a for a in appearances if a["is_start"]]
        starts = all_starts[:limit]
        outs = sum(a["outs"] for a in starts)
        earned_runs = sum(a["earned_runs"] for a in starts)
        season_start_outs = sum(a["outs"] for a in all_starts)
        role_apps = appearances[:OPENER_ROLE_APPEARANCES]
        pitch_counts = [a["pitches"] for a in role_apps if pd.notna(a["pitches"])]
        recent_starts = sum(a["is_start"] for a in role_apps)
        stretched = sum(
            a["outs"] >= int(OPENER_STRETCHED_MIN_IP * 3)
            or (pd.notna(a["pitches"]) and a["pitches"] >= OPENER_STRETCHED_MIN_PITCHES)
            for a in role_apps
        )
        out[pid] = {
            "era": round(earned_runs * 27.0 / outs, 2) if outs > 0 else np.nan,
            "starts": len(starts),
            # Average innings per recent start -- the opener signal. Openers are
            # credited a game started but pitch ~1 inning, so this runs low.
            "avg_ip": (outs / 3.0 / len(starts)) if starts else np.nan,
            "season_starts": len(all_starts),
            "season_avg_ip": (
                season_start_outs / 3.0 / len(all_starts)
                if all_starts else np.nan
            ),
            # The last ten appearances describe the pitcher's current role. A
            # short-relief profile lets us recognize a newly named opener even
            # when he has only one prior official start.
            "appearances": len(role_apps),
            "recent_starts": recent_starts,
            "relief_share": (
                (len(role_apps) - recent_starts) / len(role_apps)
                if role_apps else np.nan
            ),
            "median_ip": (
                float(np.median([a["outs"] for a in role_apps])) / 3.0
                if role_apps else np.nan
            ),
            "p80_pitches": (
                float(np.percentile(pitch_counts, 80))
                if pitch_counts else np.nan
            ),
            "pitch_count_appearances": len(pitch_counts),
            "stretched_appearances": stretched,
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


def team_batter_xwoba(team_id, batter_stat, league_fallback=np.nan):
    """PA-weighted xwOBA for Savant-listed active hitters on one team.

    Used only when a posted hitter is absent from the Savant leaderboard.
    The league prior is a last resort when no rostered hitter has a usable
    xwOBA/PA pair.
    """
    vals, weights = [], []
    for pid in roster_lineup(team_id, batter_stat):
        stat = batter_stat.get(pid) or {}
        xw = pd.to_numeric(stat.get("xwOBA"), errors="coerce")
        pa = pd.to_numeric(stat.get("PA"), errors="coerce")
        if pd.notna(xw) and pd.notna(pa) and float(pa) > 0:
            vals.append(float(xw))
            weights.append(float(pa))
    if vals:
        return float(np.average(vals, weights=weights))
    fallback = pd.to_numeric(league_fallback, errors="coerce")
    return float(fallback) if pd.notna(fallback) else np.nan


def fill_lineup_from_roster(team_id, posted_ids, batter_stat, target_size=LINEUP_SIZE):
    """Keep posted hitters in order; fill only missing slots by roster PA."""
    resolved, seen = [], set()
    for pid in posted_ids or []:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            return resolved[:target_size]
    for pid in roster_lineup(team_id, batter_stat):
        if pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            break
    return resolved[:target_size]


def resolve_lineup(game_pk, side, team_id, batter_stat, return_meta=False,
                   league_xwoba=np.nan):
    """Resolve the posted order first, then fill only genuinely open slots.

    A posted player missing from the season Savant leaderboard is retained and
    assigned the active team's PA-weighted xwOBA. That team aggregate is marked
    so the downstream player-level shrinkage step does not shrink it again.
    """
    away_ids, home_ids = gf_lineups(game_pk)
    raw = away_ids if side == "away" else home_ids
    posted, seen = [], set()
    for pid in raw:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid not in seen:
            posted.append(pid); seen.add(pid)
        if len(posted) >= LINEUP_SIZE:
            break

    missing, backfilled = [], []
    for pid in posted:
        stat = batter_stat.get(pid) or {}
        if bool(stat.get(XWOBA_TEAM_BACKFILL_COL)):
            backfilled.append(pid)
        elif pd.isna(pd.to_numeric(stat.get("xwOBA"), errors="coerce")):
            missing.append(pid)
            backfilled.append(pid)
    if missing:
        team_xwoba = team_batter_xwoba(team_id, batter_stat, league_xwoba)
        for pid in missing:
            batter_stat[pid] = {
                **(batter_stat.get(pid) or {}),
                "xwOBA": team_xwoba,
                "PA": 0.0,
                "BBE": 0.0,
                XWOBA_TEAM_BACKFILL_COL: True,
            }

    resolved = fill_lineup_from_roster(team_id, posted, batter_stat, LINEUP_SIZE)
    posted_count = min(len(posted), LINEUP_SIZE)
    filled_count = max(0, len(resolved) - posted_count)
    status = ("posted" if posted_count >= LINEUP_SIZE
              else "partial_filled" if posted_count > 0 else "projected")
    meta = {"status": status, "projected": status != "posted",
            "posted_count": int(posted_count), "filled_count": int(filled_count),
            "resolved_count": int(len(resolved)),
            "savant_backfill_count": int(len(backfilled))}
    return (resolved, meta) if return_meta else (resolved, meta["projected"])


STAT_COLS = ["BBE", "LA°", "EV", "Hard Hit%", "xwOBA", "xBA", "xSLG", "K%", "BB%"]
BB_COLS = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]

_pitcher_roster_cache = {}
_team_pitcher_role_cache = {}


def pitcher_roster(team_id):
    """Active-roster pitcher ids for a team (StatsAPI)."""
    team_id = int(team_id)
    if team_id in _pitcher_roster_cache:
        return _pitcher_roster_cache[team_id]
    data = _get_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                     {"rosterType": "active"})
    ids = [int(r["person"]["id"]) for r in data.get("roster", [])
           if (r.get("position", {}) or {}).get("abbreviation") == "P"]
    _pitcher_roster_cache[team_id] = ids
    return ids


def load_team_pitcher_roles(team_id):
    """Season-to-date workload roles for one club in a single StatsAPI call.

    Returns active/used pitchers keyed by player id with appearances, starts,
    start share, innings per appearance, and batters faced. These fields are
    used only to separate the rotation from the relief pool; no specific bulk
    follower is projected.
    """
    team_id = int(team_id)
    if team_id in _team_pitcher_role_cache:
        return _team_pitcher_role_cache[team_id]
    data = _get_json("https://statsapi.mlb.com/api/v1/stats", {
        "stats": "season",
        "group": "pitching",
        "season": SEASON,
        "sportIds": SPORT_ID,
        "teamId": team_id,
        "gameType": "R",
        "playerPool": "ALL",
        "limit": 1000,
    })
    out = {}
    for blk in data.get("stats", []):
        for sk in blk.get("splits", []):
            who = sk.get("player") or sk.get("person") or {}
            pid = who.get("id")
            if pid is None:
                continue
            st = sk.get("stat", {}) or {}
            try:
                apps = int(st.get("gamesPitched") or st.get("gamesPlayed") or 0)
                starts = int(st.get("gamesStarted") or 0)
                bf = float(st.get("battersFaced") or 0)
            except (TypeError, ValueError):
                continue
            outs = _innings_to_outs(st.get("inningsPitched"))
            out[int(pid)] = {
                "appearances": apps,
                "starts": starts,
                "start_share": starts / apps if apps > 0 else np.nan,
                "avg_ip_per_appearance": outs / 3.0 / apps if apps > 0 else np.nan,
                "batters_faced": bf,
            }
    _team_pitcher_role_cache[team_id] = out
    return out


def relief_pitcher_ids(team_id, role_stats, probable_pid=None):
    """Active-roster pitchers whose season workload is primarily relief.

    The loose three-inning ceiling intentionally keeps long/bulk relievers in
    the pooled estimate. It excludes rotation arms, and the current probable
    is removed so his starter/opener innings are not counted twice.
    """
    probable_pid = int(probable_pid) if probable_pid is not None else None
    out = []
    for pid in pitcher_roster(team_id):
        if pid == probable_pid:
            continue
        role = (role_stats or {}).get(pid) or {}
        apps = int(role.get("appearances") or 0)
        share = role.get("start_share")
        avg_ip = role.get("avg_ip_per_appearance")
        if (
            apps > 0
            and share is not None
            and pd.notna(share)
            and float(share) <= RP_MAX_START_SHARE
            and avg_ip is not None
            and pd.notna(avg_ip)
            and float(avg_ip) <= RP_MAX_IP_PER_APPEARANCE
        ):
            out.append(pid)
    return out


def bf_per_ip(role):
    """Measured batters faced per inning from a pitcher's season role line.

    The blend weight has to be a share of plate appearances, not of innings,
    because xwOBA is a per-PA rate. BF/IP is what converts one to the other and
    it is observed, not modelled: both terms come from the same season role
    call, so a pitcher who allows more baserunners shows a higher rate without
    any on-base proxy standing in for it. Returns None when either term is
    missing, which degrades the caller to the innings share.
    """
    role = role or {}
    bf = _f(role.get("batters_faced"))
    apps = _f(role.get("appearances"))
    avg_ip = _f(role.get("avg_ip_per_appearance"))
    if bf is None or apps is None or avg_ip is None:
        return None
    season_ip = float(avg_ip) * float(apps)
    if bf <= 0 or season_ip <= 0:
        return None
    return float(bf) / season_ip


def bullpen_xwoba_aggregate(team_id, probable_pid, pitcher_stat, role_stats,
                            prior, shrink_k):
    """Role-filtered bullpen xwOBA with per-pitcher empirical-Bayes shrinkage.

    Talent is shrunk by each pitcher's Savant BF before aggregation. Usage
    weights approximate relief BF as season BF × (1 - start share), preventing
    a swingman's starter work from receiving full bullpen weight.
    """
    weighted = []
    for pid in relief_pitcher_ids(team_id, role_stats, probable_pid):
        stat = pitcher_stat.get(pid) or {}
        bf = _f(stat.get("PA"))
        xw = _f(stat.get("xwOBA"))
        role = (role_stats or {}).get(pid) or {}
        share = _f(role.get("start_share"))
        role_bf = _f(role.get("batters_faced"))
        if bf is None or bf <= 0 or xw is None:
            continue
        relief_share = max(0.0, 1.0 - (share or 0.0))
        # Role BF is team-specific, whereas Savant PA follows the pitcher
        # across trades. Use team BF for expected club usage when available;
        # retain Savant PA only as the talent-shrinkage sample size.
        usage_bf = (role_bf if role_bf is not None and role_bf > 0 else bf) * relief_share
        if usage_bf <= 0:
            continue
        shrunk = _shrink_one(xw, bf, prior, shrink_k)
        if shrunk is not None and pd.notna(shrunk):
            weighted.append((pid, usage_bf, float(shrunk), bf_per_ip(role)))
    if len(weighted) < BULLPEN_MIN_PITCHERS:
        return None
    total_bf = sum(w for _, w, _, _ in weighted)
    # Pool BF/IP on the same pitchers and the same usage weights that produced
    # the pool's xwOBA, so the rate describes the innings this bullpen is
    # actually expected to throw. Pitchers missing a role line are skipped
    # here without affecting the xwOBA aggregate above.
    rated = [(w, r) for _, w, _, r in weighted if r is not None]
    rate_bf = sum(w for w, _ in rated)
    return {
        "xwOBA": sum(w * x for _, w, x, _ in weighted) / total_bf,
        "pitcher_count": len(weighted),
        "relief_bf": total_bf,
        "bf_per_ip": (sum(w * r for w, r in rated) / rate_bf) if rate_bf > 0 else None,
        "pitcher_ids": tuple(pid for pid, _, _, _ in weighted),
    }


def expected_pitcher_ip(profile, classification=None):
    """Pregame expected workload from pre-slate starts/current role.

    Normal starters blend the last-five start average with their season start
    average (or a fixed league prior when unavailable). Openers use the signal
    that classified them: repeated short starts use start IP, while a reliever
    spot start uses median recent appearance length. Values are bounded to
    plausible role ranges and never use the current slate appearance.
    """
    profile = profile or {}
    classification = classification or {}

    def finite(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    recent = finite(profile.get("avg_ip"))
    season = finite(profile.get("season_avg_ip"))
    if classification:
        if classification.get("reason") == "repeated_short_starts":
            estimate = recent
        else:
            estimate = finite(profile.get("median_ip"))
            if estimate is None:
                estimate = recent
        estimate = OPENER_IP_DEFAULT if estimate is None else estimate
        return float(np.clip(estimate, OPENER_IP_MIN, OPENER_MAX_AVG_IP))

    prior = season if season is not None else SP_IP_PRIOR
    n_recent = int(profile.get("starts") or 0)
    if recent is None or n_recent <= 0:
        estimate = prior
    else:
        estimate = (
            n_recent * recent + SP_IP_PRIOR_STARTS * prior
        ) / (n_recent + SP_IP_PRIOR_STARTS)
    return float(np.clip(estimate, SP_IP_MIN, SP_IP_MAX))


def build_pitching_plans(slate_df, recent_profiles, pitcher_stat,
                         league_baseline):
    """Expected-IP + role-filtered bullpen plan for every probable-pitcher side."""
    plans = {}
    classifications = opener_classifications(recent_profiles)
    prior = _f((league_baseline or {}).get("xwOBA"))
    shrink_k = XWOBA_SHRINK_K
    roles_by_team = {}
    bullpen_by_pair = {}

    for _, g in slate_df.iterrows():
        for side in ("away", "home"):
            pid = g.get(f"{side}_probable_pitcher_id")
            tid = g.get(f"{side}_team_id")
            if pd.isna(pid) or pd.isna(tid):
                continue
            pid, tid = int(pid), int(tid)
            classification = classifications.get(pid)
            expected_ip = expected_pitcher_ip(
                (recent_profiles or {}).get(pid), classification
            )

            if tid not in roles_by_team:
                try:
                    roles_by_team[tid] = load_team_pitcher_roles(tid)
                except Exception as e:  # noqa: BLE001
                    log(f"  bullpen roles unavailable for team {tid}: {e!r}")
                    roles_by_team[tid] = None
            pair = (tid, pid)
            if pair not in bullpen_by_pair:
                try:
                    bullpen_by_pair[pair] = bullpen_xwoba_aggregate(
                        tid, pid, pitcher_stat, roles_by_team.get(tid), prior, shrink_k
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  bullpen aggregate failed for team {tid}: {e!r}")
                    bullpen_by_pair[pair] = None
            bullpen = bullpen_by_pair[pair]
            if bullpen is None:
                basis = "starter_only_no_fullgame_lean"
            elif classification:
                basis = "opener_bullpen_sequential"
            else:
                basis = "starter_bullpen_sequential"

            key = (int(g["game_pk"]), side)
            plans[key] = {
                "opener": bool(classification),
                "opener_reason": (classification or {}).get("reason"),
                "opener_confidence": (classification or {}).get("confidence"),
                "pitching_basis": basis,
                "expected_sp_ip": expected_ip,
                "bullpen_xwOBA": None if bullpen is None else bullpen.get("xwOBA"),
                "bullpen_pitchers": None if bullpen is None else bullpen.get("pitcher_count"),
                "bullpen_relief_bf": None if bullpen is None else bullpen.get("relief_bf"),
                "sp_bf_per_ip": bf_per_ip((roles_by_team.get(tid) or {}).get(pid)),
                "bp_bf_per_ip": None if bullpen is None else bullpen.get("bf_per_ip"),
            }
            if bullpen is not None:
                log(
                    f"  sequential pitching: pid={pid} team={tid} basis={basis} "
                    f"exp_ip={expected_ip:.2f} rp_xwOBA={bullpen.get('xwOBA'):.3f} "
                    f"(n_rp={bullpen.get('pitcher_count')})"
                )
            else:
                log(
                    f"  full-game lean unavailable: pid={pid} team={tid}; "
                    "displaying starter phase only"
                )
    return plans


def opener_classifications(recent_start_era, max_avg_ip=OPENER_MAX_AVG_IP,
                           min_starts=OPENER_MIN_STARTS):
    """Classify probable pitchers whose pre-slate workload indicates an opener.

    Repeated short starts are high-confidence evidence. A short-relief role is
    a medium-confidence inference used for relievers making a first/second
    spot start. Recent starter-length work suppresses that inference.
    """
    out = {}
    for pid, m in (recent_start_era or {}).items():
        m = m or {}
        if int(m.get("stretched_appearances") or 0) > 0:
            continue
        ip = (m or {}).get("avg_ip")
        if (ip is not None and pd.notna(ip) and ip < max_avg_ip
                and int(m.get("starts") or 0) >= min_starts):
            out[int(pid)] = {
                "reason": "repeated_short_starts",
                "confidence": "high",
            }
            continue

        relief_share = m.get("relief_share")
        median_ip = m.get("median_ip")
        p80_pitches = m.get("p80_pitches")
        if (
            int(m.get("appearances") or 0) >= OPENER_ROLE_MIN_APPEARANCES
            and int(m.get("recent_starts") or 0) <= OPENER_ROLE_MAX_STARTS
            and relief_share is not None
            and pd.notna(relief_share)
            and relief_share >= OPENER_ROLE_MIN_RELIEF_SHARE
            and median_ip is not None
            and pd.notna(median_ip)
            and median_ip <= OPENER_ROLE_MAX_MEDIAN_IP
            and p80_pitches is not None
            and pd.notna(p80_pitches)
            and p80_pitches <= OPENER_ROLE_MAX_P80_PITCHES
            and int(m.get("pitch_count_appearances") or 0)
            >= OPENER_ROLE_MIN_APPEARANCES
        ):
            out[int(pid)] = {
                "reason": "reliever_spot_start",
                "confidence": "medium",
            }
    return out


def opener_pids(recent_start_era, max_avg_ip=OPENER_MAX_AVG_IP,
                min_starts=OPENER_MIN_STARTS):
    """Backward-compatible set of probable-pitcher ids classified as openers."""
    return set(opener_classifications(
        recent_start_era, max_avg_ip=max_avg_ip, min_starts=min_starts
    ))


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

        # sp_side names the block's pitcher side (structural, from the call
        # site) and is_sp marks the probable-pitcher row; both are carried on
        # every row so segment_pitcher_blocks assigns sides without name matching.
        def pitcher_row(pid, tidx, table, sp_side):
            if pd.isna(pid):
                return
            pid = int(pid)
            bio = people.get(pid, {})
            nm = bio.get("name") or f"pitcher_{pid}"
            src = (pitcher_stat if table == "stat" else pitcher_bb).get(pid, {})
            base = {**meta, "table_index": tidx, "Name": nm,
                    "sp_side": sp_side, "is_sp": True}
            if table == "stat":
                pit_rows.append({**base, "table_type": "pitchers", "Pos.": "P",
                                 **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
                                 XWOBA_TEAM_BACKFILL_COL: False,
                                 "player_id": pid, "bats": bio.get("bats"),
                                 "throws": bio.get("throws")})
            else:
                bbe = (pitcher_stat.get(pid, {}) or {}).get("BBE")
                bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": "P",
                                **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                "player_id": pid, "bats": bio.get("bats"),
                                "throws": bio.get("throws")})

        def hitter_rows(lu, tidx, table, sp_side):
            for slot, pid in enumerate(lu, start=1):
                pid = int(pid)
                bio = people.get(pid, {})
                nm = bio.get("name") or f"batter_{pid}"
                pos = bio.get("pos") or "DH"
                if pos == "P":
                    pos = "DH"
                src = (batter_stat if table == "stat" else batter_bb).get(pid, {})
                base = {**meta, "table_index": tidx, "Name": nm,
                        BATTING_ORDER_COL: slot, "sp_side": sp_side, "is_sp": False}
                if table == "stat":
                    backfill_value = src.get(XWOBA_TEAM_BACKFILL_COL)
                    pit_rows.append({**base, "table_type": "pitchers", "Pos.": pos,
                                     **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
                                     XWOBA_TEAM_BACKFILL_COL:
                                         (bool(backfill_value)
                                          if pd.notna(backfill_value) else False),
                                     "player_id": pid, "bats": bio.get("bats"),
                                     "throws": bio.get("throws")})
                else:
                    bbe = (batter_stat.get(pid, {}) or {}).get("BBE")
                    bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": pos,
                                    **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                    "player_id": pid, "bats": bio.get("bats"),
                                    "throws": bio.get("throws")})

        # Block 1 = away SP vs home lineup; block 2 = home SP vs away lineup.
        pitcher_row(asp, 1, "stat", "away"); hitter_rows(home_lu, 1, "stat", "away")
        pitcher_row(hsp, 2, "stat", "home"); hitter_rows(away_lu, 2, "stat", "home")
        pitcher_row(asp, 1, "bb", "away"); hitter_rows(home_lu, 1, "bb", "away")
        pitcher_row(hsp, 2, "bb", "home"); hitter_rows(away_lu, 2, "bb", "home")

    META = ["game_pk", "game_date", "game_datetime_utc", "matchup", "away_team", "home_team",
            "away_probable_pitcher", "home_probable_pitcher", "savant_preview_url",
            "table_type", "table_index", "Name"]
    pdf = pd.DataFrame(pit_rows)
    if not pdf.empty:
        pdf = pdf[META + ["Pos.", BATTING_ORDER_COL] + STAT_COLS
                  + ["PA", XWOBA_TEAM_BACKFILL_COL, "player_id", "bats", "throws",
                     "sp_side", "is_sp"]]
    bdf = pd.DataFrame(bb_rows)
    if not bdf.empty:
        bdf = bdf[META + ["Pos.", BATTING_ORDER_COL] + BB_COLS
                  + ["BBE", "player_id", "bats", "throws", "sp_side", "is_sp"]]
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
            league_baseline[disp] = (
                float(np.average(v[m], weights=_w[m])) if m.any() else np.nan
            )
    return league_baseline


def fetch_all(slate_date):
    """Run the full cell-1 fetch. Returns a dict of everything downstream needs."""
    log(f"Pulling slate for {slate_date} ...")
    slate_df = get_slate(slate_date, SPORT_ID)
    log(f"Games: {len(slate_df)}")
    if slate_df.empty:
        return {"slate_df": slate_df, "empty": True}

    # Log game statuses (no filtering change) so postponed/suspended games are
    # visible in the build log rather than blending into the slate count.
    if "status" in slate_df.columns:
        log(f"  game statuses: {slate_df['status'].value_counts(dropna=False).to_dict()}")
        _NORMAL_STATUS = {"Scheduled", "Pre-Game", "Warmup", "In Progress",
                          "Final", "Game Over"}
        for _, gg in slate_df.iterrows():
            st = gg.get("status")
            if st not in _NORMAL_STATUS:
                log(f"  non-standard status {st!r}: {gg.get('matchup')} "
                    f"(game_pk={int(gg['game_pk'])})")

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
        league_baseline[c] = (
            float(np.average(vals, weights=wts)) if vals else np.nan
        )
    log("  league baselines:", {k: league_baseline.get(k) for k in
        ["xwOBA", "Hard Hit%", "K%", "EV", "GB%", "FB%", "Pull%"]})

    # v8 uses one fixed pseudo-sample for every batter and pitcher xwOBA,
    # including each member of the bullpen pool. Keeping the value fixed makes
    # the shrinkage reproducible across builds and retains more of the observed
    # distribution's tails than the larger v7 per-pool estimates.
    if USE_XWOBA_SHRINK:
        prior = league_baseline.get("xwOBA")
        k_bat = k_pit = XWOBA_SHRINK_K
        log(f"  xwOBA shrink: prior={prior} | K_bat={k_bat:.0f} (fixed) "
            f"| K_pit={k_pit:.0f} (fixed)")

        # Display-only percentile reference distributions (qualified regulars),
        # stashed on league_baseline so the render layer can rank a player's
        # shrunk xwOBA without re-fetching the leaderboards. Team-games is
        # estimated once from the batter pool and shared by both qualifiers,
        # since pitchers accrue BF at a different per-game rate than hitters.
        g = _est_team_games(pd.to_numeric(batter_cust.get("pa"), errors="coerce")) if batter_cust is not None else None
        qual_bat = PCTILE_QUAL_BAT * g if g else None
        qual_pit = PCTILE_QUAL_PIT * g if g else None
        ref_bat, qb = build_pctile_ref(batter_cust, prior, k_bat, qual_bat)
        ref_pit, qp = build_pctile_ref(pitcher_cust, prior, k_pit, qual_pit)
        league_baseline["_pctile_ref_bat"] = ref_bat
        league_baseline["_pctile_ref_pit"] = ref_pit
        # K-BB% reference (pitchers), raw: K-BB stabilizes fast and the model
        # does not shrink it, so it is ranked at face value over the same
        # qualified pool.
        ref_kbb = None
        pcols = set(getattr(pitcher_cust, "columns", []))
        if pitcher_cust is not None and {"k_percent", "bb_percent"} <= pcols:
            kk = pd.to_numeric(pitcher_cust["k_percent"], errors="coerce")
            bbp = pd.to_numeric(pitcher_cust["bb_percent"], errors="coerce")
            pap = pd.to_numeric(pitcher_cust.get("pa"), errors="coerce")
            msk = kk.notna() & bbp.notna()
            if qual_pit and pap is not None:
                msk_q = msk & (pap >= qual_pit)
                if int(msk_q.sum()) >= PCTILE_MIN_POOL:
                    msk = msk_q
            ref_kbb = build_pctile_ref_raw((kk - bbp)[msk])
        league_baseline["_pctile_ref_kbb"] = ref_kbb
        log(f"  pctile refs: G~{g and round(g)} | bat n="
            f"{0 if ref_bat is None else ref_bat.size} (qual~{qb and round(qb)}) | pit n="
            f"{0 if ref_pit is None else ref_pit.size} (qual~{qp and round(qp)}) | kbb n="
            f"{0 if ref_kbb is None else ref_kbb.size}")

    log("Resolving lineups (gf -> posted/partial-fill/projected) ...")
    lineups, proj_flags, lineup_ids, prob_ids = {}, [], set(), set()
    for _, g in slate_df.iterrows():
        al, ai = resolve_lineup(
            g["game_pk"], "away", g["away_team_id"], batter_stat,
            league_xwoba=league_baseline.get("xwOBA"), return_meta=True
        )
        hl, hi = resolve_lineup(
            g["game_pk"], "home", g["home_team_id"], batter_stat,
            league_xwoba=league_baseline.get("xwOBA"), return_meta=True
        )
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
            "away_savant_backfill_count": ai["savant_backfill_count"],
            "home_savant_backfill_count": hi["savant_backfill_count"],
        })
        lineup_ids.update(al); lineup_ids.update(hl)
        for c in ("away_probable_pitcher_id", "home_probable_pitcher_id"):
            if pd.notna(g[c]):
                prob_ids.add(int(g[c]))
        # A NaN probable id drops that side's block downstream; log it so every
        # game on the slate is accounted for in the build log.
        if pd.isna(g["away_probable_pitcher_id"]) or pd.isna(g["home_probable_pitcher_id"]):
            log(f"  TBD probable: {g.get('matchup')} (game_pk={int(g['game_pk'])}) — "
                f"away={g.get('away_probable_pitcher') or 'TBD'}, "
                f"home={g.get('home_probable_pitcher') or 'TBD'}")
        time.sleep(REQUEST_DELAY)
    lineup_projection_df = pd.DataFrame(proj_flags)
    os.makedirs(DATA_DIR, exist_ok=True)
    lineup_projection_df.to_csv(os.path.join(DATA_DIR, "lineup_resolution_audit.csv"), index=False)

    log(f"Loading probable-pitcher ERA over the last {RECENT_STARTS} starts ...")
    recent_start_era = load_recent_start_era(prob_ids)

    # Pitching plans are applied after the starter's own xwOBA has been shrunk
    # in build_matchup. The starter-handedness lineup composite is used only
    # for his expected innings; the neutral composite faces the bullpen.
    pitching_plans = build_pitching_plans(
        slate_df, recent_start_era, pitcher_stat, league_baseline
    )

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
        # Statcast xERA (expected ERA) from the expected-statistics leaderboard,
        # shown against season ERA on the card. NaN wherever it's unavailable.
        xera_map = load_pitcher_xera()
        pitchers_df["xERA"] = pitchers_df.apply(
            lambda r: xera_map.get(int(r["player_id"]), np.nan)
            if r.get("Pos.") == "P" and pd.notna(r.get("player_id")) else np.nan,
            axis=1)

    side_status = pd.concat([lineup_projection_df["away_lineup_status"].rename("status"),
                             lineup_projection_df["home_lineup_status"].rename("status")],
                            ignore_index=True) if not lineup_projection_df.empty else pd.Series(dtype=str)
    n_proj = int(lineup_projection_df[["away_lineup_projected", "home_lineup_projected"]].sum().sum())
    n_backfill = int(lineup_projection_df[
        ["away_savant_backfill_count", "home_savant_backfill_count"]
    ].sum().sum())
    log(f"lineup sources: {side_status.value_counts().to_dict()} of {2 * len(slate_df)} sides; "
        f"projected_or_partial={n_proj}; savant_backfills={n_backfill}")

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
        "pitching_plans": pitching_plans,
    }


# ============================================================
# CELL 4 -- STATCAST (xwOBA) MATCHUP
# ============================================================
STATCAST_RATE_COLS = ["xwOBA", "xBA", "xSLG", "Hard Hit%", "EV", "LA°", "K%", "BB%"]
WEIGHT_COL = "BBE"
USE_WEIGHTED = True
ADD_STATS = {"EV", "LA°"}

# --- xwOBA empirical-Bayes shrinkage (v5+, fixed K in v8) ------------------
# Season xwOBA (each hitter) and xwOBA-allowed (the starter) are regressed
# toward the league xwOBA baseline by sample size before they drive the lean:
#     x* = (n*x + K*prior) / (n + K)
# so a small-sample bat or a starter with few batters faced is pulled toward
# league-average rather than taken at face value. Both sides share one shrinkage
# target (the league xwOBA baseline). v8 uses K=100 for both batters and
# pitchers instead of estimating separate values from each day's leaderboards.
# That preserves more of the observed deviations from league average and makes
# the transformation stable and directly reproducible. Shrinkage touches only
# xwOBA (the lean stat); the other columns and raw per-hitter card values are
# untouched.
USE_XWOBA_SHRINK = True
XWOBA_SHRINK_COL = "xwOBA"
XWOBA_SHRINK_K = 100.0

# Pitch-mix shadow arm (pitch_arsenal.py). OFF by default and inert when on:
# it writes shadow columns for forward testing and never moves a lean, so no
# MODEL_TAG bump is involved either way. Turn it on only once
# pitch_arsenal_probe.py shows the batter x pitch-type deviation is reliable
# enough to beat the noise it injects; see that script for the budget.
USE_PITCH_MIX_SHADOW = os.environ.get("PITCH_MIX_SHADOW", "0") == "1"

# --- SP platoon-advantage xwOBA adjustment ---------------------------------
# A one-sided hitter's shrunk xwOBA is moved by a flat +/- PLATOON_XWOBA_ADJ
# depending on whether he holds the handedness edge over tonight's starter, then
# the lineup composite is taken. Applied AFTER shrinkage on purpose: shrinkage
# estimates season talent from a noisy sample, while this is a matchup term that
# carries no sample-size uncertainty of its own and must not be regressed away
# for a low-PA bat. It is a fixed constant, not a value read off the ledger, so
# no distribution shift can invalidate it.
#
# The offset is a DEVIATION FROM THE HITTER'S OWN SEASON LINE, and that line is
# already a platoon blend weighted by his real exposure to each pitcher hand.
# So the offset only applies where the season line averages over both platoon
# states:
#
#   * A switch hitter takes the opposite side in essentially every PA, so his
#     season xwOBA IS his advantage-state number. He holds the edge -- the card
#     still marks him and the platoon-OPS lens still counts him -- but adding
#     PLATOON_XWOBA_ADJ on top would count it twice. His offset is 0.
#   * A hitter with no recorded side gets 0 for the same reason the unknown-hand
#     case does: no evidence either way, so no move.
#   * A hitter whose starter's throwing hand is unknown gets 0. An absent bio is
#     not evidence of a platoon disadvantage.
#
# Known residual: a one-sided hitter's season blend is not 50/50 either. Most
# starters are right-handed, so a LHB's line is mostly measured WITH the
# advantage and a RHB's mostly without, which makes the true deviations
# asymmetric between the two rather than the +/- one constant used here. Fixing
# that needs a platoon-gap magnitude and per-hitter exposure shares; it is not
# attempted, and the flat constant stays deliberately simple.
PLATOON_XWOBA_ADJ = 0.010
PLATOON_ADV_COL = "sp_platoon_adv"     # per-hitter tag carried from the SP block
SP_THROWS_COL = "sp_throws"            # faced starter's hand, distinct from the hitter's own
OPP = {"L": "R", "R": "L"}


def effective_stand(bats, throws):
    """Side a hitter effectively bats from against a starter throwing `throws`.

    A switch hitter -- or a hitter with no recorded side -- takes the side
    opposite the pitcher. Returns None when the starter's hand is unknown,
    where the platoon question has no answer either way.
    """
    t = str(throws or "")[:1].upper()
    if t not in ("L", "R"):
        return None
    b = str(bats or "")[:1].upper()
    return b if b in ("L", "R") else OPP[t]


def platoon_advantage(bats, throws):
    """True when the hitter holds the platoon edge over this starter.

    False when he does not, None when the starter's hand is unknown. This is the
    handedness fact -- a switch hitter holds the edge -- and it drives the card's
    marker and the platoon-OPS lens. It is NOT the xwOBA offset: see
    platoon_xwoba_offset for why holding the edge and being moved by it differ.
    """
    eff = effective_stand(bats, throws)
    return None if eff is None else eff != str(throws or "")[:1].upper()


def platoon_xwoba_offset(bats, throws):
    """How far to move this hitter's season xwOBA for tonight's platoon state.

    +PLATOON_XWOBA_ADJ with the edge, -PLATOON_XWOBA_ADJ without it -- but only
    for a one-sided hitter, whose season line averages over both platoon states.
    A switch hitter's season line is already his advantage-state number, so his
    offset is 0; so is an unrecorded bats side or an unknown starter hand.
    """
    adv = platoon_advantage(bats, throws)
    if adv is None or str(bats or "")[:1].upper() not in ("L", "R"):
        return 0.0
    return PLATOON_XWOBA_ADJ if adv else -PLATOON_XWOBA_ADJ


def platoon_xwoba_offsets(g):
    """Per-hitter xwOBA offsets for a lineup group, as a positionally-indexed
    Series (aligns with the reset-index value vectors wmean consumes).

    Derived from `bats` and the faced starter's hand rather than from a stored
    number, so the tag and the offset cannot drift apart. Returns None when the
    group carries neither column, which leaves the composite untouched.
    """
    cols = getattr(g, "columns", [])
    if "bats" not in cols or SP_THROWS_COL not in cols:
        return None
    return pd.Series([platoon_xwoba_offset(b, t)
                      for b, t in zip(g["bats"], g[SP_THROWS_COL])], dtype=float)


def matchup_value(B, P, stat, L):
    if pd.isna(B) or pd.isna(P) or pd.isna(L):
        return np.nan
    if stat in ADD_STATS:
        return B + P - L
    return (B * P / L) if L else np.nan


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


# --- percentile display scale (casual redesign) -----------------------------
# Display-only. Ranks a player's *shrunk* xwOBA against a reference population
# of qualified regulars so the casual card can show a 0-100 Statcast-style
# percentile instead of a raw decimal. It reuses the model's shrink K/prior, so
# the bars are the same regressed values the lean is built on -- it never feeds
# back into the lean, MODEL_TAG, or the ledger.
#
# Reference = Savant's qualifier (2.1 PA per team-game for batters, 1.25 for
# pitchers); games played is estimated from the leaderboard's own PA spread so
# the cut scales through the season without an extra fetch. Non-qualified
# players (call-ups, platoon bats) are still *scored against* that reference by
# their shrunk value -- they land near league, filling the projected-lineup
# coverage gap a hard in/out qualifier would leave blank.
PCTILE_QUAL_BAT = 2.1      # PA per team-game
PCTILE_QUAL_PIT = 1.25     # BF per team-game
PCTILE_PA_PER_GAME = 4.3   # ~ an everyday player's PA/team-game, to back out games
PCTILE_MIN_POOL = 20       # need a real population for a stable scale


def _est_team_games(pa_series):
    """Estimate team games played from a leaderboard's PA spread: an everyday
    player's PA (~p99) ≈ PCTILE_PA_PER_GAME × games. Robust to a lone outlier."""
    pa = pd.to_numeric(pd.Series(pa_series), errors="coerce").dropna()
    if pa.empty:
        return None
    g = float(pa.quantile(0.99)) / PCTILE_PA_PER_GAME
    return g if np.isfinite(g) and g > 0 else None


def build_pctile_ref(cust, prior, k, qual_pa):
    """Sorted array of shrunk xwOBA for qualified regulars in a custom
    leaderboard. `qual_pa` is an absolute PA/BF threshold (caller derives it
    from team-games so batters and pitchers share one games estimate). Returns
    (sorted_array | None, qual_pa | None); falls back to the whole (min=1) pool
    if qualifying leaves too few for a stable scale."""
    cols = getattr(cust, "columns", [])
    if cust is None or "xwoba" not in cols or "pa" not in cols:
        return None, None
    xw = pd.to_numeric(cust["xwoba"], errors="coerce")
    pa = pd.to_numeric(cust["pa"], errors="coerce")
    base = xw.notna() & pa.notna() & (pa > 0)
    m = (base & (pa >= qual_pa)) if qual_pa else base
    if int(m.sum()) < PCTILE_MIN_POOL:      # qualifier too strict this early -> pool everyone
        m, qual_pa = base, None
    if int(m.sum()) < PCTILE_MIN_POOL:
        return None, None
    arr = np.sort(pd.to_numeric(shrink_xwoba(xw[m], pa[m], prior, k), errors="coerce")
                  .dropna().to_numpy())
    return (arr if arr.size else None), qual_pa


def build_pctile_ref_raw(series, min_n=PCTILE_MIN_POOL):
    """Sorted array of an already-final metric (e.g. K-BB%) for ranking without
    shrinkage. None when the pool is too thin."""
    v = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return np.sort(v.to_numpy()) if len(v) >= min_n else None


def pctile_rank(value, pa, ref_arr, prior, k, invert=False):
    """Percentile (0-100) of a player's *shrunk* xwOBA within ref_arr. Set
    invert for pitchers (low xwOBA-against = high percentile). None when
    unavailable."""
    if ref_arr is None or getattr(ref_arr, "size", 0) == 0 or value is None or pd.isna(value):
        return None
    s = _shrink_one(value, pa, prior, k)
    if s is None or not np.isfinite(s):
        return None
    frac = float(np.searchsorted(ref_arr, s, side="right")) / ref_arr.size
    pct = 100.0 * (1.0 - frac if invert else frac)
    return max(0.0, min(100.0, pct))


def pctile_rank_raw(value, ref_arr, invert=False):
    """Percentile of an already-final value (no shrink) within ref_arr."""
    if ref_arr is None or getattr(ref_arr, "size", 0) == 0 or value is None or pd.isna(value):
        return None
    frac = float(np.searchsorted(ref_arr, float(value), side="right")) / ref_arr.size
    pct = 100.0 * (1.0 - frac if invert else frac)
    return max(0.0, min(100.0, pct))


def segment_pitcher_blocks(df, rate_cols):
    df = coerce_numeric(df, rate_cols)

    pitcher_rows, hitter_rows = [], []
    for _, g in df.groupby(["game_pk", "table_index"], sort=False):
        cur_p = cur_side = cur_throws = None
        for _, r in g.iterrows():
            name = r.get("Name")
            # Structural detection: is_sp marks the probable-pitcher row and
            # sp_side names its side. This replaces name matching, which failed
            # when a missing bio left a `pitcher_{pid}` fallback name and could
            # collide on identical normalized names across the two probables.
            if bool(r.get("is_sp")):
                cur_p = name
                cur_side = r.get("sp_side")
                cur_throws = r.get("throws")
                pitcher_rows.append({**r.to_dict(), "pitcher_side": cur_side})
                continue
            if cur_p is None:
                continue
            bat_side = {"away": "home", "home": "away"}.get(cur_side)
            # The starter's hand rides along under its own key -- the hitter row
            # already carries a `throws` (his own). Tagging the platoon edge here
            # rather than reading it off the platoon lens keeps the xwOBA lean
            # independent of a lens that is optional and may abstain.
            hitter_rows.append({**r.to_dict(), "faced_pitcher": cur_p,
                                "pitcher_side": cur_side, "batting_side": bat_side,
                                SP_THROWS_COL: cur_throws,
                                PLATOON_ADV_COL: platoon_advantage(r.get("bats"),
                                                                   cur_throws)})

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
            # number; other columns pass through raw. A posted Savant-missing
            # hitter already carries a team aggregate, so keep that final value
            # instead of applying player-level shrinkage a second time.
            if c == XWOBA_SHRINK_COL and shrink_prior is not None and "PA" in g.columns:
                vals = shrink_xwoba(g[c], g["PA"], shrink_prior, shrink_k)
                if XWOBA_TEAM_BACKFILL_COL in g.columns:
                    backfilled = (
                        pd.Series(g[XWOBA_TEAM_BACKFILL_COL])
                        .reset_index(drop=True)
                        .fillna(False)
                        .astype(bool)
                    )
                    raw_vals = pd.to_numeric(
                        pd.Series(g[c]).reset_index(drop=True), errors="coerce"
                    )
                    vals = vals.where(~backfilled, raw_vals)
            else:
                vals = pd.to_numeric(pd.Series(g[c]).reset_index(drop=True), errors="coerce")

            # Preserve the neutral xwOBA composite before applying the
            # probable starter's handedness term. v9 uses this B_0 against the
            # bullpen and the adjusted B_SP only during the starter phase.
            if c == XWOBA_SHRINK_COL:
                rec["opp_xwOBA_neutral_mean"] = (
                    float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
                )
                rec["opp_xwOBA_neutral_wmean"] = wmean(vals, w)
                rec["opp_xwOBA_neutral"] = (
                    rec["opp_xwOBA_neutral_wmean"]
                    if weighted
                    else rec["opp_xwOBA_neutral_mean"]
                )

                # Platoon term last, on the shrunk (or backfilled) value: it is
                # a matchup fact about tonight, not a season rate to regress.
                offsets = platoon_xwoba_offsets(g)
                if offsets is not None:
                    vals = vals + offsets
            rec[f"opp_{c}_mean"] = (
                float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
            )
            rec[f"opp_{c}_wmean"] = wmean(vals, w)
            rec[f"opp_{c}"] = rec[f"opp_{c}_wmean"] if weighted else rec[f"opp_{c}_mean"]
            if c == XWOBA_SHRINK_COL:
                # Keep opp_xwOBA as a backward-compatible alias for the
                # starter-adjusted composite used by v8 and older consumers.
                rec["opp_xwOBA_vs_sp"] = rec["opp_xwOBA"]
                rec["platoon_delta_sp"] = (
                    rec["opp_xwOBA_vs_sp"] - rec["opp_xwOBA_neutral"]
                    if pd.notna(rec["opp_xwOBA_vs_sp"])
                    and pd.notna(rec["opp_xwOBA_neutral"])
                    else np.nan
                )
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
            if c == XWOBA_SHRINK_COL:
                neutral = a.get("opp_xwOBA_neutral")
                vs_sp = a.get("opp_xwOBA_vs_sp", ov)
                rec["opp_xwOBA_neutral"] = neutral
                rec["opp_xwOBA_vs_sp"] = vs_sp
                rec["platoon_delta_sp"] = a.get("platoon_delta_sp")
                # The generic opp_xwOBA remains the starter-adjusted alias.
                ov = vs_sp
            L = league_baseline.get(c, np.nan)
            rec[f"pit_{c}"] = float(pv) if pd.notna(pv) else np.nan
            rec[f"opp_{c}"] = ov
            rec[f"lg_{c}"] = L
            M = matchup_value(float(pv) if pd.notna(pv) else np.nan,
                              float(ov) if pd.notna(ov) else np.nan, c, L)
            rec[f"mx_{c}"] = float(M) if pd.notna(M) else np.nan
            rec[f"edge_{c}"] = float(M - L) if pd.notna(M) and pd.notna(L) else np.nan
            if c == XWOBA_SHRINK_COL:
                rec["starter_xwOBA"] = rec[f"pit_{c}"]
                rec["mx_xwOBA_sp"] = rec[f"mx_{c}"]
                rec["edge_xwOBA_sp"] = rec[f"edge_{c}"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["game_pk", "side"]).reset_index(drop=True)


def build_xwoba_matchup(pitchers_df, league_baseline):
    prior = league_baseline.get(XWOBA_SHRINK_COL) if USE_XWOBA_SHRINK else None
    pitcher_rows_df, opp_hitters_df = segment_pitcher_blocks(pitchers_df, STATCAST_RATE_COLS)
    opp_lineup_agg_df = aggregate_lineup(opp_hitters_df, STATCAST_RATE_COLS, weighted=USE_WEIGHTED,
                                         shrink_prior=prior, shrink_k=XWOBA_SHRINK_K)
    matchup_df = build_matchup(pitcher_rows_df, opp_lineup_agg_df, STATCAST_RATE_COLS, league_baseline,
                               shrink_prior=prior, shrink_k=XWOBA_SHRINK_K)
    return matchup_df, pitcher_rows_df, opp_hitters_df


def attach_pitch_mix_shadow(matchup_df, pitcher_rows_df, opp_hitters_df,
                            league_baseline, batter_arsenal, pitcher_arsenal):
    """Add the pitch-mix shadow columns. Never touches an existing column.

    The arm re-weights the opposing lineup by the starter's arsenal (see
    `pitch_arsenal`) and records what the starter phase *would* have been. It
    is deliberately inert: `mx_xwOBA`, `edge_xwOBA` and every column the lean
    and the ledger's `xw_net` read are untouched, so a build with the flag on
    and a build with it off produce identical leans. The columns exist so the
    arm can be forward-tested against the record before it is allowed to move
    anything -- a control in the ledger, not a UI element.

    Sides with no arsenal coverage carry a multiplier of exactly 1.0 and a
    shadow phase equal to the real one, which is the honest reading of "no
    information", and keeps the column defined for every row.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    out = matchup_df.copy()
    lg_type = pitch_arsenal.league_by_pitch_type(batter_arsenal)
    cells, overall = pitch_arsenal.batter_relative_cells(batter_arsenal, lg_type)
    L = league_baseline.get(XWOBA_SHRINK_COL, np.nan)

    pit_ids = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, r in pitcher_rows_df.iterrows():
            pid = pd.to_numeric(r.get("player_id"), errors="coerce")
            if pd.notna(pid):
                pit_ids[(r["game_pk"], r["Name"])] = int(pid)

    # Batting order is the row order inside each (game_pk, faced_pitcher)
    # block, so the slot-PA weights line up with the composite's own weighting.
    lineups = {}
    if opp_hitters_df is not None and not opp_hitters_df.empty:
        for key, grp in opp_hitters_df.groupby(["game_pk", "faced_pitcher"],
                                               sort=False):
            rows = []
            for slot, (_, r) in enumerate(grp.iterrows(), start=1):
                pid = pd.to_numeric(r.get("player_id"), errors="coerce")
                rows.append((int(pid) if pd.notna(pid) else None,
                             LINEUP_SLOT_PA.get(slot, LINEUP_SLOT_PA[9])))
            lineups[key] = rows

    mults, covs, ns, bases = [], [], [], []
    for _, row in out.iterrows():
        key = (row["game_pk"], row["pitcher"])
        weights, basis = pitch_arsenal.pitcher_pa_weights(
            pitcher_arsenal, pit_ids.get(key))
        mult, cov, n = pitch_arsenal.lineup_mix_multiplier(
            lineups.get(key, []), weights, cells, overall)
        mults.append(mult)
        covs.append(cov)
        ns.append(n)
        bases.append(basis)
    out["mix_mult"] = mults
    out["mix_coverage"] = covs
    out["mix_hitters"] = ns
    out["mix_basis"] = bases

    b_sp = pd.to_numeric(out.get("opp_xwOBA_vs_sp"), errors="coerce")
    starter = pd.to_numeric(out.get("starter_xwOBA"), errors="coerce")
    out["opp_xwOBA_mix"] = b_sp * out["mix_mult"]
    mx = [matchup_value(b, p, "xwOBA", L)
          for b, p in zip(out["opp_xwOBA_mix"], starter)]
    out["mx_xwOBA_sp_mix"] = mx
    out["edge_xwOBA_sp_mix"] = [
        (m - L) if pd.notna(m) and pd.notna(L) else np.nan for m in mx]
    return out


def sequential_xwoba_phases(b_neutral, b_vs_sp, starter, bullpen, league,
                            expected_sp_ip, sp_bf_per_ip=None, bp_bf_per_ip=None):
    """Return the v10 starter, bullpen, and workload-weighted matchup phases.

    The starter-handedness lineup adjustment belongs only to the starter
    phase. The bullpen phase is neutral until bullpen handedness is modeled.
    A missing bullpen deliberately leaves the full-game matchup undefined.

    The phases are averaged on their share of **plate appearances**, not of
    innings. Both phase values are per-PA xwOBA rates, so the innings share
    v9 used was the wrong denominator: a pitcher who allows more baserunners
    faces more batters per inning, which means the innings share systematically
    underweights the starter exactly when he is the one being hit. Measured
    BF/IP for each phase converts innings to PA. When either rate is
    unavailable the weight falls back to the innings share, which is the same
    number whenever the two rates are equal -- the degradation is continuous,
    not a threshold.
    """
    q = np.nan
    if expected_sp_ip is not None and pd.notna(expected_sp_ip):
        ip_sp = float(np.clip(float(expected_sp_ip), 0.0, GAME_INNINGS))
        ip_bp = GAME_INNINGS - ip_sp
        r_sp, r_bp = _f(sp_bf_per_ip), _f(bp_bf_per_ip)
        if r_sp is not None and r_bp is not None and r_sp > 0 and r_bp > 0:
            pa_sp, pa_bp = ip_sp * r_sp, ip_bp * r_bp
            q = pa_sp / (pa_sp + pa_bp) if (pa_sp + pa_bp) > 0 else np.nan
        else:
            q = ip_sp / GAME_INNINGS
    bp_share = 1.0 - q if pd.notna(q) else np.nan

    mx_sp = matchup_value(b_vs_sp, starter, "xwOBA", league)
    mx_bp = matchup_value(b_neutral, bullpen, "xwOBA", league)
    mx = (
        q * mx_sp + bp_share * mx_bp
        if pd.notna(q) and pd.notna(mx_sp) and pd.notna(mx_bp)
        else np.nan
    )
    return {
        "sp_share": q,
        "bp_share": bp_share,
        "mx_xwOBA_sp": float(mx_sp) if pd.notna(mx_sp) else np.nan,
        "edge_xwOBA_sp": (
            float(mx_sp - league)
            if pd.notna(mx_sp) and league is not None and pd.notna(league)
            else np.nan
        ),
        "mx_xwOBA_bp": float(mx_bp) if pd.notna(mx_bp) else np.nan,
        "edge_xwOBA_bp": (
            float(mx_bp - league)
            if pd.notna(mx_bp) and league is not None and pd.notna(league)
            else np.nan
        ),
        "mx_xwOBA": float(mx) if pd.notna(mx) else np.nan,
        "edge_xwOBA": (
            float(mx - league)
            if pd.notna(mx) and league is not None and pd.notna(league)
            else np.nan
        ),
    }


def apply_pitching_plans(matchup_df, pitching_plans, league_baseline):
    """Apply v9's sequential starter/bullpen construction to each side.

    `build_matchup` has already shrunk the starter and computed M_SP. This step
    computes M_BP with the neutral lineup, then combines matchup values by
    expected innings. `pit_xwOBA` remains a workload-weighted pitching
    diagnostic for legacy consumers; it does not drive the v9 lean.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    out = matchup_df.copy()
    for col, default in (
        ("bullpen_xwOBA", np.nan),
        ("expected_sp_ip", np.nan),
        ("sp_share", np.nan),
        ("bp_share", np.nan),
        ("mx_xwOBA_sp", np.nan),
        ("edge_xwOBA_sp", np.nan),
        ("mx_xwOBA_bp", np.nan),
        ("edge_xwOBA_bp", np.nan),
        ("bullpen_pitchers", np.nan),
        ("bullpen_relief_bf", np.nan),
        ("sp_bf_per_ip", np.nan),
        ("bp_bf_per_ip", np.nan),
        ("opener", False),
        ("opener_reason", None),
        ("opener_confidence", None),
        ("pitching_basis", "starter_only_no_fullgame_lean"),
    ):
        if col not in out.columns:
            out[col] = default

    league = _f((league_baseline or {}).get("xwOBA"))
    for idx, row in out.iterrows():
        key = (int(row["game_pk"]), row["side"])
        plan = (pitching_plans or {}).get(key) or {}
        starter = _f(row.get("starter_xwOBA"))
        if starter is None:
            starter = _f(row.get("pit_xwOBA"))
        bullpen = _f(plan.get("bullpen_xwOBA"))
        expected_ip = _f(plan.get("expected_sp_ip"))
        b_vs_sp = _f(row.get("opp_xwOBA_vs_sp"))
        if b_vs_sp is None:
            b_vs_sp = _f(row.get("opp_xwOBA"))
        b_neutral = _f(row.get("opp_xwOBA_neutral"))
        if b_neutral is None:
            # Legacy/unit-test rows predate the dual composite. Production v9
            # always supplies B_0; this fallback means "no platoon delta."
            b_neutral = b_vs_sp

        out.at[idx, "starter_xwOBA"] = starter
        out.at[idx, "bullpen_xwOBA"] = bullpen
        out.at[idx, "expected_sp_ip"] = expected_ip
        out.at[idx, "bullpen_pitchers"] = plan.get("bullpen_pitchers")
        out.at[idx, "bullpen_relief_bf"] = plan.get("bullpen_relief_bf")
        out.at[idx, "opener"] = bool(plan.get("opener"))
        out.at[idx, "opener_reason"] = plan.get("opener_reason")
        out.at[idx, "opener_confidence"] = plan.get("opener_confidence")
        basis = plan.get("pitching_basis")
        if starter is None or bullpen is None or expected_ip is None:
            basis = "starter_only_no_fullgame_lean"
        elif not basis:
            basis = (
                "opener_bullpen_sequential"
                if bool(plan.get("opener"))
                else "starter_bullpen_sequential"
            )
        out.at[idx, "pitching_basis"] = basis

        sp_rate = _f(plan.get("sp_bf_per_ip"))
        bp_rate = _f(plan.get("bp_bf_per_ip"))
        out.at[idx, "sp_bf_per_ip"] = sp_rate
        out.at[idx, "bp_bf_per_ip"] = bp_rate
        phases = sequential_xwoba_phases(
            b_neutral, b_vs_sp, starter, bullpen, league, expected_ip,
            sp_bf_per_ip=sp_rate, bp_bf_per_ip=bp_rate,
        )
        for col, value in phases.items():
            out.at[idx, col] = value

        # A single blended pitching rate is not the v9 matchup input because
        # B_SP and B_0 differ. Keep it only as a diagnostic when both phases
        # are valid; never leave the starter value looking like a nine-inning
        # fallback when the bullpen is unavailable.
        if starter is not None and bullpen is not None and pd.notna(phases["sp_share"]):
            out.at[idx, "pit_xwOBA"] = (
                phases["sp_share"] * starter + phases["bp_share"] * bullpen
            )
        else:
            out.at[idx, "pit_xwOBA"] = np.nan
    return out


# ============================================================
# CELL 5 -- PLATOON / HANDEDNESS SPLIT MATCHUP
# ============================================================
MIN_SPLIT_PA = 30
MIN_PITCHER_SPLIT_BF = 50
K_BAT = 100
K_PIT = 200
K0 = 200
# OPP / effective_stand / platoon_advantage live in CELL 4 -- one handedness
# convention shared by the xwOBA adjustment and this lens.


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
        # NaN is truthy, so a bare `if Lcell`/`Loverall` would let an empty
        # (early-season) league cell propagate a NaN prior; guard explicitly.
        Lcell_ok = pd.notna(Lcell) and Lcell
        Loverall_ok = pd.notna(Loverall) and Loverall
        if overall is not None and not pd.isna(overall) and Loverall_ok:
            op = overall_pa or 0
            overall_reg = (op * overall + K0 * Loverall) / (op + K0)
            prior = overall_reg * (Lcell / Loverall) if Lcell_ok else overall_reg
        else:
            prior = Lcell if Lcell_ok else Loverall
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
                eff = effective_stand(b, T)
                buckets[(eff, T)].append((ops, pa)); allv.append((ops, pa))
        Lc = {}
        for k, v in buckets.items():
            if v:
                o = np.array([x[0] for x in v]); w = np.array([x[1] for x in v], float)
                Lc[k] = float(np.average(o, weights=w))
        overall = (float(np.average([x[0] for x in allv],
                                    weights=[x[1] for x in allv]))
                   if allv else np.nan)
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
        eff_stand = effective_stand(bats, T)
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
            "eff_stand": eff_stand, "platoon_adv": platoon_advantage(bats, T),
            "ops_vs_hand_raw": B_raw, "ops_vs_hand": float(B) if pd.notna(B) else np.nan,
            "pit_ops_allowed_raw": P_raw,
            "pit_ops_allowed": float(P) if pd.notna(P) else np.nan,
            "lg_cell": L, "mx_ops": float(Mi) if pd.notna(Mi) else np.nan,
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
            "opp_OPS_raw": float(opp_ops_raw) if pd.notna(opp_ops_raw) else np.nan,
            "opp_OPS_vs_hand": float(opp_ops) if pd.notna(opp_ops) else np.nan,
            "pit_OPS_raw": float(pit_ops_raw) if pd.notna(pit_ops_raw) else np.nan,
            "pit_OPS_allowed": float(pit_ops) if pd.notna(pit_ops) else np.nan,
            "mx_OPS": float(mx_ops) if pd.notna(mx_ops) else np.nan,
            "edge_OPS": float(edge) if pd.notna(edge) else np.nan,
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


def fetch_current_streaks(slate_df=None):
    """team_id -> current streak code such as 'W3' or 'L1'.

    Display-only and strictly best-effort: any failure logs and returns an
    empty map so the card header simply omits the streak rather than failing
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
        log(f"current streaks: standings fetch failed ({e!r}) -> header omits")
        return out
    for grp in js.get("records", []):
        for tr in grp.get("teamRecords", []):
            tid = (tr.get("team") or {}).get("id")
            if tid is None:
                continue
            code = str((tr.get("streak") or {}).get("streakCode") or "").upper()
            if not re.fullmatch(r"[WL]\d+", code):
                continue
            try:
                out[int(tid)] = code
            except (TypeError, ValueError):
                continue
    log(f"current streaks attached: {len(out)} teams")
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

# Compact scoreboard labels. The schedule feed supplies official full names,
# while a slate row reads more cleanly with the familiar club name beside the
# cap logo (and the abbreviation remains the fallback for degraded/test data).
TEAM_LABELS = {
    "Arizona Diamondbacks": "D-backs",
    "Athletics": "Athletics",
    "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros",
    "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins",
    "New York Mets": "Mets",
    "New York Yankees": "Yankees",
    "Philadelphia Phillies": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres",
    "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners",
    "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Washington Nationals": "Nationals",
}

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
    return "rgba(var(--warm-tx),1)" if edge * ksign >= 0 else "rgba(var(--cool-tx),1)"


def f3(v):
    return "—" if v is None else f"{v:.3f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.3f}"


def delta3(v):
    """Three-place delta display without rendering a nonzero value as zero."""
    if v is None or pd.isna(v):
        return "—"
    value = abs(float(v))
    if value > 0 and f"{value:.3f}" == "0.000":
        return "&lt;0.001"
    return f"{value:.3f}"


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
    """Bare wall clock for a game row -- no zone suffix.

    Every clock on the page is Pacific and the page says so once, in the
    legend, rather than repeating ` PT` on every card. The build stamp dropped
    its suffix too -- one zone, stated once, is the whole convention."""
    return dt.strftime("%I:%M %p").lstrip("0")


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


def _hitters_for(opp_hitters_df, detail_df, gpk, fp, lg_ops,
                 ref_bat=None, prior=None, k_bat=None):
    """Batting-order rows for the lineup this SP faces. Row order in
    opp_hitters_df IS lineup order -- never sort. Detail (platoon) join is by
    (game_pk, faced_pitcher, batter name); a hitter with no vs-hand data keeps
    the xwOBA cell and renders em-dashes for the platoon columns.

    Each row also carries xw_pctile: the hitter's season xwOBA, shrunk by his
    PA and ranked against the qualified-regular reference (display-only).

    The platoon marker reads the hitter row's own `sp_platoon_adv` tag -- the
    same tag that moved his xwOBA in the lean -- so the card still marks the
    adjusted bats on a build where the platoon-OPS lens abstained."""
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
        xw_raw = _f(r.get("xwOBA"))
        backfill_value = r.get(XWOBA_TEAM_BACKFILL_COL)
        team_backfill = bool(backfill_value) if pd.notna(backfill_value) else False
        adv_tag = r.get(PLATOON_ADV_COL)
        if adv_tag is None or not pd.notna(adv_tag):
            adv_tag = d["platoon_adv"] if d is not None else None
        rows.append(dict(
            name=r["Name"], player_id=r.get("player_id"),
            pos=str(r.get("Pos.") or ""),
            bats=(str(r.get("bats") or ""))[:1].upper(),
            xw=xw_raw,
            xw_team_backfill=team_backfill,
            xw_pctile=(
                pctile_rank_raw(xw_raw, ref_bat)
                if team_backfill
                else pctile_rank(xw_raw, _f(r.get("PA")), ref_bat, prior, k_bat)
            ),
            adv=bool(adv_tag) if adv_tag is not None and pd.notna(adv_tag) else False,
            ops=_f(d["ops_vs_hand"]) if d is not None else None,
            pa=int(d["split_pa"] or 0) if d is not None else 0,
            low=bool(d["low_sample"]) if d is not None else False,
            mx=mx, edge=edge,
        ))
    return rows


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _last(name):
    """Surname for compact spotlight pills: the last name token, skipping a
    trailing generational suffix so 'Fernando Tatis Jr.' -> 'Tatis' and
    'Michael Harris II' -> 'Harris' rather than 'Jr.'/'II'."""
    parts = str(name or "").split()
    while len(parts) > 1 and parts[-1].lower().strip(".") in _NAME_SUFFIXES:
        parts = parts[:-1]
    return parts[-1] if parts else str(name or "")


def _mlb_player_link(name, player_id, label=None):
    """Official MLB player-page link, falling back to plain text without an ID.

    MLB's stable ID-only route redirects to the current canonical name slug, so
    trades, accents, suffixes, and name changes do not require URL maintenance.
    """
    text = _esc(name if label is None else label)
    try:
        pid = int(player_id)
    except (TypeError, ValueError, OverflowError):
        return text
    return (f"<a class='player-link' href='https://www.mlb.com/player/{pid}' "
            f"target='_blank' rel='noopener noreferrer'>{text}</a>")


def _tier_word(pctile):
    """(label, css-class) plain-word starter quality from an xwOBA-against
    percentile. Class picks the accent: elite=steel, below=ember, else neutral."""
    if pctile is None:
        return None, ""
    if pctile >= 75:
        return "elite", "elite"
    if pctile >= 45:
        return "solid", "solid"
    return "below avg", "below"


def _pct_bar(pctile, kind):
    """Percentile slider (0-100). kind 'h' (hitter, ember) or 'p' (pitcher,
    steel). Fill length is the percentile; hue carries whose favor it is."""
    if pctile is None:
        return "<span class='sl na'></span><span class='pn'>—</span>"
    w = max(0.0, min(100.0, float(pctile)))
    return (f"<span class='sl {kind}'><i style='width:{w:.0f}%'></i></span>"
            f"<span class='pn'>{round(w)}</span>")


def _spotlight_html(hitters, n=3, thresh=70):
    """Top bats in a lineup by xwOBA percentile -- the 'highlight the leaders'
    ask, on a quality stat. Shows up to n at/above thresh; empty otherwise."""
    ranked = sorted((h for h in hitters if h.get("xw_pctile") is not None),
                    key=lambda h: h["xw_pctile"], reverse=True)
    top = [h for h in ranked[:n] if h["xw_pctile"] >= thresh]
    if not top:
        return ""
    pills = "".join(f"<span class='pill'>"
                    f"{_mlb_player_link(h['name'], h.get('player_id'), _last(h['name']))} "
                    f"<b>{round(h['xw_pctile'])}</b></span>" for h in top)
    return f"<div class='spot'><span class='sl-lab'>Standouts</span>{pills}</div>"


def _grade_word(edge):
    """Qualitative grade of an offense's xwOBA edge vs league, for the read."""
    if edge is None:
        return None, ""
    if edge >= 0.020:
        return "well above average", "warmtx"
    if edge >= 0.006:
        return "above average", "warmtx"
    if edge > -0.006:
        return "about average", ""
    if edge > -0.020:
        return "below average", "cooltx"
    return "well below average", "cooltx"


def _read_sentence(away_abbr, home_abbr, a, h, fav, strength_word):
    """One plain-language line under the header explaining the lean. a = away
    SP row, so a['xw_edge'] is the HOME offense edge and h -> AWAY offense."""
    home_edge, away_edge = a.get("xw_edge"), h.get("xw_edge")
    if home_edge is None or away_edge is None or fav is None:
        return ""
    # The away offense faces the HOME starter (h['p']); the home offense faces
    # the AWAY starter (a['p']). Edges were resolved to offenses above.
    ga, ca = _grade_word(away_edge)
    gh, ch = _grade_word(home_edge)
    aw = f"<b class='{ca}'>{ga}</b>" if ca else f"<b>{ga}</b>"
    hw = f"<b class='{ch}'>{gh}</b>" if ch else f"<b>{gh}</b>"
    home_sp = _mlb_player_link(h["p"], h.get("player_id"), _last(h["p"]))
    away_sp = _mlb_player_link(a["p"], a.get("player_id"), _last(a["p"]))
    away_team = _esc(away_abbr)
    home_team = _esc(home_abbr)
    return (f"<p class='read'>{away_team}'s bats grade {aw} against "
            f"{home_sp}; {home_team}'s grade {hw} against "
            f"{away_sp}. That is a <b>{strength_word or 'lean'}</b> "
            f"lean to <b>{_esc(fav)}</b>.</p>")


def _market_fav(odds, away_abbr, home_abbr):
    """Market favorite abbr from devigged home prob, else raw MLs. None if unknown."""
    if not odds:
        return None
    ph = odds.get("p_home")
    if ph is not None and pd.notna(ph):
        return home_abbr if ph >= 0.5 else away_abbr
    hm, am = odds.get("home_ml"), odds.get("away_ml")
    if hm is not None and am is not None:
        return home_abbr if float(hm) < float(am) else away_abbr
    return None


def _verdict_html(fav, odds, away_abbr, home_abbr, ctx=None):
    """Model-vs-market verdict row. Highlights the disagreement case (model on
    the underdog) -- the only bettable signal -- and stays muted on agreement.
    `ctx` is market_context_records(): where available it tails each verdict
    with the xwOBA lean's historical record in this exact spot (lean side ×
    agree/disagree), else falls back to prose. Rendered as its own full-width
    row under the odds so the prose never squeezes the four market numbers."""
    ctx = ctx or {}
    mkt = _market_fav(odds, away_abbr, home_abbr)
    if fav is None or mkt is None:
        return ("<div class='verdict'><div class='l'>Model vs market</div>"
                "<div class='vt'>No market yet.</div></div>")
    side = "home" if fav == home_abbr else "away"
    if fav == mkt:
        rec = ctx.get((side, "agree"))
        tail = (f"When the model leans the {side} favorite: <b>{rec}</b> in the ledger."
                if rec else "No edge on the line here.")
        return ("<div class='verdict'><div class='l'>Model vs market</div>"
                f"<div class='vt'>Model agrees with the market: "
                f"{_esc(fav)} favored. "
                f"{tail}</div></div>")
    price = (odds.get("home_ml") if fav == home_abbr else odds.get("away_ml"))
    px = f" ({_fmt_ml(price)})" if price is not None else ""
    rec = ctx.get((side, "disagree"))
    tail = (f"When the model leans the {side} underdog: <b>{rec}</b> in the ledger."
            if rec else "This is the spot the record is built to test.")
    return ("<div class='verdict edge'><div class='l'>Model vs market · disagree</div>"
            f"<div class='vt'>Model leans the underdog "
            f"<b>{_esc(fav)}{px}</b> against the "
            f"market's {_esc(mkt)}. {tail}</div></div>")


def _hitter_row_html(i, hr):
    """One batting-order row: name + Statcast percentile bar, raw xwOBA, and the
    ◆ platoon-advantage marker."""
    nm_text = _esc(hr["name"])
    nm_link = _mlb_player_link(hr["name"], hr.get("player_id"))
    b = f"<span class='b'>{hr['bats']}</span>" if hr["bats"] else ""
    adv = ("<span class='adv mach' title='platoon advantage vs this SP'>◆</span>"
           if hr["adv"] else "")
    bar = _pct_bar(hr.get("xw_pctile"), "h")
    backfill = bool(hr.get("xw_team_backfill"))
    xw_title = " title='team-average backfill: player absent from Savant'" if backfill else ""
    xw_mark = "<span class='pa'>*</span>" if backfill else ""
    xw_c = f"<td class='r mach'{xw_title}>{f3(hr['xw'])}{xw_mark}</td>"
    return (f"<tr><td class='ord'>{i}</td>"
            f"<td class='nm' title='{nm_text}'>{nm_link}{adv}{b}</td>"
            f"<td class='pos'>{_esc(hr['pos'])}</td>"
            f"<td class='pct r'>{bar}</td>{xw_c}</tr>")


def _lineup_details(side_d):
    st = side_d["lu_status"]
    st_lab = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, st or "—")
    st_cls = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, "projected")
    parts = []
    if side_d["opp_xw"] is not None:
        parts.append(f"xwOBA {f3(side_d['opp_xw'])}")
    summ = " · ".join(parts) if parts else ""
    head = ("<tr><th></th><th class='nm'>Hitter</th><th>Pos</th><th class='r'>xwOBA %ile</th>"
            "<th class='r mach'>xwOBA</th></tr>")
    body = "".join(_hitter_row_html(i + 1, hr) for i, hr in enumerate(side_d["hitters"]))
    if not body:
        body = "<tr><td class='na' colspan='5'>lineup unavailable</td></tr>"
    return (
        "<details class='lineup' open>"
        "<summary><span class='chev'>▶</span>"
        f"<span class='tl'>{_esc(side_d['opp_abbr'])} lineup</span>"
        f"<span class='st {st_cls}'>{st_lab}</span>"
        f"<span class='lw mach'>{summ}</span></summary>"
        f"<div class='lu-scroll'><table class='lu'>{head}{body}</table></div></details>")


def _sp_stat_cell(lab, val, fmt, sub=None, heat=""):
    s = f"<div class='s'>{sub}</div>" if sub else ""
    st = f" style='{heat}'" if heat else ""
    return (f"<div class='stat'{st}><div class='l'>{lab}</div>"
            f"<div class='v'>{fmt(val)}</div>{s}</div>")


def _side_html(sp_abbr, d, league_baseline):
    badge = f"<span class='hand'>{d['t']}HP</span>" if d["t"] in ("L", "R") else ""
    has_fullgame = "bullpen_sequential" in str(d.get("pitching_basis") or "")
    opener = ("<span class='flag warn' title='Opener: the model estimates this "
              "pitcher’s workload. The full-game lean also requires a valid "
              "role-filtered bullpen aggregate.'>"
              f"{'opener · bullpen' if has_fullgame else 'opener'}</span>") \
              if d.get("is_opener") else ""
    comp = (f"{d['R']}R/{d['L']}L" + (f"/{d['S']}S" if d["S"] else "")) if d["has_pl"] else "—"
    padv = f" · {d['padv']} plt-adv" if d["has_pl"] else ""
    lb = league_baseline or {}
    # lg K%/BB% come from the PA-weighted *batter* leaderboard; league K% and
    # BB% are symmetric (every batter K/BB is a pitcher K/BB), so they are
    # valid pitcher references and lg K-BB% = lg K% - lg BB%.
    lg = {k: _f(lb.get(k)) for k in ("xwOBA", "K%", "BB%", "ERA")}
    kbb = (d["pit_k"] - d["pit_bb"]) if (d["pit_k"] is not None
                                         and d["pit_bb"] is not None) else None
    lg_kbb = (lg["K%"] - lg["BB%"]) if (lg["K%"] is not None
                                        and lg["BB%"] is not None) else None
    # xERA (Statcast expected ERA) shown against the starter's own season ERA;
    # tinted vs league ERA (higher xERA = more offense-favorable = warm).
    xera_sub = (f"season {f2(d['era_season'])}"
                if d.get("era_season") is not None else None)
    stats = (
        _sp_stat_cell("xwOBA agn", d["pit_xw"], f3,
                      f"lg {f3(lg['xwOBA'])}" if lg["xwOBA"] is not None else None,
                      heat=heat_style(d["pit_xw"], lg["xwOBA"], HEAT_DOMAINS["xwOBA_sp"]))
        + _sp_stat_cell("K-BB%", kbb, f1,
                        f"lg {f1(lg_kbb)}" if lg_kbb is not None else None,
                        heat=heat_style(kbb, lg_kbb, HEAT_DOMAINS["K-BB%"], hi="cool"))
        + _sp_stat_cell("xERA", d.get("xera"), f2, xera_sub,
                        heat=heat_style(d.get("xera"), lg["ERA"], HEAT_DOMAINS["ERA"])))
    tier_lab, tier_cls = _tier_word(d.get("pit_xw_pctile"))
    tier = f"<span class='tier {tier_cls}'>{tier_lab}</span>" if tier_lab else ""
    bars = (f"<div class='spct'><span class='lab'>xwOBA</span>{_pct_bar(d.get('pit_xw_pctile'), 'p')}</div>"
            f"<div class='spct'><span class='lab'>K − BB%</span>{_pct_bar(d.get('kbb_pctile'), 'p')}</div>")
    exp_ip = _f(d.get("expected_sp_ip"))
    pitching_note = (
        f" · model {exp_ip:.1f} IP + bullpen"
        if exp_ip is not None and has_fullgame
        else " · starter only; no full-game lean"
        if d.get("pitching_basis") == "starter_only_no_fullgame_lean"
        else ""
    )
    sp_team = _esc(sp_abbr)
    opp_team = _esc(d["opp_abbr"])
    return (
        f"<section class='side'>"
        f"<div class='matchlab'>{sp_team} starter → {opp_team} bats</div>"
        f"<div class='sp'><div class='who'><span class='nm'>"
        f"{_mlb_player_link(d['p'], d.get('player_id'))}</span>{badge}{tier}{opener}</div>"
        f"<div class='role'>faces the {opp_team} lineup"
        f"<span class='mach'> · {comp}{padv}{pitching_note}</span></div>"
        f"<div class='sp-bars'>{bars}</div>"
        # `.sp` closes here. It was previously left open, which nested the
        # spotlight and the whole lineup table inside the starter block and let
        # `.sp .nm` (0,2,0) outrank `table.lu td` (0,1,2) -- every hitter name
        # rendered at the starter's 16px/700 instead of the table's 12.5px/400.
        f"<div class='spstats mach'>{stats}</div></div>"
        f"{_spotlight_html(d['hitters'])}"
        f"{_lineup_details(d)}"
        f"</section>")


def _market_html(o, away_abbr, home_abbr, fav=None, ctx=None):
    """Odds strip: the four market numbers share one horizontal row at every
    width (`.modds`), with the verdict on its own full-width row beneath. No
    per-card build stamp -- the page header carries the build time once, and
    every card on the page is built from the same fetch."""
    o = o or {}
    def _mlcell(prefix, lab, cur, opn):
        sub = f" <span class='mv'>← {_fmt_ml(opn)} open</span>" if opn is not None else ""
        return (f"<div class='mcell'><div class='l'>{prefix} · "
                f"{_esc(lab)}</div>"
                f"<div class='v'>{_fmt_ml(cur)}{sub}</div></div>")
    tot = f"o/u {o['total']:g}" if o.get("total") is not None else "—"
    ph = f"{o['p_home'] * 100:.1f}%" if o.get("p_home") is not None else "—"
    return (
        "<div class='market'><div class='modds'>"
        + _mlcell("DK ML", away_abbr, o.get("away_ml"), o.get("open_away_ml"))
        + _mlcell("DK ML", home_abbr, o.get("home_ml"), o.get("open_home_ml"))
        + f"<div class='mcell'><div class='l'>Total</div><div class='v'>{tot}</div></div>"
        + f"<div class='mcell'><div class='l'>Implied "
        + _esc(home_abbr)
        + f" (devig)</div><div class='v'>{ph}</div></div>"
        + "</div>"
        + _verdict_html(fav, o, away_abbr, home_abbr, ctx)
        + "</div>")


def _score_text(score):
    """Whole-run score text, or None when the schedule has not supplied one."""
    value = pd.to_numeric(score, errors="coerce")
    return str(int(value)) if pd.notna(value) else None


def _team_meta_span(g, side):
    """Render the pregame streak, replaced by the score for live/final games."""
    streak = g.get(f"{side}_streak")
    state = str(g.get("abstract_state") or "").lower()
    data = f" data-streak='{_esc(streak)}'" if streak else ""
    if state in ("live", "final"):
        score = _score_text(g.get(f"{side}_score"))
        title = "live score" if state == "live" else "final score"
        text = score if score is not None else "—"
        return (f"<span class='team-meta score {side}'{data} title='{title}'>"
                f"{text}</span>")
    if not streak:
        return f"<span class='team-meta streak {side}' data-streak='' hidden></span>"
    return (f"<span class='team-meta streak {side}'{data} "
            f"title='current win/loss streak'>{_esc(streak)}</span>")


def _game_state_span(g):
    """LIVE/FINAL header badge; hidden before first pitch."""
    state = str(g.get("abstract_state") or "").lower()
    raw_detail = g.get("status")
    detail = "" if raw_detail is None or pd.isna(raw_detail) else str(raw_detail)
    if state not in ("live", "final"):
        return "<span class='game-state' hidden></span>"
    label = "LIVE"
    if state == "live":
        inning = _inning_indicator(g)
        if inning:
            label += f" · {inning}"
    else:
        label = "FINAL"
    title = f" title='{_esc(detail)}'" if detail else ""
    return f"<span class='game-state {state}'{title}>{label}</span>"


def _base_out_text(bases, outs):
    """Screen-reader sentence for the diamond, e.g. '2 out, runners on first
    and second'. The glyph itself is aria-hidden, so this is the only reading
    of the state; an unknown out count degrades to the runners alone."""
    named = [n for n, on in zip(("first", "second", "third"), bases) if on]
    if not named:
        runners = "bases empty"
    elif len(named) == 3:
        runners = "bases loaded"
    elif len(named) == 1:
        runners = f"runner on {named[0]}"
    else:
        runners = f"runners on {' and '.join(named)}"
    if outs is None:
        return runners
    return f"{outs} out, {runners}"


def _base_out_html(g):
    """Diamond + out dots for the collapsed row, as in a scoreboard app.

    Always emitted so score_refresh_js can fill it in place, and `hidden`
    unless the game is live -- a base-out state means nothing before first
    pitch or after the last out. The three bases carry `on` when occupied and
    the dots fill up to the out count; both are aria-hidden decorations over
    `_base_out_text`."""
    state = str(g.get("abstract_state") or "").lower()
    bases = tuple(bool(g.get(k)) for k in ("on_first", "on_second", "on_third"))
    outs_val = pd.to_numeric(g.get("outs"), errors="coerce")
    outs = int(outs_val) if pd.notna(outs_val) else None
    live = state == "live"
    cls = ["b1", "b2", "b3"]
    diamond = "".join(
        f"<i class='{c}{' on' if on else ''}'></i>" for c, on in zip(cls, bases))
    dots = "".join(
        f"<i class='{'on' if outs is not None and i < outs else ''}'></i>"
        for i in range(3))
    return (f"<span class='bo'{'' if live else ' hidden'}>"
            f"<span class='diamond' aria-hidden='true'>{diamond}</span>"
            f"<span class='outs' aria-hidden='true'>{dots}</span>"
            f"<span class='bo-sr'>{_esc(_base_out_text(bases, outs))}</span>"
            "</span>")


def _inning_indicator(g):
    """Live inning as ▲2nd / ▼2nd; empty when the feed has no inning yet."""
    ordinal = g.get("current_inning_ordinal")
    if ordinal is None or pd.isna(ordinal):
        inning = pd.to_numeric(g.get("current_inning"), errors="coerce")
        if pd.isna(inning):
            return ""
        n = int(inning)
        suffix = "th" if 10 < n % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        ordinal = f"{n}{suffix}"
    half = str(g.get("inning_half") or "").lower()
    if half == "top":
        marker = "▲"
    elif half == "bottom":
        marker = "▼"
    else:
        top = g.get("is_top_inning")
        marker = ("▲" if bool(top) else "▼") if top is not None and not pd.isna(top) else ""
    return f"{marker}{_esc(ordinal)}"


def _logo_assets(games):
    """Resolve cap logos for every team on the slate.

    Returns ({team_id: (light_uri, dark_uri)}, <style-block>). Each data URI is
    emitted exactly once as a per-team CSS custom property (`--lgl`/`--lgd`); a
    single pair of theme-scoped rules switches the visible variant, mirroring the
    same prefers-color-scheme / data-theme selectors the palette vars use so the
    logo tracks the manual Theme toggle. Teams whose fetch missed are omitted and
    their cards keep the text abbreviation."""
    ids = set()
    for g in games:
        for k in ("away_team_id", "home_team_id"):
            v = g.get(k)
            if v is not None and pd.notna(v):
                try:
                    ids.add(int(v))
                except (TypeError, ValueError):
                    continue
    assets = {}
    for tid in sorted(ids):
        lt, dk = team_logo_uris(tid)
        if lt and dk:
            assets[tid] = (lt, dk)
    if not assets:
        return assets, ""
    base = (".clogo{display:inline-block;width:1.05em;height:1.05em;flex:none;"
            "vertical-align:-.16em;margin-right:.30em;background-size:contain;"
            "background-repeat:no-repeat;background-position:center;"
            "background-image:var(--lgl)}")
    per_team = "".join(
        f'.t{tid}{{--lgl:url("{lt}");--lgd:url("{dk}")}}'
        for tid, (lt, dk) in assets.items())
    dark = ('@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) '
            '.clogo{background-image:var(--lgd)}}'
            'html[data-theme="dark"] .clogo{background-image:var(--lgd)}')
    return assets, f"<style>{base}{per_team}{dark}</style>"


def _club_logo(team_id, ctx):
    """Decorative cap-logo chip for a team abbr, or '' when the logo did not
    resolve (keeping the text abbreviation as the sole label). aria-hidden since
    the adjacent abbreviation already names the team."""
    ids = (ctx or {}).get("logo_ids")
    if not ids or team_id is None or (isinstance(team_id, float) and pd.isna(team_id)):
        return ""
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return ""
    return f"<span class='clogo t{tid}' aria-hidden='true'></span>" if tid in ids else ""


def _summary_team_label(g, side):
    """Familiar team label for the scoreboard row, with abbreviation fallback."""
    official = g.get(f"{side}_team_name")
    if official is not None and not pd.isna(official):
        official = str(official)
        return TEAM_LABELS.get(official, official)
    return str(g.get(f"{side}_abbr") or "")


def _summary_market_line(g):
    """Favorite and current moneyline for the compact row."""
    odds = g.get("odds") or {}
    p_home = _f(odds.get("p_home"))
    if p_home is None:
        return "Odds pending"
    side = "home" if p_home >= .5 else "away"
    ml = odds.get(f"{side}_ml")
    if ml is None:
        return "Odds pending"
    return f"{g.get(f'{side}_abbr') or ''} {_fmt_ml(ml)}".strip()


def _summary_team_html(g, side, ctx):
    abbr = str(g.get(f"{side}_abbr") or "")
    label = _summary_team_label(g, side)
    logo = _club_logo(g.get(f"{side}_team_id"), ctx)
    if not logo:
        logo = (f"<span class='summary-logo-fallback' aria-hidden='true'>"
                f"{_esc(abbr[:1])}</span>")
    return (
        f"<span class='summary-team {side}' data-team-abbr='{_esc(abbr)}'>"
        f"{logo}<span class='summary-club'>{_esc(label)}</span>"
        f"{_team_meta_span(g, side)}</span>"
    )


def _scoreboard_summary(g, ctx=None):
    """Collapsed scoreboard row shared by modeled and pending games."""
    state = str(g.get("abstract_state") or "").lower()
    in_progress = state in ("live", "final")
    hide_pregame = " hidden" if in_progress else ""
    time_text = str(g.get("time_pt") or "Time TBD")
    market_text = _summary_market_line(g)
    game_no = (f"<span class='game-no'>{_esc(g['game_label'])}</span>"
               if g.get("game_label") else "")
    away_label = _summary_team_label(g, "away")
    home_label = _summary_team_label(g, "home")
    return (
        f"<summary class='game-summary' aria-label='{_esc(away_label)} at "
        f"{_esc(home_label)}; expand game breakdown'>"
        f"<span class='teams' data-game-pk='{int(g['game_pk'])}'>"
        f"{_summary_team_html(g, 'away', ctx)}"
        f"<span class='summary-center'>{_game_state_span(g)}"
        f"{_base_out_html(g)}"
        f"<span class='summary-time'{hide_pregame}>{_esc(time_text)}</span>"
        f"{game_no}<span class='summary-market'{hide_pregame}>"
        f"{_esc(market_text)}</span></span>"
        f"{_summary_team_html(g, 'home', ctx)}</span>"
        "<span class='summary-chevron' aria-hidden='true'>⌄</span>"
        "</summary>"
    )


def _detail_context_html(g):
    when = " · ".join(x for x in (g.get("time_pt"), g.get("venue")) if x)
    return f"<div class='detail-context'>{_esc(when)}</div>" if when else ""


def cmb_card(g, strength_scale=None, ctx=None):
    if g.get("unavailable"):
        probables = " vs ".join(
            "TBD" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
            for x in (g.get("away_probable"), g.get("home_probable")))
        return (
            "<article class='card unavailable'>"
            "<details class='game-card'>"
            + _scoreboard_summary(g, ctx)
            + "<div class='game-detail'>"
            + _detail_context_html(g)
            + "<div class='card-note'><b>Awaiting paired probable pitchers</b>"
            f"<span>{_esc(probables)} · matchup lean will appear after both starters are listed.</span></div>"
            "</div></details></article>")

    a, h = g["away"], g["home"]
    away_abbr, home_abbr = g["away_abbr"], g["home_abbr"]
    # a = away SP -> his xw_edge is the HOME offense edge. When either edge is
    # missing there is no defined lean and no read sentence. The strength word
    # ranks |Δxw| against the ledger's lean-magnitude history (display-only)
    # and appears once, in the plain-language read beneath the header.
    fav = strength_word = read_html = None
    if a["xw_edge"] is not None and h["xw_edge"] is not None:
        home_off, away_off = a["xw_edge"], h["xw_edge"]
        net = home_off - away_off
        if net != 0:
            delta = abs(net)
            fav = home_abbr if net > 0 else away_abbr
            strength_word, _ = lean_strength(delta, strength_scale)
            read_html = _read_sentence(away_abbr, home_abbr, a, h, fav, strength_word)
    return (
        "<article class='card'>"
        "<details class='game-card'>"
        + _scoreboard_summary(g, ctx)
        + "<div class='game-detail'>"
        + _detail_context_html(g)
        + (read_html or "")
        + _market_html(g.get('odds'), away_abbr, home_abbr, fav, ctx)
        + f"<div class='sides'>{_side_html(away_abbr, a, g['league_baseline'])}"
        + f"{_side_html(home_abbr, h, g['league_baseline'])}</div>"
        + "</div></details></article>")


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


def build_combined(games, strength_scale=None, ctx=None):
    cards = sorted(games, key=_game_order_key)
    return ("<div class='grid'>"
            + "".join(cmb_card(g, strength_scale, ctx) for g in cards)
            + "</div>")


def _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                          opp_hitters_df=None, detail_df=None, lg_ops=None,
                          slate_df=None, lineup_df=None,
                          league_baseline=None, odds=None, streaks=None):
    streaks = streaks or {}

    def _streak(team_id):
        try:
            return streaks.get(int(team_id))
        except (TypeError, ValueError):
            return None

    throws = {}
    player_ids = {}
    recent_era = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, pr in pitcher_rows_df.iterrows():
            key = (pr["game_pk"], pr["Name"])
            throws[key] = pr.get("throws")
            if pd.notna(pr.get("player_id")):
                player_ids[key] = int(pr["player_id"])
            recent_era[key] = (
                _f(pr.get("ERA_L5")), int(pr.get("ERA_L5_GS") or 0),
                _f(pr.get("ERA_SEASON")), _f(pr.get("xERA")))
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

    # Percentile reference distributions are stashed on league_baseline in
    # fetch_all. Missing on pre-market/degraded builds -> every pctile_rank
    # returns None and the bars fall back to em-dashes.
    _lb0 = league_baseline or {}
    _prior = _f(_lb0.get("xwOBA"))
    _k_bat = XWOBA_SHRINK_K
    _ref_bat = _lb0.get("_pctile_ref_bat")
    _ref_pit = _lb0.get("_pctile_ref_pit")
    _ref_kbb = _lb0.get("_pctile_ref_kbb")

    games = []
    for gpk, gg in xw_df.groupby("game_pk", sort=False):
        a, h = _rows_by_side(gg)
        if a is None or h is None:
            miss = ", ".join(s for s, v in (("away SP", a), ("home SP", h)) if v is None)
            matchup = gg["matchup"].iloc[0] if "matchup" in gg.columns and len(gg) else "?"
            log(f"  game skipped from paired cards — missing {miss}: "
                f"{matchup} (game_pk={gpk})")
            continue
        srow = slate_map.get(gpk)

        def mk(r):
            side = r["side"]
            t = throws.get((gpk, r["pitcher"]), "")
            opp_lu_side = "home" if side == "away" else "away"
            starter_xw = _f(r.get("starter_xwOBA"))
            if starter_xw is None:
                starter_xw = _f(r.get("pit_xwOBA"))
            d = dict(p=r["pitcher"],
                     player_id=player_ids.get((gpk, r["pitcher"])),
                     t=t if t in ("L", "R") else "",
                     opp=r["opp_team"], opp_abbr=_abbr(r["opp_team"]),
                     pit_xw=starter_xw, pit_k=_f(r.get("pit_K%")),
                     pit_bb=_f(r.get("pit_BB%")),
                     era_season=recent_era.get((gpk, r["pitcher"]), (None, 0, None, None))[2],
                     xera=recent_era.get((gpk, r["pitcher"]), (None, 0, None, None))[3],
                     opp_xw=(
                         _f(r.get("opp_xwOBA_vs_sp"))
                         if _f(r.get("opp_xwOBA_vs_sp")) is not None
                         else _f(r.get("opp_xwOBA"))
                     ),
                     xw_edge=_f(r.get("edge_xwOBA")),
                     # Display percentiles (0-100). pit_xwOBA is already shrunk
                     # in build_matchup, so it is ranked directly (invert: low
                     # xwOBA-against = high pct); K-BB% is ranked raw.
                     pit_xw_pctile=pctile_rank_raw(starter_xw, _ref_pit, invert=True),
                     kbb_pctile=pctile_rank_raw(
                         (_f(r.get("pit_K%")) - _f(r.get("pit_BB%")))
                         if (_f(r.get("pit_K%")) is not None and _f(r.get("pit_BB%")) is not None)
                         else None, _ref_kbb),
                     lu_status=(lu_map.get(gpk) or {}).get(opp_lu_side),
                     is_opener=bool(r.get("opener")),
                     expected_sp_ip=_f(r.get("expected_sp_ip")),
                     bullpen_xw=_f(r.get("bullpen_xwOBA")),
                     pitching_basis=r.get("pitching_basis"),
                     has_pl=False, R=0, L=0, S=0, padv=0,
                     pl_sp=None, pl_sp_raw=None, pl_mx=None, pl_edge=None,
                     pl_reliable=False,
                     hitters=_hitters_for(opp_hitters_df, detail_df,
                                          gpk, r["pitcher"], lg_ops_f,
                                          _ref_bat, _prior, _k_bat))
            pr = pl_map.get((gpk, side))
            if pr is not None:
                if not d["t"]:
                    pt_ = pr.get("throws"); d["t"] = pt_ if pt_ in ("L", "R") else ""
                d.update(has_pl=True,
                         R=int(pr.get("n_RHB") or 0), L=int(pr.get("n_LHB") or 0),
                         S=int(pr.get("n_SW") or 0), padv=int(pr.get("n_platoon_adv") or 0),
                         pl_sp=_f(pr.get("pit_OPS_allowed")), pl_sp_raw=_f(pr.get("pit_OPS_raw")),
                         pl_mx=_f(pr.get("mx_OPS")), pl_edge=_f(pr.get("edge_OPS")),
                         pl_reliable=bool(pr.get("reliable")))
            return d

        away_abbr = (srow.get("away_abbrev") if srow is not None else None) or _abbr(h["opp_team"])
        home_abbr = (srow.get("home_abbrev") if srow is not None else None) or _abbr(a["opp_team"])
        game_number = pd.to_numeric(srow.get("game_number"), errors="coerce") if srow is not None else np.nan
        is_dh = (str(srow.get("double_header") or "N") != "N") if srow is not None else False
        game_label = f"G{int(game_number)}" if pd.notna(game_number) and (is_dh or game_number > 1) else ""
        games.append(dict(
            away=mk(a), home=mk(h),
            away_abbr=away_abbr, home_abbr=home_abbr,
            away_team_name=(srow.get("away_team") if srow is not None else None),
            home_team_name=(srow.get("home_team") if srow is not None else None),
            away_team_id=(srow.get("away_team_id") if srow is not None else None),
            home_team_id=(srow.get("home_team_id") if srow is not None else None),
            away_streak=_streak(srow.get("away_team_id")) if srow is not None else None,
            home_streak=_streak(srow.get("home_team_id")) if srow is not None else None,
            abstract_state=(srow.get("abstract_state") if srow is not None else None),
            status=(srow.get("status") if srow is not None else None),
            away_score=(srow.get("away_score") if srow is not None else None),
            home_score=(srow.get("home_score") if srow is not None else None),
            current_inning=(srow.get("current_inning") if srow is not None else None),
            current_inning_ordinal=(
                srow.get("current_inning_ordinal") if srow is not None else None
            ),
            inning_half=(srow.get("inning_half") if srow is not None else None),
            is_top_inning=(srow.get("is_top_inning") if srow is not None else None),
            on_first=(srow.get("on_first") if srow is not None else None),
            on_second=(srow.get("on_second") if srow is not None else None),
            on_third=(srow.get("on_third") if srow is not None else None),
            outs=(srow.get("outs") if srow is not None else None),
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
                away_team_name=srow.get("away_team"),
                home_team_name=srow.get("home_team"),
                away_team_id=srow.get("away_team_id"),
                home_team_id=srow.get("home_team_id"),
                away_streak=_streak(srow.get("away_team_id")),
                home_streak=_streak(srow.get("home_team_id")),
                abstract_state=srow.get("abstract_state"),
                status=srow.get("status"),
                away_score=srow.get("away_score"),
                home_score=srow.get("home_score"),
                current_inning=srow.get("current_inning"),
                current_inning_ordinal=srow.get("current_inning_ordinal"),
                inning_half=srow.get("inning_half"),
                is_top_inning=srow.get("is_top_inning"),
                on_first=srow.get("on_first"),
                on_second=srow.get("on_second"),
                on_third=srow.get("on_third"),
                outs=srow.get("outs"),
                time_pt=_game_time_pt(srow.get("game_datetime_utc")),
                venue=str(srow.get("venue") or ""),
                away_probable=srow.get("away_probable_pitcher"),
                home_probable=srow.get("home_probable_pitcher"),
            ))
    return games


def _legend_head(model_label, built_txt):
    """Slim title bar shown above the cards; the how-to-read guide moved to
    the bottom of the page (_legend_guide) so the cards lead."""
    date = SLATE_DATE
    head = model_label
    if built_txt:
        head += f" · built {built_txt}"
    elif date:
        head += f" · {date}"
    return f"<div class='legend'><div class='lg-title'>{head}</div></div>"


def _legend_guide():
    return (
        "<div class='legend'>"
        "<div class='lg-lead'><b>How to read a card:</b> the <b>lean</b> names the model's "
        "favored side; <b>slight / clear / strong</b> shows its relative strength. "
        "It estimates matchup advantage, not the final score.</div>"
        "<div class='lg-keys'>"
        "<span class='k'><i class='sw warm'></i>warmer / longer bar = better for hitters</span>"
        "<span class='k'><i class='sw cool'></i>cooler / longer bar = better for the pitcher</span>"
        "<span class='k'><b>xwOBA %ile</b> league rank from 0–100; higher is better</span>"
        "</div>"
        "<div class='lg-notes'>"
        "<span><b>Standouts</b> are the lineup's highest-ranked bats.</span>"
        "<span><b>Starter quality</b> uses xwOBA allowed and K−BB%; higher is better.</span>"
        "<span><b>Model vs market</b> shows whether the lean agrees with the DK favorite; "
        "disagree means the model favors the underdog.</span>"
        "<span><b>◆</b> marks a platoon advantage over the starter.</span>"
        "<span>On a live game the <b>diamond</b> shows runners on base and the "
        "dots below it the outs.</span>"
        "<span class='wide'>DraftKings moneylines via ESPN at build time; first pitch "
        "times are Pacific; cards are ordered by first pitch.</span>"
        "</div></div>")


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
# (family, file, weight range). Both are variable fonts subset to the glyphs
# this site actually emits -- ASCII, Latin-1 and Latin Extended-A (the roster
# carries Jose, Pena, Suarez, Munoz...) plus the exact symbol set grepped out
# of this module. Full latin subsets were 35KB/31KB; these are 27KB/28KB.
_FONT_FACES = (
    ("Archivo", "archivo-subset.woff2", "100 900"),
    ("JetBrains Mono", "jetbrains-mono-subset.woff2", "400 800"),
)


def font_face_css():
    """`@font-face` rules with the woff2 payloads inlined as data URIs.

    The page loads no external resources -- logos are already inlined the same
    way -- and keeping the fonts local holds that. A missing or unreadable file
    degrades to the system stack behind it in `--sans`/`--mono` rather than
    failing the build: the type gets plainer, the numbers still line up, and
    the slate still ships. Licences sit next to the files in assets/fonts/
    (both OFL).
    """
    out = []
    for family, fname, wght in _FONT_FACES:
        path = os.path.join(FONT_DIR, fname)
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as e:  # noqa: BLE001
            log(f"font {fname} unavailable, falling back to system stack: {e!r}")
            continue
        out.append(
            f"@font-face{{font-family:'{family}';font-style:normal;"
            f"font-weight:{wght};font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2-variations')}}")
    return "".join(out)


CSS = r"""/* ============================================================
   MLB matchup leans -- box-score / scout-sheet tokens
   ============================================================ */
:root{
  --bg:#f3f5f7; --surface:#ffffff; --surface-2:#eef1f4; --ink:#161b20;
  --muted:#4f5a65; --faint:#636e7b; --line:#dde3e8; --line-2:#eceff2;
  /* --faint is a text colour, not a hairline: column headers, stat labels,
     the `lg .312` / `season 4.04` subs, the machinery line, the opening-price
     tail and the `projected` badge all use it, all below 12px. It must clear
     4.5:1 against BOTH --surface and --surface-2 (the market strip and the
     stat cells sit on --surface-2). The previous #98a4af measured 2.54:1 on
     white and 2.24:1 on the strip; #5f6c78 dark measured 3.15:1. --muted moved
     with it (5.54 -> 7.04:1 on white) so the two-step text hierarchy survives
     the darkening instead of collapsing into one tone. Re-measure both
     surfaces before changing either -- tests/test_casual_redesign.py checks
     the ratios and the ordering. */
  --warm:198,84,44;             /* ember  -- offense-favorable  */
  --cool:52,116,168;            /* steel  -- pitcher-favorable  */
  --lean:176,124,16;            /* amber  -- lean pill / links  */
  /* Text-only twins of the three accents. The accents above were picked as
     fills and borders, but they are also set as `color` on the tier pills,
     the LIVE badge, the read-line emphasis, the verdict label and the ledger
     badges -- all 9-13.5px, and most of them sitting on a .08-.12 tint of
     their own hue, which lifts the background toward the text. Measured in
     headless Chromium at 390px over the composited tint: warm 3.94:1 on the
     `.B` strip and 4.01:1 on its own tier tint, cool 3.90:1 under the LIVE
     badge, lean 3.12:1 on the amber verdict -- all short of 4.5:1, in light
     only (dark passed everywhere, so there the twins alias the accents).
     These are the same hue and saturation darkened in HLS until every
     background each one lands on -- both bare surfaces and a .08/.10/.12
     wash of its own hue over either -- clears 4.5:1. The .12 case is the
     binding one; solving against .08 alone left the ledger W/L badges at
     3.86:1. Use them wherever an accent
     is the `color`; keep the base tokens for fills, borders and bar hues so
     the percentile bars and legend swatches do not shift.
     tests/test_casual_redesign.py measures both. */
  --warm-tx:164,69,36;
  --cool-tx:46,103,149;
  --lean-tx:130,92,12;
  --amberbg:246,196,86;
  --chip-fg:#1b1e25;
  /* Archivo (text) and JetBrains Mono (every number on the page) are inlined
     by font_face_css() as subset variable woff2. The system stacks stay behind
     them as the fallback if a font file ever goes missing. Archivo is a
     grotesque with enough width and x-height to stay legible at the label
     sizes; JetBrains Mono was picked for its digits -- slashed zero, a 1 with
     a foot, unmistakable 6/8/9 -- because scores, odds, xwOBA and percentiles
     are the whole point of the page and they were rendering in whatever
     generic mono the OS happened to supply. */
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --sans:"Archivo",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  --shadow:0 1px 2px rgba(16,18,29,.05),0 10px 26px -20px rgba(16,18,29,.28);
  /* Corner-radius scale -- three tiers, no raw px radii anywhere else.
     --r      containers: things that hold other content (card, stat strip,
              table wrapper, button)
     --r-s    inline marks: badges, flags, status tags, percentile bars,
              legend swatches -- anything sitting inside a line of text
     --r-pill true pills: fully-rounded ends, used only where the shape is
              the point
     tests/test_casual_redesign.py enforces this; add a tier here rather
     than a one-off px value at the call site. */
  --r:6px;
  --r-s:3px;
  --r-pill:999px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#7f8b97; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  /* Dark already clears 4.5:1 on every accent-as-text pair, so the text
     twins alias the accents rather than introducing a second palette. */
  --warm-tx:236,122,72; --cool-tx:96,158,208; --lean-tx:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}}
html[data-theme="dark"]{
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#7f8b97; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  /* Dark already clears 4.5:1 on every accent-as-text pair, so the text
     twins alias the accents rather than introducing a second palette. */
  --warm-tx:236,122,72; --cool-tx:96,158,208; --lean-tx:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
/* The `hidden` attribute must win over any component `display`. `.game-state`
   set `display:inline-block`, which outranks the UA sheet's `[hidden]` rule, so
   the empty pregame badge rendered as a bare bordered chip above the time on
   every unstarted card -- and score_refresh_js re-hides it the same way. */
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 var(--sans);
  -webkit-font-smoothing:antialiased;padding:20px 16px 60px}
.mx-wrap{max-width:1060px;margin:0 auto}

.topbar{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:12px}
.brand{font:800 18px/1 var(--sans);letter-spacing:.13em;text-transform:uppercase}
.theme{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--muted);
  font:600 14px/1 var(--sans);padding:7px 11px;border-radius:var(--r);cursor:pointer}
.theme:hover{color:var(--ink)}
.theme:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}

.legend{margin:2px 2px 14px}
.lg-title{font-size:14.5px;font-weight:650;letter-spacing:.01em;margin-bottom:7px}
.lg-title em{font-style:normal;color:var(--muted);font-weight:500}
.lg-lead{font-size:13px;line-height:1.5;color:var(--muted);max-width:82ch;margin-bottom:9px}
.lg-lead b{color:var(--ink);font-weight:700}
.lg-keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted)}
.lg-keys .k{display:inline-flex;align-items:center;gap:6px}
.lg-notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 18px;
  margin-top:8px;font-size:13px;line-height:1.4;color:var(--faint)}
.lg-notes b{color:var(--muted);font-weight:700}
.lg-notes i{font-style:normal;color:var(--muted)}
.lg-notes .wide{grid-column:1/-1}
.sw{width:11px;height:11px;border-radius:var(--r-s);display:inline-block}
.sw.warm{background:rgba(var(--warm),.85)} .sw.cool{background:rgba(var(--cool),.85)}

/* One continuous scoreboard, not a stack of separate cards. The slate is a
   single surface and games are divided by a hairline -- the list reads top to
   bottom in one pass instead of as a dozen boxed objects, which is what an
   8px gap plus a border plus a shadow per game was doing. The container keeps
   the radius, the border and the shadow; the game gives all three up. `clip`
   rounds the first and last summary backgrounds off at the corners. */
.grid{display:flex;flex-direction:column;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow);overflow:clip}
.card{background:transparent;border:0;border-radius:0;box-shadow:none}
.card + .card{border-top:1px solid var(--line)}

/* ---------- collapsed scoreboard row ---------- */
.game-card{margin:0}
.game-summary{position:relative;display:flex;align-items:center;min-height:94px;
  padding:12px 16px;list-style:none;cursor:pointer;user-select:none}
.game-summary::-webkit-details-marker{display:none}
.game-summary::marker{content:""}
.game-summary:hover{background:var(--surface-2)}
.game-summary:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:-2px}
.game-card[open]>.game-summary{background:var(--surface-2);
  border-bottom:1px solid var(--line-2)}
.teams{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,.68fr) minmax(0,1fr);
  align-items:center;gap:14px;width:100%;min-width:0;padding-right:22px}
.summary-team{display:grid;align-items:center;gap:8px;min-width:0}
.summary-team.away{grid-template-columns:46px minmax(0,1fr) auto;
  grid-template-areas:"logo club meta"}
.summary-team.home{grid-template-columns:auto minmax(0,1fr) 46px;
  grid-template-areas:"meta club logo"}
.summary-team .clogo,.summary-logo-fallback{grid-area:logo}
.summary-team .clogo{width:46px;height:46px;margin:0;vertical-align:middle}
.summary-logo-fallback{display:grid;width:46px;height:46px;place-items:center;
  border:1px solid var(--line);border-radius:var(--r-pill);background:var(--surface-2);
  color:var(--muted);font:800 16.5px/1 var(--sans)}
.summary-club{grid-area:club;min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font:750 16.5px/1.15 var(--sans)}
.summary-team.home .summary-club{text-align:right}
.summary-center{display:flex;min-width:0;flex-direction:column;align-items:center;
  justify-content:center;gap:4px;text-align:center}
.summary-time{font:800 18px/1 var(--mono);font-variant-numeric:tabular-nums}
.summary-market{font:600 13px/1.1 var(--mono);color:var(--muted);
  font-variant-numeric:tabular-nums}
.summary-chevron{position:absolute;right:13px;top:50%;color:var(--faint);
  font:800 18px/1 var(--sans);transform:translateY(-50%);
  transition:transform .16s ease,color .16s ease}
.game-card[open]>.game-summary .summary-chevron{
  color:var(--ink);transform:translateY(-50%) rotate(180deg)}
.teams .team-meta{font:600 13px/1 var(--mono);color:var(--muted);letter-spacing:0;
  grid-area:meta;margin:0;vertical-align:middle;white-space:nowrap}
/* The score is the one thing a scoreboard exists to show, so it carries
   the largest type on the row -- roughly the club name doubled. It sits
   in the meta slot under the club, so the row grows to fit rather than
   anything having to move. */
.teams .score{font-size:28px;font-weight:800;color:var(--ink);
  line-height:1.05;letter-spacing:-.01em}
.game-no{font:700 13px/1 var(--mono);color:var(--muted);vertical-align:middle}
.game-state{display:inline-block;padding:3px 6px;border:1px solid var(--line);
  border-radius:var(--r-s);font:750 12px/1 var(--sans);letter-spacing:.08em;
  vertical-align:middle;color:var(--muted);background:var(--surface-2)}
.game-state.live{color:rgba(var(--cool-tx),1);border-color:rgba(var(--cool),.35);
  background:rgba(var(--cool),.10)}

/* Base-out state: three bases on a diamond over three out dots, the scoreboard
   convention. The bases are 7px squares rotated 45deg and placed on a 3-column
   grid -- second on the top row, third and first flanking below -- so the
   cluster reads as a diamond without any glyph font or image. Occupied bases
   and recorded outs fill solid; the rest stay hairline outlines. */
.bo{display:flex;flex-direction:column;align-items:center;gap:2px}
/* Diamond geometry: the bases are 6px squares rotated 45deg, so the facing
   edges of two of them are 6px apart when their centres are 6px apart along
   the 45deg diagonal. A 5.5px cell step puts adjacent centres 7.8px apart --
   6px of base plus a ~1.8px gap, tight enough to read as one cluster without
   the squares merging into a bar. The squares overflow the grid box by design;
   the padding keeps that overflow off the badge above. */
.diamond{display:grid;flex:none;grid-template-columns:repeat(3,5.5px);
  grid-template-rows:repeat(2,5.5px);place-items:center;padding:3px 2px}
.diamond i{width:6px;height:6px;flex:none;border:1px solid var(--faint);
  transform:rotate(45deg)}
.diamond .b2{grid-area:1/2}.diamond .b3{grid-area:2/1}.diamond .b1{grid-area:2/3}
.diamond i.on{background:rgba(var(--lean),1);border-color:rgba(var(--lean),1)}
.outs{display:flex;align-items:center;gap:3px}
.outs i{width:5px;height:5px;flex:none;border:1px solid var(--faint);
  border-radius:var(--r-pill)}
.outs i.on{background:var(--ink);border-color:var(--ink)}
/* The glyph is decorative; _base_out_text carries the state for readers. */
.bo-sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;
  border:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.detail-context{padding:7px 16px;border-bottom:1px solid var(--line-2);
  color:var(--muted);font:500 13px/1.3 var(--sans)}
.card-note{display:flex;flex-direction:column;gap:4px;padding:14px 16px 16px;color:var(--muted)}
.card-note b{color:var(--ink);font-size:14.5px}.card-note span{font-size:14px}

/* ---------- market strip ---------- */
/* Two stacked rows: the four odds numbers, then the verdict. .modds never
   wraps -- the cells share the width evenly and shrink together, so the strip
   reads as one horizontal line of numbers at phone widths too. */
.market{border-bottom:1px solid var(--line-2);background:var(--surface-2)}
.modds{display:flex}
.mcell{flex:1 1 0;min-width:0;display:flex;flex-direction:column;
  justify-content:space-between;padding:7px 16px;border-right:1px solid var(--line-2)}
.mcell:last-child{border-right:0}
.mcell .l,.verdict .l{font:600 12px/1.4 var(--sans);letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}
.mcell .v{font:500 14.5px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.mcell .v .mv{color:var(--faint);font-size:13px}

/* ---------- two sides ---------- */
.sides{display:grid;grid-template-columns:1fr 1fr}
.side{padding:12px 16px 14px;min-width:0}
.side + .side{border-left:1px solid var(--line-2)}
/* Side-by-side only once a column can hold the lineup table without an
   inner horizontal scroll. Measured in headless Chromium: the table
   overflows its column by 62px at 762px wide, 23px at 840px, 3px at 880px
   and 0 from ~900px up. 760px put every card on the slate into a hidden
   side-scroll across the whole tablet range. */
@media (max-width:900px){
  .sides{grid-template-columns:1fr}
  .side + .side{border-left:0;border-top:1px solid var(--line-2)}
}

/* SP block */
.sp .who{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.sp .nm{font:700 18px/1.2 var(--sans)}
.player-link{color:inherit;text-decoration:underline;
  text-decoration-color:rgba(var(--lean),.5);text-decoration-thickness:1px;
  text-underline-offset:2px}
.player-link:hover{color:rgba(var(--lean-tx),1);text-decoration-color:currentColor}
.player-link:focus-visible{
  outline:2px solid rgba(var(--lean),1);outline-offset:2px}
.hand{font:500 12.5px/1.4 var(--mono);color:var(--muted);border:1px solid var(--line);border-radius:var(--r-s);padding:0 4px}
.sp .role{font-size:13px;color:var(--faint);margin-top:1px}
.spstats{display:flex;margin-top:8px;border:1px solid var(--line-2);border-radius:var(--r);overflow:hidden}
.stat{flex:1;padding:6px 8px 5px;border-right:1px solid var(--line-2);min-width:0}
.stat:last-child{border-right:0}
.stat .l{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
.stat .v{font:500 16px/1.3 var(--mono);font-variant-numeric:tabular-nums}
.stat .s{font:400 12.5px/1.3 var(--mono);color:var(--muted)}
/* Pills on the starter's `.who` line -- the tier word and the flags -- share
   one type treatment so they read as one family. Size, tracking, case and box
   metrics live here and nowhere else; every rule below adds colour only. The
   `.tier` accents sit further down with the percentile bars. Spacing comes
   from `.sp .who`'s 8px gap: no pill carries its own margin. */
.tier,.flag{font:600 12px/1.4 var(--sans);letter-spacing:.06em;text-transform:uppercase;
  border-radius:var(--r-s);padding:1px 6px;white-space:nowrap}
.flag{color:var(--muted);border:1px solid var(--line)}
.flag.warn{color:rgba(var(--warm-tx),1);border-color:rgba(var(--warm),.45)}
.flag.mute{color:var(--faint)}

/* ---------- lineup: order rail + shared-axis edge bars ---------- */
details.lineup{margin-top:11px;border-top:1px solid var(--line-2)}
details.lineup summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:8px;
  padding:8px 0 6px;user-select:none}
details.lineup summary::-webkit-details-marker{display:none}
details.lineup summary:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}
summary .tl{font:700 13px/1.4 var(--sans);letter-spacing:.12em;text-transform:uppercase}
summary .st{font:600 12.5px/1.4 var(--sans);letter-spacing:.08em;border-radius:var(--r-s);padding:1px 6px}
summary .st.posted{color:rgba(var(--cool-tx),1);border:1px solid rgba(var(--cool),.45)}
summary .st.partial{color:rgba(var(--lean-tx),1);border:1px solid rgba(var(--amberbg),.5)}
summary .st.projected{color:var(--faint);border:1px solid var(--line)}
summary .lw{margin-left:auto;font:500 13px/1.4 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
summary .chev{font-size:12.5px;color:var(--faint);transition:transform .12s}
details[open] summary .chev{transform:rotate(90deg)}
@media (prefers-reduced-motion:reduce){summary .chev{transition:none}}

.lu-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.lu{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
table.lu th{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  text-align:right;padding:2px 6px 4px;border-bottom:1px solid var(--line-2);white-space:nowrap}
table.lu td{font:400 14px/1.4 var(--mono);text-align:right;padding:4px 6px;
  border-bottom:1px solid var(--line-2);white-space:nowrap}
table.lu tr:last-child td{border-bottom:0}
td.ord{font-size:13px;color:var(--faint);width:26px;text-align:center;border-right:1px solid var(--line-2)}
td.pos{color:var(--muted);font-size:12.5px}
td .pa{color:var(--faint);font-size:12.5px}
td.na{color:var(--faint)}
tr.low td{color:var(--muted)}

td.bar{width:86px;padding:4px 8px 4px 2px}

/* ============================================================
   Card layer: percentile bars, plain-language read, verdict, and the
   full model machinery (always shown).
   ============================================================ */
.matchlab{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);margin-bottom:6px}
.read{margin:0;padding:12px 16px;font:500 16px/1.5 var(--sans);color:var(--ink);
  border-bottom:1px solid var(--line-2)}
.read b{font-weight:700}
.read .warmtx{color:rgba(var(--warm-tx),1)} .read .cooltx{color:rgba(var(--cool-tx),1)}

/* percentile slider (fill length = percentile; hue = whose favor) */
.sl{display:inline-block;width:88px;height:9px;border-radius:var(--r-s);background:var(--surface-2);
  border:1px solid var(--line-2);vertical-align:middle;overflow:hidden;position:relative}
.sl i{display:block;height:100%;border-radius:var(--r-s) 0 0 var(--r-s)}
.sl.h i{background:rgba(var(--warm),.85)} .sl.p i{background:rgba(var(--cool),.85)}
.pn{font:600 13px/1 var(--mono);margin-left:8px;font-variant-numeric:tabular-nums;color:var(--muted)}

/* starter percentile bars + quality tier */
.sp-bars{margin-top:9px}
.spct{display:flex;align-items:center;gap:9px;margin-top:6px}
.spct .lab{font:600 12px/1.4 var(--sans);letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);min-width:64px}
/* `.tier` type + box metrics are shared with `.flag` up in the SP block. */
.tier.elite{color:rgba(var(--cool-tx),1);border:1px solid rgba(var(--cool),.5);background:rgba(var(--cool),.10)}
.tier.solid{color:var(--muted);border:1px solid var(--line)}
.tier.below{color:rgba(var(--warm-tx),1);border:1px solid rgba(var(--warm),.5);background:rgba(var(--warm),.08)}

/* spotlight (top bats by percentile) */
.spot{display:flex;flex-wrap:wrap;align-items:baseline;gap:5px 8px;margin:12px 0 2px}
.spot .sl-lab{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.spot .pill{font:600 13px/1.3 var(--mono);color:var(--ink);background:var(--surface-2);
  border:1px solid var(--line-2);border-radius:var(--r-pill);padding:2px 9px;font-variant-numeric:tabular-nums}
.spot .pill b{color:rgba(var(--warm-tx),1)}

/* model-vs-market verdict — its own full-width row beneath the odds, at every
   width. The prose grows downward instead of stealing width from the numbers. */
.verdict{padding:7px 16px;border-top:1px solid var(--line-2);border-left:3px solid var(--line)}
.verdict .l{color:var(--muted)} .verdict .vt{font:600 14px/1.4 var(--sans);color:var(--muted);margin-top:2px}
.verdict.edge{border-left-color:rgba(var(--lean),1);background:rgba(var(--amberbg),.10)}
.verdict.edge .l{color:rgba(var(--lean-tx),1)} .verdict.edge .vt{color:var(--ink)}

/* hitter row: percentile column + name cell */
td.pct{width:150px;white-space:nowrap}
table.lu td.nm,table.lu th.nm,table.lu td.pos{text-align:left}
/* Qualified so it outranks `table.lu td`'s mono face (0,1,2): hitter names are
   sans, the numbers around them stay mono. A bare `td.nm` never won this. */
table.lu td.nm{font:400 14px/1.4 var(--sans);max-width:170px;
  overflow:hidden;text-overflow:ellipsis}
td.nm .b{font:400 12px/1 var(--mono);color:var(--muted);margin-left:4px}
td.nm .adv{color:rgba(var(--warm-tx),1);font-size:12.5px;margin-left:2px}

/* Model machinery — always shown (analyst is the default and only view). The
   per-element display types match how the old Analyst toggle revealed them. */
.mach{display:block}
span.mach{display:inline}
td.mach,th.mach{display:table-cell}
tr.mach{display:table-row}
/* Keep the three starter stat cells in one horizontal strip. This specific
   flex rule must follow .mach so the generic display:block hook cannot stack
   xwOBA-against, K-BB%, and xERA vertically. */
.spstats.mach{display:flex}

/* ============================================================
   Phone layout (single breakpoint, last in the cascade)
   ------------------------------------------------------------
   One block, deliberately placed after every base rule: media queries add no
   specificity, so a mobile override declared earlier than its desktop
   counterpart silently loses (.sl and td.pct did exactly that). Keep new
   phone rules here.
   ============================================================ */
@media (max-width:540px){
  /* Tighter chrome: the card gutters, not the content, give up the width. */
  body{padding:14px 10px 44px}
  .game-summary{min-height:82px;padding:10px 10px}
  /* Centre column holds the first-pitch time on one line. At the Comfortable
     scale "7:05 PM ET" measures 84px in JetBrains Mono 14px (headless
     Chromium), so the old 82px wrapped it onto two rows on every unstarted
     card. 92px clears it with slack for a longer zone label; at 320px that
     still leaves 88px a side, and the widest thing a side carries is a 30px
     logo plus a three-letter club. */
  .teams{grid-template-columns:minmax(0,1fr) 92px minmax(0,1fr);
    gap:7px;padding-right:17px}
  .summary-team{gap:2px 6px}
  .summary-team.away{grid-template-columns:36px minmax(0,1fr);
    grid-template-areas:"logo club" "logo meta"}
  .summary-team.home{grid-template-columns:minmax(0,1fr) 36px;
    grid-template-areas:"club logo" "meta logo"}
  .summary-team.away .team-meta{justify-self:start}
  .summary-team.home .team-meta{justify-self:end;text-align:right}
  .summary-team .clogo,.summary-logo-fallback{width:36px;height:36px}
  .summary-logo-fallback{font-size:14px}
  .summary-club{font-size:14px}
  .teams .team-meta{font-size:12px}
  .teams .score{font-size:24px}
  .summary-center{gap:3px}
  .summary-time{font-size:14px;white-space:nowrap}
  .summary-market{font-size:12px}
  .summary-chevron{right:6px}
  .detail-context{padding:7px 12px}
  .read{padding:10px 12px}
  .side{padding:10px 12px 12px}
  .mcell{padding:6px 7px}
  .mcell .l{font-size:12px;letter-spacing:.04em}
  .mcell .v{font-size:14px}
  .mcell .v .mv{display:block;font-size:12px}
  .verdict{padding:7px 12px}
  .lg-notes{grid-template-columns:1fr}
  .lg-notes .wide{grid-column:auto}
  .spstats{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .stat{min-width:88px}
  /* Lineup fits inside the card instead of scrolling: trim the gutters, shrink
     the percentile bar, and let the name -- the one elastic column -- wrap
     instead of truncating. Every column the desktop card shows is kept. */
  table.lu th,table.lu td{padding:4px 3px}
  /* `table.lu td` is (0,1,2); a bare `td.nm` is (0,1,1) and loses the
     white-space battle no matter where it sits. Match the qualifier. */
  table.lu th{white-space:normal}
  table.lu td.nm{max-width:none;white-space:normal}
  td.ord,table.lu th:first-child{width:18px;padding-left:0}
  td.pct{width:auto}
  td.pos{font-size:12px}
  .sl{width:46px}
  .pn{margin-left:5px;font-size:12.5px}
}

/* ============================================================
   Touch targets (pointer axis, not width)
   ------------------------------------------------------------
   Measured in headless Chromium at 390px: the theme button rendered 28px
   tall and every lineup disclosure 32px, against a 44px floor (iOS HIG) /
   48dp (Material). The collapsed game row already clears it at 82px.
   Gated on `pointer:coarse` rather than a width breakpoint because it is
   the input device, not the viewport, that sets the floor -- a touch laptop
   at 1200px needs it and a 400px desktop window does not. Padding does the
   growing, so the type and the layout are untouched.
   ============================================================ */
@media (pointer:coarse){
  .theme{min-height:44px;padding:7px 14px}
  details.lineup summary{padding-top:14px;padding-bottom:12px}
}
"""

CSS_GRADES = r"""
.gradestrip{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:0 2px 16px;
  padding:10px 14px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
  font:600 14px/1.4 var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}
.gradestrip .lab{font:600 12px/1 var(--sans);text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.gradestrip .muted{color:var(--muted);font-weight:500}
.gradestrip a{color:rgba(var(--lean-tx),1);text-decoration:none;margin-left:auto;font:600 14px/1 var(--sans);white-space:nowrap}
.gradestrip a:hover{text-decoration:underline}

.backlink{font:600 14px/1 var(--sans);margin:0 2px 12px}
.backlink a{color:rgba(var(--lean-tx),1);text-decoration:none}
.backlink a:hover{text-decoration:underline}

/* ---------- grades page: masthead ---------- */
.gr-head{margin:0 2px 13px}
.gr-h1{font:800 19px/1.15 var(--sans);letter-spacing:-.01em;margin:0 0 5px}
.gr-lead{font:500 14px/1.55 var(--sans);color:var(--muted);max-width:80ch;margin:0}
.gr-lead b{color:var(--ink);font-weight:700}
/* `9:34 PM · 2026-07-28` is one token. Left to wrap it breaks at the
   date's own hyphen, so a phone renders `2026-` over `07-28`. */
.gr-lead .stamp{white-space:nowrap}

/* ---------- summary: one stat strip, same idiom as a card's SP stat row ---------- */
.gr-summary{display:flex;flex-wrap:wrap;margin:0 0 9px;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.gr-stat{flex:1 1 132px;min-width:0;padding:9px 14px 8px;border-right:1px solid var(--line-2)}
.gr-stat:last-child{border-right:0}
.gr-stat .l{font:600 12px/1.4 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.gr-stat .v{font:600 19px/1.2 var(--mono);font-variant-numeric:tabular-nums;
  color:var(--chip-fg);margin-top:3px}
.gr-stat .s{font:400 12.5px/1.35 var(--sans);font-variant-numeric:tabular-nums;
  color:var(--muted);margin-top:2px}
.gr-stat .v.cool{color:rgba(var(--cool-tx),1)}
.gr-stat .v.warm{color:rgba(var(--warm-tx),1)}
.gr-note{font:500 13px/1.55 var(--sans);color:var(--faint);margin:0 2px 14px}
.gr-note b{color:var(--muted);font-weight:700}

/* ---------- ledger table ---------- */
/* overflow:clip, not hidden -- `hidden` makes this a scroll container and the
   sticky column head then sticks to a box with no scrollable overflow, i.e.
   never. `clip` still rounds off the table corners. */
.gr-tablewrap{background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r);box-shadow:var(--shadow);overflow:clip}
/* Mono is for figures that line up in a column -- the lean deltas, the finals,
   the W/L badges -- and nothing else. The prose around them (the methodology
   note, the pitcher matchups, the day headers, the stat captions) is sans:
   at the Comfortable scale the note alone ran to a full screen of monospace
   on a phone, which reads as a code block rather than a sentence. */
table.gr{border-collapse:collapse;width:100%;table-layout:auto;
  font:500 14px/1.35 var(--mono);font-variant-numeric:tabular-nums}
table.gr th{font:600 12px/1 var(--sans);text-transform:uppercase;letter-spacing:.12em;color:var(--faint);
  text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;
  position:sticky;top:0;background:var(--surface);z-index:2}
table.gr td{padding:9px 12px;border-bottom:1px solid var(--line-2);white-space:nowrap;vertical-align:middle}
table.gr tbody tr:last-child td{border-bottom:none}
table.gr tbody tr.gr-row:hover td{background:var(--surface-2)}
table.gr tr.void td{opacity:.45}
table.gr .sp{display:block;margin-top:2px;font:400 12.5px/1.3 var(--sans);color:var(--faint)}
.gr-game{font:650 14px/1.3 var(--sans)}
.gr-game .at{color:var(--faint);font-weight:500}

/* Date group header -- the 300+ rows read as one undifferentiated scroll
   without it. Sticks under the column head (28px = its padded line box). */
tr.gr-day th{position:sticky;top:28px;z-index:1;background:var(--surface-2);
  padding:6px 12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line-2);
  font:700 12.5px/1.4 var(--sans);font-variant-numeric:tabular-nums;
  letter-spacing:.02em;text-transform:none;color:var(--muted)}
tr.gr-day .d{color:var(--ink)}
tr.gr-day .n{color:var(--faint);font-weight:500}
tr.gr-day .rec{float:right;color:var(--muted)}

/* ---------- badges ---------- */
.wlt{display:inline-block;min-width:17px;text-align:center;font:700 12.5px/1 var(--mono);
  padding:3px 6px;border-radius:var(--r-s)}
.wlt.W{color:rgba(var(--cool-tx),1);background:rgba(var(--cool),.12);border:1px solid rgba(var(--cool),.3)}
.wlt.L{color:rgba(var(--warm-tx),1);background:rgba(var(--warm),.12);border:1px solid rgba(var(--warm),.3)}
.wlt.T{color:var(--muted);background:var(--surface-2);border:1px solid var(--line)}
.wlt.none{color:var(--faint);background:transparent;border:1px dashed var(--line)}
.st{font:600 12px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.st.void{color:rgba(var(--warm-tx),.9)}

/* ---------- phone: each row becomes a labelled block ----------
   The table used to carry min-width:760px, so every phone view of the ledger
   was a horizontal scroll. Cells relabel themselves from data-l instead. */
@media (max-width:720px){
  table.gr thead{position:absolute;width:1px;height:1px;margin:-1px;overflow:hidden;
    clip:rect(0 0 0 0);white-space:nowrap}
  table.gr,table.gr tbody,table.gr tr,table.gr td,table.gr th{display:block}
  tr.gr-day th{position:static;top:auto}
  table.gr tr.gr-row{display:grid;grid-template-columns:minmax(0,1fr) auto;
    column-gap:10px;row-gap:4px;padding:11px 13px;border-bottom:1px solid var(--line-2)}
  table.gr tbody tr.gr-row:last-child{border-bottom:0}
  table.gr tr.gr-row td{border:0;padding:0;white-space:normal}
  table.gr td.c-game{grid-column:1;grid-row:1;min-width:0}
  table.gr td.c-res{grid-column:2;grid-row:1;justify-self:end;align-self:start}
  /* Remaining cells stack full-width under the game, each relabelled from
     data-l. Fixed label column so the values line up down the block. These
     need the table.gr prefix to outrank the display:block reset above. */
  table.gr td.c-lean,table.gr td.c-ml,table.gr td.c-final{
    grid-column:1/-1;display:flex;align-items:baseline;gap:9px;color:var(--muted)}
  table.gr td.c-lean::before,table.gr td.c-ml::before,table.gr td.c-final::before{
    content:attr(data-l);flex:0 0 46px;font:600 12px/1.7 var(--sans);
    letter-spacing:.1em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
}
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


def score_refresh_js():
    """Browser-side score refresh without rerunning the model build."""
    return rf"""
(function(){{
  var url='https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={SLATE_DATE}&hydrate=linescore';
  var headers={{}};
  document.querySelectorAll('.teams[data-game-pk]').forEach(function(el){{
    headers[String(el.getAttribute('data-game-pk'))]=el;
  }});
  function setMeta(el,score,state){{
    if(!el) return;
    if(state==='Live'||state==='Final'){{
      el.hidden=false;
      el.textContent=(score===null||score===undefined)?'—':String(score);
      el.className='team-meta score '+(el.classList.contains('home')?'home':'away');
      el.title=state==='Live'?'live score':'final score';
    }}else{{
      var streak=el.getAttribute('data-streak')||'';
      el.textContent=streak;
      el.className='team-meta streak '+(el.classList.contains('home')?'home':'away');
      el.title='current win/loss streak';
      el.hidden=!streak;
    }}
  }}
  function baseOutText(bases,outs){{
    var named=['first','second','third'].filter(function(n,i){{return bases[i];}});
    var runners=named.length===0?'bases empty':
      named.length===3?'bases loaded':
      named.length===1?'runner on '+named[0]:
      'runners on '+named.join(' and ');
    return outs===null||outs===undefined?runners:String(outs)+' out, '+runners;
  }}
  function setBaseOut(el,linescore,live){{
    if(!el) return;
    el.hidden=!live;
    if(!live) return;
    var offense=linescore.offense||{{}};
    var bases=[!!offense.first,!!offense.second,!!offense.third];
    var outs=(linescore.outs===null||linescore.outs===undefined)?null:Number(linescore.outs);
    ['b1','b2','b3'].forEach(function(cls,i){{
      var base=el.querySelector('.diamond .'+cls);
      if(base) base.className=cls+(bases[i]?' on':'');
    }});
    el.querySelectorAll('.outs i').forEach(function(dot,i){{
      dot.className=(outs!==null&&i<outs)?'on':'';
    }});
    var sr=el.querySelector('.bo-sr');
    if(sr) sr.textContent=baseOutText(bases,outs);
  }}
  function update(game){{
    var header=headers[String(game.gamePk)];
    if(!header) return;
    var status=game.status||{{}};
    var state=status.abstractGameState||'Preview';
    var teams=game.teams||{{}};
    setBaseOut(header.querySelector('.bo'),game.linescore||{{}},state==='Live');
    setMeta(header.querySelector('.team-meta.away'),(teams.away||{{}}).score,state);
    setMeta(header.querySelector('.team-meta.home'),(teams.home||{{}}).score,state);
    var inProgress=state==='Live'||state==='Final';
    var summaryTime=header.querySelector('.summary-time');
    var summaryMarket=header.querySelector('.summary-market');
    if(summaryTime) summaryTime.hidden=inProgress;
    if(summaryMarket) summaryMarket.hidden=inProgress;
    var badge=header.querySelector('.game-state');
    if(!badge) return;
    if(inProgress){{
      badge.hidden=false;
      if(state==='Live'){{
        var linescore=game.linescore||{{}};
        var ordinal=linescore.currentInningOrdinal||'';
        if(!ordinal&&linescore.currentInning!==null&&linescore.currentInning!==undefined){{
          var n=Number(linescore.currentInning);
          var mod100=n%100;
          var suffix=(mod100>10&&mod100<14)?'th':({{1:'st',2:'nd',3:'rd'}}[n%10]||'th');
          ordinal=String(n)+suffix;
        }}
        var half=String(linescore.inningHalf||'').toLowerCase();
        var marker=half==='top'?'▲':(half==='bottom'?'▼':'');
        if(!marker&&linescore.isTopInning===true) marker='▲';
        if(!marker&&linescore.isTopInning===false) marker='▼';
        badge.textContent='LIVE'+(ordinal?' · '+marker+ordinal:'');
      }}else{{
        badge.textContent='FINAL';
      }}
      badge.className='game-state '+state.toLowerCase();
      badge.title=status.detailedState||'';
    }}else{{
      badge.hidden=true;
      badge.textContent='';
      badge.className='game-state';
      badge.title='';
    }}
  }}
  function refresh(){{
    if(document.hidden) return;
    fetch(url,{{cache:'no-store'}})
      .then(function(r){{if(!r.ok) throw new Error(String(r.status));return r.json();}})
      .then(function(data){{
        (data.dates||[]).forEach(function(day){{
          (day.games||[]).forEach(update);
        }});
      }})
      .catch(function(){{}});
  }}
  refresh();
  window.setInterval(refresh,60000);
}})();
"""


def html_document(body, built_txt, title=None, extra_js=None):
    title = title or f"MLB matchup leans - {SLATE_DATE}"
    extra_script = f"<script>{extra_js}</script>" if extra_js else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        "<meta name='robots' content='noindex'>"
        f"<style>{font_face_css()}{CSS}{CSS_GRADES}</style></head><body>"
        f"<div class='mx-wrap'>"
        "<div class='topbar'><div class='brand'>MLB matchup leans</div>"
        "<div style='display:flex;gap:8px'>"
        "<button id='themeBtn' class='theme' type='button'>Theme: auto</button></div></div>"
        f"{body}</div>"
        f"<script>{THEME_JS}</script>"
        f"{extra_script}"
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
    """Graded rows whose tags share the current prediction methodology.

    Kept for the weight-fit / audit regime that must not mix prediction-math
    changes. The user-facing combined record uses _display_grades instead."""
    return led[(led["status"] == "graded") & (led["model_tag"].isin(RECORD_TAGS))]


def _display_grades(led):
    """All graded rows, every model version joined. The versions are one
    incremental lineage, so the displayed record combines them into a single
    record per market (audit slicing still lives in _record_grades)."""
    return led[led["status"] == "graded"]


# Minimum graded games in a (lean side × agree/disagree) bucket before its
# record is trusted enough to headline the verdict; thinner buckets keep prose.
VERDICT_CONTEXT_MIN = 10


def market_context_records():
    """(lean_side, relation) -> 'W-L[-T]' for graded xwOBA full-game leans.

    lean_side is 'home'/'away' (the side the model leaned); relation is
    'agree'/'disagree' vs the market favorite, read from the devigged closing
    home probability (`close_p_home` >= .5 -> home favored). Powers the
    Model-vs-market verdict's context record. Display-only and best-effort: an
    absent ledger/market column returns {}, and a bucket thinner than
    VERDICT_CONTEXT_MIN decisions is omitted so the verdict keeps its prose."""
    led = load_ledger_df()
    if led is None:
        return {}
    g = _display_grades(led)
    if g.empty or "close_p_home" not in g.columns:
        return {}
    ph = pd.to_numeric(g["close_p_home"], errors="coerce")
    lean, grade = g["xw_lean"], g["xw_full"]
    graded = ph.notna() & grade.isin(["W", "L", "T"])
    out = {}
    for side in ("home", "away"):
        side_is_lean = lean.eq(g["home"] if side == "home" else g["away"])
        mkt_is_lean = (ph >= 0.5) if side == "home" else (ph < 0.5)
        for rel in ("agree", "disagree"):
            rel_ok = mkt_is_lean if rel == "agree" else ~mkt_is_lean
            sub = grade[graded & side_is_lean & rel_ok]
            w, l, t = int((sub == "W").sum()), int((sub == "L").sum()), int((sub == "T").sum())
            if w + l + t >= VERDICT_CONTEXT_MIN:
                out[(side, rel)] = f"{w}-{l}" + (f"-{t}" if t else "")
    return out


# --- lean strength label (ranks a lean's size vs the ledger) ----------------
# Point-1 of the casual redesign: turn the raw |Δxw| (".024", ".123") into a
# word by ranking it against the historical spread of lean magnitudes. Ranks
# only against SCALE_TAGS rows (same |xw_net| units as the current model) --
# never against the full graded pool, which mixes the pre/post-v5 scales.
# Display-only -- the lean math, MODEL_TAG, and ledger are untouched.
#
# Last-resort cutoffs, used only when neither the scale-compatible ledger rows
# nor tonight's own slate reach LEAN_STRENGTH_MIN -- in practice only on the
# first build of a brand-new scale family with a tiny slate.
# Provenance: p33/p80 of |xw_net| over the 24 xw+plat_consol_v8/v9 ledger rows
# as of 2026-07-28 (0.0151 / 0.0325), rounded. The previous literal here was
# the *pre-v5 unshrunk* p33/p80 (0.021, 0.060), left in place across the v5,
# v7 and v8 shrinkage changes that moved the distribution underneath it; with
# v9's observed maximum |xw_net| of 0.0462 it made "strong" unreachable for
# every game in the family. Invalidated by any MODEL_TAG bump that changes the
# delta scale -- i.e. whenever _SCALE_FAMILIES gains a new entry. Recompute
# from the new family's rows, or delete this and let the dynamic scale run.
LEAN_STRENGTH_FALLBACK = (0.015, 0.032)   # slight < ~p33 <= clear < ~p80 <= strong
LEAN_STRENGTH_MIN = 30


def lean_strength_scale(slate_deltas=None):
    """Sorted |xw_net| on the current model's delta scale (SCALE_TAGS), for
    ranking a lean's size. Never pools across a scale change (v5's shrinkage
    halved the units) and never falls back to the full ledger.

    Every SCALE_TAGS row counts regardless of grade status: |xw_net| is a
    pregame quantity, so a scale of lean *magnitudes* needs no outcome. The old
    `status == "graded"` filter discarded pending rows and left each new scale
    family ranking against the frozen fallback for days after a MODEL_TAG bump,
    which is exactly how the pre-v5 cutoffs survived three scale changes.
    `slate_deltas` (tonight's own |Δxw|, same units by construction) tops up the
    pool so a fresh family is ranked on its own units from its first build.

    Returns None (-> LEAN_STRENGTH_FALLBACK) only when the combined pool is
    still below LEAN_STRENGTH_MIN."""
    parts = []
    led = load_ledger_df()
    if led is not None and "xw_net" in led.columns:
        scaled = (led[led["model_tag"].isin(SCALE_TAGS)]
                  if "model_tag" in led.columns else led)
        parts.append(pd.to_numeric(scaled["xw_net"], errors="coerce").abs().dropna().to_numpy())
    if slate_deltas is not None:
        parts.append(pd.to_numeric(pd.Series(list(slate_deltas)), errors="coerce")
                     .abs().dropna().to_numpy())
    if not parts:
        return None
    arr = np.sort(np.concatenate(parts))
    return arr if arr.size >= LEAN_STRENGTH_MIN else None


def _slate_deltas(games):
    """Tonight's |Δxw| per game, computed exactly as cmb_card does. An exact
    zero is an abstention (v7 rule), not a lean of size zero, so it is excluded
    from the magnitude scale just as it is excluded from the cards."""
    out = []
    for g in games or ():
        if g.get("unavailable"):
            continue
        a, h = g.get("away") or {}, g.get("home") or {}
        ae, he = a.get("xw_edge"), h.get("xw_edge")
        if ae is None or he is None:
            continue
        net = ae - he
        if net:
            out.append(abs(net))
    return out


def lean_strength(delta, scale):
    """(label, pctile|None) for a lean magnitude |Δxw|. Ranks against `scale`
    (sorted |xw_net| history); fixed p33/p80 cutoffs when the ledger is thin."""
    if delta is None or pd.isna(delta):
        return None, None
    delta = abs(float(delta))
    if scale is not None and getattr(scale, "size", 0) >= LEAN_STRENGTH_MIN:
        pct = 100.0 * float(np.searchsorted(scale, delta, side="right")) / scale.size
        c1, c2 = float(np.quantile(scale, 1 / 3)), float(np.quantile(scale, 0.80))
    else:
        pct, (c1, c2) = None, LEAN_STRENGTH_FALLBACK
    lab = "slight" if delta < c1 else ("clear" if delta < c2 else "strong")
    return lab, pct


def records_strip_html():
    led = load_ledger_df()
    if led is None:
        return ""
    g = _display_grades(led)
    if g.empty:
        inner = "<span class='muted'>no graded games yet</span>"
    else:
        bits = [f"xwOBA full {_rec_txt(g['xw_full'])}"]
        # xwOBA vs-market (z / flat ROI) — mirrors the grades-page core card;
        # market columns are absent until the first market run.
        if "close_p_home" in g.columns and g["close_p_home"].notna().any():
            try:
                from market_backfill import vs_market_summary
                m = vs_market_summary(g, verbose=False).get("xwOBA")
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
    txt = _esc(lean) + (
        f" <span class='muted'>Δ{delta3(d)}</span>" if pd.notna(d) else ""
    )
    return f"<span class='muted'>{txt}</span>" if muted else txt


def _grades_row(r, show_ml=False):
    status = str(r["status"])
    fa, fh = pd.to_numeric(r["full_away"], errors="coerce"), pd.to_numeric(r["full_home"], errors="coerce")
    if status != "graded":
        final = f"<span class='st {status}'>{_esc(status)}</span>"
    else:
        final = f"{int(fa)}–{int(fh)}" if pd.notna(fa) and pd.notna(fh) else "<span class='muted'>—</span>"
    # Muted marker when the accepted snapshot locked with a fully projected
    # lineup on either side (lineup_status_* audit columns; NaN on legacy rows).
    proj_mark = ""
    if any(str(r.get(c)) == "projected"
           for c in ("lineup_status_away", "lineup_status_home")):
        proj_mark = (" <span class='muted' title='locked with a projected "
                     "lineup'>°</span>")
    game = (f"<span class='gr-game'>{_esc(r['away'])} <span class='at'>@</span> "
            f"{_esc(r['home'])}</span>{proj_mark}"
            f"<span class='sp'>{_esc(r.get('away_sp') or '—')} v "
            f"{_esc(r.get('home_sp') or '—')}</span>")
    # (class, phone label, html). data-l only surfaces in the stacked phone
    # layout, so it stays short; the column heads carry the full names. The
    # slate date is carried by the group header row, not repeated per row.
    cells = [("c-game", "Game", game),
             ("c-lean", "Lean", _lean_cell(r["xw_lean"], r["xw_delta"]))]
    if show_ml:
        cells.append(("c-ml", "ML", _lean_ml_cell(r, "xw_lean")))
    cells += [("c-final", "Final", final),
              ("c-res", "Result", _wlt_badge(r["xw_full"]))]
    cls = "gr-row void" if status == "void" else "gr-row"
    tds = "".join(f"<td class='{c}' data-l='{lab}'>{v}</td>" for c, lab, v in cells)
    return f"<tr class='{cls}'>{tds}</tr>"


def _grades_day_header(date, day, ncols):
    """Group separator for one slate date: weekday, game count, day record.

    313 rows in one flat tbody read as an undifferentiated scroll; the ledger
    is naturally keyed by slate date, so the date leads each group instead of
    repeating in every row."""
    try:
        weekday = pd.to_datetime(date).strftime("%a")
    except Exception:  # noqa: BLE001
        weekday = ""
    n = len(day)
    lab = f"<span class='d'>{_esc(date)}</span>"
    if weekday:
        lab += f" <span class='n'>{weekday}</span>"
    lab += f" <span class='n'>· {n} game{'' if n == 1 else 's'}</span>"
    # Day record only once something on that date is graded; an all-pending
    # date would otherwise show a meaningless 0-0.
    decided = day["xw_full"].isin(["W", "L", "T"]).sum() if "xw_full" in day else 0
    if decided:
        lab += f"<span class='rec'>{_rec_parts(day['xw_full'])[0]}</span>"
    return (f"<tr class='gr-day'><th colspan='{ncols}' scope='colgroup'>"
            f"{lab}</th></tr>")


def _lock_provenance(led):
    """(verified, unverified) rows for the pregame-lock claim.

    The page asserted every row locked before first pitch. `lock_status` is
    null on every row that predates the instrumentation, and those rows carry
    no `snapshot_utc`/`scheduled_start_utc` either, so nothing in the ledger
    substantiates the claim for them. Report the split instead of asserting
    the whole."""
    n = len(led)
    if "lock_status" not in led.columns:
        return 0, n
    verified = int(led["lock_status"].astype("string")
                   .str.startswith("pregame", na=False).sum())
    return verified, n - verified


def render_grades_html(built_txt):
    back = "<div class='backlink'><a href='index.html'>← today's leans</a></div>"
    led = load_ledger_df()
    if led is None:
        body = back + ("<div class='legend'><div class='lg-title'>Grading ledger · "
                       "no graded data yet — the ledger appears after the first CI run."
                       "</div></div>")
        return html_document(body, built_txt, title="MLB lean grades")

    n_pend = int((led["status"] == "pending").sum())
    n_void = int((led["status"] == "void").sum())
    head = ("<div class='gr-head'><h1 class='gr-h1'>Grading ledger</h1>"
            "<div class='gr-lead'>Every lean this model has published, graded "
            f"against the final score. Built <span class='stamp'>{built_txt}"
            "</span>.</div></div>")

    g = _display_grades(led)
    stats, notes = [], []

    def stat(lab, val, sub=None, tone=""):
        s = f"<div class='s'>{sub}</div>" if sub else ""
        stats.append(f"<div class='gr-stat'><div class='l'>{lab}</div>"
                     f"<div class='v{(' ' + tone) if tone else ''}'>{val}</div>{s}</div>")

    if g.empty:
        summary = "<div class='gr-note'>No graded games yet.</div>"
    else:
        pend_sub = f"{n_pend} pending" + (f" · {n_void} void" if n_void else "")
        stat("Graded", str(len(g)), pend_sub)
        b, p = _rec_parts(g["xw_full"]); stat("xwOBA · full", b, p)
        # vs-market scoreboard (closing DK MLs attached by grade_leans.py via
        # market_backfill; columns absent until the first market run). z leads
        # the cell -- it is the primary metric, per the note below.
        if "close_p_home" in g.columns and g["close_p_home"].notna().any():
            try:
                from market_backfill import vs_market_summary
                mkt = vs_market_summary(g, verbose=False)
            except Exception as e:  # noqa: BLE001
                log(f"vs-market summary degraded: {e!r}")
                mkt = {}
            m = mkt.get("xwOBA")
            if m:
                stat("xwOBA · vs mkt", f"z {m['z']:+.2f}",
                     f"{m['w']}-{m['n'] - m['w']} · {m['roi_units']:+.2f}u flat",
                     tone="cool" if m["z"] > 0 else "warm")
        notes.append("xwOBA graded full-game vs devigged DK closing ML (ESPN capture); "
                     "z and flat ROI are the primary metrics")
        # Provenance, not a blanket claim: state how many rows the ledger can
        # actually show locked pregame.
        verified, unverified = _lock_provenance(led)
        lock = f"<b>{verified}</b> of <b>{len(led)}</b> rows carry a pregame lock timestamp"
        notes.append(lock + (f"; {unverified} legacy rows predate that "
                             "instrumentation and are unverified"
                             if unverified else ""))
        summary = ("<div class='gr-summary'>" + "".join(stats) + "</div>"
                   + (f"<div class='gr-note'>{'. '.join(notes)}.</div>" if notes else ""))

    show_ml = "close_home_ml" in led.columns and led["close_home_ml"].notna().any()
    heads = (["Game", "xwOBA lean"]
             + (["xw ML"] if show_ml else [])
             + ["Final", "xw F"])
    led = led.sort_values(["game_date", "game_pk"], ascending=[False, True])
    body = []
    for date, day in led.groupby("game_date", sort=False):
        body.append(_grades_day_header(date, day, len(heads)))
        body += [_grades_row(r, show_ml) for _, r in day.iterrows()]
    table = ("<div class='gr-tablewrap'><table class='gr'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in heads)
             + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    return html_document(back + head + summary + table, built_txt, title="MLB lean grades")


def render_combined_html(xw_df, pl_df, pitcher_rows_df, built_txt,
                         opp_hitters_df=None, detail_df=None, lg_ops=None,
                         slate_df=None, lineup_df=None,
                         league_baseline=None, odds=None, streaks=None):
    games = _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                                  opp_hitters_df=opp_hitters_df, detail_df=detail_df,
                                  lg_ops=lg_ops, slate_df=slate_df, lineup_df=lineup_df,
                                  league_baseline=league_baseline, odds=odds, streaks=streaks)
    head = _legend_head("MLB matchup leans — Statcast xwOBA", built_txt)
    # Cards lead; the how-to-read guide and the record strip sit below them.
    footer = _legend_guide() + records_strip_html()
    if not games:
        inner = head + "<div class='legend'><div class='lg-title'>No paired probables yet — " \
                       "probables/lineups not posted. Check back closer to first pitch.</div></div>" + footer
        return html_document(inner, built_txt, extra_js=score_refresh_js())
    strength_scale = lean_strength_scale(_slate_deltas(games))
    logo_assets, logo_css = _logo_assets(games)
    ctx = {**(market_context_records() or {}), "logo_ids": set(logo_assets)}
    body = logo_css + head + build_combined(games, strength_scale, ctx) + footer
    return html_document(body, built_txt, extra_js=score_refresh_js())


def empty_slate_html(built_txt):
    body = (
        "<div class='legend'>"
        f"<div class='lg-title'>MLB matchup leans · {SLATE_DATE} · built {built_txt}</div>"
        "<div class='lg-keys'><span class='k'>No MLB games scheduled for this date.</span></div>"
        "</div>") + records_strip_html()
    return html_document(body, built_txt)


def _lineup_status_columns(lineup_df):
    """Per-game lineup resolution (posted / partial_filled / projected +
    posted counts) keyed by game_pk, for stamping onto the lean dumps so the
    ledger can audit lineup freshness at lock. lineup_resolution_audit.csv is
    overwritten each build; the dump is what persists. Instrumentation only —
    no effect on leans or grading."""
    if lineup_df is None or lineup_df.empty:
        return {}
    idx = lineup_df.set_index("game_pk")
    missing = pd.Series(np.nan, index=idx.index)
    return {
        "lineup_status_away": idx["away_lineup_status"],
        "lineup_status_home": idx["home_lineup_status"],
        "lineup_posted_away": idx["away_posted_count"],
        "lineup_posted_home": idx["home_posted_count"],
        "lineup_savant_backfill_away":
            idx.get("away_savant_backfill_count", missing),
        "lineup_savant_backfill_home":
            idx.get("home_savant_backfill_count", missing),
    }


# ============================================================
# MAIN
# ============================================================
def _built_text_now():
    now_pt = datetime.now(PT)
    return (now_pt.strftime("%I:%M %p").lstrip("0")
            + now_pt.strftime(" · %Y-%m-%d"))


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
    matchup_df = apply_pitching_plans(
        matchup_df, data.get("pitching_plans"), data["league_baseline"]
    )

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

    if USE_PITCH_MIX_SHADOW:
        # Shadow only: extra columns for forward testing, no effect on any lean.
        # Failure-isolated like the odds fetch -- an arsenal outage must never
        # take down a build that has already fetched everything the lean needs.
        log("Attaching pitch-mix shadow columns (inert; forward test only) ...")
        try:
            batter_arsenal, pitcher_arsenal = pitch_arsenal.fetch_arsenals(
                cached_csv, SEASON)
            matchup_df = attach_pitch_mix_shadow(
                matchup_df, pitcher_rows_df, opp_hitters_df,
                data["league_baseline"], batter_arsenal, pitcher_arsenal)
            cov = pd.to_numeric(matchup_df.get("mix_coverage"), errors="coerce")
            log(f"  arsenal cells: {len(batter_arsenal)} batter rows, "
                f"{len(pitcher_arsenal)} pitcher rows; median side coverage "
                f"{cov.median():.2f}" if cov is not None and len(cov)
                else "  arsenal coverage unavailable")
        except Exception as e:  # noqa: BLE001
            log(f"pitch-mix shadow skipped: {e!r}")

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
    lu_cols = _lineup_status_columns(data["lineup_projection_df"])
    for frame in (matchup_df, matchup_platoon_df):
        if frame is not None and not frame.empty:
            frame["model_tag"] = MODEL_TAG
            frame["snapshot_utc"] = snapshot_utc
            if "game_datetime_utc" in frame.columns:
                frame["scheduled_start_utc"] = frame["game_datetime_utc"]
            for col, series in lu_cols.items():
                frame[col] = frame["game_pk"].map(series)
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

    log("Fetching current streaks (best-effort, display-only) ...")
    try:
        streaks = fetch_current_streaks(data["slate_df"])
    except Exception as e:  # noqa: BLE001
        log(f"current streaks skipped: {e!r}")
        streaks = {}

    log("Rendering index.html ...")
    html = render_combined_html(
        matchup_df, matchup_platoon_df, pitcher_rows_df, built_txt,
        opp_hitters_df=opp_hitters_df, detail_df=platoon_detail_df,
        lg_ops=league_ops_overall, slate_df=data["slate_df"],
        lineup_df=data["lineup_projection_df"],
        league_baseline=data["league_baseline"], odds=odds, streaks=streaks)
    with open(out_path, "w") as f:
        f.write(html)
    log(f"Wrote {out_path} ({len(html):,} bytes, {len(matchup_df)} matchup rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
