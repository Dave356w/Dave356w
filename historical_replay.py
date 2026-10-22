#!/usr/bin/env python3
"""Date-bounded Statcast replay of the shipped model. Never writes data/.

Historical starting orders are reconstructed from FINAL box scores, not saved
pregame announcements. Rates use strictly earlier dates; the model's own
functions calculate both sides. This is a retrospective backtest, not an
original-prediction recovery or an independent forward test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import build_site as bs
import grade_leans as gl
import starter_velocity as sv

SCHEMA = 1
BASIS = "statcast_prior_date_final_starting_lineup"
CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SAVANT_ROW_LIMIT = 25000
API = "https://statsapi.mlb.com/api/v1"
PITCH_COLS = ["game_date", "game_pk", "game_type", "at_bat_number", "pitch_number",
              "batter", "pitcher", "events", "bb_type", "inning", "inning_topbot",
              "stand", "p_throws", "pitch_type", "release_speed", "launch_speed",
              "launch_angle", "estimated_woba_using_speedangle", "woba_value",
              "woba_denom", "post_home_score", "post_away_score"]
PA_EVENTS = frozenset(("single", "double", "triple", "home_run", "field_out",
    "force_out", "grounded_into_double_play", "field_error", "fielders_choice",
    "fielders_choice_out", "double_play", "triple_play", "strikeout",
    "strikeout_double_play", "walk", "intent_walk", "hit_by_pitch", "sac_fly",
    "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf",
    "other_out"))
SOURCE_HASHES = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ("historical_replay.py", "build_site.py", "starter_velocity.py",
                              "grade_leans.py", "requirements.txt")}


class MissingHistory(RuntimeError):
    """Required historical input could not be verified."""


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


class CachedAPI:
    """Content-addressed read-only MLB transport, with fetch provenance."""
    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline
        self.failures = []

    def __call__(self, url, params=None, tries=3, headers=None):
        if not url.startswith(API + "/"):
            raise MissingHistory(f"Replay refused non-MLB or live lookup: {url}")
        key = json.dumps([url, params or {}], sort_keys=True, separators=(",", ":"))
        path = self.root / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("request") != key:
                raise MissingHistory("Historical API cache identity mismatch")
            return payload["response"]
        if self.offline:
            raise MissingHistory(f"Not cached: {url} {params or {}}")
        for attempt in range(tries):
            try:
                r = requests.get(url, params=params, headers=headers,
                                 timeout=(10, 45))
                r.raise_for_status()
                result = r.json()
                if "message" in result and "error" in str(result).lower():
                    raise MissingHistory(str(result)[:250])
                atomic_text(path, json.dumps({"request": key,
                    "fetched_utc": datetime.now(timezone.utc).isoformat(),
                    "response": result}))
                return result
            except (requests.RequestException, ValueError, MissingHistory) as exc:
                if attempt == tries - 1:
                    self.failures.append(f"{url}: {exc}")
                    raise MissingHistory(self.failures[-1]) from exc
                time.sleep(0.5 * 2 ** attempt)


def schedule(api, season, end):
    """All regular-season games, including dates before an evaluation window."""
    data = api(API + "/schedule", {"sportId": 1, "startDate": f"{season}-01-01",
        "endDate": str(end), "gameTypes": "R",
        "hydrate": "team,probablePitcher,linescore"})
    games = {}
    for day in data.get("dates", []):
        for game in day.get("games", []):
            if game.get("gameType") == "R":
                g = dict(game)
                g["_date"] = str(game.get("officialDate") or day["date"])[:10]
                games[int(g["gamePk"])] = g
    return games


def completed(g):
    return g.get("status", {}).get("detailedState") in gl._FINAL


def available_date(g):
    """Conservatively withhold a resumed game's entire line until completion.

    Statcast can date resumed innings to the original game date. Including
    them then would leak future pitches into every intervening slate.
    """
    return max(g["_date"], str(g.get("resumeDate") or g.get("resumedFrom") or "")[:10])


def suspension_blocks(games, day):
    # MLB byDateRange totals and player game logs can attribute a resumed
    # game's complete line to its original date. Until completion, the shared
    # role/profile parsers therefore cannot reproduce a clean as-of state.
    # Exclude the entire affected slate instead of silently using those totals.
    return any(g["_date"] < day <= available_date(g) for g in games.values())


def validate_pitches(raw, start, end):
    missing = sorted(set(PITCH_COLS) - set(raw.columns))
    if missing:
        raise MissingHistory(f"Statcast response missing columns: {missing}")
    d = raw[PITCH_COLS].copy()
    if d.empty:
        return d
    d["game_date"] = pd.to_datetime(d["game_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if not d["game_date"].between(str(start), str(end)).all():
        raise MissingHistory("Statcast response contains pitches outside requested dates")
    d = d[d["game_type"] == "R"].copy()
    ids = ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]
    for c in ids:
        n = pd.to_numeric(d[c], errors="coerce")
        if n.isna().any() or (n % 1 != 0).any():
            raise MissingHistory(f"Invalid Statcast identity column: {c}")
        d[c] = n.astype("int64")
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    if d.duplicated(keys).any():
        raise MissingHistory("Duplicate Statcast pitch identities")
    return d


def fetch_pitches(cache, season, end, offline=False, workers=3, chunk_days=7):
    """Cache reduced raw pitches; request bounds and schema are part of the key."""
    root = Path(cache) / "statcast" / f"v{SCHEMA}"
    root.mkdir(parents=True, exist_ok=True)
    first, last = date(season, 3, 1), date.fromisoformat(end)
    ranges = []
    cur = first
    while cur <= last:
        stop = min(cur + timedelta(days=chunk_days - 1), last)
        ranges.append((cur.isoformat(), stop.isoformat()))
        cur = stop + timedelta(days=1)

    def one(bounds):
        start, stop = bounds
        path = root / f"{start}_{stop}.csv.gz"

        def split():
            first, last = date.fromisoformat(start), date.fromisoformat(stop)
            if first == last:
                raise MissingHistory(f"Statcast {start} reached its row limit on a single day")
            middle = first + (last - first) // 2
            return pd.concat([one((start, middle.isoformat())),
                one(((middle + timedelta(days=1)).isoformat(), stop))], ignore_index=True)

        if path.exists():
            cached = validate_pitches(pd.read_csv(path, low_memory=False), start, stop)
            return split() if len(cached) >= SAVANT_ROW_LIMIT else cached
        if offline:
            raise MissingHistory(f"Statcast chunk not cached: {path.name}")
        params = {"all": "true", "type": "details", "game_date_gt": start,
                  "game_date_lt": stop, "hfGT": "R|"}
        for attempt in range(3):
            try:
                r = requests.get(CSV_URL, params=params, timeout=(15, 90))
                r.raise_for_status()
                if not r.text.strip():
                    raise MissingHistory("Statcast returned an empty body")
                raw = pd.read_csv(io.StringIO(r.text), usecols=lambda c: c in PITCH_COLS,
                                  low_memory=False)
                d = validate_pitches(raw, start, stop)
                # Savant can return HTTP 200 with exactly 25,000 rows and no
                # pagination warning. Split before trusting or caching it.
                if len(d) >= SAVANT_ROW_LIMIT:
                    return split()
                tmp = path.with_name(path.name + ".tmp")
                d.to_csv(tmp, index=False, compression="gzip")
                tmp.replace(path)
                atomic_text(path.with_suffix(".json"), json.dumps({
                    "schema": SCHEMA, "request": params, "rows": len(d),
                    "fetched_utc": datetime.now(timezone.utc).isoformat()}))
                print(f"Statcast {start}..{stop}: {len(d):,} pitches", flush=True)
                return d
            except (requests.RequestException, ValueError, MissingHistory) as exc:
                if attempt == 2:
                    raise MissingHistory(f"Statcast {start}..{stop}: {exc}") from exc
                time.sleep(1 + attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(one, ranges))
    d = pd.concat(parts, ignore_index=True)
    if d.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
        raise MissingHistory("Overlapping Statcast chunks contain duplicate pitches")
    return d


def prepare_history(raw, games):
    """Verify coverage and final scores before any earlier-date rate is trusted."""
    final = {pk: g for pk, g in games.items() if completed(g)}
    found = set(raw["game_pk"].astype(int))
    missing = sorted(set(final) - found)
    if missing:
        raise MissingHistory(f"Missing Statcast games in input history: {missing[:25]} "
                             f"({len(missing)} total); refusing partial-season inputs")
    for pk, rows in raw.groupby("game_pk"):
        if int(pk) not in final:
            continue
        g = final[int(pk)]
        for side in ("home", "away"):
            score = pd.to_numeric(rows[f"post_{side}_score"], errors="coerce").max()
            actual = g["teams"][side].get("score")
            if pd.isna(score) or actual is None or int(score) != int(actual):
                raise MissingHistory(f"Statcast final-score mismatch: {pk} {side}")
    d = raw[raw["game_pk"].isin(final)].copy()
    d["available_date"] = d["game_pk"].map({pk: available_date(g) for pk, g in final.items()})
    pa = d[d["events"].isin(PA_EVENTS)].copy()
    if pa.duplicated(["game_pk", "at_bat_number"]).any():
        raise MissingHistory("Multiple terminal events for a plate appearance")
    for c in ("woba_denom", "woba_value", "estimated_woba_using_speedangle"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")
    # Positive-denominator non-contact events (BB, HBP, K) use their actual
    # weights. Untracked contact uses the same explicit wOBA fallback as the
    # existing history collector; its count is exposed, never filled with zero.
    valid = pa["woba_denom"] > 0
    if pa.loc[valid, "woba_value"].isna().any():
        raise MissingHistory("Missing wOBA weight on a denominator-bearing PA")
    pa["untracked_bip"] = valid & pa["bb_type"].notna() & pa["estimated_woba_using_speedangle"].isna()
    pa["x_value"] = pa["estimated_woba_using_speedangle"].where(
        pa["estimated_woba_using_speedangle"].notna(), pa["woba_value"])
    pa["x_value"] = pa["x_value"].where(valid, 0.0)
    pa["w_value"] = pa["woba_value"].where(valid, 0.0)
    pa["denom"] = pa["woba_denom"].where(valid, 0.0)
    return d, pa


def rate_board(pa, player_type, before):
    """A Savant-shaped board, using only season PAs available before the date."""
    prior = pa[pa["available_date"] < before].copy()
    key = "batter" if player_type == "batter" else "pitcher"
    cols = ["player_id", "pa", "xwoba", "woba", "bbe", "k_percent", "bb_percent"]
    if prior.empty:
        return pd.DataFrame(columns=cols)
    prior["bbe"] = prior["bb_type"].notna().astype(int)
    prior["k"] = prior["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    prior["bb"] = prior["events"].isin(["walk", "intent_walk"]).astype(int)
    out = prior.groupby(key).agg(pa=(key, "size"), x=("x_value", "sum"),
        w=("w_value", "sum"), den=("denom", "sum"), bbe=("bbe", "sum"),
        k=("k", "sum"), bb=("bb", "sum")).reset_index().rename(columns={key: "player_id"})
    out["xwoba"] = out["x"] / out["den"].replace(0, np.nan)
    out["woba"] = out["w"] / out["den"].replace(0, np.nan)
    out["k_percent"], out["bb_percent"] = 100 * out["k"] / out["pa"], 100 * out["bb"] / out["pa"]
    return out[cols]


def starting_order(team):
    """Original nine batting-order slots, excluding later substitutions."""
    slots = {}
    for player in team.get("players", {}).values():
        order = str(player.get("battingOrder") or "")
        if order.isdigit() and int(order) % 100 == 0:
            slot = int(order) // 100
            if slot in slots or not 1 <= slot <= 9:
                raise MissingHistory("Invalid original batting order")
            slots[slot] = int(player["person"]["id"])
    if set(slots) != set(range(1, 10)) or len(set(slots.values())) != 9:
        raise MissingHistory("Incomplete original starting lineup")
    return [slots[i] for i in range(1, 10)]


def starting_pitcher(team):
    starters = [p for p in team.get("players", {}).values()
                if p.get("stats", {}).get("pitching", {}).get("gamesStarted") == 1]
    if len(starters) != 1:
        raise MissingHistory("Missing or ambiguous actual starting pitcher")
    return int(starters[0]["person"]["id"]), starters[0]["person"]["fullName"]


@contextmanager
def model_environment(api, season, slate_date):
    """Scope legacy loader transport/config; restore ALL mutable model caches.

    Dated loaders in build_site remain the shared parsers. No live transport,
    on-disk Savant cache, production ledger fit, or audit writer is permitted.
    """
    def forbidden(*args, **kwargs):
        raise MissingHistory("Replay attempted to read the live ledger or Savant leaderboard")

    attrs = {"_get_json": api, "SEASON": season, "SLATE_DATE": slate_date,
             "load_ledger_df": forbidden, "cached_csv": forbidden,
             "REQUEST_DELAY": 0, "USE_PLAYER_PRIORS": False}
    for key in ("_pitcher_roster_cache", "_team_pitcher_role_cache", "_roster_cache",
                "_gf_cache", "_sp_ip_cal_state", "_player_prior_state"):
        attrs[key] = {}
    old = {key: getattr(bs, key) for key in attrs}
    try:
        for key, value in attrs.items():
            setattr(bs, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(bs, key, value)


class HistoricalDataProvider(bs.LiveDataProvider):
    def __init__(self, before, season, raw, pa, games, api, velocity=None):
        super().__init__(before)
        self.season, self.raw, self.pa, self.games, self.api = season, raw, pa, games, api
        self.targets = [g for g in games.values() if g["_date"] == before]
        self.boxes, self.orders, self.exclusions, self.rosters = {}, {}, {}, {}
        self.boards = {}
        self._velocity = velocity

    def prefetch(self, workers=3):
        """Parallelize independent reads; prediction itself stays sequential."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda g: self.api(f"{API}/game/{g['gamePk']}/boxscore"),
                          [g for g in self.targets if completed(g)]))
            slate = self.get_slate(self.slate_date)
            if slate.empty:
                return
            tids = set(slate["away_team_id"]) | set(slate["home_team_id"])
            list(pool.map(self._roster, sorted(tids)))
            list(pool.map(self.load_team_pitcher_roles, sorted(tids)))
            pids = set(slate["away_probable_pitcher_id"]) | set(slate["home_probable_pitcher_id"])
            list(pool.map(lambda pid: self.api(f"{API}/people/{int(pid)}/stats", {
                "stats": "gameLog", "group": "pitching", "season": self.season,
                "gameType": "R"}), sorted(pids)))

    def get_slate(self, slate_date, sport_id=1):
        rows = []
        for g in self.targets:
            pk = int(g["gamePk"])
            if not completed(g):
                self.exclusions[pk] = "not_completed_regular_season_game"
                continue
            if available_date(g) != g["_date"]:
                self.exclusions[pk] = "resumed_game_original_pregame_state_unavailable"
                continue
            try:
                box = self.api(f"{API}/game/{pk}/boxscore")
                teams = box["teams"]
                orders = {s: starting_order(teams[s]) for s in ("away", "home")}
                starts = {s: starting_pitcher(teams[s]) for s in ("away", "home")}
            except (MissingHistory, KeyError, TypeError) as exc:
                self.exclusions[pk] = f"historical_lineup_or_starter_unavailable: {exc}"
                continue
            self.boxes[pk], self.orders[pk] = box, orders
            a, h = g["teams"]["away"]["team"], g["teams"]["home"]["team"]
            row = {"game_pk": pk, "game_type": "R", "game_date": slate_date,
                "game_datetime_utc": g["gameDate"], "game_number": g.get("gameNumber"),
                "double_header": g.get("doubleHeader"), "matchup": f"{a['name']} @ {h['name']}",
                "status": "Final", "abstract_state": "Final",
                "savant_preview_url": f"https://baseballsavant.mlb.com/preview?game_pk={pk}"}
            for side, t in (("away", a), ("home", h)):
                row.update({f"{side}_team": t["name"], f"{side}_team_id": t["id"],
                    f"{side}_abbrev": gl._ab(t["name"]),
                    f"{side}_probable_pitcher_id": starts[side][0],
                    f"{side}_probable_pitcher": starts[side][1]})
            rows.append(row)
        return pd.DataFrame(rows)

    def load_stat_lookups(self, player_type, postseason_pas=None):
        board = rate_board(self.pa, player_type, self.slate_date)
        self.boards[player_type] = board
        rename = {"xwoba": "xwOBA", "woba": bs.BLEND_RATE_INTERNAL_COL,
                  "pa": "PA", "bbe": "BBE", "k_percent": "K%", "bb_percent": "BB%"}
        stats = {int(r["player_id"]): {v: r[k] for k, v in rename.items()}
                 for r in board.to_dict("records")}
        return stats, {}, board

    def _roster(self, team_id):
        tid = int(team_id)
        if tid not in self.rosters:
            data = self.api(f"{API}/teams/{tid}/roster",
                            {"rosterType": "active", "date": self.slate_date})
            roster = data.get("roster") or []
            if not roster:
                raise MissingHistory(f"Empty historical active roster: {tid}")
            self.rosters[tid] = roster
        return self.rosters[tid]

    def pitcher_roster(self, team_id):
        return [int(p["person"]["id"]) for p in self._roster(team_id)
                if p.get("position", {}).get("abbreviation") == "P"]

    def resolve_lineup(self, game_pk, side, team_id, batter_stat,
                       return_meta=False, league_xwoba=np.nan):
        roster_ids = [int(p["person"]["id"]) for p in self._roster(team_id)
                      if p.get("position", {}).get("abbreviation") != "P"]
        # Populate the production resolver's roster seam with dated membership;
        # gf is the original order, never the final replacement batting order.
        bs._roster_cache[int(team_id)] = roster_ids
        bs._gf_cache[int(game_pk)] = (self.orders[int(game_pk)]["away"],
                                     self.orders[int(game_pk)]["home"])
        return bs.resolve_lineup(game_pk, side, team_id, batter_stat,
                                  return_meta, league_xwoba)

    def load_team_pitcher_roles(self, team_id):
        return bs.load_team_pitcher_roles(team_id, f"{self.season}-01-01",
            (date.fromisoformat(self.slate_date) - timedelta(days=1)).isoformat())

    def load_recent_start_era(self, ids):
        return bs.load_recent_start_era(ids, before_date=self.slate_date)

    def load_league_era(self):
        # Display only; skip a request that has no prediction reader.
        return np.nan

    def load_pitcher_xera(self):
        return {}  # display only; today's leaderboard is never queried

    def load_splits(self, ids, group):
        raise MissingHistory("OPS lens is disabled in the xwOBA replay")

    def load_people(self, ids):
        return bs.load_people(sorted(ids))

    def load_starter_velocity(self, ids):
        if self._velocity is None:
            self._velocity = {int(pid): sv.per_start(rows.rename(
                columns={"available_date": "_availability"}).assign(
                    game_date=rows["available_date"]))
                for pid, rows in self.raw.groupby("pitcher")}
        return {int(pid): sv.pregame_trend(self._velocity.get(int(pid)), self.slate_date)
                for pid in ids}


def prediction_rows(provider, calibration_history):
    """Use production fetch/aggregate/shrink/blend/phase code, never the ledger."""
    if calibration_history is None:
        raise MissingHistory("Replay requires an explicit prior calibration frame")
    if not calibration_history.empty:
        if "game_date" not in calibration_history:
            raise MissingHistory("Calibration rows require availability dates")
        dates = calibration_history["game_date"].astype(str)
        calibration_history = calibration_history[
            (dates < provider.slate_date) & dates.str.startswith(str(provider.season))].copy()
    with model_environment(provider.api, provider.season, provider.slate_date):
        data = bs.fetch_all(provider.slate_date, provider=provider,
                            calibration_history=calibration_history,
                            include_platoon=False, write_audit=False)
        if data.get("empty"):
            return pd.DataFrame(), data
        if not np.isfinite(data["league_baseline"].get("xwOBA", np.nan)):
            return pd.DataFrame(), data
        frame, _, _ = bs.build_xwoba_matchup(data["pitchers_df"], data["league_baseline"])
        frame = bs.apply_pitching_plans(frame, data["pitching_plans"], data["league_baseline"])
    return frame, data


def game_result(g):
    out = {"full_away": g["teams"]["away"].get("score"),
           "full_home": g["teams"]["home"].get("score"),
           "f5_away": None, "f5_home": None}
    innings = g.get("linescore", {}).get("innings", [])
    first = {int(i["num"]): i for i in innings if int(i.get("num", 0)) <= 5}
    if set(first) == set(range(1, 6)):
        for side in ("away", "home"):
            runs = [first[i].get(side, {}).get("runs") for i in range(1, 6)]
            if all(x is not None for x in runs):
                out[f"f5_{side}"] = sum(runs)
    return out


def assemble_rows(provider, frame, data):
    """One coverage row per scheduled game; calibration actuals are separate."""
    out = []
    calibration = []
    for g in provider.targets:
        pk = int(g["gamePk"])
        away = gl._ab(g["teams"]["away"]["team"]["name"])
        home = gl._ab(g["teams"]["home"]["team"]["name"])
        row = {"game_pk": pk, "game_date": provider.slate_date, "away": away,
            "home": home, "model_tag": bs.MODEL_TAG, "basis": BASIS,
            "input_cutoff": (date.fromisoformat(provider.slate_date) - timedelta(days=1)).isoformat(),
            "lineup_basis": "final_boxscore_original_order",
            "status": "excluded", "exclusion_reason": provider.exclusions.get(pk),
            "xw_lean": None, "xw_net": None, "xw_delta": None,
            "xw_full": None, "xw_f5": None, **game_result(g)}
        sides = frame[frame["game_pk"] == pk] if "game_pk" in frame else pd.DataFrame()
        a, h = bs._rows_by_side(sides) if not sides.empty else (None, None)
        if a is not None and h is not None:
            # Match production: home offence faces the AWAY starter.
            net = bs._f(a.get("edge_xwOBA"))
            other = bs._f(h.get("edge_xwOBA"))
            if net is not None and other is not None:
                net -= other
                row.update(xw_net=net, xw_delta=abs(net),
                           xw_lean=home if net > 0 else away if net < 0 else None,
                           status="graded" if net != 0 else "abstained",
                           exclusion_reason=None if net != 0 else "zero_net")
            else:
                row.update(status="abstained", exclusion_reason=";".join(sorted({
                    str(a.get("pitching_basis")), str(h.get("pitching_basis"))})))
            cal = {"game_date": provider.slate_date, "game_pk": pk}
            for side, side_row in (("away", a), ("home", h)):
                for c in ("expected_sp_ip", "expected_sp_ip_raw", "starter_xwOBA",
                          "bullpen_xwOBA", "opp_xwOBA_neutral", "opp_xwOBA_vs_sp",
                          "sp_share", "starter_velo_dv", "starter_rate_bf"):
                    row[f"{c}_{side}"] = bs._f(side_row.get(c))
                cal[f"expected_sp_ip_raw_{side}"] = bs._f(side_row.get("expected_sp_ip_raw"))
                box = provider.boxes[pk]["teams"][side]
                pid, _ = starting_pitcher(box)
                st = box["players"][f"ID{pid}"]["stats"]["pitching"]
                cal[f"act_sp_ip_{side}"] = bs._innings_to_outs(st.get("inningsPitched")) / 3
            # Completed-game actuals join only AFTER the whole slate predicts.
            calibration.append(cal)
        elif not row["exclusion_reason"]:
            row["exclusion_reason"] = "no_prior_league_history_or_missing_matchup"
        for window, allow_tie in (("full", False), ("f5", True)):
            ra, rh = row[f"{window}_away"], row[f"{window}_home"]
            if ra is not None and rh is not None:
                row[f"xw_{window}"] = gl._wlt(row["xw_lean"], away, home, ra, rh, allow_tie)
        if "lineup_projection_df" in data and not data["lineup_projection_df"].empty:
            lu = data["lineup_projection_df"]
            lu = lu[lu["game_pk"] == pk]
            if len(lu):
                for side in ("away", "home"):
                    row[f"lineup_{side}"] = ",".join(map(str, provider.orders[pk][side]))
                    row[f"backfilled_hitters_{side}"] = int(lu.iloc[0][f"{side}_savant_backfill_count"])
        out.append(row)
    return out, calibration


def report_summary(frame):
    def record(values):
        counts = Counter(values.dropna())
        n = counts["W"] + counts["L"]
        return {"wins": counts["W"], "losses": counts["L"], "ties": counts["T"],
                "decisions": n, "win_rate": counts["W"] / n if n else None}
    full = record(frame["xw_full"])
    f5 = record(frame["xw_f5"])
    paired = frame[frame["xw_full"].isin(["W", "L"])]
    home_wins = int((paired["full_home"] > paired["full_away"]).sum())
    return {"scheduled_games": len(frame), "predicted_games": int(frame["xw_lean"].notna().sum()),
        "slates": int(frame["game_date"].nunique()), "full": full, "f5": f5,
        "always_home_same_games": {"wins": home_wins, "losses": len(paired) - home_wins,
                                   "n": len(paired)},
        "exclusions": dict(Counter(frame["exclusion_reason"].dropna())),
        "native_forward_predictions": 0}


def render_report(frame, meta):
    summary = meta["summary"]
    def esc(x):
        return html.escape(str(x))
    def rec(x):
        return f"{x['wins']}–{x['losses']}" + (f"–{x['ties']}" if x.get("ties") else "")
    groups = []
    for month, rows in frame.groupby(frame["game_date"].str[:7]):
        r = report_summary(rows)
        groups.append(f"<tr><td>{esc(month)}</td><td>{len(rows)}</td><td>{r['predicted_games']}</td>"
                      f"<td>{rec(r['full'])}</td><td>{rec(r['f5'])}</td></tr>")
    games = []
    for r in frame.sort_values(["game_date", "game_pk"], ascending=[False, True]).to_dict("records"):
        games.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [r["game_date"],
            f"{r['away']} @ {r['home']}", r.get("xw_lean") or "—",
            r.get("xw_full") or "—", r.get("xw_f5") or "—",
            r.get("exclusion_reason") or ""]) + "</tr>")
    exclusions = "".join(f"<li>{esc(k)}: {v}</li>" for k, v in summary["exclusions"].items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-season backtest</title>
<style>body{{font:16px system-ui;background:#101820;color:#e8edf3;max-width:1100px;margin:40px auto;padding:0 20px}}
a{{color:#8ed5ff}}h1{{font-size:32px}}.note{{color:#b9c9d6;line-height:1.6}}.stats{{display:flex;gap:28px;flex-wrap:wrap;margin:28px 0}}
.stats strong{{font-size:26px;display:block}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #33414d}}.scroll{{overflow:auto}}details{{margin-top:28px}}</style>
<a href="https://dave356w.github.io/Dave356w/grades.html">← Recorded predictions</a><h1>Full-season backtest</h1>
<p class="note">Retrospective reconstruction · {esc(meta['model_tag'])} · {esc(meta['start'])} through {esc(meta['end'])}.
Season rates use pitches available strictly before each game date. Original starting lineups and starters come from final box scores;
announcement times are unavailable. Historical Statcast values may have been revised. This is not a live track record or an independent forward test.
The fixed model was developed after some of these games. No odds, ROI or calibrated win probabilities are inferred.</p>
<div class="stats"><div><strong>{summary['predicted_games']} / {summary['scheduled_games']}</strong>games with a lean</div>
<div><strong>{rec(summary['full'])}</strong>full game</div><div><strong>{rec(summary['f5'])}</strong>first five (ties separate)</div>
<div><strong>{rec(summary['always_home_same_games'])}</strong>always home, same full-game decisions</div></div>
<p class="note">Calibration begins {esc(meta['start'])}, uses only prior replay slates, and resets each run.
Untracked contact uses observed wOBA as an explicit fallback: {meta['untracked_bip']} historical PAs.
Resumed-game inputs are withheld until their completion date. Source commit: {esc(meta['source_commit'])}.</p>
<p><a href="predictions.csv">Predictions CSV</a> · <a href="manifest.json">Run details and coverage</a></p>
<h2>By month</h2><div class="scroll"><table><thead><tr><th>Month</th><th>Scheduled</th><th>With lean</th><th>Full</th><th>F5</th></tr></thead>
<tbody>{''.join(groups)}</tbody></table></div><h2>Exclusions and abstentions</h2><ul>{exclusions or '<li>None</li>'}</ul>
<details><summary>All games</summary><div class="scroll"><table><thead><tr><th>Date</th><th>Game</th><th>Lean</th><th>Full</th><th>F5</th><th>Reason</th></tr></thead><tbody>{''.join(games)}</tbody></table></div></details></html>"""


def safe_output(path):
    path = Path(path).resolve()
    protected = [Path(bs.DATA_DIR).resolve(), Path(bs.OUT_DIR).resolve()]
    for root in protected:
        if path == root or root in path.parents or path in root.parents:
            raise ValueError("Replay output must be separate from production data/ and public/")
    return path


def run(season, start, end, cache, output, offline=False, workers=3):
    output = safe_output(output)
    cache = safe_output(cache)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "run_status.json", json.dumps({"status": "running",
        "season": season, "start": start, "end": end}))
    api = CachedAPI(Path(cache) / "mlb", offline)
    games = schedule(api, season, end)
    raw = fetch_pitches(cache, season, end, offline, workers)
    raw, pa = prepare_history(raw, games)
    rows, calibration, velocity = [], [], None
    targets = sorted({g["_date"] for g in games.values() if start <= g["_date"] <= end})
    if not targets:
        raise MissingHistory("No regular-season games in requested evaluation window")
    with (output / "replay.log").open("w") as log:
        for day in targets:
            print(f"Replaying {day}", flush=True)
            provider = HistoricalDataProvider(day, season, raw, pa, games, api, velocity)
            prior = pd.DataFrame(calibration)
            n_fail = len(api.failures)
            with redirect_stdout(log):
                if suspension_blocks(games, day):
                    provider.exclusions.update({int(g["gamePk"]):
                        "suspended_game_interim_role_history_unavailable" for g in provider.targets})
                    frame, data = pd.DataFrame(), {}
                else:
                    with model_environment(api, season, day):
                        provider.prefetch(workers)
                    frame, data = prediction_rows(provider, prior)
                if len(api.failures) != n_fail:
                    raise MissingHistory(f"Required MLB lookup failed on {day}; see replay.log")
                day_rows, actuals = assemble_rows(provider, frame, data)
                velocity = provider._velocity
            rows.extend(day_rows)
            calibration.extend(actuals)
            # Checkpoint lives only in the separate output, never in data/.
            atomic_text(output / "checkpoint.csv", pd.DataFrame(rows).to_csv(index=False))
    frame = pd.DataFrame(rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = {"schema": SCHEMA, "basis": BASIS, "model_tag": bs.MODEL_TAG,
        "source_commit": commit, "source_sha256": SOURCE_HASHES,
        "start": start, "end": end, "season": season,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "untracked_bip": int(pa["untracked_bip"].sum()), "historical_pitches": len(raw),
        "historical_pa": len(pa), "calibration_side_games": 2 * len(calibration),
        "summary": report_summary(frame)}
    atomic_text(output / "predictions.csv", frame.to_csv(index=False))
    atomic_text(output / "backtest.html", render_report(frame, meta))
    meta["output_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                             for name in ("predictions.csv", "backtest.html")}
    atomic_text(output / "manifest.json", json.dumps(meta, indent=2, allow_nan=False))
    atomic_text(output / "run_status.json", json.dumps({"status": "complete",
        "season": season, "start": start, "end": end}))
    print(json.dumps(meta["summary"], indent=2), flush=True)
    return frame, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=date.today().year)
    p.add_argument("--start", help="first evaluation date; rates always warm up from March 1")
    p.add_argument("--end", help="last historical date, at most yesterday ET")
    p.add_argument("--cache", default="replay_cache")
    p.add_argument("--output", default="replay_output")
    p.add_argument("--offline", action="store_true", help="require all input requests in cache")
    p.add_argument("--workers", type=int, default=3, choices=range(1, 7))
    a = p.parse_args()
    yesterday = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    start = a.start or f"{a.season}-03-01"
    end = a.end or min(yesterday, date(a.season, 10, 5)).isoformat()
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first.year != a.season or last.year != a.season or first > last or last > yesterday:
        p.error("Use an ordered historical date range within the selected season")
    run(a.season, start, end, a.cache, a.output, a.offline, a.workers)


if __name__ == "__main__":
    main()
