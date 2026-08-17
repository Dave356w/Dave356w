#!/usr/bin/env python3
"""Point-in-time inputs for the current MLB matchup model.

This module owns reconstruction only.  It deliberately imports the live model
and implements the same provider surface as ``build_site.LiveDataProvider``;
all shrinkage, lineup aggregation, starter/bullpen blending, abstention and
lean math continue to run in ``build_site.py``.
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import build_site as model


SAVANT_SEARCH_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
FIDELITY_ORDER = {
    "exact_pregame": 0,
    "historical_lineup": 1,
    "reconstructed": 2,
    "insufficient_history": 3,
}


def _norm_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _number(value, default=np.nan):
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) else default


class HistoricalDataProvider:
    """Date-bounded data provider for one replay slate.

    Savant inputs end on ``slate_date - 1``.  StatsAPI workload/role queries
    use ``byDateRange`` with the same cutoff.  Starting-lineup ids come from
    the historical gamefeed; the source ledger's recorded starter names remain
    authoritative over a retroactively corrected probable-pitcher field.
    """

    def __init__(self, slate_date, source_rows, cache_dir=".walkforward_cache",
                 snapshot_dir=None, session=None):
        self.slate_date = str(slate_date)[:10]
        self.as_of_date = (
            date.fromisoformat(self.slate_date) - timedelta(days=1)
        ).isoformat()
        self.season = int(self.slate_date[:4])
        # March 1 safely precedes the regular season; hfGT=R excludes spring.
        self.season_start = f"{self.season}-03-01"
        self.source_rows = pd.DataFrame(source_rows).copy()
        if "game_pk" in self.source_rows:
            self.source_rows["game_pk"] = pd.to_numeric(
                self.source_rows["game_pk"], errors="coerce"
            ).astype("Int64")
        self.source_by_game = {
            int(r["game_pk"]): r
            for _, r in self.source_rows.dropna(subset=["game_pk"]).iterrows()
        }
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self._stat_fidelity = {}
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; mlb-walkforward/1.0)",
        })
        self._savant_frames = {}
        self._dated_rosters = {}
        self._player_ids = {}
        self._lineup_fidelity = {}
        self._starter_fidelity = {}
        self._errors = {}

    # ---- Savant ---------------------------------------------------------
    def _savant_params(self, player_type, extra=None):
        params = {
            "all": "true",
            "player_type": player_type,
            "game_date_gt": self.season_start,
            "game_date_lt": self.as_of_date,
            "group_by": "name",
            "min_pas": 1,
            "min_pitches": 0,
            "min_results": 0,
            "hfGT": "R|",
            "chk_stats_xwoba": "on",
            "chk_stats_woba": "on",
        }
        params.update(extra or {})
        return params

    def _savant_cache_path(self, player_type, extra=None):
        suffix = ""
        if extra:
            safe = "_".join(
                f"{k}-{v}" for k, v in sorted(extra.items())
            )
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", safe)
            suffix = f"_{safe}"
        return self.cache_dir / (
            f"savant_{player_type}_{self.season_start}_{self.as_of_date}"
            f"{suffix}.csv"
        )

    def _download_savant(self, player_type, extra=None):
        path = self._savant_cache_path(player_type, extra)
        if path.exists() and path.stat().st_size > 100:
            return path
        response = self.session.get(
            SAVANT_SEARCH_CSV,
            params=self._savant_params(player_type, extra),
            timeout=120,
        )
        response.raise_for_status()
        if len(response.content) < 100 or b"player_id" not in response.content[:1000]:
            raise RuntimeError(
                f"Savant returned an invalid grouped export for {player_type} "
                f"through {self.as_of_date}"
            )
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(response.content)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return path

    def _savant_table(self, player_type, extra=None):
        key = (player_type, tuple(sorted((extra or {}).items())))
        if key in self._savant_frames:
            return self._savant_frames[key]
        path = self._download_savant(player_type, extra)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"player_id", "xwoba", "pa"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Historical Savant export missing {sorted(missing)}: {path}"
            )
        frame["player_id"] = pd.to_numeric(
            frame["player_id"], errors="coerce"
        ).astype("Int64")
        self._savant_frames[key] = frame
        return frame

    def _exact_snapshot_table(self, player_type):
        """Load the production build's raw dated cache when it was archived."""
        if self.snapshot_dir is None:
            return None
        names = (
            f"savant_cache_custom_xwoba_v1_{player_type}_{self.slate_date}.csv",
            f"custom_xwoba_v1_{player_type}_{self.slate_date}.csv",
        )
        candidates = tuple(
            root / name
            for root in (self.snapshot_dir, self.snapshot_dir / ".savant_cache")
            for name in names
        )
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"player_id", "xwoba", "pa"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Exact Savant snapshot missing {sorted(missing)}: {path}"
            )
        return frame

    def load_stat_lookups(self, player_type):
        exact = self._exact_snapshot_table(player_type)
        self._stat_fidelity[player_type] = (
            "exact_pregame" if exact is not None else "historical_lineup"
        )
        frame = (exact if exact is not None
                 else self._savant_table(player_type)).copy()
        # Normalize grouped-search labels to the custom-leaderboard labels the
        # live parser and league-baseline code already consume.
        aliases = {
            "launch_speed": "exit_velocity_avg",
            "launch_angle": "launch_angle_avg",
            "hardhit_percent": "hard_hit_percent",
        }
        for source, target in aliases.items():
            if source in frame and target not in frame:
                frame[target] = frame[source]

        stat = {}
        for _, row in frame.dropna(subset=["player_id"]).iterrows():
            pid = int(row["player_id"])
            stat[pid] = {
                "xwOBA": _number(row.get("xwoba")),
                "xBA": _number(row.get("xba")),
                "xSLG": _number(row.get("xslg")),
                "EV": _number(row.get("launch_speed")),
                "LA°": _number(row.get("launch_angle")),
                "Hard Hit%": _number(row.get("hardhit_percent")),
                "K%": _number(row.get("k_percent")),
                "BB%": _number(row.get("bb_percent")),
                "PA": _number(row.get("pa"), 0.0),
                "BBE": _number(row.get("bip"), 0.0),
            }
        # Batted-ball direction is display-only and not part of the v12 lean.
        return stat, {}, frame

    # ---- Slate, identity and lineup -------------------------------------
    def _resolve_player_id(self, name, scheduled_id=None, scheduled_name=None):
        key = _norm_name(name)
        if not key:
            return None
        if key in self._player_ids:
            return self._player_ids[key]
        if scheduled_id is not None and key == _norm_name(scheduled_name):
            self._player_ids[key] = int(scheduled_id)
            return int(scheduled_id)
        data = model._get_json(
            "https://statsapi.mlb.com/api/v1/people/search",
            {"names": str(name), "sportIds": model.SPORT_ID},
        )
        candidates = data.get("people", [])
        exact = [p for p in candidates if _norm_name(p.get("fullName")) == key]
        pitchers = [
            p for p in (exact or candidates)
            if (p.get("primaryPosition") or {}).get("abbreviation") == "P"
        ]
        match = (pitchers or exact or candidates)
        pid = int(match[0]["id"]) if match else None
        self._player_ids[key] = pid
        return pid

    def get_slate(self, slate_date, sport_id=model.SPORT_ID):
        slate = model.get_slate(slate_date, sport_id)
        wanted = set(self.source_by_game)
        slate = slate[
            pd.to_numeric(slate.get("game_pk"), errors="coerce").isin(wanted)
        ].copy()
        present = set(pd.to_numeric(slate.get("game_pk"), errors="coerce").dropna().astype(int))
        for missing in sorted(wanted - present):
            self._errors[missing] = "game_not_found_on_historical_schedule"

        for idx, game in slate.iterrows():
            gpk = int(game["game_pk"])
            source = self.source_by_game[gpk]
            for side in ("away", "home"):
                recorded = source.get(f"{side}_sp")
                scheduled_id = game.get(f"{side}_probable_pitcher_id")
                scheduled_name = game.get(f"{side}_probable_pitcher")
                try:
                    pid = self._resolve_player_id(
                        recorded, scheduled_id=scheduled_id,
                        scheduled_name=scheduled_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    pid = None
                    self._errors[gpk] = f"starter_resolution_failed:{exc!r}"
                slate.at[idx, f"{side}_probable_pitcher"] = recorded
                slate.at[idx, f"{side}_probable_pitcher_id"] = (
                    pid if pid is not None else np.nan
                )
                self._starter_fidelity[(gpk, side)] = (
                    "exact_pregame" if pid is not None
                    else "insufficient_history"
                )
        return slate.reset_index(drop=True)

    def _dated_roster(self, team_id):
        team_id = int(team_id)
        if team_id in self._dated_rosters:
            return self._dated_rosters[team_id]
        data = model._get_json(
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
            {"rosterType": "active", "date": self.slate_date},
        )
        rows = []
        for row in data.get("roster", []):
            person = row.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            rows.append({
                "id": int(pid),
                "position": (row.get("position") or {}).get("abbreviation"),
            })
        self._dated_rosters[team_id] = rows
        return rows

    def pitcher_roster(self, team_id):
        return [r["id"] for r in self._dated_roster(team_id)
                if r["position"] == "P"]

    def _hitter_roster(self, team_id, batter_stat):
        ids = [r["id"] for r in self._dated_roster(team_id)
               if r["position"] != "P" and r["id"] in batter_stat]
        return sorted(
            ids,
            key=lambda pid: _number((batter_stat.get(pid) or {}).get("PA"), 0.0),
            reverse=True,
        )

    def _team_rate(self, team_id, batter_stat, fallback=np.nan):
        values, weights = [], []
        for pid in self._hitter_roster(team_id, batter_stat):
            row = batter_stat.get(pid) or {}
            rate, pa = _number(row.get("xwOBA")), _number(row.get("PA"), 0.0)
            if pd.notna(rate) and pa > 0:
                values.append(rate)
                weights.append(pa)
        if values:
            return float(np.average(values, weights=weights))
        return _number(fallback)

    def resolve_lineup(self, game_pk, side, team_id, batter_stat,
                       return_meta=False, league_xwoba=np.nan):
        game_pk = int(game_pk)
        away, home = model.gf_lineups(game_pk)
        raw = away if side == "away" else home
        posted, seen = [], set()
        for value in raw:
            try:
                pid = int(value)
            except (TypeError, ValueError):
                continue
            if pid not in seen:
                posted.append(pid)
                seen.add(pid)
            if len(posted) >= model.LINEUP_SIZE:
                break

        backfilled = []
        missing = [
            pid for pid in posted
            if pd.isna(_number((batter_stat.get(pid) or {}).get("xwOBA")))
        ]
        if missing:
            team_rate = self._team_rate(team_id, batter_stat, league_xwoba)
            for pid in missing:
                batter_stat[pid] = {
                    **(batter_stat.get(pid) or {}),
                    "xwOBA": team_rate,
                    "PA": 0.0,
                    "BBE": 0.0,
                    model.MODEL_RATE_TEAM_BACKFILL_COL: True,
                }
                backfilled.append(pid)

        resolved = list(posted)
        for pid in self._hitter_roster(team_id, batter_stat):
            if len(resolved) >= model.LINEUP_SIZE:
                break
            if pid not in seen:
                resolved.append(pid)
                seen.add(pid)

        source = self.source_by_game.get(game_pk)
        source_status = None if source is None else source.get(f"lineup_status_{side}")
        if len(resolved) < model.LINEUP_SIZE:
            fidelity = "insufficient_history"
        elif len(posted) < model.LINEUP_SIZE:
            fidelity = "reconstructed"
        elif str(source_status).lower() == "posted":
            fidelity = "exact_pregame"
        else:
            fidelity = "historical_lineup"
        self._lineup_fidelity[(game_pk, side)] = fidelity

        posted_count = min(len(posted), model.LINEUP_SIZE)
        filled_count = max(0, len(resolved) - posted_count)
        status = (
            "posted" if posted_count >= model.LINEUP_SIZE
            else "partial_filled" if posted_count else "projected"
        )
        meta = {
            "status": status,
            "projected": status != "posted",
            "posted_count": int(posted_count),
            "filled_count": int(filled_count),
            "resolved_count": int(len(resolved)),
            "savant_backfill_count": int(len(backfilled)),
        }
        answer = resolved[:model.LINEUP_SIZE]
        return (answer, meta) if return_meta else (answer, meta["projected"])

    # ---- Other point-in-time inputs -------------------------------------
    def load_recent_start_era(self, ids):
        return model.load_recent_start_era(
            ids, before_date=self.slate_date
        )

    def load_team_pitcher_roles(self, team_id):
        return model.load_team_pitcher_roles(
            team_id, start_date=self.season_start, end_date=self.as_of_date
        )

    def load_league_era(self):
        return model.load_league_era(
            start_date=self.season_start, end_date=self.as_of_date
        )

    def load_people(self, ids):
        # Handedness/identity are biographical fields, not season-to-date
        # outcomes.  The same endpoint is safe for live and replay providers.
        return model.load_people(ids)

    def _split_frame(self, group, hand):
        if group == "hitting":
            return self._savant_table(
                "batter", {"pitcher_throws": hand}
            )
        return self._savant_table(
            "pitcher", {"batter_stands": hand}
        )

    @staticmethod
    def _split_values(row):
        return {
            "ops": _number(row.get("ops")),
            "obp": _number(row.get("obp")),
            "slg": _number(row.get("slg")),
            "era": _number(row.get("era")),
            "pa": int(_number(row.get("pa"), 0.0)),
            "k": _number(row.get("so")),
            "bb": _number(row.get("bb")),
        }

    def load_splits(self, ids, group):
        """Date-bounded vL/vR splits from grouped Savant exports.

        Walk-forward's decision-only default does not request these, but the
        provider supports the optional platoon lens without falling back to a
        current full-season StatsAPI split.
        """
        wanted = {int(v) for v in ids if pd.notna(v)}
        overall = self._savant_table(
            "batter" if group == "hitting" else "pitcher"
        ).set_index("player_id", drop=False)
        left = self._split_frame(group, "L").set_index("player_id", drop=False)
        right = self._split_frame(group, "R").set_index("player_id", drop=False)
        out = {}
        for pid in wanted:
            rec = {"L": {}, "R": {}, "overall": {}}
            if pid in left.index:
                rec["L"] = self._split_values(left.loc[pid])
            if pid in right.index:
                rec["R"] = self._split_values(right.loc[pid])
            if pid in overall.index:
                values = self._split_values(overall.loc[pid])
                rec["overall"] = {
                    "ops": values["ops"], "pa": values["pa"],
                    "era": values["era"],
                }
            out[pid] = rec
        return out

    def load_pitcher_xera(self):
        # xERA is display-only and the expected-statistics leaderboard has no
        # historical cutoff. Returning no display value is safer than leaking.
        return {}

    # ---- Audit -----------------------------------------------------------
    def input_fidelity(self, game_pk):
        game_pk = int(game_pk)
        labels = [
            self._stat_fidelity.get("batter", "historical_lineup"),
            self._stat_fidelity.get("pitcher", "historical_lineup"),
        ]
        for side in ("away", "home"):
            labels.append(self._starter_fidelity.get(
                (game_pk, side), "insufficient_history"
            ))
            labels.append(self._lineup_fidelity.get(
                (game_pk, side), "insufficient_history"
            ))
        return max(labels, key=lambda label: FIDELITY_ORDER[label])

    def error_for(self, game_pk):
        return self._errors.get(int(game_pk))
