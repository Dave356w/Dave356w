#!/usr/bin/env python3
"""Daily entry point: build the Yahoo top-N position rankings and publish JSON.

Writes into `site/`, which GitHub Pages serves:

    site/data/latest.json          the current slate, read by index.html
    site/data/latest.csv           the same rankings as a flat download
    site/data/history/<date>.json  one archived copy per run date
    site/data/index.json           the archive listing, newest first

Failure policy mirrors the build: on any error nothing is written and the
process exits non-zero, so the previously published page stays live rather
than being replaced by a half-built one. The workflow surfaces the failure.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import notebook as nb

SITE = Path(__file__).resolve().parent / "site"
DATA = SITE / "data"
HISTORY = DATA / "history"

# Column in the ranking view -> key in the published JSON.
FIELDS = {
    "Rank": "rank",
    "Player": "player",
    "Team": "team",
    "Opponent": "opponent",
    "Estimated FP": "fp",
    "Yahoo FPPG": "fppg",
    "Salary": "salary",
    "FP / salary": "fp_per_salary",
    "Depth": "depth",
    "Kickoff UTC": "kickoff_utc",
    "Projection method": "method",
}


class _Tee(io.TextIOBase):
    """Echo the run's own progress output while keeping a copy for the page."""

    def __init__(self, stream):
        self._stream = stream
        self.lines: list[str] = []
        self._buffer = ""

    def write(self, text):
        self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.lines.append(line.rstrip())
        return len(text)

    def flush(self):
        self._stream.flush()


def _clean(value):
    """JSON has no NaN. Nulls render as an em dash on the page."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else str(value)
    if hasattr(value, "item"):          # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _rows(view):
    out = []
    for record in view.to_dict("records"):
        out.append({key: _clean(record.get(column)) for column, key in FIELDS.items()})
    return out


def _games(frame):
    if frame is None or frame.empty:
        return []
    games = []
    for record in frame.to_dict("records"):
        games.append({
            "game_id": _clean(record.get("Game ID")),
            "matchup": _clean(record.get("Matchup")),
            "away": _clean(record.get("Away Team")),
            "home": _clean(record.get("Home Team")),
            "kickoff_utc": _clean(record.get("Game Time")),
        })
    return games


def _market_summary(results):
    report = results.get("market_report")
    audit = results.get("market_audit") or {}
    summary = {
        "feeds": list(audit.get("feeds") or []),
        "notes": list(audit.get("notes") or []),
        "matched": None,
        "accepted": None,
        "skill_rows": None,
        "calibration_pairs": _clean(audit.get("calibration_pairs")),
    }
    if isinstance(report, pd.DataFrame) and not report.empty:
        summary["matched"] = int(report["Market matched"].sum())
        summary["accepted"] = int(report["Market accepted"].sum())
        summary["skill_rows"] = int(report["Position"].ne("DEF").sum())
    return summary


def _method_counts(results):
    counts = {}
    players = results.get("players")
    if isinstance(players, pd.DataFrame) and "Projection_Source" in players:
        raw = players["Projection_Source"].astype(str).value_counts()
        counts = {str(k): int(v) for k, v in raw.items()}
    return counts


def build_payload(results, top_n, log_lines):
    now = datetime.now(timezone.utc)
    rankings = {
        position: _rows(view)
        for position, view in results.get("rankings", {}).items()
    }
    games = _games(results.get("games"))
    removed = results.get("nflverse_removed")
    return {
        "schema": 1,
        "status": "ok",
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": now.strftime("%Y-%m-%d"),
        "top_n": int(top_n),
        "positions": [p for p in rankings if rankings[p]],
        "slate": {
            "games": games,
            "game_count": len(games),
            "player_count": int(len(results.get("players", []))),
            "first_kickoff_utc": games[0]["kickoff_utc"] if games else None,
        },
        "rankings": rankings,
        "market": _market_summary(results),
        "availability_removed": int(len(removed)) if isinstance(removed, pd.DataFrame) else 0,
        "method_counts": _method_counts(results),
        "log": log_lines,
    }


def write_outputs(payload, combined):
    DATA.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, indent=2, sort_keys=False)
    (DATA / "latest.json").write_text(text + "\n", encoding="utf-8")
    (HISTORY / f"{payload['run_date']}.json").write_text(text + "\n", encoding="utf-8")

    if isinstance(combined, pd.DataFrame) and not combined.empty:
        combined.to_csv(DATA / "latest.csv", index=False)
        combined.to_csv(HISTORY / f"{payload['run_date']}.csv", index=False)

    entries = []
    for path in sorted(HISTORY.glob("*.json"), reverse=True):
        try:
            archived = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append({
            "date": path.stem,
            "generated_utc": archived.get("generated_utc"),
            "game_count": archived.get("slate", {}).get("game_count"),
            "json": f"history/{path.name}",
            "csv": f"history/{path.stem}.csv" if (HISTORY / f"{path.stem}.csv").exists() else None,
        })
    (DATA / "index.json").write_text(
        json.dumps({"runs": entries}, indent=2) + "\n", encoding="utf-8"
    )
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=25,
                        help="players published per position (default 25)")
    parser.add_argument("--positions", default=",".join(nb.VALID_POSITIONS),
                        help="comma-separated positions to publish")
    args = parser.parse_args(argv)

    positions = tuple(p.strip() for p in args.positions.split(",") if p.strip())
    tee = _Tee(sys.stdout)

    try:
        with contextlib.redirect_stdout(tee):
            results = nb.run_position_rankings(
                top_n=args.top_n, positions=positions, export_csv=False
            )
    except Exception:
        traceback.print_exc()
        print(
            "\nRun failed; no files were written. The previously published "
            "page stays live.",
            file=sys.stderr,
        )
        return 1

    payload = build_payload(results, args.top_n, tee.lines)
    if not any(payload["rankings"].values()):
        print("Run produced no ranked players; refusing to publish.", file=sys.stderr)
        return 1

    entries = write_outputs(payload, results.get("combined"))
    total = sum(len(rows) for rows in payload["rankings"].values())
    print(
        f"\nPublished {total} ranked players across "
        f"{len(payload['positions'])} position(s); {len(entries)} run(s) archived."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
