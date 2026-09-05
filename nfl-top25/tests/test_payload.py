"""Shape tests for the published JSON.

The pipeline itself needs the live Yahoo, sportsbook and nflverse feeds, so it
cannot be exercised here. What can be pinned without a network is the contract
between `run_daily.build_payload` and `site/index.html`: every key the page
reads, the NaN handling that would otherwise emit invalid JSON, and the archive
index the history picker walks.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_daily


def _view(rows):
    return pd.DataFrame(rows, columns=list(run_daily.FIELDS))


def _results():
    qb = _view([
        [1, "Josh Allen", "BUF", "@MIA", 22.41, 21.0, 40, 0.56, 1,
         "2026-09-07 17:00", "market"],
        [2, "Lamar Jackson", "BAL", "vs KC", 21.02, np.nan, 38, 0.553, 1,
         "2026-09-07 20:25", "yahoo-prior"],
    ])
    rb = _view([
        [1, "Bijan Robinson", "ATL", "vs TB", 18.9, 17.4, 35, 0.54, 1,
         "2026-09-07 17:00", "market"],
    ])
    games = pd.DataFrame([
        {"Game ID": "g1", "Game Time": "2026-09-07 17:00", "Away Team": "BUF",
         "Home Team": "MIA", "Matchup": "BUF vs MIA"},
    ])
    players = pd.DataFrame({
        "Name": ["Josh Allen", "Lamar Jackson", "Bijan Robinson"],
        "Projection_Source": ["market", "yahoo-prior", "market"],
    })
    market_report = pd.DataFrame({
        "Market matched": [True, True, False],
        "Market accepted": [True, False, False],
        "Position": ["QB", "QB", "DEF"],
    })
    return {
        "rankings": {"QB": qb, "RB": rb, "WR": _view([]), "TE": _view([]),
                     "DEF": _view([])},
        "combined": pd.concat(
            [qb.assign(Position="QB"), rb.assign(Position="RB")], ignore_index=True
        ),
        "players": players,
        "games": games,
        "market_audit": {"feeds": ["underdog", "bovada"], "notes": ["cached"],
                         "calibration_pairs": 41},
        "market_report": market_report,
        "nflverse_removed": pd.DataFrame({"Name": ["Someone"]}),
    }


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.payload = run_daily.build_payload(_results(), 25, ["line one"])

    def test_it_serializes_without_nan(self):
        # json.dumps emits a bare NaN token, which JSON.parse rejects in the
        # browser. A missing Yahoo FPPG must land as null, not NaN.
        text = json.dumps(self.payload)
        self.assertNotIn("NaN", text)
        self.assertIsNone(self.payload["rankings"]["QB"][1]["fppg"])

    def test_it_carries_every_key_the_page_reads(self):
        for key in ("status", "generated_utc", "run_date", "top_n", "positions",
                    "slate", "rankings", "market", "method_counts", "log"):
            self.assertIn(key, self.payload)
        row = self.payload["rankings"]["QB"][0]
        for key in run_daily.FIELDS.values():
            self.assertIn(key, row)

    def test_empty_positions_are_not_advertised(self):
        # The page builds its tab strip from `positions`; a tab that opens on an
        # empty table is worse than an absent tab.
        self.assertEqual(self.payload["positions"], ["QB", "RB"])

    def test_market_counts_come_from_the_report(self):
        self.assertEqual(self.payload["market"]["matched"], 2)
        self.assertEqual(self.payload["market"]["accepted"], 1)
        self.assertEqual(self.payload["market"]["skill_rows"], 2)
        self.assertEqual(self.payload["availability_removed"], 1)

    def test_write_outputs_archives_and_indexes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_daily.DATA = root / "data"
            run_daily.HISTORY = root / "data" / "history"
            entries = run_daily.write_outputs(self.payload, _results()["combined"])

            latest = json.loads((run_daily.DATA / "latest.json").read_text())
            self.assertEqual(latest["run_date"], self.payload["run_date"])
            self.assertTrue((run_daily.DATA / "latest.csv").exists())
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["date"], self.payload["run_date"])
            self.assertTrue(entries[0]["csv"].endswith(".csv"))

            index = json.loads((run_daily.DATA / "index.json").read_text())
            self.assertEqual(index["runs"], entries)


class LogTeeTests(unittest.TestCase):
    def test_it_splits_lines_and_drops_blanks(self):
        import io as _io

        tee = run_daily._Tee(_io.StringIO())
        tee.write("first\n\nsecond line\n")
        tee.write("partial")
        self.assertEqual(tee.lines, ["first", "second line"])


if __name__ == "__main__":
    unittest.main()
