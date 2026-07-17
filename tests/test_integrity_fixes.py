import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import build_site
import grade_leans


def _dump_rows(game_pk, game_date, snapshot, start):
    common = dict(
        game_pk=game_pk,
        game_date=game_date,
        snapshot_utc=snapshot,
        scheduled_start_utc=start,
        model_tag="test_v3",
    )
    return pd.DataFrame([
        dict(common, side="away", pitcher="Away Pitcher", opp_team="Home Team",
             opp_xwOBA=.330, pit_xwOBA=.310, edge_xwOBA=.004),
        dict(common, side="home", pitcher="Home Pitcher", opp_team="Away Team",
             opp_xwOBA=.320, pit_xwOBA=.300, edge_xwOBA=-.016),
    ])


class LedgerLockTests(unittest.TestCase):
    def test_load_ledger_preserves_market_history(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.csv")
            pd.DataFrame([dict(
                game_pk=321,
                game_date="2026-07-16",
                status="graded",
                close_home_ml=-120,
                close_p_home=.54545,
            )]).to_csv(path, index=False)
            with mock.patch.object(grade_leans, "LEDGER_PATH", path):
                ledger = grade_leans.load_ledger()
            self.assertEqual(ledger.at[0, "close_home_ml"], -120)
            self.assertAlmostEqual(ledger.at[0, "close_p_home"], .54545)

    def test_snapshot_lock_status(self):
        self.assertEqual(
            grade_leans._lock_status("2026-07-17T18:00:00Z", "2026-07-17T19:00:00Z"),
            "pregame",
        )
        self.assertEqual(
            grade_leans._lock_status("2026-07-17T19:00:00Z", "2026-07-17T19:00:00Z"),
            "late_snapshot",
        )
        self.assertEqual(grade_leans._lock_status(None, None), "legacy_unverified")

    def test_rescheduled_date_gets_distinct_row(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                123,
                "2026-07-11",
                "2026-07-11T12:00:00Z",
                "2026-07-11T16:00:00Z",
            )
            xw.to_csv(os.path.join(td, "leans_2026-07-11_xw.csv"), index=False)
            old = {c: None for c in grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS}
            old.update(game_pk=123, game_date="2026-07-10", status="void")
            ledger = pd.DataFrame([old])
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            self.assertEqual(len(out), 2)
            self.assertEqual(set(out["game_date"]), {"2026-07-10", "2026-07-11"})
            new = out[out["game_date"] == "2026-07-11"].iloc[0]
            self.assertEqual(new["lock_status"], "pregame")

    def test_late_new_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                456,
                "2026-07-17",
                "2026-07-17T20:01:00Z",
                "2026-07-17T20:00:00Z",
            )
            xw.to_csv(os.path.join(td, "leans_2026-07-17_xw.csv"), index=False)
            ledger = pd.DataFrame(columns=grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            self.assertTrue(out.empty)


class SlateCompletenessTests(unittest.TestCase):
    def test_game_without_paired_probables_renders_placeholder(self):
        slate = pd.DataFrame([dict(
            game_pk=789,
            away_abbrev="TB",
            home_abbrev="BOS",
            away_team="Tampa Bay Rays",
            home_team="Boston Red Sox",
            away_probable_pitcher=None,
            home_probable_pitcher=None,
            game_datetime_utc="2026-07-17T23:10:00Z",
            game_number=2,
            double_header="Y",
            venue="Fenway Park",
        )])
        html = build_site.render_combined_html(
            pd.DataFrame(columns=["game_pk", "side"]),
            pd.DataFrame(),
            pd.DataFrame(),
            "test build",
            slate_df=slate,
        )
        self.assertIn("Awaiting paired probable pitchers", html)
        self.assertIn("G2", html)
        self.assertIn("TB", html)
        self.assertIn("BOS", html)


if __name__ == "__main__":
    unittest.main()
