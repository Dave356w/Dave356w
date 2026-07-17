import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import build_site
import grade_leans
import schedule_gate


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
    def test_v2_and_v3_share_one_record_family(self):
        ledger = pd.DataFrame([
            dict(game_pk=1, status="graded", model_tag="xw+plat_consol_v1"),
            dict(game_pk=2, status="graded", model_tag="xw+plat_consol_v2"),
            dict(game_pk=3, status="graded", model_tag="xw+plat_consol_v3"),
            dict(game_pk=4, status="pending", model_tag="xw+plat_consol_v3"),
        ])
        self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {2, 3})
        self.assertEqual(set(grade_leans._record_grades(ledger)["game_pk"]), {2, 3})

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
    def test_games_sort_chronologically_with_stable_doubleheader_ties(self):
        games = [
            dict(game_pk=30, game_number=1, game_datetime_utc="2026-07-17T23:10:00Z"),
            dict(game_pk=20, game_number=2, game_datetime_utc="2026-07-17T20:05:00Z"),
            dict(game_pk=10, game_number=1, game_datetime_utc="2026-07-17T20:05:00Z"),
            dict(game_pk=40, game_number=1, game_datetime_utc=None),
        ]
        ordered = sorted(games, key=build_site._game_order_key)
        self.assertEqual([g["game_pk"] for g in ordered], [10, 20, 30, 40])

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


class ScheduleGateTests(unittest.TestCase):
    NOW = pd.Timestamp("2026-07-17T16:00:00Z").to_pydatetime()

    @staticmethod
    def game(game_pk, minutes, state="Preview"):
        start = ScheduleGateTests.NOW + pd.Timedelta(minutes=minutes)
        return {
            "gamePk": game_pk,
            "gameDate": start.isoformat().replace("+00:00", "Z"),
            "status": {"abstractGameState": state},
        }

    def test_scheduled_poll_runs_near_first_pitch(self):
        run, day, reason = schedule_gate.decision(
            "schedule", "7,22,37,52 10-23 * * *", self.NOW,
            games=[self.game(101, 30)],
        )
        self.assertTrue(run)
        self.assertEqual(day, "2026-07-17")
        self.assertIn("101", reason)

    def test_scheduled_poll_skips_outside_window_and_final_games(self):
        games = [self.game(101, 10), self.game(102, 30, state="Final")]
        run, _, reason = schedule_gate.decision(
            "schedule", "7,22,37,52 10-23 * * *", self.NOW, games=games,
        )
        self.assertFalse(run)
        self.assertIn("no game", reason)

    def test_grade_push_and_manual_events_always_run(self):
        grade = schedule_gate.decision("schedule", schedule_gate.DAILY_GRADE_CRON,
                                       self.NOW, games=[])
        push = schedule_gate.decision("push", "", self.NOW, games=[])
        manual = schedule_gate.decision("workflow_dispatch", "", self.NOW, games=[])
        self.assertTrue(grade[0])
        self.assertTrue(push[0])
        self.assertTrue(manual[0])


if __name__ == "__main__":
    unittest.main()
