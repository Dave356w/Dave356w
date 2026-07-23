import os
import tempfile
import unittest
from unittest import mock

import numpy as np
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
            dict(game_pk=5, status="graded", model_tag="xw+plat_consol_v4"),
            dict(game_pk=6, status="graded", model_tag="xw+plat_consol_v5"),
            dict(game_pk=7, status="graded", model_tag="xw+plat_consol_v6"),
        ])
        v3_family = ("xw+plat_consol_v2", "xw+plat_consol_v3")
        with mock.patch.object(build_site, "RECORD_TAGS", v3_family), \
                mock.patch.object(grade_leans, "RECORD_TAGS", v3_family):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {2, 3})
            self.assertEqual(set(grade_leans._record_grades(ledger)["game_pk"]), {2, 3})
        # v4 re-weights the lineup composites (slot-PA), so it starts a fresh
        # record family and never mixes with the v2/v3 prediction math.
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v4",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {5})
        # v5 and v6 each changed prediction math and remain isolated.
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v5",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {6})
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v6",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {7})
        self.assertEqual(
            [label for label, _ in build_site._model_family_grades(ledger)],
            ["v2/v3", "v4", "v5", "v6", "xw+plat_consol_v1"],
        )

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

    def test_grades_page_preserves_family_history_and_row_labels(self):
        ledger = pd.DataFrame([
            dict(game_pk=1, game_date="2026-07-20", away="A", home="B",
                 away_sp="P1", home_sp="P2", status="graded",
                 model_tag="xw+plat_consol_v5", xw_lean="A", xw_delta=.01,
                 xw_full="W", xw_f5="T", full_away=4, full_home=2),
            dict(game_pk=2, game_date="2026-07-21", away="C", home="D",
                 away_sp="P3", home_sp="P4", status="graded",
                 model_tag="xw+plat_consol_v6", xw_lean="D", xw_delta=.02,
                 xw_full="L", xw_f5="L", full_away=5, full_home=3),
        ])
        history = build_site.model_family_history_html(ledger)
        self.assertIn("Model-family history", history)
        self.assertIn("<td>v5</td>", history)
        self.assertIn("<td>v6</td>", history)
        self.assertIn("1-0 (1.000)", history)
        self.assertIn("0-1 (0.000)", history)
        row = build_site._grades_row(ledger.iloc[1], show_model=True)
        self.assertIn("<td>v6</td>", row)

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

    def test_lineup_status_columns_carry_into_ledger_and_refresh(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                777,
                "2026-07-20",
                "2026-07-20T12:00:00Z",
                "2026-07-20T23:00:00Z",
            )
            xw["lineup_status_away"] = "projected"
            xw["lineup_status_home"] = "posted"
            xw["lineup_posted_away"] = 0
            xw["lineup_posted_home"] = 9
            path = os.path.join(td, "leans_2026-07-20_xw.csv")
            xw.to_csv(path, index=False)
            ledger = pd.DataFrame(columns=grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            row = out.iloc[0]
            self.assertEqual(row["lineup_status_away"], "projected")
            self.assertEqual(row["lineup_status_home"], "posted")
            self.assertEqual(int(row["lineup_posted_away"]), 0)
            self.assertEqual(int(row["lineup_posted_home"]), 9)
            # A later pregame snapshot refreshes the pending row's status: the
            # lock keeps whatever the LAST accepted snapshot said.
            xw["snapshot_utc"] = "2026-07-20T22:30:00Z"
            xw["lineup_status_away"] = "posted"
            xw["lineup_posted_away"] = 9
            xw.to_csv(path, index=False)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(out)
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["lineup_status_away"], "posted")
            self.assertEqual(int(out.iloc[0]["lineup_posted_away"]), 9)

    def test_pending_refresh_writes_status_into_legacy_ledger(self):
        # A ledger persisted before the lineup columns existed reads them back
        # as all-NaN float64; a pending refresh must still be able to write
        # string statuses into them (load_ledger forces object dtype).
        with tempfile.TemporaryDirectory() as td:
            legacy_cols = grade_leans.LEDGER_COLS + [
                "snapshot_utc", "scheduled_start_utc", "lock_status"]
            row = {c: None for c in legacy_cols}
            row.update(game_pk=777, game_date="2026-07-20", status="pending",
                       away="AWA", home="HOM", away_sp="Away Pitcher",
                       home_sp="Home Pitcher", model_tag="test_v3",
                       xw_lean="HOM", xw_net=0.02, xw_delta=0.02,
                       ops_valid=False, consensus="NA",
                       snapshot_utc="2026-07-20T12:00:00Z",
                       scheduled_start_utc="2026-07-20T23:00:00Z",
                       lock_status="pregame")
            path = os.path.join(td, "mlb_lean_ledger.csv")
            pd.DataFrame([row]).to_csv(path, index=False)
            xw = _dump_rows(
                777,
                "2026-07-20",
                "2026-07-20T22:30:00Z",
                "2026-07-20T23:00:00Z",
            )
            xw["lineup_status_away"] = "posted"
            xw["lineup_status_home"] = "partial_filled"
            xw["lineup_posted_away"] = 9
            xw["lineup_posted_home"] = 6
            xw.to_csv(os.path.join(td, "leans_2026-07-20_xw.csv"), index=False)
            with mock.patch.object(grade_leans, "DATA_DIR", td), \
                    mock.patch.object(grade_leans, "LEDGER_PATH", path):
                out = grade_leans.ingest(grade_leans.load_ledger())
            self.assertEqual(out.iloc[0]["lineup_status_away"], "posted")
            self.assertEqual(out.iloc[0]["lineup_status_home"], "partial_filled")
            self.assertEqual(int(out.iloc[0]["lineup_posted_home"]), 6)

    def test_legacy_dump_without_lineup_columns_stays_nan(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                888,
                "2026-07-20",
                "2026-07-20T12:00:00Z",
                "2026-07-20T23:00:00Z",
            )
            xw.to_csv(os.path.join(td, "leans_2026-07-20_xw.csv"), index=False)
            ledger = pd.DataFrame(columns=grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            self.assertTrue(pd.isna(out.iloc[0]["lineup_status_away"]))
            self.assertTrue(pd.isna(out.iloc[0]["lineup_posted_home"]))

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

    def test_opener_flag_carries_into_ledger_per_side(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                999,
                "2026-07-20",
                "2026-07-20T12:00:00Z",
                "2026-07-20T23:00:00Z",
            )
            # away starter is an opener, home starter is not.
            xw.loc[xw["side"] == "away", "opener"] = True
            xw.loc[xw["side"] == "home", "opener"] = False
            xw.loc[xw["side"] == "away", "opener_reason"] = "reliever_spot_start"
            xw.loc[xw["side"] == "away", "opener_confidence"] = "medium"
            xw.loc[xw["side"] == "away", "pitching_basis"] = "opener_bullpen_blend"
            xw.loc[xw["side"] == "home", "pitching_basis"] = "starter_bullpen_blend"
            xw.loc[xw["side"] == "away", "starter_xwOBA"] = .310
            xw.loc[xw["side"] == "home", "starter_xwOBA"] = .300
            xw.loc[xw["side"] == "away", "bullpen_xwOBA"] = .325
            xw.loc[xw["side"] == "home", "bullpen_xwOBA"] = .315
            xw.loc[xw["side"] == "away", "expected_sp_ip"] = 1.5
            xw.loc[xw["side"] == "home", "expected_sp_ip"] = 5.8
            xw.loc[:, "bullpen_pitchers"] = 7
            xw.loc[:, "bullpen_relief_bf"] = 900
            xw.to_csv(os.path.join(td, "leans_2026-07-20_xw.csv"), index=False)
            ledger = pd.DataFrame(columns=grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            self.assertEqual(bool(out.iloc[0]["opener_away"]), True)
            self.assertEqual(bool(out.iloc[0]["opener_home"]), False)
            self.assertEqual(out.iloc[0]["opener_reason_away"], "reliever_spot_start")
            self.assertEqual(out.iloc[0]["opener_confidence_away"], "medium")
            self.assertEqual(out.iloc[0]["pitching_basis_away"], "opener_bullpen_blend")
            self.assertEqual(out.iloc[0]["pitching_basis_home"], "starter_bullpen_blend")
            self.assertAlmostEqual(out.iloc[0]["starter_xwoba_away"], .310)
            self.assertAlmostEqual(out.iloc[0]["bullpen_xwoba_home"], .315)
            self.assertAlmostEqual(out.iloc[0]["expected_sp_ip_away"], 1.5)
            self.assertEqual(int(out.iloc[0]["bullpen_pitchers_home"]), 7)
            self.assertEqual(int(out.iloc[0]["bullpen_relief_bf_away"]), 900)

    def test_legacy_dump_without_opener_column_stays_nan(self):
        with tempfile.TemporaryDirectory() as td:
            xw = _dump_rows(
                998,
                "2026-07-20",
                "2026-07-20T12:00:00Z",
                "2026-07-20T23:00:00Z",
            )
            xw.to_csv(os.path.join(td, "leans_2026-07-20_xw.csv"), index=False)
            ledger = pd.DataFrame(columns=grade_leans.LEDGER_COLS + grade_leans.AUDIT_COLS)
            with mock.patch.object(grade_leans, "DATA_DIR", td):
                out = grade_leans.ingest(ledger)
            self.assertTrue(pd.isna(out.iloc[0]["opener_away"]))
            self.assertTrue(pd.isna(out.iloc[0]["opener_home"]))
            self.assertTrue(pd.isna(out.iloc[0]["opener_reason_away"]))
            self.assertTrue(pd.isna(out.iloc[0]["opener_confidence_home"]))
            self.assertTrue(pd.isna(out.iloc[0]["pitching_basis_away"]))


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

    def test_legend_guide_and_record_strip_render_below_cards(self):
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
        # Title bar leads, cards next, how-to-read guide after the cards, and
        # the record strip (when a ledger exists) at the very bottom. Match on
        # element markup, not bare class names, which also appear in the CSS.
        self.assertLess(html.index("<div class='lg-title'>"),
                        html.index("Awaiting paired"))
        self.assertLess(html.index("Awaiting paired"),
                        html.index("How to read a card"))
        if "<div class='gradestrip'>" in html:
            self.assertLess(html.index("How to read a card"),
                            html.index("<div class='gradestrip'>"))


class RecentStarterEraTests(unittest.TestCase):
    def test_last_five_era_excludes_relief_current_day_and_older_starts(self):
        rows = [
            ("2026-07-16", "6.0", 2, 1),
            ("2026-07-10", "5.2", 3, 1),
            ("2026-07-04", "7.0", 1, 1),
            ("2026-06-28", "4.1", 4, 1),
            ("2026-06-22", "6.2", 0, 1),
            ("2026-06-16", "1.0", 9, 1),  # older sixth start
            ("2026-07-15", "2.0", 0, 0),  # relief appearance
            ("2026-07-17", "3.0", 5, 1),  # current slate
        ]
        response = {"stats": [{"splits": [
            {"date": date, "game": {"gamePk": i + 1},
             "stat": {"inningsPitched": ip, "earnedRuns": er, "gamesStarted": gs}}
            for i, (date, ip, er, gs) in enumerate(rows)
        ]}]}
        with mock.patch.object(build_site, "SLATE_DATE", "2026-07-17"), \
                mock.patch.object(build_site, "_get_json", return_value=response), \
                mock.patch.object(build_site.time, "sleep"):
            result = build_site.load_recent_start_era([123])
        # 89 outs over 5 starts -> avg_ip 5.93; ERA excludes relief + current day.
        self.assertEqual(
            {k: result[123][k] for k in ("era", "starts", "avg_ip")},
            {"era": 3.03, "starts": 5, "avg_ip": 5.93},
        )
        self.assertEqual(result[123]["appearances"], 7)
        self.assertEqual(result[123]["recent_starts"], 6)
        self.assertEqual(result[123]["stretched_appearances"], 5)

    def test_avg_ip_flags_opener_from_short_starts(self):
        # Three ~1-inning "starts" (opener pattern) -> avg_ip well below 3.
        rows = [("2026-07-16", "1.0", 1, 1), ("2026-07-10", "1.1", 0, 1),
                ("2026-07-04", "0.2", 2, 1)]
        response = {"stats": [{"splits": [
            {"date": d, "game": {"gamePk": i + 1},
             "stat": {"inningsPitched": ip, "earnedRuns": er, "gamesStarted": gs}}
            for i, (d, ip, er, gs) in enumerate(rows)
        ]}]}
        with mock.patch.object(build_site, "SLATE_DATE", "2026-07-17"), \
                mock.patch.object(build_site, "_get_json", return_value=response), \
                mock.patch.object(build_site.time, "sleep"):
            result = build_site.load_recent_start_era([55])
        self.assertLess(result[55]["avg_ip"], build_site.OPENER_MAX_AVG_IP)
        self.assertEqual(build_site.opener_pids(result), {55})

    def test_kyle_hart_reliever_profile_flags_first_spot_start(self):
        # Regression for 2026-07-23: Hart had one prior official start, so the
        # old two-start rule missed him despite an unmistakable relief workload.
        rows = [
            ("2026-07-20", "1.0", 0, 0, 20),
            ("2026-07-17", "0.1", 0, 0, 14),
            ("2026-07-12", "1.0", 0, 0, 16),
            ("2026-07-08", "2.0", 0, 0, 37),
            ("2026-07-04", "2.0", 0, 0, 31),
            ("2026-07-01", "2.0", 0, 0, 52),
            ("2026-06-29", "0.2", 0, 0, 12),
            ("2026-06-27", "2.0", 1, 1, 36),
            ("2026-06-23", "2.0", 0, 0, 45),
            ("2026-06-21", "1.0", 0, 0, 11),
            # Same-day result must not influence pregame classification.
            ("2026-07-23", "1.0", 0, 1, 18),
        ]
        response = {"stats": [{"splits": [
            {"date": date, "game": {"gamePk": i + 1},
             "stat": {"inningsPitched": ip, "earnedRuns": er,
                      "gamesStarted": gs, "numberOfPitches": pitches}}
            for i, (date, ip, er, gs, pitches) in enumerate(rows)
        ]}]}
        with mock.patch.object(build_site, "SLATE_DATE", "2026-07-23"), \
                mock.patch.object(build_site, "_get_json", return_value=response), \
                mock.patch.object(build_site.time, "sleep"):
            result = build_site.load_recent_start_era([606996])

        profile = result[606996]
        self.assertEqual(profile["starts"], 1)
        self.assertEqual(profile["recent_starts"], 1)
        self.assertEqual(profile["relief_share"], .9)
        self.assertLessEqual(profile["median_ip"], 2.0)
        self.assertLessEqual(profile["p80_pitches"], 55.0)
        self.assertEqual(profile["pitch_count_appearances"], 10)
        self.assertEqual(
            build_site.opener_classifications(result)[606996],
            {"reason": "reliever_spot_start", "confidence": "medium"},
        )

    def test_league_era_uses_earned_runs_and_baseball_innings(self):
        response = {"stats": [{"splits": [
            {"stat": {"inningsPitched": "10.0", "earnedRuns": 4}},
            {"stat": {"inningsPitched": "5.2", "earnedRuns": 2}},
        ]}]}
        with mock.patch.object(build_site, "_get_json", return_value=response):
            self.assertEqual(build_site.load_league_era(), 3.45)

    def test_pitching_splits_include_season_era(self):
        response = {"people": [{
            "id": 123,
            "stats": [{
                "type": {"displayName": "season"},
                "splits": [{"split": {}, "stat": {"era": "3.85", "battersFaced": 400}}],
            }],
        }]}
        with mock.patch.object(build_site, "_get_json", return_value=response), \
                mock.patch.object(build_site.time, "sleep"):
            result = build_site.load_splits([123], "pitching")
        self.assertEqual(result[123]["overall"]["era"], 3.85)

    def test_pitcher_card_shows_xera_vs_season_era(self):
        side = dict(
            t="R", pl_fl={}, R=5, L=4, S=0, has_pl=False, padv=0,
            era_season=3.85, xera=3.03,
            pit_xw=.310, pit_k=27.1, pit_bb=7.5, pit_hh=35.0,
            xw_edge=-.015, p="Test Pitcher", opp_abbr="TST", lu_status="posted",
            opp_xw=None, hitters=[],
        )
        html = build_site._side_html(
            "AWAY", side,
            {"ERA": 4.20, "xwOBA": .320, "K%": 22.0, "Hard Hit%": 39.0},
        )
        self.assertIn("3.03", html)
        self.assertIn("season 3.85", html)
        self.assertNotIn("ERA · L5", html)      # last-5 ERA removed
        self.assertNotIn("OPS alwd", html)      # xOPS-against removed
        # xERA tinted vs league ERA (below league -> cool = pitcher-favorable).
        self.assertIn(
            "<div class='stat' style='background:rgba(var(--cool),0.23)'>"
            "<div class='l'>xERA</div>",
            html,
        )


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

    def test_pregame_window_spans_15_to_90_minutes(self):
        # The window is wide enough that Actions cron jitter (runs delayed
        # 5-20+ min) can't skip a slate entirely: T-60 triggers, while games
        # inside the late cutoff (T-10) or beyond the window (T-95) do not.
        for minutes, expected in ((60, True), (10, False), (95, False)):
            run, _, _ = schedule_gate.decision(
                "schedule", "7,22,37,52 10-23 * * *", self.NOW,
                games=[self.game(101, minutes)],
            )
            self.assertEqual(run, expected, f"T-{minutes}")
        self.assertEqual(schedule_gate.MIN_MINUTES_BEFORE, 15)
        self.assertEqual(schedule_gate.MAX_MINUTES_BEFORE, 90)

    def test_grade_push_and_manual_events_always_run(self):
        grade = schedule_gate.decision("schedule", schedule_gate.DAILY_GRADE_CRON,
                                       self.NOW, games=[])
        push = schedule_gate.decision("push", "", self.NOW, games=[])
        manual = schedule_gate.decision("workflow_dispatch", "", self.NOW, games=[])
        self.assertTrue(grade[0])
        self.assertTrue(push[0])
        self.assertTrue(manual[0])


class LineupStatusDumpTests(unittest.TestCase):
    def test_lineup_status_columns_map_by_game_pk(self):
        lu = pd.DataFrame([dict(
            game_pk=1, away_lineup_status="posted", home_lineup_status="projected",
            away_posted_count=9, home_posted_count=0,
        )])
        frame = pd.DataFrame({"game_pk": [1, 1], "side": ["away", "home"]})
        for col, series in build_site._lineup_status_columns(lu).items():
            frame[col] = frame["game_pk"].map(series)
        self.assertEqual(list(frame["lineup_status_away"]), ["posted", "posted"])
        self.assertEqual(list(frame["lineup_status_home"]), ["projected", "projected"])
        self.assertEqual(list(frame["lineup_posted_away"]), [9, 9])
        self.assertEqual(list(frame["lineup_posted_home"]), [0, 0])

    def test_empty_or_missing_lineup_df_yields_no_columns(self):
        self.assertEqual(build_site._lineup_status_columns(pd.DataFrame()), {})
        self.assertEqual(build_site._lineup_status_columns(None), {})


class OpenerFallbackTests(unittest.TestCase):
    def test_opener_pids_respects_ip_and_starts_thresholds(self):
        era = {
            1: {"avg_ip": 1.2, "starts": 3},   # opener
            2: {"avg_ip": 5.8, "starts": 20},  # workhorse
            3: {"avg_ip": 1.0, "starts": 1},   # one short start only -> not yet
            4: {"avg_ip": np.nan, "starts": 0},  # no starts
        }
        self.assertEqual(build_site.opener_pids(era), {1})

    def test_reliever_role_rejects_recent_starter_length_work(self):
        profile = {
            10: {
                "avg_ip": 2.0, "starts": 1, "appearances": 10,
                "recent_starts": 1, "relief_share": .9, "median_ip": 1.1,
                "p80_pitches": 42.0, "pitch_count_appearances": 10,
                "stretched_appearances": 1,
            },
        }
        self.assertEqual(build_site.opener_classifications(profile), {})

    def test_team_pitching_aggregate_is_bf_weighted(self):
        pitcher_stat = {
            10: {"xwOBA": .300, "K%": 28.0, "BB%": 6.0, "PA": 400, "BBE": 200,
                 "xBA": .230, "xSLG": .380, "EV": 88.0, "LA°": 12.0, "Hard Hit%": 35.0},
            11: {"xwOBA": .360, "K%": 20.0, "BB%": 9.0, "PA": 100, "BBE": 60,
                 "xBA": .270, "xSLG": .440, "EV": 90.0, "LA°": 14.0, "Hard Hit%": 42.0},
            12: {"xwOBA": .320, "K%": 24.0, "BB%": 7.0, "PA": 250, "BBE": 130,
                 "xBA": .245, "xSLG": .400, "EV": 89.0, "LA°": 13.0, "Hard Hit%": 38.0},
        }
        pitcher_bb = {
            10: {"GB%": 45.0, "FB%": 25.0, "LD%": 20.0, "PU%": 10.0,
                 "Pull%": 40.0, "Straight%": 35.0, "Oppo%": 25.0},
            11: {"GB%": 40.0, "FB%": 30.0, "LD%": 20.0, "PU%": 10.0,
                 "Pull%": 42.0, "Straight%": 34.0, "Oppo%": 24.0},
            12: {"GB%": 50.0, "FB%": 22.0, "LD%": 18.0, "PU%": 10.0,
                 "Pull%": 38.0, "Straight%": 36.0, "Oppo%": 26.0},
        }
        with mock.patch.object(build_site, "pitcher_roster", return_value=[10, 11, 12]):
            stat, bb = build_site.team_pitching_aggregate(1, pitcher_stat, pitcher_bb)
        exp_xw = round((.300 * 400 + .360 * 100 + .320 * 250) / 750, 3)
        self.assertEqual(stat["xwOBA"], exp_xw)
        self.assertEqual(stat["PA"], 750.0)          # staff total BF -> minimal shrink
        self.assertEqual(stat["BBE"], 390.0)         # summed, not averaged
        exp_gb = round((45.0 * 200 + 40.0 * 60 + 50.0 * 130) / 390, 3)
        self.assertEqual(bb["GB%"], exp_gb)          # batted-ball rates weighted by BBE

    def test_team_aggregate_none_when_staff_too_thin(self):
        pitcher_stat = {10: {"xwOBA": .300, "PA": 400, "BBE": 200}}
        with mock.patch.object(build_site, "pitcher_roster", return_value=[10]):
            stat, bb = build_site.team_pitching_aggregate(1, pitcher_stat, {})
        self.assertIsNone(stat)
        self.assertIsNone(bb)

    def test_team_pitcher_roles_parse_one_team_response(self):
        response = {"stats": [{"splits": [
            {"player": {"id": 10}, "stat": {
                "gamesPitched": 20, "gamesStarted": 0,
                "inningsPitched": "18.2", "battersFaced": 80,
            }},
            {"player": {"id": 11}, "stat": {
                "gamesPitched": 12, "gamesStarted": 12,
                "inningsPitched": "66.0", "battersFaced": 270,
            }},
        ]}]}
        build_site._team_pitcher_role_cache.clear()
        with mock.patch.object(build_site, "_get_json", return_value=response):
            roles = build_site.load_team_pitcher_roles(1)
        self.assertEqual(roles[10]["start_share"], 0)
        self.assertAlmostEqual(roles[10]["avg_ip_per_appearance"], 56 / 3 / 20)
        self.assertEqual(roles[11]["start_share"], 1)

    def test_relief_filter_keeps_long_relief_but_excludes_rotation(self):
        roles = {
            10: {"appearances": 20, "start_share": 0.0, "avg_ip_per_appearance": 1.0},
            11: {"appearances": 15, "start_share": .20, "avg_ip_per_appearance": 2.8},
            12: {"appearances": 12, "start_share": 1.0, "avg_ip_per_appearance": 5.5},
            13: {"appearances": 10, "start_share": .40, "avg_ip_per_appearance": 2.5},
        }
        with mock.patch.object(build_site, "pitcher_roster", return_value=[10, 11, 12, 13]):
            self.assertEqual(build_site.relief_pitcher_ids(1, roles, probable_pid=10), [11])

    def test_bullpen_aggregate_shrinks_each_reliever_then_usage_weights(self):
        pitcher_stat = {
            10: {"xwOBA": .280, "PA": 200},
            11: {"xwOBA": .340, "PA": 100},
            12: {"xwOBA": .370, "PA": 50},
        }
        roles = {
            10: {"appearances": 30, "start_share": 0.0, "avg_ip_per_appearance": 1.0},
            11: {"appearances": 20, "start_share": .20, "avg_ip_per_appearance": 1.5},
            12: {"appearances": 15, "start_share": 0.0, "avg_ip_per_appearance": 2.5},
        }
        with mock.patch.object(build_site, "pitcher_roster", return_value=[10, 11, 12]):
            out = build_site.bullpen_xwoba_aggregate(
                1, 999, pitcher_stat, roles, prior=.317, shrink_k=100
            )
        x10 = (200 * .280 + 100 * .317) / 300
        x11 = (100 * .340 + 100 * .317) / 200
        x12 = (50 * .370 + 100 * .317) / 150
        expected = (200 * x10 + 80 * x11 + 50 * x12) / 330
        self.assertAlmostEqual(out["xwOBA"], expected)
        self.assertEqual(out["pitcher_count"], 3)
        self.assertEqual(out["relief_bf"], 330)

    def test_expected_ip_uses_role_without_projecting_bulk_follower(self):
        normal = {"avg_ip": 5.8, "starts": 5, "season_avg_ip": 5.5}
        expected = (5 * 5.8 + 3 * 5.5) / 8
        self.assertAlmostEqual(build_site.expected_pitcher_ip(normal), round(expected, 2))
        short = {"avg_ip": 1.4, "starts": 3, "season_avg_ip": 1.4}
        self.assertEqual(
            build_site.expected_pitcher_ip(
                short, {"reason": "repeated_short_starts"}
            ),
            1.4,
        )
        reliever = {"median_ip": 1.2, "avg_ip": 2.0, "starts": 1}
        self.assertEqual(
            build_site.expected_pitcher_ip(
                reliever, {"reason": "reliever_spot_start"}
            ),
            1.2,
        )

    def test_nine_inning_blend_drives_matchup_but_preserves_starter_card_value(self):
        matchup = pd.DataFrame([{
            "game_pk": 1, "side": "away", "pit_xwOBA": .305,
            "opp_xwOBA": .325, "mx_xwOBA": .313, "edge_xwOBA": -.004,
        }])
        plans = {(1, "away"): {
            "expected_sp_ip": 6.0,
            "bullpen_xwOBA": .330,
            "bullpen_pitchers": 7,
            "bullpen_relief_bf": 900,
            "pitching_basis": "starter_bullpen_blend",
            "opener": False,
        }}
        out = build_site.apply_pitching_plans(matchup, plans, {"xwOBA": .317})
        expected_pitching = round((6 * .305 + 3 * .330) / 9, 3)
        expected_matchup = build_site.matchup_value(.325, expected_pitching, "xwOBA", .317)
        self.assertEqual(out.loc[0, "starter_xwOBA"], .305)
        self.assertEqual(out.loc[0, "pit_xwOBA"], expected_pitching)
        self.assertAlmostEqual(out.loc[0, "mx_xwOBA"], round(expected_matchup, 3))
        self.assertEqual(out.loc[0, "expected_sp_ip"], 6.0)
        self.assertEqual(out.loc[0, "bullpen_pitchers"], 7)

    def test_opener_badge_renders_only_when_flagged(self):
        side = dict(
            t="R", pl_fl={}, R=5, L=4, S=0, has_pl=False, padv=0,
            era_l5=0.0, era_l5_gs=1, era_season=3.65, is_opener=True,
            pit_xw=.311, pit_k=24.0, pit_bb=10.0, pit_hh=35.0,
            pl_sp=None, pl_sp_raw=None, pl_edge=None, pl_reliable=False,
            xw_edge=-.009, p="Braydon Fisher", opp_abbr="TB", lu_status="posted",
            opp_xw=None, pl_mx=None, hitters=[],
        )
        lg = {"ERA": 4.20, "xwOBA": .317, "K%": 22.0, "Hard Hit%": 39.0, "OPS": .720}
        side["expected_sp_ip"] = 1.5
        side["pitching_basis"] = "opener_bullpen_blend"
        self.assertIn("opener · bullpen blend", build_site._side_html("HOME", side, lg))
        self.assertIn("model 1.5 IP + bullpen", build_site._side_html("HOME", side, lg))
        side["is_opener"] = False
        self.assertNotIn("opener · bullpen blend", build_site._side_html("HOME", side, lg))


class BattingOrderSlotWeightingTests(unittest.TestCase):
    def test_slot_pa_weights_follow_lineup_position(self):
        w = build_site.slot_pa_weights([1, 5, 9])
        self.assertEqual(list(w), [4.61, 4.18, 3.76])
        # Top of the order carries more in-game exposure than the bottom.
        self.assertGreater(w.iloc[0], w.iloc[2])

    def test_slot_pa_weights_out_of_range_is_nan(self):
        w = build_site.slot_pa_weights([1, 0, 12, None])
        self.assertEqual(w.iloc[0], 4.61)
        self.assertTrue(pd.isna(w.iloc[1]))
        self.assertTrue(pd.isna(w.iloc[2]))
        self.assertTrue(pd.isna(w.iloc[3]))

    @staticmethod
    def _lineup(order_xwoba_bbe):
        rows = []
        for order, xwoba, bbe in order_xwoba_bbe:
            rows.append(dict(
                game_pk=1, faced_pitcher="SP", pitcher_side="away",
                batting_side="home", xwOBA=xwoba, BBE=bbe,
                batting_order=order,
            ))
        return pd.DataFrame(rows)

    def test_aggregate_lineup_weights_by_batting_slot_not_season_volume(self):
        # Slot 1 (4.61 PA) hits .400, slot 9 (3.76 PA) hits .300, but season BBE
        # is lopsided toward the .300 hitter -- slot weighting must ignore that.
        H = self._lineup([(1, .400, 100), (9, .300, 400)])
        with mock.patch.object(build_site, "USE_SLOT_PA_WEIGHTS", True):
            out = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        expected = (4.61 * .400 + 3.76 * .300) / (4.61 + 3.76)
        self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], round(expected, 3))
        # Distinct from both the equal mean (.350) and the BBE mean (.320).
        self.assertNotAlmostEqual(out.loc[0, "opp_xwOBA"], .350)
        self.assertNotAlmostEqual(out.loc[0, "opp_xwOBA"], .320)

    def test_aggregate_lineup_falls_back_to_bbe_when_disabled(self):
        H = self._lineup([(1, .400, 100), (9, .300, 400)])
        with mock.patch.object(build_site, "USE_SLOT_PA_WEIGHTS", False):
            out = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        expected = (100 * .400 + 400 * .300) / 500  # BBE-weighted == .320
        self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], round(expected, 3))

    def test_aggregate_lineup_falls_back_when_order_missing(self):
        H = self._lineup([(1, .400, 100), (9, .300, 400)]).drop(columns=["batting_order"])
        with mock.patch.object(build_site, "USE_SLOT_PA_WEIGHTS", True):
            out = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        expected = (100 * .400 + 400 * .300) / 500  # falls back to BBE == .320
        self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], round(expected, 3))


class XwobaShrinkageTests(unittest.TestCase):
    def test_shrink_pulls_low_sample_toward_prior(self):
        prior, k = 0.317, 100.0
        # 20-PA .450 hitter is mostly prior; 600-PA .450 hitter is mostly self.
        low = float(build_site.shrink_xwoba([0.450], [20], prior, k).iloc[0])
        high = float(build_site.shrink_xwoba([0.450], [600], prior, k).iloc[0])
        self.assertAlmostEqual(low, (20 * .450 + k * prior) / (20 + k), places=6)
        self.assertLess(low, high)                 # small sample regressed harder
        self.assertGreater(low, prior)             # but still above league
        self.assertLess(high, 0.450)               # even big sample shrinks a bit

    def test_shrink_missing_rate_or_zero_n_is_prior(self):
        prior, k = 0.317, 200.0
        out = build_site.shrink_xwoba([np.nan, 0.400], [500, 0], prior, k)
        self.assertAlmostEqual(float(out.iloc[0]), prior, places=6)  # NaN rate -> prior
        self.assertAlmostEqual(float(out.iloc[1]), prior, places=6)  # n=0 -> prior

    def test_shrink_disabled_is_passthrough(self):
        with mock.patch.object(build_site, "USE_XWOBA_SHRINK", False):
            out = build_site.shrink_xwoba([0.450], [20], 0.317, 100.0)
        self.assertAlmostEqual(float(out.iloc[0]), 0.450, places=6)

    def test_scalar_shrink_matches_series(self):
        prior, k = 0.317, 300.0
        s = float(build_site.shrink_xwoba([0.360], [150], prior, k).iloc[0])
        self.assertAlmostEqual(build_site._shrink_one(0.360, 150, prior, k), s, places=6)

    def test_mom_k_recovers_planted_ratio(self):
        # Plant tau^2 and sigma^2, synthesize a pool, recover K = sigma^2/tau^2.
        rng = np.random.default_rng(0)
        prior, tau, sigma = 0.317, 0.030, 0.5
        n = rng.integers(30, 650, size=1200).astype(float)
        theta = rng.normal(prior, tau, size=1200)          # true talent
        x = rng.normal(theta, sigma / np.sqrt(n))          # observed, noisier at low n
        k, note = build_site.estimate_shrink_k(list(zip(x, n)), prior,
                                               build_site.K_BAT_DEFAULT, (10.0, 5000.0))
        planted = sigma ** 2 / tau ** 2                    # ~278
        self.assertEqual(note, "ok")
        self.assertLess(abs(k - planted) / planted, 0.35)  # within ~35% of truth

    def test_mom_k_falls_back_on_thin_pool(self):
        k, note = build_site.estimate_shrink_k([(0.4, 100), (0.3, 200)], 0.317,
                                               build_site.K_PIT_DEFAULT, build_site.K_PIT_BAND)
        self.assertEqual(k, build_site.K_PIT_DEFAULT)
        self.assertTrue(note.startswith("fallback"))

    def _lineup_H(self, rows):
        # rows: (batting_order, xwOBA, PA); one game vs one pitcher.
        return pd.DataFrame([
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=xw, PA=pa, BBE=50, batting_order=o)
            for (o, xw, pa) in rows
        ])

    def test_aggregate_lineup_shrinks_bats_before_compositing(self):
        prior, k = 0.317, 150.0
        H = self._lineup_H([(1, 0.500, 15), (2, 0.300, 550)])
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True,
                                          shrink_prior=prior, shrink_k=k)
        s1 = (15 * .500 + k * prior) / (15 + k)
        s2 = (550 * .300 + k * prior) / (550 + k)
        w1, w2 = build_site.LINEUP_SLOT_PA[1], build_site.LINEUP_SLOT_PA[2]
        expected = round((w1 * s1 + w2 * s2) / (w1 + w2), 3)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], expected)
        # Without shrinkage the hot 15-PA bat would drag the composite higher.
        raw = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        self.assertLess(agg.loc[0, "opp_xwOBA"], raw.loc[0, "opp_xwOBA"])


class MarketContextRecordsTests(unittest.TestCase):
    COLS = ["status", "xw_lean", "xw_full", "home", "away", "close_p_home"]

    def _led(self, rows):
        return pd.DataFrame(rows, columns=self.COLS)

    def test_buckets_by_lean_side_and_market_agreement(self):
        rows = [
            # away lean, away is the market favorite (ph<.5) -> agree: W W L
            ("graded", "NYY", "W", "TB", "NYY", 0.40),
            ("graded", "NYY", "W", "TB", "NYY", 0.45),
            ("graded", "NYY", "L", "TB", "NYY", 0.48),
            # away lean, home favored (ph>=.5) -> disagree (away underdog): L
            ("graded", "NYY", "L", "TB", "NYY", 0.60),
            # home lean, home favored -> agree: W
            ("graded", "TB", "W", "TB", "NYY", 0.55),
            # ignored: not graded, and graded-without-market
            ("pending", "TB", None, "TB", "NYY", 0.55),
            ("graded", "TB", "W", "TB", "NYY", float("nan")),
        ]
        with mock.patch.object(build_site, "load_ledger_df", return_value=self._led(rows)), \
                mock.patch.object(build_site, "VERDICT_CONTEXT_MIN", 1):
            ctx = build_site.market_context_records()
        self.assertEqual(ctx[("away", "agree")], "2-1")
        self.assertEqual(ctx[("away", "disagree")], "0-1")
        self.assertEqual(ctx[("home", "agree")], "1-0")
        self.assertNotIn(("home", "disagree"), ctx)

    def test_thin_bucket_is_omitted(self):
        rows = [("graded", "NYY", "W", "TB", "NYY", 0.40)]  # 1 game < default min
        with mock.patch.object(build_site, "load_ledger_df", return_value=self._led(rows)):
            ctx = build_site.market_context_records()
        self.assertEqual(ctx, {})


if __name__ == "__main__":
    unittest.main()
