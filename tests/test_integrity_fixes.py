import math
import inspect
import os
import re
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

import build_site
import compare_v8_v9
import grade_leans
import market_backfill
import schedule_gate

# Read, never copied. This column is named for the active metric, so a literal
# here would silently stop matching the moment the metric changes -- and the
# failure mode is not a red test but a green one measuring the wrong thing: an
# unrecognised backfill flag just means the bat gets shrunk like any other.
BACKFILL_COL = build_site.MODEL_RATE_TEAM_BACKFILL_COL


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
            dict(game_pk=8, status="graded", model_tag="xw+plat_consol_v7"),
            dict(game_pk=9, status="graded", model_tag="xw+plat_consol_v8"),
            dict(game_pk=10, status="graded", model_tag="xw+plat_consol_v9"),
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
        # v5 through v9 each changed prediction math and remain isolated.
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v5",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {6})
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v6",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {7})
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v7",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {8})
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v8",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {9})
        with mock.patch.object(build_site, "RECORD_TAGS", ("xw+plat_consol_v9",)):
            self.assertEqual(set(build_site._record_grades(ledger)["game_pk"]), {10})
        self.assertEqual(
            [label for label, _ in grade_leans._model_family_grades(ledger)],
            ["v2/v3", "v4", "v5", "v6", "v7", "v8", "v9/v10",
             "xw+plat_consol_v1"],
        )

    def test_v9_and_v10_share_one_record_family(self):
        # v10 re-weights the phase blend from innings share to PA share. A
        # BF/IP-ratio sweep over 0.95-1.10 flips 0 of 12 leans, so the two
        # models agree on the decision and pool into one win-loss line.
        fam = ("xw+plat_consol_v9", "xw+plat_consol_v10")
        for tag in fam:
            self.assertEqual(build_site._RECORD_FAMILIES[tag], fam)
            self.assertEqual(grade_leans._RECORD_FAMILIES[tag], fam)

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

    def test_missing_bullpen_snapshot_is_audited_without_a_fullgame_lean(self):
        xw = _dump_rows(
            321,
            "2026-07-27",
            "2026-07-27T15:00:00+00:00",
            "2026-07-27T22:00:00+00:00",
        )
        xw.loc[xw["side"] == "away", "edge_xwOBA"] = np.nan
        xw["pitching_basis"] = "starter_only_no_fullgame_lean"
        rows = grade_leans.rows_from_dump(xw, None)
        self.assertEqual(len(rows), 1)
        self.assertTrue(pd.isna(rows[0]["xw_net"]))
        self.assertIsNone(rows[0]["xw_lean"])
        self.assertTrue(pd.isna(rows[0]["xw_delta"]))
        self.assertEqual(
            rows[0]["pitching_basis_away"],
            "starter_only_no_fullgame_lean",
        )

    def test_grades_page_omits_family_history_and_model_column(self):
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
        with mock.patch.object(build_site, "load_ledger_df", return_value=ledger):
            page = build_site.render_grades_html("test build")
        self.assertNotIn("Model-family history", page)
        self.assertNotIn("<th>Family</th>", page)
        self.assertNotIn("<th>Model</th>", page)
        self.assertNotIn("<td>v5</td>", page)
        self.assertNotIn("<td>v6</td>", page)
        self.assertNotIn("mlb.com/standings", page)

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

    def test_exact_zero_deltas_produce_no_lean(self):
        xw = _dump_rows(
            123, "2026-07-24", "2026-07-24T12:00:00Z",
            "2026-07-24T19:00:00Z",
        )
        xw["edge_xwOBA"] = .0123456789
        pl = pd.DataFrame([
            dict(game_pk=123, side="away", edge_OPS=.023456789,
                 reliable=True),
            dict(game_pk=123, side="home", edge_OPS=.023456789,
                 reliable=True),
        ])
        row = grade_leans.rows_from_dump(xw, pl)[0]
        self.assertIsNone(row["xw_lean"])
        self.assertIsNone(row["ops_lean"])
        self.assertEqual(row["xw_delta"], 0)
        self.assertEqual(row["ops_delta"], 0)
        self.assertEqual(row["consensus"], "NA")

    def test_sub_display_precision_delta_remains_a_decision(self):
        xw = _dump_rows(
            124, "2026-07-24", "2026-07-24T12:00:00Z",
            "2026-07-24T19:00:00Z",
        )
        xw.loc[xw["side"] == "away", "edge_xwOBA"] = .01234567891
        xw.loc[xw["side"] == "home", "edge_xwOBA"] = .01234567890
        row = grade_leans.rows_from_dump(xw, None)[0]
        self.assertEqual(row["xw_lean"], "HOM")
        self.assertGreater(row["xw_net"], 0)
        self.assertNotEqual(row["xw_delta"], 0)

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
            xw.loc[xw["side"] == "away", "pitching_basis"] = "opener_bullpen_sequential"
            xw.loc[xw["side"] == "home", "pitching_basis"] = "starter_bullpen_sequential"
            xw.loc[xw["side"] == "away", "starter_xwOBA"] = .310
            xw.loc[xw["side"] == "home", "starter_xwOBA"] = .300
            xw.loc[xw["side"] == "away", "bullpen_xwOBA"] = .325
            xw.loc[xw["side"] == "home", "bullpen_xwOBA"] = .315
            xw.loc[xw["side"] == "away", "expected_sp_ip"] = 1.5
            xw.loc[xw["side"] == "home", "expected_sp_ip"] = 5.8
            xw.loc[xw["side"] == "away", "opp_xwOBA_neutral"] = .320
            xw.loc[xw["side"] == "home", "opp_xwOBA_neutral"] = .315
            xw.loc[xw["side"] == "away", "opp_xwOBA_vs_sp"] = .330
            xw.loc[xw["side"] == "home", "opp_xwOBA_vs_sp"] = .310
            xw.loc[:, "platoon_delta_sp"] = [.010, -.005]
            xw.loc[:, "sp_share"] = [1.5 / 9, 5.8 / 9]
            xw.loc[:, "bp_share"] = [7.5 / 9, 3.2 / 9]
            xw.loc[:, "mx_xwOBA_sp"] = [.323, .301]
            xw.loc[:, "mx_xwOBA_bp"] = [.328, .310]
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
            self.assertEqual(out.iloc[0]["pitching_basis_away"], "opener_bullpen_sequential")
            self.assertEqual(out.iloc[0]["pitching_basis_home"], "starter_bullpen_sequential")
            self.assertAlmostEqual(out.iloc[0]["starter_xwoba_away"], .310)
            self.assertAlmostEqual(out.iloc[0]["bullpen_xwoba_home"], .315)
            self.assertAlmostEqual(out.iloc[0]["expected_sp_ip_away"], 1.5)
            self.assertEqual(int(out.iloc[0]["bullpen_pitchers_home"]), 7)
            self.assertEqual(int(out.iloc[0]["bullpen_relief_bf_away"]), 900)
            self.assertAlmostEqual(out.iloc[0]["opp_xwoba_neutral_away"], .320)
            self.assertAlmostEqual(out.iloc[0]["opp_xwoba_vs_sp_home"], .310)
            self.assertAlmostEqual(out.iloc[0]["sp_share_away"], 1.5 / 9)
            self.assertAlmostEqual(out.iloc[0]["mx_xwoba_bp_home"], .310)

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

    def test_record_strip_and_build_stamp_render_below_cards(self):
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
        # Cards lead, the record strip (when a ledger exists) sits under them,
        # and the model/build stamp is last. Match on element markup, not bare
        # class names, which also appear in the CSS.
        stamp = html.index("<div class='lg-title'>")
        self.assertLess(html.index("Awaiting paired"), stamp)
        if "<div class='gradestrip'>" in html:
            self.assertLess(html.index("Awaiting paired"),
                            html.index("<div class='gradestrip'>"))
            self.assertLess(html.index("<div class='gradestrip'>"), stamp)


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
        # 89 outs over 5 starts; ERA excludes relief + current day.
        self.assertEqual(result[123]["era"], 3.03)
        self.assertEqual(result[123]["starts"], 5)
        self.assertAlmostEqual(result[123]["avg_ip"], 89 / 3 / 5)
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
            away_savant_backfill_count=1, home_savant_backfill_count=0,
        )])
        frame = pd.DataFrame({"game_pk": [1, 1], "side": ["away", "home"]})
        for col, series in build_site._lineup_status_columns(lu).items():
            frame[col] = frame["game_pk"].map(series)
        self.assertEqual(list(frame["lineup_status_away"]), ["posted", "posted"])
        self.assertEqual(list(frame["lineup_status_home"]), ["projected", "projected"])
        self.assertEqual(list(frame["lineup_posted_away"]), [9, 9])
        self.assertEqual(list(frame["lineup_posted_home"]), [0, 0])
        self.assertEqual(list(frame["lineup_savant_backfill_away"]), [1, 1])
        self.assertEqual(list(frame["lineup_savant_backfill_home"]), [0, 0])

    def test_legacy_lineup_audit_without_backfill_counts_is_supported(self):
        lu = pd.DataFrame([dict(
            game_pk=1, away_lineup_status="posted", home_lineup_status="posted",
            away_posted_count=9, home_posted_count=9,
        )])
        cols = build_site._lineup_status_columns(lu)
        self.assertTrue(cols["lineup_savant_backfill_away"].isna().all())
        self.assertTrue(cols["lineup_savant_backfill_home"].isna().all())

    def test_empty_or_missing_lineup_df_yields_no_columns(self):
        self.assertEqual(build_site._lineup_status_columns(pd.DataFrame()), {})
        self.assertEqual(build_site._lineup_status_columns(None), {})


class PostedLineupBackfillTests(unittest.TestCase):
    def test_posted_savant_missing_hitter_is_kept_with_team_average(self):
        batter_stat = {
            pid: {"xwOBA": .280 + pid / 1000, "PA": 100.0 * pid, "BBE": 50}
            for pid in range(1, 10)
        }
        roster = list(range(1, 10))
        expected = np.average(
            [batter_stat[pid]["xwOBA"] for pid in roster],
            weights=[batter_stat[pid]["PA"] for pid in roster],
        )
        with mock.patch.object(build_site, "gf_lineups",
                               return_value=([999, 1], [])), \
                mock.patch.object(build_site, "roster_lineup",
                                  return_value=roster):
            resolved, meta = build_site.resolve_lineup(
                10, "away", 20, batter_stat,
                league_xwoba=.317, return_meta=True,
            )
        self.assertEqual(resolved[:2], [999, 1])
        self.assertEqual(len(resolved), 9)
        self.assertEqual(meta["posted_count"], 2)
        self.assertEqual(meta["filled_count"], 7)
        self.assertEqual(meta["savant_backfill_count"], 1)
        self.assertAlmostEqual(batter_stat[999]["xwOBA"], expected)
        self.assertEqual(batter_stat[999]["PA"], 0)
        self.assertTrue(
            batter_stat[999][build_site.MODEL_RATE_TEAM_BACKFILL_COL]
        )

    def test_team_backfill_bypasses_player_level_shrinkage(self):
        H = pd.DataFrame([
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=.340, PA=0, BBE=0,
                 batting_order=1, **{BACKFILL_COL: True}),
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=.300, PA=600, BBE=200,
                 batting_order=2, **{BACKFILL_COL: False}),
        ])
        prior, k = .317, 175.0
        out = build_site.aggregate_lineup(
            H, ["xwOBA"], weighted=True, shrink_prior=prior, shrink_k=k
        )
        regular = (600 * .300 + k * prior) / (600 + k)
        w1, w2 = build_site.LINEUP_SLOT_PA[1], build_site.LINEUP_SLOT_PA[2]
        expected = (w1 * .340 + w2 * regular) / (w1 + w2)
        self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], expected)


class PitchingPlanTests(unittest.TestCase):
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
                1, 999, pitcher_stat, roles, prior=.317,
                shrink_k=build_site.XWOBA_SHRINK_K,
            )
        k = build_site.XWOBA_SHRINK_K
        x10 = (200 * .280 + k * .317) / (200 + k)
        x11 = (100 * .340 + k * .317) / (100 + k)
        x12 = (50 * .370 + k * .317) / (50 + k)
        expected = (200 * x10 + 80 * x11 + 50 * x12) / 330
        self.assertAlmostEqual(out["xwOBA"], expected)
        self.assertEqual(out["pitcher_count"], 3)
        self.assertEqual(out["relief_bf"], 330)

    def test_expected_ip_uses_role_without_projecting_bulk_follower(self):
        normal = {"avg_ip": 5.8, "starts": 5, "season_avg_ip": 5.5}
        expected = (5 * 5.8 + 3 * 5.5) / 8
        self.assertAlmostEqual(build_site.expected_pitcher_ip(normal), expected)
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

    def test_sequential_phases_drive_matchup_but_preserve_starter_card_value(self):
        matchup = pd.DataFrame([{
            "game_pk": 1, "side": "away", "pit_xwOBA": .305,
            "opp_xwOBA": .325, "opp_xwOBA_vs_sp": .325,
            "opp_xwOBA_neutral": .315, "platoon_delta_sp": .010,
            "mx_xwOBA": .313, "edge_xwOBA": -.004,
        }])
        plans = {(1, "away"): {
            "expected_sp_ip": 6.0,
            "bullpen_xwOBA": .330,
            "bullpen_pitchers": 7,
            "bullpen_relief_bf": 900,
            "pitching_basis": "starter_bullpen_sequential",
            "opener": False,
        }}
        out = build_site.apply_pitching_plans(matchup, plans, {"xwOBA": .317})
        expected_pitching = (6 * .305 + 3 * .330) / 9
        expected_sp = build_site.matchup_value(.325, .305, "xwOBA", .317)
        expected_bp = build_site.matchup_value(.315, .330, "xwOBA", .317)
        expected_matchup = (6 * expected_sp + 3 * expected_bp) / 9
        self.assertEqual(out.loc[0, "starter_xwOBA"], .305)
        self.assertAlmostEqual(out.loc[0, "pit_xwOBA"], expected_pitching)
        self.assertAlmostEqual(out.loc[0, "mx_xwOBA_sp"], expected_sp)
        self.assertAlmostEqual(out.loc[0, "mx_xwOBA_bp"], expected_bp)
        self.assertAlmostEqual(out.loc[0, "mx_xwOBA"], expected_matchup)
        self.assertAlmostEqual(out.loc[0, "sp_share"], 2 / 3)
        self.assertAlmostEqual(out.loc[0, "bp_share"], 1 / 3)
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
        side["pitching_basis"] = "opener_bullpen_sequential"
        self.assertIn("opener · bullpen", build_site._side_html("HOME", side, lg))
        self.assertIn("model 1.5 IP + bullpen", build_site._side_html("HOME", side, lg))
        side["is_opener"] = False
        self.assertNotIn("opener · bullpen", build_site._side_html("HOME", side, lg))


class SequentialPitchingModelTests(unittest.TestCase):
    L = .317

    def phases(self, expected_ip, b0=.320, bsp=.330, sp=.305, bp=.325):
        return build_site.sequential_xwoba_phases(
            b0, bsp, sp, bp, self.L, expected_ip
        )

    def test_q_endpoints_equal_the_corresponding_phase(self):
        all_sp = self.phases(9.0)
        all_bp = self.phases(0.0)
        self.assertAlmostEqual(all_sp["mx_xwOBA"], all_sp["mx_xwOBA_sp"])
        self.assertAlmostEqual(all_bp["mx_xwOBA"], all_bp["mx_xwOBA_bp"])
        self.assertEqual(all_sp["sp_share"], 1.0)
        self.assertEqual(all_bp["sp_share"], 0.0)

    def test_pa_share_weight_uses_measured_bf_per_ip(self):
        # xwOBA is a per-PA rate, so the blend weight is the PA share.
        ip, r_sp, r_bp = 5.4, 4.60, 4.20
        p = build_site.sequential_xwoba_phases(
            .320, .330, .305, .325, self.L, ip,
            sp_bf_per_ip=r_sp, bp_bf_per_ip=r_bp,
        )
        pa_sp, pa_bp = ip * r_sp, (9 - ip) * r_bp
        self.assertAlmostEqual(p["sp_share"], pa_sp / (pa_sp + pa_bp))
        self.assertAlmostEqual(p["sp_share"] + p["bp_share"], 1.0)
        # A starter allowing more baserunners faces more batters per inning, so
        # the PA share must exceed the innings share he was credited in v9.
        self.assertGreater(p["sp_share"], ip / 9)

    def test_equal_rates_reproduce_the_innings_share(self):
        # The correction is centred, not a bias: identical BF/IP on both sides
        # is exactly v9's weight, so the change degrades continuously.
        for rate in (3.9, 4.3, 5.0):
            p = build_site.sequential_xwoba_phases(
                .320, .330, .305, .325, self.L, 5.4,
                sp_bf_per_ip=rate, bp_bf_per_ip=rate,
            )
            self.assertAlmostEqual(p["sp_share"], 5.4 / 9)

    def test_missing_rate_falls_back_to_innings_share(self):
        for kw in ({}, {"sp_bf_per_ip": 4.4}, {"bp_bf_per_ip": 4.4},
                   {"sp_bf_per_ip": 0, "bp_bf_per_ip": 4.4},
                   {"sp_bf_per_ip": None, "bp_bf_per_ip": None}):
            p = build_site.sequential_xwoba_phases(
                .320, .330, .305, .325, self.L, 5.4, **kw)
            self.assertAlmostEqual(p["sp_share"], 5.4 / 9)

    def test_pa_share_preserves_endpoints(self):
        # Nine starter innings leaves the bullpen zero PAs whatever the rates.
        hi = build_site.sequential_xwoba_phases(
            .320, .330, .305, .325, self.L, 9.0,
            sp_bf_per_ip=4.6, bp_bf_per_ip=4.0)
        lo = build_site.sequential_xwoba_phases(
            .320, .330, .305, .325, self.L, 0.0,
            sp_bf_per_ip=4.6, bp_bf_per_ip=4.0)
        self.assertEqual(hi["sp_share"], 1.0)
        self.assertEqual(lo["sp_share"], 0.0)

    def test_bf_per_ip_from_role_line(self):
        self.assertAlmostEqual(
            build_site.bf_per_ip(
                {"batters_faced": 430, "appearances": 20,
                 "avg_ip_per_appearance": 5.0}),
            430 / 100.0)
        for bad in ({}, None,
                    {"batters_faced": 0, "appearances": 20,
                     "avg_ip_per_appearance": 5.0},
                    {"batters_faced": 430, "appearances": 0,
                     "avg_ip_per_appearance": 5.0},
                    {"batters_faced": 430, "appearances": 20}):
            self.assertIsNone(build_site.bf_per_ip(bad))

    def test_no_platoon_delta_matches_v8_blended_pitching_formula(self):
        b, sp, bp, expected_ip = .320, .305, .325, 5.4
        phases = self.phases(expected_ip, b0=b, bsp=b, sp=sp, bp=bp)
        q = expected_ip / 9
        v8_pitching = q * sp + (1 - q) * bp
        v8 = build_site.matchup_value(b, v8_pitching, "xwOBA", self.L)
        self.assertAlmostEqual(phases["mx_xwOBA"], v8)

    def test_missing_bullpen_suppresses_fullgame_but_keeps_starter_phase(self):
        matchup = pd.DataFrame([{
            "game_pk": 1,
            "side": "away",
            "pit_xwOBA": .305,
            "starter_xwOBA": .305,
            "opp_xwOBA": .330,
            "opp_xwOBA_vs_sp": .330,
            "opp_xwOBA_neutral": .320,
        }])
        plans = {(1, "away"): {
            "expected_sp_ip": 1.5,
            "bullpen_xwOBA": None,
            "opener": True,
        }}
        out = build_site.apply_pitching_plans(matchup, plans, {"xwOBA": self.L})
        self.assertTrue(pd.notna(out.loc[0, "mx_xwOBA_sp"]))
        self.assertTrue(pd.isna(out.loc[0, "mx_xwOBA_bp"]))
        self.assertTrue(pd.isna(out.loc[0, "mx_xwOBA"]))
        self.assertTrue(pd.isna(out.loc[0, "edge_xwOBA"]))
        self.assertTrue(pd.isna(out.loc[0, "pit_xwOBA"]))
        self.assertEqual(
            out.loc[0, "pitching_basis"],
            "starter_only_no_fullgame_lean",
        )

    def test_shorter_workload_reduces_starter_platoon_influence(self):
        neutral_short = self.phases(1.5, bsp=.320)
        adjusted_short = self.phases(1.5, bsp=.330)
        neutral_long = self.phases(6.0, bsp=.320)
        adjusted_long = self.phases(6.0, bsp=.330)
        short_effect = adjusted_short["mx_xwOBA"] - neutral_short["mx_xwOBA"]
        long_effect = adjusted_long["mx_xwOBA"] - neutral_long["mx_xwOBA"]
        self.assertGreater(long_effect, short_effect)
        self.assertAlmostEqual(long_effect / short_effect, 4.0)

    def test_opener_hand_changes_only_projected_starter_innings(self):
        versus_left = self.phases(1.5, bsp=.330)
        versus_right = self.phases(1.5, bsp=.310)
        self.assertAlmostEqual(
            versus_left["mx_xwOBA_bp"],
            versus_right["mx_xwOBA_bp"],
        )
        expected_delta = (1.5 / 9) * (
            versus_left["mx_xwOBA_sp"] - versus_right["mx_xwOBA_sp"]
        )
        self.assertAlmostEqual(
            versus_left["mx_xwOBA"] - versus_right["mx_xwOBA"],
            expected_delta,
        )

    def test_opener_missing_bullpen_never_calls_whole_staff_fallback(self):
        slate = pd.DataFrame([{
            "game_pk": 1,
            "away_probable_pitcher_id": 10,
            "away_team_id": 100,
            "home_probable_pitcher_id": np.nan,
            "home_team_id": np.nan,
        }])
        profiles = {10: {"avg_ip": 1.0, "starts": 3}}
        with mock.patch.object(build_site, "load_team_pitcher_roles", return_value={}), \
                mock.patch.object(build_site, "bullpen_xwoba_aggregate", return_value=None):
            plans = build_site.build_pitching_plans(
                slate, profiles, {}, {"xwOBA": self.L}
            )
        self.assertEqual(
            plans[(1, "away")]["pitching_basis"],
            "starter_only_no_fullgame_lean",
        )


class V8V9ComparisonTests(unittest.TestCase):
    def test_replay_uses_identical_phase_inputs_for_both_formulas(self):
        snapshot = pd.DataFrame([
            {
                "game_pk": 1, "game_date": "2026-07-27", "side": "away",
                "opp_team": "HOME", "opp_xwOBA_neutral": .320,
                "opp_xwOBA_vs_sp": .330, "starter_xwOBA": .305,
                "bullpen_xwOBA": .325, "expected_sp_ip": 1.8,
                "lg_xwOBA": .317, "opener": True,
            },
            {
                "game_pk": 1, "game_date": "2026-07-27", "side": "home",
                "opp_team": "AWAY", "opp_xwOBA_neutral": .315,
                "opp_xwOBA_vs_sp": .315, "starter_xwOBA": .310,
                "bullpen_xwOBA": .320, "expected_sp_ip": 5.4,
                "lg_xwOBA": .317, "opener": False,
            },
        ])
        sides = compare_v8_v9.recalculate_sides(snapshot)
        q = 1.8 / 9
        v8_pitching = q * .305 + (1 - q) * .325
        expected_v8 = .330 * v8_pitching / .317
        expected_v9 = q * (.330 * .305 / .317) + (1 - q) * (.320 * .325 / .317)
        self.assertAlmostEqual(sides.loc[0, "mx_xwOBA_v8_shadow"], expected_v8)
        self.assertAlmostEqual(sides.loc[0, "mx_xwOBA_v9_recalc"], expected_v9)
        games = compare_v8_v9.pair_games(sides)
        self.assertEqual(len(games), 1)
        self.assertTrue(bool(games.loc[0, "eligible"]))
        self.assertTrue(bool(games.loc[0, "opener"]))

    def test_legacy_snapshot_without_neutral_composite_is_ineligible(self):
        legacy = pd.DataFrame([{
            "opp_xwOBA": .320,
            "pit_xwOBA": .310,
            "lg_xwOBA": .317,
        }])
        out = compare_v8_v9.recalculate_sides(legacy)
        self.assertFalse(bool(out.loc[0, "comparison_eligible"]))
        self.assertTrue(pd.isna(out.loc[0, "mx_xwOBA_v9_recalc"]))


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
        self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], expected)
        # Distinct from both the equal mean (.350) and the BBE mean (.320).
        self.assertNotAlmostEqual(out.loc[0, "opp_xwOBA"], .350)
        self.assertNotAlmostEqual(out.loc[0, "opp_xwOBA"], .320)

    def test_no_order_is_an_equal_mean_not_a_bbe_mean(self):
        """The BBE fallback is gone in both ways it could be reached.

        These two asserted the removed behaviour -- a per-PA rate weighted by a
        per-BBE denominator, which underweights three-true-outcome bats. BBE is
        now ignored by the lineup composite whether the order is missing or the
        slot weighting is switched off; `wmean` takes an equal mean, and the
        .320 BBE answer is specifically what must not come back.
        """
        H = self._lineup([(1, .400, 100), (9, .300, 400)])
        no_order = H.drop(columns=["batting_order"])
        with mock.patch.object(build_site, "USE_SLOT_PA_WEIGHTS", False):
            disabled = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        with mock.patch.object(build_site, "USE_SLOT_PA_WEIGHTS", True):
            missing = build_site.aggregate_lineup(no_order, ["xwOBA"],
                                                  weighted=True)
        for out in (disabled, missing):
            self.assertAlmostEqual(out.loc[0, "opp_xwOBA"], .350)
            self.assertNotAlmostEqual(out.loc[0, "opp_xwOBA"], .320)

    def test_lineup_weight_takes_no_fallback_column(self):
        """Signature is the guard: nothing can pass a season-volume column in
        again without this failing."""
        self.assertEqual(
            list(inspect.signature(build_site.lineup_weight).parameters), ["g"])
        self.assertFalse(hasattr(build_site, "WEIGHT_COL"))

    def test_every_lineup_group_from_the_real_frame_builder_carries_its_order(self):
        """Why deleting the fallback moves nothing: it was unreachable.

        Drives the actual path -- build_tables -> segment_pitcher_blocks -> the
        groupby aggregate_lineup uses -- and asserts every lineup group yields
        usable slot weights, including a 10-man card and a hitter missing from
        the leaderboard. If that ever stops holding, the composite silently
        becomes an equal mean, so this is the assertion that has to fail first.
        """
        people, batter_stat, lineups = {}, {}, {}
        pid = 100
        for gpk in (1, 2):
            sides = []
            for n in ((9, 10) if gpk == 1 else (9, 9)):
                lu = []
                for slot in range(n):
                    pid += 1
                    people[pid] = {"name": f"B{pid}", "bats": "R", "pos": "SS",
                                   "throws": "R"}
                    if not (gpk == 2 and slot == 4):      # one absent from Savant
                        batter_stat[pid] = {"PA": 300.0, "BBE": 200.0,
                                            "xwOBA": 0.320}
                    lu.append(pid)
                sides.append(lu)
            lineups[gpk] = (sides[0], sides[1])
        sp = {}
        for gpk in (1, 2):
            for who in ("away", "home"):
                pid += 1
                people[pid] = {"name": f"SP{pid}", "bats": "R", "pos": "P",
                               "throws": "R"}
                batter_stat.setdefault(pid, {"PA": 400.0, "BBE": 150.0,
                                             "xwOBA": 0.300})
                sp[(gpk, who)] = pid
        slate = pd.DataFrame([{
            "game_pk": gpk, "game_date": "2026-08-06",
            "game_datetime_utc": "2026-08-06T23:05:00Z",
            "matchup": f"A{gpk} @ H{gpk}", "away_team": f"A{gpk}",
            "home_team": f"H{gpk}",
            "away_probable_pitcher": f"SP{sp[(gpk, 'away')]}",
            "home_probable_pitcher": f"SP{sp[(gpk, 'home')]}",
            "away_probable_pitcher_id": sp[(gpk, "away")],
            "home_probable_pitcher_id": sp[(gpk, "home")],
            "savant_preview_url": "",
        } for gpk in (1, 2)])

        pdf, _ = build_site.build_tables(slate, lineups, batter_stat,
                                         batter_stat, {}, {}, people)
        _, H = build_site.segment_pitcher_blocks(pdf,
                                                 build_site.STATCAST_RATE_COLS)
        self.assertIn(build_site.BATTING_ORDER_COL, H.columns)
        groups = list(H.groupby(["game_pk", "faced_pitcher"], sort=False))
        self.assertEqual(len(groups), 4)
        for _, g in groups:
            w = build_site.lineup_weight(g)
            self.assertIsNotNone(w)
            self.assertTrue(pd.Series(w).reset_index(drop=True).equals(
                build_site.slot_pa_weights(g[build_site.BATTING_ORDER_COL])))


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

    def test_model_uses_one_fixed_k_for_batters_and_pitchers(self):
        """One K reaches both pools -- the claim, not the numeral.

        This asserted the literal 100.0 and so failed on the v3 bump for a
        reason with nothing to do with what it guards, which is that batters
        and pitchers are handed the SAME constant. The value is read off the
        module; a role split would still fail here, which is the point.
        """
        k = build_site.XWOBA_SHRINK_K
        self.assertGreater(k, 0)
        with (
            mock.patch.object(
                build_site, "segment_pitcher_blocks",
                return_value=("pitcher_rows", "opposing_hitters"),
            ),
            mock.patch.object(
                build_site, "aggregate_lineup", return_value="lineup_aggregate"
            ) as aggregate,
            mock.patch.object(
                build_site, "build_matchup", return_value=pd.DataFrame()
            ) as matchup,
        ):
            build_site.build_xwoba_matchup(pd.DataFrame(), {"xwOBA": .317})

        self.assertEqual(
            aggregate.call_args.kwargs["shrink_k"],
            build_site.XWOBA_SHRINK_K,
        )
        self.assertEqual(
            matchup.call_args.kwargs["shrink_k"],
            build_site.XWOBA_SHRINK_K,
        )

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
        expected = (w1 * s1 + w2 * s2) / (w1 + w2)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], expected)
        # Without shrinkage the hot 15-PA bat would drag the composite higher.
        raw = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        self.assertLess(agg.loc[0, "opp_xwOBA"], raw.loc[0, "opp_xwOBA"])


class PriorPopulationCentreTests(unittest.TestCase):
    """The shrinkage-target diagnostic. Log-only, so these assert the arithmetic
    and the pool predicates -- never a centre read off a real leaderboard, which
    would be the frozen-constant anti-pattern with a diagnostic's name on it."""

    SRC = build_site.MODEL_RATE_SOURCE_COL

    def _cust(self, rows):
        # rows: (player_id, rate, pa)
        return pd.DataFrame([{"player_id": p, self.SRC: r, "pa": n}
                             for (p, r, n) in rows])

    def test_recovers_planted_weighted_and_unweighted_centres(self):
        prior, k = 0.300, 100.0
        # A high-PA .340 regular and a low-PA .260 bench bat: PA weighting puts
        # the centre near the regular, the player-level centre sits at .300.
        cust = self._cust([(1, 0.340, 600), (2, 0.260, 40)])
        rows = dict(build_site.prior_population_centres(cust, None, prior, k))
        m = rows["batters"]
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["weighted"], (600 * .340 + 40 * .260) / 640, places=9)
        self.assertAlmostEqual(m["unweighted"], 0.300, places=9)
        self.assertAlmostEqual(
            m["mean_w"], ((k / (600 + k)) + (k / (40 + k))) / 2, places=9)
        self.assertAlmostEqual(m["bias"], m["mean_w"] * (m["unweighted"] - prior),
                               places=9)
        # The gap the diagnostic exists to surface: PA weighting flatters the
        # pool relative to the players in it.
        self.assertGreater(m["weighted"], m["unweighted"])

    def test_zero_gap_pool_reports_zero_bias(self):
        prior, k = 0.317, 100.0
        cust = self._cust([(1, 0.317, 500), (2, 0.317, 20)])
        m = dict(build_site.prior_population_centres(cust, None, prior, k))["batters"]
        self.assertAlmostEqual(m["unweighted"], prior, places=9)
        self.assertAlmostEqual(m["bias"], 0.0, places=9)

    def test_pitcher_subpools_use_the_relief_pitcher_ids_predicate(self):
        """A swingman on the boundary must land in the same pool the model
        would shrink him in. If RP_MAX_START_SHARE moves, this fails here
        rather than silently describing a pool the bullpen never used."""
        prior, k = 0.310, 100.0
        share = build_site.RP_MAX_START_SHARE
        ip_cap = build_site.RP_MAX_IP_PER_APPEARANCE
        roles = {
            11: {"start_share": share, "avg_ip_per_appearance": ip_cap},      # RP
            12: {"start_share": share + 0.01, "avg_ip_per_appearance": 1.0},  # SP
            13: {"start_share": 0.0, "avg_ip_per_appearance": ip_cap + 0.1},  # neither
        }
        cust = self._cust([(11, 0.280, 200), (12, 0.330, 500), (13, 0.300, 90)])
        rows = dict(build_site.prior_population_centres(
            cust, cust, prior, k, role_map=roles))
        self.assertEqual(rows["  pitchers:RP"]["n"], 1)
        self.assertAlmostEqual(rows["  pitchers:RP"]["unweighted"], 0.280, places=9)
        self.assertEqual(rows["  pitchers:SP"]["n"], 1)
        self.assertAlmostEqual(rows["  pitchers:SP"]["unweighted"], 0.330, places=9)
        # The long reliever is excluded from both, exactly as relief_pitcher_ids
        # excludes him and the rotation split does not claim him.
        self.assertEqual(rows["pitchers"]["n"], 3)

    def test_slate_probable_pool_is_the_starters_actually_shrunk(self):
        prior, k = 0.310, 100.0
        cust = self._cust([(21, 0.330, 500), (22, 0.290, 480), (23, 0.360, 120)])
        rows = dict(build_site.prior_population_centres(
            cust, cust, prior, k, probable_ids={21, 23}))
        m = rows["  pitchers:SP(slate)"]
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["unweighted"], (0.330 + 0.360) / 2, places=9)

    def test_degrades_without_roles_and_without_a_usable_prior(self):
        cust = self._cust([(1, 0.320, 300)])
        # No role map: the pitcher pool stays undifferentiated rather than
        # guessing a split -- the current model's assumption, stated.
        labels = [l for l, _ in build_site.prior_population_centres(
            cust, cust, 0.310, 100.0)]
        self.assertIn("batters", labels)
        self.assertIn("pitchers", labels)
        self.assertNotIn("  pitchers:RP", labels)
        # An unusable target yields nothing and must not raise.
        for bad in (None, float("nan")):
            self.assertEqual(
                build_site.prior_population_centres(cust, cust, bad, 100.0), [])

    def test_empty_and_zero_weight_pools_are_dropped_not_zero_filled(self):
        self.assertIsNone(build_site._pool_moments([], [], 0.31, 100.0))
        self.assertIsNone(build_site._pool_moments([0.3], [0], 0.31, 100.0))
        self.assertIsNone(build_site._pool_moments([np.nan], [500], 0.31, 100.0))

    def test_diagnostic_does_not_mutate_its_inputs(self):
        cust = self._cust([(1, 0.340, 600), (2, 0.260, 40)])
        before = cust.copy(deep=True)
        build_site.log_prior_population_centres(cust, cust, 0.31, 100.0,
                                                role_map={1: {"start_share": 0.0,
                                                              "avg_ip_per_appearance": 1.0}},
                                                probable_ids={2})
        pd.testing.assert_frame_equal(cust, before)


class PoolShrinkTargetTests(unittest.TestCase):
    """The batter percentile target. Display-only: the lean path must keep the
    league rate, and these lock that in."""

    SRC = build_site.MODEL_RATE_SOURCE_COL

    def _cust(self, rows):
        return pd.DataFrame([{"player_id": p, self.SRC: r, "pa": n}
                             for (p, r, n) in rows])

    def test_is_the_unweighted_pool_mean(self):
        rows = [(1, 0.340, 600), (2, 0.260, 40), (3, 0.300, 250)]
        self.assertAlmostEqual(
            build_site.pool_shrink_target(self._cust(rows)),
            (0.340 + 0.260 + 0.300) / 3, places=9)

    def test_sits_below_the_pa_weighted_rate_when_playing_time_tracks_talent(self):
        """The gap this exists to close. Equal by construction when playing
        time carries no information -- the target is not a blanket subtraction."""
        cust = self._cust([(1, 0.360, 600), (2, 0.330, 400),
                           (3, 0.280, 60), (4, 0.240, 10)])
        pa_weighted = float(np.average(cust[self.SRC], weights=cust["pa"]))
        self.assertLess(build_site.pool_shrink_target(cust), pa_weighted)

        flat = self._cust([(1, 0.360, 300), (2, 0.240, 300)])
        self.assertAlmostEqual(
            build_site.pool_shrink_target(flat),
            float(np.average(flat[self.SRC], weights=flat["pa"])), places=9)

    def test_tiny_samples_scatter_without_biasing_the_mean(self):
        """Sampling noise on a rate is mean-zero (`E[k/n] = p`), so a min=1
        board's small lines do not drag the centre -- the reason unweighted is
        usable here even though it would wreck a variance or a quantile.

        This is a claim about bias, so it is tested across draws. A single
        draw of a pool this small scatters with sd ~0.013; asserting one draw
        near talent would be a coin flip dressed as a test."""
        talent = 0.300
        base = [(i, talent, 500) for i in range(60)]
        self.assertAlmostEqual(
            build_site.pool_shrink_target(self._cust(base)), talent, places=9)

        ests = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            noisy = list(base)
            for i in range(200):
                pa = int(rng.integers(1, 15))
                noisy.append((1000 + i, float(rng.binomial(pa, talent)) / pa, pa))
            ests.append(build_site.pool_shrink_target(self._cust(noisy)))
        ests = np.array(ests)
        self.assertAlmostEqual(float(ests.mean()), talent, delta=0.002)
        # And the scatter is real: the target does wobble build to build.
        self.assertGreater(float(ests.std()), 0.005)

    def test_no_threshold_anywhere_in_the_derivation(self):
        """Sweeping one player's PA cannot move the target at all: PA decides
        only membership (n > 0), never a weight or a cut. A pool centre gated
        on `>= N` would step as that player crossed it."""
        vals = [build_site.pool_shrink_target(
            self._cust([(1, 0.340, 600), (2, 0.240, pa)]))
            for pa in range(1, 400, 3)]
        self.assertEqual(len(set(round(v, 12) for v in vals)), 1)

    def test_unusable_inputs_return_none_so_the_caller_falls_back(self):
        self.assertIsNone(build_site.pool_shrink_target(None))
        self.assertIsNone(build_site.pool_shrink_target(pd.DataFrame()))
        self.assertIsNone(build_site.pool_shrink_target(self._cust([(1, 0.3, 0)])))
        self.assertIsNone(build_site.pool_shrink_target(
            self._cust([(1, np.nan, 500)])))

    def test_lower_target_drops_a_low_pa_bat_far_more_than_a_regular(self):
        """The defect this fixes: sub-qualified bats rank high because they are
        regressed toward a centre above their own pool. Regulars move too --
        every shrunk value shifts by `w_i x delta` -- but `w` is 5x smaller for
        them, so the ranking between the two corrects."""
        k, league, pool = 100.0, 0.3163, 0.3050
        # Dense reference so the ratio is not read through searchsorted's
        # discretization; the exact ratio is the weight ratio
        # (100/140) / (100/700) = 5.0.
        ref = np.sort(np.linspace(0.290, 0.360, 4000))
        fringe_drop = (build_site.pctile_rank(0.300, 40, ref, league, k)
                       - build_site.pctile_rank(0.300, 40, ref, pool, k))
        reg_drop = (build_site.pctile_rank(0.300, 600, ref, league, k)
                    - build_site.pctile_rank(0.300, 600, ref, pool, k))
        self.assertGreater(fringe_drop, 0.0)
        self.assertGreater(reg_drop, 0.0)
        self.assertAlmostEqual(fringe_drop / reg_drop, 5.0, delta=0.2)

    def test_lean_path_keeps_the_league_rate_not_the_display_target(self):
        """Guards the claim that makes this display-only: if the lean ever
        starts reading the percentile target, it becomes prediction math and
        owes a MODEL_TAG bump. Fail here instead."""
        lb = {"xwOBA": 0.3163, "_pctile_prior_bat": 0.3050}
        with (
            mock.patch.object(
                build_site, "segment_pitcher_blocks",
                return_value=("pitcher_rows", "opposing_hitters"),
            ),
            mock.patch.object(
                build_site, "aggregate_lineup", return_value="lineup_aggregate"
            ) as aggregate,
            mock.patch.object(
                build_site, "build_matchup", return_value=pd.DataFrame()
            ) as matchup,
        ):
            build_site.build_xwoba_matchup(pd.DataFrame(), lb)
        self.assertEqual(aggregate.call_args.kwargs["shrink_prior"], 0.3163)
        self.assertEqual(matchup.call_args.kwargs["shrink_prior"], 0.3163)


class PlatoonXwobaAdjustmentTests(unittest.TestCase):
    OFFSETS = build_site.PLATOON_XWOBA_OFFSETS
    ADV = build_site.PLATOON_ADV_COL
    THR = build_site.SP_THROWS_COL

    # --- the handedness fact: who holds the edge -----------------------------

    def test_tag_follows_handedness_convention(self):
        self.assertTrue(build_site.platoon_advantage("L", "R"))
        self.assertTrue(build_site.platoon_advantage("R", "L"))
        self.assertFalse(build_site.platoon_advantage("R", "R"))
        self.assertFalse(build_site.platoon_advantage("L", "L"))
        # A switch hitter takes the opposite side, so he always holds the edge.
        self.assertTrue(build_site.platoon_advantage("S", "R"))
        self.assertTrue(build_site.platoon_advantage("S", "L"))
        # So does a hitter with no recorded side -- same convention as the lens.
        self.assertTrue(build_site.platoon_advantage(None, "L"))

    def test_unknown_starter_hand_is_untagged(self):
        self.assertIsNone(build_site.platoon_advantage("R", None))
        self.assertIsNone(build_site.platoon_advantage("R", ""))
        self.assertIsNone(build_site.effective_stand("R", None))

    def test_tag_matches_the_platoon_lens_tag(self):
        # One convention drives the card marker and the lens's platoon_adv
        # column; they must never disagree about the same hitter.
        for bats in ("L", "R", "S", None):
            for throws in ("L", "R"):
                eff = build_site.effective_stand(bats, throws)
                self.assertEqual(build_site.platoon_advantage(bats, throws),
                                 eff != throws)

    # --- the offset: who gets moved, and by how much -------------------------

    def test_one_sided_hitters_use_the_centered_matchup_table(self):
        for matchup, expected in self.OFFSETS.items():
            self.assertAlmostEqual(build_site.platoon_xwoba_offset(*matchup), expected)

    def test_switch_hitter_holds_the_edge_but_is_not_moved(self):
        # He bats opposite the starter in essentially every PA, so his season
        # xwOBA already IS his advantage-state number -- adding the term again
        # would count the same edge twice.
        for throws in ("L", "R"):
            self.assertTrue(build_site.platoon_advantage("S", throws))
            self.assertEqual(build_site.platoon_xwoba_offset("S", throws), 0.0)

    def test_unknown_side_or_hand_is_not_moved(self):
        # No evidence either way is not evidence of a disadvantage.
        self.assertEqual(build_site.platoon_xwoba_offset(None, "R"), 0.0)
        self.assertEqual(build_site.platoon_xwoba_offset("", "R"), 0.0)
        self.assertEqual(build_site.platoon_xwoba_offset("R", None), 0.0)
        self.assertEqual(build_site.platoon_xwoba_offset("S", None), 0.0)

    def test_each_one_sided_hitter_keeps_the_same_point_021_gap(self):
        self.assertAlmostEqual(
            build_site.platoon_xwoba_offset("L", "R")
            - build_site.platoon_xwoba_offset("L", "L"),
            .021,
        )
        self.assertAlmostEqual(
            build_site.platoon_xwoba_offset("R", "L")
            - build_site.platoon_xwoba_offset("R", "R"),
            .021,
        )

    def test_offsets_are_zero_at_their_assumed_exposure_mix(self):
        # Rounded research priors: LHB hold the edge about 16/21 of the time;
        # RHB about 6/21. Each season exposure blend returns exactly to zero.
        lhb = ((16 / 21) * build_site.platoon_xwoba_offset("L", "R")
               + (5 / 21) * build_site.platoon_xwoba_offset("L", "L"))
        rhb = ((6 / 21) * build_site.platoon_xwoba_offset("R", "L")
               + (15 / 21) * build_site.platoon_xwoba_offset("R", "R"))
        self.assertAlmostEqual(lhb, 0.0)
        self.assertAlmostEqual(rhb, 0.0)

    # --- through aggregate_lineup --------------------------------------------

    @staticmethod
    def _lineup(rows, hand="R"):
        # rows: (batting_order, xwOBA, PA, bats) facing a starter throwing `hand`
        return pd.DataFrame([
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=xw, PA=pa, BBE=50, batting_order=o,
                 bats=b, **{build_site.SP_THROWS_COL: hand})
            for (o, xw, pa, b) in rows
        ])

    def test_advantage_bats_up_and_the_rest_down_after_shrinkage(self):
        prior, k = 0.317, 150.0
        H = self._lineup([(1, 0.500, 15, "L"), (2, 0.300, 550, "R")], hand="R")
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True,
                                          shrink_prior=prior, shrink_k=k)
        # Shrink first, then apply the population cell. The term itself is not
        # diluted by a hitter's overall-PA sample.
        s1 = ((15 * .500 + k * prior) / (15 + k)
              + self.OFFSETS[("L", "R")])
        s2 = ((550 * .300 + k * prior) / (550 + k)
              + self.OFFSETS[("R", "R")])
        w1, w2 = build_site.LINEUP_SLOT_PA[1], build_site.LINEUP_SLOT_PA[2]
        expected = (w1 * s1 + w2 * s2) / (w1 + w2)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], expected)

    def test_aggregate_preserves_neutral_and_starter_adjusted_composites(self):
        rows = [(1, .320, 400, "L"), (2, .320, 400, "L")]
        agg = build_site.aggregate_lineup(
            self._lineup(rows, hand="R"), ["xwOBA"], weighted=True
        )
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA_neutral"], .320)
        adj = self.OFFSETS[("L", "R")]
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA_vs_sp"], .320 + adj)
        self.assertAlmostEqual(agg.loc[0, "platoon_delta_sp"], adj)
        self.assertAlmostEqual(
            agg.loc[0, "opp_xwOBA"],
            agg.loc[0, "opp_xwOBA_vs_sp"],
        )

    def test_all_lhb_lineup_spread_matches_the_point_021_gap(self):
        prior, k = 0.317, 150.0
        rows = [(1, .340, 400, "L"), (2, .300, 500, "L"), (3, .280, 600, "L")]
        adv = build_site.aggregate_lineup(self._lineup(rows, hand="R"), ["xwOBA"],
                                          weighted=True, shrink_prior=prior, shrink_k=k)
        none = build_site.aggregate_lineup(self._lineup(rows, hand="L"), ["xwOBA"],
                                           weighted=True, shrink_prior=prior, shrink_k=k)
        self.assertAlmostEqual(adv.loc[0, "opp_xwOBA"] - none.loc[0, "opp_xwOBA"],
                               .021)

    def test_all_switch_lineup_composite_is_untouched(self):
        prior, k = 0.317, 150.0
        rows = [(1, .340, 400, "S"), (2, .300, 500, "S")]
        with_term = build_site.aggregate_lineup(self._lineup(rows), ["xwOBA"],
                                                weighted=True, shrink_prior=prior,
                                                shrink_k=k)
        plain = build_site.aggregate_lineup(
            self._lineup(rows).drop(columns=[self.THR]), ["xwOBA"],
            weighted=True, shrink_prior=prior, shrink_k=k)
        self.assertAlmostEqual(with_term.loc[0, "opp_xwOBA"], plain.loc[0, "opp_xwOBA"])

    def test_switch_bat_does_not_lift_a_mixed_lineup(self):
        # The regression this guards: a switch hitter next to a same-handed bat
        # used to cancel him out and leave the composite at league-ish level.
        rows = [(1, .320, 400, "R"), (2, .320, 400, "S")]
        agg = build_site.aggregate_lineup(self._lineup(rows, hand="R"), ["xwOBA"],
                                          weighted=True)
        w1, w2 = build_site.LINEUP_SLOT_PA[1], build_site.LINEUP_SLOT_PA[2]
        expected = (w1 * (.320 + self.OFFSETS[("R", "R")]) + w2 * .320) / (w1 + w2)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], expected)
        self.assertLess(agg.loc[0, "opp_xwOBA"], .320)

    def test_lineup_without_the_columns_is_left_alone(self):
        prior, k = 0.317, 150.0
        rows = [(1, .340, 400, "R"), (2, .300, 500, "L")]
        H = self._lineup(rows).drop(columns=["bats", self.THR])
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True,
                                          shrink_prior=prior, shrink_k=k)
        s1 = (400 * .340 + k * prior) / (400 + k)
        s2 = (500 * .300 + k * prior) / (500 + k)
        w1, w2 = build_site.LINEUP_SLOT_PA[1], build_site.LINEUP_SLOT_PA[2]
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], (w1 * s1 + w2 * s2) / (w1 + w2))

    def test_team_backfilled_bat_still_gets_the_platoon_term(self):
        # The backfill bypasses player-level shrinkage, not the matchup term.
        H = pd.DataFrame([
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=.340, PA=0, BBE=0, batting_order=1,
                 bats="L", **{BACKFILL_COL: True, self.THR: "R"}),
        ])
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True,
                                          shrink_prior=.317, shrink_k=175.0)
        self.assertAlmostEqual(
            agg.loc[0, "opp_xwOBA"], .340 + self.OFFSETS[("L", "R")]
        )

    def test_only_xwoba_moves(self):
        H = pd.DataFrame([
            dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
                 batting_side="home", xwOBA=.340, xSLG=.450, PA=400, BBE=50,
                 batting_order=1, bats="L", **{self.THR: "R"}),
        ])
        agg = build_site.aggregate_lineup(H, ["xwOBA", "xSLG"], weighted=True)
        self.assertAlmostEqual(
            agg.loc[0, "opp_xwOBA"], .340 + self.OFFSETS[("L", "R")]
        )
        self.assertAlmostEqual(agg.loc[0, "opp_xSLG"], .450)

    # --- through the real segment -> aggregate -> render path ----------------

    @staticmethod
    def _block(sp_throws, bats):
        # One SP block: the probable, then his opposing lineup.
        meta = dict(game_pk=1, table_index=1, sp_side="away")
        rows = [dict(meta, Name="SP", is_sp=True, throws=sp_throws, bats="R",
                     xwOBA=.300, PA=500)]
        for i, b in enumerate(bats, start=1):
            # Hitters throw right regardless -- the tag must read the SP's hand.
            rows.append(dict(meta, Name=f"H{i}", is_sp=False, throws="R", bats=b,
                             xwOBA=.320, PA=400, batting_order=i))
        return pd.DataFrame(rows)

    def test_segment_tags_hitters_with_the_faced_starters_hand(self):
        _, H = build_site.segment_pitcher_blocks(
            self._block("L", ["L", "R", "S"]), ["xwOBA"])
        self.assertEqual(list(H[self.THR]), ["L", "L", "L"])
        # LHB vs LHP: no edge. RHB and switch vs LHP: edge.
        self.assertEqual(list(H[self.ADV]), [False, True, True])

    def test_segment_leaves_the_tag_null_when_the_starter_hand_is_missing(self):
        _, H = build_site.segment_pitcher_blocks(
            self._block(None, ["L", "R"]), ["xwOBA"])
        self.assertTrue(all(v is None for v in H[self.ADV]))
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], .320)

    def test_switch_bat_is_marked_on_the_card_but_not_moved_in_the_lean(self):
        # The two questions differ for exactly this hitter: the card must still
        # show the marker, and the composite must not move for him.
        _, H = build_site.segment_pitcher_blocks(
            self._block("R", ["S", "S"]), ["xwOBA"])
        hitters = build_site._hitters_for(H, pd.DataFrame(), 1, "SP", None)
        self.assertEqual([h["adv"] for h in hitters], [True, True])
        agg = build_site.aggregate_lineup(H, ["xwOBA"], weighted=True)
        self.assertAlmostEqual(agg.loc[0, "opp_xwOBA"], .320)

    def test_card_marks_the_bats_that_hold_the_edge(self):
        # The platoon-OPS lens is optional and can abstain; the marker must
        # still come up without it.
        _, H = build_site.segment_pitcher_blocks(
            self._block("R", ["L", "R"]), ["xwOBA"])
        hitters = build_site._hitters_for(H, pd.DataFrame(), 1, "SP", None)
        self.assertEqual([h["adv"] for h in hitters], [True, False])


class PriceBandRecordsTests(unittest.TestCase):
    """The per-game verdict's record, scored against price rather than raw.

    Structural: no expected record or band boundary is frozen beyond the ones
    the function itself defines.
    """

    COLS = ["status", "xw_lean", "xw_full", "home", "away", "close_p_home"]

    def _led(self, rows):
        return pd.DataFrame(rows, columns=self.COLS)

    def test_bands_key_off_the_leaned_side_not_the_home_side(self):
        """A game reads the same whichever team the model picked.

        Two mirror-image rows: the model leans a side priced at .62 in both.
        They must land in the same band, which the old home-relative split
        could not express.
        """
        rows = [("graded", "NYY", "W", "TB", "NYY", 0.38),   # away lean, away at .62
                ("graded", "TB", "W", "TB", "NYY", 0.62)]    # home lean, home at .62
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=self._led(rows)), \
                mock.patch.object(build_site, "VERDICT_CONTEXT_MIN", 1):
            ctx = build_site.price_band_records()
        self.assertEqual(list(ctx), [("band", "fav")])
        self.assertEqual(ctx[("band", "fav")]["n"], 2)
        self.assertAlmostEqual(ctx[("band", "fav")]["implied"], 0.62, places=9)

    def test_excess_is_measured_against_price_not_against_half(self):
        rows = [("graded", "TB", "W", "TB", "NYY", 0.60),
                ("graded", "TB", "L", "TB", "NYY", 0.60)]
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=self._led(rows)), \
                mock.patch.object(build_site, "VERDICT_CONTEXT_MIN", 1):
            parts = build_site.price_band_records()[("band", "fav")]
        # 1-1 at a .60 price is a 10-point shortfall against the price, not a
        # coin flip that happened to land even.
        self.assertAlmostEqual(parts["actual"], 0.50, places=9)
        self.assertAlmostEqual(parts["excess"], -0.10, places=9)
        self.assertAlmostEqual(parts["se"], build_site._excess_se([.6, .6]),
                               places=12)

    def test_exact_pickem_is_not_filed_as_a_disagreement(self):
        """The old split read `close_p_home >= .5` as 'home favored'.

        On a .500 line there is no favorite, so an away lean was recorded as
        disagreeing with a market that had no opinion. A band on the leaned
        side's own price cannot make that claim.
        """
        rows = [("graded", "NYY", "W", "TB", "NYY", 0.50),
                ("graded", "TB", "L", "TB", "NYY", 0.50)]
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=self._led(rows)), \
                mock.patch.object(build_site, "VERDICT_CONTEXT_MIN", 1):
            ctx = build_site.price_band_records()
        self.assertEqual(list(ctx), [("band", "even")])
        self.assertEqual(ctx[("band", "even")]["n"], 2)

    def test_abstained_and_unpriced_rows_never_enter(self):
        rows = [("graded", None, None, "TB", "NYY", 0.60),      # v5 abstention
                ("graded", "TB", "W", "TB", "NYY", float("nan")),  # no market
                ("pending", "TB", "W", "TB", "NYY", 0.60),      # not graded
                ("graded", "TB", "W", "TB", "NYY", 0.60)]
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=self._led(rows)), \
                mock.patch.object(build_site, "VERDICT_CONTEXT_MIN", 1):
            ctx = build_site.price_band_records()
        self.assertEqual(ctx[("band", "fav")]["n"], 1)

    def test_thin_band_is_omitted_rather_than_published(self):
        rows = [("graded", "NYY", "W", "TB", "NYY", 0.40)]
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=self._led(rows)):
            self.assertEqual(build_site.price_band_records(), {})

    def test_bands_partition_every_representable_price(self):
        """No price may fall through, and none may match two bands."""
        for i in range(1, 1000):
            p = i / 1000.0
            hits = [k for k, lo, hi in build_site._PRICE_BANDS if lo <= p < hi]
            self.assertEqual(len(hits), 1, f"p={p} matched {hits}")
        for bad in (0.0, 1.0, None, float("nan")):
            self.assertIsNone(build_site._price_band(bad))


class BaselineControlTests(unittest.TestCase):
    """The grades page publishes a record; these are what it is measured against."""

    COLS = ["full_away", "full_home", "close_p_home"]

    def _g(self, rows):
        return pd.DataFrame(rows, columns=self.COLS)

    def test_home_and_market_controls_over_the_same_rows(self):
        rows = [
            (2, 4, 0.60),   # home won, home favored  -> home W, chalk W
            (5, 1, 0.60),   # away won, home favored  -> home L, chalk L
            (5, 1, 0.40),   # away won, away favored  -> home L, chalk W
            (1, 3, 0.40),   # home won, away favored  -> home W, chalk L
            (6, 2, 0.45),   # away won, away favored  -> home L, chalk W
        ]
        ctl = dict((k, (w, l)) for k, w, l in build_site._baseline_controls(self._g(rows)))
        self.assertEqual(ctl["home"], (2, 3))
        self.assertEqual(ctl["market"], (3, 2))

    def test_a_pick_em_close_counts_as_a_home_favorite(self):
        # p_home == .5 has to fall on one side; the verdict context already
        # reads >= .5 as home-favored, so the control must agree with it.
        ctl = dict((k, (w, l)) for k, w, l in
                   build_site._baseline_controls(self._g([(1, 3, 0.50)])))
        self.assertEqual(ctl["market"], (1, 0))

    def test_rows_without_a_close_are_scored_by_the_home_control_only(self):
        rows = [(1, 3, float("nan")), (1, 3, 0.60)]
        ctl = dict((k, (w, l)) for k, w, l in build_site._baseline_controls(self._g(rows)))
        self.assertEqual(ctl["home"], (2, 0))
        self.assertEqual(ctl["market"], (1, 0))

    def test_unplayed_and_tied_rows_score_nothing(self):
        rows = [(float("nan"), float("nan"), 0.60), (3, 3, 0.60)]
        self.assertEqual(build_site._baseline_controls(self._g(rows)), [])

    def test_market_control_is_omitted_when_no_row_carries_a_close(self):
        g = pd.DataFrame([(1, 3)], columns=["full_away", "full_home"])
        self.assertEqual([k for k, _, _ in build_site._baseline_controls(g)], ["home"])

    def test_page_shows_the_controls_next_to_the_record(self):
        # Tagged MODEL_TAG, not a frozen historical tag: the header scores
        # RECORD_TAGS, so a literal here would silently stop exercising the
        # page at the next bump -- the memorised-constant failure this repo
        # has already had in a test once.
        ledger = pd.DataFrame([
            dict(game_pk=1, game_date="2026-07-20", away="A", home="B",
                 away_sp="P1", home_sp="P2", status="graded",
                 model_tag=build_site.MODEL_TAG, xw_lean="B", xw_delta=.01,
                 xw_full="W", xw_f5="W", full_away=2, full_home=4,
                 close_p_home=.6, lock_status="pregame"),
            dict(game_pk=2, game_date="2026-07-21", away="C", home="D",
                 away_sp="P3", home_sp="P4", status="graded",
                 model_tag=build_site.MODEL_TAG, xw_lean="C", xw_delta=.02,
                 xw_full="W", xw_f5="W", full_away=5, full_home=3,
                 close_p_home=.6, lock_status="pregame"),
        ])
        with mock.patch.object(build_site, "load_ledger_df", return_value=ledger):
            page = build_site.render_grades_html("test build")
        self.assertIn("Always home", page)
        self.assertIn("Always chalk", page)
        self.assertIn("controls on the same decided rows", page)

    def test_an_abstained_game_is_scored_by_neither_the_record_nor_a_control(self):
        """v5 abstains, so a graded row can carry no lean. A control needs no
        lean to score a game, so handed every graded row it would publish a
        baseline over more games than the record it sits beside -- and the
        `n=` marker would stay silent, because that control's n matches the
        graded count exactly."""
        # MODEL_TAG rather than the v5 literal that first shipped the
        # abstention: the page scores RECORD_TAGS, and what is under test is
        # the page's arithmetic over a leanless row, not which version can
        # produce one. v7's zero-delta rule means the current family can carry
        # one too.
        def row(pk, lean, full, fa, fh):
            return dict(game_pk=pk, game_date="2026-08-07", away="A", home="B",
                        away_sp="P1", home_sp="P2", status="graded",
                        model_tag=build_site.MODEL_TAG, xw_lean=lean,
                        xw_delta=.01, xw_full=full, xw_f5=full,
                        full_away=fa, full_home=fh, close_p_home=.6,
                        lock_status="pregame")
        ledger = pd.DataFrame([
            row(1, "B", "W", 2, 4),          # decided, home won
            row(2, "B", "L", 5, 3),          # decided, away won
            row(3, None, None, 1, 7),        # ABSTAINED -- played, never called
        ])
        decided = ledger[ledger["xw_lean"].notna()]
        self.assertEqual(
            dict((k, (w, l)) for k, w, l in build_site._baseline_controls(decided)),
            {"home": (1, 1), "market": (1, 1)},
            "controls must ignore the game the model abstained on",
        )
        # Scored over every graded row instead, the home control would read
        # 2-1 -- three games against the record's two.
        self.assertEqual(
            dict((k, (w, l)) for k, w, l in build_site._baseline_controls(ledger))["home"],
            (2, 1))
        with mock.patch.object(build_site, "load_ledger_df", return_value=ledger):
            page = build_site.render_grades_html("test build")
        self.assertIn("1 abstained", page)          # the count is stated
        self.assertIn(">1-1<", page)                # controls over the 2 decided
        self.assertNotIn(">2-1<", page)             # never the 3-game baseline


class RecordScopeTests(unittest.TestCase):
    """The two surfaces that publish a record score RECORD_TAGS.

    A record is a claim about THIS model. Pooling twelve prediction families
    into one win-loss line answers a different question, and the repo already
    has the authority for which families may share a line -- RECORD_TAGS,
    the same rule data/ledger_report.txt scores. These pin that the public
    surfaces and the internal report cannot drift back apart."""

    def _row(self, pk, tag, lean, full, fa, fh):
        return dict(game_pk=pk, game_date="2026-08-07", away="A", home="B",
                    away_sp="P1", home_sp="P2", status="graded",
                    model_tag=tag, model_metric="xwOBA", xw_lean=lean,
                    xw_delta=.01, xw_full=full, xw_f5=full,
                    full_away=fa, full_home=fh, close_p_home=.6,
                    lock_status="pregame")

    def _mixed(self):
        """One win in the current family; three losses in an older one."""
        cur = build_site.MODEL_TAG
        return pd.DataFrame([
            self._row(1, cur, "B", "W", 2, 4),
            self._row(2, "xw+plat_consol_v2", "B", "L", 5, 3),
            self._row(3, "xw+plat_consol_v2", "B", "L", 5, 3),
            self._row(4, "xw+plat_consol_v2", "B", "L", 5, 3),
        ])

    # Assertions run against the HEADER, never the whole page: the archive
    # table below it prints a per-day record for every family, and the base64
    # font blob matches most short digit strings. Both are correct and
    # neither is the claim under test.
    def _pages(self, led):
        with mock.patch.object(build_site, "load_ledger_df", return_value=led):
            page = build_site.render_grades_html("test build")
            return {"strip": build_site.records_strip_html(),
                    "grades page": page.split("<div class='gr-tablewrap'>")[0]}

    def _marks(self, rec):
        return {"strip": f"full {rec}", "grades page": f">{rec}<"}

    def test_strip_and_header_score_only_the_current_family(self):
        pages = self._pages(self._mixed())
        # 1-0 is the current family. 1-3 is every graded row, which is what
        # both surfaces used to print.
        mine, pooled = self._marks("1-0"), self._marks("1-3")
        for name, html in pages.items():
            self.assertIn(mine[name], html,
                          f"{name} lost the current-family record")
            self.assertNotIn(pooled[name], html,
                             f"{name} still pools every family")

    def test_the_scope_of_the_number_is_stated_next_to_it(self):
        """A record over a subset, printed above a table of every row, has to
        say which rows -- otherwise it reads as a summary of the table."""
        for name, html in self._pages(self._mixed()).items():
            self.assertIn("1 of 4 graded rows", html, f"{name} states no scope")
            self.assertIn(build_site.MODEL_TAG, html, f"{name} names no family")

    def test_no_scope_note_when_the_family_is_every_graded_row(self):
        """The note has to disappear on its own, or it becomes noise that a
        reader learns to skip."""
        cur = build_site.MODEL_TAG
        led = pd.DataFrame([self._row(1, cur, "B", "W", 2, 4)])
        strip = self._pages(led)["strip"]
        self.assertIn("full 1-0", strip)
        self.assertNotIn("graded rows are", strip)

    def test_an_empty_family_never_falls_back_to_the_pooled_record(self):
        """The regression this scoping could most easily introduce.

        A MODEL_TAG bump empties the family until its first row grades -- v11
        shipped and was superseded without ever producing one. Showing the
        previous family's record under the new model's name would be the
        `wOBA full 217-164` substitution again, with the tag rather than the
        metric label as the lie."""
        led = pd.DataFrame([
            self._row(2, "xw+plat_consol_v2", "B", "W", 2, 4),
            self._row(3, "xw+plat_consol_v2", "B", "W", 2, 4),
        ])
        foreign = self._marks("2-0")
        for name, html in self._pages(led).items():
            self.assertNotIn(foreign[name], html,
                             f"{name} published a foreign family's record")
            self.assertIn(f"graded games yet under {build_site.MODEL_TAG}",
                          html, f"{name} did not say the family is empty")
            # ...and it still points at where the history went, so an empty
            # headline never reads as "this model has never been graded".
            self.assertIn("2 graded", html.replace("; 2 rows graded", "; 2 graded"),
                          f"{name} hid the historical rows entirely")

    def test_the_team_page_still_pools_and_says_so(self):
        """Per-club accuracy needs volume, not comparability -- a single
        family leaves most clubs one or two games. It keeps pooling, and the
        lead states it so the two pages cannot read as contradicting."""
        led = self._mixed()
        with mock.patch.object(build_site, "load_ledger_df", return_value=led):
            page = build_site.render_team_grades_html("test build")
        self.assertIn("Pooled over every graded model family", page)


class LockProvenanceTests(unittest.TestCase):
    def _led(self, statuses):
        return pd.DataFrame({"lock_status": pd.Series(statuses, dtype="object")})

    def test_late_snapshots_are_not_counted_as_legacy_rows(self):
        led = self._led(["pregame", "pregame_recovered", None,
                         "legacy_unverified", "late_snapshot"])
        self.assertEqual(build_site._lock_provenance(led), (2, 2, 1))

    def test_a_ledger_without_the_column_is_wholly_unverified(self):
        led = pd.DataFrame({"game_pk": [1, 2, 3]})
        self.assertEqual(build_site._lock_provenance(led), (0, 3, 0))

    def test_page_states_late_snapshots_separately(self):
        # MODEL_TAG so the row survives the header's RECORD_TAGS filter; the
        # lock note itself is whole-ledger, but the page needs a non-empty
        # family to render the summary block that carries it.
        ledger = pd.DataFrame([
            dict(game_pk=1, game_date="2026-07-20", away="A", home="B",
                 away_sp="P1", home_sp="P2", status="graded",
                 model_tag=build_site.MODEL_TAG, xw_lean="B", xw_delta=.01,
                 xw_full="W", xw_f5="W", full_away=2, full_home=4,
                 lock_status="late_snapshot"),
        ])
        with mock.patch.object(build_site, "load_ledger_df", return_value=ledger):
            page = build_site.render_grades_html("test build")
        self.assertIn("1 snapshotted after first pitch", page)
        self.assertNotIn("1 legacy rows", page)


class ModelTagProvenanceTests(unittest.TestCase):
    """The ledger's lineage stamp must describe the math that produced the row.

    The v10 bump moved both modules' defaults and family maps but left
    `.github/workflows/build.yml` pinning MODEL_TAG=xw+plat_consol_v9, so CI ran
    v10's PA-share weighting and stamped every row v9. The workflow env was a
    third copy of the version that nothing kept in sync.
    """

    WORKFLOW = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".github", "workflows", "build.yml",
    )

    def test_modules_agree_on_the_default_tag(self):
        self.assertEqual(build_site.MODEL_TAG, grade_leans.MODEL_TAG)

    def test_modules_agree_on_the_record_family(self):
        self.assertEqual(build_site.RECORD_TAGS, grade_leans.RECORD_TAGS)

    def test_current_tag_is_in_its_own_record_and_scale_families(self):
        self.assertIn(build_site.MODEL_TAG, build_site.RECORD_TAGS)
        self.assertIn(build_site.MODEL_TAG, build_site.SCALE_TAGS)

    def test_centered_platoon_bump_resets_record_but_keeps_woba_scale(self):
        """v2's own family decisions, which are history and cannot change.

        Previously read the LIVE tags, so every later bump broke a test about
        v2. The maps are the authority (CLAUDE.md), so ask them directly.
        """
        self.assertEqual(build_site._RECORD_FAMILIES["woba+plat_consol_v2"],
                         ("woba+plat_consol_v2",))
        self.assertEqual(grade_leans._RECORD_FAMILIES["woba+plat_consol_v2"],
                         ("woba+plat_consol_v2",))
        self.assertEqual(build_site._SCALE_FAMILIES["woba+plat_consol_v2"],
                         ("woba+plat_consol_v1", "woba+plat_consol_v2"))

    def test_v3_starts_a_new_record_and_a_new_scale_family(self):
        """Both halves of the v3 decision, argued in _RECORD_FAMILIES.

        K 100->400 changes every shrunk rate and the relief target moves every
        bullpen number, so the record resets; and quadrupling K compresses
        |xw_net|, so magnitudes are not comparable with v2 either.
        """
        self.assertEqual(build_site._RECORD_FAMILIES["woba+plat_consol_v3"],
                         ("woba+plat_consol_v3",))
        self.assertEqual(grade_leans._RECORD_FAMILIES["woba+plat_consol_v3"],
                         ("woba+plat_consol_v3",))
        self.assertEqual(build_site._SCALE_FAMILIES["woba+plat_consol_v3"],
                         ("woba+plat_consol_v3",))
        self.assertNotIn("woba+plat_consol_v2",
                         build_site._SCALE_FAMILIES["woba+plat_consol_v3"])

    def test_every_active_rate_source_agrees_on_one_metric(self):
        """The five constants that name the active rate must not disagree.

        The subject is INTERNAL CONSISTENCY, not which metric won. This used to
        assert wOBA specifically and would have had to be rewritten on the v11
        revert for a reason that has nothing to do with what it protects -- the
        real hazard is a half-applied metric switch, where the tag says one
        thing and the fetched column is another. Written this way it holds
        through the next swap too.

        `woba` and `xwoba` are BOTH real Savant selections, which is exactly
        why a half-switch would not raise: the wrong one resolves fine.
        """
        prefix, label, col = {
            "wOBA": ("woba+", "wOBA", "woba"),
            "xwOBA": ("xw+", "xwOBA", "xwoba"),
        }[build_site.MODEL_RATE_LABEL]
        self.assertTrue(build_site.MODEL_TAG.startswith(prefix))
        self.assertEqual(build_site.MODEL_RATE_SOURCE_COL, col)
        self.assertEqual(build_site.MODEL_RATE_LABEL, label)
        self.assertIn(col, build_site.STATCAST_SELECTIONS)
        # Exactly one rate is fetched: the other must not ride along.
        other = "xwoba" if col == "woba" else "woba"
        self.assertNotIn(other, build_site.STATCAST_SELECTIONS)
        # The cache namespace is keyed to the selection set; a stale namespace
        # serves a CSV that has no column under the new name.
        self.assertIn(col, build_site.STATCAST_CACHE_NS)
        self.assertEqual(build_site.XWOBA_SHRINK_COL,
                         build_site.MODEL_RATE_INTERNAL_COL)
        # The shadow arm must run the OTHER metric, or it is not a comparison.
        import shadow_metric
        self.assertEqual(shadow_metric.SHADOW_SOURCE_COL, other)

    def test_active_metric_wins_even_if_export_contains_both_rates(self):
        """A Savant export carrying both rates must yield the ACTIVE one.

        The internal key is `xwOBA` under either metric -- it is a
        compatibility schema, not a claim about the statistic -- so reading the
        wrong column produces a plausible number under the right key and
        nothing raises. Values are deliberately far apart so a mix-up cannot
        pass on rounding.
        """
        active = build_site.MODEL_RATE_SOURCE_COL
        other = "xwoba" if active == "woba" else "woba"
        custom = pd.DataFrame({
            "player_id": [123], "pa": [50],
            active: [.401], other: [.201],
        })
        batted = pd.DataFrame({"id": [123], "bbe": [20]})
        with mock.patch.object(build_site, "cached_csv",
                               side_effect=[custom, batted]) as fetch:
            stat, _, _ = build_site.load_stat_lookups("batter")
        self.assertAlmostEqual(stat[123][build_site.MODEL_RATE_INTERNAL_COL], .401)
        custom_url, cache_name = fetch.call_args_list[0].args
        self.assertIn(f"selections=pa,k_percent,bb_percent,{active},", custom_url)
        self.assertNotIn(f",{other},", custom_url)
        self.assertEqual(cache_name, f"{build_site.STATCAST_CACHE_NS}_batter")

    def test_pooled_record_is_named_for_its_rows_not_the_running_build(self):
        """A public record labelled with MODEL_RATE_LABEL is a false claim.

        The record strip, grades headline and team page all render
        _display_grades -- every graded family pooled. On 2026-08-03 the tag
        flipped to wOBA while all 381 graded rows were xwOBA, and those
        surfaces published "wOBA full 217-164 (.570)" over zero wOBA games.
        """
        legacy = pd.DataFrame({"model_tag": ["xw+plat_consol_v9",
                                             "xw+plat_consol_v10"]})
        self.assertEqual(market_backfill.metric_label(legacy), "xwOBA")

        # model_metric absent on every pre-wOBA row, so the tag must carry it.
        legacy_nulls = legacy.assign(model_metric=[np.nan, None])
        self.assertEqual(market_backfill.metric_label(legacy_nulls), "xwOBA")

        woba = pd.DataFrame({"model_tag": ["woba+plat_consol_v2"] * 2,
                             "model_metric": ["wOBA", "wOBA"]})
        self.assertEqual(market_backfill.metric_label(woba), "wOBA")

        # The abandoned split lineage remains nameable as immutable history,
        # even when an older dump lacks the explicit model_metric column.
        split = pd.DataFrame({"model_tag": ["split+plat_consol_v1"]})
        self.assertEqual(market_backfill.metric_label(split), "wOBA/xwOBA")

        # Mixed history must not claim either metric.
        self.assertEqual(
            market_backfill.metric_label(
                pd.concat([legacy_nulls, woba], ignore_index=True)),
            "Model",
        )

        # Filtered frames arrive with gaps in the index.
        self.assertEqual(
            market_backfill.metric_label(woba.set_axis([4, 17])), "wOBA")

    def test_metric_series_labels_each_row_and_keeps_the_frame_index(self):
        """`metric_label` is this collapsed to one name. The per-row form is
        what lets the per-slate block GROUP by metric instead of re-deriving
        the tag-prefix rule beside this one; a caller that reindexed silently
        would attach a wOBA label to an xwOBA row on the one mixed date.
        """
        mixed = pd.DataFrame(
            {"model_tag": ["xw+plat_consol_v10", "woba+plat_consol_v1",
                           "split+plat_consol_v1"],
             "model_metric": [np.nan, "wOBA", None]},
            index=[7, 2, 9],
        )
        s = market_backfill.metric_series(mixed)
        self.assertEqual(list(s.index), [7, 2, 9])
        self.assertEqual(list(s), ["xwOBA", "wOBA", "wOBA/xwOBA"])
        self.assertEqual(market_backfill.metric_label(mixed), "Model")

    def test_public_pages_do_not_name_a_metric_no_graded_row_used(self):
        """End-to-end guard on the three surfaces that publish the record.

        The 2026-08-03 state: every graded row on one metric while
        MODEL_RATE_LABEL names the other. Built from whichever metric the
        running build is NOT, so the mismatch keeps reproducing through any
        future revert instead of quietly becoming a no-op the day the tag
        happens to agree with the rows.

        The rows carry MODEL_TAG with a conflicting `model_metric`, which is
        what keeps this an end-to-end guard now the record surfaces score
        RECORD_TAGS. Tagging them with a foreign tag would leave those two
        pages empty and quietly reduce this to a team-page test. It is also
        the faithful form of the invariant: `metric_label` is defined to read
        `model_metric` off the rows, with the tag prefix only as a legacy
        fallback, so a current-family row whose metric disagrees with the
        build constant is exactly the case a constant-reader gets wrong.
        """
        other_label = ("wOBA" if build_site.MODEL_RATE_LABEL == "xwOBA"
                       else "xwOBA")
        ledger = pd.DataFrame([
            dict(game_pk=1, game_date="2026-07-20", away="A", home="B",
                 away_sp="P1", home_sp="P2", status="graded",
                 model_tag=build_site.MODEL_TAG, model_metric=other_label,
                 xw_lean="B", xw_delta=.01,
                 xw_full="W", xw_f5="W", full_away=2, full_home=4,
                 close_p_home=.6, close_home_ml=-140, close_away_ml=120,
                 f5_away=1, f5_home=2, f5_close_p_home=np.nan,
                 f5_close_home_ml=np.nan, f5_close_away_ml=np.nan,
                 ops_valid=False, ops_lean=None, lock_status="pregame"),
            dict(game_pk=2, game_date="2026-07-21", away="C", home="D",
                 away_sp="P3", home_sp="P4", status="graded",
                 model_tag=build_site.MODEL_TAG, model_metric=other_label,
                 xw_lean="C", xw_delta=.02,
                 xw_full="L", xw_f5="L", full_away=1, full_home=3,
                 close_p_home=.6, close_home_ml=-140, close_away_ml=120,
                 f5_away=0, f5_home=2, f5_close_p_home=np.nan,
                 f5_close_home_ml=np.nan, f5_close_away_ml=np.nan,
                 ops_valid=False, ops_lean=None, lock_status="pregame"),
        ])
        self.assertNotEqual(build_site.MODEL_RATE_LABEL, other_label)
        with mock.patch.object(build_site, "load_ledger_df", return_value=ledger):
            pages = {
                "record strip": build_site.records_strip_html(),
                "grades page": build_site.render_grades_html("test build"),
                "team page": build_site.render_team_grades_html("test build"),
            }
        for name, page in pages.items():
            self.assertIn(other_label, page, f"{name} lost its metric label")
            # CSS comments discuss both metrics by name and are not content.
            # Stripping them is narrower than relaxing the assertion: what is
            # under test is what a reader SEES credited to these rows.
            body = re.sub(r"/\*.*?\*/", "", page, flags=re.S)
            if other_label == "xwOBA":
                # A bare "wOBA" is the bug; the tail of "xwOBA" is not.
                self.assertIsNone(
                    re.search(r"(?<!x)wOBA", body),
                    f"{name} credits xwOBA rows to wOBA")
            else:
                self.assertNotIn("xwOBA", body,
                                 f"{name} credits wOBA rows to xwOBA")
        # The vs-market cell is looked up by that same label; a mismatch used
        # to drop it silently rather than raise.
        self.assertIn("vs mkt", pages["record strip"])
        self.assertIn("vs mkt", pages["grades page"])

    def test_vs_market_cell_key_matches_the_label_it_is_looked_up_by(self):
        """The vs-market bucket and its lookup must share one derivation.

        They did not: vs_market_summary keyed the bucket off the rows
        ("xwOBA") while build_site looked it up under MODEL_RATE_LABEL
        ("wOBA") with a "Model" fallback that only fires on mixed metrics.
        Both missed, so the z / flat-ROI cell -- the primary metric -- silently
        vanished from grades.html and the front strip instead of erroring.
        """
        d = pd.DataFrame({
            "model_tag": ["xw+plat_consol_v10"] * 4,
            "close_p_home": [.55, .48, .60, .42],
            "close_home_ml": [-120, 105, -150, 115],
            "close_away_ml": [100, -125, 130, -135],
            "full_home": [5, 2, 6, 1], "full_away": [3, 4, 2, 3],
            "f5_home": [3, 1, 4, 0], "f5_away": [1, 2, 1, 2],
            "f5_close_p_home": [np.nan] * 4,
            "f5_close_home_ml": [np.nan] * 4,
            "f5_close_away_ml": [np.nan] * 4,
            "xw_lean": ["HOU", "TOR", "HOU", "TOR"],
            "home": ["HOU"] * 4, "away": ["TOR"] * 4,
            "ops_valid": [False] * 4, "ops_lean": [None] * 4,
        })
        summary = market_backfill.vs_market_summary(d, verbose=False)
        label = market_backfill.metric_label(d)
        self.assertIn(label, summary)
        self.assertIsNotNone(summary.get(label))

    def test_workflow_does_not_pin_model_tags(self):
        with open(self.WORKFLOW, encoding="utf-8") as fh:
            lines = [ln.split("#", 1)[0] for ln in fh]
        for var in ("MODEL_TAG", "RECORD_TAGS", "SCALE_TAGS"):
            pinned = [ln.strip() for ln in lines if ln.strip().startswith(f"{var}:")]
            self.assertEqual(
                pinned, [],
                f"{var} is pinned in build.yml; it overrides the module defaults "
                "and silently drifts from them on the next version bump",
            )

class MarketCalibrationTests(unittest.TestCase):
    """Invariants of the market-calibration page.

    These are structural, not expectations about how the market performed: a
    failure here means the table is built wrong, not that the season went
    differently. That is the same line test_ledger_invariants.py holds.
    """

    def test_odds_ladder_tiles_every_representable_price(self):
        """No price may fall through, and none may match two rungs.

        The first version used `lo < ml <= hi`, which dropped exactly +100 --
        not greater than 100, not <= -100 -- so one observation per such game
        vanished silently and the home/away counts went 385 vs 384.
        """
        prices = list(range(-2000, -99)) + list(range(100, 2001))
        for ml in prices:
            hits = [lab for lo, hi, lab in build_site._ODDS_LADDER
                    if (lo is None or ml >= lo) and (hi is None or ml <= hi)]
            self.assertEqual(len(hits), 1,
                             f"American price {ml:+d} matched rungs {hits}")

    def _frame(self):
        return pd.DataFrame({
            "full_home": [5, 2, 6, 1, 4, 3],
            "full_away": [3, 4, 2, 3, 1, 7],
            "close_p_home": [.55, .48, .70, .42, .50, .62],
            # deliberately includes an even-money +100 price, the case that broke
            "close_home_ml": [-120, 105, -250, 115, 100, -160],
            "close_away_ml": [100, -125, 210, -135, -120, 140],
        })

    def test_each_game_contributes_one_home_and_one_away_observation(self):
        rows, totals = build_site._market_calibration_rows(self._frame())
        self.assertEqual(totals["home"]["n"], totals["away"]["n"])
        self.assertEqual(totals["all"]["n"],
                         totals["home"]["n"] + totals["away"]["n"])
        per_rung = sum(r["all"]["n"] for r in rows)
        self.assertEqual(per_rung, totals["all"]["n"],
                         "rung counts must sum to the total; a dropped price "
                         "shows up here first")

    def test_both_sides_implied_and_actual_are_exactly_one_half(self):
        """Forced by construction, so a deviation is a bug, not a result.

        Devigged probabilities sum to 1 across the two sides of a game, and
        exactly one side wins. Pooling both sides must therefore give .500 on
        both the implied and the realised column, whatever the season did.
        """
        _rows, totals = build_site._market_calibration_rows(self._frame())
        self.assertAlmostEqual(totals["all"]["implied"], 0.5, places=9)
        self.assertAlmostEqual(totals["all"]["actual"], 0.5, places=9)

    def test_rows_without_a_close_are_skipped_not_imputed(self):
        d = self._frame()
        d.loc[0, "close_p_home"] = np.nan
        _rows, totals = build_site._market_calibration_rows(d)
        self.assertEqual(totals["home"]["n"], 5)
        self.assertEqual(totals["away"]["n"], 5)

    def test_standard_error_never_collapses_on_a_one_sided_rung(self):
        """sqrt(phat(1-phat)/n) returns exactly 0.0 when a bucket goes all-W.

        Two rungs on the live page did precisely that -- a single-game bucket
        printing a 27-point miss against implied with +-0.0 beside it. The
        rate is estimated from the same outcomes the gap is being tested on,
        so it carries no information about how uncertain that gap is. The
        Poisson-binomial SE is fixed by the prices instead.
        """
        d = self._frame()
        # force every home side to win, so the pooled home bucket is all-W
        d["full_home"] = 9
        d["full_away"] = 1
        _rows, totals = build_site._market_calibration_rows(d)
        home = totals["home"]
        self.assertEqual(home["actual"], 1.0)
        self.assertGreater(home["se"], 0.0,
                           "an all-W bucket is the least certain kind there "
                           "is; it must not report the smallest error bar")
        probs = [.55, .48, .70, .42, .50, .62]
        expected = math.sqrt(sum(q * (1 - q) for q in probs)) / len(probs)
        self.assertAlmostEqual(home["se"], expected, places=12)

    def test_both_surfaces_share_one_standard_error_derivation(self):
        """The ladder and the value panel ask the same question of the same
        page, so the error bar beside each has one definition, not two."""
        probs = [.61, .44, .5]
        expected = math.sqrt(sum(q * (1 - q) for q in probs)) / len(probs)
        self.assertAlmostEqual(build_site._excess_se(probs), expected, places=12)
        self.assertAlmostEqual(build_site._excess_se(pd.Series(probs)),
                               expected, places=12)
        # defined at n=1, undefined at n=0, never zero for a live price
        self.assertAlmostEqual(build_site._excess_se([.5]), .5, places=12)
        self.assertTrue(np.isnan(build_site._excess_se([])))
        for q in (.01, .25, .5, .75, .99):
            self.assertGreater(build_site._excess_se([q] * 4), 0.0)

    def test_no_close_column_yields_no_rows_rather_than_raising(self):
        rows, totals = build_site._market_calibration_rows(
            pd.DataFrame({"full_home": [1], "full_away": [2]}))
        self.assertEqual(rows, [])
        self.assertEqual(totals, {})


if __name__ == "__main__":
    unittest.main()


class ReliefPoolPriorTests(unittest.TestCase):
    """v3's shrink target: the relief pool's own UNWEIGHTED centre.

    The three failure modes are all silent -- a wrong pool, a weighted mean, or
    a frozen literal would each produce a plausible number and shift every
    bullpen rate by ~0.01 without erroring.
    """

    def _roles(self, n=60):
        r = {i: {"appearances": 50, "starts": 0, "start_share": 0.0,
                 "avg_ip_per_appearance": 1.0, "batters_faced": 150.0}
             for i in range(1, n)}
        r[99] = {"appearances": 28, "starts": 28, "start_share": 1.0,
                 "avg_ip_per_appearance": 6.0, "batters_faced": 700.0}
        return r

    def _stat(self, n=60):
        s = {i: {"xwOBA": 0.290 + 0.001 * i} for i in range(1, n)}
        s[99] = {"xwOBA": 0.900}          # rotation arm, must never enter
        return s

    def test_prior_is_the_unweighted_mean_over_the_shipped_filter(self):
        build_site._pitcher_roster_cache[1] = list(range(1, 60)) + [99]
        got = build_site.relief_pool_prior([1], self._stat(), {1: self._roles()})
        expect = sum(0.290 + 0.001 * i for i in range(1, 60)) / 59
        self.assertAlmostEqual(got, expect, places=12)

    def test_usage_weighting_is_not_used(self):
        """EB wants the population centre, and the pool's usage-weighted centre
        sits ~0.012 the other side of it because the good arms get the innings.
        A BF-weighted mean here would over-correct past the shipped target."""
        roles = self._roles()
        for i in range(1, 30):            # give the best arms all the usage
            roles[i]["batters_faced"] = 900.0
        build_site._pitcher_roster_cache[2] = list(range(1, 60))
        got = build_site.relief_pool_prior([2], self._stat(), {2: roles})
        expect = sum(0.290 + 0.001 * i for i in range(1, 60)) / 59
        self.assertAlmostEqual(got, expect, places=12)

    def test_rotation_arms_cannot_reach_the_prior(self):
        build_site._pitcher_roster_cache[3] = [99]
        self.assertIsNone(
            build_site.relief_pool_prior([3], self._stat(), {3: self._roles()}))

    def test_too_few_arms_returns_none_so_the_caller_keeps_the_league_baseline(self):
        """A centre off a handful of pitchers is worse than the target it
        replaces, so v3 declines rather than shrinking toward noise."""
        build_site._pitcher_roster_cache[4] = list(range(1, 10))
        roles = {i: self._roles()[i] for i in range(1, 10)}
        self.assertIsNone(
            build_site.relief_pool_prior([4], self._stat(), {4: roles}))

    def test_the_prior_is_not_a_literal(self):
        """Derived per build. If it were frozen, changing the inputs could not
        move it -- which is the constants-frozen-from-data trap."""
        build_site._pitcher_roster_cache[5] = list(range(1, 60))
        base = build_site.relief_pool_prior([5], self._stat(), {5: self._roles()})
        hotter = {i: {"xwOBA": v["xwOBA"] + 0.050}
                  for i, v in self._stat().items()}
        moved = build_site.relief_pool_prior([5], hotter, {5: self._roles()})
        self.assertAlmostEqual(moved - base, 0.050, places=12)


class StarterRateBasisTests(unittest.TestCase):
    """A defaulted starter rate has to be distinguishable from a measured one.

    A starter missing from the leaderboard arrives at `build_matchup` with a
    NaN rate and no BF; `_shrink_one` then returns the prior, so the card and
    the ledger carry a plausible number that contains no observation of that
    arm. Under wOBA v4 the prior is his own regressed history, which makes the
    defaulted value indistinguishable by eye from a measured one. These assert
    the instrumentation that separates the two -- not any change to the lean.
    """

    LEAGUE = {"xwOBA": 0.3163}

    def _frames(self, rate, pa):
        col = build_site.XWOBA_SHRINK_COL
        P = pd.DataFrame([{
            "game_pk": 1, "Name": "Arm", "game_date": "2026-08-06",
            "game_datetime_utc": "2026-08-06T23:05:00Z", "matchup": "AAA @ BBB",
            "away_team": "AAA", "home_team": "BBB", "player_id": 99,
            "PA": pa, col: rate,
        }])
        agg = pd.DataFrame([{
            "game_pk": 1, "faced_pitcher": "Arm", "pitcher_side": "away",
            "n_opp_hitters": 9, f"opp_{col}": 0.320,
            "opp_xwOBA_neutral": 0.318, "opp_xwOBA_vs_sp": 0.320,
            "platoon_delta_sp": 0.002,
        }])
        return P, agg

    def _row(self, rate, pa):
        col = build_site.XWOBA_SHRINK_COL
        P, agg = self._frames(rate, pa)
        out = build_site.build_matchup(P, agg, [col], self.LEAGUE,
                                       shrink_prior=self.LEAGUE[col],
                                       shrink_k=build_site.XWOBA_SHRINK_K)
        self.assertEqual(len(out), 1)
        return out.iloc[0]

    def test_a_measured_arm_is_labelled_measured_with_its_bf(self):
        r = self._row(0.290, 400)
        self.assertEqual(r["starter_rate_basis"], "measured")
        self.assertEqual(r["starter_rate_bf"], 400.0)

    def test_a_missing_arm_is_labelled_prior_only(self):
        r = self._row(np.nan, np.nan)
        self.assertEqual(r["starter_rate_basis"], "prior_only")
        self.assertEqual(r["starter_rate_bf"], 0.0)

    def test_the_published_rate_alone_cannot_reveal_the_default(self):
        """Why the flag is needed: the defaulted value is exactly the prior,
        a number the model publishes for measured arms too."""
        missing = self._row(np.nan, np.nan)
        self.assertAlmostEqual(float(missing["pit_xwOBA"]),
                               self.LEAGUE["xwOBA"], places=12)

    def test_flagging_does_not_move_the_lean(self):
        """Instrumentation only -- no MODEL_TAG implication. The edge a
        measured arm produces is unchanged by the two new keys, which is the
        claim that keeps this out of prediction math."""
        r = self._row(0.290, 400)
        shrunk = build_site._shrink_one(
            0.290, 400, build_site.player_prior_one(99, self.LEAGUE["xwOBA"]),
            build_site.XWOBA_SHRINK_K)
        self.assertAlmostEqual(float(r["pit_xwOBA"]), shrunk, places=12)
        self.assertAlmostEqual(
            float(r["edge_xwOBA_sp"]),
            build_site.matchup_value(shrunk, 0.320, build_site.XWOBA_SHRINK_COL,
                                     self.LEAGUE["xwOBA"]) - self.LEAGUE["xwOBA"],
            places=12)

    def test_prior_only_arm_is_marked_on_the_card(self):
        """The whole point of the flag: a defaulted rate is visible as one, and
        a measured arm carries no badge -- one that always showed would say
        nothing."""
        side = dict(
            t="R", pl_fl={}, R=5, L=4, S=0, has_pl=False, padv=0,
            era_l5=0.0, era_l5_gs=1, era_season=3.65, is_opener=False,
            pit_xw=.311, pit_k=24.0, pit_bb=10.0, pit_hh=35.0,
            pl_sp=None, pl_sp_raw=None, pl_edge=None, pl_reliable=False,
            xw_edge=-.009, p="Arm", opp_abbr="TB", lu_status="posted",
            opp_xw=None, pl_mx=None, hitters=[],
            pitching_basis="starter_bullpen_sequential", expected_sp_ip=5.4,
            sp_rate_basis="prior_only", sp_rate_bf=0.0,
        )
        lg = {"ERA": 4.20, "xwOBA": .317, "K%": 22.0, "Hard Hit%": 39.0,
              "OPS": .720}
        self.assertIn("prior only", build_site._side_html("HOME", side, lg))
        side.update(sp_rate_basis="measured", sp_rate_bf=420.0)
        self.assertNotIn("prior only", build_site._side_html("HOME", side, lg))

    def test_both_fields_persist_and_refresh_in_the_ledger(self):
        """A field the pregame refresh cannot update would freeze at the first
        snapshot of the day, which is how a scratched starter would keep the
        replaced arm's label."""
        for c in ("sp_rate_basis_away", "sp_rate_basis_home",
                  "sp_rate_bf_away", "sp_rate_bf_home"):
            self.assertIn(c, grade_leans.AUDIT_COLS)
            self.assertIn(c, grade_leans.MODEL_FIELDS)


class StarterAbstentionTests(unittest.TestCase):
    """wOBA v5: a side whose starter has no measured rate publishes no lean.

    The rate itself is unchanged -- `starter_xwOBA` still carries the prior the
    model used, because blanking it would hide the abstention's own cause. What
    goes undefined is everything the decision rests on, through the same
    NaN-edge path a missing bullpen already takes.
    """

    L = {"xwOBA": 0.3163}

    def _matchup(self, basis):
        return pd.DataFrame([{
            "game_pk": 1, "side": "away", "pitcher": "Arm",
            "starter_xwOBA": 0.305, "opp_xwOBA_vs_sp": 0.325,
            "opp_xwOBA_neutral": 0.315, "starter_rate_basis": basis,
            "starter_rate_bf": 0.0 if basis == "prior_only" else 400.0,
        }])

    _PLAN = {(1, "away"): {
        "bullpen_xwOBA": 0.330, "expected_sp_ip": 6.0, "bullpen_pitchers": 7,
        "bullpen_relief_bf": 900, "sp_bf_per_ip": 4.3, "bp_bf_per_ip": 4.2,
        "pitching_basis": "starter_bullpen_sequential", "opener": False,
    }}

    def _apply(self, basis):
        return build_site.apply_pitching_plans(
            self._matchup(basis), self._PLAN, self.L).iloc[0]

    def test_a_measured_starter_still_produces_an_edge(self):
        r = self._apply("measured")
        self.assertTrue(pd.notna(r["edge_xwOBA"]))
        self.assertEqual(r["pitching_basis"], "starter_bullpen_sequential")

    def test_an_unmeasured_starter_suppresses_the_lean(self):
        r = self._apply("prior_only")
        for c in ("edge_xwOBA", "mx_xwOBA", "edge_xwOBA_sp", "mx_xwOBA_sp",
                  "pit_xwOBA"):
            self.assertTrue(pd.isna(r[c]), c)
        self.assertEqual(r["pitching_basis"], "starter_unmeasured_no_lean")

    def test_the_abstention_keeps_the_rate_it_abstained_on(self):
        """Auditability: the ledger has to show what was defaulted, not just
        that something was."""
        r = self._apply("prior_only")
        self.assertEqual(r["starter_xwOBA"], 0.305)
        self.assertEqual(r["starter_rate_basis"], "prior_only")

    def test_the_bullpen_phase_survives_the_abstention(self):
        """Only the starter half is unmeasured. Nulling the bullpen phase too
        would lose a measurement the model did make."""
        r = self._apply("prior_only")
        self.assertTrue(pd.notna(r["mx_xwOBA_bp"]))
        self.assertTrue(pd.notna(r["edge_xwOBA_bp"]))

    def test_a_suppressed_edge_becomes_no_ledger_lean(self):
        """End of the chain: a NaN edge on one side leaves the game with no
        xw_net and therefore no lean to grade."""
        self.assertIsNone(grade_leans._fx(float("nan")))
        r = self._apply("prior_only")
        self.assertIsNone(grade_leans._fx(r["edge_xwOBA"]))

    def test_the_card_says_why_there_is_no_lean(self):
        side = dict(
            t="R", pl_fl={}, R=5, L=4, S=0, has_pl=False, padv=0,
            era_l5=0.0, era_l5_gs=1, era_season=3.65, is_opener=False,
            pit_xw=None, pit_k=24.0, pit_bb=10.0, pit_hh=35.0,
            pl_sp=None, pl_sp_raw=None, pl_edge=None, pl_reliable=False,
            xw_edge=None, p="Arm", opp_abbr="TB", lu_status="posted",
            opp_xw=None, pl_mx=None, hitters=[], expected_sp_ip=5.4,
            pitching_basis="starter_unmeasured_no_lean",
            sp_rate_basis="prior_only", sp_rate_bf=0.0,
        )
        html = build_site._side_html("HOME", side, {"ERA": 4.2, "xwOBA": .317,
                                                    "K%": 22.0, "OPS": .720})
        self.assertIn("starter unmeasured; no lean", html)
        self.assertIn("prior only", html)

    def test_v5_isolates_the_record_and_shares_v4s_scale(self):
        """Both questions, decided explicitly -- silence defaults to the wrong
        answer about half the time."""
        v4, v5 = "woba+plat_consol_v4", "woba+plat_consol_v5"
        self.assertEqual(build_site._RECORD_FAMILIES[v5], (v5,))
        self.assertEqual(build_site._SCALE_FAMILIES[v5], (v4, v5))
        self.assertEqual(build_site._SCALE_FAMILIES[v4], (v4, v5))
        self.assertEqual(grade_leans._RECORD_FAMILIES[v5], (v5,))


class AbstentionReportingTests(unittest.TestCase):
    """An abstained game is graded but undecided. Every count that mixes the
    two has to say which it is, or the record silently loses a game."""

    def _led(self):
        return pd.DataFrame([
            {"status": "graded", "xw_lean": "NYM", "xw_full": "W",
             "xw_f5": "W", "model_tag": "woba+plat_consol_v5"},
            {"status": "graded", "xw_lean": "TB", "xw_full": "L",
             "xw_f5": "L", "model_tag": "woba+plat_consol_v5"},
            {"status": "graded", "xw_lean": np.nan, "xw_full": np.nan,
             "xw_f5": np.nan, "model_tag": "woba+plat_consol_v5"},
        ])

    def test_abstentions_are_counted_from_the_lean_not_by_subtraction(self):
        led = self._led()
        self.assertEqual(grade_leans._abstained(led), 1)
        # A tie is not an abstention: it has a lean and a T grade.
        tie = pd.DataFrame([{"status": "graded", "xw_lean": "NYM",
                             "xw_full": np.nan, "xw_f5": "T",
                             "model_tag": "woba+plat_consol_v5"}])
        self.assertEqual(grade_leans._abstained(tie), 0)

    def test_the_record_line_excludes_the_abstained_game(self):
        led = self._led()
        self.assertEqual(grade_leans._rec(led["xw_full"]), "1-1  (0.500)")
        self.assertEqual(len(led), 3)

    def test_the_grades_page_names_the_abstention(self):
        r = pd.Series({
            "status": "graded", "away": "MIA", "home": "NYM",
            "away_sp": "A", "home_sp": "B", "xw_lean": np.nan,
            "xw_delta": np.nan, "full_away": 3, "full_home": 5,
            "xw_full": np.nan, "pitching_basis_away": "starter_unmeasured_no_lean",
            "pitching_basis_home": "starter_bullpen_sequential",
        })
        html = build_site._grades_row(r)
        self.assertIn("no lean", html)


class LeanMarketValueTests(unittest.TestCase):
    """Invariants of the model x market value panel on market-calibration.html.

    Structural only, in the same spirit as MarketCalibrationTests: a failure
    here means the panel is built wrong, not that the season went differently.
    No expected record, ROI or threshold is frozen into this class.
    """

    @staticmethod
    def _frame(n=8, tag=None, lean_home=True, won=True, ml=-150, p_home=.60):
        """Graded current-family rows, all leaning the same way by default."""
        tag = build_site.MODEL_TAG if tag is None else tag
        return pd.DataFrame({
            "status": ["graded"] * n,
            "model_tag": [tag] * n,
            "home": ["HOU"] * n,
            "away": ["SEA"] * n,
            "xw_lean": (["HOU"] if lean_home else ["SEA"]) * n,
            "xw_full": (["W"] if won else ["L"]) * n,
            "xw_delta": np.linspace(.005, .045, n),
            "close_p_home": [p_home] * n,
            "close_home_ml": [ml] * n,
            "close_away_ml": [-ml if ml < 0 else -ml] * n,
        })

    def test_american_unit_profit_pays_the_right_price(self):
        """Including both even-money forms and the invalid band between them."""
        self.assertAlmostEqual(build_site._american_unit_profit(-150, True), 2 / 3)
        self.assertAlmostEqual(build_site._american_unit_profit(+150, True), 1.5)
        self.assertEqual(build_site._american_unit_profit(-150, False), -1.0)
        self.assertEqual(build_site._american_unit_profit(+150, False), -1.0)
        # +/-100 are both even money and must pay, not fall in the dead band
        self.assertAlmostEqual(build_site._american_unit_profit(+100, True), 1.0)
        self.assertAlmostEqual(build_site._american_unit_profit(-100, True), 1.0)
        for bad in (0, 50, -50, 99, -99, None, np.nan):
            self.assertTrue(np.isnan(build_site._american_unit_profit(bad, True)),
                            f"{bad!r} is not an American price and must not pay")

    def test_standard_error_does_not_collapse_on_a_one_sided_bucket(self):
        """The regression this panel's SE form exists for.

        A bucket that goes all-W or all-L has no outcome variance, so the
        sample sd of the residuals -- and sqrt(phat(1-phat)/n) -- both go to
        ~0 and report the least certain bucket on the page as the most
        certain. The Poisson-binomial SE is fixed by the market prices, so it
        cannot degenerate.
        """
        obs = build_site._lean_market_observations(self._frame(n=4, won=False))
        self.assertEqual(len(obs), 4)
        parts = build_site._lean_market_agg(obs, pd.Series(True, index=obs.index))
        self.assertEqual((parts["w"], parts["l"]), (0, 4))
        expected = math.sqrt((obs["market_p"] * (1 - obs["market_p"])).sum()) / 4
        self.assertAlmostEqual(parts["excess_se"], expected, places=12)
        self.assertGreater(parts["excess_se"], .2,
                           "a 4-row all-loss bucket at even-ish prices cannot "
                           "carry a small standard error")
        # and it must be far larger than either estimator it replaced, both
        # of which collapse toward zero exactly here
        resid = obs["won"].to_numpy(float) - obs["market_p"].to_numpy(float)
        sd_form = float(np.std(resid, ddof=1)) / 2.0        # sd(resid)/sqrt(n)
        binom_form = math.sqrt(max(0.0 * (1 - 0.0), 0.0) / 4)  # sqrt(phat(1-phat)/n)
        self.assertEqual(binom_form, 0.0)
        self.assertGreater(parts["excess_se"], 10 * sd_form)

    def test_observation_frame_carries_no_unrendered_column(self):
        """price_dislocation was computed, returned and rendered nowhere.

        Not a style point: the note on the page described it as a diagnostic
        the reader could see, because nothing tied the prose to what the
        tables actually emit. Every column here must reach a surface.
        """
        obs = build_site._lean_market_observations(self._frame(n=6))
        self.assertNotIn("price_dislocation", obs.columns)
        a = build_site._lean_market_value_analysis(self._frame(n=6))
        self.assertNotIn("price_dislocation", a["obs"].columns)

    def test_degenerate_spread_yields_nan_not_a_numpy_warning(self):
        """A frame with one distinct price has no correlation to report.

        corrcoef divides by the sd of each axis, so a constant column returns
        nan from a divide-by-zero rather than raising. The page renders that
        as an em dash, but the arithmetic must not be attempted at all --
        which is what the slope beside it already did.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a = build_site._lean_market_value_analysis(self._frame(n=6))
        self.assertTrue(np.isnan(a["corr"]))
        self.assertTrue(np.isnan(a["slope"]))
        html = build_site._render_lean_market_value_panel(self._frame(n=6))
        self.assertIn("—", html)

    def test_standard_error_is_defined_for_a_single_row(self):
        obs = build_site._lean_market_observations(self._frame(n=1))
        parts = build_site._lean_market_agg(obs, pd.Series(True, index=obs.index))
        self.assertTrue(np.isfinite(parts["excess_se"]))

    def test_abstained_rows_never_enter_the_panel(self):
        """v5 can grade a row with no lean; a price panel must not score it.

        Same rule as _baseline_controls: the model is scored on the decided
        rows, so anything sitting beside it must be too.
        """
        d = self._frame(n=6)
        d.loc[0, ["xw_lean", "xw_full", "xw_delta"]] = np.nan
        obs = build_site._lean_market_observations(d)
        self.assertEqual(len(obs), 5)

    def test_only_the_current_record_family_is_scored(self):
        """Pooling old prediction math would answer a different question."""
        d = pd.concat([self._frame(n=4),
                       self._frame(n=4, tag="xw+plat_consol_v2")],
                      ignore_index=True)
        obs = build_site._lean_market_observations(d)
        self.assertEqual(len(obs), 4)

    def test_market_p_follows_the_lean_not_the_home_side(self):
        """market_p is the price of the LEANED team, whichever side that is."""
        home = build_site._lean_market_observations(
            self._frame(n=3, lean_home=True, p_home=.62))
        away = build_site._lean_market_observations(
            self._frame(n=3, lean_home=False, p_home=.62))
        self.assertTrue(np.allclose(home["market_p"], .62))
        self.assertTrue(np.allclose(away["market_p"], .38))

    def test_ungraded_and_unpriced_rows_are_skipped_not_imputed(self):
        d = self._frame(n=6)
        d.loc[0, "close_p_home"] = np.nan
        d.loc[1, "status"] = "pending"
        self.assertEqual(len(build_site._lean_market_observations(d)), 4)

    def test_panel_renders_without_rows_rather_than_raising(self):
        """A MODEL_TAG bump empties this panel; it must say so, not crash."""
        for d in (None, pd.DataFrame(),
                  self._frame(n=4, tag="xw+plat_consol_v0")):
            self.assertEqual(build_site._lean_market_value_analysis(d), {})
            html = build_site._render_lean_market_value_panel(d)
            self.assertIn("Model × market value", html)
            self.assertIn("once graded leans", html)

    def test_every_value_table_row_emits_four_cells(self):
        """Including the empty buckets, which render an em dash, not nothing."""
        rows = [("filled", {"n": 3, "w": 2, "l": 1, "implied": .5, "actual": .667,
                            "excess": .167, "excess_se": .28, "roi": .1,
                            "units": .3}),
                ("empty", None)]
        html = build_site._lean_market_value_table(rows)
        self.assertEqual(html.count("<tr class='gr-row'>"), 2)
        self.assertEqual(html.count("<td "), 8)
