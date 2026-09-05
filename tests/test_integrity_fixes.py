import math
import inspect
import os
import pathlib
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

    def test_hybrid_selection_and_two_sided_market_are_snapshotted(self):
        xw = _dump_rows(
            321, "2026-07-27", "2026-07-27T15:00:00+00:00",
            "2026-07-27T22:00:00+00:00")
        stamp = "2026-07-27T15:01:00+00:00"
        build_site.attach_hybrid_snapshot(
            xw, {321: {"away_ml": -140, "home_ml": 125, "p_home": .30}},
            stamp)
        for col in ("selection_rule_tag", "pregame_market_utc",
                    "pregame_away_ml", "pregame_home_ml", "pregame_p_home",
                    "hybrid_action", "hybrid_selection", "hybrid_p",
                    "hybrid_ml"):
            self.assertEqual(xw[col].nunique(dropna=False), 1, col)
        self.assertEqual(xw.iloc[0]["selection_rule_tag"],
                         build_site.HYBRID_RULE_TAG)
        self.assertEqual(xw.iloc[0]["pregame_market_utc"], stamp)
        self.assertEqual(xw.iloc[0]["hybrid_action"], "FADE")
        self.assertEqual(xw.iloc[0]["hybrid_selection"], "AWA")
        self.assertAlmostEqual(xw.iloc[0]["hybrid_p"], .70)
        self.assertEqual(xw.iloc[0]["hybrid_ml"], -140)

        row = grade_leans.rows_from_dump(xw, None)[0]
        self.assertEqual(row["hybrid_selection"], "AWA")
        self.assertEqual(row["pregame_away_ml"], -140)
        self.assertIsNone(row["hybrid_full"])

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


# Any schedule that is not the grading cron takes the pregame path. Named
# rather than pasted so changing the real cron is not mistaken for a test
# change -- test_window_outlasts_the_poll_interval reads build.yml for that.
PREGAME_POLL = "* * * * *"


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
            "schedule", PREGAME_POLL, self.NOW,
            games=[self.game(101, 30)],
        )
        self.assertTrue(run)
        self.assertEqual(day, "2026-07-17")
        self.assertIn("101", reason)

    def test_scheduled_poll_skips_outside_window_and_final_games(self):
        games = [self.game(101, 10), self.game(102, 30, state="Final")]
        run, _, reason = schedule_gate.decision(
            "schedule", PREGAME_POLL, self.NOW, games=games,
        )
        self.assertFalse(run)
        self.assertIn("no game", reason)

    def test_pregame_window_covers_the_slate_and_excludes_the_edges(self):
        """T-60 and T-300 build; T-10 (too late) and T-400 (too early) do not."""
        for minutes, expected in ((60, True), (300, True),
                                  (10, False), (400, False)):
            run, _, _ = schedule_gate.decision(
                "schedule", PREGAME_POLL, self.NOW,
                games=[self.game(101, minutes)],
            )
            self.assertEqual(run, expected, f"T-{minutes}")

    def test_window_outlasts_the_poll_interval(self):
        """The window must outlast the gap between polls, read from build.yml.

        These two numbers are one decision in two files. If the cron fires
        every N minutes, a game whose whole pregame window falls between two
        consecutive polls is never sampled and its rows are lost -- and
        no-lookahead means they cannot be rebuilt. So the window is pinned
        against the real cron rather than against a literal here.

        The margin is deliberately large. GitHub does not merely delay
        scheduled runs, it drops them: 56 requested polls on 2026-08-27
        returned 2, with a 3h53m gap between deliveries. Covering the nominal
        interval is the floor; the window is sized for the observed gaps.
        """
        workflow = pathlib.Path(".github/workflows/build.yml").read_text()
        crons = re.findall(r"cron:\s*'([^']+)'", workflow)
        pregame = [c for c in crons if c != schedule_gate.DAILY_GRADE_CRON]
        self.assertEqual(len(pregame), 1, f"expected one pregame cron, got {pregame}")
        minute_field = pregame[0].split()[0]
        if minute_field.startswith("*/"):
            fires_per_hour = 60 // int(minute_field[2:])
        else:
            fires_per_hour = len(minute_field.split(","))
        interval = 60 / fires_per_hour
        width = schedule_gate.MAX_MINUTES_BEFORE - schedule_gate.MIN_MINUTES_BEFORE
        self.assertGreaterEqual(
            width, interval,
            f"cron fires every {interval:.0f}min but the window is only "
            f"{width:.0f}min wide; a game can pass through unsampled")
        # and the margin for dropped polls, not just the nominal interval
        self.assertGreaterEqual(
            width, 4 * interval,
            "window should carry several polls' worth of margin, because "
            "GitHub drops scheduled runs rather than merely delaying them")

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
        self.assertIn("<b>XWOBA SIDE</b> at 45% or higher", page)

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
                    close_home_ml=-140, close_away_ml=120,
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

    # Assertions run against the header so the base64 font blob cannot match
    # short digit strings and create a false positive.
    def _pages(self, led):
        with mock.patch.object(build_site, "load_ledger_df", return_value=led):
            page = build_site.render_grades_html("test build")
            return {"strip": build_site.records_strip_html(),
                    "grades page": page.split("<div class='gr-tablewrap'>")[0]}

    def _marks(self, rec):
        # The record itself plus the opening paren of its rate, NOT the strip's
        # surrounding copy. The marker used to be `full {rec}`, which pinned a
        # word that is not the claim under test -- the claim is which ROWS the
        # number covers -- so a wording change failed this for a reason with
        # nothing to do with family scoping.
        return {"strip": f"{rec} (", "grades page": f">{rec}<"}

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

    def test_the_strip_and_the_grades_page_headline_the_same_record(self):
        """One click apart, so they cannot be allowed to disagree.

        The strip used to headline the raw lean while the grades page it links
        to headlined the published rule's selection -- 139-84 against 146-77 on
        the same rows. That is the internal-vs-public artifacts defect with
        both artifacts public. Both now read the same aggregate.
        """
        led = build_site.load_ledger_df()
        if led is None:
            self.skipTest("ledger unavailable")
        obs = build_site._lean_market_observations(led)
        if obs.empty:
            self.skipTest("no priced current-family rows")
        rule = build_site._lean_market_agg(
            obs, obs["won"].notna(), won="hybrid_won", p="hybrid_p",
            resid="hybrid_resid", profit="hybrid_profit")
        rec = f"{rule['w']}-{rule['l']}"
        strip = build_site.records_strip_html()
        header = build_site.render_grades_html("test build").split(
            "<div class='gr-tablewrap'>")[0]
        self.assertIn(f"{rec} (", strip)
        self.assertIn(f">{rec}<", header)
        # And neither may be showing the unmodified lean instead.
        lean = build_site._lean_market_agg(obs, obs["won"].notna())
        lean_rec = f"{lean['w']}-{lean['l']}"
        if lean_rec != rec:
            self.assertNotIn(f"{lean_rec} (", strip)

    def test_the_scope_of_the_number_is_stated_next_to_it(self):
        """The strip states its ledger scope; the public table is v12-only."""
        pages = self._pages(self._mixed())
        self.assertIn("1 of 4 graded rows", pages["strip"])
        self.assertIn(build_site.MODEL_TAG, pages["strip"])
        self.assertNotIn("1 of 4 graded rows", pages["grades page"])
        self.assertIn("V12 selections and results", pages["grades page"])

    def test_public_record_surfaces_use_plain_current_labels(self):
        for name, html in self._pages(self._mixed()).items():
            visible = re.sub(r"<(?:style|script)\b.*?</(?:style|script)>", "",
                             html, flags=re.S | re.I)
            self.assertNotRegex(visible.lower(), r"\bretro(?:spective)?\b", name)

    def test_no_scope_note_when_the_family_is_every_graded_row(self):
        """The note has to disappear on its own, or it becomes noise that a
        reader learns to skip."""
        cur = build_site.MODEL_TAG
        led = pd.DataFrame([self._row(1, cur, "B", "W", 2, 4)])
        strip = self._pages(led)["strip"]
        self.assertIn("1-0 (", strip)
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
        # The compact strip may explain that the internal ledger has older
        # rows, but the public archive intentionally does not display them.
        pages = self._pages(led)
        self.assertIn("2 graded", pages["strip"].replace(
            "; 2 rows graded", "; 2 graded"))
        self.assertNotIn("xw+plat_consol_v2", pages["grades page"])

class GradesHeaderClaimsTests(unittest.TestCase):
    """What the ledger page says about the numbers it publishes.

    Structural only: no record, rate or count is frozen here. Each test pins a
    CLAIM the page has to make, not a value it has to print.
    """

    @staticmethod
    def _row(pk, **kw):
        row = dict(game_pk=pk, game_date="2026-08-07", away="A", home="B",
                   away_sp="P1", home_sp="P2", status="graded",
                   model_tag=build_site.MODEL_TAG, model_metric="xwOBA",
                   xw_lean="B", xw_delta=.01, xw_full="W", xw_f5="W",
                   full_away=2, full_home=4, close_p_home=.6,
                   close_home_ml=-140, close_away_ml=120,
                   lock_status="pregame")
        row.update(kw)
        return row

    def _header(self, led):
        with mock.patch.object(build_site, "load_ledger_df", return_value=led):
            page = build_site.render_grades_html("test build")
        return page.split("<div class='gr-tablewrap'>")[0]

    def test_the_header_says_its_own_figures_are_a_discovery_result(self):
        """This page leads with a z-score for a rule fitted on these rows.

        The calibration panel says so in its lead and the per-game card stamps
        "not a forward test" on every branch history. The page publishing the
        largest version of the number said nothing at all, so a reader met
        `z +2.86` with no way to know the threshold above it was chosen on the
        same games it is scored over.
        """
        head = self._header(pd.DataFrame([self._row(1), self._row(2, xw_full="L",
                                                    full_away=5, full_home=3)]))
        self.assertIn("Discovery", head)
        self.assertIn("not a forward test", head)
        self.assertIn("data/ledger_report.txt", head,
                      "the caveat must point at the registered forward test")
        self.assertIn(f"{100 * build_site.HYBRID_THRESHOLD:.0f}%", head)

    def test_the_ml_column_says_which_price_it_shows(self):
        """One heading, two bases, and every aggregate scored at the close.

        A row carrying a locked pregame snapshot renders that price; every
        other row renders the close. The gap is small and flips no branch on
        the committed ledger, but a reader reconciling a row's price against
        the unit figures in the header cannot see which basis they are on.
        """
        head = self._header(pd.DataFrame([self._row(1), self._row(2)]))
        self.assertIn("locked pregame", head)
        self.assertIn("scored at the close", head)

    def test_a_decided_row_the_record_cannot_score_is_named_not_subtracted(self):
        """The header's tiles are scored on a stricter set than "graded".

        Today the only excluded rows are abstentions, so 284 graded minus 7
        abstained happens to land exactly on the record's 277 and the page
        reconciles by arithmetic the reader has to do. A decided row with no
        close moves that denominator with nothing saying so -- the same defect
        as a control whose n is never stated. Each excluded row is counted
        from its own columns, never by subtracting one denominator from
        another.
        """
        led = pd.DataFrame([
            self._row(1),
            self._row(2, xw_full="L", full_away=5, full_home=3),
            self._row(3, close_p_home=np.nan, close_home_ml=np.nan,
                      close_away_ml=np.nan),
        ])
        head = self._header(led)
        self.assertIn("2 scored", head)
        self.assertIn("1 unpriced", head)

    def test_the_empty_family_message_claims_no_rows_the_table_omits(self):
        """The clause deleted here offered to count rows "listed below".

        It could not fire -- its count came from the already family-filtered
        frame, so it was zero exactly when the branch ran -- and had it fired
        it would have pointed at rows the table filters out.
        """
        old = self._row(1, model_tag="xw+plat_consol_v2")
        with mock.patch.object(build_site, "load_ledger_df",
                               return_value=pd.DataFrame([old])):
            page = build_site.render_grades_html("test build")
        self.assertIn(f"No graded games yet under {build_site.MODEL_TAG}", page)
        self.assertNotIn("listed below", page)
        self.assertNotIn("xw+plat_consol_v2", page)

    def test_the_observation_frame_keeps_its_ledger_row_labels(self):
        """Membership is what lets a surface NAME the rows it dropped.

        With a fresh RangeIndex the only way to size the excluded set is to
        subtract one denominator from another, which produces a count no label
        can honestly be attached to.
        """
        led = pd.DataFrame([self._row(1), self._row(2, xw_full="L",
                                                   full_away=5, full_home=3)],
                           index=[41, 57])
        obs = build_site._lean_market_observations(led)
        self.assertEqual(list(obs.index), [41, 57])


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

    # MODEL_TAG so the row survives the header's RECORD_TAGS filter; the page
    # needs a non-empty family to render the summary block that carries the
    # note.
    @staticmethod
    def _page_row(pk, lock):
        return dict(game_pk=pk, game_date="2026-07-20", away="A", home="B",
                    away_sp="P1", home_sp="P2", status="graded",
                    model_tag=build_site.MODEL_TAG, xw_lean="B", xw_delta=.01,
                    xw_full="W", xw_f5="W", full_away=2, full_home=4,
                    lock_status=lock)

    def _page(self, *locks):
        led = pd.DataFrame([self._page_row(i, l)
                            for i, l in enumerate(locks, 1)])
        with mock.patch.object(build_site, "load_ledger_df", return_value=led):
            return build_site.render_grades_html("test build")

    def test_the_page_reports_the_lock_split_never_asserts_the_whole(self):
        """The claim came back; the verbose block it replaces did not.

        This test used to assert the page said NOTHING about locking, pinning
        a V12-redesign decision that left `_lock_provenance` with three tests
        and no caller while the public ledger made no provenance claim at all.
        What the redesign was right to remove is the three-clause block. The
        claim itself belongs on this page, because the no-lookahead invariant
        is the reason the ledger is worth reading -- and the one thing it may
        never do is assert coverage it cannot substantiate, which is what the
        split is for.
        """
        page = self._page("late_snapshot")
        self.assertIn("0 of 1 rows carry", page)
        self.assertIn("snapshotted after first pitch", page)
        self.assertNotIn("All 1 rows", page)

    def test_a_fully_locked_page_says_so_once_and_inline(self):
        """Compact is the property the old test was really protecting.

        One sentence inside the header note, never a section of its own -- so
        a future edit that grows it back into a block fails here rather than
        being noticed on the live page.
        """
        page = self._page("pregame", "pregame_recovered")
        self.assertIn("All 2 rows carry", page)
        self.assertEqual(page.count("pregame lock"), 1,
                         "the lock claim is made once, not restated")
        note = re.search(r"<div class='gr-note'>(.*?)</div>", page, re.S)
        self.assertIsNotNone(note)
        self.assertIn("pregame lock", note.group(1),
                      "the claim rides in the header note, not its own block")

    def test_the_claim_never_publishes_only_the_clean_case(self):
        """Silence on a mixed page would be the anti-pattern inverted.

        A page that states provenance when every row is verified and drops it
        when some are not publishes a claim exactly when it flatters -- worse
        than never making one.
        """
        for locks in (("pregame",), ("pregame", None),
                      (None, "late_snapshot"), ("legacy_unverified",)):
            with self.subTest(locks=locks):
                self.assertIn("pregame lock", self._page(*locks))


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
        # The grades page scores the published RULE's selection, so its tile is
        # "vs market" rather than the strip's raw-lean "vs mkt". Different
        # statistics, deliberately different labels.
        self.assertIn("vs market", pages["grades page"])

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

class WorkflowStepTimeoutTests(unittest.TestCase):
    """A step timeout must kill the process, not just the step.

    `timeout-minutes` kills the step's shell and moves on; a python child
    survives as an orphan the runner only reaps in post-job cleanup. Run
    33073467257 is the instance: a step timed out at 12:55:07, kept running,
    and rewrote a file under data/ between the commit step's `git add data/`
    and its `git pull --rebase` -- which refused with "cannot rebase: You have
    unstaged changes", so that slate's pregame rows were committed locally and
    never pushed. Any step that expects to be timed out must bound its own
    process, so nothing can write to data/ after the step returns.

    THE STEP THIS WAS WRITTEN FOR NO LONGER EXISTS. The walk-forward replay
    was removed, and with it the only step-level `timeout-minutes` in the
    repository, so this currently guards nothing live. It is kept because the
    hazard belongs to the PATTERN and not to that step: any future step that
    expects to be timed out beside a step that writes data/ reintroduces it,
    and the incident above is why. The count floor is deliberately gone -- an
    assertion that some bounded step exists would now fail for the good reason
    that none does.
    """

    WORKFLOW_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".github", "workflows",
    )

    def _workflows(self):
        """Every workflow, not just build.yml.

        The rule is about any step that expects to be timed out beside a step
        that writes data/, and the walk-forward replay -- the step the orphan
        incident happened to -- has since moved to its own workflow. Scanning
        one file would have let the invariant walk out of the suite with it.
        """
        return sorted(
            os.path.join(self.WORKFLOW_DIR, n)
            for n in os.listdir(self.WORKFLOW_DIR) if n.endswith(".yml")
        )

    def _steps(self, path):
        """Yield (name, body-lines) for each step of every job in a workflow."""
        steps, current = [], None
        for line in open(path, encoding="utf-8"):
            stripped = line.strip()
            if stripped.startswith("- ") and line.startswith(" " * 6):
                if current is not None:
                    steps.append(current)
                current = (stripped, [stripped[2:]])
            elif current is not None:
                if stripped and not line.startswith(" " * 8):
                    steps.append(current)
                    current = None
                elif stripped and not stripped.startswith("#"):
                    current[1].append(stripped)
        if current is not None:
            steps.append(current)
        return steps

    def test_every_step_level_timeout_bounds_its_own_process(self):
        found = 0
        for path in self._workflows():
            for name, body in self._steps(path):
                if not any(ln.startswith("timeout-minutes:") for ln in body):
                    continue
                found += 1
                run = [ln for ln in body if ln.startswith("run:")]
                self.assertTrue(run, f"{path}: {name} has timeout-minutes but no run:")
                self.assertIn(
                    "timeout ", run[0],
                    f"{path}: {name} relies on timeout-minutes alone; the step "
                    "dies but its process does not, and it can still write to "
                    "data/ after the commit step has staged it",
                )
        # No floor on `found`: see the class docstring. The repository
        # currently has no step-level timeout at all, and requiring one would
        # fail for the good reason that none exists rather than for a defect.


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
        per_rung = sum(r["all"]["n"] for r in rows)
        self.assertEqual(per_rung, totals["home"]["n"] + totals["away"]["n"],
                         "rung counts must sum to the total; a dropped price "
                         "shows up here first")

    def test_pooling_both_sides_is_an_identity_not_a_measurement(self):
        """Which is why no such total is returned, and none is published.

        Devigged probabilities sum to 1 across the two sides of a game and
        exactly one side wins, so a pooled both-sides bucket is .500 against
        .500 implied whatever the season did -- and it led this page for
        weeks, carrying the smallest error bar on it. The invariant is real
        and is asserted here, on the two totals that ARE rendered; what
        changed is that the page no longer prints its own arithmetic back to
        the reader as a result.
        """
        _rows, totals = build_site._market_calibration_rows(self._frame())
        h, a = totals["home"], totals["away"]
        self.assertAlmostEqual(h["implied"] + a["implied"], 1.0, places=9)
        self.assertAlmostEqual(h["actual"] + a["actual"], 1.0, places=9)
        self.assertEqual(h["se"], a["se"])
        self.assertAlmostEqual(h["diff"], -a["diff"], places=12)
        self.assertNotIn("all", totals,
                         "a pooled both-sides total is forced to .500; "
                         "computing one invites publishing it again")

    def test_the_favourite_pool_is_one_observation_per_priced_game(self):
        """The non-degenerate version of the pooled question.

        Nothing cancels here because a game contributes ONE row, not two, so
        the gap is free to be non-zero -- and it is the axis the page's own
        note names as where a favourite-longshot bias would show.
        """
        d = self._frame()
        _rows, totals = build_site._market_calibration_rows(d)
        fav = totals["favourite"]
        priced = int((pd.to_numeric(d["close_p_home"]) != .5).sum())
        self.assertEqual(fav["n"], priced,
                         "one favourite per game, not two sides")
        self.assertLess(fav["n"], totals["home"]["n"],
                        "this fixture holds a pick'em, which has no favourite")
        self.assertGreater(fav["implied"], 0.5,
                           "every observation in this pool is priced over even")

    def test_a_pickem_game_has_no_favourite_to_grade(self):
        """Both sides sit at .500, so neither is the favourite.

        Handing the tie to the home side would let an arbitrary convention
        move the realised rate of a published bucket. Dropping the game costs
        one row and decides nothing.
        """
        d = self._frame()
        d["close_p_home"] = .5
        _rows, totals = build_site._market_calibration_rows(d)
        self.assertIsNone(totals["favourite"])
        self.assertEqual(totals["home"]["n"], len(d))

    def test_the_strip_publishes_neither_a_forced_nor_a_mirrored_tile(self):
        """Three tiles, one number, one of them identically zero.

        The strip showed Both sides / Home / Away. The first cannot take any
        other value and the third is one minus the second, so a reader saw a
        single measurement three times over. Tiles are keyed by their own
        label markup rather than by the words, which also appear in the
        table's column headings.
        """
        html = build_site.render_market_calibration_html("built")
        if "once the market backfill has run" in html:
            self.skipTest("no priced rows in the committed ledger")
        self.assertIn("<div class='l'>Favourites</div>", html)
        self.assertIn("<div class='l'>Home sides</div>", html)
        for gone in ("Both sides", "Away"):
            self.assertNotIn(f"<div class='l'>{gone}</div>", html)
        # and the reader is told why, rather than left to wonder
        self.assertIn("No both-sides total", html)

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

    # Every column each frame is allowed to carry, and the surface that reads
    # it. Adding a column means adding it here, which forces the question the
    # blacklist below could not ask: what renders this?
    OBS_COLUMNS = {
        "delta":         "x-axis of the slope fit -> 'market response' tile",
        "market_p":      "decides the branch; agg implied/excess on the lean",
        "close_ml":      "input to profit, and the price the lean is scored at",
        "opp_ml":        "input to hybrid_ml / chalk / home profit on a fade",
        "won":           "agg w / actual for the 'model lean' control row",
        "market_edge":   "y-axis of the slope fit -> 'market response' tile",
        "market_resid":  "agg excess -> 'model lean' actual vs implied",
        "profit":        "agg roi / units -> 'model lean' flat ROI",
        "hybrid_follow": "masks the FOLLOW / FADE branch rows",
        "hybrid_won":    "agg w / actual for each branch",
        "hybrid_p":      "agg implied and the SE for each branch",
        "hybrid_ml":     "input to hybrid_profit; the price a selection is at",
        "hybrid_resid":  "agg excess -> branch 'actual vs implied' cell",
        "hybrid_profit": "agg roi / units -> branch 'flat close ROI' cell",
        "chalk_won":     "agg w / actual for the always-chalk control",
        "chalk_p":       "agg implied and the SE for the chalk control",
        "chalk_resid":   "agg excess -> chalk control 'actual vs implied'",
        "chalk_profit":  "agg roi / units -> chalk control 'flat close ROI'",
        "home_won":      "agg w / actual for the always-home control",
        "home_p":        "agg implied and the SE for the home control",
        "home_resid":    "agg excess -> home control 'actual vs implied'",
        "home_profit":   "agg roi / units -> home control 'flat close ROI'",
    }
    # The analysis no longer decorates the frame: the 2x3 `cell` column is
    # gone with the grid it masked, and every column the surfaces read is now
    # derived once in `_lean_market_observations`.
    ANALYSIS_ONLY_COLUMNS = {}

    def test_observation_frame_carries_no_unrendered_column(self):
        """Every column on these frames must reach a surface.

        price_dislocation was computed, returned and rendered nowhere, and the
        note on the page described it as a diagnostic the reader could see --
        because nothing tied the prose to what the tables actually emit.

        This asserts the INVARIANT, not one blacklisted name. The previous
        version only banned "price_dislocation", so it went on passing when
        the V12 verdict rewrite left `delta_bucket` behind: a column whose
        two values duplicate the prefix of `cell` and which nothing read. A
        test that memorises one instance cannot catch the next one -- the
        same defect as freezing a measured number into an assertion.
        """
        obs = build_site._lean_market_observations(self._frame(n=6))
        self.assertEqual(set(obs.columns), set(self.OBS_COLUMNS),
                         "observation frame gained/lost a column; name the "
                         "surface that reads it in OBS_COLUMNS")
        a = build_site._lean_market_value_analysis(self._frame(n=6))
        self.assertEqual(
            set(a["obs"].columns),
            set(self.OBS_COLUMNS) | set(self.ANALYSIS_ONLY_COLUMNS),
            "analysis frame gained/lost a column; name the surface that "
            "reads it in ANALYSIS_ONLY_COLUMNS")

    # The moneyline columns are the ones the analysis does not read back: they
    # are the prices each row was scored at and the inputs the profit columns
    # were derived from, kept so a row's payout stays auditable against its own
    # price. `opp_ml` in particular is what makes a faded row scoreable at all
    # -- the rule bets the other side. Stated as exemptions rather than left to
    # look like consumed columns.
    RETAINED_INPUT_COLUMNS = {"close_ml", "opp_ml", "hybrid_ml"}

    def test_every_declared_observation_column_is_actually_read(self):
        """The allowlist is a claim about consumption; hold it to that.

        Removing a consumed column must break the analysis. Without this, a
        column could be added to OBS_COLUMNS with a plausible-sounding surface
        beside it that nothing actually reads -- the allowlist becomes a rubber
        stamp and the guard above degrades back into the blacklist it replaced.
        """
        real = build_site._lean_market_observations
        for col in set(self.OBS_COLUMNS) - self.RETAINED_INPUT_COLUMNS:
            with self.subTest(column=col):
                def without(led, _c=col):
                    return real(led).drop(columns=[_c])
                with mock.patch.object(build_site,
                                       "_lean_market_observations", without):
                    with self.assertRaises(KeyError, msg=(
                            f"{col} is declared consumed in OBS_COLUMNS but "
                            "the analysis runs without it")):
                        build_site._lean_market_value_analysis(self._frame(n=6))

    def test_degenerate_spread_yields_nan_not_a_numpy_warning(self):
        """A frame with one distinct price has no slope to report.

        polyfit on a constant axis is a rank-deficient fit, so the arithmetic
        must not be attempted at all rather than left to emit a warning. The
        page renders the result as an em dash.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a = build_site._lean_market_value_analysis(self._frame(n=6))
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
            self.assertIn(build_site.PUBLIC_MODEL_NAME, html)
            self.assertIn("once graded leans", html)

    @classmethod
    def _spread_frame(cls, n=8):
        """Rows whose price and delta both vary, so a slope exists.

        The default fixture prices every game identically, which is the
        degenerate case the panel refuses to fit. Prices straddle the
        threshold so the FADE branch is populated too.
        """
        d = cls._frame(n=n)
        d["close_p_home"] = np.linspace(.36, .74, n)
        return d

    def test_the_market_response_slope_is_published_with_its_standard_error(self):
        """A bare slope is a number the reader cannot judge.

        Every other statistic on this page carries its spread -- the ladder,
        both branch tables and both controls all go through `_excess_se`, and
        this file already pins that an SE may never be estimated from the
        outcomes under test. The slope tile was the one figure printed alone.
        """
        d = self._spread_frame()
        a = build_site._lean_market_value_analysis(d)
        x = a["obs"]["delta"].to_numpy(float)
        y = a["obs"]["market_edge"].to_numpy(float)
        resid = y - (a["slope"] * x + a["intercept"])
        expected = math.sqrt(float((resid ** 2).sum()) / (len(x) - 2)
                             / float(((x - x.mean()) ** 2).sum()))
        self.assertAlmostEqual(a["slope_se"], expected, places=12)
        self.assertGreater(a["slope_se"], 0.0)
        # and it reaches the tile, in the same units as the value above it
        html = build_site._render_lean_market_value_panel(d)
        self.assertIn(f"± {a['slope_se'] * 0.010 * 100:.2f} · leaned-team p",
                      html)

    def test_a_slope_that_cannot_be_fitted_reports_no_standard_error(self):
        """One distinct price: no slope, so nothing to put a spread on."""
        a = build_site._lean_market_value_analysis(self._frame(n=6))
        self.assertTrue(np.isnan(a["slope"]))
        self.assertTrue(np.isnan(a["slope_se"]))

    def test_the_chalk_identity_is_stated_in_words_not_by_adjacency(self):
        """The FADE branch and the chalk control print the same numbers.

        Backing the other side of a lean priced under the threshold always
        lands on the favourite, so on those rows the branch IS the chalk bet.
        The per-game card already says that in words; this page printed the
        two lines four rows apart with nothing joining them, and told the
        reader it was showing "two other ways" while listing four.
        """
        d = self._spread_frame()
        a = build_site._lean_market_value_analysis(d)
        label = build_site.hybrid_public_label("FADE")
        fade = dict(a["branch_rows"])[
            f"{label} · XWOBA-side p "
            f"< {100 * build_site.HYBRID_THRESHOLD:.0f}%"]
        chalk = dict(a["control_rows"])[f"Always chalk · {label} rows only"]
        self.assertIsNotNone(fade, "fixture must populate the fade branch")
        self.assertEqual(fade, chalk,
                         "these are the same bet on the same rows; if they "
                         "differ the two were scored over different games")
        html = build_site._render_lean_market_value_panel(d)
        self.assertIn("three other ways", html)
        # The clause that says WHY the two lines match, wherever its wording
        # lands: the label comes from its one home and the identity is stated,
        # not left to adjacency.
        self.assertIn(f"must equal <b>{build_site.hybrid_public_label('FADE')}"
                      "</b> above", html)
        self.assertIn("the two are the same bet", html)

    def test_every_value_table_row_emits_four_cells(self):
        """Including the empty buckets, which render an em dash, not nothing."""
        rows = [("filled", {"n": 3, "w": 2, "l": 1, "implied": .5, "actual": .667,
                            "excess": .167, "excess_se": .28, "roi": .1,
                            "units": .3}),
                ("empty", None)]
        html = build_site._lean_market_value_table(rows)
        self.assertEqual(html.count("<tr class='gr-row'>"), 2)
        self.assertEqual(html.count("<td "), 8)


class HybridRuleTests(unittest.TestCase):
    """The published selection rule, shared by the leans and calibration pages.

    Structural only: no branch record is frozen here. The one constant that IS
    pinned is that there is exactly ONE threshold, imported from the
    registration in `hybrid_test` rather than restated -- see
    `test_the_threshold_has_exactly_one_home`.
    """

    def test_card_and_calibration_table_place_a_game_identically(self):
        """One derivation, or the card and the table will drift apart.

        The calibration page's branch rows and the per-game panel must never
        disagree about which branch a game is in -- the metric_label() lesson
        applied to a selection rule. Both go through `hybrid_action`, so this
        re-places every bucketed row and demands the same answer.
        """
        led = build_site.load_ledger_df()
        a = build_site._lean_market_value_analysis(led)
        if not a:
            self.skipTest("no current-family rows to place")
        obs = a["obs"]
        for mp, follow in zip(obs["market_p"], obs["hybrid_follow"]):
            self.assertEqual(build_site.hybrid_action(mp),
                             "FOLLOW" if follow else "FADE")

    def test_the_threshold_has_exactly_one_home(self):
        """The display rule and the registered forward test are one constant.

        Two copies of a threshold is the "one value, three homes" defect that
        stamped 14 ledger rows with v10 math under a v9 tag. If this ever
        fails, someone has written a second 0.45 and the site can now publish
        a selection the forward test would not score.
        """
        import hybrid_test
        self.assertIs(build_site.HYBRID_THRESHOLD, hybrid_test.THRESHOLD)
        src = open(build_site.__file__).read()
        # The literal may appear in prose or a docstring, but never as a
        # standalone assignment that could drift from the registration.
        self.assertNotRegex(src, r"(?m)^_?[A-Z_]*THRESHOLD[A-Z_]*\s*=\s*0\.45")

    def test_branch_boundaries_are_closed_on_the_follow_side(self):
        """Exactly at the threshold FOLLOWS. `>` would reverse a boundary game."""
        self.assertEqual(build_site.hybrid_action(.449999), "FADE")
        self.assertEqual(build_site.hybrid_action(.45), "FOLLOW")
        self.assertEqual(build_site.hybrid_action(.450001), "FOLLOW")
        self.assertEqual(build_site.hybrid_action(.99), "FOLLOW")
        self.assertEqual(build_site.hybrid_action(.01), "FADE")

    def test_daily_record_aggregates_the_hybrid_grade(self):
        day = pd.DataFrame([dict(
            model_tag=build_site.MODEL_TAG,
            selection_rule_tag=build_site.HYBRID_RULE_TAG,
            hybrid_action="FADE", hybrid_selection="A", hybrid_full="L",
            xw_lean="H", xw_full="W", away="A", home="H")])
        html = build_site._grades_day_header("2026-08-31", day, 6)
        self.assertIn("0-1", html)
        self.assertNotIn("1-0", html)

    def test_public_labels_describe_the_selected_side_not_rule_jargon(self):
        game = {"away_abbr": "A", "home_abbr": "H"}
        # The model can keep a slight underdog; calling this "follow" hid the
        # fact a user actually needs to understand.
        game["odds"] = {"p_home": .48, "home_ml": 105, "away_ml": -125}
        self.assertEqual(build_site._summary_market_line(game, "H"),
                         "H +105 · XWOBA SIDE")
        # Below the threshold the opposing side is necessarily the
        # favourite -- which is the ticket, not the decision, so the label
        # names the decision and is read from its one home rather than
        # restated here.
        game["odds"] = {"p_home": .30, "home_ml": 220, "away_ml": -260}
        self.assertEqual(
            build_site._summary_market_line(game, "H"),
            f"A -260 · {build_site.hybrid_public_label('FADE')}")

    def test_an_exact_pickem_follows_the_model(self):
        """A devigged .500 market has no favourite, and sits well above .45.

        The grid this replaced had to file a no-favourite market into one of
        three direction bands and got the boundary claim wrong once. The
        hybrid makes no such claim: a pick'em is simply above the threshold,
        so the rule follows the model and the card says exactly that.
        """
        pk = build_site._lean_implied_p(
            {"home_ml": -110, "away_ml": -110}, "H", "A", "H")
        self.assertEqual(pk, 0.5)
        self.assertEqual(build_site.hybrid_action(pk), "FOLLOW")
        html = build_site._verdict_html(
            "H", {"home_ml": -110, "away_ml": -110}, "A", "H", {}, .02)
        self.assertIn("XWOBA SIDE → H", html)
        self.assertIn("50.0% no-vig", html)
        # Nothing on a followed game may read as opposition or as an accent.
        self.assertNotIn("verdict edge", html)
        for banned in ("OPPOSE", "opposes", "against"):
            self.assertNotIn(banned, html)

    def test_the_warm_accent_marks_a_fade_and_nothing_else(self):
        """The accent means the rule departed from the model's own lean.

        It used to mean "the market is not backing this", which fired on games
        the rule follows anyway -- an accent with no decision behind it.
        """
        follow = build_site._verdict_html(
            "H", {"home_ml": -110, "away_ml": -110}, "A", "H", {}, .02)
        self.assertNotIn("verdict edge", follow)
        fade = build_site._verdict_html(
            "A", {"home_ml": -260, "away_ml": 215}, "A", "H", {}, .02)
        self.assertIn("verdict edge", fade)
        self.assertIn(f"{build_site.hybrid_public_label('FADE')} → H", fade)

    def test_unusable_prices_abstain_rather_than_defaulting_to_a_branch(self):
        """No price is not a fade. Defaulting either way invents a selection."""
        for mp in (None, float("nan"), 0.0, 1.0, "x", -0.1, 1.5):
            self.assertIsNone(build_site.hybrid_action(mp))
            self.assertIsNone(build_site.hybrid_selection("H", "A", "H", mp))
        # A usable price with no lean is also an abstention.
        self.assertIsNone(build_site.hybrid_selection(None, "A", "H", .60))
        self.assertIsNone(build_site.hybrid_selection("", "A", "H", .60))

    def test_a_fade_selects_the_other_club_on_either_side(self):
        """The mirror has to work whichever side the model leaned."""
        self.assertEqual(build_site.hybrid_selection("H", "A", "H", .30), "A")
        self.assertEqual(build_site.hybrid_selection("A", "A", "H", .30), "H")
        self.assertEqual(build_site.hybrid_selection("H", "A", "H", .60), "H")
        # A lean naming neither club cannot be mirrored, so it abstains.
        self.assertIsNone(build_site.hybrid_selection("XXX", "A", "H", .30))

    def test_branch_panel_is_shown_with_market_adjusted_detail(self):
        ctx = {
            ("branch", "FOLLOW"): dict(
                n=14, w=10, l=4, implied=.481, actual=.714, excess=.233,
                excess_se=.121, roi=.352, units=4.93,
            ),
        }
        html = build_site._verdict_html(
            "PIT", dict(p_home=.529, away_ml=103), "PIT", "SD", ctx, .0187,
        )
        self.assertIn("Past V12 model-side selections · 14 completed games", html)
        self.assertIn("Won</span><span>10-4 (71.4%)", html)
        # Named for what it is. This sat directly under the game's own
        # "47.1% no-vig" as a bare "Market implied 48.1%", two unrelated
        # percentages a line apart with nothing saying they differ.
        self.assertIn("Their average price</span><span>48.1% implied", html)
        self.assertIn("Beat that price by</span><span><b>+23.3 pp", html)
        self.assertIn("within noise", html)

    def test_pit_acceptance_panel_has_the_requested_reads(self):
        ctx = {("branch", "FOLLOW"): dict(
            n=14, w=10, l=4, implied=.481, actual=.714, excess=.233,
            excess_se=.121, roi=.352, units=4.93,
        )}
        html = build_site._verdict_html(
            "PIT", dict(p_home=.529, away_ml=103), "PIT", "SD", ctx, .0187,
        )
        for expected in (
            "This game",
            "Model lean</span><span>PIT · V12 Δ .0187 (MEDIUM)",
            "Market price</span><span>PIT +103 · 47.1% no-vig",
            "Rule</span><span><b>XWOBA SIDE → PIT</b> +103",
            "remains the XWOBA side",
            "Past V12 model-side selections · 14 completed games",
            "not a prediction for this game",
            "Won</span><span>10-4 (71.4%)",
            "Their average price</span><span>48.1% implied",
            "Beat that price by</span><span><b>+23.3 pp",
            "within noise",
        ):
            self.assertIn(expected, html)
        for banned in ("value bet", "best bet", "free money", "lock"):
            self.assertNotIn(banned, html.lower())

    def test_records_respect_the_branch_floor(self):
        led = build_site.load_ledger_df()
        with mock.patch.object(build_site, "BRANCH_RECORD_MIN", 10**6):
            out = build_site.hybrid_branch_records()
        self.assertEqual([k for k in out if isinstance(k, tuple)], [])
        self.assertIn("threshold", out)

    def test_the_fade_branch_record_equals_its_chalk_control(self):
        """Not a coincidence to be observed -- a construction to be enforced.

        Fading a lean priced below .45 backs a side priced above .55, which is
        the favourite by definition. If these two ever differ, the record and
        its control were computed over different rows, which is the exact
        defect the shared observation frame exists to make impossible.
        """
        out = build_site.hybrid_branch_records()
        fade, chalk = out.get(("branch", "FADE")), out.get(("chalk", "FADE"))
        if not fade or not chalk:
            self.skipTest("no faded rows in the committed ledger yet")
        self.assertEqual(fade, chalk)

    def test_the_site_and_the_forward_test_decide_every_row_identically(self):
        """Two implementations of one rule, held against each other.

        `build_site._row_hybrid` walks the ledger a row at a time for the
        table; `hybrid_test` derives the same rule in bulk for the registered
        forward test. They share the threshold but not the code, so this is
        the check that the surface a reader sees and the instrument that will
        judge the rule cannot drift apart -- the failure mode that let the site
        publish a pooled record under a current-family label.
        """
        import hybrid_test
        led = build_site.load_ledger_df()
        if led is None:
            self.skipTest("ledger unavailable")
        g = led[(led["status"] == "graded") & led["xw_lean"].notna()
                & led["close_p_home"].notna()
                & led["model_tag"].isin(build_site.RECORD_TAGS)]
        if g.empty:
            self.skipTest("no priced current-family rows")
        lean_home = (g["xw_lean"] == g["home"]).to_numpy()
        home_won = (g["full_home"] > g["full_away"]).to_numpy()
        p_lean = np.where(lean_home, g["close_p_home"], 1 - g["close_p_home"])
        follow = p_lean >= hybrid_test.THRESHOLD
        lean_won = np.where(lean_home, home_won, ~home_won)
        expected_won = np.where(follow, lean_won, ~lean_won)
        rows = [build_site._row_hybrid(r) for _, r in g.iterrows()]
        self.assertTrue(
            (np.array([a == "FOLLOW" for a, _, _ in rows]) == follow).all(),
            "the table and the forward test disagree about a game's branch")
        self.assertTrue(
            (np.array([gr == "W" for _, _, gr in rows]) == expected_won).all(),
            "the table and the forward test disagree about a selection's result")

    def test_the_panel_never_presents_history_as_this_games_chances(self):
        """The clarity defect this layout exists to fix.

        The panel used to lead its history block with "Selection won  73.3%"
        directly under tonight's two clubs, which reads as this pick's win
        probability. It is the rate at which PAST picks in the same branch won,
        and the site publishes no per-game probability at all. So the block is
        headed by its own sample, every value row is past tense, and the two
        percentages that used to sit unlabelled a line apart -- this game's
        no-vig price and the branch's average price -- are each named.
        """
        ctx = {("branch", "FADE"): dict(n=15, w=11, l=4, implied=.586,
                                        actual=.733, excess=.147,
                                        excess_se=.127, roi=.238, units=3.56)}
        h = build_site._verdict_html(
            "LAD", dict(p_home=.70, away_ml=200, home_ml=-260), "LAD", "ARI",
            ctx, .005)
        self.assertIn("Past V12 market-side selections · 15 completed games", h)
        self.assertIn("not a prediction for this game", h)
        # The bare rate must not appear as its own value; it is qualified by
        # the record it came from.
        self.assertNotIn("<span>73.3%</span>", h)
        self.assertIn("Won</span><span>11-4 (73.3%)", h)
        # The game's own price and the branch's average price are distinct
        # numbers and must be distinctly labelled.
        self.assertIn("30.0% no-vig", h)
        self.assertIn("Their average price</span><span>58.6% implied", h)

    def test_the_ledger_labels_each_undecidable_case_distinctly(self):
        """Three different reasons the rule did not act, three different marks.

        A lean sitting unlabelled under a "Selection" heading is the
        substitution this repo already shipped once. Out-of-family, no lean,
        and no-price are separate facts and a reader must be able to tell which
        one a row is.
        """
        def row(tag, lean, ph, basis=None):
            return pd.Series(dict(
                model_tag=tag, xw_lean=lean, close_p_home=ph, home="H",
                away="A", away_sp="P1", home_sp="P2", xw_full="W",
                xw_delta=.01, status="graded", full_away=1, full_home=3,
                close_home_ml=-140, close_away_ml=120,
                pitching_basis_away=basis, pitching_basis_home=None))
        older = build_site._grades_row(row("woba+plat_consol_v5", "H", .60), True)
        self.assertIn("lean only", older)
        self.assertIn(build_site.MODEL_TAG, older)
        unpriced = build_site._grades_row(row(build_site.MODEL_TAG, "H", np.nan), True)
        self.assertIn("awaiting market", unpriced)
        # A .30 home lean is a hybrid FADE to A: show A, A's price, and the
        # inverted selection result while preserving the write-once raw fields.
        faded = build_site._grades_row(row(build_site.MODEL_TAG, "H", .30), True)
        self.assertIn("data-l='Selection'>A", faded)
        self.assertIn("data-l='ML'>+120", faded)
        self.assertIn("<span class='wlt L'>L</span>", faded)
        self.assertIn(build_site.hybrid_public_label("FADE"), faded)
        noleaan = build_site._grades_row(
            row(build_site.MODEL_TAG, np.nan, .60, basis="starter_unmeasured_no_lean"),
            True)
        self.assertIn("no lean", noleaan)
        # Each mark is unique to its own case.
        self.assertNotIn("awaiting market", older)
        self.assertNotIn("lean only", unpriced)

        page = build_site.render_grades_html("test build")
        self.assertIn("<th>Selection</th>", page)
        self.assertNotIn("<th>Model lean</th>", page)

    def test_the_report_and_the_site_publish_the_same_hybrid_record(self):
        """Third artifact in the chain, and the one that had drifted.

        data/ledger_report.txt headlined the raw lean while the public pages
        headlined the rule's selection -- 139-84 against 146-77 on the same
        games. Both now derive from `hybrid_test.apply_rule`, so this asserts
        the whole chain agrees: the report, the grades page and the forward
        test's own arithmetic.
        """
        import grade_leans
        import hybrid_test
        led = build_site.load_ledger_df()
        if led is None:
            self.skipTest("ledger unavailable")
        obs = build_site._lean_market_observations(led)
        if obs.empty:
            self.skipTest("no priced current-family rows")
        rule = build_site._lean_market_agg(
            obs, obs["won"].notna(), won="hybrid_won", p="hybrid_p",
            resid="hybrid_resid", profit="hybrid_profit")
        lines = grade_leans._hybrid_retrospective_lines(
            grade_leans._record_grades(led))
        self.assertTrue(lines, "the report prints no hybrid line")
        head = lines[0]
        self.assertIn(f"{rule['w']}-{rule['l']}", head)
        self.assertIn(f"n={rule['n']}", head)
        # The units figure is the one most likely to drift silently, because
        # it depends on the price each selection was scored at.
        self.assertIn(f"{rule['units']:+.2f}u", head)
        # And the retrospective label is not optional: without it the line
        # reads as an out-of-sample result.
        self.assertTrue(any("RETROSPECTIVE" in ln for ln in lines))

    def test_the_report_hybrid_line_carries_its_control(self):
        """A record with no yardstick beside it is the defect this repo has
        an entry for. The report's lean line is one control; always-chalk on
        the identical rows is the other, and it must travel with the record
        rather than being left to the reader to find elsewhere."""
        import grade_leans
        led = build_site.load_ledger_df()
        if led is None:
            self.skipTest("ledger unavailable")
        g = grade_leans._record_grades(led)
        lines = grade_leans._hybrid_retrospective_lines(g)
        if not lines:
            self.skipTest("no decidable current-family rows")
        joined = " ".join(lines)
        self.assertIn("always chalk", joined)
        # Scored on the same rows as the record, and it says so.
        n = int(lines[0].split("n=")[1].split(",")[0])
        self.assertIn(f"same {n} rows", joined)

    # ---- the three 2026-09-03 registrations, printed in sample -------------
    # They CAN be computed over all of v12 -- that is where every frozen
    # discovery constant came from -- and printing them is the same choice the
    # hybrid retrospective above makes. What these pin is that doing so cannot
    # be mistaken for the forward reading.

    def _retro(self):
        import grade_leans
        led = build_site.load_ledger_df()
        if led is None:
            self.skipTest("ledger unavailable")
        g = grade_leans._record_grades(led)
        lines = grade_leans._registration_retrospective_lines(g)
        if not lines:
            self.skipTest("no decidable current-family rows")
        return g, lines

    def test_the_registration_retrospectives_print_all_three(self):
        _, lines = self._retro()
        joined = " ".join(lines)
        for name in ("|Δ| filter", "abstain", "dog contrast"):
            with self.subTest(rule=name):
                self.assertIn(name, joined)

    def test_each_retrospective_matches_its_own_modules_arithmetic(self):
        """One rule, one implementation. A line here must not be a local copy.

        This is the property that makes the retrospective safe to print at
        all: if it could drift from the module, a reader comparing it against
        the forward block below would be comparing two different rules.
        """
        import abstain_test
        import dog_contrast_test
        import hybrid_test
        g, lines = self._retro()
        joined = " ".join(lines)
        h = hybrid_test.apply_rule(hybrid_test.decidable(g))

        m, _, n_d = abstain_test.fade_minus_abstain(h)
        if n_d:
            self.assertIn(f"{m:+.3f}u/declined game", joined)

        dogs = h[h["model_side_p"].astype(float) < dog_contrast_test.DOG_MAX]
        c, _, na, nb = dog_contrast_test.contrast(dogs)
        if na and nb:
            self.assertIn(f"{100 * c:+6.2f}pp", joined)
            self.assertIn(f"n={na} above, {nb} below", joined)

    def test_the_retrospective_splits_discovery_from_later_rows(self):
        """A retrospective over 'all v12' is a MIXTURE. Watching it grow is
        not watching evidence accumulate, and the header has to say so."""
        import delta_filter_test
        _, lines = self._retro()
        head = lines[0]
        self.assertIn("discovery rows", head)
        self.assertIn(delta_filter_test.REGISTERED_ON, head)
        self.assertRegex(head, r"n=\d+: \d+ discovery rows \+ \d+ since")

    def test_the_retrospective_says_it_is_not_evidence_for_itself(self):
        _, lines = self._retro()
        joined = " ".join(lines)
        self.assertIn("in sample", joined.lower())
        self.assertIn("none of these is evidence for itself", joined)
        self.assertIn("forward", joined.lower())

    def test_the_delta_filter_line_names_the_direction_that_vindicates_it(self):
        """Its registered headline is positive today and NEGATIVE would mean
        the rule works. Printed without that, the line reads backwards."""
        _, lines = self._retro()
        joined = " ".join(lines)
        self.assertIn("NEGATIVE would vindicate", joined)

    def test_each_line_carries_its_frozen_discovery_value(self):
        """So drift shows up as drift rather than as news."""
        import abstain_test
        import delta_filter_test
        import dog_contrast_test
        _, lines = self._retro()
        joined = " ".join(lines)
        for v in (f"{delta_filter_test.DISCOVERY_DROPPED_EXCESS:+.2f}",
                  f"{abstain_test.DISCOVERY_FADE_MINUS_ABSTAIN:+.3f}",
                  f"{dog_contrast_test.DISCOVERY_CONTRAST:+.2f}"):
            with self.subTest(discovery=v):
                self.assertIn(v, joined)

    def test_the_retrospective_never_raises_on_a_degenerate_frame(self):
        """It runs inside the job that ingests pregame rows. A diagnostic line
        must never be able to cost a slate."""
        import grade_leans
        import pandas as _pd
        for bad in (_pd.DataFrame(), _pd.DataFrame({"status": ["graded"]})):
            with self.subTest(frame=list(bad.columns)):
                self.assertIsInstance(
                    grade_leans._registration_retrospective_lines(bad), list)

    def test_the_locked_hybrid_has_separate_ledger_fields(self):
        """The public selection is stored without overwriting the raw lean."""
        import grade_leans
        writer = set(grade_leans.LEDGER_COLS) | set(grade_leans.AUDIT_COLS)
        self.assertIn("xw_full", writer)
        self.assertIn("hybrid_full", writer)
        self.assertTrue({"selection_rule_tag", "pregame_market_utc",
                         "pregame_away_ml", "pregame_home_ml",
                         "pregame_p_home", "hybrid_action",
                         "hybrid_selection", "hybrid_p", "hybrid_ml"}
                        .issubset(writer))

    def test_locked_row_uses_stored_selection_and_grade(self):
        row = pd.Series(dict(
            model_tag=build_site.MODEL_TAG,
            selection_rule_tag=build_site.HYBRID_RULE_TAG,
            hybrid_action="FADE", hybrid_selection="A", hybrid_full="L",
            xw_lean="H", xw_full="W", close_p_home=.80,
            away="A", home="H"))
        self.assertEqual(build_site._row_hybrid(row), ("FADE", "A", "L"))

    def test_a_faded_ledger_row_inverts_the_leans_grade(self):
        """The rule backed the other side, so the lean's W is the rule's L."""
        def row(lean, ph, grade, tag=None):
            return pd.Series(dict(
                xw_lean=lean, close_p_home=ph, home="H", away="A",
                xw_full=grade,
                model_tag=build_site.MODEL_TAG if tag is None else tag))
        # Lean priced at .30 -> faded onto the home side, grade inverts.
        self.assertEqual(build_site._row_hybrid(row("A", .70, "L")),
                         ("FADE", "H", "W"))
        self.assertEqual(build_site._row_hybrid(row("A", .70, "W")),
                         ("FADE", "H", "L"))
        # A tie stays a tie rather than being swallowed by an inversion.
        self.assertEqual(build_site._row_hybrid(row("A", .70, "T")),
                         ("FADE", "H", "T"))
        # Followed rows pass the grade straight through.
        self.assertEqual(build_site._row_hybrid(row("H", .60, "W")),
                         ("FOLLOW", "H", "W"))
        # No lean and no price both abstain rather than guessing.
        self.assertEqual(build_site._row_hybrid(row(None, .60, "W")),
                         (None, None, None))
        self.assertEqual(build_site._row_hybrid(row("H", np.nan, "W")),
                         (None, None, None))
        # And a row from an earlier prediction family is out of scope: the
        # rule is registered against the current one, so applying it there
        # would publish a selection nobody could have made -- under lean math
        # the rule was never paired with, and with a grade this function would
        # then invert.
        self.assertEqual(
            build_site._row_hybrid(row("A", .70, "L", tag="woba+plat_consol_v5")),
            (None, None, None))
        for tag in build_site.RECORD_TAGS:
            self.assertEqual(build_site._row_hybrid(row("A", .70, "L", tag=tag)),
                             ("FADE", "H", "W"))


class DevigDerivationTests(unittest.TestCase):
    """`_imp_ml` is the one American-odds conversion in this module.

    `_lean_implied_p`'s moneyline fallback used to restate it inline, spelling
    the negative branch `abs(hm)/(abs(hm)+100)` against `_imp_ml`'s
    `-ml/(-ml+100)`. The two agreed, which is exactly why they were merged --
    two copies of one statistic drift, and the reader cannot see which they
    are reading. The same lesson as `_excess_se` serving both calibration
    surfaces, and as `metric_label` being the single metric derivation.
    """

    def test_fallback_calls_imp_ml_rather_than_restating_it(self):
        """Proves the call, not just agreement.

        Agreement is what a second copy looks like right up until it drifts,
        so this perturbs `_imp_ml` and requires the fallback to move with it.
        A reintroduced inline copy passes an equality check and fails here.
        """
        odds = {"home_ml": -150, "away_ml": 130}
        real = build_site._lean_implied_p(odds, "H", "A", "H")
        self.assertIsNotNone(real)
        with mock.patch.object(build_site, "_imp_ml",
                               lambda ml: 0.25 if ml > 0 else 0.75):
            patched = build_site._lean_implied_p(odds, "H", "A", "H")
        self.assertAlmostEqual(patched, 0.75)
        self.assertNotAlmostEqual(patched, real)

    def test_fallback_matches_imp_ml_across_the_american_range(self):
        """The fold must not have moved a single admissible price."""
        for hm in list(range(-3000, -100, 17)) + list(range(100, 3000, 17)):
            for am in (-3000, -450, -150, -110, 100, 110, 250, 2900):
                ih, ia = build_site._imp_ml(hm), build_site._imp_ml(am)
                expect = ih / (ih + ia)
                got = build_site._lean_implied_p(
                    {"home_ml": hm, "away_ml": am}, "H", "A", "H")
                self.assertAlmostEqual(got, expect, places=12,
                                       msg=f"home_ml={hm} away_ml={am}")

    def test_prices_inside_the_american_gap_are_refused(self):
        """No real moneyline lies strictly between -100 and +100.

        The guard sits ahead of `_imp_ml`, which has no opinion on whether its
        argument is a price. It is also what keeps 0 out -- the one input
        where the old inline spelling and `_imp_ml` would have differed.
        """
        for hm, am in ((0, 0), (50, -150), (-150, 50), (-99, 120), (99, -120)):
            self.assertIsNone(
                build_site._lean_implied_p(
                    {"home_ml": hm, "away_ml": am}, "H", "A", "H"),
                msg=f"home_ml={hm} away_ml={am}")

    def test_devigged_price_is_normalised_and_side_aware(self):
        """The vig is removed, and the answer follows the leaned side."""
        odds = {"home_ml": -150, "away_ml": 130}
        home = build_site._lean_implied_p(odds, "H", "A", "H")
        away = build_site._lean_implied_p(odds, "A", "A", "H")
        self.assertAlmostEqual(home + away, 1.0, places=12)
        self.assertGreater(home, away)
        # the raw book prices overround; the devigged pair must not
        raw = build_site._imp_ml(-150) + build_site._imp_ml(130)
        self.assertGreater(raw, 1.0)

    def test_supplied_p_home_wins_over_the_moneylines(self):
        """A feed-supplied devigged probability is preferred, not recomputed."""
        odds = {"p_home": 0.42, "home_ml": -150, "away_ml": 130}
        self.assertAlmostEqual(
            build_site._lean_implied_p(odds, "H", "A", "H"), 0.42, places=12)
        self.assertAlmostEqual(
            build_site._lean_implied_p(odds, "A", "A", "H"), 0.58, places=12)

    def test_unusable_inputs_return_none_rather_than_a_default(self):
        for odds in (None, {}, {"home_ml": None, "away_ml": 130},
                     {"home_ml": -150}, {"home_ml": "x", "away_ml": 130},
                     {"p_home": 0.0, "home_ml": -150, "away_ml": 130},
                     {"p_home": 1.0, "home_ml": -150, "away_ml": 130}):
            self.assertIsNone(
                build_site._lean_implied_p(odds, "H", "A", "H"), msg=str(odds))
        self.assertIsNone(
            build_site._lean_implied_p({"p_home": 0.5}, None, "A", "H"))
