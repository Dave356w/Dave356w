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
        self.assertEqual(result[123], {"era": 3.03, "starts": 5})

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

    def test_pitcher_card_shows_season_era_but_colors_l5_vs_league(self):
        side = dict(
            t="R", pl_fl={}, R=5, L=4, S=0, has_pl=False, padv=0,
            era_l5=3.03, era_l5_gs=5, era_season=3.85,
            pit_xw=.310, pit_k=27.1, pit_bb=7.5, pit_hh=35.0,
            pl_sp=None, pl_sp_raw=None, pl_edge=None, pl_reliable=False,
            xw_edge=-.015, p="Test Pitcher", opp_abbr="TST", lu_status="posted",
            opp_xw=None, pl_mx=None, hitters=[],
        )
        html = build_site._side_html(
            "AWAY", side,
            {"ERA": 4.20, "xwOBA": .320, "K%": 22.0, "Hard Hit%": 39.0,
             "OPS": .720},
        )
        self.assertIn("ERA · L5", html)
        self.assertIn("3.03", html)
        self.assertIn("season 3.85", html)
        self.assertNotIn("5 GS", html)
        self.assertIn(
            "<div class='stat' style='background:rgba(var(--cool),0.23)'>"
            "<div class='l'>ERA · L5</div>",
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


if __name__ == "__main__":
    unittest.main()
