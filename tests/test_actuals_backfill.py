import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import actuals_backfill as ab
import grade_leans


def _bat(pa=38, ab_=34, h=9, d=2, t=0, hr=1, bb=3, ibb=0, hbp=1, sf=0):
    return {"plateAppearances": pa, "atBats": ab_, "hits": h, "doubles": d,
            "triples": t, "homeRuns": hr, "baseOnBalls": bb,
            "intentionalWalks": ibb, "hitByPitch": hbp, "sacFlies": sf}


def _box(away_bat=None, home_bat=None, away_sp=("5.2", 24), home_sp=("6.0", 23),
         away_started=True, home_started=True):
    def team(bat, sp, started):
        ip, bf = sp
        players = {"ID1": {"stats": {"pitching": {
            "gamesStarted": 1 if started else 0,
            "inningsPitched": ip, "battersFaced": bf}}}}
        # A reliever listed FIRST, to catch anything that takes position for role.
        players = {"ID0": {"stats": {"pitching": {
            "gamesStarted": 0, "inningsPitched": "1.0", "battersFaced": 4}}},
            **players}
        return {"teamStats": {"batting": bat if bat is not None else _bat()},
                "players": players}
    return {"teams": {"away": team(away_bat, away_sp, away_started),
                      "home": team(home_bat, home_sp, home_started)}}


class InningsParsingTests(unittest.TestCase):
    def test_thirds_not_decimals(self):
        """'5.2' is 5 and two thirds, not 5.2. float() would be wrong by up to
        0.47 IP per start and biased one way, which lands straight in q."""
        self.assertEqual(ab.innings_to_outs("5.2"), 17)
        self.assertEqual(ab.innings_to_outs("6.0"), 18)
        self.assertEqual(ab.innings_to_outs("0.1"), 1)
        self.assertEqual(ab.innings_to_outs(7), 21)

    def test_refuses_impossible_notation_rather_than_guessing(self):
        self.assertIsNone(ab.innings_to_outs("5.7"))
        for bad in (None, "", "-", "nan"):
            self.assertIsNone(ab.innings_to_outs(bad))


class WobaComputationTests(unittest.TestCase):
    def test_matches_the_weights_by_hand(self):
        c = dict(ab=34, h=9, **{"2b": 2, "3b": 1}, hr=1, bb=3, ibb=1, hbp=1, sf=1)
        singles = 9 - 2 - 1 - 1
        ubb = 3 - 1
        w = ab.WOBA_W
        num = (w["bb"] * ubb + w["hbp"] * 1 + w["1b"] * singles
               + w["2b"] * 2 + w["3b"] * 1 + w["hr"] * 1)
        self.assertAlmostEqual(ab.woba_from_components(c), num / (34 + ubb + 1 + 1),
                               places=12)

    def test_impossible_lines_return_none_not_a_number(self):
        # more XBH than hits, and more IBB than BB: a stored 0.0 would read as
        # a real shutout rather than a broken line.
        self.assertIsNone(ab.woba_from_components(
            dict(ab=30, h=1, **{"2b": 2, "3b": 0}, hr=0, bb=1, ibb=0, hbp=0, sf=0)))
        self.assertIsNone(ab.woba_from_components(
            dict(ab=30, h=5, **{"2b": 1, "3b": 0}, hr=0, bb=1, ibb=3, hbp=0, sf=0)))
        self.assertIsNone(ab.woba_from_components(
            dict(ab=0, h=0, **{"2b": 0, "3b": 0}, hr=0, bb=0, ibb=0, hbp=0, sf=0)))
        self.assertIsNone(ab.woba_from_components({"ab": 30}))


class BoxscoreParsingTests(unittest.TestCase):
    def test_starter_is_found_by_gamesStarted_not_by_list_order(self):
        p = ab.parse_boxscore(_box(away_sp=("5.2", 24)))
        self.assertAlmostEqual(p["away"]["sp_ip"], 17 / 3.0, places=12)
        self.assertEqual(p["away"]["sp_bf"], 24)

    def test_no_starter_flagged_leaves_the_pitching_line_null(self):
        p = ab.parse_boxscore(_box(away_started=False))
        self.assertIsNone(p["away"]["sp_ip"])
        self.assertIsNone(p["away"]["sp_bf"])
        # batting is independent and must survive
        self.assertIsNotNone(ab.woba_from_components(p["away"]))


class AttachActualsTests(unittest.TestCase):
    def _led(self, **over):
        base = dict(game_pk=[1], game_date=["2026-07-01"], away=["SF"], home=["LAD"],
                    model_tag=["xw+plat_consol_v10"], status=["graded"],
                    full_away=[3.0], full_home=[5.0], gamePk=[770001])
        base.update({k: [v] for k, v in over.items()})
        return pd.DataFrame(base)

    def test_attaches_and_is_idempotent(self):
        led = self._led()
        with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
            out = ab.attach_actuals(led.copy(), verbose=False)
            self.assertEqual(gj.call_count, 1)
            self.assertTrue(out["act_woba_away"].notna().all())
            self.assertTrue(out["act_sp_ip_home"].notna().all())
            again = ab.attach_actuals(out, verbose=False)
            self.assertEqual(gj.call_count, 1)      # second pass fetches nothing
        pd.testing.assert_frame_equal(out, again)

    def test_actuals_survive_ledger_reload_and_second_pass_fetches_nothing(self):
        led = self._led()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.csv")
            with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
                first = ab.attach_actuals(led.copy(), verbose=False)
                first.to_csv(path, index=False)
                with mock.patch.object(grade_leans, "LEDGER_PATH", path):
                    loaded = grade_leans.load_ledger()
                second = ab.attach_actuals(loaded, verbose=False)

            self.assertEqual(gj.call_count, 1)
            self.assertAlmostEqual(
                second.at[0, "act_woba_away"],
                first.at[0, "act_woba_away"],
            )
            self.assertAlmostEqual(
                second.at[0, "act_sp_ip_home"],
                first.at[0, "act_sp_ip_home"],
            )

    def test_pending_rows_are_never_touched(self):
        """An actual is an outcome. A pending row's prediction can still be
        refreshed before first pitch, so an outcome must not reach it."""
        led = self._led(full_away=np.nan, full_home=np.nan, status="pending")
        with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
            out = ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, 0)
        self.assertTrue(out["act_woba_away"].isna().all())

    def test_existing_actuals_are_never_overwritten(self):
        """An actual is immutable once recorded, even when the row is revisited.

        A row backfilled under an older schema IS re-fetched now, to pick up
        columns added since -- so "never overwritten" can no longer lean on the
        row never being visited. The values themselves must survive the visit.
        """
        led = self._led()
        led["act_woba_away"] = 0.111
        led["act_woba_home"] = 0.222
        led["act_schema"] = 1
        with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
            out = ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, 1)                    # revisited
        self.assertAlmostEqual(float(out["act_woba_away"].iloc[0]), 0.111)
        self.assertAlmostEqual(float(out["act_woba_home"].iloc[0]), 0.222)
        self.assertEqual(float(out["act_schema"].iloc[0]), ab.ACT_SCHEMA)

    def test_a_row_at_the_current_schema_is_not_refetched(self):
        led = self._led()
        led["act_woba_away"] = 0.111
        led["act_woba_home"] = 0.222
        led["act_schema"] = ab.ACT_SCHEMA
        with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
            ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, 0)

    def test_row_without_a_gamePk_is_skipped_not_guessed(self):
        led = self._led(gamePk=np.nan)
        with mock.patch.object(ab, "_get_json", return_value=_box()) as gj:
            out = ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, 0)
        self.assertTrue(out["act_woba_away"].isna().all())

    def test_incomplete_boxscore_is_refused_wholesale(self):
        broken = _box(away_bat={"atBats": 30})     # missing everything else
        led = self._led()
        with mock.patch.object(ab, "_get_json", return_value=broken):
            out = ab.attach_actuals(led, verbose=False)
        self.assertTrue(out["act_woba_away"].isna().all())
        self.assertTrue(out["act_woba_home"].isna().all())   # neither side partial
        self.assertEqual(len(out.attrs["actual_skips"]), 1)

    def test_fetch_failure_leaves_the_row_for_next_run(self):
        led = self._led()
        with mock.patch.object(ab, "_get_json", side_effect=RuntimeError("503")):
            out = ab.attach_actuals(led, verbose=False)
        self.assertTrue(out["act_woba_away"].isna().all())
        self.assertIn("boxscore fetch failed", out.attrs["actual_skips"][0][1])


class WorkBoundTests(unittest.TestCase):
    """This fetches one box score PER ROW. The build job has a 15-minute
    timeout that exists so an upstream hang cannot take the site build down;
    an unbounded per-row loop over ~105 rows of history would defeat it."""

    def _led(self, n):
        return pd.DataFrame({
            "game_pk": range(n), "game_date": ["2026-07-01"] * n,
            "away": ["SF"] * n, "home": ["LAD"] * n,
            "model_tag": ["xw+plat_consol_v10"] * n, "status": ["graded"] * n,
            "full_away": [3.0] * n, "full_home": [5.0] * n,
            "gamePk": list(range(770001, 770001 + n)),
        })

    def test_stops_after_consecutive_failures_instead_of_asking_105_times(self):
        led = self._led(60)
        with mock.patch.object(ab, "_get_json", side_effect=RuntimeError("503")) as gj:
            out = ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, ab.MAX_CONSEC_FAIL)
        self.assertIn("consecutive fetch failures", out.attrs["actual_stopped"])

    def test_an_intermittent_failure_does_not_trip_the_abort(self):
        led = self._led(12)
        seq = [RuntimeError("x") if i % 3 == 0 else _box() for i in range(12)]
        with mock.patch.object(ab, "_get_json", side_effect=seq) as gj:
            out = ab.attach_actuals(led, verbose=False)
        self.assertEqual(gj.call_count, 12)                 # ran the whole set
        self.assertIsNone(out.attrs["actual_stopped"])
        self.assertEqual(int(out["act_woba_away"].notna().sum()), 8)

    def test_time_budget_stops_the_loop_and_leaves_rows_for_next_run(self):
        led = self._led(40)
        clock = {"t": 0.0}

        def slow(_url, **_kw):
            clock["t"] += 30.0          # each fetch "costs" 30s
            return _box()

        with mock.patch.object(ab, "_get_json", side_effect=slow), \
             mock.patch.object(ab.time, "monotonic", side_effect=lambda: clock["t"]):
            out = ab.attach_actuals(led, verbose=False)
        self.assertIn("time budget", out.attrs["actual_stopped"])
        done = int(out["act_woba_away"].notna().sum())
        self.assertLess(done, 40)
        self.assertGreater(done, 0)
        # Unfinished rows are simply still null -- the same state as before, so
        # the next run picks them up with no repair step.
        self.assertEqual(int(out["act_woba_away"].isna().sum()), 40 - done)


class PairingTests(unittest.TestCase):
    """The cross is the one thing here that fails silently if wrong: pairing
    the wrong sides yields a plausible near-zero correlation, not an error."""

    def test_mx_away_pairs_with_the_home_offense(self):
        led = pd.DataFrame([dict(
            game_pk=1, model_tag="v", mx_xwoba_away=0.330, mx_xwoba_home=0.290,
            act_woba_away=0.111, act_woba_home=0.999,
            act_pa_away=38, act_pa_home=41)])
        p = ab.paired_rates(led)
        away_row = p[p.pitching_side == "away"].iloc[0]
        self.assertEqual(away_row.batting_side, "home")
        self.assertAlmostEqual(away_row.pred, 0.330)
        self.assertAlmostEqual(away_row.act, 0.999)      # HOME offense
        self.assertAlmostEqual(away_row.pa, 41)
        home_row = p[p.pitching_side == "home"].iloc[0]
        self.assertAlmostEqual(home_row.pred, 0.290)
        self.assertAlmostEqual(home_row.act, 0.111)      # AWAY offense

    def test_sp_ip_pairs_on_the_same_side(self):
        led = pd.DataFrame([dict(expected_sp_ip_away=5.0, expected_sp_ip_home=6.0,
                                 act_sp_ip_away=4.0, act_sp_ip_home=7.0)])
        p = ab.paired_sp_ip(led)
        a = p[p.side == "away"].iloc[0]
        self.assertAlmostEqual(a.pred, 5.0)
        self.assertAlmostEqual(a.act, 4.0)

    def test_empty_inputs_give_empty_frames_not_exceptions(self):
        self.assertTrue(ab.paired_rates(pd.DataFrame()).empty)
        self.assertTrue(ab.paired_sp_ip(pd.DataFrame()).empty)
        self.assertTrue(ab.paired_net(pd.DataFrame()).empty)

    def test_net_pairs_home_minus_away_on_both_sides(self):
        """xw_net favours HOME when positive, so it pairs with home - away.

        Signed the other way this still produces a slope, just a negative one,
        which reads as a broken model rather than as a broken pairing.
        """
        led = pd.DataFrame([dict(game_pk=1, model_tag="v", xw_net=0.02,
                                 act_woba_home=0.400, act_woba_away=0.300)])
        p = ab.paired_net(led)
        self.assertEqual(len(p), 1)
        self.assertAlmostEqual(p.iloc[0].pred, 0.02)
        self.assertAlmostEqual(p.iloc[0].act, 0.100)     # home - away

    def test_net_is_one_row_per_game_not_two(self):
        led = pd.DataFrame([dict(game_pk=1, model_tag="v", xw_net=0.01,
                                 act_woba_home=0.35, act_woba_away=0.30),
                            dict(game_pk=2, model_tag="v", xw_net=-0.01,
                                 act_woba_home=0.28, act_woba_away=0.33)])
        self.assertEqual(len(ab.paired_net(led)), 2)


class PhaseSplitTests(unittest.TestCase):
    """The bullpen line is a residual, so it has to be validated, not assumed."""

    def _row(self, **over):
        r = dict(
            game_pk=1, model_tag="t",
            act_sp_ab_away=20, act_sp_h_away=5, act_sp_2b_away=1,
            act_sp_3b_away=0, act_sp_hr_away=1, act_sp_bb_away=2,
            act_sp_ibb_away=0, act_sp_hbp_away=0, act_sp_sf_away=0,
            act_ab_home=33, act_h_home=9, act_2b_home=2, act_3b_home=0,
            act_hr_home=2, act_bb_home=4, act_ibb_home=0, act_hbp_home=1,
            act_sf_home=1, act_pa_home=39,
            act_woba_home=0.400, act_woba_away=0.250,
            starter_xwoba_away=0.310, bullpen_xwoba_away=0.300,
            opp_xwoba_neutral_away=0.320)
        r.update(over)
        return pd.DataFrame([r])

    def test_bullpen_is_the_batting_line_minus_the_starter(self):
        sp, bp = ab.phase_lines(self._row().iloc[0], "away")
        self.assertEqual(sp["ab"], 20)
        self.assertEqual(bp["ab"], 13)      # 33 - 20
        self.assertEqual(bp["h"], 4)        # 9 - 5
        self.assertEqual(bp["hr"], 1)       # 2 - 1

    def test_a_negative_residual_refuses_to_be_a_bullpen_line(self):
        """A starter line larger than the team line means the two disagree.
        A rate built on that would be fiction, so it is not produced."""
        sp, bp = ab.phase_lines(self._row(act_sp_h_away=99).iloc[0], "away")
        self.assertIsNotNone(sp)
        self.assertIsNone(bp)

    def test_missing_starter_line_yields_neither_phase(self):
        sp, bp = ab.phase_lines(self._row(act_sp_ab_away=None).iloc[0], "away")
        self.assertIsNone(sp)
        self.assertIsNone(bp)

    def test_sp_and_bp_pair_on_the_same_side_but_lineup_crosses(self):
        """The three components do not pair alike; a wrong cross reads as a
        weak model rather than as an error."""
        p = ab.paired_components(self._row())
        lu = p[p.component == "lineup"].iloc[0]
        self.assertEqual(lu.side, "away")
        self.assertAlmostEqual(lu.pred, 0.320)
        self.assertAlmostEqual(lu.act, 0.400)      # the HOME offense it faced
        sp = p[p.component == "SP"].iloc[0]
        self.assertAlmostEqual(sp.pred, 0.310)     # away starter, same side
        self.assertAlmostEqual(sp.act, ab.woba_from_components(
            ab.phase_lines(self._row().iloc[0], "away")[0]))

    def test_components_absent_rather_than_guessed_without_a_phase_line(self):
        """Today's ledger has no stored starter line; only lineup may pair."""
        led = self._row()
        for c in [c for c in led.columns if c.startswith("act_sp_")]:
            led[c] = np.nan
        comps = set(ab.paired_components(led).component)
        self.assertEqual(comps, {"lineup"})

    def test_schema_gate_revisits_rows_backfilled_under_an_older_schema(self):
        """The old gate asked only about act_woba, which would strand every
        column added later on the rows that already had one -- i.e. all."""
        df = pd.DataFrame([dict(full_away=1, full_home=2, gamePk=7,
                                act_woba_away=0.3, act_woba_home=0.3,
                                act_schema=1)])
        self.assertEqual(ab._todo_index(df).tolist(), [0])
        df.loc[0, "act_schema"] = ab.ACT_SCHEMA
        self.assertEqual(ab._todo_index(df).tolist(), [])


class NetCalibrationScopeTests(unittest.TestCase):
    """Pooling scale families produces a slope describing no model that ran.

    This is not hypothetical: the pooled ledger reads +0.48 ("over-dispersed")
    purely because unshrunk pre-v5 rows have ~2.4x the delta spread of the
    shrunk families, and the current lineage reads +1.45 on its own. The guard
    is what keeps that number off the report.
    """

    def _mixed(self):
        rng = np.random.default_rng(7)
        rows = []
        # wide family: predictions 2.5x too spread (true slope 0.4)
        for _ in range(300):
            p = rng.normal(0, 0.060)
            rows.append(dict(game_pk=len(rows), model_tag="wide", xw_net=p,
                             act_woba_home=0.30 + 0.4 * p + rng.normal(0, 0.10),
                             act_woba_away=0.30))
        # narrow family: calibrated (true slope 1.0)
        for _ in range(300):
            p = rng.normal(0, 0.024)
            rows.append(dict(game_pk=len(rows), model_tag="narrow", xw_net=p,
                             act_woba_home=0.30 + 1.0 * p + rng.normal(0, 0.10),
                             act_woba_away=0.30))
        return pd.DataFrame(rows)

    def test_pooling_scale_families_distorts_the_slope(self):
        led = self._mixed()
        wide = ab.calibration(*ab.paired_net(led[led.model_tag == "wide"])[["pred", "act"]].T.to_numpy())
        narrow = ab.calibration(*ab.paired_net(led[led.model_tag == "narrow"])[["pred", "act"]].T.to_numpy())
        pooled = ab.calibration(*ab.paired_net(led)[["pred", "act"]].T.to_numpy())
        self.assertLess(abs(wide["slope"] - 0.4), 0.25)
        self.assertLess(abs(narrow["slope"] - 1.0), 0.45)
        # the pooled slope is dragged toward the wide family and misses the
        # narrow family's truth by more than that family's own error
        self.assertLess(pooled["slope"], narrow["slope"] - 0.2)

    def test_unscoped_summary_refuses_to_print_a_net_slope(self):
        led = self._mixed()
        self.assertNotIn("lean delta", "\n".join(ab.actuals_summary(led)))
        scoped = "\n".join(ab.actuals_summary(led, tags=["narrow"]))
        self.assertIn("lean delta", scoped)


class CalibrationTests(unittest.TestCase):
    def test_recovers_a_planted_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.316, 0.017, 4000)
        y = 0.05 + 0.8 * x + rng.normal(0, 0.084, 4000)
        c = ab.calibration(x, y)
        self.assertAlmostEqual(c["slope"], 0.8, delta=4 * c["se_slope"])
        self.assertEqual(c["n"], 4000)

    def test_se_reflects_the_real_noise_floor(self):
        """At the sample sizes this repo will have, the slope's se is large.
        The report leans on this number to label itself under-powered."""
        rng = np.random.default_rng(1)
        x = rng.normal(0.316, 0.017, 200)
        y = x + rng.normal(0, 0.084, 200)
        self.assertGreater(ab.calibration(x, y)["se_slope"], 0.25)

    def test_undefined_slope_returns_none_rather_than_zero(self):
        self.assertIsNone(ab.calibration([1, 2], [1, 2]))            # n < 3
        self.assertIsNone(ab.calibration([0.3] * 9, [0.2] * 9))      # no spread

    def test_skill_is_zero_when_the_model_equals_the_baseline(self):
        y = np.random.default_rng(2).normal(0.316, 0.084, 500)
        s = ab.skill_vs_baseline(np.full(500, 0.316), y, 0.316)
        self.assertAlmostEqual(s["skill"], 0.0, places=12)

    def test_skill_is_positive_for_a_predictor_with_real_signal(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(0.316, 0.03, 4000)
        pred = truth
        act = truth + rng.normal(0, 0.05, 4000)
        self.assertGreater(ab.skill_vs_baseline(pred, act, 0.316)["skill"], 0.1)


class SummaryTests(unittest.TestCase):
    def test_nothing_paired_yields_no_lines(self):
        self.assertEqual(ab.actuals_summary(pd.DataFrame()), [])

    def test_under_powered_slope_is_labelled(self):
        rng = np.random.default_rng(5)
        n = 120
        led = pd.DataFrame({
            "game_pk": range(n), "model_tag": ["v10"] * n,
            "mx_xwoba_away": rng.normal(0.316, 0.017, n),
            "mx_xwoba_home": rng.normal(0.316, 0.017, n),
            "act_woba_away": rng.normal(0.316, 0.084, n),
            "act_woba_home": rng.normal(0.316, 0.084, n),
            "act_pa_away": 38, "act_pa_home": 38,
            "expected_sp_ip_away": rng.normal(5.2, 0.9, n),
            "expected_sp_ip_home": rng.normal(5.2, 0.9, n),
            "act_sp_ip_away": rng.normal(5.0, 1.5, n),
            "act_sp_ip_home": rng.normal(5.0, 1.5, n),
        })
        out = "\n".join(ab.actuals_summary(led, baseline=0.316))
        self.assertIn("starter IP", out)
        self.assertIn("calibration slope", out)
        self.assertIn("UNDER-POWERED", out)

    def test_family_line_gives_each_figure_its_own_n(self):
        """v6/v7 stored expected_sp_ip but never an mx. A single shared n would
        print a measured IP bias beside a sample size it did not come from."""
        fam = pd.DataFrame([dict(game_pk=1, model_tag="v7",
                                 expected_sp_ip_away=5.0, expected_sp_ip_home=6.0,
                                 act_sp_ip_away=4.5, act_sp_ip_home=6.5)])
        ln = ab.actuals_family_line("v7", fam)
        self.assertIn("SP IP n=  2", ln)
        self.assertNotIn("wOBA", ln)          # no rate count is claimed at all

    def test_family_line_is_none_when_nothing_pairs(self):
        self.assertIsNone(ab.actuals_family_line("v2", pd.DataFrame([{"game_pk": 1}])))


    def test_ip_pools_across_families_while_rates_stay_scoped(self):
        """expected_pitcher_ip is one estimator shared since v6, so its slope
        must pool; the rates must not, because families predict from different
        inputs. Scoping IP to the record family would show n=2 for months
        after every bump and hide the sample the line exists to surface."""
        led = pd.DataFrame({
            "game_pk": [1, 2], "model_tag": ["woba+plat_consol_v1", "xw+plat_consol_v10"],
            "mx_xwoba_away": [0.32, 0.32], "mx_xwoba_home": [0.31, 0.31],
            "act_woba_away": [0.30, 0.40], "act_woba_home": [0.33, 0.28],
            "act_pa_away": [38, 38], "act_pa_home": [38, 38],
            "expected_sp_ip_away": [5.0, 6.0], "expected_sp_ip_home": [5.5, 4.5],
            "act_sp_ip_away": [4.0, 6.5], "act_sp_ip_home": [6.0, 4.0],
        })
        out = "\n".join(ab.actuals_summary(led, tags=["woba+plat_consol_v1"]))
        self.assertIn("starter IP   n=4", out)      # both families
        self.assertIn("offense wOBA n=2", out)      # current family only

    def test_tag_filter_scopes_to_one_family(self):
        led = pd.DataFrame({
            "game_pk": [1, 2], "model_tag": ["v10", "v2"],
            "mx_xwoba_away": [0.32, 0.32], "mx_xwoba_home": [0.31, 0.31],
            "act_woba_away": [0.30, 0.90], "act_woba_home": [0.33, 0.90],
            "act_pa_away": [38, 38], "act_pa_home": [38, 38],
        })
        out = "\n".join(ab.actuals_summary(led, tags=["v10"]))
        self.assertIn("n=2", out)          # one game, two sides -- not four


if __name__ == "__main__":
    unittest.main()
