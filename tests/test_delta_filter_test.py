"""The |delta| filter registration block must not drift.

Same reasoning as tests/test_forward_test.py and tests/test_hybrid_test.py, and
the same deliberate exception to this repo's rule against freezing measured
numbers into tests: here the literals ARE the subject. A pre-registered rule
whose threshold can be edited after seeing results is not pre-registered.

What is NOT asserted here, on purpose: that the discovery constants still
reproduce from the committed ledger. They were measured over the 252 decidable
v12 rows that existed on 2026-09-03 and that set grows every slate, so such a
test would fail for arithmetic the Actions bot did overnight -- the exact
defect recorded in CLAUDE.md under "Freezing a measured number into a test".
The constants are the registration; the ledger is not their source any more.

Beyond the constants, the structural properties the honest reading depends on
are pinned, because none is obvious from the formula:

  * exactly 0.012 is KEPT (`>=`), so a boundary game is played;
  * `apply_filter` carries no date filter, so the retrospective and forward
    readings cannot drift apart;
  * the always-chalk control is computed on the same rows, because the entire
    case against this rule is that the kept half is favourite-heavy;
  * the registered prior is NEGATIVE and the report says so, so a good forward
    run has to overcome an expectation rather than merely arrive.
"""

import datetime as _dt
import unittest

import numpy as np
import pandas as pd

import delta_filter_test as dft


def _led(deltas, lean_home=True, home_won=True, p_home=0.55,
         date="2026-09-10", status="graded"):
    """A minimal graded ledger frame at the given deltas and one flat price."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    lean_home = np.broadcast_to(np.asarray(lean_home, dtype=bool), (n,))
    home_won = np.broadcast_to(np.asarray(home_won, dtype=bool), (n,))
    p_home = np.broadcast_to(np.asarray(p_home, dtype=float), (n,))
    home_ml = np.where(p_home >= .5, -np.round(100 * p_home / (1 - p_home)),
                       np.round(100 * (1 - p_home) / p_home))
    away_ml = np.where(p_home >= .5, np.round(100 * p_home / (1 - p_home)),
                       -np.round(100 * (1 - p_home) / p_home))
    return pd.DataFrame({
        "status": status, "game_date": date, "home": "H", "away": "A",
        "close_p_home": p_home,
        "xw_lean": np.where(lean_home, "H", "A"),
        # Sign is irrelevant to the filter -- it reads the magnitude -- so the
        # frames alternate it to prove that.
        "xw_net": deltas * np.where(np.arange(n) % 2 == 0, 1.0, -1.0),
        "full_home": np.where(home_won, 1, 0),
        "full_away": np.where(home_won, 0, 1),
        "close_home_ml": home_ml, "close_away_ml": away_ml,
    })


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(dft.REGISTERED_ON, "2026-09-03")
        self.assertAlmostEqual(dft.DELTA_THRESHOLD, 0.012, places=10)
        self.assertAlmostEqual(dft.STAKE, 1.0, places=10)
        self.assertEqual(dft.RULE_TAG, "xwoba_delta_filter_v1")
        self.assertEqual(dft.PRIOR, "negative")

    def test_discovery_constants_are_exactly_as_measured(self):
        self.assertEqual(dft.DISCOVERY_DROPPED, 82)
        self.assertAlmostEqual(dft.DISCOVERY_DROP_RATE, 0.325, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_DROPPED_EXCESS, 5.53, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_DROPPED_SE, 5.47, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_CONTRAST, 1.77, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_CONTRAST_SE, 6.63, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_CHALK_GAP_DROPPED, 6.37, places=10)
        self.assertAlmostEqual(dft.DISCOVERY_CHALK_GAP_KEPT, 3.14, places=10)
        self.assertAlmostEqual(dft.NULL_MAX_P, 0.693, places=10)

    def test_gates_are_exactly_as_registered(self):
        self.assertEqual(dft.GATE_DROPPED, 393)
        self.assertEqual(dft.GATE_DROPPED_REALISTIC, 1090)

    def test_registration_date_is_a_real_past_date(self):
        d = _dt.date.fromisoformat(dft.REGISTERED_ON)
        self.assertLessEqual(d, _dt.date.today())

    def test_no_band_constants_exist(self):
        """One frozen number. Tiers on a z = +0.27 contrast are noise-mining."""
        src = open(dft.__file__).read()
        for banned in ("PRICE_BANDS", "DELTA_BANDS", "_BANDS = ", "TIERS"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, src)

    def test_the_discovery_counts_are_internally_consistent(self):
        """Arithmetic on frozen values, so it cannot go stale with the ledger."""
        self.assertAlmostEqual(dft.DISCOVERY_DROPPED / 252,
                               dft.DISCOVERY_DROP_RATE, places=3)
        z = dft.DISCOVERY_CONTRAST / dft.DISCOVERY_CONTRAST_SE
        self.assertAlmostEqual(z, 0.27, places=2)
        self.assertLess(abs(z), 0.5, "a registered contrast this size is null")

    def test_the_null_max_test_did_not_clear(self):
        """Above 0.5 means the search returned less than noise typically does.

        Pinned because it is the single strongest reason this is registered
        rather than shipped, and because hybrid_test's 0.019 sits in the same
        repo -- the two must not be confused for one another.
        """
        self.assertGreater(dft.NULL_MAX_P, 0.5)

    def test_the_prior_is_negative_not_null(self):
        """hybrid_test is null; this one is negative. The difference is the
        chalk control, and conflating them would let a hot run read as news."""
        self.assertEqual(dft.PRIOR, "negative")
        self.assertGreater(dft.DISCOVERY_CHALK_GAP_DROPPED,
                           dft.DISCOVERY_CHALK_GAP_KEPT)


class ThresholdBoundaryTests(unittest.TestCase):
    def test_exactly_at_the_threshold_is_kept(self):
        g = dft.apply_filter(dft.decidable(_led([dft.DELTA_THRESHOLD])))
        self.assertTrue(bool(g["kept"].iloc[0]))

    def test_just_below_the_threshold_is_dropped(self):
        g = dft.apply_filter(dft.decidable(_led([dft.DELTA_THRESHOLD - 1e-9])))
        self.assertFalse(bool(g["kept"].iloc[0]))

    def test_the_filter_reads_magnitude_not_sign(self):
        g = dft.apply_filter(dft.decidable(_led([0.02, 0.02])))
        self.assertLess(float(g["xw_net"].iloc[0]) * float(g["xw_net"].iloc[1]), 0)
        self.assertTrue(bool(g["kept"].all()))

    def test_kept_and_dropped_partition_the_rows(self):
        g = dft.apply_filter(dft.decidable(_led([.001, .012, .05, .0119])))
        self.assertEqual(int(g["kept"].sum()) + int((~g["kept"]).sum()), len(g))
        self.assertEqual(list(g["kept"]), [False, True, True, False])


class PurityTests(unittest.TestCase):
    def test_apply_filter_has_no_date_filter(self):
        """The retrospective and forward readings must share one arithmetic."""
        old = _led([0.05], date="2020-01-01")
        self.assertEqual(len(dft.apply_filter(dft.decidable(old))), 1)

    def test_apply_filter_does_not_mutate_its_input(self):
        led = dft.decidable(_led([0.05]))
        before = list(led.columns)
        dft.apply_filter(led)
        self.assertEqual(list(led.columns), before)

    def test_the_threshold_parameter_defaults_to_the_registered_value(self):
        a = dft.apply_filter(dft.decidable(_led([0.013])))
        b = dft.apply_filter(dft.decidable(_led([0.013])),
                             threshold=dft.DELTA_THRESHOLD)
        self.assertEqual(list(a["kept"]), list(b["kept"]))

    def test_no_caller_in_this_repo_passes_a_different_threshold(self):
        """The parameter exists to reproduce the sweep, not to retune."""
        import glob
        import os
        import re
        bad = []
        for path in glob.glob("*.py") + glob.glob("tests/*.py"):
            if os.path.basename(path) in ("delta_filter_test.py",
                                          "test_delta_filter_test.py"):
                continue
            for m in re.finditer(r"apply_filter\(([^)]*)\)", open(path).read()):
                if "threshold" in m.group(1):
                    bad.append(f"{path}: {m.group(0)}")
        self.assertEqual(bad, [])


class ScopeTests(unittest.TestCase):
    def test_scored_rows_excludes_the_registration_date_itself(self):
        led = _led([0.05], date=dft.REGISTERED_ON)
        self.assertEqual(len(dft.scored_rows(led)), 0)

    def test_scored_rows_includes_the_day_after(self):
        after = (_dt.date.fromisoformat(dft.REGISTERED_ON)
                 + _dt.timedelta(days=1)).isoformat()
        self.assertEqual(len(dft.scored_rows(_led([0.05], date=after))), 1)

    def test_the_committed_discovery_rows_are_out_of_scope(self):
        """Nothing on or before the registration date may ever be scored."""
        led = pd.read_csv(dft.LEDGER, low_memory=False)
        g = dft.scored_rows(led)
        self.assertIsNotNone(g)
        if len(g):
            self.assertTrue((g["game_date"].astype(str)
                             > dft.REGISTERED_ON).all())

    def test_a_row_with_no_lean_is_excluded(self):
        """A v5 abstention is the model declining; a filter cannot re-decline."""
        led = _led([0.05, 0.05])
        led.loc[0, "xw_lean"] = np.nan
        self.assertEqual(len(dft.decidable(led)), 1)

    def test_xw_net_is_required(self):
        """Opposite of hybrid_test, which reads a direction and never the size."""
        self.assertIn("xw_net", dft.REQUIRED_COLUMNS)
        led = _led([0.05])
        led["xw_net"] = np.nan
        self.assertEqual(len(dft.decidable(led)), 0)

    def test_a_pending_row_is_not_scored(self):
        self.assertEqual(len(dft.scored_rows(_led([0.05], status="pending"))), 0)

    def test_a_ledger_missing_columns_is_not_scored_rather_than_guessed(self):
        self.assertIsNone(dft.decidable(pd.DataFrame({"status": ["graded"]})))
        self.assertIsNone(dft.scored_rows(pd.DataFrame({"status": ["graded"]})))


class ControlTests(unittest.TestCase):
    def test_always_chalk_is_scored_on_the_identical_rows(self):
        g = dft.apply_filter(dft.decidable(_led([.005, .02, .05])))
        self.assertEqual(len(g["chalk_won"].dropna()), len(g))
        self.assertEqual(len(g["chalk_profit"].dropna()), len(g))

    def test_always_chalk_backs_the_larger_devigged_probability(self):
        g = dft.apply_filter(dft.decidable(
            _led([.05, .05], p_home=[0.62, 0.38], home_won=True)))
        self.assertEqual(list(g["chalk_won"]), [True, False])

    def test_the_control_is_independent_of_the_filter(self):
        """Chalk needs no lean and no delta, so it must not move with them."""
        a = dft.apply_filter(dft.decidable(_led([.001, .001])))
        b = dft.apply_filter(dft.decidable(_led([.900, .900])))
        self.assertEqual(list(a["chalk_won"]), list(b["chalk_won"]))
        self.assertEqual(list(a["chalk_profit"]), list(b["chalk_profit"]))


class ExcessTests(unittest.TestCase):
    def test_the_error_bar_is_defined_at_n_equals_one(self):
        """A p-hat based form would print +/-0.0 on an all-W bucket."""
        e, se = dft._excess(np.array([True]), np.array([0.55]))
        self.assertAlmostEqual(e, 0.45, places=10)
        self.assertGreater(se, 0.0)

    def test_it_uses_the_markets_probabilities_not_the_outcomes(self):
        _, se = dft._excess(np.array([True, True]), np.array([0.5, 0.5]))
        self.assertAlmostEqual(se, float(np.sqrt(0.5)) / 2, places=12)

    def test_an_empty_bucket_is_nan_not_zero(self):
        e, se = dft._excess(np.array([], dtype=bool), np.array([]))
        self.assertTrue(np.isnan(e) and np.isnan(se))


class ReportTests(unittest.TestCase):
    def test_report_lines_never_raise_and_always_name_the_gate(self):
        body = "\n".join(dft.report_lines())
        self.assertIn("GATE", body)
        self.assertIn(str(dft.GATE_DROPPED), body)

    def test_the_report_states_the_negative_prior_when_empty(self):
        body = "\n".join(dft.report_lines(_led([0.05], date="2020-01-01")))
        self.assertIn("NEGATIVE", body)

    def test_the_report_says_which_direction_vindicates_the_rule(self):
        """A reader must not be able to mistake a positive run for success."""
        after = (_dt.date.fromisoformat(dft.REGISTERED_ON)
                 + _dt.timedelta(days=1)).isoformat()
        body = "\n".join(dft.report_lines(
            _led([.005, .005, .05, .05], date=after)))
        self.assertIn("NEGATIVE", body)
        self.assertIn("discarding winning bets", body)

    def test_the_report_carries_the_chalk_control_beside_the_headline(self):
        after = (_dt.date.fromisoformat(dft.REGISTERED_ON)
                 + _dt.timedelta(days=1)).isoformat()
        body = "\n".join(dft.report_lines(
            _led([.005, .005, .05, .05], date=after)))
        self.assertIn("always-chalk", body)
        self.assertIn("model-minus-chalk", body)

    def test_the_report_carries_no_betting_recommendation(self):
        body = "\n".join(dft.report_lines()).lower()
        for banned in ("value bet", "recommend", "you should", "edge found",
                       "profitable"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, body)

    def test_report_lines_survive_a_ledger_missing_columns(self):
        body = "\n".join(dft.report_lines(pd.DataFrame({"status": ["graded"]})))
        self.assertIn("not scored", body)

    def test_report_lines_survive_a_ledger_with_no_eligible_rows(self):
        body = "\n".join(dft.report_lines(_led([0.05], date="2020-01-01")))
        self.assertIn("eligible rows since registration: 0", body)

    def test_the_committed_ledger_reports_without_raising(self):
        led = pd.read_csv(dft.LEDGER, low_memory=False)
        self.assertTrue(dft.report_lines(led))


class LedgerReportWiringTests(unittest.TestCase):
    def test_grade_leans_prints_this_registration(self):
        """All three registrations belong in one artifact, or a reader
        comparing them has to know a fourth place to look."""
        src = open("grade_leans.py").read()
        self.assertIn("delta_filter_test", src)
