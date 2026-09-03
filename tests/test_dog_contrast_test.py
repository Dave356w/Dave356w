"""The underdog sign-flip registration block must not drift.

Same exception to the no-frozen-numbers rule as the other three: the literals
ARE the subject.

This module has one property the others don't, and most of these tests defend
it. The obvious thing to register was the 0.45-0.50 BAND -- it is the
flattering number and the one a reader will want quoted. It is deliberately
not the registered quantity, because it shares 34 of its 47 discovery rows
with `forward_test`'s arm 2 and fails its own search test at P = 0.2805. The
CONTRAST uses both halves, so removing the losing half is what it measures
rather than what it does. If a later edit quietly makes the band the headline,
these tests are what should stop it.
"""

import datetime as _dt
import unittest

import numpy as np
import pandas as pd

import dog_contrast_test as dc
import hybrid_test as ht


def _locked(p_home, lean_home=True, home_won=True, date="2026-09-20"):
    """A graded ledger frame carrying the locked pregame columns."""
    p_home = np.asarray(p_home, dtype=float)
    n = len(p_home)
    lean_home = np.broadcast_to(np.asarray(lean_home, dtype=bool), (n,))
    home_won = np.broadcast_to(np.asarray(home_won, dtype=bool), (n,))
    home_ml = np.where(p_home >= .5, -np.round(100 * p_home / (1 - p_home)),
                       np.round(100 * (1 - p_home) / p_home))
    away_ml = np.where(p_home >= .5, np.round(100 * p_home / (1 - p_home)),
                       -np.round(100 * (1 - p_home) / p_home))
    model_p = np.where(lean_home, p_home, 1 - p_home)
    follow = model_p >= ht.THRESHOLD
    bet_home = np.where(follow, lean_home, ~lean_home)
    bet_won = np.where(bet_home, home_won, ~home_won)
    return pd.DataFrame({
        "status": "graded", "game_date": date, "home": "H", "away": "A",
        "close_p_home": p_home,
        "xw_lean": np.where(lean_home, "H", "A"),
        "full_home": np.where(home_won, 1, 0),
        "full_away": np.where(home_won, 0, 1),
        "close_home_ml": home_ml, "close_away_ml": away_ml,
        "selection_rule_tag": ht.RULE_TAG,
        "pregame_p_home": p_home,
        "pregame_home_ml": home_ml, "pregame_away_ml": away_ml,
        "hybrid_action": np.where(follow, "FOLLOW", "FADE"),
        "hybrid_selection": np.where(bet_home, "H", "A"),
        "hybrid_p": np.where(bet_home, p_home, 1 - p_home),
        "hybrid_ml": np.where(bet_home, home_ml, away_ml),
        "hybrid_full": np.where(bet_won, "W", "L"),
    })


AFTER = (_dt.date.fromisoformat(dc.REGISTERED_ON)
         + _dt.timedelta(days=1)).isoformat()


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(dc.REGISTERED_ON, "2026-09-03")
        self.assertAlmostEqual(dc.DOG_MAX, 0.50, places=10)
        self.assertEqual(dc.RULE_TAG, "xwoba_dog_contrast_v1")
        self.assertEqual(dc.PRIOR, "null")

    def test_discovery_constants_are_exactly_as_measured(self):
        self.assertEqual(dc.DISCOVERY_ABOVE_N, 47)
        self.assertAlmostEqual(dc.DISCOVERY_ABOVE_EXCESS, 16.11, places=10)
        self.assertAlmostEqual(dc.DISCOVERY_ABOVE_SE, 7.28, places=10)
        self.assertEqual(dc.DISCOVERY_BELOW_N, 20)
        self.assertAlmostEqual(dc.DISCOVERY_BELOW_EXCESS, -11.48, places=10)
        self.assertAlmostEqual(dc.DISCOVERY_BELOW_SE, 11.00, places=10)
        self.assertAlmostEqual(dc.DISCOVERY_CONTRAST, 27.59, places=10)
        self.assertAlmostEqual(dc.DISCOVERY_CONTRAST_SE, 13.20, places=10)

    def test_the_search_test_constants_are_frozen(self):
        self.assertAlmostEqual(dc.NULL_MAX_P, 0.0707, places=10)
        self.assertAlmostEqual(dc.BAND_NULL_MAX_P, 0.2805, places=10)
        self.assertEqual(dc.ARM2_OVERLAP, (34, 47))

    def test_gates_are_exactly_as_registered(self):
        self.assertEqual(dc.GATE_DOG_LEANS, 88)
        self.assertEqual(dc.GATE_DOG_LEANS_REALISTIC, 198)
        self.assertAlmostEqual(dc.DISCOVERY_DOG_LEANS_PER_SLATE, 3.53, places=10)

    def test_registration_date_is_a_real_past_date(self):
        self.assertLessEqual(_dt.date.fromisoformat(dc.REGISTERED_ON),
                             _dt.date.today())

    def test_the_discovery_contrast_is_internally_consistent(self):
        """Arithmetic on frozen values; cannot go stale with the ledger."""
        self.assertAlmostEqual(
            dc.DISCOVERY_CONTRAST,
            dc.DISCOVERY_ABOVE_EXCESS - dc.DISCOVERY_BELOW_EXCESS, places=2)
        self.assertAlmostEqual(
            dc.DISCOVERY_CONTRAST_SE,
            float(np.hypot(dc.DISCOVERY_ABOVE_SE, dc.DISCOVERY_BELOW_SE)),
            places=1)
        z = dc.DISCOVERY_CONTRAST / dc.DISCOVERY_CONTRAST_SE
        self.assertAlmostEqual(z, 2.09, places=2)

    def test_the_gate_is_not_sized_to_the_discovery_effect(self):
        """A selected maximum reproducing itself over ~7 slates proves nothing,
        so the gate is sized to a smaller effect than the one observed."""
        self.assertGreater(dc.GATE_DOG_LEANS, 26)
        self.assertGreater(dc.GATE_DOG_LEANS_REALISTIC, dc.GATE_DOG_LEANS)


class WhyNotTheBandTests(unittest.TestCase):
    """The band is the flattering number. These say why it is not registered."""

    def test_the_band_failed_its_own_search_test(self):
        self.assertGreater(dc.BAND_NULL_MAX_P, 0.25)

    def test_the_contrast_did_better_than_the_band_on_the_same_kind_of_test(self):
        self.assertLess(dc.NULL_MAX_P, dc.BAND_NULL_MAX_P)

    def test_neither_cleared_a_conventional_bar(self):
        """Pinned so the 0.0707 is never quoted as significance."""
        self.assertGreater(dc.NULL_MAX_P, 0.05)

    def test_the_band_overlaps_an_existing_registration(self):
        shared, total = dc.ARM2_OVERLAP
        self.assertGreater(shared / total, 0.5)

    def test_the_headline_requires_both_halves(self):
        """A contrast from one half is the band. Must refuse, not degrade."""
        only_above = dc.scored_rows(_locked([0.53], lean_home=False, date=AFTER))
        c, se, na, nb = dc.contrast(only_above)
        self.assertEqual(nb, 0)
        self.assertTrue(np.isnan(c))

    def test_the_report_names_the_band_as_deliberately_not_registered(self):
        body = "\n".join(dc.report_lines(
            _locked([0.53, 0.60], lean_home=False, date=AFTER)))
        self.assertIn("deliberately NOT the registered quantity", body)


class SplitTests(unittest.TestCase):
    def test_the_split_is_hybrid_tests_own_object(self):
        """A second 0.45 would break the 'a priori split point' claim."""
        self.assertIs(dc.SPLIT, ht.THRESHOLD)

    def test_no_standalone_split_assignment_in_the_source(self):
        import re
        src = open(dc.__file__).read()
        self.assertNotRegex(src, r"(?m)^_?[A-Z_]*(SPLIT|THRESHOLD)[A-Z_]*\s*=\s*0\.45")

    def test_exactly_at_the_split_is_above(self):
        """`>=` splits above, matching the hybrid's own boundary convention.

        Constructed with a HOME lean so `model_side_p` is the stored price
        itself. See the next test for why the away side cannot express this.
        """
        g = dc.scored_rows(_locked([ht.THRESHOLD], lean_home=True, date=AFTER))
        self.assertEqual(len(dc.above(g)), 1)
        self.assertEqual(len(dc.below(g)), 0)

    def test_the_away_side_boundary_is_one_ulp_low_and_that_is_pre_existing(self):
        """Known, inert, and NOT introduced here -- pinned so it stays known.

        `model_side_p` for an away lean is `1 - pregame_p_home`, and
        `1 - (1 - 0.45)` is 0.44999999999999996, so a game priced at exactly
        0.55 with an away lean lands BELOW a 0.45 split. The same arithmetic is
        in `hybrid_test.apply_rule` and `apply_locked_rule`, so this is the
        shipped rule's boundary, not this module's.

        Deliberately not "fixed": changing it would move a registered rule's
        branch assignment, and a devigged price landing on exactly 0.55 to full
        float precision does not occur -- these come from integer money lines.
        A test that documents an inert asymmetry beats a silent one.
        """
        self.assertLess(1 - (1 - ht.THRESHOLD), ht.THRESHOLD)
        g = dc.scored_rows(_locked([1 - ht.THRESHOLD], lean_home=False,
                                   date=AFTER))
        self.assertEqual(len(dc.below(g)), 1)
        self.assertEqual(len(dc.above(g)), 0)

    def test_just_under_the_split_is_below(self):
        g = dc.scored_rows(_locked([0.5501], lean_home=False, date=AFTER))
        self.assertEqual(len(dc.below(g)), 1)

    def test_above_and_below_partition_the_dog_leans(self):
        g = dc.scored_rows(_locked([0.52, 0.56, 0.62, 0.51],
                                   lean_home=False, date=AFTER))
        self.assertEqual(len(dc.above(g)) + len(dc.below(g)), len(g))


class ScopeTests(unittest.TestCase):
    def test_only_dog_leans_are_scored(self):
        """A favourite lean is not this hypothesis and must not dilute it."""
        g = dc.scored_rows(_locked([0.60, 0.40], lean_home=True, date=AFTER))
        self.assertTrue((g["model_side_p"].astype(float) < dc.DOG_MAX).all())

    def test_it_splits_on_the_models_side_not_the_rules_selection(self):
        """On a faded row the hybrid backs the OTHER side; splitting on that
        would measure the rule rather than the model's signal."""
        led = _locked([0.62], lean_home=False, date=AFTER)   # lean is the dog
        g = dc.scored_rows(led)
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(float(g["model_side_p"].iloc[0]), 0.38, places=9)
        self.assertEqual(len(dc.below(g)), 1)

    def test_this_registration_does_not_inherit_the_hybrids_date(self):
        self.assertGreater(dc.REGISTERED_ON, ht.REGISTERED_ON)

    def test_a_row_between_the_two_registration_dates_is_not_scored(self):
        between = "2026-09-02"
        led = _locked([0.62], lean_home=False, date=between)
        self.assertGreater(len(ht.scored_rows(led)), 0)
        self.assertEqual(len(dc.scored_rows(led)), 0)

    def test_the_registration_date_itself_is_excluded(self):
        self.assertEqual(len(dc.scored_rows(
            _locked([0.62], lean_home=False, date=dc.REGISTERED_ON))), 0)

    def test_the_committed_ledger_scores_nothing_before_the_date(self):
        led = pd.read_csv(dc.LEDGER, low_memory=False)
        g = dc.scored_rows(led)
        self.assertIsNotNone(g)
        if len(g):
            self.assertTrue((g["game_date"].astype(str) > dc.REGISTERED_ON).all())


class ExcessTests(unittest.TestCase):
    def test_the_excess_is_of_the_lean_not_the_hybrid_selection(self):
        # Lean is the dog at .38 and loses; the hybrid would fade and win.
        led = _locked([0.62], lean_home=False, home_won=True, date=AFTER)
        g = dc.scored_rows(led)
        e, se, n = dc._excess(g)
        self.assertEqual(n, 1)
        self.assertLess(e, 0)                 # the LEAN lost

    def test_the_error_bar_is_defined_at_n_equals_one(self):
        led = _locked([0.62], lean_home=False, date=AFTER)
        _, se, n = dc._excess(dc.scored_rows(led))
        self.assertEqual(n, 1)
        self.assertGreater(se, 0.0)

    def test_an_empty_half_is_nan_not_zero(self):
        e, se, n = dc._excess(None)
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(e) and np.isnan(se))


class ReportTests(unittest.TestCase):
    def test_report_lines_never_raise_and_name_the_gate(self):
        body = "\n".join(dc.report_lines())
        self.assertIn("GATE", body)
        self.assertIn(str(dc.GATE_DOG_LEANS), body)

    def test_the_report_declares_it_is_not_independent(self):
        """Three readings of one small set of games is not three samples."""
        body = "\n".join(dc.report_lines(
            _locked([0.53, 0.62], lean_home=False, date=AFTER)))
        self.assertIn("NOT independent", body)
        self.assertIn("arm 2", body)

    def test_the_report_states_the_null_prior_and_its_search_test(self):
        body = "\n".join(dc.report_lines(_locked([0.62], lean_home=False,
                                                 date="2020-01-01")))
        self.assertIn("NULL", body)
        self.assertIn("0.0707", body)

    def test_the_report_carries_no_betting_recommendation(self):
        body = "\n".join(dc.report_lines()).lower()
        for banned in ("value bet", "recommend", "should bet", "edge found",
                       "profitable system"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, body)

    def test_report_lines_survive_a_ledger_missing_columns(self):
        body = "\n".join(dc.report_lines(pd.DataFrame({"status": ["graded"]})))
        self.assertIn("not scored", body)

    def test_the_committed_ledger_reports_without_raising(self):
        led = pd.read_csv(dc.LEDGER, low_memory=False)
        self.assertTrue(dc.report_lines(led))


class LedgerReportWiringTests(unittest.TestCase):
    def test_grade_leans_prints_this_registration(self):
        self.assertIn("dog_contrast_test", open("grade_leans.py").read())
