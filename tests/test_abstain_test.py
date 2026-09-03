"""The abstain-vs-fade registration block must not drift.

Same reasoning as the other three registration tests, and the same deliberate
exception to the rule against freezing measured numbers: the literals ARE the
subject.

Two properties get more attention than the constants, because this module is
the only registration that DELEGATES part of its row selection:

  * it must share `hybrid_test`'s declined set exactly -- the two rules are the
    same decision on every followed game, so if they disagreed about which
    games are declined this would stop being a comparison;
  * it must NOT share `hybrid_test`'s registration date. That bug shipped for
    one run: delegating wholesale scored two slates that are part of THIS
    module's discovery sample, and both numbers looked like forward rows. The
    date-bound tests below are the ones that would have caught it.
"""

import datetime as _dt
import unittest

import numpy as np
import pandas as pd

import abstain_test as at
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


AFTER = (_dt.date.fromisoformat(at.REGISTERED_ON)
         + _dt.timedelta(days=1)).isoformat()


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(at.REGISTERED_ON, "2026-09-03")
        self.assertAlmostEqual(at.STAKE, 1.0, places=10)
        self.assertEqual(at.RULE_TAG, "xwoba_market_abstain_v1")
        self.assertEqual(at.PRIOR, "null")

    def test_discovery_constants_are_exactly_as_measured(self):
        self.assertEqual(at.DISCOVERY_DECLINED, 20)
        self.assertAlmostEqual(at.DISCOVERY_DECLINE_RATE, 0.079, places=10)
        self.assertAlmostEqual(at.DISCOVERY_FADE_MINUS_ABSTAIN, 0.1754, places=10)
        self.assertAlmostEqual(at.DISCOVERY_SD, 0.7918, places=10)
        self.assertEqual(at.DISCOVERY_CI, (-0.168, 0.503))
        self.assertEqual(at.DISCOVERY_LEAN_RECORD_ON_DECLINED, (6, 14))
        self.assertAlmostEqual(at.DISCOVERY_CHALK_EXCESS_PP, 2.54, places=10)

    def test_gates_are_exactly_as_registered(self):
        self.assertEqual(at.GATE_DECLINED, 82)
        self.assertEqual(at.GATE_DECLINED_REALISTIC, 251)

    def test_registration_date_is_a_real_past_date(self):
        self.assertLessEqual(_dt.date.fromisoformat(at.REGISTERED_ON),
                             _dt.date.today())

    def test_the_discovery_headline_is_not_significant(self):
        """Arithmetic on frozen values; cannot go stale with the ledger.

        z ~ +0.99 is why the prior is null rather than positive, and pinning it
        stops a later reader quoting +0.18u as though it were established.
        """
        se = at.DISCOVERY_SD / np.sqrt(at.DISCOVERY_DECLINED)
        self.assertLess(abs(at.DISCOVERY_FADE_MINUS_ABSTAIN / se), 2.0)
        self.assertLess(at.DISCOVERY_CI[0], 0.0)   # the CI spans zero
        self.assertGreater(at.DISCOVERY_CI[1], 0.0)


class OneThresholdTests(unittest.TestCase):
    def test_the_threshold_is_hybrid_tests_own_object(self):
        """Not a second 0.45. The declined set must be the set the shipped
        rule fades, or this stops being a comparison."""
        self.assertIs(at.THRESHOLD, ht.THRESHOLD)

    def test_no_standalone_threshold_assignment_in_the_source(self):
        import re
        src = open(at.__file__).read()
        self.assertNotRegex(src, r"(?m)^_?[A-Z_]*THRESHOLD[A-Z_]*\s*=\s*0\.45")

    def test_the_declined_set_is_exactly_the_hybrids_fade_set(self):
        led = _locked([0.62, 0.38, 0.55, 0.47], lean_home=True, date=AFTER)
        g = at.scored_rows(led)
        mine = set(at.declined(g).index)
        theirs = set(g[g["hybrid_action"].eq("FADE")].index)
        self.assertEqual(mine, theirs)

    def test_kept_and_declined_partition_the_rows(self):
        led = _locked([0.62, 0.38, 0.55, 0.47], date=AFTER)
        g = at.scored_rows(led)
        self.assertEqual(len(at.kept(g)) + len(at.declined(g)), len(g))
        self.assertFalse(set(at.kept(g).index) & set(at.declined(g).index))


class DateBoundTests(unittest.TestCase):
    """The bug that shipped for one run. These are the tests that catch it."""

    def test_this_registration_does_not_inherit_the_hybrids_date(self):
        self.assertNotEqual(at.REGISTERED_ON, ht.REGISTERED_ON)
        self.assertGreater(at.REGISTERED_ON, ht.REGISTERED_ON)

    def test_a_row_between_the_two_registration_dates_is_not_scored(self):
        """Scored by hybrid_test, must NOT be scored here -- it is part of this
        module's discovery sample."""
        between = "2026-09-02"
        self.assertGreater(between, ht.REGISTERED_ON)
        self.assertLess(between, at.REGISTERED_ON)
        led = _locked([0.62], date=between)
        self.assertGreater(len(ht.scored_rows(led)), 0)
        self.assertEqual(len(at.scored_rows(led)), 0)

    def test_the_registration_date_itself_is_excluded(self):
        self.assertEqual(len(at.scored_rows(_locked([0.62],
                                                    date=at.REGISTERED_ON))), 0)

    def test_the_day_after_is_included(self):
        self.assertEqual(len(at.scored_rows(_locked([0.62], date=AFTER))), 1)

    def test_the_committed_ledger_scores_nothing_before_the_date(self):
        led = pd.read_csv(at.LEDGER, low_memory=False)
        g = at.scored_rows(led)
        self.assertIsNotNone(g)
        if len(g):
            self.assertTrue((g["game_date"].astype(str)
                             > at.REGISTERED_ON).all())


class HeadlineTests(unittest.TestCase):
    def test_a_followed_game_contributes_identically_zero(self):
        """The two rules place the same bet there, so the headline cannot
        absorb the model's own performance."""
        led = _locked([0.62, 0.70], lean_home=True, date=AFTER)
        g = at.scored_rows(led)
        self.assertEqual(len(at.declined(g)), 0)
        m, se, n = at.fade_minus_abstain(g)
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(m))

    def test_the_headline_is_the_fade_branchs_profit_on_declined_games(self):
        led = _locked([0.62, 0.62], lean_home=False, home_won=True, date=AFTER)
        g = at.scored_rows(led)
        d = at.declined(g)
        self.assertEqual(len(d), 2)
        m, _, n = at.fade_minus_abstain(g)
        self.assertAlmostEqual(m, float(d["profit"].mean()), places=12)

    def test_a_winning_fade_gives_a_positive_headline(self):
        """Positive keeps the shipped rule; the sign convention must not flip."""
        led = _locked([0.62], lean_home=False, home_won=True, date=AFTER)
        m, _, n = at.fade_minus_abstain(at.scored_rows(led))
        self.assertEqual(n, 1)
        self.assertGreater(m, 0)

    def test_a_losing_fade_gives_a_negative_headline(self):
        led = _locked([0.62], lean_home=False, home_won=False, date=AFTER)
        m, _, n = at.fade_minus_abstain(at.scored_rows(led))
        self.assertEqual(n, 1)
        self.assertLess(m, 0)

    def test_an_empty_frame_is_nan_not_zero(self):
        m, se, n = at.fade_minus_abstain(at.scored_rows(_locked([0.62],
                                                                date="2020-01-01")))
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(m) and np.isnan(se))


class FadeIsChalkTests(unittest.TestCase):
    def test_every_faded_game_backs_the_favourite(self):
        """By construction: fading a side priced under .45 backs one over .55.

        This is why the prior is null -- the branch under test carries no model
        content, so a good forward run may only be saying favourites are hot.
        """
        led = _locked([0.62, 0.70, 0.38, 0.30], lean_home=False, date=AFTER)
        d = at.declined(at.scored_rows(led))
        self.assertTrue(len(d))
        fav_home = d["pregame_p_home"].values >= 0.5
        self.assertTrue(bool((d["bet_home"].values == fav_home).all()))


class ReportTests(unittest.TestCase):
    def test_report_lines_never_raise_and_name_the_gate(self):
        body = "\n".join(at.report_lines())
        self.assertIn("GATE", body)
        self.assertIn(str(at.GATE_DECLINED), body)

    def test_the_report_states_which_direction_keeps_the_shipped_rule(self):
        body = "\n".join(at.report_lines(
            _locked([0.62, 0.38], lean_home=False, date=AFTER)))
        self.assertIn("positive keeps the shipped fade branch", body)
        self.assertIn("negative says", body)

    def test_the_report_carries_the_chalk_control(self):
        body = "\n".join(at.report_lines(
            _locked([0.62, 0.38], lean_home=False, date=AFTER)))
        self.assertIn("always-chalk", body)

    def test_the_report_states_the_null_prior_when_empty(self):
        body = "\n".join(at.report_lines(_locked([0.62], date="2020-01-01")))
        self.assertIn("NULL", body)

    def test_the_report_carries_no_betting_recommendation(self):
        body = "\n".join(at.report_lines()).lower()
        for banned in ("value bet", "recommend", "should bet", "edge found",
                       "profitable system"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, body)

    def test_report_lines_survive_a_ledger_missing_columns(self):
        body = "\n".join(at.report_lines(pd.DataFrame({"status": ["graded"]})))
        self.assertIn("not scored", body)

    def test_the_committed_ledger_reports_without_raising(self):
        led = pd.read_csv(at.LEDGER, low_memory=False)
        self.assertTrue(at.report_lines(led))


class LedgerReportWiringTests(unittest.TestCase):
    def test_grade_leans_prints_this_registration(self):
        self.assertIn("abstain_test", open("grade_leans.py").read())
