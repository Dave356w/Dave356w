"""The pre-registered forward test's registration block must not drift.

This file is deliberately the opposite of the repo's usual rule against
freezing measured numbers into tests. Everywhere else a literal goes stale as
the ledger grows and the test fails for reasons unrelated to the diff. Here the
literals ARE the subject: a pre-registered rule whose parameters can be edited
after seeing results is not pre-registered, and the whole value of
`forward_test.py` is that its numbers were fixed before the rows it scores
existed.

So these assertions exist to make tuning DELIBERATE. Changing a constant means
editing this file too, in a commit that says so -- which is a decision a
reviewer can see, rather than a quiet re-fit. If a parameter genuinely must
change, the honest move is a NEW registration with a new date, scoring from
scratch, not an edit to the old one.
"""

import datetime as _dt
import unittest

import forward_test as ft


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(ft.REGISTERED_ON, "2026-08-29")
        self.assertAlmostEqual(ft.COEF_A, 0.1292658378, places=10)
        self.assertAlmostEqual(ft.COEF_B, 5.7751338037, places=10)
        self.assertAlmostEqual(ft.GAP_THRESHOLD, 0.06, places=10)
        self.assertEqual(ft.DIRECTION, "fade")
        self.assertAlmostEqual(ft.STAKE, 1.0, places=10)
        self.assertEqual(ft.SECONDARY_THRESHOLDS, (0.00, 0.02, 0.04, 0.06, 0.10))
        self.assertEqual(ft.GATE_BETS, 1300)

    def test_dog_arm_constants_are_exactly_as_registered(self):
        self.assertAlmostEqual(ft.DOG_ARM_STAKE, 1.0, places=10)
        self.assertEqual(ft.DOG_ARM_GATE_BETS, 1200)
        self.assertEqual(ft.DOG_ARM_SECONDARY_BANDS, (
            ("+100..+130", 0.435, 0.500),
            ("+130..+175", 0.363, 0.435),
            ("+175 or longer", 0.000, 0.363),
        ))

    def test_dog_arm_is_registered_unbanded(self):
        """The bands were chosen after seeing them; the rule must not use one.

        Registering `+100..+130` -- the band that ran +7.5% in discovery --
        would smuggle the search back into a pre-registration. The a-priori
        hypothesis is "the model leans a plus-money dog", so the headline is
        computed over every plus-money lean and the bands are context.
        """
        import numpy as np
        import pandas as pd
        n = 60
        rng = np.random.default_rng(5)
        p = rng.uniform(.30, .48, n)          # every row a plus-money dog
        led = pd.DataFrame({
            "status": "graded", "game_date": "2026-09-05",
            "close_p_home": p, "xw_lean": "H", "home": "H",
            "xw_net": rng.normal(0, .02, n),
            "full_home": 1, "full_away": 0,
            "close_home_ml": np.round(100 * (1 - p) / p).astype(int),
            "close_away_ml": -np.round(100 * p / (1 - p)).astype(int),
        })
        line = [l for l in ft.report_lines(led) if "REGISTERED" in l][0]
        # All 60 rows qualify, not just those inside the flattering band.
        self.assertIn("n=60", line)

    def test_primary_threshold_is_among_the_reported_secondaries(self):
        # Otherwise the headline is a cell the surrounding table never shows,
        # which is how a quietly-retuned threshold would hide.
        self.assertIn(ft.GAP_THRESHOLD, ft.SECONDARY_THRESHOLDS)

    def test_registration_date_is_a_real_past_date(self):
        d = _dt.date.fromisoformat(ft.REGISTERED_ON)
        self.assertLessEqual(d, _dt.date.today())


class ScopeTests(unittest.TestCase):
    """The discovery sample must be unreachable, whatever the ledger holds."""

    def test_scored_rows_excludes_the_registration_date_itself(self):
        g = ft.scored_rows()
        if g is None:
            self.skipTest("ledger unavailable")
        if len(g):
            # Strictly after -- `>=` would readmit graded rows dated on the
            # registration day, which exist.
            self.assertTrue((g["game_date"].astype(str) > ft.REGISTERED_ON).all())

    def test_report_lines_never_raise_and_always_name_the_prior(self):
        lines = ft.report_lines()
        self.assertTrue(lines)
        joined = " ".join(lines).lower()
        self.assertIn("pre-registered", joined)
        # A reader must never see a running total without being told the
        # registered prior is negative.
        self.assertTrue("negative" in joined or "gate" in joined)

    def test_report_lines_survive_a_ledger_with_no_eligible_rows(self):
        import pandas as pd
        empty = pd.DataFrame(columns=[
            "status", "game_date", "close_p_home", "xw_lean", "xw_net", "home",
            "full_home", "full_away", "close_home_ml", "close_away_ml"])
        lines = ft.report_lines(empty)
        self.assertTrue(any("0" in ln or "no " in ln.lower() for ln in lines))

    def test_report_lines_survive_a_ledger_missing_columns(self):
        import pandas as pd
        lines = ft.report_lines(pd.DataFrame({"status": ["graded"]}))
        self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
