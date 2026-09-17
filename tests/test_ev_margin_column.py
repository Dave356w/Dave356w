"""The EV column beside the calibration excess on the magnitude x price grid.

No cell values are pinned: the grid is recomputed from the ledger every build.
What is pinned is the arithmetic and the degradation behaviour.
"""
import unittest

import numpy as np

import grade_leans as gl
from market_backfill import breakeven_prob


class BreakevenProbTests(unittest.TestCase):
    def test_it_is_the_price_a_bet_must_clear(self):
        """breakeven == 1/(1+b) for the payout that price actually returns."""
        for ml in (-350, -175, -120, -105, 105, 120, 175, 350):
            b = (ml / 100.0) if ml > 0 else (100.0 / abs(ml))
            self.assertAlmostEqual(
                breakeven_prob([ml])[0], 1.0 / (1.0 + b), places=12,
                msg=f"breakeven must equal 1/(1+payout) at {ml:+d}")

    def test_the_two_sides_of_a_game_sum_to_more_than_one(self):
        """That excess IS the hold; if it were <= 1 there would be no vig."""
        total = breakeven_prob([-150])[0] + breakeven_prob([130])[0]
        self.assertGreater(total, 1.0)

    def test_an_unusable_price_yields_nan_rather_than_a_guess(self):
        out = breakeven_prob([np.nan, 0, -110])
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[1]))
        self.assertTrue(np.isfinite(out[2]))


class GridCellEVTests(unittest.TestCase):
    def test_ev_is_the_rate_against_breakeven_not_against_q(self):
        won = np.array([True, True, False, False])
        q = np.array([0.50, 0.50, 0.50, 0.50])
        be = np.array([0.52, 0.52, 0.52, 0.52])
        line = gl._grid_cell_line("t", won, q, breakeven=be)
        self.assertIn("excess   +0.0", line)   # .500 realised against .500 devigged
        self.assertIn("EV   -2.0", line)       # ... and -2.0 against a .520 breakeven

    def test_the_column_is_absent_without_a_price_and_present_with_one(self):
        won = np.array([True, False]); q = np.array([0.5, 0.5])
        self.assertNotIn("EV", gl._grid_cell_line("t", won, q))
        self.assertIn("EV", gl._grid_cell_line("t", won, q,
                                               breakeven=np.array([0.52, 0.52])))

    def test_a_partial_price_column_drops_EV_rather_than_changing_n(self):
        """Mixing denominators between the two numbers on one line is the
        harder error to see, so the cell drops EV instead."""
        won = np.array([True, False, True]); q = np.array([0.5, 0.5, 0.5])
        line = gl._grid_cell_line("t", won, q,
                                  breakeven=np.array([0.52, np.nan, 0.52]))
        self.assertIn("n=  3", line)
        self.assertNotIn("EV", line)

    def test_an_empty_cell_still_says_so(self):
        self.assertIn("no rows",
                      gl._grid_cell_line("t", np.array([]), np.array([]),
                                         breakeven=np.array([])))


if __name__ == "__main__":
    unittest.main()
