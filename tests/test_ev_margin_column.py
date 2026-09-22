"""The EV column beside the calibration excess on the magnitude x price grid.

No cell values are pinned: the grid is recomputed from the ledger every build.
What is pinned is the arithmetic and the degradation behaviour.
"""
import re
import unittest

import numpy as np

import grade_leans as gl
from market_backfill import breakeven_prob, ev_null


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


class EVNullTests(unittest.TestCase):
    """The EV column's null is PRINTED, because stating the rule was not enough.

    The block already said "EV on minus the hold" and printed the SE beside
    both columns, and a reader still compared `EV +3.0 pp` against zero and
    reported it as null -- when its own null was -2.6 and a market-correct
    simulation put the same cell at P = 0.051. Reading EV against zero credits
    the model with exactly one hold it never earned.

    Nothing here pins a cell value. What is pinned is that the null is on the
    line, that it is DERIVED from the cell's own two references, and that it
    moves to 0.0 on a vig-free book -- so it cannot be a literal.
    """

    def test_the_null_is_minus_the_hold_in_points(self):
        q = np.array([0.50, 0.50])
        be = breakeven_prob([-110, -110])
        self.assertAlmostEqual(ev_null(q, be), float(q.mean() - be.mean()), places=12)
        self.assertLess(ev_null(q, be), 0.0, "a vigged book must give a negative null")

    def test_a_vig_free_book_gives_exactly_zero(self):
        """Derived, not a literal: no hold, no offset."""
        be = breakeven_prob([100, -100])
        self.assertEqual(ev_null(np.array([0.5, 0.5]), be), 0.0)

    def test_unusable_input_answers_nan_rather_than_zero(self):
        """Zero is the one answer it must never invent -- that IS the defect."""
        self.assertTrue(np.isnan(ev_null([0.5], [0.5, 0.5])))
        self.assertTrue(np.isnan(ev_null([0.5, 0.5], [np.nan, 0.5])))
        self.assertTrue(np.isnan(ev_null([], [])))

    def test_every_rendered_EV_carries_its_own_null(self):
        """A rule over the whole block, not one cell.

        Any line holding `EV` must hold a `(null` on the same line. Pinning one
        cell would pass while a later branch printed a bare EV elsewhere.
        """
        lines = gl._magnitude_price_grid_lines(gl.load_ledger())
        ev_lines = [l for l in lines if " EV " in l and "n=" in l]
        self.assertTrue(ev_lines, "the committed ledger renders no EV cells")
        for l in ev_lines:
            self.assertIn("(null", l, f"EV printed with no null: {l!r}")

    def test_the_printed_null_matches_the_derivation(self):
        won = np.array([True, True, False, False])
        q = np.array([0.50, 0.50, 0.50, 0.50])
        be = breakeven_prob([-110, -110, -110, -110])
        line = gl._grid_cell_line("t", won, q, breakeven=be)
        self.assertIn(f"(null {100 * ev_null(q, be):+5.1f})", line)

    def test_EV_minus_its_null_reconciles_to_the_excess_on_every_line(self):
        """The identity that makes the printed null self-checking.

        excess = rate - mean q;  EV = rate - mean be;  null = mean q - mean be.
        So EV - null == excess exactly, and a reader can verify the null from
        the same line rather than trusting it. Asserted by PARSING the rendered
        grid, because the point is that the three numbers agree as printed --
        recomputing them from the frame would test the arithmetic twice and the
        rendering not at all.
        """
        pat = re.compile(r"excess\s+([+-]\d+\.\d)\s+\+-\s+\d+\.\d pp\s+"
                         r"EV\s+([+-]\d+\.\d) pp \(null\s+([+-]\d+\.\d)\)")
        hits = 0
        for line in gl._magnitude_price_grid_lines(gl.load_ledger()):
            m = pat.search(line)
            if not m:
                continue
            hits += 1
            excess, ev, null = (float(x) for x in m.groups())
            self.assertAlmostEqual(
                ev - null, excess, delta=0.11,       # one rounding step on each
                msg=f"EV - null does not reconcile to excess: {line!r}")
        self.assertGreater(hits, 10, "parsed too few cells to be a real check")

    def test_the_copy_forbids_reading_EV_against_zero(self):
        """The claim, not its wording: the block must say zero is wrong here."""
        blob = " ".join(gl._magnitude_price_grid_lines(gl.load_ledger()))
        self.assertIn("NEVER against zero", blob)
        self.assertIn("already the z for BOTH", blob)


if __name__ == "__main__":
    unittest.main()
