"""K% probe: the invariants that decide whether its numbers can be believed.

The probe measures rather than predicts, so these guard construction, not
performance. Three of them are the corrections that separate this from a naive
K% layer: log5 must be identity against a league-average pitcher (or the
starter's K% is counted twice, once here and once in his own xwOBA-allowed),
the contact residual must never go negative for a high-K starter against a
high-walk lineup, and the decomposition must round-trip -- the probe's whole
method is inverting a blended xwOBA it never observes the components of.
"""
import unittest

import numpy as np

import k_interaction_probe as kp


class Log5Test(unittest.TestCase):
    def test_identity_when_all_three_match(self):
        for p in (0.10, 0.222, 0.35):
            self.assertAlmostEqual(kp.log5(p, p, p), p, places=9)

    def test_league_average_pitcher_is_identity(self):
        """The correction that keeps the starter's K% from being double-counted."""
        lam = 0.222
        for p_bat in (0.12, 0.20, 0.222, 0.31):
            self.assertAlmostEqual(kp.log5(p_bat, lam, lam), p_bat, places=9)

    def test_league_average_batter_is_identity(self):
        lam = 0.222
        for p_pit in (0.14, 0.222, 0.33):
            self.assertAlmostEqual(kp.log5(lam, p_pit, lam), p_pit, places=9)

    def test_monotone_increasing_in_both_arguments(self):
        lam = 0.222
        grid = np.linspace(0.05, 0.50, 40)
        prev = -1.0
        for b in grid:
            v = kp.log5(b, 0.26, lam)
            self.assertGreater(v, prev)
            prev = v
        prev = -1.0
        for q in grid:
            v = kp.log5(0.19, q, lam)
            self.assertGreater(v, prev)
            prev = v

    def test_damp_one_is_the_closed_form_odds_ratio(self):
        b, q, lam = 0.27, 0.185, 0.222
        r = ((b * q) / lam) / (((1 - b) * (1 - q)) / (1 - lam))
        self.assertAlmostEqual(kp.log5(b, q, lam, damp=1.0), r / (1 + r), places=12)


class LambdaReferenceTest(unittest.TestCase):
    """Guards `lg_K%` against a "fix" that would introduce a bias.

    An earlier revision of docs/k_interaction_phase0.md filed the overall
    league K% at build_site.py:2866-2869 as a bug, on the grounds that
    `mx_K%` divides a *starter's* K% by an *all-pitcher* rate. That was wrong,
    and the proposed replacement -- an SP-only lambda -- is the actual error.
    These tests encode why, because the argument for it is superficially
    convincing and will be made again.
    """

    LG = 0.2213   # overall league K%: all-pitcher == all-batter, symmetric

    def test_lambda_is_defined_by_the_neutral_opponent_identity(self):
        """log5(b, L, L) == b forces what L must be: the average *pitcher*."""
        for b in np.linspace(0.12, 0.34, 12):
            self.assertAlmostEqual(kp.log5(b, self.LG, self.LG), b, places=12)

    def test_average_starter_must_come_in_below_the_season_rate(self):
        """A season K% is vs all pitchers, relievers included, and relievers
        strike out more. So an average *starter* must suppress it, not leave
        it alone. Overall-lambda does this; SP-only-lambda does not.
        """
        b, sp_only = 0.2129, 0.2050          # lineup rate, avg-SP rate
        correct = kp.log5(b, sp_only, self.LG)
        wrong = kp.log5(b, sp_only, sp_only)
        self.assertLess(correct, b)
        self.assertAlmostEqual(wrong, b, places=12)

    def test_sp_only_lambda_inflates_every_estimate(self):
        """Quantifies the damage: +1.3 to +2.5pp on a realistic matchup."""
        b, p = 0.2129, 0.2600
        base = kp.log5(b, p, self.LG)
        for sp_only in (0.200, 0.205, 0.210):
            self.assertGreater(kp.log5(b, p, sp_only) - base, 0.012)

    def test_probable_starters_do_not_run_below_league(self):
        """The empirical premise was false too. Measured over the committed
        slates, probable SPs sit *above* the stored league rate, because
        probables are a selected population. "Bullpen K% runs high" is true of
        the reliever pool and does not transfer to the arms the model faces.
        """
        try:
            df = kp.load_sides()
        except SystemExit:
            self.skipTest("no slate CSVs available")
        self.assertGreater(df["p_k"].mean(), df["lg_k"].mean())


class BbeShareTest(unittest.TestCase):
    def test_never_negative_over_the_full_cross_product(self):
        """The specific failure the multiplicative form exists to prevent.

        A subtractive 1 - k - bb - hbp goes negative in the high-K/high-walk
        corner of this grid; this form cannot.
        """
        for k in np.linspace(0.0, 0.60, 25):
            for bb in np.linspace(0.0, 0.30, 16):
                for hbp in np.linspace(0.0, 0.05, 6):
                    v = kp.bbe_share(k, bb, hbp)
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    def test_brief_grid_does_not_actually_reach_the_negative_corner(self):
        """Premise correction, recorded rather than asserted away.

        The brief motivates the multiplicative form by a subtractive
        1 - k - bb - hbp going negative "for a high-K SP against a high-walk
        batter", and asks for that to be checked over
        K in [0,.60] x BB in [0,.30] x HBP in [0,.05]. On that grid it cannot
        happen: the corner sums to 0.95, so the subtractive form bottoms out at
        +0.05. Going negative needs k+bb+hbp > 1, outside the stated range and
        outside anything baseball produces.

        The multiplicative form is still the right choice -- it is closed under
        [0,1] for every input, with no range precondition to remember -- but it
        is chosen for robustness, not because the cited failure occurs here.
        """
        k, bb, hbp = 0.60, 0.30, 0.05
        self.assertAlmostEqual(1.0 - k - bb - hbp, 0.05, places=9)
        self.assertGreater(kp.bbe_share(k, bb, hbp), 0.0)
        # Outside the grid, the subtractive form does fail and this one does not.
        self.assertLess(1.0 - 0.70 - 0.35 - 0.05, 0.0)
        self.assertGreaterEqual(kp.bbe_share(0.70, 0.35, 0.05), 0.0)


class ReconstructionTest(unittest.TestCase):
    def test_implied_xwobacon_round_trips(self):
        """The probe's method: it never observes xwOBAcon, it inverts for it."""
        for k, bb, con in [(0.18, 0.075, 0.352), (0.222, 0.085, 0.3643),
                           (0.31, 0.11, 0.402)]:
            x = kp.xwoba_from_branches(k, bb, kp.LG_HBP, con)
            self.assertAlmostEqual(kp.implied_xwobacon(x, k, bb), con, places=9)

    def test_zero_k_reproduces_the_unmodulated_blend(self):
        bb, hbp, con = 0.085, 0.011, 0.370
        self.assertAlmostEqual(
            kp.xwoba_from_branches(0.0, bb, hbp, con),
            bb * kp.W_BB + hbp * kp.W_HBP + (1 - bb - hbp) * con, places=12)

    def test_k_suppresses_xwoba_at_the_expected_rate(self):
        """Sets the error budget: ~-3.3 pts per 1pp at league contact quality."""
        d = (kp.xwoba_from_branches(0.21, 0.085, kp.LG_HBP, 0.370)
             - kp.xwoba_from_branches(0.20, 0.085, kp.LG_HBP, 0.370)) * 1000
        self.assertLess(d, -3.0)
        self.assertGreater(d, -3.7)


class MeasurementTest(unittest.TestCase):
    """Guards the two reported statistics against silent breakage."""

    def test_composition_residual_is_bounded_by_raw_spread(self):
        rng = np.random.default_rng(3)
        import pandas as pd
        n = 400
        x = rng.normal(0.323, 0.010, n)
        df = pd.DataFrame({"b_k": 0.212 + 0.5 * (x - 0.323) + rng.normal(0, 0.02, n),
                           "b_xwoba": x})
        c = kp.composition_residual(df)
        self.assertLessEqual(c["k_sd_resid"], c["k_sd_raw"] + 1e-12)
        self.assertGreaterEqual(c["variance_explained"], 0.0)
        self.assertLessEqual(c["variance_explained"], 1.0)

    def test_perfectly_predictable_k_leaves_no_residual(self):
        """If K% were fully determined by blended xwOBA, the layer has no room."""
        import pandas as pd
        x = np.linspace(0.30, 0.35, 50)
        df = pd.DataFrame({"b_k": 0.5 - x, "b_xwoba": x})
        c = kp.composition_residual(df)
        self.assertAlmostEqual(c["k_sd_resid"], 0.0, places=9)
        self.assertAlmostEqual(c["variance_explained"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
