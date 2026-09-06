"""The lineup-window probe pins its ARITHMETIC and its PAIRING, never a number.

`ledger_report.txt`'s component slopes move every time the Actions bot grades a
slate, so an assertion on -0.83 would go red for a reason with nothing to do
with this probe -- the anti-pattern CLAUDE.md files under "freezing a measured
number into a test". What is asserted instead is that the probe's unconditioned
fit IS the report's, computed from `actuals_backfill`'s own pairing, so the
conditioned fit beside it is a like-for-like comparison whatever the numbers do.
"""
import unittest

import numpy as np
import pandas as pd

import actuals_backfill as ab
import lineup_window_probe as lw


class OlsTests(unittest.TestCase):
    def test_single_regressor_ols_equals_calibration(self):
        """The probe's multi-regressor OLS must reduce EXACTLY to the estimator
        the report already uses; otherwise "the slope moved" could be two
        estimators disagreeing rather than the covariate doing anything."""
        rng = np.random.default_rng(0)
        x = rng.normal(0.31, 0.02, 200)
        y = 0.4 * x + rng.normal(0, 0.05, 200)
        got = lw.ols(y, [x], ["x"])
        want = ab.calibration(x, y)
        self.assertAlmostEqual(got["pred" if "pred" in got else "x"][0],
                               want["slope"], places=10)
        self.assertAlmostEqual(got["x"][1], want["se_slope"], places=10)

    def test_ols_returns_none_when_underdetermined(self):
        self.assertIsNone(lw.ols([1.0, 2.0], [[1.0, 2.0]], ["x"]))


def _row(**kw):
    """One ledger row with both starters' allowed lines and both batting lines."""
    base = {"model_tag": "T"}
    for side in ("away", "home"):
        for f, v in (("ab", 30), ("h", 8), ("2b", 2), ("3b", 0), ("hr", 1),
                     ("bb", 3), ("ibb", 0), ("hbp", 0), ("sf", 0)):
            base[f"act_sp_{f}_{side}"] = v
        for f, v in (("pa", 38), ("ab", 34), ("h", 9), ("2b", 2), ("3b", 0),
                     ("hr", 1), ("bb", 3), ("ibb", 0), ("hbp", 0), ("sf", 0)):
            base[f"act_{f}_{side}"] = v
        base[f"act_sp_bf_{side}"] = 24.0
        base[f"starter_xwoba_{side}"] = 0.310
        base[f"opp_xwoba_neutral_{side}"] = 0.320
        base[f"act_woba_{side}"] = 0.300
    base.update(kw)
    return base


class PairingTests(unittest.TestCase):
    def test_lineup_actual_is_crossed_and_sp_actual_is_not(self):
        """A reversed lineup cross yields a plausible near-zero slope rather
        than an error, so it is pinned on a row whose two sides differ."""
        f = lw.frame(pd.DataFrame([_row(act_woba_away=0.111, act_woba_home=0.999)]),
                     tags=None)
        lu = f[f.component == "lineup"].set_index("side")["act"]
        # opp_xwoba_neutral_away describes the offense the AWAY staff faces,
        # which is the HOME bats.
        self.assertAlmostEqual(lu["away"], 0.999)
        self.assertAlmostEqual(lu["home"], 0.111)
        sp = f[f.component == "SP"]
        # The SP actual comes from the starter's own allowed line, so it is
        # unaffected by either team's whole-game batting rate.
        self.assertEqual(len(sp), 2)
        self.assertTrue((sp["act"] > 0.2).all() and (sp["act"] < 0.6).all())

    def test_window_column_keys_on_the_pitching_side(self):
        """`bf` is the start's length and both components key it on the pitching
        side -- the lineup's ACTUAL takes the cross, its WINDOW does not."""
        f = lw.frame(pd.DataFrame([_row(act_sp_bf_away=11.0, act_sp_bf_home=27.0)]),
                     tags=None)
        for comp in ("lineup", "SP"):
            s = f[f.component == comp].set_index("side")["bf"]
            self.assertAlmostEqual(s["away"], 11.0)
            self.assertAlmostEqual(s["home"], 27.0)

    def test_tags_filter_selects_the_family(self):
        df = pd.DataFrame([_row(model_tag="A"), _row(model_tag="B")])
        self.assertEqual(len(lw.frame(df, tags=("A",))), 4)   # 2 sides x 2 comps
        self.assertEqual(len(lw.frame(df, tags=("A", "B"))), 8)


class ReproducesTheReportTests(unittest.TestCase):
    """The probe's unconditioned numbers are the report's, on the live ledger."""

    def setUp(self):
        try:
            self.led = pd.read_csv(lw.LEDGER)
        except FileNotFoundError:  # pragma: no cover
            self.skipTest("no committed ledger")

    def test_unconditioned_slopes_match_paired_components(self):
        from build_site import RECORD_TAGS
        mine = lw.frame(self.led, RECORD_TAGS)
        theirs = ab.paired_components(
            self.led[self.led["model_tag"].astype(str).isin(set(RECORD_TAGS))])
        if theirs.empty:
            self.skipTest("no paired component rows in this family")
        for comp in ("lineup", "SP"):
            a = mine[mine.component == comp]
            b = theirs[theirs.component == comp]
            self.assertEqual(len(a), len(b), f"{comp} row count")
            ca = ab.calibration(a["pred"], a["act"])
            cb = ab.calibration(b["pred"], b["act"])
            self.assertAlmostEqual(ca["slope"], cb["slope"], places=10,
                                   msg=f"{comp} slope")
            self.assertAlmostEqual(ca["se_slope"], cb["se_slope"], places=10,
                                   msg=f"{comp} se")


if __name__ == "__main__":
    unittest.main()
