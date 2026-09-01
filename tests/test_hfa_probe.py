"""Structural invariants of the HFA-in-the-lean probe.

No measured record, shift or verdict is frozen here: the ledger grows and the
numbers move. What is pinned is the reasoning the probe has to keep doing --
fitting inside one xw_net scale, refusing to score its own fit in-sample, and
printing the controls that make its answer readable.
"""

import unittest

import numpy as np
import pandas as pd

import build_site
import hfa_probe as hp


def _frame(n=200, seed=0, tag=None, hfa=0.0, slope=20.0):
    """Synthetic graded rows whose home-win outcomes follow a known logit."""
    rng = np.random.default_rng(seed)
    tag = tag or build_site.SCALE_TAGS[-1]
    net = rng.normal(0, 0.02, n)
    p = 1 / (1 + np.exp(-(hfa + slope * net)))
    home_won = rng.random(n) < p
    dates = pd.to_datetime("2026-07-01") + pd.to_timedelta(
        np.repeat(np.arange(n // 10 + 1), 10)[:n], unit="D")
    return pd.DataFrame({
        "model_tag": tag, "status": "graded",
        "game_date": dates.astype(str), "game_pk": np.arange(n),
        "xw_lean": np.where(net > 0, "H", "A"), "xw_net": net,
        "home": "H", "away": "A",
        "full_home": np.where(home_won, 3, 1),
        "full_away": np.where(home_won, 1, 3),
        "close_p_home": np.clip(0.5 + net * 5, .05, .95),
        "close_home_ml": -120, "close_away_ml": 110,
    })


class ScopeTests(unittest.TestCase):
    def test_only_one_xw_net_scale_is_fitted(self):
        """`h = a/b`, so mixing scales attenuates b and INFLATES the shift.

        This is the trap the probe's header documents: pooling every family
        made the correction look 2.4x larger than it is. A tag outside
        SCALE_TAGS must never reach the fit.
        """
        d = pd.concat([_frame(120, seed=1),
                       _frame(120, seed=2, tag="woba+plat_consol_v5")],
                      ignore_index=True)
        g = hp.load(d)
        self.assertEqual(len(g), 120)
        self.assertTrue(g["model_tag"].isin(build_site.SCALE_TAGS).all())

    def test_the_scale_family_is_read_live_not_pinned(self):
        """A bump must carry the probe forward with no edit.

        `interaction_probe` scored a lineage the build had stopped running for
        two versions because its row selector was a hardcoded tag list. A
        constant is not only a number: the set of rows a probe reads is one.
        """
        src = open(hp.__file__).read()
        self.assertIn("build_site.SCALE_TAGS", src)
        for tag in ("xw+plat_consol_v12", "xw+plat_consol_v10"):
            self.assertNotIn(f'"{tag}"', src)

    def test_exact_zero_deltas_are_dropped(self):
        """v7 made zero an abstention; a zero row has no side to assign."""
        d = _frame(60, seed=3)
        d.loc[0, "xw_net"] = 0.0
        self.assertEqual(len(hp.load(d)), 59)

    def test_load_survives_a_ledger_it_cannot_use(self):
        self.assertIsNone(hp.load(pd.DataFrame({"status": ["graded"]})))
        empty = _frame(10, seed=4, tag="woba+plat_consol_v5")
        self.assertTrue(hp.load(empty).empty)


class ArithmeticTests(unittest.TestCase):
    def test_the_shift_recovers_a_known_home_field_term(self):
        """h = a/b must invert the logit it was defined from."""
        g = hp.load(_frame(4000, seed=5, hfa=0.40, slope=20.0))
        h, a, _, b, _ = hp.fit_shift(g)
        self.assertAlmostEqual(a, 0.40, delta=0.15)
        self.assertAlmostEqual(b, 20.0, delta=4.0)
        self.assertAlmostEqual(h, 0.40 / 20.0, delta=0.008)

    def test_no_home_field_yields_a_shift_near_zero(self):
        g = hp.load(_frame(4000, seed=6, hfa=0.0, slope=20.0))
        h, _, _, _, _ = hp.fit_shift(g)
        self.assertLess(abs(h), 0.006)

    def test_a_flip_only_happens_below_the_shift(self):
        """A constant shift moves the decision exactly where |xw_net| < |h|.

        This is the probe's mechanical explanation for why the correction
        cannot help much: it only ever touches the weakest leans.
        """
        g = hp.load(_frame(400, seed=7, hfa=0.5))
        h, *_ = hp.fit_shift(g)
        net = g["xw_net"].to_numpy(float)
        flipped = (net > 0) != (net + h > 0)
        self.assertTrue((np.abs(net[flipped]) < abs(h) + 1e-12).all())
        self.assertFalse(flipped[np.abs(net) > abs(h)].any())


class HonestyTests(unittest.TestCase):
    def test_the_walk_forward_never_scores_a_row_it_fitted_on(self):
        """The whole value of the design. A slate's h must come from strictly
        earlier slates, or the probe is grading its own fit."""
        g = hp.load(_frame(300, seed=8, hfa=0.4))
        w = hp.walk_forward(g)
        self.assertFalse(w.empty)
        for d, sub in w.groupby("game_date"):
            prior = g[g["game_date"] < d]
            self.assertGreaterEqual(len(prior), hp.MIN_FIT)
            expected, *_ = hp.fit_shift(prior)
            self.assertTrue(np.allclose(sub["h"].to_numpy(float), expected))

    def test_the_report_prints_both_designs_and_both_controls(self):
        """A per-slate refit alone cannot tell an effect from fitting noise,
        and a record with no control beside it is unreadable."""
        lines = hp.report_lines(_frame(300, seed=9, hfa=0.4))
        joined = " ".join(lines)
        self.assertIn("WALK-FORWARD", joined)
        self.assertIn("FIXED-h HOLDOUT", joined)
        self.assertIn("always home (control)", joined)
        self.assertIn("always chalk (control)", joined)
        self.assertIn("GATE", joined)
        self.assertIn("STABILITY", joined)

    def test_the_report_never_recommends_shipping(self):
        """Same rule the published surfaces follow: report, never recommend."""
        joined = " ".join(hp.report_lines(_frame(300, seed=10))).lower()
        for banned in ("should ship", "recommend", "bump model_tag",
                       "we should", "clear win"):
            self.assertNotIn(banned, joined)

    def test_report_lines_never_raise_on_a_thin_or_broken_ledger(self):
        for d in (None, pd.DataFrame(), _frame(5, seed=11),
                  pd.DataFrame({"status": ["graded"]})):
            lines = hp.report_lines(d)
            self.assertTrue(lines)
            self.assertIn("HFA-in-the-lean probe", lines[0])


if __name__ == "__main__":
    unittest.main()
