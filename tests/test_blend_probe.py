"""The blend probe's claims, asserted as properties rather than as numbers.

The probe's whole finding rests on two things being true: that rebuilding an
arm from its persisted components reproduces what that arm published, and that
the blend's apparent gain is the noise-averaging identity.  Both are pinned
here.  No correlation, record or CI is frozen -- those move with every slate
the bot grades, and freezing one would be the constants-from-data defect this
repo has recorded repeatedly.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blend_probe as bp
import shadow_report as sr


def _committed_pairs():
    return [(d, s, p) for d, s, p in sr._slate_dates()]


class ReconstructionTests(unittest.TestCase):
    """Rebuilding a shipped arm from its own components must be exact."""

    def test_every_committed_dump_rebuilds_its_own_published_edge(self):
        checked = 0
        for _, spath, ppath in _committed_pairs():
            for path in (spath, ppath):
                df = pd.read_csv(path, float_precision="round_trip")
                if "sp_share" not in df.columns:
                    continue
                err, n = bp.reconstruction_error(df)
                if not n:
                    continue
                checked += n
                self.assertEqual(
                    err, 0.0,
                    f"{os.path.basename(path)}: rebuilt edge_xwOBA differs from "
                    f"the published column by {err:.3e}; the blend built the "
                    "same way is not the build either")
        self.assertGreater(checked, 0, "no committed dump carried phase columns")

    def test_the_probe_refuses_to_report_on_an_inexact_rebuild(self):
        """The check is load-bearing, so it must stop the report, not warn."""
        real = bp.paired_arms

        def broken(*a, **k):
            f, diag = real(*a, **k)
            return f, {**diag, "reconstruction_error": 1e-9}

        bp.paired_arms = broken
        try:
            with self.assertRaises(SystemExit):
                bp.report(n_boot=2)
        finally:
            bp.paired_arms = real


class BlendArithmeticTests(unittest.TestCase):
    """The identity the finding turns on, and the endpoints of the sweep."""

    def test_the_ceiling_formula_matches_a_brute_force_average(self):
        rng = np.random.default_rng(7)
        for rho_target in (0.5, 0.84, 0.95):
            n = 200_000
            y = rng.normal(size=n)
            a = 0.15 * y + rng.normal(size=n)
            base = rng.normal(size=n)
            b = (rho_target * a / np.std(a)
                 + np.sqrt(max(1 - rho_target ** 2, 0.0)) * base)
            za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
            r_a, r_b = bp._corr(za, y), bp._corr(zb, y)
            rho = bp._corr(za, zb)
            pred, _ = bp.blend_ceiling(r_a, r_b, rho)
            self.assertAlmostEqual(
                pred, bp._corr(za + zb, y), places=6,
                msg=f"identity disagreed with the actual average at rho={rho:.3f}")

    def test_the_ceiling_is_zero_when_the_arms_are_identical(self):
        """rho = 1 means the average IS the arm, so there is no headroom."""
        _, head = bp.blend_ceiling(0.15, 0.15, 1.0)
        self.assertAlmostEqual(head, 0.0, places=12)

    def test_headroom_grows_as_the_arms_decorrelate(self):
        heads = [bp.blend_ceiling(0.15, 0.15, r)[1] for r in (0.95, 0.84, 0.5)]
        self.assertEqual(heads, sorted(heads),
                         "a lower rho must leave MORE room, not less")

    def test_the_sweep_endpoints_reproduce_the_two_shipped_arms(self):
        """lambda 0 and 1 are not blends; they must be the arms themselves."""
        f0, _ = bp.paired_arms(lam=0.0)
        f1, _ = bp.paired_arms(lam=1.0)
        for f, col, lab in ((f0, "net_x", "lambda=0 against the xwOBA arm"),
                            (f1, "net_w", "lambda=1 against the wOBA arm")):
            m = f["net_b"].notna() & f[col].notna()
            self.assertTrue(m.any(), f"no overlapping rows for {lab}")
            self.assertLess(float((f.loc[m, "net_b"] - f.loc[m, col]).abs().max()),
                            1e-12, lab)


class ScopeTests(unittest.TestCase):
    """It is a probe: it reads committed artifacts and ships nothing."""

    def test_it_declares_no_model_tag_and_no_registered_constant(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "blend_probe.py"), encoding="utf-8").read()
        for banned in ("MODEL_TAG =", "REGISTERED_ON", "GATE_"):
            self.assertNotIn(banned, src,
                             f"{banned} belongs to a registration, not a probe")

    def test_it_never_writes(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "blend_probe.py"), encoding="utf-8").read()
        for banned in ("to_csv", "open(", "os.remove", "makedirs"):
            self.assertNotIn(banned, src, f"a probe must not {banned}")

    def test_the_probe_stamps_no_tag_of_its_own_into_any_family_map(self):
        """A probe's reconstructed arm shares a record line with nothing.

        RESTATED. This used to ban the substring "blend" from the family maps,
        which was a correct spelling of the rule only while no shipped model
        had that word in its tag. v13 does -- `xw+starter_blend_v13` is a real
        prediction family with a real record line -- so the substring test now
        forbids a legitimate entry and says nothing about the probe.

        The property was never about the word. It is that THIS MODULE defines
        no tag and contributes no row to any family, so a reconstruction it
        builds can never be pooled into a record or a delta scale. That is
        asserted directly, and it holds whatever a shipped tag is called.
        """
        import build_site as bs
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "blend_probe.py"),
            encoding="utf-8").read()
        self.assertNotIn("MODEL_TAG =", src)
        for m in (bs._RECORD_FAMILIES, bs._SCALE_FAMILIES):
            for tag, fam in m.items():
                self.assertNotIn("shadow", str(tag).lower())
                for t in fam:
                    self.assertNotIn("shadow", str(t).lower())


class SignCriterionTests(unittest.TestCase):
    """McNemar over lean calls: the criterion the record actually uses."""

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows, columns=["net_a", "net_b", "margin"])

    def test_concordant_games_cancel(self):
        """Games both arms call the same way carry no information and must not
        move the test -- that is what makes it paired."""
        base = [(+0.02, -0.02, +1), (-0.02, +0.02, -1), (+0.03, -0.03, +2)]
        a1, b1, z1, _ = bp.sign_contrast(self._frame(base), "net_a", "net_b")
        padded = base + [(+0.05, +0.05, +3)] * 40 + [(-0.05, -0.05, -3)] * 40
        a2, b2, z2, _ = bp.sign_contrast(self._frame(padded), "net_a", "net_b")
        self.assertEqual((a1, b1), (a2, b2))
        self.assertEqual(z1, z2)

    def test_an_arm_that_wins_every_disagreement_scores_positive(self):
        rows = [(+0.02, -0.02, +1)] * 9
        a, b, z, p = bp.sign_contrast(self._frame(rows), "net_a", "net_b")
        self.assertEqual((a, b), (9, 0))
        self.assertGreater(z, 0)
        self.assertLess(p, 0.01)

    def test_it_is_antisymmetric_in_its_two_arms(self):
        rows = [(+0.02, -0.02, +1)] * 7 + [(-0.02, +0.02, +1)] * 3
        a, b, z, p = bp.sign_contrast(self._frame(rows), "net_a", "net_b")
        b2, a2, z2, p2 = bp.sign_contrast(self._frame(rows), "net_b", "net_a")
        self.assertEqual((a, b), (a2, b2))
        self.assertAlmostEqual(z, -z2)
        self.assertAlmostEqual(p, p2)

    def test_a_tied_game_is_excluded_rather_than_scored(self):
        """A 0-run margin has no winner, so it cannot grade either arm."""
        rows = [(+0.02, -0.02, 0)] * 8
        a, b, z, p = bp.sign_contrast(self._frame(rows), "net_a", "net_b")
        self.assertEqual((a, b), (0, 0))
        self.assertTrue(np.isnan(z) and np.isnan(p))

    def test_no_disagreement_yields_no_verdict_rather_than_zero(self):
        rows = [(+0.02, +0.02, +1)] * 6
        _, _, z, p = bp.sign_contrast(self._frame(rows), "net_a", "net_b")
        self.assertTrue(np.isnan(z) and np.isnan(p),
                        "an empty discordant set is unmeasured, not a tie")


if __name__ == "__main__":
    unittest.main()
