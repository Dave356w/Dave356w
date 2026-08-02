import unittest

import numpy as np
import pandas as pd

import bp_ablation


def _row(**kw):
    base = dict(
        game_pk=1, game_date="2026-07-29", away="AAA", home="HHH",
        model_tag="xw+plat_consol_v10",
        edge_xwoba_away=0.010, edge_xwoba_home=0.004,
        edge_xwoba_sp_away=0.002, edge_xwoba_sp_home=0.009,
        xw_net=0.006, xw_lean="HHH",
        close_p_home=0.55,
    )
    base.update(kw)
    return base


class DevigTest(unittest.TestCase):
    def test_devig_matches_ledger_close_p_home(self):
        """close_p_home is the proportional de-vig of the two moneylines.

        Locks the convention -- if it ever changes, CLV silently shifts.
        """
        d = pd.read_csv("data/mlb_lean_ledger.csv")
        ph = bp_ablation.ml_to_prob(pd.to_numeric(d["close_home_ml"], errors="coerce"))
        pa = bp_ablation.ml_to_prob(pd.to_numeric(d["close_away_ml"], errors="coerce"))
        recon = bp_ablation.devig(ph, pa)
        shipped = pd.to_numeric(d["close_p_home"], errors="coerce")
        both = recon.notna() & shipped.notna()
        self.assertGreater(both.sum(), 0)
        self.assertLess((recon[both] - shipped[both]).abs().max(), 1e-9)

    def test_one_sided_quote_is_nan_not_the_vigged_price(self):
        """Returning the raw price here would feed the overround into CLV."""
        ph = bp_ablation.ml_to_prob(pd.Series([-200.0, -200.0]))
        pa = bp_ablation.ml_to_prob(pd.Series([np.nan, 170.0]))
        out = bp_ablation.devig(ph, pa)
        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertAlmostEqual(out.iloc[1], ph.iloc[1] / (ph.iloc[1] + pa.iloc[1]))
        self.assertLess(out.iloc[1], ph.iloc[1])   # de-vig removes the overround

    def test_ledger_has_no_one_sided_quotes(self):
        d = pd.read_csv("data/mlb_lean_ledger.csv")
        for h, a in (("open_home_ml", "open_away_ml"),
                     ("f5_open_home_ml", "f5_open_away_ml"),
                     ("close_home_ml", "close_away_ml")):
            H = pd.to_numeric(d[h], errors="coerce").notna()
            A = pd.to_numeric(d[a], errors="coerce").notna()
            self.assertEqual(int((H ^ A).sum()), 0, f"{h}/{a}")

    def test_ml_to_prob_signs(self):
        p = bp_ablation.ml_to_prob(pd.Series([-200.0, 100.0, 150.0]))
        self.assertAlmostEqual(p.iloc[0], 200 / 300)
        self.assertAlmostEqual(p.iloc[1], 0.5)
        self.assertAlmostEqual(p.iloc[2], 100 / 250)


class VariantTest(unittest.TestCase):
    def test_ablation_is_q_equals_one(self):
        """no-BP must be the starter phase alone, not the blended value."""
        d = pd.DataFrame([_row()])
        with_bp, no_bp, ok = bp_ablation.variants(d)
        self.assertTrue(ok.all())
        self.assertAlmostEqual(with_bp.iloc[0], 0.010 - 0.004)
        self.assertAlmostEqual(no_bp.iloc[0], 0.002 - 0.009)
        # the two lean opposite sides -- this row is a disagreement
        self.assertEqual(bp_ablation.side_of(with_bp).iloc[0], "HOME")
        self.assertEqual(bp_ablation.side_of(no_bp).iloc[0], "AWAY")

    def test_reconstruction_drift_aborts(self):
        """A phase/lean mismatch must fail loudly, not report a wrong baseline."""
        d = pd.DataFrame([_row(xw_net=0.999)])
        with self.assertRaises(SystemExit):
            bp_ablation.variants(d)

    def test_shipped_lean_mismatch_aborts(self):
        d = pd.DataFrame([_row(xw_lean="AAA")])
        with self.assertRaises(SystemExit):
            bp_ablation.variants(d)

    def test_exact_zero_is_abstention(self):
        s = bp_ablation.side_of(pd.Series([0.0, 1e-9, -1e-9]))
        self.assertEqual(list(s), ["", "HOME", "AWAY"])

    def test_ledger_reconstruction_matches_shipped_lean(self):
        """The real ledger must survive the assertions on every ablatable row."""
        d = pd.read_csv("data/mlb_lean_ledger.csv")
        with_bp, no_bp, ok = bp_ablation.variants(d)
        self.assertGreater(int(ok.sum()), 0)
        lean = pd.Series(
            np.where(with_bp > 0, d["home"], np.where(with_bp < 0, d["away"], None)),
            index=d.index,
        )
        self.assertTrue((lean[ok] == d["xw_lean"][ok]).all())


class ClvTest(unittest.TestCase):
    def test_away_clv_is_negated_home_move(self):
        """An away lean gains exactly what the home price loses."""
        side = pd.Series(["HOME", "AWAY", ""])
        entry = pd.Series([0.50, 0.50, 0.50])
        close = pd.Series([0.55, 0.55, 0.55])
        v = bp_ablation.clv(side, entry, close)
        self.assertAlmostEqual(v.iloc[0], +0.05)
        self.assertAlmostEqual(v.iloc[1], -0.05)
        self.assertAlmostEqual(v.iloc[2], 0.0)   # abstention takes no position

    def test_agreement_rows_contribute_exactly_zero(self):
        """Paired delta must be structurally zero wherever the sides agree."""
        side = pd.Series(["HOME", "AWAY", ""])
        entry = pd.Series([0.50, 0.60, 0.40])
        close = pd.Series([0.55, 0.52, 0.44])
        a = bp_ablation.clv(side, entry, close)
        b = bp_ablation.clv(side.copy(), entry, close)
        self.assertTrue(((b - a) == 0).all())


class GradeTest(unittest.TestCase):
    def test_grade_from_linescore(self):
        side = pd.Series(["HOME", "AWAY", "HOME", ""])
        away = pd.Series([1.0, 1.0, 3.0, 5.0])
        home = pd.Series([4.0, 4.0, 3.0, 1.0])
        g = bp_ablation.grade(side, away, home)
        self.assertEqual(list(g[:3]), ["W", "L", "T"])
        self.assertTrue(pd.isna(g.iloc[3]))       # no lean, no grade

    def test_record_reproduces_ledger_report(self):
        """with-BP full-game record must match data/ledger_report.txt (39-32).

        This is the cross-check that the replay scores the shipped model, not
        some near-miss of it.
        """
        d = pd.read_csv("data/mlb_lean_ledger.csv")
        with_bp, _, ok = bp_ablation.variants(d)
        side = bp_ablation.side_of(with_bp)
        g = bp_ablation.grade(side,
                              pd.to_numeric(d["full_away"], errors="coerce"),
                              pd.to_numeric(d["full_home"], errors="coerce"))
        w, l, t, _ = bp_ablation.record(g[ok])
        self.assertEqual((w, l, t), (39, 32, 0))

    def test_scoring_agrees_with_shipped_grades(self):
        d = pd.read_csv("data/mlb_lean_ledger.csv")
        with_bp, _, ok = bp_ablation.variants(d)
        side = bp_ablation.side_of(with_bp)
        for a_sc, h_sc, col in (("full_away", "full_home", "xw_full"),
                                ("f5_away", "f5_home", "xw_f5")):
            g = bp_ablation.grade(side,
                                  pd.to_numeric(d[a_sc], errors="coerce"),
                                  pd.to_numeric(d[h_sc], errors="coerce"))
            m = ok & g.notna() & d[col].notna()
            self.assertGreater(int(m.sum()), 0, col)
            self.assertTrue((g[m] == d[col][m]).all(), col)


if __name__ == "__main__":
    unittest.main()
