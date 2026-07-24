"""Card redesign: percentile bars, ledger-ranked lean strength, the
plain-language read, the model-vs-market verdict, and the always-shown model
machinery.

All display-only -- these guard the render layer, not the lean math."""
import unittest

import numpy as np
import pandas as pd

import build_site as b


def _hitter(name, pos, bats, xw, pct, adv=False):
    return dict(name=name, pos=pos, bats=bats, xw=xw, xw_pctile=pct, adv=adv,
                ops=None, pa=0, low=False, mx=None, edge=None)


def _side(p, opp_abbr, pit_xw_pct, kbb_pct, xw_edge, hitters):
    return dict(p=p, t="R", opp_abbr=opp_abbr, pit_xw=.300, pit_k=.24, pit_bb=.07,
                era_season=3.6, xera=3.4, opp_xw=.330, xw_edge=xw_edge,
                pit_xw_pctile=pit_xw_pct, kbb_pctile=kbb_pct, lu_status="posted",
                is_opener=False, has_pl=True, R=5, L=4, S=0, padv=3, pl_fl={},
                hitters=hitters)


def _game(away, home, a, h, odds):
    return dict(away=a, home=h, away_abbr=away, home_abbr=home, away_l10="5-5",
                home_l10="5-5", game_pk=1, game_number=1, game_label="",
                game_datetime_utc=None, time_pt="7:05 PM ET", venue="Park", odds=odds,
                league_baseline={"xwOBA": .312, "K%": .22, "BB%": .08, "OPS": .715, "ERA": 4.1})


class LeanStrengthTests(unittest.TestCase):
    def test_fixed_fallback_buckets(self):
        # No ledger scale -> fixed cutoffs (0.021 / 0.060), no percentile.
        self.assertEqual(b.lean_strength(0.010, None)[0], "slight")
        self.assertEqual(b.lean_strength(0.040, None)[0], "clear")
        self.assertEqual(b.lean_strength(0.090, None)[0], "strong")
        self.assertIsNone(b.lean_strength(0.040, None)[1])

    def test_ranked_against_scale(self):
        scale = np.sort(np.array([round(0.01 * i, 2) for i in range(1, 11)] * 4, float))
        lab, pct = b.lean_strength(0.10, scale)
        self.assertEqual(lab, "strong")
        self.assertGreater(pct, 90)

    def test_none_delta(self):
        self.assertEqual(b.lean_strength(None, None), (None, None))

    def _ledger(self, rows):
        # rows: list of (model_tag, xw_net); all graded.
        return pd.DataFrame([{"model_tag": t, "xw_net": v, "status": "graded"}
                             for t, v in rows])

    def test_scale_excludes_incompatible_units(self):
        # 40 current-scale rows (small, shrunk deltas) + 200 pre-v5 rows (large
        # deltas, 2x scale). The scale must rank against the current units only,
        # never the mixed pool -- so its max stays on the shrunk scale.
        rows = ([("xw+plat_consol_v6", 0.015)] * 40
                + [("xw+plat_consol_v2", 0.040)] * 200)
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v5", "xw+plat_consol_v6")):
            scale = b.lean_strength_scale()
        self.assertEqual(scale.size, 40)
        self.assertLess(scale.max(), 0.02)

    def test_scale_thin_pool_no_allgraded_fallback(self):
        # Too few scale-compatible rows -> None (fixed cutoffs), NOT a fallback
        # to the 200 incompatible pre-v5 rows.
        rows = ([("xw+plat_consol_v6", 0.015)] * 10
                + [("xw+plat_consol_v2", 0.040)] * 200)
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v5", "xw+plat_consol_v6")):
            self.assertIsNone(b.lean_strength_scale())


class PercentileTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 400
        pa = rng.integers(5, 650, n)
        true = rng.normal(.315, .030, n)
        obs = true + rng.normal(0, .35 / np.sqrt(np.maximum(pa, 1)), n)
        self.cust = pd.DataFrame({"xwoba": obs, "pa": pa})

    def test_shrinkage_pulls_small_sample_toward_middle(self):
        ref, _ = b.build_pctile_ref(self.cust, .312, 130.0, 300.0)
        hot_big = b.pctile_rank(.430, 600, ref, .312, 130.0)   # real elite
        hot_tiny = b.pctile_rank(.430, 20, ref, .312, 130.0)   # 20-PA fluke
        self.assertGreater(hot_big, 90)
        self.assertLess(hot_tiny, hot_big - 15)

    def test_pitcher_inversion(self):
        ref, _ = b.build_pctile_ref(self.cust, .312, 130.0, 300.0)
        good = b.pctile_rank(.270, 700, ref, .312, 130.0, invert=True)
        bad = b.pctile_rank(.360, 700, ref, .312, 130.0, invert=True)
        self.assertGreater(good, bad)

    def test_none_inputs_degrade(self):
        self.assertIsNone(b.pctile_rank(None, 100, np.array([.3, .31, .32]), .312, 130.0))
        self.assertIsNone(b.pctile_rank(.3, 100, None, .312, 130.0))


class RenderTests(unittest.TestCase):
    def _cards(self):
        ari = [_hitter(f"A{i}", "LF", "R", .33, 60) for i in range(9)]
        lad = ([_hitter("Betts", "RF", "R", .372, 95),
                _hitter("Ohtani", "DH", "L", .401, 99, adv=True)]
               + [_hitter(f"L{i}", "LF", "R", .33, 60) for i in range(7)])
        a = _side("Glasnow", "ARI", 91, 93, xw_edge=-0.050, hitters=ari)
        h = _side("Gallen", "LAD", 42, 55, xw_edge=+0.042, hitters=lad)
        g_agree = _game("LAD", "ARI", a, h, dict(away_ml=-160, home_ml=135, p_home=.395))
        a2 = _side("Bibee", "NYY", 66, 71, xw_edge=+0.019, hitters=ari)
        h2 = _side("Rodon", "CLE", 35, 62, xw_edge=+0.030, hitters=lad)
        g_dis = _game("CLE", "NYY", a2, h2, dict(away_ml=148, home_ml=-175, p_home=.62))
        return g_agree, g_dis

    def test_casual_card_structure(self):
        g, _ = self._cards()
        html = b.cmb_card(g, "9:00 AM PT", None)
        self.assertIn("class='sl h'", html)          # hitter percentile bar
        self.assertIn("class='sl p'", html)          # starter percentile bar
        self.assertIn("Standouts", html)
        self.assertIn("class='read'", html)
        self.assertIn("class='matchlab'", html)      # matchup-labeled columns
        self.assertIn("LAD bats", html)
        self.assertIn("class='tier", html)

    def test_read_names_the_opposing_starter(self):
        # away offense faces the HOME starter; home offense the AWAY starter.
        g, _ = self._cards()
        read = b.cmb_card(g, "9:00 AM PT", None).split("class='read'")[1].split("</p>")[0]
        self.assertIn("LAD's bats grade", read)
        self.assertIn("against Gallen", read)       # LAD faces home SP Gallen
        self.assertIn("against Glasnow", read)      # ARI faces away SP Glasnow

    def test_model_machinery_renders(self):
        # The model machinery (formerly gated behind an Analyst toggle) is now
        # always part of the card; the .mach classes remain as layout hooks.
        g, _ = self._cards()
        html = b.cmb_card(g, "9:00 AM PT", None)
        self.assertIn("Δxw", html)
        self.assertIn("<span class='mach'>", html)
        self.assertIn("spstats mach", html)
        # The redundant pct-lean suffix and the secondary xwOBA consensus line
        # were removed; the lean pill's Δxw is the single readout.
        self.assertNotIn("pct lean", html)
        self.assertNotIn("xwOBA →", html)

    def test_pitcher_card_shows_xera_not_removed_lenses(self):
        g, _ = self._cards()
        html = b.cmb_card(g, "9:00 AM PT", None)
        self.assertIn("xERA", html)                 # xERA cell present
        self.assertIn("season 3.6", html)           # ...vs season ERA
        # The generic .mach display hook must not stack the three starter
        # stat cells; its flex override belongs later in the cascade.
        self.assertIn(".spstats.mach{display:flex}", b.CSS)
        self.assertGreater(b.CSS.index(".spstats.mach{display:flex}"),
                           b.CSS.index(".mach{display:block}"))
        self.assertNotIn("xw edge (drives lean)", html.lower())
        self.assertNotIn("class='agg mach'", html)
        self.assertNotIn("OPS alwd", html)          # xOPS-against removed
        self.assertNotIn("xOPS edge", html)         # xOPS edge removed
        self.assertNotIn("pythag", html)            # pythag control removed
        self.assertNotIn("DK F5", html)             # F5 odds removed

    def test_verdict_agree_and_disagree(self):
        g_agree, g_dis = self._cards()
        self.assertIn("agrees with the market", b.cmb_card(g_agree, "x", None))
        dis = b.cmb_card(g_dis, "x", None)
        self.assertIn("verdict edge", dis)
        self.assertIn("underdog", dis)

    def test_verdict_shows_context_record_when_available(self):
        # ctx = market_context_records(): (lean side, agree/disagree) -> 'W-L'.
        ctx = {("home", "agree"): "30-27", ("away", "disagree"): "27-30"}
        # Agree: model favors the home side, which is also the market favorite.
        agree = b._verdict_html("ARI", dict(p_home=.62), "LAD", "ARI", ctx)
        self.assertIn("home favorite", agree)
        self.assertIn("30-27", agree)
        self.assertNotIn("No edge on the line", agree)
        # Disagree: model leans the away underdog against a home market favorite.
        dis = b._verdict_html("LAD", dict(p_home=.62, away_ml=140), "LAD", "ARI", ctx)
        self.assertIn("away underdog", dis)
        self.assertIn("27-30", dis)
        self.assertNotIn("record is built to test", dis)
        # Missing bucket -> prose fallback (no fabricated record).
        fb = b._verdict_html("ARI", dict(p_home=.62), "LAD", "ARI", {})
        self.assertIn("No edge on the line", fb)

    def test_no_lean_pill_when_edge_missing(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = None
        self.assertIn("no lean", b.cmb_card(g, "x", None))

    def test_exact_zero_delta_is_neutral_not_home(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = .012345
        g["home"]["xw_edge"] = .012345
        html = b.cmb_card(g, "x", None)
        self.assertIn("no lean", html)
        self.assertIn("Δxw 0.000", html)
        self.assertNotIn("<span class='lt'>ARI</span>", html)

    def test_nonzero_sub_display_delta_is_not_rendered_as_zero(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = .01234567891
        g["home"]["xw_edge"] = .01234567890
        html = b.cmb_card(g, "x", None)
        self.assertIn("Δxw &lt;0.001", html)
        self.assertNotIn("no lean", html)


if __name__ == "__main__":
    unittest.main()
