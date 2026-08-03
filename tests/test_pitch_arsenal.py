"""Pitch-mix arm: the invariants that decide whether it may ever drive a lean.

The arm is shadow-only today, so these guard construction rather than
performance. Two of them are the corrections that separate this from the naive
"weight batter pitch-type wOBA by usage rates" version: a league-average
hitter must return exactly 1.0 against any arsenal (or the arsenal's mix
effect is counted twice, once here and once in the starter's own
wOBA-allowed), and a hitter with no cells must land on 1.0 continuously
rather than at a coverage threshold.
"""
import unittest

import numpy as np
import pandas as pd

import pitch_arsenal as pa


def _arsenal(rows):
    """rows: (player_id, pitch_type, pa, xwoba, usage)."""
    return pd.DataFrame(rows, columns=["player_id", "Pitch Type", "PA",
                                       "woba", "pitch_usage"])


class MetricSourceTests(unittest.TestCase):
    def test_shadow_arm_requires_observed_woba(self):
        self.assertEqual(pa._MODEL_WOBA_COLS, ("woba",))
        # An xwOBA-only export must abstain instead of silently contaminating
        # one arm of the wOBA forward test.
        xw_only = LEAGUE.rename(columns={"woba": "xwoba"})
        self.assertTrue(pa.normalize_arsenal(xw_only).empty)


# A two-pitch league: sliders suppress offense, four-seamers do not. This is
# the gap that double-counts if the multiplier is not league-relative.
LEAGUE = _arsenal([
    (1, "FF", 1000, .350, 55.0),
    (1, "SL", 1000, .270, 45.0),
    (2, "FF", 1000, .350, 55.0),
    (2, "SL", 1000, .270, 45.0),
])


class NormalizeTests(unittest.TestCase):
    def test_column_aliases_and_types(self):
        tidy = pa.normalize_arsenal(LEAGUE)
        self.assertEqual(list(tidy.columns),
                         ["player_id", "pitch_type", "pa", "xwoba", "usage"])
        self.assertEqual(tidy["player_id"].dtype.kind, "i")
        self.assertEqual(set(tidy["pitch_type"]), {"FF", "SL"})

    def test_missing_columns_degrade_to_empty(self):
        # A renamed leaderboard column must abstain, not fabricate averages.
        self.assertTrue(pa.normalize_arsenal(
            pd.DataFrame({"nope": [1]})).empty)
        self.assertTrue(pa.normalize_arsenal(None).empty)

    def test_league_by_pitch_type_is_pa_weighted(self):
        tidy = pa.normalize_arsenal(_arsenal([
            (1, "FF", 300, .400, 50.0), (2, "FF", 100, .200, 50.0)]))
        lg = pa.league_by_pitch_type(tidy)
        self.assertAlmostEqual(lg["FF"], (300 * .400 + 100 * .200) / 400)


class MultiplierTests(unittest.TestCase):
    def setUp(self):
        self.tidy = pa.normalize_arsenal(LEAGUE)
        self.lg = pa.league_by_pitch_type(self.tidy)

    def _cells(self, rows, k=0.0):
        cells, overall = pa.batter_relative_cells(
            pa.normalize_arsenal(_arsenal(rows)), self.lg, k=k)
        return cells, overall

    def test_league_average_hitter_is_neutral_against_any_arsenal(self):
        # Hits both pitch types exactly at league -> 1.0 whether the starter is
        # fastball-heavy or slider-heavy. Without the league-relative ratio the
        # slider-heavy arsenal would score him lower on mix alone.
        cells, overall = self._cells([(9, "FF", 400, .350, 0), (9, "SL", 400, .270, 0)])
        for weights in ({"FF": .9, "SL": .1}, {"FF": .1, "SL": .9}):
            m, cov = pa.batter_mix_multiplier(9, weights, cells, overall)
            self.assertAlmostEqual(m, 1.0, places=6)
            self.assertAlmostEqual(cov, 1.0, places=6)

    def test_pitch_specific_weakness_survives_general_ability(self):
        # Equally above league on both pitches -> no pitch-specific signal.
        flat, ov_flat = self._cells([(3, "FF", 400, .385, 0), (3, "SL", 400, .297, 0)])
        m_flat, _ = pa.batter_mix_multiplier(3, {"FF": .5, "SL": .5}, flat, ov_flat)
        self.assertAlmostEqual(m_flat, 1.0, places=6)
        # Same overall level, but concentrated against sliders.
        skew, ov_skew = self._cells([(4, "FF", 400, .420, 0), (4, "SL", 400, .262, 0)])
        m_fast, _ = pa.batter_mix_multiplier(4, {"FF": .9, "SL": .1}, skew, ov_skew)
        m_slid, _ = pa.batter_mix_multiplier(4, {"FF": .1, "SL": .9}, skew, ov_skew)
        self.assertGreater(m_fast, 1.0)
        self.assertLess(m_slid, 1.0)

    def test_absent_hitter_is_neutral_not_league(self):
        cells, overall = self._cells([(5, "FF", 400, .420, 0)])
        m, cov = pa.batter_mix_multiplier(999, {"FF": 1.0}, cells, overall)
        self.assertEqual((m, cov), (1.0, 0.0))

    def test_uncovered_arsenal_weight_degrades_continuously(self):
        # Only the fastball cell exists; the slider half of the arsenal must
        # pull toward this hitter's own centre, not toward league, and the
        # multiplier must move smoothly with covered weight.
        cells, overall = self._cells([(6, "FF", 400, .371, 0), (6, "SL", 400, .276, 0)])
        full, cov_full = pa.batter_mix_multiplier(6, {"FF": 1.0}, cells, overall)
        self.assertLess(full - 1.0, pa.MIX_CLAMP)   # unclamped, so linearity holds
        partial, cov_part = pa.batter_mix_multiplier(
            6, {"FF": .5, "CH": .5}, cells, overall)
        self.assertAlmostEqual(cov_full, 1.0)
        self.assertAlmostEqual(cov_part, 0.5)
        self.assertAlmostEqual(partial - 1.0, (full - 1.0) / 2, places=6)

    def test_shrinkage_keeps_n_over_n_plus_k_of_the_cell_deviation(self):
        # Asserted on the cell itself: the multiplier clamps, the cell does not.
        rows = [(7, "FF", 100, .450, 0), (7, "SL", 100, .250, 0)]
        raw, ov = self._cells(rows, k=0.0)
        shrunk, _ = self._cells(rows, k=600.0)
        centre = ov[7]
        kept = (shrunk[(7, "FF")] - centre) / (raw[(7, "FF")] - centre)
        self.assertAlmostEqual(kept, 100 / (100 + 600), places=6)
        # ...and the surviving deviation still points the right way.
        m_shrunk, _ = pa.batter_mix_multiplier(7, {"FF": 1.0}, shrunk, ov)
        self.assertGreater(m_shrunk, 1.0)

    def test_clamp_bounds_one_bad_cell(self):
        cells, overall = self._cells([(8, "FF", 400, 1.500, 0), (8, "SL", 400, .100, 0)],
                                     k=0.0)
        m, _ = pa.batter_mix_multiplier(8, {"FF": 1.0}, cells, overall)
        self.assertLessEqual(m, 1.0 + pa.MIX_CLAMP + 1e-9)


class WeightTests(unittest.TestCase):
    def test_pa_share_is_preferred_over_usage(self):
        # Usage says 60/40 fastball; PA says the slider ends more of them.
        tidy = pa.normalize_arsenal(_arsenal([
            (11, "FF", 60, .350, 60.0), (11, "SL", 90, .270, 40.0)]))
        w, basis = pa.pitcher_pa_weights(tidy, 11)
        self.assertEqual(basis, "pa_share")
        self.assertAlmostEqual(w["SL"], 90 / 150)

    def test_usage_is_the_flagged_fallback(self):
        tidy = pa.normalize_arsenal(_arsenal([
            (12, "FF", np.nan, .350, 60.0), (12, "SL", np.nan, .270, 40.0)]))
        w, basis = pa.pitcher_pa_weights(tidy, 12)
        self.assertEqual(basis, "usage_share")
        self.assertAlmostEqual(w["FF"], 0.6)

    def test_unknown_pitcher_abstains(self):
        tidy = pa.normalize_arsenal(LEAGUE)
        self.assertEqual(pa.pitcher_pa_weights(tidy, 4242), ({}, "none"))
        self.assertEqual(pa.pitcher_pa_weights(tidy, None), ({}, "none"))


class LineupTests(unittest.TestCase):
    def setUp(self):
        self.lg = pa.league_by_pitch_type(pa.normalize_arsenal(LEAGUE))
        self.cells, self.overall = pa.batter_relative_cells(
            pa.normalize_arsenal(_arsenal([
                (21, "FF", 400, .420, 0), (21, "SL", 400, .262, 0),
                (22, "FF", 400, .350, 0), (22, "SL", 400, .270, 0)])),
            self.lg, k=0.0)

    def test_lineup_multiplier_is_slot_weighted(self):
        weights = {"FF": 1.0}
        even, cov, n = pa.lineup_mix_multiplier(
            [(21, 1.0), (22, 1.0)], weights, self.cells, self.overall)
        heavy, _, _ = pa.lineup_mix_multiplier(
            [(21, 3.0), (22, 1.0)], weights, self.cells, self.overall)
        self.assertGreater(heavy, even)       # more exposure to the FF-strong bat
        self.assertEqual((round(cov, 6), n), (1.0, 2))

    def test_empty_or_uncovered_lineup_is_neutral(self):
        self.assertEqual(pa.lineup_mix_multiplier([], {"FF": 1.0},
                                                  self.cells, self.overall),
                         (1.0, 0.0, 0))
        self.assertEqual(pa.lineup_mix_multiplier([(99, 1.0)], {"FF": 1.0},
                                                  self.cells, self.overall),
                         (1.0, 0.0, 0))
        self.assertEqual(pa.lineup_mix_multiplier([(21, 1.0)], {},
                                                  self.cells, self.overall),
                         (1.0, 0.0, 0))


class FetchTests(unittest.TestCase):
    def test_fetch_failure_abstains_rather_than_raising(self):
        def boom(url, cache):
            raise RuntimeError("savant down")
        bat, pit = pa.fetch_arsenals(boom, 2026)
        self.assertTrue(bat.empty and pit.empty)

    def test_fetch_passes_season_and_caches_per_leaderboard(self):
        seen = []

        def fetch(url, cache):
            seen.append((url, cache))
            return LEAGUE
        bat, pit = pa.fetch_arsenals(fetch, 2026, min_pa=30)
        self.assertEqual([c for _, c in seen],
                         ["arsenal_woba_v1_batter", "arsenal_woba_v1_pitcher"])
        self.assertTrue(all("year=2026" in u and "min=30" in u for u, _ in seen))
        self.assertFalse(bat.empty or pit.empty)


if __name__ == "__main__":
    unittest.main()


class ShadowWiringTests(unittest.TestCase):
    """The arm's whole licence to exist is that it changes nothing.

    A build with PITCH_MIX_SHADOW on and one with it off must produce the same
    leans, so these assert the columns the lean and the ledger's `xw_net` read
    are byte-identical across the attach, and that the flag defaults off."""

    def _frames(self):
        matchup = pd.DataFrame([
            {"game_pk": 1, "side": "away", "pitcher": "Ace",
             "opp_xwOBA_vs_sp": .330, "opp_xwOBA_neutral": .325,
             "starter_xwOBA": .300, "mx_xwOBA": .318, "edge_xwOBA": .006,
             "mx_xwOBA_sp": .318, "edge_xwOBA_sp": .006},
            {"game_pk": 1, "side": "home", "pitcher": "Deuce",
             "opp_xwOBA_vs_sp": .310, "opp_xwOBA_neutral": .308,
             "starter_xwOBA": .320, "mx_xwOBA": .315, "edge_xwOBA": .003,
             "mx_xwOBA_sp": .315, "edge_xwOBA_sp": .003},
        ])
        pitchers = pd.DataFrame([
            {"game_pk": 1, "Name": "Ace", "player_id": 11},
            {"game_pk": 1, "Name": "Deuce", "player_id": 12},
        ])
        hitters = pd.DataFrame(
            [{"game_pk": 1, "faced_pitcher": "Ace", "player_id": 21 + i}
             for i in range(9)]
            + [{"game_pk": 1, "faced_pitcher": "Deuce", "player_id": 31 + i}
               for i in range(9)])
        bat = pa.normalize_arsenal(_arsenal(
            [(1, "FF", 5000, .350, 0), (2, "SL", 5000, .270, 0)]
            + [(21 + i, "FF", 400, .400, 0) for i in range(9)]
            + [(21 + i, "SL", 400, .250, 0) for i in range(9)]))
        pit = pa.normalize_arsenal(_arsenal([
            (11, "FF", 300, .310, 60.0), (11, "SL", 200, .280, 40.0),
            (12, "FF", 300, .310, 60.0), (12, "SL", 200, .280, 40.0)]))
        return matchup, pitchers, hitters, bat, pit

    def test_flag_is_off_by_default(self):
        import build_site as b
        self.assertFalse(b.USE_PITCH_MIX_SHADOW)

    def test_attach_leaves_every_lean_column_untouched(self):
        import build_site as b
        matchup, pitchers, hitters, bat, pit = self._frames()
        out = b.attach_pitch_mix_shadow(matchup, pitchers, hitters,
                                        {"xwOBA": .312}, bat, pit)
        for col in ("opp_xwOBA_vs_sp", "opp_xwOBA_neutral", "starter_xwOBA",
                    "mx_xwOBA", "edge_xwOBA", "mx_xwOBA_sp", "edge_xwOBA_sp"):
            pd.testing.assert_series_equal(out[col], matchup[col])

    def test_attach_adds_the_shadow_columns_and_they_move(self):
        import build_site as b
        matchup, pitchers, hitters, bat, pit = self._frames()
        out = b.attach_pitch_mix_shadow(matchup, pitchers, hitters,
                                        {"xwOBA": .312}, bat, pit)
        for col in ("mix_mult", "mix_coverage", "mix_hitters", "mix_basis",
                    "opp_xwOBA_mix", "mx_xwOBA_sp_mix", "edge_xwOBA_sp_mix"):
            self.assertIn(col, out.columns)
        # These hitters crush fastballs; a 60% fastball arsenal must read above
        # the neutral composite, and the shadow phase must follow it.
        self.assertGreater(out.loc[0, "mix_mult"], 1.0)
        self.assertGreater(out.loc[0, "opp_xwOBA_mix"],
                           matchup.loc[0, "opp_xwOBA_vs_sp"])
        self.assertGreater(out.loc[0, "edge_xwOBA_sp_mix"],
                           matchup.loc[0, "edge_xwOBA_sp"])
        self.assertEqual(out.loc[0, "mix_basis"], "pa_share")

    def test_no_arsenal_data_is_a_neutral_multiplier_not_a_gap(self):
        import build_site as b
        matchup, pitchers, hitters, _, _ = self._frames()
        empty = pa.normalize_arsenal(None)
        out = b.attach_pitch_mix_shadow(matchup, pitchers, hitters,
                                        {"xwOBA": .312}, empty, empty)
        self.assertTrue((out["mix_mult"] == 1.0).all())
        self.assertTrue((out["mix_coverage"] == 0.0).all())
        pd.testing.assert_series_equal(out["opp_xwOBA_mix"],
                                       matchup["opp_xwOBA_vs_sp"],
                                       check_names=False)

    def test_empty_matchup_frame_is_returned_untouched(self):
        import build_site as b
        empty = pd.DataFrame()
        self.assertTrue(b.attach_pitch_mix_shadow(
            empty, None, None, {"xwOBA": .312}, empty, empty).empty)

    def test_ledger_carries_the_shadow_columns_as_audit_only(self):
        import grade_leans as gl
        for col in ("mix_mult_away", "mix_mult_home", "mix_coverage_away",
                    "mix_basis_home", "opp_xwoba_mix_away",
                    "mx_xwoba_sp_mix_home", "edge_xwoba_sp_mix_away"):
            self.assertIn(col, gl.AUDIT_COLS)
            self.assertNotIn(col, gl.LEDGER_COLS)   # never read by grading
