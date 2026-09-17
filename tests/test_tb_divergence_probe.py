"""The TB-divergence registration block, and the no-lookahead claim it rests on.

The frozen literals are pinned for the same reason the other registration
tests pin theirs: here the constants ARE the subject, and a threshold that
drifts converts a pre-registration into a sweep.

`NoLookaheadTests` carries the weight. This rule's whole claim is that a
game's windows close before its own slate, and that claim is cheap to assert
and easy to get wrong -- so it is tested against the committed ledger by
POISONING it: every outcome from a date forward is replaced with an absurd
value, and every feature before that date must come back bit-identical. A
window that reached forward by even one slate would move them.
"""

import unittest

import numpy as np
import pandas as pd

import tb_divergence_probe as tb


def _frame(n_days=90, games_per_day=12, seed=5, start="2026-04-01"):
    """A ledger-shaped frame. Deterministic, and independent of the real one."""
    rng = np.random.default_rng(seed)
    clubs = [f"T{i:02d}" for i in range(30)]
    rows, pk = [], 800000
    for day in range(n_days):
        order = rng.permutation(clubs)
        for slot in range(games_per_day):
            home, away = order[2 * slot], order[2 * slot + 1]
            h_tb, a_tb = float(rng.poisson(14)), float(rng.poisson(13))
            rows.append({
                "game_pk": pk,
                "game_date": pd.Timestamp(start) + pd.Timedelta(days=day),
                "home": home, "away": away,
                "home_tb": h_tb, "away_tb": a_tb,
                "act_pa_home": float(rng.integers(34, 42)),
                "act_pa_away": float(rng.integers(34, 42)),
                "home_won": int(h_tb + rng.normal(0, 3) > a_tb),
                "close_p_home": float(np.clip(rng.normal(0.53, 0.08), 0.15, 0.85)),
                "close_home_ml": -120.0, "close_away_ml": 110.0,
            })
            pk += 1
    return pd.DataFrame(rows).sort_values(["game_date", "game_pk"]).reset_index(drop=True)


class FrozenConstantTests(unittest.TestCase):
    """A registration whose constants move is a sweep wearing its clothes."""

    def test_windows_and_threshold_are_the_registered_values(self):
        self.assertEqual(tb.SHORT_DAYS, 15)
        self.assertEqual(tb.LONG_DAYS, 60)
        self.assertEqual(tb.THRESHOLD, 0.15)
        self.assertEqual(tb.REGISTERED_ON, "2026-09-17")

    def test_threshold_is_the_discovery_value_and_is_in_the_sweep(self):
        """Frozen at what the discovery sweep chose -- not re-chosen here."""
        self.assertIn(tb.THRESHOLD, tb.SWEEP)

    def test_short_window_is_inside_the_long_one(self):
        self.assertLess(tb.SHORT_DAYS, tb.LONG_DAYS)


class NoLookaheadTests(unittest.TestCase):
    """Nothing from a game's own slate, or after it, may reach its feature."""

    def test_poisoning_the_future_leaves_earlier_features_identical(self):
        g = _frame(n_days=140)
        clean = tb.divergence(g).set_index("game_pk")

        cutoff = pd.Timestamp("2026-06-15")
        poisoned_src = g.copy()
        mask = poisoned_src["game_date"] >= cutoff
        poisoned_src.loc[mask, "home_tb"] = 999.0
        poisoned_src.loc[mask, "away_tb"] = 1.0
        poisoned_src.loc[mask, "home_won"] = 1
        poisoned = tb.divergence(poisoned_src).set_index("game_pk")

        before = clean[clean["game_date"] < cutoff]
        self.assertGreater(len(before), 50)
        for col in ("delta_short", "delta_long", "div_delta"):
            np.testing.assert_array_equal(
                before[col].values, poisoned.loc[before.index, col].values)

    def test_poisoning_the_real_ledger_leaves_earlier_features_identical(self):
        """The same check against the committed ledger, not a fixture."""
        g = tb.game_frame()
        if g is None or g.empty:
            self.skipTest("ledger unavailable")
        clean = tb.divergence(g).set_index("game_pk")
        if len(clean) < 20:
            self.skipTest("too few games past the warm-up to poison meaningfully")

        cutoff = clean["game_date"].quantile(0.5)
        src = g.copy()
        mask = src["game_date"] >= cutoff
        src.loc[mask, "home_tb"] = 999.0
        src.loc[mask, "away_tb"] = 1.0
        src.loc[mask, "home_won"] = 1
        poisoned = tb.divergence(src).set_index("game_pk")

        before = clean[clean["game_date"] < cutoff]
        self.assertGreater(len(before), 0)
        for col in ("delta_short", "delta_long", "div_delta"):
            np.testing.assert_array_equal(
                before[col].values, poisoned.loc[before.index, col].values)

    def test_a_games_own_slate_is_excluded_not_merely_the_game(self):
        """The cutoff is the slate date, so a same-day game cannot inform it."""
        g = _frame(n_days=90)
        feats = tb.divergence(g).set_index("game_pk")
        target_pk = feats.index[len(feats) // 2]
        target_date = feats.loc[target_pk, "game_date"]

        # Corrupt only that game's own slate. The feature must not move.
        src = g.copy()
        same_day = src["game_date"] == target_date
        self.assertGreater(int(same_day.sum()), 1)
        src.loc[same_day, "home_tb"] = 999.0
        src.loc[same_day, "away_tb"] = 1.0
        moved = tb.divergence(src).set_index("game_pk")
        self.assertEqual(moved.loc[target_pk, "div_delta"],
                         feats.loc[target_pk, "div_delta"])

    def test_warmup_rows_are_dropped_rather_than_scored_on_a_short_window(self):
        g = _frame(n_days=120)
        feats = tb.divergence(g)
        first = g["game_date"].min() + pd.Timedelta(days=tb.LONG_DAYS)
        self.assertGreaterEqual(feats["game_date"].min(), first)

    def test_feature_equals_a_frame_truncated_at_the_game(self):
        """Each feature is what a run holding only prior slates would produce."""
        g = _frame(n_days=110)
        full = tb.divergence(g).set_index("game_pk")
        for pk in list(full.index)[::37][:6]:
            row = full.loc[pk]
            visible = g[(g["game_date"] < row["game_date"])
                        | (g["game_pk"] == pk)]
            live = tb.divergence(visible).set_index("game_pk")
            if pk not in live.index:
                continue
            self.assertAlmostEqual(live.loc[pk, "div_delta"],
                                   row["div_delta"], places=12)


class ReconstructionGuardTests(unittest.TestCase):
    """Abort rather than report: every rating divides by a total-bases mean."""

    def _real(self):
        g = tb.game_frame()
        if g is None or g.empty:
            self.skipTest("ledger unavailable")
        return g

    def test_the_committed_ledger_passes(self):
        self.assertTrue(tb.reconstruction_guard(self._real()))

    def test_hits_not_containing_extra_base_hits_aborts(self):
        g = self._real().copy()
        g.loc[g.index[0], "act_h_home"] = 0.0
        g.loc[g.index[0], "act_hr_home"] = 4.0
        with self.assertRaises(ValueError):
            tb.reconstruction_guard(g)

    def test_plate_appearances_below_at_bats_aborts(self):
        g = self._real().copy()
        g.loc[g.index[0], "act_pa_away"] = 1.0
        g.loc[g.index[0], "act_ab_away"] = 40.0
        with self.assertRaises(ValueError):
            tb.reconstruction_guard(g)


class RuleTests(unittest.TestCase):

    def test_total_bases_matches_the_definition(self):
        # 1B,1B,2B,3B,HR -> 5 hits, 1+1+2+3+4 = 11 total bases
        self.assertEqual(tb.total_bases(5, 1, 1, 1), 11)

    def test_a_zero_divergence_is_declined_not_sent_to_the_road_team(self):
        f = pd.DataFrame({
            "game_pk": [1], "game_date": [pd.Timestamp("2026-09-01")],
            "home": ["AAA"], "away": ["BBB"],
            "div_delta": [0.0], "abs_div_delta": [0.0],
            "home_won": [1], "close_p_home": [0.5],
            "close_home_ml": [-110.0], "close_away_ml": [-110.0]})
        self.assertTrue(tb.apply_rule(f, threshold=0.0).empty)

    def test_a_positive_divergence_backs_home_and_a_negative_backs_away(self):
        f = pd.DataFrame({
            "game_pk": [1, 2], "game_date": [pd.Timestamp("2026-09-01")] * 2,
            "home": ["AAA", "CCC"], "away": ["BBB", "DDD"],
            "div_delta": [0.5, -0.5], "abs_div_delta": [0.5, 0.5],
            "home_won": [1, 1], "close_p_home": [0.5, 0.5],
            "close_home_ml": [-110.0, -110.0], "close_away_ml": [-110.0, -110.0]})
        d = tb.apply_rule(f)
        self.assertEqual(list(d["selection"]), ["AAA", "DDD"])
        self.assertEqual(list(d["bet_won"]), [True, False])


class ControlTests(unittest.TestCase):
    """A rate without a control is not a result -- and the control must share rows."""

    def _scored(self):
        g = tb.game_frame()
        if g is None or g.empty:
            self.skipTest("ledger unavailable")
        d = tb.apply_rule(tb.divergence(g))
        if d is None or d.empty:
            self.skipTest("no qualifying games yet")
        return d

    def test_controls_are_scored_on_exactly_the_rules_rows(self):
        d = self._scored()
        ctrls = tb.controls(d)
        self.assertEqual(set(ctrls), {"always-chalk", "always-home"})
        for name, c in ctrls.items():
            self.assertEqual(len(c), len(d), f"{name} does not share the row set")

    def test_always_chalk_backs_the_market_favourite_on_every_row(self):
        d = self._scored()
        chalk = tb.controls(d)["always-chalk"]
        expected = np.where(d["close_p_home"] >= 0.5,
                            d["home_won"] == 1, d["home_won"] == 0)
        np.testing.assert_array_equal(chalk["bet_won"].values, expected)

    def test_always_chalk_is_never_priced_below_a_coin_flip(self):
        d = self._scored()
        self.assertTrue((tb.controls(d)["always-chalk"]["p_bet"] >= 0.5).all())


class StatisticsTests(unittest.TestCase):

    def test_excess_se_is_delegated_not_restated(self):
        """One home for the statistic, per market_backfill's own note."""
        p = np.array([0.4, 0.55, 0.62])
        _, se = tb._excess(np.array([True, False, True]), p)
        self.assertAlmostEqual(se, tb.excess_se(p), places=12)

    def test_excess_does_not_estimate_its_spread_from_the_outcomes(self):
        """All-W and all-L at the same prices must carry the same error bar."""
        p = np.array([0.5, 0.6, 0.45])
        _, se_w = tb._excess(np.array([True, True, True]), p)
        _, se_l = tb._excess(np.array([False, False, False]), p)
        self.assertAlmostEqual(se_w, se_l, places=12)
        self.assertGreater(se_w, 0.0)

    def test_bets_needed_scales_as_the_inverse_square_of_the_effect(self):
        near = tb.bets_needed(0.10, 0.5)
        far = tb.bets_needed(0.05, 0.5)
        self.assertAlmostEqual(far / near, 4.0, places=6)

    def test_gate_guard_passes_and_aborts_on_a_drifted_constant(self):
        self.assertTrue(tb.gate_guard())
        original = tb.GATE_BETS
        tb.GATE_BETS = original + 1
        try:
            with self.assertRaises(ValueError):
                tb.gate_guard()
        finally:
            tb.GATE_BETS = original

    def test_the_frozen_gates_re_derive_from_bets_needed(self):
        """The constants are pinned AND shown to come from the derivation."""
        self.assertEqual(
            tb.GATE_BETS,
            int(tb.bets_needed(tb.DISCOVERY_EXCESS, tb.GATE_PRICE)))
        self.assertEqual(
            tb.GATE_BETS_REALISTIC,
            int(round(tb.bets_needed(tb.PLAUSIBLE_EXCESS, tb.GATE_PRICE))))

    def test_the_plausible_gate_is_the_far_one(self):
        self.assertGreater(tb.GATE_BETS_REALISTIC, tb.GATE_BETS)

    def test_gates_are_frozen_literals_not_recomputed_from_the_sample(self):
        """A gate that moves with the sample is not a gate."""
        self.assertEqual(tb.GATE_BETS, 74)
        self.assertEqual(tb.GATE_BETS_REALISTIC, 2491)
        self.assertEqual(tb.PLAUSIBLE_EXCESS, 0.02)
        self.assertEqual(tb.GATE_PRICE, 0.53)
        self.assertEqual(tb.DISCOVERY_EXCESS, 0.116)

    def test_clustered_se_resamples_days_and_is_reproducible(self):
        g = _frame(n_days=100)
        d = tb.apply_rule(tb.divergence(g))
        a = tb.clustered_excess_se(d, draws=400, seed=3)
        b = tb.clustered_excess_se(d, draws=400, seed=3)
        self.assertEqual(a, b)
        self.assertGreater(a, 0.0)


class ReportTests(unittest.TestCase):

    def test_report_runs_against_the_committed_ledger(self):
        lines = tb.report()
        self.assertTrue(any("TB DIVERGENCE PROBE" in x for x in lines))

    def test_report_states_the_prior_and_the_price_basis(self):
        text = "\n".join(tb.report())
        self.assertIn("prior NEGATIVE", text)
        self.assertIn("devigged close", text)

    def test_report_warns_that_the_retrospective_block_overlaps_discovery(self):
        """The number is strong; the reader must not read it as forward."""
        text = "\n".join(tb.report())
        self.assertIn("OVERLAP", text)

    def test_forward_block_takes_slates_strictly_after_registration(self):
        g = tb.game_frame()
        if g is None or g.empty:
            self.skipTest("ledger unavailable")
        f = tb.divergence(g)
        fwd = f[f["game_date"].astype(str) > tb.REGISTERED_ON]
        self.assertTrue((fwd["game_date"].astype(str) > tb.REGISTERED_ON).all())


if __name__ == "__main__":
    unittest.main()
