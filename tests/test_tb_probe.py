"""Does the probe's logit recover a TB effect that was planted in the data?

The probe cannot be run here -- StatsAPI is denied by this environment's
network policy, which is why it lives behind a workflow and an offline
`--tb-csv` path. What can be checked without a network is the half that would
be wrong SILENTLY: a logit that returns a confident coefficient for a variable
carrying nothing prints a report indistinguishable from a correct one, and so
does one that returns a null for a variable carrying a real effect.

So the arms are run against synthetic slates with a KNOWN TB coefficient --
one where TB genuinely moves the outcome on top of price, and one where it is
pure noise -- and the feature builder is checked for the property the whole
reading rests on: a game's context reads only games dated strictly before it.
"""
import unittest

import numpy as np
import pandas as pd

import tb_probe as P


def synth(beta_tb, n=4000, seed=0, beta_mkt=1.0):
    """Games whose outcome is generated from price and TB with a planted beta.

    The price is generated first and TB is drawn INDEPENDENTLY of it, so a
    non-zero recovered coefficient cannot be the price leaking through -- which
    is the exact confound the probe exists to rule out on the real rows.
    """
    rng = np.random.default_rng(seed)
    p_home = np.clip(rng.beta(5, 5, n) * 0.6 + 0.2, 0.05, 0.95)
    tb = rng.normal(0, 0.2, n)
    eta = beta_mkt * np.log(p_home / (1 - p_home)) + beta_tb * (tb / tb.std())
    home_won = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return pd.DataFrame({
        "game_pk": np.arange(n),
        "game_date": pd.to_datetime("2026-04-01") + pd.to_timedelta(np.arange(n) // 12, "D"),
        "p_home": p_home, "tb_delta": tb, "abs_tb": np.abs(tb), "home_won": home_won,
    })


class PlantedEffectTests(unittest.TestCase):
    def test_a_planted_tb_effect_is_recovered(self):
        for seed in range(4):
            g = synth(0.30, seed=seed)
            (_, _, _, _), (_, b, se, z) = P.fit_arm(
                g["home_won"], [P.logit(g["p_home"].values), P._z(g["tb_delta"])],
                ["market logit", "z(TB)"])
            self.assertGreater(z, 3.0, f"seed {seed}: planted effect not recovered")
            self.assertAlmostEqual(b, 0.30, delta=4 * se)

    def test_a_tb_variable_carrying_nothing_reports_nothing(self):
        zs = []
        for seed in range(12):
            g = synth(0.0, seed=100 + seed)
            (_, _, _, _), (_, _, _, z) = P.fit_arm(
                g["home_won"], [P.logit(g["p_home"].values), P._z(g["tb_delta"])],
                ["market logit", "z(TB)"])
            zs.append(z)
        # Under no effect the z should behave like a standard normal, not like
        # a variable the estimator flatters: at most one of twelve above |2|.
        self.assertLessEqual(sum(abs(z) > 2 for z in zs), 1, f"z values {zs}")
        self.assertLess(abs(np.mean(zs)), 1.0)

    def test_log_loss_ranks_a_noise_feature_below_the_price_alone(self):
        g = synth(0.0, n=3000, seed=7)
        market = P.oos_log_loss(g, True, False)
        with_tb = P.oos_log_loss(g, True, True)
        self.assertGreater(with_tb, market,
                           "a noise feature must not improve out-of-sample loss")

    def test_log_loss_ranks_a_real_feature_above_the_price_alone(self):
        g = synth(0.45, n=3000, seed=7)
        self.assertLess(P.oos_log_loss(g, True, True), P.oos_log_loss(g, True, False))


class NoLookaheadTests(unittest.TestCase):
    def _games(self):
        rng = np.random.default_rng(3)
        rows, pk = [], 0
        for day in range(40):
            for _ in range(8):
                pk += 1
                rows.append({
                    "game_pk": pk, "season": 2026,
                    "date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day),
                    "home_id": int(rng.integers(1, 31)), "away_id": int(rng.integers(1, 31)),
                    "home_tb": float(rng.integers(4, 20)), "away_tb": float(rng.integers(4, 20)),
                })
        g = pd.DataFrame(rows)
        return g[g["home_id"] != g["away_id"]].reset_index(drop=True)

    def test_a_games_feature_does_not_move_when_later_games_change(self):
        g = self._games()
        base = P.tb_features(g)
        cut = g["date"].min() + pd.Timedelta(days=25)
        tampered = g.copy()
        later = tampered["date"] >= cut
        tampered.loc[later, "home_tb"] = 99.0      # nonsense in the future only
        after = P.tb_features(tampered)
        m = base.merge(after, on="game_pk", suffixes=("_base", "_after"))
        early = m[m["game_pk"].isin(g.loc[g["date"] < cut, "game_pk"])]
        self.assertGreater(len(early), 0)
        np.testing.assert_allclose(early["tb_delta_base"], early["tb_delta_after"],
                                   atol=0, rtol=0)

    def test_the_window_is_bounded_and_the_burn_in_holds(self):
        g = self._games()
        feats = P.tb_features(g)
        dated = g.set_index("game_pk")["date"]
        # Burn-in: no feature before the logs hold BURNIN_LOG_ROWS team-games.
        # 8 games a day = 16 team-games, so the 150-row burn-in clears on day 10.
        self.assertGreaterEqual(feats["game_pk"].map(dated).min(),
                                g["date"].min() + pd.Timedelta(days=9))


class CorrelationTests(unittest.TestCase):
    """Marginal and partial correlations must recover what was planted.

    The block exists because a MARGINAL correlation with winning and a
    CONDITIONAL one can disagree for a magnitude variable, so both are
    published. A partial correlation that silently returned the marginal --
    or an interval that did not widen with a smaller sample -- would print a
    report indistinguishable from a correct one.
    """

    def test_a_variable_that_only_proxies_price_reads_null_once_conditioned(self):
        """The exact confound the block is written to expose: a predictor with
        NO own effect, correlated with price, must show marginally and vanish
        conditionally."""
        rng = np.random.default_rng(5)
        n = 4000
        p_home = np.clip(rng.beta(5, 5, n) * 0.6 + 0.2, 0.05, 0.95)
        lg = np.log(p_home / (1 - p_home))
        proxy = lg + rng.normal(0, 0.5, n)          # correlated with price only
        home_won = (rng.random(n) < p_home).astype(float)
        marginal = P._corr(proxy, home_won)
        partial = P._partial_corr(proxy, home_won, lg)
        self.assertGreater(abs(marginal), 0.10, "planted proxy should show marginally")
        self.assertLess(abs(partial), 0.04, "and should vanish once price is removed")

    def test_a_real_effect_survives_conditioning_on_price(self):
        rng = np.random.default_rng(6)
        n = 4000
        p_home = np.clip(rng.beta(5, 5, n) * 0.6 + 0.2, 0.05, 0.95)
        lg = np.log(p_home / (1 - p_home))
        x = rng.normal(0, 1, n)
        home_won = (rng.random(n) < 1 / (1 + np.exp(-(lg + 0.6 * x)))).astype(float)
        self.assertGreater(P._partial_corr(x, home_won, lg), 0.08)

    def test_the_interval_widens_as_the_sample_shrinks(self):
        wide = P._fisher_ci(0.05, 40)
        tight = P._fisher_ci(0.05, 4000)
        self.assertGreater(wide[1] - wide[0], tight[1] - tight[0])
        for lo, hi, _ in (wide, tight):
            self.assertLess(lo, 0.05)
            self.assertGreater(hi, 0.05)

    def test_a_constant_column_returns_nan_rather_than_a_divide_by_zero(self):
        """`np.corrcoef` returns a silent nan from a zero-variance column; this
        repo has an entry for exactly that, so the guard is explicit."""
        self.assertTrue(np.isnan(P._corr(np.ones(50), np.arange(50.0))))
        self.assertTrue(np.isnan(P._fisher_ci(float("nan"), 100)[0]))

    def test_the_binary_tier_correlation_matches_the_closed_form(self):
        """Point-biserial has an exact form from the 2x2, so the block can be
        checked against arithmetic rather than against itself."""
        rng = np.random.default_rng(9)
        tier = (rng.random(600) < 0.39).astype(float)
        won = (rng.random(600) < np.where(tier > 0, 0.643, 0.596)).astype(float)
        r = P._corr(tier, won)
        p, s_ = won.mean(), won.std()
        q = tier.mean()
        closed = ((won[tier > 0].mean() - won[tier == 0].mean()) / s_) * np.sqrt(q * (1 - q))
        self.assertAlmostEqual(r, closed, places=6)


class DuplicateGameTests(unittest.TestCase):
    """A game listed twice must not reach the window, the feature or the join.

    Found in production, not in review: the first runner report printed
    `TB coverage: 996 of 982 graded decided ledger rows`, which cannot be true
    of an inner join. The schedule serves resumed and rescheduled games under
    more than one date, and a doubled game enters every later 60-day window
    twice -- so this is a wrong-feature bug, not only a wrong-n bug.
    """

    def _games(self, dup=False):
        rng = np.random.default_rng(11)
        rows, pk = [], 0
        for day in range(30):
            for _ in range(8):
                pk += 1
                rows.append({
                    "game_pk": pk, "season": 2026,
                    "date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day),
                    "home_id": int(rng.integers(1, 31)), "away_id": int(rng.integers(1, 31)),
                    "home_tb": float(rng.integers(4, 20)), "away_tb": float(rng.integers(4, 20)),
                })
        g = pd.DataFrame(rows)
        g = g[g["home_id"] != g["away_id"]].reset_index(drop=True)
        if dup:
            # The real shape: same game_pk, a LATER date -- a resumed game.
            again = g.iloc[[3, 17]].copy()
            again["date"] = again["date"] + pd.Timedelta(days=2)
            g = pd.concat([g, again], ignore_index=True)
        return g

    def test_a_duplicated_game_changes_no_other_games_feature(self):
        clean = P.tb_features(self._games(False))
        dirty = P.tb_features(self._games(True))
        self.assertEqual(len(clean), len(dirty))
        m = clean.merge(dirty, on="game_pk", suffixes=("_c", "_d"))
        self.assertEqual(len(m), len(clean))
        np.testing.assert_allclose(m["tb_delta_c"], m["tb_delta_d"], atol=0, rtol=0)

    def test_the_feature_frame_is_unique_on_game_pk(self):
        f = P.tb_features(self._games(True))
        self.assertFalse(f["game_pk"].duplicated().any())

    def test_the_join_refuses_to_fan_out(self):
        """The guard is structural: pandas raises rather than the report
        printing a coverage figure above its own denominator."""
        led = pd.DataFrame({"game_pk": [1, 2, 3]})
        tb = pd.DataFrame({"game_pk": [1, 1, 2], "tb_delta": [0.1, 0.2, 0.3]})
        with self.assertRaises(Exception):
            led.merge(tb, on="game_pk", how="inner", validate="one_to_one")

    def test_load_tb_csv_deduplicates(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            pd.DataFrame({"game_pk": [1, 1, 2],
                          "delta_tb_60": [0.1, 0.9, -0.2]}).to_csv(path, index=False)
            f = P.load_tb_csv(path)
            self.assertEqual(len(f), 2)
            self.assertFalse(f["game_pk"].duplicated().any())
        finally:
            os.unlink(path)


class OfflinePathTests(unittest.TestCase):
    def test_either_column_spelling_loads(self):
        import tempfile, os
        for cols in ({"game_pk": [1, 2], "delta_tb_60": [0.1, -0.3]},
                     {"game_pk": [1, 2], "tb_delta": [0.1, -0.3]},
                     {"game_pk": [1, 2], "abs_tb_60": [0.1, 0.3]}):
            fd, path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            try:
                pd.DataFrame(cols).to_csv(path, index=False)
                f = P.load_tb_csv(path)
                self.assertEqual(list(f.columns), ["game_pk", "tb_delta", "abs_tb"])
                np.testing.assert_allclose(f["abs_tb"], [0.1, 0.3])
            finally:
                os.unlink(path)

    def test_a_magnitude_only_frame_leaves_the_signed_arm_unusable(self):
        """Magnitude-only input must not silently masquerade as directional.

        The pooled arm reads the SIGNED delta. A frame carrying only |TB| has
        no direction to give it, and the report says SKIPPED rather than fitting
        a column of NaN -- the alternative is a published null that is really a
        missing input."""
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            pd.DataFrame({"game_pk": [1, 2], "abs_tb_60": [0.1, 0.3]}).to_csv(path, index=False)
            f = P.load_tb_csv(path)
            self.assertTrue(f["tb_delta"].isna().all())
        finally:
            os.unlink(path)


class ProbeShipsNothingTests(unittest.TestCase):
    def test_the_probe_never_writes_the_ledger(self):
        src = open("tb_probe.py", encoding="utf-8").read()
        self.assertNotIn("to_csv(LEDGER", src)
        self.assertNotIn("mlb_lean_ledger.csv\", index", src)

    def test_the_frozen_threshold_is_not_refitted_from_the_scored_rows(self):
        """p50 is a-priori or the tier arm is a search. Pin that it is a literal
        with no percentile call anywhere in the module."""
        src = open("tb_probe.py", encoding="utf-8").read()
        self.assertIn("TB_P50_FROZEN = 0.139793", src)
        self.assertNotIn("percentile", src)
        self.assertNotIn("quantile", src)


if __name__ == "__main__":
    unittest.main()
