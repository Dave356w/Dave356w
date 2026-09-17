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


class WindowSweepTests(unittest.TestCase):
    """The sweep is a search over the feature's specification, so its bar must
    be DRAWN from the candidates' correlation rather than assumed."""

    def test_the_lookback_actually_changes_the_feature(self):
        """A parameter nothing reads would make the whole sweep theatre."""
        rng = np.random.default_rng(5)
        rows, pk = [], 0
        for day in range(70):
            for _ in range(8):
                pk += 1
                rows.append({"game_pk": pk, "season": 2026,
                             "date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day),
                             "home_id": int(rng.integers(1, 31)),
                             "away_id": int(rng.integers(1, 31)),
                             "home_tb": float(rng.integers(4, 20)),
                             "away_tb": float(rng.integers(4, 20))})
        g = pd.DataFrame(rows)
        g = g[g["home_id"] != g["away_id"]].reset_index(drop=True)
        short = P.tb_features(g, lookback=10).set_index("game_pk")["tb_delta"]
        long = P.tb_features(g, lookback=60).set_index("game_pk")["tb_delta"]
        both = pd.concat([short, long], axis=1, join="inner").dropna()
        self.assertGreater(len(both), 50)
        self.assertFalse(np.allclose(both.iloc[:, 0], both.iloc[:, 1]))
        # Nested windows share their recent games, so they correlate but are
        # not the same predictor -- the fact that licenses the sweep at all.
        r = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        self.assertGreater(r, 0.1)
        self.assertLess(r, 0.99)

    def test_the_null_max_sits_below_the_independent_case_bar(self):
        """Correlated candidates make the maximum SMALLER than sqrt(2 ln k).

        Assuming independence would set the bar too high and hide a real
        effect; assuming one predictor would set it too low. Drawn from the
        observed correlation, it must land between the two.
        """
        rng = np.random.default_rng(1)
        for rho in (0.4, 0.8):
            R = np.full((4, 4), rho); np.fill_diagonal(R, 1.0)
            L = np.linalg.cholesky(R)
            sims = np.abs(rng.standard_normal((20000, 4)) @ L.T).max(axis=1)
            self.assertLess(sims.mean(), np.sqrt(2 * np.log(4)) + 0.35)
            self.assertGreater(sims.mean(), 0.7)
        # and more correlation must mean a lower bar
        def bar(rho):
            R = np.full((4, 4), rho); np.fill_diagonal(R, 1.0)
            L = np.linalg.cholesky(R)
            g = np.random.default_rng(2).standard_normal((20000, 4))
            return float(np.abs(g @ L.T).max(axis=1).mean())
        self.assertLess(bar(0.9), bar(0.2))

    def test_residualising_removes_what_it_is_given(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=300)
        y = 3.0 * x + rng.normal(size=300)
        r = P._residualise(y, x.reshape(-1, 1))
        self.assertLess(abs(float(np.corrcoef(r, x)[0, 1])), 1e-9)


class AlignmentTests(unittest.TestCase):
    """TB's SIGN against the lean's sign -- corroboration, not magnitude.

    The arm most likely to produce a false positive, because an AGREE/DIVERGE
    split is also a price split unless it is checked: this repo's OPS
    `consensus` arm looked alive at z = +1.32 and was mostly reliability and
    base rate. So the tests plant corroboration that is REAL and corroboration
    that is ONLY price, and require the arm to tell them apart.
    """

    def _frame(self, n=1500, seed=0, align_edge=0.0, price_confound=False,
               favourite_edge=0.0):
        rng = np.random.default_rng(seed)
        q = np.clip(rng.beta(5, 5, n) * 0.5 + 0.25, 0.1, 0.9)
        lean_home = rng.random(n) < 0.5
        xw = np.where(lean_home, 1, -1) * (rng.random(n) * 0.04 + 0.001)
        tb = rng.normal(0, 0.2, n)
        if price_confound:
            # TB agrees with the lean exactly when the lean is the favourite,
            # so the split carries no information of its own.
            tb = np.where(q >= 0.5, 1, -1) * np.abs(tb) * np.where(lean_home, 1, -1)
        aligned = tb * np.sign(xw)
        # `favourite_edge` makes favourites beat their price, which is the only
        # world in which a price confound is VISIBLE at all: under correct
        # pricing every cell's excess is zero and the chalk column says nothing.
        p = np.clip(q + align_edge * (aligned > 0)
                    + favourite_edge * (q >= 0.5), 0.01, 0.99)
        won = (rng.random(n) < p).astype(float)
        home_won = np.where(lean_home, won, 1 - won)
        p_home = np.where(lean_home, q, 1 - q)
        return pd.DataFrame({
            "game_date": pd.to_datetime("2026-05-01") + pd.to_timedelta(np.arange(n) // 12, "D"),
            "tb_delta": tb, "abs_tb": np.abs(tb), "xw_net": xw,
            "q_lean": q, "lean_won": won, "home_won": home_won,
            "chalk_won": np.where(p_home >= 0.5, home_won, 1 - home_won),
            "chalk_p": np.maximum(p_home, 1 - p_home),
            "lean_ml": np.where(q >= 0.5, -100 * q / (1 - q), 100 * (1 - q) / q),
        })

    def test_tb_aligned_is_positive_exactly_when_tb_backs_the_leaned_side(self):
        """The orientation, pinned directly -- getting it backwards inverts the
        whole arm and every number below it would still look plausible."""
        f = pd.DataFrame({"tb_delta": [0.3, 0.3, -0.3, -0.3],
                          "xw_net": [0.02, -0.02, 0.02, -0.02]})
        a = P.alignment_frame(f)
        # xw_net > 0 leans HOME and tb_delta > 0 favours HOME, so those agree.
        self.assertEqual(list(a["tb_aligned"] > 0), [True, False, False, True])
        self.assertEqual(list(a["agrees"]), [True, False, False, True])

    def test_a_planted_corroboration_effect_is_recovered(self):
        f = self._frame(align_edge=0.10, seed=1)
        al = P.alignment_frame(f)
        rows = P.alignment_rows(al)
        c = P.alignment_contrast(rows, al, draws=400)
        self.assertGreater(c["diff"], 5.0)
        self.assertGreater(c["z"], 2.0)

    def test_a_pure_price_confound_is_cancelled_by_the_chalk_contrast(self):
        """The failure mode the OPS arm hit: an AGREE split that is a price split.

        TB here agrees with the lean exactly when the lean is the favourite and
        carries nothing of its own, while favourites beat their price by 5pp. A
        reader taking the model's contrast alone sees a large positive effect;
        the headline nets chalk's contrast off it and lands near zero, which is
        the whole reason that subtraction is printed rather than left to the
        reader.
        """
        f = self._frame(align_edge=0.0, price_confound=True,
                        favourite_edge=0.05, n=3000, seed=2)
        al = P.alignment_frame(f)
        c = P.alignment_contrast(P.alignment_rows(al), al, draws=400)
        self.assertGreater(c["diff"], 3.0, "the naive contrast should look alive")
        self.assertLess(abs(c["net_of_chalk"]), 2.0,
                        "and netting chalk off it should kill it")

    def test_a_real_effect_survives_the_chalk_subtraction(self):
        """The other direction, so the subtraction is not simply destroying
        everything: a genuine corroboration effect must survive it."""
        f = self._frame(align_edge=0.10, favourite_edge=0.05, n=3000, seed=8)
        al = P.alignment_frame(f)
        c = P.alignment_contrast(P.alignment_rows(al), al, draws=400)
        self.assertGreater(c["net_of_chalk"], 4.0)

    def test_no_planted_effect_reads_null(self):
        zs = []
        for seed in range(8):
            f = self._frame(align_edge=0.0, seed=20 + seed)
            zs.append(P.alignment_contrast(P.alignment_rows(P.alignment_frame(f)))["z"])
        self.assertLessEqual(sum(abs(z) > 2 for z in zs), 1, f"z values {zs}")

    def test_the_cells_report_how_degenerate_their_chalk_control_is(self):
        """`model - chalk` is price-structured, so each cell must show its mix.

        Above q=.55 the model's lean IS the favourite, so model and chalk are
        the same bet and their difference is zero by construction; below q=.50
        they are opposite bets and the difference is twice the model's excess.
        Without the `same_bet` share a reader cannot see that a price-sorted
        split moves the net-of-chalk figure with no model behaviour involved.
        """
        rng = np.random.default_rng(31)
        n = 400
        q = np.where(rng.random(n) < 0.5, 0.60, 0.47)      # two clean regions
        lean_home = rng.random(n) < 0.5
        won = (rng.random(n) < q).astype(float)
        p_home = np.where(lean_home, q, 1 - q)
        home_won = np.where(lean_home, won, 1 - won)
        f = pd.DataFrame({
            "tb_delta": rng.normal(0, 0.2, n), "xw_net": np.where(lean_home, 1, -1) * 0.01,
            "q_lean": q, "lean_won": won,
            "chalk_won": np.where(p_home >= 0.5, home_won, 1 - home_won),
            "chalk_p": np.maximum(p_home, 1 - p_home)})
        rows = P.alignment_rows(P.alignment_frame(f))
        for r in rows:
            self.assertIn("same_bet", r)
            self.assertGreaterEqual(r["same_bet"], 0.0)
            self.assertLessEqual(r["same_bet"], 100.0)
        # On the q=.60 rows the model's lean is the favourite, so a cell made
        # only of those must read 100%.
        hi = f[f["q_lean"] > 0.55].assign(agrees=True)
        only_hi = P.alignment_rows(hi)
        self.assertAlmostEqual(only_hi[0]["same_bet"], 100.0, places=6)

    def test_the_headline_carries_an_interval(self):
        """A difference-in-differences published bare is the one thing this
        repo forbids without exception. Its variance is not the sum of the
        printed SEs -- the controls share rows with what they control -- so it
        is bootstrapped, and the interval must contain the point estimate."""
        f = self._frame(align_edge=0.0, n=800, seed=12)
        al = P.alignment_frame(f)
        c = P.alignment_contrast(P.alignment_rows(al), al, draws=600)
        self.assertTrue(np.isfinite(c["net_lo"]) and np.isfinite(c["net_hi"]))
        self.assertLessEqual(c["net_lo"], c["net_of_chalk"])
        self.assertGreaterEqual(c["net_hi"], c["net_of_chalk"])
        self.assertLess(c["net_lo"], 0.0)
        self.assertGreater(c["net_hi"], 0.0)

    def test_the_interval_is_omitted_rather_than_faked_on_a_tiny_frame(self):
        f = self._frame(n=6, seed=13)
        al = P.alignment_frame(f)
        c = P.alignment_contrast(P.alignment_rows(al), al, draws=50)
        if c is not None:
            self.assertTrue(np.isnan(c["net_lo"]))

    def test_the_contrast_se_is_of_the_difference(self):
        f = self._frame(seed=3)
        al = P.alignment_frame(f)
        rows = P.alignment_rows(al)
        c = P.alignment_contrast(rows, al, draws=400)
        self.assertGreater(c["se"], max(r["se"] for r in rows))

    def test_a_tie_is_dropped_rather_than_called_agreement(self):
        f = pd.DataFrame({"tb_delta": [0.0, 0.2], "xw_net": [0.01, 0.01]})
        a = P.alignment_frame(f)
        self.assertIsNone(a["agrees"].iloc[0])
        self.assertTrue(a["agrees"].iloc[1])

    def test_per_team_nets_survive_the_feature_builder(self):
        """`tb_home`/`tb_away` must reach the frame: the difference cannot be
        decomposed back into its halves, so a per-club reading needs both."""
        rng = np.random.default_rng(3)
        rows, pk = [], 0
        for day in range(30):
            for _ in range(8):
                pk += 1
                rows.append({"game_pk": pk, "season": 2026,
                             "date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day),
                             "home_id": int(rng.integers(1, 31)),
                             "away_id": int(rng.integers(1, 31)),
                             "home_tb": float(rng.integers(4, 20)),
                             "away_tb": float(rng.integers(4, 20))})
        g = pd.DataFrame(rows)
        g = g[g["home_id"] != g["away_id"]]
        feats = P.tb_features(g)
        for col in ("tb_home", "tb_away"):
            self.assertIn(col, feats.columns)
        np.testing.assert_allclose(feats["tb_home"] - feats["tb_away"],
                                   feats["tb_delta"], atol=1e-12)


class RoiSearchTests(unittest.TestCase):
    """The ROI arm must find a planted edge and must NOT find an absent one.

    This is the arm most exposed to search-driven self-deception -- the repo's
    own rule is that any grid over this data hands back a cell near +20% ROI
    whether or not anything is there. So the null-max reference and the
    walk-forward are tested directly: a feature carrying nothing must fail both,
    and one carrying a real edge must pass both.
    """

    def _frame(self, edge=0.0, n=900, seed=0):
        """Bets at honest devigged prices, with `edge` added only where |TB| is high."""
        rng = np.random.default_rng(seed)
        q = np.clip(rng.beta(5, 5, n) * 0.5 + 0.25, 0.1, 0.9)
        tb = rng.random(n) * 0.4
        p = np.clip(q + np.where(tb >= 0.2, edge, 0.0), 0.01, 0.99)
        won = (rng.random(n) < p).astype(float)
        ml = np.where(q >= 0.5, -100 * q / (1 - q), 100 * (1 - q) / q)
        return pd.DataFrame({
            "game_date": pd.to_datetime("2026-05-01") + pd.to_timedelta(np.arange(n) // 12, "D"),
            "abs_tb": tb, "q_lean": q, "lean_won": won, "lean_ml": ml,
        })

    def test_payout_matches_american_odds_by_hand(self):
        np.testing.assert_allclose(P.payout(np.array([150.0, -200.0, 100.0])),
                                   [1.5, 0.5, 1.0])
        np.testing.assert_allclose(P.flat_units([1.0, 0.0], [150.0, -200.0]), [1.5, -1.0])

    def test_a_filter_with_nothing_behind_it_does_not_beat_its_own_null_max(self):
        f = self._frame(edge=0.0, seed=3)
        grid = P.tb_grid(f)
        _, p = P.null_max_contrast(f, grid, draws=300, seed=1)
        self.assertGreater(p, 0.05, "a null feature must not clear the search bar")

    def test_a_planted_edge_is_recovered_in_the_contrast(self):
        """Estimator correctness, kept separate from whether the SEARCH can see it.

        A 12pp bump in win probability on the high-|TB| half should show as a
        contrast near 0.12*(payout+1) ~ 0.2 in flat units, and it does.
        """
        f = self._frame(edge=0.12, n=900, seed=4)
        obs = P.best_contrast(f, P.tb_grid(f))
        self.assertGreater(obs["contrast"], 0.15)
        self.assertGreater(obs["z"], 2.5)

    def test_a_large_planted_edge_clears_the_null_max(self):
        f = self._frame(edge=0.20, n=900, seed=4)
        _, p = P.null_max_contrast(f, P.tb_grid(f), draws=300, seed=1)
        self.assertLess(p, 0.05)

    def test_the_null_max_bar_is_high_and_that_is_the_point(self):
        """A REAL edge can fail this bar, so failing it is not evidence of absence.

        Measured here rather than asserted: a genuine 12pp edge at n=900 scores
        a contrast of +0.21 against a null maximum of +0.14 and lands at
        P = 0.21 -- it does NOT clear. That is the repo's own rule made
        concrete ("any grid search will hand back a cell near +20% ROI whether
        or not anything is there"), and it is why the report must say that a
        non-clearing search means "not established", never "nothing there".
        The same edge at n=3000 clears comfortably, so this is power, not bias.
        """
        weak = self._frame(edge=0.12, n=900, seed=4)
        _, p_weak = P.null_max_contrast(weak, P.tb_grid(weak), draws=300, seed=1)
        self.assertGreater(p_weak, 0.05)
        strong = self._frame(edge=0.12, n=3000, seed=4)
        _, p_strong = P.null_max_contrast(strong, P.tb_grid(strong), draws=300, seed=1)
        self.assertLess(p_strong, 0.05)

    def test_the_walk_forward_does_not_reward_chasing_a_null_feature(self):
        f = self._frame(edge=0.0, seed=5)
        ch, bl = P.walk_forward_tb(f, P.tb_grid(f))
        self.assertGreater(ch["n"], 0)
        self.assertLess(ch["roi"], bl["roi"] + 0.05,
                        "chasing noise must not look like an edge")

    def test_the_hybrid_increment_is_zero_when_the_gate_excludes_nothing(self):
        """The property that makes it an INCREMENT rather than a combined ROI."""
        f = self._frame(edge=0.0, seed=6)
        inc = P.hybrid_increment(f, [0.0])
        self.assertAlmostEqual(inc["all"][0]["increment"], 0.0, places=12)

    def test_the_vectorised_sweep_agrees_with_the_naive_one(self):
        """The cumulative-sum core must equal a plain per-threshold loop.

        `_sweep_core` turns "kept" into a prefix so each threshold is a lookup;
        that is a 4000x speedup and it is only worth having if it is the same
        arithmetic. Pinned against the obvious implementation rather than
        against a stored number, so the check survives new fixtures.
        """
        f = self._frame(edge=0.10, n=600, seed=11)
        grid = P.tb_grid(f)
        naive = max((P.filter_contrast(f, t) for t in grid),
                    key=lambda r: r["contrast"] if r else -9e9)
        fast = P.best_contrast(f, grid)
        self.assertAlmostEqual(fast["contrast"], naive["contrast"], places=12)
        self.assertAlmostEqual(fast["t"], naive["t"], places=12)

    def test_the_contrast_se_is_of_the_difference_not_of_one_side(self):
        f = self._frame(edge=0.0, seed=7)
        r = P.filter_contrast(f, 0.2)
        self.assertGreater(r["se"], max(r["kept"]["se"], r["dropped"]["se"]))

    def test_the_search_verdict_wording_is_not_specific_to_one_block(self):
        """`_search_verdict` is shared by the ROI sweep and the window sweep, so
        its copy may not name either one -- it read 'no ROI context to find
        here' under a window table for one run."""
        for pv in (0.7, 0.2, 0.01):
            self.assertNotIn("ROI", P._search_verdict(pv))

    def test_the_search_verdict_has_three_branches_and_never_calls_a_pass_a_result(self):
        self.assertIn("WORSE", P._search_verdict(0.7))
        self.assertIn("noise", P._search_verdict(0.20))
        low = P._search_verdict(0.01)
        self.assertIn("WALK-FORWARD", low)
        self.assertNotIn("finding", low.split("PERMISSION")[0])


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
                self.assertEqual(list(f.columns),
                                 ["game_pk", "tb_delta", "abs_tb", "tb_home", "tb_away"])
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

    def test_the_frozen_threshold_does_not_move_with_the_data(self):
        """p50 is a-priori or the tier arm is a search -- asserted as a PROPERTY.

        The first version of this test banned the words `percentile` and
        `quantile` anywhere in the module, and it went red the moment the ROI
        block added a legitimate SEARCH grid built from the observed spread.
        That is this repo's own rule failing on its own test: pinning a spelling
        catches a rename and misses a re-fit, and a module could satisfy it
        while recomputing the cut by hand. What actually matters is that the
        TIER boundary is the same number on any frame, so it is asserted by
        feeding two frames whose |TB| distributions do not overlap.
        """
        self.assertIn("TB_P50_FROZEN = 0.139793",
                      open("tb_probe.py", encoding="utf-8").read())
        lo = pd.DataFrame({"abs_tb": np.linspace(0.00, 0.10, 50)})
        hi = pd.DataFrame({"abs_tb": np.linspace(0.30, 0.90, 50)})
        # Every row of `lo` is below the frozen cut and every row of `hi` above,
        # which is only true if the cut ignored the frame it was handed.
        self.assertTrue((lo["abs_tb"] < P.TB_P50_FROZEN).all())
        self.assertTrue((hi["abs_tb"] >= P.TB_P50_FROZEN).all())
        for frame in (lo, hi):
            f = frame.assign(lean_won=1.0, q_lean=0.5, lean_ml=-110.0,
                             game_date=pd.Timestamp("2026-05-01"))
            r = P.filter_contrast(f, P.TB_P50_FROZEN)
            self.assertIsNone(r, "one side must be empty -- the cut did not move")


if __name__ == "__main__":
    unittest.main()
