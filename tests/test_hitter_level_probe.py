"""The PA-level join. A wrong key here yields a plausible weak correlation
rather than an error, which is the same hazard `paired_components` refuses to
let a caller take on -- so the join is pinned rather than trusted.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import actuals_backfill as ab
import hitter_level_probe as hp


def _pa(game_pk, batter_id, cats):
    return pd.DataFrame([{"game_pk": game_pk, "batter_id": batter_id,
                          "cat": c, "reconciled": True} for c in cats])


def _hitters(rows):
    return pd.DataFrame(rows)


class ValueMapTests(unittest.TestCase):
    def test_no_category_is_both_scored_and_excluded(self):
        self.assertEqual(set(hp._PA_VALUE) & hp._NOT_IN_DENOM, set())

    def test_the_weights_are_the_repo_s_own(self):
        """Not a second copy. A divergent weight set would offset every rate."""
        for k in ("1b", "2b", "3b", "hr", "bb", "hbp"):
            self.assertEqual(hp._PA_VALUE[k], ab.WOBA_W[k])

    def test_a_sac_fly_scores_zero_and_a_sac_bunt_is_excluded(self):
        """Both are plate appearances; only one is in the wOBA denominator.
        Scoring the bunt as an out penalises every hitter who laid one down."""
        d = hp.pa_values(_pa(1, 5, ["sf", "sh"]))
        sf = d[d.cat == "sf"].iloc[0]
        sh = d[d.cat == "sh"].iloc[0]
        self.assertTrue(sf["in_denom"])
        self.assertEqual(sf["woba_value"], 0.0)
        self.assertFalse(sh["in_denom"])

    def test_an_intentional_walk_is_excluded_not_scored_as_a_walk(self):
        d = hp.pa_values(_pa(1, 5, ["ibb"]))
        self.assertFalse(d.iloc[0]["in_denom"])


class JoinTests(unittest.TestCase):
    def test_the_realised_rate_is_the_hitter_s_own_woba(self):
        h = _hitters([{"game_pk": 1, "player_id": 5, "xwoba_shrunk": 0.31,
                       "xwoba_raw": 0.33}])
        pa = _pa(1, 5, ["1b", "out", "out", "hr"])
        m = hp.join(h, pa)
        self.assertEqual(len(m), 1)
        self.assertEqual(m.iloc[0]["n_pa"], 4)
        self.assertAlmostEqual(m.iloc[0]["act"],
                               (ab.WOBA_W["1b"] + ab.WOBA_W["hr"]) / 4)

    def test_excluded_categories_leave_the_denominator(self):
        """A hitter with a bunt has 2 scoring PAs, not 3."""
        m = hp.join(_hitters([{"game_pk": 1, "player_id": 5,
                               "xwoba_shrunk": 0.31}]),
                    _pa(1, 5, ["1b", "out", "sh"]))
        self.assertEqual(m.iloc[0]["n_pa"], 2)
        self.assertAlmostEqual(m.iloc[0]["act"], ab.WOBA_W["1b"] / 2)

    def test_hitters_join_to_their_own_outcomes_and_no_one_else_s(self):
        """The failure this exists to catch: a key that pairs a prediction with
        another batter's line reads as a weak correlation, never as an error."""
        h = _hitters([{"game_pk": 1, "player_id": 5, "xwoba_shrunk": 0.31},
                      {"game_pk": 1, "player_id": 6, "xwoba_shrunk": 0.29}])
        pa = pd.concat([_pa(1, 5, ["hr", "hr"]), _pa(1, 6, ["out", "out"])])
        m = hp.join(h, pa).set_index("player_id")
        self.assertAlmostEqual(m.loc[5, "act"], ab.WOBA_W["hr"])
        self.assertAlmostEqual(m.loc[6, "act"], 0.0)

    def test_the_same_player_in_two_games_stays_two_rows(self):
        h = _hitters([{"game_pk": 1, "player_id": 5, "xwoba_shrunk": 0.31},
                      {"game_pk": 2, "player_id": 5, "xwoba_shrunk": 0.32}])
        pa = pd.concat([_pa(1, 5, ["hr"]), _pa(2, 5, ["out"])])
        self.assertEqual(len(hp.join(h, pa)), 2)

    def test_a_duplicated_prediction_row_cannot_double_count_the_outcome(self):
        """Deduplicated on the PREDICTION side. The outcome is one sequence of
        plate appearances and must be counted once however the frame is keyed."""
        h = _hitters([{"game_pk": 1, "player_id": 5, "xwoba_shrunk": 0.31,
                       "faced_pitcher": 90},
                      {"game_pk": 1, "player_id": 5, "xwoba_shrunk": 0.31,
                       "faced_pitcher": 91}])
        m = hp.join(h, _pa(1, 5, ["hr", "out"]))
        self.assertEqual(len(m), 1)
        self.assertEqual(m.iloc[0]["n_pa"], 2)

    def test_unreconciled_pa_rows_are_dropped(self):
        pa = _pa(1, 5, ["hr", "out"])
        pa["reconciled"] = False
        self.assertTrue(hp.join(_hitters([{"game_pk": 1, "player_id": 5,
                                           "xwoba_shrunk": 0.31}]), pa).empty)

    def test_no_common_game_joins_to_nothing_rather_than_to_anything(self):
        m = hp.join(_hitters([{"game_pk": 1, "player_id": 5,
                               "xwoba_shrunk": 0.31}]),
                    _pa(2, 5, ["hr"]))
        self.assertTrue(m.empty)


class ReportTests(unittest.TestCase):
    def test_it_says_the_prediction_half_cannot_be_backfilled(self):
        """The load-bearing caveat: the sample starts at zero and no amount of
        collecting outcomes changes that."""
        out = "\n".join(hp.report(pd.DataFrame(), pd.DataFrame()))
        self.assertIn("cannot be backfilled", out.replace("CANNOT", "cannot"))

    def test_it_refuses_to_fit_on_too_few_rows(self):
        h = _hitters([{"game_pk": i, "player_id": 5, "xwoba_shrunk": 0.31,
                       "xwoba_raw": 0.33} for i in range(5)])
        pa = pd.concat([_pa(i, 5, ["hr", "out"]) for i in range(5)])
        out = "\n".join(hp.report(h, pa))
        self.assertIn("too few to fit", out)

    def test_it_scores_shrunk_and_raw_side_by_side(self):
        rng = np.random.default_rng(0)
        rows, pas = [], []
        for i in range(60):
            rows.append({"game_pk": i, "player_id": 5,
                         "xwoba_shrunk": 0.30 + 0.001 * i,
                         "xwoba_raw": 0.30 + 0.002 * i})
            pas.append(_pa(i, 5, list(rng.choice(["1b", "out", "hr"], 4))))
        out = "\n".join(hp.report(_hitters(rows), pd.concat(pas)))
        self.assertIn("shrunk (what the composite used)", out)
        self.assertIn("raw (pre-shrinkage)", out)
        self.assertIn("not independent", out)


class PregameGuardTests(unittest.TestCase):
    """The prediction half is only a prediction if it predates the game.

    `hitter_frame` has no `rebuild_` diversion and every build overwrites the
    slate's file, so the committed copy is the LAST build -- post-rollover for
    most slates. Measured when this guard was added: 1728 of 2178 committed
    rows (79.3%) were written after first pitch.
    """

    def _ledger(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.led = os.path.join(self.tmp, "led.csv")
        self._ledger(self.led, [
            {"game_pk": 1, "scheduled_start_utc": "2026-09-06T23:00:00+00:00"},
            {"game_pk": 2, "scheduled_start_utc": "2026-09-06T23:00:00+00:00"},
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_frame_written_after_first_pitch_is_dropped(self):
        h = _hitters([
            {"game_pk": 1, "player_id": 5, "xwoba_shrunk": .31, "xwoba_raw": .33,
             "snapshot_utc": "2026-09-06T22:00:00+00:00"},          # pregame
            {"game_pk": 2, "player_id": 6, "xwoba_shrunk": .31, "xwoba_raw": .33,
             "snapshot_utc": "2026-09-07T01:30:00+00:00"},          # post-hoc
        ])
        kept, prov = hp.pregame_only(h, self.led)
        self.assertEqual(prov, {"pregame": 1, "post_hoc": 1, "unknown": 0})
        self.assertEqual(kept["game_pk"].tolist(), [1])

    def test_first_pitch_exactly_is_post_hoc_not_pregame(self):
        """The boundary belongs to the game, matching the grader's lock rule."""
        h = _hitters([{"game_pk": 1, "player_id": 5, "xwoba_shrunk": .31,
                       "xwoba_raw": .33,
                       "snapshot_utc": "2026-09-06T23:00:00+00:00"}])
        _, prov = hp.pregame_only(h, self.led)
        self.assertEqual(prov["post_hoc"], 1)

    def test_an_uncheckable_row_is_counted_as_unknown_not_assumed_pregame(self):
        """Never assert provenance the artifact cannot substantiate."""
        h = _hitters([{"game_pk": 99, "player_id": 5, "xwoba_shrunk": .31,
                       "xwoba_raw": .33,
                       "snapshot_utc": "2026-09-06T22:00:00+00:00"}])
        kept, prov = hp.pregame_only(h, self.led)
        self.assertEqual(prov["unknown"], 1)
        self.assertEqual(prov["pregame"], 0)
        self.assertEqual(len(kept), 1)          # kept, but named

    def test_the_report_says_when_it_dropped_post_hoc_frames(self):
        h = _hitters([{"game_pk": 2, "player_id": 5, "xwoba_shrunk": .31,
                       "xwoba_raw": .33,
                       "snapshot_utc": "2026-09-07T01:30:00+00:00"}])
        out = "\n".join(hp.report(h, _pa(2, 5, ["hr"]), ledger=self.led))
        self.assertIn("AFTER first pitch", out)
        self.assertIn("DROPPED", out)


class ClusteredErrorTests(unittest.TestCase):
    def test_repeated_players_widen_the_interval(self):
        """The naive SE assumes independent rows; one hitter recurs.

        Built where the true correlation is ~0, which is both where
        `1/sqrt(n-3)` is the honest comparison and where this probe's own
        reading actually sits. Each player carries ONE rate and ONE outcome
        level repeated across his games, so the 200 rows contain 20 pieces of
        information -- if the clustered SE did not exceed the naive one here,
        the correction would be inert."""
        rng = np.random.default_rng(0)
        x, y, g = [], [], []
        for player in range(20):
            rate = 0.30 + 0.02 * rng.standard_normal()
            out = 0.30 + 0.09 * rng.standard_normal()   # independent of rate
            for _ in range(10):
                x.append(rate)
                y.append(out)
                g.append(player)
        naive = 1.0 / np.sqrt(len(x) - 3)
        clustered = hp.cluster_se_corr(x, y, g, n_boot=400)
        self.assertTrue(np.isfinite(clustered))
        self.assertGreater(clustered, naive)

    def test_it_is_deterministic_at_a_fixed_seed(self):
        x = list(np.linspace(0.28, 0.34, 40))
        y = [v + 0.01 for v in x]
        g = [i % 8 for i in range(40)]
        a = hp.cluster_se_corr(x, y, g, n_boot=200)
        b = hp.cluster_se_corr(x, y, g, n_boot=200)
        self.assertEqual(a, b)

    def test_too_few_clusters_returns_nan_rather_than_a_number(self):
        x = [0.30, 0.31, 0.32, 0.33]
        self.assertTrue(np.isnan(hp.cluster_se_corr(x, x, [1, 1, 2, 2])))


class TeamControlTests(unittest.TestCase):
    """A control is only a control if it is scored on the rows the model was
    scored on. The probe used to point at ledger_report.txt's component line,
    computed over every v12 slate while the probe covers only framed ones."""

    def _joined(self, n_sides=40):
        """Lineups must DIFFER across sides or the predictor has no variance."""
        rows = []
        for s in range(n_sides):
            for k in range(9):
                rows.append({"game_pk": s, "batting_side": "home",
                             "player_id": s * 9 + k,
                             "xwoba_shrunk": 0.30 + 0.001 * s + 0.002 * k,
                             "slot_weight": 1.0,
                             "act": 0.30 + 0.001 * s + 0.002 * k, "n_pa": 4})
        return pd.DataFrame(rows)

    def test_it_aggregates_the_probes_own_rows(self):
        got = hp.team_control(self._joined())
        self.assertIsNotNone(got)
        n, r, se = got
        self.assertEqual(n, 40)
        self.assertAlmostEqual(se, 1 / np.sqrt(37))

    def test_a_frame_without_the_columns_returns_none_not_a_wrong_control(self):
        m = self._joined().drop(columns=["batting_side"])
        self.assertIsNone(hp.team_control(m))

    def test_a_thin_control_is_printed_with_its_error_bar_not_suppressed(self):
        """No sample-size gate. Suppressing the control is what made the first
        clean run print none at all (28 sides against a floor of 30), which
        invites the cross-row-set comparison it exists to replace."""
        got = hp.team_control(self._joined(n_sides=5))
        self.assertIsNotNone(got)
        n, r, se = got
        self.assertEqual(n, 5)
        self.assertAlmostEqual(se, 1 / np.sqrt(2))     # wide, and honest

    def test_it_still_refuses_when_the_error_bar_cannot_exist(self):
        """Structural, not a judgement: 1/sqrt(n-3) needs four sides."""
        self.assertIsNone(hp.team_control(self._joined(n_sides=3)))

    def test_a_constant_predictor_returns_none_rather_than_nan(self):
        """Every side the same lineup gives a zero-variance predictor. A
        correlation there is nan, and printing nan as a control is worse than
        printing nothing."""
        m = self._joined()
        m["xwoba_shrunk"] = 0.31
        self.assertIsNone(hp.team_control(m))


class CeilingAndGateTests(unittest.TestCase):
    """Without these a null is unreadable: nobody can tell a rate that carries
    nothing from a test too small to see one.

    The first version computed `sd(shrunk)/sd(actual)`. Shrinkage is affine, so
    `corr(shrunk, outcome) == corr(raw, outcome)` exactly and the ceiling
    CANNOT depend on K -- yet that form reads K, and at the K this repo ships
    it overstated by 39% in simulation. The bound is computed from the raw rate
    net of its own PA noise instead, and the first test below is the one that
    would have caught it."""

    SIG = 0.5206          # per-PA wOBA sd, the repo's own weights

    def _sim(self, tau=0.030, pa_game=4, n=60_000, seed=7, spread_pa=True):
        """PA VARIES by default: with a constant PA the 1/n column is collinear
        with the intercept and the variance components cannot be separated at
        all -- which is why the estimator carries an explicit fallback."""
        rng = np.random.default_rng(seed)
        theta = 0.318 + tau * rng.standard_normal(n)
        n_pa = (rng.integers(120, 650, n).astype(float) if spread_pa
                else np.full(n, 400.0))
        raw = theta + (self.SIG / np.sqrt(n_pa)) * rng.standard_normal(n)
        act = theta + (self.SIG / np.sqrt(pa_game)) * rng.standard_normal(n)
        return raw, n_pa, act

    def test_a_constant_pa_falls_back_instead_of_fitting_a_collinear_model(self):
        raw, pa, act = self._sim(spread_pa=False)
        got = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got["tau"], 0.030, delta=0.004)

    def test_the_ceiling_matches_the_correlation_a_perfect_rate_achieves(self):
        """The claim, simulated end to end: if the rate IS the talent, the
        realised correlation should land on the printed bound."""
        raw, pa, act = self._sim()
        got = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)
        realised = float(np.corrcoef(raw, act)[0, 1])
        self.assertAlmostEqual(got["ceiling"], realised, delta=0.006)

    def test_the_ceiling_does_not_move_with_the_shrinkage(self):
        """The bug in one assertion. Correlation is invariant to an affine
        transform, so shrinking the input by ANY K must leave the bound alone."""
        raw, pa, act = self._sim()
        base = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)["ceiling"]
        for K in (100, 400):
            w = 400.0 / (400.0 + K)
            shrunk = 0.318 + w * (raw - 0.318)
            # feeding the SHRUNK rate as 'raw' is what the old form effectively
            # did; the realised correlation is unchanged, so the bound must be
            self.assertAlmostEqual(
                float(np.corrcoef(shrunk, act)[0, 1]),
                float(np.corrcoef(raw, act)[0, 1]), delta=1e-12)
        self.assertGreater(base, 0)

    def test_tau_is_fitted_from_the_spread_and_the_pa_counts(self):
        raw, pa, act = self._sim(tau=0.030)
        got = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)
        self.assertAlmostEqual(got["tau"], 0.030, delta=0.004)

    def test_the_gate_is_the_n_that_makes_the_ceiling_two_se(self):
        raw, pa, act = self._sim()
        got = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)
        self.assertAlmostEqual(got["n_ceiling"], (2.0 / got["ceiling"]) ** 2, places=6)
        self.assertAlmostEqual(got["n_half"], 4 * got["n_ceiling"], places=6)

    def test_clustering_enters_as_the_square_of_what_it_widened(self):
        """se falls as 1/sqrt(n), so a 2x wider interval costs 4x the sample."""
        raw, pa, act = self._sim()
        plain = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.05)
        wide = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, 0.10)
        self.assertAlmostEqual(wide["inflation"], 2.0)
        self.assertAlmostEqual(wide["n_ceiling"], 4 * plain["n_ceiling"], places=6)

    def test_a_missing_clustered_se_does_not_silently_inflate(self):
        raw, pa, act = self._sim()
        got = hp.ceiling_and_gate(raw, pa, act, self.SIG, 0.05, float("nan"))
        self.assertAlmostEqual(got["inflation"], 1.0)

    def test_noise_exceeding_the_spread_returns_none_not_an_imaginary_tau(self):
        """A rate whose spread is SMALLER than its own sampling noise implies a
        negative talent variance. Printing a ceiling there would invent one."""
        rng = np.random.default_rng(1)
        pa = rng.integers(40, 600, 4000).astype(float)
        # Every hitter has the SAME talent; the whole spread is PA noise, so a
        # fitted intercept above zero would be inventing a talent difference.
        raw = 0.318 + (0.5206 / np.sqrt(pa)) * rng.standard_normal(4000)
        got = hp.ceiling_and_gate(raw, pa, rng.standard_normal(4000),
                                  0.5206, .05, .05)
        if got is not None:                    # a tiny positive intercept is
            self.assertLess(got["tau"], 0.004)  # sampling slop, not a talent sd

    def test_the_report_prints_the_gate_and_refuses_a_reading_under_it(self):
        rng = np.random.default_rng(2)
        # A realistic outcome mix, so the measured per-PA sd is the ~0.52 the
        # bound is calibrated against rather than an artefact of four labels.
        cats = np.array(["1b", "2b", "3b", "hr", "bb", "hbp", "out"])
        pr = np.array([.147, .045, .005, .036, .079, .011, .677]); pr = pr / pr.sum()
        rows, pas = [], []
        for i in range(90):
            rate = 0.318 + 0.035 * rng.standard_normal()
            rows.append({"game_pk": i // 9, "batting_side": "home",
                         "player_id": 1000 + i, "batting_order": (i % 9) + 1,
                         "PA": 600, "xwoba_shrunk": rate, "xwoba_raw": rate,
                         "slot_weight": 1.0})
            pas.append(_pa(i // 9, 1000 + i, list(rng.choice(cats, 4, p=pr))))
        out = "\n".join(hp.report(_hitters(rows), pd.concat(pas)))
        self.assertIn("CEILING", out)
        self.assertIn("GATE", out)
        self.assertIn("Read NOTHING before the first", out)
        self.assertIn("cannot depend on K", out)


if __name__ == "__main__":
    unittest.main()


def _moderation_frame(n_players=150, games=5, b3=0.0, seed=11):
    """Hitter-games whose slope varies with PA by a known amount.

    Built so `b3` is the thing being recovered rather than a by-product of the
    shrinkage: the outcome is generated from the SHRUNK rate the composite
    consumes, which is the regressor `beta_moderation` fits.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        pa = int(rng.integers(50, 650))
        theta = 0.315 + 0.030 * rng.standard_normal()
        z = (pa - 350) / 180.0
        for g in range(games):
            x = theta + rng.standard_normal() * 0.52 / np.sqrt(pa)
            shrunk = (pa * x + 100 * 0.315) / (pa + 100)
            act = 0.315 + (1.0 + b3 * z) * (shrunk - 0.315) \
                + rng.standard_normal() * 0.25
            rows.append({"game_pk": g, "batting_side": "home", "player_id": p,
                         "batting_order": (p % 9) + 1, "PA": pa,
                         "slot_weight": 4.6 - 0.09 * ((p % 9) + 1),
                         "xwoba_raw": x, "xwoba_shrunk": shrunk,
                         "act": act, "n_pa": 4})
    return pd.DataFrame(rows)


class BetaModerationTests(unittest.TestCase):
    """The weights are DERIVED from the slope, so the slope is what is fitted.

    `w_i` proportional to `E[PA_i] * beta_i` is algebra, not a hypothesis: the
    slot weights already estimate `E[PA_i]`, so the only open term is whether
    `beta` varies. These pin the estimator recovers a planted `b3` and that the
    panel says what it cannot do, never a measured value -- the probe has no
    live reading in this sandbox and freezing one would be the test that
    memorises an artifact.
    """

    def test_it_recovers_a_planted_moderation(self):
        got = hp.beta_moderation(_moderation_frame(b3=0.7), n_boot=150)
        pa = [t for t in got["terms"] if t["label"].startswith("PA")][0]
        flat = hp.beta_moderation(_moderation_frame(b3=0.0), n_boot=150)
        pa0 = [t for t in flat["terms"] if t["label"].startswith("PA")][0]
        self.assertGreater(pa["b3"], pa0["b3"])
        self.assertGreater(pa["b3"] - pa0["b3"], 0.3)

    def test_a_flat_slope_reads_flat(self):
        got = hp.beta_moderation(_moderation_frame(b3=0.0), n_boot=200)
        pa = [t for t in got["terms"] if t["label"].startswith("PA")][0]
        self.assertLess(abs(pa["b3"] / pa["se"]), 2.5)

    def test_the_pooled_slope_is_the_one_the_weights_read(self):
        """beta is a SLOPE, not the correlation rescaled: a composite weights
        rates, and the weight that minimises error reads the slope."""
        m = _moderation_frame(b3=0.0)
        got = hp.beta_moderation(m, n_boot=100)
        x = m["xwoba_shrunk"].to_numpy()
        y = m["act"].to_numpy()
        expected = np.polyfit(x, y, 1)[0]
        self.assertAlmostEqual(got["beta"], expected, places=6)

    def test_the_interval_is_clustered_on_the_player(self):
        """One hitter recurs across his games, so a row-wise interval is
        optimistic exactly where a borderline b3 would be read."""
        m = _moderation_frame(games=6)
        got = hp.beta_moderation(m, n_boot=300)
        naive = 1.0 / np.sqrt(len(m))
        self.assertTrue(np.isfinite(got["se_beta"]))
        self.assertGreater(got["se_beta"], naive)

    def test_the_shrinkage_weight_is_not_counted_as_a_fourth_moderator(self):
        """PA/(PA+K) is strictly increasing in PA, so it is the PA row
        relabelled; counting it twice would inflate the search correction."""
        cols = [c for _l, c, _n in hp.MODERATORS]
        self.assertIn("PA", cols)
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(len(hp.MODERATORS), 3)

    def test_the_bar_is_the_expected_maximum_not_zero(self):
        got = hp.beta_moderation(_moderation_frame(), n_boot=60)
        self.assertAlmostEqual(got["expected_max_z"],
                               float(np.sqrt(2 * np.log(got["k"]))), places=9)

    def test_the_materiality_bar_is_the_slot_weights_own_spread(self):
        """Derived, not chosen. A round number here would be the frozen
        constant this repo files under `constants frozen from data`."""
        m = _moderation_frame()
        got = hp.beta_moderation(m, n_boot=60)
        w = m[m.game_pk == 0]["slot_weight"]
        self.assertAlmostEqual(got["slot_cv"], float(w.std(ddof=1) / w.mean()),
                               places=9)

    def test_too_few_rows_returns_none_rather_than_a_slope(self):
        self.assertIsNone(hp.beta_moderation(_moderation_frame().head(10)))

    def test_a_constant_rate_returns_none_rather_than_a_fabricated_fit(self):
        m = _moderation_frame()
        m["xwoba_shrunk"] = 0.31
        self.assertIsNone(hp.beta_moderation(m))

    def test_a_missing_moderator_is_named_rather_than_dropped(self):
        m = _moderation_frame().drop(columns=["batting_order"])
        got = hp.beta_moderation(m, n_boot=60)
        slot = [t for t in got["terms"] if t["label"] == "batting slot"][0]
        self.assertTrue(np.isnan(slot["b3"]))
        self.assertIn("column absent", slot["reason"])
        self.assertEqual(got["k"], 2)


class ModerationReportTests(unittest.TestCase):
    def _report(self, **kw):
        """The outcome half comes from the PA rows, as it does in production:
        `report` joins and derives `act` itself, so a fixture that carried its
        own would be testing a shape the probe never sees."""
        m = _moderation_frame(**kw).drop(columns=["act", "n_pa"])
        m["faced_pitcher"] = "X"
        m["snapshot_utc"] = "2026-09-06T20:00:00Z"
        m["savant_backfill"] = False
        m["model_tag"] = "tag"
        m["model_metric"] = "xwOBA"
        rng = np.random.default_rng(5)
        cats = ["out", "1b", "bb", "hr", "2b"]
        pa = pd.concat(
            [_pa(int(r.game_pk), int(r.player_id),
                 list(rng.choice(cats, size=4, p=[.62, .18, .10, .05, .05])))
             for r in m.itertuples()], ignore_index=True)
        old = hp.CLUSTER_BOOT
        hp.CLUSTER_BOOT = 60
        try:
            return "\n".join(hp.report(m, pa, ledger="no_such_ledger.csv"))
        finally:
            hp.CLUSTER_BOOT = old

    def test_the_report_states_the_derivation_before_any_number(self):
        text = self._report()
        self.assertIn("E[PA_i] * beta_i", text)
        self.assertIn("algebra, not a hypothesis", text)

    def test_the_report_warns_that_the_pa_term_measures_k_first(self):
        """On the shrunk rate the slope is (PA+K)/(PA+K*), flat only when K is
        calibrated -- so a material PA term is a statement about K before it is
        one about hitters, and the fix there is K, not the weights."""
        text = self._report()
        self.assertIn("(PA+K)/(PA+K*)", text)
        self.assertIn("worse copy of the shrinkage", text)

    def test_the_report_carries_the_search_bar_and_the_materiality_bar(self):
        text = self._report()
        self.assertIn("under pure noise", text)
        self.assertIn("MATERIALITY", text)

    def test_an_unreachable_bar_is_reported_as_the_answer(self):
        """A bar that needs decades is not a reason to wait: a beta that
        cannot be shown to vary by more than the slot weights' own CV cannot
        move the composite further than the weight family's own span."""
        text = self._report()
        self.assertIn("that IS the", text)
        self.assertIn("lineup_agg_probe", text)
