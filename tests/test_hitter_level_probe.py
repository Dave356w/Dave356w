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

    def test_too_few_sides_returns_none(self):
        self.assertIsNone(hp.team_control(self._joined(n_sides=5)))

    def test_a_constant_predictor_returns_none_rather_than_nan(self):
        """Every side the same lineup gives a zero-variance predictor. A
        correlation there is nan, and printing nan as a control is worse than
        printing nothing."""
        m = self._joined()
        m["xwoba_shrunk"] = 0.31
        self.assertIsNone(hp.team_control(m))


if __name__ == "__main__":
    unittest.main()
