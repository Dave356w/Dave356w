"""The PA-level join. A wrong key here yields a plausible weak correlation
rather than an error, which is the same hazard `paired_components` refuses to
let a caller take on -- so the join is pinned rather than trusted.
"""
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


if __name__ == "__main__":
    unittest.main()
