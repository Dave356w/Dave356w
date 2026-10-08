"""Postseason PAs fold into the season rate boards, pregame only.

The chain held here: a slate with a postseason game asks for postseason PAs ->
only PAs from games dated strictly before the slate are folded -> the board's
`pa` and both model rates move by PA weight, display columns do not ->
`load_stat_lookups` builds the stat dict from the folded board -> a failed
fetch leaves the board as Savant serves it instead of failing the build.
"""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

import build_site
import postseason_rates


def _pitches(rows):
    """Raw Statcast-shaped frame; one dict per pitch."""
    base = dict(game_type="D", events=None, woba_value=np.nan, woba_denom=np.nan,
                estimated_woba_using_speedangle=np.nan, game_pk=1)
    return pd.DataFrame([{**base, **r} for r in rows])


def _board():
    return pd.DataFrame({"player_id": [10, 20, 30], "pa": [90, 400, 100],
                         "xwoba": [.300, .350, .320], "woba": [.310, .340, .330],
                         "k_percent": [20.0, 25.0, 22.0]})


class PaRows(unittest.TestCase):

    def test_one_row_per_completed_postseason_pa(self):
        raw = _pitches([
            # ball mid-PA: no event, not a PA
            dict(game_date="2026-10-01", batter=10, pitcher=7),
            # batted ball: xwOBA numerator is the speed/angle estimate
            dict(game_date="2026-10-01", batter=10, pitcher=7, events="single",
                 woba_value=.9, woba_denom=1, estimated_woba_using_speedangle=.5),
            # strikeout: no estimate, xwOBA numerator falls back to woba_value
            dict(game_date="2026-10-01", batter=10, pitcher=7,
                 events="strikeout", woba_value=0, woba_denom=1),
            # a regular-season row in the response is not a postseason PA
            dict(game_date="2026-09-27", batter=10, pitcher=7, events="walk",
                 woba_value=.7, woba_denom=1, game_type="R"),
        ])
        pas = postseason_rates.pa_rows(raw)
        self.assertEqual(len(pas), 2)
        self.assertEqual(list(pas["xw_num"]), [.5, 0.0])
        self.assertEqual(list(pas["w_num"]), [.9, 0.0])

    def test_empty_response_is_no_pas(self):
        self.assertTrue(postseason_rates.pa_rows(pd.DataFrame()).empty)

    def test_response_at_the_row_cap_is_refused(self):
        raw = _pitches([dict(game_date="2026-10-01", batter=10, pitcher=7)]
                       * postseason_rates.STATCAST_ROW_CAP)
        with self.assertRaises(ValueError):
            postseason_rates.pa_rows(raw)


class AugmentBoard(unittest.TestCase):

    def _pas(self):
        return pd.DataFrame({
            "game_date": ["2026-10-01"] * 10 + ["2026-10-05"] * 5,
            "game_pk": [1] * 10 + [2] * 5,
            "batter": [10] * 10 + [10] * 5,
            "pitcher": [7] * 15,
            "xw_num": [.4] * 10 + [5.0] * 5,
            "w_num": [.5] * 10 + [5.0] * 5,
            "den": [1.0] * 15,
        })

    def test_folds_prior_pas_by_pa_weight(self):
        out, n = postseason_rates.augment_board(
            _board(), self._pas(), "batter", "2026-10-05", ("xwoba", "woba"))
        self.assertEqual(n, 1)
        row = out.set_index("player_id").loc[10]
        self.assertEqual(row["pa"], 100)                       # 90 + 10
        self.assertAlmostEqual(row["xwoba"], (.300 * 90 + .4 * 10) / 100)
        self.assertAlmostEqual(row["woba"], (.310 * 90 + .5 * 10) / 100)
        self.assertEqual(row["k_percent"], 20.0)               # display column stays

    def test_slate_day_and_later_pas_are_never_read(self):
        # Every PA on the slate date is worth 5.0: any leak is unmissable.
        out, _ = postseason_rates.augment_board(
            _board(), self._pas(), "batter", "2026-10-05", ("xwoba",))
        self.assertLess(out.set_index("player_id").loc[10, "xwoba"], .4)
        out, n = postseason_rates.augment_board(
            _board(), self._pas(), "batter", "2026-10-01", ("xwoba",))
        self.assertEqual(n, 0)
        pd.testing.assert_frame_equal(out, _board())

    def test_untouched_players_and_input_are_unchanged(self):
        board = _board()
        out, _ = postseason_rates.augment_board(
            board, self._pas(), "batter", "2026-10-06", ("xwoba", "woba"))
        pd.testing.assert_frame_equal(board, _board())         # no mutation
        pd.testing.assert_frame_equal(out.iloc[1:].reset_index(drop=True),
                                      _board().iloc[1:].reset_index(drop=True),
                                      check_dtype=False)

    def test_pitcher_board_keys_on_pitcher(self):
        board = pd.DataFrame({"player_id": [7], "pa": [85], "xwoba": [.300]})
        out, n = postseason_rates.augment_board(
            board, self._pas(), "pitcher", "2026-10-05", ("xwoba",))
        self.assertEqual(n, 1)
        self.assertEqual(out.loc[0, "pa"], 95)
        self.assertAlmostEqual(out.loc[0, "xwoba"], (.300 * 85 + .4 * 10) / 95)

    def test_shared_column_folds_once(self):
        # The wOBA shadow arm points both model rates at `woba`.
        out, _ = postseason_rates.augment_board(
            _board(), self._pas(), "batter", "2026-10-05", ("woba", "woba"))
        self.assertAlmostEqual(out.loc[0, "woba"], (.310 * 90 + .5 * 10) / 100)


class BuildWiring(unittest.TestCase):

    def test_only_postseason_slates_ask(self):
        has = build_site.slate_has_postseason
        self.assertFalse(has(pd.DataFrame({"game_type": ["R", "R"]})))
        self.assertFalse(has(pd.DataFrame({"game_type": [None]})))
        self.assertFalse(has(pd.DataFrame({"game_pk": [1]})))
        self.assertTrue(has(pd.DataFrame({"game_type": ["R", "D"]})))
        self.assertTrue(has(pd.DataFrame({"game_type": ["W"]})))

    def test_failed_fetch_falls_back_to_season_board(self):
        with mock.patch.object(build_site, "cached_csv",
                               side_effect=RuntimeError("savant down")):
            self.assertIsNone(build_site.load_postseason_pas("2026-10-08"))

    def test_fetch_window_ends_the_day_before_the_slate(self):
        with mock.patch.object(build_site, "cached_csv",
                               return_value=pd.DataFrame()) as fetch:
            pas = build_site.load_postseason_pas("2026-10-08")
        url, name = fetch.call_args.args
        self.assertIn("game_date_lt=2026-10-07", url)
        self.assertIn("hfGT=F%7CD%7CL%7CW%7C", url)
        self.assertEqual(name, "postseason_pa")
        self.assertTrue(pas.empty)

    def test_stat_lookups_read_the_folded_board(self):
        rate = build_site.MODEL_RATE_SOURCE_COL
        custom = pd.DataFrame({"player_id": [10], "pa": [90],
                               "xwoba": [.300], "woba": [.310]})
        batted = pd.DataFrame({"id": [10], "bbe": [50]})
        pas = pd.DataFrame({"game_date": ["2026-10-01"] * 10, "game_pk": [1] * 10,
                            "batter": [10] * 10, "pitcher": [7] * 10,
                            "xw_num": [.4] * 10, "w_num": [.5] * 10,
                            "den": [1.0] * 10})
        num = {"xwoba": .4, "woba": .5}
        with mock.patch.object(build_site, "SLATE_DATE", "2026-10-05"):
            with mock.patch.object(build_site, "cached_csv",
                                   side_effect=[custom, batted]):
                stat, _, cust = build_site.load_stat_lookups(
                    "batter", postseason_pas=pas)
            with mock.patch.object(build_site, "cached_csv",
                                   side_effect=[custom, batted]):
                frozen, _, _ = build_site.load_stat_lookups("batter")
        want = (custom.loc[0, rate] * 90 + num[rate] * 10) / 100
        self.assertAlmostEqual(stat[10][build_site.MODEL_RATE_INTERNAL_COL], want)
        self.assertEqual(stat[10]["PA"], 100)
        self.assertEqual(cust.loc[0, "pa"], 100)
        self.assertAlmostEqual(frozen[10][build_site.MODEL_RATE_INTERNAL_COL],
                               custom.loc[0, rate])
        self.assertEqual(frozen[10]["PA"], 90)


if __name__ == "__main__":
    unittest.main()
