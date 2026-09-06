"""The per-hitter frame: that it stores what was CONSUMED, and costs nothing.

Two properties carry this change. The stored value must be the vector
`aggregate_lineup` actually composited -- a re-derivation would be a second
home for it and the two would drift, the way v10 math drifted under a v9 tag.
And the write must be unable to cost a slate: it happens after the irreplaceable
pregame dumps, by a caller that swallows its failures.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import build_site as bs
import grade_leans
import hitter_frame


def _lineup(n=9, rate=None, pa=None, backfill=None, order=True):
    rate = rate if rate is not None else [0.30 + 0.01 * i for i in range(n)]
    pa = pa if pa is not None else [400] * n
    g = pd.DataFrame({
        "game_pk": [7] * n,
        "faced_pitcher": [99] * n,
        "pitcher_side": ["away"] * n,
        "batting_side": ["home"] * n,
        "player_id": list(range(1, n + 1)),
        bs.XWOBA_SHRINK_COL: rate,
        "PA": pa,
    })
    if order:
        g["batting_order"] = list(range(1, n + 1))
    if backfill is not None:
        g[bs.MODEL_RATE_TEAM_BACKFILL_COL] = backfill
    return g


class StoresWhatWasConsumedTests(unittest.TestCase):
    def test_the_stored_vector_reproduces_the_composite_exactly(self):
        """The point of the whole change. If the weighted mean of the stored
        column is not the composite the build published, the frame describes a
        lineup the model never used."""
        g = _lineup()
        sink = []
        agg = bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                                  shrink_prior=0.320, shrink_k=100.0,
                                  hitter_sink=sink)
        self.assertEqual(len(sink), 9)
        f = hitter_frame.frame(sink)
        got = np.average(f["xwoba_shrunk"], weights=f["slot_weight"])
        self.assertAlmostEqual(got, float(agg["opp_xwOBA_neutral"].iloc[0]),
                               places=12)

    def test_the_stored_value_is_neutral_not_platoon_adjusted(self):
        """`opp_xwOBA_neutral` is what the lineup component is scored on, so it
        is what is worth storing. Capturing after the platoon offset would
        persist a different quantity under the same name."""
        g = _lineup()
        sink = []
        agg = bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                                  shrink_prior=0.320, shrink_k=100.0,
                                  hitter_sink=sink)
        f = hitter_frame.frame(sink)
        neutral = float(agg["opp_xwOBA_neutral"].iloc[0])
        vs_sp = float(agg["opp_xwOBA_vs_sp"].iloc[0])
        got = np.average(f["xwoba_shrunk"], weights=f["slot_weight"])
        self.assertAlmostEqual(got, neutral, places=12)
        if not np.isclose(neutral, vs_sp):
            self.assertNotAlmostEqual(got, vs_sp, places=6)

    def test_shrinkage_is_recorded_alongside_the_raw_rate(self):
        """Both, because a later K cannot be evaluated from the shrunk value
        alone -- the same reason expected_sp_ip_raw_* exists."""
        g = _lineup(rate=[0.400] * 9, pa=[100] * 9)
        sink = []
        bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                            shrink_prior=0.300, shrink_k=100.0, hitter_sink=sink)
        f = hitter_frame.frame(sink)
        self.assertTrue((f["xwoba_raw"] == 0.400).all())
        # (100*0.400 + 100*0.300)/200 = 0.350
        self.assertTrue(np.allclose(f["xwoba_shrunk"], 0.350))

    def test_a_savant_backfilled_hitter_is_flagged_and_left_unshrunk(self):
        """He already carries a team aggregate, so the build declines to shrink
        him twice. A reader who cannot see that flag would treat him as a
        measured bat sitting at the mean."""
        bf = [False] * 8 + [True]
        g = _lineup(rate=[0.400] * 9, pa=[100] * 9, backfill=bf)
        sink = []
        bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                            shrink_prior=0.300, shrink_k=100.0, hitter_sink=sink)
        f = hitter_frame.frame(sink)
        self.assertEqual(int(f["savant_backfill"].sum()), 1)
        self.assertAlmostEqual(
            float(f.loc[f["savant_backfill"], "xwoba_shrunk"].iloc[0]), 0.400)

    def test_rows_align_positionally_with_the_hitters(self):
        """`vals` and the slot weights are positionally reindexed by their
        producers. Pairing them against the group's ORIGINAL index is how one
        hitter's rate lands on another hitter's row."""
        g = _lineup().iloc[::-1]          # non-monotonic index
        sink = []
        bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                            shrink_prior=0.320, shrink_k=100.0, hitter_sink=sink)
        f = hitter_frame.frame(sink)
        for _, row in f.iterrows():
            src = g[g["player_id"] == row["player_id"]].iloc[0]
            self.assertAlmostEqual(row["xwoba_raw"],
                                   float(src[bs.XWOBA_SHRINK_COL]))
            self.assertEqual(row["batting_order"], src["batting_order"])

    def test_no_sink_is_a_strict_no_op(self):
        """The default path must be byte-identical to before the change."""
        g = _lineup()
        a = bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                                shrink_prior=0.320, shrink_k=100.0)
        b = bs.aggregate_lineup(g, [bs.XWOBA_SHRINK_COL], weighted=True,
                                shrink_prior=0.320, shrink_k=100.0,
                                hitter_sink=[])
        pd.testing.assert_frame_equal(a, b)


class NamingTests(unittest.TestCase):
    def test_the_prefix_cannot_be_ingested_as_a_pending_lean(self):
        """A prefix decision, not a naming preference: grade_leans globs
        `leans_*`, so a suffix-based name would have been ledgered as a real
        pending row. Asserted against the grader's own globs."""
        import fnmatch
        for post_hoc in (False, True):
            name = bs.dump_path(hitter_frame.PREFIX, "2026-09-06",
                                bs.DUMP_SUFFIX, post_hoc).split("/")[-1]
            for pat in ("leans_*_xw.csv", "leans_*_split.csv", "leans_*_woba.csv"):
                self.assertFalse(fnmatch.fnmatch(name, pat),
                                 f"{name} would be ingested by {pat}")

    def test_the_naive_suffix_form_WOULD_have_matched(self):
        """So the test above cannot quietly become theatre. The colliding name
        is the one that keeps the metric suffix LAST -- which is exactly what
        naming this by suffix rather than prefix would have produced, and what
        the shadow arm's own naming decision was taken to avoid."""
        import fnmatch
        self.assertTrue(fnmatch.fnmatch("leans_2026-09-06_hitters_xw.csv",
                                        "leans_*_xw.csv"))

    def test_the_grader_globs_are_the_ones_this_asserts_against(self):
        """Read from grade_leans' source, so a new glob there fails here."""
        import inspect
        src = inspect.getsource(grade_leans)
        for pat in ("leans_*_xw.csv", "leans_*_split.csv", "leans_*_woba.csv"):
            self.assertIn(pat, src)


class WriteTests(unittest.TestCase):
    def test_an_empty_frame_writes_nothing_rather_than_a_header(self):
        """A slate whose lineups never posted has no per-hitter truth. A
        header-only file reads as one that did and found nine missing bats."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hitters_2026-09-06_xw.csv")
            self.assertEqual(hitter_frame.write([], p), 0)
            self.assertFalse(os.path.exists(p))

    def test_provenance_is_stamped_once_for_the_whole_frame(self):
        f = hitter_frame.frame([{"game_pk": 1, "player_id": 2}],
                               model_tag="T", model_metric="xwOBA",
                               snapshot_utc="Z")
        self.assertEqual(list(f.columns), hitter_frame.COLUMNS)
        self.assertEqual(f["model_tag"].tolist(), ["T"])
        self.assertEqual(f["model_metric"].tolist(), ["xwOBA"])

    def test_a_failure_writing_the_frame_cannot_cost_the_slate(self):
        """The property the whole placement exists for. The pregame dumps are
        already on disk when this runs, and the caller swallows its failures --
        so a fault here costs a diagnostic file and never irreplaceable rows."""
        import inspect
        src = inspect.getsource(bs.main) if hasattr(bs, "main") else None
        if src is None:
            self.skipTest("no main() to inspect")
        i_leans = src.index('dump_path("leans"')
        i_hit = src.index("hitter_frame.write")
        self.assertLess(i_leans, i_hit, "hitter frame must be written AFTER "
                                        "the pregame dumps")
        tail = src[i_hit - 400:i_hit + 400]
        self.assertIn("try:", tail)
        self.assertIn("except Exception", tail)


if __name__ == "__main__":
    unittest.main()
