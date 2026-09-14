"""The per-hitter frame: that it stores what was CONSUMED, and costs nothing.

Two properties carry this change. The stored value must be the vector
`aggregate_lineup` actually composited -- a re-derivation would be a second
home for it and the two would drift, the way v10 math drifted under a v9 tag.
And the write must be unable to cost a slate: it happens after the irreplaceable
pregame dumps, by a caller that swallows its failures.
"""
import os
import shutil
import tempfile
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
        import ast
        import inspect
        import textwrap
        src = inspect.getsource(bs.main) if hasattr(bs, "main") else None
        if src is None:
            self.skipTest("no main() to inspect")
        tree = ast.parse(textwrap.dedent(src))

        def _calls(node, name):
            """Line numbers of `<mod>.<name>(...)` calls inside `node`."""
            return [n.lineno for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == name]

        hit = _calls(tree, "write")
        hit = [ln for ln in hit
               if "hitter_frame.write" in textwrap.dedent(src).splitlines()[ln - 1]
               or "hitter_frame.write" in "".join(
                   textwrap.dedent(src).splitlines()[ln - 1:ln + 2])]
        self.assertTrue(hit, "no hitter_frame.write call found in main()")
        i_hit = min(hit)

        leans = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "dump_path"
                 and n.args and isinstance(n.args[0], ast.Constant)
                 and n.args[0].value == "leans"]
        self.assertTrue(leans, "no leans dump_path call found in main()")
        self.assertLess(min(leans), i_hit,
                        "hitter frame must be written AFTER the pregame dumps")

        # Structural, not a character window: the previous form sliced 400
        # chars around the call and went red the moment a comment was added
        # beside it, which is a false failure about formatting rather than
        # about the property. Find the Try that actually encloses the call.
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            body_lines = [n.lineno for b in node.body for n in ast.walk(b)
                          if hasattr(n, "lineno")]
            if i_hit not in body_lines:
                continue
            for h in node.handlers:
                t = h.type
                if t is None or (isinstance(t, ast.Name) and t.id == "Exception"):
                    guarded = True
        self.assertTrue(guarded, "hitter_frame.write must sit inside a "
                                 "try/except Exception so a fault here cannot "
                                 "cost the slate")


class PregamePreservationTests(unittest.TestCase):
    """The defect: `dump_is_post_hoc` diverts only when EVERY game on the slate
    has started, and every slate has a straggler -- so the live file was
    rewritten by the post-rollover build. 1728 of 2178 committed rows (79.3%)
    were written after their game started, median 172 minutes late."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "hitters_2026-09-06_xw.csv")
        self.start = "2026-09-06T23:00:00+00:00"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self, rate, side="home", game_pk=1):
        return [{"game_pk": game_pk, "faced_pitcher": 7, "pitcher_side": "away",
                 "batting_side": side, "player_id": 100 + i, "batting_order": i + 1,
                 "PA": 400, "xwoba_raw": rate, "xwoba_shrunk": rate,
                 "slot_weight": 1.0, "savant_backfill": False} for i in range(9)]

    def _write(self, rate, snap, side="home", game_pk=1):
        return hitter_frame.write(self._rows(rate, side, game_pk), self.path,
                        model_tag="t", model_metric="xwOBA", snapshot_utc=snap,
                        starts={game_pk: self.start})

    def test_a_post_hoc_build_cannot_overwrite_the_pregame_lineup(self):
        self._write(0.310, "2026-09-06T22:00:00+00:00")           # pregame
        self._write(0.999, "2026-09-07T01:31:00+00:00")           # post-rollover
        got = pd.read_csv(self.path)
        self.assertEqual(sorted(got["xwoba_shrunk"].unique()), [0.310])
        self.assertEqual(sorted(got["lock_status"].unique()), ["pregame"])

    def test_a_later_pregame_poll_still_refreshes_the_lineup(self):
        """Preserving pregame must not freeze the FIRST one -- the stored row
        should be the last snapshot taken before first pitch."""
        self._write(0.310, "2026-09-06T18:00:00+00:00")
        self._write(0.320, "2026-09-06T22:30:00+00:00")
        got = pd.read_csv(self.path)
        self.assertEqual(sorted(got["xwoba_shrunk"].unique()), [0.320])

    def test_a_game_side_with_no_pregame_frame_still_gets_the_post_hoc_one(self):
        """Protecting a pregame row is not a reason to store nothing for a
        game that never had one."""
        self._write(0.310, "2026-09-06T22:00:00+00:00", side="home")
        self._write(0.999, "2026-09-07T01:31:00+00:00", side="away")
        got = pd.read_csv(self.path)
        self.assertEqual(len(got), 18)
        by = got.groupby("batting_side").xwoba_shrunk.first().to_dict()
        self.assertEqual(by["home"], 0.310)
        self.assertEqual(by["away"], 0.999)

    def test_one_straggler_does_not_expose_the_started_games(self):
        """The real shape: most games done, one not. Each game-side is decided
        on its OWN start time, so the finished ones keep their pregame rows."""
        snap_late = "2026-09-07T01:31:00+00:00"
        hitter_frame.write(self._rows(0.310, game_pk=1), self.path, model_tag="t",
                 model_metric="xwOBA", snapshot_utc="2026-09-06T22:00:00+00:00",
                 starts={1: self.start})
        hitter_frame.write(self._rows(0.999, game_pk=1) + self._rows(0.400, game_pk=2),
                 self.path, model_tag="t", model_metric="xwOBA",
                 snapshot_utc=snap_late,
                 starts={1: self.start, 2: "2026-09-07T02:00:00+00:00"})
        got = pd.read_csv(self.path)
        self.assertEqual(got[got.game_pk == 1].xwoba_shrunk.iloc[0], 0.310)
        self.assertEqual(got[got.game_pk == 2].xwoba_shrunk.iloc[0], 0.400)
        self.assertEqual(got[got.game_pk == 2].lock_status.iloc[0], "pregame")

    def test_an_empty_build_leaves_the_existing_file_alone(self):
        self._write(0.310, "2026-09-06T22:00:00+00:00")
        self.assertEqual(hitter_frame.write([], self.path, snapshot_utc="x"), 0)
        self.assertEqual(len(pd.read_csv(self.path)), 9)

    def test_an_unreadable_existing_file_is_treated_as_absent_not_fatal(self):
        with open(self.path, "w") as fh:
            fh.write("\x00\x00 not,a,csv\n\"unterminated\n")
        n = self._write(0.310, "2026-09-06T22:00:00+00:00")
        self.assertEqual(n, 9)

    def test_lock_status_matches_the_graders_rule_including_the_boundary(self):
        self.assertEqual(hitter_frame.lock_status("2026-09-06T22:59:59+00:00", self.start),
                         "pregame")
        self.assertEqual(hitter_frame.lock_status(self.start, self.start), "late_snapshot")
        self.assertEqual(hitter_frame.lock_status(None, self.start), "legacy_unverified")
        self.assertEqual(hitter_frame.lock_status("2026-09-06T22:00:00+00:00", None),
                         "legacy_unverified")

    def test_a_legacy_file_without_lock_status_is_not_treated_as_pregame(self):
        """Absent provenance must not be read as a pregame claim."""
        old = hitter_frame.frame(self._rows(0.310), "t", "xwOBA",
                       "2026-09-06T22:00:00+00:00").drop(columns=["lock_status"])
        new = hitter_frame.frame(self._rows(0.999), "t", "xwOBA",
                       "2026-09-07T01:31:00+00:00", starts={1: self.start})
        got = hitter_frame.merge_preserving_pregame(old, new)
        self.assertEqual(sorted(got["xwoba_shrunk"].unique()), [0.999])


if __name__ == "__main__":
    unittest.main()
