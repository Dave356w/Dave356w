"""Invariants of the long-poll pregame watcher.

Structural, not expectations about a slate: a failure here means the watcher
would dispatch wrongly, hold the build group, or outlive its job -- not that
the schedule changed.
"""

import json
import os
import pathlib
import re
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule_gate
import watch_slate

BASE = datetime(2026, 8, 28, 16, 0, tzinfo=timezone.utc)


def _game(pk, start, state="Preview"):
    return {
        "gamePk": pk,
        "gameDate": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": {"abstractGameState": state},
    }


class NoHandRolledDedupeTests(unittest.TestCase):
    """The watcher must not re-implement build.yml's concurrency group.

    A "skip if a build is already queued or running" check was written and
    then removed. `concurrency: site-build` already collapses duplicates --
    GitHub cancels the PENDING run when a newer one joins, so at most one
    build waits -- while the hand-rolled check adds a way to lose a whole
    shift: one run stuck in `queued` (run 33081633410 sat there ten hours)
    suppresses every dispatch after it, and does it silently.

    The `fired` set inside watch() is not that check and does not reinstate
    it: it is local knowledge of targets this shift has already spent, it asks
    GitHub nothing, and no external run's state can wedge it.
    """

    def test_module_exposes_no_active_run_probe(self):
        for gone in ("poll_active", "build_is_active", "TERMINAL_STATUSES"):
            self.assertFalse(
                hasattr(watch_slate, gone),
                f"{gone} is back; see this class's docstring before re-adding it",
            )


class DispatchRequestTests(unittest.TestCase):
    """Exercise the real request builder.

    The loop tests stub dispatch_build, so nothing else in this file would
    notice if the module stopped importing its own constants -- which it did
    once, when removing the dedupe probe took `API` out with it and every test
    still passed.
    """

    def _capture(self):
        seen = {}

        class _Response:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["method"] = request.get_method()
            seen["body"] = request.data
            seen["headers"] = {k.lower(): v for k, v in request.header_items()}
            return _Response()

        return seen, fake_urlopen

    def test_dispatch_posts_to_the_build_workflow_on_main(self):
        seen, fake = self._capture()
        original = watch_slate.urlopen
        watch_slate.urlopen = fake
        try:
            watch_slate.dispatch_build("Dave356w/Dave356w", "tok")
        finally:
            watch_slate.urlopen = original
        self.assertEqual(
            seen["url"],
            "https://api.github.com/repos/Dave356w/Dave356w"
            "/actions/workflows/build.yml/dispatches",
        )
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(json.loads(seen["body"]), {"ref": "main"})
        self.assertEqual(seen["headers"]["authorization"], "Bearer tok")


class TargetTests(unittest.TestCase):
    """What the shift wakes up for."""

    def test_one_cluster_yields_one_target_per_offset(self):
        games = [_game(1, BASE + timedelta(hours=3))]
        targets = watch_slate.dispatch_targets(games, BASE)
        self.assertEqual(
            [(t - BASE).total_seconds() / 60 for t, _pks, _o in targets],
            [140.0, 160.0],
        )
        self.assertEqual([o for _t, _p, o in targets], [40, 20])

    def test_games_starting_close_together_share_a_dispatch(self):
        """A build renders the whole slate, so three 7:05-7:15 games are one
        capture, not three."""
        games = [
            _game(1, BASE + timedelta(hours=3)),
            _game(2, BASE + timedelta(hours=3, minutes=5)),
            _game(3, BASE + timedelta(hours=3, minutes=10)),
        ]
        targets = watch_slate.dispatch_targets(games, BASE)
        self.assertEqual(len(targets), len(watch_slate.TARGET_MINUTES))
        self.assertEqual(targets[0][1], (1, 2, 3))

    def test_targets_are_measured_from_the_earliest_start_in_a_cluster(self):
        """Measuring from any later game would dispatch after the first one
        has already started."""
        games = [
            _game(1, BASE + timedelta(hours=3)),
            _game(2, BASE + timedelta(hours=3, minutes=15)),
        ]
        first = min(t for t, _p, _o in watch_slate.dispatch_targets(games, BASE))
        self.assertEqual(first, BASE + timedelta(hours=3) - timedelta(minutes=40))

    def test_a_separate_slate_block_gets_its_own_targets(self):
        games = [
            _game(1, BASE + timedelta(hours=1)),
            _game(2, BASE + timedelta(hours=6)),
        ]
        targets = watch_slate.dispatch_targets(games, BASE)
        self.assertEqual(len(targets), 2 * len(watch_slate.TARGET_MINUTES))
        self.assertEqual({pks for _t, pks, _o in targets}, {(1,), (2,)})

    def test_a_game_inside_the_gate_floor_is_dropped(self):
        """schedule_gate would refuse it and the snapshot could not be locked
        pregame anyway."""
        close = BASE + timedelta(minutes=schedule_gate.MIN_MINUTES_BEFORE - 1)
        self.assertEqual(watch_slate.dispatch_targets([_game(1, close)], BASE), [])

    def test_a_started_game_is_dropped(self):
        games = [_game(1, BASE + timedelta(hours=3), state="Live")]
        self.assertEqual(watch_slate.dispatch_targets(games, BASE), [])

    def test_a_past_target_survives_so_a_late_shift_captures_now(self):
        """A shift that starts mid-window must dispatch immediately rather
        than skip the cluster it landed inside."""
        games = [_game(1, BASE + timedelta(minutes=25))]
        targets = watch_slate.dispatch_targets(games, BASE)
        self.assertTrue(any(t <= BASE for t, _p, _o in targets))

    def test_targets_clear_the_gate_floor(self):
        """Read against schedule_gate, not a copy of its numbers. The tight
        target must still leave room for a build (~12 min on 2026-08-27) to
        finish before first pitch."""
        self.assertGreater(min(watch_slate.TARGET_MINUTES),
                           schedule_gate.MIN_MINUTES_BEFORE)
        self.assertLessEqual(max(watch_slate.TARGET_MINUTES),
                             schedule_gate.MAX_MINUTES_BEFORE)


class _Clock:
    """A fake clock driving both the loop's timer and its wall clock."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 1.0)

    def now(self):
        return BASE + timedelta(seconds=self.t)


class WatchLoopTests(unittest.TestCase):
    """The loop is driven through injected seams so no API or wall clock runs."""

    def _run(self, fetch, deadline=4 * 3600.0, dispatch_raises=None):
        clock = _Clock()
        calls = []

        def fake_dispatch(repo, token):
            calls.append(clock())
            if dispatch_raises is not None:
                raise dispatch_raises

        original = watch_slate.dispatch_build
        watch_slate.dispatch_build = fake_dispatch
        try:
            count = watch_slate.watch(
                "o/r", "tok", deadline=deadline, clock=clock,
                sleep=clock.sleep, now_utc=clock.now, log=lambda *a: None,
                fetch=fetch,
            )
        finally:
            watch_slate.dispatch_build = original
        return count, calls, clock

    def test_dispatches_once_per_target_not_once_per_poll(self):
        """The whole point of the rewrite. The fixed 20-minute poll fired
        about twelve times over this window; two targets means two builds."""
        games = [_game(1, BASE + timedelta(hours=3))]
        count, calls, _ = self._run(fetch=lambda day: games)
        self.assertEqual(count, 2)
        self.assertEqual(len(calls), 2)

    def test_it_wakes_at_the_target_not_before_it(self):
        games = [_game(1, BASE + timedelta(hours=3))]
        _count, calls, _ = self._run(fetch=lambda day: games)
        self.assertEqual(calls, [140 * 60.0, 160 * 60.0])

    def test_holds_when_the_slate_is_empty(self):
        count, calls, _ = self._run(fetch=lambda day: [])
        self.assertEqual(count, 0)
        self.assertEqual(calls, [])

    def test_stops_at_its_own_deadline(self):
        _count, _calls, clock = self._run(fetch=lambda day: [], deadline=100.0)
        self.assertGreaterEqual(clock(), 100.0)
        self.assertLess(clock(), 100.0 + watch_slate.RECHECK_SECONDS)

    def test_a_raising_fetch_fails_open(self):
        """A missed pregame row cannot be re-derived; a redundant build can be
        thrown away. schedule_gate makes the same call on a StatsAPI error."""
        def boom(day):
            raise RuntimeError("statsapi down")

        count, _calls, _ = self._run(fetch=boom, deadline=3600.0)
        # Backed off to RECHECK_SECONDS rather than spinning.
        self.assertEqual(count, 3600 // watch_slate.RECHECK_SECONDS)

    def test_a_failing_dispatch_retries_and_does_not_end_the_shift(self):
        games = [_game(1, BASE + timedelta(hours=3))]
        count, calls, clock = self._run(
            fetch=lambda day: games,
            dispatch_raises=OSError("connection reset"),
        )
        self.assertEqual(count, 0)               # nothing counted as dispatched
        self.assertGreater(len(calls), 2)        # but it kept retrying
        self.assertGreaterEqual(clock(), 4 * 3600.0)
        retries = [b - a for a, b in zip(calls, calls[1:])]
        self.assertTrue(all(gap <= watch_slate.RETRY_SECONDS for gap in retries))

    def test_a_moved_start_does_not_refire_a_spent_target(self):
        """Keyed on (gamePks, offset), not on the target time -- otherwise a
        rain delay re-fires every target the shift has already spent."""
        start = BASE + timedelta(hours=2)
        moved = BASE + timedelta(hours=3)
        clock = _Clock()
        calls = []

        def fetch(day):
            # Slips an hour once the T-40 target has been dispatched.
            return [_game(1, start if clock.t <= 80 * 60 else moved)]

        original = watch_slate.dispatch_build
        watch_slate.dispatch_build = lambda repo, token: calls.append(clock())
        try:
            count = watch_slate.watch(
                "o/r", "tok", deadline=4 * 3600.0, clock=clock,
                sleep=clock.sleep, now_utc=clock.now, log=lambda *a: None,
                fetch=fetch,
            )
        finally:
            watch_slate.dispatch_build = original
        # T-40 spent against the original 18:00Z start; the slip re-plans only
        # T-20, which has not fired -- it does not resurrect T-40.
        self.assertEqual(count, 2)
        self.assertEqual(calls, [80 * 60.0, 160 * 60.0])


class WatchWorkflowTests(unittest.TestCase):
    """Read against the real workflow, not a copy of its numbers."""

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    WATCH = ROOT / ".github" / "workflows" / "watch.yml"
    BUILD = ROOT / ".github" / "workflows" / "build.yml"

    def setUp(self):
        self.text = self.WATCH.read_text(encoding="utf-8")

    def test_does_not_join_the_build_concurrency_group(self):
        """A five-hour job in `site-build` parks every push behind itself for
        the rest of the slate. The watcher writes nothing, so it needs no
        serialisation against builds -- only against itself."""
        group = re.search(r"concurrency:\s*\n\s*group:\s*(\S+)", self.text)
        self.assertIsNotNone(group)
        self.assertNotEqual(group.group(1), "site-build")

    def test_can_actually_dispatch(self):
        """workflow_dispatch is one of the two events GITHUB_TOKEN may trigger,
        but it still needs actions: write. Without it the shift runs for five
        hours and silently dispatches nothing."""
        self.assertRegex(self.text, r"actions:\s*write")

    def test_process_bound_outlasts_the_shift(self):
        """One decision in two files. The `timeout` wrapper is the backstop for
        a hung loop, so it must sit ABOVE the loop's own deadline -- if it
        drops below, the step starts killing healthy shifts early and the
        loop's clock stops meaning anything."""
        wrapper = re.search(r"timeout --signal=INT --kill-after=60 (\d+)", self.text)
        self.assertIsNotNone(wrapper, "the step must bound its own process")
        self.assertGreater(int(wrapper.group(1)), watch_slate.SHIFT_SECONDS)

    def test_shifts_cannot_overlap_themselves(self):
        """A pending shift would be cancelled by the next one arriving, so the
        gap between starts has to exceed a shift's length."""
        crons = re.findall(r"cron:\s*'(\d+)\s+(\d+) \* \* \*'", self.text)
        self.assertGreaterEqual(len(crons), 2)
        starts = sorted(int(h) * 60 + int(m) for m, h in crons)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertTrue(gaps, "need at least two shifts to have a gap")
        self.assertGreater(min(gaps) * 60, watch_slate.SHIFT_SECONDS)

    def test_build_keeps_its_own_schedule(self):
        """This adds a path; it does not replace one. A dropped watch cron must
        still leave build.yml's own polling behind it -- and it is also the
        only retry behind a target that dispatches into a failing build."""
        self.assertIn("schedule:", self.BUILD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
