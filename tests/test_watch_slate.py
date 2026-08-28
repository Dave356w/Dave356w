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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule_gate
import watch_slate


class NoHandRolledDedupeTests(unittest.TestCase):
    """The watcher must not re-implement build.yml's concurrency group.

    A "skip if a build is already queued or running" check was written and
    then removed. `concurrency: site-build` already collapses duplicates --
    GitHub cancels the PENDING run when a newer one joins, so at most one
    build waits -- while the hand-rolled check adds a way to lose a whole
    shift: one run stuck in `queued` (run 33081633410 sat there ten hours)
    suppresses every dispatch after it, and does it silently.
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


class _Clock:
    """A fake clock that only advances when the loop sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 1.0)


class WatchLoopTests(unittest.TestCase):
    """The loop is driven through injected seams so no API or wall clock runs."""

    def _run(self, gate, deadline=3600.0, dispatch_raises=None):
        clock = _Clock()
        calls = []

        def fake_dispatch(repo, token):
            calls.append(clock())
            if dispatch_raises is not None:
                raise dispatch_raises

        original = (watch_slate.gate, watch_slate.dispatch_build)
        watch_slate.gate = gate
        watch_slate.dispatch_build = fake_dispatch
        try:
            count = watch_slate.watch(
                "o/r", "tok", deadline=deadline, clock=clock,
                sleep=clock.sleep, log=lambda *a: None,
            )
        finally:
            (watch_slate.gate, watch_slate.dispatch_build) = original
        return count, calls, clock

    def test_dispatches_while_the_gate_fires(self):
        count, calls, _ = self._run(
            gate=lambda now: (True, "pregame window: 1"),
        )
        # One hour of deadline at a 20 minute interval.
        self.assertEqual(count, 3)
        self.assertEqual(len(calls), 3)

    def test_holds_when_no_game_is_near(self):
        count, calls, _ = self._run(
            gate=lambda now: (False, "no game 15-360 minutes from first pitch"),
        )
        self.assertEqual(count, 0)
        self.assertEqual(calls, [])

    def test_stops_at_its_own_deadline(self):
        _count, _calls, clock = self._run(
            gate=lambda now: (True, "pregame window: 1"),
            deadline=100.0,
        )
        self.assertGreaterEqual(clock(), 100.0)
        self.assertLess(clock(), 100.0 + watch_slate.POLL_SECONDS)

    def test_a_raising_gate_fails_open(self):
        """A missed pregame row cannot be re-derived; a redundant build can be
        thrown away. schedule_gate makes the same call on a StatsAPI error."""
        def boom(now):
            raise RuntimeError("statsapi down")

        count, _calls, _ = self._run(gate=boom)
        self.assertEqual(count, 3)

    def test_a_failing_dispatch_does_not_end_the_shift(self):
        count, calls, clock = self._run(
            gate=lambda now: (True, "pregame window: 1"),
            dispatch_raises=OSError("connection reset"),
        )
        self.assertEqual(count, 0)          # nothing counted as dispatched
        self.assertEqual(len(calls), 3)     # but it kept trying to the deadline
        self.assertGreaterEqual(clock(), 3600.0)


class GateWiringTests(unittest.TestCase):
    def test_watch_takes_the_pregame_path_not_the_always_run_path(self):
        """decision() returns True for every non-schedule event, so passing
        "workflow_dispatch" here would defeat the gate entirely."""
        seen = {}

        def spy(event_name, event_schedule, now):
            seen["event"] = event_name
            seen["schedule"] = event_schedule
            return False, "2026-08-27", "no game"

        original = schedule_gate.decision
        watch_slate.schedule_gate.decision = spy
        try:
            should_run, _reason = watch_slate.gate(now=None)
        finally:
            watch_slate.schedule_gate.decision = original
        self.assertEqual(seen["event"], "schedule")
        self.assertNotEqual(seen["schedule"], schedule_gate.DAILY_GRADE_CRON)
        self.assertFalse(should_run)


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
        still leave build.yml's own polling behind it."""
        self.assertIn("schedule:", self.BUILD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
