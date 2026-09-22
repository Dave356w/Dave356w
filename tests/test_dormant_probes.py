"""The two probes that answer about a metric this build does not run.

`player_prior_probe` measures a shrinkage TARGET the build refuses to load, and
`reliever_shrink_probe` fits a wOBA-denominated `K` against a build that shrinks
xwOBA. Both said so in their headers, and a header is prose: they still ran and
still printed numbers a reader could quote. The claim now prints at RUNTIME, so
it is load-bearing and gets pinned here.

Every assertion below is a PROPERTY rather than the wording of a line. Pinning
the copy would pass just as happily if a later edit kept the sentence and broke
the derivation, which is the failure mode these two exist to prevent.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import build_site                                    # noqa: E402
import player_prior_probe as ppp                     # noqa: E402
import reliever_shrink_probe as rsp                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = (("player_prior_probe", ppp), ("reliever_shrink_probe", rsp))


class StandingLineIsDerivedTests(unittest.TestCase):
    """The line must follow the build, not a literal."""

    def test_both_probes_declare_themselves_dormant_on_this_build(self):
        # The build runs xwOBA, so both subjects are unreachable and both
        # reports must open by saying so.
        self.assertEqual(build_site.MODEL_RATE_LABEL, "xwOBA")
        for name, mod in PROBES:
            with self.subTest(probe=name):
                line = mod.metric_standing_line()
                self.assertIsNotNone(line, f"{name} prints nothing on an xwOBA build")
                self.assertTrue(line.strip(), f"{name} prints an empty line")
                # The build's own label is NAMED, so a reader can see which
                # metric the mismatch is against rather than inferring it.
                self.assertIn("xwOBA", line)

    def test_a_woba_build_removes_the_line_on_its_own(self):
        # The whole reason to derive it: restoring a wOBA build makes the
        # caveat disappear without anyone editing these modules. A literal
        # would go on printing after the thing it warns about was fixed --
        # the constants-frozen-from-data defect in prose.
        for name, mod in PROBES:
            with self.subTest(probe=name):
                with mock.patch.object(build_site, "MODEL_RATE_LABEL", "wOBA"):
                    self.assertIsNone(mod.metric_standing_line())

    def test_neither_probe_refuses_to_run_while_dormant(self):
        # Labelling, never suppressing: exiting would destroy an instrument
        # that measures its own question correctly for a wOBA build. The
        # standing line is a string, not a SystemExit.
        for name, mod in PROBES:
            with self.subTest(probe=name):
                self.assertIsInstance(mod.metric_standing_line(), str)


class UnreadableBuildIsItsOwnStateTests(unittest.TestCase):
    """`build_site` RAISES at import on a non-`xw+` MODEL_TAG."""

    def test_an_unreadable_build_neither_crashes_nor_claims_a_match(self):
        # `reliever_shrink_probe` imports build_site lazily, so it can reach
        # this function on a build whose metric cannot be read. Returning None
        # there would assert the build is fine because it could not be read --
        # the `_lock_note` rule inverted. It must say so instead.
        boom = RuntimeError("refusing to stamp it with a non-xwOBA MODEL_TAG")
        real = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __builtins__.__import__

        def fail(name, *a, **k):
            if name == "build_site":
                raise boom
            return real(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fail):
            line = rsp.metric_standing_line()

        self.assertIsNotNone(line, "an unreadable build must not read as a match")
        # The half that holds regardless -- the fit's own denomination -- is
        # still stated, and the half that cannot be known is named as unknown.
        self.assertIn("wOBA", line)
        self.assertIn(type(boom).__name__, line)

    def test_the_unguarded_import_is_licensed_by_a_module_level_one(self):
        # `player_prior_probe.metric_standing_line` imports build_site WITHOUT
        # a guard, and that is only safe because the module already imports it
        # at the top: a MODEL_TAG build_site refuses makes the whole probe
        # unimportable long before the function runs, so there is no state in
        # which it can be reached and fail to read the label. Assert that
        # licence rather than the asymmetry's comment.
        tree = ast.parse((ROOT / "player_prior_probe.py").read_text())
        top = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("build_site", top)


class TheLineOpensTheReportTests(unittest.TestCase):
    """A caveat below the figures it qualifies is a caveat a reader skips."""

    def test_each_probe_emits_the_standing_line_before_any_figure(self):
        # Structural, not a text window: walk `main` and require the statement
        # holding the `metric_standing_line` call to come before the first
        # `say` that carries a VALUE -- an f-string or a formatted call. A
        # character distance broke a test in this repo once when a comment
        # moved, and a statement-index threshold would be a frozen literal
        # with the same shortcoming. Banners and constant title lines are
        # allowed above it; a measurement is not, because the whole point of
        # moving this claim out of the docstring is that a reader meets it
        # before anything it qualifies.
        for name, mod in PROBES:
            with self.subTest(probe=name):
                tree = ast.parse((ROOT / f"{name}.py").read_text())
                main = next(n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == "main")
                standing_at = first_figure = None
                for i, stmt in enumerate(main.body):
                    for node in ast.walk(stmt):
                        if not (isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)):
                            continue
                        if node.func.id == "metric_standing_line":
                            standing_at = i if standing_at is None else standing_at
                        elif node.func.id == "say" and node.args:
                            carries_value = any(
                                isinstance(sub, (ast.JoinedStr, ast.Call))
                                for arg in node.args for sub in ast.walk(arg)
                            )
                            if carries_value and first_figure is None:
                                first_figure = i
                self.assertIsNotNone(
                    standing_at, f"{name}.main never calls metric_standing_line")
                self.assertIsNotNone(
                    first_figure, f"{name}.main prints no figures to qualify")
                self.assertLess(
                    standing_at, first_figure,
                    f"{name} prints a figure at statement {first_figure} before "
                    f"its standing line at {standing_at}")


if __name__ == "__main__":
    unittest.main()
