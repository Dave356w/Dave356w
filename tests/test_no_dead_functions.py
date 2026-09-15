"""Every top-level production function has a production caller.

This repo has produced the same defect five times: a deletion removes the call
sites, leaves the callee, and a note in CLAUDE.md or a docstring goes on
describing the surface that used to read it.  `_lock_provenance`,
`price_band_records`, the unrendered `price_dislocation` column,
`interaction_probe`'s stale row selector, and `_display_grades` with
`_ledger_club_labels` / `_team_record_parts` / `opener_pids` beside it.

Each was found by reading prose, which is the expensive way and the way that
had already failed.  The cheap detection is a reference count, so it is a test
rather than a rule: a function nothing in production names is either dead or
reached by a mechanism this scan cannot see, and both deserve a sentence.

Deliberately scoped to PRODUCTION references.  A function whose only callers
are tests is the exact shape of `opener_pids` -- a shim kept alive by the
assertions written for it -- so tests must not count toward liveness here.
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reached by a mechanism the AST scan cannot see (string dispatch, a
# console-script entry point, an external caller).  Every entry needs a reason;
# an empty list is the healthy state and it is empty today.
ALLOWED_WITHOUT_CALLER: dict[tuple[str, str], str] = {}


def _production_files():
    return sorted(f for f in os.listdir(ROOT)
                  if f.endswith(".py") and os.path.isfile(os.path.join(ROOT, f)))


def _parse(fname):
    with open(os.path.join(ROOT, fname), encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=fname)


class NoDeadProductionFunctionsTests(unittest.TestCase):
    def test_every_top_level_function_is_named_somewhere_in_production(self):
        files = _production_files()
        trees = {f: _parse(f) for f in files}

        # Every identifier production code *uses*: bare names and attribute
        # access alike, so `helper(...)` and `build_site.helper(...)` both
        # count.  A def statement creates no Name node, so a function that is
        # only ever defined never lands in here.
        used = set()
        for tree in trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)

        dead = []
        for fname, tree in trees.items():
            module = fname[:-3]
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("__"):
                    continue
                if node.name in used:
                    continue
                if (module, node.name) in ALLOWED_WITHOUT_CALLER:
                    continue
                dead.append(f"{fname}:{node.lineno} {node.name}()")

        self.assertEqual(
            dead, [],
            "These top-level functions are defined but never named by any "
            "production module:\n  " + "\n  ".join(dead) + "\n\n"
            "Delete them, or restore the caller the deletion removed, or add "
            "them to ALLOWED_WITHOUT_CALLER with the mechanism that reaches "
            "them. Check the docstrings and CLAUDE.md beside them too -- in "
            "every past instance the prose still claimed a live caller.")


if __name__ == "__main__":
    unittest.main()
