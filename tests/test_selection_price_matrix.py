"""The |delta| x selected-side closing-price matrix in `ledger_report.txt`.

Every assertion pins a RULE, not a number off tonight's ledger. The
load-bearing one is the first: this block exists so the internal artifact and
the per-game card cannot publish different numbers for the same cell, and a
test that only checked the block rendered would pass just as happily while the
two drifted apart.
"""
import re
from unittest import mock

import numpy as np
import pandas as pd

import build_site
import grade_leans
import market_backfill



def _populated_family_rows(led):
    """Graded rows of the family the LEDGER actually holds most of.

    These blocks are family-scoped by design, so on the day a `MODEL_TAG` bump
    lands they correctly render nothing -- and a test that reads the committed
    ledger through `_record_grades` then asserts against an empty block for a
    reason that has nothing to do with the rule under test.

    Reading the family off the ledger keeps the RULES pinned on both sides of a
    bump. It is the same correction this repo made to `interaction_probe`, whose
    hardcoded row selector went on answering about a model the build had stopped
    running: a constant is not only a number, the set of rows is one too.
    """
    graded = led[led["status"].astype(str).eq("graded")]
    if graded.empty:
        return graded
    fam = graded["model_tag"].value_counts().idxmax()
    return graded[graded["model_tag"].astype(str).eq(str(fam))].copy()

def _lines():
    led = pd.read_csv(grade_leans.LEDGER_PATH)
    return grade_leans._selection_price_matrix_lines(_populated_family_rows(led))


def _cells(lines):
    """Parse the `n:W-L` panel back into {(band, rung): (n, w)}."""
    out, in_panel = {}, False
    for ln in lines:
        if ln.strip() == "n and W-L":
            in_panel = True
            continue
        if in_panel and ln.strip().startswith("|delta|"):
            # drop the ROW margin: it is a total over the cells, not a cell,
            # and counting it would double every denominator below.
            heads = ln.split()[1:-1]
            continue
        if in_panel and (ln.strip().startswith("-") or not ln.strip()):
            continue
        if in_panel and ln.strip().startswith(("[", "COLUMN")):
            if ln.strip().startswith("COLUMN"):
                break
            toks = ln.split()
            band, vals = toks[0] + toks[1], toks[2:-1]
            for head, v in zip(heads, vals):
                if v != "·":
                    n, wl = v.split(":")
                    out[(band, head)] = (int(n), int(wl.split("-")[0]))
    return out


def test_every_cell_equals_the_card_cell_for_the_same_bucket():
    """The report and the game card must never disagree about one cell.

    Both derive from `hybrid_v2.apply_rule` and `market_backfill.ladder_rung`,
    so this holds by construction -- which is exactly why it is worth pinning:
    the construction is the claim, and a local copy of either would break it
    silently while both surfaces kept rendering.
    """
    # BOTH sides are scoped to the same family, and that is the whole point of
    # the test rather than a detail of it. The report block reads the family
    # the ledger holds; if the card were left on the build's current family the
    # two would be compared across different row sets, and the assertion would
    # then be about scoping rather than about the shared construction. A
    # `MODEL_TAG` bump makes that difference real, so it is pinned explicitly.
    led = pd.read_csv(grade_leans.LEDGER_PATH)
    rows = _populated_family_rows(led)
    fam = tuple(sorted(set(rows["model_tag"].astype(str))))
    # MODEL_TAG moves with the family, and not only for tidiness. build_site
    # publishes a row whose tag is not MODEL_TAG through its v13
    # RECONSTRUCTION, while grade_leans scores the lean that row's own build
    # published -- so leaving MODEL_TAG where it was would compare a
    # re-decided cell against a published one and fail for a reason that has
    # nothing to do with the shared construction under test.
    live = rows["model_tag"].astype(str).value_counts().idxmax()
    with mock.patch.object(build_site, "RECORD_TAGS", fam), \
            mock.patch.object(build_site, "MODEL_TAG", live), \
            mock.patch.object(grade_leans, "RECORD_TAGS", fam):
        ctx = build_site.hybrid_branch_records()
        report = _cells(_lines())
    card = {}
    for key, val in ctx.items():
        if isinstance(key, tuple) and key[0] == "delta_price_follow":
            lo, hi = build_site._LEAN_HISTORY_BINS[key[1]]
            band = grade_leans._band_label(lo, hi).replace(" ", "")
            rung = key[2].replace(" to ", "/").replace(" ", "")
            card[(band, rung)] = (val["model"]["n"], val["model"]["w"])
    assert report, "the matrix rendered no cells"
    assert report == card


def test_an_empty_cell_is_rendered_rather_than_skipped():
    """A bucket the model never selects is a finding about the model.

    The two longest-price columns are empty on the committed ledger: the rule
    has never followed a lean priced above +174. That is only visible if the
    cell is printed empty instead of omitted, so the row must carry one marker
    per rung whether or not it holds games.
    """
    lines = _lines()
    body = [l for l in lines if l.strip().startswith("[")]
    assert body
    n_rungs = len(market_backfill.ODDS_LADDER)
    for ln in body:
        # band label is two whitespace-separated tokens, then a cell per rung,
        # then the ROW margin.
        assert len(ln.split()) - 2 == n_rungs + 1, ln
    assert any("·" in l for l in body), "no empty cell rendered"


def test_the_header_states_the_denominator_and_the_price_basis():
    """A record over a subset must name what it dropped, from its own count.

    The fade rows are excluded because a fade backs the favourite by
    construction; the closing basis is what separates this grid from the
    saved-pregame one above it. Both are claims a reader needs in place, so
    both are pinned -- as claims, not as wording.
    """
    head = " ".join(_lines()[:4]).lower()
    assert "follow" in head and "faded" in head and "excluded" in head
    assert "closing" in head and "pregame" in head
    assert "retrospective" in head


def test_the_grid_prints_its_own_null_maximum():
    """A grid is a search, so its best cell is read against a maximum.

    On this ledger any grid hands back a flattering cell whether or not
    anything is there. The reference line is what makes the block readable,
    and it must name the reference AND the observed best.
    """
    ref = [l for l in _lines() if "best-cell reference" in l]
    assert len(ref) == 1
    assert "never against zero" in ref[0]
    assert re.search(r"observed best is [+-]\d", ref[0])


def test_the_price_ladder_has_exactly_one_home():
    """Two copies of the rung edges would let the card and the report put the
    same game in different cells -- the "one value, three homes" defect on the
    axis both surfaces bucket on. build_site must alias the shared object
    rather than restate it.
    """
    assert build_site._ODDS_LADDER is market_backfill.ODDS_LADDER
    assert build_site._ladder_rung is market_backfill.ladder_rung
    src = open("build_site.py", encoding="utf-8").read()
    assert "(None, -250," not in src, "build_site restates the ladder edges"


def test_the_matrix_covers_every_followed_row_exactly_once():
    """The bands and the ladder both tile, so no followed row may be dropped.

    A row whose price fell outside every rung would vanish from the grid with
    nothing saying so, which is the silent-denominator failure the header line
    exists to prevent.
    """
    cells = _cells(_lines())
    total_n = sum(n for n, _w in cells.values())
    head = _lines()[1]
    stated = int(re.search(r"rows: (\d+) followed", head).group(1))
    assert total_n == stated
