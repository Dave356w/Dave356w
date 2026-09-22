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
import pytest

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


def test_the_card_shows_no_cell_and_the_report_still_shows_the_whole_grid():
    """Replaces `test_every_cell_equals_the_card_cell_for_the_same_bucket`.

    That test held the card's delta x price cell equal to the report's, cell
    by cell, and it earned its place: the two HAD disagreed on 24 of 26 cells
    in production. Its subject is gone -- the card published three cells of
    that grid on 2026-09-22 and now publishes none -- so the equality has no
    two surfaces to hold together.

    It is RESTATED rather than deleted, which is this repo's own precedent
    twice over (`test_a_thin_branch_…`, the `within noise` marker). Deleting
    it would leave nothing stopping a later trim from taking the report's grid
    as well, which is the deletion `Deleting controls as clutter` forbids: the
    grid was MOVED to the analyst artifact, not removed. So the claim becomes
    a biconditional across the two surfaces -- absent from the card, present
    in full on the report, WITH the error bars and the null maximum that made
    it readable there and never reached the card.
    """
    led = pd.read_csv(grade_leans.LEDGER_PATH)
    rows = _populated_family_rows(led)
    fam = tuple(sorted(set(rows["model_tag"].astype(str))))
    with mock.patch.object(build_site, "RECORD_TAGS", fam), \
            mock.patch.object(grade_leans, "RECORD_TAGS", fam):
        ctx = build_site.hybrid_branch_records()
        report = _cells(_lines())
        text = "\n".join(grade_leans._selection_price_matrix_lines(led))

    # The card carries no cell of the grid, in any spelling.
    assert [k for k in ctx if isinstance(k, tuple)] == [], list(ctx)
    assert set(ctx["pooled"]) == {"n", "excess_be", "excess_se", "hold"}

    # The report still renders the whole thing, and still renders what made it
    # readable: a spread per cell and a search reference. A grid published with
    # neither is what came off the card.
    assert len(report) >= 20, f"the matrix rendered only {len(report)} cells"
    assert "beat its own price (pp +- se)" in text
    assert "best-cell reference" in text
    assert "under 'every game settles at its own price'" in text
    # And the report must not still claim the card shows a cell of it. That
    # sentence was the header for four minutes after the cells came off, which
    # is the "prose describing a surface that no longer exists" defect this
    # repo records; it is caught here because this is the test that reads both
    # surfaces at once.
    assert "the grid the game card shows one cell of" not in text
    assert "the game card publishes no cell of it" in text


def test_the_two_grids_are_scored_on_the_same_published_rows():
    """The equality above is only meaningful if the bases can differ.

    A test that pins two numbers equal proves nothing when the fixture makes
    them equal by construction -- which is what patching `MODEL_TAG` did here
    for a day. So this asserts the precondition directly: the ledger holds
    retained rows whose published lean and delta are NOT the ones stored on
    them, so the comparison above has something to catch.

    If this ever skips because every row is current-family, the equality test
    above has stopped being a test and needs its own fixture.
    """
    led = pd.read_csv(grade_leans.LEDGER_PATH, low_memory=False)
    fam = build_site._record_grades(led)
    retained = fam[~fam["model_tag"].astype(str).eq(build_site.MODEL_TAG)]
    if retained.empty:
        pytest.skip("no retained rows: the equality test above is vacuous")
    pub = build_site._published_grades(led)
    joined = pub.merge(
        led[["game_pk", "game_date", "xw_lean", "xw_net"]],
        on=["game_pk", "game_date"], suffixes=("_pub", "_raw"))
    moved_lean = (joined["xw_lean_pub"] != joined["xw_lean_raw"]).sum()
    moved_delta = (
        pd.to_numeric(joined["xw_net_pub"], errors="coerce").abs()
        - pd.to_numeric(joined["xw_net_raw"], errors="coerce").abs()
    ).abs().gt(1e-12).sum()
    assert moved_lean > 0 or moved_delta > 0, (
        "no published row differs from its stored row, so the cell-equality "
        "test above cannot fail and is not testing the shared basis")


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
    """The header must state its denominator and its price basis.

    It used to state a SUBSET -- follow rows only, faded rows excluded -- and
    that claim went when the retired rule came off the pages on 2026-09-18:
    the grid now scores every decidable row on the lean's own price. The
    requirement is unchanged and is restated against the new denominator,
    because a grid that stopped naming what it covers is the silent-
    denominator defect whichever direction the row set moved.

    The closing basis is what separates this grid from the saved-pregame one
    above it, and that claim is untouched.
    """
    head = " ".join(_lines()[:4]).lower()
    assert "all" in head and "decidable" in head
    # And the retired rule's denominator claim must not come back.
    assert "faded rows excluded" not in head
    assert "closing" in head and "pregame" in head


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


def test_the_matrix_covers_every_decidable_row_exactly_once():
    """The bands and the ladder both tile, so no row may be dropped.

    A row whose price fell outside every rung would vanish from the grid with
    nothing saying so, which is the silent-denominator failure the header line
    exists to prevent. Renamed from `..._every_followed_row_...` when the
    retired rule's FOLLOW filter went: the property is the same and its
    subject is now every decidable row.
    """
    cells = _cells(_lines())
    total_n = sum(n for n, _w in cells.values())
    head = _lines()[1]
    stated = int(re.search(r"rows: all (\d+) decidable", head).group(1))
    assert total_n == stated
