"""The |delta| x saved-pregame price grid: basis, denominators, and spread.

Every assertion here pins a RULE rather than a number off tonight's ledger.
The one number that is pinned is an arithmetic identity (the Poisson-binomial
SE at a known price), which is the subject rather than an observation.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import grade_leans
import hybrid_v2


def _frame(net, lean, full, pregame_p, home="HOME", away="AWAY",
           close_p=0.5, date="2026-09-01"):
    n = len(net)

    def _col(v):
        return list(v) if isinstance(v, (list, tuple)) else [v] * n

    return pd.DataFrame({
        "xw_net": _col(net), "xw_lean": _col(lean), "xw_full": _col(full),
        "home": _col(home), "away": _col(away),
        "pregame_p_home": _col(pregame_p), "close_p_home": _col(close_p),
        "game_date": _col(date),
    })


def _grid(g):
    return text_of(grade_leans._magnitude_price_grid_lines(g))


def text_of(lines):
    return "\n".join(lines)


def test_the_close_never_fills_a_missing_pregame_price():
    """The two snapshots are not interchangeable; a row without one is out.

    The reverse guard already exists for the closing-price block above. This
    is the direction that matters more: `close_p_home` is populated on every
    settled row, so a fallback would silently restate the whole family at a
    price no decision was taken at.
    """
    g = _frame([0.005, 0.005], "HOME", "W", [np.nan, 0.60], close_p=0.99)
    text = _grid(g)
    assert "Included 1 of 2 graded rows" in text
    assert "1  no usable saved pregame price" in text
    # The surviving row is priced at its own .600, not at the .990 close.
    assert "mean q .600" in text
    assert ".990" not in text


def test_q_is_the_leaned_sides_price_not_the_homes():
    """An away lean at a .60 home price is a .40 side, and lands in q<.450."""
    g = _frame([0.005], "AWAY", "W", 0.60)
    priced = [l.strip() for l in grade_leans._magnitude_price_grid_lines(g)
              if l.startswith("    q ") and "n=  1" in l]
    assert priced                      # the cell, and its column margin
    assert all(l.startswith("q <.450") and "mean q .400" in l for l in priced)


def test_an_unrecognised_lean_is_counted_out_rather_than_graded():
    """`_wlt`'s else-branch once graded an unmatched selection a LOSS.

    A club naming neither side of the game is a namespace mismatch, not a
    result, so it leaves the grid through its own counter.
    """
    g = _frame([0.005, 0.005], ["HOME", "ARI"], "W", 0.60)
    text = _grid(g)
    assert "Included 1 of 2 graded rows" in text
    assert "1  no lean, or a lean naming neither club" in text


def test_every_exclusion_is_counted_from_its_own_column():
    """A row failing two conditions appears under both, never by subtraction.

    Included + excluded therefore need not reconcile, and the block says so.
    One row here is abstained (no lean, no grade, no delta) and must be
    counted three times rather than once.
    """
    g = _frame([np.nan, 0.005, 0.005], [None, "HOME", "HOME"],
               [None, "T", "W"], [0.60, 0.60, 0.60])
    text = _grid(g)
    assert "Included 1 of 3 graded rows" in text
    assert "1  no lean, or a lean naming neither club" in text
    assert "2  no W/L full-game decision" in text        # the None and the tie
    assert "1  no finite xw_net" in text


def test_empty_cells_are_printed_rather_than_dropped():
    """A region with no games is the diagnostic, not a gap in the table.

    "The model never strongly disagrees with the market on a big delta" is
    only visible if the cell that would hold those games is rendered empty.
    """
    g = _frame([0.060], "HOME", "W", 0.70)
    lines = grade_leans._magnitude_price_grid_lines(g)
    bands = len(grade_leans.FIXED_MAGNITUDE_EDGES) - 1
    columns = len(grade_leans._fixed_price_edges()) - 1
    interior = [l for l in lines if l.startswith("    q ")
                and "all q" not in l][:bands * columns]
    assert len(interior) == bands * columns
    assert sum("(no rows)" in l for l in interior) == bands * columns - 1
    assert "q <.450        n=  0   (no rows)" in text_of(lines)


def test_a_one_game_cell_prints_its_spread_and_is_never_suppressed():
    """The p-hat form reads +-0.0 here, which is the failure this SE avoids."""
    g = _frame([0.005], "HOME", "W", 0.60)
    text = _grid(g)
    expected = 100 * float(np.sqrt(0.6 * 0.4))
    assert f"excess  +40.0 +- {expected:4.1f} pp" in text
    assert "+- 0.0 pp" not in text


def test_the_price_edges_have_exactly_one_home():
    """A second 0.45 would let the display drift from the rule it is read against."""
    edges = grade_leans._fixed_price_edges()
    assert edges[1] == hybrid_v2.THRESHOLD
    assert edges[3] == pytest.approx(1.0 - hybrid_v2.THRESHOLD)
    assert edges[2] == 0.50
    src = open(grade_leans.__file__, encoding="utf-8").read()
    assert not pd.Series([src]).str.contains(
        r"(?m)^_?[A-Z_]*THRESHOLD[A-Z_]*\s*=\s*0\.45").iloc[0]


def test_the_mirror_edge_is_rounded_so_a_ulp_cannot_move_a_column():
    """`1 - 0.45` is 0.55000000000000004; a displayed row must not care.

    The same asymmetry inside the registered rule is pinned as inert in
    tests/test_dog_contrast_test.py. It is a boundary of a frozen rule there
    and merely a column edge here, so here it is rounded away.
    """
    assert grade_leans._fixed_price_edges()[3] == 0.55
    g = _frame([0.005], "AWAY", "W", 0.45)      # q = 1 - .45 exactly
    cells = [l for l in grade_leans._magnitude_price_grid_lines(g)
             if "n=  1" in l]
    assert cells[0].strip().startswith("q .550+")


def test_both_blocks_read_the_same_magnitude_edges():
    """One home for the band cutoffs, so the two blocks cannot disagree."""
    text = _grid(_frame([0.005], "HOME", "W", 0.60))
    for lo, hi in zip(grade_leans.FIXED_MAGNITUDE_EDGES,
                      grade_leans.FIXED_MAGNITUDE_EDGES[1:]):
        label = f"[{lo:.3f}, {hi:.3f})" if np.isfinite(hi) else f"[{lo:.3f}, inf)"
        assert f"|delta| {label}" in text


def test_the_grid_is_pure():
    g = _frame([0.005, 0.030], "HOME", ["W", "L"], 0.60)
    before = g.copy(deep=True)
    _grid(g)
    assert_frame_equal(g, before)


def test_missing_columns_degrade_to_one_line():
    g = _frame([0.005], "HOME", "W", 0.60).drop(columns=["pregame_p_home"])
    assert _grid(g) == (f"{grade_leans.MODEL_METRIC_LABEL} |delta| x "
                        "saved-pregame market probability: unavailable "
                        "(missing report inputs)")


def test_an_all_unpriced_family_says_so_instead_of_rendering_cells():
    g = _frame([0.005], "HOME", "W", np.nan)
    text = _grid(g)
    assert "no pregame-priced rows yet" in text
    assert "(no rows)" not in text


def test_the_null_best_reference_is_deterministic_and_positive():
    """A committed artifact cannot carry a line that moves every build.

    And the reference must be well above zero at these cell sizes -- that is
    the whole point of printing it: the best of twenty thin cells is large
    under a market that is exactly right.
    """
    cells = [np.full(3, 0.55), np.full(4, 0.48), np.full(2, 0.41)]
    first = grade_leans._grid_null_best_excess(cells)
    assert first == grade_leans._grid_null_best_excess(cells)
    assert first > 0.10
    assert np.isnan(grade_leans._grid_null_best_excess([]))
    assert np.isnan(grade_leans._grid_null_best_excess([np.array([])]))


def test_the_block_states_its_basis_and_refuses_the_probability_reading():
    """Two claims, each load-bearing, pinned as claims rather than as copy."""
    text = _grid(_frame([0.005], "HOME", "W", 0.60))
    head = text.split("|delta| [")[0]
    assert "pregame_p_home" in head and "NO close fallback" in head
    assert "not a win probability" in head
    # The search caveat belongs to a frame that HAS a search; it is pinned in
    # test_the_search_correction_returns_once_there_is_a_maximum, because on
    # this one-cell frame printing it would be the degenerate reference.


def test_report_renders_the_grid_after_the_fixed_bands():
    ledger = pd.read_csv(grade_leans.LEDGER_PATH, low_memory=False)
    if grade_leans._record_grades(ledger).empty:
        return
    text = grade_leans.report_text(ledger)
    bands = text.index("fixed |delta| bands (current record family")
    grid = text.index("|delta| x saved-pregame market probability")
    assert bands < grid
    assert "unavailable (" not in text[grid:grid + 200]


def test_the_poisson_binomial_se_has_exactly_one_home():
    """Three callers, one derivation.

    `grade_leans` cannot import `build_site` -- that module refuses a
    non-xwOBA MODEL_TAG at import time and this one supports wOBA tags -- so
    the copy that would have been made here lives in `market_backfill`
    instead, beside `metric_label`, which has one home for the same reason.
    """
    import build_site
    import market_backfill
    probs = [0.4, 0.55, 0.62]
    assert build_site._excess_se(probs) == market_backfill.excess_se(probs)
    assert grade_leans.excess_se is market_backfill.excess_se
    src = open(build_site.__file__, encoding="utf-8").read()
    assert "(p * (1.0 - p)).sum()" not in src


def test_the_band_label_has_one_home_so_a_cell_can_never_go_missing():
    """The grid indexes cells by this string and renders by it too.

    Two copies drifting would not raise: the band would print with no cells
    under it. Pinned as the rule (every band renders its full row of columns)
    rather than as the format.
    """
    lines = grade_leans._magnitude_price_grid_lines(
        _frame([0.005, 0.015, 0.025, 0.035, 0.055], "HOME", "W", 0.60))
    columns = len(grade_leans._fixed_price_edges()) - 1
    starts = [i for i, l in enumerate(lines) if l.startswith("  |delta| [")]
    assert len(starts) == len(grade_leans.FIXED_MAGNITUDE_EDGES) - 1
    for i in starts:
        row = lines[i + 1:i + 1 + columns]
        assert all(l.startswith("    q ") for l in row)
        assert lines[i + 1 + columns].strip().startswith("all q")


def test_a_single_cell_is_not_reported_as_a_search():
    """The null best of one cell is zero by construction.

    Printed as a reference it would read "+40.0 against +0.0", which is the
    compare-against-zero the line's own last clause forbids. With nothing
    maximised over there is no correction to make, and the block says so
    rather than publishing a flattering null.
    """
    text = _grid(_frame([0.005], "HOME", "W", 0.60))
    assert "nothing was maximised over" in text
    assert "non-empty cells averages" not in text
    assert "A grid is a search" not in text


def test_the_search_correction_returns_once_there_is_a_maximum():
    g = _frame([0.005, 0.005], "HOME", ["W", "L"], [0.60, 0.47])
    text = _grid(g)
    assert "best of these 2 non-empty cells" in text
    assert "never against zero" in text


def test_a_one_column_band_names_the_margin_it_cannot_differ_from():
    """Two identical lines need a clause, or they read as a bug.

    The precedent is the chalk control on the per-game card: adjacency alone
    failed there, because equality to the decimal looks like duplicated data
    until something says why it is forced.
    """
    lines = grade_leans._magnitude_price_grid_lines(_frame([0.060], "HOME", "W", 0.70))
    i = next(i for i, l in enumerate(lines) if l.startswith("  |delta| [0.050"))
    band = lines[i:i + 7]
    assert band[-2].strip().startswith("all q")
    assert "can only repeat that cell" in band[-1]
    # A band spread over two columns states nothing of the sort.
    spread = grade_leans._magnitude_price_grid_lines(
        _frame([0.060, 0.060], "HOME", ["W", "L"], [0.70, 0.47]))
    assert not any("can only repeat" in l for l in spread)
