"""Reporting-only guards for calibration, price provenance, and field labels."""

import numpy as np
import pandas as pd

import abstain_test
import delta_filter_test
import dog_contrast_test
import forward_test
import hybrid_test


def test_every_registered_report_names_its_price_snapshot_and_fallback():
    close_modules = (forward_test, delta_filter_test)
    pregame_modules = (hybrid_test, abstain_test, dog_contrast_test)
    unusable = pd.DataFrame({"status": ["graded"]})
    for module in close_modules:
        text = "\n".join(module.report_lines(unusable))
        assert "price source" in text
        assert "closing" in text
        assert "No pregame fallback" in text
    for module in pregame_modules:
        text = "\n".join(module.report_lines(unusable))
        assert "price source" in text
        assert "pregame" in text
        assert "closing fallback" in text or "close fallback" in text


def test_same_selection_price_reconciliation_is_derived_from_the_ledger():
    ledger = pd.read_csv(abstain_test.LEDGER, low_memory=False)
    g = abstain_test.scored_rows(ledger)
    text = "\n".join(abstain_test.report_lines(ledger))
    if g is None or g.empty:
        return
    chalk_home = (g["pregame_p_home"] >= 0.5).to_numpy()
    close_ml = np.where(chalk_home, g["close_home_ml"], g["close_away_ml"])
    comparable = np.isfinite(pd.to_numeric(close_ml, errors="coerce"))
    won = g["chalk_won"].astype(bool).to_numpy()[comparable]
    saved = float(g.loc[g.index[comparable], "chalk_profit"].sum())
    close = float(np.where(won, abstain_test._payout(close_ml[comparable]), -1).sum())
    expected = (f"n={int(comparable.sum())}, {int(won.sum())}-"
                f"{int(comparable.sum() - won.sum())}")
    assert expected in text
    assert f"saved pregame {saved:+.4f}u" in text
    assert f"closing {close:+.4f}u" in text
    assert "selections are not recalculated" in text


def test_historical_sections_name_closing_prices():
    import grade_leans

    ledger = pd.read_csv(grade_leans.LEDGER_PATH, low_memory=False)
    text = grade_leans.report_text(ledger)
    assert "price source — all retrospectives: closing close_p_home" in text
    assert "price source — selection/eligibility and market comparison: closing" in text


def test_every_registration_prints_the_slate_it_last_scored():
    """A forward sample reads as accruing, so a stalled one has to be visible.

    Three of the five stopped dead on 2026-09-11 -- `hybrid_test` and, through
    its delegated row selector, `abstain_test` and `dog_contrast_test` -- and
    every block went on printing gates that implied rows were still arriving.
    Nothing on any surface carried the one fact that would have shown it.

    Pinned across ALL SIX registrations rather than the three that stalled,
    because the next stall will be somewhere else, and pinned through
    `row_supply_line` so the clause has one home: six modules spelled this line
    six times, and a seventh copy is how one of them goes stale unnoticed.
    """
    import hybrid_v2
    import market_backfill
    modules = (forward_test, delta_filter_test, hybrid_test, hybrid_v2,
               abstain_test, dog_contrast_test)
    led = pd.read_csv("data/mlb_lean_ledger.csv", low_memory=False)
    for module in modules:
        src = open(module.__name__ + ".py", encoding="utf-8").read()
        assert "row_supply_line" in src, module.__name__
        assert "since registration: {len(g)}" not in src, module.__name__
        text = "\n".join(module.report_lines(led))
        assert "since registration:" in text, module.__name__
        assert "(last scored 2026-" in text, module.__name__

    # The clause says what it claims to: the LAST slate in the scored frame.
    g = pd.DataFrame({"game_date": ["2026-09-04", "2026-09-16", "2026-09-04"]})
    line = market_backfill.row_supply_line(g)
    assert "3 over 2 slates (last scored 2026-09-16)" in line
    assert market_backfill.row_supply_line(g, "dog leans").startswith(
        "    dog leans since registration:")
    # An empty sample has no last slate to name and must not invent one.
    for empty in (None, pd.DataFrame({"game_date": []})):
        assert "last scored" not in market_backfill.row_supply_line(empty)


def test_both_surfaces_naming_the_abstain_branch_name_the_same_one():
    """The registered declined set is the Q-GATE fade, not the shipped rule's.

    v2 added a delta gate on 2026-09-11 and fades a strict subset of what this
    registration declines -- 16 of the 37 current-family q-gate fades are games
    the shipped rule FOLLOWS. The 2026-09-17 correction that established this
    rewrote every copy of the claim inside `abstain_test` and missed the one in
    `grade_leans`'s retrospective block, which is the MORE prominent of the
    two: it prints near the top of the report while the corrected wording sits
    300 lines below. A caveat travels with the line someone wrote it on, so
    this asserts the property on BOTH surfaces at once -- a test naming one
    would pass while the other drifted, which is exactly what happened.
    """
    import grade_leans
    led = pd.read_csv("data/mlb_lean_ledger.csv", low_memory=False)
    g = led[led["model_tag"].astype(str).eq(grade_leans.MODEL_TAG)]
    surfaces = {
        "retrospective": "\n".join(grade_leans._registration_retrospective_lines(g)),
        "forward": "\n".join(abstain_test.report_lines(led)),
    }
    for name, text in surfaces.items():
        assert "q-gate fade branch" in text, name
        assert "shipped fade branch" not in text, name
