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
