"""Fixed report boundaries and paired denominators; no live outcome literals."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import grade_leans


def _frame(values):
    n = len(values)
    return pd.DataFrame({
        "xw_net": values, "xw_lean": ["HOME"] * n,
        "xw_full": ["W"] * n, "xw_f5": ["T"] * n,
        "close_p_home": [0.6] * n,
        "full_home": [4] * n, "full_away": [2] * n,
    })


def test_fixed_boundaries_absolute_values_and_empty_bands():
    g = _frame([0.0, 0.009999, -0.010, 0.020, 0.030, 0.050,
                np.nan, np.inf, "invalid", 0.005])
    g.loc[0, "xw_lean"] = None
    g.loc[9, "xw_full"] = None
    before = g.copy(deep=True)
    text = "\n".join(grade_leans._fixed_magnitude_lines(g))
    assert "Included 5 of 10 graded rows; excluded 5" in text
    for label in ("[0.000, 0.010)", "[0.010, 0.020)", "[0.020, 0.030)",
                  "[0.030, 0.050)", "[0.050, inf)"):
        assert f"{label} n=1" in text
    assert "decisions=0; 95% CI unavailable" in text
    assert_frame_equal(g, before)
    empty = "\n".join(grade_leans._fixed_magnitude_lines(g.iloc[:0]))
    assert empty.count(") n=0") == 5


def test_favorite_and_model_share_only_valid_closing_price_rows():
    g = _frame([0.005] * 5)
    g["xw_full"] = ["W", "L", "W", "W", "W"]
    g.loc[1, "xw_lean"] = "AWAY"
    g["close_p_home"] = [0.6, 0.4, np.nan, 1.0, 0.7]
    g.loc[4, "full_away"] = 4  # Tied full score is not a comparable decision.
    g["pregame_p_home"] = 0.8  # Must never fill a missing close.
    text = "\n".join(grade_leans._fixed_magnitude_lines(g))
    assert "closing-price paired n=2; excluded=3" in text
    assert f"{grade_leans.MODEL_METRIC_LABEL} 1-1  (0.500); decisions=2" in text
    assert "favorite 1-1  (0.500); decisions=2" in text


def test_intervals_handle_small_samples_and_f5_ties():
    record = grade_leans._magnitude_record(pd.Series(["W", "L", "T"]))
    assert "1-1-1  (0.500); decisions=2; 95% CI [0.095, 0.905]" in record
    single = grade_leans._magnitude_record(pd.Series(["W"]))
    assert "95% CI [0.207, 1.000]" in single


def test_report_includes_fixed_bands_without_changing_existing_terciles():
    ledger = pd.read_csv(grade_leans.LEDGER_PATH, low_memory=False)
    text = grade_leans.report_text(ledger)
    current = grade_leans._record_grades(ledger)
    if current.empty:
        return
    assert "fixed |delta| bands (current record family; descriptive)" in text
    if len(current) >= 9:
        assert "F5 by |" in text and "tercile:" in text
