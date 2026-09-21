import numpy as np
import pandas as pd
import pytest

from research import delta_market_audit as audit


def rows():
    # Ten days, twenty games a day, with mixed outcomes and market strengths.
    i = np.arange(200)
    q = .55 + (i % 4) * .05
    return pd.DataFrame({
        "game_pk": i+1,
        "game_date": [f"2026-08-{1+j//20:02d}" for j in i],
        "cell_id": "D3_M4", "market_q": q,
        "executable_breakeven": q+.02, "won": i%3 != 0,
        "abs_delta": .020+(i%5)*.001, "row_basis": "reconstructed",
    })


def test_current_slate_outcomes_cannot_change_its_forecasts():
    frame = rows()
    original = audit.replay(frame)
    altered = frame.copy()
    altered.loc[altered.game_date.ge("2026-08-08"), "won"] = ~altered.loc[altered.game_date.ge("2026-08-08"), "won"]
    changed = audit.replay(altered)
    columns = [c for c in original if c.startswith("p_")]
    pd.testing.assert_frame_equal(
        original.loc[original.game_date.le("2026-08-08"), columns],
        changed.loc[changed.game_date.le("2026-08-08"), columns],
    )
    assert original.loc[original.game_date.eq("2026-08-08"), "train_n"].eq(140).all()
    assert original.loc[original.game_date.eq("2026-08-08"), "train_slates"].eq(7).all()


def test_residual_correction_preserves_current_price_within_cell():
    out = audit.replay(rows())
    day = out[out.game_date.eq("2026-08-02")]
    assert day.p_existing_grid.nunique() == 1
    np.testing.assert_allclose(day.p_residual_grid-day.p_market,
                               np.repeat((day.p_residual_grid-day.p_market).iloc[0], len(day)))
    assert day.p_residual_grid.nunique() == 4


def test_duplicate_game_is_rejected():
    frame = rows()
    with pytest.raises(ValueError, match="one identified row per game"):
        audit.replay(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_row_order_and_basis_boundary_do_not_reset_history():
    frame = rows()
    frame.loc[frame.game_date.ge("2026-08-08"), "row_basis"] = "native"
    expected = audit.replay(frame)
    shuffled = audit.replay(frame.sample(frac=1, random_state=7).reset_index(drop=True))
    columns = [c for c in expected if c.startswith("p_")]+["train_n", "train_slates"]
    pd.testing.assert_frame_equal(
        expected.set_index("game_pk")[columns].sort_index(),
        shuffled.set_index("game_pk")[columns].sort_index(),
    )
    assert expected.loc[expected.row_basis.eq("native"), "train_n"].min() == 140
