import numpy as np
import pandas as pd

import price_calibration_shadow as pcs


def _prepared(rows):
    return pd.DataFrame(rows)


def test_fixed_band_boundaries():
    assert pcs.M0 == 10.0
    assert pcs.RULE_TAG == "v13_dynamic_price_shadow_m10_h015"

    assert pcs.delta_band(0.000)[0] == "D1"
    assert pcs.delta_band(0.010)[0] == "D2"
    assert pcs.delta_band(0.020)[0] == "D3"
    assert pcs.delta_band(0.030)[0] == "D4"
    assert pcs.delta_band(0.050)[0] == "D5"

    assert pcs.market_band(0.4499)[0] == "M1"
    assert pcs.market_band(0.4500)[0] == "M2"
    assert pcs.market_band(0.5000)[0] == "M3"
    assert pcs.market_band(0.5500)[0] == "M4"


def test_same_slate_rows_share_frozen_state_then_next_slate_updates():
    rows = _prepared([
        {
            "game_date": "2026-07-01",
            "cell_id": "D4_M4",
            "market_q": 0.60,
            "executable_breakeven": 0.61,
            "won": True,
        },
        {
            "game_date": "2026-07-01",
            "cell_id": "D4_M4",
            "market_q": 0.62,
            "executable_breakeven": 0.61,
            "won": False,
        },
        {
            "game_date": "2026-07-02",
            "cell_id": "D4_M4",
            "market_q": 0.61,
            "executable_breakeven": 0.58,
            "won": True,
        },
    ])

    out = pcs.evaluate_rows(rows)

    # Neither same-day game may train the other.
    assert out.loc[0, "cell_n_pregame"] == 0
    assert out.loc[1, "cell_n_pregame"] == 0
    assert out.loc[0, "shrinkage_lambda_pregame"] == 0
    assert out.loc[1, "shrinkage_lambda_pregame"] == 0
    assert out.loc[0, "adjusted_probability_pregame"] == 0.60
    assert out.loc[1, "adjusted_probability_pregame"] == 0.62

    # The next slate sees both prior games: n=2, W=1, qbar=.61.
    assert out.loc[2, "cell_n_pregame"] == 2
    assert out.loc[2, "cell_wins_pregame"] == 1
    assert np.isclose(out.loc[2, "cell_mean_market_p_pregame"], 0.61)
    assert np.isclose(out.loc[2, "shrinkage_lambda_pregame"], 2 / 12)
    assert np.isclose(out.loc[2, "adjusted_probability_pregame"], (1 + 10 * 0.61) / 12)
    assert np.isclose(out.loc[2, "estimated_edge"], ((1 + 10 * 0.61) / 12) - 0.58)
    assert out.loc[2, "decision_action"] == "QUALIFY"


def test_sparse_heavy_favorite_example_shrinks_to_abstain():
    # Eight prior games in one slate: 6-2 at mean q=.665. The next slate sees
    # exactly the specification's sparse-cell state before evaluating -240.
    rows = []
    for i in range(8):
        rows.append(
            {
                "game_date": "2026-07-01",
                "cell_id": "D5_M4",
                "market_q": 0.665,
                "executable_breakeven": 0.68,
                "won": i < 6,
            }
        )
    rows.append(
        {
            "game_date": "2026-07-02",
            "cell_id": "D5_M4",
            "market_q": 0.665,
            "executable_breakeven": 240 / 340,
            "won": True,
        }
    )

    out = pcs.evaluate_rows(_prepared(rows))
    r = out.iloc[-1]

    assert r["cell_n_pregame"] == 8
    assert r["cell_wins_pregame"] == 6
    assert np.isclose(r["cell_mean_market_p_pregame"], 0.665)
    assert np.isclose(r["shrinkage_lambda_pregame"], 8 / 18)
    assert np.isclose(r["adjusted_probability_pregame"], (6 + 10 * 0.665) / 18)
    assert r["decision_action"] == "ABSTAIN"


def test_report_combines_reconstructed_and_native_without_reset():
    led = pd.DataFrame([
        {
            "status": "graded",
            "model_tag": "xw+plat_consol_v12",
            "game_date": "2026-07-01",
            "game_pk": 1,
            "home": "AAA",
            "away": "BBB",
            "xw_lean": "BBB",
            "xw_net": -0.020,
            "xw_full": "L",
            "full_home": 5,
            "full_away": 3,
            "close_p_home": 0.60,
            "close_home_ml": -150,
            "close_away_ml": 130,
            "v13_net_recon": 0.020,
            "v13_lean_recon": "AAA",
            "v13_delta_recon": 0.020,
            "v13_recon_basis": "mixed_hindsight",
        },
        {
            "status": "graded",
            "model_tag": "xw+starter_blend_v13",
            "game_date": "2026-07-02",
            "game_pk": 2,
            "home": "CCC",
            "away": "DDD",
            "xw_lean": "CCC",
            "xw_net": 0.021,
            "xw_full": "W",
            "full_home": 4,
            "full_away": 2,
            "close_p_home": 0.60,
            "close_home_ml": -150,
            "close_away_ml": 130,
            "v13_net_recon": np.nan,
            "v13_lean_recon": np.nan,
            "v13_delta_recon": np.nan,
            "v13_recon_basis": np.nan,
        },
    ])

    out = pcs.analysis_frame(led)
    assert list(out["row_basis"]) == ["reconstructed", "native"]
    assert out.iloc[0]["cell_n_pregame"] == 0
    assert out.iloc[1]["cell_n_pregame"] == 1

    text = "\n".join(pcs.report_lines(led))
    assert "1 reconstructed + 1 native v13" in text
    assert "native boundary does NOT reset n" in text
    assert "CLOSING" in text
    assert "descriptive shadow only" in text
