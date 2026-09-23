"""Presentation-only tests for the game-card equal-count market distribution.

It MUST share the exact partition with grade_leans' market report. These are
all-family closing-price distributions, not the model's selection history, and
current quotes must never enter the historical sample.
"""
import numpy as np
import pandas as pd

import build_site as b
import grade_leans as gl
from market_backfill import percentile_band_index, percentile_price_edges


BOOK = [
    (-400, 280), (-220, 180), (-160, 140), (-130, 115),
    (-120, 105), (-110, -110), (-105, -105), (105, -125),
]


def _ledger(n=64):
    rows = []
    for i in range(n):
        hm, am = BOOK[i % len(BOOK)]
        hi = b._imp_ml(hm)
        ai = b._imp_ml(am)
        ph = hi / (hi + ai)
        rows.append({
            "status": "graded",
            "model_tag": "woba+plat_consol_v5" if i % 2 else "xw+starter_blend_v13",
            "game_pk": 2000 + i,
            "full_home": 5 if i % 3 else 2,
            "full_away": 2 if i % 3 else 5,
            "close_p_home": ph,
            "close_home_ml": hm,
            "close_away_ml": am,
        })
    return pd.DataFrame(rows)


def test_equal_count_edges_match_analyst_market_report():
    led = _ledger()
    dist = b._market_price_distribution(led)
    ml = np.r_[led["close_home_ml"].to_numpy(), led["close_away_ml"].to_numpy()]
    assert dist["edges"] == gl._percentile_price_edges(ml, 8)
    assert dist["edges"] == percentile_price_edges(ml, 8)
    assert np.array_equal(
        percentile_band_index(ml, dist["edges"]),
        gl._percentile_band_index(ml, dist["edges"]),
    )
    assert dist["games"] == len(led)
    assert dist["sides"] == 2 * len(led)
    assert sum(r["n"] for r in dist["bands"]) == dist["sides"]
    assert sum(r["pairs"] for r in dist["bands"]) <= dist["games"]


def test_each_band_scores_both_team_sides_without_using_model_tags():
    led = _ledger()
    one = b._market_price_distribution(led)
    # Model tags cannot change a market-only statistic.
    led["model_tag"] = "unrelated_historical_tag"
    two = b._market_price_distribution(led)
    assert one == two
    for rec in one["bands"]:
        assert np.isclose(rec["gap"], rec["actual"] - rec["implied"])
        assert 0 < rec["se"] < 0.5
        assert rec["lo"] <= rec["hi"]


def test_game_card_highlights_only_its_matching_historical_price_band():
    dist = b._market_price_distribution(_ledger())
    ctx = {
        "market_distribution": dist,
        "pooled": dict(n=493, excess_be=.060, excess_se=.022),
    }
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120), "SEA", "MIN", ctx, .0016)
    band = int(percentile_band_index([-120], dist["edges"])[0])
    assert "Historical market context" in h
    assert f"Current -120 falls in band {band + 1} of {len(dist['bands'])}" in h
    assert f"· {dist['games']} games · closing prices only" in h
    assert h.count("vband-step selected") == 1
    assert "Historic prices" in h and "Implied wins" in h and "Actual wins" in h
    assert "Historical calibration gap:" in h
    assert "not this model's performance or a forecast" in h
    assert h.index("Historical market context") < h.index("Combined V12/V13")
    assert "<details class='vhistory'>" in h
    assert "Past margin over closing break-even" in h
    assert "Full model-family record" in h


def test_no_historical_band_is_invented_for_an_unpriced_or_outside_quote():
    dist = b._market_price_distribution(_ledger())
    ctx = {"market_distribution": dist}
    assert b._market_band_context_html(ctx, None) == ""
    assert b._market_band_context_html(ctx, -80) == ""
    x = b._market_band_context_html(ctx, -1000)
    assert "outside the archive" in x
    assert "Historical calibration gap:" not in x


def test_thin_ledger_has_no_market_band_panel():
    thin = b._market_price_distribution(_ledger(6))
    assert thin is None
    ctx = {"market_distribution": thin}
    assert b._market_band_context_html(ctx, -120) == ""
