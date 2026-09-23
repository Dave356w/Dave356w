"""The game card compares V13 outcomes with the market on IDENTICAL rows.

One selected side per V13-represented graded game. The eight equal-count
bands are computed from V13-selected closing prices, not all other model
families or both sides of the broader market. No current odds enter history.
"""
import numpy as np
import pandas as pd
import build_site as b
from market_backfill import percentile_band_index, percentile_price_edges


BOOK = [
    (-400, 280), (-260, 225), (-220, 180), (-175, 155),
    (-160, 140), (-145, 125), (-130, 115), (-120, 105),
    (-110, -110), (-105, -105), (+105, -125), (+115, -135),
    (+130, -150), (+150, -180), (+175, -205), (+210, -245),
]


def _ledger(n=144):
    rows = []
    for i in range(n):
        hm, am = BOOK[i % len(BOOK)]
        h = b._imp_ml(hm)
        a = b._imp_ml(am)
        home_wins = i % 3 != 0
        home = 5 if home_wins else 2
        away = 2 if home_wins else 5
        source = (b.MODEL_TAG if i < 96 else
                  "xw+plat_consol_v12" if i < 120 else
                  "woba+plat_consol_v5")
        original_lean = "H" if i % 4 else "A"
        rebuilt_lean = (("A" if original_lean == "H" else "H")
                        if i % 7 == 0 else original_lean)
        # Original V12 result remains immutable; model uses the reconstructed
        # result for the comparable family, never the old side's record.
        rows.append({
            "status": "graded", "model_tag": source, "game_pk": 3000 + i,
            "game_date": "2026-09-01", "home": "H", "away": "A",
            "full_home": home, "full_away": away,
            "xw_lean": original_lean, "xw_net": .02,
            "xw_full": ("W" if (original_lean == "H") == home_wins else "L"),
            "close_p_home": h / (h + a),
            "close_home_ml": hm, "close_away_ml": am,
            "v13_recon_basis": "paired" if source == "xw+plat_consol_v12" else None,
            "v13_lean_recon": rebuilt_lean if source == "xw+plat_consol_v12" else None,
            "v13_net_recon": .025 if source == "xw+plat_consol_v12" else np.nan,
        })
    return pd.DataFrame(rows)


def test_bands_include_only_v13_represented_selected_sides():
    led = _ledger()
    obs = b._lean_market_observations(led)
    dist = b._market_price_distribution(led)
    assert len(obs) == 120  # 96 native + 24 V12 re-scored, no old wOBA
    assert dist["games"] == len(obs)
    assert sum(x["n"] for x in dist["bands"]) == len(obs)
    assert sum(x["native_n"] for x in dist["bands"]) == 96
    assert sum(x["reconstructed_n"] for x in dist["bands"]) == 24
    assert dist["edges"] == percentile_price_edges(obs["close_ml"], 8)
    ix = percentile_band_index(obs["close_ml"], dist["edges"])
    assert sum(int((ix == rec["index"]).sum()) for rec in dist["bands"]) == len(obs)


def test_market_and_v13_measurements_use_identical_game_rows():
    led = _ledger()
    obs = b._lean_market_observations(led)
    dist = b._market_price_distribution(led, obs)
    ix = percentile_band_index(obs["close_ml"], dist["edges"])
    for rec in dist["bands"]:
        same = obs.iloc[np.flatnonzero(ix == rec["index"])]
        be = np.asarray(b._mb_breakeven_prob(same["close_ml"]), dtype=float)
        assert rec["n"] == len(same)
        assert rec["w"] == int(same["won"].sum())
        assert rec["l"] == len(same) - rec["w"]
        assert np.isclose(rec["implied"], same["market_p"].mean())
        assert np.isclose(rec["actual"], same["won"].mean())
        assert np.isclose(rec["breakeven"], be.mean())
        assert np.isclose(rec["gap"], rec["actual"] - rec["implied"])
        assert np.isclose(rec["excess_be"], rec["actual"] - be.mean())
        assert np.isclose(rec["se"], b._excess_se(same["market_p"]))


def test_unrelated_model_family_cannot_change_market_baseline():
    led = _ledger()
    one = b._market_price_distribution(led)
    led.loc[led["model_tag"].eq("woba+plat_consol_v5"), "full_home"] = 50
    two = b._market_price_distribution(led)
    assert one == two


def test_card_compares_market_and_model_within_exact_matching_band():
    led = _ledger()
    dist = b._market_price_distribution(led)
    ctx = {
        "model_distribution": dist,
        "pooled": {"n": 493, "excess_be": .060, "excess_se": .022},
    }
    h = b._verdict_html(
        "MIN", {"p_home": .521, "home_ml": -120},
        "SEA", "MIN", ctx, .0016,
    )
    band = int(percentile_band_index([-120], dist["edges"])[0])
    rec = next(x for x in dist["bands"] if x["index"] == band)
    assert "V13 · matched historical price band" in h
    assert f"{rec['n']} model selections" in h
    assert f"{100*rec['implied']:.1f}%" in h
    assert f"{100*rec['actual']:.1f}%" in h
    assert f"{rec['w']}–{rec['l']}" in h
    assert f"{rec['reconstructed_n']} V12 re-scored" in h
    assert f"{rec['native_n']} native V13" in h
    assert "Market implied" in h and "V13 realised" in h
    assert "Vs market" in h and "vs closing break-even" in h
    assert h.count("vband-step selected") == 1
    assert "Historical market context" not in h
    assert "Combined V12/V13 historical performance" not in h
    assert "+6.0 ± 2.2 pp" not in h


def test_current_quote_changes_only_selected_comparison_not_historical_rows():
    dist = b._market_price_distribution(_ledger())
    ctx = {"model_distribution": dist}
    a = b._market_band_context_html(ctx, -400)
    z = b._market_band_context_html(ctx, +210)
    assert a != z
    assert dist["games"] == 120
    assert b._market_band_context_html(ctx, None) == ""
    assert b._market_band_context_html(ctx, -80) == ""
    assert "outside the observed V13" in b._market_band_context_html(ctx, -1000)


def test_thin_v13_sample_does_not_invent_price_bands():
    thin = b._market_price_distribution(_ledger(6))
    assert thin is None
    assert b._market_band_context_html({"model_distribution": thin}, -120) == ""
