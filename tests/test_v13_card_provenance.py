"""The compact historical comparison uses V13-represented selection rows only.

Original model_tag separates native locked V13 from V12→V13 re-scoring. The
same price-band observations furnish the market's expectation and the
realised record; the overall 493-game figure is not repeated per matchup.
"""
from unittest import mock
import numpy as np
import pandas as pd
import build_site as b
from market_backfill import percentile_band_index


def test_matched_card_never_renders_full_family_average():
    ctx = {
        "native": dict(n=48, w=35, l=13, excess_be=.138, excess_se=.071),
        "reconstructed_n": 445,
        "pooled": dict(n=493, excess_be=.060, excess_se=.022),
        "model_distribution": {
            "games": 493, "price_min": -200, "price_max": 200,
            "edges": [],
            "bands": [dict(index=0, lo=-200, hi=200, n=12,
                           w=7, l=5, implied=.52, actual=7/12,
                           breakeven=.55, gap=7/12-.52,
                           excess_be=7/12-.55, se=.144,
                           native_n=2, reconstructed_n=10)],
        },
    }
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120),
        "SEA", "MIN", ctx, .0016)
    assert "V13 · matched historical price band" in h
    assert "12 model selections" in h
    assert "10 V12-era games re-decided by V13 / 2 native V13 picks" in h
    assert "Market implied" in h and "V13 realised" in h
    assert "Retrospective, 1 SE" in h
    assert "+6.0 ± 2.2 pp · 493" not in h
    assert "35–13; native closing margin" not in h
    assert "Combined V12/V13 historical performance" not in h


def test_each_band_provenance_and_scores_match_original_ledger_tags():
    ledger = b.load_ledger_df()
    obs = b._lean_market_observations(ledger)
    assert len(obs), "committed ledger has no graded V13-represented leans"
    ctx = b.hybrid_branch_records()
    dist = ctx["model_distribution"]
    assert dist["games"] == len(obs) == ctx["n"]
    native_mask = ledger.loc[obs.index, "model_tag"].astype(str).eq(b.MODEL_TAG)
    native_obs = obs.loc[native_mask]
    assert ctx["native"]["n"] == len(native_obs)
    assert ctx["reconstructed_n"] == len(obs) - len(native_obs)
    bucket = percentile_band_index(obs["close_ml"], dist["edges"])
    for rec in dist["bands"]:
        mask = bucket == rec["index"]
        subset = obs.iloc[np.flatnonzero(mask)]
        assert rec["n"] == len(subset)
        assert rec["w"] == int(subset["won"].sum())
        assert rec["native_n"] == int(native_mask.iloc[np.flatnonzero(mask)].sum())
        assert rec["reconstructed_n"] == rec["n"] - rec["native_n"]
        assert np.isclose(rec["implied"], subset["market_p"].mean())
        be = np.asarray(b._mb_breakeven_prob(subset["close_ml"]), dtype=float)
        assert np.isclose(rec["excess_be"], subset["won"].mean()-be.mean())
    assert sum(x["native_n"] for x in dist["bands"]) == ctx["native"]["n"]
    assert sum(x["reconstructed_n"] for x in dist["bands"]) == ctx["reconstructed_n"]


def test_reconstructed_only_band_does_not_claim_native_provenance():
    ctx = {"model_distribution": dict(
        games=12, price_min=-200, price_max=200, edges=[],
        bands=[dict(index=0, lo=-200, hi=200, n=12,
                    w=8, l=4, implied=.50, actual=8/12,
                    breakeven=.53, gap=8/12-.5,
                    excess_be=8/12-.53, se=.144,
                    native_n=0, reconstructed_n=12)])}
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120),
        "SEA", "MIN", ctx, .0016)
    assert "12 V12-era games re-decided by V13 / 0 native V13 picks" in h
    assert "Full history" in h
    assert "not a game-specific probability" in h
    assert "completed games" not in h


def test_missing_source_tag_never_invents_native_rows():
    obs = pd.DataFrame({
        "won": [1., 0.], "market_p": [.55, .45],
        "market_resid": [.45, -.45], "close_ml": [-120., 115.],
        "profit": [100 / 120, -1.],
    })
    with mock.patch.object(b, "load_ledger_df",
                           return_value=pd.DataFrame([{}])):
        with mock.patch.object(b, "_lean_market_observations",
                               return_value=obs):
            ctx = b.hybrid_branch_records()
    assert ctx["n"] == 2
    assert "native" not in ctx
    assert "reconstructed_n" not in ctx
    assert "model_distribution" not in ctx  # fewer than 32 data points
