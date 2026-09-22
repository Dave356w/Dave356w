"""Guard V13 card provenance: native pregame vs reconstructed history.

V12 and V13 share a prediction family. The V13 SP rate blend changed only a
minority of near-zero selections on the historical paired rows. This does not
make reconstructed rows prospective bets: most saved wOBA shadow timestamps
are after their own game's scheduled start. Render both bases explicitly.
"""
from unittest import mock
import numpy as np
import pandas as pd
import build_site as b


def test_card_names_native_and_mixed_historical_bases():
    ctx = {
        "native": dict(n=48, w=35, l=13, excess_be=.018, excess_se=.069),
        "reconstructed_n": 445,
        "pooled": dict(n=493, excess_be=.060, excess_se=.022),
    }
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120),
        "SEA", "MIN", ctx, .0016,
    )
    assert "Historical context · native V13" in h
    assert "35–13 · 48 completed games" in h
    assert "Versus historical closing prices" in h
    assert "+1.8 ± 6.9 pts (1 SE)" in h
    assert "445 mixed-basis reconstructed rows are excluded" in h
    assert "Mixed-basis retrospective diagnostic" in h
    assert "+6.0 ± 2.2 pts · 493 completed games" in h
    assert "Not a prospective V13 record" in h
    assert "not the locked pregame quotes" in h
    assert "Beating this price" not in h


def test_native_aggregation_uses_original_model_tags():
    ledger = b.load_ledger_df()
    obs = b._lean_market_observations(ledger)
    assert len(obs), "committed ledger has no graded V13-represented leans"
    old_tags = ledger.loc[obs.index, "model_tag"].astype(str)
    native_obs = obs.loc[old_tags.eq(b.MODEL_TAG)]
    assert len(native_obs), "committed ledger has no native V13 games"
    ctx = b.hybrid_branch_records()
    native = ctx["native"]
    assert native["n"] == len(native_obs)
    assert native["w"] == int(native_obs["won"].sum())
    assert native["l"] == len(native_obs) - native["w"]
    assert ctx["reconstructed_n"] == len(obs) - len(native_obs)
    assert ctx["reconstructed_n"] + native["n"] == ctx["n"]
    be = b._mb_breakeven_prob(native_obs["close_ml"])
    expected_excess = float(native_obs["won"].mean() - np.mean(be))
    assert np.isclose(native["excess_be"], expected_excess)
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120), "SEA", "MIN",
        ctx, .0016,
    )
    assert f"{native['n']} completed games" in h
    assert f"{ctx['reconstructed_n']} mixed-basis reconstructed rows" in h


def test_reconstructed_only_card_never_invents_native_record():
    ctx = {"reconstructed_n": 12, "pooled": dict(
        n=12, excess_be=.15, excess_se=.08)}
    h = b._verdict_html(
        "MIN", dict(p_home=.521, home_ml=-120), "SEA", "MIN",
        ctx, .0016,
    )
    assert "No native V13 games graded yet" in h
    assert "12 mixed-basis reconstructions" in h
    assert "prospective record" in h
    assert "Cleared the posted price by" not in h


def test_missing_row_provenance_never_creates_a_native_record():
    """A synthetic observation frame cannot be assigned live V13 status."""
    obs = pd.DataFrame({
        "won": [1.0, 0.0],
        "market_p": [.55, .45],
        "market_resid": [.45, -.45],
        "close_ml": [-120.0, 115.0],
        "profit": [100 / 120, -1.0],
    })
    with mock.patch.object(b, "load_ledger_df",
                           return_value=pd.DataFrame([{}])):
        with mock.patch.object(b, "_lean_market_observations",
                               return_value=obs):
            ctx = b.hybrid_branch_records()
    assert ctx["n"] == 2
    assert ctx["pooled"]["n"] == 2
    assert "native" not in ctx
    assert "reconstructed_n" not in ctx
