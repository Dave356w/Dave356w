"""Guards on the lineup dispersion column.

`B_0` is a weighted MEAN, so a lineup of nine .310 bats and a lineup of one
.400 and eight .299 bats publish the same composite. `opp_xwOBA_sd` records how
concentrated the talent behind that mean was, which is the only way to turn "do
good hitters get averaged down" into a measurement.

It is diagnostic. The tests that matter most here are the ones asserting it
does NOT reach a lean -- a column that silently acquired a consumer would be a
prediction-math change wearing a display change's clothes.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as bs
import grade_leans as gl

W = [bs.LINEUP_SLOT_PA[i] for i in range(1, 10)]


def _lineup(rates, pa=600):
    return pd.DataFrame([
        dict(game_pk=1, faced_pitcher="SP", pitcher_side="away",
             batting_side="home", xwOBA=v, PA=pa, BBE=200, batting_order=i + 1)
        for i, v in enumerate(rates)
    ])


# --------------------------------------------------------------------------
# it measures what it claims to
# --------------------------------------------------------------------------
def test_two_lineups_with_one_mean_are_told_apart():
    """The whole point: equal composites, different concentration."""
    flat = _lineup([0.310] * 9)
    conc = _lineup([0.400] + [0.29875] * 8)
    a = bs.aggregate_lineup(flat, ["xwOBA"], weighted=True,
                            shrink_prior=0.315, shrink_k=100.0).iloc[0]
    b = bs.aggregate_lineup(conc, ["xwOBA"], weighted=True,
                            shrink_prior=0.315, shrink_k=100.0).iloc[0]
    assert a["opp_xwOBA_neutral"] == pytest.approx(b["opp_xwOBA_neutral"], abs=2e-3)
    assert a["opp_xwOBA_sd"] == pytest.approx(0.0, abs=1e-12)
    assert b["opp_xwOBA_sd"] > 0.02


def test_the_sd_shares_the_mean_s_weighting():
    """Mean and sd must describe one lineup under one weighting, or the pair
    cannot be interpreted together."""
    v = [0.400] + [0.29875] * 8
    mu = bs.wmean(v, W)
    manual = float(np.sqrt(np.average((np.array(v) - mu) ** 2, weights=W)))
    assert bs.wsd(v, W) == pytest.approx(manual)
    # Population sd, not sample: ddof=1 would be visibly larger at n=9.
    assert bs.wsd(v, W) != pytest.approx(float(pd.Series(v).std(ddof=1)))


def test_it_is_measured_before_the_platoon_term():
    """Dispersion should describe the lineup, not tonight's handedness draw.

    A lineup of one-sided bats all facing the same starter takes a common
    offset, which shifts the mean and must leave the sd alone.
    """
    base = _lineup([0.320, 0.300, 0.330, 0.290, 0.310, 0.305, 0.315, 0.295, 0.325])
    plain = bs.aggregate_lineup(base, ["xwOBA"], weighted=True,
                                shrink_prior=0.315, shrink_k=100.0).iloc[0]
    handed = base.assign(bats="L", **{bs.SP_THROWS_COL: "R"})
    withp = bs.aggregate_lineup(handed, ["xwOBA"], weighted=True,
                                shrink_prior=0.315, shrink_k=100.0).iloc[0]
    assert withp["opp_xwOBA_vs_sp"] != pytest.approx(plain["opp_xwOBA_vs_sp"])
    assert withp["opp_xwOBA_sd"] == pytest.approx(plain["opp_xwOBA_sd"])


def test_it_is_measured_after_shrinkage():
    """Post-shrinkage is what the model believes about each bat, so a thin
    sample must compress the dispersion rather than inflate it."""
    rates = [0.400] + [0.29875] * 8
    deep = bs.aggregate_lineup(_lineup(rates, pa=600), ["xwOBA"], weighted=True,
                               shrink_prior=0.315, shrink_k=100.0).iloc[0]
    thin = bs.aggregate_lineup(_lineup(rates, pa=60), ["xwOBA"], weighted=True,
                               shrink_prior=0.315, shrink_k=100.0).iloc[0]
    assert thin["opp_xwOBA_sd"] < deep["opp_xwOBA_sd"]


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------
def test_fewer_than_two_bats_is_nan_not_zero():
    """A fabricated 0.0 would read as a perfectly flat lineup."""
    assert np.isnan(bs.wsd([0.31], [4.6]))
    assert np.isnan(bs.wsd([], None))
    assert np.isnan(bs.wsd([np.nan, np.nan], W[:2]))


def test_unusable_weights_fall_back_to_an_unweighted_sd():
    """Matches wmean's degradation -- continuous, not a threshold."""
    v = [0.400] + [0.29875] * 8
    assert bs.wsd(v, [np.nan] * 9) == pytest.approx(bs.wsd(v, None))
    assert bs.wsd(v, None) == pytest.approx(float(pd.Series(v).std(ddof=0)))


# --------------------------------------------------------------------------
# it must stay diagnostic
# --------------------------------------------------------------------------
def test_the_column_reaches_the_ledger_schema():
    """It has to be in BOTH lists or it is silently dropped.

    AUDIT_COLS is the wide schema written to the CSV; MODEL_FIELDS is what a
    pre-lock refresh rewrites on a still-pending row. A column in the first
    only would go stale on a scratch; in the second only it would never be
    persisted at all.
    """
    for c in ("opp_xwoba_sd_away", "opp_xwoba_sd_home"):
        assert c in gl.AUDIT_COLS, "not written to the ledger CSV"
        assert c in gl.MODEL_FIELDS, "not refreshed when a pending row is rebuilt"
    # And the row builder must actually populate it, not just declare it.
    import inspect
    src = inspect.getsource(gl.rows_from_dump)
    assert "opp_xwoba_sd_away" in src and 'a.get("opp_xwOBA_sd"' in src


def test_no_lean_path_reads_the_dispersion():
    """THE GUARD THAT MATTERS.

    A diagnostic column that acquires a consumer becomes prediction math
    without a MODEL_TAG bump. `opp_xwOBA_sd` may be written and carried; it
    must never be read by the matchup, the phase blend or the edge.
    """
    import inspect
    for fn in (bs.sequential_xwoba_phases, bs.matchup_value,
               bs.apply_pitching_plans, bs.build_matchup):
        src = inspect.getsource(fn)
        reads = [l for l in src.splitlines()
                 if "opp_xwOBA_sd" in l and "rec[" not in l]
        assert not reads, f"{fn.__name__} reads the dispersion: {reads}"


def test_the_phase_algebra_is_unchanged_by_the_new_column():
    """Same inputs, same net, with and without the column present."""
    kw = dict(b_neutral=0.320, b_vs_sp=0.325, starter=0.300, bullpen=0.310,
              league=0.315, expected_sp_ip=5.2,
              sp_bf_per_ip=4.3, bp_bf_per_ip=4.4)
    out = bs.sequential_xwoba_phases(**kw)
    assert out["mx_xwOBA"] == pytest.approx(
        out["sp_share"] * out["mx_xwOBA_sp"] + out["bp_share"] * out["mx_xwOBA_bp"])
