"""Guards on the interaction probe's calibrated-q candidate.

The probe benchmarks alternatives against the shipped rule, so its failure mode
is not a crash -- it is a candidate that quietly stops meaning what its label
says and makes the shipped model look better or worse than it is.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as bs
import interaction_probe as ip


def _frame(iph, iph_raw, qh, r_sp=4.3, r_bp=4.3):
    n = len(iph)
    return pd.DataFrame({
        "iph": iph, "iph_raw": iph_raw, "qh": qh,
        "rh_sp": [r_sp] * n, "rh_bp": [r_bp] * n,
        "ipa": iph, "ipa_raw": iph_raw, "qa": qh,
        "ra_sp": [r_sp] * n, "ra_bp": [r_bp] * n,
    })


def test_q_from_ip_matches_the_shipped_formula():
    """Mirrored, not re-derived -- if these drift the probe benchmarks a model
    that never ran."""
    for ipv, rsp, rbp in [(5.2, 4.3, 4.5), (3.0, 4.0, 4.0), (7.0, 4.6, 4.2)]:
        want = bs.sequential_xwoba_phases(
            0.320, 0.320, 0.310, 0.300, 0.315, ipv,
            sp_bf_per_ip=rsp, bp_bf_per_ip=rbp)["sp_share"]
        got = ip.q_from_ip(pd.Series([ipv]), pd.Series([rsp]), pd.Series([rbp]))[0]
        assert got == pytest.approx(want, abs=1e-12)


def test_a_missing_bf_ip_pair_falls_back_to_the_innings_share():
    """Same degradation the build makes -- continuous, not a threshold."""
    got = ip.q_from_ip(pd.Series([4.5]), pd.Series([np.nan]), pd.Series([4.3]))[0]
    assert got == pytest.approx(4.5 / 9.0)


def test_the_calibration_is_not_applied_twice_to_a_v12_row(monkeypatch):
    """THE REGRESSION THIS FILE EXISTS FOR.

    A v12 row's `sp_share_*` already contains the calibration. Rebuilding the
    candidate off that shipped weight would apply the correction a second time,
    so the probe would show a difference that the shipped model does not have.
    Rebuilt off the RAW estimate, the candidate must reproduce the shipped q
    exactly on such a row.
    """
    fit = (1.4337, 0.7345, 586, 0.9214)
    monkeypatch.setitem(bs._sp_ip_cal_state, "fit", fit)
    raw = [3.2, 5.2, 6.8]
    published = [bs.calibrate_sp_ip(v, fit) for v in raw]
    shipped_q = list(ip.q_from_ip(pd.Series(published),
                                 pd.Series([4.3] * 3), pd.Series([4.3] * 3)))
    f = _frame(published, raw, shipped_q)
    qh, qa, label = ip.calibrated_q(f)
    assert list(qh) == pytest.approx(shipped_q, abs=1e-12)
    assert list(qa) == pytest.approx(shipped_q, abs=1e-12)
    assert "calibrated IP" in label


def test_a_pre_v12_row_still_shows_the_counterfactual(monkeypatch):
    """Rows written before the raw column existed hold a RAW value under the
    published name, so the candidate must differ from the shipped weight --
    that is the whole comparison on historical rows."""
    fit = (1.4337, 0.7345, 586, 0.9214)
    monkeypatch.setitem(bs._sp_ip_cal_state, "fit", fit)
    raw = [3.2, 6.8]
    shipped_q = list(ip.q_from_ip(pd.Series(raw), pd.Series([4.3] * 2),
                                 pd.Series([4.3] * 2)))
    f = _frame(raw, [np.nan, np.nan], shipped_q)   # no raw column populated
    qh, _qa, _label = ip.calibrated_q(f)
    assert list(qh) != pytest.approx(shipped_q, abs=1e-9)
    # Short start pulled up, deep start pulled down -- q moves the same way.
    assert qh.iloc[0] > shipped_q[0]
    assert qh.iloc[1] < shipped_q[1]


def test_no_fit_leaves_the_candidate_equal_to_the_shipped_weight(monkeypatch):
    monkeypatch.setitem(bs._sp_ip_cal_state, "fit", None)
    shipped_q = [0.4, 0.6]
    f = _frame([4.0, 6.0], [4.0, 6.0], shipped_q)
    qh, qa, label = ip.calibrated_q(f)
    assert list(qh) == shipped_q and list(qa) == shipped_q
    assert "no fit" in label


def test_no_calibration_slope_is_frozen_into_this_file():
    """It was `qs = 0.756`, measured on 2026-08-04 and stale by v12. The probe
    must read the live fit, or it benchmarks the model against an old copy of
    itself.

    Checked over the AST rather than the text, deliberately: the docstring in
    `calibrated_q` quotes the old literal on purpose, and a substring search
    cannot tell that apart from a live one. What must not exist is a numeric
    CONSTANT in the executable code -- prose about a retired constant is the
    record of why it went.
    """
    import ast
    tree = ast.parse(open(ip.__file__, encoding="utf-8").read())
    frozen = [n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)
              and 0.5 < n.value < 1.0]
    assert not frozen, f"a slope-shaped literal is back in the code: {frozen}"
    assert "build_site.sp_ip_calibration()" in open(
        ip.__file__, encoding="utf-8").read()
