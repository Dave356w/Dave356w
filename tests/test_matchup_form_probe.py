"""Does the combiner fit recover coefficients that were planted in the data?

The probe reports b1 and b2 as evidence about whether the shipped equal-weight
restriction holds. An estimator that returns confident coefficients for the
wrong weights produces a report indistinguishable from a correct one, so the
weights are planted and recovered before any of it is believed.
"""
import math

import numpy as np
import pandas as pd
import pytest

import matchup_form_probe as M


def synth(b1, b2, n_games=400, noise=0.11, seed=0, L=0.315):
    """Phase observations with planted component weights.

    Noise is set to the realised per-phase scale (~0.11) so recovery is
    demonstrated at the signal-to-noise the real data actually has, not at a
    flattering one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for side in ("away", "home"):
            B = rng.normal(L, 0.012)
            P = rng.normal(L, 0.020)
            w = rng.integers(12, 34)
            act = b1 * B + b2 * P + (1 - b1 - b2) * L + rng.normal(0, noise)
            rows.append({"game_pk": g, "model_tag": "t", "phase": "SP",
                         "side": side, "B": B, "P": P, "L": L,
                         "mx": B * P / L, "act": act, "w": float(w)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("b1,b2", [(1.0, 1.0), (0.5, 1.5), (1.5, 0.5)])
def test_free_fit_recovers_planted_weights(b1, b2):
    """Median recovery across seeds lands near the truth.

    Median over seeds, not a single fit: at sd 0.11 against component spreads
    of ~0.02 the per-draw error is large by construction, and the claim is that
    the estimator is centred -- which is what makes its standard errors usable.
    """
    got1, got2 = [], []
    for seed in range(9):
        s = synth(b1, b2, seed=seed)
        beta, se = M.wls(s.act, np.column_stack([s.B, s.P]), s.w)
        got1.append(beta[1])
        got2.append(beta[2])
    assert abs(float(np.median(got1)) - b1) < 0.45, f"b1 {np.median(got1):.2f} vs {b1}"
    assert abs(float(np.median(got2)) - b2) < 0.45, f"b2 {np.median(got2):.2f} vs {b2}"


def test_standard_errors_are_calibrated():
    """The ~68% of fits inside one se is what licenses reading b-1 as a z."""
    inside = 0
    trials = 40
    for seed in range(trials):
        s = synth(1.0, 1.0, seed=100 + seed)
        beta, se = M.wls(s.act, np.column_stack([s.B, s.P]), s.w)
        if se is not None and abs(beta[2] - 1.0) <= se[2]:
            inside += 1
    assert 0.5 <= inside / trials <= 0.9, f"{inside}/{trials} within 1 se"


def test_wls_weights_actually_bind():
    """An unweighted fit is a different estimator; verify w is not ignored."""
    s = synth(1.0, 1.0, seed=3)
    heavy = s.copy()
    heavy["w"] = np.where(heavy.B > heavy.B.median(), 1000.0, 1.0)
    a, _ = M.wls(s.act, np.column_stack([s.B, s.P]), s.w)
    b, _ = M.wls(heavy.act, np.column_stack([heavy.B, heavy.P]), heavy.w)
    assert not np.allclose(a, b)


def test_reconstruction_guard_aborts_on_misidentified_terms():
    """A wrong B or P still yields a fit; only this guard catches it."""
    s = synth(1.0, 1.0, n_games=20, seed=1)
    M.check_reconstruction(s)                      # consistent -> passes
    s.loc[0, "mx"] = 0.9                           # stored mx disagrees
    with pytest.raises(SystemExit):
        M.check_reconstruction(s)


def test_reconstruction_tolerates_ledger_rounding():
    """Stored values are rounded; the guard must not fire on that alone."""
    s = synth(1.0, 1.0, n_games=50, seed=2)
    s["mx"] = s.mx.round(4)
    M.check_reconstruction(s)


def test_empty_ledger_reports_instead_of_crashing():
    assert M.observations(pd.DataFrame()).empty
    M.check_reconstruction(pd.DataFrame(columns=["B", "P", "L", "mx"]))


def test_separation_arithmetic_is_the_interaction_term():
    """mult - add must equal (B-L)(P-L)/L exactly, since that identity is the
    entire argument for not comparing the two forms."""
    s = synth(1.0, 1.0, n_games=30, seed=4)
    mult = s.B * s.P / s.L
    add = s.B + s.P - s.L
    ident = (s.B - s.L) * (s.P - s.L) / s.L
    assert np.allclose(mult - add, ident)
