"""Guards on the v12 expected-IP calibration.

The estimator is over-dispersed and `q` -- the starter's share of plate
appearances -- is a function of it. What these assert is the part that would be
wrong *silently*: a correction that compounds against its own output, a fit
that reads a slate it is about to predict, and a hard sample-size gate that
makes every workload estimate jump on the day it is crossed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as bs


@pytest.fixture(autouse=True)
def _clear_cache():
    """The fit is memoised per build; these tests each supply their own."""
    bs._sp_ip_cal_state.pop("fit", None)
    yield
    bs._sp_ip_cal_state.pop("fit", None)


def _led(pred, act, raw=None, cols=True):
    n = len(pred)
    d = {
        "expected_sp_ip_away": pred,
        "act_sp_ip_away": act,
        "expected_sp_ip_home": [np.nan] * n,
        "act_sp_ip_home": [np.nan] * n,
    }
    if raw is not None:
        d["expected_sp_ip_raw_away"] = raw
        d["expected_sp_ip_raw_home"] = [np.nan] * n
    elif cols:
        pass
    return pd.DataFrame(d)


# --------------------------------------------------------------------------
# the identity limit -- no gate, no cliff
# --------------------------------------------------------------------------
def test_no_ledger_leaves_the_estimate_untouched():
    bs._sp_ip_cal_state["fit"] = None
    assert bs.calibrate_sp_ip(5.2) == 5.2
    assert bs.calibrate_sp_ip(None) is None


def test_one_pair_is_not_enough_to_fit():
    """A slope through one point is undefined; a fabricated one would read as
    a measurement."""
    assert bs.sp_ip_calibration(_led([5.0], [6.0])) is None


def test_the_correction_grows_continuously_with_sample_size():
    """No `if n >= N` gate, so no day on which every workload estimate jumps.

    The threshold-cliff lesson: the fix for "use the fit once there is enough
    of it" is a weight, not a better minimum.
    """
    moves = []
    for n in (2, 10, 50, 200, 1000):
        bs._sp_ip_cal_state.pop("fit", None)
        rng = np.random.default_rng(0)
        pred = rng.uniform(3.5, 7.0, n)
        act = 1.4 + 0.735 * pred          # a clean over-dispersed relationship
        fit = bs.sp_ip_calibration(_led(list(pred), list(act)))
        moves.append(abs(bs.calibrate_sp_ip(7.0, fit) - 7.0))
    # Monotone in n, and it starts near zero rather than jumping.
    assert moves == sorted(moves)
    assert moves[0] < moves[-1] / 5
    # w = n/(n+K) never reaches 1, so the raw estimate is never fully discarded.
    assert moves[-1] < abs((1.4 + 0.735 * 7.0) - 7.0)


def test_a_calibrated_estimator_is_left_alone():
    """Slope 1, intercept 0 means already calibrated -- the map must be inert."""
    pred = list(np.linspace(3.5, 7.0, 60))
    fit = bs.sp_ip_calibration(_led(pred, list(pred)))
    assert fit is not None
    for v in (3.5, 5.2, 7.0):
        assert bs.calibrate_sp_ip(v, fit) == pytest.approx(v, abs=1e-9)


# --------------------------------------------------------------------------
# the fit must not consume its own output
# --------------------------------------------------------------------------
def test_the_fit_reads_the_raw_column_when_it_exists():
    """Regressing against the PUBLISHED value would compound the correction
    every build -- the estimator would be pulled toward the mean without limit.

    Raw and published are deliberately given different slopes here, so a fit
    that read the wrong column cannot coincidentally agree.
    """
    raw = list(np.linspace(3.0, 7.0, 40))
    act = [1.4 + 0.735 * r for r in raw]
    published = [bs.calibrate_sp_ip(r, (1.4, 0.735, 1e6, 1.0)) for r in raw]
    fit = bs.sp_ip_calibration(_led(published, act, raw=raw))
    assert fit is not None
    intercept, slope, _n, _w = fit
    assert slope == pytest.approx(0.735, abs=1e-6)
    assert intercept == pytest.approx(1.4, abs=1e-6)


def test_pre_v12_rows_join_the_fit_through_the_published_column():
    """Rows written before the raw column existed hold a raw value under the
    published name, so they must not be dropped -- that is the whole sample on
    the first build after this ships."""
    pred = list(np.linspace(3.0, 7.0, 30))
    act = [1.4 + 0.735 * p for p in pred]
    fit = bs.sp_ip_calibration(_led(pred, act))          # no raw column at all
    assert fit is not None
    assert fit[2] == 30
    assert fit[1] == pytest.approx(0.735, abs=1e-6)


def test_rows_are_mixed_raw_and_legacy_without_a_branch():
    """A ledger spanning the bump holds both shapes at once."""
    raw = list(np.linspace(3.0, 7.0, 20))
    act = [1.4 + 0.735 * r for r in raw]
    # First ten rows legacy (raw column NaN), last ten carry it.
    mixed_raw = [np.nan] * 10 + raw[10:]
    fit = bs.sp_ip_calibration(_led(raw, act, raw=mixed_raw))
    assert fit[2] == 20
    assert fit[1] == pytest.approx(0.735, abs=1e-6)


# --------------------------------------------------------------------------
# no lookahead
# --------------------------------------------------------------------------
def test_only_completed_games_can_enter_the_fit():
    """An actual IP exists only after the box score is backfilled, so a pending
    row contributes nothing. Tonight's slate is never calibrated against
    itself."""
    pred = list(np.linspace(3.0, 7.0, 30)) + [5.0, 6.0]
    act = [1.4 + 0.735 * p for p in np.linspace(3.0, 7.0, 30)] + [np.nan, np.nan]
    fit = bs.sp_ip_calibration(_led(pred, act))
    assert fit[2] == 30


# --------------------------------------------------------------------------
# the shape of the correction
# --------------------------------------------------------------------------
def test_over_dispersion_is_pulled_toward_the_mean_from_both_ends():
    """The measured defect is spread, not level: short predictions run long and
    deep ones run short. The correction must move them toward each other."""
    pred = list(np.linspace(3.0, 7.0, 200))
    act = [1.4 + 0.735 * p for p in pred]
    fit = bs.sp_ip_calibration(_led(pred, act))
    lo_raw, hi_raw = 3.5, 7.0
    lo, hi = bs.calibrate_sp_ip(lo_raw, fit), bs.calibrate_sp_ip(hi_raw, fit)
    assert lo > lo_raw                      # predicted short -> longer
    assert hi < hi_raw                      # predicted deep  -> shorter
    assert (hi - lo) < (hi_raw - lo_raw)     # spread compressed


def test_the_benchmarked_constant_is_in_the_flat_region_it_was_chosen_from():
    """K was picked by walk-forward benchmark over 10-100, where the curve is
    flat. Pinning the value stops a later edit drifting outside the range the
    measurement actually covered -- it is not a claim that 50 beats 25."""
    assert 10 <= bs.SP_IP_CALIBRATION_K <= 100
