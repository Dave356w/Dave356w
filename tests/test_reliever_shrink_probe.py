"""Does the probe's fit recover a K that was planted in the data?

The probe cannot be run here -- StatsAPI is not reachable from every dev
environment, which is why it lives behind a workflow. What can be checked
without a network is the part that would be wrong silently: an estimator that
returns a confident number for the wrong K prints a report that looks exactly
like a correct one.

So the two estimators are run against synthetic pools with a known
K = sigma^2 / tau^2, over several seeds, and the arithmetic they sit on
(wOBA reconstruction, innings parsing, role classification) is checked against
hand-computed values.
"""
import math

import numpy as np
import pytest

import reliever_shrink_probe as P


def synth(k_true, n_pitchers=900, sigma2=0.30, seed=0, bf_lo=8, bf_hi=320):
    """Pool with a planted K. Returns the probe's (n1, w1, n2, w2, bf/ip) array.

    tau^2 is derived from the target K rather than set independently, so the
    planted quantity is exactly the one being recovered.
    """
    rng = np.random.default_rng(seed)
    tau2 = sigma2 / k_true
    theta = rng.normal(0.315, math.sqrt(tau2), n_pitchers)
    n1 = rng.integers(bf_lo, bf_hi, n_pitchers).astype(float)
    n2 = rng.integers(bf_lo, bf_hi, n_pitchers).astype(float)
    w1 = theta + rng.normal(0, np.sqrt(sigma2 / n1))
    w2 = theta + rng.normal(0, np.sqrt(sigma2 / n2))
    return np.column_stack([n1, w1, n2, w2, np.full(n_pitchers, 4.3)])


@pytest.mark.parametrize("k_true", [40.0, 150.0, 400.0])
def test_fit_k_recovers_planted_constant(k_true):
    """Median fitted K across seeds lands within 25% of the truth.

    A point estimate off one draw is not the claim -- weighted MSE is flat near
    its minimum, so single-seed error is large by construction. The claim is
    that the estimator is centred, which is what the median over seeds tests
    and what makes the probe's bootstrap CI meaningful.
    """
    fits = []
    for seed in range(12):
        r = synth(k_true, seed=seed)
        mu = P._pool_mu(r)
        fits.append(P.fit_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3], mu))
    med = float(np.median(fits))
    assert abs(med - k_true) / k_true < 0.25, f"planted {k_true}, recovered {med:.1f}"


@pytest.mark.parametrize("k_true", [40.0, 150.0, 400.0])
def test_moments_k_recovers_planted_constant(k_true):
    """The variance-components arm agrees, sharing no code with the fit."""
    fits = []
    for seed in range(12):
        r = synth(k_true, seed=seed)
        k, tau2, sigma2 = P.moments_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3])
        if math.isfinite(k):
            fits.append(k)
    med = float(np.median(fits))
    assert abs(med - k_true) / k_true < 0.25, f"planted {k_true}, recovered {med:.1f}"


def test_the_two_estimators_agree_on_the_same_pool():
    """They are only a cross-check if they can disagree -- verify they don't."""
    r = synth(200.0, seed=7)
    k_fit = P.fit_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3], P._pool_mu(r))
    k_mom, _, _ = P.moments_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3])
    assert abs(math.log(k_fit) - math.log(k_mom)) < math.log(1.6)


def test_k_is_invariant_to_the_woba_scale_constant():
    """The docstring claims scaling every weight leaves K unchanged. Check it."""
    r = synth(150.0, seed=3)
    k1 = P.fit_k(r[:, 0], r[:, 1], r[:, 2], r[:, 3], P._pool_mu(r))
    s = r.copy()
    s[:, 1] *= 1.25
    s[:, 3] *= 1.25
    k2 = P.fit_k(s[:, 0], s[:, 1], s[:, 2], s[:, 3], P._pool_mu(s))
    assert abs(k1 - k2) / k1 < 1e-6


def test_bootstrap_interval_covers_the_planted_constant():
    lo, hi = P.bootstrap_k(synth(150.0, seed=11), None, draws=300)
    assert lo < 150.0 < hi, f"CI [{lo:.1f}, {hi:.1f}] misses 150"


def test_woba_matches_hand_computation():
    rec = {"AB": 100.0, "H": 25.0, "2B": 5.0, "3B": 1.0, "HR": 4.0,
           "BB": 12.0, "IBB": 2.0, "HBP": 1.0, "SF": 2.0, "BF": 0.0}
    rate, den = P.woba(rec)
    assert den == 100 + 12 - 2 + 2 + 1
    singles = 25 - 5 - 1 - 4
    expect = (0.690 * 10 + 0.720 * 1 + 0.880 * singles
              + 1.250 * 5 + 1.590 * 1 + 2.030 * 4) / den
    assert rate == pytest.approx(expect)
    # An appearance with no completed PA leaves the pool rather than counting
    # as a zero -- that distinction is what keeps sigma^2 honest.
    empty, den0 = P.woba({k: 0.0 for k in rec})
    assert den0 == 0 and math.isnan(empty)


def test_innings_parse_treats_the_third_digit_as_outs():
    assert P._outs("62.1") == 187
    assert P._outs("62.2") == 188
    assert P._outs("62.0") == 186
    assert P._outs(None) == 0


def test_role_is_classified_on_the_first_period_only():
    """A swingman who starts only after the split must still count as relief.

    Classifying on the full season would let the predicted period choose the
    pool, which is selection on the answer.
    """
    rp = {"AB": 60.0, "H": 15.0, "2B": 3.0, "3B": 0.0, "HR": 2.0, "BB": 6.0,
          "IBB": 0.0, "HBP": 1.0, "SF": 1.0, "BF": 70.0, "G": 30.0, "GS": 0.0,
          "outs": 180.0}
    sp = dict(rp, G=10.0, GS=10.0)
    periods = {(1, 4): rp, (1, 8): sp}     # relief early, rotation late
    rows = P.build_pairs(periods, 6, "RP", 0.20)
    assert len(rows) == 1, "first-period relief role was overridden by period 2"
    assert P.build_pairs({(1, 4): sp, (1, 8): rp}, 6, "RP", 0.20).shape[0] == 0


def test_pairs_require_both_periods():
    rec = {"AB": 40.0, "H": 10.0, "2B": 2.0, "3B": 0.0, "HR": 1.0, "BB": 4.0,
           "IBB": 0.0, "HBP": 0.0, "SF": 0.0, "BF": 44.0, "G": 20.0, "GS": 0.0,
           "outs": 120.0}
    assert P.build_pairs({(1, 4): rec}, 6, "RP", 0.20).shape[0] == 0
    assert P.build_pairs({(1, 4): rec, (1, 7): rec}, 6, "RP", 0.20).shape[0] == 1


def test_flat_loss_surface_returns_no_fit():
    """A pool with no talent spread must not yield a confident K.

    This is the failure the probe's first dry run actually produced: identical
    rates for every pitcher made the loss ~1e-34 at every K, and the argmin of
    that float noise printed as K = 3016 with a -10% dMSE. Both read as
    measurements.
    """
    n = np.full(200, 50.0)
    w = np.full(200, 0.315)
    assert math.isnan(P.fit_k(n, w, n, w, 0.315))


def test_a_fit_never_loses_to_the_shipped_constant():
    """dMSE cannot be negative: the fit minimises the metric dMSE is computed on.

    A negative value in that column means the search returned a non-minimiser,
    which is the one way this probe could recommend a worse K than the one it
    is arguing against.
    """
    for seed in range(6):
        r = synth(150.0, seed=seed)
        s = P.summarise(r, "t", P._pool_mu(r), shipped=100.0)
        assert s["dmse"] >= -1e-9, f"fit lost to shipped K by {-s['dmse']:.3f}%"


def test_bound_pinned_minima_are_flagged_not_printed_as_estimates():
    assert P._fmt_k(P.K_HI).strip().startswith(">")
    assert P._fmt_k(P.K_LO).strip().startswith("<")
    assert P._fmt_k(float("nan")).strip() == "--"


def test_empty_pull_raises_rather_than_reporting():
    """An empty fetch must not reach the report as a zero-row arm."""
    with pytest.raises(SystemExit):
        P._bail("no usable splits", 2026, "pitching", "byMonth", {"stats": []})
