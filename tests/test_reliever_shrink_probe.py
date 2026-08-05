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


# --- what CI measured, pinned so it cannot regress -------------------------
# The first CI run of this probe died on `stats=byMonth&sportIds=1`, which
# returned zero stat blocks for every season. Two defects, not one: the guessed
# call shape, and a probe that let a dead arm kill an arm using a proven shape.

def _payload(pid, month, bf, code="1"):
    return {"stats": [{"splits": [{
        "player": {"id": pid}, "position": {"code": code}, "month": month,
        "stat": {"atBats": bf * 0.9, "hits": bf * 0.2, "doubles": 0,
                 "triples": 0, "homeRuns": bf * 0.03, "baseOnBalls": bf * 0.08,
                 "intentionalWalks": 0, "hitByPitch": 0, "sacFlies": 0,
                 "battersFaced": bf, "gamesPitched": 10, "gamesStarted": 0,
                 "inningsPitched": "20.0"},
    }]}]}


def test_a_dead_shape_falls_through_to_a_working_one(monkeypatch):
    """The shape that failed in CI must not be able to take the probe down."""
    tried = []

    def fake_get(url, params, tries=4):
        tried.append(params.get("stats"))
        if "sportIds" in params:          # the shape CI measured as empty
            return {"stats": []}
        return {"stats": [{"splits": [
            _payload(1, 4, 40)["stats"][0]["splits"][0],
            _payload(1, 7, 55)["stats"][0]["splits"][0],
        ]}]}

    monkeypatch.setattr(P, "_get", fake_get)
    P._working_shape.clear()
    P._shape_log.clear()
    out = P.fetch_month_periods(2026, "pitching", verbose=False)
    assert len(out) == 2, "fell through to no working shape"
    assert {p for _, p in out} == {4, 7}


def test_no_working_shape_returns_empty_instead_of_exiting(monkeypatch):
    """A within-season split that cannot be built must not kill the run.

    The season-pair arm uses `stats=season`, a shape this repo runs on every
    build. Hard-failing here would discard a working arm because a different
    one is unavailable -- which is exactly what the first CI run did.
    """
    monkeypatch.setattr(P, "_get", lambda url, params, tries=4: {"stats": []})
    monkeypatch.setattr(P, "team_ids", lambda season: [108])
    P._working_shape.clear()
    P._shape_log.clear()
    assert P.fetch_month_periods(2026, "pitching", verbose=False) == {}
    assert P._shape_log, "a failed shape must be recorded, not swallowed"


def test_season_totals_still_hard_fail_on_an_empty_pull(monkeypatch):
    """The proven shape keeps its fail-loudly contract."""
    monkeypatch.setattr(P, "_get", lambda url, params, tries=4: {"stats": []})
    with pytest.raises(SystemExit):
        P.fetch_season_totals(2026, "pitching", verbose=False)


def test_a_traded_player_accumulates_across_calls():
    """Per-team shapes write the same (pid, month) once per club.

    Assigning instead of accumulating would keep only the last club's line and
    silently drop the rest of the pitcher's season.
    """
    into, dropped = {}, __import__("collections").defaultdict(int)
    P._collect(_payload(7, 5, 30), "pitching", None, into, dropped)
    P._collect(_payload(7, 5, 20), "pitching", None, into, dropped)
    assert into[(7, 5)]["BF"] == 50


def test_centre_reports_weighted_and_unweighted_separately():
    """The two centres answer different questions and must not collapse.

    EB wants the pool's player-level (unweighted) centre; the build's target is
    PA-weighted. Reporting one as the other would hide the first of the two
    gaps _pool_moments exists to separate.
    """
    c = P._centre([(0.300, 100.0), (0.400, 300.0)])
    assert c["wtd"] == pytest.approx(0.375)
    assert c["unw"] == pytest.approx(0.350)
    assert c["n"] == 2 and c["med"] == pytest.approx(200.0)
    assert P._centre([]) is None
    assert P._centre([(0.3, 0.0)]) is None      # zero weight is not a member


def test_role_centres_split_by_role_and_throwing_hand(monkeypatch):
    def rec(bf, starts, apps):
        return {"AB": bf * 0.9, "H": bf * 0.2, "2B": 0.0, "3B": 0.0,
                "HR": bf * 0.03, "BB": bf * 0.08, "IBB": 0.0, "HBP": 0.0,
                "SF": 0.0, "BF": float(bf), "G": float(apps),
                "GS": float(starts), "outs": float(bf)}

    pit = {(1, 0): rec(600, 25, 25),    # SP, throws L
           (2, 0): rec(650, 26, 26),    # SP, throws R
           (3, 0): rec(70, 0, 60),      # RP, throws L
           (4, 0): rec(80, 0, 65)}      # RP, throws R
    bat = {(5, 0): rec(500, 0, 140)}
    hands = {1: {"throws": "L", "bats": "L"}, 2: {"throws": "R", "bats": "R"},
             3: {"throws": "L", "bats": "L"}, 4: {"throws": "R", "bats": "R"},
             5: {"throws": "R", "bats": "L"}}
    monkeypatch.setattr(P, "fetch_season_totals",
                        lambda s, g, verbose=True: pit if g == "pitching" else bat)
    monkeypatch.setattr(P, "fetch_hands", lambda pids, verbose=True: hands)

    cent = P.role_centres(2026, 0.20, verbose=False)
    assert cent["SP"]["n"] == 2 and cent["RP"]["n"] == 2
    assert cent["SP-L"]["n"] == 1 and cent["RP-R"]["n"] == 1
    assert cent["BAT"]["n"] == 1 and cent["BAT-L"]["n"] == 1
    # A starter must never land in the relief pool via the hand split.
    assert "SP-L" in cent and cent["RP-L"]["n"] == 1


def test_role_centres_survive_an_unknown_hand(monkeypatch):
    """A missing hand must bucket as '?', not crash or silently join L or R."""
    r = {"AB": 90.0, "H": 20.0, "2B": 0.0, "3B": 0.0, "HR": 3.0, "BB": 8.0,
         "IBB": 0.0, "HBP": 0.0, "SF": 0.0, "BF": 100.0, "G": 50.0, "GS": 0.0,
         "outs": 100.0}
    monkeypatch.setattr(P, "fetch_season_totals",
                        lambda s, g, verbose=True: {(9, 0): r} if g == "pitching" else {})
    monkeypatch.setattr(P, "fetch_hands", lambda pids, verbose=True: {})
    cent = P.role_centres(2026, 0.20, verbose=False)
    assert cent["RP-?"]["n"] == 1
    assert "RP-L" not in cent and "RP-R" not in cent


def test_position_players_are_dropped_from_the_pitching_pool():
    into, dropped = {}, __import__("collections").defaultdict(int)
    P._collect(_payload(9, 5, 6, code="3"), "pitching", None, into, dropped)
    assert into == {} and dropped["non-pitcher"] == 1
