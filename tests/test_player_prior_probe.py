"""Does the probe recover a planted C, and refuse to reward useless history?

The probe cannot be run here -- StatsAPI is not reachable from every dev
environment, which is why it lives behind a workflow. What can be checked
without a network is the part that would be wrong silently.

Two failure modes are checked, and the second is the important one:

  * A fitter that returns a confident C for the wrong one prints a report
    indistinguishable from a correct one. So C is planted and recovered.

  * An arm that "improves" on rows where the history carries NO talent signal
    is measuring its own flexibility. Personal priors are exactly the kind of
    change that looks better in sample, so the no-signal regime is planted too,
    and the requirement there is that the arm cannot win.
"""
import math

import numpy as np
import pytest

import player_prior_probe as P


def synth(n_players=1200, sigma2=0.30, tau2=None, k_true=400.0, seed=0,
          persist=1.0, hist_bf=(60, 600), cur_bf=(40, 400), n_seasons=3,
          mu=0.315):
    """Rows with a planted history constant.

    Talent is drawn once per player and each prior season observes it with
    sampling noise sigma^2/n. With flat recency weights and `persist = 1`, the
    posterior mean of talent given H batters faced of history is exactly

        (H*theta_hist + (sigma^2/tau^2)*mu) / (H + sigma^2/tau^2)

    so the optimal C IS sigma^2/tau^2 -- the same quantity K is. That identity
    is what makes C recoverable rather than merely fittable, and it is why the
    recovery test uses flat weights: the recency ladder discounts H and moves
    the optimum away from a closed form.

    `persist < 1` re-draws part of talent each season, which is the regime
    where history should buy nothing.
    """
    rng = np.random.default_rng(seed)
    tau2 = sigma2 / k_true if tau2 is None else tau2
    base = rng.normal(mu, math.sqrt(tau2), n_players)
    rows = []
    for i in range(n_players):
        hist = {}
        for lag in range(1, n_seasons + 1):
            n = float(rng.integers(*hist_bf))
            # persist=1: the same talent generated every season. persist=0:
            # each season is an independent draw, so history predicts nothing.
            theta_s = (persist * base[i]
                       + (1 - persist) * rng.normal(mu, math.sqrt(tau2)))
            hist[lag] = (theta_s + rng.normal(0, math.sqrt(sigma2 / n)), n)
        theta_now = base[i]
        n1 = float(rng.integers(*cur_bf))
        n2 = float(rng.integers(*cur_bf))
        rows.append({
            "pid": i, "season": 2025,
            "n1": n1, "w1": theta_now + rng.normal(0, math.sqrt(sigma2 / n1)),
            "n2": n2, "w2": theta_now + rng.normal(0, math.sqrt(sigma2 / n2)),
            "mu": mu,
            "_hist_raw": hist,
        })
    return rows


def attach(rows, weights, key="recency_role"):
    """Aggregate the planted per-season history under one ladder."""
    for r in rows:
        num = den = 0.0
        for lag, (rate, n) in r["_hist_raw"].items():
            v = weights[lag - 1] if lag - 1 < len(weights) else 0.0
            if v <= 0:
                continue
            num += v * n * rate
            den += v * n
        r.setdefault("hist", {})[key] = ((num / den, den) if den > 0
                                         else (float("nan"), 0.0))
    return rows


FLAT = (1.0, 1.0, 1.0)


def attach_all(rows, weights=None):
    """Fill every history spec `score_arms` reads, from the same planted data."""
    weights = weights or P.RECENCY_WEIGHTS
    attach(rows, weights, "recency_role")
    attach(rows, weights, "recency_blind")
    attach(rows, FLAT, "career_role")
    attach(rows, (1.0,), "prev_role")
    # The synthetic league does not drift, so the centred arm sees the same
    # history as the uncentred one -- which is exactly the null it should show.
    attach(rows, weights, "recency_centred")
    # Nothing frozen: the Savant arm has no history at all. Kept in the fixture
    # rather than omitted, because "no priors committed yet" is the state the
    # probe runs in until `priors_snapshot.py` has been run, and it must be a
    # tie with arm 1 rather than a crash.
    attach(rows, (0.0, 0.0, 0.0), "savant_centred")
    return rows


# --------------------------------------------------------------------------
# the fitter
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k_true", [150.0, 400.0, 1000.0])
def test_fit_c_recovers_the_planted_constant(k_true):
    """Median fitted C across seeds lands within 30% of sigma^2/tau^2.

    A point estimate off one draw is not the claim -- weighted MSE is flat near
    its minimum, which is the same reason `reliever_shrink_probe` tests the
    median over seeds rather than any single fit. The claim is that the
    estimator is centred, which is what makes the bootstrap CI mean anything.
    """
    fits = []
    for seed in range(10):
        rows = attach(synth(k_true=k_true, seed=seed), FLAT)
        n1, w1, n2, w2, mu, th, h = P.to_arrays(rows, "recency_role")
        fits.append(P.fit_c(n1, w1, n2, w2, mu, th, h, k=P.SHIPPED_K))
    med = float(np.median(fits))
    assert abs(med - k_true) / k_true < 0.30, f"planted {k_true}, got {med:.0f}"


def test_fit_c_is_flagged_not_invented_when_there_is_no_history():
    """No usable history anywhere must return nan, never a four-digit C."""
    rows = attach(synth(n_players=200, seed=1), (0.0, 0.0, 0.0))
    n1, w1, n2, w2, mu, th, h = P.to_arrays(rows, "recency_role")
    assert not math.isfinite(P.fit_c(n1, w1, n2, w2, mu, th, h))


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def _dmse(rows, spec="recency_role", c=None, fit=False, k=P.SHIPPED_K):
    n1, w1, n2, w2, mu, th, h = P.to_arrays(rows, spec)
    base = P.arm_loss(None, n1, w1, n2, w2, mu, mu, np.zeros_like(mu), k)
    if fit:
        c = P.fit_c(n1, w1, n2, w2, mu, th, h, k)
    mse = P.arm_loss(c, n1, w1, n2, w2, mu, th, h, k)
    return (base - mse) / base * 100.0


def test_a_persistent_talent_signal_is_found():
    """When history really is talent, the regressed arm must beat the centre."""
    rows = attach(synth(seed=5, persist=1.0), P.RECENCY_WEIGHTS)
    assert _dmse(rows, fit=True) > 1.0


def test_history_with_no_signal_cannot_win():
    """Talent redrawn every season: the arm must not manufacture a gain.

    This is the guard the whole probe rests on. A personal prior has more
    freedom than a population centre, so the question is never whether it can
    fit -- it is whether the fitted C collapses to "ignore history" when there
    is nothing there. The tolerance is one tenth of a percent, which is the
    fitting noise floor at this sample size, not a licence to lose.
    """
    for seed in range(4):
        rows = attach(synth(seed=seed, persist=0.0), P.RECENCY_WEIGHTS)
        assert _dmse(rows, fit=True) > -0.10


def test_raw_career_loses_to_the_regressed_prior_on_thin_history():
    """The 70-PA case: an unregressed personal prior is worse than the centre.

    This is the arm the proposal expects to lose and the reason level (2) is in
    the report at all. If it ever stops losing here, the regression step is not
    earning its constant and the probe should say so.
    """
    rows = attach(synth(seed=11, hist_bf=(20, 90)), P.RECENCY_WEIGHTS)
    raw = _dmse(rows, c=None)          # C = 0: the career rate, unadjusted
    reg = _dmse(rows, fit=True)
    assert reg > raw, f"regressed {reg:.2f}% vs raw {raw:.2f}%"
    assert raw < 0.0, f"raw career should lose to the centre, got {raw:+.2f}%"


def test_more_history_beats_less_when_talent_persists():
    """The recency ladder over three seasons beats one season of history."""
    three = attach(synth(seed=3), P.RECENCY_WEIGHTS)
    one = attach(synth(seed=3), (1.0,))
    assert _dmse(three, fit=True) > _dmse(one, fit=True)


# --------------------------------------------------------------------------
# the expression's limits -- the reason there are no branches
# --------------------------------------------------------------------------
def test_prior_limits():
    """H=0 is the population centre; C=0 is the raw history. One expression."""
    assert P.personal_prior(0.400, 0.0, 0.315, 400.0) == pytest.approx(0.315)
    assert P.personal_prior(float("nan"), 500.0, 0.315, 400.0) == pytest.approx(0.315)
    assert P.personal_prior(0.400, 500.0, 0.315, 0.0) == pytest.approx(0.400)
    assert P.personal_prior(0.400, 400.0, 0.300, 400.0) == pytest.approx(0.350)


def test_vector_priors_match_the_scalar_form():
    """`priors` is the only thing the fit calls; it must be `personal_prior`."""
    mu = np.array([0.315, 0.315, 0.320])
    th = np.array([0.400, 0.280, 0.315])
    h = np.array([600.0, 0.0, 120.0])
    got = P.priors(mu, th, h, 400.0)
    want = [P.personal_prior(t, n, m, 400.0) for t, n, m in zip(th, h, mu)]
    assert got == pytest.approx(want)
    raw = P.priors(mu, th, h, None)
    assert raw == pytest.approx([0.400, 0.315, 0.315])


def test_no_history_rows_score_identically_under_every_arm():
    """Coverage is a bound, not a footnote -- verify the bound is real."""
    rows = attach(synth(n_players=300, seed=9), (0.0, 0.0, 0.0))
    assert _dmse(rows, c=None) == pytest.approx(0.0, abs=1e-9)
    assert _dmse(rows, c=400.0) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# the arithmetic underneath
# --------------------------------------------------------------------------
def test_weighted_history_weights_by_recency_and_sample():
    hist = {2024: (0.400, 100.0, "BAT"),
            2023: (0.300, 100.0, "BAT"),
            2022: (0.200, 100.0, "BAT")}
    rate, h = P.weighted_history(hist, 2025, (0.5, 0.3, 0.2))
    assert h == pytest.approx(100.0)
    assert rate == pytest.approx(0.5 * 0.400 + 0.3 * 0.300 + 0.2 * 0.200)


def test_weighted_history_separates_roles():
    hist = {2024: (0.330, 200.0, "RP"), 2023: (0.290, 600.0, "SP")}
    rate, h = P.weighted_history(hist, 2025, (0.5, 0.3), role="RP")
    assert h == pytest.approx(0.5 * 200.0)
    assert rate == pytest.approx(0.330)
    blind, hb = P.weighted_history(hist, 2025, (0.5, 0.3))
    assert hb == pytest.approx(0.5 * 200.0 + 0.3 * 600.0)
    assert blind != pytest.approx(0.330)


def test_weighted_history_reports_no_history_as_zero_weight():
    assert P.weighted_history({}, 2025, (0.5, 0.3, 0.2))[1] == 0.0
    # A season outside the ladder contributes nothing rather than everything.
    hist = {2019: (0.400, 500.0, "BAT")}
    assert P.weighted_history(hist, 2025, (0.5, 0.3, 0.2))[1] == 0.0


def test_the_shipped_k_is_read_from_build_site_not_copied():
    """One value, one home. The probe reports against what the build ships."""
    import build_site

    assert P.SHIPPED_K == pytest.approx(build_site.XWOBA_SHRINK_K)


def test_season_role_classifies_on_start_share():
    assert P.season_role({"G": 30.0, "GS": 30.0}, 0.20) == "SP"
    assert P.season_role({"G": 60.0, "GS": 2.0}, 0.20) == "RP"
    assert P.season_role({"G": 0.0, "GS": 0.0}, 0.20) is None


def test_shipped_mu_mirrors_build_site_not_a_tidier_version():
    """Relievers get their pool's UNWEIGHTED centre; batters and starters the
    league PA-weighted batter rate. Arm 1 is only the shipped baseline if this
    matches `relief_pool_prior` and `build_pitching_plans`."""
    centres = {"BAT": {"wtd": 0.318, "unw": 0.300},
               "RP": {"wtd": 0.295, "unw": 0.308},
               "SP": {"wtd": 0.312, "unw": 0.315}}
    assert P.shipped_mu("RP", centres) == pytest.approx(0.308)
    assert P.shipped_mu("BAT", centres) == pytest.approx(0.318)
    assert P.shipped_mu("SP", centres) == pytest.approx(0.318)
    assert P.shipped_mu("SP", None) is None


def test_early_split_takes_the_first_months_not_the_balanced_one():
    """The two splits answer different questions and must not coincide."""
    periods = {(1, 4): {"BF": 100.0}, (1, 5): {"BF": 100.0},
               (2, 6): {"BF": 600.0}, (2, 7): {"BF": 100.0}}
    assert P.early_split(periods, 1) == 4
    assert P.early_split(periods, 2) == 5
    assert P.balanced_split(periods) == 5
    # Not enough months to leave anything on the other side of the boundary.
    assert P.early_split({(1, 4): {"BF": 10.0}}, 1) is None
    assert P.early_split(periods, 4) is None


def test_balanced_split_divides_batters_faced_not_months():
    periods = {(1, 4): {"BF": 100.0}, (1, 5): {"BF": 100.0},
               (2, 6): {"BF": 600.0}, (2, 7): {"BF": 100.0}}
    # 4|rest = 100/800, 5|rest = 200/700, 6|rest = 800/100 -> 5 is the balance.
    assert P.balanced_split(periods) == 5
    assert P.balanced_split({(1, 4): {"BF": 10.0}}) is None


def test_build_rows_classifies_role_on_the_first_period_only():
    """A reliever who joins the rotation in July is a reliever for this row."""
    periods = {
        (7, 4): {"AB": 60.0, "H": 15.0, "2B": 3.0, "3B": 0.0, "HR": 2.0,
                 "BB": 6.0, "IBB": 0.0, "HBP": 1.0, "SF": 1.0, "BF": 70.0,
                 "G": 25.0, "GS": 0.0, "outs": 60.0},
        (7, 8): {"AB": 90.0, "H": 20.0, "2B": 4.0, "3B": 0.0, "HR": 3.0,
                 "BB": 8.0, "IBB": 0.0, "HBP": 0.0, "SF": 1.0, "BF": 100.0,
                 "G": 8.0, "GS": 8.0, "outs": 140.0},
    }
    hist = {7: {2024: (0.290, 300.0, "RP")}}
    rp = P.build_rows(2025, periods, hist, 5, "RP", 0.20, 0.315,
                      P.RECENCY_WEIGHTS, 3)
    sp = P.build_rows(2025, periods, hist, 5, "SP", 0.20, 0.315,
                      P.RECENCY_WEIGHTS, 3)
    assert len(rp) == 1 and not sp
    assert rp[0]["hist"]["recency_role"][1] == pytest.approx(0.5 * 300.0)


def test_score_arms_reports_the_league_arm_as_the_zero():
    rows = attach_all(synth(n_players=400, seed=2))
    scored = P.score_arms(rows)
    assert [s["label"][:2] for s in scored] == ["1 ", "2 ", "3 ", "3b", "3c",
                                                "3d", "3e"]
    assert scored[0]["dmse"] == pytest.approx(0.0)
    assert scored[0]["c"] is None
    # Planted persistent talent: the regressed arm beats the centre, and the
    # unregressed career arm is the one that gives it back.
    assert scored[2]["dmse"] > 0.0
    assert scored[2]["dmse"] > scored[1]["dmse"]
    # An arm with nothing frozen behind it must tie arm 1 exactly -- reporting
    # "no data" as a zero, never as a number.
    savant = next(s for s in scored if s["spec"] == "savant_centred")
    assert savant["dmse"] == pytest.approx(0.0, abs=1e-9)
    assert savant["cover"] == pytest.approx(0.0)


def _fake_line(rng, pa, gs_share, mu=0.315):
    """A StatsAPI-shaped counting line whose wOBA lands near `mu`."""
    hits = rng.binomial(int(pa), 0.24)
    hr = rng.binomial(int(hits), 0.14)
    bb = rng.binomial(int(pa), 0.085)
    g = max(1.0, pa / 4.0 if gs_share > 0.5 else pa / 4.0)
    return {"AB": float(pa - bb), "H": float(hits), "2B": float(hits // 5),
            "3B": 0.0, "HR": float(hr), "BB": float(bb), "IBB": 0.0,
            "HBP": 1.0, "SF": 1.0, "BF": float(pa), "G": g,
            "GS": g * gs_share, "outs": float(pa)}


def _stub_fetches(monkeypatch, n=260):
    """Wire the probe's two fetch calls to synthetic pulls, no network."""
    rng = np.random.default_rng(17)

    def season_totals(season, group, verbose=True):
        out = {}
        for pid in range(n):
            share = 0.0 if group == "hitting" else (0.9 if pid % 3 == 0 else 0.0)
            out[(pid, 0)] = _fake_line(rng, 480 if share else 220, share)
        return out

    def month_periods(season, group, verbose=True):
        out = {}
        for pid in range(n):
            share = 0.0 if group == "hitting" else (0.9 if pid % 3 == 0 else 0.0)
            for month in (4, 5, 6, 7, 8, 9):
                out[(pid, month)] = _fake_line(rng, 80 if share else 45, share)
        return out

    monkeypatch.setattr(P, "fetch_season_totals", season_totals)
    monkeypatch.setattr(P, "fetch_month_periods", month_periods)


def test_main_produces_a_report_without_a_network(monkeypatch, tmp_path, capsys):
    """The report path only ever runs on a runner. Exercise it here anyway.

    Every estimator above is tested on arrays; nothing tested the code that
    turns them into the artifact a decision gets read off. A probe that raises
    in its own formatter after a 20-minute pull is a wasted run, and this repo
    has that failure on record for a fetch shape.
    """
    _stub_fetches(monkeypatch)
    out = tmp_path / "report.txt"
    rc = P.main(["--targets", "2024,2025", "--history", "2",
                 "--draws", "40", "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    for want in ("Coverage", "1 league/role centre", "2 raw career",
                 "3 regressed recency", "By current-season sample",
                 "Cross-season transfer", "Verdict"):
        assert want in text, f"missing section: {want}"
    # Arm 1 is the baseline by construction; it must print as exactly zero.
    assert "+0.00%" in text or " 0.00%" in text


def test_cross_tab_separates_history_size_from_current_sample():
    """Planted: history helps ONLY where history is large, n1 carries nothing.

    The synthetic pool draws n1 and history size independently, and the talent
    signal is real for every player. So the gain must line up with H and be
    flat across n1 -- which is the pattern that would show the first report's
    n1 table was reading H through a proxy. If `cross_tab` cannot recover a
    planted separation it cannot adjudicate the real one.
    """
    # Two pools with the same talent distribution and the same n1 draw, differing
    # only in how much history each player has. Thinning an existing pool by
    # scaling H would be wrong: it leaves a full-precision rate attached to a
    # small sample, which is a better prior than any real thin history, and the
    # arm would correctly exploit it. The noise has to be regenerated at the
    # smaller n, so the history is drawn that way from the start.
    rows = attach_all(synth(n_players=800, seed=8, hist_bf=(200, 600))
                      + synth(n_players=800, seed=9, hist_bf=(3, 12)))
    n1, w1, n2, w2, mu, th, h = P.to_arrays(rows, "recency_role")
    c = P.fit_c(n1, w1, n2, w2, mu, th, h)
    tab = P.cross_tab(rows, "recency_role", P.SHIPPED_K, c)
    assert tab is not None
    # n1 and H were drawn independently, so the probe must report them as such.
    # This is the number that adjudicates the real table: on live data it is
    # strongly positive, which is what makes the n1 buckets unreadable alone.
    assert abs(tab["corr"]) < 0.15
    hi_h = [tab[(False, True)][1], tab[(True, True)][1]]
    lo_h = [tab[(False, False)][1], tab[(True, False)][1]]
    assert min(hi_h) > max(lo_h), f"high-H {hi_h} did not beat low-H {lo_h}"


def test_cross_tab_declines_to_report_when_there_is_no_history():
    rows = attach_all(synth(n_players=200, seed=6), (0.0, 0.0, 0.0))
    assert P.cross_tab(rows, "recency_role", P.SHIPPED_K, 200.0) is None


def test_paired_bootstrap_brackets_the_point_estimate_on_a_real_signal():
    rows = attach(synth(n_players=600, seed=4), P.RECENCY_WEIGHTS)
    lo, hi = P.paired_bootstrap(rows, "recency_role", True, draws=200)
    point = _dmse(rows, fit=True)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < point < hi
    assert lo > 0.0
