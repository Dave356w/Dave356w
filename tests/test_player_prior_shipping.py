"""v4 wiring: does the personal prior reach the lean, and degrade correctly?

The probe measured whether a personal prior is BETTER. These tests check the
much dumber question that ships bugs: is it wired in, does it fall back to the
v3 behaviour everywhere it should, and does the vector path reproduce the
scalar one it replaced.

The last of those is the important one. A per-player prior is a substitution
into an expression the build already had, so the strongest guarantee available
is that a vector of one repeated value gives byte-identical results to passing
that value as a scalar. If that fails, v4 changed something it did not mean to.
"""
import numpy as np
import pandas as pd
import pytest

import build_site as B
import player_priors as PP


MU = 0.320


@pytest.fixture
def frozen(monkeypatch):
    """A tiny frozen history: one established bat, one arm, one unknown."""
    hist = {
        101: {2025: (0.400, 600.0, "BAT"), 2024: (0.390, 600.0, "BAT"),
              2023: (0.380, 600.0, "BAT")},
        202: {2025: (0.260, 300.0, "RP"), 2024: (0.270, 300.0, "RP")},
    }
    centres = {
        (2023, "BAT"): {"wtd": 0.320, "unw": 0.300, "n": 500, "pa": 1e5},
        (2024, "BAT"): {"wtd": 0.320, "unw": 0.300, "n": 500, "pa": 1e5},
        (2025, "BAT"): {"wtd": 0.320, "unw": 0.300, "n": 500, "pa": 1e5},
        (2024, "RP"): {"wtd": 0.310, "unw": 0.330, "n": 400, "pa": 5e4},
        (2025, "RP"): {"wtd": 0.310, "unw": 0.330, "n": 400, "pa": 5e4},
    }
    monkeypatch.setitem(B._player_prior_state, "loaded", (hist, centres))
    monkeypatch.setattr(B, "SEASON", 2026)
    monkeypatch.setattr(B, "USE_PLAYER_PRIORS", True)
    return hist, centres


# --------------------------------------------------------------------------
# the substitution must be exactly that
# --------------------------------------------------------------------------
def test_a_constant_vector_prior_reproduces_the_scalar_path():
    """The v4 change is a substitution, not a second code path. Prove it."""
    x = pd.Series([0.400, 0.280, np.nan, 0.333])
    n = pd.Series([600.0, 50.0, 120.0, 0.0])
    scalar = B.shrink_xwoba(x, n, MU, B.XWOBA_SHRINK_K)
    vector = B.shrink_xwoba(x, n, [MU] * 4, B.XWOBA_SHRINK_K)
    assert scalar.tolist() == pytest.approx(vector.tolist(), nan_ok=True)


def test_each_row_uses_its_own_prior():
    priors = [0.400, 0.280]
    got = B.shrink_xwoba(pd.Series([0.350, 0.350]), pd.Series([400.0, 400.0]),
                         priors, 400.0)
    # K = n, so each published rate is the midpoint of the rate and ITS prior.
    assert got.tolist() == pytest.approx([(0.350 + 0.400) / 2,
                                          (0.350 + 0.280) / 2])


def test_a_row_with_no_usable_prior_keeps_its_raw_rate():
    """Degrade to unshrunk, never to NaN -- the same choice the scalar guard
    makes, applied per row."""
    got = B.shrink_xwoba(pd.Series([0.350, 0.350]), pd.Series([400.0, 400.0]),
                         [np.nan, 0.280], 400.0)
    assert got.iloc[0] == pytest.approx(0.350)
    assert got.iloc[1] == pytest.approx((0.350 + 0.280) / 2)


def test_a_mismatched_prior_vector_is_refused_not_recycled():
    """Wrong length means the alignment assumption is broken. Fall back to the
    raw rate rather than silently pairing players with other players' priors."""
    x = pd.Series([0.350, 0.360, 0.370])
    got = B.shrink_xwoba(x, pd.Series([400.0] * 3), [0.300, 0.310], 400.0)
    assert got.tolist() == pytest.approx(x.tolist())


# --------------------------------------------------------------------------
# the fallbacks -- every one of them is the v3 behaviour
# --------------------------------------------------------------------------
def test_a_player_with_no_history_gets_the_population_centre(frozen):
    assert B.player_prior_one(999999, MU) == pytest.approx(MU)
    assert B.player_prior_vector([999999, None], MU) == pytest.approx([MU, MU])


def test_priors_switched_off_returns_the_scalar_centre(frozen, monkeypatch):
    monkeypatch.setattr(B, "USE_PLAYER_PRIORS", False)
    assert B.player_prior_vector([101, 202], MU) == pytest.approx(MU)
    assert B.player_prior_one(101, MU) == pytest.approx(MU)


def test_nothing_frozen_returns_the_scalar_centre(monkeypatch):
    """A missing snapshot must not fail the daily build."""
    monkeypatch.setitem(B._player_prior_state, "loaded", ({}, {}))
    assert B.player_prior_vector([101], MU) == pytest.approx(MU)
    assert B.player_prior_one(101, MU) == pytest.approx(MU)


def test_an_unreadable_priors_dir_degrades_rather_than_raises(monkeypatch):
    monkeypatch.delitem(B._player_prior_state, "loaded", raising=False)
    monkeypatch.setattr(B, "PRIORS_DIR", "/nonexistent/priors/dir")
    assert B.player_prior_history() == ({}, {})


# --------------------------------------------------------------------------
# the value itself
# --------------------------------------------------------------------------
def test_an_established_bat_is_pulled_toward_his_own_line(frozen):
    """Deviations +.080/+.070/+.060 on equal samples, weighted .5/.3/.2, give
    +0.073 on H = 600. Against C = 200 the prior keeps 600/800 of that."""
    dev = 0.5 * 0.080 + 0.3 * 0.070 + 0.2 * 0.060
    assert dev == pytest.approx(0.073)
    got = B.player_prior_one(101, MU)
    assert got == pytest.approx((600.0 * (MU + dev) + 200.0 * MU) / 800.0,
                                abs=1e-9)
    assert MU < got < MU + dev


def test_a_reliever_reads_against_his_pool_not_the_league(frozen):
    """RP history is centred on the relief pool's UNWEIGHTED centre (0.330),
    which is where the shipped target sits -- not the league rate."""
    got = B.player_prior_one(202, MU)
    # 2025: 0.260 - 0.330 = -0.070 at weight .5*300; 2024: same at .3*300.
    dev = (0.5 * 300 * (0.260 - 0.330) + 0.3 * 300 * (0.270 - 0.330)) / (
        0.5 * 300 + 0.3 * 300)
    h = 0.5 * 300 + 0.3 * 300
    assert got == pytest.approx((h * (MU + dev) + 200.0 * MU) / (h + 200.0))
    assert got < MU          # a good reliever must land BELOW the centre


def test_the_prior_never_leaves_the_interval_between_centre_and_history(frozen):
    """Empirical Bayes cannot extrapolate. A prior outside [mu, theta_hist] is
    a sign the weights are wrong, and it would move a lean in a direction no
    input supports."""
    for pid in (101, 202):
        th, h = PP.centred_history(B.player_prior_history()[0][pid],
                                   B.player_prior_history()[1], 2026, MU)
        got = B.player_prior_one(pid, MU)
        assert min(MU, th) - 1e-9 <= got <= max(MU, th) + 1e-9


# --------------------------------------------------------------------------
# the tag, which is how any of this stays readable later
# --------------------------------------------------------------------------
def test_v4_starts_its_own_record_and_scale_family():
    assert B.MODEL_TAG == "woba+plat_consol_v4"
    assert B._RECORD_FAMILIES["woba+plat_consol_v4"] == ("woba+plat_consol_v4",)
    assert B._SCALE_FAMILIES["woba+plat_consol_v4"] == ("woba+plat_consol_v4",)
    # And it must not have been quietly added to an older family.
    for tag, fam in B._RECORD_FAMILIES.items():
        if tag != "woba+plat_consol_v4":
            assert "woba+plat_consol_v4" not in fam


def test_the_grader_agrees_with_the_build_about_the_tag():
    """One value, two modules -- the workflow-pin entry in CLAUDE.md is what
    happens when these drift."""
    import grade_leans

    assert grade_leans.MODEL_TAG == B.MODEL_TAG
    assert (grade_leans._RECORD_FAMILIES["woba+plat_consol_v4"]
            == B._RECORD_FAMILIES["woba+plat_consol_v4"])


def test_the_constant_is_read_from_one_place():
    assert B.player_priors.PRIOR_HISTORY_C == PP.PRIOR_HISTORY_C
    assert PP.RECENCY == (0.50, 0.30, 0.20)


def test_the_committed_snapshot_covers_the_three_prior_seasons():
    """The build reads 2023-2025 for a 2026 slate; if a season were missing the
    ladder would silently drop a rung."""
    hist, centres = PP.load_priors("data")
    assert len(hist) > 1000
    for season in (2023, 2024, 2025):
        assert (season, "BAT") in centres
        assert (season, "RP") in centres
        assert any(season in seasons for seasons in hist.values())
