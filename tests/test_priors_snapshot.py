"""Guards on the frozen prior files: no lookahead, no silent rewrite, no drift.

The fetch needs Savant and cannot run here. Everything around it can, and the
parts that would be wrong *silently* are the ones tested: a season that has not
finished, a season file quietly rewritten under a ledger row that cites it, and
a rate carried across seasons without its centre.
"""
import os

import pandas as pd
import pytest

import build_site
import player_priors
import priors_snapshot as S


BAT = [
    {"season": 2024, "player_id": 1, "player_name": "A", "role": "BAT",
     "pa": 600.0, "woba": 0.400},
    {"season": 2024, "player_id": 2, "player_name": "B", "role": "BAT",
     "pa": 200.0, "woba": 0.300},
]
ARMS = [
    {"season": 2024, "player_id": 3, "player_name": "C", "role": "SP",
     "pa": 700.0, "woba": 0.290},
    {"season": 2024, "player_id": 4, "player_name": "D", "role": "RP",
     "pa": 100.0, "woba": 0.260},
    {"season": 2024, "player_id": 5, "player_name": "E", "role": "RP",
     "pa": 300.0, "woba": 0.320},
]


# --------------------------------------------------------------------------
# no-lookahead, enforced rather than trusted
# --------------------------------------------------------------------------
def test_the_current_season_is_refused():
    """An in-progress leaderboard is a slate-dependent snapshot, which is the
    thing `.savant_cache/` is gitignored to prevent. Refuse it at the door."""
    with pytest.raises(SystemExit) as e:
        S.main(["--seasons", str(build_site.SEASON), "--dry-run"])
    assert "completed season" in str(e.value)


def test_a_future_season_is_refused():
    with pytest.raises(SystemExit):
        S.main(["--seasons", str(build_site.SEASON + 1), "--dry-run"])


def test_an_existing_season_file_is_never_silently_rewritten(tmp_path, monkeypatch):
    """A ledger row built against a prior must stay reconcilable."""
    monkeypatch.setattr(S, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(S, "PRIORS_TEMPLATE",
                        os.path.join(str(tmp_path), "woba_priors_{season}.csv"))
    monkeypatch.setattr(S, "CENTRES_PATH",
                        os.path.join(str(tmp_path), "woba_prior_centres.csv"))
    rows = BAT + ARMS
    S.write_season(2024, rows, S.season_centres(rows))
    with pytest.raises(SystemExit) as e:
        S.write_season(2024, rows, S.season_centres(rows))
    assert "immutable" in str(e.value)
    # --force is the documented repair path, and it must actually work.
    S.write_season(2024, rows, S.season_centres(rows), force=True)


def test_a_dry_run_is_not_blocked_by_an_already_frozen_season(tmp_path, monkeypatch):
    """A dry run writes nothing, so the immutability guard must not stop it.

    Regression: once 2023-2025 were frozen, the PR-event dry run started
    failing on `data/woba_priors_2023.csv already exists` -- the fetch check
    this workflow exists for became a permanent red the moment it had done its
    job once. The guard now protects the write, not the report.
    """
    monkeypatch.setattr(S, "PRIORS_TEMPLATE",
                        os.path.join(str(tmp_path), "woba_priors_{season}.csv"))
    monkeypatch.setattr(S, "CENTRES_PATH",
                        os.path.join(str(tmp_path), "woba_prior_centres.csv"))
    rows = BAT + ARMS
    S.write_season(2024, rows, S.season_centres(rows))
    S.write_season(2024, rows, S.season_centres(rows), dry_run=True)
    # ...and it still wrote nothing: the file is byte-identical.
    before = (tmp_path / "woba_priors_2024.csv").read_bytes()
    S.write_season(2024, [BAT[0]], S.season_centres(BAT), dry_run=True)
    assert (tmp_path / "woba_priors_2024.csv").read_bytes() == before


def test_an_empty_pull_never_becomes_a_prior_file(tmp_path, monkeypatch):
    """A report over an empty pull looks like a measurement. So does a prior."""
    monkeypatch.setattr(S, "build_season", lambda *a, **k: ([], []))
    monkeypatch.setattr(build_site, "SEASON", 2026)
    with pytest.raises(SystemExit) as e:
        S.main(["--seasons", "2024"])
    assert "empty prior file" in str(e.value)


def test_a_renamed_savant_column_fails_loudly():
    """Savant renaming its export must not yield a file of dropped rows."""
    df = pd.DataFrame([{"player_id": 1, "plate_appearances": 600,
                        "xwoba": 0.400}])
    with pytest.raises(RuntimeError) as e:
        S._rows_from(df, 2024, lambda _pid: "BAT")
    assert "Refusing" in str(e.value)


# --------------------------------------------------------------------------
# what gets frozen
# --------------------------------------------------------------------------
def test_centres_are_stored_both_weighted_and_unweighted():
    """The build uses both, for different populations and different reasons."""
    by_pool = {c["pool"]: c for c in S.season_centres(BAT + ARMS)}
    assert set(by_pool) == {"BAT", "SP", "RP", "P"}
    bat = by_pool["BAT"]
    assert bat["woba_wtd"] == pytest.approx((0.400 * 600 + 0.300 * 200) / 800)
    assert bat["woba_unw"] == pytest.approx(0.350)
    rp = by_pool["RP"]
    assert rp["woba_unw"] == pytest.approx(0.290)
    assert rp["woba_wtd"] == pytest.approx((0.260 * 100 + 0.320 * 300) / 400)
    # The pitcher pool is every arm, so it can be read even when a role is
    # unresolved. It must not double-count.
    assert by_pool["P"]["n"] == 3


def test_role_unresolved_pitchers_are_kept_not_dropped():
    df = pd.DataFrame([{"player_id": 9, "pa": 120, "woba": 0.310}])
    rows = S._rows_from(df, 2024, lambda pid: {}.get(pid, "P"))
    assert rows and rows[0]["role"] == "P"


def test_rows_without_a_usable_rate_are_dropped_not_zero_filled():
    df = pd.DataFrame([
        {"player_id": 1, "pa": 0, "woba": 0.400},
        {"player_id": 2, "pa": 500, "woba": None},
        {"player_id": 3, "pa": 500, "woba": 0.330},
    ])
    rows = S._rows_from(df, 2024, lambda _pid: "BAT")
    assert [r["player_id"] for r in rows] == [3]


def test_the_frozen_rate_column_matches_the_files_it_writes():
    """The prior files are wOBA-denominated and immutable, so RATE_COL is
    pinned to wOBA rather than following the build's active metric.

    This test used to assert the opposite (`RATE_COL == MODEL_RATE_SOURCE_COL`)
    and that was right for exactly as long as the build ran wOBA. Following the
    build across the v11 revert would have written xwOBA rows into
    `woba_priors_<season>.csv`, under `woba_wtd`/`woba_unw` headers, with no
    error anywhere -- and every ledger row built against the old priors would
    have stopped being reconcilable. What actually has to hold is that the
    column, the filenames and the centre headers all name ONE metric.
    """
    assert S.RATE_COL == "woba"
    assert "woba_priors_" in S.PRIORS_TEMPLATE
    assert S.CENTRES_PATH.endswith("woba_prior_centres.csv")
    assert {"woba_wtd", "woba_unw"} <= set(S.CENTRE_COLUMNS)
    # The reading side must name the same metric, or the pair drifts.
    assert player_priors.PRIORS_PREFIX == "woba_priors_"
    assert player_priors.CENTRES_NAME == "woba_prior_centres.csv"


def test_a_non_woba_build_cannot_reach_the_frozen_woba_priors(monkeypatch):
    """The guard that makes the pin above safe rather than merely tidy."""
    monkeypatch.setattr(build_site, "MODEL_RATE_LABEL", "xwOBA")
    assert build_site.player_prior_history() == ({}, {})


# --------------------------------------------------------------------------
# the reading side
# --------------------------------------------------------------------------
def _seed(tmp_path):
    rows = BAT + ARMS
    pd.DataFrame(rows, columns=S.PRIOR_COLUMNS).to_csv(
        tmp_path / "woba_priors_2024.csv", index=False)
    older = [dict(r, season=2023, woba=r["woba"] - 0.010) for r in rows]
    pd.DataFrame(older, columns=S.PRIOR_COLUMNS).to_csv(
        tmp_path / "woba_priors_2023.csv", index=False)
    centres = S.season_centres(rows) + S.season_centres(older)
    pd.DataFrame(centres, columns=S.CENTRE_COLUMNS).to_csv(
        tmp_path / "woba_prior_centres.csv", index=False)
    return rows


def test_load_priors_round_trips_what_was_written(tmp_path):
    _seed(tmp_path)
    hist, centres = S.load_priors(str(tmp_path))
    assert hist[1][2024] == (0.400, 600.0, "BAT")
    assert hist[1][2023][0] == pytest.approx(0.390)
    assert (2024, "BAT") in centres and (2023, "RP") in centres


def test_load_priors_degrades_to_empty_rather_than_failing(tmp_path):
    """Nothing committed yet must leave the caller on the shipped behaviour."""
    hist, centres = S.load_priors(str(tmp_path))
    assert hist == {} and centres == {}


def test_centred_history_carries_the_deviation_not_the_level(tmp_path):
    """A 2023 rate must not import 2023's offence into today's prediction.

    Player 1 sits +0.025 above the PA-weighted batter centre in 2024 (0.400 vs
    0.375). The seeded 2023 file shifts EVERY rate down 0.010, so his 2023
    deviation is the same +0.025 -- the league move cancels, which is the whole
    point. Read against a present-day centre of 0.330 his history must land at
    0.355, whatever the league did in either season.
    """
    _seed(tmp_path)
    hist, centres = S.load_priors(str(tmp_path))
    theta, h = S.centred_history(hist[1], centres, 2025, 0.330,
                                 weights=(0.5, 0.3))
    assert h == pytest.approx(0.5 * 600.0 + 0.3 * 600.0)
    assert theta == pytest.approx(0.355, abs=1e-9)


def test_centred_history_reads_relievers_against_their_unweighted_centre(tmp_path):
    """RP is the one pool the build shrinks toward an UNWEIGHTED centre."""
    _seed(tmp_path)
    hist, centres = S.load_priors(str(tmp_path))
    theta, _ = S.centred_history(hist[4], centres, 2025, 0.300, weights=(1.0,))
    rp_unw = centres[(2024, "RP")]["unw"]
    assert theta == pytest.approx(0.300 + (0.260 - rp_unw))


def test_centred_history_is_the_population_centre_when_there_is_no_history(tmp_path):
    """The rookie fallback is H=0, not a branch. Verify there is no branch."""
    _seed(tmp_path)
    hist, centres = S.load_priors(str(tmp_path))
    theta, h = S.centred_history(hist.get(999), centres, 2025, 0.330,
                                 weights=(0.5, 0.3))
    assert h == 0.0
    import player_prior_probe as P
    assert P.personal_prior(theta, h, 0.330, 400.0) == pytest.approx(0.330)


def test_centred_history_skips_a_season_with_no_stored_centre(tmp_path):
    """A rate with no centre beside it cannot be de-levelled, so it is not used
    -- silently importing its raw level is the failure this guards."""
    _seed(tmp_path)
    hist, centres = S.load_priors(str(tmp_path))
    centres.pop((2024, "BAT"))
    theta, h = S.centred_history(hist[1], centres, 2025, 0.330,
                                 weights=(0.5, 0.3))
    # Only 2023 survives: rate 0.390 against that season's weighted centre
    # 0.365, so the same +0.025 deviation on a third of the weight.
    assert h == pytest.approx(0.3 * 600.0)
    assert theta == pytest.approx(0.330 + 0.025, abs=1e-9)


def test_committed_prior_files_validate_if_they_exist():
    """Whatever is committed must pass the same gate every other CSV does."""
    import validate_data_files as V

    for name in sorted(os.listdir("data")):
        if name.startswith("woba_prior") and name.endswith(".csv"):
            V.validate_csv(os.path.join("data", name))
