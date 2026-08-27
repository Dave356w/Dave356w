import math
import re
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as model
import historical_data
import walkforward


def _source(**overrides):
    row = {
        "game_pk": 1,
        "game_date": "2026-07-10",
        "away": "ARI",
        "home": "BOS",
        "away_sp": "Away Pitcher",
        "home_sp": "Home Pitcher",
        "lineup_status_away": "posted",
        "lineup_status_home": "posted",
        "full_away": 2,
        "full_home": 5,
        "f5_away": 1,
        "f5_home": 3,
        "xw_lean": "ARI",
        "xw_net": -0.01,
        "xw_full": "L",
        "xw_f5": "L",
        "model_tag": "xw+plat_consol_v9",
        "close_p_home": 0.55,
        "close_away_ml": 130,
        "close_home_ml": 150,
        "act_sp_ip_away": 5.0,
        "act_sp_ip_home": 6.0,
    }
    row.update(overrides)
    return row


def test_explicit_ip_history_never_reuses_live_process_cache():
    model._sp_ip_cal_state.clear()
    model._sp_ip_cal_state["fit"] = (99.0, 99.0, 999, 1.0)
    history = pd.DataFrame({
        "expected_sp_ip_raw_away": [4.0, 6.0],
        "act_sp_ip_away": [5.0, 5.0],
    })
    fit = model.sp_ip_calibration(history)
    assert fit is not None
    intercept, slope, n, weight = fit
    assert intercept == pytest.approx(5.0)
    assert slope == pytest.approx(0.0)
    assert n == 2
    assert weight == pytest.approx(2 / 52)
    assert model._sp_ip_cal_state["fit"] == (99.0, 99.0, 999, 1.0)


def test_missing_historical_roster_never_falls_back_to_live(monkeypatch):
    monkeypatch.setattr(
        model, "pitcher_roster",
        lambda team_id: (_ for _ in ()).throw(AssertionError("live roster leak")),
    )
    roles = {
        10: {"appearances": 10, "start_share": 0.0,
             "avg_ip_per_appearance": 1.0},
    }
    assert model.relief_pitcher_ids(1, roles, roster_ids=None) == []


def test_historical_provider_uses_previous_day_for_every_bounded_source(
        tmp_path, monkeypatch):
    provider = historical_data.HistoricalDataProvider(
        "2026-07-10", pd.DataFrame([_source()]), cache_dir=tmp_path
    )
    params = provider._savant_params("batter")
    assert params["game_date_lt"] == "2026-07-09"
    assert params["game_date_gt"] == "2026-03-01"
    assert params["hfGT"] == "R|"

    calls = []
    monkeypatch.setattr(
        historical_data.model,
        "load_team_pitcher_roles",
        lambda team_id, start_date=None, end_date=None: calls.append(
            (team_id, start_date, end_date)
        ) or {},
    )
    provider.load_team_pitcher_roles(147)
    assert calls == [(147, "2026-03-01", "2026-07-09")]


def test_historical_lineup_fidelity_uses_source_pregame_status(
        tmp_path, monkeypatch):
    provider = historical_data.HistoricalDataProvider(
        "2026-07-10", pd.DataFrame([_source()]), cache_dir=tmp_path
    )
    monkeypatch.setattr(
        historical_data.model, "gf_lineups",
        lambda game_pk: (list(range(1, 10)), list(range(11, 20))),
    )
    monkeypatch.setattr(provider, "_hitter_roster", lambda team_id, stat: [])
    stats = {pid: {"xwOBA": 0.300, "PA": 100} for pid in range(1, 20)}
    lineup, meta = provider.resolve_lineup(
        1, "away", 109, stats, return_meta=True, league_xwoba=0.315
    )
    assert lineup == list(range(1, 10))
    assert meta["status"] == "posted"
    provider._starter_fidelity[(1, "away")] = "exact_pregame"
    provider._starter_fidelity[(1, "home")] = "exact_pregame"
    provider._lineup_fidelity[(1, "home")] = "exact_pregame"
    provider._stat_fidelity = {
        "batter": "exact_pregame", "pitcher": "exact_pregame",
    }
    assert provider.input_fidelity(1) == "exact_pregame"


def test_archived_live_savant_cache_upgrades_stat_fidelity(tmp_path):
    for player_type in ("batter", "pitcher"):
        pd.DataFrame([{
            "player_id": 10, "xwoba": 0.301, "pa": 50,
            "xba": 0.250, "xslg": 0.400,
        }]).to_csv(
            tmp_path /
            f"savant_cache_custom_xwoba_v1_{player_type}_2026-07-10.csv",
            index=False,
        )
    provider = historical_data.HistoricalDataProvider(
        "2026-07-10", pd.DataFrame([_source()]),
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path,
    )
    for player_type in ("batter", "pitcher"):
        stats, _bb, _frame = provider.load_stat_lookups(player_type)
        assert stats[10]["xwOBA"] == pytest.approx(0.301)
    provider._starter_fidelity = {
        (1, "away"): "exact_pregame", (1, "home"): "exact_pregame",
    }
    provider._lineup_fidelity = {
        (1, "away"): "exact_pregame", (1, "home"): "exact_pregame",
    }
    assert provider.input_fidelity(1) == "exact_pregame"


def test_compose_output_grades_current_lean_and_closing_roi():
    class Provider:
        as_of_date = "2026-07-09"

        @staticmethod
        def input_fidelity(game_pk):
            return "exact_pregame"

        @staticmethod
        def error_for(game_pk):
            return None

    prediction = {
        "xw_lean": "BOS",
        "xw_net": 0.02,
        "xw_delta": 0.02,
        "home_off_edge": 0.01,
        "away_off_edge": -0.01,
        "expected_sp_ip_raw_away": 5.2,
        "expected_sp_ip_raw_home": 5.8,
        "expected_sp_ip_away": 5.3,
        "expected_sp_ip_home": 5.7,
    }
    row = walkforward.compose_output_row(
        pd.Series(_source()), prediction, Provider(), "abc123"
    )
    assert row["wf_full_grade"] == "W"
    assert row["wf_f5_grade"] == "W"
    assert row["changed_from_original"] is True
    assert row["wf_vs_market"] == "AGREE"
    assert row["wf_close_ml"] == 150
    assert row["wf_roi"] == pytest.approx(1.5)


def test_predict_slate_routes_through_current_model_functions(monkeypatch):
    calls = []

    class Provider:
        def __init__(self, *args, **kwargs):
            calls.append("provider")

    monkeypatch.setattr(walkforward, "HistoricalDataProvider", Provider)
    monkeypatch.setattr(
        walkforward.model, "fetch_all",
        lambda *args, **kwargs: calls.append(("fetch", kwargs)) or {
            "empty": False,
            "pitchers_df": pd.DataFrame({"x": [1]}),
            "league_baseline": {"xwOBA": 0.315},
            "pitching_plans": {},
        },
    )
    monkeypatch.setattr(
        walkforward.model, "build_xwoba_matchup",
        lambda *args: calls.append("matchup") or (
            pd.DataFrame({
                "game_pk": [1], "side": ["away"],
                "game_datetime_utc": ["2026-07-10T23:00:00Z"],
            }),
            pd.DataFrame(), pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        walkforward.model, "apply_pitching_plans",
        lambda frame, *args: calls.append("plans") or frame,
    )
    monkeypatch.setattr(
        walkforward.grader, "rows_from_dump",
        lambda frame, pl: calls.append("rows") or [{"game_pk": 1}],
    )

    rows, _provider = walkforward.predict_slate(
        "2026-07-10", pd.DataFrame([_source()]), pd.DataFrame(), "cache"
    )
    assert rows == {1: {"game_pk": 1}}
    assert calls[0] == "provider"
    assert calls[1][0] == "fetch"
    assert calls[1][1]["write_audit"] is False
    assert calls[1][1]["include_platoon"] is False
    assert calls[2:] == ["matchup", "plans", "rows"]


def test_walkforward_refuses_to_overwrite_production_ledger(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("game_pk,game_date\n")
    with pytest.raises(ValueError, match="must not overwrite"):
        walkforward.run_walkforward(ledger_path=ledger, output_path=ledger)


def test_market_roi_edge_cases():
    assert walkforward.flat_unit_roi("W", -200) == pytest.approx(0.5)
    assert walkforward.flat_unit_roi("L", 200) == pytest.approx(-1.0)
    assert walkforward.flat_unit_roi("T", 200) == pytest.approx(0.0)
    assert math.isnan(walkforward.flat_unit_roi(None, 200))


def test_replay_has_its_own_workflow_off_the_pregame_path():
    """The replay must not sit inside the build.

    It did, bounded at 450s so it could never delay the capture of pregame
    rows -- which cannot be re-derived later without lookahead. That bound is
    also why it never finished: on 2026-08-27 the committed replay ended at
    2026-08-11 against a ledger running to 08-27. Its own workflow is what
    lets it reach the present, so a regression that moves it back into
    build.yml has to fail here.
    """
    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    build = (root / "build.yml").read_text()
    replay = (root / "walkforward.yml").read_text()

    assert "walkforward.py" not in build.replace(
        "walkforward.yml", ""), "the replay is back on the pregame build path"
    assert "python walkforward.py" in replay
    assert "path: .walkforward_cache" in replay

    # Bounded on the process, not only on the step: a step timeout kills the
    # shell and leaves the python child writing to data/ while the commit step
    # reads it (see WorkflowStepTimeoutTests in tests/test_integrity_fixes.py).
    assert re.search(r"run: timeout\b.*\bpython walkforward\.py", replay)
    assert "continue-on-error: true" in replay

    # It writes data/, so it commits through the signed path like every other
    # writer, and validates before doing so.
    assert "python validate_data_files.py" in replay
    assert "python commit_data.py" in replay

    # Deliberately NOT in the site-build group: that group is serialized with
    # cancel-in-progress:false, so joining it would queue a long replay ahead
    # of a pregame build.
    assert "group: walkforward" in replay
    assert "group: site-build" not in replay


def test_replay_cache_key_is_code_not_bytes(tmp_path):
    """The cache key is the model's CODE, not the file's bytes.

    Keyed on bytes, every edit discarded the whole replay and restarted it
    from the first slate -- three merged display-only commits did exactly that
    on 2026-08-27, one of them almost entirely deletions of dead code. Paired
    with a bounded run, a cache that resets on comments never finishes.

    Both directions are asserted. Relaxing the key is only safe while a real
    code change still invalidates it: a stale row silently mixed into a new
    replay is a much worse failure than a spurious reset.

    Operates on temp files rather than mutating the repo's own build_site.py:
    a test that rewrites a production module leaves it corrupted if the run is
    interrupted between the write and the restore.
    """
    base = "X = 1\n\n\ndef f(a):\n    return a + X\n"
    src = tmp_path / "m.py"

    src.write_text(base)
    before = walkforward._code_only(src)

    src.write_text(base + "\n# a comment that changes nothing\n")
    assert walkforward._code_only(src) == before, "a comment changed the key"

    src.write_text("# leading comment\n" + base)
    assert walkforward._code_only(src) == before, "a line shift changed the key"

    src.write_text('"""A module docstring."""\n' + base)
    assert walkforward._code_only(src) == before, "a docstring changed the key"

    src.write_text(base.replace("return a + X", '"""Doc."""\n    return a + X'))
    assert walkforward._code_only(src) == before, "a function docstring changed the key"

    # ...but any real change must still invalidate it
    for changed, why in (
        ("X = 2\n\n\ndef f(a):\n    return a + X\n", "a constant's value"),
        ("X = 1\n\n\ndef f(a):\n    return a - X\n", "an operator"),
        ("X = 1\n\n\ndef f(a=0):\n    return a + X\n", "an argument default"),
        ("X = 1\n\n\ndef g(a):\n    return a + X\n", "a function name"),
    ):
        src.write_text(changed)
        assert walkforward._code_only(src) != before, f"{why} did NOT invalidate the key"


def test_replay_config_hash_is_stable_and_covers_the_model(tmp_path):
    """The real key is deterministic and keyed to the model tag."""
    first = walkforward.replay_config_hash()
    assert first == walkforward.replay_config_hash()
    assert len(first) == 16
    with mock.patch.object(walkforward.model, "MODEL_TAG", "xw+plat_consol_v999"):
        assert walkforward.replay_config_hash() != first


def test_report_declares_a_partial_replay(tmp_path):
    """A truncated replay must not read as a complete backtest.

    Dates replay in order, so what a bounded run drops is always the MOST
    RECENT window -- the one a reader most wants. The report on disk on
    2026-08-27 said "Games 690" beside a frame of 500 rows ending 08-11.
    """
    frame = pd.DataFrame({
        "game_date": ["2026-07-01", "2026-07-02"],
        "wf_lean": ["NYY", "TB"],
        "wf_full_grade": ["W", "L"],
        "wf_f5_grade": ["W", "L"],
        "wf_market_pick_prob": [0.55, 0.45],
        "wf_roi": [0.9, -1.0],
        "input_fidelity": ["historical_lineup"] * 2,
        "original_full_grade": ["W", "L"],
        "changed_from_original": [False, False],
    })
    source = pd.DataFrame({"game_date": ["2026-07-01", "2026-07-02",
                                         "2026-07-03", "2026-07-04"]})

    partial = walkforward.render_report(frame, source)
    assert "PARTIAL REPLAY" in partial
    assert "MOST RECENT" in partial
    assert "2026-07-03" in partial and "2026-07-04" in partial
    assert "NOT REPLAYED 2 games over 2 dates" in partial

    complete = walkforward.render_report(frame, source.iloc[:2])
    assert "PARTIAL REPLAY" not in complete
    assert "Complete" in complete

    # and with no source at all it must not silently claim completeness
    assert "Complete" not in walkforward.render_report(frame)


def test_coverage_counts_dates_and_games_it_did_not_reach():
    frame = pd.DataFrame({"game_date": ["2026-07-01"] * 3})
    source = pd.DataFrame({"game_date": ["2026-07-01"] * 3 + ["2026-07-02"] * 5})
    cov = walkforward.coverage(frame, source)
    assert cov["games"] == 3 and cov["dates"] == 1
    assert cov["missing_dates"] == ["2026-07-02"]
    assert cov["missing_games"] == 5
    assert cov["partial"] is True
    assert walkforward.coverage(frame, frame)["partial"] is False
