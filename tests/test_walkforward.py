import math
import re
import os
import sys
from pathlib import Path

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


def test_site_workflow_appends_walkforward_after_grading_before_build():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "build.yml"
    ).read_text()
    grade = workflow.index("- name: Grade leans (pre-build)")
    replay = workflow.index(
        "- name: Append current-model walk-forward ledger (pre-build)"
    )
    build = workflow.index("- name: Build static site")
    assert grade < replay < build
    replay_block = workflow[replay:build]
    assert "continue-on-error: true" in replay_block
    assert "timeout-minutes: 8" in replay_block
    # Bounded on the process, not only on the step: a step timeout kills the
    # shell and leaves the python child writing to data/ (see
    # WorkflowStepTimeoutTests in tests/test_integrity_fixes.py).
    assert re.search(r"run: timeout\b.*\bpython walkforward\.py", replay_block)
    assert "path: .walkforward_cache" in workflow
    assert "git add data/" in workflow
