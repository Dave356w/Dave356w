"""Historical input causality, production-path parity and independent grading."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import build_site as bs
import historical_replay as hr


def pitches():
    rows = []
    for day, pk in (("2026-03-28", 1), ("2026-03-29", 2), ("2026-03-30", 3)):
        for ab, (event, val, est, bb) in enumerate([
                ("single", .9, .6, "line_drive"), ("strikeout", 0, None, None),
                ("walk", .7, None, None), ("intent_walk", 0, None, None),
                ("field_out", 0, None, "fly_ball")], 1):
            r = dict.fromkeys(hr.PITCH_COLS)
            r.update(game_date=day, game_pk=pk, game_type="R", at_bat_number=ab,
                pitch_number=1, batter=10, pitcher=20, events=event, bb_type=bb,
                inning=1, inning_topbot="Top", stand="R", p_throws="L",
                pitch_type="FF", release_speed=95, launch_speed=90, launch_angle=20,
                estimated_woba_using_speedangle=est, woba_value=val,
                woba_denom=0 if event == "intent_walk" else 1,
                post_home_score=2, post_away_score=1)
            rows.append(r)
    return pd.DataFrame(rows)


def game(pk=1, day="2026-03-28", home=2, away=1):
    return {"gamePk": pk, "_date": day, "officialDate": day, "gameType": "R",
            "gameDate": day + "T20:00:00Z", "status": {"detailedState": "Final"},
            "teams": {"home": {"team": {"id": 100, "name": "Boston Red Sox"}, "score": home},
                      "away": {"team": {"id": 200, "name": "New York Yankees"}, "score": away}}}


def history():
    raw = hr.validate_pitches(pitches(), "2026-03-28", "2026-03-30")
    gs = {pk: game(pk, day) for pk, day in [(1,"2026-03-28"),(2,"2026-03-29"),(3,"2026-03-30")]}
    return (*hr.prepare_history(raw, gs), gs)


def test_rate_board_uses_pa_denominator_and_keeps_walks_strikeouts():
    _, pa, _ = history()
    b = hr.rate_board(pa, "batter", "2026-03-29").iloc[0]
    assert b["pa"] == 5  # includes IBB in sample size, excludes it in wOBA denominator
    assert b["xwoba"] == pytest.approx((.6 + 0 + .7 + 0) / 4)
    assert b["woba"] == pytest.approx((.9 + 0 + .7 + 0) / 4)
    assert b["k_percent"] == 20
    assert b["bbe"] == 2
    assert pa["untracked_bip"].sum() == 3


def test_same_day_doubleheader_and_future_results_cannot_change_inputs():
    _, pa, _ = history()
    expected = hr.rate_board(pa, "pitcher", "2026-03-29")
    changed = pa.copy()
    changed.loc[changed.available_date >= "2026-03-29", ["x_value", "w_value"]] = 999
    pd.testing.assert_frame_equal(expected, hr.rate_board(changed, "pitcher", "2026-03-29"))
    assert hr.rate_board(pa, "pitcher", "2026-03-28").empty


def test_resumed_game_is_unavailable_until_completion():
    raw = pitches().query("game_pk == 1")
    g = game()
    g["resumeDate"] = "2026-03-30T20:00:00Z"
    _, pa = hr.prepare_history(raw, {1: g})
    assert hr.rate_board(pa, "batter", "2026-03-30").empty
    assert not hr.rate_board(pa, "batter", "2026-03-31").empty


@pytest.mark.parametrize("bad", ["missing_game", "score", "column", "duplicate"])
def test_incomplete_history_stops_instead_of_creating_false_coverage(bad):
    raw = pitches().query("game_pk == 1").copy()
    gs = {1: game()}
    with pytest.raises(hr.MissingHistory):
        if bad == "missing_game":
            gs[2] = game(2)
        elif bad == "score":
            raw["post_home_score"] = 12
        elif bad == "column":
            hr.validate_pitches(raw.drop(columns="pitcher"), "2026-03-28", "2026-03-28")
        else:
            hr.validate_pitches(pd.concat([raw, raw]), "2026-03-28", "2026-03-28")
        hr.prepare_history(raw, gs)


def team(start=1, pitcher=20):
    players = {f"ID{start+i}": {"person": {"id": start+i}, "battingOrder": str((i+1)*100)}
               for i in range(9)}
    players["ID99"] = {"person": {"id": 99}, "battingOrder": "101"}
    players[f"ID{pitcher}"] = {"person": {"id": pitcher, "fullName": f"Pitcher {pitcher}"},
        "stats": {"pitching": {"gamesStarted": 1, "inningsPitched": "5.2"}}}
    return {"players": players, "pitchers": [pitcher, 99]}


def test_original_order_does_not_include_pinch_hitters():
    assert hr.starting_order(team()) == list(range(1,10))
    assert hr.starting_pitcher(team())[0] == 20
    broken = team()
    del broken["players"]["ID3"]
    with pytest.raises(hr.MissingHistory):
        hr.starting_order(broken)


def test_empty_historical_fit_cannot_read_todays_ledger(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("production ledger accessed")
    monkeypatch.setattr(bs, "load_ledger_df", forbidden)
    assert bs.calibrate_sp_ip(5.2, None) == 5.2
    assert bs.sp_ip_calibration(pd.DataFrame()) is None
    # Explicitly absent fit stays identity even when a live fit is cached.
    bs._sp_ip_cal_state["fit"] = (99, 99, 99, 1)
    assert bs.calibrate_sp_ip(5.2, None) == 5.2
    bs._sp_ip_cal_state.clear()


def test_model_environment_restores_live_caches_even_after_exception():
    old = bs._roster_cache
    old_season = bs.SEASON
    with pytest.raises(RuntimeError):
        with hr.model_environment(lambda *a, **kw: {}, 2024, "2024-04-01"):
            bs._roster_cache[1] = [99]
            raise RuntimeError("test")
    assert bs._roster_cache is old
    assert bs.SEASON == old_season


def test_output_and_cache_cannot_replace_production_files():
    for p in [bs.DATA_DIR, bs.OUT_DIR, str(bs.DATA_DIR) + "/backtest", "."]:
        with pytest.raises(ValueError):
            hr.safe_output(p)


def test_grading_uses_game_id_correct_side_and_f5_ties():
    raw, pa, gs = history()
    g = game(1, "2026-03-28", home=2, away=1)
    g["linescore"] = {"innings": [{"num": i, "away": {"runs": 0}, "home": {"runs": 0}}
                                    for i in range(1, 6)]}
    provider = hr.HistoricalDataProvider("2026-03-28", 2026, raw, pa, {1:g}, None)
    provider.boxes[1] = {"teams": {"away": team(1,20), "home": team(30,40)}}
    frame = pd.DataFrame([
        {"game_pk": 1, "side": "away", "edge_xwOBA": .02, "expected_sp_ip_raw": 5.5},
        {"game_pk": 1, "side": "home", "edge_xwOBA": -.01, "expected_sp_ip_raw": 6.0}])
    rows, cal = hr.assemble_rows(provider, frame, {})
    assert rows[0]["xw_net"] == pytest.approx(.03)
    assert rows[0]["xw_lean"] == "BOS"
    assert rows[0]["xw_full"] == "W"
    assert rows[0]["xw_f5"] == "T"
    assert cal[0]["act_sp_ip_away"] == pytest.approx(5 + 2/3)
    assert hr.report_summary(pd.DataFrame(rows))["f5"]["decisions"] == 0
    # A fifth inning not completed on BOTH sides cannot be graded as a tie.
    del g["linescore"]["innings"][-1]["home"]["runs"]
    assert hr.game_result(g)["f5_home"] is None


def test_cached_api_has_offline_reproducibility_and_rejects_live_savant(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"roster": [{"x": 1}]}
    monkeypatch.setattr(hr.requests, "get", lambda *a, **kw: Response())
    api = hr.CachedAPI(tmp_path)
    url = hr.API + "/teams/1/roster"
    result = api(url, {"date": "2026-03-28"})
    offline = hr.CachedAPI(tmp_path, offline=True)
    assert offline(url, {"date": "2026-03-28"}) == result
    with pytest.raises(hr.MissingHistory):
        offline(url, {"date": "2026-03-29"})
    with pytest.raises(hr.MissingHistory):
        api("https://baseballsavant.mlb.com/leaderboard/custom")


def test_report_labels_reconstruction_and_counts_exclusions():
    frame = pd.DataFrame([dict(game_pk=1, game_date="2026-03-28", away="NYY", home="BOS",
        xw_lean="BOS", xw_full="W", xw_f5="T", full_home=2, full_away=1,
        exclusion_reason=None), dict(game_pk=2, game_date="2026-03-28", away="NYY", home="BOS",
        xw_lean=None, xw_full=None, xw_f5=None, full_home=2, full_away=1,
        exclusion_reason="missing roster")])
    summary = hr.report_summary(frame)
    assert summary["scheduled_games"] == 2 and summary["predicted_games"] == 1
    assert summary["always_home_same_games"]["n"] == 1
    meta = dict(summary=summary, model_tag=bs.MODEL_TAG, start="2026-03-28", end="2026-03-28",
                untracked_bip=1, source_commit="abc")
    page = hr.render_report(frame, meta)
    assert "Retrospective reconstruction" in page and "not a live track record" in page
    assert "missing roster: 1" in page


def test_prediction_wrapper_shares_production_math_and_filters_calibration(monkeypatch):
    # Real production aggregation, blend, shrink, velocity and phase functions;
    # only the fetch result is replaced. Unequal lineups make sign errors visible.
    raw, pa, gs = history()
    g = game(3, "2026-03-30")
    slate = pd.DataFrame([dict(game_pk=3, game_date="2026-03-30",
        game_datetime_utc=g["gameDate"], matchup="NYY @ BOS", away_team="New York Yankees",
        home_team="Boston Red Sox", away_probable_pitcher="Pitcher 20",
        home_probable_pitcher="Pitcher 40", away_probable_pitcher_id=20,
        home_probable_pitcher_id=40, savant_preview_url="")])
    people = {i: dict(name=f"Player {i}", bats="L" if i % 2 else "R", throws="R", pos="DH")
              for i in list(range(1,10)) + list(range(30,39)) + [20,40]}
    pitchers = {20: {"xwOBA": .30, "wOBA_blend": .33, "PA": 150},
                40: {"xwOBA": .35, "wOBA_blend": .34, "PA": 170}}
    batters = {i: {"xwOBA": .29 if i < 10 else .36, "PA": 50+i} for i in people if i not in pitchers}
    pdf, _ = bs.build_tables(slate, {3: (list(range(1,10)), list(range(30,39)))},
                            batters, pitchers, {}, {}, people)
    pdf["velo_dv"] = pdf["player_id"].map({20: 1.2, 40: -1.0})
    league = {"xwOBA": .315, "wOBA_blend": .31}
    plans = {(3, side): dict(expected_sp_ip=5.5, expected_sp_ip_raw=5.6,
        bullpen_xwOBA=.31, bullpen_pitchers=3, sp_bf_per_ip=4.3, bp_bf_per_ip=4.1)
        for side in ("away", "home")}
    fake = dict(empty=False, pitchers_df=pdf, league_baseline=league, pitching_plans=plans)
    captured = {}
    def fetch(*args, **kwargs):
        captured.update(kwargs)
        return fake
    monkeypatch.setattr(bs, "fetch_all", fetch)
    provider = hr.HistoricalDataProvider("2026-03-30", 2026, raw, pa, gs, lambda *a, **k: {})
    cal = pd.DataFrame({"game_date": ["2025-03-01", "2026-03-29", "2026-03-30", "2026-03-31"]})
    actual, _ = hr.prediction_rows(provider, cal)
    expected, _, _ = bs.build_xwoba_matchup(pdf, league)
    expected = bs.apply_pitching_plans(expected, plans, league)
    pd.testing.assert_frame_equal(actual, expected)
    assert captured["calibration_history"]["game_date"].tolist() == ["2026-03-29"]
    assert captured["write_audit"] is False and captured["include_platoon"] is False
    assert expected["starter_rate_blended"].all()
    assert expected["mx_xwOBA"].notna().all()
    with pytest.raises(hr.MissingHistory):
        hr.prediction_rows(provider, None)


def test_savant_row_cap_splits_before_caching_a_truncated_response(tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "SAVANT_ROW_LIMIT", 3)
    requests = []
    class Response:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass
    def get(url, params, **kwargs):
        start, stop = params["game_date_gt"], params["game_date_lt"]
        requests.append((start, stop))
        if start != stop:
            raw = pitches().query("game_pk == 1").iloc[:3].copy()
            raw["game_date"] = start
        else:
            raw = pitches().query("game_pk == 1").iloc[:1].copy()
            raw["game_date"] = start
            raw["game_pk"] = int(start[-2:])
        return Response(raw.to_csv(index=False))
    monkeypatch.setattr(hr.requests, "get", get)
    raw = hr.fetch_pitches(tmp_path, 2026, "2026-03-02", workers=1, chunk_days=2)
    assert len(raw) == 2
    assert requests == [("2026-03-01", "2026-03-02"), ("2026-03-01", "2026-03-01"),
                        ("2026-03-02", "2026-03-02")]
    assert not list(tmp_path.rglob("2026-03-01_2026-03-02.csv.gz"))


def test_suspended_game_cannot_leak_through_mlb_role_totals():
    g = game()
    g['resumeDate'] = '2026-04-05T20:00:00Z'
    assert not hr.suspension_blocks({1:g}, '2026-03-28')
    assert hr.suspension_blocks({1:g}, '2026-03-29')
    assert hr.suspension_blocks({1:g}, '2026-04-05')
    assert not hr.suspension_blocks({1:g}, '2026-04-06')
