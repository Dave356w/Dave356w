"""v14 starter velocity term: the rule, the live adjustment, the retroactive
re-decision and what the site publishes from it."""
import numpy as np
import pandas as pd
import pytest

import build_site as bs
import market_backfill as mb
import reconstruct_v14_velocity as rv
import starter_velocity as sv


def _starts(velos, n_fb=40, start="2026-04-01"):
    d = pd.Timestamp(start)
    return pd.DataFrame({
        "game_date": [(d + pd.Timedelta(days=5 * i)).strftime("%Y-%m-%d")
                      for i in range(len(velos))],
        "game_pk": range(1, len(velos) + 1),
        "velo": velos, "n_fb": n_fb})


# ------------------------------------------------------------------ the rule

def test_dv_is_last_start_minus_earlier_starts():
    t = sv.pregame_trend(_starts([95.0, 95.0, 93.5]), "2026-05-01")
    assert t["dv"] == pytest.approx(-1.5)
    assert t["velo_last"] == 93.5 and t["velo_base"] == 95.0


def test_dv_never_reads_the_game_day_or_later():
    s = _starts([95.0, 95.0, 95.0, 90.0])          # 4th start on 2026-04-16
    t = sv.pregame_trend(s, "2026-04-16")
    assert t["dv"] == pytest.approx(0.0)           # the 90.0 is not visible
    assert t["n_starts"] == 3


def test_too_few_starts_gives_no_trend_and_no_adjustment():
    t = sv.pregame_trend(_starts([95.0, 94.0]), "2026-05-01")
    assert t["dv"] is None and t["velo_last"] == 94.0
    assert sv.adjust(0.310, t["dv"]) == 0.310


def test_thin_last_start_gives_no_trend():
    s = _starts([95.0, 95.0, 92.0])
    s.loc[2, "n_fb"] = sv.MIN_FB - 1
    assert sv.pregame_trend(s, "2026-05-01")["dv"] is None


def test_velocity_up_lowers_the_rate_allowed():
    assert sv.adjust(0.310, +1.0) == pytest.approx(0.310 + sv.BETA_V)
    assert sv.adjust(0.310, +1.0) < 0.310 < sv.adjust(0.310, -1.0)


def test_per_start_keeps_starts_and_fastballs_only():
    raw = pd.DataFrame({
        "game_date": ["2026-04-01"] * 3 + ["2026-04-03"] * 2,
        "game_pk": [1, 1, 1, 2, 2], "game_type": ["R"] * 5,
        "inning": [1, 1, 2, 6, 6],                     # game 2 is relief
        "pitch_type": ["FF", "SL", "SI", "FF", "FF"],
        "release_speed": [96.0, 86.0, 94.0, 97.0, 97.0]})
    s = sv.per_start(raw)
    assert list(s["game_pk"]) == [1]
    assert s.iloc[0]["velo"] == pytest.approx(95.0) and s.iloc[0]["n_fb"] == 2


def test_research_study_uses_the_same_rule():
    """`research/velocity_trend` and the shipped rule must agree on dv."""
    import sys
    sys.path.insert(0, "tests")
    from test_history_studies import world
    from research import velocity_trend as vt
    w = world(seed=21, days=40)
    study = vt.start_velocity(w["pa"], w["velo"]).set_index(["pitcher", "game_pk"])["dv"]
    live = rv.starter_dv(w["pa"], w["velo"])
    from research import statcast_history as sh
    st = sh.starters(w["pa"]).merge(
        w["pa"][["game_pk", "home_team"]].drop_duplicates(), on="game_pk")
    st["side"] = np.where(st["fld_team"] == st["home_team"], "home", "away")
    checked = 0
    for r in st.itertuples():
        key = (r.starter, r.game_pk)
        if key in study.index:
            assert live[(r.game_pk, r.side)] == pytest.approx(study[key])
            checked += 1
    assert checked > 50


# -------------------------------------------------------- the live adjustment

def _matchup(dv):
    col = bs.XWOBA_SHRINK_COL
    P = pd.DataFrame([{
        "game_pk": 1, "Name": "Arm", "game_date": "2026-09-25",
        "game_datetime_utc": "2026-09-25T23:05:00Z", "matchup": "AAA @ BBB",
        "away_team": "AAA", "home_team": "BBB", "player_id": 99,
        "PA": 400, col: 0.300, "velo_dv": dv, "velo_last": 94.0,
        "velo_base": 94.0 - (dv or 0.0), "velo_starts": 10}])
    agg = pd.DataFrame([{
        "game_pk": 1, "faced_pitcher": "Arm", "pitcher_side": "away",
        "n_opp_hitters": 9, f"opp_{col}": 0.320, "opp_xwOBA_neutral": 0.318,
        "opp_xwOBA_vs_sp": 0.320, "platoon_delta_sp": 0.002}])
    lg = {"xwOBA": 0.3163}
    return bs.build_matchup(P, agg, [col], lg, shrink_prior=lg[col],
                            shrink_k=bs.XWOBA_SHRINK_K).iloc[0]


def test_build_matchup_applies_the_velocity_term():
    flat, down = _matchup(None), _matchup(-2.0)
    assert down["starter_rate_prevelo"] == pytest.approx(flat["starter_xwOBA"])
    assert down["starter_xwOBA"] == pytest.approx(
        flat["starter_xwOBA"] - 2.0 * sv.BETA_V)
    assert down["starter_velo_dv"] == -2.0
    assert down["mx_xwOBA_sp"] > flat["mx_xwOBA_sp"]   # hitters benefit


def test_card_shows_velocity_and_direction():
    down = bs._velo_cell({"velo_last": 93.4, "velo_dv": -1.2})
    up = bs._velo_cell({"velo_last": 96.0, "velo_dv": 0.7})
    assert "93.4" in down and "▼" in down and "-1.2" in down and "warm" in down
    assert "▲" in up and "+0.7" in up and "cool" in up
    assert "no trend yet" in bs._velo_cell({"velo_last": 95.0, "velo_dv": None})


def test_v14_shares_the_v12_v13_record_line():
    assert bs.MODEL_TAG == "xw+starter_velo_v14"
    assert set(bs.RECORD_TAGS) == {"xw+plat_consol_v12", "xw+starter_blend_v13",
                                   "xw+starter_velo_v14"}
    import grade_leans as gl
    assert gl.MODEL_TAG == bs.MODEL_TAG
    assert gl._RECORD_FAMILIES[bs.MODEL_TAG] == bs._RECORD_FAMILIES[bs.MODEL_TAG]


# --------------------------------------------------- retroactive re-decision

def _ledger():
    led = pd.read_csv(bs.LEDGER_PATH, float_precision="round_trip",
                      low_memory=False)
    return led


def test_no_velocity_data_changes_no_lean_and_no_protected_column():
    led = _ledger()
    out, n = rv.reconstruct(led, {})
    assert n > 400 and rv._protected_unchanged(led, out) == []
    re = out[out["velo_recon_basis"].notna()]
    base = re["xw_lean"].where(re["model_tag"].eq(rv.V13_TAG), re["v13_lean_recon"])
    assert (re["velo_lean_recon"] == base).all()


def test_velocity_delta_matches_a_full_rebuild_of_the_net():
    """net_delta is exact: the same shift a rebuild with a moved P gives."""
    from research import velocity_trend as vt
    led = _ledger()
    v13 = led[led["model_tag"].eq(rv.V13_TAG) & led["xw_lean"].notna()].reset_index(drop=True)
    # Keyed by game, as the migration keys it (a game can hold two rows).
    odd = v13["game_pk"].astype(int) % 2 == 1
    dv = {}
    for g, o in zip(v13["game_pk"].astype(int), odd):
        dv[(g, "home")] = 1.0 if o else None
        dv[(g, "away")] = -1.5
    out, _ = rv.reconstruct(v13, dv)
    dh = np.where(odd, sv.BETA_V * 1.0, 0.0)
    rebuilt = vt.ledger_net(v13, d_home=dh, d_away=sv.BETA_V * -1.5)
    ok = out["velo_net_recon"].notna()
    assert ok.sum() > 50
    assert np.allclose(out.loc[ok, "velo_net_recon"], rebuilt[ok], atol=1e-12)


def test_abstained_rows_are_never_re_decided():
    led = _ledger()
    out, _ = rv.reconstruct(led, {})
    assert out.loc[led["xw_lean"].isna(), "velo_recon_basis"].isna().all()


def test_write_columns_keeps_every_other_byte(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("a,b\n1.10,x\n2.000000000000001,y\n")
    f = pd.DataFrame({"velo_net_recon": [0.5, np.nan]})
    rv.write_columns(str(p), f, ["velo_net_recon"])
    assert p.read_text() == ("a,b,velo_net_recon\n1.10,x,0.5\n"
                             "2.000000000000001,y,\n")
    f2 = pd.DataFrame({"velo_net_recon": [0.25, 0.75]})
    rv.write_columns(str(p), f2, ["velo_net_recon"])        # re-run
    assert p.read_text() == ("a,b,velo_net_recon\n1.10,x,0.25\n"
                             "2.000000000000001,y,0.75\n")


# ------------------------------------------------------------ what publishes

def _g(**over):
    base = dict(model_tag="xw+starter_blend_v13", home="HHH", away="AAA",
                full_home=5, full_away=3, xw_lean="AAA", xw_full="L",
                xw_net=-0.01, xw_delta=0.01, status="graded",
                v13_lean_recon=np.nan, v13_net_recon=np.nan,
                v13_recon_basis=np.nan, velo_lean_recon=np.nan,
                velo_net_recon=np.nan, velo_recon_basis=np.nan)
    base.update(over)
    return pd.DataFrame([base])


def test_publish_uses_the_velocity_re_decision():
    g = _g(velo_lean_recon="HHH", velo_net_recon=0.002,
           velo_recon_basis="v13_native+velocity")
    p = mb.publish_reconstruction(g, "xw+starter_velo_v14")
    assert p.iloc[0]["xw_lean"] == "HHH" and p.iloc[0]["xw_full"] == "W"


def test_publish_passes_a_v13_row_through_before_the_migration():
    p = mb.publish_reconstruction(_g(), "xw+starter_velo_v14")
    assert len(p) == 1 and p.iloc[0]["xw_lean"] == "AAA"


def test_publish_falls_back_to_v13_recon_for_v12_rows():
    g = _g(model_tag="xw+plat_consol_v12", v13_lean_recon="HHH",
           v13_net_recon=0.01, v13_recon_basis="post_hoc_shadow_pair")
    p = mb.publish_reconstruction(g, "xw+starter_velo_v14")
    assert p.iloc[0]["xw_lean"] == "HHH"
    assert len(mb.publish_reconstruction(_g(model_tag="xw+plat_consol_v12"),
                                         "xw+starter_velo_v14")) == 0


def test_publish_leaves_current_rows_untouched():
    g = _g(model_tag="xw+starter_velo_v14", velo_lean_recon="HHH",
           velo_net_recon=0.5, velo_recon_basis="x")
    p = mb.publish_reconstruction(g, "xw+starter_velo_v14")
    assert p.iloc[0]["xw_lean"] == "AAA"
