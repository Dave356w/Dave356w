"""The three Statcast-history studies: lookahead guards, joins, and recovery
of effects planted in a synthetic season.

Rules, not numbers: each study is run on a world where the true answer is
known (a starter K, a team defence, a velocity effect) and on one where the
effect is absent, so a study that "finds" its effect in both fails here.
Sizes are small -- the whole suite has a five-minute CI budget.
"""
import numpy as np
import pandas as pd
import pytest

from research import statcast_history as sh
from research import starter_shrink_history as ssh
from research import defense_history as dh
from research import velocity_trend as vt

L0 = 0.315


def world(seed=0, days=70, teams=10, tau_p=0.02, tau_d=0.0, park_sd=0.0,
          velo_effect=0.0, sigma=0.4):
    """A season of PAs. Starter talent sd `tau_p` sets the true starter K:
    K* = sigma^2 / tau_p^2 (400 at the defaults, against a shipped 100)."""
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    rot = {t: [1000 + 10 * i + j for j in range(5)] for i, t in enumerate(names)}
    talent_p = {p: rng.normal(0, tau_p) for ps in rot.values() for p in ps}
    base_v = {p: rng.normal(93, 2) for p in talent_p}
    state = {p: 0.0 for p in talent_p}
    bats = {t: [5000 + 20 * i + j for j in range(9)] for i, t in enumerate(names)}
    talent_b = {b: rng.normal(0, 0.03) for bs in bats.values() for b in bs}
    defence = {t: rng.normal(0, tau_d) for t in names}
    park = {t: rng.normal(0, park_sd) for t in names}
    starts = {t: 0 for t in names}
    pa, velo, games = [], [], []
    gpk = 100000
    for day in range(days):
        gd = (pd.Timestamp("2025-04-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        order = rng.permutation(names)
        for h, a in zip(order[::2], order[1::2]):
            gpk += 1
            sp = {}
            for t in (h, a):
                p = rot[t][starts[t] % 5]
                starts[t] += 1
                # velocity state: AR(1), persistent across a pitcher's starts
                state[p] = 0.7 * state[p] + rng.normal(0, 1.0)
                velo.append({"game_date": gd, "game_pk": gpk, "pitcher": p,
                             "velo": base_v[p] + state[p], "n_fb": 40})
                sp[t] = p
            ab = 0
            score = {h: 0.0, a: 0.0}
            for bat, fld in ((a, h), (h, a)):
                for i in range(38):
                    ab += 1
                    starter_up = i < 24
                    pit = sp[fld] if starter_up else 9000 + names.index(fld)
                    b = bats[bat][i % 9]
                    mu = L0 + talent_b[b] + (talent_p[pit] if starter_up else 0.0)
                    if starter_up:
                        mu += -velo_effect * state[pit]
                    x = mu + rng.normal(0, sigma)
                    bip = rng.random() < 0.7
                    w = x + (defence[fld] + park[h] + rng.normal(0, 0.3) if bip else 0.0)
                    score[bat] += w
                    pa.append({"game_date": gd, "game_pk": gpk, "at_bat_number": ab,
                               "inning": 1 + i // 4, "home_team": h, "away_team": a,
                               "fld_team": fld, "bat_team": bat, "batter": b,
                               "pitcher": pit, "bip": bip, "untracked_bip": False,
                               "woba_value": w, "xwoba_pa": x})
            games.append({"game_pk": gpk, "game_date": gd, "home_team": h,
                          "away_team": a, "home_score": score[h] + rng.normal(0, 1),
                          "away_score": score[a] + rng.normal(0, 1)})
    return {"pa": pd.DataFrame(pa), "velo": pd.DataFrame(velo),
            "games": pd.DataFrame(games)}


# ------------------------------------------------------------- data plumbing

def test_asof_never_sees_its_own_date():
    d = pd.DataFrame({"k": [1, 1, 1, 1], "v": [1.0, 10.0, 100.0, 1000.0],
                      "game_date": ["2025-04-01", "2025-04-02", "2025-04-02",
                                    "2025-04-03"]})
    a = sh.asof(d, "k", "v", "x").set_index("game_date")
    assert a.loc["2025-04-01", "x_num"] == 0
    assert a.loc["2025-04-02", "x_num"] == 1          # not the same-day 10 or 100
    assert a.loc["2025-04-03", "x_num"] == 111
    assert a.loc["2025-04-03", "x_n"] == 3


def test_asof_to_answers_on_dates_without_rows():
    rows = pd.DataFrame({"k": ["A", "A"], "v": [2.0, 4.0],
                         "game_date": ["2025-04-01", "2025-04-03"]})
    t = pd.DataFrame({"k": ["A", "A", "A", "A"],
                      "game_date": ["2025-04-01", "2025-04-02", "2025-04-03",
                                    "2025-04-04"]})
    a = sh.asof_to(rows, "k", "v", "x", t).set_index("game_date")
    assert list(a["x_num"]) == [0.0, 2.0, 2.0, 6.0]


def _raw(n=6):
    return pd.DataFrame({
        "game_date": ["2025-04-01"] * n, "game_pk": [1] * n,
        "game_type": ["R"] * (n - 1) + ["S"],
        "pitch_type": ["FF", "SL", "SI", "FF", "CH", "FF"],
        "release_speed": [95.0, 85.0, 93.0, 96.0, 86.0, 99.0],
        "batter": [10, 10, 11, 12, 13, 14], "pitcher": [7, 7, 7, 7, 8, 7],
        "events": [None, "single", "strikeout", "walk", "field_out", "single"],
        "home_team": ["H"] * n, "away_team": ["A"] * n,
        "bb_type": [None, "line_drive", None, None, "ground_ball", "fly_ball"],
        "inning": [1] * n, "inning_topbot": ["Top"] * 4 + ["Bot", "Top"],
        "at_bat_number": [1, 1, 2, 3, 4, 5],
        "estimated_woba_using_speedangle": [None, 0.8, None, None, None, 0.5],
        "woba_value": [None, 0.9, 0.0, 0.7, 0.0, 0.9],
        "woba_denom": [None, 1, 1, 1, 1, 1],
        "post_home_score": [0, 0, 0, 0, 1, 0], "post_away_score": [0] * n,
    })


def test_reduce_chunk_maps_sides_and_xwoba():
    pa, velo, games = sh.reduce_chunk(_raw())
    assert len(pa) == 4                                   # spring row dropped
    s = pa.set_index("at_bat_number")
    assert s.loc[1, "xwoba_pa"] == 0.8                    # contact: estimate
    assert s.loc[2, "xwoba_pa"] == 0.0                    # strikeout: actual
    assert s.loc[3, "xwoba_pa"] == 0.7                    # walk: actual
    assert s.loc[4, "xwoba_pa"] == 0.0 and s.loc[4, "untracked_bip"]
    assert s.loc[1, "fld_team"] == "H" and s.loc[4, "fld_team"] == "A"
    v = velo.set_index("pitcher").loc[7]
    assert v["n_fb"] == 3 and v["velo"] == pytest.approx((95 + 93 + 96) / 3)
    assert games.iloc[0]["home_score"] == 1


def test_reduce_chunk_refuses_a_missing_column():
    with pytest.raises(SystemExit, match="woba_denom"):
        sh.reduce_chunk(_raw().drop(columns="woba_denom"))


def test_starter_is_the_first_pitcher_even_when_an_opener():
    pa = pd.DataFrame({"game_pk": [1, 1, 1], "fld_team": ["H", "H", "H"],
                       "at_bat_number": [3, 1, 2], "pitcher": [30, 10, 20]})
    assert sh.starters(pa).iloc[0]["starter"] == 10


def test_shipped_k_matches_the_build():
    import build_site
    assert sh.SHIPPED_K == build_site.XWOBA_SHRINK_K


# ----------------------------------------------------------- starter shrink

def test_undershrunk_starters_prefer_a_larger_k():
    rows = ssh.build_rows(world(seed=1, tau_p=0.02)["pa"])   # K* = 400
    s = ssh.slope_fit(rows, "y_x", n_boot=20)
    assert s["beta_p"] < 0.85
    grid = {g["k"]: g for g in ssh.error_grid(rows, "y_x", n_boot=20)}
    assert grid[400]["d"] < 0 and grid[400]["d"] < grid[50]["d"]


def test_well_shrunk_starters_do_not_prefer_a_larger_k():
    rows = ssh.build_rows(world(seed=2, tau_p=0.04)["pa"])   # K* = 100
    grid = {g["k"]: g for g in ssh.error_grid(rows, "y_x", n_boot=20)}
    assert grid[1200]["d"] > 0                             # over-shrinking hurts


def test_starter_rows_are_starters_only_and_pregame():
    w = world(seed=3, days=20)
    rows = ssh.build_rows(w["pa"])
    assert (rows["pitcher"] < 9000).all()                  # no bullpen arms
    first = rows.sort_values("game_date").groupby("pitcher").head(1)
    # a starter's first appearance has no earlier sample and is excluded
    assert (rows["p_n"] > 0).all()
    assert len(first) > 0


# ---------------------------------------------------------------- defence

def test_defence_signal_is_recovered_through_park_noise():
    w = world(seed=4, tau_d=0.03, park_sd=0.03, days=80)
    g = dh.game_frame(w["pa"], k=dh.variance_components(w["pa"])["k"])
    r = dh.same_game_slope(g, n_boot=20)
    assert r["slope"] > 0.5


def test_parks_alone_do_not_make_a_defence_signal():
    w = world(seed=5, tau_d=0.0, park_sd=0.05, days=80)
    g = dh.game_frame(w["pa"], k=400.0)
    r = dh.same_game_slope(g, n_boot=20)
    assert abs(r["slope"]) < 3 * r["se"] + 0.1


def test_variance_components_find_real_spread():
    real = dh.variance_components(world(seed=6, tau_d=0.03, days=60)["pa"])
    none = dh.variance_components(world(seed=6, tau_d=0.0, days=60)["pa"])
    assert real["tau2"] > none["tau2"]
    assert real["k"] < none["k"]


# --------------------------------------------------------------- velocity

def test_velocity_change_persists_and_predicts_when_real():
    w = world(seed=7, velo_effect=0.02, days=80)
    sv = vt.start_velocity(w["pa"], w["velo"])
    p = vt.persistence(sv, n_boot=20)
    assert 0.3 < p["slope"] < 0.95                         # AR(1) at 0.7
    _, rows = vt.season_rows(w)
    r = vt.results_fit(rows, "y_x", n_boot=20)
    assert r["beta_v"] < 0


def test_velocity_dv_uses_only_earlier_starts():
    w = world(seed=8, days=40)
    sv = vt.start_velocity(w["pa"], w["velo"])
    v = w["velo"].sort_values(["pitcher", "game_date"])
    one = sv.iloc[0]
    hist = v[(v.pitcher == one.pitcher) & (v.game_date < one.game_date)]
    prev = hist.iloc[-1]
    base = hist.iloc[:-1]
    want = prev.velo - np.average(base.velo, weights=base.n_fb)
    assert one.dv == pytest.approx(want)


# -------------------------------------------------- retrospective ledger arm

def _ledger():
    import build_site
    led = pd.read_csv("data/mlb_lean_ledger.csv", low_memory=False)
    return led[led["model_tag"].isin(build_site.RECORD_TAGS)
               & led["model_metric"].eq("xwOBA")].reset_index(drop=True)


def test_ledger_rebuild_reproduces_xw_net():
    led = _ledger()
    ship = pd.to_numeric(led["xw_net"], errors="coerce")
    base = vt.ledger_net(led)
    ok = ship.notna() & base.notna()
    assert ok.sum() > 100
    assert float((base - ship)[ok].abs().max()) < 1e-9


def test_worse_away_starter_raises_the_home_side():
    led = _ledger()
    base = vt.ledger_net(led)
    worse = vt.ledger_net(led, d_away=+0.010)      # away starter allows more
    ok = base.notna()
    assert (worse[ok] > base[ok]).all()


def test_velocity_arm_is_inert_without_matching_velocity():
    w = world(seed=9, days=30)                      # game_pks match no ledger row
    lines = "\n".join(vt.ledger_shadow(-0.006, w["pa"], w["velo"], season=2026))
    assert "reconstruction matches xw_net" in lines
    vel = next(l for l in lines.splitlines() if l.strip().startswith("velocity"))
    assert vel.split()[1] == "0"                   # no dv -> no flips


def test_ledger_arm_is_scoped_to_its_season():
    w = world(seed=10, days=10)
    lines = "\n".join(vt.ledger_shadow(-0.006, w["pa"], w["velo"], season=1999))
    assert "no current-family ledger rows in 1999" in lines
