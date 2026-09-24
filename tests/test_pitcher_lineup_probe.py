"""The pitcher x lineup probe's joins and its fit.

Rules, not numbers: every fixture is synthetic with a KNOWN generating
interaction, so a crossed side or a mis-centred regressor fails a test rather
than printing a plausible weak coefficient.
"""
import numpy as np
import pandas as pd
import pytest

import pitcher_lineup_probe as plp

TAG = "tag"
L = 0.315


def _world(n_games=120, beta_bp=0.0, seed=0, noise=0.05, scratch=None):
    """Hitter frames, a ledger and per-PA rows from a known model."""
    rng = np.random.default_rng(seed)
    frames, led, pas = [], [], []
    hitters = np.arange(1000, 1060)
    hit_rate = dict(zip(hitters, L + rng.normal(0, 0.03, len(hitters))))
    for g in range(n_games):
        gp = 500000 + g
        P = {"away": L + rng.normal(0, 0.03), "home": L + rng.normal(0, 0.03)}
        sp = {"away": f"Away Arm {g % 25}", "home": f"Home Arm {g % 25}"}
        led.append({
            "game_pk": gp, "model_tag": TAG, "model_metric": "xwOBA",
            "away_sp": sp["away"], "home_sp": sp["home"],
            "starter_xwoba_away": P["away"], "starter_xwoba_home": P["home"],
            "mx_xwoba_sp_away": L + 0.01, "edge_xwoba_sp_away": 0.01,
            "scheduled_start_utc": "2026-09-06T23:00:00Z",
        })
        for bat in ("away", "home"):
            pit = "home" if bat == "away" else "away"
            # Disjoint pools per side: one player never bats for both teams.
            pool = hitters[:30] if bat == "away" else hitters[30:]
            nine = rng.choice(pool, 9, replace=False)
            for i, pid in enumerate(nine):
                frames.append({
                    "game_pk": gp, "faced_pitcher": sp[pit],
                    "pitcher_side": pit, "batting_side": bat,
                    "player_id": pid, "batting_order": i + 1, "PA": 400,
                    "xwoba_raw": hit_rate[pid], "xwoba_shrunk": hit_rate[pid],
                    "slot_weight": 4.0, "savant_backfill": False,
                    "model_tag": TAG, "model_metric": "xwOBA",
                    "snapshot_utc": "2026-09-06T20:00:00Z",
                    "scheduled_start_utc": "2026-09-06T23:00:00Z",
                    "lock_status": "pregame",
                })
                b, p = hit_rate[pid] - L, P[pit] - L
                mu = L + b + p + beta_bp * b * p
                thrower = sp[pit]
                if scratch is not None and (gp, bat) in scratch:
                    thrower = "Someone Else"
                for _ in range(3):
                    pas.append({
                        "game_pk": gp, "bat_side": bat, "pit_side": pit,
                        "batter_id": pid, "pitcher_name": thrower,
                        "vs_starter": True, "reconciled": True,
                        "cat": "1b", "_mu": mu + rng.normal(0, noise),
                    })
    pa = pd.DataFrame(pas)
    return pd.DataFrame(frames), pd.DataFrame(led), pa


def _rows(frames, led, pa, tmp_path):
    path = tmp_path / "ledger.csv"
    led.to_csv(path, index=False)
    m, c = plp.build_rows(frames, pa, ledger=str(path), tags=(TAG,))
    return m, c, str(path)


def _with_outcome(m):
    """Swap the categorical wOBA value for the continuous generated one, so
    the fit can be checked against its generating coefficients."""
    m = m.copy()
    m["y"] = m["_mu"] - m["L"]
    return m


@pytest.mark.parametrize("truth", [0.0, 1.0 / L])
def test_fit_recovers_the_generating_interaction(tmp_path, truth):
    frames, led, pa = _world(beta_bp=truth, noise=0.0005)
    m, c, _ = _rows(frames, led, pa, tmp_path)
    est = plp.fit(_with_outcome(m))
    assert est["beta_b"] == pytest.approx(1.0, abs=0.05)
    assert est["beta_p"] == pytest.approx(1.0, abs=0.05)
    assert est["beta_bp"] == pytest.approx(truth, abs=0.6)


def test_pitcher_side_pairs_with_the_ledger_starter(tmp_path):
    """`pitcher_side` is the pitcher's team: P must be that side's starter.
    Crossing it would pair every lineup with its own team's starter."""
    frames, led, pa = _world(n_games=10)
    m, c, _ = _rows(frames, led, pa, tmp_path)
    assert c["frame_sp_mismatch"] == 0
    for _, r in m.drop_duplicates(["game_pk", "batting_side"]).iterrows():
        lr = led[led.game_pk == r.game_pk].iloc[0]
        assert r["P"] == pytest.approx(lr[f"starter_xwoba_{r['pitcher_side']}"])
        assert r["pitcher_side"] != r["batting_side"]


def test_frame_starter_not_in_ledger_is_dropped(tmp_path):
    frames, led, pa = _world(n_games=10)
    frames.loc[frames.game_pk == 500000, "faced_pitcher"] = "Wrong Guy"
    m, c, _ = _rows(frames, led, pa, tmp_path)
    assert c["frame_sp_mismatch"] == 18
    assert 500000 not in set(m["game_pk"])


def test_scratched_starter_drops_the_whole_lineup(tmp_path):
    frames, led, pa = _world(n_games=10, scratch={(500003, "home")})
    m, c, _ = _rows(frames, led, pa, tmp_path)
    assert c["identity"] == "verified"
    assert c["starter_changed"] == 1
    keep = m[(m.game_pk == 500003)]
    assert set(keep["batting_side"]) == {"away"}


def test_missing_pitcher_name_is_reported_unverified(tmp_path):
    frames, led, pa = _world(n_games=10)
    m, c, _ = _rows(frames, led, pa.drop(columns="pitcher_name"), tmp_path)
    assert c["identity"] == "unverified"
    assert not m.empty


def test_accents_do_not_read_as_a_scratch():
    a = pd.Series(["José Ramírez", "A.J. Puk"])
    b = pd.Series(["Jose Ramirez", "AJ Puk"])
    assert (plp._norm_name(a) == plp._norm_name(b)).all()


def test_bullpen_and_post_hoc_rows_are_not_scored(tmp_path):
    frames, led, pa = _world(n_games=10)
    pa.loc[pa.game_pk == 500001, "vs_starter"] = False
    frames.loc[frames.game_pk == 500002, "snapshot_utc"] = "2026-09-07T01:00:00Z"
    m, c, _ = _rows(frames, led, pa, tmp_path)
    assert not {500001, 500002} & set(m["game_pk"])
    assert c["post_hoc"] == 18


def test_other_record_family_is_excluded(tmp_path):
    frames, led, pa = _world(n_games=10)
    frames["model_tag"] = "old_family"
    m, _, _ = _rows(frames, led, pa, tmp_path)
    assert m.empty


def test_regressors_are_centred_on_the_ledger_league(tmp_path):
    frames, led, pa = _world(n_games=5)
    m, _, _ = _rows(frames, led, pa, tmp_path)
    assert np.allclose(m["L"], L)
    assert np.allclose(m["b"], m["xwoba_shrunk"] - L)
    assert np.allclose(m["bp"], m["b"] * m["p"])


def test_report_runs_and_states_empty_sample(tmp_path):
    frames, led, pa = _world(n_games=40)
    path = tmp_path / "ledger.csv"
    led.to_csv(path, index=False)
    out = "\n".join(plp.report(frames, pa, ledger=str(path), tags=(TAG,),
                               n_boot=30))
    assert "beta_bp" in out and "paired MSE" in out
    empty = "\n".join(plp.report(frames, pa.iloc[0:0], ledger=str(path),
                                 tags=(TAG,)))
    assert "not a result" in empty
