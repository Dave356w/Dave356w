"""Guards on the dispersion pairing and the probe that reads it.

The failure this file exists for is not a crash. `opp_xwoba_sd_away` sits on the
away-PITCHER row and describes the HOME offense, so a reversed cross yields a
plausible near-zero correlation rather than an error -- and at the game level it
yields a real effect with the wrong sign. Both are asserted against constructed
frames where the right answer is known by construction.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import actuals_backfill as ab
import dispersion_probe as dp


def _row(**kw):
    base = dict(
        game_pk=1, model_tag="xw+plat_consol_v12", status="graded",
        mx_xwoba_away=0.300, mx_xwoba_home=0.320,
        act_woba_away=0.280, act_woba_home=0.340,
        act_pa_away=38, act_pa_home=39,
        opp_xwoba_sd_away=0.040, opp_xwoba_sd_home=0.010,
        lineup_savant_backfill_away=0, lineup_savant_backfill_home=2,
        xw_net=0.020, full_home=5, full_away=3,
    )
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# the cross
# --------------------------------------------------------------------------
def test_dispersion_takes_the_same_cross_as_the_predicted_rate():
    """`opp_xwoba_sd_away` is on the away-pitcher row, so it describes the HOME
    offense -- the identical cross `mx_xwoba_away` takes.

    Values are chosen so a reversed cross cannot coincidentally agree: the two
    sides' sd differ by 4x and their backfill counts differ too.
    """
    r = ab.paired_rates(pd.DataFrame([_row()]))
    home = r[r.batting_side == "home"].iloc[0]
    away = r[r.batting_side == "away"].iloc[0]

    # The home offense's row must carry the AWAY-suffixed dispersion...
    assert home["sd"] == pytest.approx(0.040)
    assert home["backfill"] == pytest.approx(0)
    # ...and pair it with the away-suffixed prediction and home actual.
    assert home["pred"] == pytest.approx(0.300)
    assert home["act"] == pytest.approx(0.340)

    assert away["sd"] == pytest.approx(0.010)
    assert away["backfill"] == pytest.approx(2)
    assert away["pred"] == pytest.approx(0.320)
    assert away["act"] == pytest.approx(0.280)


def test_sd_travels_with_pred_on_every_row():
    """Stated as an invariant rather than two literals: whatever cross `pred`
    takes, `sd` must take. This keeps holding if the cross is ever revisited."""
    rows = [_row(game_pk=i, mx_xwoba_away=0.30 + i / 1000,
                 opp_xwoba_sd_away=0.01 + i / 1000) for i in range(6)]
    r = ab.paired_rates(pd.DataFrame(rows))
    home = r[r.batting_side == "home"]
    # pred and sd were both written on the away row with the same offset.
    assert np.allclose(home["pred"] - 0.30, home["sd"] - 0.01)


def test_game_level_dispersion_gap_is_oriented_like_the_net():
    """`xw_net` is home-minus-away, so `d_sd` must be too.

    Reversed, this reports a real effect with the wrong sign -- the one failure
    mode worse than reporting nothing.
    """
    n = ab.paired_net(pd.DataFrame([_row()])).iloc[0]
    # home offense's dispersion (0.040, from the away row) minus away's (0.010)
    assert n["d_sd"] == pytest.approx(0.030)
    assert n["margin"] == pytest.approx(2.0)      # full_home - full_away
    assert n["act"] == pytest.approx(0.340 - 0.280)
    assert n["pred"] == pytest.approx(0.020)


def test_missing_dispersion_columns_do_not_break_the_pairing():
    """Every row graded before 2026-08-15 lacks these columns entirely, and
    they are the whole historical ledger."""
    row = _row()
    for c in ("opp_xwoba_sd_away", "opp_xwoba_sd_home",
              "lineup_savant_backfill_away", "lineup_savant_backfill_home"):
        row.pop(c)
    r = ab.paired_rates(pd.DataFrame([row]))
    assert len(r) == 2 and r["sd"].isna().all()
    n = ab.paired_net(pd.DataFrame([row]))
    assert len(n) == 1 and n["d_sd"].isna().all()


# --------------------------------------------------------------------------
# the probe's arithmetic
# --------------------------------------------------------------------------
def test_ols_recovers_a_planted_slope_and_calls_a_null_a_null():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.010, 0.045, 400)
    y = 0.9 * (x - x.mean()) + rng.normal(0, 0.08, 400)
    fit = dp.ols(x, y)
    assert abs(fit["slope"] / fit["se"]) > 2
    assert fit["slope"] - 1.96 * fit["se"] < 0.9 < fit["slope"] + 1.96 * fit["se"]
    null = dp.ols(x, rng.normal(0, 0.08, 400))
    assert abs(null["slope"] / null["se"]) < 2


def test_the_control_removes_a_planted_confounder():
    """H1's covariate has to actually do something, or the controlled line is
    decoration."""
    rng = np.random.default_rng(3)
    n = 600
    conf = rng.normal(0, 1, n)
    x = 0.02 + 0.004 * conf + rng.normal(0, 0.004, n)
    y = 0.05 * conf + rng.normal(0, 0.02, n)      # x has NO direct effect
    naive = dp.ols(x, y)
    controlled = dp.ols2(x, conf, y)
    assert abs(naive["slope"]) > abs(controlled["slope"])
    assert abs(controlled["slope"] / controlled["se"]) < 2


def test_power_target_is_stated_and_sane():
    n = dp.n_for_r(0.15)
    assert 300 < n < 400
    assert dp.n_for_r(0.5) < n          # bigger effects need fewer rows
    assert dp.ols([1, 2], [1, 2]) is None   # under three points, no fit


def test_an_empty_read_says_so_rather_than_going_quiet(tmp_path, monkeypatch):
    """A suppressed number invites recomputing it without the caveat."""
    led = pd.DataFrame([_row()])
    for c in ("opp_xwoba_sd_away", "opp_xwoba_sd_home"):
        led[c] = np.nan
    path = tmp_path / "led.csv"
    led.to_csv(path, index=False)
    monkeypatch.setattr(dp, "LEDGER", str(path))
    out = "\n".join(dp.run())
    assert "rows with a dispersion value AND an outcome: 0" in out
    assert "side-games" in out          # the expected-first-read estimate
