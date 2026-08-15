"""Offline guards for the paired shadow report.

Nothing here fetches, and nothing here asserts a record, a correlation or a
count -- those move with every slate and freezing one into a test is the
anti-pattern `test_record_reproduces_ledger_report` was fixed for. What is
pinned is the arithmetic that would be wrong silently: the sign convention of
the net, that a missing edge abstains rather than reading as zero, and that the
report classifies a post-first-pitch dump as post-hoc instead of assuming it
was pregame.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as bs
import grade_leans as gl
import shadow_metric as sm
import shadow_report as sr


def _frame(edge_away, edge_home, snap="2026-08-10T18:00:00+00:00",
           start="2026-08-10T23:07:00Z"):
    return pd.DataFrame([
        {"game_pk": 1, "side": "away", "matchup": "AAA @ BBB",
         "opp_team": "BBB", "edge_xwOBA": edge_away,
         "snapshot_utc": snap, "scheduled_start_utc": start},
        {"game_pk": 1, "side": "home", "matchup": "AAA @ BBB",
         "opp_team": "AAA", "edge_xwOBA": edge_home,
         "snapshot_utc": snap, "scheduled_start_utc": start},
    ])


def test_net_matches_grade_leans_construction():
    """The report must derive the net the way the ledger does, or it is
    comparing two different quantities and calling the difference a metric
    effect. Asserted against grade_leans' own function, not a copy of it."""
    df = _frame(0.020, -0.005)
    mine = sr.game_nets(df).loc[1, "net"]
    theirs = gl.rows_from_dump(df, None)[0]["xw_net"]
    assert mine == pytest.approx(theirs)
    assert mine == pytest.approx(0.025)  # away-row edge minus home-row edge


def test_missing_edge_abstains_rather_than_reading_as_zero():
    """A zero net is a v7 abstention; a NaN edge is a missing input. Collapsing
    the second into the first would publish a home lean for a game the arm
    could not price."""
    assert np.isnan(sr.game_nets(_frame(np.nan, -0.005)).loc[1, "net"])
    assert sr.game_nets(_frame(0.0, 0.0)).loc[1, "net"] == 0.0


def test_provenance_calls_a_rebuilt_dump_post_hoc():
    pre = sr._provenance(sr.game_nets(_frame(0.01, 0.0, snap="2026-08-10T22:00:00+00:00")))
    post = sr._provenance(sr.game_nets(_frame(0.01, 0.0, snap="2026-08-11T06:50:00+00:00")))
    assert pre[0] == "pregame"
    assert post[0] == "post-first-pitch"


def test_record_drops_ties_and_abstentions():
    net = pd.Series([0.01, -0.01, 0.01, np.nan, 0.0])
    margin = pd.Series([2.0, -3.0, -1.0, 5.0, 5.0])
    assert sr._record(net, margin) == (2, 1)


def test_report_refuses_to_pool_a_shadow_row(monkeypatch):
    """The runtime guard, exercised: if the shadow tag ever enters a family map
    the report must stop rather than quietly fold shadow rows into a record."""
    monkeypatch.setitem(bs._RECORD_FAMILIES, sm.SHADOW_TAG, (sm.SHADOW_TAG,))
    with pytest.raises(AssertionError):
        sr.report(n_boot=1)


# --------------------------------------------------------------------------
# which side is which metric (v11 swapped them)
# --------------------------------------------------------------------------
def _metric_frame(metric, edge_away, edge_home):
    f = _frame(edge_away, edge_home)
    f["model_metric"] = metric
    return f


def test_dump_metric_is_read_from_the_rows_not_the_filename():
    """`shadow_*` names a SIDE of the arm, not a statistic.

    Before v11 the shadow ran xwOBA; after it the shadow runs wOBA and the
    primary runs xwOBA. Keying off the filename or off the running build would
    label half the committed dumps with the wrong metric and silently flip the
    sign of d_corr on that half.
    """
    assert sr.dump_metric(_metric_frame("wOBA", .01, 0.0), "x.csv") == "wOBA"
    assert sr.dump_metric(_metric_frame("xwOBA", .01, 0.0), "x.csv") == "xwOBA"
    # A dump written before the column existed falls back to its suffix.
    bare = _frame(.01, 0.0)
    assert sr.dump_metric(bare, "shadow_2026-08-10_xw.csv") == "xwOBA"
    assert sr.dump_metric(bare, "leans_2026-08-10_woba.csv") == "wOBA"
    # A frame spanning both names no metric rather than guessing one.
    mixed = _metric_frame("wOBA", .01, 0.0)
    mixed.loc[1, "model_metric"] = "xwOBA"
    assert sr.dump_metric(mixed, "shadow_2026-08-10_woba.csv") is None


def test_net_columns_follow_the_metric_across_the_v11_swap(tmp_path, monkeypatch):
    """net_x is the xwOBA net on EVERY slate, whichever arm produced it.

    Built as two slates with the roles reversed -- the pre-v11 assignment and
    the post-v11 one -- with the same xwOBA net on both. If the report keyed
    off primary/shadow instead of off the metric, the second slate's nets would
    land in the opposite columns and d_corr would be measuring the changeover.
    """
    monkeypatch.setattr(sr, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sr, "LEDGER", str(tmp_path / "led.csv"))
    pd.DataFrame([{"game_pk": 1, "status": "graded", "xw_net": .01,
                   "model_metric": "xwOBA", "full_home": 4, "full_away": 2}]
                 ).to_csv(tmp_path / "led.csv", index=False)

    # 2026-08-10: shadow was xwOBA, primary was wOBA (pre-v11).
    _metric_frame("xwOBA", .02, 0.0).to_csv(
        tmp_path / "shadow_2026-08-10_xw.csv", index=False)
    _metric_frame("wOBA", .05, 0.0).to_csv(
        tmp_path / "leans_2026-08-10_woba.csv", index=False)
    # 2026-08-20: shadow is wOBA, primary is xwOBA (post-v11). Same values,
    # opposite sides.
    _metric_frame("wOBA", .05, 0.0).to_csv(
        tmp_path / "shadow_2026-08-20_woba.csv", index=False)
    _metric_frame("xwOBA", .02, 0.0).to_csv(
        tmp_path / "leans_2026-08-20_xw.csv", index=False)

    pairs = sr._slate_dates()
    assert [d for d, _, _ in pairs] == ["2026-08-10", "2026-08-20"]
    f, _ = sr.build_frame(pairs)
    # net = away-row edge - home-row edge, per game_nets.
    assert list(f.net_x.round(6)) == [.02, .02]
    assert list(f.net_w.round(6)) == [.05, .05]


def test_a_slate_whose_dumps_share_one_metric_is_skipped_not_pooled(
        tmp_path, monkeypatch):
    """Two xwOBA dumps are not a comparison. Skip loudly, never silently."""
    monkeypatch.setattr(sr, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sr, "LEDGER", str(tmp_path / "led.csv"))
    pd.DataFrame([{"game_pk": 1, "status": "graded", "xw_net": .01,
                   "model_metric": "xwOBA", "full_home": 4, "full_away": 2}]
                 ).to_csv(tmp_path / "led.csv", index=False)
    _metric_frame("xwOBA", .02, 0.0).to_csv(
        tmp_path / "shadow_2026-08-20_woba.csv", index=False)
    _metric_frame("xwOBA", .02, 0.0).to_csv(
        tmp_path / "leans_2026-08-20_xw.csv", index=False)
    with pytest.raises(SystemExit):
        sr.build_frame(sr._slate_dates())


def test_ledger_join_substitutes_into_the_column_matching_its_metric(
        tmp_path, monkeypatch):
    """`xw_net` is a legacy KEY name; pre-v11 rows hold wOBA, v11 rows xwOBA.

    Overwriting net_w unconditionally was right only while the ledger was
    wOBA. Under v11 it would compare an xwOBA ledger net against the wOBA arm
    and report the metric difference as contamination.
    """
    monkeypatch.setattr(sr, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sr, "LEDGER", str(tmp_path / "led.csv"))
    pd.DataFrame([{"game_pk": 1, "status": "graded", "xw_net": .033,
                   "model_metric": "xwOBA", "full_home": 4, "full_away": 2}]
                 ).to_csv(tmp_path / "led.csv", index=False)
    _metric_frame("wOBA", .05, 0.0).to_csv(
        tmp_path / "shadow_2026-08-20_woba.csv", index=False)
    _metric_frame("xwOBA", .02, 0.0).to_csv(
        tmp_path / "leans_2026-08-20_xw.csv", index=False)
    f, _ = sr.build_frame(sr._slate_dates(), use_ledger_net=True)
    assert f.net_x.iloc[0] == pytest.approx(.033)   # ledger is xwOBA
    assert f.net_w.iloc[0] == pytest.approx(.05)    # wOBA arm untouched
