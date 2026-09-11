import datetime as dt

import numpy as np
import pandas as pd

import hybrid_test
import hybrid_v2
import grade_leans
import migrate_hybrid_v2


def _rows(q, delta, won=True, date="2026-09-12", source="saved_pregame"):
    home_ml, away_ml = (-150, 130) if q >= .5 else (130, -150)
    row = dict(
        game_pk=1, game_date=date, away="A", home="H", status="graded",
        model_tag="xw+plat_consol_v12", xw_lean="H", xw_net=delta,
        full_away=1 if won else 4, full_home=4 if won else 1,
        xw_full="W" if won else "L", close_p_home=q,
        close_home_ml=home_ml, close_away_ml=away_ml,
        pregame_p_home=q, pregame_home_ml=home_ml, pregame_away_ml=away_ml,
        selection_rule_tag=hybrid_v2.RULE_TAG,
        hybrid_action="FADE" if q < .45 and abs(delta) < .012 else "FOLLOW",
        hybrid_selection="A" if q < .45 and abs(delta) < .012 else "H",
        hybrid_p=1-q if q < .45 and abs(delta) < .012 else q,
        hybrid_ml=away_ml if q < .45 and abs(delta) < .012 else home_ml,
        hybrid_full="L" if q < .45 and abs(delta) < .012 and won else "W",
        hybrid_price_source=source,
    )
    return pd.DataFrame([row])


def test_v1_registration_remains_frozen_and_v2_is_distinct():
    assert hybrid_test.RULE_TAG == "xwoba_market_hybrid_v1"
    assert hybrid_test.REGISTERED_ON == "2026-09-01"
    assert hybrid_v2.RULE_TAG == "xwoba_market_hybrid_v2"
    assert hybrid_v2.REGISTERED_ON == "2026-09-11"
    assert hybrid_v2.REGISTERED_ON > hybrid_test.REGISTERED_ON


def test_v2_truth_table_and_closed_boundaries():
    assert not bool(hybrid_v2.follows(.449999, .011999))
    assert bool(hybrid_v2.follows(.45, .011999))
    assert bool(hybrid_v2.follows(.449999, .012))
    assert bool(hybrid_v2.follows(.80, .001))


def test_apply_rule_changes_only_the_weak_low_price_cell():
    frames = [_rows(.40, .005), _rows(.40, .020), _rows(.60, .005)]
    d = pd.concat(frames, ignore_index=True)
    d["game_pk"] = [1, 2, 3]
    got = hybrid_v2.apply_rule(hybrid_v2.decidable(d))
    assert got["follow"].tolist() == [False, True, True]


def test_forward_window_is_strict_and_has_no_close_fallback():
    on = _rows(.40, .005, date=hybrid_v2.REGISTERED_ON)
    after = _rows(.40, .005, date=str(
        dt.date.fromisoformat(hybrid_v2.REGISTERED_ON) + dt.timedelta(days=1)))
    assert len(hybrid_v2.scored_rows(on)) == 0
    assert len(hybrid_v2.scored_rows(after)) == 1
    after["pregame_p_home"] = np.nan
    assert len(hybrid_v2.scored_rows(after)) == 0
    assert hybrid_v2.unscorable(after) == 1


def test_migration_preserves_every_nonhybrid_value_and_price_basis():
    pre = _rows(.40, .020)
    legacy = _rows(.40, .005, source="closing")
    legacy["game_pk"] = 2
    legacy[["pregame_p_home", "pregame_home_ml", "pregame_away_ml"]] = np.nan
    source = pd.concat([pre, legacy], ignore_index=True)
    protected = [c for c in source.columns if c not in {
        "selection_rule_tag", "hybrid_action", "hybrid_selection", "hybrid_p",
        "hybrid_ml", "hybrid_full", "hybrid_price_source"}]
    got, changed = migrate_hybrid_v2.migrate(source)
    pd.testing.assert_frame_equal(source[protected], got[protected])
    assert changed == 1  # only the captured row gains a v1 archive decision
    assert got["hybrid_price_source"].tolist() == ["saved_pregame", "closing"]
    assert got["hybrid_action"].tolist() == ["FOLLOW", "FADE"]


def test_materialized_report_identifies_each_rows_price_basis():
    close = _rows(.40, .005, source="closing")
    saved = _rows(.40, .020, source="saved_pregame")
    saved["game_pk"] = 2
    lines = grade_leans._hybrid_materialized_lines(
        pd.concat([close, saved], ignore_index=True))
    text = "\n".join(lines)
    assert "row basis" in text
    assert "1 closing, 1 saved pregame" in text
    assert "no cross-snapshot fallback" in text
