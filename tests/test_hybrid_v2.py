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


def _linescore(game_pk, away_runs, home_runs):
    return {int(game_pk): {
        "status": {"detailedState": "Final"},
        "linescore": {
            "teams": {"away": {"runs": away_runs}, "home": {"runs": home_runs}},
            "innings": [{"away": {"runs": 0}, "home": {"runs": 0}}] * 5,
        },
    }}


def test_grading_a_pending_row_grades_its_v1_archive_too(monkeypatch):
    """A row pending when the v2 migration ran must not lose its v1 grade.

    migrate_hybrid_v2 writes hybrid_v1_full once, gated on the row already
    being graded, and nothing else wrote it -- so the 13 rows still pending on
    2026-09-11 kept a valid v1 selection and no grade, and dropped out of
    hybrid_test, abstain_test and dog_contrast_test alike. This pins the
    grader's counterpart, and pins that a row with NO archive stays ungraded
    rather than being handed a fabricated result.
    """
    led = _rows(.40, .020)
    led["status"] = "pending"
    led[["full_away", "full_home", "xw_full", "hybrid_full"]] = np.nan
    # The columns grade() touches beyond the hybrid pair.
    led[["f5_away", "f5_home", "xw_f5", "ops_full", "ops_f5"]] = np.nan
    led["ops_valid"] = False
    led["ops_lean"] = np.nan
    led["hybrid_v1_action"] = "FOLLOW"
    led["hybrid_v1_selection"] = "H"
    led["hybrid_v1_full"] = np.nan
    bare = led.copy()
    bare["game_pk"] = 2
    bare[["hybrid_v1_action", "hybrid_v1_selection"]] = np.nan
    led = pd.concat([led, bare], ignore_index=True)
    # grade_leans.load() forces the W/L/T columns to object dtype before
    # grading, because an all-NaN grade column reads back from CSV as float64
    # and pandas >=3 refuses a string assignment into it. hybrid_v1_full is
    # already in that list; this mirrors it so the test frame matches what
    # production hands grade(), rather than the grader learning to cast.
    for col in ("xw_full", "xw_f5", "ops_full", "ops_f5",
                "hybrid_full", "hybrid_v1_full"):
        led[col] = led[col].astype(object)

    monkeypatch.setattr(grade_leans, "_linescores_for",
                        lambda day: {**_linescore(1, 1, 4), **_linescore(2, 1, 4)})
    got = grade_leans.grade(led)

    assert got.at[0, "hybrid_v1_full"] == "W"      # H won, and H was archived
    assert got.at[0, "hybrid_full"] == "W"
    assert got.at[1, "hybrid_v1_full"] is None     # no archive, so no grade
    assert (got["status"] == "graded").all()


def test_a_migration_rerun_refreshes_its_archive_and_mints_no_new_one():
    """The archive describes the row, so a re-run re-derives it from the row.

    A pending row's lean and pregame price are rebuilt by every pregame poll
    (grade_leans.MODEL_FIELDS), which carries no hybrid_v1_* entry -- so an
    archive written mid-slate can name a side the final lock never chose. A
    re-run must fix that. It must equally NOT mint an archive for a row that
    has none: v1 is retired, and widening a frozen registration's row set is
    not a repair.
    """
    stale = _rows(.60, .020, date="2026-09-11")
    stale["hybrid_v1_action"] = "FOLLOW"
    stale["hybrid_v1_selection"] = "A"     # superseded: the lock reads H
    stale["hybrid_v1_p"] = .48
    stale["hybrid_v1_ml"] = 130
    stale["hybrid_v1_full"] = np.nan
    fresh = _rows(.60, .020, date="2026-09-13")
    fresh["game_pk"] = 2
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        fresh[col] = np.nan

    got, changed = migrate_hybrid_v2.migrate(
        pd.concat([stale, fresh], ignore_index=True))

    assert got.at[0, "hybrid_v1_selection"] == "H"   # re-derived from the lock
    assert got.at[0, "hybrid_v1_p"] == .60
    assert got.at[0, "hybrid_v1_full"] == "W"        # graded, no longer orphaned
    assert got.loc[1, migrate_hybrid_v2.V1_ARCHIVE].isna().all()
    assert changed == 1


def test_a_first_migration_still_mints_the_archive():
    """The no-minting rule must not disable the initial migration."""
    src = _rows(.60, .020, date="2026-09-11")
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        src[col] = np.nan
    got, _ = migrate_hybrid_v2.migrate(src)
    assert got.at[0, "hybrid_v1_selection"] == "H"
    assert got.at[0, "hybrid_v1_full"] == "W"
