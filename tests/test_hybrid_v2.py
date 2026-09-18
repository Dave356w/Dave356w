import datetime as dt

import numpy as np
import pandas as pd
import pytest

import hybrid_test
import hybrid_v2
import grade_leans
import migrate_hybrid_v2


def _rows(q, delta, won=True, date="2026-09-12", source="saved_pregame"):
    home_ml, away_ml = (-150, 130) if q >= .5 else (130, -150)
    row = dict(
        game_pk=1, game_date=date, away="A", home="H", status="graded",
        model_tag=grade_leans.MODEL_TAG, xw_lean="H", xw_net=delta,
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


def test_a_migration_rerun_refreshes_its_archive_and_mints_the_missing_ones():
    """The archive describes the row, so a re-run re-derives it from the row.

    A pending row's lean and pregame price are rebuilt by every pregame poll
    (grade_leans.MODEL_FIELDS), which carries no hybrid_v1_* entry -- so an
    archive written mid-slate can name a side the final lock never chose. A
    re-run must fix that.

    THE SECOND HALF IS THE REVERSE OF WHAT IT ASSERTED UNTIL 2026-09-17, and
    the reversal is deliberate rather than a loosened assertion. It used to
    pin that a row with NO archive is left alone, on the ground that v1 is a
    retired registration and widening it is not a repair. True of v1's own
    headline and false of everything downstream: the archive is also the row
    SELECTOR that `abstain_test` and `dog_contrast_test` delegate to, and both
    are live registrations with open gates, so leaving the row alone dropped
    it out of two tests that are not retired. Both froze at 2026-09-11 with no
    surface saying so. Minting is a pure re-derivation from write-once locked
    columns -- `hybrid_test.locked_v1_decision` -- not new evidence.
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
    assert got.at[1, "hybrid_v1_selection"] == "H"   # minted, not skipped
    assert got.at[1, "hybrid_v1_full"] == "W"
    assert changed == 2


def test_a_migration_never_overwrites_an_archive_it_can_reproduce():
    """Minting must not become mutation of immutable history.

    The v1 decision is a function of `xw_lean` and the locked pregame market,
    all write-once once a row grades, so a re-derivation of a row that already
    carries an archive has to return that archive unchanged. Asserted rather
    than assumed, because minting-where-absent and rewriting-what-is-there are
    one line apart: on the real ledger this was checked over all 141 archives
    minted live and reproduced every field on every one.
    """
    src = _rows(.60, .020, date="2026-09-11")
    first, _ = migrate_hybrid_v2.migrate(src)
    second, changed = migrate_hybrid_v2.migrate(first)
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        assert str(second.at[0, col]) == str(first.at[0, col])
    assert changed == 0


def test_a_first_migration_still_mints_the_archive():
    """The no-minting rule must not disable the initial migration."""
    src = _rows(.60, .020, date="2026-09-11")
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        src[col] = np.nan
    got, _ = migrate_hybrid_v2.migrate(src)
    assert got.at[0, "hybrid_v1_selection"] == "H"
    assert got.at[0, "hybrid_v1_full"] == "W"


def _pair(idx, q, delta, won, date, tag=None):
    tag = grade_leans.MODEL_TAG if tag is None else tag
    r = _rows(q, delta, won=won, date=date)
    r["game_pk"] = idx
    r["model_tag"] = tag
    return r


def test_the_registered_headline_is_the_switch_delta_not_the_combined_line():
    """v1's point 3 applies unchanged to v2: every FOLLOWED row is the model
    untouched, so a combined ROI can only restate what the model already does.

    v2 shipped without this. `apply_rule` computed `switch_delta` all along and
    the forward block printed `combined hybrid` as its most prominent number
    while v1, the RETIRED registration, printed the headline and the gate. The
    assertion is on ORDER, not wording: the headline must reach the reader
    before the line it exists to demote.
    """
    led = pd.concat([_pair(1, .40, .005, True, "2026-09-12"),
                     _pair(2, .40, .005, False, "2026-09-13"),
                     _pair(3, .60, .030, True, "2026-09-14")],
                    ignore_index=True)
    out = hybrid_v2.report_lines(led)
    text = "\n".join(out)
    assert "SWITCH DELTA (registered)" in text
    assert "GATE:" in text
    head = next(i for i, l in enumerate(out) if "SWITCH DELTA (registered)" in l)
    comb = next(i for i, l in enumerate(out) if "combined hybrid" in l)
    assert head < comb
    assert "mostly the model, not the rule" in text


def test_the_gate_constants_have_one_home():
    """A second literal is the `one value, three homes` defect. v2's gate is
    v1's arithmetic on the same effect size, so it is imported; the report says
    in words that v2 accrues switches more slowly."""
    assert hybrid_v2.GATE_SWITCHES is hybrid_test.GATE_SWITCHES
    assert (hybrid_v2.GATE_SWITCHES_REALISTIC
            is hybrid_test.GATE_SWITCHES_REALISTIC)
    src = open(hybrid_v2.__file__).read()
    body = src.split("GATE_SWITCHES_REALISTIC = v1.GATE_SWITCHES_REALISTIC", 1)[1]
    assert f"= {hybrid_test.GATE_SWITCHES}" not in body


def test_the_discovery_reference_is_scoped_to_the_rules_own_family():
    """The gate is denominated in the CURRENT delta scale, so splitting the
    whole ledger at the registration date reaches back through families whose
    `xw_net` means something else -- the `_SCALE_FAMILIES` error inside one
    number. Measured on the committed ledger when this was written, the
    unscoped split read +0.049u over 44 switches against the family-scoped
    +0.679u over 16: a reference 14x off, in the direction that flatters a
    negative forward reading.

    Pinned as a PROPERTY: an out-of-family row that WOULD be faded must not
    move the figure, whatever its result.
    """
    pre, post = "2026-09-10", "2026-09-12"
    base = [_pair(1, .40, .005, True, pre), _pair(2, .60, .030, True, post),
            _pair(3, .40, .005, False, post)]
    clean = hybrid_v2.discovery_switch_delta(pd.concat(base, ignore_index=True))
    assert clean is not None and clean["n"] == 1
    assert clean["families"] == (grade_leans.MODEL_TAG,)

    # Same frame plus pre-registration rows from an older family, faded and
    # LOSING, which would drag an unscoped mean down hard.
    older = [_pair(10 + i, .40, .005, False, pre, tag="woba+plat_consol_v5")
             for i in range(6)]
    mixed = hybrid_v2.discovery_switch_delta(
        pd.concat(base + older, ignore_index=True))
    assert mixed["n"] == clean["n"]
    assert mixed["mean"] == pytest.approx(clean["mean"])
    assert mixed["families"] == (grade_leans.MODEL_TAG,)


def test_no_forward_rows_means_no_discovery_reference():
    """The figure exists to be read against a forward number. With no forward
    rows there is nothing to read it against, and the family cannot be derived
    from them either -- so it is withheld rather than guessed."""
    led = _pair(1, .40, .005, True, "2026-09-10")
    assert hybrid_v2.discovery_switch_delta(led) is None
    text = "\n".join(hybrid_v2.report_lines(led))
    assert "discovery was" not in text


# --------------------------------------------------------------------------
# The v1 archive as a ROW SELECTOR: the mechanism that froze three
# registrations on 2026-09-11, and the three places that now stop it.
# --------------------------------------------------------------------------

def test_the_v1_decision_has_exactly_one_derivation():
    """`locked_v1_decision` is the rule; no caller may respell it.

    The gate lived in `hybrid_test` while the arithmetic reading it lived
    inline in `migrate_hybrid_v2`, which is "one value, three homes" across a
    module boundary -- a later edit to either could publish an archive the v1
    scorer disagrees with. Pinned as a property of the SOURCE rather than of
    one output, because a second copy passes every value test until it drifts.
    """
    import re
    for mod in ("migrate_hybrid_v2.py", "grade_leans.py"):
        src = open(mod, encoding="utf-8").read()
        assert "hybrid_test.THRESHOLD" not in src, mod
        assert not re.search(r'hybrid_v1_action"?\]?\s*=\s*\(?"(FOLLOW|FADE)"',
                             src), mod


def test_the_v1_decision_follows_at_the_threshold_and_refuses_bad_input():
    at = hybrid_test.locked_v1_decision("H", "H", "A", hybrid_test.THRESHOLD,
                                        -150, 130)
    assert at[0] == "FOLLOW" and at[1] == "H"        # `>=` follows, exactly
    below = hybrid_test.locked_v1_decision("H", "H", "A", .30, 130, -150)
    assert below[0] == "FADE" and below[1] == "A"    # fade backs the other side
    assert below[2] == pytest.approx(.70)            # p is the BET's price
    assert below[3] == -150
    # An AWAY lean reads q off the other side of the same price.
    assert hybrid_test.locked_v1_decision("A", "H", "A", .40, 130, -150) \
        == ("FOLLOW", "A", pytest.approx(.60), -150)
    # v1 is unconditional on the delta: it is not even an argument.
    import inspect
    assert "delta" not in inspect.signature(
        hybrid_test.locked_v1_decision).parameters
    for bad in (dict(lean="X"), dict(p_home=float("nan")), dict(p_home=0.0),
                dict(p_home=1.0), dict(home_ml=50), dict(away_ml=None)):
        kw = dict(lean="H", home="H", away="A", p_home=.60,
                  home_ml=-150, away_ml=130)
        kw.update(bad)
        assert hybrid_test.locked_v1_decision(**kw) is None, bad


def test_ingest_mints_the_archive_for_pending_rows_and_never_for_graded_ones():
    """The writer-side fix. A graded archive is immutable; a pending one is
    re-derived, because MODEL_FIELDS rebuilds the lean and the price under it.
    """
    pending = _rows(.60, .020, date="2026-09-20")
    pending["status"] = "pending"
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        pending[col] = np.nan
    graded = _rows(.60, .020, date="2026-09-19")
    graded["game_pk"] = 2
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        graded[col] = np.nan
    led = pd.concat([pending, graded], ignore_index=True)

    assert grade_leans._mint_v1_archive(led) == 1
    assert led.at[0, "hybrid_v1_action"] == "FOLLOW"
    assert led.at[0, "hybrid_v1_selection"] == "H"
    assert pd.isna(led.at[1, "hybrid_v1_action"])   # graded: left alone

    # A pending row whose lean flipped on a later poll is re-derived from it.
    # At this price the flipped lean is a sub-.45 dog, so v1 fades it back --
    # a different ACTION off the same market, which is the thing a once-only
    # mint at insert would have missed.
    led.at[0, "xw_lean"] = "A"
    grade_leans._mint_v1_archive(led)
    assert led.at[0, "hybrid_v1_action"] == "FADE"
    assert led.at[0, "hybrid_v1_selection"] == "H"


def test_the_archive_is_minted_for_every_family_row_a_live_test_selects():
    """The regression this whole change exists for, stated as a property.

    `abstain_test` and `dog_contrast_test` are LIVE registrations that select
    rows through `hybrid_test.scored_rows`, which requires the archive. So a
    current-family row carrying a usable locked market and no archive silently
    leaves two open gates -- and invisibly, because it is dropped inside
    `_committed`, one step before the `unscorable` counter that exists to make
    a shrinking forward denominator visible.
    """
    led = pd.read_csv(grade_leans.LEDGER_PATH, low_memory=False)
    # The family is read off the LEDGER, not off the running build. A
    # `MODEL_TAG` bump makes the build's current family empty until its first
    # slate lands, and a test keyed to the build would then either fail on an
    # empty frame or -- worse -- pass vacuously for exactly as long as the
    # invariant was unobservable. Reading it off the ledger's own most recent
    # graded rows keeps the property under test on both sides of a bump. This
    # is the version-note-asserts-rows failure that CLAUDE.md records three
    # times, arriving in a test instead of in prose.
    graded = led[led["status"].astype(str).eq("graded")]
    family = graded.loc[graded["game_date"].astype(str).idxmax(), "model_tag"]
    cur = led[led["model_tag"].astype(str).eq(str(family))]
    usable = cur[[hybrid_test.locked_v1_decision(
        r.xw_lean, r.home, r.away, r.pregame_p_home,
        r.pregame_home_ml, r.pregame_away_ml) is not None
        for r in cur.itertuples()]]
    assert len(usable)
    assert usable["hybrid_v1_action"].notna().all(), (
        sorted(usable.loc[usable["hybrid_v1_action"].isna(), "game_date"]
               .unique()))
    # And the scored window reaches the ledger's own latest graded slate.
    scored = hybrid_test.scored_rows(led)
    latest = cur.loc[cur["status"].eq("graded"), "game_date"].max()
    assert scored["game_date"].max() == latest


def test_a_migration_leaves_every_cell_it_did_not_change_byte_identical(tmp_path):
    """A repair's diff has to be reviewable, so `to_csv` is not good enough.

    A read/write round-trip of this ledger rewrites cells the migration never
    touched: `read_csv`'s default float parser is inexact, and pandas renders
    floats to fewer digits than `build_site` writes. Both are pre-existing and
    neither belongs in a backfill's commit. Pinned with a literal `read_csv`
    provably cannot round-trip -- it parses this one two digits short -- so the
    test fails if the writer ever reaches for the frame's value instead of the
    file's bytes.
    """
    src = _rows(.60, .020, date="2026-09-13")
    src["d_lineup"] = 0.0
    # Canonical column order first: `migrate` reorders to the persisted list,
    # and a header that moves is the one case with no cell-for-cell
    # correspondence, where the writer correctly falls back to a whole file.
    src, _ = migrate_hybrid_v2.migrate(src)
    for col in migrate_hybrid_v2.V1_ARCHIVE:
        src[col] = np.nan
    path = tmp_path / "led.csv"
    src.to_csv(path, index=False)
    precise = "0.0008232366754536979"
    lines = path.read_text().splitlines()
    cells = lines[1].split(",")
    cells[lines[0].split(",").index("d_lineup")] = precise
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")
    assert repr(pd.read_csv(path)["d_lineup"].iloc[0]) != precise   # the hazard

    before = path.read_text().splitlines()
    led = pd.read_csv(path, low_memory=False)
    migrated, changed = migrate_hybrid_v2.migrate(led)
    migrate_hybrid_v2.write_changed_cells(str(path), migrated)
    after = path.read_text().splitlines()

    assert changed == 1
    hdr = before[0].split(",")
    a, b = before[1].split(","), after[1].split(",")
    assert b[hdr.index("d_lineup")] == precise      # untouched, not truncated
    moved = {hdr[i] for i, (x, y) in enumerate(zip(a, b)) if x != y}
    assert moved == set(migrate_hybrid_v2.V1_ARCHIVE)
