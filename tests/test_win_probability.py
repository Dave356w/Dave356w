"""Probability recovery, time leakage, and paired benchmark integrity."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import win_probability as wp

TAG = "xw+plat_consol_v12"
SMALL = wp.Settings(min_train=20, min_slates=2)


def ledger(n=120, seed=11):
    rng = np.random.default_rng(seed)
    day = pd.Timestamp("2026-08-01") + pd.to_timedelta(np.arange(n) // 10, unit="D")
    x = rng.normal(0, .025, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(.15 + 20 * x))))
    return pd.DataFrame(dict(
        game_pk=np.arange(n) + 1, game_date=day.strftime("%Y-%m-%d"),
        model_tag=TAG, model_metric="xwOBA", status="graded", xw_net=x,
        full_home=np.where(y, 5, 2), full_away=np.where(y, 2, 5),
        snapshot_utc=day.strftime("%Y-%m-%dT15:00:00Z"),
        scheduled_start_utc=day.strftime("%Y-%m-%dT19:00:00Z"), lock_status="pregame",
        pregame_market_utc=day.strftime("%Y-%m-%dT14:59:00Z"),
        pregame_p_home=1 / (1 + np.exp(-(.12 + 16 * x))),
        close_p_home=.99,
    ))


def replay(frame):
    rows, _ = wp.prepare_rows(frame, TAG)
    return wp.walk_forward(rows, SMALL)


@pytest.mark.parametrize("seed", range(5))
def test_recovers_planted_probability_mapping(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .025, 8000)
    true_p = 1 / (1 + np.exp(-(.2 + 25 * x)))
    model = wp.fit_logit(x, rng.binomial(1, true_p))
    assert abs(model.intercept - .2) < .09
    assert abs(model.slope_per_unit / model.delta_unit - 25) < 3
    assert np.mean((model.predict(x) - true_p) ** 2) < .001
    assert model.predict(.02) > model.predict(-.02)


def test_penalty_keeps_separation_and_one_class_fits_finite():
    x = np.linspace(-.1, .1, 100)
    for y in ((x > 0).astype(int), np.ones(100), np.zeros(100)):
        model = wp.fit_logit(x, y)
        assert np.isfinite([model.intercept, model.slope_per_unit]).all()
        p = model.predict(x)
        assert ((p > 0) & (p < 1)).all()
    weak = wp.fit_logit(x, x > 0, replace(SMALL, ridge=.1))
    strong = wp.fit_logit(x, x > 0, replace(SMALL, ridge=20))
    assert abs(strong.slope_per_unit) < abs(weak.slope_per_unit)


def test_zero_delta_remains_a_valid_probability_input():
    frame = ledger()
    frame.loc[0, "xw_net"] = 0
    frame["xw_lean"] = np.nan
    rows, _ = wp.prepare_rows(frame, TAG)
    assert len(rows) == len(frame)
    assert rows.iloc[0].xw_net == 0


def test_training_preserves_whole_slates_and_uses_only_earlier_dates():
    frame = ledger()
    pred, folds = replay(frame)
    assert len(pred) == 100
    assert pred.groupby("game_date").size().eq(10).all()
    assert folds.train_last_date.lt(folds.train_before_date).all()
    assert folds.train_before_date.le(folds.game_date).all()
    for date, g in pred.groupby("game_date"):
        assert g.train_n.nunique() == 1
        assert g.intercept.nunique() == g.slope_per_unit.nunique() == 1
        assert g.train_n.iloc[0] == frame.game_date.lt(date).sum()


def test_current_and_future_outcomes_cannot_change_earlier_predictions():
    frame = ledger()
    before, _ = replay(frame)
    cutoff = "2026-08-06"
    future = frame.game_date.ge(cutoff)
    frame.loc[future, ["full_home", "full_away"]] = frame.loc[
        future, ["full_away", "full_home"]].to_numpy()
    after, _ = replay(frame)
    cols = ["game_pk", "p_home", "p_prior_home", "intercept", "slope_per_unit"]
    pd.testing.assert_frame_equal(before.loc[before.game_date.le(cutoff), cols],
                                  after.loc[after.game_date.le(cutoff), cols])


def test_input_order_and_future_feature_scale_do_not_change_earlier_fits():
    frame = ledger()
    before, _ = replay(frame)
    future = frame.game_date.gt("2026-08-06")
    frame.loc[future, "xw_net"] *= 15
    after, _ = replay(frame.sample(frac=1, random_state=2))
    cols = ["game_pk", "p_home", "p_prior_home"]
    pd.testing.assert_frame_equal(before.loc[before.game_date.le("2026-08-06"), cols],
                                  after.loc[after.game_date.le("2026-08-06"), cols])


def test_early_forecast_moves_entire_slate_training_cutoff_back():
    frame = ledger()
    frame.loc[frame.game_date.eq("2026-08-06"), "snapshot_utc"] = "2026-08-05T15:00:00Z"
    pred, _ = replay(frame)
    g = pred.loc[pred.game_date.eq("2026-08-06")]
    assert g.train_before_date.eq("2026-08-05").all()
    assert g.train_last_date.eq("2026-08-04").all()


def test_market_and_closing_values_never_enter_the_fit():
    frame = ledger()
    before, _ = replay(frame)
    frame["pregame_p_home"] = .01
    frame["close_p_home"] = 1 - frame.close_p_home
    frame["close_home_ml"], frame["hybrid_p"] = -9999, .999
    after, _ = replay(frame)
    np.testing.assert_array_equal(before.p_home, after.p_home)
    assert after.pregame_p_home.eq(.01).all()


def test_missing_and_late_markets_only_exclude_paired_scoring():
    frame = ledger()
    original, _ = replay(frame)
    frame.loc[25, "pregame_p_home"] = np.nan
    frame.loc[26, "pregame_market_utc"] = "2026-08-03T15:01:00Z"
    frame.loc[27, "pregame_market_utc"] = "2026-08-03T14:00:00"  # naive
    frame.loc[28, "pregame_p_home"] = 1.0
    pred, _ = replay(frame)
    np.testing.assert_array_equal(original.p_home, pred.p_home)
    scores = wp.evaluate(pred, bootstrap_repeats=100)
    assert scores["all_oos"]["xwoba"]["n"] == 100
    for row in scores["paired_market"].values():
        assert row["n"] == 96
    assert pred.loc[~pred.market_eligible, "pregame_p_home"].isna().sum() == 1


def test_invalid_features_scores_and_late_snapshots_are_counted():
    frame = ledger(30)
    frame.loc[0, "model_tag"] = "unrelated"
    frame.loc[1, "model_metric"] = "wOBA"
    frame.loc[2, "status"] = "pending"
    frame.loc[3, "game_date"] = "bad"
    frame.loc[4, "xw_net"] = np.inf
    frame.loc[5, "full_home"] = frame.loc[5, "full_away"]
    frame.loc[6, "snapshot_utc"] = frame.loc[6, "scheduled_start_utc"]
    rows, audit = wp.prepare_rows(frame, TAG)
    assert len(rows) == 23
    assert audit["source_rows"] == len(rows) + sum(audit["excluded"].values())
    assert all(n == 1 for n in audit["excluded"].values())


def test_duplicates_and_missing_schema_fail_explicitly():
    frame = ledger()
    with pytest.raises(ValueError, match="Duplicate"):
        wp.prepare_rows(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), TAG)
    with pytest.raises(ValueError, match="Missing ledger columns"):
        wp.prepare_rows(frame.drop(columns="snapshot_utc"), TAG)


def test_probability_scores_match_hand_calculation_and_pair_exactly():
    pred, _ = replay(ledger())
    pred["p_home"] = .5
    pred["pregame_p_home"] = .5
    summary = wp.evaluate(pred, bootstrap_repeats=100)
    for m in (summary["all_oos"]["xwoba"], summary["paired_market"]["market"]):
        assert m["brier"] == pytest.approx(.25)
        assert m["log_loss"] == pytest.approx(np.log(2))
    for m in summary["model_minus_market"].values():
        assert m["difference"] == 0
        assert m["ci95"] == [0., 0.]
    bins = wp.calibration_rows(pred)
    assert bins.groupby("model").n.sum().eq(100).all()


def test_no_market_and_no_evaluation_rows_are_reportable():
    frame = ledger(10)
    rows, _ = wp.prepare_rows(frame, TAG)
    pred, _ = wp.walk_forward(rows, SMALL)
    assert wp.evaluate(pred)["all_oos"]["xwoba"]["n"] == 0
    frame = ledger().drop(columns=["pregame_p_home", "pregame_market_utc"])
    pred, _ = replay(frame)
    assert wp.evaluate(pred)["paired_market"]["market"]["n"] == 0
    assert wp.calibration_rows(pred).n.sum() == 0


def test_preparation_and_evaluation_do_not_mutate_source_ledger():
    frame = ledger()
    original = frame.copy(deep=True)
    pred, _ = replay(frame)
    wp.evaluate(pred, bootstrap_repeats=100)
    pd.testing.assert_frame_equal(original, frame)


def test_the_source_scope_is_the_scale_family_and_not_one_tag():
    """A delta-to-probability map is a UNITS question, so `_SCALE_FAMILIES`
    decides its row set -- not the running `MODEL_TAG`, which resets the
    sample at every bump, and not `RECORD_TAGS`, which pools v12 with v13
    across a deliberate scale change.

    Asserted as the RULE against build_site rather than as a tag literal: a
    test naming `xw+starter_blend_v13` would reproduce the very defect it
    guards, going stale at the next bump.
    """
    import build_site

    frame = ledger(40)
    frame.loc[20:, "model_tag"] = "xw+plat_consol_v10"
    both = wp.as_family([TAG, "xw+plat_consol_v10"])

    rows, audit = wp.prepare_rows(frame, both)
    assert len(rows) == 40
    assert audit["source_tags"] == list(both)
    assert audit["rows_by_tag"] == {TAG: 20, "xw+plat_consol_v10": 20}
    # A tag outside the family is still excluded -- widening the scope is not
    # the same as dropping it.
    assert len(wp.prepare_rows(frame, TAG)[0]) == 20

    # The default main() would run: the scale family, whatever it resolves to.
    assert wp.as_family(",".join(build_site.SCALE_TAGS)) == tuple(build_site.SCALE_TAGS)
    # Structural, not a text window: this module may DISCUSS `RECORD_TAGS` in
    # the comment explaining why it is the wrong relation, and a substring
    # search cannot tell that from reading it. Walk the AST for a real use.
    import ast

    tree = ast.parse(open("win_probability.py", encoding="utf-8").read())
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "SCALE_TAGS" in used
    assert "RECORD_TAGS" not in used
    assert "MODEL_TAG" not in used


def test_a_run_that_scores_nothing_says_so_instead_of_printing_a_warm_up_count():
    """The blind spot this module actually had: at the v13 bump its sample
    reset below the warm-up and four days of runs rendered as
    `Warm-up/unscored: 47` beside two `no eligible games` lines -- true, and
    indistinguishable from an empty ledger or a broken join.

    Both directions, because a line that always printed would be no better.
    """
    settings = wp.Settings(min_train=20, min_slates=2)

    thin = ledger(8)                      # 8 rows over 1 slate: both bind
    rows, audit = wp.prepare_rows(thin, TAG)
    predictions, _ = wp.walk_forward(rows, settings)
    assert predictions.empty
    reason = wp.blind_reason(rows, predictions, settings, audit["source_tags"])
    assert reason and "SCORES NOTHING" in reason
    assert TAG in reason                          # names the family it was blind on
    assert "12 more eligible rows" in reason      # and the binding shortfall
    assert "1 more slates" in reason
    assert wp.blind_reason(rows, predictions, settings, audit["source_tags"]) in \
        wp.report_text(dict(audit=audit, source_model_tag=TAG, settings={},
                            unscored_rows=len(rows), oos_first_date=None,
                            oos_last_date=None, metrics=wp.evaluate(predictions),
                            blind_reason=reason))

    fat = ledger(120)                     # scores, so the line must be absent
    rows, audit = wp.prepare_rows(fat, TAG)
    predictions, _ = wp.walk_forward(rows, settings)
    assert len(predictions)
    assert wp.blind_reason(rows, predictions, settings, audit["source_tags"]) is None
    text = wp.report_text(dict(audit=audit, source_model_tag=TAG, settings={},
                               unscored_rows=0, oos_first_date="a", oos_last_date="b",
                               metrics=wp.evaluate(predictions), blind_reason=None))
    assert "SCORES NOTHING" not in text


def test_only_the_row_count_shortfall_is_named_when_slates_already_clear():
    """It names whichever threshold BINDS, derived, so it cannot claim a
    shortfall that does not exist."""
    settings = wp.Settings(min_train=200, min_slates=2)
    rows, audit = wp.prepare_rows(ledger(60), TAG)
    predictions, _ = wp.walk_forward(rows, settings)
    reason = wp.blind_reason(rows, predictions, settings, audit["source_tags"])
    assert "140 more eligible rows" in reason
    assert "more slates" not in reason
