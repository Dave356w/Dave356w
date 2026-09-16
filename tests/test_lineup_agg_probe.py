"""The aggregation panel's guards, the join it could get silently wrong, and
the two claims its prose makes about arithmetic.

Every assertion here is about a RULE rather than a number. The panel's figures
move with every committed frame, and a test that froze one would be the
`test_record_reproduces_ledger_report` instance again -- the suite failing for
arithmetic the Actions bot did overnight.
"""
import numpy as np
import pandas as pd
import pytest

import lineup_agg_probe as lap


def _lineup(game_pk, side, rates, weights=None, order=None, pa=None):
    n = len(rates)
    return pd.DataFrame({
        "game_pk": game_pk,
        "faced_pitcher": "X",
        "pitcher_side": "R",
        "batting_side": side,
        "player_id": range(100 * game_pk, 100 * game_pk + n),
        "batting_order": order if order is not None else range(1, n + 1),
        "PA": pa if pa is not None else [400] * n,
        "xwoba_raw": rates,
        "xwoba_shrunk": rates,
        "slot_weight": weights if weights is not None else [4.0] * n,
        "savant_backfill": False,
        "model_tag": "tag",
        "model_metric": "xwOBA",
        "snapshot_utc": "2026-09-06T20:00:00Z",
        "scheduled_start_utc": "2026-09-06T23:00:00Z",
        "lock_status": "pregame",
    })


def _panel_frame(n_sides=40, seed=0):
    """Sides whose composite genuinely predicts, so the guards have something
    to guard rather than a degenerate frame that trips every refusal."""
    rng = np.random.default_rng(seed)
    rows, acts = [], []
    for i in range(n_sides):
        theta = 0.30 + 0.02 * rng.standard_normal()
        rates = list(theta + 0.02 * rng.standard_normal(9))
        side = "home" if i % 2 else "away"
        rows.append(_lineup(i, side, rates))
        acts.append({"game_pk": i, "batting_side": side,
                     "act": theta + 0.09 * rng.standard_normal()})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(acts)


def _merged(n_sides=40, seed=0):
    h, a = _panel_frame(n_sides, seed)
    preds = []
    for (gp, side), g in h.groupby(["game_pk", "batting_side"]):
        row = {"game_pk": gp, "batting_side": side}
        for key, _l, fn, _n in lap.VARIANTS:
            row[key] = fn(g)
        preds.append(row)
    return pd.DataFrame(preds).merge(a, on=["game_pk", "batting_side"]), h


class TestTheJoinThatCouldBeSilentlyWrong:
    """`batting_side` is the OFFENSE. Crossing it returns a plausible weak
    correlation rather than an error, which is why this is pinned against a
    constructed frame and never against the sign of a real correlation."""

    def test_a_side_is_scored_against_its_own_realised_woba(self, tmp_path):
        hot = [0.40] * 9
        cold = [0.20] * 9
        h = pd.concat([_lineup(1, "away", hot), _lineup(1, "home", cold)],
                      ignore_index=True)
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        led = pd.DataFrame([{
            "game_pk": 1, "scheduled_start_utc": "2026-09-06T23:00:00Z",
            "act_woba_away": 0.55, "act_woba_home": 0.11}])
        lp = tmp_path / "ledger.csv"
        led.to_csv(lp, index=False)
        m, _prov, _h = lap.load_sides(str(d), str(lp), tags=())
        got = dict(zip(m["batting_side"], m["act"]))
        assert got == {"away": 0.55, "home": 0.11}
        hot_row = m[m["batting_side"] == "away"].iloc[0]
        assert hot_row[lap.BASELINE] == pytest.approx(0.40)

    def test_a_frame_written_after_first_pitch_is_not_scored(self, tmp_path):
        """The pregame rule is hitter_level_probe's, not a second copy: a
        post-hoc frame carries a leaderboard the game is already inside."""
        h = _lineup(1, "away", [0.30] * 9)
        h["snapshot_utc"] = "2026-09-07T02:00:00Z"
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        lp = tmp_path / "ledger.csv"
        pd.DataFrame([{"game_pk": 1,
                       "scheduled_start_utc": "2026-09-06T23:00:00Z",
                       "act_woba_away": 0.30, "act_woba_home": 0.30}
                      ]).to_csv(lp, index=False)
        m, prov, _h = lap.load_sides(str(d), str(lp), tags=())
        assert prov["post_hoc"] == 9
        assert m.empty

    def test_a_tag_outside_the_family_is_not_pooled(self, tmp_path):
        h = _lineup(1, "away", [0.30] * 9)
        h["model_tag"] = "some_older_family"
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        lp = tmp_path / "ledger.csv"
        pd.DataFrame([{"game_pk": 1,
                       "scheduled_start_utc": "2026-09-06T23:00:00Z",
                       "act_woba_away": 0.30, "act_woba_home": 0.30}
                      ]).to_csv(lp, index=False)
        m, _p, _h = lap.load_sides(str(d), str(lp), tags=("only_this_one",))
        assert m.empty


class TestTheDeclaredOrder:
    def test_the_variants_are_never_reordered_by_result(self):
        m, _h = _merged()
        rows = lap.panel(m, n_boot=50)
        assert [r["key"] for r in rows] == [k for k, _l, _f, _n in lap.VARIANTS]

    def test_the_baseline_is_what_ships_and_takes_no_paired_difference(self):
        m, _h = _merged()
        rows = {r["key"]: r for r in lap.panel(m, n_boot=50)}
        assert lap.BASELINE == lap.VARIANTS[0][0]
        assert not np.isfinite(rows[lap.BASELINE]["d_corr"])

    def test_the_row_selector_is_not_a_hardcoded_tag(self):
        """`interaction_probe` carried a frozen tag list through two model
        bumps and went on scoring a lineage the build no longer ran."""
        import build_site
        src = open("lineup_agg_probe.py", encoding="utf-8").read()
        assert "build_site.RECORD_TAGS" in src
        for tag in build_site.RECORD_TAGS:
            assert tag not in src


class TestTheStandaloneGuard:
    def test_a_row_that_cannot_reach_two_se_is_unusable(self):
        m, _h = _merged(n_sides=40)
        for r in lap.panel(m, n_boot=50):
            if not np.isfinite(r["ceiling"]) or not np.isfinite(r["se"]):
                continue
            assert r["usable"] == (r["se"] <= r["ceiling"] / 2.0)

    def test_the_gate_is_the_n_that_makes_the_ceiling_two_se(self):
        m, _h = _merged()
        for r in lap.panel(m, n_boot=50):
            if np.isfinite(r["n_usable"]):
                assert r["n_usable"] == pytest.approx(
                    3.0 + 4.0 / r["ceiling"] ** 2)

    def test_the_report_says_unusable_rather_than_naming_a_winner(self, tmp_path):
        """At a sample where nothing is readable the panel must not print an
        ordering -- the best of eight unreadable correlations is the largest of
        eight draws from noise."""
        h, a = _panel_frame(n_sides=12)
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        led = a.pivot(index="game_pk", columns="batting_side",
                      values="act").reset_index()
        for c in ("away", "home"):
            if c not in led:
                led[c] = np.nan
        led = led.rename(columns={"away": "act_woba_away",
                                  "home": "act_woba_home"})
        led["scheduled_start_utc"] = "2026-09-06T23:00:00Z"
        lp = tmp_path / "ledger.csv"
        led.to_csv(lp, index=False)
        text = "\n".join(lap.report(str(d), str(lp), tags=(), n_boot=50))
        assert "UNUSABLE" in text
        assert "takes no place in any ordering" in text

    def test_the_ceiling_is_the_variants_own_spread_not_a_shared_one(self):
        """Comparing raw correlations across variants compares headroom: a
        composite that spreads more has more room to correlate."""
        m, _h = _merged()
        rows = {r["key"]: r for r in lap.panel(m, n_boot=50)}
        wide, narrow = rows["max_shrunk"], rows[lap.BASELINE]
        assert wide["sd_pred"] > narrow["sd_pred"]
        assert wide["ceiling"] > narrow["ceiling"]


class TestThePairedColumn:
    def test_pairing_is_tighter_than_the_standalone_interval(self):
        """The whole reason the paired column exists: two predictors that
        share their games and most of their spread cancel that noise."""
        m, _h = _merged(n_sides=60)
        rows = {r["key"]: r for r in lap.panel(m, n_boot=400)}
        flat = rows["flat_shrunk"]
        assert flat["rho_base"] > 0.99
        assert flat["d_se"] < flat["se"]

    def test_an_identical_variant_has_exactly_zero_difference(self):
        m, _h = _merged()
        m = m.copy()
        m["flat_shrunk"] = m[lap.BASELINE]
        rows = {r["key"]: r for r in lap.panel(m, n_boot=50)}
        assert rows["flat_shrunk"]["d_corr"] == pytest.approx(0.0, abs=1e-12)
        assert rows["flat_shrunk"]["d_se"] == pytest.approx(0.0, abs=1e-12)

    def test_the_search_bar_is_printed_and_grows_with_the_comparisons(self,
                                                                     tmp_path):
        h, a = _panel_frame(n_sides=30)
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        led = a.pivot(index="game_pk", columns="batting_side",
                      values="act").reset_index()
        for c in ("away", "home"):
            if c not in led:
                led[c] = np.nan
        led = led.rename(columns={"away": "act_woba_away",
                                  "home": "act_woba_home"})
        led["scheduled_start_utc"] = "2026-09-06T23:00:00Z"
        lp = tmp_path / "ledger.csv"
        led.to_csv(lp, index=False)
        text = "\n".join(lap.report(str(d), str(lp), tags=(), n_boot=50))
        assert "comparisons, so the largest |z| among them averages" in text
        k = sum(1 for _ in lap.VARIANTS) - 1
        assert f"{np.sqrt(2 * np.log(k)):.2f}" in text


class TestTheClosures:
    def test_the_predicted_shift_matches_the_observed_one(self):
        """A derivation nobody checks against the data is how a comment
        becomes wrong. The bound is approximate, so this asserts the order of
        magnitude rather than an equality."""
        m, h = _merged(n_sides=60)
        c = lap.weight_family_closure(h)
        observed = float(np.std(m[lap.BASELINE] - m["flat_shrunk"], ddof=1))
        # Equal weights in the fixture, so the two ends coincide exactly and
        # the bound must not claim a shift larger than the CV allows.
        assert c["cv"] == pytest.approx(0.0, abs=1e-12)
        assert observed == pytest.approx(0.0, abs=1e-12)
        assert c["predicted_shift"] == pytest.approx(0.0, abs=1e-12)

    def test_a_real_weight_spread_bounds_the_real_shift(self):
        rng = np.random.default_rng(4)
        rows, obs_lin, obs_flat = [], [], []
        for i in range(60):
            rates = list(0.31 + 0.025 * rng.standard_normal(9))
            w = list(4.6 - 0.1 * np.arange(9))
            rows.append(_lineup(i, "home", rates, weights=w))
            obs_lin.append(np.average(rates, weights=w))
            obs_flat.append(float(np.mean(rates)))
        h = pd.concat(rows, ignore_index=True)
        c = lap.weight_family_closure(h)
        observed = float(np.std(np.array(obs_lin) - np.array(obs_flat), ddof=1))
        assert c["cv"] > 0
        # c*s/sqrt(n) is a scale, not an equality -- it must land within a
        # factor of a few of the shift it predicts, in the right direction.
        assert observed < 3 * c["predicted_shift"]
        assert observed > c["predicted_shift"] / 5

    def test_the_log_odds_composite_tracks_the_linear_one(self):
        m, _h = _merged(n_sides=60)
        lo = lap.logodds_closure(m)
        assert lo["pearson"] > 0.999
        assert lo["max_abs"] < lo["sd_lin"]

    def test_closed_is_not_claimed_to_mean_indistinguishable(self, tmp_path):
        """The first run separated the log-odds row in the paired column while
        the prose above it said no sample ever would. The claim went, not the
        measurement -- and this pins that it stays gone."""
        h, a = _panel_frame(n_sides=30)
        d = tmp_path / "data"
        d.mkdir()
        h.to_csv(d / "hitters_2026-09-06_xw.csv", index=False)
        led = a.pivot(index="game_pk", columns="batting_side",
                      values="act").reset_index()
        for c in ("away", "home"):
            if c not in led:
                led[c] = np.nan
        led = led.rename(columns={"away": "act_woba_away",
                                  "home": "act_woba_home"})
        led["scheduled_start_utc"] = "2026-09-06T23:00:00Z"
        lp = tmp_path / "ledger.csv"
        led.to_csv(lp, index=False)
        text = "\n".join(lap.report(str(d), str(lp), tags=(), n_boot=50))
        assert "no sample will make it one" not in text
        assert "not 'indistinguishable'" in text
        src = open("lineup_agg_probe.py", encoding="utf-8").read()
        assert "no sample ever would" not in src.split('"""')[2]


class TestItSaysWhatItCannotDo:
    def test_an_empty_frame_reports_rather_than_returning_a_panel(self,
                                                                 tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        lp = tmp_path / "ledger.csv"
        pd.DataFrame([{"game_pk": 1}]).to_csv(lp, index=False)
        text = "\n".join(lap.report(str(d), str(lp), tags=()))
        assert "no scorable sides yet" in text
        assert "cannot be backfilled" in text

    def test_it_names_the_probe_that_holds_the_other_half(self, tmp_path):
        m, _h = _merged()
        text = "\n".join(lap.report("data", lap.LEDGER, tags=("nothing",)))
        assert "hitter_level_probe" in text

    def test_it_claims_no_betting_signal(self):
        text = "\n".join(lap.report("data", lap.LEDGER, tags=("nothing",)))
        assert "feeds no lean" in text
