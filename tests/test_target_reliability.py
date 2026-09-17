"""The target-reliability monitor and the within-starter lineup slope.

Both exist because of one measurement: realised team offence in this ledger
cannot tell batting clubs apart (F = 0.941 over 1,882 team-games, below
chance) while the SAME plate appearances grouped by the starter who threw them
can (F = 1.503). Every component diagnostic in this repo scored a predictor
against a target whose own reliability nothing had ever checked, so a lineup
slope that is UNDEFINED was read for weeks as a measured null.

These tests pin the RULES, never the current figures. The numbers above move
with every grading pass; the properties below do not, and freezing a figure
here would reproduce the constants-frozen-from-data anti-pattern inside the one
gate meant to catch it.
"""
import unittest

import os

import numpy as np
import pandas as pd

import actuals_backfill as ab


class OneWayIccTests(unittest.TestCase):
    def test_no_group_structure_centres_F_on_one(self):
        """The case the monitor exists for, asserted across replications rather
        than on one draw. A single noise sample routinely returns F = 1.4 at 30
        units -- pinning one seed would be pinning that draw, and asserting a
        loose bound on it would pass while the statistic was broken."""
        fs = []
        for seed in range(60):
            rng = np.random.default_rng(seed)
            fs.append(ab.one_way_icc(
                [f"club{i % 30}" for i in range(600)],
                rng.normal(0.31, 0.09, 600))["f"])
        self.assertAlmostEqual(float(np.mean(fs)), 1.0, delta=0.12)

    def test_the_null_false_positive_rate_is_near_alpha(self):
        """Why the verdict is gated on p and not on a threshold for F. Before
        this, pure noise at 30 units was labelled 'measurable' outright."""
        called = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            r = ab.one_way_icc([f"club{i % 30}" for i in range(600)],
                               rng.normal(0.31, 0.09, 600))
            called += r["p"] < ab.RELIABILITY_ALPHA
        self.assertLess(called / 200.0, 0.12)

    def test_the_f_tail_matches_published_critical_values(self):
        """A hand-rolled incomplete beta is the kind of thing that is subtly
        wrong and never noticed, so it is checked against the F table."""
        self.assertAlmostEqual(ab.f_upper_tail(1.0, 10, 10), 0.5, places=6)
        self.assertAlmostEqual(ab.f_upper_tail(2.978, 10, 10), 0.05, places=3)
        self.assertAlmostEqual(ab.f_upper_tail(2.070, 20, 100), 0.01, places=2)

    def test_real_group_structure_is_recovered(self):
        """The monitor must be able to say yes, or a null from it means
        nothing. Groups given genuinely different means read F well above 1."""
        rng = np.random.default_rng(1)
        vals, labels = [], []
        for i in range(30):
            centre = 0.31 + 0.05 * (i - 15) / 15.0
            vals.extend(rng.normal(centre, 0.02, 20))
            labels.extend([f"club{i}"] * 20)
        r = ab.one_way_icc(labels, vals)
        self.assertGreater(r["f"], 3.0)
        self.assertGreater(r["icc"], 0.2)
        self.assertGreater(r["between_sd"], 0.0)

    def test_a_negative_variance_component_is_reported_not_hidden(self):
        """F below 1 means the units differ LESS than chance. The between-sd is
        clipped to zero, so the flag is the only thing distinguishing 'exactly
        zero' from 'the estimate went negative' -- and the second is the
        finding."""
        vals = [0.0, 1.0] * 30
        labels = [f"g{i // 2}" for i in range(60)]
        r = ab.one_way_icc(labels, vals, min_per_group=2)
        self.assertLessEqual(r["f"], 1.0)
        self.assertTrue(r["negative_variance"])
        self.assertEqual(r["between_sd"], 0.0)

    def test_too_few_units_returns_none_rather_than_a_number(self):
        self.assertIsNone(ab.one_way_icc(["a"] * 10, list(range(10))))

    def test_singleton_units_are_excluded_from_the_decomposition(self):
        """A unit seen once contributes a zero within-group deviation, which
        deflates MSW and inflates F for a reason unrelated to reliability."""
        labels = ["a"] * 8 + ["b"] * 8 + [f"solo{i}" for i in range(40)]
        vals = list(np.linspace(0, 1, 56))
        r = ab.one_way_icc(labels, vals)
        self.assertEqual(r["n_groups"], 2)
        self.assertEqual(r["n_obs"], 16)


def _pair_rows(n_pitchers=12, starts=8, lineup_effect=1.0, pitcher_effect=0.0,
               noise=0.0, seed=0):
    """Long component frame with a KNOWN lineup slope and pitcher spread."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pitchers):
        p_off = pitcher_effect * (p - n_pitchers / 2) / n_pitchers
        for k in range(starts):
            # The lineups a pitcher draws are CORRELATED with his own effect.
            # Without that the pitcher term is orthogonal to `pred` and biases
            # nothing, so a pooled-vs-FE comparison built on it would compare
            # two unbiased estimators and prove the opposite of its claim.
            pred = (0.30 + 0.02 * ((k % starts) - starts / 2) / starts
                    + 0.01 * (p - n_pitchers / 2) / n_pitchers)
            act = 0.31 + lineup_effect * (pred - 0.30) + p_off
            if noise:
                act += rng.normal(0, noise)
            rows.append({"game_pk": f"{p}-{k}", "model_tag": "t",
                         "component": "lineup", "side": "away",
                         "pred": pred, "act": act, "den": 38,
                         "unit": f"club{k}", "faced_sp": f"pitcher{p}"})
    return pd.DataFrame(rows)


class WithinStarterSlopeTests(unittest.TestCase):
    def test_recovers_a_known_slope_with_the_pitcher_absorbed(self):
        """A pitcher effect 5x the lineup signal is exactly the confound the
        pooled slope cannot see. Demeaning within the starter must return the
        lineup slope it was built with."""
        p = _pair_rows(lineup_effect=1.0, pitcher_effect=0.10)
        slope, se, n, n_pit = ab.lineup_within_pitcher_slope(p)
        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertEqual(n_pit, 12)

    def test_a_pooled_fit_is_biased_by_the_same_confound(self):
        """Why the FE line is worth printing beside the pooled one rather than
        instead of it: on identical rows the two disagree, and the pooled one
        is the one carrying the pitcher's variance."""
        p = _pair_rows(lineup_effect=1.0, pitcher_effect=0.40, noise=0.01, seed=3)
        pooled = ab.calibration(p["pred"], p["act"])["slope"]
        fe = ab.lineup_within_pitcher_slope(p)[0]
        self.assertGreater(abs(pooled - 1.0), abs(fe - 1.0))

    def test_zero_lineup_effect_reads_zero_not_the_pitcher_spread(self):
        p = _pair_rows(lineup_effect=0.0, pitcher_effect=0.30)
        slope, se, _n, _k = ab.lineup_within_pitcher_slope(p)
        self.assertAlmostEqual(slope, 0.0, places=6)

    def test_it_reads_only_the_lineup_component(self):
        """For SP the starter IS the subject, so absorbing him would remove the
        variance under test and return a confident null about nothing. The
        function must refuse rather than oblige."""
        p = _pair_rows()
        p["component"] = "SP"
        self.assertIsNone(ab.lineup_within_pitcher_slope(p))

    def test_singleton_starters_contribute_nothing_and_are_not_counted(self):
        p = _pair_rows(n_pitchers=4, starts=6)
        solo = p.iloc[:1].copy()
        solo["faced_sp"] = "one-start-guy"
        out = ab.lineup_within_pitcher_slope(pd.concat([p, solo], ignore_index=True))
        self.assertEqual(out[3], 4)

    def test_the_standard_error_charges_for_absorbed_starters(self):
        """A design with two starts per pitcher spends almost all its degrees
        of freedom on fixed effects. The se must reflect that rather than
        flattering a slope fitted on what is left."""
        wide = _pair_rows(n_pitchers=6, starts=16, noise=0.02, seed=5)
        thin = _pair_rows(n_pitchers=48, starts=2, noise=0.02, seed=5)
        self.assertGreater(ab.lineup_within_pitcher_slope(thin)[1],
                           ab.lineup_within_pitcher_slope(wide)[1])

    def test_no_usable_rows_returns_none(self):
        self.assertIsNone(ab.lineup_within_pitcher_slope(pd.DataFrame()))


class TargetReliabilityReportTests(unittest.TestCase):
    def _led(self, home_offence):
        rows = []
        for i, act in enumerate(home_offence):
            rows.append(dict(
                game_pk=i, model_tag="t", away="AWY", home=f"H{i % 30}",
                away_sp=f"sp{i % 20}", home_sp=f"sp{100 + i % 20}",
                starter_xwoba_away=0.31, bullpen_xwoba_away=0.30,
                opp_xwoba_neutral_away=0.32, act_woba_home=act,
                act_woba_away=0.30, act_pa_home=38, act_pa_away=38))
        return pd.DataFrame(rows)

    def test_a_target_that_cannot_separate_its_units_says_so_in_words(self):
        """The whole point. A component whose target cannot separate its units
        must be labelled, because its fitted slope is undefined rather than
        null and a reader cannot tell those apart from a number.

        Asserted over replications: one noise draw in twenty clears alpha by
        construction, so a single-seed version of this test would be pinning a
        coin flip."""
        labelled = 0
        for seed in range(40):
            rng = np.random.default_rng(seed)
            lines = ab.target_reliability(
                self._led(rng.normal(0.31, 0.09, 900)), min_per_group=8)
            lu = [l for l in lines if l.strip().startswith("lineup")]
            self.assertTrue(lu)
            labelled += "UNMEASURABLE" in lu[0]
        self.assertGreater(labelled, 34)

    def test_below_chance_is_distinguished_from_merely_unproven(self):
        """F < 1 and F = 1.2-at-p-0.3 are both unmeasurable and they are not
        the same fact. The ledger's lineup row is the first kind."""
        # Every club receives the identical multiset of outcomes, so the
        # between-club sum of squares is exactly zero while within-club
        # variance is large -- units that provably do not differ.
        lines = ab.target_reliability(
            self._led([0.20 + 0.01 * (i // 30) for i in range(900)]),
            min_per_group=8)
        lu = [l for l in lines if l.strip().startswith("lineup")][0]
        self.assertIn("less than chance", lu)

    def test_a_measurable_target_is_not_labelled_unmeasurable(self):
        rng = np.random.default_rng(4)
        vals = [0.31 + 0.06 * ((i % 30) - 15) / 15.0 + rng.normal(0, 0.01)
                for i in range(900)]
        lines = ab.target_reliability(self._led(vals), min_per_group=8)
        lu = [l for l in lines if l.strip().startswith("lineup")][0]
        self.assertNotIn("UNMEASURABLE", lu)
        self.assertIn("measurable", lu)

    def test_every_unmeasurable_component_is_marked_on_its_own_slope(self):
        """The marker travels with the statistic, not with one component.

        The caveat was prose under the LINEUP row only, so BP's slope printed
        bare under an UNMEASURABLE verdict of its own -- live on 2026-09-17 at
        F = 1.131 (p = 0.288). This asserts the RULE: for every component, the
        component block marks its slope if and only if the reliability block
        calls that component's target unmeasurable.
        """
        # The COMMITTED ledger, because the constructed frames here carry a
        # lineup component only and the defect was on BP. Nothing about a
        # verdict is pinned -- those move with the data -- only that the two
        # blocks agree, which cannot go stale.
        led = pd.read_csv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "mlb_lean_ledger.csv"))
        rel = ab.reliability_verdicts(led)
        comp = ab.components_summary(led)
        self.assertGreaterEqual(len(rel), 2)
        for name, (_r, verdict) in rel.items():
            row = [l for l in comp if l.strip().startswith(name)]
            if not row:
                continue
            with self.subTest(component=name):
                marked = "target UNMEASURABLE" in row[0]
                self.assertEqual(
                    marked, bool(verdict and verdict.startswith("UNMEASURABLE")),
                    f"{name}: slope marker disagrees with its own verdict")

    def test_the_caveat_no_longer_names_F_equals_1_as_the_test(self):
        """"Clears F=1" is the wrong bar and it passed the component that
        needed it most: F = 1.131 clears 1 and still fails on p. The caveat
        must point at the verdict rather than restate half of it."""
        led = self._led([0.20 + 0.01 * (i // 30) for i in range(900)])
        body = "\n".join(ab.components_summary(led))
        self.assertNotIn("clears F=1", body)
        self.assertIn("target-reliability block", body)

    def test_both_failure_modes_earn_the_marker_not_just_the_F_under_1_one(self):
        """F <= 1 and F > 1 with p above alpha are different facts and the
        same consequence. A marker keyed on F alone would catch one."""
        self.assertTrue(ab._reliability_verdict(
            dict(f=0.9, p=0.6)).startswith("UNMEASURABLE"))
        self.assertTrue(ab._reliability_verdict(
            dict(f=1.131, p=0.288)).startswith("UNMEASURABLE"))
        self.assertEqual(ab._reliability_verdict(dict(f=1.36, p=0.011)),
                         "measurable")
        self.assertIsNone(ab._reliability_verdict(None))

    def test_the_marker_is_scoped_to_every_family_while_the_row_is_not(self):
        """The component row is family-scoped and its marker deliberately is
        not: the target is metric-free, so scoping the ANOVA would discard
        rows for a reason that cannot apply to it. Half the frame is retagged,
        so the scoped row carries half the games while the marker still comes
        from all of them."""
        led = self._led([0.20 + 0.01 * (i // 30) for i in range(900)])
        led.loc[led.index[:450], "model_tag"] = "other"
        row = [l for l in ab.components_summary(led, tags={"t"})
               if l.strip().startswith("lineup")][0]
        self.assertIn("n=450", row)          # the row saw half the rows
        unscoped = ab.reliability_verdicts(led)["lineup"][1]
        self.assertEqual("target UNMEASURABLE" in row,
                         unscoped.startswith("UNMEASURABLE"))
        # ... and the verdict genuinely came from 900 rows, not 450.
        self.assertEqual(ab.reliability_verdicts(led)["lineup"][0]["n_obs"],
                         ab.reliability_verdicts(
                             led.assign(model_tag="t"))["lineup"][0]["n_obs"])

    def test_it_pools_families_because_the_actual_is_metric_free(self):
        """Scoping this to RECORD_TAGS would discard rows for a reason that
        cannot apply -- a realised wOBA is not a property of the tag beside
        it -- and the component block above already states its own scope."""
        rng = np.random.default_rng(6)
        led = self._led(rng.normal(0.31, 0.09, 900))
        led["model_tag"] = ["a" if i % 2 else "b" for i in range(len(led))]
        pooled = ab.target_reliability(led, min_per_group=8)
        lu = [l for l in pooled if l.strip().startswith("lineup")][0]
        self.assertIn("n=900", lu)

    def test_empty_input_yields_no_lines_rather_than_a_header(self):
        self.assertEqual(ab.target_reliability(pd.DataFrame()), [])


class PairedComponentIdentityTests(unittest.TestCase):
    def test_the_lineup_unit_is_the_batting_club_not_the_pitching_one(self):
        """The same cross the pred/act pairing makes. Getting it backwards
        would group a club's OPPONENTS' offence under its own name and read as
        a weak-but-plausible reliability rather than as an error."""
        led = pd.DataFrame([dict(
            game_pk=1, model_tag="t", away="AWY", home="HOM",
            away_sp="ace", home_sp="other",
            starter_xwoba_away=0.31, bullpen_xwoba_away=0.30,
            opp_xwoba_neutral_away=0.32,
            act_woba_home=0.40, act_woba_away=0.25, act_pa_home=38)])
        p = ab.paired_components(led)
        lu = p[p.component == "lineup"].iloc[0]
        self.assertEqual(lu.unit, "HOM")        # the offense that batted
        self.assertEqual(lu.faced_sp, "ace")    # the pitcher it faced
        self.assertNotEqual(lu.unit, lu.faced_sp)

    def test_the_sp_unit_is_the_starter_himself(self):
        led = pd.DataFrame([dict(
            game_pk=1, model_tag="t", away="AWY", home="HOM",
            away_sp="ace", home_sp="other",
            starter_xwoba_away=0.31,
            act_sp_ab_away=20, act_sp_h_away=5, act_sp_2b_away=1,
            act_sp_3b_away=0, act_sp_hr_away=1, act_sp_bb_away=2,
            act_sp_ibb_away=0, act_sp_hbp_away=0, act_sp_sf_away=0,
            act_woba_home=0.40)])
        sp = ab.paired_components(led)
        sp = sp[sp.component == "SP"].iloc[0]
        self.assertEqual(sp.unit, "ace")

    def test_the_empty_frame_carries_the_same_columns(self):
        """A caller reading `unit` off an empty frame must get an empty column,
        not a KeyError -- the report runs on an empty ledger on day one."""
        empty = ab.paired_components(pd.DataFrame())
        self.assertIn("unit", empty.columns)
        self.assertIn("faced_sp", empty.columns)


if __name__ == "__main__":
    unittest.main()
