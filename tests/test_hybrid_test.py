"""The hybrid market-direction registration block must not drift.

Same reasoning as tests/test_forward_test.py, and the same deliberate
exception to this repo's rule against freezing measured numbers into tests:
here the literals ARE the subject. A pre-registered rule whose threshold can be
edited after seeing results is not pre-registered, and the whole value of
`hybrid_test.py` is that 0.45 was fixed before the rows it scores existed.

These assertions exist to make tuning DELIBERATE. If the threshold genuinely
must move, the honest move is a NEW registration with a new date, scoring from
scratch -- not an edit to this one.

Beyond the constants, two structural properties are pinned because the rule's
honest reading depends on them and neither is obvious from the formula:

  * exactly 0.45 FOLLOWS the model (the specification says so, and `>` instead
    of `>=` would silently reverse a boundary game);
  * the fade branch is always-chalk BY CONSTRUCTION, so a forward run that
    looks good may only be saying favourites are running hot.
"""

import datetime as _dt
import unittest

import numpy as np
import pandas as pd

import hybrid_test as ht


def _led(p_home, lean_home, home_won, date="2026-09-05"):
    """A minimal graded ledger frame at the given devigged home prices."""
    p_home = np.asarray(p_home, dtype=float)
    n = len(p_home)
    lean_home = np.broadcast_to(np.asarray(lean_home, dtype=bool), (n,))
    home_won = np.broadcast_to(np.asarray(home_won, dtype=bool), (n,))
    home_ml = np.where(
        p_home >= .5, -np.round(100 * p_home / (1 - p_home)),
        np.round(100 * (1 - p_home) / p_home))
    away_ml = np.where(
        p_home >= .5, np.round(100 * p_home / (1 - p_home)),
        -np.round(100 * (1 - p_home) / p_home))
    model_p = np.where(lean_home, p_home, 1 - p_home)
    follow = model_p >= ht.THRESHOLD
    bet_home = np.where(follow, lean_home, ~lean_home)
    bet_won = np.where(bet_home, home_won, ~home_won)
    return pd.DataFrame({
        "status": "graded", "game_date": date, "home": "H", "away": "A",
        "close_p_home": p_home,
        "xw_lean": np.where(lean_home, "H", "A"),
        "full_home": np.where(home_won, 1, 0),
        "full_away": np.where(home_won, 0, 1),
        # Fair prices from the devigged probabilities: the payout arithmetic is
        # not under test here, only the direction the rule takes.
        "close_home_ml": home_ml, "close_away_ml": away_ml,
        "selection_rule_tag": ht.RULE_TAG,
        "pregame_p_home": p_home,
        "pregame_home_ml": home_ml, "pregame_away_ml": away_ml,
        "hybrid_action": np.where(follow, "FOLLOW", "FADE"),
        "hybrid_selection": np.where(bet_home, "H", "A"),
        "hybrid_p": np.where(bet_home, p_home, 1 - p_home),
        "hybrid_ml": np.where(bet_home, home_ml, away_ml),
        "hybrid_full": np.where(bet_won, "W", "L"),
    })


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(ht.REGISTERED_ON, "2026-09-01")
        self.assertAlmostEqual(ht.THRESHOLD, 0.45, places=10)
        self.assertAlmostEqual(ht.STAKE, 1.0, places=10)
        self.assertEqual(ht.RULE_TAG, "xwoba_market_hybrid_v1")
        self.assertEqual(ht.PRIOR, "null")

    def test_discovery_constants_are_exactly_as_measured(self):
        self.assertEqual(ht.DISCOVERY_SWITCHES, 15)
        self.assertAlmostEqual(ht.DISCOVERY_SWITCH_DELTA, 0.592, places=10)
        self.assertAlmostEqual(ht.DISCOVERY_SWITCH_SD, 1.884, places=10)
        self.assertEqual(ht.DISCOVERY_PAIRED_ROI_CI, (-2.6, 10.4))
        self.assertEqual(ht.GATE_SWITCHES, 41)
        self.assertEqual(ht.GATE_SWITCHES_REALISTIC, 1420)

    def test_registration_date_is_a_real_past_date(self):
        d = _dt.date.fromisoformat(ht.REGISTERED_ON)
        self.assertLessEqual(d, _dt.date.today())

    def test_no_band_constants_exist(self):
        """The specification freezes ONE number, deliberately.

        value_probe's grid search returned a +20% ROI cell out of pure noise on
        this ledger, so adding a delta or price band to a rule whose active
        branch holds 15 games is the known way to manufacture one. If a band
        ever appears here it must be a new registration, not an extra constant.
        """
        banned = [n for n in dir(ht)
                  if n.isupper() and ("BAND" in n or "TIER" in n)]
        self.assertEqual(banned, [])


class SwitchRuleTests(unittest.TestCase):
    def test_exactly_at_the_threshold_follows_the_model(self):
        """The specification is explicit: a probability of exactly 45% follows.

        `>` instead of `>=` would reverse the selection on a boundary game, so
        this pins the comparison rather than trusting it.
        """
        g = ht.scored_rows(_led([0.45], lean_home=True, home_won=True))
        self.assertTrue(bool(g["follow"].iloc[0]))
        self.assertTrue(bool(g["bet_home"].iloc[0]))
        # And a hair below it fades.
        g = ht.scored_rows(_led([0.4499], lean_home=True, home_won=True))
        self.assertFalse(bool(g["follow"].iloc[0]))
        self.assertFalse(bool(g["bet_home"].iloc[0]))

    def test_an_away_lean_is_mirrored_correctly(self):
        """ModelSideP = 1 - close_p_home when the model leans the away side."""
        # Home priced .70 -> the away lean's q is .30, which is below .45, so
        # the rule fades onto the home side.
        g = ht.scored_rows(_led([0.70], lean_home=False, home_won=True))
        self.assertAlmostEqual(float(g["model_side_p"].iloc[0]), 0.30, places=9)
        self.assertFalse(bool(g["follow"].iloc[0]))
        self.assertTrue(bool(g["bet_home"].iloc[0]))
        self.assertTrue(bool(g["bet_won"].iloc[0]))

    def test_forward_scoring_uses_the_locked_market_not_the_close(self):
        """A later close cannot rewrite the side or price that was published."""
        led = _led([.46], lean_home=True, home_won=True)
        led["close_p_home"] = .20
        led["close_home_ml"] = 300
        led["close_away_ml"] = -400
        g = ht.scored_rows(led)
        self.assertTrue(bool(g["follow"].iloc[0]))
        self.assertTrue(bool(g["bet_home"].iloc[0]))
        self.assertAlmostEqual(float(g["p_bet"].iloc[0]), .46)
        self.assertEqual(float(g["ml_bet"].iloc[0]), 117)

        # Closing-market attachment is optional for forward scoring; a market
        # outage must not erase a settled locked selection.
        no_close = led.drop(columns=[
            "close_p_home", "close_home_ml", "close_away_ml"])
        self.assertEqual(len(ht.scored_rows(no_close)), 1)

    def test_the_fade_branch_is_always_chalk_by_construction(self):
        """Fading a sub-.45 lean always backs the favourite. Never a finding.

        This is why the module prints an always-chalk control beside the
        record: the fade branch's discovery result (11-4, +23.8%) IS the chalk
        record on those rows, in a window where chalk beat its price by 4pp.
        A forward run that looks good must be read against the control.
        """
        rng = np.random.default_rng(11)
        p = rng.uniform(0.02, 0.98, 400)
        g = ht.scored_rows(_led(p, lean_home=rng.random(400) < 0.5,
                                home_won=rng.random(400) < 0.5))
        sw = g[~g["follow"]]
        self.assertTrue(len(sw))
        chalk_home = (sw["close_p_home"] >= 0.5).values
        np.testing.assert_array_equal(sw["bet_home"].values, chalk_home)

    def test_always_chalk_is_the_larger_devigged_probability(self):
        """The chalk control selects max(p_home, 1-p_home), not the lean."""
        p = np.array([.20, .49, .51, .80])
        g = ht.scored_rows(_led(p, lean_home=[True, False, True, False],
                                home_won=True))
        expected_home = p > (1 - p)
        actual_home = np.where(g["chalk_won"].to_numpy(), True, False)
        np.testing.assert_array_equal(actual_home, expected_home)
        np.testing.assert_allclose(g["chalk_p"].to_numpy(), np.maximum(p, 1 - p))

    def test_hybrid_is_not_always_chalk(self):
        """A model side at 45%-49.9% is followed even though it is the dog."""
        g = ht.scored_rows(_led([.45, .49], lean_home=True, home_won=True))
        self.assertTrue(g["follow"].all())
        self.assertTrue(g["bet_home"].all())
        self.assertTrue((g["close_p_home"] < .5).all())

    def test_a_followed_game_has_exactly_zero_switch_delta(self):
        """The registered headline counts only switched games.

        On a followed game the hybrid and the plain lean are the same bet, so
        their difference must be identically zero -- otherwise the headline
        would quietly absorb the model's own performance, which is the defect
        the registration block calls reason 3.
        """
        rng = np.random.default_rng(3)
        p = rng.uniform(0.46, 0.98, 200)      # every row follows
        g = ht.scored_rows(_led(p, lean_home=True, home_won=rng.random(200) < .5))
        self.assertTrue(g["follow"].all())
        np.testing.assert_allclose(g["switch_delta"].values, 0.0, atol=1e-12)

    def test_a_row_with_no_lean_is_abstained_not_faded(self):
        """A v5 abstention publishes no direction, so there is nothing to fade.

        Treating a null lean as "the model likes the other side" would invent a
        selection the model declined to make.
        """
        led = _led([0.30, 0.30], lean_home=True, home_won=True)
        led.loc[0, "xw_lean"] = np.nan
        g = ht.scored_rows(led)
        self.assertEqual(len(g), 1)


class ScopeTests(unittest.TestCase):
    """The discovery sample must be unreachable, whatever the ledger holds."""

    def test_scored_rows_excludes_the_registration_date_itself(self):
        g = ht.scored_rows()
        if g is None:
            self.skipTest("ledger unavailable")
        if len(g):
            self.assertTrue((g["game_date"].astype(str) > ht.REGISTERED_ON).all())

    def test_the_committed_v12_discovery_rows_are_out_of_scope(self):
        """Every row behind the discovery table predates the registration.

        If this ever fails, the registration date was set after rows it claims
        to exclude -- which would make the forward test score its own search.
        """
        import os
        if not os.path.exists(ht.LEDGER):
            self.skipTest("ledger unavailable")
        led = pd.read_csv(ht.LEDGER, low_memory=False)
        v12 = led[led["model_tag"] == "xw+plat_consol_v12"]
        if not len(v12):
            self.skipTest("no v12 rows committed")
        g = ht.scored_rows(v12)
        self.assertEqual(len(g), 0)

    def test_report_lines_never_raise_and_always_name_the_gate(self):
        lines = ht.report_lines()
        self.assertTrue(lines)
        joined = " ".join(lines).lower()
        self.assertIn("pre-registered", joined)
        # A reader must never see a running total without a gate or a prior.
        self.assertTrue("gate" in joined or "prior" in joined)

    def test_the_report_carries_no_betting_recommendation(self):
        """Same rule the market verdict row follows: context, never a call."""
        joined = " ".join(ht.report_lines(
            _led(np.linspace(.05, .95, 40), lean_home=True,
                 home_won=True))).lower()
        for word in ("value bet", "recommend", "should bet", "edge found",
                     "profitable system"):
            self.assertNotIn(word, joined)

    def test_report_lines_survive_a_ledger_with_no_eligible_rows(self):
        empty = pd.DataFrame(columns=[
            "status", "game_date", "close_p_home", "xw_lean", "home",
            "full_home", "full_away", "close_home_ml", "close_away_ml"])
        lines = ht.report_lines(empty)
        self.assertTrue(any("0" in ln or "no " in ln.lower() for ln in lines))

    def test_report_lines_survive_a_ledger_missing_columns(self):
        lines = ht.report_lines(pd.DataFrame({"status": ["graded"]}))
        self.assertTrue(lines)

    def test_xw_net_is_not_required(self):
        """The rule reads a direction and a price, never the delta's magnitude.

        Requiring `xw_net` would silently drop rows the rule can decide, and
        would couple this arm to forward_test's eligibility for no reason.
        """
        led = _led([0.30], lean_home=True, home_won=True)
        self.assertNotIn("xw_net", led.columns)
        self.assertEqual(len(ht.scored_rows(led)), 1)


if __name__ == "__main__":
    unittest.main()
