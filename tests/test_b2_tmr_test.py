"""Candidate B2 registration, state reconstruction, and report tests.

The constants are intentionally literal. B2 is a pre-registered forward
candidate, so changing its rule in place is not maintenance; it is a new
candidate and requires a new registration.
"""

import unittest

import numpy as np
import pandas as pd

import b2_tmr_test as b2


def _row(game_pk, game_date, home, away, home_won, *, p_home=.5,
         model_tag="legacy", xw_lean=None, pregame_p_home=.5,
         pregame_home_ml=-110, pregame_away_ml=-110,
         close_home_ml=-110, close_away_ml=-110):
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home": home,
        "away": away,
        "status": "graded",
        "full_home": 1 if home_won else 0,
        "full_away": 0 if home_won else 1,
        "close_p_home": p_home,
        "close_home_ml": close_home_ml,
        "close_away_ml": close_away_ml,
        "model_tag": model_tag,
        "xw_lean": xw_lean,
        "pregame_p_home": pregame_p_home,
        "pregame_home_ml": pregame_home_ml,
        "pregame_away_ml": pregame_away_ml,
    }


def _positive_extreme_fixture(future_date="2026-09-19", *,
                              future_pregame=.55, xw_lean="HOT"):
    """HOT has +3.16 SD TMR; NEU is centered at zero after ten games."""
    rows = []
    pk = 1
    for i in range(10):
        day = f"2026-09-{i + 1:02d}"
        rows.append(_row(pk, day, "HOT", f"H{i}", True, p_home=.5))
        pk += 1
        rows.append(_row(pk, day, "NEU", f"N{i}", i % 2 == 0, p_home=.5))
        pk += 1

    rows.append(_row(
        pk, future_date, "HOT", "NEU", False, p_home=.55,
        model_tag=b2.BASE_MODEL_TAG, xw_lean=xw_lean,
        pregame_p_home=future_pregame,
        pregame_home_ml=-120, pregame_away_ml=110,
    ))
    return pd.DataFrame(rows)


def _reconstruction_fixture():
    """One historical reconstruction row where B2 and reconstructed v13 disagree."""
    led = _positive_extreme_fixture(
        future_date="2026-09-17", future_pregame=.80, xw_lean="HOT"
    )
    i = led.index[-1]
    led.loc[i, "model_tag"] = "xw+plat_consol_v12"
    led.loc[i, "v13_recon_basis"] = b2.V13_RECON_BASIS
    led.loc[i, "v13_lean_recon"] = "HOT"
    led.loc[i, "close_p_home"] = .55
    led.loc[i, "close_home_ml"] = -120
    led.loc[i, "close_away_ml"] = 110
    return led


class RegistrationFrozenTests(unittest.TestCase):
    def test_registration_constants_are_exactly_as_registered(self):
        self.assertEqual(b2.REGISTERED_ON, "2026-09-18")
        self.assertEqual(b2.BASE_MODEL_TAG, "xw+starter_blend_v13")
        self.assertEqual(b2.LOOKBACK_N, 10)
        self.assertAlmostEqual(b2.Z_THRESHOLD, 1.50, places=12)
        self.assertAlmostEqual(b2.STAKE, 1.0, places=12)
        self.assertEqual(b2.EARLY_CHECKPOINT, 20)
        self.assertEqual(b2.INTERMEDIATE_CHECKPOINT, 40)
        self.assertEqual(b2.SUBSTANTIVE_CHECKPOINT, 75)


class StateTests(unittest.TestCase):
    def test_positive_extreme_fades_hot_team_via_lower_raw_tmr(self):
        led = _positive_extreme_fixture()
        s = b2.attach_b2_state(led)
        row = s.iloc[-1]

        self.assertTrue(bool(row["b2_signal"]))
        self.assertEqual(row["b2_side"], "NEU")
        self.assertEqual(row["b2_trigger_type"], "positive_extreme")
        self.assertGreater(row["tmr10_z_home"], b2.Z_THRESHOLD)
        self.assertAlmostEqual(row["tmr10_z_away"], 0.0, places=12)
        self.assertGreater(row["tmr10_home"], row["tmr10_away"])

    def test_same_day_state_is_frozen_across_doubleheader(self):
        base = _positive_extreme_fixture(future_date="2026-09-11")
        first = base.iloc[:-1].copy()
        pk = int(first["game_pk"].max()) + 1
        dh = pd.DataFrame([
            _row(pk, "2026-09-11", "HOT", "NEU", True, p_home=.5),
            _row(pk + 1, "2026-09-11", "HOT", "NEU", False, p_home=.5),
        ])
        s = b2.attach_b2_state(pd.concat([first, dh], ignore_index=True))
        day = s[s["game_date"] == "2026-09-11"]

        self.assertEqual(len(day), 2)
        self.assertAlmostEqual(
            day.iloc[0]["tmr10_home"], day.iloc[1]["tmr10_home"], places=12
        )
        self.assertAlmostEqual(
            day.iloc[0]["tmr10_away"], day.iloc[1]["tmr10_away"], places=12
        )
        self.assertAlmostEqual(
            day.iloc[0]["tmr10_z_home"], day.iloc[1]["tmr10_z_home"], places=12
        )
        self.assertEqual(day.iloc[0]["b2_side"], day.iloc[1]["b2_side"])

    def test_ten_prior_priced_games_are_required_for_both_teams(self):
        led = _positive_extreme_fixture().iloc[:-2].copy()
        # Remove one NEU history game and then restore the future matchup.
        full = _positive_extreme_fixture()
        future = full.iloc[[-1]]
        s = b2.attach_b2_state(pd.concat([led, future], ignore_index=True))
        row = s.iloc[-1]
        self.assertFalse(bool(row["b2_signal"]))
        self.assertTrue(np.isnan(row["tmr10_away"]))


class ForwardScopeTests(unittest.TestCase):
    def test_registration_date_itself_is_never_scored(self):
        led = _positive_extreme_fixture(future_date=b2.REGISTERED_ON)
        g = b2.forward_rows(led)
        self.assertIsNotNone(g)
        self.assertEqual(len(g), 0)

    def test_only_actual_v13_rows_are_scored(self):
        led = _positive_extreme_fixture()
        led.loc[led.index[-1], "model_tag"] = "xw+plat_consol_v12"
        g = b2.forward_rows(led)
        self.assertEqual(len(g), 0)

    def test_saved_pregame_price_is_required_for_scoring_not_for_signal(self):
        led = _positive_extreme_fixture(future_pregame=np.nan)
        g = b2.forward_rows(led)
        self.assertEqual(len(g), 1)
        self.assertTrue(bool(g.iloc[0]["b2_signal"]))
        self.assertFalse(bool(g.iloc[0]["b2_scorable"]))

    def test_disagreement_scores_b2_and_unchanged_v13_on_same_game(self):
        led = _positive_extreme_fixture(xw_lean="HOT")
        g = b2.forward_rows(led)
        self.assertEqual(len(g), 1)
        row = g.iloc[0]

        self.assertEqual(row["b2_interaction"], "DISAGREE")
        self.assertTrue(bool(row["b2_won"]))
        self.assertFalse(bool(row["v13_won"]))
        self.assertGreater(row["b2_residual"], row["v13_residual"])
        self.assertGreater(row["b2_profit"], row["v13_profit"])


class ReconstructionScopeTests(unittest.TestCase):
    def test_reconstruction_rows_score_b2_against_reconstructed_v13(self):
        led = _reconstruction_fixture()
        r = b2.reconstruction_rows(led)
        self.assertEqual(len(r), 1)
        row = r.iloc[0]

        self.assertEqual(row["b2_interaction"], "DISAGREE")
        self.assertTrue(bool(row["b2_won"]))
        self.assertFalse(bool(row["v13_won"]))
        self.assertAlmostEqual(float(row["b2_p"]), .45, places=12)
        self.assertAlmostEqual(float(row["v13_p"]), .55, places=12)
        self.assertGreater(row["b2_residual"], row["v13_residual"])
        self.assertGreater(row["b2_profit"], row["v13_profit"])

    def test_reconstruction_rows_never_enter_forward_accumulator(self):
        led = _reconstruction_fixture()
        self.assertEqual(len(b2.reconstruction_rows(led)), 1)
        self.assertEqual(len(b2.forward_rows(led)), 0)


class ReportTests(unittest.TestCase):
    def test_report_names_rule_price_basis_and_forward_scope(self):
        lines = b2.report_lines(_positive_extreme_fixture(xw_lean="HOT"))
        joined = "\n".join(lines)
        self.assertIn("pre-registered Candidate B2", joined)
        self.assertIn("lower raw TMR10", joined)
        self.assertIn("prior-game closing close_p_home", joined)
        self.assertIn("saved pregame_p_home", joined)
        self.assertIn("forward scoring — actual xw+starter_blend_v13 only", joined)
        self.assertIn("RECONSTRUCTION DIAGNOSTIC", joined)
        self.assertIn("REGISTERED FORWARD ACCUMULATION", joined)
        self.assertIn("PRIMARY", joined)
        self.assertIn("DISAGREE", joined)
        self.assertIn("paired gain", joined)
        self.assertIn("reconstruction rows above never enter", joined)

    def test_report_survives_missing_columns(self):
        lines = b2.report_lines(pd.DataFrame({"status": ["graded"]}))
        self.assertTrue(lines)
        self.assertIn("not scored", " ".join(lines))

    def test_report_survives_no_forward_rows(self):
        led = _positive_extreme_fixture(future_date=b2.REGISTERED_ON)
        lines = b2.report_lines(led)
        self.assertIn("nothing to score yet", " ".join(lines))


if __name__ == "__main__":
    unittest.main()
