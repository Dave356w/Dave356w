"""A game priced at exactly .500 has no favourite, and every surface here that
publishes an always-chalk record has to decide what to do about that.

Three answers were live in this repo on 2026-09-17, and two of them were
published side by side off the same 436 rows: `ledger_report.txt` said the
always-chalk control went 257-179 and `grades.html` said 256-180. The cause was
not a row-set difference -- the thing every control note on the site promises
against -- but a tie-break. `grade_leans` broke the tie toward HOME; build_site
broke it on `market_p >= .50`, the LEANED side's price, which on a pick'em
hands the row to whichever side the model picked. That makes the control agree
with the thing it controls by fiat, on exactly the near-pick'em games where
this repo's own measurements put most of the model's contribution.

The convention now lives once, in `market_backfill.chalk_is_home`, and these
assert the two properties that matter rather than the spelling:

  * the control's side does not depend on the model's lean;
  * the internal artifact and the public page publish the same record.

The third answer -- `p > .5`, dropping the game -- stays correct where it is,
on the favourite POOL of `market-calibration.html`. That tile is one
observation per game, so a game with no favourite can leave it and nothing is
left unpaired. A CONTROL cannot do that: a chalk record over n-2 beside a model
record over n is the defect the grades page already shipped once.
"""
import unittest

import numpy as np
import pandas as pd

import build_site
import delta_filter_test as dft
import grade_leans
import hybrid_test as ht
import hybrid_v2
from market_backfill import chalk_is_home, is_pickem


def _led(p_home, lean_home, home_won, xw_net=0.02):
    """Graded current-family rows with closing prices and a lean."""
    p_home = np.asarray(p_home, dtype=float)
    n = len(p_home)
    lean_home = np.broadcast_to(np.asarray(lean_home, dtype=bool), (n,))
    home_won = np.broadcast_to(np.asarray(home_won, dtype=bool), (n,))
    ml = np.where(p_home >= .5, -np.round(100 * p_home / (1 - p_home)),
                  np.round(100 * (1 - p_home) / p_home))
    opp = np.where(p_home >= .5, np.round(100 * p_home / (1 - p_home)),
                   -np.round(100 * (1 - p_home) / p_home))
    lean_won = np.where(lean_home, home_won, ~home_won)
    return pd.DataFrame({
        "status": "graded",
        "game_date": "2026-09-20",
        "game_pk": np.arange(n) + 1,
        "model_tag": sorted(build_site.RECORD_TAGS)[0],
        "home": "H", "away": "A",
        "xw_lean": np.where(lean_home, "H", "A"),
        "xw_net": np.broadcast_to(np.asarray(xw_net, dtype=float), (n,)),
        "xw_full": np.where(lean_won, "W", "L"),
        "full_home": np.where(home_won, 1, 0),
        "full_away": np.where(home_won, 0, 1),
        "close_p_home": p_home,
        "close_home_ml": ml, "close_away_ml": opp,
        "xw_f5": np.where(lean_won, "W", "L"),
    })


def _site_chalk(led):
    obs = build_site._lean_market_observations(led)
    a = build_site._lean_market_agg(
        obs, obs["won"].notna(), won="chalk_won", p="chalk_p",
        resid="chalk_resid", profit="chalk_profit")
    return a["w"], a["l"]


def _report_chalk(led):
    g = hybrid_v2.apply_rule(hybrid_v2.decidable(led))
    won = g["chalk_won"].astype(bool)
    return int(won.sum()), int(len(g) - won.sum())


class TheConventionHasOneHome(unittest.TestCase):
    def test_a_pickem_is_recognised_as_having_no_favourite(self):
        self.assertTrue(bool(is_pickem(0.5)))
        self.assertFalse(bool(is_pickem(0.5001)))
        self.assertFalse(bool(is_pickem(0.4999)))

    def test_every_chalk_derivation_agrees_on_a_pickem(self):
        """Not a grep for the spelling: each module is asked the question."""
        p = pd.Series([0.5, 0.62, 0.38])
        want = list(chalk_is_home(p))
        for mod, fn in (
                ("hybrid_test", lambda l: ht.apply_rule(ht.decidable(l))),
                ("hybrid_v2", lambda l: hybrid_v2.apply_rule(
                    hybrid_v2.decidable(l))),
                ("delta_filter_test", lambda l: dft.apply_filter(
                    dft.decidable(l)))):
            with self.subTest(module=mod):
                g = fn(_led([0.5, 0.62, 0.38], lean_home=True, home_won=True))
                # chalk backed home exactly where the shared rule says it does
                got = (g["chalk_p"].to_numpy(dtype=float)
                       == g["close_p_home"].to_numpy(dtype=float))
                self.assertEqual(list(got), want)


class TheControlDoesNotCopyTheModel(unittest.TestCase):
    """The property the old `market_p >= .50` form broke.

    Flipping the model's lean must leave the control untouched. On every other
    price it did; on a pick'em the control followed the lean, so the one place
    the convention could bite was the one place it was model-dependent.
    """

    def _chalk_won(self, lean_home):
        led = _led([0.5], lean_home=lean_home, home_won=True)
        obs = build_site._lean_market_observations(led)
        return float(obs["chalk_won"].iloc[0])

    def test_flipping_the_lean_does_not_move_the_chalk_control(self):
        self.assertEqual(self._chalk_won(True), self._chalk_won(False))

    def test_on_a_pickem_chalk_takes_home_whatever_the_model_did(self):
        for lean_home in (True, False):
            with self.subTest(lean_home=lean_home):
                # home won, so home-side chalk is a win either way
                self.assertEqual(self._chalk_won(lean_home), 1.0)

    def test_an_ordinary_price_is_unaffected_by_the_change(self):
        """The fix must move nothing except the pick'em rows.

        Home wins both. At .62 chalk is home and wins; at .38 chalk is away
        and loses -- and neither answer depends on which side was leaned.
        """
        led = _led([0.62, 0.38], lean_home=[True, False], home_won=[True, True])
        obs = build_site._lean_market_observations(led)
        self.assertEqual(list(obs["chalk_won"]), [1.0, 0.0])
        flipped = _led([0.62, 0.38], lean_home=[False, True],
                       home_won=[True, True])
        self.assertEqual(
            list(build_site._lean_market_observations(flipped)["chalk_won"]),
            [1.0, 0.0])


class TheTwoArtifactsAgree(unittest.TestCase):
    def test_the_page_and_the_report_publish_the_same_chalk_record(self):
        led = _led([0.5, 0.5, 0.62, 0.38, 0.55],
                   lean_home=[True, False, False, True, True],
                   home_won=[True, False, True, False, True])
        self.assertEqual(_site_chalk(led), _report_chalk(led))

    def test_they_agree_on_the_committed_ledger(self):
        """The live instance. 257-179 both sides, not 257-179 and 256-180."""
        led = pd.read_csv(grade_leans.LEDGER_PATH)
        self.assertEqual(_site_chalk(led), _report_chalk(
            led[led["model_tag"].isin(build_site.RECORD_TAGS)]))


class TheFootprintIsDisclosed(unittest.TestCase):
    """A record resting on a tie-break says how many rows are in that position.

    Keeping the row is right for a control and it is not free: those games were
    decided by a convention rather than by a price, and a reader cannot see
    that from the record alone.
    """

    def test_the_page_names_the_pickem_count_when_there_is_one(self):
        led = _led([0.5, 0.62, 0.38], lean_home=True, home_won=True)
        note = build_site._pickem_note(build_site._lean_market_observations(led))
        self.assertIn("1 of 3", note)
        self.assertIn("no favourite", note)

    def test_the_page_says_nothing_when_there_are_none(self):
        led = _led([0.62, 0.38], lean_home=True, home_won=True)
        self.assertEqual(
            build_site._pickem_note(build_site._lean_market_observations(led)),
            "")

    def test_the_band_block_names_it_too(self):
        led = _led([0.5, 0.62, 0.38], lean_home=True, home_won=True)
        body = "\n".join(grade_leans._fixed_magnitude_lines(led))
        self.assertIn("exactly .500", body)

    def test_the_band_block_says_nothing_when_there_are_none(self):
        led = _led([0.62, 0.38], lean_home=True, home_won=True)
        body = "\n".join(grade_leans._fixed_magnitude_lines(led))
        self.assertNotIn("exactly .500", body)


if __name__ == "__main__":
    unittest.main()
