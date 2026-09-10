"""A started game's card shows the locked pregame row, not a rebuild of it.

Display-only: these guard which artifact the card reads, never the lean math.
The defect they exist for is a card publishing a side nobody could have taken
-- 2026-09-10 TEX@SEA locked at net +0.000191 (SEA) and rebuilt four hours
after first pitch at -0.004925 (TEX), because Texas's batting order changed
after the snapshot.
"""
import re
import unittest

import numpy as np
import pandas as pd

import build_site as b


LOCK = "2026-09-10T19:30:57.949581+00:00"
START = "2026-09-10T20:10:00Z"
AFTER = "2026-09-10T23:20:54.658902+00:00"
BEFORE = "2026-09-10T18:00:00.000000+00:00"


def _dump_side(side, edge, snapshot):
    """One side of a dump row, carrying only what the card actually reads."""
    return pd.Series({
        "game_pk": 1, "side": side, "pitcher": f"{side} SP",
        "opp_team": "Seattle Mariners" if side == "away" else "Texas Rangers",
        "edge_xwOBA": edge, "starter_xwOBA": 0.400, "pit_xwOBA": 0.400,
        "opp_xwOBA_vs_sp": 0.400, "opp_xwOBA": 0.400,
        "opp_xwOBA_neutral": 0.400, "bullpen_xwOBA": 0.400,
        "expected_sp_ip": 9.0, "platoon_delta_sp": 0.099,
        "starter_rate_basis": "live", "starter_rate_bf": 999,
        "pitching_basis": "bullpen_sequential", "opener": False,
        "snapshot_utc": snapshot, "scheduled_start_utc": START,
    })


def _ledger_row(**over):
    """A locked v12 ledger row for the same game, with different numbers.

    The frozen values are deliberately nothing like the live ones above, so a
    test can only pass by actually reading the ledger.
    """
    row = {
        "game_pk": 1, "game_date": "2026-09-10", "away": "TEX", "home": "SEA",
        "model_tag": b.MODEL_TAG, "lock_status": "pregame", "status": "graded",
        "snapshot_utc": LOCK, "scheduled_start_utc": START,
        "xw_net": 0.000191, "xw_lean": "SEA",
        "edge_xwoba_away": -0.013789, "edge_xwoba_home": -0.013980,
        "starter_xwoba_away": 0.295322, "starter_xwoba_home": 0.301057,
        "opp_xwoba_vs_sp_away": 0.323258, "opp_xwoba_vs_sp_home": 0.313618,
        "opp_xwoba_neutral_away": 0.321863, "opp_xwoba_neutral_home": 0.313380,
        "bullpen_xwoba_away": 0.291026, "bullpen_xwoba_home": 0.302740,
        "expected_sp_ip_away": 5.0798, "expected_sp_ip_home": 5.3434,
        "platoon_delta_sp_away": 0.021, "platoon_delta_sp_home": 0.021,
        "sp_rate_basis_away": "season", "sp_rate_basis_home": "season",
        "sp_rate_bf_away": 700, "sp_rate_bf_home": 700,
        "pitching_basis_away": "bullpen_sequential",
        "pitching_basis_home": "bullpen_sequential",
        "opener_away": False, "opener_home": False,
        "pregame_away_ml": 100.0, "pregame_home_ml": -120.0,
        "pregame_p_home": 0.5217391304347826,
    }
    row.update(over)
    return pd.Series(row)


LIVE_ODDS = dict(away_ml=-140, home_ml=118, open_away_ml=109,
                 open_home_ml=-132, total=8.5, p_home=0.44)


class PostHocDetectionTests(unittest.TestCase):
    """`game_is_post_hoc` is the ledger's own lock rule, per game."""

    def test_a_snapshot_after_first_pitch_is_post_hoc(self):
        self.assertTrue(b.game_is_post_hoc(_dump_side("away", 0.0, AFTER)))

    def test_a_snapshot_before_first_pitch_is_not(self):
        self.assertFalse(b.game_is_post_hoc(_dump_side("away", 0.0, BEFORE)))

    def test_it_agrees_with_the_graders_lock_rule_at_every_offset(self):
        # The card must freeze at exactly the instant the ledger would refuse
        # the row this build computed -- otherwise there is a window in which
        # the page publishes what the ledger will not accept, which is the
        # whole defect. Compared against grade_leans' own function rather than
        # a restatement of it.
        import grade_leans as gl
        for mins in (-120, -1, 0, 1, 120):
            snap = (pd.Timestamp(START) + pd.Timedelta(minutes=mins)).isoformat()
            with self.subTest(minutes=mins):
                self.assertEqual(
                    b.game_is_post_hoc(_dump_side("away", 0.0, snap)),
                    gl._lock_status(snap, START) != "pregame")

    def test_unknowable_stays_live(self):
        # dump_path's precedent: where the answer is not knowable, keep the
        # behaviour every existing surface already has.
        self.assertFalse(b.game_is_post_hoc(_dump_side("away", 0.0, None)))
        row = _dump_side("away", 0.0, AFTER)
        row["scheduled_start_utc"] = "not a timestamp"
        self.assertFalse(b.game_is_post_hoc(row))


class FreezeTests(unittest.TestCase):
    def test_the_frozen_net_is_the_ledgers_net(self):
        # The live rows below say TEX; the ledger says SEA. Freezing has to
        # reproduce the ledger's net exactly, not merely land on its side.
        a = _dump_side("away", -0.013679, AFTER)
        h = _dump_side("home", -0.008754, AFTER)
        self.assertLess(a["edge_xwOBA"] - h["edge_xwOBA"], 0)          # live: TEX
        fa, fh, _, _ = b.freeze_to_lock(a, h, _ledger_row(), None)
        self.assertAlmostEqual(fa["edge_xwOBA"] - fh["edge_xwOBA"],
                               0.000191, places=12)                     # locked: SEA

    def test_every_rate_the_card_renders_comes_from_the_ledger(self):
        fa, fh, _, _ = b.freeze_to_lock(_dump_side("away", -0.013679, AFTER),
                                        _dump_side("home", -0.008754, AFTER),
                                        _ledger_row(), None)
        led = _ledger_row()
        for dump_col, stem in b._FROZEN_SIDE_COLS:
            for side, row in (("away", fa), ("home", fh)):
                with self.subTest(col=dump_col, side=side):
                    self.assertEqual(row[dump_col], led[f"{stem}_{side}"])

    def test_the_cards_fallback_reads_are_frozen_too(self):
        # `mk()` reads pit_xwOBA / opp_xwOBA when the columns above them are
        # absent. If only the primaries were frozen a live value would sit one
        # `or` away from a card claiming to be pregame.
        fa, _, _, _ = b.freeze_to_lock(_dump_side("away", -0.013679, AFTER),
                                       _dump_side("home", -0.008754, AFTER),
                                       _ledger_row(), None)
        self.assertEqual(fa["pit_xwOBA"], _ledger_row()["starter_xwoba_away"])
        self.assertEqual(fa["opp_xwOBA"], _ledger_row()["opp_xwoba_vs_sp_away"])

    def test_the_freeze_does_not_mutate_the_dump_rows(self):
        # Display-only means display-only: the frame the dump is written from
        # must still hold the live values after the page is rendered.
        a = _dump_side("away", -0.013679, AFTER)
        h = _dump_side("home", -0.008754, AFTER)
        b.freeze_to_lock(a, h, _ledger_row(), None)
        self.assertEqual(a["edge_xwOBA"], -0.013679)
        self.assertEqual(h["edge_xwOBA"], -0.008754)

    def test_a_missing_edge_refuses_the_freeze_rather_than_half_doing_it(self):
        # A card frozen on one side and live on the other publishes a net
        # neither artifact ever computed.
        for side in ("away", "home"):
            with self.subTest(side=side):
                self.assertIsNone(b.freeze_to_lock(
                    _dump_side("away", -0.013679, AFTER),
                    _dump_side("home", -0.008754, AFTER),
                    _ledger_row(**{f"edge_xwoba_{side}": np.nan}), None))

    def test_no_locked_row_refuses_the_freeze(self):
        self.assertIsNone(b.freeze_to_lock(_dump_side("away", 0.0, AFTER),
                                           _dump_side("home", 0.0, AFTER),
                                           None, None))


class FrozenMarketTests(unittest.TestCase):
    def test_the_price_is_the_locked_one_not_the_live_one(self):
        _, _, odds, _ = b.freeze_to_lock(_dump_side("away", 0.0, AFTER),
                                         _dump_side("home", 0.0, AFTER),
                                         _ledger_row(), LIVE_ODDS)
        self.assertEqual((odds["away_ml"], odds["home_ml"]), (100, -120))
        self.assertAlmostEqual(odds["p_home"], 0.5217391304347826)

    def test_moneylines_render_as_integers(self):
        # Regression: the ledger holds floats and `_fmt_ml` str()s what it
        # gets, so an uncoerced freeze printed "-120.0" beside the live path's
        # "-120" on the same page.
        _, _, odds, _ = b.freeze_to_lock(_dump_side("away", 0.0, AFTER),
                                         _dump_side("home", 0.0, AFTER),
                                         _ledger_row(), LIVE_ODDS)
        self.assertEqual(b._fmt_ml(odds["home_ml"]), "-120")
        self.assertEqual(b._fmt_ml(odds["away_ml"]), "+100")

    def test_the_total_is_dropped_rather_than_carried_live(self):
        # There is no pregame total in the ledger. A closing total beside two
        # locked moneylines is the mixed basis this freeze exists to remove.
        _, _, odds, _ = b.freeze_to_lock(_dump_side("away", 0.0, AFTER),
                                         _dump_side("home", 0.0, AFTER),
                                         _ledger_row(), LIVE_ODDS)
        self.assertIsNone(odds["total"])

    def test_the_opens_are_kept(self):
        # An opener is a pregame quantity whichever build reads it, and the
        # ledger's own open_* are attached post-hoc so they are null on
        # exactly the rows that need them here.
        _, _, odds, _ = b.freeze_to_lock(_dump_side("away", 0.0, AFTER),
                                         _dump_side("home", 0.0, AFTER),
                                         _ledger_row(), LIVE_ODDS)
        self.assertEqual((odds["open_away_ml"], odds["open_home_ml"]), (109, -132))

    def test_a_row_with_no_locked_market_yields_no_odds(self):
        _, _, odds, _ = b.freeze_to_lock(
            _dump_side("away", 0.0, AFTER), _dump_side("home", 0.0, AFTER),
            _ledger_row(pregame_away_ml=np.nan, pregame_home_ml=np.nan,
                        pregame_p_home=np.nan), LIVE_ODDS)
        self.assertIsNone(odds)


class LockNoteTests(unittest.TestCase):
    @staticmethod
    def _text(g):
        return re.sub("<[^>]+>", " ", b._pregame_lock_note(g))

    def test_a_pregame_card_carries_no_note(self):
        # Before first pitch the live build IS the pregame view; a note on
        # every card of a normal slate would be noise.
        self.assertEqual(b._pregame_lock_note(
            dict(post_hoc=False, frozen=False, locked_at=None)), "")

    def test_a_frozen_card_states_the_lock_and_excepts_the_lineup(self):
        txt = self._text(dict(post_hoc=True, frozen=True, locked_at="12:30 PM"))
        self.assertIn("12:30 PM", txt)
        self.assertIn("before first pitch", txt)
        # The one thing the freeze cannot reach: the ledger stores no
        # per-hitter rows, so the lineup list stays live and must say so.
        self.assertIn("lineup", txt)

    def test_a_started_game_with_no_lock_says_it_is_a_rebuild(self):
        # The `_lock_note` rule: never assert coverage the artifact cannot
        # substantiate. Silence here would leave a rebuild reading as a
        # pregame publication, which is the defect with an extra step.
        txt = self._text(dict(post_hoc=True, frozen=False, locked_at=None))
        self.assertIn("rebuild", txt)
        self.assertNotIn("Locked", txt)

    def test_an_unparseable_stamp_does_not_read_as_a_missing_lock(self):
        # The note keys on whether the freeze HAPPENED, never on whether the
        # clock formatted -- those are different facts.
        txt = self._text(dict(post_hoc=True, frozen=True, locked_at=None))
        self.assertIn("Locked", txt)
        self.assertNotIn("No pregame lock", txt)

    def test_the_clock_parses_the_snapshot_format_the_ledger_actually_holds(self):
        # Regression: `_game_time_pt` parses ESPN's `...Z` schedule format and
        # returns None for a full ISO stamp with microseconds, which is what
        # snapshot_utc is -- so the note silently lost its time.
        self.assertEqual(b._lock_clock(LOCK), "12:30 PM")
        self.assertIsNone(b._lock_clock("nonsense"))


class CardIntegrationTests(unittest.TestCase):
    """The whole path, on the committed slate that produced the defect."""

    def setUp(self):
        self.xw = pd.read_csv("data/leans_2026-09-10_xw.csv")
        slate = self.xw.drop_duplicates("game_pk")[
            ["game_pk", "game_datetime_utc", "matchup"]].copy()
        slate["away_abbrev"] = slate.matchup.str.split(" @ ").str[0]
        slate["home_abbrev"] = slate.matchup.str.split(" @ ").str[1]
        for c in ("away_team", "home_team", "away_team_id", "home_team_id",
                  "venue", "status", "away_score", "home_score"):
            slate[c] = None
        slate["double_header"], slate["game_number"] = "N", 1
        slate["abstract_state"] = "Final"
        self.slate = slate
        self.locked = b.locked_pregame_rows(slate_date="2026-09-10")

    def _games(self, **kw):
        return b._df_to_combined_games(
            self.xw, pd.DataFrame(), pd.DataFrame(), slate_df=self.slate,
            league_baseline={"xwOBA": .312}, **kw)

    def test_locked_rows_are_found_for_the_slate(self):
        self.assertTrue(self.locked)
        for row in self.locked.values():
            self.assertTrue(str(row["lock_status"]).startswith("pregame"))

    def test_every_started_card_reproduces_its_ledger_lean(self):
        led = pd.read_csv("data/mlb_lean_ledger.csv")
        led = led[led.game_date == "2026-09-10"].set_index("game_pk")
        started = 0
        for g in self._games(odds={}, locked=self.locked):
            if not g.get("frozen"):
                continue
            started += 1
            r = led.loc[g["game_pk"]]
            net = g["away"]["xw_edge"] - g["home"]["xw_edge"]
            with self.subTest(game=g["game_pk"]):
                self.assertAlmostEqual(net, r["xw_net"], places=12)
                self.assertEqual(r["home"] if net > 0 else r["away"],
                                 r["xw_lean"])
        self.assertGreater(started, 0)

    def test_it_fixes_the_case_it_was_written_for(self):
        # TEX@SEA: the rebuild says TEX, the ledger says SEA.
        g = next(x for x in self._games(odds={}, locked=self.locked)
                 if x["game_pk"] == 823088)
        self.assertTrue(g["frozen"])
        self.assertGreater(g["away"]["xw_edge"] - g["home"]["xw_edge"], 0)  # SEA

    def test_a_live_market_cannot_move_a_frozen_cards_selection(self):
        # -140/+118 puts the leaned side under the hybrid threshold, which on
        # a live card flips the published branch to a fade.
        live = {823088: LIVE_ODDS}
        g = next(x for x in self._games(odds=live, locked=self.locked)
                 if x["game_pk"] == 823088)
        html = b.cmb_card(g, None, {})
        self.assertIn("-120", html)
        self.assertNotIn("+118", html)
        self.assertIn(b.hybrid_public_label("FOLLOW"), html)

    def test_a_game_that_has_not_started_is_untouched(self):
        # The frozen path must not change a normal pregame slate at all.
        before = self._games(odds={}, locked={})
        after = self._games(odds={}, locked=self.locked)
        for x, y in zip(before, after):
            if y.get("post_hoc"):
                continue
            with self.subTest(game=y.get("game_pk")):
                self.assertFalse(y.get("frozen"))
                self.assertEqual(b.cmb_card(x, None, {}), b.cmb_card(y, None, {}))

    def test_a_started_game_with_no_locked_row_still_renders_and_warns(self):
        g = next(x for x in self._games(odds={}, locked={})
                 if x["game_pk"] == 823088)
        self.assertTrue(g["post_hoc"])
        self.assertFalse(g["frozen"])
        self.assertIn("rebuild", b.cmb_card(g, None, {}))


if __name__ == "__main__":
    unittest.main()
