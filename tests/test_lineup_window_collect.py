"""The event map and the prefix logic, verified where they can be.

StatsAPI is unreachable from the dev sandbox and from CI's test job, so these
run on constructed payloads. That is the point rather than a compromise: the
mapping from `eventType` to wOBA components is the step that fails SILENTLY on
live data -- a mis-mapped event shifts every rate a little and reads as a
finding -- so it is pinned here, and the collector's own reconciliation against
the box score is what catches whatever a constructed payload could not
anticipate.
"""
import unittest

import pandas as pd

import lineup_window_collect as lw


def _play(event_type, half="top", batter=1, pitcher=100, idx=0, inning=1):
    return {"result": {"eventType": event_type},
            "about": {"halfInning": half, "atBatIndex": idx, "inning": inning},
            "matchup": {"batter": {"id": batter}, "pitcher": {"id": pitcher}}}


def _feed(plays, away_sp=100, home_sp=200):
    def team(sp_id):
        return {"players": {
            # A reliever listed FIRST, to catch anything taking position for role.
            "ID9": {"person": {"id": 999},
                    "stats": {"pitching": {"gamesStarted": 0}}},
            f"ID{sp_id}": {"person": {"id": sp_id},
                           "stats": {"pitching": {"gamesStarted": 1}}},
        }}
    return {"liveData": {"plays": {"allPlays": plays},
                         "boxscore": {"teams": {"away": team(away_sp),
                                                "home": team(home_sp)}}}}


class EventMapTests(unittest.TestCase):
    def test_ab_is_derived_by_identity_and_excludes_the_right_events(self):
        """PA - bb - hbp - sf - sh - ci = ab. A sac bunt and a catcher
        interference are plate appearances that are NOT at-bats; counting
        either as an out would inflate the denominator on every game."""
        rows = [{"cat": c} for c in
                ("out", "1b", "bb", "ibb", "hbp", "sf", "sh", "ci", "hr")]
        c = lw.components(rows)
        self.assertEqual(c["pa"], 9)
        # 9 PA - 2 walks - 1 hbp - 1 sf - 1 sh - 1 ci = 3 at-bats
        self.assertEqual(c["ab"], 3)

    def test_intentional_walk_counts_in_both_bb_and_ibb(self):
        """The box score's `baseOnBalls` INCLUDES intentional ones, and
        `woba_from_components` derives unintentional as bb - ibb. Counting an
        IBB only in ibb would make that difference negative."""
        c = lw.components([{"cat": "ibb"}, {"cat": "walk"} if False else {"cat": "bb"}])
        self.assertEqual(c["bb"], 2)
        self.assertEqual(c["ibb"], 1)

    def test_extra_base_hits_count_in_h_as_well_as_their_own_column(self):
        c = lw.components([{"cat": "2b"}, {"cat": "3b"}, {"cat": "hr"},
                           {"cat": "1b"}])
        self.assertEqual(c["h"], 4)
        self.assertEqual((c["2b"], c["3b"], c["hr"]), (1, 1, 1))

    def test_non_pa_events_are_dropped_and_unknown_ones_are_reported(self):
        """An unknown eventType must reach the report. Treated as an out it
        would move every rate a little, which is the failure this module is
        built to make loud."""
        rows, unmapped = lw.plate_appearances(_feed([
            _play("single"), _play("stolen_base_2b"), _play("wild_pitch"),
            _play("some_new_event_2027"),
        ]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(unmapped, ["some_new_event_2027"])

    def test_every_mapped_category_is_one_components_understands(self):
        self.assertEqual(set(lw._EVENT.values()),
                         {"1b", "2b", "3b", "hr", "bb", "ibb", "hbp",
                          "sf", "sh", "ci", "out"})

    def test_the_two_event_sets_are_disjoint(self):
        """An event in both maps would be counted or dropped depending on
        which lookup ran first."""
        self.assertEqual(set(lw._EVENT) & lw._NOT_A_PA, set())


class SideAndStarterTests(unittest.TestCase):
    def test_top_of_the_inning_is_the_away_team_batting(self):
        """Reversed, this pairs every prediction with the wrong offense and
        yields a plausible near-zero slope rather than an error -- the same
        cross that `paired_components` refuses to let a caller do."""
        rows, _ = lw.plate_appearances(_feed([
            _play("single", half="top"), _play("double", half="bottom")]))
        self.assertEqual(rows[0]["bat_side"], "away")
        self.assertEqual(rows[0]["pit_side"], "home")
        self.assertEqual(rows[1]["bat_side"], "home")

    def test_starter_is_identified_by_gamesStarted_not_list_position(self):
        """An opener is listed first and IS the starter; a promoted reliever is
        listed first and is not. The box score's own flag settles it."""
        feed = _feed([_play("single", half="top", pitcher=200),
                      _play("single", half="top", pitcher=999)],
                     home_sp=200)
        rows, _ = lw.plate_appearances(feed)
        self.assertTrue(rows[0]["vs_starter"])
        self.assertFalse(rows[1]["vs_starter"])


class ValidationTests(unittest.TestCase):
    def _consistent_feed(self):
        plays = [_play("single", half="top", batter=b, pitcher=200)
                 for b in range(1, 4)]
        plays += [_play("field_out", half="top", batter=b, pitcher=200)
                  for b in range(4, 7)]
        feed = _feed(plays, home_sp=200)
        feed["liveData"]["boxscore"]["teams"]["away"]["teamStats"] = {
            "batting": {"plateAppearances": 6, "atBats": 6, "hits": 3,
                        "doubles": 0, "triples": 0, "homeRuns": 0,
                        "baseOnBalls": 0, "intentionalWalks": 0,
                        "hitByPitch": 0, "sacFlies": 0}}
        feed["liveData"]["boxscore"]["teams"]["home"]["teamStats"] = {
            "batting": {"plateAppearances": 0, "atBats": 0, "hits": 0,
                        "doubles": 0, "triples": 0, "homeRuns": 0,
                        "baseOnBalls": 0, "intentionalWalks": 0,
                        "hitByPitch": 0, "sacFlies": 0}}
        return feed

    def test_a_faithful_reconstruction_reconciles(self):
        feed = self._consistent_feed()
        rows, _ = lw.plate_appearances(feed)
        self.assertEqual(lw.validate(1, feed, rows, None),
                         {"map": [], "ledger": []})

    def test_a_mismapped_event_is_caught_by_the_box_score(self):
        """The reconciliation is the safety net for exactly the thing a
        constructed test cannot foresee: an event this map gets wrong."""
        feed = self._consistent_feed()
        rows, _ = lw.plate_appearances(feed)
        rows[0]["cat"] = "out"          # pretend the map said out, not single
        fails = lw.validate(1, feed, rows, None)
        self.assertTrue(fails["map"], fails)
        self.assertFalse(fails["ledger"], fails)

    def test_starter_window_length_is_checked_against_the_ledger(self):
        """Totals reconcile even if the SEQUENCE is shuffled; the starter's
        batters-faced count is what pins the ordering."""
        feed = self._consistent_feed()
        rows, _ = lw.plate_appearances(feed)
        led = {"act_sp_bf_home": 5.0}   # pbp says 6
        fails = lw.validate(1, feed, rows, led)
        self.assertTrue(any("SP bf" in f for f in fails["ledger"]), fails)
        self.assertFalse(fails["map"], fails)

    def test_a_scoring_revision_is_a_ledger_failure_not_a_map_failure(self):
        """The case that actually fired: the PBP agrees with its own box score
        and the stored row is older. Naming that a map defect would blame this
        module for someone else's stat correction -- and, worse, would let a
        real map defect hide among them."""
        feed = self._consistent_feed()
        rows, _ = lw.plate_appearances(feed)
        led = {"act_woba_away": 0.999}          # stale stored actual
        fails = lw.validate(1, feed, rows, led)
        self.assertFalse(fails["map"], fails)
        self.assertTrue(any("woba" in f for f in fails["ledger"]), fails)


class WindowTests(unittest.TestCase):
    def _pa_frame(self, n_away):
        rows = []
        for k in range(1, n_away + 1):
            rows.append({"game_pk": 7, "bat_side": "away", "pa_seq": k,
                         "batter_id": (k - 1) % 9 + 1, "cat": "out",
                         "reconciled": True})
        return pd.DataFrame(rows)

    def _led(self):
        return pd.DataFrame([{"game_date": "2026-08-29", "gamePk": 7,
                              "opp_xwoba_neutral_home": 0.320,
                              "act_sp_bf_home": 20.0}])

    def test_the_window_is_exactly_two_turns_through_the_order(self):
        """18 is not an arbitrary number: it is why the fixed-window
        prediction would be the UNWEIGHTED nine-hitter mean."""
        w = lw.windows(self._pa_frame(30), self._led(), "2026-08-29")
        row = w[w.bat_side == "away"].iloc[0]
        self.assertEqual(row["distinct_batters_in_window"], 9)
        self.assertEqual(lw.FIXED_WINDOW, 18)

    def test_a_short_game_yields_no_fixed_window_rather_than_a_partial_one(self):
        """A rain-shortened side with 15 PA must not be scored as though it had
        a full window -- that would silently reintroduce a variable window,
        which is the exact thing being tested against."""
        w = lw.windows(self._pa_frame(15), self._led(), "2026-08-29")
        self.assertTrue(pd.isna(w[w.bat_side == "away"].iloc[0]["act_fixed"]))

    def test_unreconciled_games_never_reach_the_windows(self):
        pa = self._pa_frame(30)
        pa["reconciled"] = False
        self.assertTrue(lw.windows(pa, self._led(), "2026-08-29").empty)

    def test_the_prediction_takes_the_cross(self):
        """The away offense is predicted by `opp_xwoba_neutral_home` -- the
        column on the HOME pitcher's side, since that is who it faces."""
        w = lw.windows(self._pa_frame(30), self._led(), "2026-08-29")
        self.assertAlmostEqual(w[w.bat_side == "away"].iloc[0]["pred"], 0.320)


class NoSweepTests(unittest.TestCase):
    def test_the_window_is_a_constant_not_an_argument(self):
        """Exposing the window as a CLI flag is how a pre-registered test
        becomes the search this repo has twice been fooled by. The registration
        fixed 18; nothing here may offer a second one."""
        import inspect
        src = inspect.getsource(lw)
        self.assertNotIn("--window", src)
        self.assertNotIn("add_argument(\"--fixed", src)
        # windows() must read the module constant, not take a parameter.
        self.assertEqual(list(inspect.signature(lw.windows).parameters),
                         ["pa", "led", "date"])


if __name__ == "__main__":
    unittest.main()
