"""The persisted Hybrid selection must name a club the ledger can recognise.

Two club-abbreviation namespaces meet in this repo. `build_site.ABBR` follows
StatsAPI, where Arizona is `AZ`; `grade_leans.ABBR` persists ESPN/ledger-style
`ARI`. Every other team field crosses that boundary as a full team name and is
abbreviated by grade_leans' own map, so the two can only disagree where an
abbreviation itself is written into the ledger -- which `hybrid_selection` does.

It disagreed. `attach_hybrid_snapshot` wrote `AZ`, `grade_leans` copied it
through verbatim, and the consequences were both silent:

  * `_wlt` compared it against the ledger's `ARI`/opponent, matched neither,
    fell through to the else and graded the row `L` WHATEVER the score -- into
    a column that is immutable once written;
  * `hybrid_test.scored_rows` requires the selection to name one of the two
    clubs, so the row vanished from the registered forward test's denominator
    without appearing anywhere as a rejection.

These tests pin the RULE rather than the instance. Asserting `AZ -> ARI` alone
would pass while a thirty-first divergence went unnoticed, so the first test
walks every club in both maps. `test_ledger_abbr_is_the_only_alias` is the one
that would fail if a future club diverged.
"""

import math
import unittest

import numpy as np
import pandas as pd

import build_site as B
import grade_leans as G
import hybrid_test as ht


class NamespaceAgreementTests(unittest.TestCase):
    def test_every_club_the_model_can_emit_is_a_club_the_ledger_knows(self):
        ledger_abbrs = set(G.ABBR.values())
        for full_name, model_abbr in B.ABBR.items():
            with self.subTest(club=full_name):
                self.assertIn(B.ledger_abbr(model_abbr), ledger_abbrs)

    def test_ledger_abbr_agrees_with_grade_leans_club_by_club(self):
        """The translation must land on the SAME club, not merely a valid one."""
        for full_name, model_abbr in B.ABBR.items():
            with self.subTest(club=full_name):
                self.assertEqual(B.ledger_abbr(model_abbr), G.ABBR[full_name])

    def test_ledger_abbr_is_the_only_alias(self):
        """Arizona is the sole divergence; a new one must be added explicitly.

        This is the test that fails if StatsAPI or the grader renames a club.
        It is not decoration: the whole defect was one club, silently.
        """
        diverging = {n for n, a in B.ABBR.items() if a != G.ABBR[n]}
        self.assertEqual(diverging, {"Arizona Diamondbacks"})

    def test_ledger_abbr_leaves_an_already_ledger_shaped_abbr_alone(self):
        for abbr in sorted(set(G.ABBR.values())):
            with self.subTest(abbr=abbr):
                self.assertEqual(B.ledger_abbr(abbr), abbr)

    def test_the_two_maps_cover_the_same_clubs(self):
        self.assertEqual(set(B.ABBR), set(G.ABBR))


def _frame(home_full, away_full, home_edge, away_edge):
    """Two dump rows for one game, in the shape attach_hybrid_snapshot reads.

    `opp_team` is the OPPONENT, so the away row names the home club.
    """
    return pd.DataFrame([
        {"game_pk": 1, "side": "away", "opp_team": home_full,
         "edge_xwOBA": home_edge},
        {"game_pk": 1, "side": "home", "opp_team": away_full,
         "edge_xwOBA": away_edge},
    ])


_ODDS = {1: {"p_home": 0.60, "home_ml": -150, "away_ml": 130}}
_ODDS_FADE = {1: {"p_home": 0.62, "home_ml": -163, "away_ml": 137}}


class AttachHybridSnapshotNamespaceTests(unittest.TestCase):
    def test_an_arizona_selection_is_persisted_in_the_ledger_namespace(self):
        # Arizona at home, home edge higher -> the lean is Arizona, priced .60,
        # so the rule FOLLOWs and selects Arizona.
        f = B.attach_hybrid_snapshot(
            _frame("Arizona Diamondbacks", "Philadelphia Phillies", 0.02, 0.01),
            _ODDS, "2026-09-02T18:00Z")
        self.assertEqual(f["hybrid_action"].iloc[0], "FOLLOW")
        self.assertEqual(f["hybrid_selection"].iloc[0], "ARI")
        self.assertNotIn("AZ", set(f["hybrid_selection"]))

    def test_an_arizona_selection_reached_by_a_fade_is_translated_too(self):
        # Arizona at home; the lean is the away club, priced 1 - .62 = .38 < the
        # threshold, so the rule fades onto Arizona.
        f = B.attach_hybrid_snapshot(
            _frame("Arizona Diamondbacks", "Philadelphia Phillies", 0.01, 0.02),
            _ODDS_FADE, "2026-09-02T18:00Z")
        self.assertEqual(f["hybrid_action"].iloc[0], "FADE")
        self.assertEqual(f["hybrid_selection"].iloc[0], "ARI")

    def test_the_selected_price_still_matches_the_selected_club(self):
        """Translating the club must not shift which side's money line is stored.

        `pick` is compared against the model-namespace `home`/`away` to choose
        the price, so the translation has to happen after that and not before.
        """
        f = B.attach_hybrid_snapshot(
            _frame("Arizona Diamondbacks", "Philadelphia Phillies", 0.02, 0.01),
            _ODDS, "2026-09-02T18:00Z")
        self.assertEqual(float(f["hybrid_ml"].iloc[0]), -150.0)
        self.assertAlmostEqual(float(f["hybrid_p"].iloc[0]), 0.60, places=10)

    def test_a_non_arizona_selection_is_unchanged(self):
        f = B.attach_hybrid_snapshot(
            _frame("Philadelphia Phillies", "Arizona Diamondbacks", 0.02, 0.01),
            _ODDS, "2026-09-02T18:00Z")
        self.assertEqual(f["hybrid_selection"].iloc[0], "PHI")

    def test_every_persisted_selection_names_a_club_grade_leans_knows(self):
        """Sweep both home/away assignments for all 30 clubs against both branches."""
        ledger_abbrs = set(G.ABBR.values())
        clubs = sorted(B.ABBR)
        for club in clubs:
            other = "Chicago Cubs" if club != "Chicago Cubs" else "Miami Marlins"
            for odds in (_ODDS, _ODDS_FADE):
                for edges in ((0.02, 0.01), (0.01, 0.02)):
                    f = B.attach_hybrid_snapshot(
                        _frame(club, other, *edges), odds, "2026-09-02T18:00Z")
                    pick = f["hybrid_selection"].iloc[0]
                    with self.subTest(club=club, edges=edges):
                        self.assertIn(pick, ledger_abbrs)


class WltNeverFabricatesAGradeTests(unittest.TestCase):
    def test_a_selection_naming_neither_club_is_not_a_loss(self):
        # The exact shape of the defect: an untranslated Arizona selection.
        self.assertIsNone(G._wlt("AZ", "PHI", "ARI", 1, 7, False))
        self.assertIsNone(G._wlt("AZ", "PHI", "ARI", 7, 1, False))

    def test_the_old_behaviour_would_have_graded_a_win_as_a_loss(self):
        """Pins WHY the guard exists, so it cannot be removed as redundant.

        Reproduces the pre-fix expression directly. If this ever stops
        returning "L" for a winning Arizona, the guard above is no longer
        load-bearing and the reasoning in `_wlt` should be revisited -- but
        while it does, deleting the guard silently reinstates the bug.
        """
        lean, away, home, ra, rh = "AZ", "PHI", "ARI", 1, 7
        naive = "W" if lean == (home if rh > ra else away) else "L"
        self.assertEqual(naive, "L")          # Arizona won 7-1 and graded L

    def test_a_valid_selection_still_grades_normally(self):
        self.assertEqual(G._wlt("ARI", "PHI", "ARI", 1, 7, False), "W")
        self.assertEqual(G._wlt("ARI", "PHI", "ARI", 7, 1, False), "L")
        self.assertEqual(G._wlt("PHI", "PHI", "ARI", 7, 1, False), "W")

    def test_an_absent_selection_is_still_none(self):
        self.assertIsNone(G._wlt(None, "PHI", "ARI", 1, 7, False))
        self.assertIsNone(G._wlt(float("nan"), "PHI", "ARI", 1, 7, False))

    def test_a_tie_is_unchanged_for_a_valid_selection(self):
        self.assertEqual(G._wlt("ARI", "PHI", "ARI", 3, 3, True), "T")
        self.assertIsNone(G._wlt("ARI", "PHI", "ARI", 3, 3, False))

    def test_the_guard_precedes_the_tie_branch(self):
        """An unrecognised selection is never a tie either -- there is no bet."""
        self.assertIsNone(G._wlt("AZ", "PHI", "ARI", 3, 3, True))


def _committed_row(selection, **over):
    """One graded, tagged, in-window forward row."""
    row = {
        "status": "graded", "game_date": "2026-09-05", "home": "ARI",
        "away": "PHI", "close_p_home": 0.6, "xw_lean": "ARI",
        "full_home": 7, "full_away": 1,
        "close_home_ml": -150, "close_away_ml": 130,
        "selection_rule_tag": ht.RULE_TAG, "pregame_p_home": 0.6,
        "pregame_home_ml": -150, "pregame_away_ml": 130,
        "hybrid_action": "FOLLOW", "hybrid_selection": selection,
        "hybrid_p": 0.6, "hybrid_ml": -150, "hybrid_full": "W",
    }
    row.update(over)
    return row


class UnscorableIsReportedTests(unittest.TestCase):
    def test_a_selection_naming_neither_club_is_counted_not_swallowed(self):
        led = pd.DataFrame([_committed_row("AZ"), _committed_row("ARI")])
        self.assertEqual(len(ht.scored_rows(led)), 1)
        self.assertEqual(ht.unscorable(led), 1)

    def test_a_clean_ledger_reports_nothing_unscorable(self):
        led = pd.DataFrame([_committed_row("ARI"), _committed_row("PHI")])
        self.assertEqual(len(ht.scored_rows(led)), 2)
        self.assertEqual(ht.unscorable(led), 0)

    def test_the_report_names_the_dropped_rows(self):
        led = pd.DataFrame([_committed_row("AZ"), _committed_row("ARI")])
        body = "\n".join(ht.report_lines(led))
        self.assertIn("WARNING", body)
        self.assertIn("eligible rows since registration: 1", body)

    def test_a_clean_report_carries_no_warning(self):
        led = pd.DataFrame([_committed_row("ARI"), _committed_row("PHI")])
        self.assertNotIn("WARNING", "\n".join(ht.report_lines(led)))

    def test_a_missing_price_is_counted_too(self):
        """The counter is about scorability, not only about namespaces."""
        led = pd.DataFrame([_committed_row("ARI", hybrid_ml=np.nan),
                            _committed_row("ARI")])
        self.assertEqual(ht.unscorable(led), 1)

    def test_an_abstention_is_not_counted_as_unscorable(self):
        """No lean is the rule's own abstain branch, not a dropped commitment."""
        led = pd.DataFrame([
            _committed_row("ARI"),
            _committed_row(None, xw_lean=None, hybrid_action=None,
                           hybrid_full=None),
        ])
        self.assertEqual(ht.unscorable(led), 0)

    def test_a_pre_registration_row_is_out_of_the_window_not_unscorable(self):
        led = pd.DataFrame([_committed_row("ARI"),
                            _committed_row("AZ", game_date="2026-08-20")])
        self.assertEqual(ht.unscorable(led), 0)

    def test_a_ledger_with_no_locked_columns_reports_zero(self):
        led = pd.DataFrame([{
            "status": "graded", "game_date": "2026-09-05", "xw_lean": "ARI",
            "home": "ARI", "full_home": 7, "full_away": 1,
        }])
        self.assertEqual(ht.unscorable(led), 0)
        self.assertEqual(len(ht.scored_rows(led)), 0)


class CommittedRowsAreASupersetTests(unittest.TestCase):
    def test_the_committed_set_contains_every_scored_row(self):
        """`unscorable` is a difference of two counts; pin that it cannot go negative."""
        led = pd.read_csv(ht.LEDGER, low_memory=False)
        scored = ht.scored_rows(led)
        self.assertIsNotNone(scored)
        committed = ht._committed(led)
        self.assertTrue(set(scored.index).issubset(set(committed.index)))
        self.assertGreaterEqual(ht.unscorable(led), 0)


class CommittedLedgerIsCleanTests(unittest.TestCase):
    def test_no_ledger_row_carries_a_foreign_club_abbreviation(self):
        """The committed artifact itself, not a constructed frame.

        Rows written before the fix may still hold one; this asserts the count
        does not GROW, and names the offenders when it fails.
        """
        led = pd.read_csv(ht.LEDGER, low_memory=False)
        if "hybrid_selection" not in led.columns:
            self.skipTest("ledger predates decision-time locking")
        sel = led["hybrid_selection"]
        bad = led[sel.notna() & ~(sel.eq(led["home"]) | sel.eq(led["away"]))]
        graded = bad[bad["status"] == "graded"]
        self.assertEqual(
            len(graded), 0,
            "graded rows whose hybrid_selection names neither club: "
            + repr(graded[["game_date", "away", "home",
                           "hybrid_selection"]].to_dict("records")))
