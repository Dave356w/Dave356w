"""v13's starter blend, end to end from the leaderboard to the card.

The defect these are written against: `blend_starter_rate` was correct, its
constants were pinned, its fetch was pinned -- and the blend never fired once
in production, because `build_tables` copied STAT_COLS into the frame and the
blend rate is not in STAT_COLS. Every v13 row carried v12 math under a v13
tag, and `blend_starter_rate`'s own degrade-to-primary rule made it silent.

So every test here drives the REAL path -- load_stat_lookups' output shape ->
build_tables -> build_xwoba_matchup -> the card -- rather than handing a
constructed frame to the blend site. A fixture that starts downstream of
`build_tables` cannot represent this failure, which is why the three tests
that already covered v13 all passed while it was live.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import build_site

PRIMARY = build_site.MODEL_RATE_INTERNAL_COL
BLEND = build_site.BLEND_RATE_INTERNAL_COL
K = build_site.XWOBA_SHRINK_K
LG = {PRIMARY: 0.31500, BLEND: 0.31650}

# Far apart on purpose: a blend that silently returns the primary, or one that
# reads the wrong centre, cannot pass on rounding.
SP = {10: {PRIMARY: 0.300, BLEND: 0.340, "PA": 500.0, "BBE": 100.0},
      20: {PRIMARY: 0.320, BLEND: 0.280, "PA": 500.0, "BBE": 100.0}}
BATS = {p: {PRIMARY: 0.315, BLEND: 0.316, "PA": 400.0, "BBE": 90.0}
        for p in (101, 102, 201, 202)}
PEOPLE = {i: {"name": f"p{i}", "bats": "R", "throws": "R",
              "pos": "P" if i in SP else "OF"}
          for i in (*SP, *BATS)}
SLATE = pd.DataFrame([{
    "game_pk": 1, "game_date": "2026-09-18",
    "game_datetime_utc": "2026-09-18T23:05:00Z",
    "matchup": "AAA @ BBB", "away_team": "AAA", "home_team": "BBB",
    "away_probable_pitcher": "p10", "home_probable_pitcher": "p20",
    "away_probable_pitcher_id": 10, "home_probable_pitcher_id": 20,
    "savant_preview_url": "",
}])
LINEUPS = {1: ([101, 102], [201, 202])}


def _expected(pid):
    """The published rate, derived here rather than read off the module.

    Shrink each metric toward ITS OWN centre at the same K and the same BF,
    then average the two deviations and re-express on the primary centre.
    """
    raw = SP[pid]
    n = raw["PA"]
    p = (n * raw[PRIMARY] + K * LG[PRIMARY]) / (n + K)
    b = (n * raw[BLEND] + K * LG[BLEND]) / (n + K)
    w = build_site.STARTER_BLEND_WEIGHT
    return LG[PRIMARY] + (1 - w) * (p - LG[PRIMARY]) + w * (b - LG[BLEND])


def _frames(stat_pitchers, stat_batters):
    pdf, _ = build_site.build_tables(SLATE, LINEUPS, stat_batters,
                                     stat_pitchers, {}, {}, PEOPLE)
    out = build_site.build_xwoba_matchup(pdf, dict(LG))
    return pdf, (out[0] if isinstance(out, tuple) else out)


class StarterBlendReachesTheStarterTests(unittest.TestCase):
    def test_the_blend_rate_survives_the_frame_builder(self):
        """The gap itself. `segment_pitcher_blocks` passes a starter's whole
        row through to the blend site, so a rate that is not a frame column is
        a rate the blend never sees -- and `blend_starter_rate` answers a
        missing input by returning the primary, so nothing raises and nothing
        logs. Asserted on the starter's ROW, not on any column list, so a
        later refactor that renames the list still has to keep the value."""
        pdf, _ = _frames(SP, BATS)
        self.assertIn(BLEND, pdf.columns)
        sp = pdf[pdf["is_sp"].astype(bool)]
        self.assertEqual(len(sp), 2)
        for _, r in sp.iterrows():
            self.assertAlmostEqual(float(r[BLEND]),
                                   SP[int(r["player_id"])][BLEND], places=12)

    def test_a_v13_build_publishes_the_blend_and_not_the_primary(self):
        """What the tag claims. Both halves are asserted: the published rate
        equals the centred blend to full precision, and it is NOT the primary
        -- the second half is the one that was false in production."""
        _, mx = _frames(SP, BATS)
        self.assertEqual(len(mx), 2)
        self.assertTrue(mx["starter_rate_blended"].all())
        for _, r in mx.iterrows():
            pid = 10 if r["side"] == "away" else 20
            with self.subTest(side=r["side"]):
                self.assertAlmostEqual(float(r["starter_xwOBA"]),
                                       _expected(pid), places=12)
                self.assertNotAlmostEqual(float(r["starter_xwOBA"]),
                                          float(r["starter_rate_primary"]),
                                          places=6)

    def test_the_audit_columns_describe_the_blend_that_happened(self):
        """`starter_rate_primary` and `starter_rate_blend_in` are the dump's
        only record of the two inputs -- the ledger carries neither -- so a
        blend is auditable from the dump alone or not at all."""
        _, mx = _frames(SP, BATS)
        for _, r in mx.iterrows():
            pid = 10 if r["side"] == "away" else 20
            n, w = SP[pid]["PA"], build_site.STARTER_BLEND_WEIGHT
            with self.subTest(side=r["side"]):
                self.assertAlmostEqual(
                    float(r["starter_rate_primary"]),
                    (n * SP[pid][PRIMARY] + K * LG[PRIMARY]) / (n + K),
                    places=12)
                self.assertAlmostEqual(
                    float(r["starter_rate_blend_in"]),
                    (n * SP[pid][BLEND] + K * LG[BLEND]) / (n + K), places=12)
                # The published rate is exactly what the two stored inputs say.
                self.assertAlmostEqual(
                    float(r["starter_xwOBA"]),
                    LG[PRIMARY]
                    + (1 - w) * (float(r["starter_rate_primary"]) - LG[PRIMARY])
                    + w * (float(r["starter_rate_blend_in"]) - LG[BLEND]),
                    places=12)

    def test_the_blend_is_centred_on_each_metrics_own_league_rate(self):
        """A starter who is exactly league-average on both metrics comes back
        at exactly the primary centre. A raw average would return the midpoint
        of the two centres and carry that level shift into the starter phase
        only -- which is the bullpen comparison silently biased."""
        flat = {10: {PRIMARY: LG[PRIMARY], BLEND: LG[BLEND],
                     "PA": 500.0, "BBE": 100.0},
                20: {PRIMARY: LG[PRIMARY], BLEND: LG[BLEND],
                     "PA": 500.0, "BBE": 100.0}}
        _, mx = _frames(flat, BATS)
        # Without this the case is vacuous: a build that never blends returns
        # the primary centre here too, and passes for the wrong reason.
        self.assertTrue(mx["starter_rate_blended"].all())
        for _, r in mx.iterrows():
            self.assertAlmostEqual(float(r["starter_xwOBA"]), LG[PRIMARY],
                                   places=12)
            # A raw average would land on the midpoint of the two centres.
            self.assertNotAlmostEqual(float(r["starter_xwOBA"]),
                                      (LG[PRIMARY] + LG[BLEND]) / 2, places=6)


class MissingBlendRateCostsTheRefinementTests(unittest.TestCase):
    """A slate's pregame rows cannot be re-derived afterwards without
    lookahead, so an absent optional column must cost the blend and never the
    build. Both ways it can go wrong are pinned: raising, and pretending."""

    NO_BLEND = {pid: {k: v for k, v in s.items() if k != BLEND}
                for pid, s in SP.items()}

    def test_a_slate_still_builds_when_savant_serves_no_blend_rate(self):
        pdf, mx = _frames(self.NO_BLEND,
                          {p: {k: v for k, v in s.items() if k != BLEND}
                           for p, s in BATS.items()})
        # The frame's schema must not depend on the optional column: an absent
        # key raises in build_tables' projection, which costs the slate.
        self.assertIn(BLEND, pdf.columns)
        self.assertEqual(len(mx), 2)

    def test_an_unblended_rate_is_the_pure_primary_and_says_so(self):
        _, mx = _frames(self.NO_BLEND, BATS)
        self.assertFalse(mx["starter_rate_blended"].any())
        self.assertTrue(mx["starter_rate_blend_in"].isna().all())
        for _, r in mx.iterrows():
            pid = 10 if r["side"] == "away" else 20
            n = SP[pid]["PA"]
            self.assertAlmostEqual(
                float(r["starter_xwOBA"]),
                (n * SP[pid][PRIMARY] + K * LG[PRIMARY]) / (n + K), places=12)

    def test_a_missing_league_centre_for_the_blend_falls_back_too(self):
        """The centre comes from the batter leaderboard. If that board serves
        the primary and not the blend, the deviation has no centre to be
        measured against and the refinement is not available."""
        pdf, _ = build_site.build_tables(SLATE, LINEUPS, BATS, SP, {}, {},
                                         PEOPLE)
        out = build_site.build_xwoba_matchup(pdf, {PRIMARY: LG[PRIMARY]})
        mx = out[0] if isinstance(out, tuple) else out
        self.assertFalse(mx["starter_rate_blended"].any())
        for _, r in mx.iterrows():
            pid = 10 if r["side"] == "away" else 20
            n = SP[pid]["PA"]
            self.assertAlmostEqual(
                float(r["starter_xwOBA"]),
                (n * SP[pid][PRIMARY] + K * LG[PRIMARY]) / (n + K), places=12)


class OnlyTheStarterBlendsTests(unittest.TestCase):
    def test_the_blend_rate_builds_no_matchup_value_edge_or_bar(self):
        """v13 changes ONE input. A rate in STATCAST_RATE_COLS acquires a
        matchup value, an edge and a percentile bar that no surface publishes,
        and a second metric running through the whole construction is a
        different model from the one this tag names."""
        self.assertNotIn(BLEND, build_site.STATCAST_RATE_COLS)
        pdf, mx = _frames(SP, BATS)
        for stem in ("", "pit_", "opp_", "lg_", "mx_", "edge_"):
            self.assertNotIn(f"{stem}{BLEND}", mx.columns)
        # The rate rides the FRAME and stops there: the dump is built from an
        # explicit record, so a frame column is not a schema change.
        self.assertIn(BLEND, pdf.columns)
        # Hitters carry the key and no value -- the schema is unconditional,
        # the reader is the starter alone.
        self.assertTrue(pdf.loc[~pdf["is_sp"].astype(bool), BLEND].isna().all())

    def test_the_lineup_composite_is_untouched_by_the_blend_rate(self):
        """The opposing lineup is aggregated on the primary rate alone, so a
        hitter's blend value -- if one ever reaches the frame -- must not move
        the composite the starter is scored against."""
        _, with_blend = _frames(SP, BATS)
        _, without = _frames(SP, {p: {k: v for k, v in s.items() if k != BLEND}
                                  for p, s in BATS.items()})
        pd.testing.assert_series_equal(with_blend["opp_xwOBA_neutral"],
                                       without["opp_xwOBA_neutral"])


class TheCardPublishesTheBlendTests(unittest.TestCase):
    """The operator's call, 2026-09-18: the card shows the blended rate under
    the label it already had. So the claim under test is that the number in
    the `<LABEL> agn` cell is the blend -- the label naming one metric over a
    value that is a mixture of two is recorded in CLAUDE.md, not corrected
    here."""

    def _cards(self):
        _, mx = _frames(SP, BATS)
        slate = SLATE.copy()
        for c in ("away_abbrev", "home_abbrev", "away_team_id", "home_team_id",
                  "venue", "status", "away_score", "home_score"):
            slate[c] = None
        slate["abstract_state"] = "Preview"
        slate["double_header"], slate["game_number"] = "N", 1
        return mx, build_site._df_to_combined_games(
            mx, pd.DataFrame(), pd.DataFrame(), slate_df=slate,
            league_baseline=dict(LG))

    def test_the_card_reads_the_starters_own_rate_not_the_workload_blend(self):
        """`pit_xwOBA` is overwritten downstream with the workload-weighted
        SP+BP diagnostic, so a card sourcing the cell from it would publish a
        pitching-staff composite under a starter's name."""
        mx, games = self._cards()
        self.assertEqual(len(games), 1)
        for side in ("away", "home"):
            row = mx[mx["side"] == side].iloc[0]
            self.assertAlmostEqual(games[0][side]["pit_xw"],
                                   float(row["starter_xwOBA"]), places=12)

    def test_the_blended_rate_renders_in_the_cell_labelled_for_the_primary(self):
        _, games = self._cards()
        html = build_site._side_html("AAA", games[0]["away"], dict(LG))
        label = f"{build_site.MODEL_RATE_LABEL} agn"
        self.assertIn(label, html)
        shown = build_site.f3(_expected(10))
        primary = build_site.f3((500 * SP[10][PRIMARY] + K * LG[PRIMARY]) / 600)
        self.assertNotEqual(shown, primary)   # or the test proves nothing
        self.assertIn(f"<div class='l'>{label}</div><div class='v'>{shown}</div>",
                      html)


if __name__ == "__main__":
    unittest.main()
