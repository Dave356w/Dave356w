"""Card redesign: percentile bars, ledger-ranked lean strength, the
plain-language read, the model-vs-market verdict, and the always-shown model
machinery.

All display-only -- these guard the render layer, not the lean math."""
import re
import unittest
import unittest.mock

import numpy as np
import pandas as pd

import build_site as b


def _hitter(name, pos, bats, xw, pct, adv=False, player_id=None):
    return dict(name=name, player_id=player_id, pos=pos, bats=bats, xw=xw,
                xw_pctile=pct, adv=adv,
                ops=None, pa=0, low=False, mx=None, edge=None)


def _side(p, opp_abbr, pit_xw_pct, kbb_pct, xw_edge, hitters, player_id=None):
    return dict(p=p, player_id=player_id, t="R", opp_abbr=opp_abbr,
                pit_xw=.300, pit_k=.24, pit_bb=.07,
                era_season=3.6, xera=3.4, opp_xw=.330, xw_edge=xw_edge,
                pit_xw_pctile=pit_xw_pct, kbb_pctile=kbb_pct, lu_status="posted",
                is_opener=False, has_pl=True, R=5, L=4, S=0, padv=3, pl_fl={},
                hitters=hitters)


def _game(away, home, a, h, odds):
    return dict(away=a, home=h, away_abbr=away, home_abbr=home, away_streak="W3",
                home_streak="L1", game_pk=1, game_number=1, game_label="",
                abstract_state="Preview", status="Scheduled",
                away_score=None, home_score=None,
                current_inning=None, current_inning_ordinal=None,
                inning_half=None, is_top_inning=None,
                game_datetime_utc=None, time_pt="7:05 PM ET", venue="Park", odds=odds,
                league_baseline={"xwOBA": .312, "K%": .22, "BB%": .08, "OPS": .715, "ERA": 4.1})


class LeanStrengthTests(unittest.TestCase):
    def test_fixed_fallback_buckets(self):
        # No scale at all -> fixed cutoffs (0.015 / 0.032), no percentile.
        c1, c2 = b.LEAN_STRENGTH_FALLBACK
        self.assertEqual(b.lean_strength(c1 / 2, None)[0], "slight")
        self.assertEqual(b.lean_strength((c1 + c2) / 2, None)[0], "clear")
        self.assertEqual(b.lean_strength(c2 * 2, None)[0], "strong")
        self.assertIsNone(b.lean_strength((c1 + c2) / 2, None)[1])

    def test_fallback_is_on_the_current_shrunk_scale(self):
        # Regression: the fallback was the pre-v5 unshrunk p33/p80 (.021/.060),
        # which no v8/v9 delta can reach -- v9's observed max |xw_net| is
        # .0462, so "strong" was unreachable for the entire family.
        self.assertLess(b.LEAN_STRENGTH_FALLBACK[1], 0.0462)

    # --- cutoff continuity (the p33/p80 gate that used to sit at pool >= 30) --

    def _c2(self, scale):
        """The clear/strong boundary lean_strength is *actually* using, found by
        bisection on its own output. Deliberately does not recompute the
        formula, so the test fails if the implementation stops matching it."""
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if b.lean_strength(mid, scale)[0] == "strong":
                hi = mid
            else:
                lo = mid
        return hi

    # Pool geometry matters for these two: the shipped cliff only bites when the
    # observed p80 sits ABOVE the frozen 0.032 prior, which is the real
    # v8/v9/v10 situation (observed 0.0357). linspace(0.002, 0.06, 40) has p80
    # 0.0353 at n=29, reproducing that within 0.0004. A pool whose p80 falls
    # below 0.032 hides the bug -- an earlier draft of these tests used one and
    # passed against the unfixed code.
    CLIFF_POOL = (0.002, 0.06, 40)

    def _pool(self):
        import numpy as _np
        lo, hi, n = self.CLIFF_POOL
        return _np.sort(_np.linspace(lo, hi, n))

    def test_no_cutoff_jump_at_the_old_gate(self):
        # The old `pool >= 30` switch moved p80 in one step on the row that
        # tripped it. Growing the pool across that point must now be smooth.
        pool = self._pool()
        cuts = [self._c2(pool[:n]) for n in range(25, 36)]
        steps = [abs(y - x) for x, y in zip(cuts, cuts[1:])]
        self.assertLess(max(steps), 0.002, f"cutoff jumped: {steps}")
        # and the n=0 -> n=1 transition is continuous too (no branch there)
        self.assertLess(abs(self._c2(pool[:1]) - self._c2(None)), 0.002)

    def test_one_row_relabels_only_a_narrow_band(self):
        # A moving cutoff must eventually cross any fixed delta -- that is not
        # the bug. The bug was how *wide* the band relabelled by a single row
        # was: the gate swapped the 0.032 prior for a ~0.0365 observed p80 in
        # one step, relabelling every game in between at once. Measure the band
        # instead of demanding no game ever changes.
        import numpy as _np
        pool = self._pool()
        grid = _np.linspace(0.001, 0.05, 200)
        labs = lambda s: [b.lean_strength(d, s)[0] for d in grid]
        worst = max(sum(x != y for x, y in zip(labs(pool[:n]), labs(pool[:n + 1])))
                    for n in range(5, 39))
        # shipped cliff relabels ~18/200 on the row that trips the gate
        self.assertLessEqual(worst, 5, f"a single row relabelled {worst}/200 probes")

    def test_empty_pool_equals_the_prior_exactly(self):
        # n=0 is the limit of the shrinkage expression, not a separate branch.
        c1, c2 = b.LEAN_STRENGTH_FALLBACK
        self.assertEqual(b.lean_strength(c1 - 1e-9, None)[0], "slight")
        self.assertEqual(b.lean_strength(c2 - 1e-9, None)[0], "clear")
        self.assertEqual(b.lean_strength(c2 + 1e-9, None)[0], "strong")

    def test_large_pool_converges_toward_observed_quantiles(self):
        # Shrinkage must decay: at a big pool the cutoffs track the data, not
        # the prior. Guards against a K so large the ledger never takes over.
        import numpy as _np
        pool = _np.sort(_np.linspace(0.10, 0.30, 4000))   # far from the prior
        w = 4000 / (4000 + b.LEAN_STRENGTH_PRIOR_N)
        self.assertGreater(w, 0.97)
        # a delta below the observed p33 must read "slight" despite the prior
        # cutoffs (0.015/0.032) sitting far below this pool entirely
        self.assertEqual(b.lean_strength(0.11, pool)[0], "slight")

    def test_ranked_against_scale(self):
        scale = np.sort(np.array([round(0.01 * i, 2) for i in range(1, 11)] * 4, float))
        lab, pct = b.lean_strength(0.10, scale)
        self.assertEqual(lab, "strong")
        self.assertGreater(pct, 90)

    def test_none_delta(self):
        self.assertEqual(b.lean_strength(None, None), (None, None))

    def _ledger(self, rows):
        # rows: list of (model_tag, xw_net); all graded.
        return pd.DataFrame([{"model_tag": t, "xw_net": v, "status": "graded"}
                             for t, v in rows])

    def test_scale_excludes_incompatible_units(self):
        # 40 current-scale rows (small, shrunk deltas) + 200 pre-v5 rows (large
        # deltas, 2x scale). The scale must rank against the current units only,
        # never the mixed pool -- so its max stays on the shrunk scale.
        rows = ([("xw+plat_consol_v6", 0.015)] * 40
                + [("xw+plat_consol_v2", 0.040)] * 200)
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v5", "xw+plat_consol_v6")):
            scale = b.lean_strength_scale()
        self.assertEqual(scale.size, 40)
        self.assertLess(scale.max(), 0.02)

    def test_scale_thin_pool_no_allgraded_fallback(self):
        # A thin pool must still never reach for the 200 incompatible pre-v5
        # rows. It is kept (and shrunk toward the prior downstream) rather than
        # discarded, so assert its *contents*, not its size against a gate.
        rows = ([("xw+plat_consol_v6", 0.015)] * 10
                + [("xw+plat_consol_v2", 0.040)] * 200)
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v5", "xw+plat_consol_v6")):
            scale = b.lean_strength_scale()
        self.assertEqual(scale.size, 10)
        self.assertLess(scale.max(), 0.02)   # never the 0.040 pre-v5 units

    def test_scale_counts_pending_rows(self):
        # |xw_net| is a pregame quantity: a magnitude scale needs no outcome.
        # 20 graded + 20 pending must reach the pool, not just the 20 graded.
        led = self._ledger([("xw+plat_consol_v9", 0.02)] * 40)
        led.loc[20:, "status"] = "pending"
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=led), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v9",)):
            scale = b.lean_strength_scale()
        self.assertIsNotNone(scale)
        self.assertEqual(scale.size, 40)

    def test_slate_deltas_top_up_thin_ledger(self):
        # A fresh scale family: tonight's own deltas are the same units and
        # join the ledger rows in one pool.
        rows = [("xw+plat_consol_v9", 0.02)] * 24
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v9",)):
            bare = b.lean_strength_scale()
            topped = b.lean_strength_scale([0.03] * 12)
        self.assertEqual(bare.size, 24)
        self.assertEqual(topped.size, 36)

    def test_empty_pool_is_none(self):
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=None):
            self.assertIsNone(b.lean_strength_scale())
            self.assertIsNone(b.lean_strength_scale([]))

    def test_v8_v9_v10_share_a_scale_family(self):
        # v9 - v8 is one term worth ~6% of matchup dispersion and flips no
        # leans; v10 only re-weights a convex combination of the same two
        # phases. All three measure |xw_net| on the same units.
        fam = ("xw+plat_consol_v8", "xw+plat_consol_v9", "xw+plat_consol_v10")
        for tag in fam:
            self.assertEqual(b._SCALE_FAMILIES[tag], fam)

    def test_slate_deltas_helper(self):
        def game(ae, he, **kw):
            return {"away": {"xw_edge": ae}, "home": {"xw_edge": he}, **kw}
        games = [game(0.02, 0.01),            # |net| = 0.01
                 game(0.01, 0.04),            # |net| = 0.03
                 game(0.02, 0.02),            # exact zero -> abstention
                 game(None, 0.01),            # missing edge -> no lean
                 game(0.05, 0.01, unavailable=True)]
        out = b._slate_deltas(games)
        self.assertEqual([round(x, 4) for x in out], [0.01, 0.03])
        self.assertEqual(b._slate_deltas([]), [])
        self.assertEqual(b._slate_deltas(None), [])


class PercentileTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 400
        pa = rng.integers(5, 650, n)
        true = rng.normal(.315, .030, n)
        obs = true + rng.normal(0, .35 / np.sqrt(np.maximum(pa, 1)), n)
        self.cust = pd.DataFrame({"woba": obs, "pa": pa})

    def test_shrinkage_pulls_small_sample_toward_middle(self):
        ref, _ = b.build_pctile_ref(self.cust, .312, 130.0, 300.0)
        hot_big = b.pctile_rank(.430, 600, ref, .312, 130.0)   # real elite
        hot_tiny = b.pctile_rank(.430, 20, ref, .312, 130.0)   # 20-PA fluke
        self.assertGreater(hot_big, 90)
        self.assertLess(hot_tiny, hot_big - 15)

    def test_pitcher_inversion(self):
        ref, _ = b.build_pctile_ref(self.cust, .312, 130.0, 300.0)
        good = b.pctile_rank(.270, 700, ref, .312, 130.0, invert=True)
        bad = b.pctile_rank(.360, 700, ref, .312, 130.0, invert=True)
        self.assertGreater(good, bad)

    def test_none_inputs_degrade(self):
        self.assertIsNone(b.pctile_rank(None, 100, np.array([.3, .31, .32]), .312, 130.0))
        self.assertIsNone(b.pctile_rank(.3, 100, None, .312, 130.0))


class CurrentStreakTests(unittest.TestCase):
    def test_reads_official_streak_code_and_ignores_invalid_values(self):
        standings = {"records": [{"teamRecords": [
            {"team": {"id": 119}, "streak": {"streakCode": "W3"}},
            {"team": {"id": 147}, "streak": {"streakCode": "L1"}},
            {"team": {"id": 999}, "streak": {"streakCode": "—"}},
        ]}]}
        from unittest import mock
        with mock.patch.object(b, "_get_json", return_value=standings):
            self.assertEqual(b.fetch_current_streaks(), {119: "W3", 147: "L1"})


class CurrentScoreTests(unittest.TestCase):
    def test_slate_carries_official_state_and_scores(self):
        schedule = {"dates": [{"date": "2026-07-27", "games": [{
            "gamePk": 123,
            "gameDate": "2026-07-27T23:10:00Z",
            "gameNumber": 1,
            "doubleHeader": "N",
            "status": {
                "abstractGameState": "Live",
                "detailedState": "In Progress",
            },
            "linescore": {
                "currentInning": 2,
                "currentInningOrdinal": "2nd",
                "inningHalf": "Top",
                "isTopInning": True,
            },
            "teams": {
                "away": {"team": {"id": 119, "name": "Dodgers"}, "score": 3},
                "home": {"team": {"id": 147, "name": "Yankees"}, "score": 2},
            },
            "venue": {"name": "Test Park"},
        }]}]}
        from unittest import mock
        with mock.patch.object(b, "_get_json", return_value=schedule):
            row = b.get_slate("2026-07-27").iloc[0]
        self.assertEqual(row["abstract_state"], "Live")
        self.assertEqual(row["status"], "In Progress")
        self.assertEqual(row["away_score"], 3)
        self.assertEqual(row["home_score"], 2)
        self.assertEqual(row["current_inning_ordinal"], "2nd")
        self.assertEqual(row["inning_half"], "Top")

    def test_browser_refresh_uses_one_slate_request_per_minute(self):
        js = b.score_refresh_js()
        self.assertIn("statsapi.mlb.com/api/v1/schedule", js)
        self.assertIn("setInterval(refresh,60000)", js)
        self.assertIn("state==='Live'||state==='Final'", js)
        self.assertIn("linescore.currentInningOrdinal", js)
        self.assertIn("summaryTime.hidden=inProgress", js)
        self.assertIn("summaryMarket.hidden=inProgress", js)


class RenderTests(unittest.TestCase):
    def _cards(self):
        ari = [_hitter(f"A{i}", "LF", "R", .33, 60) for i in range(9)]
        lad = ([_hitter("Betts", "RF", "R", .372, 95),
                _hitter("Ohtani", "DH", "L", .401, 99, adv=True)]
               + [_hitter(f"L{i}", "LF", "R", .33, 60) for i in range(7)])
        a = _side("Glasnow", "ARI", 91, 93, xw_edge=-0.050, hitters=ari)
        h = _side("Gallen", "LAD", 42, 55, xw_edge=+0.042, hitters=lad)
        g_agree = _game("LAD", "ARI", a, h, dict(away_ml=-160, home_ml=135, p_home=.395))
        a2 = _side("Bibee", "NYY", 66, 71, xw_edge=+0.019, hitters=ari)
        h2 = _side("Rodon", "CLE", 35, 62, xw_edge=+0.030, hitters=lad)
        g_dis = _game("CLE", "NYY", a2, h2, dict(away_ml=148, home_ml=-175, p_home=.62))
        return g_agree, g_dis

    def test_casual_card_structure(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        text = re.sub(r"<[^>]+>", "", html)
        self.assertIn("class='sl h'", html)          # hitter percentile bar
        self.assertIn("class='sl p'", html)          # starter percentile bar
        self.assertIn("Standouts", html)
        self.assertIn("class='read'", html)
        self.assertIn("class='matchlab'", html)      # matchup-labeled columns
        self.assertIn("LAD bats", text)
        self.assertIn("class='tier", html)

    def test_game_is_collapsed_scoreboard_with_existing_breakdown_inside(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        self.assertIn("<details class='game-card'>", html)
        self.assertNotIn("<details class='game-card' open>", html)
        self.assertIn("<summary class='game-summary'", html)
        self.assertIn("class='summary-team away'", html)
        self.assertIn("class='summary-center'", html)
        self.assertIn("class='summary-team home'", html)
        self.assertIn(">7:05 PM ET</span>", html)
        self.assertIn(">LAD -160</span>", html)
        detail = html.split("<div class='game-detail'>", 1)[1]
        self.assertIn("class='read'", detail)
        self.assertIn("class='market'", detail)
        self.assertIn("class='sides'", detail)

    def test_scoreboard_uses_familiar_team_labels_when_schedule_names_exist(self):
        g, _ = self._cards()
        g.update(away_team_name="Los Angeles Dodgers",
                 home_team_name="Arizona Diamondbacks")
        html = b.cmb_card(g, None)
        self.assertIn("<span class='summary-club'>Dodgers</span>", html)
        self.assertIn("<span class='summary-club'>D-backs</span>", html)

    def test_team_headers_show_current_streaks(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        self.assertIn("class='team-meta streak away'", html)
        self.assertIn("class='team-meta streak home'", html)
        self.assertIn(">W3</span>", html)
        self.assertIn(">L1</span>", html)
        self.assertIn("class='game-state' hidden", html)
        self.assertNotIn("last 10 games", html)

    def test_live_game_replaces_streaks_with_scores(self):
        g, _ = self._cards()
        g.update(abstract_state="Live", status="In Progress",
                 away_score=3, home_score=2,
                 current_inning=2, current_inning_ordinal="2nd",
                 inning_half="Top", is_top_inning=True)
        html = b.cmb_card(g, None)
        self.assertIn("class='team-meta score away'", html)
        self.assertIn("class='team-meta score home'", html)
        self.assertIn(">3</span>", html)
        self.assertIn(">2</span>", html)
        self.assertIn("class='game-state live'", html)
        self.assertIn(">LIVE · ▲2nd</span>", html)
        self.assertNotIn(">W3</span>", html)
        self.assertNotIn(">L1</span>", html)
        self.assertIn("class='summary-time' hidden", html)
        self.assertIn("class='summary-market' hidden", html)

    def test_live_game_uses_down_marker_for_bottom_half(self):
        g, _ = self._cards()
        g.update(abstract_state="Live", status="In Progress",
                 away_score=1, home_score=4,
                 current_inning=1, current_inning_ordinal="1st",
                 inning_half="Bottom", is_top_inning=False)
        html = b.cmb_card(g, None)
        self.assertIn(">LIVE · ▼1st</span>", html)

    def test_final_game_replaces_streaks_with_final_score(self):
        g, _ = self._cards()
        g.update(abstract_state="Final", status="Final",
                 away_score=6, home_score=4)
        html = b.cmb_card(g, None)
        self.assertIn(">6</span>", html)
        self.assertIn(">4</span>", html)
        self.assertIn("class='game-state final'", html)
        self.assertIn(">FINAL</span>", html)

    def test_footer_carries_no_how_to_read_guide(self):
        # The guide is gone entirely -- the card is expected to read on its own.
        self.assertFalse(hasattr(b, "_legend_guide"))
        html = b.render_combined_html(
            pd.DataFrame(columns=["game_pk", "side"]), pd.DataFrame(),
            pd.DataFrame(), "9:00 AM")
        for gone in ("How to read a card", "warmer / longer bar",
                     "xwOBA %ile", "highest-ranked bats", "Starter quality",
                     "marks a platoon advantage"):
            self.assertNotIn(gone, html)
        # Its CSS went with it; `.lg-keys` stays for the empty-slate page.
        for dead in (".lg-lead", ".lg-notes", ".sw{"):
            self.assertNotIn(dead, b.CSS)
        self.assertIn(".lg-keys{", b.CSS)

    def test_read_names_the_opposing_starter(self):
        # away offense faces the HOME starter; home offense the AWAY starter.
        g, _ = self._cards()
        read = b.cmb_card(g, None).split("class='read'")[1].split("</p>")[0]
        text = re.sub(r"<[^>]+>", "", read)
        self.assertIn("LAD's bats grade", text)
        self.assertIn("against Gallen", text)       # LAD faces home SP Gallen
        self.assertIn("against Glasnow", text)      # ARI faces away SP Glasnow

    def test_player_names_link_to_official_mlb_profiles(self):
        g, _ = self._cards()
        g["away"]["player_id"] = 675512
        g["away"]["hitters"][0]["name"] = "Curtis Mead"
        g["away"]["hitters"][0]["player_id"] = 678554
        html = b.cmb_card(g, None)
        self.assertIn("href='https://www.mlb.com/player/675512'", html)
        self.assertIn("href='https://www.mlb.com/player/678554'", html)
        self.assertIn("target='_blank' rel='noopener noreferrer'", html)
        self.assertIn(".player-link:focus-visible", b.CSS)

    def test_player_link_falls_back_to_escaped_text_without_id(self):
        self.assertEqual(b._mlb_player_link("A & B", None), "A &amp; B")

    def test_collapsed_team_labels_do_not_link_to_mlb_standings(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        self.assertNotIn("https://www.mlb.com/standings/", html)
        self.assertNotIn("team-link", html)
        self.assertNotIn(".team-link", b.CSS)

    def test_model_machinery_renders(self):
        # The model machinery (formerly gated behind an Analyst toggle) is now
        # always part of the card; the .mach classes remain as layout hooks.
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        self.assertIn("<span class='mach'>", html)
        self.assertIn("spstats mach", html)
        # The redundant pct-lean suffix and the secondary xwOBA consensus line
        # were removed; the strength word is the card's single lean readout.
        self.assertNotIn("pct lean", html)
        self.assertNotIn("xwOBA →", html)

    def test_pitcher_card_shows_xera_not_removed_lenses(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        self.assertIn("xERA", html)                 # xERA cell present
        self.assertIn("season 3.6", html)           # ...vs season ERA
        # The generic .mach display hook must not stack the three starter
        # stat cells; its flex override belongs later in the cascade.
        self.assertIn(".spstats.mach{display:flex}", b.CSS)
        self.assertGreater(b.CSS.index(".spstats.mach{display:flex}"),
                           b.CSS.index(".mach{display:block}"))
        self.assertNotIn("xw edge (drives lean)", html.lower())
        self.assertNotIn("class='agg mach'", html)
        self.assertNotIn("OPS alwd", html)          # xOPS-against removed
        self.assertNotIn("xOPS edge", html)         # xOPS edge removed
        self.assertNotIn("pythag", html)            # pythag control removed
        self.assertNotIn("DK F5", html)             # F5 odds removed

    def test_verdict_agree_and_disagree(self):
        g_agree, g_dis = self._cards()
        self.assertIn("agrees with the market", b.cmb_card(g_agree, None))
        dis = b.cmb_card(g_dis, None)
        self.assertIn("verdict edge", dis)
        self.assertIn("underdog", dis)

    def test_verdict_shows_context_record_when_available(self):
        # ctx = market_context_records(): (lean side, agree/disagree) -> 'W-L'.
        ctx = {("home", "agree"): "30-27", ("away", "disagree"): "27-30"}
        # Agree: model favors the home side, which is also the market favorite.
        agree = b._verdict_html("ARI", dict(p_home=.62), "LAD", "ARI", ctx)
        self.assertIn("home favorite", agree)
        self.assertIn("30-27", agree)
        self.assertNotIn("No edge on the line", agree)
        # Disagree: model leans the away underdog against a home market favorite.
        dis = b._verdict_html("LAD", dict(p_home=.62, away_ml=140), "LAD", "ARI", ctx)
        self.assertIn("away underdog", dis)
        self.assertIn("27-30", dis)
        self.assertNotIn("record is built to test", dis)
        # Missing bucket -> prose fallback (no fabricated record).
        fb = b._verdict_html("ARI", dict(p_home=.62), "LAD", "ARI", {})
        self.assertIn("No edge on the line", fb)

    def test_missing_edge_has_no_lean_read_or_header_pill(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = None
        html = b.cmb_card(g, None)
        self.assertNotIn("class='read'", html)
        self.assertNotIn("class='lean", html)

    def test_exact_zero_delta_has_no_lean_read(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = .012345
        g["home"]["xw_edge"] = .012345
        html = b.cmb_card(g, None)
        self.assertNotIn("class='read'", html)
        self.assertNotIn("class='lean", html)

    def test_nonzero_sub_display_delta_still_names_a_favorite(self):
        # A delta too small to have printed at three decimals is still a lean,
        # and the plain-language read must name the side.
        g, _ = self._cards()
        g["away"]["xw_edge"] = .01234567891
        g["home"]["xw_edge"] = .01234567890
        html = b.cmb_card(g, None)
        self.assertIn("class='read'", html)
        self.assertIn("lean to <b>ARI</b>", html)
        self.assertNotIn("class='lean", html)


class UnlabeledPercentileTests(unittest.TestCase):
    """Percentiles are shown as bar length only, and the lineup table runs
    without a header row. Both are display decisions: the ranking that orders
    the Standouts pills and sizes the bars is unchanged, so no lean moves."""

    def _card(self):
        g, _ = RenderTests()._cards()
        return b.cmb_card(g, None)

    def test_bar_carries_no_printed_percentile(self):
        self.assertEqual(b._pct_bar(72, "h"),
                         "<span class='sl h'><i style='width:72%'></i></span>")
        # A missing percentile is an empty track, not a placeholder character.
        self.assertEqual(b._pct_bar(None, "h"), "<span class='sl na'></span>")
        html = self._card()
        self.assertIn("class='sl h'", html)     # bars survive the number's removal
        self.assertIn("class='sl p'", html)
        self.assertNotIn("class='pn'", html)
        self.assertNotIn(".pn{", b.CSS)         # rule deleted with its markup

    def test_standouts_pill_is_name_only(self):
        hitters = [dict(name="Mookie Betts", xw_pctile=95, player_id=605141)]
        pill = b._spotlight_html(hitters, n=3, thresh=70)
        self.assertIn("Standouts", pill)
        self.assertIn("Betts", pill)
        self.assertNotIn("95", re.sub(r"<[^>]+>", "", pill))
        self.assertNotIn("<b>", pill)

    def test_percentile_still_ranks_and_gates_the_pills(self):
        # Removing the printed number must not disturb selection or ordering.
        # Pills print surnames, so the three must not share one.
        hitters = [dict(name="Some Gamma", xw_pctile=40),
                   dict(name="Some Beta", xw_pctile=80),
                   dict(name="Some Alpha", xw_pctile=99)]
        text = re.sub(r"<[^>]+>", " ", b._spotlight_html(hitters, n=3, thresh=70))
        self.assertLess(text.index("Alpha"), text.index("Beta"))
        self.assertNotIn("Gamma", text)          # below the 70 threshold

    def test_lineup_table_has_no_header_row(self):
        html = self._card()
        self.assertIn("<table class='lu'>", html)
        self.assertNotIn("xwOBA %ile</th>", html)
        self.assertNotIn("<th", html.split("<table class='lu'>")[1])
        self.assertNotIn("table.lu th", b.CSS)   # rules deleted with the markup
        # The columns themselves stay: order, name, position, bar, raw xwOBA.
        row = html.split("<table class='lu'>")[1].split("</tr>")[0]
        for cls in ("class='ord'", "class='nm'", "class='pos'", "class='pct r'"):
            self.assertIn(cls, row)


class PlatoonHandMarkerTests(unittest.TestCase):
    """The batting-hand letter doubles as the platoon-advantage marker: warm
    when this batter holds the edge over the starter, muted otherwise. It
    replaced a separate ◆ that only ever rendered next to the same letter."""

    def _card(self):
        g, _ = RenderTests()._cards()
        return b.cmb_card(g, None)

    def _row(self, name):
        html = self._card()
        return next(r for r in html.split("<tr>") if f"title='{name}'" in r)

    def test_diamond_marker_is_gone(self):
        html = self._card()
        self.assertNotIn("◆", html)
        self.assertNotIn("class='adv mach'", html)

    def test_advantaged_hand_is_accented_and_titled(self):
        # Ohtani is the fixture's one platoon-advantaged bat.
        row = self._row("Ohtani")
        self.assertIn("<span class='b adv' title='platoon advantage vs this SP'>L</span>",
                      row)

    def test_unadvantaged_hand_stays_muted_and_untitled(self):
        row = self._row("Betts")
        self.assertIn("<span class='b'>R</span>", row)
        self.assertNotIn("platoon advantage", row)

    def test_hand_omitted_entirely_when_unknown(self):
        hr = dict(name="No Hand", player_id=None, pos="LF", bats="", xw=.300,
                  xw_pctile=50, adv=False, ops=None, pa=0, low=False,
                  mx=None, edge=None)
        self.assertNotIn("<span class='b", b._hitter_row_html(1, hr))
        # ...and an advantage with no recorded hand still has nothing to mark.
        self.assertNotIn("adv", b._hitter_row_html(1, dict(hr, adv=True)))

    def test_accent_rule_beats_the_muted_base(self):
        # `td.nm .b.adv` (0,3,1) must outrank `td.nm .b` (0,2,1), and it must
        # use the contrast-audited text twin, not the bar-fill token.
        self.assertIn("td.nm .b.adv{color:rgba(var(--warm-tx),1)", b.CSS)
        self.assertLess(b.CSS.index("td.nm .b{"), b.CSS.index("td.nm .b.adv{"))


class LeanDescriptionTests(unittest.TestCase):
    """The read sentence is the card's single lean explanation."""

    def _card(self, **over):
        r = RenderTests()
        g, _ = r._cards()
        for k, v in over.items():
            g[k[:4]]["xw_edge"] = v
        return b.cmb_card(g, None)

    def test_read_keeps_strength_word_without_header_pill(self):
        html = self._card()
        self.assertIn("That is a <b>strong</b> lean to", html)
        self.assertNotIn("class='lean", html)
        self.assertNotIn("Δxw", html)
        self.assertNotIn("<span class='mach'> · Δ", html)

    def test_missing_edge_has_neither_read_nor_pill(self):
        html = self._card(away=None)
        self.assertNotIn("class='read'", html)
        self.assertNotIn("class='lean", html)
        self.assertNotIn("Δxw", html)

    def test_header_pill_css_is_removed(self):
        self.assertNotIn(".lean{", b.CSS)
        self.assertNotIn(".lean.", b.CSS)
        self.assertNotIn(".lean ", b.CSS)


def _mobile_rules():
    """{selector: {prop: value}} for the single max-width:540px block.

    There must be exactly one such block and it must be last in the sheet:
    media queries add no specificity, so a phone rule declared above its
    desktop counterpart loses the cascade. `.sl` and `td.pct` were silently
    dead that way before this block was consolidated."""
    blocks = b.CSS.split("@media (max-width:540px){")
    assert len(blocks) == 2, f"expected 1 phone block, found {len(blocks) - 1}"
    body = blocks[1].split("\n}")[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    rules = {}
    for sel, decls in re.findall(r"([^{}]+)\{([^{}]*)\}", body):
        props = {}
        for d in decls.split(";"):
            if ":" in d:
                k, _, v = d.partition(":")
                props[k.strip()] = v.strip()
        rules[sel.strip()] = props
    return rules


class MobileLayoutTests(unittest.TestCase):
    """Responsive rules the mobile card depends on. These assert the CSS
    contract, not pixel output -- the render itself was measured in a headless
    Chromium at 320/360/390/414/540px, where the lineup table went from
    136-206px of horizontal overflow to 0 at every phone width."""

    def _html(self):
        r = RenderTests()
        g, _ = r._cards()
        return b.cmb_card(g, None)

    def test_four_odds_cells_share_one_unwrapped_row(self):
        html = self._html()
        odds = html.split("<div class='modds'>")[1].split("</div><div class='verdict")[0]
        self.assertEqual(odds.count("<div class='mcell'>"), 4)
        # One flex row that never wraps; cells shrink together instead.
        self.assertIn(".modds{display:flex}", b.CSS)
        self.assertNotIn("flex-wrap:wrap", b.CSS.split(".modds{")[1].split("}")[0])
        self.assertIn(".mcell{flex:1 1 0;min-width:0", b.CSS)
        # The old mobile rule broke the row into 45%-wide pairs.
        self.assertNotIn("flex:1 1 45%", b.CSS)

    def test_verdict_is_its_own_full_width_row(self):
        html = self._html()
        # Verdict sits outside .modds, as a sibling row under it.
        self.assertIn("</div><div class='verdict", html)
        self.assertNotIn("mcell verdict", html)
        # ...at every width: no margin-left:auto / max-width squeeze left over.
        verdict_css = b.CSS.split("\n.verdict{")[1].split("}")[0]
        self.assertNotIn("margin-left:auto", verdict_css)
        self.assertNotIn("max-width", verdict_css)

    def test_no_per_card_build_stamp(self):
        html = self._html()
        self.assertNotIn("as of build", html)
        self.assertNotIn("mcell note", html)
        # The page footer still carries the build time exactly once.
        self.assertIn("built 9:00 AM PT", b._legend_head("MLB matchup leans", "9:00 AM PT"))

    def test_lineups_open_by_default(self):
        html = self._html()
        self.assertEqual(html.count("<details class='lineup' open>"), 2)
        self.assertNotIn("<details class='lineup'>", html)

    def test_lineup_table_has_no_mobile_scroll_floor(self):
        # A 460px min-width forced horizontal scroll inside a ~314px card.
        self.assertNotIn("min-width:460px", b.CSS)
        m = _mobile_rules()
        self.assertEqual(m["table.lu td"]["padding"], "4px 3px")
        # The bar shrinks so the name column has room to fit on one line.
        self.assertLess(int(m[".sl"]["width"].rstrip("px")), 88)
        # The name is the one elastic column: it wraps rather than truncating,
        # and the selector must outrank `table.lu td`'s nowrap (0,1,2).
        nm = m["table.lu td.nm"]
        self.assertEqual(nm["white-space"], "normal")
        self.assertEqual(nm["max-width"], "none")

    def test_mobile_trims_card_padding(self):
        m = _mobile_rules()
        # Each phone padding must be tighter than the desktop rule it overrides.
        for sel, desktop_first in ((".game-summary", 12), (".side", 12),
                                   (".read", 12), (".mcell", 7)):
            pad = m[sel]["padding"].split()
            self.assertLessEqual(int(pad[0].rstrip("px")), desktop_first,
                                 f"{sel} vertical padding not reduced")
            self.assertLess(int(pad[1].rstrip("px")), 16,
                            f"{sel} horizontal padding not reduced")

    def test_scoreboard_keeps_team_center_team_columns(self):
        m = _mobile_rules()
        # Three columns, the middle one fixed: team / centre / team. The centre
        # width tracks the type scale -- it has to hold "7:05 PM ET" on one
        # line, which measures 84px at the Comfortable scale (JetBrains Mono
        # 14px, headless Chromium). Assert the shape and the floor, not a
        # frozen literal, so a type change fails on the thing that matters.
        cols = m[".teams"]["grid-template-columns"].split()
        self.assertEqual(cols[0], "minmax(0,1fr)")
        self.assertEqual(cols[2], "minmax(0,1fr)")
        self.assertGreaterEqual(int(cols[1].rstrip("px")), 84,
                                "centre column cannot hold the time on one line")
        self.assertEqual(m[".summary-time"].get("white-space"), "nowrap",
                         "the time can still wrap if the column ever narrows")
        self.assertIn(".game-card[open]>.game-summary", b.CSS)
        self.assertIn(".summary-chevron", b.CSS)
        self.assertNotIn(".lean", m)
        self.assertNotIn(".lean{", b.CSS)

    def test_mobile_team_meta_stays_under_its_team_name(self):
        m = _mobile_rules()
        self.assertEqual(
            m[".summary-team.away .team-meta"]["justify-self"], "start"
        )
        home = m[".summary-team.home .team-meta"]
        self.assertEqual(home["justify-self"], "end")
        self.assertEqual(home["text-align"], "right")


class StarterBlockTests(unittest.TestCase):
    """`.sp` was left unclosed, nesting the spotlight and the lineup table
    inside the starter block. `.sp .nm` (0,2,0) then outranked `table.lu td`
    (0,1,2) and every hitter name rendered at the starter's 16px/700 instead
    of the table's 12.5px/400 -- which is what made the rows tall enough to
    need horizontal scroll."""

    def _side(self):
        r = RenderTests()
        g, _ = r._cards()
        return b.cmb_card(g, None).split("<section class='side'>")[1] \
                                 .split("</section>")[0]

    def test_div_tags_balance_inside_each_side(self):
        side = self._side()
        self.assertEqual(len(re.findall(r"<div\b", side)),
                         len(re.findall(r"</div>", side)))

    def test_lineup_is_not_nested_inside_the_starter_block(self):
        side = self._side()
        # Everything from `.sp` up to the lineup must close out, so the
        # <details> opens as a sibling of `.sp` rather than a descendant.
        head = side[side.index("<div class='sp'>"):side.index("<details class='lineup'")]
        self.assertEqual(len(re.findall(r"<div\b", head)),
                         len(re.findall(r"</div>", head)),
                         "div.sp still open where the lineup begins")

    def test_starter_name_rule_stays_scoped_to_the_starter(self):
        # The starter's name rule is his alone; the hitter cell keeps its own
        # smaller sans face. That rule has to be qualified -- a bare `td.nm`
        # (0,1,1) loses to `table.lu td`'s mono (0,1,2), which is what closing
        # div.sp exposed. Sizes move with the type scale, so what is asserted
        # is the qualifier and the ordering, not the literals.
        starter = re.search(r"\.sp \.nm\{font:700 ([\d.]+)px/", b.CSS)
        hitter = re.search(r"table\.lu td\.nm\{font:400 ([\d.]+)px/1\.4 var\(--sans\)",
                           b.CSS)
        self.assertIsNotNone(starter, "starter name rule missing or unqualified")
        self.assertIsNotNone(hitter, "hitter name cell lost its sans qualifier")
        self.assertGreater(float(starter.group(1)), float(hitter.group(1)),
                           "starter name no longer outsizes the hitter cell")
        self.assertNotIn("\ntd.nm{font:", b.CSS)


def _token(name, block):
    """Value of a CSS custom property inside a `{...}` token block."""
    m = re.search(rf"{name}:\s*(#[0-9a-fA-F]{{6}})", block)
    assert m, f"{name} not found"
    return m.group(1)


def _relative_luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b_):
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b_)),
                    reverse=True)
    return (la + 0.05) / (lb + 0.05)


class ReadabilityTests(unittest.TestCase):
    """--faint carries real text, so it is held to a text contrast ratio.

    It colours the lineup column headers, the stat labels, the `lg .312` /
    `season 4.04` subs, the machinery line, the `<- 142 open` price tail and
    the `projected` badge -- all under 12px, so WCAG AA is 4.5:1. The shipped
    #98a4af measured 2.54:1 on --surface and 2.24:1 on --surface-2; the dark
    #5f6c78 measured 3.15:1. Both surfaces matter: the market strip and the
    stat cells sit on --surface-2, everything else on --surface."""

    def _theme_blocks(self):
        light = b.CSS.split(":root{", 1)[1].split("}", 1)[0]
        dark = b.CSS.split('html[data-theme="dark"]{', 1)[1].split("}", 1)[0]
        return light, dark

    def test_faint_text_meets_aa_on_both_surfaces(self):
        for label, block in zip(("light", "dark"), self._theme_blocks()):
            faint = _token("--faint", block)
            for surface in ("--surface", "--surface-2"):
                ratio = _contrast(faint, _token(surface, block))
                self.assertGreaterEqual(
                    round(ratio, 2), 4.5,
                    f"{label} --faint {faint} on {surface}: {ratio:.2f}:1")

    def test_faint_stays_lighter_than_muted(self):
        # The fix must not collapse the two-step text hierarchy.
        for block in self._theme_blocks():
            surface = _token("--surface", block)
            self.assertLess(_contrast(_token("--faint", block), surface),
                            _contrast(_token("--muted", block), surface))


def _rgb_token(name, block):
    """Value of an `r,g,b` custom property (the accents) as a hex string."""
    m = re.search(rf"{name}:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", block)
    assert m, f"{name} not found"
    return "#" + "".join(f"{int(v):02x}" for v in m.groups())


def _over(fg_hex, alpha, bg_hex):
    """`fg` at `alpha` composited over opaque `bg` -- what the eye gets when a
    tier pill lays a .08-.12 wash of its own hue over the card surface."""
    f = [int(fg_hex[i:i + 2], 16) for i in (1, 3, 5)]
    b_ = [int(bg_hex[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(f[i] * alpha + b_[i] * (1 - alpha)):02x}"
                         for i in range(3))


class AccentTextContrastTests(unittest.TestCase):
    """The accents double as text colours, and mostly on a tint of themselves.

    `--warm/--cool/--lean` were chosen as fills and borders. They are also set
    as `color` on the tier pills, the LIVE badge, the read-line emphasis, the
    verdict label, the ledger W/L badges and the machinery numbers -- 9 to
    13.5px, so WCAG AA is 4.5:1. Most of those sit on a .08-.12 wash of the
    same hue, which lifts the background toward the text and costs roughly
    half a point of ratio. Measured in headless Chromium at 390px, light
    failed on all three (warm 3.94:1, cool 3.90:1, lean 3.12:1); dark passed.
    The `-tx` twins are the fix, and this is what holds them."""

    # Every tint alpha an accent is set as text over, on either surface. .12
    # is the binding case (the ledger W/L badges); solving the twins against
    # the tier pills' .08 alone left warm at 3.86:1 there.
    ALPHAS = (0.08, 0.10, 0.12)
    TOKENS = ("--warm-tx", "--cool-tx", "--lean-tx")

    def _theme_blocks(self):
        light = b.CSS.split(":root{", 1)[1].split("}", 1)[0]
        dark = b.CSS.split('html[data-theme="dark"]{', 1)[1].split("}", 1)[0]
        return ("light", light), ("dark", dark)

    def test_accent_text_meets_aa_over_its_own_tint(self):
        for label, block in self._theme_blocks():
            for token in self.TOKENS:
                tx = _rgb_token(token, block)
                for alpha in self.ALPHAS:
                    for surface in ("--surface", "--surface-2"):
                        bg = _over(tx, alpha, _token(surface, block))
                        ratio = _contrast(tx, bg)
                        self.assertGreaterEqual(
                            round(ratio, 2), 4.5,
                            f"{label} {token} {tx} on {alpha} tint over "
                            f"{surface}: {ratio:.2f}:1")

    def test_accent_text_meets_aa_on_bare_surfaces(self):
        # `.read .warmtx` and `td.nm .adv` carry no tint at all.
        for label, block in self._theme_blocks():
            for token in self.TOKENS:
                tx = _rgb_token(token, block)
                for surface in ("--surface", "--surface-2"):
                    ratio = _contrast(tx, _token(surface, block))
                    self.assertGreaterEqual(
                        round(ratio, 2), 4.5,
                        f"{label} {token} {tx} on {surface}: {ratio:.2f}:1")

    def test_fills_and_bars_keep_the_base_accents(self):
        # The twins are darker. If they leak into the percentile bars the chart
        # hues shift, which is the thing this split exists to prevent. (The
        # legend swatches this also covered went with the how-to-read guide.)
        self.assertIn(".sl.h i{background:rgba(var(--warm),.85)}", b.CSS)
        self.assertIn(".sl.p i{background:rgba(var(--cool),.85)}", b.CSS)
        self.assertIn("background:rgba(var(--cool),.10)", b.CSS)

    def test_no_accent_is_still_used_as_a_bare_text_colour(self):
        # Any bare `color:rgba(var(--warm|cool|lean),...)` left in the sheet
        # is a spot this audit missed. `border-color` and
        # `text-decoration-color` are not text and keep the base accents, so
        # the lookbehind has to exclude them.
        stray = re.findall(
            r"(?<![-\w])color:\s*rgba\(var\(--(warm|cool|lean)\),", b.CSS)
        self.assertEqual(stray, [], f"accent used as text colour: {stray}")


class TouchTargetTests(unittest.TestCase):
    """44px (iOS HIG) / 48dp (Material) floor on the two controls that missed.

    Measured in headless Chromium at 390px: the theme button rendered 28px
    tall and every lineup disclosure 32px. The collapsed game row already
    cleared it at 82px. Gated on the pointer axis, not a width breakpoint --
    a touch laptop needs the floor and a narrow desktop window does not."""

    def _coarse_block(self):
        # The media query holds nested rule blocks, so stop at the newline-
        # anchored brace that closes the query, not the first `}`.
        return b.CSS.split("@media (pointer:coarse){", 1)[1].split("\n}", 1)[0]

    def test_theme_button_clears_the_floor(self):
        self.assertIn("min-height:44px", self._coarse_block())

    def test_lineup_disclosure_grows_by_padding(self):
        # Padding, so the type and the layout are untouched: 32px of content
        # box plus 14+12 of padding replaces the base 8+6.
        block = self._coarse_block()
        self.assertIn("details.lineup summary{padding-top:14px", block)
        self.assertIn("padding-bottom:12px", block)


class LosingScoreTests(unittest.TestCase):
    """The loser's number recedes, so a final reads as a result at a glance.

    Final only: a trailing team in a live game has not lost anything yet, and
    dimming it would read as a verdict on a game still being played."""

    def _g(self, state, away, home):
        return dict(abstract_state=state, away_score=away, home_score=home,
                    away_streak="W3", home_streak="L1")

    def test_final_dims_only_the_loser(self):
        g = self._g("Final", 6, 5)
        self.assertFalse(b._lost(g, "away"))
        self.assertTrue(b._lost(g, "home"))
        self.assertIn("score away'", b._team_meta_span(g, "away"))
        self.assertIn("score home lost'", b._team_meta_span(g, "home"))

    def test_shutout_dims_the_zero(self):
        g = self._g("Final", 0, 14)
        self.assertTrue(b._lost(g, "away"))
        self.assertFalse(b._lost(g, "home"))

    def test_live_game_dims_nothing(self):
        # 4-7 in the 8th is not a loss.
        g = self._g("Live", 4, 7)
        for side in ("away", "home"):
            self.assertFalse(b._lost(g, side))
            self.assertNotIn("lost", b._team_meta_span(g, side))

    def test_tie_and_missing_score_dim_nothing(self):
        for g in (self._g("Final", 5, 5), self._g("Final", None, 3),
                  self._g("Final", None, None)):
            for side in ("away", "home"):
                self.assertFalse(b._lost(g, side))

    def test_dim_survives_a_client_score_refresh(self):
        # setMeta rewrites className wholesale. If it does not recompute the
        # class, the first background refresh silently undoes every dim on the
        # page -- which is exactly the kind of thing that ships unnoticed.
        js = b.score_refresh_js()
        self.assertIn("(lost?' lost':'')", js)
        self.assertIn("Number(score)<Number(otherScore)", js)
        # and the caller has to hand it the other side's score to compare with
        self.assertIn("setMeta(header.querySelector('.team-meta.away'),"
                      "awayScore,state,homeScore)", js)

    def test_dim_uses_the_faint_text_token(self):
        # --faint is the one tone already held to a contrast ratio by
        # ReadabilityTests; a bespoke grey here would not be.
        self.assertIn(".teams .score.lost{color:var(--faint)}", b.CSS)


class HiddenAttributeTests(unittest.TestCase):
    """`hidden` has to beat component display rules.

    `.game-state{display:inline-block}` outranks the UA sheet's `[hidden]`,
    so the empty pregame badge rendered as a bare bordered chip above the
    first-pitch time on every unstarted card. score_refresh_js re-hides the
    same element by setting `.hidden`, so the JS path needed it too."""

    def test_hidden_attribute_is_enforced(self):
        self.assertIn("[hidden]{display:none!important}", b.CSS)

    def test_pregame_badge_is_emitted_hidden(self):
        g, _ = RenderTests()._cards()
        self.assertIn("<span class='game-state' hidden></span>", b.cmb_card(g, None))


class SidesBreakpointTests(unittest.TestCase):
    """The two-column split may not hand the lineup table a hidden scroll.

    Measured in headless Chromium against a full rendered slate: each
    `.lu-scroll` overflowed its column by 62px at 762px viewport width, 43px
    at 800, 23px at 840, 3px at 880 and 0 from ~900 up."""

    def test_single_column_until_the_table_fits(self):
        m = re.search(r"@media \(max-width:(\d+)px\)\{\s*\.sides\{"
                      r"grid-template-columns:1fr\}", b.CSS)
        self.assertIsNotNone(m, ".sides single-column rule not found")
        self.assertGreaterEqual(int(m.group(1)), 900)

    def test_dead_lineup_name_rules_are_gone(self):
        # The table emits class `nm`; the `.n` twins never matched anything.
        self.assertNotIn("table.lu th.n,", b.CSS)
        self.assertNotIn("\ntd.n{", b.CSS)
        self.assertNotIn("td.n .adv", b.CSS)

    def test_market_cells_bottom_align_their_values(self):
        # "Implied XXX (devig)" wraps to two lines on a phone; without this the
        # fourth number sat a line below the other three.
        mcell = b.CSS.split(".mcell{", 1)[1].split("}", 1)[0]
        self.assertIn("display:flex", mcell)
        self.assertIn("flex-direction:column", mcell)
        self.assertIn("justify-content:space-between", mcell)


class BaseOutStateTests(unittest.TestCase):
    """Scoreboard-app base-out state on the collapsed row.

    The diamond and the out dots only mean something between first pitch and
    the last out, so the markup is always emitted (score_refresh_js fills it in
    place) and `hidden` off the live path."""

    def _live(self, **over):
        g, _ = RenderTests()._cards()
        g.update(abstract_state="Live", status="In Progress",
                 away_score=3, home_score=2, current_inning=9,
                 current_inning_ordinal="9th", inning_half="Top",
                 is_top_inning=True)
        g.update(over)
        return b.cmb_card(g, None)

    def test_live_game_shows_occupied_bases_and_outs(self):
        html = self._live(on_first=True, on_second=True, on_third=False, outs=2)
        bo = html.split("<span class='bo'", 1)[1].split("</span></span>", 1)[0]
        self.assertIn("<i class='b1 on'></i>", bo)
        self.assertIn("<i class='b2 on'></i>", bo)
        self.assertIn("<i class='b3'></i>", bo)          # empty base stays outline
        self.assertEqual(bo.count("<i class='on'></i>"), 2)   # two out dots
        self.assertIn("2 out, runners on first and second", bo)

    def test_indicator_is_hidden_off_the_live_path(self):
        g, _ = RenderTests()._cards()
        for state in (None, "Preview", "Final"):
            g.update(abstract_state=state, on_first=True, outs=1)
            self.assertIn("<span class='bo' hidden>", b.cmb_card(g, None))
        self.assertIn("<span class='bo'>", self._live(on_first=True, outs=1))

    def test_missing_state_degrades_to_empty_bases(self):
        html = self._live()
        bo = html.split("<span class='bo'", 1)[1].split("</span></span>", 1)[0]
        self.assertNotIn(" on'", bo)
        self.assertIn("bases empty", bo)

    def test_base_out_sentences(self):
        self.assertEqual(b._base_out_text((False,) * 3, 0), "0 out, bases empty")
        self.assertEqual(b._base_out_text((True,) * 3, 2), "2 out, bases loaded")
        self.assertEqual(b._base_out_text((False, False, True), 1),
                         "1 out, runner on third")
        # An unknown out count still reads the runners.
        self.assertEqual(b._base_out_text((True, False, False), None),
                         "runner on first")

    def test_refresh_updates_the_indicator_live(self):
        js = b.score_refresh_js()
        self.assertIn("setBaseOut(header.querySelector('.bo')", js)
        self.assertIn("offense.first", js)
        self.assertIn("linescore.outs", js)

    def test_slate_carries_base_out_columns(self):
        # Presence of the key is the signal; StatsAPI omits empty bases.
        payload = {"dates": [{"date": "2026-07-28", "games": [{
            "gamePk": 1, "gameDate": "2026-07-28T23:05:00Z", "gameNumber": 1,
            "status": {"abstractGameState": "Live", "detailedState": "In Progress"},
            "teams": {"away": {"team": {"id": 1, "name": "A", "abbreviation": "AAA"}},
                      "home": {"team": {"id": 2, "name": "H", "abbreviation": "HHH"}}},
            "linescore": {"currentInning": 9, "outs": 2,
                          "offense": {"first": {"id": 10}, "third": {"id": 11}}},
        }]}]}
        with unittest.mock.patch.object(b, "_get_json", return_value=payload):
            row = b.get_slate("2026-07-28").iloc[0]
        self.assertTrue(row["on_first"])
        self.assertFalse(row["on_second"])
        self.assertTrue(row["on_third"])
        self.assertEqual(row["outs"], 2)


class GameClockTests(unittest.TestCase):
    """First-pitch times drop the zone suffix; the page states it once."""

    def test_game_time_has_no_zone_suffix(self):
        self.assertEqual(b._game_time_pt("2026-07-28T23:05:00Z"), "4:05 PM")
        self.assertEqual(b._game_time_pt(""), "")

    def test_zone_is_declared_once_in_the_footer_stamp(self):
        # The how-to-read guide used to carry this; when it was deleted the
        # clause moved to the stamp rather than being dropped, because bare
        # local clocks with no stated local are ambiguous.
        stamp = b._legend_head("MLB matchup leans", "9:00 AM")
        self.assertIn("first pitch times Pacific", stamp)
        html = b.render_combined_html(
            pd.DataFrame(columns=["game_pk", "side"]), pd.DataFrame(),
            pd.DataFrame(), "9:00 AM")
        self.assertEqual(html.count("first pitch times Pacific"), 1)

    def test_build_stamp_keeps_its_zone(self):
        # A timestamp without a zone is ambiguous; a slate of local first
        # pitches is not, once the page has said which local.
        self.assertIn("built 9:00 AM PT",
                      b._legend_head("MLB matchup leans", "9:00 AM PT"))


class ShapeScaleTests(unittest.TestCase):
    """One documented corner-radius scale, mechanically enforced.

    The card had four undocumented radii (2/3/4/6/20px) chosen per-element.
    A mixed scale is only defensible with a written rule, so the rule is the
    token set and this test is the enforcement."""

    TIERS = ("var(--r)", "var(--r-s)", "var(--r-pill)")

    def test_scale_tokens_defined(self):
        for tok, val in (("--r", "6px"), ("--r-s", "3px"), ("--r-pill", "999px")):
            self.assertIn(f"{tok}:{val};", b.CSS + ";")

    def test_no_raw_pixel_radii(self):
        # Every border-radius resolves through a tier. A new one-off px value
        # is the failure this catches.
        raw = [d for d in re.findall(r"border-radius:([^;}]+)", b.CSS)
               if not all(part in self.TIERS or part == "0"
                          for part in d.split())]
        self.assertEqual(raw, [], f"raw px radii outside the scale: {raw}")

    def test_every_radius_is_used_and_tiered(self):
        decls = re.findall(r"border-radius:([^;}]+)", b.CSS)
        self.assertGreater(len(decls), 10)
        # Each tier earns its place; an unused tier is a tier too many.
        for tier in self.TIERS:
            self.assertTrue(any(tier in d for d in decls), f"{tier} unused")

    def test_containers_and_inline_marks_do_not_swap_tiers(self):
        # Spot-check the two ends of the scale so a future edit cannot
        # quietly give a badge a container radius.
        # `.grid` is the container now -- the slate is one surface and the
        # games inside it are hairline-separated rather than boxed.
        self.assertIn("border:1px solid var(--line);border-radius:var(--r);", b.CSS)
        self.assertIn(".card{background:transparent;border:0;border-radius:0", b.CSS)
        self.assertIn("border-radius:var(--r-s);padding:1px 6px}", b.CSS)


class CardCopyTests(unittest.TestCase):
    """Rendered prose carries no em-dash. The character survives only as the
    'no value' placeholder, where the prescribed replacement (a hyphen) would
    read as a minus sign next to signed moneylines like -185."""

    def _card(self, **over):
        r = RenderTests()
        g, _ = r._cards()
        g.update(over)
        return b.cmb_card(g, None)

    def _visible(self, html):
        return re.sub(r"<[^>]+>", " ", html)

    def _prose(self, html):
        """Visible text with whole-cell placeholders removed, so what remains
        is prose. A placeholder is an em-dash that IS the element's entire
        content (`>—<`); anything else is a sentence connector."""
        return self._visible(re.sub(r">\s*[—–]\s*<", "><", html))

    def test_no_em_dash_in_card_prose(self):
        prose = self._prose(self._card())
        self.assertNotIn("—", prose)
        self.assertNotIn("–", prose)          # en-dash separator, same ban
        # Guard the guard: the stripper must not be hiding prose em-dashes.
        self.assertIn("—", self._visible(self._card()))   # placeholder present

    def test_verdict_and_read_use_sentence_punctuation(self):
        html = self._card()
        self.assertIn("Model agrees with the market:", html)
        self.assertIn("That is a <b>", html)

    def test_placeholder_em_dash_survives(self):
        # No market: the em-dash is data, not prose.
        g_html = self._card(odds={})
        self.assertIn("<div class='v'>—</div>", g_html)
        no_edge = self._card(
            odds={}, away=dict(RenderTests()._cards()[0]["away"], xw_edge=None)
        )
        self.assertNotIn("class='lean", no_edge)

    def test_footer_stamp_prose_is_em_dash_free(self):
        # The stamp's own copy, minus the model label it is handed -- that
        # label is a fixed title ("... — Statcast wOBA"), not prose.
        stamp = self._visible(b._legend_head("MLB matchup leans", "9:00 AM"))
        self.assertNotIn("—", stamp)


if __name__ == "__main__":
    unittest.main()
