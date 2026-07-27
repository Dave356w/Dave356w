"""Card redesign: percentile bars, ledger-ranked lean strength, the
plain-language read, the model-vs-market verdict, and the always-shown model
machinery.

All display-only -- these guard the render layer, not the lean math."""
import re
import unittest

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
                game_datetime_utc=None, time_pt="7:05 PM ET", venue="Park", odds=odds,
                league_baseline={"xwOBA": .312, "K%": .22, "BB%": .08, "OPS": .715, "ERA": 4.1})


class LeanStrengthTests(unittest.TestCase):
    def test_fixed_fallback_buckets(self):
        # No ledger scale -> fixed cutoffs (0.021 / 0.060), no percentile.
        self.assertEqual(b.lean_strength(0.010, None)[0], "slight")
        self.assertEqual(b.lean_strength(0.040, None)[0], "clear")
        self.assertEqual(b.lean_strength(0.090, None)[0], "strong")
        self.assertIsNone(b.lean_strength(0.040, None)[1])

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
        # Too few scale-compatible rows -> None (fixed cutoffs), NOT a fallback
        # to the 200 incompatible pre-v5 rows.
        rows = ([("xw+plat_consol_v6", 0.015)] * 10
                + [("xw+plat_consol_v2", 0.040)] * 200)
        from unittest import mock
        with mock.patch.object(b, "load_ledger_df", return_value=self._ledger(rows)), \
                mock.patch.object(b, "SCALE_TAGS", ("xw+plat_consol_v5", "xw+plat_consol_v6")):
            self.assertIsNone(b.lean_strength_scale())


class PercentileTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 400
        pa = rng.integers(5, 650, n)
        true = rng.normal(.315, .030, n)
        obs = true + rng.normal(0, .35 / np.sqrt(np.maximum(pa, 1)), n)
        self.cust = pd.DataFrame({"xwoba": obs, "pa": pa})

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

    def test_browser_refresh_uses_one_slate_request_per_minute(self):
        js = b.score_refresh_js()
        self.assertIn("statsapi.mlb.com/api/v1/schedule", js)
        self.assertIn("setInterval(refresh,60000)", js)
        self.assertIn("state==='Live'||state==='Final'", js)


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
                 away_score=3, home_score=2)
        html = b.cmb_card(g, None)
        self.assertIn("class='team-meta score away'", html)
        self.assertIn("class='team-meta score home'", html)
        self.assertIn(">3</span>", html)
        self.assertIn(">2</span>", html)
        self.assertIn("class='game-state live'", html)
        self.assertIn(">LIVE</span>", html)
        self.assertNotIn(">W3</span>", html)
        self.assertNotIn(">L1</span>", html)

    def test_final_game_replaces_streaks_with_final_score(self):
        g, _ = self._cards()
        g.update(abstract_state="Final", status="Final",
                 away_score=6, home_score=4)
        html = b.cmb_card(g, None)
        self.assertIn(">6</span>", html)
        self.assertIn(">4</span>", html)
        self.assertIn("class='game-state final'", html)
        self.assertIn(">FINAL</span>", html)

    def test_condensed_footer_keeps_only_casual_guide(self):
        html = b._legend_guide()
        self.assertIn("How to read a card", html)
        self.assertIn("warmer / longer bar", html)
        self.assertIn("xwOBA %ile", html)
        self.assertIn("Model vs market", html)
        self.assertIn("platoon advantage", html)
        self.assertNotIn("role-filtered bullpen", html)
        self.assertNotIn("0.010 xwOBA", html)

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

    def test_team_names_and_abbreviations_link_to_mlb_standings(self):
        g, _ = self._cards()
        html = b.cmb_card(g, None)
        href = "href='https://www.mlb.com/standings/'"
        self.assertGreaterEqual(html.count(href), 12)
        self.assertIn(f"{href} target='_blank' rel='noopener noreferrer'>LAD</a>", html)
        self.assertIn(f"{href} target='_blank' rel='noopener noreferrer'>ARI</a>", html)
        self.assertIn(".team-link:focus-visible", b.CSS)

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

    def test_no_lean_pill_when_edge_missing(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = None
        self.assertIn("no lean", b.cmb_card(g, None))

    def test_exact_zero_delta_is_neutral_not_home(self):
        g, _ = self._cards()
        g["away"]["xw_edge"] = .012345
        g["home"]["xw_edge"] = .012345
        html = b.cmb_card(g, None)
        self.assertIn("no lean", html)
        self.assertNotIn("<span class='lt'><a", html)

    def test_nonzero_sub_display_delta_still_names_a_favorite(self):
        # A delta too small to have printed at three decimals is still a lean:
        # the pill must name the side, not collapse to the neutral "no lean".
        g, _ = self._cards()
        g["away"]["xw_edge"] = .01234567891
        g["home"]["xw_edge"] = .01234567890
        html = b.cmb_card(g, None)
        self.assertNotIn("no lean", html)
        self.assertIn("<span class='lt'><a class='team-link'", html)
        self.assertIn(">ARI</a></span>", html)


class LeanPillTests(unittest.TestCase):
    """The pill shows the ledger-ranked strength word only. |Δxw| itself lives
    in the ledger (xw_net / xw_delta, written on every row), so it is relocated
    rather than discarded -- the card is not the record."""

    def _card(self, **over):
        r = RenderTests()
        g, _ = r._cards()
        for k, v in over.items():
            g[k[:4]]["xw_edge"] = v
        return b.cmb_card(g, None)

    def test_pill_keeps_strength_word_without_the_number(self):
        html = self._card()
        self.assertIn("<span class='ls'>strong</span>", html)
        self.assertIn("class='lean strong'", html)
        self.assertNotIn("Δxw", html)
        self.assertNotIn("<span class='mach'> · Δ", html)

    def test_neutral_pill_carries_no_number_either(self):
        html = self._card(away=None)
        self.assertIn("<span class='ls'>no lean</span>", html)
        self.assertNotIn("Δxw", html)

    def test_delta_styling_hook_is_gone_from_css(self):
        # .ld styled the removed delta readout; .lean .ls .mach styled its
        # mono suffix. Neither has markup left to hook.
        self.assertNotIn(".lean .ld{", b.CSS)
        self.assertNotIn(".lean .ls .mach{", b.CSS)


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
        # The page header still carries the build time exactly once.
        self.assertIn("built 9:00 AM PT", b._legend_head("MLB matchup leans", "9:00 AM PT"))

    def test_lineups_open_by_default(self):
        html = self._html()
        self.assertEqual(html.count("<details class='lineup' open>"), 2)
        self.assertNotIn("<details class='lineup'>", html)

    def test_lineup_table_has_no_mobile_scroll_floor(self):
        # A 460px min-width forced horizontal scroll inside a ~314px card.
        self.assertNotIn("min-width:460px", b.CSS)
        m = _mobile_rules()
        self.assertEqual(m["table.lu th,table.lu td"]["padding"], "4px 3px")
        # The bar shrinks so the name column has room to fit on one line.
        self.assertLess(int(m[".sl"]["width"].rstrip("px")), 88)
        # The name is the one elastic column: it wraps rather than truncating,
        # and the selector must outrank `table.lu td`'s nowrap (0,1,2).
        nm = m["table.lu td.nm,table.lu td.n"]
        self.assertEqual(nm["white-space"], "normal")
        self.assertEqual(nm["max-width"], "none")

    def test_mobile_trims_card_padding(self):
        m = _mobile_rules()
        # Each phone padding must be tighter than the desktop rule it overrides.
        for sel, desktop_first in ((".gamehead", 12), (".side", 12),
                                   (".read", 12), (".mcell", 7)):
            pad = m[sel]["padding"].split()
            self.assertLessEqual(int(pad[0].rstrip("px")), desktop_first,
                                 f"{sel} vertical padding not reduced")
            self.assertLess(int(pad[1].rstrip("px")), 16,
                            f"{sel} horizontal padding not reduced")

    def test_gamehead_wraps_teams_then_time_and_lean(self):
        m = _mobile_rules()
        self.assertEqual(m[".teams"]["flex"], "1 1 100%")     # teams own a line
        self.assertEqual(m[".lean"]["margin-left"], "auto")   # lean right of time
        # The old rule pinned the lean hard-left under the teams.
        self.assertNotIn(".lean{margin-left:0}", b.CSS)


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
        # The 16px/700 rule is the starter's alone; the hitter cell keeps its
        # own smaller sans face. That rule has to be qualified -- a bare
        # `td.nm` (0,1,1) loses to `table.lu td`'s mono (0,1,2), which is what
        # closing div.sp exposed.
        self.assertIn(".sp .nm{font:700 16px/1.2 var(--sans)}", b.CSS)
        self.assertIn("table.lu td.nm{font:400 12.5px/1.4 var(--sans)", b.CSS)
        self.assertNotIn("\ntd.nm{font:", b.CSS)


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
        self.assertIn(".card{background:var(--surface);border:1px solid var(--line);"
                      "border-radius:var(--r)", b.CSS)
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
        # No market and no lean: the em-dash is data, not prose.
        g_html = self._card(odds={})
        self.assertIn("<div class='v'>—</div>", g_html)
        self.assertIn("<span class='lt'>—</span>",
                      self._card(odds={}, away=dict(RenderTests()._cards()[0]["away"],
                                                    xw_edge=None)))

    def test_legend_prose_is_em_dash_free(self):
        self.assertNotIn("—", self._visible(b._legend_guide()))


if __name__ == "__main__":
    unittest.main()
