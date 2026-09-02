"""The season leaderboard is a display surface over the model's own shrinkage.

Two boards, one shrink function. What these tests pin is mostly the reasoning
in `leaderboard_boards`' comments rather than the arithmetic, because the
arithmetic is `shrink_xwoba` -- already covered where it lives:

  * K is the model's K and the regression is the model's, so a leaderboard
    number is the same number the lean is built on;
  * there is NO playing-time qualifier and none is needed -- the regression
    does that work continuously, which is the threshold-cliff fix this repo
    has already applied twice;
  * the two boards regress toward DIFFERENT targets, matching the two
    percentile references, so a leaderboard rank cannot disagree with the
    percentile bar on the same player's card;
  * "SP" is `relief_pitcher_ids`' own predicate, not a batters-faced proxy;
  * the bio lookup is proportional to what is PUBLISHED, not to the season
    board, and its failure costs names rather than the page.
"""

import re
import unittest

import numpy as np
import pandas as pd

import build_site as b


SRC = b.MODEL_RATE_SOURCE_COL
LG = {b.MODEL_RATE_INTERNAL_COL: 0.3150, "_pctile_prior_bat": 0.3080}


def _cust(rows):
    """rows: (player_id, pa, rate)."""
    return pd.DataFrame([{"player_id": p, "pa": n, SRC: r} for p, n, r in rows])


def _roles(**by_id):
    return {int(p): {"start_share": s, "avg_ip_per_appearance": 5.4 if s > .5 else 1.0}
            for p, s in by_id.items()}


def _people(rows):
    """rows: (player_id, name, pos)."""
    return {int(p): {"name": nm, "pos": ps} for p, nm, ps in rows}


class ShrunkRateBoardTests(unittest.TestCase):
    def test_it_is_the_models_own_shrinkage(self):
        board = b.shrunk_rate_board(_cust([(1, 200, 0.400)]), 0.300, 100.0)
        expected = float(b.shrink_xwoba(pd.Series([0.400]), pd.Series([200]),
                                        0.300, 100.0).iloc[0])
        self.assertAlmostEqual(float(board["shrunk"].iloc[0]), expected, places=12)
        # (200*.400 + 100*.300)/300
        self.assertAlmostEqual(float(board["shrunk"].iloc[0]), 0.3666666667, places=9)

    def test_default_k_is_the_model_constant(self):
        self.assertEqual(b.XWOBA_SHRINK_K, 100.0)
        boards = b.leaderboard_boards(
            _cust([(1, 100, .400)]), _cust([(9, 100, .250)]), LG,
            _people([(1, "A", "C")]), _roles(**{"9": 0.9}))
        self.assertEqual(boards["meta"]["k"], b.XWOBA_SHRINK_K)

    def test_a_thin_sample_is_pulled_almost_all_the_way_to_the_target(self):
        """The reason no qualifier is needed, stated as a number."""
        board = b.shrunk_rate_board(_cust([(1, 12, 0.700)]), 0.310, 100.0)
        got = float(board["shrunk"].iloc[0])
        self.assertAlmostEqual(got, (12 * .700 + 100 * .310) / 112, places=12)
        self.assertLess(got - 0.310, 0.048)      # 12 PA buys < 5 points

    def test_an_unusable_board_is_empty_not_none(self):
        for bad in (None, pd.DataFrame(), pd.DataFrame({"player_id": [1]})):
            with self.subTest(bad=type(bad)):
                out = b.shrunk_rate_board(bad, 0.31, 100.0)
                self.assertTrue(out.empty)
                self.assertEqual(list(out.columns),
                                 ["player_id", "pa", "rate", "shrunk"])

    def test_zero_and_missing_samples_are_dropped(self):
        out = b.shrunk_rate_board(
            _cust([(1, 0, .400), (2, np.nan, .400), (3, 50, np.nan), (4, 50, .400)]),
            0.310, 100.0)
        self.assertEqual(list(out["player_id"]), [4])

    def test_keep_ids_filters_who_without_changing_what(self):
        rows = [(1, 300, .280), (2, 300, .300), (3, 300, .320)]
        full = b.shrunk_rate_board(_cust(rows), 0.310, 100.0)
        cut = b.shrunk_rate_board(_cust(rows), 0.310, 100.0, keep_ids={1, 3})
        self.assertEqual(list(cut["player_id"]), [1, 3])
        for pid in (1, 3):
            self.assertAlmostEqual(
                float(cut.loc[cut.player_id == pid, "shrunk"].iloc[0]),
                float(full.loc[full.player_id == pid, "shrunk"].iloc[0]),
                places=12)


class StarterIdentificationTests(unittest.TestCase):
    def test_sp_uses_the_same_predicate_as_the_relief_pool(self):
        """Not a BF proxy. The boundary is RP_MAX_START_SHARE, exclusive."""
        cust = _cust([(1, 500, .3), (2, 500, .3), (3, 500, .3)])
        roles = {1: {"start_share": b.RP_MAX_START_SHARE + .01},
                 2: {"start_share": b.RP_MAX_START_SHARE},
                 3: {"start_share": b.RP_MAX_START_SHARE - .01}}
        self.assertEqual(b.starter_ids(cust, roles), {1})

    def test_a_high_workload_reliever_is_not_a_starter(self):
        """The proxy this deliberately avoids would rank him as one."""
        cust = _cust([(1, 900, .250), (2, 90, .250)])
        ids = b.starter_ids(cust, {1: {"start_share": 0.0},
                                   2: {"start_share": 1.0}})
        self.assertEqual(ids, {2})

    def test_no_role_data_is_cannot_say_not_an_empty_rotation(self):
        cust = _cust([(1, 500, .3)])
        self.assertIsNone(b.starter_ids(cust, {}))
        self.assertIsNone(b.starter_ids(cust, None))

    def test_a_pitcher_with_no_role_line_is_omitted_not_assumed(self):
        cust = _cust([(1, 500, .3), (2, 500, .3)])
        self.assertEqual(b.starter_ids(cust, {1: {"start_share": 0.9}}), {1})

    def test_a_null_start_share_is_not_a_starter(self):
        cust = _cust([(1, 500, .3)])
        self.assertEqual(b.starter_ids(cust, {1: {"start_share": np.nan}}), set())


def _demo(n_pit=140, n_bat=300, seed=3):
    rng = np.random.default_rng(seed)
    pit = _cust([(1000 + i, int(rng.integers(5, 700)),
                  round(float(rng.normal(.315, .045)), 4)) for i in range(n_pit)])
    bat = _cust([(5000 + i, int(rng.integers(1, 650)),
                  round(float(rng.normal(.318, .050)), 4)) for i in range(n_bat)])
    roles = {int(p): {"start_share": 0.9 if i % 2 == 0 else 0.05}
             for i, p in enumerate(pit["player_id"])}
    pos = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    people = _people([(p, f"Bat {p}", pos[i % len(pos)])
                      for i, p in enumerate(bat["player_id"])])
    return bat, pit, people, roles


class BoardShapeTests(unittest.TestCase):
    def setUp(self):
        self.bat, self.pit, self.people, self.roles = _demo()
        self.boards = b.leaderboard_boards(
            self.bat, self.pit, LG, self.people, self.roles,
            load_people=lambda ids: {int(i): {"name": f"SP {i}"} for i in ids})

    def test_the_sp_board_is_capped_and_ascending(self):
        sp = self.boards["sp"]
        self.assertLessEqual(len(sp), b.LEADERBOARD_SP_N)
        self.assertTrue(sp["shrunk"].is_monotonic_increasing)

    def test_each_position_board_is_capped_and_descending(self):
        for pos, g in self.boards["bat"]:
            with self.subTest(pos=pos):
                self.assertLessEqual(len(g), b.LEADERBOARD_BAT_N)
                self.assertTrue(g["shrunk"].is_monotonic_decreasing)

    def test_lower_is_better_for_pitchers_and_higher_for_batters(self):
        """The two boards sort in OPPOSITE directions; pin it explicitly."""
        sp = self.boards["sp"]
        self.assertEqual(float(sp["shrunk"].iloc[0]), float(sp["shrunk"].min()))
        _, g = self.boards["bat"][0]
        self.assertEqual(float(g["shrunk"].iloc[0]), float(g["shrunk"].max()))

    def test_the_published_counts_are_the_requested_ones(self):
        self.assertEqual(b.LEADERBOARD_SP_N, 100)
        self.assertEqual(b.LEADERBOARD_BAT_N, 15)

    def test_every_row_carries_a_name_and_its_sample(self):
        frames = [self.boards["sp"]] + [g for _, g in self.boards["bat"]]
        for f in frames:
            self.assertTrue(f["name"].notna().all())
            self.assertTrue(f["pa"].notna().all())
            self.assertTrue(f["rate"].notna().all())


class PositionGroupingTests(unittest.TestCase):
    def _boards(self, people_rows, bat_rows=None):
        bat = _cust(bat_rows or [(p, 300, .350) for p, _, _ in people_rows])
        return b.leaderboard_boards(bat, _cust([(9, 500, .25)]), LG,
                                    _people(people_rows), _roles(**{"9": .9}))

    def test_a_pitcher_who_batted_is_not_on_a_batter_board(self):
        boards = self._boards([(1, "Bat", "C"), (2, "Arm", "P")])
        self.assertEqual([p for p, _ in boards["bat"]], ["C"])

    def test_a_batter_with_no_position_is_not_filed_under_a_guess(self):
        boards = self._boards([(1, "Bat", "C"), (2, "Unknown", None)])
        ranked = {int(i) for _, g in boards["bat"] for i in g["player_id"]}
        self.assertEqual(ranked, {1})

    def test_positions_render_in_fielding_order(self):
        rows = [(i, f"P{i}", pos) for i, pos in
                enumerate(["DH", "C", "SS", "1B"], start=1)]
        boards = self._boards(rows)
        self.assertEqual([p for p, _ in boards["bat"]], ["C", "1B", "SS", "DH"])

    def test_an_unlisted_position_is_appended_not_dropped(self):
        """A position we failed to anticipate must look unfamiliar, not vanish."""
        rows = [(1, "A", "C"), (2, "B", "PH"), (3, "C", "1B")]
        boards = self._boards(rows)
        self.assertEqual([p for p, _ in boards["bat"]], ["C", "1B", "PH"])

    def test_every_ordered_position_sorts_ahead_of_every_unordered_one(self):
        for known in b.LEADERBOARD_POS_ORDER:
            with self.subTest(pos=known):
                self.assertLess(b._pos_sort_key(known), b._pos_sort_key("ZZZ"))


class ShrinkTargetTests(unittest.TestCase):
    """The two boards use two targets, matching the two percentile references."""

    def test_batters_use_the_pool_centre_and_pitchers_the_league_rate(self):
        boards = b.leaderboard_boards(
            _cust([(1, 100, .400)]), _cust([(9, 100, .250)]), LG,
            _people([(1, "A", "C")]), _roles(**{"9": 0.9}))
        self.assertAlmostEqual(boards["meta"]["prior_bat"], 0.3080, places=9)
        self.assertAlmostEqual(boards["meta"]["prior_pit"], 0.3150, places=9)
        _, g = boards["bat"][0]
        self.assertAlmostEqual(float(g["shrunk"].iloc[0]),
                               (100 * .400 + 100 * .3080) / 200, places=12)
        self.assertAlmostEqual(float(boards["sp"]["shrunk"].iloc[0]),
                               (100 * .250 + 100 * .3150) / 200, places=12)

    def test_the_batter_target_is_the_one_the_percentile_bars_rank_against(self):
        """Otherwise a #1 on this page could show a middling bar on his card.

        `_pctile_prior_bat` is what fetch_all stashes from `pool_shrink_target`
        and what `build_pctile_ref` was handed; reading it here is what keeps
        the leaderboard's order and the card's percentile the same statement.
        """
        cust = _cust([(i, 300, 0.250 + 0.001 * i) for i in range(1, 61)])
        target = b.pool_shrink_target(cust)
        lg = {b.MODEL_RATE_INTERNAL_COL: 0.3150, "_pctile_prior_bat": target}
        ref, _ = b.build_pctile_ref(cust, target, b.XWOBA_SHRINK_K, None)
        boards = b.leaderboard_boards(
            cust, _cust([(9, 500, .25)]), lg,
            _people([(i, f"B{i}", "C") for i in range(1, 61)]),
            _roles(**{"9": .9}))
        _, g = boards["bat"][0]
        # Top of the board = top of the reference distribution it is ranked in.
        self.assertAlmostEqual(float(g["shrunk"].iloc[0]), float(ref[-1]), places=12)
        pct = b.pctile_rank(float(g["rate"].iloc[0]), float(g["pa"].iloc[0]),
                            ref, target, b.XWOBA_SHRINK_K)
        self.assertEqual(pct, 100.0)

    def test_a_missing_batter_target_falls_back_to_the_league_rate(self):
        boards = b.leaderboard_boards(
            _cust([(1, 100, .400)]), _cust([(9, 100, .250)]),
            {b.MODEL_RATE_INTERNAL_COL: 0.3150}, _people([(1, "A", "C")]),
            _roles(**{"9": 0.9}))
        self.assertAlmostEqual(boards["meta"]["prior_bat"], 0.3150, places=9)

    def test_no_target_at_all_yields_no_boards(self):
        self.assertIsNone(b.leaderboard_boards(
            _cust([(1, 100, .4)]), _cust([(9, 100, .25)]), {}, {}, {}))


class BioLookupTests(unittest.TestCase):
    def test_the_lookup_is_sized_by_what_is_published_not_by_the_board(self):
        bat, pit, people, roles = _demo(n_pit=400, n_bat=600)
        seen = []

        def fake(ids):
            seen.append(list(ids))
            return {int(i): {"name": f"SP {i}"} for i in ids}

        boards = b.leaderboard_boards(bat, pit, LG, people, roles,
                                      load_people=fake)
        self.assertEqual(len(seen), 1, "one batched call, after selection")
        self.assertEqual(set(seen[0]), set(boards["sp"]["player_id"]))
        self.assertLessEqual(len(seen[0]), b.LEADERBOARD_SP_N)
        self.assertLess(len(seen[0]), len(pit))

    def test_batters_need_no_lookup_because_the_build_already_has_them(self):
        bat, pit, people, roles = _demo()
        seen = []
        b.leaderboard_boards(bat, pit, LG, people, roles,
                             load_people=lambda ids: seen.extend(ids) or {})
        self.assertFalse(set(seen) & set(bat["player_id"]))

    def test_a_failing_lookup_costs_names_not_the_page(self):
        bat, pit, people, roles = _demo()

        def boom(ids):
            raise RuntimeError("statsapi down")

        boards = b.leaderboard_boards(bat, pit, LG, people, roles,
                                      load_people=boom)
        self.assertTrue(len(boards["sp"]))
        self.assertTrue(boards["sp"]["name"].str.startswith("player ").all())

    def test_no_lookup_at_all_still_renders(self):
        bat, pit, people, roles = _demo()
        boards = b.leaderboard_boards(bat, pit, LG, people, roles)
        self.assertTrue(len(boards["sp"]))
        self.assertIn("Season leaderboard",
                      b.render_leaderboard_html("now", boards))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.bat, self.pit, self.people, self.roles = _demo()
        self.boards = b.leaderboard_boards(
            self.bat, self.pit, LG, self.people, self.roles,
            load_people=lambda ids: {int(i): {"name": f"SP {i}"} for i in ids})
        self.html = b.render_leaderboard_html("built now", self.boards)

    def test_it_links_back_to_the_leans_page(self):
        self.assertIn("index.html", self.html)

    def test_the_leans_page_links_to_it(self):
        """The request was for a page linked FROM the leans page."""
        self.assertIn("leaderboard.html", b.records_strip_html())

    def test_it_states_the_shrinkage_it_used(self):
        """The copy is terse now; the constant is the one thing it must keep."""
        self.assertIn("K&nbsp;=&nbsp;100", self.html)

    def test_it_states_that_there_is_no_qualifier(self):
        self.assertIn("No playing-time cut", self.html)

    def test_it_reports_role_coverage_rather_than_asserting_it(self):
        """Report provenance, don't assert coverage -- survives the trim."""
        self.assertIn("qualified as starters", self.html)
        self.assertIn("of the", self.html)

    def test_it_reports_how_many_batters_were_ranked(self):
        self.assertRegex(self.html, r"\d+ of \d+ batters ranked")

    def test_an_empty_role_map_says_so_instead_of_showing_a_short_rotation(self):
        boards = b.leaderboard_boards(self.bat, self.pit, LG, self.people, {})
        html = b.render_leaderboard_html("now", boards)
        self.assertIn("No role data on this build", html)
        self.assertEqual(len(boards["sp"]), 0)

    def test_the_copy_is_short(self):
        """The trim is the requirement, so it needs a test that would notice.

        Prose here means the lead paragraphs, not the rows: a board of 15
        players is not verbose. Counted as characters of `gr-lead` text, which
        is where every description on this page lives.
        """
        leads = re.findall(r"<div class='gr-lead'>(.*?)</div>", self.html)
        prose = re.sub(r"<[^>]+>", "", " ".join(leads))
        self.assertLess(len(prose), 500, f"lead copy grew back to {len(prose)}")
        self.assertLessEqual(len(leads), 3, "one lead per section at most")

    def test_the_copy_does_not_grow_with_the_number_of_boards(self):
        """Nine tables each explaining themselves is what got trimmed.

        Counting leads rather than grepping after a marker: the Batters
        divider carries one legitimately, so the invariant is that adding
        positions adds tables and not prose.
        """
        def n_leads(people_rows, bat_rows):
            boards = b.leaderboard_boards(
                _cust(bat_rows), _cust([(9000, 500, .25)]), LG,
                _people(people_rows), _roles(**{"9000": .9}))
            html = b.render_leaderboard_html("now", boards)
            return len(re.findall(r"<div class='gr-lead'>", html)), len(boards["bat"])

        few, n_few = n_leads([(1, "a", "C"), (2, "b", "SS")],
                             [(1, 300, .40), (2, 300, .39)])
        many, n_many = n_leads(
            [(i, f"p{i}", p) for i, p in
             enumerate(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"], 1)],
            [(i, 300, .40) for i in range(1, 10)])
        self.assertGreater(n_many, n_few)
        self.assertEqual(few, many, "lead copy scales with the board count")


    def test_no_boards_renders_a_page_rather_than_raising(self):
        for empty in (None, {}):
            with self.subTest(boards=empty):
                html = b.render_leaderboard_html("now", empty)
                self.assertIn("Leaderboard unavailable", html)
                self.assertIn("index.html", html)

    def test_a_player_name_is_escaped(self):
        bat = _cust([(1, 300, .400)])
        boards = b.leaderboard_boards(
            bat, _cust([(9, 500, .25)]), LG,
            _people([(1, "<script>x</script>", "C")]), _roles(**{"9": .9}))
        html = b.render_leaderboard_html("now", boards)
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_it_reuses_the_sites_table_idiom(self):
        """No second table system: same wrap, same classes, same phone labels."""
        for cls in ("gr-tablewrap", "table class='gr'", "c-game", "data-l="):
            with self.subTest(cls=cls):
                self.assertIn(cls, self.html)


class HeadingTests(unittest.TestCase):
    """A heading is plain text and is escaped exactly once.

    Shipped once escaped twice: the title arrived carrying an `&mdash;` and
    `_leaderboard_board_html` escaped it again, so every position header on the
    live page read `C &mdash; top 15`. These pin the rule rather than the
    instance -- any markup in an escaped value fails the second test.
    """

    def setUp(self):
        bat, pit, people, roles = _demo()
        self.boards = b.leaderboard_boards(bat, pit, LG, people, roles)
        self.html = b.render_leaderboard_html("now", self.boards)
        self.heads = re.findall(r"<h[12][^>]*>(.*?)</h[12]>", self.html)

    def test_no_heading_contains_a_double_escaped_entity(self):
        for h in self.heads:
            with self.subTest(heading=h):
                self.assertNotIn("&amp;", h)

    def test_a_position_heading_is_just_the_position_and_its_count(self):
        got = [h for h in self.heads if h.startswith("C<")]
        self.assertEqual(len(got), 1)
        self.assertRegex(got[0], r"^C<span class='lb-n'>\d+</span>$")

    def test_the_heading_count_is_the_rows_shown_not_the_cap(self):
        """A board of nine must not be headed 15."""
        people = _people([(i, f"B{i}", "C") for i in range(1, 10)])
        bat = _cust([(i, 300, 0.300 + 0.001 * i) for i in range(1, 10)])
        boards = b.leaderboard_boards(bat, _cust([(9000, 500, .25)]), LG,
                                      people, _roles(**{"9000": .9}))
        html = b.render_leaderboard_html("now", boards)
        self.assertIn("<span class='lb-n'>9</span>", html)
        self.assertNotIn("<span class='lb-n'>15</span>", html)

    def test_the_batter_boards_are_announced_once(self):
        """One divider, not a description above each of nine tables."""
        self.assertEqual(self.html.count(">Batters</h2>"), 1)


class PositionPoolingTests(unittest.TestCase):
    """`OF` and `TWP` pool into DH; nothing else pools.

    Neither names a corner this page can rank against its own kind -- `OF` is
    the same job as LF/CF/RF but cannot be assigned to one of them without
    inventing which, and `TWP` is one or two players a season. DH is the
    bat-only board, so that is where they land.
    """

    def _boards(self, people_rows):
        bat = _cust([(p, 300, .300 + .001 * p) for p, _, _ in people_rows])
        return b.leaderboard_boards(bat, _cust([(9000, 500, .25)]), LG,
                                    _people(people_rows),
                                    _roles(**{"9000": .9}))

    def test_the_pool_map_is_exactly_of_and_twp_to_dh(self):
        self.assertEqual(b.LEADERBOARD_POS_POOL, {"OF": "DH", "TWP": "DH"})

    def test_neither_of_nor_twp_is_published_as_its_own_board(self):
        boards = self._boards([(1, "a", "OF"), (2, "b", "TWP"), (3, "c", "DH")])
        self.assertEqual([p for p, _ in boards["bat"]], ["DH"])

    def test_a_pooled_player_is_ranked_on_the_dh_board(self):
        boards = self._boards([(1, "of", "OF"), (2, "twp", "TWP"),
                               (3, "dh", "DH")])
        names = list(dict(boards["bat"])["DH"]["name"])
        self.assertCountEqual(names, ["of", "twp", "dh"])

    def test_pooling_happens_before_the_cut_so_slots_are_competed_for(self):
        """A pooled player displaces a weaker DH rather than being appended."""
        rows = ([(i, f"dh{i}", "DH") for i in range(1, 16)]
                + [(99, "of", "OF")])
        bat = _cust([(p, 300, .300) for p, _, _ in rows[:-1]]
                    + [(99, 300, .900)])
        boards = b.leaderboard_boards(bat, _cust([(9000, 500, .25)]), LG,
                                      _people(rows), _roles(**{"9000": .9}))
        dh = dict(boards["bat"])["DH"]
        self.assertEqual(len(dh), b.LEADERBOARD_BAT_N)
        self.assertEqual(dh["name"].iloc[0], "of")

    def test_the_specific_outfield_boards_are_untouched(self):
        boards = self._boards([(1, "a", "LF"), (2, "b", "CF"), (3, "c", "RF"),
                               (4, "d", "OF")])
        self.assertEqual([p for p, _ in boards["bat"]], ["LF", "CF", "RF", "DH"])

    def test_an_unpooled_unknown_position_still_stands_alone(self):
        """Pooling is a named map, not a catch-all for anything unfamiliar."""
        boards = self._boards([(1, "a", "DH"), (2, "b", "PH")])
        self.assertEqual([p for p, _ in boards["bat"]], ["DH", "PH"])

    def test_of_and_twp_are_no_longer_in_the_display_order(self):
        for pos in ("OF", "TWP"):
            with self.subTest(pos=pos):
                self.assertNotIn(pos, b.LEADERBOARD_POS_ORDER)

class FailureIsolationTests(unittest.TestCase):
    def test_write_leaderboard_page_never_raises(self):
        class Boom:
            def __getitem__(self, k):
                raise RuntimeError("bad boards")

        self.assertIsNone(b.write_leaderboard_page("now", Boom()))

    def test_the_page_is_display_only(self):
        """No dump column, no ledger column, no MODEL_TAG dependency.

        A leaderboard that could move a lean would need a model-version
        decision; this one cannot, and that is why the PR bumps nothing.
        """
        bat, pit, people, roles = _demo()
        boards = b.leaderboard_boards(bat, pit, LG, people, roles)
        cols = set(boards["sp"].columns)
        for _, g in boards["bat"]:
            cols |= set(g.columns)
        self.assertEqual(cols, {"player_id", "pa", "rate", "shrunk", "name", "pos"}
                         | {"player_id", "pa", "rate", "shrunk", "name"})
        self.assertNotIn("model_tag", cols)
        self.assertNotIn("xw_lean", cols)
