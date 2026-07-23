"""Team cap-logo inlining and standout-surname display.

Both display-only -- these guard the render/asset layer, not the lean math.
The logo tests seed the on-disk cache so they never touch the MLB CDN.
"""
import os
import tempfile
import unittest

import build_site as b


class LastNameSurnameTests(unittest.TestCase):
    """_last powers the Standouts pills and the plain-language read; it must
    return the surname, not a generational suffix."""

    def test_strips_generational_suffix(self):
        self.assertEqual(b._last("Fernando Tatis Jr."), "Tatis")
        self.assertEqual(b._last("Michael Harris II"), "Harris")
        self.assertEqual(b._last("Ken Griffey Jr."), "Griffey")
        self.assertEqual(b._last("Ronald Acuna III"), "Acuna")
        self.assertEqual(b._last("Some Player IV"), "Player")

    def test_plain_names_unchanged(self):
        self.assertEqual(b._last("Ozzie Albies"), "Albies")
        self.assertEqual(b._last("Sung-Mun Song"), "Song")

    def test_degenerate_input(self):
        # A bare suffix (never a real full name) and empty input must not blow up
        # or return an empty surname where a token exists.
        self.assertEqual(b._last("Jr."), "Jr.")
        self.assertEqual(b._last(""), "")

    def test_spotlight_pill_shows_surname(self):
        hitters = [dict(name="Fernando Tatis Jr.", xw_pctile=96),
                   dict(name="Michael Harris II", xw_pctile=88)]
        html = b._spotlight_html(hitters, n=3, thresh=70)
        self.assertIn("Tatis", html)
        self.assertIn("Harris", html)
        self.assertNotIn(">Jr. ", html)
        self.assertNotIn(">II ", html)


class TeamLogoTests(unittest.TestCase):
    def setUp(self):
        # Isolated cache dir + a clean memo/breaker per test.
        self._tmp = tempfile.TemporaryDirectory()
        self._cache_patch = b.CACHE_DIR
        b.CACHE_DIR = self._tmp.name
        b._team_logo_cache.clear()
        b._logo_cdn_down = False

    def tearDown(self):
        b.CACHE_DIR = self._cache_patch
        b._team_logo_cache.clear()
        b._logo_cdn_down = False
        self._tmp.cleanup()

    def _seed(self, tid, variant, body="<svg>x</svg>"):
        path = os.path.join(b.CACHE_DIR, f"logo_cap_{variant}_{tid}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)

    def test_both_variants_resolve_distinctly(self):
        self._seed(135, "light", "<svg>L</svg>")
        self._seed(135, "dark", "<svg>D</svg>")
        lt, dk = b.team_logo_uris(135)
        self.assertTrue(lt and dk and lt != dk)
        self.assertTrue(lt.startswith("data:image/svg+xml;base64,"))

    def test_single_variant_used_for_both(self):
        # Only the light cap is cached; the missing dark falls back to light so
        # the team still gets a chip rather than dropping to text.
        self._seed(144, "light", "<svg>L</svg>")
        b._logo_cdn_down = True          # force the dark fetch to skip the network
        lt, dk = b.team_logo_uris(144)
        self.assertTrue(lt and dk and lt == dk)

    def test_unresolved_team_returns_none(self):
        b._logo_cdn_down = True          # nothing cached, network disabled
        self.assertEqual(b.team_logo_uris(999), (None, None))

    def test_disabled_flag_short_circuits(self):
        prev = b.USE_TEAM_LOGOS
        b.USE_TEAM_LOGOS = False
        try:
            self.assertEqual(b.team_logo_uris(135), (None, None))
        finally:
            b.USE_TEAM_LOGOS = prev

    def test_logo_assets_css_and_dedup(self):
        for tid in (135, 144):
            self._seed(tid, "light", f"<svg>L{tid}</svg>")
            self._seed(tid, "dark", f"<svg>D{tid}</svg>")
        games = [{"away_team_id": 135, "home_team_id": 144},
                 {"away_team_id": 999, "home_team_id": 135}]  # 999 unresolved; 135 repeats
        b._logo_cdn_down = True           # 999 has no cache -> stays unresolved
        assets, css = b._logo_assets(games)
        self.assertEqual(sorted(assets), [135, 144])
        # Theme-scoped swap mirrors the palette selectors.
        self.assertIn("prefers-color-scheme:dark", css)
        self.assertIn('html[data-theme="dark"]', css)
        # Each team's data URIs appear exactly once (custom-property dedup) even
        # though 135 is on two games.
        self.assertEqual(css.count(f'.t135{{'), 1)
        self.assertEqual(css.count(assets[135][0]), 1)

    def test_club_logo_chip_only_for_resolved(self):
        ctx = {"logo_ids": {135}}
        self.assertIn("clogo t135", b._club_logo(135, ctx))
        self.assertEqual(b._club_logo(144, ctx), "")     # not in resolved set
        self.assertEqual(b._club_logo(None, ctx), "")
        self.assertEqual(b._club_logo(135, {}), "")      # no logo_ids in ctx

    def test_circuit_breaker_trips_on_failure(self):
        # No cache + a fetch that raises -> breaker trips so later teams skip the
        # network entirely. Force the session to fail fast.
        class _Boom:
            def get(self, *a, **k):
                raise RuntimeError("cdn down")
        prev = b.session
        b.session = _Boom()
        try:
            self.assertEqual(b.team_logo_uris(135), (None, None))
            self.assertTrue(b._logo_cdn_down)
            # A second uncached team must not attempt the network again.
            b.session = None             # any network use would AttributeError
            self.assertEqual(b.team_logo_uris(144), (None, None))
        finally:
            b.session = prev


if __name__ == "__main__":
    unittest.main()
