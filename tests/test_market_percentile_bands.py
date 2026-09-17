"""Properties of the equal-count market calibration block.

Deliberately no counts, records or edges: the bands are recomputed from the
rows every build, so any literal here would be a frozen-from-data constant one
artifact out -- the thing the block's own caveat warns the reader about. What
is pinned is the behaviour that makes the block readable.
"""
import unittest

import numpy as np
import pandas as pd

import grade_leans as gl


def _ledger(n=240, tag="xw+plat_consol_v12", seed=0, spread=True):
    """Graded rows with closing prices spanning a plausible book."""
    rng = np.random.default_rng(seed)
    # Real (home, away) pairs only. American odds never fall strictly between
    # -100 and +100, so a fixture that derives one side by arithmetic invents
    # prices the book cannot quote -- which is what the label test catches.
    book = [(-350, 285), (-240, 200), (-175, 155), (-140, 120), (-120, 100),
            (-105, -105), (105, -125), (120, -140), (140, -160), (175, -195),
            (240, -280), (320, -380)]
    pick = rng.integers(0, len(book), size=n)
    ml = np.array([book[i][0] for i in pick]) if spread else np.full(n, -120)
    other = np.array([book[i][1] for i in pick]) if spread else np.full(n, 100)
    p_home = np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    home_runs = rng.integers(0, 9, n)
    away_runs = np.where(home_runs == 4, 7, 4)
    # The other side of a real book flips sign: a -120 favourite faces a
    # plus-money dog, never a "-80", which American odds cannot represent.
    return pd.DataFrame({
        "game_pk": np.arange(n) + 1, "game_date": "2026-09-01",
        "status": "graded", "model_tag": tag,
        "close_p_home": p_home, "close_home_ml": ml, "close_away_ml": other,
        "full_home": home_runs, "full_away": away_runs,
        "home": "H", "away": "A", "xw_lean": "H", "xw_full": "W",
        "xw_net": 0.02,
    })


class MarketPercentileBandTests(unittest.TestCase):
    def test_it_never_publishes_the_pooled_both_sides_identity(self):
        """Pooling every side is forced to 50.0 vs 50.0 and is not a result."""
        lines = gl._market_percentile_band_lines(_ledger())
        self.assertTrue(lines)
        for ln in lines:
            stripped = ln.strip().lower()
            self.assertFalse(
                stripped.startswith("all ") or stripped.startswith("pooled "),
                f"a pooled both-sides row would be an identity: {ln!r}")

    def test_the_bands_are_actually_balanced(self):
        """The block's whole reason to exist is equal n; assert it delivers."""
        lines = gl._market_percentile_band_lines(_ledger(n=400, seed=3))
        ns = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 8 and ".." in parts[0]:
                ns.append(int(parts[1]))
        self.assertGreaterEqual(len(ns), 4)
        # Ties are kept whole, so this is balance rather than exact equality.
        self.assertLessEqual(max(ns) / min(ns), 2.0,
                             f"bands are meant to be equal-count, got {ns}")

    def test_it_scores_every_family_not_just_the_current_one(self):
        """A rate against a devigged close does not know which model wrote the row."""
        cur = _ledger(n=200, tag=gl.RECORD_TAGS[0], seed=1)
        old = _ledger(n=200, tag="woba+plat_consol_v5", seed=2)
        old["game_pk"] = old["game_pk"] + 10_000
        both = pd.concat([cur, old], ignore_index=True)
        head = gl._market_percentile_band_lines(both)[1]
        self.assertIn("400 games", head,
                      "scoping to RECORD_TAGS would halve the sample for nothing")

    def test_the_pooling_verdict_follows_the_number_it_prints(self):
        """The licence sentence is derived, never asserted."""
        cur = _ledger(n=220, tag=gl.RECORD_TAGS[0], seed=5)
        old = _ledger(n=220, tag="woba+plat_consol_v5", seed=6)
        old["game_pk"] = old["game_pk"] + 10_000
        lines = gl._market_percentile_band_lines(
            pd.concat([cur, old], ignore_index=True))
        head = [l for l in lines if "pooling licence" in l]
        tail = [l for l in lines if l.strip().startswith("against ")
                and "expected from noise" in l]
        self.assertTrue(head and tail, "two families must produce a licence line")
        got = float(head[0].split("max |z|")[1].split()[0].rstrip(","))
        exp = float(tail[0].strip().split()[1])
        text = head[0] + " " + tail[0]
        if got <= exp:
            self.assertIn("licensed", text)
            self.assertNotIn("ABOVE", text)
        else:
            self.assertIn("ABOVE", text)
            self.assertNotIn("no sign", text)

    def test_a_band_label_never_names_an_impossible_moneyline(self):
        """Nothing lies strictly between -100 and +100; a label must be checkable."""
        for ln in gl._market_percentile_band_lines(_ledger(n=400, seed=7)):
            parts = ln.split()
            if not (len(parts) >= 8 and ".." in parts[0]):
                continue
            for edge in parts[0].split(".."):
                value = int(edge)
                self.assertFalse(-100 < value < 100,
                                 f"{value} is not a real moneyline, in {ln!r}")

    def test_a_thin_ledger_says_so_rather_than_banding_noise(self):
        lines = gl._market_percentile_band_lines(_ledger(n=6))
        self.assertTrue(any("fewer than" in l for l in lines), lines)

    def test_it_never_takes_the_report_down(self):
        self.assertEqual(gl._market_percentile_band_lines(pd.DataFrame()), [])
        self.assertEqual(gl._market_percentile_band_lines(None), [])
        junk = pd.DataFrame({"status": ["graded"]})
        self.assertIn("unavailable", " ".join(gl._market_percentile_band_lines(junk)))


if __name__ == "__main__":
    unittest.main()
