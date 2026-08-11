"""Structural invariants of the shipped ledger, checked against the real file.

The unit tests elsewhere check the model's functions on constructed frames.
Nothing checked the artifact those functions actually produce, so a drift
between `build_site.py`'s phase decomposition and the columns it persists
would have shown up only when someone read the CSV by hand.

Every assertion here is an *invariant*, never a snapshot: no expected count,
record, or quantile is written down. The ledger grows twice an hour and a
frozen number in a test fails for reasons that have nothing to do with the
diff under test (see `test_record_reproduces_ledger_report`, which was fixed
after doing exactly that).

Rows are immutable once graded, so a violation here is either a live bug in
the writer or a legacy row -- and the legacy carve-outs below are stated as
"pre-vN", not as a count, so a new violation cannot hide inside one.
"""

import os
import re
import unittest

import numpy as np
import pandas as pd

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")
GAME_INNINGS = 9.0
TOL = 1e-9

# The v7 convention: an exact-zero delta is an abstention, not a lean. Rows
# built before v7 recorded a side anyway, so they are excluded by version --
# not by row count, which would let a new violation slip in under the quota.
ABSTENTION_RULE_FROM = 7


def _load():
    return pd.read_csv(LEDGER)


def _num(d, col):
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    return pd.to_numeric(d[col], errors="coerce")


def _version(tag):
    """Trailing version for a recognised production lineage, else None."""
    m = re.fullmatch(r"(xw|woba|split)\+plat_consol_v(\d+)", str(tag))
    return int(m.group(2)) if m else None


class LedgerShapeTest(unittest.TestCase):
    def test_row_identity_is_game_plus_date_not_game_alone(self):
        """A postponed game is re-listed on its makeup date under the same pk.

        `game_pk` alone is therefore NOT unique and must never be used as the
        ledger key -- the original row stays as `void` and the makeup date
        carries the graded one.
        """
        d = _load()
        dup = d.duplicated(subset=["game_pk", "game_date"], keep=False)
        self.assertEqual(int(dup.sum()), 0,
                         "duplicate (game_pk, game_date) rows")
        repeats = d[d.duplicated(subset=["game_pk"], keep=False)]
        for pk, grp in repeats.groupby("game_pk"):
            self.assertEqual(int((grp["status"] == "void").sum()), len(grp) - 1,
                             f"game_pk {pk} repeats without a void original")

    def test_model_tags_are_all_well_formed(self):
        """Guards a typo'd or env-injected tag silently forming its own family."""
        bad = sorted({t for t in _load()["model_tag"].dropna().unique()
                      if _version(t) is None})
        self.assertEqual(bad, [], f"unrecognised model_tag(s): {bad}")


class PhaseAlgebraTest(unittest.TestCase):
    """v9/v10 persist both phases and the blend; they must still reconcile."""

    def test_matchup_is_the_share_weighted_blend_of_its_two_phases(self):
        d = _load()
        for side in ("away", "home"):
            q, bp = _num(d, f"sp_share_{side}"), _num(d, f"bp_share_{side}")
            blend = q * _num(d, f"mx_xwoba_sp_{side}") + bp * _num(d, f"mx_xwoba_bp_{side}")
            m = blend.notna() & _num(d, f"mx_xwoba_{side}").notna()
            self.assertGreater(int(m.sum()), 0, side)
            self.assertLess((blend[m] - _num(d, f"mx_xwoba_{side}")[m]).abs().max(),
                            TOL, f"mx_xwoba_{side} != q*sp + (1-q)*bp")

    def test_phase_shares_partition_the_game(self):
        d = _load()
        for side in ("away", "home"):
            q, bp = _num(d, f"sp_share_{side}"), _num(d, f"bp_share_{side}")
            m = q.notna() & bp.notna()
            self.assertGreater(int(m.sum()), 0, side)
            self.assertLess((q[m] + bp[m] - 1.0).abs().max(), TOL,
                            f"sp_share_{side} + bp_share_{side} != 1")
            self.assertTrue(((q[m] >= 0) & (q[m] <= 1)).all(), side)

    def test_every_phase_edge_is_measured_off_the_same_league_baseline(self):
        """edge = mx - L for each phase, so mx - edge must recover one L."""
        d = _load()
        for side in ("away", "home"):
            l_all = _num(d, f"mx_xwoba_{side}") - _num(d, f"edge_xwoba_{side}")
            l_sp = _num(d, f"mx_xwoba_sp_{side}") - _num(d, f"edge_xwoba_sp_{side}")
            l_bp = _num(d, f"mx_xwoba_bp_{side}") - _num(d, f"edge_xwoba_bp_{side}")
            m = l_all.notna() & l_sp.notna() & l_bp.notna()
            self.assertGreater(int(m.sum()), 0, side)
            self.assertLess((l_all[m] - l_sp[m]).abs().max(), TOL, side)
            self.assertLess((l_all[m] - l_bp[m]).abs().max(), TOL, side)

    def test_starter_share_is_the_pa_share_when_rates_were_measured(self):
        """v10's whole change: weight by PA share, using measured BF/IP.

        Where a rate is missing the weight degrades to the innings share --
        continuously, since the two coincide when the rates are equal.
        """
        d = _load()
        for side in ("away", "home"):
            q = _num(d, f"sp_share_{side}")
            ip = _num(d, f"expected_sp_ip_{side}").clip(0.0, GAME_INNINGS)
            r_sp, r_bp = _num(d, f"sp_bf_per_ip_{side}"), _num(d, f"bp_bf_per_ip_{side}")
            measured = (r_sp > 0) & (r_bp > 0)

            pa_sp, pa_bp = ip * r_sp, (GAME_INNINGS - ip) * r_bp
            pa_share = pa_sp / (pa_sp + pa_bp)
            m = q.notna() & pa_share.notna() & measured
            self.assertGreater(int(m.sum()), 0, side)
            self.assertLess((q[m] - pa_share[m]).abs().max(), TOL,
                            f"sp_share_{side} is not the measured PA share")

            m2 = q.notna() & ip.notna() & ~measured
            if m2.any():
                self.assertLess((q[m2] - ip[m2] / GAME_INNINGS).abs().max(), TOL,
                                f"sp_share_{side} did not fall back to innings share")


class LeanConventionTest(unittest.TestCase):
    def test_delta_is_the_magnitude_of_the_net(self):
        d = _load()
        net, delta = _num(d, "xw_net"), _num(d, "xw_delta")
        m = net.notna() & delta.notna()
        self.assertGreater(int(m.sum()), 0)
        self.assertLess((delta[m] - net[m].abs()).max(), TOL)

    def test_lean_side_follows_the_sign_of_the_net(self):
        d = _load()
        net = _num(d, "xw_net")
        want = pd.Series(np.where(net > 0, d["home"], np.where(net < 0, d["away"], "")),
                         index=d.index)
        m = net.notna() & net.ne(0)
        self.assertGreater(int(m.sum()), 0)
        self.assertTrue((want[m] == d["xw_lean"][m]).all(),
                        "xw_lean disagrees with sign(xw_net)")

    def test_zero_delta_abstains_on_every_row_built_since_v7(self):
        d = _load()
        net = _num(d, "xw_net")
        offending = net.eq(0) & d["xw_lean"].notna()
        versions = {_version(t) for t in d.loc[offending, "model_tag"]}
        self.assertTrue(all(v is not None and v < ABSTENTION_RULE_FROM for v in versions),
                        f"a lean was recorded at xw_net == 0 under tag version(s) "
                        f"{sorted(v for v in versions if v is None or v >= ABSTENTION_RULE_FROM)}")


class GradingTest(unittest.TestCase):
    def test_shipped_grades_agree_with_the_linescores_beside_them(self):
        d = _load()
        for away_col, home_col, grade_col in (("full_away", "full_home", "xw_full"),
                                              ("f5_away", "f5_home", "xw_f5")):
            margin = _num(d, home_col) - _num(d, away_col)
            signed = pd.Series(
                np.where(d["xw_lean"] == d["home"], margin,
                         np.where(d["xw_lean"] == d["away"], -margin, np.nan)),
                index=d.index)
            want = pd.Series(np.where(signed > 0, "W",
                                      np.where(signed < 0, "L",
                                               np.where(signed == 0, "T", None))),
                             index=d.index)
            m = want.notna() & d[grade_col].notna()
            self.assertGreater(int(m.sum()), 0, grade_col)
            self.assertTrue((want[m] == d[grade_col][m]).all(),
                            f"{grade_col} disagrees with {away_col}/{home_col}")

    def test_pending_rows_carry_no_closing_line(self):
        """run_market_update.py's no-lookahead invariant, checked on the artifact.

        A close attached to a row that has not been graded is a closing price
        recorded against a game whose result the ledger does not yet know --
        the shape of a lookahead leak.
        """
        d = _load()
        pending = d["status"].eq("pending")
        # An empty pending set is a legitimate state of the artifact, not a
        # violation: between the morning grading pass and the first pregame
        # build of the next slate every row is settled, and on 2026-08-11 the
        # committed ledger held 485 graded / 7 void / 0 pending -- which turned
        # this gate red for a reason with nothing to do with lookahead. The
        # invariant is vacuous there, so skip rather than assert a precondition
        # the bot controls.
        if not int(pending.sum()):
            self.skipTest("no pending rows in the committed ledger; "
                          "the invariant is vacuous")
        closes = [c for c in ("close_away_ml", "close_home_ml", "close_p_home",
                              "f5_close_away_ml", "f5_close_home_ml", "f5_close_p_home")
                  if c in d.columns]
        leaked = pending & d[closes].notna().any(axis=1)
        self.assertEqual(int(leaked.sum()), 0,
                         f"{int(leaked.sum())} pending row(s) carry a closing line")


if __name__ == "__main__":
    unittest.main()
