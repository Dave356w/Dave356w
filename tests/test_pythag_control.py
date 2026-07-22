"""Walk-forward + merge-contract tests for the Pythagorean control arm.

These encode the two guarantees the arm makes:

  * NO LOOKAHEAD — a game on date D uses only games FINAL strictly before D.
    Checked by an independent brute-force recount of prior finals, per team
    and per venue split (the same 40-game audit cited in the design note).
  * MERGE CONTRACT — the ledger merge mirrors market_backfill: verified join on
    gamePk, existing values never overwritten, unmatched rows kept NaN for the
    next run, idempotent on re-run.

Run: python -m unittest tests.test_pythag_control
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import pythag_control as pc


def _synth_season(n_dates=40, n_teams=10, pending_tail=2, seed=0):
    """A deterministic synthetic schedule: round-robin-ish daily slates of
    Poisson-scored finals, with the last `pending_tail` dates left unplayed."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    strength = {t: rng.normal(4.6, 0.5) for t in teams}
    dates = pd.date_range("2026-04-01", periods=n_dates, freq="D")
    rows, pk = [], 1000
    for d in dates:
        ds = d.date().isoformat()
        order = teams[:]
        rng.shuffle(order)
        for a, h in zip(order[0::2], order[1::2]):
            pk += 1
            final = d < dates[-pending_tail]
            hr = max(0, int(rng.poisson(strength[h])))
            ar = max(0, int(rng.poisson(strength[a])))
            if hr == ar:
                hr += 1  # no ties in baseball
            rows.append(dict(
                gamePk=str(pk), date=ds, commence=ds + "T18:00:00Z",
                home=h, away=a, venue=f"{h} Park",
                status_detailed="Final" if final else "Scheduled",
                is_final=final,
                final_home_runs=float(hr) if final else np.nan,
                final_away_runs=float(ar) if final else np.nan))
    games = pd.DataFrame(rows)
    games["is_doubleheader_pair"] = games.duplicated(["date", "home", "away"], keep=False)
    return games


class NoLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.games = _synth_season()
        self.wf = pc.build_walkforward(self.games)
        self.finals = self.games[self.games["is_final"]
                                 & self.games["final_home_runs"].notna()]

    def _prior(self, team, date):
        f = self.finals
        pooled = int(((f["date"] < date)
                      & ((f["home"] == team) | (f["away"] == team))).sum())
        split_home = int(((f["date"] < date) & (f["home"] == team)).sum())
        split_away = int(((f["date"] < date) & (f["away"] == team)).sum())
        return pooled, split_home, split_away

    def test_prior_counts_match_bruteforce_recount(self):
        """The 40-game leakage audit: every prior-game count equals an
        independent strictly-prior recount. Any lookahead shows up here."""
        sample = self.wf.sample(min(40, len(self.wf)), random_state=7)
        for _, r in sample.iterrows():
            hp, hsh, _ = self._prior(r["home"], r["date"])
            ap, _, asa = self._prior(r["away"], r["date"])
            self.assertEqual(int(r["home_prior_games_all"]), hp)
            self.assertEqual(int(r["away_prior_games_all"]), ap)
            # home club's split state is its HOME games; away club's, its AWAY games
            self.assertEqual(int(r["home_prior_games_split"]), hsh)
            self.assertEqual(int(r["away_prior_games_split"]), asa)

    def test_opening_date_has_no_prior_state(self):
        first = self.wf["date"].min()
        op = self.wf[self.wf["date"] == first]
        self.assertTrue((op["home_prior_games_all"] == 0).all())
        self.assertTrue((op["away_prior_games_all"] == 0).all())
        # with no team state, every opening game falls back to the league prior
        self.assertTrue((op["pythag_basis"] == "league").all())

    def test_probabilities_are_complementary(self):
        self.assertTrue(np.allclose(
            self.wf["pythag_p_home"] + self.wf["pythag_p_away"], 1.0))

    def test_doubleheader_pair_shares_one_probability(self):
        """DH pairs share prior state by construction, so they share the
        estimate — flagged, not hidden."""
        g = _synth_season(seed=1)
        # force a same-day rematch (a doubleheader) into the schedule
        row = g.iloc[0].copy()
        row["gamePk"] = "999999"
        g = pd.concat([g, pd.DataFrame([row])], ignore_index=True)
        g["is_doubleheader_pair"] = g.duplicated(["date", "home", "away"], keep=False)
        wf = pc.build_walkforward(g)
        pair = wf[wf["is_doubleheader_pair"]]
        self.assertGreaterEqual(len(pair), 2)
        for _, grp in pair.groupby(["date", "home", "away"]):
            self.assertLessEqual(grp["pythag_p_home"].nunique(), 1)

    def test_grade_matches_realized_winner(self):
        d = self.wf[self.wf["home_win"].notna()]
        picked_home = d["pythag_p_home"] >= 0.5
        correct = (d["home_win"] == picked_home.astype(float))
        self.assertTrue(((d["pythag_grade_fg"] == "W") == correct).all())
        self.assertFalse((self.wf["pythag_grade_fg"] == "pending").all())


class MergeContractTests(unittest.TestCase):
    def _ledger(self, path, pks):
        pd.DataFrame({
            "game_pk": pks, "gamePk": pks,
            "game_date": "2026-07-01", "away": "A", "home": "H",
            "close_p_home": 0.5,
        }).to_csv(path, index=False)

    def _wf(self, pks):
        return pd.DataFrame({
            "gamePk": [str(p) for p in pks],
            "date": "2026-07-01", "home": "H", "away": "A",
            "pythag_tag": "pythag_v1", "pythag_basis": "pooled",
            "pythag_p_home": np.linspace(0.4, 0.6, len(pks)),
            "pythag_split_p_home": np.linspace(0.41, 0.59, len(pks)),
            "pythag_lean": "H", "pythag_lean_p": 0.55,
            "pythag_grade_fg": "W", "league_hfa_p": 0.529,
            "home_prior_games_all": 50, "away_prior_games_all": 50,
        })

    def test_join_fills_only_covered_rows_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "led.csv")
            all_pks = [str(1000 + i) for i in range(20)]
            self._ledger(path, all_pks)
            wf = self._wf(all_pks[:15])  # 5 rows deliberately not backfilled yet

            pc.merge_into_ledger(wf, path)
            a = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            self.assertEqual(len(a), 20)                       # no row growth
            self.assertEqual(a["pythag_p_home"].notna().sum(), 15)
            self.assertEqual(a["pythag_p_home"].isna().sum(), 5)  # retried next run

            pc.merge_into_ledger(wf, path)                      # re-run
            b = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            self.assertTrue((a["pythag_p_home"].fillna("X")
                             == b["pythag_p_home"].fillna("X")).all())

    def test_existing_value_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "led.csv")
            pks = [str(1000 + i) for i in range(5)]
            self._ledger(path, pks)
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            df["pythag_p_home"] = pd.Series([np.nan] * len(df), dtype="object")
            df.loc[0, "pythag_p_home"] = "0.999"               # human-set sentinel
            df.to_csv(path, index=False)

            wf = self._wf(pks)
            wf.loc[0, "pythag_p_home"] = 0.111                 # conflicting value
            pc.merge_into_ledger(wf, path)
            out = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            self.assertEqual(out.loc[0, "pythag_p_home"], "0.999")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "led.csv")
            pks = [str(1000 + i) for i in range(5)]
            self._ledger(path, pks)
            before = open(path, encoding="utf-8").read()
            pc.merge_into_ledger(self._wf(pks), path, dry_run=True)
            self.assertEqual(open(path, encoding="utf-8").read(), before)


if __name__ == "__main__":
    unittest.main()
