#!/usr/bin/env python3
"""
pythag_control.py — walk-forward Pythagorean control arm for the MLB lean ledger.

Purpose
-------
Produce a deliberately naive, no-lookahead win probability for every game, so the
xwOBA and platoon-OPS lenses can be measured against something other than the
market. This is a CONTROL, not a lean: it uses season-to-date runs scored/allowed
and nothing else — no probables, no lineups, no park factors, no opponent
adjustment, no market prices.

Two variants are computed for every game:

  pooled  team strength from ALL prior final games, Log5, plus an explicit
          league home-field term.   -> `pythag_p_home`
  split   team strength from venue-matched prior final games only (home club's
          home games, road club's road games), Log5, plus the same league
          home-field term applied relative to the league's average split
          matchup.                  -> `pythag_split_p_home`

Both are empirical-Bayes shrunk toward the walk-forward league mean rate.

Why the home-field term is explicit
-----------------------------------
Venue-split run rates do NOT carry home-field advantage. Home teams skip the
bottom of the ninth when they are ahead, so their runs scored are truncated
exactly in the games they win. In the 2026 YTD sample the league's home clubs
outscored road clubs by 0.013 R/G while winning 52.5% of games; feeding those
split rates through Log5 yields a mean P(home) of 0.506 against a market mean of
0.530. The home edge has to be added back as a league constant, walk-forward.

No-lookahead guarantees
-----------------------
  * team state for a game on date D uses only games FINAL on dates strictly < D
    (merge_asof with allow_exact_matches=False);
  * league means, league home-win rate, and the shrinkage denominators are all
    computed on the same strictly-prior window;
  * non-final games never contribute to any state;
  * doubleheaders share one probability by construction — flagged, not hidden.

Usage
-----
  # one-off season backfill -> data/pythag_walkforward_2026.csv
  python pythag_control.py --season 2026

  # backfill + idempotent merge into the lean ledger
  python pythag_control.py --season 2026 --ledger data/mlb_lean_ledger.csv --merge-ledger

  # today's slate only, for the matchup cards
  python pythag_control.py --season 2026 --slate-date 2026-07-22 --out-slate data/pythag_slate.csv

  # evaluation report against market + existing lean arms
  python pythag_control.py --season 2026 --ledger data/mlb_lean_ledger.csv --report

Design notes worth keeping visible
----------------------------------
  * `PYTHAG_TAG` is versioned separately from `MODEL_TAG`. Changing the exponent,
    the shrinkage weights, or the home-field prior must bump it; the xwOBA and
    platoon records are unaffected by this arm and must not be re-tagged.
  * Rows never carry a silent default. Every row states its `pythag_basis`
    (`split` / `pooled` / `league`) and its prior-game counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"

PYTHAG_TAG = os.environ.get("PYTHAG_TAG", "pythag_v1")

# ── model constants (bump PYTHAG_TAG if any of these change) ──────────────────
PYTHAG_EXPONENT = 1.83

# Empirical-Bayes prior weight, in games, for shrinking a club's run rate toward
# the walk-forward league mean. Derived from a variance decomposition on 2026 YTD
# split rates: observed cross-club variance minus sampling variance (sigma^2_game
# ~ 9.6 runs) implies k ~ 20-90 games by rate for venue splits and k ~ 125 pooled.
# A single conservative value is used for both rather than two tuned constants,
# because tuning k on the same season the arm is evaluated in is a lookahead.
SHRINK_GAMES_SPLIT = 70.0
SHRINK_GAMES_POOLED = 120.0

# League home-field prior, blended with the walk-forward season-to-date home win
# rate. Weight is in games; 400 keeps April sane without dominating by June.
HFA_PRIOR_P = 0.530
HFA_PRIOR_WEIGHT = 400.0

# Below this many prior games in the venue split, the split variant falls back to
# pooled; below the pooled floor it falls back to the league average matchup.
MIN_SPLIT_GAMES = 12
MIN_POOLED_GAMES = 20

OUT_COLS = [
    "gamePk", "date", "away", "home", "venue",
    "pythag_tag", "pythag_basis",
    "home_prior_games_all", "away_prior_games_all",
    "home_prior_games_split", "away_prior_games_split",
    "home_rs_hat", "home_ra_hat", "away_rs_hat", "away_ra_hat",
    "home_strength", "away_strength",
    "league_hfa_p", "pythag_p_home", "pythag_p_away",
    "pythag_split_p_home",
    "pythag_lean", "pythag_lean_p",
    "final_home_runs", "final_away_runs", "home_win",
    "pythag_grade_fg", "is_doubleheader_pair",
]


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _pythag(rs, ra, exponent: float = PYTHAG_EXPONENT):
    rs = np.asarray(rs, float)
    ra = np.asarray(ra, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.power(np.clip(rs, 1e-6, None), exponent)
        b = np.power(np.clip(ra, 1e-6, None), exponent)
        return a / (a + b)


def _log5(a, b):
    """Head-to-head probability for strength `a` vs strength `b`, both on a
    common 0.500 baseline."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    num = a * (1.0 - b)
    den = num + (1.0 - a) * b
    return np.where(den > 1e-12, num / den, 0.5)


def _brier(p, y) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    return float(((p[m] - y[m]) ** 2).mean()) if m.sum() else float("nan")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def _cluster_boot_mean_ci(values, clusters, n_boot: int = 2000, seed: int = 7):
    """Mean and percentile bootstrap CI, resampling whole dates.

    Same helper as the market audit. Games on one slate share weather, umpires,
    and market conditions, and doubleheader pairs share a Pythagorean estimate,
    so the effective sample is closer to the number of dates than the number of
    games.
    """
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    m = np.isfinite(v) & pd.notna(c)
    v, c = v[m], c[m]
    if len(v) == 0:
        return (float("nan"),) * 3
    uniq = np.unique(c)
    sums = np.array([v[c == u].sum() for u in uniq], float)
    cnts = np.array([(c == u).sum() for u in uniq], float)
    est = float(v.mean())
    if len(uniq) < 2 or n_boot <= 0:
        return (est, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (est, float(lo), float(hi))


# ─────────────────────────────────────────────────────────────────────────────
# schedule
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(url: str, params: dict, tries: int = 3, timeout: int = 30):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": f"pythag-control/{PYTHAG_TAG}"})
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.25 * (k + 1))
                continue
            return None
        except Exception as e:
            last = str(e)
            time.sleep(1.25 * (k + 1))
    print(f"[warn] GET failed {url} {params}: {last}")
    return None


def _is_final(status: dict) -> bool:
    st = status or {}
    if str(st.get("abstractGameState", "")).lower() == "final":
        return True
    if str(st.get("codedGameState", "")).upper() in {"F", "O"}:
        return True
    d = str(st.get("detailedState", "")).lower()
    return any(x in d for x in ("final", "game over", "completed"))


def fetch_schedule(season: int, start: str, end: str, game_types: str = "R") -> pd.DataFrame:
    """One StatsAPI call. Returns one row per gamePk, finals scored."""
    js = _get_json(f"{MLB_API}/schedule", {
        "sportId": 1, "season": season, "startDate": start, "endDate": end,
        "gameTypes": game_types, "hydrate": "team,venue,linescore",
    }) or {}

    rows: list[dict[str, Any]] = []
    for d in js.get("dates", []) or []:
        for g in d.get("games", []) or []:
            teams = g.get("teams") or {}
            h, a = teams.get("home") or {}, teams.get("away") or {}
            ls = ((g.get("linescore") or {}).get("teams") or {})
            fin = _is_final(g.get("status") or {})
            hr = (ls.get("home") or {}).get("runs", h.get("score"))
            ar = (ls.get("away") or {}).get("runs", a.get("score"))
            rows.append({
                "gamePk": str(g.get("gamePk", "")),
                "date": d.get("date", ""),
                "commence": g.get("gameDate", ""),
                "home": (h.get("team") or {}).get("name", ""),
                "away": (a.get("team") or {}).get("name", ""),
                "venue": (g.get("venue") or {}).get("name", ""),
                "status_detailed": (g.get("status") or {}).get("detailedState", ""),
                "is_final": bool(fin),
                "final_home_runs": float(hr) if fin and hr is not None else np.nan,
                "final_away_runs": float(ar) if fin and ar is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("StatsAPI schedule returned no rows — check season/start/end.")

    # Dedupe repeated gamePk rows (suspended/resumed games recur in wide ranges),
    # keeping the most complete record.
    df["_q"] = df["final_home_runs"].notna().astype(int) * 2 + df["is_final"].astype(int)
    df = (df.sort_values("_q", ascending=False)
            .drop_duplicates("gamePk", keep="first")
            .drop(columns="_q")
            .sort_values(["date", "commence", "gamePk"])
            .reset_index(drop=True))

    df["is_doubleheader_pair"] = df.duplicated(["date", "home", "away"], keep=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# walk-forward state
# ─────────────────────────────────────────────────────────────────────────────

def _team_long(finals: pd.DataFrame) -> pd.DataFrame:
    """One row per team per final game, with venue split."""
    home = pd.DataFrame({
        "team": finals["home"], "date": finals["date"], "split": "home",
        "rs": finals["final_home_runs"], "ra": finals["final_away_runs"],
        "won": (finals["final_home_runs"] > finals["final_away_runs"]).astype(float),
    })
    away = pd.DataFrame({
        "team": finals["away"], "date": finals["date"], "split": "away",
        "rs": finals["final_away_runs"], "ra": finals["final_home_runs"],
        "won": (finals["final_away_runs"] > finals["final_home_runs"]).astype(float),
    })
    long = pd.concat([home, away], ignore_index=True)
    long["g"] = 1.0
    long["_d"] = pd.to_datetime(long["date"], errors="coerce")
    return long


def _asof_state(long: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    """Cumulative rs/ra/g by `keys` through each played date, keyed for a
    strictly-prior asof join."""
    daily = (long.groupby(keys + ["_d"], as_index=False)[["rs", "ra", "g"]]
                 .sum()
                 .sort_values("_d"))
    daily[[f"{prefix}_rs", f"{prefix}_ra", f"{prefix}_g"]] = (
        daily.groupby(keys)[["rs", "ra", "g"]].cumsum()
    )
    return daily[keys + ["_d", f"{prefix}_rs", f"{prefix}_ra", f"{prefix}_g"]]


def _join_prior(games: pd.DataFrame, state: pd.DataFrame,
                by: list[str], left_on: dict[str, str]) -> pd.DataFrame:
    """merge_asof on date, strictly prior (allow_exact_matches=False)."""
    left = games.rename(columns=left_on).sort_values("_d")
    right = state.sort_values("_d")
    return pd.merge_asof(left, right, on="_d", by=by,
                         direction="backward", allow_exact_matches=False)


def _league_prior(long: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward league mean split rates and league home-win rate, by date."""
    lg = (long.groupby(["split", "_d"], as_index=False)[["rs", "ra", "g"]].sum()
              .sort_values("_d"))
    lg[["c_rs", "c_ra", "c_g"]] = lg.groupby("split")[["rs", "ra", "g"]].cumsum()
    wide = lg.pivot(index="_d", columns="split", values=["c_rs", "c_ra", "c_g"])
    wide.columns = [f"lg_{a}_{b}" for a, b in wide.columns]
    wide = wide.sort_index().ffill()

    f = finals.copy()
    f["_d"] = pd.to_datetime(f["date"], errors="coerce")
    hw = (f.assign(hw=(f["final_home_runs"] > f["final_away_runs"]).astype(float), n=1.0)
           .groupby("_d", as_index=False)[["hw", "n"]].sum()
           .sort_values("_d"))
    hw[["c_hw", "c_n"]] = hw[["hw", "n"]].cumsum()
    out = wide.join(hw.set_index("_d")[["c_hw", "c_n"]], how="outer").sort_index().ffill()
    out[["c_hw", "c_n"]] = out[["c_hw", "c_n"]].fillna(0.0)
    return out.reset_index()


def build_walkforward(games: pd.DataFrame,
                      exponent: float = PYTHAG_EXPONENT) -> pd.DataFrame:
    """Attach a no-lookahead Pythagorean probability to every game."""
    finals = games[games["is_final"] & games["final_home_runs"].notna()].copy()
    long = _team_long(finals)

    split_state = _asof_state(long, ["team", "split"], "s")
    pooled_state = _asof_state(long, ["team"], "p")
    league = _league_prior(long, finals)

    g = games.copy()
    g["_d"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.sort_values("_d").reset_index(drop=True)

    # ---- prior state for each side -----------------------------------------
    for side, split in (("home", "home"), ("away", "away")):
        sp = split_state[split_state["split"] == split].drop(columns="split")
        g = _join_prior(g, sp.rename(columns={"team": f"{side}_team_key"}),
                        by=[f"{side}_team_key"], left_on={side: f"{side}_team_key"})
        g = g.rename(columns={f"{side}_team_key": side,
                              "s_rs": f"{side}_s_rs", "s_ra": f"{side}_s_ra",
                              "s_g": f"{side}_s_g"})
        g = _join_prior(g, pooled_state.rename(columns={"team": f"{side}_team_key"}),
                        by=[f"{side}_team_key"], left_on={side: f"{side}_team_key"})
        g = g.rename(columns={f"{side}_team_key": side,
                              "p_rs": f"{side}_p_rs", "p_ra": f"{side}_p_ra",
                              "p_g": f"{side}_p_g"})

    g = pd.merge_asof(g.sort_values("_d"), league.sort_values("_d"), on="_d",
                      direction="backward", allow_exact_matches=False)
    # A game on the season's opening date has no prior state at all.
    for c in [c for c in g.columns if c.endswith(("_s_rs", "_s_ra", "_s_g",
                                                  "_p_rs", "_p_ra", "_p_g"))]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    for c in ("lg_c_rs_home", "lg_c_ra_home", "lg_c_rs_away", "lg_c_ra_away",
              "lg_c_g_home", "lg_c_g_away", "c_hw", "c_n"):
        if c not in g.columns:
            g[c] = 0.0
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    # ---- league walk-forward means -----------------------------------------
    def _rate(num, den, fallback):
        den = np.asarray(den, float)
        return np.where(den > 0, np.asarray(num, float) / np.maximum(den, 1e-9), fallback)

    lg_home_rs = _rate(g["lg_c_rs_home"], g["lg_c_g_home"], 4.5)
    lg_home_ra = _rate(g["lg_c_ra_home"], g["lg_c_g_home"], 4.5)
    lg_away_rs = _rate(g["lg_c_rs_away"], g["lg_c_g_away"], 4.5)
    lg_away_ra = _rate(g["lg_c_ra_away"], g["lg_c_g_away"], 4.5)
    lg_all_rs = _rate(g["lg_c_rs_home"] + g["lg_c_rs_away"],
                      g["lg_c_g_home"] + g["lg_c_g_away"], 4.5)
    lg_all_ra = _rate(g["lg_c_ra_home"] + g["lg_c_ra_away"],
                      g["lg_c_g_home"] + g["lg_c_g_away"], 4.5)

    # league home-field, shrunk toward the prior
    hfa_p = ((g["c_hw"] + HFA_PRIOR_P * HFA_PRIOR_WEIGHT) /
             (g["c_n"] + HFA_PRIOR_WEIGHT)).to_numpy(float)
    g["league_hfa_p"] = hfa_p

    # ---- shrunk rates -------------------------------------------------------
    def shrunk(num, cnt, prior_rate, k):
        return (np.asarray(num, float) + k * np.asarray(prior_rate, float)) / \
               (np.asarray(cnt, float) + k)

    k_s, k_p = SHRINK_GAMES_SPLIT, SHRINK_GAMES_POOLED
    h_s_rs = shrunk(g["home_s_rs"], g["home_s_g"], lg_home_rs, k_s)
    h_s_ra = shrunk(g["home_s_ra"], g["home_s_g"], lg_home_ra, k_s)
    a_s_rs = shrunk(g["away_s_rs"], g["away_s_g"], lg_away_rs, k_s)
    a_s_ra = shrunk(g["away_s_ra"], g["away_s_g"], lg_away_ra, k_s)

    h_p_rs = shrunk(g["home_p_rs"], g["home_p_g"], lg_all_rs, k_p)
    h_p_ra = shrunk(g["home_p_ra"], g["home_p_g"], lg_all_ra, k_p)
    a_p_rs = shrunk(g["away_p_rs"], g["away_p_g"], lg_all_rs, k_p)
    a_p_ra = shrunk(g["away_p_ra"], g["away_p_g"], lg_all_ra, k_p)

    # ---- probabilities ------------------------------------------------------
    # POOLED: strengths sit on a natural 0.500 baseline, so the reference matchup
    # is exactly 0.500 and the home-field term is the league rate outright.
    h_pool = _pythag(h_p_rs, h_p_ra, exponent)
    a_pool = _pythag(a_p_rs, a_p_ra, exponent)
    raw_pool = _log5(h_pool, a_pool)
    p_pool = _expit(_logit(raw_pool) - _logit(0.5) + _logit(hfa_p))

    # SPLIT: the league's average split matchup is not 0.500 (it came out to
    # ~0.506 in 2026 YTD), so the signal is taken relative to that reference
    # before the league home-field term is applied. Without this the split
    # variant silently prices home clubs ~2 points light.
    h_split = _pythag(h_s_rs, h_s_ra, exponent)
    a_split = _pythag(a_s_rs, a_s_ra, exponent)
    ref_split = _log5(_pythag(lg_home_rs, lg_home_ra, exponent),
                      _pythag(lg_away_rs, lg_away_ra, exponent))
    raw_split = _log5(h_split, a_split)
    p_split = _expit(_logit(raw_split) - _logit(ref_split) + _logit(hfa_p))

    # ---- basis / fallback ---------------------------------------------------
    enough_split = ((g["home_s_g"] >= MIN_SPLIT_GAMES) & (g["away_s_g"] >= MIN_SPLIT_GAMES)).to_numpy()
    enough_pool = ((g["home_p_g"] >= MIN_POOLED_GAMES) & (g["away_p_g"] >= MIN_POOLED_GAMES)).to_numpy()

    basis = np.where(enough_pool, "pooled", "league")
    p_home = np.where(enough_pool, p_pool, hfa_p)
    p_split_out = np.where(enough_split, p_split, np.where(enough_pool, p_pool, hfa_p))
    basis_split = np.where(enough_split, "split", np.where(enough_pool, "pooled", "league"))

    out = pd.DataFrame({
        "gamePk": g["gamePk"].astype(str),
        "date": g["date"],
        "away": g["away"], "home": g["home"], "venue": g["venue"],
        "pythag_tag": PYTHAG_TAG,
        "pythag_basis": basis,
        "pythag_split_basis": basis_split,
        "home_prior_games_all": g["home_p_g"], "away_prior_games_all": g["away_p_g"],
        "home_prior_games_split": g["home_s_g"], "away_prior_games_split": g["away_s_g"],
        "home_rs_hat": h_p_rs, "home_ra_hat": h_p_ra,
        "away_rs_hat": a_p_rs, "away_ra_hat": a_p_ra,
        "home_strength": h_pool, "away_strength": a_pool,
        "league_hfa_p": hfa_p,
        "pythag_p_home": p_home,
        "pythag_p_away": 1.0 - p_home,
        "pythag_split_p_home": p_split_out,
        "final_home_runs": g["final_home_runs"],
        "final_away_runs": g["final_away_runs"],
        "is_doubleheader_pair": g["is_doubleheader_pair"],
    })

    out["pythag_lean"] = np.where(out["pythag_p_home"] >= 0.5, out["home"], out["away"])
    out["pythag_lean_p"] = np.maximum(out["pythag_p_home"], 1.0 - out["pythag_p_home"])

    fin = out["final_home_runs"].notna() & out["final_away_runs"].notna()
    out["home_win"] = np.where(
        fin, (out["final_home_runs"] > out["final_away_runs"]).astype(float), np.nan)
    picked_home = out["pythag_p_home"] >= 0.5
    out["pythag_grade_fg"] = np.where(
        ~fin, "pending",
        np.where(out["home_win"] == picked_home.astype(float), "W", "L"))

    # Stable, documented column order: the OUT_COLS schema first (those present),
    # then any extra columns (e.g. pythag_split_basis) appended, never dropped.
    ordered = [c for c in OUT_COLS if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    out = out[ordered]

    return out.sort_values(["date", "gamePk"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ledger merge
# ─────────────────────────────────────────────────────────────────────────────

MERGE_COLS = [
    "pythag_tag", "pythag_basis", "pythag_p_home", "pythag_split_p_home",
    "pythag_lean", "pythag_lean_p", "pythag_grade_fg", "league_hfa_p",
    "home_prior_games_all", "away_prior_games_all",
]


def merge_into_ledger(wf: pd.DataFrame, ledger_path: str,
                      dry_run: bool = False) -> pd.DataFrame:
    """Idempotent left-join onto the lean ledger.

    Joins on gamePk when the ledger has it, otherwise on (date, home, away) with
    a loud warning — that fallback cannot disambiguate doubleheaders. Existing
    non-null values are never overwritten; a row that cannot be matched keeps NaN
    and is retried on the next run, matching market_backfill's contract.
    """
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(ledger_path)
    led = pd.read_csv(ledger_path, dtype=str, keep_default_na=False, na_values=[""])

    key = None
    for cand in ("gamePk", "game_pk", "gamepk"):
        if cand in led.columns:
            key = cand
            break

    src = wf[["gamePk", "date", "home", "away"] + MERGE_COLS].copy()
    if key:
        led["_k"] = led[key].astype(str).str.strip()
        src["_k"] = src["gamePk"].astype(str).str.strip()
        join_on = ["_k"]
        print(f"[merge] joining on ledger column '{key}'")
    else:
        print("[merge] WARNING: no gamePk column in ledger; falling back to "
              "(date, home, away). Doubleheaders will not disambiguate.")
        hcol = next((c for c in led.columns if c.lower() in ("home", "home_team")), None)
        acol = next((c for c in led.columns if c.lower() in ("away", "away_team")), None)
        dcol = next((c for c in led.columns if c.lower() in ("date", "slate_date", "game_date")), None)
        if not all((hcol, acol, dcol)):
            raise RuntimeError("Ledger lacks both gamePk and (date, home, away) columns.")
        led["_k"] = (led[dcol].astype(str) + "|" + led[hcol].astype(str) + "|" + led[acol].astype(str))
        src["_k"] = (src["date"].astype(str) + "|" + src["home"].astype(str) + "|" + src["away"].astype(str))
        join_on = ["_k"]

    src = src.drop(columns=["gamePk", "date", "home", "away"]).drop_duplicates("_k")
    merged = led.merge(src, on=join_on, how="left", suffixes=("", "_new"))

    filled = 0
    for c in MERGE_COLS:
        new = f"{c}_new"
        if new in merged.columns:
            if c in led.columns:
                # The ledger is read as strings (dtype=str); on pandas>=3 that is
                # the StringDtype, which rejects the numeric values coming from
                # the walk-forward frame. Widen the target to object so a column
                # already written on a prior run accepts freshly-filled numbers
                # and round-trips through to_csv like the first-run float column.
                if merged[c].dtype != object:
                    merged[c] = merged[c].astype(object)
                take = merged[c].isna() & merged[new].notna()
                merged.loc[take, c] = merged.loc[take, new].to_numpy()
                filled += int(take.sum())
            else:
                merged[c] = merged[new]
                filled += int(merged[c].notna().sum())
            merged = merged.drop(columns=[new])
        elif c in merged.columns:
            filled += int(merged[c].notna().sum())

    merged = merged.drop(columns=["_k"])
    matched = int(merged["pythag_p_home"].notna().sum()) if "pythag_p_home" in merged else 0
    print(f"[merge] ledger rows {len(led)}  matched {matched}  cells filled {filled}")
    if matched < len(led):
        print(f"[merge] {len(led) - matched} row(s) unmatched — left NaN for retry")

    if dry_run:
        print("[merge] --dry-run: nothing written")
        return merged
    merged.to_csv(ledger_path, index=False)
    print(f"[merge] wrote {ledger_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(wf: pd.DataFrame, ledger: pd.DataFrame | None = None) -> str:
    R: list[str] = [
        f"# Pythagorean control — walk-forward evaluation ({PYTHAG_TAG})",
        "",
        "Control arm. Season-to-date runs scored/allowed, empirical-Bayes shrunk, "
        "Log5, plus a league home-field term. No probables, no lineups, no park "
        "factors, no opponent adjustment, no market input.",
        "",
    ]
    d = wf[wf["home_win"].notna()].copy()
    if d.empty:
        R.append("_No graded games yet._")
        return "\n".join(R)

    R.append("## Calibration\n")
    for label, col in (("pooled + HFA", "pythag_p_home"),
                       ("venue split + HFA", "pythag_split_p_home")):
        p, y = d[col].to_numpy(float), d["home_win"].to_numpy(float)
        base = float(y.mean())
        b, b0 = _brier(p, y), _brier(np.full(len(y), base), y)
        R.append(f"{label:<20} n={len(d)}  mean P(home) {p.mean():.4f}  "
                 f"realized {base:.4f}  Brier {b:.4f}  vs constant {b0:.4f}  "
                 f"skill {100*(1-b/b0):+.1f}%")
    R.append("")

    R.append("reliability (pooled):")
    d["_b"] = pd.cut(d["pythag_p_home"], [0, .40, .45, .50, .55, .60, 1])
    R.append(f"  {'P(home) bin':>14} | {'n':>5} | {'pred':>6} | {'real':>6} | {'95% CI':>14}")
    for b, s in d.groupby("_b", observed=True):
        pp, lo, hi = _wilson(int(s["home_win"].sum()), len(s))
        R.append(f"  {str(b):>14} | {len(s):>5d} | {s['pythag_p_home'].mean():>6.3f} | "
                 f"{pp:>6.3f} | [{lo:.2f},{hi:.2f}]")
    R.append("")

    wins = int((d["pythag_grade_fg"] == "W").sum())
    p, lo, hi = _wilson(wins, len(d))
    R.append(f"straight-up side record: {wins}-{len(d)-wins} ({p:.3f}) "
             f"[{lo:.3f}, {hi:.3f}]\n")

    if ledger is None:
        R.append("_No ledger supplied — market and lean comparisons skipped._")
        return "\n".join(R)

    mcol = next((c for c in ledger.columns if c.lower() in
                 ("close_p_home", "market_close_p_home", "p_home_close")), None)
    if mcol is None:
        R.append("_Ledger has no devigged close probability column; market "
                 "comparison skipped._")
        return "\n".join(R)

    L = ledger.copy()
    for c in (mcol, "pythag_p_home", "pythag_split_p_home"):
        if c in L.columns:
            L[c] = pd.to_numeric(L[c], errors="coerce")
    dcol = next((c for c in L.columns if c.lower() in ("date", "slate_date", "game_date")), None)
    key = next((c for c in L.columns if c.lower() in ("gamepk", "game_pk")), None)
    if key:
        L = L.merge(d[["gamePk", "home_win"]].assign(gamePk=lambda x: x["gamePk"].astype(str)),
                    left_on=L[key].astype(str), right_on="gamePk", how="inner")
    m = L[L[mcol].notna() & L["pythag_p_home"].notna() & L["home_win"].notna()].copy()

    R.append("## Versus the market close (paired, same games)\n")
    if len(m) < 30:
        R.append(f"_Only {len(m)} rows with both a close probability and a "
                 f"control probability — not reportable yet._")
        return "\n".join(R)

    y = m["home_win"].to_numpy(float)
    bp, bm = _brier(m["pythag_p_home"], y), _brier(m[mcol], y)
    diff = (m["pythag_p_home"].to_numpy(float) - y) ** 2 - (m[mcol].to_numpy(float) - y) ** 2
    clusters = m[dcol].astype(str).to_numpy() if dcol else np.arange(len(m))
    est, lo, hi = _cluster_boot_mean_ci(diff, clusters, seed=41)
    R.append(f"n={len(m)}   control Brier {bp:.4f}   market Brier {bm:.4f}")
    R.append(f"paired ΔBrier (control − market): {est:+.5f}  "
             f"date-cluster 95% CI [{lo:+.5f}, {hi:+.5f}]")
    R.append("Note: the market's entire edge over a constant base rate is ~0.003 "
             "Brier. A single season resolves ΔBrier only to about ±0.005, so this "
             "line is a sanity check, not a verdict.\n")

    R.append("## Head-to-head on disagreement (the higher-powered test)\n")
    dis = m[(m["pythag_p_home"] >= 0.5) != (m[mcol] >= 0.5)].copy()
    if len(dis) >= 30:
        w = int((dis["home_win"] == (dis["pythag_p_home"] >= 0.5).astype(float)).sum())
        p, lo, hi = _wilson(w, len(dis))
        R.append(f"control vs market, different sides: n={len(dis)}  "
                 f"control {w}-{len(dis)-w} ({p:.3f}) [{lo:.3f}, {hi:.3f}]")
    else:
        R.append(f"_Only {len(dis)} disagreements so far._")

    lean_col = next((c for c in L.columns if "lean" in c.lower() and "pythag" not in c.lower()), None)
    if lean_col:
        R.append(f"\n_Ledger lean column detected: `{lean_col}` — wire the same "
                 "disagreement split against it for the xwOBA-vs-control test._")
    return "\n".join(R)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward Pythagorean control arm.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: Mar 1 of season)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today, America/New_York)")
    ap.add_argument("--game-types", default="R")
    ap.add_argument("--out", default=None, help="backfill CSV path")
    ap.add_argument("--slate-date", default=None, help="also emit just this date's games")
    ap.add_argument("--out-slate", default=None)
    ap.add_argument("--ledger", default=None, help="path to data/mlb_lean_ledger.csv")
    ap.add_argument("--merge-ledger", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exponent", type=float, default=PYTHAG_EXPONENT)
    args = ap.parse_args(argv)

    start = args.start or f"{args.season}-03-01"
    if args.end:
        end = args.end
    else:
        try:
            from zoneinfo import ZoneInfo
            end = dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        except Exception:
            end = dt.date.today().isoformat()

    out_csv = args.out or f"data/pythag_walkforward_{args.season}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print(f"[config] tag={PYTHAG_TAG} season={args.season} {start} -> {end} "
          f"exponent={args.exponent:g} k_split={SHRINK_GAMES_SPLIT:g} "
          f"k_pooled={SHRINK_GAMES_POOLED:g} hfa_prior={HFA_PRIOR_P:g}/{HFA_PRIOR_WEIGHT:g}")

    games = fetch_schedule(args.season, start, end, args.game_types)
    print(f"[schedule] {len(games)} games, {int(games['is_final'].sum())} final")

    wf = build_walkforward(games, exponent=args.exponent)
    wf.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}  ({len(wf)} rows)")

    if args.slate_date:
        s = wf[wf["date"] == args.slate_date].copy()
        path = args.out_slate or f"data/pythag_slate_{args.slate_date}.csv"
        s.to_csv(path, index=False)
        print(f"[saved] {path}  ({len(s)} games on {args.slate_date})")

    ledger_df = None
    if args.ledger and args.merge_ledger:
        ledger_df = merge_into_ledger(wf, args.ledger, dry_run=args.dry_run)
    elif args.ledger and os.path.exists(args.ledger):
        ledger_df = pd.read_csv(args.ledger)

    if args.report:
        rep = evaluate(wf, ledger_df)
        print("\n" + rep)
        path = args.out_report or f"data/pythag_report_{args.season}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(rep)
        print(f"\n[saved] {path}")

    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        raise SystemExit(rc)
