"""v14 starter velocity term: the one implementation the build, the ledger
migration and the card all read.

    dv = fastball velo of the starter's LAST start
         - fastball-weighted mean of every start before it (this season)

    starter rate += BETA_V * dv

Pregame by construction: only starts strictly before the game's date are
read. No dv (fewer than MIN_PRIOR_STARTS earlier starts, or a last start with
under MIN_FB fastballs, or a failed fetch) means no adjustment -- the rate is
v13's.

BETA_V is frozen from `research/velocity_trend.py` fitted on 2023-2025 only
(actual wOBA outcome, beyond the shrunk hitter and starter rates): -4.67 wOBA
points per mph. Positive dv (velocity up) lowers the rate the starter is
expected to allow.

Dependency-light on purpose (pandas/numpy only): build_site, the migration and
tests import it, and `research/velocity_trend.start_velocity` is pinned to it
by a test so the rule has one spelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA_V = -0.00467          # per mph, per-PA wOBA units; fitted on 2023-2025
FASTBALLS = ("FF", "SI")
MIN_PRIOR_STARTS = 3       # the last start plus at least two before it
MIN_FB = 10                # fastballs in the last start


def per_start(pitches: pd.DataFrame) -> pd.DataFrame:
    """One pitcher's Statcast pitch rows -> one row per regular-season START.

    A start is an appearance whose first pitch came in the first inning --
    the per-pitcher counterpart of "first pitcher his team used", so an
    opener counts, as he does everywhere else in the model.
    """
    need = {"game_date", "game_pk", "inning", "pitch_type", "release_speed"}
    if pitches is None or pitches.empty or not need <= set(pitches.columns):
        return pd.DataFrame(columns=["game_date", "game_pk", "velo", "n_fb"])
    d = pitches
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str) == "R"]
    d = d.assign(game_date=pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d"),
                 inning=pd.to_numeric(d["inning"], errors="coerce"),
                 release_speed=pd.to_numeric(d["release_speed"], errors="coerce"))
    first = d.groupby("game_pk")["inning"].min()
    starts = set(first[first == 1].index)
    fb = d[d["game_pk"].isin(starts) & d["pitch_type"].isin(FASTBALLS)
           & d["release_speed"].notna()]
    return (fb.groupby(["game_date", "game_pk"])["release_speed"]
              .agg(velo="mean", n_fb="size").reset_index()
              .sort_values(["game_date", "game_pk"]).reset_index(drop=True))


def pregame_trend(starts: pd.DataFrame, before_date: str) -> dict:
    """{dv, velo_last, velo_base, n_starts} from starts STRICTLY before
    `before_date`; dv is None when the rule's minimums are not met."""
    s = starts[starts["game_date"].astype(str) < str(before_date)[:10]] \
        if starts is not None and len(starts) else pd.DataFrame()
    out = {"dv": None, "velo_last": None, "velo_base": None,
           "n_starts": int(len(s))}
    if len(s) < MIN_PRIOR_STARTS:
        if len(s):
            out["velo_last"] = float(s["velo"].iloc[-1])
        return out
    last, base = s.iloc[-1], s.iloc[:-1]
    out["velo_last"] = float(last["velo"])
    w = base["n_fb"].to_numpy(float)
    out["velo_base"] = float(np.average(base["velo"].to_numpy(float), weights=w)) \
        if w.sum() > 0 else None
    if last["n_fb"] >= MIN_FB and out["velo_base"] is not None:
        out["dv"] = out["velo_last"] - out["velo_base"]
    return out


def adjust(rate, dv):
    """The starter rate with the velocity term applied; unchanged without dv."""
    if rate is None or dv is None:
        return rate
    try:
        r, v = float(rate), float(dv)
    except (TypeError, ValueError):
        return rate
    if not (np.isfinite(r) and np.isfinite(v)):
        return rate
    return r + BETA_V * v


def net_delta(dv_home, dv_away, q_home, q_away, b_home_side, b_away_side, league):
    """Change in `net = home_off - away_off` from the velocity term alone.

    The starter phase is log5, mx_sp = B*P/L, LINEAR in P, so moving P by
    BETA_V*dv moves that phase's edge by q*B*BETA_V*dv/L exactly. Home's
    offence faces the AWAY starter (`_away` columns) and vice versa, matching
    the ledger's pitching-side field convention. A missing dv contributes 0.
    """
    def one(dv, q, b):
        if dv is None or not np.isfinite(dv):
            return 0.0
        return float(q) * float(b) * BETA_V * float(dv) / float(league)
    return one(dv_away, q_away, b_away_side) - one(dv_home, q_home, b_home_side)
