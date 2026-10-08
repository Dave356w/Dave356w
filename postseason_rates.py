"""Fold postseason plate appearances into the season rate boards.

Savant's custom leaderboard serves regular-season PAs only (checked
2026-10-08 against StatsAPI), so without this the season rates would stop
moving at the last regular-season game. On a slate with postseason games the
build fetches pitch-level Statcast for this season's postseason types and
folds every PA from a game dated strictly BEFORE the slate into `pa` and the
model rate columns, by PA weight. Nothing dated on or after the slate is
read, so the board stays pregame.

Per PA: the xwOBA numerator is `estimated_woba_using_speedangle` on a batted
ball and `woba_value` otherwise; the wOBA numerator is `woba_value`; both are
over `woba_denom`. Players absent from the season board are not added, and
the display-only columns (K%, BB%, xBA, ...) stay regular season.

Pure functions only: the fetch and its caching live in build_site, so the
research replay (research/postseason_inputs_shadow.py) shares this code.
"""

import numpy as np
import pandas as pd

POSTSEASON_GAME_TYPES = ("F", "D", "L", "W")
# Savant's statcast_search caps one CSV at this many pitches. A response at
# the cap may be truncated, and a silently partial board is worse than the
# regular-season one, so it is refused.
STATCAST_ROW_CAP = 25000

# Board rate column -> its per-PA numerator below.
NUMERATORS = {"xwoba": "xw_num", "woba": "w_num"}

PA_COLS = ["game_date", "game_pk", "batter", "pitcher", "xw_num", "w_num", "den"]


def pa_rows(raw):
    """Raw Statcast pitches -> one row per completed postseason PA."""
    if raw is None or raw.empty or "events" not in raw.columns:
        return pd.DataFrame(columns=PA_COLS)
    if len(raw) >= STATCAST_ROW_CAP:
        raise ValueError(f"statcast response has {len(raw)} rows, at the "
                         f"{STATCAST_ROW_CAP}-row cap; may be truncated")
    d = raw[raw["events"].notna()
            & raw["game_type"].astype(str).isin(POSTSEASON_GAME_TYPES)].copy()
    wv = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0.0)
    est = pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce")
    d["den"] = pd.to_numeric(d["woba_denom"], errors="coerce").fillna(0.0)
    d["w_num"] = wv
    d["xw_num"] = np.where(est.notna(), est, wv)
    d["game_date"] = d["game_date"].astype(str).str[:10]
    return d[PA_COLS].reset_index(drop=True)


def augment_board(cust, pas, player_type, before, rate_cols):
    """(board, n players changed): `cust` with PAs dated < `before` folded in.

    `rate_cols` names the board's rate columns to fold; each must be a key of
    NUMERATORS. Returns a copy; `cust` is not modified.
    """
    out = cust.copy()
    if pas is None or pas.empty:
        return out, 0
    key = "batter" if player_type == "batter" else "pitcher"
    p = pas[pas["game_date"] < str(before)[:10]]
    if p.empty:
        return out, 0
    g = p.groupby(key).agg(n=("den", "size"), den=("den", "sum"),
                          xw_num=("xw_num", "sum"), w_num=("w_num", "sum"))
    ids = pd.to_numeric(out["player_id"], errors="coerce")
    hit = ids.isin(g.index)
    pa = pd.to_numeric(out["pa"], errors="coerce")
    n_add = ids.map(g["n"]).fillna(0.0)
    den = ids.map(g["den"]).replace(0.0, np.nan)
    for col in dict.fromkeys(rate_cols):
        num = NUMERATORS[col]
        if col not in out.columns:
            continue
        season_rate = pd.to_numeric(out[col], errors="coerce")
        post_rate = ids.map(g[num]) / den
        mixed = (season_rate * pa + post_rate * n_add) / (pa + n_add)
        out[col] = np.where(hit & post_rate.notna() & season_rate.notna()
                            & pa.notna(), mixed, season_rate)
    out["pa"] = np.where(hit & pa.notna(), pa + n_add, pa)
    return out, int(hit.sum())
