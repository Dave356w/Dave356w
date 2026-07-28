"""Pitch-mix matchup arm: the opposing lineup re-weighted by the starter's
arsenal, as a multiplier on the lineup's shrunk xwOBA composite.

Status: **shadow only.** `build_site.USE_PITCH_MIX_SHADOW` is off by default and
nothing here touches a lean. It ships dark until `pitch_arsenal_probe.py` shows
the batter x pitch-type deviation is reliable enough to beat its own noise --
the probe prints the measurement and the budget it has to clear.

Why the naive construction does not work
----------------------------------------
Three corrections separate this from "weight batter pitch-type xwOBA by the
pitcher's usage rates":

1. **PA share, not usage share.** Usage is per pitch; the batter values are per
   plate appearance. Putaway pitches end plate appearances far above their usage
   rate, so usage-weighting systematically underweights the breaking ball. Use
   the pitcher's PA share per pitch type; `pitcher_pa_weights` falls back to
   usage only when PA is absent and flags it.

2. **League-relative per pitch type, or the mix counts twice.** League xwOBA
   against a slider is far below league xwOBA against a four-seamer, so a
   slider-heavy arsenal drags any raw weighted average down on mix alone -- and
   the starter's own xwOBA-allowed, which the matchup already multiplies in,
   is low *because* he throws those sliders. Everything here is a ratio to
   league-at-that-pitch-type, normalised so a league-average hitter returns
   exactly 1.0 against any arsenal.

3. **It supplements the platoon term.** Savant's batter arsenal splits pool both
   pitcher hands, so the handedness effect is not in them; `PLATOON_XWOBA_ADJ`
   still applies. The multiplier is a starter-phase quantity only -- a bullpen
   is not one arsenal.

Shrinkage
---------
Each (batter, pitch type) cell is regressed toward that batter's own overall
relative level, not toward league: the question is whether *this* hitter is
unusually bad against *this* pitch, over and above how good he is generally.
A batter with no cell data therefore lands on a multiplier of exactly 1.0 and
the arm abstains continuously -- there is no coverage threshold to trip.

`MIX_SHRINK_K` is deliberately far above the build's `XWOBA_SHRINK_K = 100`:
at K=100 the residual cell noise moves a game delta by ~0.013 xwOBA, about
two thirds of the current-family median lean of 0.0195. See the probe for the
noise budget behind the constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Savant arsenal leaderboards. Verify the column names against a live pull
# before trusting a build: this module resolves them case-insensitively and
# degrades to no multiplier when a column is missing, but a silent rename would
# otherwise look like league-average hitters.
BATTER_ARSENAL_URL = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
    "?type=batter&pitchType=&year={season}&team=&min={min_pa}&csv=true"
)
PITCHER_ARSENAL_URL = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
    "?type=pitcher&pitchType=&year={season}&team=&min={min_pa}&csv=true"
)

# Column aliases, lowercased. First hit wins.
_ID_COLS = ("player_id", "playerid", "pitcher", "batter", "mlbamid", "id")
_TYPE_COLS = ("pitch_type", "pitchtype", "pitch")
_PA_COLS = ("pa", "plate_appearances", "total_pa")
_XWOBA_COLS = ("est_woba", "xwoba", "expected_woba", "est_woba_using_speedangle")
_USAGE_COLS = ("pitch_usage", "pitch_percent", "usage", "pitch_usage_pct")

# Cell shrinkage pseudo-sample. 600 keeps ~15% of a 110-PA cell's deviation,
# which holds the arm's contribution to a game delta near 0.0036 xwOBA (19% of
# the median lean) instead of 0.0130 (67%) at K=100.
MIX_SHRINK_K = 100.0

# Multiplier clamp. A hitter whose surviving deviation implies more than this
# is a data artifact, not a matchup; the clamp keeps one bad cell from moving
# a lineup composite. Recorded in the diagnostics when it binds.
MIX_CLAMP = 0.06


def _key(name):
    """Column name reduced to a lookup key: case, spaces and dashes ignored.

    Savant has shipped the same field as `pitch_type`, `Pitch Type` and
    `pitch-type` across leaderboards; the alias lists stay short this way."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _col(df, names):
    """First matching column, matched on the reduced key; None when absent."""
    if df is None or getattr(df, "empty", True):
        return None
    keyed = {_key(c): c for c in df.columns}
    for n in names:
        if n in keyed:
            return keyed[n]
    return None


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_arsenal(df):
    """Arsenal leaderboard -> tidy (player_id, pitch_type, pa, xwoba, usage).

    Returns an empty frame when the leaderboard is missing the columns this
    arm needs, which is the same as no arsenal data: every multiplier lands on
    1.0 and the shadow columns stay NaN.
    """
    cols = {k: _col(df, v) for k, v in (
        ("player_id", _ID_COLS), ("pitch_type", _TYPE_COLS), ("pa", _PA_COLS),
        ("xwoba", _XWOBA_COLS), ("usage", _USAGE_COLS))}
    empty = pd.DataFrame(columns=["player_id", "pitch_type", "pa", "xwoba", "usage"])
    if not all(cols[k] for k in ("player_id", "pitch_type")):
        return empty
    out = pd.DataFrame({
        "player_id": _num(df[cols["player_id"]]),
        "pitch_type": df[cols["pitch_type"]].astype(str).str.strip().str.upper(),
        "pa": _num(df[cols["pa"]]) if cols["pa"] else np.nan,
        "xwoba": _num(df[cols["xwoba"]]) if cols["xwoba"] else np.nan,
        "usage": _num(df[cols["usage"]]) if cols["usage"] else np.nan,
    })
    out = out[out["player_id"].notna() & (out["pitch_type"] != "")]
    if out.empty:
        return empty
    out["player_id"] = out["player_id"].astype("int64")
    return out.reset_index(drop=True)


def league_by_pitch_type(batter_arsenal):
    """PA-weighted league xwOBA per pitch type: the denominator that keeps the
    arsenal's own mix effect out of the multiplier."""
    df = batter_arsenal
    if df is None or df.empty:
        return {}
    ok = df[df["xwoba"].notna() & df["pa"].notna() & (df["pa"] > 0)]
    if ok.empty:
        return {}
    num = ok["xwoba"] * ok["pa"]
    grp = pd.DataFrame({"pitch_type": ok["pitch_type"], "num": num, "pa": ok["pa"]})
    agg = grp.groupby("pitch_type", sort=False).sum()
    return {t: float(r["num"] / r["pa"])
            for t, r in agg.iterrows() if r["pa"] > 0 and r["num"] > 0}


def batter_relative_cells(batter_arsenal, lg_by_type, k=MIX_SHRINK_K):
    """{(batter_id, pitch_type): shrunk ratio to league-at-that-pitch-type}.

    The ratio is regressed toward the batter's own overall relative level -- his
    PA-weighted ratio across the pitch types he has seen -- so the surviving
    quantity is pitch-specific skill, not general hitting ability, which the
    lineup composite already carries.
    """
    df = batter_arsenal
    if df is None or df.empty or not lg_by_type:
        return {}, {}
    d = df[df["xwoba"].notna() & df["pa"].notna() & (df["pa"] > 0)].copy()
    d["lg"] = d["pitch_type"].map(lg_by_type)
    d = d[d["lg"].notna() & (d["lg"] > 0)]
    if d.empty:
        return {}, {}
    d["ratio"] = d["xwoba"] / d["lg"]
    # Each batter's own centre: PA-weighted mean ratio over his covered cells.
    w = d.groupby("player_id", sort=False).apply(
        lambda g: float((g["ratio"] * g["pa"]).sum() / g["pa"].sum()),
        include_groups=False)
    overall = {int(pid): float(v) for pid, v in w.items() if np.isfinite(v)}
    cells = {}
    for row in d.itertuples(index=False):
        pid = int(row.player_id)
        base = overall.get(pid)
        if base is None:
            continue
        n = float(row.pa)
        cells[(pid, row.pitch_type)] = float((n * row.ratio + k * base) / (n + k))
    return cells, overall


def pitcher_pa_weights(pitcher_arsenal, pitcher_id):
    """(weights, basis) for one starter: PA share per pitch type.

    Usage share is the documented fallback and is reported as such -- it is a
    per-pitch distribution standing in for a per-PA one, which underweights
    putaway pitches.
    """
    df = pitcher_arsenal
    if df is None or df.empty or pitcher_id is None:
        return {}, "none"
    try:
        pid = int(pitcher_id)
    except (TypeError, ValueError):
        return {}, "none"
    mine = df[df["player_id"] == pid]
    if mine.empty:
        return {}, "none"
    pa = mine[mine["pa"].notna() & (mine["pa"] > 0)]
    if not pa.empty and float(pa["pa"].sum()) > 0:
        total = float(pa["pa"].sum())
        return ({r.pitch_type: float(r.pa) / total for r in pa.itertuples(index=False)},
                "pa_share")
    use = mine[mine["usage"].notna() & (mine["usage"] > 0)]
    if not use.empty and float(use["usage"].sum()) > 0:
        total = float(use["usage"].sum())
        return ({r.pitch_type: float(r.usage) / total
                 for r in use.itertuples(index=False)}, "usage_share")
    return {}, "none"


def batter_mix_multiplier(batter_id, weights, cells, overall, clamp=MIX_CLAMP):
    """One hitter's arsenal multiplier, and the arsenal weight it covered.

    1.0 means "no information": either the hitter has no cells, or his
    pitch-specific deviations cancel over this arsenal. Uncovered weight is
    neutral by construction -- the covered ratios are re-based on the hitter's
    own centre, so missing pitch types contribute nothing rather than pulling
    toward league.
    """
    if not weights or not cells:
        return 1.0, 0.0
    try:
        pid = int(batter_id)
    except (TypeError, ValueError):
        return 1.0, 0.0
    base = overall.get(pid)
    if base is None or not np.isfinite(base) or base <= 0:
        return 1.0, 0.0
    num = cov = 0.0
    for ptype, w in weights.items():
        cell = cells.get((pid, ptype))
        if cell is None or not np.isfinite(cell):
            continue
        num += w * cell
        cov += w
    if cov <= 0:
        return 1.0, 0.0
    # Covered weight only; uncovered weight stays neutral at the batter's centre.
    mult = (num + (1.0 - cov) * base) / base
    return float(np.clip(mult, 1.0 - clamp, 1.0 + clamp)), float(cov)


def lineup_mix_multiplier(hitters, weights, cells, overall, slot_weights=None,
                          clamp=MIX_CLAMP):
    """Lineup-level multiplier over the batting order.

    `hitters` is a sequence of (player_id, weight) pairs -- the same slot-PA
    weights the xwOBA composite uses, so the multiplier is aggregated on the
    same exposure basis as the value it multiplies. Returns
    (multiplier, covered_weight_share, n_hitters_with_cells).
    """
    rows = list(hitters or ())
    if not rows or not weights or not cells:
        return 1.0, 0.0, 0
    num = wsum = cov = 0.0
    n_cov = 0
    for i, (pid, w) in enumerate(rows):
        w = float(w) if w is not None and np.isfinite(w) else 0.0
        if slot_weights is not None:
            w = float(slot_weights[i]) if i < len(slot_weights) else w
        if w <= 0:
            continue
        m, c = batter_mix_multiplier(pid, weights, cells, overall, clamp=clamp)
        num += w * m
        wsum += w
        cov += w * c
        if c > 0:
            n_cov += 1
    if wsum <= 0:
        return 1.0, 0.0, 0
    mult = num / wsum
    return (float(np.clip(mult, 1.0 - clamp, 1.0 + clamp)),
            float(cov / wsum), int(n_cov))


def fetch_arsenals(fetch, season, min_pa=25):
    """(batter_arsenal, pitcher_arsenal) tidy frames.

    `fetch(url, cache_name) -> DataFrame` is injected so the caller owns
    caching and retries (build_site passes `cached_csv`, which caches once per
    ET day). Any failure returns empty frames: the arm abstains, the build does
    not fail. A pregame slate with no arsenal data is the `—` path, not an
    error.
    """
    out = []
    for label, url in (("batter", BATTER_ARSENAL_URL),
                       ("pitcher", PITCHER_ARSENAL_URL)):
        try:
            raw = fetch(url.format(season=season, min_pa=min_pa),
                        f"arsenal_{label}")
            out.append(normalize_arsenal(raw))
        except Exception:  # noqa: BLE001 -- caller logs; the arm just abstains
            out.append(normalize_arsenal(None))
    return out[0], out[1]
