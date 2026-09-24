#!/usr/bin/env python3
"""Shared Statcast history for the three off-season studies.

  starter_shrink_history.py   what K should shrink a starter's xwOBA-allowed?
  defense_history.py          does team defence carry signal xwOBA strips out?
  velocity_trend.py           does a starter's recent velocity change predict him?

All three need the same thing the live probes cannot give them: several
seasons of plate appearances with every rate rebuilt AS OF THE DAY BEFORE each
game. That is reconstruction, not prospective evidence -- it is the right tool
for choosing a constant or screening a feature, and the wrong one for claiming
a live record. Every report says so.

NO LOOKAHEAD, BY CONSTRUCTION. `asof` sums a key's rows over strictly EARLIER
dates within the season, so a doubleheader's second game never sees the first
and no game sees itself. Seasons never leak into each other: the build's own
rates are current-season leaderboards, and these mirror that.

FETCH. Savant is unreachable from the dev sandbox, so `fetch` runs in CI
(`.github/workflows/history-studies.yml`) through `pybaseball.statcast`, one
week at a time, and keeps only what the studies read -- a season of raw
pitches is ~700k rows of ~90 columns, the reduced tables are a few MB:

  pa_<season>.csv.gz     one row per plate appearance (regular season)
  velo_<season>.csv.gz   mean four-seam/sinker velocity per pitcher per game
  games_<season>.csv.gz  final score per game

A missing column is a hard stop naming the column, never a silently empty
study -- the same rule `build_site` applies to a Savant selection that comes
back absent.

Research only: nothing here reaches a lean, a delta, a grade or data/.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = os.environ.get("HISTORY_DIR", str(ROOT / "history_cache"))

# The shipped constant the studies are read against. Imported rather than
# copied would drag build_site (and its tag guards) into a research job, so it
# is restated here and a test pins it to build_site's value.
SHIPPED_K = 100.0

# Fastball types for the velocity series: four-seam and sinker. A pitcher's
# mix across the two moves with his arsenal, not his arm, but both are his
# hardest pitch and pooling them is the standard velocity-tracking choice.
FASTBALLS = ("FF", "SI")

NEEDED = ("game_date", "game_pk", "game_type", "pitch_type", "release_speed",
          "batter", "pitcher", "events", "home_team", "away_team", "bb_type",
          "inning", "inning_topbot", "at_bat_number",
          "estimated_woba_using_speedangle", "woba_value", "woba_denom",
          "post_home_score", "post_away_score")


# --------------------------------------------------------------------- reduce

def reduce_chunk(raw: pd.DataFrame):
    """Raw Statcast pitches -> (pa, velo, games) for the regular season."""
    missing = [c for c in NEEDED if c not in raw.columns]
    if missing:
        raise SystemExit(f"Statcast response is missing columns: {missing}")
    d = raw[raw["game_type"].astype(str) == "R"].copy()
    d["game_date"] = pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d")
    top = d["inning_topbot"].astype(str).str.lower().eq("top")
    d["fld_team"] = np.where(top, d["home_team"], d["away_team"])
    d["bat_team"] = np.where(top, d["away_team"], d["home_team"])

    ev = d["events"].astype(str)
    pa = d[d["events"].notna() & ev.ne("") & ev.ne("nan")].copy()
    pa["woba_denom"] = pd.to_numeric(pa["woba_denom"], errors="coerce").fillna(0)
    pa["woba_value"] = pd.to_numeric(pa["woba_value"], errors="coerce").fillna(0)
    est = pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce")
    pa["bip"] = pa["bb_type"].notna() & pa["bb_type"].astype(str).ne("nan")
    # xwOBA per PA: the batted-ball estimate on contact, the actual weight
    # otherwise (a strikeout is 0 and a walk its linear weight either way).
    # A ball in play Savant could not track has no estimate and falls back to
    # its actual value -- a small bias toward wOBA, counted in the report.
    pa["xwoba_pa"] = np.where(pa["bip"] & est.notna(), est, pa["woba_value"])
    pa["untracked_bip"] = pa["bip"] & est.isna()
    pa = pa[pa["woba_denom"] > 0]
    pa = pa[["game_date", "game_pk", "at_bat_number", "inning", "home_team",
             "away_team", "fld_team", "bat_team", "batter", "pitcher", "bip",
             "untracked_bip", "woba_value", "xwoba_pa"]]

    fb = d[d["pitch_type"].isin(FASTBALLS)].copy()
    fb["release_speed"] = pd.to_numeric(fb["release_speed"], errors="coerce")
    velo = (fb.dropna(subset=["release_speed"])
              .groupby(["game_date", "game_pk", "pitcher"])["release_speed"]
              .agg(velo="mean", n_fb="size").reset_index())

    sc = d.assign(h=pd.to_numeric(d["post_home_score"], errors="coerce"),
                  a=pd.to_numeric(d["post_away_score"], errors="coerce"))
    games = (sc.groupby(["game_pk", "game_date", "home_team", "away_team"])
               .agg(home_score=("h", "max"), away_score=("a", "max"))
               .reset_index())
    return pa, velo, games


def fetch(season: int, out_dir: str, start: str | None = None,
          end: str | None = None, days: int = 7):
    """Pull one season a week at a time and write the three reduced tables."""
    from pybaseball import statcast  # CI-only dependency; see the workflow
    start = date.fromisoformat(start or f"{season}-03-15")
    end = date.fromisoformat(end or f"{season}-10-05")
    parts = {"pa": [], "velo": [], "games": []}
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        raw = statcast(cur.isoformat(), stop.isoformat(), verbose=False)
        if raw is not None and len(raw):
            for k, v in zip(parts, reduce_chunk(raw)):
                parts[k].append(v)
        print(f"  {season}: {cur}..{stop}  "
              f"{0 if raw is None else len(raw):,} pitches", flush=True)
        cur = stop + timedelta(days=1)
    os.makedirs(out_dir, exist_ok=True)
    for k, v in parts.items():
        df = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        df.to_csv(Path(out_dir) / f"{k}_{season}.csv.gz", index=False)
        print(f"  wrote {k}_{season}.csv.gz: {len(df):,} rows")


def load(season: int, data_dir: str = DEFAULT_DIR):
    p = Path(data_dir)
    out = {}
    for k in ("pa", "velo", "games"):
        f = p / f"{k}_{season}.csv.gz"
        out[k] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


def seasons_available(data_dir: str = DEFAULT_DIR):
    return sorted(int(f.name[3:7]) for f in Path(data_dir).glob("pa_*.csv.gz"))


# ---------------------------------------------------------------- as-of rates

def asof(rows: pd.DataFrame, key, value: str, name: str) -> pd.DataFrame:
    """Per (key, game_date): sum and count of `value` over STRICTLY EARLIER dates.

    `key` may be a column name or a list of them. The result is keyed by the
    dates `rows` itself contains, so merging it back onto any frame of
    (key, game_date) pairs drawn from the same season is lookahead-free.
    """
    keys = [key] if isinstance(key, str) else list(key)
    daily = (rows.groupby(keys + ["game_date"])[value]
                 .agg(["sum", "size"]).reset_index()
                 .sort_values(keys + ["game_date"]))
    g = daily.groupby(keys, sort=False)
    daily[f"{name}_num"] = g["sum"].cumsum() - daily["sum"]
    daily[f"{name}_n"] = g["size"].cumsum() - daily["size"]
    return daily[keys + ["game_date", f"{name}_num", f"{name}_n"]]


def asof_to(rows: pd.DataFrame, key: str, value: str, name: str,
            targets: pd.DataFrame) -> pd.DataFrame:
    """`asof`, evaluated at arbitrary (key, game_date) targets.

    For when the rows being summed are a SUBSET of the dates being asked
    about -- a team's road-fielded PAs, asked on its home dates too. Uses the
    running total through the last date STRICTLY before each target
    (`allow_exact_matches=False`), so a target date never sees itself.
    Targets with no earlier row get 0 / 0.
    """
    daily = (rows.groupby([key, "game_date"])[value]
                 .agg(["sum", "size"]).reset_index())
    daily = daily.sort_values([key, "game_date"])
    g = daily.groupby(key, sort=False)
    daily[f"{name}_num"] = g["sum"].cumsum()
    daily[f"{name}_n"] = g["size"].cumsum()
    daily["_d"] = pd.to_datetime(daily["game_date"])
    t = targets[[key, "game_date"]].drop_duplicates().copy()
    t["_d"] = pd.to_datetime(t["game_date"])
    t = t.sort_values("_d")
    daily = daily.sort_values("_d")
    m = pd.merge_asof(t, daily[[key, "_d", f"{name}_num", f"{name}_n"]],
                      on="_d", by=key, allow_exact_matches=False)
    m[[f"{name}_num", f"{name}_n"]] = m[[f"{name}_num", f"{name}_n"]].fillna(0.0)
    return m.drop(columns="_d")


def league_asof(pa: pd.DataFrame, value: str = "xwoba_pa") -> pd.DataFrame:
    d = pa.assign(_all=1)
    a = asof(d, "_all", value, "lg")
    a["L"] = a["lg_num"] / a["lg_n"].replace(0, np.nan)
    return a[["game_date", "L", "lg_n"]]


def shrink(num, n, prior, k):
    """(n*x + k*prior)/(n + k), with x = num/n -- i.e. (num + k*prior)/(n + k)."""
    return (np.asarray(num, float) + k * np.asarray(prior, float)) / (
        np.asarray(n, float) + k)


def starters(pa: pd.DataFrame) -> pd.DataFrame:
    """(game_pk, fld_team) -> the pitcher who faced that game's first batter.

    First PA by `at_bat_number`, the Statcast counterpart of the box score's
    `gamesStarted`: an opener IS the starter here, as he is in the ledger.
    """
    first = pa.sort_values("at_bat_number").drop_duplicates(["game_pk", "fld_team"])
    return first[["game_pk", "fld_team", "pitcher"]].rename(
        columns={"pitcher": "starter"})


# ------------------------------------------------------------------ inference

def ols(X, y, w=None):
    if w is not None:
        sw = np.sqrt(w)
        X, y = X * sw[:, None], y * sw
    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < X.shape[1]:
        return None
    return np.linalg.solve(xtx, X.T @ y)


def _groups(labels):
    _, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_se(stat, labels, n_boot=200, seed=20260924):
    """Cluster bootstrap SE of `stat(rows)`. O(N) grouping -- the sibling
    probes' per-cluster `flatnonzero` is O(N x clusters), which is fine at
    one slate's PAs and hours at three seasons'."""
    idx = _groups(labels)
    if len(idx) < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        v = stat(np.concatenate([idx[i] for i in pick]))
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def cluster_se_twoway(stat, a, b, n_boot=200, seed=20260924):
    """Cameron-Gelbach-Miller V_a + V_b - V_row, as `hitter_level_probe` does;
    the wider one-way SE when the estimate goes non-positive, labelled."""
    va = cluster_se(stat, a, n_boot, seed)
    vb = cluster_se(stat, b, n_boot, seed)
    vr = cluster_se(stat, np.arange(len(np.asarray(a))), n_boot, seed)
    if not all(np.isfinite(v) for v in (va, vb, vr)):
        fin = [v for v in (va, vb) if np.isfinite(v)]
        return (max(fin), "one-way") if fin else (float("nan"), "none")
    v = va ** 2 + vb ** 2 - vr ** 2
    return (max(va, vb), "degenerate") if v <= 0 else (float(np.sqrt(v)), "two-way")


def main():
    p = argparse.ArgumentParser(description="Fetch and reduce Statcast seasons")
    p.add_argument("--seasons", required=True, help="comma-separated, e.g. 2023,2024")
    p.add_argument("--out", default=DEFAULT_DIR)
    p.add_argument("--end", default=None, help="last date for the final season")
    a = p.parse_args()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    for s in seasons:
        have = Path(a.out) / f"pa_{s}.csv.gz"
        if have.exists() and s != seasons[-1]:
            print(f"{s}: cached, skipping")
            continue
        fetch(s, a.out, end=a.end if s == seasons[-1] else None)


if __name__ == "__main__":
    sys.exit(main())
