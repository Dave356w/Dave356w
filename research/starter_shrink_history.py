#!/usr/bin/env python3
"""What K should shrink a starter's xwOBA-allowed? Several seasons, as-of-date.

THE LIVE READING THIS FOLLOWS UP. `pitcher_lineup_probe` put the starter's
coefficient at 0.68 +/- 0.32 on 5,685 pregame PAs, rising with the starter's
sample and implying K* ~347 against the shipped 100 -- the shape an
UNDER-shrunk rate produces, with an interval (0 to unbounded) that settles
nothing. At ~330 usable PAs a slate the live sample needs most of another
season. Statcast history has ~110k starter PAs a season now.

TWO QUESTIONS, answered on the same rows:

  1. SLOPE. Fit y - L = a + beta_b*b + beta_p*p + beta_bp*b*p per PA against
     the starter, with p shrunk at the shipped K. Under shrinkage
     beta_p = (n+K)/(n+K*), so beta_p < 1 reads as under-shrunk and the
     implied K* is (n+K)/beta_p - n at the median sample. The live probe's
     exact fit, so the two readings are comparable.

  2. ERROR. For each K on a FIXED grid, the model's own log5 prediction
     (L+b)(L+p_K)/L with NO fitted coefficient, scored by squared error
     against the PA's actual wOBA and, separately, its xwOBA. Nothing is
     fitted, so every season is out of sample for every K; the only search
     is the grid itself, printed whole in declared order. Paired against
     K = 100 on the same rows, clustered by starter.

THE EVIDENCE CLASS. Reconstructed history: every rate is rebuilt as of the day
before the game from that season's data only, which is what the build's
current-season leaderboards see -- but it is not the build's own output, and
it is not a forward record. It is the right class for choosing a shrinkage
constant, and a K that wins here goes to a registered SHADOW, never straight
into v13.

WHAT IT DOES NOT COPY FROM THE BUILD. The shrinkage target is the league
as-of-date rate, not v13's population/role targets, and the starter rate is
pure xwOBA, not v13's centred 50/50 xwOBA/wOBA blend. Both move the level a
K acts on; neither changes which direction the error curve slopes. Stated in
the report rather than assumed away.

Research only: nothing here reaches a lean, a delta, a grade or data/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import statcast_history as sh  # noqa: E402

K_GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1200)
MIN_LEAGUE_PA = 5000     # the as-of league rate needs a few days of season
BOOT = 200


def build_rows(pa: pd.DataFrame) -> pd.DataFrame:
    """PAs against the starter, with as-of-date raw rates and samples."""
    if pa.empty:
        return pa
    st = sh.starters(pa)
    rows = pa.merge(st, on=["game_pk", "fld_team"])
    rows = rows[rows["pitcher"] == rows["starter"]]
    rows = rows.merge(sh.asof(pa, "pitcher", "xwoba_pa", "p"),
                      on=["pitcher", "game_date"], how="left")
    rows = rows.merge(sh.asof(pa, "batter", "xwoba_pa", "b"),
                      on=["batter", "game_date"], how="left")
    rows = rows.merge(sh.league_asof(pa), on="game_date", how="left")
    rows = rows[(rows["lg_n"] >= MIN_LEAGUE_PA) & (rows["p_n"] > 0)]
    rows = rows.fillna({"b_num": 0.0, "b_n": 0.0})
    L = rows["L"].to_numpy(float)
    rows = rows.assign(
        b=sh.shrink(rows["b_num"], rows["b_n"], L, sh.SHIPPED_K) - L,
        y_w=rows["woba_value"] - L,
        y_x=rows["xwoba_pa"] - L)
    return rows.reset_index(drop=True)


def p_at(rows, k):
    L = rows["L"].to_numpy(float)
    return sh.shrink(rows["p_num"], rows["p_n"], L, k) - L


def slope_fit(rows, y="y_w", k=sh.SHIPPED_K, n_boot=BOOT):
    b = rows["b"].to_numpy(float)
    p = p_at(rows, k)
    X = np.column_stack([np.ones(len(rows)), b, p, b * p])
    yy = rows[y].to_numpy(float)

    def bp(r):
        c = sh.ols(X[r], yy[r])
        return np.nan if c is None else c[2]

    c = sh.ols(X, yy)
    if c is None:
        return None
    se, how = sh.cluster_se_twoway(bp, rows["pitcher"].to_numpy(),
                                   rows["batter"].to_numpy(), n_boot=n_boot)
    n_med = float(np.median(rows["p_n"]))
    beta = float(c[2])
    k_star = (n_med + k) / beta - n_med if beta > 0.1 else float("nan")
    lo = beta + 2 * se
    hi = beta - 2 * se
    k_lo = max((n_med + k) / lo - n_med, 0.0) if lo > 0.1 else float("nan")
    k_hi = (n_med + k) / hi - n_med if hi > 0.1 else float("inf")
    return {"beta_b": float(c[1]), "beta_p": beta, "se": se, "how": how,
            "n_med": n_med, "k_star": k_star, "k_lo": k_lo, "k_hi": k_hi}


def error_grid(rows, y="y_w", grid=K_GRID, n_boot=BOOT):
    """Paired squared-error change vs the shipped K, per K on the grid."""
    L = rows["L"].to_numpy(float)
    B = L + rows["b"].to_numpy(float)
    yy = rows[y].to_numpy(float) + L

    def sq(k):
        pred = B * (L + p_at(rows, k)) / L
        return (yy - pred) ** 2

    base = sq(sh.SHIPPED_K)
    starters = rows["pitcher"].to_numpy()
    out = []
    for k in grid:
        d = sq(k) - base
        se = 0.0 if k == sh.SHIPPED_K else sh.cluster_se(
            lambda r, d=d: float(d[r].mean()), starters, n_boot=n_boot)
        out.append({"k": k, "mse": float((base + d).mean()),
                    "d": float(d.mean()), "se": se})
    return out


def season_block(label, rows, n_boot=BOOT):
    say = []
    if rows.empty:
        return [f"{label}: no rows"]
    say.append(f"{label}: {len(rows):,} PAs vs {rows['pitcher'].nunique():,} "
               f"starters, {rows['game_pk'].nunique():,} games; starter BF "
               f"p10/median/p90 {np.percentile(rows['p_n'], 10):.0f}/"
               f"{np.median(rows['p_n']):.0f}/{np.percentile(rows['p_n'], 90):.0f}")
    for y, name in (("y_w", "actual wOBA"), ("y_x", "xwOBA")):
        s = slope_fit(rows, y, n_boot=n_boot)
        if s is None:
            say.append(f"  [{name}] fit failed")
            continue
        hi = "unbounded" if not np.isfinite(s["k_hi"]) else f"{s['k_hi']:.0f}"
        say.append(f"  [{name}] beta_p at K=100 {s['beta_p']:+.3f} +/- "
                   f"{s['se']:.3f} ({s['how']}); beta_b {s['beta_b']:+.3f}; "
                   f"implied K* ~{s['k_star']:.0f}, +/-2se {s['k_lo']:.0f} to {hi}")
        grid = error_grid(rows, y, n_boot=n_boot)
        best = min(grid, key=lambda g: g["mse"])
        cells = "  ".join(
            f"{g['k']}:{g['d'] * 1e6:+.1f}" + ("" if g["k"] == sh.SHIPPED_K
                                               else f"±{g['se'] * 1e6:.1f}")
            for g in grid)
        say.append(f"  [{name}] MSE change vs K=100 (x1e-6, ±se by starter): {cells}")
        z = best["d"] / best["se"] if best["se"] > 0 else float("nan")
        say.append(f"  [{name}] lowest error at K={best['k']} "
                   f"({best['d'] * 1e6:+.1f}e-6, z {z:+.2f})")
    return say


def report(data_dir=sh.DEFAULT_DIR, seasons=None, n_boot=BOOT):
    seasons = seasons or sh.seasons_available(data_dir)
    out = ["STARTER SHRINKAGE -- several seasons, every rate as of the day before",
           "  reconstructed history: the right class for choosing K, not a "
           "forward record", ""]
    allrows = []
    for s in seasons:
        rows = build_rows(sh.load(s, data_dir)["pa"])
        if not rows.empty:
            rows = rows.assign(season=s)
            allrows.append(rows)
        out += season_block(str(s), rows, n_boot)
        out.append("")
    if len(allrows) > 1:
        out += season_block("POOLED", pd.concat(allrows, ignore_index=True), n_boot)
        out.append("")
    out += [
        "  How to read it. The error grid is the decision: a K whose paired",
        "  change is below zero by more than 2se, in the same direction in",
        "  most seasons and on both outcomes, is the candidate for a SHADOW.",
        "  The slope and K* restate the live probe's reading on far more",
        "  rows; they should agree with the grid's direction. The grid is",
        "  ten fixed values printed in order -- expect the smallest of nine",
        "  noisy comparisons to sit near -1.5se under a null, so read a lone",
        "  z of -2 in one season as noise.",
        "  Not copied from the build: league as-of target (not v13's",
        "  population/role targets) and a pure xwOBA starter rate (not the",
        "  50/50 blend). Both move the level K acts on, not the direction.",
    ]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=sh.DEFAULT_DIR)
    p.add_argument("--boot", type=int, default=BOOT)
    a = p.parse_args()
    print("\n".join(report(a.data, n_boot=a.boot)))


if __name__ == "__main__":
    main()
