#!/usr/bin/env python3
"""Does a starter's recent velocity change predict him beyond his shrunk rate?

WHY THIS AND NOT A ROLLING xwOBA WINDOW. Recent xwOBA needs hundreds of
batters to separate from noise, which is why recency-weighted rates have not
survived in this repo. Fastball velocity is measured on every pitch, settles
within a start, and moves before results do when a pitcher is tired or hurt.
It is asymmetric between the two teams, known before first pitch, and absent
from the season rate the model shrinks.

THE PREDICTOR, fixed in advance. For a pitcher's k-th start of the season:

    dv = velo(start k-1) - mean velo over starts 1..k-2 (fastball-weighted)

four-seam and sinker only, pregame by construction (it never reads start k),
and only when the baseline has at least two starts and the previous start at
least MIN_FB fastballs. One predictor, so the bar is |z| = 2.

TWO TESTS:

  1. PERSISTENCE. Does dv carry into start k's own velocity? Regress
     velo(k) - baseline(k) on dv. A slope near 0 means last start's change
     was transient -- and a transient change cannot predict results.

  2. RESULTS. Per PA against the starter in start k:
         y - L = a + beta_b*b + beta_p*p + beta_bp*b*p + beta_v*dv
     with b and p the shipped-K shrunk as-of rates (the same rows as the
     starter-shrinkage study). beta_v < 0 = a velocity gain lowers what he
     allows beyond his rate. Two-way clustered by starter and batter.

Reconstructed history: a screen for a shadow input, not a forward record.
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
from research import starter_shrink_history as ssh  # noqa: E402

BOOT = 200
MIN_FB = 10


def start_velocity(pa: pd.DataFrame, velo: pd.DataFrame) -> pd.DataFrame:
    """One row per start with its velocity and the pregame dv."""
    st = sh.starters(pa).merge(pa[["game_pk", "game_date"]].drop_duplicates(),
                               on="game_pk")
    v = st.merge(velo.rename(columns={"pitcher": "starter"}),
                 on=["game_pk", "game_date", "starter"], how="inner")
    v = v.sort_values(["starter", "game_date", "game_pk"]).reset_index(drop=True)
    g = v.groupby("starter", sort=False)
    wsum = (v["velo"] * v["n_fb"])
    # Cumulative fastball-weighted mean of every start BEFORE this one...
    cw = g["n_fb"].cumsum() - v["n_fb"]
    cv = wsum.groupby(v["starter"]).cumsum() - wsum
    v["base_k"] = cv / cw.replace(0, np.nan)          # baseline for start k
    v["n_prior"] = g.cumcount()
    # ...and the pieces for dv: the previous start, against the baseline
    # that excluded it.
    v["prev_velo"] = g["velo"].shift(1)
    v["prev_nfb"] = g["n_fb"].shift(1)
    v["base_prev"] = g["base_k"].shift(1)
    v["dv"] = v["prev_velo"] - v["base_prev"]
    ok = (v["n_prior"] >= 3) & (v["prev_nfb"] >= MIN_FB) & v["dv"].notna()
    return v[ok].rename(columns={"starter": "pitcher"})


def persistence(sv: pd.DataFrame, n_boot=BOOT):
    y = (sv["velo"] - sv["base_k"]).to_numpy(float)
    X = np.column_stack([np.ones(len(sv)), sv["dv"]])
    c = sh.ols(X, y)
    if c is None:
        return None
    se = sh.cluster_se(lambda r: (lambda cc: np.nan if cc is None else cc[1])(
        sh.ols(X[r], y[r])), sv["pitcher"].to_numpy(), n_boot=n_boot)
    return {"slope": float(c[1]), "se": se, "n": len(sv),
            "sd_dv": float(sv["dv"].std()),
            "share_drop1": float((sv["dv"] <= -1.0).mean())}


def results_fit(rows: pd.DataFrame, y="y_w", n_boot=BOOT):
    b = rows["b"].to_numpy(float)
    p = ssh.p_at(rows, sh.SHIPPED_K)
    dv = rows["dv"].to_numpy(float)
    X = np.column_stack([np.ones(len(rows)), b, p, b * p, dv])
    yy = rows[y].to_numpy(float)
    c = sh.ols(X, yy)
    if c is None:
        return None
    se, how = sh.cluster_se_twoway(
        lambda r: (lambda cc: np.nan if cc is None else cc[4])(sh.ols(X[r], yy[r])),
        rows["pitcher"].to_numpy(), rows["batter"].to_numpy(), n_boot=n_boot)
    return {"beta_v": float(c[4]), "se": se, "how": how, "n": len(rows),
            "starts": rows[["game_pk", "pitcher"]].drop_duplicates().shape[0]}


def season_rows(d):
    if d["pa"].empty or d["velo"].empty:
        return pd.DataFrame(), pd.DataFrame()
    sv = start_velocity(d["pa"], d["velo"])
    rows = ssh.build_rows(d["pa"]).merge(
        sv[["game_pk", "pitcher", "dv"]], on=["game_pk", "pitcher"])
    return sv, rows


def report(data_dir=sh.DEFAULT_DIR, seasons=None, n_boot=BOOT):
    seasons = seasons or sh.seasons_available(data_dir)
    out = ["STARTER VELOCITY TREND -- last start's fastball velo vs his season baseline",
           "  reconstructed history: screens an input, is not a forward record", ""]
    svs, rws = [], []
    for s in seasons:
        sv, rows = season_rows(sh.load(s, data_dir))
        if sv.empty:
            out.append(f"{s}: no velocity rows")
            continue
        svs.append(sv)
        rws.append(rows)
        out += _block(str(s), sv, rows, n_boot)
    if len(svs) > 1:
        out += _block("POOLED", pd.concat(svs, ignore_index=True),
                      pd.concat(rws, ignore_index=True), n_boot)
    out += [
        "  How to read it. Persistence first: a slope near 0 means the change",
        "  did not carry into the next start, and beta_v should then be ~0",
        "  too. A persistent change with beta_v clearly below 0 (per mph) on",
        "  both outcomes is the case for a SHADOW that nudges the starter's",
        "  rate by beta_v * dv. Scale: multiply beta_v by sd(dv) for the",
        "  typical per-PA shift and compare it with the starter-rate spread",
        "  (~0.023) the model already uses.",
    ]
    return out


def _block(label, sv, rows, n_boot):
    out = []
    p = persistence(sv, n_boot)
    if p:
        out.append(f"{label}: {p['n']:,} starts with a pregame dv; sd(dv) "
                   f"{p['sd_dv']:.2f} mph; {p['share_drop1']:.1%} down >= 1 mph")
        out.append(f"  persistence into the next start: slope {p['slope']:+.2f} "
                   f"+/- {p['se']:.2f}")
    for y, name in (("y_w", "actual wOBA"), ("y_x", "xwOBA")):
        r = results_fit(rows, y, n_boot) if not rows.empty else None
        if r:
            z = r["beta_v"] / r["se"] if r["se"] > 0 else float("nan")
            shift = abs(r["beta_v"]) * (p["sd_dv"] if p else float("nan"))
            out.append(f"  [{name}] beta_v {r['beta_v'] * 1000:+.2f} +/- "
                       f"{r['se'] * 1000:.2f} wOBA points per mph (z {z:+.2f}, "
                       f"{r['how']}); 1 sd of dv ~ {shift * 1000:.1f} points; "
                       f"{r['n']:,} PAs in {r['starts']:,} starts")
    out.append("")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=sh.DEFAULT_DIR)
    p.add_argument("--boot", type=int, default=BOOT)
    a = p.parse_args()
    print("\n".join(report(a.data, n_boot=a.boot)))


if __name__ == "__main__":
    main()
