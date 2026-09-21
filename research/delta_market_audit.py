#!/usr/bin/env python3
"""Descriptive delta/market separation audit; never emits selection actions.

Run from any directory: python research/delta_market_audit.py
Uses the stored v13 representation, including hindsight reconstruction, and
closing prices. Chronological fitting does NOT make those inputs prospective.
All alternatives and regularization settings are exploratory, not registrations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import price_calibration_shadow as pcs

MIN_TRAIN = 100
MIN_SLATES = 7
RIDGE = 1.0
N_BOOT = 2000
SEEDS = (17, 41, 73)


def sigmoid(eta):
    return np.exp(-np.logaddexp(0, -eta))


def fit_offset(x, y, offset):
    """Ridge Bernoulli fit with coefficients shrunk to the market offset.

Fixed feature units and penalty; no scale or hyperparameter chosen on test rows.
"""
    beta = np.zeros(x.shape[1])

    def objective(b):
        eta = offset + x @ b
        return np.sum(np.logaddexp(0, eta) - y * eta) + RIDGE * (b @ b) / 2

    for _ in range(100):
        p = sigmoid(offset + x @ beta)
        grad = x.T @ (p - y) + RIDGE * beta
        if np.max(np.abs(grad)) < 1e-8:
            return beta
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + RIDGE * np.eye(len(beta))
        step = np.linalg.solve(hessian, grad)
        scale, before = 1., objective(beta)
        while scale > 2 ** -30:
            candidate = beta - scale * step
            if objective(candidate) <= before - 1e-4 * scale * (grad @ step):
                break
            scale /= 2
        else:
            raise RuntimeError("Offset fit line search failed")
        beta = candidate
        if np.max(np.abs(scale * step)) < 1e-10:
            return beta
    raise RuntimeError("Offset fit did not converge")


def replay(rows):
    """Every slate reads only earlier dates; reconstructed/native counts continue.

First 100 games / seven slates are a common warmup. They still train every
candidate. Evaluating the grid on the same remaining rows prevents a comparison
of its cold start with a fitted alternative's mature state.
"""
    g = pcs.evaluate_rows(rows).copy()
    if g.empty:
        raise ValueError("No usable v13-represented rows")
    if g.game_pk.isna().any() or g.game_pk.duplicated().any():
        raise ValueError("Audit requires one identified row per game")
    if pd.to_datetime(g.game_date, errors="coerce").isna().any():
        raise ValueError("Invalid slate date")
    q = g.market_q.to_numpy(float)
    y = g.won.to_numpy(float)
    z = np.log(q / (1 - q))
    # Fixed centering and units; no full-history standardization.
    d = (g.abs_delta.to_numpy(float) - .020) / .020
    designs = {
        "pooled_offset": np.ones((len(g), 1)),
        "market_recalibrated": np.column_stack([np.ones(len(g)), z]),
        "market_plus_delta": np.column_stack([np.ones(len(g)), z, d]),
        "market_delta_interaction": np.column_stack([np.ones(len(g)), z, d, z*d]),
    }
    g["p_market"] = q
    g["p_existing_grid"] = g.adjusted_probability_pregame
    # Algebraic diagnostic: preserve current q, shrink past cell excess to zero.
    # Clipping makes this a bounded heuristic, not a beta-binomial posterior.
    n = g.cell_n_pregame.to_numpy(float)
    qb = g.cell_mean_market_p_pregame.fillna(g.market_q).to_numpy(float)
    residual = (g.cell_wins_pregame.to_numpy(float) - n * qb) / (n + pcs.M0)
    g["p_residual_grid"] = np.clip(q + residual, 1e-6, 1 - 1e-6)
    for name in designs:
        g["p_" + name] = np.nan
    g["train_n"] = 0
    g["train_slates"] = 0
    for date, idx in g.groupby("game_date", sort=True).groups.items():
        train = g.game_date.lt(date).to_numpy()
        n_train = int(train.sum())
        n_slates = g.loc[train, "game_date"].nunique()
        g.loc[idx, "train_n"] = n_train
        g.loc[idx, "train_slates"] = n_slates
        if n_train < MIN_TRAIN or n_slates < MIN_SLATES:
            continue
        for name, design in designs.items():
            beta = fit_offset(design[train], y[train], z[train])
            g.loc[idx, "p_" + name] = sigmoid(z[idx] + design[idx] @ beta)
    return g


def loss(y, p, kind):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (p-y)**2 if kind == "brier" else -(y*np.log(p)+(1-y)*np.log1p(-p))


def paired_interval(g, candidate, reference, kind, seed):
    """Whole-slate percentile bootstrap conditional on already fitted forecasts.

This does not refit models, model serial dependence across slates, or adjust
for the search across alternatives. Seeds check Monte Carlo stability only.
"""
    y = g.won.to_numpy(float)
    diff = loss(y, g[candidate].to_numpy(float), kind) - loss(y, g[reference].to_numpy(float), kind)
    blocks = pd.DataFrame({"date": g.game_date.to_numpy(), "diff": diff}).groupby("date")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    means = blocks["sum"].to_numpy()[draws].sum(axis=1) / blocks["count"].to_numpy()[draws].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def summarize(g):
    scored = g[g.p_market_plus_delta.notna()]
    if scored.empty:
        raise ValueError("Insufficient earlier games/slates for comparison")
    cells = g.groupby("cell_id").size()
    out = {
        "interpretation": "Exploratory chronological retrospective; hindsight reconstruction and closing prices; no prospective evidence or selection changes.",
        "settings": {"min_train": MIN_TRAIN, "min_slates": MIN_SLATES, "ridge": RIDGE, "delta_center_and_unit": .020, "bootstrap_draws_per_seed": N_BOOT, "seeds": SEEDS},
        "rows": len(g), "basis": g.row_basis.value_counts().to_dict(),
        "date_range": [str(g.game_date.min()), str(g.game_date.max())],
        "delta_market_correlation": float(g.abs_delta.corr(g.market_q)),
        "cells": {"possible": 20, "populated": len(cells), "under_10_including_empty": int((cells < 10).sum())+20-len(cells)},
        "all_row_scores": {}, "common_replay_scores": {}, "paired_differences": {},
        "evaluation_n": len(scored), "evaluation_slates": scored.game_date.nunique(),
        "evaluation_dates": [str(scored.game_date.min()), str(scored.game_date.max())],
        "basis_scores": {},
    }
    for name in ("p_market", "p_existing_grid", "p_residual_grid"):
        out["all_row_scores"][name] = {k: float(loss(g.won.to_numpy(float), g[name].to_numpy(float), k).mean()) for k in ("brier", "logloss")}
    for name in [c for c in g if c.startswith("p_")]:
        out["common_replay_scores"][name] = {k: float(loss(scored.won.to_numpy(float), scored[name].to_numpy(float), k).mean()) for k in ("brier", "logloss")}
    contrasts = (
        ("p_existing_grid", "p_market"),
        ("p_residual_grid", "p_existing_grid"),
        ("p_residual_grid", "p_market"),
        ("p_pooled_offset", "p_market"),
        ("p_market_plus_delta", "p_market_recalibrated"),
        ("p_market_delta_interaction", "p_market_plus_delta"),
    )
    for a, b in contrasts:
        item = {}
        for kind in ("brier", "logloss"):
            item[kind] = {"difference": out["common_replay_scores"][a][kind]-out["common_replay_scores"][b][kind], "conditional_95pct_intervals_by_seed": {str(seed): paired_interval(scored, a, b, kind, seed) for seed in SEEDS}}
        out["paired_differences"][a+" minus "+b] = item
    for basis, frame in scored.groupby("row_basis"):
        out["basis_scores"][basis] = {"n": len(frame), "slates": frame.game_date.nunique(), "brier": {c: float(loss(frame.won.to_numpy(float), frame[c].to_numpy(float), "brier").mean()) for c in out["common_replay_scores"]}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=ROOT / pcs.LEDGER)
    args = ap.parse_args()
    ledger = pd.read_csv(args.ledger, low_memory=False)
    print(json.dumps(summarize(replay(pcs.prepare_rows(ledger))), indent=2))


if __name__ == "__main__":
    main()
