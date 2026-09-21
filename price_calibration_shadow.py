#!/usr/bin/env python3
"""Retrospective v13 dynamic-price calibration shadow arm.

This module does NOT change the xwOBA model, any model-family tag, or any
historical ledger decision. It reads the same current-v13 historical view the
UI already publishes:

* native xw+starter_blend_v13 rows use their stored v13 lean/net;
* retained xw+plat_consol_v12 rows use the stored v13 reconstruction through
  market_backfill.publish_reconstruction.

Those rows form one chronological calibration stream. The earliest usable
v13-represented slate starts the cell counts at n=0; reconstructed history and
native v13 then accumulate continuously with no reset at the model-tag boundary.

The retrospective uses the lean's CLOSING market for every row so the full
reconstructed history has one uniform price basis:
  q  = devigged leaned-side close probability (close_p_home or 1-close_p_home)
  be = raw implied breakeven from the leaned side's closing moneyline

That is deliberately different from a future production decision, which should
use a saved executable pregame snapshot. This is a SHADOW / descriptive arm.
Reconstruction itself is mixed-basis hindsight, and closing prices are not
decision-time snapshots. The report says both explicitly.

Sequential discipline is slate-frozen: every game on one game_date sees the
same cell state from PRIOR dates only. The entire slate is appended only after
all of that slate's shadow decisions have been evaluated.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from market_backfill import breakeven_prob, publish_reconstruction

M0 = 10.0
HURDLE = 0.015
RULE_TAG = "v13_dynamic_price_shadow_m10_h015"
MODEL_TAG = "xw+starter_blend_v13"
SOURCE_TAGS = ("xw+plat_consol_v12", MODEL_TAG)

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")

DELTA_BANDS = (
    ("D1", "[0.000,0.010)", 0.000, 0.010),
    ("D2", "[0.010,0.020)", 0.010, 0.020),
    ("D3", "[0.020,0.030)", 0.020, 0.030),
    ("D4", "[0.030,0.050)", 0.030, 0.050),
    ("D5", "[0.050,+)", 0.050, math.inf),
)
MARKET_BANDS = (
    ("M1", "q<.450", 0.000, 0.450),
    ("M2", "q .450-.500", 0.450, 0.500),
    ("M3", "q .500-.550", 0.500, 0.550),
    ("M4", "q .550+", 0.550, 1.0000000001),
)


def delta_band(value):
    """Return (id, label) for a fixed |xwOBA delta| band, else (None, None)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(x) or x < 0:
        return None, None
    for ident, label, lo, hi in DELTA_BANDS:
        if lo <= x < hi:
            return ident, label
    return None, None


def market_band(value):
    """Return (id, label) for the fixed leaned-side market-probability band."""
    try:
        q = float(value)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(q) or not (0 < q < 1):
        return None, None
    for ident, label, lo, hi in MARKET_BANDS:
        if lo <= q < hi:
            return ident, label
    return None, None


def _profit_on_win(ml):
    ml = float(ml)
    return ml / 100.0 if ml > 0 else 100.0 / abs(ml)


def prepare_rows(ledger, model_tag=MODEL_TAG):
    """Build the one-row-per-game v13-represented retrospective stream.

    The input may be the whole ledger or grade_leans' already-scoped current
    record family. Only graded v12/v13 source rows are retained. Reconstruction
    is applied to a COPY and never written back to the ledger.
    """
    if ledger is None or not len(ledger):
        return pd.DataFrame()

    need = {
        "status", "model_tag", "game_date", "home", "away", "xw_lean", "xw_net",
        "xw_full", "full_home", "full_away", "close_p_home", "close_home_ml",
        "close_away_ml",
    }
    if not need.issubset(ledger.columns):
        return pd.DataFrame()

    raw = ledger[
        ledger["status"].eq("graded")
        & ledger["model_tag"].astype(str).isin(SOURCE_TAGS)
    ].copy()
    if raw.empty:
        return raw

    # Same derivation the public v13 historical surfaces use.
    pub = publish_reconstruction(raw, model_tag)
    if pub is None or pub.empty:
        return pd.DataFrame()

    lean = pub["xw_lean"].astype(str).str.strip()
    home = pub["home"].astype(str).str.strip()
    away = pub["away"].astype(str).str.strip()
    lean_home = lean.eq(home)
    sided = lean_home | lean.eq(away)

    delta = pd.to_numeric(pub["xw_net"], errors="coerce").abs()
    p_home = pd.to_numeric(pub["close_p_home"], errors="coerce")
    q = np.where(lean_home, p_home, 1.0 - p_home)

    hml = pd.to_numeric(pub["close_home_ml"], errors="coerce")
    aml = pd.to_numeric(pub["close_away_ml"], errors="coerce")
    ml = np.where(lean_home, hml, aml)
    be = breakeven_prob(ml)

    decided = pub["xw_full"].isin(["W", "L"])
    valid = (
        sided
        & decided
        & np.isfinite(delta)
        & np.isfinite(q)
        & (q > 0)
        & (q < 1)
        & np.isfinite(ml)
        & (np.asarray(ml, dtype=float) != 0)
        & np.isfinite(be)
    )
    if not bool(np.asarray(valid).any()):
        return pd.DataFrame()

    out = pub.loc[valid].copy()
    keep = np.asarray(valid)
    out["abs_delta"] = delta.loc[valid].to_numpy(float)
    out["market_q"] = np.asarray(q, dtype=float)[keep]
    out["offered_ml"] = np.asarray(ml, dtype=float)[keep]
    out["executable_breakeven"] = np.asarray(be, dtype=float)[keep]
    out["won"] = out["xw_full"].eq("W").to_numpy(bool)
    out["row_basis"] = np.where(
        out["model_tag"].astype(str).eq(model_tag),
        "native",
        "reconstructed",
    )

    db = [delta_band(v) for v in out["abs_delta"]]
    mb = [market_band(v) for v in out["market_q"]]
    out["delta_band"] = [x[0] for x in db]
    out["delta_band_label"] = [x[1] for x in db]
    out["market_band"] = [x[0] for x in mb]
    out["market_band_label"] = [x[1] for x in mb]
    out["cell_id"] = out["delta_band"].astype(str) + "_" + out["market_band"].astype(str)

    valid_band = out["delta_band"].notna() & out["market_band"].notna()
    out = out[valid_band].copy()
    if out.empty:
        return out

    out["profit_if_bet"] = np.where(
        out["won"].to_numpy(bool),
        [_profit_on_win(v) for v in out["offered_ml"].to_numpy(float)],
        -1.0,
    )

    # Stable sort. Same-date games are still evaluated from ONE frozen state,
    # so the ordering inside a date cannot change any pregame statistic.
    sort_cols = ["game_date"]
    if "scheduled_start_utc" in out.columns:
        sort_cols.append("scheduled_start_utc")
    if "game_pk" in out.columns:
        sort_cols.append("game_pk")
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def evaluate_rows(rows, m0=M0, hurdle=HURDLE):
    """Apply slate-frozen sequential M0 shrinkage to prepared rows."""
    if rows is None or not len(rows):
        return pd.DataFrame()

    need = {
        "game_date", "cell_id", "market_q", "executable_breakeven", "won",
    }
    if not need.issubset(rows.columns):
        raise ValueError(f"prepared rows missing {sorted(need - set(rows.columns))}")
    if m0 <= 0:
        raise ValueError("m0 must be positive")

    out = rows.copy().reset_index(drop=True)
    n_pre = np.zeros(len(out), dtype=int)
    w_pre = np.zeros(len(out), dtype=int)
    qbar_pre = np.full(len(out), np.nan, dtype=float)
    lam = np.zeros(len(out), dtype=float)
    p_hat = np.full(len(out), np.nan, dtype=float)
    edge = np.full(len(out), np.nan, dtype=float)
    qualify = np.zeros(len(out), dtype=bool)

    # cell -> [n, wins, sum_q]
    state = {}

    # game_date is the repo's slate identity. Every row on a date reads the
    # same prior state; no game from the slate trains another game on that slate.
    for _date, idx in out.groupby(out["game_date"].astype(str), sort=True).groups.items():
        idx = list(idx)

        for i in idx:
            cell = str(out.at[i, "cell_id"])
            n, w, qsum = state.get(cell, (0, 0, 0.0))
            q = float(out.at[i, "market_q"])
            be = float(out.at[i, "executable_breakeven"])

            n_pre[i] = int(n)
            w_pre[i] = int(w)
            if n:
                qb = qsum / n
                qbar_pre[i] = qb
                lam[i] = n / (n + m0)
                ph = (w + m0 * qb) / (n + m0)
            else:
                # qbar is undefined before the first observation. The market is
                # the null, so the current game's q is the zero-alpha estimate.
                lam[i] = 0.0
                ph = q

            p_hat[i] = ph
            edge[i] = ph - be
            qualify[i] = edge[i] >= hurdle

        # Only after the whole slate has been evaluated may its outcomes enter.
        for i in idx:
            cell = str(out.at[i, "cell_id"])
            n, w, qsum = state.get(cell, (0, 0, 0.0))
            state[cell] = (
                n + 1,
                w + int(bool(out.at[i, "won"])),
                qsum + float(out.at[i, "market_q"]),
            )

    out["cell_n_pregame"] = n_pre
    out["cell_wins_pregame"] = w_pre
    out["cell_mean_market_p_pregame"] = qbar_pre
    out["shrinkage_prior_weight_m0"] = float(m0)
    out["shrinkage_lambda_pregame"] = lam
    out["adjusted_probability_pregame"] = p_hat
    out["estimated_edge"] = edge
    out["estimated_edge_pp"] = 100.0 * edge
    out["predeclared_hurdle_margin"] = float(hurdle)
    out["decision_action"] = np.where(qualify, "QUALIFY", "ABSTAIN")
    out["rule_tag"] = RULE_TAG
    return out


def analysis_frame(ledger, model_tag=MODEL_TAG, m0=M0, hurdle=HURDLE):
    return evaluate_rows(prepare_rows(ledger, model_tag=model_tag), m0=m0, hurdle=hurdle)


def _summary_line(g, label, counterfactual=False):
    n = len(g)
    if not n:
        return f"  {label:<17} n=0"
    won = g["won"].to_numpy(bool)
    w = int(won.sum())
    units = float(pd.to_numeric(g["profit_if_bet"], errors="coerce").sum())
    roi = units / n
    prefix = "counterfactual " if counterfactual else ""
    return (
        f"  {label:<17} n={n:3d}  {w}-{n-w} ({w/n:.3f})  "
        f"{prefix}{units:+.2f}u  ROI {100*roi:+.1f}%  "
        f"mean p_hat {g['adjusted_probability_pregame'].mean():.3f}  "
        f"mean BE {g['executable_breakeven'].mean():.3f}  "
        f"mean edge {g['estimated_edge_pp'].mean():+.1f}pp"
    )


def _basis_line(g, basis):
    b = g[g["row_basis"].eq(basis)]
    q = b[b["decision_action"].eq("QUALIFY")]
    if b.empty:
        return f"    {basis}: n=0"
    if q.empty:
        return f"    {basis}: n={len(b)}, QUALIFY 0"
    w = int(q["won"].sum())
    u = float(q["profit_if_bet"].sum())
    return (
        f"    {basis}: n={len(b)}, QUALIFY {len(q)}  "
        f"{w}-{len(q)-w}  {u:+.2f}u ({100*u/len(q):+.1f}% ROI)"
    )


def report_lines(ledger, model_tag=MODEL_TAG, m0=M0, hurdle=HURDLE):
    """Human-readable retrospective block for data/ledger_report.txt."""
    g = analysis_frame(ledger, model_tag=model_tag, m0=m0, hurdle=hurdle)
    title = (
        f"v13 dynamic-price calibration SHADOW — retrospective "
        f"(M0={int(m0)}, hurdle +{100*hurdle:.1f}pp)"
    )
    if g.empty:
        return [title, "  no usable reconstructed/native v13 rows with closing prices."]

    qual = g[g["decision_action"].eq("QUALIFY")]
    abst = g[g["decision_action"].eq("ABSTAIN")]
    first = str(g["game_date"].astype(str).min())
    last = str(g["game_date"].astype(str).max())
    n_recon = int(g["row_basis"].eq("reconstructed").sum())
    n_native = int(g["row_basis"].eq("native").sum())

    y = g["won"].to_numpy(float)
    p = np.clip(g["adjusted_probability_pregame"].to_numpy(float), 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    out = [
        title,
        f"  rule — fixed 5x4 |delta| x market-q cells; p_hat=(W+{int(m0)}*qbar)/(n+{int(m0)}); "
        f"QUALIFY when p_hat - posted breakeven >= {100*hurdle:.1f}pp.",
        "  state — one expanding stream from the earliest usable v13-represented slate; "
        "same-date games share a frozen prior state and are appended only after the slate.",
        f"  basis — {len(g)} usable rows over {g['game_date'].astype(str).nunique()} slates "
        f"({first}..{last}): {n_recon} reconstructed + {n_native} native v13; "
        "the native boundary does NOT reset n.",
        "  price basis — leaned-side CLOSING devigged q for cell/prior; leaned-side posted "
        "closing ML for executable breakeven and flat-stake return. No saved-pregame fallback.",
        "  INTERPRETATION — descriptive shadow only. Reconstructed v13 is mixed-basis hindsight "
        "and close is not the execution snapshot; nothing here changes the model or a ledger decision.",
        _summary_line(g, "all v13 leans"),
        _summary_line(qual, "QUALIFY"),
        _summary_line(abst, "ABSTAIN", counterfactual=True),
        f"  probability diagnostics (all rows): Brier {brier:.4f}  log loss {logloss:.4f}",
        "  basis split (QUALIFY performance; both use the same continuously accumulated cell state):",
        _basis_line(g, "reconstructed"),
        _basis_line(g, "native"),
        "  QUALIFY / total by fixed cell:",
        "                     q<.450   .450-.500   .500-.550      .550+",
    ]

    for did, dlabel, _lo, _hi in DELTA_BANDS:
        vals = []
        for mid, _mlabel, _mlo, _mhi in MARKET_BANDS:
            c = g[g["cell_id"].eq(f"{did}_{mid}")]
            nq = int(c["decision_action"].eq("QUALIFY").sum())
            vals.append(f"{nq:>3}/{len(c):<3}")
        out.append(f"    {dlabel:<16} " + "   ".join(f"{v:>8}" for v in vals))

    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=LEDGER)
    args = ap.parse_args(argv)
    led = pd.read_csv(args.ledger, low_memory=False)
    print("\n".join(report_lines(led)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
