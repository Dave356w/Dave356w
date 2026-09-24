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


def beta_only(rows, y="y_w"):
    """Point estimate of beta_v, for carrying a coefficient forward."""
    if rows.empty:
        return None
    b = rows["b"].to_numpy(float)
    p = ssh.p_at(rows, sh.SHIPPED_K)
    X = np.column_stack([np.ones(len(rows)), b, p, b * p, rows["dv"]])
    c = sh.ols(X, rows[y].to_numpy(float))
    return None if c is None else float(c[4])


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
    svs, rws, seasons_used = [], [], []
    for s in seasons:
        sv, rows = season_rows(sh.load(s, data_dir))
        if sv.empty:
            out.append(f"{s}: no velocity rows")
            continue
        svs.append(sv)
        rws.append(rows)
        seasons_used.append(s)
        out += _block(str(s), sv, rows, n_boot)
    if len(svs) > 1:
        out += _block("POOLED", pd.concat(svs, ignore_index=True),
                      pd.concat(rws, ignore_index=True), n_boot)
    cur = max(seasons) if seasons else None
    prior = [r for s, r in zip(seasons_used, rws) if s < cur]
    if cur is not None and prior:
        b = beta_only(pd.concat(prior, ignore_index=True))
        out.append(f"RETROSPECTIVE ON THE {cur} LEDGER (v12/v13), beta_v from "
                   f"{', '.join(str(s) for s in seasons_used if s < cur)}")
        d = sh.load(cur, data_dir)
        out += (ledger_shadow(b, d["pa"], d["velo"], season=cur) if b is not None
                else ["  no earlier-season fit"])
        out.append("")
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


# ------------------------------------------------- retrospective ledger arm

def ledger_net(led: pd.DataFrame, d_home=0.0, d_away=0.0, k=None):
    """Rebuild `xw_net` from the ledger's own inputs, optionally moving each
    starter's rate by `d_<side>` (per-PA wOBA units) and re-shrinking it at
    `k` instead of the shipped K.

    Same reconstruction the pitcher x lineup analysis verified against the
    stored `xw_net` (max difference 3e-16 over 540 rows) -- checked again
    by the caller on every run, never assumed. Home offence faces the AWAY
    staff, so `_away` columns build home's edge.

    The re-shrink un-does K=100 toward the league value rather than v13's
    population/role target, which the ledger does not store: approximate,
    and labelled so wherever it is printed.
    """
    def n(c):
        return pd.to_numeric(led[c], errors="coerce")
    L = n("mx_xwoba_sp_away") - n("edge_xwoba_sp_away")
    edge = {}
    for s, d in (("away", d_away), ("home", d_home)):
        B, Bn = n(f"opp_xwoba_vs_sp_{s}"), n(f"opp_xwoba_neutral_{s}")
        P, Pb, q = n(f"starter_xwoba_{s}"), n(f"bullpen_xwoba_{s}"), n(f"sp_share_{s}")
        if k is not None:
            bf = n(f"sp_rate_bf_{s}").fillna(0.0)
            P = L + (P - L) * (bf + sh.SHIPPED_K) / (bf + k)
        P = P + d
        edge[s] = q * B * P / L + (1 - q) * Bn * Pb / L - L
    return edge["away"] - edge["home"]


def ledger_shadow(beta_v, pa: pd.DataFrame, velo: pd.DataFrame,
                  ledger="data/mlb_lean_ledger.csv", season=None):
    """Apply a velocity term fitted on EARLIER seasons to the current ledger.

    Each starter's rate moves by beta_v * dv, dv being his pregame velocity
    change from this season's Statcast (0 when he has fewer than three prior
    starts). The coefficient never sees these games. Reported beside the same
    arms with K = 300, because the shrinkage study says that is the larger
    and better-established correction.
    """
    out = []
    try:
        import build_site
        led = pd.read_csv(Path(sh.ROOT) / ledger, low_memory=False)
    except Exception as e:  # noqa: BLE001
        return [f"  ledger arm unavailable ({type(e).__name__})"]
    led = led[led["model_tag"].isin(build_site.RECORD_TAGS)
              & led["model_metric"].eq("xwOBA")]
    if season is not None:
        led = led[led["game_date"].astype(str).str.startswith(str(season))]
    led = led.reset_index(drop=True)
    if led.empty:
        return [f"  no current-family ledger rows in {season}"]
    ship = pd.to_numeric(led["xw_net"], errors="coerce")
    base = ledger_net(led)
    ok = ship.notna() & base.notna()
    drift = float((base - ship)[ok].abs().max()) if ok.any() else float("nan")
    if not np.isfinite(drift) or drift > 1e-9:
        return [f"  ledger arm refused: reconstruction differs from xw_net "
                f"by {drift:.2e}; nothing below would be v12/v13"]

    sv = start_velocity(pa, velo)[["game_pk", "pitcher", "dv"]]
    side = sh.starters(pa).merge(pa[["game_pk", "home_team"]].drop_duplicates(),
                                 on="game_pk")
    side["side"] = np.where(side["fld_team"] == side["home_team"], "home", "away")
    side = side.merge(sv.rename(columns={"pitcher": "starter"}),
                      on=["game_pk", "starter"], how="left")
    dv = side.pivot_table(index="game_pk", columns="side", values="dv")
    gp = pd.to_numeric(led["game_pk"], errors="coerce")
    dv_h = gp.map(dv.get("home", pd.Series(dtype=float))).fillna(0.0)
    dv_a = gp.map(dv.get("away", pd.Series(dtype=float))).fillna(0.0)
    has = gp.map(dv.get("home", pd.Series(dtype=float))).notna() | \
        gp.map(dv.get("away", pd.Series(dtype=float))).notna()

    fh = pd.to_numeric(led["full_home"], errors="coerce")
    fa = pd.to_numeric(led["full_away"], errors="coerce")
    graded = led["status"].eq("graded") & fh.notna() & fa.notna() & fh.ne(fa)
    home_won = fh > fa
    rows = ok & graded

    def rec(net):
        right = ((net > 0) == home_won)[rows]
        return int(right.sum()), int(len(right))

    ship_w, n_g = rec(base)
    out.append(f"  v12/v13 ledger: {int(ok.sum())} leans, {n_g} graded; "
               f"reconstruction matches xw_net (max diff {drift:.1e}); "
               f"{int((has & ok).sum())} carry a pregame dv for at least one starter")
    out.append(f"  beta_v {beta_v * 1000:+.2f} points/mph, fitted on earlier seasons only")
    out.append(f"  {'arm':<24} {'flips':>6} {'graded flips W-L (arm)':>24} {'record':>10}")
    out.append(f"  {'shipped v12/v13':<24} {'-':>6} {'-':>24} "
               f"{ship_w:>4}-{n_g - ship_w:<5}")
    arms = (("velocity", None, True), ("K=300", 300.0, False),
            ("K=300 + velocity", 300.0, True))
    for name, k, vel in arms:
        net = ledger_net(led, d_home=beta_v * dv_h if vel else 0.0,
                         d_away=beta_v * dv_a if vel else 0.0, k=k)
        flip = (np.sign(net) != np.sign(base)) & ok
        f = flip & rows
        arm_w = int(((net > 0) == home_won)[f].sum())
        w, _ = rec(net)
        out.append(f"  {name:<24} {int(flip.sum()):>6} "
                   f"{arm_w:>12}-{int(f.sum()) - arm_w:<11} {w:>4}-{n_g - w:<5}")
    out.append("  Only flipped games can change a record, and they are few: read the")
    out.append("  W-L of the flips against a coin (sd ~ sqrt(n)/2 wins), not the")
    out.append("  headline record. RECONSTRUCTED and post hoc -- the velocity data")
    out.append("  is pregame but was not saved pregame, and the K=300 re-shrink")
    out.append("  approximates v13's target. Evidence for a shadow, not a result.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=sh.DEFAULT_DIR)
    p.add_argument("--boot", type=int, default=BOOT)
    a = p.parse_args()
    print("\n".join(report(a.data, n_boot=a.boot)))


if __name__ == "__main__":
    main()
