#!/usr/bin/env python3
"""Does team defence carry signal the xwOBA model strips out? As-of-date.

WHY THIS INPUT. xwOBA prices a batted ball by exit velocity and launch angle
and ignores who fields it -- deliberately, which is why it is stable. The
runs a team actually allows on contact also depend on its fielders. That
difference is ASYMMETRIC between the two teams in a game (unlike park,
weather or umpire, which both sides share and which therefore cancel out of
`net = home_off - away_off`), plausibly STABLE within a season, and entirely
ABSENT from the xwOBA lean. v13's 50/50 starter blend re-admits some of it,
mixed with luck, for the starter phase only.

THE MEASURE. Per plate appearance, r = wOBA - xwOBA. It is zero off contact
(a strikeout or walk scores the same either way) and on contact it is what
the play produced minus what the batted ball was worth: defence + park +
luck. A team's DEFENCE is its as-of-date mean r on ROAD fielded PAs only --
its own park never enters -- shrunk toward the league road mean with a K
estimated from OTHER seasons' variance components (method of moments, the
same identity `hitter_level_probe` uses for hitters).

THREE TESTS, pre-specified, in order of how much they decide:

  1. RELIABILITY. Is team road-r mostly talent or mostly noise? tau^2 and
     K = sigma^2/tau^2 per season. A tau^2 near zero ends the study here.

  2. SAME-GAME PREDICTION. Within one game both defences play in the same
     park, so park cancels exactly in the difference
         dr = r(home fielding) - r(away fielding)
     regressed on the pregame dd = d_home - d_away. Slope 1 = calibrated
     signal, 0 = none. Two-way clustered by home and away team.

  3. WINS BEYOND STRENGTH. Logit(home win) on a pregame xwOBA team-strength
     differential and D = d_away - d_home. Both are per-PA wOBA units, so a
     calibrated defence term should carry a coefficient of the same size as
     strength's -- a run saved on contact is worth the same as one saved by
     contact quality.

Then, only as a picture of scale: on the committed 2026 ledger (current
record family), `xw_net + (d_away - d_home)` beside the shipped `xw_net` --
how many leans would flip, and the flipped games' record. Nothing is fitted
on those rows (the coefficient is 1, not estimated there), but they are few
and reconstructed; the report labels them exploratory.

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

BOOT = 200
PA_PER_TEAM_GAME = 38.0     # for the runs-per-game translation only
WOBA_SCALE = 1.25           # approximate wOBA-to-runs scale; display only
STRENGTH_K = 300.0          # mild shrink for the strength control only


def with_r(pa: pd.DataFrame) -> pd.DataFrame:
    return pa.assign(r=pa["woba_value"] - pa["xwoba_pa"])


def variance_components(pa: pd.DataFrame):
    """End-of-season (tau^2, sigma^2, K) for team road-fielded r."""
    road = with_r(pa)
    road = road[road["fld_team"] == road["away_team"]]
    if road.empty:
        return None
    g = road.groupby("fld_team")["r"].agg(["mean", "size"])
    sigma2 = float(road["r"].var())
    tau2 = float(g["mean"].var() - (sigma2 / g["size"]).mean())
    k = sigma2 / tau2 if tau2 > 0 else float("inf")
    return {"tau2": tau2, "sigma2": sigma2, "k": k, "teams": len(g),
            "pa_per_team": float(g["size"].mean())}


def team_defence(pa: pd.DataFrame, k: float, targets: pd.DataFrame):
    """(fld_team, game_date) -> shrunk as-of road defence, league-centred."""
    road = with_r(pa)
    road = road[road["fld_team"] == road["away_team"]]
    t = targets.rename(columns={"team": "fld_team"})
    a = sh.asof_to(road, "fld_team", "r", "d", t)
    lg = sh.asof_to(road.assign(_all=1), "_all", "r", "lr",
                    t.assign(_all=1)[["_all", "game_date"]])
    a = a.merge(lg.drop(columns="_all"), on="game_date", how="left")
    lr = (a["lr_num"] / a["lr_n"].replace(0, np.nan)).fillna(0.0)
    kk = 1e12 if not np.isfinite(k) else k
    a["d"] = sh.shrink(a["d_num"], a["d_n"], lr, kk) - lr
    return a.rename(columns={"fld_team": "team"})[["team", "game_date", "d", "d_n"]]


def game_frame(pa: pd.DataFrame, k: float) -> pd.DataFrame:
    """One row per game: same-game residual difference and pregame defences."""
    r = with_r(pa)
    home_fld = r["fld_team"] == r["home_team"]
    agg = (r.assign(side=np.where(home_fld, "h", "a"))
             .groupby(["game_pk", "game_date", "home_team", "away_team", "side"])["r"]
             .agg(["mean", "size"]).unstack("side"))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    g = agg.reset_index().dropna()
    targets = pd.concat([g[["home_team", "game_date"]].rename(columns={"home_team": "team"}),
                         g[["away_team", "game_date"]].rename(columns={"away_team": "team"})])
    d = team_defence(pa, k, targets)
    g = g.merge(d.rename(columns={"team": "home_team", "d": "d_home", "d_n": "n_home"}),
                on=["home_team", "game_date"], how="left")
    g = g.merge(d.rename(columns={"team": "away_team", "d": "d_away", "d_n": "n_away"}),
                on=["away_team", "game_date"], how="left")
    g["dr"] = g["mean_h"] - g["mean_a"]
    g["dd"] = g["d_home"] - g["d_away"]
    g["w"] = g["size_h"] * g["size_a"] / (g["size_h"] + g["size_a"])
    return g[(g["n_home"] > 0) & (g["n_away"] > 0)].reset_index(drop=True)


def same_game_slope(g: pd.DataFrame, n_boot=BOOT):
    X = np.column_stack([np.ones(len(g)), g["dd"]])
    y = g["dr"].to_numpy(float)
    w = g["w"].to_numpy(float)

    def slope(r):
        c = sh.ols(X[r], y[r], w[r])
        return np.nan if c is None else c[1]

    c = sh.ols(X, y, w)
    if c is None:
        return None
    se, how = sh.cluster_se_twoway(slope, g["home_team"].to_numpy(),
                                   g["away_team"].to_numpy(), n_boot=n_boot)
    return {"slope": float(c[1]), "se": se, "how": how, "n": len(g),
            "sd_dd": float(g["dd"].std())}


def strength(pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Pregame xwOBA strength: team batting minus staff allowed, per PA."""
    lg = sh.league_asof(pa)
    out = games.copy()
    for side in ("home", "away"):
        t = out[[f"{side}_team", "game_date"]].rename(columns={f"{side}_team": "team"})
        bat = sh.asof_to(pa.rename(columns={"bat_team": "team"}), "team",
                         "xwoba_pa", "bt", t)
        pit = sh.asof_to(pa.rename(columns={"fld_team": "team"}), "team",
                         "xwoba_pa", "pt", t)
        s = bat.merge(pit, on=["team", "game_date"]).merge(lg, on="game_date", how="left")
        L = s["L"].fillna(s["L"].mean())
        s[f"s_{side}"] = (sh.shrink(s["bt_num"], s["bt_n"], L, STRENGTH_K)
                          - sh.shrink(s["pt_num"], s["pt_n"], L, STRENGTH_K))
        out = out.merge(s[["team", "game_date", f"s_{side}"]].rename(
            columns={"team": f"{side}_team"}), on=[f"{side}_team", "game_date"],
            how="left")
    out["strength"] = out["s_home"] - out["s_away"]
    return out


def logit(X, y, w=None, iters=25):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        if np.linalg.matrix_rank(H) < X.shape[1]:
            return None
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def win_test(pa, games_tbl, g, n_boot=BOOT):
    gm = games_tbl.merge(g[["game_pk", "d_home", "d_away"]], on="game_pk")
    gm = gm[gm["home_score"] != gm["away_score"]]
    gm = strength(pa, gm).dropna(subset=["strength", "d_home", "d_away"])
    if len(gm) < 100:
        return None
    y = (gm["home_score"] > gm["away_score"]).astype(float).to_numpy()
    D = (gm["d_away"] - gm["d_home"]).to_numpy(float)
    X = np.column_stack([np.ones(len(gm)), gm["strength"], D])

    def coef(j):
        return lambda r: (lambda c: np.nan if c is None else c[j])(logit(X[r], y[r]))

    c = logit(X, y)
    if c is None:
        return None
    se_s, _ = sh.cluster_se_twoway(coef(1), gm["home_team"].to_numpy(),
                                   gm["away_team"].to_numpy(), n_boot=n_boot)
    se_d, how = sh.cluster_se_twoway(coef(2), gm["home_team"].to_numpy(),
                                     gm["away_team"].to_numpy(), n_boot=n_boot)
    return {"n": len(gm), "b_s": float(c[1]), "se_s": se_s,
            "b_d": float(c[2]), "se_d": se_d, "how": how}


def ledger_shadow(pa26: pd.DataFrame, k: float, ledger="data/mlb_lean_ledger.csv"):
    """Exploratory: how many current-family leans a defence term would flip."""
    try:
        import build_site
        led = pd.read_csv(Path(sh.ROOT) / ledger, low_memory=False)
    except Exception as e:  # noqa: BLE001
        return f"ledger shadow unavailable ({type(e).__name__})"
    led = led[led["model_tag"].isin(build_site.RECORD_TAGS)
              & led["model_metric"].eq("xwOBA")]
    g = game_frame(pa26, k)[["game_pk", "d_home", "d_away"]]
    m = led.merge(g, on="game_pk")
    net = pd.to_numeric(m["xw_net"], errors="coerce")
    m = m[net.notna()]
    net = net[net.notna()]
    if m.empty:
        return ("no current-family ledger leans joined this season's games "
                "(needs the current season fetched)")
    adj = net + (m["d_away"] - m["d_home"])
    flip = np.sign(adj) != np.sign(net)
    graded = m["status"].eq("graded")
    fh = pd.to_numeric(m["full_home"], errors="coerce")
    fa = pd.to_numeric(m["full_away"], errors="coerce")
    home_won = fh > fa
    f = flip & graded & fh.ne(fa)
    ship_right = int(((net[f] > 0) == home_won[f]).sum())
    return (f"2026 ledger ({len(m)} current-family leans joined): a defence term "
            f"at coefficient 1 flips {int(flip.sum())} "
            f"({flip.mean():.1%}); on the {int(f.sum())} graded flips the "
            f"shipped lean won {ship_right}, the adjusted lean "
            f"{int(f.sum()) - ship_right}. EXPLORATORY: reconstructed, few games.")


def report(data_dir=sh.DEFAULT_DIR, seasons=None, n_boot=BOOT):
    seasons = seasons or sh.seasons_available(data_dir)
    out = ["TEAM DEFENCE -- road-fielded wOBA minus xwOBA, as of the day before",
           "  reconstructed history: screens an input, is not a forward record", ""]
    data = {s: sh.load(s, data_dir) for s in seasons}
    vc = {s: variance_components(d["pa"]) for s, d in data.items() if not d["pa"].empty}
    out.append("1. RELIABILITY (end of season, road-fielded PAs)")
    for s, v in vc.items():
        if v is None:
            continue
        runs = np.sqrt(max(v["tau2"], 0)) * PA_PER_TEAM_GAME / WOBA_SCALE
        out.append(f"  {s}: tau {np.sqrt(max(v['tau2'], 0)):.4f} per PA "
                   f"(~{runs:.2f} runs/game per 1 sd of true defence), "
                   f"sigma {np.sqrt(v['sigma2']):.3f}, K {v['k']:,.0f} PA, "
                   f"{v['pa_per_team']:,.0f} road PAs a team")
    out.append("")

    out.append("2. SAME-GAME PREDICTION (park cancels; slope 1 = calibrated)")
    frames = {}
    for s, d in data.items():
        if d["pa"].empty:
            continue
        others = [v["k"] for t, v in vc.items() if t != s and v and v["tau2"] > 0]
        k = float(np.median(others)) if others else (vc[s]["k"] if vc.get(s) else float("inf"))
        g = game_frame(d["pa"], k)
        frames[s] = (g, k)
        r = same_game_slope(g, n_boot)
        if r:
            z = r["slope"] / r["se"] if r["se"] > 0 else float("nan")
            out.append(f"  {s}: slope {r['slope']:+.2f} +/- {r['se']:.2f} "
                       f"(z {z:+.2f}, {r['how']}), {r['n']:,} games, "
                       f"K {k:,.0f} from {'other seasons' if others else 'this season'}; "
                       f"sd(dd) {r['sd_dd']:.4f}/PA")
    if len(frames) > 1:
        pooled = pd.concat([g for g, _ in frames.values()], ignore_index=True)
        r = same_game_slope(pooled, n_boot)
        if r:
            out.append(f"  POOLED: slope {r['slope']:+.2f} +/- {r['se']:.2f} "
                       f"(z {r['slope'] / r['se']:+.2f}), {r['n']:,} games")
    out.append("")

    out.append("3. WINS BEYOND XWOBA STRENGTH (logit; defence should match strength's scale)")
    for s, (g, k) in frames.items():
        w = win_test(data[s]["pa"], data[s]["games"], g, n_boot)
        if w:
            out.append(f"  {s}: strength {w['b_s']:+.1f} +/- {w['se_s']:.1f}   "
                       f"defence {w['b_d']:+.1f} +/- {w['se_d']:.1f} "
                       f"(z {w['b_d'] / w['se_d']:+.2f}, {w['how']}), {w['n']:,} games")
    out.append("")

    cur = max(frames) if frames else None
    if cur is not None:
        out.append("4. SCALE ON THE CURRENT LEDGER")
        out.append("  " + ledger_shadow(data[cur]["pa"], frames[cur][1]))
        out.append("")
    out += [
        "  How to read it. Test 1 gates the rest: if tau is ~0 the defence",
        "  signal is noise and a shadow is not worth building. Test 2 is the",
        "  decision: a pooled slope clearly above 0 (and near 1) says the",
        "  pregame defence rate predicts what a defence does, with park",
        "  removed. Test 3 asks whether that reaches the scoreboard beyond",
        "  contact quality. All three positive -> register a SHADOW with a",
        "  defence term, judged forward on the same games as v13.",
        "  r also carries luck and sequencing on contact; the shrinkage K",
        "  is what separates the part that persists from the part that",
        "  does not, and it is estimated, not chosen.",
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
