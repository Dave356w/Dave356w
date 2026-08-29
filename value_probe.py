#!/usr/bin/env python3
"""Is there a tradable relationship between `xw_net` and the closing price?

The standing answer is NO, and the naive version is INVERTED -- flagging the
largest model-vs-market gaps selects the losing subset, monotonically. That
finding is recorded in CLAUDE.md ("The value-bet signal that does not exist"),
which also says: do not re-derive it by hand, and do not ship a value call
without re-running the walk-forward first.

This probe exists because the second half of that rule stopped being
satisfiable. `walkforward.py` was deleted on 2026-08-27 and was the only
out-of-sample control on the model; with it gone nothing contradicted a
flattering live window. So the rule named an instrument that no longer
existed, which is how a "measured, do not revisit" finding quietly decays into
an assumption.

WHAT THIS IS. A walk-forward over LEDGER ROWS. Every row is a decision the
model actually published before first pitch, joined to its own DK closing
price. Anything fitted here is fitted on strictly prior slates and scored on
the next, so no slate is scored on its own data.

WHAT THIS IS NOT, and cannot become. It is NOT the deleted replay and does not
restore what that did. The replay rebuilt historical predictions from scratch
and could therefore score model versions that never shipped; this can only
score decisions that were actually made. So it CANNOT answer "would version X
have done better", and it cannot close the fidelity gap that got the replay
removed -- it sidesteps that gap by never reconstructing a prediction at all.

Three further things it cannot do:

  * It cannot separate model skill from base rate on a single family. That is
    the whole point of the always-chalk control printed beside every arm: at
    these sample sizes a family's raw win rate is mostly the rate at which
    favourites happened to win that fortnight.
  * It cannot establish an edge from one family looking good. The per-family
    table exists to show the sign flipping era to era; read the pooled row and
    the control, never a single family's z.
  * It says nothing about rows with no market join (DK close via ESPN), which
    are excluded rather than imputed.
  * The band grid (section 5) cannot tell you a cell is good. It is built to
    tell you the opposite -- that a cell you already like sits inside the range
    noise produces -- because with 15 cells the best one is a maximum over 15
    draws, not an estimate. It compares against that maximum and then tests the
    selection forward; a cell surviving both is a hypothesis to test on new
    slates, never a green light.

Reads committed artifacts only. No Savant pull, no StatsAPI, no lookahead, so
it runs anywhere.

    python value_probe.py
"""
import numpy as np
import pandas as pd

# IMPORTED, not reimplemented. The SE of (realised rate - mean implied) is
# Poisson-binomial because the p_i are fixed by the market rather than
# estimated from the outcomes under test -- which is what makes it defined at
# n=1 instead of collapsing to 0.0 on an all-W bucket. CLAUDE.md records a
# live incident from getting that wrong, and its fix was "one derivation, not
# two"; a second copy here would be the same defect one module out.
from build_site import _excess_se as excess_se

LEDGER = "data/mlb_lean_ledger.csv"

# Minimum prior rows before a slate is scored. Not a significance gate -- the
# logit simply will not converge usefully below this, and a coefficient fitted
# on a handful of games would make the first slates noise that the later ones
# then inherit through the cumulative ROI line.
TRAIN_MIN = 60

PRICE_BANDS = [("big dog", 0.0, 0.40), ("small dog", 0.40, 0.50),
               ("small fav", 0.50, 0.60), ("big fav", 0.60, 1.0)]


def load():
    """Graded rows carrying a lean, a delta and a devigged closing price."""
    d = pd.read_csv(LEDGER, low_memory=False)
    g = d[(d["status"] == "graded") & d["close_p_home"].notna()
          & d["xw_lean"].notna() & d["xw_net"].notna()].copy()
    g["lean_home"] = g["xw_lean"] == g["home"]
    g["home_won"] = g["full_home"] > g["full_away"]
    g["won"] = np.where(g["lean_home"], g["home_won"], ~g["home_won"])
    # The price of the side the model actually leaned, not of the home side.
    g["p_lean"] = np.where(g["lean_home"], g["close_p_home"], 1 - g["close_p_home"])
    g["ml_lean"] = np.where(g["lean_home"], g["close_home_ml"], g["close_away_ml"])
    # Control: back the closing favourite on the identical rows. A control is
    # only a control if it is scored on the rows the model was scored on.
    g["chalk_home"] = g["close_p_home"] >= 0.5
    g["chalk_won"] = np.where(g["chalk_home"], g["home_won"], ~g["home_won"])
    return g.sort_values("game_date")


def payout(ml):
    """Profit on a 1u win at American odds."""
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def units(f):
    return float(np.where(f["won"], payout(f["ml_lean"]), -1.0).sum())


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(X, y, iters=60):
    """Newton-Raphson logit with a ridge nudge for separation safety."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-X @ b))
        W = np.clip(pr * (1 - pr), 1e-9, None)
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]),
                                   X.T @ (y - pr))
        except np.linalg.LinAlgError:
            break
        b = b + step
    return b


def arm(f, label):
    if not len(f):
        return
    rate, imp = f["won"].mean(), f["p_lean"].mean()
    se, u = excess_se(f["p_lean"]), units(f)
    chalk = f["chalk_won"].mean()
    print(f"  {label:<22} n={len(f):4d}  model {rate:.3f}  chalk {chalk:.3f} "
          f"({rate - chalk:+.3f})   vs price {rate - imp:+.4f} +/- {se:.4f} "
          f"z={(rate - imp) / se:+.2f}   {u:+7.2f}u ({u / len(f) * 100:+.1f}%)")


def walk_forward(g):
    """Per-slate model probability from a logit fitted on strictly prior slates."""
    out = []
    for s in sorted(g["game_date"].unique()):
        tr, te = g[g["game_date"] < s], g[g["game_date"] == s]
        if len(tr) < TRAIN_MIN:
            continue
        b = fit(tr["xw_net"].values, tr["home_won"].values.astype(float))
        ph = 1.0 / (1.0 + np.exp(-(b[0] + b[1] * te["xw_net"].values)))
        t = te.copy()
        t["model_p"] = np.where(t["lean_home"], ph, 1 - ph)
        t["gap"] = t["model_p"] - t["p_lean"]
        out.append(t)
    return pd.concat(out) if out else None


def oos_log_loss(g, use_market, use_net):
    mu, sd = g["xw_net"].mean(), g["xw_net"].std()
    losses = []
    for s in sorted(g["game_date"].unique()):
        tr, te = g[g["game_date"] < s], g[g["game_date"] == s]
        if len(tr) < TRAIN_MIN:
            continue

        def design(f):
            cols = []
            if use_market:
                cols.append(logit(f["close_p_home"].values))
            if use_net:
                cols.append((f["xw_net"].values - mu) / sd)
            return np.column_stack(cols)

        if use_market or use_net:
            b = fit(design(tr), tr["home_won"].values.astype(float))
            p = 1.0 / (1.0 + np.exp(-(np.column_stack(
                [np.ones(len(te)), design(te)]) @ b)))
        else:
            p = te["close_p_home"].values          # raw close, nothing fitted
        p = np.clip(p, 1e-6, 1 - 1e-6)
        y = te["home_won"].values.astype(float)
        losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.concatenate(losses).mean())


# Cell floor for the band grid. Not a significance gate -- below this a cell's
# ROI has a null sd above ~20pp, so it is a coin flip rendered as a percentage.
CELL_MIN = 25
DELTA_BANDS, PRICE_QUANTILES = 3, 5


def market_view(g, market):
    """Add p_lean / ml / profit for `market` in ("full", "f5"). F5 ties push."""
    f = g.copy()
    if market == "full":
        f["p_lean"] = np.where(f["lean_home"], f["close_p_home"], 1 - f["close_p_home"])
        f["ml"] = np.where(f["lean_home"], f["close_home_ml"], f["close_away_ml"])
        f["push"] = False
        f["hit"] = f["won"].astype(float)
    else:
        f["p_lean"] = np.where(f["lean_home"], f["f5_close_p_home"], 1 - f["f5_close_p_home"])
        f["ml"] = np.where(f["lean_home"], f["f5_close_home_ml"], f["f5_close_away_ml"])
        f["push"] = f["xw_f5"] == "T"
        f["hit"] = (f["xw_f5"] == "W").astype(float)
    f["profit"] = np.where(f["push"], 0.0,
                           np.where(f["hit"] > 0, payout(f["ml"]), -1.0))
    return f.sort_values("game_date")


def band_grid(g, market, rng):
    """3 delta bands x 5 price quintiles, against the null and then forward.

    The grid is printed because it is what gets explored; the two lines under it
    are what decide it. A 15-cell search returns a MAXIMUM, so the only honest
    reference is the distribution of that maximum when nothing is there -- and
    the only honest confirmation is picking a cell on past slates and betting it
    on the next.
    """
    f = market_view(g, market)
    f = f[f["p_lean"].notna() & f["ml"].notna()]
    if len(f) < 150:
        return
    f["dband"] = pd.qcut(f["xw_net"].abs(), DELTA_BANDS, labels=["low", "mid", "hi"])
    f["pband"] = pd.qcut(f["p_lean"], PRICE_QUANTILES,
                         labels=[f"p{i + 1}" for i in range(PRICE_QUANTILES)])
    pbs = [f"p{i + 1}" for i in range(PRICE_QUANTILES)]

    def roi(c):
        return c["profit"].sum() / len(c) * 100 if len(c) else float("nan")

    print(f"\n   {market.upper()} market, n={len(f)}")
    print(f"   {'':6s}" + "".join(f"{p:>13s}" for p in pbs) + f"{'row':>13s}")
    for db in ["low", "mid", "hi"]:
        cells = []
        for pb in pbs:
            c = f[(f["dband"] == db) & (f["pband"] == pb)]
            cells.append(f"{roi(c):+7.1f}%({len(c):3d})" if len(c) else "     --      ")
        r = f[f["dband"] == db]
        print(f"   {db:6s}" + "".join(cells) + f"{roi(r):+7.1f}%({len(r):3d})")
    print(f"   {'col':6s}" + "".join(
        f"{roi(f[f['pband'] == pb]):+7.1f}%({int((f['pband'] == pb).sum()):3d})"
        for pb in pbs) + f"{roi(f):+7.1f}%({len(f):3d})")

    groups = [f[(f["dband"] == db) & (f["pband"] == pb)]
              for db in ["low", "mid", "hi"] for pb in pbs]
    elig = [c for c in groups if len(c) >= CELL_MIN]
    if not elig:
        return
    best = max(roi(c) for c in elig)
    sims = []
    for _ in range(3000):
        mx = -1e9
        for c in elig:
            w = rng.random(len(c)) < c["p_lean"].values
            pr = np.where(c["push"].values, 0.0,
                          np.where(w, payout(c["ml"].values), -1.0))
            mx = max(mx, pr.sum() / len(c) * 100)
        sims.append(mx)
    sims = np.array(sims)
    print(f"\n   best cell (n>={CELL_MIN}): {best:+.1f}%")
    print(f"   under the null -- market correct, NO edge -- the best of "
          f"{len(elig)} cells averages {sims.mean():+.1f}% "
          f"[p5 {np.percentile(sims, 5):+.1f}, p95 {np.percentile(sims, 95):+.1f}]")
    print(f"   P(null best >= observed best) = {(sims >= best).mean():.3f}")

    bets = []
    for s in sorted(f["game_date"].unique()):
        tr, te = f[f["game_date"] < s], f[f["game_date"] == s]
        if len(tr) < 150:
            continue
        tb = tr.groupby(["dband", "pband"], observed=True)["profit"].agg(["sum", "count"])
        tb = tb[tb["count"] >= CELL_MIN]
        if tb.empty:
            continue
        pick = (tb["sum"] / tb["count"]).idxmax()
        sel = te[(te["dband"] == pick[0]) & (te["pband"] == pick[1])]
        if len(sel):
            bets.append(sel)
    if bets:
        b = pd.concat(bets)
        print(f"   FORWARD: best cell picked on prior slates only, bet on the next -- "
              f"{len(b)} bets / {b['game_date'].nunique()} slates, "
              f"{b['profit'].sum():+.2f}u, ROI {b['profit'].sum() / len(b) * 100:+.1f}%")


def report():
    g = load()
    print("=" * 92)
    print(f"VALUE PROBE -- {len(g)} graded rows with a lean and a closing price")
    print("=" * 92)

    print("\n1. DOES THE LEAN BEAT ITS OWN CLOSING PRICE?")
    print("   (always-chalk control scored on the identical rows)")
    arm(g, "all graded")
    for tag in sorted(g["model_tag"].unique()):
        if (g["model_tag"] == tag).sum() >= 100:
            arm(g[g["model_tag"] == tag], tag)

    wf = walk_forward(g)
    if wf is None:
        print("\n(insufficient history to walk forward)")
        return

    print(f"\n2. THE NAIVE VALUE BET -- walk-forward over {len(wf)} games, "
          f"{wf['game_date'].nunique()} scored slates")
    print("   flag the biggest model-vs-market gaps and back them:")
    print(f"   {'threshold':<16}{'n':>5}{'win%':>8}{'units':>9}{'ROI':>9}")
    for thr in (None, 0.00, 0.02, 0.04, 0.06, 0.10):
        f = wf if thr is None else wf[wf["gap"] > thr]
        if len(f) < 5:
            continue
        u = units(f)
        label = "all bets" if thr is None else f"gap > {thr:.2f}"
        print(f"   {label:<16}{len(f):>5}{f['won'].mean() * 100:>7.1f}%"
              f"{u:>9.2f}{u / len(f) * 100:>8.1f}%")

    print("\n   is it just dog-picking? median gap split computed WITHIN each band:")
    for name, lo, hi in PRICE_BANDS:
        b = wf[(wf["p_lean"] >= lo) & (wf["p_lean"] < hi)]
        if len(b) < 12:
            print(f"     {name:<10} n={len(b)} -- too thin to split")
            continue
        med, cells = b["gap"].median(), []
        for half, mask in (("hi-gap", b["gap"] > med), ("lo-gap", b["gap"] <= med)):
            f = b[mask]
            cells.append(f"{half} {units(f) / len(f) * 100:+6.1f}% (n={len(f):3d})")
        print(f"     {name:<10} " + "   ".join(cells))

    print("\n3. DOES xw_net ADD ANYTHING ON TOP OF PRICE?")
    z = (g["xw_net"].values - g["xw_net"].mean()) / g["xw_net"].std()
    X = np.column_stack([logit(g["close_p_home"].values), z])
    y = g["home_won"].values.astype(float)
    b = fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    pr = 1.0 / (1.0 + np.exp(-Xd @ b))
    W = np.clip(pr * (1 - pr), 1e-9, None)
    se = np.sqrt(np.diag(np.linalg.inv((Xd * W[:, None]).T @ Xd + 1e-6 * np.eye(3))))
    print(f"   n={len(g)}   market logit {b[1]:+.3f} +/- {se[1]:.3f} "
          f"(1.00 = close needs no correction)")
    print(f"           z(xw_net) {b[2]:+.3f} +/- {se[2]:.3f}   z={b[2] / se[2]:+.2f}")

    print("\n   out-of-sample log loss (lower is better):")
    rows = [("raw close (nothing fitted)", oos_log_loss(g, False, False)),
            ("market-fitted", oos_log_loss(g, True, False)),
            ("price + delta", oos_log_loss(g, True, True)),
            ("delta alone", oos_log_loss(g, False, True))]
    for name, v in sorted(rows, key=lambda kv: kv[1]):
        print(f"     {name:<28} {v:.4f}")
    print("\n   If 'price + delta' ranks BELOW 'market-fitted', the delta is")
    print("   subtracting information from the price -- what a noise feature does.")

    print("\n4. PER-FAMILY, CHRONOLOGICAL -- read the sign's STABILITY, not one row.")
    print("   A single hot family is the documented way this question gets answered wrong.")
    print(f"   {'family':<24}{'n':>5}{'model':>8}{'chalk':>8}{'edge':>8}"
          f"{'vs price':>11}{'z':>7}{'ROI':>9}")
    fams = (g.groupby("model_tag")["game_date"].min().sort_values().index)
    for tag in fams:
        f = g[g["model_tag"] == tag]
        if len(f) < 25:
            continue
        imp, rate = f["p_lean"].mean(), f["won"].mean()
        u = units(f)
        print(f"   {tag:<24}{len(f):>5}{rate:>8.3f}{f['chalk_won'].mean():>8.3f}"
              f"{rate - f['chalk_won'].mean():>+8.3f}{rate - imp:>+11.4f}"
              f"{(rate - imp) / excess_se(f['p_lean']):>+7.2f}"
              f"{u / len(f) * 100:>+8.1f}%")


    print("\n5. BAND GRID -- discretising the same signal does not create one.")
    print("   Read the null line, not the grid: the best of 15 cells is a maximum")
    print("   over 15 draws, so it is large even when nothing is there.")
    rng = np.random.default_rng(7)
    for market in ("full", "f5"):
        band_grid(g, market, rng)


if __name__ == "__main__":
    report()
