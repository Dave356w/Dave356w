#!/usr/bin/env python3
"""Does putting home-field advantage INTO the lean improve it?

THE QUESTION. `build_site`'s lean is `net = home_off - away_off`, a pure
matchup rate differential with NO home-field term anywhere. The market's price
carries one and carries it correctly (home closes: 53.0% actual against 53.2%
implied over 766 observations). So the model is home-blind while the price is
not, and the natural proposal is to add the missing term to the lean.

THE ANSWER, MEASURED: no. It makes the lean slightly worse on every design
tried, and the parameter is too unstable to be called a constant.

HOW THE CORRECTION IS DEFINED. Fit `P(home wins) = sigmoid(a + b*xw_net)`. The
model's decision boundary sits at `xw_net = 0`; a correctly centred one sits at
`xw_net = -a/b`. So the HFA-aware lean is `home if xw_net > -h` with
`h = a/b`, and h is the shift in the model's own units.

WHY THE SCALE FAMILY MATTERS, and the trap this probe exists to document.
Fitting across every graded family gives `b = +6.56` and `h = +0.0198`, which
would flip 25% of leans and push the home-lean share to 75.5% against a 53.2%
home win rate. That fit is INVALID: `xw_net` is not on one scale across
families, which is the whole reason `_SCALE_FAMILIES` exists. Pooling
incompatible units attenuates the slope, and since `h = a/b` an attenuated b
INFLATES the correction -- here by 2.4x. Fit inside `SCALE_TAGS` and the same
data gives `b = +19.04`, `h = +0.0082`, flipping 11.6%. A probe that ignored
the scale family would have reported a far more consequential change than
exists.

WHAT IT CANNOT ANSWER.
  * It cannot rule out a small real effect. Separating +0.10u per flipped game
    at |z| = 2 needs ~1,469 flips, which at ~1.3 flips a slate is ~1,100
    slates. This probe can only say the effect is not large.
  * It cannot speak for a metric the build does not run. It reads whatever
    `SCALE_TAGS` resolves to, so under a metric change it answers about the
    new one and its historical numbers do not carry over.
  * It says nothing about whether HFA belongs in a PRICE model. It tests one
    thing: shifting the lean's decision boundary.

WHY IT FAILS, mechanically. A constant shift only changes the decision where
`|xw_net| < h` -- the model's WEAKEST leans. On those games it replaces a weak
matchup read with "pick home", and home wins only ~53%. The delta itself is
informative (b = +19.04, z = +4.01), so the current lean already beats
always-home by a wide margin. The market's home-field content is worth having
when PRICING a game, and the published hybrid rule already consults the price
-- so that context reaches the selection through the market, where it is
handled once. Adding it to the lean as well double-counts it.

    python hfa_probe.py
"""
import os

import numpy as np
import pandas as pd

import build_site

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")
# Rows required before the walk-forward makes its first prediction. Not tuned:
# a logit on two parameters needs a few dozen rows to be worth reading, and the
# result below is unchanged at 40 or 100.
MIN_FIT = 60
# Effect size the power line is quoted against, in units per flipped game.
TARGET_EDGE = 0.10


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def _logit_fit(X, y, iters=80):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + np.eye(X.shape[1]) * 1e-9
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-11:
            break
    p = 1.0 / (1.0 + np.exp(-X @ b))
    W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + np.eye(X.shape[1]) * 1e-9)
    return b, np.sqrt(np.diag(cov))


def load(led=None):
    """Graded rows on ONE xw_net scale, with a close attached.

    Scoped to `build_site.SCALE_TAGS` rather than a pinned tag list, so a model
    bump carries this probe forward with no edit -- the lesson
    `interaction_probe` learned by scoring a lineage the build had stopped
    running. Exact-zero deltas are dropped: v7 made zero an abstention, and the
    three legacy rows that carry a lean at zero would otherwise be assigned a
    side by a comparison that is meaningless there.
    """
    if led is None:
        if not os.path.exists(LEDGER):
            return None
        led = pd.read_csv(LEDGER, low_memory=False)
    need = ["model_tag", "status", "game_date", "game_pk", "xw_lean", "xw_net",
            "home", "full_home", "full_away", "close_p_home",
            "close_home_ml", "close_away_ml"]
    if any(c not in getattr(led, "columns", ()) for c in need):
        return None
    g = led[led["model_tag"].isin(build_site.SCALE_TAGS)
            & (led["status"] == "graded")
            & led["xw_lean"].notna() & led["xw_net"].notna()
            & (led["xw_net"] != 0)
            & led["full_home"].notna() & led["close_p_home"].notna()].copy()
    if g.empty:
        return g
    g = g.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    g["home_won"] = g["full_home"] > g["full_away"]
    return g


def fit_shift(g):
    """(h, a, se_a, b, se_b) for one frame. h = a/b is the shift in xw_net."""
    X = np.column_stack([np.ones(len(g)), g["xw_net"].to_numpy(float)])
    y = g["home_won"].to_numpy(float)
    b, se = _logit_fit(X, y)
    h = b[0] / b[1] if b[1] != 0 else 0.0
    return h, b[0], se[0], b[1], se[1]


def _score(pick_home, g):
    """Record, excess over price and flat ROI for a home/away pick vector."""
    won = np.where(pick_home, g["home_won"], ~g["home_won"])
    p = np.where(pick_home, g["close_p_home"], 1 - g["close_p_home"])
    ml = np.where(pick_home, g["close_home_ml"], g["close_away_ml"])
    profit = np.where(won, _payout(ml), -1.0)
    n = len(won)
    se = float(np.sqrt(np.sum(p * (1 - p))) / n) if n else np.nan
    return dict(n=n, w=int(won.sum()), l=n - int(won.sum()),
                actual=float(won.mean()), implied=float(p.mean()),
                excess=float(won.mean() - p.mean()), excess_se=se,
                roi=float(profit.mean()), profit=profit, won=won)


def _line(r, label):
    z = r["excess"] / r["excess_se"] if r["excess_se"] else np.nan
    return (f"    {label:<26} {r['w']}-{r['l']}  {100 * r['actual']:5.1f}%  "
            f"vs price {100 * r['implied']:5.1f}%  "
            f"{100 * r['excess']:+5.1f} ± {100 * r['excess_se']:4.1f}  "
            f"z={z:+.2f}   ROI {100 * r['roi']:+6.1f}%")


def walk_forward(g):
    """Fit h on prior slates only, apply to the next. Returns a frame.

    A statistic recomputed per slate carries its own fitting noise into
    whatever it selects -- the defect that made `forward_test`'s arm 1 look
    like a discovery. So `report_lines` prints a FIXED-h holdout beside this,
    and the two must agree before either is believed.
    """
    out = []
    for d in sorted(g["game_date"].unique()):
        prior = g[g["game_date"] < d]
        cur = g[g["game_date"] == d]
        if len(prior) < MIN_FIT:
            continue
        h, *_ = fit_shift(prior)
        for _, r in cur.iterrows():
            out.append(dict(
                game_date=d, h=h, home_won=bool(r["home_won"]),
                base_home=bool(r["xw_net"] > 0),
                hfa_home=bool(r["xw_net"] + h > 0),
                close_p_home=float(r["close_p_home"]),
                close_home_ml=float(r["close_home_ml"]),
                close_away_ml=float(r["close_away_ml"])))
    return pd.DataFrame(out)


def report_lines(led=None):
    out = ["HFA-in-the-lean probe — does adding a home-field term to the model "
           "improve it?"]
    g = load(led)
    if g is None:
        out.append("    ledger unavailable or missing columns — not scored")
        return out
    if len(g) < MIN_FIT + 10:
        out.append(f"    only {len(g)} rows on the current xw_net scale; "
                   f"needs > {MIN_FIT + 10}")
        return out

    h, a, se_a, b, se_b = fit_shift(g)
    med = float(np.abs(g["xw_net"]).median())
    # Comma-joined: the tags themselves contain "+", so a "+".join renders
    # "xw+plat_consol_v9+xw+plat_consol_v10" and reads as one tag.
    out.append(f"    scale family: {', '.join(build_site.SCALE_TAGS)}")
    out.append(f"    rows on that one xw_net scale: {len(g)}")
    out.append(f"    a = {a:+.4f} ± {se_a:.4f}  (z={a / se_a:+.2f})   "
               "is there home-field in the outcomes?")
    out.append(f"    b = {b:+.3f} ± {se_b:.3f}  (z={b / se_b:+.2f})   "
               "does the delta itself predict?")
    out.append(f"    h = a/b = {h:+.5f}   ({h / med:.2f}x the median |xw_net|)")

    w = walk_forward(g)
    if w.empty:
        out.append("    not enough slates for a walk-forward yet")
        return out
    out.append("")
    out.append(f"  WALK-FORWARD (h refitted on prior slates only) — "
               f"{len(w)} rows over {w['game_date'].nunique()} slates")
    base = _score(w["base_home"].to_numpy(bool), w)
    hfa = _score(w["hfa_home"].to_numpy(bool), w)
    out.append(_line(base, "current lean"))
    out.append(_line(hfa, "HFA-corrected lean"))
    out.append(_line(_score(np.ones(len(w), bool), w), "always home (control)"))
    out.append(_line(_score((w["close_p_home"] >= 0.5).to_numpy(bool), w),
                     "always chalk (control)"))

    flip = w["base_home"].to_numpy(bool) != w["hfa_home"].to_numpy(bool)
    out.append(f"    leans flipped: {int(flip.sum())}/{len(w)} "
               f"({100 * flip.mean():.1f}%) — a shift only moves games where "
               f"|xw_net| < |h|, i.e. the model's WEAKEST leans")
    if flip.sum() > 1:
        d = (hfa["profit"] - base["profit"])[flip]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        out.append(f"    paired Δ on flipped games: {d.mean():+.3f}u/game ± "
                   f"{se:.3f}  z={d.mean() / se:+.2f}   total {d.sum():+.2f}u")
        need = int(np.ceil((2 * d.std(ddof=1) / TARGET_EDGE) ** 2))
        per_slate = flip.sum() / max(1, w["game_date"].nunique())
        out.append(f"    GATE: separating a real {TARGET_EDGE:+.2f}u/flip effect "
                   f"needs ~{need} flips, ~{need / max(per_slate, 1e-9):.0f} "
                   "slates at the observed flip rate")

    hs = w.groupby("game_date")["h"].first().to_numpy(float)
    out.append(f"    STABILITY: h ranged {hs.min():+.5f}..{hs.max():+.5f} "
               f"(sd {hs.std(ddof=1):.5f}); {int((hs < 0).sum())} of {len(hs)} "
               "slates fitted a NEGATIVE h — the fit changes sign, so this is "
               "not a constant")

    half = len(g) // 2
    h_fixed, *_ = fit_shift(g.iloc[:half])
    te = g.iloc[half:]
    out.append("")
    out.append(f"  FIXED-h HOLDOUT (fit on the first {half}, scored on the last "
               f"{len(te)}) — h={h_fixed:+.5f}")
    fb = _score((te["xw_net"] > 0).to_numpy(bool), te)
    fh = _score((te["xw_net"] + h_fixed > 0).to_numpy(bool), te)
    out.append(_line(fb, "current lean"))
    out.append(_line(fh, "HFA lean (fixed h)"))
    fl = (te["xw_net"] > 0).to_numpy(bool) != (te["xw_net"] + h_fixed > 0).to_numpy(bool)
    if fl.sum():
        d2 = (fh["profit"] - fb["profit"])[fl]
        out.append(f"    flipped {int(fl.sum())}/{len(te)}   paired Δ "
                   f"{d2.mean():+.3f}u/game   total {d2.sum():+.2f}u")
    out.append("    The fixed-h and walk-forward deltas must agree in sign "
               "before either is believed; a per-slate refit alone cannot "
               "distinguish an effect from its own fitting noise.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
