"""Measurement harness for the K%-interaction question. Moves nothing.

Answers one question the Phase 0 audit (`docs/k_interaction_phase0.md`) left
open: does a batter/starter K% interaction carry information the shipped
`M = B*P/L` on blended xwOBA does not already have?

That reduces to a single measurable quantity. Savant's `xwOBA` is per-PA with
strikeouts entering as zero, so a lineup's blended xwOBA already integrates
over its own K%. A K% layer can only add signal to the extent that **K% still
varies once blended xwOBA is held fixed** -- the residual spread below. If that
residual is near zero, K% is redundant with the number already in the model and
no interaction layer can help, however well specified. If it is large, two
lineups the model currently calls identical will respond differently to an
extreme starter, and the layer has something to capture.

Runs entirely on committed `data/leans_*_xw.csv`. No Savant fetch, so nothing
here depends on `.savant_cache/` and no historical state is reconstructed --
see `CLAUDE.md` No-lookahead. Precedent for the shape of this file is
`pitch_arsenal_probe.py`: a measurement that gates an arm, not the arm.

    python k_interaction_probe.py

Nothing here is imported by `build_site.py` or `grade_leans.py`. No lean, no
ledger row, and no `MODEL_TAG` is touched by running it.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# wOBA weights for the non-contact branches. Used only to decompose a blended
# xwOBA into its implied contact component; they are not fit here.
W_BB = 0.690
W_HBP = 0.720
# League HBP% is not carried in the slate CSVs and is stable enough that
# assuming it costs nothing at this precision. Stated, not measured.
LG_HBP = 0.011

DATA_DIR = os.environ.get("DATA_DIR", "data")


# --- pure functions --------------------------------------------------------

def log5(p_bat: float, p_pit: float, lam: float, damp: float = 1.0) -> float:
    """Odds-ratio matchup estimate. `lam` must be the population both component
    rates were measured against. damp=1.0 is vanilla log5; no value is fit here.
    """
    eps = 1e-9
    p_bat = min(max(p_bat, eps), 1 - eps)
    p_pit = min(max(p_pit, eps), 1 - eps)
    lam = min(max(lam, eps), 1 - eps)
    ratio = (((p_bat * p_pit) / lam) /
             (((1 - p_bat) * (1 - p_pit)) / (1 - lam))) ** damp
    return ratio / (1.0 + ratio)


def bbe_share(k: float, bb: float, hbp: float) -> float:
    """Multiplicative residual, closed under [0, 1] by construction.

    The subtractive form (1 - k - bb - hbp) can go negative for a high-K
    starter against a high-walk lineup; this cannot.
    """
    return (1.0 - k) * max(0.0, 1.0 - bb - hbp)


def xwoba_from_branches(k: float, bb: float, hbp: float, xwobacon: float) -> float:
    """Blended per-PA xwOBA from an event split. K contributes 0 by definition,
    entering only through its suppression of the contact share.
    """
    return bb * W_BB + hbp * W_HBP + bbe_share(k, bb, hbp) * xwobacon


def implied_xwobacon(xwoba: float, k: float, bb: float, hbp: float = LG_HBP) -> float:
    """Invert `xwoba_from_branches` for the contact branch.

    This is the step that makes the probe possible without a Savant xwOBAcon
    pull: given the blended rate and the event rates the slate CSVs already
    carry, the contact component is determined.
    """
    share = bbe_share(k, bb, hbp)
    if share <= 0:
        return np.nan
    return (xwoba - bb * W_BB - hbp * W_HBP) / share


# --- data ------------------------------------------------------------------

def load_sides(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """One row per game-side from the committed slate CSVs, rates as fractions.

    K%/BB% are stored in percentage points and are converted here; every
    downstream function in this module expects fractions.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "leans_*_xw.csv")))
    if not files:
        raise SystemExit(f"no slate CSVs under {data_dir}/")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    out = pd.DataFrame({"game_pk": d.get("game_pk"), "side": d.get("side")})
    for src, dst, pct in [
        ("opp_xwOBA", "b_xwoba", False), ("opp_K%", "b_k", True),
        ("opp_BB%", "b_bb", True), ("pit_xwOBA", "p_xwoba", False),
        ("pit_K%", "p_k", True), ("pit_BB%", "p_bb", True),
        ("lg_xwOBA", "lg_xwoba", False), ("lg_K%", "lg_k", True),
        ("lg_BB%", "lg_bb", True),
    ]:
        v = pd.to_numeric(d.get(src), errors="coerce")
        out[dst] = v / 100.0 if pct else v
    return out.dropna().reset_index(drop=True)


# --- the deciding measurement ---------------------------------------------

def composition_residual(df: pd.DataFrame) -> dict:
    """How much lineup K% still varies once blended xwOBA is held fixed.

    Regresses lineup K% on lineup xwOBA and reports the residual sd. That
    residual is the only room a K% layer has to work in: variation in K% that
    the number already in the model cannot see.
    """
    k, x = df["b_k"].to_numpy(), df["b_xwoba"].to_numpy()
    slope, intercept = np.polyfit(x, k, 1)
    resid = k - (slope * x + intercept)
    r = np.corrcoef(x, k)[0, 1]
    return {
        "n": int(k.size),
        "k_sd_raw": float(k.std(ddof=1)),
        "k_sd_resid": float(resid.std(ddof=1)),
        "r_k_vs_xwoba": float(r),
        "variance_explained": float(r ** 2),
        "slope_per_100pts": float(slope * 0.100),
    }


def sensitivity(xwobacon_grid=(0.340, 0.370, 0.400), bb=0.085, hbp=LG_HBP) -> list[dict]:
    """d(xwOBA)/d(K%) per 1pp. Sets the error budget for everything downstream."""
    return [{"xwobacon": c,
             "pts_per_1pp": (xwoba_from_branches(0.21, bb, hbp, c)
                             - xwoba_from_branches(0.20, bb, hbp, c)) * 1000}
            for c in xwobacon_grid]


def lean_impact(df: pd.DataFrame) -> dict:
    """Shipped `B*P/L` against a K%-interacted reconstruction, on the lean.

    Model A is what ships: the multiplicative form on blended per-PA xwOBA.
    Model B decomposes both sides, interacts K% through log5 and the other
    branches multiplicatively, and recombines -- the *replacement* form, not a
    multiplier layered on top (see `docs/pitch_mix_theory.md` for why the
    multiplier form is the weaker one).

    Reports flip rate banded by how decisive the shipped lean was, because a
    flip on a lean of 0.001 and a flip on a lean of 0.05 are not the same event.
    """
    lam = float(df["lg_k"].median())
    lg_x, lg_bb = float(df["lg_xwoba"].median()), float(df["lg_bb"].median())
    lg_con = implied_xwobacon(lg_x, lam, lg_bb)

    a, b = [], []
    for _, r in df.iterrows():
        a.append(r["b_xwoba"] * r["p_xwoba"] / lg_x)
        b.append(xwoba_from_branches(
            log5(r["b_k"], r["p_k"], lam),
            r["b_bb"] * r["p_bb"] / lg_bb, LG_HBP,
            implied_xwobacon(r["b_xwoba"], r["b_k"], r["b_bb"])
            * implied_xwobacon(r["p_xwoba"], r["p_k"], r["p_bb"]) / lg_con))
    df = df.assign(mx_a=a, mx_b=b)

    nets = []
    for _, g in df.groupby("game_pk"):
        if set(g["side"]) != {"home", "away"}:
            continue
        h, aw = g[g["side"] == "away"].iloc[0], g[g["side"] == "home"].iloc[0]
        # xw_net mirrors grade_leans.py: the away row carries the home offense.
        nets.append(((h["mx_a"] - aw["mx_a"]), (h["mx_b"] - aw["mx_b"])))
    na, nb = np.array([n[0] for n in nets]), np.array([n[1] for n in nets])
    flip = np.sign(na) != np.sign(nb)

    bands = []
    for lo, hi, lab in [(0, .005, "<.005 (noise)"), (.005, .0186, ".005-.0186"),
                        (.0186, .047, ".0186-.047"), (.047, 9, ">.047")]:
        m = (np.abs(na) >= lo) & (np.abs(na) < hi)
        if m.sum():
            bands.append({"band": lab, "n": int(m.sum()),
                          "flip_pct": float(100 * flip[m].mean())})
    return {
        "games": int(na.size),
        "median_abs_net_a": float(np.median(np.abs(na))),
        "median_abs_net_b": float(np.median(np.abs(nb))),
        "corr": float(np.corrcoef(na, nb)[0, 1]),
        "flip_pct": float(100 * flip.mean()),
        "median_abs_shift_pts": float(np.median(np.abs(nb - na)) * 1000),
        "bands": bands,
    }


def main() -> None:
    df = load_sides()
    print(f"K% interaction probe -- {len(df)} game-sides from {DATA_DIR}/\n")

    print("sensitivity  d(xwOBA)/d(K%) per 1pp:")
    for r in sensitivity():
        print(f"   xwOBAcon={r['xwobacon']:.3f}  {r['pts_per_1pp']:+.2f} pts")

    c = composition_residual(df)
    print(f"\nDECIDING MEASUREMENT -- lineup K% spread at fixed blended xwOBA (n={c['n']}):")
    print(f"   raw sd of lineup K%          {c['k_sd_raw']*100:.3f} pp")
    print(f"   residual sd, xwOBA held fixed {c['k_sd_resid']*100:.3f} pp")
    print(f"   r(K%, xwOBA) = {c['r_k_vs_xwoba']:+.4f}   "
          f"variance explained {100*c['variance_explained']:.1f}%")
    print(f"   -> {100*(1-c['variance_explained']):.1f}% of K% variation is invisible "
          f"to the shipped blended number")

    li = lean_impact(df)
    print(f"\nlean impact, shipped B*P/L vs K%-interacted ({li['games']} games):")
    print(f"   median |xw_net|  A {li['median_abs_net_a']*1000:.2f} pts   "
          f"B {li['median_abs_net_b']*1000:.2f} pts")
    print(f"   corr {li['corr']:.5f}   median |shift| {li['median_abs_shift_pts']:.2f} pts")
    print(f"   sign flips {li['flip_pct']:.2f}% overall")
    for b in li["bands"]:
        print(f"      |xw_net_A| {b['band']:14s} n={b['n']:4d}  flips {b['flip_pct']:5.2f}%")

    print("\nNothing was changed. See docs/k_interaction_phase0.md for the gate.")


if __name__ == "__main__":
    main()
