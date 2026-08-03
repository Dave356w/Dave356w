"""Does the batter x pitch-type deviation carry enough signal to be worth a
model arm? Measure before building.

Run this on a machine that can reach Savant:

    python pitch_arsenal_probe.py                 # current season vs prior
    python pitch_arsenal_probe.py --season 2026 --min-pa 25

It answers three questions and prints the budget each answer has to clear:

1. **How big is the raw spread** of a hitter's league-relative wOBA by pitch
   type, and how much of it is sampling noise? Method of moments on the cell
   PA distribution splits observed variance into within-PA (noise) and
   between-player (signal) components, which also implies the shrinkage `K`
   the arm should use -- the same estimator family the build retired in v8 for
   the season wOBA, used here only as a diagnostic.

2. **Is it stable year over year?** Season-to-season correlation of the same
   (batter, pitch type) deviation is the honest reliability proxy: the
   leaderboard is a season aggregate, so a within-season split-half is not
   available without pitch-level data.

3. **Does the implied signal beat the noise it injects?** The arm moves a game
   delta by roughly the per-hitter cell noise over sqrt(9) hitters, times
   sqrt(2) sides. Compared against the current-family median |xw_net| from the
   ledger, that is the number that decides whether the arm can pay.

Nothing here writes to `data/` or touches a lean; it prints, and optionally
writes a report to a path you name.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import pitch_arsenal as pa

# sd of a single plate appearance's wOBA. Most PAs are 0, a single ~0.9, a
# home run ~2.0. Override with --sd-pa once measured from pitch-level data;
# every noise figure below scales linearly with it.
SD_PA = 0.85
LEDGER_PATH = os.path.join("data", "mlb_lean_ledger.csv")


def _fetch_csv(url, cache_name, cache_dir=".savant_cache"):
    """Cached CSV pull, standalone so the probe does not import build_site.

    The cache key is the season, not the date: a probe re-run mid-season reuses
    the first pull. Delete `.savant_cache/probe_*.csv` to refresh. That cache is
    gitignored and never feeds a build -- reading a stale leaderboard into a
    lean would be the lookahead this repo forbids."""
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"probe_{cache_name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(r.text)
    return pd.read_csv(path)


def median_lean(ledger_path=LEDGER_PATH, scale_tags=None):
    """Current-family median |xw_net|, the budget the arm has to beat.

    Mixing scale families here would compare against the wrong units -- pre-v8
    deltas live on a different scale -- so the tags are required to mean
    anything. Returns None when the ledger cannot supply them.
    """
    if not os.path.exists(ledger_path):
        return None
    led = pd.read_csv(ledger_path)
    if scale_tags:
        led = led[led["model_tag"].isin(scale_tags)]
    d = pd.to_numeric(led.get("xw_net"), errors="coerce").abs().dropna()
    return float(d.median()) if len(d) else None


def variance_decomposition(cells, sd_pa=SD_PA):
    """(observed var, noise var, signal var, implied K) for one pitch type.

    Sampling variance of a cell mean is sd_pa^2/n, so the PA-weighted mean of
    1/n gives the noise component; what the observed dispersion has left over
    is between-player signal. K = noise_per_PA / signal is the pseudo-sample
    that balances them.
    """
    d = cells[cells["pa"].notna() & (cells["pa"] > 0) & cells["dev"].notna()]
    if len(d) < 30:
        return None
    obs = float(np.var(d["dev"], ddof=1))
    noise = float(np.mean(sd_pa ** 2 / d["pa"]))
    signal = max(obs - noise, 0.0)
    k = (sd_pa ** 2 / signal) if signal > 0 else np.inf
    return obs, noise, signal, k


def arm_noise(sd_pa, weights_pa, k, n_hitters=9, n_sides=2):
    """sd the arm adds to a game delta, given cell PA behind each weight.

    `weights_pa` is [(arsenal weight, cell PA), ...] for a representative
    starter. Cell noise after shrinkage is (sd/sqrt(n)) * n/(n+k); weights
    combine in quadrature, hitters average down by sqrt(9), and two
    independent sides widen the game delta by sqrt(2).
    """
    per_hitter = np.sqrt(sum(
        (w * (sd_pa / np.sqrt(n)) * n / (n + k)) ** 2 for w, n in weights_pa))
    return float(per_hitter / np.sqrt(n_hitters) * np.sqrt(n_sides))


def deviations(batter_arsenal):
    """Tidy cells with `dev` = shrunk-free league-relative deviation.

    dev is the raw ratio minus the batter's own PA-weighted centre, i.e. what
    the arm would keep a fraction of. Measuring it unshrunk is the point: the
    shrinkage constant is what this probe is trying to choose.
    """
    lg = pa.league_by_pitch_type(batter_arsenal)
    if not lg:
        return pd.DataFrame(columns=["player_id", "pitch_type", "pa", "dev"])
    raw, overall = pa.batter_relative_cells(batter_arsenal, lg, k=0.0)
    rows = [{"player_id": pid, "pitch_type": t, "ratio": r,
             "dev": r - overall.get(pid, np.nan)}
            for (pid, t), r in raw.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pas = batter_arsenal.set_index(["player_id", "pitch_type"])["pa"]
    out["pa"] = [pas.get((r.player_id, r.pitch_type), np.nan)
                 for r in out.itertuples(index=False)]
    return out.dropna(subset=["dev"])


def year_over_year(dev_now, dev_prior, min_pa=50):
    """Correlation of the same (batter, pitch type) deviation across seasons."""
    a = dev_now[dev_now["pa"] >= min_pa][["player_id", "pitch_type", "dev"]]
    b = dev_prior[dev_prior["pa"] >= min_pa][["player_id", "pitch_type", "dev"]]
    j = a.merge(b, on=["player_id", "pitch_type"], suffixes=("_now", "_prior"))
    if len(j) < 30:
        return None, len(j)
    return float(j["dev_now"].corr(j["dev_prior"])), len(j)


def report(season, min_pa, sd_pa, fetch=_fetch_csv, scale_tags=None):
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    bat, pit = pa.fetch_arsenals(lambda u, c: fetch(u, f"{c}_{season}"), season,
                                 min_pa=min_pa)
    if bat.empty:
        say("batter arsenal leaderboard unavailable or unrecognised -> "
            "cannot measure; check the column names in pitch_arsenal.py")
        return "\n".join(lines)

    dev = deviations(bat)
    say(f"season {season}: {bat['player_id'].nunique()} batters, "
        f"{len(bat)} cells, {len(dev)} with a computable deviation")
    say(f"cell PA: median {bat['pa'].median():.0f}, "
        f"p25 {bat['pa'].quantile(.25):.0f}, p75 {bat['pa'].quantile(.75):.0f}")
    say()

    say("per pitch type: observed sd, noise sd, implied signal sd, implied K")
    for ptype, grp in dev.groupby("pitch_type", sort=False):
        vd = variance_decomposition(grp, sd_pa)
        if vd is None:
            continue
        obs, noise, signal, k = vd
        say(f"  {ptype:<4} n={len(grp):<5} obs {np.sqrt(obs):.4f}  "
            f"noise {np.sqrt(noise):.4f}  signal {np.sqrt(signal):.4f}  "
            f"K {k:,.0f}" if np.isfinite(k) else
            f"  {ptype:<4} n={len(grp):<5} obs {np.sqrt(obs):.4f}  "
            f"noise {np.sqrt(noise):.4f}  signal 0  K inf (all noise)")
    say()

    prior_bat, _ = pa.fetch_arsenals(
        lambda u, c: fetch(u, f"{c}_{season - 1}"), season - 1, min_pa=min_pa)
    if not prior_bat.empty:
        r, n = year_over_year(dev, deviations(prior_bat))
        say(f"year-over-year reliability ({season} vs {season - 1}, "
            f"cells >= 50 PA): r = {r:.3f} over n = {n}"
            if r is not None else
            f"year-over-year reliability: too few paired cells (n = {n})")
    else:
        say(f"prior season {season - 1} unavailable -> no reliability estimate")
    say()

    med = median_lean(scale_tags=scale_tags)
    say("noise the arm would inject, vs the lean it has to improve")
    if med is None:
        say("  ledger median unavailable")
    elif scale_tags:
        say(f"  current-family median |xw_net| = {med:.4f} "
            f"(tags: {', '.join(scale_tags)})")
    else:
        say(f"  median |xw_net| = {med:.4f} -- ALL model tags pooled, which "
            "mixes delta scales; pass the current SCALE_TAGS for a real budget")
    mix = [(.55, 110), (.35, 55), (.10, 25)]     # representative 3-pitch starter
    for k in (100, 200, 400, 600, 800):
        g = arm_noise(sd_pa, mix, k)
        pct = f"{100 * g / med:.0f}% of the median lean" if med else ""
        say(f"  K={k:<4} game-delta sd {g:.4f}   {pct}")
    say()
    say("read: the arm is worth building only where the implied signal sd is a "
        "real fraction of the observed sd AND the year-over-year r is well "
        "above zero. If the noise row at the K implied above is most of the "
        "median lean, the arm cannot pay and the shadow columns should stay "
        "off.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int,
                    default=int(os.environ.get("SEASON", 2026)))
    ap.add_argument("--min-pa", type=int, default=25)
    ap.add_argument("--sd-pa", type=float, default=SD_PA)
    ap.add_argument("--out", default=None,
                    help="optional path for the printed report (not data/)")
    a = ap.parse_args(argv)
    try:
        import build_site
        scale_tags = build_site.SCALE_TAGS
    except Exception:  # noqa: BLE001 -- the probe must run standalone
        scale_tags = None
    text = report(a.season, a.min_pa, a.sd_pa, scale_tags=scale_tags)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
