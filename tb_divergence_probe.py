"""Team total-bases divergence: does a 60d-minus-15d form gap beat the price?

Runs anywhere the committed ledger is present. No network, by construction --
see "Where the inputs come from" below.

The rule
--------
For each side, rate the club only at the venue it is about to play -- the home
team from its home games, the visitor from its road games -- as its own total
bases plus its opponent's allowed, over the league split for that venue:

    delta_w = (home_scored + away_allowed)/lg_home - (away_scored + home_allowed)/lg_away

over a trailing window of w days, and take the gap between a slow and a fast
window:

    div_delta = delta_60 - delta_15        back HOME if positive, else AWAY

A positive gap means the club looks better over two months than over two
weeks, so this backs the side whose recent form UNDERSTATES its longer
baseline. It is a mean-reversion bet on team form, and it reads no pitcher and
no price.

Why the prior is negative
-------------------------
This arrives with a discovery result attached -- 41-21 (66.1%) at the frozen
threshold below, +26.2% ROI -- and that result is an artifact of how it was
measured, in three ways that were each traced rather than suspected:

1. **The threshold was chosen on the games it was scored on.** Eight were
   swept and the best reported at p = 0.0200. Holm-adjusted across the eight
   the same table's best is **p = 0.16**. Nothing in it was significant.
2. **It was priced at a flat -110 on both sides.** The rule picks a side
   without reading a price, so that ROI is a counterfactual on a book that
   does not exist. This module scores against the ledger's own devigged close
   instead, which is the only reason the number below means anything.
3. **The sample was about seven slate-days** -- 100 games at ~14/day. Games on
   one slate share the league-average denominators and often a club, so the
   binomial test behind that p-value assumed independence the sample did not
   have.

So the registered prior is NEGATIVE: this tracks a hypothesis whose supporting
evidence has already been explained away, exactly as `forward_test.py` arm 1
does, so that a good forward run reads as a hypothesis and not as a discovery.

The threshold is frozen at the value the discovery sweep selected, NOT re-swept
here. Re-sweeping forward would repeat the original defect with a longer
sample. The other thresholds print as secondary context and carry no claim.

Two construction defects, neither fixed, both measured
------------------------------------------------------
**Thin venue splits inflate the feature.** A club's 15-day window holds ~6
games at a given venue and sometimes none, after a road trip. An empty split
falls back to the league mean, which pulls `delta_15` toward zero and pushes
`|div_delta|` UP. On the ledger this is measurable and reported: the threshold
is partly selecting for games with the LEAST evidence behind them rather than
for divergence. That is a property of the rule as proposed, so it is measured
and printed rather than silently corrected.

**The bottom of the ninth biases total bases against good home teams.** A home
club that is winning does not bat in the ninth -- roughly 11% fewer plate
appearances -- so `home_tb` is depressed precisely for clubs that win at home,
while the total bases it ALLOWS are never truncated. Differencing two windows
cancels this only if the club's home win rate is stable across both, which is
the very thing the rule claims to detect a change in. `--per-pa` reports the
rate-denominated version beside the shipped one; it is context, not the
registration.

Where the inputs come from, and why this is not the deleted backtest
--------------------------------------------------------------------
Total bases are recomputed from the ledger's own graded actuals --
`act_h_*`, `act_2b_*`, `act_3b_*`, `act_hr_*` -- which are immutable facts
written at grade time, not rates reconstructed from a present-day pull. The
walk-forward backtest deleted 2026-08-27 left an unmeasured fidelity gap
because it rebuilt Savant rates a different way from the live build. There is
no analogous gap here: there is nothing to reconstruct, and no Savant call, so
the no-lookahead surface `.savant_cache/` exists to guard is not touched.

What IS a limitation: the ledger carries the slates this site built, not every
MLB game. Coverage runs ~13 games/date against a ~15-game slate, so the league
splits are a near-complete but not complete census. `--coverage` prints the
per-date counts so a thin stretch is visible rather than absorbed.

The cutoff is the slate date, strictly
--------------------------------------
A game's windows end at 00:00 of its own game_date, so nothing from its own
slate -- including an earlier first pitch that day -- can enter. That is
stricter than first-pitch ordering and does not depend on
`scheduled_start_utc`, which only 848 of 997 rows carry. It costs same-day
information and buys a bound that holds on every row.

Controls, because a rate without one is not a result
----------------------------------------------------
Always-chalk especially: a form rule that leans the better club backs the
favourite more often than not, so without that control its record reads as its
own skill. The deleted backtest's entry in CLAUDE.md is the precedent -- most
of a raw 63.1%-vs-52.9% gap there was base rate. Always-home and the shipped
model's own lean are scored on the identical rows for the same reason.

Usage:
    python tb_divergence_probe.py                # forward + retrospective
    python tb_divergence_probe.py --coverage     # per-date slate coverage
    python tb_divergence_probe.py --per-pa       # rate-denominated context
    python tb_divergence_probe.py --sweep        # secondary thresholds, no claim
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from market_backfill import excess_se

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")

# Frozen at registration. None of these is refit at run time.
REGISTERED_ON = "2026-09-17"
SHORT_DAYS = 15
LONG_DAYS = 60
THRESHOLD = 0.15

# Secondary thresholds. Context only -- the registration is THRESHOLD above.
SWEEP = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)

# A plausible sustainable edge against a devigged close, for the far gate.
PLAUSIBLE_EXCESS = 0.02

# Frozen gates, in bets, from `bets_needed` at registration against the
# retrospective block's mean price (0.53). Constants rather than a run-time
# recomputation for the reason every other registration freezes its own: a gate
# that moves with the sample is not a gate. `bets_needed` is kept as the
# derivation that produced them and a test re-derives both.
#
# The NEAR gate is sized to the discovery effect (+11.6pp). Clearing it means
# only that the rule has not disqualified itself -- a selected maximum
# reproducing itself over a handful of slates would prove nothing. The FAR gate
# is what a plausible edge needs, and is on the order of ten seasons.
GATE_BETS = 74
GATE_BETS_REALISTIC = 2491

# Mean devigged price the gates were sized at. Frozen with them.
GATE_PRICE = 0.53
DISCOVERY_EXCESS = 0.116

STAKE = 1.0


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def _excess(won, p):
    """(excess, se) against the market's own probabilities. Defined at n=1.

    `excess_se` is imported rather than restated for the reason
    `market_backfill` gives at its definition: two spellings of a statistic
    drift, and a reader cannot see which one they are looking at.
    """
    n = len(won)
    if not n:
        return float("nan"), float("nan")
    return float(np.mean(won) - np.mean(p)), float(excess_se(np.asarray(p, dtype=float)))


def total_bases(h, doubles, triples, hr):
    """TB from a box line. `h` already counts each extra-base hit once."""
    return h + doubles + 2.0 * triples + 3.0 * hr


def _ledger(led=None):
    if led is not None:
        return led
    if not os.path.exists(LEDGER):
        return None
    return pd.read_csv(LEDGER, low_memory=False)


ACT_COLS = [f"act_{s}_{side}" for side in ("home", "away")
            for s in ("h", "2b", "3b", "hr", "pa", "ab")]


def game_frame(led=None):
    """One row per game with total bases, result and the devigged close.

    Returns None when the ledger is unavailable or lacks a required column.
    The ledger carries one row per model tag, so it is deduplicated on
    `game_pk` -- the actuals and the price are properties of the GAME and are
    identical across tags, while pooling them would weight a game by how many
    model versions happened to look at it.
    """
    led = _ledger(led)
    if led is None:
        return None
    needed = ACT_COLS + ["game_pk", "game_date", "home", "away",
                         "full_home", "full_away", "close_p_home",
                         "close_home_ml", "close_away_ml"]
    if any(c not in led.columns for c in needed):
        return None

    g = led.drop_duplicates("game_pk").copy()
    for c in ACT_COLS + ["full_home", "full_away", "close_p_home",
                         "close_home_ml", "close_away_ml"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g = g[g[ACT_COLS].notna().all(axis=1)
          & g[["full_home", "full_away", "close_p_home",
               "close_home_ml", "close_away_ml"]].notna().all(axis=1)].copy()
    if g.empty:
        return g

    g = g[g["full_home"] != g["full_away"]]          # no undecided games
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["home_tb"] = total_bases(g["act_h_home"], g["act_2b_home"],
                               g["act_3b_home"], g["act_hr_home"])
    g["away_tb"] = total_bases(g["act_h_away"], g["act_2b_away"],
                               g["act_3b_away"], g["act_hr_away"])
    g["home_won"] = (g["full_home"] > g["full_away"]).astype(int)
    return g.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def reconstruction_guard(g):
    """Abort rather than report if the box columns are not what they are read as.

    Every number below divides by a league total-bases mean, so a mis-identified
    column would be plausible and wrong -- the failure `bp_ablation` and
    `matchup_form_probe` each guard against by re-deriving a stored quantity.
    There is no stored `act_tb` to check against, so the check is the algebra
    total bases must satisfy: hits contain the extra-base hits, plate
    appearances contain the at-bats, and total bases are at least hits.
    """
    problems = []
    for side in ("home", "away"):
        xbh = g[f"act_2b_{side}"] + g[f"act_3b_{side}"] + g[f"act_hr_{side}"]
        if (g[f"act_h_{side}"] < xbh).any():
            problems.append(f"act_h_{side} < 2B+3B+HR on "
                            f"{int((g[f'act_h_{side}'] < xbh).sum())} rows")
        if (g[f"act_pa_{side}"] < g[f"act_ab_{side}"]).any():
            problems.append(f"act_pa_{side} < act_ab_{side} on "
                            f"{int((g[f'act_pa_{side}'] < g[f'act_ab_{side}']).sum())} rows")
        if (g[f"{side}_tb"] < g[f"act_h_{side}"]).any():
            problems.append(f"{side}_tb < act_h_{side}")
    if problems:
        raise ValueError("total-bases reconstruction failed: " + "; ".join(problems))
    return True


def compute_delta(hist, home, away, per_pa=False):
    """Home-minus-away venue-split rating over one window. Pure.

    Returns 0.0 on an empty or degenerate window rather than raising: a club
    with no history is not evidence for either side, and the caller's
    threshold then declines the game.
    """
    if hist is None or hist.empty:
        return 0.0
    if per_pa:
        hv = hist["home_tb"] / hist["act_pa_home"]
        av = hist["away_tb"] / hist["act_pa_away"]
    else:
        hv, av = hist["home_tb"], hist["away_tb"]

    lg_home, lg_away = float(hv.mean()), float(av.mean())
    if not (np.isfinite(lg_home) and np.isfinite(lg_away)) or lg_home <= 0 or lg_away <= 0:
        return 0.0

    at_home = hist["home"] == home
    on_road = hist["away"] == away
    home_scored = float(hv[at_home].mean()) if at_home.any() else lg_home
    home_allowed = float(av[at_home].mean()) if at_home.any() else lg_away
    away_scored = float(av[on_road].mean()) if on_road.any() else lg_away
    away_allowed = float(hv[on_road].mean()) if on_road.any() else lg_home

    return ((home_scored + away_allowed) / lg_home
            - (away_scored + home_allowed) / lg_away)


def divergence(g, per_pa=False):
    """Attach `div_delta` to every game whose long window is fully covered.

    The cutoff is the game's own slate date, exclusive, so no game on that date
    -- itself included -- can enter either window. Games inside the first
    LONG_DAYS of the ledger are dropped: their long window is truncated by the
    start of the data, which rescales the feature rather than merely adding
    noise to it.
    """
    if g is None or g.empty:
        return g
    dates = g["game_date"].values
    data_start = g["game_date"].iloc[0]

    rows = []
    for row in g.itertuples(index=False):
        cutoff = row.game_date
        if cutoff < data_start + pd.Timedelta(days=LONG_DAYS):
            continue
        hi = int(np.searchsorted(dates, np.datetime64(cutoff), side="left"))
        lo_s = int(np.searchsorted(
            dates, np.datetime64(cutoff - pd.Timedelta(days=SHORT_DAYS)), side="left"))
        lo_l = int(np.searchsorted(
            dates, np.datetime64(cutoff - pd.Timedelta(days=LONG_DAYS)), side="left"))

        short_hist, long_hist = g.iloc[lo_s:hi], g.iloc[lo_l:hi]
        d_short = compute_delta(short_hist, row.home, row.away, per_pa)
        d_long = compute_delta(long_hist, row.home, row.away, per_pa)
        div = d_long - d_short

        rows.append({
            "game_pk": row.game_pk,
            "game_date": row.game_date,
            "home": row.home,
            "away": row.away,
            "delta_short": d_short,
            "delta_long": d_long,
            "div_delta": div,
            "abs_div_delta": abs(div),
            # How much venue-specific evidence the short window actually had.
            "venue_n": int(min((short_hist["home"] == row.home).sum(),
                               (short_hist["away"] == row.away).sum())),
            "home_won": row.home_won,
            "close_p_home": row.close_p_home,
            "close_home_ml": row.close_home_ml,
            "close_away_ml": row.close_away_ml,
        })
    return pd.DataFrame(rows)


def apply_rule(f, threshold=THRESHOLD):
    """Select a side and price it. Pure.

    A zero `div_delta` points nowhere, so it is declined rather than sent to
    the road team the way the discovery code sent it.
    """
    if f is None or f.empty:
        return f
    d = f[(f["abs_div_delta"] >= threshold) & (f["div_delta"] != 0.0)].copy()
    if d.empty:
        return d
    picks_home = d["div_delta"] > 0
    d["selection"] = np.where(picks_home, d["home"], d["away"])
    d["bet_won"] = np.where(picks_home, d["home_won"] == 1, d["home_won"] == 0)
    d["p_bet"] = np.where(picks_home, d["close_p_home"], 1.0 - d["close_p_home"])
    ml = np.where(picks_home, d["close_home_ml"], d["close_away_ml"])
    d["profit"] = np.where(d["bet_won"], STAKE * _payout(ml), -STAKE)
    return d


def controls(d):
    """Always-chalk, always-home and the shipped lean, on the SAME rows."""
    if d is None or d.empty:
        return {}
    chalk_home = d["close_p_home"] >= 0.5
    home_ml, away_ml = d["close_home_ml"].values, d["close_away_ml"].values
    out = {}

    out["always-chalk"] = _series(
        won=np.where(chalk_home, d["home_won"] == 1, d["home_won"] == 0),
        p=np.where(chalk_home, d["close_p_home"], 1.0 - d["close_p_home"]),
        ml=np.where(chalk_home, home_ml, away_ml))
    out["always-home"] = _series(
        won=(d["home_won"] == 1).values,
        p=d["close_p_home"].values,
        ml=home_ml)
    return out


def _series(won, p, ml):
    won = np.asarray(won, dtype=bool)
    profit = np.where(won, STAKE * _payout(ml), -STAKE)
    return pd.DataFrame({"bet_won": won, "p_bet": np.asarray(p, dtype=float),
                         "profit": profit})


def clustered_excess_se(d, draws=4000, seed=11):
    """Excess SE from resampling whole SLATE-DAYS, not individual games.

    `excess_se` treats every game as an independent Bernoulli at its own
    price. Games on one slate share the league split this rule divides by and
    sometimes a club, so that assumption is the one the discovery sweep's
    binomial p-value made and should not be taken on trust here.

    It is reported BESIDE the Poisson-binomial figure rather than replacing
    it, because on this ledger it comes back SMALLER -- day-level variation is
    milder than within-day -- and a lone clustered number that happens to
    flatter the result is exactly what a reader should be able to check.
    """
    if d is None or len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    days = [x for _, x in d.groupby("game_date", sort=True)]
    n_days = len(days)
    if n_days < 2:
        return float("nan")
    stats = np.empty(draws, dtype=float)
    for i in range(draws):
        s = pd.concat([days[j] for j in rng.integers(0, n_days, n_days)])
        stats[i] = s["bet_won"].astype(float).mean() - s["p_bet"].mean()
    return float(stats.std(ddof=1))


def bets_needed(effect, p_mean):
    """Bets to separate `effect` from zero at |z| = 2, at the market's own sd.

    Var of one Bernoulli at a fixed devigged price is p(1-p), so
    se(excess) = sqrt(p(1-p)/n) and n = 4*p(1-p)/effect^2.
    """
    if not effect or not np.isfinite(effect) or effect <= 0:
        return float("inf")
    return 4.0 * p_mean * (1.0 - p_mean) / (effect ** 2)


def gate_guard():
    """The frozen gates must still be what `bets_needed` says they are.

    The gates are literals so they cannot drift with the sample, but a literal
    and the arithmetic behind it can part company just as quietly -- the
    failure the reconstruction guard above exists to stop in the other
    direction. So the derivation runs on every report and disagreeing with its
    own constants is an abort, not a printed note.

    This is also why `bets_needed` has a production caller at all: a helper
    kept alive only by the tests written for it is the shape
    `tests/test_no_dead_functions.py` exists to catch.
    """
    near = int(bets_needed(DISCOVERY_EXCESS, GATE_PRICE))
    far = int(round(bets_needed(PLAUSIBLE_EXCESS, GATE_PRICE)))
    if (near, far) != (GATE_BETS, GATE_BETS_REALISTIC):
        raise ValueError(
            f"frozen gates disagree with their derivation: "
            f"constants ({GATE_BETS}, {GATE_BETS_REALISTIC}) vs "
            f"derived ({near}, {far})")
    return True


def _line(label, f, clustered=False):
    n = 0 if f is None else len(f)
    if not n:
        return f"    {label:<28}  no qualifying games yet"
    won = f["bet_won"].astype(bool).values
    e, se = _excess(won, f["p_bet"].values.astype(float))
    u = float(f["profit"].sum())
    z = e / se if se and np.isfinite(se) and se > 0 else float("nan")
    line = (f"    {label:<28}  n={n:<4d} {int(won.sum())}-{n - int(won.sum())}  "
            f"{100 * e:+5.1f}pp +/- {100 * se:4.1f}  z={z:+5.2f}  "
            f"{u:+7.2f}u  ROI {100 * u / n:+6.1f}%")
    if clustered:
        cse = clustered_excess_se(f)
        cz = e / cse if cse and np.isfinite(cse) and cse > 0 else float("nan")
        line += f"\n    {'  day-clustered se':<28}  {100 * cse:4.1f}pp  z={cz:+5.2f}"
    return line


def venue_profile(f):
    """Is the threshold selecting for divergence, or for missing evidence?"""
    if f is None or f.empty:
        return []
    out = []
    for t in (0.00, THRESHOLD, 0.30):
        sub = f[f["abs_div_delta"] >= t]
        if sub.empty:
            continue
        out.append(f"    |div| >= {t:.2f}   n={len(sub):<4d} "
                   f"mean venue games {sub['venue_n'].mean():5.2f}   "
                   f"empty split {100 * (sub['venue_n'] == 0).mean():4.1f}%")
    corr = float(np.corrcoef(f["abs_div_delta"], f["venue_n"])[0, 1])
    out.append(f"    corr(|div_delta|, venue games) = {corr:+.3f}")
    return out


def report(led=None, per_pa=False, do_sweep=False, coverage=False):
    g = game_frame(led)
    if g is None:
        return ["tb_divergence: ledger unavailable or missing a required column"]
    if g.empty:
        return ["tb_divergence: no gradeable games in the ledger"]
    reconstruction_guard(g)
    gate_guard()

    lines = [f"TB DIVERGENCE PROBE  (registered {REGISTERED_ON}, prior NEGATIVE)",
             f"  rule: back home if delta_{LONG_DAYS}d - delta_{SHORT_DAYS}d > 0, "
             f"act if |div| >= {THRESHOLD:.2f}",
             f"  scored against the ledger's devigged close, never a flat -110",
             ""]

    per_day = g.groupby("game_date").size()
    lines.append(f"  ledger: {len(g)} decided games over {len(per_day)} dates "
                 f"({per_day.mean():.1f}/date, median {per_day.median():.0f})")
    if coverage:
        lines.append("  per-date coverage:")
        for d, n in per_day.items():
            lines.append(f"    {d.date()}  {n:2d}" + ("   <- thin" if n < 8 else ""))

    f = divergence(g, per_pa=per_pa)
    if f is None or f.empty:
        lines.append(f"  no games survive the {LONG_DAYS}-day warm-up yet")
        return lines
    lines.append(f"  after {LONG_DAYS}d warm-up: {len(f)} games over "
                 f"{f['game_date'].nunique()} dates"
                 + ("   [per-PA context, not the registration]" if per_pa else ""))
    lines.append("")

    fwd = f[f["game_date"].astype(str) > REGISTERED_ON]
    fwd_bets = apply_rule(fwd)
    n_fwd = 0 if fwd_bets is None else len(fwd_bets)
    lines.append(f"  FORWARD (slates strictly after {REGISTERED_ON})")
    lines.append(_line(f"tb_div |div|>={THRESHOLD:.2f}", fwd_bets, clustered=True))
    lines.append("")

    lines.append("  RETROSPECTIVE (discovery-era rows; a read, not a verdict)")
    lines.append("    These rows OVERLAP the sample the rule was found on, so they are")
    lines.append("    not independent evidence for it. Only the FORWARD block above is.")
    d = apply_rule(f)
    lines.append(_line(f"tb_div |div|>={THRESHOLD:.2f}", d, clustered=True))
    for label, ctrl in controls(d).items():
        lines.append(_line(f"{label} (same rows)", ctrl))

    if d is not None and not d.empty:
        won = d["bet_won"].astype(bool).values
        e, _ = _excess(won, d["p_bet"].values.astype(float))
        p_mean = float(d["p_bet"].mean())
        lines.append("")
        lines.append("  GATES (frozen; bets to reach |z| = 2 at the market's own sd)")
        lines.append(f"    near, at the discovery {100 * DISCOVERY_EXCESS:+.1f}pp   "
                     f"{GATE_BETS:,d} bets")
        lines.append(f"    far,  at a plausible   {100 * PLAUSIBLE_EXCESS:+.1f}pp   "
                     f"{GATE_BETS_REALISTIC:,d} bets")
        lines.append(f"    forward bets so far                {n_fwd:,d}")

    lines.append("")
    lines.append("  VENUE-SAMPLE PROFILE (is the gate selecting for missing evidence?)")
    lines.extend(venue_profile(f))

    if do_sweep:
        lines.append("")
        lines.append("  SECONDARY THRESHOLDS -- context only, no claim is registered here")
        for t in SWEEP:
            lines.append(_line(f"|div| >= {t:.2f}", apply_rule(f, t)))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-pa", action="store_true",
                    help="rate-denominate total bases by plate appearances (context)")
    ap.add_argument("--sweep", action="store_true",
                    help="print secondary thresholds (context, no claim)")
    ap.add_argument("--coverage", action="store_true",
                    help="print per-date slate coverage")
    args = ap.parse_args()
    for line in report(per_pa=args.per_pa, do_sweep=args.sweep, coverage=args.coverage):
        print(line)


if __name__ == "__main__":
    main()
