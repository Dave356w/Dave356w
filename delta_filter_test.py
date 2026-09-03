#!/usr/bin/env python3
"""Pre-registered forward test: the |delta| conviction filter.

REGISTERED 2026-09-03. Every parameter below is FROZEN. Nothing is fitted at
run time, nothing is re-tuned as rows arrive, and only slates STRICTLY AFTER
the registration date are scored. Tracked separately from forward_test.py and
hybrid_test.py, which hold different hypotheses registered on different dates.

THE RULE. Abstain when the model's own |xw_net| is below 0.012; otherwise back
the lean, flat one unit. Exactly 0.012 is kept. No price is read to DECIDE --
the filter looks only at the model's conviction -- so this is a filter on the
LEAN, in the same family as forward_test's arm 1 and unlike hybrid_test, whose
threshold reads the market.

WHY IT IS REGISTERED WITH A NEGATIVE PRIOR. The proposal is the natural one:
low-conviction leans look worse than high-conviction ones on the v12 ledger,
so throwing them away should raise the model's edge. Every figure reproduces:

    band                       n     record   actual vs implied   excess    ROI
    all decidable            252   156- 96     61.9% vs 55.2%    +6.7pp  +10.0%
    kept    |d| >= 0.012     170   109- 61     64.1% vs 56.8%    +7.3pp  +10.4%
    dropped |d| <  0.012      82    47- 35     57.3% vs 51.8%    +5.5pp   +8.9%

The numbers are not the problem. Three measurements taken while implementing
this are, and each one on its own is enough to register rather than ship.

  1. THE CONTRAST IS NOT DISTINGUISHABLE FROM ZERO, AT ANY THRESHOLD. The
     headline gap is kept-minus-dropped = +1.77pp, and the standard error of
     that DIFFERENCE is +/-6.63 -- z = +0.27. Swept 0.008 / 0.010 / 0.012 /
     0.015 / 0.020 / 0.025, every z lands in [-0.18, +0.46], and at 0.020 the
     sign flips. A filter whose benefit changes sign inside the range you
     would plausibly pick from is not measuring conviction.

  2. THE SEARCH TEST SAYS THE GAP IS WORSE THAN CHANCE. value_probe's standing
     rule is that a threshold found by looking is judged against the null
     maximum, never against zero. Sweeping 0.006..0.030 in 0.001 steps, the
     best contrast the real rows offer is +4.01pp at 0.017. Simulating outcomes
     at the devigged closes under "market correct, no edge", the best of that
     same sweep averages +6.98pp. P(null best >= observed) = 0.693. So a
     search over data with NOTHING in it typically returns a bigger gap than
     this one does. Compare hybrid_test, which passed the same test at 0.019.

  3. THE ALWAYS-CHALK CONTROL INVERTS THE WHOLE FINDING, AND THIS IS THE ONE
     THAT MATTERS. On the games the filter would DROP, the model beats
     always-chalk by +6.37pp. On the games it would KEEP, by +3.14pp. The kept
     half only looks better because it is more favourite-heavy -- chalk alone
     runs +4.17pp there against -0.84pp on the dropped half. Low-|delta| games
     are near pick'em (mean implied 51.8% against 56.8%), which is exactly
     where a model has something to add over backing the favourite. So the
     filter proposes to discard the games where the model contributes MOST and
     keep the ones where the base rate is doing the work. Bootstrapped, that
     +6.37pp advantage on the dropped games has a 95% CI of [-7.8, +20.6] with
     P(<= 0) = 0.19 -- not established either, but pointing the wrong way for
     the rule.

That third point is this repo's own recorded failure mode, in a new place: a
band that looks like skill and is base rate. It is the same reading that
retired the raw win-loss "Model vs market" row, and the same one hybrid_test
prints its chalk control for.

WHAT THE REGISTERED HEADLINE IS. The excess-vs-price on the DROPPED games --
the only games this rule changes anything about. A filter's entire content is
which rows it removes, so a combined "filtered ROI" would mostly restate the
model, exactly as hybrid_test's combined line mostly restates the model. The
rule is VINDICATED if the dropped games run at a NEGATIVE excess; it is
REFUTED if they keep running positive, because then the filter is throwing
away winning bets. Discovery says +5.53pp +/- 5.47.

WHAT IT CANNOT DO.
  * It cannot make the discovery sample count. Rows on or before the
    registration date are excluded by construction.
  * It cannot separate itself from the base rate quickly. See GATE_DROPPED.
  * It is scored at the CLOSE, not at an obtainable price. The filter needs no
    price to decide, so close-scoring keeps the forward reading on the same
    basis every discovery number above was measured on -- which is the
    property that matters for reading one against the other. It is a CLV
    reading, the same caveat forward_test carries.
  * It cannot tell you the model is bad, or good. It tests one filter layered
    on the model, not the model.

    python delta_filter_test.py
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. Changing any value below invalidates the test and
# restarts it from zero: the numbers are only meaningful because they were
# fixed before the scored rows existed. tests/test_delta_filter_test.py pins
# every one of them, so editing one means deliberately editing a test that says
# not to -- which is the point.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-03"      # slates STRICTLY after this date are scored
DELTA_THRESHOLD = 0.012           # |xw_net| >= keeps the lean; below abstains
STAKE = 1.0                       # flat
RULE_TAG = "xwoba_delta_filter_v1"
# No price band, no second delta tier, no per-slate refit. One number is frozen
# and this module holds it. Adding tiers to a filter whose contrast scored
# z = +0.27 is how value_probe's grid search produced a +20% cell out of noise.

# Discovery values, recorded so a forward run is read against them rather than
# against zero. Measured 2026-09-03 on the 252 decidable v12 rows over 19
# slates. Percentage points of excess vs the devigged close unless noted.
DISCOVERY_DROPPED = 82                  # of 252 decidable rows (32.5%)
DISCOVERY_DROP_RATE = 0.325
DISCOVERY_DROPPED_EXCESS = 5.53         # the registered headline, discovery value
DISCOVERY_DROPPED_SE = 5.47
DISCOVERY_CONTRAST = 1.77               # kept minus dropped
DISCOVERY_CONTRAST_SE = 6.63            # ... i.e. z = +0.27
# The control that inverts the finding. Model excess minus always-chalk excess,
# on each half. The filter discards the half where the model adds MORE.
DISCOVERY_CHALK_GAP_DROPPED = 6.37
DISCOVERY_CHALK_GAP_KEPT = 3.14
# P(null best >= observed best) over a 0.006..0.030 threshold sweep, simulating
# at the devigged closes under "market correct, no edge". Above 0.5 means the
# observed gap is smaller than a search over pure noise typically returns.
NULL_MAX_P = 0.693
# Dropped games needed for |z| = 2 on the headline, at the observed per-game sd
# of 0.4951. The first asks whether the discovery-sized 5pp effect is real; the
# second is what a plausible 3pp one needs. At 4.32 dropped games a slate that
# is ~91 and ~253 slates. Read nothing before the first.
GATE_DROPPED = 393
GATE_DROPPED_REALISTIC = 1090
# Registered prior: NEGATIVE. Not null like hybrid_test -- the chalk control
# and the null-max test both point against the rule, so a positive forward run
# would have to overcome an expectation, not merely arrive.
PRIOR = "negative"

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")

REQUIRED_COLUMNS = ("status", "game_date", "close_p_home", "xw_lean", "xw_net",
                    "home", "full_home", "full_away", "close_home_ml",
                    "close_away_ml")


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def decidable(led):
    """Graded rows the filter can act on, with NO date filter applied.

    A v5 abstention has no lean and is excluded here -- the model already
    declined it, and a filter cannot decline it twice. A row with no two-sided
    close cannot be scored. `xw_net` IS required, unlike hybrid_test: the
    delta's magnitude is the whole of this rule's input.

    Returns None when the frame is unusable, so a caller can tell "not scored"
    from "scored, nothing qualified".
    """
    if any(c not in getattr(led, "columns", ()) for c in REQUIRED_COLUMNS):
        return None
    return led[(led["status"] == "graded") & led["close_p_home"].notna()
               & led["xw_lean"].notna() & led["xw_net"].notna()].copy()


def apply_filter(g, threshold=DELTA_THRESHOLD):
    """Attach the filter's columns to a frame from `decidable`. Pure.

    Split out so the RETROSPECTIVE reading printed in `data/ledger_report.txt`
    and the FORWARD reading registered here run the same arithmetic instead of
    two copies that can drift. The date filter lives in `scored_rows` alone and
    this function must never learn one.

    `threshold` is a parameter ONLY so the discovery sweep and the search test
    can be reproduced from the same code. Every caller in this repo leaves it
    at the registered default; a test pins that.
    """
    g = g.copy()
    lean_home = (g["xw_lean"] == g["home"]).values
    home_won = (g["full_home"] > g["full_away"]).values
    g["lean_home"] = lean_home
    g["lean_won"] = np.where(lean_home, home_won, ~home_won)
    g["lean_ml"] = np.where(lean_home, g["close_home_ml"], g["close_away_ml"])
    g["p_lean"] = np.where(lean_home, g["close_p_home"], 1 - g["close_p_home"])
    g["profit"] = np.where(g["lean_won"].astype(bool),
                           STAKE * _payout(g["lean_ml"]), -STAKE)
    g["adelta"] = pd.to_numeric(g["xw_net"], errors="coerce").abs()
    # The filter. `>=` keeps, so a game exactly at the threshold is played --
    # the registered specification, and the boundary a test pins.
    g["kept"] = (g["adelta"] >= threshold).values
    # Always-chalk on the same rows. Not decoration: the entire case against
    # this rule is that the kept half is favourite-heavy, so a forward run that
    # looks good has to be read against what chalk did on the identical games.
    chalk_home = (g["close_p_home"] >= 0.5).values
    g["chalk_won"] = np.where(chalk_home, home_won, ~home_won)
    g["chalk_p"] = np.where(chalk_home, g["close_p_home"], 1 - g["close_p_home"])
    chalk_ml = np.where(chalk_home, g["close_home_ml"], g["close_away_ml"])
    g["chalk_profit"] = np.where(g["chalk_won"],
                                 STAKE * _payout(chalk_ml), -STAKE)
    return g


def scored_rows(led=None):
    """Graded rows AFTER the registration date, with the filter applied.

    Returns None when the ledger is unavailable or lacks a required column, and
    an empty frame when nothing has been played yet -- callers distinguish the
    two, because "not scored" and "scored, no rows" are different states.
    """
    if led is None:
        if not os.path.exists(LEDGER):
            return None
        led = pd.read_csv(LEDGER, low_memory=False)
    g = decidable(led)
    if g is None:
        return None
    # Strictly after: the registration date itself already held graded rows, so
    # `>=` would silently readmit part of the discovery sample. This is the ONE
    # place the forward row set is decided.
    g = g[g["game_date"].astype(str) > REGISTERED_ON]
    if g.empty:
        return g
    return apply_filter(g)


def _excess(won, p):
    """(excess, se) of a realised rate against its own implied rate.

    Poisson-binomial: the p_i are the market's, fixed rather than estimated
    from the outcomes under test, so this is defined at n = 1 and never
    degenerates to +/-0.0 -- the error-bar rule the calibration surfaces
    already follow.
    """
    n = len(won)
    if not n:
        return float("nan"), float("nan")
    se = float(np.sqrt(np.sum(p * (1 - p)))) / n
    return float(won.mean() - p.mean()), se


def _line(f, label, won_col="lean_won", p_col="p_lean", profit_col="profit"):
    n = len(f)
    if not n:
        return f"    {label:<24}    no qualifying games yet"
    won = f[won_col].astype(bool).values
    e, se = _excess(won, f[p_col].values.astype(float))
    u = float(f[profit_col].sum())
    z = e / se if se and se > 0 else float("nan")
    return (f"    {label:<24}  n={n:<4d} {int(won.sum())}-{n - int(won.sum())}  "
            f"{100 * e:+5.1f}pp +/- {100 * se:4.1f}  z={z:+.2f}  "
            f"{u:+7.2f}u  ROI {u / n * 100:+6.1f}%")


def report_lines(led=None):
    """Report body as a list of lines. Pure -- no printing, no file writes."""
    out = [f"pre-registered |delta| filter test  (registered {REGISTERED_ON}; "
           f"abstain when |xw_net| < {DELTA_THRESHOLD:.3f})"]
    g = scored_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    slates = g["game_date"].nunique() if len(g) else 0
    out.append(f"    eligible rows since registration: {len(g)} over {slates} slates")
    if not len(g):
        out.append(f"    nothing to score yet. Prior is {PRIOR.upper()}: on the "
                   "discovery rows the model beat always-chalk by MORE on the "
                   "games this filter drops than on the ones it keeps.")
        # The gate belongs here too, not only on the populated path. Zero rows
        # is exactly when a reader wants to know how long before this says
        # anything -- without it the empty line invites daily checking.
        out.append(f"    GATE: 0 of ~{GATE_DROPPED} dropped games "
                   f"(~{GATE_DROPPED / 4.32:.0f} slates at "
                   f"{DISCOVERY_DROP_RATE:.2f} of a slate). Read nothing "
                   "before then.")
        return out

    dropped = g[~g["kept"]]
    n_dr = len(dropped)
    # THE REGISTERED HEADLINE. Everything below it is context.
    if n_dr:
        e, se = _excess(dropped["lean_won"].astype(bool).values,
                        dropped["p_lean"].values.astype(float))
        out.append(f"    DROPPED-GAME EXCESS (registered)  n={n_dr:<4d} "
                   f"{100 * e:+.2f}pp +/- {100 * se:.2f}"
                   + (f"   z={e / se:+.2f}" if se and se > 0 else ""))
        ce, _ = _excess(dropped["chalk_won"].astype(bool).values,
                        dropped["chalk_p"].values.astype(float))
        out.append(f"      vs always-chalk on the same games: "
                   f"{100 * ce:+.2f}pp; model-minus-chalk {100 * (e - ce):+.2f}pp "
                   f"(discovery {DISCOVERY_CHALK_GAP_DROPPED:+.2f})")
    else:
        out.append("    DROPPED-GAME EXCESS (registered)    no abstentions yet")
    out.append(f"    discovery was {DISCOVERY_DROPPED_EXCESS:+.2f}pp +/- "
               f"{DISCOVERY_DROPPED_SE:.2f} over {DISCOVERY_DROPPED}. The rule "
               "is VINDICATED only if this goes NEGATIVE; positive means it is "
               "discarding winning bets.")

    out.append("")
    kept = g[g["kept"]]
    out.append(_line(kept, "kept    |d|>=%.3f" % DELTA_THRESHOLD))
    out.append(_line(dropped, "dropped |d|< %.3f" % DELTA_THRESHOLD))
    out.append(_line(g, "unfiltered lean"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    if n_dr and len(kept):
        ek, sk = _excess(kept["lean_won"].astype(bool).values,
                         kept["p_lean"].values.astype(float))
        ed, sd = _excess(dropped["lean_won"].astype(bool).values,
                         dropped["p_lean"].values.astype(float))
        sdiff = float(np.hypot(sk, sd))
        out.append(f"    contrast kept-minus-dropped {100 * (ek - ed):+.2f}pp "
                   f"+/- {100 * sdiff:.2f}"
                   + (f"  z={(ek - ed) / sdiff:+.2f}" if sdiff > 0 else "")
                   + f"   (discovery {DISCOVERY_CONTRAST:+.2f} +/- "
                     f"{DISCOVERY_CONTRAST_SE:.2f})")
    out.append(f"    GATE: {n_dr} of ~{GATE_DROPPED} dropped games to test the "
               f"discovery-sized effect, ~{GATE_DROPPED_REALISTIC} for a "
               f"plausible 3pp one (~253 slates at {DISCOVERY_DROP_RATE:.2f} "
               "of a slate).")
    out.append(f"    Search test at registration put P(null best >= observed) = "
               f"{NULL_MAX_P:.3f} -- the discovery gap is SMALLER than a sweep "
               "over noise typically returns. See delta_filter_test.py.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
