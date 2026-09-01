#!/usr/bin/env python3
"""Pre-registered forward test: the hybrid market-direction rule.

REGISTERED 2026-09-01. Every parameter below is FROZEN. Nothing is fitted at
run time, nothing is re-tuned as rows arrive, and only slates STRICTLY AFTER
the registration date are scored. Tracked separately from forward_test.py's
two arms, which hold different hypotheses registered on a different date.

THE RULE. Let q be the model-selected side's two-sided no-vig market
probability. Follow the xwOBA lean when q >= 0.45; back the opposing side when
q < 0.45. Exactly 0.45 follows. Abstain when the model publishes no lean or no
valid two-sided price exists. Flat one unit.

    ModelSideP = close_p_home        if the lean is the home side
                 1 - close_p_home    otherwise

This is a hard-switch DIRECTION rule. It does not estimate the selected side's
win probability and nothing here should be read as one.

WHY IT IS REGISTERED RATHER THAN SHIPPED. Applied row by row to the v12 ledger
it looked strong -- and every figure below reproduces exactly, so the numbers
are not the problem:

    branch                  record   actual vs implied   profit     ROI
    follow  q >= 0.45       135-73    64.9% vs 56.4%    +27.94u   +13.4%
    fade    q <  0.45        11-4     73.3% vs 58.6%     +3.56u   +23.8%
    combined hybrid         146-77    65.5% vs 56.5%    +31.50u   +14.1%
    plain xwOBA lean        139-84    62.3% vs 55.4%    +22.63u   +10.1%

Measured on 243 v12 rows: 229 graded, 223 carrying a lean and a close, 6 model
abstentions and 14 pending excluded. The hybrid's aggregate price-relative
result is z = +2.72, and its bootstrap 95% ROI interval is [+3.0%, +25.2%].

FOUR REASONS THAT IS DISCOVERY EVIDENCE AND NOT A RESULT. The first two are in
the specification this module implements; the last two were measured while
implementing it and are the ones that decide how it is registered.

  1. The 0.45 threshold was chosen after seeing these rows.
  2. The fade branch holds 15 games.
  3. THE HEADLINE IS MOSTLY NOT THE RULE. +14.1% against the plain lean's
     +10.1% is a +4.0pp improvement, and the rule changes only 15 of 223
     selections -- the other 208 are the v12 model unaltered. Bootstrapped,
     that paired improvement is [-2.6, +10.4] ROI points with P(<= 0) = 0.11.
     The impressive-looking z = +2.72 is inherited from the follow branch
     (z = +2.50), which is the model, not the hybrid. So the REGISTERED
     HEADLINE BELOW IS THE PAIRED SWITCH DELTA, not the hybrid's ROI: the
     latter can only restate what the model already does.
  4. THE FADE BRANCH IS ALWAYS-CHALK, EXACTLY AND BY CONSTRUCTION. Fading a
     lean priced under 0.45 means backing a side priced over 0.55, which is
     always the favourite -- verified, 15 of 15. So the fade branch's 11-4 and
     +23.8% IS the always-chalk record on those 15 rows, to the unit, and
     carries no model content. It landed in a window where chalk beat its own
     price by +4.0pp over all 223 rows (137-86, +4.5% ROI). A rule whose only
     active branch is "back the favourite", measured over 15 games in a
     favourite-friendly stretch, is the trap CLAUDE.md names: a hot base rate
     making a derived rule look justified on the rows in front of you.

WHAT THE SEARCH TEST SAYS, since value_probe's standing rule is that a
threshold found by searching is judged against the null maximum and never
against zero. Sweeping the threshold 0.30..0.56 in 0.01 steps, 0.45 is the
argmax of 27 candidates at +14.1%; simulating outcomes at the devigged closing
prices under "market correct, no edge", the best of those 27 averages +4.7% and
P(null best >= +14.1%) = 0.019. So this one does NOT dissolve into the search
the way the band grid did (p = 0.38). That is the strongest pre-registration
case any rule in this repo has had -- and it is still not a result: the sweep
is flat near +10% below 0.44 because the fade branch is empty there, so the
whole spike is those same 15 games, and a permutation cannot manufacture the
independent sample the gate below asks for.

WHAT IT CANNOT DO.
  * It cannot make the discovery sample count. Rows on or before the
    registration date are excluded by construction, and re-including them to
    "get more power" destroys the only property this module has.
  * It cannot score at the price the protocol specifies. The registered
    protocol calls for the no-vig probability available AT DECISION TIME; the
    ledger carries only the close, because the no-lookahead invariant keeps
    every market column off a pending row. So this scores the CLOSING basis,
    which is the same basis forward_test.py uses and the same basis the
    discovery numbers above were measured on -- consistent, but a CLV reading
    rather than the obtainable-price reading. Closing the gap needs a
    decision-time price persisted pregame; until then an operator paper-
    tracking this rule should record their own obtainable price separately.
  * It cannot separate the rule from always-chalk quickly, or perhaps at all.
    See GATE_SWITCHES.
  * It cannot tell you the model is bad, or good. It tests one derived
    direction rule layered on top of the model, not the model.

    python hybrid_test.py
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. Changing any value below invalidates the test and
# restarts it from zero: the numbers are only meaningful because they were
# fixed before the scored rows existed. test_hybrid_test_registration_frozen
# pins every one of them, so editing one means deliberately editing a test that
# says not to -- which is the point.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-01"      # slates STRICTLY after this date are scored
THRESHOLD = 0.45                  # q >= THRESHOLD follows the lean; below fades
STAKE = 1.0                       # flat
# No delta band, no price band, no upper favourite cutoff. The specification
# freezes exactly one number and this module holds it: the ledger showed no
# stable point at which strong favourites become systematically overvalued, and
# adding bands to a rule this thin is how value_probe's grid search produced a
# +20% cell out of noise.

# The registered headline is the PAIRED SWITCH DELTA -- profit on a switched
# game minus what the plain lean would have returned on that same game -- for
# reason 3 above. Discovery values, recorded so a forward run is read against
# them rather than against zero:
DISCOVERY_SWITCHES = 15           # of 223 eligible v12 rows (6.7%)
DISCOVERY_SWITCH_DELTA = 0.592    # units per switched game
DISCOVERY_SWITCH_SD = 1.884       # per-switch sd, used for the gate below
DISCOVERY_PAIRED_ROI_CI = (-2.6, 10.4)   # bootstrap 95% CI, ROI points
# Switches needed for |z| = 2, at the sd above. Two gates because they answer
# different questions and only the second is the one that matters:
#   * the discovery effect is implausibly large (it is a selected maximum), so
#     41 switches only asks "is it anywhere near as big as it looked";
#   * a plausible real edge of +0.10u per switch needs ~1,420, which at the
#     observed 0.88 switches per slate is ~1,600 slates -- roughly ten seasons.
GATE_SWITCHES = 41
GATE_SWITCHES_REALISTIC = 1420
# Registered prior: NULL on the switch delta. Not negative like forward_test's
# arm 1 -- the null-max test above is genuinely encouraging -- but not positive
# either, because the branch under test is indistinguishable from always-chalk
# and its discovery window favoured chalk by +4.0pp.
PRIOR = "null"

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


REQUIRED_COLUMNS = ("status", "game_date", "close_p_home", "xw_lean", "home",
                    "full_home", "full_away", "close_home_ml", "close_away_ml")


def decidable(led):
    """Graded rows the rule can act on, with no date filter applied.

    A v5 abstention has no lean and is excluded here, which is the rule's own
    "abstain" branch; a row with no two-sided close cannot be scored at all.
    `xw_net` is deliberately NOT required: the hybrid reads a DIRECTION and a
    PRICE, never the delta's magnitude.

    Returns None when the frame is unusable, so a caller can tell "not scored"
    from "scored, nothing qualified".
    """
    if any(c not in getattr(led, "columns", ()) for c in REQUIRED_COLUMNS):
        return None
    return led[(led["status"] == "graded") & led["close_p_home"].notna()
               & led["xw_lean"].notna()].copy()


def apply_rule(g):
    """Attach the rule's columns to a frame from `decidable`. Pure.

    Split out of `scored_rows` so the RETROSPECTIVE reading printed in
    `data/ledger_report.txt` and the FORWARD reading registered here run the
    same arithmetic instead of two copies that can drift. The date filter lives
    in `scored_rows` alone and this function must never learn one: the whole
    value of the registration is that its row set is decided in exactly one
    place, and a filter here could quietly narrow it.
    """
    lean_home = (g["xw_lean"] == g["home"]).values
    home_won = (g["full_home"] > g["full_away"]).values
    g["lean_home"] = lean_home
    g["lean_won"] = np.where(lean_home, home_won, ~home_won)
    g["lean_ml"] = np.where(lean_home, g["close_home_ml"], g["close_away_ml"])
    # q: the market's probability of the side the MODEL selected.
    g["model_side_p"] = np.where(lean_home, g["close_p_home"],
                                 1 - g["close_p_home"])
    # The hard switch. `>=` follows, so exactly THRESHOLD follows the model.
    follow = (g["model_side_p"] >= THRESHOLD).values
    g["follow"] = follow
    g["bet_home"] = np.where(follow, lean_home, ~lean_home)
    g["bet_won"] = np.where(follow, g["lean_won"], ~g["lean_won"].astype(bool))
    g["p_bet"] = np.where(follow, g["model_side_p"], 1 - g["model_side_p"])
    g["ml_bet"] = np.where(g["bet_home"], g["close_home_ml"], g["close_away_ml"])
    g["profit"] = np.where(g["bet_won"], STAKE * _payout(g["ml_bet"]), -STAKE)
    # What the plain lean would have returned on the same game. The registered
    # headline is the difference on switched games only; on a followed game the
    # two are identical by construction and the difference is exactly zero.
    g["lean_profit"] = np.where(g["lean_won"].astype(bool),
                                STAKE * _payout(g["lean_ml"]), -STAKE)
    g["switch_delta"] = g["profit"] - g["lean_profit"]
    # Always-chalk on the same rows. Not decoration: the fade branch is
    # chalk-identical by construction, so this is the control that says whether
    # anything beyond "back the favourite" is happening.
    chalk_home = (g["close_p_home"] >= 0.5).values
    g["chalk_won"] = np.where(chalk_home, home_won, ~home_won)
    g["chalk_p"] = np.where(chalk_home, g["close_p_home"], 1 - g["close_p_home"])
    chalk_ml = np.where(chalk_home, g["close_home_ml"], g["close_away_ml"])
    g["chalk_profit"] = np.where(g["chalk_won"], STAKE * _payout(chalk_ml), -STAKE)
    return g


def scored_rows(led=None):
    """Graded rows AFTER the registration date, with the hybrid rule applied.

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
    return apply_rule(g)


def _excess_z(won, p):
    """Poisson-binomial z of a realised rate against its own implied rate.

    The p_i are the market's, fixed rather than estimated from the outcomes
    under test, so this is defined at n = 1 and never degenerates to +/-0.0 --
    the error-bar rule the calibration surfaces already follow.
    """
    n = len(won)
    if not n:
        return float("nan")
    se = float(np.sqrt(np.sum(p * (1 - p))) / n)
    return (won.mean() - p.mean()) / se if se > 0 else float("nan")


def _line(f, label, won_col="bet_won", p_col="p_bet", profit_col="profit"):
    n = len(f)
    if not n:
        return f"    {label:<22}    no qualifying bets yet"
    won = f[won_col].astype(bool)
    u = float(f[profit_col].sum())
    z = _excess_z(won.values, f[p_col].values)
    return (f"    {label:<22}  n={n:<4d} {int(won.sum())}-{n - int(won.sum())}  "
            f"{u:+7.2f}u  ROI {u / n * 100:+6.1f}%   vs price z={z:+.2f}")


def report_lines(led=None):
    """Report body as a list of lines. Pure -- no printing, no file writes."""
    out = [f"pre-registered hybrid market-direction test  (registered "
           f"{REGISTERED_ON}; follow the lean at q >= {THRESHOLD:.2f}, "
           f"fade below)"]
    g = scored_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    slates = g["game_date"].nunique() if len(g) else 0
    out.append(f"    eligible rows since registration: {len(g)} over {slates} slates")
    if not len(g):
        out.append(f"    nothing to score yet. Prior is {PRIOR.upper()}: the fade "
                   "branch is always-chalk by construction and its discovery "
                   "window favoured chalk by +4.0pp.")
        return out

    sw = g[~g["follow"]]
    n_sw = len(sw)
    # THE REGISTERED HEADLINE. Everything below it is context.
    if n_sw:
        d = sw["switch_delta"].values
        se = float(np.std(d, ddof=1) / np.sqrt(n_sw)) if n_sw > 1 else float("nan")
        out.append(f"    SWITCH DELTA (registered)  n={n_sw:<4d} "
                   f"{d.mean():+.3f}u per switch  +/- {se:.3f}"
                   + (f"   z={d.mean() / se:+.2f}" if se and se > 0 else "")
                   + f"   total {d.sum():+.2f}u")
    else:
        out.append("    SWITCH DELTA (registered)    no switched selections yet")
    out.append(f"    discovery was {DISCOVERY_SWITCH_DELTA:+.3f}u per switch over "
               f"{DISCOVERY_SWITCHES}; paired ROI CI "
               f"[{DISCOVERY_PAIRED_ROI_CI[0]:+.1f}, "
               f"{DISCOVERY_PAIRED_ROI_CI[1]:+.1f}] pp. Read the forward number "
               "against that, not against zero.")

    out.append("")
    out.append(_line(g[g["follow"]], "follow  q>=%.2f" % THRESHOLD))
    out.append(_line(sw, "fade    q< %.2f" % THRESHOLD))
    out.append(_line(g, "combined hybrid"))
    out.append(_line(g, "plain xwOBA lean", won_col="lean_won",
                     p_col="model_side_p", profit_col="lean_profit"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    if n_sw:
        same = int((sw["bet_home"].values == (sw["close_p_home"] >= 0.5).values).sum())
        out.append(f"    the fade branch backed the favourite in {same} of "
                   f"{n_sw} switched games (construction says all of them)")

    out.append(f"    GATE: {n_sw} of ~{GATE_SWITCHES} switches to test the "
               f"discovery-sized effect, ~{GATE_SWITCHES_REALISTIC} for a "
               "plausible +0.10u one (~10 seasons at 0.88 switches a slate).")
    out.append("    The combined-hybrid line is mostly the model, not the rule: "
               "only the switched games are the hypothesis. See hybrid_test.py.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
