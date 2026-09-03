#!/usr/bin/env python3
"""Pre-registered forward test: decline the lean instead of fading it.

REGISTERED 2026-09-03. Every parameter below is FROZEN. Nothing is fitted at
run time and only slates STRICTLY AFTER the registration date are scored.

THE RULE. Identical to the published hybrid on the FOLLOW side -- back the
xwOBA lean when its locked no-vig price q >= the registered threshold -- but
where the hybrid FADES onto the opposing side, this one makes NO BET. Flat one
unit. It is the shipped rule with its only active branch replaced by an
abstention.

WHY IT EXISTS. `hybrid_test`'s own reading is that the fade branch is
always-chalk exactly and by construction: fading a lean priced under 0.45
means backing a side priced over 0.55, which is always the favourite. So the
branch carries no model content, and the honest description of what the shipped
rule adds over the plain lean is NOT "conviction-conditional deference to the
market" but "do not back the model's pick when the market prices it under 45%"
-- a no-big-underdogs filter. Measured over the 252 decidable v12 rows, the
lean went 6-14 on the games the rule fades, and declining them is where the
whole improvement over the plain lean comes from:

    rule                          n     record   excess    profit     ROI
    plain lean                  252   156- 96    +6.73pp   +25.08u   +10.0%
    hybrid, fade the branch     252   164- 88    +8.55pp   +34.29u   +13.6%
    THIS RULE, decline instead  232   150- 82    +8.30pp   +30.79u   +13.3%

Those last two are the same decision on 232 of 252 games. Everything that
separates them happens on the 20 declined ones, which is what this module
scores and nothing else.

THE REGISTERED HEADLINE IS FADE-MINUS-ABSTAIN PER DECLINED GAME -- what
betting those games earns over not betting them. POSITIVE keeps the shipped
rule; NEGATIVE says decline. Discovery: +0.1754u per declined game over 20,
sd 0.7918, z = +0.99, bootstrap 95% CI [-0.168, +0.503] with P(<= 0) = 0.152.

WHY THE PRIOR IS NULL RATHER THAN POSITIVE, despite that +0.18u. The branch is
always-chalk, and its discovery window favoured chalk: over all 252 rows
always-chalk beat its own price by +2.54pp. A branch whose entire content is
"back the favourite", measured across 20 games in a favourite-friendly stretch,
is the trap CLAUDE.md names by name. The 14-6 is the chalk record on those
rows, to the unit, not a skill result.

WHY IT IS WORTH REGISTERING AT ALL, given the two rules are within 0.25pp:
because they are NOT equally readable. The fade branch is numerically identical
to always-chalk, so the published record mixes a model rule with a market
baseline and no reader can separate them by looking. Declining makes the record
a statement about the model alone. If the forward reading cannot separate the
two -- which the gate below says is the likely outcome -- that is itself the
argument for the simpler rule, not a reason to shrug.

WHAT IT CANNOT DO.
  * It cannot make the discovery sample count. Rows on or before the
    registration date are excluded by construction.
  * It cannot test the hybrid's FOLLOW branch. That is `hybrid_test`'s job and
    this module shares its eligibility and its follow/fade split rather than
    re-deriving them -- but NOT its registration date, which is two days
    earlier. See `scored_rows`.
  * It cannot tell you the model is bad, or good.

    python abstain_test.py
"""
import numpy as np
import pandas as pd

import hybrid_test

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. tests/test_abstain_test.py pins every value.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-03"      # slates STRICTLY after this date are scored
# NOT a second copy of 0.45. The declined set must be exactly the set the
# shipped rule fades, or this stops being a comparison and becomes a different
# rule -- the "one value, three homes" defect that put v10 math under a v9 tag.
THRESHOLD = hybrid_test.THRESHOLD
STAKE = 1.0
RULE_TAG = "xwoba_market_abstain_v1"

# Discovery values, measured 2026-09-03 on the 252 decidable v12 rows over 19
# slates, scored at the close. Units per declined game unless noted.
DISCOVERY_DECLINED = 20                    # of 252 decidable rows (7.9%)
DISCOVERY_DECLINE_RATE = 0.079
DISCOVERY_FADE_MINUS_ABSTAIN = 0.1754      # the registered headline
DISCOVERY_SD = 0.7918
DISCOVERY_CI = (-0.168, 0.503)             # bootstrap 95%, P(<= 0) = 0.152
DISCOVERY_LEAN_RECORD_ON_DECLINED = (6, 14)   # why declining helps at all
# Chalk beat its own price by this much over all 252 discovery rows. The fade
# branch IS chalk, so this is the tailwind its +0.18u was measured in.
DISCOVERY_CHALK_EXCESS_PP = 2.54
# Declined games for |z| = 2 at the observed sd: the first on the
# discovery-sized effect, the second on a plausible +0.10u one. At 1.05
# declined games a slate that is ~78 and ~239 slates.
GATE_DECLINED = 82
GATE_DECLINED_REALISTIC = 251
PRIOR = "null"

LEDGER = hybrid_test.LEDGER


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def scored_rows(led=None):
    """Forward rows: hybrid_test's row SHAPE, this module's own date bound.

    The eligibility and the follow/fade split are delegated, deliberately. The
    two rules are the same decision on every followed game and differ only on
    the declined ones, so they must agree about which games those are; deriving
    that twice is how they would come to disagree. Delegating also inherits the
    locked pregame market, which this rule needs -- it reads a price to decide,
    so it must be scored at the price obtainable when it decided, never at the
    close.

    THE DATE BOUND IS NOT DELEGATED, and that is the whole of this function.
    `hybrid_test` is registered 2026-09-01 and this module 2026-09-03, so
    handing its frame straight back would have scored two slates that are part
    of THIS registration's discovery sample -- a forward test grading its own
    search, silently, because both numbers look like forward rows. Caught on
    the first run: the delegated frame returned 14 rows over 1 slate on a day
    when this registration should have had none.
    """
    g = hybrid_test.scored_rows(led)
    if g is None or not len(g):
        return g
    # Strictly after, for the same reason hybrid_test uses `>`: the
    # registration date itself already held graded rows.
    return g[g["game_date"].astype(str) > REGISTERED_ON].copy()


def declined(g):
    """The rows where this rule and the shipped hybrid differ. Pure."""
    if g is None or not len(g):
        return g
    return g[~g["follow"].astype(bool)]


def kept(g):
    """The rows where the two rules are the same bet. Pure."""
    if g is None or not len(g):
        return g
    return g[g["follow"].astype(bool)]


def fade_minus_abstain(g):
    """(mean, se, n) of what fading earns over declining, per declined game.

    Abstaining stakes nothing and returns nothing, so the paired difference on
    a declined game is exactly the fade branch's profit. On a FOLLOWED game it
    is identically zero -- the two rules place the same bet -- which is why
    only the declined rows are the hypothesis and the followed ones cannot
    absorb the model's own performance into this number.
    """
    d = declined(g)
    n = 0 if d is None else len(d)
    if not n:
        return float("nan"), float("nan"), 0
    v = d["profit"].to_numpy(dtype=float)
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return float(v.mean()), se, n


def _excess(won, p):
    """(excess, se) against the market's own probabilities. Defined at n=1."""
    n = len(won)
    if not n:
        return float("nan"), float("nan")
    se = float(np.sqrt(np.sum(p * (1 - p)))) / n
    return float(won.mean() - p.mean()), se


def _line(f, label, won_col="bet_won", p_col="p_bet", profit_col="profit"):
    n = 0 if f is None else len(f)
    if not n:
        return f"    {label:<26}    no qualifying games yet"
    won = f[won_col].astype(bool).values
    e, se = _excess(won, f[p_col].values.astype(float))
    u = float(f[profit_col].sum())
    return (f"    {label:<26}  n={n:<4d} {int(won.sum())}-{n - int(won.sum())}  "
            f"{100 * e:+5.1f}pp +/- {100 * se:4.1f}  {u:+7.2f}u  "
            f"ROI {u / n * 100:+6.1f}%")


def report_lines(led=None):
    """Report body as a list of lines. Pure -- no printing, no file writes."""
    out = [f"pre-registered abstain-vs-fade test  (registered {REGISTERED_ON}; "
           f"decline the lean below q = {THRESHOLD:.2f} instead of fading it)"]
    g = scored_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    slates = g["game_date"].nunique() if len(g) else 0
    out.append(f"    eligible rows since registration: {len(g)} over {slates} slates")
    if not len(g):
        out.append(f"    nothing to score yet. Prior is {PRIOR.upper()}: the branch "
                   "under test is always-chalk by construction, and its "
                   f"discovery window favoured chalk by "
                   f"+{DISCOVERY_CHALK_EXCESS_PP:.2f}pp.")
        out.append(f"    GATE: 0 of ~{GATE_DECLINED} declined games "
                   f"(~{GATE_DECLINED / 1.05:.0f} slates at "
                   f"{DISCOVERY_DECLINE_RATE:.2f} of a slate).")
        return out

    m, se, n_d = fade_minus_abstain(g)
    if n_d:
        out.append(f"    FADE MINUS ABSTAIN (registered)  n={n_d:<4d} "
                   f"{m:+.3f}u per declined game +/- {se:.3f}"
                   + (f"   z={m / se:+.2f}" if se and se > 0 else ""))
        out.append("      positive keeps the shipped fade branch; negative says "
                   "decline. Discovery "
                   f"{DISCOVERY_FADE_MINUS_ABSTAIN:+.3f}u over "
                   f"{DISCOVERY_DECLINED}, CI [{DISCOVERY_CI[0]:+.3f}, "
                   f"{DISCOVERY_CI[1]:+.3f}].")
    else:
        out.append("    FADE MINUS ABSTAIN (registered)    no declined games yet")

    out.append("")
    out.append(_line(kept(g), "this rule (declined out)"))
    out.append(_line(g, "shipped hybrid (fades)"))
    out.append(_line(declined(g), "the declined games only"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    if n_d:
        d = declined(g)
        same = int((d["bet_home"].values
                    == (d["pregame_p_home"].values >= 0.5)).sum())
        out.append(f"    the faded games backed the favourite in {same} of "
                   f"{n_d} (construction says all of them, so the line above "
                   "is chalk's record, not the model's)")
    out.append(f"    GATE: {n_d} of ~{GATE_DECLINED} declined games for the "
               f"discovery-sized effect, ~{GATE_DECLINED_REALISTIC} for a "
               "plausible +0.10u one. See abstain_test.py.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
