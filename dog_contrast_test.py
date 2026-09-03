#!/usr/bin/env python3
"""Pre-registered forward test: does the model's signal FLIP SIGN on underdogs?

REGISTERED 2026-09-03. Every parameter below is FROZEN. Only slates STRICTLY
AFTER the registration date are scored. Tracked separately from
forward_test.py, hybrid_test.py, delta_filter_test.py and abstain_test.py.

THE HYPOTHESIS, which is the operator's and not this analysis's. xwOBA and the
market normally agree -- the lean is the market favourite on 73.4% of games.
Where they DIVERGE, the market has already priced whatever the model is
reading, so the model's continued insistence is anti-signal rather than news.
That predicts a SIGN FLIP: among the games where the model likes an underdog,
its endorsement should be worth something while the disagreement is mild and
worth less than nothing once it is severe.

THE REGISTERED QUANTITY IS THE CONTRAST, not either half:

    among dog leans (q < 0.50), split at the hybrid's own threshold
      above  0.45 <= q < 0.50   n=47   +16.11pp +/-  7.28   ROI +31.1%
      below         q <  0.45   n=20   -11.48pp +/- 11.00   ROI -28.6%
      CONTRAST                         +27.59pp +/- 13.20   z = +2.09

WHY THE CONTRAST AND NOT THE BAND. The 0.45-0.50 band on its own is the
flattering number and the obvious thing to register. It is also
  (a) 34 of its 47 rows shared with `forward_test`'s arm 2 -- the band is
      essentially arm 2 with its losing tail removed (arm 2 all: +5.02pp;
      inside this band: +14.72pp; outside it: -11.48pp), and refining an
      already-registered rule after seeing which part of it worked is what
      pre-registration exists to prevent. CLAUDE.md records that exact call
      being made on 2026-08-29, when arm 2 was registered DELIBERATELY
      UNBANDED for this reason; and
  (b) unable to pass its own search test: over the 180 contiguous price bands
      actually searched, noise returns +12.77pp on average and clears the
      band's +16.11pp 28% of the time (P = 0.2805).
The contrast has neither problem. It uses BOTH halves, so it cannot be "arm 2
with the bad part removed" -- removing the losing half is what it measures.
And its split point is not searched: 0.45 is `hybrid_test`'s threshold,
registered two days earlier for an unrelated reason, and 0.50 is the
definition of an underdog.

WHAT ITS OWN SEARCH TEST SAYS. Sweeping the split 0.40..0.48 within dog leans
and simulating at the devigged closes under "market correct, no edge", the
null best averages +10.54pp against the observed +27.59pp, giving
P(null best >= observed) = 0.0707. Notably the sweep's argmax IS 0.45, so
fixing the split a priori costs nothing here. That is a far better showing
than the band alone (0.2805) or `delta_filter_test` (0.6930), and still short
of `hybrid_test` (0.0190) and of any conventional bar. It does not clear 0.05.

A CONTROL THE HYPOTHESIS HAS TO SURVIVE, and does. The market is calibrated in
every price band -- within 1.2pp over 1592 sides of every graded family -- so
the below-threshold half cannot be the favourite-longshot bias. `value_probe`
looked for that separately and found none in this book. What is left when the
market is calibrated and the model-leaned subset is not is a statement about
the MODEL.

WHY THE PRIOR IS NULL. z = +2.09 against zero, P = 0.0707 against the search,
n = 20 in the half carrying the negative sign, and the whole thing was found
by slicing a price axis. Any one of those is enough to withhold a positive
prior. What it is not is the delta filter, whose search test came back worse
than chance.

WHAT IT CANNOT DO.
  * It cannot make the discovery sample count.
  * It cannot be read as independent of `forward_test` arm 2 or of
    `abstain_test`. Its below-threshold half is exactly the 20 games those
    modules already touch, from a different angle -- arm 2 bets them, this
    measures them, `abstain_test` asks whether to fade them. Three readings of
    one small set of games is three readings, not three samples.
  * It cannot tell you the model is bad, or good. A sign flip in one region
    says where the model's information stops, not whether it has any.

    python dog_contrast_test.py
"""
import numpy as np
import pandas as pd

import hybrid_test

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. tests/test_dog_contrast_test.py pins every value.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-03"      # slates STRICTLY after this date are scored
# NOT a second copy of 0.45. The split has to be the shipped rule's own
# boundary or the "a priori split point" claim above stops being true.
SPLIT = hybrid_test.THRESHOLD
# The definition of an underdog, not a searched bound.
DOG_MAX = 0.50
RULE_TAG = "xwoba_dog_contrast_v1"

# Discovery values, measured 2026-09-03 on the 252 decidable v12 rows over 19
# slates, scored at the close. Percentage points of excess vs the devigged
# close.
DISCOVERY_ABOVE_N = 47
DISCOVERY_ABOVE_EXCESS = 16.11
DISCOVERY_ABOVE_SE = 7.28
DISCOVERY_BELOW_N = 20
DISCOVERY_BELOW_EXCESS = -11.48
DISCOVERY_BELOW_SE = 11.00
DISCOVERY_CONTRAST = 27.59        # the registered headline
DISCOVERY_CONTRAST_SE = 13.20     # ... i.e. z = +2.09
# P(null best >= observed) for THIS hypothesis, sweeping the split 0.40..0.48.
NULL_MAX_P = 0.0707
# The same test for the 0.45-0.50 band ALONE, over the 180 price bands actually
# searched. Kept frozen beside it because the band is what a reader will want
# to quote, and this is the number that says not to.
BAND_NULL_MAX_P = 0.2805
# Rows the band shares with forward_test's arm 2, of the band's 47. This is
# why the band is not registered on its own.
ARM2_OVERLAP = (34, 47)
# Dog leans for |z| = 2: the first on a 15pp contrast, the second on a
# plausible 10pp one. At 3.53 dog leans a slate that is ~25 and ~56 slates --
# by far the most reachable gate of the four registrations. A 27.6pp effect
# would show at ~26, which is why that one is NOT the gate: a selected maximum
# reproducing itself over seven slates would prove nothing.
GATE_DOG_LEANS = 88
GATE_DOG_LEANS_REALISTIC = 198
DISCOVERY_DOG_LEANS_PER_SLATE = 3.53
PRIOR = "null"

LEDGER = hybrid_test.LEDGER


def scored_rows(led=None):
    """Forward dog leans: hybrid_test's row shape, this module's date bound.

    Delegates eligibility and the locked pregame market -- the split is on a
    price, so it must be the price obtainable at decision time, never the
    close. The DATE BOUND is not delegated: `hybrid_test` is registered two
    days earlier, and inheriting its bound would score slates that are part of
    this module's discovery sample. That bug shipped once in `abstain_test`
    and is the reason this function exists rather than a bare delegation.
    """
    g = hybrid_test.scored_rows(led)
    if g is None or not len(g):
        return g
    g = g[g["game_date"].astype(str) > REGISTERED_ON].copy()
    if not len(g):
        return g
    # Dog leans only. `model_side_p` is the locked price of the side the MODEL
    # picked, which is the quantity the hypothesis is about -- not the price of
    # the side the hybrid ends up backing, which on a faded row is the other
    # one.
    return g[g["model_side_p"].astype(float) < DOG_MAX].copy()


def above(g):
    """Dog leans the market prices at or above the split. Pure."""
    if g is None or not len(g):
        return g
    return g[g["model_side_p"].astype(float) >= SPLIT]


def below(g):
    """Dog leans the market prices below the split. Pure."""
    if g is None or not len(g):
        return g
    return g[g["model_side_p"].astype(float) < SPLIT]


def _excess(g):
    """(excess, se, n) of the LEAN against its own locked price.

    The lean, never the hybrid's selection: on a below-split row the shipped
    rule fades onto the other side, and scoring that would measure the rule
    rather than the model's signal, which is what this tests.
    """
    n = 0 if g is None else len(g)
    if not n:
        return float("nan"), float("nan"), 0
    won = g["lean_won"].astype(bool).to_numpy()
    p = g["model_side_p"].astype(float).to_numpy()
    se = float(np.sqrt(np.sum(p * (1 - p)))) / n
    return float(won.mean() - p.mean()), se, n


def contrast(g):
    """(contrast, se, n_above, n_below) -- the registered headline.

    above-minus-below in percentage points of excess vs price, with the SE of
    the DIFFERENCE. Both halves are required: a contrast computed from one is
    the band, and the band is what this module exists not to register.
    """
    ea, sa, na = _excess(above(g))
    eb, sb, nb = _excess(below(g))
    if not na or not nb:
        return float("nan"), float("nan"), na, nb
    return ea - eb, float(np.hypot(sa, sb)), na, nb


def _line(g, label):
    e, se, n = _excess(g)
    if not n:
        return f"    {label:<26}    no qualifying games yet"
    won = g["lean_won"].astype(bool)
    u = float(g["lean_profit"].sum()) if "lean_profit" in g.columns else float("nan")
    tail = f"  {u:+7.2f}u  ROI {u / n * 100:+6.1f}%" if np.isfinite(u) else ""
    return (f"    {label:<26}  n={n:<4d} {int(won.sum())}-{n - int(won.sum())}  "
            f"{100 * e:+6.2f}pp +/- {100 * se:5.2f}{tail}")


def report_lines(led=None):
    """Report body as a list of lines. Pure -- no printing, no file writes."""
    out = [f"pre-registered underdog sign-flip test  (registered "
           f"{REGISTERED_ON}; dog leans split at q = {SPLIT:.2f})"]
    g = scored_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    slates = g["game_date"].nunique() if len(g) else 0
    out.append(f"    dog leans since registration: {len(g)} over {slates} slates")
    if not len(g):
        out.append(f"    nothing to score yet. Prior is {PRIOR.upper()}: found by "
                   f"slicing a price axis, and P(null >= observed) = "
                   f"{NULL_MAX_P:.4f} against the split sweep.")
        out.append(f"    GATE: 0 of ~{GATE_DOG_LEANS} dog leans "
                   f"(~{GATE_DOG_LEANS / DISCOVERY_DOG_LEANS_PER_SLATE:.0f} "
                   "slates). Read nothing before then.")
        return out

    c, se, na, nb = contrast(g)
    if na and nb:
        out.append(f"    CONTRAST above-minus-below (registered)  "
                   f"{100 * c:+.2f}pp +/- {100 * se:.2f}"
                   + (f"   z={c / se:+.2f}" if se and se > 0 else "")
                   + f"   (n={na} above, {nb} below)")
    else:
        out.append(f"    CONTRAST above-minus-below (registered)    "
                   f"needs both halves; have {na} above, {nb} below")
    out.append(f"    discovery {DISCOVERY_CONTRAST:+.2f}pp +/- "
               f"{DISCOVERY_CONTRAST_SE:.2f}. Read the forward number against "
               "that, not against zero.")
    out.append("")
    out.append(_line(above(g), "above  q>=%.2f" % SPLIT))
    out.append(_line(below(g), "below  q< %.2f" % SPLIT))
    out.append(_line(g, "all dog leans"))
    out.append(f"    NOT independent of forward_test arm 2 "
               f"({ARM2_OVERLAP[0]} of {ARM2_OVERLAP[1]} discovery rows shared) "
               "or of abstain_test, which touches the same below-split games.")
    out.append(f"    GATE: {len(g)} of ~{GATE_DOG_LEANS} dog leans, "
               f"~{GATE_DOG_LEANS_REALISTIC} for a 10pp contrast. The band "
               f"alone scored P={BAND_NULL_MAX_P:.4f} on its own search test "
               "and is deliberately NOT the registered quantity.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
