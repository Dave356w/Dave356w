#!/usr/bin/env python3
"""Pre-registered forward test: decline the lean instead of fading it.

REGISTERED 2026-09-03. Every parameter below is FROZEN. Nothing is fitted at
run time and only slates STRICTLY AFTER the registration date are scored.

THE RULE. Back the xwOBA lean when its locked no-vig price q >= the registered
threshold; where the q-gate would FADE onto the opposing side, this one makes
NO BET. Flat one unit.

WHICH FADE BRANCH THAT IS, AND WHY THE ANSWER CHANGED (corrected 2026-09-17).
As registered this WAS the shipped rule with its only active branch replaced by
an abstention: on 2026-09-03 the shipped hybrid faded whenever q < 0.45, so the
declined set and the shipped fade set were the same games and the module could
call them interchangeably. Hybrid v2 shipped on 2026-09-11 and fades only when
q < 0.45 AND |xw_net| < 0.012, so the two sets came apart the moment it did.
Measured on the committed ledger at the correction: of the 436 decidable
current-family rows the q-gate fades 37 and the shipped rule fades 21, so 16 of
the q-gate's fades -- 43% -- are games the shipped rule FOLLOWS. Forward the
split is 2 of 5, and both are visible in the ledger itself: 2026-09-08 pk
824714 and 2026-09-11 pk 824631 carry hybrid_action=FOLLOW on LAA and PIT while
this module counts them as games the shipped rule fades onto BOS and CHC.

THE SELECTOR IS NOT RE-POINTED AT v2, deliberately. It is the registered rule's
row selection, and re-pointing it mid-registration would restart the test from
zero -- the one property the module has. So the q-gate set stays, every claim
that it IS the shipped fade set is corrected, and `declined_but_followed()`
measures the drift on every build so it cannot go quiet again. What follows
from that for the pre-committed decision is stated in the decision block, which
was frozen one day before this was noticed and is NOT edited here.

WHY IT EXISTS. Fading a lean priced under 0.45 means backing a side priced
over 0.55, which is always the favourite -- so every faded bet is a favourite
bet. That is a statement about the TICKET and, as first written here, it was
over-read into "the branch carries no model content". It does carry some: the
model chooses WHICH favourites, and the 20 it selects ran +11.48pp against
+5.02pp for the 135 chalk bets above 0.55 it does not select -- a selection
value of +6.45pp at z = +0.55. Not established, not nothing.

So the question this module exists to answer is not "is the branch decorative"
but "is the model's OPPOSITION to a side informative enough to be worth betting
against?" Measured over the 252 decidable v12 rows the lean went 6-14 on the
faded games, and declining them is where the whole improvement over the plain
lean comes from:

    rule                          n     record   excess    profit     ROI
    plain lean                  252   156- 96    +6.73pp   +25.08u   +10.0%
    hybrid, fade the branch     252   164- 88    +8.55pp   +34.29u   +13.6%
    THIS RULE, decline instead  232   150- 82    +8.30pp   +30.79u   +13.3%

Those last two are the same decision on 232 of 252 games. Everything that
separates them happens on the 20 declined ones, which is what this module
scores and nothing else.

THE REGISTERED HEADLINE IS FADE-MINUS-ABSTAIN PER DECLINED GAME -- what
betting those games earns over not betting them. POSITIVE keeps the q-gate
fade; NEGATIVE says decline. Discovery: +0.1754u per declined game over 20,
sd 0.7918, z = +0.99, bootstrap 95% CI [-0.168, +0.503] with P(<= 0) = 0.152.

WHY THE PRIOR IS NULL RATHER THAN POSITIVE, despite that +0.18u. Every faded
bet is a favourite bet and the discovery window favoured favourites: over all
252 rows always-chalk beat its own price by +2.54pp. Twenty games of
favourite-backing in a favourite-friendly stretch is the trap CLAUDE.md names
by name, and the +6.45pp selection value that argues the branch is more than
that is itself only z = +0.55.

WHAT A FORWARD READING WOULD SEPARATE. Two accounts of the same 20 games:

  * the branch is favourite-backing dressed as a model rule, in which case its
    edge is the window's chalk tailwind and fade-minus-abstain decays to zero;
  * the model's strong disagreement with a well-priced market is ANTI-signal,
    so opposing it is informative and fade-minus-abstain stays positive. This
    is the account `hybrid_test`'s prose now carries: in the q < 0.45 band the
    model-leaned side runs -11.5pp against a band that is calibrated at +0.2pp
    over 468 sides, i.e. the model's endorsement makes a dog worse than an
    average dog at the same price.

The second is the more interesting hypothesis and this module is the instrument
that can eventually tell them apart. Readability is a secondary argument, not
the main one: the fade branch's ticket is always a favourite, so a published
combined record mixes a model rule with a market baseline and no reader can
separate them by looking. If the forward reading cannot separate the two --
which the gate below says is the likely outcome -- that is itself the argument
for the simpler rule.

INDEPENDENT CORROBORATION OF THE DIRECTION, 2026-09-17 -- AND WHY IT IS NOT
EVIDENCE FOR THIS REGISTRATION. A separate question (can +EV sides be read off
a model x market band matrix?) produced a walk-forward that happens to bear on
the abstain-versus-fade asymmetry registered here. Over the 33 v12 slates,
fitting only on prior slates, scored at the close on the 410 rows of slates
2..33:

    rule                                  bets      ROI     profit
    decline the cells that lost            274   +9.08%   +24.88u
    bet AGAINST those same cells           410   +1.54%    +6.33u
    plain lean                             410   +8.00%   +32.80u
    shipped hybrid v2                      410   +9.00%   +36.92u

Declining is roughly neutral against the plain lean; opposing is clearly bad.
That is the same SIGN as DECISION_LIVE_FORWARD and DECISION_LIVE_RETROSPECTIVE,
reached by a different route -- worth recording because the pre-committed rule
retires the branch under exactly that sign, so a later reader should be able to
see that the direction had support from outside the registration as well as
inside it.

Deliberately prose and not a frozen constant. Every literal in the block below
is part of the registration and is pinned; this is an outside measurement of a
different rule, and giving it the same shape would invite it being read as a
fifth registered figure.

It corroborates the DIRECTION and nothing else. Four reasons it is not evidence
for this registration:

  * IT IS NOT THIS RULE. Those 136 declined rows are picked by a per-cell
    criterion refitted every slate; this module declines the games the frozen
    q-gate fades. The two sets overlap only partly, and a rule that declines a
    third of the book is not the one registered here.
  * IT IS NOT FORWARD. The window spans the discovery sample, and a selector
    refitted per slate carries its own fitting noise into whatever it picks --
    the defect that killed `forward_test`'s first arm.
  * IT DOES NOT ESTIMATE THE HEADLINE. `fade_minus_abstain` is a paired
    per-declined-game quantity on the registered set. Nothing in that table
    computes it.
  * IT IS THE SAME GAMES AGAIN. These are largely the rows the discovery
    figures were read off. Three readings of one small set of games are three
    readings, not three samples.

Nor is it an argument for declining broadly. The abstain arm LOSES units to the
plain lean (+24.88u against +32.80u), buying about a point of ROI by declining a
third of the book, and it beats the shipped rule on neither axis. What it says
is narrow -- opposing the model's own weak leans is worse than standing aside --
and that is the only part this module needs.

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
import hybrid_v2
from market_backfill import row_supply_line, window_is_closed

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. tests/test_abstain_test.py pins every value.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-03"      # slates STRICTLY after this date are scored
# NOT a second copy of 0.45: it is `hybrid_test`'s own object, so the q-gate
# cannot drift between the two modules -- the "one value, three homes" defect
# that put v10 math under a v9 tag. What it does NOT guarantee any more is that
# the declined set equals the SHIPPED fade set; v2 added a second gate this
# import knows nothing about. See the docstring, and `declined_but_followed`.
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

# --- THE DECISION RULE, PRE-COMMITTED 2026-09-16 AT n = 5 -------------------
# Frozen on the operator's call while the sample was still too small to argue
# about, which is the only moment a criterion is cheap. Deciding at n = 82 with
# the figure already on the screen is how a gate gets re-litigated.
#
# THE RULE: at GATE_DECLINED declined games, retire the q-gate fade branch --
# ship abstention as v3 -- UNLESS `fade_minus_abstain` is strictly POSITIVE.
#
# WHICH BRANCH THAT RETIRES (added 2026-09-17; no constant above or below this
# note is edited). The criterion was frozen on 2026-09-16 naming "the shipped
# fade branch", on the understanding that the declined set was the shipped fade
# set. It is not, and has not been since v2 shipped on 2026-09-11: the shipped
# branch fades a strict SUBSET, 21 of the q-gate's 37 on the current family.
# So what this criterion can retire is the q-gate fade -- fading a sub-.45 lean
# at all -- which remains an implementable action, since v2's branch is inside
# it. What it cannot do is measure v2's branch: the extra games are exactly the
# higher-conviction ones v2's delta gate was written to keep, so evidence
# against fading on the union does not transfer to the subset. A decision aimed
# at v2's branch specifically needs its own registration and its own gate.
# `declined_but_followed` prints the split every build so the reading at the
# gate carries it. The threshold and the gate are NOT moved: re-aiming a
# pre-commitment at a number that was already on the screen is the thing the
# freeze exists to prevent.
#
# A point estimate, with no significance requirement, and that asymmetry is
# deliberate rather than lax. The prior here is NULL, the two arms differ by
# 0.25pp retrospectively, and CLAUDE.md's standing preference is subtractive:
# a branch must EARN its place, so under a null the simpler rule wins. Fading
# also pays vig and publishes an always-chalk ticket as a model selection,
# while abstaining costs nothing and publishes nothing. So the branch survives
# only if the evidence points its way AT ALL; it does not get the benefit of
# an interval that spans zero.
#
# WHAT A POSITIVE READING WOULD AND WOULD NOT MEAN. It would mean the branch
# has not disqualified itself, not that it works: GATE_DECLINED is sized for
# the DISCOVERY-sized effect, and a plausible +0.10u one needs
# GATE_DECLINED_REALISTIC. Clearing the first gate keeps the branch and leaves
# the registration running; it settles nothing on its own.
#
# THE LIVE READING AT THE MOMENT OF FREEZING, recorded so a later reader can
# see the criterion was NOT set to a bar the branch conveniently clears -- it
# was set to one the branch was already failing:
#
#     forward          -0.060u per declined game over n = 5
#     retrospective    -0.022u over n = 34 on the current family
#     discovery        +0.175u over n = 20
#
# Both live readings are negative and both are far too small to act on. That
# is the point: the criterion commits the decision to a sample neither of them
# is, and it commits it in the direction the recommendation already favoured,
# stated in advance rather than discovered afterwards.
DECISION_PRE_COMMITTED_ON = "2026-09-16"
DECISION_AT_N = 5                          # declined games when this was frozen
DECISION_LIVE_FORWARD = -0.060             # u per declined game, n = 5
DECISION_LIVE_RETROSPECTIVE = -0.022       # u per declined game, n = 34
# Retire the branch unless fade-minus-abstain is strictly greater than this.
DECISION_KEEP_THRESHOLD = 0.0

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
    """The registered declined set: rows the q-gate fades. Pure.

    This is NOT the set on which this rule and the shipped hybrid differ. It
    was, at registration; v2's delta gate ended that, and the shipped rule
    follows 43% of these. `declined_but_followed` counts them every build.
    """
    if g is None or not len(g):
        return g
    return g[~g["follow"].astype(bool)]


def kept(g):
    """Rows the q-gate follows, where this rule bets the lean. Pure."""
    if g is None or not len(g):
        return g
    return g[g["follow"].astype(bool)]


def shipped_also_fades(g):
    """Declined rows the SHIPPED rule fades too. Pure, and NOT registered.

    `hybrid_v2.follows` is called rather than restated, for the reason
    `THRESHOLD` is imported rather than copied: a second spelling of the gate
    would let this diagnostic and the live rule disagree about the very thing
    it exists to measure.
    """
    d = declined(g)
    if d is None or not len(d) or "xw_net" not in d.columns:
        return None          # unanswerable, never silently "no drift"
    return d[~hybrid_v2.follows(d["model_side_p"], d["xw_net"])]


def declined_but_followed(g):
    """Declined rows the SHIPPED rule FOLLOWS -- the registered set's drift.

    Empty until 2026-09-11 and non-empty since, by construction: v2 kept the
    q-gate and added `|xw_net| >= DELTA_THRESHOLD` as a reason to follow, so
    every row in here is one the site published as a FOLLOW while this module
    scored it as a game the rule declines. That is a real disagreement between
    an artifact and a registration, not a rounding difference, which is why it
    is counted from the rows rather than asserted in prose.
    """
    d = declined(g)
    if d is None or not len(d) or "xw_net" not in d.columns:
        return None          # unanswerable, never silently "no drift"
    return d[hybrid_v2.follows(d["model_side_p"], d["xw_net"])]


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


def decision(g):
    """The pre-committed verdict, or None before the gate.

    Returns ``(verdict, n_declined, mean)`` where verdict is ``"RETIRE"`` or
    ``"KEEP"``. None until `GATE_DECLINED` declined games exist, so the rule
    cannot fire early -- a criterion that can be read at n = 5 is not a
    pre-commitment, it is a running commentary, and this module's own first
    trap was scoring rows that belonged to its discovery sample.

    Deliberately NOT consulted by any shipping code. It decides nothing on its
    own; it records in advance what the operator said the number would mean,
    so the reading at the gate is an outcome rather than an argument.
    """
    m, _se, n_d = fade_minus_abstain(g)
    if n_d < GATE_DECLINED:
        return None
    return ("KEEP" if m > DECISION_KEEP_THRESHOLD else "RETIRE"), n_d, m


def _drift_lines(g):
    """How far the registered declined set has drifted from the shipped rule.

    Printed under the headline rather than left to a reader who knows the
    history, because the headline's name for its own row set -- games the rule
    declines instead of fading -- silently stopped matching what the site does
    on 2026-09-11 and nothing in this report said so for six days.
    """
    d, f = declined(g), declined_but_followed(g)
    n_d = 0 if d is None else len(d)
    if not n_d:
        return []
    if f is None:
        # No `xw_net`, so v2's second gate cannot be evaluated. Say that
        # rather than printing "all shared", which would assert the very
        # thing this block exists to stop being assumed.
        return ["      declined set vs the SHIPPED rule: not computable on "
                "this frame (no xw_net); do not read it as agreement."]
    n_f = len(f)
    if not n_f:
        return ["      declined set vs the SHIPPED rule: all "
                f"{n_d} are games hybrid v2 fades too."]
    out = [f"      declined set vs the SHIPPED rule: {n_f} of {n_d} are games "
           "hybrid v2 FOLLOWS (q < .45 but |xw_net| >= .012),",
           "        so the registered quantity is the q-gate fade, not v2's "
           "branch, which fades a strict subset. Not a fixable"]
    m, se, _ = fade_minus_abstain(g)
    sub_rows = shipped_also_fades(g)
    n_s = 0 if sub_rows is None else len(sub_rows)
    if n_s:
        v = sub_rows["profit"].to_numpy(dtype=float)
        out.append("        mismatch mid-registration -- re-pointing the "
                   "selector restarts the test. UNREGISTERED context, on the "
                   f"{n_s} shared rows only:")
        out.append(f"        fade-minus-abstain {v.mean():+.3f}u over {n_s} "
                   f"against the registered {m:+.3f}u over {n_d}. Carries no "
                   "decision; the gate is the registered set's.")
    else:
        out.append("        mismatch mid-registration -- re-pointing the "
                   "selector restarts the test. No shared rows yet.")
    return out


def shipped_rule_retired(led=None):
    """Has the branch this decision would retire already been retired?

    Derived through `market_backfill.window_is_closed` -- the same derivation
    `hybrid_v2.window_closed` uses, reused rather than spelled a second time --
    because the honest answer is a property of what the build stamps, not of a
    date literal that would go wrong if the rule were ever re-shipped.

    Why it matters: the pre-commitment below says RETIRE THE Q-GATE FADE at the
    gate. v13 retired the entire hybrid from the shipped selection on
    2026-09-18, so the action it authorises is already taken. That does not
    void the registration -- the question "would declining those games have
    beaten fading them" is still a real one and still accruing -- but a reader
    reaching the gate must not think a decision is pending on a live branch.
    """
    if led is None:
        led = hybrid_test._ledger(None)
    if led is None:
        return None
    closed, _observed = window_is_closed(led, "selection_rule_tag",
                                         (hybrid_v2.RULE_TAG,))
    return closed


def decision_lines(g, led=None):
    """The pre-commitment, printed every build whether or not it can fire."""
    m, _se, n_d = fade_minus_abstain(g)
    out = [f"    PRE-COMMITTED {DECISION_PRE_COMMITTED_ON} (at n={DECISION_AT_N}): "
           f"at {GATE_DECLINED} declined games, RETIRE the q-gate fade branch "
           f"unless this is > {DECISION_KEEP_THRESHOLD:+.1f}."]
    d = decision(g)
    if d is None:
        out.append(f"      not at the gate: {n_d} of {GATE_DECLINED} declined "
                   "games. No verdict, and the number above is not one.")
    else:
        verdict, n_at, m_at = d
        out.append(f"      GATE REACHED at n={n_at}: verdict {verdict} "
                   f"({m_at:+.3f}u per declined game). A KEEP means the branch "
                   f"has not disqualified itself, not that it works — "
                   f"{GATE_DECLINED_REALISTIC} is the gate for a plausible "
                   "+0.10u effect.")
    if shipped_rule_retired(led):
        out.append("      ACTION ALREADY TAKEN: v13 retired the whole hybrid "
                   "from the shipped selection, so no q-gate fade branch is "
                   "live to retire.")
        out.append("      The registration still measures something real -- "
                   "whether declining those games would have beaten fading "
                   "them -- but it is now a question about a rule nothing "
                   "runs, and a RETIRE verdict at the gate ratifies a "
                   "retirement rather than causing one.")
    return out


def report_lines(led=None):
    """Report body as a list of lines. Pure -- no printing, no file writes."""
    out = [f"pre-registered abstain-vs-fade test  (registered {REGISTERED_ON}; "
           f"decline the lean below q = {THRESHOLD:.2f} instead of fading it)",
           "    price source — saved pregame eligibility/selection, probability, "
           "and return fields inherited from hybrid_test.",
           "                   No closing fallback; missing locked fields are "
           "excluded and counted there as unscorable."]
    g = scored_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    out.append(row_supply_line(g))
    if not len(g):
        out.append(f"    nothing to score yet. Prior is {PRIOR.upper()}: every "
                   "faded bet is a favourite bet and the discovery window "
                   f"favoured favourites by "
                   f"+{DISCOVERY_CHALK_EXCESS_PP:.2f}pp, but the model picks "
                   "which favourites (+6.45pp, z=+0.55). Both accounts fit.")
        out.append(f"    GATE: 0 of ~{GATE_DECLINED} declined games "
                   f"(~{GATE_DECLINED / 1.05:.0f} slates at "
                   f"{DISCOVERY_DECLINE_RATE:.2f} of a slate).")
        out.extend(decision_lines(g, led))
        return out

    m, se, n_d = fade_minus_abstain(g)
    if n_d:
        out.append(f"    FADE MINUS ABSTAIN (registered)  n={n_d:<4d} "
                   f"{m:+.3f}u per declined game +/- {se:.3f}"
                   + (f"   z={m / se:+.2f}" if se and se > 0 else ""))
        out.append("      positive keeps the q-gate fade branch; negative says "
                   "decline. Discovery "
                   f"{DISCOVERY_FADE_MINUS_ABSTAIN:+.3f}u over "
                   f"{DISCOVERY_DECLINED}, CI [{DISCOVERY_CI[0]:+.3f}, "
                   f"{DISCOVERY_CI[1]:+.3f}].")
    else:
        out.append("    FADE MINUS ABSTAIN (registered)    no declined games yet")
    out.extend(_drift_lines(g))
    out.extend(decision_lines(g, led))

    out.append("")
    out.append(_line(kept(g), "this rule (declined out)"))
    out.append(_line(g, "q-gate hybrid (fades)"))
    out.append(_line(declined(g), "the declined games only"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    # Reprice the exact saved-pregame chalk selections at the close. This is a
    # reconciliation only: eligibility, selection, and W-L remain fixed, and
    # the registered test continues to use its saved pregame price.
    if {"close_home_ml", "close_away_ml"} <= set(g.columns):
        chalk_home = (g["pregame_p_home"] >= 0.5).to_numpy()
        close_ml = np.where(chalk_home, g["close_home_ml"], g["close_away_ml"])
        comparable = np.isfinite(pd.to_numeric(close_ml, errors="coerce"))
    else:
        comparable = np.zeros(len(g), dtype=bool)
    if comparable.any():
        won = g["chalk_won"].astype(bool).to_numpy()[comparable]
        saved_u = float(g.loc[g.index[comparable], "chalk_profit"].sum())
        close_u = float(np.where(
            won, STAKE * _payout(np.asarray(close_ml)[comparable]), -STAKE).sum())
        out.append(f"    price-snapshot reconciliation — same saved-pregame "
                   f"chalk selections: n={int(comparable.sum())}, "
                   f"{int(won.sum())}-{int(comparable.sum() - won.sum())};")
        out.append(f"      saved pregame {saved_u:+.4f}u; closing {close_u:+.4f}u. "
                   "The record is identical but the ML payout changed;")
        out.append("      selections are not recalculated at the close.")
    if n_d:
        d = declined(g)
        same = int((d["bet_home"].values
                    == (d["pregame_p_home"].values >= 0.5)).sum())
        out.append(f"    the faded games backed the favourite in {same} of "
                   f"{n_d} (construction says all of them -- read the line "
                   "above against the always-chalk control, not instead of it)")
    out.append(f"    GATE: {n_d} of ~{GATE_DECLINED} declined games for the "
               f"discovery-sized effect, ~{GATE_DECLINED_REALISTIC} for a "
               "plausible +0.10u one. See abstain_test.py.")
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
