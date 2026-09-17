# The pre-registered forward tests

Seven registrations across six modules. This file explains their **logic** —
what each rule does, what number decides it, and why that number rather than
the obvious one.

**This file is a reading aid and is authoritative for nothing.** The frozen
constants live in the modules and are pinned by `tests/`; the live readings
live in `data/ledger_report.txt` and move every build. Deliberately, no forward
number appears below — a figure quoted here would be stale within a day, which
is the failure mode CLAUDE.md records against its own prose more than once.
Read a module's docstring for the argument in full and the report for where a
registration actually stands.

---

## The one idea behind all seven

**A registration's headline is deliberately not the number you would want to
look at.** The obvious number — the rule's ROI, the filtered record, the
combined line — is mostly the *model*, because these rules leave most games
untouched. A hybrid that switches 15 of 223 selections reports 208 rows of
unaltered v12 in its ROI, so a good combined line says the model is good and
says nothing about the rule.

So each registration freezes the **increment**: what the rule changes relative
to doing nothing, measured only on the rows where it acts. Everything else
follows from that choice.

Five properties every one of them has:

1. **The headline is the increment.** A switch delta over switched games, an
   excess over dropped games, a contrast between two halves. Never a combined
   record.
2. **The prior is stated in advance, and it can be negative.** Two of these are
   registered as hypotheses already believed false. A rule whose prior is
   negative cannot be rescued by a hot streak — that is what the prior is for.
3. **The gate is arithmetic, not taste.** Bets needed to separate the claimed
   effect from zero at |z| = 2, at the arm's own observed per-bet sd. Several
   carry a second, much larger gate for a *plausible* effect as opposed to the
   discovery-sized one, because clearing the first gate means only that the
   rule has not disqualified itself.
4. **Controls are scored on the same rows.** Always-chalk especially: a fade
   branch backs the favourite by construction, so without that control a chalk
   result reads as the rule's own skill.
5. **The row selector is a frozen constant too.** Slates strictly after the
   registration date, and nothing refitted at run time. A probe that refits
   per slate carries its own fitting noise into whatever it selects — the
   defect that killed the first arm below.

---

## `forward_test.py` arm 1 — the delta-vs-market gap

*Registered 2026-08-29. Prior: **negative**.*

Fits `logit(p_home) = A + B·xw_net` **once**, on the rows available at
registration, and never refits. Where the model's implied probability differs
from the market's by more than the frozen gap threshold, it bets **against**
the lean.

The frozen coefficients are the entire design. In discovery, fading the largest
gaps looked strong — and the effect was traced to the walk-forward's own
machinery. Refitting each slate on thin early samples produced slopes far above
the stable full-sample value, and an inflated slope inflates the gap on exactly
those slates, so the threshold was selecting **unsettled fits rather than
disagreement**. Holding the coefficients fixed inverts the effect at every
threshold.

That is why the prior is negative: this module tracks a hypothesis already
believed false, so that a good forward run reads as a hypothesis and not as a
discovery. The secondary thresholds print as context and carry no claim.

## `forward_test.py` arm 2 — the plus-money underdog

*Registered 2026-08-29. Prior: **null**.*

The model leans a side priced at plus money. Registered the same day as arm 1
but a separate and better-posed hypothesis: favourite-longshot bias is a
documented market phenomenon, so this had a real prior **before** anyone looked
at this ledger. That is the distinction from arm 1, which was found by
searching.

**Registered deliberately unbanded.** Cutting by price produced a far more
flattering cell, and that band was chosen after seeing it — freezing it would
smuggle the search back into a pre-registration. The a-priori hypothesis is
"the model leans a plus-money dog", so that is what is frozen; the price bands
print as secondary context. A test pins that the headline counts every
plus-money lean rather than the band's subset.

The prior is null rather than positive because the discovery measurement found
**no favourite-longshot bias here to harvest** — dogs in this book are priced
about right, and the model's selection among them was directionally better but
statistically nothing.

## `hybrid_test.py` — hybrid v1, frozen and retired

*Registered 2026-09-01. Prior: **null**. Superseded by v2; kept frozen.*

Follow the lean when the market gives the selected side at least the threshold
probability; below it, back the other side.

**The headline is the paired switch delta per switched game, never the hybrid's
ROI.** Every followed row is the model untouched, so a combined line can only
restate what the model already does. A test pins that a followed game has
*identically* zero switch delta, so the headline cannot absorb the model's own
performance.

**The fade branch is always-chalk, exactly and by construction**: fading a lean
priced below the threshold backs a side priced above its mirror, which is
always the favourite. A second test pins that. This is why the always-chalk
control prints beside it — a rule whose only active branch is favourite-backing,
measured over a handful of games in a favourite-friendly stretch, is the trap
the whole registration exists to avoid.

Two gates, because they answer different questions: the near one asks only
whether the effect is anywhere near as large as it looked; the far one is what
a plausible per-switch edge needs, and is roughly ten seasons. Read nothing
before the first and do not read the second as reachable.

## `hybrid_v2.py` — the live rule

*Registered 2026-09-11. Prior: **null**. Selection layer only — no `MODEL_TAG` bump.*

Fade only when the leaned side's saved no-vig probability is below the
threshold **and** `abs(xw_net)` is below the delta threshold. Follow on either
boundary and everywhere else. Inherits v1's threshold, stake and gates by
import rather than restating them.

**The conjunctive gate fixes a structural defect in v1.** Under v1 the fade was
unconditional, so whenever the favourite was priced above the threshold's
mirror, both branches converged on the favourite: lean the favourite and you
follow to it, lean the dog and you fade to it. The model's opinion could not
change the ticket on a clear majority of games. Requiring low conviction as
well means a dog lean the model holds with any conviction is now followed, and
the counterfactual dead zone falls roughly four-fold — bought by changing very
few tickets, which is also the reason to be careful with it.

Note what is *not* claimed: that reduction says the model's opinion can now
move the ticket on most games. It says nothing about whether moving it helps.
Only the registered forward reading can say that.

One live caveat the module states itself: the rule is specified against the
price available at **decision time**, and the registered scorer uses saved
pregame fields with no closing fallback. The retrospective line is scored
uniformly at closes and is a different basis. Never substitute one for the
other; `hybrid_price_source` records which basis each stored row carries.

## `delta_filter_test.py` — the conviction filter

*Registered 2026-09-03. Prior: **negative**.*

Abstain when `abs(xw_net)` is below the delta threshold; otherwise bet the lean.

**The headline is the excess on the DROPPED games, and the rule is vindicated
only if that number goes NEGATIVE.** A filter's whole content is which rows it
removes, so a filtered ROI could only restate the model. Positive means the
filter is discarding winning bets.

The prior is negative on three measurements, and the decisive one is the chalk
control. On the games the filter drops, the model beats chalk by more than on
the games it keeps. The kept half only looks better because it is
favourite-heavy: low-|Δ| games sit near pick'em, which is precisely where a
model has something to add over backing the favourite. **The filter proposes to
discard the games where the model contributes most.**

It is also the one registration that **failed its own search test** — the best
contrast the real rows offer is *smaller* than what the same sweep returns from
pure noise. That number is frozen in the module beside the others, because a
failed search test is evidence and deserves to be as durable as a passed one.

## `abstain_test.py` — decline instead of fade

*Registered 2026-09-03. Prior: **null**. Decision pre-committed 2026-09-16.*

Back the lean where the q-gate follows it; where the q-gate would fade onto the
opposing side, this one makes **no bet**.

**The declined set is the q-gate's fade set, not the shipped rule's** — a
correction made 2026-09-17. At registration the two were the same games,
because the shipped hybrid then faded whenever `q < .45`. Hybrid v2 shipped on
2026-09-11 with a second gate, and 43% of the q-gate's fades (16 of 37 on the
current family; 2 of 5 forward) are now games the shipped rule FOLLOWS — two of
them visible in the ledger as `hybrid_action=FOLLOW` on rows this module counts
as declined. The selector is deliberately **not** re-pointed: it is the
registered rule's row selection and re-aiming it mid-registration restarts the
test. `declined_but_followed()` prints the split every build instead.

**The headline is fade-minus-abstain per declined game** — what betting those
games earns over not betting them. Positive keeps the shipped fade branch;
negative says decline.

It exists because the two accounts of those games cannot be separated by the
combined record: either the branch is favourite-backing in a favourite-friendly
window, or the model's strong disagreement with a well-priced market is
genuinely anti-signal and opposing it is informative. This module is the
instrument that can eventually tell them apart — and if the forward reading
cannot separate them, that is itself the argument for the simpler rule.

It borrows `hybrid_test`'s row selector so the q-gate cannot be spelled twice —
but deliberately **not** its registration date. Borrowing that wholesale was a
real trap on its first run: it inherited an earlier date and scored two slates
from its own discovery sample as though they were forward rows. The borrow has
now failed in a second way, one level out: what it guaranteed was agreement
with `hybrid_test`, and the module read that as agreement with what ships.
Those stopped being the same thing when v2 shipped, and nothing raised, because
the fixture behind every test of the borrow carried no `xw_net` and so compared
v1 against v1. **A delegated selector pins you to the module you delegate to,
not to production.**

**The decision is pre-committed**, which is the part worth copying. At the
declined-game gate, retire the q-gate fade branch unless the headline is
strictly positive — a point estimate, with no significance requirement. The
asymmetry is deliberate: the prior is null, fading pays vig and publishes an
always-chalk ticket as a model selection while abstaining costs nothing, and
this repo's standing preference is subtractive. A branch must earn its place,
so under a null the simpler rule wins.

The live readings at the moment of freezing are recorded in the module, and
both were already negative. That disclosure is the point — the criterion was
set to a bar the branch was already failing, in the direction the
recommendation already favoured, stated in advance rather than discovered
afterwards. `decision()` returns None below the gate so it cannot fire early,
and nothing in shipping code consults it.

What the criterion can and cannot retire, given the correction above: retiring
the q-gate fade is implementable and v2's branch sits inside it, so the verdict
is actionable. It is not a measurement *of v2's branch* — the extra games are
exactly the higher-conviction ones v2's delta gate was written to keep, so
evidence against fading on the union does not transfer to the subset. Aiming a
decision at v2's branch needs its own registration. The threshold and the gate
are not moved for this; re-aiming a pre-commitment at a number already on the
screen is what the freeze exists to prevent.

## `dog_contrast_test.py` — the underdog sign flip

*Registered 2026-09-03. Prior: **null**. Explicitly NOT independent.*

Among leans on underdogs, the contrast between those priced above the hybrid's
threshold and those below it.

**What it registers is not the segment.** The tempting thing to freeze was the
narrow band just under even money — the best slice the rule has. Two
measurements said no: that band fails its own search test, and it is largely an
existing registration already, being arm 2 with its losing tail removed.
Refining a registered rule after seeing which part worked is exactly what
pre-registration prevents.

So the contrast is registered instead. It uses **both** halves, so removing the
losing half is what it *measures* rather than what it does. Neither split point
was searched: the upper split is `hybrid_test`'s threshold, frozen two days
earlier for an unrelated reason, and the lower bound is the definition of an
underdog. Its own search test comes back better than the band it declined to
register and far better than the delta filter's, but short of the hybrid's and
of any conventional bar: **it does not clear 0.05**, which is why the prior is
null rather than positive.

Its gate is the most reachable of the seven, and deliberately **not** sized to
the discovery effect: a selected maximum reproducing itself over a handful of
slates would prove nothing.

---

## These are not seven independent samples

`forward_test` arm 2, `abstain_test`, and `dog_contrast_test`'s below-split half
all read the same small set of games. The report says so on its own line every
build, and `dog_contrast_test` is registered with that dependence stated in its
header rather than discovered later.

**Three readings of one small set of games are three readings, not three
samples.** A reader tallying the registrations as independent pieces of
evidence is the error this note exists to prevent.

## How to read a forward block

1. **Find the registered headline.** It is the line marked as such, not the
   most prominent number and not the combined record.
2. **Read it against the discovery value printed beside it**, never against
   zero. Every one of these rules was found on rows that already existed; the
   forward question is whether the effect survived, not whether it is positive.
3. **Check the gate before reading anything at all.** Each block prints how far
   it is from the count that could separate its own claimed effect. All seven
   are currently a long way short.
4. **Read the always-chalk control on the same rows.** On any fade branch it is
   the same bet by construction, so a branch beating its price is only
   interesting relative to the control, never on its own.
5. **Do not sum them.** See above.
