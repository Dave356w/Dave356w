# Pitch-mix matchup — design notes

Design argument for a pitch-type-conditioned replacement for the batter side of
the starter phase. **Nothing here is shipped.** The implementation that exists
is `pitch_arsenal.py`, a shadow arm that is off by default (`PITCH_MIX_SHADOW=1`)
and never moves a lean; `pitch_arsenal_probe.py` is the measurement that gates
it. See `MATCHUP_SITE.md` §"Pitch-mix shadow arm" for what is actually built and
why it is dark.

These notes were previously pasted into `docs/build_logic_validation.md`, where
they did not belong — that document validates shipped build logic. The LaTeX has
been rewritten in the plain notation the rest of the repo uses (`M = B·P/L`),
because the original paste rendered its formula blocks as markdown headings.

**Framing correction carried over from the original note:** the theory is
stronger than the current one-sided arm, but it should be modelled as a
pitch-type-conditioned *replacement* for aggregate wOBA — not as another
multiplier layered on top of aggregate batter and pitcher wOBA. The shipped
shadow arm is a multiplier, which is the weaker form.

## What the model is actually estimating

For batter `b`, pitcher `p`, and pitch type `t`, decompose observed wOBA as

```
g(x̂_bpt) = g(μ_t) + a_b + d_p + u_bt + v_pt
```

| term | meaning |
|---|---|
| `μ_t` | league wOBA when a PA ends on pitch type `t` |
| `a_b` | batter's general ability |
| `d_p` | pitcher's general run-prevention ability |
| `u_bt` | batter's pitch-type-specific residual |
| `v_pt` | pitcher's pitch-type-specific residual |
| `g` | a calibrated link, probably log-relative rather than raw addition |

The matchup prediction is then a mix-weighted sum over pitch types:

```
x̂_bp = Σ_t  q_bpt · x̂_bpt
```

where `q_bpt` is the probability that a PA between this pitcher and batter
*ends* on pitch type `t`.

This is not a true batter-by-pitcher interaction in the statistical sense. It is
an **arsenal-mediated matchup**: the interaction emerges because a particular
pitcher distributes PAs across pitch types on which the batter and pitcher have
different conditional strengths. A genuine `b × p × t` term would need
substantial head-to-head history and would be hopelessly sparse.

## Why it could beat aggregate wOBA

Aggregate wOBA treats two pitchers similarly when their overall results are
similar — an elite-slider/weak-fastball arm and an elite-fastball/weak-slider
arm look alike. It likewise treats two equal-overall batters alike even if one
destroys fastballs and struggles against sliders.

The pitch-type model distinguishes those matchups. It can capture four things
aggregate wOBA discards: the pitcher's arsenal distribution, his quality within
each pitch type, the batter's relative performance by pitch type, and how the
three line up in this specific matchup. That is a real theoretical advantage.

## The rough summary-stat version

If only leaderboard summaries are available, the natural first approximation is

```
x̂_bpt = μ_t · (B̃_bt / μ_t) · (P̃_pt / μ_t) = B̃_bt · P̃_pt / μ_t
```

with `B̃_bt` the shrunk batter wOBA against type `t` and `P̃_pt` the shrunk
pitcher wOBA allowed on type `t` — both shrunk toward **their own player's
general level**, not directly toward league.

The batter and pitcher ratios must be centred so their pitch-type cells
reconstruct each player's aggregate ability. Otherwise the pitcher's arsenal
quality and his overall wOBA are counted twice. This is the single most
important constraint in these notes, and it is the one the shipped shadow arm
implements (each cell regressed toward the hitter's own overall relative level).

A calibrated version is safer:

```
log(x̂_bpt) = log(μ_t) + λ_B · log(R_bt) + λ_P · log(R_pt)
```

with `λ_B` and `λ_P` estimated from data. The prior xwOBA reliability estimate
does not carry into this wOBA test; rerun the probe before choosing either
coefficient, and expect `λ_B` may need to sit considerably below 1.

## Aggregate wOBA should remain the prior

"Instead of aggregate wOBA" must not mean discarding aggregate information.
Aggregate wOBA determines the player-level fallback:

- No batter pitch-type data → the batter's aggregate level.
- No pitcher pitch-type data → the pitcher's aggregate level.
- No reliable deviations for either → the model collapses **exactly** to the
  aggregate prediction.

Pitch-type cells should only redistribute a player's known ability across pitch
types. They must not independently re-estimate that ability from small cells.

## The largest theoretical risks

**1. Terminal pitch type is endogenous.** A PA does not randomly end on a slider
or a fastball. The terminal pitch depends on count, batter handedness, previous
pitches, takes and fouls, pitcher strategy, and game situation. A slider ending
an 0–2 PA is not comparable to a fastball ending a 3–1 PA. League-relative
normalization by pitch type helps but does not remove batter- and
pitcher-specific count distributions. At minimum the league baselines — and
preferably the player effects — should be conditioned on platoon handedness.

**2. Pitch-type labels are broad.** Two four-seamers can differ by six mph,
vertical break, release height, and location. Performance against "FF" may not
transfer from an ordinary fastball to an unusual one. Pitch-shape clusters would
eventually be more predictive than MLB pitch labels.

**3. The pitcher throws different mixes to different batters.** One
starter-wide PA-share vector assumes he attacks lefties and righties
identically. He does not. The weighting should be `q(t | p, batter side)`, and a
fully developed model would also condition on batter tendencies.

**4. Measurement noise compounds.** The joint model introduces three uncertain
quantities, and approximately

```
Var(δ_joint) ≈ Var(δ_B) + Var(δ_P) + Var(δ_q)
```

The 0.0028 game-delta noise measured at `K = 600` covers only the batter side.
Pitcher and mix uncertainty add to it. Against a current median `|xw_net|` of
0.0188 that is already a tight budget — see the noise-budget argument in
`MATCHUP_SITE.md`.

**5. Retrospective leakage is easy.** A game's PAs contribute to both the
batter's and the pitcher's full-season pitch-type numbers, so using those
season values to "predict" that same game injects the outcome into both sides of
the feature. Every backtest must use strictly pregame data or
leave-the-matchup-out features. This is the same constraint that makes the
existing ledger unusable for backtesting the arm: `.savant_cache/` is
date-keyed and gitignored, so historical leaderboard state is not recoverable.

## Best development sequence

Do not jump from the aggregate baseline to the full joint model. Run an
ablation ladder:

| Model | Batter type residual | Pitcher type residual | Arsenal mix |
|---|:--:|:--:|:--:|
| Aggregate baseline | No | No | No |
| Mix baseline | No | No | Yes |
| Batter arm *(what ships today, shadow-only)* | Yes | No | Yes |
| Pitcher arm | No | Yes | Yes |
| Joint model | Yes | Yes | Yes |

This isolates whether any improvement comes from simply knowing the pitcher's
mix, from batter-specific pitch response, from pitcher quality by pitch type, or
from the combination. The pitcher component may prove more reliable than the
batter component, because pitch shape and quality are repeatable pitcher skills
— but much of that value may already sit in aggregate pitcher wOBA, so only its
within-arsenal residual can add information.

## Bottom line

The theory is valid and more complete than aggregate wOBA alone. The clean
model is

```
league pitch-type baseline
  + batter overall + pitcher overall
  + batter-type residual + pitcher-type residual
```

weighted by a handedness-aware terminal-pitch distribution.

The key empirical question is not whether batter and pitcher pitch-type numbers
contain signal individually. It is whether the **joint, fully shrunk, time-safe
prediction improves future PA-level wOBA beyond the aggregate model.**

Given current batter reliability, expect one of three outcomes:

1. Pitcher-type residuals add useful lift while batter residuals earn very
   little weight.
2. CU/ST/SL batter effects add small matchup value but FF adds almost none.
3. The theory is directionally correct but leaderboard samples are too noisy,
   requiring pitch-shape or pitch-level modelling to pay.

**Promotion gate.** Any move from shadow to lean changes both prediction math
and `|xw_net|` units, so it starts new `RECORD_TAGS` **and** `SCALE_TAGS`
families. Run `pitch_arsenal_probe.py` first — it decomposes observed cell
dispersion into signal and sampling noise and prints the result against the
noise budget the arm must clear.
