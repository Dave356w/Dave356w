# Build-logic validation — MLB matchup-lean model

> Historical validation note: this review covers the starter-only v5
> prediction math as of 2026-07-21. v6 retains the verified shrinkage and
> lineup math but starts a new record family because it replaces the pitching
> input with an expected-IP starter/bullpen blend; see `MATCHUP_SITE.md`.
> v8 later retired the method-of-moments estimator and fixed batter and pitcher
> xwOBA shrinkage at `K=100`; section 2 below documents the historical v5-v7
> implementation rather than the current one.

**Date:** 2026-07-21
**Scope:** Statistical soundness and robustness of the daily matchup-lean build
logic in this repo (`build_site.py`, `grade_leans.py`, `market_backfill.py`)
that generates <https://dave356w.github.io/dave356w/>.

## Verdict

The build logic is **sound** and the methodology is **statistically robust**.
The model math, the empirical-Bayes shrinkage, and the vs-market significance
test were each re-derived and independently verified. The record-keeping is
protected by the right integrity controls (pregame lock, score-verified market
join, model-family separation).

One honest caveat applies to *results* (not logic): at the current sample the
model shows **no statistically significant edge over the closing market**.

## What was verified

### 1. Matchup math (`build_site.py`)
- Multiplicative ratio anchored on league average: `M = B·P/L`
  (`edge = M − L`); additive `B + P − L` for EV/LA. Confirmed in
  `matchup_value` / `build_matchup`.
- Lean signal `xw_net = home_off_edge − away_off_edge` linearizes to
  `d_lineup + d_sp`, giving the lineup and starter components equal first-order
  weight — consistent with the logit reweight check in `grade_leans.report`
  that expects an implied weight ratio ≈ +1.00.
- Team/side mapping is consistent end to end: the away-SP row carries the
  **home** offense's edge (and vice-versa); grading resolves W/L against the
  correct winner (`grade_leans._wlt`). Full-game and first-5-innings (F5)
  grades are computed separately; F5 requires all five innings present.

### 2. Empirical-Bayes xwOBA shrinkage — v5 (`shrink_xwoba`, `estimate_shrink_k`)
- `x* = (n·x + K·prior)/(n+K)`, both batters and the starter regressed toward
  the shared league xwOBA prior. Sharing the prior is correct: league-wide
  xwOBA-allowed equals league xwOBA.
- `K` is estimated per build by **method of moments**. Re-derived:
  `E[d²] = τ² + σ²/n`, so
  `Vu − Vw = σ²·(mean(1/n) − 1/mean(n))` (non-negative by AM–HM),
  yielding `σ²`; then `τ² = Vw − σ²/n̄`; then `K = σ²/τ²`. The code matches
  exactly.
- Defensive: clamped to a PA band, fallback on thin pools / non-positive
  variance components, logged each run. The planted-ratio unit test
  (`XwobaShrinkageTests.test_mom_k_recovers_planted_ratio`) recovers `K`
  within 35%.

### 3. Platoon lens (`build_platoon_matchup`)
- Two-stage empirical Bayes: each side's vs-hand OPS is regressed toward an
  `overall × (league-platoon / league-overall)` prior with fixed `K_BAT`/`K_PIT`.
- Reliability-gated: a lean is `reliable` only when the starter has ≥50 BF vs
  the relevant hand and ≤4 lineup bats fall below 30 split PA. Unreliable leans
  render as prior-driven and are excluded from the reliable-only record and the
  vs-market platoon scoreboard.

### 4. Vs-market significance test (`market_backfill.py`)
- Devig is proper two-way proportional: `p_home = imp_h/(imp_h + imp_a)`.
- The scoreboard z is a standardized **Poisson-binomial**:
  `z = (w − Σp)/√Σp(1−p)`. Monte-Carlo (200k sims) confirms the standardized
  statistic has mean ≈ 0 and std ≈ 1 under the null.
- Flat-stake ROI uses the decimal odds of the model's own side. Correct.

### 5. Integrity controls (what makes the record trustworthy)
- **Pregame snapshot lock** — snapshots captured at/after scheduled first pitch
  are rejected (`grade_leans._lock_status`); graded rows are immutable. No
  look-ahead leakage.
- **Score-verified market join** — a row that can't be verified by final score
  keeps NaN market columns, never a guessed line; a market outage never fails
  the grading run.
- **Model-family separation** — a prediction-math change bumps `MODEL_TAG` and
  starts a new `RECORD_TAGS` family, so incompatible model versions never pool
  in the record or the weight fit.
- Sample gate (`N_FIT_MIN = 120` F5 decisions) before the logit reweight;
  standardized predictors + ridge for numerical stability.

## Empirical read (committed ledger, 203 graded games with closing lines)

| Lean | Record | Market-expected W | z | ROI |
|------|--------|-------------------|-----|-----|
| xwOBA   | 117-86 | 109.4 | **+1.09** | +5.26u |
| platoon | 81-71  | 81.9  | **−0.14** | −8.69u |

- The full-game xwOBA win rate (.582 over the v2+v3 family) beats a naive
  baseline, but against the **closing line** (the sharp benchmark the z-test
  targets) it is within noise (z ≈ +1.1; ~1.96 needed for significance).
- The model agrees with the market favorite ~68% of the time and its record
  equals the favorite baseline (117-86) — it is largely tracking the favorite.

This is not a flaw in the build logic; it is the sober read the z-test is
designed to produce. The system is honest about it by construction.

## Test suite

`python -m unittest discover -s tests` → **26 passed** after fixing one stale
fixture (`test_pitcher_card_shows_season_era_but_colors_l5_vs_league` was
missing the `pit_bb` key that `_side_html` reads; production always sets it in
`_df_to_combined_games`). That fix ships in this same PR. No statistical impact.

## Notes / limitations

- The committed `data/ledger_report.txt` reads "no graded games yet" because it
  summarizes only the *current* `MODEL_TAG` family (v5, 15 recent games); the
  full v2/v3 history (189 graded) is intact in `data/mlb_lean_ledger.csv`.
- Multiple sub-splits (|Δ| terciles, DIVERGE h2h, reliable-only) are useful
  descriptively but invite multiple-comparison over-reading; the headline z vs
  market is the disciplined significance statistic.
- The MLB StatsAPI and Baseball Savant endpoints were not reachable from the
  review environment (network policy), so a live end-to-end fetch was not
  exercised; the model math, shrinkage, and market statistics were validated
  from the committed ledger and unit tests instead.

## POSSIBLE UPDATES

- The theory is stronger than the current one-sided arm, but it should be modeled as a pitch-type-conditioned replacement for aggregate xwOBA—not as another multiplier layered on top of aggregate batter and pitcher xwOBA.

## What the model is actually estimating

For batter (b), pitcher (p), and pitch type (t), decompose expected xwOBA into:

[
g(\hat{x}_{bpt})
================

g(\mu_t)

* a_b
* d_p
* u_{bt}
* v_{pt}
  ]

Where:

* (\mu_t): league xwOBA when a PA ends on pitch type (t)
* (a_b): batter’s general ability
* (d_p): pitcher’s general run-prevention ability
* (u_{bt}): batter’s pitch-type-specific residual
* (v_{pt}): pitcher’s pitch-type-specific residual
* (g): a calibrated link, probably log-relative rather than raw addition

The matchup prediction becomes:

[
\hat{x}_{bp}
============

\sum_t q_{bpt},\hat{x}_{bpt}
]

Here (q_{bpt}) is the probability that a PA between this pitcher and batter ends on pitch type (t).

This is not a true batter-by-pitcher interaction in the statistical sense. It is an **arsenal-mediated matchup**: the interaction emerges because a particular pitcher distributes PAs across pitch types on which the batter and pitcher have different conditional strengths.

A true (b \times p \times t) term would require substantial head-to-head history and would be hopelessly sparse.

## Why it could beat aggregate xwOBA

Aggregate xwOBA treats these two pitchers similarly if their overall results are similar:

* Pitcher A: elite slider, weak fastball
* Pitcher B: elite fastball, weak slider

It likewise treats two equal-overall batters similarly even if one destroys fastballs and struggles against sliders.

The pitch-type model distinguishes those matchups. It can theoretically capture information that aggregate xwOBA discards:

1. The pitcher’s arsenal distribution.
2. The pitcher’s quality within each pitch type.
3. The batter’s relative performance against each pitch type.
4. How those three elements line up in this matchup.

That is a real theoretical advantage.

## The rough summary-stat version

If only leaderboard summaries are available, the natural first approximation is:

[
\hat{x}_{bpt}
=============

\mu_t
\left(\frac{\widetilde B_{bt}}{\mu_t}\right)
\left(\frac{\widetilde P_{pt}}{\mu_t}\right)
============================================

\frac{\widetilde B_{bt}\widetilde P_{pt}}{\mu_t}
]

Where:

* (\widetilde B_{bt}) is shrunk batter xwOBA against type (t)
* (\widetilde P_{pt}) is shrunk pitcher xwOBA allowed on type (t)
* Both shrink toward their player’s general level, not directly toward league

However, the batter and pitcher ratios must be centered so that their pitch-type cells reconstruct their aggregate ability. Otherwise the pitcher’s arsenal quality and overall xwOBA get counted twice.

A calibrated version is safer:

[
\log(\hat{x}_{bpt})
===================

\log(\mu_t)
+\lambda_B\log(R^B_{bt})
+\lambda_P\log(R^P_{pt})
]

The data should estimate (\lambda_B) and (\lambda_P). Given the batter-side reliability of only 0.161, (\lambda_B) may need to be considerably below 1.

## Aggregate xwOBA should remain the prior

“Instead of aggregate xwOBA” should not mean discarding aggregate information. Aggregate xwOBA should determine the player-level fallback:

* No batter pitch-type data → batter’s aggregate level.
* No pitcher pitch-type data → pitcher’s aggregate level.
* No reliable deviations for either player → the model collapses exactly to the aggregate prediction.

The pitch-type cells should only redistribute a player’s known ability across pitch types. They should not independently re-estimate all of that ability from small cells.

## The largest theoretical risks

### 1. Terminal pitch type is endogenous

A PA does not randomly end on a slider or fastball. The terminal pitch depends on:

* Count
* Batter handedness
* Previous pitches
* Batter takes and fouls
* Pitcher strategy
* Game situation

A slider ending an 0–2 PA is not directly comparable to a fastball ending a 3–1 PA. League-relative normalization by pitch type helps, but it does not remove batter- and pitcher-specific count distributions.

At minimum, league baselines and preferably player effects should be conditioned on platoon handedness. Count conditioning would be even better.

### 2. Pitch-type labels are broad

Two four-seamers can differ by six mph, vertical break, release height, and location. “Performance against FF” may not transfer from an ordinary fastball to an unusual one.

Eventually, pitch-shape clusters could be more predictive than MLB pitch labels. That would represent a more genuine compatibility model.

### 3. The pitcher controls different mixes against different batters

Using one starter-wide PA-share vector assumes he attacks lefties and righties identically. He does not.

The weighting distribution should ideally be:

[
q(t\mid p,\text{batter side})
]

A fully developed model might also condition on batter tendencies, because batters influence which pitches survive to the terminal pitch.

### 4. Measurement noise compounds

The joint model introduces three uncertain quantities:

* Batter pitch-type effect
* Pitcher pitch-type effect
* Matchup-specific terminal-pitch distribution

Approximately:

[
\operatorname{Var}(\delta_{\text{joint}})
\approx
\operatorname{Var}(\delta_B)
+
\operatorname{Var}(\delta_P)
+
\operatorname{Var}(\delta_q)
]

The current (0.0028) game-delta noise at (K=600) covers only the batter side. Pitcher and mix uncertainty will increase the total.

### 5. Retrospective leakage is easy

A game’s PA contributes to both the batter’s and pitcher’s full-season pitch-type numbers. Using those full-season values to “predict” that same game would inject the outcome into both sides of the feature.

Every backtest must use strictly pregame data or leave-the-matchup-out features.

## Best development sequence

Do not jump directly from the aggregate baseline to the complete joint model. Run an ablation ladder:

| Model              | Batter type residual | Pitcher type residual | Arsenal mix |
| ------------------ | -------------------: | --------------------: | ----------: |
| Aggregate baseline |                   No |                    No |          No |
| Mix baseline       |                   No |                    No |         Yes |
| Batter arm         |                  Yes |                    No |         Yes |
| Pitcher arm        |                   No |                   Yes |         Yes |
| Joint model        |                  Yes |                   Yes |         Yes |

This will reveal whether any improvement comes from:

* Simply knowing the pitcher’s mix
* Batter-specific pitch response
* Pitcher quality by pitch type
* The combination of both sides

The pitcher component may prove more reliable than the batter component because pitch shape and pitch quality are repeatable pitcher skills. But much of that value may already be present in aggregate pitcher xwOBA, so only its within-arsenal residual can add new information.

## My bottom line

The theory is valid and more complete than using aggregate xwOBA alone. The clean model is:

[
\boxed{
\text{league pitch-type baseline}
+\text{batter overall}
+\text{pitcher overall}
+\text{batter-type residual}
+\text{pitcher-type residual}
}
]

weighted by a handedness-aware terminal-pitch distribution.

The key empirical question is not whether batter and pitcher pitch-type numbers contain signal individually. It is whether the **joint, fully shrunk, time-safe prediction improves future PA-level xwOBA beyond the aggregate model**.

Given the current batter reliability, I would expect one of three outcomes:

1. Pitcher-type residuals add useful lift, while batter residuals receive very little weight.
2. CU/ST/SL batter effects add a small amount of matchup value, but FF adds almost none.
3. The theory is directionally correct but the available leaderboard samples are too noisy, requiring pitch-shape or plate-appearance-level modeling to make it pay.
