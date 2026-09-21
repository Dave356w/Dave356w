# Repository cleanup and delta/market separation review

Dated 2026-09-21. Source: commit
`e09742389eb76ff3afceeeec7096771f85675508`, including its committed ledger.
Numbers below are a fixed audit snapshot, not a live results dashboard.

## Findings and scope

The immediate cleanup is clearer ownership of production code, research, and
reporting. The statistical problem is estimating delta's incremental information
after accounting for market strength. More finely divided cells would make the
current data sparsity worse.

This change corrects the README, adds a same-row market baseline to the existing
shadow report, and supplies a reproducible research audit. It does not change
v13 predictions, the M0=10 shadow rule, registrations, UI behavior, workflows,
or historical ledger fields. The new research code is needed to compare
probability forecasts on identical chronological rows; cell hit-rate summaries
cannot answer that question.

## Repository cleanup

| Priority | Verified condition | Concrete next change |
|---|---|---|
| Now | README described v12 and a live hybrid selector; source defaults to v13 and publishes the model lean | Corrected in this change; describe retained hybrid code as monitoring/reconciliation |
| Now | Shadow report printed its probability losses without a market baseline | Add baseline and shadow-minus-market losses on identical rows; no decision changes |
| Next | `build_site.py` has 8,837 lines and `grade_leans.py` 2,256 | Extract pure reporting helpers first, then rendering. Keep existing CLI entry points and imported function compatibility while updating callers |
| Next | 47 root Python files, including 17 with `probe` in the name; 14 workflows | Put future research under `research/` (started by this audit); move existing probes one at a time with import/workflow checks |
| Next | `CLAUDE.md` has 4,393 lines and `MATCHUP_SITE.md` 1,195; current instructions coexist with dated decisions | Keep a short current architecture/standards document and move historical narrative to dated docs with links; preserve decision provenance |
| Later | Fixed delta edges appear in `grade_leans.py` and `price_calibration_shadow.py`, alongside distinct display/registered cutoffs | Share descriptive band definitions in a dependency-light module; keep frozen registration cutoffs separately versioned |
| Preserve | 263 tracked files under `data/`, including historical source snapshots | Do not treat source history as regenerable build output. `public/` and caches are already ignored |

Do not delete `hybrid_test.py`, `hybrid_v2.py`, or related modules simply because
the displayed hybrid selector was retired. They still have reporting and
pregame-capture consumers. Do not rename ledger columns or conflate
`RECORD_TAGS` with `SCALE_TAGS` in a structural cleanup.

The report can eventually have a short front page, with registered monitoring,
retrospective calibration, and component diagnostics in separate sections or
artifacts. Every section should name its row basis, price snapshot, last scored
date, exclusions, and uncertainty convention. Generate those views from shared
calculations rather than copying analysis into each renderer.

## What the data can currently separate

The v13-represented frame contains 490 distinct games from 37 slates,
2026-08-15 through 2026-09-20: 445 reconstructed and 45 native v13. Its 312-178
record differs from the report's original-decision 307-183 line by design.
These bases must not be silently interchanged.

On this same v13 representation:

- Correlation between absolute delta and leaned-side closing market probability
  is **0.5250**. The unadjusted delta trend includes changing market strength.
- Eighteen of the twenty cells are populated. Six cells have fewer than ten
  games, including two empty cells.
- The `q >= .55` cell column covers actual probabilities from **.5511 to .7711**.
- The native 45 games cover only **three slates**. Splitting them out is useful
  provenance, but does not create a substantial independent validation set.

| Absolute delta | Games | Lean win rate | Mean leaned-side market q | Rate minus mean q |
|---|---:|---:|---:|---:|
| [0, .010) | 140 | 56.43% | 51.71% | +4.71 pp |
| [.010, .020) | 126 | 70.63% | 53.94% | +16.70 pp |
| [.020, .030) | 97 | 54.64% | 55.98% | -1.34 pp |
| [.030, .050) | 94 | 70.21% | 59.95% | +10.26 pp |
| [.050, infinity) | 33 | 75.76% | 65.56% | +10.20 pp |

These margins are descriptive, not a monotone confidence scale. Subtracting
mean q is a useful first adjustment, but it is not a full conditional model
and does not remove all differences in the market distribution within bands.

## A structural limitation in the current cell estimator

For a populated cell, the existing estimator is

`p_old = (W + M0*qbar)/(n + M0)`.

Its apparent excess over the current game's market probability `q_i` decomposes
exactly as:

`p_old - q_i = (qbar - q_i) + (W - n*qbar)/(n + M0)`.

The first term is a change in market level, not observed model skill. If a
cell's old mean is .60 and the current game's q is .72, even a perfectly
market-matching historical cell is assigned .60. Conversely a weaker current
favorite can receive an apparent improvement purely from its cell's old prices.

A minimal diagnostic alternative preserves current q:

`p_residual = clip(q_i + (W - n*qbar)/(n + M0), 1e-6, 1-1e-6)`.

This keeps M0=10 and the same prior-date cell observations. It is a bounded
residual heuristic, not an exact beta-binomial posterior. It is benchmarked
below but is **not substituted into the existing shadow arm**.

## Chronological benchmark

Run `python research/delta_market_audit.py` for JSON results. No network calls
or input writes are needed. The CLI resolves the default ledger relative to
the repository, so it can also run from another working directory.

All regression alternatives train on strictly earlier slate dates with a
common minimum of 100 prior games and seven prior slates. The history starts
at the earliest v13-represented slate, uses the stored reconstruction, and
continues through native v13 without resetting. All candidates are evaluated
on the same **384 games / 29 slates, 2026-08-23 through 2026-09-20**. The first
106 games form the common warmup, including training for the grid estimators.

Let `z=logit(q_i)` and `x=(abs(delta)-.020)/.020`. The regression alternatives
minimize summed Bernoulli negative log likelihood plus `sum(beta^2)/2`, with
`z` as an offset. Thus their prior is the current market probability. Coefficient
units, centering, and penalty are fixed engineering choices, not sweep winners.

| Alternative | Logit correction to z | Brier loss | Log loss |
|---|---|---:|---:|
| Market | 0 | 0.233372 | 0.658867 |
| Existing M0=10 cell grid | Existing probability formula | 0.234723 | 0.662492 |
| Residual cell grid | Probability-scale formula above | 0.232270 | 0.656721 |
| Pooled offset | a | 0.231148 | 0.653764 |
| Market recalibrated | a + b*z | 0.231977 | 0.655621 |
| Market plus delta | a + b*z + c*x | 0.233025 | 0.658111 |
| Market/delta interaction | a + b*z + c*x + d*z*x | 0.233302 | 0.660399 |

Lower losses are better. The pooled one-parameter correction is nominally best
here. Adding delta to market recalibration worsens Brier by **0.001049** and
log loss by **0.002490**. The interaction also worsens both. This does not prove
delta has no incremental information; these small, specific alternatives did
not demonstrate it on this sample. The outcome is oriented to the model's
chosen side, so this tests the additional information in **delta magnitude**
after the lean is chosen. It does not test whether signed delta is useful for
choosing that side in the first place.

Two thousand whole-slate bootstrap draws at each seed 17, 41, and 73 check
Monte Carlo stability. For the delta addition, the seed-17 conditional 95%
Brier difference interval is **[-0.000714, +0.003136]**. For residual-grid minus
existing-grid it is **[-0.005592, +0.000892]**. Both cross zero at all three
seeds. The pooled-offset improvement versus market also crosses zero (seed-17
Brier interval **[-0.007761, +0.003317]**). These are paired intervals conditional on the already fitted forecasts;
they do not include refitting uncertainty, multi-slate serial dependence,
correction for searching alternatives, or repeated monitoring.

For direct reconciliation with the existing report, the all-490-row scores
(including grid cold starts) are market **0.231777 / 0.655637** and existing
grid **0.234078 / 0.661115**, in Brier/log-loss order. The revised report prints
both and their differences.

## Better separation in the next research stage

1. **Keep market probability continuous.** Use an offset or market calibration
   as the baseline; retain fixed bands for presentation and coverage counts.
2. **Ask for delta's increment.** Compare market-only, market plus delta, then
   an interaction, on identical held-out slates. Report paired losses and
   uncertainty, not just win rates. This audit implements the first comparison.
3. **Consider partial pooling only if a richer shape is needed.** A future
   hierarchical alternative can use `logit(p_i)=logit(q_i)+a+u_delta+v_market+w_cell`,
   shrinking cell interactions more strongly than main effects. This borrows
   information across sparse cells, but it has not been benchmarked here and
   should not be promoted on principle alone. The simpler models above remain
   the controls. Avoid estimating unsupported extreme-delta/underdog cells as
   though they had direct observations.
4. **Separate retrospective fitting from forward evidence.** Stored
   reconstruction may initialize a new shadow, continuously from the earliest
   slate as requested, but future forecasts must be saved with version, actual
   pregame snapshot, training cutoff, and state counts before the game. Closing
   prices remain a retrospective diagnostic. No historical final-result
   availability timestamps are available here, so earlier-date outcomes are
   assumed available; suspended or late-final games would need snapshot evidence.
5. **Keep the existing registrations frozen.** New model comparison rules need
   a new research/version identity. Native rows inspected while designing this
   audit cannot subsequently be relabeled as its prospective evidence.

One reporting claim deserves correction separately: `grade_leans.py` calls
pooling "licensed" when a maximum family contrast is below `sqrt(2*log(k))`.
That expression is a rough scale for null extremes, not a calibrated critical
value or an equivalence test. Failure to detect a difference does not establish
exchangeability. Use "no difference detected at this resolution" with explicit
assumptions, or predefine a practically meaningful equivalence margin. Similar
care applies to labeling a noisy component "unmeasurable": a non-significant
heterogeneity test alone does not make a fitted slope mathematically undefined.

Method references: [probability scoring and calibration](https://scikit-learn.org/stable/modules/calibration.html),
[chronological validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split),
and [multilevel modeling and partial pooling](https://sites.stat.columbia.edu/gelman/research/published/multi2.pdf).
Brier and log loss assess overall probability quality, not calibration alone.
