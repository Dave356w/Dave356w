# Registered predictions — lineup-window artifact test

Registered 2026-09-06, BEFORE any output was computed or inspected.
Diagnostic only: nothing here feeds back into a lean, a delta or a grade.

## Hypothesis (fixed in advance)

Components in `actuals_backfill.components_summary` are scored "against their
own realised phase". The SP/BP boundary is endogenous to lineup quality:
strong lineups chase starters early and get short windows; weak lineups get
long windows that include third-time-through PAs. If that boundary drives the
result, the negative lineup slope (-0.83 +/- 0.51 at n=598) is a measurement
artifact of window length, not evidence the term lacks signal.

## Registered predictions

- **R1**: the lineup residual (`act - pred`) regressed on starter batters
  faced has a **POSITIVE** coefficient.
- **R2**: refitting the lineup slope with starter BF as a covariate moves the
  slope **UPWARD** (i.e. less negative than -0.83).

## Registered falsification

If the BF-conditioned lineup slope sits within ~0.1 of -0.83, the artifact
story is dead and the term is contributing noise. Say so plainly.

## Registered control

The same two fits are run for the SP component. SP's window is endogenous in
the same way, so its slope should also move. If lineup moves and SP does not,
that WEAKENS the mechanism rather than supporting it.

## Pre-specification constraints

- One test per registered prediction. No sweeps, no variants, no thresholds
  chosen after looking.
- All 598 side-games. No long/short subsampling: the unconditioned SE is
  already +/- 0.51 and halving n proves nothing.
- Task 3 (fixed 18-batter rescore) runs only if plate-appearance-level data
  with a batters-faced or batting-order index exists. The window is fixed at
  18 in advance; no other window is to be run or reported.
