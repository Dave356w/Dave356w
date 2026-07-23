# Build-logic validation — MLB matchup-lean model

> Historical validation note: this review covers the starter-only v5
> prediction math as of 2026-07-21. v6 retains the verified shrinkage and
> lineup math but starts a new record family because it replaces the pitching
> input with an expected-IP starter/bullpen blend; see `MATCHUP_SITE.md`.

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
