# Build-logic validation — MLB matchup-lean model

> **Historical document. Reviewed the v5 model on 2026-07-21 and refreshed for
> xwOBA v10; the current forward test is `woba+plat_consol_v4`.** Its structural
> checks still describe the inherited v10 construction, but every empirical
> result here predates the wOBA substitution and is not evidence for that new
> lineage. What still held at v10 and what had been superseded:
>
> | section | status against v10 |
> |---|---|
> | 1. Matchup math | **Holds**, with one amendment — the lean is now a two-phase blend, so the `d_lineup + d_sp` linearization in §1 describes the starter phase only. See the amendment note in §1. |
> | 2. Shrinkage | **Superseded twice.** v8 retired the per-build method-of-moments estimator for a fixed `XWOBA_SHRINK_K = 100`; wOBA v3 then refitted that to 400 and wOBA v4 changed the target from a population centre to each player's own history. §2 documents the v5–v7 estimator it validated, not today's code. |
> | 3. Platoon lens | **Holds.** Still computed and graded; no longer surfaced on the cards. |
> | 4. Vs-market test | **Holds.** Unchanged. |
> | 5. Integrity controls | **Holds**, and the pregame lock has since been exercised in anger — it correctly rejected post-game refreshes on 2026-07-26 and 07-28. |
> | Empirical read | **Refreshed below** against the current 310-row ledger. |
>
> The model versions between this review and now: v6 (expected-IP
> starter/bullpen blend), v7 (centre-matched shrinkage moments, full precision,
> zero-as-abstention), v8 (fixed `K=100`), v9 (starter/bullpen phase split),
> v10 (PA-share phase weighting), then the wOBA lineage — wOBA v1 (observed
> wOBA replaces xwOBA throughout), v2 (exposure-centred starter platoon prior),
> v3 (`K` refitted 100 → 400; relievers shrink toward the relief pool's own
> unweighted centre) and v4 (the shrinkage target becomes each player's own
> recency-weighted history). See `MATCHUP_SITE.md` §"The current model".
>
> Note that §2 below is doubly superseded: it documents the v5–v7 estimator,
> the table row beneath calls it out as replaced by v8's fixed `K = 100`, and
> **that value is itself now historical** — the shipped constant is
> `XWOBA_SHRINK_K = 400`, and what it regresses toward is no longer a
> population centre at all.

**Date of review:** 2026-07-21 · **Refreshed:** 2026-07-29
**Scope:** Statistical soundness and robustness of the daily matchup-lean build
logic in this repo (`build_site.py`, `grade_leans.py`, `market_backfill.py`)
that generates <https://dave356w.github.io/Dave356w/>.

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
  > **v9/v10 amendment.** The edge is now `q·M_SP + (1−q)·M_BP`, so this
  > linearization describes the **starter phase only**. The bullpen phase adds
  > a second `B_0·P_BP/L` term carrying weight `1−q` (typically 0.35–0.45), and
  > it uses the *neutral* lineup composite, so the handedness term appears at
  > weight `q` rather than 1. The equal-weight expectation for lineup vs
  > pitching still holds within each phase; the weight fit has not been
  > re-derived for the two-phase form and remains gated at 120 F5 decisions
  > (currently 21), so nothing has been fit against it either way.
- Team/side mapping is consistent end to end: the away-SP row carries the
  **home** offense's edge (and vice-versa); grading resolves W/L against the
  correct winner (`grade_leans._wlt`). Full-game and first-5-innings (F5)
  grades are computed separately; F5 requires all five innings present.

### 2. Empirical-Bayes xwOBA shrinkage — **as reviewed at v5; superseded at v8, again at wOBA v3/v4**

> **Current code:** `shrink_xwoba` applies a fixed `XWOBA_SHRINK_K = 400` — not
> the 100 this note carried through the v8–v10 era — to every batter, probable
> starter, and reliever. The *form* is still unchanged, `x* = (n·x + K·prior)/(n+K)`,
> and that is the whole reason this section's verification is still worth
> reading; but both free terms have since moved. `K` was refitted at wOBA v3
> (three independent estimators, all excluding 100). `prior` stopped being a
> single population centre at wOBA v4: each player regresses toward his own
> recency-weighted 2023–2025 history, and relievers who lack one regress toward
> the relief pool's unweighted centre rather than the league batter rate.
> `estimate_shrink_k` is no longer on the lean path.
> v7 first corrected the estimator (centring each moment on its own population
> mean), then v8 removed it entirely: a fixed pseudo-sample is reproducible
> across builds and preserves more of the player-distribution tails than the
> typically larger estimated constants it replaced. The verification below is
> retained because it is what established the estimator was *correctly
> implemented* before it was retired on other grounds.
- `x* = (n·x + K·prior)/(n+K)`, both batters and the starter regressed toward
  the shared league xwOBA prior. Sharing the prior is correct: league-wide
  xwOBA-allowed equals league xwOBA.
  > **Amendment.** That last sentence proves less than it is used for. The
  > identity is true, and empirical Bayes wants the centre of the pool each
  > estimate is drawn from — a pooled mean is not the centre of any of its
  > subpools, it only guarantees they average back to it. Batters, rotation
  > arms and the relief pool are three pools shrunk toward one target, and the
  > PA/BF weighting that builds the target lifts it above the player-level
  > centre within each. v7 named the batter half of this
  > (`MATCHUP_SITE.md`, "Centre-matched shrinkage moments"), corrected the `K`
  > estimator for the resulting between-centre offset, and deliberately left
  > the target alone; v8 then retired that estimator, so nothing measured the
  > offset any more. `prior_population_centres` now logs every pool's weighted
  > and unweighted centre against the target each build, plus `bias` — mean
  > prior weight × (pool centre − target) — so the question is settled from
  > measurement rather than from the identity above. It is log-only and moves
  > no lean; whether the target should split is a separate, unmade decision.
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

## Empirical read — refreshed 2026-07-29 (310 graded rows, 309 with closing lines)

Computed with `market_backfill.vs_market_summary`, the same function the site
renders from. Families are never pooled for a *record*; they are shown side by
side here because the pooled row is what the public page publishes.

| Scope | Lean | Record | Market-expected W | z | ROI |
|---|---|---|---|---|---|
| all graded (mixed families) | xwOBA | 174-135 | 166.5 | **+0.87** | +1.28u |
| all graded (mixed families) | platoon | 121-113 | 126.7 | **−0.76** | −19.95u |
| v2/v3 (n=188) | xwOBA | 109-79 | 101.3 | **+1.13** | +5.96u |
| v9/v10 — current (n=26) | xwOBA | 13-13 | 14.3 | **−0.50** | −2.91u |

- The 2026-07-21 review read xwOBA at z +1.09 over 203 rows. With 106 more
  games the pooled z has **fallen to +0.87** — the edge has not grown with
  sample, which is what an absent edge looks like.
- Against a trivial baseline: over all 309 rows the model is .563 while
  always-home is .502. In the current v9/v10 family it is .500 against
  always-home .462, on 26 games — far too few to read.
- Market agreement has *risen sharply*: ~69% of leans matched the closing
  favourite over the full ledger, but **84.6%** in the current family. The
  model is tracking the favourite more closely than it used to, which shrinks
  the space where it can differ from the market at all.

This is not a flaw in the build logic; it is the sober read the z-test is
designed to produce. The system is honest about it by construction.

## Test suite

`python -m pytest tests/ -q` → **211 passed** (2026-07-29). The original review
ran 26 via `unittest discover` after fixing one stale fixture
(`test_pitcher_card_shows_season_era_but_colors_l5_vs_league` was missing the
`pit_bb` key that `_side_html` reads). CI still does not run the suite —
`build.yml` has no test step and `requirements.txt` has no pytest — so this
remains a local-only gate.

## Notes / limitations

- Multiple sub-splits (|Δ| terciles, DIVERGE h2h, reliable-only) are useful
  descriptively but invite multiple-comparison over-reading; the headline z vs
  market is the disciplined significance statistic.
- The MLB StatsAPI and Baseball Savant endpoints were not reachable from the
  review environment (network policy), so a live end-to-end fetch was not
  exercised; the model math, shrinkage, and market statistics were validated
  from the committed ledger and unit tests instead.
- The v7 family's 22-23 line is internally heterogeneous — three different
  prediction models shipped under that one tag. See the versioning note at the
  end of the platoon-adjustment section in `MATCHUP_SITE.md`.
- **Superseded note:** this section used to record that
  `data/ledger_report.txt` read "no graded games yet". It now reports the
  current family (26 graded) plus immutable per-family history, so the internal
  report and the public page no longer disagree.


## Related

- `docs/pitch_mix_theory.md` — design notes for the pitch-type-conditioned
  matchup (the shadow arm in `pitch_arsenal.py`). Moved out of this file: it is
  a forward-looking design argument, not a validation of shipped build logic.
- `docs/f5_market_validation.md` — F5 market capture and first results.
