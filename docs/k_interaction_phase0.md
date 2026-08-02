# K% interaction layer — Phase 0 architecture audit

Gate result for `K_INTERACTION_BRIEF.md` Phase 0.

**Verdict: (C) — something else. Stopping here per the brief.**

Nothing was integrated, no tests were written, no constant was tuned. Phase 1
onward stays closed until the premise below is resolved.

## Premise correction: `mlb_distilled.py` does not exist

The brief instructs "Read `mlb_distilled.py`". There is no such file in this
repository, and no reference to that name in any `.py`, `.md`, or workflow file.

```
$ find / -name "mlb_distilled*"        # no results
$ grep -rl "mlb_distilled" .           # no results
```

The prediction math lives in `build_site.py` (4870 lines), with the graded
half in `grade_leans.py`. This audit reads those.

## Why not (A)

There is no simulation anywhere in the prediction path. No per-PA sampling, no
K/BB/BIP outcome draw, no Monte Carlo. The lean is a closed-form arithmetic
chain over season rates. Grep for a sampler returns nothing; the full path is
traced below.

## Why not (B)

(B) requires "a fixed or league BBE assumption." There is no BBE assumption in
the model, because **there is no plate-appearance event split at all** — not
fixed, not league, not emergent. The pipeline never decomposes a PA into
K / BB / HBP / BBE.

`WEIGHT_COL = "BBE"` (`build_site.py:1427`) is a *season batted-ball-event
count* used only as a fallback aggregation weight, and since v4 it is not even
that on the primary path — `lineup_weight()` (`build_site.py:1587`) prefers
slot-PA weights and falls back to `BBE` only when the batting order is
unavailable. It is a sample-size proxy, not a modelled BBE rate.

## What the model actually is

Per side, per game:

1. **Batter side.** Each hitter's Savant season `xwoba` (`build_site.py:415`,
   the custom leaderboard column) is shrunk toward the league prior by his `PA`
   (`aggregate_lineup`, `build_site.py:1765-1766` → `shrink_xwoba`, `:1601`).
   A platoon offset is added last, on the shrunk value (`:1797-1799`). Two
   composites are stored: `opp_xwOBA_neutral` (pre-platoon, `B_0`) and
   `opp_xwOBA_vs_sp` (post-platoon, `B_SP`).
2. **Pitcher side.** The starter's season xwOBA-allowed is shrunk by his `PA`
   (= batters faced) in `build_matchup` (`build_site.py:1842-1844` →
   `_shrink_one`, `:1615`). The bullpen pool is shrunk the same way
   (`build_site.py:990`).
3. **Combine.** `matchup_value(B, P, "xwOBA", L)` returns `B·P/L`
   (`build_site.py:1544-1549`) — the Bill James multiplicative form, on the
   *blended per-PA* rate.
4. **Phases.** `sequential_xwoba_phases` (`build_site.py:1951`) blends
   `M_SP = B_SP·P_SP/L` and `M_BP = B_0·P_BP/L` by PA share
   `q = (IP_sp·BF/IP_sp) / (IP_sp·BF/IP_sp + IP_bp·BF/IP_bp)` (`:1974-1978`).
5. **Lean.** `edge_xwOBA = mx_xwOBA − L`; `xw_net = home_off_edge −
   away_off_edge` (`grade_leans.py:320-325`); sign of `xw_net` is the lean
   (`:340-345`).

So the modelled quantity is a single blended per-PA xwOBA number per side.

## The double-count, and why it is (C) rather than (A)

Savant's `xwoba` is **per plate appearance**, with strikeouts entering as zero.
It is not `xwOBAcon`. That means:

```
xwOBA_blended ≈ (1 − K − BB − HBP)·xwOBAcon + BB·w_BB + HBP·w_HBP
```

which is exactly the identity `xwoba_from_branches` reconstructs (PDF ll. 59-70).
Every batter's blended xwOBA already integrates over **his own realized K%**, and
every pitcher's xwOBA-allowed already integrates over **his own realized K%**.
The `B·P/L` product is already an opportunity-weighted quantity.

Multiplying that product by a matchup BBE share would count the K branch twice —
the same failure (A) is written to catch, arriving through the season rate rather
than through a simulator. This is why the brief's trichotomy does not fit: the
double-count risk of (A) is present, but the mechanism of (A) is not.

**The module's required input does not exist in this pipeline.** `Batter.xwobacon`
(PDF l. 80) is mandatory and must be the BBE-conditional mean. Grep for
`xwobacon`, `est_woba_con`, `woba_con` across `build_site.py`, `grade_leans.py`,
`market_backfill.py`, `pitch_arsenal.py` returns **zero hits**. `STAT_COLS`
(`build_site.py:778`) carries `xwOBA`, `xBA`, `xSLG` — no xwOBAcon. Wiring this
layer is not an integration; it is a new Savant fetch plus a replacement of the
batter branch, and the pitcher branch has no defined counterpart at all
(`evaluate_lineup` takes only `sp.k_pct` from the pitcher — the starter's
xwOBAcon-allowed is silently dropped, so the entire pitcher contact-quality
signal that currently drives the lean would vanish).

### Direct repo precedent

`docs/pitch_mix_theory.md` records the identical framing error and its
correction, for the pitch-mix arm:

> the theory is stronger than the current one-sided arm, but it should be
> modelled as a pitch-type-conditioned *replacement* for aggregate xwOBA — not
> as another multiplier layered on top of aggregate batter and pitcher xwOBA.
> The shipped shadow arm is a multiplier, which is the weaker form.

The K% layer as written is a multiplier on aggregate xwOBA. Same weaker form,
same reason.

## The defensible smaller change

There is already a K% matchup estimate in the codebase. `STATCAST_RATE_COLS`
(`build_site.py:1426`) includes `K%`, so `build_matchup` computes `pit_K%`,
`opp_K%`, `mx_K%`, `edge_K%` for every row — they are in every
`data/leans_*_xw.csv` (columns 45-49). `K%` is not in `ADD_STATS`
(`build_site.py:1429`), so `mx_K%` is computed as `B·P/L`, not log5.

Replacing that `B·P/L` with `log5` is a well-posed, self-contained change to an
existing display column, testable against realized K outcomes without touching
the lean. It is far smaller than what the module implements, and it is the only
part of this work item with a clean home in the current architecture.

## Brief items 1-3

**1. Where xwOBA splits are shrunk, and the constants.**

| side | call site | denominator | constant |
|---|---|---|---|
| batter | `aggregate_lineup` `build_site.py:1765-1766` → `shrink_xwoba` `:1601` | `PA` | `XWOBA_SHRINK_K = 100.0` (`:1445`) |
| starter | `build_matchup` `build_site.py:1842-1844` → `_shrink_one` `:1615` | `PA` (= BF) | `XWOBA_SHRINK_K = 100.0` |
| bullpen pool | `bullpen_xwoba_aggregate` / plans `build_site.py:990` | `PA` | `XWOBA_SHRINK_K = 100.0` |

One fixed pseudo-sample for all three since v8 (`build_site.py:1255-1263`).
Form is `(n·x + k·prior)/(n + k)`; prior is `league_baseline["xwOBA"]`.
Gated by `USE_XWOBA_SHRINK = True` (`:1443`).

Note `LEAN_STRENGTH_PRIOR_N = 100` (`build_site.py:4367`) is a different
quantity that happens to share the numeral — the file says so explicitly at
`:4364-4366`. Do not sync them.

**2. Park and defense adjustments.**

**Neither is applied anywhere.** `grep -c "park_factor\|PARK_FACTOR"
build_site.py` → `0`; no park, venue-factor, or defensive-runs term exists in
the prediction path.

The brief flags this as "a live bug candidate independent of this layer." It is
not live — the bug it describes (park factors scaling a blended xwOBA, silently
scaling the walk term) cannot occur, because no park factor is applied. Phase 4's
assertion "that no park factor is applied after this layer" is vacuously true
today and would need building from nothing.

The absence is itself worth noting: the model runs park-neutral, and Coors vs.
a pitcher's park is currently unmodelled. That is a separate scope question, not
a bug, and not this work item.

**3. Lineup aggregation weighting.**

PA-weighted, by **batting-order slot**, not by playing time.
`lineup_weight()` (`build_site.py:1587`) returns `slot_pa_weights()` (`:1576`)
mapping order → `LINEUP_SLOT_PA` (`:73-74`), consumed by `wmean` (`:1565`).
`USE_SLOT_PA_WEIGHTS = True` (`:75`), `USE_WEIGHTED = True` (`:1428`).
Falls back to season `BBE` volume only when the batting order is unusable.

This was the v4 change. Both weighted and equal means are stored
(`opp_*_wmean`, `opp_*_mean`); the weighted one is selected.

**Units mismatch with the module.** `LINEUP_SLOT_PA` is 4.61 → 3.76, *full-game*
PA per slot. `PA_VS_SP` (PDF l. 27) is 3.20 → 2.40, *vs-the-starter* PA per slot.
Different quantities. The brief's Phase 4 note about replacing `PA_VS_SP` with an
empirical table should not be read as "reuse `LINEUP_SLOT_PA`" — that would
inflate the starter-phase PA count by ~44%.

## Bug candidates found, filed here, not fixed

Per deliverable 6, neither is touched.

**K-1. `lg_K%` is the all-pitcher league rate, used as the reference for a
starter.** `build_site.py:2866-2869` documents the provenance: league K% comes
from the PA-weighted batter leaderboard, justified as symmetric because "every
batter K/BB is a pitcher K/BB". That symmetry argument is correct and gives the
*all-pitcher* rate. `mx_K%` divides a **starter's** K% by it. Bullpen K% runs
above starter K%, so the SP-vs-batter league rate is lower than the all-pitcher
rate, and `mx_K% = B·P/L` is biased high for every starter by that ratio.

This is the exact bias the module's own docstring warns about (PDF l. 7:
"lambda is the SP-vs-batter league K%, NOT all-pitcher") and what the brief's
Phase 2 item 2 asks to check — and it is already present, independent of this
layer. Display-only: `mx_K%` and `edge_K%` never reach `xw_net`. Magnitude is
not quantified here because `.savant_cache/` is gitignored and keyed by slate
date; measuring it requires a live pull and would be reported against today's
slate only, never backfilled (see `CLAUDE.md` §No-lookahead).

**K-2. `K − BB%` composite on the SP card.** `build_site.py:2873` and `:3339`
compute `pit_K% − pit_BB%` for a percentile bar. The brief prohibits a K-BB%
composite "anywhere in this layer" on the grounds that it is the wrong quantity
for both BBE% and run value. The prohibition is about the layer, and this is a
pre-existing display element outside it — recorded so the two are not later
confused, not proposed for change.

## Measured: does K% carry information the blended number lacks?

`k_interaction_probe.py` answers the one question that decides whether any of
this is worth building. It runs on committed `data/leans_*_xw.csv` — no Savant
fetch, no `.savant_cache/` dependency, no historical reconstruction — and moves
nothing. `python k_interaction_probe.py`, 779 game-sides / 377 games:

```
sensitivity  d(xwOBA)/d(K%) per 1pp:
   xwOBAcon=0.340  -3.07 pts     xwOBAcon=0.370  -3.34 pts
   xwOBAcon=0.400  -3.62 pts

DECIDING MEASUREMENT -- lineup K% spread at fixed blended xwOBA (n=779):
   raw sd of lineup K%           2.177 pp
   residual sd, xwOBA held fixed 2.177 pp
   r(K%, xwOBA) = -0.0180   variance explained 0.0%

lean impact, shipped B*P/L vs K%-interacted (377 games):
   median |xw_net|  A 24.84 pts   B 23.32 pts
   corr 0.99675   median |shift| 1.91 pts   sign flips 2.39%
      |xw_net_A| <.005 (noise)  n=  44  flips 18.18%
      |xw_net_A| .005-.0186     n= 106  flips  0.94%
      |xw_net_A| .0186-.047     n= 154  flips  0.00%
      |xw_net_A| >.047          n=  73  flips  0.00%
```

Two findings, and they point opposite ways.

**K% is orthogonal to blended xwOBA at lineup level.** `r = -0.018`; the
residual sd equals the raw sd to three decimals. Holding the number the model
already has fixed removes *none* of the variation in lineup K%. The mechanism is
the power/whiff tradeoff cancelling at composite level — high-K lineups carry
higher xwOBAcon, and the two effects offset almost exactly in the blend. So the
premise behind a K% layer is sound: there is a full 2.18pp of lineup K%
variation the shipped model cannot see, and at −3.3 pts per 1pp that is not a
rounding error.

**Routing it through log5 nonetheless barely moves the lean.** Correlation
0.997, median shift 1.91 pts against a median lean of 24.84. Every flip above
`|xw_net| = .0186` is absent; the .005–.0186 band flips once in 106. The flips
concentrate where the lean was already noise (18% of the 44 games under .005,
which is close to what coin-flipping a near-zero net would give).

Both are true because `B·P/L` is *already* a multiplicative interaction, and
`log5(p_b, p_p, λ) ≈ p_b·p_p/λ` to first order. The information K% carries is
real and independent; the *interaction form* is not what was missing. This is
the distinction the brief's Phase 5 item 2 asks about — larger leans as signal
vs. double-counting — resolved before any integration, and it says the layer as
specified buys 1.91 pts of movement for four new dependencies.

The honest reading is that the promising direction is not the log5 interaction
at all. It is that lineup K% is an orthogonal axis the model currently ignores
entirely — which is an argument for decomposing the batter branch, not for
multiplying an opportunity term onto it.

## What would have to be true to proceed

Not a plan, and not authorization — the conditions a revised brief would need to
answer:

1. A source for batter and pitcher **xwOBAcon**, added to the Savant fetch.
2. A decision on the **pitcher contact branch**, which `evaluate_lineup` has no
   slot for. Without it the layer deletes signal the current lean depends on.
3. λ defined as the **SP-vs-batter** league K%, which requires a split this
   pipeline does not currently fetch (see K-1).
4. A statement of whether this is a *replacement* for the blended-xwOBA batter
   branch or a multiplier on top of it. If the latter, the double-count above
   applies and `docs/pitch_mix_theory.md` already records why that is the
   weaker form.

Until 1-4 are settled, Phase 1's tests would be testing a module against inputs
this repository cannot produce.

The measurement above reprioritises these. Condition 1 (an xwOBAcon source) is
no longer the blocker it looked like — the probe inverts for it from rates the
slate CSVs already carry, and that inversion is round-trip tested. Condition 4
is now answered by evidence rather than by argument: multiply-on-top buys 1.91
pts, so if this is built it is built as a replacement.

**Third premise correction, found while testing.** The brief asks to assert
`bbe_share` has no negatives over K% ∈ [0,.60] × BB% ∈ [0,.30] × HBP% ∈ [0,.05],
calling that "the specific failure the multiplicative form exists to prevent."
On that grid the subtractive `1 - k - bb - hbp` never goes negative either — the
corner sums to 0.95, bottoming out at +0.05. Going negative needs
`k + bb + hbp > 1`, outside the stated range and outside anything baseball
produces. The multiplicative form remains the right choice because it is closed
under [0,1] with no range precondition to remember, but it is chosen for
robustness, not because the cited failure occurs. Recorded in
`tests/test_k_interaction_probe.py::test_brief_grid_does_not_actually_reach_the_negative_corner`.

## Model-tag consequence

None. Nothing here changes prediction math, so `MODEL_TAG` stays
`xw+plat_consol_v10` and no record or scale family is opened. The brief's
proposed `v9_kint` tag (Phase 4) is also stale against the current model —
v10 shipped 2026-07-28; any future tag would branch from v10, not v9.
