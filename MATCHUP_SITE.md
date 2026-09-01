# MLB matchup-leans static site

A render-free static site that publishes daily MLB probable-pitcher vs
opponent-lineup **leans** (Statcast xwOBA + platoon-OPS), built from the
`Shrunk_mlb_matchup_render_consolidated` Colab notebook and deployed to GitHub
Pages on a schedule.

It pulls everything through keyless APIs (no browser, no secrets):

- **MLB StatsAPI** — slate, probables, rosters, bio, vL/vR splits, league baselines
- **Baseball Savant `gf?game_pk=`** — posted lineups (JSON)
- **Baseball Savant CSV leaderboards** — custom (xwOBA/xBA/xSLG/EV/LA/HardHit/K/BB) + batted-ball, cached once per day

The model uses a phase-specific multiplicative-ratio matchup anchored on league
average (`M = B·P/L`, additive for EV/LA), with `edge = M − L` as the signal.
The starter phase uses the lineup adjusted for the probable starter's hand; the
bullpen phase currently uses the neutral lineup. This is a relative-rate
heuristic, not the probability-odds form of log5. The platoon lens regresses
each side's vs-hand OPS toward an
overall×league-platoon prior and is reliability-gated. Lean is xwOBA-driven;
the platoon lens is still computed and graded into the ledger for auditing but
is no longer surfaced on the cards (the display is xwOBA-only).

---

## The current model — `xw+plat_consol_v12`

**Read this section for what the code does today.** Everything below it is a
changelog: each version note describes a *delta* against its predecessor, and
several of those deltas have since been superseded or reverted. Where a version
note and this section disagree, this section and the source are right.

Every active batter, probable-starter, reliever/bullpen, league prior, team
backfill, percentile, and optional pitch-mix cell is Statcast **xwOBA**. v11
reverted the metric after the wOBA forward test (wOBA v1–v5).

**Read the revert for what it is.** No measurement chose it. The paired shadow
arm — the surface built to answer the metric question — reports
`d_corr +0.008` with a 95% CI of `[-0.108, +0.128]` over 68 games; it does not
separate. The wOBA lineage's 63-76 record is real and is not evidence, because
always-home ran .604 over those same rows against .515 over the xwOBA ones. It
was an operator decision taken with the CIs on the table. The arm now runs
wOBA and keeps accumulating a paired answer.

Shrinkage: the pseudo-sample is back to **`K = 100`** and the target is back to
a **population centre**. Both reverts, both against their own evidence, both
deliberate — `reliever_shrink_probe` fits `K` three ways on n=53,464 and every
interval excludes 100, and the personal-prior probe reports +7% to +25%
out-of-sample MSE gains with CIs excluding zero. Neither finding is disputed;
they were overridden. Both are also **wOBA-denominated**, and the revert put the
build back on xwOBA: the probe fits `K` from StatsAPI box-line wOBA, and the
frozen priors are wOBA files an xwOBA build refuses. `K = σ²/τ²` moves with the
metric — xwOBA is near enough wOBA's conditional expectation, so its per-BF `σ²`
is strictly smaller and the implied `K` is smaller, i.e. nearer the 100 shipped.
That makes `K = 100` **unmeasured under v11 rather than overridden**, and it is
not re-measurable here (no xwOBA in StatsAPI; a per-past-date Savant pull is
lookahead). See `_RECORD_FAMILIES` under `xw+plat_consol_v11`. The one forward reading that points the other way is the
lineup component's correlation with its realised phase (+0.124 over the 194
v9/v10 side-games against −0.004 over v5's 220), and bootstrapped that
difference is `+0.128, CI [-0.054, +0.301]` — it does not separate either.

**Two post-v10 changes survive the revert**, because their evidence is
independent of metric, `K` and target:

- **v3's relief-pool target.** Relievers still shrink toward the relief pool's
  own unweighted centre, not the league PA-weighted batter rate. Measured
  before it shipped: the league batter centre is the centre of no subpool, and
  the relief pool sits 0.0102 below it. That bias exists at any `K`; `K` only
  decides how much of it lands. Reverting it would restore a measured bias.
- **v5's abstention.** A game whose starter has no leaderboard line publishes
  no lean. The input is unmeasured under xwOBA at `K=100` exactly as it was
  under wOBA at `K=400`.

The starter platoon prior is also v2's and also kept: a handedness-specific,
exposure-centred table with a conservative 0.021 strong-side/weak-side gap,
replacing the symmetric ±0.010 that every lineage through wOBA v1 used. It is
a construction fix rather than a fit — a hitter's season line is already a
platoon blend at his real exposure, so treating it as the midpoint of a 50/50
mix is arithmetically wrong whatever rate is inside it.

Everything from v6–v10 is structurally untouched: the expected-IP
starter/bullpen blend, the starter/bullpen phase split, and the PA-share phase
weighting. v12 calibrates the expected-IP estimator that feeds the phase
weight — see §"v12" — but the construction around it is v10's.

v11 starts a clean **record** family — it changes which games are decided and
moves every lean a little — while joining the v8/v9/v10 **scale** family, since
its three differences from v10 are each scale-preserving on a precedent already
set. That scale half is argued rather than measured (no-lookahead means a past
slate cannot be rebuilt to check it); the falsifier is named in
`_SCALE_FAMILIES`, and has since been run and closed without firing — the
share stands. Legacy dump/ledger columns such as `xw_net` remain a
compatibility schema and every new row carries `model_metric` explicitly —
under v11 the schema and the statistic agree again, which makes it *easier* to
mistake a key name for provenance, not harder.

The frozen `data/woba_priors_*.csv` stay wOBA-denominated and are not read by
this build: `player_prior_history()` refuses to serve them to a non-wOBA model,
so `PLAYER_PRIORS=1` alone does not restore v4 under v11.

The short-lived `split+plat_consol_v1` experiment changed the pitching side
back to xwOBA after treating a pitcher's weak cross-season residual correlation
as evidence against full wOBA. That test followed individual pitchers, while
park and defense are team context and both team membership and defensive
personnel change between seasons. It therefore could not rule out useful
same-season context. Full wOBA was restored before any split row graded; the
split tag and dump remain recognised only as immutable history, isolated from
both the wOBA record and scale families.

For one side of one game — the pitching side's staff against the opposing
lineup — the build computes:

**1. Lineup.** Posted Savant lineup is authoritative. Missing slots are filled
from the active-roster top-PA pool (`posted` / `partial_filled` / `projected`);
a posted hitter absent from the season leaderboard keeps his slot and receives
the team's PA-weighted xwOBA (`MODEL_RATE_TEAM_BACKFILL_COL`, named for the
active metric).

**2. Shrinkage.** Every hitter's season xwOBA is regressed toward the league
prior by his PA with a fixed pseudo-sample, `x* = (n·x + 100·prior)/(n+100)`
(`XWOBA_SHRINK_K = 100`). A deviation keeps `n/(n+100)` of its raw size: 50% at
100 PA, 75% at 300, 86% at 600 — so at this K the observation dominates and the
prior is a floor for thin samples rather than most of the number. That is the
substantive difference from `K = 400`, and it is why v4's personal priors are
off: at `K = 100` the target carries far less weight, which is one of the two
things the fit disagrees with (`reliever_shrink_probe` excludes 100 on all
three arms). `prior` is the population centre — the PA-weighted league rate for
batters and starters, the relief pool's own unweighted centre for relievers. A
team-backfilled hitter is already an aggregate and is **not** shrunk again.

**3. Two lineup composites**, both weighted by expected PA per batting-order
slot (`LINEUP_SLOT_PA`, 4.61 leadoff → 3.76 for the 9-hole):

- `B_0` — the neutral composite, taken before any handedness term.
- `B_SP` — after `PLATOON_XWOBA_OFFSETS` is applied to each *one-sided* hitter:

  | Batter | vs LHP | vs RHP |
  |---|---:|---:|
  | LHB | −0.016 | +0.005 |
  | RHB | +0.015 | −0.006 |
  | Switch/unknown | 0 | 0 |

  Each one-sided row preserves a 0.021 platoon gap. The unequal offsets centre
  the season blend at approximately 76.2% advantage exposure for LHB and 28.6%
  for RHB instead of assuming a 50/50 schedule. Switch hitters get **0** because
  their season line already is their advantage-state number, and remain marked ◆.
- `platoon_delta_sp = B_SP − B_0`.

**4. Two pitching inputs**, each shrunk with the same `K = 100`, each toward
its population centre:

- `P_SP` — the probable starter's season xwOBA-allowed, shrunk by BF toward the
  league centre. A starter with **no leaderboard line at all** has no rate
  and no BF, so the shrink returns the prior unchanged and the card would
  otherwise publish a defaulted number that reads like a measured one. Each side therefore
  carries `starter_rate_basis` (`measured` / `prior_only`) and
  `starter_rate_bf`, dumped and ledgered as `sp_rate_basis_*` / `sp_rate_bf_*`,
  and a `prior only` badge renders beside the starter on the card. **Since v5
  that side abstains**: the rate is still recorded, but the game publishes no
  lean (see §8). Measured over the 403 side-games in the committed dumps, 6
  (1.5%) published a starter rate equal to the league prior to full float
  precision, each with a null `K%` — no leaderboard row.
- `P_BP` — a role-filtered bullpen pool: active roster minus the probable,
  keeping pitchers with start share ≤ `0.35` and ≤ `3.0` IP per appearance
  (loose enough to retain bulk relievers, tight enough to drop rotation arms).
  Each reliever is shrunk individually toward the **relief pool's unweighted
  centre** (`relief_pool_prior`) rather than the league batter rate, which sits
  ~0.010 above it — v3's half that v11 keeps — then averaged by estimated
  relief workload (`team BF × (1 − start share)`). Needs ≥ 3 qualifying pitchers
  (`BULLPEN_MIN_PITCHERS`); below that the league baseline is the fallback
  target. The centre is derived every build from the same Savant rates the pool
  is built from, never frozen as a literal.

**5. Expected starter workload.** A normal starter blends his last-five start
average with his season IP/start, regressed toward `SP_IP_PRIOR = 5.2` and
clipped to `[3.0, 7.0]`. An opener — repeated short starts, or a reliever
spot-start profile — is clipped to `[0.33, 3.0]` instead.

**6. Sequential matchup.** The two phases are computed separately and averaged
on their share of **plate appearances**, using BF/IP measured from the same
season role call:

```
M_SP = B_SP · P_SP / L          M_BP = B_0 · P_BP / L

q    = (IP_SP·r_SP) / (IP_SP·r_SP + (9 − IP_SP)·r_BP)
M    = q·M_SP + (1 − q)·M_BP            edge = M − L
```

Both phase values are per-PA rates, so an innings share would be the wrong
denominator. When either BF/IP is unavailable the weight degrades continuously
to the innings share `IP_SP / 9` — the identical number whenever `r_SP = r_BP`,
so there is no threshold.

**7. The lean.** Legacy-schema field `xw_net = home_off_edge − away_off_edge`
(despite the name, it contains a wOBA delta in this lineage; the away-SP row
carries the *home* offense's edge). Its sign is the lean; an exact zero is an
**abstention**, not a home pick. Full precision is retained through the
decision — three decimals are a display format only, and a nonzero delta too
small to show renders `<0.001`.

**8. Degradation and abstention.** Two cases, one mechanism — the side's edge
goes undefined, so the game publishes no lean:

- If either side lacks a valid bullpen aggregate, the starter phase is still
  shown but the full-game lean is **suppressed** and the side is marked
  `pitching_basis=starter_only_no_fullgame_lean`. v10 never treats a probable
  starter as nine innings.
- **v5:** if a side's starter has no leaderboard line, his rate is his prior
  rather than a measurement (`starter_rate_basis=prior_only`), so the game
  **abstains** and the side is marked
  `pitching_basis=starter_unmeasured_no_lean`. `starter_xwOBA` still records
  the prior that was used — blanking it would hide the abstention's own cause —
  and the bullpen phase, which *was* measured, survives. The card reads
  `starter unmeasured; no lean` beside the `prior only` badge.

An abstained game is graded (it was played) but undecided, so it carries no
`xw_lean` and no W/L. That is the first mechanism here that produces such a
row: v7's exact-zero abstention has never fired at full precision. Counts that
mix graded rows with decided ones now say which is which — `ledger_report.txt`
prints the abstained count, and on the grades page the row reads `no lean`, the
Graded tile states how many rows were abstained, and **the baseline controls are
scored on the decided rows only**. That last one is the part a control gets
wrong by default: it needs no lean to score a game, so over the full graded
frame it publishes a baseline over more games than the record beside it.

Not in the lean: the platoon-OPS lens (computed, graded, ledger-only) and the
pitch-mix shadow arm (off by default). Both are described below.

---

## Version history

Each note below is the delta that version introduced, kept for provenance —
ledger rows are immutable and a row's `model_tag` is only interpretable against
the version note that produced it.

### v12 — expected starter IP calibrated against its own actuals

`expected_pitcher_ip` is **over-dispersed**. Regressing actual starter IP on the
pregame estimate over 586 backfilled side-games gives slope **+0.735 ± 0.048**,
5.5 se below 1.0: starters predicted short go longer than predicted and
starters predicted deep go shorter. Bias is +0.086 IP and not significant —
this is a spread problem, not a level one. It reaches the lean because `q`, the
starter's share of plate appearances, is a function of this number.

**The fix is a per-build refit, never a fitted literal.** `sp_ip_calibration()`
regresses actual on raw predicted across the ledger's completed games on every
build; no `a + b·pred` appears in the source. The slope is plausibly seasonal —
pitch counts climb early and clubs get cautious in September — so a September
build has to fit September's slope, and the ledger accumulates the actuals to
let it.

The correction is shrunk toward the **identity map** by sample size:

```
cal(p) = w·(a + b·p) + (1 − w)·p,    w = n / (n + SP_IP_CALIBRATION_K)
```

`n = 0` returns `p` exactly, so an empty ledger and a full one are two ends of
one expression rather than two branches with a gate between them. `K = 50`
side-games was chosen by walk-forward benchmark (fit on every prior slate,
score the next): calibration beats no-calibration by +4.1% / +4.0% / +3.8%
out-of-sample IP MSE at `K = 25 / 50 / 100`, against +3.4% at `K = 0`, so the
shrinkage earns its place early. Bootstrapped over slates, `K = 50` is the
argmin most often and every candidate's CI excludes zero. The curve is flat
from 10 to 100 — what is distinguishable is calibrated from uncalibrated, not
25 from 50.

**The dump and ledger now carry `expected_sp_ip_raw_*`.** The published
`expected_sp_ip` is calibrated, so a fit that regressed against it would
compound the correction on every build and pull the estimator toward the mean
without limit. The fit reads the raw column where it exists and falls back to
the published one for pre-v12 rows, which are raw by definition and are the
entire sample on the first build after the bump.

**Families.** New record family, and the argument is unusual: on decision
equivalence it would have *shared* with v11 — 1 lean flips in 254, mean
|Δ net| 0.00067 against a median |xw_net| of 0.01694 — but v11 had no graded
rows, so the reset cost nothing and a clean line for a lean-path change was
free. Shared scale family with v8–v11, measured and on v10's precedent: `q` is
a convex weight between the same two phases, so the delta keeps its units.

### v11 — revert to xwOBA, K=100 and population targets

`xw+plat_consol_v11` returns the three knobs the wOBA forward test moved and
keeps the three changes whose evidence did not depend on them.

**Reverted, and none of the three by a measurement that favoured the revert.**
The metric goes back to Savant xwOBA; the paired shadow arm built to settle
that question reports `d_corr +0.008, CI [-0.108, +0.128]` over 68 games and
does not separate, so the arm swaps sides and keeps measuring. `K` goes back to
100 against `reliever_shrink_probe`'s three-way fit (391 [293,519] walk-forward
on n=53,464; 600/384/577 within-season), every interval of which excludes 100 —
a fit denominated in wOBA, and so about the constant this build had rather than
the one it now has (above, and `_RECORD_FAMILIES`).
The shrinkage target goes back to the population centre against
`player_prior_probe`'s +7%–25% out-of-sample gains; the one forward reading
that points the other way — the lineup component's correlation with its
realised phase, +0.124 over the 194 v9/v10 side-games against −0.004 over v5's
220 — does not separate either (`+0.128, CI [-0.054, +0.301]`).

These were operator decisions taken with the numbers on the table. The code
comments at each site say so, and they should keep saying so: the failure mode
this repo has repeatedly hit is a later reader finding a reverted constant and
inferring that something measured it.

**Kept:** v2's exposure-centred platoon offsets (arithmetic, not a fit), v3's
relief-pool shrink target (a measured 0.0102 bias that exists at any `K`), and
v5's abstention (an unmeasured input is unmeasured under any metric).
Everything from v6–v10 is untouched.

**Families.** New record family — abstention changes which games are decided
and the platoon and relief terms move every lean. Shared scale family with
v8/v9/v10 — same metric, same `K`, same batter and starter target, and the
three deltas from v10 are each scale-preserving on a precedent already in the
table (platoon centring moved median |net| 0.7% at v1→v2; abstention left
v4/v5 quantiles bit-identical; a uniform re-centring cancels in a difference).
That half is **argued, not measured** — no-lookahead means a past slate cannot
be rebuilt to check it — so the falsifier is written into `_SCALE_FAMILIES`:
compare median |xw_net| on the first graded v11 rows against the v9/v10 pool
and split the family if it moved. Sharing gives the strength cutoffs a 99-row
pool on day one instead of a ninth restart.

That falsifier has now been run and **closed without firing**: v11 graded no
rows, so it fell to v12, where it does not separate and — more to the point —
cannot, because between-slate noise in the statistic is 7x the largest effect
the change can produce. The reads and the power argument live in
`_SCALE_FAMILIES` and in CLAUDE.md's "Instrumented and waiting"; do not
re-derive them here.

**Two collateral pins.** The frozen `data/woba_priors_*.csv` are wOBA and stay
wOBA: `priors_snapshot.RATE_COL` is pinned rather than following the build, and
`player_prior_history()` refuses to serve them to a non-wOBA model — so
`PLAYER_PRIORS=1` alone does nothing under v11. And the per-slate dump suffix
follows the metric again (`leans_<date>_xw.csv`), read from
`build_site.DUMP_SUFFIX`, because a suffix outside `grade_leans`' globs would
lose a slate's pregame rows silently and unrecoverably.

### wOBA v5 — abstain on an unmeasured starter

`woba+plat_consol_v5` publishes no lean for a game in which either side's
probable has no Savant line. Such a starter reaches the model with no rate and
no BF, so shrinkage returns his prior unchanged; under v4 that prior is his own
regressed 2023–2025 history, which makes the defaulted value indistinguishable
from a measured one in the output and unrecoverable from the ledger afterwards.
The v4 instrumentation (`starter_rate_basis`, `starter_rate_bf`, ledgered as
`sp_rate_basis_*` / `sp_rate_bf_*`) is what this reads.

Nothing about a surviving prediction changes: the abstention only nulls an
edge. **New `RECORD_TAGS` family, `SCALE_TAGS` shared with v4** — and both
halves are measured, not inherited. The record isolates because the *decided
set* changes: over the 185 ledger rows that carried starter/bullpen
instrumentation at the bump, 6 (3.2%) carried a prior-only starter on one side
and would now publish nothing. The scale is shared because every surviving
`|xw_net|` is bit-identical and dropping those 6 moves no cutoff — pooled
p33/p80 0.0090 / 0.0283 with them and 0.0090 / 0.0283 without, and
0.0127 / 0.0343 either way inside v9/v10. That is the v6 precedent (new
prediction family, inherited scale) applied to a filter.

Those 6 games graded 3-3 against 96-82 (.539) on the rest. At n=6 that is
incidence, not evidence they were bad picks — and the case for abstaining does
not rest on it. The input was never measured either way.

### wOBA v4 — personal shrinkage priors

`woba+plat_consol_v4` changes what a rate is regressed *toward*. v3 settled how
hard (`K = 400`); this settles the target. Until v4 every batter, starter and
reliever was pulled toward a population centre, so at `K = 400` a career .360
hitter and a career .290 hitter with equal samples published within a hair of
each other — the prior supplies ~70% of a median reliever's rate and about half
of a 400-PA batter's, so the target *is* most of the number.

`player_priors.prior_for` replaces that centre with

    π = (H·θ_hist + C·μ) / (H + C)

where `θ_hist` is the player's recency-weighted 2023–2025 Savant wOBA carried as
a **deviation from the pool centre he earned it against**, and `μ` is the same
population centre the build used before. `H = 0` returns `μ` exactly, so rookies
and call-ups are unchanged and need no branch. Storing the deviation rather than
the level is what keeps a 2023 rate comparable to a 2025 one; the per-season
centres live in `data/woba_prior_centres.csv`.

Measured out of sample by `player_prior_probe.py` at this K. The early-season
split — the first month, the case the build faces every April and where the
prior carries most of the weight — reports BF-weighted out-of-sample MSE gains
of **BAT +25.35%** [+18.30, +32.84], **RP +22.28%** [+16.36, +28.68] and
**SP +16.38%** [+8.84, +25.86] against the shipped centre. The balanced
mid-season split reports +7.33%, +5.91% and +2.83%, all with paired CIs
excluding zero, and `C` fitted on 2024 holds on 2025 and vice versa for batters
and relievers.

Two negative results are load-bearing. The unregressed career rate (`C = 0`)
was measured and **loses** on two of three populations (BAT −11.46%,
RP −24.29%) — a 70-PA career would otherwise supply most of a published rate.
And role-separated history buys nothing (SP +16.38 vs +16.85 role-blind), so
the build passes `role=None`.

New `RECORD_TAGS` family and new `SCALE_TAGS` family, both argued rather than
inherited. The record half: early-season out-of-sample MSE moves 16–25% across
all three populations and the published rate of an established player moves by
roughly the size of the platoon term — these are different predictions. The
scale half is arithmetic, and it is v3's argument with the sign reversed:
shrinking toward a personal prior instead of one shared centre lets two players
with equal samples keep different centres, so `|xw_net|` **disperses** where v3
compressed it. Same wOBA units, materially different spread.

Turning `PLAYER_PRIORS=0` off restores v3 behaviour exactly, and so does an
empty `data/woba_priors_*.csv` — the population centre is the `H = 0` limit. A
missing snapshot degrades to that limit and logs, rather than failing the build.

### wOBA v3 — fitted shrinkage constant and relief-pool target

`woba+plat_consol_v3` makes two changes that had to ship together.

**`XWOBA_SHRINK_K` 100 → 400.** Fitted, not chosen. `reliever_shrink_probe.py`
measures K three independent ways and every one excludes 100 by a wide margin:

| estimator | K | interval |
|---|---:|---|
| walk-forward, season-to-date → next outing (n=53,464) | 391 | [293, 519] |
| within-season split, relievers | 600 | [355, 1400] |
| within-season split, batters | 384 | [302, 503] |
| within-season split, starters | 577 | [398, 894] |
| season pair (upper bound) | 669–1048 | — |

The walk-forward arm is the one that matches what this code does — a
season-to-date rate predicting tonight's innings, not one aggregate predicting
another — and it is better powered by two orders of magnitude. Its estimator
runs 10–13% high on synthetic logs of this shape, so its point estimate
corresponds to a true K nearer 350.

400 is one value for all three populations, **not** a role split. That was the
original hypothesis and the data refused it: relievers 600 against starters 577,
intervals overlapping almost entirely. Marcel's 2.4× role gap does not appear in
BF-denominated data. Do not read the third digit — weighted MSE is flat near its
minimum, so 350 and 450 are not distinguishable here. What is distinguishable is
100 from 400.

**Reliever target → the relief pool's own unweighted centre.** The league batter
centre is the centre of no subpool: over 223 role-filtered arms on 30 active
rosters the relief pool sits **0.0102 below** it. At `K = 400` that offset lands
almost intact in every bullpen number, which is why K and the target had to move
in the same commit — raising K against the old target would have doubled a known
bias instead of removing it. The centre is **unweighted**, because empirical
Bayes wants the centre of the population a member is drawn from; the pool's
usage-weighted centre sits 0.0116 the *other* side, because the good arms get
the innings, so shrinking toward it would over-correct and land further from the
truth than the shipped target was. The obvious fix is worse than the status quo;
this is the non-obvious one. It is derived every build from the same Savant
rates the pool is built from, never frozen.

New `RECORD_TAGS` family and new `SCALE_TAGS` family. The record half is argued:
a 4× K changes every shrunk batter, starter and reliever rate, and the target
move shifts every bullpen number by a further ~0.010 before weighting — not the
v10 situation, where a reweight flipped 0 of 14 leans and earned a shared line.
The scale half is arithmetic: quadrupling K compresses `|xw_net|`, since a
median batter at ~400 PA keeps 400/500 = 80% of his deviation at `K = 100` and
400/800 = 50% at `K = 400`. The wOBA v2 family had 15 graded rows, so the cost
of resetting was 15 games.

### wOBA v2 — exposure-centred starter platoon prior

`woba+plat_consol_v2` replaces the universal ±0.010 term with the four-cell
table in the current-model section. The hitter-side gap remains deliberately
conservative at 0.021, but the offsets now preserve the season wOBA at the
assumed handedness exposure mix. Switch and unknown hitters remain zero, and
the term still applies only to the starter phase after overall-wOBA shrinkage.

This changes prediction math and therefore starts a clean `RECORD_TAGS` family.
The primary metric and practical delta units are unchanged, so v1 and v2 share
one `SCALE_TAGS` family for lean-strength calibration.

On a same-input replay of the 2026-08-04 projected slate, v2 flipped **0 of 14**
pair-complete leans. Median absolute game-net movement was **0.000587** and the
maximum was **0.002602**; median `|xw_net|` moved only **0.7%**, from 0.016169 to
0.016282. That supports shared scale units, but one projected slate is too small
to declare the win/loss decisions equivalent, so the record remains isolated.

Historical model version `xw+plat_consol_v2` added:

- **Lineup partial fill** — valid posted Savant hitters are kept in order and
  only missing slots are filled from the active-roster top-PA pool
  (`posted` / `partial_filled` / `projected`); a per-side resolution audit is
  written to `data/lineup_resolution_audit.csv` each run.
- **Full-league platoon baselines** — league OPS cells come from the entire
  Savant hitter split population (~10–15 extra batched StatsAPI calls, ~+5 s)
  instead of the day's lineups, removing slate-dependent shrinkage priors.
- **Batted-ball league anchors** — BBE-weighted full-population baselines for
  GB/FB/LD/PU/Pull/Straight/Oppo.
- **Composition-weighted SP platoon OPS** — displayed SP OPS-allowed (and all
  platoon aggregates) are lineup-composition weighted rather than simple means
  over exposed handedness cells.

Model version `xw+plat_consol_v3` leaves the prediction math unchanged and adds
a hard pregame snapshot lock. Its performance results therefore append to the
v2 series; row tags still identify the audit regime. Each dump stores its
capture and scheduled-start timestamps, and the ledger rejects new or refreshed
rows captured at/after scheduled first pitch. Ledger identity is
`(game_pk, game_date)`, so a postponed game that keeps its MLB gamePk can be
recorded again on its make-up date.

Model version `xw+plat_consol_v4` re-weights the lineup composites:

- **Batting-order slot weighting** — the xwOBA lineup composite and every
  platoon-OPS aggregate are weighted by expected plate appearances *per
  batting-order slot* (leadoff ~4.61 PA/game → 9-hole ~3.76, `LINEUP_SLOT_PA`)
  instead of by each hitter's season volume (batted-ball events / handedness
  split PA). Slot weighting reflects tonight's in-game exposure rather than who
  has simply logged the most playing time. As shipped it fell back to the old
  season-volume weights wherever a batting order was unavailable
  (`USE_SLOT_PA_WEIGHTS`) — a branch that never ran, since `hitter_rows`
  assigns every lineup slot an order, and which was **deleted on 2026-08-06**
  after being measured as a no-op on frames from the real builder. The lineup
  composite is slot-weighted or, if an order ever went missing, an equal mean;
  it is never BBE-weighted. This changes the prediction math, so v4 starts a new
  `RECORD_TAGS` family and its games never mix with v2/v3 in the records.

Model version `xw+plat_consol_v5` adds xwOBA shrinkage on top of v4:

- **Empirical-Bayes xwOBA shrinkage** — before they drive the lean, each
  hitter's season xwOBA and the starter's season xwOBA-allowed are regressed
  toward the league xwOBA baseline by sample size, `x* = (n·x + K·prior)/(n+K)`
  (`shrink_xwoba`). Both sides share the league baseline as the prior. `K` is
  estimated by **method of moments per player pool** each build
  (`estimate_shrink_k`): sampling noise scales as `1/n`, so the gap between the
  unweighted and the PA-weighted dispersion of the leaderboard identifies the
  within-PA and between-player variance components, and `K = σ²/τ²` — no fixed
  per-PA constant. The estimate is clamped to a plausible PA band with a fixed
  fallback (`K_BAT_*` / `K_PIT_*`) and logged each run. Shrinkage touches only
  xwOBA (the lean stat); other columns and the raw per-hitter card values are
  untouched. This changes the prediction math, so v5 starts a new `RECORD_TAGS`
  family and its games never mix with v4 or v2/v3.

### Historical opener fallback (v5)

A probable pitcher whose recent starts average fewer than `OPENER_MAX_AVG_IP`
innings (over at least `OPENER_MIN_STARTS` starts) is treated as an **opener**:
his own Statcast line reflects only a handful of batters and is not
representative of the innings his club will actually pitch. For those sides the
xwOBA lean substitutes a **batters-faced-weighted aggregate of the club's
rostered pitching staff** (built from the pitcher custom leaderboard the build
already fetches, plus one active-roster call per opener club) for the opener's
own numbers; the swap happens in the lookup dicts, so the whole matchup
pipeline downstream uses the staff numbers. The aggregate carries the staff's
total batters faced as its sample size, so the xwOBA shrinkage step barely pulls
it toward league. If fewer than three staff pitchers appeared in the
leaderboard, the fallback was skipped and the opener's own line was kept.

The platoon lens was deliberately left untouched — an opener's tiny vL/vR split
already fails the reliability gate, so that lens abstains on its own. Openers
were rare, so this refinement stayed inside the `xw+plat_consol_v5` family;
every affected side was flagged in the ledger.

### Full-game pitching blend (v6)

Model version `xw+plat_consol_v6` replaces the starter-only xwOBA pitching
input—and v5's opener-only whole-staff substitution—with one consistent
nine-inning construction for every side:

1. Estimate the probable pitcher's expected innings from pre-slate game logs.
   A normal starter blends his last-five start average with his season
   IP/start; sparse histories regress toward `SP_IP_PRIOR`. An opener uses the
   short-start or recent-relief workload that caused his classification.
2. Build a **role-filtered bullpen pool** from the active roster plus one
   season pitching-role call per club. Pitchers qualify when no more than 35%
   of their appearances were starts and they average no more than 3.0 IP per
   appearance. This removes regular rotation starters while retaining long or
   bulk relievers in the pooled estimate.
3. Shrink every included reliever's Savant xwOBA independently by BF, then
   average those talents using estimated relief BF
   (`season BF × (1 − start share)`).
4. Complete nine innings:
   `P_game = (E[IP_SP]·SP_xwOBA + (9−E[IP_SP])·RP_xwOBA) / 9`.

v6 deliberately does **not** identify or assign innings to a specific bulk
follower; long/bulk arms contribute only through the bullpen pool. When a
normal side lacks a trustworthy relief pool, it degrades visibly to the
probable starter's value. For an opener only, v5's whole-staff aggregate
remains an audited emergency fallback so one inning is never treated as nine.

The starter card continues to show the probable pitcher's own shrunk xwOBA,
K−BB%, and xERA. The lean uses the blended xwOBA and the card states the
expected-IP + bullpen basis. Dumps and the ledger preserve the starter xwOBA,
bullpen xwOBA, expected innings, relief-pool size/BF, opener classification,
and pitching basis for audit.

This is a prediction-math change, so v6 starts a new `RECORD_TAGS` family.
The v2/v3, v4, and v5 rows remain immutable in the ledger; the Actions report
shows family history separately while the current-family fit uses v6 only.

### Centre-matched shrinkage moments (v7)

Model version `xw+plat_consol_v7` keeps v6's full-game pitching blend and
corrects the method-of-moments estimate of xwOBA shrinkage `K`. The league
xwOBA used later as the shrinkage target is PA-weighted, but the old estimator
centred both its PA-weighted and unweighted dispersions on that target. In a
player pool whose low-PA members have a different unweighted centre, the
unweighted moment therefore included a between-centre offset and overstated
the sampling component.

v7 centres each moment on its own population mean: the PA-weighted dispersion
uses the PA-weighted pool mean, and the unweighted dispersion uses the
unweighted pool mean. The estimator no longer consumes the shrinkage target;
`sigma²`, `tau²`, and `K = sigma²/tau²` are properties of the player pool.
The league xwOBA remains the target when each resulting player estimate is
shrunk. That target was never revisited — v8 retired the estimator this
correction lived in, so the between-centre offset stopped being measured
anywhere. `prior_population_centres` now logs it each build for every pool the
target is applied to; see the amendment in `docs/build_logic_validation.md`.

The same v7 draft also removes two downstream discretization artifacts:

- **Full precision through the decision.** League anchors, player/team
  aggregates, workload estimates, matchup values, and xwOBA/OPS edges retain
  their floating-point precision through lean selection. Three-decimal card
  formatting happens only for display; dumps and ledger model fields retain
  full precision. A nonzero delta too small for the card's three-place display
  is shown as `<0.001`, never as `0.000`.
- **A zero is an abstention.** An exact zero xwOBA or OPS delta produces no
  lean; it is not assigned to the home team. Its consensus is `NA` and grading
  leaves that lens blank.

Posted lineups are authoritative even when a call-up is absent from the season
Savant leaderboard. The player remains in his posted batting-order slot and
receives the active team's PA-weighted wOBA as a transparent fallback
(`wOBA_team_backfill`); because that value is already a team aggregate, it is
not run through player-level shrinkage again. Per-side backfill counts are
carried from `lineup_resolution_audit.csv` into the lean dump and ledger audit
columns. Other available player data, including StatsAPI bio and handedness
splits, continues through the normal path.

Together these changes alter prediction math and `|xw_net|` units, so v7 starts
new `RECORD_TAGS` and `SCALE_TAGS` families without rewriting older rows.

### Fixed xwOBA shrinkage (v8)

Model version `xw+plat_consol_v8` keeps the v7 matchup, precision, lineup, and
full-game pitching logic but replaces the per-build method-of-moments
shrinkage estimates with one fixed pseudo-sample:

`XWOBA_SHRINK_K = 100`

The same value applies to every batter, probable starter, and reliever before
the bullpen pool is usage-weighted. The shrinkage target and formula stay the
same:

`x* = (n·x + 100·prior)/(n+100)`.

An observed deviation from league average therefore retains `n/(n+100)` of
its raw size: 50% at 100 PA/BF, 75% at 300, and about 86% at 600. Removing the
daily estimator makes the transformation reproducible and preserves more of
the player-distribution tails than v7's typically larger separate constants.

This changes both prediction math and `|xw_net|` units, so v8 starts new
`RECORD_TAGS` and `SCALE_TAGS` families without rewriting historical rows.

### Sequential starter/bullpen matchup (v9)

Model version `xw+plat_consol_v9` preserves two shrunk lineup composites:

- `B_0 = opp_xwOBA_neutral`, before starter-handedness adjustments.
- `B_SP = opp_xwOBA_vs_sp`, after the one-sided-hitter ±0.010 adjustment.
- `platoon_delta_sp = B_SP − B_0`.

The starter and bullpen phases are calculated separately:

`M_SP = B_SP · P_SP / L`

`M_BP = B_0 · P_BP / L`

`M_sequential = q · M_SP + (1 − q) · M_BP`, where
`q = expected_sp_ip / 9`.

The role-filtered bullpen construction is unchanged: active roster, probable
starter excluded, rotation arms excluded by role thresholds, each reliever
shrunk before relief-workload aggregation. Bullpen handedness is neutral until
a defensible follower/handedness projection exists. The OPS platoon lens remains
diagnostic and never enters the xwOBA lean.

If either side lacks a valid role-filtered bullpen aggregate, its starter phase
is still calculated and displayed, but the game's full-game xwOBA lean is
suppressed. The side is marked
`pitching_basis=starter_only_no_fullgame_lean`; v9 never treats the probable as
a nine-inning fallback. The opener whole-staff emergency aggregate is removed
because it could include rotation pitchers and double-count the probable.

Daily snapshots persist the neutral and starter-adjusted lineup composites,
starter and bullpen xwOBA, expected innings and shares, each phase matchup, the
sequential matchup/edge, and the pitching basis.

v9 started a new `RECORD_TAGS` family, isolated from v8 — but it **shares v8's
`SCALE_TAGS` family**, because v9 differs from v8 by exactly one term,
`v9 − v8 = −(1 − q)·platoon_delta_sp·P_BP/L`, which leaves `|xw_net|` units
untouched. (An earlier draft of this section claimed v9 isolated *both*
families; the code never did that, and v10 later joined v9's record family too.
See the table in `CLAUDE.md` for the authoritative mapping.)

### PA-share phase weighting (v10)

Model version `xw+plat_consol_v10` keeps v9's two phases and changes only the
weight that averages them. Both `M_SP` and `M_BP` are **per-plate-appearance**
xwOBA rates, but v9 averaged them on a share of **innings**
(`q = expected_sp_ip / 9`). Those are different denominators. A pitcher who
allows more baserunners faces more batters per inning, so the innings share
systematically underweights the starter in exactly the games where he is the
one being hit.

v10 converts innings to plate appearances with a **measured** BF/IP for each
phase, taken from the same season role call the bullpen filter already uses
(`bf_per_ip` = season batters faced ÷ season innings; no on-base proxy stands
in for it):

`q = (E[IP_SP]·r_SP) / (E[IP_SP]·r_SP + (9 − E[IP_SP])·r_BP)`

The bullpen's `r_BP` is pooled over the same pitchers and the same usage
weights that produced the pool's xwOBA. When either rate is unavailable the
weight falls back to the innings share, which is the identical number whenever
`r_SP = r_BP` — the degradation is continuous, not a threshold. Endpoints are
preserved: nine starter innings still leaves the bullpen zero PAs.

`sp_bf_per_ip` and `bp_bf_per_ip` are persisted to the dumps and the ledger, so
a side's `sp_share` can be re-derived without a rebuild.

**Versioning.** This changes prediction math, so `MODEL_TAG` moves to v10 — but
both family questions are answered by measurement rather than reflex:

- **`SCALE_TAGS` — pooled with v8/v9.** The weight is a convex combination of
  the same two phases either way; `|xw_net|` units are unchanged.
- **`RECORD_TAGS` — pooled with v9.** Sweeping the starter/bullpen BF/IP ratio
  over 0.95–1.10 on the 2026-07-28 slate moves `xw_net` with sd 0.0004–0.0009
  against a median `|xw_net|` of 0.0268 (1.6–3.4%) and flips **0 of 12** leans.
  Only an implausible 1.20 ratio flips one, on a game whose `|xw_net|` is
  0.0013. v9 and v10 agree on the decision, so they share one win-loss line.

**Honest limit.** The sweep is a bound over plausible BF/IP ratios, not a
measurement of the realised ratio distribution — the build environment could
not reach StatsAPI to fetch live role lines. The correction is justified on
dimensional grounds; with 11 graded rows in the family, the ledger cannot and
will not resolve whether it improves accuracy. The persisted `*_bf_per_ip`
columns make the realised ratios auditable from the first live build.

**Audited against the first live build (2026-07-28).** That build reached
StatsAPI and persisted real role lines for 14 of 16 games, so the sweep above
can now be replaced with a measurement of the realised ratios:

| quantity | swept bound | realised |
|---|---|---|
| starter/bullpen BF/IP ratio | 0.95–1.10 assumed | 0.913 – 1.082 (median 0.986, n=28 sides) |
| sd of `xw_net` shift, v10 − v9 | 0.0004–0.0009 | 0.00038 |
| shift vs median `\|xw_net\|` | 1.6–3.4% | 2.3% (median `\|xw_net\|` 0.0164) |
| lean flips | 0 of 12 | **0 of 14** |

The realised ratio sits just below 1 rather than at it — a starter faces
slightly *fewer* batters per inning than his own bullpen does — and 4 of 28
sides fall below the swept band's 0.95 floor, so the band was marginally
optimistic at the low end. The conclusion is unchanged and now rests on
measured rates instead of an assumed range: v9 and v10 agree on every
decision, so `RECORD_TAGS` pooling them is correct.

**Tag provenance caveat for these rows.** Those 14 games are stamped
`xw+plat_consol_v9` in the ledger, not v10, because
`.github/workflows/build.yml` pinned `MODEL_TAG` job-level and the v10 commit
updated only the module defaults. CI therefore ran v10's PA-share weighting
under a v9 tag. The workflow no longer pins the tags — the modules are the
single source of truth — but the affected rows are immutable and stay v9.
They are identifiable by carrying a non-null `sp_bf_per_ip`, which no genuine
v9 row has. Because the two tags share a `RECORD_TAGS` family this costs the
records nothing; it is recorded here so the ledger's lineage column is not
read as more precise than it is.

`python compare_v8_v9.py` recalculates the v8 shadow and v9 sequential formula
from identical persisted inputs, then reports lean flips, `xw_net` changes,
expected-IP buckets, openers, bullpen-heavy games, market disagreement, and
flat-stake ROI when settled ledger/closing-price data exists. Pre-v9 snapshots
are explicitly ineligible because they did not persist `B_0`; the utility never
rebuilds an old slate from current Savant data.

### Pitch-mix shadow arm (not in any lean)

`pitch_arsenal.py` builds a candidate replacement for the batter side of the
starter phase: the opposing lineup re-weighted by the starter's arsenal, using
each hitter's xwOBA by pitch type. It is **shadow-only and off by default**
(`PITCH_MIX_SHADOW=1` to enable). With the flag on it writes extra columns and
changes nothing else, so builds with and without it produce identical leans and
no `MODEL_TAG` bump is involved either way.

Three corrections separate the implementation from the obvious version of the
idea, and each is enforced by a test:

- **PA share, not usage share.** Usage is per pitch; the batter values are per
  plate appearance, and putaway pitches end plate appearances well above their
  usage rate. The weights are the starter's PA share per pitch type; usage is a
  flagged fallback (`mix_basis`).
- **League-relative per pitch type.** League xwOBA against a slider sits far
  below league xwOBA against a four-seamer, so a raw weighted average scores a
  slider-heavy arsenal low on mix alone — and the starter's own xwOBA-allowed,
  which the matchup already multiplies in, is low *because* of that mix.
  Everything is a ratio to league-at-that-pitch-type, normalised so a
  league-average hitter returns exactly 1.0 against any arsenal.
- **It supplements the platoon term.** Savant's batter arsenal splits pool both
  pitcher hands, so `platoon_xwoba_offset` still applies. The multiplier is a
  starter-phase quantity only: a bullpen is not one arsenal.

Each (batter, pitch type) cell is regressed toward that hitter's own overall
relative level rather than toward league, so what survives is pitch-specific
skill and not general hitting ability the composite already carries. A hitter
with no cells lands on a multiplier of exactly 1.0 — the degradation is
continuous, with no coverage threshold.

**Why it is dark.** *(This whole estimate is stale and is kept as the shape of
the argument, not as current numbers — see the note below it.)* At `K = 100`,
residual cell noise moves a game delta by about 0.013 xwOBA against a median
`|xw_net|` of **0.0188** over the 28 scale-family rows as of 2026-07-28 (all
tagged v9; the family's v8 entry has never matched a ledger row) — **69%** of a
typical lean, injected as noise. Holding that to ~19% needs `K ≈ 600`, at which
point even an 80-point raw cell deviation moves a lineup composite by 0.0008.
Whether the arm can escape that squeeze depends on the true dispersion of
batter × pitch-type skill, which has not been measured here.

**Both halves of that ratio have since moved, in opposite directions, and it has
not been recomputed.** The parenthetical it used to carry — *recompute the
median on any `SCALE_TAGS` change, this ratio is the entire gate* — has fired
four times since (wOBA v1, v2, then v3 and v4 each isolating) and nobody
recomputed. The numerator moved because `K` is now **400**, not 100, which cuts
residual cell noise; the denominator moved because v3 compressed `|xw_net|` and
v4 dispersed it again. The current `SCALE_TAGS` family is v4 alone and holds
**11 rows, none of them graded** — which does not block the median, since
`lean_strength_scale()` counts pending rows too (`|xw_net|` is pregame and needs
no outcome), but does make it noise: median `|xw_net|` 0.0062 over those 11, as
of 2026-08-06. Do not
read 69% as today's figure, and do not read `K ≈ 600` as today's requirement —
the build already sits two-thirds of the way there. Recompute both before this
arm is argued either way.

`python pitch_arsenal_probe.py` is that measurement: it decomposes the observed
cell dispersion into sampling and between-player components (implying a `K`),
estimates year-over-year reliability of the deviation, and prints both against
the noise budget the arm has to clear. **Run it before turning the flag on.**

**No-lookahead.** This cannot be backtested against the existing ledger:
`.savant_cache/` is date-keyed and gitignored, and today's leaderboard is
season-to-date, so scoring old rows against it is lookahead. The shadow columns
(`mix_*`, `opp_xwoba_mix_*`, `mx_xwoba_sp_mix_*`, `edge_xwoba_sp_mix_*` in the
ledger, NaN on every row built with the flag off) exist so the arm accrues a
forward record. Promotion to the lean would change both prediction math and
`|xw_net|` units, so it would start new `RECORD_TAGS` **and** `SCALE_TAGS`
families.

### SP platoon-advantage xwOBA adjustment

The xwOBA lineages through wOBA v1 moved a **one-sided** hitter by a flat
**±0.010** before the lineup composite. wOBA v2 supersedes that constant with
`PLATOON_XWOBA_OFFSETS`, the exposure-centred four-cell table in the current
model section. A switch hitter is still **not** moved.

Two separate questions, deliberately answered by two helpers:

- **Does he hold the edge?** `platoon_advantage(bats, throws)` — a switch
  hitter, or a hitter with no recorded side, takes the side opposite the pitcher
  and therefore always holds it. Same convention the platoon-OPS lens uses for
  `eff_stand`; both call the one helper. This drives the card's ◆ marker and the
  lens's `platoon_adv` / `n_platoon_adv` count.
- **How far does his season xwOBA move?** `platoon_xwoba_offset(bats, throws)`.
  The offset is a *deviation from the hitter's own season line*, and that line
  is already a platoon blend weighted by his real exposure to each pitcher hand.
  A switch hitter bats opposite the starter in essentially every PA, so his
  season xwOBA already **is** his advantage-state number — adding another
  advantage term would count the same edge twice. His offset is **0**, and he
  is still marked ◆. An unrecorded bats side gets 0 for the same reason an
  unknown starter hand does: no evidence either way is not evidence of a
  disadvantage.

**Where it lands.** The tag and the starter's hand (`sp_throws`) are attached in
`segment_pitcher_blocks`, so the xwOBA lean does not depend on the platoon-OPS
lens, which is optional and may abstain. The offset is applied in
`aggregate_lineup` **after** shrinkage and after the team backfill, then the
slot-PA composite is taken. Order matters: shrinkage estimates season talent
from a noisy sample, while this is a matchup term with no sample-size
uncertainty of its own — regressing it would make a 15-PA bat's platoon edge
worth less than a 550-PA bat's. A team-backfilled hitter bypasses shrinkage but
still receives the platoon term.

**What does not move.** Only the lean input. The per-hitter card xwOBA stays the
raw season rate and `xw_pctile` stays a season-talent rank against qualified
regulars, consistent with how shrinkage is already displayed. The switch-hitter
exemption prevents the page's advantage marker from overclaiming what moves the
lean. v2 implements the centred form `+(1−s)·g` with the edge and `−s·g`
without it using static population exposure shares; individual hitter splits
remain outside the model.

**Historical v1 measured effect** (279 games over the committed slate dumps,
using each side's
`n_platoon_adv` / `n_SW` / `n_opp` counts):

| quantity | switch bats moved | switch bats exempt (shipped) |
|---|---|---|
| lineup advantage share (median) | 0.667 | 0.556 |
| `opp_xwOBA` shift per side (median) | +0.0033 | +0.0011 |
| sides shifted up | 76.6% | 59.9% |
| `xw_net` change vs no adjustment (median abs) | 0.00349 | 0.00330 |
| lean flips vs no adjustment | 16 / 279 (5.7%) | 11 / 279 (3.9%) |
| median `\|xw_net\|` (0.02672 unadjusted) | 0.02677 | 0.02642 |

Switch hitters appear in 67.4% of lineups (mean 0.97 per side) and were 17.7% of
all tagged-advantage bats, so exempting them removes a systematic upward bias
rather than a rounding effect.

**Versioning — resolved.** `MODEL_TAG` was deliberately *not* bumped for the
original adjustment or for the switch-hitter exemption that followed it, which
left prediction math changing twice *inside* `xw+plat_consol_v7`: the adjustment
flipped 5.7% of leans against no adjustment, and the exemption flipped a further
3.9% against the shipped adjustment. The 45 v7 rows in the ledger therefore
share one win-loss line while having been produced by three different models,
and the flips land on exactly the games a marginal record is most sensitive to.

That was recorded here as a knowing exception to the "bump on any
prediction-math change" rule, with the note that bumping to v8 was the fix. **v8
duly landed** (fixed `K = 100`), and v9 and v10 after it, so v10 was four
families clear of the problem; the wOBA lineage (v1 through v4) is four further
families on again, each of them isolated. The v7 rows remain immutable and remain
internally heterogeneous — read that family's 22-23 line with that caveat, and
do not treat it as one model's record. The units argument still holds: median
`|xw_net|` moved only 0.02672 → 0.02642 across the two changes, so v7 rows do
pool correctly for `lean_strength_scale()` within their own scale family.

The lasting lesson is the one now at the top of `CLAUDE.md`: a bump costs a
reset of the graded sample, which is a real price, but carrying three models
under one tag costs the ability to read the record at all — and that price is
paid silently.

## Files

| Path | Purpose |
|------|---------|
| `build_site.py` | One-shot generator: fetch → matchup dataframes → writes `public/index.html` and `public/grades.html` (fully self-contained: inline CSS, dark-mode via `prefers-color-scheme`, no external assets). Also dumps the day's leans to `data/leans_<date>_{xw,pl}.csv` for the grading ledger — or to `rebuild_leans_<date>_...` when every game on the slate has already started, so a post-rollover rebuild adds a later view instead of overwriting the pregame one. |
| `grade_leans.py` | Grading ledger: ingests the lean dumps as pending rows, grades them against StatsAPI linescores (full-game + F5), attaches closing DK moneylines (via `market_backfill`), writes `data/mlb_lean_ledger.csv` + `data/ledger_report.txt`. |
| `pitch_arsenal.py` | Pitch-mix shadow arm: the opposing lineup re-weighted by the starter's arsenal. Off by default (`PITCH_MIX_SHADOW=1`), inert when on — writes shadow columns for forward testing and never moves a lean. |
| `pitch_arsenal_probe.py` | Measurement that gates the arm: cell dispersion split into signal and sampling noise, year-over-year reliability, and the noise budget the arm must clear. Prints only; writes nothing to `data/`. |
| `reliever_shrink_probe.py` | Measurement behind `XWOBA_SHRINK_K`: fits the BF-denominated regression constant out of sample three independent ways, with batters and starters through the same code path as controls. Prints only; writes nothing to `data/`. |
| `player_prior_probe.py` | Measurement on the shrinkage *target*: does a player's regressed recency-weighted career rate beat the league/role centre at the shipped K? Scores the population centre, the raw career rate and the regressed one on identical rows, paired. Prints only; writes nothing to `data/`. |
| `player_priors.py` | The wOBA v4 shrinkage target: loads the frozen snapshots and computes `prior_for(pid, ...) = (H·θ_hist + C·μ)/(H + C)`, the player's recency-weighted history regressed toward the population centre. Returns `μ` exactly at `H = 0`, so an absent player needs no branch and an absent snapshot degrades the whole build to v3 behaviour. |
| `priors_snapshot.py` | Freezes completed-season Savant wOBA into `data/woba_priors_<season>.csv` + `data/woba_prior_centres.csv`, for use as player-specific shrinkage priors. Refuses in-progress seasons and never silently rewrites a season file. Rookies get no row — the fallback is `H = 0` in the prior expression, not a branch. |
| `market_backfill.py` | Odds join: attaches ESPN/DraftKings opening + closing moneylines and the devigged home close probability to settled ledger rows (score-verified join, idempotent, no silent defaults), and computes the vs-market scoreboard. |
| `run_market_update.py` | Headless CLI for the odds join: `--dry-run` preview, one-off backfills, `--merge-backfill` for pre-enriched files. CI doesn't need it (grading calls `attach_market` directly); it's for local runs. |
| `.github/workflows/build.yml` | Scheduled + manual workflow: build → grade → commit ledger → deploy Pages. |
| `.github/workflows/tests.yml` | The gate: `validate_data_files.py` then the whole suite, on every pull request and every push to `main`. Deliberately *not* wired into `build.yml` — a test failure blocking the daily build would cost that slate's rows permanently. |
| `requirements.txt` | `requests`, `numpy`, `pandas`. |
| `data/` | Committed state: daily lean dumps, the ledger, and the latest report. |

The notebook's clean/validate cells (2–3) are intentionally not ported: they
produced `*_clean` / `*_strict` frames the matchup/render cells never consume
(the API `build_tables` output is already clean).

## Run locally

```bash
pip install -r requirements.txt
python build_site.py            # writes public/index.html for today's ET slate
open public/index.html
```

Environment variables:

- `SLATE_DATE` — force a date (`YYYY-MM-DD`); otherwise resolved in
  `America/New_York` with a ~3am ET rollover so night games don't roll early.
- `CACHE_DIR` — where the once/day Savant CSVs are cached (default `.`).
- `OUT_DIR` — output directory for `index.html` (default `public`).
- `MODEL_TAG` — row-level model/audit lineage for newly captured predictions.
- `RECORD_TAGS` — comma-separated tags whose unchanged prediction math should
  be summarized as one continuous performance family.
- `SCALE_TAGS` — comma-separated tags whose `|xw_net|` values use compatible
  units for lean-strength ranking; independent of `RECORD_TAGS`.

## Unattended-run behaviour

- **Dynamic ET date.** The runner clock is UTC; the slate date is computed in
  ET (3am rollover) so a UTC midnight rollover doesn't grab the wrong day
  mid-slate.
- **Don't clobber good output.** A hard fetch failure exits non-zero *without*
  writing `index.html`, so the deploy job is skipped and the last good page
  stays live. Per-game data gaps degrade gracefully (a side with no vs-hand
  split shows a muted "—"); a legitimately empty slate (off-day) writes a
  friendly "no games" page; a pre-slate state with no posted probables writes a
  "check back closer to first pitch" page.
- **Savant from a datacenter IP.** Requests use a real browser User-Agent with
  retries + exponential backoff. If Savant rate-limits a runner, the build
  fails and the previous page stays up rather than deploying a broken one.
- **Cache.** Savant CSV leaderboards are cached via `actions/cache` keyed to the
  ET slate date (`savant-YYYY-MM-DD`), reproducing the notebook's once/day
  behaviour across that day's runs.
- **Slate-aware cadence.** A lightweight Actions poll runs every 15 minutes
  from 10am–11:59pm ET. `schedule_gate.py` fetches that day's MLB schedule and
  launches the full build only when at least one game is 15–90 minutes from
  first pitch; the wide window rides out Actions queue jitter, and pending-row
  refresh keeps the lock at the last pregame snapshot. A separate 4:17am ET
  pass grades the prior slate. Push and manual runs always build immediately.
- **Chronological slate.** Matchup cards render by scheduled first pitch rather
  than lean strength; doubleheader game number and gamePk provide stable ties.

## One-time setup

Enable Pages with **Settings → Pages → Build and deployment → Source: GitHub
Actions**. After that the workflow deploys on each scheduled run (and on manual
`workflow_dispatch`). The site is served at `https://dave356w.github.io/dave356w/`.

## Grading ledger

Grades are also rendered into the site: the main page shows a **records
strip** for the complete model lineage, linking to the ledger.
**`grades.html`** preserves that combined headline and shows every game's
**published selection**, its closing ML, the final score and the selection's
full-game W/L/T grade, with pending and void rows included. The selection is
the hybrid rule's, not the raw lean: rows where the market priced the model's
side below the threshold are marked `FADE` and carry the opposing club, with
the grade inverted accordingly.

The Selection column is **scoped to `RECORD_TAGS`** — the family the rule is
registered against. Applying it to an earlier family would publish a selection
nobody could have made, under lean math the rule was never paired with and with
a grade the code would then invert; those rows are marked `lean only` and show
the lean they actually published. The other two cases the rule cannot act on
carry their own distinct marks rather than a shared one: `awaiting close` for a
row with no two-sided price yet (every pending row, by no-lookahead), and
`no lean` for a v5 abstention. The header states the split explicitly, so every
row of the current family is accounted for rather than only the ones the rule
could score.

Two trivial-strategy **controls** sit in the same summary strip as the record,
scored on the *identical* rows: *always home* (the null hypothesis for any
side-picking model, and the full-game twin of the F5 home baseline in
`ledger_report.txt`) and *always chalk* (the devigged closing favourite — the
strategy the rule has to beat, since the closing line is free). Both are
derived on the same observation frame as the record itself, so their
denominators cannot drift apart. The public page intentionally omits
model-family history and per-row model labels; those remain available in the
underlying ledger and Actions report.

The strip's record is a **discovery** figure: the threshold was chosen on
these rows. The out-of-sample version scores only games played after the rule
was frozen and prints every build in `data/ledger_report.txt`; the page says
so rather than leaving a reader to infer it. Both pages render purely from
`data/mlb_lean_ledger.csv`; grading runs before the build in CI (with a second
pass after it to ingest the day's fresh dumps), so the page reflects last
night's results in the same run.

The platoon-OPS lens and first-5-innings (F5) results are still computed and
recorded in the ledger for auditing (`grade_leans.py` grades both), but the
site display was pared back to the xwOBA full-game lens only; the pitcher card
shows Statcast **xERA vs season ERA** in place of the last-5-starts ERA.

Every CI run, `grade_leans.py`:

- **Ingests** timestamped pregame `data/leans_*_xw.csv` snapshots not yet in
  the ledger as `pending` rows. Re-runs refresh still-pending rows only while
  the snapshot precedes scheduled first pitch; late and legacy-unverified
  refreshes are rejected, and graded rows are immutable.
- **Grades** all pending rows via `schedule?hydrate=linescore` (one call per
  date): full-game and first-5-innings W/L/T per lean. Live games stay
  pending; postponed/cancelled go `void`.
- **Attaches market odds** to settled rows still missing them:
  ledger row → StatsAPI gamePk (date + teams, doubleheaders disambiguated by
  probable-pitcher surname, join verified by final score) → ESPN event →
  DraftKings (provider 100) opening/closing moneylines + devigged
  `close_p_home`. Idempotent; a row that can't be verified keeps NaN market
  columns and retries next run. A market outage never fails the grading run.
  `grades.html` then shows each selection's closing ML and a vs-market
  scoreboard (record vs market-expected wins → z, flat-stake ROI), and
  `market-calibration.html` splits the same rows by hybrid branch.
- **Reports** the current `RECORD_TAGS` family to the Actions log and
  `data/ledger_report.txt` (overall, reliable-only platoon subset, |Δ|
  terciles, DIVERGE head-to-head, and — once 120 graded F5 decisions
  accumulate — a pitching-vs-lineup logit weight fit), followed by immutable
  record lines for every historical model family.

The ledger persists by being committed: the workflow's `Commit ledger` step
pushes `data/` back to `main` on each run (the `contents: write` permission).
The ~4:17am ET cron is the grading pass — it runs after night games end and
grades the previous slate. A prediction-math change must bump `MODEL_TAG` and
start a new `RECORD_TAGS` family so incompatible games never mix in the records
or the weight fit. An audit-only tag change may remain in the same family.
