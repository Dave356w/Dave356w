# Working standards — Dave356w/Dave356w

MLB matchup-leans site. `build_site.py` renders daily cards to `public/`,
`grade_leans.py` grades pending rows against StatsAPI linescores,
`market_backfill.py` joins DK open/close via ESPN. GitHub Actions builds on a
pregame trigger (`schedule_gate.py`) and commits `data/` back to the repo.

Read `MATCHUP_SITE.md` for the model. Read this first for how to work here.

## Method

- **Evidence before conclusions.** Claims about this codebase get verified
  against the ledger or the source, not recalled. If you assert a record, a
  distribution, or a behaviour, run it first and paste the number.
- **Verify, don't recall.** Do not describe what a function does from its name
  or its docstring. Both have been wrong in this repo. Read the body.
- **Benchmark proposed fixes before recommending them.** A change that sounds
  principled can be worse than what it replaces. Compare candidates on the
  metric the fix is supposed to improve, across enough seeds to see variance,
  and report the loser honestly — including when the loser is your own proposal.
- **Subtractive.** Prefer deleting a branch to adding one. A fix that removes a
  special case beats a fix that adds a tier. If a change grows the code, say why
  the simpler version fails.
- **No sycophancy.** Do not open with praise. Lead with the finding. If a
  request rests on a wrong premise, say so before answering it.

## Model versioning — two namespaces, do not conflate

`MODEL_TAG` stamps every ledger row with its lineage. Bump it on any change to
prediction math. Two separate tag families gate two different questions:

- `RECORD_TAGS` — may these rows share a win-loss line? Prediction-math
  compatibility. Governs `_record_grades()` and the weight fit.
- `SCALE_TAGS` — do these rows measure the primary-rate delta on the same
  scale? (`xw_net` is the retained legacy ledger name.) Units compatibility.
  Governs `lean_strength_scale()` only.

These are different equivalence relations and they do disagree. The authority is
`_RECORD_FAMILIES` / `_SCALE_FAMILIES` in `build_site.py` (records mirrored in
`grade_leans.py`) — not this table, which is a reading aid. Current model is
**v11** — Savant xwOBA, `XWOBA_SHRINK_K = 100`, population shrinkage targets.

| tag | what changed | record family | scale family |
|---|---|---|---|
| v2 | baseline | 2 | 2 |
| v3 | pregame lock only, math identical to v2 | 2+3 | 3 |
| v4 | slot-PA lineup weighting | 4 | 4 |
| v5 | empirical-Bayes xwOBA shrinkage (halved the scale, median `\|xw_net\|` .036 → .018) | 5 | 5+6 |
| v6 | expected-IP starter/bullpen blend; inherits v5 shrinkage | 6 | 5+6 |
| v7 | centre-matched shrinkage moments; full precision; zero = abstention | 7 | 7 |
| v8 | fixed `K=100` shrinkage, widening the delta distribution | 8 | 8+9+10 |
| v9 | starter/bullpen phases split; handedness applies to starter innings only | 9+10 | 8+9+10 |
| v10 | phases weighted by PA share (measured BF/IP) not innings share | 9+10 | 8+9+10 |
| wOBA v1 | observed wOBA replaces xwOBA in every active rate input; v10 construction fixed | wOBA v1 | wOBA v1 |
| wOBA v2 | exposure-centred 0.021 starter platoon gap replaces universal ±0.010 | wOBA v2 | wOBA v1+v2 |
| wOBA v3 | `XWOBA_SHRINK_K` 100 → 400 (fitted); reliever target moves to the relief pool's own unweighted centre | wOBA v3 | wOBA v3 |
| wOBA v4 | shrinkage *target* becomes the player's own recency-weighted 2023–2025 history, not a population centre | wOBA v4 | wOBA v4 |
| wOBA v5 | abstain when a side's starter has no measured season line | wOBA v5 | wOBA v4+v5 |
| split v1 | one-slate wOBA-lineup/xwOBA-arms test; abandoned before grading | split v1 | split v1 |
| v11 | revert to xwOBA + K=100 + population target, keeping v2's platoon centring, v3's relief-pool target and v5's abstention | v11 | 8+9+10+11 |

The wOBA forward test is intentionally isolated from xwOBA in both namespaces.
Observed wOBA changes the predictions and its sampling distribution is not the
xwOBA delta scale. v2 starts a clean record because the platoon prior moves
predictions, but shares v1's strength scale: the metric is unchanged and each
handedness pair retains essentially the same total gap (0.021 versus 0.020).
Internal `xwOBA`/`xw_*` dump and ledger keys remain a compatibility schema for
immutable history; every row must carry `model_metric` explicitly. Under v11
that schema and the statistic agree again, which is *more* dangerous rather than
less: the keys stop being an obvious lie and start looking like documentation.
They are not. Read the metric from `model_metric`, never from a key name and
never from the running build's constants — `market_backfill.metric_label()` is
the one derivation, and `shadow_report.dump_metric()` is its per-dump twin.

**v11 reverts the metric to xwOBA and `K` to 100 and the shrinkage target to the
population centre, and this table's job is to stop that being read as a
finding.** No measurement said xwOBA beat wOBA: the paired shadow arm exists
because the era comparison cannot answer it, and at six slates it reports
d_corr +0.008 with CI [-0.108, +0.128]. No measurement said K=100 beat K=400 —
`reliever_shrink_probe` fits K three ways on n=53,464 and every interval
excludes 100. No measurement said the population centre beat personal priors —
the out-of-sample probe says the opposite, the forward lineup-component read
says the reverse, and bootstrapped that forward read is +0.128 with CI
[-0.054, +0.301]. It was an operator decision taken with all of that on the
table, and the code comments state it that way at each site. Do not let a later
reader convert it into evidence, and do not quote the wOBA lineage's 63-76 as
the reason: over those same rows always-home ran .604.

What v11 does *not* revert is the part with evidence independent of those three
knobs — v2's exposure-centred platoon offsets (a construction fix: the season
line is already exposure-weighted), v3's relief-pool shrink target (the league
batter centre is the centre of no subpool; the relief pool sits 0.0102 below
it, at any `K`), and v5's abstention. That split is the whole content of the
change and it is argued piece by piece in `_RECORD_FAMILIES`.

v11 is also the counter-precedent for a **shared scale across a reverted
metric**: it isolates its record and joins the v8/v9/v10 delta pool, because
the three ways it differs from v10 are each scale-preserving on a precedent
already in this table (platoon centring moved median |net| 0.7% at v1→v2;
abstention left v4/v5 quantiles identical; a uniform re-centring cancels in a
difference). Unlike v3, v4 and v10, that half is **argued, not measured** — no
lookahead means a past slate cannot be rebuilt to check it. The falsifier is
named in `_SCALE_FAMILIES`: compare median |xw_net| on the first graded v11
rows against the v9/v10 pool, and split the family if it moved.

The frozen `data/woba_priors_*.csv` are wOBA-denominated and stay that way.
`priors_snapshot.RATE_COL` is pinned to `woba` rather than following the build,
and `player_prior_history()` refuses to serve them to a non-wOBA build — so
`PLAYER_PRIORS=1` under v11 does nothing at all. Restoring v4 needs a wOBA
build, or an xwOBA prior set under its own filenames.

v3 and v4 each isolate in **both** namespaces, and in both cases the scale half
is arithmetic rather than judgement — which is what makes them the useful
counter-precedent to v10. v3 quadruples `K`, so every input keeps less of its
deviation and `|xw_net|` compresses (a 400-PA batter drops from 400/500 = 80% of
his deviation to 400/800 = 50%). v4 is the same argument with the sign reversed:
shrinking toward a *personal* prior instead of one shared centre lets two players
with equal samples keep different centres, so the delta distribution widens. Same
wOBA units both times, materially different spread, which is exactly what a scale
family separates. Neither inherited anything; both are argued in the comments
above `_RECORD_FAMILIES` and `_SCALE_FAMILIES`, which is where the argument
belongs.

**wOBA v5 is the counter-precedent to v4, one namespace at a time.** It answers
the two questions differently, and both halves are measured. Record: isolated,
because it changes *which games are decided* — 6 of the 185 ledger rows that
carried starter/bullpen instrumentation at the bump (3.2%) had a prior-only
starter on one side and would now publish nothing — and a win-loss line is a
property of the decided set, not only of the arithmetic. Scale: **shared with
v4**, because every surviving `|xw_net|` is bit-identical (the abstention only
nulls an edge; it never rescales one) and dropping the abstained games moves no
cutoff — pooled p33/p80 over those 185 rows is 0.0090 / 0.0283 with them and
0.0090 / 0.0283 without, and within v9/v10 alone 0.0127 / 0.0343 either way.
That is v6's precedent (a new prediction family inheriting a scale) applied to
a filter rather than to a construction change. The reset costs v4's graded
rows, whose count you should read off the ledger rather than from here.

The `185` is a correction: the PR, `MATCHUP_SITE.md` and this file all said
`187`, and no row set of that size exists. The quoted quantiles pin the set
exactly — `starter_xwoba_*` / `bullpen_xwoba_*` / `expected_sp_ip_*` /
`pitching_basis_*` all resolve to the same 185 rows in the ledger as of
`6b540ae~1`, and all four give 0.0090 / 0.0283. Nothing downstream moves; the
argument was right and its denominator was not. Same category as the count
below it, which is why both are fixed here rather than restated.

Two things not to read into it. The 6 abstained games graded 3-3 against
96-82 (.539) on the rest; at n=6 that is incidence, not evidence they were bad
picks, and the argument for abstaining does not depend on it — the input was
never measured either way. And v5 is the first mechanism in this repo that can
produce a **graded row with no lean** — and as of 2026-08-09 it has. The
sentence here used to read "the ledger holds no undecided rows yet"; it fired on
2026-08-08, DET@SF, where Jackson Jobe carried no measured season line
(`pitching_basis_away=starter_unmeasured_no_lean`), and that row has since
graded. It is the only leanless row in 482. v7's zero-delta abstention still has
never once fired at full precision, so v5 remains the only mechanism that
actually produces these. `_rec()` drops those rows while `len()` counts them, so
every count that mixes the two must say which it is — see the controls entry
below for the one surface that did not, now a live discrepancy rather than an
armed one. `ledger_report.txt` states the split correctly today
(`55 graded games (54 with a lean, 1 abstained)`), and the grades page scores
its controls on the 54.

`split v1` remains an isolated historical namespace because its dump and
pending ledger rows existed before full wOBA was restored. It is not an active
alternative and shares neither records nor delta scale with wOBA v1. Current
wOBA dumps ingest after split dumps, so any same-day pending split snapshot is
re-stamped into the restored lineage before first pitch; settled rows remain
immutable.

Two entries earn their keep as precedent. **v6 shares v5's units but not its
record line** — a new prediction family can inherit a scale. **v10 shares both
of v9's** — the PA-share reweight is a convex combination of the same two
phases, so units are untouched, and it flips 0 of 14 leans on measured rates,
so the win-loss line is shared rather than reset for the eighth time in a month.
A bump does not automatically mean isolation; argue it.

Known latent gap: v3's scale family is `(3,)` because `_SCALE_FAMILIES` has no
v3 entry, though v3's math is identical to v2 and the two must share units.
Inert — `SCALE_TAGS` only ever derives from the *current* tag — so it is
recorded, not patched.

Second, same category: **v8 has no rows.** The table above and `_SCALE_FAMILIES`
both treat v8 as a member of the current scale pool, but no `xw+plat_consol_v8`
row has ever been graded into the ledger. v8 shipped for a single morning
(2026-07-27); its 11 rows were all still `pending` when v9 landed, so the
pregame refresh rebuilt them under v9 math and re-stamped them — legitimate,
and the reason `MODEL_FAMILY_TAGS`' v8 line never prints. So `SCALE_TAGS`
matching v8 selects nothing, and `compare_v8_v9.py` compares against a version
that never survived into a graded row. Inert, so recorded rather than patched:
the map is the authority on a historical question and deleting the entry would
lose the answer. What it means in practice is that "the v8/v9/v10 scale pool"
is the v9/v10 pool, and any provenance note claiming v8 rows is wrong.

Third, **now settled the way it was predicted to settle: wOBA v3's graded rows
arrived the day after the note that assumed them.** The `_RECORD_FAMILIES`
comment on `woba+plat_consol_v4` says "The v3 family had its own graded rows
and they stay immutable." Written 2026-08-05, the day v3 shipped, that
described rows which did not exist: v3 held 4 rows, all `pending`, none ever
graded. They graded overnight. As of 2026-08-06 the ledger holds 4 graded v3
rows and no pending ones, and `data/ledger_report.txt` prints
`wOBA v3 n=4  wOBA full 3-1 (0.750)  F5 4-0 (1.000)`.

So the sentence in the source is true today, and it was left alone rather than
corrected and re-corrected — which is why this was recorded here instead of
patched. Keep the instance anyway: it is the third case of a version note
asserting rows a build had not yet produced (v8 above, and the
`wOBA full 217-164` front-page incident), and turning out right a day later
does not convert an assumption into a measurement. **Check the ledger before
quoting a family's record**; a 4-row line is not one either way.

When you bump `MODEL_TAG`, decide both questions explicitly in the PR body.
Silence defaults to a new record family and inherited units — which is wrong
about half the time. And when you bump it, grep for the tag: it must live in
exactly one place per module. See the workflow-pin anti-pattern below.

Display-only changes do not bump `MODEL_TAG`. Card layout, copy, CSS, legend
text: no bump. Anything that moves a lean, a delta, or a grade: bump.

## Anti-patterns with instances in this repo

Each entry names a real commit in this repo. Resolved ones are kept as
precedent — they are how the fix is known to look.

**Live — not yet fixed**

- **Constants frozen from data.** `LEAN_STRENGTH_FALLBACK` was a literal copy of
  the pooled p33/p80 at the time it was written, and stayed there through two
  model versions that changed the distribution underneath it. It was re-derived
  for the v9/v10 xwOBA family, and **every wOBA bump since has re-staled it** —
  a new `_SCALE_FAMILIES` entry is exactly the invalidation its own comment
  names, and there have now been four (wOBA v1, v2 sharing v1, then v3 and v4
  each isolating). It is deliberately not re-derived yet: the wOBA pool is still
  small enough that its p33/p80 is noise, and freezing that would be this
  anti-pattern with a fresher date on it. Shrinkage plus the slate top-up hold
  the line meanwhile — recompute the pool before quoting it rather than reading
  a number off this file, which is the same discipline the controls entry below
  demands.

  The one directional reading this entry used to carry — that as of 2026-08-04
  (wOBA v1+v2, n=16) the observed p80 ran well under the 0.032 prior while p33
  sat close to 0.015 — **no longer describes the current scale family and has
  been retired rather than restated.** v3 and v4 each started a fresh
  `_SCALE_FAMILIES` entry. Note what counts: `lean_strength_scale()`
  takes every `SCALE_TAGS` row regardless of grade status, because `|xw_net|`
  is a pregame quantity — so "v5 has no graded rows" is true and irrelevant
  here, and the constraint is thinness, not gradedness.

  This paragraph used to name a pool — "`SCALE_TAGS` today selects v4 alone,
  n=11, observed p33/p80 0.0059 / 0.0153" — and **that went stale within a day
  of being written**, which is this very entry's own failure mode in prose. v5
  shares v4's scale family, so `SCALE_TAGS` resolves to `(v4, v5)` and the pool
  is both families' rows, not v4's. The figure is deliberately not refreshed to
  a new literal: call `lean_strength_scale()` and read `.size`, then take the
  quantiles off what it returns. Carrying the old direction
  forward would have been the anti-pattern itself: a number measured on one
  distribution, quoted against another, with a bump in between that provably
  changed the spread in a *known direction opposite to v3's*. Recompute from
  whatever `SCALE_TAGS` resolves to at the time, once that family passes ~60
  rows. Expect that to take a while — the pool has restarted twice in two days.
  If a constant was read off the ledger, comment where it came from and what
  would invalidate it. Note the
  asymmetry the comment there spells out: a *scale-family* change invalidates
  it, but mere pool growth does not — it is a prior, and re-deriving it from the
  family it is shrunk against would make it the data.

  **v11 lands this entry somewhere it has not been before: back on the family
  the constant was fitted to.** `SCALE_TAGS` now resolves to the
  v8/v9/v10/v11 pool, which is the v9/v10 rows plus whatever v11 writes, and
  `LEAN_STRENGTH_FALLBACK` was re-derived for exactly that family. Measured at
  the revert, n=99, observed p33/p80 **0.0127 / 0.0343** against the frozen
  0.015 / 0.032 — the same distribution it was read off, close enough that the
  prior is doing its job rather than fighting the data. So the five wOBA-era
  bumps did not leave a stale constant behind; they left a constant temporarily
  pointed at the wrong family, and the revert points it back.

  That is a reprieve, not a fix, and the entry stays live for the reason it was
  written: the number is still a literal, and the next `_SCALE_FAMILIES` entry
  re-stales it exactly as the last four did. Do not quote 0.0127 / 0.0343
  either — recompute from whatever `SCALE_TAGS` resolves to when you need it.
  Note also that v11 is *pooled into* this family on an argued rather than
  measured basis, so the first graded v11 rows are the check on both things at
  once: if median |xw_net| has moved, the family split is wrong AND this
  constant is stale again.

  `HEAT_DOMAINS` is the same shape one level out, and v11 changes what is known
  about it rather than settling it. Its saturation ranges were calibrated on the
  xwOBA spread and the model is back on xwOBA, so the mismatch the entry was
  filed for is gone for now. The measurement behind it stands and is worth
  keeping: the one slate built under both metrics showed the starter-allowed
  rate widening under wOBA (sd 0.0161 → 0.0215) while the lineup composite did
  not move (0.0089 both ways) — a starter rate is one player's observed outcome,
  a lineup is nine shrunk ones averaged. n=14 is one slate, not a distribution,
  and it is now a fact about the metric the model does not run. Display-only,
  so no `MODEL_TAG` implication either way.

- **A measured defect deferred, with the fix that would recreate the entry
  above.** The first backfilled actuals (2026-08-04, `actuals_backfill`) show
  `expected_sp_ip` is **over-dispersed**: regressing actual on predicted over
  306 side-games gives slope **0.756 ± 0.063**, 3.9 se below 1.0. Bias on the
  same rows is +0.096 IP (t = 1.31) — a spread problem, not a level one.
  Starters predicted short go 0.69 IP longer than predicted; those predicted
  deep go 0.12 shorter. Shrinking toward the mean by 0.76 cuts IP MSE 5.2%.

  Not fixed, on two grounds. It flips 1 of 80 graded v9/v10 leans (mean
  |Δ net| 0.00067 against a median |xw_net| of 0.019), so the case is
  correctness of a directly-observed input, not performance — and correctness
  fixes can wait for a stable estimate. And 306 side-games of July/August is
  thin for a slope that is plausibly seasonal: pitch counts climb early and
  clubs get cautious in September, so this may not be the September slope.

  **Gate: re-fit at ~600 side-games** (roughly 2026-08-25). The report prints
  `IP calibration slope` every build so the number arrives without anyone
  remembering to look. If it holds near 0.75, ship it — but as a per-build
  derivation off the accumulating actuals, never as a frozen `a + b·pred`.
  A literal fitted today is exactly the constants-frozen-from-data entry
  above, with a fresher date on it. It is lean-path (it moves `q`, so `mx`),
  so it bumps `MODEL_TAG`; at 1 flip in 80 it shares both families on the v10
  precedent, but argue that in the PR rather than inheriting it.

  Two companion readings from the same backfill, both recorded rather than
  acted on. The realized phase weight (`act_sp_bf / act_pa`, the share of PAs
  the starter actually faced) carries the same over-dispersion in the units
  that matter — slope 0.746, bias +0.017, MAE 0.101 over 210 side-games — so
  the defect is in the workload estimate, not in the BF/IP conversion. And the
  rate metric says nothing yet: calibration slope 0.953 ± 0.380, corr
  0.178 ± 0.070 against a 0.196 ceiling for a perfect model. Its 95% CI spans
  near-zero to above that ceiling. Do not quote those two as findings.

- **A past slate's dump is overwritten by a post-hoc rebuild.** `SLATE_DATE`
  rolls over at 3am ET, and the daily grading cron fires at 04:17 UTC — 00:17
  ET — so it still names *yesterday's* slate. It re-runs the full build for
  that date against today's Savant leaderboard and unconditionally rewrites
  `data/leans_<yesterday>_<DUMP_SUFFIX>.csv` — `_woba.csv` while the wOBA
  lineage ran, `_xw.csv` again under v11. Measured across
  every committed dump that carries `snapshot_utc`: **every past slate's dump
  is a post-first-pitch rebuild.** `leans_2026-08-05_woba.csv` carries
  `model_tag=woba+plat_consol_v5` and `snapshot_utc=2026-08-06T04:18Z`, while
  all 15 of that slate's ledger rows are v3/v4 with pregame snapshots and
  different numbers — game 822866's `starter_xwoba_away` is 0.327266 in the
  ledger and 0.336786 in the dump. The dump on disk is stamped with a tag whose
  math produced none of that slate's rows.

  **The ledger is not affected and the guard is load-bearing.** `ingest()`
  re-reads every dump on every run and admits a row only when
  `lock_status == "pregame"`; the rebuilt 08-05 dump is `late_snapshot` on all
  30 sides and is rejected wholesale. Verified: 0 of 437 ledger rows have
  `snapshot >= scheduled_start`, and `lock_status` holds 287 `pregame`, 1
  `pregame_recovered`, 149 legacy, **0 `late_snapshot`**. Do not weaken that
  check — it is the only thing standing between the rebuild and the ledger.

  What it costs is provenance, and the cost is already being paid. The dump is
  the sole per-slate record of what the model actually saw, and for every past
  slate it has been replaced by what a later model saw with a later
  leaderboard. So: any measurement that joins ledger rows to "the dump beside
  them" is reading post-hoc data — including `FINDINGS.md`'s prior-only
  incidence ("6 of the 403 side-games in the committed dumps"), and
  `compare_v8_v9.py`, which globs `data/leans_*_xw.csv` and therefore compares
  against a v8 dump that is itself a rebuild of the 07-26 slate.

  **v11 changes that glob's population and this is recorded, not patched.**
  The primary dump suffix is `_xw` again, so `data/leans_*_xw.csv` now matches
  v11 dumps alongside the pre-wOBA ones. That is not obviously wrong — the
  script recomputes both the v8 and v9 forms from a dump's own phase columns
  and never reads `model_tag`, and a v11 dump carries the same columns at the
  same `K=100`, so pooling it measures the same formula difference on more
  slates. It IS wrong to keep describing the output as "over 24 eligible games
  of v8/v9 dumps" once v11 rows are in it. Read the row count off the run
  rather than off any prose, here or in the script. It also biases
  monitoring toward optimism: `sp_bf_per_ip` is missing on 4 of 301 committed
  side-games (1.3%), but on **3 of 22 tonight** (13.6%) — a rebuilt dump has a
  full extra day of StatsAPI behind it, so the historical rate is measured on
  data the pregame build never had.

  Not fixed here, because the fix is a behaviour change to the daily pass and
  deserves its own decision rather than a drive-by. The options are to skip the
  dump write when `SLATE_DATE` is not the current ET date, to write the rebuild
  under a distinct name, or to accept it and stop citing dumps as pregame
  evidence. Display/provenance only — no lean, grade or ledger row moves, so no
  `MODEL_TAG` implication whichever way it goes.

  **The shadow arm inherits this and has no ledger behind it.** `shadow_metric`
  runs as a step of the same build, so the shadow dump is rewritten by the same
  post-rollover pass: `shadow_2026-08-10_xw.csv` on disk is stamped
  `2026-08-11T06:50Z` against a 23:07Z first pitch, and 08-09's only committed
  version is the 03:06Z rebuild — the arm landed at 02:56Z that morning, so a
  pregame 08-09 dump never existed. For the primary this costs provenance and
  the ledger holds the pregame truth; for the shadow arm the dump *is* the
  record, and git history is the only place a pregame version survives (08-10
  has four, the last at 23:01Z, six minutes before first pitch).
  What this does **not** break is the pairing, and that distinction is the
  whole reason the arm is worth reading: primary and shadow are written 20-40
  seconds apart in the same job from the same leaderboard, so dump-against-dump
  is honest even when both are rebuilds. It is dump-against-*ledger* that is
  contaminated — a pregame wOBA decision against an xwOBA one with an extra day
  behind it. `shadow_report.py` pairs the dumps for that reason, prints each
  slate's provenance instead of assuming it, and will run the ledger join under
  `--ledger-join` so the bias can be sized rather than argued.

**Resolved — keep as precedent**

- **One value, three homes.** `.github/workflows/build.yml` pinned `MODEL_TAG`,
  `RECORD_TAGS`, and `SCALE_TAGS` job-level while both modules also defaulted
  them. The v10 commit bumped the modules and missed the workflow; the env wins,
  so CI ran v10's PA-share weighting and stamped every row `v9`. The 14 rows
  built 2026-07-28 carry v10 math under a v9 tag and are immutable. Detectable
  only because they hold a non-null `sp_bf_per_ip`, which no genuine v9 row has.
  Fixed in `2f5d922` by **deleting** the pins rather than syncing them, leaving
  a comment where they were that says why the block is empty — otherwise the
  next person re-adds them. A config value that duplicates a code default will
  drift; delete the copy rather than syncing it.

- **Freezing a measured number into a test.** Same shape as the constants entry
  above, one level out: `test_record_reproduces_ledger_report` asserted the
  ablation replay scored `39-32`, a literal copied out of `ledger_report.txt`.
  The Actions bot graded more slates, the real record moved to `45-37`, and the
  suite failed for a reason with nothing to do with the replay — the one gate
  this repo has, red on arrival. Fixed by *reading* the expected record off the
  report it names and intersecting the two row sets, so the assertion still
  fails for its real causes (replay drift, or a report not regenerated beside
  its ledger) and never for arithmetic the bot did overnight. A test that
  cross-checks two artifacts should read both, never memorise one.

- **Deleting controls as clutter.** The walk-forward Pythagorean control arm was
  added, then removed in a UI declutter three commits later, leaving the
  always-home F5 baseline in `ledger_report.txt` as the only control anywhere —
  and none at all on the public page, which published `200-151 (.570)` with
  nothing to read it against. Fixed by `_baseline_controls()`: always-home and
  always-chalk, scored on the identical graded rows, muted tiles in the same
  strip as the record. They are what makes the headline a result: at the time
  of that fix the pooled line read .570 against .504 always-home and .563
  always-chalk. Controls establish whether the model beats a trivial baseline;
  if one is visually noisy, mute it or move it to the ledger as a column — do
  not delete it.

  Read those three numbers as of that commit, not as standing facts. **And do
  not read the sentence this paragraph used to open with** — "the page scores
  `RECORD_TAGS`, so the headline reset when v9 started a new family" — which is
  wrong about the code and was verified so on 2026-08-06. `build_site`'s
  `_record_grades()` has *no production caller*; every public surface renders
  `_display_grades()`, i.e. every graded family pooled, which is exactly why
  the front page could publish `wOBA full 217-164` over 381 xwOBA games (see
  the metric-label instance below). A `MODEL_TAG` bump does not reset the
  public headline. It resets `ledger_report.txt`, which is the artifact that
  does score `RECORD_TAGS`. The 45-37 (.549) / 42-40 / 49-33 line quoted here
  for 2026-08-02 was the *report's* current-family line, not the page's.

  Controls, whatever the row set: the model ahead of the coin-flip control and
  behind the closing line, on a sample far too small to separate them — which
  is the controls doing their job. Do not quote a control figure from this
  file; recompute it. Since 2026-08-06 they are scored on the **decided** rows,
  not every graded one — see the abstention instance below.

- **Threshold cliffs.** Two instances, same shape. The old
  `use = fam if len(fam) >= 60 else pooled` scale selector switched
  discontinuously and mixed incompatible units; `SCALE_TAGS` removed that
  branch. Then `lean_strength_scale()` was found returning `None` below
  `LEAN_STRENGTH_MIN = 30`, so the cutoffs swapped the frozen
  `LEAN_STRENGTH_FALLBACK` for the pool's own p33/p80 in one step — worth up to
  0.0096 on p80, relabelling every game in the crossed band with no change to
  any lean. Fixed by deleting the gate and shrinking the observed quantiles
  toward the prior by pool size, `c = (n·c_obs + K·c_prior)/(n + K)` — the
  empirical-Bayes form already used for xwOBA, applied to a quantile. The
  prior stops being a branch and becomes the `n = 0` limit of one expression.
  **The general lesson:** when a selector has a hard `>= N`, the fix is usually
  not a better `N` — it is to make `N` a weight so nothing switches. And pick
  the weight by benchmark: the obvious `K = 30` (the old gate reinterpreted)
  measured *worse than the gate* on label stability; `K = 100` cut the worst
  single-row step 4.4× and landed nearer the population quantile.
- **Public claims the data can't support.** `grades.html` asserted every row
  locked before first pitch while `lock_status` was null on the pre-v3 rows.
  Fixed by `_lock_provenance()`, which now states the split instead of the
  whole — "*n* of *N* rows carry a pregame lock timestamp; 149 legacy rows
  predate that instrumentation." Only the 149 is fixed; the verified count and
  the total move with every build, so read them off the page or call
  `_lock_provenance()` rather than quoting a pair from here. (A pair frozen
  into this file on 2026-08-02 read 168 of 317 and was 277 of 426 four days
  later — the constants-frozen-from-data entry above, in prose.)
  Report provenance, don't assert coverage.
  The same entry recurred one level down: `_lock_provenance()` counted the
  unverified remainder *by subtraction* and the page labelled all of it
  "legacy rows predate that instrumentation", which would have described a
  `late_snapshot` row — instrumented and failed — as uninstrumented. Zero such
  rows exist today, so it was a claim waiting to go wrong rather than a live
  one. Now each outcome is counted from its own label. A count you derived by
  subtracting cannot carry a name you did not measure.

  Third instance, and the sharpest: the wOBA v1 commit replaced the literal
  `"xwOBA"` with `MODEL_RATE_LABEL` on the record strip, the grades headline and
  the team page. All three render `_display_grades()` — *every* graded family
  pooled — so the front page published **`wOBA full 217-164 (.570)`** over 381
  games of which exactly zero were wOBA; the first wOBA row had not graded yet.
  The same substitution broke the vs-market cell: `vs_market_summary()` keys its
  bucket off the rows (`"xwOBA"`), the caller looked it up under
  `MODEL_RATE_LABEL` with an `or "Model"` fallback that only fires on *mixed*
  metrics, so both missed and `z +1.09 (+3.14u)` — "the primary metrics" —
  silently vanished from two pages instead of raising. Fixed in `2d369c4`
  by `market_backfill.metric_label()`: one derivation, read off `model_metric` with
  the tag prefix as the legacy fallback, used by both the bucket and every
  lookup so a mismatch is now unrepresentable. A build-time constant must never
  name historical rows — the metric is a property of the rows, and the tag flips
  a slate before the first row under it grades.

  Fourth instance, found by an end-to-end check on 2026-08-06 and fixed the
  same day: **the grades page scored its baseline controls on every graded row
  while scoring the model on the decided ones.** A control needs no lean to
  score a game — always-home only reads the final score — so the moment v5's
  first abstention grades, `Always home` covers one more game than the record
  beside it, under a note reading "controls on the same graded rows". The `n=`
  marker that exists to catch exactly this compared the control's `w + l`
  against `len(g)`, the *graded* count, which is also inflated by the
  abstention — so the one discrepancy it was written for is the one it cannot
  see. Reproduced on a three-row frame (two decided, one abstained): the page
  rendered `wOBA · full 1-1 (.500)` beside `Always home 2-1 (.667)`, no `n=`,
  the control apparently beating the model on a game the model declined to
  call. Fixed by deriving `decided = g[g["xw_lean"].notna()]` once and passing
  it to the record, the controls and the marker alike, with the abstained count
  stated on the Graded tile from `xw_lean.isna()` — measured, not subtracted,
  per the instance above. Zero rows are affected today; v5 shipped the
  mechanism that arms it. **A control is only a control if it is scored on the
  rows the model was scored on — when a model gains the ability to abstain,
  every baseline beside it inherits that filter.**
- **Internal and public artifacts disagreeing.** `data/ledger_report.txt` once
  said the current family had no graded games while the site published a pooled
  record. Both now render from the same ledger through
  `RECORD_TAGS` — the report shows the current family plus immutable per-family
  history, the site shows the pooled headline. When you change one, change the
  other or state in the PR why they should differ.

## No-lookahead

`.savant_cache/` is gitignored and keyed by slate date. Leaderboard state as of
a past game is not recoverable, so historical rows cannot be re-derived from
today's Savant pull — that is lookahead, not backfill. Pending rows never
receive closing lines (`run_market_update.py` invariant). Do not relax either.

The one Savant pull that is *not* slate-dependent is a **completed** season:
2024's final line reads the same whenever it is fetched. `priors_snapshot.py`
freezes those into `data/woba_priors_<season>.csv` plus the per-season pool
centres in `data/woba_prior_centres.csv`, and the distinction is enforced
rather than trusted — it refuses any season not strictly earlier than
`build_site.SEASON`, and refuses to overwrite an existing season file without
`--force`. Both refusals are load-bearing. A season file is **immutable**: a
ledger row built against a prior has to stay reconcilable, and Savant does
occasionally revise history. Rewrite one only to repair a known-bad file, and
say so in the PR.

Store the deviation, not the level. A rate is only comparable across seasons
against the centre it was earned under, which is why the centres file exists
and why `centred_history()` carries `theta_s − mu_s` rather than `theta_s`.
Using a stored rate without its centre imports that season's run environment
into today's prediction — the same error as freezing a constant off the ledger,
one artifact out.

## The metric shadow arm

`shadow_metric.py` writes one extra dump per slate: the same games, built on
whichever rate the primary build is **not**. It publishes nothing — no lean, no
`index.html`, no ledger row — and its rows carry a `shadow_*` tag deliberately
absent from `_RECORD_FAMILIES` and `_SCALE_FAMILIES`. **Never pool a shadow row
into a record or a delta scale.** A test asserts the tag is in neither map.

**It swapped sides at v11 and that is the point, not a complication.** Until
v11 the primary ran wOBA and the arm ran xwOBA (`shadow_<date>_xw.csv`,
`shadow_xw+plat_consol_v1`); since v11 the primary runs xwOBA and the arm runs
wOBA (`shadow_<date>_woba.csv`, `shadow_woba+plat_consol_v1`). So *"the shadow
dump"* names a **side of the comparison, not a statistic**, and the committed
dumps hold both assignments. Anything reading them must key off `model_metric`
— `shadow_report.dump_metric()` does, falling back to the filename suffix only
for dumps written before that column existed, and `build_frame` assigns
`net_w`/`net_x` by metric so `d_corr` means "xwOBA minus wOBA" on every slate
either side of the changeover. Keying off primary-vs-shadow would silently flip
the sign of half the sample.

It exists because the wOBA-versus-xwOBA question **cannot be settled by
comparing the two eras**, and the temptation to try is exactly what produced
v11. The wOBA rows sit at 63-76 against always-home 84-55 while the xwOBA rows
sat at 217-164 against 193-188; the v9/v10-versus-wOBA-v5 gap is z = +1.63.
Real, and not evidence: the eras were *different games* — always-home ran .515
over the xwOBA rows and .604 over the wOBA ones — and five things changed in
six days, of which only wOBA v1+v2 isolates the metric, at n=16.

**What the arm has actually measured, which is the number to quote:** 6 paired
slates, 68 graded with both arms decided, metrics correlating +0.91 on net, 9
of 77 leans flipped, **d_corr +0.008 with CI [-0.108, +0.128]**. It does not
separate. Sequentially over those same 68 games the records read wOBA 29-39
against xwOBA 34-34 — a five-game gap on identical schedules, which is the
illusion pairing exists to remove. Pairing buys ~2x on se; the projection is
se ~0.045 at ~9 more slates and 80% power on a 0.09 gap at ~18.

So v11 reverted the metric **without** the arm having answered the question,
and the arm keeps running so the question keeps accumulating a paired answer
under the revert. Do not retire it because the primary is back on xwOBA — that
is precisely when a one-sided read would look most convincing.

Two things to know before touching it. It makes its **own** Savant fetch under
its own `STATCAST_CACHE_NS`, because it requests a different column and reusing
a cache written under another selection set returns a CSV missing the rate —
`STATCAST_SELECTIONS` and the cache namespace are a pair and must move together.
`patch()` now reads the primary column off `build_site` before overwriting it
rather than naming it as a literal, which is what let the two arms swap without
touching that function. And the dump is `shadow_*`, not `leans_*`, because
`grade_leans` globs `leans_*_xw.csv`, which matches any leans-prefixed name
ending `_xw.csv`: a suffix-based name would have been ledgered as a real
pending row. That is a prefix decision, not a naming preference, and a test
pins both halves — under both suffixes — against `grade_leans`' actual globs
rather than a copy of them.

**The arm paid for itself before v11 shipped, and this is the precedent to
keep.** `xwoba` is now on the primary build's critical path. It is there safely
only because the arm resolved the name against the live endpoint first: Savant
is unreachable from the dev environment and from CI, so the selection name had
never been confirmed until Actions run 31435698461 logged `shadow: rate column
'xwoba' resolved on 20/20 players` and printed a distinct league baseline
(xwOBA 0.31548 against wOBA 0.31628, same slate, same leaderboard) — proving
the column exists and is not a relabelled copy. **Never put an unverified
Savant column on the path that produces irreplaceable pregame rows.** Verify it
on the shadow arm, where being wrong costs a log line, and promote it after.

## Before opening a PR

```
python validate_data_files.py     # CSV conflict markers — has failed twice in prod
python -m pytest tests/ -q
```

`.github/workflows/tests.yml` now runs both on every pull request and every
push to main, so this is a real gate rather than an honour system. Run it
locally anyway when you touch `build_site.py`, `grade_leans.py`,
`market_backfill.py` or `actuals_backfill.py`, and paste the output in the PR —
CI tells you *that* something broke, the local run tells you before you push.

`requirements.txt` still has no pytest, on purpose: the workflow installs it
alongside, and the build job has no reason to carry a test dependency into
production. Install it yourself to run the gate.

The suite is **not** wired into `build.yml`, and that is a data-integrity
decision. The build commits pregame snapshots, and the ledger only accepts a
row whose snapshot predates first pitch — so a test failure blocking the daily
build would cost that slate's rows permanently. They cannot be re-derived
afterwards without lookahead. Tests gate the change; the build runs against a
main branch the change already passed on.

`tests/test_ledger_invariants.py` runs against the committed ledger rather than
constructed frames: phase algebra (`mx = q·mx_sp + (1−q)·mx_bp`, shares summing
to 1, one league baseline behind every phase edge), the PA-share weight against
measured BF/IP, the v7 abstention rule, grades against the linescores beside
them, and the no-lookahead invariant on pending rows. It asserts no counts or
records — a violation there is a writer bug, not a stale expectation. Any new
assertion added to it must hold that line.

Do not commit `data/` by hand — the Actions bot owns it. Do not commit
`public/`.

## Load-bearing, change with care

- `concurrency: site-build` with `cancel-in-progress: false` — serializes
  ledger commits. Removing it interleaves writes.
- Build exits non-zero without writing `index.html` on fetch failure, so the
  last good page stays live. Preserve that ordering.
- `timeout-minutes` on both jobs — an upstream API hang otherwise burns minutes.
- Score verification in `attach_market` — it correctly rejected the All-Star
  Game join. Do not loosen to raise the match rate.
