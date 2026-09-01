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
**v12** — Savant xwOBA, `XWOBA_SHRINK_K = 100`, population shrinkage
targets, calibrated expected starter IP.

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
| v12 | `expected_sp_ip` calibrated per build against its own backfilled actuals (over-dispersed, slope 0.735) | v12 | 8+9+10+11+12 |

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

**v12 is the counter-precedent to v11 on the record question: a bump whose
decision-equivalence measurement argued for *sharing* and which isolated
anyway, for a reason that has nothing to do with the model.** The expected-IP
calibration flips 1 lean in 254 with mean |Δ net| 0.00067 against a median
|xw_net| of 0.01694 — the same order as v10's reweight, which earned a shared
line. Sharing a record exists to avoid resetting a *graded sample* for a change
that decides the same games; v11 had no graded rows, so there was no sample to
protect and a clean line was free. Read that as cost-benefit, not as "a bump
means isolate" — if v11 had graded rows, the measurement above argues the other
way, and it is recorded in `_RECORD_FAMILIES` so a later reader can see which
way the evidence pointed independently of what the reset happened to cost. The
scale half is the ordinary v10 argument and is measured: `q` is a convex weight
between the same two phases, so the units are untouched.

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

**One correction to that paragraph, and it cuts the other way on K.** The
sentence above is right that no measurement favoured 100, but wrong to leave
the K fit standing as a live objection to it. `reliever_shrink_probe` builds
its rates from StatsAPI box lines (`WOBA_W`, `woba()`), so what it fits is a
**wOBA-denominated K** — correct when it ran, because the build was on wOBA and
400 shipped as wOBA v3. Under v11 the same constant shrinks xwOBA, and
`K = σ²/τ²` is a property of the metric: xwOBA is near enough wOBA's
conditional expectation given batted-ball shape, so by the law of total
variance its per-BF `σ²` is strictly smaller. `τ²` is not pinned, so that is a
direction and not a magnitude — but the direction is toward a *smaller* K, i.e.
toward the 100 now shipped. So K=100's status under v11 is **unmeasured, not
overridden**, and "every interval excludes 100" is a true statement about a
statistic this build does not use.

It also cannot be re-measured here: StatsAPI serves no xwOBA and a per-past-date
Savant pull is the lookahead `.savant_cache/` exists to forbid, so the probe
cannot be re-pointed at the live metric. Re-running it answers the wOBA
question again, accurately, about a constant this build no longer has. The same
metric-denomination caveat applies to `player_prior_probe`'s +7–25%: the frozen
priors are wOBA and `player_prior_history()` refuses them to an xwOBA build, as
the priors section below already states. Neither retracts a fit; both narrow
what the fit is about.

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
rows against the v9/v10 pool, and split the family if it moved. **It has since
been run to its limit and closed — the share stands.** v11 graded no rows so
the check fell to v12; see the "Instrumented and waiting" entry for the three
reads and for why the test cannot be sharpened by waiting.

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
graded. **It is no longer the only one**: a second fired on 2026-08-12,
CHC@WSH, on the home side (`pitching_basis_home=starter_unmeasured_no_lean`),
and has also graded. This sentence has now been wrong twice in the same way —
it read "no undecided rows yet", then "the only leanless row in 482" — so it is
fixed here as a mechanism rather than a count: **v5 abstentions are rare, they
accrue, and the number is `xw_lean.isna()` on the graded rows, not a figure in
this file.** v7's zero-delta abstention still has never once fired at full
precision, so v5 remains the only mechanism that actually produces these.
`_rec()` drops those rows while `len()` counts them, so every count that mixes
the two must say which it is — see the controls entry below for the one surface
that did not, now a live discrepancy rather than an armed one.
`ledger_report.txt` states the split per family on its history lines (the wOBA
v5 line carries its own `(n abstained)` marker), and the grades page scores its
controls on the decided rows. Read both off the artifacts; the quoted
`55 graded games (54 with a lean, 1 abstained)` that stood here was a
current-family line from a family that is no longer current.

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

**v11 is the second instance and it is not inert.** No `xw+plat_consol_v11`
row exists in the ledger at all — not graded, not pending, not void. v11 shipped
and v12 bumped before any of its rows survived a pregame refresh, exactly as v8
did. So the current scale pool is v9/v10/**v12**, and every sentence in this
file describing v11's units as "argued, not measured, first graded v11 rows are
the check" was describing rows that will never exist. The check fell to v12
instead, and it has now run — see the falsifier entry under "Instrumented and
waiting", which carries the measurement. What is *not* inert is the reading:
the revert's scale claim was never tested on the revert, only on the version
after it. That is fine here because v12's own scale argument is independent and
measured, but do not write "v11 joined the pool and its rows confirmed it."

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

- **Every dump written before 2026-08-16 is a rebuild, and a mid-slate dump is
  still a mixture.** The overwrite itself is fixed — see the resolved entry
  below — but the fix is not retroactive and does not cover the intra-day case,
  so both halves of this stay live.

  The history: `SLATE_DATE` rolls over at 3am ET and the daily grading cron
  fires at 04:17 UTC — 00:17 ET — so it still names *yesterday's* slate,
  re-runs the full build against today's Savant leaderboard, and used to
  rewrite that slate's dump in place. Measured across every committed dump
  carrying `snapshot_utc`, every past slate's dump was a post-first-pitch
  rebuild. `leans_2026-08-05_woba.csv` carries `model_tag=woba+plat_consol_v5`
  and `snapshot_utc=2026-08-06T04:18Z`, while all 15 of that slate's ledger
  rows are v3/v4 with pregame snapshots and different numbers — game 822866's
  `starter_xwoba_away` is 0.327266 in the ledger and 0.336786 in the dump. The
  dump on disk is stamped with a tag whose math produced none of that slate's
  rows. **Nothing recovers those**; the pregame versions were overwritten and
  no-lookahead forbids reconstructing them. So any measurement that joins
  ledger rows to "the dump beside them" is still reading post-hoc data for
  every slate up to 08-16 — including `FINDINGS.md`'s prior-only incidence
  ("6 of the 403 side-games in the committed dumps"), and `compare_v8_v9.py`,
  which globs `data/leans_*_xw.csv` and therefore compares against a v8 dump
  that is itself a rebuild of the 07-26 slate.

  The residue going forward is narrower and worth stating exactly. A dump is
  diverted only when **every** game on it has started, because a slate with any
  pregame game left is not a reconstruction. The 15-90 minute pregame polls and
  any push to main therefore still rewrite the live dump mid-slate, and a
  late-window build carries pregame rows for the night games beside post-hoc
  rows for the afternoon ones. Measured on the committed dumps: 13 of the 43
  instrumented ones are full rebuilds, and most of the rest are mixtures of
  exactly this kind — `shadow_2026-08-11_xw.csv` was written at 00:45Z with 12
  of its games not yet started. Each row is labelled honestly by `lock_status`,
  the ledger takes only the pregame ones, and shrinking the window further
  means merging dumps rather than naming them — a different change with a
  different risk. Do not read "not a rebuild" as "pregame throughout".

  **v11 changes `compare_v8_v9`'s glob population and this is recorded, not patched.**
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

  **The shadow arm inherited this and had no ledger behind it**, which is why
  the fix below landed on the arm and the primary together. `shadow_metric`
  runs as a step of the same build, so the shadow dump was rewritten by the
  same post-rollover pass: `shadow_2026-08-10_xw.csv` on disk is stamped
  `2026-08-11T06:50Z` against a 23:07Z first pitch, and 08-09's only committed
  version is the 03:06Z rebuild — the arm landed at 02:56Z that morning, so a
  pregame 08-09 dump never existed. For the primary this costs provenance and
  the ledger holds the pregame truth; for the shadow arm the dump *is* the
  record, and git history was the only place a pregame version survived (08-10
  has four, the last at 23:01Z, six minutes before first pitch).
  What this does **not** break is the pairing, and that distinction is the
  whole reason the arm is worth reading: primary and shadow are written 20-40
  seconds apart in the same job from the same leaderboard, so dump-against-dump
  is honest even when both are rebuilds. It is dump-against-*ledger* that is
  contaminated — a pregame wOBA decision against an xwOBA one with an extra day
  behind it. `shadow_report.py` pairs the dumps for that reason, prints each
  slate's provenance instead of assuming it, and will run the ledger join under
  `--ledger-join` so the bias can be sized rather than argued.

**Removed — recorded so the reasoning is not relitigated**

- **The walk-forward backtest, deleted 2026-08-27 on the operator's call.**
  `walkforward.py`, `historical_data.py`, `tests/test_walkforward.py`,
  `docs/walkforward.md`, `walkforward.yml`, `export-savant-cache.yml` (which
  existed only to feed the replay's `exact_pregame` fidelity) and both
  `data/walkforward_*` artifacts. Recoverable from git history; nothing else
  imported them, and `reliever_shrink_probe`'s own walk-forward K fit is
  unrelated and untouched.

  **What was and was not established, because the removal followed a review
  and should not be read as that review's conclusion.** Three defects were
  found and fixed first: the replay cache was keyed on raw file bytes so any
  edit (comments included) discarded it and restarted from the first slate;
  the report printed `Games 690` beside a 500-row frame with no marker; and a
  450-second bound inside `build.yml` meant it never finished. Those were
  plumbing. The prediction math was NOT found wrong — the replay imported
  `build_site` and ran the same code, with inputs correctly bounded to
  `slate_date - 1` and no lookahead.

  What stayed open was narrower: the replay reconstructed rates from Savant's
  `statcast_search` while the live build reads the leaderboard, and 0 of 500
  rows used the archived-cache `exact_pregame` path. That is an **unmeasured
  fidelity gap, not a demonstrated error**, and it was never closed.

  **What the repository loses.** The backtest was the only out-of-sample
  control on the model, and it disagreed with the live panel: over 490
  replayed decisions it scored -1.6pp against price (z -0.74, -7.0% flat ROI)
  where the live v12 window over 160 rows scored +7.9pp (z +2.03, +11.8%).
  Against always-chalk the same two windows are +1.8pp and -3.7pp, so most of
  the raw 63.1%-vs-52.9% gap was base rate. With the replay gone, nothing
  contradicts the live figure, and `Deleting controls as clutter` below is the
  entry this trades against. Anyone reinstating a backtest should start from
  the fidelity gap above rather than rebuilding the same reconstruction.

**Resolved — keep as precedent**

- **A timeout that kills the step but not the process.** The walk-forward
  append was `continue-on-error` with `timeout-minutes: 8` precisely so a slow
  replay could never cost a slate. It cost one anyway. (The replay itself has
  since been removed — see the entry below — so this reads as history. The
  RULE survives it, which is the point of keeping it: the hazard belongs to
  the pattern, not to that step. `WorkflowStepTimeoutTests` still enforces it
  against any future step, and now guards zero live instances by design.) A step timeout kills the
  step's shell and moves on; the `python walkforward.py` child survives as an
  orphan the runner only reaps in post-job cleanup, and it rewrites
  `data/walkforward_ledger.csv` after every date it finishes. Run
  33073467257: the step timed out at 12:55:07, the commit step ran
  `git add data/` and committed at 12:55:50, the orphan rewrote the ledger a
  fraction of a second later, and `git pull --rebase origin main` refused with
  `cannot rebase: You have unstaged changes`. The build was otherwise clean —
  7 games, 13 sides, both dumps written, 6 new ledger rows ingested — and none
  of it was pushed. Pregame rows, so no-lookahead means the next run cannot
  re-derive them.

  Fixed by bounding the **process**, not the step:
  `timeout --signal=INT --kill-after=60 450 python walkforward.py`. `timeout`
  blocks until the child is dead, so no step after this one can find `data/`
  moving underneath it; `timeout-minutes: 8` becomes a backstop that should
  never fire. SIGINT rather than SIGTERM because `_atomic_csv`'s `finally`
  removes its temp file on a `KeyboardInterrupt` and not on a default-handled
  SIGTERM — verified, `finally ran` — with SIGKILL 60s behind it. `*.tmp` is
  now gitignored for the case where it does not: `git add data/` runs *after*
  `validate_data_files.py`, so a temp file appearing in that gap would stage a
  partial CSV past the one check written to catch exactly that. The ignore
  still carries that weight under `commit_data.py`, which enumerates
  `git status --porcelain` and therefore never sees an ignored file either.

  **The general lesson: `continue-on-error` and `timeout-minutes` bound the
  step's effect on the job, not the process's effect on the working tree.** Any
  step that writes to `data/` and expects to be cut off has to bound its own
  process. Pinned by `WorkflowStepTimeoutTests`, which asserts it of *every*
  step-level `timeout-minutes` in `build.yml` rather than of this one step, so
  the next such step inherits the rule. No lean, delta, grade or ledger row
  moves; no `MODEL_TAG` implication.

- **A raw win-loss quoted where only a price-relative one means anything.** The
  per-game "Model vs market" row printed the lean's raw record in its bucket —
  side × agree/disagree with the closing favourite. Those four buckets spanned
  **24 points** of win rate (.603 home-agree down to .360 home-disagree) and
  every one of those points was base rate: scored against their own devigged
  prices the same buckets read +1.5, +0.3, +1.1 and −11.4 pp, each inside
  1.2 se of zero. A reader was shown the market's opinion of the matchup and
  invited to read it as the model's skill — the same defect as publishing a
  record with no control beside it, one surface out.

  Fixed by banding on the **leaned side's** own devigged price and reporting
  that band's gap against price with `_excess_se`, the derivation both
  calibration surfaces already share. The old home-relative split also filed
  the 10 graded rows priced at exactly .500 as "home favoured" — and the 6 of
  those where the model leaned away as *disagreeing with a market that had no
  favourite*. A band has no such boundary claim to make. `VERDICT_CONTEXT_MIN`
  went 10 → 25: a raw rate is unreadable at n=10 and a price-relative one is
  honest at any n, but a band still needs enough games for its se to mean
  something.

  **The band is gone; the finding is not.** The V12 rewrite replaced it with
  the direction-specific conviction cell (`_conviction_tail`), because a band
  pools a favourite and an underdog into one historical result and erases the
  direction the panel exists to show. That rewrite removed the three call
  sites and left the callees behind, so `price_band_records`,
  `_price_band_tail`, `_price_band`, `_PRICE_BANDS` and `VERDICT_CONTEXT_MIN`
  sat for two days as a fully-tested surface no page rendered — six tests
  pinning a function with no production caller, the same shape as the
  `_record_grades()` note in the controls entry below. All of it is now
  deleted. The numbers above are the reason the raw record went, and they
  stay here because this file is where that measurement lives; do not read
  them as describing code that still exists.

  Display-only; no lean, delta, grade or ledger row moves.

- **The value-bet signal that does not exist, and the measurement that says so.**
  Asked for a per-game "this is a value bet" badge, the honest answer turned
  out to be that no such badge is available in this data, and the naive version
  is **inverted**. Recorded here because the request is a natural one and will
  recur.

  Walk-forward over 552 games / 42 slates, fitting only on prior slates:

  * Flagging the largest model-vs-market probability gaps selects the **losing**
    subset, monotonically: gap > 0.00 → −9.5% ROI, > 0.06 → −20.2%, > 0.10 →
    −33.0% (n=40). Flagged bets total −18.00u over 89 bets across 31 slates.
  * It is not "the model's dog picks lose". Controlling for price band, the
    above-median-gap half loses to the below-median half in **all four** bands
    (big dog −18.3% vs +10.5%; small dog −14.1% vs −10.0%; small fav −7.6% vs
    +9.8%; big fav −7.6% vs −1.5%).
  * Does the delta add anything *on top of* price? Joint logit, n=627:
    market logit **+1.25 ± 0.35** (consistent with 1.00 — the close needs no
    correction), `z(xw_net)` **−0.09 ± 0.12, z = −0.78**. Out-of-sample log
    loss ranks **raw close 0.6768 < market-fitted 0.6823 < price+delta 0.6888 <
    delta alone 0.6991** — adding the delta to the price makes prediction
    *worse*, which is what a noise feature does.
  * Every arm lands within 1.4 se of its own market and loses units at the
    close: model full-game z +0.29 (−15.28u over 627), platoon full-game
    z −1.41 (−46.69u over 493), platoon vs the F5 close — the market it
    actually targets — z +0.50 (−14.59u over 417).

  So the verdict row reports price *context* and never a recommendation, and a
  test pins that its copy contains no betting language. **Do not re-derive this
  by hand and do not ship a value call without re-running the walk-forward
  first**: the failure mode is that a hot current family (v12 ran .646 over 79
  games while always-chalk ran .633 on the identical rows, in a stretch where
  favourites beat their pooled rate) makes the badge look justified on the
  rows in front of you.

  **That rule named an instrument that had been deleted, and `value_probe.py`
  is what makes it satisfiable again.** The walk-forward went on 2026-08-27,
  so between then and now "re-run the walk-forward first" could not be
  complied with — the one guardrail on the most-likely-to-recur request in this
  repo was a dangling pointer. The probe is a walk-forward over *ledger rows*
  rather than a replay: every row is a decision actually published pregame,
  joined to its own close, fitted only on prior slates. That is strictly less
  than the replay could do — it cannot score a version that never shipped, and
  its header says so — but it is enough for this question and it never
  reconstructs a prediction, so the fidelity gap that removed the replay does
  not apply to it.

  **Re-measured 2026-08-29 on 716 graded rows (the entry's own numbers were
  n=552/627), and every leg reproduces.** The naive bet is still inverted and
  still monotone: all bets −3.1%, gap > 0.00 −9.7%, > 0.02 −11.4%, > 0.04
  −13.3%, > 0.06 −17.7%, > 0.10 −28.8% (n=32). Within-band median-gap splits
  still lose in all three bands thick enough to split. The joint logit still
  says nothing: market logit +1.137 ± 0.294, `z(xw_net)` **−0.013 ± 0.095,
  z = −0.13**, and out-of-sample log loss still ranks raw close 0.6751 <
  market-fitted 0.6799 < price+delta 0.6841 < delta alone 0.6946.

  **And the trap fired exactly as predicted, which is the part to keep.** Read
  alone, v12 looks like a system: .624, +0.0725 against price, z = +1.98,
  +10.7% ROI over 181 rows. Two things dissolve it, and both are printed beside
  it by design — always-chalk ran **.613 on those identical games**, leaving a
  +1.1pp edge rather than a .624 one; and the same statistic per family flips
  sign with the era (v5 +1.09, v7 −0.68, v9 −0.50, v10 +0.73, wOBA v5 −2.74,
  v12 +1.98), which is noise, not edge. Do not quote any of these figures from
  here — run the probe.

  **Banding the delta does not rescue it, and section 5 of the probe is there
  because that is the natural next idea.** A 3-band |Δ| × 5-band price grid was
  measured on the same 716 rows, both markets. The grid looks alive — full-game
  cells run from −28.7% to +21.5% — and it is entirely noise: no cell reaches
  even 2 sd (largest deviation −1.63), because at 21–82 rows a cell's null sd is
  8–21 **percentage points** of ROI.

  The decisive comparison is against a **maximum**, not against zero, since a
  15-cell search returns the best of 15 draws. Simulating outcomes at the
  devigged prices under "market correct, no edge", the best of the 14 eligible
  cells averages **+20.4%** full-game and **+16.0%** on F5 — against observed
  bests of +21.5% and +14.6%. So the full-game winner is exactly what chance
  produces (p = 0.38) and the F5 winner is *worse* than chance (p = 0.53).
  Selecting the best cell on prior slates and betting it on the next loses in
  both markets: −13.8% over 15 bets full-game, −3.0% over 50 bets on F5. Neither
  margin ordering is monotone in either direction, which is the tell — a real
  effect would show structure, not scatter.

  **The general rule this earns: on this data, any grid search will hand back a
  cell near +20% ROI whether or not anything is there.** Judge a cell against
  the null max and a forward test, never against zero. Do not add bands to a
  signal that scored z = −0.13 undiscretised — cutting noise into bins makes
  more maxima to be fooled by, not more signal.

  **The inverted version was tested too, and it is the sharpest instance of
  search-driven self-deception in this repo.** If backing the biggest gaps
  loses monotonically, *fading* them should win — a real over/under-valuation
  signal with its sign flipped. On a walk-forward search it looked strong:
  +20.2% ROI at gap > 0.10, an inversion slope whose bootstrap CI excluded zero
  (−1.37, [−2.48, −0.15]), and — the part that made it convincing — the obvious
  debunk failed, because always-chalk *lost* −12.3% on the identical rows, so
  it was not simply favourite-backing.

  It was an artifact of the search's own machinery. The walk-forward refits
  `logit(p_home) ~ xw_net` each slate on prior slates only; early slates fit on
  75–102 rows produce slopes of **+8.5 to +9.8 against a stable full-sample
  +5.78**, and an inflated slope inflates `|gap|` on exactly those slates. So
  "gap > 0.10" was selecting **unsettled fits, not disagreement** — 69% of that
  set was fitted on <200 rows against a 20% base rate, and all five underdog
  wins carrying the result landed on three consecutive slates (2026-07-08/09/10)
  inside that region. The 27 chalk-equivalent rows contributed +3.4pp of the
  +20.2%; the five dogs contributed +16.8pp, going 5-0 against 2.3 expected.
  Hold the coefficients FIXED and the effect inverts — every threshold negative,
  −5.7% / −1.8% / −0.1% / −4.2% / −7.8%. Corrected for having searched both
  directions × five thresholds, p = 0.20.

  **Three lessons, each with the instance above attached.** A statistic
  recomputed per slate carries its own fitting noise into whatever it selects,
  so "walk-forward" is not automatically clean — freeze the parameters and
  re-run before believing a threshold effect. A failed debunk is not a
  confirmation: chalk lost those rows because it lost *the same five games*,
  one cluster counted twice. And a hypothesis generated by looking at the data
  needs its p-value computed against the search actually performed, not against
  zero — the nominal CI excluding zero was true and meaningless.

  **The plus-money underdog cut is a separate question and it was tested
  properly, because it is the one hypothesis here that was NOT found by
  searching.** Favourite-longshot bias is a documented market phenomenon, so
  this had a real prior. Measured over the 605 graded games holding a
  plus-money dog: backing **every** dog runs 41.8% against 42.0% implied,
  z = −0.09, ROI −3.8% — **no favourite-longshot bias in this book to harvest;
  the dogs are priced right.** Restricting to dogs the model leans (n=128) gives
  46.1% against 44.7% implied, z = +0.32, ROI −0.4%: better than backing them
  all, and the selection value is +3.3pp with CI [−13.2, +20.4].

  Cut by price the model's dogs at +100..+130 run **+7.5% ROI over 90 rows**,
  and this one has properties none of the earlier candidates had — it survives
  dropping its best three results (+7.5% → +3.5%, so it is not a cluster), and
  walk-forward it stays positive (67 bets, +3.8%). It is still not a finding:
  excess +4.9 ± 5.3 (z = +0.93), selection value within the band +5.7pp with
  CI [−12.4, +23.8], and against the null the best of the cells searched
  averages +12.3% — **P(chance ≥ observed) = 0.64**, i.e. the winner is *worse*
  than chance across that search.

  It is registered as `forward_test.py`'s **arm 2, deliberately UNBANDED**, with
  a NULL rather than negative prior. The band is the flattering number and a
  reader will want it registered — but it was chosen after seeing it, so
  freezing it would smuggle the search back into a pre-registration. The
  a-priori hypothesis is "the model leans a plus-money dog"; that is what is
  frozen, and the bands print as context carrying no claim. A test pins the
  distinction by asserting the headline counts every plus-money lean rather
  than the band's subset.

  `forward_test.py` is what came out of it: the rule frozen at registration
  (2026-08-29), scoring only slates strictly after that date, printed every
  build, with its registered prior stated as **negative** so a future hot streak
  cannot read as a discovery. `tests/test_forward_test.py` pins every registered
  constant — the one place in this repo where freezing measured numbers into a
  test is correct, because there the literals *are* the subject and editing one
  must be deliberate. Its gate is ~1,300 bets at ~1.6 a slate; read nothing
  before then.

  **The hybrid market-direction rule is the third registration, and it is the
  clearest instance yet of the trap this whole entry exists for.** Proposed as
  a market guardrail on top of the v12 lean — follow the lean when the market
  gives the selected side at least 45%, back the other side below that — it
  reproduces exactly on the ledger: 223 eligible v12 rows, hybrid 146-77 for
  +31.50u (+14.1% ROI, z = +2.72 against price) where the plain lean ran 139-84
  for +22.63u (+10.1%). Every figure in the proposal verified, including its
  own bootstrap intervals. It is registered rather than shipped, and the two
  measurements that decided that were not in the proposal:

  * **The headline is mostly the model, not the rule.** The switch fires on 15
    of 223 selections; the other 208 are v12 unaltered, which is where the
    z = +2.72 comes from (the follow branch alone is z = +2.50). What the rule
    *adds* is the paired switch delta: +0.592u per switched game, se 0.486,
    **z = +1.22**, paired ROI CI [−2.6, +10.4] pp with P(≤ 0) = 0.11. So the
    registered headline is the switch delta, never the hybrid's ROI — a
    combined line can only restate what the model already does.
  * **The fade branch is always-chalk, exactly and by construction.** Fading a
    lean priced under .45 backs a side priced over .55, which is always the
    favourite — verified 15 of 15. Its 11-4 and +23.8% *is* the always-chalk
    record on those rows, to the unit, in a window where chalk beat its price
    by +4.0pp over all 223 (137-86, +4.5%). A rule whose only active branch is
    favourite-backing, measured over 15 games in a favourite-friendly stretch,
    is this entry's own failure mode with a new formula on it.

  **One thing genuinely cuts the other way, and it is recorded because the
  honest answer is not "no".** Judged by the search test this repo demands — a
  threshold found by looking is scored against the null maximum, never against
  zero — 0.45 survives: sweeping 0.30..0.56 in 0.01 steps it is the argmax of
  27 candidates at +14.1%, and under "market correct, no edge" simulated at the
  devigged closes the best of those 27 averages +4.7%, giving
  **P(null best ≥ observed) = 0.019**. That is a far better showing than the
  band grid's p = 0.38. It still is not a result: the sweep is flat near +10%
  below 0.44 because the fade branch is empty there, so the entire spike is
  those same 15 games, and no permutation can manufacture the independent
  sample the gate needs.

  `hybrid_test.py` is what came out of it, alongside
  `tests/test_hybrid_test.py`. Registered 2026-09-01, scoring only slates
  strictly after, prior stated as **null**. It is a separate module rather than
  a third arm of `forward_test.py` so neither registration's frozen block can
  be edited while reaching for the other's. Its tests pin the constants and, in
  addition, the two structural properties the reading turns on: that a followed
  game has *identically* zero switch delta (so the headline cannot absorb the
  model's own performance), and that the fade branch backs the favourite on
  every row (so a good forward run is read against the always-chalk control the
  module prints beside it). Two gates, because they answer different questions:
  ~41 switches asks only whether the effect is anywhere near as large as it
  looked, and ~1,420 is what a plausible +0.10u-per-switch edge needs — at the
  observed 0.88 switches a slate, roughly ten seasons. **Read nothing before
  the first, and do not read the second as reachable.**

  **What it cannot do, and the one thing the protocol asks for that this repo
  cannot yet supply:** the rule is specified against the no-vig probability
  available *at decision time*, and the ledger carries only the close, because
  the no-lookahead invariant keeps every market column off a pending row. So
  the module scores the closing basis — the same basis `forward_test.py` uses
  and the same one the discovery numbers were measured on, so it is consistent,
  but it is a CLV reading rather than an obtainable-price one. Closing that gap
  means persisting a decision-time price pregame: `fetch_pregame_odds` already
  computes exactly that `p_home` and renders it on the card, but it is called
  *after* the dump is written and is never stored. Deliberately not done here —
  it moves the critical path that commits irreplaceable pregame rows — and the
  shape it should take if it is: leave the dump write where it is, then enrich
  it in a second best-effort pass, so a failed fetch costs the decision price
  and never the slate. Until then an operator paper-tracking this rule records
  their own obtainable price separately, and the module's header says so.

- **An error bar estimated from the outcomes it is testing.** Both surfaces on
  `market-calibration.html` print a realised rate against its implied one, and
  both sized the `±` from the results rather than from the prices. The ladder
  used `sqrt(p̂(1−p̂)/n)`, which is exactly `0.0` on any bucket that went all-W
  or all-L; the value panel used the sample sd of the residuals, which
  collapses the same way because the only variation left is in `market_p`.
  Both fail in the direction that makes noise look like signal: the *least*
  certain buckets on the page render as the most certain.

  Not latent. Two rungs were live at the fix — a one-game `≤ -250` bucket
  publishing `+27.3` against implied with `±0.0` beside it — and the panel's
  thinnest bucket read `−40.4 ± 1.6` against a true `±24.5`, an apparent
  25-sigma result on four games.

  Fixed by asking what null the number is testing. Each observation is an
  independent Bernoulli at its own devigged price, so the win count is
  Poisson-binomial: `Var(Σ wins) = Σ p(1−p)` and the SE of the mean is
  `sqrt(Σ p(1−p))/n`. The `p_i` are fixed by the market rather than estimated
  from the outcomes under test, so it is defined at `n=1` and cannot
  degenerate. On the large buckets it barely moves (pooled home `.0201` →
  `.0198`); it only bites where the old form was worthless.

  Three things worth keeping. **One derivation, not two** — `_excess_se()`
  serves both surfaces, the same move as `metric_label()` in the instance
  below, because two copies of a statistic on one page will drift and the
  reader cannot see which they are reading. **A statistic needs a spread on
  every axis it consumes, not just a second row** — the same commit guarded
  `np.corrcoef` on the sd of *both* columns, which the slope beside it already
  did and the correlation did not; a constant column returned a silent `nan`
  from a divide-by-zero. And **an SE of zero is never a result** — it is the
  estimator saying it has nothing to say. Treat one as a bug on sight, in the
  same reflex as the ratio with no sampling distribution below.

  Display-only: no lean, delta, grade or ledger row moves, so no `MODEL_TAG`
  implication. What changed is the error bars beside published numbers, not
  the numbers.

- **A column carried to no surface, second instance — and the note that made
  it invisible.** `_lean_market_agg` computes `excess_se` for every model×market
  bucket. The 2×3 calibration table renders it; the per-game 3×5 profile panel
  did not, printing `Performance vs market +33.8 pp` with nothing beside it.

  What kept it hidden was a docstring: `conviction_cell_records` said "the card
  prints `n` … the calibration table carries the error bar", and that is false.
  The table renders the **2×3 `cell`** buckets (LOW/ACTIVE × oppose/no-backing/
  agree); the card renders the **3×5 `profile`** buckets (low/medium/high ×
  five price bands). Different cuts, different n — a profile cell has no
  counterpart there, so its SE reached no surface at all. Same shape as the
  instance below: a note describing a diagnostic the reader cannot see.

  Not latent, and worst exactly where it matters. `CONVICTION_CELL_MIN = 1`, so
  a single completed game publishes a headline: live at the fix, `low × >65%`
  read **+33.8 pp off n=1**, against an SE of **±47.3**. Three of the 14
  published cells sat under n=5. The fix renders the SE the aggregate already
  returned, and a test pins the n=1 case specifically — `_excess_se` is defined
  there, where the p̂-based forms this repo already banned would print ±0.0.

  Two things deliberately NOT changed, because they are the operator's call on
  a surface requested as exploratory: the `CONVICTION_CELL_MIN = 1` floor, and
  the `n ≥ 20 → "LARGER SAMPLE"` label (n=20 still carries ±11pp). The error bar
  is what makes both readable rather than misleading, which is why it was the
  half worth fixing unasked.

  Display-only: no lean, delta, grade or ledger row moves.

- **A column carried to no surface.** The same panel computed a per-game
  `price_dislocation` residual, returned it on the observation frame, and
  rendered it nowhere — and the note beside it described the invisible
  diagnostic to the reader as though a table showed it. The prose was the
  symptom; the unread column was the cause, because nothing tied what the note
  claimed to what the tables emit. Deleted rather than surfaced: a residual-sign
  cut may be worth adding later, and adding it then is cheaper than carrying a
  column that invites a second description of something nobody can see. A test
  now asserts the frame carries no unrendered column.

- **A rebuild overwriting the record it was rebuilding.** The post-rollover
  pass rewrote each past slate's dump in place, so the one artifact saying what
  the model saw before first pitch was replaced by what a later model saw with
  a later leaderboard. Three options were on the table for a year of this file:
  skip the write, rename it, or accept it. **Renamed** — a rebuild is a
  legitimate later view of the same slate and the probes read it happily, so
  deleting information to protect information was the worst of the three.
  `dump_is_post_hoc` decides and `dump_path` names; both the primary and the
  shadow arm call them, so the two arms cannot disagree about what a slate is.

  Three things in it are the reusable part:

  * **The marker is a PREFIX, for the same reason `SHADOW_PREFIX` is.** The
    grader globs `leans_*_xw.csv`, which matches any leans-prefixed name ending
    `_xw.csv`, so `leans_<date>_xw_rebuild.csv` would have been ingested as a
    real pending row — the highest-cost silent failure available here. A test
    asserts every rebuild name against `grade_leans`' own globs, and a second
    pins that the naive suffix form *would* have matched, so the first cannot
    quietly become theatre.
  * **The rule is read off the rows, not off the clock.** "Is this a rebuild?"
    could have been `SLATE_DATE != today in ET`, which then has to re-derive
    the rollover hour, hold across DST, and is simply wrong whenever
    `SLATE_DATE` is overridden to rebuild an old slate by hand. The rows carry
    a snapshot and a scheduled start and answer it directly.
  * **Unknowable falls back to the live name.** No rows, no start column, an
    unparseable stamp — all keep the name every existing glob already finds. A
    dump wrongly marked `rebuild_` is invisible to the grader, which is the
    same slate loss the prefix exists to prevent, reached from the other side.

  Measured at the fix, on the committed dumps: 13 of the 43 instrumented ones
  are full rebuilds that would have been diverted, including both shadow dumps
  named in the live entry above. Display/provenance only — no lean, grade or
  ledger row moves, so no `MODEL_TAG` implication. The ledger's own guard is
  untouched and stays load-bearing: `ingest()` admits a row only when
  `lock_status == "pregame"`, which is what kept every one of those rebuilds
  out of the ledger while this was broken. This change means the grader is
  never offered them; it does not mean the check can be relaxed.

- **A monitor that measures its own correction.** `sp_ip_calibration()` reads
  `expected_sp_ip_raw` where present precisely so the fit cannot feed on its own
  output — and the standing monitor that prints its slope every build,
  `actuals_backfill.paired_sp_ip`, read the *published* column. From v12 that
  column is calibrated, so the printed "IP calibration slope" became a mixture:
  604 raw side-games and 30 corrected ones, with no label saying which. It read
  `+0.762` while the fit on the same 634 rows read `+0.756`.

  Harmless at 5% contamination and not harmless later, which is why it was
  fixed at sighting rather than gated. A calibrated pred is compressed by
  `w·b + (1−w) = 0.774` (measured: sd 1.338 raw → 1.017 published on those 30
  rows), so an all-v12 sample prints ≈0.98 — a monitor announcing that the
  defect it exists to watch has resolved, on a slope whose subject never moved.
  The deferral entry below it only worked because the instrument reported the
  estimator; an instrument reporting the estimator-plus-its-fix reports nothing.

  Fixed by giving the monitor the fit's own rule (raw where present, published
  where not) rather than by adding a second line for the published value.
  Nothing is lost: published is a deterministic function of raw and the fit, so
  monitoring raw monitors both. **When a correction ships, check what its
  monitor is now reading** — the column it always read may have changed meaning
  underneath it. Diagnostic only; no lean, grade or ledger row moves, so no
  `MODEL_TAG` implication.

- **A statistic with no usable sampling distribution.** `ledger_report.txt`
  printed `implied w = b_sp/b_lineup` from the SP-vs-lineup logit fit, with no
  standard error beside it — because it has none. `b_lineup` is not
  distinguishable from zero (+0.122 ± 0.227 over the 82 graded v9/v10 rows), so
  the ratio is Cauchy-like: bootstrapped, its median is +0.02 but **48% of
  resamples flip its sign**, 3.6% land beyond |5|, and its mean and sd do not
  converge with resample count. On the same ledger it read +0.12 on v9/v10,
  −2.61 pooled and +4.77 on the wOBA rows — three numbers, one underlying
  non-result, each of which reads as a measurement of a relative weight.

  Fixed by **deleting** the ratio, not by widening its gate. The hypothesis is
  unchanged and is now well posed: `w = 1` is `b_sp = b_lineup` in native
  units, so the report prints the contrast `b_lineup − b_sp·(sd_lu/sd_sp)` with
  the standard error from `c′·cov·c`. That is why `_logit_fit` now returns the
  full covariance rather than its diagonal — the off-diagonal term is part of a
  difference's variance, and discarding it is what left the ratio as the only
  available form. Same data, readable answer: `+0.107 ± 0.245, z = +0.44`, no
  departure from equal weight.

  **The gate came down as a consequence, and that is the reusable part.**
  `N_FIT_MIN` was 120 — sized to hide a statistic that is unreadable at small
  n, not to establish evidence. Coefficients printed with their standard errors
  are honest at *any* size (`+0.122 ± 0.227` says "indistinguishable from zero"
  without needing suppression), so the floor dropped to 30 and now covers only
  logit convergence. When a threshold exists to hide an unreadable number, fix
  the number and the threshold dissolves — the same move as the shrinkage
  weights elsewhere in this file, one level up: the cure for a hard gate is
  usually to remove whatever needed gating.

  Diagnostic only. Nothing in this fit feeds back into a lean, a delta or a
  grade — verified, not assumed: `b`, `se` and `cov` are locals inside
  `report()` and reach only `say()`. No `MODEL_TAG` implication.

- **A deferred defect, shipped at its own gate.** `expected_sp_ip` was measured
  **over-dispersed** on the first backfilled actuals (2026-08-04): slope
  0.756 ± 0.063 over 306 side-games, 3.9 se below 1.0, bias +0.096 IP
  (t = 1.31) — a spread problem, not a level one. It was deliberately NOT
  fixed then, on two grounds: it flipped 1 lean in 80, so the case was
  correctness of a directly-observed input rather than performance, and 306
  side-games of July/August is thin for a slope that is plausibly seasonal.
  The entry set an explicit gate — **re-fit at ~600 side-games** — and had
  `actuals_backfill` print `IP calibration slope` every build so the number
  would arrive without anyone remembering to look.

  It arrived. At n=586 the slope read **+0.735 ± 0.048**, 5.5 se below 1.0,
  and v12 shipped the fix. Keep the whole shape as precedent: a measured
  defect, a stated reason not to act, a numeric gate, a self-reporting
  instrument, and a fix at the gate rather than at the first sighting.

  **What it shipped as matters as much as when.** The deferral warned that a
  fitted literal would be the constants-frozen-from-data entry with a fresher
  date on it, so `sp_ip_calibration()` re-fits from the ledger on every build
  and no `a + b·pred` appears in the source. Two design points came out of the
  other entries here rather than out of this one:

  * The correction is shrunk toward the **identity map** by sample size,
    `cal(p) = w·(a + b·p) + (1−w)·p` with `w = n/(n+K)`, so `n = 0` returns `p`
    exactly. No `if n >= N`, no day on which every workload estimate jumps —
    the threshold-cliff entry applied a third time.
  * `K = 50` was picked by **walk-forward benchmark**, not taste: fit on every
    prior slate, score the next, over 586 side-games and 23 slates. Calibration
    beats no-calibration by +4.1% / +4.0% / +3.8% out-of-sample IP MSE at
    K = 25 / 50 / 100, against +3.4% at K = 0 — so the shrinkage earns its
    place early. Bootstrapped over slates, K=50 is the argmin most often
    (131/400) and every candidate's CI excludes zero. The curve is flat from
    10 to 100 and the comment says so: what is distinguishable is calibrated
    from uncalibrated, not 25 from 50.

  The one genuinely new hazard was **a fit that consumes its own output**.
  From v12 the published `expected_sp_ip` is calibrated, so refitting against
  it would compound the correction every build and pull the estimator toward
  the mean without limit. The dump and ledger therefore carry
  `expected_sp_ip_raw_*`, and the fit reads raw where present, falling back to
  the published column for pre-v12 rows — which are raw by definition, and are
  the entire sample on the first build after the bump. A correction that
  feeds on its own output has no fixed point worth having; store the input.

  Two companion readings from the same backfill are still **not** acted on.
  The realized phase weight (`act_sp_bf / act_pa`) carries the same
  over-dispersion in the units that matter — slope 0.746, bias +0.017, MAE
  0.101 over 210 side-games — which is why the fix targets the workload
  estimate and not the BF/IP conversion. And the rate metric still says
  nothing: calibration slope 0.953 ± 0.380, corr 0.178 ± 0.070 against a 0.196
  ceiling, a CI spanning near-zero to above that ceiling. Do not quote those
  two as findings.

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
  `RECORD_TAGS`, so the headline reset when v9 started a new family" — which
  was wrong about the code when it was written and was verified so on
  2026-08-06: `_record_grades()` then had *no production caller*, every public
  surface rendered `_display_grades()`, and that is exactly why the front page
  could publish `wOBA full 217-164` over 381 xwOBA games (see the metric-label
  instance below).

  **As of 2026-08-17 that sentence is true, and it is true because it was
  made true rather than because it was right.** The record strip and the
  grading-ledger header now score `RECORD_TAGS`, so a `MODEL_TAG` bump *does*
  reset the public headline, and the page and `ledger_report.txt` answer the
  same question over the same rows. Keep the history above: the lesson is not
  "the page scores the current family" — it is that this file asserted so for
  weeks while the code did the opposite, and the way that was caught was
  reading `build_site.py`, not re-reading this paragraph.

  Two consequences to hold onto, both deliberate:

  * **An empty family publishes no record.** The surfaces do not fall back to
    the pooled line — they say "no graded games yet under `<tag>`" and point
    at the ledger. A silent fallback would be the `wOBA full 217-164`
    substitution with the tag rather than the metric label as the lie, and
    v11 is the proof it would fire: it shipped and was superseded without ever
    grading a row. `RecordScopeTests` pins this.
  * **The pooled surfaces stayed pooled, and say so.** Per-club accuracy and
    the market verdict's context bucket still call `_display_grades()`,
    because one family leaves most clubs one or two games — comparability is
    the wrong trade there. The team page's lead states that it pools, so the
    two surfaces cannot read as contradicting each other.

  The 45-37 (.549) / 42-40 / 49-33 line quoted here for 2026-08-02 was the
  *report's* current-family line, not the page's — which at that date were
  different numbers, and now would not be.

  Controls, whatever the row set: the model ahead of the coin-flip control and
  behind the closing line, on a sample far too small to separate them — which
  is the controls doing their job. Do not quote a control figure from this
  file; recompute it. Since 2026-08-06 they are scored on the **decided** rows,
  not every graded one — see the abstention instance below.

- **Twenty-one published cells collapsed to two, and the surface renamed to
  the rule it actually publishes.** The site used to show a per-game 3×5
  delta × price *profile* grid on the leans page and a 2×3 delta × direction
  *discovery matrix* plus three direction totals on the calibration page — 21
  cells over the same few hundred rows, several of them one game. All of them
  are gone, replaced by the hybrid rule's two branches (`hybrid_action`,
  `hybrid_selection`), and `team-grades.html` is deleted outright.

  **The delta axis was the problem, not the cell counts.** Both grids bucketed
  on |Δ|, which the published rule does not read at all and which
  `value_probe`'s joint logit puts at z = −0.13 against price — adding it to
  the close makes out-of-sample prediction *worse*. So 21 cells were cutting
  the ledger on a null axis, and that is precisely the surface the same entry
  warns about: on this data any grid search returns a cell near +20% ROI
  whether or not anything is there, because at 21–82 rows a cell's null sd is
  8–21 percentage points of ROI. Two branches at n=208 and n=15 cannot produce
  that artifact the way fifteen cells at n=1 could.

  **Three things the collapse forced, each worth keeping.** The family-wise bar
  moved 2.7 → 2.2 (`_BRANCH_FAMILYWISE_Z`) because it is a property of how many
  cells are published, not a constant — a bar sized for 15 draws is simply
  wrong for 2, and leaving it would have been a stale constant of exactly the
  kind recorded above. `_lean_market_agg` became column-parameterised so the
  record, the rule's selection and both controls are **one** aggregate over
  **one** row mask. And every control is now derived in
  `_lean_market_observations` beside the record, which is what finally makes
  "controls on the same rows" true by construction rather than by a `n=`
  marker reconciling two derivations — the marker that went blind in the one
  case it existed to catch.

  **The load-bearing display fact, and the reason the control sits inside the
  panel rather than under it: the FADE branch is always-chalk, exactly.**
  Fading a lean priced below .45 backs a side priced above .55, which is the
  favourite on every such game — verified 15 of 15, and `("chalk", "FADE")`
  comes back *equal to* `("branch", "FADE")` on the committed ledger, which a
  test now asserts as a construction rather than observes as a coincidence. A
  reader shown 11-4 / +23.8% without that adjacency reads a chalk result as the
  rule's own skill, in a window where chalk beat its price by +4.0pp.

  **`HYBRID_THRESHOLD` is imported from `hybrid_test`, never restated**, and a
  test greps the source to forbid a second `= 0.45` assignment. Two copies of a
  threshold is the "one value, three homes" defect one file out: the display
  could then drift from the registration and publish a selection the forward
  test would not score. Note the historical coincidence and do not read it as
  evidence — `_CONVICTION_DEEP_OPPOSE` was *already* 0.45, so the hybrid's
  threshold matches a boundary that was chosen for a display band before it was
  chosen for a rule.

  Every branch record these pages publish is a **discovery** figure and each
  surface says so in its own copy (`NOT A FORWARD RESULT` on the card, a lead
  sentence on the calibration panel, a note on the grades page), pointing at
  `data/ledger_report.txt` for the registered out-of-sample version. That is
  the whole reason the rule could be published at all: the site shows what the
  rule selects, and `hybrid_test.py` — not the site — is what will eventually
  say whether it works.

  **`team-grades.html` went with them, and that is a deletion to justify rather
  than assume.** It pooled every graded family because one family leaves most
  clubs one or two games, so it was the last surface on the site answering a
  different question over a different row set from everything beside it — and
  club identity is not an input to the rule. It is NOT a control (the entry
  below is about controls, and always-home and always-chalk both survive and
  are now scored on stricter rows than before), and nothing else read
  `_team_performance_rows`. Recoverable from git history if per-club accuracy
  is ever wanted back.

  **The front-page strip moved with them, and it had to.** It headlined the raw
  lean (139-84, z +2.11) while the grades page one click away headlined the
  rule's selection (146-77, z +2.72) — the artifacts-disagreeing entry below,
  with *both* artifacts public and the reader able to see both. It now reads
  the same aggregate, carries always-chalk beside it, and keeps the metric
  label read off the rows: the existing provenance test caught the label being
  dropped, which is the "wOBA full 217-164" guard doing exactly its job.

  **Two follow-ups, both recorded because they are the shape of the mistake
  rather than one-off copy edits.** First, the per-game panel led its history
  block with `Selection won 73.3%` directly under tonight's two clubs. That is
  the rate at which PAST picks in the same branch won, and this site publishes
  no per-game probability at all — but placed there it reads as one, and the
  line below it (`Market implied 58.6%`) sat two rows under the same game's own
  `37.4% no-vig` with nothing saying the two percentages measure different
  things. The panel is now split into a labelled `This game` zone and a
  `Past V12 FADE picks · 15 completed games` zone, every history row is past
  tense (`Won 11-4 (73.3%)`, `Their average price`, `Beat that price by`), and
  the decision line carries a plain-English reason — on a FADE the selected
  club otherwise appears nowhere else on the panel. **A number is not made
  unambiguous by being correct; it is made unambiguous by what sits next to
  it.** The same commit found adjacency alone had failed for the chalk control:
  on FADE it is equal to the record above it *to the decimal*, so without a
  clause saying why, a reader sees duplicated data or a bug rather than "this
  branch has no model content". It now says so in words.

  Second, the ledger table applied the rule to **every** family. `_row_hybrid`
  is now gated on `RECORD_TAGS`: the rule is registered against the current
  family, and applying it to an older one publishes a selection nobody could
  have made — under lean math the rule was never paired with, and with a grade
  the function then inverts. Earlier rows are marked `lean only`. The three
  reasons the rule cannot act — out of family, no lean, no price — now carry
  three distinct marks, because a lean sitting unlabelled under a "Selection"
  heading is the substitution recorded further down this file. And the header
  now accounts for **every** row of the family (223 decided of 244; 6 with no
  lean, 15 pending), since a record quoted over 223 of 244 without saying where
  the other 21 went invites the reader to assume the rule decided them all —
  the same defect as a control whose denominator is never stated.

  Display-only throughout: no lean, delta, grade or ledger row moves, and
  `MODEL_TAG` is unchanged. What did change is which rows the published record
  is scored on — current family, decided, settled AND priced — because the
  rule needs a price to act, so a record over rows it could not have acted on
  is not this rule's record. Where no row carries a price the surfaces fall
  back to the model's own lean and **say which one they are showing**; the
  controls survive that path too, since always-home needs no price.

- **Threshold cliffs, third instance — a credibility label keyed on a row
  count.** The per-game profile panel graded its own sample
  `THIN / DEVELOPING / LARGER SAMPLE` at `n >= 10` and `n >= 20`. One completed
  game moved a cell from DEVELOPING to LARGER SAMPLE with no change in what the
  cell knew, and "LARGER SAMPLE" described an n=20 cell still carrying ±11pp —
  the label was least accurate exactly where it sounded most confident.

  Fixed the way the two below were: delete the tiers, and let the read move
  with a continuous quantity — here the cell's own standard error. The bar is
  the **family-wise** one (|z| ≥ 2.7 across the 15-cell grid), not 2 sd,
  because a cell's excess is the largest of up to 15 draws and at 2 sd roughly
  one cell clears it every build by chance. Measured on the live grid: 3 of 14
  cells cleared 2.0, **none** cleared 2.7, and `value_probe`'s permutation put
  the best cell at p = 0.20. A test pins that n crossing an old boundary with
  the excess and se held fixed changes nothing.

  The same commit gave the panel the **pooled current-family line** as its
  reference. The cell's median SE is ~15pp; pooled over the same rows it is
  ~3.6pp, so the reader's anchor had been the least measurable number on the
  card with no way to see it was a slice of something better estimated. The
  pooled figure is context and never an edge — `value_probe` measures that
  same statistic flipping sign family to family (v7 −0.68, wOBA v5 −2.74,
  v12 +1.98), which is why it prints its own spread too.

  Display-only: no lean, delta, grade or ledger row moves.

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
- **The same entry, fourth instance, caused by a display change.** The site's
  move to publish the hybrid rule left `ledger_report.txt` headlining the raw
  lean — 139-84 in the report against 146-77 on the page it links to, the same
  games, both called "the model's record". The display PR checked that the
  strip and the grades page agreed with each other and stopped there; the
  internal artifact was not in the diff and so was not in the check.

  **The fix is to print both, not to pick a winner.** The lean line is the
  control the hybrid line is read against, and always-chalk on the identical
  rows is the control they both are; the report now carries all three plus a
  `RETROSPECTIVE` marker pointing at the registered forward block further down
  the same file. The arithmetic moved into `hybrid_test.apply_rule`, a pure
  function with **no date filter**, so the retrospective and forward readings
  cannot drift; the filter stays in `scored_rows` alone, because the whole
  value of the registration is that its row set is decided in exactly one
  place.

  **The general lesson: when a display change alters what a published number
  MEANS, the internal artifact is part of that change even when it is not in
  the diff.** Grep for the other place the statistic is printed before opening
  the PR, not after.

- **Regrading the ledger under a derived rule — asked for, and correctly not
  done.** The natural follow-up to publishing the hybrid is to write it back
  into `data/mlb_lean_ledger.csv`. It is recorded here because the request is
  reasonable and will recur.

  `xw_full` is the LEAN's grade. Overwriting it under the hybrid would mutate
  immutable graded rows; destroy the lean-alone control the hybrid is read
  against, so the published comparison would become the rule against itself;
  make v12's history incomparable with every other family's on a report whose
  per-family lines exist precisely to be comparable; and silently change what
  `bp_ablation` and the SP-vs-lineup weight fit are measuring, since both read
  `xw_full` as a statement about the model's prediction rather than about a
  betting rule layered on it.

  Adding a *new* stored column was also rejected, for a weaker but sufficient
  reason: the hybrid grade is a deterministic function of `xw_lean`, `home`,
  `close_p_home` and `xw_full`, all of which are write-once (`attach_market`
  never revises a close it has already set). Storing it creates a second home
  for a value that can then drift from its derivation — the defect that put
  v10 math under a v9 tag. **The rule is a VIEW over the ledger, derived at
  read time, and a test asserts no `hybrid*` column exists in the artifact or
  in the writer's column lists.**

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

## Probes and standing monitors

Before proposing an improvement, check whether an instrument for it already
exists — several questions in this repo have been asked twice because the
answer was sitting in a probe nobody ran. And before *deferring* one, leave an
instrument behind: the expected-IP fix landed at its stated gate only because
`actuals_backfill` printed its slope every build, and the same deferral without
that line would still be waiting.

**Standing monitors** print on every build and need no one to remember them.
`data/ledger_report.txt` carries the current-family record and F5, the |Δ|
terciles, the per-family and per-slate predicted-vs-actual, the component error
(SP / BP / lineup each against its own realised phase), the IP calibration
slope, the SP-vs-lineup coefficients and their symmetry contrast, and the two
pre-registered forward tests (`forward_test.py`, `hybrid_test.py`). The grades
page carries the baseline controls and the lock provenance. Read these before
writing a new probe.

**Probes run on demand.** Seven read committed artifacts and run anywhere:

| probe | question |
|---|---|
| `value_probe.py` | is there a tradable relationship between `xw_net` and price? (incl. band grids) |
| `forward_test.py` | pre-registered fade rule, frozen 2026-08-29 (also prints every build) |
| `hybrid_test.py` | pre-registered hybrid market-direction rule, frozen 2026-09-01 (also prints every build) |
| `hfa_probe.py` | does adding a home-field term to the lean improve it? (no) |
| `interaction_probe.py` | do single signals or other combiners beat `B·P/L`? |
| `dispersion_probe.py` | does a concentrated lineup beat the mean it is averaged into? |
| `bp_ablation.py` | does removing the bullpen term change any decision? |
| `compare_v8_v9.py` | what the v9 sequential form changed against v8 |
| `shadow_report.py` | what the paired metric arm can and cannot settle |

Seven need a live API and therefore a GitHub runner — `espn_403_probe`,
`matchup_form_probe`, `phase_actuals_probe`, `pitch_arsenal_probe`,
`player_prior_probe`, `pythag_control_probe`, `reliever_shrink_probe`. Each has
a workflow under `.github/workflows/` and each says so in its own header. Savant
and StatsAPI are unreachable from the dev environment, so a probe that needs
them cannot be smoke-tested locally; run the workflow.

### Measured and rejected

- **Home-field in the lean.** The model's lean is `net = home_off - away_off`
  with **no home-field term anywhere** — verified in the source, not recalled;
  the `HFA=+0.177` in `ledger_report.txt` is a diagnostic F5 logit intercept
  that never touches a lean. The market by contrast prices it exactly: over
  766 home closes, 53.0% actual against 53.2% implied, and the model leans home
  on 50.7% of 758 graded rows while home wins 53.2%. So the omission is real
  and the natural proposal is to add the term. **Measured, it makes the lean
  worse.** `hfa_probe.py` is the instrument.

  Fitting `P(home) = σ(a + b·xw_net)` puts the correctly-centred decision
  boundary at `xw_net = -a/b`, so the shift is `h = a/b`. Inside the current
  scale family (n=320): `a = +0.156 ± 0.115` (z = +1.35, home-field present but
  **not** significant), `b = +19.04 ± 4.75` (z = +4.01, the delta itself
  clearly does predict), `h = +0.0082`, flipping 11.6% of leans.

  Walk-forward over 253 rows / 19 slates, refitting `h` on prior slates only:
  current lean 156-97 (+6.4 ± 3.1 vs price, ROI +8.7%) against the corrected
  lean's 153-100 (+5.3 ± 3.1, ROI +6.3%). On the 25 flipped games the paired
  delta is **−0.238u per flip, z = −0.62, CI [−0.98, +0.51]**. A fixed-h
  holdout (fit on the first 160, scored on the last 160) agrees in sign and
  magnitude at −0.223u per flip, so this is not the per-slate refit noise that
  killed `forward_test`'s arm 1. Neither is significant; nothing here says the
  correction hurts, only that there is no evidence it helps.

  **Three things worth keeping.**

  *The scale family is load-bearing and nearly produced a much bigger wrong
  answer.* Fitting across every graded family gives `b = +6.56` and
  `h = +0.0198` — flipping 25% of leans and pushing the home-lean share to
  75.5% against a 53.2% home win rate. That fit pools incompatible `xw_net`
  units, which attenuates `b`; and since `h = a/b`, an attenuated `b`
  **inflates** the correction, here by 2.4x. `_SCALE_FAMILIES` exists for
  exactly this, and a probe that ignored it would have reported a change
  2.4x more consequential than the one that exists.

  *`h` is not a constant.* Across the 19 walk-forward slates it ranged
  −0.0104..+0.0114 (sd 0.0055) and **3 of 19 slates fitted a NEGATIVE h** —
  16% of the time the data says correct toward the away side. A parameter that
  changes sign is not a home-field advantage.

  *Why it fails mechanically, which is the reusable part.* A constant shift
  only changes the decision where `|xw_net| < |h|` — the model's **weakest**
  leans. There it replaces a weak matchup read with "pick home", and home wins
  only ~53%, while the current lean beats always-home by eight points
  (61.7% against 53.8%). **Being blind to a factor is not the same as being
  improved by adding it**: the delta is doing real work, and a constant
  degrades the games where that work is thinnest. The market's home-field
  content is worth having when PRICING a game, and the published hybrid rule
  already consults the price — so it reaches the selection through the market,
  where it is handled once. Adding it to the lean too double-counts it.

  **Two predictions made before measuring were wrong, recorded because the
  errors are the instructive part.** It was predicted that an HFA-aware lean
  would "likely dissolve the fade branch": measured, fades go 15 → 13 of 223.
  `h` is small relative to the price distance needed to cross the 45%
  threshold, so the branch barely moves. And it was implied the correction
  would obviously improve accuracy, on the reasoning that the model was
  missing something real. It was missing something real, and adding it still
  lost. Gate if anyone revisits: ~1,469 flipped games (~1,100 slates) to
  separate a +0.10u/flip effect, so this cannot be settled by waiting.

### Rules these have earned

- **A deferred decision needs a numeric gate and a self-reporting instrument.**
  Not one or the other. `expected_sp_ip` is the worked example: measured
  over-dispersed, deliberately not fixed, gate set at ~600 side-games, slope
  printed every build, fixed at the gate.
- **Fix the test before the data exists.** `dispersion_probe` was written while
  the column it reads had zero graded rows, so its cuts could not be chosen to
  suit an outcome. A test written after seeing the data is a different and
  weaker kind of evidence; if you add one later, say so in the output.
- **Print the standard error; never suppress the number.** A statistic that is
  unreadable at small n is a statistic to fix, not to gate — see the ratio
  instance in the anti-patterns above. A hidden number invites someone to
  recompute it without the caveat.
- **Verify an unproven external input on a shadow arm before the critical
  path.** The `xwoba` selection name reached the primary build only after the
  arm resolved it against the live endpoint, where being wrong cost a log line
  instead of a slate.
- **A probe that hardcodes a model constant goes stale and starts benchmarking
  the model against an old copy of itself.** `interaction_probe` froze the IP
  calibration slope at `0.756` and, once v12 shipped a per-build fit, would have
  double-applied it. Read live values off the module.

  **Second instance in the same file, through the row selector rather than a
  number.** Its `__main__` listed two hardcoded blocks — `tags=(v9, v10)`, and
  `metric="wOBA"` labelled **"(live)"**. The revert made both wrong at once:
  the "live" block scored a lineage the build had stopped running at v11, and
  v12 grew to 217 graded rows — the largest family in the ledger, and the one
  actually shipping — without the probe ever scoring it. Nothing crashed and no
  line was false on its face; the probe simply answered about a model that no
  longer ran, which is why it survived two bumps. **A constant is not only a
  number: the set of rows a probe reads is one too.** The current block now
  derives from `build_site.RECORD_TAGS`, so a bump carries the probe forward
  with no edit; the historical blocks stay pinned, because a frozen question
  needs a frozen row set, and are labelled as history. Pinned by four tests
  that assert the *rule* — the current family appears, only it is labelled
  CURRENT, a historical block equal to it prints once, and no family tag
  outside the historical list appears in the file — because pinning
  `xw+plat_consol_v12` would reproduce the defect one file out.

  The same commit fixed a `load()` docstring reading "no v12 row has graded
  yet", which stayed there through 217 of them: the version-note-asserts-rows
  failure recorded three times above, with the sign reversed. It now states
  the rule (a column is absent on families older than the tag that introduced
  it) rather than a count.

  What the block was hiding is worth reading, not just the fix. On v12 the
  shipped rule scores corr +0.167 against the realised wOBA differential, and
  `signal: lineup only` scores **+0.017** with a paired d_corr CI of
  [-0.287, -0.022]; the in-sample lineup weight is **-0.500 ± 0.975** beside
  starter +0.562 and bullpen +1.104. That agrees with the component-error
  monitor, where the lineup phase has run a negative slope against its own
  realised rate for weeks (-0.86 → -0.78 → -0.63 as n grew to 434, corr
  -0.050). Read it as "not measurably contributing", not as "inverted" — every
  one of those intervals contains zero. It is un-acted-on and instrumented,
  not a defect: `bp_ablation` covers the bullpen term and nothing covers this
  one. The `q from calibrated IP` candidate also now reads d_corr
  [-0.001, +0.000] on v12 rows, confirming on real rows what
  `test_the_calibration_is_not_applied_twice_to_a_v12_row` pins on constructed
  ones — after the bump that candidate IS the baseline.
- **Say what a probe cannot answer.** `player_prior_probe` cannot count lean
  flips; `shadow_report` cannot settle the metric from unpaired eras; the
  dump-versus-ledger join is contaminated by the post-rollover rebuild. Each
  says so in its own header, and those sentences are load-bearing.

### Instrumented and waiting

Do not re-derive these by hand; they have readouts.

- **Lineup dispersion** — the headline slope prints in `ledger_report.txt`
  every build once rows exist, marked UNDER-POWERED until ~347 side-games
  (roughly 12 slates from 2026-08-15); `dispersion_probe.py` is the full read
  and controls for the backfill count, the zero-backfill subset and the
  game-level margin. A test pins the two together so they cannot drift into
  disagreeing. The column ships on new rows only, and no row graded before it
  existed can ever be backfilled — so the entry once read "zero rows today".

  **The gate has since been crossed and the marker has dropped on its own,
  which is the instrument working.** As of 2026-08-29 the report prints the
  slope unmarked at n=367 side-games, and the first powered read is null:
  `residual slope -0.29 +/- 0.65`, nowhere near separating from zero. Read the
  current pair off `ledger_report.txt` rather than from here — the count moves
  every build, and this is the one entry whose own threshold text has already
  gone stale once.
- **The v11/v12 scale-family share** — **settled 2026-08-29: the share
  stands and the falsifier is closed.** Argued rather than measured at the
  bump, because no-lookahead forbids rebuilding a past slate. Falsifier named in
  `_SCALE_FAMILIES`: compare median `|xw_net|` on the first graded v11/v12 rows
  against the v9/v10 pool and split the family if it moved. **First read, 15
  graded v12 rows (v11 produced none — see above): median `|xw_net|` 0.01845
  against the v9/v10 pool's 0.01853, difference −0.00009 with a bootstrap CI of
  [−0.0081, +0.0183].** No sign of a shift, and read the CI before the point
  estimate: its half-width is ±0.013 against a median of 0.018, so this can only
  rule out a shift of roughly 70% or more. It is a check that the family is not
  grossly wrong, not a confirmation that it is right. Re-read it at ~60 v12 rows
  before treating the share as settled.

  **Second read, 2026-08-17, 38 v12 rows (30 graded): median `|xw_net|` 0.01334
  against the v9/v10 pool's 0.01859 over 99, difference −0.00526 with a
  bootstrap CI of [−0.0117, +0.0002]** (graded-only: 0.01430 against 0.01853,
  −0.00423, CI [−0.0116, +0.0007] — take the all-rows figure, `|xw_net|` is a
  pregame quantity and gradedness is not a property of the scale). The point
  estimate moved from −0.00009 to −0.0053 in one read and the CI now all but
  excludes zero, which reads as the falsifier firing.

  **Do not split the family on it. The falsifier as written cannot tell a units
  change from a slate-composition change, and here the mechanism rules out the
  units change.** v12's only difference from v10 is the IP calibration, measured
  in `_SCALE_FAMILIES` at mean `|Δ net|` 0.00067 with a max of 0.00542 — the
  observed gap is 8x the mean effect and larger than the largest single-row move
  the change can produce. It cannot have caused this. What can: v12 holds 3
  slates, and per-slate median `|xw_net|` *within* a single family moves about
  this much on its own — sd 0.0027 across v10's 5 slates, 0.0039 across wOBA
  v5's 9, range 0.0103–0.0184 across v12's own 3. The gap is roughly one to two
  slate-sd on a three-slate sample.

  So the ~60-row re-read stands, and it needs a better statistic than the one
  named above: pool the slate medians rather than the rows, or size the gap
  against the 0.00067 mechanism bound instead of against zero. A row-pooled
  median over a handful of slates is measuring which games got played. Same
  category as the constants entry — a number read off one distribution and
  quoted against another — one level out into a test rather than a constant.

  **Third read, 2026-08-29, at the gate: 196 v12 rows over 15 slates. The
  falsifier does not fire, and the second read was slate composition exactly as
  called above.** Row-pooled — the form the first two reads used, kept for
  comparability — median `|xw_net|` 0.01774 against the v9/v10 pool's 0.01859
  over 99, difference **−0.00085, CI [−0.0062, +0.0023]**. The point estimate
  regressed from −0.0053 to −0.0009 on 5x the sample. The slate-pooled form
  asked for above agrees: median of per-slate medians, 15 slates against 7, gap
  **−0.00309 at 1.48 se, CI [−0.0063, +0.0021]**. Neither excludes zero and
  neither excludes the 0.00067 mechanism bound.

  **Do not re-read it again — it is not merely underpowered today, it is
  underpowered by construction.** Between-slate sd of the per-slate median is
  0.00457, **7x** the mean `|Δ net|` of 0.00067 the IP calibration can actually
  produce, so telling the mechanism apart from slate noise needs ~730 slates per
  arm — about 68 seasons. What the test does have power for is a gross units
  shift: ~11 slates per arm at 0.00542, the largest single-row move the change
  can make, and both arms clear that. It has now said the only thing it was ever
  capable of saying, which is what closes it rather than what leaves it open.

  **The reusable half is the shape of the mistake.** A two-family comparison
  routes the question through the games that happened to get played, and the
  noise that introduces swamped the effect by 7x — so the instrument could
  produce a scary-looking read (the second one) without ever being able to
  produce a decisive one. The direct paired measurement already in the v12
  `_SCALE_FAMILIES` entry — rebuild the same 254 rows both ways, mean `|Δ net|`
  0.00067 — answers it with no ledger accumulation and no slate-composition
  term at all. **Prefer measuring a change against itself over measuring two
  families against each other.** Where a falsifier needs an ever-growing sample
  to say anything, check its power against the mechanism before banking on it.
- **The metric question** — the shadow arm, running wOBA under an xwOBA
  primary. Needs roughly 18 paired slates for 80% power on a 0.09 gap.
- **`LEAN_STRENGTH_FALLBACK`** — recompute from whatever `SCALE_TAGS` resolves
  to rather than quoting any number in this file.

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
- `commit_data.py`'s fallback to `git push`. `data/` lands through the GraphQL
  `createCommitOnBranch` mutation so GitHub signs the commit and it shows
  Verified — a runner holds no key, so a pushed commit never can. The API path
  is the newer one, and a pregame slate that fails to land cannot be
  re-derived without lookahead, so any failure falls back to the old
  commit/rebase/push sequence with a workflow warning: an unverified commit
  costs provenance, a dropped slate costs rows. Keep the fallback, and keep
  `expectedHeadOid` refusing rather than overwriting when an intervening
  commit wrote one of the files being sent — that refusal is the old
  `git pull --rebase` conflict stop, which the API path would otherwise lose.
  Commits made with `GITHUB_TOKEN` do not retrigger workflows whether pushed
  or created through the API, so the no-recursion property is unchanged.
