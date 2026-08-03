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
**wOBA v1**.

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
| split v1 | wOBA lineup, **xwOBA arms**; two league anchors | split v1 | split v1 |

The wOBA forward test was intentionally isolated in both namespaces. Observed
wOBA changes the predictions and its sampling distribution is not the xwOBA
delta scale. Internal `xwOBA`/`xw_*` dump and ledger keys remain a compatibility
schema for immutable history; new rows must carry `model_metric`.

**split v1 is the measured answer to which metric belongs on which side**, and
wOBA-everywhere was half wrong. On matched 2025→2026 Savant seasons the
`wOBA − xwOBA` residual repeats for batters (r=+0.239, p=0.022, n=91) and does
not for pitchers (r=−0.117, n=41) — DIPS, in one number. Predicting next
season's wOBA, the blend `(1−w)·xwOBA + w·wOBA` is monotone in *opposite*
directions: batters peak at w=1 (r +0.288→+0.338), pitchers at w=0
(r +0.438→+0.182). wOBA v1 used w=1 on both, picking the worst available point
on the pitching side; xwOBA beats wOBA there in 99.9% of bootstrap resamples
(gap +0.256, 95% CI [+0.084, +0.442]).

Two consequences worth remembering. First, **the two sides now need two league
anchors** — `M = L_out·(B/L_bat)·(P/L_pit)`, which reduces exactly to `B·P/L`
when the metrics agree, so it is a generalisation and not a branch. Every edge
is still taken against the offensive anchor, which is why the one-baseline
phase invariant still holds. Second, **the split behaves like v10, not like
wOBA v1**: spliced from the one slate built under both metrics it flips 0 of 5
leans against v9/v10 and 1 of 5 against wOBA v1, with median |net| 0.0291 vs
0.0306 and 0.0178. That is n=5 — recorded as an observation, not as grounds to
have shared v9/v10's families. The lineup metric genuinely changed and has a
measured basis, so both namespaces reset. Argue it again when there are rows.

**Known cost of the split: it breaks a park symmetry both pure lineages had.**
Tonight's park always cancels — it is common to both offenses and the lean is a
difference of edges, so it comes out as a scalar and a game-level park factor
would buy nothing. But the park bias inside the *season lines* only cancels
when both sides carry the same dose. The matchup crosses teams (home offense =
`B_home·P_awaySP`, away offense = `B_away·P_homeSP`) and park exposure is a
team property, so under wOBA-everywhere each product carried one half-dose of
each park and they cancelled; under xwOBA-everywhere neither carried any.
Under the split the lineups carry park and the arms do not, the two products
carry `f_H` and `f_A`, and the model tilts toward whichever club has the more
hitter-friendly home park. Size unmeasured — it needs per-team park factors,
which this repo has none of.

This is **not** grounds to put the arms back on wOBA: that swaps first-order,
measured accuracy (xwOBA predicts next-season pitcher wOBA at r=+0.438 against
wOBA's +0.182) for a second-order park term. If measurement ever justifies a
fix it is to park-neutralise the *hitter* inputs, restoring the symmetry while
keeping the pitching accuracy. Neutralising one side only is what created this;
do not repeat it in reverse.

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
  for the v9/v10 xwOBA family, and **wOBA v1 and then split v1 have each made it
  stale again** — a new `_SCALE_FAMILIES` entry is exactly the invalidation its
  own comment names, and that has now happened twice in a week. It is
  deliberately not re-derived: the split pool starts at n=0, and freezing a
  fresh literal off a handful of rows would be this anti-pattern with a fresher
  date on it. Shrinkage plus the slate top-up hold the line meanwhile.
  Recompute from the split family alone once it passes ~60 rows. If a constant
  was read off the
  ledger, comment where it came from and what would invalidate it. Note the
  asymmetry the comment there spells out: a *scale-family* change invalidates
  it, but mere pool growth does not — it is a prior, and re-deriving it from the
  family it is shrunk against would make it the data.

  `HEAT_DOMAINS` is the same shape one level out and is now on notice too: its
  saturation ranges were calibrated on the xwOBA spread, and the one slate built
  under both metrics shows the starter-allowed rate widening (sd 0.0161 →
  0.0215) while the lineup composite does not move (0.0089 both ways) — a
  starter rate is one player's observed outcome, a lineup is nine shrunk ones
  averaged. Recorded in a comment rather than patched: n=14 is one slate, not a
  distribution. Display-only, so no `MODEL_TAG` implication either way.

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

  Read those three numbers as of that commit, not as standing facts — the page
  scores `RECORD_TAGS`, so the headline reset when v9 started a new family. On
  2026-08-02 the current family reads **45-37 (.549)** over 82 graded games,
  against **42-40 (.512)** always-home and **49-33 (.598)** always-chalk. The
  model is ahead of the coin-flip control and behind the closing line on a
  sample far too small to separate them — which is the controls doing their
  job. Do not quote a control figure from this file; recompute it.

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
  whole — currently "168 of 317 rows carry a pregame lock timestamp; 149 legacy
  rows predate that instrumentation." Report provenance, don't assert coverage.
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

## Before opening a PR

```
python validate_data_files.py     # CSV conflict markers — has failed twice in prod
python -m pytest tests/ -q        # CI does NOT run these yet
```

CI has no test step and `requirements.txt` has no pytest. Until that is fixed,
running the suite locally is the only gate. If you touch `build_site.py`,
`grade_leans.py`, or `market_backfill.py`, run both commands and paste the
output in the PR.

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
