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
- `SCALE_TAGS` — do these rows measure `xw_net` on the same scale? Units
  compatibility. Governs `lean_strength_scale()` only.
  - _Status: not yet implemented in the source. `SCALE_TAGS` appears nowhere in
    the tree; `lean_strength_scale()` (`build_site.py`) currently reuses
    `RECORD_TAGS`. So scale-ranking is **not** protected against the v4→v5
    shrinkage halving described below — when `RECORD_TAGS` narrows to a single
    new family with fewer than 60 graded rows, the scale falls back to all
    graded rows and mixes the pre/post-v5 units. Treat the bullet above as the
    intended split, not current behaviour._

These are different equivalence relations and can disagree. v6 is a new
prediction family (bullpen blend) but inherits v5's empirical-Bayes shrinkage,
so it shares v5's delta units. Pre-v5 tags do not: shrinkage halved the scale
(median `|xw_net|` .036 → .018).

When you bump `MODEL_TAG`, decide both questions explicitly in the PR body.
Silence defaults to a new record family and inherited units — which is wrong
about half the time.

Display-only changes do not bump `MODEL_TAG`. Card layout, copy, CSS, legend
text: no bump. Anything that moves a lean, a delta, or a grade: bump.

## Anti-patterns with instances in this repo

- **Threshold cliffs.** `use = fam if len(fam) >= 60 else pooled` switches
  discontinuously — every label in the build flipped the day the counter
  crossed. If a selector has a hard `>= N`, ask what happens on the build where
  it trips. Degrade continuously or don't degrade.
- **Constants frozen from data.** `LEAN_STRENGTH_FALLBACK` was a literal copy of
  the pooled p33/p80 at the time it was written, and stayed there through two
  model versions that changed the distribution underneath it. If a constant was
  read off the ledger, comment where it came from and what would invalidate it.
- **Deleting controls as clutter.** The walk-forward Pythagorean control arm was
  added, then removed in a UI declutter three commits later. Controls establish
  whether the model beats a trivial baseline. If a control is visually noisy,
  move it to the ledger as a column — do not delete it.
- **Public claims the data can't support.** `grades.html` asserts all rows
  locked before first pitch; `lock_status` is null on 149 of 244 rows. Before
  adding a methodology claim to a rendered page, confirm every row it covers
  carries the evidence.
- **Internal and public artifacts disagreeing.** `data/ledger_report.txt`
  reports per-family records and says the current family has no graded games.
  The site publishes a single pooled record. When you change one, change the
  other or state in the PR why they should differ.

## No-lookahead

`.savant_cache/` is gitignored and keyed by slate date. Leaderboard state as of
a past game is not recoverable, so historical rows cannot be re-derived from
today's Savant pull — that is lookahead, not backfill. Pending rows never
receive closing lines (`run_market_update.py` invariant). Do not relax either.

## Before opening a PR

```
python validate_data_files.py     # CSV conflict markers — has failed twice in prod
python -m pytest tests/ -q        # 77 tests; CI does NOT run these yet
```

CI has no test step and `requirements.txt` has no pytest. Until that is fixed,
running the suite locally is the only gate. If you touch `build_site.py`,
`grade_leans.py`, or `market_backfill.py`, run both commands and paste the
output in the PR.

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
