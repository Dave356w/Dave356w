# Is the lineup slope an artifact of the scoring window?

**Verdict: no. The artifact story is dead.** Conditioning the lineup component
on starter batters faced moves its slope from **-0.828 to -0.829** — a move of
-0.001 against a registered falsification threshold of ~0.1. The window is not
what makes the slope negative.

Registration: `docs/lineup_window_registration.md`, written before any output
here was computed. Instrument: `lineup_window_probe.py`, pinned by
`tests/test_lineup_window_probe.py`. Diagnostic only — no lean, delta, grade or
ledger row moves, and `MODEL_TAG` is unchanged.

## Task 1 — data granularity

**Committed artifacts hold box-score aggregates only. Task 3 does not run.**

Stated precisely, because the first version of this section was too broad. A
batting-order index **does** exist — at build time, in memory. `hitter_rows`
assigns `batting_order` as `enumerate(lu, start=1)` per lineup slot
(`build_site.py:1811`), and `lineup_weight` / `slot_pa_weights` use it for the
v4 slot-PA weighting (`build_site.py:2736`). It is aggregated away before
anything is written: no committed artifact carries it.

What was checked, not recalled:

* `actuals_backfill.parse_boxscore` (`actuals_backfill.py:184`) reads exactly
  two things per side from `https://statsapi.mlb.com/api/v1/game/{pk}/boxscore`
  — the team-level `teams.<side>.teamStats.batting` totals, and one pitcher
  object selected by its own `gamesStarted` flag. Both are season-to-date-style
  aggregates for the game, not event rows.
* The persisted schema, read off `data/mlb_lean_ledger.csv`'s header: columns
  129-173 are `act_pa_*` … `act_sf_*` (team batting totals), `act_woba_*`,
  `act_sp_ip_*` / `act_sp_bf_*`, the starter's allowed line `act_sp_ab_*` …
  `act_sp_sf_*`, and `act_schema`. Every one is a per-game count.
* No play-by-play endpoint is used anywhere: `grep` for
  `playByPlay|atBatIndex|battingOrder|plateAppearance` across the repo returns
  only the two team-aggregate field names (`build_site.py:892`,
  `actuals_backfill.py:191`, `reliever_shrink_probe.py:267`) and a test fixture.
* `data/lineup_resolution_audit.csv` is 15 rows of lineup-posting counts, not
  batter events.
* The per-slate dumps are one row per side and carry the lineup only as
  aggregates: `opp_xwOBA_neutral` (the slot-PA-weighted composite),
  `opp_xwOBA_sd` (its dispersion) and `n_opp` (the hitter count). Across every
  committed `leans_*`, `shadow_*` and `rebuild_*` dump the only lineup-ish
  column names are `lineup_status_*`, `lineup_posted_*` and
  `lineup_savant_backfill_*` — all per-side counts and statuses. No slot, no
  player id, no per-hitter rate.
* Per-hitter rows exist only in `.savant_cache/`, which is gitignored and
  slate-keyed, so they are not recoverable for a past slate by design.

The bullpen phase is not stored at all — `actuals_backfill.phase_lines`
reconstructs it as the team line minus the starter line, validating that no
field goes negative before it will call the residual a bullpen line.

**Consequence for the hypothesis.** `act_sp_bf_<side>` is a per-start *count*.
That is enough to CONDITION on window length, which is Task 2, and not enough
to REDEFINE it. A fixed 18-batter rescore is not computable, and no substitute
window was run.

Both halves of that rescore fail, and they fail for different reasons — worth
separating, because the order question only touches the easier half:

* **Predicted side — blocked by aggregation, and the window is a special
  case.** 18 batters faced is exactly two times through the order, so every
  slot appears exactly twice and the slot-PA weights become uniform: the
  fixed-window prediction is just the *unweighted* nine-hitter mean. That is a
  one-line change to a build that still holds the per-hitter frame. It cannot
  be reconstructed from committed artifacts, though — the dumps persist the
  weighted composite and its sd, and the gap between weighted and unweighted
  depends on the covariance between slot weight and hitter rate, which
  `(mean, sd)` does not determine.
* **Actual side — blocked by the data source, and this is the binding one.**
  Scoring what the first 18 batters actually did needs per-batter outcomes in
  sequence. The box score gives whole-game team totals and a starter-allowed
  total; neither is a prefix of the game. No batting order, persisted or not,
  supplies this — only play-by-play does.

So persisting `batting_order` would make the fixed-window *prediction*
available going forward and would still leave Task 3 uncomputable. The
conclusion is unchanged; the reason is narrower than "no order index exists".

## Task 2 — results, all 598 side-games, no subsampling

The probe reproduces the report's component block exactly before conditioning
anything (lineup -0.828/-0.83, corr -0.0664, MAE 0.0733; SP +0.606/+0.61,
corr +0.1136, MAE 0.0940), and a test asserts that against
`actuals_backfill.paired_components` rather than against a frozen literal.

| | lineup | SP (control) |
|---|---|---|
| unconditioned slope | -0.828 ± 0.510 | +0.606 ± 0.217 |
| **R1** residual ~ BF | **+0.000113 ± 0.000764** (z=+0.15) | -0.001203 ± 0.001022 (z=-1.18) |
| **R2** slope with BF as covariate | **-0.829 ± 0.511** | +0.610 ± 0.217 |
| R2 move | **-0.000** | +0.004 |

Starter BF is present on 598 of 598 side-games, mean 21.64, sd 4.93, range
1-34 (p05 9, p95 27). The covariate is not degenerate — there is real window
variation to condition on, and the null is not a restriction-of-range artifact.

**R1 — registered direction POSITIVE. Nominally satisfied, evidentially
empty.** The coefficient is +0.000113 ± 0.000764, z = +0.15. Over the full
observed BF range of 33 batters that coefficient predicts a residual swing of
0.0037 wOBA, against a lineup MAE of 0.0733. The sign matches the
registration; nothing else about it does.

**R2 — registered direction UPWARD. Fails, and fails at the falsifier.** The
slope moves from -0.828 to -0.829 — *downward*, by 0.001, against a threshold
of ~0.1. This is 100× short of the registered bar.

**Why it cannot move, which is the part worth keeping.** An OLS coefficient on
`pred` shifts when a covariate is added only to the extent the covariate
correlates with `pred`. Here `corr(pred, starter BF) = -0.0449`. **The
mechanism fails at its first link**: predicted lineup quality does not predict
how long the opposing starter lasts. The registered story — strong lineups
chase starters early — is not present in this data at a magnitude that could
move anything downstream, so no conditioning on the boundary could have
rescued the slope regardless of what the residual did.

**Task 2c — the SP control, which cuts the same way.** SP's window is
endogenous in the same manner and its slope also does not move (+0.606 →
+0.610, +0.004). Under the registration this was the discriminating case: had
lineup moved while SP stayed put, that would have weakened the mechanism.
Neither moved, so the control adds no support and no contradiction — it
confirms that BF carries essentially no information about either component's
prediction error, rather than isolating a lineup-specific effect. Note the SP
residual-on-BF coefficient is *negative* (z = -1.18), the opposite of R1's
registered direction, on the component whose window story is strongest.

## Task 3 — the fixed window, run over the whole v12 family

Task 1 said the fixed-window rescore was uncomputable from committed
artifacts. It is computable from StatsAPI play-by-play, which is backfill in
the same sense a box score is, so `lineup_window_collect.py` was built and run
over the entire family: **299 games, 22 slates, 22,760 plate appearances,
594 of 598 side-games usable.**

Fidelity first, because the slopes are worthless without it:

* **299 of 299 games reconcile** against `parse_boxscore` on the same payload,
  with **no unmapped `eventType` values** across 22,758 plate appearances.
* Reaching that took one real fix. The first family run reconciled 297 of 299,
  the two failures sharing a signature — the play-by-play counting exactly one
  more PA, and one more AB, than the box score. Diagnosed from the data rather
  than guessed: a map failure prints the game's event histogram, and
  `other_out` occurred exactly once on each of the two failing sides and on
  neither passing side of the same games. It is a baserunning out that ends an
  inning with the batter's PA incomplete, so it moved to `_NOT_A_PA`. Two
  independent confirmations followed: the reconciliation went to 299/299, and
  the PA count fell by exactly 2.
* **13 games carry a ledger row older than a scoring revision.** Across 598
  side-games the whole-game actual differs from the stored one on 12, by at
  most 0.027 wOBA. Not a defect at either end: `_fill` is write-once by design.
* **598 of 598** side-games have a full 18-batter window; distinct batters
  inside it run min 9, median 9, max 11.

| | n | slope | corr | MAE |
|---|---|---|---|---|
| report's own number (ledger actual) | 598 | -0.828 ± 0.510 | -0.0664 | 0.0733 |
| whole game, from play-by-play | 598 | **-0.811 ± 0.511** | -0.0648 | 0.0734 |
| **first 18 batters faced** | 598 | **-0.845 ± 0.671** | -0.0515 | 0.0963 |

Every row of the measurement set, with nothing dropped. The first two lines
differ by 0.017 on identical rows, accounted for by the 12 revised actuals —
the reconstruction reproduces the component block, which is what licenses
reading the third line at all.

**The registered fixed window does not move the slope: -0.811 → -0.845 on
identical rows, a move of -0.034 against the same ~0.1 band R2 was judged on,
and in the direction OPPOSITE to the hypothesis.** The artifact story predicts
the slope should rise toward zero once the endogenous boundary is removed. It
fell slightly.

**The pre-registered caveat cannot rescue it, and this is the part worth
keeping.** The predictor mismatch — a slot-PA weighted composite scored against
two unweighted turns — was registered in advance as attenuating the
fixed-window slope *toward zero*. That is the direction that would have
flattered the hypothesis, and the slope moved away from zero instead. So the
one bias big enough to worry about works against the finding rather than
producing it. What the fixed window does buy is noise: 18 PAs is a thinner
actual than ~74, so the SE grows 0.511 → 0.671 and MAE 0.0734 → 0.0963.

Two independent tests now agree. Conditioning on the window (R2) moved the
slope by 0.001; replacing the window outright moved it by -0.034. Neither
clears the registered bar, and neither moves in the hypothesised direction.

## Verdict

**The lineup slope is not an artifact of the scoring window.** Conditioning on
starter batters faced moves it by 0.001, and rescoring over a fixed 18-batter
window moves it by -0.034 — both inside the registered ~0.1 band, and the
second in the direction opposite to the hypothesis. The mechanism fails at its
first link: predicted lineup quality and starter window length are essentially
uncorrelated (-0.045) across the 598 graded side-games. Whatever explains the
sign disagreement between the component slope (-0.83 ± 0.51) and the weight fit
(b_lineup = +0.158 ± 0.130), the SP/BP boundary is not it.

Two things this does **not** establish, stated because the temptation to
over-read a clean null runs the other way here. It does not show the lineup
term is worthless: -0.83 ± 0.51 and +0.158 ± 0.130 both contain zero, and a
term contributing noise is what both measurements are consistent with — which
is the reading `interaction_probe`'s v12 block already reports (`signal:
lineup only` scores corr +0.017, in-sample weight -0.500 ± 0.975, every
interval containing zero). And it does not close the window question in
general: a fixed-window rescore is a strictly different measurement that this
data cannot produce, and the conditioning above is descriptive rather than
causal — starter BF is a post-treatment outcome of the same game, so
`act ~ pred + BF` answers "does the slope depend on the window", which is what
was registered, and not "what is the window-free lineup effect".

That is now done rather than proposed: `lineup_window_collect.py` ingests the
play-by-play and the fixed window is reported above. What remains genuinely
out of reach is the PREDICTED half — the unweighted nine-hitter mean for a past
slate needs that slate's Savant leaderboard, which `.savant_cache/` is
gitignored to forbid. A build that persisted the per-hitter frame would make
the matched-prediction version available going forward. That is a data-acquisition
change with its own no-lookahead question — a completed game's play-by-play is
an immutable historical fact like its box score, so it is backfill rather than
lookahead — and it is not proposed here.
