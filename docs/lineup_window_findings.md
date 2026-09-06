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

**Box-score aggregates only. No plate-appearance rows and no batting-order
index exist anywhere in the repository.** Task 3 does not run.

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

The bullpen phase is not stored at all — `actuals_backfill.phase_lines`
reconstructs it as the team line minus the starter line, validating that no
field goes negative before it will call the residual a bullpen line.

**Consequence for the hypothesis.** `act_sp_bf_<side>` is a per-start *count*.
That is enough to CONDITION on window length, which is Task 2, and not enough
to REDEFINE it. A fixed 18-batter rescore would need to know which batters
those were; nothing committed says. It is not computable, and no substitute
window was run.

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

## Verdict

**The lineup slope is not an artifact of the scoring window.** Conditioning on
starter batters faced moves it by 0.001 against a registered falsification
threshold of 0.1, because predicted lineup quality and starter window length
are essentially uncorrelated (-0.045) in the 598 graded side-games. Whatever
explains the sign disagreement between the component slope (-0.83 ± 0.51) and
the weight fit (b_lineup = +0.158 ± 0.130), the SP/BP boundary is not it.

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

What would settle it is play-by-play ingestion (`atBatIndex` ordering per
game), which nothing in this repo fetches today. That is a data-acquisition
change with its own no-lookahead question — a completed game's play-by-play is
an immutable historical fact like its box score, so it is backfill rather than
lookahead — and it is not proposed here.
