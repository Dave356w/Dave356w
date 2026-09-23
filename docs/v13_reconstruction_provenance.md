# V12 → V13 historical reconstruction: provenance audit

**Snapshot:** September 22, 2026. This is an audit of the committed
`data/mlb_lean_ledger.csv`, `data/shadow_<date>_woba.csv` and the existing
`reconstruct_v13.py` logic, not a new selection rule or model adjustment.

## Comparable prediction family

V13 changes the starter component from xwOBA alone to the existing xwOBA/wOBA
blend, keeping the underlying matchup machinery. V12 and V13 are therefore
closely related prediction specifications; the retained record family contains
both. Their selection overlap is directly measurable, and the original
pregame V12 decisions remain immutable.

Of **445 graded V12 games** with a usable V13 reconstruction:

- **414 / 445 (93.0%)** retain the identical selected team; **31 (7.0%)**
  change selected team.
- **27 of 31** switches have original V12 `|xw_net| < .005`; another **3**
  are in `[.005, .010)`; **1** is in `[.010, .015)`. There are **no**
  changes at original `|xw_net| >= .015`.
- On the identical 445 final outcomes, V12's originally locked selections were
  **273–172**; their V13 reconstructions are **278–167**. This is a five-win
  difference in an otherwise nearly identical set of selected teams.
- The separate *native*, first-run V13 sample at this snapshot has **48**
  completed decisions, **35–13**.

Those measurements support comparing the two specifications as closely
related. They do **not** establish that every reconstructed selection was an
available pregame V13 decision.

## Which snapshots are actually pregame?

The data answer two different timestamp questions:

1. **Original V12 decision:** all **445** have a ledger `snapshot_utc` strictly
   earlier than `scheduled_start_utc`. Their original pregame picks are real,
   locked decisions.
2. **Paired wOBA shadow used to reconstruct V13:** matching each selected
   shadow file's `snapshot_utc` to that game's
   `game_datetime_utc` yields **89 / 445** before scheduled first pitch and
   **356 / 445** after it. This count covers the 34 dated primary/shadow pairs
   from 2026-08-15 through 2026-09-17 and matches all 445 graded reconstructed
   ledger rows by date and game ID. It excludes the eight original V12
   abstentions from this scored sample.

The two arms of a same-run shadow pair are normally written seconds apart.
However, that does not make a shadow written *after* a game's first pitch the
same frozen input snapshot as the original, locked V12 decision.
`reconstruct_v13.py` already describes this as mixed-basis reconstruction.

A file timestamp after first pitch does **not alone** demonstrate leakage:
the underlying Statcast leaderboard might still contain only pregame data.
To elevate a late file to verified pregame status requires an *as-of* field or
another reproducible proof of the precise data cutoff of its wOBA rate, not an
assumption from paired dump proximity. Do not mark all 445 reconstructions
pregame-locked without that evidence.

## Reporting convention

The per-game card retains the **comparable V12→V13 family**
retrospective as the primary historical figure, rather than promoting the
shorter native V13 sample above it. Its compact provenance note states the
number of paired-snapshot V12 rows re-scored under the V13 starter blend and
the number of **native pregame V13** decisions. The native record and its
closing-price margin appear as secondary context. The current game's model
lean, no-vig market probability and posted break-even remain distinct.

The overlap measurement above supports pooling these closely related model
specifications for **descriptive family analysis**. Pooling is not proof that
all reconstructed inputs were captured pregame: that narrower claim still
requires the data-vintage audit described above. Neither historical figure is
a calibrated win probability or expected edge at today's quote.

Historical margin over break-even on these cards uses **saved historical
closing prices**, not each game's locked pregame quote and not today's posted
price. A positive pooled historical margin is not an expected return for a
particular game. The complete exploratory delta × price grid and its
multiple-search reference remain in `data/ledger_report.txt`.

No lean formula, market selection rule, ledger source field, native grade,
reconstruction, historical threshold, or tracked registration has changed
in this reporting-only pass.

## How to verify

Run `python reconstruct_v13.py --dry-run` to regenerate the reconstruction
and snapshot split without changing the ledger. Inspect the matched dated
shadow CSVs for row-level timestamps. The card source is
`build_site.hybrid_branch_records` and `build_site._xwoba_side_history`.
The reporting guard is `tests/test_v13_card_provenance.py`.
