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

## Reporting convention — matched V13 price bands

The aggregate comparable-family result remains on `grades.html` but is **not
repeated on each game card**. Each matchup instead displays one compact
historical price band drawn from the *same V13-represented selections* used to
score the model: reconstructed V12 decisions when an eligible V13 reconstruction
exists, plus originally locked native V13 decisions. Unrelated earlier model
families and the unselected opposing team are excluded.

Eight approximately equal-count bands are derived anew from the *closing
moneylines of those selected sides*, not the all-family, two-sides-per-game
market calibration distribution. Within each band, the **market's no-vig
implied win rate** and the **realised V13-selected win rate** use the exact same
games, prices and denominator. The band also shows the actual W–L record, the
realised-minus-implied gap with its one-standard-error uncertainty, and the
realised margin above those games' own historical closing-price break-even.
This is a matched historical *benchmark*, not a separately estimated
market-only outcome rate. It avoids presenting unrelated populations as a
head-to-head model-versus-market comparison.

The matched band's reconstructed/native counts come from each row's
**original ledger model tag**, not the reconstructed selection field. Native
V13 performance is not promoted over the comparable reconstructed family, but
neither is reconstructed history described as prospectively pregame-locked.
The data-vintage caveat above applies unchanged.

These are descriptive retrospective price bands, **not** calibrated
probabilities or a game-specific expected edge. Do not select a band based on
its observed record or interpret one positive cell as a prospective pricing
advantage. Bands and boundaries change as the V13-represented history grows.
Current-game odds and its posted break-even are displayed separately from the
historical closing-price comparison.

The broader eight-band *all-family, both-side* market calibration remains an
independent diagnostic in `data/ledger_report.txt` and on the calibration
page. The complete exploratory Δ × price grid and multiple-search reference
remain in `data/ledger_report.txt`. The model's selection and live pricing
logic, ledger, reconstructions and registered tests are unchanged.

## How to verify

Run `python reconstruct_v13.py --dry-run` to regenerate the reconstruction
and snapshot split without changing the ledger. Inspect the matched dated
shadow CSVs for row-level timestamps. The card's source is `build_site.hybrid_branch_records`,
`build_site._market_price_distribution` and
`build_site._market_band_context_html`. The reporting guards are
`tests/test_v13_card_provenance.py` and
`tests/test_per_game_market_bands.py`.
