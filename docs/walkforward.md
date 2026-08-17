# Current-model walk-forward replay

`walkforward.py` replays the current `build_site.py` model chronologically
against completed games in `data/mlb_lean_ledger.csv`. It writes a separate
ledger and never mutates the production record.

## What is point-in-time

For a slate date `D`:

- Savant batter and pitcher leaderboards use Statcast Search date bounds ending
  on `D-1`.
- pitcher role lines and league ERA use StatsAPI `byDateRange` ending on `D-1`;
- recent starter/opener profiles discard every appearance on or after `D`;
- the active roster is requested for `D`;
- the starter name recorded in the production ledger is authoritative, even if
  StatsAPI later shows the actual starter after a scratch;
- historical starting lineups come from the completed game's gamefeed; and
- v12 expected-IP calibration receives only prior regenerated replay rows.

The runner imports `build_site.py` for shrinkage, lineup aggregation, matchup
multiplication, starter/bullpen sequencing, workload shares and abstention. It
imports `grade_leans.py` for dump-to-game conversion and W/L/T semantics. There
is no copied walk-forward model.

## Fidelity

Each result is labelled:

- `exact_pregame`: archived production Savant cache, recorded starter, and both
  posted starting lineups are available;
- `historical_lineup`: the starting lineup is known, but Savant's currently
  reconstructed date-bounded values are being used;
- `reconstructed`: one or both lineups require the dated active-roster fallback;
- `insufficient_history`: a required game, starter or nine-man lineup could not
  be recovered.

Savant can revise historical player values after a live build. A current dated
query is temporally bounded but is not byte-identical to an old production
snapshot. The runner therefore does not call it `exact_pregame` and the strict
parity gate refuses to run on it.

The manual **Export dated Savant cache** workflow restores a retained
`savant-YYYY-MM-DD` Actions cache and exposes it as a downloadable artifact.
Extract that artifact and pass its directory with `--snapshot-dir`.

## Commands

Full sequential replay:

```bash
python walkforward.py
```

This writes:

- `data/walkforward_current.csv`
- `data/walkforward_report.txt`

Interrupted runs resume by `replay_config_hash`; dated source downloads are
cached under `.walkforward_cache/`.

Strict recent-slate parity against an archived production snapshot:

```bash
python walkforward.py \
  --acceptance-date 2026-08-16 \
  --snapshot-dir /path/to/extracted/savant-2026-08-16 \
  --acceptance-tolerance 0.002
```

Optional historical OPS/platoon reconstruction is available with
`--with-platoon`. It is off by default because the current v12 xwOBA decision
path does not read those split tables and each date requires four additional
Savant exports.

## Interpretation

This answers: “If today's specification had existed from the first ledger
date and received information sequentially, what would it have done?” It
removes temporal feature leakage. It does not remove model-selection leakage
from choosing v12 after observing part of the same season.
