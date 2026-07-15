# MLB matchup-leans static site

A render-free static site that publishes daily MLB probable-pitcher vs
opponent-lineup **leans** (Statcast xwOBA + platoon-OPS), built from the
`Shrunk_mlb_matchup_render_consolidated` Colab notebook and deployed to GitHub
Pages on a schedule.

It pulls everything through keyless APIs (no browser, no secrets):

- **MLB StatsAPI** — slate, probables, rosters, bio, vL/vR splits, league baselines
- **Baseball Savant `gf?game_pk=`** — posted lineups (JSON)
- **Baseball Savant CSV leaderboards** — custom (xwOBA/xBA/xSLG/EV/LA/HardHit/K/BB) + batted-ball, cached once per day

The model matches the updated notebook: log5 / odds-ratio matchup anchored on
league average (`M = B·P/L`, additive for EV/LA), with `edge = M − L` as the
signal. The platoon lens regresses each side's vs-hand OPS toward an
overall×league-platoon prior and is reliability-gated. Lean is xwOBA-driven;
platoon edge shows alongside with an AGREE / DIVERGE consensus tag.

Model version `xw+plat_consol_v2` (set via `MODEL_TAG` in the workflow) adds:

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

## Files

| Path | Purpose |
|------|---------|
| `build_site.py` | One-shot generator: fetch → matchup dataframes → writes `public/index.html` and `public/grades.html` (fully self-contained: inline CSS, dark-mode via `prefers-color-scheme`, no external assets). Also dumps the day's leans to `data/leans_<date>_{xw,pl}.csv` for the grading ledger. |
| `grade_leans.py` | Grading ledger: ingests the lean dumps as pending rows, grades them against StatsAPI linescores (full-game + F5), attaches closing DK moneylines (via `market_backfill`), writes `data/mlb_lean_ledger.csv` + `data/ledger_report.txt`. |
| `market_backfill.py` | Odds join: attaches ESPN/DraftKings opening + closing moneylines and the devigged home close probability to settled ledger rows (score-verified join, idempotent, no silent defaults), and computes the vs-market scoreboard. |
| `run_market_update.py` | Headless CLI for the odds join: `--dry-run` preview, one-off backfills, `--merge-backfill` for pre-enriched files. CI doesn't need it (grading calls `attach_market` directly); it's for local runs. |
| `.github/workflows/build.yml` | Scheduled + manual workflow: build → grade → commit ledger → deploy Pages. |
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
- **Cron cadence.** Scheduled passes run a morning projected pass plus a window
  through the afternoon/evening to catch posted `gf` lineups (scheduled
  workflows can lag/skip under load — fine for a leans page).

## One-time setup

Enable Pages with **Settings → Pages → Build and deployment → Source: GitHub
Actions**. After that the workflow deploys on each scheduled run (and on manual
`workflow_dispatch`). The site is served at `https://dave356w.github.io/dave356w/`.

## Grading ledger

Grades are also rendered into the site: the main page shows a **records
strip** (headline xwOBA + platoon records for the current `MODEL_TAG`,
linking to the ledger), and **`grades.html`** shows summary chips plus the
full ledger table — every game's leans, final/F5 scores, and W/L/T grades,
pending and void rows included. Both render purely from
`data/mlb_lean_ledger.csv`; grading runs before the build in CI (with a
second pass after it to ingest the day's fresh dumps), so the page reflects
last night's results in the same run.

Every CI run, `grade_leans.py`:

- **Ingests** any `data/leans_*_xw.csv` not yet in the ledger as `pending`
  rows. Re-runs on the same date refresh still-pending rows (SP scratches /
  lineup swaps up to first pitch); graded rows are immutable.
- **Grades** all pending rows via `schedule?hydrate=linescore` (one call per
  date): full-game and first-5-innings W/L/T per lean. Live games stay
  pending; postponed/cancelled go `void`.
- **Attaches market odds** to settled rows still missing them:
  ledger row → StatsAPI gamePk (date + teams, doubleheaders disambiguated by
  probable-pitcher surname, join verified by final score) → ESPN event →
  DraftKings (provider 100) opening/closing moneylines + devigged
  `close_p_home`. Idempotent; a row that can't be verified keeps NaN market
  columns and retries next run. A market outage never fails the grading run.
  `grades.html` then shows each lean's closing ML and a vs-market scoreboard
  (record vs market-expected wins → z, flat-stake ROI).
- **Reports** records to the Actions log and `data/ledger_report.txt`
  (overall, reliable-only platoon subset, |Δ| terciles, DIVERGE head-to-head,
  and — once 120 graded F5 decisions accumulate under the current
  `MODEL_TAG` — an SP-vs-lineup logit weight fit).

The ledger persists by being committed: the workflow's `Commit ledger` step
pushes `data/` back to `main` on each run (the `contents: write` permission).
The ~4:15am ET cron is the grading pass — it runs after night games end and
grades the previous slate. Any model change must bump `MODEL_TAG` (env var on
the `Grade leans` step) so pre/post-change games never mix in the records or
the weight fit.
