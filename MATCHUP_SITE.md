# MLB matchup-leans static site

A render-free static site that publishes daily MLB probable-pitcher vs
opponent-lineup **leans** (Statcast xwOBA + platoon-OPS), built from the
`Shrunk_mlb_matchup_render_consolidated` Colab notebook and deployed to GitHub
Pages on a schedule.

It pulls everything through keyless APIs (no browser, no secrets):

- **MLB StatsAPI** — slate, probables, rosters, bio, vL/vR splits, league baselines
- **Baseball Savant `gf?game_pk=`** — posted lineups (JSON)
- **Baseball Savant CSV leaderboards** — custom (xwOBA/xBA/xSLG/EV/LA/HardHit/K/BB) + batted-ball, cached once per day

The model matches the updated notebook's multiplicative-ratio matchup anchored
on league average (`M = B·P/L`, additive for EV/LA), with `edge = M − L` as the
signal. This is a relative-rate heuristic, not the probability-odds form of
log5. The platoon lens regresses each side's vs-hand OPS toward an
overall×league-platoon prior and is reliability-gated. Lean is xwOBA-driven;
the platoon lens is still computed and graded into the ledger for auditing but
is no longer surfaced on the cards (the display is xwOBA-only).

Historical model version `xw+plat_consol_v2` added:

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

Model version `xw+plat_consol_v3` leaves the prediction math unchanged and adds
a hard pregame snapshot lock. Its performance results therefore append to the
v2 series; row tags still identify the audit regime. Each dump stores its
capture and scheduled-start timestamps, and the ledger rejects new or refreshed
rows captured at/after scheduled first pitch. Ledger identity is
`(game_pk, game_date)`, so a postponed game that keeps its MLB gamePk can be
recorded again on its make-up date.

Model version `xw+plat_consol_v4` re-weights the lineup composites:

- **Batting-order slot weighting** — the xwOBA lineup composite and every
  platoon-OPS aggregate are weighted by expected plate appearances *per
  batting-order slot* (leadoff ~4.61 PA/game → 9-hole ~3.76, `LINEUP_SLOT_PA`)
  instead of by each hitter's season volume (batted-ball events / handedness
  split PA). Slot weighting reflects tonight's in-game exposure rather than who
  has simply logged the most playing time; it falls back to the old
  season-volume weights wherever a batting order is unavailable
  (`USE_SLOT_PA_WEIGHTS`). This changes the prediction math, so v4 starts a new
  `RECORD_TAGS` family and its games never mix with v2/v3 in the records.

Model version `xw+plat_consol_v5` adds xwOBA shrinkage on top of v4:

- **Empirical-Bayes xwOBA shrinkage** — before they drive the lean, each
  hitter's season xwOBA and the starter's season xwOBA-allowed are regressed
  toward the league xwOBA baseline by sample size, `x* = (n·x + K·prior)/(n+K)`
  (`shrink_xwoba`). Both sides share the league baseline as the prior. `K` is
  estimated by **method of moments per player pool** each build
  (`estimate_shrink_k`): sampling noise scales as `1/n`, so the gap between the
  unweighted and the PA-weighted dispersion of the leaderboard identifies the
  within-PA and between-player variance components, and `K = σ²/τ²` — no fixed
  per-PA constant. The estimate is clamped to a plausible PA band with a fixed
  fallback (`K_BAT_*` / `K_PIT_*`) and logged each run. Shrinkage touches only
  xwOBA (the lean stat); other columns and the raw per-hitter card values are
  untouched. This changes the prediction math, so v5 starts a new `RECORD_TAGS`
  family and its games never mix with v4 or v2/v3.

### Historical opener fallback (v5)

A probable pitcher whose recent starts average fewer than `OPENER_MAX_AVG_IP`
innings (over at least `OPENER_MIN_STARTS` starts) is treated as an **opener**:
his own Statcast line reflects only a handful of batters and is not
representative of the innings his club will actually pitch. For those sides the
xwOBA lean substitutes a **batters-faced-weighted aggregate of the club's
rostered pitching staff** (built from the pitcher custom leaderboard the build
already fetches, plus one active-roster call per opener club) for the opener's
own numbers; the swap happens in the lookup dicts, so the whole matchup
pipeline downstream uses the staff numbers. The aggregate carries the staff's
total batters faced as its sample size, so the xwOBA shrinkage step barely pulls
it toward league. If too few staff pitchers appear in the leaderboard
(`OPENER_MIN_STAFF`) the fallback is skipped and the opener's own line is kept.

The platoon lens was deliberately left untouched — an opener's tiny vL/vR split
already fails the reliability gate, so that lens abstains on its own. Openers
were rare, so this refinement stayed inside the `xw+plat_consol_v5` family;
every affected side was flagged in the ledger.

### Full-game pitching blend (v6)

Model version `xw+plat_consol_v6` replaces the starter-only xwOBA pitching
input—and v5's opener-only whole-staff substitution—with one consistent
nine-inning construction for every side:

1. Estimate the probable pitcher's expected innings from pre-slate game logs.
   A normal starter blends his last-five start average with his season
   IP/start; sparse histories regress toward `SP_IP_PRIOR`. An opener uses the
   short-start or recent-relief workload that caused his classification.
2. Build a **role-filtered bullpen pool** from the active roster plus one
   season pitching-role call per club. Pitchers qualify when no more than 35%
   of their appearances were starts and they average no more than 3.0 IP per
   appearance. This removes regular rotation starters while retaining long or
   bulk relievers in the pooled estimate.
3. Shrink every included reliever's Savant xwOBA independently by BF, then
   average those talents using estimated relief BF
   (`season BF × (1 − start share)`).
4. Complete nine innings:
   `P_game = (E[IP_SP]·SP_xwOBA + (9−E[IP_SP])·RP_xwOBA) / 9`.

v6 deliberately does **not** identify or assign innings to a specific bulk
follower; long/bulk arms contribute only through the bullpen pool. When a
normal side lacks a trustworthy relief pool, it degrades visibly to the
probable starter's value. For an opener only, v5's whole-staff aggregate
remains an audited emergency fallback so one inning is never treated as nine.

The starter card continues to show the probable pitcher's own shrunk xwOBA,
K−BB%, and xERA. The lean uses the blended xwOBA and the card states the
expected-IP + bullpen basis. Dumps and the ledger preserve the starter xwOBA,
bullpen xwOBA, expected innings, relief-pool size/BF, opener classification,
and pitching basis for audit.

This is a prediction-math change, so v6 starts a new `RECORD_TAGS` family.
The v2/v3, v4, and v5 rows remain immutable in the ledger; the Actions report
shows family history separately while the current-family fit uses v6 only.

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
- `MODEL_TAG` — row-level model/audit lineage for newly captured predictions.
- `RECORD_TAGS` — comma-separated tags whose unchanged prediction math should
  be summarized as one continuous performance family.

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
- **Slate-aware cadence.** A lightweight Actions poll runs every 15 minutes
  from 10am–11:59pm ET. `schedule_gate.py` fetches that day's MLB schedule and
  launches the full build only when at least one game is 15–90 minutes from
  first pitch; the wide window rides out Actions queue jitter, and pending-row
  refresh keeps the lock at the last pregame snapshot. A separate 4:17am ET
  pass grades the prior slate. Push and manual runs always build immediately.
- **Chronological slate.** Matchup cards render by scheduled first pitch rather
  than lean strength; doubleheader game number and gamePk provide stable ties.

## One-time setup

Enable Pages with **Settings → Pages → Build and deployment → Source: GitHub
Actions**. After that the workflow deploys on each scheduled run (and on manual
`workflow_dispatch`). The site is served at `https://dave356w.github.io/dave356w/`.

## Grading ledger

Grades are also rendered into the site: the main page shows a **records
strip** for the complete model lineage, linking to the ledger.
**`grades.html`** preserves that combined headline and shows every game's
xwOBA lean, closing ML, final score, and full-game W/L/T grade, with pending
and void rows included. The public page intentionally omits model-family
history and per-row model labels; those remain available in the underlying
ledger and Actions report. Both pages render purely from
`data/mlb_lean_ledger.csv`; grading runs before the build in CI (with a second
pass after it to ingest the day's fresh dumps), so the page reflects last
night's results in the same run.

The platoon-OPS lens and first-5-innings (F5) results are still computed and
recorded in the ledger for auditing (`grade_leans.py` grades both), but the
site display was pared back to the xwOBA full-game lens only; the pitcher card
shows Statcast **xERA vs season ERA** in place of the last-5-starts ERA.

Every CI run, `grade_leans.py`:

- **Ingests** timestamped pregame `data/leans_*_xw.csv` snapshots not yet in
  the ledger as `pending` rows. Re-runs refresh still-pending rows only while
  the snapshot precedes scheduled first pitch; late and legacy-unverified
  refreshes are rejected, and graded rows are immutable.
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
- **Reports** the current `RECORD_TAGS` family to the Actions log and
  `data/ledger_report.txt` (overall, reliable-only platoon subset, |Δ|
  terciles, DIVERGE head-to-head, and — once 120 graded F5 decisions
  accumulate — a pitching-vs-lineup logit weight fit), followed by immutable
  record lines for every historical model family.

The ledger persists by being committed: the workflow's `Commit ledger` step
pushes `data/` back to `main` on each run (the `contents: write` permission).
The ~4:17am ET cron is the grading pass — it runs after night games end and
grades the previous slate. A prediction-math change must bump `MODEL_TAG` and
start a new `RECORD_TAGS` family so incompatible games never mix in the records
or the weight fit. An audit-only tag change may remain in the same family.
