# MLB matchup-leans static site

A render-free static site that publishes daily MLB probable-pitcher vs
opponent-lineup **leans** (Statcast xwOBA + platoon-OPS), built from the
`Shrunk_mlb_matchup_render_consolidated` Colab notebook and deployed to GitHub
Pages on a schedule.

It pulls everything through keyless APIs (no browser, no secrets):

- **MLB StatsAPI** — slate, probables, rosters, bio, vL/vR splits, league baselines
- **Baseball Savant `gf?game_pk=`** — posted lineups (JSON)
- **Baseball Savant CSV leaderboards** — custom (xwOBA/xBA/xSLG/EV/LA/HardHit/K/BB) + batted-ball, cached once per day

The model is unchanged from the notebook: log5 / odds-ratio matchup anchored on
league average (`M = B·P/L`, additive for EV/LA), with `edge = M − L` as the
signal. The platoon lens regresses each side's vs-hand OPS toward an
overall×league-platoon prior and is reliability-gated. Lean is xwOBA-driven;
platoon edge shows alongside with an AGREE / DIVERGE consensus tag.

## Files

| Path | Purpose |
|------|---------|
| `build_site.py` | One-shot generator: fetch → matchup dataframes → writes `public/index.html` (fully self-contained: inline CSS, dark-mode via `prefers-color-scheme`, no external assets). |
| `.github/workflows/build.yml` | Scheduled + manual workflow that builds and deploys to Pages. |
| `requirements.txt` | `requests`, `numpy`, `pandas`. |

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

## Possible follow-ups

- Commit the dated leaderboard CSVs to a `data` branch on each run for a free
  historical archive of projected slates (model-grading / CLV snapshots). The
  generator currently doesn't emit CSVs — they'd need re-adding plus a commit
  step.
