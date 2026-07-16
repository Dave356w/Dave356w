# Handoff: matchup site render redesign (build_site.py)

Repo: `Dave356w/Dave356w` (branch `main`). Read `MATCHUP_SITE.md` and
`CLAUDE_INSTRUCTIONS.md` first; those working standards apply here.

## Scope

Replace the card render section of `build_site.py` (everything from `ABBR`
down through the page assembly / CSS) with the layout in
`design/matchup_redesign_prototype.html` (commit that file to the repo as the
markup/CSS source of truth — copy its CSS verbatim, then template the HTML
into f-string emitters). **Display-only change: no model logic, no lean
computation, no ledger schema changes. Do NOT bump MODEL_TAG** — leans are
unchanged, so pre/post records remain comparable.

Everything else in `build_site.py` (fetch, matchup, platoon cells) stays
untouched, with the two small exceptions in items 4–5 below.

## What the new layout renders

Per game card, in order:

1. **Game header** — `AWAY @ HOME` abbrevs, game time + venue (from
   `slate_df.game_datetime_utc` / `venue`), lean pill (`Δxw`, unchanged
   logic), consensus line + AGREE/DIVERGE tag (unchanged logic).
2. **Market strip** — DK moneyline both sides (current ← open), total,
   devigged home implied %. New data; see item 5. Every cell degrades to `—`
   independently. Label: `as of build <time> PT`.
3. **Two side columns** (away SP | home SP), each:
   - SP block: name, throws badge, role line
     (`<TEAM> SP · faces <OPP> lineup · nR/nL/nS · n plt-adv`), thin-SP flag
     when `pit_low_sample`.
   - 4-stat row with league refs (item 4): xwOBA-against (`lg` sub), K%
     (`lg` sub), HardHit% (`lg` sub), OPS-allowed comp-weighted (`raw` sub).
   - Aggregate edges: `xw edge` (drives lean) + `ops edge`
     (+ `prior-driven` flag when not reliable). Same values as today.
   - **Lineup `<details>`** (open by default on desktop is fine —
     prototype has game 1 open, game 2 closed; make them all `open`):
     summary = `<OPP> LINEUP` + status badge (`posted` / `partial` /
     `projected`) + `wt xwOBA <x> · mx <y>`; body = 9-row table:
     order | name + bats badge + ◆ platoon-adv | pos | xwOBA |
     vs-hand OPS (split PA, low-PA flag) | mx | diverging edge bar.
   - Edge bar: per-hitter `mx_ops − league_ops_overall`, width
     `clamp(|edge|/0.100, 0, 1) * 50%`, warm right / cool left of the
     shared center axis. Missing split → empty bar + `—` cells (muted row).
4. Cards sorted by xwOBA Δedge (unchanged). Legend per prototype footer.

## Data mapping (all existing except odds)

| UI element | Source |
|---|---|
| Per-hitter xwOBA, pos | `opp_hitters_df` (STAT_COLS + bio) |
| bats, ◆ plt-adv, vs-hand OPS, split_pa, mx_ops, low_sample | `opp_platoon_detail_df`, join key `(game_pk, faced_pitcher, batter/Name)` |
| Lineup status badge | `resolve_lineup` flags / `lineup_projection_df`; `partial` requires the per-side resolution audit status if present, else map projected→`projected`, not→`posted` |
| Lineup summary wt xwOBA | `opp_lineup_agg_df` `opp_xwOBA` |
| Lineup summary mx | PA-weighted `mx_ops` over the side's `opp_platoon_detail_df` rows (same weighting as `mx_OPS`) |
| SP block stats + agg edges + consensus + lean | existing `matchup_df` / `matchup_platoon_df` |
| Batting order | preserve `opp_hitters_df` row order within `(game_pk, faced_pitcher)` — it is lineup order; do not sort |

Note: `opp_platoon_detail_df` and per-hitter rows are already computed today
and discarded at render. No new model computation.

### Item 4 — league refs on SP stats

`league_baseline` (already computed in `compute_league_baseline`) supplies
`xwOBA`, `K%`, `Hard Hit%`. Render as small `lg <val>` subs under each of the
three cells. These are PA-weighted **batter-population** baselines — league-
identical to the pitcher-side population by construction, fine as refs; keep
the existing formatting helpers (`f3`/`f1`). OPS-allowed keeps `raw` sub
(no lg sub — the platoon lg cell is composition-dependent).

### Item 5 — pregame odds fetch (only new data path)

New function in `build_site.py` (not `market_backfill` — that module stays
closing-line/post-settlement only):

- One request per slate to ESPN scoreboard:
  `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD`
  — **verify the exact endpoint/shape against `market_backfill.py`'s ESPN
  join before writing code; reuse its event-matching approach** (date +
  teams; doubleheader disambiguation by probable-pitcher surname where
  needed). Take DraftKings (provider 100) current + open moneylines and
  total if present.
- Devig current MLs to `implied home %` (same two-way normalization as
  `market_backfill`).
- **Failure isolation:** any exception or unmatched game → that game's
  market strip renders all `—`; a total odds outage must never fail the
  build (log a warning, continue). No retries beyond the standard session
  backoff. No silent defaults — missing means `—`, never 0/`even`.
- Pregame ESPN odds may legitimately be absent early morning; that is the
  `—` path, not an error.

### Build time in PT

Header + market strip show build time as `America/Los_Angeles`
(`built H:MM AM/PM PT`). **Slate-date logic stays ET** (3am ET rollover,
unchanged) — display only.

## Constraints (unchanged invariants — verify before commit)

- Output remains a single self-contained `public/index.html`: inline CSS, no
  external assets/fonts/JS libs. `<details>` needs zero JS.
- Dark/light via `prefers-color-scheme` (prototype CSS already does this).
- Hard fetch failure of core data still exits non-zero without writing
  `index.html`. Off-day and pre-slate pages still render (touch those code
  paths only to keep them compiling against the new CSS).
- Per-side/per-hitter data gaps degrade to muted `—`, never crash a card.
- `grades.html` render untouched.
- Escape player/team/venue strings going into HTML (names contain
  apostrophes/accents; current code interpolates raw — keep at least
  `html.escape` on names in the new emitters).

## Verification gate (before any commit)

1. `pip install -r requirements.txt`
2. `SLATE_DATE=2026-07-15 python build_site.py` (a completed full slate) —
   open `public/index.html`; check: every game renders 2 SP blocks + 2
   nine-row lineups; no `nan` strings anywhere (`grep -ci nan public/index.html`
   should be 0 for value cells); lg subs present on xwOBA/K%/HardHit%.
3. Today's slate (no `SLATE_DATE`): posted vs projected badges appear;
   odds strip populated or cleanly `—`.
4. Edge cases: a date with no games → "no games" page; simulate odds fetch
   failure (bad URL) → build succeeds with `—` strips.
5. Light + dark mode visual check (prefers-color-scheme emulation).
6. Confirm `data/leans_*.csv` dumps are byte-identical to a pre-change run
   for the same `SLATE_DATE` (proves display-only).

## Commit / deploy

- Single commit to `main` after the gate passes:
  `redesign: per-hitter lineup cards + pregame DK odds strip (display-only, no MODEL_TAG bump)`
- Include `design/matchup_redesign_prototype.html` and this file in the
  commit for provenance.
- Trigger `workflow_dispatch` on `.github/workflows/build.yml` and confirm
  the Pages deploy succeeds and the live page renders. If the Actions run
  fails, revert the commit rather than leaving the deploy pipeline red —
  the "don't clobber good output" guard only protects against fetch
  failures, not template exceptions.

## Explicitly out of scope

- Any change to lean math, shrinkage constants, grading, ledger schema,
  `market_backfill.py` closing-line logic, or MODEL_TAG.
- Live-updating odds (page is as-of-build by design; cron cadence unchanged).
- grades.html restyle (candidate follow-up, separate commit).
