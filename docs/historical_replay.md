# Full-season historical replay

`historical_replay.py` reconstructs missing regular-season predictions from
historical Statcast pitches and MLB records, runs the shipped model functions,
and grades the reconstructed leans. Outputs are isolated from the original
ledger. It does not change production selection, model constants or grades.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_historical_replay.py tests/test_sp_ip_calibration.py -q
# Small real-data trial; inputs still include every earlier regular-season PA.
python historical_replay.py --season 2026 --start 2026-04-01 --end 2026-04-02 --fetch-missing-odds
# Full season; includes early-season abstentions rather than inventing inputs.
python historical_replay.py --season 2026 --workers 6 --fetch-missing-odds
# Reproduce the same run from cached source responses, with no network.
python historical_replay.py --season 2026 --workers 6 --offline
```

The **Full-season Statcast replay** GitHub Actions workflow runs a small real-data
trial on pull requests touching the replay. Its manual `full_season` option runs
the requested full range. It uploads the results as an artifact and never commits
to `data/`, deploys to Pages, or joins the `site-build` concurrency group.

For the complete repository-hosted prediction and ROI run, use **Full-season MLB
backtest** (`.github/workflows/full-season-backtest.yml`). Opening or updating a
PR that changes that workflow starts the full 2026 replay automatically; there
is no two-day smoke-test fallback. After merging, launch another run from Actions
with **Run workflow**. The season defaults to 2026, and blank date inputs evaluate
March 1 through October 5 (or yesterday for a season still in progress), filtering
to regular-season games. Override dates only when a shorter range is intentional.

This workflow always fetches missing historical closing odds after freezing all
predictions. Its Actions summary reports game coverage, priced selections, W/L/P,
unit profit and ROI separately for full-game and F5 markets. Download the
`full-season-backtest-<season>-<run-id>` artifact for the HTML report, CSV, manifest,
and logs. A failed or timed-out run retains available checkpoints and explicitly
reports incomplete progress rather than final ROI. The replay command has a
320-minute limit inside a 350-minute job so artifact upload has time to finish.

## Outputs

- `replay_output/backtest.html`: standalone report with monthly results, full-game
  and F5 W/L/T, always-home comparison on identical full-game decisions, and all
  scheduled games including abstentions and exclusions.
- `predictions.csv`: one row per regular-season game in the evaluation window;
  game IDs, input cutoff, model tag, reconstruction basis, side calculations,
  original starting orders, and grades.
- `manifest.json`: source commit, run scope, source-data counts and coverage.
- `replay.log`: the production model's diagnostics for each historical slate.

The reports include closing-price ROI, monthly coverage, same-game controls, and
per-game selected moneylines, unit profit and matching status. They can be
reviewed locally or published separately after review. They
are deliberately not inserted into the live grading ledger or headline record.

## Historical availability and limitations

- Performance features use strictly earlier **dates**, including on doubleheader
  days. Resumed games' complete lines are withheld until the recorded resumption
  date to prevent future innings entering an earlier day's inputs. Intervening
  slates are explicitly excluded: MLB role totals and game logs can otherwise
  attribute the completed line to its original date. Their original
  pregame predictions are excluded because that state cannot be reconstructed
  from a final box score alone.
- Full-season Statcast history is downloaded in date chunks and checked against
  completed MLB game IDs and final scores. Missing columns, duplicate pitches,
  missing historical games, truncated final scores and upstream failures stop the
  run; they cannot silently become partial-season inputs.
- Per-player rates use plate appearances, retaining BB/HBP/K, with wOBA's own
  denominator for the rate and total PA/BF for model shrinkage. Untracked batted
  balls use the observed wOBA weight as an explicit approximation, matching the
  existing research collector. Their count is reported. Current historical
  Statcast estimates may differ from values originally served before a game.
- Starting lineups and starters come from final box scores: original batting
  slots (`100` through `900`) exclude later pinch hitters. These identities are
  reconstructed game facts, not proof of their availability at a particular
  pregame announcement time. Historical active rosters use dated MLB requests;
  the replay never selects relievers based on who actually appeared that night.
- Batter/starter shrinkage, centred starter blending, platoon adjustment,
  bullpen membership/rates, expected-IP calibration, PA phase weights and
  velocity adjustment use production functions. Only prediction inputs are
  substituted. Display-only statistics and the optional OPS/pitch-mix shadows
  are omitted.
- Workload calibration starts empty at the requested evaluation start and learns
  only from earlier replay slates. It resets each run. A midseason smoke test's
  calibration therefore differs from a replay that began in March. No current
  production ledger or future-slate fit is read.
- This freezes today's model and retrospectively applies it. Some constants and
  model choices were selected after some evaluated games; the result is not a
  prospective record or a pristine held-out test. Earlier seasons used to fit
  velocity coefficients are not out-of-sample validation of that coefficient.
- F5 ties remain separate from wins/losses. Incomplete fifth innings are ungraded.
  Statcast has no market prices. The separate odds layer uses only verified
  archived or fetched historical closing moneylines. It does not invent odds
  or treat the matchup delta as a calibrated win probability.

The code exposes a scoped data-provider layer rather than copying the model.
One production fix makes an explicitly absent expected-IP fit remain identity;
previously `calibrate_sp_ip(raw, None)` could accidentally load today's ledger
when a replay had no usable prior fit. Omitted-fit live callers retain their
existing behavior.

## Validation

The April 1–2, 2026 real-data trial accounted for all 18 scheduled games:
16 leans (full-game 7–9; F5 6–7–3) and two unmeasured-starter abstentions.
An offline replay reproduced every selection and grade; numeric outputs agreed
within 1e-14 across the pandas versions used during setup. This small trial
validates plumbing and reproducibility, not predictive performance.

The original replay suite passed 1,537 tests with two skipped, including the
suspension-interval guard. CSV validation passed for all 397 committed CSV files.

The dependency bound `pandas<3` preserves the existing suite's numeric empty-group
behavior: pandas 3.1 raised a TypeError in the unchanged hitter-level join test.
The validated runtime uses pandas 2.3.3.

Full-season input audit: all 2,430 scheduled 2026 regular-season games passed
Statcast coverage and final-score checks, across 716,792 pitches and 183,849
plate appearances. This audits input completeness; the full-season prediction
replay has not yet been run. The 25,000-row Savant response cap was handled by
splitting date ranges.


## Odds matching and ROI

Odds are attached only after every prediction and calibration row is frozen.
The default archive is `data/mlb_lean_ledger.csv`, read without modification.
`--fetch-missing-odds` additionally uses the existing ESPN/DraftKings parsers,
with source JSON retained in `replay_cache/odds/`. A failed or absent market
remains unpriced; it cannot become zero profit or an inferred moneyline.

Existing predictions can be graded without replaying the model:

```bash
python replay_market.py --predictions replay_output/predictions.csv \
  --fetch-missing --output replay_market_output
# Reproduce the price joins and returns entirely from cached responses:
python replay_market.py --predictions replay_output/predictions.csv \
  --fetch-missing --offline --output replay_market_output
```

Use `--odds-csv PATH` in either CLI for a compatible external archive. Its
identity fields are `game_pk` (or `gamePk`), `game_date`, `away`, `home`,
`full_away`, and `full_home`; prices are `close_away_ml`/`close_home_ml` and
optionally `f5_close_away_ml`/`f5_close_home_ml`. Team abbreviations are normalized
without swapping home/away. If both game-ID columns exist they must agree.
F5 prices use the two-way market where ties push; do not supply three-way prices.

The join verifies game ID, date, teams, and final scores. Identical duplicate
quotes do not duplicate turnover; conflicting prices or identities are rejected.
Unpriced void/postponed attempts are ignored when the same ID later settles.
For live historical fetching, the MLB ID is verified against the schedule before
matching an ESPN event with the same teams and final score. Same-score
same-day doubleheaders require a unique exact scheduled start, never a nearest
or arbitrary event. Only settled ESPN events supply closing prices.

One unit is **risked** on each priced model selection. A winning +150 earns
+1.5u; a winning -200 earns +0.5u; a loss costs -1u. A priced F5 tie earns 0u
and counts 1u in turnover. ROI is net profit divided by all units staked,
including pushes. Zero priced games gives an unavailable ROI, not 0%. Missing,
partial, invalid or conflicting prices and ungraded/incomplete games are excluded
with explicit reasons. American moneylines must be finite and have absolute
value at least 100; no default is substituted for an invalid quote.

Full-game prices never substitute for F5 prices, and saved pregame prices never
substitute for closing prices. Closing odds grade the already selected model
side; no quote changes the model's lean. The same-game always-home and
always-favourite controls use the exact same priced/scored rows and stake rule.
An exact pick'em breaks to home. Headline and monthly ROI are retrospective
closing-price benchmarks, not a prospective execution result.

`predictions.csv` records each market's prices, source, ESPN ID where available,
match basis, price status, selected-side line, settlement, stake and profit.
The manifest includes coverage, exclusions, sources, fetch failures, archive and
prediction hashes, and the market-grading source hashes. The original prediction
file and live ledger stay unchanged. Each market can succeed independently.

Real-data pricing trial: all 16 model leans from April 1–2, 2026 matched ESPN/DK
full-game closing prices: 7–9, -2.853597u on 16u risked, -17.834980% ROI. Both
unmeasured-starter abstentions remained excluded. F5 prices were unavailable for
all 16 selections, so F5 ROI remains unavailable. This is a plumbing/return
validation on a small historical sample, not full-season model performance.

ROI extension validation: 49 market-specific tests and the complete repository
suite passed (1,586 tests, two skipped, 389 subtests). All 397 committed CSV
files passed validation. The integrated replay preserved all original model
selections and grades, and the cached odds rerun reproduced selected prices,
unit profits and exclusion statuses.
