# Full-season historical replay

`historical_replay.py` reconstructs missing regular-season predictions from
historical Statcast pitches and MLB records, runs the shipped model functions,
and grades the reconstructed leans. Outputs are isolated from the original
ledger. It does not change production selection, model constants or grades.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_historical_replay.py tests/test_sp_ip_calibration.py -q
# Small real-data trial; inputs still include every earlier regular-season PA.
python historical_replay.py --season 2026 --start 2026-04-01 --end 2026-04-02
# Full season; includes early-season abstentions rather than inventing inputs.
python historical_replay.py --season 2026 --workers 6
# Reproduce the same run from cached source responses, with no network.
python historical_replay.py --season 2026 --workers 6 --offline
```

The **Full-season Statcast replay** GitHub Actions workflow runs a small real-data
trial on pull requests touching the replay. Its manual `full_season` option runs
the requested full range. It uploads the results as an artifact and never commits
to `data/`, deploys to Pages, or joins the `site-build` concurrency group.

## Outputs

- `replay_output/backtest.html`: standalone report with monthly results, full-game
  and F5 W/L/T, always-home comparison on identical full-game decisions, and all
  scheduled games including abstentions and exclusions.
- `predictions.csv`: one row per regular-season game in the evaluation window;
  game IDs, input cutoff, model tag, reconstruction basis, side calculations,
  original starting orders, and grades.
- `manifest.json`: source commit, run scope, source-data counts and coverage.
- `replay.log`: the production model's diagnostics for each historical slate.

The reports can be reviewed locally or published separately after review. They
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
  Statcast has no market prices: this report does not invent odds, calculate ROI,
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

The repository suite passed 1,536 tests with two skipped before the final
suspension-interval guard; focused replay/calibration tests also cover that
guard. CSV validation passed for all 397 committed CSV files.

The dependency bound `pandas<3` preserves the existing suite's numeric empty-group
behavior: pandas 3.1 raised a TypeError in the unchanged hitter-level join test.
The validated runtime uses pandas 2.3.3.

Full-season input audit: all 2,430 scheduled 2026 regular-season games passed
Statcast coverage and final-score checks, across 716,792 pitches and 183,849
plate appearances. This audits input completeness; the full-season prediction
replay has not yet been run. The 25,000-row Savant response cap was handled by
splitting date ranges.
