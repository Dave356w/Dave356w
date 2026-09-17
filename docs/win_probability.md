# Home-win probability mapping

`win_probability.py` maps the frozen, home-oriented `xw_net` to a full-game
home-win probability. It evaluates probabilities without changing the active
lean, hybrid selections, model tags, ledger, or existing registrations.

## Model

For a saved matchup difference `d`:

```
x = d / 0.020
P(home wins) = sigmoid(a + b*x)
objective = sum(log(1 + exp(a + b*x)) - y*(a + b*x)) + (a*a + b*b)/2
```

`y=1` means the home team won. In `grade_leans.rows_from_dump`, `xw_net` is
`home_off_edge - away_off_edge`, so positive values favor home. The input
retains its sign; neither absolute magnitude nor the selected team's direction
is substituted. A finite zero is valid for this mapping, even though the
directional model abstains at zero. A missing feature is never filled with zero.

This is ridge logistic regression with fixed penalty `1.0` on both coefficients.
The intercept learns the home outcome frequency while shrinking toward zero;
the slope also shrinks toward zero. The fixed `0.020` unit makes the slope's
penalty meaningful without learning a scale from future data. These settings
are engineering defaults, not winners from a historical parameter sweep.
The numerical fit uses stable log-sum-exp arithmetic and a damped Newton step;
no new runtime dependency is required.

Only the exact requested source model tag is fitted (default:
`build_site.MODEL_TAG`), with `model_metric == xwOBA`. Sharing scale units is
not enough to pool a different forecasting lineage into this calibration.
The calibrator has its own version, `xwoba_home_logit_v1`.

## Chronological evaluation

1. Accept final, non-tied full-game results with finite deltas and verified
   `snapshot_utc < scheduled_start_utc`, plus `lock_status == pregame`.
   Reject ambiguous duplicate completed games. Exclusions are counted once,
   in the order printed in the report.
2. Group complete test slates by `game_date`. Start only after at least 100
   eligible training games and seven earlier slates.
3. Fit one model for the slate using strictly earlier dates. If a forecast
   was saved on an earlier Eastern date, move the entire slate's training
   cutoff back accordingly. There is no within-slate refit, random train/test
   split, full-history standardization, or held-out hyperparameter selection.
4. Score each eligible test game exactly once. The comparison with the
   historical home rate uses only the same earlier training games, with
   Jeffreys smoothing `(home_wins + 0.5)/(games + 1)`.
5. Compare the model and market on identical out-of-sample games. Use only
   `pregame_p_home` in `(0, 1)`, with an aware `pregame_market_utc` at or before
   that forecast's snapshot. There is **no closing-price fallback**. Missing
   or later market snapshots exclude a game from the paired comparison, not
   from model training or the overall probability evaluation.

Primary scores are mean Brier loss and log loss, where lower is better.
Accuracy is secondary. `calibration.csv` gives fixed probability bins for
both models on the identical paired games; sparse bins are descriptive.
The paired model-minus-market differences have 95% percentile intervals from
2,000 whole-slate bootstrap resamples at a fixed seed. Those intervals are
conditional on the fitted replay forecasts and assume exchangeable slate
blocks; they do not capture all fitting, serial, or cross-season uncertainty.
They are not adjusted for repeated monitoring.

This is **chronological retrospective evaluation**, not a newly registered
prospective result. The design was specified after inspecting historical
results. The ledger also lacks historical final-result availability timestamps:
the replay assumes earlier-slate outcomes were available. Suspended games,
late grading, or outcome revisions would require historical ledger snapshots
to certify exact point-in-time training availability. No Savant history is
rebuilt from current leaderboards.

## Run and inspect

```bash
python -m pip install -r requirements.txt
python win_probability.py
python win_probability.py --ledger data/mlb_lean_ledger.csv --out-dir win_probability_output
```

Outputs:

| File | Contents |
|---|---|
| `predictions.csv` | One chronological prediction per evaluated game, outcome, saved market, timestamps, training cutoff/counts, and coefficients |
| `folds.csv` | One row per evaluated slate, including its training window and sample sizes |
| `calibration.csv` | Fixed probability bins on the paired market sample |
| `summary.json` | Settings, exact input SHA-256, exclusions, scores, paired intervals, and final-fit parameters |
| `report.txt` | Readable evaluation with its limitations |

The final fit uses all eligible completed rows and is exported for later-date
predictions only. It is not used for the historical scores above. To reuse it,
construct `LogitModel` from its `intercept`, `slope_per_unit`, `delta_unit`,
and `ridge`, then call `predict(home_oriented_delta)`. Preserve the source tag,
training dates, and input hash with any subsequent forecasts; a coefficient
copied from a current fit must not be applied retrospectively and called OOS.

The **Win probability validation** workflow runs manually or on a PR touching
this module, its tests, or the workflow. It uploads the five outputs and has
read-only repository permissions. It does not enter the daily site's critical
path or automatically promote a model.

## Initial measurement — September 17, 2026

Ledger at commit `b3c2901d26fc072398fe5803e7f8d1c2e038f400`, SHA-256
`3815f6be373612b8f89fd84dcf9842de0191b44cd5bbc442280b0f89186f441e`:

- 436 eligible completed v12 games; 106 warm-up games.
- 330 chronological test games across 25 slates, August 23–September 16.
- 213 saved-market pairs across 16 slates, September 1–16.

| Same 213 games | Brier loss | Log loss | Accuracy |
|---|---:|---:|---:|
| Regularized xwOBA probability | 0.238719 | 0.670143 | 58.22% |
| Saved pregame market | 0.238908 | 0.670412 | 56.34% |
| Earlier-slate home rate | 0.249569 | 0.692290 | 54.00% |

Model-minus-market Brier difference: **-0.000189**, 95% slate-bootstrap
interval **[-0.012239, +0.010736]**. Log-loss difference: **-0.000269**,
interval **[-0.025993, +0.022608]**. The model and market are effectively tied
in this evaluation; this does not demonstrate incremental predictive value.
On all 330 test games, model Brier/log loss are **0.238686 / 0.669841**, versus
**0.250072 / 0.693305** for the earlier-slate home-rate baseline.

The final fit through September 16 is `a=0.183805`, `b=0.393426` per 0.020
xwOBA difference. These are fitted parameters, not constants embedded in the
source. Its home intercept can change the most likely winner relative to the
raw sign; this probability audit does not replace that directional forecast.

Method references: [probability calibration and proper scoring rules](https://scikit-learn.org/stable/modules/calibration.html),
[chronological validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
