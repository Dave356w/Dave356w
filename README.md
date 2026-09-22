# XWOBA MLB Matchups

Daily MLB matchup leans from the v13 Statcast model, built and published as a
static site by GitHub Actions on a pregame trigger. The site publishes the
model's own lean. The former hybrid selection rule remains in the repository
for historical reconciliation and registered monitoring.

**<https://dave356w.github.io/Dave356w/>**

Current model: `xw+starter_blend_v13` — Savant xwOBA with fixed
`K = 100` shrinkage, sequential starter/bullpen phases, PA-share workload
weighting, exposure-centred starter platoon offsets, unmeasured-starter
abstention, calibrated expected starter IP, and a centred 50/50 xwOBA/wOBA
blend on the starter rate.

Historical views have different bases: the ledger retains original decisions;
the public v13 history also includes stored v13 reconstructions of v12 rows.
Those reconstructions are hindsight analysis. The dynamic-price shadow uses
that combined history with closing prices and does not change model decisions.

| Where to look | For |
|---|---|
| [`MATCHUP_SITE.md`](MATCHUP_SITE.md) | The model. Start at §"The current model" for what the code does today; everything after it is a version changelog. |
| [`CLAUDE.md`](CLAUDE.md) | Working standards, the `RECORD_TAGS`/`SCALE_TAGS` family table, and the anti-pattern catalogue. |
| [`REGISTRATIONS.md`](REGISTRATIONS.md) | Registered tests and checkpoints. Logic only — live readings are in `data/ledger_report.txt`. |
| [`docs/repository_cleanup_review.md`](docs/repository_cleanup_review.md) | Dated repository audit, staged cleanup plan, and delta/market separation benchmark. |
| [`docs/kalshi_paper.md`](docs/kalshi_paper.md) | Read-only prospective Kalshi MLB paper fills, saved-price comparisons and isolated audit. |
| [`docs/win_probability.md`](docs/win_probability.md) | Regularized home-win probability mapping, chronological evaluation, and comparison with the saved pregame market. |
| [`docs/build_logic_validation.md`](docs/build_logic_validation.md) | Historical xwOBA review; its structural checks still describe the inherited v10 construction. |
| [`docs/f5_market_validation.md`](docs/f5_market_validation.md) | First-5-innings market capture and its data-quality audit. |
| [`docs/pitch_mix_theory.md`](docs/pitch_mix_theory.md) | Design notes for the pitch-type-conditioned matchup (shadow-only, not shipped). |

```bash
pip install -r requirements.txt
python build_site.py          # writes public/index.html for today's ET slate

python shadow_report.py       # paired wOBA-vs-xwOBA read on the shadow dumps
python win_probability.py     # probability audit; writes win_probability_output/
python research/delta_market_audit.py  # descriptive v13/closing-market comparison; JSON to stdout

pip install pytest            # deliberately not in requirements.txt
python validate_data_files.py # run both before opening a PR
python -m pytest tests/ -q    # also CI-gated on every PR and push to main
```
