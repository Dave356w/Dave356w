# MLB matchup leans

Daily MLB probable-pitcher vs opponent-lineup leans (Statcast wOBA), built and
published as a static site by GitHub Actions on a pregame trigger.

**<https://dave356w.github.io/Dave356w/>**

Current forward-test model: `xw+plat_consol_v12` — Savant xwOBA with fixed
`K = 100` shrinkage, sequential starter/bullpen phases, PA-share workload
weighting, exposure-centred starter platoon offsets, unmeasured-starter
abstention, and walk-forward-calibrated expected starter IP.

| Where to look | For |
|---|---|
| [`MATCHUP_SITE.md`](MATCHUP_SITE.md) | The model. Start at §"The current model" for what the code does today; everything after it is a version changelog. |
| [`CLAUDE.md`](CLAUDE.md) | Working standards, the `RECORD_TAGS`/`SCALE_TAGS` family table, and the anti-pattern catalogue. |
| [`docs/build_logic_validation.md`](docs/build_logic_validation.md) | Historical xwOBA review; its structural checks still describe the inherited v10 construction. |
| [`docs/f5_market_validation.md`](docs/f5_market_validation.md) | First-5-innings market capture and its data-quality audit. |
| [`docs/pitch_mix_theory.md`](docs/pitch_mix_theory.md) | Design notes for the pitch-type-conditioned matchup (shadow-only, not shipped). |
| [`docs/walkforward.md`](docs/walkforward.md) | Parallel current-model replay, point-in-time source rules, fidelity labels and acceptance gate. |

```bash
pip install -r requirements.txt
python build_site.py          # writes public/index.html for today's ET slate

python shadow_report.py       # paired wOBA-vs-xwOBA read on the shadow dumps
python walkforward.py         # separate sequential current-model replay ledger

pip install pytest            # deliberately not in requirements.txt
python validate_data_files.py # run both before opening a PR
python -m pytest tests/ -q    # also CI-gated on every PR and push to main
```
