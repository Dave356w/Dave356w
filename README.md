# MLB matchup leans

Daily MLB probable-pitcher vs opponent-lineup leans (Statcast xwOBA), built and
published as a static site by GitHub Actions on a pregame trigger.

**<https://dave356w.github.io/Dave356w/>**

Current model: `xw+plat_consol_v10` — a sequential starter/bullpen matchup whose
two phases are weighted by share of plate appearances.

| Where to look | For |
|---|---|
| [`MATCHUP_SITE.md`](MATCHUP_SITE.md) | The model. Start at §"The current model" for what the code does today; everything after it is a version changelog. |
| [`CLAUDE.md`](CLAUDE.md) | Working standards, the `RECORD_TAGS`/`SCALE_TAGS` family table, and the anti-pattern catalogue. |
| [`docs/build_logic_validation.md`](docs/build_logic_validation.md) | Independent review of the build logic, plus the current vs-market read. |
| [`docs/f5_market_validation.md`](docs/f5_market_validation.md) | First-5-innings market capture and its data-quality audit. |
| [`docs/pitch_mix_theory.md`](docs/pitch_mix_theory.md) | Design notes for the pitch-type-conditioned matchup (shadow-only, not shipped). |

```bash
pip install -r requirements.txt
python build_site.py          # writes public/index.html for today's ET slate
python -m pytest tests/ -q    # CI does not run these; local is the only gate
```
