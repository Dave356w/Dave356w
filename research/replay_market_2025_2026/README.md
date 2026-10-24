# v14 full-season replays against the market, 2025 and 2026

Research record, 2026-10-23/24. **Hindsight, not forward evidence.** The
replays freeze today's `xw+starter_velo_v14` and apply it to past games
using prior-date Statcast and each game's final starting lineups and
starters. Some constants were chosen after some of these games. Nothing
here changes the model, the ledger, `data/` or any registered rule.

`analysis.py` regenerates every table in `RESULTS.txt` from the files in
this folder plus `data/mlb_lean_ledger.csv`. Uncertainties are ±1 SE,
clustered by slate date.

## Contents

| Path | What it is |
|---|---|
| `replay_2025/`, `replay_2026/` | Full-season backtest workflow outputs: `predictions.csv`, `manifest.json`, `backtest.html`, logs. Run checkpoints are omitted (an intermediate copy of the predictions). |
| `espn_full_odds.csv` | ESPN full-game opening/closing moneylines for every graded game, all providers ESPN listed. |
| `espn_f5_odds.csv` | ESPN "1st 5 Innings Moneyline" (propBets type 136) value and opener per team, for the replay's book. |
| `espn_team_ids.csv` | ESPN team id → project abbreviation, used to orient F5 quotes. |
| `fetch_espn_odds.py` | Rebuilds the two odds CSVs from ESPN. The committed CSVs are the snapshot the analysis used. |
| `analysis.py`, `RESULTS.txt` | Season, month, price band, delta-vs-market, opening-price and native lock-time tables. |

## Runs and price basis

| Season | Workflow run | Commit | Closing prices |
|---|---|---|---|
| 2025 | [40037210833](https://github.com/Dave356w/Dave356w/actions/runs/40037210833) | `91d4c3d` | ESPN BET (provider 58) via ESPN, 2,253 full / 1,411 F5. ESPN lists no DraftKings odds for 2025; F5 lines start late May. |
| 2026 | [40037204741](https://github.com/Dave356w/Dave356w/actions/runs/40037204741) | `2673565` | DraftKings (100) via ESPN, from the ledger from Jul 2 and fetched before then. Identical game-for-game to run [39857873566](https://github.com/Dave356w/Dave356w/actions/runs/39857873566) on `4c24ee2`. |

The two books are reported separately and never pooled.

## Findings

1. **Closing market.** Full-game excess over closing no-vig q is
   −1.5 ± 1.0 pp (2025) and +0.4 ± 1.0 pp (2026). ROI is −6.7% and −3.1%,
   against market-correct nulls of −4.2% and −4.0%. F5 is +1.4 ± 1.5 pp
   (2025) and +0.1 ± 1.2 pp (2026).
2. **April.** April is the weakest month in both seasons: −3.5 ± 2.6 pp
   (2025) and −6.0 ± 2.5 pp (2026). In 2025 April is not distinct from the
   other months, so a separate early-season effect is not confirmed. The
   +6.6 pp September 2026 did not recur (−1.9 pp in 2025).
3. **Price bands** (descriptive). About 26 season × market × band cells
   give two beyond 2 SE, both in 2025: full −174..−130 at −4.6 pp and
   F5 −249..−175 at +7.0 pp. Neither recurs in 2026. These do not support
   a band filter, and the dated v13 band snapshot in `CLAUDE.md` is not
   reproduced by v14 over full seasons.
4. **Delta beyond the close.** Given the closing price, the delta
   coefficient is about zero (−0.048 ± 0.034 and +0.033 ± 0.035 per 0.020
   xwOBA). Adding it worsens held-out log loss in both directions. Alone it
   predicts outcomes clearly; the information is already in the price.
5. **Opening price.** The full-game line moves toward the model from open
   to close (+0.040 ± 0.003 and +0.013 ± 0.002 logit per 0.020; CLV
   +0.88 and +0.30 pp). At the open the model still shows no excess
   (−0.6 and +0.7 pp) and no held-out log-loss gain. Replay deltas know
   final lineups that the open usually predates, so this movement is not
   executable evidence.
6. **Native lock-time check (pregame-locked).** 355 graded native rows,
   Sep 1–27 2026, mostly locked after lineups posted. Lock → close move
   per 0.020 is +0.0018 ± 0.0006, i.e. +0.02 ± 0.02 pp CLV. Most movement
   precedes the lock (open → lock +0.0076 ± 0.0033). By lock time the
   market has largely priced what the model sees. One month, several tags.
7. **August 2026 closing hold.** ESPN's own DraftKings close for all 370
   priced games from Aug 4 to Aug 31 carries about 1.7% hold, against
   4.6% elsewhere. Re-fetching ESPN returns identical prices, so ESPN's
   data is the source. Treat August 2026 ROI as not comparable to other
   months. The cause is not established.

## Open questions and next measurements

- Whether any edge exists between the open and lineup announcements. This
  needs a forward model snapshot at the opening line using projected
  lineups, recorded under a separate registration, alongside the existing
  lock snapshot.
- An early-season effect: monitor an early-season segment in native rows
  rather than fit to two replay Aprils.
- F5: adding the delta slightly improves held-out log loss (by 0.0003 to
  0.0007 against the close or open), each within about 1 SE. This warrants
  native monitoring, not a rule.
