# Kalshi live test — guide for the 2027 season

**Written 2026-10-01.** A plan, not a registration and not a result. Every
number below is a dated snapshot of `data/ledger_report.txt` and
`data/paper_kalshi/report.txt` on `main` at that date (or a one-off scratch
calculation over the same ledger, labelled as such); re-read the live reports
before acting on any of them. Nothing here changes the model, a ledger
decision, or an existing registration. `paper_kalshi.py` stays read-only
(see `docs/kalshi_paper.md`); live orders are placed by hand or by separate,
separately reviewed code.

## 1. The rule

| Item | Choice |
|---|---|
| Model | The shipped model's own pregame lean (`MODEL_TAG`; `xw+starter_velo_v14` at writing). If the tag changes mid-season, start a new count. |
| Which games | **Every lean.** No price-band, delta-band, or favourite/dog filter. Skip abstentions, doubleheaders, and anything the paper script skips. |
| Sizing | **floor($1 ÷ price) contracts.** 1 contract above $0.50, 2 at $0.34–$0.50, 3 at $0.26–$0.33. |
| Entry price | Fee-inclusive Kalshi break-even **≤ saved sportsbook break-even** (0 pp threshold; the paper script currently allows −1 pp). |
| Order type | Resting **limit order** at or below the sportsbook break-even first; if unfilled near first pitch, take the ask only if it still qualifies. |
| Contrast to watch | Plus-money dog leans vs. favourite leans, reported separately. A monitored contrast, **not** a filter. |
| Review point | ~150 native live leans (roughly the first month), then again at ~300. |

Stake per lean averages about **$0.67** (favourites risk their price; near-even
dogs risk ~$0.90–1.00). Over 150 leans that is roughly **$100 staked**; a bad
run costs on the order of $10–20.

## 2. Why this rule (evidence at 2026-10-01)

**Every lean, not a selection.** The most realistic evidence is the
prospective registered test since 2026-08-29 at *saved pregame* prices: the
plain lean went **207–133, +13.65u, +4.0% ROI (n=340)** against always-chalk
−1.1% and the hybrid fade rule +0.1% (its 25 fades: 11–14, −27.7%) on the same
rows. Expect live ROI nearer that +4% than the larger closing-price backtest.

**Bands did not hold up out of sample.** Scratch split of the published
(mostly re-decided, hindsight) record, flat 1u at closing MLs:

| Filter | Aug 15–Sep 5 | Sep 6–27 |
|---|---|---|
| All leans | +12.2% (n=292) | +8.7% (n=286) |
| −174/−130 (best large band, 1st half) | +18.4% | +9.2% |
| −129/−100 | +5.2% | −1.5% |
| Smallest third of \|delta\| | +20.7% | −0.1% |
| Largest third of \|delta\| | +8.1% | +17.5% |

The first-half winner barely beat all leans afterwards and the delta groups
flipped. The price-calibration shadow agrees: its selected games returned
+6.5% vs +12.1% for all leans (reconstructed basis). Every closing-price
band in the report is positive (pooled +7.7 ± 2.0 pp over no-vig q, 366–212),
which supports "every lean" rather than any one band.

**Why dogs are only a contrast.** Registered arm 2 (every plus-money dog
lean) is **31–26, +9.81u, +17.2% ROI, z=+1.52 (n=57)** forward, and dogs held
up in both halves above (+20.9% → +22.9%, n=61/38). Still indistinguishable
from zero (the report's gate is ~1,200 bets), and dropping favourites would
give up most volume and profit.

**Native v14 is the open question.** The model you will actually run was
**28–22, −0.8 ± 6.9 pp vs close, −3.32u (n=50, Sep 24–27)** in the regular
season, plus 8–0 in the 2026 Wild Card round. Native v13 was 54–29, +8.3 ±
5.2 pp (n=83). Both are too small to settle anything; the live test exists to
grow this number.

**Sizing barely matters at $1.** Scratch simulation, closing prices as the
contract-price proxy, Kalshi fee rounded up per order:

| Games | floor($1/P) ROI | 1 contract/lean ROI |
|---|---|---|
| Published v14 record (n=578, hindsight) | +6.6% to +12.0% | +5.4% to +10.8% |
| Native v13 (n=83) | +2.3% to +8.2% | +5.3% to +11.7% |
| Native v14 (n=50) | −6.1% to −10.6% | −4.0% to −9.6% |

Ranges run from posted break-even with a 1× fee (worst) to no-vig price with a
0.5× fee (best). The gap between those two prices (~3.5 ROI points) is larger
than the fee gap (~1.7), so **entry price is the biggest controllable lever**:
each point of probability saved is worth ~2 points of ROI near even money.

## 3. Before Opening Day

- [ ] Confirm `MODEL_TAG` and that the paper job (`kalshi-paper.yml`, the
      `build.yml` paper step) is running and committing `data/paper_kalshi/`.
- [ ] Align the paper simulator with the live rule: floor($1/P) sizing and a
      0 pp threshold (today: fixed `--quantity 10`, `--min-savings-pp -1.0`).
      A small, reviewed PR; historical rows keep their recorded rule.
- [ ] Optional: log the best bid/ask at each paper quote so resting-order fill
      rates can be estimated (hourly snapshots cannot simulate maker fills).
- [ ] Verify Kalshi's current MLB fee: series `fee_type` / `fee_multiplier`
      (0.5× quadratic-with-maker-fees at writing) and maker vs taker rates.
- [ ] Record this rule, with its start date, in `REGISTRATIONS.md` before the
      first live order. Do not move the start date after seeing results.
- [ ] Fund a small, separate balance; decide a hard maximum loss for the test.

## 4. Daily routine

1. After the pregame build, read the day's leans on the site (or
   `data/leans_<date>_xw.csv`) and the sportsbook ML saved with them.
2. For each lean: compute the sportsbook break-even; place a limit order on the
   Kalshi YES side of the leaned team at or below it, sized floor($1/P).
3. Shortly before first pitch, if a resting order has not filled, take the ask
   only if fee-inclusive break-even still ≤ sportsbook break-even; otherwise
   record a no-fill.
4. Log every lean, filled or not (template below). No-fills and skips are
   data: they measure how often the rule is actually executable.

| Field | Example |
|---|---|
| date, game_pk, lean | 2027-04-02, 778899, NYY |
| sportsbook ML / BE | −140 / 0.583 |
| paper quote (ask, BE, time) | 0.57 / 0.579 / 16:37Z |
| order type, limit price | limit, 0.57 |
| filled? contracts, avg price, fee | yes, 1, 0.57, $0.01 |
| fill time | 17:05Z |
| result, P&L | W, +$0.42 |

The paper-vs-live gap (price and fee on the same game) is the measured
slippage.

## 5. How to judge it

Report at each review, on the **same games**:

- Record, staked $, P&L, ROI.
- Excess over the closing no-vig q of the leaned side (the model's edge) with
  its ± SE — the primary number, since ROI also depends on entry price.
- Live fill price vs paper quote vs closing price: execution cost.
- Fill rate (limit vs taker vs no-fill).
- Plus-money dog leans vs favourite leans, as the one named contrast.
- Always-chalk on the same games as the baseline.

| Outcome at review | Action |
|---|---|
| Excess over q positive with interval clear of zero **and** ROI after fees/slippage positive (~150–300 leans) | Consider a larger stake under the same rule; re-register the size change. |
| Excess positive but interval spans zero | Keep $1 stakes; keep collecting. |
| Excess ≤ 0 after ~150 leans, or live fills consistently worse than paper | Stop live orders; keep the paper job; investigate the model or execution. |
| Model tag changes | Close the count for the old tag; start a new one. |

Do not add a band or delta filter mid-season because a slice looks good;
write it down as a new registration with its own start date and judge it
only on games after that date.

## 6. Known limits

- The backtest is scored at **closing** prices; live entry is earlier and
  includes the Kalshi spread.
- Most of the published record was re-decided after the fact (537 of 588 rows
  at writing). Treat it as context, not a forecast.
- Games within a series share teams and starters; read every SE as a floor.
- The paper script does not handle postponed/voided games after a fill — such
  positions stay open there. Settle them by hand in the live log.
- `paper_kalshi.py` never places orders and should stay that way; any
  automation of live orders needs its own code, limits, and kill switch.
