# Kalshi MLB read-only paper-execution shadow

This is a separate, **zero-authentication and zero-order-submission** shadow of
the **shipped model** — whatever `MODEL_TAG` `build_site.py` currently writes
(`xw+starter_velo_v14` as of 2026-09-24; it was `xw+starter_blend_v13` when this
shadow was first written). Rows under any other tag are skipped as
`model_tag_not_current` and never filled; every row records its `model_tag`, so
fills from different model versions are never pooled silently. `--model-tag`
overrides the default. It consumes the repo's **saved** `data/leans_<ET-date>_xw.csv`
rows. It never changes `build_site.py`, model decisions, `mlb_lean_ledger.csv`, or
an existing registration. It never calls a Kalshi trading endpoint.

For the planned 2027 live test (rule, sizing, daily routine, review points),
see `docs/kalshi_live_test_guide.md`.

## Run

```sh
pip install requests
python paper_kalshi.py --slate-date 2026-09-22
python -m pytest tests/test_paper_kalshi.py -q  # offline fixtures
```

The existing `build.yml` first commits its production pregame ledger and
uploads the Pages artifact, then runs the paper capture in a **separate,
non-blocking job** against the freshly committed `main` branch. The public
`/kalshi-paper.txt` contains the latest diagnostic available at site-build time
and may lag the GitHub paper ledger by one build. A PR that changes the paper
code triggers a public API smoke run, saves its diagnostics as a workflow
artifact, and never writes to `main`. The separate `kalshi-paper.yml` provides a manual trigger and
an additional best-effort hourly snapshot at :37 ET during MLB daytime/evening.
Its schedule gate skips runs without a game in the 15–360-minute pregame window.
Scheduled/manual writer runs use their own `kalshi-paper` concurrency group,
**not** `site-build`: GitHub keeps one pending run per group and a newly queued
run replaces it, so an hourly paper run there could evict a pending production
build. The signed commit path already pins the branch tip and refuses if another
paper run wrote the same files, so the two paper writers fail closed rather
than double-filling. Read-only PR smoke tests use a separate group and never
commit data.
Paper-only commits use the signed API with `--no-fallback`, so a conflict
cannot force an unsigned push or jeopardize the model's pregame ledger. GitHub scheduled events can be skipped or late:
missing a first-pitch cutoff is a **skip**, never a retroactive fill. This is
observational infrastructure, not a continuous low-latency trading engine.

## Observations and simulated fills

* Joins Kalshi `KXMLBGAME` by **ET game date + away/home ticker codes +
  optional event start time + selected team suffix**. Unknown team codes,
  rescheduled start changes, ambiguous doubleheaders, absent live Preview
  status, stale model snapshots, market outages and missing sportsbook quotes
  produce explicit skips. The quote timestamp is local receipt time; the public
  REST order book does not provide a guaranteed per-quote exchange timestamp.
* Uses the public `orderbook_fp.no_dollars` bid ladder to reconstruct executable
  **YES asks**, sweeping visible depth for a default 10 whole contracts. No
  last-trade price, midpoint, imaginary maker fill, or guaranteed execution.
* Pulls `fee_type` / `fee_multiplier` from the **live public series API**;
  estimates quadratic **taker** fees separately at each swept price level,
  rounded upward to cents. Missing/unrecognized fee metadata means **no fill**.
  Market-specific fee overrides are not certified by this approximation.
  **Unverified:** how `fee_multiplier` (0.5 on the live MLB series) scales the
  7% taker curve. Fills use the multiplier; every priced row also stores the
  standard 1x fee (`estimated_fee_1x`, `kalshi_be_1x`, `savings_pp_1x`) and the
  report counts how many fills keep a non-negative saving at 1x. At a 55¢ ask
  the two differ by ~0.9 pp, comparable to the savings being measured, so
  confirm the rule against Kalshi's fee schedule or a real fill before reading
  the savings as settled.
* By default, records a paper fill when the **saved** side-specific sportsbook
  break-even minus the fee-inclusive Kalshi break-even is **at least -1.0
  percentage points**, inclusive. This allows up to 1 percentage point worse
  execution after fees, rather than a relative 1% difference. The CLI
  `--min-savings-pp` accepts -1 through 10; a value of 0 requires no worsening,
  and 0.5 restores the earlier improvement requirement. Both automated paper
  jobs use the CLI default. This is an execution-cost screen only: a model lean
  and sportsbook disagreement do **not** establish positive expected value.
  The new rule applies prospectively; historical skips and positions are not
  retroactively changed. New rows append `min_savings_pp` to record the applied
  threshold; older rows retain a blank value. The report states the current
  threshold and keeps the historical decisions in its all-days aggregates.
* `data/paper_kalshi/observations.csv` records **all** observations and skips;
  `positions.csv` holds at most one hypothetical position per game;
  `report.txt` summarizes diagnostic skips, open positions and hypothetical
  settlement results. No sensitive credentials are stored.
* **Same-game baseline.** The fill filter (Kalshi cheaper than a saved
  sportsbook price that can be up to 180 minutes old) may preferentially select
  games whose line moved *against* the lean after the snapshot. The report
  therefore grades, by the MLB ledger's final score, one quote per game for
  *all matched leans*, for *fills only*, and for *matched but not filled*, with
  P&L and mean fee-inclusive break-even for each. A filter worth keeping should
  beat the all-leans line on the same games; one that trails it is selecting
  adverse moves. These rows are prospective paper observations, separate from
  reconstructed history.
* At each run, pending positions settle only when the **Kalshi market itself
  reports `settled` with `yes` or `no`** AND an existing **graded MLB ledger**
  confirms the same winner. Disagreements go to `needs_review`. Postponements,
  cancellations, fair-price settlements and ambiguous outcomes are left open
  for manual inspection; **no score-only settlement** is applied.

### Important limitations

This measures observed indicative taker execution and source availability.
A public order-book snapshot does not prove that the entire simulated quantity
would fill at those prices: latency, queue depletion, changing odds and fee
rounding on actual executions may differ. The sportsbook price is the repo's
saved pregame price, not a timestamp-matched live sportsbook quote, so savings
can also reflect **market movement**. Do not interpret this report as calibrated
V13 edge, actual traded profitability or a promise of live execution.

The full Kalshi MLB event ticker may contain an originally scheduled event
start date/time; this shadow deliberately skips matches whose start conflicts
with the verified MLB game date/time rather than guessing. Credentials,
authenticated WebSockets and any POST endpoint are intentionally absent.
