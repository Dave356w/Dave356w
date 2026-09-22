# Kalshi MLB read-only paper-execution shadow

This is a separate, **zero-authentication and zero-order-submission** shadow of
`xw+starter_blend_v13`. It consumes the repo's **saved** `data/leans_<ET-date>_xw.csv`
rows. It never changes `build_site.py`, model decisions, `mlb_lean_ledger.csv`, or
an existing registration. It never calls a Kalshi trading endpoint.

## Run

```sh
pip install requests
python paper_kalshi.py --slate-date 2026-09-22
python -m pytest tests/test_paper_kalshi.py -q  # offline fixtures
```

The existing `build.yml` takes a paper snapshot after its pregame build, before
committing data. A PR that changes the paper code triggers a public API smoke
run, saves its diagnostics as a workflow artifact, and never writes to `main`. The separate `kalshi-paper.yml` provides a manual trigger and
an additional best-effort hourly snapshot at :37 ET during MLB daytime/evening.
Its schedule gate skips runs without a game in the 15–360-minute pregame window.
Scheduled/manual writer runs use the existing `site-build` concurrency group,
while read-only PR smoke tests use a separate group and never commit data. GitHub scheduled events can be skipped or late:
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
* By default, records a paper fill only if its fee-inclusive break-even
  improves on the **saved** side-specific sportsbook moneyline by at least
  0.5 percentage points. This is an execution-cost screen only: a V13 lean
  and sportsbook disagreement do **not** establish positive expected value.
* `data/paper_kalshi/observations.csv` records **all** observations and skips;
  `positions.csv` holds at most one hypothetical position per game;
  `report.txt` summarizes diagnostic skips, open positions and hypothetical
  settlement results. No sensitive credentials are stored.
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