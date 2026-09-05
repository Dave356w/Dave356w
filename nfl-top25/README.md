# NFL Top 25 by Position

A daily rebuild of the Yahoo NFL full-slate top-25 rankings for QB, RB, WR, TE
and DEF, published as a static web page. GitHub Actions runs the pipeline once
a day; GitHub Pages serves the result.

The pipeline is the Colab notebook `Yahoo_Top25_Position_Rankings_v1.ipynb`,
unchanged in substance — see [How this maps to the notebook](#how-this-maps-to-the-notebook).

## What it does

Each run:

1. Pulls the current Yahoo NFL player feed (salaries, FPPG, the slate schedule).
2. Builds market-implied projections from Underdog and Bovada player props —
   paired markets are de-vigged, yardage is fitted as Weibull and counts as
   Poisson, and the means are converted to Yahoo half-PPR scoring.
3. Cross-checks roles and availability against the published
   [nflverse](https://github.com/nflverse/nflverse-data) depth charts and weekly
   roster status, so injured-reserve and practice-squad players drop out and
   depth is read from a real depth chart rather than from salary order.
4. Applies the historical depth adjustment to fallback estimates only, then
   ranks every priced player by final projected fantasy points.
5. Writes JSON and CSV into `site/data/` and deploys `site/` to Pages.

Projection priority is: manual override → accepted market-implied mean → Yahoo
FPPG and salary prior. Only the fallback estimates carry the depth haircut.
Every row on the page shows which of the three produced it.

## Setup (one time)

1. **Enable Pages.** Settings → Pages → *Build and deployment* → Source:
   **GitHub Actions**. Nothing is published until this is set.
2. **Allow Actions to write.** Settings → Actions → General → *Workflow
   permissions* → **Read and write permissions**. The daily job commits each
   run's data back to `main` so the archive accumulates.
3. **Run it once.** Actions → *Daily rankings* → *Run workflow*. Until the
   first run finishes the page shows a "waiting for the first run" state
   rather than stale or fabricated numbers.

The site then lives at `https://<user>.github.io/<repo>/`.

No secrets or API keys are needed. Every feed the pipeline reads is public.

## Schedule

`.github/workflows/daily.yml` runs at **13:00 UTC** (09:00 ET) daily, and on
demand from the Actions tab. Change the `cron` line to move it. GitHub queues
scheduled runs under load, so the actual start time drifts by minutes.

The workflow also rebuilds on a push that touches the site or the pipeline, so
a UI change reaches Pages without waiting for the next slate.

## Layout

```
pipeline/notebook.py   the notebook's code, one module, cell banners intact
run_daily.py           entry point: run the pipeline, write site/data/*
site/index.html        the page (no build step, no dependencies)
site/data/latest.json  what the page reads
site/data/history/     one archived JSON + CSV per run date
tests/                 shape tests for the published JSON
```

## Running it locally

```bash
pip install -r requirements.txt
python run_daily.py --top-n 25
python -m http.server -d site 8000    # then open http://localhost:8000
```

`--positions QB,RB,WR` limits what gets published. The run needs outbound
network access to Yahoo, the two sportsbooks and the nflverse release CDN.

Tests do not need the network:

```bash
python -m unittest discover -s tests -v
```

## Failure behaviour

If any feed fails hard, `run_daily.py` writes **nothing** and exits non-zero.
The previously published page stays live rather than being replaced by a
half-built one, and the failure shows up as a red run in the Actions tab.

Softer failures degrade instead of stopping: an unreachable sportsbook falls
back to Yahoo FPPG and salary priors, and an unreachable nflverse release falls
back to the salary-order depth heuristic. Both are recorded in the run log,
which the page renders under *Run details*.

## How this maps to the notebook

The notebook executed every cell into a single namespace, and its functions
reach across cell boundaries for underscore-prefixed helpers as well as public
ones. `pipeline/notebook.py` therefore keeps all of that code in one module in
the original order, with a banner marking each cell, rather than splitting it
into packages — a split would need an explicit export list per cell and would
break silently the first time a private helper moved.

Two edits were made, both mechanical:

- the notebook's trailing `position_results = run_position_rankings(top_n=25)`
  is dropped, so importing the module defines functions without running a slate;
- the `google.colab.files.download` cell is dropped.

Everything else — the market engine, the correlation model, the showdown lineup
code — is carried over verbatim. The showdown/lineup half is unused by the daily
run but kept so the module stays a faithful copy of the notebook.

## Caveats

These are market-derived estimates, not predictions with a guarantee. The
Yahoo and sportsbook feeds are public but undocumented and can change shape
without notice. DEF has no dependable player-prop market and always uses the
Yahoo fallback.
