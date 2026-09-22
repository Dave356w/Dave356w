# V13 / Kalshi read-only paper monitor

This module is a separate prospective execution experiment. It **does not**
change V13, its existing decisions, historical ledger, price-calibration shadow,
or GitHub Pages matchup cards. It cannot place real orders: the client uses
public GET endpoints exclusively and accepts no trading API credentials.

## Architecture

- Existing build.yml produces the timestamped data/leans_DATE_xw.csv snapshot.
  The paper step runs **after** the main model, grading and grades-page steps,
  with continue-on-error, before the existing signed ledger commit.
- Separate kalshi_paper.yml rechecks the latest committed snapshot hourly at Eastern
  10:19–23:19 and again at 05:31 to reconcile settlements.
  Both workflows share the site-build concurrency lock.
- Public Kalshi series and markets endpoints provide KXMLBGAME markets, series
  fee multiplier, observed YES ask, top-of-book ask size and update time.
  MLB StatsAPI provides the live first-pitch status and schedule.
- The selected market is matched by **away and home team codes in order**,
  Eastern calendar date, event first-pitch clock (within 20 minutes), and the
  selected YES-outcome suffix. Zero or multiple matches means no paper fill.
- Each simulated purchase is one-per-MLB-game, at an OBSERVED ask for at most
  the available reported top ask quantity, not a guaranteed executable fill.
  We use the exchange's observed ask, **never** the display midpoint.
- Estimated standard taker fee = ceiling-to-cent of
  0.07 * quantity * ask * (1 - ask) * series fee multiplier.
  A missing series multiplier conservatively uses 1x and is labeled. The
  monitor accepts quadratic-with-maker-fees series for TAKING (not making),
  checks the event's fee overrides, and abstains if those cannot be verified.
  Actual per-order rounding can change realized charges.
- A prospective simulated trade requires a recorded V13 lean, saved-pregame
  sportsbook ML no older than 30 minutes, Kalshi quote updated in the last 15
  minutes, Preview status, model snapshot before first pitch and within six
  hours, and a live scheduled start at least 120 seconds away. No live order
  is ever sent; a threshold of **1 pp lower fee-inclusive break-even than the
  saved sportsbook** is the default cost-comparison condition, not evidence
  of model predictive edge.
- Default caps: 10 whole contracts per game, USD 20 hypothetical stake per
  game, USD 100 maximum unresolved paper exposure. Trades remain immutable
  except for settlement fields. Duplicate game IDs cannot paper fill twice.

## Running

    pip install -r requirements.txt
    python paper_kalshi.py --out-html public/paper.html
    python paper_kalshi.py --date YYYY-MM-DD --min-saved-pp 0.5
    python paper_kalshi.py --settle-only
    python -m pytest tests/test_kalshi_paper.py -q

After the branch is merged, the paper workflow may be launched manually
from GitHub Actions, or will run automatically on its schedule. The separate
[paper report](https://dave356w.github.io/Dave356w/paper.html) appears after
the next successful main site build. The separate paper-only workflow uploads
a downloadable Actions artifact immediately, even before the site republish.

## Outputs and verification

- data/kalshi_paper_trades.csv: prospective quote time, selected V13 side,
  saved sportsbook source and break-even, matching event/ticker, observed
  ask/quantity, fee source and estimated fee, cost savings in percentage
  points, simulated quantity and official Kalshi YES/NO settlement/P&L.
- data/kalshi_paper_audit.csv: all scanned model games, timestamp and exact
  reason for a simulated fill or abstention. Zero fills is not an error if
  every game fails the independently recorded preconditions.
- public/paper.html: separate descriptive paper portfolio page (not a live
  brokerage display). The dedicated workflow also uploads this report.
- tests/test_kalshi_paper.py: fixtures for V13 lean parity with the existing
  grader, ticker matching, first-pitch lock, duplicate protection, fees,
  stale prices, absence of top-ask depth, settlement and failed network reads.

Only official Kalshi YES/NO settlements grade open simulations. Unresolved
or nonbinary settlements remain open rather than being assigned fictional
win/loss outcomes. Since the repository is public, DO NOT commit real keys,
positions or personal information. This module uses no API secrets.

**Limitations:** GitHub scheduled jobs can be delayed or dropped. The separate
paper monitor is not a persistent low-latency worker. A public top-of-book
quote can change before any actual fill, and public best-ask size is not
full order-book depth. The sportsbook price is captured at the V13 snapshot,
not necessarily timestamp-identical to the Kalshi quote. Saved pp measures
the hypothetical fee-inclusive cost difference between the two recorded
observations; it is NOT model probability edge or realized trade profit.
