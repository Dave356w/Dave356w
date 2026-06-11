# mlb-close-archive

Automated MLB pregame close-price capture via GitHub Actions — append-only archive
for offline CLV grading (notebook v0.9.1).

## Setup

1. Create a new **private** repo named `mlb-close-archive` under your account.
2. Copy all files here to the new repo root  
   (`.github/workflows/capture.yml`, `capture.py`, `close_log/`, `status_log/`).
3. Add `THE_ODDS_API_KEY` to **Settings → Secrets → Actions**.
4. Trigger `workflow_dispatch` on a game day; verify a commit lands with  
   plausible rows and `credits_remaining` decremented by 2.
5. Let cron run 2–3 days; confirm each cluster has snapshots from T−90 to flip.

## File layout

```
mlb-close-archive/
├── .github/workflows/capture.yml
├── capture.py
├── close_log/          # one CSV per month: 2026-06.csv
└── status_log/         # status-flip feed (free, every tick)
```

## Schema

**`close_log/YYYY-MM.csv`** — one row per pregame event per 15-min tick:

| Column | Notes |
|---|---|
| `ts_utc` | Snapshot timestamp (ISO 8601) |
| `gamePk` | MLB Stats API game ID (join key) |
| `home` / `away` | Full official team names |
| `commence` | Scheduled first pitch (ISO Z) |
| `mlb_status` | MLB detailedState at capture time |
| `home_price` / `away_price` | Median ML across books (American) |
| `fair_home` | Median per-book devig implied prob |
| `book` | `median_N` where N = book count |
| `total_line` | Modal total line across books |
| `over_price` / `under_price` | Median prices at modal line |
| `ou_books_n` | Books offering the modal line |
| `credits_remaining` | Odds API quota after this call |

**`status_log/YYYY-MM.csv`** — every game, every tick (no Odds API spend):  
`ts_utc`, `gamePk`, `home`, `away`, `commence`, `mlb_status`.  
Used by the notebook to detect the status flip (pregame → live) and
discriminate `close_missed` vs `voided`.

## Notebook integration (v0.9.1, cell 4)

```python
CLOSE_SOURCE        = "archive"   # "archive" | "live"
CLOSE_ARCHIVE_BASE  = "https://raw.githubusercontent.com/dave356w/mlb-close-archive/main"
CLOSE_ARCHIVE_TOKEN = ""          # fine-grained PAT (contents:read) via Colab Secrets
```

## Credit budget

| Cron | `CAPTURE_LEAD_MIN` | Est. calls/day | Est. credits/month |
|---|---|---|---|
| `*/15` | 90 | ~21 | ~1,300 (fits 20K hobby tier) |
| `*/30` | 60 | ~10 | ~450 (fits 500 free tier) |

One call = 2 credits (h2h + totals, 1 region, all events in one response).
The gate (`is_pregame` + `in_window`) ensures zero credits during quiet hours.

## Design invariants

- **Capture is dumb, grading is smart.** This repo is pick-agnostic; it never
  sees your picks log.
- **The close = last pregame snapshot.** No clock math needed; delays and
  doubleheaders resolve by construction.
- **No silent failures.** Every bad outcome lands in `open` or `close_missed`,
  never in a wrong `closed` row.

## Failure modes

| Failure | Effect | Detection |
|---|---|---|
| Actions outage spanning full T−90 window | `close_missed` on that cluster | summary line > 0 |
| Cron jitter (≤15 min) | Close timestamp slightly earlier | harmless by design |
| Odds API quota exhausted | Empty odds rows → `close_missed` | `credits_remaining` column |
| Odds/Stats name mismatch | Event absent from archive → `close_missed` | add alias map entry if it fires |
| Private repo + expired PAT | Loader prints per-file error; rows stay `open` | visible at grade time |
| Suspended game resuming next day | Status flip then pregame again | last pregame snapshot before first flip is correct |
