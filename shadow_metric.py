"""Paired wOBA shadow arm: what the SAME slate would have leaned on wOBA.

WHY THIS EXISTS
---------------
This arm used to run xwOBA against a wOBA primary. v11 reverted the primary to
xwOBA, so the arm swapped sides and now runs wOBA. Its purpose is unchanged and
so is its argument: the metric question cannot be settled by comparing the two
ERAS, only by running both metrics over the SAME games.

That is not a stale caution -- it is what the arm measured. Over its first six
paired slates (68 graded, both arms decided) the two metrics correlate +0.91 on
net, flip 9 of 77 leans, and separate by d_corr +0.008 with a 95% CI of
[-0.108, +0.128]. The sequential records over the same period looked far more
decisive than that (wOBA 29-39 against xwOBA 34-34 on identical games), which is
exactly the illusion pairing exists to remove. The revert to xwOBA was an
operator decision taken with that CI on the table, not a finding that xwOBA won.

So the arm keeps running, pointed the other way, and the question keeps
accumulating a paired answer. If wOBA is genuinely better, this is the surface
that will say so -- se ~0.045 at ~9 more slates, 80% power on a 0.09 gap at ~18.
If it is not, that is worth knowing too, and neither answer comes from the
win-loss line on the front page.

WHAT THE ARM ALREADY PAID FOR
-----------------------------
The `xwoba` Savant selection name is on the PRIMARY build's critical path under
v11. It is there safely only because this arm resolved it against the live
endpoint first: Actions run 31435698461 logged `shadow: rate column 'xwoba'
resolved on 20/20 players` and printed a distinct league baseline (xwOBA
0.31548 against the primary's wOBA 0.31628 on the same slate and leaderboard),
proving the column exists and is not a relabelled copy of `woba`. Verifying an
unproven column on a shadow arm before it can ever cost a pregame row is the
posture; `woba` now sits on this side with five weeks of production behind it.

WHAT IT DOES NOT DO
-------------------
Nothing here touches a published lean, `index.html`, `MODEL_TAG`, or the
ledger's record. It writes one extra dump per slate and stops. It also does not
share the primary build's Savant cache: it fetches under its own namespace
(`STATCAST_CACHE_NS`) because it requests a different column, and reusing a
cache written under another selection set is exactly how the requested rate
comes back missing. One extra leaderboard request per slate is the whole cost.

The pairing that matters is same games / same players / same slate date, not
one HTTP response -- both arms read the same season-to-date leaderboard on the
same morning.

FAILURE POSTURE
---------------
Exits 0 on any failure unless `--strict`. The daily build commits pregame
snapshots and the ledger only accepts rows whose snapshot predates first pitch,
so a shadow fault must never be able to fail the job that produces them. Run it
as its own workflow step with `continue-on-error: true`, after the primary
build has already written its dump.

VERIFIED IN PRODUCTION (2026-08-10)
----------------------------------
This section used to say the Savant selection name `xwoba` was an assumption:
`baseballsavant.mlb.com` is blocked from the development environment and from
CI, so it had never been confirmed against the live endpoint, and the first
run's log line was nominated as the verification. It has now run. Actions run
31435698461 logged `shadow: rate column 'xwoba' resolved on 20/20 players`, and
the same job printed a distinct league baseline for the shadow arm (xwOBA
0.31548 against the primary's wOBA 0.31628 on the same slate and the same
leaderboard), so the column resolved and it is not a relabelled copy of `woba`.
The check stays in the code -- it is cheap and it is what would catch Savant
renaming the column -- but it is no longer an open question.

Note what `_coverage` actually measures: the rate on the SLATE's pitcher rows,
which is 20-30 players, not the ~800-row leaderboard. That is enough to catch a
wrong selection name (a missing column resolves on zero) and it is not a
coverage statistic for the leaderboard. Do not read it as one.

Resolving the name is not a reason to add the column to the primary
STATCAST_SELECTIONS. The arm is a forward test; the primary build has no use
for a second rate.

USAGE
-----
    python shadow_metric.py              # write the shadow dump, never fail
    python shadow_metric.py --strict     # propagate errors (CI smoke test)
    python shadow_metric.py --dry-run    # patch and report, no fetch
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import build_site as bs

# The Savant column this arm swaps in, and the label/tag its rows carry. The tag
# is deliberately OUTSIDE every family map in build_site: a shadow row shares a
# record line and a delta scale with nothing, and must never be pooled into one.
SHADOW_SOURCE_COL = "woba"
SHADOW_LABEL = "wOBA"
SHADOW_TAG = "shadow_woba+plat_consol_v1"

# The dump suffix names the metric inside it, so the six xwOBA-arm dumps
# written before v11 stay readable as what they are and the wOBA-arm dumps
# written after it are never confused for them. shadow_report reads the metric
# off each dump's `model_metric` column rather than off the filename, but the
# filenames should not lie either.
SHADOW_SUFFIX = "woba"

# The dump is `shadow_<date>_<suffix>.csv`, NOT `leans_<date>_something`. grade_leans
# ingests `leans_*_xw.csv`, `leans_*_split.csv` and `leans_*_woba.csv`, and
# `leans_*_xw.csv` matches any leans-prefixed name ending `_xw.csv` -- so a
# suffix like `leans_<date>_shadow_xw.csv` would be silently ledgered as a real
# pending row. Escaping that on the PREFIX makes the collision impossible
# instead of one character away from happening, and
# test_shadow_dump_is_not_ingestable asserts it against grade_leans' own globs
# rather than against a copy of them.
SHADOW_PREFIX = "shadow"


def patch():
    """Repoint build_site's metric constants at the shadow rate.

    Every one of these is read at CALL time except the two that are built at
    import time from MODEL_RATE_SOURCE_COL -- STATCAST_SELECTIONS and the cache
    namespace -- which is exactly why they have to be set here too. Patching the
    source column alone would request the PRIMARY's rate, read the primary's
    cache, and then look for a column that was never fetched, yielding an
    all-NaN rate that still writes a plausible-looking dump. That silent mode is
    the one this function exists to make impossible.

    The selection swap reads the primary column off `bs` BEFORE overwriting it,
    rather than naming it as a literal. When this arm ran xwOBA against a wOBA
    primary the literal `"woba"` was correct and invisible; v11 swapped which
    metric is on which side, and a literal would have quietly stopped matching
    and left the primary's rate in the selection set. Deriving it means the two
    arms can swap again without touching this function.

    MODEL_RATE_INTERNAL_COL is deliberately NOT patched: the internal/dump
    schema name is "xwOBA" for both metrics (it is a compatibility schema, not
    a statement about which statistic is inside), so leaving it alone is what
    keeps the shadow dump readable by the same tooling.
    """
    primary_col = bs.MODEL_RATE_SOURCE_COL
    bs.MODEL_RATE_SOURCE_COL = SHADOW_SOURCE_COL
    bs.MODEL_RATE_LABEL = SHADOW_LABEL
    bs.MODEL_TAG = SHADOW_TAG
    bs.STATCAST_SELECTIONS = [
        SHADOW_SOURCE_COL if c == primary_col else c
        for c in bs.STATCAST_SELECTIONS
    ]
    # Keyed to the selection set, so it MUST differ from the primary's.
    bs.STATCAST_CACHE_NS = f"custom_{SHADOW_SOURCE_COL}_v1"
    return {
        "source_col": bs.MODEL_RATE_SOURCE_COL,
        "label": bs.MODEL_RATE_LABEL,
        "tag": bs.MODEL_TAG,
        "selections": list(bs.STATCAST_SELECTIONS),
        "cache_ns": bs.STATCAST_CACHE_NS,
        "internal_col": bs.MODEL_RATE_INTERNAL_COL,
    }


def _coverage(df):
    """(resolved, total) on the primary rate -- the log line that verifies the
    Savant selection name actually exists. A near-zero numerator means the
    column was requested under a wrong name and came back absent."""
    if df is None or df.empty or bs.MODEL_RATE_INTERNAL_COL not in df.columns:
        return 0, (0 if df is None else len(df))
    v = pd.to_numeric(df[bs.MODEL_RATE_INTERNAL_COL], errors="coerce")
    return int(v.notna().sum()), int(len(v))


def run(dry_run=False):
    cfg = patch()
    bs.log(f"shadow: metric -> {cfg['label']} (Savant column '{cfg['source_col']}'), "
           f"tag {cfg['tag']}, cache ns {cfg['cache_ns']}")
    if dry_run:
        bs.log(f"shadow: dry run, selections={cfg['selections']}")
        return 0

    data = bs.fetch_all(bs.SLATE_DATE)
    if data.get("empty"):
        bs.log("shadow: no games on this slate, nothing to write")
        return 0

    matchup_df, pitcher_rows_df, _ = bs.build_xwoba_matchup(
        data["pitchers_df"], data["league_baseline"])
    matchup_df = bs.apply_pitching_plans(
        matchup_df, data.get("pitching_plans"), data["league_baseline"])

    ok, tot = _coverage(pitcher_rows_df)
    bs.log(f"shadow: rate column '{cfg['source_col']}' resolved on {ok}/{tot} players")
    if tot and ok == 0:
        # Loud, and still not fatal: the primary build has already written its
        # dump by the time this runs, and a wrong selection name must cost a log
        # line rather than a slate.
        bs.log("shadow: ABORT -- rate resolved on zero players; the Savant "
               "selection name is wrong. Primary build is unaffected.")
        return 0

    if matchup_df is None or matchup_df.empty:
        bs.log("shadow: no matchups built (probables not posted)")
        return 0

    # Same provenance columns the primary dump carries, so the shadow rows are
    # readable by the same tooling and carry their own honest lock status.
    snapshot_utc = bs.datetime.now(bs.UTC).isoformat()
    matchup_df["model_tag"] = cfg["tag"]
    matchup_df["model_metric"] = cfg["label"]
    matchup_df["snapshot_utc"] = snapshot_utc
    if "game_datetime_utc" in matchup_df.columns:
        matchup_df["scheduled_start_utc"] = matchup_df["game_datetime_utc"]

    path = f"{bs.DATA_DIR}/{SHADOW_PREFIX}_{bs.SLATE_DATE}_{SHADOW_SUFFIX}.csv"
    matchup_df.to_csv(path, index=False)
    bs.log(f"shadow: wrote {path} ({len(matchup_df)} rows)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    strict = "--strict" in argv
    try:
        return run(dry_run="--dry-run" in argv)
    except Exception as e:  # noqa: BLE001
        bs.log(f"shadow: FAILED ({e!r})")
        if strict:
            raise
        return 0


if __name__ == "__main__":
    sys.exit(main())
