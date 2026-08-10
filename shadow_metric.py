"""Paired xwOBA shadow arm: what the SAME slate would have leaned on xwOBA.

WHY THIS EXISTS
---------------
The wOBA lineage sits behind its controls (41-44 against always-home 51-34)
while the xwOBA lineage sat ahead of them, and the delta's correlation with the
realised differential fell from +0.273 (v9/v10, n=97) to +0.038 (v4+v5, n=65).
That is directionally consistent with the metric switch having cost something,
and it is NOT evidence of it, for two reasons the ledger cannot get around:

  * Five things changed in six days -- metric, platoon prior, K 100->400,
    personal priors, abstention. Only wOBA v1+v2 isolates the metric, n=16,
    CI on its correlation [-0.816, +0.288].
  * The eras were different games. Over the xwOBA rows always-home ran .515 and
    always-chalk .619; over the wOBA rows .600 and .565. A sequential
    comparison is measuring the schedule as much as the model.

Unpaired, se(d_corr) is ~0.142 and the xwOBA-minus-wOBA CI is [-0.010, +0.560]
-- it does not separate, and on this trajectory it never will. Run both metrics
over the SAME games and the metrics correlate ~0.9, so se falls to ~0.045: a
true gap of 0.09 resolves inside a month of slates. That is the entire argument
for pairing, and it is why this arm exists rather than a revert.

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

UNVERIFIED, ON PURPOSE, AND HOW TO VERIFY
-----------------------------------------
`baseballsavant.mlb.com` is blocked from the development environment, so the
Savant selection name `xwoba` could not be confirmed against the live endpoint.
It is the documented column and matches the `xba`/`xslg` convention already in
STATCAST_SELECTIONS, but it is an assumption until a real run proves it. The
first run's log line `shadow: rate column 'xwoba' resolved on N/M players` IS
the verification -- a low N means the selection name is wrong, and because this
arm has its own cache namespace and its own dump, being wrong costs a log line
and nothing else. Do not "fix" it by adding the column to the primary
STATCAST_SELECTIONS until that log says it resolved.

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
SHADOW_SOURCE_COL = "xwoba"
SHADOW_LABEL = "xwOBA"
SHADOW_TAG = "shadow_xw+plat_consol_v1"

# The dump is `shadow_<date>_xw.csv`, NOT `leans_<date>_something`. grade_leans
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
    source column alone would request `woba`, read the primary's cache, and then
    look for an `xwoba` column that was never fetched, yielding an all-NaN rate
    that still writes a plausible-looking dump. That silent mode is the one this
    function exists to make impossible.

    MODEL_RATE_INTERNAL_COL is deliberately NOT patched: the internal/dump
    schema name is already "xwOBA" for both metrics (it is a compatibility
    schema, not a statement about which statistic is inside), so leaving it
    alone is what keeps the shadow dump readable by the same tooling.
    """
    bs.MODEL_RATE_SOURCE_COL = SHADOW_SOURCE_COL
    bs.MODEL_RATE_LABEL = SHADOW_LABEL
    bs.MODEL_TAG = SHADOW_TAG
    bs.STATCAST_SELECTIONS = [
        c if c != "woba" else SHADOW_SOURCE_COL for c in bs.STATCAST_SELECTIONS
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

    path = f"{bs.DATA_DIR}/{SHADOW_PREFIX}_{bs.SLATE_DATE}_xw.csv"
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
