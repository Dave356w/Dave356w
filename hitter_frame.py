"""Persist the per-hitter inputs the lineup composite is built from.

`build_site.aggregate_lineup` shrinks each hitter's rate by his own PA, weights
the nine by expected plate appearances per batting-order slot, and writes ONE
number -- `opp_xwOBA_neutral`. The per-hitter vector it composited is discarded,
and `.savant_cache/` being gitignored and slate-keyed means a past slate's
cannot be rebuilt. So the lineup term can only ever be scored the way
`ledger_report.txt` scores it now: nine shrunk rates averaged into one number,
against one team's whole-game wOBA.

That measurement has a hard ceiling. Over the v12 family the composite's spread
is sd 0.0073 against a single-game actual of sd 0.0911, so even an exactly
correct composite could only correlate ~0.08 with the outcome -- and the
observed -0.066 [-0.146, +0.014] already excludes that ceiling. What it cannot
distinguish is WHERE the fault is: a per-hitter rate that carries no
information, or an aggregation that destroys one that does. Those are the two
hypotheses, they imply opposite fixes, and no team-level statistic separates
them.

This module stores the vector so they can be separated. Joined to per-plate-
appearance outcomes (`lineup_window_collect.py` already emits `batter_id` per
PA), each hitter's predicted rate is scored against his OWN realised results:
~1,100 PA rows a slate instead of ~30 side-games, and a correlation SE of
0.0066 at 22,758 rows against 0.0410 at 598.

FORWARD ONLY, and that is not a limitation to work around. Reconstructing a
past slate's per-hitter frame needs that slate's Savant leaderboard, which is
exactly the lookahead `.savant_cache/` exists to forbid. The first row this can
ever hold is tonight's.

WHAT IS STORED IS WHAT WAS CONSUMED, never a re-derivation. The records are
captured inside `aggregate_lineup` at the point the neutral composite is
formed -- after shrinkage, after the Savant-backfill substitution, and BEFORE
the platoon offset, because `opp_xwOBA_neutral` is the value the lineup
component is scored on. Recomputing the same quantity here would be a second
home for it, and the two would drift the way v10 math drifted under a v9 tag.

The `hitters_` prefix is a decision, not a name. `grade_leans` globs
`leans_*_xw.csv`, `leans_*_split.csv` and `leans_*_woba.csv`; a
suffix-based name would have been ingested as a real pending row. Same
reasoning as `SHADOW_PREFIX`, and a test pins it against the grader's own globs
rather than a copy of them.

Diagnostic only. Nothing here reaches a lean, a delta or a grade: the frame is
written after the primary dumps, by a caller that swallows its failures, so a
fault in this module costs a diagnostic file and never a slate.
"""
import os

import numpy as np
import pandas as pd

# Never `leans_`. See the module docstring.
PREFIX = "hitters"

# One row per hitter per lineup. `xwoba_shrunk` is the value the composite
# consumed; `xwoba_raw` is what it was before shrinkage, kept so a later K can
# be evaluated without a rebuild -- the same reason `expected_sp_ip_raw_*`
# exists, and the same hazard it avoids: a fit that reads its own output has no
# fixed point worth having.
COLUMNS = [
    "game_pk", "faced_pitcher", "pitcher_side", "batting_side",
    "player_id", "batting_order", "PA",
    "xwoba_raw", "xwoba_shrunk", "slot_weight", "savant_backfill",
    "model_tag", "model_metric", "snapshot_utc",
]


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def records(g, vals, w, game_pk, faced_pitcher, rate_col, backfill_col):
    """Per-hitter rows for one lineup, from the vectors that built its composite.

    `g` is the hitter group, `vals` the shrunk rate vector as composited, and
    `w` the slot-PA weight vector (or None when `lineup_weight` found no usable
    batting order, in which case `wmean` took an equal mean and the stored
    weight is null rather than a fabricated 1.0).

    `vals` and `w` are positionally reindexed by their producers, so `g` is too
    before zipping -- pairing a positional vector against a grouped frame's
    original index is how a hitter's rate would silently land on another
    hitter's row.
    """
    gg = g.reset_index(drop=True)
    vv = pd.to_numeric(pd.Series(vals).reset_index(drop=True), errors="coerce")
    ww = (pd.to_numeric(pd.Series(w).reset_index(drop=True), errors="coerce")
          if w is not None else None)
    raw = (pd.to_numeric(gg[rate_col], errors="coerce")
           if rate_col in gg.columns else pd.Series(np.nan, index=gg.index))
    bf = (pd.Series(gg[backfill_col]).fillna(False).astype(bool)
          if backfill_col in gg.columns
          else pd.Series(False, index=gg.index))
    out = []
    for i in range(len(gg)):
        out.append({
            "game_pk": game_pk,
            "faced_pitcher": faced_pitcher,
            "pitcher_side": gg.get("pitcher_side", pd.Series(dtype=object)).iloc[i]
            if "pitcher_side" in gg.columns else None,
            "batting_side": gg.get("batting_side", pd.Series(dtype=object)).iloc[i]
            if "batting_side" in gg.columns else None,
            "player_id": gg["player_id"].iloc[i] if "player_id" in gg.columns else None,
            "batting_order": (gg["batting_order"].iloc[i]
                              if "batting_order" in gg.columns else None),
            "PA": _f(gg["PA"].iloc[i]) if "PA" in gg.columns else None,
            "xwoba_raw": _f(raw.iloc[i]),
            "xwoba_shrunk": _f(vv.iloc[i]) if i < len(vv) else None,
            "slot_weight": (_f(ww.iloc[i]) if ww is not None and i < len(ww) else None),
            "savant_backfill": bool(bf.iloc[i]),
        })
    return out


def frame(rows, model_tag=None, model_metric=None, snapshot_utc=None):
    """Stamped DataFrame in the declared column order, or an empty one.

    Provenance is stamped here rather than at each record so a row cannot carry
    a tag that disagrees with the build that produced it -- the mismatch that
    put v10 math under a v9 stamp, one artifact out. `model_metric` is what
    tells a later reader which statistic these rates ARE; a key name never does,
    which is the standing rule for every dump in this repo.
    """
    df = pd.DataFrame(list(rows or []))
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["model_tag"] = model_tag
    df["model_metric"] = model_metric
    df["snapshot_utc"] = snapshot_utc
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS]


def write(rows, path, model_tag=None, model_metric=None, snapshot_utc=None):
    """Write the frame; return the number of rows written (0 when empty).

    An empty frame writes NOTHING rather than a header-only file: a slate whose
    lineups never posted has no per-hitter truth to record, and an empty file
    would read as one that did and found nine missing bats.
    """
    df = frame(rows, model_tag, model_metric, snapshot_utc)
    if df.empty:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)
