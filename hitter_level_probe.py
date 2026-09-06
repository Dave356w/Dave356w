"""Score each hitter's predicted xwOBA against his OWN realised plate appearances.

The lineup component is scored today as nine shrunk rates averaged into one
number, against one team's whole-game wOBA. Over the v12 family that test has a
ceiling: the composite's spread is sd 0.0073 against a single-game actual of
sd 0.0911, so an exactly correct composite could only correlate ~0.08 -- and
the observed -0.066 [-0.146, +0.014] already excludes it. What the team-level
test CANNOT do is say where the fault is. Two hypotheses fit it equally:

  A  the per-hitter rate carries no information about the hitter's outcomes.
  B  the per-hitter rate is fine and the AGGREGATION destroys it -- the slot
     weights, or the log5 combination downstream.

They imply opposite fixes, and no statistic over team-game rows separates them,
because both halves are baked into the one composite. This probe separates
them by scoring the half that is currently invisible: hitter against himself.

  team-game, today   n=598     corr SE 0.0410
  hitter PA          per slate ~1,100 rows      corr SE 0.0066 at 22,758

Reading it. A positive, well-determined per-hitter relationship alongside the
null team-level one is evidence for B -- fix the aggregation, leave the rates.
A per-hitter relationship that is ALSO null is evidence for A, and the shrunk
rate is the thing to replace. Neither is a betting signal and this probe
computes none: it scores a rate against a rate.

WHAT IT NEEDS, AND WHY IT STARTS EMPTY. Two artifacts that only exist going
forward:

  data/hitters_<date>_<suffix>.csv   `hitter_frame`, written by the build from
                                     the vector `aggregate_lineup` composited.
  the collector's per-PA rows        `lineup_window_collect.py`, which already
                                     emits `batter_id` on every PA.

The prediction half cannot be backfilled. Rebuilding a past slate's per-hitter
frame needs that slate's Savant leaderboard, which is the lookahead
`.savant_cache/` exists to forbid -- so the 299 v12 games behind the current
component block can never be scored this way, and the sample starts at zero on
the first slate after `hitter_frame` ships. The outcome half is ordinary
backfill and can be collected for any finished game.

Do NOT read a slate or two of this. At ~1,100 PA rows a slate the standard
error falls fast, but the rows are not independent -- one hitter appears many
times -- so the effective n is smaller than the row count and the printed SE is
optimistic until a few weeks have accumulated. The gate is stated in the output
rather than left to judgement.

Diagnostic only. Nothing here feeds a lean, a delta or a grade.

Usage:
    python hitter_level_probe.py
    python hitter_level_probe.py --pa lineup_window_pa.csv --hitters data/
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import actuals_backfill as ab
import hitter_frame

# wOBA value of one plate appearance, by the collector's own category. The
# denominator-excluded categories (sac bunt, catcher interference) are dropped
# rather than scored as zero -- scoring them as outs is what a naive join does
# and it biases every hitter with a bunt in his line.
# A sacrifice fly IS in the wOBA denominator and scores zero, so it is a value
# here rather than an exclusion; an intentional walk, sacrifice bunt and
# catcher interference are not in the denominator at all. Scoring the excluded
# three as outs is what a naive join does, and it penalises every hitter who
# happens to have bunted.
_PA_VALUE = {
    "1b": ab.WOBA_W["1b"], "2b": ab.WOBA_W["2b"], "3b": ab.WOBA_W["3b"],
    "hr": ab.WOBA_W["hr"], "bb": ab.WOBA_W["bb"], "hbp": ab.WOBA_W["hbp"],
    "out": 0.0, "sf": 0.0,
}
_NOT_IN_DENOM = {"ibb", "sh", "ci"}


def load_hitters(data_dir="data"):
    """Every persisted per-hitter frame, newest schema assumed throughout.

    Rebuild-prefixed frames are included: a rebuild is a legitimate later view
    of the same slate, and the PREDICTION it carries is still the one that
    slate's build made -- unlike a rebuilt leans dump, whose numbers came from
    a later leaderboard. That distinction is why this reads `model_tag` and
    `snapshot_utc` off the rows rather than trusting the filename.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, f"{hitter_frame.PREFIX}_*.csv")))
    paths += sorted(glob.glob(os.path.join(data_dir, f"rebuild_{hitter_frame.PREFIX}_*.csv")))
    if not paths:
        return pd.DataFrame(columns=hitter_frame.COLUMNS)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def pa_values(pa):
    """Per-PA wOBA value and denominator flag, from the collector's `cat`."""
    d = pa.copy()
    d["in_denom"] = ~d["cat"].isin(_NOT_IN_DENOM)
    d["woba_value"] = d["cat"].map(_PA_VALUE)
    return d


def join(hitters, pa):
    """One row per (hitter, lineup): prediction beside his realised rate.

    Joins on `(game_pk, batter_id)`. The hitter frame is keyed by
    `faced_pitcher` as well, so a hitter who bats against both an opener and a
    bulk arm would appear twice if the build ever split him -- deduplicated on
    the prediction side rather than the outcome side, because the outcome is
    one sequence of plate appearances and must not be double-counted.
    """
    if hitters.empty or pa.empty:
        return pd.DataFrame()
    h = hitters.dropna(subset=["player_id"]).copy()
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h.dropna(subset=["player_id"])
    h = h.drop_duplicates(subset=["game_pk", "player_id"], keep="first")

    d = pa_values(pa[pa.get("reconciled", True).astype(bool)]
                  if "reconciled" in pa.columns else pa)
    d = d[d["in_denom"]]
    agg = (d.groupby(["game_pk", "batter_id"])
             .agg(n_pa=("woba_value", "size"), act=("woba_value", "mean"))
             .reset_index())
    m = h.merge(agg, left_on=["game_pk", "player_id"],
                right_on=["game_pk", "batter_id"], how="inner")
    return m


def report(hitters, pa, min_pa=1):
    out = []

    def say(s=""):
        out.append(s)

    say("HITTER-LEVEL PROBE — predicted xwOBA against the hitter's own PAs")
    say("diagnostic only; feeds no lean, delta or grade")
    say()
    if hitters.empty:
        say("no persisted per-hitter frames yet.")
        say("  `hitter_frame` writes data/hitters_<date>_<suffix>.csv from the")
        say("  build. The prediction half CANNOT be backfilled -- a past slate's")
        say("  per-hitter frame needs that slate's Savant leaderboard, which is")
        say("  the lookahead .savant_cache/ exists to forbid. First rows arrive")
        say("  on the first slate after it ships.")
        return out
    if pa.empty:
        say(f"{len(hitters)} persisted hitter rows, but no per-PA rows supplied.")
        say("  Run lineup_window_collect.py and pass its CSV with --pa.")
        return out

    m = join(hitters, pa)
    say(f"persisted hitter rows: {len(hitters)}   per-PA rows: {len(pa)}")
    say(f"joined hitter-games:   {len(m)}")
    if m.empty:
        say("  nothing joined — the two artifacts cover no common game.")
        return out

    m = m[m["n_pa"] >= min_pa]
    say(f"  with >= {min_pa} scoring PA: {len(m)}   "
        f"total scoring PAs: {int(m['n_pa'].sum())}")
    say()

    for label, col in (("shrunk (what the composite used)", "xwoba_shrunk"),
                       ("raw (pre-shrinkage)", "xwoba_raw")):
        s = m.dropna(subset=[col, "act"])
        if len(s) < 30:
            say(f"  {label:34s} n={len(s)} — too few to fit")
            continue
        cal = ab.calibration(s[col], s["act"])
        r = float(np.corrcoef(s[col], s["act"])[0, 1])
        se = 1.0 / np.sqrt(len(s) - 3)
        say(f"  {label:34s} n={len(s):5d}  slope {cal['slope']:+.3f}"
            f"±{cal['se_slope']:.3f}  corr {r:+.4f}±{se:.4f}")

    say()
    say("  The shrunk line is the one that matters: it is the value the lineup")
    say("  composite consumed. The raw line beside it says whether shrinkage")
    say("  helped or hurt at the hitter level, which is the question K cannot")
    say("  be tuned on at team level -- uniform shrinkage is affine in the")
    say("  lineup mean, so it moves the composite's SPREAD and not its ORDER.")
    say()
    say("  Read against the team-level component line in ledger_report.txt.")
    say("  Positive here and null there implicates the AGGREGATION; null in")
    say("  both implicates the per-hitter rate itself.")
    say()
    say("  PAs are not independent observations -- one hitter recurs — so the")
    say("  printed SE is optimistic. Treat a few slates as a smoke test.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hitters", default="data",
                   help="directory holding hitters_*.csv (default: data)")
    p.add_argument("--pa", default="lineup_window_pa.csv",
                   help="per-PA CSV from lineup_window_collect.py")
    p.add_argument("--min-pa", type=int, default=1)
    a = p.parse_args()
    h = load_hitters(a.hitters)
    pa = (pd.read_csv(a.pa) if os.path.exists(a.pa)
          else pd.DataFrame(columns=["game_pk", "batter_id", "cat"]))
    print("\n".join(report(h, pa, a.min_pa)))


if __name__ == "__main__":
    main()
