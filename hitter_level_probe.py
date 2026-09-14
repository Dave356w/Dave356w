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

    Rebuild-prefixed frames are included HERE and filtered later, by snapshot
    rather than by filename. The docstring used to argue they were harmless --
    that a rebuilt hitter frame still carries "the one that slate's build
    made", unlike a rebuilt leans dump. That is false, and measurably so: a
    frame is written from the build's Savant leaderboard exactly as a leans
    dump is, so a frame written after first pitch carries a prediction the
    game itself is already inside. `hitter_frame` has no `rebuild_` diversion
    (only `leans_*` and `shadow_*` get one), the file is overwritten by every
    build, and the committed copy is therefore the LAST build of that slate --
    which for most slates is the post-rollover grading pass. Measured on the
    committed frames: 91 of 115 games (79.1%) were written after first pitch,
    median 172 minutes late. `pregame_only` is what keeps those out of a rate.
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


LEDGER = "data/mlb_lean_ledger.csv"


def pregame_only(hitters, ledger=LEDGER):
    """Drop hitter rows whose frame was written after that game's first pitch.

    This is the probe's no-lookahead guard and it is not optional. The
    prediction half is only a prediction if it predates the game; a frame
    written from a post-rollover build carries a leaderboard that already
    contains the game being scored, which is the contamination `.savant_cache/`
    being gitignored exists to prevent on the dump side.

    Three outcomes, each counted from its own evidence rather than by
    subtraction, because a count derived by subtraction cannot carry a name
    you did not measure:

      pregame    snapshot strictly before `scheduled_start_utc` -- scored.
      post_hoc   snapshot at or after it -- dropped.
      unknown    no snapshot on the row, or no ledger start for the game --
                 kept, and the report says so. Silently dropping these would
                 hide a schema gap; silently scoring them as pregame would
                 assert provenance the artifact cannot substantiate.

    Returns (kept_frame, counts).
    """
    counts = {"pregame": 0, "post_hoc": 0, "unknown": 0}
    if hitters.empty or "snapshot_utc" not in hitters.columns:
        counts["unknown"] = len(hitters)
        return hitters, counts
    try:
        led = pd.read_csv(ledger, low_memory=False)
    except (OSError, ValueError):
        counts["unknown"] = len(hitters)
        return hitters, counts
    gp = pd.to_numeric(led.get("game_pk"), errors="coerce")
    start = pd.to_datetime(led.get("scheduled_start_utc"), utc=True,
                           errors="coerce")
    starts = (pd.DataFrame({"gp": gp, "start": start})
                .dropna().drop_duplicates("gp").set_index("gp")["start"])

    h = hitters.copy()
    snap = pd.to_datetime(h["snapshot_utc"], utc=True, errors="coerce")
    st = pd.to_numeric(h["game_pk"], errors="coerce").map(starts)
    known = snap.notna() & st.notna()
    post = known & (snap >= st)
    counts["unknown"] = int((~known).sum())
    counts["post_hoc"] = int(post.sum())
    counts["pregame"] = int((known & ~post).sum())
    return h[~post], counts


# Cluster bootstrap settings. Fixed seed because a probe's output is read
# across runs and a churning error bar is indistinguishable from a moving one.
CLUSTER_BOOT = 2000
CLUSTER_SEED = 20260914


def cluster_se_corr(x, y, clusters, n_boot=CLUSTER_BOOT, seed=CLUSTER_SEED):
    """SE of a correlation when rows repeat within a cluster.

    `1/sqrt(n-3)` assumes independent observations. These are not: 1832
    hitter-games came from 422 players, ~4.3 apiece, and a hitter's rate is
    the same number in every one of his rows. Resampling PLAYERS rather than
    rows carries that dependence into the interval. The probe already said its
    SE was optimistic; saying so is not the same as correcting it, and a
    borderline z is exactly where the difference decides the reading.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = np.asarray(clusters)
    uniq, inv = np.unique(g, return_inverse=True)
    if len(uniq) < 3:
        return float("nan")
    idx = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx[k] for k in pick])
        xs, ys = x[rows], y[rows]
        if xs.std() == 0 or ys.std() == 0:
            continue
        out.append(float(np.corrcoef(xs, ys)[0, 1]))
    return float(np.std(out, ddof=1)) if len(out) > 1 else float("nan")


def team_control(m):
    """The SAME hitter-games, aggregated to the team level.

    This exists because the probe's own reading rule used to point at
    `ledger_report.txt`'s component line, and that line is computed over every
    v12 slate while this probe covers only the slates with a persisted frame.
    Measured on the first real run, the two row sets disagree in SIGN: the
    component line reads -0.039 over 792 side-games and the SAME games under
    this probe read +0.065 over 208. Comparing across them turns a sample
    difference into an apparent aggregation effect -- which is the finding the
    probe is built to produce, arrived at wrongly.

    So the control is derived from the probe's own joined rows and cannot
    drift from them: the prediction side is the slot-PA weighted composite
    rebuilt from the same hitters, and the actual side is their own PAs
    pooled. Returns None when a frame lacks the columns to do it, rather than
    reporting a control over a different population.
    """
    need = {"game_pk", "batting_side", "xwoba_shrunk", "act", "n_pa"}
    if m.empty or not need <= set(m.columns):
        return None
    d = m.dropna(subset=["xwoba_shrunk", "act", "n_pa"]).copy()
    if d.empty:
        return None
    w = pd.to_numeric(d["slot_weight"], errors="coerce") if "slot_weight" in d else None
    d["w"] = 1.0 if w is None else w.where(w.notna() & (w > 0), 1.0)
    d["pw"] = d["xwoba_shrunk"] * d["w"]
    d["aw"] = d["act"] * d["n_pa"]
    agg = (d.groupby(["game_pk", "batting_side"])
             .agg(pw=("pw", "sum"), w=("w", "sum"),
                  aw=("aw", "sum"), npa=("n_pa", "sum"))
             .reset_index())
    agg = agg[(agg["w"] > 0) & (agg["npa"] > 0)]
    # No sample-size gate. A hard `>= N` here is the threshold cliff this repo
    # has removed three times, and suppressing the number is worse than a wide
    # one: on the first clean run this refused at 28 sides against a floor of
    # 30, so the probe printed NO control at all -- inviting exactly the
    # cross-row-set comparison the control exists to replace. The SE says what
    # a thin control is worth. The only refusals left are structural: a
    # correlation needs at least two points to exist and `1/sqrt(n-3)` needs
    # four to be finite.
    if len(agg) < 4:
        return None
    pred = agg["pw"] / agg["w"]
    act = agg["aw"] / agg["npa"]
    if pred.std() == 0 or act.std() == 0:
        return None
    r = float(np.corrcoef(pred, act)[0, 1])
    return len(agg), r, 1.0 / np.sqrt(len(agg) - 3)


def report(hitters, pa, min_pa=1, ledger=LEDGER):
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

    # The no-lookahead guard comes FIRST, before anything is joined or scored,
    # so no number below can be computed on a frame the game is already inside.
    n_all = len(hitters)
    hitters, prov = pregame_only(hitters, ledger)
    say(f"persisted hitter rows: {n_all}   per-PA rows: {len(pa)}")
    say(f"  provenance: {prov['pregame']} pregame, "
        f"{prov['post_hoc']} written AFTER first pitch (DROPPED), "
        f"{prov['unknown']} unknown (kept)")
    if prov["post_hoc"]:
        say("    `hitter_frame` has no `rebuild_` diversion the way leans_* and")
        say("    shadow_* do, and every build overwrites the slate's file -- so")
        say("    the committed copy is the LAST build, which for most slates is")
        say("    the post-rollover pass. Those frames carry a leaderboard the")
        say("    game is already inside; scoring them would be lookahead.")
    if prov["unknown"] and not prov["pregame"]:
        say("    NO row could be checked: nothing here is verified pregame.")
    if hitters.empty:
        say()
        say("  every frame was post-hoc — nothing left to score.")
        return out

    m = join(hitters, pa)
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
        # The clustered SE is the one to read. The naive one is printed beside
        # it rather than replaced, because the gap between them IS the
        # dependence, and hiding it would leave a reader unable to see why the
        # interval is wider than the row count suggests.
        cse = (cluster_se_corr(s[col].to_numpy(), s["act"].to_numpy(),
                               s["player_id"].to_numpy())
               if "player_id" in s else float("nan"))
        line = (f"  {label:34s} n={len(s):5d}  slope {cal['slope']:+.3f}"
                f"±{cal['se_slope']:.3f}  corr {r:+.4f}")
        if np.isfinite(cse):
            say(line + f"±{cse:.4f} clustered  (±{se:.4f} if rows were independent)")
        else:
            say(line + f"±{se:.4f}")

    say()
    tc = team_control(m)
    if tc is None:
        say("  TEAM CONTROL on these same rows: not computable from this frame")
        say("    (needs game_pk, batting_side, act and n_pa, and >= 30 sides)")
    else:
        n_t, r_t, se_t = tc
        say(f"  TEAM CONTROL, the same hitter-games aggregated: "
            f"n={n_t:5d}  corr {r_t:+.4f}±{se_t:.4f}")
        say("    This, not ledger_report.txt's component line, is what the")
        say("    numbers above are read against. That line covers every v12")
        say("    slate; this probe covers only the slates with a persisted")
        say("    frame, and on the first real run the two disagreed in SIGN")
        say("    (-0.039 over 792 side-games there, +0.065 over the same 208")
        say("    games here). A control on a different row set can turn a")
        say("    sample difference into an apparent aggregation effect.")

    say()
    say("  The shrunk line is the one that matters: it is the value the lineup")
    say("  composite consumed. The raw line beside it says whether shrinkage")
    say("  helped or hurt at the hitter level, which is the question K cannot")
    say("  be tuned on at team level -- uniform shrinkage is affine in the")
    say("  lineup mean, so it moves the composite's SPREAD and not its ORDER.")
    say()
    say("  Hitter-level clearly above the TEAM CONTROL above implicates the")
    say("  AGGREGATION; the two agreeing implicates the per-hitter rate")
    say("  itself. Take the difference against the control's own rows, and")
    say("  read it against the clustered interval, not the naive one.")
    say()
    say("  PAs are not independent observations -- one hitter recurs, so the")
    say("  naive SE is optimistic and the clustered SE above corrects for it")
    say("  by resampling players. Treat a few slates as a smoke test anyway:")
    say("  clustering fixes the interval, not the slate-to-slate variation")
    say("  that moved the control by 0.10 between row sets.")
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
