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


def cluster_se(stat, clusters, n_boot=CLUSTER_BOOT, seed=CLUSTER_SEED):
    """SE of any row-level statistic when rows repeat within a cluster.

    `stat` takes an array of row indices and returns a float; a resample it
    cannot be computed on returns a non-finite value and is skipped rather
    than counted as zero.

    ONE resampler, not one per statistic. The correlation above and the
    moderation coefficients below are read against each other, and two loops
    carrying two seeds would put them on two different resamples of the same
    players -- a difference between two numbers would then partly be a
    difference between two bootstraps.
    """
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
        v = stat(rows)
        if v is not None and np.isfinite(v):
            out.append(float(v))
    return float(np.std(out, ddof=1)) if len(out) > 1 else float("nan")


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

    def corr(rows):
        xs, ys = x[rows], y[rows]
        if xs.std() == 0 or ys.std() == 0:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    return cluster_se(corr, clusters, n_boot, seed)


# The moderators, in a FIXED declared order so the panel cannot be read as a
# ranking. Three, not the four the obvious list has: the shrinkage weight
# `PA/(PA+K)` is strictly monotone in PA, so a "shrinkage weight" stratum is
# the PA stratum relabelled and would count a second time against the search
# correction below. The report MEASURES that rather than asserting it.
MODERATORS = (
    ("PA behind the rate", "PA",
     "also the K diagnostic: on the SHRUNK rate this slope is "
     "(PA+K)/(PA+K*), so it is flat exactly when K is calibrated"),
    ("batting slot", "batting_order",
     "the E[PA] axis the slot weights already carry"),
    ("rate level", "xwoba_shrunk",
     "curvature -- x*z is x^2 here, so this asks whether beta bends with the "
     "rate, not whether two groups of hitters differ"),
)


def _slot_weight_cv(m):
    """Median within-lineup coefficient of variation of the slot weights.

    This is the materiality bar, and it is derived rather than chosen. The
    shipped composite already varies its weights by this much; a beta that
    varies by LESS than this cannot change the composite more than the
    weighting it would be layered on top of, whatever its z-score. A bar
    picked by taste -- "beta must vary by 10%" -- would be the frozen constant
    this repo has filed four times.
    """
    if "slot_weight" not in m.columns:
        return None
    d = m.dropna(subset=["slot_weight"])
    if d.empty or not {"game_pk", "batting_side"} <= set(d.columns):
        return None
    cvs = []
    for _, g in d.groupby(["game_pk", "batting_side"]):
        w = pd.to_numeric(g["slot_weight"], errors="coerce").dropna()
        if len(w) < 2 or w.mean() <= 0:
            continue
        cvs.append(float(w.std(ddof=1) / w.mean()))
    return float(np.median(cvs)) if cvs else None


def beta_moderation(m, moderators=MODERATORS, n_boot=CLUSTER_BOOT,
                    seed=CLUSTER_SEED):
    """Does the per-hitter slope `beta` vary by hitter type?

    WHY THIS AND NOT A SEARCH OVER COMBINERS. If a hitter's rate `x_i`
    predicts his own plate appearances with slope `beta_i`, the linear
    composite that minimises squared error is `sum(w_i x_i)` with
    `w_i` proportional to `E[PA_i] * beta_i`. That is a derivation, not a
    hypothesis: nothing has to be tried. The slot weights are already an
    `E[PA]` estimate -- measured near-uniform at a within-lineup CV of ~0.07,
    which is what two turns through the order implies -- so the only unknown
    left in the expression is whether `beta` varies across hitters. A flat
    `beta` closes the aggregation question; a varying one hands over the
    weights directly. Either outcome is an answer, which is what the
    team-level panel cannot promise: at its own ceiling it can only ever
    return noise.

    THE FORM IS AN INTERACTION, NOT A MEDIAN SPLIT. Cutting hitters into
    high-PA and low-PA halves throws away the ordering and introduces a cut
    point that was chosen by looking -- and this repo has a `p = 0.693`
    instance of exactly that. Fitting

        act ~ b0 + b1*xc + b2*zc + b3*(xc*zc)

    on centred `x` and standardised `z` puts `b1` at `beta` for the average
    hitter and `b3` at the change in `beta` per standard deviation of the
    moderator, with no threshold anywhere. Per-stratum means are printed
    beside it as description and carry no claim of their own.

    THE PA ROW IS ALSO A MEASUREMENT OF `K`, AND THE TWO READINGS MUST NOT
    BE CONFLATED. Write the outcome as `act = theta + eps` and the season rate
    as `x = theta + e` with `Var(e) = sigma^2/PA`. The slope of the outcome on
    the RAW rate is then `PA/(PA + K*)` with `K* = sigma^2/tau^2` -- pure
    attenuation, rising with PA whatever the build does, and uninformative
    about hitter type. The composite does not consume the raw rate. It
    consumes the shrunk one, `mu + w(x - mu)` with `w = PA/(PA + K)`, and
    dividing through gives

        beta(PA) = (PA + K) / (PA + K*)

    which is flat at 1 for every PA exactly when `K = K*` and tilts otherwise.
    So a PA moderation on the shrunk rate is first a statement about the
    shrinkage constant and only second a statement about hitters: `b3 > 0`
    says `K` is too SMALL (low-PA rates reach the composite still carrying
    noise, and their slope is attenuated), `b3 < 0` says it is too large. This
    does not contradict the standing note that `K` cannot fix the lineup
    CORRELATION -- shrinkage is affine, so it moves the composite's spread and
    its slope while leaving its ORDER, and therefore `corr`, untouched. The
    weights read the slope. That is why this panel fits one.

    Read a material PA term as `K` before reading it as hitter type, and fix
    `K` rather than the weights: re-weighting by a beta that is really an
    un-shrunk residual would be a second, worse copy of the shrinkage.

    WHAT IT CANNOT DO. `b3` is descriptive: the moderators are properties of
    the hitter, not assignments, so a moderated slope says the relationship
    differs across hitters and never why. And `rate level` reuses the
    regressor as its own moderator, so its `b3` is a quadratic term rather
    than a between-group contrast -- labelled in `MODERATORS` rather than
    left for a reader to work out.

    Returns None when the frame cannot support a fit, rather than a slope
    with an invented interval.
    """
    need = {"xwoba_shrunk", "act", "player_id"}
    if m is None or getattr(m, "empty", True) or not need <= set(m.columns):
        return None
    d = m.dropna(subset=["xwoba_shrunk", "act", "player_id"]).copy()
    if len(d) < 30:
        return None
    x = pd.to_numeric(d["xwoba_shrunk"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d["act"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return None
    d, x, y = d[ok], x[ok], y[ok]
    if not float(np.std(x)) > 0:
        return None
    players = d["player_id"].to_numpy()
    xc_all = x - x.mean()

    def _slope(rows):
        xs, ys = xc_all[rows], y[rows]
        if xs.std() == 0:
            return float("nan")
        A = np.column_stack([np.ones(len(xs)), xs])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(coef[1])

    A = np.column_stack([np.ones(len(x)), xc_all])
    beta = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    se_beta = cluster_se(_slope, players, n_boot, seed)

    terms = []
    for label, col, note in moderators:
        if col not in d.columns:
            terms.append({"label": label, "note": note, "n": 0,
                          "b3": float("nan"), "se": float("nan"),
                          "reason": "column absent from this frame"})
            continue
        z = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        good = np.isfinite(z)
        if good.sum() < 30 or not float(np.std(z[good])) > 0:
            terms.append({"label": label, "note": note, "n": int(good.sum()),
                          "b3": float("nan"), "se": float("nan"),
                          "reason": "no usable spread in the moderator"})
            continue
        xg, yg, zg, pg = xc_all[good], y[good], z[good], players[good]
        zs = (zg - zg.mean()) / zg.std()

        def _b3(rows, xg=xg, yg=yg, zs=zs):
            xs, ys, zz = xg[rows], yg[rows], zs[rows]
            if xs.std() == 0 or zz.std() == 0:
                return float("nan")
            M = np.column_stack([np.ones(len(xs)), xs, zz, xs * zz])
            try:
                coef, *_ = np.linalg.lstsq(M, ys, rcond=None)
            except np.linalg.LinAlgError:
                return float("nan")
            return float(coef[3])

        b3 = _b3(np.arange(len(xg)))
        se = cluster_se(_b3, pg, n_boot, seed)
        terms.append({"label": label, "note": note, "n": int(len(xg)),
                      "b3": b3, "se": se, "reason": None})

    tested = [t for t in terms if np.isfinite(t["b3"])]
    k = max(len(tested), 1)
    return {
        "n": int(len(x)),
        "n_players": int(len(np.unique(players))),
        "beta": beta,
        "se_beta": se_beta,
        "terms": terms,
        "k": len(tested),
        # The bar a maximum is read against, not zero. `sqrt(2 ln k)` is what
        # the largest of k independent nulls typically returns, and this repo
        # has a sweep whose best |z| of 1.91 over 26 tests sat BELOW it.
        "expected_max_z": float(np.sqrt(2.0 * np.log(k))) if k > 1 else 0.0,
        "slot_cv": _slot_weight_cv(d),
    }


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


def ceiling_and_gate(raw, pa, act, sd_pa, se_naive, se_clustered):
    """The largest correlation this test could produce, and the n it needs.

    WHY A PROBE MUST PRINT THIS. Without it a null is unreadable: the reader
    cannot tell a per-hitter rate that carries nothing from a test too small to
    see one. The team-level version of this question spent weeks in that state
    -- its ceiling is 0.077 against an se of 0.050 at n=396, so it could not
    separate a PERFECT composite from a worthless one, and every null it
    produced was compatible with both.

    THE BOUND, and why it is computed from the RAW rate rather than the shrunk
    one. Shrinkage is affine, so `corr(shrunk, outcome) == corr(raw, outcome)`
    exactly -- the ceiling cannot depend on K, and a formula that reads K is
    measuring the wrong thing. The first version of this function used
    `sd(shrunk)/sd(actual)`. That is exact only when K is the well-calibrated
    `sigma^2/tau^2`; at the K this repo ships it OVERSTATED the ceiling by 39%
    in simulation (0.1214 printed against a true 0.0875), because under-
    shrinking inflates the predictor's spread with noise that cannot correlate
    with anything.

    The K-free form: with `x` the raw season rate, `x = theta + e` where
    `Var(e) = sigma^2/PA`, so the talent spread is `tau^2 = Var(x) -
    mean(sigma^2/PA)`, and against an outcome `A = theta + eps`,

        corr(x, A) = tau^2 / (sd(x) * sd(A))

    `sigma` is the per-PA wOBA sd, measured from the very plate appearances
    being scored rather than assumed -- ~0.52, against a per-hitter talent
    spread near 0.03, which is why a few PA of chance dwarf the whole signal
    and the bound lands near 0.09 however good the rate is.

    It remains an APPROXIMATION: it assumes the game noise is independent of
    talent and that the season rate's noise is binomial in PA. A measured
    correlation can still exceed it -- the team-level bullpen line does, at
    105% of its own. Read it as the order of magnitude a null is judged
    against, never as a threshold something can "beat".

    The gate is stated at the ceiling AND at half of it, because a rate that is
    real but partial is the likelier outcome and costs four times the sample.
    Clustering is folded in from the ratio the run itself measured rather than
    from a constant: `se` falls as `1/sqrt(n)`, so the n needed scales with the
    square of however much dependence widened the interval.
    """
    x = np.asarray(raw, float)
    n = np.asarray(pa, float)
    a = np.asarray(act, float)
    ok = np.isfinite(x) & np.isfinite(n) & np.isfinite(a) & (n > 0)
    if ok.sum() < 30:
        return None
    x, n, a = x[ok], n[ok], a[ok]
    sx = float(np.std(x, ddof=1))
    sa = float(np.std(a, ddof=1))
    if not (sx > 0 and sa > 0):
        return None
    # tau^2 is FITTED, not subtracted. Var(x_i) = tau^2 + sigma^2/n_i, so
    # regressing squared deviations on 1/n gives tau^2 as the intercept and
    # sigma^2 as the slope. Subtracting a flat mean(sigma^2/n) instead -- the
    # first thing tried -- is dominated by the low-PA tail: PA runs down to 4
    # here, the harmonic mean is 143 against an arithmetic 382, and tau^2 came
    # out NEGATIVE. Those rows are not PA-sized samples either; a Savant
    # backfilled hitter carries the team aggregate and sits at the mean by
    # construction, so his rate has neither the spread nor the noise the model
    # assumes. The fit is robust to them because they land at one end of 1/n
    # and move the slope rather than the intercept.
    inv = 1.0 / n
    d2 = (x - x.mean()) ** 2
    if float(np.std(inv)) <= 1e-12 * max(float(np.mean(inv)), 1e-12):
        # Every hitter carries the same PA, so 1/n cannot separate the two
        # components and the fit is collinear. There the subtraction IS exact,
        # given sigma: Var(x) = tau^2 + sigma^2/n with one known n.
        if not (sd_pa and sd_pa > 0):
            return None
        tau2 = sx ** 2 - float(sd_pa) ** 2 * float(np.mean(inv))
        sig2_fit = float(sd_pa) ** 2
    else:
        A = np.column_stack([np.ones(len(n)), inv])
        try:
            coef, *_ = np.linalg.lstsq(A, d2, rcond=None)
        except np.linalg.LinAlgError:
            return None
        tau2, sig2_fit = float(coef[0]), float(coef[1])
    if tau2 <= 0:
        return None
    ceil = tau2 / (sx * sa)
    infl = 1.0
    if (se_naive and se_naive > 0 and se_clustered is not None
            and np.isfinite(se_clustered) and se_clustered > 0):
        infl = se_clustered / se_naive
    return {"sd_raw": sx, "sd_act": sa, "tau": float(np.sqrt(tau2)),
            "sigma_fit": float(np.sqrt(sig2_fit)) if sig2_fit > 0 else float("nan"),
            "sigma_obs": float(sd_pa) if sd_pa else float("nan"),
            "ceiling": ceil, "inflation": infl,
            "n_ceiling": (2.0 * infl / ceil) ** 2,
            "n_half": (2.0 * infl / (ceil / 2.0)) ** 2}


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
    cg = None
    # The per-PA wOBA sd, from the very plate appearances being scored rather
    # than a constant: it is what converts a season rate's PA count into the
    # noise that has to be netted out of its spread.
    _v = pa_values(pa)
    _v = _v[_v["in_denom"]]["woba_value"].astype(float)
    sd_pa = float(_v.std(ddof=1)) if len(_v) > 2 else 0.0
    # Slates are counted from the rows themselves so the gate's "how long" is
    # this sample's own accrual rate, not a figure frozen from a good week.
    rate = None
    if "snapshot_utc" in m.columns:
        slates = m["snapshot_utc"].astype(str).str[:10]
        n_sl = int(slates.nunique())
        if n_sl:
            rate = (len(m) / n_sl, n_sl)

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
        if col == "xwoba_shrunk":
            cg = ceiling_and_gate(s.get("xwoba_raw"), s.get("PA"), s["act"],
                                  sd_pa, se, cse)

    say()
    if cg:
        say(f"  CEILING  fitted talent sd {cg['tau']:.4f}  (raw spread "
            f"{cg['sd_raw']:.4f}; per-PA sigma fitted {cg['sigma_fit']:.3f} "
            f"vs {cg['sigma_obs']:.3f} measured)")
        say(f"           / sd(own-PA actual) {cg['sd_act']:.4f}"
            f"  ->  r <= {cg['ceiling']:.4f}")
        say("    The two sigmas are a CHECK, not decoration: they disagree when")
        say("    the raw rates are not PA-sized samples, which is when the")
        say("    fitted talent spread -- and so this whole bound -- is soft.")
        say("    An exactly correct per-hitter rate could not beat this: a few")
        say("    plate appearances of wOBA carry an order of magnitude more")
        say("    noise than the entire spread of hitter talent. Computed from")
        say("    the RAW rate because shrinkage is affine -- the correlation,")
        say("    and so the ceiling, cannot depend on K. Approximate, and a real")
        say("    correlation CAN exceed it: read it as the scale a null is")
        say("    judged against, never as a bar to clear.")
        say(f"  GATE  {cg['n_ceiling']:,.0f} hitter-games for |z| = 2 if the rate "
            f"is perfect,")
        say(f"        {cg['n_half']:,.0f} if it is half that strong"
            + (f"  (x{cg['inflation']:.2f} for the clustering measured here)"
               if abs(cg['inflation'] - 1.0) > 0.005 else ""))
        have = len(m)
        if rate:
            say(f"        have {have:,} over {rate[1]} slates "
                f"({rate[0]:,.0f} a slate) -> "
                f"{max(0.0,(cg['n_ceiling']-have)/rate[0]):,.0f} more slates to the "
                f"first, {max(0.0,(cg['n_half']-have)/rate[0]):,.0f} to the second")
        else:
            say(f"        have {have:,}")
        say("    Read NOTHING before the first. A null under it is an")
        say("    underpowered test, not a fact about the rate -- which is the")
        say("    state the team-level version of this question sat in for weeks.")

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
    bm = beta_moderation(m)
    if bm is None:
        say("  WEIGHTS: not computable on this frame (needs xwoba_shrunk, act,")
        say("    player_id and >= 30 rows with spread in the rate).")
    else:
        say("  WEIGHTS — the composite a per-hitter slope implies, derived")
        say("    If a hitter's rate x_i predicts his own plate appearances with")
        say("    slope beta_i, the linear composite that minimises squared")
        say("    error is sum(w_i x_i) with w_i ∝ E[PA_i] * beta_i. Nothing has")
        say("    to be tried for that: it is algebra, not a hypothesis. The")
        say("    slot weights are ALREADY an E[PA] estimate, so the one open")
        say("    term is whether beta varies across hitters. Flat beta closes")
        say("    the aggregation question; a varying one hands the weights over.")
        say()
        sb = (f"±{bm['se_beta']:.4f} clustered" if np.isfinite(bm["se_beta"])
              else " (no clustered SE: fewer than 3 players)")
        say(f"    pooled beta {bm['beta']:+.4f}{sb}   "
            f"n={bm['n']:,} hitter-games over {bm['n_players']:,} players")
        say("      beta is the slope of his own realised wOBA on his predicted")
        say("      rate. It is NOT the correlation above rescaled by taste --")
        say("      the weights read the slope, so the slope is what is fitted.")
        say()
        say("    does it move? change in beta per 1 sd of the moderator:")
        for t in bm["terms"]:
            if not np.isfinite(t["b3"]):
                say(f"      {t['label']:20s} — {t['reason']}")
                continue
            z = (t["b3"] / t["se"] if np.isfinite(t["se"]) and t["se"] > 0
                 else float("nan"))
            zt = f"z {z:+.2f}" if np.isfinite(z) else "z n/a"
            say(f"      {t['label']:20s} b3 {t['b3']:+.4f}"
                f"±{t['se']:.4f}  {zt}   n={t['n']:,}")
            say(f"        {t['note']}")
        say("      A material PA term is a statement about K BEFORE it is one")
        say("      about hitters: on the shrunk rate that slope is")
        say("      (PA+K)/(PA+K*), flat only when K is calibrated. b3 > 0 says")
        say("      K is too small. Fix K there, not the weights -- re-weighting")
        say("      on a beta that is really an un-shrunk residual is a second,")
        say("      worse copy of the shrinkage. (This does not reopen the")
        say("      standing note that K cannot fix the lineup CORRELATION:")
        say("      shrinkage is affine, so it moves spread and slope and never")
        say("      order. The weights read the slope.)")
        say("      The shrinkage weight is deliberately NOT a fourth row:")
        say("      PA/(PA+K) is strictly increasing in PA, so it is the first")
        say("      row relabelled and counting it twice would inflate the")
        say("      search correction below rather than test anything.")
        say()
        if bm["k"] > 1:
            say(f"    BAR  {bm['k']} moderators, so the largest |z| among them")
            say(f"         averages {bm['expected_max_z']:.2f} under pure noise."
                f" Read the maximum")
            say("         against that, never against zero.")
        if bm["slot_cv"]:
            cv = bm["slot_cv"]
            mat = cv * abs(bm["beta"])
            say(f"    MATERIALITY  the slot weights already vary by a")
            say(f"         within-lineup CV of {cv:.3f}, so beta has to vary by")
            say(f"         at least |b3| = {mat:.4f} before re-weighting on it")
            say("         changes the composite more than the weighting it")
            say("         would sit on top of. That bar is derived from the")
            say("         shipped weights, not chosen -- a round number here")
            say("         would be the frozen constant this repo files under")
            say("         `constants frozen from data`.")
            for t in bm["terms"]:
                if not (np.isfinite(t["b3"]) and np.isfinite(t["se"])
                        and t["se"] > 0 and mat > 0):
                    continue
                # se falls as 1/sqrt(n), so the n that resolves the BAR at
                # |z| = 2 scales with the square of how far the bar sits
                # inside the current interval.
                need = t["n"] * (2.0 * t["se"] / mat) ** 2
                extra = ((f", {max(0.0, (need - len(m)) / rate[0]):,.0f} more "
                          f"slates") if rate and rate[0] > 0 else "")
                say(f"         {t['label']:18s} resolves the bar at "
                    f"n≈{need:,.0f} ({need / max(t['n'], 1):.1f}x this "
                    f"sample{extra})")
            say("         If those are decades rather than weeks, that IS the")
            say("         answer and not a reason to wait: a beta that cannot")
            say("         be shown to vary by more than the slot weights' own")
            say("         CV cannot move the composite further than the two")
            say("         ends of the weight family sit apart, which")
            say("         lineup_agg_probe measures directly (they correlate")
            say("         above 0.997). Re-weighting is then closed by")
            say("         derivation, and the open question is whether to")
            say("         DISCARD hitters at all — which that panel can")
            say("         already answer and this one cannot.")
        say()
        say("    Read it this way. Every |z| under the bar and every b3 inside")
        say("    the materiality figure: beta is flat as far as this can see,")
        say("    the optimal weights are the E[PA] ones already shipped, and")
        say("    the aggregation is closed by derivation rather than by a")
        say("    search over combiners -- which is the outcome the team-level")
        say("    panel cannot deliver at any sample it will reach. A b3 above")
        say("    BOTH: re-weight by slot_weight * (beta + b3*z) and say which")
        say("    moderator, in that order.")
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
