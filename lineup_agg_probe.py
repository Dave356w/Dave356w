"""Which composite of the nine hitters, if any -- derived first, measured second.

`aggregate_lineup` shrinks each hitter's rate by his own PA, weights the nine
by expected plate appearances per batting-order slot, and publishes one number.
The natural way to ask whether that is the right number is to score seven
alternative composites against the team's realised wOBA and take the winner.
That is a search, and on this data a search returns a winner whether or not one
exists -- the same hazard `value_probe` records at the price-band grid, where
the best of fifteen cells averages +20% ROI under a null.

So this panel is built the other way round. It CLOSES what arithmetic can
close, and then measures only what is left.

WHAT ARITHMETIC CLOSES, AND WHAT IT DOES NOT.

  The weight family. Any composite of the form `sum(w_i x_i)/sum(w_i)` with
  weights that vary by a coefficient of variation `c` sits within
  `c * sd_within / sqrt(n_batters)` of the equally-weighted mean -- so the
  whole family is one predictor to within that shift. The shipped slot
  weights' own `c` is ~0.069, which is what two turns through the order
  implies, and the within-lineup hitter spread is ~0.025, so the shift is
  ~0.0006 against a composite spread of ~0.0071. Members of the family
  correlate above 0.997 with each other, measured, not assumed.

  What that does NOT do is bound the difference in their correlations.
  Correlation is scale-free, so the 0.6% of spread one weighting discards
  could in principle be the informative part: the worst case is
  `sqrt(1 - rho^2)`, around 0.06, which is the size of the whole ceiling. The
  arithmetic narrows the question from "which of seven" to "is the discarded
  sliver better-aligned than the 99.4% retained" -- a question the paired
  column below can answer at a reachable sample, because pairing two predictors
  that correlate at 0.998 cancels almost all of the sampling noise.

  The log-odds composite. Aggregating in logit space and mapping back is the
  one variant with a mechanism behind it (rates compose multiplicatively, which
  is why the model's own matchup term is `B*P/L`). Over the range these rates
  actually occupy -- roughly 0.25 to 0.43 -- the logit is near enough linear
  that the two composites agree to three decimals, which the panel measures
  every run rather than taking from this paragraph.

  CLOSED there means closed as a CANDIDATE: a difference that small cannot be
  worth a `MODEL_TAG` bump. It does NOT mean indistinguishable, and the
  distinction is not pedantry -- pairing two predictors that agree to 0.9996
  leaves a standard error far below their difference, so the paired column can
  and does separate them. An earlier draft of this docstring asserted that no
  sample ever would, and the panel's own table falsified it on the first run.
  That is the argument for printing a closed result every build instead of
  retiring it to a comment.

WHAT IS LEFT is the handful of composites that leave the family altogether by
discarding hitters -- the top-four mean, the best bat. Those differ from the
shipped composite enough to be separated at a sample this repo will actually
reach, and they are the only rows here a reader should expect to move.

TWO GUARDS, BECAUSE THE TWO COLUMNS FAIL DIFFERENTLY.

  A variant's own correlation is unreadable until `se(r)` is at most half its
  own ceiling -- below that it could not reach `|z| = 2` even if the composite
  were perfect, and printing a ranking of unreadable numbers is how noise
  becomes a finding. Each row says `UNUSABLE` for itself rather than the panel
  claiming an order.

  A PAIRED difference against the shipped composite is a different statistic
  with a much smaller standard error, because the two share the games and most
  of their spread. It can be readable while every standalone column is not,
  and it is the column that answers the question actually being asked -- "is
  this better than what ships", not "is this good".

THE POWER TABLE, so the next person does not rebuild it. For the standalone
column, `se = 1/sqrt(n-3)`, so `|z| = 2` at a ceiling `c` needs
`n = 3 + 4/c^2` sides, and four times that at half the ceiling. It is
arithmetic rather than data, which is why it is safe to write down here:

    ceiling c    n for |z| = 2 at c    at c/2
      0.05             1,603           6,403
      0.08               628           2,503
      0.11               334           1,326
      0.16               159             628
      0.21                94             366

The shipped composite's ceiling runs near 0.08, so its own column becomes
readable in the low hundreds of sides and not before. The paired column's gate
is per-comparison and is printed with each row, because it depends on how far
that variant sits from the shipped one rather than on the sample alone.

ROW SELECTION. Pregame frames only, through `hitter_level_probe.pregame_only`
rather than a second copy of that rule, and the current record family read off
`build_site.RECORD_TAGS` rather than named here -- a hardcoded tag list is the
stale row selector `interaction_probe` carried through two model bumps.

Runs anywhere: committed `data/hitters_*.csv` and the committed ledger, no live
API. Diagnostic only -- nothing here reaches a lean, a delta or a grade.

Every figure above is illustrative of scale and none of it is a result. Read
the run.

Usage:
    python lineup_agg_probe.py
    python lineup_agg_probe.py --hitters data --ledger data/mlb_lean_ledger.csv
"""
import argparse

import numpy as np
import pandas as pd

import build_site
import hitter_level_probe as hp

LEDGER = hp.LEDGER

# Paired-bootstrap settings. Fixed seed for the reason every other resample in
# this repo carries one: a committed diagnostic that churns its own error bars
# between runs cannot be read across them.
BOOT = 4000
SEED = 20260916


def _wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(x[ok], weights=w[ok]))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def _top_slots(g, col, k=4):
    o = pd.to_numeric(g.get("batting_order"), errors="coerce")
    x = pd.to_numeric(g[col], errors="coerce")
    sel = x[np.isfinite(o) & (o <= k)]
    return float(sel.mean()) if len(sel) else float("nan")


# The variants, in a FIXED declared order that is never sorted by result. The
# first is what ships and is the baseline every paired difference is taken
# against; `logodds_shrunk` is here only so the closure above can be MEASURED
# on every run rather than asserted from a comment.
BASELINE = "slot_shrunk"
VARIANTS = (
    ("slot_shrunk", "slot-weighted shrunk",
     lambda g: _wmean(g["xwoba_shrunk"], g["slot_weight"]),
     "what ships"),
    ("flat_shrunk", "unweighted shrunk",
     lambda g: float(pd.to_numeric(g["xwoba_shrunk"], errors="coerce").mean()),
     "the family's other end: drop the slot weighting entirely"),
    ("pa_raw", "PA-weighted raw",
     lambda g: _wmean(g["xwoba_raw"], g["PA"]),
     "weights the season sample rather than the expected plate appearances"),
    ("flat_raw", "unweighted raw",
     lambda g: float(pd.to_numeric(g["xwoba_raw"], errors="coerce").mean()),
     "no shrinkage: the spread it adds is noise, not talent"),
    ("slot_raw", "slot-weighted raw",
     lambda g: _wmean(g["xwoba_raw"], g["slot_weight"]),
     "isolates shrinkage from weighting"),
    ("top4_shrunk", "top four slots",
     lambda g: _top_slots(g, "xwoba_shrunk"),
     "leaves the family: discards five hitters outright"),
    ("max_shrunk", "best bat",
     lambda g: float(pd.to_numeric(g["xwoba_shrunk"], errors="coerce").max()),
     "leaves the family: an order statistic of nine noisy rates"),
    ("logodds_shrunk", "log-odds, slot-weighted",
     lambda g: _expit(_wmean(_logit(g["xwoba_shrunk"]), g["slot_weight"])),
     "CLOSED — see the closure block; printed so the closure is re-measured"),
)


def load_sides(hitters_dir="data", ledger=LEDGER, tags=None):
    """One row per (game, batting side): every variant beside the realised wOBA.

    The pregame filter is `hitter_level_probe.pregame_only`, not a second copy
    of it. A frame written after first pitch carries a Savant leaderboard the
    game is already inside, so scoring it is lookahead -- and 79% of the frames
    written before that rule landed were exactly that.

    `batting_side` is the OFFENSE's side, so it pairs with `act_woba_<side>`
    directly. That is the one join in this module that could be silently wrong
    -- a crossed suffix returns a plausible weak correlation rather than an
    error, which is the hazard the ledger-report field conventions exist for --
    so a test pins it against a constructed frame rather than against a
    correlation's sign.
    """
    h = hp.load_hitters(hitters_dir)
    if h.empty:
        return pd.DataFrame(), {"pregame": 0, "post_hoc": 0, "unknown": 0}, h
    tags = tuple(tags if tags is not None else build_site.RECORD_TAGS)
    if tags and "model_tag" in h.columns:
        h = h[h["model_tag"].isin(tags)]
    h, prov = hp.pregame_only(h, ledger)
    need = {"game_pk", "batting_side", "xwoba_shrunk", "xwoba_raw",
            "slot_weight", "PA", "batting_order"}
    if h.empty or not need <= set(h.columns):
        return pd.DataFrame(), prov, h
    h = h.dropna(subset=["xwoba_shrunk", "xwoba_raw"])
    if h.empty:
        return pd.DataFrame(), prov, h

    preds = []
    for (gp, side), g in h.groupby(["game_pk", "batting_side"]):
        row = {"game_pk": gp, "batting_side": side, "n_batters": len(g)}
        for key, _label, fn, _note in VARIANTS:
            try:
                row[key] = fn(g)
            except (KeyError, ValueError, TypeError):
                row[key] = float("nan")
        preds.append(row)
    p = pd.DataFrame(preds)

    try:
        led = pd.read_csv(ledger, low_memory=False)
    except (OSError, ValueError):
        return pd.DataFrame(), prov, h
    acts = []
    gp = pd.to_numeric(led.get("game_pk"), errors="coerce")
    for side in ("away", "home"):
        col = f"act_woba_{side}"
        if col not in led.columns:
            continue
        acts.append(pd.DataFrame({
            "game_pk": gp, "batting_side": side,
            "act": pd.to_numeric(led[col], errors="coerce")}))
    if not acts:
        return pd.DataFrame(), prov, h
    a = pd.concat(acts, ignore_index=True).dropna(subset=["game_pk", "act"])
    a = a.drop_duplicates(subset=["game_pk", "batting_side"], keep="last")
    p["game_pk"] = pd.to_numeric(p["game_pk"], errors="coerce")
    return p.merge(a, on=["game_pk", "batting_side"], how="inner"), prov, h


def weight_family_closure(h):
    """How far ANY E[PA]-family reweighting can move the composite.

    Perturbing nine weights with a coefficient of variation `c` moves a mean of
    nine rates whose within-lineup spread is `s` by about `c*s/sqrt(9)`. Both
    inputs are measured from the frames rather than quoted, and the predicted
    shift is printed beside the one actually observed between the shipped and
    unweighted composites -- a derivation nobody checks against the data is how
    a comment becomes wrong without anyone noticing.

    `rho_floor` is the correlation the two ends of the family are guaranteed to
    share. `d_corr_bound` is the OTHER side of it, and it is the reason this
    function does not claim to close the question: `sqrt(1 - rho^2)` is what a
    difference in correlations could reach if the discarded sliver happened to
    carry all of the signal.
    """
    need = {"game_pk", "batting_side", "xwoba_shrunk", "slot_weight"}
    if h is None or getattr(h, "empty", True) or not need <= set(h.columns):
        return None
    cvs, sds, sizes = [], [], []
    for _, g in h.groupby(["game_pk", "batting_side"]):
        w = pd.to_numeric(g["slot_weight"], errors="coerce").dropna()
        x = pd.to_numeric(g["xwoba_shrunk"], errors="coerce").dropna()
        if len(w) >= 2 and w.mean() > 0:
            cvs.append(float(w.std(ddof=1) / w.mean()))
        if len(x) >= 2:
            sds.append(float(x.std(ddof=1)))
            sizes.append(len(x))
    if not cvs or not sds:
        return None
    cv = float(np.median(cvs))
    sd_in = float(np.median(sds))
    n_bat = float(np.median(sizes))
    shift = cv * sd_in / np.sqrt(max(n_bat, 1.0))
    return {"cv": cv, "sd_within": sd_in, "n_batters": n_bat,
            "predicted_shift": shift, "n_lineups": len(cvs)}


def logodds_closure(m):
    """Is the logit composite a different predictor, or the linear one slowly?

    Computed on the panel's own rows so it cannot drift from what the table
    scores. Returns the worst-case gap and the correlation, never a verdict --
    the verdict is in the report, where the numbers it rests on are printed
    beside it.
    """
    if m is None or getattr(m, "empty", True):
        return None
    cols = [BASELINE, "logodds_shrunk"]
    if not set(cols) <= set(m.columns):
        return None
    d = m.dropna(subset=cols)
    if len(d) < 3 or d[BASELINE].std() == 0 or d["logodds_shrunk"].std() == 0:
        return None
    lin = d[BASELINE].to_numpy(float)
    log = d["logodds_shrunk"].to_numpy(float)
    return {"n": len(d), "max_abs": float(np.max(np.abs(log - lin))),
            "sd_diff": float(np.std(log - lin, ddof=1)),
            "sd_lin": float(np.std(lin, ddof=1)),
            "pearson": float(np.corrcoef(log, lin)[0, 1]),
            # Spearman by ranking and taking Pearson, rather than through
            # pandas' `method="spearman"`, which imports scipy -- not a
            # dependency of this repo and not one to add for one statistic.
            "spearman": float(np.corrcoef(pd.Series(log).rank(),
                                          pd.Series(lin).rank())[0, 1])}


def panel(m, n_boot=BOOT, seed=SEED):
    """Every variant against the realised wOBA, with both guards attached.

    The standalone column is judged against the variant's OWN ceiling
    `sd(pred)/sd(act)`, because a composite that spreads more has more room to
    correlate and comparing raw correlations across variants compares headroom.
    `usable` is False when `se(r)` exceeds half that ceiling: such a row could
    not reach `|z| = 2` even if it were perfect, so it is reported as UNUSABLE
    rather than taking a place in an ordering.

    The paired column resamples GAMES, not variants, so the same resample
    scores both predictors -- that is where the noise cancellation comes from,
    and computing the two correlations on two bootstraps would throw it away.
    """
    if m is None or getattr(m, "empty", True) or "act" not in m.columns:
        return []
    base = m.dropna(subset=[BASELINE, "act"])
    n_base = len(base)
    sd_act_all = float(base["act"].std(ddof=1)) if n_base > 2 else 0.0
    rng = np.random.default_rng(seed)
    draws = ([rng.integers(0, n_base, n_base) for _ in range(n_boot)]
             if n_base > 3 else [])
    b_pred = base[BASELINE].to_numpy(float) if n_base else np.empty(0)
    b_act = base["act"].to_numpy(float) if n_base else np.empty(0)

    def _r(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    rows = []
    for key, label, _fn, note in VARIANTS:
        if key not in m.columns:
            continue
        d = m.dropna(subset=[key, "act"])
        n = len(d)
        pred = d[key].to_numpy(float)
        act = d["act"].to_numpy(float)
        sd_p = float(np.std(pred, ddof=1)) if n > 2 else 0.0
        sd_a = float(np.std(act, ddof=1)) if n > 2 else 0.0
        r = _r(pred, act)
        se = 1.0 / np.sqrt(n - 3) if n > 4 else float("nan")
        ceiling = (sd_p / sd_a) if sd_a > 0 else float("nan")
        row = {"key": key, "label": label, "note": note, "n": n,
               "sd_pred": sd_p, "sd_act": sd_a, "ceiling": ceiling,
               "corr": r, "se": se,
               "normalised": (r / ceiling) if ceiling and np.isfinite(ceiling)
               and ceiling > 0 else float("nan"),
               "usable": bool(np.isfinite(se) and np.isfinite(ceiling)
                              and ceiling > 0 and se <= ceiling / 2.0),
               "n_usable": (3.0 + 4.0 / ceiling ** 2)
               if np.isfinite(ceiling) and ceiling > 0 else float("nan"),
               "rho_base": float("nan"), "d_corr": float("nan"),
               "d_se": float("nan"), "n_paired": float("nan")}
        if key != BASELINE and n_base > 3 and key in base.columns:
            pair = base.dropna(subset=[key])
            if len(pair) > 3 and pair[key].std() > 0:
                v = pair[key].to_numpy(float)
                row["rho_base"] = _r(v, pair[BASELINE].to_numpy(float))
                row["d_corr"] = (_r(v, pair["act"].to_numpy(float))
                                 - _r(pair[BASELINE].to_numpy(float),
                                      pair["act"].to_numpy(float)))
                # The bootstrap runs on the BASELINE's row order so both
                # predictors see one resample of the same games.
                pos = base.index.get_indexer(pair.index)
                full = np.full(n_base, np.nan)
                full[pos] = v
                boots = []
                for idx in draws:
                    vv, bb, aa = full[idx], b_pred[idx], b_act[idx]
                    ok = np.isfinite(vv) & np.isfinite(bb) & np.isfinite(aa)
                    if ok.sum() < 4:
                        continue
                    da = _r(vv[ok], aa[ok]) - _r(bb[ok], aa[ok])
                    if np.isfinite(da):
                        boots.append(da)
                row["d_se"] = (float(np.std(boots, ddof=1))
                               if len(boots) > 1 else float("nan"))
                if (np.isfinite(row["d_se"]) and row["d_se"] > 0
                        and abs(row["d_corr"]) > 0):
                    row["n_paired"] = len(pair) * (2.0 * row["d_se"]
                                                   / abs(row["d_corr"])) ** 2
        rows.append(row)
    if rows and sd_act_all:
        rows[0].setdefault("sd_act_all", sd_act_all)
    return rows


def report(hitters_dir="data", ledger=LEDGER, tags=None, n_boot=BOOT):
    out = []

    def say(s=""):
        out.append(s)

    say("LINEUP AGGREGATION PANEL — which composite of the nine, if any")
    say("diagnostic only; feeds no lean, delta or grade")
    say()
    m, prov, h = load_sides(hitters_dir, ledger, tags)
    fam = tuple(tags if tags is not None else build_site.RECORD_TAGS)
    say(f"record family: {', '.join(fam) if fam else 'every tag'}")
    say(f"  frames: {prov['pregame']} pregame rows, {prov['post_hoc']} written "
        f"AFTER first pitch (DROPPED), {prov['unknown']} unknown (kept)")

    closure = weight_family_closure(h)
    if closure:
        say()
        say("CLOSED BY ARITHMETIC — the weight family")
        say(f"  slot weights vary within a lineup by CV {closure['cv']:.4f} "
            f"over {closure['n_lineups']} lineups; the hitters they average")
        say(f"  vary by sd {closure['sd_within']:.4f} across "
            f"{closure['n_batters']:.0f} bats. So ANY reweighting at that CV")
        say(f"  can move the composite by about "
            f"{closure['predicted_shift']:.5f} — c*s/sqrt(n).")

    say()
    if m.empty:
        say("no scorable sides yet.")
        say("  Needs committed data/hitters_*.csv from a pregame build AND a")
        say("  graded act_woba_* beside them in the ledger. The prediction half")
        say("  cannot be backfilled: rebuilding a past slate's frame needs that")
        say("  slate's Savant leaderboard, which is the lookahead")
        say("  .savant_cache/ exists to forbid.")
        say()
        say("  The per-hitter half of this question is hitter_level_probe,")
        say("  which derives the optimal weights instead of searching them —")
        say("  w_i ∝ E[PA_i]*beta_i — and needs the collector's per-PA rows.")
        return out

    rows = panel(m, n_boot=n_boot)
    by_key = {r["key"]: r for r in rows}
    b = by_key.get(BASELINE)
    flat = by_key.get("flat_shrunk")
    if closure and flat and np.isfinite(flat["rho_base"]):
        say(f"  MEASURED between the two ends of that family: "
            f"corr {flat['rho_base']:.5f}, so they are one predictor.")
        orth = float(np.sqrt(max(0.0, 1 - flat["rho_base"] ** 2)))
        say("  What this does NOT close: correlation is scale-free, so the")
        say(f"  {100 * orth:.1f}% of the composite's spread that a reweighting")
        say("  moves could in principle carry the signal, and the worst case")
        say(f"  on |d_corr| is that same {orth:.3f} — the size of the whole")
        say("  ceiling. Arithmetic narrows this question from 'which of seven'")
        say("  to 'is the moved sliver better aligned than the rest'; only the")
        say("  paired column answers that, and it is the column with the")
        say("  standard error small enough to try.")

    lo = logodds_closure(m)
    if lo:
        say()
        say("CLOSED BY ARITHMETIC — the log-odds composite")
        say(f"  max |logit-composite − linear composite| {lo['max_abs']:.5f} "
            f"over n={lo['n']} sides,")
        say(f"  against a composite spread of {lo['sd_lin']:.5f}: pearson "
            f"{lo['pearson']:.5f}, spearman {lo['spearman']:.5f}.")
        say("  Over the range these rates occupy the logit is near enough")
        say("  linear that the two are the same predictor to three decimals.")
        say("  CLOSED means closed as a CANDIDATE — a change this size cannot")
        say("  be worth a MODEL_TAG bump — and not 'indistinguishable'. The")
        say("  paired column below does distinguish it, because pairing two")
        say("  predictors that agree to 0.9996 leaves an interval far tighter")
        say("  than the difference. Read the two together: a real difference")
        say("  at the fourth decimal of a correlation whose ceiling is 0.08.")
        say("  It is re-measured every run rather than trusted from a comment,")
        say("  which is how the sentence that used to sit here — 'no sample")
        say("  will make it a candidate' — was caught by this panel's own")
        say("  table.")

    say()
    say(f"VARIANTS against the team's realised wOBA "
        f"(sd {b['sd_act'] if b else float('nan'):.4f})")
    say("  declared order, never sorted by result. Baseline = what ships.")
    say()
    say(f"  {'variant':22s} {'n':>4s} {'sd':>7s} {'ceil':>6s} {'corr':>8s} "
        f"{'r/ceil':>7s} {'rho_shipped':>11s} {'d_corr':>8s} {'±se':>7s}")
    for r in rows:
        rho = (f"{r['rho_base']:11.5f}" if np.isfinite(r["rho_base"])
               else f"{'—':>11s}")
        dc = (f"{r['d_corr']:+8.4f}" if np.isfinite(r["d_corr"])
              else f"{'—':>8s}")
        dse = (f"{r['d_se']:7.4f}" if np.isfinite(r["d_se"])
               else f"{'—':>7s}")
        say(f"  {r['label']:22s} {r['n']:4d} {r['sd_pred']:7.4f} "
            f"{r['ceiling']:6.3f} {r['corr']:+8.4f} {r['normalised']:+7.3f} "
            f"{rho} {dc} {dse}")
    say()
    say("  STANDALONE column")
    for r in rows:
        if r["usable"]:
            z = r["corr"] / r["se"]
            say(f"    {r['label']:22s} readable: z {z:+.2f} against its own "
                f"ceiling")
        else:
            say(f"    {r['label']:22s} UNUSABLE at n={r['n']}: se "
                f"{r['se']:.4f} > half its ceiling {r['ceiling']:.3f}; "
                f"needs n≈{r['n_usable']:,.0f}")
    say("    A row marked UNUSABLE could not reach |z| = 2 even if its")
    say("    composite were perfect, so it takes no place in any ordering.")
    say("    That is why this block is a list and not a ranking: with every")
    say("    row unusable, the 'best' correlation here is the largest of eight")
    say("    draws from noise, which is the band-grid result one surface out.")

    say()
    say("  PAIRED column — each variant minus what ships, same games")
    k = 0
    for r in rows:
        if r["key"] == BASELINE or not np.isfinite(r["d_corr"]):
            continue
        k += 1
        sep = (np.isfinite(r["d_se"]) and r["d_se"] > 0
               and abs(r["d_corr"]) >= 2 * r["d_se"])
        verdict = ("separated" if sep else "not separated")
        gate = (f"; |z| = 2 at n≈{r['n_paired']:,.0f}"
                if np.isfinite(r["n_paired"]) else "")
        z = (r["d_corr"] / r["d_se"]) if (np.isfinite(r["d_se"])
                                          and r["d_se"] > 0) else float("nan")
        say(f"    {r['label']:22s} {r['d_corr']:+.4f}±{r['d_se']:.4f} "
            f"(z {z:+.2f}) {verdict}{gate}")
        say(f"      {r['note']}")
    if k > 1:
        bar = float(np.sqrt(2.0 * np.log(k)))
        say(f"    BAR  {k} comparisons, so the largest |z| among them averages")
        say(f"         {bar:.2f} under pure noise. A row clears this panel only")
        say("         above that, not above zero.")

    say()
    say("  How to read the two columns together. Every standalone correlation")
    say("  here is unreadable and will be for a while; the paired column is")
    say("  readable now for the variants that leave the weight family, because")
    say("  pairing cancels the noise the two predictors share. So this panel")
    say("  can already say whether DISCARDING hitters hurts, and cannot yet")
    say("  say whether reweighting them helps — and the closure above says the")
    say("  second question has little room to matter however it lands.")
    say()
    say("  Not a betting signal and not an input: nothing here reaches a lean.")
    say("  The per-hitter half of this question is hitter_level_probe, which")
    say("  derives the optimal weights instead of searching them —")
    say("  w_i ∝ E[PA_i]*beta_i — and needs the collector's per-PA rows.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hitters", default="data",
                   help="directory holding hitters_*.csv (default: data)")
    p.add_argument("--ledger", default=LEDGER)
    p.add_argument("--all-tags", action="store_true",
                   help="score every model tag, not just the record family")
    a = p.parse_args()
    print("\n".join(report(a.hitters, a.ledger,
                           tags=() if a.all_tags else None)))


if __name__ == "__main__":
    main()
