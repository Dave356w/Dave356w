"""How does the starter's quality combine with each hitter's? The pitcher x
lineup interaction, scored one hitter at a time.

This is the question the per-hitter frames (`data/hitters_<date>_xw.csv`) were
stored to answer. The model combines the two sides in log5 / odds-ratio form:

    mx_sp = B * P / L        B hitter rate, P starter rate, L league

Centre both on the league (b = B - L, p = P - L) and that is

    mx_sp - L = b + p + b*p / L

so the shipped form makes ONE testable claim beyond "both sides matter": the
hitter and the pitcher INTERACT, with a coefficient of 1/L (~3.2) on `b*p`. A
good hitter gains more against a bad starter than against a good one. The plain
additive alternative, `B + P - L`, is the same expression with that coefficient
at zero. No team-level statistic can see this, because the lineup composite
averages nine `b`s into one before the starter is ever applied, and the ledger
keeps only the product.

The fit, per plate appearance against the starter the frame named:

    woba_value - L = beta_b * b + beta_p * p + beta_bp * b*p + e

  beta_b   does the hitter's stored rate predict his PAs against the starter?
  beta_p   does the starter's rate predict the PAs his opponents take?
  beta_bp  the interaction. log5 says 1/L; additive says 0.

All three are PRE-SPECIFIED, and there is no search here: the two reference
values are fixed by the shipped formula and its simplest control, not chosen
from the data.

WHAT THE SAMPLE CAN AND CANNOT RESOLVE, stated before any run so a reader does
not wait on the wrong column. The interaction's regressor is small: with `b`
and `p` each spread 0.02-0.03, `b*p` is spread well under 0.001, against a
single PA's wOBA spread of about 0.5. Its unclustered standard error is roughly

    se(beta_bp) ~ 0.5 / (sd(b*p) * sqrt(N_PA))

and clustering by hitter and starter inflates it further. Dated scale check,
not a result: the first live run (2026-09-24, 5,685 starter PAs) measured
sd(b*p) 0.0006 and se(beta_bp) 13.4 -- about four times the 1/L being tested
-- which puts a readable interaction near 400,000 PAs, several seasons. The
main effects' SEs were about 0.3 on the same rows. So the readable results
arrive in order: `beta_p` and `beta_b` first, the interaction much later. The
report prints the PA count at which the interaction's |z| would reach 2 if
log5 were exactly right, scaled from the run's OWN standard error rather than
from this paragraph. A null interaction before that count is "not yet
resolved", never "additive"; a large point estimate before it is noise-sized
too.

WHY THE STARTER'S PAs ONLY. `vs_starter` from `lineup_window_collect.py` marks
the PAs thrown by the pitcher who actually started. The frame names the pitcher
the build EXPECTED; when those differ (a scratch, an opener listed as the bulk
arm) the PAs belong to someone whose rate was never consumed. When the collector
supplies `pitcher_name`, those lineups are dropped and counted; when it does
not, the report says the starter's identity is unverified rather than
assuming it.

WHAT IS STORED IS WHAT WAS CONSUMED. `b` is the frame's `xwoba_shrunk` -- the
neutral, pre-platoon rate `aggregate_lineup` composited -- and `p` and `L` are
the ledger's own `starter_xwoba_<side>` and `mx_xwoba_sp - edge_xwoba_sp`, the
values the lean was computed from. Nothing is re-derived. The per-hitter
platoon offset is NOT stored, so the model's handedness adjustment is outside
this fit; it moves `b` for some hitters and is a known omitted term, not a
defect in the join.

ROW SELECTION reuses the siblings' rules rather than copying them: pregame
frames only through `hitter_level_probe.pregame_only`, and the current record
family off `build_site.RECORD_TAGS` -- `starter_xwoba` changed meaning at v13
(the centred 50/50 xwOBA/wOBA blend), so mixing families would score two
different `p`s as one.

Clustering. A hitter recurs across games and a starter across his lineups and
starts, and the two are crossed, so the SE is `cluster_se_twoway` over
(player, starter) -- the same estimator the hitter-level probe uses.

Diagnostic only. Nothing here reaches a lean, a delta or a grade, and no
result here changes v13 without a separately registered candidate.

Usage:
    python pitcher_lineup_probe.py --pa lineup_window_pa.csv --hitters data
"""
import argparse
import os

import numpy as np
import pandas as pd

import build_site
import hitter_level_probe as hp

LEDGER = hp.LEDGER

# Fewer resamples than the sibling probes: each draw is a least-squares fit on
# every PA, and three coefficients each take a two-way (three-bootstrap)
# SE. Fixed seed for the usual reason -- a churning error bar reads as a
# moving one.
BOOT = 600
SEED = 20260924

COEFS = ("beta_b", "beta_p", "beta_bp")


def _norm_name(s):
    """Accent-, case- and punctuation-blind, so "José Ramírez" from one feed
    matches "Jose Ramirez" from another rather than reading as a scratch."""
    return (s.astype(str).str.normalize("NFKD")
            .str.encode("ascii", "ignore").str.decode("ascii")
            .str.lower().str.replace(r"[^a-z]", "", regex=True))


def starter_inputs(ledger=LEDGER, tags=None):
    """(game_pk, pitcher_side) -> the starter's name, P and L from the ledger.

    `pitcher_side` is the PITCHER's team, so it pairs with `<side>_sp` and
    `starter_xwoba_<side>` directly. `L` is recovered per game exactly as
    `interaction_probe` does, from the away starter's matchup and edge -- the
    league value the log5 term divided by, not a season constant.
    """
    try:
        led = pd.read_csv(ledger, low_memory=False)
    except (OSError, ValueError):
        return pd.DataFrame()
    tags = tuple(tags if tags is not None else build_site.RECORD_TAGS)
    if tags and "model_tag" in led.columns:
        led = led[led["model_tag"].isin(tags)]
    if "model_metric" in led.columns:
        led = led[led["model_metric"] == "xwOBA"]

    def n(c):
        if c not in led.columns:
            return pd.Series(np.nan, index=led.index, dtype="float64")
        return pd.to_numeric(led[c], errors="coerce")

    L = n("mx_xwoba_sp_away") - n("edge_xwoba_sp_away")
    out = []
    for side in ("away", "home"):
        out.append(pd.DataFrame({
            "game_pk": pd.to_numeric(led.get("game_pk"), errors="coerce"),
            "pitcher_side": side,
            "ledger_sp": led.get(f"{side}_sp"),
            "P": n(f"starter_xwoba_{side}"),
            "L": L,
        }))
    s = pd.concat(out, ignore_index=True).dropna(subset=["game_pk", "P", "L"])
    return s.drop_duplicates(subset=["game_pk", "pitcher_side"], keep="last")


def build_rows(hitters, pa, ledger=LEDGER, tags=None):
    """One row per plate appearance against the starter, with b, p and L.

    Returns (rows, counts). Every exclusion is counted from its own evidence
    so the report can name what it dropped.
    """
    counts = {"frame_rows": len(hitters), "pregame": 0, "post_hoc": 0,
              "unknown": 0, "frame_sp_mismatch": 0, "starter_changed": 0,
              "identity": "unverified"}
    empty = pd.DataFrame()
    if hitters.empty or pa.empty:
        return empty, counts
    tags = tuple(tags if tags is not None else build_site.RECORD_TAGS)
    h = hitters
    if tags and "model_tag" in h.columns:
        h = h[h["model_tag"].isin(tags)]
    h, prov = hp.pregame_only(h, ledger)
    counts.update(prov)
    h = h.dropna(subset=["player_id", "xwoba_shrunk"]).copy()
    h["game_pk"] = pd.to_numeric(h["game_pk"], errors="coerce")
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h.drop_duplicates(subset=["game_pk", "player_id"], keep="first")

    s = starter_inputs(ledger, tags)
    if s.empty:
        return empty, counts
    h = h.merge(s, on=["game_pk", "pitcher_side"], how="inner")
    # The frame's pitcher must be the ledger's starter, or `P` belongs to
    # someone else -- a crossed side would pass silently otherwise.
    same = _norm_name(h["faced_pitcher"]) == _norm_name(h["ledger_sp"])
    counts["frame_sp_mismatch"] = int((~same).sum())
    h = h[same]

    d = pa
    if "reconciled" in d.columns:
        d = d[d["reconciled"].astype(bool)]
    if "vs_starter" not in d.columns:
        return empty, counts
    d = d[d["vs_starter"].astype(bool)]
    d = hp.pa_values(d)
    d = d[d["in_denom"]].dropna(subset=["woba_value"]).copy()
    d["game_pk"] = pd.to_numeric(d["game_pk"], errors="coerce")
    d["batter_id"] = pd.to_numeric(d["batter_id"], errors="coerce")

    m = d.merge(h, left_on=["game_pk", "batter_id", "bat_side"],
                right_on=["game_pk", "player_id", "batting_side"],
                how="inner")
    if "pitcher_name" in m.columns:
        counts["identity"] = "verified"
        ok = _norm_name(m["pitcher_name"]) == _norm_name(m["faced_pitcher"])
        # A lineup whose expected starter did not start is dropped whole:
        # its PAs were thrown by an arm whose rate was never consumed.
        bad = m.loc[~ok, ["game_pk", "batting_side"]].drop_duplicates()
        counts["starter_changed"] = len(bad)
        if len(bad):
            key = m.set_index(["game_pk", "batting_side"]).index
            drop = key.isin(bad.set_index(["game_pk", "batting_side"]).index)
            m = m[~drop]
    if m.empty:
        return empty, counts
    m = m.copy()
    m["y"] = m["woba_value"] - m["L"]
    m["b"] = m["xwoba_shrunk"] - m["L"]
    m["p"] = m["P"] - m["L"]
    m["bp"] = m["b"] * m["p"]
    return m.reset_index(drop=True), counts


def design(m):
    """(X, y) as arrays, built once: the bootstrap indexes these rather than
    slicing the frame on every draw."""
    X = np.column_stack([np.ones(len(m)), m["b"], m["p"], m["bp"]])
    return X.astype(float), m["y"].to_numpy(float)


def fit(m, rows=None, arrays=None):
    """OLS of y on (1, b, p, b*p). Intercept included so a league-level miss
    in `L` is absorbed rather than loaded onto the slopes."""
    X, y = arrays if arrays is not None else design(m)
    if rows is not None:
        X, y = X[rows], y[rows]
    if len(y) < 8:
        return None
    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < 4:
        return None
    coef = np.linalg.solve(xtx, X.T @ y)
    return dict(zip(("alpha",) + COEFS, map(float, coef)))


def paired_loss(m):
    """Mean squared-error difference, log5 minus additive, per PA.

    The two predictions differ only by `b*p/L`, so this is the interaction's
    predictive value on the same rows -- negative favours log5.
    """
    log5 = m["b"] + m["p"] + m["bp"] / m["L"]
    add = m["b"] + m["p"]
    return ((m["y"] - log5) ** 2 - (m["y"] - add) ** 2).to_numpy(float)


def report(hitters, pa, ledger=LEDGER, tags=None, n_boot=BOOT):
    out = []
    say = out.append
    say("PITCHER x LINEUP INTERACTION -- per hitter, PAs against the starter")
    say("  diagnostic only; nothing here reaches a lean, delta or grade")
    m, c = build_rows(hitters, pa, ledger, tags)
    say(f"  frame rows {c['frame_rows']}: pregame {c['pregame']}, "
        f"post-hoc dropped {c['post_hoc']}, provenance unknown {c['unknown']}")
    say(f"  frame starter != ledger starter (dropped): {c['frame_sp_mismatch']}")
    if c["identity"] == "verified":
        say(f"  lineups whose expected starter did not start (dropped): "
            f"{c['starter_changed']}")
    else:
        say("  starter identity UNVERIFIED: the PA rows carry no pitcher_name,")
        say("  so a scratched starter's replacement is scored as the starter.")
    if m.empty:
        say("")
        say("  No scorable rows. Needs committed pregame data/hitters_*.csv in")
        say("  the current record family AND the collector's per-PA rows with")
        say("  `vs_starter`. The sample starts at zero; that is not a result.")
        return out

    n_pa = len(m)
    say(f"  scored: {n_pa} PAs, {m['player_id'].nunique()} hitters, "
        f"{m['game_pk'].nunique()} games, {m['faced_pitcher'].nunique()} starters")
    say(f"  spread: sd(b) {m['b'].std():.4f}  sd(p) {m['p'].std():.4f}  "
        f"sd(b*p) {m['bp'].std():.5f}  sd(y) {m['y'].std():.3f}")
    arrays = design(m)
    est = fit(m, arrays=arrays)
    if est is None:
        say("  too few rows or no variation to fit; nothing to read")
        return out

    player = m["player_id"].to_numpy()
    starter = m["faced_pitcher"].astype(str)
    L_ref = float(m["L"].mean())
    refs = {"beta_b": (1.0, "log5 & additive: 1"),
            "beta_p": (1.0, "log5 & additive: 1"),
            "beta_bp": (1.0 / L_ref, f"log5: 1/L = {1.0 / L_ref:.2f}; additive: 0")}
    say("")
    say(f"  {'coef':<8} {'est':>8} {'se':>7}  {'z vs 0':>7}  {'z vs ref':>8}  "
        f"reference")
    for k in COEFS:
        se, how = hp.cluster_se_twoway(
            lambda r, k=k: (fit(m, r, arrays) or {}).get(k, np.nan),
            player, starter.to_numpy(), n_boot=n_boot, seed=SEED)
        ref, note = refs[k]
        z0 = est[k] / se if np.isfinite(se) and se > 0 else float("nan")
        zr = (est[k] - ref) / se if np.isfinite(se) and se > 0 else float("nan")
        say(f"  {k:<8} {est[k]:>+8.3f} {se:>7.3f}  {z0:>+7.2f}  {zr:>+8.2f}  "
            f"{note} [{how}]")
        if k == "beta_bp" and np.isfinite(se) and se > 0:
            need = n_pa * (2.0 * se / ref) ** 2
            say(f"  interaction readable (|z|=2 at log5's 1/L) near "
                f"{need:,.0f} PAs; have {n_pa:,}"
                + ("" if n_pa >= need else " -- NOT YET RESOLVED"))

    dl = paired_loss(m)
    se_dl, how = hp.cluster_se_twoway(
        lambda r: float(np.mean(dl[r])), player, starter.to_numpy(),
        n_boot=n_boot, seed=SEED)
    say("")
    say(f"  paired MSE, log5 - additive: {np.mean(dl):+.6f} "
        f"(se {se_dl:.6f}, {how}); negative favours log5")
    say("")
    say("  Read beta_p and beta_b first; they resolve long before the")
    say("  interaction does. A beta_bp interval covering both 0 and 1/L is the")
    say("  expected state for a long time and says nothing about either form.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hitters", default="data",
                   help="directory holding hitters_*.csv (default: data)")
    p.add_argument("--pa", default="lineup_window_pa.csv",
                   help="per-PA CSV from lineup_window_collect.py")
    p.add_argument("--ledger", default=LEDGER)
    a = p.parse_args()
    h = hp.load_hitters(a.hitters)
    pa = (pd.read_csv(a.pa) if os.path.exists(a.pa) else pd.DataFrame())
    print("\n".join(report(h, pa, a.ledger)))


if __name__ == "__main__":
    main()
