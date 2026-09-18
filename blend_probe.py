"""Does a 50/50 wOBA+xwOBA blend beat either metric alone? No -- and the
reason is arithmetic rather than a sample size.

WHY THIS EXISTS
---------------
"Average the surface metric and the expected one, so the lean reads neither
pure luck nor pure contact quality" is the most natural proposal left on the
metric axis, and it will recur.  It arrives with a plausible story (blending
re-anchors results to contact quality and damps BABIP/defence noise) and with
a distribution table showing the blend's spread sitting between the two
inputs.  Both halves of that story are true and neither is evidence.

The repo already owns the instrument that can answer it.  `shadow_metric.py`
writes one extra dump per slate built on whichever rate the primary is NOT, so
every paired slate carries the SAME games, players and morning under both
metrics.  A blended build is a deterministic function of those two dumps, so
it can be reconstructed exactly from committed artifacts -- no Savant call, no
rebuild of a past slate, and therefore no lookahead.

WHAT IT RECONSTRUCTS, AND WHY THE RECONSTRUCTION IS EXACT
---------------------------------------------------------
`sequential_xwoba_phases` is

    M_sp = B_vs_sp * P_sp / L ,  M_bp = B_neutral * P_bp / L
    M    = q*M_sp + (1-q)*M_bp ,  edge = M - L ,  net = edge_away - edge_home

and every term on the right is persisted per side in both dumps.  Two
properties make blending the dumps equal to blending the RATES and rebuilding:

  * Shrinkage is affine in the raw rate -- `(n*raw + K*prior)/(n+K)` -- and
    both arms shrink the same players at the same `n` under the same `K`, so
    blend-then-shrink and shrink-then-blend agree identically.  Lineup
    aggregation is a weighted mean, so it commutes for the same reason.
  * `q` is a workload share derived from `expected_sp_ip` and measured BF/IP.
    It reads no rate at all, and the dumps confirm it: `sp_share` is bitwise
    equal across the two arms on every paired side-row.

What does NOT commute is `matchup_value`, which is a RATIO and therefore
bilinear -- so `M` must be recomputed from blended components rather than
averaged.  That is the whole content of the reconstruction, and it is
self-checked: `reconstruction_error` rebuilds each arm's own published
`edge_xwOBA` from its own components and the probe refuses to report if that
is not exact.

THE FINDING, AND WHY IT IS A CEILING RATHER THAN A READING
-----------------------------------------------------------
For two standardised predictors correlating `rho` with each other and `r_w`,
`r_x` with the outcome, the equal-weight average correlates

    (r_w + r_x) / sqrt(2 + 2*rho)

with that outcome.  The two arms score almost identically (+0.1541 against
+0.1542 over 517 paired graded games) and correlate +0.84 on net, so the
formula returns +0.1607 and the reconstructed blend scores +0.1606 -- a
difference of 0.00009.  **The entire measured gain is the noise-averaging
identity.**  It would be there for any two predictors this correlated and this
equally good, it says nothing about either metric, and it is bounded above by
`1/sqrt((1+rho)/2)` = 4.3% of a correlation of 0.154, i.e. +0.0066.

Against a paired bootstrap se of 0.0125 at n=517, the largest effect the
change can produce is half its own measurement error.  That is not a wait-and-
see: separating +0.0066 from zero at 2 se needs ~7,900 paired games, roughly
590 slates.  The record disagrees with the correlation in the meantime (the
blend goes 302-215 where the shipped arm goes 307-210), which is what two
criteria do when neither is reading a signal.

WHAT IT DOES NOT ANSWER
-----------------------
Whether wOBA or xwOBA is the better rate.  That is `shadow_report.py`'s
question, it is measured there, and this probe reuses that pairing rather than
restating its answer.  It also cannot speak to a blend of RANKINGS or to
pitcher-level evaluation: it scores the model's own `net` against run margin,
which is the only thing in this repo a metric change is ultimately for.

Nothing here writes a dump, a ledger row, a page or a constant.  No
`MODEL_TAG` implication -- it ships nothing.

USAGE
-----
    python blend_probe.py                 # committed paired dumps as they stand
    python blend_probe.py --boot 4000     # bootstrap resamples (default 20000)
    python blend_probe.py --sweep-only    # just the blend-weight curve
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import shadow_report as sr

# The five rate inputs a blended build would blend.  Every one is a rate in the
# model's units; `sp_share` is deliberately absent because it is a workload
# share that reads no rate, and `platoon_delta_sp` because it is a shared
# additive constant already inside `opp_xwOBA_vs_sp` (identical across arms on
# the committed dumps, so averaging it is a no-op either way).
RATE_PARTS = ("opp_xwOBA_vs_sp", "opp_xwOBA_neutral", "starter_xwOBA",
              "bullpen_xwOBA", "lg_xwOBA")

# Equal weight is the proposal as stated.  The sweep reports the rest of the
# curve as CONTEXT and registers nothing: picking a weight off it would be the
# fitted-literal defect this repo has recorded four times, and the curve's own
# maximum is where the identity above puts it whenever the arms score alike.
BLEND_LAMBDA = 0.5


def _num(df, col):
    return pd.to_numeric(df.get(col), errors="coerce")


def edge_from_parts(parts, q):
    """`edge_xwOBA` rebuilt from the persisted components. See module header."""
    mx_sp = parts["opp_xwOBA_vs_sp"] * parts["starter_xwOBA"] / parts["lg_xwOBA"]
    mx_bp = parts["opp_xwOBA_neutral"] * parts["bullpen_xwOBA"] / parts["lg_xwOBA"]
    return q * mx_sp + (1.0 - q) * mx_bp - parts["lg_xwOBA"]


def reconstruction_error(df):
    """Max |rebuilt edge - published edge| on one dump. Exact means 0.0.

    The probe's whole claim rests on the rebuild being the build, so this is
    asserted rather than assumed -- and asserted on the SHIPPED arms, where a
    published answer exists to check against, before any blend is formed.
    """
    parts = {c: _num(df, c) for c in RATE_PARTS}
    got = edge_from_parts(parts, _num(df, "sp_share"))
    want = _num(df, "edge_xwOBA")
    m = got.notna() & want.notna()
    return (float((got[m] - want[m]).abs().max()), int(m.sum())) if m.any() else (0.0, 0)


def _sided(path):
    """One row per (game_pk, side) from a dump, or None if it predates phases."""
    df = pd.read_csv(path, float_precision="round_trip")
    if "sp_share" not in df.columns:
        return None
    df = df[df["side"].isin(["away", "home"])].copy()
    df["_key"] = list(zip(df["game_pk"], df["side"]))
    return df.drop_duplicates("_key").set_index("_key")


def _nets(edge):
    """net = away-row edge minus home-row edge, mirroring `game_nets`."""
    by = {}
    for (gpk, side), e in edge.items():
        by.setdefault(gpk, {})[side] = e
    out = {}
    for gpk, d in by.items():
        if "away" in d and "home" in d:
            a, h = d["away"], d["home"]
            out[gpk] = (a - h) if pd.notna(a) and pd.notna(h) else np.nan
    return pd.Series(out, dtype="float64")


def paired_arms(lam=BLEND_LAMBDA):
    """(frame, diagnostics) over every slate pairing one wOBA to one xwOBA dump.

    Keyed on the METRIC and never on which arm was primary that day, for the
    reason `shadow_report.build_frame` spells out: the arms swapped sides at
    v11 and keying on primary-vs-shadow would flip the sign of half the sample.
    """
    frames, recon, q_gap = [], 0.0, 0.0
    for d, spath, ppath in sr._slate_dates():
        s, p = _sided(spath), _sided(ppath)
        if s is None or p is None:
            continue
        by = {sr.dump_metric(s, spath): s, sr.dump_metric(p, ppath): p}
        if "wOBA" not in by or "xwOBA" not in by or len(by) < 2:
            continue
        w, x = by["wOBA"], by["xwOBA"]
        keys = w.index.intersection(x.index)
        if not len(keys):
            continue
        w, x = w.loc[keys], x.loc[keys]
        for arm in (w, x):
            recon = max(recon, reconstruction_error(arm)[0])
        qw, qx = _num(w, "sp_share"), _num(x, "sp_share")
        gap = (qw - qx).abs()
        q_gap = max(q_gap, float(gap.max()) if gap.notna().any() else 0.0)
        parts = {c: lam * _num(w, c) + (1.0 - lam) * _num(x, c) for c in RATE_PARTS}
        frames.append(pd.DataFrame({
            "net_b": _nets(edge_from_parts(parts, qx).to_dict()),
            "net_w": _nets(_num(w, "edge_xwOBA").to_dict()),
            "net_x": _nets(_num(x, "edge_xwOBA").to_dict()),
            "date": d,
        }))
    if not frames:
        raise SystemExit("blend_probe: no slate paired a wOBA dump against an "
                         "xwOBA dump; nothing to reconstruct")
    f = pd.concat(frames)
    led = pd.read_csv(sr.LEDGER, float_precision="round_trip")
    led = led.drop_duplicates(subset="game_pk", keep="last").set_index("game_pk")
    g = led.reindex(f.index)
    f["status"] = g["status"]
    f["margin"] = (pd.to_numeric(g["full_home"], errors="coerce")
                   - pd.to_numeric(g["full_away"], errors="coerce"))
    return f, {"reconstruction_error": recon, "sp_share_gap": q_gap,
               "slates": len(frames)}


def scored(f):
    """Graded rows on which all three arms reached a lean."""
    return f[(f["status"] == "graded") & f["margin"].notna()
             & f[["net_b", "net_w", "net_x"]].notna().all(axis=1)].copy()


def _record(net, margin):
    m = (net != 0) & margin.notna() & (margin != 0)
    won = (((net > 0) & (margin > 0)) | ((net < 0) & (margin < 0)))
    w = int(won[m].sum())
    return w, int(m.sum()) - w


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def blend_ceiling(r_w, r_x, rho):
    """(predicted corr of the equal-weight blend, its headroom over the better arm).

    The identity the whole finding turns on: an average of two standardised
    predictors correlates `(r_w + r_x)/sqrt(2 + 2*rho)` with the outcome.
    """
    pred = (r_w + r_x) / np.sqrt(2.0 + 2.0 * rho)
    return pred, max(r_w, r_x) * (1.0 / np.sqrt((1.0 + rho) / 2.0) - 1.0)


def sign_contrast(d, c1, c2):
    """McNemar on the two arms' LEAN calls: (c1-only wins, c2-only wins, z, p).

    `d_corr` is a magnitude-weighted criterion and the one `shadow_report`
    reports.  It is not the criterion this site publishes.  A lean is a SIGN,
    the record is a tally of signs, and every registered rule keys off which
    side the sign picked -- so two arms can carry identical linear information
    and still disagree about games.  Concordant games cancel by construction,
    which is what makes this the paired form: only the games the two arms call
    differently carry any information about which is better at calling them.
    """
    m = (d["margin"] != 0) & d["margin"].notna()
    g = d[m]

    def won(c):
        return (((g[c] > 0) & (g["margin"] > 0))
                | ((g[c] < 0) & (g["margin"] < 0)))

    a, b = won(c1), won(c2)
    a_only, b_only = int((a & ~b).sum()), int((b & ~a).sum())
    n = a_only + b_only
    if not n:
        return a_only, b_only, np.nan, np.nan
    z = (a_only - n / 2.0) / math.sqrt(n / 4.0)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return a_only, b_only, z, p


def _d_corr_boot(d, c1, c2, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    a1, a2 = d[c1].to_numpy(float), d[c2].to_numpy(float)
    mg = d["margin"].to_numpy(float)
    obs = _corr(a1, mg) - _corr(a2, mg)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, len(d), len(d))
        reps[i] = _corr(a1[s], mg[s]) - _corr(a2[s], mg[s])
    return obs, float(np.nanstd(reps)), np.nanpercentile(reps, [2.5, 97.5])


def sweep_lines(d_index):
    """The blend-weight curve, as context. Registers nothing -- see BLEND_LAMBDA."""
    out = ["  lambda = weight on wOBA (0.0 is the shipped xwOBA arm)"]
    for lam in np.round(np.arange(0.0, 1.01, 0.1), 2):
        f, _ = paired_arms(lam=float(lam))
        n = f["net_b"].reindex(d_index)
        m = n.notna()
        w, l = _record(n[m], f["margin"].reindex(d_index)[m])
        out.append(f"    lam={lam:.1f}  corr {_corr(n[m], f['margin'].reindex(d_index)[m]):+.4f}"
                   f"  record {w}-{l} ({w / (w + l):.3f})"
                   f"  median |net| {n[m].abs().median():.5f}")
    return out


def report(n_boot=20000, sweep_only=False):
    f, diag = paired_arms()
    if diag["reconstruction_error"] != 0.0:
        raise SystemExit(
            "blend_probe: refusing to report -- rebuilding a shipped arm's own "
            f"edge_xwOBA from its components was off by "
            f"{diag['reconstruction_error']:.3e}, so the blend built the same "
            "way is not the build either")
    d = scored(f)
    say = print
    say("=" * 64)
    say(f"WOBA/XWOBA BLEND PROBE -- {diag['slates']} paired slate(s), "
        f"{len(d)} graded games with all three arms decided")
    say(f"  reconstruction check: shipped arms' own edge rebuilt exactly "
        f"(max |diff| {diag['reconstruction_error']:.1e})")
    say(f"  sp_share identical across arms (max gap {diag['sp_share_gap']:.1e}) "
        f"-- the workload weight reads no rate")
    if sweep_only:
        say("")
        for line in sweep_lines(d.index):
            say(line)
        say("=" * 64)
        return 0

    say("")
    for name, col in (("wOBA", "net_w"), ("xwOBA (ships)", "net_x"),
                      (f"blend {BLEND_LAMBDA:.2f}/{1 - BLEND_LAMBDA:.2f}", "net_b")):
        w, l = _record(d[col], d["margin"])
        say(f"  {name:<16} record {w}-{l} ({w / (w + l):.3f})   "
            f"corr(net, margin) {_corr(d[col], d['margin']):+.4f}   "
            f"median |net| {d[col].abs().median():.5f}   sd {d[col].std():.5f}")

    r_w, r_x = _corr(d["net_w"], d["margin"]), _corr(d["net_x"], d["margin"])
    rho = _corr(d["net_w"], d["net_x"])
    pred, head = blend_ceiling(r_w, r_x, rho)
    obs = _corr(d["net_b"], d["margin"])
    say("")
    say("  the gain is the noise-averaging identity, not the metrics:")
    say(f"    (r_w + r_x)/sqrt(2+2*rho) with rho(net_w,net_x) {rho:+.4f}"
        f"  ->  predicted {pred:+.4f}")
    say(f"    observed {obs:+.4f}   difference {obs - pred:+.5f}")
    say(f"    ceiling over the better single arm: {head:+.5f} on corr "
        f"(x{1.0 / np.sqrt((1.0 + rho) / 2.0):.4f})")

    say("")
    for lab, c1, c2 in (("blend - xwOBA (ships)", "net_b", "net_x"),
                        ("blend - wOBA", "net_b", "net_w"),
                        ("xwOBA - wOBA", "net_x", "net_w")):
        o, se, ci = _d_corr_boot(d, c1, c2, n_boot)
        say(f"  d_corr {lab:<22} {o:+.4f}  se {se:.4f}  "
            f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        if c1 == "net_b" and c2 == "net_x" and se > 0:
            need = int(len(d) * (se / (head / 2.0)) ** 2) if head > 0 else 0
            say(f"    the ceiling above is {head / se:.2f} se at this n; "
                f"2 se would need ~{need:,} games "
                f"(~{need / max(len(d) / diag['slates'], 1):,.0f} slates)")

    flips = int((np.sign(d["net_b"]) != np.sign(d["net_x"])).sum())
    say("")
    say(f"  lean flips against the shipped arm: {flips} of {len(d)} "
        f"({flips / len(d):.1%})   corr(net_b, net_x) {_corr(d['net_b'], d['net_x']):+.4f}")

    say("")
    say("  the SIGN criterion -- what the record and every registered rule "
        "actually key off:")
    say("    McNemar over the games the two arms call differently; concordant "
        "games cancel")
    for lab, c1, c2 in (("blend over xwOBA (ships)", "net_b", "net_x"),
                        ("blend over wOBA", "net_b", "net_w"),
                        ("xwOBA over wOBA", "net_x", "net_w")):
        a_only, b_only, z, p = sign_contrast(d, c1, c2)
        say(f"    {lab:<25} {a_only} / {b_only} discordant of "
            f"{a_only + b_only:<4} z {z:+.2f}  p {p:.4f}")
    say("    read these against the d_corr block above, NOT instead of it: "
        "they are two")
    say("    criteria on one sample, the second was reached by noticing the "
        "first left a")
    say("    21-game record gap unexplained, and a p here carries that second "
        "look.")
    say("")
    for line in sweep_lines(d.index):
        say(line)
    say("  (context only -- the curve's maximum sits where the identity puts "
        "it whenever the arms score alike, and no weight is registered)")
    say("=" * 64)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--sweep-only", action="store_true")
    a = ap.parse_args(argv)
    return report(n_boot=a.boot, sweep_only=a.sweep_only)


if __name__ == "__main__":
    raise SystemExit(main())
