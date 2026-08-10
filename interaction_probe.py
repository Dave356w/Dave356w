"""Do individual signals beat the shipped interaction, and do alternative
combinations of the SAME signals beat it?

The shipped rule (build_site.sequential_xwoba_phases + matchup_value):

    mx_sp = B_vs_sp * P_sp / L          log5 / odds-ratio form
    mx_bp = B_neutral * P_bp / L
    mx    = q*mx_sp + (1-q)*mx_bp       q = PA share from expected_sp_ip
    edge  = mx - L
    net   = home_off_edge - away_off_edge

`rebuild check` in run() asserts the reconstruction reproduces the ledger's
own xw_net to float precision before any candidate is scored -- without that
this file would be benchmarking against something the model never ran.

Scored on a fixed row set against a fixed target, so the candidates' differing
scales do not matter: every metric is rank- or sign-based. Orientation is
"what the HOME offense faces minus what the AWAY offense faces" throughout,
matching grade_leans' d_sp = P_awaySP - P_homeSP.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260810)


def load(tags=None, metric=None):
    d = pd.read_csv("data/mlb_lean_ledger.csv")
    d = d[d["status"] == "graded"]
    if tags is not None:
        d = d[d["model_tag"].isin(tags)]
    if metric is not None:
        d = d[d["model_metric"] == metric]
    n = lambda c: pd.to_numeric(d[c], errors="coerce")

    # League baseline is recoverable per row: mx_sp - edge_sp.
    L = n("mx_xwoba_sp_away") - n("edge_xwoba_sp_away")

    f = pd.DataFrame({
        "game_pk": d["game_pk"].values,
        "tag": d["model_tag"].values,
        # home offense faces the AWAY pitching staff -> _away columns
        "Bh_sp": n("opp_xwoba_vs_sp_away"), "Bh_nu": n("opp_xwoba_neutral_away"),
        "Ph_sp": n("starter_xwoba_away"),   "Ph_bp": n("bullpen_xwoba_away"),
        "qh": n("sp_share_away"), "iph": n("expected_sp_ip_away"),
        # away offense faces the HOME staff
        "Ba_sp": n("opp_xwoba_vs_sp_home"), "Ba_nu": n("opp_xwoba_neutral_home"),
        "Pa_sp": n("starter_xwoba_home"),   "Pa_bp": n("bullpen_xwoba_home"),
        "qa": n("sp_share_home"), "ipa": n("expected_sp_ip_home"),
        "L": L,
        "net_ship": n("xw_net"),
        "act": n("act_woba_home") - n("act_woba_away"),
        "full_home": n("full_home"), "full_away": n("full_away"),
        "close_p_home": n("close_p_home"),
    })
    f["home_won"] = f["full_home"] > f["full_away"]
    played = f["full_home"].notna() & f["full_away"].notna() & (f["full_home"] != f["full_away"])
    need = ["Bh_sp", "Bh_nu", "Ph_sp", "Ph_bp", "qh",
            "Ba_sp", "Ba_nu", "Pa_sp", "Pa_bp", "qa", "L", "act"]
    return f[played & f[need].notna().all(axis=1)].reset_index(drop=True)


# ---------------------------------------------------------------- candidates
def log5(B, P, L):
    return B * P / L


def add(B, P, L):
    return B + P - L


def blend(f, comb, qh, qa):
    """One side's full-game matchup value, both sides, differenced."""
    h = qh * comb(f.Bh_sp, f.Ph_sp, f.L) + (1 - qh) * comb(f.Bh_nu, f.Ph_bp, f.L)
    a = qa * comb(f.Ba_sp, f.Pa_sp, f.L) + (1 - qa) * comb(f.Ba_nu, f.Pa_bp, f.L)
    return h - a  # edge subtracts L on both sides, cancels in the difference


def candidates(f):
    C = {}
    # --- the shipped rule, rebuilt from parts (validated against xw_net)
    C["CURRENT log5, q=PA share"] = blend(f, log5, f.qh, f.qa)

    # --- same signals, different interaction FORM
    C["additive B+P-L"] = blend(f, add, f.qh, f.qa)

    # --- same signals, different PHASE WEIGHT
    C["q fixed 0.5"] = blend(f, log5, 0.5, 0.5)
    C["q = pooled mean"] = blend(f, log5, f.qh.mean(), f.qa.mean())
    qs = 0.756  # the measured IP calibration slope, shrunk toward the mean
    C["q shrunk x0.756"] = blend(
        f, log5,
        f.qh.mean() + qs * (f.qh - f.qh.mean()),
        f.qa.mean() + qs * (f.qa - f.qa.mean()))
    C["q = 1 (starter only)"] = blend(f, log5, 1.0, 1.0)
    C["q = 0 (bullpen only)"] = blend(f, log5, 0.0, 0.0)

    # --- drop the platoon adjustment (neutral lineup in both phases)
    C["no platoon adj"] = (
        f.qh * log5(f.Bh_nu, f.Ph_sp, f.L) + (1 - f.qh) * log5(f.Bh_nu, f.Ph_bp, f.L)
        - (f.qa * log5(f.Ba_nu, f.Pa_sp, f.L) + (1 - f.qa) * log5(f.Ba_nu, f.Pa_bp, f.L)))

    # --- SINGLE SIGNALS (no interaction at all)
    # Orientation throughout: "what the HOME offense faces" minus "what the AWAY
    # offense faces", so positive favours home and matches the target
    # act_woba_home - act_woba_away. The home offense faces the AWAY staff, i.e.
    # the _away-suffixed pitching columns (grade_leans: d_sp = P_awaySP - P_homeSP).
    pit_h = f.qh * f.Ph_sp + (1 - f.qh) * f.Ph_bp   # staff home offense faces
    pit_a = f.qa * f.Pa_sp + (1 - f.qa) * f.Pa_bp   # staff away offense faces
    C["signal: lineup only"] = f.Bh_nu - f.Ba_nu
    C["signal: starter only"] = f.Ph_sp - f.Pa_sp
    C["signal: bullpen only"] = f.Ph_bp - f.Pa_bp
    C["signal: pitching only (q-wtd)"] = pit_h - pit_a
    C["signal: expected IP only"] = f.iph - f.ipa

    # --- naive sum of the two halves, no multiplicative interaction
    C["lineup + pitching (sum)"] = (f.Bh_nu - f.Ba_nu) + (pit_h - pit_a)
    return {k: pd.Series(v, index=f.index).astype(float) for k, v in C.items()}


# ------------------------------------------------------------------ scoring
def score(pred, f):
    m = pred.notna()
    p, a, hw = pred[m], f["act"][m], f["home_won"][m]
    corr = np.corrcoef(p, a)[0, 1] if p.std() > 0 else np.nan
    sign = float((np.sign(p) == np.sign(a)).mean())
    wl = float((( p > 0) == hw)[p != 0].mean())
    return {"n": int(m.sum()), "corr": corr, "sign": sign, "wl": wl}


def _boot_corr(x, y, S):
    """Correlation of x,y over each bootstrap index row of S, vectorised."""
    X, Y = x[S], y[S]
    Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    num = (Xc * Yc).sum(1)
    den = np.sqrt((Xc**2).sum(1) * (Yc**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def boot_delta(pred, base, f, key="corr", B=4000):
    """Paired bootstrap CI on (candidate - current) correlation."""
    m = (pred.notna() & base.notna()).to_numpy()
    p, b = pred.to_numpy()[m], base.to_numpy()[m]
    a = f["act"].to_numpy()[m]
    S = RNG.integers(0, len(a), size=(B, len(a)))
    d = _boot_corr(p, a, S) - _boot_corr(b, a, S)
    d = d[np.isfinite(d)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def run(label, f):
    print(f"\n{'='*88}\n{label}   n={len(f)} games\n{'='*88}")
    # sanity: does the rebuilt current rule reproduce the shipped xw_net?
    reb = blend(f, log5, f.qh, f.qa)
    ok = f["net_ship"].notna() & reb.notna()
    if ok.any():
        r = np.corrcoef(reb[ok], f["net_ship"][ok])[0, 1]
        mad = float((reb[ok] - f["net_ship"][ok]).abs().max())
        print(f"rebuild check vs shipped xw_net: corr {r:.6f}  max|diff| {mad:.2e}")
    C = candidates(f)
    base = C["CURRENT log5, q=PA share"]
    print(f"\n{'candidate':32s} {'n':>4s} {'corr':>7s} {'sign':>6s} {'W-L':>6s}   "
          f"{'d_corr 95% CI':>22s}")
    rows = []
    for k, v in C.items():
        s = score(v, f)
        if k == "CURRENT log5, q=PA share":
            ci = "        (baseline)"
        else:
            lo, hi = boot_delta(v, base, f, "corr")
            star = " *" if (lo > 0 or hi < 0) else ""
            ci = f"[{lo:+.3f}, {hi:+.3f}]{star}"
        print(f"{k:32s} {s['n']:4d} {s['corr']:+7.3f} {s['sign']:6.3f} {s['wl']:6.3f}   {ci:>22s}")
        rows.append((k, s))
    return rows


# --------------------------- fitted alternatives

import numpy as np
import pandas as pd



def parts(f):
    """The three component signals, all oriented home-favouring."""
    pit_h = f.qh * f.Ph_sp + (1 - f.qh) * f.Ph_bp
    pit_a = f.qa * f.Pa_sp + (1 - f.qa) * f.Pa_bp
    return pd.DataFrame({
        "lineup": f.Bh_nu - f.Ba_nu,
        "starter": f.Ph_sp - f.Pa_sp,
        "bullpen": f.Ph_bp - f.Pa_bp,
        "pitching": pit_h - pit_a,
    })


def cv_fit(X, y, folds=10, reps=20):
    """Repeated k-fold CV. Returns out-of-sample predictions' corr, averaged
    over reps, plus the spread across reps (variance is the point)."""
    X = np.column_stack([np.ones(len(X)), X])
    out = []
    for _ in range(reps):
        order = RNG.permutation(len(y))
        pred = np.full(len(y), np.nan)
        for k in range(folds):
            te = order[k::folds]
            tr = np.setdiff1d(order, te)
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pred[te] = X[te] @ beta
        out.append(np.corrcoef(pred, y)[0, 1])
    return float(np.mean(out)), float(np.std(out))


def run_fitted(label, f):
    print(f"\n{'='*84}\n{label}   n={len(f)}\n{'='*84}")
    P = parts(f)
    y = f["act"].to_numpy()
    base = blend(f, log5, f.qh, f.qa).to_numpy()
    print(f"shipped rule (no fitting)            corr {np.corrcoef(base, y)[0,1]:+.3f}")
    print()
    print("cross-validated refits of the SAME signals (10-fold x 20 reps):")
    for name, cols in [("lineup+starter+bullpen", ["lineup", "starter", "bullpen"]),
                       ("lineup+pitching", ["lineup", "pitching"]),
                       ("starter alone", ["starter"]),
                       ("pitching alone", ["pitching"]),
                       ("all + shipped net", None)]:
        if cols is None:
            X = np.column_stack([P[["lineup", "starter", "bullpen"]].to_numpy(), base])
        else:
            X = P[cols].to_numpy()
        m, s = cv_fit(X, y)
        print(f"  {name:24s} OOS corr {m:+.3f}  (sd across reps {s:.3f})")

    # in-sample fitted weights, for direction only
    X = np.column_stack([np.ones(len(P)), P[["lineup", "starter", "bullpen"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sig2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    print("\nin-sample weights (direction only, not a proposal):")
    for nm, b, s in zip(["const", "lineup", "starter", "bullpen"], beta, se):
        print(f"  {nm:9s} {b:+8.3f} +/- {s:.3f}  (t {b/s:+.2f})")

    # how big is the multiplicative interaction, really?
    inter_h = (f.Bh_sp - f.L) * (f.Ph_sp - f.L) / f.L
    print(f"\nlog5 minus additive, per side: (B-L)(P-L)/L")
    print(f"  mean {inter_h.mean():+.6f}  sd {inter_h.std():.6f}  "
          f"max|.| {inter_h.abs().max():.6f}")
    d = base - blend(f, add, f.qh, f.qa).to_numpy()
    print(f"  net difference log5 vs additive: sd {d.std():.6f} "
          f"against net sd {base.std():.6f}  ({d.std()/base.std():.1%})")

    # market reference: what a genuinely informative signal looks like here
    m = f["close_p_home"].notna()
    if m.sum() > 10:
        r = np.corrcoef(f["close_p_home"][m], f["act"][m])[0, 1]
        print(f"\nreference — devigged closing line vs same target: corr {r:+.3f} (n={int(m.sum())})")




if __name__ == "__main__":
    for label, kw in [("xwOBA v9/v10", dict(tags=("xw+plat_consol_v9",
                                                 "xw+plat_consol_v10"))),
                      ("wOBA lineage (live)", dict(metric="wOBA"))]:
        f = load(**kw)
        run(label, f)
        run_fitted(label, f)
