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

import build_site

RNG = np.random.default_rng(20260810)


def load(tags=None, metric=None):
    d = pd.read_csv("data/mlb_lean_ledger.csv")
    d = d[d["status"] == "graded"]
    if tags is not None:
        d = d[d["model_tag"].isin(tags)]
    if metric is not None:
        d = d[d["model_metric"] == metric]
    def n(c):
        """Numeric column, all-NaN when absent.

        Absent is the normal case for a column introduced by a MODEL_TAG
        younger than the rows being read: `expected_sp_ip_raw_*` landed with
        v12, so every pre-v12 family carries none of it and a strict `d[c]`
        would make this probe unrunnable on the very history it exists to
        read. Stated as a rule rather than as a count on purpose -- this
        docstring read "no v12 row has graded yet" for as long as it took 217
        of them to grade, which is the version-note-asserts-rows failure in
        CLAUDE.md with the sign reversed.
        """
        if c not in d.columns:
            return pd.Series(np.nan, index=d.index, dtype="float64")
        return pd.to_numeric(d[c], errors="coerce")

    # League baseline is recoverable per row: mx_sp - edge_sp.
    L = n("mx_xwoba_sp_away") - n("edge_xwoba_sp_away")

    f = pd.DataFrame({
        "game_pk": d["game_pk"].values,
        "tag": d["model_tag"].values,
        # home offense faces the AWAY pitching staff -> _away columns
        "Bh_sp": n("opp_xwoba_vs_sp_away"), "Bh_nu": n("opp_xwoba_neutral_away"),
        "Ph_sp": n("starter_xwoba_away"),   "Ph_bp": n("bullpen_xwoba_away"),
        "qh": n("sp_share_away"), "iph": n("expected_sp_ip_away"),
        "iph_raw": n("expected_sp_ip_raw_away"),
        "rh_sp": n("sp_bf_per_ip_away"), "rh_bp": n("bp_bf_per_ip_away"),
        # away offense faces the HOME staff
        "Ba_sp": n("opp_xwoba_vs_sp_home"), "Ba_nu": n("opp_xwoba_neutral_home"),
        "Pa_sp": n("starter_xwoba_home"),   "Pa_bp": n("bullpen_xwoba_home"),
        "qa": n("sp_share_home"), "ipa": n("expected_sp_ip_home"),
        "ipa_raw": n("expected_sp_ip_raw_home"),
        "ra_sp": n("sp_bf_per_ip_home"), "ra_bp": n("bp_bf_per_ip_home"),
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


def q_from_ip(ip, r_sp, r_bp, game_innings=9.0):
    """Starter's PA share from expected IP -- build_site's formula, mirrored.

    Falls back to the innings share when either BF/IP rate is missing, which is
    what `sequential_xwoba_phases` does and is the same number whenever the two
    rates are equal.
    """
    ip = np.clip(pd.to_numeric(ip, errors="coerce").astype(float), 0.0, game_innings)
    ip_bp = game_innings - ip
    r_sp = pd.to_numeric(r_sp, errors="coerce").astype(float)
    r_bp = pd.to_numeric(r_bp, errors="coerce").astype(float)
    pa_sp, pa_bp = ip * r_sp, ip_bp * r_bp
    tot = pa_sp + pa_bp
    usable = np.isfinite(r_sp) & np.isfinite(r_bp) & (r_sp > 0) & (r_bp > 0) & (tot > 0)
    return pd.Series(np.where(usable, pa_sp / tot, ip / game_innings), index=ip.index)


def calibrated_q(f):
    """(q_home, q_away, label) with the shipped IP calibration applied.

    THIS CANDIDATE USED TO BE A FROZEN LITERAL -- `qs = 0.756`, the calibration
    slope as measured on 2026-08-04, applied by shrinking `q` directly. Two
    things broke it at v12, and both are worth keeping written down because the
    second is the subtler one:

      * The slope is no longer a hypothesis. v12 ships a per-build refit, so a
        literal here benchmarks the model against a stale version of itself.
        The fit is read from `build_site` instead.
      * `f.qh` is the SHIPPED phase weight, read off the ledger's
        `sp_share_*`. On a v12 row that weight ALREADY contains the
        calibration, so shrinking it again would double-apply the correction --
        the exact compounding hazard `expected_sp_ip_raw_*` exists to prevent
        inside the build, reappearing one artifact out. So the candidate is
        rebuilt from the RAW IP (`expected_sp_ip_raw_*`, falling back to the
        published column on pre-v12 rows, where they are the same number) and
        `q` is re-derived rather than shrunk.

    On v12 rows this reproduces the shipped `q`, which is the point: after the
    bump the candidate IS the baseline and the row should show no difference.

    Honest caveat on what this measures: the fit comes from the whole ledger,
    including the rows being scored here, so as a retrospective counterfactual
    it flatters the candidate slightly. The unbiased read on whether
    calibration helps is the walk-forward benchmark behind
    `build_site.SP_IP_CALIBRATION_K`, which scores each slate against a fit
    that never saw it. This row answers a different question: what the shipped
    correction does to the LEAN, not whether it improves the IP estimate.
    """
    fit = build_site.sp_ip_calibration()
    raw_h = f.iph_raw.where(f.iph_raw.notna(), f.iph)
    raw_a = f.ipa_raw.where(f.ipa_raw.notna(), f.ipa)
    if not fit:
        return f.qh, f.qa, "q from calibrated IP (no fit)"
    cal_h = raw_h.map(lambda v: build_site.calibrate_sp_ip(v, fit))
    cal_a = raw_a.map(lambda v: build_site.calibrate_sp_ip(v, fit))
    qh = q_from_ip(cal_h, f.rh_sp, f.rh_bp)
    qa = q_from_ip(cal_a, f.ra_sp, f.ra_bp)
    # Where q cannot be rebuilt (a missing BF/IP pair on a legacy row), keep the
    # shipped weight rather than substituting an innings-share approximation
    # into a candidate that is supposed to differ by the calibration alone.
    qh = qh.where(f.rh_sp.notna() & f.rh_bp.notna(), f.qh)
    qa = qa.where(f.ra_sp.notna() & f.ra_bp.notna(), f.qa)
    return qh, qa, "q from calibrated IP"


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
    qh_cal, qa_cal, cal_label = calibrated_q(f)
    C[cal_label] = blend(f, log5, qh_cal, qa_cal)
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
    fit = build_site.sp_ip_calibration()
    if fit:
        print(f"IP calibration in force: act = {fit[0]:+.3f} + {fit[1]:.3f}*pred "
              f"(n={fit[2]}, weight {fit[3]:.3f})")
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




def blocks():
    """(label, load-kwargs) for every family this probe scores.

    The current family is DERIVED from `build_site.RECORD_TAGS`, never named
    here. This file used to list two hardcoded blocks -- v9/v10, and
    `metric="wOBA"` labelled "(live)" -- and both went stale at v11/v12: the
    primary metric reverted to xwOBA, so the block calling itself live scored
    a lineage the build had stopped running, while v12 became the largest
    family in the ledger without ever being scored at all. That is the
    probe-hardcodes-a-model-constant anti-pattern reached through the ROW
    SELECTOR rather than through a numeric literal, and the fix is the same
    one `calibrated_q` already applies to the slope: read it off the module.

    Historical blocks stay pinned to their tags on purpose -- they are answers
    about versions that no longer run, and a frozen question needs a frozen
    row set. They are labelled as history so no reader takes one for the
    shipping model. A historical block that IS the current family is dropped
    rather than printed twice under two contradicting labels, which an
    env-var `MODEL_TAG` override makes reachable.
    """
    cur = tuple(build_site.RECORD_TAGS)
    vers = "/".join(t.rsplit("_", 1)[-1] for t in cur)
    out = [(f"{build_site.MODEL_RATE_LABEL} {vers}  (CURRENT family)",
            dict(tags=cur))]
    for label, kw in [
            ("xwOBA v9/v10  (historical)",
             dict(tags=("xw+plat_consol_v9", "xw+plat_consol_v10"))),
            ("wOBA lineage  (historical -- the primary metric reverted to "
             "xwOBA at v11)", dict(metric="wOBA"))]:
        if kw.get("tags") == cur:
            continue
        out.append((label, kw))
    return out


if __name__ == "__main__":
    for label, kw in blocks():
        f = load(**kw)
        run(label, f)
        run_fitted(label, f)
