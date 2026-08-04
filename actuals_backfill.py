"""Attach realised per-game batting actuals to settled ledger rows.

The ledger reduces every game to a binary W/L. That throws away almost all of
the signal the model could be tuned on: it predicts a *rate* (`mx_xwOBA`, an
expected per-PA wOBA for one side's offense) and grades it against who won.
This module stores what that offense actually did, so predicted and actual sit
side by side on the same immutable row.

Backfill, not lookahead. A finished game's box score is an immutable historical
fact -- fetching 2026-07-15 today returns exactly what it returned then. That
is the opposite of re-deriving a *prediction* from today's Savant leaderboard,
which `.savant_cache/` being gitignored makes impossible on purpose. Only the
outcome side is recoverable; the prediction side exists only where the build
already wrote it.

Which is the binding constraint: 389 graded rows, but only 105 carry a stored
`mx_xwOBA`. Rows from v2-v7 predate those columns and are permanently
unpairable no matter how many box scores are fetched.

Joins on the `gamePk` that `market_backfill` already resolved and
score-verified, rather than repeating the date/team join. That verification
correctly rejected an All-Star Game join once; riding it means this module
cannot introduce a second, weaker version of the same check.
"""
import time

import numpy as np
import pandas as pd
import requests

BOX_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"
THROTTLE_S = 0.15
TIMEOUT = 10
TRIES = 2

# This fetches one box score PER ROW, unlike market_backfill which fetches per
# date. The first run has ~105 rows of history to fill, and an upstream hang
# would otherwise cost rows x TRIES x TIMEOUT -- enough to burn the build job's
# 15-minute budget and take the site build down with it, which is exactly what
# that timeout exists to prevent.
#
# So the work is bounded two ways, and neither loses data: unfinished rows are
# simply still null next run, which is the same state they were in before.
#   BUDGET_S       - wall clock, the bound that holds however slow the API is.
#   MAX_CONSEC_FAIL - an outage is one signal, not 105; stop asking.
BUDGET_S = 120.0
MAX_CONSEC_FAIL = 5

# wOBA linear weights. Frozen literals, so: these are the standard weights for
# recent seasons, and what invalidates them is a season whose published weights
# differ -- they drift by roughly 0.005 a year, which moves a computed wOBA by
# under 0.002. They are NOT fetched, because the source is a season-end
# publication and this runs mid-season.
#
# Two consequences to read the outputs with:
#   * the raw components are stored beside the derived rate, so a later weight
#     set can recompute every historical row without refetching anything. That
#     is the whole reason for storing ten columns instead of one.
#   * the model's input is Savant's wOBA, computed with the official yearly
#     weights. If those differ from these, predicted and actual sit on scales
#     offset by a constant. A calibration *slope* is unaffected by that; a
#     calibration *intercept* is not. Read the slope.
WOBA_W = {"bb": 0.690, "hbp": 0.720, "1b": 0.890,
          "2b": 1.271, "3b": 1.616, "hr": 2.101}

# Per batting side. `pa` is the box score's own count, not a reconstruction.
_BAT_FIELDS = ["pa", "ab", "h", "2b", "3b", "hr", "bb", "ibb", "hbp", "sf"]
ACTUAL_BAT_COLS = [f"act_{f}_{s}" for s in ("away", "home") for f in _BAT_FIELDS]
ACTUAL_RATE_COLS = [f"act_woba_{s}" for s in ("away", "home")]
# Pitching actuals are keyed by the side whose STARTER they describe, matching
# `expected_sp_ip_away` / `_home`. They pair directly; the batting columns do
# not (see `paired_rates`).
ACTUAL_PIT_COLS = [f"act_sp_{f}_{s}" for s in ("away", "home") for f in ("ip", "bf")]
ACTUAL_COLS = ACTUAL_BAT_COLS + ACTUAL_RATE_COLS + ACTUAL_PIT_COLS

_SETTLED = ("full_away", "full_home")


def _get_json(url, tries=TRIES):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.4 * (k + 1))
    raise last


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def innings_to_outs(ip):
    """StatsAPI innings-pitched string ('5.2') -> outs. Baseball's .1/.2 are
    thirds, not decimals, so float() is wrong by up to 0.47 innings per start
    and biased in one direction."""
    if ip is None:
        return None
    s = str(ip).strip()
    if not s or s in ("-", "nan", "None"):
        return None
    try:
        whole, _, frac = s.partition(".")
        outs = int(whole or 0) * 3
        if frac:
            f = int(frac[0])
            if f > 2:            # not a thirds notation; refuse rather than guess
                return None
            outs += f
        return outs
    except (TypeError, ValueError):
        return None


def woba_from_components(c):
    """Observed wOBA from box-score components, or None when the denominator
    is unusable. Singles and unintentional walks are derived, not fetched."""
    need = ("ab", "h", "2b", "3b", "hr", "bb", "ibb", "hbp", "sf")
    v = {k: _f(c.get(k)) for k in need}
    if any(x is None for x in v.values()):
        return None
    singles = v["h"] - v["2b"] - v["3b"] - v["hr"]
    ubb = v["bb"] - v["ibb"]
    if singles < 0 or ubb < 0:
        return None
    denom = v["ab"] + ubb + v["sf"] + v["hbp"]
    if denom <= 0:
        return None
    num = (WOBA_W["bb"] * ubb + WOBA_W["hbp"] * v["hbp"] + WOBA_W["1b"] * singles
           + WOBA_W["2b"] * v["2b"] + WOBA_W["3b"] * v["3b"] + WOBA_W["hr"] * v["hr"])
    return num / denom


def parse_boxscore(box):
    """{'away': {...}, 'home': {...}} of batting components + starter line.

    The starter is identified by the box score's own `gamesStarted`, never by
    position in the pitcher list -- an opener is listed first and is still the
    starter, but a reliever promoted mid-series is not.
    """
    out = {}
    teams = (box or {}).get("teams") or {}
    for side in ("away", "home"):
        t = teams.get(side) or {}
        bat = ((t.get("teamStats") or {}).get("batting")) or {}
        rec = {
            "pa": bat.get("plateAppearances"), "ab": bat.get("atBats"),
            "h": bat.get("hits"), "2b": bat.get("doubles"),
            "3b": bat.get("triples"), "hr": bat.get("homeRuns"),
            "bb": bat.get("baseOnBalls"), "ibb": bat.get("intentionalWalks"),
            "hbp": bat.get("hitByPitch"), "sf": bat.get("sacFlies"),
        }
        sp_ip = sp_bf = None
        for p in (t.get("players") or {}).values():
            ps = ((p.get("stats") or {}).get("pitching")) or {}
            if _f(ps.get("gamesStarted")):
                outs = innings_to_outs(ps.get("inningsPitched"))
                sp_ip = None if outs is None else outs / 3.0
                sp_bf = _f(ps.get("battersFaced"))
                break
        rec["sp_ip"], rec["sp_bf"] = sp_ip, sp_bf
        out[side] = rec
    return out


def attach_actuals(df, verbose=True):
    """Idempotently attach realised batting actuals to settled ledger rows.

    Write-once: a row is only filled where the column is currently null, and
    only when the game has a final score. Pending rows are never touched, which
    is the same invariant `run_market_update` holds for closing lines -- an
    actual is an outcome, and an outcome must not reach a row whose prediction
    can still be refreshed.

    Skips are collected in `df.attrs['actual_skips']` as (index, reason).
    """
    for c in ACTUAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    settled = df[_SETTLED[0]].notna() & df[_SETTLED[1]].notna()
    have = df["act_woba_away"].notna() & df["act_woba_home"].notna()
    todo = df.index[settled & ~have & df["gamePk"].notna()]
    no_pk = int((settled & ~have & df["gamePk"].isna()).sum())

    if len(todo) == 0:
        if verbose:
            print(f"actuals backfill: nothing to do"
                  + (f" ({no_pk} settled rows lack a gamePk)" if no_pk else ""))
        df.attrs["actual_skips"] = []
        return df

    skips = []
    n_ok = 0
    consec = 0
    started = time.monotonic()
    stopped = None
    for i in todo:
        if time.monotonic() - started > BUDGET_S:
            stopped = f"time budget {BUDGET_S:.0f}s reached"
            break
        if consec >= MAX_CONSEC_FAIL:
            stopped = f"{consec} consecutive fetch failures"
            break
        gpk = int(df.at[i, "gamePk"])
        try:
            box = _get_json(BOX_URL.format(gamePk=gpk))
        except Exception as e:  # noqa: BLE001
            skips.append((i, f"boxscore fetch failed ({type(e).__name__})"))
            consec += 1
            time.sleep(THROTTLE_S)
            continue
        consec = 0
        parsed = parse_boxscore(box)

        # The gamePk is already score-verified by market_backfill, so what is
        # left to catch is a box score that is present but incomplete -- a
        # suspended game, or one re-fetched mid-flight. Refuse the row rather
        # than store a partial line that would later read as a real actual.
        bad = [s for s in ("away", "home")
               if woba_from_components(parsed[s]) is None]
        if bad:
            skips.append((i, f"incomplete batting line ({','.join(bad)})"))
            time.sleep(THROTTLE_S)
            continue

        for side in ("away", "home"):
            p = parsed[side]
            for f in _BAT_FIELDS:
                df.at[i, f"act_{f}_{side}"] = _f(p.get(f))
            df.at[i, f"act_woba_{side}"] = woba_from_components(p)
            df.at[i, f"act_sp_ip_{side}"] = p.get("sp_ip")
            df.at[i, f"act_sp_bf_{side}"] = p.get("sp_bf")
        n_ok += 1
        time.sleep(THROTTLE_S)

    df.attrs["actual_skips"] = skips
    df.attrs["actual_stopped"] = stopped
    if verbose:
        remaining = len(todo) - n_ok - len(skips)
        print(f"actuals backfill: {n_ok} attached, {len(skips)} skipped"
              + (f", {no_pk} lack a gamePk" if no_pk else "")
              + (f"; STOPPED EARLY ({stopped}), {remaining} rows retry next run"
                 if stopped else ""))
        for i, why in skips[:5]:
            print(f"  SKIP row {i}: {why}")
    return df


# --------------------------------------------------------------- analysis ---
def _num(df, name):
    """Numeric column aligned to `df.index`, all-NaN when absent.

    `df.get(missing)` returns None, and `pd.to_numeric(None)` is a scalar nan,
    not a Series -- so the obvious spelling raises AttributeError on the first
    ledger that predates a column instead of degrading. Every historical
    family here is missing some of these by construction.
    """
    if name not in getattr(df, "columns", []):
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def paired_rates(df):
    """Long frame of (predicted rate, actual rate) for one offense.

    The cross lives here and only here. `mx_xwoba_away` is written on the
    away-STARTER row, and that row carries the **home** offense -- so it pairs
    with `act_woba_home`. Getting this backwards produces a plausible-looking
    near-zero correlation rather than an error, which is why no caller is
    allowed to do the pairing itself.
    """
    rows = []
    for pit_side, bat_side in (("away", "home"), ("home", "away")):
        pred = _num(df, f"mx_xwoba_{pit_side}")
        act = _num(df, f"act_woba_{bat_side}")
        pa = _num(df, f"act_pa_{bat_side}")
        m = pred.notna() & act.notna()
        if not m.any():
            continue
        rows.append(pd.DataFrame({
            "game_pk": (df.loc[m, "game_pk"].to_numpy()
                        if "game_pk" in df.columns else np.nan),
            "model_tag": (df.loc[m, "model_tag"].to_numpy()
                          if "model_tag" in df.columns else None),
            "pitching_side": pit_side, "batting_side": bat_side,
            "pred": pred[m].to_numpy(), "act": act[m].to_numpy(),
            "pa": pa[m].to_numpy(),
        }))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["game_pk", "model_tag", "pitching_side",
                                       "batting_side", "pred", "act", "pa"]))


def paired_sp_ip(df):
    """Long frame of (expected starter IP, actual starter IP). These pair on
    the same side -- both describe that side's starter -- unlike the rates."""
    rows = []
    for side in ("away", "home"):
        pred = _num(df, f"expected_sp_ip_{side}")
        act = _num(df, f"act_sp_ip_{side}")
        m = pred.notna() & act.notna()
        if not m.any():
            continue
        rows.append(pd.DataFrame({"side": side, "pred": pred[m].to_numpy(),
                                  "act": act[m].to_numpy()}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["side", "pred", "act"]))


def calibration(pred, act):
    """OLS slope/intercept of actual on predicted, with the slope's standard
    error. Slope 1 = calibrated; < 1 = predictions too spread; > 1 = too
    compressed, which is what an over-aggressive shrinkage K produces.

    Returns None below three points or with no spread in the predictor -- the
    slope is undefined there, and a fabricated 0.0 would read as a finding.
    """
    x = np.asarray(pred, dtype=float)
    y = np.asarray(act, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or x.std() <= 0:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt((resid @ resid) / (n - 2) / sxx)) if sxx > 0 else float("nan")
    return {"n": int(n), "slope": float(b), "intercept": float(a), "se_slope": se}


def skill_vs_baseline(pred, act, baseline):
    """Fractional MSE reduction against always predicting `baseline`.

    Bounded above by roughly 0.04 here whatever the model does: a team's
    ~38-PA game wOBA has sd ~0.084 against a prediction spread of ~0.017, so
    the irreducible share dominates. A small positive number is the expected
    shape of success, not a failure.
    """
    x = np.asarray(pred, dtype=float)
    y = np.asarray(act, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or baseline is None or not np.isfinite(baseline):
        return None
    mse_m = float(((y - x) ** 2).mean())
    mse_b = float(((y - baseline) ** 2).mean())
    if mse_b <= 0:
        return None
    return {"n": int(x.size), "mse_model": mse_m, "mse_baseline": mse_b,
            "skill": 1.0 - mse_m / mse_b}


def actuals_family_line(label, fam):
    """One compact predicted-vs-actual line for a past prediction family.

    Families are never pooled -- v9/v10 predicts from xwOBA inputs while
    wOBA v1 predicts from observed wOBA, and the actual here is observed wOBA,
    so the two do not even sit on the same scale against it. But the current
    family is n=1 for a long while after every bump, and the 97-game v9/v10
    sample is the only one large enough to read. Showing them separately is
    how the record block already handles exactly this, and it keeps the useful
    sample visible without making it part of a pooled number.
    """
    rates = paired_rates(fam)
    ip = paired_sp_ip(fam)
    if rates.empty and ip.empty:
        return None
    # Each figure carries its OWN n. The two do not move together: v6 and v7
    # stored expected_sp_ip but no mx, so their IP bias is real while their
    # rate count is zero. One shared n would have labelled a measured IP bias
    # with a sample size it was not computed from.
    bits = [f"  {label:9}"]
    if not rates.empty:
        cal = calibration(rates["pred"], rates["act"])
        bits.append(f"wOBA n={len(rates):3d} pred {rates['pred'].mean():.4f} "
                    f"act {rates['act'].mean():.4f} "
                    f"({rates['act'].mean() - rates['pred'].mean():+.4f})"
                    + (f" slope {cal['slope']:+.2f}±{cal['se_slope']:.2f}" if cal else ""))
    if not ip.empty:
        bits.append(f"SP IP n={len(ip):3d} bias "
                    f"{float((ip['act'] - ip['pred']).mean()):+.2f}")
    return "  ".join(bits)


def actuals_summary(df, baseline=None, tags=None):
    """Report lines for predicted-vs-actual. Empty list when nothing pairs.

    Pass the WHOLE ledger. The two metrics take different scopes on purpose:

      rates  -- scoped to `tags` (the record family). v9/v10 predicts from
                xwOBA inputs and wOBA v1 from observed wOBA; against an
                observed-wOBA actual those are not the same measurement, so
                pooling them would describe a model that never existed.
      IP     -- pooled over every family. `expected_pitcher_ip` is one
                estimator, unchanged since v6, so every row measures the same
                thing. Scoping it to the record family would have shown n=2
                for months after each bump and hidden the 306-side-game slope
                that is the whole reason the line exists.

    The per-family lines below still break IP out, so a family-specific change
    to the workload estimator would show up rather than being averaged away.
    """
    if tags is None or "model_tag" not in getattr(df, "columns", []):
        d = df
    else:
        d = df[df["model_tag"].astype(str).isin(set(tags))]
    ip = paired_sp_ip(df)          # pooled -- see above
    rates = paired_rates(d)        # family-scoped
    if ip.empty and rates.empty:
        return []

    lines = ["predicted vs actual (backfilled box scores)"]
    if not ip.empty:
        bias = float((ip["act"] - ip["pred"]).mean())
        mae = float((ip["act"] - ip["pred"]).abs().mean())
        lines.append(f"  starter IP   n={len(ip):<4d} bias {bias:+.2f} IP  "
                     f"MAE {mae:.2f} IP   (drives the phase weight q)")
        # The slope is the finding, not the bias. On 2026-08-04 it measured
        # 0.756 +/- 0.063 over 306 side-games -- 3.9 se below 1.0, i.e.
        # expected_sp_ip is over-dispersed, pushing too far from the mean in
        # both directions. Bias over the same rows was +0.10 IP (t=1.31): a
        # spread problem, not a level one. Printing it every build is what
        # makes the deferred re-fit surface on its own rather than depending
        # on anyone remembering. See CLAUDE.md for the decision and its gate.
        cal = calibration(ip["pred"], ip["act"])
        if cal:
            lines.append(f"    IP calibration slope {cal['slope']:+.3f} "
                         f"+/- {cal['se_slope']:.3f} (1.00 = calibrated; "
                         f"< 1 = over-dispersed)")
    if not rates.empty:
        n = len(rates)
        lines.append(f"  offense wOBA n={n:<4d} pred mean {rates['pred'].mean():.4f}  "
                     f"actual mean {rates['act'].mean():.4f}  "
                     f"({rates['act'].mean() - rates['pred'].mean():+.4f})")
        cal = calibration(rates["pred"], rates["act"])
        if cal:
            # se is what says whether the slope means anything yet; a slope
            # printed without it invites reading noise as miscalibration.
            lines.append(f"    calibration slope {cal['slope']:+.2f} "
                         f"+/- {cal['se_slope']:.2f} (1.00 = calibrated)")
            if cal["se_slope"] > 0.25:
                lines.append(f"    UNDER-POWERED: se {cal['se_slope']:.2f} cannot "
                             f"separate 1.00 from 0.80; needs ~6400 side-games")
        sk = skill_vs_baseline(rates["pred"], rates["act"], baseline)
        if sk:
            lines.append(f"    skill vs league-rate baseline {sk['skill']:+.4f} "
                         f"(ceiling ~0.04; per-game noise dominates)")
    return lines
