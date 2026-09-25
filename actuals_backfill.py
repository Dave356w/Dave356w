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
import math
import time

import numpy as np
import pandas as pd
import requests

from market_backfill import metric_series

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

# The starter's ALLOWED line, same fields as a batting line and keyed the same
# way as act_sp_ip: by the side whose starter it describes.
#
# This exists so the three model components can be scored separately. Until
# now the only actual was a team's whole-game offense, which is the JOINT
# outcome of that lineup against the opposing starter AND bullpen -- so a
# starter estimate and a bullpen estimate could never be wrong in measurable
# ways, only jointly. Splitting the phase is what makes them separable.
#
# The bullpen phase is deliberately NOT stored: it is the opposing side's
# batting line minus this, exactly, because a game's batters are faced by the
# starter or by a reliever and by nobody else. Storing it would be a second
# copy of a subtraction. But CLAUDE.md's rule applies -- a count derived by
# subtracting cannot carry a name you did not measure -- so `phase_lines()`
# validates the residual is non-negative in every field before it will call it
# a bullpen line, rather than assuming the identity holds.
_SP_ALLOWED_FIELDS = ["ab", "h", "2b", "3b", "hr", "bb", "ibb", "hbp", "sf"]
ACTUAL_SP_LINE_COLS = [f"act_sp_{f}_{s}"
                       for s in ("away", "home") for f in _SP_ALLOWED_FIELDS]

# Bumped whenever this module learns to store a new actual. attach_actuals
# treats a row as done only at the current schema, so rows backfilled under an
# older one are revisited and topped up instead of being stranded by a
# write-once gate that only ever asked about act_woba.
ACT_SCHEMA = 2
ACTUAL_META_COLS = ["act_schema"]

ACTUAL_COLS = (ACTUAL_BAT_COLS + ACTUAL_RATE_COLS + ACTUAL_PIT_COLS
               + ACTUAL_SP_LINE_COLS + ACTUAL_META_COLS)

# Side-games needed to separate a 0.15 correlation at 80% power, from
# dispersion_probe.n_for_r(). Stated so the report can mark its own read
# under-powered rather than printing a slope that reads as a verdict; the probe
# is the authority and a test pins the two together.
DISPERSION_N_MIN = 347

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
        sp_line = None
        for p in (t.get("players") or {}).values():
            ps = ((p.get("stats") or {}).get("pitching")) or {}
            if _f(ps.get("gamesStarted")):
                outs = innings_to_outs(ps.get("inningsPitched"))
                sp_ip = None if outs is None else outs / 3.0
                sp_bf = _f(ps.get("battersFaced"))
                # The allowed line, read from the same object. Field
                # availability on a pitching split is NOT assumed -- a missing
                # `doubles` would otherwise silently roll extra-base hits into
                # singles and bias every starter's allowed rate downward. All
                # or nothing: one absent field abandons the line, and
                # attach_actuals counts how often that happens so a schema
                # that does not carry these surfaces as a number rather than
                # as quietly-wrong rates.
                got = {f: _f(ps.get(_BOX_PIT_KEY[f])) for f in _SP_ALLOWED_FIELDS}
                if all(v is not None for v in got.values()):
                    sp_line = got
                break
        rec["sp_ip"], rec["sp_bf"] = sp_ip, sp_bf
        rec["sp_line"] = sp_line
        out[side] = rec
    return out


# Box-score pitching keys for the allowed line. Spelled out rather than reusing
# the batting names because only some of them coincide.
_BOX_PIT_KEY = {"ab": "atBats", "h": "hits", "2b": "doubles", "3b": "triples",
                "hr": "homeRuns", "bb": "baseOnBalls",
                "ibb": "intentionalWalks", "hbp": "hitByPitch",
                "sf": "sacFlies"}


def phase_lines(row, side):
    """(starter-allowed, bullpen-allowed) component dicts for one side's staff.

    `side` names the PITCHING side, matching act_sp_ip_{side}. The batters they
    faced are the other side's, so the bullpen residual is taken against that
    side's batting line.

    Returns (None, None) when the starter line is absent, and (sp, None) when
    the residual is not a valid line -- a negative count in any field means the
    starter line and the team batting line disagree, and a bullpen rate built
    on that would be fiction. Never guesses.
    """
    bat_side = "home" if side == "away" else "away"
    sp = {}
    for f in _SP_ALLOWED_FIELDS:
        v = _f(row.get(f"act_sp_{f}_{side}"))
        if v is None:
            return None, None
        sp[f] = v
    bp = {}
    for f in _SP_ALLOWED_FIELDS:
        team = _f(row.get(f"act_{f}_{bat_side}"))
        if team is None:
            return sp, None
        r = team - sp[f]
        if r < 0:
            return sp, None
        bp[f] = r
    return sp, bp


def _fill(df, i, col, val):
    """Write only into a null cell. An actual is immutable once recorded.

    The module has always documented write-once, but it was enforced by the
    todo gate rather than by the writes: the loop assigned unconditionally and
    was simply never handed a row that already had values. Adding a schema
    stamp makes revisiting normal, which turned that into a live overwrite --
    caught by the test that asserts existing actuals survive. Enforced here now,
    where the claim is made, so the gate and the guarantee are independent.
    """
    if val is None:
        return
    cur = df.at[i, col] if col in df.columns else None
    if cur is None or (isinstance(cur, float) and np.isnan(cur)) or pd.isna(cur):
        df.at[i, col] = val


def _settled_mask(df):
    return df[_SETTLED[0]].notna() & df[_SETTLED[1]].notna()


def _done_mask(df):
    """Backfilled at the CURRENT schema -- not merely "has a wOBA".

    The original gate asked only about act_woba, which would strand every
    column added later on exactly the rows that already had one, i.e. all of
    them. Revisiting is cheap and safe: the writes are fill-if-null, and a box
    score is a timestamped event rather than a leaderboard, so re-reading one
    is backfill and not lookahead. The stamp is written even when a box score
    cannot supply the newer fields, so a game that simply lacks them is not
    re-fetched on every build forever.
    """
    schema = (pd.to_numeric(df["act_schema"], errors="coerce")
              if "act_schema" in df.columns
              else pd.Series(np.nan, index=df.index))
    return (df["act_woba_away"].notna() & df["act_woba_home"].notna()
            & (schema >= ACT_SCHEMA))


def _todo_index(df):
    """Rows a backfill run should attempt, in ledger order."""
    return df.index[_settled_mask(df) & ~_done_mask(df) & df["gamePk"].notna()]


def attach_actuals(df, verbose=True):
    """Idempotently attach realised batting actuals to settled ledger rows.

    Write-once: a row is only filled where the column is currently null, and
    only when the game has a final score. Pending rows are never touched, which
    is the same invariant `run_market_update` holds for closing lines -- an
    actual is an outcome, and an outcome must not reach a row whose prediction
    can still be refreshed.

    Skips are collected in `df.attrs['actual_skips']` as (index, reason).
    """
    missing = [c for c in ACTUAL_COLS if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing)],
            axis=1,
        )

    todo = _todo_index(df)
    no_pk = int((_settled_mask(df) & ~_done_mask(df) & df["gamePk"].isna()).sum())

    if len(todo) == 0:
        if verbose:
            print(f"actuals backfill: nothing to do"
                  + (f" ({no_pk} settled rows lack a gamePk)" if no_pk else ""))
        df.attrs["actual_skips"] = []
        return df

    skips = []
    n_ok = 0
    consec = 0
    no_sp_line = 0
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
                _fill(df, i, f"act_{f}_{side}", _f(p.get(f)))
            _fill(df, i, f"act_woba_{side}", woba_from_components(p))
            _fill(df, i, f"act_sp_ip_{side}", p.get("sp_ip"))
            _fill(df, i, f"act_sp_bf_{side}", p.get("sp_bf"))
            line = p.get("sp_line")
            if line is None:
                no_sp_line += 1
            else:
                for f, v in line.items():
                    _fill(df, i, f"act_sp_{f}_{side}", v)
        # Stamped even when a starter line was unavailable: the row is as
        # complete as this box score allows, and leaving it unstamped would
        # re-fetch it on every future build.
        df.at[i, "act_schema"] = ACT_SCHEMA
        n_ok += 1
        time.sleep(THROTTLE_S)

    df.attrs["actual_skips"] = skips
    df.attrs["actual_stopped"] = stopped
    if verbose:
        remaining = len(todo) - n_ok - len(skips)
        print(f"actuals backfill: {n_ok} attached, {len(skips)} skipped"
              + (f", {no_sp_line} without a starter allowed line" if no_sp_line else "")
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
        # `opp_xwoba_sd_<pit_side>` and `lineup_savant_backfill_<pit_side>` are
        # written on the same row as `mx_xwoba_<pit_side>` and describe the same
        # offense, so they take the identical cross. They ride along here rather
        # than being re-derived by a caller, for the reason in the docstring: a
        # wrong cross yields a plausible near-zero correlation, not an error.
        # `backfill` matters because a Savant-missing hitter carries the team
        # aggregate and sits at the mean by construction, deflating `sd`.
        sd = _num(df, f"opp_xwoba_sd_{pit_side}")
        bfill = _num(df, f"lineup_savant_backfill_{pit_side}")
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
            "sd": sd[m].to_numpy(), "backfill": bfill[m].to_numpy(),
        }))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["game_pk", "model_tag", "pitching_side",
                                       "batting_side", "pred", "act", "pa",
                                       "sd", "backfill"]))


def paired_net(df):
    """Game-level frame of (predicted lean delta, realised wOBA differential).

    `xw_net = home_off_edge - away_off_edge` (grade_leans), so a positive value
    favours the HOME offense and pairs with `act_woba_home - act_woba_away`.
    This is one game-level row, not two side rows: the lean is a difference, and
    differencing two side rows that were themselves crossed is where the sign
    goes wrong. paired_rates owns the side cross; this owns the difference.

    Why it is worth its own pairing rather than reading off paired_rates: the
    differential is the quantity the LEAN is actually made from. A model can
    predict each offense well and still order the *difference* badly, and the
    difference is what the record scores.
    """
    pred = _num(df, "xw_net")
    act = _num(df, "act_woba_home") - _num(df, "act_woba_away")
    # Dispersion difference, oriented to MATCH xw_net: home offense minus away.
    # The home offense is described on the AWAY-pitcher row, so the home term is
    # `opp_xwoba_sd_away`. That is the same cross paired_rates applies, and
    # differencing it here rather than in a caller is the rule this function
    # already states -- paired_rates owns the side cross, this owns the
    # difference. Reversed, it would report a real effect with the wrong sign.
    d_sd = _num(df, "opp_xwoba_sd_away") - _num(df, "opp_xwoba_sd_home")
    # Run margin, same orientation, for the question the record actually scores.
    margin = _num(df, "full_home") - _num(df, "full_away")
    m = pred.notna() & act.notna()
    if not m.any():
        return pd.DataFrame(columns=["game_pk", "model_tag", "pred", "act",
                                     "d_sd", "margin"])
    return pd.DataFrame({
        "game_pk": (df.loc[m, "game_pk"].to_numpy()
                    if "game_pk" in df.columns else np.nan),
        "model_tag": (df.loc[m, "model_tag"].to_numpy()
                      if "model_tag" in df.columns else None),
        "pred": pred[m].to_numpy(), "act": act[m].to_numpy(),
        "d_sd": d_sd[m].to_numpy(), "margin": margin[m].to_numpy(),
    })


def paired_components(df):
    """Long frame of (predicted, actual) for each model component separately.

    The three components the lean is built from are scored apart from each
    other for the first time here. Their pairings are NOT alike, which is the
    whole reason this lives in one function:

      SP      `starter_xwoba_{side}` is that side's starter, and the realised
              starter-allowed line is keyed the same way -> SAME side.
      BP      `bullpen_xwoba_{side}` likewise -> SAME side.
      lineup  `opp_xwoba_neutral_{side}` is the lineup that side's pitching
              FACES, so it is the other side's offense -> CROSSED, exactly as
              paired_rates crosses mx.

    Getting the lineup cross backwards yields a plausible near-zero slope
    rather than an error, which is why no caller does its own pairing.

    Two identity columns ride along, because the frame could not previously
    answer "whose rate is this?" from its own contents and both new diagnostics
    below need it:

      `unit`     the entity whose talent the PREDICTOR claims to measure -- the
                 starter for SP, the pitching club for BP (a bullpen is a team
                 unit, not a person), the batting club for lineup. This is what
                 `target_reliability` groups the ACTUAL by.
      `faced_sp` the starter the batting side actually faced. Equal to `unit`
                 for SP and a NUISANCE covariate for lineup, which is the whole
                 point of `lineup_within_pitcher_slope`.

    Read slopes, not intercepts. The lineup's predicted value is a neutral
    composite while its actual is a real game against real pitching, so the
    two sit on offset levels by construction; and the weights here may differ
    from Savant's yearly set, which moves an intercept and not a slope.
    """
    rows = []
    for _, r in df.iterrows():
        tag = r.get("model_tag")
        gpk = r.get("game_pk")
        for side in ("away", "home"):
            other = "home" if side == "away" else "away"
            sp, bp = phase_lines(r, side)
            faced_sp = r.get(f"{side}_sp")
            for comp, pred_col, act, unit in (
                ("SP", f"starter_xwoba_{side}",
                 woba_from_components(sp) if sp else None, faced_sp),
                ("BP", f"bullpen_xwoba_{side}",
                 woba_from_components(bp) if bp else None, r.get(side)),
                ("lineup", f"opp_xwoba_neutral_{side}",
                 _f(r.get(f"act_woba_{other}")), r.get(other)),
            ):
                pred = _f(r.get(pred_col))
                if pred is None or act is None:
                    continue
                den = None
                if comp == "SP" and sp:
                    den = sum(sp[f] for f in ("ab", "bb", "hbp", "sf")) - sp["ibb"]
                elif comp == "BP" and bp:
                    den = sum(bp[f] for f in ("ab", "bb", "hbp", "sf")) - bp["ibb"]
                elif comp == "lineup":
                    den = _f(r.get(f"act_pa_{other}"))
                rows.append({"game_pk": gpk, "model_tag": tag, "component": comp,
                             "side": side, "pred": pred, "act": act, "den": den,
                             "unit": unit, "faced_sp": faced_sp})
    return (pd.DataFrame(rows) if rows else
            pd.DataFrame(columns=["game_pk", "model_tag", "component", "side",
                                  "pred", "act", "den", "unit", "faced_sp"]))


def _betacf(a, b, x, maxit=300, eps=3e-16, fpmin=1e-300):
    """Continued fraction for the incomplete beta (Lentz). Numerical Recipes."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_upper_tail(f, d1, d2):
    """P(F_{d1,d2} >= f). Hand-rolled because scipy is not a dependency here.

    The monitor needs this and not a bare threshold on F. An F ratio built from
    30 units has a standard deviation near sqrt(2/29) = 0.26 under the null, so
    a pure-noise target returns F = 1.45 often enough to be labelled a finding
    -- which it did, on the first constructed test written against this code.
    Judging F against its own null distribution rather than against 1.0 is the
    same rule this repo already applies to searched cells and band grids.
    """
    if not (f > 0) or d1 <= 0 or d2 <= 0:
        return 1.0
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def f_quantile(upper_tail, d1, d2):
    """The f with P(F_{d1,d2} >= f) == upper_tail, by bisection on the tail.

    Only the interval below needs it, and bisection on a monotone tail is
    exact to float precision without importing scipy.
    """
    if not (0.0 < upper_tail < 1.0) or d1 <= 0 or d2 <= 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    while f_upper_tail(hi, d1, d2) > upper_tail:
        hi *= 2.0
        if hi > 1e6:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_upper_tail(mid, d1, d2) > upper_tail:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# The entity each component's PREDICTOR claims to tell apart. `target_reliability`
# groups the ACTUAL by this to ask whether the outcome data can tell those same
# entities apart at all -- a question no diagnostic in this repo asked until a
# lineup slope had been read as a finding for weeks.
COMPONENT_UNIT_LABEL = {"SP": "starter", "BP": "pitching club",
                        "lineup": "batting club"}

# Minimum observations per unit before it enters the decomposition. Not a
# credibility gate -- the F statistic is honest at any size and is printed
# whatever it says. It exists because a unit seen once contributes a zero
# within-group deviation and biases MSW downward, which would inflate F for a
# reason that has nothing to do with reliability.
RELIABILITY_MIN_PER_UNIT = 8

# Significance bar for "the target can tell its units apart". Deliberately the
# conventional 0.05 and not tuned: this gates a WARNING about whether another
# number is readable, so the cost of a false positive (a slope read as real)
# exceeds the cost of a false negative (a caveat printed one build early).
RELIABILITY_ALPHA = 0.05

# Smallest true between-unit sd (wOBA) that counts as a spread worth measuring.
# A non-separating F alone does not make a slope undefined: at 30 clubs and
# ~53 games each the omnibus test has ~64% power against a true club sd of
# 0.010, so "not separated" is also what a real spread returns a third of the
# time. The target is called UNMEASURABLE only when the upper confidence bound
# on its between-unit sd falls BELOW this size; otherwise it is INCONCLUSIVE.
# 0.005 is set a little under the club-level spread the shipped lineup
# composite itself predicts (sd of club means of the prediction ~0.006 on the
# ledger when this was written): a target bounded below that cannot carry the
# predictor's slope, while one that cannot exclude it can. Not fitted to the
# verdict it produces -- the lineup row reads INCONCLUSIVE under any value up
# to its own bound, and the bound is printed so a reader can apply another.
RELIABILITY_MATERIAL_SD = 0.005

# Two-sided confidence level for the between-unit sd bound.
RELIABILITY_CI = 0.95


def one_way_icc(labels, values, min_per_group=RELIABILITY_MIN_PER_UNIT):
    """One-way random-effects decomposition of `values` grouped by `labels`.

    Returns a dict with the groups/observations used, the F ratio MSB/MSW, the
    intraclass correlation, and the implied true between-group sd -- or None
    when fewer than two groups survive `min_per_group`.

    Read F, not the ICC, when deciding whether a component is measurable at
    all. F < 1 means the units differ LESS than chance, i.e. the estimated
    between-unit variance is negative and clipped to zero; an ICC of -0.001
    and an ICC of +0.001 are the same answer, while F carries the direction
    and the magnitude of the miss. The between-group sd is clipped at zero for
    the same reason and must not be read as "small but real".
    """
    ok = [(str(k), float(v)) for k, v in zip(labels, values)
          if k is not None and str(k) not in ("", "nan") and _f(v) is not None]
    groups = {}
    for k, v in ok:
        groups.setdefault(k, []).append(v)
    groups = {k: v for k, v in groups.items() if len(v) >= min_per_group}
    if len(groups) < 2:
        return None
    n_groups = len(groups)
    n_obs = sum(len(v) for v in groups.values())
    if n_obs <= n_groups:
        return None
    means = {k: sum(v) / len(v) for k, v in groups.items()}
    grand = sum(sum(v) for v in groups.values()) / n_obs
    msb = sum(len(v) * (means[k] - grand) ** 2
              for k, v in groups.items()) / (n_groups - 1)
    msw = sum(sum((x - means[k]) ** 2 for x in v)
              for k, v in groups.items()) / (n_obs - n_groups)
    if msw <= 0:
        return None
    # Harmonic-style effective group size: unbalanced groups make the naive
    # n the wrong divisor for recovering the variance component.
    n0 = (n_obs - sum(len(v) ** 2 for v in groups.values()) / n_obs) / (n_groups - 1)
    var_between = (msb - msw) / n0 if n0 > 0 else float("nan")
    denom = var_between + msw
    f_stat = msb / msw
    # Upper confidence bound on the between-unit sd, from the F pivot
    # MSB/MSW ~ (1 + n0*var_b/var_w) F_{d1,d2}. A point estimate clipped to
    # zero says nothing about how large a spread the sample could hide; this
    # does, and it is what separates "no spread" from "no power".
    f_lo = f_quantile(1.0 - (1.0 - RELIABILITY_CI) / 2.0,
                      n_groups - 1, n_obs - n_groups)
    if n0 > 0 and f_lo > 0:
        vb_hi = (f_stat / f_lo - 1.0) * msw / n0
        between_sd_upper = math.sqrt(vb_hi) if vb_hi > 0 else 0.0
    else:
        between_sd_upper = float("nan")
    return {
        "n_groups": n_groups, "n_obs": n_obs, "f": f_stat,
        "p": f_upper_tail(f_stat, n_groups - 1, n_obs - n_groups),
        "icc": (var_between / denom) if denom > 0 else 0.0,
        "between_sd": math.sqrt(var_between) if var_between > 0 else 0.0,
        "within_sd": math.sqrt(msw),
        "between_sd_upper": between_sd_upper,
        "negative_variance": var_between <= 0,
    }


def _reliability_verdict(r):
    """One `one_way_icc` result -> its verdict string, or None if unfitted.

    Two ways to fail the omnibus F and they are NOT the same statement, which
    is the whole reason this is a function rather than an `f > 1` test at each
    call site: F <= 1 says the units differ less than chance, while F > 1 with
    p above alpha says they differ by no more than a search this size returns
    from noise. A component can clear the first and fail the second -- BP did,
    at F = 1.131 (p = 0.288) on 2026-09-17.

    Failing either is still not the same as an unmeasurable target. The F test
    has 29 degrees of freedom at 30 clubs and little power against a modest
    real spread, so a non-separating F is UNMEASURABLE only when the upper
    confidence bound on the between-unit sd also sits below
    RELIABILITY_MATERIAL_SD. When the bound cannot exclude a material spread
    the verdict is INCONCLUSIVE: the sample neither shows nor rules out unit
    differences, and a directed 1-df test (`unit_level_slope`) can still
    detect them. The lineup row on 2026-09-25 is that case: F = 0.945 with a
    95% upper bound of 0.0104 while unit means of prediction and actual
    correlate at +0.42.

    A result without `between_sd_upper` (a caller that built the dict by hand)
    keeps the omnibus-only verdict rather than inventing a bound.
    """
    if r is None:
        return None
    if r["f"] > 1.0 and r["p"] < RELIABILITY_ALPHA:
        return "measurable"
    below = r["f"] <= 1.0
    upper = r.get("between_sd_upper")
    if upper is not None and not (upper < RELIABILITY_MATERIAL_SD):
        up = f"{upper:.4f}" if upper == upper else "n/a"
        return ("INCONCLUSIVE -- "
                + ("units differ less than chance in this sample"
                   if below else "not separated from chance")
                + f", but a between-unit sd up to {up} "
                  f"({RELIABILITY_CI:.0%}) is not excluded")
    if below:
        return "UNMEASURABLE -- units differ less than chance"
    return "UNMEASURABLE -- not separated from chance"


def unit_level_slope(s, min_per_group=RELIABILITY_MIN_PER_UNIT):
    """Directed 1-df test: regress unit-mean actual on unit-mean prediction.

    The omnibus F asks whether units differ in ANY direction and spends 29
    degrees of freedom at 30 clubs doing it. The predictor already names a
    direction, so this asks the narrower question it is actually making --
    do units predicted higher realise higher? -- on the same units the F
    groups. It can detect a unit-level spread the omnibus test misses, which
    is why an INCONCLUSIVE verdict points here.

    `s` is one component's rows of `paired_components`. Returns
    (slope, se, t, n_units) or None. The se charges two degrees of freedom and
    treats unit means as independent, which they are across clubs; it does
    not model unequal unit sizes, so read it as approximate.
    """
    if s is None or getattr(s, "empty", True) or "unit" not in s.columns:
        return None
    d = s.dropna(subset=["unit", "pred", "act"])
    d = d[~d["unit"].astype(str).isin(("", "nan"))]
    if d.empty:
        return None
    g = d.groupby(d["unit"].astype(str)).agg(
        pred=("pred", "mean"), act=("act", "mean"), n=("act", "size"))
    g = g[g["n"] >= min_per_group]
    k = len(g)
    if k < 3:
        return None
    x = g["pred"].to_numpy(float) - g["pred"].mean()
    y = g["act"].to_numpy(float) - g["act"].mean()
    sxx = float(np.dot(x, x))
    if sxx <= 0:
        return None
    slope = float(np.dot(x, y)) / sxx
    resid = y - slope * x
    se = math.sqrt(float(np.dot(resid, resid)) / (k - 2) / sxx)
    t = slope / se if se > 0 else float("nan")
    return slope, se, t, k


def reliability_verdicts(df, min_per_group=RELIABILITY_MIN_PER_UNIT):
    """{component: (icc result or None, verdict or None)} for every component.

    Deliberately UNSCOPED by model tag, for the reason `target_reliability`
    gives: the realised rate is metric-free, so a family filter would discard
    rows for a reason that cannot apply to the target. The component block IS
    family-scoped and calls this anyway -- the two denominators differ on
    purpose and each prints its own.
    """
    p = paired_components(df)
    out = {}
    if p.empty or "unit" not in p.columns:
        return out
    for comp in ("SP", "BP", "lineup"):
        s = p[p.component == comp]
        if s.empty:
            continue
        r = one_way_icc(s["unit"], s["act"], min_per_group)
        out[comp] = (r, _reliability_verdict(r))
    return out


def target_reliability(df, min_per_group=RELIABILITY_MIN_PER_UNIT):
    """Can each component's ACTUAL tell its own units apart? Lines, or [].

    This measures the TARGET and never the prediction, which is why it exists
    and why it is not family-scoped: realised wOBA is realised wOBA whatever
    tag produced the row beside it, so scoping it would discard rows for a
    reason that cannot apply. The component block above IS family-scoped, so
    the two print different denominators on purpose and each states its own.

    Why it is worth printing every build: a correlation against a target is
    bounded by the square root of that target's reliability. A component whose
    target cannot distinguish its units has NO attainable slope, and reading
    its fitted slope as "the term contributes nothing" confuses an undefined
    measurement with a measured null. Measured on the committed ledger at the
    time this shipped, realised team offence grouped by batting club gives
    F = 0.941 over 1,882 team-games -- below chance -- while the same plate
    appearances grouped by the starter who threw them give F = 1.503.

    That first figure was read as "the lineup target carries no club-level
    signal", which the F alone cannot establish. On 2026-09-25 the same row
    (F = 0.945, 1,596 team-games) had a 95% upper bound of 0.0104 on the true
    club sd -- a real-sized spread -- so a low F at 30 clubs is as consistent
    with low power as with no spread. Each row therefore prints that bound,
    and only a bound below RELIABILITY_MATERIAL_SD earns UNMEASURABLE.
    """
    p = paired_components(df)
    if p.empty or "unit" not in p.columns:
        return []
    lines = [
        "target reliability (can the ACTUAL tell this component's units apart?)",
        "  Pools every family: the realised rate is metric-free, so the "
        "component block's tag scope cannot apply here.",
        f"  One-way ANOVA on the actual, grouped by unit, "
        f"min {min_per_group} observations per unit.",
        "  F is judged against its own null, not against 1.0: with 30 "
        "units the null sd of F is ~0.26, so a bare threshold calls noise a "
        "finding. Not separated AND bounded below a material spread => no "
        "attainable slope, and the component's fitted slope is undefined "
        "rather than null.",
        f"  UNMEASURABLE also requires the {RELIABILITY_CI:.0%} upper bound on "
        f"the true between-unit sd to fall below {RELIABILITY_MATERIAL_SD}; "
        "a non-separating F whose bound cannot exclude a material spread is "
        "INCONCLUSIVE (low power, not no spread) -- see each component's "
        "unit-level slope.",
    ]
    any_row = False
    for comp, (r, verdict) in reliability_verdicts(df, min_per_group).items():
        label = COMPONENT_UNIT_LABEL.get(comp, "unit")
        if r is None:
            lines.append(f"  {comp:<7s} by {label:<14s} too few units with "
                         f"{min_per_group}+ observations")
            any_row = True
            continue
        sd = ("0 (negative variance component)" if r["negative_variance"]
              else f"{r['between_sd']:.5f}")
        up = r.get("between_sd_upper")
        if up is not None and up == up:
            sd += f", {RELIABILITY_CI:.0%} upper {up:.4f}"
        lines.append(
            f"  {comp:<7s} by {label:<14s} units={r['n_groups']:<4d} "
            f"n={r['n_obs']:<5d} F={r['f']:.3f} (p={r['p']:.3f})  "
            f"ICC={r['icc']:+.4f}  true between-unit sd {sd}  -> {verdict}")
        any_row = True
    return lines if any_row else []


def lineup_within_pitcher_slope(p):
    """Within-starter slope of realised offence on the lineup composite.

    The pooled component slope regresses a batting club's realised wOBA on its
    predicted composite while ignoring WHO IT FACED, so opposing-starter talent
    -- the one thing this ledger's outcomes demonstrably do carry (see
    `target_reliability`) -- lands in the residual. Demeaning both sides within
    the faced starter removes it and leaves the comparison the lineup term is
    actually making: the same pitcher, different opponents.

    Applied to the LINEUP component only, and that restriction is the point. For
    SP the starter IS the subject, so demeaning by him would remove exactly the
    variance under test and return a slope of zero by construction. A caller
    passing the SP rows here would get a confident null about nothing.

    Returns (slope, se, n_obs, n_pitchers) or None. The standard error charges
    one degree of freedom per absorbed pitcher, so a design with two starts per
    pitcher is penalised for the fixed effects it spent rather than flattered
    by them.
    """
    if p is None or getattr(p, "empty", True) or "faced_sp" not in p.columns:
        return None
    s = p[p.component == "lineup"].dropna(subset=["pred", "act", "faced_sp"])
    if s.empty:
        return None
    xs, ys, n_groups = [], [], 0
    for _, grp in s.groupby(s["faced_sp"].astype(str), sort=False):
        if len(grp) < 2:          # a singleton demeans to exactly zero
            continue
        n_groups += 1
        px = pd.to_numeric(grp["pred"], errors="coerce")
        ay = pd.to_numeric(grp["act"], errors="coerce")
        xs.extend((px - px.mean()).tolist())
        ys.extend((ay - ay.mean()).tolist())
    n = len(xs)
    dof = n - n_groups - 1
    if n < 3 or dof <= 0:
        return None
    sxx = float(np.dot(xs, xs))
    if sxx <= 0:
        return None
    slope = float(np.dot(xs, ys)) / sxx
    resid = np.asarray(ys) - slope * np.asarray(xs)
    se = math.sqrt(float(np.dot(resid, resid)) / dof / sxx)
    return slope, se, n, n_groups


def components_summary(df, tags=None):
    """Per-component predicted-vs-actual lines. Empty list when nothing pairs.

    Family-scoped for the same reason the rate and delta lines are: v9/v10
    predicts these components from xwOBA inputs and the wOBA lineage from
    observed wOBA, and the actual is observed wOBA either way, so pooling them
    would describe a model that never ran.
    """
    # The reliability verdicts come from the UNSCOPED frame, before the tag
    # filter below: a realised rate does not know which model wrote the row
    # beside it, and this block needs them for every component, not just the
    # one whose caveat happened to be written out in prose.
    verdicts = reliability_verdicts(df)
    if tags is not None and "model_tag" in getattr(df, "columns", []):
        df = df[df["model_tag"].astype(str).isin(set(tags))]
    p = paired_components(df)
    if p.empty:
        return []
    lines = ["component error (each scored against its own realised phase)",
             "  pitching-side suffixes: Home offense vs away pitching; "
             "Away offense vs home pitching.",
             "  Starter and bullpen actuals, and starter IP, retain "
             "same-team pairing."]
    for comp in ("SP", "BP", "lineup"):
        s = p[p.component == comp]
        if s.empty:
            continue
        err = s["act"] - s["pred"]
        cal = calibration(s["pred"], s["act"])
        bit = (f"  {comp:<7s} n={len(s):<4d} pred {s['pred'].mean():.4f} "
               f"act {s['act'].mean():.4f} ({err.mean():+.4f})  "
               f"MAE {err.abs().mean():.4f}")
        if cal:
            r = float(np.corrcoef(s["pred"], s["act"])[0, 1])
            bit += f"  slope {cal['slope']:+.2f}±{cal['se_slope']:.2f}  corr {r:+.3f}"
        # The marker is DERIVED from the same ANOVA the block below prints,
        # per component. It used to be one sentence of prose under the lineup
        # row, which left BP's slope bare under an UNMEASURABLE verdict of its
        # own -- and phrased the test as "clears F=1", which BP's 1.131 does
        # while still failing on p. A fitted slope against a target that
        # cannot tell its units apart is undefined, not null, whichever
        # component it belongs to.
        _r, _verdict = verdicts.get(comp, (None, None))
        if _verdict and _verdict.startswith("UNMEASURABLE"):
            bit += "   [target UNMEASURABLE: this slope is undefined, not null]"
        elif _verdict and _verdict.startswith("INCONCLUSIVE"):
            bit += ("   [target INCONCLUSIVE: omnibus F not separated, "
                    "material spread not excluded]")
        lines.append(bit)
        ul = unit_level_slope(s)
        if ul:
            u_slope, u_se, u_t, u_k = ul
            label = COMPONENT_UNIT_LABEL.get(comp, "unit")
            lines.append(
                f"           unit-level slope {u_slope:+.2f}±{u_se:.2f}  "
                f"(t={u_t:+.2f}, {u_k} units by {label}, "
                f"{RELIABILITY_MIN_PER_UNIT}+ rows each; unit-mean actual on "
                f"unit-mean prediction -- the directed 1-df test the omnibus "
                f"F is not)")
        if comp == "lineup":
            fe = lineup_within_pitcher_slope(p)
            if fe:
                slope, se, n_obs, n_pit = fe
                lines.append(
                    f"           within-starter slope {slope:+.3f}±{se:.3f}  "
                    f"(n={n_obs}, {n_pit} starters absorbed; a correctly "
                    f"scaled composite implies +1.000)")
                lines.append(
                    "           the line above ignores who was faced; this "
                    "one holds the starter fixed. Read the difference, not "
                    "either alone -- and see the lineup row's own verdict in "
                    "the target-reliability block for whether either is "
                    "interpretable at all.")
    return lines


def paired_sp_ip(df, estimate="raw_fallback"):
    """Long frame of (expected starter IP, actual starter IP), same-team paired.

    ``estimate`` is ``raw_fallback`` (raw where present, otherwise stored),
    ``raw`` (raw only), or ``adjusted`` (published value only). Missing values
    stay missing. The last mode must be family-scoped because pre-v12 published
    values are raw while v12 published values are adjusted.

    In every mode the suffix names the pitching team: away pairs with
    ``act_sp_ip_away`` and home with ``act_sp_ip_home``. This differs from the
    offense-rate fields, whose suffix names the pitching side faced.
    """
    if estimate not in {"raw_fallback", "raw", "adjusted"}:
        raise ValueError(f"unknown starter-IP estimate: {estimate}")
    rows = []
    for side in ("away", "home"):
        raw = _num(df, f"expected_sp_ip_raw_{side}")
        pub = _num(df, f"expected_sp_ip_{side}")
        if estimate == "raw_fallback":
            pred = raw.where(raw.notna(), pub)
        elif estimate == "raw":
            pred = raw
        else:
            pred = pub
        act = _num(df, f"act_sp_ip_{side}")
        m = pred.notna() & act.notna()
        if not m.any():
            continue
        rows.append(pd.DataFrame({"side": side, "pred": pred[m].to_numpy(),
                                  "act": act[m].to_numpy()}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["side", "pred", "act"]))


def _sp_ip_diagnostic_line(label, pairs):
    """One calibration line whose count and errors come from ``pairs``."""
    if pairs.empty:
        return f"  {label}: n=0  slope unavailable  MAE unavailable  bias unavailable"
    err = pairs["act"] - pairs["pred"]
    cal = calibration(pairs["pred"], pairs["act"])
    slope = (f"{cal['slope']:.3f} ± {cal['se_slope']:.3f}"
             if cal else "unavailable")
    return (f"  {label}: n={len(pairs):<4d} slope {slope}  "
            f"MAE {err.abs().mean():.3f} IP  bias {err.mean():+.3f} IP")


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
    # `x.std() <= 0` never fires for a constant column: subtracting the mean
    # leaves float dust, so 60 identical values give std 1.1e-16 rather than 0.
    # polyfit then returns a meaningless slope with an se around 1e13 -- a
    # statistic with no usable sampling distribution, printed as though it were
    # a measurement, which is the exact shape CLAUDE.md files under "public
    # claims the data can't support". Compare the spread to the magnitude
    # rather than to zero.
    if n < 3 or x.std() <= 1e-12 * max(1.0, abs(float(x.mean()))):
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
    net = paired_net(fam)
    if rates.empty and ip.empty and net.empty:
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
    if not net.empty:
        cal = calibration(net["pred"], net["act"])
        bits.append(f"net n={len(net):3d}"
                    + (f" slope {cal['slope']:+.2f}±{cal['se_slope']:.2f}"
                       if cal else ""))
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
    ip = paired_sp_ip(df, "raw_fallback")  # pooled -- see above
    # Exact raw/adjusted comparisons are meaningful only within one requested
    # family. Without tags the published column mixes raw historical values
    # with adjusted v12 values, so do not manufacture a blended diagnostic.
    ip_raw = paired_sp_ip(d, "raw") if tags is not None else paired_sp_ip(d.iloc[0:0], "raw")
    ip_adjusted = (paired_sp_ip(d, "adjusted") if tags is not None
                   else paired_sp_ip(d.iloc[0:0], "adjusted"))
    rates = paired_rates(d)        # family-scoped
    # Unscoped means every scale family at once, which is the artifact
    # documented below. Refusing to print is deliberate: a missing line sends
    # the reader to that comment, a printed one sends them off with a number.
    net = paired_net(d) if tags is not None else paired_net(d).iloc[0:0]
    if ip.empty and rates.empty and net.empty:
        return []

    lines = ["predicted vs actual (backfilled box scores)"]
    if not ip.empty or not ip_raw.empty or not ip_adjusted.empty:
        family = " + ".join(tags) if tags else "requested family"
        lines.append("  starter IP calibration (bias = actual - estimate; "
                     "positive means the starter lasted longer than estimated)")
        lines.append(_sp_ip_diagnostic_line(
            "pooled raw-with-fallback, all historical model families", ip))
        lines.append(_sp_ip_diagnostic_line(
            f"current family raw [{family}; expected_sp_ip_raw_*]", ip_raw))
        lines.append(_sp_ip_diagnostic_line(
            f"current family adjusted [{family}; expected_sp_ip_*]", ip_adjusted))
        lines.append("    slope 1.00 means calibrated dispersion; below 1.00 is "
                     "over-dispersed and above 1.00 is compressed.")
        lines.append("    Pooled uses raw estimates where available and otherwise "
                     "stored estimates; it does not describe the current "
                     "family's adjusted estimates. Missing values are excluded "
                     "separately from each displayed n. (IP drives phase weight q.)")
    if not rates.empty:
        n = len(rates)
        lines.append(f"  offense wOBA n={n:<4d} pred mean {rates['pred'].mean():.4f}  "
                     f"actual mean {rates['act'].mean():.4f}  "
                     f"({rates['act'].mean() - rates['pred'].mean():+.4f})")
        lines.append("    field pairing: Home offense vs away pitching "
                     "(mx_xwoba_away -> act_woba_home);")
        lines.append("                   Away offense vs home pitching "
                     "(mx_xwoba_home -> act_woba_away).")
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

        # Lineup dispersion, printed every build for the reason the IP slope is:
        # the question needs ~12 slates of rows that did not exist when the
        # column shipped, and a check that waits for someone to remember it is
        # the deferral this repo has already got wrong once.
        #
        # This is the headline read only. `dispersion_probe.py` is the full one
        # -- it controls for the backfill count, reports the zero-backfill
        # subset, and scores the game-level differential and run margin, which
        # is the question the record actually settles. A test pins this slope
        # against the probe's H1 so the two cannot drift into disagreeing.
        #
        # `calibration(sd, residual)` is the same estimator the lines above use,
        # not a second one: slope of the residual on dispersion, with its se.
        # Positive means concentrated lineups beat the mean they are averaged
        # into. Family-scoped like every rate line here, which also happens to
        # be free -- no row predating the column can carry one.
        if "sd" in rates and rates["sd"].notna().any():
            d = rates[rates["sd"].notna()]
            dcal = calibration(d["sd"], d["act"] - d["pred"])
            if dcal:
                lines.append(
                    f"  lineup dispersion n={dcal['n']:<4d} "
                    f"residual slope {dcal['slope']:+.2f} +/- {dcal['se_slope']:.2f} "
                    f"(> 0 = concentrated lineups beat their mean)")
                if dcal["n"] < DISPERSION_N_MIN:
                    lines.append(
                        f"    UNDER-POWERED: needs ~{DISPERSION_N_MIN} side-games "
                        f"to separate a 0.15 correlation; see dispersion_probe.py")

    # The lean delta's own calibration. Family-scoped for the same reason the
    # rates are, and for one more: `xw_net` units are a scale-family property
    # (v5's shrinkage halved them), so pooling would regress a mixture of
    # scales against one actual and call the blend a slope.
    #
    # This line exists because pooling produced a wrong answer that looked
    # right. Measured 2026-08-05, the slope over all 403 graded rows is +0.48
    # -- "the delta is twice as spread as it should be", which is exactly the
    # shape of the IP slope above and reads as a finding. It is an artifact.
    # Pre-v5 rows are unshrunk (sd 0.058) and post-v5 rows are shrunk
    # (sd 0.025); regressing the mixture against one actual returns a slope
    # describing no model that ever ran. Per scale family:
    #
    #     v2      n=147  +0.48 +/- 0.19      v7      n= 45  +1.07 +/- 1.05
    #     v3      n= 41  +0.59 +/- 0.40      v9/v10  n= 97  +1.45 +/- 0.52
    #     v5/v6   n= 35  +0.02 +/- 1.13      wOBA    n= 16  -2.12 +/- 1.82
    #
    # So the over-dispersion was real for the UNSHRUNK families and is gone
    # from the current lineage: v9/v10 sits 0.87 se ABOVE 1.0, if anything
    # slightly over-shrunk. There is no delta shrink left to apply here, and
    # applying the pooled 0.48 would have compressed an already-compressed
    # delta on the strength of a mixing artifact.
    #
    # Live tension worth carrying: `calibration` reads slope > 1 as "too
    # compressed, which is what an over-aggressive shrinkage K produces",
    # while reliever_shrink_probe.py fits K well above the shipped 100 --
    # i.e. MORE shrinkage. Both sit inside their own noise, and they point
    # opposite ways. Reconcile them before moving XWOBA_SHRINK_K.
    #
    # WHAT A UNIFORM SHRINK DOES NOT FIX, stated because the obvious reading is
    # wrong: multiplying every xw_net by the slope does NOT move the clear/strong
    # labels. lean_strength shrinks the pool's OWN p33/p80 toward
    # LEAN_STRENGTH_FALLBACK, so scaling the deltas scales the observed
    # quantiles with them and the ranking is identical -- the labels shift only
    # through the frozen prior, i.e. only to the extent that prior is on a
    # different scale from the pool. It also flips no lean, being monotone. So
    # this line is a MEASUREMENT, and printing it every build is the point: it
    # is what a win-probability mapping would have to be built on, and what the
    # LEAN_STRENGTH prior must be re-derived against if the delta is ever put on
    # a calibrated scale (which would be a new _SCALE_FAMILIES entry).
    if not net.empty:
        cal = calibration(net["pred"], net["act"])
        if cal:
            lines.append(f"  lean delta   n={len(net):<4d} "
                         f"calibration slope {cal['slope']:+.2f} "
                         f"+/- {cal['se_slope']:.2f} "
                         f"(1.00 = calibrated; < 1 = over-dispersed)")
            r = float(np.corrcoef(net["pred"], net["act"])[0, 1])
            agree = float((np.sign(net["pred"]) == np.sign(net["act"])).mean())
            lines.append(f"    corr {r:+.3f}  sign agreement {agree:.3f}  "
                         f"(the lean's own hit rate against its target)")
            if cal["se_slope"] > 0.25:
                lines.append(f"    UNDER-POWERED: se {cal['se_slope']:.2f} "
                             f"cannot separate 1.00 from 0.50")
    return lines


# Display-only cap on the per-slate block. The ledger keeps every slate; this
# section answers "did anything break recently", and an unbounded listing adds
# a line a day to a report nobody would then scroll. Raising or lowering it
# changes what is printed and nothing that is computed, which is the only
# reason a bare integer is acceptable here at all -- it is not a threshold any
# number crosses, and no line reads differently on either side of it.
SLATE_WINDOW = 14


def slate_lines(df, limit=SLATE_WINDOW):
    """Per-slate predicted-vs-actual, most recent first. Empty list if nothing pairs.

    Everything else in this module aggregates over a prediction family and is
    read for a trend. This block is the opposite question -- one slate at a
    time, for the failure that shows up as a single bad day: a lineup source
    degrading, an opener misread, a slate built against stale rates. Those are
    invisible in a 366-row cumulative mean and obvious in one line.

    Grouped by (date, metric), not by date, because a date can span both --
    2026-08-03 carries `split v1` rows beside `wOBA v1` ones -- and an
    xwOBA-input prediction and a wOBA-input one are not the same measurement
    against an observed-wOBA actual. Grouping by the thing that makes pooling
    legal costs one extra line on the one day it applies, and no special case.

    Bias and MAE only. NO SLOPES, deliberately: a 15-game slate is ~30
    side-pairs, where `calibration` returns an se near +/-1.5 -- a figure whose
    own error bar spans every conclusion anyone would draw from it. The header
    prints the error sd the lines are read against for the same reason. What
    one slate can show is a gross outlier; that is all this claims to show.
    """
    if "game_date" not in getattr(df, "columns", []):
        return []
    d = df.copy()
    d["_metric"] = metric_series(d)
    groups = []
    for (date, metric), g in d.groupby(["game_date", "_metric"], dropna=False):
        rates, ip = paired_rates(g), paired_sp_ip(g)
        if rates.empty and ip.empty:
            continue
        groups.append((str(date), str(metric), len(g), rates, ip))
    if not groups:
        return []
    groups.sort(key=lambda t: t[0], reverse=True)
    shown = groups[:limit]

    head = f"by slate (most recent {len(shown)} of {len(groups)}; monitoring, not trend)"
    lines = [head]
    # The scale the lines are read against, derived here rather than quoted:
    # a slate's rate bias means nothing without the per-side spread it sits in.
    # Pooling every family for THIS number and no other is deliberate -- it is
    # the dispersion of a ~38-PA game outcome, which is a property of baseball
    # and not of the family that predicted it. The family-scoped rule upstream
    # governs LEVELS, where the metrics genuinely disagree.
    allr = paired_rates(df)
    if len(allr) > 1:
        sd = float((allr["act"] - allr["pred"]).std())
        lines.append(f"  scale: per-side rate error sd {sd:.4f} over {len(allr)} "
                     f"paired side-games — read gross outliers, not drift")
    for date, metric, n_games, rates, ip in shown:
        bits = [f"  {date}  {metric:<10} {n_games:>3d}g"]
        if not rates.empty:
            err = rates["act"] - rates["pred"]
            bits.append(f"rate n={len(rates):>3d} pred {rates['pred'].mean():.4f} "
                        f"act {rates['act'].mean():.4f} ({err.mean():+.4f}) "
                        f"MAE {err.abs().mean():.4f}")
        if not ip.empty:
            err = ip["act"] - ip["pred"]
            bits.append(f"SP IP n={len(ip):>3d} bias {err.mean():+.2f} "
                        f"MAE {err.abs().mean():.2f}")
        lines.append("  ".join(bits))
    return lines
