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
            for comp, pred_col, act in (
                ("SP", f"starter_xwoba_{side}",
                 woba_from_components(sp) if sp else None),
                ("BP", f"bullpen_xwoba_{side}",
                 woba_from_components(bp) if bp else None),
                ("lineup", f"opp_xwoba_neutral_{side}", _f(r.get(f"act_woba_{other}"))),
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
                             "side": side, "pred": pred, "act": act, "den": den})
    return (pd.DataFrame(rows) if rows else
            pd.DataFrame(columns=["game_pk", "model_tag", "component", "side",
                                  "pred", "act", "den"]))


def components_summary(df, tags=None):
    """Per-component predicted-vs-actual lines. Empty list when nothing pairs.

    Family-scoped for the same reason the rate and delta lines are: v9/v10
    predicts these components from xwOBA inputs and the wOBA lineage from
    observed wOBA, and the actual is observed wOBA either way, so pooling them
    would describe a model that never ran.
    """
    if tags is not None and "model_tag" in getattr(df, "columns", []):
        df = df[df["model_tag"].astype(str).isin(set(tags))]
    p = paired_components(df)
    if p.empty:
        return []
    lines = ["component error (each scored against its own realised phase)"]
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
        lines.append(bit)
    return lines


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
    ip = paired_sp_ip(df)          # pooled -- see above
    rates = paired_rates(d)        # family-scoped
    # Unscoped means every scale family at once, which is the artifact
    # documented below. Refusing to print is deliberate: a missing line sends
    # the reader to that comment, a printed one sends them off with a number.
    net = paired_net(d) if tags is not None else paired_net(d).iloc[0:0]
    if ip.empty and rates.empty and net.empty:
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
