"""Current XWOBA Market Hybrid selection rule and forward registration.

Version 1 remains frozen in :mod:`hybrid_test`.  This module is version 2:
fade only a weak model lean (``abs(xw_net) < .012``) whose leaned side is
priced below 45%; otherwise follow the model.  Retrospective callers use
closing prices, while the registered forward scorer uses only the immutable
pregame decision and prices stored in the ledger.
"""
import numpy as np
import pandas as pd

import hybrid_test as v1

REGISTERED_ON = "2026-09-11"      # slates STRICTLY after this date are scored
THRESHOLD = v1.THRESHOLD
DELTA_THRESHOLD = 0.012
STAKE = v1.STAKE
RULE_TAG = "xwoba_market_hybrid_v2"
PRIOR = "null"
LEDGER = v1.LEDGER
LOCKED_COLUMNS = v1.LOCKED_COLUMNS
FORWARD_BASE_COLUMNS = v1.FORWARD_BASE_COLUMNS + ("xw_net",)


def decidable(led):
    """Graded close-priced rows with a lean and finite ``xw_net``."""
    d = v1.decidable(led)
    if d is None or "xw_net" not in getattr(d, "columns", ()):
        return None
    x = pd.to_numeric(d["xw_net"], errors="coerce")
    return d[x.notna() & np.isfinite(x)].copy()


def follows(model_side_p, xw_net):
    """Vectorised v2 decision: only low-price, weak-delta rows are faded."""
    q = np.asarray(model_side_p, dtype=float)
    delta = np.abs(np.asarray(xw_net, dtype=float))
    return (q >= THRESHOLD) | (delta >= DELTA_THRESHOLD)


def apply_rule(g):
    """Apply v2 retrospectively at each row's closing market. Pure."""
    g = v1.apply_rule(g.copy())
    follow = follows(g["model_side_p"], g["xw_net"])
    lean_home = g["lean_home"].to_numpy(dtype=bool)
    lean_won = g["lean_won"].to_numpy(dtype=bool)
    g["follow"] = follow
    g["bet_home"] = np.where(follow, lean_home, ~lean_home)
    g["bet_won"] = np.where(follow, lean_won, ~lean_won)
    g["p_bet"] = np.where(follow, g["model_side_p"], 1 - g["model_side_p"])
    g["ml_bet"] = np.where(
        g["bet_home"], g["close_home_ml"], g["close_away_ml"])
    g["profit"] = np.where(
        g["bet_won"], STAKE * v1._payout(g["ml_bet"]), -STAKE)
    g["switch_delta"] = g["profit"] - g["lean_profit"]
    return g


apply_locked_rule = v1.apply_locked_rule
_excess_z = v1._excess_z
_line = v1._line


def _ledger(led):
    led = v1._ledger(led)
    if led is None or "xw_net" not in getattr(led, "columns", ()):
        return None
    return led


def _committed(led):
    return led[(led["status"] == "graded")
               & (led["game_date"].astype(str) > REGISTERED_ON)
               & led["xw_lean"].notna()
               & pd.to_numeric(led["xw_net"], errors="coerce").notna()
               & led["selection_rule_tag"].eq(RULE_TAG)
               & led["hybrid_action"].isin(["FOLLOW", "FADE"])]


def scored_rows(led=None):
    """Score only post-registration, saved-pregame v2 commitments."""
    led = _ledger(led)
    if led is None:
        return None
    if any(c not in led.columns for c in LOCKED_COLUMNS):
        return led.iloc[0:0].copy()
    g = _committed(led)
    g = g[(g["hybrid_selection"].eq(g["home"])
           | g["hybrid_selection"].eq(g["away"]))
          & g["hybrid_full"].isin(["W", "L"])
          & pd.to_numeric(g["pregame_p_home"], errors="coerce").notna()
          & pd.to_numeric(g["pregame_home_ml"], errors="coerce").notna()
          & pd.to_numeric(g["pregame_away_ml"], errors="coerce").notna()
          & pd.to_numeric(g["hybrid_p"], errors="coerce").notna()
          & pd.to_numeric(g["hybrid_ml"], errors="coerce").notna()].copy()
    return g if g.empty else apply_locked_rule(g)


def unscorable(led=None):
    led = _ledger(led)
    if led is None or any(c not in led.columns for c in LOCKED_COLUMNS):
        return 0
    scored = scored_rows(led)
    return int(len(_committed(led)) - (0 if scored is None else len(scored)))


def report_lines(led=None):
    """Forward-only v2 report; retrospective history is printed separately."""
    out = [
        f"pre-registered hybrid v2 test  (registered {REGISTERED_ON}; fade only "
        f"when q < {THRESHOLD:.2f} AND |xw_net| < {DELTA_THRESHOLD:.3f}; "
        "follow otherwise)",
        "    price source — selection/eligibility: saved pregame_p_home + "
        "locked action/selection;",
        "                   market comparison: hybrid_p/pregame_p_home; "
        "returns: hybrid_ml/pregame MLs. No close fallback.",
        "    missing inputs — malformed locked selection, grade, probability, "
        "moneyline, or xw_net: excluded and counted as unscorable.",
    ]
    led = _ledger(led)
    g = None if led is None else scored_rows(led)
    if g is None:
        return out + ["    ledger unavailable or missing columns -- not scored"]
    slates = g["game_date"].nunique() if len(g) else 0
    out.append(f"    eligible rows since registration: {len(g)} over {slates} slates")
    dropped = unscorable(led)
    if dropped:
        out.append(f"    WARNING: {dropped} committed row(s) are unscorable.")
    if not len(g):
        out.append("    nothing to score yet. Prior is NULL; all-v12 results above "
                   "are retrospective and do not count toward this gate.")
        return out
    sw = g[~g["follow"]]
    out.append(_line(g[g["follow"]], "follow"))
    out.append(_line(sw, "fade q<.45 & |d|<.012"))
    out.append(_line(g, "combined hybrid"))
    out.append(_line(g, "plain xwOBA lean", won_col="lean_won",
                     p_col="model_side_p", profit_col="lean_profit"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    return out

