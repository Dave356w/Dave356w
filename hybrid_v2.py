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
# LATENT GAP, recorded rather than patched: this constant is denominated in
# the CURRENT delta scale. `xw_net` is an xwOBA difference, and its spread is a
# property of the prediction math -- `_SCALE_FAMILIES` exists precisely because
# that spread has moved before (v5 halved it, median |xw_net| .036 -> .018;
# v3's K quadrupling compressed it again). A frozen .012 therefore tracks a
# different quantile of the distribution after any such change, exactly as
# `LEAN_STRENGTH_FALLBACK` does one file out -- and unlike that constant, this
# one has no entry saying so until now.
#
# Measured on the current family (421 decidable rows, median |xw_net| 0.01836):
# the fade branch holds 19 rows at the frozen gate, 31 if the scale halves and
# 12 if it doubles -- a 2.6x range on the only branch the rule owns.
#
# NOT re-derived here, and that is deliberate. This is a REGISTERED constant:
# it was frozen on 2026-09-11 and re-fitting it to the live pool would make the
# registration meaningless, which is the whole reason `tests/test_hybrid_test.py`
# pins these literals. What a `_SCALE_FAMILIES` entry invalidates is not the
# number but the REGISTRATION: a scale change means the forward window has been
# scoring a different statistic than the one registered, and the honest
# response is a new registration with a fresh window, not a quiet re-fit.
DELTA_THRESHOLD = 0.012
STAKE = v1.STAKE
RULE_TAG = "xwoba_market_hybrid_v2"
# Imported, never restated: v1's gate sizing is the same arithmetic on the same
# effect size, and a second literal is the "one value, three homes" defect. The
# report says in words that v2 accrues switches more slowly, since its fade
# branch is a strict subset of v1's.
GATE_SWITCHES = v1.GATE_SWITCHES
GATE_SWITCHES_REALISTIC = v1.GATE_SWITCHES_REALISTIC
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


def discovery_switch_delta(led):
    """The switch delta over the rows the rule was FOUND on, or None.

    Derived by splitting at `REGISTERED_ON` rather than frozen as a literal,
    which is the opposite of v1's choice and deliberate: v1's discovery figure
    was computed once, at registration, from a row set that no longer changes,
    so a literal there is a record. v2's has never been written down, and
    freezing one now would be a number read off today's ledger wearing a
    registration date -- the constants-frozen-from-data entry.

    Scored at the closing basis, because that is the only basis the pre-
    registration rows have: no-lookahead keeps market columns off a pending
    row, so a decision-time price exists only for rows captured after the
    capture shipped. The forward line above it is on the saved-pregame basis,
    and the two are NOT the same measurement -- which is exactly why this is
    labelled `discovery` and printed as a reference rather than pooled.

    SCOPED TO THE RULE'S OWN PREDICTION FAMILY, and the first version of this
    function was not. `decidable` applies no tag filter -- correctly, since its
    one production caller hands it a family-scoped frame -- so splitting the
    whole ledger at the registration date reaches back through every earlier
    family. The `|xw_net| < .012` gate is denominated in the CURRENT delta
    scale (see DELTA_THRESHOLD), so applying it to a wOBA-era row asks a
    different question of a different statistic, and pooling the answers is
    the `_SCALE_FAMILIES` error inside a single number.

    The family is read off the FORWARD rows rather than imported: every row
    after the registration date is current-family by construction, and taking
    the tags from there keeps this module free of a build_site import it
    cannot safely make and free of a tag literal that would go stale at the
    next bump. No forward rows means no reference is printed, which is correct
    -- there is nothing yet to read against it.
    """
    d = decidable(led)
    if d is None or d.empty:
        return None
    fwd = scored_rows(led)
    if fwd is None or not len(fwd) or "model_tag" not in fwd.columns:
        return None
    fam = set(fwd["model_tag"].dropna().unique())
    if not fam or "model_tag" not in d.columns:
        return None
    g = apply_rule(d[d["model_tag"].isin(fam)].copy())
    g = g[g["game_date"].astype(str) <= REGISTERED_ON]
    sw = g[~g["follow"]]
    if not len(sw):
        return None
    return {"n": int(len(sw)), "mean": float(sw["switch_delta"].mean()),
            "total": float(sw["switch_delta"].sum()),
            "families": tuple(sorted(fam))}


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
    n_sw = len(sw)
    # THE REGISTERED HEADLINE, and the reason it is the switch delta rather
    # than the combined line: v1's docstring point 3 applies unchanged to v2.
    # The rule alters only the switched selections; every followed row is the
    # v12 model untouched, so a combined ROI can restate what the model already
    # does and nothing else. This block shipped WITHOUT it -- v1 printed the
    # headline and the gate while v2, the rule actually in production, printed
    # `combined hybrid` as its most prominent forward number. `switch_delta` was
    # already computed by `apply_rule`; only the reporting was missing.
    if n_sw:
        d = sw["switch_delta"].values
        se = float(np.std(d, ddof=1) / np.sqrt(n_sw)) if n_sw > 1 else float("nan")
        out.append(f"    SWITCH DELTA (registered)  n={n_sw:<4d} "
                   f"{d.mean():+.3f}u per switch  +/- {se:.3f}"
                   + (f"   z={d.mean() / se:+.2f}" if se and se > 0 else "")
                   + f"   total {d.sum():+.2f}u")
    else:
        out.append("    SWITCH DELTA (registered)    no switched selections yet")
    disc = discovery_switch_delta(led)
    if disc is not None:
        out.append(f"    discovery was {disc['mean']:+.3f}u per switch over "
                   f"{disc['n']}. Read the forward number against that, not "
                   "against zero.")
    out.append("")
    out.append(_line(g[g["follow"]], "follow"))
    out.append(_line(sw, "fade q<.45 & |d|<.012"))
    out.append(_line(g, "combined hybrid"))
    out.append(_line(g, "plain xwOBA lean", won_col="lean_won",
                     p_col="model_side_p", profit_col="lean_profit"))
    out.append(_line(g, "control: always-chalk", won_col="chalk_won",
                     p_col="chalk_p", profit_col="chalk_profit"))
    if n_sw:
        same = int((sw["bet_home"].values
                    == (sw["pregame_p_home"] >= 0.5).values).sum())
        out.append(f"    the fade branch backed the favourite in {same} of "
                   f"{n_sw} switched games (construction says all of them)")
    out.append(f"    GATE: {n_sw} of ~{GATE_SWITCHES} switches to test the "
               f"discovery-sized effect, ~{GATE_SWITCHES_REALISTIC} for a "
               "plausible +0.10u one. v2 fades STRICTLY less often than v1 "
               "did, so it reaches these more slowly, not faster.")
    out.append("    The combined-hybrid line is mostly the model, not the rule: "
               "only the switched games are the hypothesis. See hybrid_v2.py.")
    return out

