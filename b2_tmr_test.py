#!/usr/bin/env python3
"""Pre-registered Candidate B2 forward test: individual-extreme TMR10.

REGISTERED 2026-09-18. Every parameter in the frozen registration block below
is fixed before the first scored slate. Rows dated on the registration day are
not scored; only slates STRICTLY AFTER it count.

B2 is a SHADOW contextual overlay to the actual forward v13 model. It never
changes the published v13 lean. The primary question is narrower: when B2 and
v13 disagree, does selecting the lower-TMR team improve market residual and
flat-unit return relative to leaving v13 unchanged?

TMR10 is a market forecast-error state, not a direct estimate of team talent
and not a generic mean-reversion claim. For each team and prior game:

    residual = realised outcome - market-implied win probability
    TMR10    = mean(last 10 residuals)
    Var      = sum[p_i (1-p_i)] / 10^2
    Z_team   = TMR10 / sqrt(Var)

The forward trigger is:

    max(|Z_home|, |Z_away|) >= 1.50

and the selected side is the team with the lower RAW TMR10. Individual Z is
only the qualification statistic; direction is not "pick the most extreme Z".

STATE PRICE BASIS. Historical Candidate-B work used the archived/current ESPN
two-sided market as a close proxy. This module therefore reconstructs the
state from each PRIOR game's stored closing `close_p_home`. That does not
create lookahead for a later game: every close in the state is from a game
already completed before the current slate. Same-day state is frozen, so one
game on a date never updates another game on that same date.

SCORING PRICE BASIS. The current game's B2-v13 comparison uses only the saved
pregame snapshot already locked into the ledger: `pregame_p_home` and the
paired saved pregame moneylines. There is deliberately no current-game closing
fallback. That keeps the forward evaluation tied to information available at
decision time.

The reconstructed historical comparison that generated B2 is NOT printed in
this block. This module is the prospective test only.
"""

from collections import defaultdict
import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION BLOCK. If any value changes, create a NEW candidate and
# a new registration date. Do not retune this module in place.
# ---------------------------------------------------------------------------
REGISTERED_ON = "2026-09-18"  # score slates STRICTLY after this date
BASE_MODEL_TAG = "xw+starter_blend_v13"
LOOKBACK_N = 10
Z_THRESHOLD = 1.50
STAKE = 1.0
EARLY_CHECKPOINT = 20
INTERMEDIATE_CHECKPOINT = 40
SUBSTANTIVE_CHECKPOINT = 75

LEDGER = os.path.join("data", "mlb_lean_ledger.csv")

_STATE_REQUIRED = {
    "status", "game_date", "away", "home", "full_away", "full_home",
    "close_p_home",
}
_FORWARD_REQUIRED = {
    "model_tag", "xw_lean", "pregame_p_home",
    "pregame_home_ml", "pregame_away_ml",
}

_STATE_COLS = (
    "tmr10_home", "tmr10_away",
    "tmr10_var_home", "tmr10_var_away",
    "tmr10_z_home", "tmr10_z_away",
    "b2_max_abs_z", "b2_signal", "b2_side", "b2_trigger_type",
)


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _valid_prob(v):
    return _finite(v) and 0.0 < float(v) < 1.0


def _valid_ml(v):
    return _finite(v) and float(v) != 0.0


def _payout(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def _team_state(history):
    """Return (TMR10, variance, z) from the last LOOKBACK_N priced games."""
    if len(history) < LOOKBACK_N:
        return (np.nan, np.nan, np.nan)
    last = history[-LOOKBACK_N:]
    residual = np.asarray([x[0] for x in last], dtype=float)
    pvar = np.asarray([x[1] for x in last], dtype=float)
    tmr = float(residual.mean())
    var = float(pvar.sum() / (LOOKBACK_N ** 2))
    z = float(tmr / math.sqrt(var)) if var > 0 else np.nan
    return (tmr, var, z)


def attach_b2_state(led):
    """Attach deterministic pre-day B2 state to graded ledger rows.

    The returned frame keeps the ledger's original rows and index. State is
    reconstructed chronologically from PRIOR completed games only. Every game
    on a calendar date sees the same pre-day history; results from the entire
    date are appended only after all rows on that date have been assigned
    state. Histories are keyed by season year, which resets state each season.
    """
    if led is None or any(c not in getattr(led, "columns", ()) for c in _STATE_REQUIRED):
        return None

    out = led.copy()
    for c in _STATE_COLS:
        if c == "b2_signal":
            out[c] = False
        else:
            out[c] = np.nan

    full_home = pd.to_numeric(out["full_home"], errors="coerce")
    full_away = pd.to_numeric(out["full_away"], errors="coerce")
    dates = pd.to_datetime(out["game_date"], errors="coerce")

    usable = (
        out["status"].astype(str).eq("graded")
        & dates.notna()
        & full_home.notna()
        & full_away.notna()
        & full_home.ne(full_away)
        & out["home"].notna()
        & out["away"].notna()
    )
    if not usable.any():
        return out

    work = out.loc[usable].copy()
    work["_date"] = dates.loc[usable].dt.date
    work["_season"] = dates.loc[usable].dt.year.astype(int)
    work["_game_sort"] = pd.to_numeric(
        work["game_pk"], errors="coerce"
    ) if "game_pk" in work.columns else np.arange(len(work), dtype=float)
    work = work.sort_values(["_date", "_game_sort"], kind="stable")

    histories = defaultdict(list)

    for (season, day), block in work.groupby(["_season", "_date"], sort=True):
        teams = set(block["home"].astype(str)) | set(block["away"].astype(str))
        states = {
            team: _team_state(histories[(int(season), team)])
            for team in teams
        }

        # Freeze: assign every row first, before any result from this date is
        # allowed into any team's history.
        for idx, row in block.iterrows():
            home = str(row["home"])
            away = str(row["away"])
            th, vh, zh = states[home]
            ta, va, za = states[away]
            out.at[idx, "tmr10_home"] = th
            out.at[idx, "tmr10_away"] = ta
            out.at[idx, "tmr10_var_home"] = vh
            out.at[idx, "tmr10_var_away"] = va
            out.at[idx, "tmr10_z_home"] = zh
            out.at[idx, "tmr10_z_away"] = za

            ready = all(_finite(v) for v in (th, ta, zh, za))
            if not ready:
                continue
            max_abs = max(abs(float(zh)), abs(float(za)))
            out.at[idx, "b2_max_abs_z"] = max_abs
            if max_abs < Z_THRESHOLD or math.isclose(float(th), float(ta), abs_tol=1e-15):
                continue

            side = home if float(th) < float(ta) else away
            home_ext = abs(float(zh)) >= Z_THRESHOLD
            away_ext = abs(float(za)) >= Z_THRESHOLD
            if home_ext and away_ext:
                trigger_type = "both"
            else:
                trigger_z = float(zh) if home_ext else float(za)
                trigger_type = (
                    "negative_extreme" if trigger_z < 0 else "positive_extreme"
                )

            out.at[idx, "b2_signal"] = True
            out.at[idx, "b2_side"] = side
            out.at[idx, "b2_trigger_type"] = trigger_type

        # Only now advance histories using this date's completed games. A game
        # without a valid two-sided close cannot enter a later TMR10 window.
        for _, row in block.iterrows():
            p_home = row.get("close_p_home")
            if not _valid_prob(p_home):
                continue
            p_home = float(p_home)
            home_won = float(row["full_home"]) > float(row["full_away"])
            home_resid = float(home_won) - p_home
            away_resid = float(not home_won) - (1.0 - p_home)
            pvar = p_home * (1.0 - p_home)
            histories[(int(season), str(row["home"]))].append((home_resid, pvar))
            histories[(int(season), str(row["away"]))].append((away_resid, pvar))

    return out


def forward_rows(led=None):
    """Return post-registration actual-v13 rows with B2 state and scoring fields."""
    if led is None:
        if not os.path.exists(LEDGER):
            return None
        led = pd.read_csv(LEDGER, low_memory=False)

    if any(c not in getattr(led, "columns", ()) for c in _FORWARD_REQUIRED):
        return None

    g = attach_b2_state(led)
    if g is None:
        return None

    g = g[
        g["status"].astype(str).eq("graded")
        & g["game_date"].astype(str).gt(REGISTERED_ON)
        & g["model_tag"].astype(str).eq(BASE_MODEL_TAG)
    ].copy()
    if g.empty:
        return g

    for c in ("pregame_p_home", "pregame_home_ml", "pregame_away_ml",
              "full_home", "full_away"):
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g["state_ready"] = (
        g["tmr10_home"].notna()
        & g["tmr10_away"].notna()
        & g["tmr10_z_home"].notna()
        & g["tmr10_z_away"].notna()
    )
    g["price_ready"] = (
        g["pregame_p_home"].map(_valid_prob)
        & g["pregame_home_ml"].map(_valid_ml)
        & g["pregame_away_ml"].map(_valid_ml)
    )
    g["base_ready"] = (
        g["xw_lean"].astype(str).eq(g["home"].astype(str))
        | g["xw_lean"].astype(str).eq(g["away"].astype(str))
    )

    g["b2_interaction"] = "NO_SIGNAL"
    sig = g["b2_signal"].fillna(False).astype(bool)
    g.loc[sig & ~g["base_ready"], "b2_interaction"] = "BASE_UNAVAILABLE"
    agree = sig & g["base_ready"] & (
        g["b2_side"].astype(str) == g["xw_lean"].astype(str)
    )
    disagree = sig & g["base_ready"] & ~agree
    g.loc[agree, "b2_interaction"] = "AGREE"
    g.loc[disagree, "b2_interaction"] = "DISAGREE"

    scorable = sig & g["state_ready"] & g["price_ready"] & g["base_ready"]
    g["b2_scorable"] = scorable

    home_won = g["full_home"] > g["full_away"]
    b2_home = g["b2_side"].astype(str) == g["home"].astype(str)
    v13_home = g["xw_lean"].astype(str) == g["home"].astype(str)

    g["b2_won"] = np.where(scorable, np.where(b2_home, home_won, ~home_won), np.nan)
    g["v13_won"] = np.where(scorable, np.where(v13_home, home_won, ~home_won), np.nan)
    g["b2_p"] = np.where(
        scorable, np.where(b2_home, g["pregame_p_home"], 1.0 - g["pregame_p_home"]),
        np.nan,
    )
    g["v13_p"] = np.where(
        scorable, np.where(v13_home, g["pregame_p_home"], 1.0 - g["pregame_p_home"]),
        np.nan,
    )
    g["b2_ml"] = np.where(
        scorable, np.where(b2_home, g["pregame_home_ml"], g["pregame_away_ml"]),
        np.nan,
    )
    g["v13_ml"] = np.where(
        scorable, np.where(v13_home, g["pregame_home_ml"], g["pregame_away_ml"]),
        np.nan,
    )

    s = g["b2_scorable"].astype(bool)
    if s.any():
        g.loc[s, "b2_residual"] = (
            g.loc[s, "b2_won"].astype(float) - g.loc[s, "b2_p"].astype(float)
        )
        g.loc[s, "v13_residual"] = (
            g.loc[s, "v13_won"].astype(float) - g.loc[s, "v13_p"].astype(float)
        )
        g.loc[s, "b2_profit"] = np.where(
            g.loc[s, "b2_won"].astype(bool),
            STAKE * _payout(g.loc[s, "b2_ml"]),
            -STAKE,
        )
        g.loc[s, "v13_profit"] = np.where(
            g.loc[s, "v13_won"].astype(bool),
            STAKE * _payout(g.loc[s, "v13_ml"]),
            -STAKE,
        )
    return g


def _performance_line(g, label, prefix):
    if not len(g):
        return f"    {label:<13} no scorable rows yet"
    won = g[f"{prefix}_won"].astype(bool)
    p = pd.to_numeric(g[f"{prefix}_p"], errors="coerce")
    resid = pd.to_numeric(g[f"{prefix}_residual"], errors="coerce")
    profit = pd.to_numeric(g[f"{prefix}_profit"], errors="coerce")
    w = int(won.sum())
    n = len(g)
    return (
        f"    {label:<13} n={n:<3d} {w}-{n-w}  mean p {p.mean():.3f}  "
        f"residual {resid.mean()*100:+.1f}pp  {profit.sum():+.2f}u"
    )


def report_lines(led=None):
    """Ledger-report block. Pure -- no printing and no file writes."""
    out = [
        "pre-registered Candidate B2 individual-extreme TMR10 test  "
        f"(registered {REGISTERED_ON}; shadow only)",
        f"    rule — max(|Z_home|, |Z_away|) >= {Z_THRESHOLD:.2f}; "
        "select the lower raw TMR10 team.",
        f"    state — prior {LOOKBACK_N} valid priced games; prior-game closing "
        "close_p_home; same-day frozen; season reset.",
        "    scoring — actual forward xw+starter_blend_v13 only; saved "
        "pregame_p_home + saved pregame MLs; no close fallback.",
        "    interpretation — conditional market-calibration / forecast-error "
        "state test, not a generic mean-reversion claim.",
    ]

    g = forward_rows(led)
    if g is None:
        out.append("    ledger unavailable or missing columns -- not scored")
        return out
    if not len(g):
        out.append(
            "    nothing to score yet; forward sample begins with slates "
            f"strictly after {REGISTERED_ON}."
        )
        return out

    state_ready = int(g["state_ready"].sum())
    signals = g[g["b2_signal"].fillna(False).astype(bool)]
    scorable = signals[signals["b2_scorable"].fillna(False).astype(bool)]
    agree = scorable[scorable["b2_interaction"] == "AGREE"]
    disagree = scorable[scorable["b2_interaction"] == "DISAGREE"]
    out.append(
        f"    eligible forward v13 rows: {len(g)}; state-ready {state_ready}; "
        f"B2 signals {len(signals)}; scorable {len(scorable)} "
        f"({len(agree)} agree, {len(disagree)} disagree)."
    )

    if len(signals):
        mix = signals["b2_trigger_type"].value_counts()
        out.append(
            "    trigger mix — "
            f"negative {int(mix.get('negative_extreme', 0))}, "
            f"positive {int(mix.get('positive_extreme', 0))}, "
            f"both {int(mix.get('both', 0))}."
        )

    if not len(scorable):
        out.append("    no scorable B2 signals yet.")
        return out

    out.append(_performance_line(scorable, "all B2", "b2"))
    out.append("    PRIMARY — B2 vs unchanged v13 on DISAGREE games")
    if not len(disagree):
        out.append("      no disagreements yet.")
    else:
        out.append(_performance_line(disagree, "B2", "b2"))
        out.append(_performance_line(disagree, "v13", "v13"))
        resid_gain = float(
            (disagree["b2_residual"] - disagree["v13_residual"]).mean()
        )
        profit_gain = float(
            (disagree["b2_profit"] - disagree["v13_profit"]).sum()
        )
        out.append(
            f"      paired gain: residual {resid_gain*100:+.1f}pp; "
            f"flat-unit profit {profit_gain:+.2f}u."
        )

    n_dis = len(disagree)
    if n_dis < EARLY_CHECKPOINT:
        checkpoint = (
            f"{n_dis} of {EARLY_CHECKPOINT} disagreements for the early "
            "directional read; no promotion decision."
        )
    elif n_dis < INTERMEDIATE_CHECKPOINT:
        checkpoint = (
            f"{n_dis} disagreements; early checkpoint reached, next stability "
            f"review at {INTERMEDIATE_CHECKPOINT}."
        )
    elif n_dis < SUBSTANTIVE_CHECKPOINT:
        checkpoint = (
            f"{n_dis} disagreements; intermediate checkpoint reached, "
            f"substantive review at {SUBSTANTIVE_CHECKPOINT}+."
        )
    else:
        checkpoint = (
            f"{n_dis} disagreements; substantive-review count reached. "
            "Interpret with clustering / leave-one-team sensitivity."
        )
    out.append(f"    CHECKPOINT: {checkpoint}")
    out.append(
        "    Reconstructed historical B2 results are deliberately excluded "
        "from this forward block."
    )
    return out


if __name__ == "__main__":
    print("\n".join(report_lines()))
