"""Collect per-plate-appearance rows for ONE slate, to make the fixed-window
lineup rescore computable.

`lineup_window_probe` established what the committed artifacts cannot do: the
lineup component is scored against a whole-game team wOBA, and the SP/BP
boundary that defines its phase is endogenous to lineup quality. The registered
test of that (docs/lineup_window_registration.md) wanted the lineup rescored
over a FIXED window -- the first 18 batters faced, regardless of when the
starter left -- and could not run, because a box score gives a whole-game total
and a starter-allowed total and neither is a PREFIX of the game.

This module fetches the one thing that supplies a prefix: StatsAPI's play-by-
play. It is BACKFILL, not lookahead. A finished game's play-by-play is an
immutable historical fact in exactly the sense `actuals_backfill` argues its box
scores are -- fetching 2026-08-29 today returns what it returned then. The
forbidden direction is re-deriving a PREDICTION from today's Savant leaderboard,
and nothing here touches the prediction side.

WHAT IT PRODUCES, and why each piece exists rather than being a sweep:

  first 18 batters faced   the registered fixed window. 18 was chosen in
                           advance and no other window is computed here --
                           adding a second one would turn a pre-registered
                           test into the search this repo has already been
                           fooled by twice.
  whole game               what the lineup component is scored against TODAY,
                           so the comparison is on identical rows; and the
                           validation target, since it must equal the ledger's
                           own `act_woba_<side>` to the last decimal.
  the starter's window     validation only: its component counts must equal
                           the ledger's `act_sp_*_<side>`, and its length must
                           equal `act_sp_bf_<side>`. This is what proves the
                           per-PA reconstruction is faithful rather than
                           plausible.

WHY THE VALIDATION IS THE POINT. Mapping StatsAPI `eventType` strings onto
wOBA components is the step that fails silently: a mis-mapped event shifts a
rate by a little on every game and looks like a finding. So this module never
reports a rate it has not reconciled. Every game is checked twice -- against
`parse_boxscore` on the SAME payload (catching the mapping) and against the
committed ledger row (catching the join) -- and a game failing either is
excluded from the output and named in the report. CLAUDE.md's rule that a count
derived by subtraction cannot carry a name you did not measure is what forces
this: `ab` here IS derived, as PA minus the non-AB outcomes, and it earns the
name `ab` only because the box score's own `atBats` confirms it.

WHAT IT CANNOT DO. It supplies the ACTUAL half of the fixed-window rescore and
not the PREDICTED half. 18 batters faced is exactly two turns through the
order, so slot-PA weights go uniform and the matching prediction is the
UNWEIGHTED nine-hitter mean -- while the ledger persists only the slot-PA
WEIGHTED composite (`opp_xwoba_neutral`), and weighted-minus-unweighted depends
on the covariance of slot weight with hitter rate, which the stored
`(mean, sd)` does not determine. Rebuilding the per-hitter frame for a past
slate would need that slate's Savant leaderboard, which `.savant_cache/` is
gitignored to forbid. So the analysis this enables holds the prediction FIXED
and varies only the actual, which is the clean form of the registered question
anyway: does scoring against a fixed window move the slope? The residual
weighting mismatch is a measurement error in the predictor that attenuates a
slope slightly; it is stated in the report rather than corrected.

ONE SLATE CANNOT ANSWER THE QUESTION. 17 games is 34 side-games against the
598 the component block scores. This is a feasibility and fidelity run: it
establishes that the reconstruction reconciles and what the pipeline costs.
Read the validation, not the slope.

Reachability: StatsAPI is blocked from the dev sandbox (the proxy answers 403
to CONNECT), so this cannot be smoke-tested locally and runs on a GitHub
runner -- the same constraint every live-API probe in this repo carries. The
event map below is therefore UNVERIFIED against the live endpoint until that
workflow runs; the validation gate is what makes a wrong map loud.

Usage:
    python lineup_window_collect.py --date 2026-08-29
    python lineup_window_collect.py --date 2026-08-29 --out pa.csv --report r.txt
"""
import argparse
import time

import numpy as np
import pandas as pd

import actuals_backfill as ab

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"
LEDGER = "data/mlb_lean_ledger.csv"
THROTTLE_S = 0.20
TIMEOUT = 20
TRIES = 3

# The registered window. Fixed in advance; see the module docstring. Not a
# parameter, deliberately -- exposing it as one is how a pre-registered window
# becomes a swept one.
FIXED_WINDOW = 18

# StatsAPI `result.eventType` -> what the plate appearance was.
#
# Spelled out rather than inferred, and the categories are the ones
# `woba_from_components` consumes. `ab` is NOT listed: it is derived as
# PA - bb - hbp - sf - sh - ci, the standard identity, and validated against
# the box score's own `atBats` before any rate built on it is reported.
#
# "out" covers every way a PA ends in an at-bat that is not a hit -- strikeouts,
# groundouts, fielder's choices, errors, double plays. They are identical for
# wOBA purposes (all denominator, no numerator), so enumerating them
# individually would add ways to be wrong without adding information.
_EVENT = {
    "single": "1b", "double": "2b", "triple": "3b", "home_run": "hr",
    "walk": "bb", "intent_walk": "ibb", "hit_by_pitch": "hbp",
    "sac_fly": "sf", "sac_fly_double_play": "sf",
    "sac_bunt": "sh", "sac_bunt_double_play": "sh",
    "catcher_interf": "ci",
    "strikeout": "out", "strikeout_double_play": "out",
    "strikeout_triple_play": "out",
    "field_out": "out", "force_out": "out", "grounded_into_double_play": "out",
    "grounded_into_triple_play": "out", "double_play": "out",
    "triple_play": "out", "fielders_choice": "out",
    "fielders_choice_out": "out", "field_error": "out",
    "batter_interference": "out", "fan_interference": "out",
    "sac_fly_error": "sf", "other_out": "out",
}

# Events that are NOT a plate appearance: base-running plays that StatsAPI
# lists as their own entry. Named so an unrecognised event is distinguishable
# from a known non-PA one -- an unknown string must reach the report, not be
# quietly treated as either.
_NOT_A_PA = {
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "wild_pitch", "passed_ball", "balk", "defensive_indiff",
    "other_advance", "runner_double_play", "run_scoring_double_play",
    "game_advisory", "ejection", "pitching_substitution",
    "offensive_substitution", "defensive_substitution",
    "defensive_switch", "pitcher_switch", "injury", "stolen_base",
    "caught_stealing", "pickoff_error_1b", "pickoff_error_2b",
    "pickoff_error_3b", "error",
}

_COMPONENTS = ("ab", "h", "2b", "3b", "hr", "bb", "ibb", "hbp", "sf")


def _get_json(url, tries=TRIES):
    import requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5 * (k + 1))
    raise last


def plate_appearances(feed):
    """Ordered PA rows for one game, or (None, reason) when the feed is unusable.

    Returns (rows, unmapped) where `rows` is a list of dicts in the order the
    plate appearances occurred, each carrying the batting side, the batter, the
    pitcher, whether that pitcher was the starter, and the wOBA category. The
    ORDER is the whole point -- a box score has the same events with the
    sequence thrown away, which is why a prefix is not recoverable from one.

    `unmapped` collects any `eventType` the map above does not know, by name.
    They are counted rather than guessed at: an unknown event silently treated
    as an out would move every rate a little, which is the failure mode this
    module exists to make loud.
    """
    plays = (((feed or {}).get("liveData") or {}).get("plays") or {}).get("allPlays")
    if not plays:
        return None, ["no allPlays in feed"]
    rows, unmapped = [], []
    starters = _starter_ids(feed)
    for p in plays:
        res = p.get("result") or {}
        et = res.get("eventType")
        if et is None:
            continue
        if et in _NOT_A_PA:
            continue
        cat = _EVENT.get(et)
        if cat is None:
            unmapped.append(et)
            continue
        about = p.get("about") or {}
        half = about.get("halfInning")
        if half not in ("top", "bottom"):
            unmapped.append(f"halfInning={half!r}")
            continue
        # Top of the inning is the AWAY team batting.
        bat_side = "away" if half == "top" else "home"
        pit_side = "home" if bat_side == "away" else "away"
        mu = p.get("matchup") or {}
        pid = ((mu.get("pitcher") or {}).get("id"))
        rows.append({
            "bat_side": bat_side,
            "pit_side": pit_side,
            "at_bat_index": about.get("atBatIndex"),
            "inning": about.get("inning"),
            "batter_id": (mu.get("batter") or {}).get("id"),
            "pitcher_id": pid,
            "vs_starter": bool(pid is not None
                               and pid == starters.get(pit_side)),
            "event_type": et,
            "cat": cat,
        })
    return rows, unmapped


def _starter_ids(feed):
    """{'away': pitcher_id, 'home': pitcher_id} by the box score's own
    `gamesStarted`, the same rule `actuals_backfill.parse_boxscore` uses --
    never position in the pitcher list, because an opener is listed first and
    IS the starter while a promoted reliever is not."""
    box = ((feed or {}).get("liveData") or {}).get("boxscore") or {}
    out = {}
    for side in ("away", "home"):
        t = ((box.get("teams") or {}).get(side)) or {}
        for p in (t.get("players") or {}).values():
            ps = ((p.get("stats") or {}).get("pitching")) or {}
            if ab._f(ps.get("gamesStarted")):
                out[side] = ((p.get("person") or {}).get("id"))
                break
    return out


def components(rows):
    """wOBA components for a list of PA rows. `ab` is derived by identity.

    PA - bb(all) - hbp - sf - sh - ci = ab. That subtraction is licensed only
    by the reconciliation in `validate`: on its own it would be a count wearing
    a name nobody measured.
    """
    c = {k: 0.0 for k in _COMPONENTS}
    n_pa = len(rows)
    sh = ci = 0
    for r in rows:
        k = r["cat"]
        if k == "1b":
            c["h"] += 1
        elif k in ("2b", "3b", "hr"):
            c["h"] += 1
            c[k] += 1
        elif k == "bb":
            c["bb"] += 1
        elif k == "ibb":
            c["bb"] += 1      # box-score baseOnBalls INCLUDES intentional
            c["ibb"] += 1
        elif k == "hbp":
            c["hbp"] += 1
        elif k == "sf":
            c["sf"] += 1
        elif k == "sh":
            sh += 1
        elif k == "ci":
            ci += 1
    c["ab"] = n_pa - c["bb"] - c["hbp"] - c["sf"] - sh - ci
    c["pa"] = n_pa
    return c


def validate(game_pk, feed, rows, led_row):
    """Reconcile the PA reconstruction twice. Returns a list of failures.

    Against `parse_boxscore` on the SAME payload -- catches the event map.
    Against the committed LEDGER row -- catches the join and the slate pick.
    Two sources because they fail differently: a wrong map reconciles with
    neither, a wrong gamePk reconciles with the box score and not the ledger.
    """
    fails = []
    box = ((feed or {}).get("liveData") or {}).get("boxscore") or {}
    parsed = ab.parse_boxscore(box)
    for bat in ("away", "home"):
        got = components([r for r in rows if r["bat_side"] == bat])
        want = parsed.get(bat) or {}
        for f in ("pa",) + _COMPONENTS:
            w = ab._f(want.get(f))
            if w is None:
                continue
            if abs(got[f] - w) > 1e-9:
                fails.append(f"{game_pk} {bat} {f}: pbp {got[f]:.0f} "
                             f"!= boxscore {w:.0f}")
        if led_row is not None:
            lw = ab._f(led_row.get(f"act_woba_{bat}"))
            mine = ab.woba_from_components(got)
            if lw is not None and mine is not None and abs(mine - lw) > 1e-6:
                fails.append(f"{game_pk} {bat} woba: pbp {mine:.6f} "
                             f"!= ledger {lw:.6f}")
        # The starter's window: length against act_sp_bf, components against
        # act_sp_*. This is the check that the ORDERING is right -- the totals
        # above would reconcile even if the sequence were shuffled.
        pit = "home" if bat == "away" else "away"
        sp_rows = [r for r in rows if r["bat_side"] == bat and r["vs_starter"]]
        sp = components(sp_rows)
        if led_row is not None:
            bf = ab._f(led_row.get(f"act_sp_bf_{pit}"))
            if bf is not None and abs(sp["pa"] - bf) > 1e-9:
                fails.append(f"{game_pk} {pit} SP bf: pbp {sp['pa']:.0f} "
                             f"!= ledger {bf:.0f}")
            for f in _COMPONENTS:
                lv = ab._f(led_row.get(f"act_sp_{f}_{pit}"))
                if lv is not None and abs(sp[f] - lv) > 1e-9:
                    fails.append(f"{game_pk} {pit} SP {f}: pbp {sp[f]:.0f} "
                                 f"!= ledger {lv:.0f}")
    return fails


def collect(date, ledger_path=LEDGER, verbose=True):
    """Fetch and reconcile every v12 game on `date`. Returns (pa_df, summary)."""
    led = pd.read_csv(ledger_path)
    d = led[led["game_date"].astype(str) == str(date)]
    d = d[d["gamePk"].notna()]
    if d.empty:
        raise SystemExit(f"no ledger rows with a gamePk on {date}")

    pa_rows, per_game, all_fails, all_unmapped = [], [], [], []
    for _, r in d.iterrows():
        gpk = int(r["gamePk"])
        try:
            feed = _get_json(FEED_URL.format(gamePk=gpk))
        except Exception as e:  # noqa: BLE001
            all_fails.append(f"{gpk}: feed fetch failed ({type(e).__name__})")
            continue
        rows, unmapped = plate_appearances(feed)
        if rows is None:
            all_fails.append(f"{gpk}: {unmapped[0]}")
            continue
        if unmapped:
            all_unmapped.extend(unmapped)
        fails = validate(gpk, feed, rows, r)
        ok = not fails and not unmapped
        all_fails.extend(fails)
        for i, row in enumerate(rows):
            row = dict(row, game_pk=gpk, game_date=str(date), reconciled=ok)
            pa_rows.append(row)
        # Sequence position WITHIN each side's offense: this is what makes a
        # prefix definable, and it is why the CSV is worth keeping.
        for bat in ("away", "home"):
            k = 0
            for row in pa_rows[-len(rows):]:
                if row["bat_side"] == bat:
                    k += 1
                    row["pa_seq"] = k
        per_game.append({"game_pk": gpk, "reconciled": ok,
                         "n_pa": len(rows), "n_fail": len(fails)})
        if verbose:
            print(f"  {gpk}: {len(rows)} PA  "
                  + ("OK" if ok else f"FAILED ({len(fails)} checks)"))
        time.sleep(THROTTLE_S)

    pa = pd.DataFrame(pa_rows)
    return pa, {"per_game": pd.DataFrame(per_game), "fails": all_fails,
                "unmapped": sorted(set(all_unmapped)), "date": str(date),
                "n_games": len(d)}


def windows(pa, led, date):
    """Per side-game: the fixed-window actual beside the whole-game one.

    One row per (game, batting side), carrying the lineup component's stored
    PREDICTION unchanged and two ACTUALS -- the whole game, which is what the
    component block scores today, and the first `FIXED_WINDOW` batters faced,
    which is the registered alternative. Only the actual varies; that is the
    clean form of the question and the only form this data supports (see the
    module docstring on the predicted half).
    """
    out = []
    d = led[led["game_date"].astype(str) == str(date)]
    for _, r in d.iterrows():
        gpk = ab._f(r.get("gamePk"))
        if gpk is None:
            continue
        g = pa[(pa.game_pk == int(gpk)) & (pa.reconciled)]
        if g.empty:
            continue
        for bat in ("away", "home"):
            pit = "home" if bat == "away" else "away"
            side = g[g.bat_side == bat].sort_values("pa_seq")
            if side.empty:
                continue
            full = ab.woba_from_components(components(side.to_dict("records")))
            head = side[side.pa_seq <= FIXED_WINDOW].to_dict("records")
            fixed = (ab.woba_from_components(components(head))
                     if len(head) == FIXED_WINDOW else None)
            out.append({
                "game_pk": int(gpk), "bat_side": bat,
                "pred": ab._f(r.get(f"opp_xwoba_neutral_{pit}")),
                "act_full": full,
                "act_fixed": fixed,
                "n_pa": len(side),
                "sp_bf": ab._f(r.get(f"act_sp_bf_{pit}")),
                "distinct_batters_in_window": len({x["batter_id"] for x in head}),
            })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-08-29",
                   help="slate date (default: the largest complete v12 slate)")
    p.add_argument("--out", default="lineup_window_pa.csv")
    p.add_argument("--report", default="lineup_window_collect_report.txt")
    a = p.parse_args()

    print(f"collecting play-by-play for {a.date}")
    pa, summ = collect(a.date)
    led = pd.read_csv(LEDGER)
    w = windows(pa, led, a.date) if not pa.empty else pd.DataFrame()

    lines = [f"LINEUP WINDOW COLLECT — slate {summ['date']}",
             f"ledger games on this slate: {summ['n_games']}",
             f"plate appearances reconstructed: {len(pa)}",
             ""]
    pg = summ["per_game"]
    ok = int(pg["reconciled"].sum()) if not pg.empty else 0
    lines.append(f"RECONCILIATION  {ok}/{len(pg)} games reconcile against BOTH "
                 f"the box score on the same payload and the committed ledger")
    if summ["unmapped"]:
        lines.append(f"  UNMAPPED eventType values (map is wrong or incomplete): "
                     f"{', '.join(summ['unmapped'])}")
    else:
        lines.append("  no unmapped eventType values")
    for f in summ["fails"][:40]:
        lines.append(f"  FAIL {f}")
    if len(summ["fails"]) > 40:
        lines.append(f"  ... and {len(summ['fails']) - 40} more")
    lines.append("")

    if not w.empty:
        good = w[w.act_fixed.notna() & w.pred.notna()]
        lines.append(f"side-games with a stored prediction and a full "
                     f"{FIXED_WINDOW}-batter window: {len(good)} of {len(w)}")
        if not good.empty:
            db = good["distinct_batters_in_window"]
            lines.append(f"  distinct batters inside the window: "
                         f"min {db.min()} median {db.median():.0f} max {db.max()} "
                         f"(9 = exactly two turns through the order)")
            for col, label in (("act_full", "whole game (scored today)"),
                               ("act_fixed", f"first {FIXED_WINDOW} batters")):
                cal = ab.calibration(good["pred"], good[col])
                if cal:
                    r = float(np.corrcoef(good["pred"], good[col])[0, 1])
                    mae = float((good[col] - good["pred"]).abs().mean())
                    lines.append(f"  {label:<28s} slope {cal['slope']:+.3f} "
                                 f"+/- {cal['se_slope']:.3f}  corr {r:+.4f}  "
                                 f"MAE {mae:.4f}")
            lines.append("")
            lines.append("  ONE SLATE. Read the reconciliation, not the slopes:")
            lines.append(f"  {len(good)} side-games against the 598 the component "
                         f"block scores, so these ses are ~4x the report's and")
            lines.append("  separate nothing. The prediction is also the slot-PA "
                         "WEIGHTED composite while the fixed")
            lines.append("  window is two whole turns, i.e. unweighted -- a "
                         "predictor mismatch that attenuates a slope.")

    open(a.report, "w").write("\n".join(lines) + "\n")
    if not pa.empty:
        pa.to_csv(a.out, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
