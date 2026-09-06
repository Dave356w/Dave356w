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
reports a rate it has not reconciled. Every game is checked against TWO
sources, and the two answer different questions, which is why their failures
are reported separately rather than pooled:

  vs `parse_boxscore` on the SAME payload   the event MAP. A disagreement here
                                            means this module read the game
                                            wrong, and the game is unusable.
  vs the committed LEDGER row               the SOURCE. The map can be perfect
                                            and this still disagree, because
                                            `actuals_backfill._fill` is
                                            write-once and StatsAPI revises
                                            box scores after the fact.

That second case is real and was hit on the first run: 2026-08-29 ARI@SF
(823176) reconciles against its own box score and differs from the ledger by
exactly one single reclassified out of the hit column -- an official scoring
change, arriving after the row was backfilled and correctly never overwriting
it. Calling that a "reconciliation failure" would name the instrument's own
correctness after someone else's stat correction, so it is classified as a
REVISION and the game is still used: its PBP is internally consistent, and both
actuals in the comparison below come from that one consistent source.

CLAUDE.md's rule that a count derived by subtraction cannot carry a name you
did not measure is what forces all of this: `ab` here IS derived, as PA minus
the non-AB outcomes, and it earns the name `ab` only because the box score's
own `atBats` confirms it.

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

# One `feed/live` per game, and the full v12 family is ~300 of them. Bounded
# the way `actuals_backfill` bounds its box-score loop, and for the same
# reason: an upstream hang would otherwise burn the job's whole budget. Neither
# bound loses anything -- this module writes no ledger row, so a short run is
# simply a smaller sample, named as such in the report.
BUDGET_S = 1500.0
MAX_CONSEC_FAIL = 5

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
    """Reconcile the PA reconstruction against two sources.

    Returns {"map": [...], "ledger": [...]} -- kept apart because they mean
    different things. A `map` failure says this module read the game wrong and
    the game is unusable. A `ledger` failure with an empty `map` says the two
    agree about the game and the stored row is older than the box score, i.e.
    an upstream scoring revision that `_fill`'s write-once rule correctly did
    not absorb. Pooling them would let a stat correction read as a defect here,
    and worse, would let a real defect hide inside a pile of them.
    """
    fails = {"map": [], "ledger": []}
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
                fails["map"].append(f"{game_pk} {bat} {f}: pbp {got[f]:.0f} "
                                    f"!= boxscore {w:.0f}")
        if led_row is not None:
            lw = ab._f(led_row.get(f"act_woba_{bat}"))
            mine = ab.woba_from_components(got)
            if lw is not None and mine is not None and abs(mine - lw) > 1e-6:
                fails["ledger"].append(f"{game_pk} {bat} woba: pbp {mine:.6f} "
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
                fails["ledger"].append(f"{game_pk} {pit} SP bf: pbp "
                                       f"{sp['pa']:.0f} != ledger {bf:.0f}")
            for f in _COMPONENTS:
                lv = ab._f(led_row.get(f"act_sp_{f}_{pit}"))
                if lv is not None and abs(sp[f] - lv) > 1e-9:
                    fails["ledger"].append(f"{game_pk} {pit} SP {f}: pbp "
                                           f"{sp[f]:.0f} != ledger {lv:.0f}")
    return fails


def ledger_scope(led, date=None, tags=None):
    """The rows a run should collect: one slate, or a whole model family.

    Restricted to games that are BACKFILLED -- a stored `act_woba` on both
    sides. That is not a convenience filter: it is exactly the population the
    component block scores, so a run over this scope is comparable to the
    number it is meant to be read against, and every game has something to
    reconcile the play-by-play against. A pending game would have neither.
    """
    d = led[led["gamePk"].notna()]
    if date:
        d = d[d["game_date"].astype(str) == str(date)]
    if tags:
        d = d[d["model_tag"].astype(str).isin(set(tags))]
    return d[d["act_woba_away"].notna() & d["act_woba_home"].notna()]


def collect(scope, verbose=True):
    """Fetch and reconcile every game in `scope`. Returns (pa_df, summary)."""
    d = scope
    if d.empty:
        raise SystemExit("no ledger rows in scope")

    pa_rows, per_game, all_unmapped = [], [], []
    all_fails = {"map": [], "ledger": [], "fetch": []}
    started = time.monotonic()
    consec = 0
    stopped = None
    for n_done, (_, r) in enumerate(d.iterrows()):
        if time.monotonic() - started > BUDGET_S:
            stopped = f"time budget {BUDGET_S:.0f}s reached after {n_done} games"
            break
        if consec >= MAX_CONSEC_FAIL:
            stopped = f"{consec} consecutive fetch failures"
            break
        gpk = int(r["gamePk"])
        try:
            feed = _get_json(FEED_URL.format(gamePk=gpk))
        except Exception as e:  # noqa: BLE001
            # A fetch failure is not a map failure and not a revision: it is a
            # game this run does not have. Counted so a short sample cannot
            # pass for a complete one, and the consecutive-failure stop below
            # treats an outage as one signal rather than 299.
            all_fails["fetch"].append(f"{gpk}: feed fetch failed "
                                      f"({type(e).__name__})")
            consec += 1
            time.sleep(THROTTLE_S)
            continue
        consec = 0
        rows, unmapped = plate_appearances(feed)
        if rows is None:
            all_fails["map"].append(f"{gpk}: {unmapped[0]}")
            continue
        if unmapped:
            all_unmapped.extend(unmapped)
        fails = validate(gpk, feed, rows, r)
        # Usable when THIS module read the game right. A ledger disagreement
        # with a clean map is the source having moved under a write-once row,
        # which says nothing about the reconstruction -- see the docstring.
        ok = not fails["map"] and not unmapped
        revised = ok and bool(fails["ledger"])
        all_fails["map"].extend(fails["map"])
        all_fails["ledger"].extend(fails["ledger"])
        if fails["map"]:
            # A map failure names a count, not a cause. The event histogram is
            # what turns "one PA too many" into a culprit -- without it the
            # next step is guessing at which eventType is miscategorised, and
            # the games cannot be refetched from the dev sandbox to look.
            from collections import Counter
            for bat in ("away", "home"):
                h = Counter(r2["event_type"] for r2 in rows
                            if r2["bat_side"] == bat)
                all_fails["map"].append(
                    f"  {gpk} {bat} events: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(h.items())))
        for i, row in enumerate(rows):
            row = dict(row, game_pk=gpk, game_date=str(r.get("game_date")),
                       reconciled=ok, ledger_revised=revised)
            pa_rows.append(row)
        # Sequence position WITHIN each side's offense: this is what makes a
        # prefix definable, and it is why the CSV is worth keeping.
        for bat in ("away", "home"):
            k = 0
            for row in pa_rows[-len(rows):]:
                if row["bat_side"] == bat:
                    k += 1
                    row["pa_seq"] = k
        per_game.append({"game_pk": gpk, "reconciled": ok, "revised": revised,
                         "n_pa": len(rows),
                         "n_map_fail": len(fails["map"]),
                         "n_ledger_fail": len(fails["ledger"])})
        if verbose and (not ok or revised or len(per_game) % 25 == 0):
            note = ("OK" if ok and not revised else
                    "OK (ledger row predates a scoring revision)" if revised
                    else f"MAP FAILURE ({len(fails['map'])} checks)")
            print(f"  [{len(per_game):3d}/{len(d)}] {gpk}: {len(rows)} PA  {note}")
        time.sleep(THROTTLE_S)

    pa = pd.DataFrame(pa_rows)
    return pa, {"per_game": pd.DataFrame(per_game), "fails": all_fails,
                "unmapped": sorted(set(all_unmapped)), "stopped": stopped,
                "n_games": len(d),
                "slates": int(d["game_date"].nunique())}


def windows(pa, scope):
    """Per side-game: the fixed-window actual beside the whole-game one.

    One row per (game, batting side), carrying the lineup component's stored
    PREDICTION unchanged and two ACTUALS -- the whole game, which is what the
    component block scores today, and the first `FIXED_WINDOW` batters faced,
    which is the registered alternative. Only the actual varies; that is the
    clean form of the question and the only form this data supports (see the
    module docstring on the predicted half).
    """
    out = []
    for _, r in scope.iterrows():
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
                "game_date": str(r.get("game_date")),
                "pred": ab._f(r.get(f"opp_xwoba_neutral_{pit}")),
                "act_full": full,
                "act_fixed": fixed,
                "n_pa": len(side),
                "sp_bf": ab._f(r.get(f"act_sp_bf_{pit}")),
                "distinct_batters_in_window": len({x["batter_id"] for x in head}),
                # The ledger's own stored actual, carried so the report can say
                # how far the play-by-play reconstruction sits from what the
                # component block actually scores -- at scale, rather than
                # game by game.
                "act_ledger": ab._f(r.get(f"act_woba_{bat}")),
            })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None,
                   help="one slate; omit to run the whole model family")
    p.add_argument("--tags", default=None,
                   help="comma-separated model tags (default: RECORD_TAGS)")
    p.add_argument("--out", default="lineup_window_pa.csv")
    p.add_argument("--report", default="lineup_window_collect_report.txt")
    a = p.parse_args()

    from build_site import RECORD_TAGS
    tags = tuple(a.tags.split(",")) if a.tags else RECORD_TAGS
    led = pd.read_csv(LEDGER)
    scope = ledger_scope(led, date=a.date, tags=tags)

    label = a.date if a.date else f"family {', '.join(sorted(set(tags)))}"
    print(f"collecting play-by-play for {label}: "
          f"{len(scope)} games over {scope['game_date'].nunique()} slates")
    pa, summ = collect(scope)
    w = windows(pa, scope) if not pa.empty else pd.DataFrame()

    lines = [f"LINEUP WINDOW COLLECT — {label}",
             f"ledger games in scope: {summ['n_games']} "
             f"over {summ['slates']} slates",
             f"plate appearances reconstructed: {len(pa)}"]
    if summ.get("stopped"):
        lines.append(f"  STOPPED EARLY: {summ['stopped']} — this is a partial "
                     f"sample, not a smaller population")
    nf = len(summ["fails"].get("fetch", []))
    if nf:
        lines.append(f"  {nf} games could not be fetched at all — also a "
                     f"partial sample; they are not map failures")
        for f in summ["fails"]["fetch"][:10]:
            lines.append(f"    FETCH {f}")
    lines.append("")

    pg = summ["per_game"]
    ok = int(pg["reconciled"].sum()) if not pg.empty else 0
    rev = int(pg["revised"].sum()) if not pg.empty else 0
    lines.append(f"EVENT MAP       {ok}/{len(pg)} fetched games reconcile "
                 f"against the box score on the SAME payload")
    if summ["unmapped"]:
        lines.append(f"  UNMAPPED eventType values (map is wrong or incomplete): "
                     f"{', '.join(summ['unmapped'])}")
    else:
        lines.append("  no unmapped eventType values")
    for f in summ["fails"]["map"][:60]:
        lines.append(f"  MAP FAIL {f}")
    if len(summ["fails"]["map"]) > 60:
        lines.append(f"  ... and {len(summ['fails']['map']) - 60} more")
    lines.append("")
    lines.append(f"SOURCE DRIFT    {rev} of those carry a ledger row that "
                 f"predates a scoring revision")
    lines.append("  Not a defect here and not one there: StatsAPI revises box "
                 "scores after the fact and")
    lines.append("  `actuals_backfill._fill` is write-once by design, so the "
                 "stored actual keeps the original.")
    lines.append("  These games ARE used -- their PBP is internally consistent "
                 "and both actuals below come")
    lines.append("  from that one source.")
    lines.append("")

    if not w.empty:
        good = w[w.act_fixed.notna() & w.pred.notna() & w.act_full.notna()]
        lines.append(f"side-games with a stored prediction and a full "
                     f"{FIXED_WINDOW}-batter window: {len(good)} of {len(w)}")
        if not good.empty:
            db = good["distinct_batters_in_window"]
            lines.append(f"  distinct batters inside the window: "
                         f"min {db.min()} median {db.median():.0f} max {db.max()} "
                         f"(9 = exactly two turns through the order)")
            drift = (good["act_full"] - good["act_ledger"]).abs()
            lines.append(f"  |pbp whole-game − ledger actual|: "
                         f"max {drift.max():.6f}, "
                         f"{int((drift > 1e-9).sum())} of {len(good)} nonzero")
            lines.append("")

            # The number this is read against is computed here, not quoted:
            # component slopes move every time the bot grades a slate, and a
            # literal in a report is the constants-frozen-from-data defect.
            comp = ab.paired_components(led[led["model_tag"].astype(str)
                                            .isin(set(tags))])
            comp = comp[comp.component == "lineup"]
            base = ab.calibration(comp["pred"], comp["act"]) if not comp.empty else None
            if base:
                r = float(np.corrcoef(comp["pred"], comp["act"])[0, 1])
                mae = float((comp["act"] - comp["pred"]).abs().mean())
                lines.append(f"  {'REPORT (ledger actual, all rows)':<38s} "
                             f"n={len(comp):<4d} slope {base['slope']:+.3f} "
                             f"+/- {base['se_slope']:.3f}  corr {r:+.4f}  "
                             f"MAE {mae:.4f}")

            for col, label2 in (("act_full", "whole game (what is scored today)"),
                                ("act_fixed", f"first {FIXED_WINDOW} batters faced")):
                cal = ab.calibration(good["pred"], good[col])
                if cal:
                    r = float(np.corrcoef(good["pred"], good[col])[0, 1])
                    mae = float((good[col] - good["pred"]).abs().mean())
                    lines.append(f"  {label2:<38s} n={len(good):<4d} "
                                 f"slope {cal['slope']:+.3f} "
                                 f"+/- {cal['se_slope']:.3f}  corr {r:+.4f}  "
                                 f"MAE {mae:.4f}")
            lines.append("")
            lines.append("  The first line is the component block's own number, "
                         "recomputed here rather than quoted.")
            lines.append("  The second is this collection reproducing it from "
                         "play-by-play on the rows that")
            lines.append("  survived; the two differing by more than the "
                         "revisions above would be a defect.")
            lines.append("  The third is the registered fixed window. Only the "
                         "ACTUAL changes between them --")
            lines.append("  the prediction is the same stored column throughout.")
            lines.append("")
            lines.append("  CAVEAT that does not shrink with n: the prediction "
                         "is the slot-PA WEIGHTED")
            lines.append("  composite, while 18 batters is two whole turns and "
                         "so an unweighted mix. That")
            lines.append("  mismatch is measurement error in the predictor and "
                         "attenuates the third slope")
            lines.append("  toward zero. It cannot be corrected from committed "
                         "artifacts -- see the module")
            lines.append("  docstring on why the predicted half is not "
                         "recoverable for a past slate.")

    open(a.report, "w").write("\n".join(lines) + "\n")
    if not pa.empty:
        pa.to_csv(a.out, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
