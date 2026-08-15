#!/usr/bin/env python3
# ============================================================
# grade_leans.py — CI grading ledger for the matchup site
#
# Companion to build_site.py. Requires build_site.py to dump the day's
# model outputs (3-line patch, see MATCHUP_SITE.md / below):
#
#     os.makedirs("data", exist_ok=True)
#     matchup_df.to_csv(f"data/leans_{SLATE_DATE}_xw.csv", index=False)
#     if matchup_platoon_df is not None and not matchup_platoon_df.empty:
#         matchup_platoon_df.to_csv(f"data/leans_{SLATE_DATE}_pl.csv", index=False)
#
# This script then, on every CI run:
#   INGEST : any data/leans_*_xw.csv not yet ledgered -> pending rows.
#            Re-runs on the same date REFRESH still-pending rows only when
#            the dump carries a snapshot timestamp before scheduled first
#            pitch (handles pregame SP scratches / lineup swaps). Late and
#            legacy-unverified refreshes are rejected; graded rows are never
#            touched.
#   GRADE  : all pending rows via schedule?hydrate=linescore, one call per
#            date. Full-game + F5 (innings 1-5). Live games stay pending;
#            postponed/cancelled -> void.
#   REPORT : stdout (Actions log) + data/ledger_report.txt.
#
# Ledger persists at data/mlb_lean_ledger.csv — commit data/ back to the
# repo in the workflow (contents: write) so state survives between runs:
#
#     - name: Grade leans
#       run: python grade_leans.py
#     - name: Commit ledger
#       run: |
#         git config user.name  "github-actions[bot]"
#         git config user.email "github-actions[bot]@users.noreply.github.com"
#         git add data/
#         git diff --cached --quiet || git commit -m "ledger $(date -u +%F)"
#         git push
#
# SP-vs-lineup weight fit: logs d_lineup / d_sp per game; once >= N_FIT_MIN
# graded F5 decisions accumulate, fits logit(home F5 win) ~ d_lineup + d_sp.
# The symmetric multiplicative-ratio matchup gives both components equal
# first-order weight, and the fit reports the CONTRAST that tests it --
# b_lineup - b_sp*(sd_lu/sd_sp), zero under equal weight -- with a standard
# error. A stable departure would motivate the reweight net_w = d_lineup +
# w*d_sp; the contrast is the evidence, and it is diagnostic output only:
# nothing here feeds back into a lean, a delta or a grade.
# ============================================================
import glob
import math
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from market_backfill import MARKET_COLS, attach_market, metric_label
from actuals_backfill import (ACTUAL_COLS, attach_actuals, actuals_summary,
                              actuals_family_line, components_summary,
                              slate_lines)

DATA_DIR    = os.environ.get("DATA_DIR", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
REPORT_PATH = os.path.join(DATA_DIR, "ledger_report.txt")
MODEL_TAG   = os.environ.get("MODEL_TAG", "xw+plat_consol_v12")
MODEL_METRIC_LABEL = os.environ.get(
    "MODEL_METRIC_LABEL",
    "wOBA" if MODEL_TAG.startswith("woba+") else "xwOBA",
)
if MODEL_TAG.startswith("woba+") != (MODEL_METRIC_LABEL == "wOBA"):
    raise RuntimeError("MODEL_TAG and MODEL_METRIC_LABEL describe different metrics")
_RECORD_FAMILIES = {
    # v3 changed only ledger locking/identity; its prediction math is v2.
    "xw+plat_consol_v3": ("xw+plat_consol_v2", "xw+plat_consol_v3"),
    # v4 re-weights lineup composites by expected PA per batting-order slot
    # (was season BBE / split PA); that changes prediction math, so it starts a
    # fresh record family and never mixes with v2/v3 in the ledger or weight fit.
    # v5 adds empirical-Bayes xwOBA shrinkage (batters + starter) on top of v4;
    # another prediction-math change, so it starts its own family again.
    # v6 uses expected starter innings plus a role-filtered bullpen aggregate;
    # it starts a new family while all prior tags remain in ledger history.
    # v7 centre-matches the moments used to estimate xwOBA shrinkage K and
    # starts another prediction family.
    # v8 fixes xwOBA shrinkage K at 100 for both batters and pitchers and
    # starts another prediction family.
    # v9 applies starter platoon adjustments only to projected starter innings
    # and uses the neutral lineup against the bullpen.
    # v10 weights the two phases by share of plate appearances (measured BF/IP)
    # instead of share of innings. Prediction math changed, so the tag moves,
    # but a BF/IP-ratio sweep over 0.95-1.10 flips 0 of 12 leans and shifts
    # xw_net by 1.6-3.4% of its median magnitude: v9 and v10 decisions agree,
    # so they share one win-loss line rather than resetting the sample again.
    "xw+plat_consol_v9": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
    "xw+plat_consol_v10": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
    "woba+plat_consol_v1": ("woba+plat_consol_v1",),
    "woba+plat_consol_v2": ("woba+plat_consol_v2",),
    # v3: K 100->400 plus a relief-pool shrink target. New prediction
    # family; see build_site._RECORD_FAMILIES for the argument.
    "woba+plat_consol_v3": ("woba+plat_consol_v3",),
    # v4: player-specific shrinkage targets. New record family --
    # see _RECORD_FAMILIES in build_site.py, which is the authority.
    "woba+plat_consol_v4": ("woba+plat_consol_v4",),
    # v5: abstain when a side's starter has no measured rate. New record
    # family -- it changes which games are decided; see build_site.
    "woba+plat_consol_v5": ("woba+plat_consol_v5",),
    # Historical one-slate experiment; isolated from the restored full-wOBA
    # family but still recognised by the immutable ledger.
    "split+plat_consol_v1": ("split+plat_consol_v1",),
    # v11: metric back to xwOBA, K back to 100, population shrinkage target --
    # keeping the exposure-centred platoon offsets, the relief-pool target and
    # the starter abstention. New record family; build_site._RECORD_FAMILIES is
    # the authority and carries the per-piece argument.
    "xw+plat_consol_v11": ("xw+plat_consol_v11",),
    # v12: expected_sp_ip calibrated per build against its own actuals. New
    # record family -- it flips 1 lean in 254, which on the v10 precedent would
    # have argued for sharing, but v11 had no graded rows so the reset is free.
    # build_site._RECORD_FAMILIES is the authority and carries the argument.
    "xw+plat_consol_v12": ("xw+plat_consol_v12",),
}
RECORD_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "RECORD_TAGS", ",".join(_RECORD_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
    ).split(",") if t.strip()
)
MODEL_FAMILY_TAGS = (
    ("v2/v3", ("xw+plat_consol_v2", "xw+plat_consol_v3")),
    ("v4", ("xw+plat_consol_v4",)),
    ("v5", ("xw+plat_consol_v5",)),
    ("v6", ("xw+plat_consol_v6",)),
    ("v7", ("xw+plat_consol_v7",)),
    ("v8", ("xw+plat_consol_v8",)),
    ("v9/v10", ("xw+plat_consol_v9", "xw+plat_consol_v10")),
    ("wOBA v1", ("woba+plat_consol_v1",)),
    ("wOBA v2", ("woba+plat_consol_v2",)),
    ("wOBA v3", ("woba+plat_consol_v3",)),
    ("wOBA v4", ("woba+plat_consol_v4",)),
    ("wOBA v5", ("woba+plat_consol_v5",)),
    ("split v1", ("split+plat_consol_v1",)),
    ("v11", ("xw+plat_consol_v11",)),
    ("v12", ("xw+plat_consol_v12",)),
)
# Numerical floor on the weight fit, NOT an evidence threshold. It was 120,
# chosen to suppress a ratio that is unreadable at small n; the ratio is gone
# and coefficients printed with their standard errors are honest at any size --
# `+0.122 +/- 0.227` says "indistinguishable from zero" without needing to be
# hidden. What remains is that a logit on very few rows can fail to converge or
# return a meaningless covariance, so this is now sized for that and nothing
# else. Raising it back to hide an uncertain number would be the claims-the-
# data-cannot-support entry inverted: withholding the uncertainty instead of
# overstating the estimate.
N_FIT_MIN   = 30
_FINAL  = {"Final", "Game Over", "Completed Early"}
_VOID   = {"Postponed", "Cancelled"}

ABBR = {
 "Arizona Diamondbacks":"ARI","Athletics":"ATH","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
 "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS","Cincinnati Reds":"CIN",
 "Cleveland Guardians":"CLE","Colorado Rockies":"COL","Detroit Tigers":"DET","Houston Astros":"HOU",
 "Kansas City Royals":"KC","Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
 "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM","New York Yankees":"NYY",
 "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
 "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB","Texas Rangers":"TEX",
 "Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

def _hj(url, params=None, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=30); r.raise_for_status()
            return r.json()
        except Exception:
            if k == tries - 1: raise
            time.sleep(0.6 * (2 ** k))

def _ab(name): return ABBR.get(name, str(name or "")[:3].upper())
def _fx(v):
    try:
        v = float(v); return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None
def _optbool(v):
    """Tri-state bool: True/False from a dump column, NaN when absent (legacy)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "1.0"):
            return True
        if s in ("false", "0", "0.0"):
            return False
        return np.nan
    return bool(v)

LEDGER_COLS = [
    "game_pk","game_date","away","home","away_sp","home_sp","model_tag",
    "model_metric",
    "B_home","B_away","P_awaySP","P_homeSP","d_lineup","d_sp",
    "home_off_edge","away_off_edge","xw_net","xw_lean","xw_delta",
    "ops_net","ops_lean","ops_delta","ops_valid","consensus",
    "status","full_away","full_home","f5_away","f5_home",
    "xw_full","xw_f5","ops_full","ops_f5",
]
# Audit-only columns. The lineup_* fields record each side's lineup
# resolution (posted / partial_filled / projected + posted and Savant-backfill
# counts) as of the accepted snapshot, so lineup freshness at lock is auditable
# per row. They are NaN on legacy rows dumped before the columns existed —
# intentionally never backfilled — and instrumentation only: no effect on
# grading.
AUDIT_COLS = [
    "snapshot_utc", "scheduled_start_utc", "lock_status",
    "lineup_status_away", "lineup_status_home",
    "lineup_posted_away", "lineup_posted_home",
    "lineup_savant_backfill_away", "lineup_savant_backfill_home",
    # True when that side's probable was classified as an opener, plus the
    # classification evidence and actual pitching input used.
    # NaN on legacy rows; never backfilled.
    "opener_away", "opener_home",
    "opener_reason_away", "opener_reason_home",
    "opener_confidence_away", "opener_confidence_home",
    "pitching_basis_away", "pitching_basis_home",
    # Whether the starter rate beside it was measured or defaulted to the
    # player's prior, and the BF behind it. A starter missing from the
    # leaderboard publishes a prior-shaped number that reads like a
    # measurement; these two make that filterable, so the incidence can be
    # counted off the ledger before anyone argues for abstaining on it.
    # NaN on legacy rows; never backfilled, never read by grading.
    "sp_rate_basis_away", "sp_rate_basis_home",
    "sp_rate_bf_away", "sp_rate_bf_home",
    # v6 workload/blend audit. The P_* fields above hold the actual model
    # pitching input; these preserve its starter and bullpen components.
    "starter_xwoba_away", "starter_xwoba_home",
    "bullpen_xwoba_away", "bullpen_xwoba_home",
    "expected_sp_ip_away", "expected_sp_ip_home",
    "expected_sp_ip_raw_away", "expected_sp_ip_raw_home",
    "bullpen_pitchers_away", "bullpen_pitchers_home",
    "bullpen_relief_bf_away", "bullpen_relief_bf_home",
    # v10 records the measured BF/IP behind the PA-share blend weight, so a
    # side's sp_share can be re-derived from the ledger without a rebuild.
    "sp_bf_per_ip_away", "sp_bf_per_ip_home",
    "bp_bf_per_ip_away", "bp_bf_per_ip_home",
    # v9 sequential-phase audit. Suffixes identify the pitcher side; the
    # opponent-lineup fields therefore describe the offense facing that side.
    "opp_xwoba_neutral_away", "opp_xwoba_neutral_home",
    "opp_xwoba_vs_sp_away", "opp_xwoba_vs_sp_home",
    "platoon_delta_sp_away", "platoon_delta_sp_home",
    "sp_share_away", "sp_share_home",
    "bp_share_away", "bp_share_home",
    "mx_xwoba_sp_away", "mx_xwoba_sp_home",
    "edge_xwoba_sp_away", "edge_xwoba_sp_home",
    "mx_xwoba_bp_away", "mx_xwoba_bp_home",
    "edge_xwoba_bp_away", "edge_xwoba_bp_home",
    "mx_xwoba_away", "mx_xwoba_home",
    "edge_xwoba_away", "edge_xwoba_home",
    # Pitch-mix shadow arm (build_site.USE_PITCH_MIX_SHADOW, default off). The
    # opposing lineup re-weighted by the starter's arsenal, recorded so the arm
    # can be scored against the record before it is allowed to move a lean.
    # NaN on every row built with the flag off, and on legacy rows; never
    # backfilled, and never read by grading.
    "mix_mult_away", "mix_mult_home",
    "mix_coverage_away", "mix_coverage_home",
    "mix_basis_away", "mix_basis_home",
    "opp_xwoba_mix_away", "opp_xwoba_mix_home",
    "mx_xwoba_sp_mix_away", "mx_xwoba_sp_mix_home",
    "edge_xwoba_sp_mix_away", "edge_xwoba_sp_mix_home",
]
MODEL_FIELDS = [
    "game_date","away","home","away_sp","home_sp","model_tag","model_metric",
    "B_home","B_away","P_awaySP","P_homeSP","d_lineup","d_sp",
    "home_off_edge","away_off_edge","xw_net","xw_lean","xw_delta",
    "ops_net","ops_lean","ops_delta","ops_valid","consensus",
    "snapshot_utc","scheduled_start_utc","lock_status",
    "lineup_status_away","lineup_status_home",
    "lineup_posted_away","lineup_posted_home",
    "lineup_savant_backfill_away","lineup_savant_backfill_home",
    "opener_away","opener_home",
    "opener_reason_away","opener_reason_home",
    "opener_confidence_away","opener_confidence_home",
    "pitching_basis_away","pitching_basis_home",
    "sp_rate_basis_away","sp_rate_basis_home",
    "sp_rate_bf_away","sp_rate_bf_home",
    "starter_xwoba_away","starter_xwoba_home",
    "bullpen_xwoba_away","bullpen_xwoba_home",
    "expected_sp_ip_away","expected_sp_ip_home",
    "expected_sp_ip_raw_away","expected_sp_ip_raw_home",
    "bullpen_pitchers_away","bullpen_pitchers_home",
    "bullpen_relief_bf_away","bullpen_relief_bf_home",
    "sp_bf_per_ip_away","sp_bf_per_ip_home",
    "bp_bf_per_ip_away","bp_bf_per_ip_home",
    "opp_xwoba_neutral_away","opp_xwoba_neutral_home",
    "opp_xwoba_vs_sp_away","opp_xwoba_vs_sp_home",
    "platoon_delta_sp_away","platoon_delta_sp_home",
    "sp_share_away","sp_share_home",
    "bp_share_away","bp_share_home",
    "mx_xwoba_sp_away","mx_xwoba_sp_home",
    "edge_xwoba_sp_away","edge_xwoba_sp_home",
    "mx_xwoba_bp_away","mx_xwoba_bp_home",
    "edge_xwoba_bp_away","edge_xwoba_bp_home",
    "mx_xwoba_away","mx_xwoba_home",
    "edge_xwoba_away","edge_xwoba_home",
]

def load_ledger():
    if os.path.exists(LEDGER_PATH):
        led = pd.read_csv(LEDGER_PATH)
        persisted_cols = list(dict.fromkeys(
            LEDGER_COLS + MARKET_COLS + AUDIT_COLS + ACTUAL_COLS
        ))
        # Add every missing column in one concat. Inserting them one at a time
        # refragmented the frame on each new audit column and pandas warns.
        missing = [c for c in persisted_cols if c not in led.columns]
        if missing:
            led = pd.concat(
                [led, pd.DataFrame(np.nan, index=led.index, columns=missing)],
                axis=1)
        # Preserve already attached market columns. Dropping them here forced
        # every CI run to refetch the full closing-odds history.
        led = led[persisted_cols]
        # W/L/T grade columns still all-NaN read back from CSV as float64;
        # pandas >=3 refuses string assignment into float columns, so force
        # object dtype before grading writes W/L/T into them. Same for the
        # lineup status audit columns, which are all-NaN on a ledger that
        # predates them but receive strings on pending refresh, and for
        # consensus, whose "NA" marker read_csv parses as NaN (a ledger with
        # no AGREE/DIVERGE row yet reloads it as float64).
        for c in ("xw_full", "xw_f5", "ops_full", "ops_f5", "consensus",
                  "lineup_status_away", "lineup_status_home",
                  "opener_away", "opener_home",
                  "opener_reason_away", "opener_reason_home",
                  "opener_confidence_away", "opener_confidence_home",
                  "pitching_basis_away", "pitching_basis_home",
                  "sp_rate_basis_away", "sp_rate_basis_home"):
            led[c] = led[c].astype(object)
        return led
    return pd.DataFrame(columns=list(dict.fromkeys(
        LEDGER_COLS + MARKET_COLS + AUDIT_COLS + ACTUAL_COLS
    )))


def _utc_datetime(value):
    """Parse an API/ISO timestamp as an aware UTC datetime, or return None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _lock_status(snapshot_utc, scheduled_start_utc):
    snap, start = _utc_datetime(snapshot_utc), _utc_datetime(scheduled_start_utc)
    if snap is None or start is None:
        return "legacy_unverified"
    return "pregame" if snap < start else "late_snapshot"

# ---- INGEST ------------------------------------------------------------
def rows_from_dump(xw_df, pl_df):
    pl_map = {}
    if pl_df is not None and not pl_df.empty:
        for _, r in pl_df.iterrows():
            pl_map[(int(r["game_pk"]), r["side"])] = r
    out = []
    for gpk, gg in xw_df.groupby("game_pk", sort=False):
        gpk = int(gpk)
        a = gg[gg["side"] == "away"]; h = gg[gg["side"] == "home"]
        if not len(a) or not len(h): continue
        a, h = a.iloc[0], h.iloc[0]
        home_team, away_team = _ab(a["opp_team"]), _ab(h["opp_team"])
        B_home = _fx(a.get("opp_xwOBA_vs_sp"))
        B_away = _fx(h.get("opp_xwOBA_vs_sp"))
        if B_home is None:
            B_home = _fx(a.get("opp_xwOBA"))
        if B_away is None:
            B_away = _fx(h.get("opp_xwOBA"))
        P_aSP = _fx(a.get("starter_xwOBA"))
        P_hSP = _fx(h.get("starter_xwOBA"))
        if P_aSP is None:
            P_aSP = _fx(a.get("pit_xwOBA"))
        if P_hSP is None:
            P_hSP = _fx(h.get("pit_xwOBA"))
        home_off, away_off = _fx(a.get("edge_xwOBA")), _fx(h.get("edge_xwOBA"))
        xw_net = (
            home_off - away_off
            if home_off is not None and away_off is not None
            else None
        )
        d_lu = (B_home - B_away) if None not in (B_home, B_away) else np.nan
        d_sp = (P_aSP - P_hSP)   if None not in (P_aSP, P_hSP)   else np.nan
        pa, ph = pl_map.get((gpk, "away")), pl_map.get((gpk, "home"))
        ops_net = ops_lean = ops_delta = None; ops_valid = False
        if pa is not None and ph is not None:
            eh, ea = _fx(pa.get("edge_OPS")), _fx(ph.get("edge_OPS"))
            if eh is not None and ea is not None:
                ops_net, ops_delta = eh - ea, abs(eh - ea)
                ops_lean = (
                    None if ops_net == 0
                    else home_team if ops_net > 0
                    else away_team
                )
                ops_valid = bool(pa.get("reliable")) and bool(ph.get("reliable"))
        xw_lean = (
            None
            if xw_net is None or xw_net == 0
            else home_team if xw_net > 0
            else away_team
        )
        consensus = (
            "NA"
            if not ops_valid or xw_lean is None or ops_lean is None
            else "AGREE" if ops_lean == xw_lean else "DIVERGE"
        )
        snapshot_utc = a.get("snapshot_utc")
        scheduled_start_utc = a.get("scheduled_start_utc")
        if scheduled_start_utc is None or pd.isna(scheduled_start_utc):
            scheduled_start_utc = a.get("game_datetime_utc")
        lock_status = _lock_status(snapshot_utc, scheduled_start_utc)
        dump_model_tag = a.get("model_tag")
        if dump_model_tag is None or (isinstance(dump_model_tag, float) and pd.isna(dump_model_tag)):
            dump_model_tag = MODEL_TAG
        dump_model_metric = a.get("model_metric")
        if dump_model_metric is None or (isinstance(dump_model_metric, float)
                                         and pd.isna(dump_model_metric)):
            tag = str(dump_model_tag)
            dump_model_metric = (
                "wOBA/xwOBA" if tag.startswith("split+")
                else "wOBA" if tag.startswith("woba+")
                else "xwOBA"
            )
        out.append(dict(
            game_pk=gpk, game_date=str(a.get("game_date")), away=away_team, home=home_team,
            away_sp=a.get("pitcher"), home_sp=h.get("pitcher"), model_tag=str(dump_model_tag),
            model_metric=str(dump_model_metric),
            B_home=B_home, B_away=B_away, P_awaySP=P_aSP, P_homeSP=P_hSP,
            d_lineup=d_lu, d_sp=d_sp,
            home_off_edge=home_off, away_off_edge=away_off,
            xw_net=(float(xw_net) if xw_net is not None else np.nan),
            xw_lean=xw_lean,
            xw_delta=(float(abs(xw_net)) if xw_net is not None else np.nan),
            ops_net=(float(ops_net) if ops_net is not None else np.nan),
            ops_lean=ops_lean,
            ops_delta=(float(ops_delta) if ops_delta is not None else np.nan),
            ops_valid=ops_valid, consensus=consensus,
            snapshot_utc=snapshot_utc, scheduled_start_utc=scheduled_start_utc,
            lock_status=lock_status,
            lineup_status_away=a.get("lineup_status_away", np.nan),
            lineup_status_home=a.get("lineup_status_home", np.nan),
            lineup_posted_away=a.get("lineup_posted_away", np.nan),
            lineup_posted_home=a.get("lineup_posted_home", np.nan),
            lineup_savant_backfill_away=a.get(
                "lineup_savant_backfill_away", np.nan
            ),
            lineup_savant_backfill_home=a.get(
                "lineup_savant_backfill_home", np.nan
            ),
            opener_away=_optbool(a.get("opener")),
            opener_home=_optbool(h.get("opener")),
            opener_reason_away=a.get("opener_reason", np.nan),
            opener_reason_home=h.get("opener_reason", np.nan),
            opener_confidence_away=a.get("opener_confidence", np.nan),
            opener_confidence_home=h.get("opener_confidence", np.nan),
            pitching_basis_away=a.get("pitching_basis", np.nan),
            pitching_basis_home=h.get("pitching_basis", np.nan),
            sp_rate_basis_away=a.get("starter_rate_basis", np.nan),
            sp_rate_basis_home=h.get("starter_rate_basis", np.nan),
            sp_rate_bf_away=a.get("starter_rate_bf", np.nan),
            sp_rate_bf_home=h.get("starter_rate_bf", np.nan),
            starter_xwoba_away=a.get("starter_xwOBA", np.nan),
            starter_xwoba_home=h.get("starter_xwOBA", np.nan),
            bullpen_xwoba_away=a.get("bullpen_xwOBA", np.nan),
            bullpen_xwoba_home=h.get("bullpen_xwOBA", np.nan),
            expected_sp_ip_away=a.get("expected_sp_ip", np.nan),
            expected_sp_ip_home=h.get("expected_sp_ip", np.nan),
            # The uncalibrated estimate, carried so every future calibration
            # fit regresses against the raw number rather than against its own
            # previous output. Absent on pre-v12 rows, where the published
            # value IS the raw one.
            expected_sp_ip_raw_away=a.get("expected_sp_ip_raw", np.nan),
            expected_sp_ip_raw_home=h.get("expected_sp_ip_raw", np.nan),
            bullpen_pitchers_away=a.get("bullpen_pitchers", np.nan),
            bullpen_pitchers_home=h.get("bullpen_pitchers", np.nan),
            bullpen_relief_bf_away=a.get("bullpen_relief_bf", np.nan),
            bullpen_relief_bf_home=h.get("bullpen_relief_bf", np.nan),
            sp_bf_per_ip_away=a.get("sp_bf_per_ip", np.nan),
            sp_bf_per_ip_home=h.get("sp_bf_per_ip", np.nan),
            bp_bf_per_ip_away=a.get("bp_bf_per_ip", np.nan),
            bp_bf_per_ip_home=h.get("bp_bf_per_ip", np.nan),
            opp_xwoba_neutral_away=a.get("opp_xwOBA_neutral", np.nan),
            opp_xwoba_neutral_home=h.get("opp_xwOBA_neutral", np.nan),
            opp_xwoba_vs_sp_away=a.get("opp_xwOBA_vs_sp", np.nan),
            opp_xwoba_vs_sp_home=h.get("opp_xwOBA_vs_sp", np.nan),
            platoon_delta_sp_away=a.get("platoon_delta_sp", np.nan),
            platoon_delta_sp_home=h.get("platoon_delta_sp", np.nan),
            sp_share_away=a.get("sp_share", np.nan),
            sp_share_home=h.get("sp_share", np.nan),
            bp_share_away=a.get("bp_share", np.nan),
            bp_share_home=h.get("bp_share", np.nan),
            mx_xwoba_sp_away=a.get("mx_xwOBA_sp", np.nan),
            mx_xwoba_sp_home=h.get("mx_xwOBA_sp", np.nan),
            edge_xwoba_sp_away=a.get("edge_xwOBA_sp", np.nan),
            edge_xwoba_sp_home=h.get("edge_xwOBA_sp", np.nan),
            mx_xwoba_bp_away=a.get("mx_xwOBA_bp", np.nan),
            mx_xwoba_bp_home=h.get("mx_xwOBA_bp", np.nan),
            edge_xwoba_bp_away=a.get("edge_xwOBA_bp", np.nan),
            edge_xwoba_bp_home=h.get("edge_xwOBA_bp", np.nan),
            mx_xwoba_away=a.get("mx_xwOBA", np.nan),
            mx_xwoba_home=h.get("mx_xwOBA", np.nan),
            edge_xwoba_away=a.get("edge_xwOBA", np.nan),
            edge_xwoba_home=h.get("edge_xwOBA", np.nan),
            mix_mult_away=a.get("mix_mult", np.nan),
            mix_mult_home=h.get("mix_mult", np.nan),
            mix_coverage_away=a.get("mix_coverage", np.nan),
            mix_coverage_home=h.get("mix_coverage", np.nan),
            mix_basis_away=a.get("mix_basis", np.nan),
            mix_basis_home=h.get("mix_basis", np.nan),
            opp_xwoba_mix_away=a.get("opp_xwOBA_mix", np.nan),
            opp_xwoba_mix_home=h.get("opp_xwOBA_mix", np.nan),
            mx_xwoba_sp_mix_away=a.get("mx_xwOBA_sp_mix", np.nan),
            mx_xwoba_sp_mix_home=h.get("mx_xwOBA_sp_mix", np.nan),
            edge_xwoba_sp_mix_away=a.get("edge_xwOBA_sp_mix", np.nan),
            edge_xwoba_sp_mix_home=h.get("edge_xwOBA_sp_mix", np.nan),
            status="pending", full_away=np.nan, full_home=np.nan,
            f5_away=np.nan, f5_home=np.nan,
            xw_full=None, xw_f5=None, ops_full=None, ops_f5=None,
        ))
    return out

def ingest(led):
    n_new = n_ref = n_late = n_legacy = 0
    if "model_metric" in led.columns:
        led["model_metric"] = led["model_metric"].astype(object)
    # Historical xwOBA and split dumps remain immutable. Current wOBA dumps are
    # ingested last so a same-day pending snapshot from either older lineage is
    # refreshed into full wOBA before first pitch; graded rows are never touched.
    dump_paths = (
        sorted(glob.glob(os.path.join(DATA_DIR, "leans_*_xw.csv")))
        + sorted(glob.glob(os.path.join(DATA_DIR, "leans_*_split.csv")))
        + sorted(glob.glob(os.path.join(DATA_DIR, "leans_*_woba.csv")))
    )
    for xw_path in dump_paths:
        pl_path = re.sub(r"_(?:xw|split|woba)\.csv$", "_pl.csv", xw_path)
        xw = pd.read_csv(xw_path)
        pl = pd.read_csv(pl_path) if os.path.exists(pl_path) else None
        for row in rows_from_dump(xw, pl):
            # gamePk is retained when MLB reschedules a postponed game, so the
            # ledger identity must include the slate date. This lets the played
            # make-up entry coexist with the original void entry.
            hit = led.index[
                (pd.to_numeric(led["game_pk"], errors="coerce") == row["game_pk"]) &
                (led["game_date"].astype(str) == row["game_date"])
            ]
            if len(hit) == 0:
                if row["lock_status"] != "pregame":
                    if row["lock_status"] == "late_snapshot": n_late += 1
                    else: n_legacy += 1
                    continue
                add = pd.DataFrame([row])[LEDGER_COLS + AUDIT_COLS]
                led = add if led.empty else pd.concat([led, add], ignore_index=True)
                n_new += 1
            elif led.at[hit[0], "status"] == "pending":
                if row["lock_status"] != "pregame":
                    if row["lock_status"] == "late_snapshot": n_late += 1
                    else: n_legacy += 1
                    continue
                for k in MODEL_FIELDS:                    # refresh scratches pre-lock
                    led.at[hit[0], k] = row[k]
                n_ref += 1
    print(f"ingest: +{n_new} new, {n_ref} pending refreshed, "
          f"{n_late} late snapshots rejected, {n_legacy} legacy refreshes skipped "
          f"({len(led)} total)")
    return led

# ---- GRADE -------------------------------------------------------------
def _linescores_for(day):
    data = _hj("https://statsapi.mlb.com/api/v1/schedule",
               {"sportId": 1, "date": day, "hydrate": "linescore"})
    out = {}
    for db in data.get("dates", []):
        for g in db.get("games", []):
            out[int(g["gamePk"])] = g
    return out

def _f5(innings, side):
    if innings is None or len(innings) < 5: return None
    tot = 0
    for inn in innings[:5]:
        r = (inn.get(side) or {}).get("runs")
        if r is None: return None
        tot += int(r)
    return tot

def _wlt(lean, away, home, ra, rh, allow_tie):
    if lean is None or (isinstance(lean, float) and math.isnan(lean)): return None
    if ra == rh: return "T" if allow_tie else None
    return "W" if lean == (home if rh > ra else away) else "L"

def grade(led):
    pend = led[led["status"] == "pending"]
    if pend.empty:
        print("grade: nothing pending."); return led
    n_g = n_v = 0
    for day in sorted(pend["game_date"].dropna().unique()):
        games = _linescores_for(day)
        for idx in pend[pend["game_date"] == day].index:
            g = games.get(int(led.at[idx, "game_pk"]))
            if g is None: continue
            state = (g.get("status") or {}).get("detailedState", "")
            if state in _VOID:
                led.at[idx, "status"] = "void"; n_v += 1; continue
            if state not in _FINAL:
                continue
            ls = g.get("linescore") or {}
            fa = (ls.get("teams", {}).get("away", {}) or {}).get("runs")
            fh = (ls.get("teams", {}).get("home", {}) or {}).get("runs")
            if fa is None or fh is None: continue
            f5a, f5h = _f5(ls.get("innings"), "away"), _f5(ls.get("innings"), "home")
            aw, hm = led.at[idx, "away"], led.at[idx, "home"]
            led.at[idx, "full_away"], led.at[idx, "full_home"] = fa, fh
            led.at[idx, "f5_away"],   led.at[idx, "f5_home"]   = f5a, f5h
            led.at[idx, "xw_full"] = _wlt(led.at[idx, "xw_lean"], aw, hm, fa, fh, False)
            if f5a is not None:
                led.at[idx, "xw_f5"] = _wlt(led.at[idx, "xw_lean"], aw, hm, f5a, f5h, True)
            if bool(led.at[idx, "ops_valid"]):
                led.at[idx, "ops_full"] = _wlt(led.at[idx, "ops_lean"], aw, hm, fa, fh, False)
                if f5a is not None:
                    led.at[idx, "ops_f5"] = _wlt(led.at[idx, "ops_lean"], aw, hm, f5a, f5h, True)
            led.at[idx, "status"] = "graded"; n_g += 1
    print(f"grade: {n_g} graded, {n_v} void, "
          f"{int((led['status'] == 'pending').sum())} still pending")
    return led

# ---- REPORT ------------------------------------------------------------
def _rec(s):
    s = s.dropna()
    w, l, t = int((s == "W").sum()), int((s == "L").sum()), int((s == "T").sum())
    base = f"{w}-{l}" + (f"-{t}" if t else "")
    return f"{base}  ({w/(w+l):.3f})" if (w + l) else base

def _logit_fit(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b)); Wd = p * (1 - p)
        H = X.T @ (X * Wd[:, None]) + np.eye(X.shape[1]) * 1e-9
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10: break
    p = 1.0 / (1.0 + np.exp(-X @ b)); Wd = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * Wd[:, None]) + np.eye(X.shape[1]) * 1e-9)
    # The full covariance is returned, not just its diagonal, because the
    # question this fit exists to answer is about a CONTRAST of two
    # coefficients and the off-diagonal term is part of its variance.
    # Discarding it here is what forced the old ratio form, which needs no
    # covariance only because it has no usable standard error at all.
    return b, np.sqrt(np.diag(cov)), cov


def symmetry_contrast(b, cov, sd_lineup, sd_sp):
    """(difference, se, z) testing equal first-order weight on the two inputs.

    Replaces `implied w = b_sp / b_lineup`, which was dropped rather than
    reformatted. `b_lineup` is not distinguishable from zero, so that ratio is
    Cauchy-like: bootstrapped over the 82 graded v9/v10 rows its median is
    +0.02, but 48% of resamples flip its sign, 3.6% land beyond |5|, and its
    mean and sd do not converge with resample count. There is no standard error
    to print beside it -- which is why none ever was -- and a bare `+0.12`
    reads as a measurement of a relative weight that the data cannot support.
    On the same ledger it read +0.12 on v9/v10, -2.61 pooled and +4.77 on the
    wOBA rows: three numbers, one underlying non-result.

    The hypothesis is unchanged and now well posed. `w = 1` means equal weight
    in NATIVE units, i.e. `b_sp/sd_sp == b_lineup/sd_lu`, so with coefficients
    fitted on standardised inputs the contrast is

        c'b,  c = (0, 1, -sd_lineup/sd_sp)

    and its variance is `c' cov c`. The off-diagonal covariance term is part of
    that and is why `_logit_fit` returns the full matrix: a difference of two
    correlated coefficients has a usable sampling distribution exactly where
    their ratio does not.

    Returns (nan, nan, None) when the contrast has no positive variance, so the
    caller prints nothing rather than a z built on a zero denominator.
    """
    b = np.asarray(b, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.isfinite(sd_sp) or sd_sp <= 0 or not np.isfinite(sd_lineup):
        return float("nan"), float("nan"), None
    c = np.array([0.0, 1.0, -(sd_lineup / sd_sp)])
    if b.shape[0] != 3 or cov.shape != (3, 3):
        return float("nan"), float("nan"), None
    diff = float(c @ b)
    var = float(c @ cov @ c)
    if not np.isfinite(var) or var <= 0:
        return diff, float("nan"), None
    se = math.sqrt(var)
    return diff, se, diff / se


def _abstained(fam):
    """Graded rows that published no lean.

    v5 abstains when a side's starter has no measured rate, so a game can be
    graded (it was played, the score is in) and still carry no decision. `_rec`
    drops those rows and `len()` counts them, so a line that prints both
    without saying so reads as an unexplained missing game. Count them from
    `xw_lean` -- the field that says whether a decision was published -- not by
    subtracting W and L, which would also swallow ties."""
    return int(fam["xw_lean"].isna().sum()) if "xw_lean" in fam else 0


def _record_grades(led):
    """Graded rows whose tags share the current prediction methodology."""
    return led[(led["status"] == "graded") & (led["model_tag"].isin(RECORD_TAGS))].copy()


def _model_family_grades(led):
    graded = led[led["status"] == "graded"]
    out, covered = [], set()
    for label, tags in MODEL_FAMILY_TAGS:
        fam = graded[graded["model_tag"].isin(tags)]
        covered.update(tags)
        if not fam.empty:
            out.append((label, fam))
    for tag in sorted(set(graded["model_tag"].dropna().astype(str)) - covered):
        fam = graded[graded["model_tag"].astype(str) == tag]
        if not fam.empty:
            out.append((tag, fam))
    return out


def report_text(led):
    """Build the report body. PURE -- no printing, no file write.

    Split out because `report()` writes REPORT_PATH as a side effect, so any
    call made to inspect the output silently overwrote `data/ledger_report.txt`
    -- a bot-owned artifact -- with whatever frame was passed in. That bit
    during development: calling it on a filtered ledger rewrote the committed
    report from a partial row set. Tests and ad-hoc inspection use this; only
    `report()` touches the filesystem.
    """
    lines = []
    say = lines.append
    g = _record_grades(led)
    if g.empty:
        # Name the scope. This said "no graded games yet." full stop, which is
        # true of RECORD_TAGS and reads as "the ledger is empty" -- on the
        # morning v12 shipped, 40 lines above a family history covering 534
        # graded rows. A MODEL_TAG bump empties this block by design and every
        # bump reproduces the sentence, so the fix belongs here rather than in
        # a release note nobody reads twice.
        #
        # The prior-family count is measured off the ledger, never derived by
        # subtracting what this block would have shown -- see _lock_provenance
        # in build_site for the same rule and the reason for it.
        prior = 0
        if "status" in getattr(led, "columns", ()):
            prior = int((led["status"] == "graded").sum())
        tag = " + ".join(RECORD_TAGS)
        if prior:
            say(f"no graded games yet under {tag} — {prior} graded rows of "
                "prior-family history below (see \"model-family history\").")
            say("  The current-family record, |Δ| terciles and weight fit "
                "resume once rows accumulate; everything printed below is "
                "either pooled across families or scoped per family, and is "
                "unaffected.")
        else:
            say("no graded games yet.")
    else:
        _abs = _abstained(g)
        say(f"LEAN LEDGER — {len(g)} graded games"
            + (f" ({len(g) - _abs} with a lean, {_abs} abstained)" if _abs else "")
            + f"  [{' + '.join(RECORD_TAGS)}]")
        say(f"{MODEL_METRIC_LABEL} lean   full: {_rec(g['xw_full'])}   F5: {_rec(g['xw_f5'])}")
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            say(f"platoon lean full: {_rec(ov['ops_full'])}   F5: {_rec(ov['ops_f5'])}   (reliable-only, n={len(ov)})")
            say(f"{MODEL_METRIC_LABEL} on same subset  full: {_rec(ov['xw_full'])}   F5: {_rec(ov['xw_f5'])}")
        if len(g) >= 9:
            g["_terc"] = pd.qcut(g["xw_delta"], 3, labels=["low", "mid", "hi"], duplicates="drop")
            say(f"{MODEL_METRIC_LABEL} F5 by |Δ| tercile:")
            for lab, gg in g.groupby("_terc", observed=True):
                say(f"  {lab:3}  {_rec(gg['xw_f5'])}   (Δ {gg['xw_delta'].min():.3f}–{gg['xw_delta'].max():.3f}, n={len(gg)})")
        dv = g[g["consensus"] == "DIVERGE"]
        if len(dv):
            say(f"DIVERGE h2h (F5): {MODEL_METRIC_LABEL} {int((dv['xw_f5']=='W').sum())} — "
                f"platoon {int((dv['ops_f5']=='W').sum())}  (n={len(dv)})")
        f5d = g.dropna(subset=["f5_away", "f5_home"])
        dec = f5d[f5d["f5_home"] != f5d["f5_away"]]
        if len(dec):
            say(f"home F5 baseline: {(dec['f5_home'] > dec['f5_away']).mean():.3f}  (n={len(dec)})")
        fit = g.dropna(subset=["d_lineup", "d_sp", "f5_away", "f5_home"])
        fit = fit[fit["f5_home"] != fit["f5_away"]]
        say(f"weight fit: {len(fit)} usable F5 decisions (gate {N_FIT_MIN})")
        if len(fit) >= N_FIT_MIN:
            dlu = (fit["d_lineup"] - fit["d_lineup"].mean()) / fit["d_lineup"].std()
            dsp = (fit["d_sp"]     - fit["d_sp"].mean())     / fit["d_sp"].std()
            X = np.column_stack([np.ones(len(fit)), dlu.values, dsp.values])
            y = (fit["f5_home"] > fit["f5_away"]).astype(float).values
            b, se, cov = _logit_fit(X, y)
            say(f"  b_lineup={b[1]:+.3f}±{se[1]:.3f}  b_sp={b[2]:+.3f}±{se[2]:.3f}  "
                f"HFA={b[0]:+.3f}  (per sd)")
            # The symmetry test, as a CONTRAST rather than a ratio.
            #
            # This line used to print `implied w = b_sp/b_lineup`. It was
            # dropped, not reformatted: b_lineup is not distinguishable from
            # zero, so the ratio is Cauchy-like -- bootstrapped over the 82
            # v9/v10 rows its median is +0.02 but 48% of resamples flip sign,
            # 3.6% land beyond |5|, and its mean and sd do not converge. There
            # is no standard error to print beside it, which is why none ever
            # was, and a bare +0.12 reads as a measurement of a relative weight
            # that the data cannot support.
            #
            # The hypothesis is unchanged and is now well posed. `w = 1` means
            # equal weight in NATIVE units, i.e. b_sp/sd_sp == b_lineup/sd_lu,
            # so the contrast is c'b with c = (0, 1, -sd_lu/sd_sp) and its
            # variance is c'(cov)c -- the off-diagonal term included. A
            # difference of two coefficients has a usable sampling
            # distribution exactly where their ratio does not.
            diff, se_diff, z = symmetry_contrast(
                b, cov, fit["d_lineup"].std(), fit["d_sp"].std())
            if z is not None:
                say(f"  symmetry test  b_lineup - b_sp*(sd_lu/sd_sp) = "
                    f"{diff:+.3f} ± {se_diff:.3f}  z={z:+.2f}"
                    f"  ({'no departure from equal weight' if abs(z) < 2 else 'DEPARTURE from equal weight'})")
                say("  (equal first-order weight is what the symmetric "
                    "multiplicative matchup implies; z is the evidence "
                    "against it. Diagnostic only -- nothing here feeds back "
                    "into a lean, a delta or a grade.)")
    # Predicted-vs-actual is scored on RECORD_TAGS only, for the same reason
    # the record is: rates from different prediction families are not
    # commensurable, and pooling them would make a calibration slope describe
    # a model that never existed. The league-rate baseline is recovered from
    # the rows themselves (mx - edge), never a literal.
    _fam = led[led["model_tag"].isin(RECORD_TAGS)]
    _lg = pd.to_numeric(_fam.get("mx_xwoba_away"), errors="coerce") - \
        pd.to_numeric(_fam.get("edge_xwoba_away"), errors="coerce")
    _lg = float(_lg.mean()) if _lg.notna().any() else None
    # Whole ledger, not _fam: actuals_summary scopes the rate metric to
    # RECORD_TAGS itself and deliberately pools the IP metric across families,
    # because expected_pitcher_ip is one estimator shared since v6.
    _act_lines = actuals_summary(led, baseline=_lg, tags=RECORD_TAGS)
    for _ln in _act_lines:
        say(_ln)
    _fam_lines = [
        ln for ln in (actuals_family_line(label, fam)
                      for label, fam in _model_family_grades(led))
        if ln is not None
    ]
    if _fam_lines:
        if not _act_lines:
            say("predicted vs actual (backfilled box scores)")
        say("  by prediction family (never pooled — different inputs, "
            "different scale against an observed-wOBA actual):")
        for _ln in _fam_lines:
            say(_ln)

    # Each component against its own realised phase, so SP, BP and the lineup
    # can be tuned separately instead of only jointly. Until the schema-2
    # backfill has run there is no realised starter line, so only the lineup
    # component pairs and the others are simply absent -- which is the honest
    # state, not a gap to fill with the joint number.
    for _ln in components_summary(led, tags=RECORD_TAGS):
        say(_ln)

    # Whole ledger, NOT RECORD_TAGS, and the exception is the point: every
    # block above scores one prediction family because a level is only
    # comparable within one. This one is scoped by DATE and keeps pooling legal
    # by grouping on the metric instead, which is what lets it survive a bump —
    # scoped to the record family it would have gone blank the morning wOBA v5
    # landed, on exactly the slates a per-slate check exists to watch.
    for _ln in slate_lines(led):
        say(_ln)

    families = _model_family_grades(led)
    if families:
        say("model-family history (never pooled into the current-family fit):")
        for label, fam in families:
            metric = metric_label(fam)
            _abs = _abstained(fam)
            say(
                f"  {label:7} n={len(fam):3}  "
                + (f"({_abs} abstained)  " if _abs else "")
                + f"{metric} full {_rec(fam['xw_full'])}  F5 {_rec(fam['xw_f5'])}"
            )
    return "\n".join(lines)


def report(led):
    """Print the report and write it to REPORT_PATH. The only writer."""
    txt = report_text(led)
    print("=" * 60); print(txt); print("=" * 60)
    with open(REPORT_PATH, "w") as f:
        f.write(txt + "\n")
    return txt

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    led = load_ledger()
    led = ingest(led)
    led = grade(led)
    try:
        led = attach_market(led)      # idempotent; settled rows missing MLs only
    except Exception as e:            # market outage must not lose the grading run
        print(f"market backfill: FAILED ({type(e).__name__}: {e}); rows retry next run")
    try:
        # Must follow attach_market: the join key is the gamePk that call
        # resolves and score-verifies. Same outage discipline -- a StatsAPI
        # box score is an outcome, and losing one must never lose the grades.
        led = attach_actuals(led)
    except Exception as e:            # noqa: BLE001
        print(f"actuals backfill: FAILED ({type(e).__name__}: {e}); rows retry next run")
    led.to_csv(LEDGER_PATH, index=False)
    report(led)

if __name__ == "__main__":
    main()
