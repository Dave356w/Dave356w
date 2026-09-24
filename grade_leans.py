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
# The write site now goes through build_site.dump_path, which diverts a dump
# whose games have ALL started to `rebuild_leans_...` so a post-rollover
# rebuild cannot overwrite the pregame record. Nothing here changes: the
# globs below never matched a `rebuild_`-prefixed name, and such a dump was
# already rejected row-by-row on lock_status. The grader is simply no longer
# offered one.
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

from market_backfill import (MARKET_COLS, ODDS_LADDER, V13_RECON_COLS,
                             V13_RECON_TEXT_COLS, attach_market,
                             breakeven_prob, chalk_is_home, ev_null, excess_se,
                             is_pickem, ladder_rung, metric_label,
                             publish_reconstruction,
                             percentile_price_edges as _percentile_price_edges,
                             percentile_band_index as _percentile_band_index)
import season_phase
from actuals_backfill import (ACTUAL_COLS, attach_actuals, actuals_summary,
                              actuals_family_line, components_summary,
                              target_reliability,
                              slate_lines)

DATA_DIR    = os.environ.get("DATA_DIR", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
# Postseason rows and rows whose game type is not yet confirmed. Written by
# save_ledger, read back only by main(). See season_phase.py for why the split
# happens here, at the single writer, rather than in every reader.
POSTSEASON_LEDGER_PATH = os.path.join(DATA_DIR, season_phase.POSTSEASON_LEDGER_NAME)
REPORT_PATH = os.path.join(DATA_DIR, "ledger_report.txt")
MODEL_TAG   = os.environ.get("MODEL_TAG", "xw+starter_blend_v13")
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
    # v13 starter blend -- ISOLATED. Mirrors build_site._RECORD_FAMILIES;
    # the argument lives there, beside the model that produces the rows.
    # v13 SHARES v12's record line. Mirrors build_site._RECORD_FAMILIES;
    # the argument -- including why a 32-of-448 flip rate does NOT sink the
    # share here, and what the retained rows cost in provenance -- lives
    # there, beside the model that produces the rows.
    "xw+starter_blend_v13": ("xw+plat_consol_v12", "xw+starter_blend_v13"),
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
    ("v13 starter blend", ("xw+starter_blend_v13",)),
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
    "xw_full","xw_f5","ops_full","ops_f5","hybrid_full",
]
# Audit-only columns. The lineup_* fields record each side's lineup
# resolution (posted / partial_filled / projected + posted and Savant-backfill
# counts) as of the accepted snapshot, so lineup freshness at lock is auditable
# per row. They are NaN on legacy rows dumped before the columns existed —
# intentionally never backfilled — and instrumentation only: no effect on
# grading.
AUDIT_COLS = [
    "snapshot_utc", "scheduled_start_utc", "lock_status",
    # XWOBA Market Hybrid decision and the exact current two-sided market used
    # at the last accepted pregame snapshot. Historical v12 rows are NaN and
    # remain available as the explicitly retrospective close-derived archive.
    "selection_rule_tag", "pregame_market_utc",
    "pregame_away_ml", "pregame_home_ml", "pregame_p_home",
    "hybrid_action", "hybrid_selection", "hybrid_p", "hybrid_ml",
    # Explicit because the v2 migration reconstructs legacy rows at their
    # close while live rows retain their saved decision-time market.
    "hybrid_price_source",
    # Frozen v1 saved-pregame decisions retained when current fields migrated
    # to v2. These keep the original registered forward test reproducible.
    "hybrid_v1_action", "hybrid_v1_selection", "hybrid_v1_p",
    "hybrid_v1_ml", "hybrid_v1_full",
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
    "opp_xwoba_sd_away", "opp_xwoba_sd_home",
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
    # StatsAPI gameType (R/F/D/L/W). Decides which ledger FILE a row is saved
    # to and nothing else; never read by grading. Last in AUDIT_COLS, so it
    # persists after the audit block and before ACTUAL_COLS and the carried
    # v13 recon columns: those shift one place right, every relative order is
    # unchanged, and all ledger readers select by name. See season_phase.py.
    "game_type",
]
MODEL_FIELDS = [
    "game_date","away","home","away_sp","home_sp","model_tag","model_metric",
    "B_home","B_away","P_awaySP","P_homeSP","d_lineup","d_sp",
    "home_off_edge","away_off_edge","xw_net","xw_lean","xw_delta",
    "ops_net","ops_lean","ops_delta","ops_valid","consensus",
    "snapshot_utc","scheduled_start_utc","lock_status",
    "selection_rule_tag","pregame_market_utc",
    "pregame_away_ml","pregame_home_ml","pregame_p_home",
    "hybrid_action","hybrid_selection","hybrid_p","hybrid_ml",
    "hybrid_price_source",
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
    "opp_xwoba_sd_away","opp_xwoba_sd_home",
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

def load_ledger(include_held=False):
    """The regular-season ledger; with include_held, the held rows too.

    Only main() passes include_held: it grades, prices and backfills every row
    alike and splits them again in save_ledger. Every other caller -- tests,
    probes, report inspection -- gets regular season only, which is what all
    of them were written against.
    """
    if os.path.exists(LEDGER_PATH):
        led = pd.read_csv(LEDGER_PATH)
        # Legacy stamp: a main ledger that predates the column is wholly
        # regular season. The only place a missing type is read as `R`.
        if season_phase.GAME_TYPE_COL not in led.columns:
            led = pd.concat([led, pd.Series(season_phase.REGULAR, index=led.index,
                                            name=season_phase.GAME_TYPE_COL)],
                            axis=1)
        if include_held and os.path.exists(POSTSEASON_LEDGER_PATH):
            held = pd.read_csv(POSTSEASON_LEDGER_PATH)
            if not held.empty:
                led = pd.concat([led, held], ignore_index=True)
        # Preserved-if-present, never minted. reconstruct_v13 writes these
        # and nothing in a build does, so enumerating them in AUDIT_COLS
        # would mint them empty on every ledger and make the migration --
        # which appends trailing fields and refuses a column that already
        # exists -- unrunnable. Leaving them out of the list entirely is what
        # deleted them: the reindex below keeps only what it is told to keep,
        # so a column this module has never heard of does not survive one
        # bot ledger commit. That is the whole failure, and it is the writer's
        # to fix rather than the migration's to repeat.
        carried = [c for c in V13_RECON_COLS if c in led.columns]
        persisted_cols = list(dict.fromkeys(
            LEDGER_COLS + MARKET_COLS + AUDIT_COLS + ACTUAL_COLS + carried
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
        for c in ("xw_full", "xw_f5", "ops_full", "ops_f5", "hybrid_full", "consensus",
                  "selection_rule_tag", "pregame_market_utc",
                  "hybrid_action", "hybrid_selection", "hybrid_price_source",
                  "hybrid_v1_action", "hybrid_v1_selection", "hybrid_v1_full",
                  "lineup_status_away", "lineup_status_home",
                  "opener_away", "opener_home",
                  "opener_reason_away", "opener_reason_home",
                  "opener_confidence_away", "opener_confidence_home",
                  "pitching_basis_away", "pitching_basis_home",
                  "sp_rate_basis_away", "sp_rate_basis_home",
                  "game_type"):
            led[c] = led[c].astype(object)
        for c in V13_RECON_TEXT_COLS:
            if c in led.columns:
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
            game_type=season_phase.clean_game_type(a.get("game_type")),
            selection_rule_tag=a.get("selection_rule_tag", np.nan),
            pregame_market_utc=a.get("pregame_market_utc", np.nan),
            pregame_away_ml=a.get("pregame_away_ml", np.nan),
            pregame_home_ml=a.get("pregame_home_ml", np.nan),
            pregame_p_home=a.get("pregame_p_home", np.nan),
            hybrid_action=a.get("hybrid_action", np.nan),
            hybrid_selection=a.get("hybrid_selection", np.nan),
            hybrid_p=a.get("hybrid_p", np.nan),
            hybrid_ml=a.get("hybrid_ml", np.nan),
            hybrid_price_source=a.get("hybrid_price_source", np.nan),
            hybrid_v1_action=np.nan,
            hybrid_v1_selection=np.nan,
            hybrid_v1_p=np.nan,
            hybrid_v1_ml=np.nan,
            hybrid_v1_full=np.nan,
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
            # Slot-PA-weighted sd of the nine shrunk rates behind the composite
            # above. Diagnostic: it makes "do good hitters get averaged down"
            # measurable, and nothing reads it back into a lean. Absent on every
            # row written before it existed, which is most of the ledger.
            opp_xwoba_sd_away=a.get("opp_xwOBA_sd", np.nan),
            opp_xwoba_sd_home=h.get("opp_xwOBA_sd", np.nan),
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
            hybrid_full=None,
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
                if ("game_type" in led.columns
                        and pd.isna(season_phase.clean_game_type(led.at[hit[0], "game_type"]))
                        and pd.notna(row["game_type"])):
                    led.at[hit[0], "game_type"] = row["game_type"]
                n_ref += 1
    n_v1 = _mint_v1_archive(led)
    print(f"ingest: +{n_new} new, {n_ref} pending refreshed, "
          f"{n_late} late snapshots rejected, {n_legacy} legacy refreshes skipped "
          f"({len(led)} total); {n_v1} pending v1 archive row(s) written")
    return led


def _mint_v1_archive(led):
    """Write each PENDING current-family row's v1 hybrid decision. Returns n.

    `hybrid_v1_*` is the archived q-gate selection. Its registration is
    retired, but it is still the ROW SELECTOR that `abstain_test` and
    `dog_contrast_test` delegate to, so a row that never receives one drops out
    of two LIVE registrations with open gates -- silently, because it is
    filtered inside `hybrid_test._committed`, one step before the `unscorable`
    counter that exists to make a shrinking forward denominator visible. That
    is how both froze at 2026-09-11 when `migrate_hybrid_v2` stopped minting:
    the guard was pointed at the rows the denominator drops, not at the
    denominator itself. Minting here is what stops the gap reopening; the
    migration only repairs the rows written while it was open.

    PENDING ONLY. A graded row's archive is immutable, for the same reason
    `xw_full` is. Refreshing a pending one is required rather than optional:
    `MODEL_FIELDS` rebuilds `xw_lean` and `pregame_p_home` on every pregame
    poll, and the 2026-09-11 slate showed 2 rows whose LEAN flipped between the
    first snapshot and the lock, so an archive minted once at insert would name
    a selection nobody locked.
    """
    import hybrid_test                    # local, matching this module's others
    cols = ("xw_lean", "home", "away", "pregame_p_home",
            "pregame_home_ml", "pregame_away_ml", "status", "model_tag")
    if any(c not in led.columns for c in cols):
        return 0
    n = 0
    # `load_ledger` already casts these, but a caller holding a frame built
    # some other way hands us all-NaN float columns, and writing a string into
    # one warns on pandas 2 and RAISES on 3.
    for c in ("hybrid_v1_action", "hybrid_v1_selection"):
        led[c] = led[c].astype(object)
    pending = led.index[led["status"].eq("pending")
                        & led["model_tag"].astype(str).eq(MODEL_TAG)]
    for idx in pending:
        v1 = hybrid_test.locked_v1_decision(
            led.at[idx, "xw_lean"], led.at[idx, "home"], led.at[idx, "away"],
            led.at[idx, "pregame_p_home"], led.at[idx, "pregame_home_ml"],
            led.at[idx, "pregame_away_ml"])
        if v1 is None:
            continue
        (led.at[idx, "hybrid_v1_action"], led.at[idx, "hybrid_v1_selection"],
         led.at[idx, "hybrid_v1_p"], led.at[idx, "hybrid_v1_ml"]) = v1
        n += 1
    return n

# ---- GRADE -------------------------------------------------------------
def _linescores_for(day):
    data = _hj("https://statsapi.mlb.com/api/v1/schedule",
               {"sportId": 1, "date": day, "hydrate": "linescore"})
    out = {}
    for db in data.get("dates", []):
        for g in db.get("games", []):
            out[int(g["gamePk"])] = g
    return out

def resolve_game_types(led):
    """Fill missing game_type from the schedule, one call per affected date.

    Dumps written before the build stamped the type, and rows whose stamp
    failed, arrive blank. A blank row is saved to the held file, never the
    main ledger, so a failed lookup here delays a regular-season row by one
    run and nothing else. Idempotent; a confirmed type is never overwritten.
    """
    col = season_phase.GAME_TYPE_COL
    blank = led[col].map(season_phase.clean_game_type).isna()
    n_set = n_fail = 0
    for day in sorted(led.loc[blank, "game_date"].dropna().astype(str).unique()):
        on_day = blank & (led["game_date"].astype(str) == day)
        try:
            games = _linescores_for(day)
        except Exception as e:                    # noqa: BLE001
            print(f"game type: schedule lookup for {day} failed ({type(e).__name__}); "
                  "rows stay held until a later run resolves them")
            n_fail += int(on_day.sum())
            continue
        for idx in led.index[on_day]:
            pk = pd.to_numeric(led.at[idx, "game_pk"], errors="coerce")
            g = games.get(int(pk)) if pd.notna(pk) else None
            gt = season_phase.clean_game_type((g or {}).get("gameType"))
            if pd.isna(gt):
                n_fail += 1
                continue
            led.at[idx, col] = gt
            n_set += 1
    if n_set or n_fail:
        print(f"game type: {n_set} resolved, {n_fail} still unconfirmed (held)")
    return led


def save_ledger(led):
    """Write regular-season rows to LEDGER_PATH and the rest to the held file.

    The held file is rewritten whenever it exists, even empty, so a row that
    resolves to `R` leaves it on the same run it joins the main ledger.
    """
    regular, held = season_phase.split_regular(led)
    regular.to_csv(LEDGER_PATH, index=False)
    if len(held) or os.path.exists(POSTSEASON_LEDGER_PATH):
        held.to_csv(POSTSEASON_LEDGER_PATH, index=False)
    return regular, held


def _held_lines(held):
    """Short, descriptive footer for rows the report above excludes."""
    if held is None or held.empty:
        return []
    types = held["game_type"].map(season_phase.clean_game_type)
    by_type = types.fillna("unconfirmed").value_counts().sort_index()
    status = held["status"].astype(str).value_counts()
    out = [
        "",
        f"HELD OUT of every number above: {len(held)} rows in "
        f"{season_phase.POSTSEASON_LEDGER_NAME}",
        "  by type: " + "  ".join(f"{k}={v}" for k, v in by_type.items())
        + "   (F wild card, D division series, L LCS, W World Series)",
        "  status: " + "  ".join(f"{k}={status.get(k, 0)}"
                                 for k in ("graded", "pending", "void")),
    ]
    post = held[types.notna()]
    wl = post["xw_full"].astype(str)
    w, l = int((wl == "W").sum()), int((wl == "L").sum())
    if w + l:
        out.append(f"  postseason xwOBA lean full: {w}-{l}  (descriptive only; "
                   "no registration covers these games)")
    if types.isna().any():
        out.append(f"  {int(types.isna().sum())} row(s) await a game-type lookup; "
                   "they join the main ledger once confirmed regular season")
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
    """W/L/T for a selection, or None when there is nothing to grade.

    A selection naming NEITHER club is an abstention, not a loss. The naive
    `"W" if lean == winner else "L"` graded such a row `L` whichever side won,
    because the mismatch fell through the else -- silently, and into a ledger
    column that is immutable once written. That fired for real: build_site
    persisted `hybrid_selection` in the model's StatsAPI namespace (`AZ`) while
    this module writes `home`/`away` as `ARI`, so every Arizona selection would
    have graded a loss regardless of the score. The namespace is fixed at the
    write site; this returns None so the next such leak is a visible gap rather
    than a fabricated result.
    """
    if lean is None or (isinstance(lean, float) and math.isnan(lean)): return None
    if lean != away and lean != home: return None
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
            if ("game_type" in led.columns
                    and pd.isna(season_phase.clean_game_type(led.at[idx, "game_type"]))):
                led.at[idx, "game_type"] = season_phase.clean_game_type(
                    g.get("gameType"))
            led.at[idx, "full_away"], led.at[idx, "full_home"] = fa, fh
            led.at[idx, "f5_away"],   led.at[idx, "f5_home"]   = f5a, f5h
            led.at[idx, "xw_full"] = _wlt(led.at[idx, "xw_lean"], aw, hm, fa, fh, False)
            led.at[idx, "hybrid_full"] = _wlt(
                led.at[idx, "hybrid_selection"], aw, hm, fa, fh, False)
            # The v1 archive is graded here too, from its own stored selection.
            # migrate_hybrid_v2 writes hybrid_v1_full once, gated on the row
            # already being graded, and nothing else ever wrote it -- so every
            # row still pending when that migration ran lost its v1 grade
            # permanently while keeping a perfectly good v1 selection. That hit
            # the whole 2026-09-11 slate: 13 rows dropped out of hybrid_test,
            # and with it out of abstain_test and dog_contrast_test, which
            # delegate row selection to it. `_wlt` returns None for a NaN
            # selection, so rows carrying no archive stay ungraded rather than
            # graded a loss.
            led.at[idx, "hybrid_v1_full"] = _wlt(
                led.at[idx, "hybrid_v1_selection"], aw, hm, fa, fh, False)
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

# Fixed reporting bands in the current family's primary-rate units. These are
# descriptive round cutoffs, introduced after inspecting historical results;
# they are not fitted thresholds, a forward registration, or selection rules.
FIXED_MAGNITUDE_EDGES = (0.0, 0.010, 0.020, 0.030, 0.050, float("inf"))


def _band_label(lo, hi):
    """One home for a band's printed name.

    Three sites format this -- the band block, the grid's cell index and the
    grid's renderer -- and the last two MATCH on the string, so two copies
    drifting would not raise, it would silently print a band with no cells.
    """
    return f"[{lo:.3f}, {hi:.3f})" if np.isfinite(hi) else f"[{lo:.3f}, inf)"


def _magnitude_record(grades):
    """Record and Wilson 95% interval; ties do not enter the rate."""
    w = int(grades.eq("W").sum())
    l = int(grades.eq("L").sum())
    n = w + l
    if not n:
        return f"{_rec(grades)}; decisions=0; 95% CI unavailable"
    p = w / n
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (f"{_rec(grades)}; decisions={n}; "
            f"95% CI [{center - half:.3f}, {center + half:.3f}]")


def _fixed_magnitude_lines(g):
    """Pure descriptive summary; caller supplies current-family graded rows."""
    required = {"xw_net", "xw_lean", "xw_full", "xw_f5"}
    if not required.issubset(g.columns):
        return ["fixed |delta| bands: unavailable (missing report inputs)"]
    magnitude = pd.to_numeric(g["xw_net"], errors="coerce").abs()
    has_lean = g["xw_lean"].notna() & g["xw_lean"].astype(str).str.strip().ne("")
    eligible = has_lean & g["xw_full"].isin(["W", "L"]) & np.isfinite(magnitude)
    out = [
        f"{MODEL_METRIC_LABEL} fixed |delta| bands (current record family; descriptive)",
        "  Fixed cutoffs: .010 / .020 / .030 / .050; lower inclusive, upper exclusive.",
        "  Introduced after inspecting history; not a registered test or calibrated confidence scale.",
        "  Rates exclude ties; intervals are Wilson 95% CIs assuming independent games.",
        f"  Included {int(eligible.sum())} of {len(g)} graded rows; "
        f"excluded {int((~eligible).sum())} without a lean, full decision, or finite xw_net.",
        "  Favorite comparison: closing close_p_home; no pregame fallback. "
        "Both records use the same valid-price, non-tied full-score rows.",
    ]
    # Same convention, same reason, same one home as the always-chalk control
    # above -- and the same disclosure, because a band's favourite record is
    # what the model's record in that band is read against.
    n_pk = int(is_pickem(pd.to_numeric(
        g.loc[eligible, "close_p_home"], errors="coerce")).sum()) \
        if "close_p_home" in g.columns else 0
    if n_pk:
        out.append(f"  {n_pk} of those rows closed at exactly .500 and have no "
                   "favourite; the comparison breaks the tie toward home.")
    price_columns = {"close_p_home", "full_home", "full_away"}
    for lower, upper in zip(FIXED_MAGNITUDE_EDGES, FIXED_MAGNITUDE_EDGES[1:]):
        band = g[eligible & magnitude.ge(lower) & magnitude.lt(upper)]
        label = _band_label(lower, upper)
        out.append(f"  {label} n={len(band)}")
        out.append(f"    full {_magnitude_record(band['xw_full'])}")
        out.append(f"    F5   {_magnitude_record(band['xw_f5'])}")
        if not price_columns.issubset(band.columns):
            out.append("    closing-price comparison unavailable (missing columns)")
            continue
        p = pd.to_numeric(band["close_p_home"], errors="coerce")
        home = pd.to_numeric(band["full_home"], errors="coerce")
        away = pd.to_numeric(band["full_away"], errors="coerce")
        valid = (np.isfinite(p) & p.gt(0) & p.lt(1)
                 & np.isfinite(home) & np.isfinite(away) & home.ne(away))
        paired = band[valid]
        favorite = pd.Series(
            np.where(chalk_is_home(p[valid]) == home[valid].gt(away[valid]),
                     "W", "L"),
            index=paired.index,
        )
        out.append(f"    closing-price paired n={len(paired)}; excluded={len(band) - len(paired)}")
        out.append(f"      {MODEL_METRIC_LABEL} {_magnitude_record(paired['xw_full'])}")
        out.append(f"      favorite {_magnitude_record(favorite)}")
    return out


# ---- magnitude x market-price grid ----------------------------------------
# The market axis of the grid below is the LEANED side's own no-vig
# probability. .500 is the definition of a favourite; the outer two edges are
# the shipped rule's fade gate and its mirror, taken from the registration
# rather than restated, so a reader comparing this grid against the rule is
# never comparing two thresholds that have drifted apart. Neither edge was
# fitted here and neither selects anything -- they cut a display.
#
# The mirror is rounded because `1 - 0.45` is 0.55000000000000004 in binary
# floating point. That one-ULP asymmetry is real and inert inside the
# registered rule, where it is pinned rather than fixed (see
# tests/test_dog_contrast_test.py); it has no business deciding which column a
# displayed row lands in.
def _fixed_price_edges():
    import hybrid_v2
    thr = float(hybrid_v2.THRESHOLD)
    return (0.0, thr, 0.50, round(1.0 - thr, 10), 1.0)


def _p3(x):
    """.417 rather than 0.417 -- every number in this grid is a probability."""
    return f"{x:.3f}".lstrip("0")


def _price_column_labels(edges):
    out = []
    for lo, hi in zip(edges, edges[1:]):
        if lo <= 0.0:
            out.append(f"q <{_p3(hi)}")
        elif hi >= 1.0:
            out.append(f"q {_p3(lo)}+")
        else:
            out.append(f"q {_p3(lo)}-{_p3(hi)}")
    return out


def _grid_cell_line(label, won, q, width=14, breakeven=None):
    """One cell: size, record, its own mean price, the gap, the SE, and EV.

    The SE is never suppressed and the cell is never gated on n. A one-game
    cell prints sqrt(p(1-p)) -- up to 50 points, at a coin-flip price -- which
    is exactly what it should say; gating it would leave a reader to recompute
    the rate without the caveat, and grading it THIN/DEVELOPING/LARGER would be
    the credibility-tier cliff this repo has already removed twice.

    TWO price-relative numbers, and they answer different questions. `excess`
    is the realised rate against the DEVIGGED price: a calibration statistic,
    and the one every other block here prints. `EV` is the same rate against
    the POSTED price's breakeven, which is what a bet actually has to clear.
    The two differ by the cell's own hold, so a cell can beat its devigged
    price and still lose money. Deliberately no cell is named here: which ones
    flip moves with the rows, and a worked example frozen into a comment is the
    constants-from-data entry in prose. Compare the two columns on the block
    itself. Printing only the first is the defect this argument names; printing
    only the second would drop the calibration read the rest of the report is
    built on.

    Note which prices these are. This block reads the PREGAME snapshot, whose
    hold runs materially wider than the closes the rest of the report scores --
    books tighten toward first pitch -- so the gap between the two columns here
    is larger than a closing-basis version of the same cell would show. That is
    a property of the snapshot, not of the model.

    One `+-` serves both, and that is arithmetic rather than economy: the
    breakeven is fixed by the market exactly as `q` is, never estimated from
    the outcomes under test, so the two statistics differ by a constant and
    share a sampling SE. What they do NOT share is a null -- `excess` is
    centred on zero when the market is right, `EV` on minus the hold.

    `breakeven` is optional and is dropped for the whole cell unless every row
    in it carries one. A partial column would quietly change the denominator
    between the two numbers on one line, which is the harder error to see.
    """
    n = int(np.asarray(q).size)
    if not n:
        return f"    {label:<{width}} n=  0   (no rows)"
    w = int(np.asarray(won).sum())
    rate = w / n
    mkt = float(np.asarray(q).mean())
    line = (f"    {label:<{width}} n={n:>3}   {f'{w}-{n - w}':>7} ({rate:.3f})   "
            f"mean q {_p3(mkt)}   excess {100 * (rate - mkt):+6.1f}"
            f" +- {100 * excess_se(q):4.1f} pp")
    if breakeven is not None:
        be = np.asarray(breakeven, dtype=float)
        if be.size == n and np.isfinite(be).all():
            # The null is PRINTED, not left to the block's prose. Stating the
            # rule and withholding the number is what let a reader compare an
            # EV figure to zero; see market_backfill.ev_null for the incident.
            line += (f"   EV {100 * (rate - float(be.mean())):+6.1f} pp"
                     f" (null {100 * ev_null(q, be):+5.1f})")
    return line


def _grid_null_best_excess(cells, draws=2000, seed=0):
    """Mean best-CELL excess when every game settles at its own price.

    A grid is a search. The largest of twenty cells is a maximum, and at these
    cell sizes a maximum is large whether or not anything is there -- the same
    arithmetic that hands back a +20% ROI cell from pure noise on this ledger.
    So the grid prints what its own best cell would read under "the market is
    right and the model adds nothing", which is the reference a cell has to
    beat. Never zero.

    Fixed seed: this report is a committed artifact, and a reference line that
    moves every build is a diff nobody can read.
    """
    cells = [np.asarray(c, dtype=float) for c in cells]
    cells = [c for c in cells if c.size]
    if not cells:
        return float("nan")
    rng = np.random.default_rng(seed)
    best = None
    for q in cells:
        exc = (rng.random((draws, q.size)) < q).mean(axis=1) - q.mean()
        best = exc if best is None else np.maximum(best, exc)
    return float(best.mean())


def _magnitude_price_cells(mag, q, edges):
    """(band label, column label, boolean mask) for every interior cell."""
    out = []
    for lo, hi in zip(FIXED_MAGNITUDE_EDGES, FIXED_MAGNITUDE_EDGES[1:]):
        band = _band_label(lo, hi)
        row = (mag >= lo) & (mag < hi)
        for name, (clo, chi) in zip(_price_column_labels(edges),
                                    zip(edges, edges[1:])):
            out.append((band, name, row & (q >= clo) & (q < chi)))
    return out


def _magnitude_price_grid_lines(g):
    """|xw_net| bands x the leaned side's saved pregame price. Descriptive.

    There is no validated mapping from magnitude to a win probability in this
    repo: `xw_net` is an xwOBA difference, and the only thing the ledger can
    say about it is what past leans of that size did against the prices those
    games actually carried. So a cell reports its size, its record, the market
    probability of the side the model took, the gap between the two, and the
    SE of that gap -- and none of that is a probability for the next game to
    land in the cell.

    The market axis is what makes the magnitude axis readable at all, because
    the two are not independent: the model leans the market's favourite more
    often as |xw_net| grows, so a band's own rate moves with the schedule that
    band drew, and it cannot separate "the model is right" from "the model
    leaned expensive favourites". The favourite control in the block above is
    one binary comparison where the games' own prices are n of them, and it
    can read as a tie while the price-relative gap is not one. The measurement
    behind that is in CLAUDE.md; the gap against each game's own price is the
    only column here that is a statement about the model rather than about the
    schedule.

    Saved pregame prices only (`pregame_p_home`), no close fallback: that is
    the price the decision was locked against, and the two snapshots are not
    interchangeable. Rows predating the pregame instrumentation carry no such
    price and are counted out rather than filled from their close -- which is
    why this block's n is smaller than the one above, and why it names its own
    earliest date instead of leaving the reader to assume full coverage.
    """
    label = f"{MODEL_METRIC_LABEL} |delta| x saved-pregame market probability"
    required = {"xw_net", "xw_lean", "xw_full", "home", "away", "pregame_p_home"}
    if not required.issubset(g.columns):
        return [f"{label}: unavailable (missing report inputs)"]
    mag = pd.to_numeric(g["xw_net"], errors="coerce").abs()
    p_home = pd.to_numeric(g["pregame_p_home"], errors="coerce")
    lean = g["xw_lean"].astype(str).str.strip()
    # An unrecognised lean is not a side. `_wlt` learned that the hard way: its
    # else-branch graded a selection matching neither club a LOSS, so a
    # namespace mismatch invented a result rather than raising one.
    sided = lean.eq(g["home"].astype(str).str.strip()) | lean.eq(
        g["away"].astype(str).str.strip())
    decided = g["xw_full"].isin(["W", "L"])
    priced = p_home.gt(0) & p_home.lt(1)
    finite = np.isfinite(mag)
    keep = sided & decided & priced & finite
    out = [
        f"{label} (current record family; descriptive)",
        "  Rows: the fixed |delta| bands above. Columns: q, the market's own probability "
        "of the side the model leaned,",
        "    taken from the pregame snapshot the decision was locked against "
        "(pregame_p_home; NO close fallback).",
        "  Cell: n; record; mean q; excess = realised rate - mean q, in points; "
        "+- Poisson-binomial SE at the cell's own prices;",
        "    EV = the same rate against the POSTED price's breakeven, which is what "
        "a bet has to clear. The two differ by the",
        "    cell's own hold, so a cell can beat its devigged q and still lose money. "
        "One SE serves both: the breakeven is fixed",
        "    by the market exactly as q is. Their NULLS differ though, and the EV "
        "column PRINTS its own: excess is centred on",
        "    zero when the market is right, EV on minus the cell's hold, which is "
        "the (null ...) beside it. Read EV against that,",
        "    NEVER against zero -- doing so credits the model with one hold it never "
        "earned. Because the two columns differ by a",
        "    constant the market fixes, excess / +- is already the z for BOTH, so no "
        "second interval is printed or needed.",
        "  Magnitude is an xwOBA difference, not a win probability, and this repo has "
        "no validated mapping between the two.",
        "    A cell's rate is what past leans in it did; it is not this model's "
        "probability for the next game in that cell.",
        "  The q column is what makes the rate readable: a band can win 77% because the "
        "model is right or because it leaned",
        "    expensive favourites, and only the gap against each game's own price "
        "separates those. Full-game only; ties excluded.",
    ]
    excluded = [(int((~sided).sum()), "no lean, or a lean naming neither club"),
                (int((~decided).sum()), "no W/L full-game decision"),
                (int((~priced).sum()), "no usable saved pregame price"),
                (int((~finite).sum()), "no finite xw_net")]
    kept = g[keep]
    first = (str(kept["game_date"].astype(str).min())
             if len(kept) and "game_date" in kept.columns else "")
    out.append(f"  Included {len(kept)} of {len(g)} graded rows"
               + (f"; earliest saved pregame price {first}." if first else "."))
    out.append("    Excluded, each counted from its own column rather than by "
               "subtraction, so a row failing two conditions appears twice:")
    for n_ex, why in excluded:
        out.append(f"      {n_ex:>4}  {why}")
    if not len(kept):
        out.append("    no pregame-priced rows yet -- no cells to show.")
        return out
    mag_k = mag[keep].to_numpy(float)
    lean_home_k = lean[keep].eq(kept["home"].astype(str).str.strip()).to_numpy(bool)
    ph_k = p_home[keep].to_numpy(float)
    q = np.where(lean_home_k, ph_k, 1.0 - ph_k)
    # The leaned side's own posted price, for the EV column. Deliberately the
    # PREGAME moneylines rather than the closes: this block is built on the
    # pregame snapshot the decision was locked against, and pairing a pregame
    # probability with a closing payout would be the mixed basis the rest of
    # the report refuses. Absent columns give an all-NaN array, which
    # `_grid_cell_line` drops per cell rather than filling.
    if {"pregame_home_ml", "pregame_away_ml"}.issubset(kept.columns):
        ml_k = np.where(lean_home_k,
                        pd.to_numeric(kept["pregame_home_ml"], errors="coerce"),
                        pd.to_numeric(kept["pregame_away_ml"], errors="coerce"))
        be_k = breakeven_prob(ml_k)
    else:
        be_k = np.full(q.shape, np.nan)
    won = kept["xw_full"].eq("W").to_numpy(bool)
    edges = _fixed_price_edges()
    cells = _magnitude_price_cells(mag_k, q, edges)
    cols = _price_column_labels(edges)
    for lo, hi in zip(FIXED_MAGNITUDE_EDGES, FIXED_MAGNITUDE_EDGES[1:]):
        band = _band_label(lo, hi)
        row = (mag_k >= lo) & (mag_k < hi)
        out.append(f"  |delta| {band}  n={int(row.sum())}")
        for band_label, name, mask in cells:
            if band_label == band:
                out.append(_grid_cell_line(name, won[mask], q[mask],
                                           breakeven=be_k[mask]))
        out.append(_grid_cell_line("all q", won[row], q[row],
                                   breakeven=be_k[row]))
        # A band confined to one column has a margin that CANNOT differ from
        # that column -- two identical lines with nothing saying why, which is
        # how a reader gets duplicated data or a suspected bug instead of a
        # fact. The fact is worth having (it is what the .050+ band looks
        # like), so the equality is named rather than the margin dropped.
        if sum(1 for bl, _, m in cells if bl == band and m.any()) == 1:
            out.append("      (this band occupies one column, so the margin "
                       "above can only repeat that cell)")
    out.append("  all |delta| bands")
    for name, (clo, chi) in zip(cols, zip(edges, edges[1:])):
        col = (q >= clo) & (q < chi)
        out.append(_grid_cell_line(name, won[col], q[col],
                                   breakeven=be_k[col]))
    out.append(_grid_cell_line("all q", won, q, breakeven=be_k))
    filled = [(b, c, m) for b, c, m in cells if m.any()]
    # With one non-empty cell there is no maximum to correct for: the null best
    # of a single cell is its own null mean, which is zero by construction, so
    # the reference line would reduce to "compare against zero" -- exactly what
    # its own last clause forbids. Say that instead of printing a +0.0 that
    # flatters whatever the one cell did.
    if len(filled) == 1:
        out.append("  best-cell reference: one non-empty cell, so nothing was "
                   "maximised over and there is no search to correct for. Read "
                   "it against its own SE above.")
    elif filled:
        null_best = _grid_null_best_excess([q[m] for _, _, m in filled])
        band, name, mask = max(
            filled, key=lambda t: won[t[2]].mean() - q[t[2]].mean())
        observed = won[mask].mean() - q[mask].mean()
        out.append(f"  best-cell reference: the best of these {len(filled)} non-empty "
                   f"cells averages {100 * null_best:+.1f} pp of excess under "
                   "'every game settles at its own price';")
        out.append(f"    the observed best is {100 * observed:+.1f} pp, |delta| {band} "
                   f"{name} (n={int(mask.sum())}) -- "
                   f"{_search_verdict(100 * observed, 100 * null_best)}. A grid is a "
                   "search, so a cell is read against that reference, never "
                   "against zero.")
    return out


def _prints_the_same(a, b, fmt):
    """Do two figures render identically at the precision they are printed at?

    The tie test for every derived verdict in this report, and deliberately
    not a tolerance. A tolerance is a second number to justify and it can
    disagree with what the line actually shows; this cannot, because it asks
    the rendering itself. The pooling licence is the case that earned it:
    2.038953 against 2.039334 is a real ordering and an invisible one, so a
    verdict that reported it as a clean pass was describing a comparison the
    reader could not make.
    """
    try:
        return format(float(a), fmt) == format(float(b), fmt)
    except (TypeError, ValueError):
        return False


def _search_verdict(observed, reference, fmt="+.1f"):
    """Derived reading of a searched maximum against its own null maximum.

    Takes the figures AS PRINTED -- already scaled to the units on the line --
    with their format, so the tie branch can ask whether the two render the
    same rather than carry a tolerance of its own.

    The three grids in this report all print `observed` beside `reference`
    and, until 2026-09-17, two of them stopped there -- leaving the reader to
    do the comparison the line itself says is the only valid one. Both were
    ABOVE their reference at the time and neither said so. The market-band
    block already derived its verdict; this is that clause with one home, so a
    fourth grid cannot arrive without one.
    """
    if not np.isfinite(observed) or not np.isfinite(reference):
        return "no reference"
    gap = observed - reference
    if _prints_the_same(observed, reference, fmt):
        # The bar is the MEAN of simulated maxima, not a threshold, so
        # clearing it by less than the line's own precision is not clearing
        # it -- and saying otherwise reports an ordering nobody can see.
        return "AT that reference, not clear of it"
    if gap > 0:
        # Said in full every time rather than left to the reader: the
        # reference is the MEAN of the null maximum, so a searched best clears
        # it roughly half the time with nothing there at all. "ABOVE it" alone
        # would read as the finding this line exists to prevent.
        return ("ABOVE it, which a search returns about half the time under "
                "no effect, so it is not a finding either")
    return "at or below it"


def _market_percentile_band_lines(led, bands=8):
    """Market calibration over EQUAL-COUNT price bands. Whole ledger.

    Why it exists. The fixed `ODDS_LADDER` is the axis every market surface
    here shares, and it is deliberately a-priori: round-number rungs chosen for
    no data reason, so a rung means the same thing on the card, on the
    calibration page and in this file. What it is not is balanced. On the rows
    it scores its coverage runs 11 to 510 sides, so its SE runs ~1.5 to ~13 pp
    and the extreme rungs are unreadable while the middle ones are precise.
    This block asks the same question of the same rows on a partition that
    equalises n instead, and prints both so the ladder can be read against it.

    It REPLACES nothing. The ladder stays the shared axis precisely because its
    labels do not move, and these bands are the complement: balanced, and for
    that reason not comparable across builds or across surfaces.

    WHOLE LEDGER, not RECORD_TAGS, and that is the point rather than an
    oversight. A realised rate against a devigged close is arithmetic on a box
    score and a price; it does not know which model wrote the row, so scoping
    it to the current family would halve the sample for nothing. That is the
    row-set rule CLAUDE.md records from the phase-gap read, where the narrower
    row set was the one that flattered the shipped version. The licence for
    pooling is measured and printed rather than assumed.

    WHAT IT DOES NOT PRINT is a pooled both-sides total. The two devigged sides
    of a game sum to 1 and exactly one of them wins, so that number is forced
    to 50.0 vs 50.0 whatever the market does -- an identity published with the
    tightest error bar on the page, which is a defect this repo has already had
    once. The per-band figures are not degenerate because a band holds only
    some of each game's sides.

    THE SE IS `market_backfill.excess_se`, the one home, which assumes the
    observations are independent. Within a band they are not quite: where both
    sides of one game land in the same band their outcomes are complementary,
    which makes the true variance SMALLER than the printed one. So the bar is
    conservative, never flattering, and each band prints how many such pairs it
    holds so the reader can see where that bites.
    """
    if led is None or not len(led):
        return []
    need = {"status", "close_p_home", "close_home_ml", "close_away_ml",
            "full_home", "full_away", "game_pk"}
    if not need.issubset(led.columns):
        return ["market percentile bands: unavailable (missing report inputs)"]
    g = led[led["status"] == "graded"]
    fa = pd.to_numeric(g.get("full_away"), errors="coerce")
    fh = pd.to_numeric(g.get("full_home"), errors="coerce")
    ph = pd.to_numeric(g.get("close_p_home"), errors="coerce")
    hml = pd.to_numeric(g.get("close_home_ml"), errors="coerce")
    aml = pd.to_numeric(g.get("close_away_ml"), errors="coerce")
    ok = (fa.notna() & fh.notna() & (fa != fh) & ph.notna()
          & ph.gt(0) & ph.lt(1) & hml.notna() & aml.notna())
    if int(ok.sum()) < bands * 4:
        return [f"market percentile bands: {int(ok.sum())} usable games, "
                f"fewer than the {bands * 4} this partition needs."]
    gk = g.loc[ok, "game_pk"].to_numpy()
    home_won = (fh[ok] > fa[ok]).to_numpy()
    ml = np.concatenate([hml[ok].to_numpy(), aml[ok].to_numpy()])
    p = np.concatenate([ph[ok].to_numpy(), 1.0 - ph[ok].to_numpy()])
    won = np.concatenate([home_won, ~home_won])
    game = np.concatenate([gk, gk])
    fam = np.concatenate([g.loc[ok, "model_tag"].isin(RECORD_TAGS).to_numpy()] * 2)

    edges = _percentile_price_edges(ml, bands)
    idx = _percentile_band_index(ml, edges)
    lo_hi = [(None, edges[0])] + [(edges[i], edges[i + 1])
                                  for i in range(len(edges) - 1)] + [(edges[-1], None)]

    out = [f"market calibration on equal-count price bands "
           f"({len(np.unique(idx))} of {bands} populated; whole ledger, every graded family)",
           f"  {int(ok.sum())} games / {len(ml)} team-side closing prices. "
           f"Bands are nearest-rank, recomputed from THESE rows every build.",
           "  Edges move as the book does, so a band label is NOT comparable "
           "across builds or with the fixed ladder above;",
           "    the ladder is the stable axis, this is the balanced one. "
           "Read a row against its own SE, never against another build's.",
           f"  {'band':>16} {'n':>5} {'pair':>4} {'act':>6} {'imp':>6} "
           f"{'gap':>7} {'se':>6} {'z':>6}"]
    cells, zs, labels = [], [], {}
    for i in range(len(lo_hi)):
        m = idx == i
        if not m.any():
            continue
        # Labelled from the band's OWN observed prices, never from the edge
        # arithmetic: an edge plus one can name a price American odds cannot
        # take (nothing lies strictly between -100 and +100), and a label that
        # names an impossible price is a label a reader cannot check.
        b_lo, b_hi = int(ml[m].min()), int(ml[m].max())
        lab = f"{b_lo:+d}..{b_hi:+d}" if b_lo != b_hi else f"{b_lo:+d}"
        labels[i] = lab
        _, counts = np.unique(game[m], return_counts=True)
        pairs = int((counts == 2).sum())
        se = excess_se(p[m])
        gap = float(won[m].mean() - p[m].mean())
        z = gap / se if se and np.isfinite(se) and se > 0 else float("nan")
        zs.append(abs(z))
        cells.append(p[m])
        out.append(f"  {lab:>16} {int(m.sum()):5d} {pairs:4d} "
                   f"{100 * won[m].mean():6.1f} {100 * p[m].mean():6.1f} "
                   f"{100 * gap:+7.1f} {100 * se:6.1f} {z:+6.2f}")
    # A partition is a search over its own bands, so the best one is a maximum.
    if len(cells) > 1:
        null_best = _grid_null_best_excess(cells)
        best = max(range(len(cells)),
                   key=lambda j: float(won[idx == j].mean() - p[idx == j].mean()))
        obs = float(won[idx == best].mean() - p[idx == best].mean())
        out.append(f"  best-band reference: the best of {len(cells)} bands averages "
                   f"{100 * null_best:+.1f} pp under 'every game settles at its own "
                   f"price'; observed best {100 * obs:+.1f} pp -- "
                   f"{_search_verdict(100 * obs, 100 * null_best)}.")
        out.append(f"    Max |z| across the bands is {max(zs):.2f} against the "
                   f"{np.sqrt(2 * np.log(len(cells))):.2f} a search this wide "
                   "typically returns from noise.")
    # Pooling licence, per band: the pooled both-sides form is the identity
    # above and cannot answer this, so the families are compared within bands.
    lz, lab_of = [], {}
    for i in range(len(lo_hi)):
        a, b = (idx == i) & fam, (idx == i) & ~fam
        if a.sum() < 5 or b.sum() < 5:
            continue
        lab_of[len(lz)] = labels[i]
        sa, sb = excess_se(p[a]), excess_se(p[b])
        d = float((won[a].mean() - p[a].mean()) - (won[b].mean() - p[b].mean()))
        s = float(np.sqrt(sa ** 2 + sb ** 2))
        if s > 0:
            lz.append(abs(d) / s)
    if lz:
        exp = float(np.sqrt(2 * np.log(len(lz))))
        # The verdict is DERIVED, never asserted. The largest of k comparisons
        # is a maximum, so it is read against what k nulls typically return and
        # not against zero -- and when it lands above that, the line has to say
        # so. A licence sentence that reads "no sign" beside a number saying
        # otherwise is the publishing-a-claim-the-data-cannot-support entry.
        # Three outcomes, not two. `max(lz) <= exp` is a knife edge and it
        # landed on the knife: 2.038953 against 2.039334 on 2026-09-17, a
        # margin of 0.0004 that both sides of the comparison round away, so
        # the line read "2.04 ... against 2.04 ... licensed". A verdict whose
        # two inputs print identically has to say it is a tie; otherwise the
        # reader is told the licence is clean when it turned on the fourth
        # decimal of a simulated bar.
        worst = lab_of[int(np.argmax(lz))]
        margin = exp - max(lz)
        if _prints_the_same(max(lz), exp, ".3f"):
            verdict = (f"level with it, in band '{worst}' -- a margin of "
                       f"{margin:+.4f}, which both figures above round away. "
                       "Read this as a tie rather than a licence: the bar is "
                       "the MEAN of simulated maxima, so landing on it is not "
                       "clearing it.")
        elif max(lz) <= exp:
            verdict = ("at or below what a search this wide returns from "
                       "noise, so pooling the families is licensed.")
        else:
            verdict = (f"ABOVE it, in band '{worst}'. One band at "
                       "the noise maximum is not a finding, but the licence is "
                       "not clean either -- read the pooled rows knowing that.")
        out.append(f"  pooling licence: current family against the rest, within bands, "
                   f"max |z| {max(lz):.3f} over {len(lz)} comparable bands,")
        out.append(f"    against {exp:.3f} expected from noise -- {verdict}")
    return out


def _selection_price_matrix_lines(g):
    """|xw_net| bands x the LEAN's own closing price rung, every decided row.

    This is the grid the per-game card publishes one cell of, brought into the
    internal artifact. It is deliberately NOT the block above it, and the two
    are meant to be read as different instruments rather than reconciled:

      * that grid bands the market's own probability `q` of the LEAN and reads
        saved PREGAME prices with no close fallback, so it scores fewer rows;
      * this one buckets the published selection on the closing moneyline
        ladder, which is the axis a reader of the card sees, and scores every
        decided row of the current family.

    **It scored FOLLOW rows only until 2026-09-18, and that filter went with
    the rule.** v13 retires the hybrid selection, so the published side is the
    model's own lean on every decided game -- and a grid that still excluded
    the 25 rows the retired rule would have faded would be cutting the ledger
    on a row set defined by a rule nothing runs. The card changed in the same
    commit for the same reason, and a test holds the two equal cell by cell.

    The LEAN columns, not the bet columns. `ml_bet` / `p_bet` / `bet_won` name
    the side the retired rule selected, which on a faded row is the opposite
    club at the opposite price; dropping the filter while keeping them would
    have silently scored 25 games at the wrong side's odds. `lean_ml`,
    `model_side_p`, `lean_won` and `lean_profit` are the lean's own throughout.

    `hybrid_v2.apply_rule` is still where the arithmetic comes from, never a
    local copy, and the rungs from `market_backfill.ladder_rung`, so a cell
    here and the same cell on the card cannot drift apart. That equality is
    the whole point of the block: the site publishes per-cell units and this
    file is their counterpart. Reading the lean columns off a function that
    also computes a retired rule's branch is deliberate -- it is the one home
    for the ledger-to-bet mapping, and a second spelling is the defect this
    repo tracks most closely.

    Retrospective, and a grid is a search -- the null-max line at the foot is
    the reference a cell is read against, never zero.
    """
    import hybrid_v2
    # PUBLISHED rows, not raw ledger rows. This block calls itself the grid
    # the game card shows one cell of, and until this was added it was not:
    # the card bands on the reconstructed delta and scores the re-decided
    # lean, while this read `xw_net` / `xw_lean` / `xw_full` straight off the
    # ledger, which on a retained row is v12's. Measured before the fix --
    # 444 of 452 deltas differed by up to 0.0215 (wider than a band), the
    # lean differed on 39 rows, and 24 of 26 cells disagreed.
    #
    # `publish_reconstruction` is the one home for that substitution and
    # build_site renders from the same function, so the two cannot drift
    # again. Scoped to THIS block: the family history lines above and every
    # registration keep scoring the lean each build actually published, which
    # is the difference `_published_basis_lines` declares on the artifact.
    g = publish_reconstruction(g, MODEL_TAG)
    d = hybrid_v2.decidable(g)
    if d is None or d.empty:
        return []
    h = hybrid_v2.apply_rule(d)
    if h.empty:
        return []

    mag = pd.to_numeric(h["xw_net"], errors="coerce").abs().to_numpy(dtype=float)
    ml = pd.to_numeric(h["lean_ml"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(h["model_side_p"], errors="coerce").to_numpy(dtype=float)
    profit = pd.to_numeric(h["lean_profit"], errors="coerce").to_numpy(dtype=float)
    won = h["lean_won"].to_numpy(dtype=bool)
    rung = np.array([ladder_rung(float(m)) if np.isfinite(m) else None
                     for m in ml], dtype=object)
    bands = list(zip(FIXED_MAGNITUDE_EDGES, FIXED_MAGNITUDE_EDGES[1:]))
    labels = [lab for _lo, _hi, lab in ODDS_LADDER]
    # Short headers only: the full rung labels are 12 characters and eight of
    # them do not fit a readable row.
    short = {lab: lab.replace(" to ", "/").replace(" ", "") for lab in labels}

    def fmt(mask, kind):
        if not mask.any():
            return "·"
        n = int(mask.sum())
        w = int(won[mask].sum())
        if kind == "wl":
            return f"{n}:{w}-{n - w}"
        if kind == "u":
            return f"{profit[mask].sum():+.2f}"
        exc = won[mask].mean() - p[mask].mean()
        return f"{100 * exc:+.0f}±{100 * excess_se(pd.Series(p[mask])):.0f}"

    n_all, w_all = len(h), int(won.sum())
    out = [
        f"{MODEL_METRIC_LABEL} |delta| x LEANED-side closing price "
        f"(analyst-only; the game card publishes no cell of it)",
        f"  rows: all {n_all} decidable rows of the current family. The v13 "
        f"selection IS the lean, so no row is excluded for a branch; this "
        f"block scored the retired rule's FOLLOW subset until 2026-09-18.",
        "  Price basis: the lean's own CLOSING moneyline -- not the saved "
        "pregame price the block above uses, so the two grids score different "
        "row sets on purpose.",
        "  Cells are descriptive history, not a validated mapping from "
        "|delta| to a win probability, and no cell is a registered rule.",
        "  The game card showed three of these cells until 2026-09-22 and now "
        "shows none: no cell here is readable, because the best of them under "
        "'market correct, no edge' clears breakeven by ~+40 pp on average and "
        "the |delta| bands are not ordered. The grid stays HERE, beside its "
        "spreads and its search reference, which is what the card never had.",
    ]
    width, lab_w = 12, 15
    for title, kind in (("n and W-L", "wl"),
                        ("flat-stake units (1u a game, at each row's own close)", "u"),
                        ("beat its own price (pp +- se)", "e")):
        out.append(f"  {title}")
        head = "    " + f"{'|delta|':<{lab_w}}" + "".join(
            f"{short[lab]:>{width}}" for lab in labels) + f"{'ROW':>{width}}"
        out.append(head)
        out.append("    " + "-" * (len(head) - 4))
        for lo, hi in bands:
            row = (mag >= lo) & (mag < hi)
            cells = [fmt(row & (rung == lab), kind) for lab in labels]
            cells.append(fmt(row, kind))
            out.append("    " + f"{_band_label(lo, hi):<{lab_w}}"
                       + "".join(f"{c:>{width}}" for c in cells))
        cols = [fmt(rung == lab, kind) for lab in labels]
        cols.append(fmt(np.ones(n_all, dtype=bool), kind))
        out.append("    " + f"{'COLUMN':<{lab_w}}"
                   + "".join(f"{c:>{width}}" for c in cols))

    # A grid is a search: the best cell is read against what the best cell
    # averages when every game settles at its own price. Empty cells are
    # rendered above and excluded here -- a cell with no rows is a finding
    # about the model, not a candidate.
    cell_p, best, best_lab = [], None, ""
    for lo, hi in bands:
        row = (mag >= lo) & (mag < hi)
        for lab in labels:
            m = row & (rung == lab)
            if not m.any():
                continue
            cell_p.append(p[m])
            exc = float(won[m].mean() - p[m].mean())
            if best is None or exc > best:
                best, best_lab = exc, f"|delta| {_band_label(lo, hi)} {lab} (n={int(m.sum())})"
    ref = _grid_null_best_excess(cell_p)
    pooled = float(won.mean() - p.mean())
    out.append(
        f"  best-cell reference: the best of these {len(cell_p)} non-empty cells averages "
        f"{100 * ref:+.1f} pp of excess under 'every game settles at its own price'; "
        f"the observed best is {100 * best:+.1f} pp, {best_lab} -- "
        f"{_search_verdict(100 * best, 100 * ref)}. "
        f"A grid is a search, so a cell is read against that reference, never against zero.")
    out.append(
        f"  pooled over all {n_all} rows: {w_all}-{n_all - w_all} "
        f"({w_all / n_all:.3f})   excess {100 * pooled:+.1f} +- "
        f"{100 * excess_se(pd.Series(p)):.1f} pp   {profit.sum():+.2f}u. "
        f"The margins are better estimated than any cell; read them first.")
    return out


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


def _registration_retrospective_lines(g):
    """In-sample readings of the three registrations frozen 2026-09-03.

    Answers the question a reader has the moment five forward blocks all say
    "nothing to score yet": what do these rules say about the v12 data that
    ALREADY exists? They can be computed over it -- that is where every frozen
    discovery constant came from -- and printing them is the same choice the
    hybrid retrospective above already makes for the shipped rule.

    Two things keep it honest, and both matter more here than for the hybrid.

    Each reading comes from the registering module's OWN pure function, never a
    local copy, so a line here cannot drift from the forward block below it.

    And the row count is split at the registration date. A retrospective over
    "all v12" is a MIXTURE of the rows the rule was found on and the rows that
    arrived afterwards, so watching it grow is NOT watching evidence
    accumulate -- the forward blocks below score the second group alone, and
    that is the only reading that is out of sample. The discovery figure is
    printed beside each so drift is visible as drift rather than as news.
    """
    try:
        import abstain_test
        import delta_filter_test
        import dog_contrast_test
        import hybrid_test
        d = hybrid_test.decidable(g)
        if d is None or d.empty:
            return []
        h = hybrid_test.apply_rule(d)
    except Exception as _exc:                      # noqa: BLE001
        # Same load-bearing guard as every other block in this report: it runs
        # inside the job that ingests pregame rows, and a diagnostic line must
        # never be able to cost a slate.
        return [f"registration retrospectives unavailable ({type(_exc).__name__})"]

    reg = delta_filter_test.REGISTERED_ON
    dates = h["game_date"].astype(str)
    n_disc = int((dates <= reg).sum())
    n_fwd = int(len(h) - n_disc)
    out = [f"REGISTRATIONS (retrospective, n={len(h)}: {n_disc} discovery rows "
           f"+ {n_fwd} since {reg}) — in sample, NOT the forward reading:"]
    out.append("  price source — all retrospectives: closing close_p_home for "
               "eligibility, selection, and implied probabilities;")
    out.append("                 closing close_home_ml/close_away_ml for returns. "
               "No saved-pregame fallback.")
    out.append("  missing prices — no close_p_home or lean: excluded; |Δ| also "
               "requires xw_net. Paired close MLs are not a separate filter.")

    # |delta| filter -- registered headline is the DROPPED games' excess.
    try:
        fd = delta_filter_test.decidable(g)
        f = delta_filter_test.apply_filter(fd) if fd is not None else None
        if f is not None and len(f):
            dr = f[~f["kept"]]
            if len(dr):
                e, se = delta_filter_test._excess(
                    dr["lean_won"].astype(bool).to_numpy(),
                    dr["p_lean"].to_numpy(dtype=float))
                out.append(
                    f"  |Δ| filter   dropped-game excess {100 * e:+6.2f}pp "
                    f"+/- {100 * se:4.2f}  (n={len(dr)}; discovery "
                    f"{delta_filter_test.DISCOVERY_DROPPED_EXCESS:+.2f}). "
                    "NEGATIVE would vindicate the rule.")
    except Exception as _exc:                      # noqa: BLE001
        out.append(f"  |Δ| filter   unavailable ({type(_exc).__name__})")

    # abstain vs fade -- registered headline is per declined game.
    try:
        m, se, n_d = abstain_test.fade_minus_abstain(h)
        if n_d:
            out.append(
                f"  abstain      fade-minus-abstain {m:+.3f}u/declined game "
                f"+/- {se:.3f}  (n={n_d}; discovery "
                f"{abstain_test.DISCOVERY_FADE_MINUS_ABSTAIN:+.3f}). "
                # NOT "the shipped fade branch". The declined set is the
                # q-gate's -- v1's unconditional q < .45 -- and v2 fades a
                # strict subset of it, following 16 of these 37. The 2026-09-17
                # correction that established this rewrote every copy of the
                # claim inside `abstain_test` and missed this one, which is the
                # MORE prominent of the two: the forward block carries the
                # corrected wording 310 lines further down a file most readers
                # skim from the top. A caveat travels with the line someone
                # wrote it on, not with the statistic.
                "POSITIVE keeps the q-gate fade branch, which v2 fades a "
                "strict subset of.")
    except Exception as _exc:                      # noqa: BLE001
        out.append(f"  abstain      unavailable ({type(_exc).__name__})")

    # underdog sign flip -- registered headline is the contrast.
    try:
        dogs = h[h["model_side_p"].astype(float) < dog_contrast_test.DOG_MAX]
        c, cse, na, nb = dog_contrast_test.contrast(dogs)
        if na and nb:
            out.append(
                f"  dog contrast above-minus-below {100 * c:+6.2f}pp "
                f"+/- {100 * cse:4.2f}  (n={na} above, {nb} below; discovery "
                f"{dog_contrast_test.DISCOVERY_CONTRAST:+.2f}).")
    except Exception as _exc:                      # noqa: BLE001
        out.append(f"  dog contrast unavailable ({type(_exc).__name__})")

    if len(out) == 1:
        return []
    out.append("  Each was FOUND on the discovery rows above, so none of these "
               "is evidence for itself. The registered forward readings are "
               "further down this report.")
    return out


def _hybrid_retrospective_lines(g):
    """The published rule's record over the CURRENT family's graded rows.

    Why this exists: the public pages headline the hybrid selection while this
    report headlined the raw lean, so the two artifacts disagreed about "the
    model's record" -- 146-77 on the site against 139-84 here, on the same
    games. That is the internal-vs-public divergence this repo has an entry
    for, and the fix is to print both from one derivation rather than to pick
    a winner: the lean line above is the control the hybrid line has to be read
    against, and the chalk line is the control they BOTH have to be read
    against.

    Retrospective, and labelled so. Both v2 gates were chosen after examining
    these rows, so nothing here is out-of-sample; the registered v2 forward
    reading further down scores only later slates.

    Arithmetic comes from `hybrid_v2.apply_rule`, never a local copy -- one
    rule, one implementation, so this line and the forward block below cannot
    drift apart.
    """
    try:
        import hybrid_v2
        d = hybrid_v2.decidable(g)
        if d is None or d.empty:
            return []
        h = hybrid_v2.apply_rule(d)
    except Exception as _exc:                      # noqa: BLE001
        # Same load-bearing guard as the forward-test block: this function runs
        # inside the job that ingests pregame rows, and a cosmetic line must
        # never be able to cost a slate.
        return [f"hybrid (retrospective) unavailable ({type(_exc).__name__})"]

    n = len(h)
    won = h["bet_won"].astype(bool)
    w, l = int(won.sum()), int(n - won.sum())
    n_fade = int((~h["follow"]).sum())
    se = float(np.sqrt(np.sum(h["p_bet"] * (1 - h["p_bet"]))) / n)
    z = (won.mean() - h["p_bet"].mean()) / se if se > 0 else float("nan")
    units = float(h["profit"].sum())
    cw = int(h["chalk_won"].astype(bool).sum())
    cse = float(np.sqrt(np.sum(h["chalk_p"] * (1 - h["chalk_p"]))) / n)
    cz = ((h["chalk_won"].astype(bool).mean() - h["chalk_p"].mean()) / cse
          if cse > 0 else float("nan"))
    out = [
        f"hybrid v2 rule  full: {w}-{l}  ({w / (w + l):.3f})"
        if (w + l) else f"hybrid v2 rule  full: {w}-{l}",
    ]
    out[0] += (f"   vs price z={z:+.2f}  {units:+.2f}u   "
               f"(n={n}, {n_fade} faded)")
    chalk_line = (f"  always chalk, same {n} rows: {cw}-{n - cw}  "
                  f"({cw / n:.3f})   vs price z={cz:+.2f}")
    # A game at exactly .500 has no favourite, so these rows sit on the
    # tie-break rather than on a price. Stated rather than absorbed: the
    # convention lives in `market_backfill.chalk_is_home` precisely so this
    # line and the grades page cannot answer it differently, and a shared
    # convention is only auditable if its footprint is printed.
    n_pk = int(is_pickem(pd.to_numeric(h["close_p_home"], errors="coerce")).sum())
    if n_pk:
        chalk_line += (f"   ({n_pk} priced exactly .500: no favourite, "
                       "tie to home)")
    out.append(chalk_line)
    out.append("  price source — selection/eligibility and market comparison: "
               "closing close_p_home;")
    out.append("                 returns: closing close_home_ml/close_away_ml. "
               "No saved-pregame fallback.")
    out.append("  missing prices — no close_p_home or lean: excluded. Paired "
               "close MLs are not a separate filter.")
    out.append("  RETROSPECTIVE: the q < 45% AND |xw_net| < .012 fade gate "
               "was chosen after examining these rows. Registered v2 starts "
               "after 2026-09-11 and is reported below.")
    return out


def _hybrid_materialized_lines(g):
    """V2 ledger fields on each row's recorded basis, kept separate from close.

    This is not the uniform historical comparison above: legacy rows use their
    close, while rows captured after decision locking retain saved pregame
    selections and prices. Printing it makes the CSV's recomputed fields
    reconcilable without pretending the two snapshots are interchangeable.
    """
    try:
        import hybrid_v2
        needed = {"selection_rule_tag", "hybrid_action", "hybrid_selection",
                  "hybrid_p", "hybrid_ml", "hybrid_full",
                  "hybrid_price_source"}
        if not needed.issubset(g.columns):
            return []
        s = g[(g["selection_rule_tag"] == hybrid_v2.RULE_TAG)
              & g["hybrid_action"].isin(["FOLLOW", "FADE"])
              & g["hybrid_full"].isin(["W", "L"])
              & pd.to_numeric(g["hybrid_p"], errors="coerce").notna()
              & pd.to_numeric(g["hybrid_ml"], errors="coerce").notna()].copy()
        if s.empty:
            return []
        won = s["hybrid_full"].eq("W").to_numpy()
        p = pd.to_numeric(s["hybrid_p"], errors="coerce").to_numpy(float)
        ml = pd.to_numeric(s["hybrid_ml"], errors="coerce").to_numpy(float)
        profit = np.where(won, np.where(ml > 0, ml / 100, 100 / np.abs(ml)), -1)
        z = hybrid_v2._excess_z(won, p)
        counts = s["hybrid_price_source"].value_counts()
        basis = ", ".join(f"{int(counts[k])} {k.replace('_', ' ')}"
                          for k in ("closing", "saved_pregame") if k in counts)
        fades = int(s["hybrid_action"].eq("FADE").sum())
        w = int(won.sum())
        return [
            f"hybrid v2 ledger fields (row basis): {w}-{len(s)-w}  "
            f"({w / len(s):.3f})   vs price z={z:+.2f}  "
            f"{profit.sum():+.2f}u   (n={len(s)}, {fades} faded)",
            f"  price source — {basis}; no cross-snapshot fallback. This mixed-"
            "basis reconciliation is not the uniform closing-price retrospective above.",
        ]
    except Exception as _exc:                      # noqa: BLE001
        return [f"hybrid v2 ledger reconciliation unavailable "
                f"({type(_exc).__name__})"]


def _published_basis_lines(g):
    """Declare the one way this report and the public pages disagree.

    `RECORD_TAGS` shares a line across v12 and v13, and the two artifacts
    score those retained v12 rows DIFFERENTLY on purpose. This report scores
    every row on the lean its own build published -- the immutable pregame
    record, and the control the new model has to be read against. The public
    pages score a retained row on `reconstruct_v13`'s re-decision of it, and
    since 2026-09-18, on the operator's instruction, they do so without
    marking which rows those are.

    So the numbers differ, and this repo's standing rule is that an internal
    and a public artifact may differ only if the difference is DECLARED --
    `ledger_report.txt` once said the current family had no graded games while
    the site published a pooled record, and nothing on either said why. This
    is that declaration, and it is the only place the split is now printed.

    Counted from the rows' own tags, never by subtracting one published
    number from another. Empty once every row in the family was built under
    the current tag, which is the state this whole clause exists to bridge to.
    """
    if g is None or not len(g) or "model_tag" not in getattr(g, "columns", ()):
        return []
    retained = g[~g["model_tag"].astype(str).eq(MODEL_TAG)]
    if retained.empty or "v13_lean_recon" not in g.columns:
        return []
    n_rebuilt = int(retained["v13_lean_recon"].notna().sum())
    if not n_rebuilt:
        return []
    return [
        f"  BASIS: {n_rebuilt} of these {len(g)} rows were published under an "
        f"earlier tag and are scored ABOVE on their own pregame lean.",
        "  The public pages re-decide those rows under the current model and "
        "blend them in unmarked, so their record is NOT this one and is "
        "hindsight on the re-decided rows.",
    ]


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
    # Regular season only. A frame carrying postseason or unconfirmed rows --
    # main()'s combined frame, or a caller's -- is split here so no block
    # below can count them; they get a footer of their own.
    led, held = season_phase.split_regular(led)
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
        for _bl in _published_basis_lines(g):
            say(_bl)
        for _hl in _hybrid_retrospective_lines(g):
            say(_hl)
        for _ml in _hybrid_materialized_lines(g):
            say(_ml)
        for _rl in _registration_retrospective_lines(g):
            say(_rl)
        ov = g[g["ops_valid"] == True]                                # noqa: E712
        if len(ov):
            say(f"platoon lean full: {_rec(ov['ops_full'])}   F5: {_rec(ov['ops_f5'])}   (reliable-only, n={len(ov)})")
            say(f"{MODEL_METRIC_LABEL} on same subset  full: {_rec(ov['xw_full'])}   F5: {_rec(ov['xw_f5'])}")
        for line in _fixed_magnitude_lines(g):
            say(line)
        # The grid the block above cannot express: the same bands crossed with
        # the market's own probability of the leaned side. Guarded like every
        # other diagnostic here -- this runs in the job that ingests pregame
        # rows, and a report line must never be able to cost a slate.
        try:
            for line in _magnitude_price_grid_lines(g):
                say(line)
        except Exception as _exc:                  # noqa: BLE001 - see above
            say(f"{MODEL_METRIC_LABEL} |delta| x saved-pregame market "
                f"probability unavailable ({type(_exc).__name__})")
        try:
            for line in _selection_price_matrix_lines(g):
                say(line)
        except Exception as _exc:                  # noqa: BLE001 - see above
            say(f"{MODEL_METRIC_LABEL} |delta| x selected-side closing price "
                f"unavailable ({type(_exc).__name__})")

        # v13 dynamic-price calibration SHADOW. This is analysis only: it reads
        # the same reconstructed/native v13 history the public UI already uses,
        # then walks it chronologically with fixed 5x4 cells and M0=10 market
        # shrinkage. It never writes a model field or changes a ledger decision.
        # Closing prices are used here on purpose so the entire reconstruction
        # has one uniform retrospective basis; the block labels that basis and
        # the reconstruction hindsight explicitly.
        try:
            import price_calibration_shadow
            for line in price_calibration_shadow.report_lines(g, model_tag=MODEL_TAG):
                say(line)
        except Exception as _exc:                  # noqa: BLE001 - see above
            say(f"v13 dynamic-price calibration shadow unavailable "
                f"({type(_exc).__name__})")

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

    # Printed directly beneath the component block, and beneath rather than
    # above on purpose: it is the thing that says whether the slopes above are
    # readable at all. Guarded like every other diagnostic in this function --
    # this runs in the job that ingests irreplaceable pregame rows, and a
    # report line must never be able to cost a slate.
    try:
        for _ln in target_reliability(led):
            say(_ln)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"target reliability unavailable ({type(_exc).__name__})")

    # Whole ledger, NOT RECORD_TAGS, and the exception is the point: every
    # block above scores one prediction family because a level is only
    # comparable within one. This one is scoped by DATE and keeps pooling legal
    # by grouping on the metric instead, which is what lets it survive a bump —
    # scoped to the record family it would have gone blank the morning wOBA v5
    # landed, on exactly the slates a per-slate check exists to watch.
    for _ln in slate_lines(led):
        say(_ln)

    # Same whole-ledger licence as the block above, for the same reason: a
    # realised rate against a devigged close is arithmetic on a box score and a
    # price, so the prediction family is not part of its definition. Wrapped
    # because this file's blocks never take the report down with them.
    try:
        for _ln in _market_percentile_band_lines(led):
            say(_ln)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"market percentile bands unavailable ({type(_exc).__name__})")

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

    # Standing monitor for the pre-registered forward test. Printed every build
    # rather than left to be run on demand, because a pre-registration nobody
    # looks at decays into a file: the whole point is that the number arrives
    # without anyone remembering it exists (the expected-IP deferral landed at
    # its gate for exactly this reason).
    #
    # Guarded, and the guard is load-bearing. This function feeds `report()`,
    # which runs inside the same job that ingests pregame rows -- and a pregame
    # row that fails to land cannot be re-derived afterwards without lookahead.
    # A cosmetic monitor must never be able to cost a slate, so any failure
    # degrades to one line and grading continues. Same reasoning as keeping the
    # test suite out of build.yml.
    try:
        import forward_test
        say("")
        for _fl in forward_test.report_lines(led):
            say(_fl)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered forward test unavailable ({type(_exc).__name__})")

    # Second registration, same reasoning and the same load-bearing guard: the
    # hybrid market-direction rule (hybrid_test.py), registered separately from
    # the two arms above because it holds a different hypothesis and a
    # different registration date. Kept as its own module rather than a third
    # arm so neither registration's frozen block can be edited while reaching
    # for the other's.
    try:
        import hybrid_test
        say("")
        for _hl in hybrid_test.report_lines(led):
            say(_hl)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered hybrid v1 test unavailable ({type(_exc).__name__})")

    # Current production rule. Version 1 above remains frozen so its forward
    # evidence is not rewritten by a later, data-informed rule change.
    try:
        import hybrid_v2
        say("")
        for _hl in hybrid_v2.report_lines(led):
            say(_hl)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered hybrid v2 test unavailable ({type(_exc).__name__})")

    # |delta| conviction filter (delta_filter_test.py), registered 2026-09-03.
    # Third module rather than a third arm, for the same reason as above. Its
    # prior is NEGATIVE where the hybrid's is null -- the always-chalk control
    # on its discovery rows points against the rule -- so the two must not be
    # read as one another, and printing them adjacently is what makes that
    # visible rather than a fact you have to go and look up.
    try:
        import delta_filter_test
        say("")
        for _dl in delta_filter_test.report_lines(led):
            say(_dl)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered delta filter test unavailable ({type(_exc).__name__})")

    # abstain-vs-fade (abstain_test.py), registered 2026-09-03. It shares the
    # hybrid's threshold and declined set on purpose -- the two rules differ
    # only in what happens on a declined game -- so it prints directly after
    # the hybrid block, where the comparison is legible. Its own registration
    # date is two days later than the hybrid's and is NOT delegated.
    try:
        import abstain_test
        say("")
        for _al in abstain_test.report_lines(led):
            say(_al)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered abstain test unavailable ({type(_exc).__name__})")

    # underdog sign-flip (dog_contrast_test.py), registered 2026-09-03. Prints
    # last of the five because it is the one that is explicitly NOT independent
    # of the others -- its below-split half is the same 20 games arm 2 bets and
    # abstain_test declines. Reading it beside them is the point; reading it as
    # a fifth sample is the error it warns about in its own report line.
    try:
        import dog_contrast_test
        say("")
        for _cl in dog_contrast_test.report_lines(led):
            say(_cl)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered dog contrast test unavailable ({type(_exc).__name__})")

    # Candidate B2 individual-extreme TMR10 shadow test. Registered 2026-09-18
    # after the historical mechanism comparison was complete. This block is
    # intentionally prospective only: b2_tmr_test reconstructs each day's
    # state from PRIOR completed ledger games, then scores only actual v13 rows
    # strictly after registration against their saved pregame market. Like the
    # other monitors it is guarded because a reporting diagnostic must never
    # be able to cost an irreplaceable pregame snapshot.
    try:
        import b2_tmr_test
        say("")
        for _b2l in b2_tmr_test.report_lines(led):
            say(_b2l)
    except Exception as _exc:                      # noqa: BLE001 - see above
        say(f"pre-registered B2 TMR10 test unavailable ({type(_exc).__name__})")

    lines.extend(_held_lines(held))
    return "\n".join(lines)


def report(led):
    """Print the report and write it to REPORT_PATH. The only writer."""
    txt = report_text(led)
    print("=" * 60); print(txt); print("=" * 60)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    led = load_ledger(include_held=True)
    led = ingest(led)
    led = resolve_game_types(led)
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
    save_ledger(led)
    report(led)

if __name__ == "__main__":
    main()
