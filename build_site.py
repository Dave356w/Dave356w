#!/usr/bin/env python3
# ============================================================
# MLB Probable-Pitcher vs Opponent-Lineup matchup -- STATIC SITE GENERATOR
#
# Render-free build of the Colab notebook
# (Shrunk_mlb_matchup_render_consolidated.ipynb), turned into a one-shot
# static-site generator: it pulls the slate via keyless APIs, builds the
# wOBA + platoon-OPS matchup tables, and writes a fully self-contained
# index.html (inline CSS, no external assets, dark-mode via
# prefers-color-scheme) for GitHub Pages.
#
# Data sources (all keyless):
#   - MLB StatsAPI          : slate, probables, rosters, bio, vL/vR splits
#   - Savant gf?game_pk=     : posted lineups (JSON)
#   - Savant CSV leaderboards (cached once/day): custom + batted-ball
#
# Pipeline mirrors notebook cells 1 (fetch) -> 4 (primary-rate matchup) ->
# 5 (platoon split) -> 6 (consolidated card render). The notebook's
# clean/validate cells (2, 3) are skipped: they produced *_clean / *_strict
# frames that the matchup/render cells never consume (the API build_tables
# output is already clean).
#
# Behaviour for unattended runs:
#   - SLATE_DATE is computed in America/New_York with a ~3am ET rollover
#     (override with the SLATE_DATE env var, YYYY-MM-DD).
#   - A top-level guard means a failed/empty core fetch raises a non-zero
#     exit WITHOUT writing index.html, so CI skips the deploy and the last
#     good page stays live. A legitimate empty slate (off-day) writes a
#     friendly "no games" page instead (a successful build).
#   - index.html carries a "built HH:MM" line beside the slate date; every
#     clock on the site is Pacific and the legend is where that is stated.
# ============================================================

import base64
import io
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import hybrid_test
import pitch_arsenal
import player_priors

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")   # display timezone (build time, game times)
UTC = timezone.utc
ROLLOVER_HOUR = 3          # hold "today" until ~3am ET so night games don't roll early
SPORT_ID = 1
LINEUP_SIZE = 9
REQUEST_DELAY = 0.25

# Column carrying a hitter's 1-9 batting-order slot through the pipeline.
BATTING_ORDER_COL = "batting_order"
# Savant xwOBA is the active model input. v11 restores it after the wOBA
# forward test (wOBA v1-v5) ran 63-76 against always-home 84-55; see
# _RECORD_FAMILIES for what the revert keeps and what the evidence for each
# piece actually was. The dump/ledger keys have always been xwOBA-shaped, so
# under this metric the compatibility schema and the statistic agree again --
# but they are still two different things, and every dump carries
# model_metric explicitly rather than letting the key names imply it.
# These constants define the source of truth for all active batter, starter,
# bullpen, league-prior, percentile, and pitch-mix inputs.
MODEL_RATE_SOURCE_COL = "xwoba"
MODEL_RATE_LABEL = "xwOBA"
PUBLIC_MODEL_NAME = "XWOBA Market Hybrid"
HYBRID_RULE_TAG = hybrid_test.RULE_TAG
MODEL_RATE_INTERNAL_COL = "xwOBA"  # stable dump/ledger schema, both metrics
# True only for a posted hitter absent from the season Savant leaderboard, who
# therefore carries the active team's PA-weighted rate. Purely an in-process
# frame column -- it reaches no dump, ledger, or audit schema -- so it carries
# one name, not an alias pair. An alias would drift the way the workflow's
# MODEL_TAG pin did.
MODEL_RATE_TEAM_BACKFILL_COL = "xwOBA_team_backfill"
# Expected plate appearances per game by batting-order slot. A lineup turns over
# from the top, so the leadoff man bats ~4.6 times while the 9-hole bats ~3.8 --
# roughly a 22% spread in in-game exposure across the order. We weight lineup
# composites by this slot expectation instead of by each hitter's season volume
# (BBE / split PA), which otherwise over-weights whoever has simply logged more
# playing time rather than who will actually see more pitches tonight. Values are
# league PA/game-by-order-position anchors (modern 9-inning norms).
LINEUP_SLOT_PA = {1: 4.61, 2: 4.50, 3: 4.39, 4: 4.29, 5: 4.18,
                  6: 4.08, 7: 3.97, 8: 3.87, 9: 3.76}
USE_SLOT_PA_WEIGHTS = True
CACHE_DIR = os.environ.get("CACHE_DIR", ".")
# Where the frozen completed-season priors live (priors_snapshot.py writes
# them). Committed, unlike CACHE_DIR, which is the gitignored slate cache.
PRIORS_DIR = os.environ.get("PRIORS_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "public")
# Team cap logos are keyed by the same StatsAPI team id the slate already
# carries. They live on MLB's static CDN (not in the API payload), so the build
# fetches the on-light / on-dark cap SVGs once per run and inlines them as data
# URIs -- keeping the page fully self-contained. Any fetch miss degrades to the
# text abbreviation, so the CDN being unreachable never breaks the build.
USE_TEAM_LOGOS = os.environ.get("USE_TEAM_LOGOS", "1") != "0"
LOGO_CDN = "https://www.mlbstatic.com/team-logos"
DATA_DIR = os.environ.get("DATA_DIR", "data")            # grading ledger home
LEDGER_PATH = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")
MODEL_TAG = os.environ.get("MODEL_TAG", "xw+plat_consol_v12")  # keep in sync with grade_leans.py
if not MODEL_TAG.startswith("xw+"):
    raise RuntimeError(
        "This build fetches Savant xwOBA; refusing to stamp it with a non-xwOBA MODEL_TAG"
    )
_RECORD_FAMILIES = {
    # v3 changed only ledger locking/identity; its prediction math is v2.
    "xw+plat_consol_v3": ("xw+plat_consol_v2", "xw+plat_consol_v3"),
    # v4 re-weights lineup composites by expected PA per batting-order slot
    # (was season BBE / split PA); that changes prediction math, so it starts a
    # fresh record family and never mixes with v2/v3 in the ledger or weight fit.
    # v5 adds empirical-Bayes xwOBA shrinkage (batters + starter) on top of v4;
    # another prediction-math change, so it starts its own family again.
    # v6 replaces the starter-only pitching input with a nine-inning blend of
    # expected starter workload and a role-filtered bullpen aggregate. It is a
    # new prediction family; older tags remain in the ledger history.
    # v7 centre-matches the weighted and unweighted moments used to estimate
    # xwOBA shrinkage K. It changes prediction math and starts another family.
    # v8 replaces the per-build estimates with fixed K=100 shrinkage for both
    # batters and pitchers. It changes prediction math and starts another family.
    # v9 applies the starter-handedness lineup adjustment only to expected
    # starter innings, then uses the neutral lineup against the bullpen. It is
    # isolated from v8 for both records and delta-scale ranking.
    #
    # v10 blends the two phases on their share of plate appearances rather than
    # of innings, using measured BF/IP. Prediction math changed, so the tag is
    # bumped -- but the record question is answered by measurement, not reflex.
    # Sweeping the starter/bullpen BF/IP ratio over 0.95-1.10 on the 2026-07-28
    # slate moves xw_net with sd 0.0004-0.0009 against a median |xw_net| of
    # 0.0268 (1.6-3.4%) and flips 0 of 12 leans; only an implausible 1.20 ratio
    # flips one, on a game whose |xw_net| is 0.0013. v9 and v10 rows are
    # therefore produced by models that agree on the decision and share a
    # win-loss line. Isolating v10 would reset the graded sample to zero for
    # the eighth time in a month and buy nothing.
    "xw+plat_consol_v9": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
    "xw+plat_consol_v10": ("xw+plat_consol_v9", "xw+plat_consol_v10"),
    # Observed wOBA replaces xwOBA in every active rate input while the rest of
    # v10's construction stays fixed. This is a new prediction family.
    "woba+plat_consol_v1": ("woba+plat_consol_v1",),
    # v2 replaces the symmetric +/-0.010 starter platoon term with a
    # handedness-specific, exposure-centred 0.021 gap. On the 2026-08-04
    # projected slate it flips 0/14 pair-complete leans, but one slate is not
    # enough to establish decision equivalence, so its forward record starts
    # clean.
    "woba+plat_consol_v2": ("woba+plat_consol_v2",),
    # v3 raises XWOBA_SHRINK_K 100 -> 400 and moves the reliever shrink target
    # from the league PA-weighted batter rate to the relief pool's own
    # unweighted centre. NEW RECORD FAMILY, argued rather than inherited: a 4x
    # K changes every shrunk batter, starter and reliever rate, and the target
    # move shifts every bullpen number by a further ~0.010 before weighting.
    # That is not the v10 situation, where a reweight flipped 0 of 14 leans and
    # earned a shared line -- these are different predictions and they get a
    # clean record. The old family had 15 graded rows, so the cost of resetting
    # is 15 games.
    "woba+plat_consol_v3": ("woba+plat_consol_v3",),
    # v4 replaces the shrinkage TARGET: every batter, starter and reliever with
    # 2023-2025 Savant history now regresses toward his own recency-weighted
    # rate instead of a population centre. NEW RECORD FAMILY, argued not
    # inherited. This is not the v10 situation, where a convex reweight of the
    # same two phases flipped 0 of 14 leans and earned a shared line. Here the
    # early-season out-of-sample MSE moves 16-25% across all three populations
    # and the published rate of an established player moves by roughly the size
    # of the platoon term. Those are different predictions and they get a clean
    # record. The v3 family had its own graded rows and they stay immutable.
    "woba+plat_consol_v4": ("woba+plat_consol_v4",),
    # v5 abstains: a side whose starter has no leaderboard line contributes his
    # prior, not a measurement, so that game publishes no lean at all. NEW
    # RECORD FAMILY, and this half is not close. It does not change any surviving
    # prediction -- every game that still leans leans identically to v4 -- but it
    # changes WHICH games are decided, and a win-loss line is a property of the
    # decided set. Measured over the 185 ledger rows carrying starter/bullpen
    # instrumentation at the bump (the set whose quantiles are quoted below), 6
    # (3.2%) carried a prior-only starter on one side and would now abstain;
    # those 6 graded 3-3 against 96-82 (.539) on the rest. That is n=6 and is
    # quoted as incidence, NOT as evidence the abstained games were bad picks --
    # the case for abstaining is that the input was never measured, which is true
    # whatever those six did. v7 is the precedent: zero-as-abstention got its own
    # family for the same reason. Cost of the reset, stated rather than elided:
    # the v4 family's graded rows stay immutable. Read the count off the ledger
    # rather than from here -- it was 10 when this shipped on 2026-08-06 and 11
    # by that afternoon, which is the whole reason not to freeze it in prose.
    "woba+plat_consol_v5": ("woba+plat_consol_v5",),
    # Short-lived wOBA-lineup/xwOBA-arms experiment. It is retained only so
    # immutable dumps and ledger rows remain recognised after full wOBA was
    # restored; it never pools with the active wOBA record.
    "split+plat_consol_v1": ("split+plat_consol_v1",),
    # v11 reverts the metric to Savant xwOBA, K to 100, and the shrinkage
    # target to the population centre -- and KEEPS the three post-v10 changes
    # whose evidence is independent of those three. NEW RECORD FAMILY, which
    # is not the interesting half; what needs arguing is the composition, so
    # each piece is listed with what actually decided it.
    #
    # REVERTED
    #   metric wOBA -> xwOBA. Decided by the operator, not by a measurement,
    #     and the honest statement of the evidence is that there is none
    #     either way: the paired shadow arm exists precisely because the
    #     sequential comparison cannot settle it, and at 6 slates it reports
    #     d_corr +0.008 with CI [-0.108, +0.128]. The wOBA lineage's 63-76
    #     against always-home 84-55 is real and is not evidence -- the eras
    #     were different games. The shadow arm now runs wOBA, so the question
    #     keeps accumulating a paired answer under the revert.
    #   K 400 -> 100. This one overrides a direct measurement and the override
    #     is deliberate: `reliever_shrink_probe.py` fits K three ways
    #     (walk-forward 391 [293,519] on n=53,464; within-season 600/384/577)
    #     and every interval excludes 100. Nothing here disputes that fit. It
    #     is reverted because the operator asked for the K=100 model back, and
    #     that is a legitimate call to make against a fit -- but do not read
    #     the revert as a finding that 100 beat 400. Nothing measured that.
    #
    #     What the revert DOES change is whether that fit is about this
    #     constant at all, and the answer is no. The probe builds its rates
    #     from StatsAPI box lines (`WOBA_W`, `woba()`), so it fits a
    #     wOBA-denominated K -- correct when it ran, because the build was on
    #     wOBA and 400 shipped as wOBA v3. Under v11 the same constant shrinks
    #     xwOBA, and K = sigma^2/tau^2 is a property of the metric: xwOBA is
    #     (near enough) wOBA's conditional expectation given batted-ball
    #     shape, so by the law of total variance its per-BF sigma^2 is strictly
    #     smaller. tau^2 is NOT pinned, so this is a direction and not a
    #     magnitude -- but the direction is toward a smaller K, i.e. toward the
    #     100 now shipped. So the honest status of K=100 under v11 is
    #     UNMEASURED, not overridden.
    #
    #     And it cannot be measured with what this repo has. StatsAPI serves no
    #     xwOBA, and a per-past-date Savant pull is the lookahead the cache
    #     rule forbids, so the probe cannot be re-pointed at the live metric.
    #     Do not re-run it and read the output as an argument about v11: it
    #     will answer the wOBA question again, accurately, and about a
    #     constant this build no longer has.
    #   personal shrinkage targets -> population centre, via PLAYER_PRIORS=0,
    #     which restores the v3 target expression exactly (H -> 0) rather than
    #     adding a branch. Note the flag alone will not undo this under v11 --
    #     the frozen priors are wOBA and an xwOBA build refuses them; see
    #     USE_PLAYER_PRIORS. The out-of-sample probe favours personal priors by
    #     +7% to +25% weighted MSE with CIs excluding zero; the forward read on
    #     the ledger points the other way (lineup-component corr +0.124 on the
    #     v9/v10 rows against -0.004 on v5's) and does NOT separate either --
    #     bootstrapped, that difference is +0.128 with CI [-0.054, +0.301].
    #     Reverted on the operator's call, with both measurements on the record.
    #
    # KEPT, each because its evidence is independent of metric, K and target
    #   exposure-centred platoon offsets (wOBA v2). Not an empirical fit: a
    #     hitter's season line is ALREADY a platoon blend at his real exposure,
    #     so treating it as the midpoint of a 50/50 mix is arithmetically wrong
    #     whatever rate is inside it. Same conservative gap as the old
    #     universal term (0.021 against 0.020).
    #   relief-pool shrink target (wOBA v3's other half). Measured before it
    #     shipped: the league PA-weighted batter centre is the centre of no
    #     subpool, and the relief pool sits 0.0102 below it. That bias exists
    #     at any K -- K only decides how much of it lands. `relief_pool_prior`
    #     notes K and this target had to RISE together, because raising K
    #     against the old target would have doubled the bias; the reverse
    #     direction is safe, since lowering K just gives a correct centre less
    #     weight. Reverting it would restore a measured bias for no reason.
    #   starter abstention (wOBA v5). A side whose starter has no leaderboard
    #     line contributes a prior, not a measurement, and that is true of
    #     xwOBA at K=100 exactly as it was of wOBA at K=400. It makes no
    #     performance claim and never did.
    #
    # Everything from v6-v10 (expected-IP blend, phase split, PA-share
    # weighting) is untouched -- the revert stops at wOBA v1.
    "xw+plat_consol_v11": ("xw+plat_consol_v11",),
    # v12 calibrates `expected_sp_ip` against its own backfilled actuals. The
    # estimator is over-dispersed -- regressing actual on predicted over 586
    # side-games gives slope +0.735 +/- 0.048, 5.5 se below 1.0 -- and `q`, the
    # starter's share of plate appearances, is a function of it. The fix is a
    # per-build OLS refit shrunk toward the identity map, never a frozen
    # `a + b*pred`; see SP_IP_CALIBRATION_K.
    #
    # NEW RECORD FAMILY, and the argument is not the usual one. On decision
    # equivalence alone this would SHARE with v11 on the v10 precedent: over
    # the 254 ledger rows carrying both phases it flips 1 lean (0.4%), with
    # mean |d net| 0.00067 against a median |xw_net| of 0.01694 (4.0%) -- the
    # same order as the reweight that earned v9/v10 a shared line.
    #
    # It is isolated anyway because the usual argument is a COST-BENEFIT one
    # and the cost is zero here. Sharing exists to avoid resetting a graded
    # sample for a change that decides the same games; v11 has no graded rows
    # yet, so there is no sample to protect and a clean line for a lean-path
    # change is free. Do not generalise this into "bump means isolate" -- if
    # v11 had graded rows the 1-flip measurement above would have argued for
    # sharing, and it is recorded here so a later reader can see which way the
    # evidence pointed independently of what the reset cost.
    "xw+plat_consol_v12": ("xw+plat_consol_v12",),
}
RECORD_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "RECORD_TAGS", ",".join(_RECORD_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
    ).split(",") if t.strip()
)
# Scale-compatibility: which tags measure |xw_net| on the SAME units. Separate
# from RECORD_TAGS (win-loss compatibility) -- the two equivalence relations
# disagree. v5 introduced empirical-Bayes shrinkage that halved the delta scale
# (median |xw_net| .036 -> .018); v6 inherits v5's shrinkage, so v5 and v6 share
# units even though v6 is a fresh RECORD family. v7 changes the estimated K and
# therefore starts its own scale family. v8 fixes K at 100 for both pools,
# widening the shrunk distribution and starting another scale family. v10
# only re-weights a convex combination of the two existing phases, so the
# delta distribution is unchanged in units. Pre-v5
# tags are on the older, unshrunk scale. The default isolates any unlisted tag
# (never mixes scales); list a tag here only to pool it with others that share
# its units.
#
# v8 and v9 share units. v9 differs from v8 by exactly one term --
# v9 - v8 == -(1 - q) * platoon_delta_sp * P_BP / L -- because the two forms
# expand to the same starter phase and differ only in which lineup composite
# meets the bullpen. Measured on the 2026-07-28 slate that term has sd 0.00103
# against a matchup dispersion of 0.01662 (6.2%), and compare_v8_v9.py reports
# 0 lean flips over 24 eligible games. That is a units-preserving change, so
# the two pool for delta-scale ranking. RECORD_TAGS is a separate question and
# is deliberately left isolated.
# v11 SHARES the v8/v9/v10 scale family, argued rather than inherited, and it
# is the one decision here that is argued rather than measured -- no-lookahead
# means a past slate cannot be rebuilt to check it. Same metric, same K=100,
# same batter and starter target, so the three ways v11 differs from v10 are
# each scale-preserving by a precedent already set in this file:
#
#   * exposure-centred platoon offsets: total gap 0.020 -> 0.021. The identical
#     change under wOBA (v1 -> v2) moved median |net| 0.7% and shared a scale
#     family on that basis.
#   * abstention: only ever nulls an edge, never rescales a surviving one --
#     the v4/v5 precedent, where pooled p33/p80 was identical with and without
#     the abstained rows.
#   * relief-pool shrink target: the ONLY piece without a direct precedent.
#     It shifts both sides' bullpen aggregates by roughly the same amount, so
#     it largely cancels in a difference -- the same argument the batter
#     re-centring diagnostic makes above ("a uniform re-centring cancels",
#     0 flips over 99 rows). It does not cancel exactly, because each
#     reliever's prior weight K/(n+K) varies with his BF, but that residual is
#     second-order and at K=100 the prior carries far less of the rate than
#     the K=400 build this target was designed against.
#
# THE FALSIFIER, since this is argued: compare median |xw_net| on the first
# graded v11 rows against the v9/v10 pool. If it has moved materially, split
# v11 into its own scale family -- do not leave the pooling in place because
# it was convenient. Sharing is what gives the strength cutoffs a real pool on
# day one instead of restarting them for the ninth time.
#
# RUN, AND NOW CLOSED -- it does not fire, and it cannot be run any finer.
# v11 produced no rows at all (see the v8 precedent), so the check fell to
# v12. Three reads of the row-pooled form, against the v9/v10 pool's 0.01859
# over 99 rows:
#     15 v12 rows   gap -0.00009  CI [-0.0081, +0.0183]
#     38 v12 rows   gap -0.00526  CI [-0.0117, +0.0002]   <- looked like firing
#    196 v12 rows   gap -0.00085  CI [-0.0062, +0.0023]
# The middle read was slate composition, exactly as it was called at the time:
# at 5x the sample the point estimate regressed from -0.0053 back to -0.0009.
# The slate-pooled form asked for in CLAUDE.md agrees and does not fire either
# -- median of per-slate medians, 15 v12 slates against 7, gap -0.00309 at
# 1.48 se, CI [-0.0063, +0.0021].
#
# Do NOT re-read this at larger n. The statistic is underpowered BY
# CONSTRUCTION, not merely today: between-slate sd of the per-slate median is
# 0.00457, 7x the mean |d net| of 0.00067 that the IP calibration can actually
# produce, so separating the mechanism from slate noise needs ~730 slates per
# arm (~68 seasons). What it does have power for is a GROSS units shift --
# ~11 slates per arm at 0.00542, the largest single-row move the change can
# make, and both arms clear that -- so it has said the only thing it was ever
# able to say. The quantity that actually bounds the units change is the
# direct paired measurement in the v12 entry below (mean |d net| 0.00067 over
# 254 rows), which needs no ledger accumulation at all. Prefer that form for
# any future shared-scale argument: measure the change against itself, not
# two families against each other through the games that happened to be
# played.
_SCALE_FAMILIES = {
    "xw+plat_consol_v5": ("xw+plat_consol_v5", "xw+plat_consol_v6"),
    "xw+plat_consol_v6": ("xw+plat_consol_v5", "xw+plat_consol_v6"),
    "xw+plat_consol_v7": ("xw+plat_consol_v7",),
    # v11 is listed in all four tuples to keep the relation symmetric. Only the
    # v11 entry can ever be selected (SCALE_TAGS derives from the CURRENT tag),
    # so the other three are documentation of the equivalence class, not live
    # behaviour -- same reason the v8 entries stay despite matching zero rows.
    "xw+plat_consol_v8": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                          "xw+plat_consol_v10", "xw+plat_consol_v11",
                          "xw+plat_consol_v12"),
    "xw+plat_consol_v9": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                          "xw+plat_consol_v10", "xw+plat_consol_v11",
                          "xw+plat_consol_v12"),
    "xw+plat_consol_v10": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                           "xw+plat_consol_v10", "xw+plat_consol_v11",
                           "xw+plat_consol_v12"),
    "xw+plat_consol_v11": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                           "xw+plat_consol_v10", "xw+plat_consol_v11",
                           "xw+plat_consol_v12"),
    # v12 SHARES the scale, and unlike v11's half this one is measured. The
    # calibration only moves `q`, a convex weight between the same two phases,
    # so the delta keeps its units by construction -- v10's argument exactly.
    # Measured on the 254 ledger rows with both phases: mean |d net| 0.00067,
    # max 0.00542, against a median |xw_net| of 0.01694. That is 4.0% of a
    # typical magnitude, inside the 1.6-3.4% band that earned v9/v10 a shared
    # scale, and one lean flips.
    "xw+plat_consol_v12": ("xw+plat_consol_v8", "xw+plat_consol_v9",
                           "xw+plat_consol_v10", "xw+plat_consol_v11",
                           "xw+plat_consol_v12"),
    # wOBA has a different sampling distribution from xwOBA, so it cannot
    # share magnitude cutoffs with any xwOBA lineage.
    # v2 changes the centre of the starter platoon prior but retains observed
    # wOBA units and essentially the same total gap (0.021 versus 0.020). On the
    # 2026-08-04 projected slate median |net| moves only 0.7% (.016169 ->
    # .016282), so v1 and v2 remain compatible for magnitude/strength
    # calibration.
    "woba+plat_consol_v1": ("woba+plat_consol_v1", "woba+plat_consol_v2"),
    "woba+plat_consol_v2": ("woba+plat_consol_v1", "woba+plat_consol_v2"),
    # v3 also starts a NEW SCALE FAMILY, and this half is not a judgement call.
    # Quadrupling K shrinks every input toward its prior, so |xw_net| compresses
    # -- a median batter at ~400 PA keeps 400/500 = 80% of his deviation at
    # K=100 and 400/800 = 50% at K=400. The delta is measured in the same wOBA
    # units but on a materially narrower spread, which is exactly what a scale
    # family exists to separate. Pooling v2 and v3 magnitudes would rank two
    # distributions against one set of cutoffs.
    "woba+plat_consol_v3": ("woba+plat_consol_v3",),
    # v4 starts a NEW SCALE FAMILY, and like v3's this half is not a judgement
    # call either -- it is the same argument with the sign reversed. Shrinking
    # toward a personal prior instead of one shared centre WIDENS the delta
    # distribution: under v3 two players with equal samples were pulled to the
    # same number and their difference compressed toward zero, while under v4
    # each keeps his own centre, so |xw_net| disperses. Same wOBA units,
    # materially wider spread, which is exactly what a scale family separates.
    # Pooling v3 and v4 magnitudes would rank two distributions against one set
    # of cutoffs and relabel every game in the crossed band.
    # v4 and v5 SHARE a scale family, argued rather than inherited. v5 removes
    # rows from the magnitude pool; it does not rescale the ones that remain --
    # every surviving |xw_net| is bit-identical to what v4 produced, because the
    # abstention only ever nulls an edge. The only question a shared scale can
    # get wrong is whether dropping the abstained games shifts the quantiles the
    # cutoffs are read from, and measured on the ledger it does not: pooled
    # p33/p80 over those 185 rows is 0.0090 / 0.0283 with the 6 prior-only games
    # and 0.0090 / 0.0283 without them, and within the v9/v10 family alone
    # 0.0127 / 0.0343 both ways -- unchanged to four decimals. That is not
    # vacuous: dropping a *random* 6 of the 185 holds both to four decimals only
    # ~19% of the time. Isolating would reset a pool that was only 11 rows deep
    # at the bump, for a filter that moves no cutoff.
    # This is the v6 precedent (a new prediction family inheriting a scale).
    "woba+plat_consol_v4": ("woba+plat_consol_v4", "woba+plat_consol_v5"),
    "woba+plat_consol_v5": ("woba+plat_consol_v4", "woba+plat_consol_v5"),
    # The mixed-metric delta has its own units. Historical recognition does
    # not make it compatible with the active full-wOBA scale.
    "split+plat_consol_v1": ("split+plat_consol_v1",),
}
SCALE_TAGS = tuple(
    t.strip() for t in os.environ.get(
        "SCALE_TAGS", ",".join(_SCALE_FAMILIES.get(MODEL_TAG, (MODEL_TAG,)))
    ).split(",") if t.strip()
)
STATCAST_SELECTIONS = ["pa", "k_percent", "bb_percent", MODEL_RATE_SOURCE_COL,
                       "xba", "xslg",
                       "exit_velocity_avg", "launch_angle_avg", "hard_hit_percent"]

# Cache namespace for the custom leaderboard. A constant rather than a literal
# at the fetch site for one reason: it is keyed to the SELECTION SET, so it must
# change whenever STATCAST_SELECTIONS does -- otherwise a same-day cache written
# under an older column set is reused and the requested column comes back
# missing. Making both a pair of module constants is what lets shadow_metric.py
# repoint the metric and its cache together; a literal at the fetch site could
# only be repointed by patching the function.
STATCAST_CACHE_NS = "custom_xwoba_v1"

# Suffix of the per-slate dump, `leans_<date>_<DUMP_SUFFIX>.csv`. One constant
# rather than a literal at the write site, because grade_leans ingests a fixed
# set of globs (`leans_*_xw.csv`, `_split.csv`, `_woba.csv`) and a dump written
# under a suffix outside that set is silently never ledgered -- a slate's rows
# lost with no error, and unrecoverable afterwards without lookahead. The
# suffix names the metric inside: xwOBA rows land in `_xw.csv` again under v11,
# and the wOBA lineage's `_woba.csv` dumps stay exactly where they are.
# test_primary_dump_name_still_ingests pins this against grade_leans' own
# globs, reading the suffix from here rather than repeating it.
DUMP_SUFFIX = "xw"

# Prefix for a dump whose every row was captured AFTER its game started, i.e.
# a reconstruction of a slate that is already over rather than a record of what
# the model saw before first pitch.
#
# The daily 4:17am ET grading pass runs at 00:17 ET, before the 3am rollover,
# so SLATE_DATE still names yesterday and the build re-runs that whole slate
# against today's Savant leaderboard. Written under the live name, that
# unconditionally replaced the pregame dump: measured across every committed
# dump carrying `snapshot_utc`, EVERY past slate's dump was a post-first-pitch
# rebuild. The ledger was never affected -- `grade_leans.ingest` admits a row
# only when `lock_status == "pregame"` and rejected those rebuilds wholesale --
# but the dump is the sole per-slate record of what the model actually saw, so
# every dump-based measurement was reading data the pregame build never had.
# The shadow arm has no ledger behind it at all: for that arm the dump IS the
# record, and git history was the only surviving pregame copy.
#
# A rebuild is worth keeping -- it is a legitimate later view of the same
# slate, and `compare_v8_v9` and the probes read it happily -- so it is renamed
# rather than skipped. The prefix, not a suffix, is what makes it safe: the
# grader globs `leans_*_xw.csv`, which matches any leans-prefixed name ending
# `_xw.csv`, so `leans_<date>_xw_rebuild.csv` would have been ingested as a
# real pending row. Same decision, and the same reason, as SHADOW_PREFIX.
#
# What this does NOT fix: a mid-slate build still overwrites the live dump for
# games that have already started, because a slate with any pregame game left
# is not a rebuild. Those dumps are a mixture, honestly labelled per row by
# `lock_status`, and shrinking that window means merging dumps rather than
# naming them -- a different change with a different risk.
REBUILD_PREFIX = "rebuild"


def dump_is_post_hoc(frame, snapshot_utc):
    """True when no row in `frame` was captured before its scheduled start.

    Decided from the rows rather than from the clock. A date comparison against
    the ET calendar would have to re-derive the rollover hour, agree with it
    across DST, and still be wrong whenever SLATE_DATE is overridden to rebuild
    an old slate by hand -- while the rows already carry both timestamps and
    answer the question directly.

    Returns False whenever the answer is not knowable (no rows, no start times,
    an unparseable stamp), which is the current behaviour: an unknown dump
    keeps the live name and stays visible to every existing glob. The failure
    this guards is overwriting a pregame record, and a dump wrongly named
    `rebuild_` would be a second way to lose one.
    """
    if frame is None or getattr(frame, "empty", True):
        return False
    if "scheduled_start_utc" not in getattr(frame, "columns", ()):
        return False
    snap = pd.to_datetime(snapshot_utc, utc=True, errors="coerce")
    if pd.isna(snap):
        return False
    starts = pd.to_datetime(frame["scheduled_start_utc"], utc=True,
                            errors="coerce").dropna()
    return bool(len(starts)) and bool((starts <= snap).all())


def dump_path(prefix, date, suffix, post_hoc):
    """`data/[rebuild_]<prefix>_<date>_<suffix>.csv`."""
    name = f"{prefix}_{date}_{suffix}.csv"
    if post_hoc:
        name = f"{REBUILD_PREFIX}_{name}"
    return os.path.join(DATA_DIR, name)

# Batted-ball direction/tendency rates with true league-wide anchors.
BATTED_RATE_COLS_FOR_BASELINE = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]

# Use all Savant-listed hitters for platoon priors instead of today's loaded lineups.
# Costs ~10-15 batched StatsAPI calls (~+5s) but removes slate-dependent shrinkage baselines.
FULL_LEAGUE_PLATOON_BASELINES = True
MIN_LEAGUE_BASELINE_PA = 1
RECENT_STARTS = 5

# Sequential starter/bullpen model. A probable pitcher is treated as an opener when either his
# recent starts are repeatedly short or his recent usage is overwhelmingly
# short relief work. The latter catches relievers named for a one-off start
# before they have accumulated two official starts. v6 estimates the probable
# pitcher's innings, then fills the balance of nine innings with a filtered
# bullpen aggregate. Long/bulk relievers remain in the bullpen pool, but the
# model deliberately does not guess which individual bulk arm will follow.
#
# The bullpen pool uses season role stats to exclude rotation starters:
# <=35% of appearances started and <=3.0 IP per appearance. Each included
# reliever's xwOBA is shrunk independently before the pool is weighted by his
# estimated relief batters faced. If the role-filtered aggregate is unavailable,
# the starter phase is still displayed but the full-game lean is suppressed.
# There is no whole-staff opener fallback: it could include rotation arms and
# double-count the probable pitcher.
# The platoon lens is left alone -- an opener's tiny vL/vR split already fails
# the reliability gate, so that lens abstains on its own. Each affected side is
# flagged with its detection reason and pitching basis so it stays auditable.
OPENER_MAX_AVG_IP = 3.0     # avg IP/start below this => treat probable as an opener
OPENER_MIN_STARTS = 2       # need a repeated pattern, not one rain-shortened start
BULLPEN_MIN_PITCHERS = 3    # minimum role-filtered relievers for an aggregate
# Below this many resolved arms the relief centre is estimated off too few
# pitchers to beat the league baseline it replaces, so v3 falls back to v2's
# target rather than shrinking toward a noisy one. A full slate resolves ~200.
RELIEF_PRIOR_MIN_ARMS = 40
OPENER_ROLE_APPEARANCES = 10
OPENER_ROLE_MIN_APPEARANCES = 3
OPENER_ROLE_MAX_STARTS = 1
OPENER_ROLE_MIN_RELIEF_SHARE = 0.70
OPENER_ROLE_MAX_MEDIAN_IP = 2.0
OPENER_ROLE_MAX_P80_PITCHES = 55.0
OPENER_STRETCHED_MIN_IP = 4.0
OPENER_STRETCHED_MIN_PITCHES = 65
GAME_INNINGS = 9.0
SP_IP_PRIOR = 5.2
SP_IP_PRIOR_STARTS = 3.0
SP_IP_MIN = 3.0
SP_IP_MAX = 7.0
OPENER_IP_DEFAULT = 1.5
OPENER_IP_MIN = 1.0 / 3.0
RP_MAX_START_SHARE = 0.35
RP_MAX_IP_PER_APPEARANCE = 3.0


def slate_date_now():
    """Slate date in ET with a ~3am rollover. Env override wins."""
    override = os.environ.get("SLATE_DATE", "").strip()
    if override:
        return override
    now_et = datetime.now(ET)
    day = now_et.date() if now_et.hour >= ROLLOVER_HOUR else (now_et - timedelta(days=1)).date()
    return day.isoformat()


SLATE_DATE = slate_date_now()
SEASON = int(SLATE_DATE[:4])

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/csv,*/*",
})


def log(*a):
    print(*a, flush=True)


# ============================================================
# CELL 1 -- FETCH (render-free / no browser)
# ============================================================
def _get_json(url, params=None, tries=4, headers=None):
    """`headers` overrides the session defaults for one call.

    Used only by the ESPN odds path, which needs the honest agent string in
    fetch_headers.py to get past that edge's bot rule. Deliberately per-call
    rather than a change to `session.headers`: Savant and StatsAPI are
    answering the browser-ish UA today, and a build that loses the Savant
    leaderboards costs a slate's rows permanently.
    """
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=30, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.6 * (2 ** k))


def cached_csv(url, cache_name, tries=4):
    """Fetch a Savant CSV leaderboard, caching the raw text once per ET day."""
    path = os.path.join(CACHE_DIR, f"savant_cache_{cache_name}_{SLATE_DATE}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            text = r.text
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
            except Exception:
                pass
            return pd.read_csv(io.StringIO(text))
        except Exception as e:  # noqa: BLE001
            last = e
            if k == tries - 1:
                break
            time.sleep(0.8 * (2 ** k))
    raise last


_team_logo_cache = {}
_logo_cdn_down = False   # tripped once a fetch exhausts retries; skips further
                         # network so an unreachable CDN can't stall the build


def _fetch_logo_svg(url, cache_name, tries=2):
    """Fetch one SVG logo, cached to CACHE_DIR (cap marks are effectively
    static). Returns the raw SVG text, or None on any failure / non-SVG body so
    the caller can fall back to the text abbreviation. A cached file is always
    served; once the CDN is known-unreachable this run, uncached teams skip the
    network entirely rather than each re-paying the retry budget."""
    global _logo_cdn_down
    path = os.path.join(CACHE_DIR, f"logo_{cache_name}.svg")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read() or None
        except Exception:  # noqa: BLE001
            pass
    if _logo_cdn_down:
        return None
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            text = r.text
            if "<svg" not in text.lower():
                return None
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:  # noqa: BLE001
                pass
            return text
        except Exception as e:  # noqa: BLE001
            last = e
            if k == tries - 1:
                break
            time.sleep(0.4 * (2 ** k))
    _logo_cdn_down = True
    log(f"  team logo fetch failed ({cache_name}); disabling logo fetch this run: {last!r}")
    return None


def _svg_data_uri(svg_text):
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def team_logo_uris(team_id):
    """(<light-bg data-uri>, <dark-bg data-uri>) cap logos for a team, or
    (None, None) when unavailable. Both cap variants share the StatsAPI team id;
    each is fetched once per build and memoized, then embedded so the page stays
    self-contained. If only one variant resolves it is used for both; if neither
    does the card keeps its text abbreviation."""
    if not USE_TEAM_LOGOS or team_id is None:
        return None, None
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return None, None
    if tid in _team_logo_cache:
        return _team_logo_cache[tid]
    light = _fetch_logo_svg(f"{LOGO_CDN}/team-cap-on-light/{tid}.svg", f"cap_light_{tid}")
    dark = _fetch_logo_svg(f"{LOGO_CDN}/team-cap-on-dark/{tid}.svg", f"cap_dark_{tid}")
    if not light and not dark:
        _team_logo_cache[tid] = (None, None)
        return None, None
    light, dark = (light or dark), (dark or light)
    uris = (_svg_data_uri(light), _svg_data_uri(dark))
    _team_logo_cache[tid] = uris
    return uris


def get_slate(slate_date, sport_id=1):
    data = _get_json("https://statsapi.mlb.com/api/v1/schedule",
                     {"sportId": sport_id, "date": slate_date,
                      "hydrate": "probablePitcher,team,venue,linescore"})
    rows = []
    for db in data.get("dates", []):
        od = db.get("date", slate_date)
        for g in db.get("games", []):
            a, h = g["teams"]["away"], g["teams"]["home"]
            linescore = g.get("linescore") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": od,
                "game_datetime_utc": g.get("gameDate"),
                "game_number": g.get("gameNumber"),
                "double_header": g.get("doubleHeader"),
                "matchup": f'{a["team"]["name"]} @ {h["team"]["name"]}',
                "away_team": a["team"]["name"], "home_team": h["team"]["name"],
                "away_team_id": a["team"]["id"], "home_team_id": h["team"]["id"],
                "away_abbrev": a["team"].get("abbreviation"),
                "home_abbrev": h["team"].get("abbreviation"),
                "away_probable_pitcher": a.get("probablePitcher", {}).get("fullName"),
                "home_probable_pitcher": h.get("probablePitcher", {}).get("fullName"),
                "away_probable_pitcher_id": a.get("probablePitcher", {}).get("id"),
                "home_probable_pitcher_id": h.get("probablePitcher", {}).get("id"),
                "abstract_state": g.get("status", {}).get("abstractGameState"),
                "status": g.get("status", {}).get("detailedState"),
                "away_score": a.get("score"),
                "home_score": h.get("score"),
                "current_inning": linescore.get("currentInning"),
                "current_inning_ordinal": linescore.get("currentInningOrdinal"),
                "inning_half": linescore.get("inningHalf"),
                "is_top_inning": linescore.get("isTopInning"),
                # Base-out state. `offense` carries a player object per occupied
                # base and omits the key entirely when the base is empty, so
                # presence is the signal. Absent on unstarted games -> all False
                # / no outs, and the card hides the indicator.
                "on_first": bool((linescore.get("offense") or {}).get("first")),
                "on_second": bool((linescore.get("offense") or {}).get("second")),
                "on_third": bool((linescore.get("offense") or {}).get("third")),
                "outs": linescore.get("outs"),
                "venue": g.get("venue", {}).get("name"),
                "savant_preview_url": f'https://baseballsavant.mlb.com/preview?game_pk={g.get("gamePk")}&game_date={od}',
            })
    return pd.DataFrame(rows)


def load_stat_lookups(player_type):
    """player_type in {'batter','pitcher'} -> (stat dict, bbprofile dict, custom df)."""
    sel = ",".join(STATCAST_SELECTIONS)
    cust = cached_csv(
        f"https://baseballsavant.mlb.com/leaderboard/custom?year={SEASON}"
        f"&type={player_type}&min=1&selections={sel}&csv=true",
        # Namespace is keyed to the selection set (see STATCAST_CACHE_NS): a
        # same-day cache written under a different column set must never be
        # reused, or the requested rate comes back missing.
        f"{STATCAST_CACHE_NS}_{player_type}")
    bb = cached_csv(
        f"https://baseballsavant.mlb.com/leaderboard/batted-ball?type={player_type}"
        f"&year={SEASON}&min=1&csv=true",
        f"battedball_{player_type}")

    # Map observed Savant wOBA into the established primary-rate schema. The
    # internal name stays xwOBA solely so old dumps, ledger readers, and audit
    # tooling remain compatible; MODEL_RATE_SOURCE_COL is the fetched value.
    REN_STAT = {MODEL_RATE_SOURCE_COL: MODEL_RATE_INTERNAL_COL,
                "xba": "xBA", "xslg": "xSLG", "exit_velocity_avg": "EV",
                "launch_angle_avg": "LA°", "hard_hit_percent": "Hard Hit%",
                "k_percent": "K%", "bb_percent": "BB%", "pa": "PA"}
    stat = {}
    for _, r in cust.iterrows():
        pid = int(r["player_id"])
        stat[pid] = {REN_STAT[k]: r.get(k) for k in REN_STAT if k in cust.columns}

    BB_REN = {"gb_rate": "GB%", "fb_rate": "FB%", "ld_rate": "LD%", "pu_rate": "PU%",
              "pull_rate": "Pull%", "straight_rate": "Straight%", "oppo_rate": "Oppo%"}
    bbprofile = {}
    for _, r in bb.iterrows():
        pid = int(r["id"])
        prof = {v: (r[k] * 100 if pd.notna(r[k]) else np.nan) for k, v in BB_REN.items() if k in bb.columns}
        bbprofile[pid] = prof
        stat.setdefault(pid, {})
        stat[pid]["BBE"] = r.get("bbe")
    return stat, bbprofile, cust


def load_pitcher_xera():
    """player_id -> Statcast xERA (expected ERA), cached once per ET day.

    xERA lives on Savant's *expected_statistics* leaderboard, not the custom
    board, so it needs its own CSV. `min=1` (not the default `min=q`) keeps
    non-qualified probables in. Strictly best-effort and display-only: any fetch
    failure or a missing/renamed column yields an empty map and the card's xERA
    cell renders an em-dash — a Savant outage never fails the build here."""
    try:
        df = cached_csv(
            "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
            f"?type=pitcher&year={SEASON}&position=&team=&filterType=bip&min=1&csv=true",
            "xstats_pitcher")
    except Exception as e:  # noqa: BLE001
        log(f"  xERA leaderboard unavailable ({e!r}) -> em-dashes")
        return {}
    # Export labels the expected-ERA column `xera`; tolerate the `est_era`
    # variant used elsewhere on the expected-stats board.
    col = next((c for c in ("xera", "est_era") if c in df.columns), None)
    if col is None:
        log(f"  xERA column absent from expected-stats board (cols={list(df.columns)[:12]}...) -> em-dashes")
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            pid = int(r["player_id"])
        except (TypeError, ValueError):
            continue
        v = _f(r.get(col))
        if v is not None:
            out[pid] = v
    log(f"  xERA leaderboard: {len(out)} pitchers (col '{col}')")
    return out


def load_people(ids):
    ids = [int(i) for i in ids if pd.notna(i)]
    info = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        data = _get_json("https://statsapi.mlb.com/api/v1/people",
                         {"personIds": ",".join(map(str, chunk))})
        for p in data.get("people", []):
            info[p["id"]] = {
                "name": p.get("fullName"),
                "pos": (p.get("primaryPosition", {}) or {}).get("abbreviation"),
                "bats": (p.get("batSide", {}) or {}).get("code"),
                "throws": (p.get("pitchHand", {}) or {}).get("code"),
            }
    return info


def _parse_rate(x):
    try:
        return float(str(x))
    except Exception:
        return np.nan


def load_splits(ids, group):
    """vL/vR splits + season overall via batched hydrate. group in {'hitting','pitching'}."""
    ids = [int(i) for i in ids if pd.notna(i)]
    pa_field = "plateAppearances" if group == "hitting" else "battersFaced"
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        hyd = f"stats(group=[{group}],type=[statSplits,season],sitCodes=[vl,vr],season={SEASON})"
        data = _get_json("https://statsapi.mlb.com/api/v1/people",
                         {"personIds": ",".join(map(str, chunk)), "hydrate": hyd})
        for p in data.get("people", []):
            rec = {"L": {}, "R": {}, "overall": {}}
            for blk in p.get("stats", []):
                tname = (blk.get("type", {}) or {}).get("displayName")
                for sk in blk.get("splits", []):
                    code = sk.get("split", {}).get("code")
                    st = sk.get("stat", {})
                    val = {"ops": _parse_rate(st.get("ops")),
                           "obp": _parse_rate(st.get("obp")),
                           "slg": _parse_rate(st.get("slg")),
                           "era": _parse_rate(st.get("era")),
                           "pa": int(st.get(pa_field) or st.get("plateAppearances") or 0),
                           "k": st.get("strikeOuts"), "bb": st.get("baseOnBalls")}
                    if tname == "statSplits" and code in ("vl", "vr"):
                        rec["L" if code == "vl" else "R"] = val
                    elif tname == "season" and code is None:
                        rec["overall"] = {"ops": val["ops"], "pa": val["pa"],
                                          "era": val["era"]}
            out[p["id"]] = rec
        time.sleep(REQUEST_DELAY)
    return out


def _innings_to_outs(value):
    """Convert MLB's baseball-notation innings (for example, 5.2) to outs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = str(value).strip()
    if not s:
        return 0
    try:
        whole, dot, frac = s.partition(".")
        extra = int(frac[:1] or 0) if dot else 0
        return max(0, int(whole) * 3 + min(max(extra, 0), 2))
    except (TypeError, ValueError):
        return 0


def load_recent_start_era(ids, limit=RECENT_STARTS, before_date=None):
    """Build recent-start and recent-role profiles strictly before a date.

    ``before_date`` is injectable for historical replay.  The live build keeps
    the existing behaviour by defaulting to ``SLATE_DATE``.
    """
    cutoff = str(before_date or SLATE_DATE)[:10]
    out = {}
    for pid in sorted({int(i) for i in ids if pd.notna(i)}):
        try:
            data = _get_json(
                f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
                {"stats": "gameLog", "group": "pitching", "season": SEASON,
                 "gameType": "R"})
        except Exception as e:  # noqa: BLE001
            log(f"  recent-start ERA unavailable for {pid}: {e!r}")
            continue

        appearances = []
        for blk in data.get("stats", []):
            for sk in blk.get("splits", []):
                st = sk.get("stat", {}) or {}
                # Never let the current slate's start leak into a pregame metric.
                game_date = str(sk.get("date") or
                                (sk.get("game", {}) or {}).get("gameDate") or "")[:10]
                if not game_date or game_date >= cutoff:
                    continue
                try:
                    er = float(st.get("earnedRuns") or 0)
                except (TypeError, ValueError):
                    continue
                game_pk = (sk.get("game", {}) or {}).get("gamePk") or 0
                try:
                    pitches = float(st.get("numberOfPitches"))
                except (TypeError, ValueError):
                    pitches = np.nan
                appearances.append({
                    "date": game_date,
                    "game_pk": int(game_pk),
                    "outs": _innings_to_outs(st.get("inningsPitched")),
                    "earned_runs": er,
                    "is_start": int(st.get("gamesStarted") or 0) >= 1,
                    "pitches": pitches,
                })

        appearances.sort(key=lambda x: (x["date"], x["game_pk"]), reverse=True)
        all_starts = [a for a in appearances if a["is_start"]]
        starts = all_starts[:limit]
        outs = sum(a["outs"] for a in starts)
        earned_runs = sum(a["earned_runs"] for a in starts)
        season_start_outs = sum(a["outs"] for a in all_starts)
        role_apps = appearances[:OPENER_ROLE_APPEARANCES]
        pitch_counts = [a["pitches"] for a in role_apps if pd.notna(a["pitches"])]
        recent_starts = sum(a["is_start"] for a in role_apps)
        stretched = sum(
            a["outs"] >= int(OPENER_STRETCHED_MIN_IP * 3)
            or (pd.notna(a["pitches"]) and a["pitches"] >= OPENER_STRETCHED_MIN_PITCHES)
            for a in role_apps
        )
        out[pid] = {
            "era": round(earned_runs * 27.0 / outs, 2) if outs > 0 else np.nan,
            "starts": len(starts),
            # Average innings per recent start -- the opener signal. Openers are
            # credited a game started but pitch ~1 inning, so this runs low.
            "avg_ip": (outs / 3.0 / len(starts)) if starts else np.nan,
            "season_starts": len(all_starts),
            "season_avg_ip": (
                season_start_outs / 3.0 / len(all_starts)
                if all_starts else np.nan
            ),
            # The last ten appearances describe the pitcher's current role. A
            # short-relief profile lets us recognize a newly named opener even
            # when he has only one prior official start.
            "appearances": len(role_apps),
            "recent_starts": recent_starts,
            "relief_share": (
                (len(role_apps) - recent_starts) / len(role_apps)
                if role_apps else np.nan
            ),
            "median_ip": (
                float(np.median([a["outs"] for a in role_apps])) / 3.0
                if role_apps else np.nan
            ),
            "p80_pitches": (
                float(np.percentile(pitch_counts, 80))
                if pitch_counts else np.nan
            ),
            "pitch_count_appearances": len(pitch_counts),
            "stretched_appearances": stretched,
        }
        time.sleep(REQUEST_DELAY)
    return out


def load_league_era(start_date=None, end_date=None):
    """MLB pitching ERA, optionally bounded for point-in-time replay."""
    endpoint = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "byDateRange" if end_date else "season",
        "group": "pitching", "season": SEASON,
        "sportIds": SPORT_ID, "gameType": "R", "playerPool": "ALL",
        "limit": 5000,
    }
    if start_date:
        params["startDate"] = str(start_date)[:10]
    if end_date:
        params["endDate"] = str(end_date)[:10]
    data = _get_json(endpoint, params)
    outs = earned_runs = 0
    for blk in data.get("stats", []):
        for sk in blk.get("splits", []):
            st = sk.get("stat", {}) or {}
            outs += _innings_to_outs(st.get("inningsPitched"))
            try:
                earned_runs += int(st.get("earnedRuns") or 0)
            except (TypeError, ValueError):
                pass
    return round(earned_runs * 27.0 / outs, 2) if outs > 0 else np.nan


_gf_cache = {}


def gf_lineups(game_pk):
    """Raw Savant posted lineup ids. Cached per game_pk."""
    if game_pk in _gf_cache:
        return _gf_cache[game_pk]
    try:
        gf = _get_json(f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}", tries=2)
        out = ([int(x) for x in gf.get("away_lineup", []) or []],
               [int(x) for x in gf.get("home_lineup", []) or []])
    except Exception:
        out = ([], [])
    _gf_cache[game_pk] = out
    return out


_roster_cache = {}


def roster_lineup(team_id, batter_stat):
    """Projected lineup pool = active-roster position players, ranked by season PA."""
    if team_id in _roster_cache:
        ids = _roster_cache[team_id]
    else:
        data = _get_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                         {"rosterType": "active"})
        ids = [r["person"]["id"] for r in data.get("roster", [])
               if (r.get("position", {}) or {}).get("abbreviation") != "P"]
        _roster_cache[team_id] = ids
    return sorted([int(i) for i in ids if int(i) in batter_stat],
                  key=lambda i: (batter_stat[i].get("PA") or 0), reverse=True)


def team_batter_xwoba(team_id, batter_stat, league_fallback=np.nan):
    """PA-weighted active model rate for Savant-listed hitters on one team.

    Used only when a posted hitter is absent from the Savant leaderboard.
    The league prior is a last resort when no rostered hitter has a usable
    wOBA/PA pair.
    """
    vals, weights = [], []
    for pid in roster_lineup(team_id, batter_stat):
        stat = batter_stat.get(pid) or {}
        xw = pd.to_numeric(stat.get("xwOBA"), errors="coerce")
        pa = pd.to_numeric(stat.get("PA"), errors="coerce")
        if pd.notna(xw) and pd.notna(pa) and float(pa) > 0:
            vals.append(float(xw))
            weights.append(float(pa))
    if vals:
        return float(np.average(vals, weights=weights))
    fallback = pd.to_numeric(league_fallback, errors="coerce")
    return float(fallback) if pd.notna(fallback) else np.nan


def fill_lineup_from_roster(team_id, posted_ids, batter_stat, target_size=LINEUP_SIZE):
    """Keep posted hitters in order; fill only missing slots by roster PA."""
    resolved, seen = [], set()
    for pid in posted_ids or []:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            return resolved[:target_size]
    for pid in roster_lineup(team_id, batter_stat):
        if pid not in seen:
            resolved.append(pid); seen.add(pid)
        if len(resolved) >= target_size:
            break
    return resolved[:target_size]


def resolve_lineup(game_pk, side, team_id, batter_stat, return_meta=False,
                   league_xwoba=np.nan):
    """Resolve the posted order first, then fill only genuinely open slots.

    A posted player missing from the season Savant leaderboard is retained and
    assigned the active team's PA-weighted xwOBA. That team aggregate is marked
    so the downstream player-level shrinkage step does not shrink it again.
    """
    away_ids, home_ids = gf_lineups(game_pk)
    raw = away_ids if side == "away" else home_ids
    posted, seen = [], set()
    for pid in raw:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid not in seen:
            posted.append(pid); seen.add(pid)
        if len(posted) >= LINEUP_SIZE:
            break

    missing, backfilled = [], []
    for pid in posted:
        stat = batter_stat.get(pid) or {}
        if bool(stat.get(MODEL_RATE_TEAM_BACKFILL_COL)):
            backfilled.append(pid)
        elif pd.isna(pd.to_numeric(stat.get("xwOBA"), errors="coerce")):
            missing.append(pid)
            backfilled.append(pid)
    if missing:
        team_xwoba = team_batter_xwoba(team_id, batter_stat, league_xwoba)
        for pid in missing:
            batter_stat[pid] = {
                **(batter_stat.get(pid) or {}),
                "xwOBA": team_xwoba,
                "PA": 0.0,
                "BBE": 0.0,
                MODEL_RATE_TEAM_BACKFILL_COL: True,
            }

    resolved = fill_lineup_from_roster(team_id, posted, batter_stat, LINEUP_SIZE)
    posted_count = min(len(posted), LINEUP_SIZE)
    filled_count = max(0, len(resolved) - posted_count)
    status = ("posted" if posted_count >= LINEUP_SIZE
              else "partial_filled" if posted_count > 0 else "projected")
    meta = {"status": status, "projected": status != "posted",
            "posted_count": int(posted_count), "filled_count": int(filled_count),
            "resolved_count": int(len(resolved)),
            "savant_backfill_count": int(len(backfilled))}
    return (resolved, meta) if return_meta else (resolved, meta["projected"])


STAT_COLS = ["BBE", "LA°", "EV", "Hard Hit%", "xwOBA", "xBA", "xSLG", "K%", "BB%"]
BB_COLS = ["GB%", "FB%", "LD%", "PU%", "Pull%", "Straight%", "Oppo%"]

_pitcher_roster_cache = {}
_team_pitcher_role_cache = {}
_ROSTER_UNSET = object()


def pitcher_roster(team_id, roster_date=None):
    """Active-roster pitcher ids, optionally on a historical date."""
    team_id = int(team_id)
    # Preserve the original integer cache key for live callers/tests; dated
    # replay entries live in a separate tuple namespace.
    key = ((team_id, str(roster_date)[:10]) if roster_date else team_id)
    if key in _pitcher_roster_cache:
        return _pitcher_roster_cache[key]
    params = {"rosterType": "active"}
    if roster_date:
        params["date"] = str(roster_date)[:10]
    data = _get_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                     params)
    ids = [int(r["person"]["id"]) for r in data.get("roster", [])
           if (r.get("position", {}) or {}).get("abbreviation") == "P"]
    _pitcher_roster_cache[key] = ids
    return ids


def load_team_pitcher_roles(team_id, start_date=None, end_date=None):
    """Workload roles for one club in a single StatsAPI call.

    Returns active/used pitchers keyed by player id with appearances, starts,
    start share, innings per appearance, and batters faced. Optional date
    bounds make the same parser safe for point-in-time replay. These fields are
    used only to separate the rotation from the relief pool; no specific bulk
    follower is projected.
    """
    team_id = int(team_id)
    # Preserve the original integer cache key for live callers/tests; dated
    # replay entries live in a separate tuple namespace.
    key = ((team_id, str(start_date)[:10] if start_date else None,
            str(end_date)[:10]) if end_date else team_id)
    if key in _team_pitcher_role_cache:
        return _team_pitcher_role_cache[key]
    params = {
        "stats": "byDateRange" if end_date else "season",
        "group": "pitching",
        "season": SEASON,
        "sportIds": SPORT_ID,
        "teamId": team_id,
        "gameType": "R",
        "playerPool": "ALL",
        "limit": 1000,
    }
    if start_date:
        params["startDate"] = str(start_date)[:10]
    if end_date:
        params["endDate"] = str(end_date)[:10]
    data = _get_json("https://statsapi.mlb.com/api/v1/stats", params)
    out = {}
    for blk in data.get("stats", []):
        for sk in blk.get("splits", []):
            who = sk.get("player") or sk.get("person") or {}
            pid = who.get("id")
            if pid is None:
                continue
            st = sk.get("stat", {}) or {}
            try:
                apps = int(st.get("gamesPitched") or st.get("gamesPlayed") or 0)
                starts = int(st.get("gamesStarted") or 0)
                bf = float(st.get("battersFaced") or 0)
            except (TypeError, ValueError):
                continue
            outs = _innings_to_outs(st.get("inningsPitched"))
            out[int(pid)] = {
                "appearances": apps,
                "starts": starts,
                "start_share": starts / apps if apps > 0 else np.nan,
                "avg_ip_per_appearance": outs / 3.0 / apps if apps > 0 else np.nan,
                "batters_faced": bf,
            }
    _team_pitcher_role_cache[key] = out
    return out


def relief_pitcher_ids(team_id, role_stats, probable_pid=None,
                       roster_ids=_ROSTER_UNSET):
    """Active-roster pitchers whose season workload is primarily relief.

    The loose three-inning ceiling intentionally keeps long/bulk relievers in
    the pooled estimate. It excludes rotation arms, and the current probable
    is removed so his starter/opener innings are not counted twice.
    """
    probable_pid = int(probable_pid) if probable_pid is not None else None
    out = []
    # ``None`` is an explicit unavailable historical roster and must stay
    # empty; only an omitted argument may fall back to the live roster call.
    ids = (pitcher_roster(team_id)
           if roster_ids is _ROSTER_UNSET else (roster_ids or []))
    for pid in ids:
        if pid == probable_pid:
            continue
        role = (role_stats or {}).get(pid) or {}
        apps = int(role.get("appearances") or 0)
        share = role.get("start_share")
        avg_ip = role.get("avg_ip_per_appearance")
        if (
            apps > 0
            and share is not None
            and pd.notna(share)
            and float(share) <= RP_MAX_START_SHARE
            and avg_ip is not None
            and pd.notna(avg_ip)
            and float(avg_ip) <= RP_MAX_IP_PER_APPEARANCE
        ):
            out.append(pid)
    return out


def bf_per_ip(role):
    """Measured batters faced per inning from a pitcher's season role line.

    The blend weight has to be a share of plate appearances, not of innings,
    because xwOBA is a per-PA rate. BF/IP is what converts one to the other and
    it is observed, not modelled: both terms come from the same season role
    call, so a pitcher who allows more baserunners shows a higher rate without
    any on-base proxy standing in for it. Returns None when either term is
    missing, which degrades the caller to the innings share.
    """
    role = role or {}
    bf = _f(role.get("batters_faced"))
    apps = _f(role.get("appearances"))
    avg_ip = _f(role.get("avg_ip_per_appearance"))
    if bf is None or apps is None or avg_ip is None:
        return None
    season_ip = float(avg_ip) * float(apps)
    if bf <= 0 or season_ip <= 0:
        return None
    return float(bf) / season_ip


def relief_pool_prior(team_ids, pitcher_stat, roles_by_team,
                      rosters_by_team=None):
    """Unweighted centre of tonight's relief pools, or None.

    v3 shrinks relievers toward this instead of toward the league PA-weighted
    batter rate. Three things make that the right target and each of them was
    measured before the change, not argued:

      * The league batter centre is the centre of no subpool. Measured over the
        223 role-filtered arms on 30 active rosters, the relief pool sits
        0.0102 BELOW it. At K=400 the prior supplies ~70% of a median
        reliever's published rate, so that offset lands almost intact in every
        bullpen number -- which is precisely why K and this target had to move
        in the same commit. Raising K against the old target would have doubled
        a known bias instead of removing it.
      * UNWEIGHTED, not usage-weighted, because empirical Bayes wants the
        centre of the population a member is drawn from. The pool's own
        usage-weighted centre -- what `bullpen_xwoba_aggregate` averages -- sits
        0.0116 the OTHER side of the unweighted centre, because the good arms
        get the innings. Shrinking toward it would over-correct and land
        further from the truth than the shipped target was. The obvious fix is
        worse than the status quo; this is the non-obvious one.
      * Derived every build from the same Savant rates the pool is built from,
        never a literal. The measured 0.3032 is on a StatsAPI reconstruction
        and is not even on Savant's scale; only the gaps between pools
        transfer. A frozen centre here would be the constants-frozen-from-data
        entry in CLAUDE.md with a fresh date on it.

    Membership is `relief_pitcher_ids` -- the same filter the aggregate uses, so
    the prior and the pool cannot describe different populations. No probable is
    excluded: the aggregate drops tonight's starter per game so his innings are
    not double counted, but for a population centre that is per-slate noise, and
    dropping one arm per club would bias toward the pool's weaker members.

    Returns None when too few arms resolve, and the caller then falls back to
    the league baseline -- the v2 behaviour -- rather than shrinking toward a
    centre estimated off a handful of pitchers.
    """
    rates, seen = [], set()
    for tid in team_ids:
        roles = (roles_by_team or {}).get(tid)
        if not roles:
            continue
        try:
            if rosters_by_team is None:
                ids = relief_pitcher_ids(tid, roles, probable_pid=None)
            else:
                ids = relief_pitcher_ids(
                    tid, roles, probable_pid=None,
                    roster_ids=rosters_by_team.get(tid),
                )
        except Exception:  # noqa: BLE001
            continue
        for pid in ids:
            if pid in seen:
                continue
            seen.add(pid)
            v = _f((pitcher_stat.get(pid) or {}).get(XWOBA_SHRINK_COL))
            if v is not None and pd.notna(v):
                rates.append(float(v))
    if len(rates) < RELIEF_PRIOR_MIN_ARMS:
        return None
    return float(np.mean(rates))


def bullpen_xwoba_aggregate(team_id, probable_pid, pitcher_stat, role_stats,
                            prior, shrink_k, roster_ids=_ROSTER_UNSET):
    """Role-filtered bullpen xwOBA with per-pitcher empirical-Bayes shrinkage.

    Talent is shrunk by each pitcher's Savant BF before aggregation. Usage
    weights approximate relief BF as season BF × (1 - start share), preventing
    a swingman's starter work from receiving full bullpen weight.
    """
    weighted = []
    for pid in relief_pitcher_ids(
        team_id, role_stats, probable_pid, roster_ids=roster_ids
    ):
        stat = pitcher_stat.get(pid) or {}
        bf = _f(stat.get("PA"))
        xw = _f(stat.get("xwOBA"))
        role = (role_stats or {}).get(pid) or {}
        share = _f(role.get("start_share"))
        role_bf = _f(role.get("batters_faced"))
        if bf is None or bf <= 0 or xw is None:
            continue
        relief_share = max(0.0, 1.0 - (share or 0.0))
        # Role BF is team-specific, whereas Savant PA follows the pitcher
        # across trades. Use team BF for expected club usage when available;
        # retain Savant PA only as the talent-shrinkage sample size.
        usage_bf = (role_bf if role_bf is not None and role_bf > 0 else bf) * relief_share
        if usage_bf <= 0:
            continue
        # v4: each arm toward his own history, and toward the relief pool's
        # centre (`prior`, the v3 target) when he has none.
        shrunk = _shrink_one(xw, bf, player_prior_one(pid, prior), shrink_k)
        if shrunk is not None and pd.notna(shrunk):
            weighted.append((pid, usage_bf, float(shrunk), bf_per_ip(role)))
    if len(weighted) < BULLPEN_MIN_PITCHERS:
        return None
    total_bf = sum(w for _, w, _, _ in weighted)
    # Pool BF/IP on the same pitchers and the same usage weights that produced
    # the pool's xwOBA, so the rate describes the innings this bullpen is
    # actually expected to throw. Pitchers missing a role line are skipped
    # here without affecting the xwOBA aggregate above.
    rated = [(w, r) for _, w, _, r in weighted if r is not None]
    rate_bf = sum(w for w, _ in rated)
    return {
        "xwOBA": sum(w * x for _, w, x, _ in weighted) / total_bf,
        "pitcher_count": len(weighted),
        "relief_bf": total_bf,
        "bf_per_ip": (sum(w * r for w, r in rated) / rate_bf) if rate_bf > 0 else None,
        "pitcher_ids": tuple(pid for pid, _, _, _ in weighted),
    }


def expected_pitcher_ip(profile, classification=None):
    """Pregame expected workload from pre-slate starts/current role.

    Normal starters blend the last-five start average with their season start
    average (or a fixed league prior when unavailable). Openers use the signal
    that classified them: repeated short starts use start IP, while a reliever
    spot start uses median recent appearance length. Values are bounded to
    plausible role ranges and never use the current slate appearance.
    """
    profile = profile or {}
    classification = classification or {}

    def finite(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    recent = finite(profile.get("avg_ip"))
    season = finite(profile.get("season_avg_ip"))
    if classification:
        if classification.get("reason") == "repeated_short_starts":
            estimate = recent
        else:
            estimate = finite(profile.get("median_ip"))
            if estimate is None:
                estimate = recent
        estimate = OPENER_IP_DEFAULT if estimate is None else estimate
        return float(np.clip(estimate, OPENER_IP_MIN, OPENER_MAX_AVG_IP))

    prior = season if season is not None else SP_IP_PRIOR
    n_recent = int(profile.get("starts") or 0)
    if recent is None or n_recent <= 0:
        estimate = prior
    else:
        estimate = (
            n_recent * recent + SP_IP_PRIOR_STARTS * prior
        ) / (n_recent + SP_IP_PRIOR_STARTS)
    return float(np.clip(estimate, SP_IP_MIN, SP_IP_MAX))


# --- v12: the expected-IP estimator is calibrated against its own actuals ---
# `expected_pitcher_ip` is OVER-DISPERSED: regressing actual starter IP on the
# pregame estimate over the backfilled box scores gives a slope well under 1,
# so starters predicted short go longer than predicted and starters predicted
# deep go shorter. Bias is small and not significant; this is a spread problem.
# It matters because `q` -- the starter's share of plate appearances -- is a
# function of this number, so every over-dispersed estimate tilts the phase
# weight behind `mx`.
#
# THE FIX IS A PER-BUILD DERIVATION, NOT A FITTED LITERAL, and that is the
# whole design. Freezing today's `a + b*pred` into the source would be the
# constants-frozen-from-data anti-pattern with a fresh date on it -- the slope
# is plausibly seasonal (pitch counts climb early, clubs get cautious in
# September), so a September build must fit September's slope. The ledger
# accumulates actuals on every grading pass, so the fit arrives on its own.
#
# The correction is shrunk toward the IDENTITY MAP by sample size:
#
#     cal(p) = w*(a + b*p) + (1 - w)*p,   w = n / (n + SP_IP_CALIBRATION_K)
#
# so n = 0 returns `p` exactly. That is the threshold-cliff lesson applied
# again: there is no `if n >= N` gate to cross, no day on which every starter's
# workload estimate jumps, and no branch to keep in sync. An empty ledger and a
# full one are the two ends of one expression.
#
# K = 50 side-games, chosen by walk-forward benchmark rather than by taste:
# fitting on every prior slate and scoring the next one, over 586 side-games /
# 23 slates, the calibration beats no-calibration by +4.1% (K=25), +4.0%
# (K=50), +3.8% (K=100) in out-of-sample IP MSE, against +3.4% for K=0 (no
# shrinkage at all -- shrinkage does earn its place early). Bootstrapping over
# slates, K=50 is the argmin most often (131/400) and every candidate's 95% CI
# excludes zero. DO NOT READ THE SECOND DIGIT: the curve is flat from 10 to
# 100 and 25 vs 50 vs 100 are not distinguishable on this sample. What is
# distinguishable is calibrated from uncalibrated.
SP_IP_CALIBRATION_K = 50.0
_sp_ip_cal_state = {}


def sp_ip_calibration(led=None):
    """(intercept, slope, n, w) mapping the raw IP estimate onto its actuals.

    Fitted on `expected_sp_ip_raw` where present and on `expected_sp_ip` where
    it is not. That fallback is what makes the fit self-sustaining rather than
    self-consuming: from v12 on the dump carries the RAW estimate beside the
    published one, and the fit must always read the raw column, or each build
    would calibrate an already-calibrated number and compound the correction.
    Rows written before v12 hold a raw value under the published name, so they
    join the same fit with no special case.

    Uses only rows whose games have finished -- an actual IP exists only after
    the box score is backfilled -- so there is no lookahead here: tonight's
    slate is calibrated against completed games, never against itself.

    Returns None when the ledger is missing, unreadable or has fewer than two
    usable pairs, and the caller then publishes the raw estimate unchanged.
    """
    # The live build calls with ``led=None`` and may reuse one fit during the
    # process. A walk-forward replay passes an explicit prior-only frame for
    # every slate; caching that fit would leak the first replay date into every
    # later one (or, worse, reuse a production-ledger fit).
    use_process_cache = led is None
    if use_process_cache and "fit" in _sp_ip_cal_state:
        return _sp_ip_cal_state["fit"]
    fit = None
    led = load_ledger_df() if led is None else led
    if led is not None:
        def col(name):
            """Numeric column as a Series, all-NaN when the column is absent.

            `pd.to_numeric(led.get(missing))` returns a scalar NaN rather than
            a Series, which is how a missing column reaches `.where` and raises
            on an attribute it does not have. Pre-v12 ledgers have no
            `expected_sp_ip_raw_*` at all, so that path is the normal one on
            the first build after this ships, not an edge case.
            """
            if name not in getattr(led, "columns", ()):
                return pd.Series(np.nan, index=led.index, dtype="float64")
            return pd.to_numeric(led[name], errors="coerce")

        pred, act = [], []
        for side in ("away", "home"):
            raw = col(f"expected_sp_ip_raw_{side}")
            pub = col(f"expected_sp_ip_{side}")
            p = raw.where(raw.notna(), pub)
            a = col(f"act_sp_ip_{side}")
            m = p.notna() & a.notna()
            pred.append(p[m].to_numpy())
            act.append(a[m].to_numpy())
        if pred:
            p = np.concatenate(pred)
            a = np.concatenate(act)
            if p.size >= 2 and float(np.std(p)) > 1e-9:
                slope, intercept = np.polyfit(p, a, 1)
                n = int(p.size)
                w = n / (n + SP_IP_CALIBRATION_K)
                fit = (float(intercept), float(slope), n, float(w))
    if use_process_cache:
        _sp_ip_cal_state["fit"] = fit
    return fit


def calibrate_sp_ip(raw, fit=None):
    """Apply `sp_ip_calibration` to one raw estimate. Identity without a fit."""
    v = _f(raw)
    if v is None:
        return raw
    fit = sp_ip_calibration() if fit is None else fit
    if not fit:
        return v
    intercept, slope, _n, w = fit
    return float(w * (intercept + slope * v) + (1.0 - w) * v)


def build_pitching_plans(slate_df, recent_profiles, pitcher_stat,
                         league_baseline, provider=None,
                         calibration_history=None):
    """Expected-IP + role-filtered bullpen plan for every probable-pitcher side."""
    plans = {}
    classifications = opener_classifications(recent_profiles)
    league_prior = _f((league_baseline or {}).get("xwOBA"))
    shrink_k = XWOBA_SHRINK_K
    # Fitted once per build, from completed games only.
    ip_cal = sp_ip_calibration(calibration_history)
    if ip_cal:
        _a, _b, _n, _w = ip_cal
        log(f"  expected-IP calibration: act = {_a:+.3f} + {_b:.3f}*pred "
            f"(n={_n}, weight {_w:.3f}); 1.000 = already calibrated")
    else:
        log("  expected-IP calibration: no usable actuals -> raw estimate")
    roles_by_team = {}
    rosters_by_team = {}
    bullpen_by_pair = {}

    # Role lines first, for every club on the slate: the relief centre has to
    # exist before any member is shrunk toward it, and it is a population
    # statistic over all of tonight's pools rather than one club's.
    for _, g in slate_df.iterrows():
        for side in ("away", "home"):
            tid = g.get(f"{side}_team_id")
            if pd.isna(tid):
                continue
            tid = int(tid)
            if tid not in roles_by_team:
                try:
                    roles_by_team[tid] = (
                        provider.load_team_pitcher_roles(tid)
                        if provider is not None else load_team_pitcher_roles(tid)
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  bullpen roles unavailable for team {tid}: {e!r}")
                    roles_by_team[tid] = None
            if tid not in rosters_by_team:
                try:
                    rosters_by_team[tid] = (
                        provider.pitcher_roster(tid)
                        if provider is not None else pitcher_roster(tid)
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  pitcher roster unavailable for team {tid}: {e!r}")
                    rosters_by_team[tid] = None
    relief_prior = relief_pool_prior(
        list(roles_by_team), pitcher_stat, roles_by_team,
        rosters_by_team=rosters_by_team,
    )
    prior = relief_prior if relief_prior is not None else league_prior
    log(f"  relief shrink target: "
        + (f"{relief_prior:.4f} (pool centre, {len(roles_by_team)} clubs)"
           if relief_prior is not None
           else f"{league_prior} (league baseline; too few arms resolved)")
        + f" | league baseline {league_prior}")

    for _, g in slate_df.iterrows():
        for side in ("away", "home"):
            pid = g.get(f"{side}_probable_pitcher_id")
            tid = g.get(f"{side}_team_id")
            if pd.isna(pid) or pd.isna(tid):
                continue
            pid, tid = int(pid), int(tid)
            classification = classifications.get(pid)
            # Raw first, then calibrated. Both are carried: the published value
            # drives `q`, and the raw one is what every future fit regresses
            # against (see sp_ip_calibration).
            raw_ip = expected_pitcher_ip(
                (recent_profiles or {}).get(pid), classification
            )
            expected_ip = calibrate_sp_ip(raw_ip, ip_cal)

            pair = (tid, pid)
            if pair not in bullpen_by_pair:
                try:
                    bullpen_by_pair[pair] = bullpen_xwoba_aggregate(
                        tid, pid, pitcher_stat, roles_by_team.get(tid), prior,
                        shrink_k, roster_ids=rosters_by_team.get(tid),
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  bullpen aggregate failed for team {tid}: {e!r}")
                    bullpen_by_pair[pair] = None
            bullpen = bullpen_by_pair[pair]
            if bullpen is None:
                basis = "starter_only_no_fullgame_lean"
            elif classification:
                basis = "opener_bullpen_sequential"
            else:
                basis = "starter_bullpen_sequential"

            key = (int(g["game_pk"]), side)
            plans[key] = {
                "opener": bool(classification),
                "opener_reason": (classification or {}).get("reason"),
                "opener_confidence": (classification or {}).get("confidence"),
                "pitching_basis": basis,
                "expected_sp_ip": expected_ip,
                "expected_sp_ip_raw": raw_ip,
                "bullpen_xwOBA": None if bullpen is None else bullpen.get("xwOBA"),
                "bullpen_pitchers": None if bullpen is None else bullpen.get("pitcher_count"),
                "bullpen_relief_bf": None if bullpen is None else bullpen.get("relief_bf"),
                "sp_bf_per_ip": bf_per_ip((roles_by_team.get(tid) or {}).get(pid)),
                "bp_bf_per_ip": None if bullpen is None else bullpen.get("bf_per_ip"),
            }
            if bullpen is not None:
                log(
                    f"  sequential pitching: pid={pid} team={tid} basis={basis} "
                    f"exp_ip={expected_ip:.2f} rp_{MODEL_RATE_LABEL}="
                    f"{bullpen.get('xwOBA'):.3f} "
                    f"(n_rp={bullpen.get('pitcher_count')})"
                )
            else:
                log(
                    f"  full-game lean unavailable: pid={pid} team={tid}; "
                    "displaying starter phase only"
                )
    return plans


def opener_classifications(recent_start_era, max_avg_ip=OPENER_MAX_AVG_IP,
                           min_starts=OPENER_MIN_STARTS):
    """Classify probable pitchers whose pre-slate workload indicates an opener.

    Repeated short starts are high-confidence evidence. A short-relief role is
    a medium-confidence inference used for relievers making a first/second
    spot start. Recent starter-length work suppresses that inference.
    """
    out = {}
    for pid, m in (recent_start_era or {}).items():
        m = m or {}
        if int(m.get("stretched_appearances") or 0) > 0:
            continue
        ip = (m or {}).get("avg_ip")
        if (ip is not None and pd.notna(ip) and ip < max_avg_ip
                and int(m.get("starts") or 0) >= min_starts):
            out[int(pid)] = {
                "reason": "repeated_short_starts",
                "confidence": "high",
            }
            continue

        relief_share = m.get("relief_share")
        median_ip = m.get("median_ip")
        p80_pitches = m.get("p80_pitches")
        if (
            int(m.get("appearances") or 0) >= OPENER_ROLE_MIN_APPEARANCES
            and int(m.get("recent_starts") or 0) <= OPENER_ROLE_MAX_STARTS
            and relief_share is not None
            and pd.notna(relief_share)
            and relief_share >= OPENER_ROLE_MIN_RELIEF_SHARE
            and median_ip is not None
            and pd.notna(median_ip)
            and median_ip <= OPENER_ROLE_MAX_MEDIAN_IP
            and p80_pitches is not None
            and pd.notna(p80_pitches)
            and p80_pitches <= OPENER_ROLE_MAX_P80_PITCHES
            and int(m.get("pitch_count_appearances") or 0)
            >= OPENER_ROLE_MIN_APPEARANCES
        ):
            out[int(pid)] = {
                "reason": "reliever_spot_start",
                "confidence": "medium",
            }
    return out


def opener_pids(recent_start_era, max_avg_ip=OPENER_MAX_AVG_IP,
                min_starts=OPENER_MIN_STARTS):
    """Backward-compatible set of probable-pitcher ids classified as openers."""
    return set(opener_classifications(
        recent_start_era, max_avg_ip=max_avg_ip, min_starts=min_starts
    ))


def _meta(row):
    return {k: row[k] for k in ["game_pk", "game_date", "game_datetime_utc",
                                "matchup", "away_team", "home_team",
                                "away_probable_pitcher", "home_probable_pitcher",
                                "savant_preview_url"]}


def build_tables(slate, lineups, batter_stat, pitcher_stat, batter_bb, pitcher_bb, people):
    pit_rows, bb_rows = [], []

    for _, g in slate.iterrows():
        meta = _meta(g)
        asp, hsp = g["away_probable_pitcher_id"], g["home_probable_pitcher_id"]
        away_lu, home_lu = lineups[g["game_pk"]]

        # sp_side names the block's pitcher side (structural, from the call
        # site) and is_sp marks the probable-pitcher row; both are carried on
        # every row so segment_pitcher_blocks assigns sides without name matching.
        def pitcher_row(pid, tidx, table, sp_side):
            if pd.isna(pid):
                return
            pid = int(pid)
            bio = people.get(pid, {})
            nm = bio.get("name") or f"pitcher_{pid}"
            src = (pitcher_stat if table == "stat" else pitcher_bb).get(pid, {})
            base = {**meta, "table_index": tidx, "Name": nm,
                    "sp_side": sp_side, "is_sp": True}
            if table == "stat":
                pit_rows.append({**base, "table_type": "pitchers", "Pos.": "P",
                                 **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
                                 MODEL_RATE_TEAM_BACKFILL_COL: False,
                                 "player_id": pid, "bats": bio.get("bats"),
                                 "throws": bio.get("throws")})
            else:
                bbe = (pitcher_stat.get(pid, {}) or {}).get("BBE")
                bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": "P",
                                **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                "player_id": pid, "bats": bio.get("bats"),
                                "throws": bio.get("throws")})

        def hitter_rows(lu, tidx, table, sp_side):
            for slot, pid in enumerate(lu, start=1):
                pid = int(pid)
                bio = people.get(pid, {})
                nm = bio.get("name") or f"batter_{pid}"
                pos = bio.get("pos") or "DH"
                if pos == "P":
                    pos = "DH"
                src = (batter_stat if table == "stat" else batter_bb).get(pid, {})
                base = {**meta, "table_index": tidx, "Name": nm,
                        BATTING_ORDER_COL: slot, "sp_side": sp_side, "is_sp": False}
                if table == "stat":
                    backfill_value = src.get(MODEL_RATE_TEAM_BACKFILL_COL)
                    pit_rows.append({**base, "table_type": "pitchers", "Pos.": pos,
                                     **{c: src.get(c) for c in STAT_COLS}, "PA": src.get("PA"),
                                     MODEL_RATE_TEAM_BACKFILL_COL:
                                         (bool(backfill_value)
                                          if pd.notna(backfill_value) else False),
                                     "player_id": pid, "bats": bio.get("bats"),
                                     "throws": bio.get("throws")})
                else:
                    bbe = (batter_stat.get(pid, {}) or {}).get("BBE")
                    bb_rows.append({**base, "table_type": "batted_ball_profile", "Pos.": pos,
                                    **{c: src.get(c) for c in BB_COLS}, "BBE": bbe,
                                    "player_id": pid, "bats": bio.get("bats"),
                                    "throws": bio.get("throws")})

        # Block 1 = away SP vs home lineup; block 2 = home SP vs away lineup.
        pitcher_row(asp, 1, "stat", "away"); hitter_rows(home_lu, 1, "stat", "away")
        pitcher_row(hsp, 2, "stat", "home"); hitter_rows(away_lu, 2, "stat", "home")
        pitcher_row(asp, 1, "bb", "away"); hitter_rows(home_lu, 1, "bb", "away")
        pitcher_row(hsp, 2, "bb", "home"); hitter_rows(away_lu, 2, "bb", "home")

    META = ["game_pk", "game_date", "game_datetime_utc", "matchup", "away_team", "home_team",
            "away_probable_pitcher", "home_probable_pitcher", "savant_preview_url",
            "table_type", "table_index", "Name"]
    pdf = pd.DataFrame(pit_rows)
    if not pdf.empty:
        pdf = pdf[META + ["Pos.", BATTING_ORDER_COL] + STAT_COLS
                  + ["PA", MODEL_RATE_TEAM_BACKFILL_COL, "player_id", "bats", "throws",
                     "sp_side", "is_sp"]]
    bdf = pd.DataFrame(bb_rows)
    if not bdf.empty:
        bdf = bdf[META + ["Pos.", BATTING_ORDER_COL] + BB_COLS
                  + ["BBE", "player_id", "bats", "throws", "sp_side", "is_sp"]]
    return pdf, bdf


def compute_league_baseline(batter_cust):
    _LB_MAP = {MODEL_RATE_SOURCE_COL: MODEL_RATE_INTERNAL_COL,
               "xba": "xBA", "xslg": "xSLG", "k_percent": "K%", "bb_percent": "BB%",
               "exit_velocity_avg": "EV", "launch_angle_avg": "LA°", "hard_hit_percent": "Hard Hit%"}
    league_baseline = {}
    _w = pd.to_numeric(batter_cust.get("pa"), errors="coerce")
    for raw, disp in _LB_MAP.items():
        if raw in batter_cust.columns:
            v = pd.to_numeric(batter_cust[raw], errors="coerce")
            m = v.notna() & _w.notna() & (_w > 0)
            league_baseline[disp] = (
                float(np.average(v[m], weights=_w[m])) if m.any() else np.nan
            )
    return league_baseline


# --- shrinkage-target population diagnostic (log-only) ----------------------
# XWOBA_SHRINK_K regresses batters, probable starters and every reliever toward
# ONE target: the PA-weighted league rate compute_league_baseline just built.
# The justification on record (docs/build_logic_validation.md) is "league-wide
# xwOBA-allowed equals league xwOBA". That identity is true and does not
# license the target. Empirical Bayes wants the centre of the pool each
# estimate is drawn from, and a pooled mean is not the centre of any of its
# subpools -- the identity only guarantees they average back to it.
#
# v7 already named the batter half of this (MATCHUP_SITE.md: low-PA members
# have a different unweighted centre than the PA-weighted target), corrected
# the K *estimator* for the resulting between-centre offset, and deliberately
# left the target alone. v8 then retired that estimator, so nothing in the
# current build measures the offset at all and the target has never been
# revisited.
#
# So this logs the centres instead of arguing them from identities. Two
# distinct gaps are separated on purpose:
#
#   weighted vs unweighted -- within one pool, how much the PA/BF weighting
#     lifts the centre above the player-level centre EB actually wants. Good
#     hitters take more PA and good relievers get more usage, so the weighted
#     figure flatters both pools relative to their members.
#   pool vs target -- whether that pool belongs at the shared target at all.
#
# `bias` multiplies the second gap by the mean prior weight to give the average
# displacement the shared target imposes on a member of the pool: what the
# wrong centre is actually worth, in rate points, before anything cancels
# between the two sides of a game.
#
# Log-only by construction. Nothing here returns into a lean, a dump, a ledger
# row or the page, so it carries no MODEL_TAG implication and cannot fail the
# build -- the caller swallows its exceptions.
def _pool_moments(rate, weight, prior, k):
    """(n, weighted centre, unweighted centre, mean prior weight, bias) or None.

    `weight` is PA for batters and BF for pitchers -- the same sample size the
    shrinkage uses, so `mean_w` is literally the average share of a member's
    published estimate that is the target rather than the player.
    """
    rate = pd.to_numeric(pd.Series(rate).reset_index(drop=True), errors="coerce")
    weight = pd.to_numeric(pd.Series(weight).reset_index(drop=True), errors="coerce")
    m = rate.notna() & weight.notna() & (weight > 0)
    if not m.any():
        return None
    r, w = rate[m], weight[m]
    unweighted = float(r.mean())
    mean_w = float((k / (w + k)).mean())
    # `r_n` is why `weighted` and `unweighted` differ at all: it is the
    # correlation between a player's rate and how much he plays. It also
    # decides whether a prior that varies with n would beat a single constant
    # (see pool_shrink_target) -- that only pays once this is solidly
    # positive, so it is measured rather than assumed.
    r_n = None
    if m.sum() >= 3 and float(r.std()) > 0 and float(w.std()) > 0:
        r_n = float(np.corrcoef(r.to_numpy(), np.log(w.to_numpy()))[0, 1])
    return {
        "n": int(m.sum()),
        "weighted": float(np.average(r, weights=w)),
        "unweighted": unweighted,
        "mean_w": mean_w,
        "bias": mean_w * (unweighted - float(prior)),
        "r_n": r_n,
    }


def prior_population_centres(batter_cust, pitcher_cust, prior, k,
                             role_map=None, probable_ids=None):
    """Centres of every pool the shared shrinkage target is applied to.

    `role_map` is player id -> season role line (load_team_pitcher_roles), which
    build_pitching_plans has already fetched and cached for the slate's teams;
    passing it costs no extra request and is what splits the rotation from the
    relief pool. Without it the pitcher pools collapse to one undifferentiated
    row -- which is the current model's assumption, stated rather than hidden.

    Returns a list of (label, moments) with unusable pools dropped.
    """
    if prior is None or not np.isfinite(float(prior)):
        return []
    src = MODEL_RATE_SOURCE_COL
    out = []

    def add(label, cust, mask=None):
        cols = getattr(cust, "columns", [])
        if cust is None or src not in cols or "pa" not in cols:
            return
        rate, weight = cust[src], cust["pa"]
        if mask is not None:
            rate, weight = rate[mask], weight[mask]
        moments = _pool_moments(rate, weight, prior, k)
        if moments:
            out.append((label, moments))

    add("batters", batter_cust)
    # The sub-qualified tail is where the target does most of its work: these
    # are the call-ups and platoon bats whose published rate is mostly prior.
    games = (_est_team_games(pd.to_numeric(batter_cust.get("pa"), errors="coerce"))
             if batter_cust is not None else None)
    if games:
        pa = pd.to_numeric(batter_cust["pa"], errors="coerce")
        add(f"batters <{PCTILE_QUAL_BAT * games:.0f}PA", batter_cust,
            pa < PCTILE_QUAL_BAT * games)

    add("pitchers", pitcher_cust)
    if role_map and pitcher_cust is not None and "player_id" in pitcher_cust.columns:
        pid = pd.to_numeric(pitcher_cust["player_id"], errors="coerce")
        share = pid.map(lambda i: (role_map.get(int(i)) or {}).get("start_share")
                        if pd.notna(i) else None)
        ip_app = pid.map(lambda i: (role_map.get(int(i)) or {}).get("avg_ip_per_appearance")
                         if pd.notna(i) else None)
        share = pd.to_numeric(share, errors="coerce")
        ip_app = pd.to_numeric(ip_app, errors="coerce")
        # Same predicate relief_pitcher_ids uses, so the RP row describes the
        # pool bullpen_xwoba_aggregate actually shrinks -- not a proxy for it.
        add("  pitchers:RP", pitcher_cust,
            (share <= RP_MAX_START_SHARE) & (ip_app <= RP_MAX_IP_PER_APPEARANCE))
        add("  pitchers:SP", pitcher_cust, share > RP_MAX_START_SHARE)
    if probable_ids and pitcher_cust is not None and "player_id" in pitcher_cust.columns:
        pid = pd.to_numeric(pitcher_cust["player_id"], errors="coerce")
        # The only starters the model ever shrinks. A rotation-wide SP centre
        # can differ from this one; that difference is selection, not talent.
        add("  pitchers:SP(slate)", pitcher_cust,
            pid.isin({int(i) for i in probable_ids}))
    return out


def log_prior_population_centres(batter_cust, pitcher_cust, prior, k,
                                 role_map=None, probable_ids=None):
    rows = prior_population_centres(batter_cust, pitcher_cust, prior, k,
                                    role_map=role_map, probable_ids=probable_ids)
    if not rows:
        return
    log(f"  shrinkage target = {prior:.5f} (PA-weighted league {MODEL_RATE_LABEL}), "
        f"K={k:.0f}; pool centres:")
    log(f"    {'pool':<24}{'n':>6}{'wtd':>9}{'unwtd':>9}{'mean w':>9}"
        f"{'unwtd-tgt':>11}{'bias':>9}{'r(rate,lnN)':>13}")
    for label, m in rows:
        r_n = "     --" if m["r_n"] is None else f"{m['r_n']:>+7.3f}"
        log(f"    {label:<24}{m['n']:>6}{m['weighted']:>9.4f}{m['unweighted']:>9.4f}"
            f"{m['mean_w']:>9.3f}{m['unweighted'] - prior:>+11.4f}{m['bias']:>+9.4f}"
            f"{r_n:>13}")


class LiveDataProvider:
    """Default source layer for the production build.

    The wrapper is intentionally thin: it changes no live behaviour, but gives
    replay code one seam at which to substitute date-bounded inputs while all
    prediction functions below remain the single source of truth.
    """

    def __init__(self, slate_date=None):
        self.slate_date = str(slate_date or SLATE_DATE)[:10]

    def get_slate(self, slate_date, sport_id=SPORT_ID):
        return get_slate(slate_date, sport_id)

    def load_stat_lookups(self, player_type):
        return load_stat_lookups(player_type)

    def resolve_lineup(self, game_pk, side, team_id, batter_stat,
                       return_meta=False, league_xwoba=np.nan):
        return resolve_lineup(
            game_pk, side, team_id, batter_stat,
            return_meta=return_meta, league_xwoba=league_xwoba,
        )

    def load_recent_start_era(self, ids):
        return load_recent_start_era(ids, before_date=self.slate_date)

    def load_team_pitcher_roles(self, team_id):
        return load_team_pitcher_roles(team_id)

    def pitcher_roster(self, team_id):
        return pitcher_roster(team_id)

    def load_league_era(self):
        return load_league_era()

    def load_people(self, ids):
        return load_people(ids)

    def load_splits(self, ids, group):
        return load_splits(ids, group)

    def load_pitcher_xera(self):
        return load_pitcher_xera()


def fetch_all(slate_date, provider=None, calibration_history=None,
              include_platoon=True, write_audit=True):
    """Run the cell-1 fetch through a live or point-in-time provider."""
    provider = provider or LiveDataProvider(slate_date)
    log(f"Pulling slate for {slate_date} ...")
    slate_df = provider.get_slate(slate_date, SPORT_ID)
    log(f"Games: {len(slate_df)}")
    if slate_df.empty:
        return {"slate_df": slate_df, "empty": True}

    # Log game statuses (no filtering change) so postponed/suspended games are
    # visible in the build log rather than blending into the slate count.
    if "status" in slate_df.columns:
        log(f"  game statuses: {slate_df['status'].value_counts(dropna=False).to_dict()}")
        _NORMAL_STATUS = {"Scheduled", "Pre-Game", "Warmup", "In Progress",
                          "Final", "Game Over"}
        for _, gg in slate_df.iterrows():
            st = gg.get("status")
            if st not in _NORMAL_STATUS:
                log(f"  non-standard status {st!r}: {gg.get('matchup')} "
                    f"(game_pk={int(gg['game_pk'])})")

    log("Loading Savant leaderboards (cached once/day) ...")
    batter_stat, batter_bb, batter_cust = provider.load_stat_lookups("batter")
    pitcher_stat, pitcher_bb, pitcher_cust = provider.load_stat_lookups("pitcher")
    log(f"  batters: {len(batter_stat)} | pitchers: {len(pitcher_stat)}")

    league_baseline = compute_league_baseline(batter_cust)
    # Full-population batted-ball anchors so GB/FB/LD/PU/Pull/Straight/Oppo matchup
    # edges are not NaN or slate-dependent. Rates are already stored as percentages.
    for c in BATTED_RATE_COLS_FOR_BASELINE:
        vals, wts = [], []
        for pid, prof in batter_bb.items():
            v = prof.get(c)
            w = (batter_stat.get(pid, {}) or {}).get("BBE")
            if pd.notna(v) and pd.notna(w) and float(w) > 0:
                vals.append(float(v)); wts.append(float(w))
        league_baseline[c] = (
            float(np.average(vals, weights=wts)) if vals else np.nan
        )
    shown_baselines = {
        MODEL_RATE_LABEL: league_baseline.get(MODEL_RATE_INTERNAL_COL),
        **{k: league_baseline.get(k) for k in
           ["Hard Hit%", "K%", "EV", "GB%", "FB%", "Pull%"]},
    }
    log("  league baselines:", shown_baselines)

    # v8 uses one fixed pseudo-sample for every batter and pitcher xwOBA,
    # including each member of the bullpen pool. Keeping the value fixed makes
    # the shrinkage reproducible across builds and retains more of the observed
    # distribution's tails than the larger v7 per-pool estimates.
    if USE_XWOBA_SHRINK:
        prior = league_baseline.get("xwOBA")
        k_bat = k_pit = XWOBA_SHRINK_K
        log(f"  {MODEL_RATE_LABEL} shrink: prior={prior} | K_bat={k_bat:.0f} (fixed) "
            f"| K_pit={k_pit:.0f} (fixed)")

        # Display-only percentile reference distributions (qualified regulars),
        # stashed on league_baseline so the render layer can rank a player's
        # shrunk xwOBA without re-fetching the leaderboards. Team-games is
        # estimated once from the batter pool and shared by both qualifiers,
        # since pitchers accrue BF at a different per-game rate than hitters.
        g = _est_team_games(pd.to_numeric(batter_cust.get("pa"), errors="coerce")) if batter_cust is not None else None
        qual_bat = PCTILE_QUAL_BAT * g if g else None
        qual_pit = PCTILE_QUAL_PIT * g if g else None

        # The bars rank a player against a reference of qualified regulars, so
        # the target that regression runs toward has to be the batter pool's
        # own centre. Shrinking a 40-PA bat toward the PA-weighted league rate
        # lands him near the middle of a population he is not drawn from, and
        # the reference barely moves under the same target because its members
        # are all high-PA -- so the sub-qualified bats rank high for a reason
        # that is arithmetic, not hitting. The 2026-08-04 diagnostic measured
        # that gap at 23 points pool-wide and 45 in the sub-qualified tail.
        #
        # DISPLAY ONLY, and deliberately not the lean's target. Replaying the
        # 99 graded v9/v10 rows with the batter target moved flips 0 leans at
        # every plausible lineup weight (mean |d net| <= 0.0004 against a median
        # |xw_net| of 0.019) -- xw_net differences two lineups regressed toward
        # the same centre, so a uniform re-centring cancels. Moving the lean's
        # target would therefore bump MODEL_TAG, reset nothing, and shift every
        # published per-side edge off zero, since `edge = mx - L` only reads as
        # "vs league" while B and P are both centred on L. The bars are ranked
        # against a population; the edge is measured against the league. Those
        # are two different anchors and this is the one the bars need.
        #
        # The reference and the ranked value MUST share a target -- a rank
        # against a differently-regressed population is not a rank at all.
        prior_bat_display = pool_shrink_target(batter_cust) or prior
        ref_bat, qb = build_pctile_ref(batter_cust, prior_bat_display, k_bat, qual_bat)
        # Pitchers keep the league rate: the same diagnostic put slate probables
        # at -0.0016 from it and the BF-weighted relief pool at -0.0024, i.e.
        # already centred. Deriving a target for them would be motion, not fix.
        ref_pit, qp = build_pctile_ref(pitcher_cust, prior, k_pit, qual_pit)
        league_baseline["_pctile_ref_bat"] = ref_bat
        league_baseline["_pctile_ref_pit"] = ref_pit
        league_baseline["_pctile_prior_bat"] = prior_bat_display
        log(f"  batter percentile target: {prior_bat_display:.5f} "
            f"(unweighted pool centre) vs league {prior:.5f} "
            f"-> {prior_bat_display - prior:+.5f}")
        # K-BB% reference (pitchers), raw: K-BB stabilizes fast and the model
        # does not shrink it, so it is ranked at face value over the same
        # qualified pool.
        ref_kbb = None
        pcols = set(getattr(pitcher_cust, "columns", []))
        if pitcher_cust is not None and {"k_percent", "bb_percent"} <= pcols:
            kk = pd.to_numeric(pitcher_cust["k_percent"], errors="coerce")
            bbp = pd.to_numeric(pitcher_cust["bb_percent"], errors="coerce")
            pap = pd.to_numeric(pitcher_cust.get("pa"), errors="coerce")
            msk = kk.notna() & bbp.notna()
            if qual_pit and pap is not None:
                msk_q = msk & (pap >= qual_pit)
                if int(msk_q.sum()) >= PCTILE_MIN_POOL:
                    msk = msk_q
            ref_kbb = build_pctile_ref_raw((kk - bbp)[msk])
        league_baseline["_pctile_ref_kbb"] = ref_kbb
        log(f"  pctile refs: G~{g and round(g)} | bat n="
            f"{0 if ref_bat is None else ref_bat.size} (qual~{qb and round(qb)}) | pit n="
            f"{0 if ref_pit is None else ref_pit.size} (qual~{qp and round(qp)}) | kbb n="
            f"{0 if ref_kbb is None else ref_kbb.size}")

    log("Resolving lineups (gf -> posted/partial-fill/projected) ...")
    lineups, proj_flags, lineup_ids, prob_ids = {}, [], set(), set()
    for _, g in slate_df.iterrows():
        al, ai = provider.resolve_lineup(
            g["game_pk"], "away", g["away_team_id"], batter_stat,
            league_xwoba=league_baseline.get("xwOBA"), return_meta=True
        )
        hl, hi = provider.resolve_lineup(
            g["game_pk"], "home", g["home_team_id"], batter_stat,
            league_xwoba=league_baseline.get("xwOBA"), return_meta=True
        )
        lineups[g["game_pk"]] = (al, hl)
        proj_flags.append({
            "game_pk": g["game_pk"],
            "away_lineup_projected": ai["projected"],
            "home_lineup_projected": hi["projected"],
            "away_lineup_status": ai["status"],
            "home_lineup_status": hi["status"],
            "away_posted_count": ai["posted_count"],
            "home_posted_count": hi["posted_count"],
            "away_filled_count": ai["filled_count"],
            "home_filled_count": hi["filled_count"],
            "away_resolved_count": ai["resolved_count"],
            "home_resolved_count": hi["resolved_count"],
            "away_savant_backfill_count": ai["savant_backfill_count"],
            "home_savant_backfill_count": hi["savant_backfill_count"],
        })
        lineup_ids.update(al); lineup_ids.update(hl)
        for c in ("away_probable_pitcher_id", "home_probable_pitcher_id"):
            if pd.notna(g[c]):
                prob_ids.add(int(g[c]))
        # A NaN probable id drops that side's block downstream; log it so every
        # game on the slate is accounted for in the build log.
        if pd.isna(g["away_probable_pitcher_id"]) or pd.isna(g["home_probable_pitcher_id"]):
            log(f"  TBD probable: {g.get('matchup')} (game_pk={int(g['game_pk'])}) — "
                f"away={g.get('away_probable_pitcher') or 'TBD'}, "
                f"home={g.get('home_probable_pitcher') or 'TBD'}")
        time.sleep(REQUEST_DELAY)
    lineup_projection_df = pd.DataFrame(proj_flags)
    if write_audit:
        os.makedirs(DATA_DIR, exist_ok=True)
        lineup_projection_df.to_csv(
            os.path.join(DATA_DIR, "lineup_resolution_audit.csv"), index=False
        )

    log(f"Loading probable-pitcher ERA over the last {RECENT_STARTS} starts ...")
    recent_start_era = provider.load_recent_start_era(prob_ids)

    # Pitching plans are applied after the starter's own xwOBA has been shrunk
    # in build_matchup. The starter-handedness lineup composite is used only
    # for his expected innings; the neutral composite faces the bullpen.
    pitching_plans = build_pitching_plans(
        slate_df, recent_start_era, pitcher_stat, league_baseline,
        provider=provider, calibration_history=calibration_history,
    )

    # Log-only; runs here because build_pitching_plans has just populated
    # _team_pitcher_role_cache, so the rotation/relief split costs no request.
    # Best-effort in both directions: a missing role line drops the pitcher
    # subpools, and any failure at all drops the whole block rather than the
    # build. See prior_population_centres.
    if USE_XWOBA_SHRINK:
        try:
            # Read the cache rather than calling load_team_pitcher_roles: a
            # diagnostic must not add a request, and a team that failed to load
            # above should drop out of the subpools instead of being retried.
            role_map = {}
            for roles in _team_pitcher_role_cache.values():
                role_map.update(roles or {})
            log_prior_population_centres(
                batter_cust, pitcher_cust, league_baseline.get("xwOBA"),
                XWOBA_SHRINK_K, role_map=role_map, probable_ids=prob_ids,
            )
        except Exception as e:  # noqa: BLE001
            log(f"  shrinkage-target population diagnostic unavailable: {e!r}")

    try:
        league_baseline["ERA"] = provider.load_league_era()
    except Exception as e:  # noqa: BLE001
        log(f"  league ERA unavailable: {e!r}")
        league_baseline["ERA"] = np.nan
    log(f"  recent ERA: {len(recent_start_era)} pitchers | league {league_baseline['ERA']}")

    log("Loading player bio + vL/vR platoon splits ...")
    league_hitter_ids = {int(pid) for pid, st in batter_stat.items()
                         if pd.notna((st or {}).get("PA"))
                         and float((st or {}).get("PA") or 0) >= MIN_LEAGUE_BASELINE_PA}
    if include_platoon and FULL_LEAGUE_PLATOON_BASELINES:
        log(f"  league hitter split population: {len(league_hitter_ids)}")
        people_league_hitters = provider.load_people(league_hitter_ids)
        player_splits_hit_league = provider.load_splits(
            league_hitter_ids, "hitting"
        )
    else:
        people_league_hitters, player_splits_hit_league = {}, {}
    people = dict(people_league_hitters)
    people.update(provider.load_people(lineup_ids | prob_ids))
    if include_platoon:
        player_splits_hit = provider.load_splits(lineup_ids, "hitting")
        player_splits_pit = provider.load_splits(prob_ids, "pitching")
    else:
        player_splits_hit, player_splits_pit = {}, {}

    log("Assembling tables ...")
    pitchers_df, batted_ball_profile_df = build_tables(
        slate_df, lineups, batter_stat, pitcher_stat, batter_bb, pitcher_bb, people)
    if not pitchers_df.empty:
        def _recent_value(r, key, default=np.nan):
            if r.get("Pos.") != "P" or pd.isna(r.get("player_id")):
                return default
            return recent_start_era.get(int(r["player_id"]), {}).get(key, default)

        pitchers_df["ERA_L5"] = pitchers_df.apply(lambda r: _recent_value(r, "era"), axis=1)
        pitchers_df["ERA_L5_GS"] = pitchers_df.apply(
            lambda r: _recent_value(r, "starts", 0), axis=1)
        pitchers_df["ERA_SEASON"] = pitchers_df.apply(
            lambda r: ((player_splits_pit.get(int(r["player_id"]), {}).get("overall", {})
                        or {}).get("era", np.nan))
            if r.get("Pos.") == "P" and pd.notna(r.get("player_id")) else np.nan,
            axis=1)
        # Statcast xERA (expected ERA) from the expected-statistics leaderboard,
        # shown against season ERA on the card. NaN wherever it's unavailable.
        xera_map = provider.load_pitcher_xera()
        pitchers_df["xERA"] = pitchers_df.apply(
            lambda r: xera_map.get(int(r["player_id"]), np.nan)
            if r.get("Pos.") == "P" and pd.notna(r.get("player_id")) else np.nan,
            axis=1)

    side_status = pd.concat([lineup_projection_df["away_lineup_status"].rename("status"),
                             lineup_projection_df["home_lineup_status"].rename("status")],
                            ignore_index=True) if not lineup_projection_df.empty else pd.Series(dtype=str)
    n_proj = int(lineup_projection_df[["away_lineup_projected", "home_lineup_projected"]].sum().sum())
    n_backfill = int(lineup_projection_df[
        ["away_savant_backfill_count", "home_savant_backfill_count"]
    ].sum().sum())
    log(f"lineup sources: {side_status.value_counts().to_dict()} of {2 * len(slate_df)} sides; "
        f"projected_or_partial={n_proj}; savant_backfills={n_backfill}")

    # Season leaderboard inputs. Best-effort in the same sense as the odds and
    # pitch-mix steps: this runs after everything the lean needs is already in
    # hand, and any failure here drops the boards rather than the build. The
    # role map is read from the cache build_pitching_plans populated -- a
    # display surface must not add a request to the critical path, and a club
    # that failed to load above stays out rather than being retried.
    leaderboard = None
    try:
        role_map = {}
        for roles in _team_pitcher_role_cache.values():
            role_map.update(roles or {})
        leaderboard = leaderboard_boards(
            batter_cust, pitcher_cust, league_baseline, people, role_map,
            load_people=provider.load_people)
        if leaderboard:
            lm = leaderboard["meta"]
            log(f"  leaderboard: {len(leaderboard['sp'])} SP of {lm['n_sp']} "
                f"identified | {lm['n_bat_ranked']} batters over "
                f"{len(leaderboard['bat'])} positions")
    except Exception as e:  # noqa: BLE001
        log(f"  leaderboard unavailable: {e!r}")

    return {
        "empty": False,
        "slate_df": slate_df,
        "pitchers_df": pitchers_df,
        "batted_ball_profile_df": batted_ball_profile_df,
        "league_baseline": league_baseline,
        "leaderboard": leaderboard,
        "people": people,
        "player_splits_hit": player_splits_hit,
        "player_splits_pit": player_splits_pit,
        "player_splits_hit_league": player_splits_hit_league,
        "people_league_hitters": people_league_hitters,
        "lineup_projection_df": lineup_projection_df,
        "pitching_plans": pitching_plans,
    }


# ============================================================
# CELL 4 -- STATCAST (xwOBA) MATCHUP
# ============================================================
STATCAST_RATE_COLS = ["xwOBA", "xBA", "xSLG", "Hard Hit%", "EV", "LA°", "K%", "BB%"]
# No WEIGHT_COL here any more. It named "BBE" as the lineup composite's weight
# and had been dead since v4 made slot-PA weighting unconditional; the constant
# outlived the branch that read it. BBE is still a real column on the stat
# frame and still weights the batted-ball league anchors, which are genuinely
# per-BBE rates -- it just no longer weights a per-PA rate.
USE_WEIGHTED = True
ADD_STATS = {"EV", "LA°"}

# --- xwOBA empirical-Bayes shrinkage (v5+, fixed K in v8) ------------------
# Season xwOBA (each hitter) and xwOBA-allowed (the starter) are regressed
# toward the league xwOBA baseline by sample size before they drive the lean:
#     x* = (n*x + K*prior) / (n + K)
# so a small-sample bat or a starter with few batters faced is pulled toward
# league-average rather than taken at face value. Both sides share one shrinkage
# target (the league xwOBA baseline). v8 uses K=100 for both batters and
# pitchers instead of estimating separate values from each day's leaderboards.
# That preserves more of the observed deviations from league average and makes
# the transformation stable and directly reproducible. Shrinkage touches only
# xwOBA (the lean stat); the other columns and raw per-hitter card values are
# untouched.
#
# The one value also regresses every member of the bullpen pool
# (`bullpen_xwoba_aggregate` -> `_shrink_one`), and whether it should is an open
# question rather than a settled one. Marcel is the precedent for expecting not:
# it carries separate baselines by role -- 200 PA for a hitter, 60 IP for a
# starter, 25 IP for a reliever. Those are IP-denominated and this is
# BF-denominated, and no published BF-denominated reliever constant exists to
# convert against, so the number has to be fitted here.
#
# `reliever_shrink_probe.py` fits it, and fits starters and batters through the
# same code path as controls -- the claim under test is a *difference* between
# the three, not a distance from 100. Run it (manual workflow, StatsAPI is not
# reachable from every dev environment) before changing this value, and read
# `dMSE vs 100` before the K column: weighted MSE is flat near its minimum, so a
# large move in K that buys nothing measurable is not a finding. Splitting K by
# role is lean-path, so it bumps MODEL_TAG.
USE_XWOBA_SHRINK = True
XWOBA_SHRINK_COL = "xwOBA"

# v3: 100 -> 400. Fitted, not chosen. `reliever_shrink_probe.py` measures K
# three independent ways and every one of them excludes 100 by a wide margin:
#
#   walk-forward, season-to-date -> next outing   391  [293, 519]  n=53,464
#   within-season split, relievers                600  [355, 1400]
#   within-season split, batters                  384  [302,  503]
#   within-season split, starters                 577  [398,  894]
#   season pair (upper bound)                     669-1048
#
# The walk-forward arm is the one that matches what this code does -- a
# season-to-date rate predicting tonight's innings, rather than one aggregate
# predicting another -- and it is the best powered by two orders of magnitude.
# Its estimator runs 10-13% high on synthetic logs of this shape, so its point
# estimate corresponds to a true K nearer 350.
#
# 400 is a single value for all three populations, NOT a role split. That was
# the original hypothesis and the data refused it: relievers 600 against
# starters 577, intervals overlapping almost entirely. Marcel's 2.4x role gap
# does not appear in BF-denominated data. 400 sits inside all three intervals,
# so one constant is both the parsimonious choice and the supported one.
#
# Do not read the third digit. Weighted MSE is flat near its minimum -- that is
# why `dMSE vs 100` is the column the probe tells you to read first -- so 350
# and 450 are not distinguishable here and precision beyond "about 400" would
# be false. What is distinguishable is 100 from 400.
#
# v11 REVERTS THIS TO 100 AND THE FIT ABOVE STILL STANDS. Read the two together
# rather than assuming the comment was updated to match: every arm above
# excludes 100, and nothing has been measured since that disputes them. K is
# back at 100 because the operator asked for the K=100 model, which is a call
# they get to make -- but it is an override of this fit, not a refutation of
# it, and the sequential record difference that motivated it does not separate
# (v9/v10 .567 against wOBA v5 .454 is z = +1.63, on eras whose always-home
# baselines were .515 and .574). If K is ever revisited, re-run the probe
# rather than re-reading this paragraph.
XWOBA_SHRINK_K = 100.0

# --- v4: the shrinkage TARGET is the player, not the pool ------------------
# K says how hard to regress. This says what to regress toward. Until v4 every
# batter, starter and reliever was pulled toward a population centre, so a
# career .360 hitter and a career .290 hitter with equal samples were published
# within a hair of each other -- at K=400 the prior supplies ~70% of a median
# reliever's rate and about half of a 400-PA batter's, so the target is most of
# the number.
#
# `player_priors.prior_for` replaces that centre with
#
#     pi = (H*theta_hist + C*mu) / (H + C)
#
# where theta_hist is the player's recency-weighted 2023-2025 Savant wOBA
# expressed as a deviation from the pool centre he earned it against, and mu is
# the SAME population centre the build used before. H = 0 returns mu exactly,
# so rookies and call-ups are unchanged and need no branch.
#
# Measured by `player_prior_probe.py` on a runner, out of sample, at this K.
# The early-season split -- period 1 = the first month, which is the case the
# build faces every April and where the prior carries most of the weight --
# reports BAT +25.35% [+18.30,+32.84], RP +22.28% [+16.36,+28.68] and
# SP +16.38% [+8.84,+25.86] in BF-weighted out-of-sample MSE against the
# shipped centre. The balanced mid-season split reports +7.33%, +5.91% and
# +2.83%, all with paired CIs excluding zero, and the C fitted on 2024 holds
# on 2025 and vice versa for batters and relievers.
#
# The unregressed career rate (C = 0) was measured too and LOSES on two of
# three populations (BAT -11.46%, RP -24.29%): a 70-PA career would otherwise
# supply most of a published rate. Role-separated history was measured and buys
# nothing (SP +16.38 vs +16.85 role-blind), so the build passes role=None.
#
# Turning this off restores the v3 behaviour exactly, and so does an empty
# data/woba_priors_*.csv -- the population centre is the H = 0 limit.
# v11 DEFAULTS THIS OFF, restoring the population centre. The probe results
# above are unchanged and unrefuted -- they measure per-player rate MSE out of
# sample, and they win there. What moved is a different quantity: on the
# ledger, the lineup component's correlation with its own realised phase ran
# +0.124 over the 194 v9/v10 side-games (population centre) against -0.004
# over v5's 220 (personal priors). Those can both be true -- better individual
# rates need not survive aggregation into a nine-batter composite -- and the
# forward difference does not separate either (+0.128, CI [-0.054, +0.301]).
# Off by operator decision with both readings on the record, not because the
# probe was found wrong.
#
# `PLAYER_PRIORS=1` alone does NOT restore v4 under v11, and the escape hatch
# should not be described as if it does: the frozen priors are wOBA-
# denominated, so `player_prior_history` refuses to serve them to an xwOBA
# build and every caller degrades to the population centre. Restoring v4 means
# a wOBA build AND this flag, or freezing an xwOBA prior set under its own
# filenames first.
USE_PLAYER_PRIORS = os.environ.get("PLAYER_PRIORS", "0") == "1"

_player_prior_state = {}


def player_prior_history():
    """The frozen priors, loaded once per build. ({}, {}) when absent.

    Absent is not an error: every caller degrades to the population centre,
    which is the v3 behaviour. A daily build must not fail to produce a page
    because a snapshot file is missing.

    A METRIC MISMATCH is an error, and it is the one this guard exists for.
    The frozen files hold wOBA (`priors_snapshot.RATE_COL` is pinned to it, and
    the centres carry `woba_wtd`/`woba_unw` headers). Under v11 the model runs
    on xwOBA with PLAYER_PRIORS off, so nothing reads them -- but turning
    PLAYER_PRIORS back on would shrink an xwOBA rate toward a wOBA-denominated
    personal centre. The two are close enough that the result looks entirely
    plausible and is silently wrong, which is exactly the failure mode
    `shadow_metric.patch` describes for its cache namespace. Refuse instead:
    degrade to the population centre and say why, rather than publish it.
    """
    if MODEL_RATE_LABEL != "wOBA":
        log(f"  player priors are wOBA-denominated and this build runs "
            f"{MODEL_RATE_LABEL} -> population centres "
            f"(frozen priors would be a metric mismatch)")
        return {}, {}
    if "loaded" not in _player_prior_state:
        try:
            hist, centres = player_priors.load_priors(PRIORS_DIR)
        except Exception as e:  # noqa: BLE001
            log(f"  player priors unavailable ({e!r}) -> population centres")
            hist, centres = {}, {}
        _player_prior_state["loaded"] = (hist, centres)
    return _player_prior_state["loaded"]


def player_prior_vector(ids, mu):
    """Per-player shrinkage targets aligned with `ids`, falling back to `mu`.

    Returns `mu` itself (a scalar) when priors are off or nothing is frozen, so
    the caller hands `shrink_xwoba` exactly what v3 handed it.
    """
    mu = _f(mu)
    if not USE_PLAYER_PRIORS or mu is None:
        return mu
    hist, centres = player_prior_history()
    if not hist or not centres:
        return mu
    return [player_priors.prior_for(pid, hist, centres, SEASON, mu,
                                    player_priors.PRIOR_HISTORY_C)
            for pid in ids]


def player_prior_one(pid, mu):
    """Scalar form, for the starter row and each member of the bullpen pool."""
    mu = _f(mu)
    if not USE_PLAYER_PRIORS or mu is None:
        return mu
    hist, centres = player_prior_history()
    if not hist or not centres:
        return mu
    return player_priors.prior_for(pid, hist, centres, SEASON, mu,
                                   player_priors.PRIOR_HISTORY_C)

# Pitch-mix shadow arm (pitch_arsenal.py). OFF by default and inert when on:
# it writes shadow columns for forward testing and never moves a lean, so no
# MODEL_TAG bump is involved either way. Turn it on only once
# pitch_arsenal_probe.py shows the batter x pitch-type deviation is reliable
# enough to beat the noise it injects; see that script for the budget.
USE_PITCH_MIX_SHADOW = os.environ.get("PITCH_MIX_SHADOW", "0") == "1"

# --- SP platoon-advantage xwOBA adjustment ---------------------------------
# A one-sided hitter's shrunk xwOBA is moved by the handedness/matchup cell in
# PLATOON_XWOBA_OFFSETS before the lineup composite is taken. Every pair keeps
# the same conservative 0.021 advantage-minus-disadvantage gap, but the season
# line is no longer treated as the midpoint of a 50/50 exposure mix:
#
#                    vs LHP     vs RHP
#       LHB           -0.016     +0.005
#       RHB           +0.015     -0.006
#
# The cells are centred at approximately 76.2% advantage exposure for LHB and
# 28.6% for RHB. Applied AFTER shrinkage on purpose: shrinkage estimates season
# talent from a noisy sample, while this is a fixed population matchup prior and
# must not be regressed away for a low-PA bat.
#
# The offset is a DEVIATION FROM THE HITTER'S OWN SEASON LINE, and that line is
# already a platoon blend weighted by his real exposure to each pitcher hand.
# So the offset only applies where the season line averages over both platoon
# states:
#
#   * A switch hitter takes the opposite side in essentially every PA, so his
#     season xwOBA IS his advantage-state number. He holds the edge -- the card
#     still marks him and the platoon-OPS lens still counts him -- but adding
#     another advantage offset on top would count it twice. His offset is 0.
#   * A hitter with no recorded side gets 0 for the same reason the unknown-hand
#     case does: no evidence either way, so no move.
#   * A hitter whose starter's throwing hand is unknown gets 0. An absent bio is
#     not evidence of a platoon disadvantage.
#
# These remain population offsets, not player-specific split estimates. A
# future per-hitter model would need separate vs-hand samples and much heavier
# regression than the K=100 overall-wOBA shrinkage used above.
PLATOON_XWOBA_OFFSETS = {
    ("L", "L"): -0.016,
    ("L", "R"): +0.005,
    ("R", "L"): +0.015,
    ("R", "R"): -0.006,
}
PLATOON_ADV_COL = "sp_platoon_adv"     # per-hitter tag carried from the SP block
SP_THROWS_COL = "sp_throws"            # faced starter's hand, distinct from the hitter's own
OPP = {"L": "R", "R": "L"}


def effective_stand(bats, throws):
    """Side a hitter effectively bats from against a starter throwing `throws`.

    A switch hitter -- or a hitter with no recorded side -- takes the side
    opposite the pitcher. Returns None when the starter's hand is unknown,
    where the platoon question has no answer either way.
    """
    t = str(throws or "")[:1].upper()
    if t not in ("L", "R"):
        return None
    b = str(bats or "")[:1].upper()
    return b if b in ("L", "R") else OPP[t]


def platoon_advantage(bats, throws):
    """True when the hitter holds the platoon edge over this starter.

    False when he does not, None when the starter's hand is unknown. This is the
    handedness fact -- a switch hitter holds the edge -- and it drives the card's
    marker and the platoon-OPS lens. It is NOT the xwOBA offset: see
    platoon_xwoba_offset for why holding the edge and being moved by it differ.
    """
    eff = effective_stand(bats, throws)
    return None if eff is None else eff != str(throws or "")[:1].upper()


def platoon_xwoba_offset(bats, throws):
    """How far to move this hitter's season xwOBA for tonight's platoon state.

    The lookup is handedness-specific and exposure-centred, but only one-sided
    hitters move. A switch hitter's season line is already his advantage-state
    number, so his offset is 0; so is an unrecorded bats side or an unknown
    starter hand.
    """
    b = str(bats or "")[:1].upper()
    t = str(throws or "")[:1].upper()
    if b not in ("L", "R") or t not in ("L", "R"):
        return 0.0
    return PLATOON_XWOBA_OFFSETS[(b, t)]


def platoon_xwoba_offsets(g):
    """Per-hitter xwOBA offsets for a lineup group, as a positionally-indexed
    Series (aligns with the reset-index value vectors wmean consumes).

    Derived from `bats` and the faced starter's hand rather than from a stored
    number, so the tag and the offset cannot drift apart. Returns None when the
    group carries neither column, which leaves the composite untouched.
    """
    cols = getattr(g, "columns", [])
    if "bats" not in cols or SP_THROWS_COL not in cols:
        return None
    return pd.Series([platoon_xwoba_offset(b, t)
                      for b, t in zip(g["bats"], g[SP_THROWS_COL])], dtype=float)


def matchup_value(B, P, stat, L):
    if pd.isna(B) or pd.isna(P) or pd.isna(L):
        return np.nan
    if stat in ADD_STATS:
        return B + P - L
    return (B * P / L) if L else np.nan


def coerce_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = (df[c].astype(str)
                 .str.replace("%", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.strip())
            s = s.where(~s.isin(["", "nan", "None", "--", "—"]), other=np.nan)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def wmean(vals, wts):
    vals = pd.to_numeric(pd.Series(vals).reset_index(drop=True), errors="coerce")
    if wts is None:
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
    wts = pd.to_numeric(pd.Series(wts).reset_index(drop=True), errors="coerce")
    m = vals.notna() & wts.notna() & (wts > 0)
    if not m.any():
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
    return float(np.average(vals[m], weights=wts[m]))


def wsd(vals, wts):
    """Slot-PA-weighted sd of a lineup's shrunk rates. NaN below two bats.

    Instrumentation for one question the composite cannot answer: `B_0` is a
    weighted MEAN, so a lineup of nine .310 bats and a lineup of one .400 and
    eight .299 bats publish the same number. This records how concentrated the
    talent behind that mean was, so "do good hitters get averaged down" becomes
    a measurement rather than an argument -- regress the rate residual, and the
    lean's error against run margin, on this.

    DIAGNOSTIC ONLY. It reaches the dump and ledger and nothing else; no lean,
    delta, edge or grade reads it, which is why adding it carries no MODEL_TAG
    implication. Two things to know before using it:

      * It is measured on the NEUTRAL, post-shrinkage values -- the same
        vector `opp_xwOBA_neutral` averages. Post-shrinkage because that is
        what the model actually believes about each bat; pre-platoon because
        dispersion should describe the lineup, not tonight's handedness draw.
      * A team-backfilled hitter carries the team aggregate, so he sits at
        (roughly) the mean by construction and deflates this. Control for it
        with `lineup_savant_backfill_*`, which the ledger already carries --
        do not read a low value as a flat lineup without checking that first.

    Population (not sample) sd, weighted by the same slot vector as the mean,
    so mean and sd describe the same weighting. Falls back to an unweighted sd
    when no usable weights exist, matching `wmean`'s degradation.
    """
    vals = pd.to_numeric(pd.Series(vals).reset_index(drop=True), errors="coerce")
    if vals.notna().sum() < 2:
        return np.nan
    if wts is None:
        return float(vals.std(ddof=0, skipna=True))
    wts = pd.to_numeric(pd.Series(wts).reset_index(drop=True), errors="coerce")
    m = vals.notna() & wts.notna() & (wts > 0)
    if m.sum() < 2:
        return float(vals.std(ddof=0, skipna=True))
    v, k = vals[m].to_numpy(float), wts[m].to_numpy(float)
    mu = float(np.average(v, weights=k))
    return float(np.sqrt(np.average((v - mu) ** 2, weights=k)))


def slot_pa_weights(order):
    """Map a batting-order (1-9) series to expected PA/game weights.

    Returns a positionally-reindexed Series (aligns with the equally
    reset-indexed value vectors used by wmean/_wmean); slots outside 1-9 map to
    NaN so wmean falls back to an equal mean for them.
    """
    o = pd.to_numeric(pd.Series(order).reset_index(drop=True), errors="coerce")
    return o.map(LINEUP_SLOT_PA)


def lineup_weight(g):
    """Slot-PA weights for a lineup group, or None when the order is unusable.

    This used to take a `fallback_col` and return that season-volume column
    (`WEIGHT_COL = "BBE"`) when the batting order was missing -- a per-BBE
    denominator under a per-PA rate, which would have underweighted
    three-true-outcome bats. It was unreachable: `hitter_rows` assigns
    `batting_order` as `enumerate(lu, start=1)` for every lineup slot and the
    column projection carries it, so `slot_pa_weights` always yields at least
    one non-null weight for a lineup group. Measured on frames built through
    `build_tables` -> `segment_pitcher_blocks` -- posted, partial, 10-man, a
    hitter absent from the leaderboard, and a side with no BBE at all -- the
    returned vector was the slot vector on every group, while the BBE
    composite differed by up to 0.008 wOBA. Dead, not equivalent, so deleting
    it moves nothing and removes the way it could have.

    None (no usable order) is left to the caller: the platoon path already
    substitutes split PA, and the lineup path lets `wmean` take an equal mean.
    """
    if USE_SLOT_PA_WEIGHTS and BATTING_ORDER_COL in getattr(g, "columns", []):
        w = slot_pa_weights(g[BATTING_ORDER_COL])
        if w.notna().any():
            return w
    return None


def shrink_xwoba(x, n, prior, k):
    """Regress observed rate(s) toward prior by sample size: (n*x + k*prior)/(n+k).

    Series in / positionally-reindexed Series out (aligns with wmean's reset
    index). A missing rate or n<=0 collapses to the prior. Pass-through when
    shrinkage is disabled or k/prior are unusable.

    v4: `prior` may be a per-player vector as well as a scalar. Nothing else
    changes -- the arithmetic is identical and a vector of one repeated value
    reproduces the scalar path exactly, which is what makes the personal-prior
    change a substitution rather than a second code path.
    """
    xs = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
    if not USE_XWOBA_SHRINK or k is None or not np.isfinite(k) or prior is None:
        return xs
    if np.isscalar(prior) or isinstance(prior, (int, float)):
        if not np.isfinite(prior):
            return xs
        ps = pd.Series(float(prior), index=xs.index)
    else:
        ps = pd.to_numeric(pd.Series(prior).reset_index(drop=True),
                           errors="coerce")
        if len(ps) != len(xs) or not ps.notna().any():
            return xs
    ns = pd.to_numeric(pd.Series(n).reset_index(drop=True), errors="coerce").fillna(0.0).clip(lower=0.0)
    out = (ns * xs.fillna(ps) + k * ps) / (ns + k)
    # A player whose personal prior is unusable falls back to the unshrunk
    # rate rather than to NaN -- the same degradation the scalar guard above
    # makes, applied per row.
    return out.where(ps.notna(), xs)


def _shrink_one(x, n, prior, k):
    """Scalar form of shrink_xwoba for a single pitcher row."""
    if not USE_XWOBA_SHRINK or k is None or not np.isfinite(k) or prior is None or pd.isna(prior):
        return x
    n = float(n) if pd.notna(n) else 0.0
    x = float(prior) if pd.isna(x) else float(x)
    return (n * x + k * prior) / (n + k)


# --- percentile display scale (casual redesign) -----------------------------
# Display-only. Ranks a player's *shrunk* xwOBA against a reference population
# of qualified regulars so the casual card can show a 0-100 Statcast-style
# percentile instead of a raw decimal. It reuses the model's shrink K/prior, so
# the bars are the same regressed values the lean is built on -- it never feeds
# back into the lean, MODEL_TAG, or the ledger.
#
# Reference = Savant's qualifier (2.1 PA per team-game for batters, 1.25 for
# pitchers); games played is estimated from the leaderboard's own PA spread so
# the cut scales through the season without an extra fetch. Non-qualified
# players (call-ups, platoon bats) are still *scored against* that reference by
# their shrunk value -- they land near league, filling the projected-lineup
# coverage gap a hard in/out qualifier would leave blank.
PCTILE_QUAL_BAT = 2.1      # PA per team-game
PCTILE_QUAL_PIT = 1.25     # BF per team-game
PCTILE_PA_PER_GAME = 4.3   # ~ an everyday player's PA/team-game, to back out games
PCTILE_MIN_POOL = 20       # need a real population for a stable scale


def _est_team_games(pa_series):
    """Estimate team games played from a leaderboard's PA spread: an everyday
    player's PA (~p99) ≈ PCTILE_PA_PER_GAME × games. Robust to a lone outlier."""
    pa = pd.to_numeric(pd.Series(pa_series), errors="coerce").dropna()
    if pa.empty:
        return None
    g = float(pa.quantile(0.99)) / PCTILE_PA_PER_GAME
    return g if np.isfinite(g) and g > 0 else None


def pool_shrink_target(cust):
    """A player pool's centre: the plain unweighted mean of its rates.

    Unweighted, not PA-weighted, and the distinction is the whole point. The
    PA-weighted rate is the mean of the *plate appearances* — it answers "what
    does a league PA look like", which is the right question for a normalizer
    and the wrong one for a shrinkage target. Playing time is awarded for
    hitting well, so weighting by it puts the centre where the regulars are,
    and every fringe bat is then regressed toward a population he is not drawn
    from. On 2026-08-04 that gap measured 23 points pool-wide, 45 in the
    sub-qualified tail.

    Unweighted is safe here despite the `min=1` board being mostly small
    samples: sampling noise on a rate is mean-zero (`E[k/n] = p` exactly), so
    tiny lines scatter around their talent without dragging the *mean*. They
    would wreck a variance or a quantile; they do not bias this.

    Two better-looking candidates were measured and rejected. Benchmark: rank
    sub-qualified bats by their published percentile against a reference of
    qualified regulars, score Spearman against known talent, 40-60 seeds per
    cell, sweeping how strongly playing time signals talent.

      talent->PA link      0.00    0.25    0.50    1.00
      PA-weighted (today) .5241   .5666   .5235   .1829
      unweighted          .5236   .5715   .5365   .2503
      per-n trend         .4759   .5815   .5978   .5390

    A precision weight `n/(n+K)` — the exchangeable EB estimator of the prior
    mean — lost to plain unweighted 0/40, because downweighting the low-PA
    members biases the centre back up toward the pole it is supposed to avoid,
    and those members are exactly the ones being ranked.

    A prior that varies with `n` (regress rate on `log PA`) is better wherever
    playing time really signals talent, and worse when it does not — it fits
    noise into the slope, and `n` also sets the shrink weight, so the error
    compounds where the weight is largest. Shrinking the slope by its own
    evidence does not rescue that column (.4821). It is the right answer only
    once `corr(rate, PA)` is known to be solidly positive on the real board,
    which is why that correlation is now logged beside the pool centres. Do not
    adopt it from this docstring; adopt it from the log.

    Unweighted is the only candidate that beats today's target in every regime
    and loses in none.
    """
    cols = getattr(cust, "columns", [])
    if cust is None or MODEL_RATE_SOURCE_COL not in cols or "pa" not in cols:
        return None
    x = pd.to_numeric(cust[MODEL_RATE_SOURCE_COL], errors="coerce")
    n = pd.to_numeric(cust["pa"], errors="coerce")
    m = x.notna() & n.notna() & (n > 0)
    if not m.any():
        return None
    return float(x[m].mean())


def build_pctile_ref(cust, prior, k, qual_pa):
    """Sorted array of shrunk xwOBA for qualified regulars in a custom
    leaderboard. `qual_pa` is an absolute PA/BF threshold (caller derives it
    from team-games so batters and pitchers share one games estimate). Returns
    (sorted_array | None, qual_pa | None); falls back to the whole (min=1) pool
    if qualifying leaves too few for a stable scale."""
    cols = getattr(cust, "columns", [])
    if cust is None or MODEL_RATE_SOURCE_COL not in cols or "pa" not in cols:
        return None, None
    xw = pd.to_numeric(cust[MODEL_RATE_SOURCE_COL], errors="coerce")
    pa = pd.to_numeric(cust["pa"], errors="coerce")
    base = xw.notna() & pa.notna() & (pa > 0)
    m = (base & (pa >= qual_pa)) if qual_pa else base
    if int(m.sum()) < PCTILE_MIN_POOL:      # qualifier too strict this early -> pool everyone
        m, qual_pa = base, None
    if int(m.sum()) < PCTILE_MIN_POOL:
        return None, None
    # v4: shrink the reference population the same way the ranked value is
    # shrunk -- toward each player's own prior. The invariant the caller states
    # is that the reference and the ranked value share a target; personal
    # priors keep that, because both sides now use the SAME player's target.
    # Ranking a personal-prior rate against a population-prior distribution
    # would break it, and the bar would stop being the percentile of the number
    # printed beside it.
    target = prior
    if "player_id" in cols:
        target = player_prior_vector(pd.Series(cust["player_id"])[m], prior)
    arr = np.sort(pd.to_numeric(shrink_xwoba(xw[m], pa[m], target, k), errors="coerce")
                  .dropna().to_numpy())
    return (arr if arr.size else None), qual_pa


def build_pctile_ref_raw(series, min_n=PCTILE_MIN_POOL):
    """Sorted array of an already-final metric (e.g. K-BB%) for ranking without
    shrinkage. None when the pool is too thin."""
    v = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return np.sort(v.to_numpy()) if len(v) >= min_n else None


def pctile_rank(value, pa, ref_arr, prior, k, invert=False, pid=None):
    """Percentile (0-100) of a player's *shrunk* xwOBA within ref_arr. Set
    invert for pitchers (low xwOBA-against = high percentile). None when
    unavailable.

    `pid` opts this player into his own v4 prior, matching how `build_pctile_ref`
    shrank the population he is being ranked against. Omitting it ranks a
    population-shrunk value against a personal-shrunk distribution, which is
    the mismatch the reference's docstring warns about."""
    if ref_arr is None or getattr(ref_arr, "size", 0) == 0 or value is None or pd.isna(value):
        return None
    s = _shrink_one(value, pa, player_prior_one(pid, prior) if pid is not None
                    else prior, k)
    if s is None or not np.isfinite(s):
        return None
    frac = float(np.searchsorted(ref_arr, s, side="right")) / ref_arr.size
    pct = 100.0 * (1.0 - frac if invert else frac)
    return max(0.0, min(100.0, pct))


def pctile_rank_raw(value, ref_arr, invert=False):
    """Percentile of an already-final value (no shrink) within ref_arr."""
    if ref_arr is None or getattr(ref_arr, "size", 0) == 0 or value is None or pd.isna(value):
        return None
    frac = float(np.searchsorted(ref_arr, float(value), side="right")) / ref_arr.size
    pct = 100.0 * (1.0 - frac if invert else frac)
    return max(0.0, min(100.0, pct))


# --- season leaderboards (display only) --------------------------------------
# Two boards built from the same Savant custom leaderboards the model already
# fetches, regressed with the model's own `shrink_xwoba` at XWOBA_SHRINK_K.
# Display only: nothing here reaches a lean, a delta, a dump or a ledger row,
# so it carries no MODEL_TAG implication.
#
# NO QUALIFIER, DELIBERATELY. A hard `PA >= N` cut is the threshold-cliff
# anti-pattern this repo has already fixed twice, and shrinkage makes it
# unnecessary rather than merely unfashionable: at K=100 a 12-PA bat keeps
# 12/112 of his deviation, so he lands within a hair of the pool centre and
# cannot top a board. The regression IS the qualifier, applied continuously.
# `pa` is still printed on every row so a reader can see how much of a number
# is the player and how much is the target.
#
# THE TWO BOARDS USE DIFFERENT SHRINK TARGETS AND THAT IS NOT AN OVERSIGHT.
# They are the same two targets `_pctile_ref_bat` / `_pctile_ref_pit` already
# use, for the reason argued at their call site: a rank against a population
# has to regress toward that population's own centre. Playing time is awarded
# for hitting well, so the PA-weighted league rate sits where the regulars are,
# and regressing a fringe bat toward it lands him in the middle of a group he
# is not drawn from -- he then ranks high for a reason that is arithmetic
# rather than hitting. That distortion was measured on 2026-08-04 at 23
# PERCENTILE points pool-wide and 45 in the sub-qualified tail (a rank error,
# not a gap between the two targets, which is far smaller). The pitcher pool
# needs no such correction: the same diagnostic put slate probables 0.0016
# below the league rate, i.e. already centred on it.
#
# So using the league rate for batters here would publish a leaderboard whose
# order disagrees with the percentile bar on the same player's card -- the
# internal/public artifacts-disagreeing defect, one surface out. Both targets
# arrive from `league_baseline`, which is where fetch_all stashes them, so
# neither is recomputed here and neither can drift from the bars.
LEADERBOARD_SP_N = 100          # SP rows published
LEADERBOARD_BAT_N = 15          # batter rows published per position
# Fielding order, then the bat-only board. Anything StatsAPI reports that is
# not listed lands after these in alphabetical order rather than being dropped:
# a position we failed to anticipate should look unfamiliar, not vanish.
LEADERBOARD_POS_ORDER = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")
# StatsAPI lists some hitters under a generic `OF` rather than LF/CF/RF, and a
# two-way player under `TWP`. Neither names a corner this page can rank against
# its own kind: `OF` is the same job as the three specific boards but cannot be
# assigned to one of them without inventing which, and `TWP` is one or two
# players a season. Both pool into DH -- the bat-only board, where a hitter is
# ranked on hitting and nothing else -- rather than standing as boards of nine
# and two. Mapped BEFORE grouping, so a pooled player competes for the same 15
# slots as everyone else on that board rather than being appended to it.
LEADERBOARD_POS_POOL = {"OF": "DH", "TWP": "DH"}


def shrunk_rate_board(cust, prior, k, keep_ids=None):
    """player_id / pa / rate / shrunk for a Savant custom leaderboard.

    `keep_ids` restricts the pool. It does not change any surviving player's
    number -- `shrink_xwoba` is per row against a shared target -- so this is a
    filter on WHO is published, never on what they are worth.

    Returns an empty frame (not None) when the board is unusable, so a caller
    can render "no rows" without distinguishing two failure shapes it would
    treat identically anyway.
    """
    empty = pd.DataFrame(columns=["player_id", "pa", "rate", "shrunk"])
    cols = getattr(cust, "columns", [])
    src = MODEL_RATE_SOURCE_COL
    if cust is None or src not in cols or "pa" not in cols or "player_id" not in cols:
        return empty
    out = pd.DataFrame({
        "player_id": pd.to_numeric(cust["player_id"], errors="coerce"),
        "pa": pd.to_numeric(cust["pa"], errors="coerce"),
        "rate": pd.to_numeric(cust[src], errors="coerce"),
    })
    out = out[out["player_id"].notna() & out["rate"].notna()
              & out["pa"].notna() & (out["pa"] > 0)]
    if keep_ids is not None:
        out = out[out["player_id"].astype("int64").isin(
            {int(i) for i in keep_ids})]
    if out.empty:
        return empty
    out = out.reset_index(drop=True)
    out["player_id"] = out["player_id"].astype("int64")
    # Personal priors, exactly as build_pctile_ref takes them. Under an xwOBA
    # build `player_prior_vector` returns the scalar unchanged (the frozen
    # priors are wOBA-denominated and player_prior_history refuses them), so
    # this is a no-op today and the right call the day an xwOBA prior set
    # ships -- at which point both surfaces move together instead of one.
    target = player_prior_vector(out["player_id"], prior)
    out["shrunk"] = pd.to_numeric(
        shrink_xwoba(out["rate"], out["pa"], target, k), errors="coerce")
    return out[out["shrunk"].notna()].reset_index(drop=True)


def starter_ids(pitcher_cust, role_map):
    """Pitcher ids whose season workload is primarily starting, or None.

    `role_map` is player id -> `load_team_pitcher_roles` line. The predicate is
    `start_share > RP_MAX_START_SHARE` -- the SAME one `relief_pitcher_ids` and
    `prior_population_centres` use, so "SP" means one thing on this site. Do
    not substitute a BF proxy: a high-BF reliever and a low-BF starter both
    exist, and the repo already has the rotation/relief split measured.

    Returns None when no role line is available at all, which a caller must
    render as "cannot say" rather than as an empty rotation.
    """
    cols = getattr(pitcher_cust, "columns", [])
    if not role_map or pitcher_cust is None or "player_id" not in cols:
        return None
    out = set()
    for pid in pd.to_numeric(pitcher_cust["player_id"], errors="coerce").dropna():
        share = (role_map.get(int(pid)) or {}).get("start_share")
        if share is not None and pd.notna(share) and float(share) > RP_MAX_START_SHARE:
            out.add(int(pid))
    return out


def _pos_sort_key(pos):
    try:
        return (0, LEADERBOARD_POS_ORDER.index(pos), "")
    except ValueError:
        return (1, 0, str(pos))


def leaderboard_boards(batter_cust, pitcher_cust, league_baseline, people,
                       role_map, k=None, load_people=None):
    """The two published boards, or None when neither can be built.

    Returns {"sp": frame, "bat": [(position, frame), ...], "meta": {...}} with
    `meta` carrying the provenance every count on the page is quoted from --
    pool sizes, the two targets, and how many pitchers carried a role line.
    Report provenance, do not assert coverage: the SP board is only as complete
    as the role map, which `fetch_all` builds from the clubs it has already
    called for, and the page says so rather than implying a full rotation.

    `people` supplies names and positions and is what the build has already
    fetched: every league hitter (so the batter board needs nothing new) plus
    the slate's probables. `load_people` is an optional one-shot lookup for the
    ids still missing after selection -- the ~100 published starters, most of
    whom are not pitching today. It is called ONCE, AFTER the boards are cut,
    so the request is proportional to what is published rather than to the
    season board. A failure there costs names, not the page.
    """
    k = XWOBA_SHRINK_K if k is None else k
    league_baseline = league_baseline or {}
    people = dict(people or {})
    prior_pit = _f(league_baseline.get(MODEL_RATE_INTERNAL_COL))
    prior_bat = _f(league_baseline.get("_pctile_prior_bat"))
    if prior_bat is None:
        prior_bat = prior_pit
    if prior_pit is None or prior_bat is None:
        return None

    sp_ids = starter_ids(pitcher_cust, role_map)
    sp = shrunk_rate_board(pitcher_cust, prior_pit, k,
                           keep_ids=sp_ids if sp_ids is not None else None)
    if sp_ids is None:
        sp = sp.iloc[0:0]
    # Lower xwOBA allowed is better, so the SP board ascends. Ties break on the
    # larger sample: two identical shrunk rates are not equally well measured.
    sp = sp.sort_values(["shrunk", "pa"], ascending=[True, False]).head(
        LEADERBOARD_SP_N).reset_index(drop=True)

    bat = shrunk_rate_board(batter_cust, prior_bat, k)
    bat["pos"] = [LEADERBOARD_POS_POOL.get(p, p) for p in
                  ((people.get(int(i)) or {}).get("pos")
                   for i in bat["player_id"])]
    # A pitcher who took a plate appearance is not on a batter leaderboard, and
    # a hitter with no bio is not filed under a position we did not read.
    bat = bat[bat["pos"].notna() & (bat["pos"] != "P")]
    boards = []
    for pos in sorted(set(bat["pos"]), key=_pos_sort_key):
        g = bat[bat["pos"] == pos]
        # Higher xwOBA is better for a bat -- the opposite of the SP board, and
        # the reason the two are separate calls rather than one parameterised
        # sort with a flag nobody reads.
        g = g.sort_values(["shrunk", "pa"], ascending=[False, False])
        boards.append((pos, g.head(LEADERBOARD_BAT_N).reset_index(drop=True)))

    # Names last, for the published rows only.
    published = set(sp["player_id"]) | {i for _, g in boards
                                        for i in g["player_id"]}
    missing = sorted(i for i in published
                     if not (people.get(int(i)) or {}).get("name"))
    if missing and load_people:
        try:
            people.update(load_people(missing) or {})
        except Exception as e:  # noqa: BLE001
            log(f"  leaderboard: bio lookup failed for {len(missing)} "
                f"players, ids shown instead ({e!r})")

    def name_of(pid):
        return (people.get(int(pid)) or {}).get("name") or f"player {int(pid)}"

    sp["name"] = [name_of(i) for i in sp["player_id"]]
    boards = [(pos, g.assign(name=[name_of(i) for i in g["player_id"]]))
              for pos, g in boards]

    meta = {
        "k": float(k),
        "prior_pit": prior_pit,
        "prior_bat": prior_bat,
        "n_pitchers": 0 if pitcher_cust is None else int(len(pitcher_cust)),
        "n_batters": 0 if batter_cust is None else int(len(batter_cust)),
        "n_sp": 0 if sp_ids is None else len(sp_ids),
        "sp_roles_known": sp_ids is not None,
        "n_role_lines": len(role_map or {}),
        "n_bat_ranked": int(len(bat)),
    }
    if sp.empty and not boards:
        return None
    return {"sp": sp, "bat": boards, "meta": meta}


def segment_pitcher_blocks(df, rate_cols):
    df = coerce_numeric(df, rate_cols)

    pitcher_rows, hitter_rows = [], []
    for _, g in df.groupby(["game_pk", "table_index"], sort=False):
        cur_p = cur_side = cur_throws = None
        for _, r in g.iterrows():
            name = r.get("Name")
            # Structural detection: is_sp marks the probable-pitcher row and
            # sp_side names its side. This replaces name matching, which failed
            # when a missing bio left a `pitcher_{pid}` fallback name and could
            # collide on identical normalized names across the two probables.
            if bool(r.get("is_sp")):
                cur_p = name
                cur_side = r.get("sp_side")
                cur_throws = r.get("throws")
                pitcher_rows.append({**r.to_dict(), "pitcher_side": cur_side})
                continue
            if cur_p is None:
                continue
            bat_side = {"away": "home", "home": "away"}.get(cur_side)
            # The starter's hand rides along under its own key -- the hitter row
            # already carries a `throws` (his own). Tagging the platoon edge here
            # rather than reading it off the platoon lens keeps the xwOBA lean
            # independent of a lens that is optional and may abstain.
            hitter_rows.append({**r.to_dict(), "faced_pitcher": cur_p,
                                "pitcher_side": cur_side, "batting_side": bat_side,
                                SP_THROWS_COL: cur_throws,
                                PLATOON_ADV_COL: platoon_advantage(r.get("bats"),
                                                                   cur_throws)})

    P = pd.DataFrame(pitcher_rows)
    H = pd.DataFrame(hitter_rows)
    if not P.empty:
        P = P.drop_duplicates(subset=["game_pk", "Name"]).reset_index(drop=True)
    if not H.empty:
        H = H.drop_duplicates(subset=["game_pk", "faced_pitcher", "Name"]).reset_index(drop=True)
    return P, H


def aggregate_lineup(H, rate_cols, weighted=True, shrink_prior=None, shrink_k=None):
    if H is None or H.empty:
        return pd.DataFrame()
    out = []
    for (gpk, fp), g in H.groupby(["game_pk", "faced_pitcher"], sort=False):
        rec = {"game_pk": gpk, "faced_pitcher": fp,
               "pitcher_side": g["pitcher_side"].iloc[0],
               "batting_side": g["batting_side"].iloc[0],
               "n_opp_hitters": len(g)}
        # Weight the lineup composite by expected PA per batting-order slot
        # (top of the order sees more pitches) rather than by season BBE volume.
        w = lineup_weight(g)
        for c in rate_cols:
            if c not in g.columns:
                continue
            # Regress each hitter's xwOBA toward the league prior by their PA
            # BEFORE compositing, so a small-sample bat can't swing the lineup
            # number; other columns pass through raw. A posted Savant-missing
            # hitter already carries a team aggregate, so keep that final value
            # instead of applying player-level shrinkage a second time.
            if c == XWOBA_SHRINK_COL and shrink_prior is not None and "PA" in g.columns:
                # v4: each hitter regresses toward his OWN prior when he has
                # history, and toward `shrink_prior` -- the league centre v3
                # used for everyone -- when he does not.
                target = shrink_prior
                if "player_id" in g.columns:
                    target = player_prior_vector(g["player_id"], shrink_prior)
                vals = shrink_xwoba(g[c], g["PA"], target, shrink_k)
                if MODEL_RATE_TEAM_BACKFILL_COL in g.columns:
                    backfilled = (
                        pd.Series(g[MODEL_RATE_TEAM_BACKFILL_COL])
                        .reset_index(drop=True)
                        .fillna(False)
                        .astype(bool)
                    )
                    raw_vals = pd.to_numeric(
                        pd.Series(g[c]).reset_index(drop=True), errors="coerce"
                    )
                    vals = vals.where(~backfilled, raw_vals)
            else:
                vals = pd.to_numeric(pd.Series(g[c]).reset_index(drop=True), errors="coerce")

            # Preserve the neutral xwOBA composite before applying the
            # probable starter's handedness term. v9 uses this B_0 against the
            # bullpen and the adjusted B_SP only during the starter phase.
            if c == XWOBA_SHRINK_COL:
                rec["opp_xwOBA_neutral_mean"] = (
                    float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
                )
                rec["opp_xwOBA_neutral_wmean"] = wmean(vals, w)
                rec["opp_xwOBA_neutral"] = (
                    rec["opp_xwOBA_neutral_wmean"]
                    if weighted
                    else rec["opp_xwOBA_neutral_mean"]
                )
                # How concentrated the talent behind that mean was. See wsd:
                # diagnostic only, and computed on this same vector so the
                # mean and its dispersion always describe one lineup.
                rec["opp_xwOBA_sd"] = wsd(vals, w if weighted else None)

                # Platoon term last, on the shrunk (or backfilled) value: it is
                # a matchup fact about tonight, not a season rate to regress.
                offsets = platoon_xwoba_offsets(g)
                if offsets is not None:
                    vals = vals + offsets
            rec[f"opp_{c}_mean"] = (
                float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
            )
            rec[f"opp_{c}_wmean"] = wmean(vals, w)
            rec[f"opp_{c}"] = rec[f"opp_{c}_wmean"] if weighted else rec[f"opp_{c}_mean"]
            if c == XWOBA_SHRINK_COL:
                # Keep opp_xwOBA as a backward-compatible alias for the
                # starter-adjusted composite used by v8 and older consumers.
                rec["opp_xwOBA_vs_sp"] = rec["opp_xwOBA"]
                rec["platoon_delta_sp"] = (
                    rec["opp_xwOBA_vs_sp"] - rec["opp_xwOBA_neutral"]
                    if pd.notna(rec["opp_xwOBA_vs_sp"])
                    and pd.notna(rec["opp_xwOBA_neutral"])
                    else np.nan
                )
        out.append(rec)
    return pd.DataFrame(out)


def build_matchup(P, agg, rate_cols, league_baseline, shrink_prior=None, shrink_k=None):
    if P.empty or agg.empty:
        return pd.DataFrame()
    Pk = P.set_index(["game_pk", "Name"])
    rows = []
    for _, a in agg.iterrows():
        key = (a["game_pk"], a["faced_pitcher"])
        if key not in Pk.index:
            continue
        pr = Pk.loc[key]
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        side = a["pitcher_side"]
        opp_team = pr.get("home_team") if side == "away" else pr.get("away_team") if side == "home" else None
        rec = {"game_pk": a["game_pk"], "game_date": pr.get("game_date"),
               "game_datetime_utc": pr.get("game_datetime_utc"),
               "matchup": pr.get("matchup"), "side": side, "pitcher": a["faced_pitcher"],
               "opp_team": opp_team, "n_opp": int(a["n_opp_hitters"])}
        for c in rate_cols:
            pv = pd.to_numeric(pr.get(c), errors="coerce")
            if c == XWOBA_SHRINK_COL:
                # Was tonight's published starter rate measured, or defaulted?
                # A starter absent from the leaderboard arrives here with no
                # rate and no BF, and _shrink_one returns the prior unchanged --
                # so the card prints a plausible number that contains no
                # observation of this arm, and until now nothing said so. At
                # K=400 against a personal prior the defaulted value is not even
                # distinguishable by eye from a measured one.
                #
                # Two states and a count, deliberately: the boolean says whether
                # an observation exists at all, and starter_rate_bf carries how
                # much of one, so "thin" stays a number to read rather than a
                # tier to cross. Instrumentation only -- pv, the lean, the
                # ledger's xw_net and every grade are untouched by these two
                # keys, which is why this carries no MODEL_TAG implication.
                _bf = pd.to_numeric(pr.get("PA"), errors="coerce")
                rec["starter_rate_basis"] = ("measured" if pd.notna(pv)
                                             else "prior_only")
                rec["starter_rate_bf"] = float(_bf) if pd.notna(_bf) else 0.0
            # Regress the starter's xwOBA-allowed toward the league prior by
            # batters faced (PA), so a short-sample starter isn't taken at face
            # value. This shrunk value drives the lean and the SP card display.
            if c == XWOBA_SHRINK_COL and shrink_prior is not None:
                # v4: the starter regresses toward his own history when he has
                # it. `player_prior_one` returns `shrink_prior` unchanged for a
                # rookie, so this is the v3 line for anyone without a record.
                pv = _shrink_one(pv, pd.to_numeric(pr.get("PA"), errors="coerce"),
                                 player_prior_one(pr.get("player_id"),
                                                  shrink_prior), shrink_k)
            ov = a.get(f"opp_{c}")
            if c == XWOBA_SHRINK_COL:
                neutral = a.get("opp_xwOBA_neutral")
                vs_sp = a.get("opp_xwOBA_vs_sp", ov)
                rec["opp_xwOBA_neutral"] = neutral
                rec["opp_xwOBA_vs_sp"] = vs_sp
                rec["platoon_delta_sp"] = a.get("platoon_delta_sp")
                # Diagnostic; carried to the dump so it reaches the ledger.
                # Never read back into a matchup value -- see wsd.
                rec["opp_xwOBA_sd"] = a.get("opp_xwOBA_sd")
                # The generic opp_xwOBA remains the starter-adjusted alias.
                ov = vs_sp
            L = league_baseline.get(c, np.nan)
            rec[f"pit_{c}"] = float(pv) if pd.notna(pv) else np.nan
            rec[f"opp_{c}"] = ov
            rec[f"lg_{c}"] = L
            M = matchup_value(float(pv) if pd.notna(pv) else np.nan,
                              float(ov) if pd.notna(ov) else np.nan, c, L)
            rec[f"mx_{c}"] = float(M) if pd.notna(M) else np.nan
            rec[f"edge_{c}"] = float(M - L) if pd.notna(M) and pd.notna(L) else np.nan
            if c == XWOBA_SHRINK_COL:
                rec["starter_xwOBA"] = rec[f"pit_{c}"]
                rec["mx_xwOBA_sp"] = rec[f"mx_{c}"]
                rec["edge_xwOBA_sp"] = rec[f"edge_{c}"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["game_pk", "side"]).reset_index(drop=True)


def build_xwoba_matchup(pitchers_df, league_baseline):
    prior = league_baseline.get(XWOBA_SHRINK_COL) if USE_XWOBA_SHRINK else None
    pitcher_rows_df, opp_hitters_df = segment_pitcher_blocks(pitchers_df, STATCAST_RATE_COLS)
    opp_lineup_agg_df = aggregate_lineup(opp_hitters_df, STATCAST_RATE_COLS, weighted=USE_WEIGHTED,
                                         shrink_prior=prior, shrink_k=XWOBA_SHRINK_K)
    matchup_df = build_matchup(pitcher_rows_df, opp_lineup_agg_df, STATCAST_RATE_COLS, league_baseline,
                               shrink_prior=prior, shrink_k=XWOBA_SHRINK_K)
    return matchup_df, pitcher_rows_df, opp_hitters_df


def attach_pitch_mix_shadow(matchup_df, pitcher_rows_df, opp_hitters_df,
                            league_baseline, batter_arsenal, pitcher_arsenal):
    """Add the pitch-mix shadow columns. Never touches an existing column.

    The arm re-weights the opposing lineup by the starter's arsenal (see
    `pitch_arsenal`) and records what the starter phase *would* have been. It
    is deliberately inert: `mx_xwOBA`, `edge_xwOBA` and every column the lean
    and the ledger's `xw_net` read are untouched, so a build with the flag on
    and a build with it off produce identical leans. The columns exist so the
    arm can be forward-tested against the record before it is allowed to move
    anything -- a control in the ledger, not a UI element.

    Sides with no arsenal coverage carry a multiplier of exactly 1.0 and a
    shadow phase equal to the real one, which is the honest reading of "no
    information", and keeps the column defined for every row.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    out = matchup_df.copy()
    lg_type = pitch_arsenal.league_by_pitch_type(batter_arsenal)
    cells, overall = pitch_arsenal.batter_relative_cells(batter_arsenal, lg_type)
    L = league_baseline.get(XWOBA_SHRINK_COL, np.nan)

    pit_ids = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, r in pitcher_rows_df.iterrows():
            pid = pd.to_numeric(r.get("player_id"), errors="coerce")
            if pd.notna(pid):
                pit_ids[(r["game_pk"], r["Name"])] = int(pid)

    # Batting order is the row order inside each (game_pk, faced_pitcher)
    # block, so the slot-PA weights line up with the composite's own weighting.
    lineups = {}
    if opp_hitters_df is not None and not opp_hitters_df.empty:
        for key, grp in opp_hitters_df.groupby(["game_pk", "faced_pitcher"],
                                               sort=False):
            rows = []
            for slot, (_, r) in enumerate(grp.iterrows(), start=1):
                pid = pd.to_numeric(r.get("player_id"), errors="coerce")
                rows.append((int(pid) if pd.notna(pid) else None,
                             LINEUP_SLOT_PA.get(slot, LINEUP_SLOT_PA[9])))
            lineups[key] = rows

    mults, covs, ns, bases = [], [], [], []
    for _, row in out.iterrows():
        key = (row["game_pk"], row["pitcher"])
        weights, basis = pitch_arsenal.pitcher_pa_weights(
            pitcher_arsenal, pit_ids.get(key))
        mult, cov, n = pitch_arsenal.lineup_mix_multiplier(
            lineups.get(key, []), weights, cells, overall)
        mults.append(mult)
        covs.append(cov)
        ns.append(n)
        bases.append(basis)
    out["mix_mult"] = mults
    out["mix_coverage"] = covs
    out["mix_hitters"] = ns
    out["mix_basis"] = bases

    b_sp = pd.to_numeric(out.get("opp_xwOBA_vs_sp"), errors="coerce")
    starter = pd.to_numeric(out.get("starter_xwOBA"), errors="coerce")
    out["opp_xwOBA_mix"] = b_sp * out["mix_mult"]
    mx = [matchup_value(b, p, "xwOBA", L)
          for b, p in zip(out["opp_xwOBA_mix"], starter)]
    out["mx_xwOBA_sp_mix"] = mx
    out["edge_xwOBA_sp_mix"] = [
        (m - L) if pd.notna(m) and pd.notna(L) else np.nan for m in mx]
    return out


def sequential_xwoba_phases(b_neutral, b_vs_sp, starter, bullpen, league,
                            expected_sp_ip, sp_bf_per_ip=None, bp_bf_per_ip=None):
    """Return the v10 starter, bullpen, and workload-weighted matchup phases.

    The starter-handedness lineup adjustment belongs only to the starter
    phase. The bullpen phase is neutral until bullpen handedness is modeled.
    A missing bullpen deliberately leaves the full-game matchup undefined.

    The phases are averaged on their share of **plate appearances**, not of
    innings. Both phase values are per-PA xwOBA rates, so the innings share
    v9 used was the wrong denominator: a pitcher who allows more baserunners
    faces more batters per inning, which means the innings share systematically
    underweights the starter exactly when he is the one being hit. Measured
    BF/IP for each phase converts innings to PA. When either rate is
    unavailable the weight falls back to the innings share, which is the same
    number whenever the two rates are equal -- the degradation is continuous,
    not a threshold.
    """
    q = np.nan
    if expected_sp_ip is not None and pd.notna(expected_sp_ip):
        ip_sp = float(np.clip(float(expected_sp_ip), 0.0, GAME_INNINGS))
        ip_bp = GAME_INNINGS - ip_sp
        r_sp, r_bp = _f(sp_bf_per_ip), _f(bp_bf_per_ip)
        if r_sp is not None and r_bp is not None and r_sp > 0 and r_bp > 0:
            pa_sp, pa_bp = ip_sp * r_sp, ip_bp * r_bp
            q = pa_sp / (pa_sp + pa_bp) if (pa_sp + pa_bp) > 0 else np.nan
        else:
            q = ip_sp / GAME_INNINGS
    bp_share = 1.0 - q if pd.notna(q) else np.nan

    mx_sp = matchup_value(b_vs_sp, starter, "xwOBA", league)
    mx_bp = matchup_value(b_neutral, bullpen, "xwOBA", league)
    mx = (
        q * mx_sp + bp_share * mx_bp
        if pd.notna(q) and pd.notna(mx_sp) and pd.notna(mx_bp)
        else np.nan
    )
    return {
        "sp_share": q,
        "bp_share": bp_share,
        "mx_xwOBA_sp": float(mx_sp) if pd.notna(mx_sp) else np.nan,
        "edge_xwOBA_sp": (
            float(mx_sp - league)
            if pd.notna(mx_sp) and league is not None and pd.notna(league)
            else np.nan
        ),
        "mx_xwOBA_bp": float(mx_bp) if pd.notna(mx_bp) else np.nan,
        "edge_xwOBA_bp": (
            float(mx_bp - league)
            if pd.notna(mx_bp) and league is not None and pd.notna(league)
            else np.nan
        ),
        "mx_xwOBA": float(mx) if pd.notna(mx) else np.nan,
        "edge_xwOBA": (
            float(mx - league)
            if pd.notna(mx) and league is not None and pd.notna(league)
            else np.nan
        ),
    }


def apply_pitching_plans(matchup_df, pitching_plans, league_baseline):
    """Apply v9's sequential starter/bullpen construction to each side.

    `build_matchup` has already shrunk the starter and computed M_SP. This step
    computes M_BP with the neutral lineup, then combines matchup values by
    expected innings. `pit_xwOBA` remains a workload-weighted pitching
    diagnostic for legacy consumers; it does not drive the v9 lean.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    out = matchup_df.copy()
    for col, default in (
        ("bullpen_xwOBA", np.nan),
        ("expected_sp_ip", np.nan),
        ("expected_sp_ip_raw", np.nan),
        ("sp_share", np.nan),
        ("bp_share", np.nan),
        ("mx_xwOBA_sp", np.nan),
        ("edge_xwOBA_sp", np.nan),
        ("mx_xwOBA_bp", np.nan),
        ("edge_xwOBA_bp", np.nan),
        ("bullpen_pitchers", np.nan),
        ("bullpen_relief_bf", np.nan),
        ("sp_bf_per_ip", np.nan),
        ("bp_bf_per_ip", np.nan),
        ("opener", False),
        ("opener_reason", None),
        ("opener_confidence", None),
        ("pitching_basis", "starter_only_no_fullgame_lean"),
    ):
        if col not in out.columns:
            out[col] = default

    league = _f((league_baseline or {}).get("xwOBA"))
    for idx, row in out.iterrows():
        key = (int(row["game_pk"]), row["side"])
        plan = (pitching_plans or {}).get(key) or {}
        starter = _f(row.get("starter_xwOBA"))
        if starter is None:
            starter = _f(row.get("pit_xwOBA"))
        bullpen = _f(plan.get("bullpen_xwOBA"))
        expected_ip = _f(plan.get("expected_sp_ip"))
        b_vs_sp = _f(row.get("opp_xwOBA_vs_sp"))
        if b_vs_sp is None:
            b_vs_sp = _f(row.get("opp_xwOBA"))
        b_neutral = _f(row.get("opp_xwOBA_neutral"))
        if b_neutral is None:
            # Legacy/unit-test rows predate the dual composite. Production v9
            # always supplies B_0; this fallback means "no platoon delta."
            b_neutral = b_vs_sp

        out.at[idx, "starter_xwOBA"] = starter
        out.at[idx, "bullpen_xwOBA"] = bullpen
        out.at[idx, "expected_sp_ip"] = expected_ip
        out.at[idx, "expected_sp_ip_raw"] = _f(plan.get("expected_sp_ip_raw"))
        out.at[idx, "bullpen_pitchers"] = plan.get("bullpen_pitchers")
        out.at[idx, "bullpen_relief_bf"] = plan.get("bullpen_relief_bf")
        out.at[idx, "opener"] = bool(plan.get("opener"))
        out.at[idx, "opener_reason"] = plan.get("opener_reason")
        out.at[idx, "opener_confidence"] = plan.get("opener_confidence")
        # v5 abstention. A starter with no leaderboard line contributes his
        # prior, not a measurement, so the starter phase -- and with it this
        # side's edge and the game's lean -- is not something this model
        # measured. Abstain rather than publish a decision built on a default.
        #
        # `starter_xwOBA` still carries the prior: the column records what was
        # used, and blanking it would hide the abstention's own cause. Only the
        # lean-bearing quantities go undefined, through the same NaN-edge path a
        # missing bullpen already takes -- no second suppression mechanism.
        unmeasured_sp = str(row.get("starter_rate_basis") or "") == "prior_only"
        lean_starter = None if unmeasured_sp else starter
        basis = plan.get("pitching_basis")
        if unmeasured_sp:
            basis = "starter_unmeasured_no_lean"
        elif starter is None or bullpen is None or expected_ip is None:
            basis = "starter_only_no_fullgame_lean"
        elif not basis:
            basis = (
                "opener_bullpen_sequential"
                if bool(plan.get("opener"))
                else "starter_bullpen_sequential"
            )
        out.at[idx, "pitching_basis"] = basis

        sp_rate = _f(plan.get("sp_bf_per_ip"))
        bp_rate = _f(plan.get("bp_bf_per_ip"))
        out.at[idx, "sp_bf_per_ip"] = sp_rate
        out.at[idx, "bp_bf_per_ip"] = bp_rate
        phases = sequential_xwoba_phases(
            b_neutral, b_vs_sp, lean_starter, bullpen, league, expected_ip,
            sp_bf_per_ip=sp_rate, bp_bf_per_ip=bp_rate,
        )
        for col, value in phases.items():
            out.at[idx, col] = value

        # A single blended pitching rate is not the v9 matchup input because
        # B_SP and B_0 differ. Keep it only as a diagnostic when both phases
        # are valid; never leave the starter value looking like a nine-inning
        # fallback when the bullpen is unavailable.
        if (lean_starter is not None and bullpen is not None
                and pd.notna(phases["sp_share"])):
            out.at[idx, "pit_xwOBA"] = (
                phases["sp_share"] * lean_starter + phases["bp_share"] * bullpen
            )
        else:
            out.at[idx, "pit_xwOBA"] = np.nan
    return out


# ============================================================
# CELL 5 -- PLATOON / HANDEDNESS SPLIT MATCHUP
# ============================================================
MIN_SPLIT_PA = 30
MIN_PITCHER_SPLIT_BF = 50
K_BAT = 100
K_PIT = 200
K0 = 200
# OPP / effective_stand / platoon_advantage live in CELL 4 -- one handedness
# convention shared by the xwOBA adjustment and this lens.


def build_platoon_matchup(pitcher_rows_df, opp_hitters_df, people,
                          player_splits_hit, player_splits_pit,
                          league_splits=None, league_people=None):
    # League OPS cells prefer the full hitter split population (fetch_all with
    # FULL_LEAGUE_PLATOON_BASELINES); fall back to today's lineups if absent.
    _league_split_source = league_splits if league_splits else player_splits_hit
    _league_people_source = league_people if league_people else people
    _pmeta = pitcher_rows_df.set_index(["game_pk", "Name"])

    def _pitcher_attr(gpk, name, attr):
        try:
            v = _pmeta.loc[(gpk, name), attr]
            return v.iloc[0] if isinstance(v, pd.Series) else v
        except KeyError:
            return None

    def _hit_split(pid, hand):
        return (player_splits_hit.get(int(pid), {}) or {}).get(hand, {}) if pd.notna(pid) else {}

    def _pit_split(pid, hand):
        return (player_splits_pit.get(int(pid), {}) or {}).get(hand, {}) if pd.notna(pid) else {}

    def _overall(splits, pid):
        o = (splits.get(int(pid), {}) or {}).get("overall", {}) or {} if pd.notna(pid) else {}
        return o.get("ops"), (o.get("pa") or 0)

    def _shrink(obs, n, overall, overall_pa, Lcell, Loverall, K):
        # NaN is truthy, so a bare `if Lcell`/`Loverall` would let an empty
        # (early-season) league cell propagate a NaN prior; guard explicitly.
        Lcell_ok = pd.notna(Lcell) and Lcell
        Loverall_ok = pd.notna(Loverall) and Loverall
        if overall is not None and not pd.isna(overall) and Loverall_ok:
            op = overall_pa or 0
            overall_reg = (op * overall + K0 * Loverall) / (op + K0)
            prior = overall_reg * (Lcell / Loverall) if Lcell_ok else overall_reg
        else:
            prior = Lcell if Lcell_ok else Loverall
        if obs is None or pd.isna(obs):
            return prior
        n = n or 0
        return (n * obs + K * prior) / (n + K) if (n + K) > 0 else prior

    def _bats_of(pid):
        return (_league_people_source.get(int(pid), {}) or {}).get("bats") if pd.notna(pid) else None

    def _compute_league_ops_cells():
        buckets = {("L", "L"): [], ("L", "R"): [], ("R", "L"): [], ("R", "R"): []}
        allv = []
        for pid, sp in _league_split_source.items():
            b = (_bats_of(pid) or "")[:1].upper()
            for T in ("L", "R"):
                s = sp.get(T, {}) or {}
                ops, pa = s.get("ops"), s.get("pa") or 0
                if ops is None or (isinstance(ops, float) and np.isnan(ops)) or pa <= 0:
                    continue
                eff = effective_stand(b, T)
                buckets[(eff, T)].append((ops, pa)); allv.append((ops, pa))
        Lc = {}
        for k, v in buckets.items():
            if v:
                o = np.array([x[0] for x in v]); w = np.array([x[1] for x in v], float)
                Lc[k] = float(np.average(o, weights=w))
        overall = (float(np.average([x[0] for x in allv],
                                    weights=[x[1] for x in allv]))
                   if allv else np.nan)
        return Lc, overall

    league_ops_cell, league_ops_overall = _compute_league_ops_cells()

    rows = []
    for _, h in opp_hitters_df.iterrows():
        gpk, fp = h["game_pk"], h["faced_pitcher"]
        T = _pitcher_attr(gpk, fp, "throws")
        if T not in ("L", "R"):
            continue
        bats = (h.get("bats") or "")[:1].upper()
        pid = h.get("player_id")
        pid_p = _pitcher_attr(gpk, fp, "player_id")
        eff_stand = effective_stand(bats, T)
        L = league_ops_cell.get((eff_stand, T), league_ops_overall)
        Lo = league_ops_overall
        B_raw = _hit_split(pid, T).get("ops"); pa_b = _hit_split(pid, T).get("pa") or 0
        P_raw = (_pit_split(pid_p, eff_stand) or {}).get("ops")
        bf_p = (_pit_split(pid_p, eff_stand) or {}).get("pa") or 0
        ov_b, ov_b_pa = _overall(player_splits_hit, pid)
        ov_p, ov_p_pa = _overall(player_splits_pit, pid_p)
        B = _shrink(B_raw, pa_b, ov_b, ov_b_pa, L, Lo, K_BAT)
        P = _shrink(P_raw, bf_p, ov_p, ov_p_pa, L, Lo, K_PIT)
        Mi = (B * P / L) if (B is not None and P is not None and L) and not (
            pd.isna(B) or pd.isna(P)) else np.nan
        rows.append({
            "game_pk": gpk, "matchup": h.get("matchup"), "faced_pitcher": fp,
            BATTING_ORDER_COL: h.get(BATTING_ORDER_COL),
            "pitcher_throws": T, "batter": h["Name"], "bats": bats or "?",
            "eff_stand": eff_stand, "platoon_adv": platoon_advantage(bats, T),
            "ops_vs_hand_raw": B_raw, "ops_vs_hand": float(B) if pd.notna(B) else np.nan,
            "pit_ops_allowed_raw": P_raw,
            "pit_ops_allowed": float(P) if pd.notna(P) else np.nan,
            "lg_cell": L, "mx_ops": float(Mi) if pd.notna(Mi) else np.nan,
            "split_pa": pa_b, "pit_split_bf": bf_p,
            "low_sample": pa_b < MIN_SPLIT_PA,
            "pit_low_sample": bf_p < MIN_PITCHER_SPLIT_BF,
        })
    opp_platoon_detail_df = pd.DataFrame(rows)

    def _wmean(v, w):
        v = pd.to_numeric(pd.Series(v).reset_index(drop=True), errors="coerce")
        w = pd.to_numeric(pd.Series(w).reset_index(drop=True), errors="coerce")
        m = v.notna() & w.notna() & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    if opp_platoon_detail_df.empty:
        return pd.DataFrame(), opp_platoon_detail_df, league_ops_overall

    plat_rows = []
    for (gpk, fp), g in opp_platoon_detail_df.groupby(["game_pk", "faced_pitcher"], sort=False):
        T = g["pitcher_throws"].iloc[0]
        # All aggregates are lineup-composition weighted by expected PA per
        # batting-order slot (top of the order carries more in-game exposure).
        # Fall back to season split PA (clip 0-PA splits to 1 so prior-driven
        # bats still count as lineup exposure, incl. SP OPS display) wherever the
        # batting order is unavailable.
        lineup_w = lineup_weight(g)
        if lineup_w is None or not pd.Series(lineup_w).notna().any():
            lineup_w = pd.to_numeric(g["split_pa"], errors="coerce").fillna(0).clip(lower=1)
        opp_ops_raw = _wmean(g["ops_vs_hand_raw"], lineup_w)
        opp_ops = _wmean(g["ops_vs_hand"], lineup_w)
        pit_ops_raw = _wmean(g["pit_ops_allowed_raw"], lineup_w)
        pit_ops = _wmean(g["pit_ops_allowed"], lineup_w)
        mx_ops = _wmean(g["mx_ops"], lineup_w)
        edge = mx_ops - league_ops_overall if pd.notna(mx_ops) else np.nan
        bf_present = [v for v in g.loc[g["pit_split_bf"] > 0, "pit_split_bf"].unique()]
        pit_min_bf = int(min(bf_present)) if bf_present else 0
        pit_low = bool(g["pit_low_sample"].any())
        meta_row = opp_hitters_df[(opp_hitters_df.game_pk == gpk) &
                                  (opp_hitters_df.faced_pitcher == fp)].iloc[0]
        opp_team = meta_row["home_team"] if meta_row["pitcher_side"] == "away" else meta_row["away_team"]
        plat_rows.append({
            "game_pk": gpk, "game_date": meta_row.get("game_date"), "matchup": meta_row["matchup"],
            "side": meta_row["pitcher_side"], "pitcher": fp, "throws": T, "opp_team": opp_team,
            "n_opp": len(g),
            "n_LHB": int((g["bats"] == "L").sum()), "n_RHB": int((g["bats"] == "R").sum()),
            "n_SW": int((g["bats"] == "S").sum()),
            "n_platoon_adv": int(g["platoon_adv"].sum()),
            "n_low_sample": int(g["low_sample"].sum()),
            "pit_min_split_bf": pit_min_bf, "pit_low_sample": pit_low,
            "reliable": (not pit_low) and (int(g["low_sample"].sum()) <= 4),
            "opp_OPS_raw": float(opp_ops_raw) if pd.notna(opp_ops_raw) else np.nan,
            "opp_OPS_vs_hand": float(opp_ops) if pd.notna(opp_ops) else np.nan,
            "pit_OPS_raw": float(pit_ops_raw) if pd.notna(pit_ops_raw) else np.nan,
            "pit_OPS_allowed": float(pit_ops) if pd.notna(pit_ops) else np.nan,
            "mx_OPS": float(mx_ops) if pd.notna(mx_ops) else np.nan,
            "edge_OPS": float(edge) if pd.notna(edge) else np.nan,
        })
    matchup_platoon_df = (pd.DataFrame(plat_rows)
                          .sort_values(["game_pk", "side"]).reset_index(drop=True))
    return matchup_platoon_df, opp_platoon_detail_df, league_ops_overall


# ============================================================
# PREGAME MARKET -- best-effort DK odds via ESPN
# Same endpoints/join as market_backfill.py (scoreboard -> core
# /odds, provider 100), but *pregame*: current + open moneylines
# and the total, devigged home implied %. Strictly best-effort:
# any failure logs and omits -- an odds outage never fails the
# build, and missing values render as em-dashes, never defaults.
# ============================================================
_ESPN2SA = {"CHW": "CWS", "ARI": "AZ", "OAK": "ATH"}  # ESPN -> StatsAPI abbrev


def _amer_ml(x):
    if isinstance(x, dict):
        x = x.get("american")
    try:
        return int(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _imp_ml(ml):
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def _parse_espn_dt(s):
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except (TypeError, ValueError):
            continue
    return None


def fetch_pregame_odds(slate_df):
    """game_pk -> {away_ml, home_ml, open_away_ml, open_home_ml, total, p_home}."""
    from fetch_headers import FETCH_HEADERS
    out = {}
    try:
        ds = SLATE_DATE.replace("-", "")
        sb = _get_json("https://site.api.espn.com/apis/site/v2/sports/baseball/"
                       f"mlb/scoreboard?dates={ds}", tries=2,
                       headers=FETCH_HEADERS)
    except Exception as e:  # noqa: BLE001
        log(f"pregame odds: scoreboard fetch failed ({e!r}) -> strip renders em-dashes")
        return out
    evs = {}
    for ev in sb.get("events", []):
        try:
            comp = ev["competitions"][0]
            t = {c["homeAway"]: _ESPN2SA.get(c["team"]["abbreviation"],
                                             c["team"]["abbreviation"])
                 for c in comp["competitors"]}
            tid = {str(c["team"]["id"]): c["homeAway"] for c in comp["competitors"]}
            evs.setdefault((t["away"], t["home"]), []).append(
                dict(eid=ev["id"], start=ev.get("date"), tid=tid))
        except Exception:  # noqa: BLE001
            continue

    def _team_odds(dk, which):
        td = dk.get(which) or {}
        cur = _amer_ml(((td.get("current") or {}).get("moneyLine")))
        if cur is None:
            cur = _amer_ml(td.get("moneyLine"))
        opn = _amer_ml(((td.get("open") or {}).get("moneyLine")))
        return cur, opn

    for _, g in slate_df.iterrows():
        cands = evs.get((g.get("away_abbrev"), g.get("home_abbrev")), [])
        if not cands:
            continue
        pick = cands[0]
        if len(cands) > 1:  # doubleheader: nearest scheduled start
            gt = _parse_espn_dt(str(g.get("game_datetime_utc") or ""))
            if gt is not None:
                def _gap(e):
                    et_ = _parse_espn_dt(str(e.get("start") or ""))
                    return abs((et_ - gt).total_seconds()) if et_ else 1e12
                pick = min(cands, key=_gap)
        try:
            odds = _get_json("https://sports.core.api.espn.com/v2/sports/baseball/"
                             f"leagues/mlb/events/{pick['eid']}/competitions/"
                             f"{pick['eid']}/odds", tries=2,
                             headers=FETCH_HEADERS)
        except Exception as e:  # noqa: BLE001
            log(f"pregame odds: game_pk={g['game_pk']} odds fetch failed ({e!r})")
            continue
        dk = next((i for i in odds.get("items", [])
                   if str((i.get("provider") or {}).get("id")) == "100"), None)
        if dk is None:
            continue
        h_cur, h_opn = _team_odds(dk, "homeTeamOdds")
        a_cur, a_opn = _team_odds(dk, "awayTeamOdds")
        try:
            total = float(dk.get("overUnder")) if dk.get("overUnder") is not None else None
        except (TypeError, ValueError):
            total = None
        p_home = None
        if h_cur is not None and a_cur is not None:
            ih, ia = _imp_ml(h_cur), _imp_ml(a_cur)
            p_home = ih / (ih + ia) if (ih + ia) > 0 else None
        # F5 (1st 5 innings) DK ML from the same event's propBets child, using
        # the validated parser from market_backfill; pregame `value` is the
        # current F5 price, `open` its opener. Best-effort: any failure omits
        # F5 and the strip renders em-dashes. Devig is CONDITIONAL on a decided
        # half (DK F5 ties push).
        f5a = f5h = f5a_opn = f5h_opn = f5_p_home = None
        try:
            from market_backfill import _espn_f5
            f5 = _espn_f5(pick["eid"], pick.get("tid") or {})
            if f5:
                f5a, f5h = f5["f5_close_away_ml"], f5["f5_close_home_ml"]
                f5a_opn, f5h_opn = f5["f5_open_away_ml"], f5["f5_open_home_ml"]
                if f5a is not None and f5h is not None:
                    ih5, ia5 = _imp_ml(f5h), _imp_ml(f5a)
                    f5_p_home = ih5 / (ih5 + ia5) if (ih5 + ia5) > 0 else None
        except Exception as e:  # noqa: BLE001
            log(f"pregame odds: game_pk={g['game_pk']} F5 fetch failed ({e!r})")
        out[g["game_pk"]] = dict(away_ml=a_cur, home_ml=h_cur,
                                 open_away_ml=a_opn, open_home_ml=h_opn,
                                 total=total, p_home=p_home,
                                 f5_away_ml=f5a, f5_home_ml=f5h,
                                 f5_open_away_ml=f5a_opn, f5_open_home_ml=f5h_opn,
                                 f5_p_home=f5_p_home)
        time.sleep(0.15)
    log(f"pregame odds attached: {len(out)}/{len(slate_df)} games")
    return out


def fetch_current_streaks(slate_df=None):
    """team_id -> current streak code such as 'W3' or 'L1'.

    Display-only and strictly best-effort: any failure logs and returns an
    empty map so the card header simply omits the streak rather than failing
    the build. `slate_df` is accepted for signature symmetry with the odds
    fetch; standings cover all teams regardless of the slate.
    """
    out = {}
    try:
        js = _get_json("https://statsapi.mlb.com/api/v1/standings",
                       {"leagueId": "103,104", "season": SEASON,
                        "date": SLATE_DATE, "standingsTypes": "regularSeason"},
                       tries=2)
    except Exception as e:  # noqa: BLE001
        log(f"current streaks: standings fetch failed ({e!r}) -> header omits")
        return out
    for grp in js.get("records", []):
        for tr in grp.get("teamRecords", []):
            tid = (tr.get("team") or {}).get("id")
            if tid is None:
                continue
            code = str((tr.get("streak") or {}).get("streakCode") or "").upper()
            if not re.fullmatch(r"[WL]\d+", code):
                continue
            try:
                out[int(tid)] = code
            except (TypeError, ValueError):
                continue
    log(f"current streaks attached: {len(out)} teams")
    return out


# ============================================================
# CELL 6 -- CARD RENDER (per-hitter lineup layout, emits HTML)
# ============================================================
ABBR = {
    "Arizona Diamondbacks": "AZ", "Athletics": "ATH", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

# Compact scoreboard labels. The schedule feed supplies official full names,
# while a slate row reads more cleanly with the familiar club name beside the
# cap logo (and the abbreviation remains the fallback for degraded/test data).
TEAM_LABELS = {
    "Arizona Diamondbacks": "D-backs",
    "Athletics": "Athletics",
    "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros",
    "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins",
    "New York Mets": "Mets",
    "New York Yankees": "Yankees",
    "Philadelphia Phillies": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres",
    "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners",
    "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Washington Nationals": "Nationals",
}

# ---------- heatmap tints (casual-UI layer; numbers unchanged) ----------
HEAT_ALPHA_MAX = 0.30
# |dev| from league at full colour saturation. Calibrated on the xwOBA spread
# and NOT re-derived for wOBA -- recorded here rather than patched, because one
# slate is not a distribution. Measured on 2026-08-03, the only day built under
# both metrics (n=13 xwOBA sides vs 14 wOBA): the starter-allowed rate widens
# materially (sd 0.0161 -> 0.0215, mean |dev| 0.0121 -> 0.0166), because it is
# one pitcher's observed outcome rather than an expectation, so xwOBA_sp=0.035
# saturates sooner -- 14% of starter cells fully tinted against 8%, mean
# saturation 0.34 -> 0.44. The lineup composite barely moves (sd 0.0089 both
# ways): nine shrunk hitters average the extra variance away, so xwOBA_bat is
# roughly still in calibration. Display-only, no MODEL_TAG implication. Widen
# xwOBA_sp only off a real wOBA sample, not off this one.
HEAT_DOMAINS = {"xwOBA_sp": 0.035, "K-BB%": 7.0,
                "OPS": 0.080, "ERA": 1.50, "xwOBA_bat": 0.045}


def heat_style(val, lg, domain, hi="warm"):
    """Cell tint vs league. hi = token when value is ABOVE league:
    'warm' (offense-favorable) or 'cool' (pitcher-favorable)."""
    v, L = _f(val), _f(lg)
    if v is None or L is None:
        return ""
    a = clamp(abs(v - L) / domain, 0.0, 1.0) * HEAT_ALPHA_MAX
    if a < 0.04:          # dead zone: ~league-average stays untinted
        return ""
    tok = hi if v > L else ("cool" if hi == "warm" else "warm")
    return f"background:rgba(var(--{tok}),{a:.2f})"


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def f3(v):
    return "—" if v is None else f"{v:.3f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.3f}"


def delta3(v):
    """Three-place delta display without rendering a nonzero value as zero."""
    if v is None or pd.isna(v):
        return "—"
    value = abs(float(v))
    if value > 0 and f"{value:.3f}" == "0.000":
        return "&lt;0.001"
    return f"{value:.3f}"


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def _abbr(name):
    return ABBR.get(name, str(name or "")[:3].upper())


def ledger_abbr(model_abbr):
    """Translate THIS module's club abbreviation into the ledger's namespace.

    Two abbreviation namespaces meet in this repo and exactly one club differs:
    the model's display map follows StatsAPI (`AZ`) while `grade_leans.ABBR`
    persists ESPN/ledger-style `ARI`. Every other field crosses that boundary
    as a full team name and is abbreviated by grade_leans' own map, so this is
    the only place the two can disagree.

    It is a named function rather than an inline conditional because it now has
    two callers, and a second copy of the alias is how `hybrid_selection` came
    to be written in the wrong namespace in the first place. Anything this
    module writes for the LEDGER goes through here; anything it renders on a
    card does not, because a card is internally consistent in the model's own
    namespace.
    """
    return "ARI" if model_abbr == "AZ" else model_abbr


def _fmt_ml(v):
    if v is None:
        return "—"
    return f"+{v}" if v > 0 else str(v)


def _fmt_pt_clock(dt):
    """Bare wall clock for a game row -- no zone suffix.

    Every clock on the page is Pacific and the page says so once, in the
    footer stamp (`_legend_head`), rather than repeating ` PT` on every card.
    The build stamp dropped its suffix too -- one zone, stated once, is the
    whole convention. Delete that clause and every time on the page becomes
    ambiguous."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _game_time_pt(iso_utc):
    dt = _parse_espn_dt(str(iso_utc or ""))
    if dt is None:
        return ""
    return _fmt_pt_clock(dt.replace(tzinfo=UTC).astimezone(PT))


def _f(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows_by_side(gg):
    a = gg[gg["side"] == "away"]; h = gg[gg["side"] == "home"]
    return (a.iloc[0] if len(a) else None), (h.iloc[0] if len(h) else None)


def _pl_lookup(pl_df):
    out = {}
    if pl_df is None or getattr(pl_df, "empty", True):
        return out
    for _, r in pl_df.iterrows():
        out[(r["game_pk"], r["side"])] = r
    return out


def _hitters_for(opp_hitters_df, detail_df, gpk, fp, lg_ops,
                 ref_bat=None, prior=None, k_bat=None):
    """Batting-order rows for the lineup this SP faces. Row order in
    opp_hitters_df IS lineup order -- never sort. Detail (platoon) join is by
    (game_pk, faced_pitcher, batter name); a hitter with no vs-hand data keeps
    the xwOBA cell and renders em-dashes for the platoon columns.

    Each row also carries xw_pctile: the hitter's season xwOBA, shrunk by his
    PA and ranked against the qualified-regular reference (display-only).

    The platoon marker reads the hitter row's own `sp_platoon_adv` tag -- the
    same tag that moved his xwOBA in the lean -- so the card still marks the
    adjusted bats on a build where the platoon-OPS lens abstained."""
    rows = []
    if opp_hitters_df is None or getattr(opp_hitters_df, "empty", True):
        return rows
    H = opp_hitters_df[(opp_hitters_df["game_pk"] == gpk)
                       & (opp_hitters_df["faced_pitcher"] == fp)]
    D = {}
    if detail_df is not None and not getattr(detail_df, "empty", True):
        dd = detail_df[(detail_df["game_pk"] == gpk)
                       & (detail_df["faced_pitcher"] == fp)]
        for _, r in dd.iterrows():
            D[r["batter"]] = r
    for _, r in H.iterrows():
        d = D.get(r["Name"])
        mx = _f(d["mx_ops"]) if d is not None else None
        edge = (mx - lg_ops) if (mx is not None and lg_ops is not None
                                 and pd.notna(lg_ops)) else None
        xw_raw = _f(r.get("xwOBA"))
        backfill_value = r.get(MODEL_RATE_TEAM_BACKFILL_COL)
        team_backfill = bool(backfill_value) if pd.notna(backfill_value) else False
        adv_tag = r.get(PLATOON_ADV_COL)
        if adv_tag is None or not pd.notna(adv_tag):
            adv_tag = d["platoon_adv"] if d is not None else None
        rows.append(dict(
            name=r["Name"], player_id=r.get("player_id"),
            pos=str(r.get("Pos.") or ""),
            bats=(str(r.get("bats") or ""))[:1].upper(),
            xw=xw_raw,
            xw_team_backfill=team_backfill,
            xw_pctile=(
                pctile_rank_raw(xw_raw, ref_bat)
                if team_backfill
                else pctile_rank(xw_raw, _f(r.get("PA")), ref_bat, prior, k_bat,
                                 pid=r.get("player_id"))
            ),
            adv=bool(adv_tag) if adv_tag is not None and pd.notna(adv_tag) else False,
            ops=_f(d["ops_vs_hand"]) if d is not None else None,
            pa=int(d["split_pa"] or 0) if d is not None else 0,
            low=bool(d["low_sample"]) if d is not None else False,
            mx=mx, edge=edge,
        ))
    return rows


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _last(name):
    """Surname for compact spotlight pills: the last name token, skipping a
    trailing generational suffix so 'Fernando Tatis Jr.' -> 'Tatis' and
    'Michael Harris II' -> 'Harris' rather than 'Jr.'/'II'."""
    parts = str(name or "").split()
    while len(parts) > 1 and parts[-1].lower().strip(".") in _NAME_SUFFIXES:
        parts = parts[:-1]
    return parts[-1] if parts else str(name or "")


def _mlb_player_link(name, player_id, label=None):
    """Official MLB player-page link, falling back to plain text without an ID.

    MLB's stable ID-only route redirects to the current canonical name slug, so
    trades, accents, suffixes, and name changes do not require URL maintenance.
    """
    text = _esc(name if label is None else label)
    try:
        pid = int(player_id)
    except (TypeError, ValueError, OverflowError):
        return text
    return (f"<a class='player-link' href='https://www.mlb.com/player/{pid}' "
            f"target='_blank' rel='noopener noreferrer'>{text}</a>")


def _tier_word(pctile):
    """(label, css-class) plain-word starter quality from an xwOBA-against
    percentile. Class picks the accent: elite=steel, below=ember, else neutral."""
    if pctile is None:
        return None, ""
    if pctile >= 75:
        return "elite", "elite"
    if pctile >= 45:
        return "solid", "solid"
    return "below avg", "below"


def _pct_bar(pctile, kind):
    """Percentile slider (0-100). kind 'h' (hitter, ember) or 'p' (pitcher,
    steel). Fill length is the percentile; hue carries whose favor it is. The
    bar is the whole reading -- the numeric percentile is deliberately not
    printed beside it. An empty track means no percentile was available."""
    if pctile is None:
        return "<span class='sl na'></span>"
    w = max(0.0, min(100.0, float(pctile)))
    return f"<span class='sl {kind}'><i style='width:{w:.0f}%'></i></span>"


def _spotlight_html(hitters, n=3, thresh=70):
    """Top bats in a lineup by xwOBA percentile -- the 'highlight the leaders'
    ask, on a quality stat. Shows up to n at/above thresh; empty otherwise."""
    ranked = sorted((h for h in hitters if h.get("xw_pctile") is not None),
                    key=lambda h: h["xw_pctile"], reverse=True)
    top = [h for h in ranked[:n] if h["xw_pctile"] >= thresh]
    if not top:
        return ""
    pills = "".join(f"<span class='pill'>"
                    f"{_mlb_player_link(h['name'], h.get('player_id'), _last(h['name']))}"
                    f"</span>" for h in top)
    return f"<div class='spot'><span class='sl-lab'>Standouts</span>{pills}</div>"


def _grade_word(edge):
    """Qualitative grade of an offense's xwOBA edge vs league, for the read."""
    if edge is None:
        return None, ""
    if edge >= 0.020:
        return "well above average", "warmtx"
    if edge >= 0.006:
        return "above average", "warmtx"
    if edge > -0.006:
        return "about average", ""
    if edge > -0.020:
        return "below average", "cooltx"
    return "well below average", "cooltx"


def _read_sentence(away_abbr, home_abbr, a, h, fav, strength_word):
    """One plain-language line under the header explaining the lean. a = away
    SP row, so a['xw_edge'] is the HOME offense edge and h -> AWAY offense."""
    home_edge, away_edge = a.get("xw_edge"), h.get("xw_edge")
    if home_edge is None or away_edge is None or fav is None:
        return ""
    # The away offense faces the HOME starter (h['p']); the home offense faces
    # the AWAY starter (a['p']). Edges were resolved to offenses above.
    ga, ca = _grade_word(away_edge)
    gh, ch = _grade_word(home_edge)
    aw = f"<b class='{ca}'>{ga}</b>" if ca else f"<b>{ga}</b>"
    hw = f"<b class='{ch}'>{gh}</b>" if ch else f"<b>{gh}</b>"
    home_sp = _mlb_player_link(h["p"], h.get("player_id"), _last(h["p"]))
    away_sp = _mlb_player_link(a["p"], a.get("player_id"), _last(a["p"]))
    away_team = _esc(away_abbr)
    home_team = _esc(home_abbr)
    return (f"<p class='read'>{away_team}'s bats grade {aw} against "
            f"{home_sp}; {home_team}'s grade {hw} against "
            f"{away_sp}. That is a <b>{strength_word or 'lean'}</b> "
            f"lean to <b>{_esc(fav)}</b>.</p>")


def _lean_implied_p(odds, fav, away_abbr, home_abbr):
    """Devigged closing-style probability for the LEANED side, or None.

    Prefers the devigged `p_home` the market feed already supplies; falls back
    to devigging the two moneylines itself so a game with prices but no
    `p_home` still resolves to a hybrid branch rather than silently losing its
    signal.

    That fallback calls `_imp_ml` rather than restating it. It used to carry
    its own copy of the American-odds conversion -- written as
    `abs(hm)/(abs(hm)+100)` where `_imp_ml` writes `-ml/(-ml+100)`, identical
    for every price the guard below admits. Two copies of one statistic in a
    single module will drift, and the reader cannot see which they are
    reading; the moment to merge them is while they still agree, which was
    verified exhaustively over the full American range before this fold.
    `_imp_ml` is the one derivation, shared with `fetch_pregame_odds` and
    mirrored by `market_backfill._imp` for the closing line.
    """
    if not odds or fav is None:
        return None
    ph = odds.get("p_home")
    if ph is None or pd.isna(ph):
        hm, am = odds.get("home_ml"), odds.get("away_ml")
        if hm is None or am is None:
            return None
        try:
            hm, am = float(hm), float(am)
        except (TypeError, ValueError):
            return None
        # American odds never fall strictly between -100 and +100. The guard
        # stays ahead of `_imp_ml`, which has no opinion on whether its
        # argument is a real price -- and it is what keeps 0 out, where the
        # two spellings of the negative branch would have differed.
        if (-100 < hm < 100) or (-100 < am < 100):
            return None
        ih, ia = _imp_ml(hm), _imp_ml(am)
        if ih + ia <= 0:
            return None
        ph = ih / (ih + ia)
    ph = float(ph)
    if not (0.0 < ph < 1.0):
        return None
    return ph if fav == home_abbr else 1.0 - ph


def _model_version_short():
    """Compact active-model label for the per-game panel (for example, V12)."""
    m = re.search(r"_v(\d+)$", MODEL_TAG, flags=re.IGNORECASE)
    return f"V{m.group(1)}" if m else MODEL_TAG


# Reference bar for the branch read. The hybrid publishes TWO branches rather
# than the 21 cells this replaces, so the family-wise correction that grid
# needed (|z| >= 2.7 across 15 draws) is no longer the right bar: at two
# branches the ~0.05 family-wise threshold is |z| ~ 2.2. Stated as a REFERENCE
# and not a gate -- the number and its spread always print, only the sentence
# beside them changes.
_BRANCH_FAMILYWISE_Z = 2.2


def _branch_read(parts):
    """One honest sentence about what a branch's excess can support.

    Moves continuously with the standard error rather than switching on a row
    count, which is the threshold-cliff fix this repo has now applied four
    times. It returns the VERDICT only -- the sample size is carried by the
    block heading above it, and printing `n` twice in six lines was one of the
    things making this panel hard to read.

    Note what it does NOT say: nothing here calls a branch profitable, because
    a branch clearing its bar on the discovery sample is a statement about rows
    the rule was fitted on.
    """
    se, excess = parts.get("excess_se"), parts.get("excess")
    try:
        ok = se is not None and np.isfinite(se) and se > 0
    except TypeError:
        ok = False
    if not ok:
        return "spread unavailable"
    z = abs(float(excess) / float(se))
    if z < _BRANCH_FAMILYWISE_Z:
        return "within noise"
    return "outside noise across both branches"


def _excess_pm(parts):
    """` ± X.X` for a published excess, or "" when no spread can be formed.

    Every excess on this site prints with one of these. `_excess_se` sizes it
    at the market's own prices under the null that each game settles at its
    close, so it is defined at n=1 and can never collapse to ±0.0 the way a
    p-hat form does on an all-W or all-L branch.
    """
    se = parts.get("excess_se")
    try:
        if se is None or not np.isfinite(se):
            return ""
    except TypeError:
        return ""
    return f" ± {100 * float(se):.1f}"


def _branch_history(ctx, action):
    """Track record of the branch this game's price puts it in.

    Every line here describes PAST games, and the wording has to make that
    impossible to misread. The version this replaces led with
    "Selection won  73.3%" directly under tonight's teams, which reads as this
    pick's win probability -- it is the historical rate of past picks in the
    same branch, and the panel publishes no per-game probability at all. The
    count therefore moves into the heading, every value row is past tense, and
    the two percentages that used to sit unlabelled beside each other (this
    game's 37.4% no-vig against the branch's 58.6% average price) are now
    named for what they are.
    """
    if not action:
        # The rule line already says the rule abstains and why; a second line
        # restating it is the redundancy this rewrite is removing.
        return ""
    history_branch = ("model-side" if action == "FOLLOW"
                      else "market-favorite")
    parts = (ctx or {}).get(("branch", action))
    if not parts:
        return ("<div class='vline hist'><span class='vk'>Track record:</span>"
                f"<span>No completed {_model_version_short()} {history_branch} selections "
                "yet.</span></div>")

    version = _model_version_short()
    n = parts["n"]
    rows = [
        ("Won", f"{parts['w']}-{parts['l']} ({100 * parts['actual']:.1f}%)"),
        ("Their average price", f"{100 * parts['implied']:.1f}% implied"),
        ("Beat that price by",
         f"<b>{100 * parts['excess']:+.1f} pp</b>{_excess_pm(parts)}"
         f" · {_esc(_branch_read(parts))}"),
    ]
    # The control, and a sentence saying what it means HERE. Adjacency alone
    # was not enough: on FADE the two lines are identical to the decimal, and
    # without a clause saying why, a reader sees duplicated data or a bug
    # rather than the point -- that this branch has no model content.
    chalk = (ctx or {}).get(("chalk", action))
    note = ""
    if chalk:
        rows.append(("Always chalk, same games",
                     f"{chalk['w']}-{chalk['l']} "
                     f"({100 * chalk['actual']:.1f}%) · "
                     f"{100 * chalk['excess']:+.1f} pp"))
        if action == "FADE":
            note = ("Identical by construction: backing the other side of a "
                    "lean priced under "
                    f"{100 * HYBRID_THRESHOLD:.0f}% always lands on the "
                    "favourite, so this branch IS the chalk bet.")
        else:
            note = ("Chalk is the yardstick: it backs the favourite on these "
                    "same games, whatever the model said.")
    body = "".join(
        f"<div class='vline'><span class='vk'>{k}</span><span>{v}</span></div>"
        for k, v in rows)
    # Directly under the chalk row, before anything else: a note explaining a
    # row two rows above it is a note the reader has to hunt for.
    if note:
        body += f"<div class='vnote'>{note}</div>"
    pooled = (ctx or {}).get("pooled")
    if pooled:
        body += (f"<div class='vline'><span class='vk'>Every {version} pick"
                 f"</span><span>{100 * pooled['excess']:+.1f} pp"
                 f"{_excess_pm(pooled)} vs price · n={pooled['n']}</span></div>")
    return (
        "<div class='vprofile'>"
        f"<div class='vprofile-title'>Past {version} {history_branch} selections · "
        f"{n} completed {'game' if n == 1 else 'games'}</div>"
        "<div class='vprofile-band'>Track record of this branch — "
        "not a prediction for this game, and not a forward test</div>"
        f"{body}"
        "</div>"
    )


def _verdict_html(fav, odds, away_abbr, home_abbr, ctx=None, delta=None):
    """Per-game panel: the model's lean, the price, and the rule's selection.

    The site's one published decision surface. It is laid out in two labelled
    zones -- what is true of THIS game, then the track record of the branch it
    landed in -- because every clarity problem this panel has had came from a
    reader carrying a historical rate down onto tonight's teams.

    Two things it deliberately does not do. It never calls a selection a bet:
    the copy carries no betting language and a test pins that. And it never
    hides the always-chalk control, because the FADE branch is chalk-identical
    by construction and says so in words rather than by adjacency alone.

    A live card uses the current pregame price while history is bucketed on
    closes, so a game can cross the threshold before first pitch.
    """
    ctx = ctx or {}
    if fav is None:
        return ("<div class='verdict'><div class='l'>Model vs market</div>"
                "<div class='vt'>No model lean — the rule abstains.</div></div>")

    p_lean = _lean_implied_p(odds, fav, away_abbr, home_abbr)
    action = hybrid_action(p_lean)
    pick = hybrid_selection(fav, away_abbr, home_abbr, p_lean)
    d = _f(delta)
    delta_txt = f"{abs(d):.4f}".lstrip("0") if d is not None else "—"
    strength = _delta_label(delta) or "—"

    odds = odds or {}
    price = odds.get("home_ml") if fav == home_abbr else odds.get("away_ml")
    price_txt = _fmt_ml(price) if price is not None else "—"
    p_txt = (f"{100 * p_lean:.1f}% no-vig" if p_lean is not None
             else "no no-vig price yet")

    thr = f"{100 * HYBRID_THRESHOLD:.0f}%"
    if action is None:
        rule_line = ("<div class='vline'><span class='vk'>Rule</span>"
                     "<span>abstains — no two-sided price yet</span></div>")
    else:
        sel_price = None
        if pick is not None:
            sel_price = (odds.get("home_ml") if pick == home_abbr
                         else odds.get("away_ml"))
        # The reason, in plain words, on the line that carries the decision.
        # Without it a reader has to reverse-engineer the threshold from two
        # numbers printed above -- and on a FADE the selected club appears
        # nowhere else on the panel.
        public_branch = hybrid_public_label(action)
        why = (f"market gives {_esc(fav)} at least {thr}, so {_esc(fav)} "
               "remains the XWOBA side" if action == "FOLLOW" else
               f"market gives {_esc(fav)} under {thr}, so the rule backs "
               f"the market favorite, {_esc(pick)}, instead")
        rule_line = (
            f"<div class='vline'><span class='vk'>Rule</span>"
            f"<span><b>{public_branch} → {_esc(pick)}</b>"
            + (f" {_fmt_ml(sel_price)}" if sel_price is not None else "")
            + f"</span></div><div class='vnote'>{why}</div>")

    history = _branch_history(ctx, action)
    # The warm accent marks a FADE -- the one case where the published
    # selection differs from the model's own lean. It has never meant "bet
    # this side".
    cls = " edge" if action == "FADE" else ""
    version = _model_version_short()
    return (
        f"<div class='verdict{cls}'><div class='l'>Model vs market</div>"
        "<div class='vt'>"
        "<div class='vprofile-title vgroup'>This game</div>"
        f"<div class='vline'><span class='vk'>Model lean</span>"
        f"<span>{_esc(fav)} · {version} Δ {delta_txt} ({strength})</span></div>"
        f"<div class='vline'><span class='vk'>Market price</span>"
        f"<span>{_esc(fav)} {price_txt} · {p_txt}</span></div>"
        f"{rule_line}"
        f"{history}</div></div>"
    )


def _hitter_row_html(i, hr):
    """One batting-order row: name + batting hand, Statcast percentile bar, and
    raw xwOBA.

    The hand letter is also the platoon-advantage marker. It takes the hitter
    bar's warm accent when this batter holds the platoon edge over the starter
    and stays muted otherwise, so one glyph that has to be on the row anyway
    carries both facts -- replacing a separate ◆ that only ever appeared
    alongside it.
    """
    nm_text = _esc(hr["name"])
    nm_link = _mlb_player_link(hr["name"], hr.get("player_id"))
    if hr["bats"]:
        adv = bool(hr["adv"])
        cls = "b adv" if adv else "b"
        title = " title='platoon advantage vs this SP'" if adv else ""
        b = f"<span class='{cls}'{title}>{hr['bats']}</span>"
    else:
        b = ""
    bar = _pct_bar(hr.get("xw_pctile"), "h")
    backfill = bool(hr.get("xw_team_backfill"))
    xw_title = " title='team-average backfill: player absent from Savant'" if backfill else ""
    xw_mark = "<span class='pa'>*</span>" if backfill else ""
    xw_c = f"<td class='r mach'{xw_title}>{f3(hr['xw'])}{xw_mark}</td>"
    return (f"<tr><td class='ord'>{i}</td>"
            f"<td class='nm' title='{nm_text}'>{nm_link}{b}</td>"
            # Roster primary, from people.primaryPosition -- the Savant lineup
            # payload this row is built from carries ids, not card positions.
            # Three SS on one card is three players whose *roster* primary is
            # SS, and the column has no header to say so, so the cell does.
            f"<td class='pos' title='roster primary position, not tonight’s "
            f"lineup card'>{_esc(hr['pos'])}</td>"
            f"<td class='pct r'>{bar}</td>{xw_c}</tr>")


def _lineup_details(side_d):
    st = side_d["lu_status"]
    st_lab = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, st or "—")
    st_cls = {"posted": "posted", "partial_filled": "partial", "projected": "projected"}.get(st, "projected")
    parts = []
    if side_d["opp_xw"] is not None:
        parts.append(f"{MODEL_RATE_LABEL} {f3(side_d['opp_xw'])}")
    summ = " · ".join(parts) if parts else ""
    # No header row: the columns (order, name, position, percentile bar, xwOBA)
    # read without labels, and the legend below the cards carries the key.
    body = "".join(_hitter_row_html(i + 1, hr) for i, hr in enumerate(side_d["hitters"]))
    if not body:
        body = "<tr><td class='na' colspan='5'>lineup unavailable</td></tr>"
    return (
        "<details class='lineup' open>"
        "<summary><span class='chev'>▶</span>"
        f"<span class='tl'>{_esc(side_d['opp_abbr'])} lineup</span>"
        f"<span class='st {st_cls}'>{st_lab}</span>"
        f"<span class='lw mach'>{summ}</span></summary>"
        f"<div class='lu-scroll'><table class='lu'>{body}</table></div></details>")


def _sp_stat_cell(lab, val, fmt, sub=None, heat=""):
    s = f"<div class='s'>{sub}</div>" if sub else ""
    st = f" style='{heat}'" if heat else ""
    return (f"<div class='stat'{st}><div class='l'>{lab}</div>"
            f"<div class='v'>{fmt(val)}</div>{s}</div>")


def _side_html(sp_abbr, d, league_baseline):
    badge = f"<span class='hand'>{d['t']}HP</span>" if d["t"] in ("L", "R") else ""
    has_fullgame = "bullpen_sequential" in str(d.get("pitching_basis") or "")
    opener = ("<span class='flag warn' title='Opener: the model estimates this "
              "pitcher’s workload. The full-game lean also requires a valid "
              "role-filtered bullpen aggregate.'>"
              f"{'opener · bullpen' if has_fullgame else 'opener'}</span>") \
              if d.get("is_opener") else ""
    # A starter with no leaderboard line still prints a rate -- his prior, which
    # under wOBA v4 is his own regressed history and looks exactly like a
    # measurement. Say which it is on the card, next to the rate it qualifies.
    # The lean is unchanged: this flag reports the input, it does not gate it.
    no_line = ("<span class='flag warn' title='No season line for this starter "
               f"on the leaderboard: the {MODEL_RATE_LABEL} shown is his prior, "
               "and the lean leans on the lineup half.'>prior only</span>"
               if str(d.get("sp_rate_basis") or "") == "prior_only" else "")
    comp = (f"{d['R']}R/{d['L']}L" + (f"/{d['S']}S" if d["S"] else "")) if d["has_pl"] else "—"
    padv = f" · {d['padv']} plt-adv" if d["has_pl"] else ""
    lb = league_baseline or {}
    # lg K%/BB% come from the PA-weighted *batter* leaderboard; league K% and
    # BB% are symmetric (every batter K/BB is a pitcher K/BB), so they are
    # valid pitcher references and lg K-BB% = lg K% - lg BB%.
    lg = {k: _f(lb.get(k)) for k in ("xwOBA", "K%", "BB%", "ERA")}
    kbb = (d["pit_k"] - d["pit_bb"]) if (d["pit_k"] is not None
                                         and d["pit_bb"] is not None) else None
    lg_kbb = (lg["K%"] - lg["BB%"]) if (lg["K%"] is not None
                                        and lg["BB%"] is not None) else None
    # xERA (Statcast expected ERA) shown against the starter's own season ERA;
    # tinted vs league ERA (higher xERA = more offense-favorable = warm).
    xera_sub = (f"season {f2(d['era_season'])}"
                if d.get("era_season") is not None else None)
    stats = (
        _sp_stat_cell(f"{MODEL_RATE_LABEL} agn", d["pit_xw"], f3,
                      f"lg {f3(lg['xwOBA'])}" if lg["xwOBA"] is not None else None,
                      heat=heat_style(d["pit_xw"], lg["xwOBA"], HEAT_DOMAINS["xwOBA_sp"]))
        + _sp_stat_cell("K-BB%", kbb, f1,
                        f"lg {f1(lg_kbb)}" if lg_kbb is not None else None,
                        heat=heat_style(kbb, lg_kbb, HEAT_DOMAINS["K-BB%"], hi="cool"))
        + _sp_stat_cell("xERA", d.get("xera"), f2, xera_sub,
                        heat=heat_style(d.get("xera"), lg["ERA"], HEAT_DOMAINS["ERA"])))
    tier_lab, tier_cls = _tier_word(d.get("pit_xw_pctile"))
    tier = f"<span class='tier {tier_cls}'>{tier_lab}</span>" if tier_lab else ""
    bars = (f"<div class='spct'><span class='lab'>{MODEL_RATE_LABEL}</span>{_pct_bar(d.get('pit_xw_pctile'), 'p')}</div>"
            f"<div class='spct'><span class='lab'>K − BB%</span>{_pct_bar(d.get('kbb_pctile'), 'p')}</div>")
    exp_ip = _f(d.get("expected_sp_ip"))
    pitching_note = (
        f" · model {exp_ip:.1f} IP + bullpen"
        if exp_ip is not None and has_fullgame
        else " · starter unmeasured; no lean"
        if d.get("pitching_basis") == "starter_unmeasured_no_lean"
        else " · starter only; no full-game lean"
        if d.get("pitching_basis") == "starter_only_no_fullgame_lean"
        else ""
    )
    sp_team = _esc(sp_abbr)
    opp_team = _esc(d["opp_abbr"])
    return (
        f"<section class='side'>"
        f"<div class='matchlab'>{sp_team} starter → {opp_team} bats</div>"
        f"<div class='sp'><div class='who'><span class='nm'>"
        f"{_mlb_player_link(d['p'], d.get('player_id'))}</span>{badge}{tier}{opener}{no_line}</div>"
        f"<div class='role'>faces the {opp_team} lineup"
        f"<span class='mach'> · {comp}{padv}{pitching_note}</span></div>"
        f"<div class='sp-bars'>{bars}</div>"
        # `.sp` closes here. It was previously left open, which nested the
        # spotlight and the whole lineup table inside the starter block and let
        # `.sp .nm` (0,2,0) outrank `table.lu td` (0,1,2) -- every hitter name
        # rendered at the starter's 16px/700 instead of the table's 12.5px/400.
        f"<div class='spstats mach'>{stats}</div></div>"
        f"{_spotlight_html(d['hitters'])}"
        f"{_lineup_details(d)}"
        f"</section>")


def _market_html(o, away_abbr, home_abbr, fav=None, ctx=None, delta=None):
    """Odds strip: the four market numbers share one horizontal row at every
    width (`.modds`), with the verdict on its own full-width row beneath. No
    per-card build stamp -- the page header carries the build time once, and
    every card on the page is built from the same fetch."""
    o = o or {}
    def _mlcell(prefix, lab, cur, opn):
        sub = f" <span class='mv'>← {_fmt_ml(opn)} open</span>" if opn is not None else ""
        return (f"<div class='mcell'><div class='l'>{prefix} · "
                f"{_esc(lab)}</div>"
                f"<div class='v'>{_fmt_ml(cur)}{sub}</div></div>")
    tot = f"o/u {o['total']:g}" if o.get("total") is not None else "—"
    ph = f"{o['p_home'] * 100:.1f}%" if o.get("p_home") is not None else "—"
    return (
        "<div class='market'><div class='modds'>"
        + _mlcell("DK ML", away_abbr, o.get("away_ml"), o.get("open_away_ml"))
        + _mlcell("DK ML", home_abbr, o.get("home_ml"), o.get("open_home_ml"))
        + f"<div class='mcell'><div class='l'>Total</div><div class='v'>{tot}</div></div>"
        + f"<div class='mcell'><div class='l'>Implied "
        + _esc(home_abbr)
        + f" (devig)</div><div class='v'>{ph}</div></div>"
        + "</div>"
        + _verdict_html(fav, o, away_abbr, home_abbr, ctx, delta)
        + "</div>")


def _score_text(score):
    """Whole-run score text, or None when the schedule has not supplied one."""
    value = pd.to_numeric(score, errors="coerce")
    return str(int(value)) if pd.notna(value) else None


def _lost(g, side):
    """True when this side finished behind. Final games only.

    A trailing team in a live game has not lost anything yet, so the dimming
    waits for the final -- which is also what makes it readable as a result
    rather than as a current state. A tie or a missing score dims neither."""
    if str(g.get("abstract_state") or "").lower() != "final":
        return False
    other = "home" if side == "away" else "away"
    mine = pd.to_numeric(g.get(f"{side}_score"), errors="coerce")
    theirs = pd.to_numeric(g.get(f"{other}_score"), errors="coerce")
    return bool(pd.notna(mine) and pd.notna(theirs) and mine < theirs)


def _team_meta_span(g, side):
    """Render the pregame streak, replaced by the score for live/final games."""
    streak = g.get(f"{side}_streak")
    state = str(g.get("abstract_state") or "").lower()
    data = f" data-streak='{_esc(streak)}'" if streak else ""
    if state in ("live", "final"):
        score = _score_text(g.get(f"{side}_score"))
        title = "live score" if state == "live" else "final score"
        text = score if score is not None else "—"
        cls = f"team-meta score {side}" + (" lost" if _lost(g, side) else "")
        return (f"<span class='{cls}'{data} title='{title}'>"
                f"{text}</span>")
    if not streak:
        return f"<span class='team-meta streak {side}' data-streak='' hidden></span>"
    return (f"<span class='team-meta streak {side}'{data} "
            f"title='current win/loss streak'>{_esc(streak)}</span>")


def _game_state_span(g):
    """LIVE/FINAL header badge; hidden before first pitch."""
    state = str(g.get("abstract_state") or "").lower()
    raw_detail = g.get("status")
    detail = "" if raw_detail is None or pd.isna(raw_detail) else str(raw_detail)
    if state not in ("live", "final"):
        return "<span class='game-state' hidden></span>"
    label = "LIVE"
    if state == "live":
        inning = _inning_indicator(g)
        if inning:
            label += f" · {inning}"
    else:
        label = "FINAL"
    title = f" title='{_esc(detail)}'" if detail else ""
    return f"<span class='game-state {state}'{title}>{label}</span>"


def _base_out_text(bases, outs):
    """Screen-reader sentence for the diamond, e.g. '2 out, runners on first
    and second'. The glyph itself is aria-hidden, so this is the only reading
    of the state; an unknown out count degrades to the runners alone."""
    named = [n for n, on in zip(("first", "second", "third"), bases) if on]
    if not named:
        runners = "bases empty"
    elif len(named) == 3:
        runners = "bases loaded"
    elif len(named) == 1:
        runners = f"runner on {named[0]}"
    else:
        runners = f"runners on {' and '.join(named)}"
    if outs is None:
        return runners
    return f"{outs} out, {runners}"


def _base_out_html(g):
    """Diamond + out dots for the collapsed row, as in a scoreboard app.

    Always emitted so score_refresh_js can fill it in place, and `hidden`
    unless the game is live -- a base-out state means nothing before first
    pitch or after the last out. The three bases carry `on` when occupied and
    the dots fill up to the out count; both are aria-hidden decorations over
    `_base_out_text`."""
    state = str(g.get("abstract_state") or "").lower()
    bases = tuple(bool(g.get(k)) for k in ("on_first", "on_second", "on_third"))
    outs_val = pd.to_numeric(g.get("outs"), errors="coerce")
    outs = int(outs_val) if pd.notna(outs_val) else None
    live = state == "live"
    cls = ["b1", "b2", "b3"]
    diamond = "".join(
        f"<i class='{c}{' on' if on else ''}'></i>" for c, on in zip(cls, bases))
    dots = "".join(
        f"<i class='{'on' if outs is not None and i < outs else ''}'></i>"
        for i in range(3))
    return (f"<span class='bo'{'' if live else ' hidden'}>"
            f"<span class='diamond' aria-hidden='true'>{diamond}</span>"
            f"<span class='outs' aria-hidden='true'>{dots}</span>"
            f"<span class='bo-sr'>{_esc(_base_out_text(bases, outs))}</span>"
            "</span>")


def _inning_indicator(g):
    """Live inning as ▲2nd / ▼2nd; empty when the feed has no inning yet."""
    ordinal = g.get("current_inning_ordinal")
    if ordinal is None or pd.isna(ordinal):
        inning = pd.to_numeric(g.get("current_inning"), errors="coerce")
        if pd.isna(inning):
            return ""
        n = int(inning)
        suffix = "th" if 10 < n % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        ordinal = f"{n}{suffix}"
    half = str(g.get("inning_half") or "").lower()
    if half == "top":
        marker = "▲"
    elif half == "bottom":
        marker = "▼"
    else:
        top = g.get("is_top_inning")
        marker = ("▲" if bool(top) else "▼") if top is not None and not pd.isna(top) else ""
    return f"{marker}{_esc(ordinal)}"


def _logo_assets(games):
    """Resolve cap logos for every team on the slate.

    Returns ({team_id: (light_uri, dark_uri)}, <style-block>). Each data URI is
    emitted exactly once as a per-team CSS custom property (`--lgl`/`--lgd`); a
    single pair of theme-scoped rules switches the visible variant, mirroring the
    same prefers-color-scheme / data-theme selectors the palette vars use so the
    logo tracks the manual Theme toggle. Teams whose fetch missed are omitted and
    their cards keep the text abbreviation."""
    ids = set()
    for g in games:
        for k in ("away_team_id", "home_team_id"):
            v = g.get(k)
            if v is not None and pd.notna(v):
                try:
                    ids.add(int(v))
                except (TypeError, ValueError):
                    continue
    assets = {}
    for tid in sorted(ids):
        lt, dk = team_logo_uris(tid)
        if lt and dk:
            assets[tid] = (lt, dk)
    if not assets:
        return assets, ""
    base = (".clogo{display:inline-block;width:1.05em;height:1.05em;flex:none;"
            "vertical-align:-.16em;margin-right:.30em;background-size:contain;"
            "background-repeat:no-repeat;background-position:center;"
            "background-image:var(--lgl)}")
    per_team = "".join(
        f'.t{tid}{{--lgl:url("{lt}");--lgd:url("{dk}")}}'
        for tid, (lt, dk) in assets.items())
    dark = ('@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) '
            '.clogo{background-image:var(--lgd)}}'
            'html[data-theme="dark"] .clogo{background-image:var(--lgd)}')
    return assets, f"<style>{base}{per_team}{dark}</style>"


def _club_logo(team_id, ctx):
    """Decorative cap-logo chip for a team abbr, or '' when the logo did not
    resolve (keeping the text abbreviation as the sole label). aria-hidden since
    the adjacent abbreviation already names the team."""
    ids = (ctx or {}).get("logo_ids")
    if not ids or team_id is None or (isinstance(team_id, float) and pd.isna(team_id)):
        return ""
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return ""
    return f"<span class='clogo t{tid}' aria-hidden='true'></span>" if tid in ids else ""


def _summary_team_label(g, side):
    """Familiar team label for the scoreboard row, with abbreviation fallback."""
    official = g.get(f"{side}_team_name")
    if official is not None and not pd.isna(official):
        official = str(official)
        return TEAM_LABELS.get(official, official)
    return str(g.get(f"{side}_abbr") or "")


def _summary_market_line(g, lean=None):
    """Locked/current Hybrid selection for the compact row."""
    odds = g.get("odds") or {}
    away, home = g.get("away_abbr"), g.get("home_abbr")
    p_lean = _lean_implied_p(odds, lean, away, home)
    action = hybrid_action(p_lean)
    pick = hybrid_selection(lean, away, home, p_lean)
    if action is None or pick is None:
        return "Selection pending"
    ml = odds.get("home_ml") if pick == home else odds.get("away_ml")
    return f"{pick} {_fmt_ml(ml)} · {hybrid_public_label(action)}".strip()


def _summary_team_html(g, side, ctx):
    abbr = str(g.get(f"{side}_abbr") or "")
    label = _summary_team_label(g, side)
    logo = _club_logo(g.get(f"{side}_team_id"), ctx)
    if not logo:
        logo = (f"<span class='summary-logo-fallback' aria-hidden='true'>"
                f"{_esc(abbr[:1])}</span>")
    return (
        f"<span class='summary-team {side}' data-team-abbr='{_esc(abbr)}'>"
        f"{logo}<span class='summary-club'>{_esc(label)}</span>"
        f"{_team_meta_span(g, side)}</span>"
    )


def _scoreboard_summary(g, ctx=None, lean=None):
    """Collapsed scoreboard row shared by modeled and pending games."""
    state = str(g.get("abstract_state") or "").lower()
    in_progress = state in ("live", "final")
    hide_pregame = " hidden" if in_progress else ""
    time_text = str(g.get("time_pt") or "Time TBD")
    market_text = _summary_market_line(g, lean)
    game_no = (f"<span class='game-no'>{_esc(g['game_label'])}</span>"
               if g.get("game_label") else "")
    away_label = _summary_team_label(g, "away")
    home_label = _summary_team_label(g, "home")
    return (
        f"<summary class='game-summary' aria-label='{_esc(away_label)} at "
        f"{_esc(home_label)}; expand game breakdown'>"
        f"<span class='teams' data-game-pk='{int(g['game_pk'])}'>"
        f"{_summary_team_html(g, 'away', ctx)}"
        f"<span class='summary-center'>{_game_state_span(g)}"
        f"{_base_out_html(g)}"
        f"<span class='summary-time'{hide_pregame}>{_esc(time_text)}</span>"
        f"{game_no}<span class='summary-market'{hide_pregame}>"
        f"{_esc(market_text)}</span></span>"
        f"{_summary_team_html(g, 'home', ctx)}</span>"
        "<span class='summary-chevron' aria-hidden='true'>⌄</span>"
        "</summary>"
    )


def _detail_context_html(g):
    when = " · ".join(x for x in (g.get("time_pt"), g.get("venue")) if x)
    return f"<div class='detail-context'>{_esc(when)}</div>" if when else ""


def cmb_card(g, strength_scale=None, ctx=None):
    if g.get("unavailable"):
        probables = " vs ".join(
            "TBD" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
            for x in (g.get("away_probable"), g.get("home_probable")))
        return (
            "<article class='card unavailable'>"
            "<details class='game-card'>"
            + _scoreboard_summary(g, ctx)
            + "<div class='game-detail'>"
            + _detail_context_html(g)
            + "<div class='card-note'><b>Awaiting paired probable pitchers</b>"
            f"<span>{_esc(probables)} · matchup lean will appear after both starters are listed.</span></div>"
            "</div></details></article>")

    a, h = g["away"], g["home"]
    away_abbr, home_abbr = g["away_abbr"], g["home_abbr"]
    # a = away SP -> his xw_edge is the HOME offense edge. When either edge is
    # missing there is no defined lean and no read sentence. The strength word
    # ranks |Δxw| against the ledger's lean-magnitude history (display-only)
    # and appears once, in the plain-language read beneath the header.
    fav = strength_word = read_html = delta = None
    if a["xw_edge"] is not None and h["xw_edge"] is not None:
        home_off, away_off = a["xw_edge"], h["xw_edge"]
        net = home_off - away_off
        if net != 0:
            delta = abs(net)
            fav = home_abbr if net > 0 else away_abbr
            strength_word, _ = lean_strength(delta, strength_scale)
            read_html = _read_sentence(away_abbr, home_abbr, a, h, fav, strength_word)
    return (
        "<article class='card'>"
        "<details class='game-card'>"
        + _scoreboard_summary(g, ctx, fav)
        + "<div class='game-detail'>"
        + _detail_context_html(g)
        + (read_html or "")
        + _market_html(g.get('odds'), away_abbr, home_abbr, fav, ctx, delta)
        + f"<div class='sides'>{_side_html(away_abbr, a, g['league_baseline'])}"
        + f"{_side_html(home_abbr, h, g['league_baseline'])}</div>"
        + "</div></details></article>")


def _game_order_key(game):
    """Chronological slate order with deterministic doubleheader tie-breaks."""
    raw_start = game.get("game_datetime_utc")
    try:
        start = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        start = start.astimezone(UTC)
    except (TypeError, ValueError):
        start = datetime.max.replace(tzinfo=UTC)
    game_number = pd.to_numeric(game.get("game_number"), errors="coerce")
    game_pk = pd.to_numeric(game.get("game_pk"), errors="coerce")
    return (
        start,
        int(game_number) if pd.notna(game_number) else 99,
        int(game_pk) if pd.notna(game_pk) else 2**63 - 1,
    )


def build_combined(games, strength_scale=None, ctx=None):
    cards = sorted(games, key=_game_order_key)
    return ("<div class='grid'>"
            + "".join(cmb_card(g, strength_scale, ctx) for g in cards)
            + "</div>")


def _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                          opp_hitters_df=None, detail_df=None, lg_ops=None,
                          slate_df=None, lineup_df=None,
                          league_baseline=None, odds=None, streaks=None):
    streaks = streaks or {}

    def _streak(team_id):
        try:
            return streaks.get(int(team_id))
        except (TypeError, ValueError):
            return None

    throws = {}
    player_ids = {}
    recent_era = {}
    if pitcher_rows_df is not None and not pitcher_rows_df.empty:
        for _, pr in pitcher_rows_df.iterrows():
            key = (pr["game_pk"], pr["Name"])
            throws[key] = pr.get("throws")
            if pd.notna(pr.get("player_id")):
                player_ids[key] = int(pr["player_id"])
            recent_era[key] = (
                _f(pr.get("ERA_L5")), int(pr.get("ERA_L5_GS") or 0),
                _f(pr.get("ERA_SEASON")), _f(pr.get("xERA")))
    pl_map = _pl_lookup(pl_df)
    slate_map = {}
    if slate_df is not None and not getattr(slate_df, "empty", True):
        for _, s in slate_df.iterrows():
            slate_map[s["game_pk"]] = s
    lu_map = {}
    if lineup_df is not None and not getattr(lineup_df, "empty", True):
        for _, s in lineup_df.iterrows():
            lu_map[s["game_pk"]] = {"away": s.get("away_lineup_status"),
                                    "home": s.get("home_lineup_status")}
    lg_ops_f = _f(lg_ops)

    # Percentile reference distributions are stashed on league_baseline in
    # fetch_all. Missing on pre-market/degraded builds -> every pctile_rank
    # returns None and the bars fall back to em-dashes.
    _lb0 = league_baseline or {}
    # Must be the same target `_pctile_ref_bat` was built with (fetch_all), or
    # the bar ranks a value regressed one way against a population regressed
    # another. Falls back to the league rate exactly when fetch_all's
    # derivation did, so the pair cannot come apart.
    _prior = _f(_lb0.get("_pctile_prior_bat"))
    if _prior is None:
        _prior = _f(_lb0.get("xwOBA"))
    _k_bat = XWOBA_SHRINK_K
    _ref_bat = _lb0.get("_pctile_ref_bat")
    _ref_pit = _lb0.get("_pctile_ref_pit")
    _ref_kbb = _lb0.get("_pctile_ref_kbb")

    games = []
    for gpk, gg in xw_df.groupby("game_pk", sort=False):
        a, h = _rows_by_side(gg)
        if a is None or h is None:
            miss = ", ".join(s for s, v in (("away SP", a), ("home SP", h)) if v is None)
            matchup = gg["matchup"].iloc[0] if "matchup" in gg.columns and len(gg) else "?"
            log(f"  game skipped from paired cards — missing {miss}: "
                f"{matchup} (game_pk={gpk})")
            continue
        srow = slate_map.get(gpk)

        def mk(r):
            side = r["side"]
            t = throws.get((gpk, r["pitcher"]), "")
            opp_lu_side = "home" if side == "away" else "away"
            starter_xw = _f(r.get("starter_xwOBA"))
            if starter_xw is None:
                starter_xw = _f(r.get("pit_xwOBA"))
            d = dict(p=r["pitcher"],
                     player_id=player_ids.get((gpk, r["pitcher"])),
                     t=t if t in ("L", "R") else "",
                     opp=r["opp_team"], opp_abbr=_abbr(r["opp_team"]),
                     pit_xw=starter_xw, pit_k=_f(r.get("pit_K%")),
                     pit_bb=_f(r.get("pit_BB%")),
                     era_season=recent_era.get((gpk, r["pitcher"]), (None, 0, None, None))[2],
                     xera=recent_era.get((gpk, r["pitcher"]), (None, 0, None, None))[3],
                     opp_xw=(
                         _f(r.get("opp_xwOBA_vs_sp"))
                         if _f(r.get("opp_xwOBA_vs_sp")) is not None
                         else _f(r.get("opp_xwOBA"))
                     ),
                     xw_edge=_f(r.get("edge_xwOBA")),
                     # Display percentiles (0-100). pit_xwOBA is already shrunk
                     # in build_matchup, so it is ranked directly (invert: low
                     # xwOBA-against = high pct); K-BB% is ranked raw.
                     pit_xw_pctile=pctile_rank_raw(starter_xw, _ref_pit, invert=True),
                     kbb_pctile=pctile_rank_raw(
                         (_f(r.get("pit_K%")) - _f(r.get("pit_BB%")))
                         if (_f(r.get("pit_K%")) is not None and _f(r.get("pit_BB%")) is not None)
                         else None, _ref_kbb),
                     lu_status=(lu_map.get(gpk) or {}).get(opp_lu_side),
                     is_opener=bool(r.get("opener")),
                     expected_sp_ip=_f(r.get("expected_sp_ip")),
                     bullpen_xw=_f(r.get("bullpen_xwOBA")),
                     pitching_basis=r.get("pitching_basis"),
                     sp_rate_basis=r.get("starter_rate_basis"),
                     sp_rate_bf=_f(r.get("starter_rate_bf")),
                     has_pl=False, R=0, L=0, S=0, padv=0,
                     pl_sp=None, pl_sp_raw=None, pl_mx=None, pl_edge=None,
                     pl_reliable=False,
                     hitters=_hitters_for(opp_hitters_df, detail_df,
                                          gpk, r["pitcher"], lg_ops_f,
                                          _ref_bat, _prior, _k_bat))
            pr = pl_map.get((gpk, side))
            if pr is not None:
                if not d["t"]:
                    pt_ = pr.get("throws"); d["t"] = pt_ if pt_ in ("L", "R") else ""
                d.update(has_pl=True,
                         R=int(pr.get("n_RHB") or 0), L=int(pr.get("n_LHB") or 0),
                         S=int(pr.get("n_SW") or 0), padv=int(pr.get("n_platoon_adv") or 0),
                         pl_sp=_f(pr.get("pit_OPS_allowed")), pl_sp_raw=_f(pr.get("pit_OPS_raw")),
                         pl_mx=_f(pr.get("mx_OPS")), pl_edge=_f(pr.get("edge_OPS")),
                         pl_reliable=bool(pr.get("reliable")))
            return d

        away_abbr = (srow.get("away_abbrev") if srow is not None else None) or _abbr(h["opp_team"])
        home_abbr = (srow.get("home_abbrev") if srow is not None else None) or _abbr(a["opp_team"])
        game_number = pd.to_numeric(srow.get("game_number"), errors="coerce") if srow is not None else np.nan
        is_dh = (str(srow.get("double_header") or "N") != "N") if srow is not None else False
        game_label = f"G{int(game_number)}" if pd.notna(game_number) and (is_dh or game_number > 1) else ""
        games.append(dict(
            away=mk(a), home=mk(h),
            away_abbr=away_abbr, home_abbr=home_abbr,
            away_team_name=(srow.get("away_team") if srow is not None else None),
            home_team_name=(srow.get("home_team") if srow is not None else None),
            away_team_id=(srow.get("away_team_id") if srow is not None else None),
            home_team_id=(srow.get("home_team_id") if srow is not None else None),
            away_streak=_streak(srow.get("away_team_id")) if srow is not None else None,
            home_streak=_streak(srow.get("home_team_id")) if srow is not None else None,
            abstract_state=(srow.get("abstract_state") if srow is not None else None),
            status=(srow.get("status") if srow is not None else None),
            away_score=(srow.get("away_score") if srow is not None else None),
            home_score=(srow.get("home_score") if srow is not None else None),
            current_inning=(srow.get("current_inning") if srow is not None else None),
            current_inning_ordinal=(
                srow.get("current_inning_ordinal") if srow is not None else None
            ),
            inning_half=(srow.get("inning_half") if srow is not None else None),
            is_top_inning=(srow.get("is_top_inning") if srow is not None else None),
            on_first=(srow.get("on_first") if srow is not None else None),
            on_second=(srow.get("on_second") if srow is not None else None),
            on_third=(srow.get("on_third") if srow is not None else None),
            outs=(srow.get("outs") if srow is not None else None),
            game_pk=gpk, game_number=game_number, game_label=game_label,
            game_datetime_utc=(srow.get("game_datetime_utc") if srow is not None else None),
            time_pt=_game_time_pt(srow.get("game_datetime_utc")) if srow is not None else "",
            venue=str(srow.get("venue") or "") if srow is not None else "",
            odds=(odds or {}).get(gpk),
            league_baseline={**(league_baseline or {}), "OPS": lg_ops_f},
        ))

    # Keep the slate complete even when one or both probable pitchers are not
    # posted yet. Previously these games silently disappeared because no paired
    # xwOBA rows could be built.
    built_game_pks = {int(g["game_pk"]) for g in games if g.get("game_pk") is not None}
    if slate_df is not None and not getattr(slate_df, "empty", True):
        for _, srow in slate_df.iterrows():
            gpk = int(srow["game_pk"])
            if gpk in built_game_pks:
                continue
            game_number = pd.to_numeric(srow.get("game_number"), errors="coerce")
            is_dh = str(srow.get("double_header") or "N") != "N"
            game_label = f"G{int(game_number)}" if pd.notna(game_number) and (is_dh or game_number > 1) else ""
            games.append(dict(
                unavailable=True, game_pk=gpk, game_number=game_number, game_label=game_label,
                game_datetime_utc=srow.get("game_datetime_utc"),
                away_abbr=srow.get("away_abbrev") or _abbr(srow.get("away_team")),
                home_abbr=srow.get("home_abbrev") or _abbr(srow.get("home_team")),
                away_team_name=srow.get("away_team"),
                home_team_name=srow.get("home_team"),
                away_team_id=srow.get("away_team_id"),
                home_team_id=srow.get("home_team_id"),
                away_streak=_streak(srow.get("away_team_id")),
                home_streak=_streak(srow.get("home_team_id")),
                abstract_state=srow.get("abstract_state"),
                status=srow.get("status"),
                away_score=srow.get("away_score"),
                home_score=srow.get("home_score"),
                current_inning=srow.get("current_inning"),
                current_inning_ordinal=srow.get("current_inning_ordinal"),
                inning_half=srow.get("inning_half"),
                is_top_inning=srow.get("is_top_inning"),
                on_first=srow.get("on_first"),
                on_second=srow.get("on_second"),
                on_third=srow.get("on_third"),
                outs=srow.get("outs"),
                time_pt=_game_time_pt(srow.get("game_datetime_utc")),
                venue=str(srow.get("venue") or ""),
                away_probable=srow.get("away_probable_pitcher"),
                home_probable=srow.get("home_probable_pitcher"),
            ))
    return games


def _legend_head(model_label, built_txt):
    """Slim model/build stamp, the last thing on the page.

    It also declares the clock zone, and is the only thing that does.
    `_fmt_pt_clock` prints bare wall clocks on the cards on the understanding
    that the page states the zone exactly once; the how-to-read guide used to
    be that once, so when the guide was deleted the clause moved here rather
    than being dropped. A slate of bare local times with no stated local is
    ambiguous in a way the rest of the de-cluttering was not.
    """
    date = SLATE_DATE
    head = model_label
    if built_txt:
        head += f" · built {built_txt}"
    elif date:
        head += f" · {date}"
    # "first pitch times" undersold it: the build stamp beside it is Pacific
    # too (_built_text_now uses datetime.now(PT)), and it is the other bare
    # clock on the page. One zone, stated once, has to cover both.
    return (f"<div class='legend'><div class='lg-title'>{head}"
            "<em> · all times Pacific</em></div></div>")


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
# (family, file, weight range). Both are variable fonts subset to the glyphs
# this site actually emits -- ASCII, Latin-1 and Latin Extended-A (the roster
# carries Jose, Pena, Suarez, Munoz...) plus the exact symbol set grepped out
# of this module. Full latin subsets were 35KB/31KB; these are 27KB/28KB.
_FONT_FACES = (
    ("Archivo", "archivo-subset.woff2", "100 900"),
    ("JetBrains Mono", "jetbrains-mono-subset.woff2", "400 800"),
)


def font_face_css():
    """`@font-face` rules with the woff2 payloads inlined as data URIs.

    The page loads no external resources -- logos are already inlined the same
    way -- and keeping the fonts local holds that. A missing or unreadable file
    degrades to the system stack behind it in `--sans`/`--mono` rather than
    failing the build: the type gets plainer, the numbers still line up, and
    the slate still ships. Licences sit next to the files in assets/fonts/
    (both OFL).
    """
    out = []
    for family, fname, wght in _FONT_FACES:
        path = os.path.join(FONT_DIR, fname)
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as e:  # noqa: BLE001
            log(f"font {fname} unavailable, falling back to system stack: {e!r}")
            continue
        out.append(
            f"@font-face{{font-family:'{family}';font-style:normal;"
            f"font-weight:{wght};font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2-variations')}}")
    return "".join(out)


CSS = r"""/* ============================================================
   MLB matchup leans -- box-score / scout-sheet tokens
   ============================================================ */
:root{
  --bg:#f3f5f7; --surface:#ffffff; --surface-2:#eef1f4; --ink:#161b20;
  --muted:#4f5a65; --faint:#636e7b; --line:#dde3e8; --line-2:#eceff2;
  /* --faint is a text colour, not a hairline: column headers, stat labels,
     the `lg .312` / `season 4.04` subs, the machinery line, the opening-price
     tail and the `projected` badge all use it, all below 12px. It must clear
     4.5:1 against BOTH --surface and --surface-2 (the market strip and the
     stat cells sit on --surface-2). The previous #98a4af measured 2.54:1 on
     white and 2.24:1 on the strip; #5f6c78 dark measured 3.15:1. --muted moved
     with it (5.54 -> 7.04:1 on white) so the two-step text hierarchy survives
     the darkening instead of collapsing into one tone. Re-measure both
     surfaces before changing either -- tests/test_casual_redesign.py checks
     the ratios and the ordering. */
  --warm:198,84,44;             /* ember  -- offense-favorable  */
  --cool:52,116,168;            /* steel  -- pitcher-favorable  */
  --lean:176,124,16;            /* amber  -- lean pill / links  */
  /* Text-only twins of the three accents. The accents above were picked as
     fills and borders, but they are also set as `color` on the tier pills,
     the LIVE badge, the read-line emphasis, the verdict label and the ledger
     badges -- all 9-13.5px, and most of them sitting on a .08-.12 tint of
     their own hue, which lifts the background toward the text. Measured in
     headless Chromium at 390px over the composited tint: warm 3.94:1 on the
     `.B` strip and 4.01:1 on its own tier tint, cool 3.90:1 under the LIVE
     badge, lean 3.12:1 on the amber verdict -- all short of 4.5:1, in light
     only (dark passed everywhere, so there the twins alias the accents).
     These are the same hue and saturation darkened in HLS until every
     background each one lands on -- both bare surfaces and a .08/.10/.12
     wash of its own hue over either -- clears 4.5:1. The .12 case is the
     binding one; solving against .08 alone left the ledger W/L badges at
     3.86:1. Use them wherever an accent
     is the `color`; keep the base tokens for fills, borders and bar hues so
     the percentile bars and legend swatches do not shift.
     tests/test_casual_redesign.py measures both. */
  --warm-tx:164,69,36;
  --cool-tx:46,103,149;
  --lean-tx:130,92,12;
  --amberbg:246,196,86;
  --chip-fg:#1b1e25;
  /* Archivo (text) and JetBrains Mono (every number on the page) are inlined
     by font_face_css() as subset variable woff2. The system stacks stay behind
     them as the fallback if a font file ever goes missing. Archivo is a
     grotesque with enough width and x-height to stay legible at the label
     sizes; JetBrains Mono was picked for its digits -- slashed zero, a 1 with
     a foot, unmistakable 6/8/9 -- because scores, odds, xwOBA and percentiles
     are the whole point of the page and they were rendering in whatever
     generic mono the OS happened to supply. */
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --sans:"Archivo",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  --shadow:0 1px 2px rgba(16,18,29,.05),0 10px 26px -20px rgba(16,18,29,.28);
  /* Corner-radius scale -- three tiers, no raw px radii anywhere else.
     --r      containers: things that hold other content (card, stat strip,
              table wrapper, button)
     --r-s    inline marks: badges, flags, status tags, percentile bars,
              legend swatches -- anything sitting inside a line of text
     --r-pill true pills: fully-rounded ends, used only where the shape is
              the point
     tests/test_casual_redesign.py enforces this; add a tier here rather
     than a one-off px value at the call site. */
  --r:6px;
  --r-s:3px;
  --r-pill:999px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#7f8b97; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  /* Dark already clears 4.5:1 on every accent-as-text pair, so the text
     twins alias the accents rather than introducing a second palette. */
  --warm-tx:236,122,72; --cool-tx:96,158,208; --lean-tx:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}}
html[data-theme="dark"]{
  --bg:#0f1418; --surface:#171d24; --surface-2:#131920; --ink:#e7ecef;
  --muted:#96a2ad; --faint:#7f8b97; --line:#242e38; --line-2:#1c242c;
  --warm:236,122,72; --cool:96,158,208; --lean:244,196,96; --amberbg:244,196,96;
  /* Dark already clears 4.5:1 on every accent-as-text pair, so the text
     twins alias the accents rather than introducing a second palette. */
  --warm-tx:236,122,72; --cool-tx:96,158,208; --lean-tx:244,196,96;
  --chip-fg:#e7ecef;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
/* The `hidden` attribute must win over any component `display`. `.game-state`
   set `display:inline-block`, which outranks the UA sheet's `[hidden]` rule, so
   the empty pregame badge rendered as a bare bordered chip above the time on
   every unstarted card -- and score_refresh_js re-hides it the same way. */
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 var(--sans);
  -webkit-font-smoothing:antialiased;padding:20px 16px 60px}
.mx-wrap{max-width:1060px;margin:0 auto}

.topbar{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:12px}
.brand{font:800 18px/1 var(--sans);letter-spacing:.13em;text-transform:uppercase}
.theme{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--muted);
  font:600 14px/1 var(--sans);padding:7px 11px;border-radius:var(--r);cursor:pointer}
.theme:hover{color:var(--ink)}
.theme:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}

.legend{margin:2px 2px 14px}
.lg-title{font-size:14.5px;font-weight:650;letter-spacing:.01em;margin-bottom:7px}
.lg-title em{font-style:normal;color:var(--muted);font-weight:500}
/* `.lg-keys` outlived the how-to-read guide: the empty-slate page still uses
   it for its one "no games scheduled" line. */
.lg-keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted)}
.lg-keys .k{display:inline-flex;align-items:center;gap:6px}

/* One continuous scoreboard, not a stack of separate cards. The slate is a
   single surface and games are divided by a hairline -- the list reads top to
   bottom in one pass instead of as a dozen boxed objects, which is what an
   8px gap plus a border plus a shadow per game was doing. The container keeps
   the radius, the border and the shadow; the game gives all three up. `clip`
   rounds the first and last summary backgrounds off at the corners. */
.grid{display:flex;flex-direction:column;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow);overflow:clip}
.card{background:transparent;border:0;border-radius:0;box-shadow:none}
.card + .card{border-top:1px solid var(--line)}

/* ---------- collapsed scoreboard row ---------- */
.game-card{margin:0}
.game-summary{position:relative;display:flex;align-items:center;min-height:94px;
  padding:12px 16px;list-style:none;cursor:pointer;user-select:none}
.game-summary::-webkit-details-marker{display:none}
.game-summary::marker{content:""}
.game-summary:hover{background:var(--surface-2)}
.game-summary:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:-2px}
.game-card[open]>.game-summary{background:var(--surface-2);
  border-bottom:1px solid var(--line-2)}
.teams{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,.68fr) minmax(0,1fr);
  align-items:center;gap:14px;width:100%;min-width:0;padding-right:22px}
.summary-team{display:grid;align-items:center;gap:8px;min-width:0}
.summary-team.away{grid-template-columns:46px minmax(0,1fr) auto;
  grid-template-areas:"logo club meta"}
.summary-team.home{grid-template-columns:auto minmax(0,1fr) 46px;
  grid-template-areas:"meta club logo"}
.summary-team .clogo,.summary-logo-fallback{grid-area:logo}
.summary-team .clogo{width:46px;height:46px;margin:0;vertical-align:middle}
.summary-logo-fallback{display:grid;width:46px;height:46px;place-items:center;
  border:1px solid var(--line);border-radius:var(--r-pill);background:var(--surface-2);
  color:var(--muted);font:800 16.5px/1 var(--sans)}
.summary-club{grid-area:club;min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font:750 16.5px/1.15 var(--sans)}
.summary-team.home .summary-club{text-align:right}
.summary-center{display:flex;min-width:0;flex-direction:column;align-items:center;
  justify-content:center;gap:4px;text-align:center}
.summary-time{font:800 18px/1 var(--mono);font-variant-numeric:tabular-nums}
.summary-market{font:600 13px/1.1 var(--mono);color:var(--muted);
  font-variant-numeric:tabular-nums}
.summary-chevron{position:absolute;right:13px;top:50%;color:var(--faint);
  font:800 18px/1 var(--sans);transform:translateY(-50%);
  transition:transform .16s ease,color .16s ease}
.game-card[open]>.game-summary .summary-chevron{
  color:var(--ink);transform:translateY(-50%) rotate(180deg)}
.teams .team-meta{font:600 13px/1 var(--mono);color:var(--muted);letter-spacing:0;
  grid-area:meta;margin:0;vertical-align:middle;white-space:nowrap}
/* The score is the one thing a scoreboard exists to show, so it carries
   the largest type on the row -- roughly the club name doubled. It sits
   in the meta slot under the club, so the row grows to fit rather than
   anything having to move. */
.teams .score{font-size:28px;font-weight:800;color:var(--ink);
  line-height:1.05;letter-spacing:-.01em}
/* The loser's number recedes so a final reads as a result before any digit is
   parsed. --faint is the site's lightest text tone and is held to 4.5:1 on
   both surfaces; at this size it is large text (3:1), so it clears with room. */
.teams .score.lost{color:var(--faint)}
.game-no{font:700 13px/1 var(--mono);color:var(--muted);vertical-align:middle}
.game-state{display:inline-block;padding:3px 6px;border:1px solid var(--line);
  border-radius:var(--r-s);font:750 12px/1 var(--sans);letter-spacing:.08em;
  vertical-align:middle;color:var(--muted);background:var(--surface-2)}
.game-state.live{color:rgba(var(--cool-tx),1);border-color:rgba(var(--cool),.35);
  background:rgba(var(--cool),.10)}

/* Base-out state: three bases on a diamond over three out dots, the scoreboard
   convention. The bases are 7px squares rotated 45deg and placed on a 3-column
   grid -- second on the top row, third and first flanking below -- so the
   cluster reads as a diamond without any glyph font or image. Occupied bases
   and recorded outs fill solid; the rest stay hairline outlines. */
.bo{display:flex;flex-direction:column;align-items:center;gap:2px}
/* Diamond geometry: the bases are 6px squares rotated 45deg, so the facing
   edges of two of them are 6px apart when their centres are 6px apart along
   the 45deg diagonal. A 5.5px cell step puts adjacent centres 7.8px apart --
   6px of base plus a ~1.8px gap, tight enough to read as one cluster without
   the squares merging into a bar. The squares overflow the grid box by design;
   the padding keeps that overflow off the badge above. */
.diamond{display:grid;flex:none;grid-template-columns:repeat(3,5.5px);
  grid-template-rows:repeat(2,5.5px);place-items:center;padding:3px 2px}
.diamond i{width:6px;height:6px;flex:none;border:1px solid var(--faint);
  transform:rotate(45deg)}
.diamond .b2{grid-area:1/2}.diamond .b3{grid-area:2/1}.diamond .b1{grid-area:2/3}
.diamond i.on{background:rgba(var(--lean),1);border-color:rgba(var(--lean),1)}
.outs{display:flex;align-items:center;gap:3px}
.outs i{width:5px;height:5px;flex:none;border:1px solid var(--faint);
  border-radius:var(--r-pill)}
.outs i.on{background:var(--ink);border-color:var(--ink)}
/* The glyph is decorative; _base_out_text carries the state for readers. */
.bo-sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;
  border:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.detail-context{padding:7px 16px;border-bottom:1px solid var(--line-2);
  color:var(--muted);font:500 13px/1.3 var(--sans)}
.card-note{display:flex;flex-direction:column;gap:4px;padding:14px 16px 16px;color:var(--muted)}
.card-note b{color:var(--ink);font-size:14.5px}.card-note span{font-size:14px}

/* ---------- market strip ---------- */
/* Two stacked rows: the four odds numbers, then the verdict. .modds never
   wraps -- the cells share the width evenly and shrink together, so the strip
   reads as one horizontal line of numbers at phone widths too. */
.market{border-bottom:1px solid var(--line-2);background:var(--surface-2)}
.modds{display:flex}
.mcell{flex:1 1 0;min-width:0;display:flex;flex-direction:column;
  justify-content:space-between;padding:7px 16px;border-right:1px solid var(--line-2)}
.mcell:last-child{border-right:0}
.mcell .l,.verdict .l{font:600 12px/1.4 var(--sans);letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}
.mcell .v{font:500 14.5px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.mcell .v .mv{color:var(--faint);font-size:13px}

/* ---------- two sides ---------- */
.sides{display:grid;grid-template-columns:1fr 1fr}
.side{padding:12px 16px 14px;min-width:0}
.side + .side{border-left:1px solid var(--line-2)}
/* Side-by-side only once a column can hold the lineup table without an
   inner horizontal scroll. Measured in headless Chromium: the table
   overflows its column by 62px at 762px wide, 23px at 840px, 3px at 880px
   and 0 from ~900px up. 760px put every card on the slate into a hidden
   side-scroll across the whole tablet range. */
@media (max-width:900px){
  .sides{grid-template-columns:1fr}
  .side + .side{border-left:0;border-top:1px solid var(--line-2)}
}

/* SP block */
.sp .who{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.sp .nm{font:700 18px/1.2 var(--sans)}
.player-link{color:inherit;text-decoration:underline;
  text-decoration-color:rgba(var(--lean),.5);text-decoration-thickness:1px;
  text-underline-offset:2px}
.player-link:hover{color:rgba(var(--lean-tx),1);text-decoration-color:currentColor}
.player-link:focus-visible{
  outline:2px solid rgba(var(--lean),1);outline-offset:2px}
.hand{font:500 12.5px/1.4 var(--mono);color:var(--muted);border:1px solid var(--line);border-radius:var(--r-s);padding:0 4px}
.sp .role{font-size:13px;color:var(--faint);margin-top:1px}
.spstats{display:flex;margin-top:8px;border:1px solid var(--line-2);border-radius:var(--r);overflow:hidden}
.stat{flex:1;padding:6px 8px 5px;border-right:1px solid var(--line-2);min-width:0}
.stat:last-child{border-right:0}
.stat .l{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
.stat .v{font:500 16px/1.3 var(--mono);font-variant-numeric:tabular-nums}
.stat .s{font:400 12.5px/1.3 var(--mono);color:var(--muted)}
/* Pills on the starter's `.who` line -- the tier word and the flags -- share
   one type treatment so they read as one family. Size, tracking, case and box
   metrics live here and nowhere else; every rule below adds colour only. The
   `.tier` accents sit further down with the percentile bars. Spacing comes
   from `.sp .who`'s 8px gap: no pill carries its own margin. */
.tier,.flag{font:600 12px/1.4 var(--sans);letter-spacing:.06em;text-transform:uppercase;
  border-radius:var(--r-s);padding:1px 6px;white-space:nowrap}
.flag{color:var(--muted);border:1px solid var(--line)}
.flag.warn{color:rgba(var(--warm-tx),1);border-color:rgba(var(--warm),.45)}

/* ---------- lineup: order rail + shared-axis edge bars ---------- */
details.lineup{margin-top:11px;border-top:1px solid var(--line-2)}
details.lineup summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:8px;
  padding:8px 0 6px;user-select:none}
details.lineup summary::-webkit-details-marker{display:none}
details.lineup summary:focus-visible{outline:2px solid rgba(var(--lean),1);outline-offset:2px}
summary .tl{font:700 13px/1.4 var(--sans);letter-spacing:.12em;text-transform:uppercase}
summary .st{font:600 12.5px/1.4 var(--sans);letter-spacing:.08em;border-radius:var(--r-s);padding:1px 6px}
summary .st.posted{color:rgba(var(--cool-tx),1);border:1px solid rgba(var(--cool),.45)}
summary .st.partial{color:rgba(var(--lean-tx),1);border:1px solid rgba(var(--amberbg),.5)}
summary .st.projected{color:var(--faint);border:1px solid var(--line)}
summary .lw{margin-left:auto;font:500 13px/1.4 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
summary .chev{font-size:12.5px;color:var(--faint);transition:transform .12s}
details[open] summary .chev{transform:rotate(90deg)}
@media (prefers-reduced-motion:reduce){summary .chev{transition:none}}

.lu-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.lu{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
table.lu td{font:400 14px/1.4 var(--mono);text-align:right;padding:4px 6px;
  border-bottom:1px solid var(--line-2);white-space:nowrap}
table.lu tr:last-child td{border-bottom:0}
td.ord{font-size:13px;color:var(--faint);width:26px;text-align:center;border-right:1px solid var(--line-2)}
td.pos{color:var(--muted);font-size:12.5px}
td .pa{color:var(--faint);font-size:12.5px}
td.na{color:var(--faint)}
tr.low td{color:var(--muted)}

td.bar{width:86px;padding:4px 8px 4px 2px}

/* ============================================================
   Card layer: percentile bars, plain-language read, verdict, and the
   full model machinery (always shown).
   ============================================================ */
.matchlab{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);margin-bottom:6px}
.read{margin:0;padding:12px 16px;font:500 16px/1.5 var(--sans);color:var(--ink);
  border-bottom:1px solid var(--line-2)}
.read b{font-weight:700}
.read .warmtx{color:rgba(var(--warm-tx),1)} .read .cooltx{color:rgba(var(--cool-tx),1)}

/* percentile slider (fill length = percentile; hue = whose favor) */
.sl{display:inline-block;width:88px;height:9px;border-radius:var(--r-s);background:var(--surface-2);
  border:1px solid var(--line-2);vertical-align:middle;overflow:hidden;position:relative}
.sl i{display:block;height:100%;border-radius:var(--r-s) 0 0 var(--r-s)}
.sl.h i{background:rgba(var(--warm),.85)} .sl.p i{background:rgba(var(--cool),.85)}

/* starter percentile bars + quality tier */
.sp-bars{margin-top:9px}
.spct{display:flex;align-items:center;gap:9px;margin-top:6px}
.spct .lab{font:600 12px/1.4 var(--sans);letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);min-width:64px}
/* `.tier` type + box metrics are shared with `.flag` up in the SP block. */
.tier.elite{color:rgba(var(--cool-tx),1);border:1px solid rgba(var(--cool),.5);background:rgba(var(--cool),.10)}
.tier.solid{color:var(--muted);border:1px solid var(--line)}
.tier.below{color:rgba(var(--warm-tx),1);border:1px solid rgba(var(--warm),.5);background:rgba(var(--warm),.08)}

/* spotlight (top bats by percentile) */
.spot{display:flex;flex-wrap:wrap;align-items:baseline;gap:5px 8px;margin:12px 0 2px}
.spot .sl-lab{font:600 12px/1.4 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.spot .pill{font:600 13px/1.3 var(--mono);color:var(--ink);background:var(--surface-2);
  border:1px solid var(--line-2);border-radius:var(--r-pill);padding:2px 9px;font-variant-numeric:tabular-nums}

/* model-vs-market verdict — its own full-width row beneath the odds, at every
   width. The prose grows downward instead of stealing width from the numbers. */
.verdict{padding:7px 16px;border-top:1px solid var(--line-2);border-left:3px solid var(--line)}
.verdict .l{color:var(--muted)} .verdict .vt{font:600 14px/1.4 var(--sans);color:var(--muted);margin-top:2px}
.verdict.edge{border-left-color:rgba(var(--lean),1);background:rgba(var(--amberbg),.10)}
.verdict.edge .l{color:rgba(var(--lean-tx),1)} .verdict.edge .vt{color:var(--ink)}
.verdict .vline{display:flex;align-items:baseline;gap:6px;font-variant-numeric:tabular-nums}
.verdict .vline + .vline{margin-top:2px}
.verdict .vk{flex:0 0 auto;color:var(--muted);font-weight:700}
.verdict .hist{display:block;margin-top:7px;padding-top:6px;border-top:1px solid var(--line-2)}
.verdict .hist .vk,.verdict .hist>span:last-child{display:block}
.verdict .hist>span:last-child{margin-top:1px;color:var(--ink)}
.verdict .vprofile{margin-top:7px;padding-top:7px;border-top:1px solid var(--line-2)}
.verdict .vprofile-title{font:700 11px/1.3 var(--sans);letter-spacing:.13em;color:var(--muted)}
.verdict .vprofile-band{margin:2px 0 6px;color:var(--muted);font-weight:600;
  font-size:12.5px;line-height:1.35}
/* Zone heading inside the 'this game' half, so the two halves of the
   panel are visibly separate rather than one list a reader carries a
   historical rate down through. */
.verdict .vprofile-title.vgroup{margin-bottom:3px}
/* One-line plain-English reason under the line it explains. Indented
   to the value column so it reads as a note on that row, not a new row. */
.verdict .vnote{margin:2px 0 1px;font:500 12.5px/1.4 var(--sans);
  color:var(--faint)}
.verdict .vprofile .vline{justify-content:space-between;gap:14px;font-weight:600}
.verdict .vprofile .vline>span:last-child{text-align:right;color:var(--ink)}

/* hitter row: percentile column + name cell. The column is the 88px bar plus
   the cell's own gutters -- it carried a printed percentile until that was
   dropped, and the freed width goes to the name column. */
td.pct{width:104px;white-space:nowrap}
table.lu td.nm,table.lu td.pos{text-align:left}
/* Qualified so it outranks `table.lu td`'s mono face (0,1,2): hitter names are
   sans, the numbers around them stay mono. A bare `td.nm` never won this. */
table.lu td.nm{font:400 14px/1.4 var(--sans);max-width:170px;
  overflow:hidden;text-overflow:ellipsis}
td.nm .b{font:400 12px/1 var(--mono);color:var(--muted);margin-left:4px}
/* The hand letter IS the platoon marker: it takes the hitter bar's accent when
   this batter holds the edge over the starter. `--warm-tx` is that accent's
   contrast-audited text twin -- the bar fill token is not legible as type. */
td.nm .b.adv{color:rgba(var(--warm-tx),1);font-weight:700}

/* Model machinery — always shown (analyst is the default and only view). The
   per-element display types match how the old Analyst toggle revealed them. */
.mach{display:block}
span.mach{display:inline}
td.mach,th.mach{display:table-cell}
tr.mach{display:table-row}
/* Keep the three starter stat cells in one horizontal strip. This specific
   flex rule must follow .mach so the generic display:block hook cannot stack
   xwOBA-against, K-BB%, and xERA vertically. */
.spstats.mach{display:flex}

/* ============================================================
   Phone layout (single breakpoint, last in the cascade)
   ------------------------------------------------------------
   One block, deliberately placed after every base rule: media queries add no
   specificity, so a mobile override declared earlier than its desktop
   counterpart silently loses (.sl and td.pct did exactly that). Keep new
   phone rules here.
   ============================================================ */
@media (max-width:540px){
  /* Tighter chrome: the card gutters, not the content, give up the width. */
  body{padding:14px 10px 44px}
  .game-summary{min-height:82px;padding:10px 10px}
  /* Centre column holds the first-pitch time on one line. At the Comfortable
     scale "7:05 PM ET" measures 84px in JetBrains Mono 14px (headless
     Chromium), so the old 82px wrapped it onto two rows on every unstarted
     card. 92px clears it with slack for a longer zone label; at 320px that
     still leaves 88px a side, and the widest thing a side carries is a 30px
     logo plus a three-letter club. */
  .teams{grid-template-columns:minmax(0,1fr) 92px minmax(0,1fr);
    gap:7px;padding-right:17px}
  .summary-team{gap:2px 6px}
  .summary-team.away{grid-template-columns:36px minmax(0,1fr);
    grid-template-areas:"logo club" "logo meta"}
  .summary-team.home{grid-template-columns:minmax(0,1fr) 36px;
    grid-template-areas:"club logo" "meta logo"}
  .summary-team.away .team-meta{justify-self:start}
  .summary-team.home .team-meta{justify-self:end;text-align:right}
  .summary-team .clogo,.summary-logo-fallback{width:36px;height:36px}
  .summary-logo-fallback{font-size:14px}
  .summary-club{font-size:14px}
  .teams .team-meta{font-size:12px}
  .teams .score{font-size:24px}
  .summary-center{gap:3px}
  .summary-time{font-size:14px;white-space:nowrap}
  .summary-market{font-size:12px}
  .summary-chevron{right:6px}
  .detail-context{padding:7px 12px}
  .read{padding:10px 12px}
  .side{padding:10px 12px 12px}
  .mcell{padding:6px 7px}
  .mcell .l{font-size:12px;letter-spacing:.04em}
  .mcell .v{font-size:14px}
  .mcell .v .mv{display:block;font-size:12px}
  .verdict{padding:7px 12px}
  .spstats{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .stat{min-width:88px}
  /* Lineup fits inside the card instead of scrolling: trim the gutters, shrink
     the percentile bar, and let the name -- the one elastic column -- wrap
     instead of truncating. Every column the desktop card shows is kept. */
  table.lu td{padding:4px 3px}
  /* `table.lu td` is (0,1,2); a bare `td.nm` is (0,1,1) and loses the
     white-space battle no matter where it sits. Match the qualifier. */
  table.lu td.nm{max-width:none;white-space:normal}
  td.ord{width:18px;padding-left:0}
  td.pct{width:auto}
  td.pos{font-size:12px}
  .sl{width:46px}
}

/* ============================================================
   Touch targets (pointer axis, not width)
   ------------------------------------------------------------
   Measured in headless Chromium at 390px: the theme button rendered 28px
   tall and every lineup disclosure 32px, against a 44px floor (iOS HIG) /
   48dp (Material). The collapsed game row already clears it at 82px.
   Gated on `pointer:coarse` rather than a width breakpoint because it is
   the input device, not the viewport, that sets the floor -- a touch laptop
   at 1200px needs it and a 400px desktop window does not. Padding does the
   growing, so the type and the layout are untouched.
   ============================================================ */
@media (pointer:coarse){
  .theme{min-height:44px;padding:7px 14px}
  details.lineup summary{padding-top:14px;padding-bottom:12px}
}
"""

CSS_GRADES = r"""
.gradestrip{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:0 2px 16px;
  padding:10px 14px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
  font:600 14px/1.4 var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}
.gradestrip .lab{font:600 12px/1 var(--sans);text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.gradestrip .muted{color:var(--muted);font-weight:500}
.gradestrip .grade-links{display:flex;align-items:center;gap:14px;margin-left:auto}
.gradestrip a{color:rgba(var(--lean-tx),1);text-decoration:none;font:600 14px/1 var(--sans);white-space:nowrap}
.gradestrip a:hover{text-decoration:underline}

.backlink{font:600 14px/1 var(--sans);margin:0 2px 12px}
.backlink a{color:rgba(var(--lean-tx),1);text-decoration:none}
.backlink a:hover{text-decoration:underline}
.backlink.ledger-nav{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 18px}

/* ---------- grades page: masthead ---------- */
.gr-head{margin:0 2px 13px}
.gr-h1{font:800 19px/1.15 var(--sans);letter-spacing:-.01em;margin:0 0 5px}
.gr-lead{font:500 14px/1.55 var(--sans);color:var(--muted);max-width:80ch;margin:0}
.gr-lead b{color:var(--ink);font-weight:700}
/* `9:34 PM · 2026-07-28` is one token. Left to wrap it breaks at the
   date's own hyphen, so a phone renders `2026-` over `07-28`. */
.gr-lead .stamp{white-space:nowrap}

/* ---------- summary: one stat strip, same idiom as a card's SP stat row ---------- */
.gr-summary{display:flex;flex-wrap:wrap;margin:0 0 9px;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.gr-stat{flex:1 1 132px;min-width:0;padding:9px 14px 8px;border-right:1px solid var(--line-2)}
.gr-stat:last-child{border-right:0}
.gr-stat .l{font:600 12px/1.4 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.gr-stat .v{font:600 19px/1.2 var(--mono);font-variant-numeric:tabular-nums;
  color:var(--chip-fg);margin-top:3px}
.gr-stat .s{font:400 12.5px/1.35 var(--sans);font-variant-numeric:tabular-nums;
  color:var(--muted);margin-top:2px}
.gr-stat .v.cool{color:rgba(var(--cool-tx),1)}
.gr-stat .v.warm{color:rgba(var(--warm-tx),1)}
/* Controls sit in the same strip as the model's own record but must not read
   as a headline: same geometry, muted weight and colour. */
.gr-stat .v.dim{color:var(--muted);font-weight:500}
.gr-note{font:500 13px/1.55 var(--sans);color:var(--faint);margin:0 2px 14px}
.gr-note b{color:var(--muted);font-weight:700}

/* ---------- ledger table ---------- */
/* overflow:clip, not hidden -- `hidden` makes this a scroll container and the
   sticky column head then sticks to a box with no scrollable overflow, i.e.
   never. `clip` still rounds off the table corners. */
.gr-tablewrap{background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r);box-shadow:var(--shadow);overflow:clip}
/* Mono is for figures that line up in a column -- the lean deltas, the finals,
   the W/L badges -- and nothing else. The prose around them (the methodology
   note, the pitcher matchups, the day headers, the stat captions) is sans:
   at the Comfortable scale the note alone ran to a full screen of monospace
   on a phone, which reads as a code block rather than a sentence. */
table.gr{border-collapse:collapse;width:100%;table-layout:auto;
  font:500 14px/1.35 var(--mono);font-variant-numeric:tabular-nums}
table.gr th{font:600 12px/1 var(--sans);text-transform:uppercase;letter-spacing:.12em;color:var(--faint);
  text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;
  position:sticky;top:0;background:var(--surface);z-index:2}
table.gr td{padding:9px 12px;border-bottom:1px solid var(--line-2);white-space:nowrap;vertical-align:middle}
table.gr tbody tr:last-child td{border-bottom:none}
table.gr tbody tr.gr-row:hover td{background:var(--surface-2)}
table.gr tr.void td{opacity:.45}
table.gr .sp{display:block;margin-top:2px;font:400 12.5px/1.3 var(--sans);color:var(--faint)}
/* Marks the rows where the published rule departed from the model's own
   lean. Warm, because it is the one thing on the row a reader scanning the
   archive needs to spot; deliberately not a badge, so it cannot read as a
   recommendation. */
table.gr .sp.fade-mark{display:inline-block;margin:2px 0 0 6px;padding:0 5px;
  border-radius:4px;font:700 10.5px/1.6 var(--sans);letter-spacing:.1em;
  color:rgba(var(--warm-tx),1);background:rgba(var(--warm-tx),.10)}
.gr-game{font:650 14px/1.3 var(--sans)}
.gr-game .at{color:var(--faint);font-weight:500}

/* Team ledger reuses the responsive ledger table, but each metric cell keeps
   the record, hit rate and sample size together so the percentage is never
   detached from its denominator. */
table.team-gr .tm-record{font-weight:700;color:var(--ink);margin-right:8px}
table.team-gr .tm-record.cool{color:rgba(var(--cool-tx),1)}
table.team-gr .tm-record.warm{color:rgba(var(--warm-tx),1)}
table.team-gr .tm-accuracy{font-weight:650;color:rgba(var(--lean-tx),1)}
table.team-gr .tm-n{font:500 12px/1 var(--sans);color:var(--faint);margin-left:6px}
table.team-gr .team-code{font:700 15px/1.2 var(--mono);color:var(--ink)}

/* Date group header -- the 300+ rows read as one undifferentiated scroll
   without it. Sticks under the column head (28px = its padded line box). */
tr.gr-day th{position:sticky;top:28px;z-index:1;background:var(--surface-2);
  padding:6px 12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line-2);
  font:700 12.5px/1.4 var(--sans);font-variant-numeric:tabular-nums;
  letter-spacing:.02em;text-transform:none;color:var(--muted)}
tr.gr-day .d{color:var(--ink)}
tr.gr-day .n{color:var(--faint);font-weight:500}
tr.gr-day .rec{float:right;color:var(--muted)}

/* ---------- season leaderboard ---------- */
/* Reuses the ledger table wholesale -- wrap, sticky head, and the phone
   relabelling -- so a second table idiom never enters the site. Only the
   rank/name/value trio inside the first two cells is new. */
.lb-head{margin-top:22px}
/* The Batters divider is the one break between two boards that mean different
   things -- one is "lowest allowed", the other "highest". Without it the first
   position table reads as a continuation of the pitcher list. */
.lb-split{margin-top:30px;padding-top:18px;border-top:1px solid var(--line)}
.lb-rank{display:inline-block;min-width:2.4ch;margin-right:10px;
  font:600 13px/1 var(--mono);font-variant-numeric:tabular-nums;color:var(--faint)}
.lb-name{font:650 14px/1.3 var(--sans);color:var(--ink)}
.lb-val{font-weight:700;color:rgba(var(--lean-tx),1)}
/* Row count, not the cap. Sits in the heading so a short board is legible as
   short rather than looking like a truncated one. */
.lb-n{margin-left:9px;font:600 12px/1 var(--mono);font-variant-numeric:tabular-nums;
  color:var(--faint);vertical-align:.12em}

/* ---------- badges ---------- */
.wlt{display:inline-block;min-width:17px;text-align:center;font:700 12.5px/1 var(--mono);
  padding:3px 6px;border-radius:var(--r-s)}
.wlt.W{color:rgba(var(--cool-tx),1);background:rgba(var(--cool),.12);border:1px solid rgba(var(--cool),.3)}
.wlt.L{color:rgba(var(--warm-tx),1);background:rgba(var(--warm),.12);border:1px solid rgba(var(--warm),.3)}
.wlt.T{color:var(--muted);background:var(--surface-2);border:1px solid var(--line)}
.wlt.none{color:var(--faint);background:transparent;border:1px dashed var(--line)}
.st{font:600 12px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.st.void{color:rgba(var(--warm-tx),.9)}

/* ---------- phone: each row becomes a labelled block ----------
   The table used to carry min-width:760px, so every phone view of the ledger
   was a horizontal scroll. Cells relabel themselves from data-l instead. */
@media (max-width:720px){
  table.gr thead{position:absolute;width:1px;height:1px;margin:-1px;overflow:hidden;
    clip:rect(0 0 0 0);white-space:nowrap}
  table.gr,table.gr tbody,table.gr tr,table.gr td,table.gr th{display:block}
  tr.gr-day th{position:static;top:auto}
  table.gr tr.gr-row{display:grid;grid-template-columns:minmax(0,1fr) auto;
    column-gap:10px;row-gap:4px;padding:11px 13px;border-bottom:1px solid var(--line-2)}
  table.gr tbody tr.gr-row:last-child{border-bottom:0}
  table.gr tr.gr-row td{border:0;padding:0;white-space:normal}
  table.gr td.c-game{grid-column:1;grid-row:1;min-width:0}
  table.gr td.c-res{grid-column:2;grid-row:1;justify-self:end;align-self:start}
  /* Remaining cells stack full-width under the game, each relabelled from
     data-l. Fixed label column so the values line up down the block. These
     need the table.gr prefix to outrank the display:block reset above. */
  table.gr td.c-lean,table.gr td.c-ml,table.gr td.c-final{
    grid-column:1/-1;display:flex;align-items:baseline;gap:9px;color:var(--muted)}
  table.gr td.c-lean::before,table.gr td.c-ml::before,table.gr td.c-final::before{
    content:attr(data-l);flex:0 0 46px;font:600 12px/1.7 var(--sans);
    letter-spacing:.1em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
}
"""

THEME_JS = r"""
(function(){
  var KEY='mlb-mx-theme';
  var btn=document.getElementById('themeBtn');
  function cur(){return document.documentElement.getAttribute('data-theme')||'auto';}
  function apply(m){
    if(m==='auto'){document.documentElement.removeAttribute('data-theme');}
    else{document.documentElement.setAttribute('data-theme',m);}
    if(btn){btn.textContent='Theme: '+m;}
  }
  try{apply(localStorage.getItem(KEY)||'auto');}catch(e){apply('auto');}
  if(btn){btn.addEventListener('click',function(){
    var order=['auto','light','dark'];var n=order[(order.indexOf(cur())+1)%3];
    apply(n);try{localStorage.setItem(KEY,n);}catch(e){}
  });}
})();
"""


def score_refresh_js():
    """Browser-side score refresh without rerunning the model build."""
    return rf"""
(function(){{
  var url='https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={SLATE_DATE}&hydrate=linescore';
  var headers={{}};
  document.querySelectorAll('.teams[data-game-pk]').forEach(function(el){{
    headers[String(el.getAttribute('data-game-pk'))]=el;
  }});
  function setMeta(el,score,state,otherScore){{
    if(!el) return;
    if(state==='Live'||state==='Final'){{
      el.hidden=false;
      el.textContent=(score===null||score===undefined)?'—':String(score);
      // This rewrites className wholesale, so the loser class has to be
      // recomputed here or a refresh would silently undo the dimming.
      var lost=state==='Final'&&score!==null&&score!==undefined
        &&otherScore!==null&&otherScore!==undefined&&Number(score)<Number(otherScore);
      el.className='team-meta score '+(el.classList.contains('home')?'home':'away')
        +(lost?' lost':'');
      el.title=state==='Live'?'live score':'final score';
    }}else{{
      var streak=el.getAttribute('data-streak')||'';
      el.textContent=streak;
      el.className='team-meta streak '+(el.classList.contains('home')?'home':'away');
      el.title='current win/loss streak';
      el.hidden=!streak;
    }}
  }}
  function baseOutText(bases,outs){{
    var named=['first','second','third'].filter(function(n,i){{return bases[i];}});
    var runners=named.length===0?'bases empty':
      named.length===3?'bases loaded':
      named.length===1?'runner on '+named[0]:
      'runners on '+named.join(' and ');
    return outs===null||outs===undefined?runners:String(outs)+' out, '+runners;
  }}
  function setBaseOut(el,linescore,live){{
    if(!el) return;
    el.hidden=!live;
    if(!live) return;
    var offense=linescore.offense||{{}};
    var bases=[!!offense.first,!!offense.second,!!offense.third];
    var outs=(linescore.outs===null||linescore.outs===undefined)?null:Number(linescore.outs);
    ['b1','b2','b3'].forEach(function(cls,i){{
      var base=el.querySelector('.diamond .'+cls);
      if(base) base.className=cls+(bases[i]?' on':'');
    }});
    el.querySelectorAll('.outs i').forEach(function(dot,i){{
      dot.className=(outs!==null&&i<outs)?'on':'';
    }});
    var sr=el.querySelector('.bo-sr');
    if(sr) sr.textContent=baseOutText(bases,outs);
  }}
  function update(game){{
    var header=headers[String(game.gamePk)];
    if(!header) return;
    var status=game.status||{{}};
    var state=status.abstractGameState||'Preview';
    var teams=game.teams||{{}};
    setBaseOut(header.querySelector('.bo'),game.linescore||{{}},state==='Live');
    var awayScore=(teams.away||{{}}).score,homeScore=(teams.home||{{}}).score;
    setMeta(header.querySelector('.team-meta.away'),awayScore,state,homeScore);
    setMeta(header.querySelector('.team-meta.home'),homeScore,state,awayScore);
    var inProgress=state==='Live'||state==='Final';
    var summaryTime=header.querySelector('.summary-time');
    var summaryMarket=header.querySelector('.summary-market');
    if(summaryTime) summaryTime.hidden=inProgress;
    if(summaryMarket) summaryMarket.hidden=inProgress;
    var badge=header.querySelector('.game-state');
    if(!badge) return;
    if(inProgress){{
      badge.hidden=false;
      if(state==='Live'){{
        var linescore=game.linescore||{{}};
        var ordinal=linescore.currentInningOrdinal||'';
        if(!ordinal&&linescore.currentInning!==null&&linescore.currentInning!==undefined){{
          var n=Number(linescore.currentInning);
          var mod100=n%100;
          var suffix=(mod100>10&&mod100<14)?'th':({{1:'st',2:'nd',3:'rd'}}[n%10]||'th');
          ordinal=String(n)+suffix;
        }}
        var half=String(linescore.inningHalf||'').toLowerCase();
        var marker=half==='top'?'▲':(half==='bottom'?'▼':'');
        if(!marker&&linescore.isTopInning===true) marker='▲';
        if(!marker&&linescore.isTopInning===false) marker='▼';
        badge.textContent='LIVE'+(ordinal?' · '+marker+ordinal:'');
      }}else{{
        badge.textContent='FINAL';
      }}
      badge.className='game-state '+state.toLowerCase();
      badge.title=status.detailedState||'';
    }}else{{
      badge.hidden=true;
      badge.textContent='';
      badge.className='game-state';
      badge.title='';
    }}
  }}
  function refresh(){{
    if(document.hidden) return;
    fetch(url,{{cache:'no-store'}})
      .then(function(r){{if(!r.ok) throw new Error(String(r.status));return r.json();}})
      .then(function(data){{
        (data.dates||[]).forEach(function(day){{
          (day.games||[]).forEach(update);
        }});
      }})
      .catch(function(){{}});
  }}
  refresh();
  window.setInterval(refresh,60000);
}})();
"""


def html_document(body, built_txt, title=None, extra_js=None):
    title = title or f"{PUBLIC_MODEL_NAME} - {SLATE_DATE}"
    extra_script = f"<script>{extra_js}</script>" if extra_js else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        "<meta name='robots' content='noindex'>"
        f"<style>{font_face_css()}{CSS}{CSS_GRADES}</style></head><body>"
        f"<div class='mx-wrap'>"
        f"<div class='topbar'><div class='brand'>{PUBLIC_MODEL_NAME}</div>"
        "<div style='display:flex;gap:8px'>"
        "<button id='themeBtn' class='theme' type='button'>Theme: auto</button></div></div>"
        f"{body}</div>"
        f"<script>{THEME_JS}</script>"
        f"{extra_script}"
        "</body></html>")


# ============================================================
# GRADES -- records strip + full ledger + per-team accuracy pages
# Renders purely from data/mlb_lean_ledger.csv, which grade_leans.py
# maintains and CI commits back to the repo; the grading pass runs
# before this build so records are current as of the run.
# ============================================================
def load_ledger_df():
    if not os.path.exists(LEDGER_PATH):
        return None
    try:
        led = pd.read_csv(LEDGER_PATH)
    except Exception as e:  # noqa: BLE001
        log(f"Ledger unreadable, grades render degraded: {e!r}")
        return None
    return None if led.empty else led


def _esc(x):
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _rec_parts(s):
    """W/L/T series -> ('W-L' or 'W-L-T', 'pct' or None)."""
    s = s.dropna()
    w, l, t = int((s == "W").sum()), int((s == "L").sum()), int((s == "T").sum())
    base = f"{w}-{l}" + (f"-{t}" if t else "")
    return base, (f"{w / (w + l):.3f}" if (w + l) else None)


def _rec_txt(s):
    base, pct = _rec_parts(s)
    return f"{base} ({pct})" if pct else base


def _ledger_club_labels():
    """Ledger abbreviation -> familiar club label for the 30 MLB clubs.

    The model's display map follows StatsAPI's Arizona abbreviation (AZ),
    while grade_leans.py persists ESPN/ledger-style ARI. `ledger_abbr` owns
    that one boundary; this reads the ledger, so it goes through it too.
    Building the set from the existing club maps also keeps
    exhibition abbreviations such as AME/NAT off the team page without a
    second hand-maintained list of clubs.
    """
    out = {}
    for full_name, stats_abbr in ABBR.items():
        out[ledger_abbr(stats_abbr)] = TEAM_LABELS.get(full_name, full_name)
    return out


def _team_record_parts(frame):
    """Raw W/L/T counts for one team-page slice."""
    grade = frame["xw_full"]
    return {
        "n": int(len(frame)),
        "w": int(grade.eq("W").sum()),
        "l": int(grade.eq("L").sum()),
        "t": int(grade.eq("T").sum()),
    }


_ODDS_LADDER = (
    (None, -250, "≤ -250"),
    (-249, -175, "-249 to -175"),
    (-174, -130, "-174 to -130"),
    (-129, -100, "-129 to -100"),
    (100, 129, "+100 to +129"),
    (130, 174, "+130 to +174"),
    (175, 249, "+175 to +249"),
    (250, None, "≥ +250"),
)


def _ladder_rung(ml):
    """Rung label for an American price, or None if it is not a valid one.

    American odds never fall strictly between -100 and +100, so the rungs above
    tile every representable price. A None return therefore means the input was
    not a real moneyline, and the caller drops it rather than guessing."""
    for lo, hi, label in _ODDS_LADDER:
        if (lo is None or ml >= lo) and (hi is None or ml <= hi):
            return label
    return None


def _excess_se(probs):
    """SE of (realised rate − mean implied) under correctly priced games.

    Both surfaces on market-calibration.html ask the same question -- did this
    bucket beat its own price -- so the standard error beside them has one
    derivation rather than two that can drift.

    Each observation is an independent Bernoulli at its own devigged price, so
    the win count is Poisson-binomial: Var(Σ wins) = Σ p(1−p), and the SE of
    the mean is sqrt(Σ p(1−p))/n.

    It deliberately does NOT estimate the spread from the outcomes. The
    obvious sqrt(p̂(1−p̂)/n) does, and therefore returns exactly 0.0 on any
    bucket that went all-W or all-L -- rendering the least certain buckets on
    the page as the most certain, which is the direction that makes noise look
    like signal. The sample sd of the residuals fails the same way for the
    same reason. Here the p_i are fixed by the market rather than estimated
    from the outcomes under test, so this is defined at n=1 and cannot
    degenerate.
    """
    p = np.asarray(list(probs), dtype=float)
    if not p.size:
        return np.nan
    return float(math.sqrt(float((p * (1.0 - p)).sum())) / p.size)


def _market_calibration_rows(led):
    """Devigged closing implied % against realised win %, by side and price.

    One graded game contributes TWO observations -- the home side at its close
    and the away side at its close -- because each side is its own bet and the
    question is asked per price, not per game. That is deliberate double
    counting of games, not of bets, and the note on the page says so.

    `totals` therefore carries `home`, `away` and `favourite` -- and NOT a
    pooled both-sides figure, which the double counting makes an identity.
    See the comment beside its construction below.

    `close_p_home` is already devigged by market_backfill, so `implied` is a
    fair probability and the gap to the raw price is the vig. Rows without a
    close are skipped rather than imputed; ties/unplayed rows never enter.
    """
    if led is None or "close_p_home" not in led.columns:
        return [], {}
    fa = pd.to_numeric(led.get("full_away"), errors="coerce")
    fh = pd.to_numeric(led.get("full_home"), errors="coerce")
    ph = pd.to_numeric(led["close_p_home"], errors="coerce")
    hml = pd.to_numeric(led.get("close_home_ml"), errors="coerce")
    aml = pd.to_numeric(led.get("close_away_ml"), errors="coerce")
    ok = fa.notna() & fh.notna() & (fa != fh) & ph.notna() & hml.notna() & aml.notna()
    if not ok.any():
        return [], {}
    home_won = (fh > fa)[ok].to_numpy()
    obs = []
    for side, ml, imp, won in (
        ("home", hml[ok].to_numpy(), ph[ok].to_numpy(), home_won),
        ("away", aml[ok].to_numpy(), (1.0 - ph[ok]).to_numpy(), ~home_won),
    ):
        for m, p, w in zip(ml, imp, won):
            rung = _ladder_rung(float(m))
            if rung is not None:
                obs.append((rung, side, float(p), bool(w)))

    def agg(sel):
        n = len(sel)
        if not n:
            return None
        wins = sum(1 for _, _, _, w in sel if w)
        imp = sum(p for _, _, p, _ in sel) / n
        act = wins / n
        return dict(n=n, w=wins, implied=imp, actual=act,
                    se=_excess_se(p for _, _, p, _ in sel),
                    diff=act - imp)

    rows = []
    for _lo, _hi, label in _ODDS_LADDER:
        at = [o for o in obs if o[0] == label]
        if not at:
            continue
        rows.append(dict(
            rung=label,
            home=agg([o for o in at if o[1] == "home"]),
            away=agg([o for o in at if o[1] == "away"]),
            all=agg(at),
        ))
    totals = {s: agg([o for o in obs if o[1] == s]) for s in ("home", "away")}
    # NOT a pooled both-sides total. Pooling every observation is forced to
    # .500 on both columns -- the two devigged sides of a game sum to 1 and
    # exactly one of them wins -- so it is an identity rather than a
    # measurement, and it used to lead this page with the tightest error bar
    # on it. The invariant is worth pinning and is, in
    # `test_pooling_both_sides_is_an_identity_not_a_measurement`; it is not
    # worth publishing.
    #
    # The favourite pool is the non-degenerate version of the same question,
    # and it is the one the page's own note names: one observation per game
    # rather than two, so nothing cancels. `p > .5` selects exactly one side
    # of every priced game and drops both sides of a pick'em, which has no
    # favourite to grade.
    totals["favourite"] = agg([o for o in obs if o[2] > 0.5])
    return rows, totals


def _calib_cell(parts):
    """Realised vs implied for one bucket, with the sample size it rests on."""
    if not parts:
        return "<span class='tm-n'>—</span>"
    d = parts["diff"]
    tone = "cool" if d > 0 else ("warm" if d < 0 else "")
    return (f"<span class='tm-record{(' ' + tone) if tone else ''}'>"
            f"{100 * parts['actual']:.1f}%</span>"
            f"<span class='tm-accuracy'>vs {100 * parts['implied']:.1f}% implied "
            f"({d * 100:+.1f})</span>"
            f"<span class='tm-n'>n={parts['n']} · ±{100 * parts['se']:.1f}</span>")


def _american_unit_profit(ml, won):
    """Flat one-unit P/L for a settled American-moneyline bet.

    The market-value diagnostic is intentionally a price diagnostic rather
    than another win-rate table. A favourite can win more often than it loses
    and still be a bad bet, while a plus-money dog can lose more games than it
    wins and still return a profit. Invalid/non-American prices return NaN and
    are excluded instead of being coerced into a payout.
    """
    if ml is None or pd.isna(ml):
        return np.nan
    ml = float(ml)
    if (-100 < ml < 100) or ml == 0:
        return np.nan
    if not won:
        return -1.0
    return ml / 100.0 if ml > 0 else 100.0 / abs(ml)


def _lean_market_observations(led):
    """One row per current-family full-game lean with a devigged DK close.

    This is deliberately scoped to `_record_grades`: unlike the market-only
    calibration ladder, this panel asks whether *this prediction family* adds
    information relative to price, so pooling old prediction math would answer
    the wrong question.

    `market_p` is the closing no-vig probability of the LEANED team, not always
    the home team. `market_edge = market_p - .50` keeps the direction: positive
    means the market agrees and prices the lean as a favourite; negative means
    the model is leaning an underdog. That sign is the whole hybrid axis, so it
    is never collapsed into a distance-from-even.

    Every column the hybrid rule needs is derived here rather than at each
    surface, for the reason `_conviction_cell` used to give and this replaces:
    two copies of a selection rule on one site will drift and the reader cannot
    see which they are reading. `opp_ml` is what makes that possible -- a faded
    row is BET AT THE OTHER SIDE'S PRICE, so a frame carrying only the leaned
    side's moneyline cannot score the rule that this site now publishes.

    The always-chalk columns ride along for a reason specific to this rule: the
    fade branch backs the favourite on every row by construction, so its record
    IS the chalk record on those rows. Deriving both here means the two can
    never be computed from different row sets, and every surface that shows a
    fade result can show the control beside it.

    The ledger has no per-game SE/SD for xw_delta. Do not manufacture one from
    lineup dispersion or opponent-rate SD -- those are different quantities.
    `delta` is retained as a reported quantity, but it is no longer a bucketing
    axis: the hybrid reads a DIRECTION and a PRICE, never the delta's size.
    """
    cols = {"close_p_home", "close_home_ml", "close_away_ml",
            "xw_lean", "xw_full", "home", "away"}
    if led is None or not cols.issubset(led.columns):
        return pd.DataFrame()
    g = _record_grades(led).copy()
    if g.empty:
        return pd.DataFrame()

    ph = pd.to_numeric(g["close_p_home"], errors="coerce")
    hml = pd.to_numeric(g["close_home_ml"], errors="coerce")
    aml = pd.to_numeric(g["close_away_ml"], errors="coerce")
    if "xw_delta" in g.columns:
        delta = pd.to_numeric(g["xw_delta"], errors="coerce").abs()
    elif "xw_net" in g.columns:
        delta = pd.to_numeric(g["xw_net"], errors="coerce").abs()
    else:
        return pd.DataFrame()

    home_lean = g["xw_lean"].eq(g["home"])
    away_lean = g["xw_lean"].eq(g["away"])
    settled = g["xw_full"].isin(["W", "L"])
    valid = ((home_lean | away_lean) & settled & ph.notna()
             & hml.notna() & aml.notna() & delta.notna())
    if not valid.any():
        return pd.DataFrame()

    gv = g.loc[valid]
    hv = home_lean.loc[valid].to_numpy()
    pv = ph.loc[valid].to_numpy(dtype=float)
    market_p = np.where(hv, pv, 1.0 - pv)
    close_ml = np.where(hv, hml.loc[valid].to_numpy(dtype=float),
                        aml.loc[valid].to_numpy(dtype=float))
    opp_ml = np.where(hv, aml.loc[valid].to_numpy(dtype=float),
                      hml.loc[valid].to_numpy(dtype=float))
    won = gv["xw_full"].eq("W").to_numpy(dtype=bool)
    dv = delta.loc[valid].to_numpy(dtype=float)

    obs = pd.DataFrame({
        "delta": dv,
        "market_p": market_p,
        "close_ml": close_ml,
        "opp_ml": opp_ml,
        "won": won.astype(float),
    })
    obs = obs[np.isfinite(obs["delta"]) & np.isfinite(obs["market_p"])
              & np.isfinite(obs["close_ml"]) & np.isfinite(obs["opp_ml"])
              & obs["market_p"].between(0.0, 1.0, inclusive="neither")].copy()
    if obs.empty:
        return obs
    obs["market_edge"] = obs["market_p"] - 0.50
    obs["market_resid"] = obs["won"] - obs["market_p"]
    obs["profit"] = [
        _american_unit_profit(ml, bool(w))
        for ml, w in zip(obs["close_ml"], obs["won"])
    ]

    # --- the published rule, derived once ---------------------------------
    # `>=` follows, so a game priced at exactly the threshold follows the
    # model. That is the specification, and hybrid_test pins the same
    # comparison on the forward-scoring side.
    follow = obs["market_p"].to_numpy(dtype=float) >= HYBRID_THRESHOLD
    lean_won = obs["won"].to_numpy(dtype=float)
    obs["hybrid_follow"] = follow
    obs["hybrid_won"] = np.where(follow, lean_won, 1.0 - lean_won)
    obs["hybrid_p"] = np.where(follow, obs["market_p"], 1.0 - obs["market_p"])
    obs["hybrid_ml"] = np.where(follow, obs["close_ml"], obs["opp_ml"])
    obs["hybrid_resid"] = obs["hybrid_won"] - obs["hybrid_p"]
    obs["hybrid_profit"] = [
        _american_unit_profit(ml, bool(w))
        for ml, w in zip(obs["hybrid_ml"], obs["hybrid_won"])
    ]

    # --- controls, on the IDENTICAL rows -----------------------------------
    # Always-chalk is the control the fade branch has to be read against --
    # that branch backs the favourite on every row by construction, so the two
    # must come out equal. Always-home is the coin-flip yardstick.
    #
    # Both are derived HERE, from the same frame as the record, so no surface
    # can score a control over a different row set than the number beside it.
    # That is not hypothetical: this page once scored its controls on every
    # graded row while scoring the model on the decided ones, and the `n=`
    # marker written to catch exactly that was blind to it.
    chalk_is_lean = obs["market_p"].to_numpy(dtype=float) >= 0.50
    obs["chalk_won"] = np.where(chalk_is_lean, lean_won, 1.0 - lean_won)
    obs["chalk_p"] = np.where(chalk_is_lean, obs["market_p"], 1.0 - obs["market_p"])
    chalk_ml = np.where(chalk_is_lean, obs["close_ml"], obs["opp_ml"])
    obs["chalk_resid"] = obs["chalk_won"] - obs["chalk_p"]
    obs["chalk_profit"] = [
        _american_unit_profit(ml, bool(w))
        for ml, w in zip(chalk_ml, obs["chalk_won"])
    ]

    obs["home_won"] = np.where(hv, lean_won, 1.0 - lean_won)
    obs["home_p"] = np.where(hv, obs["market_p"], 1.0 - obs["market_p"])
    home_ml = np.where(hv, obs["close_ml"], obs["opp_ml"])
    obs["home_resid"] = obs["home_won"] - obs["home_p"]
    obs["home_profit"] = [
        _american_unit_profit(ml, bool(w))
        for ml, w in zip(home_ml, obs["home_won"])
    ]

    obs = obs[np.isfinite(obs["profit"]) & np.isfinite(obs["hybrid_profit"])
              & np.isfinite(obs["chalk_profit"])
              & np.isfinite(obs["home_profit"])].copy()
    return obs


def _lean_market_agg(obs, mask, won="won", p="market_p",
                     resid="market_resid", profit="profit"):
    """Compact result/price summary for one bucket of the observation frame.

    Parameterised by column set so the model's own lean, the published hybrid
    selection and the always-chalk control are all summarised by ONE function
    over ONE row mask. Three copies of this arithmetic is how a control ends up
    scored on a different row set than the record beside it -- which happened
    here once already, when the grades page scored controls on every graded row
    and the model on the decided ones.
    """
    s = obs.loc[mask]
    if s.empty:
        return None
    n = int(len(s))
    w = int(s[won].sum())
    # Poisson-binomial at the market's own prices -- see `_excess_se` for why
    # no surface here may estimate this from the outcomes it is testing.
    resid_se = _excess_se(s[p])
    return {
        "n": n,
        "w": w,
        "l": n - w,
        "implied": float(s[p].mean()),
        "actual": float(s[won].mean()),
        "excess": float(s[resid].mean()),
        "excess_se": resid_se,
        "roi": float(s[profit].mean()),
        "units": float(s[profit].sum()),
    }


# --- the published selection rule ------------------------------------------
# The site publishes ONE rule and every surface reads it from here. The
# threshold is imported from `hybrid_test` rather than restated, because that
# module is the registration and a second literal is the "one value, three
# homes" defect this repo already paid for once: a config copy of a code
# default drifted, and 14 ledger rows carry v10 math under a v9 tag as a
# result. There must be exactly one 0.45 in this repository.
#
# This replaces a 2x3 delta x direction discovery grid AND a 3x5 delta x price
# profile grid -- 21 published cells, most of them thin. That was the surface
# `value_probe` warns about in the sharpest terms: on this ledger any grid
# search hands back a cell near +20% ROI whether or not anything is there,
# because at 21-82 rows a cell's null sd is 8-21 percentage points of ROI. The
# hybrid has two branches, so a reader is never shown a one-game headline, and
# the delta's MAGNITUDE leaves the display axis entirely -- it never entered
# the rule, and `value_probe`'s joint logit puts z(xw_net) at -0.13 against
# price, so bucketing by it was showing a reader structure that is not there.
#
# Display and monitoring only. The rule never enters a lean and does not bump
# MODEL_TAG.
HYBRID_THRESHOLD = hybrid_test.THRESHOLD

# Reported alongside the action, never used to bucket a record. Kept because a
# reader who has the delta on the card can see it is not what moved the
# selection; dropping it would hide that the rule ignores it.
_DELTA_MEDIUM = 0.012
_DELTA_HIGH = 0.025


def hybrid_action(market_p):
    """FOLLOW / FADE for the leaned side's no-vig price, or None when unusable.

    `>=` follows: a game priced at exactly the threshold follows the model.
    That is the registered specification and `hybrid_test` pins the identical
    comparison, so a live card and the forward test can never disagree about a
    boundary game.
    """
    if market_p is None:
        return None
    try:
        market_p = float(market_p)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(market_p) or not (0.0 < market_p < 1.0):
        return None
    return "FOLLOW" if market_p >= HYBRID_THRESHOLD else "FADE"


def hybrid_public_label(action):
    """Plain-language public label for an internal Hybrid branch code."""
    return {"FOLLOW": "XWOBA SIDE", "FADE": "MARKET FAVORITE"}.get(action, "")


def hybrid_selection(lean, away_abbr, home_abbr, market_p):
    """The team the published rule selects, or None when it cannot decide.

    Returns the lean itself on FOLLOW and the opposing club on FADE. Abstains
    (None) with no lean or no usable two-sided price -- a v5 abstention
    publishes no direction, so there is nothing to fade, and inventing one
    would manufacture a selection the model declined to make.
    """
    action = hybrid_action(market_p)
    if action is None or not lean:
        return None
    if action == "FOLLOW":
        return lean
    if lean == home_abbr:
        return away_abbr
    if lean == away_abbr:
        return home_abbr
    return None


def attach_hybrid_snapshot(frame, odds, snapshot_utc):
    """Stamp the public Hybrid decision and its two-sided pregame market.

    The same values are repeated on both pitcher-side rows for a game so the
    paired dump remains self-contained. Pending ledger rows may refresh these
    fields only from a later accepted pregame snapshot; once graded they are
    immutable. Historical v12 rows predate this instrumentation and remain the
    historical, close-derived archive.
    """
    if frame is None or frame.empty:
        return frame
    string_cols = ("selection_rule_tag", "hybrid_action", "hybrid_selection",
                   "pregame_market_utc")
    number_cols = ("pregame_away_ml", "pregame_home_ml", "pregame_p_home",
                   "hybrid_p", "hybrid_ml")
    for col in string_cols:
        frame[col] = None
    for col in number_cols:
        frame[col] = np.nan
    for game_pk, gg in frame.groupby("game_pk", sort=False):
        a = gg[gg["side"] == "away"]
        h = gg[gg["side"] == "home"]
        if a.empty or h.empty:
            continue
        a, h = a.iloc[0], h.iloc[0]
        home = _abbr(a.get("opp_team"))
        away = _abbr(h.get("opp_team"))
        home_off, away_off = _f(a.get("edge_xwOBA")), _f(h.get("edge_xwOBA"))
        lean = None
        if home_off is not None and away_off is not None and home_off != away_off:
            lean = home if home_off > away_off else away
        market = (odds or {}).get(game_pk) or (odds or {}).get(str(game_pk)) or {}
        p_home = _f(market.get("p_home"))
        p_lean = (p_home if lean == home else 1.0 - p_home
                  if lean == away and p_home is not None else None)
        action = hybrid_action(p_lean)
        pick = hybrid_selection(lean, away, home, p_lean)
        selected_ml = (market.get("home_ml") if pick == home else
                       market.get("away_ml") if pick == away else None)
        mask = frame["game_pk"].eq(game_pk)
        # `pick` stays in this module's namespace above so it can be compared
        # against `home`/`away`; it is translated on the way OUT because this
        # column is persisted to the ledger, where `hybrid_test` matches it
        # against the ledger's own `home`/`away` and `grade_leans._wlt` grades
        # it against them. An untranslated `AZ` matches neither club there.
        frame.loc[mask, "selection_rule_tag"] = HYBRID_RULE_TAG
        frame.loc[mask, "pregame_market_utc"] = snapshot_utc
        frame.loc[mask, "pregame_away_ml"] = market.get("away_ml")
        frame.loc[mask, "pregame_home_ml"] = market.get("home_ml")
        frame.loc[mask, "pregame_p_home"] = p_home
        frame.loc[mask, "hybrid_action"] = action
        frame.loc[mask, "hybrid_selection"] = ledger_abbr(pick) if pick else pick
        frame.loc[mask, "hybrid_p"] = p_lean if action == "FOLLOW" else (
            1.0 - p_lean if action == "FADE" and p_lean is not None else np.nan)
        frame.loc[mask, "hybrid_ml"] = selected_ml
    return frame


def _delta_label(delta):
    """LOW/MEDIUM/HIGH display band for a usable absolute model delta."""
    if delta is None:
        return None
    try:
        delta = abs(float(delta))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(delta):
        return None
    if delta < _DELTA_MEDIUM:
        return "LOW"
    if delta < _DELTA_HIGH:
        return "MEDIUM"
    return "HIGH"


def _lean_market_value_analysis(led):
    """Current-family hybrid-branch summary for the market calibration page.

    Returns the observation frame plus one row per published branch, the
    always-chalk control on the identical rows, and the model's own lean for
    reference. Every figure is a DISCOVERY result: the threshold was chosen on
    these rows, so nothing here is out-of-sample and the page says so.

    The 2x3 delta x direction matrix and the three direction-band totals this
    replaces are gone rather than relabelled. They cut the same rows nine ways
    on an axis -- the delta's magnitude -- that the published rule does not
    read, and `value_probe` puts that axis at z = -0.13 against price. Nine
    published cells on a null axis is the grid-search surface this repo has
    already been fooled by once.
    """
    obs = _lean_market_observations(led)
    if obs.empty:
        return {}

    # Retained because it is the evidence for the rule having NO upper
    # favourite cutoff: if the close already charged more as the model's
    # separation grew, a cutoff would have something to bite on. Reported as a
    # single slope, never used to bucket a record.
    x = obs["delta"].to_numpy(dtype=float)
    y = obs["market_edge"].to_numpy(dtype=float)
    # Both statistics need a spread on BOTH axes: a constant column makes
    # corrcoef divide by its own zero sd and hand back a silent nan.
    spread = (len(obs) > 1 and float(np.std(x)) > 0 and float(np.std(y)) > 0)
    if spread:
        slope, intercept = np.polyfit(x, y, 1)
        # And its spread. A slope published bare is a number the reader cannot
        # judge, which is the rule every other statistic on this page already
        # follows -- see `_excess_se`. OLS: se = sqrt(RSS / dof / Sxx), which
        # needs two residual degrees of freedom to exist at all.
        dof = len(x) - 2
        resid = y - (slope * x + intercept)
        sxx = float(((x - x.mean()) ** 2).sum())
        slope_se = (math.sqrt(float((resid ** 2).sum()) / dof / sxx)
                    if dof > 0 else np.nan)
    else:
        slope = intercept = slope_se = np.nan

    follow = obs["hybrid_follow"]
    hyb = dict(won="hybrid_won", p="hybrid_p", resid="hybrid_resid",
               profit="hybrid_profit")
    chalk = dict(won="chalk_won", p="chalk_p", resid="chalk_resid",
                 profit="chalk_profit")
    all_rows = obs["won"].notna()
    branch_rows = [
        (f"XWOBA SIDE · XWOBA-side p ≥ {100 * HYBRID_THRESHOLD:.0f}%",
         _lean_market_agg(obs, follow, **hyb)),
        # Plain "<": the table escapes every label through `_esc`, so a
        # pre-escaped entity here would render as literal "&lt;".
        (f"MARKET FAVORITE · XWOBA-side p < {100 * HYBRID_THRESHOLD:.0f}%",
         _lean_market_agg(obs, ~follow, **hyb)),
        ("Hybrid, both branches", _lean_market_agg(obs, all_rows, **hyb)),
    ]
    home = dict(won="home_won", p="home_p", resid="home_resid",
                profit="home_profit")
    control_rows = [
        ("Model lean, unmodified", _lean_market_agg(obs, all_rows)),
        ("Always chalk", _lean_market_agg(obs, all_rows, **chalk)),
        ("Always home", _lean_market_agg(obs, all_rows, **home)),
        # The row that makes the fade branch legible: it must match the FADE
        # line above EXACTLY. If it ever does not, the two were computed over
        # different rows and that is a bug rather than a discovery.
        ("Always chalk · market-favorite rows only",
         _lean_market_agg(obs, ~follow, **chalk)),
    ]
    return {
        "obs": obs,
        "n": int(len(obs)),
        "n_fade": int((~follow).sum()),
        "threshold": HYBRID_THRESHOLD,
        "slope": float(slope),
        "slope_se": float(slope_se),
        "intercept": float(intercept),
        "branch_rows": branch_rows,
        "control_rows": control_rows,
    }


def _lean_market_value_cell(parts, kind):
    """One cell in the exploratory market-value tables."""
    if not parts:
        return "<span class='tm-n'>—</span>"
    if kind == "record":
        return (f"<span class='tm-record'>{parts['w']}-{parts['l']}</span>"
                f"<span class='tm-n'>n={parts['n']}</span>")
    if kind == "market":
        se = (f" ± {100 * parts['excess_se']:.1f}"
              if np.isfinite(parts["excess_se"]) else "")
        tone = "cool" if parts["excess"] > 0 else ("warm" if parts["excess"] < 0 else "")
        return (f"<span class='tm-record{(' ' + tone) if tone else ''}'>"
                f"{100 * parts['actual']:.1f}%</span>"
                f"<span class='tm-accuracy'>vs {100 * parts['implied']:.1f}% "
                f"({100 * parts['excess']:+.1f}{se})</span>")
    tone = "cool" if parts["roi"] > 0 else ("warm" if parts["roi"] < 0 else "")
    return (f"<span class='tm-record{(' ' + tone) if tone else ''}'>"
            f"{100 * parts['roi']:+.1f}%</span>"
            f"<span class='tm-accuracy'>{parts['units']:+.2f}u flat</span>")


def _lean_market_value_table(rows, first_head="Situation"):
    """Responsive four-column table shared by both value-diagnostic views."""
    body = []
    for label, parts in rows:
        cells = [
            ("c-game", first_head, f"<span class='team-code'>{_esc(label)}</span>"),
            ("c-lean", "Record", _lean_market_value_cell(parts, "record")),
            ("c-ml", "vs implied", _lean_market_value_cell(parts, "market")),
            ("c-final", "Flat ROI", _lean_market_value_cell(parts, "roi")),
        ]
        tds = "".join(f"<td class='{cls}' data-l='{lab}'>{value}</td>"
                      for cls, lab, value in cells)
        body.append(f"<tr class='gr-row'>{tds}</tr>")
    heads = (first_head, "Record", "Actual vs implied", "Flat close ROI")
    return ("<div class='gr-tablewrap'><table class='gr team-gr'><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in heads)
            + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def _render_lean_market_value_panel(led):
    """Hybrid-rule validation panel for market-calibration.html."""
    a = _lean_market_value_analysis(led)
    if not a:
        return (f"<div class='gr-head'><h2 class='gr-h1'>{PUBLIC_MODEL_NAME}</h2>"
                "<div class='gr-lead'>Branch diagnostics appear once graded "
                "leans carry a closing moneyline.</div></div>")

    slope = a["slope"]
    slope_pp = slope * 0.010 * 100 if np.isfinite(slope) else np.nan
    slope_txt = f"{slope_pp:+.2f} pp" if np.isfinite(slope_pp) else "—"
    se_pp = a["slope_se"] * 0.010 * 100 if np.isfinite(a["slope_se"]) else np.nan
    slope_sub = (f"± {se_pp:.2f} · leaned-team p per +.010 Δ"
                 if np.isfinite(se_pp) else "leaned-team p per +.010 Δ")

    summary = (
        f"<div class='gr-head'><h2 class='gr-h1'>{PUBLIC_MODEL_NAME}</h2>"
        "<div class='gr-lead'><b>XWOBA SIDE</b> at "
        f"{100 * a['threshold']:.0f}% or higher; <b>MARKET FAVORITE</b> "
        "below.</div></div>"
        "<div class='gr-summary'>"
        f"<div class='gr-stat'><div class='l'>Priced decisions</div>"
        f"<div class='v'>{a['n']}</div>"
        "<div class='s'>settled full-game leans</div></div>"
        f"<div class='gr-stat'><div class='l'>Threshold</div>"
        f"<div class='v'>{100 * a['threshold']:.0f}%</div>"
        "<div class='s'>XWOBA side at or above · market favorite below</div></div>"
        f"<div class='gr-stat'><div class='l'>Selections changed</div>"
        f"<div class='v'>{a['n_fade']}</div>"
        f"<div class='s'>{100 * a['n_fade'] / a['n']:.1f}% became market-favorite selections</div></div>"
        f"<div class='gr-stat'><div class='l'>Market response</div>"
        f"<div class='v'>{slope_txt}</div>"
        f"<div class='s'>{slope_sub}</div></div>"
        "</div>"
    )
    note = ("<div class='gr-note'>Results use each selection's devigged "
            "closing price. The 45% threshold was selected from these rows; "
            "always chalk is shown on the same games.</div>")
    branch_head = (
        "<div class='gr-head'><h2 class='gr-h1'>By branch</h2>"
        "<div class='gr-lead'>What the published rule selected, and how those "
        "selections settled against their own closing prices.</div></div>"
    )
    control_head = (
        "<div class='gr-head' style='margin-top:18px'><h2 class='gr-h1'>"
        "Controls</h2><div class='gr-lead'>The same rows scored three other "
        "ways, and then the last of them restricted to the games the "
        "MARKET FAVORITE branch acts on. A record is only a result relative "
        "to these.</div></div>"
    )
    # The card's per-game panel already says this in words; the page that
    # prints the two lines four rows apart did not. Adjacency is not enough
    # when two published records are identical to the decimal -- without a
    # sentence a reader sees duplicated data or a bug, rather than the point.
    control_note = (
        "<div class='gr-note'>The last row must equal the "
        "<b>MARKET FAVORITE</b> branch above, exactly. Backing the other side "
        f"of a lean priced under {100 * a['threshold']:.0f}% always lands on "
        "the favourite, so on those games that branch <i>is</i> the chalk bet "
        "and carries no model content. If the two lines ever differ they were "
        "scored over different rows, which is a bug and not a discovery.</div>"
    )
    return (summary + note + branch_head
            + _lean_market_value_table(a["branch_rows"], first_head="Branch")
            + control_head
            + _lean_market_value_table(a["control_rows"], first_head="Control")
            + control_note)


def _record_grades(led):
    """Graded rows whose tags share the current prediction methodology.

    The row set for the weight-fit / audit regime that must not mix
    prediction-math changes, AND -- since the headline was scoped -- for the
    two public surfaces that quote a record: the leans strip and the grading
    ledger's header. `RECORD_TAGS` is the authority on which families are
    comparable, so those surfaces and data/ledger_report.txt now answer the
    same question over the same rows.

    NOT the row set for anything that needs volume rather than comparability.
    The per-club table and the market-verdict context still pool every family
    through _display_grades, because a current-family slice gives most clubs
    one or two games. Those surfaces say so on their face; see
    _record_scope_note."""
    return led[(led["status"] == "graded") & (led["model_tag"].isin(RECORD_TAGS))]


def _display_grades(led):
    """All graded rows, every model version joined.

    The pooled view. Used where a statistic needs sample size more than it
    needs methodological comparability -- per-club accuracy, the market
    verdict's context bucket -- never for a headline record. A record answers
    "how good is this model", and pooling twelve prediction families into it
    answers a different question; that is what RECORD_TAGS exists to separate
    and what _record_grades now serves."""
    return led[led["status"] == "graded"]


def _record_scope_note(led, g):
    """(scope_txt, n_family, n_all) for a record scored on `g`.

    Every surface that narrows to the current family sits above or beside
    something that does not -- the grades page prints the whole ledger table
    under its header, and the strip links to a pooled per-team page. Saying
    "19-11" over a 568-row table without saying which rows is the
    claims-the-data-can't-support anti-pattern in its purest form, so the
    scope travels with the number rather than being left to the reader.

    Returns an empty scope string when the family IS every graded row, so the
    note disappears by itself rather than needing a caller-side branch."""
    n_all = int((led["status"] == "graded").sum())
    n_fam = len(g)
    if n_fam == n_all:
        return "", n_fam, n_all
    return (f"{n_fam} of {n_all} graded rows are {MODEL_TAG}", n_fam, n_all)


def _baseline_controls(g):
    """Trivial-strategy records over the rows passed in, as (label, W, L).

    A .570 headline means nothing without something to beat. Two controls,
    both scored on the identical rows the model's record is scored on --
    which is why the caller passes the *decided* rows, not every graded one.
    A control needs no lean to score a game, so handed the full graded frame
    it would happily score the ones v5 abstained on and print a baseline over
    N+1 games beside a model line over N:

      home    -- always take the home side. The null hypothesis for any
                 side-picking model; also the shape of the F5 control already
                 in data/ledger_report.txt, so the page and the report agree.
      market  -- always take the devigged closing favourite. The strategy the
                 model has to beat to be worth anything, since the closing
                 line is free. Only defined on rows that carry a close, so it
                 reports its own n and the caller states it.

    Display-only, and best-effort: a control whose inputs are missing is
    omitted rather than approximated. Returns [] when nothing is scorable."""
    def col(name):
        """Numeric column, or an all-NaN one when the ledger predates it."""
        return pd.to_numeric(g[name], errors="coerce") if name in g.columns \
            else pd.Series(np.nan, index=g.index)

    fa, fh = col("full_away"), col("full_home")
    played = fa.notna() & fh.notna() & (fa != fh)
    if not played.any():
        return []
    home_won = fh > fa
    out = [("home", int((home_won & played).sum()),
            int((~home_won & played).sum()))]
    p_home = col("close_p_home")
    m = played & p_home.notna()
    if m.any():
        hit = ((p_home >= 0.5) == home_won) & m
        out.append(("market", int(hit.sum()), int((m & ~hit).sum())))
    return out


# Emergency kill switch only: raise it to suppress branch records entirely.
# It is 1 rather than a sample floor because with two branches a thin one is
# self-describing -- the record prints its own n and its own standard error,
# and `_branch_read` says "within noise" whenever the spread cannot support
# more. Suppressing a number invites someone to recompute it without the
# caveat, which is the rule this repo settled when it deleted `N_FIT_MIN`.
BRANCH_RECORD_MIN = 1


def hybrid_branch_records():
    """Current-family records for each hybrid branch, plus its chalk control.

    Keys: ``("branch", "FOLLOW"|"FADE")`` for the rule's own record,
    ``("chalk", ...)`` for always-chalk on the IDENTICAL rows, and ``"pooled"``
    for the whole family. Scored on `_record_grades`, because pooling older
    prediction math would answer a different question.

    THESE ARE DISCOVERY ROWS. The threshold was chosen on this sample, so a
    branch's excess here is not evidence the rule works -- `hybrid_test.py`
    holds the forward registration and is the only thing that can answer that.
    Every surface rendering these says so; that is not decoration.

    The chalk control is keyed per branch on purpose. Pooled it would be one
    number a reader has to hold against two, and on the FADE branch it is not a
    comparison at all but an identity: fading a sub-threshold lean backs the
    favourite on every row, so `("chalk", "FADE")` and `("branch", "FADE")`
    must come out equal. If they ever differ, the row sets have drifted apart
    and that is a bug, not a finding.
    """
    led = load_ledger_df()
    if led is None:
        return {}
    obs = _lean_market_observations(led)
    if obs.empty:
        return {}
    out = {"n": int(len(obs)), "threshold": HYBRID_THRESHOLD}
    # The pooled reference the per-game panel prints beside its branch. Same
    # rows, same aggregate, several times the precision.
    pooled = _lean_market_agg(obs, obs["won"].notna())
    if pooled:
        out["pooled"] = pooled
    for action, mask in (("FOLLOW", obs["hybrid_follow"]),
                         ("FADE", ~obs["hybrid_follow"])):
        parts = _lean_market_agg(obs, mask, won="hybrid_won",
                                 p="hybrid_p", resid="hybrid_resid",
                                 profit="hybrid_profit")
        if parts and parts["n"] >= BRANCH_RECORD_MIN:
            out[("branch", action)] = parts
        ctl = _lean_market_agg(obs, mask, won="chalk_won", p="chalk_p",
                               resid="chalk_resid", profit="chalk_profit")
        if ctl and ctl["n"] >= BRANCH_RECORD_MIN:
            out[("chalk", action)] = ctl
    return out


# --- lean strength label (ranks a lean's size vs the ledger) ----------------
# Point-1 of the casual redesign: turn the raw |Δxw| (".024", ".123") into a
# word by ranking it against the historical spread of lean magnitudes. Ranks
# only against SCALE_TAGS rows (same |xw_net| units as the current model) --
# never against the full graded pool, which mixes the pre/post-v5 scales.
# Display-only -- the lean math, MODEL_TAG, and ledger are untouched.
#
# Prior cutoffs. No longer a last-resort branch: they are the prior that the
# observed quantiles are shrunk toward, so they always contribute and their
# influence decays smoothly as the pool grows (see lean_strength).
# STALE BY ITS OWN RULE, KNOWINGLY. Provenance: p33/p80 of |xw_net| over the 24
# ledger rows on the *xwOBA v9/v10* scale as of 2026-07-28 (0.0151 / 0.0325),
# rounded. All 24 were tagged v9 -- an earlier wording here said "v8/v9", but no
# xw+plat_consol_v8 row has ever been graded into the ledger (v8 shipped for one
# morning; its 11 rows were still pending when v9 landed, so the pregame refresh
# rebuilt and re-stamped them). The v8 entries in _SCALE_FAMILIES match zero rows
# and are history, not a live pool. The literal before that was the *pre-v5
# unshrunk* p33/p80 (0.021, 0.060), left in place across the v5, v7 and v8
# shrinkage changes that moved the distribution underneath it; with v9's observed
# maximum |xw_net| of 0.0462 it made "strong" unreachable for every game.
#
# The rule this block has always stated: invalidated by any MODEL_TAG bump that
# changes the delta scale -- i.e. whenever _SCALE_FAMILIES gains a new entry.
# Four bumps have added one since (wOBA v1; v2 sharing v1; then v3 and v4 each
# isolating), so these two numbers are a prior carried over from a retired
# family. Do not re-derive them against whichever family this comment happens to
# name -- name the one SCALE_TAGS resolves to when you read it, since a stale
# family name here is the same defect as a stale literal. That family is v4
# alone as of 2026-08-06, and its pool is far too thin to freeze: n=11, own
# p33/p80 0.0059 / 0.0153, which at that size is noise, and copying it in would
# be the very anti-pattern this comment exists to prevent. Shrinkage plus the
# slate top-up in lean_strength_scale() bound the damage meanwhile -- at n=11
# the shrunk cutoffs are 0.0141 / 0.0303, essentially the prior.
#
# Every row in the family counts toward that n, graded or pending, because
# lean_strength_scale() ranks a pregame magnitude and needs no outcome. A fresh
# family being ungraded is therefore not what blocks a re-derivation; its size
# is.
#
# WHAT TO DO: once the current SCALE_TAGS pool passes ~60 rows, recompute
# p33/p80 from that family alone and replace these literals, then update this
# provenance. Pool growth is otherwise NOT an invalidation -- this is a prior,
# and re-deriving it from the same family it is shrunk against every build would
# make it the data.
LEAN_STRENGTH_FALLBACK = (0.015, 0.032)   # slight < ~p33 <= clear < ~p80 <= strong

# Pseudo-count for shrinking the observed p33/p80 toward LEAN_STRENGTH_FALLBACK.
# This replaced a hard `pool >= 30` gate that switched between frozen and
# dynamic cutoffs in one step: on the real v8/v9/v10 distribution that gate
# moved p80 by up to 0.0096 on the single row that tripped it, flipping a
# game's rendered label with no change to its lean.
#
# Chosen by benchmark over 60 seeded pool-growth paths resampled from the real
# scale pool, scoring continuity (largest single-row cutoff step) against
# fidelity (|cutoff - population quantile| at a realistic pool size). Both
# matter: continuity alone is degenerate, since K -> inf freezes the prior and
# scores perfectly while ignoring the ledger.
#
#     K      max step    |bias| n=300    (current hard gate: 0.00961 / 0.00151)
#     30     0.00502     0.00123
#     100    0.00218     0.00088   <- adopted
#     400    0.00136     0.00185
#
# 100 cuts the worst step 4.4x AND lands nearer the population quantile than
# the gate did, because shrinkage also damps sampling noise in the quantile
# itself. Larger K is smoother but drifts back toward a prior whose p80 is off
# by 0.0037. K=30 -- the old gate reinterpreted as a pseudo-count, the obvious
# first guess -- was measured *worse than the gate* on label stability.
#
# NOT XWOBA_SHRINK_K. Same numeral, unrelated quantity: that one is a
# pseudo-sample of plate appearances for shrinking a rate, this one is a
# pseudo-sample of games for shrinking a quantile. Do not sync them.
LEAN_STRENGTH_PRIOR_N = 100


def lean_strength_scale(slate_deltas=None):
    """Sorted |xw_net| on the current model's delta scale (SCALE_TAGS), for
    ranking a lean's size. Never pools across a scale change (v5's shrinkage
    halved the units) and never falls back to the full ledger.

    Every SCALE_TAGS row counts regardless of grade status: |xw_net| is a
    pregame quantity, so a scale of lean *magnitudes* needs no outcome. The old
    `status == "graded"` filter discarded pending rows and left each new scale
    family ranking against the frozen fallback for days after a MODEL_TAG bump,
    which is exactly how the pre-v5 cutoffs survived three scale changes.
    `slate_deltas` (tonight's own |Δxw|, same units by construction) tops up the
    pool so a fresh family is ranked on its own units from its first build.

    Returns every scale-compatible row it finds, at any size -- a thin pool is
    handled by shrinking its quantiles toward the prior in `lean_strength`, not
    by discarding it. Returns None only when there is no pool at all."""
    parts = []
    led = load_ledger_df()
    if led is not None and "xw_net" in led.columns:
        scaled = (led[led["model_tag"].isin(SCALE_TAGS)]
                  if "model_tag" in led.columns else led)
        parts.append(pd.to_numeric(scaled["xw_net"], errors="coerce").abs().dropna().to_numpy())
    if slate_deltas is not None:
        parts.append(pd.to_numeric(pd.Series(list(slate_deltas)), errors="coerce")
                     .abs().dropna().to_numpy())
    if not parts:
        return None
    arr = np.sort(np.concatenate(parts))
    return arr if arr.size else None


def _slate_deltas(games):
    """Tonight's |Δxw| per game, computed exactly as cmb_card does. An exact
    zero is an abstention (v7 rule), not a lean of size zero, so it is excluded
    from the magnitude scale just as it is excluded from the cards."""
    out = []
    for g in games or ():
        if g.get("unavailable"):
            continue
        a, h = g.get("away") or {}, g.get("home") or {}
        ae, he = a.get("xw_edge"), h.get("xw_edge")
        if ae is None or he is None:
            continue
        net = ae - he
        if net:
            out.append(abs(net))
    return out


def lean_strength(delta, scale):
    """(label, pctile|None) for a lean magnitude |Δxw|.

    Cutoffs are the pool's own p33/p80 shrunk toward `LEAN_STRENGTH_FALLBACK`
    by pool size, `c = (n·c_obs + K·c_prior)/(n + K)` -- the same
    empirical-Bayes form the model already uses for xwOBA, applied to a
    quantile instead of a rate. A one-row change in the pool moves a cutoff by
    a bounded amount at every size, so no build can relabel a game purely by
    crossing a sample-count threshold. With no pool at all the cutoffs are
    exactly the prior, which is the n=0 limit of the same expression rather
    than a separate branch.
    """
    if delta is None or pd.isna(delta):
        return None, None
    delta = abs(float(delta))
    n = int(getattr(scale, "size", 0)) if scale is not None else 0
    p1, p2 = LEAN_STRENGTH_FALLBACK
    if n:
        pct = 100.0 * float(np.searchsorted(scale, delta, side="right")) / n
        w = n / (n + LEAN_STRENGTH_PRIOR_N)
        c1 = w * float(np.quantile(scale, 1 / 3)) + (1.0 - w) * p1
        c2 = w * float(np.quantile(scale, 0.80)) + (1.0 - w) * p2
    else:
        pct, c1, c2 = None, p1, p2
    lab = "slight" if delta < c1 else ("clear" if delta < c2 else "strong")
    return lab, pct


def records_strip_html():
    led = load_ledger_df()
    if led is None:
        return ""
    # The current record family, not every graded row. A headline record is a
    # claim about THIS model, and RECORD_TAGS is the repo's own answer to
    # which families may share a win-loss line -- the same rule
    # data/ledger_report.txt scores, so the public number and the internal one
    # stop being able to disagree.
    g = _record_grades(led)
    scope, _, n_all = _record_scope_note(led, g)
    if g.empty:
        # Deliberately NOT a fallback to the pooled record. A bump resets this
        # to zero until the family's first row grades (v11 shipped and was
        # superseded without ever producing one), and quietly showing an
        # older family's record under the current model's name is the exact
        # substitution that published "wOBA full 217-164" over 381 xwOBA
        # games. Say there is nothing yet, and say where the history went.
        inner = ("<span class='muted'>no graded games yet under "
                 f"{_esc(MODEL_TAG)}</span>")
        if n_all:
            inner += (f" <span class='muted'>· {n_all} graded under earlier "
                      "families retained internally</span>")
    else:
        # Label the record with the metric those rows were predicted under,
        # not MODEL_RATE_LABEL. Still read off the rows even now the family is
        # pinned: a tag flips a slate before any row under it grades, so the
        # constant and the rows can still disagree for a day.
        from market_backfill import metric_label
        label = metric_label(g)
        # The strip headlines the SAME statistic the grades page it links to
        # headlines: the published rule's selection, over the same priced rows,
        # from the same aggregate. Two public surfaces showing different
        # records for "the model" is the artifacts-disagreeing anti-pattern,
        # and here the reader would be one click from seeing both.
        obs = _lean_market_observations(led)
        bits = []
        if obs.empty:
            # No price means the rule cannot act; fall back to the model's own
            # lean and SAY which it is, rather than letting an unlabelled
            # record stand in for the rule's.
            bits.append(f"{label} lean {_rec_txt(g['xw_full'])}")
        else:
            priced = obs["won"].notna()
            rule = _lean_market_agg(obs, priced, won="hybrid_won",
                                    p="hybrid_p", resid="hybrid_resid",
                                    profit="hybrid_profit")
            # The metric label stays on the record. The selection is built on
            # a lean predicted under a specific statistic, and dropping the
            # label is how this strip once published "wOBA full 217-164" over
            # 381 xwOBA games -- read off the ROWS, never MODEL_RATE_LABEL.
            bits.append(f"{PUBLIC_MODEL_NAME} ({label}) "
                        f"{rule['w']}-{rule['l']} "
                        f"({rule['actual']:.3f})")
            se = rule["excess_se"]
            if se is not None and np.isfinite(se) and se > 0:
                bits.append(f"vs mkt z {rule['excess'] / se:+.2f} "
                            f"({rule['units']:+.2f}u)")
            ctl = _lean_market_agg(obs, priced, won="chalk_won", p="chalk_p",
                                   resid="chalk_resid", profit="chalk_profit")
            if ctl:
                # The control travels with the headline. A record with no
                # yardstick beside it is what this strip was fixed for once
                # already, when it published .570 with nothing to read it
                # against.
                bits.append("<span class='muted'>always chalk "
                            f"{ctl['w']}-{ctl['l']}</span>")
        if scope:
            bits.append(f"<span class='muted'>{scope}</span>")
        inner = " <span class='muted'>·</span> ".join(bits)
    return ("<div class='gradestrip'><span class='lab'>V12 record</span>"
            f"<span>{inner}</span><span class='grade-links'>"
            "<a href='leaderboard.html'>leaderboard →</a>"
            "<a href='grades.html'>v12 ledger →</a></span></div>")


def _wlt_badge(v):
    if isinstance(v, str) and v in ("W", "L", "T"):
        return f"<span class='wlt {v}'>{v}</span>"
    return "<span class='wlt none'>–</span>"


def _fmt_ml_cell(v):
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return "<span class='muted'>—</span>"
    return f"+{int(v)}" if v > 0 else f"{int(v)}"


def _lean_ml_cell(r, lean_col):
    """Closing ML of the lean's side; '—' when no market data."""
    lean = r.get(lean_col)
    if not isinstance(lean, str) or not lean:
        return "<span class='muted'>—</span>"
    return _fmt_ml_cell(r.get("close_home_ml") if lean == r.get("home") else r.get("close_away_ml"))


def _pick_ml_cell(r, pick):
    """Closing moneyline of the side the rule actually selected."""
    if not isinstance(pick, str) or not pick:
        return "<span class='muted'>—</span>"
    return _fmt_ml_cell(r.get("close_home_ml") if pick == r.get("home")
                        else r.get("close_away_ml"))


def _lean_cell(lean, delta, muted=False):
    if not isinstance(lean, str) or not lean:
        return "<span class='muted'>—</span>"
    d = pd.to_numeric(delta, errors="coerce")
    txt = _esc(lean) + (
        f" <span class='muted'>Δ{delta3(d)}</span>" if pd.notna(d) else ""
    )
    return f"<span class='muted'>{txt}</span>" if muted else txt


def _row_hybrid(r):
    """(action, selection, result) for one ledger row under the published rule.

    Row-level twin of the columns `_lean_market_observations` derives in bulk,
    and the ONLY place the ledger table decides what the rule did. Returns
    ``(None, None, None)`` whenever the rule cannot act, which is three
    distinct cases the caller has to tell apart and label:

      * the row predates the current record family -- see below;
      * no lean was published (a v5 abstention);
      * no two-sided close is attached (every pending row, by no-lookahead).

    An abstention must render as an abstention rather than borrowing the
    model's own result under a selection heading.

    SCOPED TO `RECORD_TAGS`. The rule is registered against the current
    prediction family, and applying it to an older family's rows publishes a
    selection nobody could have made -- under lean math the rule was never
    paired with, and with a grade this function would then invert. The archive
    still shows those rows; it shows them as the leans they were.

    The result on a faded row is the lean's grade INVERTED: the rule backed the
    other side, so a lean that lost is a selection that won. Ties are mapped
    explicitly rather than by subtraction, which is what swallows them.
    """
    if r.get("model_tag") not in RECORD_TAGS:
        return None, None, None
    locked_pick = r.get("hybrid_selection")
    locked_action = r.get("hybrid_action")
    if (r.get("selection_rule_tag") == HYBRID_RULE_TAG
            and isinstance(locked_pick, str) and locked_pick
            and locked_action in ("FOLLOW", "FADE")):
        return locked_action, locked_pick, r.get("hybrid_full")
    lean = r.get("xw_lean")
    if not isinstance(lean, str) or not lean:
        return None, None, None
    ph = pd.to_numeric(r.get("close_p_home"), errors="coerce")
    if pd.isna(ph):
        return None, None, None
    home, away = r.get("home"), r.get("away")
    market_p = float(ph) if lean == home else 1.0 - float(ph)
    action = hybrid_action(market_p)
    pick = hybrid_selection(lean, away, home, market_p)
    if action is None or pick is None:
        return None, None, None
    grade = r.get("xw_full")
    if action == "FADE":
        grade = {"W": "L", "L": "W"}.get(grade, grade)
    return action, pick, grade


def _grades_row(r, show_ml=False):
    status = str(r["status"])
    fa, fh = pd.to_numeric(r["full_away"], errors="coerce"), pd.to_numeric(r["full_home"], errors="coerce")
    if status != "graded":
        final = f"<span class='st {status}'>{_esc(status)}</span>"
    else:
        final = f"{int(fa)}–{int(fh)}" if pd.notna(fa) and pd.notna(fh) else "<span class='muted'>—</span>"
    # Muted marker when the accepted snapshot locked with a fully projected
    # lineup on either side (lineup_status_* audit columns; NaN on legacy rows).
    proj_mark = ""
    if any(str(r.get(c)) == "projected"
           for c in ("lineup_status_away", "lineup_status_home")):
        proj_mark = (" <span class='muted' title='locked with a projected "
                     "lineup'>°</span>")
    game = (f"<span class='gr-game'>{_esc(r['away'])} <span class='at'>@</span> "
            f"{_esc(r['home'])}</span>{proj_mark}"
            f"<span class='sp'>{_esc(r.get('away_sp') or '—')} v "
            f"{_esc(r.get('home_sp') or '—')}</span>")
    # (class, phone label, html). data-l only surfaces in the stacked phone
    # layout, so it stays short; the column heads carry the full names. The
    # slate date is carried by the group header row, not repeated per row.
    # A graded row with no lean is an abstention, not a missing value. v5
    # abstains when a side's starter had no measured rate, and "—" alone would
    # read as legacy data the ledger never captured.
    abstained = (status == "graded" and not isinstance(r["xw_lean"], str)
                 and any(str(r.get(c)) == "starter_unmeasured_no_lean"
                         for c in ("pitching_basis_away", "pitching_basis_home")))
    # The table publishes the RULE's selection, so its result column has to be
    # the rule's too. Where the rule cannot act the row falls back to the
    # model's own lean and SAYS SO in the cell -- never silently, because a
    # lean result standing unlabelled in a selection column is the same
    # substitution that once published a pooled record under a current-family
    # name.
    action, pick, rule_grade = _row_hybrid(r)
    current = r.get("model_tag") in RECORD_TAGS
    if abstained:
        sel_cell = ("<span class='muted' title='no lean published: a starter "
                    "had no measured season line'>no lean</span>")
        res = r["xw_full"]
    elif action is None:
        # Three different reasons the rule did not act, and the cell has to say
        # WHICH -- an unlabelled lean sitting in a selection column is the
        # substitution that once published a pooled record under a
        # current-family label.
        sel_cell = _lean_cell(r["xw_lean"], r["xw_delta"], muted=True)
        if isinstance(r.get("xw_lean"), str) and r["xw_lean"]:
            if not current:
                why, tag = ("the hybrid rule is registered against "
                            f"{MODEL_TAG} and is not applied to earlier "
                            "prediction families; this row shows the lean it "
                            "published at the time", "lean only")
            else:
                why, tag = ("no locked pregame market exists on this row and "
                            "no closing market is attached yet", "awaiting market")
            sel_cell += f"<span class='sp' title='{why}'>{tag}</span>"
        res = r["xw_full"]
    elif action == "FADE":
        # Δ is deliberately NOT shown here. It is the model's separation in
        # favour of the side the rule just declined; printed beside the club
        # the rule selected it would read as the model rating THAT team, which
        # is the opposite of what the number means.
        sel_cell = _lean_cell(pick, None)
        sel_cell += ("<span class='sp fade-mark' title='the model side was "
                     "priced below the threshold, so this selection is the "
                     "market favorite; Δ describes the declined model side, "
                     "not this selection'>MARKET FAVORITE</span>")
        res = rule_grade
    else:
        sel_cell = _lean_cell(pick, r["xw_delta"])
        sel_cell += ("<span class='sp' title='the market gave the XWOBA side "
                     "at least the threshold'>XWOBA SIDE</span>")
        res = rule_grade
    cells = [("c-game", "Game", game),
             ("c-lean", "Selection", sel_cell)]
    if show_ml:
        cells.append(("c-ml", "ML",
                      _lean_ml_cell(r, "xw_lean") if action is None
                      else (_fmt_ml_cell(r.get("hybrid_ml"))
                            if r.get("selection_rule_tag") == HYBRID_RULE_TAG
                            and pd.notna(pd.to_numeric(r.get("hybrid_ml"), errors="coerce"))
                            else _pick_ml_cell(r, pick))))
    cells += [("c-final", "Final", final),
              ("c-res", "Result", _wlt_badge(res))]
    cls = "gr-row void" if status == "void" else "gr-row"
    tds = "".join(f"<td class='{c}' data-l='{lab}'>{v}</td>" for c, lab, v in cells)
    return f"<tr class='{cls}'>{tds}</tr>"


def _grades_day_header(date, day, ncols):
    """Group separator for one slate date: weekday, game count, day record.

    313 rows in one flat tbody read as an undifferentiated scroll; the ledger
    is naturally keyed by slate date, so the date leads each group instead of
    repeating in every row."""
    try:
        weekday = pd.to_datetime(date).strftime("%a")
    except Exception:  # noqa: BLE001
        weekday = ""
    n = len(day)
    lab = f"<span class='d'>{_esc(date)}</span>"
    if weekday:
        lab += f" <span class='n'>{weekday}</span>"
    lab += f" <span class='n'>· {n} game{'' if n == 1 else 's'}</span>"
    # Day record only once something on that date is graded; an all-pending
    # date would otherwise show a meaningless 0-0.
    displayed_grades = pd.Series(
        [_row_hybrid(r)[2] for _, r in day.iterrows()], index=day.index,
        dtype=object)
    decided = displayed_grades.isin(["W", "L", "T"]).sum()
    if decided:
        lab += f"<span class='rec'>{_rec_parts(displayed_grades)[0]}</span>"
    return (f"<tr class='gr-day'><th colspan='{ncols}' scope='colgroup'>"
            f"{lab}</th></tr>")


def _lock_provenance(led):
    """(verified, legacy, late) rows for the pregame-lock claim.

    The page asserted every row locked before first pitch. `lock_status` is
    null on every row that predates the instrumentation, and those rows carry
    no `snapshot_utc`/`scheduled_start_utc` either, so nothing in the ledger
    substantiates the claim for them. Report the split instead of asserting
    the whole.

    Three outcomes, not two. `grade_leans._lock_status` also emits
    `late_snapshot` for a row whose snapshot postdates first pitch — a row
    that is instrumented and failed, which is the opposite of an
    uninstrumented one. Counting the remainder as legacy would assert
    provenance the ledger contradicts, so each is counted from its own
    label."""
    n = len(led)
    if "lock_status" not in led.columns:
        return 0, n, 0
    st = led["lock_status"].astype("string")
    verified = int(st.str.startswith("pregame", na=False).sum())
    legacy = int((st.isna() | st.eq("legacy_unverified")).sum())
    return verified, legacy, n - verified - legacy


def render_grades_html(built_txt):
    back = ("<div class='backlink ledger-nav'>"
            "<a href='index.html'>← today's leans</a>"
            "<a href='leaderboard.html'>leaderboard</a>"
            "<a href='market-calibration.html'>market calibration →</a></div>")
    led = load_ledger_df()
    if led is None:
        body = back + (f"<div class='legend'><div class='lg-title'>{PUBLIC_MODEL_NAME} ledger · "
                       "no graded data yet — the ledger appears after the first CI run."
                       "</div></div>")
        return html_document(body, built_txt, title=f"{PUBLIC_MODEL_NAME} ledger")

    # The public archive is one product with one rule. Older prediction-family
    # leans remain in the CSV and internal report, but are not mixed into the
    # XWOBA Market Hybrid table under a shared Selection heading.
    led = led[led["model_tag"].isin(RECORD_TAGS)].copy()

    # Scoped to the current record family, like the graded count they sit
    # beside. Pooling them there would put "30 graded" next to a void count
    # drawn from twelve families -- three numbers in one tile, two of them
    # answering a different question.
    _fam = led["model_tag"].isin(RECORD_TAGS)
    n_pend = int((_fam & (led["status"] == "pending")).sum())
    n_void = int((_fam & (led["status"] == "void")).sum())
    head = (f"<div class='gr-head'><h1 class='gr-h1'>{PUBLIC_MODEL_NAME} ledger</h1>"
            "<div class='gr-lead'>V12 selections and results. Built "
            f"<span class='stamp'>{built_txt}"
            "</span>.</div></div>")

    # Header stats score the current record family; the TABLE below still
    # lists every row the ledger holds. That split is intentional -- the table
    # is the archive, the header is a claim about this model -- and it is why
    # _record_scope_note is printed rather than left implicit.
    g = _record_grades(led)
    _, _, n_all = _record_scope_note(led, g)
    stats, notes = [], []

    def stat(lab, val, sub=None, tone=""):
        s = f"<div class='s'>{sub}</div>" if sub else ""
        stats.append(f"<div class='gr-stat'><div class='l'>{lab}</div>"
                     f"<div class='v{(' ' + tone) if tone else ''}'>{val}</div>{s}</div>")

    if g.empty:
        # Same reasoning as the strip: no silent fall back to the pooled
        # record. The table below still renders every historical row, so the
        # history is on the page -- it is just not being called this model's.
        summary = (f"<div class='gr-note'>No graded games yet under "
                   f"{_esc(MODEL_TAG)}"
                   + (f"; {n_all} rows graded under earlier families are "
                      "listed below and scored per family in "
                      "data/ledger_report.txt" if n_all else "")
                   + ".</div>")
    else:
        notes = [f"<b>XWOBA SIDE</b> at {100 * HYBRID_THRESHOLD:.0f}% or higher; "
                 "<b>MARKET FAVORITE</b> below"]
        # EVERY TILE BELOW IS SCORED ON ONE ROW SET: current family, decided,
        # settled, and carrying a two-sided close. That is stricter than the
        # decided set this header used to score, and deliberately so -- the
        # published rule needs a price to act, so a record over rows it could
        # not have acted on is not this rule's record. One derivation, one
        # denominator, controls included. The alternative is the discrepancy
        # this repo already shipped once, when the controls were scored on
        # every graded row while the model was scored on the decided ones.
        obs = _lean_market_observations(led)
        decided = g[g["xw_lean"].notna()]
        n_abst = int(g["xw_lean"].isna().sum())
        pend_sub = f"{n_pend} pending" + (f" · {n_void} void" if n_void else "")
        if n_abst:
            pend_sub += f" · {n_abst} abstained"
        stat("Graded", str(len(g)), pend_sub)

        # Metric read off the graded rows, not the running build: MODEL_TAG
        # flips a slate before any row under it grades, so on that morning the
        # constant names a metric no row on this page was predicted under.
        from market_backfill import metric_label
        label = metric_label(g)
        if obs.empty:
            # No price means the rule cannot act. Show the model's own record,
            # say so, and KEEP THE CONTROLS -- a record with no yardstick
            # beside it is the defect `_baseline_controls` was written for, and
            # dropping them on this path would reintroduce it wherever the
            # market backfill has not run.
            b_, p_ = _rec_parts(decided["xw_full"])
            stat(f"{label} lean · full", b_, p_)
            ctl_labels = {"home": "Always home", "market": "Always chalk"}
            for key, w, l in _baseline_controls(decided):
                if not (w + l):
                    continue
                sub = f"{w / (w + l):.3f}"
                if w + l != len(decided):
                    sub += f" · n={w + l}"
                stat(ctl_labels[key], f"{w}-{l}", sub, tone="dim")
        else:
            hyb = dict(won="hybrid_won", p="hybrid_p", resid="hybrid_resid",
                       profit="hybrid_profit")
            priced = obs["won"].notna()
            rule = _lean_market_agg(obs, priced, **hyb)
            lean_only = _lean_market_agg(obs, priced)
            n_fade = int((~obs["hybrid_follow"]).sum())
            stat(f"V12 {label} Hybrid", f"{rule['w']}-{rule['l']}",
                 f"{rule['actual']:.3f} · lean alone "
                 f"{lean_only['w']}-{lean_only['l']} · "
                 f"{n_fade} market-favorite selections")
            # z leads. A raw rate in a price-selected sample is mostly base
            # rate -- the statistic this repo already retired from the per-game
            # panel for exactly that reason.
            se = rule["excess_se"]
            z = (rule["excess"] / se
                 if se is not None and np.isfinite(se) and se > 0 else np.nan)
            stat("vs market", f"z {z:+.2f}" if np.isfinite(z) else "—",
                 f"{100 * rule['excess']:+.1f} pp · {rule['units']:+.2f}u flat",
                 tone="cool" if np.isfinite(z) and z > 0 else "warm")
            # Controls, from the SAME aggregate over the SAME rows as the
            # record above them, so no `n=` reconciliation is needed and the
            # denominators cannot drift apart in the first place.
            for lab, cols in (
                ("Always chalk", dict(won="chalk_won", p="chalk_p",
                                      resid="chalk_resid",
                                      profit="chalk_profit")),
                ("Always home", dict(won="home_won", p="home_p",
                                     resid="home_resid",
                                     profit="home_profit")),
            ):
                ctl = _lean_market_agg(obs, priced, **cols)
                if ctl:
                    stat(lab, f"{ctl['w']}-{ctl['l']}",
                         f"{ctl['actual']:.3f}", tone="dim")
        summary = ("<div class='gr-summary'>" + "".join(stats) + "</div>"
                   + (f"<div class='gr-note'>{'. '.join(notes)}.</div>" if notes else ""))

    show_ml = (("close_home_ml" in led.columns and led["close_home_ml"].notna().any())
               or ("hybrid_ml" in led.columns and led["hybrid_ml"].notna().any()))
    # The public archive is v12-only. Pending and void rows remain visible so
    # every attempted publication in the family is accounted for.
    heads = (["Game", "Selection"]
             + (["ML"] if show_ml else [])
             + ["Final", "Result"])
    led = led.sort_values(["game_date", "game_pk"], ascending=[False, True])
    body = []
    for date, day in led.groupby("game_date", sort=False):
        body.append(_grades_day_header(date, day, len(heads)))
        body += [_grades_row(r, show_ml) for _, r in day.iterrows()]
    table = ("<div class='gr-tablewrap'><table class='gr'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in heads)
             + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    return html_document(back + head + summary + table, built_txt,
                         title=f"{PUBLIC_MODEL_NAME} ledger")


def _leaderboard_table(rows, rank_start=1, invert=False):
    """One board's rows as the site's standard responsive table body."""
    out = []
    for i, r in enumerate(rows.itertuples(index=False), start=rank_start):
        pa = "" if pd.isna(r.pa) else f"{int(round(float(r.pa))):,}"
        out.append(
            "<tr class='gr-row'>"
            f"<td class='c-game'><span class='lb-rank'>{i}</span>"
            f"<span class='lb-name'>{_esc(r.name)}</span></td>"
            f"<td class='c-res lb-val'>{float(r.shrunk):.4f}</td>"
            f"<td class='c-lean' data-l='raw'>{float(r.rate):.4f}</td>"
            f"<td class='c-ml' data-l='{'bf' if invert else 'pa'}'>{pa}</td>"
            "</tr>")
    return "".join(out)


def _leaderboard_board_html(title, sub, rows, invert=False):
    """One board. `title` is plain text and is escaped HERE, exactly once.

    It used to arrive carrying an `&mdash;` and was escaped on the way in as
    well, so every position header published the literal text `C &mdash; top
    15`. One owner for the escaping, and no markup in a value that gets
    escaped: the entity belongs in the template, never in the argument.
    """
    n = 0 if rows is None else len(rows)
    head = ("<tr><th>Player</th>"
            f"<th>Shrunk {_esc(MODEL_RATE_LABEL)}</th>"
            "<th>Raw</th>"
            f"<th>{'BF' if invert else 'PA'}</th></tr>")
    if not n:
        body = ("<tr class='gr-row'><td class='c-game' colspan='4'>"
                "No rows available for this board.</td></tr>")
    else:
        body = _leaderboard_table(rows, invert=invert)
    # The count is the row count, never the cap: a board of nine must not be
    # headed "top 15". Same rule as every other published denominator here.
    return (f"<div class='gr-head lb-head'><h2 class='gr-h1'>{_esc(title)}"
            f"<span class='lb-n'>{n}</span></h2>"
            + (f"<div class='gr-lead'>{sub}</div>" if sub else "")
            + f"<div class='gr-tablewrap'><table class='gr'><thead>{head}</thead>"
            f"<tbody>{body}</tbody></table></div>")


def render_leaderboard_html(built_txt, boards):
    """Season xwOBA boards at the model's own shrinkage.

    Display only. The copy here is deliberately short: the page's job is the
    numbers, and the reasoning behind the shrinkage lives in MATCHUP_SITE.md
    where it can be read once rather than restated above every table. What
    survives the trim is the part a reader cannot infer from the rows -- the
    constant, the fact that there is no playing-time cut, and the SP board's
    own coverage, which is a provenance statement rather than a description.
    """
    nav = ("<div class='backlink ledger-nav'>"
           "<a href='index.html'>&larr; today's selections</a>"
           "<a href='grades.html'>v12 ledger &rarr;</a></div>")
    if not boards:
        body = nav + ("<div class='legend'><div class='lg-title'>Leaderboard "
                      "unavailable &mdash; the Savant season boards did not "
                      "load on this build. The last good page stays live until "
                      "they do.</div></div>")
        return html_document(body, built_txt,
                             title=f"{PUBLIC_MODEL_NAME} leaderboard")
    m = boards["meta"]
    label = _esc(MODEL_RATE_LABEL)
    head = (f"<div class='gr-head'><h1 class='gr-h1'>Season leaderboard</h1>"
            f"<div class='gr-lead'>Season {label} shrunk at "
            f"<b>K&nbsp;=&nbsp;{m['k']:.0f}</b>, the model's own. No "
            "playing-time cut &mdash; a thin sample regresses to the target "
            "instead. Raw rate and sample shown beside every number. Built "
            f"<span class='stamp'>{built_txt}</span>.</div></div>")

    if not m["sp_roles_known"]:
        sp_sub = ("No role data on this build, so no starter could be "
                  "identified.")
    else:
        sp_sub = (f"Lowest {label} allowed. {m['n_sp']} of "
                  f"{m['n_pitchers']} pitchers qualified as starters "
                  f"(start share over {RP_MAX_START_SHARE:.0%}).")
    sections = [_leaderboard_board_html(
        "Starting pitchers", sp_sub, boards["sp"], invert=True)]

    if boards["bat"]:
        sections.append(
            f"<div class='gr-head lb-split'><h2 class='gr-h1'>Batters</h2>"
            f"<div class='gr-lead'>Highest {label}, by primary position. "
            f"{m['n_bat_ranked']} of {m['n_batters']} batters ranked.</div>"
            "</div>")
    for pos, rows in boards["bat"]:
        sections.append(_leaderboard_board_html(pos, "", rows))
    body = nav + head + "".join(sections) + _legend_head(
        f"{PUBLIC_MODEL_NAME} leaderboard", built_txt)
    return html_document(body, built_txt,
                         title=f"{PUBLIC_MODEL_NAME} leaderboard")


def render_market_calibration_html(built_txt):
    """Market-only calibration plus current-family model×price diagnostics."""
    nav = ("<div class='backlink ledger-nav'>"
           "<a href='grades.html'>← v12 ledger</a>"
           "<a href='index.html'>today's selections →</a></div>")
    head = ("<div class='gr-head'><h1 class='gr-h1'>Market calibration</h1>"
            "<div class='gr-lead'>What the devigged DK close implied, against "
            "what actually happened — split by side and by price rung. This "
            "grades the <i>market</i>, not the model. Built "
            f"<span class='stamp'>{built_txt}</span>.</div></div>")
    led = load_ledger_df()
    rows, totals = _market_calibration_rows(led)
    if not rows:
        empty = ("<div class='gr-note'>No graded rows carry a closing line yet — "
                 "this page appears once the market backfill has run.</div>")
        return html_document(nav + head + empty, built_txt,
                             title="MLB market calibration")

    # Two tiles, two questions. There is deliberately no both-sides tile: it
    # is 50.0% against 50.0% implied by construction, and it used to lead this
    # page carrying the smallest error bar on it. And no away tile: the away
    # figure is one minus the home figure, so the strip would be publishing
    # one number twice. Both facts are stated in the note below.
    stats = []
    for key, lab in (("favourite", "Favourites"), ("home", "Home sides")):
        t = totals.get(key)
        if not t:
            continue
        stats.append(
            f"<div class='gr-stat'><div class='l'>{lab}</div>"
            f"<div class='v'>{100 * t['actual']:.1f}%</div>"
            f"<div class='s'>vs {100 * t['implied']:.1f}% implied "
            f"({t['diff'] * 100:+.1f} ± {100 * t['se']:.1f}) · n={t['n']}</div></div>")

    # The honest framing has to be on the page, not just in a commit message:
    # every rung here is thin, and a calibration table invites reading a bias
    # into what is sampling noise. State the arithmetic that decides it.
    note = (
        "<div class='gr-note'><b>Implied</b> is the DK closing moneyline "
        "devigged to a fair probability, so it already excludes the hold; the "
        "gap between it and the raw price is what you pay to bet. Each graded "
        "game contributes two observations — the home side at its close and the "
        "away side at its close — because a price is what is being graded, not "
        "a game. <b>±</b> is one standard error on the realised rate. "
        "There is no both-sides total, because there is none worth reading: "
        "the two devigged sides of a game sum to 1 and exactly one of them "
        "wins, so pooling every observation returns 50.0% against 50.0% "
        "implied whatever the season did. <b>Favourites</b> asks the same "
        "question once per game rather than twice — the side priced over "
        "even, at its own close — and it is where a favourite-longshot bias "
        "would show; pick'em games have no favourite and are not in it. The "
        "away figure is one minus the home figure, so the table carries it "
        "per rung and the strip does not repeat it. "
        "A rung whose gap is smaller than about twice its ± is indistinguishable "
        "from a fairly priced market; at these sample sizes most of them are, and "
        "resolving a real favourite-longshot bias would take thousands of games "
        "rather than hundreds.</div>")

    body = []
    for r in rows:
        cells = [
            ("c-game", "Price", f"<span class='team-code'>{_esc(r['rung'])}</span>"),
            ("c-lean", "Home", _calib_cell(r["home"])),
            ("c-ml", "Away", _calib_cell(r["away"])),
            ("c-final", "All", _calib_cell(r["all"])),
        ]
        tds = "".join(f"<td class='{cls}' data-l='{lab}'>{value}</td>"
                      for cls, lab, value in cells)
        body.append(f"<tr class='gr-row'>{tds}</tr>")
    heads = ("Price rung", "Home side", "Away side", "Both")
    table = ("<div class='gr-tablewrap'><table class='gr team-gr'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in heads)
             + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    summary = "<div class='gr-summary'>" + "".join(stats) + "</div>" + note
    value_panel = _render_lean_market_value_panel(led)
    return html_document(nav + head + summary + table + value_panel, built_txt,
                         title="MLB market calibration")


def render_combined_html(xw_df, pl_df, pitcher_rows_df, built_txt,
                         opp_hitters_df=None, detail_df=None, lg_ops=None,
                         slate_df=None, lineup_df=None,
                         league_baseline=None, odds=None, streaks=None):
    games = _df_to_combined_games(xw_df, pl_df, pitcher_rows_df,
                                  opp_hitters_df=opp_hitters_df, detail_df=detail_df,
                                  lg_ops=lg_ops, slate_df=slate_df, lineup_df=lineup_df,
                                  league_baseline=league_baseline, odds=odds, streaks=streaks)
    # Cards lead. The record strip and the model/build stamp sit below them.
    footer = (records_strip_html()
              + _legend_head(f"{PUBLIC_MODEL_NAME} — Statcast {MODEL_RATE_LABEL}", built_txt))
    if not games:
        inner = "<div class='legend'><div class='lg-title'>No paired probables yet — " \
                "probables/lineups not posted. Check back closer to first pitch.</div></div>" + footer
        return html_document(inner, built_txt, extra_js=score_refresh_js())
    strength_scale = lean_strength_scale(_slate_deltas(games))
    logo_assets, logo_css = _logo_assets(games)
    # Per-game history is current-family and keyed on the published rule's own
    # branch, with the always-chalk control on the identical rows. Do not
    # reintroduce a fallback that pools the two branches: FOLLOW and FADE are
    # different bets at different prices, and on FADE the control is an
    # identity rather than a comparison.
    ctx = {**(hybrid_branch_records() or {}),
           "logo_ids": set(logo_assets)}
    body = logo_css + build_combined(games, strength_scale, ctx) + footer
    return html_document(body, built_txt, extra_js=score_refresh_js())


def empty_slate_html(built_txt):
    body = (
        "<div class='legend'>"
        f"<div class='lg-title'>{PUBLIC_MODEL_NAME} · {SLATE_DATE} · built {built_txt}</div>"
        "<div class='lg-keys'><span class='k'>No MLB games scheduled for this date.</span></div>"
        "</div>") + records_strip_html()
    return html_document(body, built_txt)


def _lineup_status_columns(lineup_df):
    """Per-game lineup resolution (posted / partial_filled / projected +
    posted counts) keyed by game_pk, for stamping onto the lean dumps so the
    ledger can audit lineup freshness at lock. lineup_resolution_audit.csv is
    overwritten each build; the dump is what persists. Instrumentation only —
    no effect on leans or grading."""
    if lineup_df is None or lineup_df.empty:
        return {}
    idx = lineup_df.set_index("game_pk")
    missing = pd.Series(np.nan, index=idx.index)
    return {
        "lineup_status_away": idx["away_lineup_status"],
        "lineup_status_home": idx["home_lineup_status"],
        "lineup_posted_away": idx["away_posted_count"],
        "lineup_posted_home": idx["home_posted_count"],
        "lineup_savant_backfill_away":
            idx.get("away_savant_backfill_count", missing),
        "lineup_savant_backfill_home":
            idx.get("home_savant_backfill_count", missing),
    }


# ============================================================
# MAIN
# ============================================================
def _built_text_now():
    now_pt = datetime.now(PT)
    return (now_pt.strftime("%I:%M %p").lstrip("0")
            + now_pt.strftime(" · %Y-%m-%d"))


def write_grades_page(built_txt=None):
    """Render both public ledger views without fetching a slate."""
    built_txt = built_txt or _built_text_now()
    os.makedirs(OUT_DIR, exist_ok=True)
    grades_path = os.path.join(OUT_DIR, "grades.html")
    with open(grades_path, "w") as f:
        f.write(render_grades_html(built_txt))
    calib_path = os.path.join(OUT_DIR, "market-calibration.html")
    with open(calib_path, "w") as f:
        f.write(render_market_calibration_html(built_txt))
    log(f"Wrote {grades_path} and {calib_path}")
    return grades_path


def write_leaderboard_page(built_txt, boards):
    """Write leaderboard.html, or leave the last good one in place.

    Returns the path written, or None. Never raises: this page is a display
    surface hanging off a build whose real product is the slate, so it follows
    the same rule as the odds fetch -- a failure costs a log line.
    """
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "leaderboard.html")
        with open(path, "w") as f:
            f.write(render_leaderboard_html(built_txt, boards))
        log(f"Wrote {path}")
        return path
    except Exception as e:  # noqa: BLE001
        log(f"Leaderboard page not written ({e!r}); last good copy stays live.")
        return None


def main():
    built_txt = _built_text_now()
    if "--grades-only" in sys.argv:
        write_grades_page(built_txt)
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")

    # Render once here for local/partial paths. CI renders it again after the
    # post-build grading pass so the deployed ledger includes newly ingested
    # matchups from this same build.
    write_grades_page(built_txt)

    # Top-level guard: a failed core fetch must NOT write index.html, so CI
    # skips the deploy and the last good page stays live.
    data = fetch_all(SLATE_DATE)

    if data.get("empty"):
        log("No games for this date -> writing friendly empty-slate page.")
        with open(out_path, "w") as f:
            f.write(empty_slate_html(built_txt))
        log(f"Wrote {out_path}")
        return 0

    # Written here rather than at the end so every later early return still
    # publishes it -- a probables-not-posted slate has no cards and the same
    # season boards. Failure-isolated: the leans page is what this build exists
    # to produce, and a leaderboard render must never be what stops it.
    write_leaderboard_page(built_txt, data.get("leaderboard"))

    log(f"Building {MODEL_RATE_LABEL} matchup ...")
    matchup_df, pitcher_rows_df, opp_hitters_df = build_xwoba_matchup(
        data["pitchers_df"], data["league_baseline"])
    matchup_df = apply_pitching_plans(
        matchup_df, data.get("pitching_plans"), data["league_baseline"]
    )

    if matchup_df is None or matchup_df.empty:
        # Probables/lineups not posted yet. Render a valid "check back" page
        # rather than failing -- this is a real pre-slate state, not a fetch error.
        log(f"No {MODEL_RATE_LABEL} matchups built (probables not posted) -> check-back page.")
        html = render_combined_html(pd.DataFrame(columns=["game_pk", "side"]),
                                    pd.DataFrame(), pitcher_rows_df, built_txt)
        with open(out_path, "w") as f:
            f.write(html)
        log(f"Wrote {out_path}")
        return 0

    if USE_PITCH_MIX_SHADOW:
        # Shadow only: extra columns for forward testing, no effect on any lean.
        # Failure-isolated like the odds fetch -- an arsenal outage must never
        # take down a build that has already fetched everything the lean needs.
        log("Attaching pitch-mix shadow columns (inert; forward test only) ...")
        try:
            batter_arsenal, pitcher_arsenal = pitch_arsenal.fetch_arsenals(
                cached_csv, SEASON)
            matchup_df = attach_pitch_mix_shadow(
                matchup_df, pitcher_rows_df, opp_hitters_df,
                data["league_baseline"], batter_arsenal, pitcher_arsenal)
            cov = pd.to_numeric(matchup_df.get("mix_coverage"), errors="coerce")
            log(f"  arsenal cells: {len(batter_arsenal)} batter rows, "
                f"{len(pitcher_arsenal)} pitcher rows; median side coverage "
                f"{cov.median():.2f}" if cov is not None and len(cov)
                else "  arsenal coverage unavailable")
        except Exception as e:  # noqa: BLE001
            log(f"pitch-mix shadow skipped: {e!r}")

    log("Building platoon-OPS matchup ...")
    try:
        matchup_platoon_df, platoon_detail_df, league_ops_overall = build_platoon_matchup(
            pitcher_rows_df, opp_hitters_df, data["people"],
            data["player_splits_hit"], data["player_splits_pit"],
            league_splits=data.get("player_splits_hit_league"),
            league_people=data.get("people_league_hitters"))
    except Exception as e:  # noqa: BLE001
        # Platoon lens is optional; degrade to xwOBA-only rather than fail.
        log(f"Platoon lens skipped: {e!r}")
        matchup_platoon_df = pd.DataFrame()
        platoon_detail_df, league_ops_overall = pd.DataFrame(), np.nan

    # The public model is the matchup direction plus the current market. Fetch
    # the market BEFORE the dump so the exact selection shown on the page is
    # also the immutable pregame selection the ledger will grade.
    log("Fetching pregame odds (best-effort) ...")
    try:
        odds = fetch_pregame_odds(data["slate_df"])
    except Exception as e:  # noqa: BLE001
        log(f"pregame odds skipped: {e!r}")
        odds = {}

    # Dump an auditable pregame snapshot for the grading ledger. The grader
    # rejects rows captured at/after their scheduled start, so in-progress
    # refreshes cannot alter the published pregame record.
    snapshot_utc = datetime.now(UTC).isoformat()
    lu_cols = _lineup_status_columns(data["lineup_projection_df"])
    for frame in (matchup_df, matchup_platoon_df):
        if frame is not None and not frame.empty:
            frame["model_tag"] = MODEL_TAG
            frame["model_metric"] = MODEL_RATE_LABEL
            frame["snapshot_utc"] = snapshot_utc
            if "game_datetime_utc" in frame.columns:
                frame["scheduled_start_utc"] = frame["game_datetime_utc"]
            for col, series in lu_cols.items():
                frame[col] = frame["game_pk"].map(series)
    attach_hybrid_snapshot(matchup_df, odds, snapshot_utc)
    attach_hybrid_snapshot(matchup_platoon_df, odds, snapshot_utc)
    os.makedirs(DATA_DIR, exist_ok=True)
    # One decision for both dumps, taken from the primary frame: they describe
    # the same games at the same instant, and naming them from separate reads
    # would let a slate land half rebuilt and half live.
    post_hoc = dump_is_post_hoc(matchup_df, snapshot_utc)
    if post_hoc:
        log(f"dump: every game on {SLATE_DATE} has already started — writing "
            f"under {REBUILD_PREFIX}_ so the pregame dump survives")
    matchup_df.to_csv(dump_path("leans", SLATE_DATE, DUMP_SUFFIX, post_hoc),
                      index=False)
    if matchup_platoon_df is not None and not matchup_platoon_df.empty:
        matchup_platoon_df.to_csv(dump_path("leans", SLATE_DATE, "pl", post_hoc),
                                  index=False)

    log("Fetching current streaks (best-effort, display-only) ...")
    try:
        streaks = fetch_current_streaks(data["slate_df"])
    except Exception as e:  # noqa: BLE001
        log(f"current streaks skipped: {e!r}")
        streaks = {}

    log("Rendering index.html ...")
    html = render_combined_html(
        matchup_df, matchup_platoon_df, pitcher_rows_df, built_txt,
        opp_hitters_df=opp_hitters_df, detail_df=platoon_detail_df,
        lg_ops=league_ops_overall, slate_df=data["slate_df"],
        lineup_df=data["lineup_projection_df"],
        league_baseline=data["league_baseline"], odds=odds, streaks=streaks)
    with open(out_path, "w") as f:
        f.write(html)
    log(f"Wrote {out_path} ({len(html):,} bytes, {len(matchup_df)} matchup rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
