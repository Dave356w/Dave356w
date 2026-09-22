# =============================================================================
# MARKET BACKFILL — attach ESPN/DK closing moneylines to the grading ledger
# =============================================================================
# Drop-in for the grading harness (notebook cell or imported module).
#
# WORKFLOW (daily, after grading):
#   df = pd.read_csv(LEDGER_CSV)
#   df = attach_market(df)          # idempotent; only touches settled rows
#   df.to_csv(LEDGER_CSV, index=False)
#   vs_market_summary(df)           # prints the scoreboard; returns dict for grades.html
#
# COLUMNS ADDED:
#   gamePk, espn_id, open_away_ml, open_home_ml, close_away_ml, close_home_ml,
#   close_p_home  (devigged two-way home win prob at close)
#   f5_open_*_ml, f5_close_*_ml, f5_close_p_home  (DK "1st 5 Innings Moneyline"
#   from the same event's propBets child; devigged prob is CONDITIONAL on a
#   decided half -- DK F5 ties push. Post-final `value` validated as the
#   pregame close in docs/f5_market_validation.md: movement identical across
#   lastUpdated stamp classes (p99 5.9 pts both, 0/302 sides > 10 pts) and
#   lookahead gain -0.0005 Brier vs the mapped-FG baseline over 151 games.)
#
# JOIN LOGIC (validated 59/59 on 07-02..07-07 slate):
#   ledger row -> MLB StatsAPI schedule (date + away + home), doubleheaders
#   disambiguated by probable-pitcher surname; join VERIFIED by final score
#   (mismatch = hard skip + log, never a guess). gamePk -> ESPN event by
#   date + teams, DH disambiguated by final score then start-time proximity.
#   ESPN core API /odds list filtered to provider id 100 (DraftKings).
#   NOTE: direct /odds/100 path 404s as of 2026-07; moneyLine is a dict —
#   read .american. 'close' is only trustworthy on settled (state=post) games.
#
# FAILURE MODE: any row that can't be joined or verified keeps NaN market
# columns and is reported in the returned skip log. No silent defaults.
# =============================================================================

import json
import time
import unicodedata
import datetime as dt
import urllib.request

import numpy as np
import pandas as pd

from fetch_headers import FETCH_HEADERS

# ---- COLMAP: adjust to the ledger CSV's actual column names ----------------
COL = dict(
    date="game_date",       # 'YYYY-MM-DD'
    away="away",            # ledger team abbr (ARI-style)
    home="home",
    p_away="away_sp",       # away starter full name (for DH disambiguation)
    p_home="home_sp",
    away_runs="full_away",  # final score; NaN/None = pending
    home_runs="full_home",
    xw_team="xw_lean",      # lean side abbrs
    pl_team="ops_lean",
    pl_reliable="ops_valid",
    f5_away_runs="f5_away",  # F5 line score; ties push in the F5 market
    f5_home_runs="f5_home",
)

THROTTLE_S = 0.15
LEDGER2SA = {"ARI": "AZ"}                       # ledger -> StatsAPI abbr
ESPN2SA = {"CHW": "CWS", "ARI": "AZ", "OAK": "ATH"}  # ESPN -> StatsAPI abbr

MARKET_COLS = ["gamePk", "espn_id", "open_away_ml", "open_home_ml",
               "close_away_ml", "close_home_ml", "close_p_home",
               "f5_open_away_ml", "f5_open_home_ml",
               "f5_close_away_ml", "f5_close_home_ml", "f5_close_p_home"]

# Columns written by reconstruct_v13.py -- a one-shot migration, never by a
# build. They live in this module for the same reason `excess_se` and
# `chalk_is_home` do: grade_leans owns the ledger writer and cannot import
# build_site, and a writer that does not know a column exists DELETES it.
# That is not hypothetical -- the first bot ledger commit after the v13
# reconstruction landed dropped all five, because `load_ledger()` reindexes
# the frame to its own column lists.
#
# Preserved when present, never minted. `reconstruct_v13.append_columns`
# appends them as trailing fields so no existing byte is re-rendered, and
# refuses to run if they already exist; minting them empty here would make
# the migration unrunnable, so grade_leans keeps only the ones the loaded
# ledger already carries.
# There is deliberately no `v13_full_recon`. The reconstructed GRADE is a
# deterministic function of the reconstructed lean and the game's own two
# finals, all write-once, so it is derived at read time -- and deriving it is
# what lets a retained row that is still PENDING publish the moment it
# settles, instead of carrying a NaN grade forever and silently leaving the
# record on the day it graded.
V13_RECON_COLS = ("v13_net_recon", "v13_lean_recon", "v13_delta_recon",
                  "v13_recon_basis")
# The two that hold text, so a reader can force object dtype: an all-NaN
# column reads back from CSV as float64 and pandas >=3 refuses a string
# assignment into one.
V13_RECON_TEXT_COLS = ("v13_lean_recon", "v13_recon_basis")


def recon_grade(lean, home, full_home, full_away):
    """W/L/T for a reconstructed lean against that game's own final score.

    DERIVED, never stored. The grade is a deterministic function of three
    write-once columns -- the reconstructed lean and the two finals -- and
    this repo's standing rule for exactly that shape is to derive it. It also
    makes a PENDING retained row work: `reconstruct_v13` writes its lean today
    and the grade appears the moment the game settles.

    Returns None when the game has no final, which is the pending case and
    not an error.
    """
    if not isinstance(lean, str) or not lean or not isinstance(home, str):
        return None
    fh = pd.to_numeric(full_home, errors="coerce")
    fa = pd.to_numeric(full_away, errors="coerce")
    if pd.isna(fh) or pd.isna(fa):
        return None
    if fh == fa:
        return "T"
    return "W" if (lean == home) == (fh > fa) else "L"


def recon_grades(g):
    """`recon_grade` over a frame, as a Series aligned to it."""
    return pd.Series(
        [recon_grade(l, h, fh, fa) for l, h, fh, fa in zip(
            g["v13_lean_recon"], g["home"], g["full_home"], g["full_away"])],
        index=g.index, dtype=object)


def publish_reconstruction(g, model_tag):
    """Substitute the v13 re-decision into every retained row of `g`.

    THE one derivation of "what this model publishes for these rows", and it
    lives here for the reason `chalk_is_home` and `excess_se` do: build_site
    renders it and grade_leans has to score the same thing, and grade_leans
    cannot import build_site. Spelled twice, it drifted -- and did.

    **The drift this was extracted to fix, measured before the fix.** The
    per-game card banded on the RECONSTRUCTED delta while
    `grade_leans._selection_price_matrix_lines`, whose own docstring calls
    itself "the grid the game card shows one cell of", banded on the ledger's
    raw `xw_net`. Over 452 rows the two deltas differed on 444, by up to
    0.0215 -- wider than a whole band -- the published LEAN differed on 39,
    and **24 of the 26 cells disagreed**. The records disagreed too: 282-170
    against 273-171.

    Rows: a row built under `model_tag` passes through untouched; a retained
    row is re-decided; a retained row with no usable reconstruction is
    DROPPED, because carrying one over on its own lean would publish an
    earlier model's result under this model's name.

    **AN ABSTAINED ROW IS NEVER RE-DECIDED.** A retained row whose own build
    published no lean passes through as it is -- kept, not substituted and
    not dropped -- because the abstention is a rule THIS model still runs.
    v5 declines a game when a side's starter has no measured season line,
    v11 kept it, and v13 changed the starter's RATE without touching that
    gate; a live build facing the same game today publishes nothing. So a
    reconstruction that hands one a lean is publishing a selection this model
    would itself refuse, which is the "a selection nobody could have made"
    defect rather than a re-decision.

    It was live. `reconstruct_v13` computes a net from the paired dumps with
    no abstention check, so all 8 of the ledger's `starter_unmeasured_no_lean`
    rows were being resurrected: the published headline read 282-171 where the
    rows this model would actually decide are 278-167, and the 8 went 4-4.
    Every surface inherited it through this one function -- the record, the
    ROI, the delta x price grid -- and the grades page reported 0 abstentions
    against a ledger holding 8.

    Passing them through rather than dropping them is deliberate: every
    surface already knows how to handle an abstention (`_rec()` skips it,
    the Graded tile counts `xw_lean.isna()`, the observation frame's
    home-or-away test excludes it), and dropping them instead would relabel
    a declined game as one this migration could not rebuild.

    What this is NOT for: a family history line, or a registration. Those
    score the lean each build actually published, which is the difference
    `grade_leans._published_basis_lines` declares on the artifact. Re-aiming
    a registration at this would restart its forward window.
    """
    if g is None or not len(g):
        return g
    cur = g["model_tag"].astype(str).eq(str(model_tag))
    need = ("v13_recon_basis", "v13_lean_recon", "v13_net_recon",
            "home", "full_home", "full_away")
    if any(c not in g.columns for c in need):
        return g[cur].copy()
    # `xw_lean.isna()` is the abstention signal every other surface in this
    # repo uses, so it is the one used here rather than `pitching_basis_*`:
    # a second spelling would miss v7's zero-delta rule, which has never
    # fired and would produce the same NaN if it ever did.
    abstained = g["xw_lean"].isna()
    has = (g["v13_recon_basis"].notna()
           & g["v13_lean_recon"].notna()
           & ~abstained
           & recon_grades(g).isin(["W", "L", "T"]))
    keep = cur | (~cur & (has | abstained))
    out = g[keep].copy()
    if out.empty:
        return out
    rebuilt = (~cur & has)[keep].to_numpy()
    if rebuilt.any():
        sub = out.loc[rebuilt]
        out.loc[rebuilt, "xw_full"] = recon_grades(sub)
        out.loc[rebuilt, "xw_lean"] = sub["v13_lean_recon"]
        net = pd.to_numeric(sub["v13_net_recon"], errors="coerce")
        out.loc[rebuilt, "xw_net"] = net
        out.loc[rebuilt, "xw_delta"] = net.abs()
    return out


# ---------------------------------------------------------------- helpers ---
def _get(url):
    # Was a bare `Mozilla/5.0`, which ESPN's edge began 403ing on 2026-08-04.
    # See fetch_headers.py for the measurement behind the replacement.
    req = urllib.request.Request(url, headers=FETCH_HEADERS)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


def _dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _amer(x):
    if x is None:
        return None
    try:
        return int(str(x).replace("+", ""))
    except ValueError:
        return None


def _imp(ml):
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def _dec(ml):
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / (-ml))


# ------------------------------------------------------------ data pulls ----
def _statsapi_day(date):
    js = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
              f"&hydrate=probablePitcher,team")
    out = []
    for dd in js.get("dates", []):
        for gm in dd.get("games", []):
            out.append(dict(
                gamePk=gm["gamePk"], gameDate=gm["gameDate"],
                away=_dig(gm, "teams", "away", "team", "abbreviation"),
                home=_dig(gm, "teams", "home", "team", "abbreviation"),
                p_away=_dig(gm, "teams", "away", "probablePitcher", "fullName"),
                p_home=_dig(gm, "teams", "home", "probablePitcher", "fullName"),
                away_score=_dig(gm, "teams", "away", "score"),
                home_score=_dig(gm, "teams", "home", "score"),
            ))
    return out


def _espn_day(date):
    ds = date.replace("-", "")
    sb = _get(f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={ds}")
    out = {}
    for ev in sb.get("events", []):
        comp = ev["competitions"][0]
        t = {c["homeAway"]: ESPN2SA.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
             for c in comp["competitors"]}
        sc = {c["homeAway"]: c.get("score") for c in comp["competitors"]}
        out.setdefault((t["away"], t["home"]), []).append(dict(
            eid=ev["id"], start=ev["date"],
            away_sc=sc["away"], home_sc=sc["home"],
            state=_dig(comp, "status", "type", "state"),
            tid={str(c["team"]["id"]): c["homeAway"] for c in comp["competitors"]},
        ))
    return out


def _espn_close(eid):
    odds = _get(f"https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
                f"/events/{eid}/competitions/{eid}/odds")
    dk = next((i for i in odds.get("items", [])
               if str(_dig(i, "provider", "id")) == "100"), None)
    if dk is None:
        return None
    return dict(
        open_home_ml=_amer(_dig(dk, "homeTeamOdds", "open", "moneyLine", "american")),
        open_away_ml=_amer(_dig(dk, "awayTeamOdds", "open", "moneyLine", "american")),
        close_home_ml=_amer(_dig(dk, "homeTeamOdds", "close", "moneyLine", "american")),
        close_away_ml=_amer(_dig(dk, "awayTeamOdds", "close", "moneyLine", "american")),
    )


def _espn_f5(eid, tid):
    """DK '1st 5 Innings Moneyline' (type.id=136) from the event's propBets.

    Items live in the propBets child (~500-700 player props per game), keyed
    by team $ref id; `tid` maps those ids to home/away. `value` is the last
    posted price and `open` the opener. On settled events the value behaves
    as the pregame close (validation: docs/f5_market_validation.md); stamps
    in `lastUpdated` mark item touches incl. settlement bookkeeping and must
    NOT be read as price-change times.
    """
    base = (f"https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
            f"/events/{eid}/competitions/{eid}/odds/100/propBets")
    try:
        pb = _get(base + "?limit=200")
        items = list(pb.get("items", []))
        for pg in range(2, int(pb.get("pageCount", 1)) + 1):
            time.sleep(THROTTLE_S)
            items += _get(base + f"?limit=200&page={pg}").get("items", [])
    except Exception:  # noqa: BLE001  (404 = props never posted for event)
        return None
    out = dict(f5_open_away_ml=None, f5_open_home_ml=None,
               f5_close_away_ml=None, f5_close_home_ml=None)
    for it in items:
        if str(_dig(it, "type", "id")) != "136":
            continue
        side = tid.get(str((_dig(it, "team", "$ref") or "")
                           .rsplit("/", 1)[-1].split("?")[0]))
        if side not in ("away", "home"):
            continue
        out[f"f5_close_{side}_ml"] = _amer(_dig(it, "odds", "american", "value"))
        out[f"f5_open_{side}_ml"] = _amer(_dig(it, "odds", "american", "open"))
    return out


# ------------------------------------------------------------- main entry ---
def attach_market(df, col=COL, verbose=True):
    """Idempotently attach gamePk + DK closing MLs to settled ledger rows.

    Returns the modified DataFrame. Skipped rows keep NaN and are listed in
    df.attrs['market_skips'] as (index, reason) tuples.
    """
    for c in MARKET_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # dtype coercion must be unconditional: CSV round-trips reload all-NaN
    # ID columns as float64, which rejects string/int assignment.
    df["espn_id"] = df["espn_id"].astype("string")
    df["gamePk"] = df["gamePk"].astype("Int64")
    settled = df[col["away_runs"]].notna() & df[col["home_runs"]].notna()
    todo = df.index[settled & (df["close_home_ml"].isna()
                               | df["f5_close_home_ml"].isna())]
    if len(todo) == 0:
        if verbose:
            print("market backfill: nothing to do")
        df.attrs["market_skips"] = []
        return df

    dates = sorted(df.loc[todo, col["date"]].unique())
    sched = {}
    espn = {}
    for d in dates:
        sched[d] = _statsapi_day(d)
        time.sleep(THROTTLE_S)
        espn[d] = _espn_day(d)
        time.sleep(THROTTLE_S)

    skips = []
    for i in todo:
        r = df.loc[i]
        d = r[col["date"]]
        aw = LEDGER2SA.get(r[col["away"]], r[col["away"]])
        hm = LEDGER2SA.get(r[col["home"]], r[col["home"]])
        a_runs, h_runs = int(r[col["away_runs"]]), int(r[col["home_runs"]])

        # --- StatsAPI join (gamePk), DH disambiguation by pitcher surname
        cands = [g for g in sched[d] if g["away"] == aw and g["home"] == hm]
        if len(cands) > 1:
            sur = _norm(str(r[col["p_away"]])).split()[-1]
            narrowed = [g for g in cands if g["p_away"] and sur in _norm(g["p_away"])]
            if len(narrowed) != 1:
                sur = _norm(str(r[col["p_home"]])).split()[-1]
                narrowed = [g for g in cands if g["p_home"] and sur in _norm(g["p_home"])]
            cands = narrowed
        if len(cands) != 1:
            skips.append((i, f"statsapi join ambiguous ({len(cands)} cands)"))
            continue
        g = cands[0]
        if (g["away_score"], g["home_score"]) != (a_runs, h_runs):
            skips.append((i, f"score mismatch statsapi {g['away_score']}-{g['home_score']}"
                             f" vs ledger {a_runs}-{h_runs}"))
            continue

        # --- ESPN event join, DH disambiguation by score then start time
        evs = espn[d].get((aw, hm), [])
        if len(evs) > 1:
            byscore = [e for e in evs
                       if (str(e["away_sc"]), str(e["home_sc"])) == (str(a_runs), str(h_runs))]
            if len(byscore) == 1:
                evs = byscore
            else:
                gd = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
                evs = sorted(evs, key=lambda e: abs(
                    (dt.datetime.fromisoformat(e["start"].replace("Z", "+00:00")) - gd)
                    .total_seconds()))[:1]
        if len(evs) != 1:
            skips.append((i, "espn event join failed"))
            continue
        if evs[0]["state"] != "post":
            skips.append((i, "espn event not settled; close unreliable"))
            continue

        eid = evs[0]["eid"]
        df.loc[i, "gamePk"] = g["gamePk"]
        df.loc[i, "espn_id"] = eid

        if pd.isna(df.loc[i, "close_home_ml"]):
            ml = _espn_close(eid)
            time.sleep(THROTTLE_S)
            if ml is None or ml["close_home_ml"] is None or ml["close_away_ml"] is None:
                skips.append((i, "no DK close on event"))
                continue
            ph, pa = _imp(ml["close_home_ml"]), _imp(ml["close_away_ml"])
            for k, v in ml.items():
                df.loc[i, k] = v
            df.loc[i, "close_p_home"] = ph / (ph + pa)

        # F5 rides the same verified event; a missing F5 market never blocks
        # or erases the full-game close -- it logs and retries next run.
        if pd.isna(df.loc[i, "f5_close_home_ml"]):
            f5 = _espn_f5(eid, evs[0].get("tid") or {})
            time.sleep(THROTTLE_S)
            if (f5 is None or f5["f5_close_home_ml"] is None
                    or f5["f5_close_away_ml"] is None):
                skips.append((i, "no DK F5 ML on event"))
            else:
                for k, v in f5.items():
                    df.loc[i, k] = v
                ph5, pa5 = _imp(f5["f5_close_home_ml"]), _imp(f5["f5_close_away_ml"])
                df.loc[i, "f5_close_p_home"] = ph5 / (ph5 + pa5)

    df.attrs["market_skips"] = skips
    if verbose:
        done = settled.sum() - len(skips)
        print(f"market backfill: {len(todo) - len(skips)} attached, {len(skips)} skipped")
        for i, why in skips:
            print(f"  SKIP row {i}: {why}")
    return df


# --------------------------------------------------------------- analysis ---
def metric_series(df):
    """Per-row metric label, index-aligned to `df`.

    `model_metric` is authoritative; rows written before that column existed
    fall back to their tag prefix.

    This is the derivation. `metric_label` is its collapse to one name, and a
    caller that needs to GROUP by metric — the per-slate block in
    actuals_backfill — takes the rows themselves rather than re-deriving the
    prefix rule beside this one. Two copies of this rule is exactly how the
    vs-market cell went missing.
    """
    tag = df.get("model_tag")
    if tag is None:
        return pd.Series("xwOBA", index=getattr(df, "index", None), dtype=object)
    tag = pd.Series(tag).astype(str)
    from_tag = pd.Series(np.select(
        [tag.str.startswith("woba+"), tag.str.startswith("split+")],
        ["wOBA", "wOBA/xwOBA"],
        default="xwOBA",
    ), index=tag.index)
    metric = df.get("model_metric")
    if metric is None:
        return from_tag
    metric = pd.Series(metric)
    return metric.where(metric.notna(), from_tag).astype(str)


def metric_label(df, mixed="Model"):
    """Name of the primary rate behind a set of ledger rows, read off the rows.

    The metric is a property of the rows being rendered, never of the running
    build: every public surface here draws pooled history, so a build-time
    constant would relabel 381 xwOBA games "wOBA" the morning the tag flips.
    A frame spanning both metrics gets `mixed`.

    One home for the derivation, deliberately: build_site's display surfaces,
    the ledger report, and the vs-market summary all need the same answer, and
    a lookup keyed on one copy while another copy names the bucket is exactly
    how the vs-market cell went missing.
    """
    uniq = metric_series(df).dropna().unique().tolist()
    if not uniq:
        return "xwOBA"
    return uniq[0] if len(uniq) == 1 else mixed


ODDS_LADDER = (
    (None, -250, "≤ -250"),
    (-249, -175, "-249 to -175"),
    (-174, -130, "-174 to -130"),
    (-129, -100, "-129 to -100"),
    (100, 129, "+100 to +129"),
    (130, 174, "+130 to +174"),
    (175, 249, "+175 to +249"),
    (250, None, "≥ +250"),
)


def ladder_rung(ml):
    """Rung label for an American price, or None if it is not a valid one.

    American odds never fall strictly between -100 and +100, so the rungs
    above tile every representable price. A None return therefore means the
    input was not a real moneyline, and the caller drops it rather than
    guessing.

    One home, for `excess_se`'s reason exactly: the per-game card buckets a
    game's history on these rungs and the ledger report's selection x price
    matrix buckets the same rows the same way. Two copies of the edges would
    let the public surface and the internal artifact disagree about which
    cell a game is in -- the artifacts-disagreeing defect, with the reader
    unable to see which ladder they are reading. It lives here rather than in
    build_site because grade_leans cannot import that module (it refuses a
    non-xwOBA MODEL_TAG at import time) and because this is a statement about
    market prices.
    """
    for lo, hi, label in ODDS_LADDER:
        if (lo is None or ml >= lo) and (hi is None or ml <= hi):
            return label
    return None


def excess_se(probs):
    """SE of (realised rate - mean implied) under correctly priced games.

    One home for a derivation three callers need -- build_site's calibration
    surfaces, value_probe, and the ledger report's magnitude x price grid --
    for the same reason `metric_label` above has one: two copies of a
    statistic drift, and a reader cannot see which one they are looking at.
    It lives here rather than in build_site because grade_leans cannot import
    that module (it refuses a non-xwOBA MODEL_TAG at import time) and because
    this is a statement about market prices.

    Each observation is an independent Bernoulli at its own devigged price, so
    the win count is Poisson-binomial: Var(sum wins) = sum p(1-p), and the SE
    of the mean is sqrt(sum p(1-p))/n.

    It deliberately does NOT estimate the spread from the outcomes. The
    obvious sqrt(p_hat(1-p_hat)/n) does, and therefore returns exactly 0.0 on
    any bucket that went all-W or all-L -- rendering the least certain buckets
    as the most certain, which is the direction that makes noise look like
    signal. The sample sd of the residuals fails the same way for the same
    reason. Here the p_i are fixed by the market rather than estimated from
    the outcomes under test, so this is defined at n=1 and cannot degenerate.
    """
    p = np.asarray(list(probs), dtype=float)
    if not p.size:
        return np.nan
    return float(np.sqrt(float((p * (1.0 - p)).sum())) / p.size)


def ev_null(probs, breakevens):
    """What a breakeven-relative excess is centred on when the market is RIGHT.

    ZERO IS THE WRONG NULL FOR AN EV COLUMN, and that is the whole reason this
    exists. `excess_se` above sizes the spread of (realised rate - a fixed
    market reference); which reference you subtract decides where the null
    sits, not how wide it is:

      * against the DEVIGGED price, a correctly priced book gives E[excess] = 0;
      * against the POSTED breakeven, it gives E[EV] = -(mean hold), because
        breakeven already contains the vig the devigged price has removed.

    So an EV figure read against zero is read against a null the market never
    offered, and the error flatters the book by exactly one hold. It happened:
    the magnitude x market grid stated that rule in its own copy and printed
    the SE beside both columns, and a reader still compared `EV +3.0 pp` to 0
    and called it null when its own null was -2.6 and a simulated
    market-correct null put it at P = 0.051. Prose asserting the rule is not
    the same as printing the number the rule is about -- the caveat-does-not-
    travel-with-the-statistic entry, one column out.

    Takes the two references a caller already holds -- the devigged prices and
    the posted breakevens -- rather than moneylines, so it cannot disagree with
    the `EV` figure printed next to it by re-deriving a breakeven a second way.
    Returns the null in PROBABILITY units (negative, or 0.0 for a vig-free
    book), so a caller renders `100 * ev_null(...)` beside a points figure.

    What it deliberately does NOT return is a second standard error. The two
    columns differ by a constant fixed by the market, so they share one
    sampling spread: `excess / excess_se` is already the z for BOTH, and the
    only thing a reader needs that they did not have is where the EV column's
    centre lies. Handing back a second `se` would invite a second, wrong z.
    """
    p = np.asarray(list(probs), dtype=float)
    be = np.asarray(list(breakevens), dtype=float)
    if not p.size or p.size != be.size or not np.isfinite(be).all():
        return np.nan
    return float(p.mean() - be.mean())


def row_supply_line(g, noun="eligible rows"):
    """The `<noun> since registration: N over S slates` line, plus its LAST.

    ONE home for the line every registration prints, for the reason
    `chalk_is_home` and `excess_se` are here: six modules spelled it six times,
    and the clause added below has to appear in all six or the one that lacks
    it is the one that goes stale unnoticed.

    The clause is the last slate the registration actually scored. A forward
    sample is read as accruing, and nothing on any block said when it last
    did. Three of the five stopped dead on 2026-09-11 -- `hybrid_test` and,
    through its delegated row selector, `abstain_test` and
    `dog_contrast_test` -- while continuing to print gates (5 of 82 declined
    games, 16 of 88 dog leans) that implied rows were still arriving. The cause
    is fixed at the writer, in `grade_leans._mint_v1_archive`; this is the
    instrument that makes the NEXT stall visible instead of leaving it to be
    found by hand, whatever stops the rows. Read it against the ledger's own
    most recent graded slate -- if it trails, the registration is not accruing.

    Deliberately not a staleness VERDICT. This module cannot know the ledger's
    latest slate without taking an argument that every caller would have to
    supply correctly, and a threshold in days would be a constant frozen off
    an operating cadence that already runs 5 to 16 games a slate.
    """
    n = 0 if g is None else len(g)
    if not n or "game_date" not in getattr(g, "columns", ()):
        return f"    {noun} since registration: {n} over 0 slates"
    dates = g["game_date"].astype(str)
    return (f"    {noun} since registration: {n} over {dates.nunique()} "
            f"slates (last scored {dates.max()})")


# The clause a registration prints once its window can take no further rows.
# ONE home, for the same reason `row_supply_line` above is here: two
# registrations need it and they must not word it differently -- a reader
# comparing two blocks has to be able to tell "closed" from "stalled" without
# knowing which module wrote which line.
def window_closed_line(reason):
    """The WINDOW CLOSED clause. `reason` is the module's own one-sentence why.

    `row_supply_line` deliberately refuses to issue a staleness VERDICT, and
    that refusal is right: it cannot know the ledger's latest slate. This is
    the other half of the same problem and it IS answerable, because the
    module that owns a registration knows what its own filter accepts. Without
    it a closed window renders exactly like the 2026-09-12 stall -- a supply
    line trailing the ledger beside a gate that implies rows are still
    arriving -- and the two call for opposite responses: one is a writer bug
    to fix, the other is the answer the registered question got.
    """
    return f"    WINDOW CLOSED -- no further row can enter this test: {reason}"


# Appended under a closed registration's GATE line. A gate is still printed
# when the window shuts, because a reader wants the sizing that was registered;
# what must not survive is the implication that it can still be reached.
GATE_UNREACHABLE = ("      The gate is not reachable from here: the counts "
                    "above are final, not accruing.")


def window_is_closed(led, column, accepted):
    """Can a registration filtering `column` to `accepted` still take a row?

    Returns `(closed, observed)` -- `closed` True when the ledger's most
    recent slate carries no accepted value, False when it carries one, and
    None when the frame cannot answer. `observed` is the distinct non-null
    values found on that slate, so the caller's reason can NAME what is being
    stamped now instead of asserting it.

    DERIVED, never asserted, and that is the whole design. A `CLOSED = True`
    literal would be this repo's constants-frozen-from-data entry in the one
    place it does most damage: a registration that reopened -- the family
    restored, the rule re-shipped -- would go on printing that it could not.
    Keyed on the most recent slate rather than on any row anywhere, because
    the question is what the build stamps NOW; pending rows count, since they
    carry the current build's tags and are the freshest evidence available.

    A slate whose `column` is entirely null answers None rather than True: an
    absent value mid-ingest is not a closed window, and reporting one would
    put the clause on the artifact for a day on a build that is fine.
    """
    if led is None:
        return None, ()
    cols = getattr(led, "columns", ())
    if column not in cols or "game_date" not in cols:
        return None, ()
    dates = led["game_date"].astype(str)
    if not len(dates):
        return None, ()
    live = led.loc[dates == dates.max(), column].dropna()
    if live.empty:
        return None, ()
    observed = tuple(sorted({str(v) for v in live}))
    return not live.isin(list(accepted)).any(), observed


def is_pickem(p_home):
    """True where the devigged home price is exactly .500. Vectorised.

    A pick'em has NO favourite, so every always-chalk record and every
    favourite comparison in this repo rests on a tie-break convention for
    these rows, and a convention that differs between two surfaces makes them
    publish different numbers off the same games. 12 of the graded ledger's
    rows are here as of 2026-09-17; the count is small and the failure it
    caused was not.
    """
    return np.asarray(p_home, dtype=float) == 0.5


def chalk_is_home(p_home):
    """Does the always-chalk control back the HOME side? Vectorised.

    ONE home for the convention, for the reason `excess_se` and `ladder_rung`
    are here: grade_leans cannot import build_site, so a rule both need has to
    live in this module or be spelled twice -- and spelled twice it drifted.
    Before 2026-09-17 there were three spellings of "which side is chalk":

      * `p_home > .5`, dropping pick'ems, on the favourite tile of
        `market-calibration.html`. Correct there and deliberately left alone:
        that tile is a one-observation-per-game POOL, so a game with no
        favourite can simply leave it and nothing is unpaired;
      * `p_home >= .5`, tie to home, in the band block, hybrid_test and the
        delta filter;
      * `market_p >= .50` on the LEANED side's price, in build_site's
        always-chalk control -- which made the control back the model's own
        pick on a pick'em, i.e. defined the control in terms of the thing it
        controls. The two live spellings published 257-179 in
        `ledger_report.txt` and 256-180 on `grades.html` over the same 436
        rows on 2026-09-17.

    The tie goes to HOME. It is arbitrary, and that is the point: it must be
    arbitrary WITH RESPECT TO THE MODEL, because this is the control the
    model's record is read against. Dropping the row instead is the other
    defensible answer and is wrong HERE for a reason specific to a control --
    `a control is only a control if it is scored on the rows the model was
    scored on`, so a chalk record over n-2 beside a model record over n is
    exactly the defect CLAUDE.md records the grades page having had. Surfaces
    that publish a chalk record state the pick'em count instead, so the reader
    knows how many rows rest on the convention rather than on a price.
    """
    return np.asarray(p_home, dtype=float) >= 0.5


def breakeven_prob(mls):
    """Win rate a bet at these American prices must clear to be +EV.

    This is the RAW implied probability, `_imp` vectorised and nothing more --
    one formula, because a second copy of an odds conversion is the "one value,
    three homes" defect in arithmetic. The name exists because the QUANTITY is
    easy to confuse with the devigged price beside it, and the confusion runs
    one way: a cell can beat its devigged `q` and still lose money.

    The gap between the two IS the per-side hold. A bet at ml with decimal
    profit b returns `p(1+b) - 1`, so it breaks even at `p = 1/(1+b)`, which is
    exactly `_imp(ml)`. Every `excess` in this file is measured against the
    DEVIGGED price and is therefore a calibration statistic; realised rate
    minus this is the EV one. On the current ledger the two differ by ~1.7 pp
    on average and by more on favourites, which is enough to flip a band's
    sign -- so they are printed side by side rather than either alone.

    NaN in, NaN out: a row with no usable price gets no breakeven, and the
    caller decides what to do about it rather than being handed a guess.
    """
    out = []
    for ml in np.asarray(mls, dtype=float).ravel():
        out.append(float("nan") if not np.isfinite(ml) or ml == 0
                   else float(_imp(float(ml))))
    return np.asarray(out, dtype=float)


def vs_market_summary(df, col=COL, verbose=True):
    """Vs-market scoreboard for both models. Returns dict for grades.html chips."""
    d = df[df["close_p_home"].notna()].copy()
    d["winner"] = np.where(d[col["home_runs"]] > d[col["away_runs"]],
                           d[col["home"]], d[col["away"]])
    d["fav"] = np.where(d["close_p_home"] >= 0.5, d[col["home"]], d[col["away"]])
    out = {}
    specs = [(metric_label(d), d, col["xw_team"]),
             ("platoon", d[d[col["pl_reliable"]] == True], col["pl_team"])]  # noqa: E712
    for label, rows, key in specs:
        rows = rows[rows[key].notna()]
        n = len(rows)
        if n == 0:
            continue
        p_side = np.where(rows[key] == rows[col["home"]],
                          rows["close_p_home"], 1 - rows["close_p_home"])
        w = (rows[key] == rows["winner"]).sum()
        exp, var = p_side.sum(), (p_side * (1 - p_side)).sum()
        z = (w - exp) / np.sqrt(var)
        ml = np.where(rows[key] == rows[col["home"]],
                      rows["close_home_ml"], rows["close_away_ml"])
        pnl = np.where(rows[key] == rows["winner"],
                       [_dec(m) - 1 for m in ml], -1.0).sum()
        fav_agree = (rows[key] == rows["fav"]).mean()
        fav_w = (rows["fav"] == rows["winner"]).sum()
        out[label] = dict(n=int(n), w=int(w), exp=round(float(exp), 1),
                          z=round(float(z), 2), roi_units=round(float(pnl), 2),
                          fav_agree=round(float(fav_agree), 3),
                          fav_baseline=f"{fav_w}-{n - fav_w}")
        if verbose:
            print(f"{label}: {w}-{n - w} | market-expected {exp:.1f}W -> z {z:+.2f} | "
                  f"ROI {pnl:+.2f}u | agrees w/ fav {fav_agree:.0%} | "
                  f"fav baseline {fav_w}-{n - fav_w}")

    # Platoon lean vs the F5 close -- the market this lean actually targets.
    # DK F5 ties push, so ties are excluded and the devigged close_p is the
    # matching conditional probability; stakes on pushes return (ROI 0).
    f = df[df["f5_close_p_home"].notna()
           & (df[col["pl_reliable"]] == True)                    # noqa: E712
           & df[col["pl_team"]].notna()
           & df[col["f5_away_runs"]].notna()
           & df[col["f5_home_runs"]].notna()].copy()
    pushes = int((f[col["f5_home_runs"]] == f[col["f5_away_runs"]]).sum())
    f = f[f[col["f5_home_runs"]] != f[col["f5_away_runs"]]]
    if len(f):
        f["winner5"] = np.where(f[col["f5_home_runs"]] > f[col["f5_away_runs"]],
                                f[col["home"]], f[col["away"]])
        p_side = np.where(f[col["pl_team"]] == f[col["home"]],
                          f["f5_close_p_home"], 1 - f["f5_close_p_home"])
        w = int((f[col["pl_team"]] == f["winner5"]).sum())
        n = len(f)
        exp, var = float(p_side.sum()), float((p_side * (1 - p_side)).sum())
        z = (w - exp) / np.sqrt(var) if var > 0 else np.nan
        ml5 = np.where(f[col["pl_team"]] == f[col["home"]],
                       f["f5_close_home_ml"], f["f5_close_away_ml"])
        pnl = float(np.where(f[col["pl_team"]] == f["winner5"],
                             [_dec(int(m)) - 1 for m in ml5], -1.0).sum())
        out["platoon_f5"] = dict(n=n, w=w, exp=round(exp, 1),
                                 z=round(float(z), 2), roi_units=round(pnl, 2),
                                 pushes=pushes)
        if verbose:
            print(f"platoon vs F5 close: {w}-{n - w} ({pushes} push) | "
                  f"market-expected {exp:.1f}W -> z {z:+.2f} | ROI {pnl:+.2f}u")
    return out
