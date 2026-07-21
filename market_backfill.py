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


# ---------------------------------------------------------------- helpers ---
def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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
def vs_market_summary(df, col=COL):
    """Vs-market scoreboard for both models. Returns dict for grades.html chips."""
    d = df[df["close_p_home"].notna()].copy()
    d["winner"] = np.where(d[col["home_runs"]] > d[col["away_runs"]],
                           d[col["home"]], d[col["away"]])
    d["fav"] = np.where(d["close_p_home"] >= 0.5, d[col["home"]], d[col["away"]])
    out = {}
    specs = [("xwOBA", d, col["xw_team"]),
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
        print(f"{label}: {w}-{n - w} | market-expected {exp:.1f}W -> z {z:+.2f} | "
              f"ROI {pnl:+.2f}u | agrees w/ fav {fav_agree:.0%} | fav baseline {fav_w}-{n - fav_w}")

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
        print(f"platoon vs F5 close: {w}-{n - w} ({pushes} push) | "
              f"market-expected {exp:.1f}W -> z {z:+.2f} | ROI {pnl:+.2f}u")
    return out
