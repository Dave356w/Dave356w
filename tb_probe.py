"""Does 60-day team total-bases context add anything to the price the market
already posts? Conditional logit, not a median split.

Run this where StatsAPI is reachable -- for this repo that means a GitHub
runner, via `.github/workflows/tb-probe.yml`. It also runs fully offline from a
CSV of pre-computed TB deltas (`--tb-csv`), which is how a Colab session that
has already built the walk-forward can get the answer without a second fetch.

Why this exists
---------------
The proposal is that `|TB delta|` is magnitude CONTEXT: split the slate at a
frozen p50 and read the model's accuracy as higher above the cut than below.
Measured that way on 436 v12 rows it reads T2-minus-T1 = +4.72pp, which is
z = +0.99 -- and the split has 80% power only against a true gap of 13.4pp, so
the median-split form cannot answer its own question at this sample.

The form that can is the one `value_probe` already uses for `xw_net`: put TB in
a logit BESIDE the price and ask whether it carries anything the price does not.
That test is far better powered than a two-cell split because it uses every
row's own magnitude rather than which side of one cut it fell on, and because
its pooled arm does not need the model at all.

Two things make this worth running rather than arguing. TB is EXTERNAL -- it is
not inside the model the way `xw_net` is, and it is not the market's own opinion
the way price is -- so unlike `delta_filter_test`'s axis it has not already been
consumed. And the frozen p50 came off prior seasons, so nothing here is a search
over thresholds: the tiers are a-priori and the continuous arms use no cut.

What it cannot answer
---------------------
It cannot say whether a DIFFERENT TB construction would separate. The feature is
fixed to the proposal's own definition (same-season, strictly prior dates, 60-day
window, league-mean denominated) so the reading is comparable to the report that
motivated it; re-specifying the feature and re-reading is a search, and the
null-max rule in CLAUDE.md applies to it.

It cannot license a selection rule. A coefficient is not a registration, and a
rule built on whatever this prints would be fitted on these rows.

And it cannot rescue a raw-accuracy reading. Raw win rate over a magnitude tier
is mostly base rate -- on the same rows a median split on the market's own
`|p-.5|` returns +11.26pp with no model information in it at all -- which is why
every tier row below carries always-chalk and excess-vs-price beside it.

No lookahead
------------
Every TB feature is built from box scores dated STRICTLY BEFORE the game it
describes, in the same season, so a row's context could have been computed the
morning of its slate. A finished game's box score is immutable, so pulling one
after the fact is backfill and not the `.savant_cache/` hazard: nothing here
reconstructs a prediction, and no ledger row is written.

Nothing in this probe ships. It moves no lean, writes no ledger row, and carries
no `MODEL_TAG` implication.

Note one import dependency, recorded rather than worked around: `logit` and the
Newton-Raphson `fit` are taken from `value_probe` so this repo keeps ONE spelling
of them, and `value_probe` imports `build_site`, which refuses a non-xwOBA
`MODEL_TAG` at import. So this probe runs under an xwOBA build only. That is the
build it was written for; under a wOBA primary the pooled arm would still be
askable and this module would not run, which is a cost accepted to avoid a
second copy of the estimator.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from market_backfill import chalk_is_home, excess_se
from value_probe import fit, logit

BASE_URL = "https://statsapi.mlb.com/api/v1"
LEDGER = "data/mlb_lean_ledger.csv"

# The proposal's own feature definition, reproduced so the reading is comparable
# to the report that motivated this. Changing any of these three re-specifies
# the statistic, which makes a second reading a search rather than a check.
LOOKBACK_DAYS = 60
BURNIN_LOG_ROWS = 150
LEAGUE_TB_FALLBACK = 13.0

# Frozen on 2024-2025 walk-forward TB history (n=4,697), outcome-independent:
# it is the median of the training distribution, not a cut chosen against wins.
# Carried as a default rather than re-fitted here -- re-deriving it against the
# rows being scored is the constants-frozen-from-data defect, and the tier arm
# is only a-priori because this number was fixed before these games were seen.
TB_P50_FROZEN = 0.139793

# Below this the logit is not worth printing; it is a convergence floor, not a
# credibility gate. Every coefficient prints with its standard error.
N_FIT_MIN = 30


def _get_json(url, params=None, retries=4, timeout=20):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
    return {}


def _total_bases(batting):
    """TB from a boxscore batting line, reconstructed when not served.

    StatsAPI supplies `totalBases` on most lines but not all, and a missing key
    silently zeroes a team-game, so the identity is spelled out as the fallback
    rather than defaulted to 0.
    """
    tb = batting.get("totalBases")
    if tb is not None:
        return float(tb)
    h = float(batting.get("hits", 0) or 0)
    d = float(batting.get("doubles", 0) or 0)
    t = float(batting.get("triples", 0) or 0)
    hr = float(batting.get("homeRuns", 0) or 0)
    return (h - d - t - hr) + 2 * d + 3 * t + 4 * hr


def fetch_schedule(season, game_type="R"):
    """Final regular-season games for one season, as (game_pk, date, ids)."""
    js = _get_json(f"{BASE_URL}/schedule", {
        "sportId": 1, "season": season, "gameType": game_type,
        "startDate": f"{season}-01-01", "endDate": f"{season}-12-31",
    })
    rows = []
    for day in js.get("dates", []):
        for gm in day.get("games", []):
            if str(gm.get("status", {}).get("abstractGameState")) != "Final":
                continue
            teams = gm.get("teams", {})
            rows.append({
                "game_pk": int(gm["gamePk"]),
                "season": int(season),
                "date": pd.Timestamp(day["date"]).normalize(),
                "home_id": int(teams["home"]["team"]["id"]),
                "away_id": int(teams["away"]["team"]["id"]),
            })
    return pd.DataFrame(rows)


def fetch_box_tb(game_pk):
    js = _get_json(f"{BASE_URL}/game/{game_pk}/boxscore")
    teams = js.get("teams", {})
    out = {}
    for side in ("home", "away"):
        stats = teams.get(side, {}).get("teamStats", {}).get("batting", {})
        out[f"{side}_tb"] = _total_bases(stats)
        out[f"{side}_runs"] = float(stats.get("runs", 0) or 0)
    return out


def dedupe_games(games):
    """One row per game_pk, keeping its earliest listed date.

    The schedule serves a resumed or rescheduled game under MORE THAN ONE date,
    so a raw concat carries the same game_pk twice. That is not a cosmetic
    duplicate: `team_logs` is built from this frame, so a doubled game enters
    every 60-day window twice and skews the context of every LATER game, not
    only its own row. It then fans out the ledger join, inflating n and
    understating every standard error on the report.

    Caught by the report's own coverage line reading `996 of 982` -- a join that
    returns more rows than the frame it joins into cannot be right, which is why
    that line is printed rather than assumed and why the merge below validates.
    """
    g = games.sort_values(["date", "game_pk"])
    return g.drop_duplicates(subset="game_pk", keep="first").reset_index(drop=True)


def fetch_tb_games(seasons, workers=16, progress=True):
    """Schedule + box scores for whole seasons, as one frame of team totals."""
    frames = [fetch_schedule(s) for s in seasons]
    games = dedupe_games(pd.concat(frames, ignore_index=True))
    pks = games["game_pk"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        boxes = list(pool.map(fetch_box_tb, pks))
    if progress:
        print(f"[tb] fetched {len(boxes)} box scores over seasons {list(seasons)}",
              file=sys.stderr)
    box = pd.DataFrame(boxes, index=games.index)
    return pd.concat([games, box], axis=1)


def team_logs(games):
    """One row per team-game: what that team hit for, and what it allowed."""
    cols = ["game_pk", "season", "date"]
    h = games[cols + ["home_id", "home_tb", "away_tb"]].copy()
    a = games[cols + ["away_id", "away_tb", "home_tb"]].copy()
    h.columns = a.columns = cols + ["team_id", "tb", "tb_opp"]
    return pd.concat([h, a], ignore_index=True).sort_values(["date", "game_pk"])


def tb_features(games, only_game_pks=None):
    """Strict prior-date, same-season, 60-day TB deltas. Home-oriented.

    A game's feature reads only team-games dated STRICTLY BEFORE it, so the row
    could have been produced on the morning of its own slate. The burn-in skips
    early dates where the window holds too little to mean anything, which is why
    the probe reports its own coverage rather than assuming the ledger's.
    """
    games = dedupe_games(games.copy())
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    logs = team_logs(games)
    rows = []
    for day in sorted(games["date"].unique()):
        today = games[games["date"] == day]
        if today.empty:
            continue
        season = int(today["season"].iloc[0])
        hist = logs[(logs["season"] == season) & (logs["date"] < day)]
        if len(hist) < BURNIN_LOG_ROWS:
            continue
        win = hist[hist["date"] >= day - pd.Timedelta(days=LOOKBACK_DAYS)]
        lg = float(win["tb"].mean()) if len(win) else np.nan
        if not np.isfinite(lg) or lg <= 0:
            lg = LEAGUE_TB_FALLBACK
        means = win.groupby("team_id")[["tb", "tb_opp"]].mean()

        def net(team_id):
            if team_id not in means.index:
                return np.nan
            row = means.loc[team_id]
            return float(row["tb"]) - float(row["tb_opp"])

        for _, gm in today.iterrows():
            pk = int(gm["game_pk"])
            if only_game_pks is not None and pk not in only_game_pks:
                continue
            delta = (net(int(gm["home_id"])) - net(int(gm["away_id"]))) / lg
            rows.append({"game_pk": pk, "tb_delta": float(delta),
                         "abs_tb": abs(float(delta))})
    return pd.DataFrame(rows)


def load_tb_csv(path):
    """Offline path: TB deltas computed elsewhere, keyed on game_pk.

    Accepts the column names this probe writes and the ones the Colab script
    writes, because the point of this path is to read a frame that already
    exists rather than to impose a schema on it. A file carrying only the
    magnitude is accepted and the pooled directional arm is then skipped and
    SAID to be skipped -- never silently dropped.
    """
    f = pd.read_csv(path, float_precision="round_trip")
    ren = {"delta_tb_60": "tb_delta", "abs_tb_60": "abs_tb"}
    f = f.rename(columns={k: v for k, v in ren.items() if k in f.columns})
    if "game_pk" not in f.columns:
        raise ValueError(f"{path}: no game_pk column")
    if "tb_delta" not in f.columns and "abs_tb" not in f.columns:
        raise ValueError(f"{path}: needs tb_delta/delta_tb_60 or abs_tb/abs_tb_60")
    if "tb_delta" not in f.columns:
        f["tb_delta"] = np.nan
    if "abs_tb" not in f.columns:
        f["abs_tb"] = f["tb_delta"].abs()
    f["game_pk"] = pd.to_numeric(f["game_pk"], errors="coerce").astype("Int64")
    f = f.dropna(subset=["game_pk"])[["game_pk", "tb_delta", "abs_tb"]]
    # Same hazard from the other direction: a frame built elsewhere may carry a
    # game twice, and the join would then double those rows silently.
    return f.drop_duplicates(subset="game_pk", keep="first").reset_index(drop=True)


def load_ledger(path=LEDGER):
    """Graded, decided ledger rows with the columns every arm below needs."""
    g = pd.read_csv(path, low_memory=False, float_precision="round_trip")
    g = g[g["status"].astype(str).eq("graded")]
    g = g[g["full_home"].notna() & g["full_away"].notna()]
    g = g[g["full_home"] != g["full_away"]].copy()
    g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce").astype("Int64")
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g["home_won"] = (g["full_home"] > g["full_away"]).astype(float)

    lean = g["xw_lean"].astype(str).str.upper().str.strip()
    home = g["home"].astype(str).str.upper().str.strip()
    away = g["away"].astype(str).str.upper().str.strip()
    g["lean_is_home"] = np.where(lean.eq(home), True,
                                 np.where(lean.eq(away), False, None))
    g["lean_won"] = np.where(
        g["lean_is_home"].isna(), np.nan,
        np.where(g["lean_is_home"].astype("boolean").fillna(False),
                 g["home_won"], 1 - g["home_won"]))
    return g


def attach_price(g, basis):
    """The leaned side's own devigged price, on ONE stated basis.

    `closing` covers the most rows and is what every other calibration surface
    in this repo scores; `pregame` is the price a bettor could have taken and
    covers fewer. They are not mixed and there is no fallback between them --
    a mixed basis is the defect `hybrid_price_source` exists to make visible.
    """
    col = "close_p_home" if basis == "closing" else "pregame_p_home"
    out = g.copy()
    out["p_home"] = pd.to_numeric(out[col], errors="coerce")
    is_home = out["lean_is_home"].astype("boolean")
    out["q_lean"] = np.where(is_home.fillna(False), out["p_home"], 1 - out["p_home"])
    out.loc[is_home.isna(), "q_lean"] = np.nan
    out["chalk_is_home"] = chalk_is_home(out["p_home"])
    out["chalk_won"] = np.where(out["chalk_is_home"], out["home_won"], 1 - out["home_won"])
    out["chalk_p"] = np.maximum(out["p_home"], 1 - out["p_home"])
    return out


def _z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if np.isfinite(sd) and sd > 0 else np.zeros_like(v)


def fit_arm(y, cols, names):
    """Logit with standard errors. Returns rows of (name, coefficient, se, z)."""
    X = np.column_stack(cols)
    y = np.asarray(y, dtype=float)
    b = fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    pr = 1.0 / (1.0 + np.exp(-Xd @ b))
    W = np.clip(pr * (1 - pr), 1e-9, None)
    cov = np.linalg.inv((Xd * W[:, None]).T @ Xd + 1e-6 * np.eye(Xd.shape[1]))
    se = np.sqrt(np.diag(cov))
    return [(nm, float(b[i + 1]), float(se[i + 1]), float(b[i + 1] / se[i + 1]))
            for i, nm in enumerate(names)]


def oos_log_loss(g, use_market, use_tb, train_min=80):
    """Walk-forward log loss: fit on strictly prior slates, score the next.

    The decisive arm, and the reason the coefficient is not read alone. If
    `price + TB` scores WORSE than `market-fitted`, TB is subtracting
    information from the price, which is what a noise feature does -- the same
    ranking `value_probe` prints for `xw_net`.
    """
    mu, sd = g["tb_delta"].mean(), g["tb_delta"].std()
    losses = []
    for s in sorted(g["game_date"].unique()):
        tr, te = g[g["game_date"] < s], g[g["game_date"] == s]
        if len(tr) < train_min or te.empty:
            continue

        def design(f):
            cols = []
            if use_market:
                cols.append(logit(f["p_home"].values))
            if use_tb:
                cols.append((f["tb_delta"].values - mu) / sd)
            return np.column_stack(cols)

        if use_market or use_tb:
            b = fit(design(tr), tr["home_won"].values.astype(float))
            p = 1.0 / (1.0 + np.exp(-(np.column_stack(
                [np.ones(len(te)), design(te)]) @ b)))
        else:
            p = te["p_home"].values
        p = np.clip(p, 1e-6, 1 - 1e-6)
        y = te["home_won"].values.astype(float)
        losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.concatenate(losses).mean()) if losses else float("nan")


def tier_rows(f, p50):
    """Per-tier record WITH its two controls, because the raw rate is base rate.

    Every tier line carries always-chalk on the identical rows and the tier's
    own mean implied price. A magnitude tier sorts games onto the price axis, so
    a tier with a high raw win rate is the expected reading whether or not TB
    knows anything -- the controls are what separate the two.
    """
    rows = []
    for name, mask in (("ALL", pd.Series(True, index=f.index)),
                       ("TB_T1", f["abs_tb"] < p50),
                       ("TB_T2", f["abs_tb"] >= p50)):
        d = f[mask]
        if not len(d):
            continue
        rate, imp = d["lean_won"].mean(), d["q_lean"].mean()
        ch, chp = d["chalk_won"].mean(), d["chalk_p"].mean()
        rows.append({
            "tier": name, "n": len(d),
            "record": f"{int(d['lean_won'].sum())}-{int((1 - d['lean_won']).sum())}",
            "raw": rate, "implied": imp,
            "excess": 100 * (rate - imp), "excess_se": 100 * excess_se(d["q_lean"]),
            "chalk_excess": 100 * (ch - chp),
            "model_minus_chalk": 100 * ((rate - imp) - (ch - chp)),
        })
    return pd.DataFrame(rows)


def report(led, tb, tags, basis, p50, out=sys.stdout):
    """Every arm, each naming its own row set and price basis."""
    say = lambda s="": print(s, file=out)
    # `one_to_one` rather than a post-hoc row count: pandas raises on a
    # duplicate key in EITHER frame, so a fan-out is impossible rather than
    # merely detectable. The coverage line below is then a real coverage
    # figure and can never again exceed its own denominator.
    f = led.merge(tb, on="game_pk", how="inner", validate="one_to_one")
    f = attach_price(f, basis)
    f = f[f["p_home"].notna()].copy()

    say("=" * 78)
    say("TB CONTEXT AGAINST PRICE -- conditional logit, not a median split")
    say("=" * 78)
    say(f"price basis   : {basis} (no fallback to the other basis)")
    say(f"frozen p50    : {p50:.6f}  (2024-2025 walk-forward median; not re-fitted here)")
    say(f"TB coverage   : {len(f)} of {len(led)} graded decided ledger rows carry a TB feature")
    say(f"date range    : {f['game_date'].min():%Y-%m-%d} .. {f['game_date'].max():%Y-%m-%d}"
        f"   ({f['game_date'].nunique()} slates)")
    has_dir = f["tb_delta"].notna().any()

    fam = f[f["model_tag"].isin(tags) & f["q_lean"].notna() & f["lean_won"].notna()].copy()
    say(f"family rows   : {len(fam)} under {sorted(tags)}")
    say()

    say("1. TIER RECORDS, WITH THE TWO CONTROLS THE MEDIAN SPLIT OMITS")
    say("   Raw rate over a magnitude tier is mostly base rate. Read the last two")
    say("   columns: what the tier beat its OWN prices by, and whether backing the")
    say("   favourite on the identical rows would have done the same.")
    if len(fam):
        t = tier_rows(fam, p50)
        say(f"   {'tier':<8}{'n':>5}{'record':>10}{'raw':>8}{'implied':>9}"
            f"{'vs price':>11}{'+/-':>7}{'chalk':>9}{'model-chalk':>13}")
        for _, r in t.iterrows():
            say(f"   {r['tier']:<8}{r['n']:>5}{r['record']:>10}{r['raw']:>8.3f}"
                f"{r['implied']:>9.3f}{r['excess']:>+11.2f}{r['excess_se']:>7.2f}"
                f"{r['chalk_excess']:>+9.2f}{r['model_minus_chalk']:>+13.2f}")
    else:
        say("   no family rows carry both a TB feature and a price on this basis.")
    say()

    say("2. DOES TB ADD ANYTHING TO THE PRICE? (pooled, metric-free)")
    say("   P(home wins) ~ logit(close p_home) + z(signed TB delta), EVERY graded")
    say("   family. The outcome is a box score and the price is a price, so neither")
    say("   knows which model wrote the row -- scoping this to one family would")
    say("   halve the sample for a reason that cannot apply.")
    pooled = f[f["p_home"].notna()]
    if has_dir and len(pooled) >= N_FIT_MIN:
        arm = fit_arm(pooled["home_won"], [logit(pooled["p_home"].values),
                                           _z(pooled["tb_delta"])],
                      ["market logit", "z(TB signed)"])
        say(f"   n={len(pooled)}")
        for nm, b, se, z in arm:
            say(f"     {nm:<16}{b:+8.3f} +/- {se:.3f}   z={z:+.2f}")
        say("   market logit at 1.00 means the close needs no correction.")
    elif not has_dir:
        say("   SKIPPED: the TB frame carries magnitude only, no signed delta.")
    else:
        say(f"   SKIPPED: n={len(pooled)} below the convergence floor of {N_FIT_MIN}.")
    say()

    say("3. DOES TB MAGNITUDE SAY WHEN TO TRUST THE LEAN? (current family)")
    say("   P(lean wins) ~ logit(leaned side's price) + z(|TB delta|). A positive")
    say("   TB coefficient is the proposal's claim: the lean is likelier right when")
    say("   the TB mismatch is large, over and above the price already saying so.")
    if len(fam) >= N_FIT_MIN:
        for label, extra, names in (
            ("price + |TB|", [_z(fam["abs_tb"])], ["z(|TB|)"]),
            ("price + |TB| tier", [(fam["abs_tb"] >= p50).to_numpy(float)], ["1[TB_T2]"]),
            ("price + |xw_net| + |TB|",
             [_z(fam["xw_net"].abs()), _z(fam["abs_tb"])], ["z(|xw_net|)", "z(|TB|)"]),
        ):
            arm = fit_arm(fam["lean_won"], [logit(fam["q_lean"].values)] + extra,
                          ["market logit"] + names)
            say(f"   {label}  (n={len(fam)})")
            for nm, b, se, z in arm:
                say(f"     {nm:<16}{b:+8.3f} +/- {se:.3f}   z={z:+.2f}")
        say("   The third arm is the one that matters for shipping: |xw_net| is")
        say("   already inside the decision, so TB has to add on top of it.")
    else:
        say(f"   SKIPPED: n={len(fam)} below the convergence floor of {N_FIT_MIN}.")
    say()

    say("4. OUT-OF-SAMPLE LOG LOSS (pooled, walk-forward; lower is better)")
    say("   Fitted on strictly prior slates, scored on the next. This is the arm")
    say("   a coefficient cannot fake: if 'price + TB' ranks BELOW 'market-fitted',")
    say("   TB is subtracting information from the price.")
    if has_dir and len(pooled) >= N_FIT_MIN:
        rows = [("raw close (nothing fitted)", oos_log_loss(pooled, False, False)),
                ("market-fitted", oos_log_loss(pooled, True, False)),
                ("price + TB", oos_log_loss(pooled, True, True)),
                ("TB alone", oos_log_loss(pooled, False, True))]
        for nm, v in sorted(rows, key=lambda kv: (np.isnan(kv[1]), kv[1])):
            say(f"     {nm:<28}{v:.4f}")
    else:
        say("   SKIPPED: no signed TB delta, or too few rows.")
    say()

    say("READING THIS")
    say("  * A tier's raw rate is not evidence. Its excess over price, read against")
    say("    the chalk column beside it, is. The measured hazard on these rows: a")
    say("    median split on the market's own |p-.5| returns +11.26pp of raw lift")
    say("    while the model's edge over chalk moves the OTHER way.")
    say("  * Nothing here registers a rule, and no threshold was fitted on these")
    say("    rows. A rule built on this output would be fitted on it, and would")
    say("    need its own forward window and its own null-max test.")
    say("  * The delta filter is the precedent this sits beside: the model's own")
    say("    magnitude reads null against price and its chalk control inverts it.")
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--tb-csv", default=None,
                    help="pre-computed TB deltas (game_pk + tb_delta/abs_tb); "
                         "skips every network call")
    ap.add_argument("--seasons", default=None,
                    help="seasons to fetch when --tb-csv is absent, e.g. 2026")
    ap.add_argument("--tags", default=None,
                    help="model tags for the family arms (default: build_site.RECORD_TAGS)")
    ap.add_argument("--basis", choices=("closing", "pregame"), default="closing")
    ap.add_argument("--p50", type=float, default=TB_P50_FROZEN)
    ap.add_argument("--tb-out", default=None, help="write the computed TB frame here")
    ap.add_argument("--out", default=None, help="write the report here as well as stdout")
    a = ap.parse_args(argv)

    led = load_ledger(a.ledger)
    if a.tags:
        tags = tuple(t.strip() for t in a.tags.split(",") if t.strip())
    else:
        from build_site import RECORD_TAGS
        tags = tuple(RECORD_TAGS)

    if a.tb_csv:
        tb = load_tb_csv(a.tb_csv)
    else:
        seasons = ([int(s) for s in a.seasons.split(",")] if a.seasons
                   else sorted({d.year for d in led["game_date"].dropna()}))
        games = fetch_tb_games(seasons)
        tb = tb_features(games, only_game_pks=set(led["game_pk"].dropna().astype(int)))
        if a.tb_out:
            tb.to_csv(a.tb_out, index=False)

    report(led, tb, tags, a.basis, a.p50)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            report(led, tb, tags, a.basis, a.p50, out=fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
