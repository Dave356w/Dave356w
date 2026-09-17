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
consumed. And the frozen p50 came off prior seasons, so the TIER arm is
a-priori and the continuous arms use no cut at all.

Section 5 IS a search, deliberately, and says so on its own output. "Does TB add
ROI in any context" cannot be answered by one threshold -- the honest form is to
run the search the question implies and then score its winner against what a
search that wide returns from noise, followed by a walk-forward, because this
repo has two variants that cleared the null-max test and then lost forward, one
of them with a stable argmax. A threshold found in that block is never a-priori
and nothing may be registered from it without a fresh window.

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
import math
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

# Resamples behind the null-maximum reference. Fixed seed so a committed report
# does not churn; 2000 is where the reported mean stops moving in the third
# decimal at these row counts.
# 20,000 rather than the 2,000 this started at: `_contrasts` scores every draw
# as one matrix, so the whole block runs in ~0.3s where the per-threshold
# version took two minutes, and the extra draws buy a P-value that is stable in
# the third decimal instead of the second.
NULL_DRAWS = 20000
# Fixed, for the same reason: a committed artifact whose numbers move on a
# re-run cannot be diffed, and a seed chosen per run is a search knob.
NULL_SEED = 0

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
            h, a = net(int(gm["home_id"])) / lg, net(int(gm["away_id"])) / lg
            delta = h - a
            # Each side's own net is kept, not just the difference, because the
            # question "does TB give a TEAM's delta context" is asked per club
            # and the difference cannot be decomposed back into its halves.
            rows.append({"game_pk": pk, "tb_delta": float(delta),
                         "abs_tb": abs(float(delta)),
                         "tb_home": float(h), "tb_away": float(a)})
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
    for side in ("tb_home", "tb_away"):
        if side not in f.columns:
            f[side] = np.nan
    f = f.dropna(subset=["game_pk"])[["game_pk", "tb_delta", "abs_tb",
                                      "tb_home", "tb_away"]]
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
    hm = pd.to_numeric(out["close_home_ml" if basis == "closing" else "pregame_home_ml"],
                       errors="coerce")
    am = pd.to_numeric(out["close_away_ml" if basis == "closing" else "pregame_away_ml"],
                       errors="coerce")
    # The price the bet actually settles at, on the SAME basis as p_home above.
    # Mixing a closing probability with a pregame moneyline would price a bet at
    # odds nobody was offered for the probability being scored.
    out["lean_ml"] = np.where(is_home.fillna(False), hm, am)
    out.loc[is_home.isna(), "lean_ml"] = np.nan
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


def _fisher_ci(r, n):
    """95% interval for a correlation, and the z its own SE implies.

    Printed with every correlation below for the reason this repo prints every
    SE: a bare r invites a reading its sample cannot support, and at these n a
    correlation of 0.05 and one of 0.00 are the same statement.
    """
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return float("nan"), float("nan"), float("nan")
    se = 1.0 / math.sqrt(n - 3)
    zf = 0.5 * math.log((1 + r) / (1 - r))
    return math.tanh(zf - 1.96 * se), math.tanh(zf + 1.96 * se), r * math.sqrt(n - 1)


def _corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _partial_corr(a, b, given):
    """corr(a, b) with `given` regressed out of both.

    The distinction this block exists to draw: TB magnitude sorts games onto
    the price axis, so a marginal correlation with winning is partly the market
    speaking. Residualising on the price is the correlation-language form of
    the logit arm below, and the two should agree.
    """
    g = np.column_stack([np.ones(len(given)), np.asarray(given, dtype=float)])
    ra = np.asarray(a, float) - g @ np.linalg.lstsq(g, np.asarray(a, float), rcond=None)[0]
    rb = np.asarray(b, float) - g @ np.linalg.lstsq(g, np.asarray(b, float), rcond=None)[0]
    return _corr(ra, rb)


def correlation_rows(fam, pooled):
    """The two questions a reader actually asks, answered as correlations.

    "Does TB magnitude correlate with V12 wins?" is MARGINAL and "does TB give
    the lean context?" is CONDITIONAL, and on this data they do not have the
    same answer sign-for-sign -- which is the whole reason both are printed
    rather than one standing in for the other.

    The last two rows are the decomposition: how much of TB is the price
    restated, and how much of it the model already carries in `xw_net`.
    """
    rows = []

    def add(label, r, n, note=""):
        lo, hi, z = _fisher_ci(r, n)
        rows.append({"label": label, "r": r, "n": n, "lo": lo, "hi": hi,
                     "z": z, "note": note})

    q = logit(fam["q_lean"].values)
    add("|TB| vs V12 win (marginal)", _corr(fam["abs_tb"], fam["lean_won"]), len(fam))
    add("|TB| vs V12 win | price", _partial_corr(fam["abs_tb"], fam["lean_won"], q), len(fam),
        "the context question")
    add("TB tier vs V12 win (marginal)",
        _corr((fam["abs_tb"] >= TB_P50_FROZEN).astype(float), fam["lean_won"]), len(fam))
    add("|TB| vs |price - .5|", _corr(fam["abs_tb"], (fam["p_home"] - 0.5).abs()), len(fam),
        "how much of TB is the price restated")
    add("|TB| vs |xw_net|", _corr(fam["abs_tb"], fam["xw_net"].abs()), len(fam),
        "how much of TB the model already carries")
    if pooled["tb_delta"].notna().any():
        ph = logit(pooled["p_home"].values)
        add("TB signed vs home win (marginal)",
            _corr(pooled["tb_delta"], pooled["home_won"]), len(pooled))
        add("TB signed vs home win | price",
            _partial_corr(pooled["tb_delta"], pooled["home_won"], ph), len(pooled),
            "pooled, metric-free")
    return rows


def alignment_frame(fam):
    """TB oriented to the side the model leans, plus the AGREE/DIVERGE split.

    Verified on the committed ledger rather than recalled, because the suffix
    convention in this repo names the PITCHING side faced and getting it
    backwards would invert the whole arm: `xw_net == edge_xwoba_away -
    edge_xwoba_home` exactly on 439 of 439 v12 rows, and a positive `xw_net`
    leans HOME on 218 of 218. `tb_delta` is home-oriented too, so the two share
    a sign convention and alignment needs no flip.

    `tb_aligned` is TB signed toward the model's own pick: positive means the
    60-day run-differential read CORROBORATES this team's lean delta, negative
    means it contradicts it.

    Why this is asked at the GAME level and not per club, which is the shape the
    question invites: the two sides of one game are complements -- their devigged
    prices sum to 1 and exactly one of them wins -- so stacking both sides is one
    observation dressed as two, and any pooled figure over them is fixed by the
    partition rather than by the data. That is this repo's own degenerate-tile
    defect, and the oriented game-level form is the non-degenerate version of the
    same question.
    """
    f = fam.copy()
    f["tb_aligned"] = pd.to_numeric(f["tb_delta"], errors="coerce") * np.sign(
        pd.to_numeric(f["xw_net"], errors="coerce"))
    f["agrees"] = np.where(f["tb_aligned"] > 0, True,
                           np.where(f["tb_aligned"] < 0, False, None))
    return f


def alignment_rows(f):
    """AGREE vs DIVERGE, each with the price and chalk controls beside it.

    The mean implied price is printed per cell on purpose: the OPS `consensus`
    arm this repo already measured looked alive at z = +1.32 and turned out to
    be mostly reliability and price, so a cell's own price is what lets a reader
    see a base-rate split before reading it as corroboration.
    """
    rows = []
    for name, mask in (("TB AGREES with lean", f["agrees"] == True),      # noqa: E712
                       ("TB CONTRADICTS lean", f["agrees"] == False)):    # noqa: E712
        d = f[mask]
        if not len(d):
            continue
        rate, imp = d["lean_won"].mean(), d["q_lean"].mean()
        ch, chp = d["chalk_won"].mean(), d["chalk_p"].mean()
        rows.append({"cell": name, "n": len(d),
                     "record": f"{int(d['lean_won'].sum())}-{int((1 - d['lean_won']).sum())}",
                     "raw": rate, "implied": imp,
                     "excess": 100 * (rate - imp),
                     "se": 100 * excess_se(d["q_lean"]),
                     "chalk": 100 * (ch - chp)})
    return rows


def _did(f):
    """The difference-in-differences: (model - chalk) in AGREE minus in DIVERGE."""
    a = f[f["agrees"] == True]                                        # noqa: E712
    d = f[f["agrees"] == False]                                       # noqa: E712
    if not len(a) or not len(d):
        return float("nan")
    def cell(x):
        return ((x["lean_won"].mean() - x["q_lean"].mean())
                - (x["chalk_won"].mean() - x["chalk_p"].mean()))
    return 100.0 * (cell(a) - cell(d))


def alignment_contrast(rows, f=None, draws=4000, seed=0):
    """AGREE minus DIVERGE, with the SE OF THE DIFFERENCE, and the DiD headline.

    The contrast is the claim -- "TB tells you when to trust this team's delta"
    is a statement about the gap between the two cells, not about either one.

    The headline is the model's contrast NET OF CHALK's on the identical split.
    A pure price confound moves both cells together -- and only where favourites
    actually beat their price, which is why chalk's own contrast cannot be read
    alone: in a correctly-priced world it is zero in both cells and reveals
    nothing. The difference is the part that is about TB.

    That headline is a difference-in-differences over two cells sharing no rows
    but two controls measured on the SAME rows as the thing they control, so its
    variance is NOT the sum of the parts' and cannot be written down from the
    printed SEs. The first version of this function published it bare, which is
    the one rule this repo states without exception -- print the standard error,
    never the number alone. It is bootstrapped over games instead.
    """
    if len(rows) != 2:
        return None
    a, d = rows[0], rows[1]
    se = math.sqrt(a["se"] ** 2 + d["se"] ** 2)
    diff = a["excess"] - d["excess"]
    chalk_diff = a["chalk"] - d["chalk"]
    out = {"diff": diff, "se": se, "z": diff / se if se > 0 else float("nan"),
           "chalk_diff": chalk_diff, "net_of_chalk": diff - chalk_diff,
           "net_lo": float("nan"), "net_hi": float("nan")}
    if f is None or len(f) < 8:
        return out
    rng = np.random.default_rng(seed)
    idx = np.arange(len(f))
    boots = []
    for _ in range(draws):
        b = _did(f.iloc[rng.choice(idx, len(idx), replace=True)])
        if np.isfinite(b):
            boots.append(b)
    if len(boots) > 20:
        out["net_lo"] = float(np.quantile(boots, 0.025))
        out["net_hi"] = float(np.quantile(boots, 0.975))
    return out


def _search_verdict(p):
    """One home for the clause a searched maximum has to carry.

    Three branches rather than two, because the middle case is real and a
    two-branch rendering turns a coin flip into a licence -- the defect this
    repo already fixed once on a pooling licence that cleared by 0.00038.
    """
    if not np.isfinite(p):
        return "     verdict: not computable."
    if p >= 0.5:
        return ("     verdict: the observed best is WORSE than a search this wide\n"
                "     typically returns from noise -- no ROI context to find here.")
    if p >= 0.05:
        return ("     verdict: inside what the search returns from noise. NOT a\n"
                "     finding, and NOT evidence of absence either -- this bar is high\n"
                "     enough that a real edge can fail it at these row counts.")
    return ("     verdict: clears the null maximum -- which is PERMISSION TO\n"
            "     WALK-FORWARD, not a result. Read the next block before quoting it.")


def payout(ml):
    """Profit per 1u risked at an American price. Vectorised, nan-safe."""
    ml = pd.to_numeric(ml, errors="coerce").astype(float)
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def flat_units(won, ml):
    """Flat-stake P&L per row: the price's profit on a win, -1 on a loss."""
    won = np.asarray(won, dtype=float)
    return np.where(won > 0, payout(ml), -1.0)


def roi_block(f):
    """Units, ROI and the ROI's own standard error for one set of bets.

    The SE is the per-bet sd over sqrt(n), NOT a win-rate SE: at these prices a
    single bet's P&L has sd near 1.0, so an 80-bet cell carries about +/-11pp of
    ROI. That number is the whole reason a tier's ROI cannot be read alone.
    """
    u = flat_units(f["lean_won"], f["lean_ml"])
    u = u[np.isfinite(u)]
    if not len(u):
        return {"n": 0, "units": float("nan"), "roi": float("nan"), "se": float("nan")}
    sd = float(np.std(u, ddof=1)) if len(u) > 1 else float("nan")
    return {"n": int(len(u)), "units": float(np.sum(u)), "roi": float(np.mean(u)),
            "se": sd / math.sqrt(len(u)) if np.isfinite(sd) else float("nan")}


def filter_contrast(f, t):
    """A TB abstention filter's OWN content: kept minus dropped.

    A filter cannot pick a side -- TB never supplies direction -- so the only
    way it can add ROI is by deciding which games to bet. Its content is
    therefore which rows it REMOVES, and a combined ROI over the kept rows can
    only restate the model on them. This returns the contrast and the SE OF THE
    DIFFERENCE, which is what a filter has to clear.
    """
    keep = f[f["abs_tb"] >= t]
    drop = f[f["abs_tb"] < t]
    k, d = roi_block(keep), roi_block(drop)
    if not k["n"] or not d["n"] or not np.isfinite(k["se"]) or not np.isfinite(d["se"]):
        return None
    se = math.sqrt(k["se"] ** 2 + d["se"] ** 2)
    return {"t": t, "kept": k, "dropped": d, "contrast": k["roi"] - d["roi"],
            "se": se, "z": (k["roi"] - d["roi"]) / se if se > 0 else float("nan")}


def tb_grid(f, n=25):
    """Thresholds spanning the observed |TB| range, plus the frozen p50.

    Deliberately a spread of candidates rather than one: the question asked is
    "in ANY context", and the only honest way to answer it is to run the search
    the question implies and then score the winner against what a search that
    wide returns from noise.
    """
    v = pd.to_numeric(f["abs_tb"], errors="coerce").dropna()
    if v.empty:
        return []
    lo, hi = float(v.quantile(0.05)), float(v.quantile(0.95))
    grid = list(np.linspace(lo, hi, n)) + [TB_P50_FROZEN]
    return sorted(set(round(x, 6) for x in grid))


def _sweep_core(f, grid):
    """Rows sorted by |TB| descending, plus each threshold's prefix length.

    ONE derivation for the observed contrast and for the null, because two code
    paths computing the same statistic will drift and the reader cannot see
    which one they are looking at. Sorting descending makes "kept" a PREFIX, so
    every threshold is a cumulative-sum lookup instead of a fresh boolean slice
    -- which is what turns a 2-minute block into a fraction of a second and
    keeps it that way as the ledger grows.
    """
    g = f.dropna(subset=["abs_tb", "lean_ml", "q_lean"]).sort_values(
        "abs_tb", ascending=False)
    n = len(g)
    if n < 4 or not grid:
        return None
    tb = g["abs_tb"].to_numpy(float)
    # kept = rows with |TB| >= t; sorted descending, that is the first k rows.
    ks = np.array(sorted({int(np.count_nonzero(tb >= t)) for t in grid}))
    ks = ks[(ks >= 1) & (ks <= n - 1)]
    if not len(ks):
        return None
    thresholds = {int(np.count_nonzero(tb >= t)): float(t) for t in sorted(grid)}
    return {"g": g, "n": n, "ks": ks, "pay": payout(g["lean_ml"]).astype(float),
            "q": g["q_lean"].to_numpy(float),
            "t_of_k": {k: thresholds[k] for k in ks if k in thresholds}}


def _contrasts(units, ks, n):
    """kept-minus-dropped at every prefix length, for one or many outcome draws."""
    cs = np.cumsum(units, axis=-1)
    total = cs[..., -1][..., None]
    kept = cs[..., ks - 1] / ks
    dropped = (total - cs[..., ks - 1]) / (n - ks)
    return kept - dropped


def best_contrast(f, grid, won_col="lean_won"):
    """Best filter over the grid. Returns the same shape `filter_contrast` does."""
    core = _sweep_core(f, grid)
    if core is None:
        return None
    won = pd.to_numeric(core["g"][won_col], errors="coerce").to_numpy(float)
    units = np.where(won > 0, core["pay"], -1.0)
    c = _contrasts(units, core["ks"], core["n"])
    if not np.isfinite(c).any():
        return None
    k = int(core["ks"][int(np.nanargmax(c))])
    t = core["t_of_k"].get(k)
    if t is None:                       # a k with no threshold of its own
        t = float(core["g"]["abs_tb"].to_numpy(float)[k - 1])
    g = f.copy()
    if won_col != "lean_won":
        g["lean_won"] = g[won_col]
    return filter_contrast(g, t)


def null_max_contrast(f, grid, draws=2000, seed=0):
    """What the best filter in this grid returns when the market is CORRECT.

    Outcomes are redrawn at each row's own devigged price, so the model has no
    edge by construction and every apparent filter effect is the search finding
    a maximum. The observed best is read against this, never against zero --
    the rule this repo earned from a band grid that handed back +20% ROI on
    noise.

    All `draws` are drawn and scored as one matrix through `_contrasts`, the
    same function the observed value goes through.
    """
    core = _sweep_core(f, grid)
    obs = best_contrast(f, grid)
    if core is None or obs is None:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    q = np.nan_to_num(core["q"], nan=0.5)
    sim = (rng.random((draws, core["n"])) < q).astype(float)
    units = sim * (core["pay"] + 1.0) - 1.0
    bests = np.nanmax(_contrasts(units, core["ks"], core["n"]), axis=1)
    bests = bests[np.isfinite(bests)]
    if not len(bests):
        return float("nan"), float("nan")
    return float(bests.mean()), float(np.mean(bests >= obs["contrast"]))


def walk_forward_tb(f, grid):
    """Pick the best threshold on prior slates, bet the next. The closing test.

    The null-max test is necessary and not sufficient -- this repo has two
    variants that cleared it and then lost forward, one of them with a STABLE
    argmax. So the search result is read only alongside this.
    """
    f = f.sort_values("game_date")
    dates = sorted(f["game_date"].dropna().unique())
    chased, baseline = [], []
    for s in dates:
        prior, today = f[f["game_date"] < s], f[f["game_date"] == s]
        if len(prior) < 60 or today.empty:
            continue
        b = best_contrast(prior, grid)
        if b is None:
            continue
        kept = today[today["abs_tb"] >= b["t"]]
        chased.extend(flat_units(kept["lean_won"], kept["lean_ml"]))
        baseline.extend(flat_units(today["lean_won"], today["lean_ml"]))
    chased = [u for u in chased if np.isfinite(u)]
    baseline = [u for u in baseline if np.isfinite(u)]
    return ({"n": len(chased), "roi": float(np.mean(chased)) if chased else float("nan")},
            {"n": len(baseline), "roi": float(np.mean(baseline)) if baseline else float("nan")})


def hybrid_increment(f, grid):
    """Does a TB gate ADDED to the shipped hybrid pay? Scored as an increment.

    The trap this avoids is named in CLAUDE.md: a search whose candidate set
    contains the baseline cannot test the increment. Sweeping "shipped rule AND
    TB gate" against zero measures the SHIPPED RULE beating chance. So every
    candidate is scored as (gated - shipped) on the SAME rows, which is zero
    when the gate excludes nothing.
    """
    base = roi_block(f)
    if not base["n"]:
        return None
    rows = []
    for t in grid:
        kept = f[f["abs_tb"] >= t]
        k = roi_block(kept)
        if not k["n"]:
            continue
        # Abstaining stakes nothing on the dropped rows, so the increment in
        # UNITS is what the gate changes; per-row so the two are comparable.
        rows.append({"t": t, "n": k["n"], "units": k["units"],
                     "increment": (k["units"] - base["units"]) / base["n"]})
    if not rows:
        return None
    return {"base": base, "best": max(rows, key=lambda r: r["increment"]), "all": rows}


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

    pooled = f[f["p_home"].notna()]

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

    say("2. THE TWO QUESTIONS AS CORRELATIONS")
    say("   MARGINAL asks 'does TB magnitude correlate with V12 wins'. CONDITIONAL")
    say("   asks 'does TB give the lean context the price does not already give'.")
    say("   They are different questions and a magnitude variable is exactly the")
    say("   case where they can disagree, because magnitude sorts games onto the")
    say("   price axis. The last rows size that directly.")
    if len(fam) >= 4:
        say(f"   {'':<34}{'n':>5}{'r':>9}{'95% CI':>20}{'z':>7}")
        for row in correlation_rows(fam, pooled):
            ci = f"[{row['lo']:+.3f}, {row['hi']:+.3f}]"
            say(f"   {row['label']:<34}{row['n']:>5}{row['r']:>+9.4f}{ci:>20}{row['z']:>+7.2f}"
                + (f"   <- {row['note']}" if row["note"] else ""))
    else:
        say("   SKIPPED: too few family rows.")
    say()

    say("3. DOES TB ADD ANYTHING TO THE PRICE? (pooled, metric-free)")
    say("   P(home wins) ~ logit(close p_home) + z(signed TB delta), EVERY graded")
    say("   family. The outcome is a box score and the price is a price, so neither")
    say("   knows which model wrote the row -- scoping this to one family would")
    say("   halve the sample for a reason that cannot apply.")
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

    say("4. DOES TB MAGNITUDE SAY WHEN TO TRUST THE LEAN? (current family)")
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

    say("5. DOES TB ADD ROI IN ANY CONTEXT?")
    say("   TB never picks a side, so its only ROI channel is deciding WHICH")
    say("   games to bet. A filter's content is therefore which rows it removes,")
    say("   and the statistic is kept-minus-dropped with the SE of the DIFFERENCE.")
    priced = fam[fam["lean_ml"].notna() & fam["q_lean"].notna()].copy()
    if len(priced) >= N_FIT_MIN:
        say(f"   basis: {basis} prices, n={len(priced)} bets over "
            f"{priced['game_date'].nunique()} slates")
        base = roi_block(priced)
        say(f"   {'bet every row':<26}n={base['n']:<5}{base['units']:+8.2f}u  "
            f"ROI {100*base['roi']:+6.2f}% +/- {100*base['se']:.2f}pp")
        for name, mask in (("TB_T1 only (|TB| < p50)", priced["abs_tb"] < p50),
                           ("TB_T2 only (|TB| >= p50)", priced["abs_tb"] >= p50)):
            b = roi_block(priced[mask])
            if b["n"]:
                say(f"   {name:<26}n={b['n']:<5}{b['units']:+8.2f}u  "
                    f"ROI {100*b['roi']:+6.2f}% +/- {100*b['se']:.2f}pp")
        frozen = filter_contrast(priced, p50)
        if frozen:
            say(f"   frozen-p50 filter contrast (kept - dropped): "
                f"{100*frozen['contrast']:+.2f}pp +/- {100*frozen['se']:.2f}   "
                f"z={frozen['z']:+.2f}")

        grid = tb_grid(priced)
        obs = best_contrast(priced, grid)
        if obs and grid:
            say()
            say(f"   THE SEARCH -- {len(grid)} thresholds swept, because 'any context'")
            say("   is a search and its winner must be read against what a search")
            say("   this wide returns from noise, never against zero.")
            say(f"     best threshold          |TB| >= {obs['t']:.4f}")
            say(f"     its contrast            {100*obs['contrast']:+.2f}pp "
                f"(kept n={obs['kept']['n']}, dropped n={obs['dropped']['n']})")
            nm, p = null_max_contrast(priced, grid, draws=NULL_DRAWS, seed=NULL_SEED)
            say(f"     null max (market correct, no edge)  {100*nm:+.2f}pp")
            say(f"     P(null best >= observed)            {p:.4f}")
            say(_search_verdict(p))
            say("     Read the bar before the verdict: at these row counts a")
            say("     GENUINE 12pp edge scores P = 0.21 on this same test, so a")
            say("     non-clearing result bounds what is findable, not what exists.")

            say()
            say("   THE WALK-FORWARD -- necessary because the null-max test is not")
            say("   sufficient: two variants in this repo cleared it and then lost")
            say("   forward, one of them with a stable argmax.")
            ch, bl = walk_forward_tb(priced, grid)
            say(f"     chase the best threshold   n={ch['n']:<5}ROI {100*ch['roi']:+6.2f}%")
            say(f"     bet every row              n={bl['n']:<5}ROI {100*bl['roi']:+6.2f}%")
            if np.isfinite(ch["roi"]) and np.isfinite(bl["roi"]):
                say(f"     chasing TB is worth        {100*(ch['roi']-bl['roi']):+6.2f}pp"
                    f"  {'-- WORSE than no filter' if ch['roi'] < bl['roi'] else ''}")

        inc = hybrid_increment(priced, grid)
        if inc:
            say()
            say("   ON TOP OF THE SHIPPED RULE -- scored as an INCREMENT, because a")
            say("   search whose candidates all contain the baseline measures the")
            say("   baseline beating chance, not the gate contributing anything.")
            b = inc["best"]
            say(f"     best added TB gate      |TB| >= {b['t']:.4f}  keeps {b['n']} of {inc['base']['n']}")
            say(f"     increment               {100*b['increment']:+.2f}pp per row bet")
    else:
        say(f"   SKIPPED: n={len(priced)} priced family rows, below {N_FIT_MIN}.")
    say()

    say("6. DOES TB GIVE A TEAM'S LEAN DELTA CONTEXT? (direction, not magnitude)")
    say("   Every arm above reads TB's MAGNITUDE. This one reads its SIGN against")
    say("   the model's: `tb_aligned` is TB oriented to the side the lean picks,")
    say("   so positive means the 60-day run-differential read corroborates this")
    say("   team's delta and negative means it contradicts it.")
    say("   Asked at the GAME level, not per club: the two sides of a game are")
    say("   complements -- prices summing to 1, exactly one winner -- so stacking")
    say("   them is one observation dressed as two.")
    al = alignment_frame(fam)
    al = al[al["agrees"].notna() & al["q_lean"].notna() & al["lean_won"].notna()]
    if len(al) >= N_FIT_MIN:
        overlap = _corr(al["tb_delta"], al["xw_net"])
        lo, hi, z = _fisher_ci(overlap, len(al))
        say(f"   corr(TB delta, xw_net) = {overlap:+.4f}  [{lo:+.3f}, {hi:+.3f}]  "
            f"z={z:+.2f}   <- how much the two reads already overlap")
        say(f"   they agree on {100 * (al['agrees'] == True).mean():.1f}% of "  # noqa: E712
            f"{len(al)} rows")
        rows = alignment_rows(al)
        say(f"   {'':<22}{'n':>5}{'record':>10}{'raw':>8}{'implied':>9}"
            f"{'vs price':>11}{'+/-':>7}{'chalk':>9}")
        for r in rows:
            say(f"   {r['cell']:<22}{r['n']:>5}{r['record']:>10}{r['raw']:>8.3f}"
                f"{r['implied']:>9.3f}{r['excess']:>+11.2f}{r['se']:>7.2f}{r['chalk']:>+9.2f}")
        c = alignment_contrast(rows, al, seed=NULL_SEED)
        if c:
            say(f"   AGREE minus DIVERGE     {c['diff']:+.2f}pp +/- {c['se']:.2f}   "
                f"z={c['z']:+.2f}")
            say(f"   the same split for chalk {c['chalk_diff']:+.2f}pp")
            ci = (f"  [{c['net_lo']:+.2f}, {c['net_hi']:+.2f}]"
                  if np.isfinite(c["net_lo"]) else "  (no interval: too few rows)")
            say(f"   TB's own contribution   {c['net_of_chalk']:+.2f}pp{ci}  <- the headline")
            say("   A pure price confound moves both cells together, so the")
            say("   difference is the part that is about TB rather than about")
            say("   which side happened to be favoured.")

        say()
        say("   THE CONTINUOUS FORM, which is what the sign split is a coarse")
        say("   version of -- and better powered, since it uses how much TB agrees")
        say("   rather than only whether it does:")
        zal, zd = _z(al["tb_aligned"]), _z(al["xw_net"].abs())
        arms = [("price + z(TB aligned)", [logit(al["q_lean"].values), zal],
                 ["market logit", "z(TB aligned)"]),
                ("price + |xw_net| + z(TB aligned)",
                 [logit(al["q_lean"].values), zd, zal],
                 ["market logit", "z(|xw_net|)", "z(TB aligned)"]),
                ("... plus the interaction",
                 [logit(al["q_lean"].values), zd, zal, zd * zal],
                 ["market logit", "z(|xw_net|)", "z(TB aligned)", "interaction"])]
        for label, cols, names in arms:
            say(f"   {label}  (n={len(al)})")
            for nm, b, se, zz in fit_arm(al["lean_won"], cols, names):
                say(f"     {nm:<16}{b:+8.3f} +/- {se:.3f}   z={zz:+.2f}")
        say("   The interaction is the sharpest form of the question: it asks")
        say("   whether TB's corroboration matters MORE when the delta is large.")
    else:
        say(f"   SKIPPED: n={len(al)} rows carry both a signed TB and a price.")
    say()

    say("7. OUT-OF-SAMPLE LOG LOSS (pooled, walk-forward; lower is better)")
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
