"""Freeze completed-season Savant wOBA as preseason priors.

Writes `data/woba_priors_<season>.csv` (one immutable file per completed
season) and `data/woba_prior_centres.csv` (the per-season, per-pool centres
those rates are read against).

Why this is stored rather than fetched at build time
----------------------------------------------------
It could be fetched -- Savant's custom leaderboard takes `year` and the build
already calls it daily. It is stored anyway for two reasons:

  * A completed season is a fixed fact. Re-fetching it every morning spends
    three extra leaderboard calls on a number that cannot change, and puts the
    build's ability to produce a lean at the mercy of Savant answering for
    2023. The build already exits non-zero without writing `index.html` on a
    fetch failure; a prior that predates the season has no business being able
    to trigger that.

  * A frozen file is auditable. `MODEL_TAG` stamps lineage on predictions, but
    a prior read live from a leaderboard leaves no record of what it said when
    a row was built. Savant does revise historical rates -- rarely, but it does
    -- and a committed snapshot means a ledger row can be reconciled against
    the exact prior that produced it.

Why Savant and not the StatsAPI reconstruction
----------------------------------------------
`player_prior_probe.py` rebuilds wOBA from StatsAPI counting stats, because
StatsAPI is what serves per-month splits. That is the right call for FITTING a
constant -- K = sigma^2/tau^2 is invariant to the wOBA scale -- and the wrong
one for SHIPPING a prior. `build_site` shrinks Savant's published `woba`, and a
prior has to be on the scale of the rate it is shrinking. CLAUDE.md already
records this for the relief centre: the StatsAPI reconstruction "is not even on
Savant's scale; only the gaps between pools transfer". Same hazard, same
answer.

Why the centres file exists
---------------------------
A rate from 2023 is not on 2026's scale either, and for a bigger reason than
metric choice: the run environment moves. So every season's pool centres are
frozen beside the rates, and the transferable quantity is the DEVIATION,

    pi = mu_now + sum_s v_s n_s (theta_s - mu_s) / sum_s v_s n_s

not the level. Without the centres a 2023 rate silently imports 2023's offence
into a 2026 prediction. With them the correction is arithmetic. This is the
same discipline as `relief_pool_prior` deriving its centre every build instead
of freezing one: a level is a property of its season, a deviation is a property
of the player.

Rookies and call-ups
--------------------
Deliberately absent. There is no rookie row, no imputed line, no minimum-PA
gate. A player with no history has H = 0, and

    pi = (H*theta_hist + C*mu) / (H + C)

returns `mu` -- the role population prior -- exactly. The fallback is the
n = 0 limit of the shipped expression rather than a branch beside it, which is
CLAUDE.md's threshold-cliff lesson: when a selector has a hard `>= N`, make N a
weight so nothing switches. A rookie's 71st plate appearance must not relabel
him.

Minor-league translation is NOT here and is not a small addition: it needs a
different source and a fitted translation factor per level, which is its own
probe. Until then a call-up is regressed to his role's centre, which is what
the model does today.

No-lookahead
------------
`.savant_cache/` is gitignored because a CURRENT-season leaderboard is
slate-dependent: today's pull cannot tell you what it read on 3 July. A
COMPLETED season has no such property -- 2024's final line reads the same
whenever it is pulled. That is what makes this a preseason prior rather than
backfill, and it is enforced rather than trusted: this script refuses any
season that is not strictly earlier than `build_site.SEASON`, and refuses to
overwrite a season file that already exists.

Usage:
  python priors_snapshot.py --seasons 2023,2024,2025
  python priors_snapshot.py --seasons 2025 --force     # only to repair a file
  python priors_snapshot.py --seasons 2023,2024,2025 --dry-run
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd

import build_site
from reliever_shrink_probe import fetch_season_totals

DATA_DIR = "data"
PRIORS_TEMPLATE = os.path.join(DATA_DIR, "woba_priors_{season}.csv")
CENTRES_PATH = os.path.join(DATA_DIR, "woba_prior_centres.csv")

# Top-level in data/, not data/priors/, on purpose. `validate_data_files.py`
# globs `data/*.csv` and would silently skip a subdirectory -- and that gate
# (unresolved conflict markers in committed CSVs) has failed twice in prod.
# Coverage of the gate beats tidiness of the tree.

PRIOR_COLUMNS = ["season", "player_id", "player_name", "role", "pa", "woba"]
CENTRE_COLUMNS = ["season", "pool", "n", "pa", "woba_wtd", "woba_unw"]

# The build's own call with `year` varied and the selection list trimmed to
# what a prior needs. Varying one parameter of a proven call is the same
# argument `reliever_shrink_probe` makes for its per-team fallback shapes.
SAVANT_CUSTOM = ("https://baseballsavant.mlb.com/leaderboard/custom"
                 "?year={season}&type={player_type}&min=1"
                 "&selections=pa,{rate}&csv=true")

RATE_COL = build_site.MODEL_RATE_SOURCE_COL      # "woba"; read, never copied


def fetch_leaderboard(season, player_type, tries=4):
    """One completed-season Savant custom leaderboard as a DataFrame.

    Deliberately NOT `build_site.cached_csv`: that cache is keyed by slate date
    and lives in the gitignored `.savant_cache/`, which is the current-season
    cache with the no-lookahead rule attached to it. A completed season has no
    business in there.
    """
    url = SAVANT_CUSTOM.format(season=season, player_type=player_type,
                               rate=RATE_COL)
    last = None
    for k in range(tries):
        try:
            r = build_site.session.get(url, timeout=60)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:  # noqa: BLE001
            last = e
            if k == tries - 1:
                break
            time.sleep(0.8 * (2 ** k))
    raise RuntimeError(f"Savant {player_type} leaderboard for {season} "
                       f"unavailable after {tries} tries: {last!r}")


def pitcher_roles(season, role_cut):
    """{pid: 'SP' | 'RP'} from StatsAPI starts/appearances for one season.

    Savant's leaderboard has no role column, so the split comes from the same
    starts-per-appearance rule `reliever_shrink_probe` and `build_site` use.
    A pitcher StatsAPI does not resolve keeps role 'P' at the call site rather
    than being dropped: the probe measured role separation as worth almost
    nothing (SP 2.83% vs 2.82% role-blind), so losing a rate to preserve a
    label would be a bad trade.
    """
    out = {}
    for (pid, _), rec in fetch_season_totals(season, "pitching",
                                             verbose=False).items():
        g, gs = rec.get("G", 0.0), rec.get("GS", 0.0)
        if g and g > 0:
            out[int(pid)] = "RP" if (gs / g) <= role_cut else "SP"
    return out


def _rows_from(df, season, role_of):
    rows = []
    if RATE_COL not in df.columns or "pa" not in df.columns:
        raise RuntimeError(
            f"Savant leaderboard for {season} has no '{RATE_COL}'/'pa' column "
            f"(got {list(df.columns)[:12]}). Refusing to write a prior file "
            "built on a renamed export.")
    for _, r in df.iterrows():
        try:
            pid = int(r["player_id"])
        except (TypeError, ValueError):
            continue
        pa = pd.to_numeric(r.get("pa"), errors="coerce")
        rate = pd.to_numeric(r.get(RATE_COL), errors="coerce")
        if not (pd.notna(pa) and pd.notna(rate)) or pa <= 0:
            continue
        rows.append({
            "season": int(season), "player_id": pid,
            "player_name": str(r.get("last_name, first_name")
                               or r.get("player_name") or "").strip(),
            "role": role_of(pid), "pa": float(pa), "woba": float(rate),
        })
    return rows


def build_season(season, role_cut, verbose=True):
    """(rows, centres) for one completed season."""
    bat = fetch_leaderboard(season, "batter")
    pit = fetch_leaderboard(season, "pitcher")
    roles = pitcher_roles(season, role_cut)
    rows = _rows_from(bat, season, lambda _pid: "BAT")
    rows += _rows_from(pit, season, lambda pid: roles.get(pid, "P"))
    if verbose:
        n_unres = sum(1 for r in rows if r["role"] == "P")
        print(f"  {season}: {len(rows)} players "
              f"({sum(1 for r in rows if r['role'] == 'BAT')} BAT, "
              f"{sum(1 for r in rows if r['role'] == 'SP')} SP, "
              f"{sum(1 for r in rows if r['role'] == 'RP')} RP, "
              f"{n_unres} role unresolved)", file=sys.stderr)
    return rows, season_centres(rows)


def season_centres(rows):
    """Pool centres for one season: both PA-weighted and unweighted.

    Both, because the shipped model uses both and for different reasons --
    batters and starters regress toward the league PA-weighted batter rate,
    relievers toward their pool's UNWEIGHTED centre, since empirical Bayes
    wants the centre of the population a member is drawn from and the good arms
    get the innings. Freezing only one would force the other to be
    reconstructed wrong later.
    """
    pools = {"BAT": [], "SP": [], "RP": [], "P": []}
    for r in rows:
        if r["role"] == "BAT":
            pools["BAT"].append(r)
        else:
            pools["P"].append(r)
            if r["role"] in pools:
                pools[r["role"]].append(r)
    out = []
    for pool, members in pools.items():
        if not members:
            continue
        w = np.array([m["pa"] for m in members], dtype=float)
        v = np.array([m["woba"] for m in members], dtype=float)
        out.append({
            "season": int(members[0]["season"]), "pool": pool,
            "n": int(len(members)), "pa": float(w.sum()),
            "woba_wtd": float(np.sum(v * w) / np.sum(w)),
            "woba_unw": float(np.mean(v)),
        })
    return out


def write_season(season, rows, centres, force=False, dry_run=False):
    """Write one season file and merge its centres. Never silently overwrites."""
    path = PRIORS_TEMPLATE.format(season=season)
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"FAIL: {path} already exists.\n"
            "  A completed season's prior is immutable -- a ledger row built\n"
            "  against it must stay reconcilable. Rewriting it silently would\n"
            "  change history. Pass --force only to repair a known-bad file,\n"
            "  and say so in the PR.")
    df = pd.DataFrame(rows, columns=PRIOR_COLUMNS).sort_values(
        ["role", "player_id"], kind="stable")
    if dry_run:
        print(f"  would write {path}: {len(df)} rows", file=sys.stderr)
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    merge_centres(centres)


def merge_centres(centres):
    """Upsert one season's centres into the shared centres file."""
    new = pd.DataFrame(centres, columns=CENTRE_COLUMNS)
    if os.path.exists(CENTRES_PATH):
        old = pd.read_csv(CENTRES_PATH)
        keep = old[~old["season"].isin(new["season"].unique())]
        new = pd.concat([keep, new], ignore_index=True)
    new.sort_values(["season", "pool"], kind="stable").to_csv(
        CENTRES_PATH, index=False)


# --------------------------------------------------------------------------
# reading side -- what a consumer (the probe today, the build later) calls
# --------------------------------------------------------------------------
def load_priors(data_dir=DATA_DIR):
    """({pid: {season: (woba, pa, role)}}, {(season, pool): centre dict}).

    Returns empty maps when nothing is committed yet, so a consumer degrades to
    the population centre -- which is the shipped behaviour -- rather than
    failing. That is the same posture `relief_pool_prior` takes when too few
    arms resolve.
    """
    hist, centres = {}, {}
    cpath = os.path.join(data_dir, os.path.basename(CENTRES_PATH))
    if os.path.exists(cpath):
        for _, r in pd.read_csv(cpath).iterrows():
            centres[(int(r["season"]), str(r["pool"]))] = {
                "wtd": float(r["woba_wtd"]), "unw": float(r["woba_unw"]),
                "n": int(r["n"]), "pa": float(r["pa"]),
            }
    for name in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if not (name.startswith("woba_priors_") and name.endswith(".csv")):
            continue
        df = pd.read_csv(os.path.join(data_dir, name))
        for _, r in df.iterrows():
            hist.setdefault(int(r["player_id"]), {})[int(r["season"])] = (
                float(r["woba"]), float(r["pa"]), str(r["role"]))
    return hist, centres


def centred_history(hist, centres, target_season, weights, mu_now,
                    pool_for_role=None, role=None, lookback=None):
    """(theta_hist_on_today's_scale, H) -- the deviation form, or (nan, 0).

    Each season's rate is read against ITS OWN pool centre and the deviation is
    carried onto `mu_now`. A level is a property of its run environment; a
    deviation is a property of the player. Without this a 2023 rate imports
    2023's offence into a 2026 prediction, and the drift is the same order as
    the talent spread the prior is trying to capture.
    """
    num = den = 0.0
    lookback = len(weights) if lookback is None else lookback
    for lag in range(1, lookback + 1):
        s = target_season - lag
        got = (hist or {}).get(s)
        if got is None:
            continue
        rate, n, r = got
        if role is not None and r != role:
            continue
        v = weights[lag - 1] if lag - 1 < len(weights) else 0.0
        if v <= 0 or n <= 0:
            continue
        pool = (pool_for_role or (lambda x: x))(r)
        c = (centres or {}).get((s, pool))
        if c is None:
            continue
        # RP against its unweighted centre, everything else against the
        # PA-weighted one -- mirroring which centre the build shrinks toward.
        base = c["unw"] if pool == "RP" else c["wtd"]
        num += v * n * (rate - base)
        den += v * n
    if den <= 0:
        return float("nan"), 0.0
    return mu_now + num / den, den


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", default="2023,2024,2025",
                    help="completed seasons to freeze (comma-separated)")
    ap.add_argument("--role-cut", type=float, default=0.20,
                    help="max share of appearances started to count as relief")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing season file (repair only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    bad = [s for s in seasons if s >= build_site.SEASON]
    if bad:
        raise SystemExit(
            f"FAIL: {bad} is not a completed season (current is "
            f"{build_site.SEASON}).\n"
            "  An in-progress leaderboard is a slate-dependent snapshot -- the\n"
            "  exact thing .savant_cache/ is gitignored to prevent. A prior\n"
            "  must predate the season it is applied to.")

    print(f"Freezing Savant '{RATE_COL}' priors for {seasons} ...",
          file=sys.stderr)
    for season in seasons:
        rows, centres = build_season(season, a.role_cut)
        if not rows:
            raise SystemExit(f"FAIL: no usable rows for {season}. Refusing to "
                             "write an empty prior file.")
        write_season(season, rows, centres, force=a.force, dry_run=a.dry_run)
    if not a.dry_run:
        print(f"wrote {len(seasons)} season file(s) and {CENTRES_PATH}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
