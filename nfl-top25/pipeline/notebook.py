from __future__ import annotations

"""Yahoo NFL weekly top-25 position rankings — pipeline.

GENERATED FROM the Colab notebook `Yahoo_Top25_Position_Rankings_v1.ipynb`.

The notebook executed every cell in one namespace, and its functions reach
across cell boundaries for both public helpers and underscore-prefixed ones.
This file therefore keeps all of that code in a single module in the original
order rather than splitting it: a package split would need an explicit export
list per cell and would break silently the first time a private helper moved.

Each notebook cell is marked with a banner below. Edit the configuration
section (cell 1) to change settings; `run_daily.py` is the entry point.
"""

# ============================================================================
# NOTEBOOK CELL 1 - Configuration, settings, and manual overrides
# ============================================================================
# Core imports — no PuLP or scikit-learn required.

import itertools
import math
import json
import re
import shutil
import time
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from IPython.display import display
except ImportError:
    display = print

YAHOO_API_ENDPOINT = "https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nfl"
VALID_POSITIONS = ("QB", "RB", "WR", "TE", "DEF")


@dataclass(frozen=True)
class Settings:
    lineup_size: int = 5

    # v3.2: raised from 5,000. At 5,000 the tail metrics were badly under-sampled:
    # `Win_Rate` had a cross-seed rank correlation of only 0.21 and the 20-lineup
    # portfolio changed by half its members when the seed changed. The vectorized
    # enumerator pays for the extra scenarios; total runtime is still lower than v3.1.
    simulations: int = 20_000
    random_seed: int = 356
    tournament_lineups: int = 20
    max_candidate_lineups: int = 25_000
    mean_candidate_reserve: int = 750
    max_enumeration_players: int = 36

    # This is a strategy filter, not a Yahoo rule. Lower it to allow more salary left unused.
    min_salary_used_pct: float = 0.75
    candidate_ceiling_weight: float = 0.85
    near_optimal_ratio: float = 0.95

    # Portfolio controls. On a five-player roster `max_shared_players = 4` permits
    # entries that differ by a single player; the v3.1 default produced 19 such pairs
    # out of 190 on the NE-SEA slate. Lowered to 3 so every pair of entries differs by
    # at least two players. The diversity report prints the overlap actually used.
    max_player_exposure: float = 0.70
    max_superstar_exposure: float = 0.35
    max_shared_players: int = 3

    # Historical backup-QB projections were badly biased. Keep them out unless
    # the user explicitly confirms a package or replacement-starter role.
    exclude_backup_qbs: bool = True

    # Optional user strategy limits. Yahoo itself does not require position limits.
    position_limits: dict | None = None

    # Print the split-half Monte Carlo reliability table with the run.
    report_reliability: bool = True

    # --- market-implied offensive means (v3.4) ------------------------------------
    # Pull Yahoo-scored projections from the bundled Bovada + Underdog market
    # engine. Manual PROJECTION_OVERRIDES remain highest priority. DEF has no
    # dependable player-prop projection and keeps the Yahoo/salary fallback.
    use_market_projections: bool = True
    market_source: str = "hybrid"  # "hybrid", "bovada", or "underdog"
    market_scoring: str = "yahoo"
    market_fallback_logit_vig: float = 0.17
    market_cache_dir: str = "market_projection_cache"
    market_cache_hours: float = 2.0

    # Direct component sums are accepted at good/fair coverage. TD-only estimates
    # are full-FP slate regressions, so they are allowed but clearly labeled.
    # Partial component sums and bare TD components are never used as full means.
    market_accepted_quality: tuple = ("good", "fair", "td-estimate")
    market_drop_unmatched: bool = False

    # --- nflverse role and availability feed (v3.3) ---------------------------------
    # Salary order is a weak proxy for a depth chart and says nothing at all about
    # who is on injured reserve. These pull the published depth chart and weekly
    # roster status from the nflverse data releases. Any failure degrades to the
    # salary heuristic with a warning; the run never dies on a network problem.
    use_nflverse: bool = True
    nflverse_apply_depth: bool = True
    nflverse_availability_filter: bool = True

    # Statuses treated as available. ACT is the active roster; DEV is the practice
    # squad, RES injured reserve, CUT released. Add "DEV" if you deliberately want
    # practice-squad elevation candidates in the pool.
    nflverse_available_status: tuple = ("ACT",)

    # A player nflverse has no roster row for is kept by default. The match rate is
    # high but not perfect, and dropping an unmatched star would be worse than
    # keeping an unmatched fringe player who will not be selected anyway.
    nflverse_drop_unmatched: bool = False

    nflverse_season: int | None = None  # None infers the season from the slate
    nflverse_timeout: int = 30
    nflverse_cache_dir: str = "nflverse_cache"


CFG = Settings()


def _cfg(cfg=None):
    """Resolve the live CFG at call time.

    v3.1 bound `cfg=CFG` as a def-time default. Editing this cell and re-running it
    without also re-running every function cell left the pipeline silently using the
    previous settings object. Passing `cfg=None` now reads the current global.
    """
    return CFG if cfg is None else cfg


# Exact fantasy-point overrides. Example: {"Player Name": 17.8}
PROJECTION_OVERRIDES = {}

# Team-position depth overrides. Example: {"Player Name": 1}
# Use this for a confirmed replacement starter. Depth drives BOTH the calibrated CV
# and the historical mean multiplier, so a promoted QB2 left at depth 2 keeps the
# 0.37x haircut even after you add the name to INCLUDE_BACKUP_QBS.
DEPTH_OVERRIDES = {}

# Supported styles: "rushing_qb", "pass_catching_rb", "committee_rb".
PLAYER_STYLE_OVERRIDES = {}

# Players to remove before optimization. Exact Yahoo names.
EXCLUDE_PLAYERS = set()

# Exact backup-QB names to retain despite the default role filter.
# Pair each name with DEPTH_OVERRIDES or PROJECTION_OVERRIDES; the run warns if you do not.
INCLUDE_BACKUP_QBS = set()

# Optional strategic limits, e.g. {"QB": (0, 2), "DEF": (0, 2)}.
# Leave empty to follow Yahoo's position-flexible single-game construction.
POSITION_LIMITS = {}
CFG = replace(CFG, position_limits=POSITION_LIMITS)


# ============================================================================
# NOTEBOOK CELL 3 - Yahoo player feed: fetch, normalize, priors, depth assumptions
# ============================================================================
def fetch_yahoo_data(endpoint=YAHOO_API_ENDPOINT, timeout=15, attempts=3):
    """Fetch the Yahoo player feed with bounded retries and clear errors."""
    last_error = None
    headers = {"User-Agent": "Mozilla/5.0 Yahoo-Showdown-Lineup-Lab/3.2"}
    for attempt in range(1, attempts + 1):
        try:
            request = Request(endpoint, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("players", {}).get("result"):
                raise ValueError("Yahoo response did not contain a player list.")
            return payload
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(0.75 * attempt)
    raise RuntimeError(f"Yahoo feed failed after {attempts} attempts: {last_error}")


def _repair_game_assignments(df):
    """Move rows filed under a game their team is not playing in, or drop them.

    Returns the frame with Game ID / Game Time / Home Team / Away Team corrected for
    any row whose team appears in exactly one game on the slate. See the caller for
    why this happens and why the team field is the half worth trusting.
    """
    misfiled = ~(df["Team"].eq(df["Home Team"]) | df["Team"].eq(df["Away Team"]))
    if not misfiled.any():
        return df

    columns = ["Game ID", "Game Time", "Home Team", "Away Team"]
    schedule = df.loc[~misfiled, columns].drop_duplicates("Game ID")
    lookup = pd.concat([
        schedule.assign(_team=schedule["Home Team"]),
        schedule.assign(_team=schedule["Away Team"]),
    ], ignore_index=True)
    # Only trust a team that plays exactly one game on this slate.
    counts = lookup["_team"].value_counts()
    lookup = lookup[lookup["_team"].isin(counts[counts.eq(1)].index)].set_index("_team")

    out = df.copy()
    resolvable = misfiled & out["Team"].isin(lookup.index)
    unresolved = misfiled & ~out["Team"].isin(lookup.index)

    if resolvable.any():
        targets = lookup.loc[out.loc[resolvable, "Team"]]
        moved = [
            f"{name} ({team}: {old_away}@{old_home} -> {new_away}@{new_home})"
            for name, team, old_away, old_home, new_away, new_home in zip(
                out.loc[resolvable, "Name"], out.loc[resolvable, "Team"],
                out.loc[resolvable, "Away Team"], out.loc[resolvable, "Home Team"],
                targets["Away Team"], targets["Home Team"],
            )
        ]
        out.loc[resolvable, columns] = targets[columns].to_numpy()
        warnings.warn(
            f"Reassigned {len(moved)} player(s) filed under a game their team is not "
            "playing in, using the team field and the slate schedule: "
            + ", ".join(sorted(moved))
        )

    if unresolved.any():
        names = [
            f"{name} ({team})"
            for name, team in zip(out.loc[unresolved, "Name"], out.loc[unresolved, "Team"])
        ]
        out = out.loc[~unresolved]
        warnings.warn(
            f"Dropped {len(names)} player(s) whose team plays no unambiguous game on "
            "this slate: " + ", ".join(sorted(names))
        )
    return out.reset_index(drop=True)


def normalize_yahoo_data(payload):
    players = payload["players"]["result"]
    df = pd.DataFrame(players).copy()
    rename = {
        "name": "Name",
        "position": "Position",
        "team": "Team",
        "salary": "Salary",
        "fppg": "FPPG",
        "gameCode": "Game ID",
        "gameStartTime": "Game Time",
        "homeTeam": "Home Team",
        "awayTeam": "Away Team",
    }
    df = df.rename(columns=rename)
    required = [
        "Name", "Position", "Team", "Salary", "FPPG", "Game ID",
        "Game Time", "Home Team", "Away Team",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Yahoo schema changed; missing columns: {missing}")

    df["Position"] = (
        df["Position"].astype(str).str.upper().replace({"D/ST": "DEF", "DST": "DEF"})
    )
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["FPPG"] = pd.to_numeric(df["FPPG"], errors="coerce").fillna(0.0)
    df["Game ID"] = df["Game ID"].astype(str)
    # v3.2 keeps Yahoo's own player id ("nfl.p.26753" -> 26753). Name matching against
    # any external depth-chart or injury feed is lossy; this is the stable key. Team
    # defenses carry a team code ("nfl.t.25") instead, so parse leniently.
    if "playerCode" in df:
        df["Yahoo ID"] = pd.to_numeric(
            df["playerCode"].astype(str).str.extract(r"nfl\.p\.(\d+)", expand=False),
            errors="coerce",
        ).astype("Int64")

    # v3.2: the live feed carries rows whose `gameCode` points at a game their team is
    # not playing in - for example Corey Kiner, correctly listed as NE, filed under
    # ARI@LAC. v3.1 gave them a wrong Opponent and, worse, they registered as a third
    # team, so `build_correlation_model` raised "Expected exactly two teams" and the
    # whole run died. On the 2026 week-one feed this killed 6 of 15 games.
    #
    # The team field is the reliable half: nflverse's depth chart confirms Kiner as a
    # New England running back. Since each team plays exactly one game on a slate, the
    # row can be moved to its team's real game rather than thrown away. Only a row
    # whose team is absent or ambiguous is dropped.
    df = _repair_game_assignments(df)
    df["Opponent"] = np.where(
        df["Team"].eq(df["Away Team"]), df["Home Team"], df["Away Team"]
    )

    df = df[
        df["Position"].isin(VALID_POSITIONS)
        & df["Salary"].notna()
        & df["Salary"].gt(0)
    ].copy()
    df = df.drop_duplicates(["Game ID", "Team", "Name", "Position"]).reset_index(drop=True)

    raw_caps = payload.get("salaryCapInfo", {}).get("result", [{}])
    cap_map = raw_caps[0].get("singleGameSalaryCapMap", {}) if raw_caps else {}
    cap_map = {str(key): float(value) for key, value in cap_map.items()}
    return df, cap_map


def _warn_unmatched(names, label, available):
    """Report override names that matched nothing, with the closest feed spellings.

    v3.1 warned on a miss but gave no hint, and a silently ignored projection
    override is the single most damaging user error in this notebook.
    """
    import difflib

    for name in names:
        if name in available:
            continue
        near = difflib.get_close_matches(str(name), list(available), n=3, cutoff=0.6)
        hint = f" Closest feed names: {', '.join(near)}." if near else ""
        warnings.warn(f"{label} did not match any Yahoo name: {name}.{hint}")


def add_projection_priors(df, projection_overrides=None):
    """
    Blend Yahoo FPPG with a regularized position/salary prior.

    This deliberately replaces the prior degree-3 regression, which could overfit a
    small current slate and assign strong projections to zero-history players.
    """
    projection_overrides = projection_overrides or {}
    out = df.copy()
    out["Salary_Prior"] = np.nan

    for position, group in out.groupby("Position"):
        train = group[np.isfinite(group["FPPG"]) & group["FPPG"].gt(0.25)]
        if len(train) >= 5 and train["Salary"].nunique() >= 3:
            x = train["Salary"].to_numpy(float)
            y = train["FPPG"].to_numpy(float)
            x_center = x - x.mean()
            # Ridge-like denominator stabilizes thin positions and clips implausible slopes.
            slope = float(np.dot(x_center, y - y.mean()) / (np.dot(x_center, x_center) + 25.0))
            slope = float(np.clip(slope, 0.05, 1.25))
            prior = y.mean() + slope * (group["Salary"].to_numpy(float) - x.mean())
        else:
            ratio = np.median(train["FPPG"] / train["Salary"]) if len(train) else 0.45
            ratio = float(np.clip(ratio, 0.15, 0.90))
            prior = group["Salary"].to_numpy(float) * ratio
        out.loc[group.index, "Salary_Prior"] = np.maximum(prior, 0.25)

    has_history = out["FPPG"].gt(0.25)
    out["Projected_FP"] = np.where(
        has_history,
        0.70 * out["FPPG"] + 0.30 * out["Salary_Prior"],
        0.80 * out["Salary_Prior"],
    )
    out["Projection_Source"] = np.where(
        has_history, "70% FPPG + 30% salary prior", "80% salary prior; zero/low FPPG"
    )

    _warn_unmatched(projection_overrides, "Projection override", set(out["Name"]))
    for name, projection in projection_overrides.items():
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Projected_FP"] = float(projection)
            out.loc[mask, "Projection_Source"] = "manual override"

    out["Projected_FP"] = out["Projected_FP"].clip(lower=0.05)
    return out


def list_games(df):
    games = (
        df[["Game ID", "Game Time", "Away Team", "Home Team"]]
        .drop_duplicates("Game ID")
        .copy()
    )
    games["_sort"] = pd.to_datetime(games["Game Time"], errors="coerce", utc=True)
    games = games.sort_values(["_sort", "Game ID"], na_position="last").drop(columns="_sort")
    games["Matchup"] = games["Away Team"] + " vs " + games["Home Team"]
    return games.reset_index(drop=True)


def select_game_interactive(games):
    print("Available games:")
    for number, row in games.iterrows():
        print(f"{number + 1}. {row['Game Time']}: {row['Matchup']}")
    while True:
        try:
            choice = int(input("Select game number: ")) - 1
            if 0 <= choice < len(games):
                return games.iloc[choice]
        except ValueError:
            pass
        print("Enter one of the listed game numbers.")


def salary_cap_for_game(cap_map, game_id):
    game_id = str(game_id)
    if game_id in cap_map:
        return float(cap_map[game_id])
    while True:
        try:
            value = float(input("Yahoo cap was unavailable. Enter the single-game salary cap: $"))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive number.")


def assign_depth_assumptions(players, depth_overrides=None, style_overrides=None):
    """Assign relative team-position ranks without claiming active status.

    The Yahoo feed does not provide a dependable live depth chart. Ranking by the
    unadjusted projection gives a reproducible fallback and avoids circularly
    re-ranking players after their depth haircut. A manual override should be used
    for injury replacements, newly promoted starters, and specialty packages.

    v3.2 breaks projection ties on salary then name instead of on feed row order, so
    two equally projected bench players always receive the same depth ranks.
    """
    depth_overrides = depth_overrides or {}
    style_overrides = style_overrides or {}
    out = players.copy()
    order = out.sort_values(
        ["Team", "Position", "Projected_FP", "Salary", "Name"],
        ascending=[True, True, False, False, True],
    )
    ranks = (order.groupby(["Team", "Position"]).cumcount() + 1).reindex(out.index)
    out["Depth_Rank"] = ranks.astype(int)
    out["Depth_Source"] = "projection heuristic"
    out["Player_Style"] = "standard"

    _warn_unmatched(depth_overrides, "Depth override", set(out["Name"]))
    for name, depth in depth_overrides.items():
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Depth_Rank"] = max(1, int(depth))
            out.loc[mask, "Depth_Source"] = "manual override"

    allowed_styles = {"standard", "rushing_qb", "pass_catching_rb", "committee_rb"}
    _warn_unmatched(style_overrides, "Style override", set(out["Name"]))
    for name, style in style_overrides.items():
        if style not in allowed_styles:
            raise ValueError(f"Unsupported style '{style}' for {name}")
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Player_Style"] = style
    return out


# Actual / rolling-pregame expectation by position and lagged-snap depth.
# Rank-one values are held at 1.0 because the small observed differences were not
# practically important. QB3 had only 23 games and is not promoted as a parameter.
DEPTH_MEAN_MULTIPLIER = {
    "QB": {1: 1.00, 2: 0.37, 3: 0.37, 4: 0.37},
    "RB": {1: 1.00, 2: 0.94, 3: 0.75, 4: 0.70},
    "WR": {1: 1.00, 2: 0.98, 3: 0.98, 4: 0.77},
    "TE": {1: 1.00, 2: 0.94, 3: 0.75, 4: 0.61},
    "DEF": {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00},
}


def _depth_bucket(depth):
    """Map all fourth-or-deeper roles to the calibrated 4+ bucket."""
    return min(max(int(depth), 1), 4)


def apply_depth_mean_adjustments(players):
    """Separate systematic depth bias from random game-level volatility.

    The calibration's deep-player forecast errors contained both dispersion and
    predictable mean overstatement. A mean-preserving lognormal cannot correct an
    inflated projection; increasing its CV only creates misleading cheap-player
    ceilings. Therefore non-manual projections receive the observed depth mean
    ratio before simulation. Exact manual projections and current market means
    remain untouched because they already contain current role information.
    """
    out = players.copy()
    out["Pre_Depth_Projected_FP"] = out["Projected_FP"].astype(float)
    out["Depth_Mean_Multiplier"] = [
        DEPTH_MEAN_MULTIPLIER[row.Position].get(_depth_bucket(row.Depth_Rank), 1.0)
        for row in out.itertuples()
    ]
    # Current market means already encode role through priced components.
    # Applying the historical depth haircut again would double-count role.
    # Depth still controls the calibrated CV and pair correlations.
    authoritative = (
        out["Projection_Source"].eq("manual override")
        | out["Projection_Source"].astype(str).str.startswith("market ")
    )
    out.loc[authoritative, "Depth_Mean_Multiplier"] = 1.0
    out["Projected_FP"] = (
        out["Pre_Depth_Projected_FP"] * out["Depth_Mean_Multiplier"]
    ).clip(lower=0.05)
    out["Projection_Adjustment"] = np.where(
        authoritative,
        np.where(
            out["Projection_Source"].eq("manual override"),
            "manual projection retained",
            "market mean retained",
        ),
        np.where(
            out["Depth_Mean_Multiplier"].lt(0.999),
            "historical depth mean adjustment",
            "none",
        ),
    )
    return out


def apply_default_role_filters(players, include_backup_qbs=None, cfg=None):
    """Remove unconfirmed backup QBs while allowing explicit named exceptions.

    QB2 averaged 3.93 points against a 10.52-point pregame expectation in the
    historical fit. That is primarily a participation problem, not useful upside.
    Default exclusion prevents a low-salary backup from entering a lineup solely
    because a high fitted CV produces a long simulated tail.
    """
    cfg = _cfg(cfg)
    include_backup_qbs = set(include_backup_qbs or set())
    out = players.copy()

    # v3.2: keeping a backup QB without also promoting its depth leaves the 0.37x
    # historical multiplier in place, silently deleting ~63% of its projection. That
    # made INCLUDE_BACKUP_QBS look broken rather than misconfigured.
    demoted = out[
        out["Name"].isin(include_backup_qbs)
        & out["Position"].eq("QB")
        & out["Depth_Rank"].gt(1)
        & ~out["Projection_Source"].eq("manual override")
    ]["Name"].tolist()
    if demoted:
        warnings.warn(
            "INCLUDE_BACKUP_QBS retained " + ", ".join(demoted)
            + " but they still rank below QB1, so the 0.37x historical QB2 mean "
            "multiplier is still applied. Add a DEPTH_OVERRIDES entry of 1, or a "
            "PROJECTION_OVERRIDES value, if you expect them to start."
        )

    if not cfg.exclude_backup_qbs:
        return out.reset_index(drop=True), []
    remove = (
        out["Position"].eq("QB")
        & out["Depth_Rank"].gt(1)
        & ~out["Name"].isin(include_backup_qbs)
    )
    removed = out.loc[remove, "Name"].tolist()
    return out.loc[~remove].reset_index(drop=True), removed


def depth_sanity_report(players):
    """Present every role assumption and its direct projection consequence."""
    rows = []
    for _, player in players.sort_values(["Team", "Position", "Depth_Rank"]).iterrows():
        flags = []
        if player["Depth_Source"] != "manual override":
            flags.append("heuristic depth")
        if (
            player["FPPG"] <= 0.25
            and player["Projection_Source"] != "manual override"
            and not str(player["Projection_Source"]).startswith("market ")
        ):
            flags.append("zero/low FPPG prior")
        if player["Position"] == "QB" and player["Depth_Rank"] > 1:
            flags.append("verify expected snaps")
        rows.append({
            "Team": player["Team"],
            "Position": player["Position"],
            "Depth": int(player["Depth_Rank"]),
            "Player": player["Name"],
            "Salary": float(player["Salary"]),
            "Raw projection": round(float(player["Pre_Depth_Projected_FP"]), 2),
            "Mean factor": round(float(player["Depth_Mean_Multiplier"]), 2),
            "Adjusted projection": round(float(player["Projected_FP"]), 2),
            "Depth source": player["Depth_Source"],
            "Review": "; ".join(flags) or "ok",
        })
    return pd.DataFrame(rows)


def apply_exclusions_interactive(players, preexcluded=None):
    """Display stable row numbers and remove explicitly selected players."""
    preexcluded = set(preexcluded or set())
    _warn_unmatched(preexcluded, "Exclusion", set(players["Name"]))
    out = players[~players["Name"].isin(preexcluded)].copy()
    view = out.sort_values(["Team", "Position", "Depth_Rank", "Salary"], ascending=[True, True, True, False])
    view = view.reset_index(drop=True)
    view.insert(0, "Row", np.arange(len(view)))
    display(view[[
        "Row", "Name", "Position", "Team", "Salary", "FPPG",
        "Pre_Depth_Projected_FP", "Depth_Mean_Multiplier", "Projected_FP",
        "Depth_Rank", "Projection_Source",
    ]].rename(columns={
        "Pre_Depth_Projected_FP": "Raw projection",
        "Depth_Mean_Multiplier": "Mean factor",
        "Projected_FP": "Adjusted projection",
        "Depth_Rank": "Depth",
    }))
    answer = input("Exclude more players? Enter comma-separated Row values, or press Enter: ").strip()
    if not answer:
        return out.reset_index(drop=True)
    try:
        numbers = [int(piece.strip()) for piece in answer.split(",")]
        bad = [number for number in numbers if number < 0 or number >= len(view)]
        if bad:
            raise ValueError(f"row numbers out of range: {bad}")
        names = set(view.iloc[numbers]["Name"])
        return out[~out["Name"].isin(names)].reset_index(drop=True)
    except ValueError as exc:
        raise ValueError(f"Invalid exclusion list: {exc}") from exc


def roster_feasibility_error(players, lineup_size):
    """Return why this pool cannot build a Yahoo-valid roster, or None if it can.

    Yahoo requires five players and at least one non-DEF athlete from each team.
    Checking that up front turns a confusing "no valid rosters" failure deep inside
    enumeration into a specific message naming the team that lost its skill players.
    """
    if len(players) < lineup_size:
        return f"Only {len(players)} players remain; {lineup_size} are required."
    teams = sorted(players["Team"].unique())
    if len(teams) != 2:
        return f"Expected exactly two teams, found {teams}."
    for team in teams:
        if players[players["Team"].eq(team) & ~players["Position"].eq("DEF")].empty:
            return (
                f"{team} has no non-DEF player left. Yahoo requires at least one skill "
                "player from each team, so no valid lineup exists."
            )
    return None


def trim_player_pool(players, cfg=None):
    """Fill the configured pool using projection, value, and balanced backfill.

    The first implementation concatenated 26 projection leaders and 10 value leaders.
    When those lists overlapped heavily, deduplication could leave only 26 players
    even though `max_enumeration_players` was 36. The missing slots are now filled by
    a 70/30 projection/value percentile score.

    v3.2 also protects roster feasibility. A pool whose leaders all belonged to one
    team could strand the other team with only its DEF; the caller's two-team guard
    still passed and enumeration then failed with an unrelated message. Each team's
    best non-DEF player is now reserved before ranking.
    """
    cfg = _cfg(cfg)
    if len(players) <= cfg.max_enumeration_players:
        return players.reset_index(drop=True), []
    work = players.copy()
    work["Value"] = work["Projected_FP"] / work["Salary"]
    target = max(1, int(cfg.max_enumeration_players))

    reserved = []
    if {"Team", "Position"}.issubset(work.columns):
        for team in work["Team"].drop_duplicates():
            skill = work[work["Team"].eq(team) & ~work["Position"].eq("DEF")]
            if len(skill):
                reserved.append(skill["Projected_FP"].idxmax())
    reserved = list(dict.fromkeys(reserved))[:target]

    value_count = min(10, max(target - 1, 0))
    anchor_count = target - value_count
    keep = list(reserved)
    keep += [i for i in work.nlargest(anchor_count, "Projected_FP").index if i not in keep]
    keep += [i for i in work.nlargest(value_count, "Value").index if i not in keep]
    keep = list(dict.fromkeys(keep))

    if len(keep) < target:
        # Percentile ranks are scale-free, so a point projection and a
        # points-per-dollar value can be combined without arbitrary units.
        work["Projection_Percentile"] = work["Projected_FP"].rank(pct=True)
        work["Value_Percentile"] = work["Value"].rank(pct=True)
        work["Pool_Priority"] = (
            0.70 * work["Projection_Percentile"]
            + 0.30 * work["Value_Percentile"]
        )
        needed = target - len(keep)
        backfill = (
            work.loc[~work.index.isin(keep)]
            .sort_values(["Pool_Priority", "Projected_FP", "Value"], ascending=False)
            .head(needed)
            .index
            .tolist()
        )
        keep.extend(backfill)
    keep = keep[:target]

    dropped = work.loc[~work.index.isin(keep), "Name"].tolist()
    helper_columns = ["Value", "Projection_Percentile", "Value_Percentile", "Pool_Priority"]
    return (
        work.loc[keep]
        .drop(columns=helper_columns, errors="ignore")
        .reset_index(drop=True),
        dropped,
    )


# ============================================================================
# NOTEBOOK CELL 5 - Market-implied projection engine (Bovada + Underdog props)
# ============================================================================
#!/usr/bin/env python3
"""Market-implied NFL fantasy projections from Underdog and Bovada props.

Why this version differs from the original:

* Alternate lines are survival probabilities (P[stat >= threshold]).  They
  cannot be normalized and averaged as if they were probability masses.
* Underdog supplies broad, paired main lines for the full slate. Bovada adds
  alternate ladders that help estimate the shape of each stat distribution.
* Two-way main props from either feed are de-vigged and used as anchors.
* One-way alternate prices are adjusted in log-odds space.  The adjustment is
  learned from main/alternate pairs on the same slate when possible.
* Yardage is fit with a non-negative Weibull survival curve.  Receptions and
  touchdown counts are fit with a Poisson survival curve.
* Players are merged across feeds only when normalized name, team, and game
  time agree; different games can never be silently combined.
* Passing interceptions and lost fumbles are included when markets exist.
* Touchdown-only players receive a slate-trained fantasy-point estimate.  The
  regression uses only good, non-quarterback projections with touchdown props
  as its targets and is labeled separately from component-based projections.

This is a market-derived estimate, not a predictive guarantee. Both feeds are
public but undocumented and may change. Make one request to each feed per run
and cache the payloads with the input flags if you are iterating.

The offensive projection omits components without a dependable prop market,
including two-point conversions and return statistics. It does not project
kickers or D/ST.
"""


import argparse
import base64
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BOVADA_ENDPOINT = (
    "https://www.bovada.lv/services/sports/event/v2/events/A/"
    "description/football/nfl"
)
UNDERDOG_ENDPOINTS = (
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
)
# Backwards-compatible singular name now points to the current endpoint.
UNDERDOG_ENDPOINT = UNDERDOG_ENDPOINTS[0]

# Backwards-compatible name for notebooks that imported ENDPOINT.
ENDPOINT = BOVADA_ENDPOINT

USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/15.0 Mobile/15E148 Safari/604.1"
)

UNDERDOG_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

EPSILON = 1e-6
DEFAULT_FALLBACK_LOGIT_VIG = 0.17
MIN_GLOBAL_TD_REGRESSION_SAMPLES = 12
MIN_POSITION_TD_REGRESSION_SAMPLES = 8
TD_REGRESSION_RESIDUAL_MAD_LIMIT = 3.5

YARD_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}
COUNT_STATS = {
    "receptions",
    "passing_touchdowns",
    "any_touchdowns",
    "interceptions",
    "fumbles_lost",
}

STAT_ALIASES = {
    "Passing Yards": "passing_yards",
    "Rushing Yards": "rushing_yards",
    "Receiving Yards": "receiving_yards",
    "Receptions": "receptions",
    "Passing Touchdowns": "passing_touchdowns",
    "Interceptions Thrown": "interceptions",
    "Passing Interceptions": "interceptions",
}

DEFAULT_WEIBULL_SHAPES = {
    "passing_yards": 4.0,
    "rushing_yards": 2.0,
    "receiving_yards": 2.0,
}

STAT_LABELS = {
    "passing_yards": "PaY",
    "rushing_yards": "RuY",
    "receiving_yards": "ReY",
    "receptions": "Rec",
    "passing_touchdowns": "PaTD",
    "any_touchdowns": "TD",
    "interceptions": "INT",
    "fumbles_lost": "FUM",
}

UNDERDOG_STAT_ALIASES = {
    "passing_yds": "passing_yards",
    "rushing_yds": "rushing_yards",
    "receiving_yds": "receiving_yards",
    "receiving_rec": "receptions",
    "passing_tds": "passing_touchdowns",
    "passing_ints": "interceptions",
    "rush_rec_tds": "any_touchdowns",
    "fumbles_lost": "fumbles_lost",
}

TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "NOR": "NO",
    "WSH": "WAS",
}

NFL_TEAM_NAMES = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


@dataclass(frozen=True)
class Scoring:
    passing_yard: float
    rushing_yard: float
    receiving_yard: float
    reception: float
    passing_touchdown: float
    rushing_receiving_touchdown: float
    interception: float
    fumble_lost: float


SCORING_PRESETS = {
    # Current Yahoo default offensive categories: half-PPR, 1/25 passing,
    # 1/10 rushing/receiving, 4-point passing TD, -1 per interception, and
    # -2 per lost fumble.
    "yahoo": Scoring(0.04, 0.10, 0.10, 0.50, 4.0, 6.0, -1.0, -2.0),
    "half-ppr": Scoring(0.04, 0.10, 0.10, 0.50, 4.0, 6.0, -2.0, -2.0),
    "ppr": Scoring(0.04, 0.10, 0.10, 1.00, 4.0, 6.0, -2.0, -2.0),
    # Preserves the coefficients and behavior of the supplied script.  Its
    # declared interceptions variable was unused, so the penalty remains 0.
    "original": Scoring(0.06, 0.125, 0.125, 1.00, 4.0, 6.0, 0.0, 0.0),
}


@dataclass
class Observation:
    threshold: float
    probability: float
    weight: float
    source: str


@dataclass
class StatMarket:
    alternate: Dict[float, List[float]] = field(default_factory=dict)
    anchors: List[Observation] = field(default_factory=list)
    market_ids: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)

    def add_alternate(
        self,
        threshold: float,
        probability: float,
        market_id: str,
        provider: str = "bovada",
    ) -> None:
        self.alternate.setdefault(float(threshold), []).append(probability)
        self.providers.add(provider)
        if market_id:
            self.market_ids.add(f"{provider}:{market_id}")

    def add_anchor(
        self,
        threshold: float,
        probability: float,
        market_id: str,
        provider: str = "bovada",
    ) -> None:
        self.anchors.append(
            Observation(float(threshold), probability, 4.0, f"{provider} total")
        )
        self.providers.add(provider)
        if market_id:
            self.market_ids.add(f"{provider}:{market_id}")


@dataclass
class PlayerMarkets:
    event_id: str
    name: str
    team: str
    matchup: str
    start_time_ms: Optional[int]
    position: str = "UNK"
    stats: Dict[str, StatMarket] = field(default_factory=dict)

    def market(self, stat: str) -> StatMarket:
        return self.stats.setdefault(stat, StatMarket())


@dataclass
class Distribution:
    family: str
    mean: float
    parameter_1: float
    parameter_2: Optional[float] = None

    def survival(self, threshold: float) -> float:
        if self.family == "weibull":
            shape = self.parameter_1
            scale = self.parameter_2 or EPSILON
            if threshold <= 0:
                return 1.0
            return math.exp(-((threshold / scale) ** shape))
        if self.family == "poisson":
            return poisson_survival(max(1, int(math.ceil(threshold))), self.parameter_1)
        raise ValueError(f"Unknown distribution family: {self.family}")


@dataclass
class Projection:
    event_id: str
    matchup: str
    start_time_utc: Optional[str]
    team: str
    player: str
    position: str
    fantasy_points: float
    quality: str
    stat_means: Dict[str, float]
    sources: Dict[str, str]
    fantasy_points_method: str = "component-sum"


@dataclass(frozen=True)
class TdRegression:
    """Linear expected-TD to fantasy-point fit learned from this slate."""

    label: str
    intercept: float
    slope: float
    sample_count: int
    r_squared: float
    maximum_target: float

    def predict(self, expected_touchdowns: float, touchdown_points: float) -> float:
        estimate = self.intercept + self.slope * max(0.0, expected_touchdowns)
        # A complete estimate should not fall below its known touchdown scoring
        # component.  Cap extreme extrapolation just beyond the strongest good
        # projection observed on the same slate.
        floor = max(0.0, expected_touchdowns * touchdown_points)
        ceiling = max(floor, self.maximum_target * 1.10)
        return min(ceiling, max(floor, estimate))


def clamp_probability(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, float(value)))


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def american_odds_to_probability(american_odds: Any) -> Optional[float]:
    """Convert American odds without prematurely rounding the probability."""

    if american_odds is None:
        return None
    text = str(american_odds).strip().upper().replace("\u2212", "-")
    if text in {"EVEN", "EVENS", "EV", "EVS"}:
        return 0.5
    text = text.replace(",", "")
    try:
        odds = float(text)
    except (TypeError, ValueError):
        return None
    if odds == 0 or not math.isfinite(odds):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_probability(price: Any) -> Optional[float]:
    """Convert a price mapping, preferring the usually finer American quote."""

    if not isinstance(price, Mapping):
        return None
    probability = american_odds_to_probability(price.get("american"))
    if probability is not None:
        return clamp_probability(probability)
    decimal = price.get("decimal")
    if decimal is not None:
        try:
            decimal_value = float(str(decimal).replace(",", ""))
            if decimal_value > 1.0 and math.isfinite(decimal_value):
                return clamp_probability(1.0 / decimal_value)
        except (TypeError, ValueError):
            pass
    return None


def no_vig_two_way(over_probability: float, under_probability: float) -> float:
    denominator = over_probability + under_probability
    if denominator <= 0:
        raise ValueError("Two-way market has no valid probability mass")
    return clamp_probability(over_probability / denominator)


def fetch_json_payload(
    url: str,
    feed_name: str,
    validator: Callable[[Any], None],
    timeout: float = 30.0,
    retries: int = 3,
    extra_headers: Optional[Mapping[str, str]] = None,
) -> Any:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if extra_headers:
        request_headers.update(extra_headers)
    request = Request(
        url,
        headers=request_headers,
    )

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            validator(payload)
            return payload
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            # Retrying an authorization/WAF rejection against the identical
            # URL is not useful. Let the caller try its next endpoint version.
            if isinstance(exc, HTTPError) and exc.code in {401, 403, 404}:
                break
            if attempt < retries:
                time.sleep(0.75 * attempt)

    raise RuntimeError(f"Unable to load a valid {feed_name} payload: {last_error}")


def fetch_bovada_payload(timeout: float = 30.0, retries: int = 3) -> Any:
    params = urlencode(
        {"preMatchOnly": "true", "eventsLimit": "5000", "lang": "en"}
    )
    return fetch_json_payload(
        f"{BOVADA_ENDPOINT}?{params}",
        "Bovada NFL",
        validate_bovada_payload,
        timeout,
        retries,
    )


def fetch_underdog_via_colab_browser(endpoint: str, timeout: float) -> Any:
    """Fetch through the user's browser when Underdog blocks Colab's VM IP.

    Underdog explicitly permits cross-origin GETs. The payload is transferred
    to Python in bounded base64 chunks so Colab's message bridge never has to
    carry the full response in one reply.
    """

    try:
        from google.colab import output as colab_output
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Colab browser bridge is unavailable") from exc

    endpoint_literal = json.dumps(endpoint)
    fetch_script = f"""
        (async () => {{
          const response = await fetch({endpoint_literal}, {{
            method: 'GET',
            mode: 'cors',
            cache: 'no-store',
            headers: {{'Accept': 'application/json'}}
          }});
          if (!response.ok) {{
            throw new Error('Underdog HTTP ' + response.status);
          }}
          const buffer = await response.arrayBuffer();
          globalThis.__nflUnderdogPayloadBytes = new Uint8Array(buffer);
          return globalThis.__nflUnderdogPayloadBytes.length;
        }})()
    """

    try:
        byte_count = int(
            colab_output.eval_js(
                fetch_script,
                timeout_sec=max(60, int(math.ceil(timeout)) + 15),
            )
        )
        if byte_count <= 2:
            raise ValueError("browser returned an empty Underdog response")

        raw = bytearray()
        chunk_size = 512 * 1024
        for start in range(0, byte_count, chunk_size):
            end = min(start + chunk_size, byte_count)
            chunk_script = f"""
                (() => {{
                  const bytes = globalThis.__nflUnderdogPayloadBytes.subarray(
                    {start}, {end}
                  );
                  let binary = '';
                  const blockSize = 32768;
                  for (let i = 0; i < bytes.length; i += blockSize) {{
                    binary += String.fromCharCode(
                      ...bytes.subarray(i, Math.min(i + blockSize, bytes.length))
                    );
                  }}
                  return btoa(binary);
                }})()
            """
            encoded = colab_output.eval_js(chunk_script, timeout_sec=30)
            if not isinstance(encoded, str):
                raise ValueError("browser returned a non-text payload chunk")
            raw.extend(base64.b64decode(encoded, validate=True))

        if len(raw) != byte_count:
            raise ValueError(
                f"browser transfer was incomplete ({len(raw)} of {byte_count} bytes)"
            )
        payload = json.loads(bytes(raw).decode("utf-8"))
        validate_underdog_payload(payload)
        return payload
    except Exception as exc:
        raise RuntimeError(f"Colab browser fetch failed: {exc}") from exc
    finally:
        try:
            colab_output.eval_js(
                "delete globalThis.__nflUnderdogPayloadBytes",
                ignore_result=True,
                timeout_sec=5,
            )
        except Exception:
            pass


def fetch_underdog_payload(timeout: float = 45.0, retries: int = 3) -> Any:
    browser_headers = {
        "User-Agent": UNDERDOG_USER_AGENT,
        "Origin": "https://underdogfantasy.com",
        "Referer": "https://underdogfantasy.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
    failures: List[str] = []
    for endpoint in UNDERDOG_ENDPOINTS:
        version_match = re.search(r"/beta/(v\d+)/", endpoint)
        version = version_match.group(1) if version_match else endpoint
        try:
            return fetch_json_payload(
                endpoint,
                f"Underdog {version}",
                validate_underdog_payload,
                timeout,
                retries,
                browser_headers,
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    if running_in_google_colab():
        print(
            "Direct Underdog request failed; retrying through the Colab "
            "browser (this can take 20-60 seconds)...",
            file=sys.stderr,
            flush=True,
        )
        for endpoint in UNDERDOG_ENDPOINTS:
            try:
                return fetch_underdog_via_colab_browser(endpoint, timeout)
            except RuntimeError as exc:
                failures.append(str(exc))

    if failures and all("HTTP Error 403" in failure for failure in failures):
        raise RuntimeError(
            "all public endpoint versions returned HTTP 403; the provider's "
            "CDN may be rejecting this hosted runtime. Supply a browser-saved "
            "payload with --underdog-input, or run the script from a local IP."
        )
    raise RuntimeError("; ".join(failures))


# Backwards-compatible Bovada fetch helper.
def fetch_payload(timeout: float = 30.0, retries: int = 3) -> Any:
    return fetch_bovada_payload(timeout, retries)


def validate_bovada_payload(payload: Any) -> None:
    wrappers: List[Any]
    if isinstance(payload, list):
        wrappers = payload
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        wrappers = [payload]
    else:
        raise ValueError(
            "Bovada returned an empty or unexpected response. The bare endpoint "
            "currently returns {}; keep the required query parameters enabled."
        )
    if not wrappers or not any(
        wrapper.get("events")
        for wrapper in wrappers
        if isinstance(wrapper, dict)
    ):
        raise ValueError("Bovada payload contains no NFL events")


def validate_underdog_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Underdog returned an unexpected non-object response")
    required_lists = ("players", "appearances", "games", "over_under_lines")
    missing = [key for key in required_lists if not isinstance(payload.get(key), list)]
    if missing:
        raise ValueError(
            "Underdog payload is missing expected lists: " + ", ".join(missing)
        )
    if not payload.get("over_under_lines"):
        raise ValueError("Underdog payload contains no over/under lines")


# Backwards-compatible Bovada validator.
def validate_payload(payload: Any) -> None:
    validate_bovada_payload(payload)


def iter_bovada_events(payload: Any) -> Iterable[Mapping[str, Any]]:
    validate_bovada_payload(payload)
    wrappers = payload if isinstance(payload, list) else [payload]
    for wrapper in wrappers:
        if not isinstance(wrapper, Mapping):
            continue
        for event in wrapper.get("events", []):
            if isinstance(event, Mapping):
                yield event


# Backwards-compatible Bovada iterator.
def iter_events(payload: Any) -> Iterable[Mapping[str, Any]]:
    yield from iter_bovada_events(payload)


def is_open_full_game_market(market: Mapping[str, Any]) -> bool:
    if str(market.get("status", "O")).upper() != "O":
        return False

    period = market.get("period")
    if isinstance(period, Mapping):
        abbreviation = str(period.get("abbreviation") or "").upper()
        description = str(period.get("description") or "").casefold()
        return bool(period.get("main")) or abbreviation == "G" or description == "game"

    # Fallback only for a future feed variant without structured period data.
    description = str(market.get("description") or "")
    partial_period = (
        r"(?:^|\W)(?:[1-4](?:ST|ND|RD|TH)?\s+QUARTER|Q[1-4]|[12]H|HALF)"
        r"(?:\W|$)"
    )
    return re.search(partial_period, description, re.I) is None


def is_open_outcome(outcome: Mapping[str, Any]) -> bool:
    return str(outcome.get("status", "O")).upper() == "O"


def split_player_team(label: Any) -> Tuple[str, str]:
    text = " ".join(str(label or "").split())
    match = re.match(r"^(.*?)\s*\(([A-Za-z]{2,4})\)\s*$", text)
    if match:
        return match.group(1).strip(), canonical_team(match.group(2))
    return text, "UNK"


def is_defense_name(name: str) -> bool:
    compact = " ".join(str(name or "").casefold().split())
    return bool(re.search(r"(?:\bd/st|\bdef/st|\bdefense)\s*$", compact))


def bovada_event_team_codes(event: Mapping[str, Any]) -> set[str]:
    """Resolve the two current teams from Bovada's event competitors."""

    teams: set[str] = set()
    competitors = event.get("competitors")
    if isinstance(competitors, list):
        for competitor in competitors:
            if not isinstance(competitor, Mapping):
                continue
            name = " ".join(str(competitor.get("name") or "").casefold().split())
            if name in NFL_TEAM_NAMES:
                teams.add(NFL_TEAM_NAMES[name])
    return teams


def normalized_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name).casefold()
    # Providers are inconsistent about generational suffixes (for example,
    # "Deebo Samuel Sr." versus "Deebo Samuel"). The game/team guard still
    # prevents this relaxed name key from merging unrelated players.
    decomposed = re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\.?\s*$", "", decomposed)
    return re.sub(r"[^a-z0-9]+", "", decomposed)


def canonical_team(team: Any) -> str:
    code = re.sub(r"[^A-Za-z]", "", str(team or "")).upper() or "UNK"
    return TEAM_ALIASES.get(code, code)


def player_identity(event_id: str, name: str, team: str) -> Tuple[str, str, str]:
    return str(event_id), normalized_name(name), canonical_team(team)


def parse_iso_time_ms(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (OverflowError, TypeError, ValueError):
        return None


def threshold_from_outcome(description: Any) -> Optional[float]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*\+", str(description or ""))
    return float(match.group(1)) if match else None


def outcome_line(outcome: Mapping[str, Any]) -> Optional[float]:
    price = outcome.get("price")
    if isinstance(price, Mapping) and price.get("handicap") is not None:
        try:
            return float(str(price.get("handicap")).replace(",", ""))
        except (TypeError, ValueError):
            pass
    match = re.search(r"-?\d+(?:\.\d+)?", str(outcome.get("description") or ""))
    return float(match.group(0)) if match else None


def parse_two_way_market(
    outcomes: Any,
) -> Optional[Tuple[float, float]]:
    """Return (integer threshold, fair over probability)."""

    if not isinstance(outcomes, list):
        return None

    by_line: Dict[float, Dict[str, float]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
            continue
        description = str(outcome.get("description") or "").strip().casefold()
        if description.startswith("over"):
            side = "over"
        elif description.startswith("under"):
            side = "under"
        else:
            continue
        line = outcome_line(outcome)
        probability = implied_probability(outcome.get("price"))
        if line is not None and probability is not None:
            by_line.setdefault(line, {})[side] = probability

    paired = [
        (line, sides)
        for line, sides in by_line.items()
        if "over" in sides and "under" in sides
    ]
    if not paired:
        return None

    # Player props normally contain one line. If the feed supplies several,
    # choose the most balanced pair, which is usually the main line.
    line, sides = min(
        paired,
        key=lambda item: abs(
            no_vig_two_way(item[1]["over"], item[1]["under"]) - 0.5
        ),
    )
    threshold = float(math.floor(line) + 1)
    return threshold, no_vig_two_way(sides["over"], sides["under"])


def parse_bovada_payload(payload: Any) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    players: Dict[Tuple[str, str, str], PlayerMarkets] = {}

    def get_player(event: Mapping[str, Any], label: Any) -> Optional[PlayerMarkets]:
        name, label_team = split_player_team(label)
        if not name or name.casefold() == "no touchdown scorer":
            return None
        # Defensive touchdown markets are not enough to project D/ST scoring.
        if is_defense_name(name):
            return None
        event_teams = bovada_event_team_codes(event)
        # Bovada sometimes leaves a former team or a college abbreviation in
        # parentheses on deep touchdown-scorer outcomes. Trust the suffix only
        # when it belongs to this event; Underdog appearance metadata can fill
        # an unknown team during the guarded same-player/same-game merge.
        team = label_team if not event_teams or label_team in event_teams else "UNK"
        event_id = str(event.get("id") or event.get("description") or "unknown-event")
        key = player_identity(event_id, name, team)
        if key not in players:
            start = event.get("startTime")
            try:
                start_ms = int(start) if start is not None else None
            except (TypeError, ValueError):
                start_ms = None
            players[key] = PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(event.get("description") or "Unknown matchup"),
                start_time_ms=start_ms,
            )
        return players[key]

    for event in iter_bovada_events(payload):
        if bool(event.get("live")):
            continue
        for display_group in event.get("displayGroups", []):
            if not isinstance(display_group, Mapping):
                continue
            for market in display_group.get("markets", []):
                if not isinstance(market, Mapping) or not is_open_full_game_market(market):
                    continue

                description = " ".join(str(market.get("description") or "").split())
                market_id = str(market.get("id") or "")

                # One-way non-passing touchdown ladders.
                td_threshold: Optional[int] = None
                if description.casefold() == "anytime touchdown scorer":
                    td_threshold = 1
                else:
                    td_match = re.fullmatch(
                        r"Player to Score\s+(\d+)\s+or More Touchdowns",
                        description,
                        flags=re.I,
                    )
                    if td_match:
                        td_threshold = int(td_match.group(1))

                if td_threshold is not None:
                    for outcome in market.get("outcomes", []):
                        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
                            continue
                        probability = implied_probability(outcome.get("price"))
                        player = get_player(event, outcome.get("description"))
                        if probability is not None and player is not None:
                            player.market("any_touchdowns").add_alternate(
                                td_threshold, probability, market_id, "bovada"
                            )
                    continue

                match = re.fullmatch(
                    r"(Alternate|Total)\s+(.+?)\s+-\s+(.+)", description, flags=re.I
                )
                if not match:
                    continue

                family, raw_stat_name, player_label = match.groups()
                canonical_stat_name = next(
                    (
                        alias
                        for label, alias in STAT_ALIASES.items()
                        if raw_stat_name.casefold() == label.casefold()
                    ),
                    None,
                )
                if canonical_stat_name is None:
                    continue
                player = get_player(event, player_label)
                if player is None:
                    continue
                stat_market = player.market(canonical_stat_name)

                if family.casefold() == "alternate":
                    for outcome in market.get("outcomes", []):
                        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
                            continue
                        threshold = threshold_from_outcome(outcome.get("description"))
                        probability = implied_probability(outcome.get("price"))
                        if threshold is not None and probability is not None:
                            stat_market.add_alternate(
                                threshold, probability, market_id, "bovada"
                            )
                else:
                    parsed = parse_two_way_market(market.get("outcomes"))
                    if parsed is not None:
                        threshold, fair_over_probability = parsed
                        stat_market.add_anchor(
                            threshold,
                            fair_over_probability,
                            market_id,
                            "bovada",
                        )

    return players


# Backwards-compatible name for callers that only use the Bovada parser.
def parse_payload(payload: Any) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    return parse_bovada_payload(payload)


def mapping_by_id(items: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("id")): item
        for item in items
        if isinstance(item, Mapping) and item.get("id") is not None
    }


def underdog_option_probability(option: Mapping[str, Any]) -> Optional[float]:
    return implied_probability(
        {
            "american": option.get("american_price"),
            "decimal": option.get("decimal_price"),
        }
    )


def underdog_team_code(
    appearance: Mapping[str, Any], game: Mapping[str, Any], player: Mapping[str, Any]
) -> str:
    title = str(game.get("abbreviated_title") or game.get("title") or "")
    parts = re.split(r"\s+@\s+", title.strip(), maxsplit=1)
    if len(parts) != 2:
        return "UNK"
    away_code, home_code = (canonical_team(part) for part in parts)
    team_id = str(appearance.get("team_id") or player.get("team_id") or "")
    if team_id and team_id == str(game.get("away_team_id") or ""):
        return away_code
    if team_id and team_id == str(game.get("home_team_id") or ""):
        return home_code
    return "UNK"


def parse_underdog_payload(
    payload: Any,
) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    """Parse active, pregame NFL player lines from Underdog's public feed."""

    validate_underdog_payload(payload)
    players_by_id = mapping_by_id(payload.get("players"))
    appearances_by_id = mapping_by_id(payload.get("appearances"))
    games_by_id = mapping_by_id(payload.get("games"))
    parsed_players: Dict[Tuple[str, str, str], PlayerMarkets] = {}

    # Keep roster metadata even when a player has only a Bovada touchdown
    # price. These empty records are used solely to repair stale Bovada team
    # suffixes and add positions during the cross-feed merge; make_projections
    # later skips any record that still has no supported market.
    for appearance in appearances_by_id.values():
        if str(appearance.get("type") or "").casefold() != "player":
            continue
        if str(appearance.get("match_type") or "").casefold() != "game":
            continue
        player = players_by_id.get(str(appearance.get("player_id")))
        game = games_by_id.get(str(appearance.get("match_id")))
        if not isinstance(player, Mapping) or not isinstance(game, Mapping):
            continue
        if str(player.get("sport_id") or "").upper() != "NFL":
            continue
        if str(game.get("sport_id") or "").upper() != "NFL":
            continue
        name = " ".join(
            part
            for part in (
                str(player.get("first_name") or "").strip(),
                str(player.get("last_name") or "").strip(),
            )
            if part
        )
        position = str(player.get("position_name") or "UNK").upper()
        if not name or is_defense_name(name) or position in {"DEF", "DST", "D/ST"}:
            continue
        team = underdog_team_code(appearance, game, player)
        game_id = str(game.get("id") or appearance.get("match_id") or "unknown")
        event_id = f"underdog:{game_id}"
        key = player_identity(event_id, name, team)
        parsed_players.setdefault(
            key,
            PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(
                    game.get("full_team_names_title")
                    or game.get("short_title")
                    or game.get("title")
                    or "Unknown matchup"
                ),
                start_time_ms=parse_iso_time_ms(game.get("scheduled_at")),
                position=position,
            ),
        )

    for line in payload.get("over_under_lines", []):
        if not isinstance(line, Mapping):
            continue
        if str(line.get("status") or "").casefold() != "active":
            continue
        if bool(line.get("live_event")):
            continue
        # Balanced lines are the ordinary, non-boosted market. Excluding
        # promotional line types avoids baking discounts into projections.
        if str(line.get("line_type") or "").casefold() != "balanced":
            continue

        over_under = line.get("over_under")
        if not isinstance(over_under, Mapping):
            continue
        if str(over_under.get("category") or "").casefold() != "player_prop":
            continue
        appearance_stat = over_under.get("appearance_stat")
        if not isinstance(appearance_stat, Mapping):
            continue
        stat = UNDERDOG_STAT_ALIASES.get(str(appearance_stat.get("stat") or ""))
        if stat is None:
            continue

        appearance = appearances_by_id.get(str(appearance_stat.get("appearance_id")))
        if not isinstance(appearance, Mapping):
            continue
        if str(appearance.get("type") or "").casefold() != "player":
            continue
        if str(appearance.get("match_type") or "").casefold() != "game":
            continue
        player = players_by_id.get(str(appearance.get("player_id")))
        game = games_by_id.get(str(appearance.get("match_id")))
        if not isinstance(player, Mapping) or not isinstance(game, Mapping):
            continue
        if str(player.get("sport_id") or "").upper() != "NFL":
            continue
        if str(game.get("sport_id") or "").upper() != "NFL":
            continue
        position = str(player.get("position_name") or "UNK").upper()

        try:
            line_value = float(str(line.get("stat_value")).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(line_value) or line_value < 0:
            continue
        # Higher than x.5 (or x with pushes possible) means an integer result
        # of floor(x) + 1 or greater, matching the survival-curve convention.
        threshold = float(math.floor(line_value) + 1)

        side_probabilities: Dict[str, List[float]] = {}
        for option in line.get("options", []):
            if not isinstance(option, Mapping):
                continue
            if str(option.get("status") or "").casefold() != "active":
                continue
            side = str(option.get("choice") or "").casefold()
            if side not in {"higher", "lower"}:
                continue
            probability = underdog_option_probability(option)
            if probability is not None:
                side_probabilities.setdefault(side, []).append(probability)

        higher_values = side_probabilities.get("higher", [])
        lower_values = side_probabilities.get("lower", [])
        if not higher_values:
            # A lower-only selection cannot be converted with the same
            # one-way vig adjustment, so omit it rather than invert it badly.
            continue
        higher_probability = statistics.median(higher_values)

        name = " ".join(
            part
            for part in (
                str(player.get("first_name") or "").strip(),
                str(player.get("last_name") or "").strip(),
            )
            if part
        )
        if not name or is_defense_name(name) or position in {"DEF", "DST", "D/ST"}:
            continue
        team = underdog_team_code(appearance, game, player)
        game_id = str(game.get("id") or appearance.get("match_id") or "unknown")
        event_id = f"underdog:{game_id}"
        key = player_identity(event_id, name, team)
        if key not in parsed_players:
            parsed_players[key] = PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(
                    game.get("full_team_names_title")
                    or game.get("short_title")
                    or game.get("title")
                    or "Unknown matchup"
                ),
                start_time_ms=parse_iso_time_ms(game.get("scheduled_at")),
                position=position,
            )

        market = parsed_players[key].market(stat)
        market_id = str(line.get("id") or line.get("stable_id") or "")
        if lower_values:
            lower_probability = statistics.median(lower_values)
            market.add_anchor(
                threshold,
                no_vig_two_way(higher_probability, lower_probability),
                market_id,
                "underdog",
            )
        else:
            market.add_alternate(
                threshold, higher_probability, market_id, "underdog"
            )

    return parsed_players


def merge_game_token(player: PlayerMarkets) -> str:
    if player.start_time_ms is not None:
        # Feeds occasionally differ by a few seconds. Minute precision is
        # strict enough to separate NFL games while tolerating that drift.
        return f"time:{int(round(player.start_time_ms / 60000.0))}"
    return "matchup:" + normalized_name(player.matchup)


def merge_player_data(target: PlayerMarkets, incoming: PlayerMarkets) -> None:
    if target.team == "UNK" and incoming.team != "UNK":
        target.team = incoming.team
    if target.position == "UNK" and incoming.position != "UNK":
        target.position = incoming.position
    if target.start_time_ms is None:
        target.start_time_ms = incoming.start_time_ms
    if len(incoming.matchup) > len(target.matchup):
        target.matchup = incoming.matchup

    for stat, incoming_market in incoming.stats.items():
        target_market = target.market(stat)
        for threshold, probabilities in incoming_market.alternate.items():
            target_market.alternate.setdefault(threshold, []).extend(probabilities)
        target_market.anchors.extend(incoming_market.anchors)
        target_market.market_ids.update(incoming_market.market_ids)
        target_market.providers.update(incoming_market.providers)


def merge_player_collections(
    *collections: Mapping[Tuple[str, str, str], PlayerMarkets],
) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    """Merge provider records for the same player in the same scheduled game."""

    merged_list: List[PlayerMarkets] = []
    loose_index: Dict[Tuple[str, str], List[PlayerMarkets]] = {}

    for collection in collections:
        for incoming in collection.values():
            loose_key = (merge_game_token(incoming), normalized_name(incoming.name))
            candidates = loose_index.get(loose_key, [])
            incoming_team = canonical_team(incoming.team)
            compatible = [
                candidate
                for candidate in candidates
                if canonical_team(candidate.team) == incoming_team
                or "UNK" in {canonical_team(candidate.team), incoming_team}
            ]
            target = compatible[0] if len(compatible) == 1 else None
            if target is None:
                incoming.team = incoming_team
                merged_list.append(incoming)
                loose_index.setdefault(loose_key, []).append(incoming)
            else:
                merge_player_data(target, incoming)

    result: Dict[Tuple[str, str, str], PlayerMarkets] = {}
    for index, player in enumerate(merged_list):
        key = player_identity(player.event_id, player.name, player.team)
        if key in result:
            key = (f"{player.event_id}:{index}", key[1], key[2])
        result[key] = player
    return result


def isotonic_nonincreasing(
    points: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Weighted pool-adjacent-violators fit for a survival curve."""

    if not points:
        return []
    ordered = sorted(points, key=lambda point: point[0])
    blocks: List[List[float]] = []
    for index, (_, probability, weight) in enumerate(ordered):
        blocks.append([float(index), float(index), weight, probability * weight])
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = previous[3] / previous[2]
            current_mean = current[3] / current[2]
            if previous_mean >= current_mean:
                break
            merged = [
                previous[0],
                current[1],
                previous[2] + current[2],
                previous[3] + current[3],
            ]
            blocks[-2:] = [merged]

    fitted = [0.0] * len(ordered)
    for start, end, weight, weighted_sum in blocks:
        mean = clamp_probability(weighted_sum / weight)
        for index in range(int(start), int(end) + 1):
            fitted[index] = mean
    return [
        (ordered[index][0], fitted[index], ordered[index][2])
        for index in range(len(ordered))
    ]


def collapsed_alternate(market: StatMarket) -> List[Tuple[float, float, float]]:
    points = [
        (threshold, statistics.median(probabilities), float(len(probabilities)))
        for threshold, probabilities in market.alternate.items()
        if probabilities
    ]
    return isotonic_nonincreasing(points)


def interpolate_logit(
    points: Sequence[Tuple[float, float, float]], threshold: float
) -> Optional[float]:
    ordered = sorted(points, key=lambda point: point[0])
    for point_threshold, probability, _ in ordered:
        if math.isclose(point_threshold, threshold, abs_tol=1e-9):
            return probability
    for left, right in zip(ordered, ordered[1:]):
        x0, p0, _ = left
        x1, p1, _ = right
        if x0 <= threshold <= x1 and x1 > x0:
            weight = (threshold - x0) / (x1 - x0)
            return logistic(logit(p0) + weight * (logit(p1) - logit(p0)))
    return None


def local_logit_vig_shift(market: StatMarket) -> Optional[float]:
    alternate = collapsed_alternate(market)
    shifts = []
    for anchor in market.anchors:
        raw_probability = interpolate_logit(alternate, anchor.threshold)
        if raw_probability is not None:
            shifts.append(logit(raw_probability) - logit(anchor.probability))
    if not shifts:
        return None
    return min(0.75, max(-0.35, statistics.median(shifts)))


def estimate_global_logit_vig(
    players: Mapping[Tuple[str, str, str], PlayerMarkets],
    fallback: float = DEFAULT_FALLBACK_LOGIT_VIG,
) -> Tuple[float, int]:
    shifts = [
        shift
        for player in players.values()
        for market in player.stats.values()
        if (shift := local_logit_vig_shift(market)) is not None
    ]
    if len(shifts) >= 3:
        return statistics.median(shifts), len(shifts)
    return fallback, len(shifts)


def fair_observations(
    market: StatMarket, global_logit_vig: float
) -> Tuple[List[Observation], str]:
    alternate = collapsed_alternate(market)
    local_shift = local_logit_vig_shift(market)
    shift = local_shift if local_shift is not None else global_logit_vig

    observations = [
        Observation(
            threshold=threshold,
            probability=logistic(logit(probability) - shift),
            weight=weight,
            source="alternate",
        )
        for threshold, probability, weight in alternate
    ]
    observations.extend(market.anchors)

    # Aggregate coincident total and alternate thresholds, then impose the
    # required non-increasing shape one final time.
    by_threshold: Dict[float, List[Observation]] = {}
    for observation in observations:
        by_threshold.setdefault(observation.threshold, []).append(observation)
    combined = []
    for threshold, entries in by_threshold.items():
        total_weight = sum(entry.weight for entry in entries)
        probability = sum(
            entry.probability * entry.weight for entry in entries
        ) / total_weight
        combined.append((threshold, probability, total_weight))
    monotone = isotonic_nonincreasing(combined)

    provider_label = "+".join(sorted(market.providers)) or "unknown"
    if alternate and market.anchors:
        source = f"{provider_label}:total+alternate"
    elif market.anchors:
        source = f"{provider_label}:total-only"
    else:
        source = f"{provider_label}:alternate-only"
    return [Observation(x, p, w, source) for x, p, w in monotone], source


def fit_weibull(
    observations: Sequence[Observation], default_shape: float
) -> Distribution:
    valid = [
        observation
        for observation in observations
        if observation.threshold > 0
        and EPSILON < observation.probability < 1.0 - EPSILON
    ]
    if not valid:
        return Distribution("weibull", 0.0, default_shape, EPSILON)

    shape = default_shape
    if len(valid) >= 2 and len({item.threshold for item in valid}) >= 2:
        x_values = [math.log(item.threshold) for item in valid]
        y_values = [math.log(-math.log(item.probability)) for item in valid]
        weights = [item.weight for item in valid]
        weight_sum = sum(weights)
        x_mean = sum(x * w for x, w in zip(x_values, weights)) / weight_sum
        y_mean = sum(y * w for y, w in zip(y_values, weights)) / weight_sum
        denominator = sum(
            w * (x - x_mean) ** 2 for x, w in zip(x_values, weights)
        )
        if denominator > 0:
            fitted_shape = sum(
                w * (x - x_mean) * (y - y_mean)
                for x, y, w in zip(x_values, y_values, weights)
            ) / denominator
            if math.isfinite(fitted_shape) and fitted_shape > 0:
                shape = min(8.0, max(0.70, fitted_shape))

    # Refit the intercept after shape clamping.
    weights = [item.weight for item in valid]
    weight_sum = sum(weights)
    intercept = sum(
        item.weight
        * (math.log(-math.log(item.probability)) - shape * math.log(item.threshold))
        for item in valid
    ) / weight_sum
    scale = math.exp(-intercept / shape)
    mean = scale * math.gamma(1.0 + 1.0 / shape)
    if not math.isfinite(mean) or mean < 0:
        mean = 0.0
    return Distribution("weibull", mean, shape, scale)


def poisson_survival(threshold: int, rate: float) -> float:
    """P[X >= threshold] for X ~ Poisson(rate)."""

    if threshold <= 0:
        return 1.0
    if rate <= 0:
        return 0.0
    term = math.exp(-rate)
    cumulative = term
    for value in range(1, threshold):
        term *= rate / value
        cumulative += term
    return clamp_probability(1.0 - cumulative)


def fit_poisson(observations: Sequence[Observation]) -> Distribution:
    valid = [
        observation
        for observation in observations
        if observation.threshold >= 1
        and EPSILON < observation.probability < 1.0 - EPSILON
    ]
    if not valid:
        return Distribution("poisson", 0.0, 0.0)

    max_threshold = max(int(round(item.threshold)) for item in valid)
    lower_log = math.log(1e-4)
    upper_log = math.log(max(12.0, max_threshold * 5.0))

    def objective(log_rate: float) -> float:
        rate = math.exp(log_rate)
        total = 0.0
        for item in valid:
            fitted = poisson_survival(int(round(item.threshold)), rate)
            residual = logit(fitted) - logit(item.probability)
            total += item.weight * residual * residual
        return total

    # Golden-section minimization in log(rate), avoiding a SciPy dependency.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower_log, upper_log
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(100):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    rate = math.exp((left + right) / 2.0)
    return Distribution("poisson", rate, rate)


def fit_stat_distribution(
    stat: str, market: StatMarket, global_logit_vig: float
) -> Tuple[Distribution, str]:
    observations, source = fair_observations(market, global_logit_vig)
    if stat in YARD_STATS:
        return fit_weibull(observations, DEFAULT_WEIBULL_SHAPES[stat]), source
    if stat in COUNT_STATS:
        return fit_poisson(observations), source
    raise ValueError(f"Unsupported stat: {stat}")


def projection_quality(
    distributions: Mapping[str, Distribution],
    sources: Mapping[str, str],
    position: str = "UNK",
) -> str:
    stats = set(distributions)
    has_total_anchor = any("total" in source for source in sources.values())
    is_quarterback = position.upper() == "QB" or bool(
        stats & {"passing_yards", "passing_touchdowns", "interceptions"}
    )
    if is_quarterback:
        quarterback_core = {"passing_yards", "passing_touchdowns", "interceptions"}
        if quarterback_core <= stats and has_total_anchor:
            return "good"
        if {"passing_yards", "passing_touchdowns"} <= stats:
            return "fair"
        return "partial"

    skill_core = sum(
        stat in stats
        for stat in ("rushing_yards", "receiving_yards", "receptions")
    )
    has_touchdown = "any_touchdowns" in stats
    if skill_core >= 2 and has_touchdown and has_total_anchor:
        return "good"
    if skill_core >= 2 or (skill_core >= 1 and has_touchdown):
        return "fair"
    if skill_core >= 1:
        return "partial"
    return "td-only" if has_touchdown else "partial"


def iso_start_time(start_time_ms: Optional[int]) -> Optional[str]:
    if start_time_ms is None:
        return None
    try:
        return datetime.fromtimestamp(
            start_time_ms / 1000.0, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _ordinary_least_squares(
    samples: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Return intercept and slope for y = intercept + slope*x."""

    if len(samples) < 2:
        return None
    x_mean = sum(x for x, _ in samples) / len(samples)
    y_mean = sum(y for _, y in samples) / len(samples)
    denominator = sum((x - x_mean) ** 2 for x, _ in samples)
    if denominator <= EPSILON:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in samples) / denominator
    intercept = y_mean - slope * x_mean
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        return None
    return intercept, slope


def fit_td_regression(
    samples: Sequence[Tuple[float, float]],
    label: str,
    minimum_samples: int,
) -> Optional[TdRegression]:
    """Fit a robust linear regression after one MAD residual trim."""

    clean = [
        (float(expected_td), float(fantasy_points))
        for expected_td, fantasy_points in samples
        if math.isfinite(expected_td)
        and math.isfinite(fantasy_points)
        and expected_td > 0
        and fantasy_points > 0
    ]
    if len(clean) < minimum_samples:
        return None

    coefficients = _ordinary_least_squares(clean)
    if coefficients is None:
        return None
    intercept, slope = coefficients
    residuals = [y - (intercept + slope * x) for x, y in clean]
    residual_median = statistics.median(residuals)
    residual_mad = statistics.median(
        abs(residual - residual_median) for residual in residuals
    )
    if residual_mad > EPSILON:
        robust_sigma = 1.4826 * residual_mad
        trimmed = [
            sample
            for sample, residual in zip(clean, residuals)
            if abs(residual - residual_median)
            <= TD_REGRESSION_RESIDUAL_MAD_LIMIT * robust_sigma
        ]
        if minimum_samples <= len(trimmed) < len(clean):
            trimmed_coefficients = _ordinary_least_squares(trimmed)
            if trimmed_coefficients is not None:
                clean = trimmed
                intercept, slope = trimmed_coefficients

    predictions = [intercept + slope * x for x, _ in clean]
    y_mean = sum(y for _, y in clean) / len(clean)
    residual_sum_squares = sum(
        (y - fitted) ** 2 for (_, y), fitted in zip(clean, predictions)
    )
    total_sum_squares = sum((y - y_mean) ** 2 for _, y in clean)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > EPSILON
        else 0.0
    )
    return TdRegression(
        label=label,
        intercept=intercept,
        slope=slope,
        sample_count=len(clean),
        r_squared=max(0.0, min(1.0, r_squared)),
        maximum_target=max(y for _, y in clean),
    )


def fit_td_regression_models(
    projections: Sequence[Projection],
) -> Dict[str, TdRegression]:
    """Train global and position fits from good non-QB projections only."""

    global_samples: List[Tuple[float, float]] = []
    position_samples: Dict[str, List[Tuple[float, float]]] = {}
    for projection in projections:
        stats = set(projection.stat_means)
        if (
            projection.quality != "good"
            or "any_touchdowns" not in stats
            or projection.position.upper() == "QB"
            or bool(stats & {"passing_yards", "passing_touchdowns"})
        ):
            continue
        sample = (
            projection.stat_means["any_touchdowns"],
            projection.fantasy_points,
        )
        global_samples.append(sample)
        position = projection.position.upper()
        if position in {"RB", "WR", "TE"}:
            position_samples.setdefault(position, []).append(sample)

    models: Dict[str, TdRegression] = {}
    global_model = fit_td_regression(
        global_samples,
        "skill",
        MIN_GLOBAL_TD_REGRESSION_SAMPLES,
    )
    if global_model is not None:
        models["*"] = global_model
    for position, samples in position_samples.items():
        model = fit_td_regression(
            samples,
            position,
            MIN_POSITION_TD_REGRESSION_SAMPLES,
        )
        if model is not None:
            models[position] = model
    return models


def estimate_td_only_fantasy_points(
    projections: Sequence[Projection], scoring: Scoring
) -> None:
    """Replace bare TD scoring with slate-regressed full-FP estimates in place."""

    models = fit_td_regression_models(projections)
    global_model = models.get("*")
    for projection in projections:
        if (
            projection.quality != "td-only"
            or "any_touchdowns" not in projection.stat_means
        ):
            continue
        position = projection.position.upper()
        model = None if position == "QB" else models.get(position, global_model)
        expected_touchdowns = projection.stat_means.get("any_touchdowns", 0.0)
        if model is None:
            projection.fantasy_points_method = "touchdown-component-only"
            continue
        projection.fantasy_points = round(
            model.predict(
                expected_touchdowns,
                scoring.rushing_receiving_touchdown,
            ),
            2,
        )
        projection.quality = "td-estimate"
        projection.fantasy_points_method = (
            f"td-regression:{model.label};n={model.sample_count};"
            f"r2={model.r_squared:.3f};intercept={model.intercept:.3f};"
            f"slope={model.slope:.3f}"
        )


def make_projections(
    players: Mapping[Tuple[str, str, str], PlayerMarkets],
    scoring: Scoring,
    fallback_logit_vig: float = DEFAULT_FALLBACK_LOGIT_VIG,
) -> Tuple[List[Projection], float, int]:
    global_logit_vig, calibration_pairs = estimate_global_logit_vig(
        players, fallback_logit_vig
    )
    projections: List[Projection] = []

    for player in players.values():
        distributions: Dict[str, Distribution] = {}
        sources: Dict[str, str] = {}
        for stat, market in player.stats.items():
            if not market.alternate and not market.anchors:
                continue
            distribution, source = fit_stat_distribution(
                stat, market, global_logit_vig
            )
            distributions[stat] = distribution
            sources[stat] = source

        if not distributions:
            continue
        means = {stat: distribution.mean for stat, distribution in distributions.items()}
        fantasy_points = (
            means.get("passing_yards", 0.0) * scoring.passing_yard
            + means.get("rushing_yards", 0.0) * scoring.rushing_yard
            + means.get("receiving_yards", 0.0) * scoring.receiving_yard
            + means.get("receptions", 0.0) * scoring.reception
            + means.get("passing_touchdowns", 0.0) * scoring.passing_touchdown
            + means.get("any_touchdowns", 0.0)
            * scoring.rushing_receiving_touchdown
            + means.get("interceptions", 0.0) * scoring.interception
            + means.get("fumbles_lost", 0.0) * scoring.fumble_lost
        )
        projections.append(
            Projection(
                event_id=player.event_id,
                matchup=player.matchup,
                start_time_utc=iso_start_time(player.start_time_ms),
                team=player.team,
                player=player.name,
                position=player.position,
                fantasy_points=round(fantasy_points, 2),
                quality=projection_quality(distributions, sources, player.position),
                stat_means={key: round(value, 3) for key, value in means.items()},
                sources=sources,
            )
        )

    estimate_td_only_fantasy_points(projections, scoring)
    projections.sort(
        key=lambda item: (
            item.start_time_utc or "9999",
            item.matchup,
            item.team,
            -item.fantasy_points,
            item.player,
        )
    )
    return projections, global_logit_vig, calibration_pairs


def print_projections(
    projections: Sequence[Projection],
    scoring_name: str,
    logit_vig: float,
    calibration_pairs: int,
    include_td_only: bool,
    loaded_feeds: Sequence[str],
) -> None:
    print(
        f"Feeds: {' + '.join(loaded_feeds)} | Scoring: {scoring_name} | "
        f"one-way logit adjustment: {logit_vig:.3f} "
        f"({calibration_pairs} local calibration pairs)"
    )
    print("Coverage labels describe available prop components, not certainty.")
    estimated_count = sum(
        projection.quality == "td-estimate" for projection in projections
    )
    if estimated_count:
        print(
            f"TD-only regression estimates: {estimated_count}; targets are good "
            "non-QB projections from this slate."
        )

    visible = [
        projection
        for projection in projections
        if include_td_only
        or projection.quality not in {"td-only", "td-estimate"}
    ]
    if not visible:
        print("No sufficiently covered player prop projections were found.")
        return

    current_group: Optional[Tuple[str, str, str]] = None
    for projection in visible:
        group = (
            projection.start_time_utc or "time unavailable",
            projection.matchup,
            projection.team,
        )
        if group != current_group:
            print(
                f"\n{projection.matchup} | {projection.start_time_utc or 'time unavailable'}"
            )
            print(f"Team: {projection.team}")
            current_group = group

        component_order = [
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "receptions",
            "passing_touchdowns",
            "any_touchdowns",
            "interceptions",
            "fumbles_lost",
        ]
        components = " ".join(
            f"{STAT_LABELS[stat]}={projection.stat_means[stat]:.2f}"
            for stat in component_order
            if stat in projection.stat_means
        )
        regression_note = ""
        if projection.quality == "td-estimate":
            match = re.search(
                r"^td-regression:([^;]+);n=(\d+);r2=([\d.]+)",
                projection.fantasy_points_method,
            )
            if match:
                label, sample_count, r_squared = match.groups()
                regression_note = (
                    f" | model={label} n={sample_count} R2={r_squared}"
                )
        print(
            f"  {projection.player}: {projection.fantasy_points:.2f} FP "
            f"[{projection.quality}] | {components}{regression_note}"
        )

    hidden_count = len(projections) - len(visible)
    if hidden_count:
        print(
            f"\nHidden: {hidden_count} touchdown-only estimates. "
            "Remove --exclude-td-only to display them."
        )


def run_self_test() -> None:
    assert math.isclose(american_odds_to_probability("EVEN") or 0, 0.5)
    assert math.isclose(american_odds_to_probability("+100") or 0, 0.5)
    assert math.isclose(
        american_odds_to_probability("-110") or 0, 110 / 210, rel_tol=1e-12
    )
    assert math.isclose(no_vig_two_way(110 / 210, 110 / 210), 0.5)
    assert canonical_team("NOR") == "NO"
    assert is_defense_name("Seattle Seahawks Def/ST")
    assert is_defense_name("CHI Bears D/ST")
    assert bovada_event_team_codes(
        {
            "competitors": [
                {"name": "New England Patriots"},
                {"name": "Seattle Seahawks"},
            ]
        }
    ) == {"NE", "SEA"}

    monotone = isotonic_nonincreasing(
        [(10, 0.80, 1), (20, 0.60, 1), (30, 0.62, 1), (40, 0.20, 1)]
    )
    assert all(
        left[1] >= right[1] for left, right in zip(monotone, monotone[1:])
    )

    rate = 3.25
    synthetic = [
        Observation(k, poisson_survival(k, rate), 1.0, "test")
        for k in range(1, 6)
    ]
    fitted = fit_poisson(synthetic)
    assert math.isclose(fitted.mean, rate, rel_tol=1e-5)

    weibull = fit_weibull(
        [
            Observation(40, math.exp(-((40 / 80) ** 2)), 1.0, "test"),
            Observation(80, math.exp(-1), 1.0, "test"),
            Observation(120, math.exp(-((120 / 80) ** 2)), 1.0, "test"),
        ],
        default_shape=2.0,
    )
    assert math.isclose(weibull.parameter_1, 2.0, rel_tol=1e-5)
    assert math.isclose(weibull.parameter_2 or 0, 80.0, rel_tol=1e-5)

    def synthetic_underdog_line(
        line_id: str,
        stat: str,
        value: str,
        options: Sequence[Tuple[str, str]],
    ) -> Dict[str, Any]:
        return {
            "id": line_id,
            "line_type": "balanced",
            "live_event": False,
            "status": "active",
            "stat_value": value,
            "options": [
                {
                    "choice": choice,
                    "american_price": price,
                    "status": "active",
                }
                for choice, price in options
            ],
            "over_under": {
                "category": "player_prop",
                "appearance_stat": {"appearance_id": "a1", "stat": stat},
            },
        }

    underdog_fixture = {
        "players": [
            {
                "id": "p1",
                "first_name": "Test",
                "last_name": "Quarterback",
                "position_name": "QB",
                "sport_id": "NFL",
                "team_id": "away-id",
            }
        ],
        "appearances": [
            {
                "id": "a1",
                "match_id": 7,
                "match_type": "Game",
                "player_id": "p1",
                "team_id": "away-id",
                "type": "Player",
            }
        ],
        "games": [
            {
                "id": 7,
                "abbreviated_title": "NE @ SEA",
                "away_team_id": "away-id",
                "home_team_id": "home-id",
                "full_team_names_title": "New England Patriots @ Seattle Seahawks",
                "scheduled_at": "2026-09-10T00:20:00Z",
                "sport_id": "NFL",
            }
        ],
        "over_under_lines": [
            synthetic_underdog_line(
                "line-yards",
                "passing_yds",
                "249.5",
                (("higher", "-110"), ("lower", "-110")),
            ),
            synthetic_underdog_line(
                "line-fumble", "fumbles_lost", "0.5", (("higher", "+350"),)
            ),
        ],
    }
    parsed_underdog = parse_underdog_payload(underdog_fixture)
    assert len(parsed_underdog) == 1
    test_player = next(iter(parsed_underdog.values()))
    assert test_player.team == "NE" and test_player.position == "QB"
    assert math.isclose(
        test_player.stats["passing_yards"].anchors[0].probability, 0.5
    )
    assert test_player.stats["passing_yards"].anchors[0].threshold == 250.0
    assert test_player.stats["fumbles_lost"].alternate[1.0]

    bovada_record = PlayerMarkets(
        event_id="bovada:7",
        name="A.J. Brown",
        team="UNK",
        matchup="New England Patriots @ Seattle Seahawks",
        start_time_ms=parse_iso_time_ms("2026-09-10T00:20:00Z"),
    )
    bovada_record.market("receiving_yards").add_alternate(
        50, 0.65, "b1", "bovada"
    )
    underdog_record = PlayerMarkets(
        event_id="underdog:7",
        name="AJ Brown",
        team="NE",
        matchup="NE @ SEA",
        start_time_ms=parse_iso_time_ms("2026-09-10T00:20:00Z"),
        position="WR",
    )
    underdog_record.market("receptions").add_anchor(
        5, 0.5, "u1", "underdog"
    )
    merged = merge_player_collections(
        {player_identity("bovada:7", "A.J. Brown", "NE"): bovada_record},
        {player_identity("underdog:7", "AJ Brown", "NE"): underdog_record},
    )
    assert len(merged) == 1
    merged_player = next(iter(merged.values()))
    assert set(merged_player.stats) == {"receiving_yards", "receptions"}
    assert merged_player.team == "NE"
    assert merged_player.position == "WR"

    regression_training = [
        Projection(
            event_id="training",
            matchup="Training @ Sample",
            start_time_utc=None,
            team="TST",
            player=f"Training Receiver {index}",
            position="WR",
            fantasy_points=4.0 + 15.0 * expected_td,
            quality="good",
            stat_means={
                "receiving_yards": 40.0,
                "receptions": 3.0,
                "any_touchdowns": expected_td,
            },
            sources={},
        )
        for index, expected_td in enumerate(
            (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
            1,
        )
    ]
    td_only_projection = Projection(
        event_id="test",
        matchup="Test @ Sample",
        start_time_utc=None,
        team="TST",
        player="Touchdown Only",
        position="UNK",
        fantasy_points=3.0,
        quality="td-only",
        stat_means={"any_touchdowns": 0.50},
        sources={"any_touchdowns": "bovada:alternate-only"},
    )
    regression_sample = regression_training + [td_only_projection]
    models = fit_td_regression_models(regression_sample)
    assert "*" in models and "WR" in models
    assert math.isclose(models["*"].intercept, 4.0, rel_tol=1e-10)
    assert math.isclose(models["*"].slope, 15.0, rel_tol=1e-10)
    estimate_td_only_fantasy_points(regression_sample, SCORING_PRESETS["yahoo"])
    assert td_only_projection.quality == "td-estimate"
    assert math.isclose(td_only_projection.fantasy_points, 11.5, rel_tol=1e-10)
    assert td_only_projection.fantasy_points_method.startswith(
        "td-regression:skill;n=12;"
    )
    assert parse_cli_args([]).include_td_only is True
    assert parse_cli_args(["--exclude-td-only"]).include_td_only is False
    print("Self-test passed")


def load_json_file(
    path: Path, validator: Callable[[Any], None], feed_name: str
) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        validator(payload)
    except ValueError as exc:
        raise ValueError(f"{feed_name} input {path}: {exc}") from exc
    return payload


def load_bovada_payload(path: Optional[Path]) -> Any:
    if path is None:
        return fetch_bovada_payload()
    return load_json_file(path, validate_bovada_payload, "Bovada")


def load_underdog_payload(path: Optional[Path]) -> Any:
    if path is None:
        return fetch_underdog_payload()
    return load_json_file(path, validate_underdog_payload, "Underdog")


# Backwards-compatible saved/live Bovada loader.
def load_payload(path: Optional[Path]) -> Any:
    return load_bovada_payload(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build market-implied NFL fantasy projections from paired "
            "Underdog props and Bovada alternate lines."
        )
    )
    parser.add_argument(
        "--source",
        choices=("hybrid", "bovada", "underdog"),
        default="hybrid",
        help="Data feed selection (default: hybrid).",
    )
    parser.add_argument(
        "--input",
        "--bovada-input",
        dest="bovada_input",
        type=Path,
        help=(
            "Read a saved Bovada JSON payload instead of requesting it live. "
            "--input remains as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--underdog-input",
        type=Path,
        help="Read a saved Underdog JSON payload instead of requesting it live.",
    )
    parser.add_argument(
        "--scoring",
        choices=sorted(SCORING_PRESETS),
        default="yahoo",
        help="Fantasy scoring preset (default: yahoo).",
    )
    parser.add_argument(
        "--fallback-logit-vig",
        type=float,
        default=DEFAULT_FALLBACK_LOGIT_VIG,
        help=(
            "Fallback log-odds adjustment for one-way prices when fewer than "
            "three local total/alternate pairs exist (default: 0.17)."
        ),
    )
    td_display = parser.add_mutually_exclusive_group()
    td_display.add_argument(
        "--include-td-only",
        dest="include_td_only",
        action="store_true",
        help=(
            "Show touchdown-only regression estimates (enabled by default; "
            "retained for backwards compatibility)."
        ),
    )
    td_display.add_argument(
        "--exclude-td-only",
        dest="include_td_only",
        action="store_false",
        help="Hide touchdown-only regression estimates.",
    )
    parser.set_defaults(include_td_only=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON rather than the formatted report.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run deterministic math checks and exit."
    )
    return parser


def running_in_notebook_kernel() -> bool:
    return (
        "ipykernel" in sys.modules
        or running_in_google_colab()
        or Path(sys.argv[0]).name == "ipykernel_launcher.py"
    )


def running_in_google_colab() -> bool:
    return (
        "google.colab" in sys.modules
        or Path(sys.argv[0]).name == "colab_kernel_launcher.py"
    )


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse user options without treating Jupyter's kernel file as an option.

    Colab/IPython launches the notebook process with ``-f kernel-....json``.
    When a complete script is pasted into a cell, that process-level argument
    is still present in ``sys.argv``.  Remove only that exact connection-file
    pair; all other unknown arguments continue to raise an argparse error.
    """

    parser = build_parser()
    if argv is not None:
        return parser.parse_args(list(argv))

    raw_arguments = list(sys.argv[1:])
    if not running_in_notebook_kernel():
        return parser.parse_args(raw_arguments)

    cleaned_arguments: List[str] = []
    index = 0
    while index < len(raw_arguments):
        argument = raw_arguments[index]
        if argument == "-f" and index + 1 < len(raw_arguments):
            connection_file = raw_arguments[index + 1]
            if re.search(r"(?:^|[/\\])kernel-[^/\\]+\.json$", connection_file):
                index += 2
                continue
        if argument.startswith("-f="):
            connection_file = argument[3:]
            if re.search(r"(?:^|[/\\])kernel-[^/\\]+\.json$", connection_file):
                index += 1
                continue
        cleaned_arguments.append(argument)
        index += 1

    return parser.parse_args(cleaned_arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    if not math.isfinite(args.fallback_logit_vig):
        print("Error: --fallback-logit-vig must be finite", file=sys.stderr)
        return 1

    collections: List[Mapping[Tuple[str, str, str], PlayerMarkets]] = []
    loaded_feeds: List[str] = []
    warnings: List[str] = []

    requested_feeds = (
        ("bovada", "underdog") if args.source == "hybrid" else (args.source,)
    )
    for feed in requested_feeds:
        try:
            if feed == "bovada":
                payload = load_bovada_payload(args.bovada_input)
                parsed = parse_bovada_payload(payload)
                display_name = "Bovada"
            else:
                payload = load_underdog_payload(args.underdog_input)
                parsed = parse_underdog_payload(payload)
                display_name = "Underdog"
            if not parsed:
                raise ValueError("no supported pregame NFL player props were found")
            collections.append(parsed)
            loaded_feeds.append(display_name)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"{feed.capitalize()} unavailable: {exc}")

    try:
        if not collections:
            details = "; ".join(warnings) or "no feeds were requested"
            raise ValueError(f"No usable player-prop feed. {details}")
        players = merge_player_collections(*collections)
        if not players:
            raise ValueError("No supported full-game player prop markets were found")
        projections, logit_vig, calibration_pairs = make_projections(
            players,
            SCORING_PRESETS[args.scoring],
            fallback_logit_vig=args.fallback_logit_vig,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        output = {
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "scoring": args.scoring,
            "source_mode": args.source,
            "feeds": loaded_feeds,
            "warnings": warnings,
            "one_way_logit_adjustment": round(logit_vig, 6),
            "calibration_pairs": calibration_pairs,
            "projections": [asdict(projection) for projection in projections],
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print_projections(
            projections,
            args.scoring,
            logit_vig,
            calibration_pairs,
            args.include_td_only,
            loaded_feeds,
        )
    return 0


# ============================================================================
# NOTEBOOK CELL 7 - Market means: Yahoo-game matching and fallback policy
# ============================================================================
MARKET_TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "NOR": "NO",
    "WSH": "WAS",
}


def _market_team(team):
    value = str(team or "").strip().upper()
    return MARKET_TEAM_ALIASES.get(value, value)


def _market_name_key(name):
    """Conservative player-name key shared by Yahoo and the prop feeds."""
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", value)
    return re.sub(r"\s+", "", value).strip()


def _atomic_json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _fresh_market_cache(path, max_age_hours):
    path = Path(path)
    if not path.exists() or max_age_hours <= 0:
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_market_feed(feed, cfg):
    """Load one raw feed, caching successful payloads for a short interval."""
    cache_path = Path(cfg.market_cache_dir) / f"{feed}.json"
    cached = _fresh_market_cache(cache_path, cfg.market_cache_hours)
    if cached is not None:
        validator = validate_bovada_payload if feed == "bovada" else validate_underdog_payload
        validator(cached)
        return cached, "cache"

    if feed == "bovada":
        payload = fetch_bovada_payload(timeout=cfg.nflverse_timeout)
        validate_bovada_payload(payload)
    elif feed == "underdog":
        payload = fetch_underdog_payload(timeout=max(45, cfg.nflverse_timeout))
        validate_underdog_payload(payload)
    else:
        raise ValueError(f"Unsupported market feed: {feed}")
    try:
        _atomic_json_write(cache_path, payload)
    except OSError as exc:
        warnings.warn(f"Could not cache {feed} market payload: {exc}")
    return payload, "live"


def load_market_projection_reference(cfg=None):
    """Return market Projection objects and a transparent feed audit.

    Hybrid mode degrades to either surviving feed. It falls back to Yahoo priors
    only if both feeds fail; one provider outage never aborts lineup generation.
    """
    cfg = _cfg(cfg)
    if cfg.market_source not in {"hybrid", "bovada", "underdog"}:
        raise ValueError("Settings.market_source must be hybrid, bovada, or underdog")
    if cfg.market_scoring not in SCORING_PRESETS:
        raise ValueError(f"Unknown market scoring preset: {cfg.market_scoring}")

    requested = (
        ("bovada", "underdog")
        if cfg.market_source == "hybrid"
        else (cfg.market_source,)
    )
    collections = []
    feed_notes = []
    loaded = []
    for feed in requested:
        try:
            payload, mode = _load_market_feed(feed, cfg)
            parsed = (
                parse_bovada_payload(payload)
                if feed == "bovada"
                else parse_underdog_payload(payload)
            )
            if not parsed:
                raise ValueError("no supported pregame NFL player props")
            collections.append(parsed)
            loaded.append(feed.capitalize())
            feed_notes.append(f"{feed}: {len(parsed):,} player-market records ({mode})")
        except (OSError, RuntimeError, ValueError, HTTPError, URLError) as exc:
            feed_notes.append(f"{feed} unavailable ({type(exc).__name__}: {exc})")

    if not collections:
        return [], {
            "feeds": [],
            "notes": feed_notes,
            "logit_vig": np.nan,
            "calibration_pairs": 0,
        }

    players = merge_player_collections(*collections)
    projections, logit_vig, calibration_pairs = make_projections(
        players,
        SCORING_PRESETS[cfg.market_scoring],
        fallback_logit_vig=cfg.market_fallback_logit_vig,
    )
    return projections, {
        "feeds": loaded,
        "notes": feed_notes,
        "logit_vig": float(logit_vig),
        "calibration_pairs": int(calibration_pairs),
    }


def build_market_projection_report(yahoo_players, projections, selected_game, cfg=None):
    """Match market means to one Yahoo game without crossing teams or games."""
    cfg = _cfg(cfg)
    columns = [
        "Player", "Team", "Position", "Yahoo projection", "Market projection",
        "Market quality", "Market method", "Market feeds", "Market matched",
        "Market accepted", "Market reason",
    ]
    if not len(yahoo_players):
        return pd.DataFrame(columns=columns)

    game_start = pd.to_datetime(selected_game["Game Time"], errors="coerce", utc=True)
    market_rows = []
    for projection in projections:
        start = pd.to_datetime(projection.start_time_utc, errors="coerce", utc=True)
        # Game time is a hard guard. Allow a small tolerance for provider rounding.
        same_time = (
            pd.notna(game_start)
            and pd.notna(start)
            and abs((start - game_start).total_seconds()) <= 15 * 60
        )
        if not same_time:
            continue
        market_rows.append({
            "_key": (_market_team(projection.team), _market_name_key(projection.player)),
            "Market projection": float(projection.fantasy_points),
            "Market quality": str(projection.quality),
            "Market method": str(projection.fantasy_points_method),
            "Market feeds": "+".join(sorted({
                provider
                for source in projection.sources.values()
                for provider in ("Bovada" if "bovada" in source.lower() else "",
                                 "Underdog" if "underdog" in source.lower() else "")
                if provider
            })) or "market",
        })

    market = pd.DataFrame(market_rows)
    ambiguous = set()
    lookup = {}
    if len(market):
        ambiguous = set(market.loc[market.duplicated("_key", keep=False), "_key"])
        lookup = market.loc[~market["_key"].isin(ambiguous)].set_index("_key").to_dict("index")

    accepted_quality = set(cfg.market_accepted_quality)
    rows = []
    for player in yahoo_players.itertuples(index=False):
        key = (_market_team(player.Team), _market_name_key(player.Name))
        hit = lookup.get(key)
        manual = str(player.Projection_Source) == "manual override"
        matched = hit is not None
        quality = hit["Market quality"] if matched else None
        finite_positive = matched and np.isfinite(hit["Market projection"]) and hit["Market projection"] > 0
        accepted = bool(matched and quality in accepted_quality and finite_positive and not manual)
        if manual:
            reason = "manual override retained"
        elif key in ambiguous:
            reason = "ambiguous market identity"
        elif not matched:
            reason = "no same-game market projection"
        elif quality not in accepted_quality:
            reason = f"quality '{quality}' not accepted"
        elif not finite_positive:
            reason = "non-positive/non-finite market projection"
        else:
            reason = "accepted"
        rows.append({
            "Player": player.Name,
            "Team": player.Team,
            "Position": player.Position,
            "Yahoo projection": float(player.Projected_FP),
            "Market projection": hit["Market projection"] if matched else np.nan,
            "Market quality": quality,
            "Market method": hit["Market method"] if matched else None,
            "Market feeds": hit["Market feeds"] if matched else None,
            "Market matched": matched,
            "Market accepted": accepted,
            "Market reason": reason,
        })
    return pd.DataFrame(rows, columns=columns)


def apply_market_projection_means(players, report, cfg=None):
    """Replace fallback means with accepted market means; preserve manual overrides."""
    cfg = _cfg(cfg)
    out = players.copy()
    if not len(report):
        return out, pd.DataFrame()
    accepted = report[report["Market accepted"]].copy()
    by_identity = accepted.set_index(["Team", "Player"])
    for idx, player in out.iterrows():
        key = (player["Team"], player["Name"])
        if key not in by_identity.index:
            continue
        row = by_identity.loc[key]
        out.loc[idx, "Fallback_Projected_FP"] = float(player["Projected_FP"])
        out.loc[idx, "Projected_FP"] = float(row["Market projection"])
        out.loc[idx, "Projection_Source"] = (
            f"market {row['Market quality']}: {row['Market feeds']}"
        )
        out.loc[idx, "Market_Quality"] = row["Market quality"]
        out.loc[idx, "Market_Method"] = row["Market method"]
    if cfg.market_drop_unmatched:
        accepted_keys = set(zip(accepted["Team"], accepted["Player"]))
        keep = out["Position"].eq("DEF") | pd.Series(
            [(team, name) in accepted_keys for team, name in zip(out["Team"], out["Name"])],
            index=out.index,
        ) | out["Projection_Source"].eq("manual override")
        out = out.loc[keep].copy()
    return out.reset_index(drop=True), accepted.reset_index(drop=True)


def market_projection_review(report):
    if not len(report):
        return report
    view = report.copy()
    view["Delta"] = view["Market projection"] - view["Yahoo projection"]
    view["Delta %"] = 100 * view["Delta"] / view["Yahoo projection"].replace(0, np.nan)
    return view.sort_values(
        ["Market accepted", "Market projection", "Yahoo projection"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


# ============================================================================
# NOTEBOOK CELL 9 - nflverse depth chart and roster-availability cross-check
# ============================================================================
NFLVERSE_RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Yahoo and nflverse disagree on exactly one abbreviation on a normal slate. Without
# this alias every Jacksonville player fails to match and the availability filter
# silently switches itself off for that team.
YAHOO_TO_NFLVERSE_TEAM = {"JAC": "JAX"}

# nflverse lists fullbacks separately; Yahoo prices them as running backs.
NFLVERSE_POSITION_ALIASES = {"FB": "RB", "HB": "RB"}


def rerank_aliased_positions(offense):
    """Place fullbacks behind their team's running backs before the ranks are used.

    nflverse ranks fullbacks within fullbacks, so a blocking FB1 naively becomes RB1
    and inherits RB1's 1.00 mean multiplier and low 0.676 CV - strictly worse than the
    salary heuristic it replaced. Offsetting by the deepest running back keeps the
    alias useful without promoting a fullback over the actual starter.
    """
    out = offense.copy()
    is_fullback = out["pos_abb"].eq("FB")
    if not is_fullback.any():
        return out
    deepest = out[out["pos_abb"].eq("RB")].groupby("team")["pos_rank"].max()
    out.loc[is_fullback, "pos_rank"] = (
        out.loc[is_fullback, "team"].map(deepest).fillna(0).astype(int)
        + out.loc[is_fullback, "pos_rank"].astype(int)
    )
    return out

# Roster status codes seen in nflverse weekly rosters.
NFLVERSE_STATUS_MEANING = {
    "ACT": "active",
    "DEV": "practice squad",
    "CUT": "released",
    "RES": "reserve / injured reserve",
    "RET": "retired",
    "EXE": "exempt list",
    "TRC": "reserve / did not report",
    "W04": "waived",
}

_NAME_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NAME_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_person_name(name):
    """Collapse a player name to a join key that survives feed spelling differences.

    Yahoo writes "Velus Jones Jr." and "A.J. Brown"; nflverse writes "Velus Jones" and
    "AJ Brown". Punctuation is replaced by a space so generational suffixes can be
    matched on word boundaries, and only then is all whitespace removed - otherwise
    "a.j. brown" collapses to "a j brown" and never meets "aj brown".
    """
    text = _NAME_NON_ALNUM.sub(" ", str(name).lower())
    text = _NAME_SUFFIXES.sub(" ", text)
    return text.replace(" ", "")


def infer_season(game_time, fallback=None):
    """Infer the NFL season year from a slate kickoff timestamp.

    A season is named for the calendar year it starts in, so January and February
    games belong to the previous season.
    """
    stamp = pd.to_datetime(game_time, errors="coerce", utc=True)
    if pd.isna(stamp):
        return fallback
    return int(stamp.year) if stamp.month >= 3 else int(stamp.year) - 1


def _read_nflverse_csv(dataset, filename, cfg):
    """Download one nflverse release asset, caching it for the current UTC day.

    The .csv.gz assets are used rather than .parquet so the notebook keeps working
    without pyarrow. Files are refreshed daily because nflverse republishes the
    depth chart every morning.
    """
    cache = Path(cfg.nflverse_cache_dir)
    stamp = time.strftime("%Y%m%d", time.gmtime())
    target = cache / f"{stamp}_{filename}"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{NFLVERSE_RELEASE_BASE}/{dataset}/{filename}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 Yahoo-Showdown-Lineup-Lab/3.3"})
        with urlopen(request, timeout=cfg.nflverse_timeout) as response:
            target.write_bytes(response.read())
        for stale in cache.glob(f"*_{filename}"):
            if stale != target:
                stale.unlink(missing_ok=True)
    return pd.read_csv(target, low_memory=False)


def fetch_nflverse_depth_chart(season, cfg=None):
    """Return the most recent depth-chart snapshot for one season, plus its timestamp.

    The 2026 asset is a running log of snapshots rather than one row per week, so the
    latest `dt` is taken. Only the offensive personnel group is kept, and `pos_rank`
    is already the player's rank within his team and position.
    """
    cfg = _cfg(cfg)
    frame = _read_nflverse_csv("depth_charts", f"depth_charts_{season}.csv.gz", cfg)
    if "dt" in frame:
        frame = frame.copy()
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce", utc=True)
        as_of = frame["dt"].max()
        frame = frame[frame["dt"].eq(as_of)]
    else:  # older seasons use a season/week schema
        as_of = None
        if "week" in frame:
            frame = frame[frame["week"].eq(frame["week"].max())]
    offense = frame[frame["pos_abb"].isin(["QB", "RB", "FB", "WR", "TE"])].copy()
    offense = rerank_aliased_positions(offense)
    offense["Position"] = offense["pos_abb"].replace(NFLVERSE_POSITION_ALIASES)
    return offense, as_of


def fetch_nflverse_roster_status(season, cfg=None):
    """Return the latest weekly roster snapshot: who is active, cut, IR, or practice squad."""
    cfg = _cfg(cfg)
    frame = _read_nflverse_csv("weekly_rosters", f"roster_weekly_{season}.csv.gz", cfg)
    if "week" in frame and frame["week"].notna().any():
        frame = frame[frame["week"].eq(frame["week"].max())]
    name_column = next(
        (c for c in ("full_name", "player_name", "football_name") if c in frame), None
    )
    if name_column is None:
        raise ValueError("nflverse roster asset has no recognizable name column.")
    out = frame[[name_column, "team", "status"]].copy()
    out.columns = ["player_name", "team", "status"]
    return out


def _keyed(frame, team_column, name_column):
    """Index a reference frame by team + normalized name, dropping ambiguous keys.

    Two different players on one team who normalize to the same key cannot be told
    apart, so both are removed rather than guessed at.
    """
    out = frame.copy()
    out["_key"] = (
        out[team_column].astype(str).str.upper()
        + "|"
        + out[name_column].map(normalize_person_name)
    )
    counts = out["_key"].value_counts()
    return out[out["_key"].isin(counts[counts.eq(1)].index)].set_index("_key")


def build_nflverse_role_report(players, depth_chart, roster_status, cfg=None):
    """Join a Yahoo player pool to nflverse depth and roster status.

    Pure function: it performs no network access, so the smoke test can exercise the
    matching and disagreement logic offline. Returns one row per Yahoo player with
    the heuristic depth, the nflverse depth, the roster status, and whether each
    lookup actually matched.
    """
    cfg = _cfg(cfg)
    allowed = set(cfg.nflverse_available_status)
    pool = players.copy()
    pool["_team"] = pool["Team"].astype(str).str.upper().replace(YAHOO_TO_NFLVERSE_TEAM)
    pool["_key"] = pool["_team"] + "|" + pool["Name"].map(normalize_person_name)

    depth_key, status_key = None, None
    if depth_chart is not None and len(depth_chart):
        depth_key = _keyed(depth_chart, "team", "player_name")
    if roster_status is not None and len(roster_status):
        status_key = _keyed(roster_status, "team", "player_name")

    rows = []
    columns = zip(
        pool["Name"], pool["Team"], pool["Position"], pool["Salary"],
        pool["Depth_Rank"], pool["_key"],
    )
    for name, team, position, salary, yahoo_depth, key in columns:
        is_defense = position == "DEF"
        nfl_depth, nfl_position, status = None, None, None
        if depth_key is not None and not is_defense and key in depth_key.index:
            match = depth_key.loc[key]
            # Only accept the depth entry if the position agrees with Yahoo's.
            if str(match["Position"]) == str(position):
                nfl_depth = int(match["pos_rank"])
                nfl_position = str(match["pos_abb"])
        if status_key is not None and not is_defense and key in status_key.index:
            status = str(status_key.loc[key]["status"])

        if is_defense:
            available, reason = True, "team defense"
        elif status is None:
            available = not cfg.nflverse_drop_unmatched
            reason = "no roster row" if available else "no roster row (dropped)"
        else:
            available = status in allowed
            reason = NFLVERSE_STATUS_MEANING.get(status, status)

        rows.append({
            "Player": name,
            "Team": team,
            "Position": position,
            "Salary": float(salary),
            "Yahoo depth": int(yahoo_depth),
            "nflverse depth": nfl_depth,
            "nflverse position": nfl_position,
            "Depth matched": nfl_depth is not None,
            "Depth agrees": (nfl_depth is not None and int(nfl_depth) == int(yahoo_depth)),
            "Roster status": status,
            "Status meaning": reason,
            "Available": available,
        })
    report = pd.DataFrame(rows)
    report["Depth change"] = np.where(
        report["Depth matched"] & ~report["Depth agrees"],
        report["Yahoo depth"].astype(str) + " -> " + report["nflverse depth"].astype("Int64").astype(str),
        "",
    )
    return report


def load_nflverse_reference(players, season, cfg=None):
    """Fetch depth chart and roster status, degrading to None on any failure.

    A network problem must never take down a lineup build, so every failure is
    reported and the run continues on the salary-based heuristic alone.
    """
    cfg = _cfg(cfg)
    depth_chart, as_of, roster_status, notes = None, None, None, []
    try:
        depth_chart, as_of = fetch_nflverse_depth_chart(season, cfg)
        notes.append(
            f"depth chart {season}: {len(depth_chart):,} offensive rows"
            + (f", snapshot {as_of:%Y-%m-%d %H:%M} UTC" if as_of is not None else "")
        )
    except Exception as exc:  # network, 404 for an unstarted season, schema drift
        notes.append(f"depth chart unavailable ({type(exc).__name__}: {exc}); keeping heuristic depth")
    try:
        roster_status = fetch_nflverse_roster_status(season, cfg)
        notes.append(f"roster status {season}: {len(roster_status):,} rows")
    except Exception as exc:
        notes.append(f"roster status unavailable ({type(exc).__name__}: {exc}); no availability filter")
    return depth_chart, roster_status, as_of, notes


def apply_nflverse_depth(players, report, cfg=None):
    """Replace heuristic depth ranks with the published depth chart.

    A manual DEPTH_OVERRIDES entry always wins: the point of that dict is to encode
    information the user has and the feed does not. Everything else that matched is
    overwritten, because a salary-ordered guess is strictly weaker evidence than a
    published depth chart.
    """
    cfg = _cfg(cfg)
    out = players.copy()
    if not cfg.nflverse_apply_depth or report is None or report.empty:
        return out, pd.DataFrame()
    usable = report[report["Depth matched"] & ~report["Depth agrees"]]
    manual = set(out.loc[out["Depth_Source"].eq("manual override"), "Name"])
    applied = usable[~usable["Player"].isin(manual)]
    lookup = dict(zip(applied["Player"], applied["nflverse depth"]))
    for name, depth in lookup.items():
        mask = out["Name"].eq(name)
        out.loc[mask, "Depth_Rank"] = max(1, int(depth))
        out.loc[mask, "Depth_Source"] = "nflverse depth chart"
    confirmed = report[report["Depth agrees"] & ~report["Player"].isin(manual)]["Player"]
    out.loc[out["Name"].isin(confirmed) & out["Depth_Source"].eq("projection heuristic"),
            "Depth_Source"] = "nflverse depth chart (confirmed)"
    skipped = usable[usable["Player"].isin(manual)]
    if len(skipped):
        warnings.warn(
            "Kept your manual DEPTH_OVERRIDES over the nflverse depth chart for: "
            + ", ".join(skipped["Player"])
        )
    return out, applied


def apply_nflverse_availability(players, report, cfg=None):
    """Hard-drop players the published roster says are not available.

    This is deliberately a removal rather than a projection haircut. A player on
    injured reserve has no distribution to simulate, and leaving him in the pool with
    a reduced mean would still let a long lognormal tail put him in a lineup.
    """
    cfg = _cfg(cfg)
    if not cfg.nflverse_availability_filter or report is None or report.empty:
        return players.reset_index(drop=True), pd.DataFrame()
    blocked = report[~report["Available"]]
    if blocked.empty:
        return players.reset_index(drop=True), blocked
    kept = players[~players["Name"].isin(set(blocked["Player"]))].reset_index(drop=True)
    return kept, blocked[["Player", "Team", "Position", "Salary", "Yahoo depth", "Roster status", "Status meaning"]]


def nflverse_disagreement_view(report):
    """The rows a human should actually look at before trusting the run."""
    interesting = report[
        (~report["Available"])
        | (report["Depth matched"] & ~report["Depth agrees"])
        | (~report["Depth matched"] & report["Position"].ne("DEF"))
    ]
    columns = [
        "Player", "Team", "Position", "Salary", "Yahoo depth", "nflverse depth",
        "Depth change", "Roster status", "Status meaning", "Available",
    ]
    return interesting[columns].sort_values(
        ["Available", "Salary"], ascending=[True, False]
    ).reset_index(drop=True)


# ============================================================================
# NOTEBOOK CELL 11 - Correlated outcome model and calibrated CVs
# ============================================================================
# Direct fitted forecast-error CV. TE4+ is held at the TE3 value because the
# small historical decline was not meaningful and a monotone risk curve is safer.
# QB3 had only 23 games, so QB2 is the fallback if backup QBs are explicitly kept.
CALIBRATED_CV = {
    "QB": {1: 0.488, 2: 0.922, 3: 0.922, 4: 0.922},
    "RB": {1: 0.676, 2: 0.921, 3: 1.154, 4: 1.207},
    "WR": {1: 0.695, 2: 0.814, 3: 1.042, 4: 1.323},
    "TE": {1: 0.885, 2: 1.324, 3: 1.658, 4: 1.658},
    "DEF": {1: 0.950, 2: 0.950, 3: 0.950, 4: 0.950},
}

QB_WR_CORR = {1: 0.217, 2: 0.171, 3: 0.129, 4: 0.076}
QB_TE_CORR = {1: 0.152, 2: 0.085, 3: 0.072, 4: 0.060}
RB_OWN_DEF_CORR = {1: 0.057, 2: 0.015, 3: 0.000, 4: 0.000}
OPPONENT_DEF_CORR = {
    "QB": {1: -0.241, 2: 0.000, 3: 0.000, 4: 0.000},
    "RB": {1: -0.172, 2: -0.104, 3: -0.069, 4: 0.000},
    "WR": {1: -0.112, 2: -0.100, 3: -0.029, 4: -0.052},
    "TE": {1: -0.059, 2: -0.013, 3: -0.021, 4: 0.000},
}
SAME_TEAM_OTHER_CORR = {
    ("QB", "QB"): -0.200,
    ("QB", "DEF"): -0.061,
    ("RB", "RB"): 0.013,
    ("RB", "WR"): -0.002,
    ("RB", "TE"): -0.010,
    ("WR", "WR"): 0.011,
    ("WR", "TE"): 0.009,
    ("TE", "TE"): -0.004,
    ("WR", "DEF"): -0.044,
    ("TE", "DEF"): -0.040,
}
OPPOSING_OFFENSE_CORR = {
    ("QB", "QB"): 0.175,
    ("QB", "RB"): 0.013,
    ("QB", "WR"): 0.042,
    ("QB", "TE"): 0.048,
    ("RB", "RB"): 0.019,
    ("RB", "WR"): 0.011,
    ("RB", "TE"): -0.006,
    ("WR", "WR"): 0.021,
    ("WR", "TE"): 0.016,
    ("TE", "TE"): 0.016,
}
POSITION_ORDER = {position: index for index, position in enumerate(VALID_POSITIONS)}


def _position_pair(first, second):
    """Return a stable order for symmetric position-pair lookup tables."""
    return tuple(sorted((first, second), key=lambda value: POSITION_ORDER[value]))


def _calibrated_cv(position, depth):
    """Return fitted total forecast-error CV for one position/depth bucket."""
    return CALIBRATED_CV[position][_depth_bucket(depth)]


def target_score_correlation(a, b):
    """Return the fitted fantasy-score correlation for a player pair.

    Depth-specific estimates are used where the historical results showed a clear
    gradient. Near-zero skill-player relationships are retained near zero instead
    of being forced upward by a shared team factor. Style overrides only affect
    QB-RB: the calibration did not separately identify role styles, so the
    pass-catching value is deliberately modest rather than presented as fitted.
    """
    pa, pb = a["Position"], b["Position"]
    same_team = a["Team"] == b["Team"]
    pair = _position_pair(pa, pb)

    if same_team:
        if pair == ("QB", "WR"):
            receiver = a if pa == "WR" else b
            return QB_WR_CORR[_depth_bucket(receiver["Depth_Rank"])]
        if pair == ("QB", "TE"):
            receiver = a if pa == "TE" else b
            return QB_TE_CORR[_depth_bucket(receiver["Depth_Rank"])]
        if pair == ("QB", "RB"):
            back = a if pa == "RB" else b
            return 0.080 if back.get("Player_Style", "standard") == "pass_catching_rb" else 0.020
        if pair == ("RB", "DEF"):
            back = a if pa == "RB" else b
            return RB_OWN_DEF_CORR[_depth_bucket(back["Depth_Rank"])]
        return SAME_TEAM_OTHER_CORR.get(pair, 0.0)

    if "DEF" in pair:
        if pa == pb == "DEF":
            return 0.0
        offense = b if pa == "DEF" else a
        return OPPONENT_DEF_CORR[offense["Position"]][
            _depth_bucket(offense["Depth_Rank"])
        ]
    return OPPOSING_OFFENSE_CORR.get(pair, 0.0)


def requested_score_correlation_matrix(players):
    """Build the symmetric matrix of historical score-correlation targets."""
    matrix = np.eye(len(players), dtype=np.float64)
    records = [players.iloc[index] for index in range(len(players))]
    for first, second in itertools.combinations(range(len(players)), 2):
        value = target_score_correlation(records[first], records[second])
        matrix[first, second] = matrix[second, first] = float(value)
    return matrix


def lognormal_correlation_bounds(cv):
    """Return the attainable score-correlation range for each lognormal pair.

    Two mean-preserving lognormals with coefficients of variation cv_i, cv_j cannot
    reach every correlation in [-1, 1]. With s_i = sqrt(log(1 + cv_i^2)), the score
    correlation is (exp(rho * s_i * s_j) - 1) / (cv_i * cv_j), so it is bounded by
    rho = -1 and rho = +1. High-CV deep-role players have a floor well above -1:
    a WR4/TE4 pair cannot be more negatively correlated than about -0.31.
    """
    sigma = np.sqrt(np.log1p(np.square(cv)))
    outer_sigma = np.outer(sigma, sigma)
    denominator = np.outer(cv, cv)
    with np.errstate(divide="ignore", invalid="ignore"):
        low = np.where(denominator > 0, np.expm1(-outer_sigma) / denominator, -1.0)
        high = np.where(denominator > 0, np.expm1(outer_sigma) / denominator, 1.0)
    return low, high


def score_to_lognormal_latent(score_corr, cv, report=None):
    """Convert desired lognormal score correlations to Gaussian correlations.

    For mean-preserving lognormal variables, score correlation is not the same as
    latent-normal correlation. Solving the closed-form covariance relationship
    before PSD repair keeps the simulated score matrix close to the historical
    targets despite large, depth-dependent CVs.

    v3.2: a target outside the attainable range above used to fall through to
    log(1e-8) and then clip to a latent -0.95, quietly delivering a different
    correlation than requested. Targets are now clipped to the attainable bound
    first, and every clipped pair is counted and reported.
    """
    low, high = lognormal_correlation_bounds(cv)
    margin = 1e-6
    feasible = np.clip(score_corr, low + margin, high - margin)
    np.fill_diagonal(feasible, 1.0)
    infeasible = np.abs(feasible - score_corr) > 1e-9
    np.fill_diagonal(infeasible, False)

    sigma = np.sqrt(np.log1p(np.square(cv)))
    latent = np.eye(len(cv), dtype=np.float64)
    for first, second in itertools.combinations(range(len(cv)), 2):
        argument = 1.0 + feasible[first, second] * cv[first] * cv[second]
        denominator = sigma[first] * sigma[second]
        value = math.log(argument) / denominator if denominator > 0 else 0.0
        latent[first, second] = latent[second, first] = float(np.clip(value, -0.999, 0.999))

    if report is not None:
        report["infeasible_pairs"] = int(infeasible.sum() // 2)
        report["max_infeasible_shift"] = (
            float(np.abs(feasible - score_corr).max()) if infeasible.any() else 0.0
        )
    if infeasible.any():
        warnings.warn(
            f"{int(infeasible.sum() // 2)} correlation target(s) were outside the range "
            "two lognormals with these coefficients of variation can attain and were "
            "clipped to the nearest attainable value (largest shift "
            f"{float(np.abs(feasible - score_corr).max()):.3f})."
        )
    return latent


def repair_correlation_matrix(matrix, eigenvalue_floor=1e-8):
    """Project a symmetric matrix to a numerically safe PSD correlation matrix.

    Independently fitted pairwise targets are not guaranteed to form a valid joint
    distribution. Eigenvalue clipping followed by diagonal renormalization makes
    simulation possible while changing the requested structure as little as this
    lightweight, dependency-free repair permits.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = eigenvectors @ np.diag(np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(repaired), eigenvalue_floor))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def lognormal_score_correlation(latent_corr, cv):
    """Return the score correlation implied by a latent lognormal matrix."""
    sigma = np.sqrt(np.log1p(np.square(cv)))
    numerator = np.expm1(latent_corr * np.outer(sigma, sigma))
    denominator = np.outer(cv, cv)
    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    score = np.clip(0.5 * (score + score.T), -0.999, 0.999)
    np.fill_diagonal(score, 1.0)
    return score


def build_correlation_model(players):
    """Build requested, latent, repaired, and effective score correlations."""
    teams = list(players["Team"].drop_duplicates())
    if len(teams) != 2:
        raise ValueError(f"Expected exactly two teams, found {teams}")
    cv = np.array([
        _calibrated_cv(row.Position, row.Depth_Rank) for row in players.itertuples()
    ])
    requested_score = requested_score_correlation_matrix(players)
    feasibility = {}
    requested_latent = score_to_lognormal_latent(requested_score, cv, report=feasibility)
    latent_corr = repair_correlation_matrix(requested_latent)
    effective_score = lognormal_score_correlation(latent_corr, cv)
    off_diagonal = ~np.eye(len(players), dtype=bool)
    max_adjustment = float(
        np.max(np.abs(effective_score[off_diagonal] - requested_score[off_diagonal]))
    ) if len(players) > 1 else 0.0
    return {
        "cv": cv,
        "target_score_corr": requested_score,
        "requested_latent_corr": requested_latent,
        "latent_corr": latent_corr,
        "score_corr": effective_score,
        "psd_max_score_adjustment": max_adjustment,
        "infeasible_pairs": feasibility.get("infeasible_pairs", 0),
        "max_infeasible_shift": feasibility.get("max_infeasible_shift", 0.0),
    }


def simulate_player_outcomes(players, cfg=None):
    """Simulate mean-preserving lognormal scores under the repaired joint model."""
    cfg = _cfg(cfg)
    model = build_correlation_model(players)
    eigenvalues, eigenvectors = np.linalg.eigh(model["latent_corr"])
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(cfg.random_seed)
    latent = rng.normal(size=(cfg.simulations, len(players))) @ root.T
    sigma = np.sqrt(np.log1p(np.square(model["cv"])))
    means = players["Projected_FP"].to_numpy(float)
    outcomes = means * np.exp(latent * sigma - 0.5 * np.square(sigma))
    return outcomes.astype(np.float32), model


CALIBRATION_SANITY_RELATIONSHIPS = [
    "Same team QB-WR",
    "Same team QB-TE",
    "Same team QB-RB",
    "Same team RB-DEF",
    "Same team WR-WR",
    "Same team WR-TE",
    "Opponent QB-DEF",
    "Opponent WR-DEF",
    "Opponent TE-DEF",
    "Opponent RB-DEF",
    "Opposing QB-QB",
]


def _pair_relationship(a, b):
    pa, pb = a["Position"], b["Position"]
    same = a["Team"] == b["Team"]
    pair = {pa, pb}
    if same:
        if pair == {"QB", "WR"}: return "Same team QB-WR"
        if pair == {"QB", "TE"}: return "Same team QB-TE"
        if pair == {"QB", "RB"}: return "Same team QB-RB"
        if pair == {"RB", "DEF"}: return "Same team RB-DEF"
        if pa == pb == "WR": return "Same team WR-WR"
        if pair == {"WR", "TE"}: return "Same team WR-TE"
        ordered = _position_pair(pa, pb)
        return f"Same team {ordered[0]}-{ordered[1]}"
    if "DEF" in pair:
        offense = pb if pa == "DEF" else pa
        return f"Opponent {offense}-DEF"
    ordered = _position_pair(pa, pb)
    return f"Opposing {ordered[0]}-{ordered[1]}"


def correlation_sanity_report(players, outcomes, factor_model):
    """Compare historical targets, PSD-repaired model values, and simulations."""
    empirical = np.corrcoef(outcomes, rowvar=False)
    target = factor_model["target_score_corr"]
    model_score = factor_model["score_corr"]
    latent = factor_model["latent_corr"]
    records = [players.iloc[index] for index in range(len(players))]
    rows = []
    for i, j in itertools.combinations(range(len(players)), 2):
        a, b = records[i], records[j]
        rows.append({
            "Relationship": _pair_relationship(a, b),
            "Player A": a["Name"],
            "Player B": b["Name"],
            "Target score corr": target[i, j],
            "PSD model score corr": model_score[i, j],
            "Latent corr": latent[i, j],
            "Simulated score corr": empirical[i, j],
            "PSD target gap": model_score[i, j] - target[i, j],
            "Simulation target gap": empirical[i, j] - target[i, j],
        })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("Relationship")
        .agg(
            Pairs=("Simulated score corr", "size"),
            Target=("Target score corr", "median"),
            Model=("PSD model score corr", "median"),
            Simulated=("Simulated score corr", "median"),
            Max_PSD_Gap=("PSD target gap", lambda values: np.abs(values).max()),
            Max_Sim_Gap=("Simulation target gap", lambda values: np.abs(values).max()),
        )
        .reset_index()
    )
    summary["Sanity"] = np.where(
        summary["Max_PSD_Gap"].le(0.04) & summary["Max_Sim_Gap"].le(0.08),
        "PASS",
        "REVIEW",
    )
    for column in ["Target", "Model", "Simulated", "Max_PSD_Gap", "Max_Sim_Gap"]:
        summary[column] = summary[column].round(3)
    return summary, detail


def analytic_covariance(players, factor_model):
    """Convert model means, fitted CVs, and effective score correlations to covariance."""
    means = players["Projected_FP"].to_numpy(float)
    std = means * factor_model["cv"]
    return np.outer(std, std) * factor_model["score_corr"]


def marginal_sanity_report(players, outcomes):
    """Confirm the simulator reproduced each player's intended mean and CV.

    The candidate screen ranks lineups with the analytic covariance while the final
    ranking uses the simulated draws. If those two descriptions of the same model
    ever diverged, the screen would silently discard good lineups. This checks the
    marginals; `analytic_vs_simulated_lineup_check` checks the joint behaviour.
    """
    intended_mean = players["Projected_FP"].to_numpy(float)
    intended_cv = np.array([
        _calibrated_cv(row.Position, row.Depth_Rank) for row in players.itertuples()
    ])
    simulated_mean = outcomes.mean(axis=0).astype(float)
    simulated_cv = outcomes.std(axis=0).astype(float) / np.maximum(simulated_mean, 1e-9)
    report = pd.DataFrame({
        "Player": players["Name"].to_numpy(),
        "Position": players["Position"].to_numpy(),
        "Depth": players["Depth_Rank"].to_numpy(),
        "Intended mean": np.round(intended_mean, 3),
        "Simulated mean": np.round(simulated_mean, 3),
        "Mean error %": np.round(100 * (simulated_mean / np.maximum(intended_mean, 1e-9) - 1), 2),
        "Intended CV": np.round(intended_cv, 3),
        "Simulated CV": np.round(simulated_cv, 3),
        "CV error %": np.round(100 * (simulated_cv / np.maximum(intended_cv, 1e-9) - 1), 2),
    })
    report["Sanity"] = np.where(
        report["Mean error %"].abs().le(3.0) & report["CV error %"].abs().le(8.0),
        "PASS",
        "REVIEW",
    )
    return report


# ============================================================================
# NOTEBOOK CELL 13 - Candidate lineup generation and shared-scenario scoring
# ============================================================================
# Guard against an enumeration the notebook cannot hold in memory. C(36,5) is
# 376,992; the budget below leaves room to raise max_enumeration_players to ~55.
MAX_COMBINATION_BUDGET = 30_000_000


def _as_range_indexed(players):
    """Every lineup id in this notebook is positional, so force a 0..n-1 index."""
    index = players.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return players
    return players.reset_index(drop=True)


def _lineup_variance(cov, combos, superstar_slot, chunk):
    """Variance of a Yahoo lineup, vectorized over many rosters at once.

    With weights w = 1 + 0.5 * e_s (the Superstar slot carries 1.5x),
        w' C w = sum(C) + rowsum_s(C) + 0.25 * C_ss
    over the 5x5 submatrix of the roster, which avoids building one quadratic form
    per candidate in Python.
    """
    total = len(combos)
    variance = np.empty(total, dtype=np.float64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = combos[start:stop]
        sub = cov[block[:, :, None], block[:, None, :]]
        rows = np.arange(stop - start)
        slots = superstar_slot[start:stop]
        variance[start:stop] = (
            sub.sum(axis=(1, 2))
            + sub.sum(axis=2)[rows, slots]
            + 0.25 * np.diagonal(sub, axis1=1, axis2=2)[rows, slots]
        )
    return np.maximum(variance, 0.0)


def enumerate_candidate_lineups(players, salary_cap, covariance, cfg=None, chunk=150_000):
    """Enumerate Yahoo-valid rosters and retain strong mean/ceiling candidates.

    v3.2 replaces the per-combination Python loop and two heaps with array
    operations and `np.argpartition`. On a 36-player pool this runs about 40x
    faster and returns the same candidate set to floating-point tolerance.
    """
    cfg = _cfg(cfg)
    players = _as_range_indexed(players)
    n = len(players)
    size = int(cfg.lineup_size)
    if n < size:
        raise ValueError(f"Only {n} eligible players remain; need {size}.")

    problem = roster_feasibility_error(players, size)
    if problem:
        raise ValueError(problem)

    total_combinations = math.comb(n, size)
    if total_combinations > MAX_COMBINATION_BUDGET:
        raise ValueError(
            f"{total_combinations:,} rosters exceed the {MAX_COMBINATION_BUDGET:,} "
            "enumeration budget. Lower Settings.max_enumeration_players."
        )

    salary = players["Salary"].to_numpy(float)
    projection = players["Projected_FP"].to_numpy(float)
    positions = players["Position"].to_numpy(str)
    team_values = players["Team"].to_numpy(str)
    teams = list(pd.unique(team_values))
    is_skill = positions != "DEF"
    covariance = np.ascontiguousarray(covariance, dtype=np.float64)
    min_salary = salary_cap * cfg.min_salary_used_pct

    combos = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), size)),
        dtype=np.int32,
        count=total_combinations * size,
    ).reshape(total_combinations, size)

    combo_salary = salary[combos].sum(axis=1)
    keep_mask = (combo_salary <= salary_cap) & (combo_salary >= min_salary)
    # Yahoo single-game rule: at least one non-defense athlete from each team.
    for team in teams:
        keep_mask &= ((team_values[combos] == team) & is_skill[combos]).any(axis=1)
    for position, (low, high) in (cfg.position_limits or {}).items():
        counts = (positions[combos] == position).sum(axis=1)
        keep_mask &= (counts >= low) & (counts <= high)

    combos = combos[keep_mask]
    combo_salary = combo_salary[keep_mask]
    valid_rosters = int(len(combos))
    if valid_rosters == 0:
        raise ValueError(
            "No valid rosters. Review exclusions, salary floor, cap, or optional position limits."
        )

    # Score every (roster, Superstar) pair: shape (rosters, lineup_size).
    base_projection = projection[combos].sum(axis=1)
    expected = base_projection[:, None] + 0.5 * projection[combos]
    ceiling = np.empty_like(expected)
    for slot in range(size):
        slots = np.full(valid_rosters, slot, dtype=np.int64)
        variance = _lineup_variance(covariance, combos, slots, chunk)
        ceiling[:, slot] = expected[:, slot] + cfg.candidate_ceiling_weight * np.sqrt(variance)

    flat_expected = expected.ravel()
    flat_ceiling = ceiling.ravel()
    population = flat_ceiling.size

    def top_indices(values, count):
        count = min(int(count), population)
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        return np.argpartition(values, population - count)[population - count:]

    selected = np.union1d(
        top_indices(flat_ceiling, cfg.max_candidate_lineups),
        top_indices(flat_expected, cfg.mean_candidate_reserve),
    )
    roster_index, slot_index = np.divmod(selected, size)
    kept = combos[roster_index]
    variance = _lineup_variance(covariance, kept, slot_index, chunk)

    candidates = pd.DataFrame({
        "Player_Ids": [tuple(int(value) for value in row) for row in kept],
        "Superstar_Id": kept[np.arange(len(kept)), slot_index].astype(int),
        "Salary": combo_salary[roster_index],
        "Expected_FP": flat_expected[selected],
        "Analytic_SD": np.sqrt(variance),
        "Pre_Sim_Score": flat_ceiling[selected],
    })
    return candidates.reset_index(drop=True), valid_rosters


def _lineup_arrays(candidates, lineup_size):
    """Materialize lineup ids once instead of re-parsing them per batch."""
    ids = np.empty((len(candidates), lineup_size), dtype=np.int32)
    for row, value in enumerate(candidates["Player_Ids"]):
        ids[row] = value
    return ids, candidates["Superstar_Id"].to_numpy(np.int32)


def _weight_matrix(ids, superstars, start, stop, player_count):
    """Build the (players x candidates) 1.0/1.5 weight block for one batch."""
    weights = np.zeros((player_count, stop - start), dtype=np.float32)
    columns = np.arange(stop - start)
    weights[ids[start:stop].T, columns] = 1.0
    weights[superstars[start:stop], columns] += 0.5
    return weights


def _tournament_score(ceiling_p90, near_optimal, sim_mean, sim_sd):
    """Blend the four ranking inputs on a common z-scale."""
    def zscore(values):
        values = np.asarray(values, dtype=float)
        scale = values.std()
        return (values - values.mean()) / scale if scale > 1e-12 else np.zeros_like(values)

    return (
        0.40 * zscore(ceiling_p90)
        + 0.25 * zscore(near_optimal)
        + 0.25 * zscore(sim_mean)
        + 0.10 * zscore(sim_sd)
    )


def score_candidates_shared_scenarios(candidates, outcomes, cfg=None, batch_size=512):
    """Two passes: distribution summaries, then performance vs each scenario's optimum.

    v3.2 also computes every metric separately on each half of the shared scenarios.
    Those half-sample copies cost almost nothing because the batch scores are already
    in memory, and they let `reliability_report` measure how much of the final ranking
    is signal rather than Monte Carlo noise.
    """
    cfg = _cfg(cfg)
    n_candidates = len(candidates)
    n_scenarios, n_players = outcomes.shape
    half = n_scenarios // 2
    halves = {"": slice(None), "_A": slice(0, half), "_B": slice(half, 2 * half)}
    ids, superstars = _lineup_arrays(candidates, int(cfg.lineup_size))

    names = ["Sim_Mean", "Sim_SD", "Floor_P25", "Ceiling_P90", "Ceiling_P95"]
    metrics = {f"{name}{tag}": np.zeros(n_candidates) for name in names for tag in halves}
    scenario_best = np.full(n_scenarios, -np.inf, dtype=np.float32)

    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        scores = outcomes @ _weight_matrix(ids, superstars, start, stop, n_players)
        for tag, window in halves.items():
            block = scores[window]
            metrics[f"Sim_Mean{tag}"][start:stop] = block.mean(axis=0)
            metrics[f"Sim_SD{tag}"][start:stop] = block.std(axis=0)
            quantiles = np.quantile(block, [0.25, 0.90, 0.95], axis=0)
            metrics[f"Floor_P25{tag}"][start:stop] = quantiles[0]
            metrics[f"Ceiling_P90{tag}"][start:stop] = quantiles[1]
            metrics[f"Ceiling_P95{tag}"][start:stop] = quantiles[2]
        np.maximum(scenario_best, scores.max(axis=1), out=scenario_best)

    rates = {f"{name}{tag}": np.zeros(n_candidates)
             for name in ("Near_Optimal_Rate", "Win_Rate") for tag in halves}
    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        scores = outcomes @ _weight_matrix(ids, superstars, start, stop, n_players)
        near = scores >= (scenario_best[:, None] * cfg.near_optimal_ratio)
        won = np.isclose(scores, scenario_best[:, None], rtol=1e-6, atol=1e-5)
        for tag, window in halves.items():
            rates[f"Near_Optimal_Rate{tag}"][start:stop] = near[window].mean(axis=0)
            rates[f"Win_Rate{tag}"][start:stop] = won[window].mean(axis=0)

    scored = candidates.copy()
    for name, values in {**metrics, **rates}.items():
        scored[name] = values

    # Binomial Monte Carlo standard errors. Both rates are tiny (a top lineup is
    # near-optimal in well under 1% of scenarios), so their sampling error is a
    # large fraction of the spread between candidates. Report it beside the value.
    for name in ("Near_Optimal_Rate", "Win_Rate"):
        rate = scored[name].to_numpy()
        scored[f"{name}_SE"] = np.sqrt(np.maximum(rate * (1 - rate), 0.0) / max(n_scenarios, 1))

    for tag in halves:
        scored[f"Tournament_Score{tag}"] = _tournament_score(
            scored[f"Ceiling_P90{tag}"], scored[f"Near_Optimal_Rate{tag}"],
            scored[f"Sim_Mean{tag}"], scored[f"Sim_SD{tag}"],
        )
    scored.attrs["simulations"] = n_scenarios
    return scored.sort_values("Tournament_Score", ascending=False).reset_index(drop=True)


RELIABILITY_METRICS = [
    "Sim_Mean", "Floor_P25", "Ceiling_P90", "Ceiling_P95",
    "Sim_SD", "Near_Optimal_Rate", "Win_Rate", "Tournament_Score",
]


def reliability_report(scored, top_n=20):
    """Estimate how much of each ranking metric is signal rather than sampling noise.

    Each metric was computed twice, on two independent halves of the shared
    scenarios. Their rank correlation is the reliability at half the sample size;
    the Spearman-Brown formula 2r / (1 + r) up-corrects it to the full run. A metric
    near 1.0 would rank the same candidates the same way under a different seed; a
    metric near 0 is re-ranking noise and should not drive lineup selection.
    """
    rows = []
    for metric in RELIABILITY_METRICS:
        left, right = f"{metric}_A", f"{metric}_B"
        if left not in scored or right not in scored:
            continue
        a, b = scored[left], scored[right]
        if a.std() < 1e-12 or b.std() < 1e-12:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(a.rank(), b.rank())[0, 1])
        full = 2 * correlation / (1 + correlation) if correlation > -1 else float("nan")
        overlap = len(set(a.nlargest(top_n).index) & set(b.nlargest(top_n).index))
        rows.append({
            "Metric": metric,
            "Split-half rank corr": round(correlation, 3),
            "Full-run reliability": round(full, 3),
            f"Top-{top_n} overlap": f"{overlap}/{top_n}",
            "Verdict": (
                "stable" if full >= 0.90 else
                "usable" if full >= 0.75 else
                "NOISY - do not rank on this alone"
            ),
        })
    return pd.DataFrame(rows)


def analytic_vs_simulated_lineup_check(scored, sample=2_000, seed=0):
    """Verify the analytic screen agrees with the simulation it is screening for.

    Candidates are enumerated and pruned with the analytic covariance, then ranked
    with the simulated draws. If the two disagreed, the pruning step would be
    discarding lineups the final objective would have liked. This compares the two
    descriptions on the retained candidates.
    """
    rng = np.random.default_rng(seed)
    take = min(int(sample), len(scored))
    rows = scored.iloc[rng.choice(len(scored), size=take, replace=False)]
    mean_error = (rows["Sim_Mean"] - rows["Expected_FP"]).abs() / rows["Expected_FP"].abs().clip(lower=1e-9)
    sd_error = (rows["Sim_SD"] - rows["Analytic_SD"]).abs() / rows["Analytic_SD"].abs().clip(lower=1e-9)
    return pd.DataFrame([{
        "Checked candidates": take,
        "Mean: median abs error %": round(100 * float(mean_error.median()), 3),
        "Mean: worst abs error %": round(100 * float(mean_error.max()), 3),
        "SD: median abs error %": round(100 * float(sd_error.median()), 3),
        "SD: worst abs error %": round(100 * float(sd_error.max()), 3),
        "Mean rank corr": round(float(np.corrcoef(rows["Expected_FP"].rank(), rows["Sim_Mean"].rank())[0, 1]), 4),
        "SD rank corr": round(float(np.corrcoef(rows["Analytic_SD"].rank(), rows["Sim_SD"].rank())[0, 1]), 4),
        "Sanity": "PASS" if float(mean_error.max()) < 0.05 and float(sd_error.max()) < 0.12 else "REVIEW",
    }])


# ============================================================================
# NOTEBOOK CELL 15 - Strongest lineup, tournament portfolio, and exports
# ============================================================================
def select_strongest_lineups(scored, count=10):
    """Rank by analytic expected points, which is seed-independent.

    The tie-breakers are simulated, so v3.2 adds Player_Ids as a final deterministic
    key. Without it two rosters with identical expected points could swap places
    purely because of the random seed.
    """
    ordered = scored.copy()
    ordered["_tiebreak"] = [tuple(ids) for ids in ordered["Player_Ids"]]
    ordered = ordered.sort_values(
        ["Expected_FP", "Floor_P25", "Ceiling_P90", "_tiebreak", "Superstar_Id"],
        ascending=[False, False, False, True, True],
    )
    return ordered.drop(columns="_tiebreak").head(count).reset_index(drop=True)


def select_tournament_portfolio(scored, cfg=None):
    cfg = _cfg(cfg)
    target = cfg.tournament_lineups
    max_player_count = max(1, math.ceil(target * cfg.max_player_exposure))
    max_superstar_count = max(1, math.ceil(target * cfg.max_superstar_exposure))
    player_counts = Counter()
    superstar_counts = Counter()
    selected_rows = []
    selected_sets = []

    for idx, candidate in scored.sort_values("Tournament_Score", ascending=False).iterrows():
        ids = tuple(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        candidate_set = set(ids)
        if any(player_counts[player] >= max_player_count for player in ids):
            continue
        if superstar_counts[superstar] >= max_superstar_count:
            continue
        if any(len(candidate_set & prior) > cfg.max_shared_players for prior in selected_sets):
            continue
        selected_rows.append(idx)
        selected_sets.append(candidate_set)
        player_counts.update(ids)
        superstar_counts.update([superstar])
        if len(selected_rows) == target:
            break

    portfolio = scored.loc[selected_rows].copy().reset_index(drop=True)
    if len(portfolio) < target:
        warnings.warn(
            f"Built {len(portfolio)} of {target} tournament lineups under the current "
            "exposure/overlap caps. Relax a cap if you need the full count."
        )
    return portfolio


def portfolio_diversity_report(portfolio, cfg=None):
    """Show how different the entries really are.

    On a five-player roster `max_shared_players = 4` allows two entries to differ by
    a single player, which is a legitimate choice for a small field but rarely what
    is wanted from twenty tournament entries. v3.2 defaults to 3; this report states
    the overlap actually achieved so the setting is never invisible.
    """
    cfg = _cfg(cfg)
    sets = [set(ids) for ids in portfolio["Player_Ids"]]
    if len(sets) < 2:
        return pd.DataFrame([{"Lineups": len(sets), "Note": "Too few lineups to compare."}])
    overlaps = [
        len(a & b) for a, b in itertools.combinations(sets, 2)
    ]
    counts = Counter(overlaps)
    size = int(cfg.lineup_size)
    near_duplicate = counts.get(size - 1, 0)
    return pd.DataFrame([{
        "Lineups": len(sets),
        "Distinct players used": len(set().union(*sets)),
        "Mean shared players": round(float(np.mean(overlaps)), 2),
        "Max shared players": int(max(overlaps)),
        "Cap in use": int(cfg.max_shared_players),
        "Pairs at the cap": counts.get(int(cfg.max_shared_players), 0),
        f"Pairs sharing {size - 1} of {size}": near_duplicate,
        "Note": (
            "Near-duplicate entries present; lower Settings.max_shared_players."
            if near_duplicate
            else f"Every pair of entries differs by at least "
                 f"{size - int(max(overlaps))} player(s)."
        ),
    }])


def _player_label(players, player_id):
    """Return a compact label that remains readable in wide ranking tables."""
    player = players.iloc[int(player_id)]
    return f"{player['Name']} ({player['Position']}-{player['Team']})"


def _lineup_team_split(players, ids):
    """Describe roster concentration without implying that a 3-2 split is required."""
    counts = players.iloc[list(ids)]["Team"].value_counts()
    return " / ".join(f"{team} {int(count)}" for team, count in counts.items())


def _lineup_construction(players, ids):
    """Summarize actual QB stacks and game-stack structure in plain language."""
    selected = players.iloc[list(ids)]
    notes = []
    qbs = selected[selected["Position"].eq("QB")]
    for _, quarterback in qbs.iterrows():
        receivers = selected[
            selected["Team"].eq(quarterback["Team"])
            & selected["Position"].isin(["WR", "TE"])
        ]
        if len(receivers):
            notes.append(f"{quarterback['Team']} QB + {len(receivers)} WR/TE")
    if len(qbs) == 2:
        notes.append("opposing-QB game stack")
    if not notes:
        notes.append("no QB-receiver stack")
    return "; ".join(notes)


def _lineup_risk_notes(players, ids, salary_left, salary_cap):
    """Flag assumptions worth checking; flags are diagnostics, not hard bans."""
    selected = players.iloc[list(ids)]
    notes = []
    deep = selected[
        selected["Depth_Rank"].ge(3) & ~selected["Position"].eq("DEF")
    ]
    if len(deep):
        notes.append("deep role: " + ", ".join(deep["Name"].tolist()))
    zero_history = selected[
        selected["Projection_Source"].astype(str).str.contains("zero/low FPPG")
        & ~selected["Projection_Source"].astype(str).str.startswith("market ")
    ]
    if len(zero_history):
        notes.append("no FPPG history: " + ", ".join(zero_history["Name"].tolist()))
    backups = selected[
        selected["Position"].eq("QB") & selected["Depth_Rank"].gt(1)
    ]
    if len(backups):
        notes.append("verify backup-QB snaps")
    for _, defense in selected[selected["Position"].eq("DEF")].iterrows():
        conflicts = selected[
            selected["Team"].eq(defense["Opponent"])
            & ~selected["Position"].eq("DEF")
        ]
        if len(conflicts):
            notes.append(f"{defense['Team']} DEF faces {len(conflicts)} selected opponent(s)")
    if salary_cap and salary_left >= 0.10 * salary_cap:
        notes.append("10%+ salary unused")
    return "; ".join(dict.fromkeys(notes)) or "no structural flags"


def lineup_summary(lineups, players, label, salary_cap, rank_start=1):
    """Create an entry-oriented ranking table with distinct mean and tail metrics.

    `Scenario-best %` is the fraction of shared simulations in which the lineup tied
    for the top score among retained candidates. It is not contest win probability
    (ownership and the opponent field are not modeled) and, per the reliability
    report, it is the noisiest column here, so its Monte Carlo standard error is
    printed beside it.
    """
    rows = []
    for number, candidate in lineups.reset_index(drop=True).iterrows():
        ids = list(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        flex = sorted(
            [idx for idx in ids if idx != superstar],
            key=lambda idx: float(players.iloc[idx]["Projected_FP"]),
            reverse=True,
        )
        salary_left = float(salary_cap) - float(candidate["Salary"])
        row = {
            "Use": f"{label} #{number + rank_start}",
            "Superstar (1.5x)": _player_label(players, superstar),
        }
        for slot, player_id in enumerate(flex, 1):
            row[f"Flex {slot}"] = _player_label(players, player_id)
        near = 100 * float(candidate["Near_Optimal_Rate"])
        near_se = 100 * float(candidate.get("Near_Optimal_Rate_SE", float("nan")))
        best = 100 * float(candidate["Win_Rate"])
        best_se = 100 * float(candidate.get("Win_Rate_SE", float("nan")))
        row.update({
            "Team split": _lineup_team_split(players, ids),
            "Construction": _lineup_construction(players, ids),
            "Salary used": round(float(candidate["Salary"]), 2),
            "Salary left": round(salary_left, 2),
            "Projected mean": round(float(candidate["Expected_FP"]), 2),
            "Simulated mean": round(float(candidate["Sim_Mean"]), 2),
            "P25 floor": round(float(candidate["Floor_P25"]), 2),
            "P90 ceiling": round(float(candidate["Ceiling_P90"]), 2),
            "P95 ceiling": round(float(candidate["Ceiling_P95"]), 2),
            "Near-optimal %": round(near, 2),
            "Near-optimal +/-": round(near_se, 3),
            "Scenario-best %": round(best, 3),
            "Scenario-best +/-": round(best_se, 3),
            "Tournament score": round(float(candidate["Tournament_Score"]), 3),
            "Review": _lineup_risk_notes(players, ids, salary_left, salary_cap),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def lineup_comparison_view(summary):
    """Keep notebook displays compact while CSV exports retain all diagnostics."""
    columns = [
        "Use", "Superstar (1.5x)", "Flex 1", "Flex 2", "Flex 3", "Flex 4",
        "Salary left", "Projected mean", "P25 floor", "P90 ceiling",
        "Near-optimal %", "Near-optimal +/-", "Review",
    ]
    return summary[[column for column in columns if column in summary.columns]]


def lineup_entry_rows(lineups, players, label):
    """Expand lineups into Yahoo-entry order and expose every modeling input."""
    rows = []
    for number, candidate in lineups.reset_index(drop=True).iterrows():
        ids = list(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        flex = sorted(
            [idx for idx in ids if idx != superstar],
            key=lambda idx: float(players.iloc[idx]["Projected_FP"]),
            reverse=True,
        )
        ordered = [superstar] + flex
        for slot_number, player_id in enumerate(ordered):
            player = players.iloc[player_id]
            multiplier = 1.5 if slot_number == 0 else 1.0
            raw_projection = player.get(
                "Pre_Depth_Projected_FP", player["Projected_FP"]
            )
            rows.append({
                "Use": label,
                "Lineup": number + 1,
                "Yahoo slot": "SUPERSTAR" if slot_number == 0 else f"FLEX {slot_number}",
                "Player": player["Name"],
                "Position": player["Position"],
                "Team": player["Team"],
                "Salary": round(float(player["Salary"]), 2),
                "Raw projection": round(float(raw_projection), 2),
                "Depth mean factor": round(float(player.get("Depth_Mean_Multiplier", 1.0)), 2),
                "Adjusted projection": round(float(player["Projected_FP"]), 2),
                "Lineup multiplier": multiplier,
                "Projected contribution": round(float(player["Projected_FP"]) * multiplier, 2),
                "Calibrated CV": round(_calibrated_cv(player["Position"], player["Depth_Rank"]), 3),
                "Depth rank": int(player["Depth_Rank"]),
                "Depth source": player["Depth_Source"],
            })
    return pd.DataFrame(rows)


def display_best_lineup(strongest, players, salary_cap):
    """Show the recommended single entry before any alternative rankings."""
    if strongest.empty:
        print("No strongest lineup was produced.")
        return
    print("\nRECOMMENDED SINGLE ENTRY - highest adjusted projected mean")
    summary = lineup_summary(strongest.head(1), players, "Recommended", salary_cap)
    entry = lineup_entry_rows(strongest.head(1), players, "recommended")
    display(entry[[
        "Yahoo slot", "Player", "Position", "Team", "Salary",
        "Adjusted projection", "Lineup multiplier", "Projected contribution",
        "Depth rank",
    ]])
    display(summary[[
        "Salary used", "Salary left", "Projected mean", "Simulated mean",
        "P25 floor", "P90 ceiling", "P95 ceiling",
        "Near-optimal %", "Near-optimal +/-", "Scenario-best %", "Scenario-best +/-",
    ]])
    print(f"Construction: {summary.iloc[0]['Construction']}")
    print(f"Review: {summary.iloc[0]['Review']}")


def display_alternatives(strongest, players, salary_cap, count=4):
    """Show a short comparison set without burying the primary recommendation."""
    alternatives = strongest.iloc[1:1 + count]
    if len(alternatives):
        print("\nNEXT-BEST SINGLE-ENTRY ALTERNATIVES")
        summary = lineup_summary(
            alternatives, players, "Single-entry rank", salary_cap, rank_start=2
        )
        display(lineup_comparison_view(summary))


def exposure_report(portfolio, players):
    """Summarize player and Superstar usage across the portfolio.

    Lineup ids are positional, so v3.2 iterates positions rather than index labels.
    The v3.1 version used `players.iterrows()` and silently reported every exposure
    as zero whenever the caller passed a frame whose index was not already 0..n-1.
    """
    total = len(portfolio)
    player_count = Counter()
    superstar_count = Counter()
    for _, candidate in portfolio.iterrows():
        player_count.update(int(value) for value in candidate["Player_Ids"])
        superstar_count.update([int(candidate["Superstar_Id"])])
    rows = []
    for position in range(len(players)):
        player = players.iloc[position]
        rows.append({
            "Player": player["Name"],
            "Position": player["Position"],
            "Team": player["Team"],
            "Lineups": player_count[position],
            "Exposure %": round(100 * player_count[position] / total, 1) if total else 0.0,
            "Superstar lineups": superstar_count[position],
            "Superstar %": round(100 * superstar_count[position] / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["Exposure %", "Superstar %", "Player"], ascending=[False, False, True]
    ).reset_index(drop=True)


def export_results(
    output_dir, strongest, portfolio, players, corr_summary, corr_detail, salary_cap,
    extra_frames=None,
):
    """Write both readable rankings and exact Yahoo-entry rows to one ZIP bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strongest_summary = lineup_summary(strongest, players, "Strongest", salary_cap)
    tournament_summary = lineup_summary(portfolio, players, "Tournament", salary_cap)
    # v3.2 exports every ranked single-entry alternative, not only the top one.
    strongest_entries = lineup_entry_rows(strongest, players, "strongest")
    tournament_entries = lineup_entry_rows(portfolio, players, "tournament")
    exposures = exposure_report(portfolio, players)

    files = {
        "strongest_lineup.csv": strongest_entries,
        "strongest_rankings.csv": strongest_summary,
        "tournament_lineups.csv": tournament_entries,
        "tournament_rankings.csv": tournament_summary,
        "player_exposures.csv": exposures,
        "player_projections.csv": players,
        "correlation_summary.csv": corr_summary,
        "correlation_detail.csv": corr_detail,
    }
    files.update(extra_frames or {})
    for filename, frame in files.items():
        frame.to_csv(output_dir / filename, index=False)
    zip_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
    return {name: str(output_dir / name) for name in files} | {"zip": str(zip_path)}


# ============================================================================
# NOTEBOOK CELL 17 - End-to-end showdown runner
# ============================================================================
def run_interactive(cfg=None):
    """Fetch one Yahoo slate, audit assumptions, simulate, rank, and export.

    The order is deliberate: depth is assigned from the unadjusted projection,
    then its historical mean bias is corrected, then unconfirmed backup QBs and
    user exclusions are removed. This prevents adjusted means from redefining the
    very depth role used to choose the adjustment.
    """
    cfg = _cfg(cfg)
    started = time.perf_counter()
    payload = fetch_yahoo_data()
    all_players, cap_map = normalize_yahoo_data(payload)
    all_players = add_projection_priors(all_players, PROJECTION_OVERRIDES)
    games = list_games(all_players)
    selected = select_game_interactive(games)
    salary_cap = salary_cap_for_game(cap_map, selected["Game ID"])

    players = all_players[all_players["Game ID"].eq(str(selected["Game ID"]))].copy()

    # Market-implied means replace the Yahoo FPPG/salary fallback before depth is
    # assigned, so the fallback depth heuristic also sees the better current mean.
    market_report = pd.DataFrame()
    market_applied = pd.DataFrame()
    market_audit = {"feeds": [], "notes": [], "logit_vig": np.nan, "calibration_pairs": 0}
    if cfg.use_market_projections:
        market_projections, market_audit = load_market_projection_reference(cfg)
        print("\nMarket projection feed:")
        for note in market_audit["notes"]:
            print(f"  - {note}")
        if market_projections:
            market_report = build_market_projection_report(
                players, market_projections, selected, cfg
            )
            players, market_applied = apply_market_projection_means(
                players, market_report, cfg
            )
            matched = int(market_report["Market matched"].sum())
            accepted = int(market_report["Market accepted"].sum())
            skill = int(market_report["Position"].ne("DEF").sum())
            print(
                f"  - feeds used: {', '.join(market_audit['feeds'])}; "
                f"one-way logit adjustment {market_audit['logit_vig']:.3f} "
                f"from {market_audit['calibration_pairs']} local pair(s)"
            )
            print(
                f"  - matched {matched}/{skill} skill players; "
                f"accepted {accepted} market means with quality "
                f"{tuple(cfg.market_accepted_quality)}"
            )
            print("\nMarket mean audit (manual overrides and fallbacks are explicit):")
            display(market_projection_review(market_report))
        else:
            warnings.warn(
                "No market feed was usable; continuing with Yahoo FPPG/salary priors."
            )
    else:
        print("\nMarket projections disabled (Settings.use_market_projections = False).")

    players = assign_depth_assumptions(players, DEPTH_OVERRIDES, PLAYER_STYLE_OVERRIDES)

    print(f"\n{selected['Matchup']} - cap ${salary_cap:g}")

    # nflverse role and availability, applied before the depth mean adjustment so the
    # corrected rank drives both the mean multiplier and the volatility prior.
    nflverse_report = pd.DataFrame()
    nflverse_applied = pd.DataFrame()
    nflverse_blocked = pd.DataFrame()
    if cfg.use_nflverse:
        season = cfg.nflverse_season or infer_season(selected["Game Time"])
        depth_chart, roster_status, as_of, notes = load_nflverse_reference(players, season, cfg)
        print("\nnflverse reference feed:")
        for note in notes:
            print(f"  - {note}")
        if depth_chart is not None or roster_status is not None:
            nflverse_report = build_nflverse_role_report(
                players, depth_chart, roster_status, cfg
            )
            players, nflverse_applied = apply_nflverse_depth(players, nflverse_report, cfg)
            matched = int(nflverse_report["Depth matched"].sum())
            skill = int(nflverse_report["Position"].ne("DEF").sum())
            statused = int(nflverse_report["Roster status"].notna().sum())
            print(
                f"  - matched {matched}/{skill} skill players to a depth-chart entry, "
                f"{statused}/{skill} to a roster status"
            )
            print(f"  - depth ranks rewritten from the published chart: {len(nflverse_applied)}")
            review = nflverse_disagreement_view(nflverse_report)
            if len(review):
                print(
                    "\nnflverse disagreements and unmatched players - review before "
                    "trusting the run:"
                )
                display(review)
    else:
        print("\nnflverse cross-check disabled (Settings.use_nflverse = False).")

    players = apply_depth_mean_adjustments(players)
    print(
        "\nDepth controls both a fitted mean correction and volatility prior; "
        "verify injuries, actives, and expected snaps."
    )
    display(depth_sanity_report(players))

    if cfg.use_nflverse and len(nflverse_report):
        players, nflverse_blocked = apply_nflverse_availability(players, nflverse_report, cfg)
        if len(nflverse_blocked):
            print(
                f"\nAVAILABILITY FILTER removed {len(nflverse_blocked)} player(s) "
                f"(allowed status: {', '.join(cfg.nflverse_available_status)}):"
            )
            display(nflverse_blocked)
        else:
            print("\nAvailability filter: every priced player is on an allowed roster status.")
    players, auto_removed_qbs = apply_default_role_filters(
        players, INCLUDE_BACKUP_QBS, cfg
    )
    if auto_removed_qbs:
        print(
            "Default backup-QB filter removed: " + ", ".join(auto_removed_qbs)
            + ". Add a confirmed replacement to INCLUDE_BACKUP_QBS (and give it a "
            "DEPTH_OVERRIDES entry of 1) to retain it."
        )
    players = apply_exclusions_interactive(players, EXCLUDE_PLAYERS)
    eligible_player_count = len(players)
    players, dropped = trim_player_pool(players, cfg)
    if dropped:
        print(
            f"Enumeration cap retained {len(players)} of {eligible_player_count} "
            f"eligible players and removed {len(dropped)} fringe players: "
            + ", ".join(dropped)
        )
    problem = roster_feasibility_error(players, cfg.lineup_size)
    if problem:
        raise ValueError(problem)

    # Keep all matrices aligned to this exact RangeIndex.
    players = players.reset_index(drop=True)
    outcomes, correlation_model = simulate_player_outcomes(players, cfg)
    corr_summary, corr_detail = correlation_sanity_report(
        players, outcomes, correlation_model
    )

    print("\nCross-position correlation sanity check:")
    focus_corr = corr_summary[
        corr_summary["Relationship"].isin(CALIBRATION_SANITY_RELATIONSHIPS)
    ]
    display(focus_corr)
    print(
        "Largest pairwise score-correlation change required for a valid joint "
        f"matrix: {correlation_model['psd_max_score_adjustment']:.3f}."
    )
    if correlation_model["infeasible_pairs"]:
        print(
            f"{correlation_model['infeasible_pairs']} target(s) were unattainable for "
            "lognormals with these CVs and were clipped (largest shift "
            f"{correlation_model['max_infeasible_shift']:.3f})."
        )
    review = focus_corr[focus_corr["Sanity"].eq("REVIEW")]
    if len(review):
        warnings.warn(
            "One or more displayed relationship groups differs materially from "
            "its fitted target after PSD repair or Monte Carlo sampling."
        )

    print("\nPer-player marginal check (simulator reproduced the intended mean and CV):")
    marginals = marginal_sanity_report(players, outcomes)
    flagged = marginals[marginals["Sanity"].eq("REVIEW")]
    display(flagged if len(flagged) else marginals.head(5))
    if len(flagged):
        warnings.warn(f"{len(flagged)} player marginal(s) drifted from the intended mean or CV.")
    else:
        print(f"All {len(marginals)} players within 3% on the mean and 8% on the CV.")

    covariance = analytic_covariance(players, correlation_model)
    candidates, valid_rosters = enumerate_candidate_lineups(players, salary_cap, covariance, cfg)
    print(
        f"\nYahoo-valid base rosters: {valid_rosters:,}; "
        f"retained Superstar candidates: {len(candidates):,}; "
        f"shared simulations: {cfg.simulations:,}."
    )
    scored = score_candidates_shared_scenarios(candidates, outcomes, cfg)

    screen_check = analytic_vs_simulated_lineup_check(scored)
    print("\nAnalytic screen vs simulation (the pruning step is ranking the same thing):")
    display(screen_check)

    reliability = reliability_report(scored, top_n=cfg.tournament_lineups)
    if cfg.report_reliability:
        print(
            "\nMonte Carlo reliability (split-half, Spearman-Brown corrected). "
            "A low value means the column re-ranks differently under another seed:"
        )
        display(reliability)
        weak = reliability[reliability["Verdict"].str.startswith("NOISY")]["Metric"].tolist()
        if weak:
            print(
                "Noisy at this sample size: " + ", ".join(weak)
                + ". Raise Settings.simulations, or read those columns as approximate."
            )

    strongest = select_strongest_lineups(scored, count=10)
    portfolio = select_tournament_portfolio(scored, cfg)

    display_best_lineup(strongest, players, salary_cap)
    display_alternatives(strongest, players, salary_cap)
    print("\nDIVERSIFIED TOURNAMENT PORTFOLIO")
    tournament_view = lineup_summary(portfolio, players, "Tournament", salary_cap)
    display(lineup_comparison_view(tournament_view))
    diversity = portfolio_diversity_report(portfolio, cfg)
    print("\nPORTFOLIO DIVERSITY")
    display(diversity)
    print("\nTOURNAMENT EXPOSURE")
    exposures = exposure_report(portfolio, players)
    display(exposures)

    safe_matchup = selected["Matchup"].replace(" ", "_").replace("/", "-")
    output_dir = f"yahoo_showdown_{safe_matchup}"
    files = export_results(
        output_dir, strongest, portfolio, players, corr_summary, corr_detail,
        salary_cap,
        extra_frames={
            "reliability_report.csv": reliability,
            "analytic_vs_simulated.csv": screen_check,
            "player_marginals.csv": marginals,
            "portfolio_diversity.csv": diversity,
            **({"market_projection_report.csv": market_report} if len(market_report) else {}),
            **({"market_means_applied.csv": market_applied} if len(market_applied) else {}),
            **({"nflverse_role_report.csv": nflverse_report} if len(nflverse_report) else {}),
            **({"nflverse_removed.csv": nflverse_blocked} if len(nflverse_blocked) else {}),
        },
    )
    print(f"\nSaved results bundle: {files['zip']}")
    print(f"Total run time: {time.perf_counter() - started:.1f}s")
    return {
        "game": selected.to_dict(),
        "salary_cap": salary_cap,
        "players": players,
        "outcomes": outcomes,
        "correlation_model": correlation_model,
        # Backward-compatible alias for code written against notebook v2.
        "factor_model": correlation_model,
        "correlation_summary": corr_summary,
        "correlation_detail": corr_detail,
        "marginals": marginals,
        "market_audit": market_audit,
        "market_report": market_report,
        "market_means_applied": market_applied,
        "nflverse_report": nflverse_report,
        "nflverse_depth_applied": nflverse_applied,
        "nflverse_removed": nflverse_blocked,
        "reliability": reliability,
        "screen_check": screen_check,
        "diversity": diversity,
        "exposures": exposures,
        "scored_candidates": scored,
        "strongest": strongest,
        "portfolio": portfolio,
        "files": files,
    }


# ============================================================================
# NOTEBOOK CELL 19 - Weekly top-N rankings by position
# ============================================================================
def run_position_rankings(top_n=25, cfg=None, positions=VALID_POSITIONS, export_csv=True):
    """Build full-slate Yahoo rankings from the notebook's final estimated FP.

    Projection priority is unchanged from the showdown model:
    manual override > accepted market mean > Yahoo FPPG/salary prior.
    Current market/manual means are not depth-haircut; fallback estimates are.
    """
    cfg = _cfg(cfg)
    top_n = int(top_n)
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    requested_positions = []
    for position in positions:
        normalized = str(position).upper().replace("D/ST", "DEF").replace("DST", "DEF")
        if normalized not in VALID_POSITIONS:
            raise ValueError(
                f"Unsupported position {position!r}; choose from {VALID_POSITIONS}"
            )
        if normalized not in requested_positions:
            requested_positions.append(normalized)

    started = time.perf_counter()
    payload = fetch_yahoo_data()
    players, _ = normalize_yahoo_data(payload)
    players = add_projection_priors(players, PROJECTION_OVERRIDES)
    games = list_games(players)
    if games.empty:
        raise ValueError("Yahoo returned no usable NFL games")

    print(
        f"Yahoo slate: {len(games)} game(s), {len(players)} priced player(s); "
        f"ranking top {top_n} per position."
    )

    # Pull the market once, then use the notebook's hard game-time guard for each game.
    market_report = pd.DataFrame()
    market_applied = pd.DataFrame()
    market_audit = {"feeds": [], "notes": [], "logit_vig": np.nan, "calibration_pairs": 0}
    if cfg.use_market_projections:
        market_projections, market_audit = load_market_projection_reference(cfg)
        for note in market_audit["notes"]:
            print(f"  Market: {note}")
        if market_projections:
            reports = []
            for _, game in games.iterrows():
                game_players = players[
                    players["Game ID"].eq(str(game["Game ID"]))
                ].copy()
                if game_players.empty:
                    continue
                reports.append(
                    build_market_projection_report(
                        game_players, market_projections, game, cfg
                    )
                )
            if reports:
                market_report = pd.concat(reports, ignore_index=True)
                players, market_applied = apply_market_projection_means(
                    players, market_report, cfg
                )
                accepted = int(market_report["Market accepted"].sum())
                matched = int(market_report["Market matched"].sum())
                skill = int(market_report["Position"].ne("DEF").sum())
                feeds = ", ".join(market_audit["feeds"]) or "none"
                print(
                    f"  Market: {feeds}; matched {matched}/{skill} skill-player rows; "
                    f"accepted {accepted} estimated means."
                )
        else:
            warnings.warn(
                "No market feed was usable; rankings use Yahoo FPPG/salary priors."
            )
    else:
        print("  Market projections disabled; using Yahoo FPPG/salary priors.")

    # Assign role using the unadjusted estimate, exactly as the showdown runner does.
    players = assign_depth_assumptions(
        players, DEPTH_OVERRIDES, PLAYER_STYLE_OVERRIDES
    )

    nflverse_report = pd.DataFrame()
    nflverse_applied = pd.DataFrame()
    nflverse_blocked = pd.DataFrame()
    if cfg.use_nflverse:
        first_kickoff = games.iloc[0]["Game Time"]
        season = cfg.nflverse_season or infer_season(first_kickoff)
        depth_chart, roster_status, as_of, notes = load_nflverse_reference(
            players, season, cfg
        )
        for note in notes:
            print(f"  nflverse: {note}")
        if depth_chart is not None or roster_status is not None:
            nflverse_report = build_nflverse_role_report(
                players, depth_chart, roster_status, cfg
            )
            players, nflverse_applied = apply_nflverse_depth(
                players, nflverse_report, cfg
            )

    players = apply_depth_mean_adjustments(players)

    if cfg.use_nflverse and not nflverse_report.empty:
        players, nflverse_blocked = apply_nflverse_availability(
            players, nflverse_report, cfg
        )
        if len(nflverse_blocked):
            print(
                f"  Availability filter removed {len(nflverse_blocked)} player(s)."
            )

    players, backup_qbs_removed = apply_default_role_filters(
        players, INCLUDE_BACKUP_QBS, cfg
    )
    if backup_qbs_removed:
        print(f"  Backup-QB filter removed {len(backup_qbs_removed)} player(s).")

    excluded = set(EXCLUDE_PLAYERS)
    _warn_unmatched(excluded, "Exclusion", set(players["Name"]))
    players = players[~players["Name"].isin(excluded)].copy()
    players["FP_per_Salary"] = players["Projected_FP"] / players["Salary"]

    # Stable tie-breaks make repeated runs deterministic when estimates are equal.
    ordered = players.sort_values(
        ["Position", "Projected_FP", "FPPG", "Salary", "Name"],
        ascending=[True, False, False, False, True],
    ).copy()

    tables = {}
    combined = []
    for position in requested_positions:
        group = ordered[ordered["Position"].eq(position)].head(top_n).copy()
        group.insert(0, "Rank", np.arange(1, len(group) + 1))
        group["Estimated FP"] = group["Projected_FP"].round(2)
        group["FP / salary"] = group["FP_per_Salary"].round(3)
        group["Yahoo FPPG"] = group["FPPG"].round(2)
        group["Kickoff UTC"] = pd.to_datetime(
            group["Game Time"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M")
        group["Projection method"] = group["Projection_Source"].astype(str)

        columns = [
            "Rank", "Name", "Team", "Opponent", "Estimated FP", "Yahoo FPPG",
            "Salary", "FP / salary", "Depth_Rank", "Kickoff UTC",
            "Projection method",
        ]
        view = group[columns].rename(
            columns={"Name": "Player", "Depth_Rank": "Depth"}
        ).reset_index(drop=True)
        tables[position] = view
        combined.append(view.assign(Position=position))

        print(f"\nTOP {min(top_n, len(view))} {position}")
        display(view)

    combined_table = (
        pd.concat(combined, ignore_index=True)
        if combined
        else pd.DataFrame()
    )
    if not combined_table.empty:
        combined_table = combined_table[
            ["Position"] + [c for c in combined_table.columns if c != "Position"]
        ]

    csv_path = None
    if export_csv:
        csv_path = f"yahoo_top_{top_n}_by_position.csv"
        combined_table.to_csv(csv_path, index=False)
        print(f"\nSaved combined rankings: {csv_path}")

    print(f"Ranking run time: {time.perf_counter() - started:.1f}s")
    return {
        "rankings": tables,
        "combined": combined_table,
        "players": players.reset_index(drop=True),
        "games": games,
        "market_audit": market_audit,
        "market_report": market_report,
        "market_means_applied": market_applied,
        "nflverse_report": nflverse_report,
        "nflverse_depth_applied": nflverse_applied,
        "nflverse_removed": nflverse_blocked,
        "csv": csv_path,
    }
