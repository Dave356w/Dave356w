"""Compare legacy v8 blending with v9 sequential SP/BP matchup phases.

The comparison is no-lookahead: both formulas consume the same values already
persisted in a daily lean snapshot. Pre-v9 files are reported as ineligible
because they do not contain the neutral lineup composite; the script never
guesses B_0 from B_SP or reconstructs old Savant state with current data.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


REQUIRED_SIDE_COLS = (
    "opp_xwOBA_neutral",
    "opp_xwOBA_vs_sp",
    "starter_xwOBA",
    "bullpen_xwOBA",
    "lg_xwOBA",
)


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "1.0"}
    return bool(value) if pd.notna(value) else False


def recalculate_sides(snapshot):
    """Return snapshot sides with v8-shadow and v9 matchup values.

    v8: B_SP * [q*P_SP + (1-q)*P_BP] / L
    v9: q*[B_SP*P_SP/L] + (1-q)*[B_0*P_BP/L]
    """
    out = snapshot.copy()
    b0 = _num(out, "opp_xwOBA_neutral")
    bsp = _num(out, "opp_xwOBA_vs_sp")
    psp = _num(out, "starter_xwOBA")
    pbp = _num(out, "bullpen_xwOBA")
    league = _num(out, "lg_xwOBA")
    q = _num(out, "sp_share")
    if q.isna().all():
        q = _num(out, "expected_sp_ip") / 9.0
    q = q.clip(lower=0.0, upper=1.0)

    valid = (
        b0.notna() & bsp.notna() & psp.notna() & pbp.notna()
        & league.notna() & league.ne(0) & q.notna()
    )
    p_v8 = q * psp + (1.0 - q) * pbp
    mx_sp = bsp * psp / league
    mx_bp = b0 * pbp / league
    mx_v8 = bsp * p_v8 / league
    mx_v9 = q * mx_sp + (1.0 - q) * mx_bp

    out["comparison_eligible"] = valid
    out["sp_share_recalc"] = q
    out["bp_share_recalc"] = 1.0 - q
    out["mx_xwOBA_v8_shadow"] = mx_v8.where(valid)
    out["mx_xwOBA_v9_recalc"] = mx_v9.where(valid)
    out["edge_xwOBA_v8_shadow"] = (mx_v8 - league).where(valid)
    out["edge_xwOBA_v9_recalc"] = (mx_v9 - league).where(valid)
    return out


def _ip_bucket(value):
    if pd.isna(value):
        return "unknown"
    if value <= 3.0:
        return "<=3.0 IP"
    if value <= 4.5:
        return "3.0-4.5 IP"
    if value <= 5.5:
        return "4.5-5.5 IP"
    return ">5.5 IP"


def pair_games(sides):
    """Pair away-SP and home-SP rows into one v8/v9 decision per game."""
    games = []
    key_cols = ["game_pk"]
    if "game_date" in sides.columns:
        key_cols.append("game_date")
    for key, group in sides.groupby(key_cols, sort=False, dropna=False):
        away = group[group["side"] == "away"]
        home = group[group["side"] == "home"]
        if away.empty or home.empty:
            continue
        away, home = away.iloc[0], home.iloc[0]
        eligible = bool(away["comparison_eligible"] and home["comparison_eligible"])
        if isinstance(key, tuple):
            game_pk, game_date = key[0], key[1]
        else:
            game_pk, game_date = key, np.nan

        # The away-pitcher row is the HOME offense matchup; the home-pitcher
        # row is the AWAY offense matchup.
        net_v8 = (
            away["edge_xwOBA_v8_shadow"] - home["edge_xwOBA_v8_shadow"]
            if eligible else np.nan
        )
        net_v9 = (
            away["edge_xwOBA_v9_recalc"] - home["edge_xwOBA_v9_recalc"]
            if eligible else np.nan
        )
        away_team = home.get("opp_team")
        home_team = away.get("opp_team")
        lean_v8 = (
            home_team if net_v8 > 0 else away_team if net_v8 < 0 else None
        ) if eligible else None
        lean_v9 = (
            home_team if net_v9 > 0 else away_team if net_v9 < 0 else None
        ) if eligible else None
        ip_away = pd.to_numeric(away.get("expected_sp_ip"), errors="coerce")
        ip_home = pd.to_numeric(home.get("expected_sp_ip"), errors="coerce")
        min_ip = np.nanmin([ip_away, ip_home]) if not (
            pd.isna(ip_away) and pd.isna(ip_home)
        ) else np.nan
        bp_values = np.asarray([
            away.get("bp_share_recalc"),
            home.get("bp_share_recalc"),
        ], dtype=float)
        bp_mean = (
            float(np.nanmean(bp_values))
            if np.isfinite(bp_values).any()
            else np.nan
        )
        games.append({
            "game_pk": game_pk,
            "game_date": game_date,
            "away": away_team,
            "home": home_team,
            "eligible": eligible,
            "xw_net_v8": net_v8,
            "xw_net_v9": net_v9,
            "xw_net_change": net_v9 - net_v8 if eligible else np.nan,
            "lean_v8": lean_v8,
            "lean_v9": lean_v9,
            "lean_flip": (
                bool(lean_v8 != lean_v9)
                if eligible and lean_v8 is not None and lean_v9 is not None
                else False
            ),
            "expected_sp_ip_min": min_ip,
            "expected_sp_ip_bucket": _ip_bucket(min_ip),
            "bp_share_mean": bp_mean,
            "bullpen_heavy": bool(pd.notna(bp_mean) and bp_mean >= 0.5),
            "opener": _bool(away.get("opener")) or _bool(home.get("opener")),
        })
    return pd.DataFrame(games)


def attach_results_and_market(games, ledger):
    if games.empty or ledger is None or ledger.empty:
        return games
    led = ledger.copy()
    led["game_pk"] = pd.to_numeric(led["game_pk"], errors="coerce")
    led["game_date"] = led["game_date"].astype(str)
    keep = [
        "game_pk", "game_date", "full_away", "full_home",
        "close_away_ml", "close_home_ml", "close_p_home",
    ]
    for col in keep:
        if col not in led.columns:
            led[col] = np.nan
    led = led[keep].drop_duplicates(["game_pk", "game_date"], keep="last")
    out = games.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
    out["game_date"] = out["game_date"].astype(str)
    return out.merge(led, on=["game_pk", "game_date"], how="left")


def _american_profit(odds):
    if pd.isna(odds) or float(odds) == 0:
        return np.nan
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def score_model(games, model):
    """Add settled-result, market-disagreement, and one-unit ROI columns."""
    out = games.copy()
    lean_col = f"lean_{model}"
    full_away = _num(out, "full_away")
    full_home = _num(out, "full_home")
    decided = full_away.notna() & full_home.notna()
    winner = np.where(
        full_home > full_away,
        out["home"],
        np.where(
            full_away > full_home,
            out["away"],
            None,
        ),
    )
    winner = pd.Series(winner, index=out.index)
    out[f"win_{model}"] = np.where(
        decided & out[lean_col].notna() & winner.notna(),
        out[lean_col].astype(str) == winner.astype(str),
        np.nan,
    )
    selected_ml = np.where(
        out[lean_col] == out["home"],
        _num(out, "close_home_ml"),
        np.where(out[lean_col] == out["away"], _num(out, "close_away_ml"), np.nan),
    )
    market_home = _num(out, "close_p_home")
    market_favorite = np.where(
        market_home >= .5, out["home"],
        np.where(market_home < .5, out["away"], None),
    )
    out[f"market_disagree_{model}"] = np.where(
        out[lean_col].notna() & pd.Series(market_favorite).notna(),
        out[lean_col].astype(str)
        != pd.Series(market_favorite, index=out.index).astype(str),
        np.nan,
    )
    roi = []
    for win, odds in zip(out[f"win_{model}"], selected_ml):
        if pd.isna(win) or pd.isna(odds):
            roi.append(np.nan)
        else:
            roi.append(_american_profit(odds) if bool(win) else -1.0)
    out[f"roi_{model}"] = roi
    return out


def _group_summary(frame, label):
    rows = []
    for name, group in frame.groupby(label, dropna=False):
        eligible = group[group["eligible"]]
        settled = eligible["win_v9"].dropna() if "win_v9" in eligible else pd.Series(dtype=float)
        roi = eligible["roi_v9"].dropna() if "roi_v9" in eligible else pd.Series(dtype=float)
        rows.append({
            label: name,
            "games": len(eligible),
            "flips": int(eligible["lean_flip"].sum()),
            "mean_abs_xw_change": eligible["xw_net_change"].abs().mean(),
            "v9_wins": int((settled == True).sum()),  # noqa: E712
            "v9_losses": int((settled == False).sum()),  # noqa: E712
            "v9_flat_roi": roi.sum(),
            "v9_roi_per_bet": roi.mean(),
        })
    return pd.DataFrame(rows)


def report(games, side_count, ineligible_side_count):
    eligible = games[games["eligible"]] if not games.empty else games
    print(
        f"sides: {side_count} total | {side_count - ineligible_side_count} eligible "
        f"| {ineligible_side_count} missing v9 phase inputs"
    )
    print(
        f"games: {len(games)} paired | {len(eligible)} eligible | "
        f"{int(eligible['lean_flip'].sum()) if len(eligible) else 0} lean flips"
    )
    if eligible.empty:
        print(
            "No comparable games. Pre-v9 snapshots lack opp_xwOBA_neutral, so "
            "a no-lookahead v8-v9 replay is impossible until v9 snapshots exist."
        )
        return
    print(
        "xw_net change: "
        f"mean={eligible['xw_net_change'].mean():+.6f} "
        f"median={eligible['xw_net_change'].median():+.6f} "
        f"mean_abs={eligible['xw_net_change'].abs().mean():.6f} "
        f"median_abs={eligible['xw_net_change'].abs().median():.6f}"
    )
    print("\nby minimum expected SP IP:")
    print(_group_summary(eligible, "expected_sp_ip_bucket").to_string(index=False))
    print("\nopener:")
    print(_group_summary(eligible, "opener").to_string(index=False))
    print("\nbullpen-heavy (mean BP share >= 0.50):")
    print(_group_summary(eligible, "bullpen_heavy").to_string(index=False))
    for model in ("v8", "v9"):
        if f"roi_{model}" not in eligible:
            continue
        roi = eligible[f"roi_{model}"].dropna()
        disagree = eligible[f"market_disagree_{model}"].dropna()
        roi_total = f"{roi.sum():+.3f}u" if len(roi) else "NA"
        roi_rate = f"{roi.mean():+.3f}u/bet" if len(roi) else "NA"
        print(
            f"{model}: market-disagree={int((disagree == True).sum())}/"  # noqa: E712
            f"{len(disagree)} | flat ROI={roi_total} "
            f"({roi_rate}, n={len(roi)})"
        )


def load_snapshots(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    frames = []
    for path in sorted(set(paths)):
        frame = pd.read_csv(path)
        frame["_snapshot_path"] = path
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "snapshots",
        nargs="*",
        default=["data/leans_*_xw.csv"],
        help="snapshot paths or glob patterns",
    )
    parser.add_argument("--ledger", default="data/mlb_lean_ledger.csv")
    parser.add_argument("--output", help="optional per-game comparison CSV")
    args = parser.parse_args()

    raw = load_snapshots(args.snapshots)
    if raw.empty:
        raise SystemExit("no snapshot rows found")
    sides = recalculate_sides(raw)
    games = pair_games(sides)
    ledger = pd.read_csv(args.ledger) if os.path.exists(args.ledger) else None
    games = attach_results_and_market(games, ledger)
    games = score_model(score_model(games, "v8"), "v9")
    report(
        games,
        side_count=len(sides),
        ineligible_side_count=int((~sides["comparison_eligible"]).sum()),
    )
    if args.output:
        games.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
