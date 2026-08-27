#!/usr/bin/env python3
"""Sequential replay of the current MLB model against the immutable ledger.

The production ledger is read-only.  Each slate is rebuilt with inputs ending
at D-1, while expected-IP calibration is fit only on regenerated rows from
earlier replay dates.  Prediction math is imported from ``build_site.py`` and
ledger-row extraction/grading semantics from ``grade_leans.py``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_site as model
import grade_leans as grader
from historical_data import HistoricalDataProvider


DEFAULT_LEDGER = Path("data/mlb_lean_ledger.csv")
DEFAULT_OUTPUT = Path("data/walkforward_current.csv")
DEFAULT_REPORT = Path("data/walkforward_report.txt")
DEFAULT_CACHE = Path(".walkforward_cache")
SOURCE_ROOT = Path(__file__).resolve().parent

CALIBRATION_COLS = [
    "expected_sp_ip_raw_away", "expected_sp_ip_raw_home",
    "expected_sp_ip_away", "expected_sp_ip_home",
    "act_sp_ip_away", "act_sp_ip_home",
]


def _num(value):
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


REPLAY_SOURCES = ("build_site.py", "grade_leans.py", "historical_data.py",
                  "walkforward.py")


def _code_only(path):
    """A module's CODE, with comments and docstrings removed.

    The replay cache is keyed on this. It used to be keyed on the raw file
    BYTES, which meant any edit at all discarded every cached row and restarted
    the replay from the first slate -- including edits that cannot move a
    prediction. Measured on 2026-08-27: three merged display-only commits
    (a1efe74, 5ea0758, 3fdc201), one of them almost entirely DELETIONS of dead
    code, each reset the backtest. Appending a bare comment to build_site.py
    changed the hash. Because the replay is also bounded (it gets one slice of
    a run before its timeout), a cache that resets on comments never finishes:
    the committed replay stopped at 2026-08-11 while the ledger ran to 08-27,
    and it had never once covered the v12 window it exists to validate.

    Comments are not in the AST at all, so they are excluded by construction.
    Docstrings ARE nodes, so they are stripped explicitly. `ast.dump` omits
    line numbers by default, so shifting code down a file changes nothing
    either.

    THIS DELIBERATELY STAYS STRICT ABOUT CODE. Any change to an expression, a
    constant's value, an argument default or the order of statements produces a
    different dump and still invalidates the cache -- which is the direction
    that matters, because a stale row silently mixed into a new replay is a
    far worse failure than a spurious reset. The relaxation is confined to
    text that the interpreter throws away.
    """
    tree = ast.parse((SOURCE_ROOT / path).read_bytes())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def replay_config_hash():
    digest = hashlib.sha256()
    digest.update(model.MODEL_TAG.encode())
    for name in REPLAY_SOURCES:
        digest.update(_code_only(name).encode())
    return digest.hexdigest()[:16]


def calibration_history(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=CALIBRATION_COLS)
    for col in CALIBRATION_COLS:
        if col not in frame:
            frame[col] = np.nan
    return frame[CALIBRATION_COLS].copy()


def grade_prediction(lean, away, home, away_runs, home_runs, allow_tie=False):
    away_runs, home_runs = _num(away_runs), _num(home_runs)
    if pd.isna(away_runs) or pd.isna(home_runs):
        return None
    return grader._wlt(lean, away, home, away_runs, home_runs, allow_tie)


def flat_unit_roi(grade, moneyline):
    """One-unit return at American closing odds; ties push."""
    moneyline = _num(moneyline)
    if grade not in {"W", "L", "T"} or pd.isna(moneyline) or moneyline == 0:
        return np.nan
    if grade == "L":
        return -1.0
    if grade == "T":
        return 0.0
    return moneyline / 100.0 if moneyline > 0 else 100.0 / abs(moneyline)


def market_fields(source, lean, grade):
    home_prob = _num(source.get("close_p_home"))
    if lean == source.get("home"):
        moneyline = _num(source.get("close_home_ml"))
        pick_prob = home_prob
    elif lean == source.get("away"):
        moneyline = _num(source.get("close_away_ml"))
        pick_prob = 1.0 - home_prob if pd.notna(home_prob) else np.nan
    else:
        moneyline = pick_prob = np.nan

    if lean is None or pd.isna(home_prob):
        versus = None
    else:
        favorite = source.get("home") if home_prob >= 0.5 else source.get("away")
        versus = "AGREE" if lean == favorite else "DIVERGE"
    return {
        "market_home_prob": home_prob,
        "wf_market_pick_prob": pick_prob,
        "wf_close_ml": moneyline,
        "wf_vs_market": versus,
        "wf_roi": flat_unit_roi(grade, moneyline),
    }


def _score_text(away, home):
    away, home = _num(away), _num(home)
    if pd.isna(away) or pd.isna(home):
        return None
    return f"{int(away)}-{int(home)}"


def _stamp_replay_dump(frame, slate_date):
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    as_of = (pd.Timestamp(slate_date) - pd.Timedelta(days=1))
    out["model_tag"] = f"{model.MODEL_TAG}_walkforward"
    out["model_metric"] = model.MODEL_RATE_LABEL
    out["snapshot_utc"] = as_of.strftime("%Y-%m-%dT23:59:59Z")
    if "game_datetime_utc" in out:
        out["scheduled_start_utc"] = out["game_datetime_utc"]
    return out


def predict_slate(slate_date, source_day, history, cache_dir,
                  include_platoon=False, snapshot_dir=None):
    provider = HistoricalDataProvider(
        slate_date, source_day, cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
    )
    fetched = model.fetch_all(
        slate_date,
        provider=provider,
        calibration_history=history,
        include_platoon=include_platoon,
        write_audit=False,
    )
    if fetched.get("empty"):
        return {}, provider
    matchup, pitcher_rows, opp_hitters = model.build_xwoba_matchup(
        fetched["pitchers_df"], fetched["league_baseline"]
    )
    matchup = model.apply_pitching_plans(
        matchup, fetched.get("pitching_plans"), fetched["league_baseline"]
    )
    matchup = _stamp_replay_dump(matchup, slate_date)
    if matchup is None or matchup.empty:
        return {}, provider
    rows = grader.rows_from_dump(matchup, None)
    return {int(row["game_pk"]): row for row in rows}, provider


def compose_output_row(source, prediction, provider, config_hash):
    gpk = int(source["game_pk"])
    away, home = source.get("away"), source.get("home")
    lean = None if prediction is None else _text(prediction.get("xw_lean"))
    full_grade = grade_prediction(
        lean, away, home, source.get("full_away"), source.get("full_home")
    )
    f5_grade = grade_prediction(
        lean, away, home, source.get("f5_away"), source.get("f5_home"),
        allow_tie=True,
    )
    original_lean = _text(source.get("xw_lean"))
    changed = (
        bool(lean != original_lean)
        if lean is not None and original_lean is not None else np.nan
    )
    row = {
        "game_pk": gpk,
        "game_date": str(source.get("game_date"))[:10],
        "away": away,
        "home": home,
        "replay_model_tag": model.MODEL_TAG,
        "replay_config_hash": config_hash,
        "replay_asof": provider.as_of_date,
        "input_fidelity": provider.input_fidelity(gpk),
        "replay_error": provider.error_for(gpk),
        "wf_home_off_edge": np.nan if prediction is None else _num(prediction.get("home_off_edge")),
        "wf_away_off_edge": np.nan if prediction is None else _num(prediction.get("away_off_edge")),
        "wf_xw_net": np.nan if prediction is None else _num(prediction.get("xw_net")),
        "wf_lean": lean,
        "wf_delta": np.nan if prediction is None else _num(prediction.get("xw_delta")),
        "actual_full": _score_text(source.get("full_away"), source.get("full_home")),
        "actual_f5": _score_text(source.get("f5_away"), source.get("f5_home")),
        "wf_full_grade": full_grade,
        "wf_f5_grade": f5_grade,
        "original_model_tag": source.get("model_tag"),
        "original_lean": original_lean,
        "original_xw_net": _num(source.get("xw_net")),
        "original_full_grade": _text(source.get("xw_full")),
        "original_f5_grade": _text(source.get("xw_f5")),
        "changed_from_original": changed,
        "source_lineup_status_away": source.get("lineup_status_away"),
        "source_lineup_status_home": source.get("lineup_status_home"),
        "source_away_sp": source.get("away_sp"),
        "source_home_sp": source.get("home_sp"),
        "act_sp_ip_away": _num(source.get("act_sp_ip_away")),
        "act_sp_ip_home": _num(source.get("act_sp_ip_home")),
    }
    for col in (
        "B_home", "B_away", "P_awaySP", "P_homeSP", "d_lineup", "d_sp",
        "expected_sp_ip_away", "expected_sp_ip_home",
        "expected_sp_ip_raw_away", "expected_sp_ip_raw_home",
        "starter_xwoba_away", "starter_xwoba_home",
        "bullpen_xwoba_away", "bullpen_xwoba_home",
        "sp_share_away", "sp_share_home",
        "bp_share_away", "bp_share_home",
    ):
        row[col] = np.nan if prediction is None else _num(prediction.get(col))
    row.update(market_fields(source, lean, full_grade))
    return row


def _atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _complete_dates(existing, source, config_hash):
    if existing is None or existing.empty:
        return set()
    if "replay_config_hash" not in existing:
        return set()
    existing = existing[existing["replay_config_hash"].astype(str) == config_hash]
    complete = set()
    for day, expected in source.groupby("game_date"):
        got = existing[existing["game_date"].astype(str) == str(day)]
        if set(pd.to_numeric(got.get("game_pk"), errors="coerce").dropna().astype(int)) == set(
            pd.to_numeric(expected["game_pk"], errors="coerce").dropna().astype(int)
        ):
            complete.add(str(day))
    return complete


def run_walkforward(ledger_path=DEFAULT_LEDGER, output_path=DEFAULT_OUTPUT,
                    report_path=DEFAULT_REPORT, cache_dir=DEFAULT_CACHE,
                    snapshot_dir=None,
                    start_date=None, end_date=None, include_pending=False,
                    include_platoon=False, resume=True):
    ledger_path, output_path = Path(ledger_path), Path(output_path)
    if ledger_path.resolve() == output_path.resolve():
        raise ValueError("walk-forward output must not overwrite the production ledger")
    source = pd.read_csv(ledger_path)
    source["game_date"] = source["game_date"].astype(str).str[:10]
    if not include_pending:
        source = source[
            (source.get("status").astype(str) == "graded")
            & pd.to_numeric(source.get("full_away"), errors="coerce").notna()
            & pd.to_numeric(source.get("full_home"), errors="coerce").notna()
        ].copy()
    if start_date:
        source = source[source["game_date"] >= str(start_date)[:10]]
    if end_date:
        source = source[source["game_date"] <= str(end_date)[:10]]
    source = source.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    if source.empty:
        raise ValueError("no ledger games match the requested replay range")

    config_hash = replay_config_hash()
    existing = None
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
    done = _complete_dates(existing, source, config_hash)
    results = []
    if existing is not None and done:
        keep = existing[
            (existing["replay_config_hash"].astype(str) == config_hash)
            & existing["game_date"].astype(str).isin(done)
        ]
        results.extend(keep.to_dict("records"))

    for slate_date, source_day in source.groupby("game_date", sort=True):
        slate_date = str(slate_date)
        if slate_date in done:
            print(f"walkforward {slate_date}: cached {len(source_day)} games")
            continue
        prior = [r for r in results if str(r.get("game_date")) < slate_date]
        print(
            f"walkforward {slate_date}: {len(source_day)} games, "
            f"{len(prior)} prior replay rows"
        )
        predictions, provider = predict_slate(
            slate_date, source_day, calibration_history(prior), cache_dir,
            include_platoon=include_platoon, snapshot_dir=snapshot_dir,
        )
        for _, source_row in source_day.iterrows():
            gpk = int(source_row["game_pk"])
            results.append(compose_output_row(
                source_row, predictions.get(gpk), provider, config_hash
            ))
        frame = pd.DataFrame(results).sort_values(
            ["game_date", "game_pk"]
        ).reset_index(drop=True)
        _atomic_csv(frame, output_path)

    frame = pd.DataFrame(results).sort_values(
        ["game_date", "game_pk"]
    ).reset_index(drop=True)
    _atomic_csv(frame, output_path)
    report = render_report(frame, source)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    return frame, report


def _record(series):
    series = pd.Series(series).dropna().astype(str)
    w, l, t = (int((series == value).sum()) for value in ("W", "L", "T"))
    pct = w / (w + l) if w + l else np.nan
    tail = f" ({pct:.3f})" if pd.notna(pct) else ""
    return f"{w}-{l}" + (f"-{t}" if t else "") + tail


def coverage(frame, source=None):
    """What the replay actually covers, against what it was asked to cover.

    A truncated replay is NOT a random subsample of the ledger. Dates are
    replayed in order, so a run that stops early always keeps the EARLIEST
    slates and drops the most recent ones -- exactly the window a reader cares
    about most. On 2026-08-27 the report on disk read "Games 690" while the
    frame beside it held 500 rows ending 2026-08-11, with the ledger running to
    08-27: sixteen dates and 209 games missing, none of them announced, and the
    BY MONTH block simply had no August-second-half line to omit visibly.

    So the report states its own coverage rather than leaving the reader to
    infer completeness from a confident-looking record.
    """
    replayed = sorted(set(frame["game_date"].astype(str)))
    out = {"games": len(frame), "dates": len(replayed),
           "first": replayed[0] if replayed else None,
           "last": replayed[-1] if replayed else None,
           "missing_dates": [], "missing_games": 0, "partial": False}
    if source is not None and len(source):
        asked = sorted(set(source["game_date"].astype(str)))
        missing = [d for d in asked if d not in set(replayed)]
        out["missing_dates"] = missing
        out["missing_games"] = int(source["game_date"].astype(str).isin(missing).sum())
        out["asked_games"] = len(source)
        out["asked_dates"] = len(asked)
        out["partial"] = bool(missing)
    return out


def render_report(frame, source=None):
    decided = frame[frame["wf_lean"].notna()].copy()
    wins = (decided["wf_full_grade"] == "W").astype(float)
    probs = pd.to_numeric(decided.get("wf_market_pick_prob"), errors="coerce")
    valid_market = probs.notna() & decided["wf_full_grade"].isin(["W", "L"])
    expected = float(probs[valid_market].sum())
    observed = float(wins[valid_market].sum())
    variance = float((probs[valid_market] * (1.0 - probs[valid_market])).sum())
    z_score = (observed - expected) / math.sqrt(variance) if variance > 0 else np.nan
    roi = pd.to_numeric(decided.get("wf_roi"), errors="coerce")
    roi_total = float(roi.sum(skipna=True))
    roi_n = int(roi.notna().sum())

    cov = coverage(frame, source)
    lines = [
        "CURRENT MODEL WALK-FORWARD",
        "",
    ]
    if cov["partial"]:
        lines += [
            "*** PARTIAL REPLAY -- every figure below covers only the dates "
            "listed under COVERAGE. ***",
            "*** Dates replay in order, so what is missing is the MOST RECENT "
            "window, not a random sample. ***",
            "",
        ]
    lines += [
        f"Model       {model.MODEL_TAG}",
        f"Games       {len(frame)}",
        f"Decisions   {len(decided)}",
        f"Full        {_record(decided['wf_full_grade'])}",
        f"F5          {_record(decided['wf_f5_grade'])}",
        f"Market exp  {expected:.1f} wins on {int(valid_market.sum())} priced decisions",
        f"Market z    {z_score:+.2f}" if pd.notna(z_score) else "Market z    —",
        (f"Flat ROI    {roi_total:+.2f}u / {roi_n} bets "
         f"({100 * roi_total / roi_n:+.1f}%)" if roi_n else "Flat ROI    —"),
        "",
        "COVERAGE",
        f"Replayed    {cov['games']} games over {cov['dates']} dates"
        + (f", {cov['first']} .. {cov['last']}" if cov["first"] else ""),
    ]
    if source is not None and len(source):
        lines.append(
            f"Ledger      {cov['asked_games']} games over {cov['asked_dates']} dates")
        if cov["partial"]:
            shown = ", ".join(cov["missing_dates"][:6])
            more = ("" if len(cov["missing_dates"]) <= 6
                    else f" (+{len(cov['missing_dates']) - 6} more)")
            lines.append(
                f"NOT REPLAYED {cov['missing_games']} games over "
                f"{len(cov['missing_dates'])} dates: {shown}{more}")
        else:
            lines.append("Complete    every ledger date in range was replayed")
    lines += [
        "",
        "INPUT FIDELITY",
    ]
    for label, group in frame.groupby("input_fidelity", dropna=False):
        picked = group[group["wf_lean"].notna()]
        lines.append(
            f"{str(label):<22} n={len(group):>3}  full {_record(picked['wf_full_grade'])}"
        )
    lines.extend(["", "BY MONTH"])
    month = frame["game_date"].astype(str).str[:7]
    for label, group in frame.groupby(month):
        picked = group[group["wf_lean"].notna()]
        lines.append(
            f"{label:<22} n={len(group):>3}  full {_record(picked['wf_full_grade'])}"
        )
    changed = frame["changed_from_original"].fillna(False).astype(bool)
    lines.extend([
        "",
        "ORIGINAL HISTORICAL MODELS",
        f"Full        {_record(frame['original_full_grade'])}",
        f"Changed     {int(changed.sum())} of {int(frame['changed_from_original'].notna().sum())} comparable decisions",
        "",
        "Method: current specification replayed sequentially with dated inputs;",
        "this removes temporal feature leakage but not model-selection leakage.",
        "",
    ])
    return "\n".join(lines)


def run_acceptance(ledger_path, slate_date, cache_dir, tolerance,
                   snapshot_dir=None):
    source = pd.read_csv(ledger_path)
    source["game_date"] = source["game_date"].astype(str).str[:10]
    day = source[source["game_date"] == str(slate_date)[:10]].copy()
    if day.empty:
        raise ValueError(f"no ledger rows for acceptance date {slate_date}")
    # This parity check reproduces what the live v12 build knew on that date.
    # The actual walk-forward runner instead calibrates on prior replay rows.
    prior = source[source["game_date"] < str(slate_date)[:10]].copy()
    predictions, provider = predict_slate(
        str(slate_date)[:10], day, calibration_history(prior), cache_dir,
        snapshot_dir=snapshot_dir,
    )
    comparisons = []
    for _, row in day.iterrows():
        prediction = predictions.get(int(row["game_pk"]))
        original = _num(row.get("xw_net"))
        regenerated = np.nan if prediction is None else _num(prediction.get("xw_net"))
        if str(row.get("model_tag")) == model.MODEL_TAG and pd.notna(original) and pd.notna(regenerated):
            comparisons.append({
                "game_pk": int(row["game_pk"]),
                "original": original,
                "regenerated": regenerated,
                "abs_error": abs(regenerated - original),
                "fidelity": provider.input_fidelity(int(row["game_pk"])),
            })
    result = pd.DataFrame(comparisons)
    if result.empty:
        raise RuntimeError("acceptance date has no comparable current-v12 rows")
    print(result.to_string(index=False))
    exact = result[result["fidelity"] == "exact_pregame"]
    if exact.empty:
        raise RuntimeError(
            "acceptance date has no exact-pregame inputs; restore the dated "
            "Actions Savant cache and pass --snapshot-dir"
        )
    max_error = float(exact["abs_error"].max())
    print(f"acceptance max |d xw_net| = {max_error:.6f} (tolerance {tolerance:.6f})")
    if max_error > tolerance:
        raise RuntimeError("recent-slate reconstruction parity failed")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--snapshot-dir",
        help="directory containing archived savant_cache_* files for exact parity",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--with-platoon", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--acceptance-date")
    parser.add_argument("--acceptance-tolerance", type=float, default=0.002)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.acceptance_date:
        run_acceptance(
            args.ledger, args.acceptance_date, args.cache_dir,
            args.acceptance_tolerance, snapshot_dir=args.snapshot_dir,
        )
        return 0
    frame, report = run_walkforward(
        ledger_path=args.ledger,
        output_path=args.output,
        report_path=args.report,
        cache_dir=args.cache_dir,
        snapshot_dir=args.snapshot_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        include_pending=args.include_pending,
        include_platoon=args.with_platoon,
        resume=not args.no_resume,
    )
    print(report)
    print(f"Wrote {args.output} ({len(frame)} rows) and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
