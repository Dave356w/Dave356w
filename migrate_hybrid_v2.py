"""One-time, idempotent migration of v12 Hybrid selection fields to v2.

Legacy rows use their existing closing market. Rows with a captured two-sided
pregame market keep that snapshot. Model fields, raw lean grades, scores,
status, dates, and eligibility are never changed.
"""
import os

import numpy as np
import pandas as pd

import grade_leans
import hybrid_test
import hybrid_v2


PRICE_SOURCE_COL = "hybrid_price_source"
V1_ARCHIVE = ["hybrid_v1_action", "hybrid_v1_selection", "hybrid_v1_p",
              "hybrid_v1_ml", "hybrid_v1_full"]


def _valid_probability(v):
    return pd.notna(v) and np.isfinite(float(v)) and 0.0 < float(v) < 1.0


def _valid_ml(v):
    return pd.notna(v) and np.isfinite(float(v)) and abs(float(v)) >= 100


def migrate(led):
    """Return a migrated copy and a count of rows whose Hybrid fields changed."""
    out = led.copy()
    if PRICE_SOURCE_COL not in out.columns:
        insert_at = out.columns.get_loc("hybrid_ml") + 1
        out.insert(insert_at, PRICE_SOURCE_COL,
                   pd.Series([None] * len(out), dtype=object))
    else:
        out[PRICE_SOURCE_COL] = out[PRICE_SOURCE_COL].astype(object)
    archive_strings = {"hybrid_v1_action", "hybrid_v1_selection",
                       "hybrid_v1_full"}
    for col in V1_ARCHIVE:
        if col not in out.columns:
            out[col] = (pd.Series([None] * len(out), dtype=object)
                        if col in archive_strings else np.nan)
        elif col in archive_strings:
            out[col] = out[col].astype(object)
    hybrid_cols = ["selection_rule_tag", "hybrid_action", "hybrid_selection",
                   "hybrid_p", "hybrid_ml", "hybrid_full", PRICE_SOURCE_COL]
    hybrid_cols += V1_ARCHIVE
    before = out[hybrid_cols].copy()

    current = out["model_tag"].astype(str).eq(grade_leans.MODEL_TAG)
    for idx, row in out[current].iterrows():
        lean, home, away = row.get("xw_lean"), row.get("home"), row.get("away")
        delta = pd.to_numeric(row.get("xw_net"), errors="coerce")
        out.at[idx, "selection_rule_tag"] = hybrid_v2.RULE_TAG
        if not isinstance(lean, str) or lean not in (home, away) or pd.isna(delta):
            for col in ("hybrid_action", "hybrid_selection", "hybrid_p",
                        "hybrid_ml", "hybrid_full", PRICE_SOURCE_COL):
                out.at[idx, col] = np.nan
            continue

        pregame = (_valid_probability(row.get("pregame_p_home"))
                   and _valid_ml(row.get("pregame_home_ml"))
                   and _valid_ml(row.get("pregame_away_ml")))
        if pregame:
            ph, hml, aml, source = (float(row["pregame_p_home"]),
                                      float(row["pregame_home_ml"]),
                                      float(row["pregame_away_ml"]),
                                      "saved_pregame")
            # Preserve the rule that was actually registered and live when
            # these snapshots were captured. It remains saved-pregame only.
            old_q = ph if lean == home else 1.0 - ph
            old_action = "FOLLOW" if old_q >= hybrid_test.THRESHOLD else "FADE"
            old_pick = (lean if old_action == "FOLLOW"
                        else (away if lean == home else home))
            out.at[idx, "hybrid_v1_action"] = old_action
            out.at[idx, "hybrid_v1_selection"] = old_pick
            out.at[idx, "hybrid_v1_p"] = (old_q if old_action == "FOLLOW"
                                            else 1.0 - old_q)
            out.at[idx, "hybrid_v1_ml"] = hml if old_pick == home else aml
            out.at[idx, "hybrid_v1_full"] = (
                grade_leans._wlt(old_pick, away, home, row.get("full_away"),
                                 row.get("full_home"), False)
                if row.get("status") == "graded" else np.nan)
        else:
            for col in V1_ARCHIVE:
                out.at[idx, col] = np.nan
            ph = pd.to_numeric(row.get("close_p_home"), errors="coerce")
            hml = pd.to_numeric(row.get("close_home_ml"), errors="coerce")
            aml = pd.to_numeric(row.get("close_away_ml"), errors="coerce")
            if not (_valid_probability(ph) and _valid_ml(hml) and _valid_ml(aml)):
                for col in ("hybrid_action", "hybrid_selection", "hybrid_p",
                            "hybrid_ml", "hybrid_full", PRICE_SOURCE_COL):
                    out.at[idx, col] = np.nan
                continue
            ph, hml, aml, source = float(ph), float(hml), float(aml), "closing"

        q = ph if lean == home else 1.0 - ph
        action = "FOLLOW" if bool(hybrid_v2.follows(q, delta)) else "FADE"
        pick = lean if action == "FOLLOW" else (away if lean == home else home)
        out.at[idx, "hybrid_action"] = action
        out.at[idx, "hybrid_selection"] = pick
        out.at[idx, "hybrid_p"] = q if action == "FOLLOW" else 1.0 - q
        out.at[idx, "hybrid_ml"] = hml if pick == home else aml
        out.at[idx, PRICE_SOURCE_COL] = source
        if row.get("status") == "graded":
            out.at[idx, "hybrid_full"] = grade_leans._wlt(
                pick, away, home, row.get("full_away"), row.get("full_home"), False)
        else:
            out.at[idx, "hybrid_full"] = np.nan

    differences = pd.DataFrame(False, index=out.index, columns=hybrid_cols)
    numeric = {"hybrid_p", "hybrid_ml", "hybrid_v1_p", "hybrid_v1_ml"}
    for col in hybrid_cols:
        if col in numeric:
            a = pd.to_numeric(before[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
            differences[col] = ~np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-12)
        else:
            differences[col] = (before[col].fillna("<NA>").astype(str)
                                != out[col].fillna("<NA>").astype(str))
    changed = int(differences.any(axis=1).sum())
    persisted = list(dict.fromkeys(
        grade_leans.LEDGER_COLS + grade_leans.MARKET_COLS
        + grade_leans.AUDIT_COLS + grade_leans.ACTUAL_COLS))
    ordered = ([c for c in persisted if c in out.columns]
               + [c for c in out.columns if c not in persisted])
    return out[ordered], changed


def main(path=grade_leans.LEDGER_PATH):
    led = pd.read_csv(path, low_memory=False)
    migrated, changed = migrate(led)
    migrated.to_csv(path, index=False)
    print(f"hybrid v2 migration: {changed} row(s) changed in {path}")


if __name__ == "__main__":
    main()
