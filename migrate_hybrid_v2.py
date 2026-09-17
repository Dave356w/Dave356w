"""One-time, idempotent migration of v12 Hybrid selection fields to v2.

Legacy rows use their existing closing market. Rows with a captured two-sided
pregame market keep that snapshot. Model fields, raw lean grades, scores,
status, dates, and eligibility are never changed.
"""
import csv
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

    # A re-run REFRESHES the v1 archive it already wrote, and MINTS one for a
    # row that carries none. Both halves matter and they fix opposite failures.
    #
    # Refreshing, because the archive describes a row and `grade_leans`
    # rebuilds a pending row's model and market fields on every pregame poll
    # (MODEL_FIELDS carries xw_lean and pregame_p_home) while carrying no
    # hybrid_v1_* entry. A row pending when this migration ran therefore froze
    # its archive at a state the final lock superseded. Measured on the
    # 2026-09-11 slate: 12 of 13 rows kept a stale probability and 2 kept a
    # stale LEAN (CWS@STL at xw_net +0.0043, BAL@TOR at -0.0027 -- both thin
    # enough to flip on a lineup refresh), so the archive named a selection
    # nobody locked. Deriving from the row's final state is the same rule the
    # first migration applied to rows that were already graded, hence already
    # final; it just could not hold for the ones that were not.
    #
    # MINTING per row, changed 2026-09-17 on the operator's call, reversing the
    # all-or-nothing `minting` flag this loop used to carry. That flag read
    # "mint only if no row anywhere has an archive", so it was False from the
    # first migration onward and every row written after 2026-09-11 got none.
    # v1 is indeed a retired registration and widening it was the thing the old
    # comment set out to avoid -- but hybrid_test is not the only reader. The
    # archive is also the ROW SELECTOR for abstain_test and dog_contrast_test,
    # which are LIVE registrations with open gates, and both delegate to
    # `hybrid_test.scored_rows`. So the flag silently froze two live tests at
    # 2026-09-11: abstain at 5 declined games of a gate of 82, with a decision
    # pre-committed on 2026-09-16 that could never fire, and dog contrast at 16
    # dog leans of 88. Measured over 09-12..09-16 alone, the freeze cost 11
    # declined games and 18 dog leans -- more than doubling both.
    #
    # What is minted is not new evidence and not a re-pointed selector: the v1
    # decision is a pure function of `xw_lean` and the locked pregame market,
    # all write-once once a row grades, so this computes what live minting
    # would have stored. Verified rather than asserted -- re-deriving the 141
    # archives that WERE minted live reproduces every field on all 141.
    # `hybrid_test.locked_v1_decision` is the single derivation; `grade_leans`
    # now mints from the same call at ingest, so the gap cannot reopen and this
    # migration is a repair rather than the mechanism.

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
            v1 = hybrid_test.locked_v1_decision(lean, home, away, ph, hml, aml)
            if v1 is not None:
                old_action, old_pick, old_p, old_ml = v1
                out.at[idx, "hybrid_v1_action"] = old_action
                out.at[idx, "hybrid_v1_selection"] = old_pick
                out.at[idx, "hybrid_v1_p"] = old_p
                out.at[idx, "hybrid_v1_ml"] = old_ml
                # Ungraded stays NaN here, but no longer permanently: the
                # grader now writes hybrid_v1_full from this selection when
                # the row grades, so a row pending at a migration keeps its
                # archive instead of losing it.
                out.at[idx, "hybrid_v1_full"] = (
                    grade_leans._wlt(old_pick, away, home,
                                     row.get("full_away"),
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
    out = out[ordered]
    # `gamePk` carries NaN on rows with no market join, so read_csv gives
    # float64 and a plain rewrite renders every one of them `822884.0`. The
    # build does not, because `attach_market` owns the `Int64` cast -- and this
    # migration does not run it.
    if "gamePk" in out.columns:
        out["gamePk"] = out["gamePk"].astype("Int64")
    return out, changed


def _cell(v):
    """One cell as `to_csv` would render it, for a surgical rewrite.

    Rendering follows the VALUE's type, never its magnitude: an integral float
    is `4.0` in a float column and `4` in an integer one, and collapsing the
    first to `4` rewrote 3013 cells on the first attempt instead of 400.
    """
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def write_changed_cells(path, migrated):
    """Rewrite only the cells whose VALUE the migration changed. Returns n.

    A migration must leave the cells it did not mean to touch byte-identical,
    and `DataFrame.to_csv` cannot promise that: a read/write round-trip of this
    ledger rewrites every cell in its own float formatting. Two ways that bit,
    both found by diffing the 2026-09-17 backfill rather than by reading code.
    `gamePk` is fixed above, at the dtype. The other is not fixable here --
    pandas renders a float to 16 decimal places while `build_site` writes
    Python's full repr, so the 8 rows of the slate that had not yet been
    through a ledger rewrite lost a digit of `d_lineup`, `d_sp` and `xw_net`.
    That truncation is what every ordinary build already does, which is why no
    older row shows the longer form; it is still not this migration's to do,
    and 28 unexplained cell changes in a repair's diff are 28 places a reviewer
    has to take on trust.

    So the frame decides WHAT changed and the original file supplies every
    other byte. Falls back to a whole-file write when the column set or row
    count moved, since then there is no cell-for-cell correspondence to use --
    a first migration adding the archive columns takes that path.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0] if rows else []
    if header != list(migrated.columns) or len(rows) - 1 != len(migrated):
        migrated.to_csv(path, index=False)
        return -1
    n = 0
    for i, idx in enumerate(migrated.index, start=1):
        for j, col in enumerate(migrated.columns):
            text = _cell(migrated.at[idx, col])
            if text != rows[i][j] and not _same_value(rows[i][j], text):
                rows[i][j] = text
                n += 1
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return n


def _same_value(a, b):
    """True when two renderings denote the same value (so neither is written).

    THE TOLERANCE IS `migrate`'s OWN, deliberately, so the row count it reports
    and the cells this writes cannot disagree about what changed. Below it sit
    two separate last-bit effects, both pre-existing and neither this
    migration's to commit:

      * a re-derived field coming back one ULP off the stored one, because the
        CSV rounded the value when it was written and the migration recomputes
        from the rounded input -- 36 graded rows' `hybrid_p`, on immutable
        rows the migration itself counts as unchanged;
      * `read_csv`'s default float parser, which is inexact: it reads this
        ledger's `0.0008232366754536979` as `...4536`, so a rewrite truncates
        any cell the build wrote at full repr and has not round-tripped yet.
        That is every reader in this repo, not this one, and the fix is
        `float_precision="round_trip"` at the reads -- a change with the whole
        ledger downstream of it, not something to smuggle into a backfill.
    """
    if a == b:
        return True
    try:
        return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-12))
    except ValueError:
        return False


def main(path=grade_leans.LEDGER_PATH):
    led = pd.read_csv(path, low_memory=False)
    migrated, changed = migrate(led)
    cells = write_changed_cells(path, migrated)
    where = "whole file rewritten" if cells < 0 else f"{cells} cell(s) written"
    print(f"hybrid v2 migration: {changed} row(s) changed in {path} ({where})")


if __name__ == "__main__":
    main()
