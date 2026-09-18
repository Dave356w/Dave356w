"""Write the v13 starter-blend RECONSTRUCTION onto earlier-family ledger rows.

WHAT THIS IS, AND THE ONE THING IT IS NOT
-----------------------------------------
v13 blends each starter's wOBA-allowed line into his xwOBA-allowed one. To show
that model over history the site needs a v13 `net` for games that were decided
under v12, and both halves of that net are recoverable: the xwOBA half from the
committed primary dump, the wOBA half from the paired shadow dump written
beside it in the same job.

**It is not a backtest and the rows it writes are not decisions.** The script
measures the split rather than asserting it, and the measurement is worse than
a clean story: MOST paired shadow rows were written after their own first pitch
(the run prints the exact count), so the wOBA rate behind most reconstructed
selections was read off a leaderboard the game had already finished inside. A
minority were genuinely pregame -- and those are not clean either, because the
xwOBA half beside them is the pregame value the ledger locked while the wOBA
half came from a dump written at a different moment. Every reconstructed lean
is therefore mixed-basis at best and hindsight at worst. No bettor could have
taken one, and no surface may render one as a record. That is
why the values land in their own `v13_*_recon` columns under an explicit
`v13_recon_basis`, and why every renderer keys off that basis rather than off
the presence of a number.

WHAT IT REFUSES TO TOUCH
------------------------
`xw_net`, `xw_lean`, `xw_full`, `model_tag` and every market column are
write-once pregame records. Two separate reasons, and either alone is
sufficient: they are immutable history, and they are also the CONTROL the new
model is read against -- overwrite them and the published comparison becomes
v13 against itself. The script asserts it changed none of them before it
writes, and refuses the whole file if it did.

It also never writes a current-family row. Those rows are produced by v13
itself, live and pregame; giving one a "reconstruction" would put a hindsight
value beside a real one under the same name.

USAGE
-----
    python reconstruct_v13.py --dry-run     # report, write nothing
    python reconstruct_v13.py               # write the recon columns
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import csv

import build_site as bs
import blend_probe as bp
import shadow_report as sr
from migrate_hybrid_v2 import _cell

LEDGER = bs.LEDGER_PATH

# Never rewritten, and asserted rather than trusted. `xw_*` are the immutable
# pregame record AND the control; the market columns are write-once by the
# no-lookahead invariant.
PROTECTED = ("xw_net", "xw_lean", "xw_full", "xw_delta", "model_tag",
             "model_metric", "status", "close_p_home", "close_home_ml",
             "close_away_ml", "pregame_p_home", "full_home", "full_away")


def blended_nets():
    """{game_pk: v13 net} over every slate pairing a wOBA dump to an xwOBA one.

    Uses `blend_probe`'s reconstruction, which is self-checked bitwise against
    each shipped arm's own published edge, and `build_site.blend_starter_rate`
    -- the SHIPPED function -- so this can never drift from the live model into
    a second spelling of the blend.
    """
    out, provenance = {}, []
    for d, spath, ppath in sr._slate_dates():
        s, p = bp._sided(spath), bp._sided(ppath)
        if s is None or p is None:
            continue
        by = {sr.dump_metric(s, spath): s, sr.dump_metric(p, ppath): p}
        if "wOBA" not in by or "xwOBA" not in by or len(by) < 2:
            continue
        w, x = by["wOBA"], by["xwOBA"]
        keys = w.index.intersection(x.index)
        if not len(keys):
            continue
        w, x = w.loc[keys], x.loc[keys]
        lg_x, lg_w = bp._num(x, "lg_xwOBA"), bp._num(w, "lg_xwOBA")
        st_x, st_w = bp._num(x, "starter_xwOBA"), bp._num(w, "starter_xwOBA")
        blended = [bs.blend_starter_rate(a, b, c, e)
                   for a, b, c, e in zip(st_x, st_w, lg_x, lg_w)]
        parts = {"opp_xwOBA_vs_sp": bp._num(x, "opp_xwOBA_vs_sp"),
                 "opp_xwOBA_neutral": bp._num(x, "opp_xwOBA_neutral"),
                 "starter_xwOBA": pd.Series(blended, index=x.index),
                 "bullpen_xwOBA": bp._num(x, "bullpen_xwOBA"),
                 "lg_xwOBA": lg_x}
        edge = bp.edge_from_parts(parts, bp._num(x, "sp_share"))
        for gpk, net in bp._nets(edge.to_dict()).items():
            out[int(gpk)] = float(net) if pd.notna(net) else np.nan
        # Provenance measured per slate, not assumed: how many of THIS slate's
        # shadow rows were written before their own game started. Anything
        # below the row count means the wOBA half carries hindsight, and the
        # header prints the total so the claim in the docstring is checkable
        # rather than quoted.
        snap = pd.to_datetime(w.get("snapshot_utc"), errors="coerce", utc=True)
        start = pd.to_datetime(w.get("game_datetime_utc"), errors="coerce",
                               utc=True)
        pre = int((snap < start).sum()) if snap is not None and start is not None else 0
        provenance.append((d, pre, int(len(w))))
    return out, provenance


def reconstruct(led, nets):
    """Return (frame, n_rows). Additive only; protected columns are untouched."""
    out = led.copy()
    # The three string columns are created as object dtype explicitly. A fresh
    # float64 NaN column raises on the first club abbreviation assigned into
    # it, which is a real failure and not a warning.
    _text = (bs.V13_RECON_LEAN_COL, bs.V13_RECON_BASIS_COL)
    for c in bs.V13_RECON_COLUMNS:
        if c not in out.columns:
            out[c] = pd.Series(np.nan, index=out.index,
                               dtype="object" if c in _text else "float64")
    # Keyed on MODEL_TAG, NOT on RECORD_TAGS. Those stopped being the same
    # question when v13 chose to SHARE v12's record line: the shared family
    # is v12 + v13, so `isin(RECORD_TAGS)` would exclude exactly the rows
    # this migration exists to rebuild. What needs a reconstruction is a row
    # not BUILT under v13 math, which is a statement about the row's own tag
    # and about nothing else.
    # RETAINED rows only: in the record family, not built under v13. Both
    # halves are load-bearing. `~eq(MODEL_TAG)` alone rebuilds pre-v12
    # families too, and nothing publishes those -- a v2 row is not in the
    # shared record line and the ledger table no longer renders one, so the
    # cells would be a column carried to no surface. `isin(RECORD_TAGS)`
    # alone rebuilds v13's own rows, which need no reconstruction because
    # they were decided under this math in the first place.
    # Deliberately NOT gated on `status == "graded"`. A retained row that is
    # still pending needs its lean written now: this migration runs once, and
    # a row reconstructed only after it grades would have to wait for a rerun
    # that never comes. Its grade is derived from the finals at read time, so
    # writing the lean early costs nothing and is what keeps a pending v12 row
    # inside the published record on the day it settles.
    # ABSTENTIONS ARE NOT RECONSTRUCTED. A row whose own build published no
    # lean was declined by a rule v13 still runs -- v5 abstains when a side's
    # starter has no measured season line, and v13 changed that starter's
    # RATE, not the gate. Handing one a blended net publishes a selection the
    # live model would refuse. Without this, all 8 of the ledger's
    # `starter_unmeasured_no_lean` rows were rebuilt and scored 4-4 into a
    # published 282-171 whose honest figure is 278-167.
    #
    # `market_backfill.publish_reconstruction` enforces the same rule at READ
    # time, which is what actually protects the surfaces -- this migration
    # runs once and its columns are already written. Both exist because a
    # re-run should not re-create what the reader then has to filter out.
    eligible = (out["model_tag"].isin(bs.RECORD_TAGS)
                & out["xw_lean"].notna()
                & ~out["model_tag"].astype(str).eq(bs.MODEL_TAG)
                & out["game_pk"].astype("Int64").isin(list(nets)))
    n = 0
    for i in out.index[eligible]:
        gpk = int(out.at[i, "game_pk"])
        net = nets.get(gpk)
        if net is None or not np.isfinite(net) or net == 0.0:
            continue
        home, away = out.at[i, "home"], out.at[i, "away"]
        if not isinstance(home, str) or not isinstance(away, str):
            continue
        lean = home if net > 0 else away
        out.at[i, bs.V13_RECON_NET_COL] = net
        out.at[i, bs.V13_RECON_DELTA_COL] = abs(net)
        out.at[i, bs.V13_RECON_LEAN_COL] = lean
        # Named for what it is. Every consumer branches on this string, never
        # on "is there a number here".
        out.at[i, bs.V13_RECON_BASIS_COL] = "post_hoc_shadow_pair"
        n += 1
    return out, n


def _protected_unchanged(before, after):
    """Names of protected columns whose values moved. Empty is the only pass."""
    moved = []
    for c in PROTECTED:
        if c not in before.columns or c not in after.columns:
            continue
        a, b = before[c], after[c]
        same = (a.isna() & b.isna()) | (a.astype(str) == b.astype(str))
        if not bool(same.all()):
            moved.append(c)
    return moved



def append_columns(path, frame, new_cols):
    """Add `new_cols` to the CSV, leaving every existing byte untouched.

    `migrate_hybrid_v2.write_changed_cells` falls back to a whole-file
    `to_csv` when the column SET moves, which is exactly this case -- and a
    whole-file rewrite of this ledger re-renders every float. That is not
    hypothetical: the file holds values written at Python's full repr, and a
    read/write round-trip through pandas has already truncated a digit of
    `xw_net` once. Reading with `float_precision="round_trip"` fixes the read
    half; nothing fixes the WRITE half except not rewriting the cell.

    So the existing rows are copied through verbatim as text and only the new
    trailing fields are rendered. Refuses rather than guesses if the row count
    or an existing header field does not line up.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("reconstruct_v13: empty ledger")
    header = rows[0]
    if len(rows) - 1 != len(frame):
        raise SystemExit(
            f"reconstruct_v13: row count moved ({len(rows) - 1} on disk, "
            f"{len(frame)} in frame); refusing to write")
    add = [c for c in new_cols if c not in header]
    if not add:
        raise SystemExit(
            "reconstruct_v13: these columns already exist; this script writes "
            "them once and is not a re-runnable migration")
    rows[0] = header + list(add)
    n = 0
    for i, idx in enumerate(frame.index, start=1):
        extra = []
        for c in add:
            text = _cell(frame.at[idx, c])
            extra.append(text)
            if text != "":
                n += 1
        rows[i] = rows[i] + extra
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    led = pd.read_csv(LEDGER, float_precision="round_trip", low_memory=False)
    nets, provenance = blended_nets()
    pre = sum(p for _, p, _ in provenance)
    tot = sum(t for _, _, t in provenance)
    print(f"paired slates with a reconstructible blend: {len(provenance)}")
    print(f"games with a v13 net: {len(nets)}")
    print(f"shadow rows written BEFORE their own first pitch: {pre} of {tot}"
          f" ({pre / max(tot, 1):.1%})")
    if pre < tot:
        print("  -> the wOBA half of this reconstruction is post-hoc on "
              f"{tot - pre} side-rows. Hindsight, not a backtest.")
    out, n = reconstruct(led, nets)
    moved = _protected_unchanged(led, out)
    if moved:
        raise SystemExit(
            "reconstruct_v13: refusing to write -- these are immutable pregame "
            f"records and the reconstruction moved them: {', '.join(moved)}")
    print(f"rows given a v13 reconstruction: {n}")
    if n:
        # Graded through the same function every surface reads, so this
        # summary cannot be a second spelling of the rule.
        out["_g"] = bs._recon_grades(out)
        g = out[out["_g"].isin(["W", "L"])]
        w = int((g["_g"] == "W").sum())
        l = int((g["_g"] == "L").sum())
        lw = int((g["xw_full"] == "W").sum())
        ll = int((g["xw_full"] == "L").sum())
        flips = int((g[bs.V13_RECON_LEAN_COL] != g["xw_lean"]).sum())
        print(f"  v13 reconstructed {w}-{l} ({w / max(w + l, 1):.3f})  "
              f"against the published lean's {lw}-{ll} "
              f"({lw / max(lw + ll, 1):.3f}) on the SAME rows")
        print(f"  leans flipped: {flips} of {w + l}")
        out = out.drop(columns=["_g"])
        print("  NOT A RECORD. The wOBA half is post-hoc on most of these "
              "rows (the split is printed above), and a reconstruction is "
              "mixed-basis even where it is not -- the xwOBA half is the "
              "pregame value the ledger locked. Hindsight, not a result.")
    if a.dry_run:
        print("dry run -- nothing written")
        return 0
    written = append_columns(LEDGER, out, bs.V13_RECON_COLUMNS)
    print(f"new cells written: {written}; every pre-existing byte left as it was")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
