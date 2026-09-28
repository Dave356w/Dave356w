"""Re-decide every earlier record-family ledger row under v14 (velocity term).

v14 = v13 + the starter velocity term (`starter_velocity.py`, BETA_V frozen
from 2023-2025). For each v12/v13 row this writes the v14 net, lean and delta
into `velo_*_recon` columns:

    base net   v13-built row: its own `xw_net`
               v12-built row: its v13 re-decision `v13_net_recon`
    + delta    `starter_velocity.net_delta` -- exact, because the starter
               phase is log5 and linear in the starter rate

Each starter's dv comes from that season's Statcast (fetched in CI by
`research/statcast_history.py`), using only his starts BEFORE the game. No dv
(too few earlier starts) adds zero, as it does live.

Nothing protected moves: `xw_*`, grades, tags and market columns are asserted
unchanged, and existing cells are copied through byte for byte -- only the
`velo_*_recon` cells are written. Re-runnable: a re-run rewrites those cells
and nothing else. Abstained rows are never re-decided.

    python reconstruct_v14_velocity.py --history history_cache --dry-run
    python reconstruct_v14_velocity.py --history history_cache
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import pandas as pd

import build_site as bs
import starter_velocity as sv
from market_backfill import VELO_RECON_COLS
from migrate_hybrid_v2 import _cell
from research import statcast_history as sh

LEDGER = bs.LEDGER_PATH
PROTECTED = ("xw_net", "xw_lean", "xw_full", "xw_delta", "model_tag",
             "model_metric", "status", "close_p_home", "close_home_ml",
             "close_away_ml", "pregame_p_home", "full_home", "full_away",
             "v13_net_recon", "v13_lean_recon")
V13_TAG = "xw+starter_blend_v13"


def starter_dv(pa: pd.DataFrame, velo: pd.DataFrame) -> dict:
    """{(game_pk, 'home'|'away'): dv} for every start in one season."""
    if pa.empty or velo.empty:
        return {}
    st = sh.starters(pa).merge(
        pa[["game_pk", "game_date", "home_team"]].drop_duplicates("game_pk"),
        on="game_pk")
    st["side"] = np.where(st["fld_team"] == st["home_team"], "home", "away")
    # One start table per pitcher, in the shape starter_velocity reads.
    starts = (st.merge(velo.rename(columns={"pitcher": "starter"}),
                       on=["game_pk", "game_date", "starter"])
                [["starter", "game_date", "game_pk", "velo", "n_fb"]]
                .sort_values(["starter", "game_date", "game_pk"]))
    by = {p: g.drop(columns="starter").reset_index(drop=True)
          for p, g in starts.groupby("starter")}
    out = {}
    for r in st.itertuples(index=False):
        t = sv.pregame_trend(by.get(r.starter, pd.DataFrame()), r.game_date)
        out[(int(r.game_pk), r.side)] = t["dv"]
    return out


def reconstruct(led: pd.DataFrame, dv: dict):
    """Return (frame, n). Additive only; protected columns are untouched."""
    out = led.copy()
    for c in VELO_RECON_COLS:
        out[c] = pd.Series(np.nan, index=out.index,
                           dtype="object" if c in ("velo_lean_recon",
                                                   "velo_recon_basis") else "float64")
    tag = out["model_tag"].astype(str)
    eligible = (tag.isin(bs.RECORD_TAGS) & ~tag.eq(bs.MODEL_TAG)
                & out["xw_lean"].notna())

    def num(i, c):
        v = pd.to_numeric(out.at[i, c], errors="coerce") if c in out.columns else np.nan
        return float(v) if pd.notna(v) else np.nan

    n = 0
    for i in out.index[eligible]:
        if tag[i] == V13_TAG:
            base, basis = num(i, "xw_net"), "v13_native+velocity"
        else:
            base, basis = num(i, "v13_net_recon"), "v13_recon+velocity"
        if not np.isfinite(base):
            continue
        gpk = int(out.at[i, "game_pk"])
        dva, dvh = dv.get((gpk, "away")), dv.get((gpk, "home"))
        L = num(i, "mx_xwoba_sp_away") - num(i, "edge_xwoba_sp_away")
        d = 0.0
        if np.isfinite(L) and L > 0:
            d = sv.net_delta(dvh, dva, num(i, "sp_share_home"), num(i, "sp_share_away"),
                             num(i, "opp_xwoba_vs_sp_home"),
                             num(i, "opp_xwoba_vs_sp_away"), L)
            d = d if np.isfinite(d) else 0.0
        net = base + d
        home, away = out.at[i, "home"], out.at[i, "away"]
        if net == 0.0 or not isinstance(home, str) or not isinstance(away, str):
            continue
        out.at[i, "velo_dv_away_recon"] = dva if dva is not None else np.nan
        out.at[i, "velo_dv_home_recon"] = dvh if dvh is not None else np.nan
        out.at[i, "velo_net_recon"] = net
        out.at[i, "velo_delta_recon"] = abs(net)
        out.at[i, "velo_lean_recon"] = home if net > 0 else away
        out.at[i, "velo_recon_basis"] = basis
        n += 1
    return out, n


def _protected_unchanged(before, after):
    moved = []
    for c in PROTECTED:
        if c in before.columns and c in after.columns:
            a, b = before[c], after[c]
            if not bool(((a.isna() & b.isna()) | (a.astype(str) == b.astype(str))).all()):
                moved.append(c)
    return moved


def write_columns(path, frame, cols):
    """Write `cols` into the CSV, every other byte copied through verbatim.

    Appends the columns when absent; rewrites only their cells when present
    (the re-run case). Refuses if the row count does not line up.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    if len(rows) - 1 != len(frame):
        raise SystemExit(f"row count moved ({len(rows) - 1} on disk, "
                         f"{len(frame)} in frame); refusing to write")
    pos = {}
    for c in cols:
        if c not in header:
            header.append(c)
            for r in rows[1:]:
                r.append("")
        pos[c] = header.index(c)
    n = 0
    for i, idx in enumerate(frame.index, start=1):
        r = rows[i]
        if len(r) < len(header):
            r.extend([""] * (len(header) - len(r)))
        for c, j in pos.items():
            r[j] = _cell(frame.at[idx, c])
            n += r[j] != ""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", default=sh.DEFAULT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    led = pd.read_csv(LEDGER, float_precision="round_trip", low_memory=False)
    years = sorted({str(d)[:4] for d in led.loc[
        led["model_tag"].isin(bs.RECORD_TAGS), "game_date"].dropna()})
    dv = {}
    for y in years:
        d = sh.load(int(y), a.history)
        if d["pa"].empty:
            raise SystemExit(f"no Statcast history for {y} in {a.history}")
        dv.update(starter_dv(d["pa"], d["velo"]))
    print(f"starts with a velocity lookup: {len(dv)}; "
          f"with a pregame trend: {sum(v is not None for v in dv.values())}")

    out, n = reconstruct(led, dv)
    moved = _protected_unchanged(led, out)
    if moved:
        raise SystemExit(f"refusing to write -- protected columns moved: {moved}")
    re = out[out["velo_recon_basis"].notna()]
    flips = int((re["velo_lean_recon"] != re["xw_lean"].where(
        re["model_tag"].eq(V13_TAG), re.get("v13_lean_recon"))).sum())
    print(f"rows re-decided under {bs.MODEL_TAG}: {n}; leans changed by the "
          f"velocity term: {flips}")
    if a.dry_run:
        print("dry run -- nothing written")
        return 0
    cells = write_columns(LEDGER, out, VELO_RECON_COLS)
    print(f"cells written: {cells}; every other byte left as it was")
    return 0


if __name__ == "__main__":
    sys.exit(main())
