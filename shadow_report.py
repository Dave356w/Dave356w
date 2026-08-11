"""Read the paired xwOBA shadow dumps and report what they can and cannot settle.

WHY A SCRIPT AND NOT A NOTE
---------------------------
`shadow_metric.py` writes one dump per slate and stops; nothing reads them. The
first two slates are on disk, so the question "has the arm produced anything
yet" was answerable only by hand -- which is how a number ends up frozen into a
markdown file and quoted against a distribution it was never measured on (see
CLAUDE.md, "Constants frozen from data"). This prints the comparison instead, so
the answer is recomputed rather than remembered.

WHAT IT PAIRS, AND WHY THAT PAIRING AND NOT THE OBVIOUS ONE
-----------------------------------------------------------
The obvious join -- shadow dump against the LEDGER's wOBA rows -- is wrong, and
quietly so. Ledger rows are locked pregame; the committed dump for any past
slate is a post-first-pitch rebuild (`SLATE_DATE` rolls at 3am ET and the 4:17am
grading pass re-runs the whole build for yesterday against today's leaderboard).
So that join compares a pregame wOBA decision against an xwOBA decision made
with an extra day of Savant behind it, and any gap it finds is partly lookahead.

This reads the wOBA net out of the dump sitting BESIDE the shadow dump in the
same commit instead. Those two are written seconds apart from the same
leaderboard, so the pairing is honest even when both are post-hoc -- which is
the whole point of the arm: same games, same players, same morning, one metric
apart. The provenance of each slate is printed rather than assumed, and
`--ledger-join` will run the contaminated version on request so the size of the
contamination is measurable instead of argued.

WHAT IT WILL NOT DO
-------------------
Nothing here writes a dump, a ledger row, or a page, and the shadow tag is
asserted at runtime to be absent from both family maps before any pooling
happens. A shadow row shares a record line and a delta scale with nothing.

USAGE
-----
    python shadow_report.py                 # committed dumps as they stand
    python shadow_report.py --ledger-join   # also print the contaminated join
    python shadow_report.py --boot 20000    # bootstrap resamples (default 20000)
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import build_site as bs
import shadow_metric as sm

DATA_DIR = getattr(bs, "DATA_DIR", "data")
LEDGER = os.path.join(DATA_DIR, "mlb_lean_ledger.csv")

# The design target from shadow_metric's docstring: the se on d_corr that makes
# a 0.09 gap resolvable. Kept here as the yardstick the projection reports
# against, NOT as a measured quantity -- it is an argument, not an observation.
TARGET_SE = 0.045
TARGET_GAP = 0.09


def _slate_dates():
    """Dates that have BOTH a shadow dump and a primary dump beside it."""
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, f"{sm.SHADOW_PREFIX}_*_xw.csv"))):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_xw\.csv$", os.path.basename(p))
        if not m:
            continue
        d = m.group(1)
        primary = [q for q in
                   (os.path.join(DATA_DIR, f"leans_{d}_{s}.csv")
                    for s in ("woba", "xw", "split"))
                   if os.path.exists(q)]
        if primary:
            out.append((d, p, primary[0]))
    return out


def game_nets(df):
    """One row per game: net = away-row edge minus home-row edge.

    Mirrors grade_leans.rows_from_dump's construction (`home_off - away_off`,
    where home_off is the AWAY row's edge). Positive favours the home side. A
    missing edge on either side yields NaN -- an abstention, not a zero.
    """
    rows = []
    for gpk, gg in df.groupby("game_pk", sort=False):
        a, h = gg[gg["side"] == "away"], gg[gg["side"] == "home"]
        if not len(a) or not len(h):
            continue
        a, h = a.iloc[0], h.iloc[0]
        ha, aw = a.get("edge_xwOBA"), h.get("edge_xwOBA")
        net = (float(ha) - float(aw)) if pd.notna(ha) and pd.notna(aw) else np.nan
        rows.append({
            "game_pk": int(gpk),
            "matchup": a.get("matchup"),
            "net": net,
            "snapshot_utc": a.get("snapshot_utc"),
            "scheduled_start_utc": a.get("scheduled_start_utc", a.get("game_datetime_utc")),
        })
    return pd.DataFrame(rows).set_index("game_pk")


def _provenance(nets):
    """('pregame'|'post-first-pitch'|'unknown', earliest snapshot, first pitch).

    Stated, never assumed: a rebuilt dump has a full extra day of StatsAPI and
    Savant behind it, and a comparison that mixes one with a pregame row on the
    other side is measuring the extra day.
    """
    snap, start = nets["snapshot_utc"].min(), nets["scheduled_start_utc"].min()
    if not isinstance(snap, str) or not isinstance(start, str):
        return "unknown", snap, start
    late = pd.Timestamp(snap) >= pd.Timestamp(start)
    return ("post-first-pitch" if late else "pregame"), snap, start


def _record(net, margin):
    """(w, l) straight-up on decided games with a decided net. Ties drop."""
    lean_home = net > 0
    decided = (margin != 0) & net.notna() & (net != 0)
    hit = (lean_home & (margin > 0)) | (~lean_home & (margin < 0))
    return int((hit & decided).sum()), int((~hit & decided).sum())


def _corr(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _d_corr_bootstrap(df, n_boot, seed=0):
    """Paired bootstrap on corr(xwOBA net, margin) - corr(wOBA net, margin).

    Paired is the entire argument for the arm: the two nets correlate ~0.95, so
    resampling GAMES (not arms independently) cancels most of the schedule
    variance that made the sequential era comparison useless.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        d = _corr(s["net_x"], s["margin"]) - _corr(s["net_w"], s["margin"])
        if not np.isnan(d):
            out.append(d)
    return np.array(out)


def build_frame(pairs, use_ledger_net=False):
    # The ledger holds two rows for a postponed game: the original date `void`
    # and the makeup date `graded` (6 game_pks, 12 rows as of 2026-08-11), so
    # reindexing on it raw raises. Keep the last row per game_pk -- the makeup,
    # since rows append in date order -- rather than joining silently.
    led = pd.read_csv(LEDGER)
    led = led.drop_duplicates(subset="game_pk", keep="last").set_index("game_pk")
    frames, prov = [], []
    for d, spath, ppath in pairs:
        xs, ws = game_nets(pd.read_csv(spath)), game_nets(pd.read_csv(ppath))
        xkind, xsnap, xstart = _provenance(xs)
        prov.append((d, os.path.basename(spath), xkind, xsnap, xstart,
                     _provenance(ws)[0]))
        f = pd.DataFrame({"net_w": ws["net"], "net_x": xs["net"]})
        f["matchup"], f["date"] = xs["matchup"], d
        frames.append(f)
    f = pd.concat(frames)
    g = led.reindex(f.index)
    f["net_led"] = pd.to_numeric(g["xw_net"], errors="coerce")
    f["status"] = g["status"]
    f["margin"] = (pd.to_numeric(g["full_home"], errors="coerce")
                   - pd.to_numeric(g["full_away"], errors="coerce"))
    if use_ledger_net:
        f["net_w"] = f["net_led"]
    return f, prov


def report(n_boot=20000, ledger_join=False):
    # Runtime guard, not decoration: everything below pools rows into records
    # and correlations, and a shadow row belongs in neither family.
    assert sm.SHADOW_TAG not in bs._RECORD_FAMILIES, "shadow tag entered the record map"
    assert sm.SHADOW_TAG not in bs._SCALE_FAMILIES, "shadow tag entered the scale map"

    pairs = _slate_dates()
    if not pairs:
        print("shadow report: no paired slates on disk yet")
        return 0

    f, prov = build_frame(pairs)
    print("=" * 64)
    print(f"XWOBA SHADOW ARM — {len(pairs)} paired slate(s), {len(f)} games")
    print("provenance (a rebuilt dump saw an extra day; both arms rebuilt "
          "together is still a fair pairing)")
    for d, name, kind, snap, start, wkind in prov:
        flag = "" if kind == "pregame" else "   <- shadow rows are post-hoc"
        print(f"  {d}  {kind:16s} snapshot {str(snap)[:19]}  first pitch "
              f"{str(start)[:19]}  primary {wkind}{flag}")

    dec = f.dropna(subset=["net_w", "net_x"])
    print(f"\nabstentions: wOBA {int(f.net_w.isna().sum())}, "
          f"xwOBA {int(f.net_x.isna().sum())}, both-decided {len(dec)}")
    print(f"corr(net_wOBA, net_xwOBA) = {_corr(dec.net_w, dec.net_x):+.4f}   "
          f"mean |net| wOBA {dec.net_w.abs().mean():.4f} xwOBA {dec.net_x.abs().mean():.4f}")
    flips = dec[(dec.net_w > 0) != (dec.net_x > 0)]
    print(f"lean flips: {len(flips)}/{len(dec)}"
          + (f"  (95% upper bound {3/len(dec):.1%} by the rule of three)"
             if len(flips) == 0 and len(dec) else ""))
    for _, r in flips.iterrows():
        print(f"    {r.date} {r.matchup}: wOBA {r.net_w:+.4f} -> xwOBA {r.net_x:+.4f}")

    graded = dec[dec.margin.notna()]
    print(f"\ngraded and both-decided: {len(graded)}")
    if len(graded) >= 3:
        for name, col in (("wOBA  (dump)", "net_w"), ("xwOBA (shadow)", "net_x")):
            w, l = _record(graded[col], graded.margin)
            pct = f"{w/(w+l):.3f}" if w + l else "  -  "
            print(f"  {name:16s} record {w}-{l} ({pct})  "
                  f"corr(net, run margin) {_corr(graded[col], graded.margin):+.3f}")
        d_obs = _corr(graded.net_x, graded.margin) - _corr(graded.net_w, graded.margin)
        b = _d_corr_bootstrap(graded, n_boot)
        se = float(b.std())
        print(f"  d_corr (xwOBA - wOBA) {d_obs:+.4f}  paired se {se:.4f}  "
              f"95% CI [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
        unpaired = 1 / np.sqrt(max(len(graded) - 3, 1))
        print(f"  for scale: se of ONE correlation at n={len(graded)} is ~{unpaired:.3f}; "
              f"pairing buys {unpaired / se:.1f}x")
        # Projection, stated as what it is: se scales as 1/sqrt(n), and this se
        # is itself estimated on n=len(graded). Read it as an order, not a date.
        n_target = int(np.ceil(len(graded) * (se / TARGET_SE) ** 2))
        n_power = int(np.ceil(len(graded) * (se / (TARGET_GAP / 2.8)) ** 2))
        per_slate = len(f) / len(pairs)
        print(f"  projection: se {TARGET_SE} at n~{n_target} games "
              f"(~{n_target/per_slate:.0f} slates); 80% power on a {TARGET_GAP} gap "
              f"at n~{n_power} (~{n_power/per_slate:.0f} slates)")
        print("  (both assume the arms keep correlating as they do now, and the "
              "se above is itself measured on this small a sample)")
    else:
        print("  too few graded games to score")

    if ledger_join:
        fl, _ = build_frame(pairs, use_ledger_net=True)
        j = fl.dropna(subset=["net_w", "net_x"])
        print("\ncontaminated join (ledger's PREGAME wOBA net vs the shadow dump) "
              "— run to size the bias, not to draw a conclusion:")
        print(f"  corr(ledger net, same-day dump net) = {_corr(f.net_led.dropna(), f.net_w.reindex(f.net_led.dropna().index)):+.4f}")
        print(f"  mean |ledger - dump| = {(f.net_led - f.net_w).abs().mean():.5f} "
              f"against median |net| {f.net_w.abs().median():.4f}")
        print(f"  lean flips ledger vs shadow: "
              f"{int(((j.net_w > 0) != (j.net_x > 0)).sum())}/{len(j)}")
    print("=" * 64)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--ledger-join", action="store_true")
    a = ap.parse_args(argv)
    return report(n_boot=a.boot, ledger_join=a.ledger_join)


if __name__ == "__main__":
    raise SystemExit(main())
