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


# Which Savant rate a dump actually holds. Read off `model_metric`, never off
# the filename or the running build's constants: v11 swapped which metric is
# primary and which is shadow, so "the shadow dump" names a SIDE, not a
# statistic, and the six pre-v11 slates hold the opposite assignment from every
# slate after them. The filename is the fallback only for a dump written before
# the column existed.
_SUFFIX_METRIC = {"xw": "xwOBA", "woba": "wOBA", "split": "wOBA/xwOBA"}


def dump_metric(df, path):
    col = df.get("model_metric")
    if col is not None:
        vals = pd.Series(col).dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return None
    m = re.search(r"_(\w+)\.csv$", os.path.basename(path))
    return _SUFFIX_METRIC.get(m.group(1)) if m else None


def _pick(*paths):
    """First path that exists, else None. Callers pass live before rebuild."""
    return next((p for p in paths if os.path.exists(p)), None)


def _slate_dates():
    """Dates that have BOTH a shadow dump and a primary dump beside it.

    Globs every shadow suffix, not just `_xw`: the arm wrote xwOBA dumps before
    v11 and wOBA dumps after it, and a report that saw only one generation
    would silently shrink its own sample at the changeover.

    Since the rebuild rename, a slate can hold two dumps per arm: the pregame
    one under the live name and a post-first-pitch reconstruction under
    `rebuild_`. The pregame pair is what the arm exists to compare, so it wins
    where it exists -- but a rebuild-only slate is still READ rather than
    dropped, because dropping it would shrink the sample silently and the
    provenance block below already prints which kind each slate is. The arm
    landed mid-morning on 2026-08-09 and that slate has no pregame dump at all;
    a report that hid it would be hiding its own history.
    """
    out, seen = [], set()
    for suf in ("xw", "woba"):
        stem = f"_*_{suf}.csv"
        pats = (os.path.join(DATA_DIR, f"{sm.SHADOW_PREFIX}{stem}"),
                os.path.join(DATA_DIR,
                             f"{bs.REBUILD_PREFIX}_{sm.SHADOW_PREFIX}{stem}"))
        for p in sorted(q for pat in pats for q in glob.glob(pat)):
            m = re.search(r"_(\d{4}-\d{2}-\d{2})_" + suf + r"\.csv$",
                          os.path.basename(p))
            if not m or m.group(1) in seen:
                continue
            d = m.group(1)
            # Prefer the pregame shadow dump for this date even when the glob
            # reached the rebuild first, so the two arms are picked by the same
            # rule rather than by iteration order.
            shadow = _pick(os.path.join(DATA_DIR,
                                        f"{sm.SHADOW_PREFIX}_{d}_{suf}.csv"), p)
            # Live before rebuild ACROSS suffixes, not within each: a pregame
            # `_xw` dump beats a rebuilt `_woba` one, and interleaving the two
            # orders would silently prefer a reconstruction.
            primary = _pick(*[os.path.join(DATA_DIR, f"{pre}leans_{d}_{s}.csv")
                              for pre in ("", f"{bs.REBUILD_PREFIX}_")
                              for s in ("woba", "xw", "split")])
            if primary:
                seen.add(d)
                out.append((d, shadow, primary))
    return sorted(out)


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
        sdf, pdf = pd.read_csv(spath), pd.read_csv(ppath)
        smet, pmet = dump_metric(sdf, spath), dump_metric(pdf, ppath)
        # net_w and net_x are keyed to the METRIC, not to which side of the
        # arm produced them, so d_corr means "xwOBA minus wOBA" on every slate
        # regardless of which one the primary was that day. Without this the
        # v11 changeover would silently flip the sign of half the sample.
        by_metric = {smet: game_nets(sdf), pmet: game_nets(pdf)}
        if smet == pmet or "wOBA" not in by_metric or "xwOBA" not in by_metric:
            print(f"  {d}: SKIPPED -- dumps do not pair one wOBA against one "
                  f"xwOBA (primary {pmet!r}, shadow {smet!r})")
            continue
        ws, xs = by_metric["wOBA"], by_metric["xwOBA"]
        xkind, xsnap, xstart = _provenance(xs)
        prov.append((d, os.path.basename(spath), xkind, xsnap, xstart,
                     _provenance(ws)[0]))
        f = pd.DataFrame({"net_w": ws["net"], "net_x": xs["net"]})
        f["matchup"], f["date"] = xs["matchup"], d
        frames.append(f)
    if not frames:
        raise SystemExit("shadow_report: no slate paired one wOBA dump against "
                         "one xwOBA dump; nothing to compare")
    f = pd.concat(frames)
    g = led.reindex(f.index)
    f["net_led"] = pd.to_numeric(g["xw_net"], errors="coerce")
    f["led_metric"] = g.get("model_metric")
    f["status"] = g["status"]
    f["margin"] = (pd.to_numeric(g["full_home"], errors="coerce")
                   - pd.to_numeric(g["full_away"], errors="coerce"))
    if use_ledger_net:
        # The ledger's `xw_net` is a legacy KEY name, not a statement of which
        # statistic is in it -- pre-v11 rows hold wOBA, v11 rows hold xwOBA. So
        # substitute it into whichever column matches each row's own
        # model_metric. Overwriting net_w unconditionally (correct while the
        # ledger was wOBA) would compare a v11 xwOBA ledger net against the
        # wOBA arm and call the metric difference contamination.
        lm = f["led_metric"].astype(str)
        for metric, col in (("wOBA", "net_w"), ("xwOBA", "net_x")):
            m = (lm == metric) & f["net_led"].notna()
            f.loc[m, col] = f.loc[m, "net_led"]
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
        # Labelled by metric only. Which arm was primary changed at v11, so
        # naming a side here would be wrong for half the sample.
        for name, col in (("wOBA", "net_w"), ("xwOBA", "net_x")):
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
        print("\ncontaminated join (the ledger's PREGAME net vs the same-day "
              "dumps) — run to size the bias, not to draw a conclusion:")
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
