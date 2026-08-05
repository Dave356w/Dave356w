"""Does the box score actually carry a starter's ALLOWED line? Measure it.

Run this where StatsAPI is reachable -- for this repo that means a GitHub
runner, via `.github/workflows/phase-actuals-probe.yml`.

Why this exists
---------------
`actuals_backfill` learned to store the starter's allowed components so SP, BP
and lineup can be scored apart from each other. That rests on an assumption
this repo has NOT verified: that a box score's per-pitcher `stats.pitching`
object carries `doubles`, `triples` and `sacFlies` alongside the hits and walks
it is already read for. If it does not, `parse_boxscore` abandons every starter
line (all-or-nothing, deliberately, because rolling extra-base hits into
singles would bias every allowed rate downward) and the whole phase split is
inert -- silently, since the report simply omits a component it cannot pair.

Guessing at field availability from a sandbox that cannot reach the API is
exactly how the `byMonth` shape in reliever_shrink_probe.py got shipped broken.
So this measures it before anything depends on it.

READ-ONLY. It never writes data/, and the workflow grants `contents: read` to
make that structural. The ledger is the Actions bot's to commit -- this runs
`attach_actuals` against an in-memory copy purely to see what comes back, and
throws the result away. Populating data/ for real happens on the bot's next
build once the code is on main; this answers "will it work, and what will it
say" without waiting for that.

What it prints
--------------
  coverage   how many settled rows got a starter allowed line, and how many
             did not -- the number the assumption above lives or dies on.
  fields     which pitching keys were present on a sampled box score, so a
             partial schema names the missing field instead of just failing.
  components the per-component error report the ledger would show, per
             prediction family. Read slopes, not intercepts: the lineup's
             predicted value is a neutral composite scored against real
             pitching, and the weights here may differ from Savant's.

Usage:
  python phase_actuals_probe.py --out phase_actuals_probe_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import actuals_backfill as ab

LEDGER = "data/mlb_lean_ledger.csv"
FAMILIES = (
    ("v9/v10", ["xw+plat_consol_v9", "xw+plat_consol_v10"]),
    ("v7", ["xw+plat_consol_v7"]),
    ("v2/v3", ["xw+plat_consol_v2", "xw+plat_consol_v3"]),
    ("wOBA v1", ["woba+plat_consol_v1"]),
    ("wOBA v2", ["woba+plat_consol_v2"]),
)


def sample_pitching_keys(df, n=3):
    """Keys actually present on a starter's pitching object, from live games.

    Printed rather than asserted: if a field is missing, the fix depends on
    which one, and a boolean 'it failed' sends the next person guessing again.
    """
    out = []
    pks = [int(v) for v in pd.to_numeric(df.get("gamePk"), errors="coerce")
           .dropna().tolist()[:n]]
    for pk in pks:
        try:
            box = ab._get_json(ab.BOX_URL.format(gamePk=pk))
        except Exception as e:  # noqa: BLE001
            out.append((pk, f"fetch failed: {e!r}"))
            continue
        found = None
        for side in ("away", "home"):
            t = ((box.get("teams") or {}).get(side)) or {}
            for p in (t.get("players") or {}).values():
                ps = ((p.get("stats") or {}).get("pitching")) or {}
                if ab._f(ps.get("gamesStarted")):
                    found = sorted(ps.keys())
                    break
            if found:
                break
        out.append((pk, found or "no starter found"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--budget", type=float, default=900.0,
                    help="seconds of fetching (module default is tuned for a build)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only attempt N rows (0 = all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    led = pd.read_csv(a.ledger)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("Phase actuals probe -- can the starter's allowed line be read at all?")
    say("=" * 74)

    # Which keys the API really returns, before interpreting any coverage number.
    say("")
    say("pitching keys present on a starter's box-score line:")
    for pk, keys in sample_pitching_keys(led):
        if isinstance(keys, str):
            say(f"  game {pk}: {keys}")
        else:
            need = set(ab._BOX_PIT_KEY.values())
            missing = sorted(need - set(keys))
            say(f"  game {pk}: {len(keys)} keys; "
                + ("ALL REQUIRED PRESENT" if not missing
                   else f"MISSING {missing}"))
            say(f"    required: {sorted(need)}")
            say(f"    returned: {json.dumps(keys)[:400]}")
    say("")

    todo = ab._todo_index(led)
    say(f"settled rows needing backfill at schema {ab.ACT_SCHEMA}: {len(todo)}")
    work = led if not a.limit else led.loc[list(todo[:a.limit]) or [led.index[0]]]
    if a.limit:
        say(f"  (limited to {a.limit} rows for this run)")

    ab.BUDGET_S = a.budget
    out = ab.attach_actuals(work.copy(), verbose=True)

    # Coverage is the finding. A starter line present on ~every settled row
    # means the phase split works; near-zero means the assumption is wrong and
    # nothing downstream of it is real.
    say("")
    n_settled = int(ab._settled_mask(out).sum())
    have_sp = 0
    for side in ("away", "home"):
        have_sp += int(pd.to_numeric(out.get(f"act_sp_h_{side}"),
                                     errors="coerce").notna().sum())
    say(f"coverage: {have_sp} starter allowed lines across {n_settled} settled "
        f"rows ({2 * n_settled} possible sides)")
    if n_settled and have_sp == 0:
        say("  !! ZERO starter lines. The pitching object does not carry the")
        say("  !! required fields -- see the key dump above for which. The")
        say("  !! phase split is inert until parse_boxscore is changed.")

    p = ab.paired_components(out)
    say(f"paired component observations: "
        f"{p.component.value_counts().to_dict() if not p.empty else {}}")
    say("")

    say("component error by prediction family (never pooled)")
    say("  read slopes, not intercepts")
    for label, tags in FAMILIES:
        fam = out[out["model_tag"].astype(str).isin(tags)] \
            if "model_tag" in out.columns else out.iloc[0:0]
        lines = ab.components_summary(fam, tags=tags)
        if not lines:
            continue
        say(f"  -- {label} (n_rows={len(fam)}) --")
        for ln in lines[1:]:          # drop the repeated header
            say(ln)
    say("")
    say("Nothing was written. data/ is the Actions bot's to commit; this ran")
    say("against an in-memory copy so the numbers arrive before the merge.")

    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
