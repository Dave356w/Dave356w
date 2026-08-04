"""Why is ESPN returning 403, and does any header set get us back in?

Run this from a machine whose egress matches the one that broke -- for this
repo that means a GitHub Actions runner, via `.github/workflows/espn-probe.yml`.
Running it from a laptop measures the laptop's IP reputation, not CI's, and
those answer different questions.

Background
----------
On 2026-08-04 the daily build logged, in the same job:

    market backfill: FAILED (HTTPError: HTTP Error 403: Forbidden)
    pregame odds: scoreboard fetch failed (403 ... /mlb/scoreboard?dates=20260804)

The 10:59Z run of the same commit had logged `pregame odds attached: 15/15
games`, so the break is upstream and sits between 11:00Z and 18:56Z. Neither
fetch path had changed since 2026-08-01 (f7150d6).

The one diagnostic already in hand: `build_site` sends a full Chrome
User-Agent and `market_backfill` sends the bare string `Mozilla/5.0`, and
*both* got 403 in the same job, while StatsAPI and Savant kept working in that
same job. That is evidence against a naive User-Agent blocklist and against a
general egress failure, and for something ESPN-side keyed on the caller --
most likely edge bot-management on the runner's address range.

This probe is what turns that inference into a measurement. It answers:

1. **Does any header set get a 200?** If a full browser-like header set works
   and the current ones do not, the fix is headers and this prints which.
2. **Is it every ESPN host or one of them?** `site.api` (scoreboard) and
   `sports.core.api` (odds) are separate edges and can be blocked separately.
3. **Is it us or the runner?** StatsAPI is pulled as a control on the same
   egress. If the control also fails, the problem is not ESPN.
4. **What does the block actually say?** Akamai denials carry a `Reference
   #...` in the body and a `Server:` banner. That string is what an ESPN
   support conversation needs, and it distinguishes a WAF denial from a
   rate-limit from a genuine 404-behind-403.

Nothing here writes to `data/`, touches a lean, or renders a page. It prints a
report and optionally writes it to a path you name.

    python espn_403_probe.py
    python espn_403_probe.py --out espn_403_probe_report.txt
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date

# The two ESPN endpoints the site actually depends on, plus a no-argument path
# on the core host. The core `/odds` URL needs a live event id, which we can
# only learn from the scoreboard -- and the scoreboard is one of the things
# that may be blocked. The league root reaches the same host and edge without
# needing an id, so host reachability stays measurable even when discovery
# fails.
SCOREBOARD = ("site.api scoreboard",
              "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
              "/scoreboard?dates={ds}")
CORE_ROOT = ("sports.core.api league root",
             "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb")
CORE_ODDS = ("sports.core.api odds",
             "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
             "/events/{eid}/competitions/{eid}/odds")

# Control on the same egress: if this fails too, the finding is about the
# runner's network, not about ESPN.
CONTROL = ("statsapi control",
           "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}")

CHROME_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")

# Ordered cheapest-to-most-browser-like. The first two are byte-for-byte what
# production sends today, so the report always contains the baseline it is
# being compared against rather than a paraphrase of it.
VARIANTS = [
    ("prod: market_backfill (bare Mozilla/5.0)", {
        "User-Agent": "Mozilla/5.0",
    }),
    ("prod: build_site (Chrome UA + Accept)", {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json,text/csv,*/*",
    }),
    ("python-urllib default (no UA override)", {}),
    ("Chrome UA + Accept + Lang", {
        "User-Agent": CHROME_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }),
    ("Chrome UA + Lang + Referer espn.com", {
        "User-Agent": CHROME_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/",
        "Origin": "https://www.espn.com",
    }),
    ("full browser set (+ Sec-Fetch/Sec-CH-UA)", {
        "User-Agent": CHROME_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.espn.com/",
        "Origin": "https://www.espn.com",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Sec-Ch-Ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", '
                     '"Chromium";v="127"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    }),
]

INTERESTING_RESP_HEADERS = ("Server", "Content-Type", "X-Cache", "X-Served-By",
                            "Via", "Retry-After", "X-Reference-Error",
                            "Akamai-GRN", "Set-Cookie")


def _attempt(url, headers, timeout=20):
    """One request. Returns a dict; never raises for an HTTP status."""
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(600)
            return dict(status=r.status, ms=int((time.time() - t0) * 1000),
                        headers=dict(r.headers), body=body[:600], err=None)
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read(600)
        except Exception:  # noqa: BLE001  -- body is a nicety, not the finding
            pass
        return dict(status=e.code, ms=int((time.time() - t0) * 1000),
                    headers=dict(e.headers or {}), body=body[:600], err=None)
    except (urllib.error.URLError, ssl.SSLError, OSError) as e:
        return dict(status=None, ms=int((time.time() - t0) * 1000),
                    headers={}, body=b"", err=f"{type(e).__name__}: {e}")


def _fmt(res):
    if res["err"]:
        return f"ERR  {res['err'][:90]}"
    return f"{res['status']}  {res['ms']:>5}ms"


def _detail(res, indent="      "):
    """The lines that make a 403 actionable rather than just red."""
    out = []
    for k in INTERESTING_RESP_HEADERS:
        for hk, hv in res["headers"].items():
            if hk.lower() == k.lower():
                out.append(f"{indent}{hk}: {str(hv)[:140]}")
    body = res["body"]
    if body:
        try:
            txt = body.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            txt = repr(body)
        txt = " ".join(txt.split())
        if txt:
            out.append(f"{indent}body[:220]: {txt[:220]}")
    return out


def _public_ip():
    """Best-effort: which address ESPN is actually judging."""
    for url in ("https://checkip.amazonaws.com",
                "https://api.ipify.org"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read(64).decode().strip()
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


def _discover_event_id(ds, report):
    """Pull one event id from whichever variant can reach the scoreboard."""
    for name, headers in VARIANTS:
        res = _attempt(SCOREBOARD[1].format(ds=ds), headers)
        if res["status"] != 200:
            continue
        # _attempt caps the body at 600 bytes for the report, which is far too
        # small to parse -- refetch in full, but only now that this variant is
        # known to be allowed through.
        try:
            req = urllib.request.Request(SCOREBOARD[1].format(ds=ds),
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                js = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            report.append(f"  event-id discovery: refetch failed ({e!r})")
            return None
        evs = js.get("events") or []
        if evs:
            report.append(f"  event-id discovery: via {name!r} -> "
                          f"{len(evs)} events, using {evs[0]['id']}")
            return evs[0]["id"]
    report.append("  event-id discovery: no variant reached the scoreboard; "
                  "odds endpoint tested by host root only")
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="slate date to request (YYYY-MM-DD, default today)")
    ap.add_argument("--event-id", default=None,
                    help="skip discovery and probe this ESPN event id's odds")
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()

    ds = args.date.replace("-", "")
    rep = []
    rep.append("=" * 68)
    rep.append("ESPN 403 PROBE")
    rep.append(f"  slate date : {args.date}")
    rep.append(f"  egress IP  : {_public_ip()}")
    rep.append(f"  python     : {sys.version.split()[0]}")
    rep.append("=" * 68)

    rep.append("")
    rep.append("CONTROL — same egress, non-ESPN host")
    res = _attempt(CONTROL[1].format(d=args.date), {"User-Agent": "Mozilla/5.0"})
    rep.append(f"  {CONTROL[0]:<34} {_fmt(res)}")
    control_ok = res["status"] == 200
    if not control_ok:
        rep.extend(_detail(res))
        rep.append("  !! control failed: read every ESPN line below as "
                   "inconclusive, the egress itself is unhealthy")

    eid = args.event_id
    if eid is None:
        rep.append("")
        rep.append("EVENT-ID DISCOVERY")
        eid = _discover_event_id(ds, rep)

    targets = [(SCOREBOARD[0], SCOREBOARD[1].format(ds=ds)),
               (CORE_ROOT[0], CORE_ROOT[1])]
    if eid:
        targets.append((CORE_ODDS[0], CORE_ODDS[1].format(eid=eid)))

    any_200 = {}
    for tname, url in targets:
        rep.append("")
        rep.append(f"{tname}")
        rep.append(f"  {url}")
        for vname, headers in VARIANTS:
            res = _attempt(url, headers)
            rep.append(f"    {vname:<42} {_fmt(res)}")
            if res["status"] == 200:
                any_200.setdefault(tname, []).append(vname)
            else:
                rep.extend(_detail(res))
            time.sleep(0.4)          # do not add a rate-limit to a 403 probe

    rep.append("")
    rep.append("=" * 68)
    rep.append("READING")
    if not control_ok:
        rep.append("  Control failed. Fix egress before concluding anything "
                   "about ESPN.")
    elif not any_200:
        rep.append("  No variant reached any ESPN endpoint while the control "
                   "passed.")
        rep.append("  => Not a User-Agent problem. Six header sets, including "
                   "a full browser")
        rep.append("     set, all denied from this address. Consistent with an "
                   "edge/IP block on")
        rep.append("     the runner's range. Header tuning will not fix this; "
                   "the options are a")
        rep.append("     different egress, a different odds source, or ESPN "
                   "allowlisting.")
        rep.append("     Quote the Reference # above to ESPN.")
    else:
        rep.append("  At least one header set was allowed through:")
        for tname, names in any_200.items():
            rep.append(f"    {tname}: {', '.join(names)}")
        rep.append("  => Fix is headers. Apply the cheapest passing variant to "
                   "the shared")
        rep.append("     fetch layer in build_site.py and market_backfill.py, "
                   "and re-run this")
        rep.append("     probe to confirm before trusting a green build.")
    rep.append("=" * 68)

    text = "\n".join(rep)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
