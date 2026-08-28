"""Poll the pregame gate inside one long-running job and dispatch build.yml.

WHY THIS EXISTS, MEASURED. GitHub delivers a fraction of the scheduled runs
this repository asks for, and the fraction has been falling. Counting runs
GitHub actually created for build.yml, by ET day: 2026-08-25 requested 56
polls and got 21, 08-26 requested 56 and got 7, 08-27 requested ~15 (the cron
went hourly at 18:20Z) and got 2. Between 2026-08-27T19:07Z and 2026-08-28
00:39Z it created none at all -- five consecutive hourly slots covering the
whole evening slate.

The cost is on the record rather than hypothetical. Every pregame lock on the
2026-08-27 slate came from a push or a manual dispatch and not one from a
cron: COL@WSH (17:05Z) locked at 13:46:57Z off a merge, the four games
starting 23:05-23:15Z all locked at 18:59:35Z off another merge -- four hours
out, on projected lineups -- and ARI@SF locked at 00:32:48Z off a hand-run
re-run. Those rows cannot be re-derived later without lookahead.

Cutting the ask was tried first: 2026-08-27 moved the cron from `7,22,37,52`
(56/day) to hourly (14/day) on the theory that a smaller ask is likelier to be
honoured. The delivered count fell rather than rose. That theory is not
supported by what happened next, so this stops tuning the cron.

WHAT THIS CHANGES. Nothing about how a build decides or what it writes. One
delivered cron starts one shift here; the shift calls schedule_gate itself
every POLL_SECONDS and dispatches build.yml when the gate fires. The day then
rests on ~3 cron deliveries instead of ~15, and a dropped delivery costs one
shift instead of one poll.

WHAT IT DOES NOT FIX, AND WHY build.yml KEEPS ITS OWN SCHEDULE. A shift is
itself started by a cron GitHub may drop. This is a second, independent path
to a build rather than a replacement, so no capture path that works today can
regress -- which is the whole reason build.yml's schedule is left alone rather
than thinned out to pay for this one.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import schedule_gate

# One knob, not two. Polling more often than we are willing to build buys
# nothing: the gate is only worth asking when a dispatch can follow, and the
# poll itself is a single StatsAPI call either way.
#
# 20 minutes is sized to the tail, which is what the lock actually depends on.
# grade_leans accepts the LAST pregame snapshot, so the number that matters is
# how close to first pitch the final build lands: the gate stops firing at
# MIN_MINUTES_BEFORE, so the last dispatch falls in [T-15-interval, T-15] and
# averages T-25 here. It is also what keeps this affordable next to a ~10
# minute build -- schedule_gate's window is 15-360 minutes, so on a full slate
# the gate says go for most of the day, and a 15-minute interval would run
# builds back to back for eight hours and hold `site-build` against every
# push. At 20 minutes that group is roughly half idle.
POLL_SECONDS = 20 * 60

# GitHub caps a job at 6 hours. 5h20m leaves headroom for checkout, setup and
# the runner's own overhead, and lets three shifts tile the slate day.
SHIFT_SECONDS = 5 * 3600 + 20 * 60

BUILD_WORKFLOW = "build.yml"
BUILD_REF = "main"

API = "https://api.github.com"


def _api(path, token, method="GET", body=None):
    request = Request(
        f"{API}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Dave356w-slate-watch/1.0",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def dispatch_build(repo, token):
    _api(
        f"/repos/{repo}/actions/workflows/{BUILD_WORKFLOW}/dispatches",
        token,
        method="POST",
        body={"ref": BUILD_REF},
    )


def gate(now):
    """Ask schedule_gate the question a pregame cron would have asked.

    Called in-process rather than through the workflow's env-var entry point
    so there is one decision function and no second copy of the window. The
    schedule name only has to differ from DAILY_GRADE_CRON to take the pregame
    path; passing "workflow_dispatch" here would take the always-run path and
    defeat the gate entirely.
    """
    should_run, _slate, reason = schedule_gate.decision("schedule", "watch", now)
    return should_run, reason


def watch(repo, token, deadline, clock=time.time, sleep=time.sleep,
          now_utc=None, log=print):
    """Poll until `deadline`, dispatching a build when the gate fires.

    Bounded by its own clock rather than by the step's timeout-minutes: a step
    timeout kills the shell and leaves the child running, which is how run
    33073467257 lost a slate. This one writes nothing under data/, so an
    orphan could not corrupt a commit -- but it could still dispatch builds
    after the job it belongs to has gone, which is the same class of surprise.
    """
    now_utc = now_utc or (lambda: datetime.now(timezone.utc))
    dispatched = 0
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            log(f"shift complete; {dispatched} build(s) dispatched")
            return dispatched
        stamp = now_utc().strftime("%H:%M:%SZ")
        try:
            should_run, reason = gate(now_utc())
        except Exception as exc:  # noqa: BLE001 - upstream must not stall the shift
            # schedule_gate already fails open on a StatsAPI error; this is the
            # backstop for anything it does not catch. Failing open matches it:
            # a missed pregame row is unrecoverable, a redundant build is not.
            should_run, reason = True, f"gate raised; fail-open ({type(exc).__name__})"
        if not should_run:
            log(f"{stamp} hold -- {reason}")
        else:
            try:
                # No "is a build already running?" check, deliberately. It was
                # written and removed: build.yml's own `concurrency: site-build`
                # already collapses duplicates -- GitHub cancels the PENDING run
                # when a newer one joins the group, so at most one build ever
                # waits -- and a hand-rolled check on top of it can deadlock,
                # because one run stuck in `queued` (run 33081633410 sat there
                # for ten hours) would suppress every dispatch for the rest of
                # the shift, silently. A redundant build costs a commit; a
                # suppressed shift costs the slate this exists to protect.
                dispatch_build(repo, token)
                dispatched += 1
                log(f"{stamp} dispatched build.yml -- {reason}")
            except (HTTPError, URLError, OSError, ValueError) as exc:
                # Never fatal. The next poll is 20 minutes away and build.yml
                # still has its own cron; killing the shift over one API blip
                # would throw away the coverage this exists to provide.
                log(f"{stamp} dispatch failed ({type(exc).__name__}: {exc})")
        sleep(min(POLL_SECONDS, max(0.0, deadline - clock())))


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 2
    shift = float(os.environ.get("SHIFT_SECONDS", SHIFT_SECONDS))
    print(f"watching {repo} for {shift / 3600:.2f}h, polling every "
          f"{POLL_SECONDS // 60}min")
    watch(repo, token, deadline=time.time() + shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
