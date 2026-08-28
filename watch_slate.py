"""Wake at each game's pregame target and dispatch build.yml.

WHY THIS EXISTS, MEASURED. GitHub delivers a fraction of the scheduled runs
this repository asks for. Counting runs GitHub actually created for build.yml,
by ET day: 2026-08-25 requested 56 polls and got 21, 08-26 requested 56 and got
7, 08-27 requested ~15 (the cron went hourly at 18:20Z) and got 2. On 08-28 it
delivered 1 of the first 2 due, dropping the 04:17 ET grading pass outright,
and dropped this workflow's own 09:50 ET shift as well -- run 1 came from a
hand dispatch at 15:02Z.

The cost is on the record rather than hypothetical. Every pregame lock on the
2026-08-27 slate came from a push or a manual dispatch and not one from a
cron: COL@WSH (17:05Z) locked at 13:46:57Z off a merge, the four games
starting 23:05-23:15Z all locked at 18:59:35Z off another merge -- four hours
out, on projected lineups -- and ARI@SF locked at 00:32:48Z off a hand-run
re-run. Those rows cannot be re-derived later without lookahead.

ONE CORRECTION TO THE ARGUMENT THAT USED TO STAND HERE. This file previously
said that cutting the ask (56/day to hourly on 08-27) "made the delivered count
fall rather than rise", and concluded the smaller-ask theory was unsupported.
That does not follow: the ask fell about 4x and the delivered count fell about
4x, so the delivery RATE is not distinguishable across the change and nothing
here ever computed it. It is evidence for neither theory. The reason to stop
tuning the cron is simpler and does not need that claim -- a schedule this
repository does not control is the wrong place to put a deadline it cannot
miss.

WHAT THIS DOES. One delivered cron starts one shift. The shift reads the day's
slate from StatsAPI, computes a dispatch target per cluster of games, sleeps
until the next one, and dispatches build.yml there. It re-reads the slate on
every wake (at most RECHECK_SECONDS apart) because start times move.

TARGETS, NOT A POLL, AND WHY THERE ARE TWO PER CLUSTER. The previous version
polled a fixed 20 minutes and asked schedule_gate "is any game 15-360 minutes
out?", which on a full slate is true for most of the day -- so it dispatched
about fifteen builds to capture a handful of distinct lineup states, and held
`site-build` against every push while doing it. Targeting the games directly
costs two builds per cluster instead: TARGET_MINUTES fires at T-40 (a safe
capture that survives a slow build) and again at T-20 (the tight lock, after
lineups post). grade_leans accepts the LAST pregame snapshot, so the second
one is what the lock actually becomes, and both sit clear of
schedule_gate.MIN_MINUTES_BEFORE with room for the ~12 minute build observed
on 2026-08-27.

A cluster is one dispatch because a build renders the WHOLE slate, not one
game: two games starting three minutes apart do not need two builds.
COALESCE_MINUTES is what "close enough to share" means.

WHAT IT DOES NOT FIX, AND WHY build.yml KEEPS ITS OWN SCHEDULE. A shift is
itself started by a cron GitHub may drop -- and on this workflow's first day,
did. This is a second, independent path to a build rather than a replacement,
so no capture path that works today can regress. It is also the redundancy
behind the one place this design is thinner than the poll it replaces: a
cluster's target fires once, so a dispatch that succeeds into a build that
then FAILS is not retried by this loop. build.yml's own hourly cron is what
covers that, which is the whole reason it is left alone rather than thinned
out to pay for this.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import schedule_gate

# Minutes before first pitch to dispatch. Two per cluster: an early capture
# that survives a slow build, and a tight one that becomes the actual lock.
# Both must clear schedule_gate.MIN_MINUTES_BEFORE by more than a build takes
# -- test_targets_clear_the_gate_floor pins that against the gate itself
# rather than against a copy of the number.
TARGET_MINUTES = (40, 20)

# Games whose first pitches fall within this of each other share one dispatch.
COALESCE_MINUTES = 20

# Longest sleep between slate re-reads. Start times move (rain, doubleheaders,
# a game added late), so a target computed once is not a target forever.
RECHECK_SECONDS = 30 * 60

# After a failed dispatch, come back soon rather than at the next target.
RETRY_SECONDS = 2 * 60

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


def _start_utc(game):
    raw = game.get("gameDate")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def dispatch_targets(games, now, offsets=TARGET_MINUTES,
                     coalesce=COALESCE_MINUTES, min_before=None):
    """Return sorted [(target_utc, gamePks, offset_minutes), ...] for `games`.

    Games are clustered by first pitch -- one build renders the whole slate --
    and each cluster yields one target per entry in `offsets`, measured from
    the EARLIEST start in the cluster so no game in it is dispatched late.

    A game already inside min_before is dropped: schedule_gate would refuse it
    and the snapshot could not be locked pregame anyway. Targets in the past
    are kept, so a shift starting mid-window captures immediately instead of
    waiting for the next one.
    """
    min_before = schedule_gate.MIN_MINUTES_BEFORE if min_before is None else min_before
    now = now.astimezone(timezone.utc)
    upcoming = []
    for game in games:
        if game.get("status", {}).get("abstractGameState") != "Preview":
            continue
        start = _start_utc(game)
        if start is None:
            continue
        if (start - now).total_seconds() / 60 < min_before:
            continue
        upcoming.append((start, int(game["gamePk"])))
    upcoming.sort()

    clusters = []
    for start, pk in upcoming:
        if clusters and (start - clusters[-1][0]).total_seconds() / 60 <= coalesce:
            clusters[-1][1].append(pk)
        else:
            clusters.append((start, [pk]))

    targets = [
        (first - timedelta(minutes=offset), tuple(sorted(pks)), offset)
        for first, pks in clusters
        for offset in offsets
    ]
    targets.sort(key=lambda t: (t[0], t[2]))
    return targets


def watch(repo, token, deadline, clock=time.time, sleep=time.sleep,
          now_utc=None, log=print, fetch=None):
    """Sleep to each target until `deadline`, dispatching a build at each.

    Bounded by its own clock rather than by the step's timeout-minutes: a step
    timeout kills the shell and leaves the child running, which is how run
    33073467257 lost a slate. This one writes nothing under data/, so an
    orphan could not corrupt a commit -- but it could still dispatch builds
    after the job it belongs to has gone, which is the same class of surprise.
    """
    now_utc = now_utc or (lambda: datetime.now(timezone.utc))
    fetch = fetch or schedule_gate.fetch_games
    # Keyed on (gamePks, offset), not on the target time: a start that moves
    # must not re-fire a target this shift has already spent.
    fired = set()
    dispatched = 0

    while True:
        if deadline - clock() <= 0:
            log(f"shift complete; {dispatched} build(s) dispatched")
            return dispatched

        now = now_utc()
        stamp = now.strftime("%H:%M:%SZ")
        try:
            targets = dispatch_targets(fetch(schedule_gate.slate_date(now)), now)
        except Exception as exc:  # noqa: BLE001 - upstream must not stall the shift
            # Fail open the way schedule_gate does: a missed pregame row cannot
            # be re-derived, a redundant build can be thrown away. Backing off
            # to RECHECK_SECONDS caps an all-day outage at two builds an hour,
            # which is under what the fixed poll cost anyway.
            log(f"{stamp} slate lookup failed ({type(exc).__name__}); dispatching anyway")
            if _dispatch(repo, token, log, stamp):
                dispatched += 1
            sleep(min(RECHECK_SECONDS, max(0.0, deadline - clock())))
            continue

        wait = RECHECK_SECONDS
        due = [t for t in targets if t[0] <= now and (t[1], t[2]) not in fired]
        if due:
            covered = sorted({pk for _t, pks, _o in due for pk in pks})
            if _dispatch(repo, token, log, stamp, covered):
                dispatched += 1
                fired.update((pks, offset) for _t, pks, offset in due)
            else:
                wait = RETRY_SECONDS

        pending = [t[0] for t in targets if t[0] > now and (t[1], t[2]) not in fired]
        if pending:
            wait = min(wait, (min(pending) - now).total_seconds())
            if not due:
                log(f"{stamp} hold -- next target {min(pending).strftime('%H:%MZ')}")
        elif not due:
            log(f"{stamp} hold -- no further target on this slate")

        sleep(min(max(1.0, wait), max(0.0, deadline - clock())))


def _dispatch(repo, token, log, stamp, pks=None):
    """Dispatch once; never fatal. Returns whether it landed.

    A dispatch failure must not end the shift: the next wake is minutes away
    and build.yml still has its own cron, so throwing away the rest of the
    coverage over one API blip is the worse trade.
    """
    detail = "" if not pks else " for " + ",".join(map(str, pks))
    try:
        dispatch_build(repo, token)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        log(f"{stamp} dispatch failed ({type(exc).__name__}: {exc})")
        return False
    log(f"{stamp} dispatched build.yml{detail}")
    return True


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 2
    shift = float(os.environ.get("SHIFT_SECONDS", SHIFT_SECONDS))
    print(f"watching {repo} for {shift / 3600:.2f}h; targets at "
          f"{', '.join(f'T-{m}' for m in TARGET_MINUTES)} minutes")
    watch(repo, token, deadline=time.time() + shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
