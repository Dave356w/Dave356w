"""Decide whether a scheduled workflow should run the full site build.

GitHub cron is static, so the workflow polls every 15 minutes and this script
gates expensive work against the live MLB slate. Push/manual events always run;
the daily early-morning schedule always runs to grade completed games.

The pregame window is 15-360 minutes before first pitch. Later builds inside
the window refresh still-pending rows, so the lock is still the LAST accepted
pregame snapshot (grade_leans.py keeps rejecting anything captured at/after
scheduled start) -- widening only adds earlier attempts, it never makes the
accepted snapshot worse.

THE WINDOW IS SIZED TO THE POLL GAP, NOT TO CRON JITTER, and that is a
correction. It read 15-90 on the theory that Actions delays scheduled runs by
"5-20+ minutes", so 75 minutes of width survived the jitter. The real failure
is not delay, it is DROP: on 2026-08-27 the workflow requested 56 polls
(`7,22,37,52` over 14 hours) and GitHub delivered 2, having delivered 10 the
day before and 18 the day before that. Measured gaps between consecutive
delivered polls ran 3h53m and 11h. A 75-minute window cannot survive a
four-hour gap, and on 2026-08-27 it did not: first pitch was 17:05Z, the
window was 15:35-16:50Z, nothing ran, and that slate's only capture came from
an unrelated push at 13:46Z -- 3h20m early, 7 games, projected lineups.

So the two knobs move together and neither is meaningful alone: the cron drops
to hourly (fewer requested runs are likelier to be honoured; a 56/day ask is
plausibly why they were dropped) and the window widens to 6 hours, which is
wider than any gap observed so far. test_window_outlasts_the_poll_interval
pins the relationship against build.yml rather than against these literals, so
changing one without the other fails.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROLLOVER_HOUR = 3
DAILY_GRADE_CRON = "17 4 * * *"
MIN_MINUTES_BEFORE = 15
# 6 hours. Not a guess at jitter -- see the module docstring: it must exceed
# the gap between DELIVERED polls, which has been measured in hours.
MAX_MINUTES_BEFORE = 360


def slate_date(now):
    now_et = now.astimezone(ET)
    day = now_et if now_et.hour >= ROLLOVER_HOUR else now_et - timedelta(days=1)
    return day.date().isoformat()


def upcoming_game_ids(games, now, min_before=MIN_MINUTES_BEFORE,
                      max_before=MAX_MINUTES_BEFORE):
    """Return gamePks whose scheduled starts fall inside the pregame window."""
    now_utc = now.astimezone(timezone.utc)
    due = []
    for game in games:
        if game.get("status", {}).get("abstractGameState") != "Preview":
            continue
        raw_start = game.get("gameDate")
        if not raw_start:
            continue
        try:
            start = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        except ValueError:
            continue
        minutes = (start.astimezone(timezone.utc) - now_utc).total_seconds() / 60
        if min_before <= minutes <= max_before:
            due.append(int(game["gamePk"]))
    return due


def fetch_games(day):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}"
    req = Request(url, headers={"User-Agent": "Dave356w-schedule-gate/1.0"})
    with urlopen(req, timeout=20) as response:
        payload = json.load(response)
    return [game for date in payload.get("dates", []) for game in date.get("games", [])]


def decision(event_name, event_schedule, now=None, games=None):
    now = now or datetime.now(timezone.utc)
    day = slate_date(now)
    if event_name != "schedule":
        return True, day, f"{event_name or 'manual'} event"
    if event_schedule == DAILY_GRADE_CRON:
        return True, day, "daily grading pass"
    try:
        games = fetch_games(day) if games is None else games
        due = upcoming_game_ids(games, now)
    except Exception as exc:  # upstream failure should not suppress a refresh
        return True, day, f"schedule lookup failed; fail-open ({type(exc).__name__})"
    if due:
        return True, day, "pregame window: " + ",".join(map(str, due))
    return False, day, (f"no game {MIN_MINUTES_BEFORE}-{MAX_MINUTES_BEFORE} "
                        "minutes from first pitch")


def emit_output(name, value):
    value = str(value).replace("\n", " ")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")
    print(f"{name}={value}")


def main():
    should_run, day, reason = decision(
        os.environ.get("EVENT_NAME", "workflow_dispatch"),
        os.environ.get("EVENT_SCHEDULE", ""),
    )
    emit_output("should_run", str(should_run).lower())
    emit_output("slate_date", day)
    emit_output("reason", reason)


if __name__ == "__main__":
    main()
