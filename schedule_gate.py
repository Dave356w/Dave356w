"""Decide whether a scheduled workflow should run the full site build.

GitHub cron is static, so the workflow polls every 15 minutes and this script
gates expensive work against the live MLB slate. Push/manual events always run;
the daily early-morning schedule always runs to grade completed games.
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
MAX_MINUTES_BEFORE = 45


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
    return False, day, "no game 15-45 minutes from first pitch"


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
