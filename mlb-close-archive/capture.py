"""Close archive capture. Stateless; append-only; pick-agnostic.
Gate order (cheap -> expensive):
  1. MLB Stats API schedule (free): any game pregame AND inside its capture
     window?  No -> exit 0, zero Odds API credits spent.
  2. One Odds API call (h2h+totals, us region = 2 credits) -> append one row
     per event to close_log/YYYY-MM.csv with the MLB status joined in.
"""
import csv, os, sys, datetime as dt
import requests

ODDS_KEY   = os.environ["THE_ODDS_API_KEY"]
MLB_API    = "https://statsapi.mlb.com/api/v1"
ODDS_API   = ("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
              "?regions=us&markets=h2h,totals&oddsFormat=american"
              f"&apiKey={ODDS_KEY}")

PREGAME_KEYS     = ("scheduled", "pre-game", "pregame", "warmup", "delayed")
CAPTURE_LEAD_MIN = 90        # start capturing T-90
CAPTURE_LAG_MIN  = 240       # keep capturing while still-pregame up to T+4h (delays)

CLOSE_FIELDS  = ["ts_utc", "gamePk", "home", "away", "commence", "mlb_status",
                 "home_price", "away_price", "fair_home", "book",
                 "total_line", "over_price", "under_price", "ou_books_n",
                 "credits_remaining"]
STATUS_FIELDS = ["ts_utc", "gamePk", "home", "away", "commence", "mlb_status"]


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def mlb_schedule(day):
    js = requests.get(f"{MLB_API}/schedule?sportId=1&date={day}", timeout=20).json()
    out = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            out.append({
                "gamePk": g["gamePk"],
                "home": g["teams"]["home"]["team"]["name"],
                "away": g["teams"]["away"]["team"]["name"],
                "commence": g["gameDate"],                       # ISO Z
                "status": g.get("status", {}).get("detailedState", "?"),
            })
    return out


def is_pregame(status):
    s = str(status).lower()
    return any(k in s for k in PREGAME_KEYS)


def in_window(g, now):
    c = dt.datetime.fromisoformat(g["commence"].replace("Z", "+00:00"))
    lead = c - dt.timedelta(minutes=CAPTURE_LEAD_MIN)
    lag  = c + dt.timedelta(minutes=CAPTURE_LAG_MIN)
    return lead <= now <= lag


def devig_home(hp, ap):
    def imp(p):
        p = float(p)
        return 100.0 / (p + 100.0) if p > 0 else -p / (-p + 100.0)
    h, a = imp(hp), imp(ap)
    return h / (h + a)


def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n//2 - 1] + xs[n//2])


def summarize_event(ev):
    """Per-book devig -> median fair_home; modal totals line with median prices.
    Mirrors notebook policy: BOOKMAKER_POLICY='median'."""
    fairs, hps, aps = [], [], []
    by_line = {}
    for bm in ev.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m["key"] == "h2h":
                hp = ap = None
                for o in m["outcomes"]:
                    if o["name"] == ev["home_team"]: hp = o["price"]
                    elif o["name"] == ev["away_team"]: ap = o["price"]
                if hp is not None and ap is not None:
                    fairs.append(devig_home(hp, ap)); hps.append(hp); aps.append(ap)
            elif m["key"] == "totals":
                op = up = pt = None
                for o in m["outcomes"]:
                    if o["name"] == "Over":  op, pt = o["price"], o.get("point")
                    elif o["name"] == "Under": up = o["price"]
                if pt is not None and op is not None and up is not None:
                    by_line.setdefault(float(pt), []).append((op, up))
    out = {"home_price": "", "away_price": "", "fair_home": "", "book": "",
           "total_line": "", "over_price": "", "under_price": "", "ou_books_n": 0}
    if fairs:
        out.update(home_price=round(median(hps)), away_price=round(median(aps)),
                   fair_home=round(median(fairs), 4), book=f"median_{len(fairs)}")
    if by_line:
        line = max(by_line, key=lambda k: len(by_line[k]))      # modal line
        ops, ups = zip(*by_line[line])
        out.update(total_line=line, over_price=round(median(list(ops))),
                   under_price=round(median(list(ups))), ou_books_n=len(by_line[line]))
    return out


def append(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    now = now_utc()
    days = [now.date().isoformat()]
    if now.hour < 8:                                  # 0-7Z = previous US date
        days.append((now.date() - dt.timedelta(days=1)).isoformat())

    sched = [g for d in days for g in mlb_schedule(d)]
    month = now.strftime("%Y-%m")
    ts = now.isoformat(timespec="seconds")

    # always log status (free, tiny) -- this is the join's status feed
    append(f"status_log/{month}.csv", STATUS_FIELDS,
           [{"ts_utc": ts, **{k: g[k] for k in ("gamePk", "home", "away", "commence")},
             "mlb_status": g["status"]} for g in sched])

    # gate: spend Odds API credits only when a capture window is open
    active = [g for g in sched if is_pregame(g["status"]) and in_window(g, now)]
    if not active:
        print("no open capture windows; 0 credits spent"); return

    r = requests.get(ODDS_API, timeout=30)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    events = r.json()

    by_names = {(e["home_team"], e["away_team"]): e for e in events}
    rows = []
    for g in sched:                                   # archive ALL pregame games
        if not is_pregame(g["status"]):
            continue
        ev = by_names.get((g["home"], g["away"]))
        if ev is None:
            continue
        rows.append({"ts_utc": ts, "gamePk": g["gamePk"], "home": g["home"],
                     "away": g["away"], "commence": g["commence"],
                     "mlb_status": g["status"], **summarize_event(ev),
                     "credits_remaining": remaining})
    append(f"close_log/{month}.csv", CLOSE_FIELDS, rows)
    print(f"captured {len(rows)} event row(s); credits remaining {remaining}")


if __name__ == "__main__":
    main()
