#!/usr/bin/env python3
"""Read-only Kalshi MLB paper fills; NEVER submits or authenticates an order.

Consumes the existing immutable pregame lean dump of the SHIPPED model (the
MODEL_TAG in build_site.py, currently v14; rows under any other tag are skipped
and logged, never filled). Captures live executable
YES asks from Kalshi's public order book, simulates immediate taker fills at
visible depth, and settles only when Kalshi AND the existing MLB ledger agree.
No model calibration, lean selection, or original ledger field is changed.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import requests

API = "https://external-api.kalshi.com/trade-api/v2"
MLB = "https://statsapi.mlb.com/api/v1"
ET = ZoneInfo("America/New_York")
D = Decimal
CENT = D("0.01")
# MLB ledger abbreviation -> accepted Kalshi event-ticker abbreviations.
# Kalshi's code is verified live only for AZ; where Kalshi may use a different
# common code (White Sox CHW, etc.) both are accepted. That is still an exact
# join -- ET date, start time, BOTH teams in order and the selected side must
# all match, and more than one matching market is skipped as ambiguous.
# Unknown codes fail closed; never fuzzy-match or join on the selected team alone.
TEAMS = {code: (code,) for code in (
    "ATL BAL BOS CHC CIN CLE COL DET HOU LAA LAD MIA MIL MIN NYM NYY "
    "PHI PIT SEA STL TEX TOR").split()}
TEAMS.update({"ARI": ("AZ", "ARI"), "ATH": ("ATH", "OAK"), "CWS": ("CWS", "CHW"),
              "KC": ("KC", "KCR"), "SD": ("SD", "SDP"), "SF": ("SF", "SFG"),
              "TB": ("TB", "TBR"), "WSH": ("WSH", "WAS")})
ABBR = {"Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
        "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
        "New York Mets": "NYM", "New York Yankees": "NYY",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
        "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"}
FIELDS = ["observed_utc", "game_pk", "game_date", "start_utc", "model_snapshot_utc",
          "model_tag", "away", "home", "lean", "xw_net", "sportsbook_ml",
          "sportsbook_be", "kalshi_ticker", "event_ticker", "status", "reason",
          "market_status", "book_best_ask", "quantity", "avg_price", "estimated_fee",
          "fee_multiplier", "kalshi_be", "savings_pp", "pnl_dollars", "settled_utc"]
FIELDS += ["estimated_fee_1x", "kalshi_be_1x", "savings_pp_1x"]
EVENT = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})(\d{4})?([A-Z]+?)(G\d+)?$")


BUILD_SITE = Path(__file__).resolve().parent / "build_site.py"
_TAG_LINE = re.compile(r'^MODEL_TAG\s*=\s*os\.environ\.get\(\s*"MODEL_TAG"\s*,\s*"([^"]+)"', re.M)


def current_model_tag(source=BUILD_SITE):
    """The tag the production build writes, read without importing build_site.

    The paper job installs only `requests`, so build_site's heavy imports are
    unavailable; its MODEL_TAG line is parsed instead (the same env override
    applies). tests/test_paper_kalshi.py pins this to build_site.MODEL_TAG.
    """
    if os.environ.get("MODEL_TAG"):
        return os.environ["MODEL_TAG"]
    match = _TAG_LINE.search(Path(source).read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"MODEL_TAG not found in {source}; refusing to guess the shipped model")
    return match.group(1)


def utc(value):
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (ValueError, TypeError):
        return None


def dec(value):
    try:
        return D(str(value)) if value not in (None, "", "nan") else None
    except (ValueError, TypeError, ArithmeticError):
        return None


def read_csv(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)
    tmp.replace(path)


def api(session, base, path, params=None):
    response = session.get(base + path, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def get_markets(session):
    markets, cursor = [], None
    for _ in range(20):
        params = {"series_ticker": "KXMLBGAME", "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = api(session, API, "/markets", params)
        markets.extend(data.get("markets", []))
        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            return markets
        cursor = next_cursor
    raise RuntimeError("Kalshi markets pagination exceeded 20 pages; refusing partial list")


def games_in_slate(rows):
    out = {}
    for row in rows:
        try:
            pk = str(int(row["game_pk"]))
        except (KeyError, ValueError):
            continue
        out.setdefault(pk, {})[row.get("side")] = row
    games = []
    for pk, sides in out.items():
        if "away" not in sides or "home" not in sides:
            continue
        away, home = sides["away"], sides["home"]
        # A pitcher row's opp_team is the TEAM BATTING AGAINST HIM.
        away_code = ABBR.get(home.get("opp_team"))
        home_code = ABBR.get(away.get("opp_team"))
        away_edge, home_edge = dec(home.get("edge_xwOBA")), dec(away.get("edge_xwOBA"))
        start = utc(away.get("scheduled_start_utc") or away.get("game_datetime_utc"))
        snapshot = utc(away.get("snapshot_utc"))
        lean = (home_code if home_edge is not None and away_edge is not None
                and home_edge > away_edge else away_code if home_edge is not None
                and away_edge is not None and away_edge > home_edge else None)
        game = {"game_pk": pk, "game_date": away.get("game_date", ""),
                "away": away_code, "home": home_code, "start": start,
                "snapshot": snapshot, "tag": away.get("model_tag", ""),
                "metric": away.get("model_metric", ""), "lean": lean,
                "xw_net": (home_edge - away_edge if home_edge is not None
                           and away_edge is not None else None),
                "book_utc": utc(away.get("pregame_market_utc")),
                "book_ml": dec(away.get("pregame_home_ml" if lean == home_code
                                else "pregame_away_ml"))}
        games.append(game)
    return games


def market_for(game, markets, all_games):
    start = game["start"]
    if not start or not game["away"] or not game["home"] or not game["lean"]:
        return None, "missing_fixture_or_lean"
    # Game numbers cannot be assigned reliably from the current lean dump.
    # Both games in a same-day doubleheader must be skipped, not cross-filled.
    duplicates = [g for g in all_games if g["game_date"] == game["game_date"]
                  and g["away"] == game["away"] and g["home"] == game["home"]]
    if len(duplicates) != 1:
        return None, "doubleheader_ambiguous"
    date = start.astimezone(ET).strftime("%y%b%d").upper()
    pairs = {a + h for a in TEAMS[game["away"]] for h in TEAMS[game["home"]]}
    candidate = []
    for m in markets:
        ticker = m.get("ticker", "")
        event = m.get("event_ticker", "")
        match = EVENT.fullmatch(event)
        if not match or match.group(1) != date or match.group(3) not in pairs:
            continue
        if match.group(4):  # cannot associate doubleheader ordinal without metadata
            continue
        if ticker not in {event + "-" + code for code in TEAMS[game["lean"]]}:
            continue
        if match.group(2):
            hhmm = match.group(2)
            et = start.astimezone(ET)
            try:
                event_start = et.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0)
            except ValueError:
                continue
            if abs((event_start - et).total_seconds()) > 20 * 60:
                continue
        candidate.append(m)
    if len(candidate) != 1:
        return None, "market_not_found" if not candidate else "market_ambiguous"
    return candidate[0], "matched_verified_event_date_pair_and_side"


def levels(book):
    # Kalshi publishes bids only: a NO bid at x is a YES ask at (1-x).
    raw = book.get("orderbook_fp", {}).get("no_dollars")
    if raw is None:
        old = book.get("orderbook", {})
        raw = old.get("no_dollars")
        if raw is None and old.get("no"):
            raw = [[str(D(p) / 100), str(q)] for p, q in old["no"]]
    result = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        price, volume = dec(item[0]), dec(item[1])
        if price is None or volume is None or not (0 < price < 1):
            continue
        count = int(volume)  # whole-contract paper fills only
        if count > 0:
            result.append((D(1) - price, count))
    return sorted(result)


def taker_fill(book, qty, multiplier):
    asks = levels(book)
    if sum(n for _, n in asks) < qty:
        return None
    remain, cost, fee, fee_1x = qty, D(0), D(0), D(0)
    for price, depth in asks:
        n = min(depth, remain)
        cost += n * price
        # Explicit cents ROUND_UP per price-level simulated execution.
        fee += (D("0.07") * multiplier * n * price * (1-price)).quantize(
            CENT, rounding=ROUND_CEILING)
        # Unconfirmed how the series fee_multiplier scales the taker curve, so
        # the standard (1x) 7% fee is carried alongside as a sensitivity check.
        fee_1x += (D("0.07") * n * price * (1-price)).quantize(
            CENT, rounding=ROUND_CEILING)
        remain -= n
        if not remain:
            break
    return {"book_best_ask": asks[0][0], "quantity": qty,
            "avg_price": cost / qty, "estimated_fee": fee,
            "kalshi_be": (cost + fee) / qty,
            "estimated_fee_1x": fee_1x, "kalshi_be_1x": (cost + fee_1x) / qty}


def ml_break_even(ml):
    if ml is None or ml == 0:
        return None
    return -ml / (D(100)-ml) if ml < 0 else D(100)/(D(100)+ml)


def assess(game, markets, schedule, session, multiplier, now, existing, qty, min_savings,
           model_tag):
    row = {k: "" for k in FIELDS}
    row.update(observed_utc=now.isoformat(), game_pk=game["game_pk"],
               game_date=game["game_date"],
               start_utc=game["start"].isoformat() if game["start"] else "",
               model_snapshot_utc=game["snapshot"].isoformat() if game["snapshot"] else "",
               model_tag=game["tag"], away=game["away"] or "", home=game["home"] or "",
               lean=game["lean"] or "", xw_net=str(game["xw_net"] or ""),
               sportsbook_ml=str(game["book_ml"] or ""),
               fee_multiplier=str(multiplier))
    def skip(reason):
        row.update(status="skipped", reason=reason)
        return row
    start, snap = game["start"], game["snapshot"]
    if game["tag"] != model_tag:
        return skip("model_tag_not_current")
    if not start or not snap or game["metric"] != "xwOBA" or not game["lean"]:
        return skip("invalid_pregame_snapshot")
    if not (snap < start and snap <= now and now-snap <= timedelta(minutes=120)):
        return skip("stale_or_postgame_model")
    if not (start-timedelta(minutes=360) <= now < start-timedelta(minutes=2)):
        return skip("outside_pregame_cutoff")
    state = schedule.get(game["game_pk"])
    if not state or state.get("status", {}).get("abstractGameState") != "Preview":
        return skip("mlb_status_not_confirmed_pregame")
    api_start = utc(state.get("gameDate"))
    if not api_start or abs((api_start-start).total_seconds()) > 15*60:
        return skip("mlb_start_time_changed")
    if not game["book_utc"] or game["book_utc"] > snap or game["book_utc"] >= start:
        return skip("missing_or_late_sportsbook_snapshot")
    if now-game["book_utc"] > timedelta(minutes=180):
        return skip("stale_sportsbook_price")
    book_be = ml_break_even(game["book_ml"])
    if book_be is None:
        return skip("missing_sportsbook_price")
    row["sportsbook_be"] = str(book_be)
    if game["game_pk"] in existing["positions"]:
        return skip("already_paper_filled")
    m, match_reason = market_for(game, markets, existing["fixtures"])
    if not m:
        return skip(match_reason)
    row.update(kalshi_ticker=m["ticker"], event_ticker=m["event_ticker"],
               market_status=m.get("status", ""))
    if m.get("status") != "active" and m.get("status") != "open":
        return skip("kalshi_market_not_open")
    try:
        book = api(session, API, "/markets/" + m["ticker"] + "/orderbook")
    except requests.RequestException:  # includes HTTP errors and invalid JSON
        return skip("kalshi_orderbook_unavailable")
    if not isinstance(book, dict):
        return skip("kalshi_orderbook_malformed")
    # Fetch completion is the earliest defensible observation timestamp.
    clock = existing.get("clock") or (
        (lambda: datetime.now(timezone.utc)) if existing.get("live_clock") else (lambda: now))
    observed = clock()
    row["observed_utc"] = observed.isoformat()
    if observed >= start-timedelta(minutes=2):
        return skip("orderbook_arrived_after_cutoff")
    # Slow upstream requests can carry inputs past their limits between the
    # run's start and quote receipt: re-apply the freshness windows at receipt.
    if observed-snap > timedelta(minutes=120):
        return skip("model_stale_at_quote_receipt")
    if observed-game["book_utc"] > timedelta(minutes=180):
        return skip("sportsbook_stale_at_quote_receipt")
    if multiplier is None or multiplier < 0:
        return skip("unverified_fee_multiplier")
    fill = taker_fill(book, qty, multiplier)
    ask = levels(book)
    if ask:
        row["book_best_ask"] = str(ask[0][0])
    if fill is None:
        return skip("insufficient_visible_ask_depth")
    row.update({key: str(value) for key, value in fill.items()})
    row["savings_pp"] = str((book_be-fill["kalshi_be"])*100)
    row["savings_pp_1x"] = str((book_be-fill["kalshi_be_1x"])*100)
    if (book_be-fill["kalshi_be"])*100 < min_savings:
        return skip("execution_savings_below_threshold")
    row.update(status="paper_filled", reason="depth_covered_fee_inclusive_quote")
    return row


def settle(positions, ledger, session, now):
    grade = {str(int(r["game_pk"])): r for r in ledger
             if r.get("game_pk", "").strip().isdigit() and r.get("status") == "graded"}
    for row in positions:
        if row["status"] != "paper_filled":
            continue
        g = grade.get(row["game_pk"])
        if not g:
            continue
        home, away = dec(g.get("full_home")), dec(g.get("full_away"))
        if home is None or away is None or home == away:
            continue
        try:
            m = api(session, API, "/markets/"+row["kalshi_ticker"])["market"]
        except (requests.RequestException, KeyError):
            continue  # leave open on API failure, try the next build
        if m.get("status") != "settled" or m.get("result") not in {"yes", "no"}:
            continue
        mlb_winner = g["home"] if home > away else g["away"]
        kalshi_winner = row["lean"] if m["result"] == "yes" else (
            row["home"] if row["lean"] == row["away"] else row["away"])
        if mlb_winner != kalshi_winner:
            row.update(status="needs_review", reason="kalshi_mlb_outcome_disagreement")
            continue
        qty = D(row["quantity"])
        payout = qty if m["result"] == "yes" else D(0)
        pnl = payout-qty*D(row["avg_price"])-D(row["estimated_fee"])
        row.update(status="paper_settled", pnl_dollars=str(pnl),
                   settled_utc=now.isoformat(), reason="confirmed_kalshi_and_mlb")


def matched_quotes(observations):
    """One depth-covered, fee-inclusive Kalshi quote per game: the filled one
    where a fill happened, else the game's first priced quote. This is the
    all-leans baseline -- the filter is judged against it on the SAME games."""
    per_game = {}
    for r in observations:
        if not r.get("kalshi_be") or not r.get("avg_price") or not r.get("quantity"):
            continue
        prior = per_game.get(r["game_pk"])
        if prior is None or (r["status"] == "paper_filled"
                             and prior["status"] != "paper_filled"):
            per_game[r["game_pk"]] = r
    return per_game


def ledger_graded(quotes, ledger):
    """Grade quotes by the MLB ledger's final score (not Kalshi settlement):
    returns (n, wins, losses, pending, pnl_dollars, mean_kalshi_be, mean_be_1x)."""
    finals = {}
    for g in ledger:
        pk = g.get("game_pk", "").strip()
        home, away = dec(g.get("full_home")), dec(g.get("full_away"))
        if (pk.isdigit() and g.get("status") == "graded" and home is not None
                and away is not None and home != away):
            finals[str(int(pk))] = g["home"] if home > away else g["away"]
    wins = losses = pending = 0
    pnl, be, be_1x, n_1x = D(0), D(0), D(0), 0
    for r in quotes:
        be += D(r["kalshi_be"])
        if r.get("kalshi_be_1x"):
            be_1x += D(r["kalshi_be_1x"])
            n_1x += 1
        winner = finals.get(r["game_pk"])
        if winner is None:
            pending += 1
            continue
        qty = D(r["quantity"])
        won = winner == r["lean"]
        wins, losses = wins + won, losses + (not won)
        pnl += (qty if won else D(0)) - qty * D(r["avg_price"]) - D(r["estimated_fee"] or 0)
    n = len(quotes)
    return (n, wins, losses, pending, pnl,
            be / n if n else None, be_1x / n_1x if n_1x else None)


def _comparison_line(label, quotes, ledger):
    n, w, l, pending, pnl, be, be_1x = ledger_graded(quotes, ledger)
    if not n:
        return f"  {label}: none yet"
    fmt = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"
    return (f"  {label}: n={n} graded {w}-{l} pending={pending} "
            f"P&L=${pnl.quantize(CENT)} mean fee-incl BE={fmt(be)} (1x fee {fmt(be_1x)})")


def summarize(observations, positions, outfile, now, note="", ledger=(), model_tag=""):
    # Every aggregate below is for the current model only; rows from earlier
    # model versions stay in the CSVs (and still settle) but are only counted.
    other = Counter(r.get("model_tag") or "untagged" for r in observations + positions
                    if model_tag and r.get("model_tag") != model_tag)
    if model_tag:
        observations = [r for r in observations if r.get("model_tag") == model_tag]
        positions = [r for r in positions if r.get("model_tag") == model_tag]
    counts = Counter(r["reason"] for r in observations if r["status"] == "skipped")
    # A fill stays a fill when settlement sends it to review; its unresolved
    # state is reported separately under Open/review.
    filled = [r for r in positions
              if r["status"] in {"paper_filled", "paper_settled", "needs_review"}]
    settled = [r for r in positions if r["status"] == "paper_settled"]
    total = sum((D(r["pnl_dollars"]) for r in settled), D(0))
    quotes = matched_quotes(observations)
    fill_pks = {r["game_pk"] for r in positions}
    text = ["KALSHI PAPER EXECUTION — READ-ONLY / NOT LIVE ORDERS",
            "As of: " + now.isoformat(),
            "Source: saved pregame dump of the shipped model (" + (model_tag or "unknown")
            + ") + live public Kalshi orderbook",
            "Decision rule: simulated immediate YES taker fill on the model lean;",
            "  only when fee-inclusive break-even improves on saved sportsbook price.",
            "This is NOT a claim of predictive edge or executable realized fills.",
            "Aggregates below: " + (model_tag or "all model tags") + " rows only. Other-tag rows "
            "(kept, not pooled): " + (", ".join(f"{k}={v}" for k, v in sorted(other.items()))
                                      or "none"),
            "Quote observations (all days): " + str(len(observations)),
            "Paper fills (all days): " + str(len(filled)),
            "Settled fills: " + str(len(settled)),
            "Settled hypothetical P&L: $" + str(total.quantize(CENT)),
            "Open/review: " + str(sum(r["status"] in {"paper_filled", "needs_review"} for r in positions)),
            "Skip reasons: " + (", ".join(f"{k}={v}" for k,v in sorted(counts.items())) or "none"),
            "Same-game comparison (graded by MLB ledger final score, not Kalshi settlement;",
            "  one quote per game; fills are a subset of all matched leans):",
            _comparison_line("All matched leans at Kalshi price", list(quotes.values()), ledger),
            _comparison_line("Paper fills only", [q for pk, q in quotes.items()
                                                  if pk in fill_pks], ledger),
            _comparison_line("Matched but not filled", [q for pk, q in quotes.items()
                                                        if pk not in fill_pks], ledger),
            "Fee sensitivity: fills whose saving stays >= 0 pp at the standard 1x fee: "
            + str(sum(1 for r in positions if r.get("savings_pp_1x")
                      and D(r["savings_pp_1x"]) >= 0)) + " of " + str(len(positions))]
    if note:
        text.append("NOTICE: " + note)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(text)+"\n", encoding="utf-8")
    print("\n".join(text))


def run(args, session=None, now=None):
    session = session or requests.Session()
    session.headers.update({"User-Agent": "xwoba-kalshi-paper/1.1"})
    now = now or datetime.now(timezone.utc)
    date = args.slate_date or now.astimezone(ET).strftime("%Y-%m-%d")
    root = Path(args.out_dir)
    quotes_file = root / "observations.csv"
    positions_file = root / "positions.csv"
    observations, positions = read_csv(quotes_file), read_csv(positions_file)
    data_path = Path(args.dumps_dir) / f"leans_{date}_xw.csv"
    ledger_path = Path(args.dumps_dir) / "mlb_lean_ledger.csv"
    model_tag = getattr(args, "model_tag", None) or current_model_tag()
    ledger = read_csv(ledger_path)
    report = lambda note: summarize(observations, positions, root/"report.txt", now,
                                    note, ledger, model_tag)
    # Settlement works even on dates without a new model dump.
    settle(positions, ledger, session, now)
    if not data_path.exists():
        write_csv(positions_file, positions)
        report("no saved pregame dump for " + date)
        return
    games = games_in_slate(read_csv(data_path))
    if not games:
        write_csv(positions_file, positions)
        report("empty model dump")
        return
    try:
        markets = get_markets(session)
        series = api(session, API, "/series/KXMLBGAME").get("series", {})
        schedule = api(session, MLB, "/schedule", {"sportId": 1, "date": date})
    except requests.RequestException as exc:
        # Observability even when the entire quote source is down. Fail closed;
        # never synthesize an ask from the last trade or yesterday's quotes.
        for game in games:
            row = {k: "" for k in FIELDS}
            row.update(observed_utc=now.isoformat(), game_pk=game["game_pk"],
                       game_date=game["game_date"], lean=game["lean"] or "",
                       model_tag=game["tag"],
                       status="skipped", reason="upstream_market_or_schedule_unavailable")
            observations.append(row)
        write_csv(quotes_file, observations)
        write_csv(positions_file, positions)
        report("Public upstream request failed: " + type(exc).__name__)
        return
    fee_type = series.get("fee_type", "")
    multiplier = (dec(series.get("fee_multiplier")) if fee_type in
                  {"quadratic", "quadratic_with_maker_fees"} else None)
    status = {str(g["gamePk"]): g for day in schedule.get("dates", [])
              for g in day.get("games", [])}
    existing = {"fixtures": games, "positions": {p["game_pk"] for p in positions},
                "live_clock": getattr(args, "use_wall_clock", False)}
    # Programmatic test clocks bypass wall-time; real CLI uses receipt time.
    for game in games:
        result = assess(game, markets, status, session, multiplier, now, existing,
                        args.quantity, D(str(args.min_savings_pp)), model_tag)
        observations.append(result)
        if result["status"] == "paper_filled":
            positions.append(result.copy())
            existing["positions"].add(game["game_pk"])
    write_csv(quotes_file, observations)
    write_csv(positions_file, positions)
    report("Kalshi series fee_type=" + str(fee_type) +
           ", fee_multiplier=" + str(series.get("fee_multiplier")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-date", help="Eastern MLB date (default: today ET)")
    parser.add_argument("--dumps-dir", default="data")
    parser.add_argument("--out-dir", default="data/paper_kalshi")
    parser.add_argument("--quantity", type=int, default=10)
    parser.add_argument("--min-savings-pp", type=float, default=0.5)
    parser.add_argument("--model-tag", help="model tag to paper-trade "
                        "(default: build_site.MODEL_TAG, the shipped model)")
    args = parser.parse_args()
    if not 1 <= args.quantity <= 100 or not 0 <= args.min_savings_pp <= 10:
        parser.error("quantity must be 1..100 and min-savings-pp 0..10")
    args.use_wall_clock = True
    run(args)


if __name__ == "__main__":
    main()