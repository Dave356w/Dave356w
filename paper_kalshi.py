#!/usr/bin/env python3
"""Read-only prospective Kalshi MLB paper trades; NEVER submits real orders."""
import argparse
import csv
import html
import json
import math
import re
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
BASE = "https://external-api.kalshi.com/trade-api/v2"
MODEL = "xw+starter_blend_v13"
NAMES = dict(zip(
    ("Arizona Diamondbacks|Athletics|Atlanta Braves|Baltimore Orioles|Boston Red Sox|"
     "Chicago Cubs|Chicago White Sox|Cincinnati Reds|Cleveland Guardians|Colorado Rockies|"
     "Detroit Tigers|Houston Astros|Kansas City Royals|Los Angeles Angels|Los Angeles Dodgers|"
     "Miami Marlins|Milwaukee Brewers|Minnesota Twins|New York Mets|New York Yankees|"
     "Philadelphia Phillies|Pittsburgh Pirates|San Diego Padres|San Francisco Giants|"
     "Seattle Mariners|St. Louis Cardinals|Tampa Bay Rays|Texas Rangers|"
     "Toronto Blue Jays|Washington Nationals").split("|"),
    "ARI ATH ATL BAL BOS CHC CWS CIN CLE COL DET HOU KC LAA LAD MIA MIL MIN NYM NYY PHI PIT SD SF SEA STL TB TEX TOR WSH".split()))
ALIASES = {code: (code,) for code in NAMES.values()}
ALIASES.update({"ARI": ("ARI", "AZ"), "ATH": ("ATH", "OAK"),
                "CWS": ("CWS", "CHW"), "KC": ("KC", "KCR"),
                "SD": ("SD", "SDP"), "SF": ("SF", "SFG"),
                "TB": ("TB", "TBR"), "WSH": ("WSH", "WAS")})
EVENT = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})(\d{4})([A-Z]{4,7})$")
TRADE_FIELDS = ("id time game_pk date away home selected model snapshot net start "
                "sportsbook_ml sportsbook_be sportsbook_time ticker event quote_time market_update "
                "ask top_size qty fee_multiplier fee_source fee effective_be saved_pp "
                "status result settled payout pnl").split()
AUDIT_FIELDS = "time game_pk away home lean status reason ticker saved_pp".split()


def timestamp(s):
    try:
        value = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return value.astimezone(UTC) if value.tzinfo else None
    except (TypeError, ValueError):
        return None


def number(v):
    try:
        d = Decimal(str(v))
        return d if d.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def break_even(ml):
    p = number(ml)
    return (Decimal(100) / (100 + p) if p > 0 else -p / (100 - p)) if p and p != -100 else None


def taker_fee(ask, qty, multiplier):
    a, m = number(ask), number(multiplier)
    if a is None or m is None or not 0 < a < 1 or not 0 <= m <= 10 or qty < 1:
        raise ValueError("invalid fee input")
    return (Decimal("0.07") * qty * a * (1 - a) * m).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     delete=False, dir=path.parent) as f:
        tmp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def model_games(path):
    if not path.exists():
        return []
    pairs = {}
    for row in read_csv(path):
        try:
            pairs.setdefault(int(row["game_pk"]), {})[row["side"]] = row
        except (ValueError, KeyError):
            continue
    out = []
    for pk, p in pairs.items():
        if set(p) != {"home", "away"}:
            continue
        a, h = p["away"], p["home"]
        home, away = NAMES.get(a.get("opp_team")), NAMES.get(h.get("opp_team"))
        start, snap = timestamp(a.get("scheduled_start_utc")), timestamp(a.get("snapshot_utc"))
        if not home or not away or not start or not snap:
            continue
        if any(r.get("model_tag") != MODEL or r.get("model_metric") != "xwOBA"
               or r.get("snapshot_utc") != a.get("snapshot_utc")
               or r.get("scheduled_start_utc") != a.get("scheduled_start_utc")
               for r in (a, h)):
            continue
        eh, ea = number(a.get("edge_xwOBA")), number(h.get("edge_xwOBA"))
        net = eh - ea if eh is not None and ea is not None else None
        lean = (home if net > 0 else away if net < 0 else None) if net is not None else None
        out.append(dict(pk=pk, date=a["game_date"], away=away, home=home,
                        start=start, snapshot=snap, net=net, lean=lean,
                        sportsbook_time=timestamp(a.get("pregame_market_utc")),
                        sportsbook_source=a.get("hybrid_price_source"),
                        sportsbook_ml=a.get("pregame_home_ml" if lean == home
                                            else "pregame_away_ml")))
    return out


def exact_market(g, markets):
    matches = []
    for m in markets:
        ticker = m.get("ticker", "")
        if "-" not in ticker:
            continue
        event, team = ticker.rsplit("-", 1)
        parsed = EVENT.fullmatch(event)
        if not parsed or m.get("event_ticker") != event:
            continue
        try:
            et = datetime.strptime(parsed[1] + parsed[2], "%y%b%d%H%M").replace(tzinfo=ET)
        except ValueError:
            continue
        if abs((et.astimezone(UTC) - g["start"]).total_seconds()) > 1200:
            continue
        pairs = {a + h for a in ALIASES[g["away"]] for h in ALIASES[g["home"]]}
        if parsed[3] in pairs and g["lean"] and team in ALIASES[g["lean"]]:
            matches.append(m)
    return (matches[0] if len(matches) == 1 else None), len(matches)


class PublicClient:
    """GET-only, public quote client. No order endpoints or keys."""
    def __init__(self, session=None):
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = "Dave356w-V13-paper/1.0"

    def kalshi(self, path, params=None):
        r = self.s.get(BASE + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def schedule(self, day):
        r = self.s.get("https://statsapi.mlb.com/api/v1/schedule",
                       params={"sportId": 1, "date": day}, timeout=15)
        r.raise_for_status()
        return {int(g["gamePk"]): g for d in r.json().get("dates", [])
                for g in d.get("games", [])}

    def markets(self):
        markets, cursor = [], None
        for _ in range(8):
            params = {"series_ticker": "KXMLBGAME", "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            page = self.kalshi("/markets", params)
            markets += page.get("markets", [])
            cursor = page.get("cursor")
            if not cursor:
                return markets
        raise RuntimeError("market pagination limit exceeded")


def pregame(g, live, now, cutoff):
    if not g["lean"]:
        return "v13_abstained"
    if g["snapshot"] >= g["start"] or g["snapshot"] > now:
        return "model_not_pregame"
    if not live or live.get("status", {}).get("abstractGameState") != "Preview":
        return "game_not_confirmed_preview"
    actual = timestamp(live.get("gameDate"))
    if not actual or abs((actual - g["start"]).total_seconds()) > 1200:
        return "first_pitch_changed"
    if now >= actual - timedelta(seconds=cutoff):
        return "first_pitch_cutoff"
    if now - g["snapshot"] > timedelta(hours=6):
        return "model_snapshot_too_old"
    return None


def quote(g, market, mult, source, now, cfg, trades, exposure):
    if any(str(t["game_pk"]) == str(g["pk"]) for t in trades):
        return None, "duplicate_game", None
    if market.get("status") not in ("open", "active"):
        return None, "market_not_open", None
    if g["sportsbook_source"] != "saved_pregame" or not g["sportsbook_time"]:
        return None, "missing_saved_sportsbook_quote", None
    if g["sportsbook_time"] > g["snapshot"] + timedelta(seconds=60):
        return None, "sportsbook_after_model", None
    if now - g["sportsbook_time"] > timedelta(minutes=cfg.max_sportsbook_age):
        return None, "sportsbook_quote_stale", None
    sb = break_even(g["sportsbook_ml"])
    if sb is None:
        return None, "missing_sportsbook_price", None
    updated = timestamp(market.get("updated_time"))
    if not updated or updated > now + timedelta(seconds=60):
        return None, "invalid_quote_timestamp", None
    if now - updated > timedelta(minutes=cfg.max_kalshi_age):
        return None, "kalshi_quote_stale", None
    ask, size = number(market.get("yes_ask_dollars")), number(market.get("yes_ask_size_fp"))
    if ask is None or not 0 < ask < 1 or size is None or size < 1:
        return None, "no_top_ask_or_depth", None
    if market.get("fee_type") not in (None, "", "quadratic", "quadratic_fee"):
        return None, "unrecognized_market_fee", None
    qty = min(cfg.max_contracts, int(size), int(Decimal(str(cfg.max_stake)) / ask))
    while qty:
        fee = taker_fee(ask, qty, mult)
        cost = ask * qty + fee
        if cost <= Decimal(str(cfg.max_stake)) and (
                exposure + cost <= Decimal(str(cfg.max_open_exposure))):
            break
        qty -= 1
    if not qty:
        return None, "paper_risk_or_depth_limit", None
    effective = ask + fee / qty
    savings = (sb - effective) * 100
    if savings < Decimal(str(cfg.min_saved_pp)):
        return None, "below_savings_threshold", savings
    ticker = market["ticker"]
    trade = dict(id=f"{g['pk']}:{ticker}:V13:YES", time=now.isoformat(),
                 game_pk=g["pk"], date=g["date"], away=g["away"], home=g["home"],
                 selected=g["lean"], model=MODEL, snapshot=g["snapshot"].isoformat(),
                 net=str(g["net"]), start=g["start"].isoformat(),
                 sportsbook_ml=g["sportsbook_ml"], sportsbook_be=str(sb),
                 sportsbook_time=g["sportsbook_time"].isoformat(), ticker=ticker,
                 event=market["event_ticker"], quote_time=now.isoformat(),
                 market_update=updated.isoformat(), ask=str(ask), top_size=str(size),
                 qty=qty, fee_multiplier=str(mult), fee_source=source, fee=str(fee),
                 effective_be=str(effective), saved_pp=str(savings),
                 status="open", result="", settled="", payout="", pnl="")
    return trade, "simulated_taker_fill_at_observed_ask", savings


def settle(trades, client, now):
    for t in trades:
        if t.get("status") != "open":
            continue
        try:
            market = client.kalshi("/markets/" + t["ticker"])["market"]
        except (requests.RequestException, ValueError, KeyError):
            continue
        result = str(market.get("result", "")).lower()
        if market.get("status") not in ("settled", "finalized") or result not in ("yes", "no"):
            continue
        qty, ask, fee = Decimal(t["qty"]), Decimal(t["ask"]), Decimal(t["fee"])
        payout = qty if result == "yes" else Decimal(0)
        t.update(status="settled", result=result, settled=now.isoformat(),
                 payout=str(payout), pnl=str(payout - qty * ask - fee))


def render(trades, audit, destination):
    closed = [t for t in trades if t["status"] == "settled"]
    pnl = sum((Decimal(t["pnl"]) for t in closed), Decimal(0))
    print(f"Kalshi paper: {len(trades)} simulated, {len(closed)} settled, P&L USD {pnl:.2f}")
    print("Audit reasons: " + json.dumps(
        {reason: sum(a["reason"] == reason for a in audit)
         for reason in sorted({a["reason"] for a in audit})}))
    if destination:
        fields = ("time", "selected", "ticker", "ask", "qty", "saved_pp", "status", "pnl")
        cells = "".join("<tr>" + "".join("<td>" + html.escape(str(t.get(f, ""))) +
                        "</td>" for f in fields) + "</tr>" for t in trades[-60:][::-1])
        page = ('<!doctype html><meta charset="utf-8"><title>V13 Kalshi Paper</title>'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}'
                'table{border-collapse:collapse;font-size:13px}td,th{padding:.5rem;'
                'border-bottom:1px solid #ccc;text-align:left}div{overflow-x:auto}</style>'
                '<h1>V13 Kalshi — paper only</h1><p>Indicative observed asks, estimated '
                'fees and hypothetical fills only. No orders submitted. Not a calibrated '
                'model edge.</p><p>' + html.escape(
                    f"{len(trades)} trades; {len(closed)} settled; P&L USD {pnl:.2f}") +
                '</p><p><a href="index.html">Matchups</a></p><div><table><tr>' +
                "".join("<th>" + html.escape(f) + "</th>" for f in fields) +
                "</tr>" + cells + "</table></div>")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page, encoding="utf-8")


def run(cfg, client=None, now=None):
    now, client = now or datetime.now(UTC), client or PublicClient()
    data = Path(cfg.data_dir)
    tp, ap = data / "kalshi_paper_trades.csv", data / "kalshi_paper_audit.csv"
    trades, audit = (read_csv(tp) if tp.exists() else [],
                     read_csv(ap) if ap.exists() else [])
    settle(trades, client, now)
    date = cfg.date or now.astimezone(ET).strftime("%Y-%m-%d")
    games = [] if cfg.settle_only else model_games(data / f"leans_{date}_xw.csv")
    if games:
        try:
            schedule = client.schedule(date)
            series = client.kalshi("/series/KXMLBGAME").get("series", {})
            if series.get("fee_type") not in (None, "", "quadratic", "quadratic_fee"):
                raise ValueError("unsupported series fee_type")
            raw = number(series.get("fee_multiplier"))
            mult = raw if raw is not None and 0 <= raw <= 10 else Decimal(1)
            source = "series" if raw is not None else "conservative_1x_default"
            markets = client.markets()
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            print(f"Read-only data unavailable: {type(exc).__name__}: {exc}")
            schedule, markets, mult, source = {}, [], Decimal(1), "unavailable"
        exposure = sum((Decimal(t["ask"]) * Decimal(t["qty"]) + Decimal(t["fee"])
                        for t in trades if t["status"] == "open"), Decimal(0))
        for g in games:
            reason = pregame(g, schedule.get(g["pk"]), now, cfg.cutoff_seconds)
            ticker, savings, status = "", None, "skip"
            if not reason and not markets:
                reason = "market_data_unavailable"
            if not reason:
                market, count = exact_market(g, markets)
                if count != 1:
                    reason = "no_exact_market" if count == 0 else "ambiguous_market"
                else:
                    ticker = market["ticker"]
                    t, reason, savings = quote(g, market, mult, source, now, cfg, trades, exposure)
                    if t:
                        trades.append(t)
                        exposure += Decimal(t["ask"]) * Decimal(t["qty"]) + Decimal(t["fee"])
                        status = "paper"
            audit.append(dict(time=now.isoformat(), game_pk=g["pk"], away=g["away"],
                              home=g["home"], lean=g["lean"] or "", status=status,
                              reason=reason, ticker=ticker,
                              saved_pp=str(savings) if savings is not None else ""))
    save_csv(tp, TRADE_FIELDS, trades)
    save_csv(ap, AUDIT_FIELDS, audit)
    render(trades, audit, Path(cfg.out_html) if cfg.out_html else None)
    return trades, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", help="Eastern slate date YYYY-MM-DD")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-html")
    p.add_argument("--settle-only", action="store_true")
    p.add_argument("--max-contracts", type=int, default=10)
    p.add_argument("--max-stake", type=float, default=20)
    p.add_argument("--max-open-exposure", type=float, default=100)
    p.add_argument("--min-saved-pp", type=float, default=1)
    p.add_argument("--max-sportsbook-age", type=int, default=30)
    p.add_argument("--max-kalshi-age", type=int, default=15)
    p.add_argument("--cutoff-seconds", type=int, default=120)
    cfg = p.parse_args()
    if min(cfg.max_contracts, cfg.max_stake, cfg.max_open_exposure) <= 0:
        p.error("positive paper risk limits required")
    run(cfg)


if __name__ == "__main__":
    main()
