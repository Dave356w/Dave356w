"""Offline tests: no trading API credentials, no real orders and no network."""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper_kalshi as p

NOW = datetime(2026, 9, 22, 23, 30, tzinfo=timezone.utc)
START = datetime(2026, 9, 23, 0, 5, tzinfo=timezone.utc)
TICKER = "KXMLBGAME-26SEP222005NYMTEX-TEX"


def config(tmp_path, **overrides):
    defaults = dict(date="2026-09-22", data_dir=str(tmp_path),
                    settle_only=False, out_html=str(tmp_path / "paper.html"),
                    max_contracts=10, max_stake=20, max_open_exposure=100,
                    min_saved_pp=1, max_sportsbook_age=30, max_kalshi_age=15,
                    cutoff_seconds=120)
    return SimpleNamespace(**(defaults | overrides))


def rows():
    return [dict(game_pk="822840", side=side, opp_team=opp,
                 game_date="2026-09-22", model_tag=p.MODEL, model_metric="xwOBA",
                 edge_xwOBA=edge, snapshot_utc="2026-09-22T23:20:00Z",
                 scheduled_start_utc=START.isoformat(), pregame_market_utc=
                 "2026-09-22T23:20:00Z", hybrid_price_source="saved_pregame",
                 pregame_home_ml="-133", pregame_away_ml="+110")
            for side, opp, edge in (
                ("away", "Texas Rangers", ".021778"),
                ("home", "New York Mets", ".000400"))]


def dump(tmp_path, data=None):
    data = data or rows()
    target = tmp_path / "leans_2026-09-22_xw.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)
    return target


def market(**overrides):
    d = dict(ticker=TICKER, event_ticker=TICKER.rsplit("-", 1)[0],
             status="active", yes_ask_dollars="0.54", yes_ask_size_fp="25.00",
             updated_time="2026-09-22T23:29:00Z")
    return d | overrides


class FakeClient:
    def __init__(self, markets=None, settled=None, schedule=None):
        self.items = [market()] if markets is None else markets
        self.settled = settled or {}
        self.live = (dict(gamePk=822840, gameDate=START.isoformat(),
                          status={"abstractGameState": "Preview"})
                     if schedule is None else schedule)
        self.calls = []

    def kalshi(self, path, params=None):
        self.calls.append(("GET", path))
        if path == "/series/KXMLBGAME":
            return {"series": {"fee_type": "quadratic", "fee_multiplier": "0.5"}}
        if path.startswith("/markets/"):
            return {"market": self.settled.get(path.rsplit("/", 1)[-1],
                                                {"status": "active", "result": ""})}
        raise AssertionError(f"unknown GET: {path}")

    def markets(self):
        self.calls.append(("GET", "/markets"))
        return self.items

    def schedule(self, day):
        self.calls.append(("GET", "/schedule"))
        return {822840: self.live}


def test_mlb_lean_has_same_orientation_as_grader(tmp_path):
    import pandas as pd
    import grade_leans
    path = dump(tmp_path)
    game = p.model_games(path)[0]
    original = grade_leans.rows_from_dump(pd.DataFrame(rows()), None)[0]
    assert game["lean"] == original["xw_lean"] == "TEX"
    assert abs(float(game["net"]) - original["xw_net"]) < 1e-10
    assert original["xw_net"] > 0


def test_abstain_if_missing_rate(tmp_path):
    values = rows()
    values[0]["edge_xwOBA"] = ""
    assert p.model_games(dump(tmp_path, values))[0]["lean"] is None


@pytest.mark.parametrize("ml,expected", [
    (-120, Decimal(120) / 220), (+110, Decimal(100) / 210),
    (-100, Decimal("0.5")), (+100, Decimal("0.5")),
])
def test_sportsbook_breakeven(ml, expected):
    assert p.break_even(ml) == expected


def test_fee_rounding_and_half_rate():
    assert p.taker_fee(Decimal(".54"), 10, Decimal(".5")) == Decimal(".09")


def test_end_to_end_paper_and_idempotency(tmp_path):
    dump(tmp_path)
    c = FakeClient()
    cfg = config(tmp_path)
    t, a = p.run(cfg, c, NOW)
    assert len(t) == 1 and a[-1]["status"] == "paper"
    assert t[0]["selected"] == "TEX" and t[0]["qty"] == 10
    assert t[0]["fee"] == "0.09" and Decimal(t[0]["saved_pp"]) > 1
    assert Path(cfg.out_html).exists()
    assert p.read_csv(tmp_path / "kalshi_paper_trades.csv") == t
    t2, a2 = p.run(cfg, c, NOW + timedelta(minutes=1))
    assert len(t2) == 1 and a2[-1]["reason"] == "duplicate_game"
    assert all(x[0] == "GET" for x in c.calls)


def test_settlement_uses_kalshi_result_only(tmp_path):
    dump(tmp_path)
    cfg, c = config(tmp_path), FakeClient()
    t, _ = p.run(cfg, c, NOW)
    c.settled[TICKER] = {"status": "settled", "result": "yes"}
    settled, _ = p.run(config(tmp_path, settle_only=True), c, NOW + timedelta(hours=5))
    assert settled[0]["status"] == "settled"
    assert Decimal(settled[0]["pnl"]) == 10 - 10 * Decimal(".54") - Decimal(".09")
    again, _ = p.run(config(tmp_path, settle_only=True), c, NOW + timedelta(hours=6))
    assert again[0]["settled"] == settled[0]["settled"]


def test_first_pitch_cutoff_rejects(tmp_path):
    dump(tmp_path)
    t, a = p.run(config(tmp_path), FakeClient(), START - timedelta(seconds=90))
    assert not t and a[-1]["reason"] == "first_pitch_cutoff"


def test_market_matching_exact_datetime_and_doubleheader(tmp_path):
    game = p.model_games(dump(tmp_path))[0]
    assert p.exact_market(game, [market()])[1] == 1
    wrong = market(ticker="KXMLBGAME-26SEP221805NYMTEX-TEX",
                   event_ticker="KXMLBGAME-26SEP221805NYMTEX")
    assert p.exact_market(game, [wrong])[1] == 0
    assert p.exact_market(game, [market(), market()])[1] == 2


def test_no_quote_depth_or_post_hoc_fill(tmp_path):
    dump(tmp_path)
    t, a = p.run(config(tmp_path), FakeClient(markets=[market(yes_ask_size_fp="0")]), NOW)
    assert not t and a[-1]["reason"] == "no_top_ask_or_depth"


def test_sportsbook_timestamp_age_rejects(tmp_path):
    r = rows()
    for row in r:
        row["pregame_market_utc"] = "2026-09-22T21:00:00Z"
    dump(tmp_path, r)
    t, a = p.run(config(tmp_path), FakeClient(), NOW)
    assert not t and a[-1]["reason"] == "sportsbook_quote_stale"


def test_savings_threshold_and_risk_cap(tmp_path):
    dump(tmp_path)
    expensive = FakeClient(markets=[market(yes_ask_dollars=".58")])
    t, a = p.run(config(tmp_path), expensive, NOW)
    assert not t and a[-1]["reason"] == "below_savings_threshold"
    t, a = p.run(config(tmp_path, max_open_exposure=2), FakeClient(), NOW)
    assert not t and a[-1]["reason"] == "paper_risk_or_depth_limit"


def test_market_api_unavailable_fails_closed(tmp_path):
    dump(tmp_path)
    class Broken(FakeClient):
        def markets(self):
            raise requests.Timeout("test")
    import requests
    t, a = p.run(config(tmp_path), Broken(), NOW)
    assert not t and a[-1]["reason"] == "market_data_unavailable"
