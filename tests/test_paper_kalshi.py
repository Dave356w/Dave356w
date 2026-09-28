"""Offline fixture tests; no real network calls or trading credentials."""
import csv
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper_kalshi as paper

NOW = datetime(2026, 9, 22, 21, 0, tzinfo=timezone.utc)
START = "2026-09-22T22:05:00Z"
EVENT = "KXMLBGAME-26SEP221805NYMTEX"
MARKET = EVENT + "-TEX"


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class Session:
    def __init__(self, settlement=None):
        self.headers = {}
        self.calls = []
        self.settlement = settlement

    def get(self, url, params=None, timeout=15):
        self.calls.append((url, params))
        if url.endswith('/markets'):
            return Response({"markets": [{"ticker": MARKET, "event_ticker": EVENT,
                                          "market_type": "binary", "status": "open"}],
                             "cursor": ""})
        if url.endswith('/series/KXMLBGAME'):
            return Response({"series": {"fee_type": "quadratic", "fee_multiplier": 0.5}})
        if url.endswith('/schedule'):
            return Response({"dates": [{"games": [{"gamePk": 822840, "gameDate": START,
                         "status": {"abstractGameState": "Preview"}}]}]})
        if url.endswith('/orderbook'):
            return Response({"orderbook_fp": {"no_dollars": [["0.4800", "25.00"]],
                                                "yes_dollars": [["0.5000", "22.00"]]}})
        if url.endswith('/markets/' + MARKET):
            return Response({"market": self.settlement or {"status": "open"}})
        raise AssertionError(f"unmocked endpoint: {url}")


def model_rows():
    a = {"game_pk": 822840, "game_date": "2026-09-22", "side": "away",
         "opp_team": "Texas Rangers", "edge_xwOBA": "0.021",
         "scheduled_start_utc": START, "snapshot_utc": "2026-09-22T20:45:00Z",
         "model_tag": paper.current_model_tag(), "model_metric": "xwOBA",
         "pregame_market_utc": "2026-09-22T20:45:00Z",
         "pregame_home_ml": "-120", "pregame_away_ml": "+110"}
    h = {**a, "side": "home", "opp_team": "New York Mets", "edge_xwOBA": "0.008"}
    return [a, h]


def write_dump(tmp, rows=None):
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    rows = rows or model_rows()
    with (data / "leans_2026-09-22_xw.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return data


def args(tmp):
    return SimpleNamespace(slate_date="2026-09-22", dumps_dir=str(tmp / "data"),
                           out_dir=str(tmp / "data/paper_kalshi"),
                           quantity=10, min_savings_pp=0.5, use_wall_clock=False)


def test_depth_and_fee_rounding():
    result = paper.taker_fill({"orderbook_fp": {"no_dollars": [["0.48", "7"],
                                                               ["0.47", "5"]]}}, 10, Decimal("0.5"))
    assert result["quantity"] == 10
    assert result["avg_price"] == Decimal("0.523")
    # Fee is rounded per level, not once for the whole aggregate fill.
    assert result["estimated_fee"] == Decimal("0.10")
    assert paper.taker_fill({"orderbook_fp": {"no_dollars": [["0.48", "7"]]}}, 10, Decimal("0.5")) is None


def test_verified_market_and_no_doubleheader():
    games = paper.games_in_slate(model_rows())
    m = {"ticker": MARKET, "event_ticker": EVENT}
    assert games[0]["lean"] == "TEX"
    assert paper.market_for(games[0], [m], games)[0] == m
    assert paper.market_for(games[0], [m], games + [dict(games[0], game_pk="99")])[1] == "doubleheader_ambiguous"
    assert paper.market_for(games[0], [{"ticker": EVENT + "-NYM", "event_ticker": EVENT}], games)[0] is None


def test_end_to_end_once_duplicate_and_settle(tmp_path):
    data = write_dump(tmp_path)
    config = args(tmp_path)
    session = Session()
    paper.run(config, session=session, now=NOW)
    positions = paper.read_csv(data / "paper_kalshi/positions.csv")
    assert len(positions) == 1
    assert positions[0]["status"] == "paper_filled"
    assert Decimal(positions[0]["avg_price"]) == Decimal("0.52")
    assert Decimal(positions[0]["estimated_fee"]) == Decimal("0.09")
    assert Decimal(positions[0]["kalshi_be"]) == Decimal("0.529")
    assert Decimal(positions[0]["savings_pp"]) > Decimal("1.6")
    paper.run(config, session=session, now=NOW)
    assert len(paper.read_csv(data / "paper_kalshi/positions.csv")) == 1
    assert paper.read_csv(data / "paper_kalshi/observations.csv")[-1]["reason"] == "already_paper_filled"
    assert all("/portfolio/" not in u for u, _ in session.calls)

    ledger = [{"game_pk": "822840", "status": "graded", "home": "TEX", "away": "NYM",
               "full_home": "4", "full_away": "2"}]
    positions = paper.read_csv(data / "paper_kalshi/positions.csv")
    paper.settle(positions, ledger, Session(settlement={"status": "settled", "result": "yes"}), NOW)
    assert positions[0]["status"] == "paper_settled"
    assert Decimal(positions[0]["pnl_dollars"]) == Decimal("4.71")


def test_cutoff_stale_price_no_fill(tmp_path):
    data = write_dump(tmp_path)
    config = args(tmp_path)
    paper.run(config, session=Session(), now=datetime(2026, 9, 22, 22, 4, tzinfo=timezone.utc))
    assert paper.read_csv(data / "paper_kalshi/observations.csv")[-1]["reason"] == "outside_pregame_cutoff"
    assert paper.read_csv(data / "paper_kalshi/positions.csv") == []


def test_settlement_disagreement_stays_ungraded():
    row = {k: "" for k in paper.FIELDS}
    row.update(status="paper_filled", game_pk="822840", lean="TEX", home="TEX", away="NYM",
               kalshi_ticker=MARKET, quantity="10", avg_price="0.52", estimated_fee="0.09")
    ledger = [{"game_pk": "822840", "status": "graded", "home": "TEX", "away": "NYM",
               "full_home": "2", "full_away": "4"}]
    paper.settle([row], ledger, Session(settlement={"status": "settled", "result": "yes"}), NOW)
    assert row["status"] == "needs_review"
    assert row["pnl_dollars"] == ""


def test_upstream_failure_persists_skip_without_synthetic_fill(tmp_path):
    class FailingSession(Session):
        def get(self, url, params=None, timeout=15):
            if url.endswith("/markets"):
                raise paper.requests.ConnectionError("mock upstream down")
            return super().get(url, params, timeout)

    data = write_dump(tmp_path)
    paper.run(args(tmp_path), session=FailingSession(), now=NOW)
    assert paper.read_csv(data / "paper_kalshi/observations.csv")[-1]["reason"] == "upstream_market_or_schedule_unavailable"
    assert paper.read_csv(data / "paper_kalshi/positions.csv") == []
    assert "Public upstream request failed" in (data / "paper_kalshi/report.txt").read_text()


def test_orderbook_failure_persists_skip_without_position(tmp_path):
    class FailingBook(Session):
        def get(self, url, params=None, timeout=15):
            if url.endswith("/orderbook"):
                raise paper.requests.Timeout("mock slow orderbook")
            return super().get(url, params, timeout)

    data = write_dump(tmp_path)
    paper.run(args(tmp_path), session=FailingBook(), now=NOW)
    assert paper.read_csv(data / "paper_kalshi/observations.csv")[-1]["reason"] == "kalshi_orderbook_unavailable"
    assert paper.read_csv(data / "paper_kalshi/positions.csv") == []
def test_live_mlb_maker_fee_series_uses_taker_curve(tmp_path):
    # MLB series can charge both maker and taker fees; takers still use 7% quadratic.
    class MakerFeeSeries(Session):
        def get(self, url, params=None, timeout=15):
            if url.endswith('/series/KXMLBGAME'):
                return Response({"series": {"fee_type": "quadratic_with_maker_fees",
                                            "fee_multiplier": 0.5}})
            return super().get(url, params, timeout)

    data = write_dump(tmp_path)
    paper.run(args(tmp_path), session=MakerFeeSeries(), now=NOW)
    positions = paper.read_csv(data / "paper_kalshi/positions.csv")
    assert len(positions) == 1
    assert positions[0]["estimated_fee"] == "0.09"
    assert "quadratic_with_maker_fees" in (data / "paper_kalshi/report.txt").read_text()


# --- shipped-model tag, fee sensitivity, same-game baseline, concurrency -----

def test_default_tag_is_the_shipped_build_site_tag(monkeypatch):
    import build_site
    monkeypatch.delenv("MODEL_TAG", raising=False)
    assert paper.current_model_tag() == build_site.MODEL_TAG
    monkeypatch.setenv("MODEL_TAG", "xw+override_v99")
    assert paper.current_model_tag() == "xw+override_v99"


def test_unparseable_build_site_refuses_to_guess(tmp_path):
    fake = tmp_path / "build_site.py"
    fake.write_text("MODEL_TAG = compute()\n")
    try:
        paper.current_model_tag(fake)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_stale_model_tag_is_logged_and_never_filled(tmp_path):
    rows = model_rows()
    for r in rows:
        r["model_tag"] = "xw+retired_v1"
    data = write_dump(tmp_path, rows)
    paper.run(args(tmp_path), session=Session(), now=NOW)
    obs = paper.read_csv(data / "paper_kalshi/observations.csv")
    assert [o["reason"] for o in obs] == ["model_tag_not_current"]
    assert obs[0]["model_tag"] == "xw+retired_v1"
    assert paper.read_csv(data / "paper_kalshi/positions.csv") == []
    # The same dump fills when that tag is the one explicitly under test.
    paper.run(SimpleNamespace(**{**vars(args(tmp_path)), "model_tag": "xw+retired_v1"}),
              session=Session(), now=NOW)
    assert len(paper.read_csv(data / "paper_kalshi/positions.csv")) == 1


def test_standard_fee_is_recorded_beside_series_multiplier():
    fill = paper.taker_fill({"orderbook_fp": {"no_dollars": [["0.48", "25"]]}}, 10, Decimal("0.5"))
    # 10 @ 0.52: 0.07*10*0.52*0.48 = 0.17472 -> 0.18 at 1x; 0.08736 -> 0.09 at 0.5x.
    assert fill["estimated_fee"] == Decimal("0.09")
    assert fill["estimated_fee_1x"] == Decimal("0.18")
    assert fill["kalshi_be_1x"] == Decimal("0.538")


def test_positions_carry_both_savings_figures(tmp_path):
    data = write_dump(tmp_path)
    paper.run(args(tmp_path), session=Session(), now=NOW)
    pos = paper.read_csv(data / "paper_kalshi/positions.csv")[0]
    # Saved sportsbook -120 -> BE 0.545454...
    assert Decimal(pos["savings_pp"]) - Decimal(pos["savings_pp_1x"]) == Decimal("0.9")
    report = (data / "paper_kalshi/report.txt").read_text()
    assert "saving stays >= 0 pp at the standard 1x fee: 1 of 1" in report


def _obs(pk, status, lean, avg, fee, be, reason=""):
    row = {k: "" for k in paper.FIELDS}
    row.update(game_pk=pk, status=status, lean=lean, quantity="10", avg_price=avg,
               estimated_fee=fee, kalshi_be=be, kalshi_be_1x=be, reason=reason)
    return row


def test_baseline_compares_fills_with_all_matched_leans_on_same_games(tmp_path):
    observations = [
        _obs("1", "skipped", "TEX", "0.50", "0.09", "0.509", "execution_savings_below_threshold"),
        _obs("1", "paper_filled", "TEX", "0.50", "0.09", "0.509"),  # later fill wins
        _obs("2", "skipped", "NYM", "0.60", "0.09", "0.609", "execution_savings_below_threshold"),
        _obs("3", "skipped", "SEA", "", "", "", "market_not_found"),  # unpriced: excluded
        _obs("4", "skipped", "BOS", "0.40", "0.09", "0.409", "execution_savings_below_threshold"),
    ]
    quotes = paper.matched_quotes(observations)
    assert set(quotes) == {"1", "2", "4"}
    assert quotes["1"]["status"] == "paper_filled"
    ledger = [{"game_pk": "1", "status": "graded", "home": "TEX", "away": "NYM",
               "full_home": "5", "full_away": "1"},
              {"game_pk": "2", "status": "graded", "home": "LAD", "away": "NYM",
               "full_home": "3", "full_away": "2"},
              {"game_pk": "4", "status": "pending", "home": "BOS", "away": "NYY",
               "full_home": "", "full_away": ""}]
    n, w, l, pending, pnl, be, _ = paper.ledger_graded(list(quotes.values()), ledger)
    assert (n, w, l, pending) == (3, 1, 1, 1)
    # Win: 10 - 5.00 - 0.09 = 4.91; loss: -6.00 - 0.09 = -6.09.
    assert pnl == Decimal("-1.18")
    positions = [dict(observations[1])]
    paper.summarize(observations, positions, tmp_path / "r.txt", NOW, "", ledger, "tag")
    text = (tmp_path / "r.txt").read_text()
    assert "All matched leans at Kalshi price: n=3 graded 1-1 pending=1 P&L=$-1.18" in text
    assert "Paper fills only: n=1 graded 1-0 pending=0 P&L=$4.91" in text
    assert "Matched but not filled: n=2 graded 0-1 pending=1 P&L=$-6.09" in text


def test_empty_dump_still_persists_settlements(tmp_path):
    data = write_dump(tmp_path)
    paper.run(args(tmp_path), session=Session(), now=NOW)
    (data / "leans_2026-09-22_xw.csv").write_text("game_pk,side\n")
    with (data / "mlb_lean_ledger.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["game_pk", "status", "home", "away",
                                           "full_home", "full_away"])
        w.writeheader()
        w.writerow({"game_pk": "822840", "status": "graded", "home": "TEX",
                    "away": "NYM", "full_home": "4", "full_away": "2"})
    paper.run(args(tmp_path), session=Session(settlement={"status": "settled", "result": "yes"}),
              now=NOW)
    assert paper.read_csv(data / "paper_kalshi/positions.csv")[0]["status"] == "paper_settled"


def test_standalone_paper_workflow_stays_out_of_site_build_group():
    # Text-level (PyYAML is not a CI dependency): the concurrency block's group
    # must never name site-build, where a queued paper run could evict a
    # pending production build.
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/kalshi-paper.yml").read_text()
    block = text.split("\nconcurrency:\n", 1)[1].split("\n\n", 1)[0]
    group = next(line for line in block.splitlines() if line.strip().startswith("group:"))
    assert "site-build" not in group, group
    assert "'kalshi-paper'" in group
    assert "cancel-in-progress: false" in block
