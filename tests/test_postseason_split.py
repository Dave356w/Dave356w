"""Postseason and type-unconfirmed rows never reach the regular-season ledger.

The chain this file holds, end to end:
  build_site records StatsAPI gameType and stamps it on the dump ->
  grade_leans carries it into the row, resolves blanks from the schedule ->
  save_ledger writes only confirmed `R` rows to mlb_lean_ledger.csv ->
  validate_data_files fails CI if anything else ever lands there ->
  every registered accumulator reads mlb_lean_ledger.csv and nothing else.
Break any link and a test below fails.
"""

import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock

import pandas as pd

import build_site
import grade_leans
import season_phase
import validate_data_files

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every module that scores a registration or feeds the published record from
# the ledger. Each must read the main ledger and only the main ledger.
REGISTERED_ACCUMULATORS = (
    "forward_test", "hybrid_test", "hybrid_v2", "delta_filter_test",
    "abstain_test", "dog_contrast_test", "b2_tmr_test", "price_calibration_shadow",
)

# The only modules allowed to know the held file exists.
# paper_kalshi.py is a read-only execution diagnostic, not a registered test:
# it already holds postseason fills (they come from the slate dump) and reads
# the held file only to settle/grade them by game_pk.
HELD_FILE_READERS = {"season_phase.py", "grade_leans.py", "build_site.py",
                     "validate_data_files.py", "paper_kalshi.py"}

DAY = "2026-10-06"
REGULAR_PK, POSTSEASON_PK, BLANK_PK = 9001, 9002, 9003


def _dump(game_pk, game_type):
    common = dict(game_pk=game_pk, game_date=DAY, model_tag="test_v3",
                  snapshot_utc=f"{DAY}T16:00:00Z",
                  scheduled_start_utc=f"{DAY}T23:00:00Z",
                  game_type=game_type)
    return [
        dict(common, side="away", pitcher="Away Pitcher", opp_team="Home Team",
             opp_xwOBA=.330, pit_xwOBA=.310, edge_xwOBA=.004),
        dict(common, side="home", pitcher="Home Pitcher", opp_team="Away Team",
             opp_xwOBA=.320, pit_xwOBA=.300, edge_xwOBA=-.016),
    ]


def _schedule(types):
    """The `_linescores_for` shape: gamePk -> game, all final 5-3 home."""
    innings = [{"away": {"runs": 0}, "home": {"runs": 1}}] * 5
    return {pk: {"gamePk": pk, "gameType": gt,
                 "status": {"detailedState": "Final"},
                 "linescore": {"teams": {"away": {"runs": 3}, "home": {"runs": 5}},
                               "innings": innings}}
            for pk, gt in types.items()}


class _TempLedger:
    """Point grade_leans at a scratch data dir and run main() there."""

    def __init__(self, td):
        self.td = td
        self.main = os.path.join(td, "mlb_lean_ledger.csv")
        self.held = os.path.join(td, season_phase.POSTSEASON_LEDGER_NAME)

    def run(self, schedule):
        lookup = (schedule if callable(schedule)
                  else (lambda day: schedule))
        with mock.patch.object(grade_leans, "DATA_DIR", self.td), \
                mock.patch.object(grade_leans, "LEDGER_PATH", self.main), \
                mock.patch.object(grade_leans, "POSTSEASON_LEDGER_PATH", self.held), \
                mock.patch.object(grade_leans, "REPORT_PATH",
                                  os.path.join(self.td, "report.txt")), \
                mock.patch.object(grade_leans, "_linescores_for", lookup), \
                mock.patch.object(grade_leans, "attach_market", lambda led: led), \
                mock.patch.object(grade_leans, "attach_actuals", lambda led: led), \
                mock.patch("builtins.print"):
            grade_leans.main()

    def pks(self, path):
        if not os.path.exists(path):
            return set()
        return set(pd.read_csv(path)["game_pk"].astype(int))


class BuildStampsGameType(unittest.TestCase):
    def test_slate_records_statsapi_game_type(self):
        payload = {"dates": [{"date": DAY, "games": [{
            "gamePk": POSTSEASON_PK, "gameType": "D", "gameDate": f"{DAY}T23:00:00Z",
            "teams": {"away": {"team": {"id": 1, "name": "A", "abbreviation": "AAA"}},
                      "home": {"team": {"id": 2, "name": "H", "abbreviation": "HHH"}}},
            "status": {}, "venue": {}}]}]}
        with mock.patch.object(build_site, "_get_json", return_value=payload):
            slate = build_site.get_slate(DAY)
        self.assertEqual(slate.loc[0, "game_type"], "D")

    def test_dump_is_stamped_by_game_pk_and_unknown_pks_stay_blank(self):
        slate = pd.DataFrame({"game_pk": [REGULAR_PK, POSTSEASON_PK],
                              "game_type": ["R", "D"]})
        frame = pd.DataFrame({"game_pk": [REGULAR_PK, POSTSEASON_PK, BLANK_PK]})
        build_site.stamp_game_type(frame, slate)
        self.assertEqual(frame["game_type"].iloc[0], "R")
        self.assertEqual(frame["game_type"].iloc[1], "D")
        self.assertTrue(pd.isna(frame["game_type"].iloc[2]))


class LedgerSplit(unittest.TestCase):
    def _write_dumps(self, td):
        rows = (_dump(REGULAR_PK, "R") + _dump(POSTSEASON_PK, "D")
                + _dump(BLANK_PK, None))
        pd.DataFrame(rows).to_csv(os.path.join(td, f"leans_{DAY}_xw.csv"), index=False)

    def test_postseason_row_is_held_and_blank_row_is_resolved_from_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_dumps(td)
            led = _TempLedger(td)
            led.run(_schedule({REGULAR_PK: "R", POSTSEASON_PK: "D", BLANK_PK: "R"}))

            self.assertEqual(led.pks(led.main), {REGULAR_PK, BLANK_PK})
            self.assertEqual(led.pks(led.held), {POSTSEASON_PK})
            # Graded in the held file too: recorded, just not counted.
            held = pd.read_csv(led.held)
            self.assertEqual(held.loc[0, "status"], "graded")
            validate_data_files.validate_data_dir(td)

    def test_unconfirmed_row_fails_closed_then_rejoins_once_resolved(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_dumps(td)
            led = _TempLedger(td)

            # The schedule does not return the blank game (lookup miss), so
            # its type cannot be confirmed on this run.
            led.run(_schedule({REGULAR_PK: "R", POSTSEASON_PK: "D"}))
            # Nothing confirmed, so nothing reaches the main ledger.
            self.assertEqual(led.pks(led.main), {REGULAR_PK})
            self.assertEqual(led.pks(led.held), {POSTSEASON_PK, BLANK_PK})
            validate_data_files.validate_data_dir(td)

            led.run(_schedule({REGULAR_PK: "R", POSTSEASON_PK: "D", BLANK_PK: "R"}))
            self.assertEqual(led.pks(led.main), {REGULAR_PK, BLANK_PK})
            self.assertEqual(led.pks(led.held), {POSTSEASON_PK})

    def test_report_numbers_exclude_held_rows_and_footer_names_them(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_dumps(td)
            led = _TempLedger(td)
            led.run(_schedule({REGULAR_PK: "R", POSTSEASON_PK: "D", BLANK_PK: "R"}))
            with mock.patch.object(grade_leans, "LEDGER_PATH", led.main), \
                    mock.patch.object(grade_leans, "POSTSEASON_LEDGER_PATH", led.held):
                combined = grade_leans.load_ledger(include_held=True)
                regular_only = grade_leans.load_ledger()
            self.assertEqual(len(combined), 3)
            self.assertEqual(set(regular_only["game_pk"].astype(int)),
                             {REGULAR_PK, BLANK_PK})
            text = grade_leans.report_text(combined)
            body, _, footer = text.partition("HELD OUT of every number above")
            self.assertIn("D=1", footer)
            self.assertEqual(body.rstrip("\n"), grade_leans.report_text(regular_only))

    def test_legacy_main_ledger_without_the_column_is_regular_season(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mlb_lean_ledger.csv")
            pd.DataFrame([dict(game_pk=1, game_date="2026-07-02", status="graded")]
                         ).to_csv(path, index=False)
            validate_data_files.validate_data_dir(td)
            with mock.patch.object(grade_leans, "LEDGER_PATH", path):
                self.assertEqual(grade_leans.load_ledger()["game_type"].tolist(), ["R"])


class PostseasonReadout(unittest.TestCase):
    """The held footer scores postseason rows without touching the body."""

    @staticmethod
    def _row(pk, gt, lean, fa, fh, p_home, away_ml, home_ml, status="graded"):
        full = None
        if status == "graded" and lean:
            full = "W" if lean == ("HHH" if fh > fa else "AAA") else "L"
        return dict(game_pk=pk, game_date=DAY, away="AAA", home="HHH",
                    model_tag="test_v3", game_type=gt, status=status,
                    xw_lean=lean, xw_full=full, xw_f5=None,
                    full_away=fa, full_home=fh, close_p_home=p_home,
                    close_away_ml=away_ml, close_home_ml=home_ml)

    def test_record_units_ev_null_and_rounds(self):
        held = pd.DataFrame([
            # home fav wins, lean home at -150: +0.667u
            self._row(1, "F", "HHH", 2, 5, 0.58, 130, -150),
            # lean away dog +130 loses: -1u; chalk (home) wins
            self._row(2, "F", "AAA", 1, 4, 0.58, 130, -150),
            # division series: lean away +120 wins: +1.2u; chalk (home) loses
            self._row(3, "D", "AAA", 6, 3, 0.53, 120, -140),
            # abstention and a pending game are counted, not scored
            self._row(4, "D", None, 6, 3, 0.53, 120, -140),
            self._row(5, "D", "HHH", None, None, None, None, None, status="pending"),
        ])
        text = "\n".join(grade_leans._held_lines(held))
        self.assertIn("POSTSEASON (not registered; descriptive only)", text)
        all_block = text.split("  ALL:")[1].split("  F wild card:")[0]
        self.assertIn("graded=4  pending=1", all_block)
        self.assertIn("abstained=1", all_block)
        self.assertIn("lean full: 2-1", all_block)
        self.assertIn("+0.87u", all_block)                  # 0.667 - 1 + 1.2
        self.assertIn("same-row always-chalk: 2-1", all_block)
        self.assertIn("F wild card: 2 rows", text)
        self.assertIn("D division series: 3 rows", text)
        self.assertNotIn("  L LCS:", text)
        # Says the season rates are frozen, once, before the scope lines.
        self.assertEqual(text.count("  Inputs: Season rates"), 1)
        self.assertLess(text.index("  Inputs:"), text.index("  ALL:"))
        self.assertIn("postseason PAs are not folded in",
                      " ".join(l.strip() for l in text.splitlines()))
        self.assertTrue(all(len(l) <= 96 for l in text.splitlines()
                            if "Inputs" in l or "folded" in l))

        # EV - null == excess on the rendered line (the #227 invariant).
        line = next(l for l in all_block.splitlines() if "vs close" in l)
        num = lambda tag: float(re.search(tag + r"\s+([+-]?\d+\.\d)", line).group(1))
        ev, exc = num(r"EV"), num(r"excess")
        null = float(re.search(r"\(null\s+([+-]?\d+\.\d)\)", line).group(1))
        self.assertAlmostEqual(ev - null, exc, delta=0.15)

    def test_unpriced_rows_say_so_instead_of_scoring(self):
        held = pd.DataFrame([self._row(1, "W", "HHH", 2, 5, None, None, None)])
        text = "\n".join(grade_leans._held_lines(held))
        self.assertIn("lean full: 1-0", text)
        self.assertIn("closing price: none attached yet", text)
        self.assertNotIn("vs close", text)

    def test_unconfirmed_rows_get_no_postseason_block(self):
        held = pd.DataFrame([self._row(1, None, "HHH", 2, 5, 0.58, 130, -150)])
        text = "\n".join(grade_leans._held_lines(held))
        self.assertNotIn("POSTSEASON (", text)
        self.assertNotIn("Inputs:", text)
        self.assertIn("await a game-type lookup", text)


class PostseasonPage(unittest.TestCase):
    """postseason.html shows confirmed postseason rows; grades.html never does."""

    @staticmethod
    def _row(pk, gt, lean, fa, fh, p_home, away_ml, home_ml, status="graded"):
        full = None
        if status == "graded" and lean:
            full = "W" if lean == ("HHH" if fh > fa else "AAA") else "L"
        return dict(game_pk=pk, game_date=DAY, away="AAA", home="HHH",
                    away_sp="A SP", home_sp="H SP",
                    model_tag=build_site.MODEL_TAG, game_type=gt, status=status,
                    xw_lean=lean, xw_full=full, xw_delta=0.02, xw_net=0.02,
                    full_away=fa, full_home=fh, close_p_home=p_home,
                    close_away_ml=away_ml, close_home_ml=home_ml)

    def _render(self, held_rows, main_rows=()):
        with tempfile.TemporaryDirectory() as td:
            main = os.path.join(td, "mlb_lean_ledger.csv")
            held = os.path.join(td, season_phase.POSTSEASON_LEDGER_NAME)
            pd.DataFrame(list(main_rows) or [self._row(1, "R", "HHH", 2, 5, .58, 130, -150)]
                         ).to_csv(main, index=False)
            if held_rows is not None:
                pd.DataFrame(held_rows).to_csv(held, index=False)
            with mock.patch.object(build_site, "LEDGER_PATH", main), \
                    mock.patch.object(build_site, "POSTSEASON_LEDGER_PATH", held):
                return (build_site.render_postseason_html("t"),
                        build_site.render_grades_html("t"))

    def test_rounds_record_units_and_controls(self):
        page, grades = self._render([
            self._row(9101, "F", "HHH", 2, 5, .58, 130, -150),   # +0.667u
            self._row(9102, "F", "AAA", 1, 4, .58, 130, -150),   # -1u
            self._row(9103, "D", "AAA", 6, 3, .53, 120, -140),   # +1.2u
            self._row(9104, "D", "HHH", None, None, None, None, None,
                      status="pending"),
            self._row(9105, None, "HHH", 2, 5, .58, 130, -150),  # unconfirmed
        ])
        self.assertIn("Postseason ledger", page)
        self.assertIn(f"<div class='l'>{build_site.PUBLIC_MODEL_NAME}</div>"
                      "<div class='v cool'>2-1</div>", page)
        self.assertIn("+0.87u", page)
        self.assertIn("<div class='l'>Wild Card</div><div class='v'>1-1</div>", page)
        self.assertIn("<div class='l'>Division Series</div><div class='v'>1-0</div>", page)
        self.assertIn("<div class='l'>Always chalk</div><div class='v dim'>2-1</div>", page)
        self.assertIn("1 pending", page)
        self.assertIn("· Wild Card, Division Series", page)
        note = build_site._esc(season_phase.POSTSEASON_INPUTS_NOTE)
        self.assertIn(note, page)
        self.assertNotIn(note, grades)
        # The unconfirmed row is neither here nor on the regular-season page.
        for html in (page, grades):
            self.assertNotIn("9105", html)
        # grades.html links the page and scores none of its games.
        self.assertIn("href='postseason.html'", grades)
        self.assertEqual(grades.count("<tr class='gr-row"), 1)  # the main row only

    def test_empty_when_no_postseason_rows(self):
        page, _ = self._render(None)
        self.assertIn("No postseason games yet", page)
        page, _ = self._render([self._row(9105, None, "HHH", 2, 5, .58, 130, -150)])
        self.assertIn("No postseason games yet", page)


class Guards(unittest.TestCase):
    def test_validator_rejects_postseason_or_blank_row_in_main_ledger(self):
        for bad in ("D", ""):
            with self.subTest(game_type=bad), tempfile.TemporaryDirectory() as td:
                pd.DataFrame([dict(game_pk=1, game_type="R"),
                              dict(game_pk=2, game_type=bad)]).to_csv(
                    os.path.join(td, "mlb_lean_ledger.csv"), index=False)
                with self.assertRaisesRegex(ValueError, "regular-season ledger"):
                    validate_data_files.validate_data_dir(td)

    def test_every_registered_accumulator_reads_only_the_main_ledger(self):
        import importlib
        for name in REGISTERED_ACCUMULATORS:
            with self.subTest(module=name):
                mod = importlib.import_module(name)
                self.assertEqual(os.path.basename(mod.LEDGER), "mlb_lean_ledger.csv")

    def test_no_other_module_reads_the_held_file(self):
        pattern = re.compile(r"POSTSEASON_LEDGER|mlb_postseason_ledger")
        offenders = [p.name for p in REPO.glob("*.py")
                     if p.name not in HELD_FILE_READERS
                     and pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])

    def test_build_reads_held_rows_only_for_the_pregame_lock(self):
        enclosing, callers = None, []
        for line in (REPO / "build_site.py").read_text(encoding="utf-8").splitlines():
            m = re.match(r"def (\w+)\(", line)
            if m:
                enclosing = m.group(1)
            elif "include_held=True" in line:
                callers.append(enclosing)
        self.assertEqual(callers, ["locked_pregame_rows"])

if __name__ == "__main__":
    unittest.main()
