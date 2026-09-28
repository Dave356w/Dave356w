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
HELD_FILE_READERS = {"season_phase.py", "grade_leans.py", "build_site.py",
                     "validate_data_files.py"}

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
