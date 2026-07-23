import tempfile
import unittest
from pathlib import Path

import validate_data_files


class DataFileValidationTests(unittest.TestCase):
    def _write(self, directory, contents):
        path = Path(directory) / "sample.csv"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_valid_csv_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, 'game_pk,matchup\n123,"Away, Home"\n')
            self.assertIsNone(validate_data_files.validate_csv(path))

    def test_conflict_marker_reports_its_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "game_pk,side\n<<<<<<< Updated upstream\n")
            with self.assertRaisesRegex(
                ValueError, r"sample\.csv:2: unresolved Git conflict marker"
            ):
                validate_data_files.validate_csv(path)

    def test_uneven_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "game_pk,side\n123,away,unexpected\n")
            with self.assertRaisesRegex(
                ValueError, r"sample\.csv:2: expected 2 columns, found 3"
            ):
                validate_data_files.validate_csv(path)


if __name__ == "__main__":
    unittest.main()
