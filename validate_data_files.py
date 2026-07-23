#!/usr/bin/env python3
"""Reject unresolved merge markers and malformed rows in committed CSV data."""

import csv
import sys
from pathlib import Path


CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def validate_csv(path):
    """Raise ValueError when a CSV contains conflict markers or uneven rows."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        rows = csv.reader(src)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty CSV") from exc
        if not header:
            raise ValueError(f"{path}: empty header")
        expected = len(header)
        for line_number, row in enumerate(rows, start=2):
            first = row[0].strip() if row else ""
            if first.startswith(CONFLICT_MARKERS):
                raise ValueError(
                    f"{path}:{line_number}: unresolved Git conflict marker"
                )
            if len(row) != expected:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected} columns, "
                    f"found {len(row)}"
                )


def validate_data_dir(data_dir="data"):
    """Validate every top-level CSV in data_dir and return the checked paths."""
    paths = sorted(Path(data_dir).glob("*.csv"))
    for path in paths:
        validate_csv(path)
    return paths


def main():
    try:
        paths = validate_data_dir()
    except (OSError, ValueError) as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"data validation: {len(paths)} CSV files OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
