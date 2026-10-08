"""Which ledger rows are regular season, and where the others live.

Every registered test, calibration, fit and report in this repo reads
data/mlb_lean_ledger.csv and was specified on regular-season games. None of
them was registered with the postseason in mind, and the StatsAPI schedule
returns postseason games (gameType F/D/L/W) from the same date query the build
uses, so without a mark they would enter every accumulator unannounced.

The rule is enforced at the one place the ledger is WRITTEN, not at the ~25
places it is read: grade_leans keeps a single in-memory frame (so ingest,
grading, market and actuals backfills run on every row alike) and splits it
on save. Rows confirmed `R` go to the main ledger; everything else -- any
postseason type, and any row whose type could not be confirmed -- goes to
POSTSEASON_LEDGER_NAME. Unconfirmed is FAIL-CLOSED: a regular-season row whose
schedule lookup failed waits in the held file and moves back once the next run
resolves it, which costs a delay; the opposite default would cost the
registrations their definition.

Legacy: ledgers written before this column existed are wholly regular season
(first row 2026-07-02, last pre-patch row 2026-09-24), so a main ledger with
no `game_type` column is stamped `R` on load. That is the ONLY place a missing
type is read as regular.
"""

import numpy as np
import pandas as pd

REGULAR = "R"
GAME_TYPE_COL = "game_type"
POSTSEASON_LEDGER_NAME = "mlb_postseason_ledger.csv"


def clean_game_type(value):
    """StatsAPI gameType as a stripped string, or NaN when absent."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def is_regular(frame):
    """Boolean Series: True only where game_type is confirmed `R`.

    A frame with no game_type column at all is a legacy or synthetic frame
    and is treated as wholly regular -- see the module docstring. Inside a
    frame that HAS the column, a missing value is not regular.
    """
    if GAME_TYPE_COL not in getattr(frame, "columns", ()):
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame[GAME_TYPE_COL].map(clean_game_type).eq(REGULAR).astype(bool)


def split_regular(frame):
    """(regular rows, held rows) -- held is postseason plus unconfirmed."""
    mask = is_regular(frame)
    return frame[mask], frame[~mask]
