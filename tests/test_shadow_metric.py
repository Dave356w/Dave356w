"""Offline guards for the metric shadow arm.

The arm ran xwOBA against a wOBA primary until v11 and runs wOBA against an
xwOBA primary after it, so nothing here names a metric as a literal. Each test
derives the primary from build_site and the shadow from shadow_metric and
asserts the RELATIONSHIP -- that they differ, that every constant moves
together, that neither leaks into the other. Written against literals these
would all have gone red on the swap for reasons unrelated to what they guard.

Savant is unreachable from CI as well as from the dev environment, so nothing
here fetches. What these assert is the part that can be wrong silently: that
the patch repoints every constant the metric is read from, that the primary
build's constants are untouched by importing the shadow, and above all that the
shadow's dump cannot be ingested into the ledger.
"""
import fnmatch
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_site as bs
import shadow_metric as sm


@pytest.fixture(autouse=True)
def _restore():
    """Every test here mutates build_site globals; put them back afterwards."""
    saved = {k: getattr(bs, k) for k in
             ("MODEL_RATE_SOURCE_COL", "MODEL_RATE_LABEL", "MODEL_TAG",
              "STATCAST_SELECTIONS", "STATCAST_CACHE_NS",
              "MODEL_RATE_INTERNAL_COL")}
    yield
    for k, v in saved.items():
        setattr(bs, k, v)


def test_importing_shadow_does_not_move_the_primary_metric():
    """Import must be inert -- the daily build imports build_site too."""
    fresh = importlib.reload(bs)
    primary = fresh.MODEL_RATE_SOURCE_COL
    assert primary != sm.SHADOW_SOURCE_COL, (
        "the shadow arm must run the metric the primary does not")
    assert {"woba", "xwoba"} == {primary, sm.SHADOW_SOURCE_COL}
    assert fresh.MODEL_RATE_LABEL == ("wOBA" if primary == "woba" else "xwOBA")
    assert primary in fresh.STATCAST_SELECTIONS
    assert sm.SHADOW_SOURCE_COL not in fresh.STATCAST_SELECTIONS


def test_patch_repoints_every_metric_constant():
    primary_ns = bs.STATCAST_CACHE_NS
    primary_col = bs.MODEL_RATE_SOURCE_COL
    cfg = sm.patch()
    assert cfg["source_col"] == sm.SHADOW_SOURCE_COL
    assert cfg["label"] == sm.SHADOW_LABEL
    assert cfg["tag"] == sm.SHADOW_TAG
    # The two built at import time from the source column -- the silent-failure
    # pair. Patching the source column alone would request the primary's rate
    # and then look for a column that was never fetched.
    assert sm.SHADOW_SOURCE_COL in cfg["selections"]
    assert primary_col not in cfg["selections"]
    assert cfg["cache_ns"] != primary_ns


def test_selection_set_swaps_one_column_and_keeps_the_rest():
    before = list(bs.STATCAST_SELECTIONS)
    primary_col = bs.MODEL_RATE_SOURCE_COL
    cfg = sm.patch()
    assert len(cfg["selections"]) == len(before)
    assert set(before) - set(cfg["selections"]) == {primary_col}
    assert set(cfg["selections"]) - set(before) == {sm.SHADOW_SOURCE_COL}


def test_internal_schema_name_is_left_alone():
    """The dump/ledger key is a compatibility schema shared by both metrics."""
    cfg = sm.patch()
    assert cfg["internal_col"] == "xwOBA" == bs.MODEL_RATE_INTERNAL_COL


def test_cache_namespace_is_keyed_to_the_selection_set():
    """Primary and shadow request different columns, so they must never share a
    cache entry -- reusing one returns a CSV without the requested column."""
    primary_ns = bs.STATCAST_CACHE_NS
    cfg = sm.patch()
    assert cfg["cache_ns"] != primary_ns
    assert sm.SHADOW_SOURCE_COL in cfg["cache_ns"]


def _grade_leans_dump_globs():
    """The ingest patterns, read out of grade_leans' source rather than copied.

    A copy here would pass while the real globs drifted, which is the failure
    mode this whole test exists to prevent.
    """
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "grade_leans.py")).read()
    pats = re.findall(r'glob\.glob\(os\.path\.join\(DATA_DIR,\s*"([^"]+)"\)', src)
    assert pats, "could not find grade_leans' dump globs -- update this reader"
    return pats


def test_shadow_dump_is_not_ingestable():
    """The guard that matters: the ledger must never see a shadow row.

    `leans_*_xw.csv` matches ANY leans-prefixed name ending `_xw.csv`, so a
    suffix-based shadow name is one character from being ledgered as a real
    pending row. Asserted against grade_leans' actual patterns.
    """
    pats = _grade_leans_dump_globs()
    name = f"{sm.SHADOW_PREFIX}_2026-08-10_{sm.SHADOW_SUFFIX}.csv"
    for p in pats:
        assert not fnmatch.fnmatch(name, p), f"shadow dump {name} matches {p}"
    # Both generations, not just the current one: the arm wrote `_xw` dumps
    # before v11 and `_woba` after it, and either would be catastrophic to
    # ingest. The prefix is what makes both safe -- prove it for both.
    for suffix in ("xw", "woba"):
        old = f"{sm.SHADOW_PREFIX}_2026-08-10_{suffix}.csv"
        for p in pats:
            assert not fnmatch.fnmatch(old, p), f"shadow dump {old} matches {p}"


def test_the_naive_suffix_would_have_been_ingested():
    """Pins WHY the prefix form was chosen -- if this stops being true the
    comment above SHADOW_PREFIX is stale and the guard above is theatre."""
    pats = _grade_leans_dump_globs()
    naive = "leans_2026-08-10_shadow_xw.csv"
    assert any(fnmatch.fnmatch(naive, p) for p in pats)


def test_primary_dump_name_still_ingests():
    """The dump the build actually writes must be one grade_leans ingests.

    This is the highest-cost silent failure in the repo: a dump written under
    a suffix outside grade_leans' globs is never ledgered, and the slate's
    pregame rows cannot be recovered afterwards without lookahead. The suffix
    moved from `woba` to `xw` at v11, so it is read off build_site rather than
    written here as a literal -- a literal would keep passing while the build
    wrote somewhere else entirely.
    """
    pats = _grade_leans_dump_globs()
    real = f"leans_2026-08-10_{bs.DUMP_SUFFIX}.csv"
    assert any(fnmatch.fnmatch(real, p) for p in pats), (
        f"{real} matches none of {pats} -- this slate would be lost")


def test_the_dump_suffix_names_the_metric_inside_it():
    """A `_woba.csv` full of xwOBA rows would mislead every later reader,
    including shadow_report's filename fallback."""
    assert bs.DUMP_SUFFIX == ("woba" if bs.MODEL_RATE_SOURCE_COL == "woba"
                              else "xw")
    assert sm.SHADOW_SUFFIX == ("woba" if sm.SHADOW_SOURCE_COL == "woba"
                                else "xw")
    assert bs.DUMP_SUFFIX != sm.SHADOW_SUFFIX


def test_shadow_tag_is_in_no_family_map():
    """A shadow row shares a record line and a delta scale with nothing."""
    assert sm.SHADOW_TAG not in bs._RECORD_FAMILIES
    assert sm.SHADOW_TAG not in bs._SCALE_FAMILIES
    for fam in bs._RECORD_FAMILIES.values():
        assert sm.SHADOW_TAG not in fam
    for fam in bs._SCALE_FAMILIES.values():
        assert sm.SHADOW_TAG not in fam


def test_failure_is_swallowed_unless_strict(monkeypatch):
    """A shadow fault must not be able to fail the job that writes the ledger."""
    def boom(*a, **k):
        raise RuntimeError("savant down")
    monkeypatch.setattr(sm, "run", boom)
    assert sm.main([]) == 0
    with pytest.raises(RuntimeError):
        sm.main(["--strict"])
