"""Offline guards for the xwOBA shadow arm.

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
    assert fresh.MODEL_RATE_SOURCE_COL == "woba"
    assert fresh.MODEL_RATE_LABEL == "wOBA"
    assert "woba" in fresh.STATCAST_SELECTIONS
    assert "xwoba" not in fresh.STATCAST_SELECTIONS


def test_patch_repoints_every_metric_constant():
    cfg = sm.patch()
    assert cfg["source_col"] == "xwoba"
    assert cfg["label"] == "xwOBA"
    assert cfg["tag"] == sm.SHADOW_TAG
    # The two built at import time from the source column -- the silent-failure
    # pair. Patching the source column alone would request `woba` and then look
    # for an `xwoba` that was never fetched.
    assert "xwoba" in cfg["selections"] and "woba" not in cfg["selections"]
    assert cfg["cache_ns"] != "custom_woba_v1"


def test_selection_set_swaps_one_column_and_keeps_the_rest():
    before = list(bs.STATCAST_SELECTIONS)
    cfg = sm.patch()
    assert len(cfg["selections"]) == len(before)
    assert set(before) - set(cfg["selections"]) == {"woba"}
    assert set(cfg["selections"]) - set(before) == {"xwoba"}


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
    name = f"{sm.SHADOW_PREFIX}_2026-08-10_xw.csv"
    for p in pats:
        assert not fnmatch.fnmatch(name, p), f"shadow dump {name} matches {p}"


def test_the_naive_suffix_would_have_been_ingested():
    """Pins WHY the prefix form was chosen -- if this stops being true the
    comment above SHADOW_PREFIX is stale and the guard above is theatre."""
    pats = _grade_leans_dump_globs()
    naive = "leans_2026-08-10_shadow_xw.csv"
    assert any(fnmatch.fnmatch(naive, p) for p in pats)


def test_primary_dump_name_still_ingests():
    """Sanity on the reader: the real dump must match, or the globs moved."""
    pats = _grade_leans_dump_globs()
    assert any(fnmatch.fnmatch("leans_2026-08-10_woba.csv", p) for p in pats)


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
