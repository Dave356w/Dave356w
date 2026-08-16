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


def test_no_rebuild_dump_is_ingestable():
    """A rebuild carries post-first-pitch rows and must never reach the ledger.

    `ingest` would reject them on `lock_status` anyway, but that check is the
    last line rather than the only one -- and the primary rebuild is a
    `leans_`-prefixed name one underscore from matching. Asserted for every
    name the build can write, against grade_leans' real globs.
    """
    pats = _grade_leans_dump_globs()
    names = [f"{bs.REBUILD_PREFIX}_leans_2026-08-10_{bs.DUMP_SUFFIX}.csv",
             f"{bs.REBUILD_PREFIX}_leans_2026-08-10_pl.csv",
             f"{bs.REBUILD_PREFIX}_{sm.SHADOW_PREFIX}_2026-08-10_"
             f"{sm.SHADOW_SUFFIX}.csv"]
    for name in names:
        for p in pats:
            assert not fnmatch.fnmatch(name, p), f"rebuild {name} matches {p}"


def test_the_naive_rebuild_suffix_would_have_been_ingested():
    """Pins WHY the rebuild marker is a prefix, exactly as SHADOW_PREFIX is."""
    pats = _grade_leans_dump_globs()
    assert any(fnmatch.fnmatch("leans_2026-08-10_rebuild_xw.csv", p)
               for p in pats)


def test_post_hoc_detection(monkeypatch):
    """Every game started -> rebuild. Any game not yet started -> live name."""
    import pandas as pd
    snap = "2026-08-11T06:50:00+00:00"
    started = pd.DataFrame({"scheduled_start_utc":
                            ["2026-08-10T23:07:00+00:00",
                             "2026-08-10T20:10:00+00:00"]})
    assert bs.dump_is_post_hoc(started, snap) is True
    mixed = pd.DataFrame({"scheduled_start_utc":
                          ["2026-08-10T23:07:00+00:00",
                           "2026-08-11T23:07:00+00:00"]})
    assert bs.dump_is_post_hoc(mixed, snap) is False


def test_unknowable_provenance_keeps_the_live_name():
    """Never name a dump `rebuild_` on a guess.

    A wrongly-renamed pregame dump is invisible to every glob that matters --
    the same slate loss the prefix exists to prevent, arrived at from the other
    side. Absent, empty and unparseable inputs all fall back to the live name.
    """
    import pandas as pd
    snap = "2026-08-11T06:50:00+00:00"
    assert bs.dump_is_post_hoc(pd.DataFrame(), snap) is False
    assert bs.dump_is_post_hoc(pd.DataFrame({"game_pk": [1]}), snap) is False
    assert bs.dump_is_post_hoc(
        pd.DataFrame({"scheduled_start_utc": ["not a timestamp"]}), snap) is False
    assert bs.dump_is_post_hoc(
        pd.DataFrame({"scheduled_start_utc": ["2026-08-10T23:07:00+00:00"]}),
        "not a timestamp") is False


def test_dump_path_renames_only_the_prefix():
    assert bs.dump_path("leans", "2026-08-10", "xw", False).endswith(
        "leans_2026-08-10_xw.csv")
    assert bs.dump_path("leans", "2026-08-10", "xw", True).endswith(
        f"{bs.REBUILD_PREFIX}_leans_2026-08-10_xw.csv")


def test_both_write_sites_go_through_dump_path():
    """The helper is only a guard if the writers call it.

    Source-level, deliberately: Savant is unreachable from CI, so neither write
    site can be exercised here, and a helper that is correct but bypassed looks
    identical to a working one in every other test in this file. Same shape as
    test_interaction_probe's check that the probe reads the live calibration.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mod, call in (("build_site.py", 'dump_path("leans"'),
                      ("shadow_metric.py", "bs.dump_path(SHADOW_PREFIX")):
        src = open(os.path.join(root, mod)).read()
        assert call in src, f"{mod} no longer writes its dump via dump_path"
        assert f'to_csv(os.path.join(DATA_DIR, f"leans_' not in src, (
            f"{mod} writes a dump path by hand, bypassing the rebuild rename")


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
