"""Tests for spool file compaction (stoptimes.record.storage.spool_base).

DuckDB never returns freed blocks to the filesystem, so the VP spool reached
2.38 GB while holding 146 rows. These tests pin the rewrite that reclaims it:
that it preserves every row and the schema contract, that it only fires when
the file is both big and mostly dead space, and -- most importantly -- that a
failure leaves the spool exactly as it was.
"""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import pytest

from stoptimes.record.storage.spool import Spool
from .test_spool import _sample


def _spool(tmp_path) -> Spool:
    return Spool(str(tmp_path / "vp_spool.duckdb"))


FOREVER = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)


def _bloat(spool: Spool, *, cycles: int = 6, rows: int = 4000) -> None:
    """Grow the file the way production does: many appends, then a flush.

    Each cycle uses a distinct day so the natural key never collides, and the
    delete keeps only the tail -- the same insert-heavy/delete-nearly-all shape
    as an hour of polling followed by an hourly flush. That is what walked the
    live spool to 2.38 GB.
    """
    for cycle in range(cycles):
        for chunk in range(4):
            spool.append(
                _sample(
                    rows,
                    day=2 + cycle,
                    hour=chunk,
                    vehicle_prefix=f"c{cycle}h{chunk}v",
                )
            )
        spool.delete_before(dt.datetime(2025, 1, 2 + cycle, 3, tzinfo=dt.timezone.utc))


# --- the trigger ----------------------------------------------------------


def test_a_small_spool_is_never_compacted(tmp_path):
    spool = _spool(tmp_path)
    spool.append(_sample(5))
    assert spool.should_compact_file() is False
    assert spool.maybe_compact_file() == {"compacted": False}


def test_the_trigger_fires_on_the_shape_that_was_actually_observed(tmp_path, monkeypatch):
    """2.38 GB holding 146 rows -- the live VP spool on 2026-09-04.

    Size is stubbed rather than grown: reaching a real 2.38 GB takes weeks of
    poll/flush cycles, and it is the policy, not DuckDB, under test here.
    """
    spool = _spool(tmp_path)
    spool.append(_sample(146))
    monkeypatch.setattr(os.path, "getsize", lambda _p: 2_380_017_664)
    assert spool.should_compact_file() is True


def test_an_empty_spool_in_a_large_file_qualifies(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    monkeypatch.setattr(os.path, "getsize", lambda _p: 2_380_017_664)
    assert spool.should_compact_file() is True


def test_a_full_spool_is_left_alone_even_when_large(tmp_path, monkeypatch):
    """A big spool that is big because it holds data -- an S3 outage backlog --
    must not be rewritten: the rewrite would be expensive and reclaim little.
    """
    spool = _spool(tmp_path)
    spool.append(_sample(4000))
    # 4000 rows is a stand-in for the millions a real backlog holds; what the
    # gate reads is the ratio, and 2.38 GB / 4M rows is ~600 B/row.
    monkeypatch.setattr(os.path, "getsize", lambda _p: 4000 * 600)
    assert spool.should_compact_file(min_bytes=1) is False


# --- the invariant: rows are never at risk --------------------------------


def test_compaction_reclaims_space_a_delete_left_behind(tmp_path):
    """The end-to-end reclaim, on a real (small) file DuckDB genuinely bloats.

    An hour of polling followed by a flush leaves the file holding its tail of
    rows in space sized for the peak; the rewrite gives that back.
    """
    spool = _spool(tmp_path)
    _bloat(spool)
    rows_before = spool.stats()["rows"]
    size = os.path.getsize(spool.path)

    result = spool.compact_file()

    assert result == {
        "compacted": True,
        "rows": rows_before,
        "bytes_before": size,
        "bytes_after": result["bytes_after"],
    }
    assert result["bytes_after"] < size
    assert spool.stats()["rows"] == rows_before


def test_compaction_preserves_every_row(tmp_path):
    spool = _spool(tmp_path)
    _bloat(spool)
    spool.append(_sample(50, day=20))
    before = spool.select_before(FOREVER)
    assert len(before) > 0

    result = spool.compact_file()
    assert result["rows"] == len(before)

    pd.testing.assert_frame_equal(before, spool.select_before(FOREVER))


def test_the_natural_key_constraint_survives_compaction(tmp_path):
    spool = _spool(tmp_path)
    _bloat(spool)
    rows = _sample(10, day=20)
    spool.append(rows)
    total = spool.stats()["rows"]
    spool.compact_file()
    # Re-offering the same rows must still be a no-op, which only holds if the
    # UNIQUE constraint was carried into the rewritten file.
    assert spool.append(rows) == 0
    assert spool.stats()["rows"] == total


def test_a_failed_compaction_leaves_the_spool_untouched(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    spool.append(_sample(20))
    before = spool.select_before(FOREVER)
    size_before = os.path.getsize(spool.path)

    def boom(*args, **kwargs):
        raise RuntimeError("swap failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError, match="swap failed"):
        spool.compact_file()

    assert os.path.getsize(spool.path) == size_before
    pd.testing.assert_frame_equal(before, spool.select_before(FOREVER))
    # And no half-written temp file left behind to leak the space we came to
    # reclaim.
    assert [p.name for p in tmp_path.glob("*.compact-*")] == []


def test_maybe_compact_reports_failure_instead_of_raising(tmp_path, monkeypatch):
    """A flush that has already written and verified its rows must not fail
    because housekeeping did."""
    spool = _spool(tmp_path)
    spool.append(_sample(5))
    monkeypatch.setattr(
        Spool, "should_compact_file", lambda self, **kw: True
    )
    monkeypatch.setattr(
        Spool, "compact_file", lambda self: (_ for _ in ()).throw(OSError("disk full"))
    )
    result = spool.maybe_compact_file()
    assert result["compacted"] is False
    assert "disk full" in result["error"]


def test_compacting_an_empty_spool_is_safe(tmp_path):
    spool = _spool(tmp_path)
    result = spool.compact_file()
    assert result == {
        "compacted": True,
        "rows": 0,
        "bytes_before": result["bytes_before"],
        "bytes_after": result["bytes_after"],
    }
    assert spool.stats()["rows"] == 0
    assert spool.append(_sample(3)) == 3
