"""Tests for the TripUpdates spool and its run-length compaction.

Run against tmp_path -- no Django settings, no S3 endpoint. The flush is
exercised against a local directory, which DuckDB reads/writes exactly as it
does ``s3://``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from rt_pipeline.storage.tu_schema import DATA_COLUMNS, NATURAL_KEY, SERIES_KEY
from rt_pipeline.storage.tu_spool import (
    TripUpdateSpool,
    compact_repeated_predictions,
    flush_tu_to_staging,
)

BASE = dt.datetime(2026, 9, 4, 12, 0, 0, tzinfo=dt.timezone.utc)


def _row(**over) -> dict:
    row = {
        "feed_name": "mbta",
        "route_id": "15",
        "trip_id": "trip-1",
        "start_date": "20260904",
        "start_time": None,
        "direction_id": 1,
        "vehicle_id": "y1234",
        "trip_schedule_relationship": "SCHEDULED",
        "revenue": True,
        "stop_id": "s3",
        "stop_sequence": 3,
        "arrival_time": BASE + dt.timedelta(minutes=5),
        "arrival_delay": 30,
        "arrival_uncertainty": None,
        "departure_time": None,
        "departure_delay": None,
        "departure_uncertainty": None,
        "stu_schedule_relationship": "SCHEDULED",
        "ts": BASE,
        "header_ts": BASE,
        "ingested_at": BASE,
    }
    row.update(over)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(DATA_COLUMNS))


def _spool(tmp_path) -> TripUpdateSpool:
    return TripUpdateSpool(str(tmp_path / "tu_spool.duckdb"))


# --- schema contract -------------------------------------------------------


def test_series_key_is_the_natural_key_without_the_as_of_instant():
    assert tuple(NATURAL_KEY) == (*SERIES_KEY, "ts")
    assert set(NATURAL_KEY) <= set(DATA_COLUMNS)


# --- append / dedup --------------------------------------------------------


def test_append_dedups_identical_polls_of_the_same_instant(tmp_path):
    sp = _spool(tmp_path)
    assert sp.append(_frame([_row()])) == 1
    # Same (trip, stop, ts) seen again -- a repeated poll of an unchanged
    # feed -- must not become a second row.
    assert sp.append(_frame([_row(ingested_at=BASE + dt.timedelta(seconds=30))])) == 0
    assert sp.stats()["rows"] == 1


def test_append_keeps_the_same_stop_at_a_later_instant(tmp_path):
    sp = _spool(tmp_path)
    sp.append(_frame([_row()]))
    assert sp.append(_frame([_row(ts=BASE + dt.timedelta(seconds=30))])) == 1


def test_append_separates_trip_instances_by_start_date(tmp_path):
    sp = _spool(tmp_path)
    sp.append(_frame([_row()]))
    assert sp.append(_frame([_row(start_date="20260905")])) == 1


def test_append_missing_columns_raises(tmp_path):
    sp = _spool(tmp_path)
    with pytest.raises(ValueError):
        sp.append(_frame([_row()]).drop(columns=["arrival_uncertainty"]))


# --- run-length compaction -------------------------------------------------


def test_unchanged_repeats_are_dropped(tmp_path):
    rows = [_row(ts=BASE + dt.timedelta(seconds=30 * i)) for i in range(4)]
    out = compact_repeated_predictions(_frame(rows))
    # Four identical predictions carry the information of one.
    assert len(out) == 1
    assert out.iloc[0]["ts"] == BASE


def test_a_changed_prediction_is_kept(tmp_path):
    rows = [
        _row(ts=BASE, arrival_time=BASE + dt.timedelta(minutes=5)),
        _row(ts=BASE + dt.timedelta(seconds=30), arrival_time=BASE + dt.timedelta(minutes=6)),
    ]
    assert len(compact_repeated_predictions(_frame(rows))) == 2


def test_a_to_b_to_a_survives_intact():
    """The reason compaction compares against the immediate predecessor and
    not against every value seen: a prediction that reverts must stay a
    three-row history, or a replay reconstructs the wrong middle."""
    a = BASE + dt.timedelta(minutes=5)
    b = BASE + dt.timedelta(minutes=6)
    rows = [
        _row(ts=BASE, arrival_time=a),
        _row(ts=BASE + dt.timedelta(seconds=30), arrival_time=b),
        _row(ts=BASE + dt.timedelta(seconds=60), arrival_time=a),
    ]
    out = compact_repeated_predictions(_frame(rows))
    assert len(out) == 3


def test_series_are_compacted_independently():
    rows = [
        _row(ts=BASE, stop_id="s3", stop_sequence=3),
        _row(ts=BASE, stop_id="s4", stop_sequence=4),
        _row(ts=BASE + dt.timedelta(seconds=30), stop_id="s3", stop_sequence=3),
        _row(ts=BASE + dt.timedelta(seconds=30), stop_id="s4", stop_sequence=4,
             arrival_time=BASE + dt.timedelta(minutes=9)),
    ]
    out = compact_repeated_predictions(_frame(rows))
    # s3 repeated unchanged (1 row); s4 changed (2 rows).
    assert sorted(out["stop_id"]) == ["s3", "s4", "s4"]


def test_uncertainty_change_alone_is_a_change():
    """A prediction that flips from realtime to schedule-plus-historical keeps
    the same arrival time but changes uncertainty; that transition is the
    whole reason the column is archived, so it must not be compacted away."""
    rows = [
        _row(ts=BASE, arrival_uncertainty=None),
        _row(ts=BASE + dt.timedelta(seconds=30), arrival_uncertainty=300),
    ]
    assert len(compact_repeated_predictions(_frame(rows))) == 2


def test_a_vehicle_swap_is_a_change_even_when_the_prediction_is_not():
    """A vehicle reassignment on an otherwise-unchanged trip is a real event,
    and it is what a join to the VehiclePositions archive keys on -- so it
    must survive compaction."""
    rows = [
        _row(ts=BASE, vehicle_id="y1234"),
        _row(ts=BASE + dt.timedelta(seconds=30), vehicle_id="y5678"),
    ]
    assert len(compact_repeated_predictions(_frame(rows))) == 2


def test_nulls_compare_as_equal_not_as_distinct():
    rows = [
        _row(ts=BASE, arrival_time=None, arrival_delay=None),
        _row(ts=BASE + dt.timedelta(seconds=30), arrival_time=None, arrival_delay=None),
    ]
    assert len(compact_repeated_predictions(_frame(rows))) == 1


def test_compact_empty_frame_is_a_noop():
    out = compact_repeated_predictions(_frame([]))
    assert out.empty


# --- flush -----------------------------------------------------------------


def test_flush_writes_day_partitions_and_reports_compaction(tmp_path):
    sp = _spool(tmp_path)
    sp.append(_frame([_row(ts=BASE + dt.timedelta(seconds=30 * i)) for i in range(4)]))
    base_uri = str(tmp_path / "tu_staging")
    cutoff = dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)

    result = flush_tu_to_staging(sp, base_uri, cutoff)

    assert result["flushed"] == 4      # rows drained from the spool
    assert result["written"] == 1      # rows that survived compaction
    assert result["days"] == ["2026-09-04"]
    assert (tmp_path / "tu_staging" / "year=2026" / "month=9" / "day=4").is_dir()
    # No route fan-out in the partitioning -- same rule as the VP flush.
    assert not any("route_id=" in str(p) for p in (tmp_path / "tu_staging").rglob("*"))
    assert sp.stats()["rows"] == 0


def test_flush_keeps_the_first_row_of_each_window(tmp_path):
    """Compaction is per flush window, so an unchanged prediction still gets
    one row per hour -- which is what keeps "last value at or before T"
    correct across file boundaries."""
    sp = _spool(tmp_path)
    sp.append(_frame([_row(ts=BASE), _row(ts=BASE + dt.timedelta(seconds=30))]))
    base_uri = str(tmp_path / "tu_staging")

    first = flush_tu_to_staging(sp, base_uri, BASE + dt.timedelta(seconds=15))
    assert first["written"] == 1

    sp.append(_frame([_row(ts=BASE + dt.timedelta(seconds=60))]))
    second = flush_tu_to_staging(sp, base_uri, dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc))
    # Unchanged from the previous window, but still written: a window never
    # depends on rows already deleted from the spool.
    assert second["written"] == 1


def test_flush_empty_spool_is_a_noop(tmp_path):
    sp = _spool(tmp_path)
    result = flush_tu_to_staging(sp, str(tmp_path / "tu_staging"), BASE)
    assert result["flushed"] == 0
    assert result["days"] == []
    assert result["written"] == 0
    # A small, healthy spool is nowhere near the compaction thresholds.
    assert result["compaction"] == {"compacted": False}


def test_flush_leaves_spool_intact_when_write_raises(tmp_path, monkeypatch):
    sp = _spool(tmp_path)
    sp.append(_frame([_row()]))

    class _FailingConnection:
        def register(self, name, df): pass
        def unregister(self, name): pass
        def execute(self, sql, *args, **kwargs):
            if "COPY" in sql:
                raise RuntimeError("simulated S3 PUT failure")
            return self
        def fetchone(self): return (0,)
        def close(self): pass

    monkeypatch.setattr(
        "rt_pipeline.storage.tu_spool.connect",
        lambda base_uri, config=None: _FailingConnection(),
    )
    with pytest.raises(RuntimeError, match="simulated S3 PUT failure"):
        flush_tu_to_staging(sp, str(tmp_path / "tu_staging"), BASE + dt.timedelta(hours=1))
    assert sp.stats()["rows"] == 1
