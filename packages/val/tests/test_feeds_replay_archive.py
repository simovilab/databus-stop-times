"""test_feeds_replay_archive — Phase A1 tests for archived-Parquet replay.

Synthetic Hive-partitioned Parquet fixtures are written at runtime under
``tmp_path`` (never depends on a real gtfs-eta archive on disk). Covers:

1. read_vp_groups — column presence, ordering, time-window + route filters.
2. read_tu_groups — per-trip merge, stop_sequence ordering, epoch AND
   timestamp arrival_time columns.
3. build_archive_snapshots — ascending distinct ticks, TU-baseline
   ("greatest ts <= tick") attachment, VP-only replay.
4. ArchiveReplayFeedSource — async iteration, wall-clock independence.
5. Static GTFS matching — latest_local_snapshot_on_or_before, resolve_static_gtfs.
6. s3:// guard — clear error when fsspec/s3fs are unavailable (skipped if present).
7. End-to-end CLI: `run --replay-vp ... --replay-gtfs ...` via CliRunner.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stoptimes.val.cli.main import app
from stoptimes.val.engine.assembly import build_replay_run
from stoptimes.val.engine.runtime import FeedSnapshot
from stoptimes.val.feeds.replay_archive import (
    ArchiveReplayFeedSource,
    build_archive_snapshots,
    latest_local_snapshot_on_or_before,
    read_tu_groups,
    read_vp_groups,
    resolve_static_gtfs,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"

runner = CliRunner()

# ── Synthetic archive fixture builders ──────────────────────────────────────

# Base tick timestamps (epoch seconds UTC), chosen to land on 2024-01-02.
T0 = 1704153600.0  # 2024-01-02T00:00:00Z
T1 = T0 + 100.0
T2 = T0 + 200.0


def _write_hive_parquet(base_dir: Path, df: pd.DataFrame, partition_cols: list[str]) -> Path:
    """Write *df* as a Hive-partitioned Parquet dataset under *base_dir*."""
    base_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(base_dir), engine="pyarrow", partition_cols=partition_cols, index=False)
    return base_dir


def _vp_df(rows: list[dict]) -> pd.DataFrame:
    """Build a VP dataframe with the collector's partition columns attached."""
    out = []
    for r in rows:
        d = dict(r)
        date = dt.datetime.fromtimestamp(d["ts"], tz=dt.UTC).date()
        d.setdefault("year", date.year)
        d.setdefault("month", date.month)
        d.setdefault("day", date.day)
        d.setdefault("feed_name", "test-feed")
        d.setdefault("bearing", 0.0)
        d.setdefault("ingested_at", d["ts"])
        out.append(d)
    return pd.DataFrame(out)


def _write_vp_archive(tmp_path: Path, rows: list[dict], name: str = "vp") -> Path:
    df = _vp_df(rows)
    return _write_hive_parquet(
        tmp_path / name, df, partition_cols=["year", "month", "day", "route_id"]
    )


def _tu_df(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = dict(r)
        date = dt.datetime.fromtimestamp(d["ts"], tz=dt.UTC).date()
        d.setdefault("year", date.year)
        d.setdefault("month", date.month)
        d.setdefault("day", date.day)
        out.append(d)
    return pd.DataFrame(out)


def _write_tu_archive(tmp_path: Path, rows: list[dict], name: str = "tu") -> Path:
    df = _tu_df(rows)
    return _write_hive_parquet(
        tmp_path / name, df, partition_cols=["year", "month", "day", "route_id"]
    )


def _base_vp_rows() -> list[dict]:
    """Three ticks of one vehicle on trip1/route r1, advancing through mini_gtfs stops."""
    return [
        {
            "vehicle_id": "v1",
            "ts": T0,
            "lat": 42.3601,
            "lon": -71.0589,
            "speed": 3.0,
            "trip_id": "trip1",
            "route_id": "r1",
            "current_stop_sequence": 1,
            "current_status": "IN_TRANSIT_TO",
            "stop_id": "s1",
        },
        {
            "vehicle_id": "v1",
            "ts": T1,
            "lat": 42.3610,
            "lon": -71.0580,
            "speed": 3.0,
            "trip_id": "trip1",
            "route_id": "r1",
            "current_stop_sequence": 2,
            "current_status": "IN_TRANSIT_TO",
            "stop_id": "s2",
        },
        {
            "vehicle_id": "v1",
            "ts": T2,
            "lat": 42.3620,
            "lon": -71.0570,
            "speed": 3.0,
            "trip_id": "trip1",
            "route_id": "r1",
            "current_stop_sequence": 3,
            "current_status": "IN_TRANSIT_TO",
            "stop_id": "s3",
        },
    ]


# ── 1. read_vp_groups ────────────────────────────────────────────────────────


def test_read_vp_groups_basic_columns_and_order(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    df = read_vp_groups(vp_dir)
    assert list(df["ts"]) == sorted(df["ts"])
    assert len(df) == 3
    for col in ("vehicle_id", "lat", "lon", "trip_id", "stop_id", "route_id"):
        assert col in df.columns


def test_read_vp_groups_window_filter(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    df = read_vp_groups(vp_dir, start=T0 + 1, end=T2 - 1)
    assert list(df["ts"]) == [T1]


def test_read_vp_groups_route_filter(tmp_path: Path) -> None:
    rows = _base_vp_rows()
    other = dict(rows[0])
    other["route_id"] = "r2"
    other["trip_id"] = "trip2"
    other["ts"] = T0
    vp_dir = _write_vp_archive(tmp_path, rows + [other])

    df_r1 = read_vp_groups(vp_dir, route_ids=["r1"])
    assert set(df_r1["route_id"].astype(str)) == {"r1"}
    assert len(df_r1) == 3

    df_r2 = read_vp_groups(vp_dir, route_ids="r2")
    assert len(df_r2) == 1
    assert df_r2.iloc[0]["trip_id"] == "trip2"


# ── 2. read_tu_groups ────────────────────────────────────────────────────────


def test_read_tu_groups_merges_per_trip_ordered_by_stop_sequence(tmp_path: Path) -> None:
    rows = [
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 3,
            "stop_id": "s3",
            "arrival_delay": 0,
            "arrival_time": T2,
            "departure_time": T2 + 10,
        },
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 2,
            "stop_id": "s2",
            "arrival_delay": 0,
            "arrival_time": T1,
            "departure_time": T1 + 10,
        },
    ]
    tu_dir = _write_tu_archive(tmp_path, rows)
    groups = read_tu_groups(tu_dir)
    assert len(groups) == 1
    g = groups[0]
    assert g.trip_id == "trip1"
    assert g.ts == T0
    seqs = [stu.stop_sequence for stu in g.stop_time_updates]
    assert seqs == sorted(seqs)
    assert [stu.stop_id for stu in g.stop_time_updates] == ["s2", "s3"]


def test_read_tu_groups_handles_epoch_arrival_time(tmp_path: Path) -> None:
    rows = [
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 1,
            "stop_id": "s1",
            "arrival_delay": 0,
            "arrival_time": T1,  # plain epoch float
            "departure_time": T1 + 10,
        }
    ]
    tu_dir = _write_tu_archive(tmp_path, rows)
    groups = read_tu_groups(tu_dir)
    assert groups[0].stop_time_updates[0].arrival_time == pytest.approx(T1)


def test_read_tu_groups_handles_timestamp_arrival_time(tmp_path: Path) -> None:
    ts_value = pd.Timestamp(T1, unit="s", tz="UTC")
    rows = [
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 1,
            "stop_id": "s1",
            "arrival_delay": 0,
            "arrival_time": ts_value,  # pandas Timestamp column
            "departure_time": ts_value,
        }
    ]
    tu_dir = _write_tu_archive(tmp_path, rows)
    groups = read_tu_groups(tu_dir)
    assert groups[0].stop_time_updates[0].arrival_time == pytest.approx(T1, abs=1.0)


def test_read_tu_groups_window_and_route_filter(tmp_path: Path) -> None:
    rows = [
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 1,
            "stop_id": "s1",
            "arrival_delay": 0,
            "arrival_time": T1,
            "departure_time": T1,
        },
        {
            "ts": T2,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 1,
            "stop_id": "s1",
            "arrival_delay": 0,
            "arrival_time": T2 + 10,
            "departure_time": T2 + 10,
        },
    ]
    tu_dir = _write_tu_archive(tmp_path, rows)
    groups = read_tu_groups(tu_dir, start=T1, end=T2 + 1)
    assert len(groups) == 1
    assert groups[0].ts == T2


# ── 3. build_archive_snapshots ───────────────────────────────────────────────


def test_build_archive_snapshots_ticks_ascending(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    snaps = build_archive_snapshots(vp_dir)
    assert [s.header_timestamp for s in snaps] == [T0, T1, T2]
    assert all(isinstance(s, FeedSnapshot) for s in snaps)
    for s in snaps:
        assert len(s.vp_records) == 1
        assert s.vp_records[0].trip_id == "trip1"


def test_build_archive_snapshots_tu_baseline_attachment(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    tu_rows = [
        {
            "ts": T0,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 1,
            "stop_id": "s1",
            "arrival_delay": 0,
            "arrival_time": T0 + 5,
            "departure_time": T0 + 5,
        },
        {
            "ts": T1,
            "trip_id": "trip1",
            "route_id": "r1",
            "stop_sequence": 2,
            "stop_id": "s2",
            "arrival_delay": 0,
            "arrival_time": T1 + 5,
            "departure_time": T1 + 5,
        },
    ]
    tu_dir = _write_tu_archive(tmp_path, tu_rows)

    snaps = build_archive_snapshots(vp_dir, tu_dir)
    assert len(snaps) == 3

    # tick T0 -> baseline ts=T0 (stop_id s1)
    assert snaps[0].tu_records[0].stop_time_updates[0].stop_id == "s1"
    # tick T1 -> baseline ts=T1 (stop_id s2)
    assert snaps[1].tu_records[0].stop_time_updates[0].stop_id == "s2"
    # tick T2 -> greatest ts <= T2 is still T1's baseline (stop_id s2)
    assert snaps[2].tu_records[0].stop_time_updates[0].stop_id == "s2"


def test_build_archive_snapshots_vp_only_no_tu_path(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    snaps = build_archive_snapshots(vp_dir, tu_path=None)
    assert len(snaps) == 3
    for s in snaps:
        assert s.tu_records == []


def test_build_archive_snapshots_empty_vp_returns_empty(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    # Window excludes every archived row -> no ticks.
    snaps = build_archive_snapshots(vp_dir, start=T2 + 1000)
    assert snaps == []


# ── 4. ArchiveReplayFeedSource ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_archive_replay_feed_source_iterates_all_ticks(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    source = ArchiveReplayFeedSource(vp_dir)
    assert len(source.snapshots) == 3

    seen: list[float] = []
    async for snap in source:
        seen.append(snap.header_timestamp)
    assert seen == [T0, T1, T2]


@pytest.mark.asyncio
async def test_archive_replay_feed_source_exhausts_with_stop_iteration(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    source = ArchiveReplayFeedSource(vp_dir)
    with pytest.raises(StopAsyncIteration):
        async for _ in source:
            pass
        await source.__anext__()


# ── 5. Static GTFS matching ──────────────────────────────────────────────────


def test_latest_local_snapshot_on_or_before_picks_newest(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs_snapshots"
    gtfs_dir.mkdir()
    for name in ("2023-12-01.zip", "2024-01-01.zip", "2024-06-01.zip"):
        shutil.copy(_MINI_GTFS, gtfs_dir / name)

    found = latest_local_snapshot_on_or_before(gtfs_dir, dt.date(2024, 3, 1))
    assert found is not None
    assert found.name == "2024-01-01.zip"


def test_latest_local_snapshot_on_or_before_returns_none_when_no_match(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs_snapshots"
    gtfs_dir.mkdir()
    shutil.copy(_MINI_GTFS, gtfs_dir / "2024-06-01.zip")

    found = latest_local_snapshot_on_or_before(gtfs_dir, dt.date(2024, 1, 1))
    assert found is None


def test_latest_local_snapshot_on_or_before_missing_dir_returns_none(tmp_path: Path) -> None:
    found = latest_local_snapshot_on_or_before(tmp_path / "does-not-exist", dt.date(2024, 1, 1))
    assert found is None


def test_resolve_static_gtfs_local_zip_priority(tmp_path: Path) -> None:
    zip_copy = tmp_path / "explicit.zip"
    shutil.copy(_MINI_GTFS, zip_copy)
    resolved = resolve_static_gtfs(local_zip=zip_copy, local_dir=tmp_path / "unused")
    assert resolved == zip_copy


def test_resolve_static_gtfs_local_dir(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs_snapshots"
    gtfs_dir.mkdir()
    shutil.copy(_MINI_GTFS, gtfs_dir / "2024-01-01.zip")

    resolved = resolve_static_gtfs(local_dir=gtfs_dir, target_date=dt.date(2024, 1, 2))
    assert resolved.name == "2024-01-01.zip"


def test_resolve_static_gtfs_local_dir_no_match_raises(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs_snapshots"
    gtfs_dir.mkdir()
    shutil.copy(_MINI_GTFS, gtfs_dir / "2024-06-01.zip")
    with pytest.raises(FileNotFoundError):
        resolve_static_gtfs(local_dir=gtfs_dir, target_date=dt.date(2024, 1, 1))


def test_resolve_static_gtfs_raises_when_nothing_given() -> None:
    with pytest.raises(ValueError, match="local_zip"):
        resolve_static_gtfs()


# ── 6. s3:// guard ────────────────────────────────────────────────────────────


def test_s3_path_without_fsspec_raises_clear_error() -> None:
    pytest.importorskip("fsspec", reason="fsspec present — s3 guard path not exercised")


def test_s3_path_raises_when_dependency_missing() -> None:
    try:
        import fsspec  # noqa: F401

        pytest.skip("fsspec is installed; s3 unavailable-dependency guard not applicable")
    except ImportError:
        pass

    with pytest.raises(RuntimeError, match="fsspec"):
        read_vp_groups("s3://some-bucket/some/prefix")


# ── 7. End-to-end CLI ─────────────────────────────────────────────────────────


def test_cli_run_replay_vp_end_to_end(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        [
            "run",
            "--source",
            "mbta",
            "--replay-vp",
            str(vp_dir),
            "--replay-gtfs",
            str(_MINI_GTFS),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Replaying" in result.output
    assert "Done. Ticks=3" in result.output


def test_cli_run_replay_requires_gtfs_source(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        ["run", "--source", "mbta", "--replay-vp", str(vp_dir)],
    )

    assert result.exit_code == 1
    assert "--replay-gtfs" in result.output


# ── build_replay_run direct coverage ─────────────────────────────────────────


def test_build_replay_run_no_matching_vp_rows_raises(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    from stoptimes.val.feeds.source_config import load_named_source

    cfg = load_named_source("mbta")
    with pytest.raises(ValueError, match="No archived"):
        build_replay_run(
            cfg,
            vp_path=vp_dir,
            static_gtfs_zip=_MINI_GTFS,
            start=T2 + 1000,  # window excludes every row
        )


def test_build_replay_run_requires_static_gtfs_source(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    from stoptimes.val.feeds.source_config import load_named_source

    cfg = load_named_source("mbta")
    with pytest.raises(ValueError, match="static_gtfs"):
        build_replay_run(cfg, vp_path=vp_dir)
