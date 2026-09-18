"""Integration tests for stoptimes.record.recorder.Recorder.

All offline: HTTP is mocked with respx, time is a FakeClock (no real
sleeping), and storage is a tmp_path directory (no real S3).
"""

from __future__ import annotations

import asyncio
import json
import signal

import httpx
import pytest
import respx
from stoptimes.core.contracts.source import ArchiveKind, FeedFormat, FeedKind, RealtimeFeed
from stoptimes.record.recorder import Recorder, RecorderConfig, spool_path_for


def _vp_body(vp_feed_dict: dict) -> bytes:
    return json.dumps(vp_feed_dict).encode()


@pytest.mark.asyncio
async def test_duration_exits_cleanly_and_flushes(make_source, vp_feed_dict, fake_clock, tmp_path):
    source = make_source()
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=12.0, jitter_frac=0.0
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=_vp_body(vp_feed_dict)))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock)
            result = await recorder.run()

    assert result.ok
    assert result.stopped_reason == "duration_elapsed"
    stats = result.stats["vehicle_positions"]
    assert stats.polls >= 2  # 12s / 5s interval
    assert stats.written == 1  # only the first poll wrote a row; rest were unchanged

    archive_root = tmp_path / "archive"
    written_files = list(archive_root.rglob("*.parquet"))
    assert len(written_files) == 1, "final flush after duration elapsed should have run"


@pytest.mark.asyncio
async def test_unchanged_payload_is_not_rewritten(make_source, vp_feed_dict, fake_clock, tmp_path):
    source = make_source()
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=20.0, jitter_frac=0.0
    )

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(feed.url).mock(
            return_value=httpx.Response(200, content=_vp_body(vp_feed_dict))
        )
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock)
            result = await recorder.run()

    assert route.call_count >= 3  # several polls
    stats = result.stats["vehicle_positions"]
    assert stats.written == 1
    assert stats.skipped_unchanged == stats.polls - 1


@pytest.mark.asyncio
async def test_500_then_success_is_recorded(make_source, vp_feed_dict, fake_clock, tmp_path):
    source = make_source()
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=1.0, jitter_frac=0.0, max_retries=3
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(
            side_effect=[httpx.Response(500), httpx.Response(200, content=_vp_body(vp_feed_dict))]
        )
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock)
            result = await recorder.run()

    stats = result.stats["vehicle_positions"]
    assert stats.written == 1
    assert stats.errors == 0  # the retry succeeded before exhausting attempts


@pytest.mark.asyncio
async def test_malformed_protobuf_does_not_crash_the_loop(make_source, fake_clock, tmp_path):
    feed = RealtimeFeed(
        kind=FeedKind.VEHICLE_POSITIONS,
        url="http://example.test/vp.pb",
        format=FeedFormat.PROTOBUF,
        poll_interval_s=5.0,
    )
    source = make_source(feeds=[feed])
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=1.0, jitter_frac=0.0
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=b"\xff\xff not protobuf"))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock)
            result = await recorder.run()  # must not raise

    stats = result.stats["vehicle_positions"]
    assert stats.errors >= 1
    assert stats.written == 0


@pytest.mark.asyncio
async def test_write_error_never_loses_spooled_data(make_source, vp_feed_dict, fake_clock, tmp_path, monkeypatch):
    """A failed flush (simulating disk-full / write error) must leave the
    spool's rows intact -- flush_spool_to_staging only deletes after a
    verified write, so a raising connect_fn must propagate without losing
    the already-spooled row."""
    source = make_source()
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=1.0, jitter_frac=0.0
    )

    def _boom_connect(base_uri, s3_config=None):
        raise OSError("No space left on device")

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=_vp_body(vp_feed_dict)))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock, connect_fn=_boom_connect)
            # Drive one poll directly, then attempt a flush that fails.
            from stoptimes.record.http import FeedCache

            await recorder.poll_once(feed, FeedCache())
            with pytest.raises(OSError):
                recorder.flush(FeedKind.VEHICLE_POSITIONS)

    spool = recorder._spool_for(FeedKind.VEHICLE_POSITIONS)
    assert spool.stats()["rows"] == 1  # row is still there -- nothing was lost


@pytest.mark.asyncio
async def test_sigterm_mid_flight_flushes_and_exits(make_source, vp_feed_dict, tmp_path):
    """Sending SIGTERM to the running process should stop the loop and flush
    whatever was spooled before the signal arrived. Uses the real clock (a
    tiny poll interval) since signal delivery needs a real event loop tick."""
    source = make_source()
    source.realtime_feeds[0].poll_interval_s = 0.05
    feed = source.realtime_feeds[0]
    config = RecorderConfig(source=source, spool_dir=tmp_path / "spool", jitter_frac=0.0)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=_vp_body(vp_feed_dict)))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client)
            run_task = asyncio.ensure_future(recorder.run())
            await asyncio.sleep(0.15)  # let at least one poll happen
            import os

            os.kill(os.getpid(), signal.SIGTERM)
            result = await asyncio.wait_for(run_task, timeout=5.0)

    assert result.stopped_reason == "signal"
    archive_root = tmp_path / "archive"
    written_files = list(archive_root.rglob("*.parquet"))
    assert len(written_files) == 1


@pytest.mark.asyncio
async def test_s3_writer_called_with_source_scoped_keys(make_source, vp_feed_dict, fake_clock, tmp_path):
    """The recorder must build its staging base_uri from the parameterized
    layout (schema.staging_prefix), not a hardcoded 'mbta' path -- verified
    by mocking connect_fn and inspecting the base_uri flush_spool_to_staging
    receives (the S3 writer's own COPY call is exercised elsewhere; this
    checks the *keys* the recorder asks it to write to)."""
    source = make_source(archive_kind=ArchiveKind.S3, archive_uri="s3://test-bucket/prefix")
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=1.0, jitter_frac=0.0
    )

    calls: list[str] = []

    class _FakeCon:
        def register(self, *a, **k):
            pass

        def unregister(self, *a, **k):
            pass

        def execute(self, sql, *a, **k):
            class _R:
                def fetchone(self_inner):
                    return (1,)

            return _R()

        def close(self):
            pass

    def _fake_connect(base_uri, s3_config=None):
        calls.append(base_uri)
        return _FakeCon()

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=_vp_body(vp_feed_dict)))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock, connect_fn=_fake_connect)
            await recorder.run()

    assert calls, "connect_fn was never called -- no flush happened"
    assert calls[0] == "s3://test-bucket/prefix/feeds/testagency/vehicle_positions_staging"


@pytest.mark.asyncio
async def test_dry_run_never_writes_spool_or_archive(make_source, vp_feed_dict, fake_clock, tmp_path):
    source = make_source()
    feed = source.realtime_feeds[0]
    config = RecorderConfig(
        source=source, spool_dir=tmp_path / "spool", duration_s=1.0, jitter_frac=0.0, dry_run=True
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.get(feed.url).mock(return_value=httpx.Response(200, content=_vp_body(vp_feed_dict)))
        async with httpx.AsyncClient() as client:
            recorder = Recorder(config, http_client=client, clock=fake_clock)
            result = await recorder.run()

    assert result.stats["vehicle_positions"].written == 0
    assert not (tmp_path / "spool").exists() or not list((tmp_path / "spool").glob("*.duckdb"))
    assert not (tmp_path / "archive").exists()


def test_spool_path_for_is_source_scoped(tmp_path):
    vp_path = spool_path_for(tmp_path, "mbta", FeedKind.VEHICLE_POSITIONS)
    tu_path = spool_path_for(tmp_path, "bucr", FeedKind.TRIP_UPDATES)
    assert vp_path.endswith("mbta_vp_spool.duckdb")
    assert tu_path.endswith("bucr_tu_spool.duckdb")
