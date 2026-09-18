"""Shared fixtures for the recorder test suite.

``FakeClock`` makes the poll loop, its jittered intervals, and its
``--duration`` cutoff deterministic and instant: ``sleep`` advances a virtual
clock and yields control once (``await asyncio.sleep(0)``) instead of
actually waiting, so a test exercising a multi-hour flush interval runs in
milliseconds.
"""

from __future__ import annotations

import asyncio

import pytest
from stoptimes.core.contracts.source import (
    Archive,
    ArchiveKind,
    FeedFormat,
    FeedKind,
    RealtimeFeed,
    Source,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0)
        await asyncio.sleep(0)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def vp_feed_dict() -> dict:
    return {
        "header": {"timestamp": 1000},
        "entity": [
            {
                "id": "1",
                "vehicle": {
                    "trip": {"trip_id": "t1", "route_id": "r1"},
                    "vehicle": {"id": "v1"},
                    "position": {
                        "latitude": 1.0,
                        "longitude": 2.0,
                        "bearing": 90,
                        "speed": 5,
                    },
                    "current_status": "IN_TRANSIT_TO",
                    "current_stop_sequence": 3,
                    "timestamp": 1000,
                },
            }
        ],
    }


@pytest.fixture
def make_source(tmp_path):
    def _make(*, archive_kind: ArchiveKind = ArchiveKind.LOCAL, archive_uri: str | None = None,
               feeds: list[RealtimeFeed] | None = None) -> Source:
        if feeds is None:
            feeds = [
                RealtimeFeed(
                    kind=FeedKind.VEHICLE_POSITIONS,
                    url="http://example.test/vp.json",
                    format=FeedFormat.JSON,
                    poll_interval_s=5.0,
                )
            ]
        archive = Archive(
            kind=archive_kind, uri=archive_uri or str(tmp_path / "archive")
        )
        return Source(
            id="testagency",
            name="Test Agency",
            gtfs_static_url="http://example.test/gtfs.zip",
            realtime_feeds=feeds,
            archive=archive,
        )

    return _make
