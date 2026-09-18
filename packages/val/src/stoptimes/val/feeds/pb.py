"""GTFS-RT protobuf feed adapters.

``stoptimes.core`` ships the GTFS-RT *JSON* reader
(``stoptimes.core.realtime.gtfsrt_json``) plus a declared-but-unused
``stoptimes-core[pb]`` extra (``gtfs-realtime-bindings``) -- nothing in core
parses the wire format yet. ``val`` is the first consumer that needs
protobuf feeds (many real agencies only publish VehiclePositions/TripUpdates
as protobuf, never JSON), so this module owns the bridge.

Design: decode the wire-format ``FeedMessage``, convert it to the exact same
nested dict shape the JSON feed already produces
(``google.protobuf.json_format.MessageToDict``,
``preserving_proto_field_name=True`` so keys come out snake_case, matching
the JSON feed's own key names), then hand that dict to
``gtfsrt_json.parse_vp_feed`` / ``parse_tu_feed``. Past the wire format,
protobuf and JSON feeds run through the *identical* parsing code -- there is
no separate "protobuf record" type to keep in sync with the JSON one.

Importing ``gtfs_realtime_pb2`` is deferred to first use (``_gtfs_realtime_pb2``)
so this module -- and anything that imports it -- loads fine without the
optional ``pb`` extra installed; only actually parsing a protobuf feed raises,
with a clear install hint.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from stoptimes.core.realtime.gtfsrt_json import (
    TripUpdateRecord,
    TuPollResult,
    VehiclePositionRecord,
    VpPollResult,
    parse_tu_feed,
    parse_vp_feed,
)

FeedFormat = Literal["json", "pb", "auto"]


def _gtfs_realtime_pb2() -> Any:
    """Import and return the ``gtfs_realtime_pb2`` module, or raise with an install hint."""
    try:
        from google.transit import gtfs_realtime_pb2
    except ImportError as exc:  # pragma: no cover - exercised by test_feeds_pb's missing-extra case
        raise ImportError(
            "Parsing GTFS-RT protobuf feeds requires the 'pb' extra: install "
            "'stoptimes-core[pb]' (pulls gtfs-realtime-bindings)."
        ) from exc
    return gtfs_realtime_pb2


def feed_message_to_dict(data: bytes) -> dict[str, Any]:
    """Decode a serialized GTFS-RT ``FeedMessage`` into the JSON-shaped dict
    that ``gtfsrt_json.parse_vp_feed`` / ``parse_tu_feed`` already read.

    Raises:
        ImportError: the optional ``pb`` extra isn't installed.
        google.protobuf.message.DecodeError: *data* isn't a valid FeedMessage.
    """
    from google.protobuf.json_format import MessageToDict

    gtfs_realtime_pb2 = _gtfs_realtime_pb2()
    message = gtfs_realtime_pb2.FeedMessage()
    message.ParseFromString(data)
    return MessageToDict(message, preserving_proto_field_name=True)


def parse_vp_feed_pb(data: bytes) -> tuple[float, list[VehiclePositionRecord]]:
    """Parse a serialized VehiclePositions ``FeedMessage``. Same return shape as ``parse_vp_feed``."""
    return parse_vp_feed(feed_message_to_dict(data))


def parse_tu_feed_pb(data: bytes) -> tuple[float, list[TripUpdateRecord]]:
    """Parse a serialized TripUpdates ``FeedMessage``. Same return shape as ``parse_tu_feed``."""
    return parse_tu_feed(feed_message_to_dict(data))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def sniff_format(content_type: str | None, body: bytes) -> Literal["json", "pb"]:
    """Guess whether *body* is a GTFS-RT JSON or protobuf feed.

    1. ``content_type`` (the response's Content-Type header), if it names one
       unambiguously (``application/json`` -> json; ``application/x-protobuf``,
       ``application/protobuf``, or ``application/octet-stream`` -> pb).
    2. Otherwise the body itself: GTFS-RT JSON always starts (after leading
       whitespace) with ``{``; protobuf's wire format never does, so anything
       else is treated as protobuf.
    """
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if "json" in ct:
            return "json"
        if "protobuf" in ct or "octet-stream" in ct:
            return "pb"
    if body.lstrip()[:1] == b"{":
        return "json"
    return "pb"


# ---------------------------------------------------------------------------
# Async pollers -- FeedAdapter-compatible (same `.poll()` interface as
# stoptimes.core.realtime.gtfsrt_json's JSON-only pollers)
# ---------------------------------------------------------------------------


class VehiclePositionsPbPoller:
    """Async GTFS-RT VehiclePositions poller for a feed always served as protobuf."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}

    async def poll(self) -> VpPollResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            header_ts, records = parse_vp_feed_pb(response.content)
        return VpPollResult(header_timestamp=header_ts, records=records)


class TripUpdatesPbPoller:
    """Async GTFS-RT TripUpdates poller for a feed always served as protobuf."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}

    async def poll(self) -> TuPollResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            header_ts, records = parse_tu_feed_pb(response.content)
        return TuPollResult(header_timestamp=header_ts, records=records)


class AutoVehiclePositionsPoller:
    """VehiclePositions poller that picks JSON or protobuf per response.

    ``format``: ``"json"`` | ``"pb"`` | ``"auto"`` (default). ``"auto"``
    sniffs the Content-Type header, falling back to the response body, on
    every poll -- so a source declared without an explicit format (or one
    that changes format upstream) still works.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        format: FeedFormat = "auto",
    ) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}
        self._format: FeedFormat = format

    async def poll(self) -> VpPollResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            fmt = self._format
            if fmt == "auto":
                fmt = sniff_format(response.headers.get("content-type"), response.content)
            if fmt == "pb":
                header_ts, records = parse_vp_feed_pb(response.content)
            else:
                header_ts, records = parse_vp_feed(response.json())
        return VpPollResult(header_timestamp=header_ts, records=records)


class AutoTripUpdatesPoller:
    """TripUpdates poller that picks JSON or protobuf per response. See ``AutoVehiclePositionsPoller``."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        format: FeedFormat = "auto",
    ) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}
        self._format: FeedFormat = format

    async def poll(self) -> TuPollResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            fmt = self._format
            if fmt == "auto":
                fmt = sniff_format(response.headers.get("content-type"), response.content)
            if fmt == "pb":
                header_ts, records = parse_tu_feed_pb(response.content)
            else:
                header_ts, records = parse_tu_feed(response.json())
        return TuPollResult(header_timestamp=header_ts, records=records)
