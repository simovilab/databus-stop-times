"""Tests for stoptimes.record.http: conditional GET, retry/backoff, dedup."""

from __future__ import annotations

import httpx
import pytest
import respx
from stoptimes.core.contracts.source import FeedFormat, FeedKind, RealtimeFeed
from stoptimes.record.http import FeedCache, fetch_feed


def _feed(url: str = "http://example.test/vp.json") -> RealtimeFeed:
    return RealtimeFeed(kind=FeedKind.VEHICLE_POSITIONS, url=url, format=FeedFormat.JSON)


@pytest.mark.asyncio
async def test_unchanged_payload_is_skipped(fake_clock):
    """Same body twice (no ETag support) -> second poll reports 'unchanged'."""
    feed = _feed()
    cache = FeedCache()
    body = b'{"header": {"timestamp": 1}, "entity": []}'

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(feed.url).mock(return_value=httpx.Response(200, content=body))
            first = await fetch_feed(client, feed, cache, clock=fake_clock)
            second = await fetch_feed(client, feed, cache, clock=fake_clock)

    assert first.status == "ok"
    assert first.payload == body
    assert second.status == "unchanged"
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_etag_304_is_not_modified(fake_clock):
    feed = _feed()
    cache = FeedCache()

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(feed.url).mock(
                side_effect=[
                    httpx.Response(200, content=b"{}", headers={"ETag": '"v1"'}),
                    httpx.Response(304),
                ]
            )
            first = await fetch_feed(client, feed, cache, clock=fake_clock)
            second = await fetch_feed(client, feed, cache, clock=fake_clock)

    assert first.status == "ok"
    assert cache.etag == '"v1"'
    assert second.status == "not_modified"


@pytest.mark.asyncio
async def test_500_then_success_retries_with_backoff(fake_clock):
    feed = _feed()
    cache = FeedCache()
    body = b'{"header": {"timestamp": 1}, "entity": []}'

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(feed.url).mock(
                side_effect=[httpx.Response(500), httpx.Response(200, content=body)]
            )
            result = await fetch_feed(
                client, feed, cache, clock=fake_clock, max_retries=3, backoff_base_s=1.0
            )

    assert result.status == "ok"
    assert result.payload == body
    assert result.attempts == 2
    assert fake_clock.t == pytest.approx(1.0)  # one backoff sleep before the retry


@pytest.mark.asyncio
async def test_timeout_then_success(fake_clock):
    feed = _feed()
    cache = FeedCache()
    body = b'{"header": {"timestamp": 1}, "entity": []}'

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(feed.url).mock(
                side_effect=[httpx.TimeoutException("timed out"), httpx.Response(200, content=body)]
            )
            result = await fetch_feed(client, feed, cache, clock=fake_clock, max_retries=3)

    assert result.status == "ok"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_exhausted_retries_report_error(fake_clock):
    feed = _feed()
    cache = FeedCache()

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(feed.url).mock(return_value=httpx.Response(503))
            result = await fetch_feed(client, feed, cache, clock=fake_clock, max_retries=2)

    assert result.status == "error"
    assert "503" in (result.error or "")
    assert result.attempts == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_auth_env_header_is_resolved(fake_clock, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    feed = RealtimeFeed(
        kind=FeedKind.VEHICLE_POSITIONS,
        url="http://example.test/vp.json",
        format=FeedFormat.JSON,
        auth_env={"x-api-key": "TEST_API_KEY"},
    )
    cache = FeedCache()

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(feed.url).mock(return_value=httpx.Response(200, content=b"{}"))
            await fetch_feed(client, feed, cache, clock=fake_clock)

    sent_headers = route.calls[0].request.headers
    assert sent_headers["x-api-key"] == "secret-value"


@pytest.mark.asyncio
async def test_missing_auth_env_omits_header(fake_clock, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    feed = RealtimeFeed(
        kind=FeedKind.VEHICLE_POSITIONS,
        url="http://example.test/vp.json",
        format=FeedFormat.JSON,
        auth_env={"x-api-key": "MISSING_KEY"},
    )
    cache = FeedCache()

    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(feed.url).mock(return_value=httpx.Response(200, content=b"{}"))
            await fetch_feed(client, feed, cache, clock=fake_clock)

    assert "x-api-key" not in route.calls[0].request.headers
