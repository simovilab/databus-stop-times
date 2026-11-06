"""A live run survives a transient feed outage (DNS/network blip, timeout, 5xx) instead of failing on
the first failed fetch; it fails only once the feed stays unreachable past `max_outage_s`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from stoptimes.val.engine.runtime import LiveFeedSource, is_transient_feed_error


class _Poller:
    def __init__(self, failures: list[BaseException], header_timestamp: float = 100.0) -> None:
        self._failures = list(failures)
        self._ts = header_timestamp
        self.calls = 0

    async def poll(self):
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return SimpleNamespace(header_timestamp=self._ts, records=[])


def _dns_error() -> httpx.ConnectError:
    return httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://feed.example/vp.pb")
    return httpx.HTTPStatusError("bad status", request=request, response=httpx.Response(code, request=request))


@pytest.mark.parametrize(
    "exc, transient",
    [
        (_dns_error(), True),
        (httpx.ReadTimeout("slow"), True),
        (asyncio.TimeoutError(), True),
        (_status_error(503), True),
        (_status_error(429), True),
        (_status_error(404), False),
        (ValueError("malformed feed"), False),
    ],
)
def test_transient_classification(exc, transient):
    assert is_transient_feed_error(exc) is transient


def test_transient_failures_are_retried_until_the_feed_comes_back():
    vp = _Poller([_dns_error(), _dns_error()])
    tu = _Poller([])
    source = LiveFeedSource(vp, tu, retry_base_s=0.0, max_outage_s=60.0)
    snapshot = asyncio.run(source.__anext__())
    assert snapshot.header_timestamp == 100.0
    assert vp.calls == 3
    assert source.feed_errors == 2
    assert "ConnectError" in source.last_error


def test_run_fails_once_the_outage_exceeds_the_limit():
    vp = _Poller([_dns_error()] * 10)
    source = LiveFeedSource(vp, _Poller([]), retry_base_s=0.0, max_outage_s=0.0)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(source.__anext__())


def test_non_transient_errors_fail_immediately():
    vp = _Poller([_status_error(404)])
    source = LiveFeedSource(vp, _Poller([]), retry_base_s=0.0)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(source.__anext__())
    assert vp.calls == 1
