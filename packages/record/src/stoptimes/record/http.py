"""Async HTTP polling: conditional GETs, retry with backoff, an injectable clock.

Kept separate from ``recorder.py`` because it is the part with real I/O and
timing concerns; everything here takes an ``httpx.AsyncClient`` and a
:class:`Clock` as parameters instead of constructing its own, so tests can
swap in ``respx``-mocked transports and an instant fake clock (no real
sleeping, no flaky timing-dependent tests).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx
from stoptimes.core.contracts.source import RealtimeFeed


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock ``Clock`` used in production."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)


@dataclass
class FeedCache:
    """Per-feed conditional-request + dedup state, carried across polls."""

    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_hash: Optional[str] = None


@dataclass(frozen=True)
class PollResult:
    """Outcome of one :func:`fetch_feed` call."""

    status: str  # "ok" | "not_modified" | "unchanged" | "error"
    payload: Optional[bytes] = None
    error: Optional[str] = None
    attempts: int = 1


def _auth_headers(feed: RealtimeFeed) -> dict[str, str]:
    """Resolve ``feed.auth_env`` (header name -> env var NAME) into real header values.

    Never logs or persists the resolved value; a missing env var silently
    omits that header rather than sending a literal env-var name as a secret.
    """
    headers: dict[str, str] = dict(feed.headers)
    for header_name, env_name in feed.auth_env.items():
        value = os.environ.get(env_name)
        if value:
            headers[header_name] = value
    return headers


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= 500


async def fetch_feed(
    client: httpx.AsyncClient,
    feed: RealtimeFeed,
    cache: FeedCache,
    *,
    clock: Clock,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    timeout_s: float = 10.0,
) -> PollResult:
    """GET ``feed.url`` with conditional headers, retrying 5xx/timeouts.

    On success, updates ``cache`` in place (ETag / Last-Modified / content
    hash) and returns ``status="ok"`` with the body, ``status="not_modified"``
    on a 304, or ``status="unchanged"`` when the server returned 200 but the
    body hash matches the previous poll (some feeds don't support conditional
    requests at all). ``status="error"`` after exhausting retries -- the
    caller decides what that means for the spool (nothing: a failed poll
    writes nothing and the spool is untouched).
    """
    headers = _auth_headers(feed)
    if cache.etag:
        headers["If-None-Match"] = cache.etag
    if cache.last_modified:
        headers["If-Modified-Since"] = cache.last_modified

    attempt = 0
    last_error = ""
    while attempt < max(1, max_retries + 1):
        attempt += 1
        try:
            response = await client.get(feed.url, headers=headers, timeout=timeout_s)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt > max_retries:
                break
            await clock.sleep(backoff_base_s * (2 ** (attempt - 1)))
            continue

        if response.status_code == 304:
            return PollResult(status="not_modified", attempts=attempt)

        if _is_retryable_status(response.status_code):
            last_error = f"HTTP {response.status_code}"
            if attempt > max_retries:
                break
            await clock.sleep(backoff_base_s * (2 ** (attempt - 1)))
            continue

        if response.status_code >= 400:
            return PollResult(
                status="error", error=f"HTTP {response.status_code}", attempts=attempt
            )

        body = response.content
        digest = hashlib.sha256(body).hexdigest()
        cache.etag = response.headers.get("ETag", cache.etag)
        cache.last_modified = response.headers.get("Last-Modified", cache.last_modified)
        if cache.content_hash == digest:
            return PollResult(status="unchanged", attempts=attempt)
        cache.content_hash = digest
        return PollResult(status="ok", payload=body, attempts=attempt)

    return PollResult(status="error", error=last_error or "unknown error", attempts=attempt)
