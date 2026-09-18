"""Standalone GTFS-RT recorder (no Django/Celery): poll -> spool -> flush.

Phase 2, track T5 (docs/PLAN.md). Replaces the Django/Celery collector
(``legacy/collector/``, read-only reference) with an async poll loop over a
``stoptimes.core.contracts.source.Source``'s realtime feeds, writing into the
existing DuckDB spool (``storage/spool.py``, ``storage/tu_spool.py``) and
flushing to the existing Parquet staging layout (``storage/schema.py``,
``storage/tu_schema.py``) via the existing writer machinery
(``storage/spool_base.flush_spool_to_staging``).

Design:

* One asyncio task per configured feed (VP, TU), each on its own jittered
  interval, so a slow/erroring feed never delays another.
* Conditional GETs (ETag / If-Modified-Since) and a body-hash fallback
  (``http.fetch_feed``) mean an unchanged payload is never re-spooled.
* 5xx / timeout errors are retried with backoff inside one poll attempt
  (``http.fetch_feed``); a poll that still fails after retries logs and
  moves on -- it never touches the spool, so a transient outage cannot lose
  already-spooled data.
* SIGINT/SIGTERM set a stop event; the run loop finishes its current poll
  (never half-writes a row), does a final flush of every spool with pending
  rows, and returns. Only an unhandled exception exits non-zero.
* Time is read through an injected :class:`stoptimes.record.http.Clock`
  (``RealClock`` in production) so the loop, its jittered intervals, and its
  ``--duration`` cutoff are all testable without real sleeping.
"""

from __future__ import annotations

import asyncio
import random
import signal
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from stoptimes.core.contracts.source import (
    ArchiveKind,
    FeedFormat,
    FeedKind,
    RealtimeFeed,
    Source,
)
from stoptimes.core.logging import get_logger

from . import parsing, status
from .http import Clock, FeedCache, PollResult, RealClock, fetch_feed
from .storage import schema as vp_schema
from .storage import tu_schema
from .storage.spool import Spool
from .storage.spool_base import flush_spool_to_staging
from .storage.tu_spool import TripUpdateSpool

log = get_logger("record.recorder")

DEFAULT_FLUSH_INTERVAL_S = 3600.0  # hourly, matches the legacy collector


@dataclass
class RecorderConfig:
    """Everything one ``record start`` invocation needs."""

    source: Source
    spool_dir: Path
    feed_kinds: Optional[list[FeedKind]] = None  # None = every configured feed
    archive_root: Optional[str] = None  # overrides source.archive.uri
    dry_run: bool = False
    duration_s: Optional[float] = None
    max_retries: int = 3
    backoff_base_s: float = 1.0
    jitter_frac: float = 0.2
    flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S
    status_dir: Optional[str] = None
    routes: Optional[frozenset[str]] = None
    max_tu_horizon_s: Optional[int] = None

    def feeds(self) -> list[RealtimeFeed]:
        wanted = set(self.feed_kinds) if self.feed_kinds is not None else None
        return [f for f in self.source.realtime_feeds if wanted is None or f.kind in wanted]


@dataclass
class FeedRunStats:
    polls: int = 0
    written: int = 0
    skipped_unchanged: int = 0
    errors: int = 0
    last_poll_utc: Optional[str] = None
    last_success_utc: Optional[str] = None
    last_error: Optional[str] = None


@dataclass
class RecorderResult:
    """What ``Recorder.run()`` hands back -- also the shape ``record start`` reports."""

    stats: dict[str, FeedRunStats] = field(default_factory=dict)
    stopped_reason: str = "duration_elapsed"  # or "signal", "no_feeds"
    ok: bool = True


def _archive_root(config: RecorderConfig) -> Optional[str]:
    if config.archive_root:
        return config.archive_root.rstrip("/")
    if config.source.archive is None:
        return None
    return config.source.archive.uri.rstrip("/")


def _base_uri(config: RecorderConfig, kind: FeedKind) -> Optional[str]:
    """Staging base URI for one feed kind, using the parameterized layout
    (``storage.schema.staging_prefix`` / ``storage.tu_schema.staging_prefix``),
    which defaults to today's ``feeds/mbta/...`` layout for ``source.id == "mbta"``.
    """
    root = _archive_root(config)
    if root is None:
        return None
    source_id = config.source.id
    if kind == FeedKind.VEHICLE_POSITIONS:
        return f"{root}/{vp_schema.staging_prefix(source_id)}"
    if kind == FeedKind.TRIP_UPDATES:
        return f"{root}/{tu_schema.staging_prefix(source_id)}"
    raise ValueError(f"no archive layout for feed kind: {kind}")


def spool_path_for(spool_dir: Path, source_id: str, kind: FeedKind) -> str:
    """Local DuckDB spool file path for one (source, feed kind).

    Public so ``record status`` can locate the same spool files the recorder
    writes without duplicating the naming convention.
    """
    name = "vp_spool.duckdb" if kind == FeedKind.VEHICLE_POSITIONS else "tu_spool.duckdb"
    return str(Path(spool_dir) / f"{source_id}_{name}")


def _spool_path(config: RecorderConfig, kind: FeedKind) -> str:
    return spool_path_for(config.spool_dir, config.source.id, kind)


class Recorder:
    """Owns the poll/spool/flush loop for one :class:`Source`."""

    def __init__(
        self,
        config: RecorderConfig,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        clock: Optional[Clock] = None,
        rng: Optional[random.Random] = None,
        connect_fn: Optional[Callable[..., Any]] = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config
        self._client = http_client
        self._owns_client = http_client is None
        self.clock: Clock = clock or RealClock()
        self.rng = rng or random.Random()
        self._now_fn = now_fn
        self._connect_fn = connect_fn
        self._spools: dict[FeedKind, Any] = {}
        self._stats: dict[str, FeedRunStats] = {}

    # --- spool access --------------------------------------------------

    def _spool_for(self, kind: FeedKind):
        if kind not in self._spools:
            path = _spool_path(self.config, kind)
            if kind == FeedKind.VEHICLE_POSITIONS:
                self._spools[kind] = Spool(path)
            elif kind == FeedKind.TRIP_UPDATES:
                self._spools[kind] = TripUpdateSpool(path)
            else:
                raise ValueError(f"unsupported feed kind for spooling: {kind}")
        return self._spools[kind]

    # --- one poll --------------------------------------------------------

    async def poll_once(self, feed: RealtimeFeed, cache: FeedCache) -> PollResult:
        """Fetch, parse, and (unless ``dry_run``) spool one feed's current payload."""
        client = self._client
        assert client is not None, "poll_once requires an active http client"

        stats = self._stats.setdefault(feed.kind.value, FeedRunStats())
        stats.polls += 1
        stats.last_poll_utc = self._now_fn().isoformat()

        result = await fetch_feed(
            client,
            feed,
            cache,
            clock=self.clock,
            max_retries=self.config.max_retries,
            backoff_base_s=self.config.backoff_base_s,
        )

        if result.status in ("not_modified", "unchanged"):
            stats.skipped_unchanged += 1
            stats.last_success_utc = self._now_fn().isoformat()
            return result

        if result.status == "error":
            stats.errors += 1
            stats.last_error = result.error
            log.warning(
                "poll failed", feed=feed.kind.value, url=feed.url, error=result.error
            )
            return result

        # status == "ok"
        ingested_at = self._now_fn()
        try:
            feed_msg = parsing.parse_payload(result.payload or b"", feed.format)
            if feed.kind == FeedKind.VEHICLE_POSITIONS:
                rows = parsing.vp_rows(
                    feed_msg, feed.format, feed_name=self.config.source.id, ingested_at=ingested_at
                )
            elif feed.kind == FeedKind.TRIP_UPDATES:
                rows = parsing.tu_rows(
                    feed_msg,
                    feed.format,
                    feed_name=self.config.source.id,
                    ingested_at=ingested_at,
                    routes=self.config.routes,
                    max_horizon_seconds=self.config.max_tu_horizon_s,
                )
            else:
                rows = []
        except parsing.FeedParseError as exc:
            stats.errors += 1
            stats.last_error = str(exc)
            log.error("malformed feed payload", feed=feed.kind.value, error=str(exc))
            return PollResult(status="error", error=str(exc), attempts=result.attempts)

        if rows and not self.config.dry_run:
            import pandas as pd

            df = pd.DataFrame.from_records(rows)
            written = self._spool_for(feed.kind).append(df)
            stats.written += written

        stats.last_success_utc = self._now_fn().isoformat()
        return result

    # --- flush -----------------------------------------------------------

    def flush(self, kind: FeedKind) -> dict[str, Any]:
        """Drain everything currently in ``kind``'s spool to staging.

        A no-op (returns ``{"flushed": 0, ...}``) when the spool was never
        opened (nothing has been polled for this kind yet) or when this
        source has no configured archive -- ``record start --dry-run`` never
        touches storage.
        """
        if kind not in self._spools:
            return {"flushed": 0, "days": [], "written": 0}
        base_uri = _base_uri(self.config, kind)
        if base_uri is None:
            return {"flushed": 0, "days": [], "written": 0, "skipped": "no archive configured"}

        spool = self._spools[kind]
        cutoff = self._now_fn()
        connect_fn = self._connect_fn
        if connect_fn is None:
            from .storage.s3_writer import connect as connect_fn  # local import: default wiring

        if not base_uri.startswith("s3://"):
            # DuckDB's local COPY does not create the target directory tree.
            Path(base_uri).mkdir(parents=True, exist_ok=True)

        # No config passed here even for an s3:// target: the default
        # connect_fn (storage.s3_writer.connect) resolves S3Config.from_env()
        # itself, lazily, only when it actually needs to open a connection --
        # so a caller-supplied connect_fn (tests) never needs real S3 env vars.
        return flush_spool_to_staging(
            spool, base_uri, cutoff, connect_fn=connect_fn, config=None
        )

    def flush_all(self) -> dict[str, Any]:
        return {kind.value: self.flush(kind) for kind in list(self._spools)}

    # --- run loop ----------------------------------------------------------

    async def run(self) -> RecorderResult:
        feeds = self.config.feeds()
        if not feeds:
            return RecorderResult(stopped_reason="no_feeds", ok=True)

        stop_event = asyncio.Event()
        # A plain dict, not a local var + nonlocal: the signal handler and the
        # duration watchdog both write it BEFORE setting stop_event, so
        # whichever fires first has already recorded the reason by the time
        # anything else wakes up on the event -- no race between "who set the
        # event" and "who gets to say why".
        reason = {"value": "duration_elapsed"}
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []

        def _on_signal() -> None:
            reason["value"] = "signal"
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_signal)
                installed.append(sig)
            except (NotImplementedError, RuntimeError):
                pass  # platform without signal-handler support (e.g. some test harnesses)

        owns_client = self._client is None
        if owns_client:
            self._client = httpx.AsyncClient()

        try:
            start = self.clock.now()

            async def watchdog() -> None:
                if self.config.duration_s is None:
                    await stop_event.wait()
                    return
                while not stop_event.is_set():
                    remaining = self.config.duration_s - (self.clock.now() - start)
                    if remaining <= 0:
                        reason["value"] = "duration_elapsed"
                        stop_event.set()
                        return
                    await self._race_sleep(min(remaining, 1.0), stop_event)

            watchdog_task = asyncio.create_task(watchdog())
            feed_tasks = [
                asyncio.create_task(self._feed_loop(feed, stop_event)) for feed in feeds
            ]

            await stop_event.wait()
            watchdog_task.cancel()
            for t in feed_tasks:
                t.cancel()
            await asyncio.gather(watchdog_task, *feed_tasks, return_exceptions=True)
        finally:
            for sig in installed:
                loop.remove_signal_handler(sig)
            if owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None

        flush_report = self.flush_all() if not self.config.dry_run else {}
        return RecorderResult(stats=dict(self._stats), stopped_reason=reason["value"], ok=True)

    async def _race_sleep(self, seconds: float, stop_event: asyncio.Event) -> None:
        sleep_task = asyncio.create_task(self.clock.sleep(seconds))
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            await asyncio.wait({sleep_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (sleep_task, stop_task):
                if not t.done():
                    t.cancel()

    async def _feed_loop(self, feed: RealtimeFeed, stop_event: asyncio.Event) -> None:
        cache = FeedCache()
        last_flush = self.clock.now()
        while not stop_event.is_set():
            try:
                await self.poll_once(feed, cache)
            except Exception:  # never let one feed's crash kill the loop
                log.error("poll_once raised", feed=feed.kind.value, exc_info=True)

            if not self.config.dry_run and (
                self.clock.now() - last_flush >= self.config.flush_interval_s
            ):
                try:
                    self.flush(feed.kind)
                except Exception:
                    log.error("periodic flush failed", feed=feed.kind.value, exc_info=True)
                last_flush = self.clock.now()

            interval = feed.poll_interval_s
            jitter = interval * self.rng.uniform(-self.config.jitter_frac, self.config.jitter_frac)
            await self._race_sleep(max(0.0, interval + jitter), stop_event)


def build_recorder(
    source: Source,
    *,
    spool_dir: Path,
    feed_kinds: Optional[list[FeedKind]] = None,
    archive_root: Optional[str] = None,
    dry_run: bool = False,
    duration_s: Optional[float] = None,
    status_dir: Optional[str] = None,
    **overrides: Any,
) -> Recorder:
    """Convenience constructor used by the CLI."""
    config = RecorderConfig(
        source=source,
        spool_dir=spool_dir,
        feed_kinds=feed_kinds,
        archive_root=archive_root,
        dry_run=dry_run,
        duration_s=duration_s,
        status_dir=status_dir,
        **overrides,
    )
    return Recorder(config)


async def run_recorder(recorder: Recorder) -> RecorderResult:
    """Run to completion, writing a status snapshot before and after.

    Split from ``Recorder.run`` so the CLI's status side-effect isn't forced
    onto every test that only cares about the poll loop.
    """
    name = recorder.config.source.id
    status_dir = recorder.config.status_dir
    run_id = uuid.uuid4().hex[:8]
    status.update(name, status_dir, event=True, state="starting", run_id=run_id)
    try:
        result = await recorder.run()
    except Exception as exc:
        status.update(name, status_dir, event=True, state="crashed", error=str(exc))
        raise
    status.update(
        name,
        status_dir,
        event=True,
        state="stopped",
        stopped_reason=result.stopped_reason,
        stats={k: vars(v) for k, v in result.stats.items()},
    )
    return result
