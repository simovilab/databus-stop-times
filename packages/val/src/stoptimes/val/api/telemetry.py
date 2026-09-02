"""Process-wide telemetry collector — the "no silent failures" seam.

The engine tick loop swallows predictor/detector exceptions by design (a
single flaky predictor must not take down a whole run).  This module gives
operators visibility into what would otherwise vanish silently: tick health,
feed poll health, per-predictor exception counts, and a rolling log of
structured events.

Thread-safety
-------------
A single :class:`threading.Lock` guards all mutation.  ``RunManager`` calls
into this from asyncio callbacks (single-threaded event loop) so contention
is minimal, but the lock keeps the collector safe to use from any thread
(e.g. a future background poller) without further changes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

EventLevel = Literal["info", "warn", "error"]


@dataclass
class _FeedHealth:
    """Last feed-poll outcome for one run."""

    last_poll_at: float
    ok: bool


@dataclass
class _RunTelemetry:
    """Mutable per-run telemetry state."""

    started_at: float
    tick_count: int = 0
    last_tick_at: float | None = None
    feed: _FeedHealth | None = None
    predictor_errors: dict[str, int] = field(default_factory=dict)


class TelemetryCollector:
    """Thread-safe, process-wide collector of run health signals.

    Instantiate one shared collector (e.g. a module-level singleton, or one
    owned by ``RunManager``) and pass it wherever ticks / feed polls /
    predictor errors occur.
    """

    def __init__(self, *, event_capacity: int = 500) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, _RunTelemetry] = {}
        self._events: deque[dict] = deque(maxlen=event_capacity)
        self.api_started_at: float = time.time()

    # ── Recording ────────────────────────────────────────────────────────────

    def _get_or_create(self, run_id: str) -> _RunTelemetry:
        rt = self._runs.get(run_id)
        if rt is None:
            rt = _RunTelemetry(started_at=time.time())
            self._runs[run_id] = rt
        return rt

    def record_tick(self, run_id: str) -> None:
        """Record that *run_id* produced a tick just now."""
        with self._lock:
            rt = self._get_or_create(run_id)
            rt.tick_count += 1
            rt.last_tick_at = time.time()

    def record_feed_poll(self, run_id: str, ok: bool) -> None:
        """Record the outcome of a feed poll for *run_id*."""
        with self._lock:
            rt = self._get_or_create(run_id)
            rt.feed = _FeedHealth(last_poll_at=time.time(), ok=ok)

    def record_predictor_error(self, run_id: str, predictor_id: str) -> None:
        """Increment the exception counter for *predictor_id* under *run_id*."""
        with self._lock:
            rt = self._get_or_create(run_id)
            rt.predictor_errors[predictor_id] = rt.predictor_errors.get(predictor_id, 0) + 1

    def log_event(
        self,
        level: EventLevel,
        source: str,
        message: str,
        run_id: str | None = None,
    ) -> None:
        """Append a structured event to the rolling ring buffer.

        Oldest events are dropped once ``event_capacity`` is exceeded.
        """
        with self._lock:
            self._events.append(
                {
                    "ts": time.time(),
                    "run_id": run_id,
                    "level": level,
                    "source": source,
                    "message": message,
                }
            )

    # ── Queries ──────────────────────────────────────────────────────────────

    def events(self, limit: int = 100) -> list[dict]:
        """Return up to *limit* most-recent events, oldest-first (newest-last)."""
        with self._lock:
            items = list(self._events)
        if limit < len(items):
            items = items[-limit:]
        return items

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all tracked run telemetry.

        Shape::

            {
                "api_uptime_s": float,
                "per_run": {
                    run_id: {
                        "ticks": int,
                        "tick_rate": float,
                        "last_tick_age_s": float | None,
                        "feed": {"last_poll_age_s": float, "ok": bool} | None,
                        "predictor_errors": {predictor_id: count},
                    },
                    ...
                },
            }
        """
        now = time.time()
        with self._lock:
            per_run: dict[str, dict] = {}
            for run_id, rt in self._runs.items():
                elapsed = max(now - rt.started_at, 1e-9)
                tick_rate = rt.tick_count / elapsed
                last_tick_age = None if rt.last_tick_at is None else now - rt.last_tick_at
                feed_dict = None
                if rt.feed is not None:
                    feed_dict = {
                        "last_poll_age_s": now - rt.feed.last_poll_at,
                        "ok": rt.feed.ok,
                    }
                per_run[run_id] = {
                    "ticks": rt.tick_count,
                    "tick_rate": tick_rate,
                    "last_tick_age_s": last_tick_age,
                    "feed": feed_dict,
                    "predictor_errors": dict(rt.predictor_errors),
                }
        return {
            "api_uptime_s": now - self.api_started_at,
            "per_run": per_run,
        }
