"""LiveSession: the per-run state behind live-feature serving.

One session lives as long as a *run* (the serve loop, or one `val` replay), not one tick.
It owns what must survive across ticks and what is worth caching:

* a :class:`~stoptimes.core.features.trip_state.TripStateStore` (in-memory by default) with one
  :class:`~stoptimes.core.features.trip_state.TripState` per vehicle-trip;
* a ``TripContext`` + ``ForwardMatcher`` per ``(trip_id, service_date)``, built once from the
  static feed (and a shape/stop-progress cache per trip_id, shared across service dates);
* the artifact's sidecar assets (segment statistics, route vocabulary), time zone and holiday
  region -- always the artifact's, never the serving source's, so the temporal features are
  what the model was trained with.

Per tick the caller feeds every new ping with :meth:`LiveSession.observe` (in arrival order)
and asks for predictions at the tick instant ``t`` with :meth:`LiveSession.predict`, which
assembles features for all fresh vehicles with
:func:`~stoptimes.core.features.live.assemble_live_features` and calls the model **once** on
the concatenated rows.

Rules that are definitions, not tuning:

* *Staleness.* A vehicle is skipped at ``t`` when its latest valid ping is more than
  ``max_ping_age_s`` (30 s, the training bound) older than ``t``.
* *Causality.* A prediction at ``t`` uses only pings with epoch <= ``t`` (the assembly ignores
  later ones even if they are already in the state).
* *Service date.* From the VehiclePosition ``trip.start_date`` when it is a valid ``YYYYMMDD``;
  otherwise derived from the static schedule by :func:`derive_service_date`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.features.live import (
    LIVE_FEATURES,
    LiveAssets,
    TripContext,
    StaticTripSource,
    assemble_live_features,
    build_trip_context,
)
from stoptimes.core.features.schedule import resolve_timezone, scheduled_epochs
from stoptimes.core.features.trip_state import (
    ForwardMatcher,
    InMemoryTripStateStore,
    Ping,
    TripState,
    TripStateStore,
    update_trip_state,
)

MAX_PING_AGE_S: float = 30.0
"""Newest valid ping may be at most this old at prediction time (training: ``LiveFrameParams.max_ping_age_s``)."""

SERVICE_DATE_SLACK_S: float = 3 * 3600.0
"""A bound service date stays bound while the ping is within this slack of the trip's scheduled window."""


class LiveSessionError(ValueError):
    """A trip cannot be served live (not in the static feed, no usable shape, schedule mismatch)."""


def needs_live_session(features: Sequence[str]) -> bool:
    """True when any of ``features`` is a registry feature that declares live inputs."""
    from stoptimes.core.features.registry import get

    for name in features:
        try:
            if get(name).inputs:
                return True
        except LookupError:
            continue
    return False


def _valid_yyyymmdd(text: str | None) -> bool:
    if not text or len(text) != 8 or not text.isdigit():
        return False
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return False
    return True


def _dist_to_window(epoch: float, lo: float, hi: float) -> float:
    return 0.0 if lo <= epoch <= hi else min(abs(epoch - lo), abs(epoch - hi))


def derive_service_date(schedule: Any, ping_epoch: float, tz: str) -> str:
    """Service date of a trip observed at ``ping_epoch`` when the feed does not say.

    Rule (no hard-coded day boundary): the candidates are the agency-local calendar dates
    of the ping minus one day, that day and plus one day; for each, the trip's scheduled
    window ``[first, last]`` is anchored per GTFS (``scheduled_epochs``) and the candidate
    whose window is closest to the ping wins (distance 0 inside the window). Ties go to
    the later date. A 25:30 departure observed at 00:30 therefore belongs to the previous
    service date, a 00:10 departure observed at 00:05 to the current one.
    """
    from zoneinfo import ZoneInfo

    local = datetime.fromtimestamp(ping_epoch, tz=ZoneInfo(tz)).date()
    best: tuple[float, int, str] | None = None
    for k, delta in enumerate((-1, 0, 1)):
        day = (local + timedelta(days=delta)).strftime("%Y%m%d")
        sched = np.asarray(scheduled_epochs(schedule, day, tz), dtype=float)
        sched = sched[np.isfinite(sched)]
        if sched.size == 0:
            continue
        cand = (_dist_to_window(ping_epoch, float(sched.min()), float(sched.max())), -k, day)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise LiveSessionError(f"trip {schedule.trip_id!r} has no scheduled times to derive a service date from")
    return best[2]


@dataclass(frozen=True)
class VehicleSlot:
    """A vehicle-trip that is fresh at some ``t``: everything needed to describe its downstream stops."""

    key: Hashable
    ctx: TripContext
    state: TripState
    ping_epoch: float
    progress_m: float
    vehicle_id: str
    lat: float = math.nan  # last observed position (informational: identity of the Observation)
    lon: float = math.nan


class LiveSession:
    """Run-scoped live-feature state. See the module docstring."""

    def __init__(
        self,
        static: StaticTripSource,
        assets: LiveAssets,
        *,
        features: Sequence[str],
        tz: str,
        holiday_region: str | None = None,
        store: TripStateStore | None = None,
        max_ping_age_s: float = MAX_PING_AGE_S,
    ) -> None:
        unknown = [f for f in features if f not in LIVE_FEATURES]
        if unknown:
            raise LiveSessionError(f"features {unknown} cannot be computed by the live assembly")
        self._static = static
        self.assets = assets
        self.features: tuple[str, ...] = tuple(features)
        self.tz = tz
        self.holiday_region = holiday_region
        self.store: TripStateStore = store if store is not None else InMemoryTripStateStore()
        self.max_ping_age_s = float(max_ping_age_s)
        self._geometry_cache: dict[str, Any] = {}
        self._contexts: dict[tuple[str, str], tuple[TripContext, ForwardMatcher] | str] = {}
        self._schedule_cache: dict[str, Any] = {}
        self._bound_date: dict[tuple[str, str], str] = {}
        self._ctx_by_key: dict[Hashable, TripContext] = {}
        self._last_ping: dict[Hashable, float] = {}
        self._last_position: dict[Hashable, tuple[float, float]] = {}

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_artifact(
        cls,
        meta: Any,
        registry_dir: Path | str,
        static: StaticTripSource,
        *,
        fallback_timezone: str | None = None,
        fallback_holiday_region: str | None = None,
        store: TripStateStore | None = None,
        max_ping_age_s: float = MAX_PING_AGE_S,
    ) -> LiveSession:
        """Session for a registry artifact: assets (hash-verified), time zone and region from ``meta``.

        ``meta.timezone`` / ``meta.holiday_region`` win; only when the artifact predates them are
        the fallbacks (the agency's ``agency.txt`` zone, then ``fallback_timezone``) used.
        """
        from stoptimes.serve.model_loading import load_artifact_assets

        assets = load_artifact_assets(meta, registry_dir)
        agency_tz = static.agency_timezone() if hasattr(static, "agency_timezone") else None
        tz = resolve_timezone(meta.timezone, resolve_timezone(agency_tz, fallback_timezone))
        region = meta.holiday_region if meta.holiday_region is not None else fallback_holiday_region
        return cls(
            static,
            assets,
            features=list(meta.features),
            tz=tz,
            holiday_region=region,
            store=store,
            max_ping_age_s=max_ping_age_s,
        )

    # -- trip context ----------------------------------------------------------------

    def _schedule(self, trip_id: str) -> Any:
        sched = self._schedule_cache.get(trip_id)
        if sched is None:
            try:
                sched = self._static.trip_schedule(trip_id)
            except ValueError as exc:
                raise LiveSessionError(str(exc)) from exc
            self._schedule_cache[trip_id] = sched
        return sched

    def service_date(self, trip_id: str, vehicle_id: str, start_date: str | None, ping_epoch: float) -> str:
        """The trip's service date (``trip.start_date`` if valid, else derived; see module docstring)."""
        if _valid_yyyymmdd(start_date):
            return str(start_date)
        bound_key = (trip_id, vehicle_id)
        schedule = self._schedule(trip_id)
        bound = self._bound_date.get(bound_key)
        if bound is not None:
            sched = np.asarray(scheduled_epochs(schedule, bound, self.tz), dtype=float)
            sched = sched[np.isfinite(sched)]
            if sched.size and _dist_to_window(ping_epoch, float(sched.min()), float(sched.max())) <= SERVICE_DATE_SLACK_S:
                return bound
        date = derive_service_date(schedule, ping_epoch, self.tz)
        self._bound_date[bound_key] = date
        return date

    def _context(self, trip_id: str, route_id: str, service_date: str) -> tuple[TripContext, ForwardMatcher]:
        key = (trip_id, service_date)
        cached = self._contexts.get(key)
        if cached is None:
            try:
                cached = build_trip_context(
                    self._static,
                    trip_id,
                    route_id,
                    service_date,
                    self.tz,
                    holiday_region=self.holiday_region,
                    geometry_cache=self._geometry_cache,
                )
            except ValueError as exc:
                cached = str(exc)  # negative cache: do not retry the shape work every ping
            self._contexts[key] = cached
        if isinstance(cached, str):
            raise LiveSessionError(cached)
        return cached

    # -- state -----------------------------------------------------------------------

    def observe(
        self,
        trip_id: str,
        route_id: str | None,
        start_date: str | None,
        vehicle_id: str | None,
        ping: Ping,
    ) -> Hashable:
        """Feed one ping (arrival order); returns the vehicle-trip key. A repeated or older ping is a no-op."""
        vehicle = vehicle_id or ""
        date = self.service_date(str(trip_id), vehicle, start_date, ping.epoch)
        ctx, matcher = self._context(str(trip_id), str(route_id or ""), date)
        key: Hashable = (ctx.trip_id, ctx.service_date, vehicle)
        state = update_trip_state(
            self.store.get(key),
            ping,
            polyline=matcher,
            stop_progress_m=ctx.stop_progress_m,
            trip_key=key,
        )
        self.store.put(state)
        self._ctx_by_key[key] = ctx
        self._last_ping[key] = state.last_update_epoch
        self._last_position[key] = (ping.lat, ping.lon)
        return key

    def last_epoch(self, key: Hashable) -> float:
        state = self.store.get(key)
        return -math.inf if state is None else state.last_update_epoch

    def fresh_keys(self, t: float) -> list[Hashable]:
        """Vehicle-trips whose newest ping is no older than ``max_ping_age_s`` at ``t``."""
        lo = t - self.max_ping_age_s
        return [k for k, e in self._last_ping.items() if e >= lo]

    def n_tracked(self) -> int:
        """Vehicle-trips with state in this session (fresh or not)."""
        return len(self._last_ping)

    def evict(self, now_epoch: float) -> int:
        """Drop state of trips that ended or went silent (store policy); keeps the context caches."""
        dropped = self.store.evict(now_epoch)
        if dropped:
            for key in [k for k in self._ctx_by_key if self.store.get(k) is None]:
                del self._ctx_by_key[key]
                self._last_ping.pop(key, None)
                self._last_position.pop(key, None)
        return dropped

    # -- queries ---------------------------------------------------------------------

    def fresh_slot(self, key: Hashable, t: float) -> VehicleSlot | None:
        """The vehicle's slot at ``t``, or ``None`` if unknown, without a valid ping <= ``t``, or stale."""
        state = self.store.get(key)
        ctx = self._ctx_by_key.get(key)
        if state is None or ctx is None:
            return None
        latest = None
        for epoch, progress, ok in reversed(state.pings):
            if ok and epoch <= t:
                latest = (epoch, progress)
                break
        if latest is None or t - latest[0] > self.max_ping_age_s:
            return None
        lat, lon = self._last_position.get(key, (math.nan, math.nan))
        return VehicleSlot(key, ctx, state, latest[0], latest[1], str(key[2]) if isinstance(key, tuple) else "", lat, lon)

    def assemble(self, keys: Sequence[Hashable], t: float) -> pd.DataFrame:
        """Feature rows (all downstream stops) of every fresh vehicle in ``keys`` at ``t``.

        Adds ``vehicle_id`` and ``start_date`` (= service date) to the assembly's identity columns.
        """
        parts: list[pd.DataFrame] = []
        for key in keys:
            slot = self.fresh_slot(key, t)
            if slot is None:
                continue
            rows = assemble_live_features(slot.ctx, self.assets, slot.state, float(t), features=self.features)
            if rows.empty:
                continue
            rows["vehicle_id"] = slot.vehicle_id
            rows["start_date"] = slot.ctx.service_date
            rows["stop_sequence_id"] = np.asarray(slot.ctx.stop_sequences, dtype=int)[rows["stop_idx"].to_numpy()]
            parts.append(rows)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def predict(self, model: Any, keys: Sequence[Hashable], t: float) -> pd.DataFrame:
        """Assemble and predict: one ``model.predict`` call for all fresh vehicles.

        Returns the assembled rows plus ``horizon_s`` (the model output: seconds from ``t`` to arrival)
        and ``stop_sequence_id`` (integer stop sequence).
        """
        frame = self.assemble(keys, t)
        if frame.empty:
            return frame
        required = list(getattr(model, "required_features", self.features))
        x = frame[required].astype(float)
        frame = frame.copy()
        frame["horizon_s"] = np.asarray(model.predict(x), dtype=float).reshape(-1)
        return frame
