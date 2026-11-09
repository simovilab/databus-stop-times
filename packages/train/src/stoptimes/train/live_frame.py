"""Training frame for the live features: archived VehiclePositions + GTFS static -> feature rows.

Agency-agnostic (timezone from agency.txt via ``resolve_timezone`` with a configured
fallback; service date derived from the trip's schedule, never from the ping's calendar
date; no agency strings).  Every feature comes from :mod:`stoptimes.core.features.live`,
the same code serving uses.

Flow for a set of service days:

1. :func:`prepare_trip_records` - pings of one service day -> one :class:`TripRecord`
   (trip context + final :class:`TripState`) per trip instance.  The state's arrivals are
   the ground truth *and* the input of the segment statistics (F3 ``arrivals_from_pings``:
   interpolated crossing time, bracketing gap <= 60 s).
2. :func:`build_live_dataset` - statistics fitted out-of-fold by day for training rows
   and on all training days for the shipped asset (validation rows use the latter, as
   serving does); features on a prediction-time grid, rows = (instant x downstream stop),
   target ``time_to_arrival_seconds = arrival - t``.
3. :func:`write_live_assets` - the sidecar assets ``stoptimes train run`` packs into the artifact.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import sha256_hex
from stoptimes.core.contracts.data import ObservationKind, coerce_utc_ns
from stoptimes.core.features.identity import RouteVocabulary
from stoptimes.core.features.live import (
    LIVE_FEATURES,
    LiveAssets,
    TripContext,
    assemble_live_batch,
    build_trip_context,
)
from stoptimes.core.features.prior import (
    OutOfFoldStats,
    SegmentStats,
    fit_out_of_fold,
    segments_from_arrivals,
)
from stoptimes.core.features.schedule import resolve_timezone
from stoptimes.core.features.trip_state import Ping, StopArrival, TripState, arrivals_from_pings

__all__ = [
    "LiveFrameParams",
    "TripRecord",
    "LiveDataset",
    "prepare_trip_records",
    "arrivals_table",
    "build_live_dataset",
    "build_day_frame",
    "write_live_assets",
    "with_radius_arrivals",
    "LIVE_ASSETS_MANIFEST",
    "TARGET_COLUMN",
]

_log = logging.getLogger(__name__)

TARGET_COLUMN = "time_to_arrival_seconds"
LIVE_ASSETS_MANIFEST = "live.json"
SEGMENT_STATS_FILE = "segment_stats.bin"
ROUTE_VOCABULARY_FILE = "route_vocabulary.json"
_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class LiveFrameParams:
    """Knobs of the live-feature frame."""

    grid_s: float = 60.0
    """Prediction instants are multiples of this many seconds on the epoch clock. Per-ping
    instants would make ``sched_rem_t`` an exact function of ``delay_now`` (age 0), which
    serving ticks never are."""

    max_ping_age_s: float = 30.0
    """An instant needs a valid ping at most this old (serving's staleness guard)."""

    min_pings: int = 10
    """Trip instances with fewer pings are skipped."""

    rows_per_day: int | None = None
    """Uniform random subsample of (instant x stop) rows per day; None keeps all."""

    arrival_radius_m: float | None = None
    """When set, a stop also counts as reached at the first matched ping within this many
    metres *before* it along the shape (the earlier of that and the crossing wins) - the
    radius rule ``stoptimes val`` scores against. Needed where buses lay over just short of
    the final stop (bUCR's Odontología: 6 m short, so the crossing only fires when the bus
    leaves 30 min later). ``None`` keeps crossing-only arrivals."""

    seed: int = 42


@dataclass(frozen=True)
class TripRecord:
    """One observed trip instance: its context and the state after all its pings."""

    ctx: TripContext
    state: TripState
    vehicle_id: str | None
    n_pings: int


@dataclass(frozen=True)
class LiveDataset:
    """The frame plus everything fitted to build it."""

    frame: pd.DataFrame
    assets: LiveAssets  # what ships with the artifact (statistics fitted on ALL training days)
    out_of_fold: OutOfFoldStats
    tz: str
    holiday_region: str | None
    train_days: tuple[str, ...]
    val_days: tuple[str, ...]


# ---------------------------------------------------------------------------
# Pings -> trip records
# ---------------------------------------------------------------------------


def _vehicle_positions(observations: pd.DataFrame) -> pd.DataFrame:
    vp = observations
    if "kind" in vp.columns:
        vp = vp[vp["kind"] == ObservationKind.VEHICLE_POSITION.value]
    vp = vp.copy()
    for col in ("lat", "lon", "ts", "trip_id", "route_id", "vehicle_id", "speed", "current_status",
                "current_stop_sequence", "start_date"):
        if col not in vp.columns:
            vp[col] = pd.NA
    vp = vp[vp["lat"].notna() & vp["lon"].notna() & vp["ts"].notna() & vp["trip_id"].notna()].copy()
    if vp.empty:
        return vp.assign(epoch=pd.Series(dtype=float))
    vp["ts"] = coerce_utc_ns(vp["ts"])
    vp["trip_id"] = vp["trip_id"].astype(str)
    vp["epoch"] = vp["ts"].astype("int64").to_numpy() / 1e9
    return vp


def _assign_service_dates(vp: pd.DataFrame, static: Any, tz: str, skipped: dict[str, int]) -> pd.DataFrame:
    """Service day of each ping: the feed's ``start_date`` when present, else from the schedule.

    A trip runs within half a day of its scheduled midpoint, so the service day is the local
    date of ``ping_time - scheduled_midpoint_offset + 12 h`` (the service-day origin is local
    midnight to within the DST hour).
    """
    mids: dict[str, float] = {}
    for trip_id in vp["trip_id"].unique():
        try:
            offsets = np.asarray(static.trip_schedule(trip_id).offsets_s, dtype=float)
            mids[trip_id] = float(np.nanmedian(offsets)) if np.isfinite(offsets).any() else np.nan
        except ValueError:
            mids[trip_id] = np.nan
    mid = vp["trip_id"].map(mids)
    known = mid.notna()
    skipped["unknown_trip_pings"] = skipped.get("unknown_trip_pings", 0) + int((~known).sum())
    vp = vp[known].copy()
    shifted = vp["epoch"].to_numpy() - mid[known].to_numpy() + _SECONDS_PER_DAY / 2
    derived = pd.to_datetime(shifted, unit="s", utc=True).tz_convert(tz).strftime("%Y%m%d")
    given = vp["start_date"].astype("string").str.replace("-", "", regex=False)
    vp["service_date"] = np.where(given.notna() & (given.str.len() == 8), given.fillna(""), derived)
    return vp


def prepare_trip_records(
    observations: pd.DataFrame,
    static: Any,
    *,
    config_timezone: str | None,
    holiday_region: str | None,
    params: LiveFrameParams | None = None,
    service_dates: Sequence[str] | None = None,
    geometry_cache: dict | None = None,
) -> tuple[dict[str, list[TripRecord]], dict[str, int]]:
    """Trip records by service date (``YYYYMMDD``) from VehiclePosition observations.

    ``static`` is a loaded static feed (``GtfsStaticLoader``).  The timezone is agency.txt's,
    falling back to ``config_timezone``.  Per trip instance the most-pinged vehicle is
    kept, duplicate timestamps dropped.  Returns ``(records, skipped_counts)``.
    """
    params = params or LiveFrameParams()
    agency_tz = static.agency_timezone() if hasattr(static, "agency_timezone") else None
    tz = resolve_timezone(agency_tz, config_timezone)
    skipped: dict[str, int] = {}
    vp = _vehicle_positions(observations)
    if vp.empty:
        return {}, skipped
    vp = _assign_service_dates(vp, static, tz, skipped)
    if service_dates is not None:
        vp = vp[vp["service_date"].isin(set(service_dates))]
    cache = geometry_cache if geometry_cache is not None else {}
    by_day: dict[str, list[TripRecord]] = {}
    for (trip_id, service_date), group in vp.groupby(["trip_id", "service_date"], sort=True):
        record = _record_for_instance(
            static, str(trip_id), str(service_date), group, tz, holiday_region, params, cache, skipped
        )
        if record is not None:
            by_day.setdefault(str(service_date), []).append(record)
    return by_day, skipped


def _mode(values: pd.Series) -> Any:
    clean = values.dropna()
    return None if clean.empty else clean.astype(str).value_counts().index[0]


def _record_for_instance(
    static: Any,
    trip_id: str,
    service_date: str,
    group: pd.DataFrame,
    tz: str,
    holiday_region: str | None,
    params: LiveFrameParams,
    cache: dict,
    skipped: dict[str, int],
) -> TripRecord | None:
    vehicle = _mode(group["vehicle_id"])
    if vehicle is not None:
        group = group[group["vehicle_id"].astype(str) == vehicle]
    group = group.sort_values("epoch", kind="stable").drop_duplicates("epoch")
    if len(group) < params.min_pings:
        skipped["few_pings"] = skipped.get("few_pings", 0) + 1
        return None
    route_id = _mode(group["route_id"])
    if route_id is None and hasattr(static, "trips"):
        row = static.trips[static.trips["trip_id"] == trip_id]
        route_id = None if row.empty else str(row["route_id"].iloc[0])
    if route_id is None:
        skipped["no_route"] = skipped.get("no_route", 0) + 1
        return None
    try:
        ctx, matcher = build_trip_context(
            static, trip_id, route_id, service_date, tz, holiday_region=holiday_region, geometry_cache=cache
        )
    except ValueError:
        skipped["bad_geometry"] = skipped.get("bad_geometry", 0) + 1
        return None
    pings = [
        Ping(float(e), float(la), float(lo))
        for e, la, lo in zip(group["epoch"], group["lat"], group["lon"], strict=True)
    ]
    state = arrivals_from_pings(pings, matcher, ctx.stop_progress_m, trip_key=ctx.trip_key)
    if params.arrival_radius_m is not None:
        state = with_radius_arrivals(state, params.arrival_radius_m)
    return TripRecord(ctx, state, vehicle, len(pings))


def with_radius_arrivals(state: TripState, radius_m: float) -> TripState:
    """``state`` with each stop's arrival moved to the first valid ping within ``radius_m``
    before the stop along the shape, when that ping is earlier than the crossing (or the
    crossing was never seen). Such an arrival is known at its own ping (no lag, no gap)."""
    valid = [(e, p) for e, p, ok in state.pings if ok]
    arrivals = list(state.arrivals)
    for i, stop_m in enumerate(state.stop_progress_m):
        hit = next((e for e, p in valid if p >= stop_m - radius_m), None)
        if hit is not None and (arrivals[i] is None or hit < arrivals[i].epoch):
            arrivals[i] = StopArrival(epoch=float(hit), detect_epoch=float(hit), gap_s=0.0)
    return replace(state, arrivals=tuple(arrivals))


# ---------------------------------------------------------------------------
# Arrivals (ground truth) and statistics
# ---------------------------------------------------------------------------


def arrivals_table(records: Sequence[TripRecord]) -> pd.DataFrame:
    """One row per (trip instance, stop): the input of ``segments_from_arrivals``."""
    parts = []
    for rec in records:
        arrival, _, _ = rec.state.arrival_arrays()
        n = rec.ctx.n_stops
        parts.append(
            pd.DataFrame(
                {
                    "trip_key": f"{rec.ctx.trip_id}|{rec.ctx.service_date}",
                    "service_date": rec.ctx.service_date,
                    "route_id": rec.ctx.route_id,
                    "stop_index": np.arange(n),
                    "stop_id": rec.ctx.stop_ids,
                    "stop_progress_m": rec.ctx.stop_progress_m,
                    "arrival_epoch": arrival,
                }
            )
        )
    columns = ["trip_key", "service_date", "route_id", "stop_index", "stop_id", "stop_progress_m", "arrival_epoch"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def _instants(rec: TripRecord, params: LiveFrameParams) -> np.ndarray:
    valid = np.array([e for e, _, ok in rec.state.pings if ok])
    if valid.size == 0:
        return np.empty(0)
    first = np.ceil(valid[0] / params.grid_s) * params.grid_s
    grid = np.arange(first, valid[-1] + 1e-9, params.grid_s)
    pos = np.searchsorted(valid, grid, side="right") - 1
    ok = (pos >= 0) & (grid - valid[np.clip(pos, 0, None)] <= params.max_ping_age_s)
    return grid[ok]


def build_day_frame(
    records: Sequence[TripRecord], assets: LiveAssets, params: LiveFrameParams, day: str, split: str
) -> pd.DataFrame:
    """Rows of one service day (``split`` is ``'train'`` or ``'val'``) with the given assets.

    Training days must be given their out-of-fold statistics, validation days the shipped ones;
    :func:`build_live_dataset` does that bookkeeping.
    """
    parts = []
    for rec in records:
        times = _instants(rec, params)
        if times.size == 0:
            continue
        feats = assemble_live_batch(rec.ctx, assets, rec.state, times)
        if feats.empty:
            continue
        arrival, _, _ = rec.state.arrival_arrays()
        feats[TARGET_COLUMN] = arrival[feats["stop_idx"].to_numpy()] - feats["t"].to_numpy()
        parts.append(feats[feats[TARGET_COLUMN] > 0])
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    if params.rows_per_day is not None and len(frame) > params.rows_per_day:
        rng = np.random.default_rng([params.seed, int(day)])
        keep = np.sort(rng.choice(len(frame), size=params.rows_per_day, replace=False))
        frame = frame.iloc[keep].reset_index(drop=True)
    frame["day"] = day
    frame["split"] = split
    frame["vp_ts"] = pd.to_datetime(frame["t"].to_numpy(), unit="s", utc=True)
    frame[list(LIVE_FEATURES)] = frame[list(LIVE_FEATURES)].astype(np.float32)
    return frame


def build_live_dataset(
    train_days: Sequence[str],
    val_days: Sequence[str],
    load_records: Callable[[str], Sequence[TripRecord]],
    *,
    tz: str,
    holiday_region: str | None,
    params: LiveFrameParams | None = None,
) -> LiveDataset:
    """Fit assets and build the frame for ``train_days`` (``split='train'``) and ``val_days`` (``'val'``).

    ``load_records(day)`` returns the trip records of a service day (it is called twice per
    day: once to fit the statistics, once to build rows, so a caller can cache or stream).
    Training rows use statistics fitted on the other folds of ``train_days``; validation rows
    use statistics fitted on all training days, which is also the asset that ships.  A day
    in both lists, or a validation day inside the training span, is an error.
    """
    params = params or LiveFrameParams()
    train_days, val_days = tuple(train_days), tuple(val_days)
    overlap = set(train_days) & set(val_days)
    if overlap:
        raise ValueError(f"days in both train and val: {sorted(overlap)}")
    tables, routes = [], set()
    for day in train_days:
        records = load_records(day)
        tables.append(arrivals_table(records))
        routes.update(r.ctx.route_id for r in records)
    arrivals = pd.concat(tables, ignore_index=True) if tables else arrivals_table([])
    segments = segments_from_arrivals(arrivals, tz)
    oof = fit_out_of_fold(segments, [pd.Timestamp(d).date().isoformat() for d in train_days])
    vocabulary = RouteVocabulary.fit(sorted(routes))
    shipped = LiveAssets(segment_stats=oof.full, route_vocabulary=vocabulary)

    frames = []
    for day in (*train_days, *val_days):
        is_train = day in train_days
        stats = oof.for_day(pd.Timestamp(day).date().isoformat()) if is_train else oof.full
        frame = build_day_frame(
            load_records(day), LiveAssets(stats, vocabulary), params, day, "train" if is_train else "val"
        )
        _log.info("live frame %s (%s): %d rows", day, "train" if is_train else "val", len(frame))
        if len(frame):
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return LiveDataset(frame, shipped, oof, tz, holiday_region, train_days, val_days)


# ---------------------------------------------------------------------------
# Assets on disk (read by `stoptimes train run`)
# ---------------------------------------------------------------------------


def write_live_assets(directory: str | Path, dataset: LiveDataset) -> dict[str, str]:
    """Write ``segment_stats.bin``, ``route_vocabulary.json`` and ``live.json``; return the hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert dataset.assets.segment_stats is not None and dataset.assets.route_vocabulary is not None
    stats_sha = dataset.assets.segment_stats.save(directory / SEGMENT_STATS_FILE)
    vocab_bytes = json.dumps(dataset.assets.route_vocabulary.to_dict(), sort_keys=True).encode("utf-8")
    (directory / ROUTE_VOCABULARY_FILE).write_bytes(vocab_bytes)
    manifest = {
        "timezone": dataset.tz,
        "holiday_region": dataset.holiday_region,
        "features": list(LIVE_FEATURES),
        "train_days": list(dataset.train_days),
        "val_days": list(dataset.val_days),
        "assets": {
            "segment_stats": {"file": SEGMENT_STATS_FILE, "sha256": stats_sha},
            "route_vocabulary": {"file": ROUTE_VOCABULARY_FILE, "sha256": sha256_hex(vocab_bytes)},
        },
    }
    (directory / LIVE_ASSETS_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {name: a["sha256"] for name, a in manifest["assets"].items()}
