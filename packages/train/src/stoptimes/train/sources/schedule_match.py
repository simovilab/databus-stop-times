"""Match inferred trip instances to scheduled GTFS trips (agency-agnostic).

An AVL feed with no trip identity (bUCR's navsat) yields *trip instances* from trip
inference: a vehicle, a shape and a run of pings. The live features need the scheduled
trip each instance ran (``delay_now``, ``sched_rem``... come from its ``stop_times``),
exactly what a GTFS-RT feed hands over. This module recovers it:

1. :func:`stop_passages` - the time each instance passed each of its shape's stops
   (forward-only progress along the shape, linear interpolation between pings).
2. :func:`match_instances` - per service day and direction, a one-to-one assignment
   (Hungarian) between instances and the day's scheduled trips minimising
   ``|median(passage - scheduled)|``, plus a penalty when the scheduled trip's stop
   pattern differs from the instance's inferred shape (the geometry is a prior, the
   timetable decides) and per scheduled stop the instance did not cover. Pairs further
   apart than ``max_offset_s`` stay unmatched.

The output carries ``offset_s`` (positive = late), ``spread_s`` (how much the offset
varies along the trip) and ``margin_s`` (cost gap to the instance's best alternative
trip: small means the match is ambiguous), so callers can drop doubtful matches.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "MatchParams",
    "MATCH_COLUMNS",
    "gtfs_time_to_seconds",
    "scheduled_stop_times",
    "stop_passages",
    "match_instances",
]

MATCH_COLUMNS = [
    "instance_id",
    "service_date",
    "direction_id",
    "inferred_shape_id",
    "trip_id",
    "matched",
    "offset_s",
    "spread_s",
    "n_shared_stops",
    "pattern_agrees",
    "margin_s",
]

_UNMATCHABLE = 1e9


@dataclass(frozen=True)
class MatchParams:
    """Knobs of the instance -> scheduled trip assignment."""

    max_offset_s: float = 1200.0
    """A pair whose median offset exceeds this is never matched (20 min)."""
    min_shared_stops: int = 3
    """Stops the instance and the scheduled trip must both have, for the offset to mean anything."""
    pattern_penalty_s: float = 300.0
    """Cost added when the scheduled trip's shape differs from the instance's inferred shape."""
    missing_stop_penalty_s: float = 60.0
    """Cost added per scheduled stop the instance has no passage for."""


def gtfs_time_to_seconds(value: str) -> float:
    """GTFS ``HH:MM:SS`` (hours may exceed 23) -> seconds after the service day's origin."""
    h, m, s = (int(x) for x in str(value).split(":"))
    return float(h * 3600 + m * 60 + s)


def scheduled_stop_times(stop_times: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """``trip_id, direction_id, shape_id, stop_id, sched_s`` from GTFS ``stop_times``/``trips``."""
    cols = ["trip_id", "direction_id", "shape_id"]
    out = stop_times[["trip_id", "stop_id", "arrival_time"]].merge(
        trips[cols], on="trip_id", how="inner"
    )
    out = out[out["arrival_time"].notna()].copy()
    out["sched_s"] = out["arrival_time"].map(gtfs_time_to_seconds)
    out["direction_id"] = out["direction_id"].astype(int)
    out["stop_id"] = out["stop_id"].astype(str)
    return out[["trip_id", "direction_id", "shape_id", "stop_id", "sched_s"]].reset_index(drop=True)


def stop_passages(
    pings: pd.DataFrame,
    stop_progress_m: Sequence[tuple[str, float]],
    progress_of: Callable[[float, float], float],
) -> dict[str, float]:
    """Epoch seconds at which one instance passed each stop it spanned.

    ``pings`` are the instance's ``ts`` (UTC), ``lat``, ``lon`` in time order;
    ``stop_progress_m`` is ``(stop_id, progress along the shape)`` in stop order;
    ``progress_of(lat, lon)`` projects a point onto the same shape. Progress is made
    forward-only (running maximum) so GPS jitter at a stop never un-passes it; a stop
    before the first ping's progress (by more than 1 m) or past the last one is skipped.
    """
    if pings.empty:
        return {}
    prog = np.maximum.accumulate(
        np.array(
            [
                progress_of(float(a), float(b))
                for a, b in zip(pings["lat"], pings["lon"], strict=True)
            ]
        )
    )
    t = (
        pd.to_datetime(pings["ts"], utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
        / 1e9
    )
    out: dict[str, float] = {}
    for stop_id, p in stop_progress_m:
        if p < prog[0] - 1.0 or p > prog[-1] + 1.0:
            continue
        i = int(np.searchsorted(prog, p, side="left"))
        if i == 0 or i >= len(prog):
            out[str(stop_id)] = float(t[min(i, len(t) - 1)])
            continue
        p0, p1 = prog[i - 1], prog[i]
        frac = (p - p0) / (p1 - p0) if p1 > p0 else 0.0
        out[str(stop_id)] = float(t[i - 1] + (t[i] - t[i - 1]) * frac)
    return out


def _cost_matrix(
    obs: pd.DataFrame,
    sched: pd.DataFrame,
    inferred: Sequence[str],
    sched_shapes: Sequence[str],
    params: MatchParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(cost, median offset, spread, n shared)``, each ``instances x scheduled trips``."""
    stops = sorted(set(obs.columns) | set(sched.columns))
    o = obs.reindex(columns=stops).to_numpy(float)
    s = sched.reindex(columns=stops).to_numpy(float)
    diff = o[:, None, :] - s[None, :, :]
    shared = np.isfinite(diff).sum(axis=2)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pairs: unmatchable below
        med = np.nanmedian(diff, axis=2)
        spread = np.nanstd(diff, axis=2)
    missing = (np.isfinite(s)[None, :, :] & ~np.isfinite(o)[:, None, :]).sum(axis=2)
    pattern = np.array([[a != b for b in sched_shapes] for a in inferred], dtype=float)
    ok = (shared >= params.min_shared_stops) & (np.abs(med) <= params.max_offset_s)
    cost = (
        np.abs(med) + params.pattern_penalty_s * pattern + params.missing_stop_penalty_s * missing
    )
    return np.where(ok, cost, _UNMATCHABLE), med, spread, shared


def match_instances(
    passages: pd.DataFrame,
    instances: pd.DataFrame,
    schedule: pd.DataFrame,
    active_trips: Callable[[str], Iterable[str]],
    params: MatchParams | None = None,
) -> pd.DataFrame:
    """Assign each instance at most one scheduled trip, one-to-one per service day.

    Args:
        passages: ``instance_id, stop_id, passage_s`` with ``passage_s`` in seconds after
            the service day's local midnight (the clock ``stop_times`` uses).
        instances: ``instance_id, service_date (YYYYMMDD), direction_id, shape_id``.
        schedule: :func:`scheduled_stop_times` output.
        active_trips: ``service_date -> trip_ids`` running that day (calendar + exceptions).
        params: :class:`MatchParams`.

    Returns one row per instance with :data:`MATCH_COLUMNS`.
    """
    from scipy.optimize import linear_sum_assignment

    params = params or MatchParams()
    shape_of = schedule.drop_duplicates("trip_id").set_index("trip_id")["shape_id"].astype(str)
    pivots = {
        int(d): g.pivot_table(index="trip_id", columns="stop_id", values="sched_s", aggfunc="min")
        for d, g in schedule.groupby("direction_id")
    }
    obs_all = passages.pivot_table(
        index="instance_id", columns="stop_id", values="passage_s", aggfunc="min"
    )
    rows: list[dict[str, Any]] = []
    for (day, direction), inst in instances.groupby(["service_date", "direction_id"], sort=True):
        ids = [i for i in inst["instance_id"] if i in obs_all.index]
        inferred = inst.set_index("instance_id").loc[ids, "shape_id"].astype(str).tolist()
        base = {"service_date": str(day), "direction_id": int(direction)}
        running = set(active_trips(str(day)))
        sched = pivots.get(int(direction))
        if sched is not None:
            sched = sched[sched.index.isin(running)]
        if not ids or sched is None or sched.empty:
            rows += [_unmatched(base, i, s) for i, s in zip(ids, inferred, strict=True)]
            continue
        obs = obs_all.loc[ids].dropna(axis=1, how="all")
        cost, med, spread, shared = _cost_matrix(
            obs, sched, inferred, shape_of.loc[sched.index].tolist(), params
        )
        r, c = linear_sum_assignment(cost)
        chosen = {int(i): int(j) for i, j in zip(r, c, strict=True) if cost[i, j] < _UNMATCHABLE}
        for k, inst_id in enumerate(ids):
            if k not in chosen:
                rows.append(_unmatched(base, inst_id, inferred[k]))
                continue
            j = chosen[k]
            others = np.delete(cost[k], j)
            runner_up = float(others.min()) if others.size else _UNMATCHABLE
            trip = str(sched.index[j])
            rows.append(
                {
                    **base,
                    "instance_id": inst_id,
                    "inferred_shape_id": inferred[k],
                    "trip_id": trip,
                    "matched": True,
                    "offset_s": float(med[k, j]),
                    "spread_s": float(spread[k, j]),
                    "n_shared_stops": int(shared[k, j]),
                    "pattern_agrees": bool(shape_of.loc[trip] == inferred[k]),
                    "margin_s": float(min(runner_up, _UNMATCHABLE) - cost[k, j]),
                }
            )
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _unmatched(base: Mapping[str, Any], instance_id: Any, shape: str) -> dict[str, Any]:
    return {
        **base,
        "instance_id": instance_id,
        "inferred_shape_id": shape,
        "trip_id": None,
        "matched": False,
        "offset_s": np.nan,
        "spread_s": np.nan,
        "n_shared_stops": 0,
        "pattern_agrees": False,
        "margin_s": np.nan,
    }
