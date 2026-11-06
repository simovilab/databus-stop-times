"""Historical segment statistics and the path prior (``prior_rem``).

Behavioral port of the 2026-11 MBTA study (``rs_stats.py`` / ``rs_feat._trip_block``); keyed only by
GTFS ids, timezone is always a parameter.

A *segment* is (route_id, from_stop_id, to_stop_id) between consecutive stops of a trip; its value is the
arrival-to-arrival time (traversal plus dwell at the from-stop).  ``SegmentStats`` holds a dense table
``H[segment, daytype, hour]`` of medians with the hierarchical fallback

    (seg, daytype, hour) n>=5 -> (seg, daytype, hour+-1) n>=8 -> (seg, hour+-1) n>=8
        -> (seg, daytype) n>=3 -> (seg) -> max(seglen, 30) / route median speed + 15

Hour and daytype are evaluated in the agency timezone using the UTC offset at local noon of the service
date (constant for the whole service day, as GTFS anchors times; on a DST-change day hours before the
change are therefore shifted by one, exactly as in the study).

Out-of-fold use: a training day's features must come from stats fitted on the *other* folds
(fold = day index mod K), otherwise the model over-trusts the prior.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TT_MAX = 1800.0
MIN_N: Mapping[str, int] = MappingProxyType({"l0": 5, "l1": 8, "l1b": 8, "l2": 3})
K_FOLDS = 6
FORMAT_VERSION = 1
_MAGIC = b"STOPSEGSTAT"
_SEP = "\x1f"  # joins ids into a key; cannot occur in a GTFS id


# ---------------------------------------------------------------------------- time helpers

def _as_date(day: Any) -> dt.date:
    return pd.Timestamp(str(day)).date()


def daytype_of(service_date: Any) -> int:
    """0 weekday, 1 Saturday, 2 Sunday (from the service date itself)."""
    w = _as_date(service_date).weekday()
    return 0 if w < 5 else (1 if w == 5 else 2)


def local_offset_s(service_date: Any, tz: str) -> float:
    """UTC offset in seconds at local noon of the service date."""
    d = _as_date(service_date)
    return float(pd.Timestamp(dt.datetime(d.year, d.month, d.day, 12), tz=tz).utcoffset().total_seconds())


def local_hour(epoch: Any, offset_s: Any) -> np.ndarray:
    return np.floor((np.asarray(epoch, float) + offset_s) / 3600.0).astype(np.int64) % 24


# ---------------------------------------------------------------------------- segments

SEGMENT_COLUMNS = ("route_id", "from_stop_id", "to_stop_id", "travel_s", "seglen_m", "hour", "daytype", "service_date")


def segments_from_arrivals(arrivals: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Consecutive-stop segments from per-(trip instance, stop) arrivals.

    Input columns: trip_key, service_date, route_id, stop_index, stop_id, stop_progress_m, arrival_epoch
    (NaN allowed).  Keeps segments with 0 < travel_s <= 1800.
    """
    e = arrivals.sort_values(["trip_key", "stop_index"], kind="stable").reset_index(drop=True)
    nxt = e.shift(-1)
    ok = (
        (nxt["trip_key"] == e["trip_key"])
        & (nxt["service_date"] == e["service_date"])
        & (nxt["stop_index"] == e["stop_index"] + 1)
        & e["arrival_epoch"].notna()
        & nxt["arrival_epoch"].notna()
    )
    tt = (nxt["arrival_epoch"] - e["arrival_epoch"])[ok].to_numpy(float)
    a_from = e["arrival_epoch"][ok].to_numpy(float)
    dates = e["service_date"][ok].map(lambda d: _as_date(d).isoformat())
    out = pd.DataFrame({
        "route_id": e["route_id"][ok].astype(str).to_numpy(),
        "from_stop_id": e["stop_id"][ok].astype(str).to_numpy(),
        "to_stop_id": nxt["stop_id"][ok].astype(str).to_numpy(),
        "travel_s": tt,
        "seglen_m": (nxt["stop_progress_m"] - e["stop_progress_m"])[ok].to_numpy(float),
        "service_date": dates.to_numpy(),
    })
    out["hour"] = 0
    out["daytype"] = 0
    for d in out["service_date"].unique():
        m = (out["service_date"] == d).to_numpy()
        out.loc[m, "hour"] = local_hour(a_from[m], local_offset_s(d, tz))
        out.loc[m, "daytype"] = daytype_of(d)
    out = out[(out["travel_s"] > 0) & (out["travel_s"] <= TT_MAX)].reset_index(drop=True)
    return out.astype({"hour": np.int64, "daytype": np.int64})[list(SEGMENT_COLUMNS)]


# ---------------------------------------------------------------------------- stats

def _key(route: Any, frm: Any, to: Any) -> np.ndarray:
    r, f, t = (np.asarray(x, dtype=str) for x in (route, frm, to))
    return np.char.add(np.char.add(np.char.add(r, _SEP), np.char.add(f, _SEP)), t)


@dataclass(frozen=True, eq=False)
class SegmentStats:
    """Dense median table with fallbacks.  ``keys`` are the sorted-by-first-seen segment keys."""

    keys: tuple[str, ...]
    table: np.ndarray  # float32 [S, 3, 24]
    route_speed: Mapping[str, float]  # m/s median per route
    speed_default: float
    n_obs: int

    def __post_init__(self) -> None:
        table = np.array(self.table, dtype=np.float32)
        table.setflags(write=False)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "route_speed", MappingProxyType(dict(self.route_speed)))
        object.__setattr__(self, "_index", {k: i for i, k in enumerate(self.keys)})

    # -- lookup
    def segment_ids(self, route_id: Any, from_stop_id: Any, to_stop_id: Any) -> np.ndarray:
        """Row of each segment in the table, -1 when unseen."""
        keys = _key(*np.broadcast_arrays(np.asarray(route_id, str), np.asarray(from_stop_id, str), np.asarray(to_stop_id, str)))
        idx = self._index  # type: ignore[attr-defined]
        return np.fromiter((idx.get(k, -1) for k in keys.ravel()), dtype=np.int64, count=keys.size).reshape(keys.shape)

    def lookup_ids(self, sid: Any, daytype: Any, hour: Any, seglen_m: Any, route_id: Any) -> np.ndarray:
        sid, daytype, hour, seglen, route = np.broadcast_arrays(
            np.asarray(sid, np.int64), np.asarray(daytype, np.int64), np.asarray(hour, np.int64),
            np.asarray(seglen_m, float), np.asarray(route_id, str))
        ok = sid >= 0
        if self.table.shape[0]:
            v = np.where(ok, self.table[np.where(ok, sid, 0), daytype, hour % 24], np.nan)
        else:
            v = np.full(sid.shape, np.nan)
        speed = pd.Series(route.ravel()).map(self.route_speed).fillna(self.speed_default).to_numpy().reshape(route.shape)
        fb = np.maximum(seglen, 30.0) / speed + 15.0
        return np.where(np.isfinite(v), v, fb).astype(float)

    def lookup(self, route_id: Any, from_stop_id: Any, to_stop_id: Any, daytype: Any, hour: Any, seglen_m: Any) -> np.ndarray:
        """Median arrival-to-arrival seconds; arguments broadcast.  Unseen segments use the speed fallback."""
        sid = self.segment_ids(route_id, from_stop_id, to_stop_id)
        return self.lookup_ids(sid, daytype, hour, seglen_m, np.asarray(route_id, str))

    # -- persistence
    def to_bytes(self) -> bytes:
        table = np.ascontiguousarray(self.table, dtype="<f4")
        header = {
            "version": FORMAT_VERSION,
            "keys": list(self.keys),
            "route_speed": sorted(self.route_speed.items()),
            "speed_default": float(self.speed_default),
            "n_obs": int(self.n_obs),
            "shape": list(table.shape),
        }
        h = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return _MAGIC + struct.pack("<Q", len(h)) + h + table.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SegmentStats":
        if not raw.startswith(_MAGIC):
            raise ValueError("not a segment-statistics file (bad magic)")
        p = len(_MAGIC)
        (n,) = struct.unpack("<Q", raw[p:p + 8])
        header = json.loads(raw[p + 8:p + 8 + n].decode("ascii"))
        if header.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported segment-statistics format version {header.get('version')!r}; expected {FORMAT_VERSION}")
        shape = tuple(header["shape"])
        body = raw[p + 8 + n:]
        if len(body) != int(np.prod(shape)) * 4:
            raise ValueError("segment-statistics file is truncated or has trailing bytes")
        table = np.frombuffer(body, dtype="<f4").reshape(shape).astype(np.float32)
        return cls(tuple(header["keys"]), table, {k: float(v) for k, v in header["route_speed"]},
                   float(header["speed_default"]), int(header["n_obs"]))

    def save(self, path: str | os.PathLike[str]) -> str:
        """Write deterministically (same stats -> same bytes); returns the sha256 hex of the bytes."""
        raw = self.to_bytes()
        tmp = f"{os.fspath(path)}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SegmentStats":
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())


def _grp_median(d: pd.DataFrame, by: list[str], min_n: int) -> pd.DataFrame:
    g = d.groupby(by)["tt"].agg(["median", "count"]).reset_index()
    return g[g["count"] >= min_n]


def fit(segments: pd.DataFrame) -> SegmentStats:
    """Fit the median table (with hierarchical fallback) on segment rows from ``segments_from_arrivals``."""
    if len(segments) == 0:
        return SegmentStats((), np.zeros((0, 3, 24), np.float32), {}, 5.0, 0)
    seg = segments.reset_index(drop=True)
    keys = _key(seg["route_id"], seg["from_stop_id"], seg["to_stop_id"])
    sid, uniq = pd.factorize(pd.Series(keys))
    S = len(uniq)
    tt = seg["travel_s"].to_numpy(float)
    d = pd.DataFrame({"sid": sid, "dtype": seg["daytype"].to_numpy(np.int64), "hour": seg["hour"].to_numpy(np.int64), "tt": tt})
    H = np.full((S, 3, 24), np.nan, np.float32)

    g = _grp_median(d, ["sid", "dtype", "hour"], MIN_N["l0"])
    H[g.sid.to_numpy(), g.dtype.to_numpy(), g.hour.to_numpy()] = g["median"].to_numpy(np.float32)

    dd = pd.concat([d.assign(hour=(d.hour + s) % 24) for s in (-1, 0, 1)], ignore_index=True)
    g = _grp_median(dd, ["sid", "dtype", "hour"], MIN_N["l1"])
    ix = (g.sid.to_numpy(), g.dtype.to_numpy(), g.hour.to_numpy())
    cur = H[ix]
    H[ix] = np.where(np.isnan(cur), g["median"].to_numpy(np.float32), cur)

    g = _grp_median(dd, ["sid", "hour"], MIN_N["l1b"])
    for dtp in range(3):
        ix = (g.sid.to_numpy(), np.full(len(g), dtp), g.hour.to_numpy())
        cur = H[ix]
        H[ix] = np.where(np.isnan(cur), g["median"].to_numpy(np.float32), cur)

    g = _grp_median(d, ["sid", "dtype"], MIN_N["l2"])
    m2 = np.full((S, 3), np.nan, np.float32)
    m2[g.sid.to_numpy(), g.dtype.to_numpy()] = g["median"].to_numpy(np.float32)
    m3 = d.groupby("sid")["tt"].median().reindex(range(S)).to_numpy(np.float32)
    fillv = np.where(np.isnan(m2), m3[:, None], m2)
    H = np.where(np.isnan(H), fillv[:, :, None], H).astype(np.float32)

    sl = seg["seglen_m"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vv = sl / tt
        okv = (tt > 5) & (sl > 20)
    rs = pd.Series(vv[okv]).groupby(seg["route_id"].to_numpy()[okv]).median().to_dict()
    sp_def = float(np.median(vv[okv])) if okv.any() else 5.0
    return SegmentStats(tuple(str(u) for u in uniq), H, {str(k): float(v) for k, v in rs.items()}, sp_def, len(seg))


@dataclass(frozen=True)
class OutOfFoldStats:
    """Per-fold stats (each fitted on the OTHER folds), full-data stats and the day -> fold map."""

    by_fold: tuple[SegmentStats, ...]
    full: SegmentStats
    day_fold: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "day_fold", MappingProxyType(dict(self.day_fold)))

    def for_day(self, service_date: Any) -> SegmentStats:
        """Stats a *training* day's rows must use (its own fold excluded)."""
        return self.by_fold[self.day_fold[_as_date(service_date).isoformat()]]


def fit_out_of_fold(segments: pd.DataFrame, days: Sequence[Any], k: int = K_FOLDS) -> OutOfFoldStats:
    """fold(day) = position of the day in ``days`` (order given, duplicates dropped) mod k."""
    ordered = list(dict.fromkeys(_as_date(d).isoformat() for d in days))
    fold = {d: i % k for i, d in enumerate(ordered)}
    seg = segments[segments["service_date"].map(lambda d: _as_date(d).isoformat()).isin(fold)].reset_index(drop=True)
    seg_fold = seg["service_date"].map(lambda d: fold[_as_date(d).isoformat()]).to_numpy()
    by_fold = tuple(fit(seg[seg_fold != f].reset_index(drop=True)) for f in range(k))
    return OutOfFoldStats(by_fold, fit(seg), fold)


# ---------------------------------------------------------------------------- path prior

def prior_remaining(
    stats: SegmentStats,
    route_id: str,
    stop_ids: Sequence[str],
    stop_progress_m: Sequence[float],
    progress_m: float,
    t: float,
    tz: str,
    service_date: Any,
    target_idx: Sequence[int],
) -> np.ndarray:
    """Seconds from ``t`` to arrival at each target stop, summing median segment times along the path.

    The current segment contributes ``(1 - frac) * H`` with ``frac`` the progress through it; each later
    segment is looked up at the local hour of ``t + eta`` so the hour rolls over mid-path.  Stops already
    passed (index < j0 = searchsorted(stop_progress, progress, 'right')) are NaN.
    """
    sprog = np.asarray(stop_progress_m, float)
    sids = [str(s) for s in stop_ids]
    n = len(sids)
    targets = np.asarray(target_idx, np.int64)
    out = np.full(targets.shape, np.nan)
    j0 = int(np.searchsorted(sprog, progress_m, side="right"))
    if n < 2 or j0 >= n or targets.size == 0:
        return out
    off = local_offset_s(service_date, tz)
    dtp = daytype_of(service_date)
    seg_ids = stats.segment_ids(np.full(n - 1, str(route_id)), sids[:-1], sids[1:])
    seg_len = np.diff(sprog)
    route = str(route_id)

    def seg_time(k: int, when: float) -> float:
        return float(stats.lookup_ids(seg_ids[k], dtp, local_hour(when, off), seg_len[k], route))

    s = j0 - 1
    if s >= 0:
        span = max(sprog[s + 1] - sprog[s], 1.0)
        frac = float(np.clip((progress_m - sprog[s]) / span, 0.0, 1.0))
        cur = (1.0 - frac) * seg_time(s, t)
    else:
        cur = 0.0
    top = int(min(targets.max(), n - 1))
    eta = np.full(n, np.nan)
    eta[j0] = cur
    for j in range(j0 + 1, top + 1):
        eta[j] = eta[j - 1] + seg_time(j - 1, t + eta[j - 1])
    valid = (targets >= 0) & (targets < n)
    out[valid] = eta[targets[valid]]
    return out
