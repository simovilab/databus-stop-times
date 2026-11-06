"""Synthetic live-serving kit: a stub static feed, assets and pings (no network, no files).

Geometry: a straight eastbound line, 13 stops 500 m apart. ``trip1`` and ``trip2`` run on it
(08:00 and 08:20 on 2026-10-14, America/New_York); ``trip_late`` departs 25:30 of the *previous*
service day (to exercise service-date derivation). Duplicated verbatim in packages/val/tests.
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    Provenance,
    write_asset,
    write_metadata,
)
from stoptimes.core.domain.models import Stop
from stoptimes.core.features.identity import RouteVocabulary
from stoptimes.core.features.live import LIVE_FEATURES, LiveAssets
from stoptimes.core.features.prior import fit, segments_from_arrivals
from stoptimes.core.features.schedule import build_trip_schedule
from stoptimes.core.features.trip_state import Ping

LAT0, LON0 = 42.0, -71.0
M_DEG = 6371000.0 * math.pi / 180.0
COS = math.cos(math.radians(LAT0))
TZ = "America/New_York"
DAY = "20261014"
STOP_PROGRESS = tuple(float(x) for x in range(0, 6001, 500))
STOP_IDS = tuple(f"s{i}" for i in range(len(STOP_PROGRESS)))
MIDNIGHT = pd.Timestamp("2026-10-14 00:00", tz=TZ).timestamp()  # local midnight (no DST change that day)


def east(progress_m: float, north_m: float = 0.0) -> tuple[float, float]:
    return LAT0 + north_m / M_DEG, LON0 + progress_m / (COS * M_DEG)


def _hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# trip -> (route, first departure, seconds from the service-day origin)
TRIPS = {
    "trip1": ("R1", 8 * 3600),
    "trip2": ("R1", 8 * 3600 + 1200),
    "trip_late": ("R1", 25 * 3600 + 1800),
}


class StubStatic:
    """Duck-types the loader surface the live session needs."""

    def __init__(self) -> None:
        self.calls = {"trip_schedule": 0, "trip_geometry": 0}

    def agency_timezone(self) -> str | None:
        return TZ

    def trip_geometry(self, trip_id: str):
        if trip_id not in TRIPS:
            raise ValueError(f"trip_id {trip_id!r} not found in trips.txt")
        self.calls["trip_geometry"] += 1
        shape = [(*east(i * 250.0), None) for i in range(31)]
        stops = [
            Stop(stop_id=sid, stop_sequence=i + 1, lat=east(p)[0], lon=east(p)[1], progress_m=None)
            for i, (sid, p) in enumerate(zip(STOP_IDS, STOP_PROGRESS))
        ]
        return shape, stops

    def trip_schedule(self, trip_id: str):
        if trip_id not in TRIPS:
            raise ValueError(f"trip_id {trip_id!r} not found in trips.txt")
        self.calls["trip_schedule"] += 1
        _route, t0 = TRIPS[trip_id]
        times = [_hms(t0 + 60 * i) for i in range(len(STOP_IDS))]
        return build_trip_schedule(trip_id, STOP_IDS, list(range(1, len(STOP_IDS) + 1)), times, times, None)


def make_assets(route: str = "R1") -> LiveAssets:
    rng = np.random.default_rng(7)
    rows = []
    for d in range(12):
        date = f"2026-09-{(d % 10) + 1:02d}"
        for trip in range(4):
            t0 = pd.Timestamp(f"{date} 07:00", tz=TZ).timestamp() + trip * 1800.0
            tt = np.cumsum(rng.uniform(40, 90, len(STOP_IDS)))
            for i, sid in enumerate(STOP_IDS):
                rows.append(
                    {
                        "trip_key": f"{d}-{trip}",
                        "service_date": date,
                        "route_id": route,
                        "stop_index": i,
                        "stop_id": sid,
                        "stop_progress_m": STOP_PROGRESS[i],
                        "arrival_epoch": t0 + tt[i],
                    }
                )
    return LiveAssets(fit(segments_from_arrivals(pd.DataFrame(rows), TZ)), RouteVocabulary.fit(["R0", route, "R9"]))


def drive(trip_id: str, *, start_m: float = 0.0, seed: int = 1, cadence=(5.0, 12.0), delay_s: float = 0.0) -> list[Ping]:
    """A noisy drive along the line, starting at the trip's scheduled departure plus ``delay_s``."""
    rng = np.random.default_rng(seed)
    t = (MIDNIGHT - (86400 if trip_id == "trip_late" else 0)) + TRIPS[trip_id][1] + delay_s
    p = start_m
    out: list[Ping] = []
    while p < 6100:
        out.append(Ping(t, *east(p, north_m=float(rng.normal(0, 3)))))
        t += float(rng.uniform(*cadence))
        p += float(rng.uniform(20, 140))
    return out


class SumModel:
    """Deterministic stand-in for a trained model: horizon = sched_rem_t + delay_now (a plain ETA)."""

    family = "stub"
    artifact_id = "stub-live"

    def __init__(self, features=LIVE_FEATURES) -> None:
        self.required_features = list(features)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert list(x.columns) == self.required_features
        return (x["sched_rem_t"] + x["delay_now"].fillna(0.0)).to_numpy(float)


def write_live_registry(registry: Path, artifact_id: str = "live-synth", route: str = "R1") -> ArtifactMetadata:
    """A real hash-consistent artifact: a tiny fitted xgboost on the 19 live features plus both assets."""
    import xgboost as xgb

    directory = registry / artifact_id
    assets = make_assets(route)
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(300, len(LIVE_FEATURES))), columns=list(LIVE_FEATURES))
    model = xgb.XGBRegressor(n_estimators=6, max_depth=3, n_jobs=1).fit(x, rng.normal(300, 60, 300))
    model.required_features = list(LIVE_FEATURES)
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="xgboost",
        target="time_to_arrival_seconds",
        features=list(LIVE_FEATURES),
        provenance=Provenance(created_at=datetime.now(UTC)),
        timezone=TZ,
        holiday_region="US_MA",
        assets={
            "segment_stats": write_asset(directory, "segment_stats.bin", assets.segment_stats.to_bytes()),
            "route_vocabulary": write_asset(
                directory, "route_vocabulary.json", json.dumps(assets.route_vocabulary.to_dict()).encode()
            ),
        },
    )
    write_metadata(meta, directory)
    with open(directory / meta.model_filename, "wb") as fh:
        pickle.dump(model, fh)
    return meta
