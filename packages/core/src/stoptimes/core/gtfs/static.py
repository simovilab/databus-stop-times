"""GTFS static .zip loader (stdlib zipfile + pandas). Phase P1.

GtfsStaticLoader: loads stops, trips, stop_times, shapes, routes from a
local GTFS .zip file and provides trip_geometry() for spatial work (P2).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from stoptimes.core.domain.models import Stop
from stoptimes.core.features.schedule import TripSchedule, build_trip_schedule


class GtfsStaticLoader:
    """Load and expose GTFS static data from a local .zip file.

    DataFrames are available as attributes:
      - stops       (stops.txt)
      - trips       (trips.txt)
      - stop_times  (stop_times.txt)
      - shapes      (shapes.txt, empty DataFrame if absent)
      - routes      (routes.txt, empty DataFrame if absent)

    Numeric columns are coerced after loading so consumers get float/int values.
    """

    def __init__(self, zip_path: str | Path) -> None:
        self._zip_path = Path(zip_path)
        if not self._zip_path.exists():
            raise FileNotFoundError(self._zip_path)
        self._schedule_cache: dict[str, TripSchedule] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with zipfile.ZipFile(self._zip_path) as zf:
            names = set(zf.namelist())
            self.stops: pd.DataFrame = pd.read_csv(zf.open("stops.txt"), dtype=str)
            self.trips: pd.DataFrame = pd.read_csv(zf.open("trips.txt"), dtype=str)
            self.stop_times: pd.DataFrame = pd.read_csv(zf.open("stop_times.txt"), dtype=str)
            self.shapes: pd.DataFrame = (
                pd.read_csv(zf.open("shapes.txt"), dtype=str)
                if "shapes.txt" in names
                else pd.DataFrame()
            )
            self.routes: pd.DataFrame = (
                pd.read_csv(zf.open("routes.txt"), dtype=str)
                if "routes.txt" in names
                else pd.DataFrame()
            )

            self.agency: pd.DataFrame = (
                pd.read_csv(zf.open("agency.txt"), dtype=str)
                if "agency.txt" in names
                else pd.DataFrame()
            )

        # Coerce numeric columns
        for col in ("stop_lat", "stop_lon"):
            if col in self.stops.columns:
                self.stops[col] = pd.to_numeric(self.stops[col], errors="coerce")

        if not self.shapes.empty:
            for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"):
                if col in self.shapes.columns:
                    self.shapes[col] = pd.to_numeric(self.shapes[col], errors="coerce")

        if "stop_sequence" in self.stop_times.columns:
            self.stop_times["stop_sequence"] = pd.to_numeric(
                self.stop_times["stop_sequence"], errors="coerce"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trip_geometry(
        self,
        trip_id: str,
    ) -> tuple[list[tuple[float, float, float | None]], list[Stop]]:
        """Return (shape_points, ordered_stops) for *trip_id*.

        shape_points: [(lat, lon, shape_dist_traveled_or_None), ...]
            Ordered by shape_pt_sequence from shapes.txt for the trip's shape_id.

        ordered_stops: [Stop, ...]
            Built by joining stop_times (for this trip, ordered by stop_sequence)
            to stops.txt.  progress_m is left None — assigned in P2 (spatial).

        Raises ValueError if trip_id is not found in trips.txt.
        """
        trip_rows = self.trips[self.trips["trip_id"] == trip_id]
        if trip_rows.empty:
            raise ValueError(f"trip_id {trip_id!r} not found in trips.txt")

        trip_row = trip_rows.iloc[0]
        shape_id: str | None = (
            str(trip_row["shape_id"])
            if "shape_id" in trip_row.index and not pd.isna(trip_row["shape_id"])
            else None
        )

        # --- shape points ---
        shape_points: list[tuple[float, float, float | None]] = []
        if shape_id and not self.shapes.empty and "shape_id" in self.shapes.columns:
            shape_rows = (
                self.shapes[self.shapes["shape_id"] == shape_id]
                .sort_values("shape_pt_sequence")
            )
            has_dist = "shape_dist_traveled" in shape_rows.columns
            for _, row in shape_rows.iterrows():
                dist: float | None = (
                    float(row["shape_dist_traveled"])
                    if has_dist and not pd.isna(row["shape_dist_traveled"])
                    else None
                )
                shape_points.append((float(row["shape_pt_lat"]), float(row["shape_pt_lon"]), dist))

        # --- ordered stops ---
        st_rows = (
            self.stop_times[self.stop_times["trip_id"] == trip_id]
            .sort_values("stop_sequence")
        )
        stops_indexed = self.stops.set_index("stop_id")
        ordered_stops: list[Stop] = []
        for _, st_row in st_rows.iterrows():
            stop_id = str(st_row["stop_id"])
            if stop_id not in stops_indexed.index:
                continue
            stop_row = stops_indexed.loc[stop_id]
            # Handle rare case where stop_id is duplicated (returns DataFrame row)
            if isinstance(stop_row, pd.DataFrame):
                stop_row = stop_row.iloc[0]
            ordered_stops.append(
                Stop(
                    stop_id=stop_id,
                    stop_sequence=int(st_row["stop_sequence"]),
                    lat=float(stop_row["stop_lat"]),
                    lon=float(stop_row["stop_lon"]),
                    progress_m=None,
                )
            )

        return shape_points, ordered_stops

    def agency_timezone(self) -> str | None:
        """First agency's ``agency_timezone`` from agency.txt, or None if absent/blank."""
        if self.agency.empty or "agency_timezone" not in self.agency.columns:
            return None
        value = self.agency["agency_timezone"].iloc[0]
        if pd.isna(value) or not str(value).strip():
            return None
        return str(value).strip()

    def trip_schedule(self, trip_id: str) -> TripSchedule:
        """Scheduled stop times of *trip_id* (cached). Raises ValueError if unknown."""
        cached = self._schedule_cache.get(trip_id)
        if cached is not None:
            return cached
        if not (self.trips["trip_id"] == trip_id).any():
            raise ValueError(f"trip_id {trip_id!r} not found in trips.txt")
        rows = self.stop_times[self.stop_times["trip_id"] == trip_id]
        dist = (
            pd.to_numeric(rows["shape_dist_traveled"], errors="coerce").tolist()
            if "shape_dist_traveled" in rows.columns
            else None
        )

        def col(name: str) -> list[object]:
            return rows[name].tolist() if name in rows.columns else [None] * len(rows)

        schedule = build_trip_schedule(
            trip_id,
            rows["stop_id"].astype(str).tolist(),
            rows["stop_sequence"].astype(int).tolist(),
            col("arrival_time"),
            col("departure_time"),
            dist,
        )
        self._schedule_cache[trip_id] = schedule
        return schedule
