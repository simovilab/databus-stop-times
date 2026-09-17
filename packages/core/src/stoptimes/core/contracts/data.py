"""Contract: canonical data tables.

Three shapes travel through every stoptimes track:

- `StaticFeed`   which GTFS static tables a loaded feed has, and which columns
                 each requires. Backed by `stoptimes.core.gtfs.static.GtfsStaticLoader`.
- `Observation`  one realtime record — a vehicle position OR a trip-update
                 stop-time prediction — source-agnostic (no agency names).
                 Backed by `stoptimes.record.storage.schema` (VP) and
                 `.tu_schema` (TU); see `contracts.adapters` for the mapping.
- `Arrival`      one actual stop arrival, with how it was derived.
                 Backed by `stoptimes.core.domain.models.ActualArrival`.

Timezone rule (non-negotiable): every timestamp in every table is
`datetime64[ns, UTC]` — timezone-aware, UTC, nanosecond resolution. Pandas 2.x
infers coarser units (`datetime64[us]`) from some inputs; `coerce_utc_ns`
below is the one place that pins it back to `ns`. Never compare a naive
timestamp to a table column — `coerce_utc_ns` treats a naive value as already
UTC, matching `record.storage.schema.add_partition_columns`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

UTC_DTYPE = "datetime64[ns, UTC]"


def coerce_utc_ns(series: pd.Series) -> pd.Series:
    """Return *series* as tz-aware UTC, nanosecond resolution.

    A naive timestamp is interpreted as UTC (never local time) — the same
    rule `record.storage.schema.add_partition_columns` uses, so a series
    built from either path lands on the identical dtype and values.
    """
    out = pd.to_datetime(series, utc=True)
    if hasattr(out, "dt") and hasattr(out.dt, "as_unit"):
        out = out.dt.as_unit("ns")
    else:  # pragma: no cover - pandas < 2.0 fallback
        out = out.astype(UTC_DTYPE)
    return out


# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------


class ColumnSpec(BaseModel):
    """One column of a table schema."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    """A canonical dtype string: one of "string", "int64", "float64", "bool",
    or `UTC_DTYPE` ("datetime64[ns, UTC]")."""
    required: bool = True
    description: str = ""


class TableSchema(BaseModel):
    """An ordered set of columns a table must (or may) have."""

    model_config = ConfigDict(extra="forbid")

    name: str
    columns: list[ColumnSpec]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)


def validate_frame(df: pd.DataFrame, schema: TableSchema) -> list[str]:
    """Return a list of human-readable problems with *df* against *schema*.

    Empty list means *df* satisfies the schema. Checks, in order:
    - every required column is present,
    - every present column with a declared dtype matches it (datetime columns
      must be exactly `UTC_DTYPE` — not merely "some datetime dtype").
    """
    problems: list[str] = []
    for col in schema.columns:
        if col.name not in df.columns:
            if col.required:
                problems.append(f"missing required column: {col.name}")
            continue
        actual = df[col.name].dtype
        if col.dtype == UTC_DTYPE:
            if not (isinstance(actual, pd.DatetimeTZDtype) and str(actual.tz) == "UTC" and actual.unit == "ns"):
                problems.append(
                    f"column {col.name!r}: expected {UTC_DTYPE}, got {actual} "
                    "(use contracts.data.coerce_utc_ns before validating)"
                )
        elif col.dtype == "string":
            if actual != object and not pd.api.types.is_string_dtype(actual):
                problems.append(f"column {col.name!r}: expected string, got {actual}")
        elif col.dtype in ("int64", "float64", "bool"):
            if str(actual) != col.dtype and not (
                col.dtype == "int64" and pd.api.types.is_integer_dtype(actual)
            ):
                problems.append(f"column {col.name!r}: expected {col.dtype}, got {actual}")
    return problems


# ---------------------------------------------------------------------------
# StaticFeed — GTFS static tables
# ---------------------------------------------------------------------------

STOPS_SCHEMA = TableSchema(
    name="stops",
    columns=[
        ColumnSpec(name="stop_id", dtype="string"),
        ColumnSpec(name="stop_lat", dtype="float64"),
        ColumnSpec(name="stop_lon", dtype="float64"),
        ColumnSpec(name="stop_name", dtype="string", required=False),
    ],
)

TRIPS_SCHEMA = TableSchema(
    name="trips",
    columns=[
        ColumnSpec(name="trip_id", dtype="string"),
        ColumnSpec(name="route_id", dtype="string"),
        ColumnSpec(name="service_id", dtype="string"),
        ColumnSpec(name="direction_id", dtype="int64", required=False),
        ColumnSpec(name="shape_id", dtype="string", required=False),
    ],
)

STOP_TIMES_SCHEMA = TableSchema(
    name="stop_times",
    columns=[
        ColumnSpec(name="trip_id", dtype="string"),
        ColumnSpec(name="stop_id", dtype="string"),
        ColumnSpec(name="stop_sequence", dtype="int64"),
        ColumnSpec(name="arrival_time", dtype="string", required=False),
        ColumnSpec(name="departure_time", dtype="string", required=False),
        ColumnSpec(name="shape_dist_traveled", dtype="float64", required=False),
    ],
)

SHAPES_SCHEMA = TableSchema(
    name="shapes",
    columns=[
        ColumnSpec(name="shape_id", dtype="string"),
        ColumnSpec(name="shape_pt_lat", dtype="float64"),
        ColumnSpec(name="shape_pt_lon", dtype="float64"),
        ColumnSpec(name="shape_pt_sequence", dtype="int64"),
        ColumnSpec(name="shape_dist_traveled", dtype="float64", required=False),
    ],
)

ROUTES_SCHEMA = TableSchema(
    name="routes",
    columns=[
        ColumnSpec(name="route_id", dtype="string"),
        ColumnSpec(name="route_short_name", dtype="string", required=False),
        ColumnSpec(name="route_long_name", dtype="string", required=False),
    ],
)

CALENDAR_SCHEMA = TableSchema(
    name="calendar",
    columns=[
        ColumnSpec(name="service_id", dtype="string"),
        *(ColumnSpec(name=day, dtype="int64") for day in (
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")),
        ColumnSpec(name="start_date", dtype="string"),
        ColumnSpec(name="end_date", dtype="string"),
    ],
)

CALENDAR_DATES_SCHEMA = TableSchema(
    name="calendar_dates",
    columns=[
        ColumnSpec(name="service_id", dtype="string"),
        ColumnSpec(name="date", dtype="string"),
        ColumnSpec(name="exception_type", dtype="int64"),
    ],
)

REQUIRED_STATIC_TABLES: tuple[str, ...] = ("stops", "trips", "stop_times")
# calendar/calendar_dates resolve which service day a trip ran on (TripUpdate
# start_date, arrival derivation). GTFS requires at least one of the two.
OPTIONAL_STATIC_TABLES: tuple[str, ...] = ("shapes", "routes", "calendar", "calendar_dates")

STATIC_TABLE_SCHEMAS: dict[str, TableSchema] = {
    "stops": STOPS_SCHEMA,
    "trips": TRIPS_SCHEMA,
    "stop_times": STOP_TIMES_SCHEMA,
    "shapes": SHAPES_SCHEMA,
    "routes": ROUTES_SCHEMA,
    "calendar": CALENDAR_SCHEMA,
    "calendar_dates": CALENDAR_DATES_SCHEMA,
}


class StaticFeed(BaseModel):
    """Which GTFS static tables a loaded feed has.

    Not a data container — `stoptimes.core.gtfs.static.GtfsStaticLoader`
    (T1) owns the actual DataFrames. This just records which of them loaded,
    so callers can check completeness before using them.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    tables: list[str] = Field(default_factory=list)

    def has(self, table: str) -> bool:
        return table in self.tables

    def missing_required(self) -> list[str]:
        return [t for t in REQUIRED_STATIC_TABLES if t not in self.tables]


# ---------------------------------------------------------------------------
# Observation — one realtime record (VP or TU), source-agnostic
# ---------------------------------------------------------------------------


class ObservationKind(str, Enum):
    VEHICLE_POSITION = "vehicle_position"
    TRIP_UPDATE = "trip_update"


class Observation(BaseModel):
    """One realtime record: a vehicle position sample, or one stop-time
    prediction from a TripUpdate.

    `kind` selects which optional field group is populated. Column names
    below mirror `record.storage.schema.DATA_COLUMNS` (VP) and
    `record.storage.tu_schema.DATA_COLUMNS` (TU) exactly, so archive rows map
    onto this without renaming (see `contracts.adapters`).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: ObservationKind
    source_id: str
    feed_name: str | None = None
    trip_id: str
    route_id: str | None = None
    direction_id: int | None = None
    start_date: str | None = None
    """YYYYMMDD, GTFS-RT `trip.start_date`."""
    vehicle_id: str | None = None
    ts: pd.Timestamp | None = None
    """The instant this record describes — VP: the position fix; TU: the
    prediction's `timestamp` (falling back to the feed header)."""
    ingested_at: pd.Timestamp | None = None

    # --- vehicle_position fields ---
    lat: float | None = None
    lon: float | None = None
    bearing: float | None = None
    speed: float | None = None
    current_status: str | None = None
    current_stop_sequence: int | None = None

    # --- trip_update fields (one stop_time_update row) ---
    stop_id: str | None = None
    stop_sequence: int | None = None
    arrival_time: pd.Timestamp | None = None
    arrival_delay: int | None = None
    arrival_uncertainty: int | None = None
    departure_time: pd.Timestamp | None = None
    departure_delay: int | None = None
    departure_uncertainty: int | None = None
    schedule_relationship: str | None = None
    """`stop_time_update.schedule_relationship` (e.g. SCHEDULED, SKIPPED, NO_DATA)."""
    trip_schedule_relationship: str | None = None
    revenue: bool | None = None

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> "Observation":
        if self.kind == ObservationKind.VEHICLE_POSITION:
            if self.lat is None or self.lon is None:
                raise ValueError("vehicle_position observation requires lat and lon")
        elif self.kind == ObservationKind.TRIP_UPDATE:
            if self.stop_id is None or self.stop_sequence is None:
                raise ValueError("trip_update observation requires stop_id and stop_sequence")
        return self


# Column schemas mirroring the record archive tables (documentation + validate_frame).
OBSERVATION_VP_SCHEMA = TableSchema(
    name="observation_vehicle_position",
    columns=[
        ColumnSpec(name="feed_name", dtype="string", required=False),
        ColumnSpec(name="vehicle_id", dtype="string", required=False),
        ColumnSpec(name="ts", dtype=UTC_DTYPE),
        ColumnSpec(name="lat", dtype="float64"),
        ColumnSpec(name="lon", dtype="float64"),
        ColumnSpec(name="bearing", dtype="float64", required=False),
        ColumnSpec(name="speed", dtype="float64", required=False),
        ColumnSpec(name="trip_id", dtype="string"),
        ColumnSpec(name="route_id", dtype="string", required=False),
        ColumnSpec(name="current_stop_sequence", dtype="int64", required=False),
        ColumnSpec(name="current_status", dtype="string", required=False),
        ColumnSpec(name="stop_id", dtype="string", required=False),
        ColumnSpec(name="ingested_at", dtype=UTC_DTYPE, required=False),
    ],
)

OBSERVATION_TU_SCHEMA = TableSchema(
    name="observation_trip_update",
    columns=[
        ColumnSpec(name="feed_name", dtype="string", required=False),
        ColumnSpec(name="route_id", dtype="string", required=False),
        ColumnSpec(name="trip_id", dtype="string"),
        ColumnSpec(name="start_date", dtype="string", required=False),
        ColumnSpec(name="direction_id", dtype="int64", required=False),
        ColumnSpec(name="vehicle_id", dtype="string", required=False),
        ColumnSpec(name="trip_schedule_relationship", dtype="string", required=False),
        ColumnSpec(name="revenue", dtype="bool", required=False),
        ColumnSpec(name="stop_id", dtype="string"),
        ColumnSpec(name="stop_sequence", dtype="int64"),
        ColumnSpec(name="arrival_time", dtype=UTC_DTYPE, required=False),
        ColumnSpec(name="arrival_delay", dtype="int64", required=False),
        ColumnSpec(name="arrival_uncertainty", dtype="int64", required=False),
        ColumnSpec(name="departure_time", dtype=UTC_DTYPE, required=False),
        ColumnSpec(name="departure_delay", dtype="int64", required=False),
        ColumnSpec(name="departure_uncertainty", dtype="int64", required=False),
        ColumnSpec(name="ts", dtype=UTC_DTYPE),
        ColumnSpec(name="ingested_at", dtype=UTC_DTYPE, required=False),
    ],
)


class PartitionLayout(BaseModel):
    """A Hive-style parquet partition layout, agency/bucket agnostic."""

    model_config = ConfigDict(extra="forbid")

    columns: tuple[str, ...]
    template: str
    """e.g. "year={year}/month={month}/day={day}/route_id={route_id}"."""

    def path_for(self, **values: Any) -> str:
        return self.template.format(**values)


OBSERVATION_VP_PARTITION = PartitionLayout(
    columns=("year", "month", "day", "route_id"),
    template="year={year}/month={month}/day={day}/route_id={route_id}",
)
OBSERVATION_TU_PARTITION = PartitionLayout(
    columns=("year", "month", "day"),
    template="year={year}/month={month}/day={day}",
)


# ---------------------------------------------------------------------------
# Arrival — actual stop arrival, however derived
# ---------------------------------------------------------------------------


class ArrivalMethod(str, Enum):
    """How an actual arrival time was determined."""

    REPORTED = "reported"
    """Taken directly from a published feed (e.g. a TripUpdate marked ARRIVED)."""
    MAP_MATCHED = "map_matched"
    """Derived from vehicle positions projected onto the route shape (T2)."""
    INTERPOLATED = "interpolated"
    """Derived by interpolating between two bracketing observations."""


class Arrival(BaseModel):
    """One actual (derived or reported) arrival at a stop."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source_id: str
    trip_id: str
    start_date: str
    route_id: str | None = None
    vehicle_id: str | None = None
    stop_id: str
    stop_sequence: int
    arrival_time: pd.Timestamp
    method: ArrivalMethod
    confidence: float | None = None
    """Optional [0, 1] confidence for derived methods; None for `reported`."""


ARRIVAL_SCHEMA = TableSchema(
    name="arrival",
    columns=[
        ColumnSpec(name="source_id", dtype="string"),
        ColumnSpec(name="trip_id", dtype="string"),
        ColumnSpec(name="start_date", dtype="string"),
        ColumnSpec(name="route_id", dtype="string", required=False),
        ColumnSpec(name="vehicle_id", dtype="string", required=False),
        ColumnSpec(name="stop_id", dtype="string"),
        ColumnSpec(name="stop_sequence", dtype="int64"),
        ColumnSpec(name="arrival_time", dtype=UTC_DTYPE),
        ColumnSpec(name="method", dtype="string"),
        ColumnSpec(name="confidence", dtype="float64", required=False),
    ],
)

ARRIVAL_PARTITION = PartitionLayout(
    columns=("year", "month", "day"),
    template="year={year}/month={month}/day={day}",
)
