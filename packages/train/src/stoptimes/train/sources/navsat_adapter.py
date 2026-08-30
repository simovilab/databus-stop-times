"""Pure field-mapping adapter: raw BUCR navsat AVL rows -> canonical VP frame.

BUCR (navsat AVL) hands over none of the structural columns MBTA's GTFS-RT
feed supplies (``route_id``/``trip_id``/``stop_id``/``current_stop_sequence``)
-- those are inferred in a later step (``bucr_trip_inference.py``, not this
module). This adapter only renames/rescales/recasts the raw navsat fields
into the subset of the canonical VP frame (see ``feature_engineering.rt_source
.OUTPUT_COLUMNS``) that can be derived from BUCR data alone, plus
``odometer_km`` which downstream BUCR steps need but which has no MBTA
equivalent.

Pure and deterministic: no I/O, no network, no DB, no mutation of the input
frame. Cleaning (dropping stale/null fixes) and trip inference are explicitly
OUT OF SCOPE here -- this module only maps fields 1:1, so NaN/None values in
the raw frame are passed through (as NaN) rather than dropped or raised on.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Raw BUCR navsat columns this adapter actually consumes. `cr_datetime`,
# `ingested_at_utc`, and `lugar` are part of the raw contract (see
# docs/BUCR_DATASET_PROMPT.md) but are not needed to produce this frame --
# `ingested_at_utc` is consumed by trace cleaning (a later step), and
# `cr_datetime`/`lugar` are not part of the canonical VP mapping at all.
REQUIRED_RAW_COLUMNS: List[str] = [
    "plate_number",
    "cr_datetime_utc",
    "lat",
    "lon",
    "speed_kmh",
    "estado",
    "odometer_km",
]

# estado -> GTFS-RT VehicleStopStatus name. Literals copied from the MBTA
# path (feature_engineering/dataset_builder.py:146, .../tests/test_rt_source.py:32)
# so the two frames use exactly the same status vocabulary downstream.
_ESTADO_TO_STATUS = {
    "detenido": "STOPPED_AT",
    "movimiento": "IN_TRANSIT_TO",
}

KMH_TO_MS = 1.0 / 3.6

# Columns this adapter emits, in order. Deliberately NOT included:
# route_id/trip_id/stop_id/current_stop_sequence (structural columns filled
# in by the later trip-inference step) and `bearing` (BUCR navsat has no
# heading field -- emitted as an explicit NaN column below, never fabricated,
# so the frame still matches rt_source.OUTPUT_COLUMNS's column *set* for any
# consumer that reindexes against it).
OUTPUT_COLUMNS: List[str] = [
    "vehicle_id",
    "ts",
    "lat",
    "lon",
    "bearing",
    "speed",
    "current_status",
    "odometer_km",
]


def _missing_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]


def map_estado_to_status(estado: pd.Series) -> pd.Series:
    """Map BUCR ``estado`` strings to GTFS-RT ``current_status`` literals.

    Unknown/null values map to NaN rather than raising -- estado validation
    belongs to trace cleaning (a later step), not this pure field mapper.
    """
    return estado.map(_ESTADO_TO_STATUS)


def convert_speed_kmh_to_ms(speed_kmh: pd.Series) -> pd.Series:
    """km/h -> m/s, matching the canonical VP frame's ``speed`` unit.

    NaN in -> NaN out (no crash, no silent drop); negative/absurd values are
    passed through unchanged -- that is trace cleaning's job, not this one's.
    """
    return pd.to_numeric(speed_kmh, errors="coerce") * KMH_TO_MS


def navsat_to_vehicle_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw BUCR navsat rows to the canonical VP frame (minus structural cols).

    Args:
        df: Raw navsat rows with (at least) ``REQUIRED_RAW_COLUMNS``. Row
            order is preserved and treated as significant by the caller (the
            natural dedup key ``(plate_number, cr_datetime)`` is preserved via
            ``vehicle_id``/``ts``, but this function does not itself dedup).

    Returns:
        A new DataFrame with columns ``OUTPUT_COLUMNS``, one row per input
        row, in the same order. ``ts`` is a tz-aware UTC ``datetime64[ns, UTC]``
        Series, matching the convention
        ``rt_pipeline.storage.schema.add_partition_columns`` relies on
        (tz-naive inputs are interpreted as UTC, never local time).

    Raises:
        TypeError: if ``df`` is not a ``pandas.DataFrame``.
        ValueError: if any of ``REQUIRED_RAW_COLUMNS`` is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"navsat_to_vehicle_positions expects a DataFrame, got {type(df)!r}")

    missing = _missing_columns(df)
    if missing:
        raise ValueError(f"navsat frame missing required columns: {missing}")

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(index=df.index)
    out["vehicle_id"] = df["plate_number"]
    # utc=True: tz-aware values normalize to UTC, tz-naive values are
    # interpreted AS UTC (never local time) -- same rule as
    # rt_pipeline.storage.schema.add_partition_columns. `cr_datetime_utc` is
    # already UTC by name/contract, so this only guarantees the dtype.
    out["ts"] = pd.to_datetime(df["cr_datetime_utc"], utc=True, errors="coerce")
    out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    out["bearing"] = np.nan  # BUCR navsat has no heading field -- never fabricated.
    out["speed"] = convert_speed_kmh_to_ms(df["speed_kmh"])
    out["current_status"] = map_estado_to_status(df["estado"])
    out["odometer_km"] = pd.to_numeric(df["odometer_km"], errors="coerce")

    return out.reset_index(drop=True)[OUTPUT_COLUMNS]
