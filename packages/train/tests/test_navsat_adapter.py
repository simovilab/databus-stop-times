"""Tests for the pure BUCR navsat -> canonical VP field-mapping adapter."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from feature_engineering.navsat_adapter import (
    OUTPUT_COLUMNS,
    navsat_to_vehicle_positions,
)


def _raw(n: int = 3, **overrides) -> pd.DataFrame:
    t0 = dt.datetime(2026, 8, 24, 12, 0, tzinfo=dt.timezone.utc)
    df = pd.DataFrame(
        {
            "plate_number": [f"P{i}" for i in range(n)],
            "cr_datetime": [(t0 - dt.timedelta(hours=6)) + dt.timedelta(seconds=30 * i) for i in range(n)],
            "cr_datetime_utc": [t0 + dt.timedelta(seconds=30 * i) for i in range(n)],
            "ingested_at_utc": [t0 + dt.timedelta(seconds=30 * i, milliseconds=200) for i in range(n)],
            "lat": [9.9 + 0.001 * i for i in range(n)],
            "lon": [-84.0 - 0.001 * i for i in range(n)],
            "speed_kmh": [36.0] * n,
            "odometer_km": [100.0 + i for i in range(n)],
            "estado": ["movimiento" if i % 2 == 0 else "detenido" for i in range(n)],
            "lugar": ["San Jose"] * n,
        }
    )
    for col, val in overrides.items():
        df[col] = val
    return df


def test_output_columns_and_row_count():
    out = navsat_to_vehicle_positions(_raw(4))
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 4


def test_vehicle_id_and_odometer_passthrough():
    raw = _raw(3)
    out = navsat_to_vehicle_positions(raw)
    assert list(out["vehicle_id"]) == list(raw["plate_number"])
    assert list(out["odometer_km"]) == list(raw["odometer_km"])


def test_lat_lon_passthrough():
    raw = _raw(3)
    out = navsat_to_vehicle_positions(raw)
    assert list(out["lat"]) == list(raw["lat"])
    assert list(out["lon"]) == list(raw["lon"])


def test_ts_is_utc_from_cr_datetime_utc():
    raw = _raw(2)
    out = navsat_to_vehicle_positions(raw)
    assert str(out["ts"].dtype) == "datetime64[ns, UTC]"
    assert list(out["ts"]) == list(pd.to_datetime(raw["cr_datetime_utc"], utc=True))


def test_speed_kmh_to_ms_conversion():
    raw = _raw(1, speed_kmh=[36.0])
    out = navsat_to_vehicle_positions(raw)
    assert out["speed"].iloc[0] == pytest.approx(10.0)  # 36 km/h == 10 m/s


def test_estado_movimiento_maps_to_in_transit_to():
    raw = _raw(1, estado=["movimiento"])
    out = navsat_to_vehicle_positions(raw)
    assert out["current_status"].iloc[0] == "IN_TRANSIT_TO"


def test_estado_detenido_maps_to_stopped_at():
    raw = _raw(1, estado=["detenido"])
    out = navsat_to_vehicle_positions(raw)
    assert out["current_status"].iloc[0] == "STOPPED_AT"


def test_bearing_is_null_not_fabricated():
    out = navsat_to_vehicle_positions(_raw(3))
    assert out["bearing"].isna().all()


def test_null_coords_and_speed_pass_through_as_nan_without_crashing():
    raw = _raw(3)
    raw.loc[1, "lat"] = np.nan
    raw.loc[1, "lon"] = None
    raw.loc[2, "speed_kmh"] = np.nan
    out = navsat_to_vehicle_positions(raw)
    assert len(out) == 3  # cleaning (row dropping) is a later step, not this one
    assert pd.isna(out["lat"].iloc[1])
    assert pd.isna(out["lon"].iloc[1])
    assert pd.isna(out["speed"].iloc[2])


def test_unknown_estado_maps_to_nan_status_without_crashing():
    raw = _raw(2, estado=["movimiento", "desconocido"])
    out = navsat_to_vehicle_positions(raw)
    assert out["current_status"].iloc[0] == "IN_TRANSIT_TO"
    assert pd.isna(out["current_status"].iloc[1])


def test_missing_required_column_raises_value_error():
    raw = _raw(2).drop(columns=["speed_kmh"])
    with pytest.raises(ValueError, match="speed_kmh"):
        navsat_to_vehicle_positions(raw)


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError):
        navsat_to_vehicle_positions([{"plate_number": "P0"}])


def test_empty_dataframe_returns_empty_with_correct_columns():
    raw = _raw(0)
    out = navsat_to_vehicle_positions(raw)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_order_preservation():
    raw = _raw(5)
    out = navsat_to_vehicle_positions(raw)
    assert list(out["vehicle_id"]) == list(raw["plate_number"])
    assert list(out["ts"]) == list(pd.to_datetime(raw["cr_datetime_utc"], utc=True))


def test_natural_key_fields_both_present_and_uniquely_identify_rows():
    raw = _raw(4)
    out = navsat_to_vehicle_positions(raw)
    key = list(zip(out["vehicle_id"], out["ts"]))
    assert len(set(key)) == len(out)


def test_structural_columns_not_emitted():
    out = navsat_to_vehicle_positions(_raw(2))
    for col in ("route_id", "trip_id", "stop_id", "current_stop_sequence"):
        assert col not in out.columns


def test_does_not_mutate_input_frame():
    raw = _raw(3)
    raw_copy = raw.copy(deep=True)
    navsat_to_vehicle_positions(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)
