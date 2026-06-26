"""P0 storage tests — SqliteStorage round-trip verification."""

from __future__ import annotations

import pytest

from etaval.domain.models import ActualArrival, PredictionSample, ValidationResult
from etaval.storage.sqlite import SqliteStorage

# ── fixtures & helpers ────────────────────────────────────────────────────────


@pytest.fixture
def store() -> SqliteStorage:
    """In-memory SQLite store, auto-closed after each test."""
    with SqliteStorage(":memory:") as s:
        yield s


_TRIP_KEY = "src:trip1:veh1:20240101"


def make_sample(**overrides: object) -> PredictionSample:
    defaults: dict[str, object] = dict(
        trip_key=_TRIP_KEY,
        predictor_id="gtfsrt",
        stop_id="stop_A",
        stop_sequence=1,
        predicted_arrival_epoch=1_700_000_100.0,
        made_at_epoch=1_700_000_000.0,
        vehicle_progress_m_at_made=500.0,
        dist_remaining_m=200.0,
        stops_remaining=3,
    )
    defaults.update(overrides)
    return PredictionSample(**defaults)  # type: ignore[arg-type]


def make_arrival(**overrides: object) -> ActualArrival:
    defaults: dict[str, object] = dict(
        trip_key=_TRIP_KEY,
        stop_id="stop_A",
        stop_sequence=1,
        actual_arrival_epoch=1_700_000_090.0,
        method="shape_distance",
    )
    defaults.update(overrides)
    return ActualArrival(**defaults)  # type: ignore[arg-type]


def make_result(**overrides: object) -> ValidationResult:
    defaults: dict[str, object] = dict(
        trip_key=_TRIP_KEY,
        predictor_id="gtfsrt",
        stop_id="stop_A",
        stop_sequence=1,
        route_id="42",
        error_s=10.0,
        horizon_s=90.0,
        dist_remaining_m=200.0,
        stops_remaining=3,
        predicted_arrival_epoch=1_700_000_100.0,
        actual_arrival_epoch=1_700_000_090.0,
        made_at_epoch=1_700_000_000.0,
    )
    defaults.update(overrides)
    return ValidationResult(**defaults)  # type: ignore[arg-type]


# ── prediction_samples ────────────────────────────────────────────────────────


def test_prediction_samples_stored(store: SqliteStorage) -> None:
    sample = make_sample()
    store.save_samples([sample])

    rows = store._conn.execute("SELECT * FROM prediction_samples").fetchall()
    assert len(rows) == 1
    row = rows[0]

    assert row["trip_key"] == sample.trip_key
    assert row["predictor_id"] == sample.predictor_id
    assert row["stop_id"] == sample.stop_id
    assert row["stop_sequence"] == sample.stop_sequence
    assert row["predicted_arrival_epoch"] == pytest.approx(sample.predicted_arrival_epoch)
    assert row["made_at_epoch"] == pytest.approx(sample.made_at_epoch)
    assert row["vehicle_progress_m_at_made"] == pytest.approx(sample.vehicle_progress_m_at_made)
    assert row["dist_remaining_m"] == pytest.approx(sample.dist_remaining_m)
    assert row["stops_remaining"] == sample.stops_remaining


def test_prediction_samples_null_optionals(store: SqliteStorage) -> None:
    sample = make_sample(
        vehicle_progress_m_at_made=None,
        dist_remaining_m=None,
        stops_remaining=None,
    )
    store.save_samples([sample])

    rows = store._conn.execute("SELECT * FROM prediction_samples").fetchall()
    assert len(rows) == 1
    assert rows[0]["vehicle_progress_m_at_made"] is None
    assert rows[0]["dist_remaining_m"] is None
    assert rows[0]["stops_remaining"] is None


def test_save_multiple_samples(store: SqliteStorage) -> None:
    samples = [make_sample(stop_id=f"stop_{i}", stop_sequence=i) for i in range(5)]
    store.save_samples(samples)

    count = store._conn.execute("SELECT COUNT(*) FROM prediction_samples").fetchone()[0]
    assert count == 5


# ── actual_arrivals ───────────────────────────────────────────────────────────


def test_actual_arrivals_stored(store: SqliteStorage) -> None:
    arrival = make_arrival()
    store.save_arrivals([arrival])

    rows = store._conn.execute("SELECT * FROM actual_arrivals").fetchall()
    assert len(rows) == 1
    row = rows[0]

    assert row["trip_key"] == arrival.trip_key
    assert row["stop_id"] == arrival.stop_id
    assert row["stop_sequence"] == arrival.stop_sequence
    assert row["actual_arrival_epoch"] == pytest.approx(arrival.actual_arrival_epoch)
    assert row["method"] == arrival.method


def test_save_multiple_arrivals(store: SqliteStorage) -> None:
    arrivals = [make_arrival(stop_id=f"stop_{i}", stop_sequence=i) for i in range(3)]
    store.save_arrivals(arrivals)

    count = store._conn.execute("SELECT COUNT(*) FROM actual_arrivals").fetchone()[0]
    assert count == 3


# ── validation_results ────────────────────────────────────────────────────────


def test_save_and_load_results_roundtrip(store: SqliteStorage) -> None:
    expected = make_result()
    store.save_results([expected])
    loaded = store.load_results()

    assert len(loaded) == 1
    r = loaded[0]

    assert r.trip_key == expected.trip_key
    assert r.predictor_id == expected.predictor_id
    assert r.stop_id == expected.stop_id
    assert r.stop_sequence == expected.stop_sequence
    assert r.route_id == expected.route_id
    assert r.error_s == pytest.approx(expected.error_s)
    assert r.horizon_s == pytest.approx(expected.horizon_s)
    assert r.dist_remaining_m == pytest.approx(expected.dist_remaining_m)
    assert r.stops_remaining == expected.stops_remaining
    assert r.predicted_arrival_epoch == pytest.approx(expected.predicted_arrival_epoch)
    assert r.actual_arrival_epoch == pytest.approx(expected.actual_arrival_epoch)
    assert r.made_at_epoch == pytest.approx(expected.made_at_epoch)


def test_load_results_returns_validation_result_instances(store: SqliteStorage) -> None:
    store.save_results([make_result()])
    loaded = store.load_results()
    assert all(isinstance(r, ValidationResult) for r in loaded)


def test_load_results_filter_by_predictor_id(store: SqliteStorage) -> None:
    r1 = make_result(predictor_id="gtfsrt")
    r2 = make_result(predictor_id="baseline:constant_speed", error_s=5.0)
    store.save_results([r1, r2])

    gtfsrt = store.load_results(predictor_id="gtfsrt")
    assert len(gtfsrt) == 1
    assert gtfsrt[0].predictor_id == "gtfsrt"

    baseline = store.load_results(predictor_id="baseline:constant_speed")
    assert len(baseline) == 1
    assert baseline[0].predictor_id == "baseline:constant_speed"

    all_results = store.load_results()
    assert len(all_results) == 2


def test_load_results_unknown_predictor_returns_empty(store: SqliteStorage) -> None:
    store.save_results([make_result()])
    assert store.load_results(predictor_id="nonexistent") == []


def test_load_results_empty_store(store: SqliteStorage) -> None:
    assert store.load_results() == []


def test_result_null_optional_fields(store: SqliteStorage) -> None:
    result = make_result(route_id=None, dist_remaining_m=None, stops_remaining=None)
    store.save_results([result])
    loaded = store.load_results()

    assert len(loaded) == 1
    r = loaded[0]
    assert r.route_id is None
    assert r.dist_remaining_m is None
    assert r.stops_remaining is None


# ── context manager ───────────────────────────────────────────────────────────


def test_context_manager_closes_connection() -> None:
    with SqliteStorage(":memory:") as store:
        store.save_results([make_result()])
        loaded = store.load_results()
    assert len(loaded) == 1


# ── ValidationResult.from_sample_and_arrival ─────────────────────────────────


def test_from_sample_and_arrival_computes_error_and_horizon() -> None:
    sample = make_sample(
        predicted_arrival_epoch=1_700_000_100.0,
        made_at_epoch=1_700_000_000.0,
    )
    arrival = make_arrival(actual_arrival_epoch=1_700_000_090.0)

    result = ValidationResult.from_sample_and_arrival(sample, arrival, route_id="42")

    # error_s = predicted - actual = 100 - 90 = 10
    assert result.error_s == pytest.approx(10.0)
    # horizon_s = actual - made_at = 90 - 0 (relative offset)
    assert result.horizon_s == pytest.approx(1_700_000_090.0 - 1_700_000_000.0)
    assert result.route_id == "42"
    assert result.predictor_id == sample.predictor_id
    assert result.stop_id == sample.stop_id
    assert result.trip_key == sample.trip_key


def test_from_sample_and_arrival_no_route_id() -> None:
    sample = make_sample()
    arrival = make_arrival()
    result = ValidationResult.from_sample_and_arrival(sample, arrival)
    assert result.route_id is None


def test_from_sample_and_arrival_propagates_optional_fields() -> None:
    sample = make_sample(
        dist_remaining_m=150.0,
        stops_remaining=2,
        vehicle_progress_m_at_made=300.0,
    )
    arrival = make_arrival()
    result = ValidationResult.from_sample_and_arrival(sample, arrival)

    assert result.dist_remaining_m == pytest.approx(150.0)
    assert result.stops_remaining == 2
