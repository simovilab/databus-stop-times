"""test_engine_replay — offline replay test for engine/runtime.py. Phase P5.

Scenario
--------
Uses tests/fixtures/mini_gtfs.zip (trip1, shape1, 4 stops s1–s4).
Builds a synthetic multi-tick replay:
  - Vehicle starts at s1 and advances through all 4 stops over 5 ticks.
  - Each tick provides one VP entity (vehicle position advancing along shape).
  - Each tick provides one TU entity predicting arrivals for upcoming stops.
  - header_timestamp advances by 100 s per tick (t0=1000, t1=1100, ..., t4=1400).

Assertions
----------
- PredictionSamples non-empty for BOTH predictors.
- ActualArrivals fire for stops the vehicle crossed (shape_distance detector).
- ValidationResults populated; error_s and horizon_s are finite.
- At least one result in a recognisable horizon bucket (>10m for early ticks).
- MetricsReport built from results is non-empty with per-predictor MAE.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from etaval.engine.runtime import FeedSnapshot, ReplayFeedSource, run_engine
from etaval.feeds.gtfsrt_json import parse_tu_feed, parse_vp_feed
from etaval.feeds.static import GtfsStaticLoader
from etaval.groundtruth.position_radius import PositionRadiusDetector
from etaval.groundtruth.shape_distance import ShapeDistanceDetector
from etaval.predictors.constant_speed import ConstantSpeedPredictor
from etaval.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor
from etaval.scoring.metrics import compute_metrics
from etaval.storage.sqlite import SqliteStorage

# ── Fixture paths ─────────────────────────────────────────────────────────────

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"

# ── Stop coordinates from mini_gtfs.zip ───────────────────────────────────────
# stops.txt:
#   s1  42.3601  -71.0589
#   s2  42.3610  -71.0580
#   s3  42.3620  -71.0570
#   s4  42.3630  -71.0560

_STOPS = [
    ("s1", 42.3601, -71.0589, 1),
    ("s2", 42.3610, -71.0580, 2),
    ("s3", 42.3620, -71.0570, 3),
    ("s4", 42.3630, -71.0560, 4),
]

# ── Snapshot builder ──────────────────────────────────────────────────────────


def _make_snapshot(
    t: float,
    lat: float,
    lon: float,
    speed: float,
    current_stop_seq: int,
    *,
    tu_predictions: list[tuple[str, int, float]],  # (stop_id, stop_seq, arrival_epoch)
) -> FeedSnapshot:
    """Build one FeedSnapshot from raw dict representations."""
    vp_dict = {
        "header": {"timestamp": t},
        "entity": [
            {
                "id": "e1",
                "vehicle": {
                    "trip": {
                        "trip_id": "trip1",
                        "route_id": "r1",
                        "direction_id": 0,
                        "start_date": "20240101",
                    },
                    "vehicle": {"id": "v1"},
                    "position": {"latitude": lat, "longitude": lon, "speed": speed},
                    "timestamp": t,
                    "current_status": "IN_TRANSIT_TO",
                    "current_stop_sequence": current_stop_seq,
                },
            }
        ],
    }

    stu_entries = [
        {
            "stop_sequence": seq,
            "stop_id": sid,
            "arrival": {"time": arr_t},
            "departure": {"time": arr_t + 30},
        }
        for sid, seq, arr_t in tu_predictions
    ]

    tu_dict = {
        "header": {"timestamp": t},
        "entity": [
            {
                "id": "e1",
                "trip_update": {
                    "trip": {
                        "trip_id": "trip1",
                        "route_id": "r1",
                        "direction_id": 0,
                        "start_date": "20240101",
                    },
                    "stop_time_update": stu_entries,
                },
            }
        ],
    }

    vp_ts, vp_records = parse_vp_feed(vp_dict)
    tu_ts, tu_records = parse_tu_feed(tu_dict)
    return FeedSnapshot(
        header_timestamp=vp_ts,
        vp_records=vp_records,
        tu_header_timestamp=tu_ts,
        tu_records=tu_records,
    )


def _build_snapshots() -> list[FeedSnapshot]:
    """Build a 5-tick replay: vehicle moves from s1 to s4 over 400 s."""
    # Arrival epochs: each stop is 100 s apart.
    t0 = 1000.0   # t=1000: vehicle is at s1
    t1 = 1100.0   # t=1100: vehicle is at s2
    t2 = 1200.0   # t=1200: vehicle is at s3
    t3 = 1300.0   # t=1300: vehicle is at s4

    snaps = [
        # Tick 0: vehicle at s1, TU predicts s2/s3/s4
        _make_snapshot(
            t=t0,
            lat=42.3601, lon=-71.0589, speed=3.0,
            current_stop_seq=1,
            tu_predictions=[
                ("s2", 2, t1),
                ("s3", 3, t2),
                ("s4", 4, t3),
            ],
        ),
        # Tick 1: vehicle at s2, TU predicts s3/s4
        _make_snapshot(
            t=t1,
            lat=42.3610, lon=-71.0580, speed=3.0,
            current_stop_seq=2,
            tu_predictions=[
                ("s3", 3, t2),
                ("s4", 4, t3),
            ],
        ),
        # Tick 2: vehicle at s3, TU predicts s4
        _make_snapshot(
            t=t2,
            lat=42.3620, lon=-71.0570, speed=3.0,
            current_stop_seq=3,
            tu_predictions=[
                ("s4", 4, t3),
            ],
        ),
        # Tick 3: vehicle at s4, no TU remaining
        _make_snapshot(
            t=t3,
            lat=42.3630, lon=-71.0560, speed=3.0,
            current_stop_seq=4,
            tu_predictions=[],
        ),
        # Tick 4: vehicle past s4 (same position, different timestamp)
        _make_snapshot(
            t=1400.0,
            lat=42.3630, lon=-71.0560, speed=0.0,
            current_stop_seq=4,
            tu_predictions=[],
        ),
    ]
    return snaps


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_replay_produces_samples_and_results() -> None:
    """Full engine replay: samples, arrivals, results, and metrics all populate."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [
        ShapeDistanceDetector(shape_threshold_m=30.0),
        PositionRadiusDetector(arrival_radius_m=50.0),
    ]

    snapshots = _build_snapshots()
    feed_source = ReplayFeedSource(snapshots)

    summary = await run_engine(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=predictors,
        detectors=detectors,
        storage=storage,
        primary_method="shape_distance",
        source_name="test",
    )

    # ── Engine ran all ticks ──────────────────────────────────────────────────
    assert summary.ticks == len(snapshots)

    # ── Samples: both predictors produced predictions ─────────────────────────
    all_samples = storage._conn.execute(
        "SELECT DISTINCT predictor_id FROM prediction_samples"
    ).fetchall()
    predictor_ids_in_db = {row[0] for row in all_samples}
    assert "gtfsrt" in predictor_ids_in_db, (
        "GtfsRtTripUpdatesPredictor should have produced samples"
    )
    assert "baseline:constant_speed" in predictor_ids_in_db, (
        "ConstantSpeedPredictor should have produced samples"
    )

    # ── Arrivals: shape_distance detector should have fired ───────────────────
    arrivals = storage._conn.execute(
        "SELECT * FROM actual_arrivals WHERE method='shape_distance'"
    ).fetchall()
    assert len(arrivals) > 0, "ShapeDistanceDetector should fire for passed stops"

    # Epochs must be finite positive floats.
    for row in arrivals:
        assert row["actual_arrival_epoch"] > 0
        assert row["actual_arrival_epoch"] < float("inf")

    # ── Results: validation results populated ─────────────────────────────────
    results = storage.load_results()
    assert len(results) > 0, "At least one ValidationResult should be scored"

    for r in results:
        assert isinstance(r.error_s, float)
        assert isinstance(r.horizon_s, float)
        assert r.horizon_s > 0 or r.horizon_s == pytest.approx(0.0)  # non-negative
        assert not (r.error_s != r.error_s)   # not NaN
        assert not (r.horizon_s != r.horizon_s)

    # ── At least one result should have a recognisable horizon bucket ─────────
    from etaval.scoring.metrics import _assign_bucket

    buckets_seen = {_assign_bucket(r.horizon_s) for r in results}
    # The first tick has horizon ≈ 100–300 s (2-5m or 5-10m range).
    all_bucket_labels = {"<2m", "2-5m", "5-10m", ">10m"}
    assert buckets_seen & all_bucket_labels, f"Expected at least one bucket, got {buckets_seen}"

    # ── Metrics report non-empty and has per-predictor MAE ────────────────────
    report = compute_metrics(results)
    assert len(report.per_predictor) > 0, "Metrics report should have at least one predictor row"
    for m in report.per_predictor:
        assert m.mae >= 0.0
        assert m.count > 0

    # ── Summary counts are consistent ─────────────────────────────────────────
    assert summary.samples_saved > 0
    assert summary.arrivals_saved > 0
    assert summary.results_saved > 0


@pytest.mark.asyncio
async def test_replay_deduplicates_arrivals() -> None:
    """Each (trip_key, stop_id, method) arrival is persisted at most once."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor()]
    detectors = [ShapeDistanceDetector(shape_threshold_m=30.0)]

    # Run the same snapshot twice to verify dedup.
    snap = _build_snapshots()[0]  # tick 0: vehicle at s1
    # Add a second tick at the same position.
    feed_source = ReplayFeedSource([snap, snap])

    await run_engine(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=predictors,
        detectors=detectors,
        storage=storage,
        primary_method="shape_distance",
        source_name="test",
    )

    rows = storage._conn.execute(
        "SELECT trip_key, stop_id, method, COUNT(*) as cnt "
        "FROM actual_arrivals GROUP BY trip_key, stop_id, method HAVING cnt > 1"
    ).fetchall()
    assert len(rows) == 0, "Duplicate arrivals should be deduplicated by the engine"


@pytest.mark.asyncio
async def test_replay_source_exhausts_correctly() -> None:
    """ReplayFeedSource yields exactly len(snapshots) ticks and then stops."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    snaps = _build_snapshots()

    summary = await run_engine(
        feed_source=ReplayFeedSource(snaps),
        static_loader=static_loader,
        predictors=[],
        detectors=[],
        storage=storage,
        source_name="test",
    )
    assert summary.ticks == len(snaps)


@pytest.mark.asyncio
async def test_max_ticks_cap() -> None:
    """max_ticks parameter stops the engine after the specified number of ticks."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    snaps = _build_snapshots()  # 5 snapshots

    summary = await run_engine(
        feed_source=ReplayFeedSource(snaps),
        static_loader=static_loader,
        predictors=[],
        detectors=[],
        storage=storage,
        max_ticks=2,
        source_name="test",
    )
    assert summary.ticks == 2


# ── on_tick streaming hook ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_tick_hook_fires_per_tick() -> None:
    """on_tick coroutine receives one TickEvent per engine tick."""
    import math

    from etaval.engine.runtime import TickEvent

    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [ShapeDistanceDetector(shape_threshold_m=30.0)]

    snapshots = _build_snapshots()  # 5 ticks
    feed_source = ReplayFeedSource(snapshots)

    captured: list[TickEvent] = []

    async def on_tick(event: TickEvent) -> None:
        captured.append(event)

    summary = await run_engine(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=predictors,
        detectors=detectors,
        storage=storage,
        primary_method="shape_distance",
        source_name="test",
        on_tick=on_tick,
    )

    # One event per tick.
    assert len(captured) == summary.ticks == len(snapshots)

    # Tick numbers are 1-based and strictly increasing.
    assert [e.tick for e in captured] == list(range(1, len(snapshots) + 1))

    # Cumulative counts are monotonically non-decreasing.
    for prev, curr in zip(captured, captured[1:], strict=False):
        assert curr.samples_saved >= prev.samples_saved
        assert curr.arrivals_saved >= prev.arrivals_saved
        assert curr.results_saved >= prev.results_saved

    # Final event's cumulative counts match the RunSummary totals.
    last = captured[-1]
    assert last.samples_saved == summary.samples_saved
    assert last.arrivals_saved == summary.arrivals_saved
    assert last.results_saved == summary.results_saved

    # At least one tick carries new_results (this tick's scored results).
    ticks_with_results = [e for e in captured if e.new_results]
    assert len(ticks_with_results) > 0, "Expected at least one tick with ValidationResults"

    # At least one tick has vehicle pings, and every ping has finite lat/lon.
    ticks_with_vehicles = [e for e in captured if e.vehicles]
    assert len(ticks_with_vehicles) > 0, "Expected at least one tick with vehicle pings"
    for event in captured:
        for ping in event.vehicles:
            assert math.isfinite(ping.lat)
            assert math.isfinite(ping.lon)
            if ping.progress_m is not None:
                assert math.isfinite(ping.progress_m)


@pytest.mark.asyncio
async def test_on_tick_none_is_backward_compatible() -> None:
    """on_tick=None (default) produces the same behaviour as the old signature."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [ShapeDistanceDetector(shape_threshold_m=30.0)]

    snapshots = _build_snapshots()

    summary = await run_engine(
        feed_source=ReplayFeedSource(snapshots),
        static_loader=static_loader,
        predictors=predictors,
        detectors=detectors,
        storage=storage,
        primary_method="shape_distance",
        source_name="test",
        on_tick=None,
    )

    assert summary.ticks == len(snapshots)
    assert summary.samples_saved > 0
    assert summary.arrivals_saved > 0
    assert summary.results_saved > 0
