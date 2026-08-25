"""Tests for the BUCR trip-inference quality report's pure aggregation helpers.

All synthetic -- no network, no filesystem, deterministic. Exercises
``bucr_quality_report``'s aggregation functions directly against hand-built
``TraceCleaningReport``/``InferenceStats`` instances and small ``out_df``
frames shaped like ``infer_bucr_trips`` output, rather than running the full
pipeline (that needs the real navsat/GTFS snapshots and is a script
invocation, not a unit test -- see the module for how those are wired).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from feature_engineering.bucr_quality_report import (
    SELF_LOOP_SHAPE_IDS,
    SELF_LOOP_SIBLING_SHAPES,
    CleaningSummary,
    MatchRateSummary,
    parse_trip_id,
    render_markdown_report,
    QualityReport,
    sanity_check,
    self_loop_measurement,
    spot_check_table,
    summarize_cleaning,
    summarize_match_rate,
    trip_instance_table,
    trips_per_day_table,
)
from feature_engineering.bucr_trip_inference import InferenceStats, OUTPUT_COLUMNS
from feature_engineering.navsat_cleaning import TraceCleaningReport


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _cleaning_report(rows_in=100, kept=90, stale=5, null_coord=3, zero_coord=1, oor=1, bad_ts=0, threshold=900.0):
    return TraceCleaningReport(
        rows_in=rows_in,
        dropped_bad_timestamp=bad_ts,
        dropped_stale=stale,
        dropped_null_coord=null_coord,
        dropped_zero_coord=zero_coord,
        dropped_out_of_range=oor,
        rows_kept=kept,
        staleness_threshold_seconds=threshold,
    )


def _inference_stats(
    points_in=100, assigned=80, parked=15, anomaly=5, trips=4, per_variant=None,
    dropped_as_noise=0, instances_dropped_as_noise=0,
):
    if per_variant is None:
        per_variant = (("R1", 0, "shapeA", trips),)
    return InferenceStats(
        points_in=points_in,
        points_assigned=assigned,
        points_parked_ambiguous=parked,
        points_anomaly_rejected=anomaly,
        points_dropped_as_noise=dropped_as_noise,
        trips_inferred=trips,
        instances_dropped_as_noise=instances_dropped_as_noise,
        trips_per_direction_variant=per_variant,
    )


def _trip_id(vehicle_id: str, route_id: str, direction_id: int, shape_id: str, start: dt.datetime) -> str:
    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    return f"bucr:{vehicle_id}:{route_id}:{direction_id}:{shape_id}:{start_str}"


def _synthetic_out_df(rows: list[dict]) -> pd.DataFrame:
    """Build an out_df shaped like ``infer_bucr_trips`` output.

    Each row dict: vehicle_id, route_id, direction_id, shape_id, start (dt),
    n_points, seconds_between_points, n_stops (stop ids cycle 0..n_stops-1).
    """
    frames = []
    for r in rows:
        trip_id = _trip_id(r["vehicle_id"], r["route_id"], r["direction_id"], r["shape_id"], r["start"])
        n = r["n_points"]
        step = r.get("seconds_between_points", 30)
        n_stops = r.get("n_stops", 3)
        ts = [r["start"] + dt.timedelta(seconds=step * i) for i in range(n)]
        stop_ids = [f"stop{i % n_stops}" for i in range(n)]
        frames.append(
            pd.DataFrame(
                {
                    "trip_id": trip_id,
                    "vehicle_id": r["vehicle_id"],
                    "ts": pd.to_datetime(ts, utc=True),
                    "lat": 9.9,
                    "lon": -84.0,
                    "bearing": np.nan,
                    "speed": 5.0,
                    "current_stop_sequence": list(range(n)),
                    "current_status": "IN_TRANSIT_TO",
                    "stop_id": stop_ids,
                    "route_id": r["route_id"],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.reindex(columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# CleaningSummary
# ---------------------------------------------------------------------------


def test_summarize_cleaning_sums_across_days():
    reports = [_cleaning_report(rows_in=100, kept=90, stale=5, null_coord=3, zero_coord=1, oor=1),
               _cleaning_report(rows_in=50, kept=45, stale=2, null_coord=2, zero_coord=0, oor=1)]
    summary = summarize_cleaning(reports)
    assert summary.days == 2
    assert summary.rows_in == 150
    assert summary.rows_kept == 135
    assert summary.dropped_stale == 7
    assert summary.dropped_total == 15
    assert summary.drop_rate == pytest.approx(15 / 150)


def test_summarize_cleaning_drop_rate_by_reason_sums_to_drop_rate():
    reports = [_cleaning_report()]
    summary = summarize_cleaning(reports)
    by_reason = summary.drop_rate_by_reason()
    assert sum(by_reason.values()) == pytest.approx(summary.drop_rate)


def test_summarize_cleaning_empty_raises():
    with pytest.raises(ValueError):
        summarize_cleaning([])


def test_summarize_cleaning_inconsistent_thresholds_raises():
    reports = [_cleaning_report(threshold=900.0), _cleaning_report(threshold=600.0)]
    with pytest.raises(ValueError):
        summarize_cleaning(reports)


def test_summarize_cleaning_zero_rows_in_gives_zero_drop_rate():
    summary = summarize_cleaning([_cleaning_report(rows_in=0, kept=0, stale=0, null_coord=0, zero_coord=0, oor=0)])
    assert summary.drop_rate == 0.0
    assert all(v == 0.0 for v in summary.drop_rate_by_reason().values())


# ---------------------------------------------------------------------------
# MatchRateSummary
# ---------------------------------------------------------------------------


def test_summarize_match_rate_sums_across_days():
    stats = [
        _inference_stats(points_in=100, assigned=80, parked=15, anomaly=5, trips=4,
                          per_variant=(("R1", 0, "shapeA", 4),)),
        _inference_stats(points_in=50, assigned=40, parked=5, anomaly=5, trips=2,
                          per_variant=(("R1", 0, "shapeA", 1), ("R1", 1, "shapeB", 1))),
    ]
    summary = summarize_match_rate(stats)
    assert summary.days == 2
    assert summary.points_in == 150
    assert summary.points_assigned == 120
    assert summary.points_parked_ambiguous == 20
    assert summary.points_anomaly_rejected == 10
    assert summary.trips_inferred == 6
    assert summary.assigned_rate == pytest.approx(120 / 150)
    assert summary.parked_rate == pytest.approx(20 / 150)
    assert summary.anomaly_rate == pytest.approx(10 / 150)
    # per_variant merges the (R1, 0, shapeA) key across days: 4 + 1 = 5
    per_variant = dict(((r, d, s), c) for r, d, s, c in summary.per_variant)
    assert per_variant[("R1", 0, "shapeA")] == 5
    assert per_variant[("R1", 1, "shapeB")] == 1


def test_summarize_match_rate_rates_are_fractions_that_sum_to_le_one():
    stats = [_inference_stats()]
    summary = summarize_match_rate(stats)
    assert summary.assigned_rate + summary.parked_rate + summary.anomaly_rate == pytest.approx(1.0)


def test_summarize_match_rate_empty_raises():
    with pytest.raises(ValueError):
        summarize_match_rate([])


def test_summarize_match_rate_zero_points_in_gives_zero_rates():
    summary = summarize_match_rate([_inference_stats(points_in=0, assigned=0, parked=0, anomaly=0, trips=0, per_variant=())])
    assert summary.assigned_rate == 0.0
    assert summary.parked_rate == 0.0
    assert summary.anomaly_rate == 0.0


# ---------------------------------------------------------------------------
# parse_trip_id / trip_instance_table / trips_per_day_table
# ---------------------------------------------------------------------------


def test_parse_trip_id_round_trips_the_documented_format():
    start = dt.datetime(2026, 8, 18, 12, 30, 45, tzinfo=dt.timezone.utc)
    trip_id = _trip_id("299-1015", "R1", 1, "shapeA", start)
    meta = parse_trip_id(trip_id)
    assert meta == {
        "vehicle_id": "299-1015",
        "route_id": "R1",
        "direction_id": 1,
        "shape_id": "shapeA",
        "start_ts": "20260818T123045Z",
    }


def test_parse_trip_id_rejects_malformed_input():
    with pytest.raises(ValueError):
        parse_trip_id("not-a-bucr-trip-id")


def test_trip_instance_table_one_row_per_trip_with_expected_stats():
    start = dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
    out_df = _synthetic_out_df(
        [
            {"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA",
             "start": start, "n_points": 10, "seconds_between_points": 30, "n_stops": 5},
        ]
    )
    table = trip_instance_table(out_df)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["vehicle_id"] == "V1"
    assert row["route_id"] == "R1"
    assert row["direction_id"] == 0
    assert row["shape_id"] == "shapeA"
    assert row["n_points"] == 10
    assert row["n_stops_covered"] == 5
    assert row["duration_s"] == pytest.approx(30 * 9)


def test_trip_instance_table_empty_input_returns_empty_with_columns():
    table = trip_instance_table(pd.DataFrame(columns=OUTPUT_COLUMNS))
    assert table.empty
    assert "trip_id" in table.columns


def test_trips_per_day_table_counts_correctly():
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    out_df = _synthetic_out_df(
        [
            {"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA", "start": start, "n_points": 5},
            {"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA",
             "start": start + dt.timedelta(hours=2), "n_points": 5},
            {"vehicle_id": "V2", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA", "start": start, "n_points": 5},
        ]
    )
    table = trip_instance_table(out_df)
    per_day = trips_per_day_table(table)
    v1_row = per_day[(per_day["vehicle_id"] == "V1")]
    v2_row = per_day[(per_day["vehicle_id"] == "V2")]
    assert v1_row["n_trips"].iloc[0] == 2
    assert v2_row["n_trips"].iloc[0] == 1


def test_trips_per_day_table_empty_input():
    per_day = trips_per_day_table(trip_instance_table(pd.DataFrame(columns=OUTPUT_COLUMNS)))
    assert per_day.empty
    assert "n_trips" in per_day.columns


# ---------------------------------------------------------------------------
# self_loop_measurement
# ---------------------------------------------------------------------------


def test_self_loop_measurement_flags_shorter_fragmented_trips():
    """Synthetic scenario: the flagged shape produces MORE, SHORTER instances
    per vehicle-day than its con_milla sibling for the same vehicle -- the
    exact signature the self-loop bug is expected to leave. Inflation
    estimate should come back > 1."""
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    flagged_shape = SELF_LOOP_SHAPE_IDS[0]
    sibling_shape = SELF_LOOP_SIBLING_SHAPES[flagged_shape]

    rows = []
    # Flagged shape: 3 short fragments for V1 on one service day.
    for i in range(3):
        rows.append({"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": flagged_shape,
                      "start": start + dt.timedelta(minutes=15 * i), "n_points": 5, "seconds_between_points": 20})
    # Sibling shape: 1 long, unfragmented trip for V1 on the same day.
    rows.append({"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": sibling_shape,
                  "start": start + dt.timedelta(hours=3), "n_points": 40, "seconds_between_points": 20})
    # An unrelated "other" shape trip, for the others-group comparison.
    rows.append({"vehicle_id": "V1", "route_id": "R2", "direction_id": 0, "shape_id": "unrelated_shape",
                  "start": start + dt.timedelta(hours=5), "n_points": 30, "seconds_between_points": 20})

    out_df = _synthetic_out_df(rows)
    table = trip_instance_table(out_df)
    result = self_loop_measurement(table)

    assert result["flagged"]["n_trips"] == 3
    assert result["siblings"]["n_trips"] == 1
    assert result["others"]["n_trips"] == 1
    # Flagged trips are shorter (fewer points / shorter duration) than sibling.
    assert result["flagged"]["duration_s"]["median"] < result["siblings"]["duration_s"]["median"]
    assert result["flagged"]["n_points"]["median"] < result["siblings"]["n_points"]["median"]
    # 3 flagged trips vs 1 sibling trip on the same vehicle-day -> inflation ~3x.
    assert result["inflation_estimate"][flagged_shape] == pytest.approx(3.0)


def test_self_loop_measurement_nan_inflation_when_no_sibling_trips():
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    flagged_shape = SELF_LOOP_SHAPE_IDS[0]
    out_df = _synthetic_out_df(
        [{"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": flagged_shape,
          "start": start, "n_points": 5}]
    )
    table = trip_instance_table(out_df)
    result = self_loop_measurement(table)
    assert np.isnan(result["inflation_estimate"][flagged_shape])


def test_self_loop_measurement_empty_table():
    result = self_loop_measurement(trip_instance_table(pd.DataFrame(columns=OUTPUT_COLUMNS)))
    assert result["flagged"]["n_trips"] == 0
    assert result["siblings"]["n_trips"] == 0
    assert result["others"]["n_trips"] == 0
    assert result["inflation_estimate"] == {}


# ---------------------------------------------------------------------------
# spot_check_table (uses a fake BucrGtfs-like object)
# ---------------------------------------------------------------------------


class _FakeGtfs:
    def __init__(self, trips: pd.DataFrame, stop_times: pd.DataFrame):
        self.trips = trips
        self.stop_times = stop_times


def _fake_gtfs() -> _FakeGtfs:
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3"],
            "route_id": ["R1", "R1", "R1"],
            "direction_id": [0, 0, 0],
            "shape_id": ["shapeA", "shapeA", "shapeA"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3"],
            "stop_sequence": [1, 1, 1],
            "departure_time": ["06:00:00", "06:30:00", "07:00:00"],
        }
    )
    return _FakeGtfs(trips, stop_times)


def test_spot_check_table_finds_nearest_scheduled_start():
    start = dt.datetime(2026, 8, 18, 12, 3, 0, tzinfo=dt.timezone.utc)  # 06:03 America/Costa_Rica (UTC-6)
    out_df = _synthetic_out_df(
        [{"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA", "start": start, "n_points": 5}]
    )
    table = trip_instance_table(out_df)
    result = spot_check_table(table, _fake_gtfs(), n=10, seed=0)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["nearest_scheduled_start"] == "06:00:00"
    assert row["delta_minutes"] == pytest.approx(3.0)
    assert row["n_scheduled_trips_same_direction"] == 3


def test_spot_check_table_samples_at_most_n():
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    rows = [
        {"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA",
         "start": start + dt.timedelta(hours=i), "n_points": 5}
        for i in range(5)
    ]
    table = trip_instance_table(_synthetic_out_df(rows))
    result = spot_check_table(table, _fake_gtfs(), n=2, seed=0)
    assert len(result) == 2


def test_spot_check_table_empty_input():
    result = spot_check_table(trip_instance_table(pd.DataFrame(columns=OUTPUT_COLUMNS)), _fake_gtfs())
    assert result.empty
    assert "delta_minutes" in result.columns


# ---------------------------------------------------------------------------
# sanity_check
# ---------------------------------------------------------------------------


def test_sanity_check_reports_timetable_size_and_max_trips():
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    rows = [
        {"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA",
         "start": start + dt.timedelta(hours=i), "n_points": 5}
        for i in range(4)
    ]
    table = trip_instance_table(_synthetic_out_df(rows))
    per_day = trips_per_day_table(table)
    result = sanity_check(per_day, _fake_gtfs())
    assert result["timetable_trip_count"] == 3
    assert result["max_trips_single_vehicle_day_shape"] == 4
    assert result["vehicle_days_observed"] == 1


def test_sanity_check_empty_trips_per_day():
    result = sanity_check(pd.DataFrame(columns=["service_day", "vehicle_id", "route_id", "direction_id", "shape_id", "n_trips"]), _fake_gtfs())
    assert result["max_trips_single_vehicle_day_shape"] == 0
    assert result["vehicle_days_observed"] == 0


# ---------------------------------------------------------------------------
# render_markdown_report (smoke: produces a non-empty string covering all sections)
# ---------------------------------------------------------------------------


def test_render_markdown_report_includes_all_required_sections():
    start = dt.datetime(2026, 8, 18, 6, 0, 0, tzinfo=dt.timezone.utc)
    out_df = _synthetic_out_df(
        [{"vehicle_id": "V1", "route_id": "R1", "direction_id": 0, "shape_id": "shapeA", "start": start, "n_points": 5}]
    )
    table = trip_instance_table(out_df)
    per_day = trips_per_day_table(table)
    report = QualityReport(
        cleaning=summarize_cleaning([_cleaning_report()]),
        match_rate=summarize_match_rate([_inference_stats()]),
        trip_table=table,
        trips_per_day=per_day,
        self_loop=self_loop_measurement(table),
        spot_check=spot_check_table(table, _fake_gtfs()),
        sanity=sanity_check(per_day, _fake_gtfs()),
    )
    md = render_markdown_report(report, verdict="TEST VERDICT TEXT")
    for heading in ["## Cleaning", "## Inference match rate", "## Trips/day", "## Self-loop artifact",
                     "## Spot-check vs. timetable", "## Sanity", "## Verdict"]:
        assert heading in md
    assert "TEST VERDICT TEXT" in md
