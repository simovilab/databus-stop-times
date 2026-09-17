"""test_cli — Phase P6 tests for CLI commands, helpers, and parquet round-trip.

Test groups
-----------
1. parse_duration — unit tests for the helper function.
2. sources command — CliRunner smoke tests.
3. Parquet round-trip — export_results → load_results_parquet; edge cases.
4. report command — CliRunner tests including missing-file error path.
5. run core (offline) — _run_validation called directly with ReplayFeedSource +
   SqliteStorage(":memory:"), no network, no sleeping; full path to report.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from etaval.cli.main import _run_validation, app, parse_duration
from etaval.domain.models import ValidationResult
from etaval.engine.runtime import FeedSnapshot, ReplayFeedSource
from etaval.feeds.gtfsrt_json import parse_tu_feed, parse_vp_feed
from etaval.feeds.static import GtfsStaticLoader
from etaval.groundtruth.position_radius import PositionRadiusDetector
from etaval.groundtruth.shape_distance import ShapeDistanceDetector
from etaval.groundtruth.stop_status import StopStatusDetector
from etaval.predictors.constant_speed import ConstantSpeedPredictor
from etaval.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor
from etaval.storage.parquet_export import export_results, load_results_parquet
from etaval.storage.sqlite import SqliteStorage

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"

# ── parse_duration ────────────────────────────────────────────────────────────


def test_parse_duration_minutes() -> None:
    assert parse_duration("30m") == pytest.approx(1800.0)


def test_parse_duration_seconds_suffix() -> None:
    assert parse_duration("90s") == pytest.approx(90.0)


def test_parse_duration_hours() -> None:
    assert parse_duration("1h") == pytest.approx(3600.0)


def test_parse_duration_bare_number() -> None:
    assert parse_duration("45") == pytest.approx(45.0)


def test_parse_duration_bare_float() -> None:
    assert parse_duration("1.5") == pytest.approx(1.5)


def test_parse_duration_minutes_float() -> None:
    assert parse_duration("2.5m") == pytest.approx(150.0)


def test_parse_duration_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration("10x")


def test_parse_duration_empty_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        parse_duration("")


def test_parse_duration_whitespace_only_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        parse_duration("   ")


# ── sources command ───────────────────────────────────────────────────────────


def test_sources_exits_zero() -> None:
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0, result.output


def test_sources_contains_mbta() -> None:
    result = runner.invoke(app, ["sources"])
    assert "mbta" in result.output


def test_sources_contains_bucr() -> None:
    result = runner.invoke(app, ["sources"])
    assert "bucr" in result.output


def test_sources_contains_poll_interval() -> None:
    """Table should show numeric poll intervals."""
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    # MBTA has 15, bUCR has 5 — at least one digit appears.
    assert re.search(r"\d+", result.output)


# ── Parquet round-trip ────────────────────────────────────────────────────────


def _make_result(
    trip_key: str = "test:trip1:v1:20240101",
    predictor_id: str = "gtfsrt",
    route_id: str | None = "r1",
) -> ValidationResult:
    """Return a minimal synthetic ValidationResult for testing."""
    return ValidationResult(
        trip_key=trip_key,
        predictor_id=predictor_id,
        stop_id="s2",
        stop_sequence=2,
        route_id=route_id,
        error_s=10.0,
        horizon_s=120.0,
        dist_remaining_m=500.0,
        stops_remaining=3,
        predicted_arrival_epoch=1110.0,
        actual_arrival_epoch=1100.0,
        made_at_epoch=980.0,
    )


def test_parquet_round_trip_basic(tmp_path: Path) -> None:
    results = [_make_result("k1", "gtfsrt"), _make_result("k2", "baseline:constant_speed")]
    out = tmp_path / "results.parquet"
    export_results(results, out)
    loaded = load_results_parquet(out)
    assert len(loaded) == 2
    assert loaded[0].trip_key == "k1"
    assert loaded[0].predictor_id == "gtfsrt"
    assert loaded[1].trip_key == "k2"
    assert loaded[0].error_s == pytest.approx(10.0)
    assert loaded[0].horizon_s == pytest.approx(120.0)
    assert loaded[0].route_id == "r1"
    assert loaded[0].dist_remaining_m == pytest.approx(500.0)
    assert loaded[0].stops_remaining == 3


def test_parquet_round_trip_none_fields(tmp_path: Path) -> None:
    """Optional fields that are None should round-trip as None."""
    result = _make_result(route_id=None)
    result = result.model_copy(update={"dist_remaining_m": None, "stops_remaining": None})
    out = tmp_path / "none_fields.parquet"
    export_results([result], out)
    loaded = load_results_parquet(out)
    assert len(loaded) == 1
    assert loaded[0].route_id is None
    assert loaded[0].dist_remaining_m is None
    assert loaded[0].stops_remaining is None


def test_parquet_empty_list(tmp_path: Path) -> None:
    """Empty list should write a valid file and read back as empty list."""
    out = tmp_path / "empty.parquet"
    export_results([], out)
    loaded = load_results_parquet(out)
    assert loaded == []


def test_parquet_preserves_all_numeric_fields(tmp_path: Path) -> None:
    result = ValidationResult(
        trip_key="t",
        predictor_id="gtfsrt",
        stop_id="s1",
        stop_sequence=1,
        route_id="r1",
        error_s=-5.5,
        horizon_s=300.0,
        dist_remaining_m=123.456,
        stops_remaining=7,
        predicted_arrival_epoch=2000.0,
        actual_arrival_epoch=2005.5,
        made_at_epoch=1700.0,
    )
    out = tmp_path / "numeric.parquet"
    export_results([result], out)
    loaded = load_results_parquet(out)
    r = loaded[0]
    assert r.error_s == pytest.approx(-5.5)
    assert r.horizon_s == pytest.approx(300.0)
    assert r.dist_remaining_m == pytest.approx(123.456)
    assert r.stops_remaining == 7
    assert r.predicted_arrival_epoch == pytest.approx(2000.0)
    assert r.actual_arrival_epoch == pytest.approx(2005.5)
    assert r.made_at_epoch == pytest.approx(1700.0)


# ── report command ────────────────────────────────────────────────────────────


def test_report_exits_zero(tmp_path: Path) -> None:
    results = [_make_result("k1", "gtfsrt"), _make_result("k2", "baseline:constant_speed")]
    out = tmp_path / "results.parquet"
    export_results(results, out)
    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output


def test_report_shows_predictor_id(tmp_path: Path) -> None:
    results = [_make_result("k1", "gtfsrt"), _make_result("k2", "gtfsrt")]
    out = tmp_path / "results.parquet"
    export_results(results, out)
    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output
    assert "gtfsrt" in result.output


def test_report_shows_mae_value(tmp_path: Path) -> None:
    """MAE for error_s=10.0 → '10.0' should appear in the table."""
    results = [_make_result("k1", "gtfsrt"), _make_result("k2", "gtfsrt")]
    out = tmp_path / "results.parquet"
    export_results(results, out)
    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output
    # MAE = mean(|10.0|) = 10.0; appears as "10.0" in the table.
    assert re.search(r"10\.0", result.output), (
        f"Expected MAE '10.0' in output; got:\n{result.output}"
    )


def test_report_missing_file_nonzero_exit(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.parquet"
    result = runner.invoke(app, ["report", str(missing)])
    assert result.exit_code != 0


def test_report_empty_file_exits_zero(tmp_path: Path) -> None:
    """An empty-row Parquet should exit 0 with a 'no results' message."""
    out = tmp_path / "empty.parquet"
    export_results([], out)
    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output
    assert "No results" in result.output or "0 rows" in result.output


# ── Snapshot builder (mirrors test_engine_replay.py helpers) ──────────────────


def _make_snapshot(
    t: float,
    lat: float,
    lon: float,
    speed: float,
    current_stop_seq: int,
    *,
    tu_predictions: list[tuple[str, int, float]],  # (stop_id, stop_seq, arrival_epoch)
) -> FeedSnapshot:
    """Build one FeedSnapshot from raw-dict representations (offline)."""
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
    """5-tick scenario: vehicle moves from s1 to s4 (matches mini_gtfs.zip)."""
    t0, t1, t2, t3 = 1000.0, 1100.0, 1200.0, 1300.0
    return [
        _make_snapshot(
            t0, 42.3601, -71.0589, 3.0, 1,
            tu_predictions=[("s2", 2, t1), ("s3", 3, t2), ("s4", 4, t3)],
        ),
        _make_snapshot(
            t1, 42.3610, -71.0580, 3.0, 2,
            tu_predictions=[("s3", 3, t2), ("s4", 4, t3)],
        ),
        _make_snapshot(
            t2, 42.3620, -71.0570, 3.0, 3,
            tu_predictions=[("s4", 4, t3)],
        ),
        _make_snapshot(t3, 42.3630, -71.0560, 3.0, 4, tu_predictions=[]),
        _make_snapshot(1400.0, 42.3630, -71.0560, 0.0, 4, tu_predictions=[]),
    ]


# ── run core (offline) ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_validation_core_summary_counts() -> None:
    """_run_validation with ReplayFeedSource: offline, no network, no sleeping."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [
        ShapeDistanceDetector(shape_threshold_m=30.0),
        PositionRadiusDetector(arrival_radius_m=50.0),
        StopStatusDetector(),
    ]
    feed_source = ReplayFeedSource(_build_snapshots())

    summary = await _run_validation(
        feed_source,
        static_loader,
        predictors,
        detectors,
        storage,
        primary_method="shape_distance",
        source_name="test",
    )

    assert summary.ticks == 5
    assert summary.samples_saved > 0, "Expected prediction samples from both predictors"
    assert summary.arrivals_saved > 0, "Expected actual arrivals from detectors"
    assert summary.results_saved > 0, "Expected scored ValidationResults"

    results = storage.load_results()
    assert len(results) == summary.results_saved
    storage.close()


@pytest.mark.asyncio
async def test_run_validation_core_both_predictors_produce_samples() -> None:
    """Both gtfsrt and constant_speed predictors should score results."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [ShapeDistanceDetector(shape_threshold_m=30.0)]
    feed_source = ReplayFeedSource(_build_snapshots())

    await _run_validation(
        feed_source,
        static_loader,
        predictors,
        detectors,
        storage,
        primary_method="shape_distance",
        source_name="test",
    )

    results = storage.load_results()
    predictor_ids_in_results = {r.predictor_id for r in results}
    assert "gtfsrt" in predictor_ids_in_results
    assert "baseline:constant_speed" in predictor_ids_in_results
    storage.close()


@pytest.mark.asyncio
async def test_run_validation_core_to_parquet_and_report(tmp_path: Path) -> None:
    """Full offline path: engine → parquet export → CLI report renders."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [
        ShapeDistanceDetector(shape_threshold_m=30.0),
        PositionRadiusDetector(arrival_radius_m=50.0),
        StopStatusDetector(),
    ]
    feed_source = ReplayFeedSource(_build_snapshots())

    await _run_validation(
        feed_source,
        static_loader,
        predictors,
        detectors,
        storage,
        primary_method="shape_distance",
        source_name="test",
    )

    results = storage.load_results()
    assert len(results) > 0, "Engine produced no results — cannot test report"

    out = tmp_path / "results.parquet"
    export_results(results, out)

    # Parquet file is valid and round-trips.
    loaded = load_results_parquet(out)
    assert len(loaded) == len(results)

    # report command renders without error.
    cli_result = runner.invoke(app, ["report", str(out)])
    assert cli_result.exit_code == 0, cli_result.output

    # Output must contain at least one predictor id.
    assert "gtfsrt" in cli_result.output or "baseline" in cli_result.output, (
        f"Expected a predictor ID in report output; got:\n{cli_result.output}"
    )

    # Output must contain a MAE value (a decimal number).
    assert re.search(r"\d+\.\d", cli_result.output), (
        f"Expected numeric MAE in report output; got:\n{cli_result.output}"
    )

    storage.close()


# ── run --seed (offline replay/fixture path) — A7/A8 CLI stamp + sidecars ────


def test_run_command_with_seed_writes_sidecars_and_prints_stamp(tmp_path: Path) -> None:
    """Full CliRunner invocation of ``etaval run --replay-vp ... --seed ...``.

    Exercises the real ``run`` command end-to-end (offline archive replay, no
    network) and asserts both sidecars land on disk and the one-line
    provenance stamp is printed.
    """
    from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out = tmp_path / "results.parquet"

    result = runner.invoke(
        app,
        [
            "run",
            "--source", "mbta",
            "--replay-vp", str(vp_dir),
            "--replay-gtfs", str(_MINI_GTFS),
            "--out", str(out),
            "--seed", "1234",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Provenance: etaval" in result.output
    assert "seed=1234" in result.output

    prov_sidecar = Path(f"{out}.provenance.json")
    summary_sidecar = Path(f"{out}.summary.json")
    assert prov_sidecar.is_file()
    assert summary_sidecar.is_file()

    import json

    prov_data = json.loads(prov_sidecar.read_text())
    assert prov_data["seed"] == 1234
    assert prov_data["source_name"] == "mbta"

    summary_data = json.loads(summary_sidecar.read_text())
    assert summary_data["provenance"]["seed"] == 1234
    assert "inference_latency" in summary_data


def test_run_command_without_out_skips_sidecars(tmp_path: Path) -> None:
    """No --out means no sidecars are written (mirrors parquet-export gating)."""
    from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        [
            "run",
            "--source", "mbta",
            "--replay-vp", str(vp_dir),
            "--replay-gtfs", str(_MINI_GTFS),
            "--seed", "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Provenance: etaval" in result.output
    assert not (tmp_path / "results.parquet.provenance.json").exists()
