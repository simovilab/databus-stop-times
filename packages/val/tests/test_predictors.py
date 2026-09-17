"""Tests for Phase P3 — predictors.

Coverage
--------
- GtfsRtTripUpdatesPredictor: stop matching (stop_id / stop_sequence fallback),
  skipping unmatched / null-arrival stops, header_timestamp → made_at_epoch.
- ConstantSpeedPredictor: history-based speed, reported-speed fallback,
  guards (speed ≤ 0, no progress_m, stop behind vehicle), multi-stop ordering.
- StopPrediction.to_sample: field mapping to PredictionSample.
"""

from __future__ import annotations

import pytest

from stoptimes.core.domain.models import PredictionSample, Stop, TripInstance, VehicleState
from stoptimes.core.realtime.gtfsrt_json import StopTimeUpdate
from stoptimes.val.predictors.base import StopPrediction
from stoptimes.val.predictors.constant_speed import ConstantSpeedPredictor
from stoptimes.val.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor, TripUpdateInput

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

TRIP = TripInstance(
    source="test",
    trip_id="t1",
    vehicle_id="v1",
    route_id="r1",
    direction_id=0,
    shape_id="s1",
    start_date="20240101",
    started_at=1_000_000.0,
)

BASE_STATE = VehicleState(
    epoch=1_000_100.0,
    lat=10.0,
    lon=10.0,
    progress_m=200.0,
    speed=10.0,
)


def _stop(stop_id: str, stop_sequence: int, progress_m: float | None = None) -> Stop:
    return Stop(
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        lat=10.0,
        lon=10.0,
        progress_m=progress_m,
    )


# ---------------------------------------------------------------------------
# GtfsRtTripUpdatesPredictor
# ---------------------------------------------------------------------------


class TestGtfsRtPredictor:
    PREDICTOR = GtfsRtTripUpdatesPredictor()

    def _tu(
        self,
        header_ts: float,
        entries: list[tuple[str | None, int | None, float | None]],
    ) -> TripUpdateInput:
        """Build a TripUpdateInput from (stop_id, stop_sequence, arrival_time) tuples."""
        return TripUpdateInput(
            header_timestamp=header_ts,
            stop_time_updates=[
                StopTimeUpdate(
                    stop_sequence=seq,
                    stop_id=sid,
                    arrival_time=arr,
                    departure_time=None,
                )
                for sid, seq, arr in entries
            ],
        )

    def test_matched_stops_emit_predictions(self) -> None:
        """Stops matched by stop_id get a StopPrediction with correct fields."""
        upcoming = [_stop("s1", 1, 300.0), _stop("s2", 2, 500.0)]
        tu = self._tu(
            1_000_050.0,
            [("s1", 1, 1_000_300.0), ("s2", 2, 1_000_500.0)],
        )
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, tu)

        assert len(preds) == 2
        assert preds[0].stop_id == "s1"
        assert preds[0].stop_sequence == 1
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_300.0)
        assert preds[0].made_at_epoch == pytest.approx(1_000_050.0)
        assert preds[0].stops_remaining == 2  # 2 upcoming stops remain at index 0
        assert preds[1].stop_id == "s2"
        assert preds[1].predicted_arrival_epoch == pytest.approx(1_000_500.0)
        assert preds[1].stops_remaining == 1  # only 1 at index 1

    def test_unmatched_stop_is_skipped(self) -> None:
        """Upcoming stop with no TU entry is silently omitted."""
        upcoming = [_stop("s1", 1, 300.0), _stop("s3", 3, 600.0)]
        tu = self._tu(1_000_050.0, [("s1", 1, 1_000_300.0)])  # s3 absent
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, tu)

        assert len(preds) == 1
        assert preds[0].stop_id == "s1"

    def test_null_arrival_time_skipped(self) -> None:
        """StopTimeUpdate with arrival_time=None is silently skipped."""
        upcoming = [_stop("s1", 1, 300.0)]
        tu = self._tu(1_000_050.0, [("s1", 1, None)])
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, tu)
        assert preds == []

    def test_fallback_to_stop_sequence(self) -> None:
        """When stop_id is absent in the TU entry, match by stop_sequence."""
        upcoming = [_stop("s1", 1, 300.0)]
        tu = self._tu(1_000_050.0, [(None, 1, 1_000_300.0)])  # no stop_id
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, tu)

        assert len(preds) == 1
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_300.0)

    def test_none_trip_updates_returns_empty(self) -> None:
        """None trip_updates → empty list, no crash."""
        upcoming = [_stop("s1", 1, 300.0)]
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, None)
        assert preds == []

    def test_wrong_type_trip_updates_returns_empty(self) -> None:
        """Non-TripUpdateInput trip_updates → empty list, no crash."""
        preds = self.PREDICTOR.predict(TRIP, [_stop("s1", 1, 300.0)], BASE_STATE, {"raw": "data"})
        assert preds == []

    def test_empty_upcoming_stops(self) -> None:
        """Empty upcoming_stops → empty list regardless of TU content."""
        tu = self._tu(1_000_050.0, [("s1", 1, 1_000_300.0)])
        preds = self.PREDICTOR.predict(TRIP, [], BASE_STATE, tu)
        assert preds == []

    def test_dist_remaining_m_is_none(self) -> None:
        """TU predictor does not set dist_remaining_m (feed carries no shape distance)."""
        upcoming = [_stop("s1", 1, 300.0)]
        tu = self._tu(1_000_050.0, [("s1", 1, 1_000_300.0)])
        preds = self.PREDICTOR.predict(TRIP, upcoming, BASE_STATE, tu)
        assert preds[0].dist_remaining_m is None

    def test_predictor_id(self) -> None:
        assert self.PREDICTOR.predictor_id == "gtfsrt"


# ---------------------------------------------------------------------------
# ConstantSpeedPredictor
# ---------------------------------------------------------------------------


class TestConstantSpeedPredictor:
    PREDICTOR = ConstantSpeedPredictor()

    def test_history_based_speed(self) -> None:
        """Speed from Δprogress_m/Δt is used; ETA, dist_remaining_m, stops_remaining set."""
        state1 = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0)
        state2 = VehicleState(epoch=1_000_010.0, lat=10.0, lon=10.0, progress_m=200.0)
        # speed = (200 - 100) / (1000010 - 1000000) = 10 m/s
        upcoming = [_stop("s1", 1, 500.0)]
        # dist = 500 - 200 = 300 m  →  ETA = 1000010 + 30 = 1000040
        preds = self.PREDICTOR.predict(
            TRIP, upcoming, state2, None, position_history=[state1, state2]
        )

        assert len(preds) == 1
        assert preds[0].stop_id == "s1"
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_040.0)
        assert preds[0].dist_remaining_m == pytest.approx(300.0)
        assert preds[0].stops_remaining == 1
        assert preds[0].made_at_epoch == pytest.approx(1_000_010.0)

    def test_reported_speed_fallback(self) -> None:
        """Falls back to vehicle_state.speed when history is None."""
        state = VehicleState(
            epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0, speed=5.0
        )
        upcoming = [_stop("s1", 1, 200.0)]
        # dist = 100 m at 5 m/s → 20 s → ETA = 1_000_020
        preds = self.PREDICTOR.predict(TRIP, upcoming, state, None, position_history=None)

        assert len(preds) == 1
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_020.0)
        assert preds[0].dist_remaining_m == pytest.approx(100.0)

    def test_single_history_entry_uses_reported_speed(self) -> None:
        """Only one history entry (insufficient for Δ) → falls back to speed."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0, speed=8.0)
        upcoming = [_stop("s1", 1, 180.0)]
        # dist = 80 m at 8 m/s → 10 s → ETA = 1_000_010
        preds = self.PREDICTOR.predict(
            TRIP, upcoming, state, None, position_history=[state]
        )

        assert len(preds) == 1
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_010.0)

    def test_zero_speed_emits_nothing(self) -> None:
        """speed = 0 (stationary vehicle) → no predictions, no crash."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0, speed=0.0)
        upcoming = [_stop("s1", 1, 200.0)]
        preds = self.PREDICTOR.predict(TRIP, upcoming, state, None)
        assert preds == []

    def test_no_speed_no_history_emits_nothing(self) -> None:
        """Neither reported speed nor sufficient history → empty list."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0, speed=None)
        upcoming = [_stop("s1", 1, 200.0)]
        preds = self.PREDICTOR.predict(TRIP, upcoming, state, None)
        assert preds == []

    def test_no_progress_on_vehicle_state_emits_nothing(self) -> None:
        """Vehicle has no progress_m → cannot compute dist_remaining → empty list."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=None, speed=10.0)
        upcoming = [_stop("s1", 1, 200.0)]
        preds = self.PREDICTOR.predict(TRIP, upcoming, state, None)
        assert preds == []

    def test_stop_behind_vehicle_skipped(self) -> None:
        """Stop with progress_m < vehicle progress_m has dist_remaining ≤ 0; skipped."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=300.0, speed=10.0)
        upcoming = [_stop("s1", 1, 200.0)]  # behind the vehicle
        preds = self.PREDICTOR.predict(TRIP, upcoming, state, None)
        assert preds == []

    def test_stop_without_progress_m_skipped(self) -> None:
        """Stop with progress_m=None is silently skipped."""
        state = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=100.0, speed=10.0)
        unprojected = Stop(stop_id="sx", stop_sequence=5, lat=10.0, lon=10.0, progress_m=None)
        preds = self.PREDICTOR.predict(TRIP, [unprojected], state, None)
        assert preds == []

    def test_multiple_stops_stops_remaining_and_etas(self) -> None:
        """Multiple upcoming stops are all predicted; stops_remaining decreases per stop."""
        s1 = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=0.0)
        s2 = VehicleState(epoch=1_000_010.0, lat=10.0, lon=10.0, progress_m=100.0)
        # speed = 10 m/s
        upcoming = [
            _stop("a", 1, 200.0),
            _stop("b", 2, 400.0),
            _stop("c", 3, 600.0),
        ]
        preds = self.PREDICTOR.predict(
            TRIP, upcoming, s2, None, position_history=[s1, s2]
        )

        assert len(preds) == 3
        assert preds[0].stops_remaining == 3
        assert preds[1].stops_remaining == 2
        assert preds[2].stops_remaining == 1
        # ETAs: now=1000010, speed=10 m/s, vehicle at 100 m
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_010.0 + 10.0)  # +100m
        assert preds[1].predicted_arrival_epoch == pytest.approx(1_000_010.0 + 30.0)  # +300m
        assert preds[2].predicted_arrival_epoch == pytest.approx(1_000_010.0 + 50.0)  # +500m

    def test_predictor_id(self) -> None:
        assert self.PREDICTOR.predictor_id == "baseline:constant_speed"

    def test_negative_dp_history_falls_back_to_reported_speed(self) -> None:
        """Δprogress_m < 0 (backward) in history → skip and use reported speed."""
        s1 = VehicleState(epoch=1_000_000.0, lat=10.0, lon=10.0, progress_m=200.0)
        s2 = VehicleState(epoch=1_000_010.0, lat=10.0, lon=10.0, progress_m=100.0, speed=5.0)
        # dp = -100 (backward) → skip; fallback to speed=5 m/s
        upcoming = [_stop("s1", 1, 200.0)]
        # dist = 200 - 100 = 100 m at 5 m/s → 20 s
        preds = self.PREDICTOR.predict(
            TRIP, upcoming, s2, None, position_history=[s1, s2]
        )
        assert len(preds) == 1
        assert preds[0].predicted_arrival_epoch == pytest.approx(1_000_010.0 + 20.0)


# ---------------------------------------------------------------------------
# StopPrediction.to_sample
# ---------------------------------------------------------------------------


class TestStopPredictionToSample:
    def test_to_sample_all_fields(self) -> None:
        """to_sample maps every StopPrediction field to the corresponding PredictionSample field."""
        pred = StopPrediction(
            stop_id="s1",
            stop_sequence=1,
            predicted_arrival_epoch=1_000_300.0,
            made_at_epoch=1_000_050.0,
            dist_remaining_m=250.0,
            stops_remaining=2,
        )
        sample = pred.to_sample(
            trip_key="test:t1:v1:20240101",
            predictor_id="gtfsrt",
            vehicle_progress_m_at_made=50.0,
        )

        assert sample.trip_key == "test:t1:v1:20240101"
        assert sample.predictor_id == "gtfsrt"
        assert sample.stop_id == "s1"
        assert sample.stop_sequence == 1
        assert sample.predicted_arrival_epoch == pytest.approx(1_000_300.0)
        assert sample.made_at_epoch == pytest.approx(1_000_050.0)
        assert sample.vehicle_progress_m_at_made == pytest.approx(50.0)
        assert sample.dist_remaining_m == pytest.approx(250.0)
        assert sample.stops_remaining == 2

    def test_to_sample_optional_fields_default_to_none(self) -> None:
        """to_sample with no optional args leaves nullable fields as None."""
        pred = StopPrediction(
            stop_id="s1",
            stop_sequence=1,
            predicted_arrival_epoch=1_000_300.0,
            made_at_epoch=1_000_050.0,
        )
        sample = pred.to_sample(
            trip_key="test:t1:v1:20240101",
            predictor_id="baseline:constant_speed",
        )

        assert sample.dist_remaining_m is None
        assert sample.stops_remaining is None
        assert sample.vehicle_progress_m_at_made is None

    def test_to_sample_returns_prediction_sample_type(self) -> None:
        """to_sample returns a PredictionSample (frozen Pydantic model)."""
        pred = StopPrediction(
            stop_id="s2",
            stop_sequence=2,
            predicted_arrival_epoch=1_000_400.0,
            made_at_epoch=1_000_100.0,
        )
        sample = pred.to_sample("k", "gtfsrt")
        assert isinstance(sample, PredictionSample)

    def test_to_sample_trip_key_matches(self) -> None:
        """to_sample uses the trip_key from TripInstance.trip_key correctly."""
        pred = StopPrediction(
            stop_id="s1",
            stop_sequence=1,
            predicted_arrival_epoch=1_000_300.0,
            made_at_epoch=1_000_050.0,
        )
        expected_key = TRIP.trip_key  # "test:t1:v1:20240101"
        sample = pred.to_sample(trip_key=expected_key, predictor_id="gtfsrt")
        assert sample.trip_key == expected_key


# ---------------------------------------------------------------------------
# R1 — PREDICTOR_REGISTRY contents (gtfs_eta:ewma retired upstream)
# ---------------------------------------------------------------------------


class TestPredictorRegistry:
    def test_registry_contains_expected_ids(self) -> None:
        """PREDICTOR_REGISTRY exposes the live gtfsrt/baseline/gtfs_eta predictors."""
        from stoptimes.val.engine.assembly import PREDICTOR_REGISTRY

        expected = {
            "gtfsrt",
            "baseline:constant_speed",
            "gtfs_eta:historical_mean",
            "gtfs_eta:polyreg_distance",
            "gtfs_eta:polyreg_time",
            "gtfs_eta:xgboost",
        }
        assert expected <= set(PREDICTOR_REGISTRY)

    def test_registry_does_not_contain_retired_ewma(self) -> None:
        """gtfs_eta:ewma was retired upstream and must not be selectable."""
        from stoptimes.val.engine.assembly import PREDICTOR_REGISTRY

        assert "gtfs_eta:ewma" not in PREDICTOR_REGISTRY


# ---------------------------------------------------------------------------
# M3 — declarative accepts_position_history opt-in
# ---------------------------------------------------------------------------


class TestAcceptsPositionHistory:
    def test_constant_speed_opts_in(self) -> None:
        assert ConstantSpeedPredictor.accepts_position_history is True
        assert ConstantSpeedPredictor().accepts_position_history is True

    def test_default_predictor_does_not_opt_in(self) -> None:
        """GtfsRtTripUpdatesPredictor does not declare it -> defaults False."""
        predictor = GtfsRtTripUpdatesPredictor()
        assert getattr(predictor, "accepts_position_history", False) is False

    @pytest.mark.asyncio
    async def test_engine_passes_history_only_when_opted_in(self) -> None:
        """run_engine dispatches position_history via getattr, not signature sniffing."""
        from pathlib import Path

        from stoptimes.val.engine.runtime import FeedSnapshot, ReplayFeedSource, run_engine
        from stoptimes.core.realtime.gtfsrt_json import parse_tu_feed, parse_vp_feed
        from stoptimes.core.gtfs.static import GtfsStaticLoader
        from stoptimes.val.groundtruth.shape_distance import ShapeDistanceDetector
        from stoptimes.val.storage.sqlite import SqliteStorage

        fixtures = Path(__file__).parent / "fixtures"
        static_loader = GtfsStaticLoader(fixtures / "mini_gtfs.zip")
        storage = SqliteStorage(":memory:")

        calls: list[bool] = []

        class RecordingPredictor:
            predictor_id = "recording"
            accepts_position_history = True

            def predict(self, trip, upcoming_stops, vehicle_state, trip_updates, **kwargs):
                calls.append("position_history" in kwargs)
                return []

        class NonOptedPredictor:
            predictor_id = "non_opted"

            def predict(self, trip, upcoming_stops, vehicle_state, trip_updates):
                calls.append(False)
                return []

        vp_dict = {
            "header": {"timestamp": 1000.0},
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
                        "position": {
                            "latitude": 42.3601,
                            "longitude": -71.0589,
                            "speed": 3.0,
                        },
                        "timestamp": 1000.0,
                        "current_status": "IN_TRANSIT_TO",
                        "current_stop_sequence": 1,
                    },
                }
            ],
        }
        tu_dict = {"header": {"timestamp": 1000.0}, "entity": []}
        vp_ts, vp_records = parse_vp_feed(vp_dict)
        tu_ts, tu_records = parse_tu_feed(tu_dict)
        snap = FeedSnapshot(
            header_timestamp=vp_ts,
            vp_records=vp_records,
            tu_header_timestamp=tu_ts,
            tu_records=tu_records,
        )

        await run_engine(
            feed_source=ReplayFeedSource([snap]),
            static_loader=static_loader,
            predictors=[RecordingPredictor(), NonOptedPredictor()],
            detectors=[ShapeDistanceDetector(shape_threshold_m=30.0)],
            storage=storage,
            source_name="test",
        )

        assert calls == [True, False]
