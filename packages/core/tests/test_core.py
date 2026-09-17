"""
Tests for core utilities: config, logging, validation, and exceptions.

Run with: pytest core/tests/test_core.py -v
"""

import os
import pytest
from datetime import datetime, timezone
from pathlib import Path


class TestConfig:
    """Tests for stoptimes.core.config module."""

    def test_get_config_returns_singleton(self):
        """Config should return the same instance."""
        from stoptimes.core.config import get_config, reset_config

        reset_config()  # Start fresh
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_registry_dir_uses_stoptimes_home(self):
        """STOPTIMES_HOME, when set, anchors the registry at $STOPTIMES_HOME/models."""
        from stoptimes.core.config import get_config, reset_config

        reset_config()
        os.environ.pop("MODEL_REGISTRY_DIR", None)
        os.environ["STOPTIMES_HOME"] = "/tmp/stoptimes-home-test"
        try:
            config = get_config()
            assert config.model_registry_dir == Path("/tmp/stoptimes-home-test").resolve() / "models"
        finally:
            os.environ.pop("STOPTIMES_HOME", None)
            reset_config()

    def test_config_model_registry_dir(self):
        """Model registry dir should be configurable via env var, and default
        to ./artifacts/models relative to the current working directory when
        neither MODEL_REGISTRY_DIR nor STOPTIMES_HOME is set."""
        from stoptimes.core.config import get_config, reset_config

        # Test default
        reset_config()
        os.environ.pop("MODEL_REGISTRY_DIR", None)
        os.environ.pop("STOPTIMES_HOME", None)
        config = get_config()
        assert config.model_registry_dir == Path.cwd() / "artifacts" / "models"

        # Explicit override wins
        reset_config()
        os.environ["MODEL_REGISTRY_DIR"] = "/tmp/stoptimes-registry-test"
        try:
            config = get_config()
            assert config.model_registry_dir == Path("/tmp/stoptimes-registry-test").resolve()
        finally:
            os.environ.pop("MODEL_REGISTRY_DIR", None)
            reset_config()

    def test_config_default_values(self):
        """Config should have sensible defaults.

        Default agency is mbta (the agency the estimator actually serves),
        so temporal defaults resolve to America/New_York / US_MA -- see
        AGENCY_TEMPORAL_DEFAULTS. Regression test for the 2026-08-14 finding
        that the estimator was silently defaulting to America/Costa_Rica / CR
        while serving MBTA predictions, a train/serve skew from models
        trained via dataset_builder's America/New_York / US_MA default.
        """
        from stoptimes.core.config import get_config, reset_config

        reset_config()
        for var in ("ETA_AGENCY", "ETA_TIMEZONE", "ETA_HOLIDAY_REGION"):
            os.environ.pop(var, None)
        config = get_config()
        assert config.agency == "mbta"
        assert config.default_timezone == "America/New_York"
        assert config.default_region == "US_MA"
        assert config.redis_default_port == 6379
        assert config.predictions_ttl_seconds == 300

    def test_config_agency_temporal_defaults_per_agency(self):
        """Each agency's timezone/region resolves from one shared table."""
        from stoptimes.core.config import AGENCY_TEMPORAL_DEFAULTS, get_config, reset_config

        reset_config()
        os.environ["ETA_AGENCY"] = "bucr"
        for var in ("ETA_TIMEZONE", "ETA_HOLIDAY_REGION"):
            os.environ.pop(var, None)
        try:
            config = get_config()
            assert config.default_timezone == AGENCY_TEMPORAL_DEFAULTS["bucr"]["timezone"]
            assert config.default_region == AGENCY_TEMPORAL_DEFAULTS["bucr"]["region"]
        finally:
            os.environ.pop("ETA_AGENCY", None)
            reset_config()

    def test_config_explicit_overrides_win_over_agency_default(self):
        """ETA_TIMEZONE/ETA_HOLIDAY_REGION still override the agency default."""
        from stoptimes.core.config import get_config, reset_config

        reset_config()
        os.environ["ETA_AGENCY"] = "mbta"
        os.environ["ETA_TIMEZONE"] = "America/Chicago"
        os.environ["ETA_HOLIDAY_REGION"] = "US"
        try:
            config = get_config()
            assert config.default_timezone == "America/Chicago"
            assert config.default_region == "US"
        finally:
            for var in ("ETA_AGENCY", "ETA_TIMEZONE", "ETA_HOLIDAY_REGION"):
                os.environ.pop(var, None)
            reset_config()

    def test_config_has_no_repo_root_discovery(self):
        """Config no longer walks the filesystem for a project root: there is
        no single meaningful "project root" in a uv workspace of independently
        installable packages. See docs/PLAN.md P0-A / docs/MIGRATION_MAP.md."""
        import stoptimes.core.config as config_module
        from stoptimes.core.config import get_config

        config = get_config()
        for stale_attr in (
            "project_root",
            "core_dir",
            "models_dir",
            "feature_engineering_dir",
            "eta_service_dir",
            "datasets_dir",
            "prefect_dir",
        ):
            assert not hasattr(config, stale_attr)
        assert not hasattr(config_module, "_find_project_root")


class TestLogging:
    """Tests for stoptimes.core.logging module."""

    def test_get_logger_returns_logger(self):
        """get_logger should return an ETALogger instance."""
        from stoptimes.core.logging import get_logger, ETALogger

        logger = get_logger("test")
        assert isinstance(logger, ETALogger)

    def test_logger_levels(self):
        """Logger should support all standard levels."""
        from stoptimes.core.logging import get_logger

        logger = get_logger("test_levels")
        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_with_fields(self):
        """Logger should accept structured fields."""
        from stoptimes.core.logging import get_logger

        logger = get_logger("test_fields")
        # This should not raise
        logger.info("Test message", vehicle_id="V123", route="171")

    def test_logger_caching(self):
        """Loggers should be cached by component name."""
        from stoptimes.core.logging import get_logger

        logger1 = get_logger("cached_test")
        logger2 = get_logger("cached_test")
        # Same component should return same logger (cached)
        assert logger1 is logger2


class TestExceptions:
    """Tests for stoptimes.core.exceptions module."""

    def test_base_exception(self):
        """ETAError should format messages correctly."""
        from stoptimes.core.exceptions import ETAError

        err = ETAError("Test error", code="TEST_CODE", key1="val1")
        assert "TEST_CODE" in str(err)
        assert "Test error" in str(err)
        assert "key1=val1" in str(err)

    def test_exception_to_dict(self):
        """Exceptions should convert to dict for JSON serialization."""
        from stoptimes.core.exceptions import ETAError

        err = ETAError("Test", foo="bar")
        d = err.to_dict()
        assert d["message"] == "Test"
        assert d["context"]["foo"] == "bar"

    def test_model_not_found_error(self):
        """ModelNotFoundError should include model details."""
        from stoptimes.core.exceptions import ModelNotFoundError

        err = ModelNotFoundError(model_key="xgb_global")
        assert "xgb_global" in str(err)
        assert "MODEL_NOT_FOUND" in str(err)

    def test_validation_error_hierarchy(self):
        """Validation errors should inherit from ETAError."""
        from stoptimes.core.exceptions import (
            ETAError,
            ValidationError,
            InvalidVehiclePositionError,
            InvalidStopError,
        )

        assert issubclass(ValidationError, ETAError)
        assert issubclass(InvalidVehiclePositionError, ValidationError)
        assert issubclass(InvalidStopError, ValidationError)


class TestValidation:
    """Tests for stoptimes.core.validation module."""

    def test_validate_vehicle_position_valid(self):
        """Valid vehicle position should pass validation."""
        from stoptimes.core.validation import validate_vehicle_position, VehiclePosition

        data = {
            "vehicle_id": "V123",
            "lat": 9.93,
            "lon": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "route": "171",
        }
        result = validate_vehicle_position(data)
        assert isinstance(result, VehiclePosition)
        assert result.vehicle_id == "V123"
        assert result.lat == 9.93
        assert result.lon == -84.08
        assert result.route_id == "171"

    def test_validate_vehicle_position_missing_id(self):
        """Missing vehicle_id should raise error."""
        from stoptimes.core.validation import validate_vehicle_position
        from stoptimes.core.exceptions import InvalidVehiclePositionError

        data = {
            "lat": 9.93,
            "lon": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "route": "171",
        }
        with pytest.raises(InvalidVehiclePositionError) as exc_info:
            validate_vehicle_position(data)
        assert "vehicle_id" in str(exc_info.value).lower()

    def test_validate_vehicle_position_invalid_lat(self):
        """Invalid latitude should raise error."""
        from stoptimes.core.validation import validate_vehicle_position
        from stoptimes.core.exceptions import InvalidVehiclePositionError

        data = {
            "vehicle_id": "V123",
            "lat": 91.0,  # Invalid: > 90
            "lon": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "route": "171",
        }
        with pytest.raises(InvalidVehiclePositionError) as exc_info:
            validate_vehicle_position(data)
        assert "latitude" in str(exc_info.value).lower() or "lat" in str(exc_info.value).lower()

    def test_validate_vehicle_position_alternate_field_names(self):
        """Should accept alternate field names."""
        from stoptimes.core.validation import validate_vehicle_position

        # Using alternate field names
        data = {
            "id": "V456",
            "latitude": 9.93,
            "longitude": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "routeId": "172",
        }
        result = validate_vehicle_position(data)
        assert result.vehicle_id == "V456"
        assert result.route_id == "172"

    def test_validate_stop_valid(self):
        """Valid stop should pass validation."""
        from stoptimes.core.validation import validate_stop, Stop

        data = {
            "stop_id": "S100",
            "lat": 9.94,
            "lon": -84.07,
            "stop_sequence": 5,
        }
        result = validate_stop(data)
        assert isinstance(result, Stop)
        assert result.stop_id == "S100"
        assert result.stop_sequence == 5

    def test_validate_stop_missing_coordinates(self):
        """Missing coordinates should raise error."""
        from stoptimes.core.validation import validate_stop
        from stoptimes.core.exceptions import InvalidStopError

        data = {
            "stop_id": "S100",
            "stop_sequence": 5,
        }
        with pytest.raises(InvalidStopError):
            validate_stop(data)

    def test_validate_stops_list(self):
        """Should validate a list of stops."""
        from stoptimes.core.validation import validate_stops_list

        data = [
            {"stop_id": "S1", "lat": 9.93, "lon": -84.08},
            {"stop_id": "S2", "lat": 9.94, "lon": -84.07},
        ]
        result = validate_stops_list(data)
        assert len(result) == 2
        assert result[0].stop_id == "S1"
        assert result[0].stop_sequence == 1  # Auto-assigned
        assert result[1].stop_sequence == 2

    def test_validate_stops_list_from_dict(self):
        """Should handle dict with 'stops' key."""
        from stoptimes.core.validation import validate_stops_list

        data = {
            "stops": [
                {"stop_id": "S1", "lat": 9.93, "lon": -84.08},
            ]
        }
        result = validate_stops_list(data)
        assert len(result) == 1

    def test_is_valid_vehicle_position_helper(self):
        """is_valid_vehicle_position should return bool."""
        from stoptimes.core.validation import is_valid_vehicle_position

        valid_data = {
            "vehicle_id": "V1",
            "lat": 9.93,
            "lon": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "route": "171",
        }
        invalid_data = {"vehicle_id": "V1"}

        assert is_valid_vehicle_position(valid_data) is True
        assert is_valid_vehicle_position(invalid_data) is False

    def test_to_dict_compatibility(self):
        """Validated objects should convert back to dicts."""
        from stoptimes.core.validation import validate_vehicle_position, validate_stop

        vp_data = {
            "vehicle_id": "V1",
            "lat": 9.93,
            "lon": -84.08,
            "timestamp": "2024-01-15T10:00:00Z",
            "route": "171",
        }
        vp = validate_vehicle_position(vp_data)
        vp_dict = vp.to_dict()
        assert vp_dict["vehicle_id"] == "V1"
        assert vp_dict["lat"] == 9.93

        stop_data = {"stop_id": "S1", "lat": 9.93, "lon": -84.08}
        stop = validate_stop(stop_data)
        stop_dict = stop.to_dict()
        assert stop_dict["stop_id"] == "S1"


class TestIntegration:
    """Integration tests for core modules working together."""

    def test_logging_with_validation_error(self):
        """Logging should work with validation errors."""
        from stoptimes.core.logging import get_logger
        from stoptimes.core.validation import validate_vehicle_position
        from stoptimes.core.exceptions import InvalidVehiclePositionError

        logger = get_logger("integration_test")

        try:
            validate_vehicle_position({})
        except InvalidVehiclePositionError as e:
            logger.error("Validation failed", error=str(e), code=e.code)
            # Should not raise

    def test_config_paths_importable(self):
        """Sibling stoptimes.core modules are importable via the normal
        package mechanism (uv workspace editable install) -- config no
        longer needs to (and doesn't) mutate sys.path for this to work."""
        from stoptimes.core.config import get_config

        get_config()

        from stoptimes.core.features.temporal import extract_temporal_features

        assert callable(extract_temporal_features)


class TestAgencyTemporalParity:
    """Roadmap Phase 1.1: builder (training) and estimator (serving) must
    resolve timezone/holiday-region identically for a given agency, both
    sourced from stoptimes.core.config.AGENCY_TEMPORAL_DEFAULTS -- not two independent
    hardcoded defaults that can drift apart. Regression coverage for the
    2026-08-14 finding: dataset_builder hardcoded America/New_York/US_MA
    while the deployed estimator defaulted to America/Costa_Rica/CR.
    """

    def _reset_env(self):
        for var in ("ETA_AGENCY", "ETA_TIMEZONE", "ETA_HOLIDAY_REGION"):
            os.environ.pop(var, None)

    def test_builder_default_matches_estimator_config_for_mbta(self):
        # dataset_builder.build_vp_training_dataset's `agency` parameter
        # defaults to stoptimes.core.config.DEFAULT_AGENCY itself (not a re-typed
        # string literal), so this constant *is* what a caller gets if they
        # don't pass `agency` -- import-checking it directly, rather than
        # importing the (Django + sch_pipeline-coupled) builder module,
        # keeps this test dependency-light while still exercising the real
        # shared default.
        from stoptimes.core.config import AGENCY_TEMPORAL_DEFAULTS, DEFAULT_AGENCY, get_config, reset_config

        reset_config()
        self._reset_env()
        try:
            builder_tz = AGENCY_TEMPORAL_DEFAULTS[DEFAULT_AGENCY]["timezone"]
            builder_region = AGENCY_TEMPORAL_DEFAULTS[DEFAULT_AGENCY]["region"]

            # The estimator's own resolution path (eta_service/estimator.py
            # calls extract_temporal_features(tz=_config.default_timezone,
            # region=_config.default_region)).
            serving_config = get_config()
            serving_tz = serving_config.default_timezone
            serving_region = serving_config.default_region

            assert builder_tz == serving_tz == "America/New_York"
            assert builder_region == serving_region == "US_MA"
        finally:
            self._reset_env()
            reset_config()

    def test_same_timestamp_produces_identical_features_builder_vs_serving(self):
        from datetime import datetime, timezone as dt_timezone
        from stoptimes.core.config import get_config, reset_config
        from stoptimes.core.features.temporal import extract_temporal_features

        reset_config()
        self._reset_env()
        try:
            serving_config = get_config()

            # A weekday morning-peak UTC timestamp and a US federal holiday,
            # so a tz/region mismatch would actually flip is_peak_hour /
            # is_holiday rather than passing by coincidence.
            timestamps = [
                datetime(2026, 3, 10, 12, 30, tzinfo=dt_timezone.utc),  # Tue 08:30 ET peak
                datetime(2026, 7, 4, 15, 0, tzinfo=dt_timezone.utc),    # US Independence Day
            ]
            for ts in timestamps:
                builder_feats = extract_temporal_features(
                    ts, tz="America/New_York", region="US_MA"
                )
                serving_feats = extract_temporal_features(
                    ts, tz=serving_config.default_timezone, region=serving_config.default_region
                )
                assert builder_feats == serving_feats
        finally:
            self._reset_env()
            reset_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
