"""
Centralized configuration for stoptimes: model registry location, agency
defaults, and other cross-cutting settings.

There is no single "project root" concept here: stoptimes ships as a uv
workspace of independently installable packages (`packages/*`), so a
package can be imported from inside this repo, from an editable install, or
as a regular dependency of an unrelated project with no repo checkout at
all. Anything that needs a location (the model registry) is resolved from
`STOPTIMES_HOME` / an explicit override / the current working directory --
never by walking up from this file looking for a `pyproject.toml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Per-agency timezone + holiday-calendar region. Single source of truth for
# temporal feature extraction -- the dataset builder (training) and the
# estimator (serving) must both resolve through this table so a given
# timestamp produces identical hour/is_weekend/is_holiday/is_peak_hour
# features whether it's being trained on or predicted against.
AGENCY_TEMPORAL_DEFAULTS: dict[str, dict[str, str]] = {
    "mbta": {"timezone": "America/New_York", "region": "US_MA"},
    "bucr": {"timezone": "America/Costa_Rica", "region": "CR"},
}
DEFAULT_AGENCY = "mbta"

# Per-agency served-model pin: the registry key the databus path (via
# stoptimes.serve.estimator.estimate_stop_times) uses when the caller doesn't
# pass an explicit model_key, plus a fallback key tried if the primary is
# missing from the registry. This is what keeps registry.get_best_model()'s
# metric-only ranking (which has no dataset/agency awareness and would
# happily rank an MBTA model above a BUCR one) from ever being consulted for
# an agency that has a pinned model -- see estimate_stop_times's smart model
# selection block. Update these when a newer/better BUCR model is trained
# and promoted, or override per-deployment with ETA_MODEL_KEY /
# ETA_MODEL_FALLBACK_KEY.
AGENCY_MODEL_DEFAULTS: dict[str, dict[str, Optional[str]]] = {
    "bucr": {
        "primary": (
            "xgboost_bucr_segments_corpus_spatial-temporal_global_20260824_215905"
            "_handle_nan=drop_learning_rate=0.05_max_depth=5_n_estimators=200"
        ),
        "fallback": (
            "polyreg_distance_bucr_segments_corpus_distance_global_20260824_215904"
            "_degree=2_route_specific=no"
        ),
    },
    "mbta": {"primary": None, "fallback": None},
}


@dataclass
class ProjectConfig:
    """
    Centralized stoptimes configuration.

    Usage:
        from stoptimes.core.config import get_config

        config = get_config()
        print(config.model_registry_dir)
    """

    # Environment overrides (populated from env vars)
    _model_registry_override: Optional[str] = field(default=None, repr=False)
    _stoptimes_home_override: Optional[str] = field(default=None, repr=False)
    _agency_override: Optional[str] = field(default=None, repr=False)
    _timezone_override: Optional[str] = field(default=None, repr=False)
    _region_override: Optional[str] = field(default=None, repr=False)
    _served_model_key_override: Optional[str] = field(default=None, repr=False)
    _served_model_fallback_override: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Load environment variable overrides."""
        self._model_registry_override = os.environ.get("MODEL_REGISTRY_DIR")
        self._stoptimes_home_override = os.environ.get("STOPTIMES_HOME")
        self._agency_override = os.environ.get("ETA_AGENCY")
        self._timezone_override = os.environ.get("ETA_TIMEZONE")
        self._region_override = os.environ.get("ETA_HOLIDAY_REGION")
        self._served_model_key_override = os.environ.get("ETA_MODEL_KEY")
        self._served_model_fallback_override = os.environ.get("ETA_MODEL_FALLBACK_KEY")

    # =========================================================================
    # Directory Paths
    # =========================================================================

    @property
    def stoptimes_home(self) -> Optional[Path]:
        """`STOPTIMES_HOME`, if set: a directory stoptimes can keep its own
        state under (today: the model registry, at `$STOPTIMES_HOME/models`)."""
        if self._stoptimes_home_override:
            return Path(self._stoptimes_home_override).expanduser().resolve()
        return None

    @property
    def model_registry_dir(self) -> Path:
        """
        Path to trained model storage. Resolution order:

        1. `MODEL_REGISTRY_DIR` env var (or `set_model_registry_dir`) --
           an explicit, single-purpose override.
        2. `$STOPTIMES_HOME/models`, if `STOPTIMES_HOME` is set.
        3. `./artifacts/models`, relative to the current working directory.
        """
        if self._model_registry_override:
            return Path(self._model_registry_override).expanduser().resolve()
        if self.stoptimes_home is not None:
            return self.stoptimes_home / "models"
        return Path.cwd() / "artifacts" / "models"

    # =========================================================================
    # Default Values (eliminates magic numbers)
    # =========================================================================

    @property
    def agency(self) -> str:
        """Which agency this deployment serves. See AGENCY_TEMPORAL_DEFAULTS."""
        return self._agency_override or DEFAULT_AGENCY

    @property
    def default_timezone(self) -> str:
        """Default timezone for temporal features, per `agency`."""
        return self._timezone_override or AGENCY_TEMPORAL_DEFAULTS[self.agency]["timezone"]

    @property
    def default_region(self) -> str:
        """Default holiday-calendar region for temporal features, per `agency`."""
        return self._region_override or AGENCY_TEMPORAL_DEFAULTS[self.agency]["region"]

    @property
    def served_model_key(self) -> Optional[str]:
        """Registry key the databus path should serve by default.

        ETA_MODEL_KEY overrides this per-deployment; otherwise it's the
        `agency`'s entry in AGENCY_MODEL_DEFAULTS (None for agencies with no
        pin, e.g. mbta, which keeps the pre-existing get_best_model ranking
        behavior).
        """
        return self._served_model_key_override or AGENCY_MODEL_DEFAULTS.get(self.agency, {}).get("primary")

    @property
    def served_model_fallback_key(self) -> Optional[str]:
        """Registry key to try if `served_model_key` isn't in the registry."""
        return self._served_model_fallback_override or AGENCY_MODEL_DEFAULTS.get(self.agency, {}).get("fallback")

    # Redis defaults
    redis_default_host: str = "localhost"
    redis_default_port: int = 6379
    redis_default_db: int = 0

    # Cache settings
    predictions_ttl_seconds: int = 300
    route_stops_cache_size: int = 100
    shape_cache_size: int = 100

    # Distance thresholds (meters)
    min_distance_threshold_m: float = 10.0

    # Model defaults
    default_model_type: str = "xgboost"
    default_metric: str = "test_mae_seconds"

    # Weather defaults (placeholder values)
    default_temperature_c: float = 25.0
    default_precipitation_mm: float = 0.0
    default_wind_speed_kmh: Optional[float] = None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def ensure_dirs_exist(self) -> None:
        """Create essential directories if they don't exist."""
        self.model_registry_dir.mkdir(parents=True, exist_ok=True)

    def set_model_registry_dir(self, path: str) -> None:
        """
        Override the model registry directory at runtime.

        Also updates the environment variable for child processes.
        """
        resolved = str(Path(path).expanduser().resolve())
        self._model_registry_override = resolved
        os.environ["MODEL_REGISTRY_DIR"] = resolved

    def get_model_path(self, model_key: str) -> Path:
        """Get the full path for a model artifact."""
        return self.model_registry_dir / f"{model_key}.pkl"

    def get_model_metadata_path(self, model_key: str) -> Path:
        """Get the full path for a model's metadata file."""
        return self.model_registry_dir / f"{model_key}_meta.json"


# Global config instance (lazy initialization)
_config: Optional[ProjectConfig] = None


def get_config() -> ProjectConfig:
    """
    Get or create the global configuration instance.

    Usage:
        from stoptimes.core.config import get_config

        config = get_config()
        print(f"Model registry: {config.model_registry_dir}")
    """
    global _config
    if _config is None:
        _config = ProjectConfig()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
