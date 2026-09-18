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

Agency profiles
----------------
stoptimes ships no agency-specific code or data (see docs/PLAN.md §1 design
rule 4). Per-agency timezone, holiday-calendar region, and served-model pins
live in ``examples/agencies/<agency>.yaml`` (one file per agency, shared with
``stoptimes.val``'s source configs -- see ``sources_search_path`` below and
``examples/agencies/README.md``). Resolution order for each setting is:

1. An explicit env var (``STOPTIMES_TIMEZONE``, ``STOPTIMES_HOLIDAY_REGION``,
   ``STOPTIMES_MODEL_KEY``, ``STOPTIMES_MODEL_FALLBACK_KEY``).
2. The active agency's profile file: ``STOPTIMES_AGENCY_FILE`` (explicit
   path) or ``STOPTIMES_AGENCY``/``ETA_AGENCY`` (an id looked up on
   ``sources_search_path()``).
3. Generic defaults: timezone ``UTC``, no holiday region, no pinned model.

Old ``ETA_*`` env var names are still honored (with a ``DeprecationWarning``)
so existing SIMOVI deployments (``ETA_AGENCY=bucr`` on the databus producer)
keep working; the new name wins if both are set.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Deprecated legacy fallback (Phase 1 C2)
# ---------------------------------------------------------------------------
# SIMOVI's production databus/VPS deploys set ETA_AGENCY=bucr today with no
# examples/agencies/*.yaml checked out alongside them, and rely on the
# hardcoded bUCR served-model pin to keep working. This table exists ONLY to
# preserve that behavior when no agency profile file can be found on
# `sources_search_path()`; it is consulted nowhere else, always emits a
# DeprecationWarning when used, and is deleted in Phase 4 (see docs/PLAN.md
# §8 G1) once every deployment ships a profile file instead.
_LEGACY_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "mbta": {
        "timezone": "America/New_York",
        "holiday_region": "US_MA",
        "serving": {"primary_model": None, "fallback_model": None},
    },
    "bucr": {
        "timezone": "America/Costa_Rica",
        "holiday_region": "CR",
        "serving": {
            "primary_model": (
                "xgboost_bucr_segments_corpus_spatial-temporal_global_20260824_215905"
                "_handle_nan=drop_learning_rate=0.05_max_depth=5_n_estimators=200"
            ),
            "fallback_model": (
                "polyreg_distance_bucr_segments_corpus_distance_global_20260824_215904"
                "_degree=2_route_specific=no"
            ),
        },
    },
}

DEFAULT_AGENCY = "mbta"

# Generic defaults used when neither an env var nor a profile file resolves
# a setting -- see the module docstring's resolution order.
_GENERIC_TIMEZONE = "UTC"
_GENERIC_HOLIDAY_REGION: Optional[str] = None


def sources_search_path(explicit: "str | os.PathLike[str] | None" = None) -> list[Path]:
    """Ordered directories to search for agency profile / source-config YAML.

    1. `explicit` (e.g. a CLI ``--sources-dir`` flag), if given -- exclusive.
    2. `STOPTIMES_SOURCES`, `os.pathsep`-separated, in listed order.
    3. `./examples/agencies`, relative to the current working directory.

    Shared by ``stoptimes.val``'s source lookup (``feeds/source_config.py``)
    and this module's agency-profile lookup, so both resolve agency
    configuration from the same directories.
    """
    if explicit is not None:
        return [Path(explicit)]
    dirs: list[Path] = []
    env = os.environ.get("STOPTIMES_SOURCES")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    dirs.append(Path.cwd() / "examples" / "agencies")
    return dirs


def _read_yaml_profile(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _find_profile_file(agency: str) -> Optional[Path]:
    """First `<agency>.yaml` found on `sources_search_path()`, if any."""
    for directory in sources_search_path():
        candidate = Path(directory) / f"{agency}.yaml"
        if candidate.is_file():
            return candidate
    return None


def _profile_for_named_agency(agency: str) -> Optional[dict[str, Any]]:
    """Resolve `agency`'s profile by name: a file on the search path, else
    (with a DeprecationWarning) the legacy builtin, else None."""
    found = _find_profile_file(agency)
    if found is not None:
        return _read_yaml_profile(found)
    if agency in _LEGACY_BUILTIN_PROFILES:
        warnings.warn(
            f"No agency profile file found for {agency!r} on the sources "
            "search path; falling back to deprecated built-in defaults. "
            "This fallback is removed in Phase 4 (docs/PLAN.md §8 G1) -- "
            f"ship examples/agencies/{agency}.yaml (or point STOPTIMES_SOURCES "
            "at a directory that has one) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _LEGACY_BUILTIN_PROFILES[agency]
    return None


def _known_agency_ids() -> list[str]:
    ids = set(_LEGACY_BUILTIN_PROFILES)
    for directory in sources_search_path():
        directory = Path(directory)
        if directory.is_dir():
            ids.update(p.stem for p in directory.glob("*.yaml"))
    return sorted(ids)


class _AgencyTemporalDefaultsTable(Mapping):
    """Back-compat view over agency profile files: ``AGENCY_TEMPORAL_DEFAULTS[agency]``
    still returns ``{"timezone": ..., "region": ...}`` for any agency with a
    resolvable profile (a file on ``sources_search_path()``, or -- deprecated,
    with a warning -- ``_LEGACY_BUILTIN_PROFILES``).

    Kept for ``stoptimes.core.segments.resolve_agency_temporal`` and
    ``stoptimes.train.datasets.dataset_builder`` (and their tests), which
    index this table directly by agency id rather than going through
    ``ProjectConfig``.
    """

    def __getitem__(self, agency: str) -> dict[str, Optional[str]]:
        profile = _profile_for_named_agency(agency)
        if profile is None:
            raise KeyError(agency)
        return {
            "timezone": profile.get("timezone") or _GENERIC_TIMEZONE,
            "region": profile.get("holiday_region", _GENERIC_HOLIDAY_REGION),
        }

    def __iter__(self):
        return iter(_known_agency_ids())

    def __len__(self) -> int:
        return len(_known_agency_ids())

    def __contains__(self, agency: object) -> bool:
        if not isinstance(agency, str):
            return False
        return _profile_for_named_agency(agency) is not None


class _AgencyModelDefaultsTable(Mapping):
    """Back-compat view over agency profile files for served-model pins --
    ``AGENCY_MODEL_DEFAULTS[agency]`` returns ``{"primary": ..., "fallback": ...}``.
    See ``_AgencyTemporalDefaultsTable`` for the same pattern applied to
    timezone/holiday-region.
    """

    def __getitem__(self, agency: str) -> dict[str, Optional[str]]:
        profile = _profile_for_named_agency(agency)
        if profile is None:
            raise KeyError(agency)
        serving = profile.get("serving") or {}
        return {
            "primary": serving.get("primary_model"),
            "fallback": serving.get("fallback_model"),
        }

    def __iter__(self):
        return iter(_known_agency_ids())

    def __len__(self) -> int:
        return len(_known_agency_ids())

    def __contains__(self, agency: object) -> bool:
        if not isinstance(agency, str):
            return False
        return _profile_for_named_agency(agency) is not None


# Back-compat module-level names -- see the classes above. Existing call
# sites (`stoptimes.core.segments`, `stoptimes.train.datasets.dataset_builder`)
# and tests index these directly by agency id; nothing new should import
# them -- prefer `ProjectConfig.default_timezone` / `.default_region` /
# `.served_model_key` / `.served_model_fallback_key`, which additionally
# apply the env-var overrides.
AGENCY_TEMPORAL_DEFAULTS: Mapping[str, dict[str, Optional[str]]] = _AgencyTemporalDefaultsTable()
AGENCY_MODEL_DEFAULTS: Mapping[str, dict[str, Optional[str]]] = _AgencyModelDefaultsTable()


def _read_env(new: str, old: str) -> Optional[str]:
    """Read `new`, falling back to the deprecated `old` name with a warning.
    The new name wins when both are set."""
    new_val = os.environ.get(new)
    if new_val is not None:
        return new_val
    old_val = os.environ.get(old)
    if old_val is not None:
        warnings.warn(
            f"{old} is deprecated; use {new} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return old_val
    return None


@dataclass
class ProjectConfig:
    """
    Centralized stoptimes configuration.

    Usage:
        from stoptimes.core.config import get_config

        config = get_config()
        print(config.model_registry_dir)
    """

    # Set only by `set_model_registry_dir`; an explicit call beats the
    # environment. Everything else is read from the environment on access
    # (see below) rather than frozen when this object is first created --
    # `get_config()` is a process-global singleton, and a config that
    # snapshots os.environ makes any later change (a test's monkeypatch, a
    # CLI setting STOPTIMES_HOME before doing its work) silently ineffective.
    _model_registry_explicit: Optional[str] = field(default=None, repr=False)

    @property
    def _model_registry_override(self) -> Optional[str]:
        return self._model_registry_explicit or _read_env(
            "STOPTIMES_MODEL_REGISTRY_DIR", "MODEL_REGISTRY_DIR"
        )

    @property
    def _stoptimes_home_override(self) -> Optional[str]:
        return os.environ.get("STOPTIMES_HOME")

    @property
    def _agency_override(self) -> Optional[str]:
        return _read_env("STOPTIMES_AGENCY", "ETA_AGENCY")

    @property
    def _timezone_override(self) -> Optional[str]:
        return _read_env("STOPTIMES_TIMEZONE", "ETA_TIMEZONE")

    @property
    def _region_override(self) -> Optional[str]:
        return _read_env("STOPTIMES_HOLIDAY_REGION", "ETA_HOLIDAY_REGION")

    @property
    def _served_model_key_override(self) -> Optional[str]:
        return _read_env("STOPTIMES_MODEL_KEY", "ETA_MODEL_KEY")

    @property
    def _served_model_fallback_override(self) -> Optional[str]:
        return _read_env("STOPTIMES_MODEL_FALLBACK_KEY", "ETA_MODEL_FALLBACK_KEY")

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

        1. `STOPTIMES_MODEL_REGISTRY_DIR` env var (or `set_model_registry_dir`)
           -- an explicit, single-purpose override. The deprecated
           `MODEL_REGISTRY_DIR` name still works too (see `_read_env`).
        2. `$STOPTIMES_HOME/models`, if `STOPTIMES_HOME` is set.
        3. `./artifacts/models`, relative to the current working directory.
        """
        if self._model_registry_override:
            return Path(self._model_registry_override).expanduser().resolve()
        if self.stoptimes_home is not None:
            return self.stoptimes_home / "models"
        return Path.cwd() / "artifacts" / "models"

    # =========================================================================
    # Agency profile resolution
    # =========================================================================

    @property
    def agency(self) -> str:
        """Which agency this deployment serves."""
        return self._agency_override or DEFAULT_AGENCY

    @property
    def _profile(self) -> dict[str, Any]:
        """Resolved profile for the active agency: `STOPTIMES_AGENCY_FILE`
        (explicit path) if set, else a `<agency>.yaml` found on
        `sources_search_path()`, else (deprecated) the legacy builtin, else
        `{}` (generic defaults apply)."""
        explicit_file = os.environ.get("STOPTIMES_AGENCY_FILE")
        if explicit_file:
            path = Path(explicit_file).expanduser()
            if path.is_file():
                return _read_yaml_profile(path)
            return {}
        profile = _profile_for_named_agency(self.agency)
        return profile if profile is not None else {}

    @property
    def default_timezone(self) -> str:
        """Default timezone for temporal features, per `agency`'s profile."""
        return self._timezone_override or self._profile.get("timezone") or _GENERIC_TIMEZONE

    @property
    def default_region(self) -> Optional[str]:
        """Default holiday-calendar region for temporal features, per
        `agency`'s profile (`None` if the agency has none configured)."""
        return self._region_override or self._profile.get("holiday_region") or _GENERIC_HOLIDAY_REGION

    @property
    def served_model_key(self) -> Optional[str]:
        """Registry key the databus path should serve by default.

        `STOPTIMES_MODEL_KEY` (or deprecated `ETA_MODEL_KEY`) overrides this
        per-deployment; otherwise it's the `agency`'s profile's
        `serving.primary_model` (`None` for agencies with no pin, e.g. mbta,
        which keeps the pre-existing get_best_model ranking behavior).
        """
        serving = self._profile.get("serving") or {}
        return self._served_model_key_override or serving.get("primary_model")

    @property
    def served_model_fallback_key(self) -> Optional[str]:
        """Registry key to try if `served_model_key` isn't in the registry."""
        serving = self._profile.get("serving") or {}
        return self._served_model_fallback_override or serving.get("fallback_model")

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

        Also updates both the preferred and deprecated environment variables
        for child processes / direct `os.environ` readers elsewhere in the
        codebase (e.g. `stoptimes.models.common.registry`).
        """
        resolved = str(Path(path).expanduser().resolve())
        self._model_registry_explicit = resolved
        os.environ["STOPTIMES_MODEL_REGISTRY_DIR"] = resolved
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
