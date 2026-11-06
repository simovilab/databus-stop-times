"""Contract: model artifacts on disk.

Fixed by the orchestrator at the start of Phase 1 so C1 (contracts) and C3
(`stoptimes models migrate`) build on the same layout. C1 may add optional
fields; changing or removing a field needs an orchestrator-approved note.

Layout — one directory per artifact under a registry root::

    <registry>/
    └── <artifact_id>/
        ├── artifact.json   ArtifactMetadata (this module), always present
        ├── model.<ext>     the fitted model, ext from ModelFormat
        ├── card.md         human-readable model card (optional)
        └── <asset files>   sidecar assets listed in ArtifactMetadata.assets (optional)

`artifact.json` never needs the model's own library to be read, so listing,
cards and compatibility checks work in a serve-only install.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTIFACT_SCHEMA_VERSION = 2
"""2 adds optional ``assets`` / ``timezone`` / ``holiday_region``; version-1 files load unchanged."""
METADATA_FILENAME = "artifact.json"
CARD_FILENAME = "card.md"


class ModelFormat(str, Enum):
    """How `model.<ext>` is serialized."""

    PICKLE = "pickle"
    XGBOOST_JSON = "xgboost-json"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"

    @property
    def filename(self) -> str:
        return {
            ModelFormat.PICKLE: "model.pkl",
            ModelFormat.XGBOOST_JSON: "model.json",
            ModelFormat.ONNX: "model.onnx",
            ModelFormat.TORCHSCRIPT: "model.pt",
        }[self]


class ScopeLevel(str, Enum):
    """What a model is specialized to. Selection prefers the most specific match."""

    GLOBAL = "global"
    ROUTE = "route"
    DIRECTION = "direction"
    TRIP = "trip"


class ModelScope(BaseModel):
    """Which trips an artifact is allowed to serve.

    A registry normally holds one global model plus route- (or trip-) specific
    ones; `stoptimes.core.contracts.selection` ranks them with this.
    """

    model_config = ConfigDict(extra="forbid")

    level: ScopeLevel = ScopeLevel.GLOBAL
    source_id: str | None = None
    """Agency the model was trained for. None = source-agnostic (serves any)."""

    route_ids: list[str] = Field(default_factory=list)
    direction_id: int | None = None
    trip_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _keys_match_level(self) -> "ModelScope":
        if self.level in (ScopeLevel.ROUTE, ScopeLevel.DIRECTION) and not self.route_ids:
            raise ValueError(f"scope level {self.level.value} needs route_ids")
        if self.level is ScopeLevel.DIRECTION and self.direction_id is None:
            raise ValueError("scope level direction needs direction_id")
        if self.level is ScopeLevel.TRIP and not self.trip_ids:
            raise ValueError("scope level trip needs trip_ids")
        return self

    @property
    def specificity(self) -> int:
        return list(ScopeLevel).index(self.level)

    def matches(
        self,
        *,
        source_id: str | None = None,
        route_id: str | None = None,
        direction_id: int | None = None,
        trip_id: str | None = None,
    ) -> bool:
        """True if this artifact may serve the given trip."""
        if self.source_id is not None and source_id is not None and self.source_id != source_id:
            return False
        if self.level is ScopeLevel.GLOBAL:
            return True
        if self.level is ScopeLevel.TRIP:
            return trip_id in self.trip_ids
        if route_id not in self.route_ids:
            return False
        if self.level is ScopeLevel.DIRECTION:
            return direction_id == self.direction_id
        return True


class DataWindow(BaseModel):
    """What the model was trained on."""

    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    dataset_sha256: str | None = None
    sources: list[str] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    n_samples: int | None = None
    n_train: int | None = None
    n_test: int | None = None


class Provenance(BaseModel):
    """How and with what the model was built."""

    model_config = ConfigDict(extra="allow")

    created_at: datetime
    git_sha: str | None = None
    python_version: str | None = None
    lib_versions: dict[str, str | None] = Field(default_factory=dict)
    seed: int | None = None


class HorizonScore(BaseModel):
    """Error for predictions made `horizon_min_s`..`horizon_max_s` before the arrival."""

    model_config = ConfigDict(extra="forbid")

    horizon_min_s: int
    horizon_max_s: int | None = None
    n: int
    mae_s: float
    rmse_s: float | None = None
    median_ae_s: float | None = None
    p90_ae_s: float | None = None


_ASSET_FILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactAsset(BaseModel):
    """A sidecar file next to the model that its features need (e.g. segment statistics)."""

    model_config = ConfigDict(extra="forbid")

    file: str
    """File name inside the artifact directory (no path separators)."""

    sha256: str
    """Hex SHA-256 of the file's bytes; loaders refuse a file that does not match."""

    @model_validator(mode="after")
    def _safe(self) -> "ArtifactAsset":
        if not _ASSET_FILE_PATTERN.match(self.file) or self.file in (METADATA_FILENAME, CARD_FILENAME):
            raise ValueError(f"asset file {self.file!r} must be a plain file name inside the artifact directory")
        if not _SHA256_PATTERN.match(self.sha256):
            raise ValueError("asset sha256 must be 64 lowercase hex characters")
        return self


class ArtifactMetadata(BaseModel):
    """Contents of `artifact.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
    family: str
    target: str = "arrival_delay_s"
    format: ModelFormat = ModelFormat.PICKLE
    scope: ModelScope = Field(default_factory=ModelScope)
    features: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    data: DataWindow = Field(default_factory=DataWindow)
    provenance: Provenance
    metrics: dict[str, float | None] = Field(default_factory=dict)
    scores_by_horizon: list[HorizonScore] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    legacy: dict[str, Any] | None = None
    """Original pre-migration metadata, verbatim, for artifacts made by `models migrate`."""
    assets: dict[str, ArtifactAsset] = Field(default_factory=dict)
    """Sidecar assets by name (``segment_stats``, ``route_vocabulary``): file + sha256."""
    timezone: str | None = None
    """Agency timezone the live features were trained with (IANA name), if the model uses live features."""
    holiday_region: str | None = None
    """Holiday-calendar region the temporal features were trained with."""

    @property
    def model_filename(self) -> str:
        return self.format.filename


def artifact_dir(registry: Path, artifact_id: str) -> Path:
    return Path(registry) / artifact_id


def read_metadata(directory: Path) -> ArtifactMetadata:
    path = Path(directory) / METADATA_FILENAME
    return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def write_metadata(meta: ArtifactMetadata, directory: Path) -> Path:
    """Write `artifact.json` atomically (temp file + rename)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / METADATA_FILENAME
    payload = json.dumps(meta.model_dump(mode="json"), indent=2)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".artifact.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_asset(directory: Path, filename: str, data: bytes) -> ArtifactAsset:
    """Write a sidecar asset atomically into an artifact directory and describe it."""
    asset = ArtifactAsset(file=filename, sha256=sha256_hex(data))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".asset.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, directory / filename)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return asset


def read_asset_bytes(directory: Path, name: str, asset: ArtifactAsset) -> bytes:
    """The bytes of a recorded asset, verified against its sha256.

    Raises ``FileNotFoundError`` when the file is gone and ``ValueError`` on a hash mismatch.
    """
    path = Path(directory) / asset.file
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"asset {name!r}: file {asset.file!r} is missing from {directory}") from None
    actual = sha256_hex(data)
    if actual != asset.sha256:
        raise ValueError(
            f"asset {name!r} ({asset.file}) does not match its recorded sha256: "
            f"expected {asset.sha256[:12]}..., found {actual[:12]}..."
        )
    return data


def version_mismatches(meta: ArtifactMetadata, installed: dict[str, str | None]) -> list[str]:
    """Libraries whose major.minor differs between build time and now.

    Loaders must refuse (or require an explicit override) when this is non-empty
    for the library that deserializes the model — never unpickle silently across versions.
    """

    def major_minor(version: str | None) -> tuple[str, ...] | None:
        return tuple(version.split(".")[:2]) if version else None

    problems = []
    for lib, built in sorted(meta.provenance.lib_versions.items()):
        if built is None or lib not in installed:
            continue
        if major_minor(built) != major_minor(installed[lib]):
            problems.append(f"{lib}: built with {built}, installed {installed[lib]}")
    return problems
