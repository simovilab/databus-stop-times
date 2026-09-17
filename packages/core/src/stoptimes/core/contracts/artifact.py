"""Contract: model artifacts on disk.

Fixed by the orchestrator at the start of Phase 1 so C1 (contracts) and C3
(`stoptimes models migrate`) build on the same layout. C1 may add optional
fields; changing or removing a field needs an orchestrator-approved note.

Layout — one directory per artifact under a registry root::

    <registry>/
    └── <artifact_id>/
        ├── artifact.json   ArtifactMetadata (this module), always present
        ├── model.<ext>     the fitted model, ext from ModelFormat
        └── card.md         human-readable model card (optional)

`artifact.json` never needs the model's own library to be read, so listing,
cards and compatibility checks work in a serve-only install.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_SCHEMA_VERSION = 1
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


class ArtifactMetadata(BaseModel):
    """Contents of `artifact.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
    family: str
    target: str = "arrival_delay_s"
    format: ModelFormat = ModelFormat.PICKLE
    features: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    data: DataWindow = Field(default_factory=DataWindow)
    provenance: Provenance
    metrics: dict[str, float | None] = Field(default_factory=dict)
    scores_by_horizon: list[HorizonScore] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    legacy: dict[str, Any] | None = None
    """Original pre-migration metadata, verbatim, for artifacts made by `models migrate`."""

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
