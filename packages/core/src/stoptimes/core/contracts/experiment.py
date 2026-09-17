"""Contract: experiment.yaml.

An `Experiment` is the one document that fully determines a training run:
what data to pull, what to predict, which model family with which
hyperparameters (fixed values now, ranges/choices later for tuning), which
named features to compute, how to split, the seed, and where outputs go.
The TUI (Phase 3 U1) only ever writes this file; the Python API and CLI only
ever read it — see docs/PLAN.md rule 2.

Existing starting point: `stoptimes.train.train_all_models` CLI args
(`--dataset`, `--by-route`, `--models {historical_mean,polyreg_distance,
polyreg_time,ewma,xgboost,all}`, `--no-save`). This contract generalizes that
one flat "which models" list into one `model` (a run trains one model spec;
sweeping many is a loop over many `Experiment`s, not a wider contract).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_BASELINES: tuple[str, ...] = ("schedule", "constant_speed")
"""Always included in every report (docs/PLAN.md rule 5)."""


class ParamRange(BaseModel):
    """A continuous hyperparameter search range, for later tuning."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["range"] = "range"
    low: float
    high: float
    step: float | None = None
    log: bool = False


class ParamChoices(BaseModel):
    """A discrete hyperparameter search set, for later tuning."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["choices"] = "choices"
    choices: list[Any] = Field(min_length=1)


ParamValue = float | int | str | bool | ParamRange | ParamChoices
"""A hyperparameter is a fixed scalar today, or a search space for a future tuner."""


class ModelSpec(BaseModel):
    """Which model family to train, and with what hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    family: str
    """Free-form family name (e.g. "xgboost", "historical_mean",
    "polyreg_distance", "polyreg_time"), not an enum — new families never
    require a contract change."""
    params: dict[str, ParamValue] = Field(default_factory=dict)


class DataSelection(BaseModel):
    """Which recorded data feeds this experiment's dataset."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    """Source ids (see contracts.source.Source.id) to pull from."""
    datasets: list[str] = Field(default_factory=list)
    """Named datasets in the local data catalog (T1), as an alternative or
    supplement to naming raw sources."""
    start_date: date | None = None
    end_date: date | None = None
    routes: list[str] | None = None
    """None = all routes."""

    @model_validator(mode="after")
    def _some_selection(self) -> "DataSelection":
        if not self.sources and not self.datasets:
            raise ValueError("data selection needs at least one of sources or datasets")
        return self


class SplitStrategy(BaseModel):
    """How the dataset is divided into train/test (/folds)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["time", "trip", "kfold"]
    train_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    test_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    n_folds: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> "SplitStrategy":
        if self.kind == "kfold":
            if self.n_folds is None:
                raise ValueError("split kind 'kfold' requires n_folds")
        else:
            if self.train_fraction is None or self.test_fraction is None:
                raise ValueError(f"split kind '{self.kind}' requires train_fraction and test_fraction")
            total = self.train_fraction + self.test_fraction
            if total > 1.0 + 1e-9:
                raise ValueError("train_fraction + test_fraction must not exceed 1.0")
        return self


class OutputSpec(BaseModel):
    """Where trained artifacts and their cards are written."""

    model_config = ConfigDict(extra="forbid")

    registry_dir: str = "artifacts/models"
    tags: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    """The full definition of one training run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    data: DataSelection
    target: str = "arrival_delay_s"
    model: ModelSpec
    features: list[str] = Field(default_factory=list)
    """Names resolved against the feature registry (core.features.registry, T3)."""
    split: SplitStrategy
    seed: int = 42
    outputs: OutputSpec = Field(default_factory=OutputSpec)
    baselines: list[str] = Field(default_factory=lambda: list(DEFAULT_BASELINES))
    """Baseline predictors always scored alongside `model` (docs/PLAN.md rule 5).
    Not emptyable by omission — pass `baselines: []` explicitly to opt out."""


def load_experiment(path: str | Path) -> Experiment:
    """Parse an experiment.yaml file into an `Experiment`."""
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    return Experiment.model_validate(data)


def dump_experiment(experiment: Experiment, path: str | Path) -> Path:
    """Write *experiment* to *path* as YAML, round-trippable with `load_experiment`."""
    target = Path(path)
    target.write_text(
        yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return target
