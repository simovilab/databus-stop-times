"""Experiment builder logic: an editable draft, validation, and the
write-then-reload round trip -- everything the TUI needs that isn't a
widget. No Textual import here, so this module (and its tests) works
without the optional 'tui' extra installed.

docs/PLAN.md rule 2 ("UIs write config; the CLI runs config") means this
module's only side effect is writing an experiment.yaml through the real
`stoptimes.core.contracts.experiment` loader/dumper -- it never trains
anything and holds no state the YAML can't express (`ExperimentDraft` is
a 1:1 pre-image of `Experiment`, nothing more).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from stoptimes.core.contracts.experiment import (
    DEFAULT_BASELINES,
    DataSelection,
    Experiment,
    ModelSpec,
    OutputSpec,
    ParamChoices,
    ParamRange,
    ParamValue,
    SplitStrategy,
    dump_experiment,
    load_experiment,
)
from stoptimes.core.features.registry import FeatureNotFoundError, get as get_feature

SplitKind = Literal["time", "trip", "kfold"]


class ExperimentDraftError(ValueError):
    """Raised by `validate_draft`/`write_draft` with every problem found,
    not just the first -- so the TUI can show the whole list at once."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(problems) if problems else "invalid experiment draft")


@dataclass
class ExperimentDraft:
    """Everything the builder screens collect, shaped 1:1 with `Experiment`
    (see `to_experiment`/`from_experiment`). Plain data -- safe to construct,
    copy and compare in tests without touching Textual or the filesystem."""

    name: str = ""
    target: str = "arrival_delay_s"

    sources: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    routes: list[str] | None = None

    family: str = ""
    params: dict[str, ParamValue] = field(default_factory=dict)

    features: list[str] = field(default_factory=list)

    split_kind: SplitKind = "time"
    train_fraction: float | None = 0.8
    test_fraction: float | None = 0.2
    n_folds: int | None = None

    seed: int = 42

    registry_dir: str = "artifacts/models"
    tags: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=lambda: list(DEFAULT_BASELINES))

    def to_experiment(self) -> Experiment:
        """Build an `Experiment` from this draft. Raises `pydantic.ValidationError`
        for anything the contract itself rejects (e.g. no sources/datasets,
        a split missing its required fields) -- `validate_draft` turns that
        into readable problem strings."""
        return Experiment(
            name=self.name,
            data=DataSelection(
                sources=list(self.sources),
                datasets=list(self.datasets),
                start_date=self.start_date,
                end_date=self.end_date,
                routes=self.routes,
            ),
            target=self.target,
            model=ModelSpec(family=self.family, params=dict(self.params)),
            features=list(self.features),
            split=SplitStrategy(
                kind=self.split_kind,
                train_fraction=self.train_fraction,
                test_fraction=self.test_fraction,
                n_folds=self.n_folds,
            ),
            seed=self.seed,
            outputs=OutputSpec(registry_dir=self.registry_dir, tags=list(self.tags)),
            baselines=list(self.baselines),
        )

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> "ExperimentDraft":
        """The inverse of `to_experiment` -- used to open an existing
        experiment.yaml for editing."""
        return cls(
            name=experiment.name,
            target=experiment.target,
            sources=list(experiment.data.sources),
            datasets=list(experiment.data.datasets),
            start_date=experiment.data.start_date,
            end_date=experiment.data.end_date,
            routes=list(experiment.data.routes) if experiment.data.routes else None,
            family=experiment.model.family,
            params=dict(experiment.model.params),
            features=list(experiment.features),
            split_kind=experiment.split.kind,
            train_fraction=experiment.split.train_fraction,
            test_fraction=experiment.split.test_fraction,
            n_folds=experiment.split.n_folds,
            seed=experiment.seed,
            registry_dir=experiment.outputs.registry_dir,
            tags=list(experiment.outputs.tags),
            baselines=list(experiment.baselines),
        )


def validate_draft(draft: ExperimentDraft) -> list[str]:
    """Every problem with `draft`, human-readable, empty when it's ready to
    write. Checks feature names against the real feature registry first
    (a clearer message than the generic one `feature_registry.build` would
    raise later, at train time), then defers everything else -- data
    selection, split shape, model family presence -- to the `Experiment`
    contract itself, so this can never drift from what `train run` accepts."""
    problems: list[str] = []

    if not draft.name.strip():
        problems.append("name is required")
    if not draft.family.strip():
        problems.append("model family is required")
    if not draft.sources and not draft.datasets:
        problems.append("at least one data source or dataset is required")

    for feature_name in draft.features:
        try:
            get_feature(feature_name)
        except FeatureNotFoundError as exc:
            problems.append(str(exc))

    try:
        draft.to_experiment()
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(p) for p in error["loc"]) or "experiment"
            problems.append(f"{loc}: {error['msg']}")

    return problems


def write_draft(draft: ExperimentDraft, path: str | Path) -> Experiment:
    """Validate `draft`, write it to `path` as experiment.yaml, and confirm
    it round-trips losslessly through `load_experiment` (docs/PLAN.md U1
    requirement). Raises `ExperimentDraftError` instead of writing anything
    when validation fails."""
    problems = validate_draft(draft)
    if problems:
        raise ExperimentDraftError(problems)

    experiment = draft.to_experiment()
    target = dump_experiment(experiment, path)
    reloaded = load_experiment(target)
    if reloaded != experiment:
        raise ExperimentDraftError(
            [f"internal error: {target} did not round-trip through load_experiment unchanged"]
        )
    return experiment


def open_draft(path: str | Path) -> ExperimentDraft:
    """Load an existing experiment.yaml into an editable draft."""
    return ExperimentDraft.from_experiment(load_experiment(path))


def make_param_range(low: float, high: float, *, step: float | None = None, log: bool = False) -> ParamRange:
    """Convenience wrapper so the TUI (and tests) don't import
    `contracts.experiment.ParamRange` directly for the common case."""
    return ParamRange(low=low, high=high, step=step, log=log)


def make_param_choices(choices: list) -> ParamChoices:
    return ParamChoices(choices=list(choices))
