"""Thin accessors the experiment builder uses to populate its choices from
real sources -- never a hardcoded list of names alone (docs/PLAN.md §7, U1).

- **Features**: `stoptimes.core.features.registry` is already a real,
  documented registry (name, description, tags). `list_features()` here is
  a plain re-export so the TUI has one import surface.
- **Targets**: no target registry exists anywhere in the codebase yet --
  `contracts.experiment.Experiment.target` is a free `str` with a single
  default (`"arrival_delay_s"`), and `train.experiment._resolve_target_column`
  recognizes exactly one legacy alias (`"time_to_arrival_seconds"`, what
  pre-Phase-2 datasets call the same column). `list_targets()` offers both,
  described, and the builder still accepts free text for anything else --
  filed as a gap in docs/notes/p3-u1.md.
- **Model families**: `stoptimes.models.common.contract_adapter.get_model_class`
  is the only real registry, but it's private (`_FAMILY_FACTORIES` /
  `_BASELINE_FACTORIES`) and exposes no descriptions or hyperparameter
  metadata. `list_model_families()` reads that same private mapping (so the
  set of valid family names always matches what `get_model_class` actually
  accepts -- it is never redeclared here) and pairs it with per-family
  human descriptions plus hyperparameters introspected live from each
  family's real `__init__`/`train_lstm` signature via `inspect`, not
  hand-copied defaults. Only the descriptive text is authored in this file.
  Filed as a gap in docs/notes/p3-u1.md: `stoptimes-models` should grow a
  public `list_families()` with the same shape, and this module should
  become a thin re-export of it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from stoptimes.core.features.registry import Feature
from stoptimes.core.features.registry import list_features as _list_core_features


@dataclass(frozen=True)
class TargetInfo:
    name: str
    description: str


_KNOWN_TARGETS: tuple[TargetInfo, ...] = (
    TargetInfo(
        name="arrival_delay_s",
        description=(
            "Seconds between actual and scheduled arrival at the target stop "
            "(the Experiment.target default)."
        ),
    ),
    TargetInfo(
        name="time_to_arrival_seconds",
        description=(
            "Legacy target column name carried by pre-Phase-2 datasets (ETADataset). "
            "train.experiment resolves it automatically when 'arrival_delay_s' isn't "
            "present in the dataset."
        ),
    ),
)


def list_targets() -> list[TargetInfo]:
    """Known prediction targets. Not exhaustive -- the builder also accepts a
    free-text target, since `Experiment.target` is validated as a plain `str`."""
    return list(_KNOWN_TARGETS)


def list_features(tag: str | None = None) -> list[Feature]:
    """Re-export of `core.features.registry.list_features` for one TUI import surface."""
    return _list_core_features(tag=tag)


@dataclass(frozen=True)
class ParamInfo:
    """One hyperparameter of a model family, introspected from its real
    `__init__` (or `train_lstm`, for `lstm`) signature."""

    name: str
    default: Any
    type_hint: str
    description: str
    tunable: bool
    """Whether this param is numeric and so eligible for a `ParamRange` /
    `ParamChoices` search space in the TUI (docs/PLAN.md: "ranges where a
    family supports tuning")."""


@dataclass(frozen=True)
class ModelFamilyInfo:
    name: str
    description: str
    params: list[ParamInfo] = field(default_factory=list)
    is_baseline: bool = False


# Per-parameter descriptions shared across families -- authored copy, not
# sourced from a registry (none of the family classes carry per-arg
# docstrings a machine could reliably parse). Falls back to a generic
# sentence for anything not listed here, so a new family/param never
# crashes the TUI, only shows a thinner description.
_PARAM_DESCRIPTIONS: dict[str, str] = {
    "group_by": "Columns to group observations by when computing the estimate.",
    "handle_nan": "How to handle rows with missing inputs: 'drop', 'impute', or 'error'.",
    "alpha": "Regularization strength (higher = smoother fit, higher bias).",
    "degree": "Polynomial degree of the distance-to-time curve (2 or 3 typical).",
    "poly_degree": "Polynomial degree of the distance-to-time curve (2 or 3 typical).",
    "route_specific": "Whether to fit a separate model per route instead of one global model.",
    "include_temporal": "Whether to include time-of-day/day-of-week features.",
    "include_spatial": "Whether to include spatial/kinematic/status features.",
    "min_observations": "Minimum prior observations required before trusting the rolling estimate.",
    "max_depth": "Maximum depth of each decision tree (higher = more capacity, more overfit risk).",
    "n_estimators": "Number of boosting rounds / trees.",
    "learning_rate": "Shrinkage applied to each boosting step (lower = slower, often better generalization).",
    "subsample": "Fraction of rows sampled per boosting round.",
    "colsample_bytree": "Fraction of columns sampled per tree.",
    "min_child_weight": "Minimum sum of instance weight needed in a leaf (higher = more conservative splits).",
    "reg_alpha": "L1 regularization on leaf weights.",
    "reg_lambda": "L2 regularization on leaf weights.",
    "gamma": "Minimum loss reduction required to make a further split.",
    "early_stopping_rounds": "Stop boosting if the validation metric hasn't improved in this many rounds.",
    "random_state": "Random seed for the model library's own internal randomness.",
    "hidden_size": "Width of the LSTM/GRU hidden state.",
    "num_layers": "Number of stacked recurrent layers.",
    "dropout": "Dropout probability applied between recurrent layers.",
    "rnn_type": "Recurrent cell type: 'lstm' or 'gru'.",
    "batch_size": "Number of sequences per training batch.",
    "max_epochs": "Maximum number of training epochs.",
    "patience": "Epochs of no validation-metric improvement before early stopping.",
    "weight_decay": "L2 weight regularization applied by the optimizer.",
    "grad_clip": "Maximum gradient norm before clipping, for training stability.",
    "loss": "Training loss function (e.g. 'huber', 'mse').",
    "target_transform": "Transform applied to the target before training ('none' or a named transform).",
    "service_offset_h": "Hours after midnight a transit service day is considered to start (purges same-trip leakage across the split).",
    "seed": "Random seed for the split/initialization performed by this family.",
    "val_fraction": "Fraction of the training window carved off as validation for early stopping.",
}

_GENERIC_PARAM_DESCRIPTION = "Hyperparameter of this model family's implementation."

_TUNABLE_ANNOTATIONS = (int, float)


def _param_description(name: str) -> str:
    return _PARAM_DESCRIPTIONS.get(name, _GENERIC_PARAM_DESCRIPTION)


def _type_hint_str(annotation: Any, default: Any) -> str:
    if annotation is not inspect.Parameter.empty:
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation)
    if default is not inspect.Parameter.empty and default is not None:
        return type(default).__name__
    return "str"


def _is_tunable(annotation: Any, default: Any) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, bool):
        return False
    if isinstance(annotation, type) and issubclass(annotation, _TUNABLE_ANNOTATIONS):
        return True
    if annotation is inspect.Parameter.empty and isinstance(default, _TUNABLE_ANNOTATIONS) and not isinstance(
        default, bool
    ):
        return True
    return False


def _params_from_signature(func: Any, *, only: frozenset[str] | None = None) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return params
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if only is not None and name not in only:
            continue
        default = None if param.default is inspect.Parameter.empty else param.default
        params.append(
            ParamInfo(
                name=name,
                default=default,
                type_hint=_type_hint_str(param.annotation, param.default),
                description=_param_description(name),
                tunable=_is_tunable(param.annotation, param.default),
            )
        )
    return params


@dataclass(frozen=True)
class _FamilySource:
    module: str
    cls_name: str
    description: str


_FAMILY_SOURCES: dict[str, _FamilySource] = {
    "historical_mean": _FamilySource(
        "stoptimes.models.historical_mean.train",
        "HistoricalMeanModel",
        "Predicts arrival delay as the historical mean of similar past observations "
        "(grouped by route, stop, hour, ...). Simple, fast, and a strong baseline when "
        "there's enough history per group -- weak where a group has few or no prior rows.",
    ),
    "ewma": _FamilySource(
        "stoptimes.models.ewma.train",
        "EWMAModel",
        "Exponentially-weighted moving average of recent observations per group -- "
        "like historical_mean but weights recent traversals more, so it adapts faster "
        "to changing conditions (e.g. ongoing congestion) at the cost of more noise.",
    ),
    "polyreg_distance": _FamilySource(
        "stoptimes.models.polyreg_distance.train",
        "PolyRegDistanceModel",
        "Ridge-regularized polynomial regression of arrival time against distance to "
        "the target stop only. Cheap and interpretable; ignores time-of-day and traffic.",
    ),
    "polyreg_time": _FamilySource(
        "stoptimes.models.polyreg_time.train",
        "PolyRegTimeModel",
        "Ridge-regularized polynomial regression of arrival time against distance plus "
        "(optionally) temporal and spatial features -- a richer version of polyreg_distance.",
    ),
    "xgb": _FamilySource(
        "stoptimes.models.xgb.train",
        "XGBTimeModel",
        "Gradient-boosted trees (XGBoost) over whatever features the experiment selects. "
        "Usually the strongest family when there's enough training data; needs the "
        "'xgboost' extra installed and is slower to train than the simpler families.",
    ),
}
_FAMILY_ALIASES: dict[str, str] = {"xgboost": "xgb"}

_LSTM_DESCRIPTION = (
    "A recurrent (LSTM/GRU) model over the full sequence of segments in a trip, rather "
    "than one flat row -- can capture within-trip dynamics the row-level families can't. "
    "Needs the 'torch' extra installed, more data, and more time to train."
)

_BASELINE_DESCRIPTIONS: dict[str, str] = {
    "schedule": "Always predicts zero delay -- i.e. the published schedule is exactly right. "
    "The floor every real model must beat (docs/PLAN.md rule 5: baselines always ship).",
    "constant_speed": "Predicts time-to-arrival as remaining distance over one average speed "
    "fit at train time. A second, slightly stronger floor than the schedule baseline.",
}


def _known_family_and_baseline_names() -> tuple[set[str], set[str]]:
    """The real, authoritative name sets `get_model_class` accepts -- read from
    `stoptimes.models.common.contract_adapter`'s own (private) mapping so this
    module can never drift from what actually resolves at train time."""
    from stoptimes.models.common import contract_adapter as ca

    families = {*ca._FAMILY_FACTORIES, "lstm"}  # noqa: SLF001 -- see module docstring
    baselines = set(ca._BASELINE_FACTORIES)  # noqa: SLF001
    return families, baselines


def _family_info(name: str) -> ModelFamilyInfo:
    if name == "lstm":
        from stoptimes.models.common.contract_adapter import LSTMModelAdapter

        try:
            from stoptimes.models.lstm.train import train_lstm

            params = _params_from_signature(train_lstm, only=LSTMModelAdapter._LSTM_FIT_KWARGS)  # noqa: SLF001
        except ImportError:
            # torch extra not installed -- still list the family (it's a valid
            # choice to write into experiment.yaml), just without introspected params.
            params = [
                ParamInfo(
                    name=n,
                    default=None,
                    type_hint="unknown",
                    description=_param_description(n),
                    tunable=False,
                )
                for n in sorted(LSTMModelAdapter._LSTM_FIT_KWARGS)  # noqa: SLF001
            ]
        return ModelFamilyInfo(name="lstm", description=_LSTM_DESCRIPTION, params=params)

    canonical = _FAMILY_ALIASES.get(name, name)
    source = _FAMILY_SOURCES.get(canonical)
    if source is None:
        return ModelFamilyInfo(
            name=name,
            description=f"Family '{name}' (no description authored yet in stoptimes-train's TUI registry).",
            params=[],
        )
    import importlib

    cls = getattr(importlib.import_module(source.module), source.cls_name)
    params = _params_from_signature(cls.__init__)
    return ModelFamilyInfo(name=name, description=source.description, params=params)


def list_model_families(*, include_baselines: bool = False) -> list[ModelFamilyInfo]:
    """Every model family `get_model_class` actually accepts, each with a
    human description and its real hyperparameters (introspected, not
    hand-copied). `xgb` and `xgboost` both appear (both are valid
    `ModelSpec.family` values). Baselines (`schedule`, `constant_speed`) are
    included only when `include_baselines=True` -- they're always trained
    for comparison (`Experiment.baselines`) rather than chosen as `model.family`.
    """
    family_names, baseline_names = _known_family_and_baseline_names()
    names = set(family_names) | set(_FAMILY_ALIASES)
    infos = [_family_info(name) for name in sorted(names)]
    if include_baselines:
        infos += [
            ModelFamilyInfo(
                name=name,
                description=_BASELINE_DESCRIPTIONS.get(name, "Baseline predictor."),
                params=[],
                is_baseline=True,
            )
            for name in sorted(baseline_names)
        ]
    return infos


def list_baselines() -> list[ModelFamilyInfo]:
    """Baseline predictors available for `Experiment.baselines`."""
    _, baseline_names = _known_family_and_baseline_names()
    return [
        ModelFamilyInfo(
            name=name,
            description=_BASELINE_DESCRIPTIONS.get(name, "Baseline predictor."),
            params=[],
            is_baseline=True,
        )
        for name in sorted(baseline_names)
    ]
