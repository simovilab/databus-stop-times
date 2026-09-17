"""Contract: the model interface.

`Model` is what every family — `historical_mean`, `polyreg_distance`,
`polyreg_time`, `xgboost`, and anything added later — must satisfy so
`stoptimes.train` can fit it and `stoptimes.serve` can call it without
knowing which family it is. It is a `typing.Protocol`, not a base class:
existing predict.py modules (functions, not objects) satisfy it through a
thin adapter rather than a rewrite (see `contracts.adapters`).

Bring-your-own models (pickle, ONNX, a plain callable, or an HTTP endpoint)
don't implement `Model` directly — they're described by a `ModelRef`, and a
`ModelLoader` turns a `ModelRef` into something that satisfies `Model`. That
split keeps `stoptimes.core` free of pickle/onnx/requests-specific code: T4
(serve) owns the loader implementations, this module only owns the shapes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stoptimes.core.contracts.artifact import ModelFormat


@runtime_checkable
class Model(Protocol):
    """What a trained model must provide.

    `runtime_checkable` allows `isinstance(obj, Model)` as a cheap sanity
    check (it verifies the methods exist, not their signatures — call
    `fit`/`predict` to actually exercise a model).
    """

    family: str
    """Model family name, matching `contracts.experiment.ModelSpec.family`
    and `contracts.artifact.ArtifactMetadata.family`."""

    required_features: list[str]
    """Feature names (from the feature registry, T3) this model needs in `X`."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "Model":
        """Fit on features `X` and target `y`; return a fitted `Model`
        (the same instance or a new one — callers must use the return value).
        `ctx` carries anything a family needs beyond X/y (e.g. `sample_weight`,
        `eval_set`), all keyword-only so adding one never breaks callers."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one prediction of the experiment's `target` per row of `X`."""
        ...

    def save(self, directory: Path) -> ModelFormat:
        """Serialize the fitted model under `directory` (see contracts.artifact
        for the directory layout) and return the format used."""
        ...

    @classmethod
    def load(cls, directory: Path) -> "Model":
        """Deserialize a model previously written by `save` from `directory`."""
        ...


class ModelLoader(Protocol):
    """Resolves a `ModelRef` (bring-your-own) into something that satisfies `Model`.

    Distinct from `Model.load`: a family's own `load` reconstructs *that*
    family from its own artifact directory; a `ModelLoader` reconstructs
    *any* family from a `ModelRef` whose `kind` it knows how to handle
    (pickle/onnx/callable/http) — the thing T4's BYO support needs.
    """

    def load(self, ref: ModelRef) -> Model:
        ...


class ModelRefKind(str, Enum):
    """How a bring-your-own model is packaged."""

    PICKLE = "pickle"
    ONNX = "onnx"
    CALLABLE = "callable"
    """A Python import path `"module.sub:function"`; `function(X: pd.DataFrame) -> np.ndarray`."""
    HTTP = "http"
    """A prediction endpoint; request/response shape is serve's concern (T4), not this contract's."""


class ModelRef(BaseModel):
    """Points at a bring-your-own model without saying how to run it.

    A `ModelLoader` implementation (T4) resolves a `ModelRef` into a `Model`.
    `stoptimes.core` never imports pickle-unsafe code, onnxruntime, or an
    HTTP client on behalf of a `ModelRef` — that happens only where the
    matching loader lives.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ModelRefKind
    location: str
    """A filesystem path (pickle/onnx), an import path `module:function`
    (callable), or a URL (http)."""
    family: str = "byo"
    required_features: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    """HTTP only: static request headers."""
    auth_env: dict[str, str] = Field(default_factory=dict)
    """HTTP only: header name -> environment variable NAME holding its value
    (never a secret value in the ref itself), matching `contracts.source.RealtimeFeed.auth_env`."""
    timeout_s: float = 5.0
    """HTTP only: request timeout."""
