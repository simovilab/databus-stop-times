"""Model resolution: pick an artifact per trip, then load it.

Two separate concerns, both required by docs/CONTRACTS.md §4/§7b:

- **Selection** (`resolve_artifact`): which `artifact.json` in the registry
  should serve a given trip. Ranks by reading `artifact.json` files only --
  never opens a pickle to choose -- via `core.contracts.selection.select_artifact`.
- **Loading** (`ModelLoaders`, `load_model_ref`): bring-your-own models
  described by a `core.contracts.model.ModelRef` (pickle / onnx / a Python
  callable / an HTTP endpoint). Heavy/optional imports (`onnxruntime`,
  `httpx`) are deferred into each loader's `load` so `import stoptimes.serve`
  stays light and an uninstalled optional part fails only when actually used.
"""

from __future__ import annotations

import importlib
import pickle  # noqa: S403 - BYO pickle loading is an explicit, documented opt-in
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import ArtifactMetadata, ModelFormat, artifact_dir, read_metadata
from stoptimes.core.contracts.model import Model, ModelRef, ModelRefKind
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact

__all__ = [
    "iter_registry_artifacts",
    "resolve_artifact",
    "load_artifact_model",
    "PickleModelLoader",
    "OnnxModelLoader",
    "CallableModelLoader",
    "HttpModelLoader",
    "load_model_ref",
    "CallableModel",
    "HttpModel",
]


# ---------------------------------------------------------------------------
# Selection: which artifact.json wins
# ---------------------------------------------------------------------------


def iter_registry_artifacts(registry_dir: Path | str) -> list[ArtifactMetadata]:
    """Read every `<registry_dir>/<artifact_id>/artifact.json`, skipping
    directories that don't have one (nothing is ranked from a pickle)."""
    root = Path(registry_dir)
    if not root.is_dir():
        return []
    metadatas: list[ArtifactMetadata] = []
    for child in sorted(root.iterdir()):
        meta_path = child / "artifact.json"
        if meta_path.is_file():
            metadatas.append(read_metadata(child))
    return metadatas


def resolve_artifact(
    registry_dir: Path | str,
    *,
    source_id: str,
    policy: SelectionPolicy | None = None,
    route_id: str | None = None,
    direction_id: int | None = None,
    trip_id: str | None = None,
    artifact_id: str | None = None,
) -> ArtifactMetadata | None:
    """The artifact that should serve this trip.

    `artifact_id` is a hard `--artifact` CLI override: when given, it wins
    outright (looked up directly) rather than going through selection at all.
    """
    candidates = iter_registry_artifacts(registry_dir)
    if artifact_id is not None:
        return next((c for c in candidates if c.artifact_id == artifact_id), None)
    return select_artifact(
        candidates, policy, source_id=source_id, route_id=route_id, direction_id=direction_id, trip_id=trip_id
    )


def load_artifact_model(meta: ArtifactMetadata, registry_dir: Path | str) -> Model:
    """Load the fitted model a registry artifact points at.

    Only `ModelFormat.PICKLE` is wired up here: it covers every family
    `stoptimes.models.common.registry.ModelRegistry.save_model` produces
    today (`historical_mean`, `polyreg_distance`, `polyreg_time`, `xgboost`).
    A pickled object that doesn't already carry `family`/`required_features`/
    `artifact_id` (pre-`Model`-protocol pickles) gets them duck-typed on from
    the metadata, so it satisfies `contracts.model.Model` without a rewrite.
    Other formats (`xgboost-json`, `onnx`, `torchscript`) need a per-family
    loader T3 owns (`contracts.model.Model.load`); this raises a clear error
    rather than guessing.
    """
    directory = artifact_dir(registry_dir, meta.artifact_id)
    if meta.format is ModelFormat.PICKLE:
        with open(directory / meta.model_filename, "rb") as fh:  # noqa: S301 - registry-owned artifact
            obj = pickle.load(fh)
        if not hasattr(obj, "predict"):
            raise TypeError(f"artifact {meta.artifact_id!r} does not look like a Model (no .predict)")
        if not getattr(obj, "family", None):
            obj.family = meta.family
        if not getattr(obj, "required_features", None):
            obj.required_features = meta.features
        if not getattr(obj, "artifact_id", None):
            obj.artifact_id = meta.artifact_id
        return obj
    raise NotImplementedError(
        f"artifact {meta.artifact_id!r} uses format {meta.format.value!r}; only pickle artifacts "
        "load generically today (T3 owns per-family loaders for xgboost-json/onnx/torchscript)"
    )


# ---------------------------------------------------------------------------
# Loading: ModelRef -> Model
# ---------------------------------------------------------------------------


class CallableModel:
    """Wraps a plain `function(X: pd.DataFrame) -> np.ndarray` as a `Model`."""

    def __init__(self, func: Any, *, family: str, required_features: list[str]) -> None:
        self._func = func
        self.family = family
        self.required_features = required_features

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "CallableModel":
        raise NotImplementedError("a BYO callable model is inference-only; train it out-of-band")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._func(X))

    def save(self, directory: Path) -> Any:
        raise NotImplementedError("a BYO callable model is not re-saved by stoptimes")

    @classmethod
    def load(cls, directory: Path) -> "CallableModel":
        raise NotImplementedError("use ModelLoader/load_model_ref for a BYO callable, not Model.load")


class HttpModel:
    """Wraps a remote prediction endpoint as a `Model`.

    POSTs `{"rows": [{feature: value, ...}, ...]}` and expects
    `{"predictions": [float, ...]}` back, one per row, same order.
    """

    def __init__(
        self,
        url: str,
        *,
        family: str,
        required_features: list[str],
        headers: dict[str, str] | None = None,
        auth_env: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._url = url
        self.family = family
        self.required_features = required_features
        self._headers = dict(headers or {})
        self._auth_env = dict(auth_env or {})
        self._timeout_s = timeout_s

    def _request_headers(self) -> dict[str, str]:
        import os

        headers = dict(self._headers)
        for header_name, env_var in self._auth_env.items():
            value = os.environ.get(env_var)
            if value is not None:
                headers[header_name] = value
        return headers

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "HttpModel":
        raise NotImplementedError("an HTTP model is inference-only")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import httpx

        rows = X.to_dict(orient="records")
        try:
            response = httpx.post(
                self._url, json={"rows": rows}, headers=self._request_headers(), timeout=self._timeout_s
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP model endpoint {self._url!r} failed: {exc}") from exc
        payload = response.json()
        predictions = payload.get("predictions") if isinstance(payload, dict) else payload
        if predictions is None or len(predictions) != len(rows):
            raise RuntimeError(
                f"HTTP model endpoint {self._url!r} returned {len(predictions or [])} "
                f"predictions for {len(rows)} rows"
            )
        return np.asarray(predictions, dtype=float)

    def save(self, directory: Path) -> Any:
        raise NotImplementedError("an HTTP model is not re-saved by stoptimes")

    @classmethod
    def load(cls, directory: Path) -> "HttpModel":
        raise NotImplementedError("use ModelLoader/load_model_ref for an HTTP model, not Model.load")


class PickleModelLoader:
    """`ModelRef(kind=pickle)` -> whatever `Model` was pickled at `location`."""

    def load(self, ref: ModelRef) -> Model:
        with open(ref.location, "rb") as fh:  # noqa: S301 - documented BYO opt-in
            obj = pickle.load(fh)
        if not hasattr(obj, "predict"):
            raise TypeError(f"pickle at {ref.location!r} does not look like a Model (no .predict)")
        if not hasattr(obj, "family"):
            obj.family = ref.family
        if not hasattr(obj, "required_features") or not obj.required_features:
            obj.required_features = ref.required_features
        return obj


class OnnxModelLoader:
    """`ModelRef(kind=onnx)` -> an in-process ONNXRuntime session wrapped as `Model`.

    `onnxruntime` is an optional dependency; the import is deferred here so
    `import stoptimes.serve` never requires it, and only a caller that
    actually resolves an ONNX `ModelRef` needs it installed.
    """

    def load(self, ref: ModelRef) -> Model:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised only when onnxruntime is absent
            raise ImportError(
                "ONNX models need the optional 'onnxruntime' package: pip install onnxruntime"
            ) from exc

        session = ort.InferenceSession(ref.location)
        input_name = session.get_inputs()[0].name
        required = ref.required_features

        class _OnnxModel:
            family = ref.family
            required_features = required

            def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> Any:
                raise NotImplementedError("an ONNX model is inference-only")

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                cols = required or list(X.columns)
                inputs = X[cols].to_numpy(dtype=np.float32)
                (outputs,) = session.run(None, {input_name: inputs})
                return np.asarray(outputs).reshape(-1)

            def save(self, directory: Path) -> Any:
                raise NotImplementedError("an ONNX model is not re-saved by stoptimes")

            @classmethod
            def load(cls, directory: Path) -> Any:
                raise NotImplementedError("use ModelLoader/load_model_ref for ONNX, not Model.load")

        return _OnnxModel()


class CallableModelLoader:
    """`ModelRef(kind=callable)` -> a `"module.sub:function"` import."""

    def load(self, ref: ModelRef) -> Model:
        if ":" not in ref.location:
            raise ValueError(f"callable ModelRef.location must be 'module:function', got {ref.location!r}")
        module_name, _, func_name = ref.location.rpartition(":")
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return CallableModel(func, family=ref.family, required_features=ref.required_features)


class HttpModelLoader:
    """`ModelRef(kind=http)` -> a remote endpoint wrapped as `Model`."""

    def load(self, ref: ModelRef) -> Model:
        return HttpModel(
            ref.location,
            family=ref.family,
            required_features=ref.required_features,
            headers=ref.headers,
            auth_env=ref.auth_env,
            timeout_s=ref.timeout_s,
        )


_LOADERS: dict[ModelRefKind, Any] = {
    ModelRefKind.PICKLE: PickleModelLoader(),
    ModelRefKind.ONNX: OnnxModelLoader(),
    ModelRefKind.CALLABLE: CallableModelLoader(),
    ModelRefKind.HTTP: HttpModelLoader(),
}


def load_model_ref(ref: ModelRef) -> Model:
    """Dispatch a `ModelRef` to the loader matching its `kind`."""
    return _LOADERS[ref.kind].load(ref)
