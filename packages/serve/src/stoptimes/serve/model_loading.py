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
import os
import pickle  # noqa: S403 - BYO pickle loading is an explicit, documented opt-in
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    ModelFormat,
    artifact_dir,
    read_metadata,
    version_mismatches,
)
from stoptimes.core.contracts.model import Model, ModelRef, ModelRefKind
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact
from stoptimes.core.logging import get_logger
from stoptimes.models.common.provenance import lib_versions as _installed_lib_versions

__all__ = [
    "iter_registry_artifacts",
    "resolve_artifact",
    "load_artifact_model",
    "check_version_drift",
    "ArtifactVersionDriftError",
    "LOAD_CRITICAL_LIBS",
    "ALLOW_VERSION_DRIFT_ENV",
    "PickleModelLoader",
    "OnnxModelLoader",
    "CallableModelLoader",
    "HttpModelLoader",
    "load_model_ref",
    "CallableModel",
    "HttpModel",
]

_logger = get_logger("model_loading")

ALLOW_VERSION_DRIFT_ENV = "STOPTIMES_ALLOW_VERSION_DRIFT"
"""Override switch for a refused artifact (see `load_artifact_model`).

Set to ``"1"``/``"true"``/``"all"``/``"*"`` to override for every artifact,
or to a comma-separated list of artifact ids (e.g.
``"xgb_mbta_2026,poly_v3"``) to override only those. Unset/empty means no
override -- the default is to refuse, never to load quietly across a
drifted load-critical library.
"""

LOAD_CRITICAL_LIBS: dict[ModelFormat, frozenset[str]] = {
    # PICKLE covers every family produced by `models.common.contract_adapter`
    # today (historical_mean, ewma, polyreg_*, xgb, lstm baselines) -- the
    # pickled object's class may come from either library depending on
    # family, so both are load-critical for this format. numpy/pandas are
    # deliberately excluded: they are far more stable across minor versions
    # for the plain arrays/frames these models carry, and treating every
    # numpy point release as unloadable would make the refusal useless noise
    # (see docs/PLAN.md U7 and docs/CONTRACTS.md).
    ModelFormat.PICKLE: frozenset({"sklearn", "xgboost"}),
    # Not yet loaded generically (see `load_artifact_model`'s NotImplementedError
    # below), but declared here so the table is complete once a per-family
    # loader is wired for it.
    ModelFormat.XGBOOST_JSON: frozenset({"xgboost"}),
    ModelFormat.ONNX: frozenset({"onnxruntime"}),
    ModelFormat.TORCHSCRIPT: frozenset({"torch"}),
}
"""Which libraries, if drifted, make an artifact of this `ModelFormat` unsafe
to deserialize. A drift in any other tracked library (e.g. numpy, pandas, or
anything not in this set -- a matplotlib drift, say) is informational only:
logged as a warning, never a refusal. See `version_mismatches` in
contracts/artifact.py for the comparison itself (major.minor, built vs.
installed)."""


class ArtifactVersionDriftError(RuntimeError):
    """Raised when a load-critical library has drifted since an artifact was
    built and no override was given. See `check_version_drift`."""


def _lib_name(mismatch: str) -> str:
    """`"xgboost: built with 3.1.2, installed 3.0.5"` -> `"xgboost"`."""
    return mismatch.split(":", 1)[0]


def _drift_override_requested(artifact_id: str) -> bool:
    raw = os.environ.get(ALLOW_VERSION_DRIFT_ENV, "").strip()
    if not raw:
        return False
    if raw.lower() in ("1", "true", "all", "*"):
        return True
    allowed_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return artifact_id in allowed_ids


def check_version_drift(meta: ArtifactMetadata, *, allow_override: bool | None = None) -> None:
    """Refuse to proceed if a load-critical library has drifted since
    `meta` was built, unless overridden.

    - Artifacts with no recorded `provenance.lib_versions` (pre-Phase-1, or
      any producer that never populated it) are never refused: absence of
      information is not evidence of drift.
    - A drift in a library that is not load-critical for `meta.format`
      (see `LOAD_CRITICAL_LIBS`) is logged as a warning and never blocks
      loading.
    - A drift in a load-critical library raises `ArtifactVersionDriftError`
      unless `allow_override` is True (explicit, per-call) or the
      `STOPTIMES_ALLOW_VERSION_DRIFT` env var names this artifact (or
      "all"/"1"/"true"/"*"). An override is always logged loudly, naming the
      drift, so it is never a silent default.
    """
    if not meta.provenance.lib_versions:
        return

    mismatches = version_mismatches(meta, _installed_lib_versions())
    if not mismatches:
        return

    critical_libs = LOAD_CRITICAL_LIBS.get(meta.format, frozenset())
    critical = [m for m in mismatches if _lib_name(m) in critical_libs]
    informational = [m for m in mismatches if _lib_name(m) not in critical_libs]

    for note in informational:
        _logger.warning(
            "artifact library drift (non-load-critical, proceeding)",
            artifact_id=meta.artifact_id,
            drift=note,
        )

    if not critical:
        return

    override = _drift_override_requested(meta.artifact_id) if allow_override is None else allow_override
    detail = "; ".join(critical)
    if override:
        _logger.warning(
            "OVERRIDE: loading artifact despite load-critical library drift",
            artifact_id=meta.artifact_id,
            drift=detail,
        )
        return

    raise ArtifactVersionDriftError(
        f"artifact {meta.artifact_id!r} was built with different library versions than are "
        f"installed here, for a library that deserializes the model: {detail}. Re-export the "
        "artifact under the versions installed on this host (or the deploy target's versions, "
        "if this is a deploy), or, if you know this artifact is still good, override with "
        f"{ALLOW_VERSION_DRIFT_ENV}=all (or a comma-separated list of artifact ids including "
        f"{meta.artifact_id!r}) -- this is logged loudly, never the default."
    )


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


def load_artifact_model(
    meta: ArtifactMetadata, registry_dir: Path | str, *, allow_version_drift: bool | None = None
) -> Model:
    """Load the fitted model a registry artifact points at.

    Before opening anything, refuses (see `check_version_drift`) when a
    library that deserializes this `meta.format` has drifted between build
    time (`meta.provenance.lib_versions`) and what's installed now, unless
    `allow_version_drift` (or the `STOPTIMES_ALLOW_VERSION_DRIFT` env var)
    overrides it for this artifact.

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
    check_version_drift(meta, allow_override=allow_version_drift)
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
