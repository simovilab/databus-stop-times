"""`MlflowRegistry`: artifacts stored as MLflow model versions (docs/MLFLOW_INTEGRATION.md §0).

Layout on the server (D1):
- one registered model per agency, ``<model_prefix><source_id>`` (``stoptimes-mbta``); artifacts
  without a source go to ``<model_prefix>global``;
- one model version per published artifact. Its files are a pyfunc model whose
  ``artifacts/<artifact_id>/`` directory is the stoptimes artifact, unchanged;
- version tags (D3) carry what selection needs, including the whole ``artifact.json``
  (``stoptimes.metadata``, gzip+base64) when it fits, so listing downloads nothing;
- each publish also records a run in experiment ``<experiment_prefix><source_id>``.

Loading (D2) downloads a version once into an immutable local cache and hands the normal
filesystem loader a directory, so drift refusal, asset hashes and feature checks still apply.

Only `mlflow-skinny` (extra ``stoptimes-core[mlflow]``) is needed to read; every MLflow import is
lazy. Credentials come from MLflow itself: ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``
or ``~/.mlflow/credentials`` (D7). Nothing here reads or logs them.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import gzip
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from stoptimes.core.contracts.artifact import METADATA_FILENAME, ArtifactMetadata, artifact_dir, read_metadata
from stoptimes.core.registry.base import ArtifactExistsError, RegistryError, SkipHandler

TAG_PREFIX = "stoptimes."
TAG_ARTIFACT_ID = "stoptimes.artifact_id"
TAG_METADATA = "stoptimes.metadata"
#: Model-version tag values are capped server-side (5000 chars on older stores); stay under it.
MAX_TAG_VALUE = 4900
CACHE_ENV = "STOPTIMES_CACHE_DIR"
_COMPLETE_MARKER = ".complete"
_INSTALL_HINT = "the MLflow registry backend needs the extra: pip install 'stoptimes-core[mlflow]'"
_AUTH_HINT = (
    "check MLflow credentials: MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, or ~/.mlflow/credentials"
)

# mlflow's fluent state (tracking/registry URI) is process-global; publishing binds it under this lock.
_GLOBAL_URI_LOCK = threading.Lock()


@dataclass(frozen=True)
class RegistryVersion:
    """One model version that holds a stoptimes artifact."""

    name: str
    version: int
    artifact_id: str
    aliases: tuple[str, ...]
    run_id: str | None
    created_ms: int
    meta: ArtifactMetadata


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RegistryError(_INSTALL_HINT) from exc
    return mlflow


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return slug or "global"


def encode_metadata(meta: ArtifactMetadata) -> str | None:
    """`artifact.json` as a compact tag value, or None when it would not fit in one tag."""
    raw = meta.model_dump_json().encode("utf-8")
    value = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    return value if len(value) <= MAX_TAG_VALUE else None


def decode_metadata(value: str) -> ArtifactMetadata:
    return ArtifactMetadata.model_validate_json(gzip.decompress(base64.b64decode(value)))


def version_tags(meta: ArtifactMetadata) -> dict[str, str]:
    """The tags a published version carries (D3). Values are strings, as MLflow requires."""
    tags = {
        TAG_ARTIFACT_ID: meta.artifact_id,
        "stoptimes.family": meta.family,
        "stoptimes.target": meta.target,
        "stoptimes.format": meta.format.value,
        "stoptimes.scope_level": meta.scope.level.value,
        "stoptimes.schema_version": str(meta.schema_version),
    }
    optional = {
        "stoptimes.source_id": meta.scope.source_id,
        "stoptimes.route_ids": ",".join(meta.scope.route_ids) or None,
        "stoptimes.timezone": meta.timezone,
        "stoptimes.holiday_region": meta.holiday_region,
        "stoptimes.test_mae_s": _fmt(meta.metrics.get("test_mae_seconds")),
        "stoptimes.val_mae_s": _fmt(meta.metrics.get("val_mae_seconds")),
    }
    tags.update({k: v for k, v in optional.items() if v is not None})
    encoded = encode_metadata(meta)
    if encoded is not None:
        tags[TAG_METADATA] = encoded
    return tags


def _fmt(value: float | None) -> str | None:
    return None if value is None else f"{value:.3f}"


class MlflowRegistry:
    """A `RegistryBackend` on an MLflow tracking server (or any MLflow store URI)."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        registry_uri: str | None = None,
        model_prefix: str = "stoptimes-",
        experiment_prefix: str = "stoptimes/",
        cache_dir: Path | str | None = None,
        artifact_location: str | None = None,
        list_ttl_s: float = 60.0,
    ) -> None:
        """
        Args:
            tracking_uri: e.g. ``https://ml.simovi.org`` or ``sqlite:///…`` (tests).
            artifact_location: where experiments this backend *creates* store files; None lets the
                server decide (proxied ``mlflow-artifacts:`` on ml.simovi.org). Tests pass a tmp dir.
            list_ttl_s: how long a listing is reused; MLflow calls can take seconds each.
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.model_prefix = model_prefix
        self.experiment_prefix = experiment_prefix
        self.artifact_location = artifact_location
        self.list_ttl_s = list_ttl_s
        host = _slug(urlparse(tracking_uri).netloc or urlparse(tracking_uri).path or tracking_uri)
        base = Path(cache_dir) if cache_dir is not None else _default_cache_root()
        self.cache_dir = base.expanduser() / host
        self._lock = threading.RLock()
        self._listing: tuple[float, dict[str, RegistryVersion], tuple[RegistryVersion, ...]] | None = None
        self._by_fingerprint: dict[str, RegistryVersion] = {}
        self._client_obj: Any = None

    def __repr__(self) -> str:
        return f"MlflowRegistry({self.tracking_uri!r})"

    @property
    def uri(self) -> str:
        return f"mlflow+{self.tracking_uri}"

    # -- RegistryBackend -----------------------------------------------------------------------

    def list(self, *, strict: bool = False, on_skip: SkipHandler | None = None) -> list[ArtifactMetadata]:
        latest = self._latest(strict=strict, on_skip=on_skip)
        return [latest[aid].meta for aid in sorted(latest)]

    def get(self, artifact_id: str) -> ArtifactMetadata | None:
        found = self._latest().get(artifact_id)
        return found.meta if found is not None else None

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._latest(refresh=True)

    def describe(self, artifact_id: str) -> str:
        found = self._latest().get(artifact_id)
        if found is None:
            return f"{self.uri} (artifact {artifact_id})"
        return f"{self.uri} ({found.name} v{found.version})"

    def local_root(self, meta: ArtifactMetadata) -> Path:
        return self._download(self._version_for(meta))

    def card(self, artifact_id: str) -> str | None:
        found = self._latest().get(artifact_id)
        if found is None:
            return None
        path = artifact_dir(self._download(found), artifact_id) / "card.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    @contextlib.contextmanager
    def staged(self, artifact_id: str, *, overwrite: bool = False) -> Iterator[Path]:
        if not overwrite and self.exists(artifact_id):
            raise ArtifactExistsError(f"artifact {artifact_id!r} already exists at {self.describe(artifact_id)}")
        with tempfile.TemporaryDirectory(prefix="stoptimes-publish-") as tmp:
            directory = Path(tmp) / artifact_id  # the basename becomes artifacts/<artifact_id>/ on the server
            directory.mkdir()
            yield directory
            self.publish(directory)

    # -- MLflow-specific -----------------------------------------------------------------------

    def model_name(self, meta: ArtifactMetadata) -> str:
        return f"{self.model_prefix}{_slug(meta.scope.source_id or 'global')}"

    def experiment_name(self, meta: ArtifactMetadata) -> str:
        return f"{self.experiment_prefix}{_slug(meta.scope.source_id or 'global')}"

    def versions(self, name: str | None = None) -> builtins.list[RegistryVersion]:
        """Every stoptimes version (optionally of one registered model), oldest first."""
        found = self._search(name=name)
        return sorted(found, key=lambda v: (v.name, v.version))

    def listing(self, *, refresh: bool = False) -> builtins.list[RegistryVersion]:
        """Every stoptimes version, sorted by (name, version), from the same TTL cache as `list()`.

        Versions that can't be read are left out. Use this where a request must stay cheap (the val
        API lists it on every page load); `versions()` always asks the server.
        """
        self._latest(refresh=refresh)
        with self._lock:
            assert self._listing is not None
            return builtins.list(self._listing[2])

    def get_version(self, name: str, version: int | str) -> RegistryVersion:
        mv = self._call(lambda c: c.get_model_version(name, str(version)), f"{name} v{version}")
        return self._to_version(mv, strict=True)

    def resolve(self, ref: str) -> RegistryVersion:
        """``<name>@<alias>``, ``<name>/<version>`` or a bare artifact id (its latest version)."""
        if "@" in ref:
            name, alias = ref.split("@", 1)
            mv = self._call(lambda c: c.get_model_version_by_alias(name, alias), ref)
            return self._to_version(mv, strict=True)
        if "/" in ref:
            name, version = ref.rsplit("/", 1)
            return self.get_version(name, version)
        found = self._latest().get(ref)
        if found is None:
            raise RegistryError(f"{self.uri}: no artifact {ref!r}")
        return found

    def set_alias(self, name: str, alias: str, version: int | str) -> None:
        self._call(lambda c: c.set_registered_model_alias(name, alias, str(version)), f"{name}@{alias}")
        self._invalidate()

    def delete_alias(self, name: str, alias: str) -> None:
        self._call(lambda c: c.delete_registered_model_alias(name, alias), f"{name}@{alias}")
        self._invalidate()

    def publish(self, directory: Path) -> RegistryVersion:
        """Upload an artifact directory as a new model version (D1). Returns the version."""
        from mlflow.exceptions import MlflowException

        try:
            return self._publish(directory)
        except MlflowException as exc:
            code = getattr(exc, "error_code", "")
            hint = f"; {_AUTH_HINT}" if code in ("UNAUTHENTICATED", "PERMISSION_DENIED") else ""
            raise RegistryError(f"{self.uri}: publish {directory.name!r}: {exc}{hint}") from exc

    def _publish(self, directory: Path) -> RegistryVersion:
        mlflow = _mlflow()
        from mlflow.entities import LoggedModelStatus

        from stoptimes.core.registry._pyfunc import ARTIFACT_KEY, StoptimesPyfunc

        meta = read_metadata(directory)
        if meta.artifact_id != directory.name:
            raise RegistryError(f"artifact.json says {meta.artifact_id!r} but the directory is {directory.name!r}")
        name = self.model_name(meta)
        client = self._client()
        with _bound_uris(mlflow, self.tracking_uri, self.registry_uri):
            experiment_id = self._ensure_experiment(client, self.experiment_name(meta))
            run = client.create_run(
                experiment_id, run_name=meta.artifact_id, tags={TAG_ARTIFACT_ID: meta.artifact_id}
            )
            run_id = run.info.run_id
            try:
                _log_artifact_summary(client, run_id, meta, directory)
                with tempfile.TemporaryDirectory(prefix="stoptimes-pyfunc-") as tmp:
                    model_path = Path(tmp) / "model"
                    with _quiet_mlflow():
                        mlflow.pyfunc.save_model(
                            path=str(model_path),
                            python_model=StoptimesPyfunc(),
                            artifacts={ARTIFACT_KEY: str(directory)},
                            pip_requirements=_requirements(meta),
                        )
                    logged = client.create_logged_model(
                        experiment_id=experiment_id,
                        name=meta.artifact_id,
                        source_run_id=run_id,
                        tags={TAG_ARTIFACT_ID: meta.artifact_id},
                    )
                    client.log_model_artifacts(logged.model_id, str(model_path))
                    client.finalize_logged_model(logged.model_id, LoggedModelStatus.READY)
                self._ensure_registered_model(client, name)
                mv = client.create_model_version(
                    name,
                    source=f"models:/{logged.model_id}",
                    run_id=run_id,
                    tags=version_tags(meta),
                    model_id=logged.model_id,
                )
                client.set_terminated(run_id, "FINISHED")
            except BaseException:
                with contextlib.suppress(Exception):
                    client.set_terminated(run_id, "FAILED")
                raise
        self._invalidate()
        return self._to_version(mv, strict=True, meta=meta)

    # -- internals -----------------------------------------------------------------------------

    def _client(self) -> Any:
        if self._client_obj is None:
            from mlflow import MlflowClient

            _mlflow()
            self._client_obj = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
        return self._client_obj

    def _call(self, fn: Any, what: str) -> Any:
        from mlflow.exceptions import MlflowException

        try:
            return fn(self._client())
        except MlflowException as exc:
            code = getattr(exc, "error_code", "")
            hint = f"; {_AUTH_HINT}" if code in ("UNAUTHENTICATED", "PERMISSION_DENIED") else ""
            raise RegistryError(f"{self.uri}: {what}: {exc.message if hasattr(exc, 'message') else exc}{hint}") from exc

    def _invalidate(self) -> None:
        with self._lock:
            self._listing = None

    def _latest(
        self, *, refresh: bool = False, strict: bool = False, on_skip: SkipHandler | None = None
    ) -> dict[str, RegistryVersion]:
        with self._lock:
            now = time.monotonic()
            if not refresh and self._listing is not None and now - self._listing[0] < self.list_ttl_s:
                return self._listing[1]
            every = sorted(self._search(strict=strict, on_skip=on_skip), key=lambda v: (v.name, v.version))
            latest: dict[str, RegistryVersion] = {}
            for v in every:
                current = latest.get(v.artifact_id)
                if current is None or (v.created_ms, v.version) > (current.created_ms, current.version):
                    latest[v.artifact_id] = v
            self._listing = (now, latest, tuple(every))
            return latest

    def _search(
        self, *, name: str | None = None, strict: bool = False, on_skip: SkipHandler | None = None
    ) -> builtins.list[RegistryVersion]:
        filter_string = f"name = '{name}'" if name else f"name LIKE '{self.model_prefix}%'"
        aliases = self._aliases(filter_string)
        found: builtins.list[RegistryVersion] = []
        token: str | None = None
        while True:
            page = self._call(
                lambda c, t=token: c.search_model_versions(filter_string, max_results=1000, page_token=t),
                "search model versions",
            )
            for mv in page:
                if TAG_ARTIFACT_ID not in (mv.tags or {}) or str(getattr(mv, "status", "READY")) != "READY":
                    continue
                try:
                    found.append(self._to_version(mv, strict=True, aliases=aliases.get((mv.name, int(mv.version)))))
                except Exception as exc:  # noqa: BLE001 - one bad version must not hide the rest
                    if strict:
                        raise
                    if on_skip is not None:
                        on_skip(f"{mv.name} v{mv.version}", exc)
            token = getattr(page, "token", None)
            if not token:
                return found

    def _aliases(self, filter_string: str) -> dict[tuple[str, int], tuple[str, ...]]:
        """(registered model, version) -> its aliases (search_model_versions leaves them out)."""
        found: dict[tuple[str, int], builtins.list[str]] = {}
        token: str | None = None
        while True:
            page = self._call(
                lambda c, t=token: c.search_registered_models(filter_string, max_results=1000, page_token=t),
                "search registered models",
            )
            for rm in page:
                for alias, version in (getattr(rm, "aliases", None) or {}).items():
                    found.setdefault((rm.name, int(version)), []).append(alias)
            token = getattr(page, "token", None)
            if not token:
                return {key: tuple(sorted(names)) for key, names in found.items()}

    def _to_version(
        self,
        mv: Any,
        *,
        strict: bool,
        meta: ArtifactMetadata | None = None,
        aliases: tuple[str, ...] | None = None,
    ) -> RegistryVersion:
        tags = dict(mv.tags or {})
        artifact_id = tags.get(TAG_ARTIFACT_ID)
        if not artifact_id:
            raise RegistryError(f"{mv.name} v{mv.version} is not a stoptimes artifact (no {TAG_ARTIFACT_ID} tag)")
        if meta is None:
            encoded = tags.get(TAG_METADATA)
            meta = decode_metadata(encoded) if encoded else self._fetch_metadata(mv.name, mv.version, artifact_id)
        version = RegistryVersion(
            name=mv.name,
            version=int(mv.version),
            artifact_id=artifact_id,
            aliases=aliases if aliases is not None else tuple(sorted(getattr(mv, "aliases", None) or ())),
            run_id=getattr(mv, "run_id", None),
            created_ms=int(getattr(mv, "creation_timestamp", 0) or 0),
            meta=meta,
        )
        with self._lock:
            self._by_fingerprint[_fingerprint(meta)] = version
        return version

    def _fetch_metadata(self, name: str, version: Any, artifact_id: str) -> ArtifactMetadata:
        """`artifact.json` for a version whose tags couldn't hold it: downloaded once, cached."""
        target = self._version_cache(name, int(version)) / "metadata" / METADATA_FILENAME
        if not target.is_file():
            mlflow = _mlflow()
            with tempfile.TemporaryDirectory(prefix="stoptimes-meta-") as tmp, _bound_uris(
                mlflow, self.tracking_uri, self.registry_uri
            ):
                path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{name}/{version}/artifacts/{artifact_id}/{METADATA_FILENAME}",
                    dst_path=tmp,
                    tracking_uri=self.tracking_uri,
                    registry_uri=self.registry_uri,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return read_metadata(target.parent)

    def _version_for(self, meta: ArtifactMetadata) -> RegistryVersion:
        with self._lock:
            found = self._by_fingerprint.get(_fingerprint(meta))
        if found is not None:
            return found
        latest = self._latest(refresh=True).get(meta.artifact_id)
        if latest is not None and _fingerprint(latest.meta) == _fingerprint(meta):
            return latest
        raise RegistryError(f"{self.uri}: no model version holds this exact artifact {meta.artifact_id!r}")

    def _version_cache(self, name: str, version: int) -> Path:
        return self.cache_dir / name / str(version)

    def _download(self, v: RegistryVersion) -> Path:
        """Root directory holding ``<artifact_id>/`` for this version; downloaded once (immutable)."""
        cache = self._version_cache(v.name, v.version)
        root = cache / "files" / "artifacts"
        if (cache / _COMPLETE_MARKER).is_file():
            return root
        mlflow = _mlflow()
        cache.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cache, prefix=".download-"))
        try:
            with _bound_uris(mlflow, self.tracking_uri, self.registry_uri):
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{v.name}/{v.version}",
                    dst_path=str(tmp),
                    tracking_uri=self.tracking_uri,
                    registry_uri=self.registry_uri,
                )
            downloaded = read_metadata(artifact_dir(tmp / "artifacts", v.artifact_id))
            if _fingerprint(downloaded) != _fingerprint(v.meta):
                raise RegistryError(
                    f"{v.name} v{v.version}: artifact.json in the files differs from the version's tags"
                )
            if (cache / "files").exists():
                shutil.rmtree(cache / "files")
            os.rename(tmp, cache / "files")
            (cache / _COMPLETE_MARKER).write_text(f"{v.name} v{v.version}\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return root

    def _ensure_experiment(self, client: Any, name: str) -> str:
        existing = client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return client.create_experiment(name, artifact_location=self.artifact_location)

    @staticmethod
    def _ensure_registered_model(client: Any, name: str) -> None:
        """Create *name* unless it exists.

        A server with basic auth answers PERMISSION_DENIED, not RESOURCE_DOES_NOT_EXIST, when a
        non-admin looks up a model nobody has granted them -- which includes one that doesn't exist
        yet (ml.simovi.org does). Creating it is then the way to find out: any signed-in user may
        create a model and becomes its manager; if it exists and belongs to someone else, the
        create fails and that is reported.
        """
        from mlflow.exceptions import MlflowException

        try:
            client.get_registered_model(name)
            return
        except MlflowException as exc:
            denied = getattr(exc, "error_code", "") == "PERMISSION_DENIED"
            if not denied and getattr(exc, "error_code", "") != "RESOURCE_DOES_NOT_EXIST":
                raise
        try:
            client.create_registered_model(name)
        except MlflowException as exc:
            if not denied:
                raise
            raise RegistryError(
                f"registered model {name!r}: no permission to read it, and creating it failed "
                f"({getattr(exc, 'error_code', '')}): it probably exists and belongs to another user; "
                "ask an MLflow admin to grant you EDIT on it"
            ) from exc


def _log_artifact_summary(client: Any, run_id: str, meta: ArtifactMetadata, directory: Path) -> None:
    """The publish run carries the artifact's params, metrics, per-horizon scores and card (D6)."""
    from stoptimes.core.tracking import Tracker, artifact_metrics, artifact_params, artifact_tags

    Tracker._log_batches(client, run_id, artifact_params(meta), artifact_metrics(meta), artifact_tags(meta))
    card = directory / "card.md"
    if card.is_file():
        client.log_text(run_id, card.read_text(encoding="utf-8"), "card.md")


def _fingerprint(meta: ArtifactMetadata) -> str:
    return meta.model_dump_json()


def _requirements(meta: ArtifactMetadata) -> list[str]:
    return sorted(f"{lib}=={ver}" for lib, ver in meta.provenance.lib_versions.items() if ver)


def _default_cache_root() -> Path:
    env = os.environ.get(CACHE_ENV, "").strip()
    base = Path(env) if env else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "stoptimes"
    return base / "mlflow"


@contextlib.contextmanager
def _bound_uris(mlflow: Any, tracking_uri: str, registry_uri: str) -> Iterator[None]:
    """Point mlflow's process-global URIs at this registry while publishing or downloading.

    Resolving ``models:/…`` URIs (registering a logged model, downloading a version) goes partly
    through the global URIs rather than the client's or the arguments', so they must match. Restored afterwards; serialized so threads don't interleave.
    """
    with _GLOBAL_URI_LOCK:
        previous = (mlflow.get_tracking_uri(), mlflow.get_registry_uri())
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        try:
            yield
        finally:
            mlflow.set_tracking_uri(previous[0])
            mlflow.set_registry_uri(previous[1])


@contextlib.contextmanager
def _quiet_mlflow() -> Iterator[None]:
    """Silence mlflow's dependency-mismatch and pickling warnings while saving the pyfunc."""
    import logging
    import warnings

    logger = logging.getLogger("mlflow")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(level)
