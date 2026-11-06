"""Experiment tracking on an MLflow server (docs/MLFLOW_INTEGRATION.md §0 D6).

Training runs go to experiment ``stoptimes/<source_id>``, validation runs to
``stoptimes-val/<source_id>``. A run is described by a `RunRecord` (pure data, built by the
callers' `*_record` helpers) and written by a `Tracker`.

Tracking is **best-effort**: `Tracker.log` never raises. A server that is down, slow or refuses
the write costs a warning, never a training or validation run. It is off unless asked for:
``$STOPTIMES_TRACKING`` (a tracking URI such as ``https://ml.simovi.org``) or a ``--track`` flag.

Every MLflow import is lazy (extra ``stoptimes-core[mlflow]``). Credentials come from MLflow itself
(``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` or ``~/.mlflow/credentials``); nothing
here reads or logs them.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from stoptimes.core.contracts.artifact import ArtifactMetadata, HorizonScore

log = logging.getLogger(__name__)

#: Turns tracking on: the MLflow tracking URI to log runs to (``mlflow+`` prefix optional).
TRACKING_ENV = "STOPTIMES_TRACKING"
TRAIN_EXPERIMENT_PREFIX = "stoptimes/"
VAL_EXPERIMENT_PREFIX = "stoptimes-val/"
TAG_KIND = "stoptimes.kind"

_MAX_PARAM_VALUE = 500  # MLflow's param value limit (older stores)
_PARAMS_PER_BATCH = 100
_METRICS_PER_BATCH = 900
_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_\-. /]")


@dataclass(frozen=True)
class RunRecord:
    """One MLflow run to write: where, what, and its child runs."""

    experiment: str
    run_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    texts: Mapping[str, str] = field(default_factory=dict)
    """Small text files to attach (``card.md``, ``report.json``), by artifact path."""
    children: tuple[RunRecord, ...] = ()


def metric_key(*parts: str) -> str:
    """A metric/param key MLflow accepts: unsupported characters (``:``, ``@``) become ``_``."""
    return ".".join(_KEY_UNSAFE.sub("_", p) for p in parts if p)


def source_slug(source_id: str | None) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", (source_id or "global").lower()).strip("-") or "global"


def tracking_uri(explicit: str | None = None) -> str | None:
    """The tracking URI: *explicit*, else ``$STOPTIMES_TRACKING``; None when tracking is off."""
    uri = (explicit if explicit is not None else os.environ.get(TRACKING_ENV, "")).strip()
    if uri.startswith("mlflow+"):
        uri = uri[len("mlflow+") :]
    return uri or None


def get_tracker(explicit: str | None = None) -> Tracker | None:
    """A `Tracker` for `tracking_uri(explicit)`, or None when tracking is off."""
    uri = tracking_uri(explicit)
    return Tracker(uri) if uri else None


# -- what a trained artifact contributes to a run ----------------------------------------------


def horizon_label(h: HorizonScore) -> str:
    return (
        f"h{h.horizon_min_s}-{h.horizon_max_s}s"
        if h.horizon_max_s is not None
        else f"h{h.horizon_min_s}s-inf"
    )


def artifact_params(meta: ArtifactMetadata) -> dict[str, Any]:
    """Run params for a trained artifact: hyperparameters, data window and features."""
    params: dict[str, Any] = {f"model.{k}": v for k, v in sorted(meta.params.items())}
    params.update(
        {
            "family": meta.family,
            "target": meta.target,
            "features": ",".join(meta.features),
            "n_features": len(meta.features),
            "scope.level": meta.scope.level.value,
            "scope.source_id": meta.scope.source_id,
            "scope.route_ids": ",".join(meta.scope.route_ids),
            "data.dataset": meta.data.dataset,
            "data.start": meta.data.start.isoformat() if meta.data.start else None,
            "data.end": meta.data.end.isoformat() if meta.data.end else None,
            "data.n_samples": meta.data.n_samples,
            "data.n_train": meta.data.n_train,
            "data.n_test": meta.data.n_test,
            "seed": meta.provenance.seed,
        }
    )
    return params


def artifact_metrics(meta: ArtifactMetadata) -> dict[str, float | None]:
    """Run metrics for a trained artifact: its headline metrics plus the per-horizon scores."""
    metrics: dict[str, float | None] = dict(meta.metrics)
    for h in meta.scores_by_horizon:
        label = horizon_label(h)
        metrics[metric_key(label, "mae_s")] = h.mae_s
        metrics[metric_key(label, "rmse_s")] = h.rmse_s
        metrics[metric_key(label, "p90_ae_s")] = h.p90_ae_s
        metrics[metric_key(label, "n")] = float(h.n)
    return metrics


def artifact_tags(meta: ArtifactMetadata) -> dict[str, str]:
    tags = {
        TAG_KIND: "train",
        "stoptimes.artifact_id": meta.artifact_id,
        "stoptimes.family": meta.family,
        "stoptimes.source_id": meta.scope.source_id or "",
    }
    if meta.provenance.git_sha:
        tags["mlflow.source.git.commit"] = meta.provenance.git_sha
    if meta.tags:
        tags["stoptimes.tags"] = ",".join(meta.tags)
    return tags


def training_record(
    meta: ArtifactMetadata,
    *,
    card: str | None = None,
    baselines: Mapping[str, Mapping[str, float | None]] | None = None,
    experiment_json: str | None = None,
) -> RunRecord:
    """The training run for *meta*; each baseline becomes a child run with its own metrics."""
    experiment = TRAIN_EXPERIMENT_PREFIX + source_slug(meta.scope.source_id)
    texts: dict[str, str] = {"artifact.json": meta.model_dump_json(indent=2)}
    if card:
        texts["card.md"] = card
    if experiment_json:
        texts["experiment.json"] = experiment_json
    children = tuple(
        RunRecord(
            experiment=experiment,
            run_name=f"{meta.artifact_id} baseline {name}",
            params={"family": name, "baseline_of": meta.artifact_id},
            metrics=dict(metrics),
            tags={TAG_KIND: "train-baseline", "stoptimes.artifact_id": meta.artifact_id},
        )
        for name, metrics in sorted((baselines or {}).items())
    )
    return RunRecord(
        experiment=experiment,
        run_name=meta.artifact_id,
        params=artifact_params(meta),
        metrics=artifact_metrics(meta),
        tags=artifact_tags(meta),
        texts=texts,
        children=children,
    )


# -- writing ------------------------------------------------------------------------------------


class Tracker:
    """Writes `RunRecord`s to one MLflow tracking server, best-effort."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._client_obj: Any = None
        self._experiments: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"Tracker({self.uri!r})"

    def run_url(self, experiment: str, run_id: str) -> str | None:
        """A link to a run this tracker wrote, in the MLflow UI (http(s) servers only)."""
        exp_id = self._experiments.get(experiment)
        if exp_id is None or not self.uri.startswith(("http://", "https://")):
            return None
        return f"{self.uri.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"

    def log(self, record: RunRecord, *, run_id: str | None = None) -> str | None:
        """Write *record* (into the existing run *run_id* if given); its run id, or None on failure.

        Never raises: tracking must not fail the work it describes.
        """
        started = time.monotonic()
        try:
            run = self._write(record, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - best-effort by design (D6)
            log.warning(
                "MLflow tracking to %s failed (%s: %s); the run itself is unaffected",
                self.uri,
                type(exc).__name__,
                exc,
            )
            return None
        log.info(
            "logged %r to MLflow run %s in %.1fs", record.run_name, run, time.monotonic() - started
        )
        return run

    def _client(self) -> Any:
        if self._client_obj is None:
            from mlflow import MlflowClient

            self._client_obj = MlflowClient(tracking_uri=self.uri)
        return self._client_obj

    def _experiment_id(self, name: str) -> str:
        if name not in self._experiments:
            client = self._client()
            existing = client.get_experiment_by_name(name)
            self._experiments[name] = (
                existing.experiment_id if existing else client.create_experiment(name)
            )
        return self._experiments[name]

    def _write(self, record: RunRecord, *, run_id: str | None, parent: str | None = None) -> str:
        client = self._client()
        tags = dict(record.tags)
        if parent:
            tags["mlflow.parentRunId"] = parent
        own = run_id is None
        if own:
            run_id = client.create_run(
                self._experiment_id(record.experiment), run_name=record.run_name, tags=tags
            ).info.run_id
            tags = {}
        assert run_id is not None
        try:
            self._log_batches(client, run_id, record.params, record.metrics, tags)
            for path, text in record.texts.items():
                client.log_text(run_id, text, path)
            for child in record.children:
                self._write(child, run_id=None, parent=run_id)
        except BaseException:
            if own:
                try:
                    client.set_terminated(run_id, "FAILED")
                except Exception:  # noqa: BLE001
                    pass
            raise
        if own:
            client.set_terminated(run_id, "FINISHED")
        return run_id

    @staticmethod
    def _log_batches(
        client: Any,
        run_id: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float | None],
        tags: Mapping[str, str],
    ) -> None:
        from mlflow.entities import Metric, Param, RunTag

        now_ms = int(time.time() * 1000)
        param_list = [
            Param(metric_key(k), str(v)[:_MAX_PARAM_VALUE])
            for k, v in params.items()
            if v is not None and v != ""
        ]
        metric_list = [
            Metric(metric_key(k), float(v), now_ms, 0)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]
        tag_list = [RunTag(k, str(v)[:5000]) for k, v in tags.items()]
        for chunk_params, chunk_metrics, chunk_tags in _chunks(param_list, metric_list, tag_list):
            client.log_batch(run_id, metrics=chunk_metrics, params=chunk_params, tags=chunk_tags)


def _chunks(
    params: list[Any], metrics: list[Any], tags: list[Any]
) -> Iterable[tuple[list[Any], list[Any], list[Any]]]:
    """Split into `log_batch` calls within MLflow's limits (100 params, 100 tags, 1000 entities)."""
    p = m = t = 0
    while p < len(params) or m < len(metrics) or t < len(tags) or (p == m == t == 0):
        cp = params[p : p + _PARAMS_PER_BATCH]
        ct = tags[t : t + _PARAMS_PER_BATCH]
        cm = metrics[m : m + _METRICS_PER_BATCH - len(cp) - len(ct)]
        if not (cp or ct or cm):
            return
        yield cp, cm, ct
        p, m, t = p + len(cp), m + len(cm), t + len(ct)
