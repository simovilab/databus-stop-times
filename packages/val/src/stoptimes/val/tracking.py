"""Log a validation run to MLflow (docs/MLFLOW_INTEGRATION.md §0 D6), best-effort.

One run per validation run, in experiment ``stoptimes-val/<source>``. Metrics are per predictor,
keyed by a readable name (``mlflow:stoptimes-mbta@staging`` -> ``mlflow_stoptimes-mbta_staging``),
so one MLflow run holds the model and its baselines (``gtfsrt``, ``baseline:schedule``) side by
side; the MLflow UI then compares runs over time. Which model version an ``mlflow:`` predictor
scored is a tag (``stoptimes.model.<predictor>``), since an alias moves.

A failing or slow MLflow server costs a warning, never the validation run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from stoptimes.core.tracking import (
    TAG_KIND,
    VAL_EXPERIMENT_PREFIX,
    RunRecord,
    Tracker,
    metric_key,
    source_slug,
)
from stoptimes.val.scoring.metrics import MetricsReport, compute_metrics

log = logging.getLogger(__name__)

_BUCKET_KEYS = {"<2m": "lt2m", "2-5m": "2-5m", "5-10m": "5-10m", ">10m": "gt10m"}
_PREDICTOR_FIELDS = {
    "mae": "mae_s",
    "rmse": "rmse_s",
    "bias": "bias_s",
    "medae": "medae_s",
    "abs_error_p90": "p90_ae_s",
    "within_60s": "within_60s",
    "count": "n",
}


def predictor_key(predictor_id: str) -> str:
    return metric_key(predictor_id.replace(":", "_").replace("@", "_").replace("/", "_v"))


def predictor_models(predictors: Iterable[Any]) -> dict[str, dict[str, str]]:
    """What each model predictor scored: ``{pid: {"version": ..., "artifact_id": ...}}``."""
    found: dict[str, dict[str, str]] = {}
    for p in predictors:
        pid = getattr(p, "predictor_id", None)
        if not pid:
            continue
        info: dict[str, str] = {}
        version = getattr(p, "resolved_version", None)
        if version:
            info["version"] = str(version)
        artifact = getattr(p, "_static_artifact", None)
        if artifact is not None:
            info["artifact_id"] = artifact.artifact_id
        if info:
            found[str(pid)] = info
    return found


def report_metrics(report: MetricsReport) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for m in report.per_predictor:
        key = predictor_key(m.predictor_id)
        for attr, name in _PREDICTOR_FIELDS.items():
            metrics[metric_key(key, name)] = float(getattr(m, attr))
    for b in report.horizon_buckets:
        bucket = _BUCKET_KEYS.get(b.label, b.label)
        key = predictor_key(b.predictor_id)
        metrics[metric_key(key, bucket, "mae_s")] = float(b.mae)
        metrics[metric_key(key, bucket, "bias_s")] = float(b.bias)
        metrics[metric_key(key, bucket, "n")] = float(b.count)
    return metrics


def validation_record(
    report: MetricsReport,
    *,
    source: str,
    mode: str,
    run_id: str,
    predictor_ids: list[str],
    routes: list[str] | None = None,
    duration_s: float | None = None,
    seed: int | None = None,
    status: str = "done",
    error: str | None = None,
    models: Mapping[str, Mapping[str, str]] | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> RunRecord:
    tags = {
        TAG_KIND: "val",
        "stoptimes.val.run_id": run_id,
        "stoptimes.val.mode": mode,
        "stoptimes.val.status": status,
        "stoptimes.source_id": source,
    }
    if error:
        tags["stoptimes.val.error"] = error[:4000]
    for pid, info in (models or {}).items():
        if "version" in info:
            tags[f"stoptimes.model.{predictor_key(pid)}"] = info["version"]
        if "artifact_id" in info:
            tags[f"stoptimes.artifact.{predictor_key(pid)}"] = info["artifact_id"]
    params: dict[str, Any] = {
        "source": source,
        "mode": mode,
        "predictors": ",".join(predictor_ids),
        "routes": ",".join(routes or []) or "all",
        "duration_s": duration_s,
        "seed": seed,
        **(extra_params or {}),
    }
    return RunRecord(
        experiment=VAL_EXPERIMENT_PREFIX + source_slug(source),
        run_name=f"{mode} {source} {run_id[:8]}",
        params=params,
        metrics=report_metrics(report),
        tags=tags,
        texts={"report.json": json.dumps(report.summary(), default=str, indent=1)},
    )


def log_validation(
    tracker: Tracker | None,
    results: list[Any],
    *,
    tz: str = "UTC",
    **record_kwargs: Any,
) -> str | None:
    """Score *results* and log the run.

    Returns the MLflow run id, or None (tracking off, no results, or the write failed).
    """
    if tracker is None:
        return None
    if not results:
        log.info("no scored results; nothing to log to MLflow")
        return None
    try:
        record = validation_record(compute_metrics(results, tz=tz), **record_kwargs)
    except Exception as exc:  # noqa: BLE001 - best-effort (D6)
        log.warning("could not build the MLflow record for this validation run: %s", exc)
        return None
    return tracker.log(record)
