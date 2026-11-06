"""Log a finished training run to MLflow (docs/MLFLOW_INTEGRATION.md §0 D6), best-effort.

When the artifact was published to an MLflow registry on the same server, the publish already made
a run in ``stoptimes/<source_id>`` and the model version points at it; the training details
(baselines as child runs, the experiment) go into that run, so version and run stay one record.
Otherwise a new run is created in that experiment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from stoptimes.core.contracts.artifact import ArtifactMetadata
from stoptimes.core.tracking import Tracker, training_record

log = logging.getLogger(__name__)


def log_training(
    tracker: Tracker | None,
    meta: ArtifactMetadata,
    registry: Any,
    *,
    card: str | None,
    baselines: Mapping[str, Mapping[str, float | None]] | None = None,
    experiment: Any = None,
) -> str | None:
    """Log *meta*'s training run; returns the MLflow run id, or None (tracking off or failed)."""
    if tracker is None:
        return None
    record = training_record(
        meta,
        card=card,
        baselines=baselines,
        experiment_json=experiment.model_dump_json(indent=2) if experiment is not None else None,
    )
    return tracker.log(record, run_id=_publish_run(registry, tracker, meta))


def _publish_run(registry: Any, tracker: Tracker, meta: ArtifactMetadata) -> str | None:
    """The run the MLflow registry's publish created for *meta*, if it's on the tracker's server."""
    if getattr(registry, "tracking_uri", None) != tracker.uri or not hasattr(registry, "resolve"):
        return None
    try:
        return registry.resolve(meta.artifact_id).run_id
    except Exception as exc:  # noqa: BLE001 - fall back to a separate run
        log.warning("no publish run found for %s (%s); logging a separate run", meta.artifact_id, exc)
        return None
