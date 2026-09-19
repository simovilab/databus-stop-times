"""Python API: the `stoptimes train`/`serve` verbs that need training and
serving dependencies -- pandas/sklearn/the pipeline -- so they can't live in
`stoptimes.core.api` without violating docs/PLAN.md §1 rule 1 (`core` must
not import `train`/`serve`).

Phase 3, track U4. See `stoptimes.core.api`'s module docstring for the full
picture and docs/notes/p3-u4.md for the CLI-command-to-API-function map the
orchestrator should apply.

Verbs here:

- `build_frame` -- `stoptimes data frame`: observations + arrivals + static
  -> the training frame.
- `plan_experiment`, `load_experiment` -- re-exported unchanged from
  `stoptimes.core.contracts.experiment` / `stoptimes.train.experiment`;
  listed here too so a notebook needs only one import line.
- `run_experiment`, `run_experiment_from_file` -- `stoptimes train run`.
- `select_model` -- the selection half of `stoptimes serve`'s `--model`
  resolution (`stoptimes.serve.model_loading.resolve_artifact`), exposed
  here (not from `stoptimes.serve`, which this track does not own) so
  "list/select models" doesn't need a direct `stoptimes.serve` import in
  notebook code.
- `predict_once` -- the prediction half of `stoptimes serve`: run one
  pipeline pass and return its trip updates instead of publishing them.
  This mirrors `stoptimes.cli.commands.serve.serve`'s `--once` path as of
  Phase 3 T4; once U6 rebuilds `serve.estimator` onto the pipeline this
  should be revisited (see docs/notes/p3-u4.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from stoptimes.core.contracts.experiment import Experiment, load_experiment
from stoptimes.core.contracts.source import Source
from stoptimes.train.experiment import ArtifactExistsError, plan_experiment, run_experiment
from stoptimes.train.frame import FrameParams, build_training_frame

__all__ = [
    "FrameParams",
    "build_frame",
    "Experiment",
    "load_experiment",
    "plan_experiment",
    "run_experiment",
    "run_experiment_from_file",
    "ArtifactExistsError",
    "select_model",
    "predict_once",
]


def build_frame(
    *,
    observations: str,
    arrivals: str,
    static: str,
    source: str,
    out: str | Path,
    max_stops_ahead: int = 5,
    keep_unlabeled: bool = False,
    sources_dir: str | Path | None = None,
    register: str | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the training frame: observations + arrivals + static GTFS ->
    one row per (vehicle position, upcoming stop), ready for `run_experiment`.

    Equivalent to `stoptimes data frame`.

    Args:
        observations: a `stoptimes.core.api.add_dataset` name, or a
            path/URI `stoptimes.core.readers.read_observations` understands.
        arrivals: a catalog dataset name, or a path/URI to an Arrival parquet
            (e.g. from `stoptimes.core.api.build_arrivals_dataset`).
        static: a GTFS static `.zip` path or URL.
        source: the agency id (`examples/agencies/<source>.yaml`).
        out: where to write the training-frame parquet.
        max_stops_ahead: upcoming stops emitted per vehicle-position row.
        keep_unlabeled: keep rows with no matching arrival (target left NaN)
            instead of dropping them.
        sources_dir: agency config directory; defaults to
            `$STOPTIMES_SOURCES` / `./examples/agencies`.
        register: also register the output under this catalog name.
        overwrite: replace an existing `out` file / catalog entry.

    Returns:
        `(frame, summary)` where `summary` matches `stoptimes data frame
        --json`'s ``{"source", "observations", "arrivals", "rows",
        "labeled_rows", "static_tables", "out", "registered"}``.

    Example:
        >>> from stoptimes.train.api import build_frame  # doctest: +SKIP
        >>> frame, summary = build_frame(
        ...     observations="positions", arrivals="arrivals", static="gtfs.zip",
        ...     source="demo", out="frame.parquet", register="frame",
        ... )  # doctest: +SKIP
        >>> summary["rows"] > 0  # doctest: +SKIP
        True

    Raises:
        FileExistsError: `out` already exists and `overwrite` is False.
    """
    from stoptimes.core import readers as readers_module
    from stoptimes.core.api import load_arrivals_frame, load_observations, resolve_source
    from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetKind

    src: Source = resolve_source(source, sources_dir=sources_dir)
    obs_frame = load_observations(observations, source_id=source)
    arrivals_frame = load_arrivals_frame(arrivals)
    loader, static_feed = readers_module.read_static(static, source)
    if loader is None:
        raise ValueError(f"{static!r} is missing required static table(s): {static_feed.missing_required()}")

    out_path = Path(out)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists (pass overwrite=True to replace it)")

    params = FrameParams(max_stops_ahead=max_stops_ahead, keep_unlabeled=keep_unlabeled)
    result = build_training_frame(obs_frame, arrivals_frame, loader, source=src, params=params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    date_start = date_end = None
    if not result.empty:
        date_start = str(result["vp_ts"].min())
        date_end = str(result["vp_ts"].max())

    if register:
        entry = DatasetEntry(
            name=register,
            source_id=source,
            kind=DatasetKind.FRAME,
            location=str(out_path),
            date_start=date_start,
            date_end=date_end,
            row_count=int(len(result)),
            added_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            extra={
                "derived_from_observations": observations,
                "derived_from_arrivals": arrivals,
                "static": static,
                "max_stops_ahead": max_stops_ahead,
            },
        )
        Catalog().add(entry, overwrite=overwrite)

    n_labeled = int(result["time_to_arrival_seconds"].notna().sum()) if not result.empty else 0
    summary = {
        "source": source,
        "observations": int(len(obs_frame)),
        "arrivals": int(len(arrivals_frame)),
        "rows": int(len(result)),
        "labeled_rows": n_labeled,
        "static_tables": static_feed.tables,
        "out": str(out_path),
        "registered": register,
    }
    return result, summary


def run_experiment_from_file(
    experiment_path: str | Path,
    *,
    registry_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    overwrite: bool = False,
):
    """Load `experiment_path` and run it end to end.

    Equivalent to `stoptimes train run experiment.yaml`. See
    `stoptimes.train.experiment.run_experiment` for the full behavior; this
    just adds the `load_experiment` step so a notebook needs one call, not two.

    Example:
        >>> from stoptimes.train.api import run_experiment_from_file  # doctest: +SKIP
        >>> meta = run_experiment_from_file("experiment.yaml")  # doctest: +SKIP
        >>> meta.family  # doctest: +SKIP
        'historical_mean'

    Raises:
        ArtifactExistsError: an artifact with this experiment's id already
            exists in the registry and `overwrite` is False.
    """
    experiment = load_experiment(experiment_path)
    return run_experiment(experiment, registry_dir=registry_dir, dataset_dir=dataset_dir, overwrite=overwrite)


def select_model(
    *,
    registry: str | Path,
    source_id: str,
    policy: Any = None,
    route_id: str | None = None,
    direction_id: int | None = None,
    trip_id: str | None = None,
    artifact_id: str | None = None,
):
    """Pick the artifact that should serve a trip -- the selection half of
    `stoptimes serve`'s `--model` resolution.

    Thin wrapper over `stoptimes.serve.model_loading.resolve_artifact`
    (imported lazily so `import stoptimes.train.api` never pulls in
    `stoptimes.serve`'s dependency stack unless this is actually called).

    Args:
        registry: an artifact registry directory
            (`stoptimes.core.contracts.artifact` layout).
        source_id: the agency id to select for.
        policy: a `stoptimes.core.contracts.selection.SelectionPolicy`
            (e.g. `source.serving.selection`); `None` uses the registry's
            best match with no pins.
        route_id, direction_id, trip_id: narrow the selection to a specific
            trip; all optional.
        artifact_id: a hard override -- if given, it wins outright (looked
            up directly) instead of going through selection.

    Returns:
        The winning `ArtifactMetadata`, or `None` if nothing in `registry`
        matches.

    Example:
        >>> from stoptimes.train.api import select_model  # doctest: +SKIP
        >>> meta = select_model(registry="artifacts/models", source_id="demo")  # doctest: +SKIP
    """
    from stoptimes.serve.model_loading import resolve_artifact

    return resolve_artifact(
        registry,
        source_id=source_id,
        policy=policy,
        route_id=route_id,
        direction_id=direction_id,
        trip_id=trip_id,
        artifact_id=artifact_id,
    )


def predict_once(
    *,
    source: Source,
    registry: str | Path | None = None,
    artifact_id: str | None = None,
    model: Any = None,
    pipeline: str | None = None,
    sink: str | None = None,
    format: str = "json",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one serving pipeline pass and return its summary -- the
    prediction half of `stoptimes serve --once`, without the CLI's argv
    parsing or agency-file loading (pass an already-resolved `source`, e.g.
    from `stoptimes.core.api.resolve_source`).

    Args:
        source: the resolved `Source` to serve.
        registry: artifact registry directory; used to resolve `artifact_id`
            into a model unless `model` is given directly.
        artifact_id: an artifact id in `registry` to serve with.
        model: an already-loaded `stoptimes.core.contracts.model.Model`,
            overriding `registry`/`artifact_id` entirely.
        pipeline: a custom `Pipeline` spec (`"module:attr"` or
            `"path/to/file.py:attr"`); the default pipeline is used if omitted.
        sink: `file:PATH` / `http(s)://...` / `mqtt://host/topic` to publish
            to; predictions are computed but not published if omitted.
        format: feed encoding, `"pb"` or `"json"`.
        now: the pipeline's logical clock; defaults to the current UTC time.

    Returns:
        ``{"source", "artifact_id", "generated_at", "trip_update_count",
        "errors", "published"}`` -- the same shape `stoptimes serve --once
        --json` prints, as data.

    Example:
        >>> from stoptimes.train.api import predict_once  # doctest: +SKIP
        >>> from stoptimes.core.api import resolve_source  # doctest: +SKIP
        >>> src = resolve_source("demo", sources_dir="examples/agencies")  # doctest: +SKIP
        >>> summary = predict_once(
        ...     source=src, registry="artifacts/models", artifact_id="e2e-1",
        ... )  # doctest: +SKIP
        >>> summary["errors"]  # doctest: +SKIP
        []
    """
    from stoptimes.core.contracts.pipeline import PipelineContext
    from stoptimes.serve.model_loading import load_artifact_model, resolve_artifact
    from stoptimes.serve.pipeline import build_default_pipeline
    from stoptimes.serve.pipeline import load_pipeline as load_user_pipeline
    from stoptimes.serve.sinks import sink_from_uri

    resolved_model = model
    resolved_artifact_id = artifact_id
    if resolved_model is None and registry is not None:
        meta = resolve_artifact(
            registry, source_id=source.id, policy=source.serving.selection, artifact_id=artifact_id
        )
        if meta is not None:
            resolved_artifact_id = meta.artifact_id
            resolved_model = load_artifact_model(meta, registry)

    run_pipeline = load_user_pipeline(pipeline) if pipeline is not None else build_default_pipeline()
    active_sink = sink_from_uri(sink) if sink else None

    ctx = PipelineContext(
        source=source,
        static_feed=None,
        model=resolved_model,
        now=now or datetime.now(timezone.utc),
        config={"sink": active_sink, "format": format},
    )
    run_pipeline.run(ctx)
    trip_updates = ctx.state.get("trip_updates", [])
    errors = ctx.state.get("errors", [])
    return {
        "source": source.id,
        "artifact_id": resolved_artifact_id,
        "generated_at": ctx.now.isoformat(),
        "trip_update_count": len(trip_updates),
        "errors": errors,
        "published": ctx.state.get("published", False),
    }
