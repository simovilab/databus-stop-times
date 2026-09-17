"""The contracts every part of stoptimes agrees on.

Frozen in Phase 1 (docs/PLAN.md §5). Phase 2 tracks import from here, not
from the individual submodules, so the public surface is this file's
`__all__`. See docs/CONTRACTS.md for the human-readable version of each
contract: purpose, fields, invariants, producers/consumers.

    source      how a feed / agency / archive is described
    data        tables for static GTFS, realtime observations and derived arrivals
    experiment  the experiment.yaml schema (target, model, features, params, split)
    model       the interface every model family implements
    pipeline    the serving stages users can override
    prediction  what serving emits (and how it maps to GTFS-RT TripUpdates)
    artifact    a trained model on disk + its model card
    adapters    pure functions from existing structures to the shapes above
"""

from __future__ import annotations

from stoptimes.core.contracts.adapters import (
    arrival_from_actual_arrival,
    observation_from_tu_row,
    observation_from_vp_row,
    source_from_archive_uri,
    source_from_val_yaml,
    stop_time_prediction_from_sample,
)
from stoptimes.core.contracts.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactMetadata,
    DataWindow,
    HorizonScore,
    ModelFormat,
    Provenance,
    artifact_dir,
    read_metadata,
    version_mismatches,
    write_metadata,
)
from stoptimes.core.contracts.data import (
    ARRIVAL_PARTITION,
    ARRIVAL_SCHEMA,
    OBSERVATION_TU_PARTITION,
    OBSERVATION_TU_SCHEMA,
    OBSERVATION_VP_PARTITION,
    OBSERVATION_VP_SCHEMA,
    REQUIRED_STATIC_TABLES,
    STATIC_TABLE_SCHEMAS,
    UTC_DTYPE,
    Arrival,
    ArrivalMethod,
    ColumnSpec,
    Observation,
    ObservationKind,
    PartitionLayout,
    StaticFeed,
    TableSchema,
    coerce_utc_ns,
    validate_frame,
)
from stoptimes.core.contracts.experiment import (
    DEFAULT_BASELINES,
    DataSelection,
    Experiment,
    ModelSpec,
    OutputSpec,
    ParamChoices,
    ParamRange,
    SplitStrategy,
    dump_experiment,
    load_experiment,
)
from stoptimes.core.contracts.model import Model, ModelLoader, ModelRef, ModelRefKind
from stoptimes.core.contracts.pipeline import (
    STAGE_ORDER,
    Pipeline,
    PipelineContext,
    SequentialPipeline,
    StageHook,
)
from stoptimes.core.contracts.prediction import (
    StopScheduleRelationship,
    StopTimePrediction,
    TripScheduleRelationship,
    TripUpdatePrediction,
)
from stoptimes.core.contracts.source import (
    Archive,
    ArchiveKind,
    FeedFormat,
    FeedKind,
    GroundTruthDetector,
    RealtimeFeed,
    Source,
    dump_source,
    load_source,
)

CONTRACTS_VERSION = "1.0.0-draft"
"""Bumped whenever a contract's shape changes. "-draft" until the user signs
off docs/CONTRACTS.md (docs/PLAN.md §5 gate); after sign-off, a contract
change needs an orchestrator-approved note (docs/PLAN.md §10)."""

__all__ = [
    "CONTRACTS_VERSION",
    # source
    "Source",
    "RealtimeFeed",
    "FeedKind",
    "FeedFormat",
    "Archive",
    "ArchiveKind",
    "GroundTruthDetector",
    "load_source",
    "dump_source",
    # data
    "StaticFeed",
    "Observation",
    "ObservationKind",
    "Arrival",
    "ArrivalMethod",
    "ColumnSpec",
    "TableSchema",
    "PartitionLayout",
    "UTC_DTYPE",
    "coerce_utc_ns",
    "validate_frame",
    "REQUIRED_STATIC_TABLES",
    "STATIC_TABLE_SCHEMAS",
    "OBSERVATION_VP_SCHEMA",
    "OBSERVATION_TU_SCHEMA",
    "OBSERVATION_VP_PARTITION",
    "OBSERVATION_TU_PARTITION",
    "ARRIVAL_SCHEMA",
    "ARRIVAL_PARTITION",
    # experiment
    "Experiment",
    "DataSelection",
    "ModelSpec",
    "ParamRange",
    "ParamChoices",
    "SplitStrategy",
    "OutputSpec",
    "DEFAULT_BASELINES",
    "load_experiment",
    "dump_experiment",
    # model
    "Model",
    "ModelLoader",
    "ModelRef",
    "ModelRefKind",
    # pipeline
    "Pipeline",
    "PipelineContext",
    "SequentialPipeline",
    "StageHook",
    "STAGE_ORDER",
    # prediction
    "StopTimePrediction",
    "TripUpdatePrediction",
    "StopScheduleRelationship",
    "TripScheduleRelationship",
    # artifact
    "ArtifactMetadata",
    "ModelFormat",
    "DataWindow",
    "Provenance",
    "HorizonScore",
    "ARTIFACT_SCHEMA_VERSION",
    "artifact_dir",
    "read_metadata",
    "write_metadata",
    "version_mismatches",
    # adapters
    "source_from_val_yaml",
    "source_from_archive_uri",
    "observation_from_vp_row",
    "observation_from_tu_row",
    "arrival_from_actual_arrival",
    "stop_time_prediction_from_sample",
]
