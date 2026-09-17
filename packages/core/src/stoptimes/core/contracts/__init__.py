"""The contracts every part of stoptimes agrees on.

PLACEHOLDER — designed and frozen in Phase 1 (docs/PLAN.md). Until then these
modules only name each contract and what it must cover, so Phase 2 tracks have
a single place to import from once they exist.

    data        tables for static GTFS, realtime observations and derived arrivals
    experiment  the experiment.yaml schema (target, model, features, params, split)
    artifact    a trained model on disk + its model card
    model       the interface every model family implements
    pipeline    the serving stages users can override
    prediction  what serving emits (and how it maps to GTFS-RT TripUpdates)
    source      how a feed / agency / archive is described
"""
