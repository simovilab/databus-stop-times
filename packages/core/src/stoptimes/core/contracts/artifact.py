"""Contract: model artifacts and model cards.

PLACEHOLDER — Phase 1, track C1 (docs/PLAN.md). Must cover:
- On-disk layout, metadata (data window, features, params, code version,
  library versions), scores by horizon, and a loader that never unpickles
  across incompatible versions silently.
- Loading of pre-migration pickles (module paths models.*, core.*, eta_service.*).
Existing starting points: stoptimes.models.common.registry, .provenance.
"""
