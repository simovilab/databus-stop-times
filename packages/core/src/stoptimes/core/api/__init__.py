"""Python API: the same verbs as the `stoptimes` CLI, as importable functions.

Phase 3, track U4 (docs/PLAN.md). A researcher in a notebook does the whole
end-to-end chain -- read observations, derive arrivals, build a training
frame, run an experiment, list/select models, predict, replay-score -- with
these functions instead of shelling out to the CLI. Every function returns
data (a `pandas.DataFrame` or one of the `stoptimes.core.contracts` pydantic
models), never prints.

Layout (docs/PLAN.md §1 rule 1: `core` must not import `train`/`serve`/`val`):

- This package (`stoptimes.core.api`) covers verbs that only ever touch
  `stoptimes.core`: the local data catalog (`add_dataset`, `list_datasets`,
  `get_dataset`, `remove_dataset`, `load_observations`,
  `load_arrivals_frame`), arrivals derivation (`build_arrivals_dataset`,
  `derive_arrivals`), feed diagnostics (`diagnose`), dataset coverage
  (`inspect_dataset`), agency config loading (`resolve_source`), and the
  read side of the model registry (`list_models`, `model_card`,
  `promote_model` -- these only read `artifact.json`/profile YAML, no
  `stoptimes.models` import needed).
- Verbs that need `stoptimes.train` (`build_frame`, `plan_experiment`,
  `run_experiment`, `select_model`, `predict_once`) live in
  `stoptimes.train.api` instead -- import from there directly, or lazily via
  `stoptimes.train` if it's installed. Trying to import them from here
  would violate the dependency direction (`core` importing `train`).
- Verbs that need `stoptimes.val` (`replay_score`, `score_report`) live in
  `stoptimes.val.api_verbs`.

A note on `import stoptimes` (see docs/notes/p3-u4.md for the full writeup):
`stoptimes` itself is an *implicit namespace package* (PEP 420, no
`stoptimes/__init__.py` anywhere in the workspace -- docs/PLAN.md §2) with
no code of its own, so `import stoptimes; stoptimes.add_dataset(...)` does
not work and can't be made to work without adding a real `stoptimes/__init__.py`
somewhere, which would need an orchestrator decision (it affects every
package, not just this one). The verbs are "importable as `stoptimes`" in
the sense this module's import path starts with `stoptimes`:

    from stoptimes.core.api import add_dataset, diagnose, resolve_source
    from stoptimes.train.api import build_frame, run_experiment_from_file
    from stoptimes.val.api_verbs import replay_score

Example (the full core-only half of the end-to-end chain; see
`examples/notebooks/01_end_to_end.ipynb` for the complete chain including
`stoptimes.train.api`/`stoptimes.val.api_verbs`):

    >>> from stoptimes.core.api import add_dataset, build_arrivals_dataset  # doctest: +SKIP
    >>> add_dataset("positions", "archive/", source="demo")  # doctest: +SKIP
    >>> arrivals, summary = build_arrivals_dataset(
    ...     observations="positions", static="gtfs.zip", source="demo",
    ...     out="arrivals.parquet", register="arrivals",
    ... )  # doctest: +SKIP
"""

from __future__ import annotations

from stoptimes.core.api._arrivals import ArrivalParams, build_arrivals_dataset, derive_arrivals
from stoptimes.core.api._catalog import (
    DatasetEntry,
    DatasetExistsError,
    DatasetKind,
    add_dataset,
    get_dataset,
    guess_format,
    list_datasets,
    load_arrivals_frame,
    load_observations,
    remove_dataset,
)
from stoptimes.core.api._diagnose import DiagnoseParams, DiagnosticReport, diagnose
from stoptimes.core.api._inspect import inspect_dataset
from stoptimes.core.api._models import list_models, model_card, promote_model
from stoptimes.core.api._sources import resolve_source

__all__ = [
    # catalog
    "add_dataset",
    "list_datasets",
    "get_dataset",
    "remove_dataset",
    "load_observations",
    "load_arrivals_frame",
    "guess_format",
    "DatasetEntry",
    "DatasetExistsError",
    "DatasetKind",
    # arrivals
    "build_arrivals_dataset",
    "derive_arrivals",
    "ArrivalParams",
    # diagnose
    "diagnose",
    "DiagnoseParams",
    "DiagnosticReport",
    # inspect
    "inspect_dataset",
    # models (read side)
    "list_models",
    "model_card",
    "promote_model",
    # sources
    "resolve_source",
]
