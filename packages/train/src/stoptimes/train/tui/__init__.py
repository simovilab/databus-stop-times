"""Textual experiment builder: pick target, model, features, params; writes experiment.yaml.

Phase 3, track U1 (docs/PLAN.md). `textual` is an optional dependency of
`stoptimes-train` (see docs/notes/p3-u1.md) -- importing this package (or
`stoptimes.train` generally) never requires it. Only `stoptimes.train.tui.app`
imports Textual; `registries` (choice sources) and `builder` (draft assembly,
validation, YAML round trip) are plain Python and safe to import/test
without the 'tui' extra installed.

`run_app(path)` is a lazy re-export of `stoptimes.train.tui.app.run_app` --
importing *this* module doesn't import Textual, calling `run_app` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_app(path: "str | Path | None" = None) -> None:
    """Launch the experiment builder TUI. Raises `ImportError` with an
    install hint if Textual isn't installed -- callers that already did a
    `find_spec("textual")` check (e.g. the CLI's `train ui` command) won't
    hit that path."""
    from stoptimes.train.tui.app import run_app as _run_app

    _run_app(path)
