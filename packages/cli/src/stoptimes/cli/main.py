"""`stoptimes` — one command for the whole toolkit.

Every sub-command imports its implementation lazily, so `stoptimes serve` never
loads training dependencies, and a missing optional part produces an install hint
instead of a traceback.

Commands marked "planned" are placeholders: they name the phase in docs/PLAN.md
that implements them and exit with status 2.

This module is only the assembler: each command group lives in its own module
under `stoptimes.cli.commands`, one per Phase 2 track, so a track can own its
file without touching the others.
"""

from __future__ import annotations

import typer
from stoptimes.cli.commands import data, diagnose, models, record, serve, train, val

app = typer.Typer(
    help="stoptimes — train, serve and validate transit arrival-time models on GTFS data.",
    no_args_is_help=True,
    # Plain click-style help instead of typer's rich-rendered help: rich's
    # markdown help formatter pulls in `rich.markdown` -> `markdown_it`
    # (~150-200ms) on every `--help`, which alone blows the <300ms startup
    # budget (docs/PLAN.md §7 U2). See docs/notes/p3-u2.md.
    rich_markup_mode=None,
)

app.add_typer(data.app, name="data")
app.add_typer(record.app, name="record")
app.add_typer(train.app, name="train")
app.add_typer(models.app, name="models")
app.command("serve")(serve.serve)
app.command("diagnose")(diagnose.diagnose)

val.mount(app)

if __name__ == "__main__":
    app()
