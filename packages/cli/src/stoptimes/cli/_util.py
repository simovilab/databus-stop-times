"""Helpers shared by the stoptimes CLI."""

from __future__ import annotations

import importlib.util
import runpy
import sys

import typer

PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []}
PLANNED_EXIT = 2


def run_module(module: str, prog: str, args: list[str]) -> None:
    """Run a legacy module as ``__main__`` with the given argv (passthrough commands)."""
    sys.argv = [prog, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def require(module: str, extra: str) -> None:
    """Exit with an install hint when an optional part of the toolkit is missing.

    The hint names commands that actually work today: `stoptimes` isn't on PyPI
    yet (docs/PLAN.md §9), so `pip install 'stoptimes[...]'` resolves to
    nothing, and `uv sync --extra ...` only works from `packages/cli`, not from
    the repo root a reader is most likely standing in.
    """
    if importlib.util.find_spec(module) is None:
        typer.secho(f"✗ This command needs the '{extra}' part of stoptimes.", fg="red", err=True)
        typer.echo(
            f"  Install it with:  uv pip install './packages/cli[{extra}]'"
            "   (from a clone of the repo)",
            err=True,
        )
        typer.echo("  Contributing to stoptimes itself?  uv sync --group dev", err=True)
        raise typer.Exit(1)


def planned(command: str, phase: str) -> None:
    """Placeholder for a command that is designed but not built yet."""
    typer.secho(f"• `stoptimes {command}` is planned — {phase}. See docs/PLAN.md.", fg="yellow",
                err=True)
    raise typer.Exit(PLANNED_EXIT)


# ── Shared presentation (docs/PLAN.md §7 U2: "consistent tables ... errors to
#    stderr, data to stdout") ─────────────────────────────────────────────────
#
# Every command that prints something people read (as opposed to `--json`
# output) should go through these instead of ad hoc `typer.echo`/`typer.secho`
# calls, so error/success/warning styling and table rendering look the same
# everywhere. `--json` output never goes through these -- it's already a
# single `typer.echo(json.dumps(...))` call per command.


def error(message: str) -> None:
    """A user-facing error: red, prefixed with '✗', to stderr. Callers still
    raise `typer.Exit(1)` (or another non-zero code) themselves afterwards --
    this only prints."""
    typer.secho(f"✗ {message}", fg="red", err=True)


def warn(message: str) -> None:
    """A non-fatal warning: yellow, prefixed with '⚠', to stderr."""
    typer.secho(f"⚠ {message}", fg="yellow", err=True)


def success(message: str) -> None:
    """A success message: green, prefixed with '✓', to stdout."""
    typer.secho(f"✓ {message}", fg="green")


def print_table(
    title: str, columns: list[str], rows: list[list[str]], *, empty_message: str | None = None
) -> None:
    """Render `rows` as a table on stdout, via `rich`, the one shape every
    non-JSON command output uses.

    `rows` are pre-stringified: callers own their own formatting (e.g. "—"
    for missing values), this only lays them out.
    """
    if not rows and empty_message is not None:
        typer.echo(empty_message)
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    Console().print(table)
