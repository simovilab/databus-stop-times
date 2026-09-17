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
    """Exit with an install hint when an optional part of the toolkit is missing."""
    if importlib.util.find_spec(module) is None:
        typer.secho(f"✗ This command needs the '{extra}' part of stoptimes.", fg="red", err=True)
        typer.echo(
            f"  Install it with:  uv sync --extra {extra}   (or pip install 'stoptimes[{extra}]')",
            err=True,
        )
        raise typer.Exit(1)


def planned(command: str, phase: str) -> None:
    """Placeholder for a command that is designed but not built yet."""
    typer.secho(f"• `stoptimes {command}` is planned — {phase}. See docs/PLAN.md.", fg="yellow",
                err=True)
    raise typer.Exit(PLANNED_EXIT)
