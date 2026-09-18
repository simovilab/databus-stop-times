"""`stoptimes val` — mount the etaval CLI, plus its API/TUI console scripts.

etaval shipped two console scripts (`etaval-api`, `etaval-tui`) alongside its
Typer CLI. Here they become `stoptimes val api` / `stoptimes val tui`. The
rest of the real val commands (`version`, `sources`, `run`, `replay`,
`report`, `walk-forward`, `figures`) are mounted as thin forwarding stubs so
`stoptimes val --help` stays fast; see "Startup budget" below.

Falls back to an install hint (rather than a traceback) only when a
`stoptimes*` module is missing — any other import error propagates.

Startup budget
---------------
`stoptimes.val.cli.main` transitively imports `stoptimes.val.engine.assembly`
(-> `stoptimes.models.common.metrics` -> `sklearn`) and `stoptimes.val.figures`
(-> `matplotlib.pyplot`) at module scope: about 3s combined, measured with
`python -X importtime -c "import stoptimes.val.cli.main"` (see
docs/notes/p3-u2.md). Those eager imports live in `packages/val` and
`packages/models`, outside this track's owned paths, so making
`stoptimes.val.cli.main` itself lazy isn't an option here.

Instead, each real val command is mounted as a `context_settings=PASSTHROUGH`
stub (the same pattern already used for `record compact` / `models check` /
`train legacy-all`) whose `help=` text is a static one-liner, so `--help`
never needs to import the module to render the command list. The stub's body
imports `stoptimes.val.cli.main` and forwards argv to the real click command
only when the subcommand is actually invoked (including that subcommand's
own `--help`, which is not covered by the group-level speed budget).
"""

from __future__ import annotations

import importlib
import sys

import typer
from stoptimes.cli._util import PASSTHROUGH, require

VAL_MISSING_HELP = "Validate predictions (not installed)."
VAL_HELP = "Validate arrival predictions against ground truth (etaval)."

# name -> short help, kept in sync by hand with the docstrings in
# stoptimes.val.cli.main (packages/val). There is no way to derive this table
# without the import it exists to avoid.
_FORWARDED_COMMANDS: dict[str, str] = {
    "version": "Print the stoptimes val version.",
    "sources": "List all configured GTFS-RT sources.",
    "run": "Run headless ETA validation against a live GTFS-RT source, or replay an archive.",
    "replay": "Score predictions from a recorded archive instead of a live feed.",
    "report": "Render a metrics report from a Parquet results file.",
    "walk-forward": "Walk-forward backtest: per-window MAE + CI over an archived replay window.",
    "figures": "Render publication figures (300 dpi PNG) from a Parquet results file.",
}


def _run_real_val_command(name: str, args: list[str]) -> None:
    """Import the real `stoptimes.val` CLI and hand off to its `name` command."""
    module = importlib.import_module("stoptimes.val.cli.main")
    real_group = typer.main.get_command(module.app)
    real_command = real_group.commands[name]
    real_command.main(args=list(args), prog_name=f"stoptimes val {name}", standalone_mode=True)


def _make_forwarder(name: str):
    def _forward(ctx: typer.Context) -> None:
        _run_real_val_command(name, ctx.args)

    _forward.__name__ = f"val_{name.replace('-', '_')}"
    return _forward


def _add_api_tui_commands(val_app: typer.Typer) -> None:
    @val_app.command("api")
    def val_api() -> None:
        """Run the REST + WebSocket API (was the `etaval-api` console script)."""
        from stoptimes.val.api.app import main as api_main

        # api_main() reads settings from the environment; it takes no argv.
        api_main()

    @val_app.command("tui", context_settings=PASSTHROUGH)
    def val_tui(ctx: typer.Context) -> None:
        """Launch the terminal ops console (was the `etaval-tui` console script)."""
        from stoptimes.val.tui.app import main as tui_main

        # tui_main() parses sys.argv itself (argparse), so pass extra args through.
        sys.argv = ["stoptimes val tui", *ctx.args]
        tui_main()


def mount(app: typer.Typer) -> None:
    """Mount `val` onto the assembled `app`, or an install-hint stub if absent.

    Checking `find_spec` (rather than importing) keeps this cheap even when
    `stoptimes.val` isn't installed at all.
    """
    import importlib.util

    if importlib.util.find_spec("stoptimes.val") is None:

        @app.command("val", help=VAL_MISSING_HELP)
        def val_missing() -> None:
            require("stoptimes.val", "val")

        return

    val_app = typer.Typer(help=VAL_HELP, no_args_is_help=True)
    for name, help_text in _FORWARDED_COMMANDS.items():
        val_app.command(name, context_settings=PASSTHROUGH, help=help_text)(_make_forwarder(name))
    _add_api_tui_commands(val_app)

    app.add_typer(val_app, name="val", help=VAL_HELP)
