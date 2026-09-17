"""`stoptimes val` — mount the etaval CLI, plus its API/TUI console scripts.

etaval shipped two console scripts (`etaval-api`, `etaval-tui`) alongside its
Typer CLI. Here they become `stoptimes val api` / `stoptimes val tui`, added
directly onto the imported `stoptimes.val.cli.main:app` instance so they show
up next to `version`/`sources`/`run`/`report` under one `stoptimes val --help`.

Falls back to an install hint (rather than a traceback) only when a
`stoptimes*` module is missing — any other import error propagates.
"""

from __future__ import annotations

import importlib
import sys

import typer
from stoptimes.cli._util import PASSTHROUGH, require

VAL_MISSING_HELP = "Validate predictions (not installed)."
VAL_HELP = "Validate arrival predictions against ground truth (etaval)."


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
    """Mount `val` onto the assembled `app`, or an install-hint stub if absent."""
    try:
        module = importlib.import_module("stoptimes.val.cli.main")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing.startswith("stoptimes"):
            raise

        @app.command("val", help=VAL_MISSING_HELP)
        def val_missing() -> None:
            require("stoptimes.val", "val")

        return

    _add_api_tui_commands(module.app)
    app.add_typer(module.app, name="val", help=VAL_HELP)
