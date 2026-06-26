"""Typer CLI. Phase P6.

Commands: sources (list configured), run (drive runtime headless, persist to SQLite,
export Parquet), report (per-predictor MAE + horizon breakdown + model comparison via rich).

P0 scaffold ships only `version`; the rest land in P6.
"""

import typer

import etaval

app = typer.Typer(help="etaval — ETA Prediction Validation Tool", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """etaval — ETA Prediction Validation Tool."""


@app.command()
def version() -> None:
    """Print the etaval version."""
    typer.echo(etaval.__version__)


if __name__ == "__main__":
    app()
