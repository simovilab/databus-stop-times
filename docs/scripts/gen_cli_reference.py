"""Generate docs/reference/cli.md from the live Typer app.

This walks `stoptimes.cli.main.app`'s command tree with Click's own help
formatter, so the reference always matches what `--help` actually prints —
it cannot drift the way a hand-copied reference would.

Regenerate after any CLI change:

    uv run --no-sync python docs/scripts/gen_cli_reference.py

Requires nothing beyond the project's own dev environment (no mkdocs
dependency) so any track can run it.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import click
import typer

from stoptimes.cli.main import app

OUT = Path(__file__).resolve().parent.parent / "reference" / "cli.md"

HEADER = """# CLI reference

Generated from the `stoptimes` Typer app's own `--help` output by
`docs/scripts/gen_cli_reference.py` — run that script after any CLI change;
do not hand-edit this file.

Every block below is the literal output of `stoptimes <path> --help` in this
checkout. Options shown as `[required]` have no default and must be passed.
"""


def _help_text(cmd: click.Command, path: list[str]) -> str:
    ctx = click.Context(cmd, info_name=" ".join(["stoptimes", *path]))
    return cmd.get_help(ctx)


def _walk(cmd: click.Command, path: list[str], lines: list[str]) -> None:
    heading = "stoptimes " + " ".join(path) if path else "stoptimes"
    level = "##" if len(path) <= 1 else "###"
    lines.append(f"{level} `{heading}`\n")
    lines.append("```text")
    lines.append(_help_text(cmd, path).rstrip())
    lines.append("```\n")
    # Typer vendors its own click fork (`typer.core.TyperGroup`), which is
    # not an instance of the real `click.Group` — duck-type on `.commands`
    # instead of `isinstance(cmd, click.Group)`.
    subcommands = getattr(cmd, "commands", None)
    if subcommands:
        for name, sub in sorted(subcommands.items()):
            _walk(sub, [*path, name], lines)


def main() -> None:
    root = typer.main.get_command(app)
    lines: list[str] = [HEADER, ""]
    _walk(root, [], lines)
    lines.append(f"\n<!-- generated {datetime.date.today().isoformat()} -->\n")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
