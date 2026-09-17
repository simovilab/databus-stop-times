from typer.testing import CliRunner

from stoptimes.cli.main import app

runner = CliRunner()


def test_help_lists_every_area():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("data", "record", "train", "models", "serve", "diagnose", "val"):
        assert name in result.output


def test_planned_command_exits_2_and_names_phase():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_val_is_mounted():
    result = runner.invoke(app, ["val", "--help"])
    assert result.exit_code == 0
