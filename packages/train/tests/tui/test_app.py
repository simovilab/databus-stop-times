"""Textual pilot smoke tests for ExperimentBuilderApp -- widget wiring only
(mount, fill in the minimum fields, save). Deeper logic (validation, YAML
round trip) is covered without Textual in test_builder.py; per the track
brief this file stays to "at most a couple of smoke tests."
"""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual", reason="textual is an optional dependency of stoptimes-train")

from textual.widgets import Input, RadioButton, SelectionList, Static  # noqa: E402

from stoptimes.train.tui.app import ExperimentBuilderApp  # noqa: E402
from stoptimes.train.tui.builder import ExperimentDraft  # noqa: E402


def _static_content(widget: Static) -> str:
    """Plain-text content last passed to a Static widget's ``.update()``."""
    return str(widget._Static__content)  # noqa: SLF001 -- textual stores it name-mangled


async def test_app_mounts_with_all_tabs() -> None:
    app = ExperimentBuilderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#name", Input) is not None
        assert app.query_one("#feature-list", SelectionList) is not None


async def test_filling_minimum_fields_and_saving_writes_a_loadable_experiment(tmp_path) -> None:
    from stoptimes.core.contracts.experiment import load_experiment

    out_path = tmp_path / "experiment.yaml"
    app = ExperimentBuilderApp(save_path=str(out_path))

    async with app.run_test() as pilot:
        await pilot.pause()

        app.query_one("#name", Input).value = "smoke-test"
        app.query_one("#sources", Input).value = "mbta"

        family_radio = app.query_one("#family-xgb", RadioButton)
        family_radio.value = True
        await pilot.pause()

        app.query_one("#save-path", Input).value = str(out_path)
        app.action_save()
        await pilot.pause()

        status = app.query_one("#status", Static)
        assert "Saved" in _static_content(status)

    assert out_path.exists()
    experiment = load_experiment(out_path)
    assert experiment.name == "smoke-test"
    assert experiment.model.family == "xgb"
    assert experiment.data.sources == ["mbta"]


async def test_opening_an_existing_experiment_prefills_the_form() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    example = repo_root / "examples" / "experiments" / "xgb-baseline.yaml"

    from stoptimes.train.tui.builder import open_draft

    draft = open_draft(example)
    app = ExperimentBuilderApp(draft=draft, save_path=str(example))

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#name", Input).value == "xgb-baseline"
        assert app.query_one("#sources", Input).value == "mbta"
