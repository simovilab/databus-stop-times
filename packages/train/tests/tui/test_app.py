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
        # Regression: start_date/end_date used to have no widget at all, so
        # opening a dated experiment and saving silently dropped the date
        # range (round-tripped fine in builder.py, just never reached the
        # form). Assert the fields are actually prefilled here, and assert
        # the full open -> save cycle preserves them below.
        assert app.query_one("#start-date", Input).value == "2026-06-01"
        assert app.query_one("#end-date", Input).value == "2026-08-31"


async def test_editing_and_resaving_a_dated_experiment_preserves_its_date_range(tmp_path) -> None:
    """The regression test for the date-range silent-drop bug: open a real
    shipped experiment through the TUI's own load path, save it back out
    *without touching the date fields*, and confirm the date range in the
    written YAML still matches the original -- not just that `builder.py`'s
    dataclass round-trips in isolation (test_builder.py already covers
    that), but that the actual widget-sync-then-save path preserves it."""
    from pathlib import Path

    from stoptimes.core.contracts.experiment import load_experiment
    from stoptimes.train.tui.builder import open_draft

    repo_root = Path(__file__).resolve().parents[4]
    example = repo_root / "examples" / "experiments" / "xgb-baseline.yaml"
    original = load_experiment(example)
    assert original.data.start_date is not None and original.data.end_date is not None

    out_path = tmp_path / "resaved.yaml"
    draft = open_draft(example)
    app = ExperimentBuilderApp(draft=draft, save_path=str(out_path))

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#save-path", Input).value = str(out_path)
        app.action_save()
        await pilot.pause()

        status = app.query_one("#status", Static)
        assert "Saved" in _static_content(status)

    resaved = load_experiment(out_path)
    assert resaved.data.start_date == original.data.start_date
    assert resaved.data.end_date == original.data.end_date


async def test_invalid_date_input_blocks_save_without_crashing_or_dropping_the_field(tmp_path) -> None:
    out_path = tmp_path / "should-not-exist.yaml"
    app = ExperimentBuilderApp(save_path=str(out_path))

    async with app.run_test() as pilot:
        await pilot.pause()

        app.query_one("#name", Input).value = "bad-date-test"
        app.query_one("#sources", Input).value = "mbta"
        app.query_one("#family-xgb", RadioButton).value = True
        app.query_one("#start-date", Input).value = "not-a-date"
        await pilot.pause()

        app.query_one("#save-path", Input).value = str(out_path)
        app.action_save()  # must not raise
        await pilot.pause()

        problems = app.query_one("#problems", Static)
        assert "start_date" in _static_content(problems)

    assert not out_path.exists()
