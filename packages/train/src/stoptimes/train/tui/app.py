"""ExperimentBuilderApp -- the Textual experiment builder (docs/PLAN.md
Phase 3, track U1).

Imports Textual, so this module is only ever imported after the caller has
confirmed it's installed (`stoptimes train ui`'s CLI body does a
`find_spec("textual")` check first, per docs/PLAN.md rule about optional
parts giving a clear install hint rather than a traceback). Nothing in
`stoptimes.train`'s package `__init__` imports this module.

Five tabs -- Data & target, Model, Features, Split & output, Review -- built
on the `ExperimentDraft` from `builder.py`. Saving calls `builder.write_draft`
and nothing else: this app never trains a model or touches the model/dataset
libraries directly (docs/PLAN.md rule 2).
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.selection_list import Selection

from stoptimes.core.contracts.experiment import DEFAULT_BASELINES
from stoptimes.train.tui.builder import ExperimentDraft, ExperimentDraftError, open_draft, write_draft
from stoptimes.train.tui.registries import list_baselines, list_features, list_model_families, list_targets


def _param_value_to_str(value: object) -> str:
    return "" if value is None else str(value)


class ExperimentBuilderApp(App):
    """Interactive experiment.yaml builder. `run_app(path)` is the entry point."""

    CSS = """
    #problems { color: red; height: auto; }
    #status { color: green; height: auto; }
    .section-title { text-style: bold; padding-top: 1; }
    """

    TITLE = "stoptimes experiment builder"

    BINDINGS = [("ctrl+s", "save", "Save"), ("q", "quit", "Quit")]

    def __init__(self, draft: ExperimentDraft | None = None, save_path: str | Path | None = None) -> None:
        super().__init__()
        self.draft = draft if draft is not None else ExperimentDraft()
        self.save_path = str(save_path) if save_path is not None else "experiment.yaml"
        self._families = list_model_families()
        self._baselines = list_baselines()
        self._features = list_features()
        self._targets = list_targets()

    # ── layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="data"):
            with TabPane("Data & target", id="data"):
                yield from self._compose_data_tab()
            with TabPane("Model", id="model"):
                yield from self._compose_model_tab()
            with TabPane("Features", id="features"):
                yield from self._compose_features_tab()
            with TabPane("Split & output", id="split"):
                yield from self._compose_split_tab()
            with TabPane("Review & save", id="review"):
                yield from self._compose_review_tab()
        yield Footer()

    def _compose_data_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label("Experiment name")
            yield Input(value=self.draft.name, placeholder="e.g. xgb-baseline", id="name")

            yield Label("Data sources (comma-separated source ids, e.g. mbta,bucr)")
            yield Input(value=",".join(self.draft.sources), id="sources")

            yield Label("Named datasets (comma-separated, alternative to sources)")
            yield Input(value=",".join(self.draft.datasets), id="datasets")

            yield Label("Routes (comma-separated route ids, blank = all routes)")
            yield Input(value=",".join(self.draft.routes or []), id="routes")

            yield Static("Target", classes="section-title")
            options = [(f"{t.name} -- {t.description}", t.name) for t in self._targets]
            yield Select(options, value=self.draft.target or None, id="target-select", allow_blank=True)
            yield Label("Custom target (overrides the dropdown if set)")
            yield Input(value="" if self.draft.target in {t.name for t in self._targets} else self.draft.target, id="target-custom")

    def _compose_model_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield Static("Model family", classes="section-title")
            for family in self._families:
                yield RadioButton(
                    f"{family.name} -- {family.description}",
                    value=(family.name == self.draft.family),
                    id=f"family-{family.name}",
                )
            yield Static("", id="family-description")

            yield Static("Hyperparameters (one per line: name=value)", classes="section-title")
            yield Label(
                "Leave a value blank to use the family's default. See the family's "
                "description above for what each parameter means."
            )
            params_text = "\n".join(f"{k}={_param_value_to_str(v)}" for k, v in self.draft.params.items())
            yield Input(value=params_text.replace("\n", ";") if params_text else "", id="params", placeholder="max_depth=6;learning_rate=0.05")

            yield Static("Baselines always scored alongside the model", classes="section-title")
            baseline_selections = [
                Selection(f"{b.name} -- {b.description}", b.name, b.name in self.draft.baselines)
                for b in self._baselines
            ]
            yield SelectionList(*baseline_selections, id="baselines")

    def _compose_features_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield Static(
                "Select the named features this experiment computes. Descriptions come "
                "from stoptimes.core.features.registry.",
                classes="section-title",
            )
            selections = [
                Selection(f"{f.name} [{','.join(f.tags)}] -- {f.description}", f.name, f.name in self.draft.features)
                for f in self._features
            ]
            yield SelectionList(*selections, id="feature-list")

    def _compose_split_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield Static("Split strategy", classes="section-title")
            with RadioSet(id="split-kind"):
                yield RadioButton("time -- chronological train/test split", value=self.draft.split_kind == "time", id="split-time")
                yield RadioButton(
                    "trip -- chronological split, purged so no trip straddles train/test",
                    value=self.draft.split_kind == "trip",
                    id="split-trip",
                )
                yield RadioButton(
                    "kfold -- last 1/n_folds of rows (time-ordered) held out",
                    value=self.draft.split_kind == "kfold",
                    id="split-kfold",
                )
            with Horizontal():
                yield Vertical(
                    Label("train_fraction"),
                    Input(value=_param_value_to_str(self.draft.train_fraction), id="train-fraction"),
                )
                yield Vertical(
                    Label("test_fraction"),
                    Input(value=_param_value_to_str(self.draft.test_fraction), id="test-fraction"),
                )
                yield Vertical(
                    Label("n_folds (kfold only)"),
                    Input(value=_param_value_to_str(self.draft.n_folds), id="n-folds"),
                )

            yield Static("Output", classes="section-title")
            yield Label("Registry directory")
            yield Input(value=self.draft.registry_dir, id="registry-dir")
            yield Label("Tags (comma-separated)")
            yield Input(value=",".join(self.draft.tags), id="tags")
            yield Label("Seed")
            yield Input(value=str(self.draft.seed), id="seed")

    def _compose_review_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label("Write to (path)")
            yield Input(value=self.save_path, id="save-path")
            yield Button("Save experiment.yaml", id="save-button", variant="primary")
            yield Static("", id="problems")
            yield Static("", id="status")

    # ── event handling ───────────────────────────────────────────────────

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "split-kind" and event.pressed is not None:
            mapping = {"split-time": "time", "split-trip": "trip", "split-kfold": "kfold"}
            self.draft.split_kind = mapping.get(event.pressed.id, "time")  # type: ignore[assignment]

    def on_radio_button_changed(self, event: RadioButton.Changed) -> None:
        button_id = event.radio_button.id or ""
        if button_id.startswith("family-") and event.value:
            self.draft.family = button_id[len("family-") :]
            info = next((f for f in self._families if f.name == self.draft.family), None)
            desc_widget = self.query_one("#family-description", Static)
            desc_widget.update(info.description if info else "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-button":
            self.action_save()

    def _sync_draft_from_widgets(self) -> None:
        draft = self.draft
        draft.name = self.query_one("#name", Input).value.strip()
        draft.sources = _split_csv(self.query_one("#sources", Input).value)
        draft.datasets = _split_csv(self.query_one("#datasets", Input).value)
        routes = _split_csv(self.query_one("#routes", Input).value)
        draft.routes = routes or None

        custom_target = self.query_one("#target-custom", Input).value.strip()
        target_select = self.query_one("#target-select", Select)
        selected = target_select.value
        draft.target = custom_target or (selected if isinstance(selected, str) else draft.target)

        draft.params = _parse_params(self.query_one("#params", Input).value)

        baselines_widget = self.query_one("#baselines", SelectionList)
        draft.baselines = list(baselines_widget.selected) or list(DEFAULT_BASELINES)

        features_widget = self.query_one("#feature-list", SelectionList)
        draft.features = list(features_widget.selected)

        draft.train_fraction = _parse_optional_float(self.query_one("#train-fraction", Input).value)
        draft.test_fraction = _parse_optional_float(self.query_one("#test-fraction", Input).value)
        draft.n_folds = _parse_optional_int(self.query_one("#n-folds", Input).value)

        draft.registry_dir = self.query_one("#registry-dir", Input).value.strip() or "artifacts/models"
        draft.tags = _split_csv(self.query_one("#tags", Input).value)
        seed_value = _parse_optional_int(self.query_one("#seed", Input).value)
        draft.seed = seed_value if seed_value is not None else 42

    def action_save(self) -> None:
        self._sync_draft_from_widgets()
        path = self.query_one("#save-path", Input).value.strip() or self.save_path
        problems_widget = self.query_one("#problems", Static)
        status_widget = self.query_one("#status", Static)
        try:
            write_draft(self.draft, path)
        except ExperimentDraftError as exc:
            problems_widget.update("\n".join(f"- {p}" for p in exc.problems))
            status_widget.update("")
            return
        problems_widget.update("")
        status_widget.update(f"Saved {path}. Run it with:  stoptimes train run {path}")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _parse_params(value: str) -> dict[str, object]:
    """Parse `name=value` pairs separated by `;`, with best-effort scalar
    typing (int -> float -> bool -> str). Ranges/choices for tuning aren't
    entered through this compact field yet -- see docs/notes/p3-u1.md."""
    params: dict[str, object] = {}
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key or not raw:
            continue
        params[key] = _coerce_scalar(raw)
    return params


def _coerce_scalar(raw: str) -> object:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def run_app(path: str | Path | None = None) -> None:
    """Entry point called by `stoptimes train ui`. Opens `path` for editing
    if it exists, otherwise starts a blank draft that will be written to
    `path` (or `experiment.yaml` if `path` is omitted)."""
    if path is not None and Path(path).exists():
        draft = open_draft(path)
    else:
        draft = ExperimentDraft()
    app = ExperimentBuilderApp(draft=draft, save_path=path)
    app.run()
