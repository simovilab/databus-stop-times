"""The four etaval-tui screens: CONSOLE, LEADERBOARD, EXPLORER, TELEMETRY.

Every screen talks to the API exclusively through ``self.app.client`` (an
:class:`~stoptimes.val.tui.client.ApiClient` or a test fake with the same method
signatures) — no widget imports httpx / websockets directly.  This keeps the
screens testable with ``App.run_test()`` and a fake client (see
``tests/tui/test_app.py``).

Stats surfaces (CIs, significance, provenance, calibration) are rendered on
LEADERBOARD's run-detail panel (for DONE runs, via ``get_metrics(...,
stats=1)`` + ``get_run(...)`` for provenance) and on EXPLORER's calibration
panel. Every stats field is optional/nullable end to end — a missing value
renders as ``—`` rather than raising.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    SelectionList,
    Static,
)

# ── CONSOLE ──────────────────────────────────────────────────────────────────


class ConsoleScreen(Screen):
    """Launch a new run and watch it live over the WS stream."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back", show=False)]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="panel"):
            yield Label("SOURCE", classes="micro-label")
            yield Select([], id="source-select", prompt="Select a source")
            yield Label("PREDICTORS", classes="micro-label")
            yield SelectionList(id="predictor-list")
            with Horizontal():
                with Vertical():
                    yield Label("ROUTES (comma-separated)", classes="micro-label")
                    yield Input(placeholder="e.g. 1,66,CR-Fairmount", id="routes-input")
                with Vertical():
                    yield Label("DURATION", classes="micro-label")
                    yield Input(value="30m", id="duration-input")
            yield Button("START RUN", id="start-button", variant="success")
        with Horizontal(classes="panel"):
            yield Static("TICKS\n0", id="tile-ticks", classes="tile")
            yield Static("SAMPLES\n0", id="tile-samples", classes="tile")
            yield Static("ARRIVALS\n0", id="tile-arrivals", classes="tile")
            yield Static("RESULTS\n0", id="tile-results", classes="tile")
        yield DataTable(id="results-table")
        yield Static("idle", id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("PREDICTOR", "STOP", "ERROR_S", "HORIZON_S")
        self.run_worker(self._load_sources(), exclusive=True, group="load-sources")
        self.run_worker(self._load_predictors(), exclusive=True, group="load-predictors")

    async def _load_sources(self) -> None:
        try:
            sources = await self.app.client.get_sources()
        except Exception:  # noqa: BLE001
            sources = []
        select = self.query_one("#source-select", Select)
        select.set_options([(s["name"], s["name"]) for s in sources])

    async def _load_predictors(self) -> None:
        try:
            predictors: list[str] = await self.app.client.get_predictors()
        except Exception:  # noqa: BLE001
            predictors = []
        selection_list = self.query_one("#predictor-list", SelectionList)
        selection_list.clear_options()
        for pid in predictors:
            selection_list.add_option((pid, pid, False))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            self.run_worker(self._start_run(), exclusive=True, group="start-run")

    async def _start_run(self) -> None:
        source_select = self.query_one("#source-select", Select)
        source = source_select.value
        if source is Select.BLANK or not source:
            self._set_status("Pick a source first", level="warn")
            return

        predictor_list = self.query_one("#predictor-list", SelectionList)
        predictors = list(predictor_list.selected)

        routes_raw = self.query_one("#routes-input", Input).value
        routes = [r.strip() for r in routes_raw.split(",") if r.strip()]

        duration = self.query_one("#duration-input", Input).value or "30m"

        payload = {
            "source": source,
            "predictors": predictors,
            "routes": routes,
            "duration": duration,
        }
        self._set_status("starting run...")
        try:
            resp = await self.app.client.create_run(payload)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"failed to start run: {exc}", level="err")
            return

        run_id = resp["run_id"]
        self._set_status(f"run {run_id} started — streaming")
        self.run_worker(self._watch_run(run_id), exclusive=True, group="watch-run")

    async def _watch_run(self, run_id: str) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        try:
            async for frame in self.app.client.stream(run_id):
                ftype = frame.get("type")
                if ftype == "tick":
                    data = frame["data"]
                    self._update_tile("tile-ticks", "TICKS", data.get("tick", 0))
                    self._update_tile("tile-samples", "SAMPLES", data.get("samples_saved", 0))
                    self._update_tile("tile-arrivals", "ARRIVALS", data.get("arrivals_saved", 0))
                    self._update_tile("tile-results", "RESULTS", data.get("results_saved", 0))
                    for row in data.get("new_results", [])[:20]:
                        table.add_row(
                            row.get("predictor_id", ""),
                            row.get("stop_id", ""),
                            f"{row.get('error_s', 0):.1f}",
                            f"{row.get('horizon_s', 0):.0f}",
                        )
                elif ftype == "end":
                    status = frame["data"].get("status", "done")
                    self._set_status(f"run ended: {status}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"stream error: {exc}", level="err")

    def _update_tile(self, widget_id: str, label: str, value: Any) -> None:
        self.query_one(f"#{widget_id}", Static).update(f"{label}\n{value}")

    def _set_status(self, message: str, level: str = "info") -> None:
        self.query_one("#status-line", Static).update(message)


# ── LEADERBOARD ──────────────────────────────────────────────────────────────


class LeaderboardScreen(Screen):
    """Browse all known runs; Enter opens a per-run detail view."""

    BINDINGS = [Binding("escape", "back", "Back", show=False)]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="runs-table")
        with Vertical(id="run-detail", classes="panel"):
            yield Static("", id="winner-banner")
            yield DataTable(id="predictor-metrics-table")
            yield DataTable(id="horizon-buckets-table")
            yield Label("LATENCY", classes="micro-label")
            yield DataTable(id="latency-table")
            yield Label("SIGNIFICANCE", classes="micro-label")
            yield Static("", id="significance-block")
            yield Label("PROVENANCE", classes="micro-label")
            yield Static("", id="provenance-block")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns("RUN_ID", "STATUS", "SOURCE", "TICKS", "RESULTS")
        latency_table = self.query_one("#latency-table", DataTable)
        latency_table.add_columns("PREDICTOR", "P50_S", "P95_S", "P99_S", "COUNT")
        self.query_one("#run-detail").display = False
        self.run_worker(self._load_runs(), exclusive=True, group="load-runs")

    async def _load_runs(self) -> None:
        try:
            runs = await self.app.client.list_runs()
        except Exception:  # noqa: BLE001
            runs = []
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        self._runs_by_row: list[str] = []
        for run in runs:
            summary = run.get("summary") or {}
            params = run.get("params") or {}
            table.add_row(
                run["id"][:8],
                run.get("status", ""),
                params.get("source", ""),
                str(summary.get("ticks", 0)),
                str(summary.get("results_saved", 0)),
            )
            self._runs_by_row.append(run["id"])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "runs-table":
            return
        idx = event.cursor_row
        if 0 <= idx < len(getattr(self, "_runs_by_row", [])):
            run_id = self._runs_by_row[idx]
            self.run_worker(self._open_detail(run_id), exclusive=True, group="run-detail")

    async def _open_detail(self, run_id: str) -> None:
        run_info: dict[str, Any] = {}
        try:
            run_info = await self.app.client.get_run(run_id)
        except Exception:  # noqa: BLE001
            pass
        is_done = run_info.get("status") == "done"

        try:
            metrics = await self.app.client.get_metrics(run_id, stats=1 if is_done else 0)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#winner-banner", Static).update(f"metrics unavailable: {exc}")
            self.query_one("#run-detail").display = True
            return

        per_predictor = metrics.get("per_predictor", [])
        winner = min(per_predictor, key=lambda p: p.get("mae_s", float("inf")), default=None)
        banner = f"WINNER: {winner['predictor_id']}" if winner else "No results yet"
        self.query_one("#winner-banner", Static).update(banner)

        pred_table = self.query_one("#predictor-metrics-table", DataTable)
        pred_table.clear(columns=True)
        pred_table.add_columns("PREDICTOR", "MAE [lo–hi]", "N")
        for row in per_predictor:
            pred_table.add_row(
                row.get("predictor_id", ""), self._mae_ci_cell(row), str(row.get("n", 0))
            )

        horizon_table = self.query_one("#horizon-buckets-table", DataTable)
        horizon_table.clear(columns=True)
        horizon_table.add_columns("HORIZON_BUCKET", "MAE_S", "N")
        for row in metrics.get("horizon_buckets", []):
            horizon_table.add_row(
                str(row.get("bucket", "")),
                f"{row.get('mae_s', 0):.1f}",
                str(row.get("n", 0)),
            )

        self._render_latency(per_predictor)
        self._render_significance(metrics.get("significance"))
        self._render_provenance(run_info.get("provenance"), metrics.get("seed"))

        self.query_one("#run-detail").display = True

    def _mae_ci_cell(self, row: dict[str, Any]) -> Text:
        """``MAE [lo–hi]`` cell — MAE plain, CI in the dim token color, or ``—`` when absent."""
        mae = row.get("mae")
        mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else "—"
        ci = row.get("mae_ci")
        if not ci:
            return Text(mae_str)
        lo, hi = ci.get("lo"), ci.get("hi")
        if lo is None or hi is None:
            return Text(mae_str)
        text = Text(mae_str)
        text.append(f" [{lo:.1f}–{hi:.1f}]", style="dim")
        return text

    def _render_latency(self, per_predictor: list[dict[str, Any]]) -> None:
        table = self.query_one("#latency-table", DataTable)
        table.clear()
        for row in per_predictor:
            latency = row.get("latency")
            pid = row.get("predictor_id", "")
            if not latency:
                table.add_row(pid, "—", "—", "—", "—")
                continue
            table.add_row(
                pid,
                f"{latency.get('p50_s', 0):.3f}",
                f"{latency.get('p95_s', 0):.3f}",
                f"{latency.get('p99_s', 0):.3f}",
                str(latency.get("count", 0)),
            )

    def _render_significance(self, significance: list[dict[str, Any]] | None) -> None:
        widget = self.query_one("#significance-block", Static)
        if not significance:
            widget.update("no significance data" if significance is not None else "—")
            return
        lines = []
        for row in significance:
            a, b = row.get("a", "?"), row.get("b", "?")
            diff_mean = row.get("diff_mean", 0.0) or 0.0
            better, worse = (a, b) if diff_mean <= 0 else (b, a)
            star = "*" if row.get("reject") else ""
            p_adj = row.get("p_adjusted", row.get("p_value", 0.0))
            lines.append(f"[{row.get('test', '?')}] {better} > {worse} (p={p_adj:.2f}{star})")
        widget.update("\n".join(lines))

    def _render_provenance(
        self, provenance: dict[str, Any] | None, seed: int | None
    ) -> None:
        widget = self.query_one("#provenance-block", Static)
        if not provenance:
            widget.update("—")
            return

        def _sha(sha: str | None, dirty: bool | None) -> str:
            if not sha:
                return "—"
            return f"{sha[:8]}{'*' if dirty else ''}"

        etaval_sha = _sha(provenance.get("etaval_git_sha"), provenance.get("etaval_dirty"))
        gtfs_eta_sha = _sha(provenance.get("gtfs_eta_git_sha"), provenance.get("gtfs_eta_dirty"))
        lines = [
            f"etaval:    {etaval_sha}",
            f"gtfs-eta:  {gtfs_eta_sha}",
            f"snapshot:  {provenance.get('gtfs_snapshot') or '—'}",
            f"seed:      {seed if seed is not None else provenance.get('seed', '—')}",
            f"config:    {(provenance.get('source_config_sha256') or '—')[:12]}",
            f"created:   {provenance.get('created_at') or '—'}",
        ]
        widget.update("\n".join(lines))

    def action_back(self) -> None:
        detail = self.query_one("#run-detail")
        if detail.display:
            detail.display = False
        else:
            self.app.pop_screen()


# ── EXPLORER ─────────────────────────────────────────────────────────────────


class ExplorerScreen(Screen):
    """Results table with predictor/route filters and a horizon-MAE bar chart."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(classes="panel"):
            with Vertical():
                yield Label("PREDICTOR FILTER", classes="micro-label")
                yield Input(placeholder="predictor_id", id="predictor-filter")
            with Vertical():
                yield Label("ROUTE FILTER", classes="micro-label")
                yield Input(placeholder="route_id", id="route-filter")
            yield Button("APPLY", id="apply-filter")
        yield DataTable(id="explorer-table")
        yield Static("", id="horizon-chart", classes="panel")
        yield Label("CALIBRATION (nominal vs empirical coverage)", classes="micro-label")
        yield Static("", id="calibration-block", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#explorer-table", DataTable)
        table.add_columns("PREDICTOR", "ROUTE", "STOP", "ERROR_S", "HORIZON_S")
        self._run_id: str | None = None

    def set_run(self, run_id: str) -> None:
        """Load results for *run_id* (called externally or wired to Leaderboard later)."""
        self._run_id = run_id
        self.run_worker(self._load_results(), exclusive=True, group="load-results")
        self.run_worker(self._load_calibration(), exclusive=True, group="load-calibration")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-filter":
            self.run_worker(self._load_results(), exclusive=True, group="load-results")

    async def _load_results(self) -> None:
        if self._run_id is None:
            return
        predictor_filter = self.query_one("#predictor-filter", Input).value.strip() or None
        route_filter = self.query_one("#route-filter", Input).value.strip() or None

        try:
            results = await self.app.client.get_results(
                self._run_id, predictor=predictor_filter
            )
        except Exception:  # noqa: BLE001
            results = []

        if route_filter:
            results = [r for r in results if r.get("route_id") == route_filter]

        table = self.query_one("#explorer-table", DataTable)
        table.clear()
        for row in results[:500]:
            table.add_row(
                row.get("predictor_id", ""),
                row.get("route_id", "") or "",
                row.get("stop_id", ""),
                f"{row.get('error_s', 0):.1f}",
                f"{row.get('horizon_s', 0):.0f}",
            )

        self._render_horizon_chart(results)

    def _render_horizon_chart(self, results: list[dict[str, Any]]) -> None:
        """A simple text-bar horizon-MAE chart, bucketed by 60s horizon bins."""
        buckets: dict[int, list[float]] = {}
        for row in results:
            horizon = row.get("horizon_s", 0)
            bucket = int(horizon // 60) * 60
            buckets.setdefault(bucket, []).append(abs(row.get("error_s", 0)))

        lines = []
        max_mae = max((sum(v) / len(v) for v in buckets.values()), default=0.0)
        for bucket in sorted(buckets):
            values = buckets[bucket]
            mae = sum(values) / len(values)
            bar_len = int((mae / max_mae) * 30) if max_mae > 0 else 0
            lines.append(f"{bucket:>5}s | {'#' * bar_len} {mae:.1f}")

        self.query_one("#horizon-chart", Static).update("\n".join(lines) or "no data")

    async def _load_calibration(self) -> None:
        widget = self.query_one("#calibration-block", Static)
        if self._run_id is None:
            widget.update("—")
            return
        try:
            metrics = await self.app.client.get_metrics(self._run_id, stats=1)
        except Exception:  # noqa: BLE001
            widget.update("—")
            return

        calibration = metrics.get("calibration")
        if not calibration:
            widget.update("no calibration data")
            return

        lines = []
        for row in calibration:
            band = row.get("band_label", "?")
            tol = row.get("tolerance_s", 0)
            nominal = row.get("nominal_coverage", 0.0) or 0.0
            empirical = row.get("empirical_coverage", 0.0) or 0.0
            n = row.get("n", 0)
            lines.append(
                f"{band:>6} @{tol:>4.0f}s  nominal={nominal:5.2f}  "
                f"empirical={empirical:5.2f}  n={n}"
            )
        widget.update("\n".join(lines))


# ── TELEMETRY ────────────────────────────────────────────────────────────────


class TelemetryScreen(Screen):
    """Poll /api/telemetry every 5s and tail /api/telemetry/events."""

    POLL_INTERVAL_S = 5.0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(classes="panel"):
            yield Static("UPTIME\n--", id="tile-uptime", classes="tile")
            yield Static("ACTIVE RUNS\n--", id="tile-active", classes="tile")
        yield DataTable(id="telemetry-table")
        yield Label("EVENTS", classes="micro-label")
        yield RichLog(id="event-log", markup=True, max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#telemetry-table", DataTable)
        table.add_columns("RUN_ID", "STATUS", "TICKS", "TICK_RATE", "LAST_TICK_AGE_S", "FEED_OK")
        self._last_event_count = 0
        self.run_worker(self._poll_once(), exclusive=True, group="telemetry-poll")
        self.set_interval(self.POLL_INTERVAL_S, self._schedule_poll)

    def _schedule_poll(self) -> None:
        self.run_worker(self._poll_once(), exclusive=True, group="telemetry-poll")

    async def _poll_once(self) -> None:
        try:
            snap = await self.app.client.get_telemetry()
        except Exception:  # noqa: BLE001
            return

        self.query_one("#tile-uptime", Static).update(
            f"UPTIME\n{snap.get('api_uptime_s', 0):.0f}s"
        )
        self.query_one("#tile-active", Static).update(
            f"ACTIVE RUNS\n{snap.get('active_runs', 0)}"
        )

        table = self.query_one("#telemetry-table", DataTable)
        table.clear()
        for row in snap.get("per_run", []):
            feed = row.get("feed") or {}
            table.add_row(
                str(row.get("run_id", ""))[:8],
                row.get("status", "") or "",
                str(row.get("ticks", 0)),
                f"{row.get('tick_rate', 0):.2f}",
                f"{row.get('last_tick_age_s', 0) or 0:.1f}",
                "yes" if feed.get("ok") else "no",
            )

        try:
            events = await self.app.client.get_telemetry_events(limit=100)
        except Exception:  # noqa: BLE001
            events = []

        log = self.query_one("#event-log", RichLog)
        new_events = events[self._last_event_count :]
        for event in new_events:
            color = {"info": "cyan", "warn": "yellow", "error": "red"}.get(
                event.get("level", "info"), "white"
            )
            log.write(
                f"[{color}]{event.get('level', 'info').upper()}[/] "
                f"{event.get('source', '')}: {event.get('message', '')}"
            )
        self._last_event_count = len(events)
