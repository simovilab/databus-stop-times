"""EtavalApp — Textual entrypoint for the ``etaval-tui`` ops console.

Four screens (F1..F4): CONSOLE (launch + watch a run live), LEADERBOARD
(browse completed runs), EXPLORER (results table + horizon-MAE bars),
TELEMETRY (poll /api/telemetry + tail /api/telemetry/events).

The app owns a single :class:`~etaval.tui.client.ApiClient` (or an injected
fake for tests) reachable from any screen via ``self.app.client``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from textual.app import App
from textual.binding import Binding

from etaval.tui.client import ApiClient
from etaval.tui.screens import (
    ConsoleScreen,
    ExplorerScreen,
    LeaderboardScreen,
    TelemetryScreen,
)

_CSS_PATH = Path(__file__).parent / "styles.tcss"


class EtavalApp(App):
    """Ops-console Textual app with four F-key-switchable screens."""

    CSS_PATH = str(_CSS_PATH)
    TITLE = "etaval ops console"

    BINDINGS = [
        Binding("f1", "show_console", "Console"),
        Binding("f2", "show_leaderboard", "Leaderboard"),
        Binding("f3", "show_explorer", "Explorer"),
        Binding("f4", "show_telemetry", "Telemetry"),
        Binding("q", "quit", "Quit"),
    ]

    SCREENS = {
        "console": ConsoleScreen,
        "leaderboard": LeaderboardScreen,
        "explorer": ExplorerScreen,
        "telemetry": TelemetryScreen,
    }

    def __init__(self, client: ApiClient | None = None, api_url: str | None = None) -> None:
        super().__init__()
        self.client = client if client is not None else ApiClient(api_url=api_url)

    def on_mount(self) -> None:
        self.push_screen("console")

    # ── Screen switching ─────────────────────────────────────────────────────

    def action_show_console(self) -> None:
        self._switch("console")

    def action_show_leaderboard(self) -> None:
        self._switch("leaderboard")

    def action_show_explorer(self) -> None:
        self._switch("explorer")

    def action_show_telemetry(self) -> None:
        self._switch("telemetry")

    def _switch(self, name: str) -> None:
        # Pop back to the base screen before switching so the SCREENS stack
        # never grows unbounded as the operator hops between F-keys.
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen(name)

    async def on_unmount(self) -> None:
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()


def main() -> None:
    """Console-script entry point: ``etaval-tui [--api-url URL]``."""
    parser = argparse.ArgumentParser(prog="etaval-tui")
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of the etaval REST API (default: $ETVAL_API_URL or "
        "http://localhost:8010)",
    )
    args = parser.parse_args()
    app = EtavalApp(api_url=args.api_url)
    app.run()


if __name__ == "__main__":
    main()
