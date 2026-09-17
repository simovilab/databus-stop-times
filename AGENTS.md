# Agent guide

Read `docs/PLAN.md` first — §10 is the protocol every agent follows (worktrees, owned
paths, notes instead of cross-track edits, test command, report format).

Quick facts:
- uv workspace; members in `packages/*`, all under the `stoptimes.*` namespace (no `stoptimes/__init__.py`).
- Tests: `uv run --no-sync python -m pytest packages/<pkg> -q -p no:cacheprovider`.
- Dependency direction: core ← models ← serve ← train; core ← record ← train; nothing depends on val.
- Never edit `uv.lock` or the root `pyproject.toml` from a track.
- `legacy/` is reference only; nothing in `packages/` may import it.
