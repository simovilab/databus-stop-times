# stoptimes (CLI)

The single entry point for the toolkit. Install only the parts you need:

```bash
# from a clone of the repo, at its root
uv pip install './packages/cli[serve]'   # inference only
uv pip install './packages/cli[train]'   # training
uv pip install './packages/cli[val]'     # validation (API + TUI)
uv pip install './packages/cli[all]'     # everything
```

The `uv sync --extra <name>` form does the same thing, but only from this
directory (`packages/cli`) — from the repo root it resolves against the
workspace, which declares no extras. Contributing? `uv sync --group dev`.

Run `stoptimes --help`. Commands that are designed but not built yet say so and
point at the phase in `docs/PLAN.md` that delivers them.
