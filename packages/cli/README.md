# stoptimes (CLI)

The single entry point for the toolkit. Install only the parts you need:

```bash
uv sync --extra serve      # inference only
uv sync --extra train      # training
uv sync --extra val        # validation (API + TUI)
uv sync --all-extras       # everything
```

Run `stoptimes --help`. Commands that are designed but not built yet say so and
point at the phase in `docs/PLAN.md` that delivers them.
