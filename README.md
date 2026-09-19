<img width="250" alt="databus" src="https://github.com/user-attachments/assets/b2ad45ac-83e5-44cf-a93e-898868763530" />

# Stop Times

**stoptimes** is a toolkit for arrival-time prediction on GTFS data: record GTFS-Realtime
feeds, train models on them, serve GTFS-RT TripUpdates, and validate any prediction
source against what actually happened.

> **Status: under construction** (Phase 3 of [docs/PLAN.md](docs/PLAN.md)). The code
> from SIMOVI's `gtfs-eta` and `etaval` repositories has been migrated here with its
> full history; see [docs/MIGRATION_MAP.md](docs/MIGRATION_MAP.md).

**Docs:** the [documentation site](docs/index.md) has a full quickstart ("I have a GTFS
zip and a realtime URL"), concept explanations, per-task guides, a generated CLI
reference, and the SIMOVI (bUCR/MBTA) case study. Build it locally with:

```bash
uv run --with mkdocs-material --no-project mkdocs build -f mkdocs.yml
uv run --with mkdocs-material --no-project mkdocs serve -f mkdocs.yml
```

Two things get called "serving" in this project, and they're different: `stoptimes.serve`
is a **prediction library** (load a model, compute its inputs, produce stop time updates)
that any caller with its own feed builder — like SIMOVI's `databus` — calls directly, and
`stoptimes serve` is a **standalone publisher** for an agency without one, and for watching
your own model work end to end. Neither one is a solution to the operations problem of
running a public realtime feed at scale. See the docs' [selection policy
page](docs/concepts/selection-policy.md#two-things-called-serving) for the full picture.

## Layout

| Package | What it does |
|---|---|
| `packages/core` | GTFS static + realtime readers, domain models, map-matching, features, contracts |
| `packages/models` | Model families and the model registry |
| `packages/train` | Datasets, feature engineering, training, offline evaluation |
| `packages/serve` | Inference and TripUpdates publishing |
| `packages/record` | Archiving live GTFS-RT feeds |
| `packages/val` | Validation (formerly etaval): API, TUI, scoring and statistics |
| `packages/cli` | The `stoptimes` command |
| `web/` | Validation web console |

## Development

```bash
uv sync --group dev          # add --group dl for the LSTM (torch)
uv run stoptimes --help
uv run --no-sync python -m pytest packages/<pkg> -q
```

Python 3.12 is pinned for development (`.python-version`) because the LSTM's torch 2.2
has no newer wheels.

**Checkout under iCloud (`~/Desktop`, `~/Documents`)?** Keep the virtualenv outside it.
iCloud marks files hidden, Python skips hidden `.pth` files, and editable workspace
installs then stop importing at random. One-time setup:

```bash
uv venv --python 3.12 ~/.venvs/databus-stop-times
ln -s ~/.venvs/databus-stop-times .venv
uv sync --group dev
```

## License

MIT
