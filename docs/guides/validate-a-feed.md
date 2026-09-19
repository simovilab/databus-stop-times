# Guide: validate any feed

For the skeptic user stories: "score *any* TripUpdates feed against what happened without
access to how it was produced," "replay one recorded day across several prediction sources
over identical trips," "break error down by horizon, route and time of day," "watch a live
feed's accuracy degrade."

Validation never reads training internals — it scores predictions purely through what a
`Predictor` id resolves to and what actually happened
([Arrivals](../concepts/arrivals.md)), per `docs/PLAN.md` rule 3.

!!! note "A few seconds' delay is normal, not stuck"
    `stoptimes val --help` (the group) is fast. Running an actual `val` subcommand — including
    that subcommand's own `--help`, e.g. `val replay --help` — still imports scikit-learn and
    matplotlib and takes a few seconds (measured 4-6s on this machine) before printing
    anything. Noticeably slower than the other command groups, but not a hang.

## Replay a recorded day

```bash
stoptimes val replay \
  --source myagency --sources-dir ./agencies \
  --archive ./archive --gtfs ./gtfs.zip \
  --from 2026-09-17 --to 2026-09-18 \
  --predictor baseline:schedule --predictor stoptimes:<artifact_id> \
  --out ./scores.parquet
```

Deterministic: the same archive, window, and predictor list always produce the same scores —
no wall-clock dependency, no live network — which is what makes a run reproducible for CI or
for a paper's numbers. Run for real against the quickstart's fixture (single predictor
shown; repeat `--predictor` to score several sources over the identical trips in one run):

```text
Replaying: source='demo', archive='./archive', ..., predictors=['baseline:schedule'], ...
Done. Ticks=100 Samples=176 Arrivals=32 Results=176
Exported 176 results → ./scores.parquet
```

Predictor ids: `baseline:schedule` / `baseline:constant_speed` need no trained artifact;
`stoptimes:<artifact_id>` scores a registry artifact; `pipeline:<file.py>` scores a full
serving pipeline the way `stoptimes serve` would run it; `gtfs_eta:*` aliases exist for
scoring a legacy artifact directly.

`stoptimes:<artifact_id>` resolves its registry through the `STOPTIMES_MODEL_REGISTRY_DIR`
environment variable, not a flag — `val replay`/`val run` have no `--registry` option. Set it
before invoking the CLI:

```bash
STOPTIMES_MODEL_REGISTRY_DIR=./registry stoptimes val replay --predictor stoptimes:<artifact_id> ...
```

(The deprecated `MODEL_REGISTRY_DIR` name is also read, for compatibility with older
scripts.)

## Turn scores into a report

```bash
stoptimes val report ./scores.parquet
```

Run for real against the quickstart's `scores.parquet`, producing a per-predictor metrics
table and an error-vs-horizon-bucket table (shown in full on the
[quickstart](../quickstart.md#6-score-it-against-what-actually-happened)) — this is the
horizon/route/time-of-day breakdown the skeptic story asks for; `--route`/time filters narrow
either report further. `stoptimes val figures` renders the same breakdown as 300 dpi PNGs for
a paper or a report.

## Score a live feed instead of a replay

```bash
stoptimes val run --source myagency --sources-dir ./agencies
```

Runs headless validation against a *live* GTFS-RT source rather than an archive — the
"watch a live feed's accuracy degrade" story. `stoptimes val walk-forward` runs a
walk-forward backtest (per-window MAE with confidence intervals) over an archive instead of
one fixed split, and `stoptimes val api`/`stoptimes val tui` (was `etaval-api`/`etaval-tui`)
run the REST+WebSocket API and the terminal ops console respectively for watching this
continuously rather than one command at a time.

## Scoring your own producer, not just `stoptimes`'s

Nothing about `val replay`/`run` requires the predictions under test to have come from
`stoptimes train`/`serve` at all — `pipeline:<file.py>` scores any pipeline (yours or ours),
and any TripUpdates feed reachable at a URL or archived in the recorder's layout can be
scored the same way. This is deliberate: validation is meant to answer "is this feed any
good," independent of who or what produced it.
