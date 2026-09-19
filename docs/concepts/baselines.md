# Baselines

Every `experiment.yaml` trains its chosen model family plus a set of **baselines** —
`schedule` (predict the published schedule, unmodified) and `constant_speed` (predict from
the vehicle's own recent speed, no learning at all) by default — and every report, card, and
validation run shows the baseline's numbers alongside the trained model's.

```yaml
baselines: ["schedule", "constant_speed"]   # the default; the quickstart's toy example sets [] to keep its output minimal
```

## Why this is non-negotiable (docs/PLAN.md rule 5)

A trained model that can't beat "just publish the schedule" isn't worth serving, and the
only way to know that reliably is to compute the schedule baseline on the *same* split, with
the *same* horizon breakdown, every time — never as a separate, easy-to-skip step. This is
also a direct answer to a newcomer user story: "be told plainly when the model is *not*
better than the printed schedule." Baselines shipping unconditionally, rather than as an
opt-in flag, is what makes that plain rather than something you have to remember to check
for.

## Where you'll see them

- **Model cards** — a baseline's row sits next to the trained family's in the scores table.
- **`stoptimes val replay`/`run`** — `baseline:schedule` is a first-class predictor id, usable
  standalone (as in the [quickstart](../quickstart.md#6-score-it-against-what-actually-happened))
  or alongside a trained artifact for comparison.
- **`stoptimes train run --dry-run`** — the resolved plan shows which baselines will be
  computed before you spend time on a real run.

Baselines have no hyperparameters to tune (`ConstantSpeedBaseline` fits its one internal
parameter, `speed_m_s`, from data rather than accepting it as a setting) — there is nothing
to configure beyond including or excluding them.
