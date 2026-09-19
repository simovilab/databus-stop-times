# Selection policy

When a registry holds several eligible artifacts for a trip — a global model and a
route-specific one, say — something has to decide which one actually serves a given
prediction. `select_artifact(candidates, policy, *, source_id, route_id=, direction_id=,
trip_id=)` is that decision, and it's a pure function over `artifact.json` metadata: no model
file is opened to choose.

## The rule, and why it's in this order

1. **Explicit pins first.** `SelectionPolicy.pins` maps a scope key (`"global"`,
   `"route:22"`, `"route:22/1"`, `"trip:t1"`) to an artifact id. A pin naming a missing
   artifact is ignored rather than erroring, so a deleted model degrades gracefully to
   ranking instead of refusing to serve anything.
2. **Then the most specific matching scope** in `scope_order` (`trip → direction → route →
   global` by default).
3. **Within that level, the best `metric`** (default `test_mae_seconds`, lower is better),
   ties broken by `artifact_id` so the choice is deterministic — the same registry state
   always picks the same artifact, which matters for reproducing a serving decision later.

Artifacts lacking the metric are never selected. `None` is a real, valid answer: the caller
decides between skipping the trip and falling back to a baseline.

This replaces two things that existed before the contract: `registry.get_best_model`'s
metric-only ranking (no agency awareness — the reason bUCR needed a hard pin in the first
place) and the route/global if-else chain that used to live in `serve.estimator` directly.
Centralizing it means training reports ("which artifact would be chosen") and serving ask
the identical question through the identical function — a user story from the modeler and
operator perspectives that this design directly answers: "get per-route/per-trip model
choice automatically with a manual pin available."

Full shape: [Contracts §7b](../CONTRACTS.md#7b-selection-which-artifact-serves-a-trip).

## Two things called "serving" {#two-things-called-serving}

The plan is explicit that two different things get called serving, and documentation must
not blur them:

- **The prediction library** — load a model, compute its inputs, produce stop time updates —
  is load-bearing regardless of who's asking. `databus` (SIMOVI's production backend) calls
  exactly this, and `stoptimes val` scores through the same seam. This is the part that
  matters even if you never run the standalone publisher below.
- **The standalone publisher** (`stoptimes serve`: poll on a timer, predict, publish a feed)
  is what an operator who already owns a feed builder — like databus — does *not* need,
  because they own their own feed. It exists because an agency *without* a databus would
  otherwise be left with a trained model file and nowhere to put it, and because the loop
  around the prediction code is thin enough that shipping it costs little. It is the
  straightforward path for a small agency, and the way to watch your own model work end to
  end — it is **not** a solution to the operations problem of running a public realtime feed
  (uptime, scaling, multi-region delivery). Anyone with that infrastructure already calls the
  library from their own publisher, the way databus does.

See [Serve predictions](../guides/serve-predictions.md) for what this means in practice, and
[the SIMOVI case study](../case-study/simovi.md) for how databus actually uses the library
without the publisher.
