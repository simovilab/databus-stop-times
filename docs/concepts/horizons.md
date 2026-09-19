# Horizons

A prediction made 30 seconds before a vehicle reaches a stop and one made 8 minutes before
are different problems: near-term predictions can lean on the vehicle's current speed and
position; long-range ones can't, and schedule adherence dominates instead. Reporting one
blended error number across both hides which regime a model is actually good at.

`stoptimes` breaks error out by **horizon** — the time remaining until the predicted
arrival — everywhere accuracy is reported:

- **Model cards** (`stoptimes models card`) show a `scores_by_horizon` table (`0-60s`,
  `60-180s`, `180-360s`, ... — see a real one in the
  [quickstart](../quickstart.md#4-train-a-model)).
- **Validation reports** (`stoptimes val report`) show the same breakdown for any scored
  predictor, live or replayed.

## Why this is a modeler and skeptic user story, not an implementation detail

The [user stories](../PLAN.md#71-user-stories-the-acceptance-list) ask, from the modeler's
side, to "compare families over one dataset with errors broken out by prediction horizon,"
and from the skeptic's side, to "break error down by horizon, route and time of day." Both
exist because a single MAE number can hide a model that's excellent close-in and useless far
out (or vice versa) — exactly the failure mode a horizon breakdown is built to surface before
you commit to serving a model.

`uncertainty_s` on a [`StopTimePrediction`](../CONTRACTS.md#6-prediction-serving-output)
follows the horizon convention too: the recommendation in
[Contracts](../CONTRACTS.md#open-questions-for-sign-off) is that it means the model's held-out
mean absolute error, in seconds, for the prediction's own horizon bucket — not one blended
number for the whole model — so a consumer reading GTFS-RT's `uncertainty` field gets a
number that means something for *that* prediction's time-to-arrival.
