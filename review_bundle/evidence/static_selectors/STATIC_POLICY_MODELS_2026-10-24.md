# Start-only policy selector comparison

Authorized on 2026-10-24. The threshold pilots end at 18:00 Asia/Jerusalem;
actual project fitting starts only after that time and after both light-pilot
units have stopped. Original long benchmark and meta campaigns remain frozen.

## Inputs available at deployment

Each model chooses once at time zero and commits to that policy until every
delivery finishes. Inputs are public graph geometry, charger/cluster summaries,
configured fleet size and type shares, known full initial batteries and idle
state, known run horizon, and historical calibration demand priors already
supplied to the dispatch policies. The feature API never accepts a scenario
order list, future arrivals/importance, scenario seed, future queue outcomes,
actual demand of that scenario, completed-run loss, or a city-name indicator.
The candidate policy identity is an input to its outcome scorer.

## Training data and versions

All 310 valid completed fixed-policy results in the five compatible earlier-city
benchmark cohorts are used: precision v2, readiness v3, queue v4, coordination v5,
and MLE v6. These are only 25 independent scenario workloads in Tel Aviv, Haifa,
Manhattan, NYC, and Barcelona and only five distinct start feature profiles.
Hundreds of policy outcomes are not hundreds of independent city contexts.

The older benchmark split names are historical. They are development data for
this new model; final test cities are Beijing, Paris, New Delhi, Johannesburg,
Moscow, and Sydney, and none of their maps, scenario outcomes or features enters
fitting, normalization, architecture selection or validation.

Old eight-minute readiness versions have separate auxiliary action heads from
the current readiness versions. Uncertainty-policy historical results also have
auxiliary heads. They train the shared scorer, but deployment chooses among the
ten policies in the current comparison. Missing action labels are never zero
filled. Superseded v1 numeric-bug outputs, incomplete runs, dynamic-policy
trajectory returns and short-horizon rollout targets are not valid full-run
labels for a fixed policy and are excluded.

## Four frozen configurations

- FCNN: mean of three fixed-seed, 16-hidden-tanh-unit networks; L-BFGS, L2=10.
- XGBoost: 200 depth-two trees, learning rate 0.03, L2=10, one numerical thread.
- Decision tree: maximum depth three, at least ten observed outcome rows per leaf.
- Extra Trees: 128 trees, depth five, at least six observed outcome rows per leaf.

They use identical input features and target definitions. The target is complete
run loss per robot divided by a positive training-city scale; the final report
also retains raw loss. This is a declared equal-city fitting objective rather
than an assertion that normalized and raw aggregate loss have the same winner.

No hyperparameter search uses final test cities. Grouped validation holds out
whole workloads (all their policies/versions together). A separate leave-one-city
out check measures harder transfer. Scalers and target scales are fitted anew
inside each fold. Baselines are training-only best fixed policy and historical
city lookup. Missing validation labels remain explicitly missing. Configurations
are fixed, not chosen by their final test results.

## Test pilot

Freeze all four model artifacts first. Then publish all four policy choices for
each test city using only start features, before opening any test outcome.
Use the first two scenario IDs from each city's existing held-out suite:
4 models × 6 cities × 2 scenarios = 48 logical model evaluations.

Choosing a fixed policy produces exactly that policy's full-run simulation.
Reuse a completed, validated matching result, with its hash and source receipt.
Run only missing selected-policy outcomes in a new output folder. Deduplicate
identical choices across selectors. Never thaw a paused original service or
duplicate a preserved unfinished original worker; a matching paused worker is
reported for attention instead. New simulations have progress logs, not full
event-state checkpoints; completed outputs are reusable, interrupted new cases
restart only that unfinished case. Original campaigns remain untouched.

At most eight model-validation worker processes, each with one numerical thread.
New simulations use at most two concurrent slots, inline idle/MLE scoring and
exact shared routing caches. Maintain a 64 GiB available-memory reserve and
reserve 48 GiB per prospective new worker. The isolated service has a 96 GiB
cgroup ceiling. CPU-only XGBoost is installed only in this run's dependency
folder; original Python packages are unchanged.

## Locations and commands

VM output root:
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_static_policy_nn_v1_20261024`

Readable entry point: `operations/static_policy_nn_20261024.py`.
Service: `robot-static-policy-models-20261024.service`.
Log: `workflow.log`. Status: `status.json`.
Artifacts: `models/*.json`, `model_registry.json`, `training_report.json`,
`test_choices/<family>/<city>.json`, `evaluation_plan.json`, `test_summary.json`.

The safe resumable workflow command is the existing service's ordinary argv,
ending in `workflow --processes 8`. Do not launch a second controller while it
is active. Its exclusive lock rejects duplicates, and completed result hashes
are checked before reuse. `predict --model-kind fcnn --features <start.json>`
prints the chosen fixed policy; replace `fcnn` with another supported family.

Six correctness checks verify strict input schema, version separation, complete
finite labels, NN export equivalence, all four finite family predictions and
the before-18:00 fitting guard. This small test-city comparison is exploratory;
selecting a winner on it does not create a new untouched final test set.
