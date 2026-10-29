# Switching-model ablation without Full MLE — 29 October 2026

Repository: https://github.com/NoamAvital8/dynamic-delivery-robot-fleet

Branch: `codex/small-city-switching-review-20261025`.

The owner authorized fresh training of both XGBoost and the small NN, with the
same 15-minute switching protocol and `full_mle_reservation` removed from the
candidate portfolio. `queue_mle_reservation` remains available. This is a
separate campaign; the completed ten-action campaign and its artifacts are
preserved. Launch status and measured resource use are recorded separately in
`review_bundle/evidence/switching_no_full_mle_20261029/`.

## Protocol and scope

The nine choices, in model-output order, are:

1. `myopic_ab`
2. `reactive_insertion`
3. `paper_sa_adapted`
4. `full_no_nn`
5. `full_no_idle`
6. `full`
7. `full_queue_aware`
8. `full_coordinated_idle`
9. `queue_mle_reservation`

At each state the actor records its choice before the teacher runs. All nine
benchmarks then branch from the identical simulator snapshot for 15 minutes.
Labels retain realized loss plus projected completion loss for every admitted
unfinished order, including shared FIFO charging queues. Only the physical
boundary of the actor-selected branch becomes the next state. Projection
does not become the next state. Each model collects its own trajectories.

The bootstrap remains Reactive Insertion. Model updates still occur after
2,000 new committed training states, with the same learners, features, loss,
and episode weighting. XGBoost now fits nine regressors; the 64/32-unit NN has
nine output units. The original ten-action configuration remains supported.
Model loads and replay schemas reject mixing the two portfolios. Action
portfolios are included in science, learner and campaign identities.

This removes a selectable dispatch policy. Simulator internals, observable
feature names, Queue MLE, MLE estimator resets and MLE gates are unchanged.
Existing features for an inactive action can remain constant.

There are 100 training and 25 validation scenarios per development city:
Tel Aviv, Haifa, Manhattan, New Delhi, Delft, Leuven, Oxford and Palo Alto.
That is 800 training inputs and 1,600 independent actor episodes across both
families. Validation evaluates two frozen selectors and nine fixed policies
on 200 inputs, producing 2,200 outcomes. Scenarios retain 12 hours of arrivals
and drain admitted orders to completion. The original base seed, maps,
scenario generator and profile schedule are retained for a controlled
ablation; existing teacher labels and model weights are never reused.

Cambridge UK, Ghent, Salamanca and Lucca remain excluded from this campaign's
preparation, collection, fitting and validation. The launcher never prepares
or evaluates final test. The old campaign already has separately evaluated
test results; the new experiment is a subsequent research ablation and has
no new confirmatory test result yet.

## Files and execution

- Configuration: `configs/small_city_switching_no_full_mle_20261029.json`.
- Read-only preflight / explicit service submission:
  `operations/launch_switching_no_full_mle.py` (requires `--execute` to start).
- Supervised prepare/train/validation:
  `operations/run_switching_no_full_mle.py`.
- Core: `src/fleet_switching/{config,collection,models,provenance,workflow,reporting}.py`.
- Simulator and projection: `backend.py`, `backend_lifecycle.py`, `projection.py`.
- Immutable inputs and replay: `data.py`, `replay.py`.
- Process handling: `execution.py`; it incorporates the tested recovery used
  by the completed campaign, retains successful results while processes
  finish cleanup, and does not abort healthy fits on elapsed time. The
  historical `fit_timeout_seconds` config field is retained for compatibility
  and is not an enforced timer. Memory guards and real failure checks remain.
- Regression coverage: `tests/test_switching_portfolio.py`,
  `tests/test_switching_execution_cleanup.py` and the existing switching tests.

New VM directory:
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_no_full_mle_v1_20261029`.

The backend remains clean at `5e76f5384aec98e9d5a538b27dae205df301fa28`.
The launcher checks source hashes, the backend commit, development roster,
memory admission and an unopened test set before submitting the user service
`robot-switching-no-full-mle-20261029.service`. It runs independently of SSH,
restarts on failure after 30 seconds, and resumes durable campaign state.
There is no runtime limit or automatic test phase.

The resource ceiling is 67 single-thread episode workers, one four-thread
fitter and coordinator capacity on the 72-logical-CPU VM. Admission retains
64 GiB of available RAM plus 8 GiB for fitting. This campaign budgets 1.25
GiB per worker, above the previous pilot's 0.794-GiB observed peak; actual
available memory can reduce admission. Existing unrelated processes are
preserved.

Sources and the compressed prepared maps, fixed FCNN dependencies and
historical development scenarios are in Git. Live replay, checkpoints,
logs and generated new model versions reside on the VM. The new source
manifest is `review_bundle/CURRENT_BUILD_SOURCES.json`; the previous manifest
is archived with this campaign's evidence. Old source is also recoverable
from its original Git commits and unchanged VM directory. Dated old build
verification receipts describe the original campaign, not this new build.
