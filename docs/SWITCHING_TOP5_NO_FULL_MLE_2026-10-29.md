# Five-benchmark switching campaign — 29 October 2026

Repository: https://github.com/NoamAvital8/dynamic-delivery-robot-fleet

Branch: `codex/small-city-switching-review-20261025`.

The owner replaced the nine-benchmark ablation with the best five eligible
benchmarks from the completed original ten-action campaign. Full MLE remains
excluded. Only the nine-action service was stopped; its artifacts and all
completed original campaign artifacts are preserved. The replacement uses a
new output directory, fresh model versions and fresh model-led trajectories.
No nine-action model weights, replay labels or saved actor states are reused.

## Selection from development validation

Rank fixed benchmarks by the original primary metric: equal-city mean final
episode loss per robot, lower is better, across 200 development-validation
inputs in the eight development cities. Remove `full_mle_reservation` before
selecting the best five. Final-test outcomes are not used for this selection.

| Eligible rank | Policy | Mean loss per robot |
|---|---|---:|
| 1 | `full_coordinated_idle` | 230.788588 |
| 2 | `queue_mle_reservation` | 231.541625 |
| 3 | `full_queue_aware` | 233.124642 |
| 4 | `full_no_nn` | 233.331046 |
| 5 | `full` | 237.414991 |

The excluded lower-ranking eligible policies are Reactive Insertion,
Full without Idle, Paper SA Adapted and Myopic AB. Full MLE ranked first in
the original portfolio but is excluded by the owner's earlier instruction.

Selection evidence:
`review_bundle/evidence/switching_top5_no_full_mle_20261029/POLICY_SELECTION.json`.
The source report is
`review_bundle/evidence/final_test_launch_20261026/validation_summary.json`
(the file contains development validation, despite its enclosing launch-folder
name). Its SHA-256 is
`3619eb2243303eecc254b22bcef92076a8ebc1b53341aa8dd8c690afc3b2daf0`.
The policy-selection regression checks its ranking and validation-only split.

Model-output order retains the canonical ordering of the original portfolio:
`full_no_nn`, `full`, `full_queue_aware`, `full_coordinated_idle`,
`queue_mle_reservation`. Ranking and output-index order are distinct.

## Training and held-out scope

Both XGBoost and the 64/32-unit NN train from scratch with five cost outputs.
Each actor chooses before any teacher scores are available. All five
benchmarks branch from the same snapshot for the next 15 minutes. Scores
retain realized loss plus projected completion loss, including shared FIFO
charging queues. Only the actor-selected physical boundary becomes the next
state. Each learner collects its own trajectories and updates after 2,000
new committed training states, as in the original campaign.

The initial bootstrap is now `full_coordinated_idle`, the best eligible
validation benchmark, because Reactive Insertion is outside the five choices.
Runtime initialization may represent an initial internal policy before the
first decision; it is not a selectable sixth benchmark or a simulated branch.
Observable feature names, loss, 12-hour arrival horizon, drain-to-completion
procedure, pinned simulator and fixed FCNN dependencies are unchanged.

The eight development cities retain 100 training and 25 validation inputs
each, with identical seeds and scenario generation. There are 800 training
inputs, 1,600 independent training episodes across both models, and roughly
400,000 training teacher branches if episodes average 50 decisions. Development
validation evaluates two frozen selectors plus five fixed policies on 200
inputs: 1,400 outcomes. This service never opens final test.

Cambridge UK, Ghent, Salamanca and Lucca remain excluded from this replacement
campaign's generation, collection, fitting and validation. Prior separately
published test results belong to the original campaign. No new final-test
outcomes are used to rank the five policies.

## Execution and relevant files

- Config: `configs/small_city_switching_top5_no_full_mle_20261029.json`.
- Preflight and explicit service launch:
  `operations/launch_switching_top5_no_full_mle.py` (`--execute` required).
- Prepare/train/validation supervisor:
  `operations/run_switching_top5_no_full_mle.py`.
- Core: `src/fleet_switching/config.py`, `models.py`, `collection.py`,
  `provenance.py`, `workflow.py`, `reporting.py`, `projection.py`, `backend.py`,
  `backend_lifecycle.py`, `replay.py`, `data.py`, `execution.py`.
- Coverage: `tests/test_switching_top5.py`, `test_switching_portfolio.py`,
  `test_switching_execution_cleanup.py`, and existing switching tests.
- Current source manifest: `review_bundle/CURRENT_BUILD_SOURCES.json`.
- Launch evidence: `review_bundle/evidence/switching_top5_no_full_mle_20261029/`.

New VM root:
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_top5_no_full_mle_v1_20261029`.

New service: `robot-switching-top5-no-full-mle-20261029.service`.
Superseded stopped service: `robot-switching-no-full-mle-20261029.service`.

The resource ceiling remains 67 one-thread episode workers and a four-thread
fitter on 72 logical CPUs, subject to a 64-GiB available-RAM reserve, 8-GiB
trainer allowance and 1.25-GiB worker allowance. The service is independent
of SSH, restarts on failure after 30 seconds, and has no elapsed runtime
limit. Successful process results survive slow cleanup. Other VM jobs are
not stopped or changed.

Five-, nine- and ten-action artifacts have incompatible scientific identities
and schemas. Older completed/stopped VM source trees remain unchanged. The
compressed development scenario inputs already in Git can be paired by SHA;
no old labels are reused. Live replay, checkpoints, logs and later fitted
model versions remain on the VM until explicitly exported. Initial launch
evidence and model snapshots are dated records, not live status.
