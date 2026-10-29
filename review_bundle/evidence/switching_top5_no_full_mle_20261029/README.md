# Five-benchmark replacement campaign — 29 October 2026

The nine-benchmark service was stopped at the owner's request and remains
inactive. Its files are preserved. Fresh XGBoost and NN training started at
**13:11:50 Israel time (11:11:50 UTC)**, using only the five strongest eligible
benchmarks from the original ten-action campaign's development validation.
Full MLE is excluded before ranking; final-test outcomes are not used.

| Rank | Policy | Equal-city mean loss per robot (lower is better) |
|---|---|---:|
| 1 | `full_coordinated_idle` | 230.788588 |
| 2 | `queue_mle_reservation` | 231.541625 |
| 3 | `full_queue_aware` | 233.124642 |
| 4 | `full_no_nn` | 233.331046 |
| 5 | `full` | 237.414991 |

Source code commit: `dc2f75788210fda4ce9887bba1c68a7e97d9c822` on
`codex/small-city-switching-review-20261025`.
Current VM service: `robot-switching-top5-no-full-mle-20261029.service`.
Output directory: `runs/icaps_small_city_switching_top5_no_full_mle_v1_20261029`.

Decisions remain 15 minutes. The first bootstrap choice is Full Coordinated
Idle because Reactive Insertion is outside the selected five. Both models
collect their own new trajectories and refit every 2,000 committed states.
There are 800 training inputs, 1,600 independent training actor episodes and
1,400 development-validation outcomes. The four final-test cities stay held
out; this service does not prepare or evaluate final test.

`LAUNCH_AUDIT.json` is a dated snapshot at 2026-10-29T11:16:26.712298+00:00, not live status.
It verified 4,897 fresh rows with exactly
five costs each, across development cities only, and 70
completed training episodes. First fitted model states: XGBoost 2,017,
NN 2,087; both exported five outputs and were already being used by actors.
CPU use measured 92.27% of the 72 logical CPUs
with 67 workers. Available RAM was 134.65
GiB, above the 64-GiB reserve. The service had zero restarts.

- `POLICY_SELECTION.json`: exact ranking, validation-report SHA, selected set,
  output ordering, and bootstrap. The source report is already in Git at
  `review_bundle/evidence/final_test_launch_20261026/validation_summary.json`;
  its contents are development validation despite its parent directory name.
- `BUILD_VERIFICATION.json`, `VM_TEST_OUTPUT.txt`: 90 scoped checks passed
  locally and on the VM, including actual five-output fits, actor-led
  trajectory fixtures, wrong-portfolio rejection, and legacy compatibility.
- `LAUNCH_RECEIPT.json`, `LAUNCH_PREFLIGHT.json`, `campaign.json`,
  `execution_settings.json`: run identity and explicit scope.
- `SUPERSEDED_NINE_ACTION_STATUS.json`: stopped old service and partial
  training status; no states, replay or model weights are reused.
- `STARTUP_AUDIT.json`, `LAUNCH_AUDIT.json`, `audit_live_campaign.py`: initial
  and post-fit read-only audits. No simulations or training are launched by
  the audit script.
- `FULL_DEVELOPMENT_DATA_VERIFIED.json`, `development_manifest.json.gz`:
  all 1,000 development inputs match original seeds, graph/scenario/workload
  hashes and fleets. Exact scenario bytes are already archived in Git at
  `review_bundle/evidence/full_development_launch_20261025/DEVELOPMENT_SCENARIOS.jsonl.gz`.
- `INITIAL_MODEL_EXPORT.json`, `*_initial_model.json.gz`: exact first fitted
  models and uncompressed SHA-256 values; these are preliminary, not final.
- `PREVIOUS_NINE_ACTION_BUILD_SOURCES.json`: archived prior source manifest.
- `runtime_plan.json`: inherited conservative planning arithmetic, not a
  measured completion forecast for this new five-policy campaign.

The original completed ten-action models are unchanged. Live replay,
checkpoints, subsequent fitted models and full logs remain on the VM. The
service survives SSH disconnection and restarts failed work from durable
state. Unrelated VM jobs were not changed.
