# Nine-action campaign launch evidence — 29 October 2026

Full development training started at **12:56:30 Israel time (10:56:30 UTC)**.
Scientific source commit: `7ec43896ceca8c9b8a19a43d500d444069d3027f` on
`codex/small-city-switching-review-20261025`.

Both XGBoost and NN are running in the separate VM service
`robot-switching-no-full-mle-20261029.service`, under
`runs/icaps_small_city_switching_no_full_mle_v1_20261029`.
The candidate portfolio contains nine benchmarks, retains Queue MLE, and
excludes Full MLE. Decisions remain 15 minutes. There are 1,600 training
actor episodes and 2,200 development-validation outcomes. Final test is not
opened or evaluated by this service.

`LAUNCH_AUDIT.json` is a dated snapshot, not live status. At 11:02:29 UTC,
74 training episodes were complete; all 5,020 replay rows contained exactly
nine action costs, belonged to the eight development cities, and excluded
the removed action. Both learners had published and were being used:
XGBoost at 2,015 states, NN at 2,097 states. CPU use measured 92.36% of all
72 logical CPUs with 67 episode workers. Available RAM was 134.09 GiB,
above the 64-GiB reserve. The service had zero restarts.

The preliminary total runtime expectation is 4–6 hours including development
validation, based on the earlier full campaign and pilot. Fewer teacher
branches reduce work, but new trajectories can change runtime; this is not
a measured completion prediction for the new campaign.

- `BUILD_VERIFICATION.json` and `VM_TEST_OUTPUT.txt`: 83 scoped checks passed
  locally and on the VM, including tiny real nine-output fits and legacy
  ten-output compatibility.
- `LAUNCH_PREFLIGHT.json`, `LAUNCH_RECEIPT.json`, `campaign.json`,
  `execution_settings.json`: exact run identity, settings and launch scope.
- `FULL_DEVELOPMENT_DATA_VERIFIED.json`, `development_manifest.json.gz`:
  all 1,000 development inputs, with no held-out city. Scenario bytes, seeds,
  graph hashes, workload hashes and fleets match the original campaign.
  The exact scenario payloads are already in Git at
  `review_bundle/evidence/full_development_launch_20261025/DEVELOPMENT_SCENARIOS.jsonl.gz`.
  That archive supplies scenario inputs only; no teacher labels are reused.
- `INITIAL_MODEL_EXPORT.json`, `*_initial_model.json.gz`: exact first fitted
  models, compressed with their original payload hashes. These are preliminary
  artifacts; live model versions continue to update on the VM.
- `runtime_plan.json`: inherited conservative arithmetic for planning, not a
  fresh runtime measurement; the launch receipt records the provisional estimate.
- `audit_live_campaign.py`: read-only audit used on the VM. It never trains,
  simulates, opens test inputs or changes the experiment.
- `PREVIOUS_BUILD_SOURCES.json`: preserved prior ten-action source manifest.

The audit also verifies that both original completed selector model payloads
are unchanged. Live SQLite replay, checkpoints, full logs and subsequent model
versions remain VM-only. The supervised service survives SSH disconnects and
restarts failed work from durable checkpoints. No unrelated VM jobs were changed.
