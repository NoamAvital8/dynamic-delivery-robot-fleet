# Frozen switching models: final test launched

The owner explicitly authorized opening the final test on 26 October 2026.
The VM service started at **12:57 Israel time** (`10:57 UTC`). This document
records that launch; `test_run_status.json` on the VM supplies live progress.

## Fixed protocol

- Final-test cities: **Cambridge UK, Ghent, Salamanca and Lucca**.
- **50 fresh scenarios per city**, 200 inputs total, with the same 12-hour
  arrival horizon, five profile types and 15-minute selector decisions.
- **Both frozen selectors and all ten fixed policies** use each input:
  **2,400 matched complete outcomes**.
- The declared primary metric remains equal-city mean loss per robot.
- The deployable fixed baseline remains **Full MLE Reservation**, selected
  from completed development validation before final-test generation.
- There is no optimizer/model-fitting path in the supervisor. Test inputs and
  outcomes are not appended to training replay and do not change model choice.

The final models use 40,743 XGBoost states and 40,852 NN states. Their payloads
are archived under `review_bundle/evidence/full_development_status_20261026/models/`.
The exact frozen receipt is included with the launch evidence:

```text
freeze_id: f59e0d0d82873acbec1293f0ef189f09e126d99519f6c8dfadf29b33a814b7d5
frozen receipt SHA-256: df928637a11de0528d2a40414f77f3f41112ce18253d309db130ee9531bf33b9
```

All 200 test inputs have unique seeds and workload identities, with no overlap
with the 1,000 development inputs. Opening `test_opened.json` activates the
existing development barrier: this campaign must not resume training or tune
models after observing final-test results.

## Execution and startup checks

Service: `robot-switching-final-test-20261026.service`.

The explicit entry point is `operations/run_switching_final_test.py`. It
verifies the frozen model/benchmark payloads and sealed source files, prepares
only the four-city test split, and calls the existing `evaluate` workflow. The
previously tested execution-only cleanup fix is loaded explicitly; simulator,
projection, features and learner code remain unchanged.

Admission allows **up to 67 one-thread episode workers**, retaining the 64-GiB
available-memory reserve, 8-GiB trainer allowance and 1.5-GiB prospective worker
budget. No fitter is launched for final test. The service restarts on failure
after 30 seconds, resumes saved progress, has no runtime cutoff, and survives
SSH disconnection. Unrelated old jobs remain in their existing state.

At **13:00 Israel time**, the saved startup audit observed **181/2,400 completed
outcomes**, 66 active episode workers, **93.4% VM CPU utilization** and about
**150 GiB available RAM**. These are timestamped startup measurements.

The frozen receipt, both selected model pointers, all ten fixed-policy payloads,
completed development reports, development manifest and full training replay
were hashed before opening test. A subsequent audit verified those bytes had
not changed. The supervisor checks these protected artifacts again on completion.

Provisional planning estimate: **about 1–2 hours** for this test comparison.
Actual time depends on city, workload and benchmark-policy tails. The startup
snapshot is not a performance result or a completion claim.

## VM files and Git evidence

Campaign root:

```text
/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_v4_pilot_fixes_20261025
```

- `FULL_TEST_LAUNCH.json`: exact command, service, scope and freeze hashes.
- `test_run_status.json`: current phase, completed outcomes, RAM and limit.
- `final_test/evaluation.log`: delivery-runtime and exception output.
- `final_test/TEST_DATA_VERIFIED.json`: 200-input/four-city separation receipt.
- `final_test/PROTECTED_DEVELOPMENT_FILES.json`: development artifact hashes.
- `final_test/STARTUP_AUDIT.json`: measured CPU, worker and integrity snapshot.
- `final_test/COMPLETE.json`, `test_summary.json`: completion/report artifacts
  once all 2,400 matched outcomes succeed.

`full_run_status.json` describes the earlier development campaign. Its old
`final_test_opened:false` belongs to that dated completion snapshot; use the
separate final-test files for the newly authorized evaluation.

Git evidence is under `review_bundle/evidence/final_test_launch_20261026/`:
the launch and opening receipts, model freeze, exact fixed-policy payloads,
completed development results, all 200 test documents and supervisor snapshot.
Large evolving test checkpoints/logs remain on the VM. No new scientific source
or hyperparameter was changed after the model freeze.
