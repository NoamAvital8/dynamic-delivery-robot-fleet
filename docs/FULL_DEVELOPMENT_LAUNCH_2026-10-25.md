# Full small-city switching development training launched

Both XGBoost and the small NN were launched on **25 October 2026 at 22:45
Israel time** (`2026-10-25T20:45:05Z`). This document records the launch; use VM
status files for live progress. Final-test generation and evaluation remain
unauthorized and unopened.

The verified simulator, collector, projector and learner code are unchanged
from commit `7846354e189dc2e7b03e0404d6f76e8c1bc836fe`. Later documentation and
launch metadata describe the running campaign. The exact verified build
manifest and verification receipt from before launch are preserved under
`review_bundle/evidence/full_development_launch_20261025/`.

## Campaign

- Eight development cities: Tel Aviv, Haifa, Manhattan, New Delhi, Delft,
  Leuven, Oxford and Palo Alto.
- **800 fresh training inputs**: 100 per city. Both model families collect
  their own actor-selected trajectories, giving **1,600 training episodes**.
- **200 separate development validation inputs**: 25 per city. After final
  fitting, both frozen selectors and all ten benchmarks run on those inputs,
  giving **2,400 matched validation outcomes**.
- All 1,000 input seeds and workload identities are unique. Input files were
  verified against the manifest. The full compressed input archive is in Git.
- Five rotating profiles: ordinary, surge, spatial shift, priority shift and
  charging pressure. Arrival horizon remains 12 hours; decisions remain 15
  minutes. Every admitted order must finish before an outcome is accepted.
- **Cambridge UK, Ghent, Salamanca and Lucca remain held out.** There is no
  `prepare-test` or `evaluate` step in the supervisor.

The proposed full-run estimate remains **4–6 hours centrally / 6–12 hours as a
planning budget**, including development validation. This is a pilot-based
estimate, not a guarantee. Final-test work is additional and will require a
separate instruction.

## Resources and observed startup

The VM exposes 72 logical CPUs. The campaign admits up to **64 fresh episode
processes**, each with one numerical thread, and **one asynchronous four-thread
fitter**. Eight CPU slots are reserved for fitting, coordination and other VM
work. Existing unrelated jobs were preserved.

The first measured steady sample had **64 simulation workers**, about **63.5
CPU-core equivalents** consumed by the training process tree, **94.1% VM CPU
utilization**, 0% observed steal and negligible I/O wait. RAM remained well above
the 64-GiB available-memory reserve. These are startup measurements, not a
promise that the same utilization will hold during every phase or tail.

A subsequent startup snapshot confirmed **73 completed training episodes**,
with both models publishing their first real updates: XGBoost fitted on **2,021
states**, NN on **2,062 states**. The numerical-library runtime reports confirmed
four threads inside both fitting processes. These are timestamped startup
counts; collection continues beyond them.

Admission retains the verified 1.5-GiB prospective episode budget, 8-GiB trainer
budget and 64-GiB available-memory reserve, including pending startup allocations.
The existing controller checkpoints and stops owned workers if that reserve is
crossed. Both families refit at approximately 2,000 additional labeled states;
new actor versions are adopted only at decision boundaries.

## Location, supervision and progress

VM output root:

```text
/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_v4_pilot_fixes_20261025
```

`operations/run_switching_development.py` is the explicit development-only
supervisor. It verifies sealed source bytes, runs `prepare --execute`, verifies
the 800/200 development split, then runs `train --execute`, which includes
matched development validation and final model freezing. It has no final-test
command. It is detached from SSH and continues when that connection closes.

Launch supervisor PID: **1789179**; training controller PID at startup:
**1789396**. PIDs are historical identifiers; verify process identity before
acting on one. Do not start a second controller in this output directory.

Useful VM files:

- `FULL_RUN_LAUNCH.json`: exact command, hashes, scope and resource settings.
- `full_run_status.json`: supervisor stage, counts, controller state and RAM.
- `training_status.json`: collection progress, active workers, replay coverage
  and fitted-state counts for both families.
- `FULL_DEVELOPMENT_DATA_VERIFIED.json`: the completed 1,000-input roster check.
- `logs/supervisor.log`, `logs/prepare.log`, `logs/train.log`: durable output.
- `last_shutdown.json`: controller failure/stop details if a failure occurs.
- `FULL_DEVELOPMENT_COMPLETE.json`, `training_summary.json`,
  `validation_summary.json`, `frozen_models.json`: expected completion evidence.

A read-only progress check can read these JSON/log files. Launch authorization
does not authorize terminating unrelated VM jobs or opening the final test.

## Git evidence and limitations

`review_bundle/evidence/full_development_launch_20261025/` contains the launch
receipt, exact config, dataset manifest, campaign identity, measured startup
resources, supervisor snapshot and all 1,000 development scenario documents.
`DEVELOPMENT_SCENARIOS.jsonl.gz` preserves each scenario's exact UTF-8 JSON text
and SHA-256. Live replay, evolving actor versions and bulky runtime checkpoints
remain on the VM while the campaign is running; the launch snapshot does not
claim that training has completed.

The pilot's scientific limitation remains: projected completion includes known
charging queues but omits future fleet value. Its controlled action rankings
can differ from completed remaining-episode rankings. The full development
comparison will assess both selectors against all ten fixed benchmarks; no
performance improvement is guaranteed and no final-test result exists yet.
