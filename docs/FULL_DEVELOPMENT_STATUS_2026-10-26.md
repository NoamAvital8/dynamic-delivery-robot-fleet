# Full switching development status — 26 October 2026

Snapshot at **11:27 Israel time** (`09:27 UTC`). This is a saved observation,
not a live dashboard. Final-test generation/evaluation remains unopened.

## Training complete; development validation running

| Model | Completed training episodes | Final fit states |
|---|---:|---:|
| XGBoost | 800 / 800 | 40,743 |
| NN | 800 / 800 | 40,852 |

Both post-training model payloads were saved and their content hashes verified.
They remain fixed during the current development comparison. The campaign's
final model-freeze receipt is created only after that comparison completes.

All **1,600 training runs** have finished. **109 of 2,400 matched development
validation runs** were complete at this snapshot. Participants are the two
selectors and all ten fixed benchmarks on the same 200 development inputs.
Incomplete validation is not a basis for reporting a winning model.

## Remaining time and resource use

The remaining work is development validation and report generation. Estimate
**about 1–2 additional hours**, with **up to 3 hours** allowed for hard-city
tails. This is a planning estimate, not a guarantee.

A sample of 8,211 labeled states from 160 completed training actor episodes,
balanced across all eight cities and five profile positions, gives about 26.7
worker-hours of raw-window validation proxy work. At 67 processes this is 0.40
hours before trajectory corrections; applying the pilot's 3.1 calibration gives
1.23 hours. Fixed-policy trajectories, startup and expensive Manhattan cases
can differ from teacher branches, hence the wider practical band. The first
109 actual validation completions took about 4.6 minutes, with unfinished hard
jobs still active; extrapolating that count alone would understate uncertainty.

During the measured five-second interval:

- Execution admission allowed **up to 67 episode processes** on 72 logical CPUs.
- The campaign process tree consumed about **62.6 CPU-core equivalents** and
  the VM showed **93.3% CPU utilization**.
- I/O wait was about **0.008%**.
- Available RAM was **135.9 GiB**, above the retained **64-GiB reserve**.
- Four-thread fitting is complete; fixed validation now consumes the parallel
  episode capacity. Existing memory admission and owned-process cleanup remain.

Low utilization during the final two training episodes was a short workload
tail. The validation phase subsequently ramped back up without starting any
additional training scenarios or opening final test.

## Overnight interruption and recovery

The original controller stopped on **25 October at 23:48 Israel time** after
it mishandled slow process cleanup following a successful saved XGBoost fit.
It waited one second for child exit and classified a still-cleaning-up process
as a fit failure. Collection was checkpointed and owned workers were stopped.

The campaign resumed on **26 October at 09:16 Israel time**, preserving the
completed outcomes, saved action intents and replay. Approximately **9.5 hours
of downtime** account for the large difference between wall-clock elapsed time
and active training time. The initial estimate did not include that interruption.

Recovery uses a separately hashed execution-only override, waiting for a child
to exit after returning a successful result and still rejecting genuine failures.
The preserved simulator, projection, feature, learner and input logic and their
campaign/science/learner identities remain unchanged. The saved recovery notes
record focused process-control regression checks. The recovery service was
active with zero subsequent systemd restarts at this inspection.

The active recovery scripts and post-training model payloads have now been
archived for review under
`review_bundle/evidence/full_development_status_20261026/`. The original sealed
execution source is retained; the override is explicitly identified rather
than silently presented as the original implementation.

## Four-city holdout audit

The audit checked all **1,000 development input files** against their hashes,
the 800/200 training/validation split, and both final training model payloads.

**Cambridge UK, Ghent, Salamanca and Lucca** were absent from development inputs
and both models' training-city metadata. There was no `test_opened.json`, no
test manifest and no test actor output directory. Completed validation results
were checked for valid development input identity and complete order delivery.

These observations confirm the holdout at this timestamp. Current status can
be read from the existing campaign's `full_run_status.json` and recovery log.
`training_status.json` can retain the last collection snapshot during validation;
its old running count is not the current validation-worker count. Do not start
a duplicate controller or open final test to check progress.
