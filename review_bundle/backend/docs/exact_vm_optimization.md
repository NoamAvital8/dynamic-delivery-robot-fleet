# Exact VM execution optimization — 11 October 2026

## What changes and what does not

The ten-policy, six-new-city campaign still contains 300 evaluations. Maps,
scenarios, seeds, trained models, candidate counts, loss functions, charging,
reservations, relocation utilities, event ordering and tie-breaking are unchanged.
All original `scripts/` and `src/` files remain byte-for-byte unchanged. The
accelerators and continuation controller live under `operations/`.

Execution improvements:

- Charger-to-node float64 distances and int32 predecessor tables are computed
  once per exact CSR/node/station ordering and SciPy version. Independent source
  rows use up to four preprocessing processes. Later simulations reuse read-only
  memory-mapped files, so the OS can share pages instead of allocating another
  private dense table for every simulator.
- Exact charger queries and station predecessor paths use bounded caches, with
  no rounded cutoff or battery keys. Returned dictionaries are copies, preserving
  mutable public API behavior. Each cache has an approximately 64 MiB allowance.
- The original compiled distance-row cache expands from eight rows to a maximum
  of 256 rows, capped at approximately 512 MiB. It returns the same float64 rows;
  only Dijkstra recomputation/hit diagnostics can change.
- The scheduler permits up to 24 concurrent simulations, subject to a maximum
  of 60 main/preprocessing/idle compute processes, a 16 GiB memory reserve, and
  conservative allowances for preprocessing and per-city private allocations.
  It can select another eligible small job when the next large job cannot fit.
  Numerical-library internal threads remain capped at one. Four original idle
  workers and all MLE settings remain unchanged.

This is not parallel route-candidate selection, a different policy, or a claim
of a measured whole-campaign speedup. Existing route-search logic is preserved.
Cache bounds/memory admission are estimates, not a kernel-enforced RAM limit;
memory pressure must continue to be monitored.

## Correctness and provenance

The shared cache includes all CSR inputs, node/station order, directedness, and
SciPy version in its identity. Builds are serialized per cache key, workers write
disjoint rows, and a ready manifest is published only after all rows finish.
SHA-256 checksums and shape/dtype checks validate every load; partial builds are
not accepted. The original station-pair oracle seeding order is preserved.

Tests compare shared distances/predecessors bit-for-bit, cutoff precision,
parallel edges, path ties, read-only mappings, cache eviction, returned-dict
mutation, corrupted cache rejection and repeated cache reuse. End-to-end tests
run all ten original and optimized policies on an independent synthetic scenario
and compare all non-performance/non-file-path result fields plus complete queue
audit records. Windows tests skip Linux-only mmap/locking checks; the full suite
is run on the VM's actual Python/SciPy runtime.

Scientific input signatures remain those of the original policy source and
commands. Optimized evaluations also save `result.execution.json` and stamp
execution code digests/actual wrapper commands explicitly. Consequently old and
new compatible policy-quality results can be retained, while runtime comparisons
must acknowledge the different execution versions and concurrent workloads.

## Continuation and recovery

`handoff_exact_controller.py` verifies the scheduler PID, start ticks and argv,
pauses only the scheduler while capturing every live simulation, saves SHA-256
hashes of completed outputs, and replaces only that scheduler PID. It never
signals simulator process groups. The new controller adopts those existing
simulators from their current in-memory state; only pending evaluations launch
the optimized wrapper. Completed results are not rewritten or rerun.

The current simulators **do not have event-state checkpoints**. A healthy live
job therefore stays on its original loaded code until it finishes. After a VM
crash, compatible completed evaluations can be reused; unfinished evaluations
must restart at their scenario beginning. Do not imply progress JSON is a full
checkpoint, inject new code into live interpreters, or blindly launch a second
controller. If a controller fails but children survive, capture and verify all
live identities and actual commands before making a fresh adoption record.
The first handoff helper deliberately rejects a second handoff with the same
record, rather than silently overwriting recovery evidence.

## VM paths and commands

Checkout:
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-new-city-benchmarks`

Campaign:
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies`

New scheduler log: `stages/benchmarks_exact_optimized.log`.
Shared tables: `shared_exact_index/`.
Controller identity: latest `launch.json`; initial identities/results:
`optimization_handoff.json`; prior launch: `launch.before_exact_optimization.json`.
`campaign_status.json` reports running/pending, worker budget and available RAM.
The original manifests and all older cohort results/models are preserved.

Read-only live summary from PowerShell remains `Show-FleetSummary` after loading
`operations/Show-FleetSummary.ps1`. On the VM:

```bash
tail -f /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies/stages/benchmarks_exact_optimized.log
```

Controller parameters are `--max-simulators`, `--max-workers`,
`--preprocessing-workers` and `--memory-reserve-gib`. Do not rerun a controller
command while an existing controller or unadopted children are alive.

## ETA limitations

The previous 10–20-day estimate was superseded by the observed multi-day large
map jobs. A provisional 4–8 weeks (possibly longer) remains a planning estimate,
not a measured optimized ETA. Sharing/caching and better admission can accelerate
pending work, but the preserved slow Beijing/Sydney/Moscow jobs continue on their
original code. Refine the estimate using completed optimized large-map runs and
their full delivery-drain time, not only fast New Delhi/Paris results or percent
of completed evaluations.
