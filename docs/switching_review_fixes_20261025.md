# Review fixes: small-city switching selectors

This is the current implementation addendum after the two external reviews of
commit `2ffd002`. A separately authorized bounded pilot is now documented in
[the pilot report](SWITCHING_PILOT_RESULTS_2026-10-25.md); full training and
final-test evaluation remain unstarted. Training and evaluation require explicit
`--execute`. The original review inputs are archived under
`review_bundle/evidence/reviews/`.

## Correctness and operational changes

| Review issue | Implementation |
|---|---|
| Retained graphs and growing initialization wrappers | `backend_lifecycle.py` installs the legacy graph/CSR hooks once, removes the closing runtime's oracle/row-cache entries, and the adapter drops its runtime references. `execution.py` additionally gives every episode a fresh process. The preserved backend source remains unchanged. |
| Charger FIFO ties | `Step` and `Session` carry known physical event IDs. The adapter validates current movement/service/charge events against the saved heap. Handling/charging completions use priority zero; movement uses priority one. Travel is represented edge by edge using physical path lengths. New projected events start from the boundary event counter. |
| Stale prepared route legs | Cached legs must match current node and battery. Reconstruction uses the physical engine's next-distinct-stop lookahead when applicable. Independent stop, pickup/dropoff, capacity and released-order conservation checks reject invalid labels. |
| Incompatible campaign reuse | `provenance.py` covers backend source including `operations`, fixed weight bytes, generator/collector/projector/features, all new implementation source, actor learner code, scientific settings and package/Python/platform identities. Worker-count changes are recorded separately and do not change the campaign identity. Finished jobs are checked against current input hashes and job identity before reuse. |
| Row-ID-only fitting provenance | Every row has a canonical full-content hash. The append-only SQLite catalog rejects edits to an existing key; fits validate row contents and active-manifest membership. Model versions include the actual payload, learner identity and content-hashed replay. |
| First-fit thread budget | Each fit starts in a fresh process with numerical environment limits set before import. Native libraries load before entering the dynamic limiter; XGBoost also has explicit thread limits. Effective native-library settings are recorded. |
| Failure hangs | Workers check a cooperative stop flag between branches/decisions. The coordinator stops admission on failure, requests checkpoint-safe exit, and then terminates only its own unresponsive children after a configurable grace period. A shutdown receipt lists resumable jobs. Fits have a wall-clock bound and are cleaned up separately. |
| Checkpoint durability | JSON and snapshot writes flush/fsync before atomic replacement; POSIX parent directories are fsynced. Initial state is persisted before the first intent. The current and previous checkpoint/state pairs are retained. Recovery never substitutes a new actor action. Exact-byte recovery is required when rebuilding a missing committed boundary; a mismatch fails closed. |
| Final-test lifecycle | Preparation, training and evaluation share one phase lock. Immutable campaign/dataset/freeze/opening receipts bind inputs and models. Opening final test prevents further development in the same campaign. |

Event fidelity is conditional on the frozen completion rule: known committed
event order is preserved, and synthesized continuation events have a deterministic
counter. This does not turn the projector into a normal dispatcher with future
arrivals, route insertion, interruptible background charging or future fleet value.

Checkpoint changes improve the storage ordering contract; they are not a promise
to recover arbitrary disk corruption. On Windows the file is flushed but a POSIX
directory fsync is unavailable. Production cross-process snapshot reproduction,
filesystem power-loss behavior, and the corrected runtime/projector integration
still need separately authorized verification. In particular, a regenerated
boundary whose diagnostic timing changes its serialized bytes is rejected rather
than silently changing an immutable training row.

## Speed and observability

- Pending shortlisting on these undirected maps uses one pickup-rooted distance
  row to query all eligible robot tails. Directed graphs retain the forward-query
  fallback. Path/tie parity and actual speedup remain production pilot questions.
- Immutable JSON rows remain audit evidence; their producer appends them to a
  SQLite WAL catalog. The coordinator reads counts/watermarks and aggregated
  coverage/regret metrics instead of repeatedly reparsing every JSON file.
- One bounded asynchronous fitter reads an immutable catalog prefix. Whole-data
  refits still use all accumulated compatible rows; the coordinator keeps
  admitting jobs, watching memory and handling failures during fitting.
- NN inference caches numeric weight/bias arrays instead of rebuilding them at
  every decision. Training math is separately exposed for fixed-weight finite-
  difference gradient checks without optimizing a model.
- New observable features summarize remaining route stops/edges, active handling,
  concentrated queues and recent demand versus free-robot spatial alignment.
- Row diagnostics time restore/advance/snapshot and terminal projection separately.
  Status reports include city/profile/policy coverage, bootstrap exposure, teacher
  action gaps and local regret. These metrics do not establish episode superiority.

## Complete matched comparisons

`train` now follows actor-led collection and final fitting with **2,400 development
validation episodes**: 200 inputs times two selectors plus ten fixed policies.
The fixed baseline for deployment is selected on development data using
**equal-city mean loss per robot**, a declared metric. Both learned models remain
frozen for the final comparison.

`evaluate` now runs **2,400 final-test episodes**: 200 held-out inputs times those
same 12 participants. Every comparison matches complete input identity, graph,
scenario and fleet. Reports require all outcomes; missing, duplicate, non-finite
or mismatched results are rejected rather than silently excluded. They include
raw/per-city/equal-city scores, paired differences against each fixed policy,
hindsight best fixed per scenario, wins/ties/losses, and stratified episode
bootstrap intervals. Those intervals are conditional on the observed cities;
four cities do not establish universal geographic generalization.

Outcome receipts include on-time/late totals, policy occupancy, post-initial
switches and queue waits. Queue waits use the backend's combined delivery and
background metric; they must not be described as mission-only waits. Finish time
is explicitly the last decision boundary, not the exact final dropoff timestamp.

## Resources, compatibility and estimate

Defaults remain up to 64 one-thread episode processes on 72 logical CPUs, four
fit threads and a 64-GiB available-memory reserve. Admission additionally reserves
8 GiB for fitting and accounts for submitted workers whose allocations are not
yet visible. Crossing the reserve triggers checkpoint-safe shutdown, with bounded
termination for unresponsive owned processes. This is monitoring/admission, not a
hard operating-system memory limit or protection against unrelated VM allocations.

Label version: **`projected_completion_fifo_v2_event_order`**. Existing v1 replay
and checkpoints are intentionally incompatible. The default VM output is the new
`runs/icaps_small_city_switching_v4_pilot_fixes_20261025` directory. Old builds,
campaigns, maps and weights remain preserved.

The plan now includes approximately 800,000 training branches plus 120,000
validation windows. Reusing the unmeasured 0.7–2.5-second planning band gives
approximately **7–24 elapsed hours**, including an assumed 2–6-hour overhead.
Budget **24–48 hours for the first full development campaign** until the corrected
edge-level scorer, fresh-process startup, replay, fits and memory are measured.
Final-test comparison is additional. These are conditional estimates, not timings.

## Verification and remaining work

Current scoped test results and build hashes are in
`review_bundle/CURRENT_BUILD_VERIFICATION.json` and `CURRENT_BUILD_SOURCES.json`.
Tests use pure projected itineraries, backend-shaped objects, stub event
transitions, child-process control stubs, content-tampering fixtures, a fixed
weight gradient check and a stubbed publisher. No actual optimizer, scenario
generator or production event engine is invoked by this suite.

A separately authorized production pilot now covers all eight development
cities. Read [its report](SWITCHING_PILOT_RESULTS_2026-10-25.md) for actual runtime,
queue, fit, restart, branch-order and matched outcome checks. It supersedes the
unmeasured timing budget above. The pilot found a completed-episode projection
bug, fixed with a guarded zero-unfinished-work result, and campaign identity now
includes online refit cadence. These source changes are staged in the new v4
source-only build. Full training remains unstarted. Longer lookahead, explicit
exploration and a learned terminal fleet value remain proposed experiments;
the 15-minute decisions, 12-hour arrival horizon and four held-out cities are
unchanged.

`requirements-switching-vm-pins.txt` pins the observed direct VM packages from the
existing environment inventory. It is not a full transitive/native lock, and no
packages are installed by this build. The current verification receipt records
whether the revised source could actually be staged on the VM; a target path in
the config is not proof that an SSH transfer succeeded.
