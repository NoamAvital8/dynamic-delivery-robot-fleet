# Assignment-triggered coordinated idle relocation v1

This opt-in version builds on `codex/queue-aware-assignment` (`9c712d7`). It
preserves the old policies and their results. No new NN or separate per-cluster
scoring networks are trained: cluster history already updates the online
Gamma-Poisson posterior from **order arrivals**, not only completed deliveries.

## One shared fleet decision

Each committed delivery assignment requests an idle-planning refresh. Requests
at the same simulated timestamp are coalesced into one pass after order,
street-edge, charging and service events at that timestamp settle. This avoids
repeated work on intermediate assignment states. Periodic replanning and the
existing delivery/charging completion triggers remain active.

The coordinator forecasts each robot's future service location, remaining
mission/handling/charging/waiting time, battery and importance eligibility.
Existing relocation destinations count as **planned coverage**, once, rather
than treating travelling robots as available at their original location.
Parallel workers compute coverage vectors from immutable inputs; only the
coordinator selects actions and updates the common coverage and port calendar.

For each candidate, compare its marginal fleet utility against **continuing
its current plan** (or staying, for an uncommitted idle robot). Select the best
worthwhile move, update shared coverage and any charging reservation, then
recompute all remaining marginal gains. Stop when no move passes its threshold
or the bounded action budget is exhausted.

The existing surrogate remains:

`U = sum_(cluster,importance) W * (1 - exp(-sum_robot coverage))`

`W = horizon * posterior_arrival_rate * (importance + 1)^2`.

Coverage includes remaining availability delay, Haversine response travel,
battery after travel, robot capability and priority eligibility. Moving replaces
the robot's old contribution, so lost origin coverage counts as a cost. Travel
time and energy consumption reduce destination coverage; no second arbitrary
travel penalty is added. This is a **loss-weighted response/coverage surrogate**,
not an exact prediction of future total delivery loss or a learned local utility.

## Overlap, charging and stability

- Response coverage overlaps spatially even across different cluster labels.
  Sending one robot toward a hotspot reduces the next robot's marginal benefit.
  There is no blanket one-robot-per-cluster restriction: multiple robots are
  allowed when their marginal gains remain worthwhile. No uniform spreading is
  forced, and inactive regions need not receive a robot.
- Active charging, queued charging and committed trips to charge cannot be
  interrupted by idle planning. Background charging continues to 100%; only a
  real delivery assignment may interrupt it, as before. Planned charging slots
  do not overlap on a physical port. These are forecasts, not protected future
  appointments in the actual FIFO queue; actual mission arrivals can change waits.
- A relocation can be retargeted only after a cooldown (default **2 minutes**),
  and its gain over continuing must exceed both the ordinary move threshold
  (default **0.1%** of total demand weight) and a switching threshold (default
  **0.5%**). These are tunable heuristics, not constants derived from the papers.
- Retargeting preserves the already committed street edge, ETA and energy.
  Only the route after the next node changes, reusing the existing arrival event.
  Stopping at that next node is also an option. A rejected/no-change decision
  leaves the original intent and dispatchability intact.
- Unassigned real orders block speculative moves. Robots on delivery work
  contribute forecast coverage but are never relocated by the idle controller.

Limits remain: at most 32 demand clusters, five relocation destinations and
three charging alternatives per robot, and 32 selected actions per epoch by
default. Coverage scoring is process-parallel (`--idle-processes`); selection is
serialized deliberately to avoid races. Busy forecasts reuse cached phases,
not repeated battery-route searches. Assignment-triggered planning increases
the number of epochs; benchmark `idle_planning_seconds` rather than assuming
it is free or globally optimal. Nearby geometry is currently Haversine-based;
barriers and road topology can still make this approximation optimistic.

## Versions and measurements

| Policy | Queue-aware delivery assignment | Assignment-triggered coordinated idle | NN |
|---|---|---|---|
| `full` | No | No | Existing spatial NN |
| `full_queue_aware` | Yes | No | Same NN |
| `full_coordinated_idle` | Yes | Yes | Same NN |
| `full_coordinated_idle_no_nn` | Yes | Yes | None |

The default six policies remain unchanged. To isolate this relocation change,
compare `full_queue_aware` with `full_coordinated_idle` on identical held-out
scenarios and unchanged model files. Use the corresponding no-NN variants for
an ablation. Comparing directly with `full` also includes the previous charging
queue change, so it cannot isolate the effect of relocation coordination.

`result.idle.jsonl` durably appends each committed idle decision, including the
previous destination, legal decision node/time, destination, arrival forecast,
battery and whether an existing edge was preserved. `result.queue.jsonl` still
records mission/background waiting separately. Result/summary fields identify
`policy_version=coordinated_idle_v1`, assignment refresh requests/epochs,
retargets, preserved edges, cooldown protections, rejected commits, thresholds,
planning runtime and the decision-ledger path.

Completed scenario/policy evaluations are restart-friendly using the existing
input/source/model signatures. An interrupted simulator restarts its evaluation;
the event ledger is not a full mid-simulation checkpoint. Preserve old results
and use a new campaign directory for this version.

## VM comparison (not automatically started)

Checkout: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-coordinated-idle`.
Existing campaigns, graphs, original virtual environment and trained NNs remain
unchanged. This example runs 50 paired evaluations across five cities/five seeds.
Check currently active workers before launching; keep total simulator plus nested
idle workers below the agreed 60-core budget. Two simulators with two idle workers
each add at most six worker processes, excluding controller overhead.

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet-coordinated-idle
coord_original=/data/workspace/robot_delivery/dynamic-delivery-robot-fleet
coord_run="$coord_original/runs/icaps_coordinated_idle_v5"
mkdir -p "$coord_run/stages"
nohup "$coord_original/.venv/bin/python" -u scripts/run_multicity_experiments.py \
  "$coord_original/runs/icaps_multicity_v2_precision/suite/suite.json" \
  "$coord_run/benchmarks" \
  --spatial-model "$coord_original/runs/icaps_multicity_v2_precision/training/spatial/reservation_fcnn.npz" \
  --paper-model "$coord_original/runs/icaps_multicity_v2_precision/training/paper/reservation_fcnn.npz" \
  --policies full_queue_aware full_coordinated_idle \
  --processes 2 --idle-processes 2 --timeout-seconds 0 \
  --idle-switch-gain-fraction 0.005 --idle-retarget-cooldown-min 2 \
  > "$coord_run/stages/benchmarks.log" 2>&1 < /dev/null &
echo "Comparison controller PID: $!"
```

Append `full_queue_aware_no_nn full_coordinated_idle_no_nn` for 100 evaluations.
Use `--dry-run` without `nohup`/redirection to validate inputs and count jobs
without starting simulations. Rerun the same command to retain compatible
completed evaluations. Do not start duplicate controllers in the same directory.

```bash
tail -n 40 -F /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_coordinated_idle_v5/stages/benchmarks.log
```
