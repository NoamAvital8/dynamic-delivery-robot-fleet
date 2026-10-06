# Queue-aware assignment v1

This is a separate, opt-in version, based on the full algorithm at commit
`39c6d15`. It does not replace the original dispatcher, NN, posterior, idle
utility, 100%-background-charging rule, or existing experiment results.

## Versions to compare

| Policy | Assignment queue forecast | NN reservation | Predictive idle |
|---|---|---|---|
| `full` | No | Yes | Yes |
| `full_queue_aware` | Known-traffic FIFO | Same existing NN | Same controller |
| `full_no_nn` | No | No | Yes |
| `full_queue_aware_no_nn` | Known-traffic FIFO | No | Same controller |

The multicity runner's default six policies are unchanged. Queue-aware versions
must be explicitly selected. Reusing the unchanged NN isolates the effect of
queue-aware planning; subsequent NN retraining is a separate experiment.

## Forecast and route scoring

For each decision, copy active sessions, actual FIFO queues, committed
background charging trips, and charging visits in released orders' cached
remaining routes. Both delivery and background traffic occupy ports.

Current charge/queue/service and committed street edges contribute only their
remaining duration. Future ETAs use cached exact phase distances, handling,
scheduled energy, and waits against observed occupancy/queues. These arrival
forecasts are one-pass estimates: later visits are not iteratively updated for
all interactions between other robots' predicted future visits.

For every candidate robot, exclude its old active session and queued/planned
requests because real reassignment cancels those editable commitments. Give
each candidate itinerary a private FIFO replay. Schedule requests arriving
before it onto the earliest-free physical port. Queue waiting is

`max(0, earliest_port_free - candidate_arrival)`.

Later predicted arrivals are **not protected reservations**. They cannot force
an earlier-arriving candidate to wait. Equal-ETA known requests are processed
before the candidate conservatively. Forecasts never inspect hidden future
scenario orders, and candidate evaluation never changes real queues.

Propagate every predicted wait through all subsequent stops and recompute
deadline/importance loss for all orders on the candidate robot. Rebuild the
forecast on each selection and invalidate baseline loss caches; identical
position/battery does not imply identical queue-dependent loss.

When a prefix predicts waiting, also compare detours through reachable chargers.
Defaults bound this to three alternatives among twelve nearest reachable
stations. `--queue-charger-alternatives` and `--queue-charger-scan-limit` control
those computational limits; zero alternatives disables detour search. Candidate
detours retain battery feasibility and cannot trade away the original prefix's
arrival battery. Prepared phases execute the selected detour on actual graph
edges. This is bounded prefix search, **not a globally optimal station/route
solver**. The Haversine top-K shortlist remains unchanged.

The score measures forecast loss changes on the rerouted robot. It does not
price induced waiting/loss changes for every other robot, model unobserved
future charging requests, or anticipate unknown charge interruptions. Routing,
forecast uncertainty and priority reservation therefore remain approximations.

## Exact mission-only measurements

`result.queue.jsonl` records each queue episode when charging starts or a queued
request is canceled, including zero waits. Each record contains robot, station,
purpose (`delivery`/`background`), arrival, outcome, observed waiting, and matched
assignment prediction if available. Charging duration is never part of waiting.
Episodes still queued at a crash have not ended and are not included yet.

`result.json` and the rolling summary expose mission-only total/mean waiting,
mean among positive waits, p95, maximum, canceled count/wait, prediction sample
count/MAE/bias, and forecast diagnostics. The city summary pools sums and visit
counts rather than averaging scenario averages. Old versions' unmeasured queue
fields stay blank, not falsely zero. Comparison directories are separate and
finished evaluations remain resumable using the existing source/model/input
signatures; an unfinished simulation still restarts its evaluation.

## VM deployment and paired comparison

New checkout: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-queue-aware`.
Existing checkouts/campaigns remain unchanged. The original virtual environment,
graphs, suite and trained models are reused read-only.

Run the following **only when ready to start a new comparison**. This evaluates
two policies on the same five cities and five seeds (50 evaluations), detached
from SSH. It does not retrain the NN or overwrite old campaign directories.

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet-queue-aware
queue_baseline_repo=/data/workspace/robot_delivery/dynamic-delivery-robot-fleet
queue_run_dir="$queue_baseline_repo/runs/icaps_queue_aware_v4"
mkdir -p "$queue_run_dir/stages"
nohup "$queue_baseline_repo/.venv/bin/python" -u scripts/run_multicity_experiments.py \
  "$queue_baseline_repo/runs/icaps_multicity_v2_precision/suite/suite.json" \
  "$queue_run_dir/benchmarks" \
  --spatial-model "$queue_baseline_repo/runs/icaps_multicity_v2_precision/training/spatial/reservation_fcnn.npz" \
  --paper-model "$queue_baseline_repo/runs/icaps_multicity_v2_precision/training/paper/reservation_fcnn.npz" \
  --policies full full_queue_aware \
  --processes 8 --idle-processes 4 --timeout-seconds 0 \
  > "$queue_run_dir/stages/benchmarks.log" 2>&1 < /dev/null &
echo "Comparison controller PID: $!"
```

Append `full_no_nn full_queue_aware_no_nn` to `--policies` for a 100-evaluation
four-policy ablation. Account for existing simulations when choosing workers:
eight simulator processes plus up to four idle workers each use at most forty
processes, before other campaigns. Do not launch duplicate controllers against
the same output directory. Re-run the same completed/interrupted comparison
command to resume compatible completed jobs; use a fresh directory for changed
models, algorithms, or inputs.

```bash
tail -n 40 -F /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_queue_aware_v4/stages/benchmarks.log
```
