# Uncertainty-gated online MLE reservation v1

This is a separate, opt-in NN replacement on
`codex/uncertainty-gated-mle-reservation`. Existing policies, trained NNs and
completed experiments are unchanged. **No new full campaign has been launched.**

## What changes

Start with no priority reservation. Every newly revealed request updates the
arrival counters, including requests that have not been assigned or delivered.
Estimate demand, check uncertainty, then search a bounded set of reservation
fractions. Enable priority restrictions only when the gates below pass. Keep
learning and checking after activation; return to unrestricted dispatch if
the evidence weakens, an epoch resets, the arrival horizon ends, or real orders
are pending. There is no mandatory two-hour exploration period.

The fallback uses the existing **queue-aware Haversine Top-K greedy dispatcher**
without priority restrictions. It is **not** the exhaustive `reactive_insertion`
benchmark. Exact shortlisted route evaluation, capacity/battery checks,
charger-queue forecasting, all-order incremental loss and predictive idle
planning remain active in both fallback and reservation mode. In particular,
fallback does not mean disabling idle planning or reserving no charging ports.

The online Gamma-Poisson model remains unchanged for idle-location utility.
Only the NN that outputs delivery-priority reservation fractions is replaced.
No NN is loaded or retrained by either new policy.

| New policy | Idle planner retained | Matched NN comparator |
|---|---|---|
| `queue_mle_reservation` | Queue-aware, legacy periodic idle | `full_queue_aware` |
| `full_mle_reservation` | Assignment-triggered coordinated idle | `full_coordinated_idle` |

The default six benchmarks are unchanged. Comparing these two new policies
with each other changes idle coordination, not the MLE estimator. Comparing
each with its matched NN policy isolates the overall reservation-controller
replacement, which includes a different readiness forecast and switch/fallback
logic, not just a different mathematical estimator.

## Rates and confidence: what is guaranteed

For a fixed cluster z and importance c, after observing n arrivals over t
simulation minutes in the current epoch:

`lambda_hat[z,c] = n[z,c] / t` (requests/minute).

Aggregate importance-class rates are estimated in the same way. Counters and
the bounded request-template buffer reset every fixed **60 simulation minutes**,
matching the existing suite's hourly piecewise-constant Poisson demand generator.
The old idle posterior does not reset. Fixed epochs permit rates to differ
between epochs; they do not detect arbitrary changes within an epoch.

Ordinary 95% intervals do not retain 95% coverage if checked repeatedly and
stopped whenever they become narrow. This implementation uses an **anytime
Poisson confidence sequence**. For candidate rate lambda, use a fixed Gamma
mixing distribution with a=0.5 and b=10 minutes, chosen before observing data:

`log M(lambda) = a log b + log Gamma(a+n) - log Gamma(a)
                 - (a+n) log(b+t) + lambda*t - n log(lambda)`.

Invert `M(lambda) < 1/delta_cell_epoch` to obtain the interval. Ville's
inequality controls boundary crossing across all inspection times for a
constant-rate Poisson process. Split the total error probability across the
fixed cluster/importance cells, aggregate importance cells, and all epochs:

`delta_cell_epoch = (1-confidence) * 6 / (pi^2 * (epoch+1)^2) / cell_family_size`.

The union bound gives joint, time-uniform coverage at the configured confidence
level under those assumptions; independence between the aggregate and spatial
counters is not required. Zero observed arrivals still have a positive upper
bound. At zero exposure the bound is uninformative and reservation is disabled.
The Gamma mixing distribution here is a mathematical device, not a fitted
posterior and not the idle Gamma-Poisson prior.

**The guarantee is about arrival rates, not optimal alpha, simulator loss or
winning against the NN.** Within-epoch nonstationarity, dependence or bursty
non-Poisson traffic can invalidate the coverage claim. Pooling several of the
suite's hourly buckets into a longer constant-rate epoch would not preserve
that justification. Epoch length must be chosen in advance, without tuning on
held-out city results. Bounds describe the current epoch rate, not an unknown
next-hour rate. See the general time-uniform inference
framework in [Howard et al.](https://arxiv.org/abs/1810.08240).

## From rates to robots: the bounded optimizer

At most once every **15 simulation minutes**, forecast the next **45 minutes**,
limited by both the remaining arrival horizon and the next epoch boundary.
There must first be at least **20
observed arrivals in this epoch**. The rate-width gate requires

`sum_class(upper - lower) / sum_class(lambda_hat) <= 0.75`.

This is the width of the summed class-rate interval relative to its estimate,
not a requirement that every rare cluster achieve a tiny relative interval.
The spatial uncertainty still enters the stress tests. Narrow intervals alone
do not imply that any robot should be reserved. Low-demand cities may never
collect enough information within an hour to pass this conservative gate;
staying with unrestricted dispatch is an intended outcome, not a promised
eventual switch. Inspect activation logs before interpreting the results.

Construct up to seven reservation candidates, including unrestricted dispatch:
nominal reserved shares `0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50`. Adjust these
by robot-type response efficiency and divide reserved capacity among higher
importance levels using estimated demand times `(importance+1)^2`. Fractions
sum to one for each type and keep at least 25% general capacity per type.
Existing integer apportionment and concrete robot assignment are reused:
prefer robots with lower forecast response time to relevant clusters and
protect the largest-capacity general-purpose robot. Duplicate integer
assignments are removed. These are designed heuristics, not a closed-form
statistically optimal solution.

Score each candidate on identical sampled future traces (common random numbers
across candidates), using four rollouts per stress case. For three importance
classes there are 11 cases: central MLE, eight class-rate lower/upper corners,
and two additional spatial-weight stresses. Traces sample only already
revealed pickup/dropoff distances and item sizes; an unseen cluster borrows
known trip/item templates at that cluster's representative. No future test
orders, outcomes or offline labels are read.

The bounded surrogate treats robots as nonpreemptive servers. Service estimates
include Haversine approach/trip distance divided by speed, pickup/dropoff
handling, charging for an estimated energy deficit, and known charger waits
when charging is required. Busy readiness includes remaining travel, handling,
planned charging and waiting, with projected final location/battery. It uses
the project's importance/deadline loss formula. Reservations alter which
servers can accept each importance; already committed missions are not canceled.

Important limitations: the surrogate does not perform exact route insertion,
propagate future robot locations/battery depletion, or update induced future
charger traffic between sampled jobs. It uses Haversine-based deadline
allowances, while actual simulation deadlines use graph distance. It is not
the actual simulator and cannot certify true expected loss. Its best candidate
is only the best of this bounded candidate set. Exact graph/battery route
feasibility remains enforced by the real dispatcher.

Activate only if the central best candidate is a real reservation, the
largest difference between its **realized integer fractions** and the stress
cases' best fractions is at most **0.10**, and that same candidate improves
mean surrogate loss by more than **1% in every stress case**. This finite grid
does not cover every point inside the joint confidence region. Monte Carlo
loss differences have no separate confidence certificate in v1. If the
surrogate is infeasible or a sampled trace exceeds the default 1,000-request
budget, fall back; never truncate that workload and claim a win.

## Parallelism, logging and restart behavior

`--processes` controls simultaneous city/scenario simulators.
`--idle-processes` controls the unchanged idle-scoring pool.
`--mle-processes` controls the separate candidate-scoring pool; **1 is inline**
and is the default because at most seven candidates often do not justify IPC.
The multicity runner caps numerical-library threads at one and rejects MLE
campaigns whose worst-case compute-worker budget exceeds 60:

`simulators * (1 + idle_children + mle_children) <= 60`.

A setting of `--processes 12 --idle-processes 4 --mle-processes 1` uses at most
60 compute workers. It is one campaign's limit, not an automatic accounting of
unrelated jobs already running on the VM. Use fewer workers if other jobs exist.
Candidate scores are deterministic across serial/process-parallel execution
for the same seeds and inputs.

Each evaluation saves `result.json`, `result.progress.json`, `runner.log`,
`status.json`, `stamp.json`, idle/mission charger records, and
`result.reservation.jsonl`. The reservation audit logs time, epoch, counts,
activation/fallback reason, rate bounds and interval width when inspected,
selected fractions, stability and stress gain when scored. Final results also
contain full MLE configuration, first activation, active simulation minutes,
planning runtime, reason counts and bounded-budget fallbacks. Inspect whether
reservation actually activates before interpreting policy comparisons.

The campaign updates `summary.csv` and `city_policy_summary.csv` after every
evaluation. Repeating the exact command reuses scientifically compatible
completed results and restarts unfinished evaluations. Progress/audits survive
a crash, but **v1 cannot resume the internal event state of an unfinished
simulation**. Keep the same checkout, parameters and output directory for
restarts; incompatible signatures are rejected. Do not adopt old NN outputs
as MLE results or overwrite prior campaigns. Timeout defaults to disabled.

## VM command (when authorized to launch)

Checkout: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-mle-reservation`.
Shared suite/models/venv remain under the original repository. Both model
arguments below are required by the existing campaign CLI for metadata; the
two MLE simulators do not load them.

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet-mle-reservation
base=/data/workspace/robot_delivery/dynamic-delivery-robot-fleet
out="$base/runs/icaps_mle_reservation_v6"
mkdir -p "$out/stages"
nohup "$base/.venv/bin/python" -u scripts/run_multicity_experiments.py \
  "$base/runs/icaps_multicity_v2_precision/suite/suite.json" \
  "$out/benchmarks" \
  --spatial-model "$base/runs/icaps_multicity_v2_precision/training/spatial/reservation_fcnn.npz" \
  --paper-model "$base/runs/icaps_multicity_v2_precision/training/paper/reservation_fcnn.npz" \
  --policies queue_mle_reservation full_mle_reservation \
  --processes 12 --idle-processes 4 --mle-processes 1 \
  --timeout-seconds 0 \
  > "$out/stages/benchmarks.log" 2>&1 < /dev/null &
```

This plans **50 evaluations**: five cities, five held-out scenarios and two
new policies. It runs detached from SSH. Add `--dry-run` to the Python command
without `nohup`/redirection to validate the 50 planned jobs without running
them. The direct single-scenario runner accepts `--mle-idle-mode legacy` or
`coordinated`. All estimator/search defaults can be overridden with the
corresponding `--mle-*` flags shown by `--help`.

```bash
tail -f /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_mle_reservation_v6/stages/benchmarks.log
```

Evaluate paired losses on the same held-out scenarios against preserved queue
and coordinated NN results, and the corresponding no-reservation ablations.
Report activation frequency, loss, on-time/late delivery, mission-only charger
waits and computation cost. Do not choose the final parameters from held-out
results and then present those results as untouched test evidence.
