# Small-city switching: build-only handoff

This implementation is staged for review. No scenario preparation, delivery
simulation, model fitting, or training service is launched by importing it or
by running `plan`. Every other command requires an explicit `--execute` flag.

## Experiment

- Eight development cities: Tel Aviv, Haifa, Manhattan, New Delhi, Delft,
  Leuven, Oxford and Palo Alto.
- Four final-test cities: Cambridge (UK), Ghent, Salamanca and Lucca.
  Their names are a hard-coded exclusion from training and validation, checked
  again at map selection, dataset loading, collection and model fitting.
- One hundred training and 25 validation scenarios per development city.
  There are 800 training + 200 validation inputs. Both learners use those same
  inputs, but collect their own trajectories: **1,600 training runs and 400
  frozen validation runs** in total.
- Twelve-hour arrival scenarios, **15-minute decisions**, and all ten supported
  benchmark policies as candidate actions. Continue decisions after the arrival
  horizon if deliveries remain; incomplete work is never accepted as a result.
- Balanced ordinary, surge, spatial-shift, priority-shift and charging-pressure
  profiles. Fleet size and initial battery vary within declared bounds. Public
  demand priors are configured expectations, never actual test arrival counts.
- XGBoost: ten cost-difference regressors, 200 depth-four trees each.
- Small NN: shared ReLU layers of 64 and 32 units, ten linear outputs, weighted
  Adam optimization in NumPy. Both models predict per-robot cost differences
  from the state-wise portfolio mean. Episode weights prevent long trajectories
  from dominating merely because they supply more decision windows.

The initial selector is a documented constant Reactive Insertion initializer.
After 2,000 new states per family, the controller fits and publishes a learned
checkpoint. Workers adopt updates at decision boundaries. No actor chooses its
action using the teacher branch scores. Every decision records the exact model
hash that made it. Validation uses one frozen final checkpoint per family and
never contributes labels or scalers to fitting.

## Completion score

For each benchmark, restore the same complete simulator snapshot, run exactly
the next 15 minutes, and save its boundary snapshot **before** terminal scoring.
The label is completed-delivery loss added during that window plus projected
full request-to-delivery loss for every released unfinished order.

The read-only projector preserves the remaining committed edge, current handling
time, subsequent stop order, charging energy/duration, active charging sessions,
existing FIFO waiters and known incoming background charging. Its joint event
calendar propagates charger delays between robots and across subsequent stations.
It does not simulate new arrivals or run the dispatch engine beyond the boundary.

Unassigned pending orders are not free. A common terminal rule appends them in
release order to a feasible robot after its frozen commitments. It chooses the
earliest no-new-queue completion among a bounded shortlist of feasible robots,
then recomputes **all** itineraries jointly, including any delays the added
charging requests cause to previously committed missions. Missing cached legs
are rebuilt battery-feasibly without changing the service-stop sequence and are
counted in diagnostics. Any incomplete/infeasible projection invalidates the label.

This is a fast terminal approximation. It does not estimate demand arriving
after the decision window, or the future value of an empty robot's location.
Background charging is frozen rather than interrupted by the terminal pending
assignment rule. A later authorized pilot should check projection quality and
full-episode policy performance before relying on these labels at scale.

Only the actor-selected **15-minute** boundary becomes the next actual state.
Atomic action intents, teacher rows and state hashes preserve that choice across
interruption, even if a newer model is published before resumption. Finished
results and model artifacts are immutable and reused on resume.

Old collection labels are **not** merged: they used six actions, sampled 25% of
decisions and had a different continuation target. All existing artifacts stay
preserved. The new implementation reads the preserved VM backend as a dependency
and never edits it or the original campaign sources.

## Parallel execution

The VM reports 72 logical CPUs and 36 physical core IDs. Default admission is up
to 64 episode workers, four fitting threads, and remaining capacity for the
coordinator and existing work. Each simulation worker uses one numerical thread;
there is no nested pool multiplying workers by ten candidate benchmarks. The old
60-process cap is not used. Worker count also respects a 64 GiB available-memory
reserve and a configurable 1.5 GiB prospective worker allowance. Admission pauses
if the reserve is threatened; actual peak memory and throughput still need a pilot.

## Build verification and runtime estimate

Verification uses only synthetic arithmetic fixtures, deterministic collector
test doubles, serialization/inference checks and a read-only planning command.
It does not construct the production FleetRuntime, generate scenarios, advance
delivery events or optimize either model. Production integration and timings are
therefore intentionally unbenchmarked at this handoff.

Historical mean 15-minute branch-window costs, excluding old scoring tails:

| Existing city | Seconds per branch |
|---|---:|
| New Delhi | 0.0863 |
| Haifa | 0.2774 |
| Tel Aviv | 0.4054 |
| Manhattan | 2.0260 |

The new design has about **800,000 training branches**: 1,600 actor episodes x
approximately 50 decisions including drain x ten actions. Assuming 0.7-2.5 seconds
per branch including projection gives 156-556 worker-hours. At a conservative
36 effective workers plus 2-6 hours for preparation, refits, validation and
overhead, the planning range is **about 6-22 elapsed hours for both learners**.
Allow **24-36 hours for the first full campaign** until new maps, heavy profiles,
projection overhead, cache warm-up and memory-limited concurrency are measured.
This estimate excludes the separate held-out four-city benchmark campaign.

## Commands for a later explicit launch

On the VM the build is under:

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_v2_20261025`

The readable launcher is `operations/switching_meta.py`; configuration is
`config.json`. Replace the placeholders below with those absolute paths.

```text
python -B operations/switching_meta.py plan --config config.json
python -B operations/switching_meta.py prepare --config config.json --execute
python -B operations/switching_meta.py train --config config.json --execute
```

The first command is read-only. The second prepares development inputs only.
The third is the **explicit training launch**, and is not executed during build.
Do not run a second controller for the same output directory.

Only after models have been frozen by development training:

```text
python -B operations/switching_meta.py prepare-test --config config.json --execute
python -B operations/switching_meta.py evaluate --config config.json --execute
```

Neither command updates a model. `evaluate` runs the two frozen selectors; an
additional fixed-benchmark evaluation campaign is separate future work. No test
scenario is generated by the development preparation command.
