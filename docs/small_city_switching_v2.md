# Small-city switching: current build guide

Read [the project handoff](../PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md) and
[review-fix addendum](switching_review_fixes_20261025.md) before reviewing or
launching this code. The original 2ffd002 build receipts are historical; current
sources are described by review_bundle/CURRENT_BUILD_SOURCES.json.

## Experiment

- Eight development cities: Tel Aviv, Haifa, Manhattan, New Delhi, Delft,
  Leuven, Oxford and Palo Alto.
- Four final-only cities: Cambridge UK, Ghent, Salamanca and Lucca.
- Per development city: 100 training and 25 disjoint validation scenarios.
  Each model collects its own actor-chosen trajectories: 1,600 training episodes.
- Full development validation: 200 inputs times two frozen selectors and ten
  fixed benchmarks = 2,400 matched episodes. Final test later uses 200 new inputs
  times the same 12 participants = another 2,400 episodes.
- Twelve-hour arrivals; 15-minute decisions through completion of all deliveries.
- XGBoost: ten depth-four, 200-tree regressors. NN: ReLU 64/32, ten outputs.
- Both predict centered per-robot costs. All scalers and fitting weights use
  training only; old six-action rollout labels are incompatible.

## Collection, projection and model updates

The actor records its action before teacher labels. Every candidate starts from
the same complete snapshot. Its label is the actual full delivery loss added in
the next window plus projected full loss for all released unfinished work.
Only the actor-selected pre-projection boundary becomes the next physical state.

The joint FIFO projector preserves saved current event IDs and physical event
priorities, projects travel edge by edge, and includes current/known incoming
charging, handling and pending orders. Pending assignments use the common bounded
earliest-feasible completion rule and are jointly reprojected. Missing or stale
legs are reconstructed with start-state/feasibility guards. Independent order,
stop and capacity checks reject incomplete labels.

This is a frozen completion approximation without new future demand or terminal
fleet value. It does not establish long-run action optimality.

Each family begins with Reactive Insertion. A bounded asynchronous fitter refits
after roughly 2,000 new compatible states from that family's own trajectories.
It reads an immutable SQLite replay prefix; the coordinator keeps processing
workers and failures. Model adoption happens only between decisions. Frozen
validation/test does not collect teacher labels or update weights.

## Safety and resources

Every episode and fit uses a fresh owned process. Legacy graph hooks are
idempotent and runtime-owned registry entries are released. Up to 64 episode
processes, one numerical thread each, coexist with one four-thread fitter.
Admission reserves 64 GiB available RAM, 8 GiB for the trainer, and 1.5 GiB per
prospective episode, including not-yet-visible startup allocations. Reserve
crossings/failures request checkpoint-safe stop, then terminate unresponsive
owned children after a bounded grace period. These are admission estimates,
not a hard system memory limit.

Campaign, dataset, model/freeze and row-content identities prevent incompatible
reuse. Initial state and current/previous checkpoint pairs are retained with
durable file writes and POSIX directory fsync. Exact-byte recovery is required
for missing committed boundaries; mismatches fail closed. Opening final test
prevents further development in the same campaign. Windows/storage limitations
and remaining production verification are described in the addendum.

## Source-only VM build

Target directory (see review_bundle/CURRENT_BUILD_VERIFICATION.json for actual
staging status; SSH may be unavailable):

/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_v4_pilot_fixes_20261025

Launcher: operations/switching_meta.py.
Config: configs/small_city_switching_v2.json.
Old v2 build and all historical campaigns are preserved.

The only read-only operational command is:

    python -B operations/switching_meta.py plan --config configs/small_city_switching_v2.json

For a future explicitly authorized launch, the command sequence is prepare,
train, prepare-test, evaluate, each requiring --execute. Train now includes
matched development baselines. Evaluate includes both frozen selectors and all
ten matched fixed benchmarks, complete reporting and a development-selected
fixed baseline. Do not run a second controller against the same output.

A Git-only reviewer can run review_bundle/materialize.py to verify hashes.
Its optional --execute only inflates the bundled maps and writes portable paths.
It never creates scenarios or runs a simulator. Production collection targets
Linux; portable source inspection and the scoped synthetic suite work on Windows.

## Conditional estimate and verification

Approximately 800,000 training branches plus 120,000 validation windows are
planned. The old unmeasured 7–24-hour arithmetic and 24–48-hour precautionary
budget are superseded by [the measured pilot report](SWITCHING_PILOT_RESULTS_2026-10-25.md)
and `review_bundle/evidence/switching_pilot_20261025/TRAINING_TIME_ESTIMATE.json`.
Final-test runtime remains unmeasured and is additional.

Current scoped counts are in review_bundle/CURRENT_BUILD_VERIFICATION.json.
The synthetic suite uses pure fixtures, stub transitions/publication,
child-process controls and fixed-weight arithmetic. A separate bounded pilot
has exercised the production runtime, queue projection, actual fitting,
serialization, interruption/resume and matched development comparisons. Its
evidence and immutable source snapshot are bundled. No full campaign or final
test has been launched. The future campaign must use its own fresh output.
