# Read-only review: new small-city switching meta-model

**Review date:** 25 October 2026  
**Repository:** NoamAvital8/dynamic-delivery-robot-fleet  
**Branch requested:** `codex/small-city-switching-review-20261025`  
**Exact reviewed commit:** `2ffd002a480cef66b56ecc0ce73316a4b2ee3b33` (`2ffd002`)  
**Vendored simulator revision recorded by the bundle:** `5e76f5384aec98e9d5a538b27dae205df301fa28`

## Executive assessment

The implementation substantially follows the proposed **actor-led, counterfactual supervised-learning loop**. It records the actor's choice before teacher scoring, branches from the same snapshot, and adopts the chosen policy's physical boundary rather than the teacher's best or projected terminal state. The new development/final-test split is explicit. I found no direct future-arrival input in the inspected feature path and no obvious algebraic error in the new loss equation or NN backpropagation.

**I would not launch the full 1,600-episode collection yet.** There is a confirmed mismatch between the projector's event priorities and the simulator's priorities, and experiment identity does not cover all dependencies needed for safe reuse. Failure shutdown and crash durability also need attention. The remaining projection assumptions require small, separately authorized integration experiments before their labels justify a large training campaign.

There is **no production performance result for the new selectors**. A correct implementation of the proposed objective would still not guarantee superiority over the best fixed policy: the terminal objective omits future arrivals and future fleet value, and the predictor sees an aggregated, partially observed state.

### Scope and restrictions honored

- Read the supplied `PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md` completely, including its scientific limitations and review constraints.
- Read the pinned files through the GitHub connector, including the new source/configuration/tests, the relevant vendored runtime and generated-policy sources, portable-bundle helper/provenance, and historical reports.
- Resolved the exact commit through GitHub commit metadata. The existing local checkout was on a different revision; I did **not** check out, fetch into, or modify it. Accordingly, the SHA above is API-resolved, not a claim that local `git rev-parse HEAD` returned this revision.
- Did **not** execute project code, tests, materialization, fitting, scenario generation, simulators, or training. Did not access the VM for this review. Did not modify a repository, commit, push, or create a PR.
- This standalone report, outside the repository, is the only deliverable. All failure cases below are source-derived/hand-worked, **not newly executed reproductions**.
- I inspected the exported reports and metadata, not every compressed historical JSONL record or every binary graph/weight. I did not independently rerun the bundle-wide checksum verification. The build receipts are evidence of what their author reports, not fresh verification by this review.

All source references below refer to the exact reviewed commit. Backend paths mean **`review_bundle/backend/…`**, never the unrelated older top-level `src/delivery_fleet`.

## Priority summary

| ID | Priority | Classification | Finding |
|---|---|---|---|
| F1 | P1 | Confirmed correctness defect | Projected service completions have the wrong event priority; simultaneous charger requests can get the wrong FIFO order. |
| F2 | P1 | Confirmed reuse/identity defect | Dependencies affecting labels, model versions and outcomes are omitted from combined reuse guards. |
| F3 | P2 | Confirmed failure-control defect | Cancelling futures does not stop already-running episodes; pool shutdown can wait indefinitely after one failure. |
| F4 | P2 | Confirmed durability gap | Atomic replacement is not durable checkpointing across VM/power failure; previous states are deleted without durable commit ordering. |
| R1 | P1 before scale-up | Documented but unmeasured operational risk | Memory admission is an estimate, not enforcement of the 64 GiB reserve. |
| R2 | P2 | Scientific limitation | The teacher is a truncated, frozen-itinerary objective; it has no future-demand or terminal fleet value. |
| R3 | P2 | Untested integration risk | Production route/state conversion, fitting, publication and multi-step resume lack new integration evidence. |
| R4 | P2 | Protocol/reporting gap | Final-test immutability, matched fixed benchmarks and complete outcome reporting are not fully enforced/provided. |

P1 means address before substantial collection or relying on reusable results. No P0 production failure was demonstrated in this read-only review.

## Confirmed findings

### F1 — P1: projector changes simultaneous-event semantics and therefore charger FIFO order

**References:**

- `src/fleet_switching/projection.py`, `project()`, lines 123–151, especially line 151.
- `src/fleet_switching/backend.py`, `completion_projection()`, lines 117–151.
- `review_bundle/backend/scripts/run_nyc_benchmark5.py`, `start_service()`, lines 1271–1285; node-arrival handling, lines 1382–1403; `fill_free_ports()`, lines 511–514.
- `review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py`, `FleetRuntime.advance()`, lines 646–652: it executes the original event ordering.

**Evidence:** the production engine schedules `service_complete` with priority **0**, and movement/node arrivals with priority **1**. The projector schedules both `Step("service", …)` and `Step("travel", …)` completion through `event(..., 1, "advance", ...)`. It seeds these events by sorted robot ID and its own sequence counter, rather than preserving physical event order. Existing queued requests are handled before new requests, but the ordering of simultaneous **new** requests is not faithful.

**Concrete hand-worked failure case:** one charging port at station S, no initial waiters, and two robots:

- R1, lower ID: travel 1 minute to S; charge 10 minutes; travel 1 minute; dropoff handling 1 minute.
- R2: finish 1 minute of current pickup handling at S; charge 1 minute; travel 1 minute; dropoff handling 1 minute.
- Give both orders sufficiently generous deadlines and importance 1.

At minute 1, the real service completion runs before the travel arrival. R2 charges from 1–2 and finishes delivery at 4; R1 charges from 2–12 and finishes at 14. Total remaining completion-time loss is 18 when request times are zero. In the projector, both boundary events have priority 1 and R1's earlier insertion wins: R1 finishes at 13, R2 at 14, giving 27. No future arrivals or replanning are needed for the discrepancy. Importance/deadline penalties can magnify it.

**Impact:** labels can reverse action rankings even within the declared frozen-itinerary approximation. Equal-time arrivals can also disagree because the projector substitutes robot-ID ordering for the simulator's event sequence. This is a fidelity defect, not merely uncertainty about future behavior.

**Suggested fix:** preserve event kind/priority in projected steps and retain the relevant boundary-event sequence for current commitments. Specify a consistent rule for future equal-time events. Keep existing FIFO waiters ahead of new arrivals, but do not replace service/travel precedence with robot ID. If a simplified tie convention is deliberately retained, label it as an approximation and measure its ranking impact; do not claim exact runtime queue semantics.

**Validation:** differential tests for service-vs-travel, travel-vs-travel and charge-complete ties, with robot IDs permuted, existing waiters, two ports and repeated station visits. Include a high-importance order so a wrong ordering is visible in loss, not just readiness.

### F2 — P1: experiment identity is insufficient for safe reuse across scientific changes

**References:**

- `src/fleet_switching/data.py`, `science_id()`, lines 15–23; `load_dataset()`, lines 107–131.
- `src/fleet_switching/collection.py`, `collect_episode()`, lines 45–60, 71–78 and 133–145.
- `src/fleet_switching/models.py`, `fit()`, lines 52–59.
- `src/fleet_switching/workflow.py`, `_rows()`, lines 51–58; `_publish()`, lines 61–76; `train()`, lines 123–131.
- `review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py`, `_use_shared_index()`, lines 195–223; input/snapshot checks, lines 390–397 and 682–697.

**Evidence:** `science_id()` hashes backend package Python, backend top-level scripts, `backend.py`, `projection.py`, label version and policy order. It does **not** hash the new collector, generator, learner/configuration semantics, fixed reservation weights, or backend `operations/exact_runtime.py`, which is actually imported and changes routing execution. Dataset loading compares only six configuration fields. Episode signatures include episode metadata, family and this incomplete science identity; they do not include the configured base-model bytes. Training completion receipts are returned before constructing a runtime or rechecking input files.

The backend snapshot input digest is a useful additional guard: it includes base-model bytes for **that runtime**. It does not repair replay/finished-result identity or protect fresh, not-yet-checkpointed episodes from joining an older replay stream.

**Concrete failure case:** collect some training episodes using reservation weights A. Change `config.spatial_model` to weights B, keeping the same output directory and prepared dataset. The six checked preparation settings and `science_id` remain unchanged. Completed training episodes and their rows can be reused; new episodes without checkpoints can use B under the same science ID. An existing checkpoint may reject the change, but that rejection does not prevent the earlier mixed replay or stale finished-result reuse in the other cases.

Similarly, changing collector behavior without bumping `LABEL_VERSION`, or altering the exact-runtime dependency, need not invalidate existing row/episode identities. `training_records_sha256` hashes only the list of row IDs, not their feature/target contents; modifying an existing row under the same ID is not reflected in that identifier. `_publish()` may then fail with a version collision rather than produce a correctly identified new model.

**Suggested fix:** separate and explicitly record:

1. **Environment/label identity:** all simulator and projector dependencies, collector transition semantics, base-model hashes, physical/controller settings, feature schema, fleet/battery settings and generator identity.
2. **Replay identity:** content hashes of validated immutable rows, bound to episode/input/intent identities.
3. **Learner identity:** model code, hyperparameters, sample weighting, scaler schema, relevant dependency versions and training-record content digest.
4. **Execution identity:** resource settings that need not invalidate scientific labels, but must remain recorded.

Check appropriate identities before accepting a finished receipt as well as before restoring a checkpoint. Bind `_rows()` to the current manifest, not just family/science fields. Hash `operations/exact_runtime.py`; its location outside `src` does not make it irrelevant to correctness. Preserve old artifacts and use a new run/version rather than silently migrating incompatible work.

**Validation:** independently alter each dependency/configuration/row component and verify that incompatible replay, snapshots, model pointers and completed outcomes are rejected at the correct boundary. Verify that changing only a harmless worker limit does not unnecessarily invalidate scientific results.

### F3 — P2: one failed episode can leave the controller waiting for all long-running episodes

**Reference:** `src/fleet_switching/workflow.py`, `_pool()`, lines 88–114, especially 108–111.

**Evidence:** on a failed future, the code calls `cancel()` on active futures and raises inside a `with ProcessPoolExecutor(...)` block. Cancelling an already-running future does not stop its episode. Exiting the executor context waits for running tasks. There is no cooperative cancellation flag, boundary-level stop request, branch timeout or controlled shutdown protocol.

**Failure case:** one worker raises because a terminal projection is infeasible while another is in a very long insertion search. The controller can remain in executor shutdown until the second finishes; it cannot promptly return the error or publish further models. Running workers can continue writing rows under the last published actor, so this is not an immediate clean stop of the learning loop.

**Suggested fix:** use explicit cooperative shutdown checked at safe decision/branch boundaries, record failed jobs and a final controller state, and stop admitting new work. Decide whether unaffected jobs should finish or checkpoint and exit. Handle a worker that cannot reach a safe boundary with a separately designed bounded termination policy; do not assume `Future.cancel()` achieves that. Preserve compatible rows and checkpoints.

**Validation:** mock one failing job and another long-running/cooperative job; assert bounded error reporting, no new admissions, preserved checkpoint data, and a single clean resume. Include a broken worker process, not only a normal Python exception.

### F4 — P2: checkpoint transactions cover process interruption better than VM/power failure

**References:** `src/fleet_switching/config.py`, `atomic()`, lines 34–39; `src/fleet_switching/collection.py`, `save_state()`, lines 31–37; checkpoint replacement and old-state deletion, lines 139–149.

**Evidence:** files are written and renamed but neither file contents nor parent directories are `fsync`ed. After replacing the checkpoint, the previous state file is immediately deleted. Atomic rename prevents readers from observing an ordinary half-written JSON file, but does not guarantee the new state, row, rename and deletion survive a VM/power failure in the required order.

**Failure case:** the new checkpoint/state exists in the page cache, the old state is unlinked, and the VM loses power. The durable filesystem can lack a valid state matching the surviving checkpoint. The current crash-injection test covers a Python exception before checkpoint publication, not storage durability.

**Suggested fix:** document the supported failure model. If VM-crash resilience is required, flush/fsync state and row files, publish/fsync the checkpoint and directory, then retire old states. Keep at least a previous valid checkpoint/state pair and validate fallbacks. Record a checksum for row contents too. This is not a claim that every ordinary process crash currently loses work.

**Validation:** simulated interrupted transactions at every write/rename/delete cut point, corrupted/missing latest state, and recovery from the retained previous pair. Real storage/power-loss behavior needs an appropriate controlled environment, not this source review.

## Risks, limitations and missing evidence

### R1 — memory reserve is prospective admission, not a hard safety guarantee

**References:** `config.py:139–152`, `workflow.py:80–113`, and configuration fields `worker_memory_gib=1.5`, `worker_limit=64`, `memory_reserve_gib=64`.

The code checks Linux `MemAvailable` before submission. It never pauses or stops already-running workers when memory drops, and `startup_allowance` does not track every already-submitted worker's still-unrealized allocation. The initial `slots` calculation is an estimate, not a cgroup limit. When there is no active work and memory is insufficient, the controller raises; it does not wait indefinitely for resources as the phrase “admission pauses” might suggest.

I am **not** claiming 1.5 GiB is already disproved: the old exported round summaries show peak RSS below that allowance for the four existing small development cities. Those figures do not measure the new maps, new projection, accumulated replay, first-time cache builds or fitting memory. Parent replay/fit memory has no separate admission reservation. Seventy-two logical CPUs are also not 72 independent physical cores, and available CPU affinity does not measure other VM workload utilization.

Before scale-up: measure process/cgroup peak memory on representative profiles, reserve trainer/coordinator memory separately, limit in-flight startup allocations, and add an operational containment/stop policy. Retain the 64 GiB reserve. Increase concurrency only from measured throughput/RAM, not node count or unused CPU alone.

### R2 — the terminal objective can systematically misrank reservation/relocation benefits

**References:** `collection.py:102–123`, `backend.py:93–115,153–185`, and the handoff's terminal-approximation section.

The label correctly combines newly realized **full** delivered-order loss with projected **full** request-to-delivery loss of unfinished released orders. Previously completed loss is excluded and there is no explicit double counting of completed versus unfinished work in the normal route representation. These overlapping state targets must not be summed as episode return; the collector correctly reports `runtime.loss` instead.

However, no future demand or value of an empty robot's location/battery enters terminal loss. At a boundary with no unfinished orders, all terminal costs are zero even when actions leave very different fleet readiness. Pending assignments freeze existing work and background charging, shortlist up to five feasible robots by readiness/pickup travel, choose a no-new-queue finish, and only then recompute joint queues. Queue delays induced by newly appended pending work are not fed back into earlier candidate selection. Those are declared approximations, **not newly discovered arithmetic bugs**.

There is also no scalar prediction that can always recover the true best action from the current aggregate features: different spatial layouts/itineraries can have similar summaries but different best choices. A better fit cannot remove all partial-observability or target bias.

Measure projection error and **action-ranking regret**, stratified by pending backlog, importance, active service, queue pressure and relocation. Compare against an authorized reference completion using the same admitted workload and separately against actual longer-run outcomes. Do not demand equality with an unconstrained replanning drain and then blame every difference on projector arithmetic: the two terminal policies are different.

### R3 — new production integration and fitting remain unverified

`review_bundle/evidence/build/BUILD_VERIFICATION.json` explicitly records 31 synthetic tests, `production_runtime_instantiated=false`, `model_fit_started=false`, and zero new delivery simulations. That is appropriately candid. Existing vendored `tests/test_meta_runtime.py` and `test_meta_fixes.py` contain useful fixed-policy equivalence, snapshot, controller and feature tests, but their presence does not establish that the **new** adapter/projector/fitter passed production integration.

Specific missing cases:

- All ten policies with the new fleet-count monkeypatch and lower initial batteries.
- Active/queued delivery charging, background charging and in-transit intents, multiple/repeated charger visits, service locks and route versions invalidated by insertion.
- Global released-work coverage: `project()` checks completion against the cost map the adapter itself constructed. Add an independent count/invariant such as `released_count - delivered_count == represented_unfinished_count`; an order accidentally absent from both pending/schedules and that map is not detected by set equality alone.
- Prepared-route start-node/start-battery consistency, capacity/pickup-before-dropoff invariants, and negative battery checks after remaining-route subtraction. The projector validates many steps but is not a full independent validator of carried payload state.
- Multi-decision crash/resume. The new collector tests use a one-step fake runtime, not a multi-boundary production state.
- Real fitting and export/import round trips for both model families in the intended dependency environment.

### R4 — frozen evaluation protocol and reporting need completion

**References:** `workflow.py:117–175,179–190`, `data.py:35–40`, and `collection.py:150–159`.

Validation/test pass an explicit frozen model and do not collect teacher labels; this is correct. But `train()` does not refuse to train again after `frozen_models.json` or a test manifest exists, and `prepare(test=True)` checks model hashes without fully binding the test preparation to all development experiment metadata. The separate training/test locks are not a single experiment-phase lock. This does not cause automatic leakage in the intended one-way CLI sequence, but leaves the “frozen before test” protocol dependent on operator discipline.

The collector's completion receipt preserves loss, order count and fleet size, but not on-time/late totals, finish time, switch counts, policy occupancy or charging diagnostics for frozen evaluation. Training-row diagnostics do not substitute for those outcomes because validation/test do not compute projections. `evaluate()` writes raw selector results only: there is no implemented new matched ten-fixed-policy campaign or full paired statistical comparison. The handoff acknowledges the separate benchmark campaign; it still must exist before claiming a win.

Add a phase lock/immutable freeze record with model, dataset, feature and scientific identities; reject later fitting in the same tested experiment unless explicitly creating a new version. Export physical outcome and switching diagnostics from the already available runtime metrics. Build the fixed-policy comparison with exactly matching maps, fleets, initial batteries, scenarios, public priors, base weights and runtime semantics. Report failures and matched coverage explicitly.

## What appears correct in the intended design

1. **Actor ownership:** `collect_episode()` writes an intent before any counterfactual label, restores `before` for every candidate, saves each branch boundary before projection, and restores `chosen_blob`. The teacher argmin never selects the actual next state.
2. **Resume action ownership:** an existing matching intent is reused, even after the live registry changes. An already committed row/boundary is checked and reused. The ordinary interruption ordering is sensibly designed, subject to F2/F4.
3. **Branch isolation:** the backend serializes mutable execution state in one pickle stream, preserving robot/controller aliases, event counter, queues, schedules, demand/controller state and reservation/queue forecast. It excludes static routing references and resets transient caches on restore. The new collector restores a pre-projection boundary, protecting the actual state from scoring cache writes. Production differential testing is still required; I found no source basis to assert a specific RNG leak in this deterministic event engine.
4. **Information separation:** the inspected feature extractor reads released `_meta_observed`, active/pending work, physical fleet/queue state, configured horizon and public graph/prior summaries. Scenario seeds and hashes are stored as provenance, not passed as features. The full future scenario is present in the runtime for event release, but I found no direct future order/count access in the feature path. Name-based checks alone would not establish this; the observed data flow is the meaningful safeguard.
5. **City separation:** eight development cities and four final-only cities are hard-coded and rechecked. Train/validation generation uses disjoint deterministic seeds and checks workload identities; neither learner fits validation/test rows. Do not reuse the old six-city “test” terminology for this experiment.
6. **Fitting math:** row targets are per-robot costs minus a state-wise common mean, preserving action ordering. Per-episode inverse-row-count weights sum to equal episode mass; weighted feature/NN target statistics use training rows. XGBoost rescales sample weights to mean one. The inspected ReLU derivatives, weighted MSE normalization, Adam bias corrections and NN inverse target transform are algebraically consistent. This is source analysis, not a passed gradient/fitting test.
7. **Parallelism:** branches are sequential inside an episode; idle/MLE execution and shared-index preparation are configured inline. There is one outer episode pool rather than a tenfold nested candidate pool. Activation checks the imported backend package path.

## Improvements worth prioritizing

### Correctness and evidence first

- Fix F1 and F2 before labeling at scale. Keep old artifacts; changing projector semantics requires a new compatible label identity.
- Add the small integration and failure-injection tests below. More synthetic imports or planner output will not resolve production semantics.
- Preserve per-decision model hashes, actor intents and trajectory boundaries. Log local regret and teacher margin for diagnosis, never as a guarantee of full-episode optimality.
- Lock dependency versions for the eventual executable environment. The declared minimum requirements are not a numerical reproducibility lock; the bundle records Python 3.14.4 and CPU XGBoost 3.4.1, while the new fit path has not been exercised. Set numerical thread limits **before** importing numerical libraries in the coordinator as well as workers; the first `_publish()` enters `threadpool_limits()` before `fit()` may first import those libraries.

### Training speed without changing the policy objective

- `_rows()` reparses all JSON rows for each family on repeated scans, and each update refits the entire history. Maintain an append-only content-hashed replay index plus a stable consumed-row watermark. Retain all raw rows for audit.
- `_publish()` runs synchronously inside the pool's progress callback. During a long refit the coordinator cannot replenish freed episode slots or report new failures. Move fitting to a bounded, separately budgeted fit worker, using an immutable replay snapshot and atomic publication. Avoid two uncontrolled fits multiplying BLAS/XGBoost threads.
- Cache immutable NN weight arrays in `Predictor.__init__`; `choose()` currently converts every layer's JSON lists to arrays on each decision.
- Profile projection routing before optimizing it: memoize repeated exact distance/route queries within a projection where safe, batch distinct sources, and retain exact float64/tie behavior. Pending-order sorting evaluates pickup distances for every eligible robot, so it can dominate in backlog states.
- Consider per-candidate committed branch receipts to avoid recomputing all earlier candidates after an interruption. This is an optimization only if identity and chosen-boundary transaction guarantees remain intact.
- Keep retraining-from-scratch as the correctness reference. Warm starts require deliberate handling of changing feature/target scalers, replay weights and optimizer state; do not simply continue an old NN optimizer or booster under newly scaled data.

### Model performance and experiment quality

- Validate **ranking/regret**, not only cost MSE. The ten-action portfolio includes potentially very costly actions; centering by its mean preserves exact rankings but squared-error fitting can still be dominated by extreme action differences. Compare a reference-policy advantage or ranking-aware objective using only development data.
- The Reactive Insertion bootstrap provides counterfactual actions but not diverse actual next-state trajectories. Measure chosen-policy entropy, fraction of episodes/states collected under bootstrap, model-version adoption and non-tied teacher labels. If coverage is poor, consider an explicitly specified, seeded exploratory actor that still chooses **before** labels; that changes experiment design and needs authorization.
- Use an inner episode-grouped training split for early stopping/model diagnostics if needed; do not repeatedly tune on the reserved final cities. Monitor performance by city/profile/load/importance and compare with a strong fixed baseline.
- Add spatial fleet-versus-demand mismatch and route/remaining-work summaries if diagnostics show perceptual aliasing. Every added feature must remain observable and its schema/version recorded.
- Measure whether charging-pressure and shift profiles actually produce queues, backlog and useful action disagreements. Lower batteries or a “surge” name alone do not establish stressful workloads.
- Do not import old six-action rollout labels or old full-run static outcomes as if they were compatible new projected labels. Their targets and behavior distributions differ.

## Historical evidence: what it supports and what it does not

- The bundled heavy guard pilot has 18 completed cases. Unrestricted switching gives many more switches but wins only two of six workloads and loses four. This motivates checking teacher quality and decision ranking; it does not prove the new objective fixes them.
- The bundled static summary is explicitly partial: 27/48 logical evaluations. Trees choosing Reactive Insertion everywhere, sometimes through prediction ties, do not establish learned adaptation or a final ranking.
- The original dynamic comparison distinguishes training-seen cities from unseen ones. Paris, New Delhi, Johannesburg and Moscow were in its training scope. They cannot be cited as unseen-city generalization evidence for that original model. The new reserved Cambridge UK/Ghent/Salamanca/Lucca split is separate.
- The original comparison labels a best available completed policy as “hindsight best.” With missing benchmark runs, that is only the best **observed** fixed comparator, not a complete oracle. A selector beating it on an incompletely covered day does not prove it beat all fixed policies.
- The new build receipt expressly reports no production runtime, new fit or simulation. Thus the 6–22-hour planning band / 24–36-hour allowance is conditional arithmetic, not measured evidence. Validate branch, projection, startup, replay/refit and drain costs separately before giving a completion estimate.

## Open questions for the implementing AI / experiment owner

1. Is exact service/travel/tie event ordering part of the projector's promised semantics? If not, what approximation tolerance and ranking-error budget is acceptable?
2. Does “restart friendly” include VM/power loss, or only interruption of a worker process? That determines the required F4 durability contract.
3. Which complete identity should define an experiment, including base weights and exact-runtime operations? Which changes intentionally preserve label compatibility?
4. Should pending terminal completion freeze all background charging even when the active dispatch policy could interrupt it? Is that deliberate conservative bias acceptable across all ten policies?
5. What measurable benefit over Reactive Insertion / a development-selected fixed policy must the projected teacher show before full collection? What is the maximum permitted wall-clock/RAM budget for that gate?
6. Must the 64 GiB reserve be continuously enforced under coexistence with other VM jobs, and by what bounded stop/pause mechanism?
7. Which metric selects between XGBoost and NN: raw total loss, equal-city per-robot loss, a prespecified regret measure, or a combination? Fix this on development data before final testing.
8. What is the planned matched fixed-benchmark runner and reporting artifact for the four final cities? `evaluate()` alone does not provide it.
9. Can the base reservation weights' original training-city provenance be documented explicitly? The bundled weight metadata identifies their old datasets but does not contain the entire original target dataset. No held-out leakage was established by this review.

## Proposed validation plan — only after separate execution authorization

Nothing below was run during this review.

### Gate A: pure fixtures and transaction tests

- Add F1 event-order fixtures, joint-station propagation with multi-port and repeated visits, and future-only queue-wait accounting.
- Cover elapsed waiting before the boundary, exact-boundary dropoff, partial pickup/dropoff handling, current active/queued charging, and assigned plus unassigned order coverage.
- Check per-episode fitting weights and train-only normalization by hand-computable examples. Verify NN gradients by finite differences and both predictors' serialized inference against their fitted objects on a tiny fixture.
- Inject interruptions after intent, each candidate, chosen-state write, row write, checkpoint replacement and before state retirement. Publish a new actor during recovery; the retained intent must still win. Resume multiple consecutive steps, not just one.
- Mutate every scientifically relevant dependency/input and test rejection. Test a mismatched frozen model and accidental train-after-test invocation.

**Pass:** exact accounting/invariants and the documented numerical tolerance hold; no teacher-selected transition, duplicate row, invalid reuse or unreported incomplete work.

### Gate B: bounded production integration

- Start with the existing tiny vendored fixed-policy fixtures, then the new adapter with varied fleet size/battery. Exercise all ten policies, uninterrupted and with 15-minute snapshots/restores.
- Compare fixed-policy loss, delivered IDs/counts, on-time counts, service/charge timing and physical commitments with matched original execution.
- Compare projected frozen itineraries against an execution constrained to the same itineraries and no new arrivals. Separately measure the approximation gap to a replanning completion policy.
- Permute branch evaluation order; chosen boundary and per-action costs must be invariant. Check robot/controller aliasing and no persistent scoring-induced mutation.

**Pass:** zero unexplained physical-state/accounting mismatch. Any intentional approximation is reported separately from numerical error.

### Gate C: small development-only learning pilot

- Use a prespecified small sample of fresh development workloads across an easy map and a larger map, including ordinary, spatial/priority shift and charging pressure. Keep the four final cities unopened for selection.
- Start with low concurrency and a fixed wall-clock budget. Measure cache startup, per-branch projection cost, peak RSS, trainer/replay memory, actual queues/backlog, model adoption and post-horizon drain length.
- Measure projected-teacher versus reference action ranking on sampled states, and frozen learned-selector versus Reactive Insertion and a training-selected strong fixed policy on matching validation episodes.
- Define the operationally meaningful loss improvement and acceptable worst-case regressions **before** examining outcomes. More switches or lower training MSE is not a pass condition.

**Pass:** reliable physical integration, useful out-of-sample action-ranking signal and enough full-episode evidence to justify expanding collection. A small pilot cannot establish high-probability or universal superiority.

### Gate D: scale only after the earlier gates pass

Run the intended actor-led collection with measured worker admission, immutable replay/model identities and recoverable checkpoints. Freeze model choice using development validation, then prepare the four held-out cities and run selectors plus matched fixed baselines. Publish paired scenario differences, per-city/raw/equal-city normalized results, uncertainty clustered at the episode/city level as appropriate, all failures and coverage, policy occupancy/switches, and mission-only charger waits.

Four held-out cities support a limited generalization claim, not a universal claim. Compare the best single fixed policy selected on development data separately from the hindsight best fixed policy per scenario, and never silently exclude failed or missing comparisons.

## Recommended immediate decision

**Fix F1/F2, strengthen the failure/resume contract, then request authorization for Gates A/B and a bounded development pilot.** Preserve the old artifacts and the actor-chosen transition design. Do not spend another large simulation budget until the new projected teacher and production adapter demonstrate useful, trustworthy rankings.

### Pinned source links

- [New collector](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/collection.py)
- [Projection calendar](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/projection.py)
- [Production projection adapter](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/backend.py)
- [Workflow](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py)
- [Fitting and inference](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/models.py)
- [Dataset and identity guards](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/data.py)
- [Pinned simulator runtime](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py)
- [Build evidence](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/evidence/build/BUILD_VERIFICATION.json)
