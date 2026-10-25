# Switching meta-model: read-only implementation review

Date: 25 October 2026  
Repository: [NoamAvital8/dynamic-delivery-robot-fleet](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet)  
Branch: `codex/small-city-switching-review-20261025`  
Reviewed revision: **`2ffd002a480cef66b56ecc0ce73316a4b2ee3b33`**  
Pinned simulator dependency: **`5e76f5384aec98e9d5a538b27dae205df301fa28`**, as exported under `review_bundle/backend/`.

## Assessment

**The core model-owned learning loop is implemented as intended, but this version is not ready for the full training campaign.** Fix the reused-worker lifetime defect and experiment-identity gap first. Address the projection tie-order mismatch and operational safeguards, then run a small, separately authorized integration pilot before committing to 1,600 training episodes.

I found no source-demonstrated substitution of the teacher's best branch for the actor's chosen branch, no direct future-order leakage in the inspected feature/prior path, and no obvious NN backpropagation error. These are bounded static conclusions, not runtime certifications. The largest scientific uncertainty remains whether the frozen-itinerary target ranks actions usefully for completed episode loss.

### Scope and evidence boundaries

- Read the supplied `PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md` completely before reviewing implementation. Its launch examples and historical recommendations were treated as context, not authorization.
- Inspected the new collector, adapter, projector, models, data preparation, configuration, workflow, launcher, tests, portable helper and requirements; traced relevant paths into the **bundled** simulator, generated-policy source patches, routing/cache hooks, features, reservation controllers and physical event engine. The older top-level simulator was not substituted for this dependency.
- Inspected existing build receipts, map/base-model metadata, historical dynamic records and comparison reports, static-selector evidence, and threshold-pilot evidence.
- **No project code or tests were executed or imported.** No models were fitted; no scenarios, labels, simulations or VM jobs were started. No VM access, repository edits, commits, pushes or PRs were made. Inspection utilities only retrieved/read existing files, parsed existing artifacts and computed checksums. This report is the sole deliverable.
- The branch reference resolved to the full revision above. After a network failure during clone, the exact revision was retrieved through GitHub's tree/raw interfaces: **all 319 file contents matched their Git blob hashes**. Consequently there is no successful local checkout `HEAD` to report; the revision is established by the remote reference and commit-addressed content, not a claimed `git rev-parse` result.
- Independently checked **177 export SHA-256 values and 18 staged-build source hashes**, all matching. Read all 12 compressed maps without writing inflated maps: their raw hashes matched, totaling **204,739,704 bytes**. XML node/edge/charger/cluster counts matched the manifest for all 12; all are undirected. The prepared New Delhi map has the documented 25 chargers.

“Confirmed” below means the defect follows from inspected source and a concrete failure path. Hand-derived examples were **not executed**. P1 should block the full campaign; P2 should be resolved or explicitly bounded before relying on its results. Risks and scientific limitations are separate from confirmed defects.

## Prioritized confirmed findings

| ID | Priority | Finding | Main consequence |
|---|---|---|---|
| F1 | P1 | Reused workers retain episode graphs/oracles and stack initialization wrappers | Growing memory and initialization cost; possible eventual worker failure |
| F2 | P1 | Campaign identity omits behavior-changing dependencies | Incompatible labels/actor histories can enter the same replay campaign |
| F3 | P2 | Projection changes equal-time physical event ordering | Wrong FIFO allocation can change loss and teacher rankings |
| F4 | P2 | Worker failure does not stop running peers | Expensive work continues while the controller waits to report failure |
| F5 | P2 | Training-record hash authenticates row IDs, not training contents | Changed labels/features can retain the same recorded training identity |
| F6 | P2 | First-fit thread limiter is entered before native libraries load | Coordinator's initial fit can escape the advertised thread budget |

### F1 — Simulator lifetime is unsafe for reused episode workers

**References:** [`run_nyc_benchmark2_fast.py:45–52,99–114`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark2_fast.py#L45-L114), [`routing.py:46–59`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/routing.py#L46-L59), [`exact_runtime.py:87–91`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/operations/exact_runtime.py#L87-L91), [`FleetRuntime.close():714–719`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py#L714-L719), [`workflow._pool():88–99`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L88-L99).

Each runtime's CSR hook stores a strong reference in `_oracle_by_graph_id[id(self.graph)] = self.oracle`. The oracle retains its graph, CSR and caches. There is no registry removal in the bundled backend, including `FleetRuntime.close()`. The shared-index path still invokes this hook. The process pool reuses workers without a task-count lifetime limit.

**Failure case:** one worker completes several episodes, even for the same city. Every construction reads a new graph object and inserts another registry entry. Closing an episode leaves those graphs reachable. The eight-entry `_distance_rows` limit does not bound this separate registry. Therefore the 1.5-GiB prospective worker allowance cannot be treated as a stable lifetime footprint. The actual retained bytes and time to memory pressure are unmeasured.

There is a second lifetime defect: each construction calls the installers at [`run_nyc_benchmark5.py:273–274`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark5.py#L273-L274). `_install_in_memory_graph_initialization()` captures the **current** `nx.read_graphml` and wraps it again; every previous wrapper repeats normalization and charger scans. The CSR and nearest-target hooks also wrap previously installed methods. Initialization cost grows with worker history; sufficiently long single-worker campaigns can encounter recursion limits.

**Fix:** make installation idempotent; give runtime-owned registrations explicit cleanup or weak-reference ownership. Remove only the closing runtime's cache entries. If sharing graphs deliberately, use a bounded cache keyed by verified immutable graph identity. Recycling workers after a small number of jobs is useful containment, but does not replace cleanup and should be measured against index/startup cost.

**Validation:** repeatedly construct/close runtimes in one process, first without episodes and then across completed episodes. Check registry size, weak-reference reclamation, stable wrapper identity/depth, and RSS after cleanup. Expect a bounded plateau rather than linear graph retention. Include several cities and a one-worker configuration.

### F2 — Combined identity guards do not prevent mixed scientific dependencies

**References:** [`data.science_id():15–23`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/data.py#L15-L23), [`load_dataset():107–132`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/data.py#L107-L132), [`collect_episode():51–60`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/collection.py#L51-L60), [`workflow._rows/_publish():51–76`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L51-L76).

`science_id()` hashes backend package/scripts plus the new adapter/projector, but omits the fixed reservation weights, backend `operations/exact_runtime.py`, and new data/collection/model/workflow/config implementation. Some omissions could reasonably be separate identities; currently their combined checks are insufficient.

**Failure case:** finish some episodes, change the spatial reservation NPZ, then resume. The dataset science ID remains unchanged. Completed receipts return before runtime construction; new episodes use the changed weights. `_rows()` accepts both because science and family match. Their different teachers enter one fit. This is a demonstrated guard hole, **not a claim that the supplied artifacts already contain mixed labels**.

The runtime's input digest includes base weights and protects incompatible **snapshot restores** ([runtime.py:395–397,689–693](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py#L395-L397)). That does not protect completed replay or fresh episodes. The handoff explicitly flagged this concern; the other guards do not close it.

**Fix:** persist an immutable campaign manifest with graph/scenario identities, base-weight hashes, generator and behavior-affecting source identities, scientific settings and environment versions. Bind it to jobs, rows, model publication and freeze receipts; validate before cached-result reuse. Separate **label compatibility**, **learner configuration**, and **execution provenance**, so an intentional compatible refit does not unnecessarily invalidate expensive labels. Include learner identity in model version names; `_publish()` currently derives the version from science, family and row IDs only.

**Validation:** mutate each dependency independently and test completed, incomplete and unstarted jobs. All incompatible paths must reject before fitting or collection. Explicitly test changed base weights and `operations/exact_runtime.py`, not only changed adapter code.

### F3 — Joint projection does not preserve known equal-time FIFO ordering

**References:** [`projection.project():91–151`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/projection.py#L91-L151), [`physical event ordering:348–355`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark5.py#L348-L355), [`movement/service priorities:1224–1285`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark5.py#L1224-L1285).

The physical engine retains `(time, priority, event_counter)` ordering. The projector starts robots in sorted ID order and manufactures a new counter. It also gives handling and travel completions the same priority, whereas physical handling completion has priority 0 and movement priority 1. Existing queued requests retain their list order, but equal-time **incoming** requests need not retain their physical order.

**Hand-derived failure case, not an observed trace:** a two-port station has one background session occupying a port until minute 30. Robots 1 and 2 are already on their final incoming edges to the station, both arriving at minute 5; the saved heap orders robot 2 first. Use the default medium-robot speed/energy values, 4 m/s and 0.075 Wh/m. Each arrives with zero battery and has a carried order whose destination is 1,200 m away, with that station also the destination's nearest charger. Charging 180 Wh covers the 90-Wh outward trip plus 90-Wh reserve: 5.4 minutes at 2,000 W, followed by five minutes travel and one minute dropoff handling. Both orders were released at time zero; importance is 1 for robot 1 and 5 for robot 2, with deadlines after minute 21.8. Physical order gives `r2=16.4, r1=21.8`, loss `5×16.4 + 21.8 = 103.8`. Projector ID order gives `r1=16.4, r2=21.8`, loss `16.4 + 5×21.8 = 125.4`. A competing action with cost between these values can be ranked incorrectly. Frequency in real workloads is unmeasured.

**Fix:** carry known event priorities and tie ordering into boundary-to-itinerary conversion. Define consistent ordering for newly synthesized pending work and later events. If exact tie fidelity is deliberately sacrificed, document it as a projection approximation and measure sensitivity instead of implying complete physical FIFO fidelity.

**Validation:** reversed saved arrival sequence with equal ETAs, occupied ports, unequal importance/charge durations, and simultaneous handling, movement and charge completion. Existing projector fixtures cannot catch this because they do not include the source event heap's sequence.

### F4 — Failure reporting waits for other running episodes to finish

**References:** [`workflow._pool():88–111`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L88-L111), [`collection.py:78–146`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/collection.py#L78-L146).

When one future raises, `_pool()` calls `cancel()` on peers and raises inside the executor context. Cancellation does not stop already running work, and context exit waits for executor shutdown. Peers have no cooperative stop signal. A failed campaign can therefore continue up to 63 other episodes and their checkpoint writes before returning the error. The default post-horizon allowance is 10,080 simulated minutes; it is not a wall-clock cancellation bound.

**Fix:** choose and implement an explicit policy: fail promptly with cooperative peer cancellation, or deliberately continue peers and report failures without pretending to stop. For fail-fast behavior, check a shared stop signal at safe branch/decision boundaries, retain valid committed transactions, and emit cancelled/resumable receipts. Define a bounded fallback for an unresponsive worker.

**Validation:** a stub job fails while another remains long-running. Verify bounded error reporting, no uncontrolled continued collection, and valid resume state. Include failure during a synchronous fit, which currently also delays coordinator attention to worker failures.

### F5 — “Training-record hash” does not hash training data contents

**References:** [`collection.py:130–138`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/collection.py#L130-L138), [`models.check_rows()/fit():12–58`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/models.py#L12-L58), [`workflow._rows():51–58`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L51-L58).

A row ID hashes job signature, step and pre-state hash. `training_records_sha256` hashes the sorted row IDs, not features or labels. Replay loading checks science/family, then schema and finite values, but does not verify full-row content or membership in the active dataset.

**Failure case:** a finite cost or feature is changed while its row ID remains unchanged. It passes the fitting checks and keeps the recorded training-record identity. An existing model-file comparison may detect a changed resulting payload **after fitting**, but this is not replay integrity verification and does not protect a first publication.

**Fix:** keep a stable row key and a separate canonical content digest covering features, all ten costs, input identity, intent and boundary hash. Build fit provenance from these content digests. Validate each row against the active episode/job manifest; reject orphan, incompatible and duplicate records. Enforce expected city/split cardinalities as well—`load_dataset()` currently does not enforce them.

**Validation:** modify a finite cost/feature without changing the key; insert a same-science row from an absent episode; remove a required city from the manifest. Reject before fitting. No supplied-row corruption was observed; this finding concerns the claimed integrity guarantee.

### F6 — The coordinator's first native fit may escape the thread limit

**References:** [`workflow._publish():61–66`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L61-L66), [`models.fit():33–34,60–102`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/models.py#L33-L102), [threadpoolctl implementation/documentation](https://github.com/joblib/threadpoolctl/blob/master/threadpoolctl.py).

`_publish()` enters `threadpool_limits()` before `fit()` imports NumPy. The limiter controls native libraries already loaded when it builds its controller. A fresh parent process has no guaranteed prior NumPy import, so newly loaded BLAS can escape the first context. An NN-first publication is a concrete path. XGBoost has explicit `n_jobs`; worker initialization also sets thread environment variables before numerical imports. Those do not establish the coordinator's first BLAS limit.

**Fix:** initialize parent numerical libraries before entering the limiter, or set parent BLAS/OpenMP environment limits before any numerical import; retain explicit XGBoost limits. Capture the effective library/thread configuration with the run manifest.

**Validation:** cold-process first fit for each family, including NN reaching the threshold first. Record `threadpool_info()` and process CPU use. Actual oversubscription was not measured in this review.

## Suspected risks and missing safeguards

### R1 — Prepared route legs lack the physical engine's compatibility checks (P2)

[`backend.completion_projection():132–149`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/backend.py#L132-L149) consumes `prepared[stop].phases` without checking the cached step's start node or battery. Physical [`continue_delivery():1313–1326`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark5.py#L1313-L1326) checks both before reuse. The adapter reconstructs missing legs using a single final target; physical [`compute_first_step():1237–1260`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/scripts/run_nyc_benchmark5.py#L1237-L1260) uses next-distinct-stop lookahead.

A reconstructed leg could arrive with battery 20 while its cached successor assumed 50 and spends 30. Projection then rejects an otherwise re-plannable suffix, or a mismatched start node can yield misleading travel arithmetic. **A reachable production snapshot with this mismatch was not established**; normal assignments populate prepared steps. Add start-state checks, define reconstruction semantics, and rebuild an incompatible suffix consistently. Test a missing nonterminal leg followed by a stale cached leg.

### R2 — Production isolation and full order conservation remain unproved (P2)

The collector restores one complete runtime snapshot before each candidate and saves boundaries before projection. The pinned runtime preserves aliasing, event counter, queues, schedules, controllers, observations and scenario; it clears selected transient caches on restore. No concrete actor-state contamination defect was identified.

However, exact-routing objects and some caches are shared references outside the snapshot. Their intended observational equivalence needs **candidate-permutation and repeated-branch tests using the new adapter**, including cache-hit/miss paths. Existing new collector tests use a small JSON-state stub. Legacy runtime tests are relevant specifications, not evidence that this new path was exercised.

`project()` proves equality between produced dropoffs and its supplied `orders` dictionary. The adapter builds that dictionary from pending/scheduled work; it does not independently reconcile it with all released orders minus delivered orders. A lost order absent from both structures would be invisible to that check. No lost-order occurrence was found. Add an independent conservation invariant: released = delivered + pending + assigned, disjoint by ID; also verify pickup-before-dropoff and payload feasibility when reconstructing routes.

### R3 — Final-test lifecycle and fixed-model provenance need stronger custody (P2)

[`prepare(test=True):35–40,101–103`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/data.py#L35-L40) checks the current frozen artifacts, but the test dataset manifest does not bind their hashes into an immutable freeze receipt. [`train()/evaluate()`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/workflow.py#L117-L190) use separate locks and there is no campaign-level “final test opened” barrier. Existing completed evaluation signatures help reject changed models for those jobs, but do not enforce the entire study lifecycle.

Bind test preparation to model hashes, development evidence and learner configuration; block subsequent development mutation within that final-test campaign. If previously exposed cities are reused in another study, disclose that exposure. This is a custody weakness, **not evidence of current leakage**: the new test inputs/results do not yet exist.

The two reservation `.training.json` files provide training settings/losses and VM dataset paths, but not a source/workload/city roster sufficient to independently prove their exclusion of the four final cities. Obtain that provenance before making an “untouched by any learned component” claim. Sharing the same fixed dependencies across selectors and benchmarks makes comparisons more controlled, but does not establish unseen-geography provenance.

### R4 — Resume support is narrower than durable, fully replayable history (P2/P3)

[`collect_episode():70–150`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/collection.py#L70-L150) correctly reuses a saved intent and commits boundary → row → checkpoint. Still:

- The first pre-decision snapshot is initially memory-only. A crash after its intent but before the first committed checkpoint reconstructs the initial runtime and requires the regenerated pickle bytes to match. Cross-process first-intent recovery is not established by the JSON stub test. Persist the initial snapshot before publishing the first intent.
- Prior boundaries are deleted after checkpoint replacement, leaving hashes/rows without every historical physical snapshot. This is a reasonable disk-saving resume strategy, but does not provide random-access replay of every decision. State the retention policy and preserve selected diagnostic snapshots if needed.
- JSON/state writes use rename without file/directory `fsync`. Ordinary process interruption and machine/power-loss durability are different guarantees. Power-loss recovery needs a durable ordering policy before deletion of the preceding state.

Test interruptions after intent, boundary, row, checkpoint and completion publication, including a newer model published between attempts. Verify chosen action, physical state and accumulated loss; do not validate only the final scalar loss.

## Intended design: what the source supports

| Requirement | Static assessment |
|---|---|
| Actor chooses before seeing labels | Implemented: intent is written before branch scoring; a saved intent takes precedence on resume (`collection.py:88–100`). |
| Actor action determines the actual next state | Implemented: only its pre-projection boundary is restored (`collection.py:113–139`); teacher argmin does not drive state. |
| Same exogenous scenario across branches | Implemented through restoring the same snapshot/scenario and event queue. No fresh demand sampling inside collection was found. |
| Realized increment plus unfinished full loss | Implemented (`collection.py:118–125`): delivered-window loss is added once; projected costs use request-to-delivery duration and deadline allowance. Final episode receipt uses runtime loss, not a sum of overlapping labels. |
| Charging, queues and pending work | Active sessions, stored FIFO waiters, known incoming background charging, remaining handling/edge travel, and cross-station delay propagation are represented. Pending work is appended and jointly reprojected. F3/R1/R2 limit fidelity claims. |
| Full delivery completion | `runtime.outstanding()` includes future and unfinished orders; receipt rejects `delivered != orders`. Drain overflow fails rather than reporting partial loss as success. |
| Separate learner trajectories | Separate family directories and family checks are present. Inputs are shared, selected states are not. |
| Train/validation/test separation | Explicit city allowlists/roles, separate seeds/workloads, and fitting rejection of non-training rows are present. Normal development preparation selects only development maps. |
| Observable features and priors | Features use observed arrivals, pending/assigned work and public map/fleet/horizon state. Demand is updated on actual arrival. New priors are configured fleet-scaled constants, not realized future counts. No direct future-order leak was found. Map summaries can nevertheless identify development cities indirectly. |
| Fitting mathematics | Per-robot cost normalization and within-state centering preserve ranking. Train-only episode-weighted feature normalization is implemented. XGBoost weights have mean one; NN weighted gradients, layer derivatives and Adam correction appear internally coherent. |
| Frozen evaluation | An explicit frozen model is required for non-training episodes; only the chosen action runs, with no counterfactual labels fed back into fitting. Final fitting occurs before validation. |
| Parallelism | Candidate branches are sequential. Worker numerical limits and inline one-worker index construction avoid nested branch/index process pools. F1/F4/F6 and measured memory remain important exceptions to resource assurances. |

References for the feature/model conclusions: [`features.py:62–143`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/meta_policy/features.py#L62-L143), [`runtime._observe():535–544`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/backend/src/delivery_fleet/meta_policy/runtime.py#L535-L544), [`models.py:33–105`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/models.py#L33-L105).

A checked hypothesis was rejected: reducing initial batteries after runtime construction does not currently preserve an initially full-battery MLE reservation assignment, because the MLE time-zero gate returns no assignment before using those snapshots. Collection also takes a fresh snapshot after battery adjustment. Initializing batteries inside the fleet factory would be cleaner, but this was not promoted to a bug.

## Scientific limitations and model-performance improvements

### L1 — The terminal target can reward locally cheap but strategically poor states

No-future-arrival completion intentionally omits later demand, coverage and reserve-battery value. It can undervalue relocation/reservation even when those actions help later. Conversely, retaining background charging commitments can overprice pending work compared with a future dispatcher that interrupts charging. The common pending rule is deterministic, but its five-feasible-candidate shortlist and no-new-queue selection are approximations; a final joint reproject corrects realized queue interactions **after assignment**, not the assignment decision itself.

Measure target usefulness before increasing model size: on a small, preselected development set, compare projected action differences with independently completed continuations and measure ranking errors/regret by queue pressure, pending count, importance and profile. Keep exact frozen-plan arithmetic validation separate from predictive accuracy under actual replanning. Only then consider development-only ablations of a longer lookahead, a learned terminal value, or a modest future-coverage feature/target. Such changes define a new label version.

### L2 — Aggregate features hide action-relevant spatial relationships

The extractor records counts, battery summaries, aggregate queue pressure, slack moments and pickup entropy, but not robot-to-demand spatial alignment or remaining route workload. Two states on one map can have identical current features while free robots are near versus far from a hotspot. Additional trees cannot recover information absent from their input.

Prioritize observable, city-independent features: demand-versus-ready-capacity imbalance by cluster; travel-to-pickup/charger quantiles; remaining route/service workload; slack quantiles by importance; concentrated station congestion. Use only released observations and known commitments. Test incremental value on development validation and leave-one-development-city-out splits; eight city fingerprints do not establish geographic generalization.

### L3 — Greedy cold start and asynchronous replay constrain coverage and reproducibility

All-action labels at visited states are valuable, but deterministic Reactive Insertion bootstrap plus greedy argmin does not explore alternative multi-step trajectories. Accumulating old actor versions is intended replay, not an on-policy stationary dataset. Fast cities and shorter episodes can enter early fits sooner; asynchronous timing can alter the learned trajectories of otherwise seeded runs.

Record actor-version exposure, policy occupancy, profile/city coverage and publication timing. Preserve all original action intents. Compare the greedy baseline with a bounded, explicitly specified exploration or balanced-bootstrap variant only in development and only while the actor's actual action still owns the next state. For a scientific comparison of model families, repeat seeds/schedules or document the asynchronous schedule as part of the treatment.

### L4 — Cost MSE and equal episode weight are useful choices, not the final objective

Centering removes a common within-state offset but does not guarantee correct ordering of close actions. Per-robot normalization, equal episode weights and partial episodes during online collection define a different fitting emphasis from raw total campaign loss. A newly started episode's few rows initially receive the same total weight as a much longer completed episode; this should be intentional and monitored.

Report chosen-action regret, action-gap calibration, per-importance tail loss and completed-episode loss, not only regression MSE or switch count. Consider robust/ranking-aware losses and validation-based fit stopping after label fidelity is established. Exact ties currently select the first policy, `myopic_ab` (`models.py:151`), rather than the Reactive Insertion bootstrap; measure tie frequency and consider incumbent/default tie handling. Tune any margin or hysteresis only on development data.

### L5 — The experiment still lacks its matched fixed-policy evaluation campaign

`workflow.evaluate()` runs two selectors only. It does not run all ten matched fixed policies or produce the complete statistical comparison. Add this before claiming a win. Match graph/scenario bytes, fleet composition/count, initial battery, priors, base weights and physical semantics. Select the deployable best fixed baseline using development data; report hindsight best fixed per scenario separately. A sparse set of completed fixed baselines is not an all-policy hindsight oracle.

Use paired episode differences, per-city results, raw total loss and equal-city/per-robot summaries, uncertainty, completion/failure counts, and common completed coverage. Treat city-level generalization cautiously with only four final cities. Keep the synthetic demand distribution's limits visible, including uniformly sampled dropoffs; broader urban realism is not established by varied profiles alone.

## Training-speed and resource improvements, in order

1. **Fix F1 before increasing concurrency.** Measure process lifetime memory, not just first-episode RSS. Check actual available memory alongside initialization and fit peaks. The current admission ramp does not explicitly reserve the latent RSS of already submitted initializations; the 64-GiB reserve is a heuristic admission threshold, not a hard memory limit.
2. **Eliminate repeated shortest-path rows in pending shortlisting.** [`backend.py:160–161`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/src/fleet_switching/backend.py#L160-L161) ranks each pending order by requesting distances from every candidate tail. With up to 80 robots and an eight-row LRU, this can thrash. All bundled graphs are undirected: compute one row from the pickup and query every tail, preserving exact rankings. Use a reverse-graph approach if directed maps are later supported. Validate parity before timing.
3. **Ingest immutable replay incrementally.** `_rows()` reparses all JSON records roughly every 15 seconds, including diagnostics unused by fitting. Use an indexed append-only catalog and compact numeric training arrays, while retaining original evidence. Bind each fit to a content-hashed snapshot, not a directory changing underneath it.
4. **Separate fitting from coordinator responsiveness.** `_publish()` runs synchronously inside the progress callback; existing workers continue, but job refill and error handling wait. Use one bounded fit process with explicit thread/memory budget and atomic publication. Coalesce update requests; never expose a partially written model or silently fit a moving dataset.
5. **Measure startup separately from branches and projection.** Reuse verified read-only routing indexes, serialize cold index initialization where useful, and account for lock waits and coordinator memory. Profile constructor, snapshot/restore, each policy's 15-minute branch, terminal projection, replay I/O and refit separately. Preserve exact semantics when caching.
6. **Do not trade correctness for label volume prematurely.** Branch deduplication is safe only after proving equality of all relevant physical/controller state and realized loss. Changing candidate coverage, sample frequency, horizon or exploration changes the protocol and needs development validation plus a new identity where appropriate.

The 0.7–2.5 seconds/branch estimate and 24–36-hour first-campaign budget remain assumptions. Nothing inspected establishes new-projector throughput or supports a speedup factor for these suggestions.

## Existing evidence: what it does and does not establish

- [`BUILD_VERIFICATION.json`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/evidence/build/BUILD_VERIFICATION.json) and [`PORTABLE_REVIEW_VERIFICATION.json`](https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/2ffd002a480cef66b56ecc0ce73316a4b2ee3b33/review_bundle/evidence/build/PORTABLE_REVIEW_VERIFICATION.json) report **31 prior synthetic passes**, with production runtime, fitting and training controller explicitly unrun. This review did not repeat those tests.
- The four new test files cover closed-plan arithmetic, four backend-shaped fixtures, guard/inference behavior and stub collection. The collector interruption test covers failure at checkpoint writing; it does not cover the complete crash matrix. Fixed-weight NN inference is not gradient/fit validation. New fitted XGBoost round-trip behavior and production runtime integration remain unverified.
- Legacy `test_meta_runtime.py`/`test_meta_fixes.py` contain useful fixed-policy equivalence, aliasing, event-counter and switching tests. Their existence is relevant, but no new execution result is claimed here, and they do not cover the new terminal projection or reused-worker lifecycle.
- Independently parsed historical archives contain **3,942 examples (3,132 train/810 validation), 23,652 rollouts, 15,462 decisions, 225 completions and 225 scenarios**, matching the handoff. These are old targets/trajectories, not compatible new training rows.
- Historical dynamic comparisons have unequal baseline denominators; the static snapshot is **27/48 complete**. The heavy threshold evidence shows that more switches did not produce a consistent gain. None establishes new-selector superiority; historical “test city” terminology must not redefine this experiment's four-city holdout.
- Portable paths and dependency activation are present in source. Minimum-version requirements and a VM inventory are not a reproducible environment lock. No installation, native-library compatibility check, helper execution or XGBoost fit was performed. Linux `/proc/meminfo` admission is intentional; portable inspection does not imply portable production training.

## Proposed validation plan — only after separate execution authorization

Run these gates in order, with a fixed time/memory budget, stop conditions and preserved failures. No stage below was executed during this review.

| Gate | Smallest useful scope | Acceptance evidence |
|---|---|---|
| 1. Identity and lifecycle | Synthetic manifest/row tampering; repeated initializer installation; worker-failure stub; all transaction crash points | Incompatible dependencies/content rejected before use; bounded stop behavior; saved actor choice survives publication/interruption; no expanding wrapper chain |
| 2. Learner correctness | Tiny synthetic fit per family in cold processes; NN finite-difference gradients before clipping; save/load prediction comparisons; unequal episode lengths | Weighted objective/scalers correct; predictions finite and identical after serialization within specified tolerance; train-only statistics; observed native thread limit |
| 3. Runtime integration | Tiny deterministic production-runtime fixtures for all ten fixed policies and selected switches; configured fleet count and low initial batteries; several sequential episodes in one worker | Matched fixed-policy loss/deliveries, preserved commitments, released-order conservation, branch-order invariance, no retained graph growth |
| 4. Projector fidelity | Partial edges; locked pickup/dropoff; queued/active charges; two ports; repeated/cross-station visits; equal-time reversed ordering; infeasible pending and missing/stale legs; exact-boundary deliveries | Hand-calculated or independent frozen-plan reference times/losses; one contribution per unfinished order; invalid states fail closed; physical tie semantics explained |
| 5. Target usefulness and throughput | Preselect a small development-only pilot spanning one cheap and one larger city, ordinary and charging/surge pressure; collect both actor streams with a strict cap | Compare projected rankings to completed continuations; record phase timings, p50/p95 branch time, memory growth, minimum available RAM, fit lag and failures; revise budget from measured data |
| 6. Frozen validation and final custody | Matched development validation including fixed benchmarks; then immutable freeze receipt; only afterward final four-city preparation/evaluation | Completed paired outcomes, uncertainty and failure coverage; no test-driven retuning; exact frozen models and dependencies recorded |

For Gate 4, a normal dispatcher that replans is **not an exact oracle for a frozen-itinerary target**. First compare against a reference with the same commitments, pending assignment and no-future-arrival semantics. Then use normal completed continuations in Gate 5 to quantify the target's predictive limitations. Likewise, repeat/permutation checks should compare physical state and event order, not only a coincidentally equal loss.

## Open questions for the implementing AI

1. What is the intended lifetime of global backend hooks/registries across pool jobs, and which component owns cleanup?
2. Which source/configuration changes intentionally preserve label compatibility, and which require a new actor/model campaign? Define these identities before resumption is supported across changes.
3. Should terminal equal-time ties reproduce physical ordering exactly? If not, what ranking-error tolerance is acceptable?
4. Are missing/stale prepared legs reachable in production switching states? Supply a captured development boundary or an invariant proving when the fallback can occur.
5. Can the base reservation training data be traced to cities/workloads and verified against the four held-out cities?
6. Is interruption recovery required only for process crashes, or also machine/power loss? How much historical state must remain independently replayable?
7. What pilot thresholds will justify full collection: minimum target-ranking usefulness, acceptable failures, memory ceiling and measured completion budget?
8. Which complete fixed-policy campaign and statistical comparison will accompany the selector evaluation?

**Recommended next action:** implement F1/F2 and the targeted lifecycle/projection safeguards, then request authorization for Gates 1–4. Do not start the full collection campaign merely because the prior synthetic suite passed.
