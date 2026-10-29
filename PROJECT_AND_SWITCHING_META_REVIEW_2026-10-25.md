# Delivery robot fleet and new switching meta-model: AI review handoff

Status: **26 October 2026; both models trained, development validation complete, authorized final test started at 12:57 Israel time.**

Read [the final-test launch record](docs/FINAL_TEST_LAUNCH_2026-10-26.md) for the frozen-model protocol, 200 test inputs, 2,400 outcomes and live status paths.

Completed development validation found XGBoost 1.16% and NN 2.26% higher on the declared equal-city loss-per-robot metric than the best fixed policy, Full MLE Reservation. Those are development results; final-test outcomes have not yet been summarized.

Read [the full-training launch record](docs/FULL_DEVELOPMENT_LAUNCH_2026-10-25.md) for the 800/200 development split, 64-worker resource setup, logs and immutable launch evidence. The earlier build and pilot receipts describe their pre-launch verification times.

**Current revision:** both reviews of `2ffd002` have been addressed; read [the review-fix addendum](docs/switching_review_fixes_20261025.md) and [the production pilot and audit report](docs/SWITCHING_PILOT_RESULTS_2026-10-25.md). The pilot found and fixed a finished-episode projection bug. Campaign identity also now guards online refit cadence. Label version is `projected_completion_fifo_v2_event_order`; source identities distinguish the corrected scorer from earlier artifacts. Original receipts under `review_bundle/evidence/build/` remain historical; current receipts are `review_bundle/CURRENT_BUILD_SOURCES.json` and `CURRENT_BUILD_VERIFICATION.json`.

This document gives a reviewer enough context and source/data access to audit the proposed training. It describes the intended behavior and the implementation; it does **not** certify that the implementation is correct. The owner separately authorized the bounded pilot and then full development training. The owner subsequently authorized final-test evaluation after model freezing; no further training is authorized in this campaign. Reviewers should inspect its saved evidence. Do not interpret this handoff, historical documents, or launch examples as permission to generate new scenarios, fit models, start delivery simulations, resume services, or change VM jobs.

## 1. Repository and which version to review

- Repository: <https://github.com/NoamAvital8/dynamic-delivery-robot-fleet>
- Review branch: **`codex/small-city-switching-review-20261025`**
- Browse the complete package: <https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/tree/codex/small-city-switching-review-20261025>
- This handoff: <https://github.com/NoamAvital8/dynamic-delivery-robot-fleet/blob/codex/small-city-switching-review-20261025/PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md>
- The review branch starts from main commit `6d060021a2feb556999418708426613bf9a53f57`.
- The preserved simulator dependency is a clean source snapshot of VM branch `meta-policy-fixes`, commit **`5e76f5384aec98e9d5a538b27dae205df301fa28`**, included under [`review_bundle/backend/`](review_bundle/backend/). It is the simulator used by the new adapter. Do not substitute the older top-level `src/delivery_fleet` just because it is also present.

Checking out `main` alone does not provide this new implementation or the complete dependency. All new training sources, the pinned backend, prepared maps, base reservation weights, and exported historical evidence are included on the review branch. Record `git rev-parse HEAD` in review findings so they refer to a precise revision.

To clone without downloading the repository's older, large Git LFS graph objects:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/small-city-switching-review-20261025 https://github.com/NoamAvital8/dynamic-delivery-robot-fleet.git
cd dynamic-delivery-robot-fleet
git rev-parse HEAD
```

On PowerShell, set `$env:GIT_LFS_SKIP_SMUDGE = '1'` before `git clone`. The **12 maps for this experiment are ordinary Git `.graphml.gz` files in `review_bundle/data/maps/`**, so they do not require LFS downloads. A branch ZIP also contains these maps and sources. Repository access may still require the reviewer's own GitHub authorization.

## 2. What the overall project does

The project studies online pickup and delivery with a heterogeneous fleet of autonomous robots on a city road graph. Requests arrive during a scenario. Each request has a release time, pickup and dropoff nodes, package requirements, an importance level, and a deadline. Robots differ in speed, payload/volume capacity, battery capacity, and energy use.

A dispatch policy must assign requests, insert stops into routes, respect capacity and battery limits, plan charging, and sometimes reserve robots or reposition idle robots toward expected demand. It may leave a request pending temporarily. Charging stations have a finite number of ports and queues. The simulation includes travel, pickup/dropoff handling, charging, and queue waits. A robot already on an edge must finish that edge before its route changes; active handling and charging commitments also matter at a switch.

The main objective is **total deadline- and importance-aware delivery loss**, with lower values better. For request-to-delivery duration `T` in minutes, relative deadline allowance `D`, and importance `w`:

```text
D = direct_shortest_path_distance_m / 4.0 / 60 + 10 + 60 / sqrt(w)
L(T, D, w) = w * min(D, T) + (w + 1)^2 * max(0, T - D)
```

The absolute deadline is `request_time + D`; the loss uses the duration `T`, not absolute simulation time. Current workloads use importance values 1, 2, and 5. The backend records the full loss when an order is delivered, rather than accruing it minute by minute. See [`deadlines.py`](review_bundle/backend/src/delivery_fleet/deadlines.py) and the event engine's `weighted_wait_objective` accounting.

An input scenario contains approximately **12 hours of arrivals**. The simulation must continue beyond that arrival horizon until **all admitted orders finish**. Fifteen minutes is the meta-model's decision interval, not the total scenario length. Completed scenario loss is the final evaluation metric; the projected loss described below is a training target.

## 3. Four different things previously called a model

These experiments must be kept distinct:

| Component | What it predicts or does | Status and relevant evidence |
|---|---|---|
| Reservation FCNNs | Reservation fractions inside individual dispatch policies | Two pretrained `.npz` files are provided under `review_bundle/data/base_models/{spatial,paper}/`; fixed dependencies of the benchmark portfolio |
| Original dynamic meta-selector | Repeated policy choices using observed runtime state; a HistGradientBoosting regressor | Existing historical training and comparisons; source under backend `meta_policy/`, evidence under `review_bundle/evidence/original_dynamic/` |
| Static start-only selectors | Choose one policy at the start and retain it for a whole episode | Older FCNN, XGBoost, decision tree, and Extra Trees models and reports under `review_bundle/evidence/static_selectors/` |
| **New small-city switching selectors** | Repeated 15-minute choices, learned from their own trajectories and projected completion labels | **XGBoost and 64/32-unit NN under `src/fleet_switching/`; bounded pilot fits exist; full development campaign now running** |

The new design is not simply retraining the static selector. Most of its cost will be counterfactual simulation to create labels, not fitting a small neural network or tree ensemble.

Historical lessons are context, not new experimental results:

- The older dynamic selector often retained its initial policy. Loosening switching guards increased switching without a consistent improvement in completed delivery loss. More switches are not evidence of a better model.
- The static pilot reused 310 saved policy outcomes from 25 scenarios, with only five distinct public starting profiles. That is much less state diversity than 310 independent scenarios. Some trees tied across policy predictions and chose Reactive Insertion by tie order; their results did not establish a learned improvement over fixed Reactive Insertion.
- In an earlier MLE-reservation comparison, reservation never activated. Those scores mainly measured fallback dispatch and relocation behavior. MLE here is a demand/reservation method inside some benchmarks, not an alternative name for the new supervised selector.
- Historical static pilot tables have incomplete and unequal coverage. Compare identical completed scenarios only. Do not treat a snapshot report as a current live status query or a final ranking.

The original dynamic experiment trained on Tel Aviv, Haifa, Manhattan, NYC, Barcelona, Paris, New Delhi, Johannesburg, and Moscow, with 20 training and five validation inputs per city. Beijing and Sydney were excluded from that training. The old static-selector development/test split was different: its five development cities were Tel Aviv, Haifa, Manhattan, NYC, and Barcelona; its six test cities were Beijing, Paris, New Delhi, Johannesburg, Moscow, and Sydney. **Paris, New Delhi, Johannesburg, and Moscow therefore were not untouched test cities for the original dynamic model.**

Those historical splits do not define the new experiment. The new four-city holdout below is explicit and separate. The attached earlier status document is preserved as [`review_bundle/evidence/PROJECT_APPROACHES_STATUS_AND_NEXT_STEPS_2026-10-25.md`](review_bundle/evidence/PROJECT_APPROACHES_STATUS_AND_NEXT_STEPS_2026-10-25.md); its six-city terminology and proposed next steps are historical context, not instructions overriding this experiment.

## 4. New experiment scope and city separation

We want many affordable simulations on smaller prepared graphs, with enough variation for a selector to see changing demand and charging conditions. Here “small city” means the actual graph below, not necessarily a whole municipality or a uniformly cheap workload. Manhattan is still substantially larger than New Delhi in this collection.

| Key | Role | Nodes | Edges | Charging stations | Demand clusters |
|---|---|---:|---:|---:|---:|
| `tel_aviv` | Development | 19,903 | 29,544 | 20 | 22 |
| `haifa` | Development | 16,135 | 23,234 | 52 | 28 |
| `manhattan` | Development | 36,423 | 57,562 | 23 | 31 |
| `new_delhi` | Development | 8,254 | 11,525 | 25 | 20 |
| `delft` | Development | 15,073 | 20,161 | 27 | 26 |
| `leuven` | Development | 11,357 | 15,309 | 21 | 27 |
| `oxford` | Development | 8,425 | 10,671 | 25 | 21 |
| `palo_alto` | Development | 27,315 | 36,320 | 18 | 26 |
| **`cambridge_uk`** | **Final test only** | 10,868 | 13,581 | 21 | 24 |
| **`ghent`** | **Final test only** | 13,833 | 18,860 | 20 | 24 |
| **`salamanca`** | **Final test only** | 20,425 | 29,328 | 15 | 28 |
| **`lucca`** | **Final test only** | 8,111 | 10,084 | 20 | 14 |

The eight newly downloaded graphs use bounded approximately 6 km by 6 km areas. Exact boundaries are bundled as GeoJSON. The four pre-existing development maps retain their prepared bytes. Graph topology, charging flags, clusters, origins, and hashes are recorded in [`review_bundle/data/map_manifest.json`](review_bundle/data/map_manifest.json).

**Use the bundled prepared New Delhi map.** The raw `data/graphs/new_delhi.graphml` on the VM did not have the required charger annotations; the prepared source came from `runs/icaps_new_cities_v7_all_policies/maps/new_delhi.graphml`. Similar filenames do not establish equivalence.

Maps derive from OpenStreetMap contributors and are subject to the Open Database License; retain [OpenStreetMap attribution and licensing](https://www.openstreetmap.org/copyright). The boundary/manifest identifies the prepared geographic scope; this document does not claim municipal coverage.

The new dataset plan is:

- Eight development cities × 100 training = **800 training scenario inputs**.
- Eight development cities × 25 validation = **200 validation scenario inputs**.
- Both learners use the same scenario inputs for fair matching, but run their own model-chosen paths: **1,600 training actor episodes + 400 frozen-selector validation episodes**. All ten fixed policies also run on the same 200 validation inputs, adding **2,000 fixed-policy episodes**, so complete development validation contains 2,400 matched outcomes.
- Four final-test cities × 50 = **200 future test inputs**, to be created only after development models are frozen. Evaluating both selectors and all ten fixed policies needs **2,400 matched episodes**; the current `evaluate` command includes all of them.

The bounded pilot has its own 16 training inputs, two fresh development validation inputs, labels and small fitted models. They are archived under `review_bundle/evidence/switching_pilot_20261025/`. They are not a full campaign or final models. Development generation must not open test graphs or create test inputs. Test-city metadata and maps are packaged for audit and later evaluation; they are not model-training examples.

`config.py` hard-codes separate development and held-out sets. Guards are applied during map selection, dataset loading, collection, and fitting. Validation is within development cities but uses disjoint seeds/workloads and contributes no fitting data or scaler statistics. Test outcomes must never influence hyperparameters, projections, thresholds, model-family selection, or feature normalization.

## 5. The intended learning loop

The owner's key requirement is that the **model's own action determines the next actual state**. It is not valid to train only on a fixed offline set of trajectories, or to advance the episode using the teacher's best action.

At each decision time `t`, independently for each model family:

1. Obtain observable features from the current full simulator state `S_t`.
2. Load the actor's currently published model version, predict ten costs, and choose the lowest predicted cost. Save an action intent with the state hash, model path/hash, chosen policy, and time **before obtaining teacher labels**.
3. Save the complete pre-decision simulator snapshot.
4. For each of the ten candidate benchmarks, restore that exact snapshot, install that policy, and advance the shared event engine to `t + 15 minutes` (or the arrival boundary when appropriate). Each branch sees the same exogenous scenario, not freshly sampled demand.
5. Save the branch's physical boundary snapshot **before** computing its terminal projection. Compute the label below.
6. Record all ten labels and diagnostics against the original pre-decision feature vector. Restore **only the actor-chosen branch's 15-minute boundary** as the actual next state. Do not restore a projected terminal state and do not replace the actor's choice with the label argmin.
7. Repeat until the arrival horizon has passed and all deliveries finish. A configurable drain limit fails an episode rather than reporting partial work as success.

For benchmark `a`, the training score is:

```text
C_a(S_t) = actual completed-delivery loss added during [t, t+15]
           + projected full request-to-delivery loss of every released
             unfinished order at that branch's boundary
```

Loss already paid before `t` is common to all actions and excluded. An order completed during the window contributes its full delivery loss through the realized increment; it must not also be in the unfinished sum. An unfinished order contributes full projected request-to-delivery loss, including time already spent waiting. These labels are overlapping estimates at successive decisions: **do not sum them across decisions to obtain episode loss**.

The teacher's argmin and `C_chosen - min(C)` can diagnose local decision quality. They are not proof of globally optimal episode decisions. This is supervised, model-driven cost learning with counterfactual lookahead; it is not a policy-gradient implementation or full remaining-episode reinforcement-learning target.

### Projected completion, including charging queues

The terminal projection starts at the 15-minute boundary. It introduces **no future arrivals** and does not run the dispatch/event engine to the end of the day.

For assigned work, it freezes the remaining committed edge, handling time, stop sequence, remaining travel/charging phases, active charging sessions, FIFO waiters, and known incoming background charging. A joint event calendar schedules robots and charger ports together. Current movement/service/charge events retain saved physical IDs. Handling/charging completions have priority zero, movement priority one, and travel paths are represented edge by edge. Synthesized continuation events use a counter starting at the saved boundary counter. A wait at one station shifts downstream arrivals at other stations, so queue effects propagate between robots and visits.

It must count every released unfinished order exactly once. Pending, unassigned orders are included through a common deterministic terminal completion rule: process them by release time/id and append pickup/dropoff after a feasible robot's frozen commitments. Robots are ordered by estimated readiness plus pickup travel; the rule considers up to five feasible candidates, expanding the search when candidates are infeasible, and chooses the earliest estimated **no-new-queue** finish. After appending all pending work, the code jointly projects all plans again, so newly introduced charging can delay previously committed deliveries.

This pending assignment is an approximation. Its candidate choice does not repeatedly optimize against all newly induced queue delays. Missing cached route legs are reconstructed battery-feasibly while keeping service-stop order, with a diagnostic counter. Frozen background charging is not interrupted to serve pending work. An incomplete, infeasible, duplicate, future, or non-finite projection must invalidate the label rather than provide a deceptively low score.

The completion estimate is conditional on frozen itineraries, fixed speed/handling/charging behavior, and known queue commitments. It is **not an exact prediction of eventual delivery times under subsequent replanning and arrivals**. It also lacks a terminal value for an empty robot's position, battery reserve for unseen demand, or future coverage. Those omissions can bias against reservation and relocation policies even when their benefits appear later. Review label usefulness as well as arithmetic correctness.

### Model updates and trajectory ownership

Both families start from an explicit constant **Reactive Insertion bootstrap selector**. After enough labels exist, learned checkpoints replace it. This cold start is unavoidable without a trained new selector and should be considered when judging state coverage.

The controller refits each family after roughly **2,000 additional labeled states**, scanning progress periodically. Workers continue collecting while fitting happens, and adopt newly published versions only at decision boundaries. Updates do **not** wait for every running scenario to finish. Each refit uses the accumulated compatible rows from that family's own trajectories, including earlier model versions; it retrains from scratch rather than incrementally updating a booster or continuing NN optimizer state.

XGBoost and NN have separate replay/actor directories. They share inputs, not the chosen next states or the teacher-best trajectory. Validation/test use one explicitly frozen model for the whole episode and do not collect counterfactual training labels.

Action intents, full row-content hashes, per-job locks, fsynced atomic replacement, a persisted initial snapshot, and retained current/previous checkpoint pairs support interruption/resume. A saved intent must retain its original action if a newer model is published during interruption. The adapter restores the selected pre-projection snapshot to prevent scoring cache writes from contaminating subsequent physical states. A lost committed boundary is accepted only if exact serialized bytes can be reproduced from a retained state and original intent; otherwise recovery fails closed. A real interruption/resume intent gate now passes. Power-loss behavior and arbitrary storage failures remain untested. Async publication timing can change trajectories between clean runs even with fixed random seeds; per-decision model hashes are therefore essential provenance.

## 6. Models and input features

Both learners predict ten **cost differences**. Each label is divided by the episode's robot count, then its state-wise ten-policy mean is subtracted. The actor chooses the lowest prediction. This preserves the within-state action ranking while reducing scale differences across city/fleet workloads; it does not directly regress win/loss classes.

Each episode has equal total fitting weight: a row gets weight inversely proportional to its episode's collected row count. Train-only weighted means and standard deviations normalize features. This balances episodes, not necessarily all rare operating states. No validation/test labels may enter these calculations.

| Family | Current implementation |
|---|---|
| XGBoost | Ten separate squared-error regressors; each has 200 trees, max depth 4, learning rate 0.04, L2 regularization 10; histogram tree method; deterministic per-head seeds; four fit threads, one inference thread |
| Small NN | NumPy MLP with ReLU hidden layers 64 and 32, ten linear outputs; weighted squared-error objective; target centering/scaling from training only; Adam, learning rate 0.001, 60 epochs, minibatches 512, small L2 term and clipped gradients |

New model serialization is JSON, including feature schema/scalers, candidate order, label version, training-record identity, cities, episodes, and model-specific parameters. New live simulator state snapshots use pickle internally for resume; these are different artifacts from model JSON.

Base feature extraction is inherited from [`review_bundle/backend/src/delivery_fleet/meta_policy/features.py`](review_bundle/backend/src/delivery_fleet/meta_policy/features.py), called by new `Backend.features()`. It includes map node/edge counts and geometry, fleet composition/availability and battery summaries, current policy and its age, pending/active work and deadline slack, recent observed arrival rates and demand posterior uncertainty, charging occupancy/waits, and recent realized loss. Read the function for the exact feature schema; the examples here are not an exhaustive specification.

The simulator necessarily contains future scenario orders so it can release them later. **The actor's feature extractor and demand estimator must only use released observations and configured public priors.** No future arrival list/count, seed, scenario ID, city-name ID, terminal teacher cost, or final outcome may be an input. The name-based feature checks are only one guard; inspect the actual data flow. Public prior rates are configured expectations, not rates inferred from the realized test scenario.

The candidate order is frozen in `config.POLICIES`:

| Policy ID | Main distinction to inspect in the backend |
|---|---|
| `myopic_ab` | Myopic availability/battery-aware assignment |
| `reactive_insertion` | Reactive insertion into existing plans |
| `paper_sa_adapted` | Paper-adapted reservation/assignment policy |
| `full_no_nn` | Full-policy ablation without learned reservation NN |
| `full_no_idle` | Full-policy ablation without idle relocation |
| `full` | Full spatial-reservation policy |
| `full_queue_aware` | Queue-aware full policy |
| `full_coordinated_idle` | Coordinated idle relocation variant |
| `queue_mle_reservation` | Queue-aware MLE-reservation variant |
| `full_mle_reservation` | Coordinated/full MLE-reservation variant |

Use `runtime.sources()`, `controller_key()`, `set_policy()`, and the generated benchmark scripts as the definitive behavior. Policies share one physical engine; a switch changes decision/controller behavior rather than transplanting physical state from another simulation. Effects of prior assignments and reservations persist according to the backend's switching semantics.

## 7. Scenario diversity and splitting

[`data.py`](src/fleet_switching/data.py) explicitly prepares fresh seeded inputs; it does not execute a delivery policy. The five balanced profiles are ordinary, surge, spatial shift, priority shift, and charging pressure. It varies overall demand, moves a spatial demand emphasis between halves of the day, changes priority mix in a time band, and uses lower starting batteries for charging pressure.

Nominal robot count is `ceil(nodes / 500)`, varied by approximately 0.75–1.25 and bounded to 5–80. Charging-pressure initial battery fraction is 0.4–0.75; other profiles start full. Demand levels are generated through the pinned `create_multicity_suite.make_demand()` and modified by the profile. Exact constants and units are in source, not assumptions to infer from city size.

Seeds derive deterministically from experiment seed, city, split, and index. Scenario bytes, graph bytes, workload identities, split roles, and configuration values are recorded and checked. The old historical labels are **not** reused for the new fit: they used a six-action teacher subset, sampled approximately 25% of decisions, and a different continuation/scoring target. Merely having historical data in the repository does not make it compatible replay.

## 8. Source map: review these files first

Paths below are relative to this branch, so they work in a clone or on GitHub.

| File | Responsibility / review entry point |
|---|---|
| [`operations/switching_meta.py`](operations/switching_meta.py) | Standalone readable launcher; resolves local `src/` or VM staged `code/src/` |
| [`operations/run_switching_final_test.py`](operations/run_switching_final_test.py) | Frozen-model final-test supervisor, protected development artifacts, automatic recovery and no fitting path |
| [`operations/run_switching_development.py`](operations/run_switching_development.py) | Detached full-development supervisor: verifies source/scope, prepares 800/200 inputs, trains both families and validates; no final-test command |
| [`operations/switching_pilot.py`](operations/switching_pilot.py) | Explicit capped pilot: 16 training inputs, two development validation inputs, at most eight episode workers and shared two-hour guard |
| [`operations/switching_pilot_validation.py`](operations/switching_pilot_validation.py) | Twenty controlled candidate continuations on two existing development inputs; no fitting or final test |
| [`operations/switching_pilot_report.py`](operations/switching_pilot_report.py) | Offline evidence/hash checks and measured time extrapolation; imports no simulator and never fits |
| [`src/fleet_switching/cli.py`](src/fleet_switching/cli.py), [`__main__.py`](src/fleet_switching/__main__.py) | Read-only `plan`; all other commands reject without `--execute` |
| [`configs/small_city_switching_v2.json`](configs/small_city_switching_v2.json) | VM paths and experiment/resource settings; generate a portable copy before use elsewhere |
| [`src/fleet_switching/config.py`](src/fleet_switching/config.py) | Fixed city/action sets, `Config`, split guards, identity/hash helpers, atomic JSON, locks, CPU/RAM admission |
| [`src/fleet_switching/data.py`](src/fleet_switching/data.py) | `science_id`, `prepare`, `load_dataset`; workload diversity and train/validation/test manifests |
| [`src/fleet_switching/backend.py`](src/fleet_switching/backend.py) | `activate`, `Backend`, `completion_projection`; production state-to-itinerary conversion, pending completion, fleet/battery setup |
| [`src/fleet_switching/projection.py`](src/fleet_switching/projection.py) | `Step`, `Session`, `Waiting`, `OrderCost`, `project`; independent joint FIFO event calendar and projected loss |
| [`src/fleet_switching/collection.py`](src/fleet_switching/collection.py) | `collect_episode`; choose-before-label, identical branch starts, actor-state adoption, transactional checkpoint/resume |
| [`src/fleet_switching/models.py`](src/fleet_switching/models.py) | `check_rows`, `fit`, `Predictor`; weighting/scalers, both regressors, JSON inference |
| [`src/fleet_switching/workflow.py`](src/fleet_switching/workflow.py) | `_pool`, `_publish`, `train`, `evaluate`, `estimate`; worker budget, refits, frozen validation/test |
| [`src/fleet_switching/__init__.py`](src/fleet_switching/__init__.py) | Target identifier `projected_completion_fifo_v2_event_order` |
| [`src/fleet_switching/provenance.py`](src/fleet_switching/provenance.py) | Campaign, learner, input, content and freeze identities |
| [`src/fleet_switching/replay.py`](src/fleet_switching/replay.py) | Append-only SQLite replay, fixed fit watermarks, content/membership guards and coverage/regret summaries |
| [`src/fleet_switching/execution.py`](src/fleet_switching/execution.py) | Fresh process per episode/fit, async fitting, memory accounting and bounded owned-process shutdown |
| [`src/fleet_switching/backend_lifecycle.py`](src/fleet_switching/backend_lifecycle.py) | Idempotent legacy hooks and runtime-owned cache cleanup |
| [`src/fleet_switching/features.py`](src/fleet_switching/features.py) | Additional observable route-work, queue and spatial supply/demand features |
| [`src/fleet_switching/reporting.py`](src/fleet_switching/reporting.py) | Matched 12-participant comparisons, development-selected fixed baseline and paired intervals |
| [`docs/switching_review_fixes_20261025.md`](docs/switching_review_fixes_20261025.md) | Current fix list, estimates and verification limits |
| [`docs/SWITCHING_PILOT_RESULTS_2026-10-25.md`](docs/SWITCHING_PILOT_RESULTS_2026-10-25.md) | Production pilot, provenance audit, matched outcomes, target-ranking limitation and measured time estimate |
| [`docs/small_city_switching_v2.md`](docs/small_city_switching_v2.md) | Original build notes, VM staging details, future launch commands and estimate |
| [`requirements-switching.txt`](requirements-switching.txt) | New learner dependencies; not a complete simulator environment lock |
| [`review_bundle/requirements-review.txt`](review_bundle/requirements-review.txt) | Broader packages for local source/test review; prepared graphs do not need a new OSM download |
| [`review_bundle/materialize.py`](review_bundle/materialize.py) | Stdlib-only hash verification, optional map decompression and portable config; imports no simulator or learner |
| [`pytest.ini`](pytest.ini) | Excludes the vendored backend from automatic duplicate test collection |

The pinned backend is an active dependency, not just background reading:

| Backend file(s), under `review_bundle/backend/` | Why they matter |
|---|---|
| `src/delivery_fleet/meta_policy/runtime.py` | `FleetRuntime`, snapshot/restore, event advance, policy switching, source compilation and exact-cache hookup |
| `src/delivery_fleet/meta_policy/features.py` | Observable features, FIFO wait feature, map summaries; audit information leakage here |
| `src/delivery_fleet/meta_policy/separation.py` | Workload identity and prior split/provenance machinery |
| `scripts/run_nyc_benchmark*.py`, `run_nyc_*policy.py` | Actual benchmark policy and event-engine sources compiled by runtime; follow `runtime.sources()` |
| `scripts/create_multicity_suite.py`, `scripts/create_nyc_reference_scenario.py` | Demand generator and GraphML node reader used by new preparation |
| `src/delivery_fleet/scenario_creator.py`, `fleet.py`, `robot.py`, `defaults.py` | Scenario/robot definitions, heterogeneous default fleet, physical constants |
| `src/delivery_fleet/deadlines.py` | Reference deadline and loss equations |
| `src/delivery_fleet/battery_routing.py`, `routing.py`, `insertion_routing.py` | Feasibility, paths, route phases and stop insertion |
| `src/delivery_fleet/charging*.py`, `queue_forecasting.py` | Actual queue/service behavior against which the new projector should be checked |
| `src/delivery_fleet/reservation*.py`, `mle_*.py`, `coordinated_idle.py`, `idle_planning.py` | Reservation and relocation behavior; effects when changing controllers |
| `operations/exact_runtime.py` | Shared exact routing indexes, locks and acceleration; required runtime dependency |
| `docs/adaptive_meta_training.md`, `tests/test_meta_*.py` | Original switching architecture/equivalence assumptions and legacy tests; reference material, not newly executed validation |

The snapshot includes backend `src`, `scripts`, `operations`, `configs`, `docs`, `tests`, and README. There is no requirement to fetch its unpublished VM branch separately. Use a fresh process when activating it: the adapter rejects an already imported, different `delivery_fleet` package.

## 9. Data and evidence included in Git

[`review_bundle/PROVENANCE.json`](review_bundle/PROVENANCE.json) records the backend revision, clean snapshot status, original export checksums, and omissions. The verifier checks 177 original exported files plus the current source manifest. The original 18-file receipt is historical and is not the checksum specification for revised code. Additional handoff/helper/environment files are versioned normally. `.gitattributes` preserves the bundle's exact bytes across Windows/Linux checkouts.

| Bundled material | Size/count and purpose |
|---|---|
| `data/maps/*.graphml.gz` | All 12 prepared graphs: **22,656,638 compressed bytes**, **204,739,704 inflated bytes**; ordinary Git objects |
| `data/boundaries/*.geojson`, `data/map_manifest.json` | Eight new geographic boundaries; roles, original VM paths, topology counts, raw/compressed hashes |
| `data/base_models/{spatial,paper}/reservation_fcnn.npz` and `.training.json` | Both existing reservation model weights/metadata needed by candidate policies; these are not new meta-model weights |
| `evidence/build/BUILD_SOURCES.json`, `BUILD_VERIFICATION.json`, `planning_estimate.json` | Staged source hashes, synthetic build verification, conditional runtime arithmetic |
| `evidence/build/ENVIRONMENT.json` | Read-only VM Python/package inventory, including separately installed CPU XGBoost |
| `evidence/original_dynamic/examples.jsonl.gz` | 3,942 old teacher examples: 3,132 train and 810 validation; diagnostic only |
| `evidence/original_dynamic/rollouts.jsonl.gz` | 23,652 old counterfactual records, corresponding to six candidate labels per old example |
| `evidence/original_dynamic/decisions.jsonl.gz`, `completions.jsonl.gz` | 15,462 historical decision records and 225 episode completion records |
| `evidence/original_dynamic/original_scenarios.jsonl.gz` | All 225 scenario documents used by the exported original dynamic round: nine cities × (20 train + five validation) |
| `evidence/original_dynamic/manifest.json`, `round0/`, `comparison/` | Original scope, timing/progress/completion summaries, model report and paired comparison results |
| `evidence/static_selectors/` | Static training data, training report, four JSON models, model registry and dated partial evaluation summary |
| `evidence/threshold_pilots/` | Completed heavy/light switching-guard pilot reports and available analysis |
| `evidence/PROJECT_APPROACHES_STATUS_AND_NEXT_STEPS_2026-10-25.md` | Earlier user-supplied project-status context; historical split/status terminology |

The compressed historical JSONL files wrap source records rather than merging them without provenance. Most have `{"source_path": "...", "record": {...}}`; the scenario archive uses `{"source_path": "...", "scenario": {...}}`. Example for inspection only:

```python
import gzip, json
with gzip.open("review_bundle/evidence/original_dynamic/examples.jsonl.gz", "rt", encoding="utf-8") as f:
    first = json.loads(next(f))
print(first["source_path"], first["record"].keys())
```

Historical JSON metadata often retains absolute VM paths as provenance. Those paths are not all runnable outside the VM. The portable helper rewrites the **new experiment's** map/model/backend configuration, not every historical experiment. This is a focused review package, not a complete archive of every earlier campaign or a promise that every old experiment can be replayed from it.

### What is absent or remains VM-dependent

- **New training/validation/test scenarios, teacher rows, XGBoost/NN checkpoints and scores do not exist yet.** They cannot be included without launching work the owner has not authorized. Existing maps and base weights are included.
- Shared exact routing indexes and query caches are not included. They are derived, large, and rebuildable; an eventual real run may spend significant startup time and RAM regenerating them. The new output directory is separate from older campaigns.
- Live process memory, unfinished historical worker checkpoints, complete per-event traces, and every historical campaign's caches/checkpoints are not included. Some legacy serialized predictors are also VM-only. They are not dependencies of new-model training; the old label/data/report evidence needed for context is supplied.
- Virtual environments, native shared libraries, credentials and SSH keys are not versioned. Package requirements and the VM inventory are supplied instead. Native dependency compatibility and exact numerical reproducibility still need checking in a fresh environment.
- Older large raw graph files in the repository may use Git LFS; those are separate from the 12 ordinary-Git compressed maps in this package. New training does not need to redownload maps from OpenStreetMap.

No essential training source, prepared map, or fixed reservation weight is intentionally left only on the VM. All 1,000 full-development inputs and launch evidence are now archived in Git. Live replay, evolving model versions, logs and full runtime checkpoints remain on the VM while the campaign runs; they are not represented as completed/final artifacts.

## 10. Review without VM access

Python 3.12+ is a reasonable review environment; the VM inventory records Python 3.14.4. VM packages include NumPy 2.5.3, SciPy 1.18.1, NetworkX 3.7, scikit-learn 1.9.1, pytest 9.1.1, and threadpoolctl 3.7.0. XGBoost comes from the **`xgboost-cpu` 3.4.1** distribution installed in a separate dependency directory, not the VM venv itself. The requirements files specify minimums rather than an exact cross-platform lock; inspect `ENVIRONMENT.json` for the recorded versions.

Read-only integrity check; standard library only:

```bash
python -B review_bundle/materialize.py
```

Optional data extraction and path rewriting, **not scenario generation or training**:

```bash
python -B review_bundle/materialize.py --execute
```

This writes `.review-work/maps/`, `.review-work/map_manifest.json`, and `.review-work/config.json`, verifies inflated graph hashes, points the backend at `review_bundle/backend`, and points the two base models at their bundled weights. `.review-work/` is ignored by Git. `--output-dir` can choose another scratch location. About 205 MB is needed for inflated maps. A differing existing map is preserved and causes an error.

Read-only experiment plan after that extraction:

```bash
python -B operations/switching_meta.py plan --config .review-work/config.json
```

The source checkout's VM config has absolute VM paths; do not use it directly on another machine. The planner's default `--cpus 72 --available-gib 171` are **planning assumptions**, not measurements of the reviewer's computer. Full production collection currently reads Linux `/proc/meminfo`, so source review/synthetic tests can run on Windows, while the training resource-admission path targets Linux.

If desired, install review dependencies in your own isolated environment from `review_bundle/requirements-review.txt`. There is no need to run a training command to inspect the logic. To run only the already scoped synthetic tests, set `PYTHONPATH=src` and use:

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_switching_projection.py tests/test_switching_backend_projection.py tests/test_switching_guards.py tests/test_switching_collector.py tests/test_switching_hardening.py tests/test_switching_execution.py
```

On PowerShell use `$env:PYTHONPATH = 'src'`. These tests exercise synthetic arithmetic, backend-shaped fixtures, a stub collector, and fixed-weight inference/serialization and derivative arithmetic, content-tampering cases, stubbed publication, and child-process control stubs. They do not instantiate the production FleetRuntime, optimize models, generate scenarios, or advance a real delivery simulation. Avoid running every historical operational script simply because it is included in the bundle.

The future operational flow is `prepare` → `train` → frozen models → `prepare-test` → `evaluate`, each requiring explicit `--execute`. `prepare` creates only development inputs. `train` performs model-driven collection, bounded asynchronous refits, final fits and matched validation for both selectors plus all ten fixed policies. It freezes models and the development-selected fixed baseline. `evaluate` runs all 12 participants on held-out inputs. Opening final test blocks subsequent development in the same campaign. These commands are described for review, **not authorized to be launched by this handoff**.

## 11. VM layout, if access is available

The owner originally provided SSH access as `ssh gal.pinto@132.68.161.40`. This document contains no login secret. Use only access and operations separately authorized by the owner.

| VM location | Purpose |
|---|---|
| `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet` | Main VM project and venv |
| `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-meta-policy` | Preserved simulator checkout at the pinned backend commit |
| Main project `runs/icaps_small_city_maps_v1_20261025` | Prepared small-city manifest and new maps |
| Main project `runs/icaps_small_city_switching_v4_pilot_fixes_20261025` | Verified v4 build and active full-development campaign; see the launch record and live status files |
| Main project `.venv/bin/python` | Existing VM interpreter |
| Main project `runs/icaps_static_policy_nn_v1_20261024/dependencies` | Existing isolated CPU-XGBoost dependency directory referenced by VM config |

Older campaigns, paused workers, existing maps and legacy models are preserved. Do not thaw/resume them or overwrite their outputs. A real new campaign should use its own output directory and a single controller. The revised Git/VM sources are tied by `review_bundle/CURRENT_BUILD_SOURCES.json`. The original 18-file staging receipt remains historical; the materializer verifies current source hashes.

## 12. Parallelism and expected training time

The VM reports **72 logical CPUs / 36 physical core IDs**. Admission allows up to **64 simultaneous episode processes**, each with one numerical thread. Every episode gets a fresh process. One separately budgeted asynchronous fitter uses four threads; branches remain sequential inside an episode.

Admission retains a **64 GiB available-memory reserve**, budgets **1.5 GiB per prospective episode**, accounts for submitted workers whose allocation is not yet visible, and separately reserves **8 GiB for fitting**. Crossing the reserve triggers cooperative shutdown and bounded termination of unresponsive owned processes. This is monitored admission, not a hard cgroup limit or protection against unrelated VM allocations. Startup, RSS and eight-worker throughput are now measured in the pilot report; 64-worker scaling remains an estimate.

Both learners need approximately **800,000 training branches**. Complete development validation adds approximately **120,000 windows**: 400 selector episodes and 2,000 fixed-policy episodes. Read [the measured pilot report](docs/SWITCHING_PILOT_RESULTS_2026-10-25.md) and `review_bundle/evidence/switching_pilot_20261025/TRAINING_TIME_ESTIMATE.json` for the current estimate and its assumptions. The old unmeasured 7–24-hour arithmetic / 24–48-hour precautionary budget is superseded. The final-test phase is separate; its authorized launch and live status are described in the final-test launch record.

## 13. What has and has not been verified

The original build had 31 synthetic tests passing locally and on the VM. Current scoped test counts and hashes are recorded in [CURRENT_BUILD_VERIFICATION.json](review_bundle/CURRENT_BUILD_VERIFICATION.json) and [CURRENT_BUILD_SOURCES.json](review_bundle/CURRENT_BUILD_SOURCES.json). The revised suite exercises pure queue/event arithmetic, backend-shaped fixtures, stub collector transitions, content tampering, checkpoint fallback, native-thread limits with a stubbed publisher, process failure/cleanup and fixed-weight NN derivatives. No actual optimizer or production event engine is used.

The portable helper verifies all 177 original export hashes and current-source hashes; the 12 prepared maps remain unchanged. A bounded pilot now supplies actual scenario generation, learner optimization, queue projection, branch-order, fitting/serialization, interruption/resume, memory, runtime and matched outcome evidence. Full development training and all 2,400 validation outcomes subsequently completed; the owner then authorized the final test, now running with frozen models. The reservation-model provenance audit reconstructed both 100-row historical training archives and found no held-out-city overlap; its historical-immutability qualification is recorded in the report.

Power-loss behavior and arbitrary storage failures remain untested. The pilot's eight-worker throughput does not establish 64-worker throughput, and two validation inputs do not establish geographic generalization.

The 15-minute actor-chosen transition rule, 12-hour arrival horizon and four held-out cities are unchanged. The terminal score still omits future demand and future fleet value. Longer lookahead, explicit exploration and a learned terminal value remain research proposals. New observable route-work and spatial supply/demand features reduce some input aliasing without supplying future arrivals.

There are small development-pilot performance results in the report, not final-test results. Matched reporting covers both selectors and all ten fixed policies, a fixed baseline selected on development data, and hindsight best fixed per scenario as a separate diagnostic. Incomplete/mismatched comparisons are rejected. Paired intervals are conditional on the observed cities; four final cities do not establish universal geographic generalization.

## 14. Specific questions for the reviewing AI

Please audit code and scientific assumptions, not just whether files import. Useful findings should identify a file/function and explain a concrete failure case or bias, its severity, and a proposed correction/verification. Separate confirmed defects from hypotheses and untested concerns. Prioritize:

1. **Actor versus teacher:** Is the action irrevocably recorded before labels? Does every branch begin from a complete identical snapshot, including RNG/controller/event/queue state? Is the next actual state always the chosen 15-minute boundary, including crash/resume and model publication races?
2. **Physical projection fidelity:** Do partial edges, remaining handling, locked stops, payload/battery changes, route phases and active charge energy match the actual backend's semantics? Do reconstructed legs preserve feasibility and future stop order? Check charging sessions, FIFO ordering/ties, multi-port stations, repeated visits, cross-station propagation and background charging against real runtime definitions.
3. **Loss coverage:** Is every released order represented exactly once, including unassigned pending orders, carried packages and orders delivered exactly at a boundary? Are absolute deadlines and elapsed times converted correctly? Can an infeasible pending completion be silently omitted? Are teacher labels accidentally summed as episode outcomes?
4. **Terminal approximation:** Does the pending shortlist/no-new-queue selection bias comparisons? Could freezing background charging or omitting future coverage/battery value systematically favor a benchmark? What later small, explicitly authorized production pilot would quantify projection error and decision-ranking usefulness?
5. **Information separation:** Inspect feature and demand-prior data flow, not just feature names. Can future scenario information, realized order counts, city/test identities, validation labels, test outcomes or incompatible historical data enter fitting/scaling or policy choice?
6. **Learning correctness:** Check weighted centering/scaling, per-episode weights, ten-head XGBoost fitting/inference, NN backpropagation/Adam/regularization and serialization. Is the cost-regression target appropriate for regret and rare high-priority failures? How limiting is deterministic Reactive Insertion bootstrap with no explicit exploration? Old model-version replay and asynchronous update schedules also deserve scrutiny.
7. **Identity and resume:** Audit combined scientific/campaign/learner/content identities, including backend operations, fixed weights, all new source, dependency versions and actor intents. Confirm worker-limit changes remain operational. Check SQLite fit-watermark consistency, active-manifest membership, immutable dataset/freeze receipts, the final-test barrier, durable writes and retained state pairs. Regeneration that changes committed snapshot bytes must fail closed.
8. **Resources and speed:** Audit process/thread limits, import order for thread limits, shared cache locks/index initialization, 1.5-GiB admission assumptions, memory behavior of collecting/re-reading replay and repeated full refits, pool failure/exit behavior, and coexistence with existing VM work. The VM has 72 logical CPUs, not 72 independent physical cores.
9. **Validation/evaluation completeness:** Frozen validation must run full episodes without teacher-label feedback. The current `evaluate()` covers selectors and all ten fixed policies. Verify complete input matching, the frozen development-selected baseline, paired reporting, failures and uncertainty before anyone claims a win.
10. **Portability:** Verify the pinned dependency is actually used, all source/data paths resolve after materialization, exact source/model/map identities survive a fresh checkout, and the minimum requirements are compatible. Do not assume synthetic tests establish XGBoost fitting or real event-loop compatibility.

Suggested review output: a prioritized findings list with source locations and evidence; any missing safeguards or evaluation components; a short statement of what is supported versus untested; and the smallest targeted validation plan to run **after separate launch authorization**. If a premise in this document disagrees with the code, report the disagreement explicitly.


## New nine-action ablation (29 October 2026)

The owner authorized a separate XGBoost and NN training campaign with only
`full_mle_reservation` removed from the action portfolio. Queue MLE remains.
See [the ablation protocol](docs/SWITCHING_NO_FULL_MLE_2026-10-29.md).
The new campaign retains 15-minute decisions and excludes all four test cities.
Dated records above describe the original completed ten-action campaign.


## Replacement five-benchmark campaign (29 October 2026)

The owner subsequently requested the best five eligible policies from the
original development-validation results. The nine-action service was stopped
and its artifacts preserved. The replacement trains both models from scratch
with Full Coordinated Idle, Queue MLE Reservation, Full Queue Aware, Full
without NN, and Full. Full MLE stays excluded; all four test cities stay held
out. See [the current five-benchmark protocol](docs/SWITCHING_TOP5_NO_FULL_MLE_2026-10-29.md).
