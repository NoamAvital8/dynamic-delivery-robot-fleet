# Small-city switching pilot and provenance audit — 25 October 2026

Repository: https://github.com/NoamAvital8/dynamic-delivery-robot-fleet

Branch: `codex/small-city-switching-review-20261025`.

This report records the completed pilot before full training. The owner later
authorized [the full development campaign](FULL_DEVELOPMENT_LAUNCH_2026-10-25.md),
which started at 22:45 Israel time on 25 October. The four final-test cities remain unopened.

This is a bounded development pilot, authorized separately from full training.
It uses two fresh 12-hour scenarios in each of the eight development cities,
one ordinary and one charging-pressure input. Both XGBoost and the small NN
collect their own actor-selected trajectories: **16 inputs, 32 training runs**.
All ten candidate policies are scored at each 15-minute training decision.
The four final-test cities — Cambridge UK, Ghent, Salamanca and Lucca — are
excluded from pilot generation, collection, fitting and evaluation.

## Provenance audit

The two fixed reservation FCNN dependencies each have 100 historical training
rows: 20 each from Tel Aviv, Haifa, Manhattan, New York City and Barcelona.
Every row ID belongs to the old training roster, with no overlap with its test
roster or the four new held-out cities. Reconstructing all input features,
candidate-winner selections, alpha targets and perfect-information losses from
the saved scenarios and candidate outcomes passed for both target archives.
Model metadata links the supplied weight files to those archives.

This establishes consistency of the artifacts available today. The historical
training metadata did not record dataset-content hashes at fitting time, so it
does not prove that historical files have never changed. This qualification is
retained in the machine-readable audit, rather than claiming cryptographic
historical immutability.

Evidence: `review_bundle/evidence/switching_pilot_20261025/audit/`, including
`RESERVATION_PROVENANCE_AUDIT.json`, both target matrices, training manifests,
model-training metadata and all 100 source scenario documents compressed as JSONL.
The original pinned reservation weights remain under `review_bundle/data/base_models/`.

## Measured training collection

Times below are seconds **per complete model/scenario run**, averaged over the
two model families. They include initialization, all ten counterfactual policies,
projection, durable labels and checkpoints. They are not times for a single
15-minute decision. The two columns use different seeds, fleets and demand;
their difference is not a controlled estimate of the effect of low batteries.

| Development city | Ordinary | Charging pressure |
|---|---:|---:|
| Delft | 75.1 | 112.5 |
| Haifa | 194.4 | 129.3 |
| Leuven | 40.5 | 67.6 |
| Manhattan | 683.7 | 708.3 |
| New Delhi | 42.9 | 39.2 |
| Oxford | 50.2 | 29.6 |
| Palo Alto | 382.0 | 142.6 |
| Tel Aviv | 93.0 | 226.4 |

The corrected collection finished in **1,073.7 seconds (17.9 minutes)** with at
most eight episode workers and one four-thread fitter. Summed episode-worker
time was **6,034.8 seconds (100.6 worker-minutes)**. The difference matters:
Manhattan's late-running jobs create a tail, so multiplying pilot elapsed time
by 50 would exaggerate the cost of the much larger, better-filled campaign.

Manhattan accounted for **46.1% of all measured collection-worker time**.
Palo Alto was the next expensive city. Graph bounds alone do not determine
runtime: order count, fleet size, route lengths and the policy trajectory also
matter. The agreed eight-city roster has been retained.

The final replays contain **829 XGBoost states and 810 NN states**, producing
**16,390 candidate branches**. XGBoost fits at 200–829 states took 2.89–4.14
seconds; NN fits at 203–810 states took 0.47–1.01 seconds. Small optimizer fits
are a minor cost compared with teacher simulations. The pilot refit cadence was
200 new states per family, to exercise real publication and actor adoption;
the full proposed campaign retains the declared 2,000-state cadence.

Recorded per-process peak RSS reached **0.794 GiB**. This Linux high-water metric
can include an inherited process high-water mark; it is a conservative process
measurement, not total campaign RAM or a proof of throughput at 64 workers.
The configured 1.5-GiB episode budget, 8-GiB trainer budget and 64-GiB available
reserve are retained. The pilot measured eight-worker operation; full 64-worker
scaling remains an estimate.

The minimum observed available RAM during collection was **166.45 GiB**,
comfortably above the retained reserve. Of recorded teacher work, **5,653.6
seconds** were restore/advance/snapshot windows and **218.0 seconds** were
completion projection. The simulator windows dominate this pilot's cost.

## Fresh matched development comparison

Both final pilot models were frozen before evaluation. Two additional fresh
surge inputs, one each in New Delhi and Manhattan, were run with those models
and all ten fixed benchmarks: **24 complete outcomes**, with identical inputs
and fleets within each comparison. Validation took **520.7 seconds (8.7
minutes)**; summed validation-worker time was **2,600.3 seconds**. Every admitted
order was delivered. The corrected collection plus validation therefore took
**26.6 minutes**, excluding the earlier gates, failed attempt and separate
continuation diagnostic.

| Fresh validation input | Best fixed benchmark loss | XGBoost loss (vs best fixed) | NN loss (vs best fixed) |
|---|---:|---:|---:|
| New Delhi, 94 orders / 21 robots | Full without NN: 3,029.37 | 3,691.36 (+21.85%) | 3,022.04 (−0.24%) |
| Manhattan, 303 orders / 80 robots | Reactive Insertion: 17,929.63 | 17,415.43 (−2.87%) | 17,653.39 (−1.54%) |

Lower loss is better. The NN beat the hindsight best fixed policy on both
inputs. Its combined raw loss was **1.35% lower**, and its equal-city mean loss
per robot was **1.03% lower**, than the hindsight best fixed policy per input.
XGBoost won Manhattan but lost New Delhi; its corresponding aggregate differences
were **+0.71% raw loss / +6.81% equal-city loss per robot**.

The single fixed policy selected across these development inputs by equal-city
loss per robot was `full_no_nn`. This is distinct from choosing the best fixed
benchmark separately after seeing each input's outcomes.

These are promising feasibility results for the NN, not a reliable improvement
estimate. There is only one validation input per city, so within-city bootstrap
intervals in the raw reporting output are degenerate and provide no useful
uncertainty estimate. No final-test performance is claimed. XGBoost's New Delhi
failure is also a reason to retain a broad matched validation campaign rather
than declaring a winning family from two inputs.

The models do adopt different actions: validation post-initial switch counts
were XGBoost 36/35 and NN 41/44 in New Delhi/Manhattan. Switching frequency is a
diagnostic, not an objective. In training, all ten candidate costs tied in
249/829 XGBoost states and 214/810 NN states. Fitted predictors can still choose
different actions around those ties; future experiments may measure hysteresis
or tie handling, without assuming more switches mean better performance.

## Projected ranking versus remaining-episode continuation

A separate controlled diagnostic reused the ordinary training input for New
Delhi and Manhattan. Every candidate started from the same Reactive Insertion
state at minute 360, ran its own policy to minute 375, and received the current
projected-completion score. The saved minute-375 boundary was then restored and
Reactive Insertion completed the remaining episode with all actual future
arrivals retained. These **20 completed candidate continuations** generated no
new training labels and no new input scenarios.

| Development state | Projected choice | Best under common continuation | Remaining-cost regret of projected choice |
|---|---|---|---:|
| New Delhi at 360 min | Reactive Insertion, among nine tied candidates | Full without NN / Queue MLE reservation | 64.81 (+4.25%) |
| Manhattan at 360 min | Queue MLE reservation | Full MLE reservation | 82.49 (+0.83%) |

The frozen queue fixture passed exact arithmetic, but this diagnostic shows
that exact frozen-route completion does **not** ensure the best longer-term
action ranking. In New Delhi, actions with equal current completion scores led
to different future episode costs. Supply position, residual battery and the
effect of future orders matter. More examples alone cannot guarantee removal
of that target blind spot.

The percentages use the best remaining-episode cost under the common reference
policy as denominator, not whole-day loss or an optimal policy value. Only two
states were examined. Reactive Insertion is a controlled continuation, not the
learned actor's actual future behavior. This is evidence of a limitation, not
a population estimate of target error or a new physical-scoring bug.

My assessment: the integration gates support a full development experiment as
a research baseline, and the NN's small matched wins give it a plausible chance.
They do not justify a claim that more training will beat every benchmark. A
later controlled comparison of a short common continuation or a terminal fleet
value would address the observed ranking limitation. No such objective change
or full campaign had been launched at the time of this pilot assessment. Keep the complete fixed-policy validation
comparison and the four-city final holdout when choosing the next experiment.

## Updated full-campaign time estimate

For **100 training and 25 validation inputs per development city**, both
learners require 1,600 training episodes; matched frozen validation requires
another 2,400 episodes. My central expectation is **about 4–6 elapsed hours**
on this VM, with a **6–12-hour first-campaign planning budget**. Final-test
evaluation is additional and remains unmeasured.

This replaces the old unmeasured 24–48-hour precautionary budget. It is an
extrapolation, not a completion-time guarantee or statistical confidence band:

- Summed pilot collection cost scales to **83.8 worker-hours** for the same
  ordinary/charging mix; an 80/20 ordinary proxy gives **85.6 worker-hours**.
  Neither mix measures all five production profiles across many random seeds.
- Frozen validation is separately calibrated: the 24 measured outcomes were
  **3.09 times** the corresponding training-state raw-window proxy. Applying
  that observed ratio to all eight cities gives about **36.4 worker-hours**
  for 2,400 validation episodes. Transferring the two-city calibration to the
  other six cities is an assumption.
- Full admission allows up to **64 one-thread episode processes**, plus one
  four-thread fitter, on 72 logical CPUs / 36 physical core IDs. The estimate
  assumes **24–36 effective worker equivalents**, rather than linear 8→64
  scaling. Memory reserve and admission accounting remain enabled.
- Refit-size extrapolation projects roughly **41k states per family** and
  **21 refits per family**. Depending on assumed size scaling, all fits occupy
  about **0.8–2.0 trainer-hours**, much of which can overlap collection.
  Final fitting still has to finish before validation.
- Allowing twice the measured collection work, 0.7–1.5 times calibrated
  validation work, imperfect concurrency and fitting overhead gives an explicit
  arithmetic envelope of **3.36–11.95 hours**. The central expectation and
  practical budget above add planning judgment to this broad envelope.

`TRAINING_TIME_ESTIMATE.json` records every multiplier and fit-size assumption.
`operations/switching_pilot_report.py --evidence review_bundle/evidence/switching_pilot_20261025`
reproduces the offline analysis without importing a simulator or fitting a model.
The estimate can change with demand profile, policy adoption, long drains,
other VM load and 64-worker disk/replay contention.

## Integration checks and fixes

- All ten fixed policies passed the preserved runtime's short and charging
  snapshot tests: **25 tests passed**.
- A production-engine queue fixture delivered all three orders. Projected and
  actual completion loss both equaled **813.2172429218303**, with **30.4866667
  minutes** of actual/projected FIFO queue waiting. This validates the frozen
  commitment calculation for that fixture, not every future dispatch outcome.
- Both real learners passed fitting, JSON serialization and prediction
  roundtrips with zero prediction discrepancy.
- Interrupting collection after the durable action intent, publishing a
  different live actor, then resuming preserved the original action and exact
  completed loss (**82.14502164502126**), matching uninterrupted collection.
- Reversing candidate evaluation order produced exactly equal scores. Physical
  state digests also matched after removing only the MLE controller's diagnostic
  wall-clock `planning_seconds` from the digest calculation. The pinned backend
  is unchanged; its raw digest includes this nondeterministic diagnostic timer.
- The first pilot exposed a real adapter bug after the last delivery: the engine
  stops processing physical events at completion but advances its displayed
  clock to the requested decision boundary. Remaining background charging events
  can then appear in the past. The adapter now returns zero unfinished loss only
  after verifying that all scenario orders are delivered and no assigned,
  pending or service-locked work remains. Inconsistent finished states still
  fail closed. The failed attempt is preserved; corrected collection uses a
  fresh output and scientific identity, without mixing its replay.
- Campaign identity now includes the online refit cadence, so changing that
  trajectory-affecting setting cannot silently resume an existing campaign.
  This additional guard was applied after the immutable pilot snapshot; it
  does not change label arithmetic. The pilot already hashes its entire config.
- The updated scoped synthetic suite passes **64 tests locally**. These tests
  supplement the production checks; they do not certify power-loss recovery or
  arbitrary filesystem failure behavior.

## Scope and locations

The pinned simulator remains commit
`5e76f5384aec98e9d5a538b27dae205df301fa28` in the preserved VM checkout and
`review_bundle/backend/`. The source-only full-campaign target is
`runs/icaps_small_city_switching_v4_pilot_fixes_20261025`; it was initially verified without launching training, and now runs the
separately authorized development campaign linked above. The prior v2/v3 builds remain preserved.

The failed pilot is `runs/icaps_small_city_pilot_v1_20261025`; the corrected
pilot is `runs/icaps_small_city_pilot_v2_20261025`, both under the main VM project.
The bounded runner is `operations/switching_pilot.py`. A separate diagnostic,
`operations/switching_pilot_validation.py`, reuses existing development inputs
to compare projected rankings with a common remaining-episode continuation.
It does not add training labels or open the final test.

Pilot snapshots and weights are development evidence, not frozen final models
and not compatible replay to import silently into a future full campaign.
The model and row metadata retain original VM paths and content hashes. Maps and
pinned dependencies are already bundled; live routing caches and bulky runtime
checkpoints stay on the VM.
