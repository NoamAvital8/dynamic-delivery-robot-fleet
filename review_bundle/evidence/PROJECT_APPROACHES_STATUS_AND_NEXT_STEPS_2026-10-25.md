# Robot delivery project: approaches, current work, and next steps

Status checked on 2026-10-25 at 08:28 UTC. This is a handoff snapshot, not a final results report.

## 1. What we are trying to achieve

Select a strong dispatch policy for delivery robots across different cities, fleet configurations and demand conditions. We have tried both improving individual policies and learning which policy to use. Lower loss is better. A more complex model or more frequent switching is useful only if it improves measured performance.

The six test cities are **Beijing, Paris, New Delhi, Johannesburg, Moscow and Sydney**. The earlier cities used to develop the new static selectors are **Tel Aviv, Haifa, Manhattan, NYC and Barcelona**.

## 2. Approaches explored

| Approach | Idea | What we have learned |
|---|---|---|
| Fixed-policy benchmarks | Compare Myopic A+B, Reactive Insertion, paper-adapted, and our full policy and ablations. | Different policies perform differently by city. There is no established universal winner. |
| Original reservation NN | Estimate reservation parameters controlling which importance levels robots can serve. | Having a trained NN does not by itself establish better dispatch outcomes. |
| Better readiness and charger estimates | Replace the eight-minutes-per-stop approximation with remaining travel, handling, charging and waiting; account for charging queues in assignment. | These are separate policy versions. Preserve their results and compare compatible versions explicitly. |
| Coordinated idle relocation | Coordinate robots' intended destinations, account for coverage already supplied by other robots, and discourage unnecessary relocation changes. | Coordination is promising in some settings, but must justify its cost and loss relative to simpler alternatives. |
| Uncertainty-gated MLE reservation | Estimate demand rates and activate reservation only when confidence intervals are sufficiently narrow. | In the completed earlier MLE comparison, reservation never activated. Those scores measured fallback dispatch/relocation, not successful MLE reservation optimization. |
| Dynamic switching meta-model | Choose a policy repeatedly during a run using the observed system state. | The deployed selector often keeps its initial policy. Relaxing guards causes more switches, but has not shown a consistent loss improvement. |
| Static start-only policy selectors | Use public starting information to choose one policy and keep it for the entire run. | Four model families are fitted; a small six-city comparison is running. Trees currently lead the fair completed subset. |

The original dynamic switching model is a **HistGradientBoosting regressor**, distinct from both the reservation FCNN and the new static FCNN. These should not all be called the same NN.

## 3. Completed threshold experiments

We kept the dynamic predictor unchanged and compared three settings. Both the gain guard and error guard changed, so their individual effects were not isolated.

| Setting | Minimum relative gain | Error multiplier | Heavy-pilot total loss | Post-initial switches |
|---|---:|---:|---:|---:|
| Current | 5% | 1 | 249,382.198 | 0 |
| Relaxed | 1% | 0.25 | 249,418.740 | 2 |
| Unrestricted | 0% | 0 | 250,011.384 | 78 |

The heavy pilot finished **18/18 evaluations**: two fresh workloads in each of Tel Aviv, Haifa and Manhattan. Relaxed loss was 0.0147% worse in aggregate; unrestricted loss was 0.2523% worse, with two workload wins and four losses versus current guards.

The light pilot finished **14/18** before the user-approved cutoff. Four incomplete cases were excluded from loss comparisons. It used short calibration-arrival prefixes on the six test cities, with a common no-idle continuation after selector control ended. Several completed cases had identical final losses despite switching. This was exploratory evidence, not hyperparameter selection on an untouched final test set.

Conclusion: lowering guards alone is not currently supported as a reliable fix. This does not prove adaptation cannot work or that guards never matter. Prediction quality, training targets, state coverage and the consequences of switching remain candidates for investigation.

Both pilots ended; neither is queued to restart.

## 4. Static start-only models: completed first training stage

The following models finished fitting and grouped validation on 2026-10-24:

- FCNN: ensemble of three small networks, each with 16 tanh hidden units.
- XGBoost.
- Decision tree.
- Extra Trees.

We reused **310 valid saved fixed-policy outcomes from 25 scenarios**, representing only **five distinct public starting profiles**. This is not 310 independent scenarios. Historical policy versions are distinguished rather than silently merged.

Inputs include public map properties, configured fleet, initial battery/idle state, known horizon, historical demand priors and candidate-policy identity. They exclude actual future arrivals, scenario seeds, test outcomes and city-name IDs. All six test cities were excluded from fitting, scaling and tuning. Model choices were frozen before opening their test outcomes.

The current pilot uses two original held-out scenarios per test city: **48 logical selector evaluations**, sharing **36 unique policy/scenario cases**. Exact matching completed fixed-policy outcomes are reused; selecting a fixed policy does not require duplicating its simulation for each selector family.

### Frozen choices

| City | FCNN | XGBoost | Decision tree / Extra Trees |
|---|---|---|---|
| Beijing | Coordinated + NN | Full coordinated + MLE | Reactive Insertion |
| Paris | Coordinated + NN | Full coordinated + MLE | Reactive Insertion |
| New Delhi | Coordinated + NN | Full coordinated + MLE | Reactive Insertion |
| Johannesburg | Coordinated + NN | Full coordinated + MLE | Reactive Insertion |
| Moscow | Queue-aware + MLE | Full coordinated + MLE | Reactive Insertion |
| Sydney | Queue-aware + MLE | Full coordinated + MLE | Reactive Insertion |

These are model choices, not claims that the chosen policy is optimal. In particular, choosing different policies across cities does not make FCNN better than trees.

### Current results: mean loss, with completed scenario count

| City | FCNN | XGBoost | Decision tree | Extra Trees |
|---|---:|---:|---:|---:|
| Beijing | — (0/2) | — (0/2) | — (0/2) | — (0/2) |
| Paris | 40,379.835 (2/2) | 40,080.916 (2/2) | 36,857.589 (2/2) | 36,857.589 (2/2) |
| New Delhi | 2,826.787 (2/2) | 2,694.941 (2/2) | 3,016.228 (2/2) | 3,016.228 (2/2) |
| Johannesburg | 968,903.339 (2/2) | 449,490.004 (1/2) | 421,730.874 (1/2) | 421,730.874 (1/2) |
| Moscow | 1,616,415.328 (2/2) | 2,277,377.002 (1/2) | — (0/2) | — (0/2) |
| Sydney | 5,624,090.094 (2/2) | 4,616,055.071 (1/2) | — (0/2) | — (0/2) |

**Do not rank cells with different scenario coverage.** On the five scenarios completed for all four families, the fair total-loss comparison is:

| Rank | Selector | Matched total loss |
|---:|---|---:|
| 1= | Decision tree | 501,478.509 |
| 1= | Extra Trees | 501,478.509 |
| 3 | XGBoost | 535,041.717 |
| 4 | FCNN | 535,903.248 |

Trees are approximately 6.4% better than FCNN on this subset. Both trees select Reactive Insertion everywhere, so this has not yet shown an improvement over simply using Reactive Insertion directly. It is not a final six-city ranking.

## 5. What is running now

**27/48 logical evaluations are available; 21 remain outstanding. Training is finished.**

| Active simulation | Policy | How it started |
|---|---|---|
| Beijing test 001 | `full_coordinated_idle` | New isolated evaluation |
| Beijing test 000 | `full_mle_reservation` | New isolated evaluation |
| Beijing test 000 | `full_coordinated_idle` | Explicitly approved resumption of its preserved original worker |

The other approved original dependency, Sydney test 001 / `queue_mle_reservation`, finished. It delivered all 4,184 orders, with loss 6,631,499.043; its result was verified and adopted into the new comparison without rewriting original outputs.

The comparison controller and approved-dependency watchdog are active. The VM has approximately 373 GiB RAM, with 171 GiB available at this check. New simulations are limited to two slots, use inline scoring and single numerical-library threads; the existing 64 GiB global memory reserve remains enforced.

These full-map simulations are expensive. Fitting took seconds; unfinished evaluation cases can take days. We do not have a reliable overall finish estimate and should not extrapolate from cached results or the small maps.

## 6. What is queued, and what is paused

Within the authorized static comparison, missing cases are:

| City | Missing unique policy/scenario cases, including running work |
|---|---|
| Beijing | Scenarios 000 and 001 for coordinated + NN, full coordinated + MLE, and Reactive Insertion. Three of these six cases are currently running. |
| Johannesburg | Scenario 001 for full coordinated + MLE and Reactive Insertion. |
| Moscow | Scenario 001 for full coordinated + MLE; scenarios 000 and 001 for Reactive Insertion. |
| Sydney | Scenario 001 for full coordinated + MLE; scenarios 000 and 001 for Reactive Insertion. |
| Paris / New Delhi | None for this pilot. |

Reactive Insertion results serve both tree selectors, so physical cases and logical model-evaluation counts differ. This table describes remaining scope, not a promise of scheduling order.

The original full fixed-policy and dynamic meta-evaluation campaigns remain **paused/frozen**, not scheduled to resume automatically. Nineteen other original simulator roots remain stopped. Only the two specifically approved dependency workers were resumed; one has now completed. No new training, extra scenarios or threshold pilots are queued.

## 7. What we think is limiting the learned selectors

1. **Limited independent environments.** Hundreds of policy outcomes do not replace varied maps, fleets and starting demand profiles. The static training set has only five distinct starting profiles.
2. **Ranking errors.** The dynamic selector must distinguish small policy advantages, not merely predict overall loss. An audit on earlier-city stored validation states found approximately 58% strict pairwise direction accuracy, with a substantial train/validation gap. This is a diagnostic of its stored rollout labels, not a full-run performance metric.
3. **Target mismatch.** Dynamic labels score a short candidate-policy segment followed by a fixed continuation. Repeated policy selection during deployment can have different long-term effects. This is a plausible limitation, not a proven sole cause.
4. **State distribution and missing information.** Deployment may reach states poorly represented in training, and aggregate features may omit information needed to rank policies.
5. **Switching has consequences.** Being able to reproduce every fixed policy by never switching does not guarantee a learned selector can identify the best trajectory. More switches can worsen decisions.

We have not established one runtime bug explaining all poor results. A larger NN or a different model family is not automatically a solution.

## 8. Proposed way forward

The user asked us to preserve this work as the completed first training stage. All datasets, preprocessing definitions, validation results and fitted model artifacts should be reused when choosing the next model. Do not repeat completed simulations unnecessarily.

Recommended sequence:

1. Finish the currently authorized comparison, preserving frozen choices and evaluating only matching complete scenarios. Report raw and equal-city normalized loss separately.
2. Before another expensive training campaign, audit whether existing training/validation data supports a policy-selection advantage over a strong fixed policy. Measure ranking regret, performance by city and generalization on held-out episodes/cities.
3. For the dynamic model, consider a small residual calibration model using frozen predictor outputs plus a few existing state features. Fit using held-out or cross-fitted predictions; do not evaluate corrections on the same samples used to fit them.
4. Compare that low-cost correction with a simple static selector and a strong fixed-policy baseline. Require an offline validation improvement before launching more simulations.
5. If promising, run a bounded fresh training/validation-city pilot with a fixed time budget and success criteria chosen in advance. Do not begin another multi-day label-generation campaign merely because an architecture sounds stronger.
6. If existing labels contain too little useful signal, collect a small, targeted batch of more informative scenarios or longer-horizon counterfactuals rather than regenerating everything.

Residual fitting could take minutes; implementation and verification may take hours. Simulation cost remains the uncertain part. These are recommendations, **not authorization to launch additional fitting, experiments or paused campaigns**.

### Evaluation discipline

Use only training/validation cities and fresh validation data for future hyperparameter selection. The current test-city pilot is exploratory and has already been inspected. Do not tune against it and then claim untouched final-test superiority. If its outcomes are ever explicitly repurposed for training, a new independent final test is required.

JSON model exports preserve inference, but are not necessarily optimizer-resumable training checkpoints. Depending on the selected family, later improvement may mean fitting a correction or refitting on the saved dataset. That does not require repeating the expensive completed simulations.

## 9. Files and operational handoff

Authoritative VM comparison directory:

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_static_policy_nn_v1_20261024`

Important files inside it:

- `models/*.json` and `model_registry.json`: four frozen models and hashes.
- `training_data.json`, `training_report.json`: preserved first-stage data and validation.
- `evaluation_plan.json`, `test_choices/`: frozen evaluation inputs and policy selections.
- `CURRENT_SUMMARY.json` and `CURRENT_SUMMARY.md`: current results; the report-only helper refreshes these from verified completed artifacts.
- `approved_dependency_resume_status.json`: status of the two specifically approved original workers.
- `operations/static_policy_nn_20261024.py`: readable training/evaluation entry point.
- `operations/report_static_selectors_20261024.py`: report-only refresh; does not launch simulations or train.

The older controller `status.json` can lag available cached outcomes; use the refreshed `CURRENT_SUMMARY` for coverage. Historical resolved source-guard failures remain in logs and must not be mistaken for a new failure.

Threshold final report:

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_meta_threshold_pilot_20261024/COMBINED_FINAL_REPORT_2026-10-24.md`

Original fixed-policy campaign and trained reservation models remain under:

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies`

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_multicity_v2_precision/training/`

Local verified snapshot:

`C:/Users/galpi/.codex/visualizations/2026/09/22/01a0c8d1-386e-73c0-8bd5-2d0f0d4cead4/static_selector_snapshot_2026-10-25/`

Preserve all completed results and models. Do not duplicate live simulations, thaw whole original services, weaken the memory reserve, or use encoded/opaque launch commands. Use named readable scripts and inspect exact process identity before any process action.
