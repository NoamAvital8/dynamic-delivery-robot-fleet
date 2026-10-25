# Combined final threshold-pilot assessment — 24 October 2026

Light pilot: 14/18 complete; stopped at the requested 18:00 Israel cutoff. Heavy pilot: 18/18 complete.
All 462 protected original artifacts are unchanged. Both original full-campaign services remain frozen.
Missing scores are excluded from comparisons. Last committed pilot checkpoints were verified and preserved.

## Light workloads

Full maps/fleets; first 60 minutes of calibration arrivals; selector until 120 minutes, then common full_no_idle delivery drain.
| City | Orders | Current loss (switches) | Relaxed loss (switches) | No guards loss (switches) |
|---|---:|---:|---:|---:|
| new_delhi | 2 | 128.906 (0) | 128.906 (0) | 128.906 (1) |
| paris | 26 | 1,197.305 (0) | 1,197.305 (0) | 1,197.305 (4) |
| johannesburg | 42 | 6,718.296 (0) | 6,718.296 (0) | 6,718.296 (5) |
| moscow | 182 | 29,045.377 (0) | 29,045.377 (1) | Incomplete at cutoff |
| sydney | 135 | 58,500.188 (0) | 58,500.188 (1) | 58,391.711 (5) |
| beijing | 158 | Incomplete at cutoff | Incomplete at cutoff | Incomplete at cutoff |

Sydney no-guard loss was 0.18543% lower with five switches. Delhi, Paris and Johannesburg tied across all three settings; Moscow current/relaxed tied. No complete three-way comparison exists for Moscow or Beijing.

## Heavy development workloads

Only training-city maps Tel Aviv, Haifa and Manhattan; two fresh steady/shift workloads per city. Two-hour arrivals; native control until all deliveries finish, with no forced common drain policy.
| City / workload | Current loss | Relaxed loss | No guards loss | Switches current / relaxed / no guards |
|---|---:|---:|---:|---:|
| tel_aviv / steady_heavy | 17,876.169 | 17,876.169 | 18,273.380 | 0 / 0 / 12 |
| tel_aviv / shift_heavy | 30,272.413 | 30,272.413 | 29,434.696 | 0 / 1 / 14 |
| haifa / steady_heavy | 15,475.927 | 15,475.927 | 14,721.906 | 0 / 0 / 13 |
| haifa / shift_heavy | 59,696.997 | 59,696.997 | 60,851.379 | 0 / 0 / 17 |
| manhattan / steady_heavy | 49,152.290 | 49,152.290 | 49,665.119 | 0 / 0 / 11 |
| manhattan / shift_heavy | 76,908.402 | 76,944.944 | 77,064.903 | 0 / 1 / 11 |

Heavy totals: current 249,382.198; relaxed 249,418.740 (+0.01465%); no guards 250,011.384 (+0.25230%).
Actual switches: 0, 2 and 78. No guards improved two of six workloads and worsened four. Equal-city aggregate change was +0.04955%.
Peak busy share was approximately 96–100%, with charging queues up to 16 robots: these workloads created actual fleet pressure.

## Conclusions and next step

Lowering both guards creates real switches and different physical trajectories, but has not consistently improved loss. The guard-only change is not a reliable repair. This does not prove that guards are irrelevant or that dynamic selection cannot work.
Both minimum gain and error multiplier changed together, so their individual effects are not isolated. One light case per city and two heavy cases per training city cannot establish high-probability superiority or a full-benchmark winner.
The predictor and training-target quality deserve attention: the saved offline ranking diagnostics were already worse than a strong fixed baseline without the deployment guards. Candidate-control labels used 15 minutes followed by full_no_idle continuation, which may not value longer-term effects as deployed.
Before another multi-day label collection, reuse existing complete fixed-policy runs to compare cheap start-only selectors with training-only grouped validation and a strong fixed-policy/city-lookup baseline. Fit scalers inside each fold, keep all policies from a workload together, and freeze configurations before test evaluation. Fitting is minutes; reserve a bounded 2–4-hour offline investigation, then reassess the cost of genuinely new simulation labels.
The user subsequently authorized FCNN, XGBoost, decision-tree and Extra-Trees start-only models, fitted exclusively on the five earlier cities, followed by a small comparative test-city pilot. That is a separate workflow; none of these threshold outputs is used to tune those models. No new multi-day dynamic-meta training is authorized.
Any future hyperparameter selection must use training/validation cities and fresh validation data. The six final benchmark cities remain separate from fitting; exploratory comparisons on their outcomes must be described as exploratory.

## Files

/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_meta_threshold_pilot_20261024
/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_meta_threshold_heavy_pilot_20261024
Heavy details: FINAL_REPORT.md and FINAL_ANALYSIS.json in the heavy folder.
Complete light details and missing/checkpoint records: COMBINED_FINAL_ANALYSIS_2026-10-24.json in the light folder.
Original long campaigns remain paused; only the separately authorized static-selector workflow may run.
