# Full MLE Reservation: training decision optimality

Analysis of immutable training labels only; no new simulations or fitting. Counts include bootstrap and learned-actor decisions.

A policy is best when its saved projected cost is within `1e-7` absolute loss units of the minimum among all ten policies.

| Trajectory family | States | Best including ties | Unique best | Shared best | All ten tied |
|---|---:|---:|---:|---:|---:|
| nn | 40,852 | 34,441 (84.31%) | 2,392 (5.86%) | 32,049 (78.45%) | 15,322 (37.51%) |
| xgboost | 40,743 | 34,460 (84.58%) | 1,977 (4.85%) | 32,483 (79.73%) | 15,463 (37.95%) |
| combined | 81,595 | 68,901 (84.44%) | 4,369 (5.35%) | 64,532 (79.09%) | 30,785 (37.73%) |

## Excluding states where all policies tied

| Family | States with different scores | Full MLE among best | Episode-weighted best rate, all states | Actor actually chose Full MLE |
|---|---:|---:|---:|---:|
| nn | 25,530 | 19,119 (74.89%) | 84.49% | 8,674 (21.23%) |
| xgboost | 25,280 | 18,997 (75.15%) | 84.75% | 10,206 (25.05%) |

XGBoost and NN visited different states on their own trajectories. Pooled counts are state visits, not independent scenarios. Episode-weighted percentages give each trajectory equal total weight, matching the fitting weight scheme.

Every row content hash and input provenance was checked. The complete row-ID/content-hash set for each family exactly matches the final frozen model's `training_records_sha256` and state count.

**Meaning of right:** lowest 15-minute-plus-projected-completion training target. It omits future demand and terminal fleet value; this is not guaranteed full-episode optimality. Ties do not imply a unique correct action.

The JSON report includes per-city/profile and bootstrap/learned breakdowns, all-policy frequencies, tolerance sensitivity and projected regret diagnostics. It reads training rows only and uses no validation/test outcomes.

## All ten policies

Best-including-ties counts overlap; they are not mutually exclusive class frequencies.

| Policy | Best/tied, XGBoost states | Best/tied, NN states | Unique best, combined |
|---|---:|---:|---:|
| full_mle_reservation | 34,460 (84.58%) | 34,441 (84.31%) | 4,369 |
| queue_mle_reservation | 32,318 (79.32%) | 31,646 (77.46%) | 237 |
| full_no_nn | 31,943 (78.40%) | 31,354 (76.75%) | 141 |
| reactive_insertion | 31,200 (76.58%) | 30,365 (74.33%) | 1,935 |
| full_coordinated_idle | 25,603 (62.84%) | 25,666 (62.83%) | 996 |
| full_queue_aware | 24,337 (59.73%) | 24,097 (58.99%) | 57 |
| full | 24,257 (59.54%) | 24,031 (58.82%) | 31 |
| full_no_idle | 23,895 (58.65%) | 23,416 (57.32%) | 208 |
| paper_sa_adapted | 23,282 (57.14%) | 22,879 (56.00%) | 706 |
| myopic_ab | 21,924 (53.81%) | 21,339 (52.23%) | 97 |

Full MLE is the most frequently best/tied policy on both families' visited training states. Choosing another tied-best policy is also correct under this target. Therefore its 84.44% best/tied frequency is not a requirement that the actor must literally choose Full MLE 84.44% of the time.

Tie-tolerance sensitivity is recorded in the JSON: changing from exact equality to the declared tolerance changes only one Full-MLE best/tied count out of 81,595 states.
