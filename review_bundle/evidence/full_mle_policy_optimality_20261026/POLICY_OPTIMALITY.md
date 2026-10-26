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
