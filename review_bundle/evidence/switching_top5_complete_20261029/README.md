# Completed five-benchmark switching campaign — 29 October 2026

Training and development validation finished successfully at **16:32:11 Israel
time (14:32:11 UTC)**. All **1,600 training episodes** and **1,400 matched
validation outcomes** completed, with zero missing or failed outcomes. The
service exited successfully. The four final-test cities remain unopened.

The scientific source is commit `dc2f75788210fda4ce9887bba1c68a7e97d9c822`
on `codex/small-city-switching-review-20261025`. The 60-file source-manifest
SHA-256 remains `1bc632436a200bb661d41942d95ed029d8150c729576b72953f13c97de0847c4`.

Both learners used only the five selected policies and made decisions every
15 minutes. Final training states: **40,661 XGBoost**, **40,668 NN**. The
original and stopped nine-action experiment artifacts remain separate.

## Completed development validation

Metric: equal-city mean final episode loss per robot; lower is better.
Each row evaluates the same 200 development-validation inputs in eight cities.

| Policy/model | Loss per robot |
|---|---:|
| full_coordinated_idle | 230.788588 |
| xgboost | 231.456765 |
| queue_mle_reservation | 231.541625 |
| nn | 232.025143 |
| full_queue_aware | 233.124642 |
| full_no_nn | 233.331046 |
| full | 237.414991 |

Full Coordinated Idle remains the strongest fixed benchmark. XGBoost's mean
loss is **0.2895% higher**, and NN's is **0.5358% higher**. These descriptive
means do not establish statistical significance. Paired outcome differences,
win/tie/loss counts and bootstrap intervals are in the complete validation
report. This is development validation, not a new test-set result.

## Verified artifacts

- `COMPLETION_AUDIT.json`: completed scope, scores, identities and holdout checks.
- `FULL_DEVELOPMENT_COMPLETE.json`, `full_run_status.json`: terminal receipts.
- `frozen_models.json`: immutable freeze ID
  `a4716cea96802b7a3c1369c86e9e72eb04149dd5953da45521713a45ff311b9f`.
- `campaign.json`: scientific campaign settings and source identities.
- `FINAL_MODEL_EXPORT.json`, `*_final_model.json.gz`: exact final model payloads,
  compressed with their original uncompressed SHA-256 values and state counts.
- `fixed_*.json`: the five exact frozen fixed-policy payloads.
- `validation_summary.json.gz`: complete 1,400-outcome report and paired statistics.

The VM verification used the existing freeze guard to check the frozen model
hashes, campaign identity, development manifest and validation-report hash.
No simulations, fitting or final-test preparation were performed during export.

The campaign began at 13:11:50 Israel time, was stopped at 14:34 before a VM
reboot, and resumed at 15:02:50. The `started_utc` in the completion receipt
refers to the resumed supervisor invocation. See
`review_bundle/evidence/switching_top5_reboot_recovery_20261029/` for the
verified checkpoint recovery and persistent service definition. Its startup
conditions skip the now-complete campaign and any opened final-test campaign.

All 1,000 input manifests are archived with launch evidence; exact matching
scenario bytes are already in Git under
`review_bundle/evidence/full_development_launch_20261025/DEVELOPMENT_SCENARIOS.jsonl.gz`.
Full live replay, actor checkpoints, complete logs and intermediate model
versions remain on the VM. The final exported models and validation results
are available for offline review from Git.
