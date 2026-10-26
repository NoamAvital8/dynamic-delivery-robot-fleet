# Final switching test results — 26 October 2026

Final test completed at **13:28 Israel time**. All **2,400/2,400** matched outcomes completed, with **zero missing or failed results**. Elapsed time from service launch was about **31 minutes**. The service exited normally without restarts.

Both selectors and all ten fixed benchmarks used 200 inputs: 50 each in Cambridge UK, Ghent, Salamanca and Lucca. Models, development artifacts and training replay were verified unchanged after evaluation. No retraining used the test data.

## Primary comparison

The declared metric is equal-city mean loss per robot; **lower is better**. Full MLE Reservation was selected on development data before test and also has the lowest test score among the ten fixed policies.

| Participant | Primary loss | Difference versus selected fixed baseline |
|---|---:|---:|
| Full MLE Reservation | 155.91240 | +0.00% |
| XGBoost selector | 157.34780 | +0.92% |
| NN selector | 161.83022 | +3.80% |

Neither selector beat the strongest fixed policy overall. XGBoost performed better than the NN on this test.

## Per-scenario and per-city results

| Selector | Wins versus selected Full MLE baseline | Wins versus hindsight best fixed policy per scenario |
|---|---:|---:|
| XGBoost | 71/200 | 37/200 |
| NN | 40/200 | 20/200 |

Hindsight best fixed policy per scenario is an outcome-dependent diagnostic, distinct from the single deployable baseline selected before test.

| City | Full MLE | XGBoost | NN |
|---|---:|---:|---:|
| cambridge_uk | 177.0107 | 178.9753 | 188.6337 |
| ghent | 161.2080 | 161.8896 | 164.6988 |
| salamanca | 133.9567 | 135.8441 | 136.4712 |
| lucca | 151.4741 | 152.6822 | 157.5172 |

## Integrity and saved evidence

`review_bundle/evidence/final_test_results_20261026/` contains the full matched report, completion receipts and checksums. All test inputs, the pre-test freeze, benchmark payloads and supervisor source are in the corresponding final-test launch evidence. Selector payloads are in the 26 October development-status evidence.

The report also contains paired differences and stratified episode-bootstrap intervals. Those intervals are conditional on the four observed cities, not a population-level geographic guarantee. No model, threshold or fixed baseline was selected or updated from these test outcomes.
