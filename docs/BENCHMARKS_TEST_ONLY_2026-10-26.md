# Fixed benchmarks: final-test comparison

Only the ten fixed benchmarks are included. No meta-model outcomes, training scores or validation scores enter this comparison.

The same **200 test scenarios** (50 each in Cambridge UK, Ghent, Salamanca and Lucca) give **2,000 complete matched benchmark outcomes**. Input identities, outcome receipt hashes, fleets and delivery completion were verified.

Primary metric: **equal-city mean loss per robot**, lower is better. These are deadline/importance-weighted loss values.

| Rank | Benchmark | Primary loss | Higher loss versus best | On-time deliveries | Best/tied scenarios |
|---|---|---:|---:|---:|---:|
| 1 | Full MLE Reservation | 155.91240 | +0.00% | 99.98% | 79/200 |
| 2 | Full Coordinated Idle | 157.17277 | +0.81% | 99.99% | 43/200 |
| 3 | Queue MLE Reservation | 157.32096 | +0.90% | 99.95% | 38/200 |
| 4 | Full without NN | 157.67521 | +1.13% | 99.93% | 38/200 |
| 5 | Full Queue Aware | 158.62002 | +1.74% | 99.98% | 22/200 |
| 6 | Full | 158.70892 | +1.79% | 99.98% | 23/200 |
| 7 | Reactive Insertion | 164.95051 | +5.80% | 99.99% | 22/200 |
| 8 | Full without Idle | 165.04377 | +5.86% | 99.98% | 14/200 |
| 9 | Paper SA Adapted | 179.67951 | +15.24% | 99.23% | 15/200 |
| 10 | Myopic AB | 597.73099 | +283.38% | 88.21% | 0/200 |

Best/tied scenario counts overlap when several fixed policies tie. They do not sum to 200. All admitted orders completed; on-time percentage is a separate metric from weighted loss.

## Per-city primary scores

| Benchmark | Cambridge UK | Ghent | Salamanca | Lucca |
|---|---:|---:|---:|---:|
| Full MLE Reservation | 177.0107 | 161.2080 | 133.9567 | 151.4741 |
| Full Coordinated Idle | 178.6384 | 162.9929 | 134.2356 | 152.8242 |
| Queue MLE Reservation | 178.5497 | 162.8572 | 135.9817 | 151.8952 |
| Full without NN | 178.5481 | 162.9020 | 137.3556 | 151.8952 |
| Full Queue Aware | 180.1985 | 165.0446 | 135.8703 | 153.3666 |
| Full | 180.1935 | 165.0939 | 136.1817 | 153.3666 |
| Reactive Insertion | 187.9971 | 170.4274 | 135.2657 | 166.1119 |
| Full without Idle | 187.9842 | 170.6666 | 135.1894 | 166.3348 |
| Paper SA Adapted | 196.6390 | 185.0430 | 136.9275 | 200.1086 |
| Myopic AB | 634.0930 | 552.8739 | 702.9621 | 500.9950 |

## Per-city winners

- cambridge_uk: **Full MLE Reservation**, 177.01072.
- ghent: **Full MLE Reservation**, 161.20805.
- lucca: **Full MLE Reservation**, 151.47415.
- salamanca: **Full MLE Reservation**, 133.95669.

## Paired uncertainty against the preselected reference

Full MLE Reservation was selected on development before test. Positive differences favor that reference. The JSON contains 5,000-draw stratified paired bootstrap intervals, conditional on the four observed cities and unadjusted for multiple comparisons. The table reports observed rankings, not universal geographic superiority.

This compares composite benchmark implementations. The separate reservation-activation audit is relevant when attributing any benefit to the MLE component itself. No new simulation or fitting was performed for this report.
