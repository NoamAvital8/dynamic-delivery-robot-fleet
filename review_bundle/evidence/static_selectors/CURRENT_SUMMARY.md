# Static start-only selector pilot

Updated 2026-10-25T13:30:01.550272+00:00; complete 27/48 logical evaluations.

| City | FCNN mean loss | XGBoost mean loss | Decision tree mean loss | Extra Trees mean loss |
|---|---:|---:|---:|---:|
| beijing | — (0/2) | — (0/2) | — (0/2) | — (0/2) |
| paris | 40,379.835 (2/2) | 40,080.916 (2/2) | 36,857.589 (2/2) | 36,857.589 (2/2) |
| new_delhi | 2,826.787 (2/2) | 2,694.941 (2/2) | 3,016.228 (2/2) | 3,016.228 (2/2) |
| johannesburg | 968,903.339 (2/2) | 449,490.004 (1/2) | 421,730.874 (1/2) | 421,730.874 (1/2) |
| moscow | 1,616,415.328 (2/2) | 2,277,377.002 (1/2) | — (0/2) | — (0/2) |
| sydney | 5,624,090.094 (2/2) | 4,616,055.071 (1/2) | — (0/2) | — (0/2) |

## Frozen choices

| City | FCNN | XGBoost | Decision tree | Extra Trees |
|---|---|---|---|---|
| beijing | full_coordinated_idle | full_mle_reservation | reactive_insertion | reactive_insertion |
| paris | full_coordinated_idle | full_mle_reservation | reactive_insertion | reactive_insertion |
| new_delhi | full_coordinated_idle | full_mle_reservation | reactive_insertion | reactive_insertion |
| johannesburg | full_coordinated_idle | full_mle_reservation | reactive_insertion | reactive_insertion |
| moscow | queue_mle_reservation | full_mle_reservation | reactive_insertion | reactive_insertion |
| sydney | queue_mle_reservation | full_mle_reservation | reactive_insertion | reactive_insertion |

Only identical scenario subsets can be compared. Completed matching fixed-policy outcomes are reused with hashes.
The original campaigns remain frozen; a preserved unfinished worker is not duplicated or thawed.
Two scenarios per city are an exploratory comparison, not a high-confidence superiority result.
