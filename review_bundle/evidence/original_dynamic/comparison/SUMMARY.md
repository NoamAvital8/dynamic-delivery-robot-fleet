# Frozen meta-policy comparison

Each cell: mean over test days of (meta loss - baseline loss) / baseline loss, then (days the meta-policy was better / days compared). Negative = the meta-policy has lower loss.

- Switching bar: 5%, chosen on validation days (VALIDATION.md).
- Best fixed policy, chosen in advance from training labels: **full_mle_reservation**. Per city: barcelona = full_mle_reservation, haifa = full_mle_reservation, johannesburg = full_mle_reservation, manhattan = full_mle_reservation, moscow = queue_mle_reservation, new_delhi = full_mle_reservation, new_york_city = full_coordinated_idle, paris = full_mle_reservation, tel_aviv = full_mle_reservation.
- Hindsight best: the lowest-loss completed policy for each day, picked after seeing every result (any of the benchmark policies). An upper bar, not a policy anyone could deploy.
- Baseline losses are complete fixed-policy benchmark runs on the same test scenarios; (vN) is the benchmark campaign, i.e. code version, that produced each one.

## Cities seen in training

### Against the chosen baselines

| City | best fixed (training) | best fixed for city (training) | hindsight best |
|---|---:|---:|---:|
| barcelona | +0.14% (1/5 better) | +0.14% (1/5 better) | +14.37% (0/5 better) |
| haifa | +0.00% (0/5 better) | +0.00% (0/5 better) | +1.36% (0/5 better) |
| johannesburg | +0.00% (0/1 better) | +0.00% (0/1 better) | +6.00% (0/5 better) |
| manhattan | +0.00% (0/5 better) | +0.00% (0/5 better) | +1.82% (0/5 better) |
| moscow | +2.03% (0/1 better) | +0.22% (2/4 better) | -10.79% (1/5 better) |
| new_delhi | +1.15% (0/5 better) | +1.15% (0/5 better) | +2.35% (0/5 better) |
| new_york_city | -0.28% (2/5 better) | +0.00% (0/5 better) | +3.08% (0/5 better) |
| paris | +0.44% (0/5 better) | +0.44% (0/5 better) | +9.80% (0/5 better) |
| tel_aviv | +0.00% (0/5 better) | +0.00% (0/5 better) | +1.44% (0/5 better) |
| **all** | +0.25% (3/37 better) | +0.24% (3/40 better) | +3.27% (1/45 better) |

### Against each portfolio policy

| City | paper_sa_adapted | full_no_idle | full_queue_aware | full_coordinated_idle | queue_mle_reservation | full_mle_reservation |
|---|---:|---:|---:|---:|---:|---:|
| barcelona | +13.40% (0/5 better) | +14.06% (0/5 better) | -1.11% (5/5 better) | +0.07% (0/5 better) | -1.42% (5/5 better) | +0.14% (1/5 better) |
| haifa | -9.05% (4/5 better) | -6.94% (4/5 better) | -2.66% (4/5 better) | +0.05% (2/5 better) | -0.20% (3/5 better) | +0.00% (0/5 better) |
| johannesburg | +4.10% (2/5 better) | +1.01% (3/5 better) | -0.40% (2/5 better) | +0.00% (0/5 better) | +0.19% (1/5 better) | +0.00% (0/1 better) |
| manhattan | -8.71% (4/5 better) | -9.55% (4/5 better) | -0.73% (4/5 better) | +1.82% (1/5 better) | -2.62% (5/5 better) | +0.00% (0/5 better) |
| moscow | -52.54% (5/5 better) | -52.51% (5/5 better) | -0.27% (3/4 better) | +0.00% (0/4 better) | +0.22% (2/4 better) | +2.03% (0/1 better) |
| new_delhi | -10.17% (5/5 better) | -8.91% (5/5 better) | -1.58% (4/5 better) | -1.68% (4/5 better) | +1.68% (1/5 better) | +1.15% (0/5 better) |
| new_york_city | -73.88% (5/5 better) | -72.62% (5/5 better) | +1.43% (2/5 better) | +0.00% (0/5 better) | +0.81% (3/5 better) | -0.28% (2/5 better) |
| paris | +9.45% (0/5 better) | +9.39% (0/5 better) | -0.89% (4/5 better) | +0.00% (0/5 better) | -0.89% (4/5 better) | +0.44% (0/5 better) |
| tel_aviv | -5.52% (3/5 better) | -6.16% (3/5 better) | -0.35% (3/5 better) | +0.43% (1/5 better) | -1.38% (4/5 better) | +0.00% (0/5 better) |
| **all** | -14.77% (28/45 better) | -14.69% (29/45 better) | -0.74% (31/44 better) | +0.08% (8/44 better) | -0.42% (28/44 better) | +0.25% (3/37 better) |

## New cities (never in training)

### Against the chosen baselines

| City | best fixed (training) | best fixed for city (training) | hindsight best |
|---|---:|---:|---:|
| sydney | +2.30% (0/1 better) | n/a | -26.30% (2/4 better) |
| **all** | +2.30% (0/1 better) | n/a | -26.30% (2/4 better) |

### Against each portfolio policy

| City | paper_sa_adapted | full_no_idle | full_queue_aware | full_coordinated_idle | queue_mle_reservation | full_mle_reservation |
|---|---:|---:|---:|---:|---:|---:|
| sydney | -54.70% (4/4 better) | -54.14% (4/4 better) | +0.14% (1/2 better) | +0.00% (0/2 better) | +1.03% (1/2 better) | +2.30% (0/1 better) |
| **all** | -54.70% (4/4 better) | -54.14% (4/4 better) | +0.14% (1/2 better) | +0.00% (0/2 better) | +1.03% (1/2 better) | +2.30% (0/1 better) |

## Per test day

| City | Scenario | Meta loss | best fixed (training) | best fixed for city (training) | hindsight best |
|---|---|---:|---|---|---|
| barcelona | barcelona_test_000 | 27,625.901 | +0.25% full_mle_reservation (v6) | +0.25% full_mle_reservation (v6) | +17.20% reactive_insertion (v2) |
| barcelona | barcelona_test_001 | 36,772.292 | -0.44% full_mle_reservation (v6) | -0.44% full_mle_reservation (v6) | +10.77% reactive_insertion (v2) |
| barcelona | barcelona_test_002 | 26,344.720 | +0.15% full_mle_reservation (v6) | +0.15% full_mle_reservation (v6) | +12.05% full_no_idle (v2) |
| barcelona | barcelona_test_003 | 24,490.733 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +20.00% full_no_idle (v2) |
| barcelona | barcelona_test_004 | 21,723.055 | +0.75% full_mle_reservation (v6) | +0.75% full_mle_reservation (v6) | +11.86% paper_sa_adapted (v2) |
| haifa | haifa_test_000 | 3,742.644 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.90% reactive_insertion (v2) |
| haifa | haifa_test_001 | 8,418.235 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +4.32% queue_mle_reservation (v6) |
| haifa | haifa_test_002 | 6,667.701 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +1.59% queue_mle_reservation (v6) |
| haifa | haifa_test_003 | 9,864.946 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) |
| haifa | haifa_test_004 | 7,475.244 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) |
| johannesburg | johannesburg_test_000 | 449,490.004 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +7.66% paper_sa_adapted (v7) |
| johannesburg | johannesburg_test_001 | 1,488,316.674 | missing full_mle_reservation | missing full_mle_reservation | +2.85% queue_mle_reservation (v7) |
| johannesburg | johannesburg_test_002 | 403,975.455 | missing full_mle_reservation | missing full_mle_reservation | +17.73% paper_sa_adapted (v7) |
| johannesburg | johannesburg_test_003 | 533,553.925 | missing full_mle_reservation | missing full_mle_reservation | +0.91% queue_mle_reservation (v7) |
| johannesburg | johannesburg_test_004 | 629,450.834 | missing full_mle_reservation | missing full_mle_reservation | +0.83% full_queue_aware (v7) |
| manhattan | manhattan_test_000 | 12,849.019 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) |
| manhattan | manhattan_test_001 | 24,932.958 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +7.36% full_coordinated_idle (v5) |
| manhattan | manhattan_test_002 | 15,859.859 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.21% full_coordinated_idle (v5) |
| manhattan | manhattan_test_003 | 16,268.081 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) |
| manhattan | manhattan_test_004 | 14,046.111 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +1.53% full_coordinated_idle (v5) |
| moscow | moscow_test_000 | 2,323,626.495 | +2.03% full_mle_reservation (v7) | +1.75% queue_mle_reservation (v7) | +2.03% full_mle_reservation (v7) |
| moscow | moscow_test_001 | 952,329.152 | missing full_mle_reservation | +0.33% queue_mle_reservation (v7) | +0.33% full_queue_aware (v7) |
| moscow | moscow_test_002 | 1,627,997.418 | missing full_mle_reservation | -0.88% queue_mle_reservation (v7) | +0.00% full_coordinated_idle (v7) |
| moscow | moscow_test_003 | 1,834,935.517 | missing full_mle_reservation | -0.32% queue_mle_reservation (v7) | +0.00% full_coordinated_idle (v7) |
| moscow | moscow_test_004 | 2,076,414.086 | missing full_mle_reservation | missing queue_mle_reservation | -56.29% paper_sa_adapted (v7) |
| new_delhi | new_delhi_test_000 | 2,568.769 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +3.52% queue_mle_reservation (v7) |
| new_delhi | new_delhi_test_001 | 2,983.582 | +5.76% full_mle_reservation (v7) | +5.76% full_mle_reservation (v7) | +5.76% full_mle_reservation (v7) |
| new_delhi | new_delhi_test_002 | 2,452.802 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +2.28% full_coordinated_idle (v7) |
| new_delhi | new_delhi_test_003 | 2,017.828 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +0.20% queue_mle_reservation (v7) |
| new_delhi | new_delhi_test_004 | 1,564.029 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) |
| new_york_city | new_york_city_test_000 | 1,039,296.306 | +3.20% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) | +4.71% queue_mle_reservation (v6) |
| new_york_city | new_york_city_test_001 | 751,150.790 | +0.80% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) | +0.80% full_mle_reservation (v6) |
| new_york_city | new_york_city_test_002 | 539,558.450 | -2.65% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) | +4.78% full_queue_aware (v4) |
| new_york_city | new_york_city_test_003 | 396,552.994 | +0.00% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) | +5.11% full_queue_aware (v4) |
| new_york_city | new_york_city_test_004 | 1,540,431.845 | -2.75% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) | +0.00% full_coordinated_idle (v5) |
| paris | paris_test_000 | 39,851.636 | +0.50% full_mle_reservation (v7) | +0.50% full_mle_reservation (v7) | +4.43% paper_sa_adapted (v7) |
| paris | paris_test_001 | 40,908.034 | +0.99% full_mle_reservation (v7) | +0.99% full_mle_reservation (v7) | +15.11% reactive_insertion (v7) |
| paris | paris_test_002 | 16,097.739 | +0.01% full_mle_reservation (v7) | +0.01% full_mle_reservation (v7) | +6.94% full_no_idle (v7) |
| paris | paris_test_003 | 34,442.722 | +0.00% full_mle_reservation (v7) | +0.00% full_mle_reservation (v7) | +10.30% reactive_insertion (v7) |
| paris | paris_test_004 | 25,207.328 | +0.72% full_mle_reservation (v7) | +0.72% full_mle_reservation (v7) | +12.21% paper_sa_adapted (v7) |
| sydney | sydney_test_000 | 4,722,062.993 | +2.30% full_mle_reservation (v7) | n/a | +2.30% full_mle_reservation (v7) |
| sydney | sydney_test_002 | 1,468,980.767 | missing full_mle_reservation | n/a | +0.00% full_coordinated_idle (v7) |
| sydney | sydney_test_003 | 2,555,068.417 | missing full_mle_reservation | n/a | -55.57% paper_sa_adapted (v7) |
| sydney | sydney_test_004 | 1,622,916.992 | missing full_mle_reservation | n/a | -51.93% paper_sa_adapted (v7) |
| tel_aviv | tel_aviv_test_000 | 7,864.920 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +2.39% full_queue_aware (v4) |
| tel_aviv | tel_aviv_test_001 | 7,917.659 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.77% full_coordinated_idle (v5) |
| tel_aviv | tel_aviv_test_002 | 7,350.643 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +3.24% reactive_insertion (v2) |
| tel_aviv | tel_aviv_test_003 | 6,212.637 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.78% full_no_idle (v2) |
| tel_aviv | tel_aviv_test_004 | 6,593.675 | +0.00% full_mle_reservation (v6) | +0.00% full_mle_reservation (v6) | +0.00% full_coordinated_idle (v5) |

Days without any completed matched baseline are not ranked.
