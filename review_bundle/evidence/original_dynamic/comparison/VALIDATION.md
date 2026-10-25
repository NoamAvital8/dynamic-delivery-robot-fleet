# Switching-bar choice on validation days

Chosen: **5% bar** (gain_5pct), compared on 44 validation days that were never used for fitting or testing.

Each cell: loss compared with the 1% bar, averaged per city (negative = lower loss).

| City | 1% bar | 0% bar | 5% bar |
|---|---:|---:|---:|
| barcelona | +0.00% | +0.00% | +0.00% |
| haifa | +0.00% | +0.00% | +0.00% |
| johannesburg | +0.00% | -0.53% | +0.57% |
| manhattan | +0.00% | +0.00% | +0.18% |
| moscow | +0.00% | +0.43% | -0.20% |
| new_delhi | +0.00% | +0.00% | +0.00% |
| new_york_city | +0.00% | +1.25% | -0.23% |
| paris | +0.00% | +0.00% | -1.18% |
| tel_aviv | +0.00% | +0.00% | -0.37% |
| **all cities** | +0.00% | +0.13% | -0.14% |
| policy switches per day | 1.4 | 4.3 | 1.0 |


## Successful retry reconciliation

Verified 135/135 complete validation runs across all 45 paired validation days, including the two successful Johannesburg retries. There are zero unresolved validation failures.

The original 5% switching threshold was frozen using 44 paired days before held-out testing. Including the missing validation day confirms the same threshold. The original selection record and all selector model files remain byte-for-byte unchanged; existing test runs stay valid.

See validation/reconciliation.json and validation_status.json for verified counts, receipt signatures and model hashes.
