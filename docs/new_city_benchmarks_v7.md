# Six new cities: isolated all-policy benchmark campaign

Source maps are from `uncertainty-aware-idle-relocation` at
`6d060021a2feb556999418708426613bf9a53f57`, not the older main checkout.
The six GraphML files are Git LFS objects. Their sizes and SHA-256 checksums
are verified against the committed pointers before copying to the VM.

| City | Nodes | Edges |
|---|---:|---:|
| Beijing | 354,619 | 495,374 |
| Sydney | 421,271 | 593,192 |
| Moscow | 557,061 | 803,014 |
| Johannesburg | 149,160 | 202,512 |
| New Delhi | 8,254 | 11,525 |
| Paris | 77,129 | 110,313 |

All original maps and experiments are preserved. None of Tel Aviv, Haifa,
Manhattan, NYC or Barcelona is part of this new campaign. The New Delhi map
is the repository's **New Delhi administrative-area** map, not the whole Delhi
metropolitan region. Do not describe these unequal administrative/map extents
as identically sized city coverage.

## What is run

Five held-out scenarios per new city, each with a 12-hour arrival horizon,
paired across all **14 implemented policies**: 420 evaluations total.

- `full`: original spatial-NN full policy.
- `myopic_ab`: myopic reactive insertion/completion-time baseline.
- `reactive_insertion`: exhaustive incremental-loss insertion baseline.
- `paper_sa_adapted`: paper-style moving-average reservation NN.
- `full_no_nn`: original full policy without reservation NN.
- `full_no_idle`: original reservation policy without predictive idle planning.
- `full_uncertainty_idle`: original NN with posterior-uncertainty idle penalty.
- `full_uncertainty_idle_no_nn`: the corresponding no-NN ablation.
- `full_queue_aware`: NN with charger-queue-aware assignment and legacy idle.
- `full_queue_aware_no_nn`: its no-NN ablation.
- `full_coordinated_idle`: NN with assignment-triggered coordinated idle.
- `full_coordinated_idle_no_nn`: its no-NN ablation.
- `queue_mle_reservation`: uncertainty-gated MLE reservation with legacy idle.
- `full_mle_reservation`: the same MLE controller with coordinated idle.

The miniature offline oracle is not an exact solver for these large maps and
is not included in this city-scale campaign.

Existing spatial and paper NNs are reused unchanged. Twenty separate calibration
scenarios per new city estimate city priors; they are **not used to retrain the
NNs** or choose parameters from the held-out test scenarios. Base seed is
20261007. Demand still follows the existing independently generated hourly
node-level Poisson model; graph clusters do not define the demand generator.

The downloaded graphs have clusters but no charger annotations. The launcher
validates topology, geometry, cluster coverage and edge lengths, then applies
the **existing** deterministic charger-placement algorithm once per map.
Prepared copies retain the saved HDBSCAN labels and live in the new campaign's
`maps/` folder; raw downloaded map bytes stay unchanged under `data/graphs/`.
All policies see the same prepared graph for a city. Charger layouts are
cached with source/configuration and output hashes.

## Resources, durability and timing

`scripts/run_new_city_benchmarks.py` performs preparation, scenario generation
and benchmarks in sequence. Preparation uses six processes (one per map). Benchmarking uses
12 simultaneous simulators, four idle-scoring workers each, inline MLE scoring
and numerical-library threads capped at one: at most **60 compute workers**.
The phases do not overlap their worker pools. The round-robin job order starts
different cities early rather than completing an entire city before starting
the next. It changes scheduling, not policies or scenario contents.

Timeouts are disabled. Every completed evaluation is saved with signed inputs;
per-simulation progress, idle/queue/MLE audits and runner logs remain available.
Prepared maps and exactly matching generated scenarios are reused on restart,
as are scientifically compatible completed benchmark results. An unfinished
simulation restarts from its scenario beginning; there is no event-state
checkpoint. Incompatible inputs are rejected, not silently mixed.

The initial planning estimate is **about 5–10 days for all 420 evaluations**,
not a measured ETA. The slowest old NYC reactive-insertion completion took
about 18 hours, and Moscow is roughly twice NYC's node count with about twice
its fleet size under the same density rule. Exhaustive insertion can scale
nonlinearly; those jobs may extend the campaign beyond this range. Small-city
results arrive earlier. Re-estimate from completed large-city jobs rather than
the controller's early ETA, which is biased by fast maps/policies.

## VM locations and command

Experiment code: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-new-city-benchmarks`,
branch `codex/new-city-benchmark-suite`.

Raw maps: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/data/graphs/`.
Exact downloaded source checkout: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-new-city-map-source`.

Outputs: `/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies/`:

- `launch.json`: detached controller identity, exact command and source commit.
- `campaign.json` / `campaign_status.json`: immutable inputs and current stage.
- `prepared_maps.json` / `maps/*.prepared.json`: validation/preparation records.
- `suite/suite.json`: only the six new cities and paired scenario definitions.
- `stages/preparation.log`, `stages/suite.log`, `stages/benchmarks.log`: stage logs.
- `benchmarks/summary.csv` / `city_policy_summary.csv`: durable result tables.
- `benchmarks/<city>/<scenario>/<policy>/`: results, statuses, stamps and audits.

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet-new-city-benchmarks
base=/data/workspace/robot_delivery/dynamic-delivery-robot-fleet
"$base/.venv/bin/python" -u scripts/run_new_city_benchmarks.py \
  --graph-dir "$base/data/graphs" \
  --output-dir "$base/runs/icaps_new_cities_v7_all_policies" \
  --spatial-model "$base/runs/icaps_multicity_v2_precision/training/spatial/reservation_fcnn.npz" \
  --paper-model "$base/runs/icaps_multicity_v2_precision/training/paper/reservation_fcnn.npz" \
  --processes 12 --idle-processes 4 --prepare-processes 6
```

Use that exact command for a restart **only after confirming that no previous
controller or child simulator is still alive**. The authorized initial launch
is detached from SSH, with its identity saved in `launch.json`; do not run a
second live copy. Preserve the existing source checkout to keep signatures
compatible.

```bash
tail -f /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies/stages/preparation.log
# After preparation/scenario generation:
tail -f /data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies/stages/benchmarks.log
```
