# Adaptive meta-policy training: implementation guide

This implements the offline training loop in `adaptive_meta_policy_proposal.md`.
No training, campaign evaluation, Git push or campaign recovery is triggered
by adding or importing these files. All commands below are for a future
explicit launch. Unit tests use independent tiny synthetic fixtures.

Work in the dedicated `dynamic-delivery-robot-fleet-meta-policy` worktree
(branch `meta-policy-fixes`). Do **not** copy the meta scripts into the frozen
benchmark checkout: `run_multicity_experiments.py` fingerprints every
`scripts/*.py`, so extra scripts there block resuming the v7 campaign.
`docs/adaptive_meta_changes.md` lists what changed after the review.

## Training semantics

The default portfolio is six complementary policies chosen from the 35
completed test scenarios in the training cities:

- `full_no_idle` and `paper_sa_adapted`: no idle control, not queue-aware.
  Best in Paris, Barcelona and Johannesburg; up to 4.6x worse when chargers
  congest.
- `full_queue_aware` and `full_coordinated_idle` (NN reservation) and their
  no-reservation twins `queue_mle_reservation` and `full_mle_reservation`:
  queue-aware with legacy or coordinated idle control. Never worse than 1.2x
  the best; best in Moscow, NYC and the small cities. MLE reservation never
  activated in any of the 72 campaign runs, so the two MLE policies behave as
  "no reservation", which gives a clean NN-on/off contrast for both idle
  styles. `full_mle_reservation` is also the best single fixed policy.

Choosing per scenario among these six equals the best of all eight
candidates; the best single fixed policy is 4.4% worse and the best policy
per city (in hindsight) 0.6% worse, so the selector must beat a static
per-city choice to show real adaptivity. `reactive_insertion` (never best,
~8x slower on big maps), `myopic_ab`, `full` and `full_no_nn` (dominated) are
not in the default portfolio but remain available. Beijing and Sydney are excluded from training
by default (`--exclude-cities`): one episode would cost >=1,100 and ~290 CPU
hours. One shared
physical event engine owns robots, batteries, cargo, service locks, prepared
routes, charger sessions/FIFO queues, pending orders and the event heap. Its
candidate search functions are taken from each original generated policy
program. The original scripts remain unchanged. Assertions on source structure
fail closed if an upstream patch moves the physical event loop.

At a decision time t:

1. Extract only observable features once and choose an action BEFORE any
   branch is evaluated. Bootstrap rounds use deterministic exploration:
   a fraction of episodes follow one fixed policy throughout, otherwise the
   previous policy is kept with probability `stickiness` (default 0.75) and a
   portfolio policy is drawn uniformly otherwise. Later rounds follow a frozen
   selector. The first decision of an episode has no incumbent, so the
   selector picks its plain argmin there (no hysteresis).
2. Decide whether this step is labelled (probability `label_step_probability`,
   default 0.25, deterministic per seed/episode/step). Unlabelled steps only
   run the chosen policy.
3. For a labelled step, save a complete state snapshot (including all
   controllers, the current reservation assignment, the queue forecast and
   the event tie-breaking counter) and restore it for every policy and
   holding duration. All controllers observe every arrival, including when
   inactive. One realized training future per snapshot is used; additional
   independent episodes provide more samples.
4. Execute the candidate for 15 simulated minutes by default. Events at the
   boundary timestamp settle under the outgoing policy. Re-selecting the
   active policy is a no-op, so "stay" is exactly the uninterrupted policy.
5. Save the selected policy's boundary state before any scoring continuation.
   The actual episode always continues from the chosen policy's branch; the
   other candidates are only scored (for their labels) and then discarded.
   Branches never expand further: there is no tree of alternatives. A test
   replays only the recorded choices, without any alternatives, and reproduces
   the collected episode exactly.
6. For scoring only, admit orders released through t + 60 minutes, switch to the
   common continuation policy (default `full_no_idle`, a fast top-K shortlist
   policy; `reactive_insertion` searches every robot and is 6–14× slower in
   the campaign data) and finish every outstanding delivery. The label is the
   final loss minus the already completed loss at t. If two candidates reach a
   byte-identical tail-start state, the tail is computed once and its loss is
   reused (the simulator is deterministic; this is exact). A failed or
   incomplete drain is an error, never a zero label.
7. Commit only the chosen boundary snapshot, checkpoint it atomically and repeat.
   With `decide_until = all_delivered` (default for new manifests), decisions
   continue after the last arrival until every order is delivered; in the big
   cities most deliveries happen after the 720-minute arrival horizon.

This is a finite-horizon supervised counterfactual target with a fixed
continuation policy. It does not establish optimal long-term switching. The
60-minute horizon and the continuation choice must be validated in the pilot
(do the policy rankings agree under continuation = candidate?).
Realized training futures may be used as privileged LABELS, never FEATURES.
`--mode validate/evaluate` runs only the chosen policy, never the tournament.

## Features and models

Features (version 2) include graph size/density, bounding-box area, edge
lengths, cluster counts/sizes, charger density, fleet size/type mix and
density, busy/charging/waiting fractions, projected battery mean/std/min/low
fraction per type, revealed outstanding orders per importance, deadline
slack/overdue share, recent arrival rates and their 15-minute rate trend,
Gamma-Poisson posterior rate means/std, pickup entropy, FIFO charger wait
estimates, shares of robots repositioning or holding a charge intent, whether
a reservation is active, the current policy, policy age, previous-window loss
rate, whether the arrival horizon has passed, and the proposed holding
duration. Extraction is read-only (it writes no simulator cache). No city
one-hot is used, but map features can still identify a city, so claims about
unseen cities need leave-one-city-out evaluation.

The default model predicts **advantages**: for each state, the per-robot loss
of every policy minus the portfolio mean. One histogram gradient-boosted head
per policy models that difference and one "level" head models the mean, so
absolute predictions remain available. Most of each label is common to all
policies; removing it stops independent heads from making independent errors
on the shared part. `--target absolute` keeps the old per-policy target for
ablations. The optional NumPy MLP uses all rows with episode-weighted
minibatch losses (no bootstrap resampling).

Switching: a challenger must beat the incumbent by at least 5% of the
incumbent's predicted loss AND by `margin_scale ×` the 90th percentile of
validation residuals of the **predicted difference** between that pair of
policies (episode-weighted). These are heuristics, not confidence intervals.
With adaptive holding, a longer hold requires the same winner with a margin at
every shorter tested horizon too. Keep adaptive holding off until a fixed
15-minute selector has been validated.

Explicit train/validation groups are required. The same episode, scenario seed,
or actual workload cannot cross splits; test labels are rejected. Fitting
reports both the raw-argmin regret and the regret of the deployed `choose()`
rule (with hysteresis), the "stay with the incumbent" and "best fixed policy"
regrets, top-1 accuracy, rank correlation, switch and veto rates — per row,
per episode and per city. These are offline diagnostics; online validation is
still required (below).

## Fresh data and leakage protection

The manifest preparer creates NEW scenarios: 20 training and 5 validation per
city by default, with distinct deterministic seeds. Several reference suites
can be combined (`--reference-suite` repeatable), e.g. the 5 original and the
6 new cities for one 11-city selector. Existing suites supply prepared graphs
and the same demand-generator distribution. Priors use only the NEW training
arrivals, with a documented 0.5 pseudocount per importance class.

All archived suites discovered in the reference repository are pinned as
reserved data (`--reserved-suite` adds others). Collection verifies their
checksums and rejects any training/validation seed or workload matching a
reserved scenario; workload fingerprints ignore names, order ids and seeds.

Selector provenance records every scenario that informed it AND which of
them were fitting vs held-out validation. That enables three modes:

| Mode | Episodes | Allowed scenarios |
|---|---|---|
| `collect` | train + validation | fresh only (reserved data rejected) |
| `validate` | validation | only this selector's held-out validation scenarios |
| `evaluate` | test | none that informed the selector |

`<manifest>_comparison.json` points to the original suites' exact test files,
maps and priors. Report only paired cases with completed, strictly validated
baselines, and parity-check that a fixed policy in the shared runtime
reproduces a stored baseline before reusing it. These old tests have informed
design, so describe this as an existing-benchmark comparison; keep a separate
untouched final test set for independent conclusions.

## Identity and resume

Labels depend only on the **simulator identity**: the ten generated policy
programs plus the local import closure of `runtime.py`, `labels.py` and
`exact_runtime.py` (computed by parsing imports), the base models, controller
settings, label horizon and continuation. Editing `learning.py`, `pipeline.py`,
the training script or an unrelated benchmark script no longer invalidates
collected labels or blocks evaluation. Features have a separate identity
(`features.py` + `VERSION`); a selector is only used with the feature
extractor it was trained with.

Every rollout writes an atomic JSON receipt (compatibility fingerprint, not a
cryptographic signature of its content). Restart the same command and output
directory to reuse completed episodes, rollouts and committed states. One
failed episode no longer stops the others: failures are recorded in
`failure.json` and raised at the end. `progress.json` is finalized as
`complete`, `pilot_stopped` or `failed`, and every status file records PID,
host, boot id and process start time, so a stale "running" label can be
recognized after a crash or reboot.

## Parallelism and memory

`--processes 60` is the TOTAL compute-process ceiling: one coordinator plus at
most 59 spawned episode workers, further capped by CPU count, episode count
and available RAM after a 16 GiB reserve. Each episode gets a fresh worker
process (`max_tasks_per_child=1`), and episodes are started longest-first
(largest map, most orders). Within a worker, candidates are serial and restore
the same snapshot; there are no nested idle, MLE, index-preparation or BLAS
pools.

On Linux, the exact distance/predecessor index is shared through memory-mapped
files. It is SHA-256-verified once; a `meta_verified.json` record of file size,
mtime and inode then lets later simulators skip re-hashing (Beijing's index is
23 GiB) unless a file changes. To reuse an existing index, COPY its key folder
into the meta cache folder rather than pointing at a live campaign's folder.
The idle planners share the engine's graph-sized tables, and controller
cluster tables are shared as soon as each controller is built.

Measured index sizes: Beijing 23.2 GiB, Sydney 8.7, Moscow 6.5, Johannesburg
1.2, Paris 0.03, New Delhi 0.002.

Cost model (from the v3-v7 runner logs): almost all simulator wall time is
spent while orders arrive (the first 720 minutes); the post-arrival delivery
phase is nearly free (Sydney: 70 simulated hours in under 30 s). One episode
therefore costs about one fixed run for the behaviour trajectory plus, per
labelled decision, 15 minutes of each candidate and 45 minutes of the
continuation. With the default 6 policies, label probability 0.25 and the 9
training cities, collection is estimated at ~3,050 CPU hours and ~3.9 days of
wall time (25 Moscow episodes of ~93 h each run in parallel; all other cities
finish within hours). With 25 instead of 59 workers it is ~5.1 days. These are
estimates; the pilot replaces them with measurements.

## Commands for a later launch (not executed)

Run from the meta worktree with its Python environment
(`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/.venv/bin/python`).

1. Fresh scenarios and both manifests (DATA ONLY; no policies are run). With the
   defaults this gives 9 cities (Beijing and Sydney excluded) x (20 + 5) new
   scenarios and the 6-policy portfolio:

```bash
python scripts/prepare_meta_manifest.py \
  --reference-suite /abs/runs/icaps_multicity_v2_precision/suite/suite.json \
  --reference-suite /abs/runs/icaps_new_cities_v7_all_policies/suite/suite.json \
  --spatial-model /abs/runs/icaps_multicity_v2_precision/training/spatial/reservation_fcnn.npz \
  --paper-model /abs/runs/icaps_multicity_v2_precision/training/paper/reservation_fcnn.npz \
  --output runs/meta_v2/manifest.json
```

2. Pilot: measure cost before anything big (stops each episode after N
   decisions; resumable later with the same output folder):

```bash
python scripts/generate_meta_training_data.py --manifest runs/meta_v2/manifest.json \
  --output-dir runs/meta_v2/round0 --processes 8 --memory-per-worker-gib 24 --max-steps 4
python scripts/summarize_meta_run.py runs/meta_v2/round0
```

3. Collect and fit:

```bash
python scripts/generate_meta_training_data.py --manifest runs/meta_v2/manifest.json \
  --output-dir runs/meta_v2/round0 --processes 60 --memory-per-worker-gib 16
python scripts/train_meta_policy.py --data runs/meta_v2/round0 --output runs/meta_v2/selector_round0.pkl
```

4. Tune the switching rule ONLINE on held-out validation episodes:

```bash
python scripts/configure_meta_selector.py --selector runs/meta_v2/selector_round0.pkl \
  --output runs/meta_v2/selector_round0_m05.pkl --margin-scale 0.5
python scripts/generate_meta_training_data.py --mode validate --manifest runs/meta_v2/manifest.json \
  --selector runs/meta_v2/selector_round0_m05.pkl --output-dir runs/meta_v2/validate_m05
```

5. Optional DAgger round following the frozen selector (new seed and output),
   then refit on all rounds. Leave-one-city-out: `--exclude-city X
   --validation-city X` at fitting time.
6. Final evaluation of ONE frozen selector variant:

```bash
python scripts/generate_meta_training_data.py --mode evaluate \
  --manifest runs/meta_v2/manifest_comparison.json \
  --selector runs/meta_v2/selector_final.pkl --output-dir runs/meta_v2/comparison
```

Use the same portfolio, windows and settings throughout. Report actual online
losses (paired by scenario, with episode-level uncertainty), not only offline
prediction regret. Fixed-policy baselines for "best fixed" and "best fixed per
city" should be chosen on the fresh validation scenarios, not on test results.

## Not implemented yet (see `adaptive_meta_changes.md`)

- Parallel candidate branches inside one episode (fork-based workers).
- A terminal value instead of a full drain in heavily overloaded regimes.
- Per-city memory classes for worker admission.
- A static per-city selector baseline and a multi-future rollout selector.
