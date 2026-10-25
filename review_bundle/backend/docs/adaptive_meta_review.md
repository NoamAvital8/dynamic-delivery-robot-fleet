# Adaptive meta-policy: reviewer entry point

This is a review implementation, not a launched training campaign. Uploading or
importing these files does not generate project scenarios, train a selector,
start benchmark simulations, or recover any stopped campaign.

## What to share

Give reviewers read access to the complete repository checkout, because the
adaptive runtime reuses the existing simulator and policy implementations.
The VM checkout is:

```text
/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-meta-policy   (branch meta-policy-fixes)
```

This is a dedicated worktree, separate from the frozen benchmark checkout
(`...-new-city-benchmarks`), so the meta scripts do not change the benchmark
fingerprint there. The adaptive files are uncommitted in this worktree; the
GitHub branch alone does not contain them. `docs/adaptive_meta_changes.md`
lists every change made after the first review. Existing benchmark outputs and
trained reservation models remain in the original repository, not this code
directory. Reviewing the code does not require launching those workloads.

## Read in this order

| Area | Files | Main review questions |
| --- | --- | --- |
| Intended behavior | `docs/adaptive_meta_policy_proposal.md`, `docs/adaptive_meta_training.md` | Are the time-window decisions and finite-horizon counterfactual targets appropriate? |
| Shared physical state | `src/delivery_fleet/meta_policy/runtime.py`, `src/delivery_fleet/meta_policy/labels.py` | Do policy switches preserve routes, cargo, battery, queues, event order and all demand observers? Does snapshot/restore isolate candidate branches? |
| Fresh data and leakage protection | `src/delivery_fleet/meta_policy/fresh_scenarios.py`, `src/delivery_fleet/meta_policy/separation.py`, `scripts/prepare_meta_manifest.py` | Are training and validation entirely new and disjoint from every reserved scenario? Are original comparison scenarios preserved exactly? |
| Features and selector | `src/delivery_fleet/meta_policy/features.py`, `src/delivery_fleet/meta_policy/learning.py` | Do features exclude unrevealed orders? Are normalization, fitting, margins and holding decisions statistically sound? |
| Collection, resources and resume | `src/delivery_fleet/meta_policy/pipeline.py`, `src/delivery_fleet/meta_policy/artifacts.py` | Is the selected action chosen before observing candidate outcomes? Is only its boundary state retained? Are process/RAM limits, input fingerprints and atomic resume correct? |
| Entry points and dependencies | `scripts/generate_meta_training_data.py`, `scripts/train_meta_policy.py`, `scripts/configure_meta_selector.py`, `scripts/summarize_meta_run.py`, `configs/meta_training.example.json`, `configs/meta_training.requirements.txt` | Are launch, evaluation and dependency instructions clear and safe? |
| Correctness tests | `tests/test_meta_runtime.py`, `tests/test_meta_training.py`, `tests/test_meta_separation.py`, `tests/test_meta_fixes.py` | Are fixed-policy equivalence, mixed switches, branch isolation, charging, fresh-data separation and interruption/resume covered adequately? |

The reused code in `scripts/run_*`, `src/delivery_fleet/` outside `meta_policy`,
and `operations/exact_runtime.py` is part of the review context. The new files
do not modify those original policy implementations.

The adaptive runtime uses an explicitly pickleable integer event counter for
Python 3.14 compatibility. Its first value is the original engine counter's
exact next value; checkpointing/restoring must preserve tie-breaking ids without
advancing them. Dedicated regression tests cover this behavior.

## Checks reviewers may run

From the checkout root, use an environment with the existing simulator
dependencies and the additional dependencies listed in
`configs/meta_training.requirements.txt`. On this VM, the existing environment is
`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/.venv/bin/python`.

```bash
PYTHON=/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/.venv/bin/python
"$PYTHON" -m pytest -q tests/test_meta_runtime.py tests/test_meta_training.py tests/test_meta_separation.py tests/test_meta_fixes.py
"$PYTHON" scripts/prepare_meta_manifest.py --help
"$PYTHON" scripts/generate_meta_training_data.py --help
"$PYTHON" scripts/train_meta_policy.py --help
```

The tests use tiny independent synthetic fixtures, not benchmark maps/scenarios.
Some tests fit tiny temporary models and run tiny fixture simulations; they do
not launch the project training or benchmark campaigns. Do not run the full
launch examples in the training guide without the user's explicit approval.

## Important limitations to scrutinize

- Labels use one realized future per state and a common reactive continuation,
  not an exact optimal value function or a multi-future Monte Carlo expectation.
- Validation-error margins and adaptive holding rules are heuristics, not
  calibrated confidence probabilities or guarantees of selecting the best policy.
- Fixed-policy agreement on tiny fixtures is evidence, not proof of equivalence
  on all maps, states and policy-switch sequences.
- Large-map memory/runtime and Linux shared-index behavior need representative
  validation before production training. A 60-process ceiling is not a promise
  that 60 large-map workers fit safely in RAM.
- Old benchmark results already informed design. Comparison against them is a
  matched existing-benchmark evaluation, not an untouched final test set.
- Existing reservation NNs stay frozen. Their own pretraining provenance must
  be disclosed separately from the new selector's fresh training/validation data.
- Pickle snapshots/models are trusted-local artifacts only.

Please report findings with file/line references and a reproducible small test
where possible. Do not edit the running/stopped benchmark campaigns or retrain
the fixed reservation NNs as part of review.
