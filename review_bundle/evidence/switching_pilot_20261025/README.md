# Bounded switching pilot evidence

Read [the report](../../../docs/SWITCHING_PILOT_RESULTS_2026-10-25.md).
This folder is saved evidence, not permission to launch or resume work.

- `PILOT_LIMITS.json`, `PILOT_CONFIG.json`, `PILOT_DATASET.json`: exact bounded
  scope, config and 18-input roster. There are two training inputs in each of
  eight cities and two separate development validation inputs.
- `inputs/`: exact scenario JSON bytes for those 18 inputs. The 12 public map
  files and fixed reservation weights are already in `review_bundle/data/`.
- `code/`: immutable adapter/runner snapshot actually used by the corrected
  pilot. The current top-level implementation additionally guards online refit
  cadence and points the planner at the measured estimate. Do not silently mix
  scientific identities or paste pilot replay into a full campaign.
- `TRAINING_ROWS.jsonl.gz`: all 1,639 immutable labeled training rows, wrapped
  as `{"source_path": "original VM path", "row": {...}}`. Each row retains the
  actor intent, model hash, ten costs, projection/window diagnostics and full
  canonical content hash. This replaces the SQLite catalog in this export.
- `run/models/`: tiny gate/pilot model payloads and pointers, including all
  published actor versions and final pilot fits. Original VM paths are retained
  as provenance. For portable inspection resolve the suffix under this folder;
  do not follow a stale absolute path or assume these are final selected models.
- `PILOT_TRAINING_RESULTS.json`, `PILOT_VALIDATION_RESULTS.json`: all 32 training
  outcomes and all 24 matched frozen validation outcomes, including timings,
  memory profiles and completed loss. Every admitted order finished.
- `PILOT_ANALYSIS.json`, `TRAINING_TIME_ESTIMATE.json`: derived offline checks
  and explicit extrapolation assumptions. Reproduce with
  `python -B operations/switching_pilot_report.py --evidence review_bundle/evidence/switching_pilot_20261025`
  from the repository root. This reads saved artifacts and writes these two
  analysis files; it never imports the production engine or fits a model.
- `gates/`: fixed-policy runtime tests, real fitting/serialization checks,
  frozen queue-completion parity, branch-order checks and durable intent/resume
  evidence. The raw legacy digest includes a nondeterministic MLE planning timer;
  the normalized gate removes only that diagnostic while computing its digest.
- `audit/`: reservation-weight training lineage, 100-row target matrices,
  source scenario documents, training metadata and numerical reconstruction.
  Its historical-immutability qualification is explicit.
- `failed_attempt/`: preserved first-attempt failure record. It exposed the
  completed-delivery/background-event clock issue fixed before the corrected
  pilot. Failed-attempt labels were not mixed into corrected training.
- `CONTINUATION_RANK_DIAGNOSTICS.json`: additional controlled action-ranking
  checks on two existing training inputs, ten candidates each. These twenty
  continuations add no scenarios or fitting labels. A common Reactive Insertion
  continuation retains remaining actual arrivals; it is a reference policy,
  not an oracle for the learned model's eventual future decisions.
- `EXPORTED_FILES.json`: exact hashes of the original VM export. Additional
  analysis, audit and final diagnostic files are covered by
  `PILOT_EVIDENCE_SOURCES.json`. `review_bundle/materialize.py` verifies that
  complete evidence manifest, as well as original maps/backend and current code.

The original VM roots are
`runs/icaps_small_city_pilot_v1_20261025` (failed, preserved) and
`runs/icaps_small_city_pilot_v2_20261025` (corrected). The two-hour safety budget
is shared across attempts and the continuation diagnostic. Up to eight episode
workers and one four-thread fitter were allowed. Full 64-worker training was
not launched. Cambridge UK, Ghent, Salamanca and Lucca were not used.

Live routing caches, the full city runtime checkpoint objects, scratch event
traces and the SQLite catalog stay on the VM; they are unnecessary for offline
review of the supplied inputs, labels, model payloads, outcomes and estimates.
Small independent gate checkpoint fixtures are supplied. The existing environment
is not vendored; its direct package pins and fitting-thread diagnostics are
provided elsewhere in this bundle.
