# Full-development status evidence, 26 October 2026

See [the dated status report](../../../docs/FULL_DEVELOPMENT_STATUS_2026-10-26.md).
These are timestamped read-only observations and exact copies of existing VM
artifacts. No training, scenario generation, service restart or final test was
started by this status check.

- `STATUS_AUDIT.json`: completed training counts, saved model pointers, partial
  validation counts, five-second resource measurements, holdout checks and ETA.
- `models/xgboost.json`, `models/nn.json`: exact post-training payloads currently
  fixed for development validation. Their hashes match the audit pointers.
  They are not accompanied by a completed validation/model-selection claim.
- `recovery/`: the active process-lifecycle override, resume entries, focused
  tests and existing recovery receipts/notes copied from the VM. These require
  the original VM campaign context; inspect them rather than launching them.
- `STATUS_EVIDENCE_SOURCES.json`: exact file hashes for this saved snapshot.

The execution-only recovery was already active when this check began. It is
documented separately from the original sealed `src/fleet_switching` source;
no scientific training logic or active process was modified in this turn.
Large live replay/checkpoint data and the full pre-recovery backup stay on the
VM. All 1,000 immutable development scenarios are already archived in the
previous launch evidence folder. The four final-test cities remain unopened.
