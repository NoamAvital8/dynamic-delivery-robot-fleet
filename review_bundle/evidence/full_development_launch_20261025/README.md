# Full development campaign launch evidence

See [the launch record](../../../docs/FULL_DEVELOPMENT_LAUNCH_2026-10-25.md).
These are timestamped startup snapshots, not current live progress or completed
training results. The four final-test cities remain unopened.

- `FULL_RUN_LAUNCH.json`: detached command, launch time, scope, hashes and PID.
- `LAUNCH_CONFIG.json`, `campaign.json`: exact settings and scientific identity.
- `SOURCE_MANIFEST_AT_LAUNCH.json`, `SOURCE_VERIFICATION_BEFORE_LAUNCH.json`:
  exact verified build evidence before authorization to start full training.
  Later Git documentation records that training is now running. The underlying
  simulator/collector/projector/learner package remains unchanged.
- `run_switching_development.py`: exact byte-preserved supervisor used at launch;
  its hash matches the launch receipt. The normal entry file is also tracked
  under the repository's `operations/` directory.
- `FULL_DEVELOPMENT_DATA_VERIFIED.json`, `development_manifest.json`: all 800
  training and 200 disjoint development validation inputs and content hashes.
- `DEVELOPMENT_SCENARIOS.jsonl.gz`: all 1,000 exact scenario documents. Each line
  contains `episode`, `relative_path`, `sha256`, and `scenario_json_utf8`.
  To reconstruct a file, write `record['scenario_json_utf8'].encode('utf-8')`
  to a chosen portable location and verify its SHA-256. Do not execute a
  simulator to inspect this archive.
- `STARTUP_RESOURCE_CHECK.json`: five-second CPU measurement and controller
  snapshot with 64 active simulation workers; values can change during the run.
- `STARTUP_STATUS.json`: a later timestamped progress snapshot, including both
  families' first real model updates when observed.
- `LAUNCH_EVIDENCE_SOURCES.json`: checksums for the archived launch evidence.

Raw maps and fixed reservation weights are already in `review_bundle/data/`.
Live replay, evolving model files, routing caches, logs and full city checkpoints
remain on the VM. They are not represented as completed/final artifacts here.
No held-out inputs, simulations or final-test outcomes are included.
