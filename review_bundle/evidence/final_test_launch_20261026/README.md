# Final-test launch evidence

Read [the final-test launch record](../../../docs/FINAL_TEST_LAUNCH_2026-10-26.md).
These are saved observations at launch, not completed test results.

- `FULL_TEST_LAUNCH.json`: the human-authorized opening, command, service and scope.
- `frozen_models.json`: exact pre-test model/benchmark freeze and the baseline
  selected on development data. Selector payloads with matching hashes are in
  `../full_development_status_20261026/models/`.
- `fixed/`: the exact ten fixed-policy payloads named in the freeze.
- `validation_summary.json`, `FULL_DEVELOPMENT_COMPLETE.json`: completed
  development evidence. These are not final-test results.
- `test_manifest.json`, `TEST_DATA_VERIFIED.json`, `test_opened.json`: 200
  final-test inputs, 50 in each of four cities, unique and disjoint from development.
- `TEST_SCENARIOS.jsonl.gz`: exact UTF-8 input bytes for all 200 test scenarios.
  Each JSONL record contains `episode`, `relative_path`, `sha256` and
  `scenario_json_utf8`. Writing the latter as UTF-8 reproduces the original file.
- `PROTECTED_DEVELOPMENT_FILES.json`, `STARTUP_AUDIT.json`: pre-test hashes and
  observed unchanged models, development artifacts and training replay.
- `run_switching_final_test.py`: byte-preserved launch supervisor. Its hash
  matches the receipt; normal source is also under `operations/`.
- `SOURCE_MANIFEST_BEFORE_TEST.json`, `SOURCE_VERIFICATION_BEFORE_TEST.json`:
  preserved source-verification history before this new test launch.
- `TEST_LAUNCH_EVIDENCE_SOURCES.json`: exact evidence hashes.

The four test cities are now opened for evaluation, following model freezing.
They remain excluded from all training and development model selection. No
partial test outcome is used to update weights, thresholds or the fixed baseline.
Live test checkpoints and logs stay on the VM while evaluation runs.
