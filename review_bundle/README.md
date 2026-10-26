# Portable review bundle

Start with [the project and switching-model handoff](../PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md) on branch `codex/small-city-switching-review-20261025`.

- `backend/`: preserved simulator source at commit `5e76f5384aec98e9d5a538b27dae205df301fa28`.
- `data/`: 12 compressed prepared graphs, eight boundaries, map provenance, and two fixed reservation FCNN weights.
- `evidence/`: new build receipts and historical training/scenario/comparison exports; historical data is not new-model replay.
- `PROVENANCE.json`: original export checksums and omissions.
- `requirements-review.txt`: packages for source and synthetic-test review.
- `materialize.py`: read-only verification by default; optional `--execute` only decompresses maps and writes portable paths in `.review-work/`. It does not import a simulator, generate scenarios, or train models.

A separately authorized bounded development pilot has run. See
[its report](../docs/SWITCHING_PILOT_RESULTS_2026-10-25.md) and
`evidence/switching_pilot_20261025/` for input rosters, replay, small fitted
models, matched outcomes, integration gates, reservation-model provenance and
the measured training-time estimate. The full development campaign subsequently started; see
[the launch record](../docs/FULL_DEVELOPMENT_LAUNCH_2026-10-25.md) and
`evidence/full_development_launch_20261025/`. The four test cities were held out throughout development; their separately authorized evaluation is now running. Read the main handoff's
limitations before interpreting these small development results.

The owner subsequently authorized final test on 26 October after model freezing.
See [the final-test launch record](../docs/FINAL_TEST_LAUNCH_2026-10-26.md) and
`evidence/final_test_launch_20261026/`. Test outcomes are evaluation-only; no
training, tuning or fixed-baseline selection uses those outcomes.
