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
the measured training-time estimate. Full training and final-test evaluation
remain unstarted. The 4 test cities remain held out. Read the main handoff's
limitations before interpreting these small development results.
