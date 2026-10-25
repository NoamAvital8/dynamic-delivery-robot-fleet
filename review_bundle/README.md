# Portable review bundle

Start with [the project and switching-model handoff](../PROJECT_AND_SWITCHING_META_REVIEW_2026-10-25.md) on branch `codex/small-city-switching-review-20261025`.

- `backend/`: preserved simulator source at commit `5e76f5384aec98e9d5a538b27dae205df301fa28`.
- `data/`: 12 compressed prepared graphs, eight boundaries, map provenance, and two fixed reservation FCNN weights.
- `evidence/`: new build receipts and historical training/scenario/comparison exports; historical data is not new-model replay.
- `PROVENANCE.json`: original export checksums and omissions.
- `requirements-review.txt`: packages for source and synthetic-test review.
- `materialize.py`: read-only verification by default; optional `--execute` only decompresses maps and writes portable paths in `.review-work/`. It does not import a simulator, generate scenarios, or train models.

No new model has been trained and no new delivery simulation has run. Read the main handoff's limitations and review questions before interpreting test results or historical comparisons.
