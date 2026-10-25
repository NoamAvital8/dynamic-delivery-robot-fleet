"""Explicit opt-in entry point; importing it does not run simulations.

Modes:
  collect   counterfactual labels on fresh train/validation episodes
  validate  run a frozen selector online on its own held-out validation episodes
  evaluate  run a frozen selector on test episodes (never used for fitting/validation)
Use --max-steps N for a pilot that measures cost without finishing episodes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from delivery_fleet.meta_policy.pipeline import MODES, run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=MODES, default="collect")
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--processes", type=int, default=60, help="Total compute budget INCLUDING coordinator, maximum 60.")
    parser.add_argument("--memory-per-worker-gib", type=float, default=8., help="Estimated private worker RAM; increase for large maps.")
    parser.add_argument("--memory-reserve-gib", type=float, default=16.)
    parser.add_argument("--shared-cache", type=Path,
                        help="Exact-index cache folder. Copy an existing index here rather than pointing at a live campaign.")
    parser.add_argument("--seed", type=int, default=42, help="Change for a new exploration round and output directory.")
    parser.add_argument("--max-steps", type=int,
                        help="Pilot: stop each episode after N decisions (no drain, resumable later).")
    parser.add_argument("--describe", action="store_true", help="Read inputs and print budget; do not load maps or run an episode.")
    args = parser.parse_args()
    run(args.manifest, args.output_dir, processes=args.processes,
        memory_per_worker_gib=args.memory_per_worker_gib, reserve_gib=args.memory_reserve_gib,
        selector=args.selector, seed=args.seed, mode=args.mode,
        shared_cache=args.shared_cache, describe=args.describe, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
