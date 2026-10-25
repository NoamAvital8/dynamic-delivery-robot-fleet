"""Fit a loss selector from already generated labels; never run simulators."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, type=Path, help="One or more compatible collection round directories.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("trees", "nn"), default="trees")
    parser.add_argument("--target", choices=("advantage", "absolute"), default="advantage",
                        help="advantage: per-policy loss minus the portfolio mean (default); absolute: legacy.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--default-holding-minutes", type=float, default=15.)
    parser.add_argument("--minimum-gain-fraction", type=float, default=.05)
    parser.add_argument("--margin-scale", type=float, default=1.,
                        help="Multiplier on validation pair-difference residual margins (tune online in validate mode).")
    parser.add_argument("--adaptive-holding", action="store_true")
    parser.add_argument("--exclude-city", action="append", default=[],
                        help="Drop this city's rows from fitting (leave-one-city-out). Repeatable.")
    parser.add_argument("--validation-city", action="append", default=[],
                        help="Keep only these cities' validation rows. Repeatable.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cpu_capacity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if not 1 <= args.threads <= cpu_capacity:
        parser.error(f"--threads must be in 1..{cpu_capacity}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threadpoolctl import threadpool_limits
    from delivery_fleet.meta_policy.pipeline import collect_examples
    from delivery_fleet.meta_policy.learning import build_provenance, fit_selector
    from delivery_fleet.meta_policy.artifacts import RunLock, atomic_json, digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with RunLock(args.output.with_suffix(".lock")), threadpool_limits(limits=args.threads):
        if args.output.exists():
            parser.error("model output exists; choose a new versioned model filename")
        records = collect_examples(args.data)
        excluded = set(args.exclude_city)
        records = [r for r in records if not (r["split"] == "train" and r["city"] in excluded)]
        if args.validation_city:
            keep = set(args.validation_city)
            records = [r for r in records if r["split"] == "train" or r["city"] in keep]
        selector, report = fit_selector(records, kind=args.model, seed=args.seed,
            default_window=args.default_holding_minutes, adaptive=args.adaptive_holding,
            minimum_gain_fraction=args.minimum_gain_fraction, max_iter=args.max_iter,
            checkpoint_dir=args.output.with_suffix(".fit_checkpoints"), target=args.target,
            margin_scale=args.margin_scale, progress=lambda message: print(message, flush=True))

        selector.provenance = build_provenance(records,
            excluded_training_cities=sorted(excluded), validation_cities=sorted(args.validation_city),
            data_directories=[str(p.resolve()) for p in args.data], seed=args.seed,
            kind=args.model, target=args.target, max_iter=args.max_iter)
        selector.save(args.output)
        atomic_json(args.output.with_suffix(".report.json"), {**report,
            "model_sha256": digest(args.output), "provenance": selector.provenance})
        decisions = report["validation_decisions"].get("per_episode_mean", {})
        print(f"saved {args.output}; validation per-episode regret/robot: "
              f"raw argmin={decisions.get('raw_argmin_regret', float('nan')):.6g} "
              f"deployed rule={decisions.get('rule_regret', float('nan')):.6g} "
              f"best fixed={decisions.get('best_fixed_regret', float('nan')):.6g}")


if __name__ == "__main__":
    main()
