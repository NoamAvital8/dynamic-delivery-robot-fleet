from __future__ import annotations

"""Train paper-aligned heterogeneous reservation FCNNs in parallel."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.fleet import RobotType
from delivery_fleet.reservation_nn import ReservationFCNN


def _fit_restart(payload: dict[str, Any]) -> dict[str, Any]:
    model = ReservationFCNN(
        payload["input_dim"],
        payload["robot_types"],
        payload["importance_levels"],
        hidden_dim=payload["hidden_dim"],
        seed=payload["seed"],
    )
    history = model.fit(
        payload["train_x"],
        payload["train_y"],
        epochs=payload["epochs"],
        learning_rate=payload["learning_rate"],
        weight_decay=payload["weight_decay"],
    )
    validation_loss = model.loss(payload["validation_x"], payload["validation_y"])
    return {
        "seed": payload["seed"],
        "training_loss": history[-1],
        "validation_loss": validation_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument(
        "--restarts",
        type=int,
        default=None,
        help="Independent initializations; defaults to --processes.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="0 uses the paper default of C hidden tanh neurons.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads-per-process", type=int, default=1)
    args = parser.parse_args()
    restarts = args.processes if args.restarts is None else args.restarts
    if args.processes <= 0 or restarts <= 0:
        parser.error("--processes and --restarts must be positive")
    if args.threads_per_process <= 0:
        parser.error("--threads-per-process must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")

    thread_count = str(args.threads_per_process)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count

    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=float)
        targets = np.asarray(data["targets"], dtype=float)
        metadata = (
            json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        )
    if len(features) < 2:
        raise ValueError("at least two training instances are required")
    robot_types = tuple(
        metadata.get("robot_types", [item.value for item in RobotType])
    )
    levels = tuple(
        float(value) for value in metadata.get("importance_levels", (1, 2, 5))
    )
    if features.shape[1] != len(levels) + 1:
        raise ValueError("dataset must use the paper's C+1 input feature layout")
    ReservationFCNN(
        features.shape[1], robot_types, levels, hidden_dim=None, seed=args.seed
    ).loss(features, targets)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(features))
    validation_count = min(
        len(features) - 1,
        max(1, int(round(len(features) * args.validation_fraction))),
    )
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    hidden_dim = None if args.hidden_dim == 0 else args.hidden_dim
    base_payload = {
        "input_dim": features.shape[1],
        "robot_types": robot_types,
        "importance_levels": levels,
        "hidden_dim": hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_x": features[train_indices],
        "train_y": targets[train_indices],
        "validation_x": features[validation_indices],
        "validation_y": targets[validation_indices],
    }

    worker_count = min(args.processes, restarts)
    print(
        f"samples={len(features)} train={len(train_indices)} "
        f"validation={validation_count} restarts={restarts} processes={worker_count}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = []
        for index in range(restarts):
            payload = dict(base_payload)
            payload["seed"] = args.seed + index
            futures.append(executor.submit(_fit_restart, payload))
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed_count}/{restarts}] seed={result['seed']} "
                f"train={result['training_loss']:.6f} "
                f"validation={result['validation_loss']:.6f}",
                flush=True,
            )

    best = min(
        results, key=lambda result: (result["validation_loss"], result["seed"])
    )
    model = ReservationFCNN(
        features.shape[1],
        robot_types,
        levels,
        hidden_dim=hidden_dim,
        seed=best["seed"],
    )
    final_history = model.fit(
        features,
        targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    report = {
        "dataset": str(args.dataset.resolve()),
        "model": str(args.output.resolve()),
        "paper_architecture": "C+1 inputs, one C-neuron tanh hidden layer",
        "heterogeneous_output": "joint KxC output with per-type softmax",
        "processes": worker_count,
        "restarts": restarts,
        "selected_seed": best["seed"],
        "selection_training_loss": best["training_loss"],
        "selection_validation_loss": best["validation_loss"],
        "final_all_training_instances_loss": final_history[-1],
    }
    report_path = args.output.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
