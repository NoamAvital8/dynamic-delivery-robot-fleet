from __future__ import annotations

"""Train the joint reservation FCNN from an offline target dataset.

The input NPZ must contain ``features`` with shape ``(N, F)`` and ``targets``
with shape ``(N, K, C)``.  Targets are perfect-information reservation
fractions generated on training scenarios, never from the test future.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.fleet import RobotType
from delivery_fleet.reservation_nn import ReservationFCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--importance-levels", default="1,2,5")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    levels = tuple(float(value) for value in args.importance_levels.split(","))
    robot_types = tuple(robot_type.value for robot_type in RobotType)
    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=float)
        targets = np.asarray(data["targets"], dtype=float)

    model = ReservationFCNN(
        features.shape[1],
        robot_types,
        levels,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    initial_loss = model.loss(features, targets)
    history = model.fit(
        features,
        targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"saved={args.output} initial_loss={initial_loss:.6f} final_loss={history[-1]:.6f}")


if __name__ == "__main__":
    main()
