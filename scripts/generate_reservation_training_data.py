from __future__ import annotations

"""Build FCNN targets from perfect-information candidate evaluations.

The JSON input is a list of records.  Each record contains a ``features``
vector and ``candidates``; every candidate contains a ``fractions`` matrix and
the final simulated ``loss`` obtained on that offline training scenario.  The
minimum-loss candidate becomes the supervised target.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.reservation_nn import select_perfect_information_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.evaluations.read_text(encoding="utf-8"))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    for record in records:
        candidates = [np.asarray(item["fractions"], dtype=float) for item in record["candidates"]]
        by_bytes = {
            candidate.tobytes(): float(item["loss"])
            for candidate, item in zip(candidates, record["candidates"], strict=True)
        }
        target, loss = select_perfect_information_target(
            candidates, lambda candidate: by_bytes[candidate.tobytes()]
        )
        features.append(np.asarray(record["features"], dtype=float))
        targets.append(target)
        losses.append(loss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        targets=np.stack(targets),
        perfect_information_loss=np.asarray(losses),
    )
    print(f"saved={args.output} samples={len(features)}")


if __name__ == "__main__":
    main()
