from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.scenario_creator import NodeDemandProfile, ScenarioCreator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=ROOT / "data/graphs/tel_aviv.graphml")
    parser.add_argument("--output", type=Path, default=ROOT / "data/scenarios/tel_aviv_small_3h_seed42.json")
    args = parser.parse_args()

    graph = nx.read_graphml(args.graph, node_type=int)
    nodes = np.asarray(list(graph.nodes), dtype=np.int64)
    if len(nodes) < 2:
        raise RuntimeError("graph is too small")

    robots = max(1, math.ceil(len(nodes) / 500))
    total = min(240, max(120, int(round(3.0 * robots))))
    hourly = np.asarray([0.28, 0.42, 0.30], dtype=np.float64) * total

    lat = np.asarray([float(graph.nodes[int(n)]["y"]) for n in nodes])
    lon = np.asarray([float(graph.nodes[int(n)]["x"]) for n in nodes])
    centers = [
        (32.0741, 34.7922, 1.8),
        (32.0780, 34.7740, 1.6),
        (32.0505, 34.7505, 2.0),
    ]

    def field(clat: float, clon: float, sigma: float) -> np.ndarray:
        x = (lon - clon) * 94.0
        y = (lat - clat) * 111.0
        d2 = x * x + y * y
        return np.exp(-0.5 * d2 / (sigma * sigma))

    spatial = 0.20 + 1.4 * field(*centers[0]) + 1.15 * field(*centers[1]) + field(*centers[2])
    rates = np.empty((3, len(nodes)), dtype=np.float64)
    for i, target in enumerate(hourly):
        scale = [0.95, 1.20, 1.05][i]
        raw = 0.20 + scale * spatial
        rates[i] = raw * (target / raw.sum())

    scenario = ScenarioCreator().create(
        graph_name="tel_aviv_small",
        demand=NodeDemandProfile(nodes, rates, bucket_minutes=60.0),
        seed=42,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenario.save_json(args.output)
    print(f"nodes={len(nodes)} robots={robots} expected_orders={total} realized_orders={len(scenario.orders)}")
    print(args.output)


if __name__ == "__main__":
    main()
