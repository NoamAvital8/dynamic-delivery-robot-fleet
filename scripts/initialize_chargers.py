from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.charging import add_charging_stations


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    args = parser.parse_args()

    start = time.perf_counter()
    graph = nx.read_graphml(args.graph, node_type=int)
    existing = [
        node
        for node, data in graph.nodes(data=True)
        if truthy(data.get("is_charging_station", False))
    ]
    if existing:
        print(f"graph already has {len(existing):,} charging stations")
        return

    print(
        f"placing charging stations on graph with {graph.number_of_nodes():,} nodes...",
        flush=True,
    )
    graph, stations = add_charging_stations(graph, copy_graph=False)
    print(
        f"placed {len(stations):,} charging stations in {time.perf_counter()-start:.1f}s; saving graph...",
        flush=True,
    )
    nx.write_graphml(graph, args.graph)
    print(f"saved initialized graph to {args.graph}", flush=True)


if __name__ == "__main__":
    main()
