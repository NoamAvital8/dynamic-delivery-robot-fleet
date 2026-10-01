from __future__ import annotations

"""Measure idle-decision overhead on a reproducible synthetic line graph.

The O(1) line-distance index isolates the planner's candidate, parallel-vector,
calendar, and coordinated-selection costs.  This is a scaling check, not an
end-to-end NYC simulation benchmark.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.idle_planning import (
    ChargingCalendar, IdleFleetPlanner, IdlePlanningConfig, IdleRobotSnapshot,
)
from delivery_fleet.spatial_demand import GammaPoissonDemandModel


class LineDistanceIndex:
    def __init__(self, stations: tuple[int, ...], edge_m: float) -> None:
        self.stations = stations
        self.edge_m = edge_m
        self.oracle = self

    def distance(self, a: int, b: int) -> float:
        return abs(int(a) - int(b)) * self.edge_m

    def distances_to_stations(self, node: int) -> dict[int, float]:
        return {station: self.distance(node, station) for station in self.stations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=300)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if min(args.robots, args.clusters, args.processes, args.iterations) <= 0:
        parser.error("all counts must be positive")

    nodes_per_cluster = 10
    graph = nx.path_graph(args.clusters * nodes_per_cluster)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.002,
            y=32.0,
            in_cluster=node // nodes_per_cluster,
            is_cluster_representative=(node % nodes_per_cluster == 5),
        )
    nx.set_edge_attributes(graph, 200.0, "length")
    stations = tuple(range(0, len(graph), 40))
    index = LineDistanceIndex(stations, 200.0)
    demand = GammaPoissonDemandModel(
        graph, {1.0: 120.0, 2.0: 22.5, 5.0: 7.5}
    )
    rng = np.random.default_rng(42)
    robots = [
        IdleRobotSnapshot(
            robot_id=i,
            node_id=int(rng.integers(0, len(graph))),
            speed_mps=4.0,
            energy_per_meter_wh=0.075,
            battery_wh=float(rng.uniform(300.0, 1_200.0)),
            battery_capacity_wh=1_200.0,
            max_payload_kg=12.0,
            max_volume_l=30.0,
        )
        for i in range(args.robots)
    ]
    elapsed = []
    action_counts = []
    with IdleFleetPlanner(
        graph, index, demand, stations, 2_000.0,
        IdlePlanningConfig(
            max_demand_clusters=args.clusters,
            processes=args.processes,
        ),
    ) as planner:
        for _ in range(args.iterations):
            calendar = ChargingCalendar({station: 2 for station in stations})
            start = time.perf_counter()
            chosen = planner.plan(0.0, robots, range(args.robots), calendar)
            elapsed.append(time.perf_counter() - start)
            action_counts.append(len(chosen))
    print(json.dumps({
        "benchmark": "synthetic_idle_planner_only",
        "robots": args.robots,
        "clusters": args.clusters,
        "processes": args.processes,
        "iterations": args.iterations,
        "seconds_each": elapsed,
        "selected_actions_each": action_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
