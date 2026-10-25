from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.charging import (
    DEFAULT_CHARGING_POWER_W,
    DEFAULT_NUMBER_OF_PORTS,
)
from delivery_fleet.defaults import MAX_DISTANCE_TO_CHARGING_STATION_M

EDGE_WEIGHT = "length"
PLACEMENT_FRACTION = 0.75
SEED = 42


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_edge_lengths(graph: nx.Graph) -> None:
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            data[EDGE_WEIGHT] = float(data.get(EDGE_WEIGHT, 1.0))
    else:
        for _, _, data in graph.edges(data=True):
            data[EDGE_WEIGHT] = float(data.get(EDGE_WEIGHT, 1.0))


def edge_distance(graph: nx.Graph, u: int, v: int) -> float:
    data = graph.get_edge_data(u, v)
    if graph.is_multigraph():
        return min(float(attrs.get(EDGE_WEIGHT, 1.0)) for attrs in data.values())
    return float(data.get(EDGE_WEIGHT, 1.0))


def node_at_fraction(graph: nx.Graph, path: list[int], excluded: set[int]) -> int:
    cumulative = [0.0]
    for u, v in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + edge_distance(graph, u, v))
    target = PLACEMENT_FRACTION * cumulative[-1]
    candidates = [
        (abs(distance - target), index, node)
        for index, (node, distance) in enumerate(zip(path, cumulative, strict=True))
        if node not in excluded
    ]
    if not candidates:
        raise RuntimeError("no unused node available on charger-placement path")
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def update_nearest_from_new_station(
    graph: nx.Graph,
    new_station: int,
    nearest_distance: dict[int, float],
    nearest_station: dict[int, int],
    node_rank: dict[int, int],
) -> None:
    """Exact nearest-station update, pruning branches that cannot improve.

    If d(new,node) is already strictly larger than node's current distance to a
    charger, then no continuation through node can improve any neighbor because
    that existing charger can reach the neighbor through node at no greater
    cost. Equal-distance states are retained so deterministic tie-breaking can
    propagate exactly as in the full-Dijkstra implementation.
    """

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, new_station)]
    seen: dict[int, float] = {new_station: 0.0}
    new_rank = node_rank[new_station]

    while heap:
        distance, _, node = heapq.heappop(heap)
        if distance != seen.get(node):
            continue

        old_distance = nearest_distance[node]
        if distance > old_distance and not math.isclose(distance, old_distance):
            continue

        if distance < old_distance:
            nearest_distance[node] = distance
            nearest_station[node] = new_station
        elif math.isclose(distance, old_distance):
            if new_rank < node_rank[nearest_station[node]]:
                nearest_station[node] = new_station

        for neighbor in graph.neighbors(node):
            candidate = distance + edge_distance(graph, node, neighbor)
            old_neighbor_distance = nearest_distance[neighbor]
            if candidate > old_neighbor_distance and not math.isclose(
                candidate, old_neighbor_distance
            ):
                continue
            if candidate < seen.get(neighbor, math.inf):
                seen[neighbor] = candidate
                counter += 1
                heapq.heappush(heap, (candidate, counter, neighbor))


def initialize_chargers(graph: nx.Graph) -> tuple[int, ...]:
    if graph.is_directed() or not nx.is_connected(graph):
        raise ValueError("charger placement expects one connected undirected graph")

    nodes = sorted(graph.nodes())
    node_rank = {node: rank for rank, node in enumerate(nodes)}
    rng = np.random.default_rng(SEED)
    first_station = nodes[int(rng.integers(0, len(nodes)))]

    stations = [first_station]
    station_set = {first_station}
    nearest_distance = dict(
        nx.single_source_dijkstra_path_length(
            graph, first_station, weight=EDGE_WEIGHT
        )
    )
    nearest_station = {node: first_station for node in nearest_distance}

    while True:
        farthest = max(
            nodes,
            key=lambda node: (nearest_distance[node], -node_rank[node]),
        )
        max_distance = nearest_distance[farthest]
        if max_distance <= MAX_DISTANCE_TO_CHARGING_STATION_M:
            break

        source_station = nearest_station[farthest]
        path = nx.shortest_path(
            graph,
            source=source_station,
            target=farthest,
            weight=EDGE_WEIGHT,
            method="dijkstra",
        )
        new_station = node_at_fraction(graph, path, station_set)
        stations.append(new_station)
        station_set.add(new_station)
        update_nearest_from_new_station(
            graph,
            new_station,
            nearest_distance,
            nearest_station,
            node_rank,
        )

        if len(stations) % 25 == 0:
            print(
                f"  stations={len(stations):,}, current max distance={max_distance:.1f} m",
                flush=True,
            )

    nx.set_node_attributes(graph, False, "is_charging_station")
    nx.set_node_attributes(
        graph, nearest_distance, "distance_to_nearest_charging_station_m"
    )
    nx.set_node_attributes(
        graph, nearest_station, "nearest_charging_station_node"
    )
    for station_id, node in enumerate(stations):
        graph.nodes[node]["is_charging_station"] = True
        graph.nodes[node]["charging_station_id"] = station_id
        graph.nodes[node]["charging_power_w"] = DEFAULT_CHARGING_POWER_W
        graph.nodes[node]["charging_ports"] = DEFAULT_NUMBER_OF_PORTS

    return tuple(stations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    args = parser.parse_args()

    start = time.perf_counter()
    graph = nx.read_graphml(args.graph, node_type=int)
    normalize_edge_lengths(graph)

    existing = [
        node
        for node, data in graph.nodes(data=True)
        if truthy(data.get("is_charging_station", False))
    ]
    if existing:
        print(f"graph already has {len(existing):,} charging stations")
        nx.write_graphml(graph, args.graph)
        return

    print(
        f"placing charging stations on graph with {graph.number_of_nodes():,} nodes...",
        flush=True,
    )
    stations = initialize_chargers(graph)
    print(
        f"placed {len(stations):,} charging stations in {time.perf_counter()-start:.1f}s; saving graph...",
        flush=True,
    )
    nx.write_graphml(graph, args.graph)
    print(f"saved initialized graph to {args.graph}", flush=True)


if __name__ == "__main__":
    main()
