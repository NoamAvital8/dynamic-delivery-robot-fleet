"""Charging-station placement and charger-distance preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Hashable

import networkx as nx
import numpy as np

from .defaults import MAX_DISTANCE_TO_CHARGING_STATION_M

NodeId = Hashable

DEFAULT_CHARGING_POWER_W = 500.0
DEFAULT_NUMBER_OF_PORTS = 2

NEAREST_CHARGING_STATION_NODE_ATTR = "nearest_charging_station_node"
DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR = (
    "distance_to_nearest_charging_station_m"
)


@dataclass(frozen=True, slots=True)
class ChargingStation:
    id: int
    node_id: NodeId
    charging_power_w: float = DEFAULT_CHARGING_POWER_W
    number_of_ports: int = DEFAULT_NUMBER_OF_PORTS


@dataclass(frozen=True, slots=True)
class ChargingConfig:
    max_distance_to_station_m: float = MAX_DISTANCE_TO_CHARGING_STATION_M
    placement_fraction: float = 0.75
    seed: int = 42
    edge_weight: str = "length"

    def __post_init__(self) -> None:
        if self.max_distance_to_station_m <= 0:
            raise ValueError("max_distance_to_station_m must be positive")
        if not (0.0 < self.placement_fraction <= 1.0):
            raise ValueError("placement_fraction must be in (0, 1]")
        if not self.edge_weight:
            raise ValueError("edge_weight cannot be empty")


def _edge_weight_value(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    weight: str,
) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        raise ValueError(f"path contains missing edge {u!r} -- {v!r}")

    if graph.is_multigraph():
        value = min(float(attrs.get(weight, 1.0)) for attrs in data.values())
    else:
        value = float(data.get(weight, 1.0))

    if value < 0:
        raise ValueError("charging placement requires non-negative edge weights")
    return value


def _node_at_path_fraction(
    graph: nx.Graph,
    path: list[NodeId],
    fraction: float,
    weight: str,
    excluded: set[NodeId],
) -> NodeId:
    if len(path) < 2:
        raise ValueError("path must contain at least two nodes")

    cumulative = [0.0]
    for u, v in zip(path, path[1:], strict=True):
        cumulative.append(cumulative[-1] + _edge_weight_value(graph, u, v, weight))

    target = fraction * cumulative[-1]
    candidates = [
        (abs(distance - target), index, node)
        for index, (node, distance) in enumerate(zip(path, cumulative, strict=True))
        if node not in excluded
    ]
    if not candidates:
        raise RuntimeError("no unused node is available on the selected path")
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def select_charging_station_nodes(
    graph: nx.Graph,
    config: ChargingConfig | None = None,
) -> tuple[NodeId, ...]:
    """Place chargers until max_v min_s d(v,s) is within the coverage radius.

    The first charger is uniformly random (seeded); every later choice is
    deterministic. New chargers are placed 75% of the path distance toward
    the currently farthest node by default.
    """

    config = config or ChargingConfig()
    if graph.is_directed():
        raise ValueError("charging placement expects an undirected graph")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph is empty")
    if not nx.is_connected(graph):
        raise ValueError("charging placement expects a connected graph")

    nodes = sorted(graph.nodes())
    node_rank = {node: rank for rank, node in enumerate(nodes)}
    rng = np.random.default_rng(config.seed)
    first_station = nodes[int(rng.integers(0, len(nodes)))]

    stations: list[NodeId] = [first_station]
    station_set: set[NodeId] = {first_station}
    nearest_distance = dict(
        nx.single_source_dijkstra_path_length(
            graph, first_station, weight=config.edge_weight
        )
    )
    nearest_station = {node: first_station for node in nearest_distance}

    while True:
        farthest = max(
            nodes,
            key=lambda node: (nearest_distance[node], -node_rank[node]),
        )
        if nearest_distance[farthest] <= config.max_distance_to_station_m:
            break

        source_station = nearest_station[farthest]
        path = nx.shortest_path(
            graph,
            source=source_station,
            target=farthest,
            weight=config.edge_weight,
            method="dijkstra",
        )
        new_station = _node_at_path_fraction(
            graph,
            path,
            config.placement_fraction,
            config.edge_weight,
            station_set,
        )
        stations.append(new_station)
        station_set.add(new_station)

        distances_from_new = nx.single_source_dijkstra_path_length(
            graph, new_station, weight=config.edge_weight
        )
        for node, new_distance in distances_from_new.items():
            old_distance = nearest_distance[node]
            if new_distance < old_distance:
                nearest_distance[node] = new_distance
                nearest_station[node] = new_station
            elif math.isclose(new_distance, old_distance):
                if node_rank[new_station] < node_rank[nearest_station[node]]:
                    nearest_station[node] = new_station

    return tuple(stations)


def nearest_station_data(
    graph: nx.Graph,
    station_nodes: tuple[NodeId, ...] | list[NodeId],
    *,
    edge_weight: str = "length",
) -> tuple[dict[NodeId, float], dict[NodeId, NodeId]]:
    """One multi-source Dijkstra returning distance and nearest charger/node.

    This deliberately stores only one distance and one source label per graph
    node, rather than full source-to-node paths, so it stays memory-efficient
    on the large NYC graph.
    """

    if not station_nodes:
        raise ValueError("at least one charging station is required")

    stations = tuple(station_nodes)
    for station in stations:
        if station not in graph:
            raise ValueError(f"charging station {station!r} is not in graph")

    station_rank = {
        station: rank
        for rank, station in enumerate(sorted(stations, key=lambda node: str(node)))
    }
    distances: dict[NodeId, float] = {}
    nearest: dict[NodeId, NodeId] = {}
    counter = itertools.count()
    heap: list[tuple[float, int, int, NodeId, NodeId]] = []

    for station in stations:
        distances[station] = 0.0
        nearest[station] = station
        heapq.heappush(
            heap,
            (0.0, station_rank[station], next(counter), station, station),
        )

    while heap:
        distance, source_rank, _, node, source = heapq.heappop(heap)
        if distance != distances.get(node) or source != nearest.get(node):
            continue

        for neighbor in graph.neighbors(node):
            candidate = distance + _edge_weight_value(
                graph, node, neighbor, edge_weight
            )
            old_distance = distances.get(neighbor, math.inf)
            old_source = nearest.get(neighbor)
            better_tie = (
                math.isclose(candidate, old_distance)
                and old_source is not None
                and source_rank < station_rank[old_source]
            )
            if candidate < old_distance or better_tie:
                distances[neighbor] = candidate
                nearest[neighbor] = source
                heapq.heappush(
                    heap,
                    (
                        candidate,
                        source_rank,
                        next(counter),
                        neighbor,
                        source,
                    ),
                )

    if len(distances) != graph.number_of_nodes():
        raise ValueError("every graph node must be reachable from a charging station")
    return distances, nearest


def annotate_nearest_charging_stations(
    graph: nx.Graph,
    station_nodes: tuple[NodeId, ...] | list[NodeId],
    *,
    edge_weight: str = "length",
) -> None:
    """Persist nearest-charger lookup data directly on every graph node."""

    distances, nearest = nearest_station_data(
        graph, station_nodes, edge_weight=edge_weight
    )
    nx.set_node_attributes(
        graph,
        distances,
        DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR,
    )
    nx.set_node_attributes(
        graph,
        nearest,
        NEAREST_CHARGING_STATION_NODE_ATTR,
    )


def max_distance_to_nearest_station(
    graph: nx.Graph,
    station_nodes: tuple[NodeId, ...] | list[NodeId],
    *,
    edge_weight: str = "length",
) -> float:
    distances, _ = nearest_station_data(
        graph, station_nodes, edge_weight=edge_weight
    )
    return float(max(distances.values(), default=0.0))


def add_charging_stations(
    graph: nx.Graph,
    config: ChargingConfig | None = None,
    *,
    copy_graph: bool = True,
) -> tuple[nx.Graph, tuple[ChargingStation, ...]]:
    """Place chargers and initialize charger-related attributes on every node."""

    config = config or ChargingConfig()
    result = graph.copy() if copy_graph else graph
    station_nodes = select_charging_station_nodes(result, config)

    nx.set_node_attributes(result, False, "is_charging_station")
    stations: list[ChargingStation] = []
    for station_id, node in enumerate(station_nodes):
        station = ChargingStation(id=station_id, node_id=node)
        stations.append(station)
        result.nodes[node]["is_charging_station"] = True
        result.nodes[node]["charging_station_id"] = station.id
        result.nodes[node]["charging_power_w"] = station.charging_power_w
        result.nodes[node]["charging_ports"] = station.number_of_ports

    # One initialization pass gives future policies O(1) access to both the
    # closest charger and its distance from any graph node.
    annotate_nearest_charging_stations(
        result, station_nodes, edge_weight=config.edge_weight
    )
    return result, tuple(stations)
