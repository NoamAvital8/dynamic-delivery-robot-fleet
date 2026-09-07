"""Charging-station placement for delivery-robot city graphs.

The first station is chosen uniformly at random from the graph nodes. After
that, placement is deterministic given that first station:

1. Find the node whose graph distance to its nearest existing charger is
   maximal.
2. Find the nearest existing charger to that node.
3. Compute the shortest path from that charger to the farthest node.
4. Place the new charger at ``placement_fraction`` of the path distance
   (75% by default), rather than at the farthest endpoint itself.

Moving the station inward avoids repeatedly placing chargers on peripheral
endpoints while still pushing coverage toward poorly served parts of the map.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import networkx as nx
import numpy as np


NodeId = Hashable


@dataclass(frozen=True, slots=True)
class ChargingStation:
    """A charging station attached to one graph node."""

    id: int
    node_id: NodeId
    charging_power_w: float
    number_of_ports: int


@dataclass(frozen=True, slots=True)
class ChargingConfig:
    """Configuration for charging-station placement.

    By default, roughly 0.1% of graph nodes become charging stations. Set
    ``station_count`` to override the fraction with an exact count.
    """

    station_fraction: float = 0.001
    station_count: int | None = None
    placement_fraction: float = 0.75
    charging_power_w: float = 500.0
    number_of_ports: int = 2
    seed: int = 42
    edge_weight: str = "length"

    def __post_init__(self) -> None:
        if self.station_fraction <= 0:
            raise ValueError("station_fraction must be positive")
        if self.station_count is not None and self.station_count <= 0:
            raise ValueError("station_count must be positive when provided")
        if not (0.0 < self.placement_fraction <= 1.0):
            raise ValueError("placement_fraction must be in (0, 1]")
        if self.charging_power_w <= 0:
            raise ValueError("charging_power_w must be positive")
        if self.number_of_ports <= 0:
            raise ValueError("number_of_ports must be positive")
        if not self.edge_weight:
            raise ValueError("edge_weight cannot be empty")

    def resolved_station_count(self, number_of_nodes: int) -> int:
        if number_of_nodes <= 0:
            raise ValueError("graph must contain at least one node")

        count = (
            self.station_count
            if self.station_count is not None
            else max(1, round(number_of_nodes * self.station_fraction))
        )
        if count > number_of_nodes:
            raise ValueError("station count cannot exceed number of graph nodes")
        return count


def _edge_weight_value(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    weight: str,
) -> float:
    """Return the same effective edge weight used by NetworkX shortest paths."""

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
    """Choose the non-excluded path node closest to a fraction of path length."""

    if len(path) < 2:
        raise ValueError("path must contain at least two nodes")

    cumulative = [0.0]
    for u, v in zip(path, path[1:], strict=True):
        cumulative.append(cumulative[-1] + _edge_weight_value(graph, u, v, weight))

    total = cumulative[-1]
    target = fraction * total

    candidates = [
        (abs(distance - target), index, node)
        for index, (node, distance) in enumerate(zip(path, cumulative, strict=True))
        if node not in excluded
    ]
    if not candidates:
        raise RuntimeError("no unused node is available on the selected path")

    # index is a deterministic tie-breaker: prefer the node encountered first.
    _, _, node = min(candidates, key=lambda row: (row[0], row[1]))
    return node


def select_charging_station_nodes(
    graph: nx.Graph,
    config: ChargingConfig | None = None,
) -> tuple[NodeId, ...]:
    """Select charging-station nodes.

    The graph must be undirected and connected. The only random choice is the
    first station. Every subsequent station is completely determined by the
    graph, the first station, and the configuration.
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
    target_count = config.resolved_station_count(len(nodes))

    rng = np.random.default_rng(config.seed)
    first_station = nodes[int(rng.integers(0, len(nodes)))]

    stations: list[NodeId] = [first_station]
    station_set: set[NodeId] = {first_station}

    nearest_distance = dict(
        nx.single_source_dijkstra_path_length(
            graph,
            first_station,
            weight=config.edge_weight,
        )
    )
    nearest_station: dict[NodeId, NodeId] = {
        node: first_station for node in nearest_distance
    }

    while len(stations) < target_count:
        # Deterministic tie-breaking by node order after the random first choice.
        farthest = max(
            nodes,
            key=lambda node: (nearest_distance[node], -node_rank[node]),
        )
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

        # Only one new full Dijkstra is required. Distances to the nearest
        # charger are updated by comparing against distances from this station.
        distances_from_new = nx.single_source_dijkstra_path_length(
            graph,
            new_station,
            weight=config.edge_weight,
        )

        for node, new_distance in distances_from_new.items():
            old_distance = nearest_distance[node]
            if new_distance < old_distance:
                nearest_distance[node] = new_distance
                nearest_station[node] = new_station
            elif math.isclose(new_distance, old_distance):
                # Stable deterministic tie-break between equidistant stations.
                if node_rank[new_station] < node_rank[nearest_station[node]]:
                    nearest_station[node] = new_station

    return tuple(stations)


def add_charging_stations(
    graph: nx.Graph,
    config: ChargingConfig | None = None,
    *,
    copy_graph: bool = True,
) -> tuple[nx.Graph, tuple[ChargingStation, ...]]:
    """Select stations and annotate their graph nodes.

    Returns ``(graph_with_chargers, stations)``. By default the input graph is
    copied so adding chargers does not modify the original OSM graph.
    """

    config = config or ChargingConfig()
    result = graph.copy() if copy_graph else graph
    station_nodes = select_charging_station_nodes(result, config)

    # Explicit false values make downstream checks simple and unambiguous.
    nx.set_node_attributes(result, False, "is_charging_station")

    stations: list[ChargingStation] = []
    for station_id, node in enumerate(station_nodes):
        station = ChargingStation(
            id=station_id,
            node_id=node,
            charging_power_w=config.charging_power_w,
            number_of_ports=config.number_of_ports,
        )
        stations.append(station)

        result.nodes[node]["is_charging_station"] = True
        result.nodes[node]["charging_station_id"] = station.id
        result.nodes[node]["charging_power_w"] = station.charging_power_w
        result.nodes[node]["charging_ports"] = station.number_of_ports

    return result, tuple(stations)
