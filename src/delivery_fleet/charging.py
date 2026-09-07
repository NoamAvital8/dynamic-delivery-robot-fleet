"""Charging-station placement for delivery-robot city graphs.

The first station is chosen uniformly at random from the graph nodes. After
that, placement is deterministic given that first station:

1. Find the node whose graph distance to its nearest existing charger is
   maximal.
2. If that distance is at most the configured coverage radius (2 km by
   default), stop.
3. Otherwise find the nearest existing charger to that farthest node.
4. Compute the shortest path from that charger to the farthest node.
5. Place the new charger at ``placement_fraction`` of the path distance
   (75% by default), rather than at the farthest endpoint itself.

Thus the number of stations is not hard-coded. Enough stations are added to
ensure every graph node is within the required graph distance of a charger.

For now, all charging stations are identical: they use one global charging
power and one fixed number of ports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import networkx as nx
import numpy as np

from .defaults import MAX_DISTANCE_TO_CHARGING_STATION_M


NodeId = Hashable

# V1 station capabilities. These are intentionally global constants so all
# charging stations are identical for now.
DEFAULT_CHARGING_POWER_W = 500.0
DEFAULT_NUMBER_OF_PORTS = 2


@dataclass(frozen=True, slots=True)
class ChargingStation:
    """A charging station attached to one graph node."""

    id: int
    node_id: NodeId
    charging_power_w: float = DEFAULT_CHARGING_POWER_W
    number_of_ports: int = DEFAULT_NUMBER_OF_PORTS


@dataclass(frozen=True, slots=True)
class ChargingConfig:
    """Configuration for charging-station placement only.

    Stations are added until every graph node is within
    ``max_distance_to_station_m`` of at least one station. Charging power and
    port count are intentionally not configurable in V1: every station uses
    ``DEFAULT_CHARGING_POWER_W`` and ``DEFAULT_NUMBER_OF_PORTS``.
    """

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
    """Select charger nodes until the requested coverage radius is satisfied.

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

    while True:
        # Deterministic tie-breaking by node order after the random first choice.
        farthest = max(
            nodes,
            key=lambda node: (nearest_distance[node], -node_rank[node]),
        )
        farthest_distance = nearest_distance[farthest]

        if farthest_distance <= config.max_distance_to_station_m:
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


def max_distance_to_nearest_station(
    graph: nx.Graph,
    station_nodes: tuple[NodeId, ...] | list[NodeId],
    *,
    edge_weight: str = "length",
) -> float:
    """Return max_v min_s d(v, s) for a station set."""

    if not station_nodes:
        raise ValueError("at least one charging station is required")

    distances = nx.multi_source_dijkstra_path_length(
        graph,
        station_nodes,
        weight=edge_weight,
    )
    if len(distances) != graph.number_of_nodes():
        raise ValueError("every graph node must be reachable from a charging station")
    return float(max(distances.values(), default=0.0))


def add_charging_stations(
    graph: nx.Graph,
    config: ChargingConfig | None = None,
    *,
    copy_graph: bool = True,
) -> tuple[nx.Graph, tuple[ChargingStation, ...]]:
    """Select identical stations and annotate their graph nodes.

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
        station = ChargingStation(id=station_id, node_id=node)
        stations.append(station)

        result.nodes[node]["is_charging_station"] = True
        result.nodes[node]["charging_station_id"] = station.id
        result.nodes[node]["charging_power_w"] = station.charging_power_w
        result.nodes[node]["charging_ports"] = station.number_of_ports

    return result, tuple(stations)
