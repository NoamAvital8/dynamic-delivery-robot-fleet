"""Optimal battery-feasible routing through charging stations.

For the V1 model all charging stations have the same charging power, charging
is linear and partial charging is allowed, and there is no charger queueing
cost in route planning. Under those assumptions, for a route of total distance
D the minimum charging time is a monotone function of D. Therefore the
minimum-time feasible route is exactly the minimum-distance feasible route.

The router searches a meta-graph of charging stops while enforcing the order
robot_location -> pickup -> dropoff and a final battery reserve sufficient to
reach the nearest charger from the dropoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Hashable, Iterable

import networkx as nx

from .charging import (
    DEFAULT_CHARGING_POWER_W,
    DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR,
)
from .robot import RobotState

NodeId = Hashable
_EPS = 1e-8


class NoFeasibleBatteryRoute(RuntimeError):
    """Raised when the selected robot cannot safely serve the requested route."""


@dataclass(frozen=True, slots=True)
class ChargeEvent:
    node_id: NodeId
    energy_added_wh: float
    duration_min: float
    battery_before_wh: float
    battery_after_wh: float


@dataclass(frozen=True, slots=True)
class RouteSegment:
    waypoints: tuple[NodeId, ...]
    node_path: tuple[NodeId, ...]
    distance_m: float
    ends_at_charger: bool


@dataclass(frozen=True, slots=True)
class BatteryFeasibleRoute:
    pickup_node: NodeId
    dropoff_node: NodeId
    segments: tuple[RouteSegment, ...]
    charging_events: tuple[ChargeEvent, ...]
    total_distance_m: float
    travel_time_min: float
    charging_time_min: float
    total_time_min: float
    arrival_battery_wh: float
    required_dropoff_reserve_wh: float

    @property
    def node_path(self) -> tuple[NodeId, ...]:
        if not self.segments:
            return ()
        merged: list[NodeId] = list(self.segments[0].node_path)
        for segment in self.segments[1:]:
            merged.extend(segment.node_path[1:])
        return tuple(merged)


@dataclass(frozen=True, slots=True)
class _MetaEdge:
    distance_m: float
    waypoints: tuple[NodeId, ...]


class BatteryFeasibleRouter:
    """Find the fastest safe route for one already-selected robot."""

    def __init__(
        self,
        graph: nx.Graph,
        station_nodes: Iterable[NodeId] | None = None,
        *,
        edge_weight: str = "length",
    ) -> None:
        if graph.is_directed():
            raise ValueError("battery router currently expects an undirected graph")
        if not nx.is_connected(graph):
            raise ValueError("battery router expects a connected graph")

        self.graph = graph
        self.edge_weight = edge_weight
        if station_nodes is None:
            station_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if bool(data.get("is_charging_station"))
            ]
        self.station_nodes = tuple(sorted(set(station_nodes), key=lambda x: str(x)))
        self.station_set = set(self.station_nodes)
        if not self.station_nodes:
            raise ValueError("graph has no charging stations")

        missing = [
            node
            for node in graph.nodes
            if DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR not in graph.nodes[node]
        ]
        if missing:
            raise ValueError(
                "graph must be initialized with distance_to_nearest_charging_station_m "
                "for every node"
            )

        self._station_neighbor_cache: dict[
            tuple[NodeId, float], dict[NodeId, float]
        ] = {}

    def _distances_within(self, source: NodeId, cutoff_m: float) -> dict[NodeId, float]:
        return dict(
            nx.single_source_dijkstra_path_length(
                self.graph,
                source,
                cutoff=max(0.0, cutoff_m),
                weight=self.edge_weight,
            )
        )

    def _station_neighbors(
        self, source: NodeId, cutoff_m: float
    ) -> dict[NodeId, float]:
        key = (source, round(float(cutoff_m), 6))
        if source in self.station_set and key in self._station_neighbor_cache:
            return self._station_neighbor_cache[key]

        distances = self._distances_within(source, cutoff_m)
        result = {
            station: float(distances[station])
            for station in self.station_nodes
            if station in distances and station != source
        }
        if source in self.station_set:
            self._station_neighbor_cache[key] = result
        return result

    def _path_length(self, path: list[NodeId]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            data = self.graph.get_edge_data(a, b)
            if self.graph.is_multigraph():
                total += min(
                    float(attrs.get(self.edge_weight, 1.0))
                    for attrs in data.values()
                )
            else:
                total += float(data.get(self.edge_weight, 1.0))
        return total

    def _materialize_segment(
        self,
        edge: _MetaEdge,
        *,
        ends_at_charger: bool,
    ) -> RouteSegment:
        waypoints = edge.waypoints
        if not waypoints:
            raise RuntimeError("internal route edge has no waypoints")

        full_path: list[NodeId] = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            if a == b:
                continue
            leg = nx.shortest_path(
                self.graph,
                source=a,
                target=b,
                weight=self.edge_weight,
                method="dijkstra",
            )
            full_path.extend(leg[1:])

        actual_distance = self._path_length(full_path)
        if not math.isclose(
            actual_distance, edge.distance_m, rel_tol=1e-8, abs_tol=1e-6
        ):
            raise RuntimeError("materialized route length disagrees with meta route")

        return RouteSegment(
            waypoints=waypoints,
            node_path=tuple(full_path),
            distance_m=float(actual_distance),
            ends_at_charger=ends_at_charger,
        )

    def plan(
        self,
        robot: RobotState,
        pickup_node: NodeId,
        dropoff_node: NodeId,
    ) -> BatteryFeasibleRoute:
        if pickup_node not in self.graph or dropoff_node not in self.graph:
            raise ValueError("pickup and dropoff must be graph nodes")
        if pickup_node == dropoff_node:
            raise ValueError("pickup and dropoff must be different")

        start = robot.node_id
        spec = robot.spec
        full_range = spec.full_battery_range_m
        current_range = robot.remaining_range_m
        reserve_distance = float(
            self.graph.nodes[dropoff_node][
                DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR
            ]
        )
        reserve_wh = reserve_distance * spec.energy_per_meter_wh

        start_is_station = start in self.station_set
        start_nearest_distance = float(
            self.graph.nodes[start][DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR]
        )
        if not start_is_station and current_range + _EPS < start_nearest_distance:
            raise NoFeasibleBatteryRoute(
                "robot is already below the safety reserve needed to reach a charger"
            )

        u_dist = self._distances_within(pickup_node, full_range)
        v_dist = self._distances_within(dropoff_node, full_range)

        START = ("start",)
        DONE = ("done",)

        def physical_node(state: tuple) -> NodeId:
            return start if state == START else state[1]

        def departure_range(state: tuple) -> float:
            if state == START:
                return full_range if start_is_station else current_range
            return full_range

        def neighbors(state: tuple):
            if state == DONE:
                return

            phase = "pre" if state == START else state[0]
            source = physical_node(state)
            available_range = departure_range(state)

            if phase == "pre":
                for station, distance in self._station_neighbors(
                    source, available_range
                ).items():
                    yield (
                        ("pre", station),
                        _MetaEdge(distance, (source, station)),
                    )

                d_to_pickup = u_dist.get(source)
                if d_to_pickup is None or d_to_pickup > available_range + _EPS:
                    return
                d_to_pickup = float(d_to_pickup)

                if pickup_node in self.station_set:
                    yield (
                        ("post", pickup_node),
                        _MetaEdge(d_to_pickup, (source, pickup_node)),
                    )
                    return

                for station in self.station_nodes:
                    d_from_pickup = u_dist.get(station)
                    if d_from_pickup is None:
                        continue
                    segment_distance = d_to_pickup + float(d_from_pickup)
                    if segment_distance <= available_range + _EPS:
                        yield (
                            ("post", station),
                            _MetaEdge(
                                segment_distance,
                                (source, pickup_node, station),
                            ),
                        )

                d_pickup_to_dropoff = u_dist.get(dropoff_node)
                if d_pickup_to_dropoff is not None:
                    segment_distance = d_to_pickup + float(d_pickup_to_dropoff)
                    if (
                        segment_distance + reserve_distance
                        <= available_range + _EPS
                    ):
                        yield (
                            DONE,
                            _MetaEdge(
                                segment_distance,
                                (source, pickup_node, dropoff_node),
                            ),
                        )
                return

            for station, distance in self._station_neighbors(
                source, full_range
            ).items():
                yield (
                    ("post", station),
                    _MetaEdge(distance, (source, station)),
                )

            d_to_dropoff = v_dist.get(source)
            if d_to_dropoff is not None:
                d_to_dropoff = float(d_to_dropoff)
                if d_to_dropoff + reserve_distance <= full_range + _EPS:
                    yield (
                        DONE,
                        _MetaEdge(d_to_dropoff, (source, dropoff_node)),
                    )

        best: dict[tuple, float] = {START: 0.0}
        previous: dict[tuple, tuple[tuple, _MetaEdge]] = {}
        counter = itertools.count()
        heap: list[tuple[float, int, tuple]] = [(0.0, next(counter), START)]

        while heap:
            distance_so_far, _, state = heapq.heappop(heap)
            if distance_so_far != best.get(state):
                continue
            if state == DONE:
                break

            for next_state, edge in neighbors(state):
                candidate = distance_so_far + edge.distance_m
                if candidate + _EPS < best.get(next_state, math.inf):
                    best[next_state] = candidate
                    previous[next_state] = (state, edge)
                    heapq.heappush(
                        heap,
                        (candidate, next(counter), next_state),
                    )

        if DONE not in best:
            raise NoFeasibleBatteryRoute(
                "no battery-feasible route exists for the selected robot"
            )

        meta_edges: list[tuple[_MetaEdge, bool]] = []
        state = DONE
        while state != START:
            prev_state, edge = previous[state]
            meta_edges.append((edge, state != DONE))
            state = prev_state
        meta_edges.reverse()

        segments = tuple(
            self._materialize_segment(edge, ends_at_charger=ends_at_charger)
            for edge, ends_at_charger in meta_edges
        )

        battery = robot.battery_wh
        charge_events: list[ChargeEvent] = []
        total_charging_time = 0.0

        for index, segment in enumerate(segments):
            is_final = index == len(segments) - 1
            required_after_segment = reserve_wh if is_final else 0.0
            required_departure = (
                segment.distance_m * spec.energy_per_meter_wh
                + required_after_segment
            )
            if required_departure > spec.battery_capacity_wh + 1e-6:
                raise RuntimeError("meta route contains an infeasible battery segment")

            if battery + _EPS < required_departure:
                start_node = segment.node_path[0]
                if start_node not in self.station_set:
                    raise RuntimeError("route requires charging at a non-station node")

                energy_added = required_departure - battery
                before = battery
                battery += energy_added
                duration_min = energy_added / DEFAULT_CHARGING_POWER_W * 60.0
                total_charging_time += duration_min
                charge_events.append(
                    ChargeEvent(
                        node_id=start_node,
                        energy_added_wh=float(energy_added),
                        duration_min=float(duration_min),
                        battery_before_wh=float(before),
                        battery_after_wh=float(battery),
                    )
                )

            battery -= segment.distance_m * spec.energy_per_meter_wh
            if battery < -1e-6:
                raise RuntimeError("battery became negative on a feasible route")
            battery = max(0.0, battery)

        if battery + 1e-6 < reserve_wh:
            raise RuntimeError("route violates required dropoff charger reserve")

        total_distance = sum(segment.distance_m for segment in segments)
        travel_time = total_distance / spec.speed_mps / 60.0
        total_time = travel_time + total_charging_time

        return BatteryFeasibleRoute(
            pickup_node=pickup_node,
            dropoff_node=dropoff_node,
            segments=segments,
            charging_events=tuple(charge_events),
            total_distance_m=float(total_distance),
            travel_time_min=float(travel_time),
            charging_time_min=float(total_charging_time),
            total_time_min=float(total_time),
            arrival_battery_wh=float(battery),
            required_dropoff_reserve_wh=float(reserve_wh),
        )
