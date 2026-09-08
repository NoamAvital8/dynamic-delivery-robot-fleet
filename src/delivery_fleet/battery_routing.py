"""Optimal battery-feasible routing through charging stations.

The expensive street-graph work is delegated to the shared routing layer.  A
policy can first call :meth:`BatteryFeasibleRouter.evaluate` for many candidate
assignments and only materialize the full node-by-node path for the route it
actually commits.

For the V1 model all charging stations have the same charging power, charging
is linear and partial charging is allowed, and there is no charger queueing
cost in route planning. Under those assumptions minimum feasible total time is
obtained by the minimum-distance battery-feasible route.
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
from .routing import ChargerDistanceIndex, DistanceOracle

NodeId = Hashable
_EPS = 1e-3  # dense charger index stores city-scale distances as float32


class NoFeasibleBatteryRoute(RuntimeError):
    """Raised when a robot cannot safely serve the requested route."""


@dataclass(frozen=True, slots=True)
class ChargeEvent:
    node_id: NodeId
    energy_added_wh: float
    duration_min: float
    battery_before_wh: float
    battery_after_wh: float


@dataclass(frozen=True, slots=True)
class RouteQuoteSegment:
    """A route segment described only by semantic waypoints and exact distance."""

    waypoints: tuple[NodeId, ...]
    distance_m: float
    ends_at_charger: bool


@dataclass(frozen=True, slots=True)
class BatteryRouteQuote:
    """Cheap exact route evaluation before street-node path materialization."""

    pickup_node: NodeId
    dropoff_node: NodeId
    segments: tuple[RouteQuoteSegment, ...]
    charging_events: tuple[ChargeEvent, ...]
    total_distance_m: float
    travel_time_min: float
    charging_time_min: float
    total_time_min: float
    arrival_battery_wh: float
    required_dropoff_reserve_wh: float


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


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


class BatteryFeasibleRouter:
    """Evaluate and materialize safe routes using shared immutable indices."""

    def __init__(
        self,
        graph: nx.Graph,
        station_nodes: Iterable[NodeId] | None = None,
        *,
        edge_weight: str = "length",
        distance_oracle: DistanceOracle | None = None,
        charger_index: ChargerDistanceIndex | None = None,
        build_dense_charger_index: bool = True,
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
                if _truthy(data.get("is_charging_station", False))
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

        self.distance_oracle = distance_oracle or DistanceOracle(
            graph,
            edge_weight=edge_weight,
        )
        if charger_index is None:
            charger_index = ChargerDistanceIndex(
                graph,
                self.station_nodes,
                oracle=self.distance_oracle,
                edge_weight=edge_weight,
                build_dense=build_dense_charger_index,
            )
        if set(charger_index.station_nodes) != self.station_set:
            raise ValueError("charger index stations do not match router stations")
        self.charger_index = charger_index

    def _path_length(self, path: tuple[NodeId, ...] | list[NodeId]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            data = self.graph.get_edge_data(a, b)
            if data is None:
                raise RuntimeError(f"materialized path contains missing edge {a!r}->{b!r}")
            if self.graph.is_multigraph():
                total += min(
                    float(attrs.get(self.edge_weight, 1.0))
                    for attrs in data.values()
                )
            else:
                total += float(data.get(self.edge_weight, 1.0))
        return total

    def _materialize_segment(self, segment: RouteQuoteSegment) -> RouteSegment:
        if not segment.waypoints:
            raise RuntimeError("route segment has no waypoints")

        full_path: list[NodeId] = [segment.waypoints[0]]
        for a, b in zip(segment.waypoints, segment.waypoints[1:]):
            if a == b:
                continue
            leg = self.charger_index.path(a, b)
            full_path.extend(leg[1:])

        actual_distance = self._path_length(full_path)
        # The dense station index stores float32 distances, so allow a few cm
        # of representation error when validating the reconstructed exact path.
        if not math.isclose(
            actual_distance,
            segment.distance_m,
            rel_tol=2e-6,
            abs_tol=0.05,
        ):
            raise RuntimeError(
                "materialized route length disagrees with route quote: "
                f"{actual_distance} != {segment.distance_m}"
            )

        return RouteSegment(
            waypoints=segment.waypoints,
            node_path=tuple(full_path),
            distance_m=float(actual_distance),
            ends_at_charger=segment.ends_at_charger,
        )

    def evaluate(
        self,
        robot: RobotState,
        pickup_node: NodeId,
        dropoff_node: NodeId,
        *,
        start_to_pickup_m: float | None = None,
        pickup_to_dropoff_m: float | None = None,
    ) -> BatteryRouteQuote:
        """Return an exact cost/feasibility quote without building street paths.

        Optional known pair distances let a policy reuse work it already did
        while ranking candidates (for example the outward nearest-robot search).
        """

        if pickup_node not in self.graph or dropoff_node not in self.graph:
            raise ValueError("pickup and dropoff must be graph nodes")
        if pickup_node == dropoff_node:
            raise ValueError("pickup and dropoff must be different")

        start = robot.node_id
        spec = robot.spec
        full_range = float(spec.full_battery_range_m)
        current_range = float(robot.remaining_range_m)
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

        # These are O(number_of_stations) lookups with the dense index.  The
        # old implementation ran two large cutoff Dijkstras per candidate.
        pickup_station_dist = self.charger_index.distances_to_stations(
            pickup_node,
            cutoff_m=full_range,
        )
        dropoff_station_dist = self.charger_index.distances_to_stations(
            dropoff_node,
            cutoff_m=full_range,
        )

        if pickup_to_dropoff_m is None:
            pickup_to_dropoff_m = self.distance_oracle.distance(
                pickup_node,
                dropoff_node,
            )
        else:
            pickup_to_dropoff_m = float(pickup_to_dropoff_m)
            self.distance_oracle.remember_distance(
                pickup_node,
                dropoff_node,
                pickup_to_dropoff_m,
            )

        if start_to_pickup_m is None:
            if start in self.station_set:
                start_to_pickup_m = pickup_station_dist.get(start)
                if start_to_pickup_m is None:
                    # It can be outside full range; exact pair distance is still
                    # useful to decide that direct pickup is impossible.
                    start_to_pickup_m = self.charger_index.distance(start, pickup_node)
            else:
                start_to_pickup_m = self.distance_oracle.distance(start, pickup_node)
        else:
            start_to_pickup_m = float(start_to_pickup_m)
            self.distance_oracle.remember_distance(start, pickup_node, start_to_pickup_m)

        START = ("start",)
        DONE = ("done",)

        def physical_node(state: tuple) -> NodeId:
            return start if state == START else state[1]

        def departure_range(state: tuple) -> float:
            if state == START:
                return full_range if start_is_station else current_range
            return full_range

        def station_neighbors(source: NodeId, cutoff_m: float) -> dict[NodeId, float]:
            if source in self.station_set:
                return self.charger_index.station_neighbors(source, cutoff_m)
            return self.charger_index.distances_to_stations(
                source,
                cutoff_m=cutoff_m,
                include_self=False,
            )

        def distance_station_to_pickup(source: NodeId) -> float | None:
            if source == start:
                return float(start_to_pickup_m)
            value = pickup_station_dist.get(source)
            return None if value is None else float(value)

        def neighbors(state: tuple):
            if state == DONE:
                return

            phase = "pre" if state == START else state[0]
            source = physical_node(state)
            available_range = departure_range(state)

            if phase == "pre":
                for station, distance in station_neighbors(
                    source,
                    available_range,
                ).items():
                    yield (
                        ("pre", station),
                        _MetaEdge(float(distance), (source, station)),
                    )

                d_to_pickup = distance_station_to_pickup(source)
                if d_to_pickup is None or d_to_pickup > available_range + _EPS:
                    return

                if pickup_node in self.station_set:
                    yield (
                        ("post", pickup_node),
                        _MetaEdge(d_to_pickup, (source, pickup_node)),
                    )
                    return

                for station, d_from_pickup in pickup_station_dist.items():
                    segment_distance = d_to_pickup + float(d_from_pickup)
                    if segment_distance <= available_range + _EPS:
                        yield (
                            ("post", station),
                            _MetaEdge(
                                segment_distance,
                                (source, pickup_node, station),
                            ),
                        )

                direct_segment = d_to_pickup + pickup_to_dropoff_m
                if direct_segment + reserve_distance <= available_range + _EPS:
                    yield (
                        DONE,
                        _MetaEdge(
                            direct_segment,
                            (source, pickup_node, dropoff_node),
                        ),
                    )
                return

            for station, distance in station_neighbors(source, full_range).items():
                yield (
                    ("post", station),
                    _MetaEdge(float(distance), (source, station)),
                )

            d_to_dropoff = dropoff_station_dist.get(source)
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
            RouteQuoteSegment(
                waypoints=edge.waypoints,
                distance_m=float(edge.distance_m),
                ends_at_charger=ends_at_charger,
            )
            for edge, ends_at_charger in meta_edges
        )

        battery = float(robot.battery_wh)
        charge_events: list[ChargeEvent] = []
        total_charging_time = 0.0

        for index, segment in enumerate(segments):
            is_final = index == len(segments) - 1
            required_after_segment = reserve_wh if is_final else 0.0
            required_departure = (
                segment.distance_m * spec.energy_per_meter_wh
                + required_after_segment
            )
            if required_departure > spec.battery_capacity_wh + 1e-4:
                raise RuntimeError("meta route contains an infeasible battery segment")

            if battery + _EPS < required_departure:
                start_node = segment.waypoints[0]
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
            if battery < -1e-4:
                raise RuntimeError("battery became negative on a feasible route")
            battery = max(0.0, battery)

        if battery + 1e-4 < reserve_wh:
            raise RuntimeError("route violates required dropoff charger reserve")

        total_distance = sum(segment.distance_m for segment in segments)
        travel_time = total_distance / spec.speed_mps / 60.0
        total_time = travel_time + total_charging_time

        return BatteryRouteQuote(
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

    def materialize(
        self,
        quote: BatteryRouteQuote,
        *,
        speed_mps: float,
    ) -> BatteryFeasibleRoute:
        """Build node-by-node street paths only for a route that will be used."""

        segments = tuple(self._materialize_segment(segment) for segment in quote.segments)
        actual_total_distance = sum(segment.distance_m for segment in segments)
        actual_travel_time = actual_total_distance / float(speed_mps) / 60.0
        return BatteryFeasibleRoute(
            pickup_node=quote.pickup_node,
            dropoff_node=quote.dropoff_node,
            segments=segments,
            charging_events=quote.charging_events,
            total_distance_m=float(actual_total_distance),
            travel_time_min=float(actual_travel_time),
            charging_time_min=quote.charging_time_min,
            total_time_min=float(actual_travel_time + quote.charging_time_min),
            arrival_battery_wh=quote.arrival_battery_wh,
            required_dropoff_reserve_wh=quote.required_dropoff_reserve_wh,
        )

    def plan(
        self,
        robot: RobotState,
        pickup_node: NodeId,
        dropoff_node: NodeId,
        *,
        start_to_pickup_m: float | None = None,
        pickup_to_dropoff_m: float | None = None,
    ) -> BatteryFeasibleRoute:
        """Compatibility helper: evaluate and immediately materialize one route."""

        quote = self.evaluate(
            robot,
            pickup_node,
            dropoff_node,
            start_to_pickup_m=start_to_pickup_m,
            pickup_to_dropoff_m=pickup_to_dropoff_m,
        )
        return self.materialize(quote, speed_mps=robot.spec.speed_mps)
