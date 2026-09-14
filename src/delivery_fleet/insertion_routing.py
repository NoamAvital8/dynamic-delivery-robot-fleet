"""Battery routing helpers for interruptible insertion policies.

The standard :meth:`BatteryFeasibleRouter.evaluate` requires an idle robot at a
non-charger node to retain enough battery to reach a charger immediately.  That
is a useful dispatch safety invariant, but it is too strict at an intermediate
service/decision node of an already-committed multi-stop route: the robot may be
below that standalone reserve while still having a fully battery-feasible path
through the next mandatory stop to a charger.

``evaluate_pair_from_committed_state`` uses the same meta-graph and charging
semantics as ``BatteryFeasibleRouter.evaluate`` but omits only that standalone
start-reserve precheck.  All actual route legs remain range-feasible and the
second mandatory node still ends with the normal charger reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Hashable

from .battery_routing import (
    BatteryFeasibleRouter,
    BatteryRouteQuote,
    ChargeEvent,
    NoFeasibleBatteryRoute,
    RouteQuoteSegment,
)
from .charging import DEFAULT_CHARGING_POWER_W, DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR
from .robot import BATTERY_EPS_WH, RobotState

NodeId = Hashable
_DISTANCE_EPS_M = 1e-3


@dataclass(frozen=True, slots=True)
class _MetaEdge:
    distance_m: float
    waypoints: tuple[NodeId, ...]


def evaluate_pair_from_committed_state(
    router: BatteryFeasibleRouter,
    robot: RobotState,
    pickup_node: NodeId,
    dropoff_node: NodeId,
    *,
    start_to_pickup_m: float | None = None,
    pickup_to_dropoff_m: float | None = None,
) -> BatteryRouteQuote:
    """Evaluate start -> pickup -> dropoff for an already-committed route.

    This is intentionally identical to ``BatteryFeasibleRouter.evaluate``
    except that it does not reject a non-charger *start* merely because the
    current battery is below the distance-to-nearest-charger reserve.  The
    route itself must still be feasible with the actual battery, and the final
    dropoff retains the normal reserve needed to reach a charger.
    """

    graph = router.graph
    if pickup_node not in graph or dropoff_node not in graph:
        raise ValueError("pickup and dropoff must be graph nodes")
    if pickup_node == dropoff_node:
        raise ValueError("pickup and dropoff must be different")

    start = robot.node_id
    spec = robot.spec
    full_range = float(spec.full_battery_range_m)
    current_range = float(robot.remaining_range_m)
    reserve_distance = float(
        graph.nodes[dropoff_node][DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR]
    )
    reserve_wh = reserve_distance * spec.energy_per_meter_wh
    start_is_station = start in router.station_set

    pickup_station_dist = router.charger_index.distances_to_stations(
        pickup_node,
        cutoff_m=full_range,
    )
    dropoff_station_dist = router.charger_index.distances_to_stations(
        dropoff_node,
        cutoff_m=full_range,
    )

    if pickup_to_dropoff_m is None:
        pickup_to_dropoff_m = router.distance_oracle.distance(pickup_node, dropoff_node)
    else:
        pickup_to_dropoff_m = float(pickup_to_dropoff_m)
        router.distance_oracle.remember_distance(
            pickup_node,
            dropoff_node,
            pickup_to_dropoff_m,
        )

    if start_to_pickup_m is None:
        if start in router.station_set:
            start_to_pickup_m = pickup_station_dist.get(start)
            if start_to_pickup_m is None:
                start_to_pickup_m = router.charger_index.distance(start, pickup_node)
        else:
            start_to_pickup_m = router.distance_oracle.distance(start, pickup_node)
    else:
        start_to_pickup_m = float(start_to_pickup_m)
        router.distance_oracle.remember_distance(start, pickup_node, start_to_pickup_m)

    START = ("start",)
    DONE = ("done",)

    def physical_node(state: tuple) -> NodeId:
        return start if state == START else state[1]

    def departure_range(state: tuple) -> float:
        if state == START:
            return full_range if start_is_station else current_range
        return full_range

    def station_neighbors(source: NodeId, cutoff_m: float) -> dict[NodeId, float]:
        if source in router.station_set:
            return router.charger_index.station_neighbors(source, cutoff_m)
        return router.charger_index.distances_to_stations(
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
            for station, distance in station_neighbors(source, available_range).items():
                yield (
                    ("pre", station),
                    _MetaEdge(float(distance), (source, station)),
                )

            d_to_pickup = distance_station_to_pickup(source)
            if d_to_pickup is None or d_to_pickup > available_range + _DISTANCE_EPS_M:
                return

            if pickup_node in router.station_set:
                yield (
                    ("post", pickup_node),
                    _MetaEdge(d_to_pickup, (source, pickup_node)),
                )
                return

            for station, d_from_pickup in pickup_station_dist.items():
                segment_distance = d_to_pickup + float(d_from_pickup)
                if segment_distance <= available_range + _DISTANCE_EPS_M:
                    yield (
                        ("post", station),
                        _MetaEdge(segment_distance, (source, pickup_node, station)),
                    )

            direct_segment = d_to_pickup + pickup_to_dropoff_m
            if direct_segment + reserve_distance <= available_range + _DISTANCE_EPS_M:
                yield (
                    DONE,
                    _MetaEdge(direct_segment, (source, pickup_node, dropoff_node)),
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
            if d_to_dropoff + reserve_distance <= full_range + _DISTANCE_EPS_M:
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
            if candidate + _DISTANCE_EPS_M < best.get(next_state, math.inf):
                best[next_state] = candidate
                previous[next_state] = (state, edge)
                heapq.heappush(heap, (candidate, next(counter), next_state))

    if DONE not in best:
        raise NoFeasibleBatteryRoute(
            "no battery-feasible committed pair route exists for the selected robot"
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
            segment.distance_m * spec.energy_per_meter_wh + required_after_segment
        )
        if required_departure > spec.battery_capacity_wh + BATTERY_EPS_WH:
            raise RuntimeError("meta route contains an infeasible battery segment")

        if battery + BATTERY_EPS_WH < required_departure:
            start_node = segment.waypoints[0]
            if start_node not in router.station_set:
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
        if battery < -BATTERY_EPS_WH:
            raise RuntimeError("battery became negative on a feasible route")
        battery = max(0.0, battery)

    if battery + BATTERY_EPS_WH < reserve_wh:
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
