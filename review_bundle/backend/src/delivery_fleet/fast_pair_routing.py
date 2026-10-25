"""Exact fast paths for committed two-stop battery routing.

Most B5 candidate evaluations do not need a charger detour at all.  The generic
committed-state helper nevertheless builds/searches the charger meta graph.
This wrapper proves when the physically shortest mandatory route

    start -> first stop -> second stop

is already battery-feasible.  In that case no charger detour can have shorter
physical distance (each leg is itself a shortest-path distance), so we can
construct exactly the same direct quote in O(1).  Only candidates that really
need an intermediate charger fall back to the full meta-graph search.
"""

from __future__ import annotations

from .battery_routing import BatteryRouteQuote, ChargeEvent, RouteQuoteSegment
from .charging import DEFAULT_CHARGING_POWER_W, DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR
from .insertion_routing import evaluate_pair_from_committed_state as _full_evaluate
from .robot import BATTERY_EPS_WH, RobotState

_DISTANCE_EPS_M = 1e-3


def evaluate_pair_from_committed_state_fast(
    router,
    robot: RobotState,
    pickup_node,
    dropoff_node,
    *,
    start_to_pickup_m: float | None = None,
    pickup_to_dropoff_m: float | None = None,
) -> BatteryRouteQuote:
    """Return the exact direct quote when provably optimal, else use full search."""
    start = robot.node_id
    spec = robot.spec
    rate = float(spec.energy_per_meter_wh)
    full_range = float(spec.full_battery_range_m)
    start_is_station = start in router.station_set

    if pickup_to_dropoff_m is None:
        pickup_to_dropoff_m = float(
            router.distance_oracle.distance(pickup_node, dropoff_node)
        )
    else:
        pickup_to_dropoff_m = float(pickup_to_dropoff_m)
        router.distance_oracle.remember_distance(
            pickup_node, dropoff_node, pickup_to_dropoff_m
        )

    if start_to_pickup_m is None:
        if start in router.station_set:
            start_to_pickup_m = float(router.charger_index.distance(start, pickup_node))
        else:
            start_to_pickup_m = float(
                router.distance_oracle.distance(start, pickup_node)
            )
    else:
        start_to_pickup_m = float(start_to_pickup_m)
        router.distance_oracle.remember_distance(start, pickup_node, start_to_pickup_m)

    reserve_distance = float(
        router.graph.nodes[dropoff_node][DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR]
    )
    reserve_wh = reserve_distance * rate
    mandatory_distance = start_to_pickup_m + pickup_to_dropoff_m
    required_range = mandatory_distance + reserve_distance
    available_range = full_range if start_is_station else float(robot.remaining_range_m)

    if required_range <= available_range + _DISTANCE_EPS_M:
        required_departure_wh = mandatory_distance * rate + reserve_wh
        battery = float(robot.battery_wh)
        charging_events: tuple[ChargeEvent, ...] = ()
        charging_time = 0.0
        if battery + BATTERY_EPS_WH < required_departure_wh:
            # This can only happen at a charging-station start because a
            # non-station direct route was checked against the current range.
            if not start_is_station:
                return _full_evaluate(
                    router,
                    robot,
                    pickup_node,
                    dropoff_node,
                    start_to_pickup_m=start_to_pickup_m,
                    pickup_to_dropoff_m=pickup_to_dropoff_m,
                )
            energy = required_departure_wh - battery
            before = battery
            battery += energy
            charging_time = energy / DEFAULT_CHARGING_POWER_W * 60.0
            charging_events = (
                ChargeEvent(
                    node_id=start,
                    energy_added_wh=float(energy),
                    duration_min=float(charging_time),
                    battery_before_wh=float(before),
                    battery_after_wh=float(battery),
                ),
            )

        battery -= mandatory_distance * rate
        segment = RouteQuoteSegment(
            waypoints=(start, pickup_node, dropoff_node),
            distance_m=float(mandatory_distance),
            ends_at_charger=False,
        )
        travel_time = mandatory_distance / float(spec.speed_mps) / 60.0
        return BatteryRouteQuote(
            pickup_node=pickup_node,
            dropoff_node=dropoff_node,
            segments=(segment,),
            charging_events=charging_events,
            total_distance_m=float(mandatory_distance),
            travel_time_min=float(travel_time),
            charging_time_min=float(charging_time),
            total_time_min=float(travel_time + charging_time),
            arrival_battery_wh=float(max(0.0, battery)),
            required_dropoff_reserve_wh=float(reserve_wh),
        )

    return _full_evaluate(
        router,
        robot,
        pickup_node,
        dropoff_node,
        start_to_pickup_m=start_to_pickup_m,
        pickup_to_dropoff_m=pickup_to_dropoff_m,
    )
