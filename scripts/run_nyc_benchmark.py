from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import argparse
import heapq
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterator, Literal

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.battery_routing import (
    BatteryFeasibleRouter,
    BatteryRouteQuote,
    NoFeasibleBatteryRoute,
)
from delivery_fleet.charging import DEFAULT_CHARGING_POWER_W, DEFAULT_NUMBER_OF_PORTS
from delivery_fleet.deadlines import delivery_deadline_min
from delivery_fleet.fleet import create_default_fleet, fleet_type_summary
from delivery_fleet.routing import ChargerDistanceIndex, DistanceOracle
from delivery_fleet.scenario_creator import Order, Scenario

PICKUP_HANDLING_MIN = 1.0
DROPOFF_HANDLING_MIN = 1.0
EDGE_WEIGHT = "length"


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal["travel", "pickup", "charge", "dropoff"]
    value: float = 0.0
    node_id: int | None = None


@dataclass(slots=True)
class PlanState:
    order: Order
    route: BatteryRouteQuote
    assigned_at_min: float
    direct_distance_m: float
    deadline_min: float
    actions: tuple[Action, ...]
    action_index: int = 0
    pickup_completed_at_min: float | None = None


@dataclass(slots=True)
class ChargeRequest:
    robot_id: int
    energy_added_wh: float
    arrival_time_min: float


@dataclass(slots=True)
class StationState:
    active: int = 0
    queue: deque[ChargeRequest] = field(default_factory=deque)
    peak_queue: int = 0


@dataclass(slots=True)
class Metrics:
    assignment_waits: list[float] = field(default_factory=list)
    delivery_times: list[float] = field(default_factory=list)
    direct_distances_m: list[float] = field(default_factory=list)
    route_distances_m: list[float] = field(default_factory=list)
    actual_service_times: list[float] = field(default_factory=list)
    planned_service_times: list[float] = field(default_factory=list)
    queue_waits_min: list[float] = field(default_factory=list)
    assignments_by_type: Counter[str] = field(default_factory=Counter)
    on_time: int = 0
    weighted_wait_objective: float = 0.0
    charge_sessions: int = 0
    queued_charge_sessions: int = 0
    assigned_by_scenario_end: int = 0
    delivered_by_scenario_end: int = 0


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def direct_distance_m(graph: nx.Graph, order: Order) -> float:
    """Exact A* pickup->dropoff distance using Haversine as lower bound."""

    def heuristic(a: int, b: int) -> float:
        da, db = graph.nodes[a], graph.nodes[b]
        lat1 = math.radians(float(da["y"]))
        lat2 = math.radians(float(db["y"]))
        dlat = lat2 - lat1
        dlon = math.radians(float(db["x"]) - float(da["x"]))
        h = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        )
        return 2.0 * 6_371_008.8 * math.asin(math.sqrt(h))

    return float(
        nx.astar_path_length(
            graph,
            order.pickup_node,
            order.dropoff_node,
            heuristic=heuristic,
            weight=EDGE_WEIGHT,
        )
    )


def candidate_robots_in_distance_order(
    order: Order,
    robots,
    oracle: DistanceOracle,
) -> Iterator[tuple[object, float]]:
    """Yield available capable robots from nearest to farthest with one search."""

    candidates = [
        robot
        for robot in robots
        if robot.available and robot.can_hold(order.item)
    ]
    if not candidates:
        return

    by_node: dict[int, list] = {}
    for robot in candidates:
        by_node.setdefault(int(robot.node_id), []).append(robot)
    for node_robots in by_node.values():
        node_robots.sort(key=lambda robot: robot.spec.id)

    # One outward Dijkstra from the pickup.  If the nearest robot is
    # battery-infeasible, iteration simply continues to the next target; the
    # shortest-path search is not restarted.
    for nearest in oracle.iter_nearest_targets(order.pickup_node, by_node):
        for robot in by_node[int(nearest.node_id)]:
            yield robot, nearest.distance_m


def leg_distance_m(router: BatteryFeasibleRouter, source: int, target: int) -> float:
    """Exact waypoint-leg distance without constructing its street-node path."""

    if source == target:
        return 0.0
    if source in router.station_set:
        return router.charger_index.distance(source, target)
    if target in router.station_set:
        return router.charger_index.distance(target, source)
    return router.distance_oracle.distance(source, target)


def build_actions(
    router: BatteryFeasibleRouter,
    route: BatteryRouteQuote,
) -> tuple[Action, ...]:
    """Build semantic actions directly from a route quote.

    This benchmark never reroutes a busy robot, so it only needs exact travel
    distances/times between semantic waypoints.  It deliberately avoids
    materializing full node-by-node street paths.  The general router can still
    materialize a quote when a policy/simulator needs edge-level movement.
    """

    actions: list[Action] = []
    pickup_added = False
    charge_index = 0

    for segment in route.segments:
        start = int(segment.waypoints[0])
        if start == route.pickup_node and not pickup_added:
            actions.append(Action("pickup", node_id=start))
            pickup_added = True

        if charge_index < len(route.charging_events):
            event = route.charging_events[charge_index]
            if event.node_id == segment.waypoints[0]:
                actions.append(
                    Action(
                        "charge",
                        value=float(event.energy_added_wh),
                        node_id=start,
                    )
                )
                charge_index += 1

        for source, target in zip(segment.waypoints, segment.waypoints[1:]):
            distance = leg_distance_m(router, int(source), int(target))
            if distance > 0:
                actions.append(
                    Action("travel", value=float(distance), node_id=int(target))
                )
            if target == route.pickup_node and not pickup_added:
                actions.append(Action("pickup", node_id=int(target)))
                pickup_added = True

    if not pickup_added:
        raise RuntimeError("route actions never reached pickup")
    if charge_index != len(route.charging_events):
        remaining = route.charging_events[charge_index:]
        raise RuntimeError(
            "not every planned charging event was materialized in route order: "
            f"next_unmatched={remaining[0].node_id!r}"
        )

    actions.append(Action("dropoff", node_id=int(route.dropoff_node)))
    travel_distance = sum(action.value for action in actions if action.kind == "travel")
    if not math.isclose(
        travel_distance,
        route.total_distance_m,
        rel_tol=2e-6,
        abs_tol=0.1,
    ):
        raise RuntimeError(
            f"action distance {travel_distance} != route distance {route.total_distance_m}"
        )
    return tuple(actions)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data" / "graphs" / "new_york_city.graphml",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=ROOT / "data" / "scenarios" / "nyc_reference_12h_seed42.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results.json")
    args = parser.parse_args()

    wall_start = time.perf_counter()
    print(f"loading graph: {args.graph}", flush=True)
    graph = nx.read_graphml(args.graph, node_type=int)
    print(
        f"graph nodes={graph.number_of_nodes():,} edges={graph.number_of_edges():,}",
        flush=True,
    )
    scenario = Scenario.load_json(args.scenario)
    print(
        f"orders={len(scenario.orders):,} duration={scenario.duration_minutes:.1f} min",
        flush=True,
    )

    station_nodes = tuple(
        int(node)
        for node, data in graph.nodes(data=True)
        if truthy(data.get("is_charging_station", False))
    )
    print(f"charging stations={len(station_nodes):,}", flush=True)

    robots = create_default_fleet(graph, seed=scenario.seed)
    robot_by_id = {r.spec.id: r for r in robots}
    print(f"robots={len(robots):,} types={fleet_type_summary(robots)}", flush=True)

    # Shared immutable routing services.  The charger index replaces repeated
    # 272k-node cutoff Dijkstras inside every candidate route evaluation.
    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)
    index_start = time.perf_counter()
    print("building shared charger distance index...", flush=True)
    charger_index = ChargerDistanceIndex(
        graph,
        station_nodes,
        oracle=oracle,
        edge_weight=EDGE_WEIGHT,
        build_dense=True,
    )
    print(
        f"charger index dense={charger_index.is_dense} "
        f"build={time.perf_counter()-index_start:.1f}s",
        flush=True,
    )
    router = BatteryFeasibleRouter(
        graph,
        station_nodes=station_nodes,
        edge_weight=EDGE_WEIGHT,
        distance_oracle=oracle,
        charger_index=charger_index,
    )

    station_state = {node: StationState() for node in station_nodes}
    plans: dict[int, PlanState] = {}
    pending: list[Order] = []
    metrics = Metrics()
    direct_cache: dict[int, float] = {}
    events: list[tuple[float, int, int, str, object]] = []
    counter = itertools.count()

    def push_event(time_min: float, priority: int, kind: str, payload: object) -> None:
        heapq.heappush(events, (float(time_min), priority, next(counter), kind, payload))

    for order in scenario.orders:
        push_event(order.request_time_min, 2, "order_arrival", order)

    def get_direct_distance(order: Order) -> float:
        direct = direct_cache.get(order.id)
        if direct is None:
            direct = direct_distance_m(graph, order)
            direct_cache[order.id] = direct
            oracle.remember_distance(order.pickup_node, order.dropoff_node, direct)
        return direct

    def start_charge(
        robot_id: int,
        station_node: int,
        energy_wh: float,
        now: float,
        queued_arrival: float | None = None,
    ) -> None:
        state = station_state[station_node]
        if state.active >= DEFAULT_NUMBER_OF_PORTS:
            raise RuntimeError("start_charge called without a free port")
        state.active += 1
        metrics.charge_sessions += 1
        if queued_arrival is not None:
            metrics.queue_waits_min.append(now - queued_arrival)
        duration = energy_wh / DEFAULT_CHARGING_POWER_W * 60.0
        push_event(now + duration, 0, "charge_complete", (robot_id, station_node))

    def request_charge(
        robot_id: int,
        station_node: int,
        energy_wh: float,
        now: float,
    ) -> None:
        state = station_state[station_node]
        if state.active < DEFAULT_NUMBER_OF_PORTS:
            start_charge(robot_id, station_node, energy_wh, now)
            return
        metrics.queued_charge_sessions += 1
        state.queue.append(ChargeRequest(robot_id, energy_wh, now))
        state.peak_queue = max(state.peak_queue, len(state.queue))

    def advance_robot(robot_id: int, now: float) -> None:
        plan = plans[robot_id]
        robot = robot_by_id[robot_id]
        while plan.action_index < len(plan.actions):
            action = plan.actions[plan.action_index]
            plan.action_index += 1
            if action.kind == "travel":
                push_event(
                    now + action.value / robot.spec.speed_mps / 60.0,
                    1,
                    "robot_ready",
                    robot_id,
                )
                return
            if action.kind == "pickup":
                plan.pickup_completed_at_min = now + PICKUP_HANDLING_MIN
                push_event(now + PICKUP_HANDLING_MIN, 1, "robot_ready", robot_id)
                return
            if action.kind == "charge":
                assert action.node_id is not None
                request_charge(robot_id, action.node_id, action.value, now)
                return
            if action.kind == "dropoff":
                push_event(
                    now + DROPOFF_HANDLING_MIN,
                    0,
                    "delivery_complete",
                    robot_id,
                )
                return
        raise RuntimeError("robot plan exhausted without dropoff")

    def nearest_feasible_robot_and_quote(
        order: Order,
    ) -> tuple[object, BatteryRouteQuote, float] | None:
        direct = get_direct_distance(order)
        for robot, start_to_pickup in candidate_robots_in_distance_order(
            order,
            robots,
            oracle,
        ):
            try:
                quote = router.evaluate(
                    robot,
                    order.pickup_node,
                    order.dropoff_node,
                    start_to_pickup_m=start_to_pickup,
                    pickup_to_dropoff_m=direct,
                )
            except NoFeasibleBatteryRoute:
                continue
            return robot, quote, direct
        return None

    def assign_order(
        order: Order,
        robot,
        route: BatteryRouteQuote,
        direct: float,
        now: float,
    ) -> None:
        deadline = delivery_deadline_min(
            order.request_time_min,
            direct,
            order.importance,
        )
        actions = build_actions(router, route)

        robot.available = False
        robot.current_order_id = order.id
        plans[robot.spec.id] = PlanState(
            order=order,
            route=route,
            assigned_at_min=now,
            direct_distance_m=direct,
            deadline_min=deadline,
            actions=actions,
        )
        metrics.assignment_waits.append(now - order.request_time_min)
        metrics.route_distances_m.append(route.total_distance_m)
        metrics.direct_distances_m.append(direct)
        metrics.planned_service_times.append(
            route.travel_time_min
            + route.charging_time_min
            + PICKUP_HANDLING_MIN
            + DROPOFF_HANDLING_MIN
        )
        metrics.assignments_by_type[
            getattr(robot.spec, "display_name", type(robot.spec).__name__)
        ] += 1
        if now <= scenario.duration_minutes + 1e-9:
            metrics.assigned_by_scenario_end += 1
        advance_robot(robot.spec.id, now)

    def dispatch_pending(now: float) -> None:
        while pending:
            assigned = False
            for idx, order in enumerate(pending):
                feasible = nearest_feasible_robot_and_quote(order)
                if feasible is None:
                    continue
                robot, route, direct = feasible
                pending.pop(idx)
                assign_order(order, robot, route, direct, now)
                assigned = True
                break
            if not assigned:
                return

    delivered = 0
    last_progress = time.perf_counter()
    now = 0.0

    while events:
        now, _, _, kind, payload = heapq.heappop(events)

        if kind == "order_arrival":
            pending.append(payload)
            dispatch_pending(now)
        elif kind == "robot_ready":
            advance_robot(int(payload), now)
        elif kind == "charge_complete":
            robot_id, station_node = payload
            state = station_state[station_node]
            state.active -= 1
            if state.active < 0:
                raise RuntimeError("negative charger occupancy")
            advance_robot(robot_id, now)
            if state.queue:
                request = state.queue.popleft()
                start_charge(
                    request.robot_id,
                    station_node,
                    request.energy_added_wh,
                    now,
                    queued_arrival=request.arrival_time_min,
                )
        elif kind == "delivery_complete":
            robot_id = int(payload)
            robot = robot_by_id[robot_id]
            plan = plans.pop(robot_id)
            delivery_time = now - plan.order.request_time_min
            metrics.delivery_times.append(delivery_time)
            metrics.actual_service_times.append(now - plan.assigned_at_min)
            metrics.weighted_wait_objective += plan.order.importance * delivery_time
            if now <= plan.deadline_min + 1e-9:
                metrics.on_time += 1
            if now <= scenario.duration_minutes + 1e-9:
                metrics.delivered_by_scenario_end += 1

            robot.node_id = plan.route.dropoff_node
            robot.battery_wh = plan.route.arrival_battery_wh
            robot.current_order_id = None
            robot.available = True
            delivered += 1
            dispatch_pending(now)
        else:
            raise RuntimeError(f"unknown event kind: {kind}")

        if time.perf_counter() - last_progress >= 30.0:
            print(
                f"progress delivered={delivered}/{len(scenario.orders)} "
                f"pending={len(pending)} active={len(plans)} "
                f"sim_t={now:.1f} wall={time.perf_counter()-wall_start:.1f}s",
                flush=True,
            )
            last_progress = time.perf_counter()

    if pending or plans or delivered != len(scenario.orders):
        raise RuntimeError(
            f"simulation ended incomplete: delivered={delivered}, "
            f"pending={len(pending)}, active={len(plans)}"
        )

    queue_total = float(sum(metrics.queue_waits_min))
    peak_queues = sorted(
        (state.peak_queue for state in station_state.values()),
        reverse=True,
    )
    results = {
        "scenario": scenario.graph_name,
        "scenario_seed": scenario.seed,
        "orders": len(scenario.orders),
        "robots": len(robots),
        "fleet_types": {
            str(k.value): int(v)
            for k, v in fleet_type_summary(robots).items()
        },
        "charging_stations": len(station_nodes),
        "charger_power_w": DEFAULT_CHARGING_POWER_W,
        "ports_per_station": DEFAULT_NUMBER_OF_PORTS,
        "routing_index_dense": charger_index.is_dense,
        "routing_index_build_seconds": time.perf_counter() - index_start,
        "delivered": delivered,
        "on_time": metrics.on_time,
        "on_time_pct": 100.0 * metrics.on_time / delivered,
        "weighted_wait_objective": metrics.weighted_wait_objective,
        "mean_request_to_delivery_min": float(np.mean(metrics.delivery_times)),
        "median_request_to_delivery_min": float(np.median(metrics.delivery_times)),
        "p95_request_to_delivery_min": percentile(metrics.delivery_times, 95),
        "mean_assignment_wait_min": float(np.mean(metrics.assignment_waits)),
        "median_assignment_wait_min": float(np.median(metrics.assignment_waits)),
        "p95_assignment_wait_min": percentile(metrics.assignment_waits, 95),
        "mean_actual_service_min": float(np.mean(metrics.actual_service_times)),
        "mean_planned_service_no_queue_min": float(np.mean(metrics.planned_service_times)),
        "mean_direct_distance_km": float(np.mean(metrics.direct_distances_m)) / 1000.0,
        "median_direct_distance_km": float(np.median(metrics.direct_distances_m)) / 1000.0,
        "mean_route_distance_km": float(np.mean(metrics.route_distances_m)) / 1000.0,
        "total_robot_distance_km": float(sum(metrics.route_distances_m)) / 1000.0,
        "charge_sessions": metrics.charge_sessions,
        "queued_charge_sessions": metrics.queued_charge_sessions,
        "queued_charge_pct": (
            100.0 * metrics.queued_charge_sessions / max(1, metrics.charge_sessions)
        ),
        "total_charger_queue_wait_min": queue_total,
        "mean_queue_wait_if_queued_min": (
            float(np.mean(metrics.queue_waits_min)) if metrics.queue_waits_min else 0.0
        ),
        "median_queue_wait_if_queued_min": (
            float(np.median(metrics.queue_waits_min)) if metrics.queue_waits_min else 0.0
        ),
        "max_queue_wait_min": max(metrics.queue_waits_min, default=0.0),
        "max_station_queue_length": peak_queues[0] if peak_queues else 0,
        "second_max_station_queue_length": peak_queues[1] if len(peak_queues) > 1 else 0,
        "assigned_by_scenario_end": metrics.assigned_by_scenario_end,
        "delivered_by_scenario_end": metrics.delivered_by_scenario_end,
        "simulation_finish_min": now if scenario.orders else 0.0,
        "simulation_finish_hours": (now / 60.0) if scenario.orders else 0.0,
        "assignments_by_robot_type": dict(metrics.assignments_by_type),
        "wall_clock_seconds": time.perf_counter() - wall_start,
    }

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("BENCHMARK_RESULTS_JSON")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
