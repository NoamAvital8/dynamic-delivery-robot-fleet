from __future__ import annotations

"""NYC Benchmark 2: interruptible return-to-charge NAR.

Same scenario/fleet/routing/chargers as Benchmark 1, with one behavioral change:
after a delivery, an unassigned robot heads to its nearest charging station and
fills to 100%. Repositioning, charger-queue waiting, and background charging are
interruptible by this policy. Delivery work itself is never interrupted.

A moving background robot must finish its already-committed graph edge before a
new delivery route can take effect, matching RobotState's node-event movement
model. RobotState.available is not used as the source of truth here: whether a
physical state is considered dispatchable is a policy decision.
"""

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
from delivery_fleet.robot import RobotActivity, RobotNodeArrivalEvent, RobotState
from delivery_fleet.routing import ChargerDistanceIndex, DistanceOracle
from delivery_fleet.scenario_creator import Order, Scenario

PICKUP_HANDLING_MIN = 1.0
DROPOFF_HANDLING_MIN = 1.0
EDGE_WEIGHT = "length"
EPS = 1e-9


@dataclass(frozen=True, slots=True)
class DeliveryAction:
    kind: Literal["travel", "pickup", "charge", "dropoff"]
    value: float = 0.0
    node_id: int | None = None


@dataclass(slots=True)
class DeliveryPlan:
    order: Order
    route: BatteryRouteQuote
    assigned_at_min: float
    route_start_time_min: float
    direct_distance_m: float
    deadline_min: float
    actions: tuple[DeliveryAction, ...]
    action_index: int = 0


@dataclass(slots=True)
class DeferredAssignment:
    plan: DeliveryPlan
    expected_start_node: int
    expected_start_time_min: float


@dataclass(slots=True)
class ChargeRequest:
    robot_id: int
    energy_wh: float
    arrival_time_min: float
    purpose: Literal["delivery", "background"]


@dataclass(slots=True)
class ActiveCharge:
    robot_id: int
    station_node: int
    energy_wh: float
    start_time_min: float
    start_battery_wh: float
    purpose: Literal["delivery", "background"]
    token: int


@dataclass(slots=True)
class StationState:
    active: dict[int, ActiveCharge] = field(default_factory=dict)
    queue: deque[ChargeRequest] = field(default_factory=deque)
    peak_queue: int = 0


@dataclass(frozen=True, slots=True)
class Candidate:
    robot: RobotState
    policy_distance_m: float
    route_start_node: int
    route_start_time_min: float
    route_start_battery_wh: float
    decision_to_pickup_m: float


@dataclass(slots=True)
class Metrics:
    assignment_waits: list[float] = field(default_factory=list)
    delivery_times: list[float] = field(default_factory=list)
    direct_distances_m: list[float] = field(default_factory=list)
    delivery_route_distances_m: list[float] = field(default_factory=list)
    actual_service_times: list[float] = field(default_factory=list)
    planned_service_times: list[float] = field(default_factory=list)
    queue_waits_min: list[float] = field(default_factory=list)
    assignments_by_type: Counter[str] = field(default_factory=Counter)
    on_time: int = 0
    weighted_wait_objective: float = 0.0
    delivery_charge_sessions: int = 0
    background_charge_sessions: int = 0
    queued_delivery_charge_sessions: int = 0
    queued_background_charge_sessions: int = 0
    assigned_by_scenario_end: int = 0
    delivered_by_scenario_end: int = 0
    background_reposition_distance_m: float = 0.0
    interrupted_repositioning: int = 0
    interrupted_charge_queue: int = 0
    interrupted_background_charging: int = 0


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def direct_distance_m(graph: nx.Graph, order: Order) -> float:
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


def leg_distance_m(router: BatteryFeasibleRouter, source: int, target: int) -> float:
    if source == target:
        return 0.0
    if source in router.station_set:
        return router.charger_index.distance(source, target)
    if target in router.station_set:
        return router.charger_index.distance(target, source)
    return router.distance_oracle.distance(source, target)


def build_delivery_actions(
    router: BatteryFeasibleRouter,
    route: BatteryRouteQuote,
) -> tuple[DeliveryAction, ...]:
    actions: list[DeliveryAction] = []
    pickup_added = False
    charge_index = 0

    for segment in route.segments:
        start = int(segment.waypoints[0])
        if start == route.pickup_node and not pickup_added:
            actions.append(DeliveryAction("pickup", node_id=start))
            pickup_added = True

        if charge_index < len(route.charging_events):
            event = route.charging_events[charge_index]
            if event.node_id == segment.waypoints[0]:
                actions.append(
                    DeliveryAction(
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
                    DeliveryAction("travel", value=float(distance), node_id=int(target))
                )
            if target == route.pickup_node and not pickup_added:
                actions.append(DeliveryAction("pickup", node_id=int(target)))
                pickup_added = True

    if not pickup_added:
        raise RuntimeError("delivery route never reached pickup")
    if charge_index != len(route.charging_events):
        raise RuntimeError("not every delivery charging event was materialized")

    actions.append(DeliveryAction("dropoff", node_id=int(route.dropoff_node)))
    travel_distance = sum(a.value for a in actions if a.kind == "travel")
    if not math.isclose(
        travel_distance,
        route.total_distance_m,
        rel_tol=2e-6,
        abs_tol=0.1,
    ):
        raise RuntimeError(
            f"delivery action distance {travel_distance} != route {route.total_distance_m}"
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark2_results.json",
    )
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
    robot_by_id = {robot.spec.id: robot for robot in robots}
    print(f"robots={len(robots):,} types={fleet_type_summary(robots)}", flush=True)

    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)
    print("building shared charger distance index...", flush=True)
    index_start = time.perf_counter()
    charger_index = ChargerDistanceIndex(
        graph,
        station_nodes,
        oracle=oracle,
        edge_weight=EDGE_WEIGHT,
        build_dense=True,
    )
    index_seconds = time.perf_counter() - index_start
    print(
        f"charger index dense={charger_index.is_dense} build={index_seconds:.1f}s",
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
    delivery_plans: dict[int, DeliveryPlan] = {}
    deferred: dict[int, DeferredAssignment] = {}
    pending: list[Order] = []
    direct_cache: dict[int, float] = {}
    metrics = Metrics()
    charge_token: dict[int, int] = {robot.spec.id: 0 for robot in robots}

    events: list[tuple[float, int, int, str, object]] = []
    counter = itertools.count()

    def push_event(time_min: float, priority: int, kind: str, payload: object) -> None:
        heapq.heappush(
            events,
            (float(time_min), int(priority), next(counter), kind, payload),
        )

    for order in scenario.orders:
        push_event(order.request_time_min, 3, "order_arrival", order)

    def get_direct_distance(order: Order) -> float:
        direct = direct_cache.get(order.id)
        if direct is None:
            direct = direct_distance_m(graph, order)
            direct_cache[order.id] = direct
            oracle.remember_distance(order.pickup_node, order.dropoff_node, direct)
        return direct

    def next_token(robot_id: int) -> int:
        charge_token[robot_id] += 1
        return charge_token[robot_id]

    def projected_background_charge_battery(robot: RobotState, now: float) -> float:
        if robot.activity is not RobotActivity.CHARGING:
            return float(robot.battery_wh)
        for state in station_state.values():
            session = state.active.get(robot.spec.id)
            if session is None:
                continue
            if session.purpose != "background":
                return float(robot.battery_wh)
            elapsed = max(0.0, now - session.start_time_min)
            added = min(
                session.energy_wh,
                elapsed * DEFAULT_CHARGING_POWER_W / 60.0,
            )
            return min(robot.spec.battery_capacity_wh, session.start_battery_wh + added)
        return float(robot.battery_wh)

    def start_charge(request: ChargeRequest, station_node: int, now: float) -> None:
        state = station_state[station_node]
        if len(state.active) >= DEFAULT_NUMBER_OF_PORTS:
            raise RuntimeError("start_charge called without a free port")

        robot = robot_by_id[request.robot_id]
        token = next_token(request.robot_id)
        session = ActiveCharge(
            robot_id=request.robot_id,
            station_node=station_node,
            energy_wh=float(request.energy_wh),
            start_time_min=float(now),
            start_battery_wh=float(robot.battery_wh),
            purpose=request.purpose,
            token=token,
        )
        state.active[request.robot_id] = session
        robot.activity = RobotActivity.CHARGING
        robot.available = False

        if request.purpose == "delivery":
            metrics.delivery_charge_sessions += 1
        else:
            metrics.background_charge_sessions += 1

        if request.arrival_time_min + EPS < now:
            metrics.queue_waits_min.append(now - request.arrival_time_min)

        duration = request.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0
        push_event(
            now + duration,
            0,
            "charge_complete",
            (request.robot_id, station_node, token),
        )

    def fill_free_ports(station_node: int, now: float) -> None:
        state = station_state[station_node]
        while state.queue and len(state.active) < DEFAULT_NUMBER_OF_PORTS:
            start_charge(state.queue.popleft(), station_node, now)

    def request_charge(
        robot_id: int,
        station_node: int,
        energy_wh: float,
        now: float,
        purpose: Literal["delivery", "background"],
    ) -> None:
        robot = robot_by_id[robot_id]
        energy_wh = max(
            0.0,
            min(
                float(energy_wh),
                robot.spec.battery_capacity_wh - robot.battery_wh,
            ),
        )
        if energy_wh <= 1e-8:
            if purpose == "delivery":
                advance_delivery(robot_id, now)
            else:
                robot.activity = RobotActivity.IDLE
                robot.available = True
                if pending:
                    dispatch_pending(now)
            return

        request = ChargeRequest(
            robot_id=robot_id,
            energy_wh=energy_wh,
            arrival_time_min=float(now),
            purpose=purpose,
        )
        state = station_state[station_node]
        if len(state.active) < DEFAULT_NUMBER_OF_PORTS:
            start_charge(request, station_node, now)
            return

        state.queue.append(request)
        state.peak_queue = max(state.peak_queue, len(state.queue))
        robot.activity = RobotActivity.WAITING
        robot.available = False
        if purpose == "delivery":
            metrics.queued_delivery_charge_sessions += 1
        else:
            metrics.queued_background_charge_sessions += 1

    def remove_background_queue_request(robot_id: int) -> bool:
        for state in station_state.values():
            if not state.queue:
                continue
            kept: deque[ChargeRequest] = deque()
            removed = False
            while state.queue:
                req = state.queue.popleft()
                if (
                    not removed
                    and req.robot_id == robot_id
                    and req.purpose == "background"
                ):
                    removed = True
                    continue
                kept.append(req)
            state.queue = kept
            if removed:
                return True
        return False

    def interrupt_background_charging(robot: RobotState, now: float) -> None:
        robot_id = robot.spec.id
        for station_node, state in station_state.items():
            session = state.active.get(robot_id)
            if session is None or session.purpose != "background":
                continue

            robot.battery_wh = projected_background_charge_battery(robot, now)
            del state.active[robot_id]
            next_token(robot_id)
            robot.activity = RobotActivity.IDLE
            metrics.interrupted_background_charging += 1
            fill_free_ports(station_node, now)
            return
        raise RuntimeError("background-charging robot has no active session")

    def cancel_stationary_background(robot: RobotState, now: float) -> None:
        was_repositioning_at_node = bool(robot.remaining_route)
        if robot.activity is RobotActivity.CHARGING:
            interrupt_background_charging(robot, now)
        elif robot.activity is RobotActivity.WAITING:
            if not remove_background_queue_request(robot.spec.id):
                raise RuntimeError("waiting background robot was not found in charger queue")
            robot.activity = RobotActivity.IDLE
            metrics.interrupted_charge_queue += 1
        elif robot.activity is not RobotActivity.IDLE:
            raise RuntimeError(
                f"stationary background cancellation from {robot.activity}"
            )

        if was_repositioning_at_node:
            metrics.interrupted_repositioning += 1
        robot.available = False
        robot.clear_movement_plan()

    def nearest_station(node: int) -> int:
        distances = charger_index.distances_to_stations(node)
        return int(min(distances.items(), key=lambda item: (item[1], str(item[0])))[0])

    def begin_background_return(robot: RobotState, now: float) -> None:
        robot_id = robot.spec.id
        if robot_id in delivery_plans or robot_id in deferred:
            return
        if robot.current_order_id is not None:
            return

        if robot.battery_wh >= robot.spec.battery_capacity_wh - 1e-8:
            robot.activity = RobotActivity.IDLE
            robot.available = True
            return

        station = nearest_station(int(robot.node_id))
        if int(robot.node_id) == station:
            energy = robot.spec.battery_capacity_wh - robot.battery_wh
            request_charge(robot_id, station, energy, now, "background")
            return

        path = charger_index.path(int(robot.node_id), station)
        robot.activity = RobotActivity.IDLE
        robot.available = False
        robot.set_planned_path(path)
        event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
        if event is None:
            raise RuntimeError("non-station background route had no edge")
        push_event(event.time_min, 1, "background_node_arrival", event)

    def background_snapshot(
        robot: RobotState,
        now: float,
    ) -> tuple[int, float, float, float]:
        if robot.activity is RobotActivity.MOVING:
            start_node = int(robot.decision_node)
            start_time = robot.decision_time_min(now)
            battery = float(robot.battery_at_decision_node_wh)
            residual_m = max(0.0, start_time - now) * robot.spec.speed_mps * 60.0
            return start_node, start_time, battery, residual_m

        battery = (
            projected_background_charge_battery(robot, now)
            if robot.activity is RobotActivity.CHARGING
            else float(robot.battery_wh)
        )
        return int(robot.node_id), float(now), battery, 0.0

    def policy_candidates(order: Order, now: float) -> Iterator[Candidate]:
        candidates = [
            robot
            for robot in robots
            if robot.spec.id not in delivery_plans
            and robot.spec.id not in deferred
            and robot.current_order_id is None
            and robot.can_hold(order.item)
        ]
        if not candidates:
            return

        by_node: dict[int, list[tuple[RobotState, float, float, float]]] = {}
        for robot in candidates:
            node, start_time, battery, residual_m = background_snapshot(robot, now)
            by_node.setdefault(node, []).append((robot, start_time, battery, residual_m))
        for values in by_node.values():
            values.sort(key=lambda item: item[0].spec.id)

        buffered: list[tuple[float, int, Candidate]] = []
        for nearest in oracle.iter_nearest_targets(order.pickup_node, by_node):
            base = float(nearest.distance_m)
            while buffered and buffered[0][0] < base - 1e-9:
                yield heapq.heappop(buffered)[2]

            for robot, start_time, battery, residual_m in by_node[int(nearest.node_id)]:
                candidate = Candidate(
                    robot=robot,
                    policy_distance_m=residual_m + base,
                    route_start_node=int(nearest.node_id),
                    route_start_time_min=float(start_time),
                    route_start_battery_wh=float(battery),
                    decision_to_pickup_m=base,
                )
                heapq.heappush(
                    buffered,
                    (candidate.policy_distance_m, robot.spec.id, candidate),
                )

        while buffered:
            yield heapq.heappop(buffered)[2]

    def nearest_feasible_assignment(
        order: Order,
        now: float,
    ) -> tuple[Candidate, BatteryRouteQuote, float] | None:
        direct = get_direct_distance(order)
        for candidate in policy_candidates(order, now):
            snapshot = RobotState(
                spec=candidate.robot.spec,
                node_id=candidate.route_start_node,
                battery_wh=candidate.route_start_battery_wh,
            )
            try:
                quote = router.evaluate(
                    snapshot,
                    order.pickup_node,
                    order.dropoff_node,
                    start_to_pickup_m=candidate.decision_to_pickup_m,
                    pickup_to_dropoff_m=direct,
                )
            except NoFeasibleBatteryRoute:
                continue
            return candidate, quote, direct
        return None

    def advance_delivery(robot_id: int, now: float) -> None:
        plan = delivery_plans[robot_id]
        robot = robot_by_id[robot_id]

        while plan.action_index < len(plan.actions):
            action = plan.actions[plan.action_index]
            plan.action_index += 1

            if action.kind == "travel":
                assert action.node_id is not None
                travel_time = action.value / robot.spec.speed_mps / 60.0
                push_event(
                    now + travel_time,
                    1,
                    "delivery_travel_complete",
                    (robot_id, action.node_id, action.value),
                )
                return

            if action.kind == "pickup":
                push_event(now + PICKUP_HANDLING_MIN, 1, "delivery_ready", robot_id)
                return

            if action.kind == "charge":
                assert action.node_id is not None
                if int(robot.node_id) != int(action.node_id):
                    raise RuntimeError("delivery charge action at wrong node")
                request_charge(robot_id, action.node_id, action.value, now, "delivery")
                return

            if action.kind == "dropoff":
                push_event(now + DROPOFF_HANDLING_MIN, 0, "delivery_complete", robot_id)
                return

        raise RuntimeError("delivery plan exhausted without dropoff")

    def activate_assignment(robot: RobotState, plan: DeliveryPlan, now: float) -> None:
        robot_id = robot.spec.id
        start_node = int(plan.route.segments[0].waypoints[0])
        if int(robot.node_id) != start_node:
            raise RuntimeError("assignment activated from wrong start node")

        robot.clear_movement_plan()
        robot.available = False
        robot.current_order_id = plan.order.id
        delivery_plans[robot_id] = plan
        deferred.pop(robot_id, None)
        advance_delivery(robot_id, now)

    def assign_order(
        order: Order,
        candidate: Candidate,
        route: BatteryRouteQuote,
        direct: float,
        now: float,
    ) -> None:
        robot = candidate.robot
        robot_id = robot.spec.id
        plan = DeliveryPlan(
            order=order,
            route=route,
            assigned_at_min=float(now),
            route_start_time_min=candidate.route_start_time_min,
            direct_distance_m=direct,
            deadline_min=delivery_deadline_min(
                order.request_time_min,
                direct,
                order.importance,
            ),
            actions=build_delivery_actions(router, route),
        )

        metrics.assignment_waits.append(now - order.request_time_min)
        metrics.delivery_route_distances_m.append(route.total_distance_m)
        metrics.direct_distances_m.append(direct)
        metrics.planned_service_times.append(
            (candidate.route_start_time_min - now)
            + route.travel_time_min
            + route.charging_time_min
            + PICKUP_HANDLING_MIN
            + DROPOFF_HANDLING_MIN
        )
        metrics.assignments_by_type[
            getattr(robot.spec, "display_name", type(robot.spec).__name__)
        ] += 1
        if now <= scenario.duration_minutes + EPS:
            metrics.assigned_by_scenario_end += 1

        if robot.activity is RobotActivity.MOVING:
            deferred[robot_id] = DeferredAssignment(
                plan=plan,
                expected_start_node=candidate.route_start_node,
                expected_start_time_min=candidate.route_start_time_min,
            )
            robot.available = False
            metrics.interrupted_repositioning += 1
            return

        cancel_stationary_background(robot, now)
        if not math.isclose(
            robot.battery_wh,
            candidate.route_start_battery_wh,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("assignment battery snapshot changed during interruption")
        activate_assignment(robot, plan, now)

    def dispatch_pending(now: float) -> None:
        while pending:
            assigned = False
            for index, order in enumerate(pending):
                feasible = nearest_feasible_assignment(order, now)
                if feasible is None:
                    continue
                candidate, route, direct = feasible
                pending.pop(index)
                assign_order(order, candidate, route, direct, now)
                assigned = True
                break
            if not assigned:
                return

    delivered = 0
    now = 0.0
    last_progress = time.perf_counter()

    while events:
        now, _, _, kind, payload = heapq.heappop(events)

        if kind == "order_arrival":
            pending.append(payload)
            dispatch_pending(now)

        elif kind == "background_node_arrival":
            event = payload
            assert isinstance(event, RobotNodeArrivalEvent)
            robot = robot_by_id[event.robot_id]
            if not robot.is_moving:
                continue

            edge_m = float(robot.current_edge_distance_m or 0.0)
            robot.arrive_at_next_node(event)
            metrics.background_reposition_distance_m += edge_m

            deferred_assignment = deferred.get(robot.spec.id)
            if deferred_assignment is not None:
                if int(robot.node_id) != deferred_assignment.expected_start_node:
                    raise RuntimeError("deferred assignment reached wrong decision node")
                if not math.isclose(
                    now,
                    deferred_assignment.expected_start_time_min,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError("deferred assignment started at wrong time")
                robot.remaining_route.clear()
                activate_assignment(robot, deferred_assignment.plan, now)
                continue

            if pending:
                dispatch_pending(now)
            if robot.spec.id in delivery_plans or robot.spec.id in deferred:
                continue

            if robot.remaining_route:
                next_event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
                if next_event is None:
                    raise RuntimeError("background route unexpectedly stopped")
                push_event(next_event.time_min, 1, "background_node_arrival", next_event)
            else:
                station = int(robot.node_id)
                if station not in router.station_set:
                    raise RuntimeError("background route ended at non-charger")
                energy = robot.spec.battery_capacity_wh - robot.battery_wh
                request_charge(robot.spec.id, station, energy, now, "background")
                if pending:
                    dispatch_pending(now)

        elif kind == "delivery_travel_complete":
            robot_id, target, distance_m = payload
            robot_id = int(robot_id)
            robot = robot_by_id[robot_id]
            robot.battery_wh -= float(distance_m) * robot.spec.energy_per_meter_wh
            if robot.battery_wh < -1e-6:
                raise RuntimeError("delivery travel made battery negative")
            robot.battery_wh = max(0.0, robot.battery_wh)
            robot.node_id = int(target)
            advance_delivery(robot_id, now)

        elif kind == "delivery_ready":
            advance_delivery(int(payload), now)

        elif kind == "charge_complete":
            robot_id, station_node, token = payload
            robot_id = int(robot_id)
            station_node = int(station_node)
            state = station_state[station_node]
            session = state.active.get(robot_id)
            if session is None or session.token != int(token):
                continue

            del state.active[robot_id]
            robot = robot_by_id[robot_id]
            robot.battery_wh = min(
                robot.spec.battery_capacity_wh,
                session.start_battery_wh + session.energy_wh,
            )
            robot.activity = RobotActivity.IDLE

            fill_free_ports(station_node, now)

            if session.purpose == "delivery":
                advance_delivery(robot_id, now)
            else:
                robot.available = True
                if pending:
                    dispatch_pending(now)

        elif kind == "delivery_complete":
            robot_id = int(payload)
            robot = robot_by_id[robot_id]
            plan = delivery_plans.pop(robot_id)

            if int(robot.node_id) != int(plan.route.dropoff_node):
                raise RuntimeError("delivery completed away from dropoff")
            if not math.isclose(
                robot.battery_wh,
                plan.route.arrival_battery_wh,
                rel_tol=2e-6,
                abs_tol=0.05,
            ):
                raise RuntimeError(
                    "runtime battery disagrees with planned delivery arrival battery: "
                    f"{robot.battery_wh} != {plan.route.arrival_battery_wh}"
                )

            delivery_time = now - plan.order.request_time_min
            metrics.delivery_times.append(delivery_time)
            metrics.actual_service_times.append(now - plan.assigned_at_min)
            metrics.weighted_wait_objective += plan.order.importance * delivery_time
            if now <= plan.deadline_min + EPS:
                metrics.on_time += 1
            if now <= scenario.duration_minutes + EPS:
                metrics.delivered_by_scenario_end += 1

            robot.current_order_id = None
            robot.activity = RobotActivity.IDLE
            robot.available = True
            delivered += 1

            if pending:
                dispatch_pending(now)
            if robot_id not in delivery_plans and robot_id not in deferred:
                begin_background_return(robot, now)

        else:
            raise RuntimeError(f"unknown event kind: {kind}")

        if (
            delivered == len(scenario.orders)
            and not pending
            and not delivery_plans
            and not deferred
        ):
            break

        if time.perf_counter() - last_progress >= 30.0:
            bg_moving = sum(
                1
                for robot in robots
                if robot.current_order_id is None
                and robot.spec.id not in deferred
                and robot.activity is RobotActivity.MOVING
            )
            bg_charging = sum(
                1
                for robot in robots
                if robot.current_order_id is None
                and robot.spec.id not in deferred
                and robot.activity is RobotActivity.CHARGING
            )
            print(
                f"progress delivered={delivered}/{len(scenario.orders)} "
                f"pending={len(pending)} busy={len(delivery_plans)+len(deferred)} "
                f"bg_moving={bg_moving} bg_charging={bg_charging} "
                f"sim_t={now:.1f} wall={time.perf_counter()-wall_start:.1f}s",
                flush=True,
            )
            last_progress = time.perf_counter()

    if pending or delivery_plans or deferred or delivered != len(scenario.orders):
        raise RuntimeError(
            "simulation ended incomplete: "
            f"delivered={delivered}, pending={len(pending)}, "
            f"active={len(delivery_plans)}, deferred={len(deferred)}"
        )

    peak_queues = sorted(
        (state.peak_queue for state in station_state.values()), reverse=True
    )
    delivery_distance = float(sum(metrics.delivery_route_distances_m))
    total_distance = delivery_distance + metrics.background_reposition_distance_m
    charge_sessions = metrics.delivery_charge_sessions + metrics.background_charge_sessions
    queued_sessions = (
        metrics.queued_delivery_charge_sessions
        + metrics.queued_background_charge_sessions
    )

    results = {
        "benchmark": 2,
        "policy": "nearest_available_robot_with_interruptible_return_to_charge",
        "scenario": scenario.graph_name,
        "scenario_seed": scenario.seed,
        "orders": len(scenario.orders),
        "robots": len(robots),
        "fleet_types": {
            str(key.value): int(value)
            for key, value in fleet_type_summary(robots).items()
        },
        "charging_stations": len(station_nodes),
        "charger_power_w": DEFAULT_CHARGING_POWER_W,
        "ports_per_station": DEFAULT_NUMBER_OF_PORTS,
        "routing_index_dense": charger_index.is_dense,
        "routing_index_build_seconds": index_seconds,
        "delivered": delivered,
        "on_time": metrics.on_time,
        "on_time_pct": 100.0 * metrics.on_time / max(1, delivered),
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
        "mean_delivery_route_distance_km": (
            float(np.mean(metrics.delivery_route_distances_m)) / 1000.0
        ),
        "delivery_route_distance_km": delivery_distance / 1000.0,
        "background_reposition_distance_km": (
            metrics.background_reposition_distance_m / 1000.0
        ),
        "total_robot_distance_km": total_distance / 1000.0,
        "charge_sessions": charge_sessions,
        "delivery_charge_sessions": metrics.delivery_charge_sessions,
        "background_charge_sessions": metrics.background_charge_sessions,
        "queued_charge_sessions": queued_sessions,
        "queued_delivery_charge_sessions": metrics.queued_delivery_charge_sessions,
        "queued_background_charge_sessions": metrics.queued_background_charge_sessions,
        "queued_charge_pct": 100.0 * queued_sessions / max(1, charge_sessions),
        "total_charger_queue_wait_min": float(sum(metrics.queue_waits_min)),
        "mean_queue_wait_if_queued_min": (
            float(np.mean(metrics.queue_waits_min)) if metrics.queue_waits_min else 0.0
        ),
        "median_queue_wait_if_queued_min": (
            float(np.median(metrics.queue_waits_min)) if metrics.queue_waits_min else 0.0
        ),
        "max_queue_wait_min": max(metrics.queue_waits_min, default=0.0),
        "max_station_queue_length": peak_queues[0] if peak_queues else 0,
        "second_max_station_queue_length": (
            peak_queues[1] if len(peak_queues) > 1 else 0
        ),
        "interrupted_repositioning": metrics.interrupted_repositioning,
        "interrupted_charge_queue": metrics.interrupted_charge_queue,
        "interrupted_background_charging": metrics.interrupted_background_charging,
        "assigned_by_scenario_end": metrics.assigned_by_scenario_end,
        "delivered_by_scenario_end": metrics.delivered_by_scenario_end,
        "simulation_finish_min": now if scenario.orders else 0.0,
        "simulation_finish_hours": (now / 60.0) if scenario.orders else 0.0,
        "assignments_by_robot_type": dict(metrics.assignments_by_type),
        "wall_clock_seconds": time.perf_counter() - wall_start,
    }

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("BENCHMARK2_RESULTS_JSON")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
