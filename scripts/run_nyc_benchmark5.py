from __future__ import annotations

"""NYC Benchmark 5: interruptible reactive insertion.

Benchmark 5 keeps Benchmark 4's myopic objective -- minimize the newly arrived
request's planned completion time -- but removes B4's append-only restriction.
For every robot, the new pickup and drop-off may be inserted at any
precedence-feasible positions in the robot's remaining service-stop sequence.
Existing stops keep their relative order; the selected robot may reroute after
its currently committed street edge.

Important implementation details:
- cumulative payload/volume capacity is enforced at every service stop;
- delivery motion is edge-event based, so an en-route robot can be rerouted at
  the next graph node rather than only after finishing its current delivery;
- candidate search is exact for the stated objective, using admissible
  lower bounds to prune robots/insertion positions before expensive
  battery-feasible evaluation;
- only exact candidates that can beat the incumbent are sent through the
  battery router; the chosen sequence is cached as prepared step plans;
- charger queues are simulated exactly, but (as in B4) future queue delay is
  deliberately not predicted during assignment.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import run_nyc_benchmark2 as base
import run_nyc_benchmark2_fast as fast

from delivery_fleet.battery_routing import (
    BatteryFeasibleRouter,
    BatteryRouteQuote,
    ChargeEvent,
    NoFeasibleBatteryRoute,
    RouteQuoteSegment,
)
from delivery_fleet.charging import DEFAULT_CHARGING_POWER_W, DEFAULT_NUMBER_OF_PORTS
from delivery_fleet.deadlines import delivery_deadline_min, delivery_loss
from delivery_fleet.fleet import create_default_fleet, fleet_type_summary
from delivery_fleet.insertion_routing import evaluate_pair_from_committed_state
from delivery_fleet.robot import BATTERY_EPS_WH, RobotActivity, RobotNodeArrivalEvent, RobotState
from delivery_fleet.routing import ChargerDistanceIndex, DistanceOracle
from delivery_fleet.scenario_creator import Order, Scenario

PICKUP_HANDLING_MIN = 1.0
DROPOFF_HANDLING_MIN = 1.0
EDGE_WEIGHT = "length"
EPS = 1e-9
DISTANCE_EPS_M = 1e-3


@dataclass(frozen=True, slots=True)
class ServiceStop:
    order_id: int
    kind: Literal["pickup", "dropoff"]
    node_id: int


@dataclass(slots=True)
class ActiveOrder:
    order: Order
    direct_distance_m: float
    deadline_min: float
    assigned_at_min: float
    picked_up: bool = False


@dataclass(frozen=True, slots=True)
class RoutePhase:
    kind: Literal["travel", "charge"]
    target_node: int
    distance_m: float = 0.0
    energy_wh: float = 0.0


@dataclass(frozen=True, slots=True)
class StepPlan:
    stop: ServiceStop
    start_node: int
    start_battery_wh: float
    phases: tuple[RoutePhase, ...]
    route_time_min: float
    route_distance_m: float
    arrival_battery_wh: float


@dataclass(slots=True)
class RobotSchedule:
    orders: dict[int, ActiveOrder] = field(default_factory=dict)
    stops: list[ServiceStop] = field(default_factory=list)
    version: int = 0
    prepared_steps: list[StepPlan] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    node_id: int
    time_min: float
    battery_wh: float


@dataclass(frozen=True, slots=True)
class PrefixPlan:
    phases: tuple[RoutePhase, ...]
    total_time_min: float
    total_distance_m: float
    arrival_battery_wh: float


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    steps: tuple[StepPlan, ...]
    completion_by_order: dict[int, float]
    total_distance_m: float
    projected_loss: float
    new_completion_time_min: float


@dataclass(frozen=True, slots=True)
class InsertionChoice:
    robot_id: int
    pickup_pos: int
    dropoff_pos: int
    stops: tuple[ServiceStop, ...]
    evaluation: SequenceEvaluation
    snapshot: DecisionSnapshot


@dataclass(frozen=True, slots=True)
class TargetQuote:
    segments: tuple[RouteQuoteSegment, ...]
    charging_events: tuple[ChargeEvent, ...]
    total_distance_m: float
    travel_time_min: float
    charging_time_min: float
    total_time_min: float
    arrival_battery_wh: float


B5_STATS: dict[str, float] = {
    "selection_calls": 0.0,
    "robots_considered": 0.0,
    "robots_lb_pruned": 0.0,
    "insertion_candidates": 0.0,
    "capacity_pruned": 0.0,
    "insertion_lb_pruned": 0.0,
    "exact_sequence_evaluations": 0.0,
    "exact_sequence_seconds": 0.0,
    "no_feasible_sequence": 0.0,
    "assignments_to_busy_robots": 0.0,
    "non_append_insertions": 0.0,
    "interrupted_delivery_routes": 0.0,
    "interrupted_delivery_charging": 0.0,
    "interrupted_delivery_charge_queue": 0.0,
    "max_remaining_stops": 0.0,
    "max_active_orders": 0.0,
    "prepared_step_hits": 0.0,
    "prepared_step_misses": 0.0,
}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if values else 0.0


def _enumerate_insertions(
    existing: tuple[ServiceStop, ...],
    pickup: ServiceStop,
    dropoff: ServiceStop,
):
    """Yield all pickup-before-dropoff insertions preserving old stop order."""
    n = len(existing)
    for pickup_pos in range(n + 1):
        with_pickup = existing[:pickup_pos] + (pickup,) + existing[pickup_pos:]
        for dropoff_pos in range(pickup_pos + 1, n + 2):
            yield (
                pickup_pos,
                dropoff_pos,
                with_pickup[:dropoff_pos] + (dropoff,) + with_pickup[dropoff_pos:],
            )


def _capacity_feasible(
    robot: RobotState,
    stops: tuple[ServiceStop, ...],
    order_states: dict[int, ActiveOrder],
) -> bool:
    carried: set[int] = {
        order_id for order_id, state in order_states.items() if state.picked_up
    }
    weight = sum(order_states[oid].order.item.weight_kg for oid in carried)
    volume = sum(order_states[oid].order.item.volume_l for oid in carried)
    if weight > robot.spec.max_payload_kg + EPS or volume > robot.spec.max_volume_l + EPS:
        return False

    seen_pickups = set(carried)
    for stop in stops:
        state = order_states.get(stop.order_id)
        if state is None:
            return False
        item = state.order.item
        if stop.kind == "pickup":
            if stop.order_id in seen_pickups:
                return False
            seen_pickups.add(stop.order_id)
            carried.add(stop.order_id)
            weight += item.weight_kg
            volume += item.volume_l
            if (
                weight > robot.spec.max_payload_kg + EPS
                or volume > robot.spec.max_volume_l + EPS
            ):
                return False
        else:
            if stop.order_id not in carried:
                return False
            carried.remove(stop.order_id)
            weight -= item.weight_kg
            volume -= item.volume_l

    return not carried


def main() -> None:
    import argparse

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
        default=ROOT / "benchmark5_results.json",
    )
    args = parser.parse_args()

    fast._install_in_memory_graph_initialization()
    distance_row = fast._install_compiled_distance_backend()

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
        if _truthy(data.get("is_charging_station", False))
    )
    print(f"charging stations={len(station_nodes):,}", flush=True)

    robots = create_default_fleet(graph, seed=scenario.seed)
    robot_by_id = {robot.spec.id: robot for robot in robots}
    print(f"robots={len(robots):,} types={fleet_type_summary(robots)}", flush=True)

    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)
    print("building shared charger distance index...", flush=True)
    index_started = time.perf_counter()
    charger_index = ChargerDistanceIndex(
        graph,
        station_nodes,
        oracle=oracle,
        edge_weight=EDGE_WEIGHT,
        build_dense=True,
    )
    index_seconds = time.perf_counter() - index_started
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

    node_index = getattr(oracle, "_fast_node_index", None)
    if node_index is None:
        raise RuntimeError("compiled routing backend did not expose node index")

    schedules = {robot.spec.id: RobotSchedule() for robot in robots}
    station_state = {node: base.StationState() for node in station_nodes}
    charge_token = {robot.spec.id: 0 for robot in robots}
    pending: list[Order] = []
    metrics = base.Metrics()

    active_phases: dict[int, deque[RoutePhase]] = {
        robot.spec.id: deque() for robot in robots
    }
    phase_version: dict[int, int] = {robot.spec.id: -1 for robot in robots}
    service_lock: dict[int, tuple[int, ServiceStop]] = {}
    service_token = {robot.spec.id: 0 for robot in robots}

    pair_distance_cache: dict[frozenset[int], float] = {}
    coord_cache: dict[int, tuple[float, float]] = {}
    direct_distance_cache: dict[int, float] = {}

    delivery_distance_m = 0.0
    background_distance_m = 0.0

    events: list[tuple[float, int, int, str, object]] = []
    event_counter = itertools.count()

    def push_event(time_min: float, priority: int, kind: str, payload: object) -> None:
        heapq.heappush(
            events,
            (float(time_min), int(priority), next(event_counter), kind, payload),
        )

    for order in scenario.orders:
        push_event(order.request_time_min, 3, "order_arrival", order)

    def pair_key(a: int, b: int) -> frozenset[int]:
        return frozenset((int(a), int(b)))

    def remember_distance(a: int, b: int, value: float) -> float:
        value = float(value)
        if a != b:
            pair_distance_cache[pair_key(a, b)] = value
            oracle.remember_distance(a, b, value)
        return value

    def row_distance(row: np.ndarray, target: int) -> float:
        value = float(row[int(node_index[int(target)])])
        if not math.isfinite(value):
            raise nx.NetworkXNoPath(f"no path to node {target!r}")
        return value

    def exact_distance(
        a: int,
        b: int,
        special_rows: dict[int, np.ndarray] | None = None,
    ) -> float:
        a, b = int(a), int(b)
        if a == b:
            return 0.0
        cached = pair_distance_cache.get(pair_key(a, b))
        if cached is not None:
            return float(cached)
        if special_rows:
            row = special_rows.get(a)
            if row is not None:
                return remember_distance(a, b, row_distance(row, b))
            row = special_rows.get(b)
            if row is not None and not graph.is_directed():
                return remember_distance(a, b, row_distance(row, a))
        if a in router.station_set:
            return remember_distance(a, b, charger_index.distance(a, b))
        if b in router.station_set and not graph.is_directed():
            return remember_distance(a, b, charger_index.distance(b, a))
        return remember_distance(a, b, oracle.distance(a, b))

    def coords(node: int) -> tuple[float, float]:
        node = int(node)
        cached = coord_cache.get(node)
        if cached is not None:
            return cached
        data = graph.nodes[node]
        value = (float(data["y"]), float(data["x"]))
        coord_cache[node] = value
        return value

    def haversine_lb(a: int, b: int) -> float:
        if a == b:
            return 0.0
        lat1, lon1 = coords(a)
        lat2, lon2 = coords(b)
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dphi = p2 - p1
        dlambda = math.radians(lon2 - lon1)
        h = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
        )
        return 2.0 * 6_371_008.8 * math.asin(math.sqrt(h))
    def lower_bound_distance(
        a: int,
        b: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
        if a == b:
            return 0.0
        row = special_rows.get(int(a))
        if row is not None:
            return row_distance(row, int(b))
        row = special_rows.get(int(b))
        if row is not None and not graph.is_directed():
            return row_distance(row, int(a))
        cached = pair_distance_cache.get(pair_key(int(a), int(b)))
        if cached is not None:
            return float(cached)
        return haversine_lb(int(a), int(b))

    def get_direct_distance(order: Order, pickup_row: np.ndarray | None = None) -> float:
        cached = direct_distance_cache.get(order.id)
        if cached is not None:
            return cached
        if pickup_row is None:
            pickup_row = distance_row(oracle, order.pickup_node)
        value = row_distance(pickup_row, order.dropoff_node)
        remember_distance(order.pickup_node, order.dropoff_node, value)
        direct_distance_cache[order.id] = value
        return value

    def next_charge_token(robot_id: int) -> int:
        charge_token[robot_id] += 1
        return charge_token[robot_id]

    def find_active_charge(robot_id: int):
        for station_node, state in station_state.items():
            session = state.active.get(robot_id)
            if session is not None:
                return int(station_node), state, session
        return None

    def projected_charge_battery(robot: RobotState, now: float) -> float:
        found = find_active_charge(robot.spec.id)
        if found is None:
            return float(robot.battery_wh)
        _, _, session = found
        elapsed = max(0.0, now - session.start_time_min)
        added = min(
            session.energy_wh,
            elapsed * DEFAULT_CHARGING_POWER_W / 60.0,
        )
        return min(
            robot.spec.battery_capacity_wh,
            session.start_battery_wh + added,
        )

    def start_charge(request: base.ChargeRequest, station_node: int, now: float) -> None:
        state = station_state[station_node]
        if len(state.active) >= DEFAULT_NUMBER_OF_PORTS:
            raise RuntimeError("start_charge called without a free port")
        robot = robot_by_id[request.robot_id]
        token = next_charge_token(request.robot_id)
        session = base.ActiveCharge(
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
            min(float(energy_wh), robot.spec.battery_capacity_wh - robot.battery_wh),
        )
        if energy_wh <= 1e-8:
            robot.activity = RobotActivity.IDLE
            if purpose == "delivery":
                continue_delivery(robot_id, now)
            else:
                robot.available = True
                if pending:
                    dispatch_pending(now)
            return
        request = base.ChargeRequest(
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

    def remove_queue_request(robot_id: int):
        for station_node, state in station_state.items():
            if not state.queue:
                continue
            kept: deque[base.ChargeRequest] = deque()
            removed = None
            while state.queue:
                request = state.queue.popleft()
                if removed is None and request.robot_id == robot_id:
                    removed = request
                    continue
                kept.append(request)
            state.queue = kept
            if removed is not None:
                return int(station_node), removed
        return None

    def interrupt_charge(robot: RobotState, now: float) -> None:
        found = find_active_charge(robot.spec.id)
        if found is None:
            raise RuntimeError("charging robot has no active charging session")
        station_node, state, session = found
        robot.battery_wh = projected_charge_battery(robot, now)
        del state.active[robot.spec.id]
        next_charge_token(robot.spec.id)
        robot.activity = RobotActivity.IDLE
        if session.purpose == "background":
            metrics.interrupted_background_charging += 1
        else:
            B5_STATS["interrupted_delivery_charging"] += 1.0
        fill_free_ports(station_node, now)

    def cancel_stationary_wait(robot: RobotState) -> None:
        removed = remove_queue_request(robot.spec.id)
        if removed is None:
            raise RuntimeError("waiting robot not found in charger queue")
        _, request = removed
        robot.activity = RobotActivity.IDLE
        if request.purpose == "background":
            metrics.interrupted_charge_queue += 1
        else:
            B5_STATS["interrupted_delivery_charge_queue"] += 1.0

    def nearest_station(node: int) -> int:
        distances = charger_index.distances_to_stations(node)
        return int(min(distances.items(), key=lambda item: (item[1], str(item[0])))[0])

    def decision_snapshot(robot: RobotState, now: float) -> DecisionSnapshot | None:
        robot_id = robot.spec.id
        if robot_id in service_lock:
            return None
        if robot.activity is RobotActivity.MOVING:
            return DecisionSnapshot(
                node_id=int(robot.decision_node),
                time_min=float(robot.decision_time_min(now)),
                battery_wh=float(robot.battery_at_decision_node_wh),
            )
        if robot.activity is RobotActivity.CHARGING:
            return DecisionSnapshot(
                node_id=int(robot.node_id),
                time_min=float(now),
                battery_wh=float(projected_charge_battery(robot, now)),
            )
        return DecisionSnapshot(
            node_id=int(robot.node_id),
            time_min=float(now),
            battery_wh=float(robot.battery_wh),
        )

    def plan_final_target(
        robot: RobotState,
        target_node: int,
        *,
        start_to_target_m: float,
    ) -> PrefixPlan:
        """Shortest battery-feasible route from robot.node_id to one final stop."""
        start = int(robot.node_id)
        target_node = int(target_node)
        if start == target_node:
            return PrefixPlan((), 0.0, 0.0, float(robot.battery_wh))

        spec = robot.spec
        full_range = float(spec.full_battery_range_m)
        current_range = float(robot.remaining_range_m)
        reserve_distance = 0.0
        if target_node not in router.station_set:
            reserve_distance = min(charger_index.distances_to_stations(target_node).values())
        reserve_wh = reserve_distance * spec.energy_per_meter_wh

        start_is_station = start in router.station_set

        START = ("start",)
        DONE = ("done",)

        def physical_node(state: tuple) -> int:
            return start if state == START else int(state[1])

        def departure_range(state: tuple) -> float:
            if state == START:
                return full_range if start_is_station else current_range
            return full_range

        def distance_to_target(source: int) -> float:
            if source == start:
                return float(start_to_target_m)
            return float(charger_index.distance(source, target_node))

        best: dict[tuple, float] = {START: 0.0}
        previous: dict[tuple, tuple[tuple, tuple[float, tuple[int, ...]], bool]] = {}
        heap_counter = itertools.count()
        heap: list[tuple[float, int, tuple]] = [(0.0, next(heap_counter), START)]

        while heap:
            distance_so_far, _, state = heapq.heappop(heap)
            if distance_so_far != best.get(state):
                continue
            if state == DONE:
                break
            source = physical_node(state)
            available = departure_range(state)

            if source in router.station_set:
                station_neighbors = charger_index.station_neighbors(source, available)
            else:
                station_neighbors = charger_index.distances_to_stations(
                    source,
                    cutoff_m=available,
                    include_self=False,
                )
            for station, distance in station_neighbors.items():
                next_state = ("station", int(station))
                candidate = distance_so_far + float(distance)
                if candidate + DISTANCE_EPS_M < best.get(next_state, math.inf):
                    best[next_state] = candidate
                    previous[next_state] = (
                        state,
                        (float(distance), (source, int(station))),
                        True,
                    )
                    heapq.heappush(heap, (candidate, next(heap_counter), next_state))

            d_target = distance_to_target(source)
            if d_target + reserve_distance <= available + DISTANCE_EPS_M:
                candidate = distance_so_far + d_target
                if candidate + DISTANCE_EPS_M < best.get(DONE, math.inf):
                    best[DONE] = candidate
                    previous[DONE] = (
                        state,
                        (float(d_target), (source, target_node)),
                        False,
                    )
                    heapq.heappush(heap, (candidate, next(heap_counter), DONE))

        if DONE not in best:
            raise NoFeasibleBatteryRoute("no battery-feasible route to final service stop")

        raw_segments: list[tuple[float, tuple[int, ...], bool]] = []
        state = DONE
        while state != START:
            prev, edge, ends_at_charger = previous[state]
            raw_segments.append((edge[0], edge[1], ends_at_charger))
            state = prev
        raw_segments.reverse()

        battery = float(robot.battery_wh)
        phases: list[RoutePhase] = []
        charging_time = 0.0
        travel_time = 0.0
        total_distance = 0.0

        for index, (distance, waypoints, _) in enumerate(raw_segments):
            final = index == len(raw_segments) - 1
            required_after = reserve_wh if final else 0.0
            required_departure = distance * spec.energy_per_meter_wh + required_after
            if battery + BATTERY_EPS_WH < required_departure:
                node = int(waypoints[0])
                if node not in router.station_set:
                    raise RuntimeError("single-target route needs charge at non-station")
                energy = required_departure - battery
                phases.append(RoutePhase("charge", node, energy_wh=float(energy)))
                battery += energy
                charging_time += energy / DEFAULT_CHARGING_POWER_W * 60.0

            source, target = int(waypoints[0]), int(waypoints[-1])
            phases.append(RoutePhase("travel", target, distance_m=float(distance)))
            battery -= distance * spec.energy_per_meter_wh
            if battery < -BATTERY_EPS_WH:
                raise RuntimeError("battery became negative on single-target route")
            battery = max(0.0, battery)
            total_distance += distance
            travel_time += distance / spec.speed_mps / 60.0

        return PrefixPlan(
            phases=tuple(phases),
            total_time_min=float(travel_time + charging_time),
            total_distance_m=float(total_distance),
            arrival_battery_wh=float(battery),
        )

    def prefix_from_pair_quote(
        robot: RobotState,
        quote: BatteryRouteQuote,
        first_stop_node: int,
    ) -> PrefixPlan:
        first_stop_node = int(first_stop_node)
        battery = float(robot.battery_wh)
        rate = float(robot.spec.energy_per_meter_wh)
        event_index = 0
        phases: list[RoutePhase] = []
        elapsed = 0.0
        total_distance = 0.0

        for segment in quote.segments:
            if event_index < len(quote.charging_events):
                event = quote.charging_events[event_index]
                if (
                    int(event.node_id) == int(segment.waypoints[0])
                    and math.isclose(
                        float(event.battery_before_wh),
                        battery,
                        rel_tol=0.0,
                        abs_tol=0.02,
                    )
                ):
                    phases.append(
                        RoutePhase(
                            "charge",
                            int(event.node_id),
                            energy_wh=float(event.energy_added_wh),
                        )
                    )
                    battery = float(event.battery_after_wh)
                    elapsed += float(event.duration_min)
                    event_index += 1

            for source, target in zip(segment.waypoints, segment.waypoints[1:]):
                source, target = int(source), int(target)
                distance = float(base.leg_distance_m(router, source, target))
                phases.append(RoutePhase("travel", target, distance_m=distance))
                elapsed += distance / robot.spec.speed_mps / 60.0
                total_distance += distance
                battery -= distance * rate
                if battery < -BATTERY_EPS_WH:
                    raise RuntimeError("pair-prefix battery became negative")
                battery = max(0.0, battery)
                if target == first_stop_node:
                    return PrefixPlan(
                        phases=tuple(phases),
                        total_time_min=float(elapsed),
                        total_distance_m=float(total_distance),
                        arrival_battery_wh=float(battery),
                    )

        raise RuntimeError("battery quote did not reach first service stop")

    def first_distinct_later_node(stops: tuple[ServiceStop, ...], index: int) -> int | None:
        node = int(stops[index].node_id)
        for later in stops[index + 1 :]:
            if int(later.node_id) != node:
                return int(later.node_id)
        return None

    def evaluate_sequence(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        order_states: dict[int, ActiveOrder],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> SequenceEvaluation:
        started = time.perf_counter()
        B5_STATS["exact_sequence_evaluations"] += 1.0
        try:
            node = int(snapshot.node_id)
            battery = float(snapshot.battery_wh)
            now = float(snapshot.time_min)
            steps: list[StepPlan] = []
            completion: dict[int, float] = {}
            total_distance = 0.0

            for i, stop in enumerate(stops):
                if node == int(stop.node_id):
                    prefix = PrefixPlan((), 0.0, 0.0, battery)
                else:
                    lookahead = first_distinct_later_node(stops, i)
                    snapshot_robot = RobotState(
                        spec=robot.spec,
                        node_id=node,
                        battery_wh=battery,
                        available=False,
                    )
                    start_to_first = exact_distance(node, int(stop.node_id), special_rows)
                    if lookahead is None:
                        prefix = plan_final_target(
                            snapshot_robot,
                            int(stop.node_id),
                            start_to_target_m=start_to_first,
                        )
                    else:
                        first_to_next = exact_distance(
                            int(stop.node_id),
                            int(lookahead),
                            special_rows,
                        )
                        quote = evaluate_pair_from_committed_state(
                            router,
                            snapshot_robot,
                            int(stop.node_id),
                            int(lookahead),
                            start_to_pickup_m=start_to_first,
                            pickup_to_dropoff_m=first_to_next,
                        )
                        prefix = prefix_from_pair_quote(
                            snapshot_robot,
                            quote,
                            int(stop.node_id),
                        )

                steps.append(
                    StepPlan(
                        stop=stop,
                        start_node=node,
                        start_battery_wh=battery,
                        phases=prefix.phases,
                        route_time_min=prefix.total_time_min,
                        route_distance_m=prefix.total_distance_m,
                        arrival_battery_wh=prefix.arrival_battery_wh,
                    )
                )
                now += prefix.total_time_min
                total_distance += prefix.total_distance_m
                node = int(stop.node_id)
                battery = float(prefix.arrival_battery_wh)

                if stop.kind == "pickup":
                    now += PICKUP_HANDLING_MIN
                else:
                    now += DROPOFF_HANDLING_MIN
                    completion[stop.order_id] = float(now)

            if new_order_id not in completion:
                raise RuntimeError("candidate sequence never completed the new order")

            projected_loss = 0.0
            for order_id, state in order_states.items():
                done_at = completion.get(order_id)
                if done_at is None:
                    raise RuntimeError(f"candidate sequence did not complete order {order_id}")
                allowance = state.deadline_min - state.order.request_time_min
                projected_loss += delivery_loss(
                    done_at - state.order.request_time_min,
                    allowance,
                    state.order.importance,
                )

            return SequenceEvaluation(
                steps=tuple(steps),
                completion_by_order=completion,
                total_distance_m=float(total_distance),
                projected_loss=float(projected_loss),
                new_completion_time_min=float(completion[new_order_id]),
            )
        finally:
            B5_STATS["exact_sequence_seconds"] += time.perf_counter() - started

    def sequence_completion_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
        now = float(snapshot.time_min)
        node = int(snapshot.node_id)
        for stop in stops:
            distance = lower_bound_distance(node, int(stop.node_id), special_rows)
            now += distance / robot.spec.speed_mps / 60.0
            now += PICKUP_HANDLING_MIN if stop.kind == "pickup" else DROPOFF_HANDLING_MIN
            node = int(stop.node_id)
            if stop.kind == "dropoff" and stop.order_id == new_order_id:
                return float(now)
        return math.inf

    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:
        B5_STATS["selection_calls"] += 1.0
        pickup_row = distance_row(oracle, order.pickup_node)
        dropoff_row = distance_row(oracle, order.dropoff_node)
        special_rows = {
            int(order.pickup_node): pickup_row,
            int(order.dropoff_node): dropoff_row,
        }
        direct = get_direct_distance(order, pickup_row)
        direct_distance_cache[order.id] = direct

        pickup_stop = ServiceStop(order.id, "pickup", int(order.pickup_node))
        dropoff_stop = ServiceStop(order.id, "dropoff", int(order.dropoff_node))

        robot_rows: list[tuple[float, int, DecisionSnapshot]] = []
        for robot in robots:
            if not robot.can_hold(order.item):
                continue
            snapshot = decision_snapshot(robot, now)
            if snapshot is None:
                continue
            start_to_pickup = row_distance(pickup_row, snapshot.node_id)
            universal_lb = (
                snapshot.time_min
                + start_to_pickup / robot.spec.speed_mps / 60.0
                + PICKUP_HANDLING_MIN
                + direct / robot.spec.speed_mps / 60.0
                + DROPOFF_HANDLING_MIN
            )
            robot_rows.append((float(universal_lb), robot.spec.id, snapshot))

        robot_rows.sort(key=lambda row: (row[0], row[1]))
        best: InsertionChoice | None = None
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf

        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):
            if universal_lb >= best_completion - 1e-9:
                B5_STATS["robots_lb_pruned"] += float(len(robot_rows) - robot_index)
                break
            B5_STATS["robots_considered"] += 1.0
            robot = robot_by_id[robot_id]
            schedule = schedules[robot_id]
            existing = tuple(schedule.stops)

            new_state = ActiveOrder(
                order=order,
                direct_distance_m=direct,
                deadline_min=delivery_deadline_min(
                    order.request_time_min,
                    direct,
                    order.importance,
                ),
                assigned_at_min=float(now),
                picked_up=False,
            )
            order_states = dict(schedule.orders)
            order_states[order.id] = new_state

            candidates: list[tuple[float, int, int, tuple[ServiceStop, ...]]] = []
            for pickup_pos, dropoff_pos, stops in _enumerate_insertions(
                existing, pickup_stop, dropoff_stop
            ):
                B5_STATS["insertion_candidates"] += 1.0
                if not _capacity_feasible(robot, stops, order_states):
                    B5_STATS["capacity_pruned"] += 1.0
                    continue
                lb = sequence_completion_lower_bound(
                    robot,
                    snapshot,
                    stops,
                    order.id,
                    special_rows,
                )
                if lb >= best_completion - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += 1.0
                    continue
                candidates.append((lb, pickup_pos, dropoff_pos, stops))

            candidates.sort(key=lambda row: (row[0], row[1], row[2]))
            for candidate_index, (lb, pickup_pos, dropoff_pos, stops) in enumerate(candidates):
                if lb >= best_completion - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += float(
                        len(candidates) - candidate_index
                    )
                    break
                try:
                    evaluation = evaluate_sequence(
                        robot,
                        snapshot,
                        stops,
                        order_states,
                        order.id,
                        special_rows,
                    )
                except NoFeasibleBatteryRoute:
                    B5_STATS["no_feasible_sequence"] += 1.0
                    continue

                completion = evaluation.new_completion_time_min
                loss = evaluation.projected_loss
                distance = evaluation.total_distance_m
                better = (
                    completion < best_completion - 1e-9
                    or (
                        math.isclose(completion, best_completion, abs_tol=1e-9, rel_tol=0.0)
                        and (
                            loss < best_loss - 1e-6
                            or (
                                math.isclose(loss, best_loss, abs_tol=1e-6, rel_tol=0.0)
                                and (
                                    distance < best_distance - 1e-6
                                    or (
                                        math.isclose(
                                            distance,
                                            best_distance,
                                            abs_tol=1e-6,
                                            rel_tol=0.0,
                                        )
                                        and (
                                            best is None
                                            or (robot_id, pickup_pos, dropoff_pos)
                                            < (best.robot_id, best.pickup_pos, best.dropoff_pos)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                if better:
                    best_completion = completion
                    best_loss = loss
                    best_distance = distance
                    best = InsertionChoice(
                        robot_id=robot_id,
                        pickup_pos=pickup_pos,
                        dropoff_pos=dropoff_pos,
                        stops=stops,
                        evaluation=evaluation,
                        snapshot=snapshot,
                    )

        return best

    def cancel_editable_route_for_replan(robot: RobotState, now: float, was_busy: bool) -> None:
        robot_id = robot.spec.id
        active_phases[robot_id].clear()
        phase_version[robot_id] = -1

        if robot.activity is RobotActivity.MOVING:
            robot.replan_from_decision_node((robot.decision_node,))
            if was_busy:
                B5_STATS["interrupted_delivery_routes"] += 1.0
            else:
                metrics.interrupted_repositioning += 1
            return
        if robot.activity is RobotActivity.CHARGING:
            interrupt_charge(robot, now)
        elif robot.activity is RobotActivity.WAITING:
            cancel_stationary_wait(robot)
        elif robot.activity is RobotActivity.IDLE:
            if robot.remaining_route:
                robot.clear_movement_plan()
        else:
            raise RuntimeError(f"unsupported replan activity {robot.activity}")

    def commit_assignment(order: Order, choice: InsertionChoice, now: float) -> None:
        robot = robot_by_id[choice.robot_id]
        schedule = schedules[choice.robot_id]
        was_busy = bool(schedule.orders)
        old_stop_count = len(schedule.stops)

        if was_busy:
            B5_STATS["assignments_to_busy_robots"] += 1.0
        if choice.pickup_pos != old_stop_count or choice.dropoff_pos != old_stop_count + 1:
            B5_STATS["non_append_insertions"] += 1.0

        direct = direct_distance_cache[order.id]
        schedule.orders[order.id] = ActiveOrder(
            order=order,
            direct_distance_m=direct,
            deadline_min=delivery_deadline_min(
                order.request_time_min,
                direct,
                order.importance,
            ),
            assigned_at_min=float(now),
            picked_up=False,
        )
        schedule.stops = list(choice.stops)
        schedule.version += 1
        schedule.prepared_steps = list(choice.evaluation.steps)

        B5_STATS["max_remaining_stops"] = max(
            B5_STATS["max_remaining_stops"], float(len(schedule.stops))
        )
        B5_STATS["max_active_orders"] = max(
            B5_STATS["max_active_orders"], float(len(schedule.orders))
        )

        metrics.assignment_waits.append(now - order.request_time_min)
        metrics.direct_distances_m.append(direct)
        metrics.planned_service_times.append(
            choice.evaluation.new_completion_time_min - now
        )
        metrics.assignments_by_type[
            getattr(robot.spec, "display_name", type(robot.spec).__name__)
        ] += 1
        if now <= scenario.duration_minutes + EPS:
            metrics.assigned_by_scenario_end += 1

        cancel_editable_route_for_replan(robot, now, was_busy)
        robot.available = False
        robot.current_order_id = schedule.stops[0].order_id if schedule.stops else None

        if robot.activity is not RobotActivity.MOVING:
            continue_delivery(robot.spec.id, now)

    def dispatch_pending(now: float) -> None:
        while pending:
            assigned = False
            for index, order in enumerate(tuple(pending)):
                choice = choose_insertion(order, now)
                if choice is None:
                    continue
                pending.pop(index)
                commit_assignment(order, choice, now)
                assigned = True
                break
            if not assigned:
                return

    def begin_background_return(robot: RobotState, now: float) -> None:
        robot_id = robot.spec.id
        if schedules[robot_id].orders or robot_id in service_lock:
            return
        robot.current_order_id = None
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
            raise RuntimeError("background route had no edge")
        push_event(event.time_min, 1, "background_node_arrival", event)

    def path_distance(path: tuple[int, ...]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            data = graph.get_edge_data(a, b)
            if graph.is_multigraph():
                total += min(float(attrs.get(EDGE_WEIGHT, 1.0)) for attrs in data.values())
            else:
                total += float(data.get(EDGE_WEIGHT, 1.0))
        return total

    def start_travel_phase(robot: RobotState, phase: RoutePhase, now: float) -> None:
        if phase.kind != "travel":
            raise RuntimeError("start_travel_phase received non-travel phase")
        source = int(robot.node_id)
        target = int(phase.target_node)
        if source == target:
            continue_delivery(robot.spec.id, now)
            return
        path = charger_index.path(source, target)
        actual = path_distance(tuple(int(x) for x in path))
        if not math.isclose(actual, phase.distance_m, rel_tol=2e-6, abs_tol=0.1):
            raise RuntimeError(
                f"materialized phase distance {actual} != planned {phase.distance_m}"
            )
        robot.set_planned_path(path)
        event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
        if event is None:
            raise RuntimeError("travel phase had no edge")
        push_event(event.time_min, 1, "delivery_node_arrival", event)

    def compute_first_step(robot: RobotState, now: float) -> StepPlan:
        schedule = schedules[robot.spec.id]
        if not schedule.stops:
            raise RuntimeError("compute_first_step without service stops")
        snapshot = DecisionSnapshot(int(robot.node_id), float(now), float(robot.battery_wh))
        stops = tuple(schedule.stops)
        stop = stops[0]
        if int(robot.node_id) == int(stop.node_id):
            prefix = PrefixPlan((), 0.0, 0.0, float(robot.battery_wh))
        else:
            start_to_first = exact_distance(int(robot.node_id), int(stop.node_id))
            lookahead = first_distinct_later_node(stops, 0)
            snapshot_robot = RobotState(
                spec=robot.spec,
                node_id=int(robot.node_id),
                battery_wh=float(robot.battery_wh),
                available=False,
            )
            if lookahead is None:
                prefix = plan_final_target(
                    snapshot_robot,
                    int(stop.node_id),
                    start_to_target_m=start_to_first,
                )
            else:
                first_to_next = exact_distance(int(stop.node_id), int(lookahead))
                quote = evaluate_pair_from_committed_state(
                    router,
                    snapshot_robot,
                    int(stop.node_id),
                    int(lookahead),
                    start_to_pickup_m=start_to_first,
                    pickup_to_dropoff_m=first_to_next,
                )
                prefix = prefix_from_pair_quote(snapshot_robot, quote, int(stop.node_id))
        return StepPlan(
            stop=stop,
            start_node=int(robot.node_id),
            start_battery_wh=float(robot.battery_wh),
            phases=prefix.phases,
            route_time_min=prefix.total_time_min,
            route_distance_m=prefix.total_distance_m,
            arrival_battery_wh=prefix.arrival_battery_wh,
        )

    def start_service(robot_id: int, now: float) -> None:
        robot = robot_by_id[robot_id]
        schedule = schedules[robot_id]
        if not schedule.stops:
            raise RuntimeError("start_service without stop")
        stop = schedule.stops[0]
        if int(robot.node_id) != int(stop.node_id):
            raise RuntimeError("service started at wrong node")
        service_token[robot_id] += 1
        token = service_token[robot_id]
        service_lock[robot_id] = (token, stop)
        robot.activity = RobotActivity.IDLE
        robot.available = False
        duration = PICKUP_HANDLING_MIN if stop.kind == "pickup" else DROPOFF_HANDLING_MIN
        push_event(now + duration, 0, "service_complete", (robot_id, token))

    def continue_delivery(robot_id: int, now: float) -> None:
        robot = robot_by_id[robot_id]
        schedule = schedules[robot_id]
        if not schedule.orders:
            begin_background_return(robot, now)
            return
        if robot_id in service_lock:
            return
        if robot.activity in {RobotActivity.MOVING, RobotActivity.CHARGING, RobotActivity.WAITING}:
            return
        if not schedule.stops:
            raise RuntimeError("active schedule has no stops")

        robot.available = False
        robot.current_order_id = schedule.stops[0].order_id
        if int(robot.node_id) == int(schedule.stops[0].node_id):
            active_phases[robot_id].clear()
            phase_version[robot_id] = schedule.version
            start_service(robot_id, now)
            return

        if phase_version[robot_id] != schedule.version or not active_phases[robot_id]:
            step = None
            if schedule.prepared_steps:
                candidate = schedule.prepared_steps[0]
                if (
                    candidate.stop == schedule.stops[0]
                    and candidate.start_node == int(robot.node_id)
                    and math.isclose(
                        candidate.start_battery_wh,
                        robot.battery_wh,
                        rel_tol=0.0,
                        abs_tol=0.05,
                    )
                ):
                    step = candidate
                    B5_STATS["prepared_step_hits"] += 1.0
            if step is None:
                B5_STATS["prepared_step_misses"] += 1.0
                step = compute_first_step(robot, now)
            active_phases[robot_id] = deque(step.phases)
            phase_version[robot_id] = schedule.version

        if not active_phases[robot_id]:
            if int(robot.node_id) != int(schedule.stops[0].node_id):
                raise RuntimeError("empty phase list before reaching service stop")
            start_service(robot_id, now)
            return

        phase = active_phases[robot_id].popleft()
        if phase.kind == "charge":
            if int(robot.node_id) != int(phase.target_node):
                raise RuntimeError("charge phase at wrong node")
            request_charge(
                robot_id,
                int(phase.target_node),
                float(phase.energy_wh),
                now,
                "delivery",
            )
            return
        start_travel_phase(robot, phase, now)

    delivered = 0
    now = 0.0
    last_progress = time.perf_counter()

    while events:
        now, _, _, kind, payload = heapq.heappop(events)

        if kind == "order_arrival":
            pending.append(payload)
            dispatch_pending(now)

        elif kind in {"background_node_arrival", "delivery_node_arrival"}:
            event = payload
            assert isinstance(event, RobotNodeArrivalEvent)
            robot = robot_by_id[event.robot_id]
            if not robot.is_moving:
                continue
            edge_m = float(robot.current_edge_distance_m or 0.0)
            robot.arrive_at_next_node(event)
            if kind == "delivery_node_arrival":
                delivery_distance_m += edge_m
            else:
                background_distance_m += edge_m

            robot_id = robot.spec.id
            if schedules[robot_id].orders:
                if robot.remaining_route:
                    next_event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
                    if next_event is None:
                        raise RuntimeError("delivery path unexpectedly stopped")
                    push_event(next_event.time_min, 1, "delivery_node_arrival", next_event)
                else:
                    continue_delivery(robot_id, now)
                continue

            if pending:
                dispatch_pending(now)
            if schedules[robot_id].orders:
                continue_delivery(robot_id, now)
                continue

            if robot.remaining_route:
                next_event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
                if next_event is None:
                    raise RuntimeError("background path unexpectedly stopped")
                push_event(next_event.time_min, 1, "background_node_arrival", next_event)
            else:
                station = int(robot.node_id)
                if station not in router.station_set:
                    raise RuntimeError("background route ended at non-charger")
                energy = robot.spec.battery_capacity_wh - robot.battery_wh
                request_charge(robot_id, station, energy, now, "background")
                if pending:
                    dispatch_pending(now)

        elif kind == "charge_complete":
            robot_id, station_node, token = payload
            robot_id, station_node = int(robot_id), int(station_node)
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
            if schedules[robot_id].orders:
                continue_delivery(robot_id, now)
            else:
                robot.available = True
                if pending:
                    dispatch_pending(now)

        elif kind == "service_complete":
            robot_id, token = payload
            robot_id, token = int(robot_id), int(token)
            locked = service_lock.get(robot_id)
            if locked is None or locked[0] != token:
                continue
            _, stop = service_lock.pop(robot_id)
            schedule = schedules[robot_id]
            robot = robot_by_id[robot_id]
            if not schedule.stops or schedule.stops[0] != stop:
                raise RuntimeError("service completion no longer matches route front")
            schedule.stops.pop(0)
            if schedule.prepared_steps and schedule.prepared_steps[0].stop == stop:
                schedule.prepared_steps.pop(0)
            active_phases[robot_id].clear()
            phase_version[robot_id] = schedule.version

            state = schedule.orders[stop.order_id]
            if stop.kind == "pickup":
                if state.picked_up:
                    raise RuntimeError("order picked up twice")
                state.picked_up = True
            else:
                if not state.picked_up:
                    raise RuntimeError("dropoff before pickup")
                delivery_time = now - state.order.request_time_min
                metrics.delivery_times.append(delivery_time)
                metrics.actual_service_times.append(now - state.assigned_at_min)
                allowance = state.deadline_min - state.order.request_time_min
                metrics.weighted_wait_objective += delivery_loss(
                    delivery_time,
                    allowance,
                    state.order.importance,
                )
                if now <= state.deadline_min + EPS:
                    metrics.on_time += 1
                if now <= scenario.duration_minutes + EPS:
                    metrics.delivered_by_scenario_end += 1
                del schedule.orders[stop.order_id]
                delivered += 1

            robot.activity = RobotActivity.IDLE
            robot.available = False
            robot.current_order_id = schedule.stops[0].order_id if schedule.stops else None

            if pending:
                dispatch_pending(now)
            if schedules[robot_id].orders:
                continue_delivery(robot_id, now)
            else:
                robot.current_order_id = None
                begin_background_return(robot, now)

        else:
            raise RuntimeError(f"unknown event kind: {kind}")

        if (
            delivered == len(scenario.orders)
            and not pending
            and not any(schedule.orders for schedule in schedules.values())
            and not service_lock
        ):
            break

        if time.perf_counter() - last_progress >= 30.0:
            busy = sum(1 for schedule in schedules.values() if schedule.orders)
            scheduled = sum(len(schedule.orders) for schedule in schedules.values())
            print(
                f"progress delivered={delivered}/{len(scenario.orders)} "
                f"pending={len(pending)} busy_robots={busy} active_orders={scheduled} "
                f"sim_t={now:.1f} wall={time.perf_counter()-wall_start:.1f}s",
                flush=True,
            )
            last_progress = time.perf_counter()

    if (
        pending
        or any(schedule.orders for schedule in schedules.values())
        or service_lock
        or delivered != len(scenario.orders)
    ):
        raise RuntimeError(
            "simulation ended incomplete: "
            f"delivered={delivered}, pending={len(pending)}, "
            f"active_orders={sum(len(s.orders) for s in schedules.values())}, "
            f"service_locked={len(service_lock)}"
        )

    peak_queues = sorted(
        (state.peak_queue for state in station_state.values()), reverse=True
    )
    charge_sessions = metrics.delivery_charge_sessions + metrics.background_charge_sessions
    queued_sessions = (
        metrics.queued_delivery_charge_sessions + metrics.queued_background_charge_sessions
    )
    total_distance_m = delivery_distance_m + background_distance_m

    results = {
        "benchmark": 5,
        "policy": "reactive_insertion_interruptible_busy_routes",
        "selection_objective": "earliest_new_request_completion_time",
        "insertion_rule": "insert_new_pickup_and_dropoff_anywhere_feasible_preserving_existing_stop_order",
        "reroute_rule": "current_graph_edge_is_immutable_then_route_may_change",
        "scenario": scenario.graph_name,
        "scenario_seed": scenario.seed,
        "orders": len(scenario.orders),
        "robots": len(robots),
        "fleet_types": {
            str(key.value): int(value) for key, value in fleet_type_summary(robots).items()
        },
        "charging_stations": len(station_nodes),
        "charger_power_w": DEFAULT_CHARGING_POWER_W,
        "ports_per_station": DEFAULT_NUMBER_OF_PORTS,
        "routing_index_dense": charger_index.is_dense,
        "routing_index_build_seconds": index_seconds,
        "delivered": delivered,
        "on_time": metrics.on_time,
        "on_time_pct": 100.0 * metrics.on_time / max(1, delivered),
        "loss_formula": "w*min(D,T)+(w+1)^2*max(0,T-D)",
        "loss_objective": metrics.weighted_wait_objective,
        "weighted_wait_objective": metrics.weighted_wait_objective,
        "mean_request_to_delivery_min": float(np.mean(metrics.delivery_times)),
        "median_request_to_delivery_min": float(np.median(metrics.delivery_times)),
        "p95_request_to_delivery_min": _percentile(metrics.delivery_times, 95),
        "mean_assignment_wait_min": float(np.mean(metrics.assignment_waits)),
        "median_assignment_wait_min": float(np.median(metrics.assignment_waits)),
        "p95_assignment_wait_min": _percentile(metrics.assignment_waits, 95),
        "mean_actual_service_min": float(np.mean(metrics.actual_service_times)),
        "mean_planned_service_no_queue_min": float(np.mean(metrics.planned_service_times)),
        "mean_direct_distance_km": float(np.mean(metrics.direct_distances_m)) / 1000.0,
        "median_direct_distance_km": float(np.median(metrics.direct_distances_m)) / 1000.0,
        "mean_delivery_route_distance_km": delivery_distance_m / max(1, delivered) / 1000.0,
        "delivery_route_distance_km": delivery_distance_m / 1000.0,
        "background_reposition_distance_km": background_distance_m / 1000.0,
        "total_robot_distance_km": total_distance_m / 1000.0,
        "distance_metric_note": "B5 delivery distance is physical shared robot travel counted once",
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
        "second_max_station_queue_length": peak_queues[1] if len(peak_queues) > 1 else 0,
        "interrupted_repositioning": metrics.interrupted_repositioning,
        "interrupted_charge_queue": metrics.interrupted_charge_queue,
        "interrupted_background_charging": metrics.interrupted_background_charging,
        "assigned_by_scenario_end": metrics.assigned_by_scenario_end,
        "delivered_by_scenario_end": metrics.delivered_by_scenario_end,
        "simulation_finish_min": now if scenario.orders else 0.0,
        "simulation_finish_hours": (now / 60.0) if scenario.orders else 0.0,
        "assignments_by_robot_type": dict(metrics.assignments_by_type),
        "candidate_selection_calls": int(B5_STATS["selection_calls"]),
        "candidate_robots_considered": int(B5_STATS["robots_considered"]),
        "candidate_robots_lb_pruned": int(B5_STATS["robots_lb_pruned"]),
        "insertion_candidates": int(B5_STATS["insertion_candidates"]),
        "capacity_pruned_insertions": int(B5_STATS["capacity_pruned"]),
        "insertion_lb_pruned": int(B5_STATS["insertion_lb_pruned"]),
        "exact_sequence_evaluations": int(B5_STATS["exact_sequence_evaluations"]),
        "exact_sequence_seconds": float(B5_STATS["exact_sequence_seconds"]),
        "exact_sequence_mean_ms": (
            1000.0 * B5_STATS["exact_sequence_seconds"]
            / max(1.0, B5_STATS["exact_sequence_evaluations"])
        ),
        "no_feasible_sequence": int(B5_STATS["no_feasible_sequence"]),
        "assignments_to_busy_robots": int(B5_STATS["assignments_to_busy_robots"]),
        "non_append_insertions": int(B5_STATS["non_append_insertions"]),
        "interrupted_delivery_routes": int(B5_STATS["interrupted_delivery_routes"]),
        "interrupted_delivery_charging": int(B5_STATS["interrupted_delivery_charging"]),
        "interrupted_delivery_charge_queue": int(B5_STATS["interrupted_delivery_charge_queue"]),
        "max_remaining_stops": int(B5_STATS["max_remaining_stops"]),
        "max_active_orders_on_robot": int(B5_STATS["max_active_orders"]),
        "prepared_step_hits": int(B5_STATS["prepared_step_hits"]),
        "prepared_step_misses": int(B5_STATS["prepared_step_misses"]),
        "compiled_dijkstra_calls": int(fast._fast_dijkstra_calls),
        "compiled_dijkstra_cache_hits": int(fast._fast_dijkstra_cache_hits),
        "compiled_dijkstra_seconds": float(fast._fast_dijkstra_seconds),
        "wall_clock_seconds": time.perf_counter() - wall_start,
    }

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("BENCHMARK5_RESULTS_JSON")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
