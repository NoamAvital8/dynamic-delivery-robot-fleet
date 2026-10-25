from __future__ import annotations

"""Run the exact-loss dispatcher with coordinated predictive idle control.

The existing Haversine top-K dispatcher is unchanged.  When a robot has no
delivery, this runner chooses STAY, REPOSITION or CHARGE using only the current
Gamma-Poisson posterior.  Background charge always continues to 100% unless
the robot receives a real delivery assignment.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_nyc_heuristic_policy as heuristic


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def _patch_idle(source: str) -> str:
    source = _replace_once(
        source,
        "from delivery_fleet.spatial_demand import haversine_node_distance_m\n",
        "from functools import lru_cache\n"
        "from delivery_fleet.spatial_demand import (\n"
        "    GammaPoissonDemandModel, haversine_node_distance_m,\n"
        ")\n"
        "from delivery_fleet.idle_planning import (\n"
        "    ChargingCalendar, IdleFleetPlanner, IdlePlanningConfig,\n"
        "    IdleRobotSnapshot,\n"
        "    ReadinessStep, project_remaining_readiness,\n"
        ")\n",
        "idle planning imports",
    )
    source = _replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    parser.add_argument('--idle-processes', type=int, default=8,\n"
        "                        help='Workers for large idle-action vector batches.')\n"
        "    parser.add_argument('--idle-horizon-min', type=float, default=45.0)\n"
        "    parser.add_argument('--idle-replan-interval-min', type=float, default=15.0)\n"
        "    parser.add_argument('--idle-candidate-clusters', type=int, default=5)\n"
        "    parser.add_argument('--idle-candidate-chargers', type=int, default=3)\n"
        "    parser.add_argument('--idle-max-demand-clusters', type=int, default=32)\n"
        "    parser.add_argument('--idle-max-actions-per-epoch', type=int, default=32)\n"
        "    parser.add_argument('--idle-minimum-gain-fraction', type=float, default=0.001)\n"
        "    parser.add_argument('--idle-relocation-uncertainty-penalty', type=float, default=0.0,\n"
        "                        help='Subtract this many posterior gain standard deviations from relocation; 0 uses the original mean score.')\n"
        "    args = parser.parse_args()\n"
        "    if args.idle_replan_interval_min <= 0:\n"
        "        parser.error('--idle-replan-interval-min must be positive')\n",
        "idle CLI",
    )
    source = _replace_once(
        source,
        "    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)\n",
        "    if reservation_policy is None:\n"
        "        try:\n"
        "            idle_prior_rates = {\n"
        "                float(key): float(value)\n"
        "                for key, value in json.loads(\n"
        "                    args.importance_prior_rates_per_hour\n"
        "                ).items()\n"
        "            }\n"
        "        except (TypeError, ValueError, json.JSONDecodeError) as exc:\n"
        "            parser.error(f'invalid importance prior rates JSON: {exc}')\n"
        "        idle_demand = GammaPoissonDemandModel(\n"
        "            graph, idle_prior_rates,\n"
        "            prior_concentration=args.prior_concentration,\n"
        "        )\n"
        "    else:\n"
        "        idle_demand = reservation_policy.demand\n\n"
        "    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)\n",
        "idle posterior initialization",
    )
    source = _replace_once(
        source,
        '    node_index = getattr(oracle, "_fast_node_index", None)\n',
        "    idle_planner = IdleFleetPlanner(\n"
        "        graph, charger_index, idle_demand, station_nodes,\n"
        "        DEFAULT_CHARGING_POWER_W,\n"
        "        IdlePlanningConfig(\n"
        "            horizon_min=args.idle_horizon_min,\n"
        "            max_demand_clusters=args.idle_max_demand_clusters,\n"
        "            candidate_clusters=args.idle_candidate_clusters,\n"
        "            candidate_chargers=args.idle_candidate_chargers,\n"
        "            max_actions_per_epoch=args.idle_max_actions_per_epoch,\n"
        "            minimum_gain_fraction=args.idle_minimum_gain_fraction,\n"
        "            relocation_uncertainty_penalty=args.idle_relocation_uncertainty_penalty,\n"
        "            processes=args.idle_processes,\n"
        "        ),\n"
        "    )\n"
        '    node_index = getattr(oracle, "_fast_node_index", None)\n',
        "idle planner initialization",
    )
    source = _replace_once(
        source,
        "    station_state = {node: base.StationState() for node in station_nodes}\n",
        "    station_state = {node: base.StationState() for node in station_nodes}\n"
        "    idle_intents = {}  # robot id -> chosen background travel intent\n"
        "    idle_stats = Counter()\n",
        "idle runtime state",
    )

    source = _replace_once(
        source,
        "    service_token = {robot.spec.id: 0 for robot in robots}\n",
        "    service_token = {robot.spec.id: 0 for robot in robots}\n"
        "    idle_service_finish_min = {}\n",
        "track unfinished handling",
    )
    source = _replace_once(
        source,
        '        push_event(now + duration, 0, "service_complete", (robot_id, token))\n',
        '        idle_service_finish_min[robot_id] = now + duration\n'
        '        push_event(now + duration, 0, "service_complete", (robot_id, token))\n',
        "record handling completion time",
    )
    source = _replace_once(
        source,
        "            _, stop = service_lock.pop(robot_id)\n",
        "            _, stop = service_lock.pop(robot_id)\n"
        "            idle_service_finish_min.pop(robot_id, None)\n",
        "remove completed handling time",
    )

    helper_marker = "    def begin_background_return(robot: RobotState, now: float) -> None:\n"
    helpers = '''    def build_charging_calendar(now: float) -> ChargingCalendar:
        """Reserve active, FIFO queued, then in-transit background charging."""
        calendar = ChargingCalendar(
            {station: DEFAULT_NUMBER_OF_PORTS for station in station_nodes}
        )
        for station, state in station_state.items():
            for session in state.active.values():
                finish = session.start_time_min + (
                    session.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0
                )
                remaining = max(1e-8, finish - now)
                calendar.reserve(session.robot_id, station, now, remaining)
            for request in state.queue:
                calendar.reserve(
                    request.robot_id, station, now,
                    max(1e-8, request.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0),
                )
        planned = sorted(
            (intent for intent in idle_intents.values()
             if intent["kind"] == "charge"),
            key=lambda intent: (intent["arrival_min"], intent["robot_id"]),
        )
        for intent in planned:
            robot = robot_by_id[intent["robot_id"]]
            energy = robot.spec.battery_capacity_wh - intent["battery_arrival_wh"]
            intent["slot"] = calendar.reserve(
                intent["robot_id"], intent["target_node"],
                max(now, intent["arrival_min"]),
                max(1e-8, energy / DEFAULT_CHARGING_POWER_W * 60.0),
            )
        return calendar

    def start_idle_action(robot: RobotState, action, now: float) -> None:
        """Commit one exact graph path; delivery may interrupt at a node."""
        robot_id = robot.spec.id
        target = int(action.target_node)
        if action.kind == "charge" and target == int(robot.node_id):
            request_charge(
                robot_id, target,
                robot.spec.battery_capacity_wh - robot.battery_wh,
                now, "background",
            )
            idle_stats["charge_actions"] += 1
            return
        path = charger_index.path(int(robot.node_id), target)
        exact_m = path_distance(tuple(int(node) for node in path))
        battery_after = robot.battery_wh - exact_m * robot.spec.energy_per_meter_wh
        if action.kind == "reposition":
            reserve_m = min(charger_index.distances_to_stations(target).values())
            if battery_after + BATTERY_EPS_WH < (
                reserve_m * robot.spec.energy_per_meter_wh
            ):
                robot.available = True
                idle_stats["stay_decisions"] += 1
                return
        elif battery_after < -BATTERY_EPS_WH:
            robot.available = True
            idle_stats["stay_decisions"] += 1
            return
        robot.activity = RobotActivity.IDLE
        robot.available = False
        robot.set_planned_path(path)
        idle_intents[robot_id] = {
            "robot_id": robot_id,
            "kind": action.kind,
            "target_node": target,
            "arrival_min": now + exact_m / robot.spec.speed_mps / 60.0,
            "battery_arrival_wh": max(0.0, battery_after),
            "slot": action.slot,
        }
        event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
        if event is None:
            raise RuntimeError("idle travel has no graph edge")
        push_event(event.time_min, 1, "background_node_arrival", event)
        idle_stats[f"{action.kind}_actions"] += 1

    @lru_cache(maxsize=65536)
    def idle_haversine_distance(source_node: int, target_node: int) -> float:
        return haversine_node_distance_m(graph, source_node, target_node)

    def busy_readiness_steps(robot: RobotState):
        """Read cached remaining phases; do not rerun battery route search."""
        robot_id = robot.spec.id
        schedule = schedules[robot_id]
        prepared = {step.stop: step for step in schedule.prepared_steps}
        steps = []
        for index, stop in enumerate(schedule.stops):
            if index == 0 and robot_id in service_lock:
                # Its exact remaining service time is in the initial snapshot.
                continue
            if index == 0 and phase_version[robot_id] == schedule.version:
                if robot.is_moving:
                    target = (robot.remaining_route[-1]
                              if robot.remaining_route else robot.decision_node)
                    steps.append(ReadinessStep("travel", int(target)))
                phases = tuple(active_phases[robot_id])
            else:
                cached = prepared.get(stop)
                if cached is None:
                    # No fabricated duration or charging: mark unavailable if
                    # the cached battery-feasible itinerary is incomplete.
                    return None
                phases = cached.phases
            for phase in phases:
                steps.append(ReadinessStep(
                    phase.kind, int(phase.target_node), float(phase.energy_wh),
                ))
            steps.append(ReadinessStep("travel", int(stop.node_id)))
            steps.append(ReadinessStep(stop.kind, int(stop.node_id)))
        return steps

    def plan_idle_fleet(now: float, only_ids: set[int] | None = None) -> None:
        started = time.perf_counter()
        calendar = build_charging_calendar(now)
        snapshots = []
        candidate_ids = set()
        for robot in robots:
            robot_id = robot.spec.id
            intent = idle_intents.get(robot_id)
            if intent is not None:
                node = intent["target_node"]
                battery = intent["battery_arrival_wh"]
                ready = max(0.0, intent["arrival_min"] - now)
                if intent["kind"] == "charge":
                    battery = robot.spec.battery_capacity_wh
                    ready = max(0.0, intent["slot"].finish_min - now)
            elif robot.activity is RobotActivity.CHARGING:
                found = find_active_charge(robot_id)
                node = int(robot.node_id)
                if found is None:
                    raise RuntimeError("charging robot has no active session")
                _, _, session = found
                battery = min(
                    robot.spec.battery_capacity_wh,
                    session.start_battery_wh + session.energy_wh,
                )
                ready = max(
                    0.0,
                    session.start_time_min
                    + session.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0
                    - now,
                )
            elif robot.activity is RobotActivity.WAITING:
                node = int(robot.node_id)
                booked = next(
                    (slot for slot in calendar.slots(node)
                     if slot.robot_id == robot_id), None
                )
                if booked is None:
                    raise RuntimeError("waiting robot has no projected charge slot")
                queued = next(
                    (request for request in station_state[node].queue
                     if request.robot_id == robot_id), None
                )
                if queued is None:
                    raise RuntimeError("waiting robot has no charging request")
                battery = min(
                    robot.spec.battery_capacity_wh,
                    robot.battery_wh + queued.energy_wh,
                )
                ready = max(0.0, booked.finish_min - now)
            else:
                decision = decision_snapshot(robot, now)
                node = int(decision.node_id) if decision else int(robot.node_id)
                battery = float(decision.battery_wh) if decision else float(robot.battery_wh)
                ready = max(0.0, decision.time_min - now) if decision else 0.0
            if schedules[robot_id].orders:
                if robot_id in service_lock:
                    ready = max(0.0, idle_service_finish_min[robot_id] - now)
                estimate_start = IdleRobotSnapshot(
                    robot_id, node, robot.spec.speed_mps,
                    robot.spec.energy_per_meter_wh, battery,
                    robot.spec.battery_capacity_wh, robot.spec.max_payload_kg,
                    robot.spec.max_volume_l, available_in_min=ready,
                )
                steps = busy_readiness_steps(robot)
                if steps is None:
                    node = int(schedules[robot_id].stops[-1].node_id)
                    battery, ready = 0.0, math.inf
                    idle_stats["readiness_missing_plan"] += 1
                else:
                    projection = project_remaining_readiness(
                        estimate_start, now, steps, calendar,
                        distance_m=idle_haversine_distance,
                        charger_power_w=DEFAULT_CHARGING_POWER_W,
                        pickup_handling_min=PICKUP_HANDLING_MIN,
                        dropoff_handling_min=DROPOFF_HANDLING_MIN,
                    )
                    node = int(projection.node_id)
                    battery = projection.battery_wh
                    ready = max(0.0, projection.finish_min - now)
                    idle_stats["readiness_projections"] += 1
                    idle_stats["readiness_queue_min"] += projection.queue_min
                    if not math.isfinite(ready):
                        idle_stats["readiness_infeasible_plan"] += 1
            elif robot_id in service_lock:
                ready = max(0.0, idle_service_finish_min[robot_id] - now)
            elif (intent is None and robot.activity is RobotActivity.IDLE
                  and not robot.remaining_route and not pending
                  and (only_ids is None or robot_id in only_ids)):
                candidate_ids.add(robot_id)
            snapshots.append(IdleRobotSnapshot(
                robot_id=robot_id,
                node_id=node,
                speed_mps=robot.spec.speed_mps,
                energy_per_meter_wh=robot.spec.energy_per_meter_wh,
                battery_wh=battery,
                battery_capacity_wh=robot.spec.battery_capacity_wh,
                max_payload_kg=robot.spec.max_payload_kg,
                max_volume_l=robot.spec.max_volume_l,
                available_in_min=ready,
                min_importance=(
                    reservation_assignment.threshold_by_robot_id[robot_id]
                    if reservation_assignment is not None else 0.0
                ),
            ))
        if candidate_ids:
            chosen = idle_planner.plan(now, snapshots, candidate_ids, calendar)
            for robot_id in sorted(candidate_ids):
                action = chosen.get(robot_id)
                if action is None:
                    robot_by_id[robot_id].available = True
                    idle_stats["stay_decisions"] += 1
                else:
                    start_idle_action(robot_by_id[robot_id], action, now)
            idle_stats["planning_epochs"] += 1
        idle_stats["planning_seconds"] += time.perf_counter() - started

'''
    source = _replace_once(
        source, helper_marker, helpers + helper_marker, "idle runtime helpers"
    )
    source = _replace_once(
        source,
        '''    def begin_background_return(robot: RobotState, now: float) -> None:
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

''',
        '''    def begin_background_return(robot: RobotState, now: float) -> None:
        robot_id = robot.spec.id
        if schedules[robot_id].orders or robot_id in service_lock:
            return
        robot.current_order_id = None
        robot.activity = RobotActivity.IDLE
        robot.available = True
        plan_idle_fleet(now, {robot_id})

''',
        "idle policy after delivery",
    )
    source = _replace_once(
        source,
        '''        cancel_editable_route_for_replan(robot, now, was_busy)
        robot.available = False
''',
        '''        idle_intents.pop(robot.spec.id, None)
        cancel_editable_route_for_replan(robot, now, was_busy)
        robot.available = False
''',
        "cancel idle intent on delivery",
    )
    source = _replace_once(
        source,
        '''            if reservation_policy is not None:
                reservation_policy.observe(
                    order.pickup_node, order.importance, now
                )
                refresh_reservations(now)
            dispatch_pending(now)
''',
        '''            if reservation_policy is not None:
                reservation_policy.observe(
                    order.pickup_node, order.importance, now
                )
                refresh_reservations(now)
            else:
                idle_demand.observe(order.pickup_node, order.importance, now)
            dispatch_pending(now)
''',
        "online idle posterior",
    )
    source = _replace_once(
        source,
        '''        elif kind in {"background_node_arrival", "delivery_node_arrival"}:
''',
        '''        elif kind == "idle_replan":
            plan_idle_fleet(now)

        elif kind in {"background_node_arrival", "delivery_node_arrival"}:
''',
        "idle replan event",
    )
    source = _replace_once(
        source,
        '''            else:
                station = int(robot.node_id)
                if station not in router.station_set:
                    raise RuntimeError("background route ended at non-charger")
                energy = robot.spec.battery_capacity_wh - robot.battery_wh
                request_charge(robot_id, station, energy, now, "background")
                if pending:
                    dispatch_pending(now)
''',
        '''            else:
                intent = idle_intents.pop(robot_id, None)
                if intent is None:
                    raise RuntimeError("background route ended without idle intent")
                if intent["kind"] == "charge":
                    station = int(robot.node_id)
                    if station not in router.station_set:
                        raise RuntimeError("idle charging route ended at non-charger")
                    energy = robot.spec.battery_capacity_wh - robot.battery_wh
                    request_charge(robot_id, station, energy, now, "background")
                else:
                    robot.activity = RobotActivity.IDLE
                    robot.available = True
                if pending:
                    dispatch_pending(now)
''',
        "idle travel arrival",
    )
    source = _replace_once(
        source,
        '''            else:
                robot.available = True
                if pending:
                    dispatch_pending(now)

        elif kind == "service_complete":
''',
        '''            else:
                robot.available = True
                if pending:
                    dispatch_pending(now)
                if not schedules[robot_id].orders:
                    plan_idle_fleet(now, {robot_id})

        elif kind == "service_complete":
''',
        "charge completion idle planning",
    )
    source = _replace_once(
        source,
        '''    while events:
''',
        '''    if reservation_policy is not None:
        refresh_reservations(0.0)
    tick = 0.0
    while tick < scenario.duration_minutes:
        push_event(tick, 4, "idle_replan", None)
        tick += args.idle_replan_interval_min

    while events:
''',
        "bounded periodic replanning",
    )
    source = _replace_once(
        source,
        '''    if (
        pending
        or any(schedule.orders for schedule in schedules.values())
''',
        '''    idle_planner.close()
    if (
        pending
        or any(schedule.orders for schedule in schedules.values())
''',
        "close process pool",
    )
    source = _replace_once(
        source,
        '        "wall_clock_seconds": time.perf_counter() - wall_start,\n',
        '        "idle_policy": ("posterior_uncertainty_gated"\n'
        '                        if args.idle_relocation_uncertainty_penalty > 0\n'
        '                        else "posterior_marginal_utility"),\n'
        '        "idle_processes": int(args.idle_processes),\n'
        '        "idle_max_actions_per_epoch": int(args.idle_max_actions_per_epoch),\n'
        '        "idle_minimum_gain_fraction": float(args.idle_minimum_gain_fraction),\n'
        '        "idle_relocation_uncertainty_penalty": float(args.idle_relocation_uncertainty_penalty),\n'
        '        "idle_horizon_min": float(args.idle_horizon_min),\n'
        '        "idle_reposition_actions": int(idle_stats["reposition_actions"]),\n'
        '        "idle_charge_actions": int(idle_stats["charge_actions"]),\n'
        '        "idle_stay_decisions": int(idle_stats["stay_decisions"]),\n'
        '        "idle_planning_epochs": int(idle_stats["planning_epochs"]),\n'
        '        "idle_planning_seconds": float(idle_stats["planning_seconds"]),\n'
        '        "idle_readiness_estimator": "haversine_route_handling_charge_queue_v1",\n'
        '        "idle_readiness_projections": int(idle_stats["readiness_projections"]),\n'
        '        "idle_readiness_missing_plan": int(idle_stats["readiness_missing_plan"]),\n'
        '        "idle_readiness_infeasible_plan": int(idle_stats["readiness_infeasible_plan"]),\n'
        '        "idle_readiness_queue_min": float(idle_stats["readiness_queue_min"]),\n'
        '        "wall_clock_seconds": time.perf_counter() - wall_start,\n',
        "idle result metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = heuristic.corrected._patch_source(
        heuristic.corrected.TARGET.read_text(encoding="utf-8")
    )
    source = heuristic._patch_heuristic(source)
    source = _patch_idle(source)
    name = "nyc_anticipatory_idle_policy_target"
    module = types.ModuleType(name)
    module.__file__ = str(heuristic.corrected.TARGET)
    module.__package__ = None
    sys.modules[name] = module
    exec(compile(source, str(heuristic.corrected.TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    _load_namespace()["main"]()


if __name__ == "__main__":
    main()
