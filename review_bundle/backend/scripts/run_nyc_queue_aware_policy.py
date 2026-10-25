from __future__ import annotations

"""Opt-in queue-aware version of the full anticipatory idle policy.

The legacy dispatcher/idle runners are untouched. This version includes known
FIFO charging waits in affected-order loss, compares bounded charger detours,
and records observed mission/background waits separately, including cancellations.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_nyc_anticipatory_idle_policy as idle


def _patch_queue(source: str) -> str:
    replace = idle._replace_once
    source = replace(source, "from functools import lru_cache\n",
                     "from functools import lru_cache\n"
                     "from delivery_fleet.queue_forecasting import (\n"
                     "    ActiveChargingForecast, KnownChargingRequest, KnownChargingForecast,\n"
                     ")\n", "queue forecast imports")
    source = replace(source, "    args = parser.parse_args()\n",
                     "    parser.add_argument('--queue-charger-alternatives', type=int, default=3)\n"
                     "    parser.add_argument('--queue-charger-scan-limit', type=int, default=12)\n"
                     "    args = parser.parse_args()\n"
                     "    if args.queue_charger_alternatives < 0 or args.queue_charger_scan_limit <= 0:\n"
                     "        parser.error('queue charger limits must be non-negative/positive')\n",
                     "queue forecast CLI")
    source = replace(source,
                     "    new_completion_time_min: float\n\n\n@dataclass(frozen=True, slots=True)\nclass InsertionChoice:",
                     "    new_completion_time_min: float\n"
                     "    queue_predictions: tuple = ()\n\n\n@dataclass(frozen=True, slots=True)\nclass InsertionChoice:",
                     "retain selected route predictions")
    source = replace(source, "    idle_stats = Counter()\n",
                     "    idle_stats = Counter()\n"
                     "    assignment_queue_forecast = None\n"
                     "    queue_stats = Counter()\n"
                     "    queue_predictions_by_robot = {}\n"
                     "    queue_request_predictions = {}\n"
                     "    queue_observed_waits = {'delivery': [], 'background': []}\n"
                     "    queue_log_path = args.output.with_suffix('.queue.jsonl')\n"
                     "    args.output.parent.mkdir(parents=True, exist_ok=True)\n"
                     "    queue_log_path.write_text('', encoding='utf-8')\n",
                     "queue-aware runtime state")
    helpers = '''    def record_queue_wait(request, station, now, outcome):
        wait = max(0.0, now - request.arrival_time_min)
        purpose = request.purpose
        queue_observed_waits[purpose].append(wait)
        queue_stats[f"{purpose}_queue_{outcome}_count"] += 1
        queue_stats[f"{purpose}_queue_{outcome}_total_min"] += wait
        prediction = queue_request_predictions.pop(
            (request.robot_id, request.arrival_time_min), None
        )
        if purpose == "delivery" and outcome == "started" and prediction is not None:
            queue_stats["prediction_samples"] += 1
            queue_stats["prediction_absolute_error_min"] += abs(wait - prediction.wait_min)
            queue_stats["prediction_signed_error_min"] += wait - prediction.wait_min
        payload = {
            "robot_id": request.robot_id, "station_node": int(station),
            "purpose": purpose, "outcome": outcome,
            "arrival_min": request.arrival_time_min, "observed_at_min": now,
            "wait_min": wait,
            "predicted_wait_min": prediction.wait_min if prediction else None,
            "predicted_arrival_min": prediction.arrival_min if prediction else None,
        }
        with queue_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\\n")

    def queue_remaining_phases(robot):
        """Cached exact distances; count only the unfinished committed edge."""
        rid = robot.spec.id
        schedule = schedules[rid]
        prepared = {step.stop: step for step in schedule.prepared_steps}
        result = []
        for index, stop in enumerate(schedule.stops):
            if index == 0 and rid in service_lock:
                continue
            if index == 0 and phase_version[rid] == schedule.version:
                if robot.is_moving:
                    path = (int(robot.decision_node), *map(int, robot.remaining_route))
                    if len(path) > 1:
                        result.append(RoutePhase("travel", path[-1], distance_m=path_distance(path)))
                phases = tuple(active_phases[rid])
            else:
                cached = prepared.get(stop)
                if cached is None:
                    queue_stats["missing_cached_plans"] += 1
                    return None
                phases = cached.phases
            result.extend(phases)
            result.append((stop.kind, int(stop.node_id)))
        return result

    def build_assignment_queue_forecast(now):
        active, requests = [], []
        current_finish, current_energy = {}, {}
        sequence = itertools.count()
        for station, state in station_state.items():
            free = [now] * (DEFAULT_NUMBER_OF_PORTS - len(state.active))
            for session in state.active.values():
                finish = max(now, session.start_time_min + session.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0)
                active.append(ActiveChargingForecast(session.robot_id, station, finish))
                free.append(finish)
                current_finish[session.robot_id] = finish
                current_energy[session.robot_id] = min(
                    robot_by_id[session.robot_id].spec.battery_capacity_wh,
                    session.start_battery_wh + session.energy_wh,
                )
            heapq.heapify(free)
            for request in state.queue:
                duration = request.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0
                finish = max(now, heapq.heappop(free)) + duration
                heapq.heappush(free, finish)
                current_finish[request.robot_id] = finish
                current_energy[request.robot_id] = min(
                    robot_by_id[request.robot_id].spec.battery_capacity_wh,
                    robot_by_id[request.robot_id].battery_wh + request.energy_wh,
                )
                requests.append(KnownChargingRequest(
                    request.robot_id, station, request.arrival_time_min, duration, next(sequence)
                ))
        ports = {station: DEFAULT_NUMBER_OF_PORTS for station in station_nodes}
        observed = KnownChargingForecast(now, ports, active, requests)
        for rid, intent in idle_intents.items():
            if intent["kind"] == "charge":
                energy = robot_by_id[rid].spec.battery_capacity_wh - intent["battery_arrival_wh"]
                if energy > 1e-8:
                    requests.append(KnownChargingRequest(
                        rid, intent["target_node"], max(now, intent["arrival_min"]),
                        energy / DEFAULT_CHARGING_POWER_W * 60.0, next(sequence),
                    ))
        for robot in robots:
            rid = robot.spec.id
            if not schedules[rid].orders:
                continue
            phases = queue_remaining_phases(robot)
            if phases is None:
                continue
            if rid in current_finish:
                time_min = current_finish[rid]
                battery = current_energy[rid]
            elif rid in service_lock:
                time_min = max(now, idle_service_finish_min[rid])
                battery = robot.battery_wh
            else:
                snapshot = decision_snapshot(robot, now)
                time_min, battery = snapshot.time_min, snapshot.battery_wh
            projection = observed.project(rid)
            future = []
            for phase in phases:
                if isinstance(phase, tuple):
                    time_min += PICKUP_HANDLING_MIN if phase[0] == "pickup" else DROPOFF_HANDLING_MIN
                elif phase.kind == "travel":
                    time_min += phase.distance_m / robot.spec.speed_mps / 60.0
                    battery -= phase.distance_m * robot.spec.energy_per_meter_wh
                    if battery < -BATTERY_EPS_WH:
                        queue_stats["infeasible_cached_plans"] += 1
                        future = []
                        break
                else:
                    energy = min(max(0.0, phase.energy_wh), robot.spec.battery_capacity_wh - max(0.0, battery))
                    if energy > 1e-8:
                        duration = energy / DEFAULT_CHARGING_POWER_W * 60.0
                        future.append(KnownChargingRequest(rid, phase.target_node, time_min, duration, next(sequence)))
                        time_min = projection.serve(phase.target_node, time_min, duration).finish_min
                        battery += energy
            requests.extend(future)
        queue_stats["forecast_rebuilds"] += 1
        queue_stats["known_requests_forecast"] += len(requests)
        return KnownChargingForecast(now, ports, active, requests)

    def project_prefix_queue(robot, prefix, start_min, projection):
        copy = projection.clone()
        time_min = start_min
        predictions = []
        for phase in prefix.phases:
            if phase.kind == "travel":
                time_min += phase.distance_m / robot.spec.speed_mps / 60.0
            elif phase.energy_wh > 1e-8:
                prediction = copy.serve(phase.target_node, time_min, phase.energy_wh / DEFAULT_CHARGING_POWER_W * 60.0)
                predictions.append(prediction)
                time_min = prediction.finish_min
        return time_min, copy, tuple(predictions)

    def queue_aware_prefix(robot, snapshot_robot, prefix, target, lookahead, start_min, projection, special_rows):
        best_time, best_projection, best_predictions = project_prefix_queue(robot, prefix, start_min, projection)
        best_prefix = prefix
        queue_stats["prefix_evaluations"] += 1
        queue_stats["positive_wait_prefixes"] += int(any(p.wait_min > EPS for p in best_predictions))
        if not best_predictions or not any(p.wait_min > EPS for p in best_predictions) or not args.queue_charger_alternatives:
            return best_prefix, best_projection, best_predictions
        source = int(snapshot_robot.node_id)
        nearest = sorted(
            charger_index.distances_to_stations(source, cutoff_m=snapshot_robot.remaining_range_m).items(),
            key=lambda item: (item[1], str(item[0])),
        )[:args.queue_charger_scan_limit]
        options = []
        for station, distance in nearest:
            if station == source or station == target:
                continue
            eta = start_min + distance / robot.spec.speed_mps / 60.0
            options.append((eta + projection.wait_at(station, eta), float(distance), int(station)))
        for _, distance, station in sorted(options)[:args.queue_charger_alternatives]:
            battery = snapshot_robot.battery_wh - distance * robot.spec.energy_per_meter_wh
            if battery < -BATTERY_EPS_WH:
                continue
            at_station = RobotState(spec=robot.spec, node_id=station, battery_wh=max(0.0, battery), available=False)
            try:
                distance_to_target = exact_distance(station, target, special_rows)
                if lookahead is None:
                    tail = plan_final_target(at_station, target, start_to_target_m=distance_to_target)
                else:
                    quote = evaluate_pair_from_committed_state(
                        router, at_station, target, lookahead,
                        start_to_pickup_m=distance_to_target,
                        pickup_to_dropoff_m=exact_distance(target, lookahead, special_rows),
                    )
                    tail = prefix_from_pair_quote(at_station, quote, target)
            except NoFeasibleBatteryRoute:
                continue
            candidate = PrefixPlan(
                (RoutePhase("travel", station, distance_m=distance), *tail.phases),
                distance / robot.spec.speed_mps / 60.0 + tail.total_time_min,
                distance + tail.total_distance_m, tail.arrival_battery_wh,
            )
            queue_stats["alternative_prefix_evaluations"] += 1
            end, copy, predictions = project_prefix_queue(robot, candidate, start_min, projection)
            # Preserve downstream feasibility: do not trade away arrival battery.
            if (candidate.arrival_battery_wh + BATTERY_EPS_WH >= prefix.arrival_battery_wh
                    and (end, candidate.total_distance_m) < (best_time, best_prefix.total_distance_m)):
                best_time, best_projection, best_predictions = end, copy, predictions
                best_prefix = candidate
        if best_prefix is not prefix:
            queue_stats["alternative_prefix_selected"] += 1
        return best_prefix, best_projection, best_predictions

'''
    source = replace(source, "    def evaluate_sequence(\n", helpers + "    def evaluate_sequence(\n",
                     "queue forecast and accounting helpers")
    source = replace(source, "            completion: dict[int, float] = {}\n",
                     "            completion: dict[int, float] = {}\n"
                     "            queue_projection = assignment_queue_forecast.project(robot.spec.id)\n"
                     "            queue_predictions = []\n", "private candidate queue projection")
    source = replace(source, "                steps.append(\n                    StepPlan(\n",
                     "                prefix_wait = 0.0\n"
                     "                if prefix.phases:\n"
                     "                    snapshot_robot = RobotState(spec=robot.spec, node_id=node, battery_wh=battery, available=False)\n"
                     "                    prefix, queue_projection, predictions = queue_aware_prefix(\n"
                     "                        robot, snapshot_robot, prefix, int(stop.node_id),\n"
                     "                        first_distinct_later_node(stops, i), now, queue_projection, special_rows,\n"
                     "                    )\n"
                     "                    prefix_wait = sum(p.wait_min for p in predictions)\n"
                     "                    queue_predictions.extend(predictions)\n"
                     "                steps.append(\n                    StepPlan(\n", "queue-aware prefix choice")
    source = replace(source, "                        route_time_min=prefix.total_time_min,\n",
                     "                        route_time_min=prefix.total_time_min + prefix_wait,\n", "queue-inclusive planned step time")
    source = replace(source, "                now += prefix.total_time_min\n",
                     "                now += prefix.total_time_min + prefix_wait\n", "propagate waits to all deliveries")
    source = replace(source, "                total_distance_m=float(total_distance),\n                projected_loss=float(projected_loss),\n",
                     "                total_distance_m=float(total_distance),\n                projected_loss=float(projected_loss),\n"
                     "                queue_predictions=tuple(queue_predictions),\n", "save candidate predictions")
    source = replace(source, "    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:\n",
                     "    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:\n"
                     "        nonlocal assignment_queue_forecast\n"
                     "        assignment_queue_forecast = build_assignment_queue_forecast(now)\n"
                     "        baseline_projection_cache.clear()  # live queues invalidate old baseline scores\n",
                     "refresh live queue forecast before selection")
    source = replace(source, "        schedule.prepared_steps = list(choice.evaluation.steps)\n",
                     "        schedule.prepared_steps = list(choice.evaluation.steps)\n"
                     "        queue_predictions_by_robot[robot.spec.id] = list(choice.evaluation.queue_predictions)\n",
                     "commit only the selected predictions")
    source = replace(source, "        robot = robot_by_id[request.robot_id]\n        token = next_charge_token(request.robot_id)\n",
                     "        robot = robot_by_id[request.robot_id]\n"
                     "        record_queue_wait(request, station_node, now, 'started')\n"
                     "        token = next_charge_token(request.robot_id)\n", "measure observed wait excluding charging")
    source = replace(source, "        state = station_state[station_node]\n        if len(state.active) < DEFAULT_NUMBER_OF_PORTS:\n",
                     "        plans = queue_predictions_by_robot.get(robot_id, [])\n"
                     "        if purpose == 'delivery' and plans and int(plans[0].station_node) == station_node:\n"
                     "            queue_request_predictions[(robot_id, float(now))] = plans.pop(0)\n"
                     "        state = station_state[station_node]\n        if len(state.active) < DEFAULT_NUMBER_OF_PORTS:\n",
                     "attach forecast at actual charger arrival")
    source = replace(source, "    def cancel_stationary_wait(robot: RobotState) -> None:\n",
                     "    def cancel_stationary_wait(robot: RobotState, now: float) -> None:\n", "timestamp canceled wait")
    source = replace(source, "        _, request = removed\n        robot.activity = RobotActivity.IDLE\n",
                     "        station, request = removed\n"
                     "        record_queue_wait(request, station, now, 'cancelled')\n"
                     "        robot.activity = RobotActivity.IDLE\n", "measure canceled wait")
    source = replace(source, "            cancel_stationary_wait(robot)\n",
                     "            cancel_stationary_wait(robot, now)\n", "pass cancellation timestamp")
    source = replace(source, '        "wall_clock_seconds": time.perf_counter() - wall_start,\n',
                     '        "assignment_queue_forecast": "known_traffic_fifo_v1",\n'
                     '        "policy_version": "queue_aware_v1",\n'
                     '        "queue_charger_alternatives": args.queue_charger_alternatives,\n'
                     '        "queue_forecast_rebuilds": int(queue_stats["forecast_rebuilds"]),\n'
                     '        "queue_forecast_missing_cached_plans": int(queue_stats["missing_cached_plans"]),\n'
                     '        "queue_forecast_infeasible_cached_plans": int(queue_stats["infeasible_cached_plans"]),\n'
                     '        "queue_positive_wait_prefixes": int(queue_stats["positive_wait_prefixes"]),\n'
                     '        "queue_alternative_prefix_evaluations": int(queue_stats["alternative_prefix_evaluations"]),\n'
                     '        "queue_alternative_prefix_selected": int(queue_stats["alternative_prefix_selected"]),\n'
                     '        "queue_wait_records_file": str(queue_log_path),\n'
                     '        "delivery_queue_wait_total_min": float(sum(queue_observed_waits["delivery"])),\n'
                     '        "delivery_queue_visit_count": len(queue_observed_waits["delivery"]),\n'
                     '        "delivery_queue_waited_count": sum(w > EPS for w in queue_observed_waits["delivery"]),\n'
                     '        "delivery_queue_wait_mean_min": float(np.mean(queue_observed_waits["delivery"])) if queue_observed_waits["delivery"] else 0.0,\n'
                     '        "delivery_queue_wait_mean_if_waited_min": float(np.mean([w for w in queue_observed_waits["delivery"] if w > EPS])) if any(w > EPS for w in queue_observed_waits["delivery"]) else 0.0,\n'
                     '        "delivery_queue_wait_p95_min": _percentile(queue_observed_waits["delivery"], 95),\n'
                     '        "delivery_queue_wait_max_min": max(queue_observed_waits["delivery"], default=0.0),\n'
                     '        "delivery_queue_cancelled_count": int(queue_stats["delivery_queue_cancelled_count"]),\n'
                     '        "delivery_queue_cancelled_wait_total_min": float(queue_stats["delivery_queue_cancelled_total_min"]),\n'
                     '        "background_queue_wait_total_min": float(sum(queue_observed_waits["background"])),\n'
                     '        "delivery_queue_prediction_samples": int(queue_stats["prediction_samples"]),\n'
                     '        "delivery_queue_prediction_mae_min": queue_stats["prediction_absolute_error_min"] / max(1, queue_stats["prediction_samples"]),\n'
                     '        "delivery_queue_prediction_bias_min": queue_stats["prediction_signed_error_min"] / max(1, queue_stats["prediction_samples"]),\n'
                     '        "wall_clock_seconds": time.perf_counter() - wall_start,\n', "queue forecast result metadata")
    return source


def _load_namespace() -> dict[str, object]:
    source = idle.heuristic.corrected._patch_source(idle.heuristic.corrected.TARGET.read_text(encoding="utf-8"))
    source = idle.heuristic._patch_heuristic(source)
    source = idle._patch_idle(source)
    source = _patch_queue(source)
    module = types.ModuleType("nyc_queue_aware_policy_target")
    module.__file__ = str(idle.heuristic.corrected.TARGET)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    _load_namespace()["main"]()


if __name__ == "__main__":
    main()
