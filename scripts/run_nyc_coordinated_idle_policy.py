from __future__ import annotations

"""Versioned, assignment-triggered shared idle planning on queue-aware dispatch.

Parallel workers score immutable candidates. One coordinator updates shared
coverage and port bookings after every selection. No independent robot bidding,
hidden future orders, or interruption of background charging is introduced.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_nyc_queue_aware_policy as queue


def _patch_coordinated(source: str) -> str:
    replace = queue.idle._replace_once
    source = replace(source, "    ReadinessStep, project_remaining_readiness,\n)\n",
                     "    ReadinessStep, project_remaining_readiness, IdleAction,\n)\n"
                     "from delivery_fleet.coordinated_idle import (\n"
                     "    CoordinatedIdleFleetPlanner as IdleFleetPlanner,\n)\n",
                     "coordinated planner import")
    source = replace(source, "    args = parser.parse_args()\n",
                     "    parser.add_argument('--idle-switch-gain-fraction', type=float, default=0.005,\n"
                     "                        help='Extra required gain as a fraction of total demand weight for changing an existing relocation.')\n"
                     "    parser.add_argument('--idle-retarget-cooldown-min', type=float, default=2.0)\n"
                     "    args = parser.parse_args()\n"
                     "    if any(not math.isfinite(v) or v < 0 for v in (args.idle_switch_gain_fraction, args.idle_retarget_cooldown_min)):\n"
                     "        parser.error('idle switch fraction and cooldown must be finite and non-negative')\n",
                     "coordinated CLI")
    source = replace(source, "    idle_stats = Counter()\n",
                     "    idle_stats = Counter()\n"
                     "    coordinated_pending_times = set()\n"
                     "    idle_last_target_change = {}\n"
                     "    idle_decisions_path = args.output.with_suffix('.idle.jsonl')\n"
                     "    args.output.parent.mkdir(parents=True, exist_ok=True)\n"
                     "    idle_decisions_path.write_text('', encoding='utf-8')\n",
                     "coordinated runtime state")
    helpers = '''    def request_coordinated_idle_replan(now: float) -> None:
        # Order/service/node events at one timestamp settle first. Every real
        # assignment requests a refresh; a burst gets one shared fleet decision.
        idle_stats["assignment_refresh_requests"] += 1
        if now not in coordinated_pending_times:
            coordinated_pending_times.add(now)
            push_event(now, 4, "coordinated_idle_replan", None)

    def start_idle_action(robot: RobotState, action, now: float) -> None:
        rid = robot.spec.id
        if (schedules[rid].orders or rid in service_lock
                or robot.activity in {RobotActivity.CHARGING, RobotActivity.WAITING}):
            raise RuntimeError("idle relocation cannot interrupt delivery or charging")
        previous = idle_intents.get(rid)
        if previous is not None and previous["kind"] == "charge":
            raise RuntimeError("idle relocation cannot cancel a committed charging trip")
        snapshot = decision_snapshot(robot, now)
        target = int(action.target_node)
        path = tuple(int(node) for node in charger_index.path(int(snapshot.node_id), target))
        exact_m = path_distance(path)
        battery_after = snapshot.battery_wh - exact_m * robot.spec.energy_per_meter_wh
        if action.kind != "charge":
            reserve_m = min(charger_index.distances_to_stations(target).values())
            if battery_after + BATTERY_EPS_WH < reserve_m * robot.spec.energy_per_meter_wh:
                idle_stats["commit_rejected"] += 1
                return
        elif battery_after < -BATTERY_EPS_WH:
            idle_stats["commit_rejected"] += 1
            return
        moving = robot.is_moving
        robot.replan_from_decision_node(path)
        idle_last_target_change[rid] = now
        if previous is not None:
            idle_stats["retarget_actions"] += 1
            if moving:
                idle_stats["retarget_preserved_edges"] += 1
        record = {
            "time_min": now, "robot_id": rid, "kind": action.kind,
            "previous_target": previous["target_node"] if previous else None,
            "target_node": target, "decision_node": int(snapshot.node_id),
            "decision_time_min": snapshot.time_min,
            "arrival_min": snapshot.time_min + exact_m / robot.spec.speed_mps / 60.0,
            "ready_in_min": action.ready_min,
            "battery_arrival_wh": max(0.0, battery_after),
            "committed_edge_preserved": moving,
        }
        with idle_decisions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\\n")
        if not moving and len(path) == 1:
            idle_intents.pop(rid, None)
            if action.kind == "charge":
                request_charge(rid, target, robot.spec.battery_capacity_wh - robot.battery_wh,
                               now, "background")
                idle_stats["charge_actions"] += 1
            else:
                robot.available = True
                idle_stats["stay_decisions"] += 1
            return
        # For STAY while moving, finish the committed edge, then stop at its
        # endpoint. Reuse its one existing arrival event; never teleport.
        idle_intents[rid] = {
            "robot_id": rid, "kind": "charge" if action.kind == "charge" else "reposition",
            "target_node": target, "arrival_min": record["arrival_min"],
            "battery_arrival_wh": record["battery_arrival_wh"], "slot": action.slot,
        }
        robot.available = False
        idle_stats[f"{action.kind}_actions"] += 1
        if not moving:
            event = robot.depart_next_edge(graph, now, edge_weight=EDGE_WEIGHT)
            if event is None:
                raise RuntimeError("coordinated idle travel has no edge")
            push_event(event.time_min, 1, "background_node_arrival", event)

'''
    begin = source.index("    def start_idle_action(")
    end = source.index("    @lru_cache(maxsize=65536)\n    def idle_haversine_distance", begin)
    source = replace(source, source[begin:end], helpers, "safe idle retargeting")
    source = replace(source, "        candidate_ids = set()\n        for robot in robots:\n",
                     "        candidate_ids = set()\n"
                     "        baseline_actions = {}\n"
                     "        for robot in robots:\n", "continue-current-plan baselines")
    source = replace(source,
                     "    def plan_idle_fleet(now: float, only_ids: set[int] | None = None) -> None:\n"
                     "        started = time.perf_counter()\n",
                     "    def plan_idle_fleet(now: float, only_ids: set[int] | None = None) -> None:\n"
                     "        if pending:\n"
                     "            idle_stats[\"pending_blocked_epochs\"] += 1\n"
                     "            return\n"
                     "        started = time.perf_counter()\n", "skip work when real orders block idle moves")
    source = replace(source, "            if schedules[robot_id].orders:\n                if robot_id in service_lock:\n",
                     '''            if (intent is not None and intent["kind"] == "reposition"
                    and not schedules[robot_id].orders and not pending
                    and (only_ids is None or robot_id in only_ids)):
                if now - idle_last_target_change.get(robot_id, -math.inf) >= args.idle_retarget_cooldown_min:
                    baseline_actions[robot_id] = IdleAction(
                        robot_id, "stay", node, 0.0, intent["arrival_min"], ready, battery,
                    )
                    decision = decision_snapshot(robot, now)
                    node, battery = int(decision.node_id), float(decision.battery_wh)
                    ready = max(0.0, decision.time_min - now)
                    candidate_ids.add(robot_id)
                else:
                    idle_stats["cooldown_protected"] += 1
            if schedules[robot_id].orders:
                if robot_id in service_lock:
''', "editable relocation snapshots")
    source = replace(source, "            chosen = idle_planner.plan(now, snapshots, candidate_ids, calendar)\n",
                     "            chosen = idle_planner.plan(\n"
                     "                now, snapshots, candidate_ids, calendar,\n"
                     "                baseline_actions=baseline_actions,\n"
                     "                switching_gain_fraction=args.idle_switch_gain_fraction,\n"
                     "            )\n", "coordinated switch threshold")
    source = replace(source, "                if action is None:\n                    robot_by_id[robot_id].available = True\n",
                     "                if action is None:\n"
                     "                    if robot_id not in baseline_actions:\n"
                     "                        robot_by_id[robot_id].available = True\n",
                     "unchanged moving robots remain unavailable")
    source = replace(source,
                     "        if robot.activity is not RobotActivity.MOVING:\n"
                     "            continue_delivery(robot.spec.id, now)\n\n    def dispatch_pending",
                     "        if robot.activity is not RobotActivity.MOVING:\n"
                     "            continue_delivery(robot.spec.id, now)\n"
                     "        request_coordinated_idle_replan(now)\n\n    def dispatch_pending",
                     "replan after every real assignment")
    source = replace(source, "        elif kind == \"idle_replan\":\n            plan_idle_fleet(now)\n",
                     "        elif kind == \"coordinated_idle_replan\":\n"
                     "            coordinated_pending_times.discard(now)\n"
                     "            idle_stats[\"assignment_refresh_epochs\"] += 1\n"
                     "            plan_idle_fleet(now)\n\n"
                     "        elif kind == \"idle_replan\":\n            plan_idle_fleet(now)\n",
                     "settled assignment refresh event")
    source = replace(source, '        "policy_version": "queue_aware_v1",\n',
                     '        "policy_version": "coordinated_idle_v1",\n'
                     '        "idle_coordination": "shared_spatial_marginal_coverage_v1",\n'
                     '        "idle_assignment_refresh_requests": int(idle_stats["assignment_refresh_requests"]),\n'
                     '        "idle_assignment_refresh_epochs": int(idle_stats["assignment_refresh_epochs"]),\n'
                     '        "idle_retarget_actions": int(idle_stats["retarget_actions"]),\n'
                     '        "idle_retarget_preserved_edges": int(idle_stats["retarget_preserved_edges"]),\n'
                     '        "idle_cooldown_protected": int(idle_stats["cooldown_protected"]),\n'
                     '        "idle_commit_rejected": int(idle_stats["commit_rejected"]),\n'
                     '        "idle_pending_blocked_epochs": int(idle_stats["pending_blocked_epochs"]),\n'
                     '        "idle_switch_gain_fraction": args.idle_switch_gain_fraction,\n'
                     '        "idle_retarget_cooldown_min": args.idle_retarget_cooldown_min,\n'
                     '        "idle_decisions_file": str(idle_decisions_path),\n', "versioned coordination diagnostics")
    return source


def _load_namespace() -> dict[str, object]:
    source = queue.idle.heuristic.corrected._patch_source(queue.idle.heuristic.corrected.TARGET.read_text(encoding="utf-8"))
    source = queue.idle.heuristic._patch_heuristic(source)
    source = queue.idle._patch_idle(source)
    source = queue._patch_queue(source)
    source = _patch_coordinated(source)
    module = types.ModuleType("nyc_coordinated_idle_policy_target")
    module.__file__ = str(queue.idle.heuristic.corrected.TARGET)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    _load_namespace()["main"]()


if __name__ == "__main__":
    main()
