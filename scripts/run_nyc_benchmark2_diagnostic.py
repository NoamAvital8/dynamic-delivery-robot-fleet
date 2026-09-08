from __future__ import annotations

"""Run Benchmark 2 with low-overhead assignment/battery diagnostics.

The benchmark itself stays untouched. A profile hook records the latest assignment
and activation snapshot for each robot. If the deterministic benchmark fails, this
wrapper inspects the live traceback frame and prints a structured JSON dump with
the exact robot, order, battery, travel leg, route and interruption context.
"""

import json
from pathlib import Path
import runpy
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "run_nyc_benchmark2.py"

assignment_history: dict[int, dict[str, Any]] = {}
activation_history: dict[int, dict[str, Any]] = {}


def _activity_value(activity: object) -> object:
    return getattr(activity, "value", str(activity))


def _robot_snapshot(robot: object) -> dict[str, Any]:
    spec = robot.spec
    return {
        "robot_id": int(spec.id),
        "robot_type": getattr(spec, "display_name", type(spec).__name__),
        "activity": _activity_value(robot.activity),
        "node_id": int(robot.node_id),
        "battery_wh": float(robot.battery_wh),
        "battery_capacity_wh": float(spec.battery_capacity_wh),
        "energy_per_meter_wh": float(spec.energy_per_meter_wh),
        "remaining_range_m": float(robot.remaining_range_m),
        "current_order_id": robot.current_order_id,
        "available_flag": bool(robot.available),
        "next_node": None if robot.next_node is None else int(robot.next_node),
        "next_node_arrival_time_min": robot.next_node_arrival_time_min,
        "edge_departure_time_min": robot.edge_departure_time_min,
        "current_edge_distance_m": robot.current_edge_distance_m,
        "remaining_route_nodes": len(robot.remaining_route),
        "remaining_route_head": [int(x) for x in robot.remaining_route[:8]],
    }


def _action_dict(action: object, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "kind": action.kind,
        "value": float(action.value),
        "node_id": None if action.node_id is None else int(action.node_id),
    }


def _route_dict(route: object) -> dict[str, Any]:
    return {
        "pickup_node": int(route.pickup_node),
        "dropoff_node": int(route.dropoff_node),
        "total_distance_m": float(route.total_distance_m),
        "travel_time_min": float(route.travel_time_min),
        "charging_time_min": float(route.charging_time_min),
        "total_time_min": float(route.total_time_min),
        "arrival_battery_wh": float(route.arrival_battery_wh),
        "required_dropoff_reserve_wh": float(route.required_dropoff_reserve_wh),
        "segments": [
            {
                "waypoints": [int(x) for x in segment.waypoints],
                "distance_m": float(segment.distance_m),
                "ends_at_charger": bool(segment.ends_at_charger),
            }
            for segment in route.segments
        ],
        "charging_events": [
            {
                "node_id": int(event.node_id),
                "energy_added_wh": float(event.energy_added_wh),
                "duration_min": float(event.duration_min),
                "battery_before_wh": float(event.battery_before_wh),
                "battery_after_wh": float(event.battery_after_wh),
            }
            for event in route.charging_events
        ],
    }


def profiler(frame, event: str, arg):
    if event != "call":
        return
    if Path(frame.f_code.co_filename).name != TARGET.name:
        return

    name = frame.f_code.co_name
    loc = frame.f_locals

    if name == "assign_order":
        candidate = loc.get("candidate")
        order = loc.get("order")
        route = loc.get("route")
        now = loc.get("now")
        if candidate is None or order is None or route is None:
            return
        robot = candidate.robot
        assignment_history[int(robot.spec.id)] = {
            "assignment_time_min": float(now),
            "order_id": int(order.id),
            "order_request_time_min": float(order.request_time_min),
            "order_pickup_node": int(order.pickup_node),
            "order_dropoff_node": int(order.dropoff_node),
            "order_importance": float(order.importance),
            "pre_assignment_robot": _robot_snapshot(robot),
            "candidate_policy_distance_m": float(candidate.policy_distance_m),
            "candidate_route_start_node": int(candidate.route_start_node),
            "candidate_route_start_time_min": float(candidate.route_start_time_min),
            "candidate_route_start_battery_wh": float(candidate.route_start_battery_wh),
            "candidate_decision_to_pickup_m": float(candidate.decision_to_pickup_m),
            "route": _route_dict(route),
        }

    elif name == "activate_assignment":
        robot = loc.get("robot")
        plan = loc.get("plan")
        now = loc.get("now")
        if robot is None or plan is None:
            return
        activation_history[int(robot.spec.id)] = {
            "activation_time_min": float(now),
            "order_id": int(plan.order.id),
            "robot_at_activation": _robot_snapshot(robot),
            "route_start_time_min": float(plan.route_start_time_min),
            "route_start_node": int(plan.route.segments[0].waypoints[0]),
        }


def _find_main_frame(tb):
    chosen = None
    while tb is not None:
        frame = tb.tb_frame
        if Path(frame.f_code.co_filename).name == TARGET.name and frame.f_code.co_name == "main":
            chosen = frame
        tb = tb.tb_next
    return chosen


def _failure_dump(exc: BaseException) -> dict[str, Any]:
    frame = _find_main_frame(exc.__traceback__)
    result: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }
    if frame is None:
        result["diagnostic_error"] = "benchmark main frame not found"
        return result

    loc = frame.f_locals
    now = loc.get("now")
    result["simulation_time_min"] = None if now is None else float(now)
    result["event_kind"] = loc.get("kind")
    result["event_payload"] = repr(loc.get("payload"))
    result["delivered_count"] = int(loc.get("delivered", 0))
    result["pending_count"] = len(loc.get("pending", ()))
    result["active_delivery_count"] = len(loc.get("delivery_plans", {}))
    result["deferred_count"] = len(loc.get("deferred", {}))

    robot_id_raw = loc.get("robot_id")
    if robot_id_raw is None:
        return result
    robot_id = int(robot_id_raw)
    result["robot_id"] = robot_id

    robot_by_id = loc.get("robot_by_id", {})
    robot = robot_by_id.get(robot_id)
    if robot is not None:
        result["robot_runtime"] = _robot_snapshot(robot)

    if loc.get("kind") == "delivery_travel_complete":
        distance_m = float(loc.get("distance_m", 0.0))
        target = loc.get("target")
        before = None
        if robot is not None:
            # The benchmark subtracts energy immediately before raising. Recover
            # the pre-leg battery from the post-subtraction value in the frame.
            needed = distance_m * float(robot.spec.energy_per_meter_wh)
            before = float(robot.battery_wh) + needed
            result["failing_leg"] = {
                "from_node": int(robot.node_id),
                "target_node": None if target is None else int(target),
                "distance_m": distance_m,
                "energy_required_wh": needed,
                "battery_before_leg_wh": before,
                "battery_after_subtraction_wh": float(robot.battery_wh),
                "shortfall_wh": max(0.0, needed - before),
            }

    plans = loc.get("delivery_plans", {})
    plan = plans.get(robot_id)
    if plan is not None:
        result["current_plan"] = {
            "order_id": int(plan.order.id),
            "assigned_at_min": float(plan.assigned_at_min),
            "route_start_time_min": float(plan.route_start_time_min),
            "action_index_after_event_schedule": int(plan.action_index),
            "route": _route_dict(plan.route),
            "all_actions": [
                _action_dict(action, i) for i, action in enumerate(plan.actions)
            ],
            "remaining_actions": [
                _action_dict(action, i)
                for i, action in enumerate(plan.actions)
                if i >= plan.action_index
            ],
        }

    result["assignment_snapshot"] = assignment_history.get(robot_id)
    result["activation_snapshot"] = activation_history.get(robot_id)

    station_state = loc.get("station_state", {})
    charger_context = []
    for station_node, state in station_state.items():
        active = state.active.get(robot_id)
        queued = [req for req in state.queue if req.robot_id == robot_id]
        if active is not None or queued:
            charger_context.append(
                {
                    "station_node": int(station_node),
                    "active": None
                    if active is None
                    else {
                        "purpose": active.purpose,
                        "energy_wh": float(active.energy_wh),
                        "start_time_min": float(active.start_time_min),
                        "start_battery_wh": float(active.start_battery_wh),
                        "token": int(active.token),
                    },
                    "queued": [
                        {
                            "purpose": req.purpose,
                            "energy_wh": float(req.energy_wh),
                            "arrival_time_min": float(req.arrival_time_min),
                        }
                        for req in queued
                    ],
                }
            )
    result["charger_context"] = charger_context
    return result


def main() -> None:
    sys.setprofile(profiler)
    try:
        runpy.run_path(str(TARGET), run_name="__main__")
    except BaseException as exc:
        sys.setprofile(None)
        print("BENCHMARK2_DIAGNOSTIC_FAILURE_JSON", flush=True)
        print(json.dumps(_failure_dump(exc), indent=2, sort_keys=True), flush=True)
        raise
    finally:
        sys.setprofile(None)


if __name__ == "__main__":
    main()
