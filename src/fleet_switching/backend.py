"""Adapter to the preserved VM backend; never imports it until explicitly used.

The terminal rule freezes committed travel/handling/charging, including known
background charging. Pending requests are appended FIFO to the earliest
feasible projected robot (bounded shortlist, expanded if necessary). All plans
are then projected jointly, so a newly appended charge can delay an existing
mission at a shared station. This is a terminal approximation, not future demand.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

from .projection import OrderCost, Session, Step, Waiting, project


def activate(config):
    root = Path(config.backend_root).resolve()
    existing = sys.modules.get("delivery_fleet")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different delivery_fleet backend is already imported; use a fresh process")
    if config.dependency_dir:
        sys.path.insert(0, config.dependency_dir)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    return importlib.import_module("delivery_fleet.meta_policy.runtime")


def exact_distance(ns, source, target):
    return float(ns["row_distance"](ns["distance_row"](ns["oracle"], int(source)), int(target)))


def completion_projection(ns):
    now = float(ns["now"])
    power = float(ns["DEFAULT_CHARGING_POWER_W"])
    ports = {int(s): int(ns["DEFAULT_NUMBER_OF_PORTS"]) for s in ns["station_state"]}
    active, waiting, blocked, battery_after = [], [], set(), {}
    for station, state in ns["station_state"].items():
        for rid, session in state.active.items():
            finish = session.start_time_min + session.energy_wh / power * 60.
            active.append(Session(rid, int(station), max(now, finish)))
            blocked.add(rid)
            battery_after[rid] = min(ns["robot_by_id"][rid].spec.battery_capacity_wh,
                                     session.start_battery_wh + session.energy_wh)
        for request in state.queue:
            rid = request.robot_id
            waiting.append(Waiting(rid, int(station), request.energy_wh / power * 60., request.arrival_time_min))
            blocked.add(rid)
            battery_after[rid] = min(ns["robot_by_id"][rid].spec.battery_capacity_wh,
                                     ns["robot_by_id"][rid].battery_wh + request.energy_wh)
    plans, costs, tails = {}, {}, {}
    fallback_steps = 0

    def append_phases(output, phases, robot, node, battery):
        for phase in phases:
            if phase.kind == "travel":
                distance = float(phase.distance_m)
                battery -= distance * robot.spec.energy_per_meter_wh
                if battery < -1e-3:
                    raise ValueError("Frozen itinerary has infeasible travel energy")
                output.append(Step("travel", distance / robot.spec.speed_mps / 60.))
                node = int(phase.target_node)
            elif phase.kind == "charge":
                if int(phase.target_node) != node:
                    raise ValueError("Frozen charge occurs at the wrong node")
                energy = min(max(0., phase.energy_wh), robot.spec.battery_capacity_wh - max(0., battery))
                if energy > 1e-8:
                    output.append(Step("charge", energy / power * 60., node))
                    battery += energy
            else:
                raise ValueError("Unknown route phase")
        return node, max(0., battery)

    def route_to(robot, node, battery, target):
        if node == int(target):
            return ()
        proxy = ns["RobotState"](spec=robot.spec, node_id=node, battery_wh=battery, available=False)
        return ns["plan_final_target"](proxy, int(target),
            start_to_target_m=exact_distance(ns, node, target)).phases

    for robot in sorted(ns["robots"], key=lambda r: r.spec.id):
        rid = robot.spec.id
        schedule = ns["schedules"][rid]
        output = plans[rid] = []
        node, battery = int(robot.node_id), float(battery_after.get(rid, robot.battery_wh))
        for oid, state in schedule.orders.items():
            order = state.order
            if order.request_time_min > now + 1e-8 or oid in costs:
                raise ValueError("Unknown future or duplicate active order")
            costs[oid] = OrderCost(order.request_time_min, state.deadline_min, order.importance)
        if not schedule.orders:
            if rid not in blocked:
                intent = ns.get("idle_intents", {}).get(rid)
                if intent:
                    output.append(Step("travel", max(0., intent["arrival_min"] - now)))
                    node, battery = int(intent["target_node"]), float(intent["battery_arrival_wh"])
                    if intent["kind"] == "charge":
                        output.append(Step("charge", max(0., robot.spec.battery_capacity_wh - battery) / power * 60., node))
                        battery = robot.spec.battery_capacity_wh
                elif robot.is_moving:
                    snap = ns["decision_snapshot"](robot, now)
                    output.append(Step("travel", max(0., snap.time_min - now)))
                    node, battery = int(snap.node_id), float(snap.battery_wh)
                    path = (node, *map(int, robot.remaining_route))
                    if len(path) > 1:
                        distance = float(ns["path_distance"](path))
                        output.append(Step("travel", distance / robot.spec.speed_mps / 60.))
                        node, battery = path[-1], battery - distance * robot.spec.energy_per_meter_wh
                    if node in ports:
                        output.append(Step("charge", max(0., robot.spec.battery_capacity_wh - battery) / power * 60., node))
                        battery = robot.spec.battery_capacity_wh
            tails[rid] = (node, battery)
            continue
        stops = list(schedule.stops)
        if rid in ns["service_lock"]:
            locked = ns["service_lock"][rid][1]
            if not stops or stops[0] != locked:
                raise ValueError("Service lock disagrees with the route")
            finish = ns["idle_service_finish_min"].get(rid)
            if finish is None:
                raise ValueError("Missing active-service finish time")
            output.append(Step("service", max(0., finish - now), dropoff=locked.order_id if locked.kind == "dropoff" else None))
            stops.pop(0)
        elif robot.is_moving:
            snap = ns["decision_snapshot"](robot, now)
            output.append(Step("travel", max(0., snap.time_min - now)))
            node, battery = int(snap.node_id), float(snap.battery_wh)
        prepared = {s.stop: s for s in schedule.prepared_steps}
        for index, stop in enumerate(stops):
            is_front = bool(schedule.stops) and stop == schedule.stops[0]
            if is_front and ns["phase_version"][rid] == schedule.version:
                if robot.is_moving:
                    path = (node, *map(int, robot.remaining_route))
                    if len(path) > 1:
                        distance = float(ns["path_distance"](path))
                        output.append(Step("travel", distance / robot.spec.speed_mps / 60.))
                        node, battery = path[-1], battery - distance * robot.spec.energy_per_meter_wh
                phases = tuple(ns["active_phases"][rid])
            elif stop in prepared:
                phases = prepared[stop].phases
            else:
                # Keep the service sequence; reconstruct only the missing leg.
                phases = route_to(robot, node, battery, stop.node_id)
                fallback_steps += 1
            node, battery = append_phases(output, phases, robot, node, battery)
            if node != int(stop.node_id):
                raise ValueError("Frozen route does not reach its service stop")
            handling = ns["PICKUP_HANDLING_MIN"] if stop.kind == "pickup" else ns["DROPOFF_HANDLING_MIN"]
            output.append(Step("service", float(handling), dropoff=stop.order_id if stop.kind == "dropoff" else None))
        tails[rid] = (node, battery)
    initial = project(now, plans, ports, costs, active, waiting)
    ready = dict(initial.robot_ready)
    pending_ids = set()
    for order in sorted(ns["pending"], key=lambda o: (o.request_time_min, o.id)):
        if order.id in costs or order.id in pending_ids or order.request_time_min > now + 1e-8:
            raise ValueError("Duplicate/future pending order")
        pending_ids.add(order.id)
        eligible = [r for r in ns["robots"] if r.spec.can_hold(order.item)]
        eligible.sort(key=lambda r: (ready[r.spec.id] + exact_distance(ns, tails[r.spec.id][0], order.pickup_node) / r.spec.speed_mps / 60., r.spec.id))
        options = []
        for robot in eligible:
            rid = robot.spec.id
            node, battery = tails[rid]
            steps = []
            try:
                for kind, target in (("pickup", order.pickup_node), ("dropoff", order.dropoff_node)):
                    node, battery = append_phases(steps, route_to(robot, node, battery, target), robot, node, battery)
                    handling = ns["PICKUP_HANDLING_MIN"] if kind == "pickup" else ns["DROPOFF_HANDLING_MIN"]
                    steps.append(Step("service", float(handling), dropoff=order.id if kind == "dropoff" else None))
            except ns["NoFeasibleBatteryRoute"]:
                continue
            options.append((ready[rid] + sum(s.minutes for s in steps), rid, steps, node, battery))
            if len(options) >= 5:
                break
        if not options:
            raise ValueError("Pending order has no feasible terminal completion; invalid label")
        finish, rid, steps, node, battery = min(options, key=lambda x: (x[0], x[1]))
        plans[rid].extend(steps)
        ready[rid], tails[rid] = finish, (node, battery)
        direct = ns["get_direct_distance"](order)
        costs[order.id] = OrderCost(order.request_time_min,
            ns["delivery_deadline_min"](order.request_time_min, direct, order.importance), order.importance)
    final = project(now, plans, ports, costs, active, waiting) if pending_ids else initial
    return final, {"pending_orders": len(pending_ids), "assigned_orders": len(costs) - len(pending_ids),
                   "reconstructed_route_legs": fallback_steps, "projected_queue_wait_min": final.queue_wait_minutes}


class Backend:
    def __init__(self, config, episode, scratch):
        module = activate(config)
        from delivery_fleet import fleet
        original_factory = fleet.create_default_fleet
        def configured_fleet(graph, **kwargs):
            kwargs["robot_count"] = episode["fleet_count"]
            return original_factory(graph, **kwargs)
        fleet.create_default_fleet = configured_fleet
        try:
            self.runtime = module.FleetRuntime(episode["graph"], episode["scenario"], config.spatial_model,
                config.paper_model, episode["prior_rates_per_hour"], scratch=scratch,
                shared_cache=Path(config.output) / "routing_cache")
        finally:
            fleet.create_default_fleet = original_factory
        for robot in self.runtime.ns["robots"]:
            robot.battery_wh = robot.spec.battery_capacity_wh * episode.get("initial_battery_fraction", 1.)
        from delivery_fleet.meta_policy.features import map_features
        self.static = map_features(self.runtime.ns["graph"])

    def features(self, age=0., recent=0.):
        from delivery_fleet.meta_policy.features import extract
        return extract(self.runtime, self.static, holding_min=15., policy_age_min=age, recent_loss_rate=recent)

    def project(self):
        return completion_projection(self.runtime.ns)

    def close(self):
        self.runtime.close()
