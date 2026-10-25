"""Proven hindsight optimum for deliberately tiny, nonbinding-energy cases.

Every assignment and precedence-respecting pickup/drop-off order is enumerated.
For a fixed stop order, earliest travel/service is optimal because delivery loss
is nondecreasing in completion time. A conservative certificate proves that
no candidate route needs charging or exceeds payload/volume capacity. This is
an offline clairvoyant lower bound, not a scalable online dispatch policy.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
from typing import Any

import networkx as nx

from .deadlines import delivery_loss, delivery_time_allowance_min
from .fleet import create_default_fleet
from .scenario_creator import Scenario


HANDLING_MIN = 1.0


def solve_small_hindsight(
    graph: nx.Graph,
    scenario: Scenario,
    *,
    max_orders: int = 5,
    max_robots: int = 3,
) -> dict[str, Any]:
    """Return an exhaustive optimum, or reject a case without a valid proof."""

    orders = scenario.orders
    if not 1 <= len(orders) <= max_orders:
        raise ValueError(f"exact oracle supports 1..{max_orders} orders")
    if len({order.id for order in orders}) != len(orders):
        raise ValueError("order ids must be distinct")
    robots = create_default_fleet(graph, seed=scenario.seed)
    if not 1 <= len(robots) <= max_robots:
        raise ValueError(f"exact oracle supports 1..{max_robots} robots")

    nodes = {int(robot.node_id) for robot in robots}
    for order in orders:
        nodes.add(int(order.pickup_node))
        nodes.add(int(order.dropoff_node))
    distance_rows = {
        node: nx.single_source_dijkstra_path_length(graph, node, weight="length")
        for node in nodes
    }
    distances: dict[tuple[int, int], float] = {}
    for source in nodes:
        for destination in nodes:
            try:
                distance = float(distance_rows[source][destination])
            except KeyError as exc:
                raise ValueError("all oracle service nodes must be mutually reachable") from exc
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("invalid service-node graph distance")
            distances[source, destination] = distance

    # These deliberately tiny tests are constructed so charging and capacity
    # cannot bind, even in the worst possible stop order. The resulting search
    # is therefore exact for the simulator's full feasibility model here.
    max_leg_m = max(distances.values())
    max_route_m = 2 * len(orders) * max_leg_m
    total_weight = sum(order.item.weight_kg for order in orders)
    total_volume = sum(order.item.volume_l for order in orders)
    for robot in robots:
        spec = robot.spec
        if total_weight > spec.max_payload_kg or total_volume > spec.max_volume_l:
            raise ValueError("payload/volume is binding; exact oracle cannot certify this case")
        if max_route_m * spec.energy_per_meter_wh > robot.battery_wh + 1e-9:
            raise ValueError("charging could bind; exact oracle cannot certify this case")

    allowances = [
        delivery_time_allowance_min(
            distances[int(order.pickup_node), int(order.dropoff_node)],
            order.importance,
        )
        for order in orders
    ]

    @lru_cache(maxsize=None)
    def best_robot_route(robot_index: int, subset: int) -> tuple[float, tuple[tuple[Any, ...], ...]]:
        if subset == 0:
            return 0.0, ()
        robot = robots[robot_index]
        speed = robot.spec.speed_mps
        best_loss = math.inf
        best_steps: tuple[tuple[Any, ...], ...] = ()

        def search(
            node: int, now: float, picked: int, delivered: int,
            loss: float, steps: tuple[tuple[Any, ...], ...],
        ) -> None:
            nonlocal best_loss, best_steps
            if loss >= best_loss - 1e-12:
                return
            if delivered == subset:
                best_loss, best_steps = loss, steps
                return
            for index, order in enumerate(orders):
                bit = 1 << index
                if not subset & bit or delivered & bit:
                    continue
                if not picked & bit:
                    target = int(order.pickup_node)
                    arrival = now + distances[node, target] / speed / 60.0
                    service_start = max(arrival, order.request_time_min)
                    finish = service_start + HANDLING_MIN
                    search(target, finish, picked | bit, delivered, loss, steps + (
                        ("pickup", order.id, target, arrival, service_start, finish),
                    ))
                else:
                    target = int(order.dropoff_node)
                    arrival = now + distances[node, target] / speed / 60.0
                    finish = arrival + HANDLING_MIN
                    increment = delivery_loss(
                        finish - order.request_time_min, allowances[index], order.importance
                    )
                    search(target, finish, picked, delivered | bit, loss + increment,
                           steps + (("dropoff", order.id, target, arrival, arrival, finish),))

        search(int(robot.node_id), 0.0, 0, 0, 0.0, ())
        return best_loss, best_steps

    best_total = math.inf
    best_assignment: tuple[int, ...] | None = None
    best_routes: tuple[tuple[tuple[Any, ...], ...], ...] = ()
    for assignment in product(range(len(robots)), repeat=len(orders)):
        masks = [0] * len(robots)
        for order_index, robot_index in enumerate(assignment):
            masks[robot_index] |= 1 << order_index
        solutions = [best_robot_route(index, mask) for index, mask in enumerate(masks)]
        total = sum(solution[0] for solution in solutions)
        if total < best_total - 1e-12:
            best_total = total
            best_assignment = assignment
            best_routes = tuple(solution[1] for solution in solutions)
    if best_assignment is None or not math.isfinite(best_total):
        raise RuntimeError("exact enumeration found no complete solution")

    completed = {
        int(step[1]): float(step[5])
        for route in best_routes for step in route if step[0] == "dropoff"
    }
    on_time = sum(
        completed[order.id] <= order.request_time_min + allowances[index] + 1e-9
        for index, order in enumerate(orders)
    )
    return {
        "scenario": scenario.graph_name,
        "seed": scenario.seed,
        "orders": len(orders),
        "robots": len(robots),
        "optimal_loss": best_total,
        "on_time": on_time,
        "late": len(orders) - on_time,
        "completed_at_min": max(completed.values()),
        "clairvoyant": True,
        "proven_optimal": True,
        "certificate": {
            "max_any_route_distance_m": max_route_m,
            "max_total_payload_kg": total_weight,
            "max_total_volume_l": total_volume,
            "charging_cannot_help": True,
            "enumeration": "all robot assignments and precedence-feasible stop orders",
        },
        "routes": [
            {
                "robot_id": robot.spec.id,
                "robot_type": robot.spec.robot_type.value,
                "start_node": int(robot.node_id),
                "assigned_order_ids": [
                    orders[i].id for i, assigned in enumerate(best_assignment)
                    if assigned == robot_index
                ],
                "stops": [
                    {
                        "kind": step[0], "order_id": step[1], "node": step[2],
                        "arrival_min": step[3], "service_start_min": step[4],
                        "finish_min": step[5],
                    }
                    for step in best_routes[robot_index]
                ],
            }
            for robot_index, robot in enumerate(robots)
        ],
    }
