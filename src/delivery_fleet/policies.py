"""Benchmark dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .battery_routing import BatteryFeasibleRoute, BatteryFeasibleRouter
from .robot import RobotState
from .scenario_creator import Order


@dataclass(frozen=True, slots=True)
class Assignment:
    order_id: int
    robot_id: int
    assigned_at_min: float
    completion_time_min: float
    route: BatteryFeasibleRoute


class NearestAvailableRobotPolicy:
    """Nearest-Available Robot (NAR) baseline.

    On each revealed order:
      1. Keep only currently available robots that can carry the item.
      2. Choose the one with minimum graph distance to the pickup, ignoring
         battery for the robot-selection step.
      3. For that robot, compute the minimum-time battery-feasible route through
         any required charging stations, with partial charging allowed.

    A None result means no capable robot is currently available; the simulator
    should keep that order pending and retry when robots become available.
    """

    def __init__(
        self,
        graph: nx.Graph,
        router: BatteryFeasibleRouter | None = None,
        *,
        edge_weight: str = "length",
    ) -> None:
        self.graph = graph
        self.edge_weight = edge_weight
        self.router = router or BatteryFeasibleRouter(
            graph, edge_weight=edge_weight
        )

    def _choose_robot(
        self,
        order: Order,
        robots: Iterable[RobotState],
    ) -> RobotState | None:
        candidates = [
            robot
            for robot in robots
            if robot.available and robot.can_hold(order.item)
        ]
        if not candidates:
            return None

        # One Dijkstra from u gives d(u, robot_location) for the whole candidate
        # set on our undirected graph, rather than one search per robot.
        distances = nx.single_source_dijkstra_path_length(
            self.graph,
            order.pickup_node,
            weight=self.edge_weight,
        )

        unreachable = [robot for robot in candidates if robot.node_id not in distances]
        if unreachable:
            raise ValueError("an available robot is disconnected from the pickup")

        return min(
            candidates,
            key=lambda robot: (distances[robot.node_id], robot.spec.id),
        )

    def on_order(
        self,
        order: Order,
        robots: Iterable[RobotState],
        *,
        now_min: float | None = None,
    ) -> Assignment | None:
        """Try to dispatch a newly revealed (or retried pending) order."""

        if now_min is None:
            now_min = order.request_time_min
        if now_min + 1e-9 < order.request_time_min:
            raise ValueError("cannot dispatch an order before its request time")

        robot = self._choose_robot(order, robots)
        if robot is None:
            return None

        route = self.router.plan(
            robot,
            pickup_node=order.pickup_node,
            dropoff_node=order.dropoff_node,
        )

        robot.available = False
        robot.current_order_id = order.id

        return Assignment(
            order_id=order.id,
            robot_id=robot.spec.id,
            assigned_at_min=float(now_min),
            completion_time_min=float(now_min + route.total_time_min),
            route=route,
        )

    def complete_delivery(
        self,
        assignment: Assignment,
        robot: RobotState,
    ) -> None:
        """Apply the assignment's terminal state when its completion event fires."""

        if robot.spec.id != assignment.robot_id:
            raise ValueError("assignment belongs to a different robot")
        if robot.available:
            raise ValueError("cannot complete an assignment for an available robot")
        if robot.current_order_id != assignment.order_id:
            raise ValueError("robot is serving a different order")

        robot.node_id = assignment.route.dropoff_node
        robot.battery_wh = assignment.route.arrival_battery_wh
        robot.current_order_id = None
        robot.available = True
