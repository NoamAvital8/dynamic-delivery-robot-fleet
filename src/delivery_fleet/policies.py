"""Benchmark dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import networkx as nx

from .battery_routing import (
    BatteryFeasibleRoute,
    BatteryFeasibleRouter,
    BatteryRouteQuote,
    NoFeasibleBatteryRoute,
)
from .robot import RobotNodeArrivalEvent, RobotState
from .routing import DistanceOracle
from .scenario_creator import Order


@dataclass(frozen=True, slots=True)
class Assignment:
    order_id: int
    robot_id: int
    assigned_at_min: float
    completion_time_min: float
    route: BatteryFeasibleRoute
    # If the route can depart immediately, this is the first committed
    # node-arrival event. It is None when a charge at the current node must
    # happen before departure.
    first_node_arrival_event: RobotNodeArrivalEvent | None = None


class NearestAvailableRobotPolicy:
    """Nearest-Available Robot (NAR) baseline.

    Available payload-capable robots are considered in exact graph-distance
    order from the pickup.  The outward shortest-path search is run once and is
    continued if a nearer robot is battery-infeasible.  Route feasibility/cost
    is evaluated without constructing the street-node path; only the chosen
    candidate's route is materialized.

    A None result means no currently available capable robot has a safe route;
    the simulator should keep the order pending and retry later.
    """

    def __init__(
        self,
        graph: nx.Graph,
        router: BatteryFeasibleRouter | None = None,
        *,
        edge_weight: str = "length",
        distance_oracle: DistanceOracle | None = None,
    ) -> None:
        self.graph = graph
        self.edge_weight = edge_weight
        if router is not None:
            self.router = router
            self.distance_oracle = router.distance_oracle
        else:
            self.distance_oracle = distance_oracle or DistanceOracle(
                graph,
                edge_weight=edge_weight,
            )
            self.router = BatteryFeasibleRouter(
                graph,
                edge_weight=edge_weight,
                distance_oracle=self.distance_oracle,
            )

    def _candidate_robots_in_distance_order(
        self,
        order: Order,
        robots: Iterable[RobotState],
    ) -> Iterator[tuple[RobotState, float]]:
        candidates = [
            robot
            for robot in robots
            if robot.available and robot.can_hold(order.item)
        ]
        if not candidates:
            return

        by_node: dict[object, list[RobotState]] = {}
        for robot in candidates:
            by_node.setdefault(robot.node_id, []).append(robot)
        for robots_at_node in by_node.values():
            robots_at_node.sort(key=lambda robot: robot.spec.id)

        for nearest in self.distance_oracle.iter_nearest_targets(
            order.pickup_node,
            by_node,
        ):
            for robot in by_node[nearest.node_id]:
                yield robot, nearest.distance_m

    def _choose_feasible_quote(
        self,
        order: Order,
        robots: Iterable[RobotState],
    ) -> tuple[RobotState, BatteryRouteQuote] | None:
        pickup_to_dropoff = self.distance_oracle.distance(
            order.pickup_node,
            order.dropoff_node,
        )
        for robot, start_to_pickup in self._candidate_robots_in_distance_order(
            order,
            robots,
        ):
            try:
                quote = self.router.evaluate(
                    robot,
                    order.pickup_node,
                    order.dropoff_node,
                    start_to_pickup_m=start_to_pickup,
                    pickup_to_dropoff_m=pickup_to_dropoff,
                )
            except NoFeasibleBatteryRoute:
                continue
            return robot, quote
        return None

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

        selected = self._choose_feasible_quote(order, robots)
        if selected is None:
            return None
        robot, quote = selected
        route = self.router.materialize(
            quote,
            speed_mps=robot.spec.speed_mps,
        )

        robot.available = False
        robot.current_order_id = order.id
        robot.set_planned_path(route.node_path)

        # If the route starts with a partial charge at the robot's current node,
        # movement starts only after that charging event. Otherwise commit the
        # first graph edge now so the simulator immediately knows next_node+ETA.
        initial_charge = (
            route.charging_events[0]
            if route.charging_events
            and route.charging_events[0].node_id == robot.node_id
            else None
        )
        first_arrival = None
        if initial_charge is None:
            first_arrival = robot.depart_next_edge(
                self.graph,
                float(now_min),
                edge_weight=self.edge_weight,
            )

        return Assignment(
            order_id=order.id,
            robot_id=robot.spec.id,
            assigned_at_min=float(now_min),
            completion_time_min=float(now_min + route.total_time_min),
            route=route,
            first_node_arrival_event=first_arrival,
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

        robot.clear_movement_plan()
        robot.node_id = assignment.route.dropoff_node
        robot.battery_wh = assignment.route.arrival_battery_wh
        robot.current_order_id = None
        robot.available = True
