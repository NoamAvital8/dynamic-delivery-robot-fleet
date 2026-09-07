"""Robot capabilities and mutable runtime state.

Movement is intentionally discrete in space and continuous in time. A robot is
never represented by an interpolated coordinate inside an edge. While an edge
is being traversed we store only:

- the last graph node actually reached,
- the committed next graph node,
- the time that next node will be reached.

The committed edge cannot be changed mid-edge. The route *after* the next node
is mutable, which gives future policies a clean place to reroute a robot when a
new request appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Hashable, Iterable

import networkx as nx

from .defaults import MIN_ROBOT_FULL_BATTERY_RANGE_M
from .scenario_creator import Item

NodeId = Hashable
_EPS = 1e-9


class RobotActivity(str, Enum):
    """What the robot is physically doing at the current simulation time."""

    IDLE = "idle"
    MOVING = "moving"
    CHARGING = "charging"
    WAITING = "waiting"


@dataclass(frozen=True, slots=True)
class RobotNodeArrivalEvent:
    """The only movement event needed by the event-driven simulator."""

    robot_id: int
    from_node: NodeId
    node_id: NodeId
    time_min: float


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """Fixed heterogeneous capabilities of one delivery robot."""

    id: int
    speed_mps: float
    max_payload_kg: float
    max_volume_l: float
    battery_capacity_wh: float
    energy_per_meter_wh: float

    def __post_init__(self) -> None:
        if self.speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        if self.max_payload_kg <= 0:
            raise ValueError("max_payload_kg must be positive")
        if self.max_volume_l <= 0:
            raise ValueError("max_volume_l must be positive")
        if self.battery_capacity_wh <= 0:
            raise ValueError("battery_capacity_wh must be positive")
        if self.energy_per_meter_wh <= 0:
            raise ValueError("energy_per_meter_wh must be positive")
        if self.full_battery_range_m + _EPS < MIN_ROBOT_FULL_BATTERY_RANGE_M:
            raise ValueError(
                "robot full-battery range must be at least "
                f"{MIN_ROBOT_FULL_BATTERY_RANGE_M:.0f} m"
            )

    @property
    def full_battery_range_m(self) -> float:
        return self.battery_capacity_wh / self.energy_per_meter_wh

    def can_hold(self, item: Item) -> bool:
        return (
            item.weight_kg <= self.max_payload_kg
            and item.volume_l <= self.max_volume_l
        )


def _edge_distance_m(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    edge_weight: str,
) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        raise ValueError(f"{u!r} and {v!r} are not adjacent graph nodes")

    if graph.is_multigraph():
        distance = min(
            float(attrs.get(edge_weight, 1.0)) for attrs in data.values()
        )
    else:
        distance = float(data.get(edge_weight, 1.0))

    if distance < 0:
        raise ValueError("edge distance cannot be negative")
    return distance


@dataclass(slots=True)
class RobotState:
    """Mutable runtime state of one robot.

    ``node_id`` is always the last node the robot has actually reached. If the
    robot is moving, ``next_node`` and ``next_node_arrival_time_min`` describe
    the committed edge. ``remaining_route`` contains only nodes *after* that
    committed next node, so it may safely be replaced by a future policy while
    the robot is moving.

    ``available`` means dispatchable by the current benchmark policy. It is
    deliberately separate from ``current_order_id`` so future repositioning or
    transfer actions may make a robot temporarily unavailable without it being
    on a delivery.
    """

    spec: RobotSpec
    node_id: NodeId
    battery_wh: float
    available: bool = True
    current_order_id: int | None = None

    activity: RobotActivity = RobotActivity.IDLE

    # Committed edge. These fields are either all populated or all empty.
    next_node: NodeId | None = None
    next_node_arrival_time_min: float | None = None
    edge_departure_time_min: float | None = None
    current_edge_distance_m: float | None = None

    # Nodes after the committed next node (or after node_id when stationary).
    remaining_route: list[NodeId] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.battery_wh < -_EPS:
            raise ValueError("battery_wh cannot be negative")
        if self.battery_wh > self.spec.battery_capacity_wh + _EPS:
            raise ValueError("battery_wh cannot exceed battery capacity")
        if self.available and self.current_order_id is not None:
            raise ValueError("available robot cannot have a current order")

        edge_fields = (
            self.next_node,
            self.next_node_arrival_time_min,
            self.edge_departure_time_min,
            self.current_edge_distance_m,
        )
        populated = [value is not None for value in edge_fields]
        if any(populated) and not all(populated):
            raise ValueError("committed-edge fields must be populated together")
        if self.activity is RobotActivity.MOVING and not all(populated):
            raise ValueError("moving robot must have a committed next node")
        if self.activity is not RobotActivity.MOVING and any(populated):
            raise ValueError("only a moving robot may have a committed edge")
        if self.available and self.activity is not RobotActivity.IDLE:
            raise ValueError("an available robot must be idle")

    @classmethod
    def fully_charged(cls, spec: RobotSpec, node_id: NodeId) -> "RobotState":
        return cls(
            spec=spec,
            node_id=node_id,
            battery_wh=spec.battery_capacity_wh,
        )

    @property
    def remaining_range_m(self) -> float:
        return self.battery_wh / self.spec.energy_per_meter_wh

    @property
    def is_moving(self) -> bool:
        return self.activity is RobotActivity.MOVING

    @property
    def decision_node(self) -> NodeId:
        """Earliest node at which a new route may take effect.

        While moving this is the already-committed next node. Otherwise it is
        the robot's current node.
        """

        return self.next_node if self.is_moving else self.node_id

    @property
    def battery_at_decision_node_wh(self) -> float:
        """Battery expected at the earliest legal rerouting node."""

        if not self.is_moving:
            return self.battery_wh
        assert self.current_edge_distance_m is not None
        return self.battery_wh - (
            self.current_edge_distance_m * self.spec.energy_per_meter_wh
        )

    def decision_time_min(self, now_min: float) -> float:
        """Earliest time at which a newly planned route may take effect."""

        if not self.is_moving:
            return float(now_min)
        assert self.next_node_arrival_time_min is not None
        return float(self.next_node_arrival_time_min)

    def can_hold(self, item: Item) -> bool:
        return self.spec.can_hold(item)

    def set_planned_path(self, node_path: Iterable[NodeId]) -> None:
        """Set a path while stationary without starting movement yet.

        The first node must be the robot's current node. This separation is
        useful when charging or another node action must happen before the first
        edge departure.
        """

        if self.is_moving:
            raise ValueError("cannot replace the committed edge while moving")

        path = list(node_path)
        if not path:
            raise ValueError("node_path cannot be empty")
        if path[0] != self.node_id:
            raise ValueError("planned path must start at the robot's current node")
        self.remaining_route = path[1:]

    def replan_from_decision_node(self, node_path: Iterable[NodeId]) -> None:
        """Replace only the editable part of the route.

        If the robot is moving, the first node must be ``next_node`` and the
        current edge/ETA are left untouched. If it is stationary, the first
        node must be ``node_id``.
        """

        path = list(node_path)
        if not path:
            raise ValueError("node_path cannot be empty")
        if path[0] != self.decision_node:
            raise ValueError("new path must start at the earliest decision node")
        self.remaining_route = path[1:]

    def depart_next_edge(
        self,
        graph: nx.Graph,
        now_min: float,
        *,
        edge_weight: str = "length",
    ) -> RobotNodeArrivalEvent | None:
        """Commit the next graph edge and return its future arrival event.

        No continuous position is created. Until the returned event fires, the
        robot is represented by ``node_id -> next_node`` plus the ETA.
        """

        if self.is_moving:
            raise ValueError("robot already has a committed edge")
        if self.activity is RobotActivity.CHARGING:
            raise ValueError("robot cannot depart while charging")
        if not self.remaining_route:
            return None

        target = self.remaining_route[0]
        distance_m = _edge_distance_m(graph, self.node_id, target, edge_weight)
        energy_wh = distance_m * self.spec.energy_per_meter_wh
        if energy_wh > self.battery_wh + _EPS:
            raise ValueError("robot does not have enough battery for the next edge")

        travel_time_min = distance_m / self.spec.speed_mps / 60.0
        arrival_time = float(now_min + travel_time_min)

        self.remaining_route.pop(0)
        self.next_node = target
        self.next_node_arrival_time_min = arrival_time
        self.edge_departure_time_min = float(now_min)
        self.current_edge_distance_m = float(distance_m)
        self.activity = RobotActivity.MOVING
        self.available = False

        return RobotNodeArrivalEvent(
            robot_id=self.spec.id,
            from_node=self.node_id,
            node_id=target,
            time_min=arrival_time,
        )

    def arrive_at_next_node(self, event: RobotNodeArrivalEvent) -> None:
        """Apply a node-arrival event and open a new rerouting decision point."""

        if not self.is_moving:
            raise ValueError("robot is not moving")
        if event.robot_id != self.spec.id:
            raise ValueError("arrival event belongs to a different robot")
        if event.from_node != self.node_id or event.node_id != self.next_node:
            raise ValueError("arrival event does not match the committed edge")
        assert self.next_node_arrival_time_min is not None
        if not math.isclose(
            event.time_min,
            self.next_node_arrival_time_min,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("arrival event time does not match committed ETA")

        assert self.current_edge_distance_m is not None
        energy_wh = self.current_edge_distance_m * self.spec.energy_per_meter_wh
        self.battery_wh -= energy_wh
        if self.battery_wh < -1e-7:
            raise RuntimeError("robot battery became negative while traversing an edge")
        self.battery_wh = max(0.0, self.battery_wh)

        self.node_id = self.next_node
        self.next_node = None
        self.next_node_arrival_time_min = None
        self.edge_departure_time_min = None
        self.current_edge_distance_m = None
        self.activity = RobotActivity.IDLE

        # Deliberately do not auto-depart. The simulator/policy gets a decision
        # point at every reached node and may change remaining_route first.

    def projected_travel_arrivals(
        self,
        graph: nx.Graph,
        now_min: float,
        *,
        edge_weight: str = "length",
    ) -> tuple[tuple[NodeId, float], ...]:
        """Travel-only ETAs for the currently planned nodes.

        The next-node ETA is committed and exact under the current edge travel
        time. Later ETAs are projections only: they assume immediate departure
        at each node and therefore intentionally exclude future charging/waiting
        delays and may change after rerouting.
        """

        result: list[tuple[NodeId, float]] = []
        if self.is_moving:
            assert self.next_node is not None
            assert self.next_node_arrival_time_min is not None
            node = self.next_node
            time_min = self.next_node_arrival_time_min
            result.append((node, float(time_min)))
        else:
            node = self.node_id
            time_min = float(now_min)

        for target in self.remaining_route:
            distance_m = _edge_distance_m(graph, node, target, edge_weight)
            time_min += distance_m / self.spec.speed_mps / 60.0
            result.append((target, float(time_min)))
            node = target

        return tuple(result)

    def clear_movement_plan(self) -> None:
        """Clear committed/planned movement, e.g. after terminal completion."""

        self.next_node = None
        self.next_node_arrival_time_min = None
        self.edge_departure_time_min = None
        self.current_edge_distance_m = None
        self.remaining_route.clear()
        self.activity = RobotActivity.IDLE
