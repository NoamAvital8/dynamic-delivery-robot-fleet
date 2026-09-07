"""Robot capabilities and mutable runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .defaults import MIN_ROBOT_FULL_BATTERY_RANGE_M
from .scenario_creator import Item

NodeId = Hashable


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
        if self.full_battery_range_m + 1e-9 < MIN_ROBOT_FULL_BATTERY_RANGE_M:
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


@dataclass(slots=True)
class RobotState:
    """Mutable simulation state for one robot.

    The benchmark policy handles only one delivery at a time, so an available
    robot carries no active order. More complex cargo/route state can be added
    later for insertion policies.
    """

    spec: RobotSpec
    node_id: NodeId
    battery_wh: float
    available: bool = True
    current_order_id: int | None = None

    def __post_init__(self) -> None:
        if self.battery_wh < -1e-9:
            raise ValueError("battery_wh cannot be negative")
        if self.battery_wh > self.spec.battery_capacity_wh + 1e-9:
            raise ValueError("battery_wh cannot exceed battery capacity")
        if self.available and self.current_order_id is not None:
            raise ValueError("available robot cannot have a current order")
        if not self.available and self.current_order_id is None:
            raise ValueError("busy robot must have a current order")

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

    def can_hold(self, item: Item) -> bool:
        return self.spec.can_hold(item)
