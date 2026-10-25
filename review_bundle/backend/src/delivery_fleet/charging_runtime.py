"""Runtime charging-port occupancy and FIFO waiting queues.

Each station has a fixed number of independent charging ports.  A robot that
arrives when every port is occupied waits at the station.  Charging robots are
not affected by waiting robots; when one charging session finishes, the oldest
waiting request immediately takes the newly freed port.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .charging import ChargingStation
from .robot import RobotActivity, RobotState

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class ChargingCompleteEvent:
    """Future event emitted when one robot's requested partial charge finishes."""

    station_id: int
    robot_id: int
    time_min: float
    energy_added_wh: float


@dataclass(frozen=True, slots=True)
class ChargingSession:
    station_id: int
    robot_id: int
    start_time_min: float
    finish_time_min: float
    energy_added_wh: float


@dataclass(slots=True)
class _QueuedCharge:
    robot: RobotState
    energy_added_wh: float
    arrival_time_min: float


@dataclass(frozen=True, slots=True)
class ChargeCompletionResult:
    """Result of freeing one charging port.

    ``next_event`` is populated only when a queued robot immediately takes the
    freed port.  Other active sessions at the station are deliberately left
    untouched.
    """

    completed_robot_id: int
    next_event: ChargingCompleteEvent | None


class ChargingStationRuntime:
    """Mutable runtime state for one charging station."""

    def __init__(self, station: ChargingStation) -> None:
        if station.charging_power_w <= 0:
            raise ValueError("charging power must be positive")
        if station.number_of_ports <= 0:
            raise ValueError("number_of_ports must be positive")
        self.station = station
        self._active: dict[int, tuple[RobotState, ChargingSession]] = {}
        self._waiting: deque[_QueuedCharge] = deque()

    @property
    def active_robot_ids(self) -> tuple[int, ...]:
        return tuple(self._active)

    @property
    def waiting_robot_ids(self) -> tuple[int, ...]:
        return tuple(request.robot.spec.id for request in self._waiting)

    @property
    def free_ports(self) -> int:
        return self.station.number_of_ports - len(self._active)

    def _validate_request(
        self,
        robot: RobotState,
        energy_added_wh: float,
        now_min: float,
    ) -> None:
        if now_min < 0:
            raise ValueError("now_min cannot be negative")
        if energy_added_wh <= 0:
            raise ValueError("energy_added_wh must be positive")
        if robot.node_id != self.station.node_id:
            raise ValueError("robot must be physically at the charging station")
        if robot.is_moving:
            raise ValueError("moving robot cannot start or queue for charging")
        if robot.spec.id in self._active or robot.spec.id in self.waiting_robot_ids:
            raise ValueError("robot already has a charging request at this station")
        if robot.activity in (RobotActivity.CHARGING, RobotActivity.WAITING):
            raise ValueError("robot is already charging or waiting")
        if robot.battery_wh + energy_added_wh > robot.spec.battery_capacity_wh + _EPS:
            raise ValueError("requested charge would exceed robot battery capacity")

    def _start_charge(
        self,
        robot: RobotState,
        energy_added_wh: float,
        now_min: float,
    ) -> ChargingCompleteEvent:
        if self.free_ports <= 0:
            raise RuntimeError("cannot start charging without a free port")

        duration_min = energy_added_wh / self.station.charging_power_w * 60.0
        finish_time = float(now_min + duration_min)
        session = ChargingSession(
            station_id=self.station.id,
            robot_id=robot.spec.id,
            start_time_min=float(now_min),
            finish_time_min=finish_time,
            energy_added_wh=float(energy_added_wh),
        )
        self._active[robot.spec.id] = (robot, session)
        robot.activity = RobotActivity.CHARGING
        robot.available = False

        return ChargingCompleteEvent(
            station_id=self.station.id,
            robot_id=robot.spec.id,
            time_min=finish_time,
            energy_added_wh=float(energy_added_wh),
        )

    def request_charge(
        self,
        robot: RobotState,
        energy_added_wh: float,
        now_min: float,
    ) -> ChargingCompleteEvent | None:
        """Start immediately if a port is free; otherwise join the FIFO queue."""

        self._validate_request(robot, energy_added_wh, now_min)
        if self.free_ports > 0:
            return self._start_charge(robot, energy_added_wh, now_min)

        robot.activity = RobotActivity.WAITING
        robot.available = False
        self._waiting.append(
            _QueuedCharge(
                robot=robot,
                energy_added_wh=float(energy_added_wh),
                arrival_time_min=float(now_min),
            )
        )
        return None

    def complete_charge(
        self,
        event: ChargingCompleteEvent,
    ) -> ChargeCompletionResult:
        """Finish one session and, if needed, start the oldest waiting robot."""

        if event.station_id != self.station.id:
            raise ValueError("charging event belongs to a different station")
        active = self._active.get(event.robot_id)
        if active is None:
            raise ValueError("robot is not actively charging at this station")

        robot, session = active
        if not math.isclose(
            event.time_min,
            session.finish_time_min,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("charging event time does not match active session")
        if not math.isclose(
            event.energy_added_wh,
            session.energy_added_wh,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("charging event energy does not match active session")

        robot.battery_wh += session.energy_added_wh
        if robot.battery_wh > robot.spec.battery_capacity_wh + 1e-7:
            raise RuntimeError("robot battery exceeded capacity after charging")
        robot.battery_wh = min(robot.battery_wh, robot.spec.battery_capacity_wh)
        robot.activity = RobotActivity.IDLE
        robot.available = robot.current_order_id is None
        del self._active[event.robot_id]

        next_event = None
        if self._waiting:
            request = self._waiting.popleft()
            # The robot has already waited at this node; starting the newly free
            # port does not alter any of the station's other active sessions.
            request.robot.activity = RobotActivity.IDLE
            next_event = self._start_charge(
                request.robot,
                request.energy_added_wh,
                event.time_min,
            )

        return ChargeCompletionResult(
            completed_robot_id=event.robot_id,
            next_event=next_event,
        )


class ChargingNetworkRuntime:
    """Collection of independent station runtimes keyed by station id/node."""

    def __init__(self, stations: tuple[ChargingStation, ...] | list[ChargingStation]) -> None:
        stations = tuple(stations)
        if not stations:
            raise ValueError("at least one charging station is required")
        if len({station.id for station in stations}) != len(stations):
            raise ValueError("charging station ids must be unique")
        if len({station.node_id for station in stations}) != len(stations):
            raise ValueError("charging station nodes must be unique")

        self.by_id = {
            station.id: ChargingStationRuntime(station) for station in stations
        }
        self.station_id_by_node = {
            station.node_id: station.id for station in stations
        }

    def at_node(self, node_id) -> ChargingStationRuntime:
        try:
            station_id = self.station_id_by_node[node_id]
        except KeyError as exc:
            raise ValueError("node is not a charging station") from exc
        return self.by_id[station_id]
