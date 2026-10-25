"""Side-effect-free, rolling FIFO forecasts for identical charging ports.

Future requests are predicted arrivals, not reservations. A later arrival never
blocks an earlier candidate merely because its charge would overlap a forecast.
Each hypothetical route gets a private projection; no live queue is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Hashable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ActiveChargingForecast:
    robot_id: int
    station_node: Hashable
    finish_min: float


@dataclass(frozen=True, slots=True)
class KnownChargingRequest:
    robot_id: int
    station_node: Hashable
    arrival_min: float
    duration_min: float
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class QueuePrediction:
    station_node: Hashable
    arrival_min: float
    start_min: float
    finish_min: float

    @property
    def wait_min(self) -> float:
        return max(0.0, self.start_min - self.arrival_min)


class KnownChargingForecast:
    """Immutable observed occupancy plus known queued/predicted requests.

    ETA ties put existing requests before the candidate conservatively. Excluding
    the candidate's own old requests/session models cancellation on reassignment.
    Forecasts use only released orders and already-committed idle movements.
    """

    def __init__(
        self, now_min: float, ports_by_station: Mapping[Hashable, int],
        active: Iterable[ActiveChargingForecast] = (),
        requests: Iterable[KnownChargingRequest] = (),
    ) -> None:
        if not math.isfinite(now_min) or now_min < 0:
            raise ValueError("forecast time must be finite and non-negative")
        self.now_min = float(now_min)
        self.ports = dict(ports_by_station)
        if any(int(n) != n or n <= 0 for n in self.ports.values()):
            raise ValueError("stations must have a positive integer port count")
        self.active: dict[Hashable, tuple[ActiveChargingForecast, ...]] = {}
        self.requests: dict[Hashable, tuple[KnownChargingRequest, ...]] = {}
        active_lists = {station: [] for station in self.ports}
        request_lists = {station: [] for station in self.ports}
        for session in active:
            if session.station_node not in self.ports or not math.isfinite(session.finish_min):
                raise ValueError("invalid active charging session")
            active_lists[session.station_node].append(session)
        for request in requests:
            if (request.station_node not in self.ports
                    or not math.isfinite(request.arrival_min)
                    or request.arrival_min < 0
                    or not math.isfinite(request.duration_min)
                    or request.duration_min <= 0):
                raise ValueError("invalid known charging request")
            request_lists[request.station_node].append(request)
        for station, count in self.ports.items():
            if len(active_lists[station]) > count:
                raise ValueError("active sessions exceed the physical port count")
            self.active[station] = tuple(active_lists[station])
            self.requests[station] = tuple(sorted(
                request_lists[station],
                key=lambda request: (request.arrival_min, request.sequence, request.robot_id),
            ))
        self._views: dict[tuple[int, Hashable], tuple[tuple[float, ...], tuple[KnownChargingRequest, ...]]] = {}

    def project(self, robot_id: int) -> "QueueProjection":
        return QueueProjection(self, int(robot_id))

    def _view(self, robot_id: int, station: Hashable):
        key = (robot_id, station)
        if key not in self._views:
            sessions = [s for s in self.active[station] if s.robot_id != robot_id]
            free = [max(self.now_min, s.finish_min) for s in sessions]
            free.extend([self.now_min] * (self.ports[station] - len(sessions)))
            requests = tuple(r for r in self.requests[station] if r.robot_id != robot_id)
            self._views[key] = (tuple(free), requests)
        return self._views[key]


class QueueProjection:
    """Private FIFO replay, including earlier visits of this candidate route."""

    def __init__(self, forecast: KnownChargingForecast, robot_id: int) -> None:
        self.forecast = forecast
        self.robot_id = robot_id
        self._states: dict[Hashable, tuple[list[float], int, float]] = {}

    def clone(self) -> "QueueProjection":
        result = QueueProjection(self.forecast, self.robot_id)
        result._states = {
            station: (list(free), cursor, last_arrival)
            for station, (free, cursor, last_arrival) in self._states.items()
        }
        return result

    def _before(self, station: Hashable, arrival_min: float) -> list[float]:
        if not math.isfinite(arrival_min) or arrival_min < self.forecast.now_min - 1e-8:
            raise ValueError("candidate arrival must not precede forecast time")
        base_free, requests = self.forecast._view(self.robot_id, station)
        if station not in self._states:
            free = list(base_free)
            heapq.heapify(free)
            self._states[station] = (free, 0, self.forecast.now_min)
        free, cursor, last_arrival = self._states[station]
        if arrival_min < last_arrival - 1e-8:
            raise ValueError("a candidate's visits must be chronological")
        while cursor < len(requests) and requests[cursor].arrival_min <= arrival_min:
            request = requests[cursor]
            available = heapq.heappop(free)
            start = max(self.forecast.now_min, request.arrival_min, available)
            heapq.heappush(free, start + request.duration_min)
            cursor += 1
        self._states[station] = (free, cursor, arrival_min)
        return free

    def wait_at(self, station: Hashable, arrival_min: float) -> float:
        """Read-only preview for alternative-station ranking."""
        copy = self.clone()
        return max(0.0, min(copy._before(station, arrival_min)) - arrival_min)

    def serve(self, station: Hashable, arrival_min: float, duration_min: float) -> QueuePrediction:
        if not math.isfinite(duration_min) or duration_min <= 0:
            raise ValueError("charging duration must be finite and positive")
        free = self._before(station, arrival_min)
        start = max(arrival_min, heapq.heappop(free))
        finish = start + duration_min
        heapq.heappush(free, finish)
        return QueuePrediction(station, arrival_min, start, finish)
