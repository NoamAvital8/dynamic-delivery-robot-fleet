"""Read-only, joint FIFO completion of frozen itineraries; no dispatch engine.

Only already released work is represented. Travel/handling durations and charge
durations are fixed; queue-induced delays propagate through all subsequent
stops, including visits to other stations. Pending work is appended by the
backend's common deterministic completion rule before this calculation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import math


@dataclass(frozen=True)
class Step:
    kind: str
    minutes: float
    station: int | None = None
    dropoff: int | None = None

    def __post_init__(self):
        if self.kind not in ("travel", "service", "charge") or not math.isfinite(self.minutes) or self.minutes < 0:
            raise ValueError("Invalid fixed itinerary step")
        if self.kind == "charge" and self.station is None:
            raise ValueError("Charge step requires a station")
        if self.dropoff is not None and self.kind != "service":
            raise ValueError("Only service completion may deliver an order")


@dataclass(frozen=True)
class Session:
    robot: int
    station: int
    finish: float

    def __post_init__(self):
        if not math.isfinite(self.finish) or self.finish < 0:
            raise ValueError("Invalid active-charge finish time")


@dataclass(frozen=True)
class Waiting:
    robot: int
    station: int
    minutes: float
    arrival: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.minutes,self.arrival)):
            raise ValueError("Invalid current FIFO request")


@dataclass(frozen=True)
class OrderCost:
    request: float
    deadline: float
    importance: float

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.request,self.deadline,self.importance))
                or self.request < 0 or self.deadline < self.request or self.importance <= 0):
            raise ValueError("Invalid released order cost")

    def loss(self, finished):
        elapsed, allowed = finished - self.request, self.deadline - self.request
        if elapsed < 0 or allowed < 0 or self.importance <= 0:
            raise ValueError("Invalid request/deadline/completion")
        return self.importance * min(elapsed, allowed) + (self.importance + 1)**2 * max(0., elapsed - allowed)


@dataclass(frozen=True)
class Result:
    loss: float
    completed: dict[int, float]
    robot_ready: dict[int, float]
    queue_wait_minutes: float


def project(now, plans, ports, orders, active=(), waiting=()):
    """Compute each remaining dropoff once; never mutate any supplied object."""
    if not math.isfinite(now) or now < 0 or any(int(p) != p or p < 1 for p in ports.values()):
        raise ValueError("Invalid projection clock or station ports")
    plans = {rid: tuple(steps) for rid, steps in plans.items()}
    positions = dict.fromkeys(plans, 0)
    busy = {station: set() for station in ports}
    queues = {station: deque() for station in ports}
    blocked, completed, ready = set(), {}, {}
    events, sequence = [], itertools.count()
    queue_wait = 0.

    def event(time, priority, kind, rid, station=None, dropoff=None):
        if not math.isfinite(time) or time < now - 1e-8:
            raise ValueError("Invalid projected event time")
        heapq.heappush(events, (time, priority, next(sequence), kind, rid, station, dropoff))

    def fill(station, time):
        nonlocal queue_wait
        while queues[station] and len(busy[station]) < ports[station]:
            rid, duration, arrival = queues[station].popleft()
            queue_wait += max(0., time - arrival)
            busy[station].add(rid)
            event(time + duration, 0, "charged", rid, station)

    for session in active:
        if session.robot not in plans or session.station not in ports or session.robot in blocked:
            raise ValueError("Invalid/duplicate active charging session")
        busy[session.station].add(session.robot)
        blocked.add(session.robot)
        event(max(now, session.finish), 0, "charged", session.robot, session.station)
    if any(len(busy[s]) > ports[s] for s in ports):
        raise ValueError("Active charging sessions exceed physical ports")
    for request in waiting:
        if (request.robot not in plans or request.station not in ports or request.robot in blocked
                or request.arrival > now or not math.isfinite(request.minutes) or request.minutes < 0):
            raise ValueError("Invalid/duplicate current FIFO request")
        blocked.add(request.robot)
        queues[request.station].append((request.robot, request.minutes, now))
    for station in ports:
        fill(station, now)
    for rid in sorted(plans):
        if rid not in blocked:
            event(now, 1, "advance", rid)
    processed = 0
    while events:
        time, _, _, kind, rid, station, dropoff = heapq.heappop(events)
        processed += 1
        if processed > 10_000_000:
            raise ValueError("Projection event bound exceeded")
        if kind == "charged":
            busy[station].remove(rid)
            fill(station, time)  # Existing FIFO waiters precede a new request.
        if dropoff is not None:
            if dropoff in completed or dropoff not in orders:
                raise ValueError("Duplicate or unknown projected dropoff")
            completed[dropoff] = time
        index = positions[rid]
        if index == len(plans[rid]):
            ready[rid] = time
            continue
        step = plans[rid][index]
        positions[rid] += 1
        if step.kind == "charge":
            if step.station not in ports:
                raise ValueError("Itinerary charges at an unknown station")
            queues[step.station].append((rid, step.minutes, time))
            fill(step.station, time)
        else:
            event(time + step.minutes, 1, "advance", rid, dropoff=step.dropoff)
    if set(completed) != set(orders) or set(ready) != set(plans):
        raise ValueError("Incomplete projected work must never receive a low-loss label")
    loss = sum(orders[key].loss(finish) for key, finish in completed.items())
    if not math.isfinite(loss):
        raise ValueError("Non-finite projected loss")
    return Result(loss, completed, ready, queue_wait)
