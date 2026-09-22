"""Priority-reservation allocation for a heterogeneous robot fleet.

The learned policy predicts a probability vector over priority thresholds for
each robot type.  This module converts those fractions to exact integer counts
and selects the physical robots with the smallest demand-weighted response
scores.  Larger numerical importance values are more important.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

from .robot import RobotState


RobotTypeKey = Hashable


@dataclass(frozen=True, slots=True)
class ReservationAssignment:
    """Concrete reservation thresholds and their realized integer counts."""

    threshold_by_robot_id: dict[int, float]
    counts_by_type: dict[RobotTypeKey, dict[float, int]]
    general_service_robot_id: int | None

    def permits(self, robot_id: int, importance: float) -> bool:
        """Return whether ``robot_id`` may serve the given importance."""

        try:
            threshold = self.threshold_by_robot_id[int(robot_id)]
        except KeyError as exc:
            raise KeyError(f"robot {robot_id!r} has no reservation assignment") from exc
        return float(importance) >= threshold


def _validate_levels(importance_levels: Sequence[float]) -> tuple[float, ...]:
    levels = tuple(sorted(float(value) for value in importance_levels))
    if not levels or any(value <= 0 or not math.isfinite(value) for value in levels):
        raise ValueError("importance levels must be finite and positive")
    if len(set(levels)) != len(levels):
        raise ValueError("importance levels must be unique")
    return levels


def apportion_reservation_counts(
    fractions_by_type: Mapping[RobotTypeKey, Sequence[float]],
    fleet_counts_by_type: Mapping[RobotTypeKey, int],
    importance_levels: Sequence[float],
) -> dict[RobotTypeKey, dict[float, int]]:
    """Convert per-type reservation fractions into exact fleet counts.

    Largest-remainder apportionment is applied independently to every robot
    type, so every type's integer counts sum exactly to its fleet size.
    Fractions describe disjoint strata (one threshold per robot), not
    cumulative reservations.
    """

    levels = _validate_levels(importance_levels)
    if set(fractions_by_type) != set(fleet_counts_by_type):
        raise ValueError("fraction and fleet-count mappings must have the same types")

    result: dict[RobotTypeKey, dict[float, int]] = {}
    for robot_type, raw_count in fleet_counts_by_type.items():
        count = int(raw_count)
        if count < 0:
            raise ValueError("fleet counts cannot be negative")
        fractions = tuple(float(value) for value in fractions_by_type[robot_type])
        if len(fractions) != len(levels):
            raise ValueError("every fraction row must match the importance levels")
        if any(value < 0 or not math.isfinite(value) for value in fractions):
            raise ValueError("reservation fractions must be finite and non-negative")
        total = sum(fractions)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("reservation fractions for each type must sum to one")

        exact = [count * value for value in fractions]
        allocated = [math.floor(value) for value in exact]
        remainder = count - sum(allocated)
        rank = sorted(
            range(len(levels)),
            key=lambda index: (-(exact[index] - allocated[index]), index),
        )
        for index in rank[:remainder]:
            allocated[index] += 1
        result[robot_type] = dict(zip(levels, allocated, strict=True))
    return result


def _robot_type(robot: RobotState) -> RobotTypeKey:
    robot_type = getattr(robot.spec, "robot_type", None)
    if robot_type is None:
        raise ValueError(f"robot {robot.spec.id} has no robot_type")
    return robot_type


def _capacity_rank(robot: RobotState) -> tuple[float, float, float, int]:
    return (
        float(robot.spec.max_payload_kg),
        float(robot.spec.max_volume_l),
        float(robot.spec.battery_capacity_wh),
        -int(robot.spec.id),
    )


def assign_reservation_thresholds(
    robots: Iterable[RobotState],
    counts_by_type: Mapping[RobotTypeKey, Mapping[float, int]],
    scores: Mapping[tuple[int, float], float],
    importance_levels: Sequence[float],
    *,
    keep_largest_capacity_general: bool = True,
) -> ReservationAssignment:
    """Choose the physical robots assigned to every priority stratum.

    Thresholds are processed from highest importance to lowest.  Within each
    robot type, the unassigned robots having the lowest ``H_reserve(r, c)``
    score receive threshold ``c``.  Any remainder belongs to the lowest,
    general-service threshold.

    When requested, one globally largest-capacity robot is protected for the
    general-service stratum.  It is excluded from restrictive strata even if
    the learned fraction would otherwise reserve every such robot.
    """

    levels = _validate_levels(importance_levels)
    general = levels[0]
    fleet = list(robots)
    if len({robot.spec.id for robot in fleet}) != len(fleet):
        raise ValueError("robot ids must be unique")

    by_type: dict[RobotTypeKey, list[RobotState]] = defaultdict(list)
    for robot in fleet:
        by_type[_robot_type(robot)].append(robot)
    if set(by_type) != set(counts_by_type):
        raise ValueError("reservation counts must cover exactly the fleet robot types")

    protected_id: int | None = None
    if keep_largest_capacity_general and fleet:
        protected_id = max(fleet, key=_capacity_rank).spec.id

    threshold_by_robot: dict[int, float] = {}
    for robot_type, type_robots in by_type.items():
        requested = {float(level): int(value) for level, value in counts_by_type[robot_type].items()}
        if set(requested) != set(levels):
            raise ValueError("reservation counts must contain every importance level")
        if any(value < 0 for value in requested.values()):
            raise ValueError("reservation counts cannot be negative")
        if sum(requested.values()) != len(type_robots):
            raise ValueError("reservation counts must sum to the number of robots per type")

        unassigned = {robot.spec.id: robot for robot in type_robots}
        for threshold in reversed(levels[1:]):
            desired = requested[threshold]
            candidates = [
                robot
                for robot in unassigned.values()
                if robot.spec.id != protected_id
            ]
            candidates.sort(
                key=lambda robot: (
                    float(scores.get((robot.spec.id, threshold), math.inf)),
                    robot.spec.id,
                )
            )
            for robot in candidates[:desired]:
                score = float(scores.get((robot.spec.id, threshold), math.inf))
                if not math.isfinite(score):
                    raise ValueError(
                        f"missing finite reservation score for robot {robot.spec.id}, "
                        f"threshold {threshold}"
                    )
                threshold_by_robot[robot.spec.id] = threshold
                del unassigned[robot.spec.id]

        for robot_id in sorted(unassigned):
            threshold_by_robot[robot_id] = general

    realized: dict[RobotTypeKey, dict[float, int]] = {
        robot_type: {level: 0 for level in levels} for robot_type in by_type
    }
    robot_by_id = {robot.spec.id: robot for robot in fleet}
    for robot_id, threshold in threshold_by_robot.items():
        realized[_robot_type(robot_by_id[robot_id])][threshold] += 1

    return ReservationAssignment(
        threshold_by_robot_id=threshold_by_robot,
        counts_by_type=realized,
        general_service_robot_id=protected_id,
    )


def reservation_eligibility(
    assignment: ReservationAssignment | None,
    robot_id: int,
    importance: float,
) -> bool:
    """Reservation feasibility predicate; ``None`` means reservation is off."""

    return assignment is None or assignment.permits(robot_id, importance)
