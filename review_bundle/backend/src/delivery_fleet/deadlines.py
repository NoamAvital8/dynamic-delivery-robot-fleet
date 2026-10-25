"""Distance- and importance-dependent delivery deadlines and loss.

V1 deadlines are relative to request release time and use the direct graph
shortest-path distance between pickup and drop-off:

    deadline = request_time
             + direct_distance / reference_speed
             + fixed_buffer
             + importance_slack / sqrt(importance)

Distance is measured in meters, speed in meters/second, and all returned times
are in minutes.

The delivery loss uses elapsed request-to-delivery time ``T`` and the relative
deadline allowance ``D`` (not the absolute simulation-time deadline):

    loss = w * min(D, T) + (w + 1)^2 * max(0, T - D)

Thus every minute up to the deadline has the regular importance weight ``w``,
while every minute after the deadline has the larger weight ``(w + 1)^2``.
"""

from __future__ import annotations

import math


DEADLINE_REFERENCE_SPEED_MPS = 4.0
DEADLINE_FIXED_BUFFER_MIN = 10.0
DEADLINE_IMPORTANCE_SLACK_MIN = 60.0


def delivery_time_allowance_min(
    direct_distance_m: float,
    importance: float,
    *,
    reference_speed_mps: float = DEADLINE_REFERENCE_SPEED_MPS,
    fixed_buffer_min: float = DEADLINE_FIXED_BUFFER_MIN,
    importance_slack_min: float = DEADLINE_IMPORTANCE_SLACK_MIN,
) -> float:
    """Return minutes allowed from request release until delivery.

    ``direct_distance_m`` should be the graph shortest-path distance from the
    order pickup node to its drop-off node. Importance affects only the extra
    slack; it never changes the unavoidable distance-dependent travel term.
    """

    if direct_distance_m < 0:
        raise ValueError("direct_distance_m cannot be negative")
    if importance <= 0:
        raise ValueError("importance must be positive")
    if reference_speed_mps <= 0:
        raise ValueError("reference_speed_mps must be positive")
    if fixed_buffer_min < 0:
        raise ValueError("fixed_buffer_min cannot be negative")
    if importance_slack_min < 0:
        raise ValueError("importance_slack_min cannot be negative")

    direct_travel_min = direct_distance_m / reference_speed_mps / 60.0
    urgency_slack_min = importance_slack_min / math.sqrt(importance)
    return direct_travel_min + fixed_buffer_min + urgency_slack_min


def delivery_deadline_min(
    request_time_min: float,
    direct_distance_m: float,
    importance: float,
    *,
    reference_speed_mps: float = DEADLINE_REFERENCE_SPEED_MPS,
    fixed_buffer_min: float = DEADLINE_FIXED_BUFFER_MIN,
    importance_slack_min: float = DEADLINE_IMPORTANCE_SLACK_MIN,
) -> float:
    """Return the absolute simulation-time deadline for one order."""

    if request_time_min < 0:
        raise ValueError("request_time_min cannot be negative")

    return request_time_min + delivery_time_allowance_min(
        direct_distance_m,
        importance,
        reference_speed_mps=reference_speed_mps,
        fixed_buffer_min=fixed_buffer_min,
        importance_slack_min=importance_slack_min,
    )


def delivery_loss(
    delivery_time_min: float,
    deadline_allowance_min: float,
    importance: float,
) -> float:
    """Return deadline-aware loss for one delivered order.

    ``delivery_time_min`` is elapsed request-to-delivery time ``T`` and
    ``deadline_allowance_min`` is the allowed request-to-delivery duration ``D``.
    The first ``min(D, T)`` minutes cost ``importance`` per minute. Any lateness
    beyond ``D`` costs ``(importance + 1)^2`` per minute.
    """

    if delivery_time_min < 0:
        raise ValueError("delivery_time_min cannot be negative")
    if deadline_allowance_min < 0:
        raise ValueError("deadline_allowance_min cannot be negative")
    if importance <= 0:
        raise ValueError("importance must be positive")

    regular_minutes = min(deadline_allowance_min, delivery_time_min)
    late_minutes = max(0.0, delivery_time_min - deadline_allowance_min)
    return (
        importance * regular_minutes
        + (importance + 1.0) ** 2 * late_minutes
    )
