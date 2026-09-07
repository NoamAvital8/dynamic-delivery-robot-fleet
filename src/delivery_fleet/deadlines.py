"""Distance- and importance-dependent delivery deadlines.

V1 deadlines are relative to request release time and use the direct graph
shortest-path distance between pickup and drop-off:

    deadline = request_time
             + direct_distance / reference_speed
             + fixed_buffer
             + importance_slack / sqrt(importance)

Distance is measured in meters, speed in meters/second, and all returned times
are in minutes.
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
