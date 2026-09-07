import math

import pytest

from delivery_fleet.deadlines import (
    DEADLINE_FIXED_BUFFER_MIN,
    DEADLINE_IMPORTANCE_SLACK_MIN,
    DEADLINE_REFERENCE_SPEED_MPS,
    delivery_deadline_min,
    delivery_time_allowance_min,
)


def test_default_deadline_parameters() -> None:
    assert DEADLINE_REFERENCE_SPEED_MPS == 4.0
    assert DEADLINE_FIXED_BUFFER_MIN == 10.0
    assert DEADLINE_IMPORTANCE_SLACK_MIN == 60.0


def test_three_km_allowances_match_reference_values() -> None:
    assert math.isclose(delivery_time_allowance_min(3_000.0, 1.0), 82.5)
    assert math.isclose(
        delivery_time_allowance_min(3_000.0, 2.0),
        12.5 + 10.0 + 60.0 / math.sqrt(2.0),
    )
    assert math.isclose(
        delivery_time_allowance_min(3_000.0, 5.0),
        12.5 + 10.0 + 60.0 / math.sqrt(5.0),
    )


def test_deadline_is_absolute_simulation_time() -> None:
    request_time_min = 137.25
    allowance = delivery_time_allowance_min(10_000.0, 5.0)

    assert math.isclose(
        delivery_deadline_min(request_time_min, 10_000.0, 5.0),
        request_time_min + allowance,
    )


def test_more_important_orders_have_tighter_deadlines() -> None:
    normal = delivery_time_allowance_min(5_000.0, 1.0)
    important = delivery_time_allowance_min(5_000.0, 2.0)
    urgent = delivery_time_allowance_min(5_000.0, 5.0)

    assert urgent < important < normal


def test_invalid_deadline_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        delivery_time_allowance_min(-1.0, 1.0)
    with pytest.raises(ValueError):
        delivery_time_allowance_min(1_000.0, 0.0)
    with pytest.raises(ValueError):
        delivery_deadline_min(-0.1, 1_000.0, 1.0)
