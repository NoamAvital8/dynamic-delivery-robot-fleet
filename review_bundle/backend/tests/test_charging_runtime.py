import math

from delivery_fleet.charging import ChargingStation
from delivery_fleet.charging_runtime import ChargingStationRuntime
from delivery_fleet.robot import RobotActivity, RobotSpec, RobotState


def robot(robot_id: int, *, current_order_id: int | None = None) -> RobotState:
    spec = RobotSpec(
        id=robot_id,
        speed_mps=4.0,
        max_payload_kg=10.0,
        max_volume_l=20.0,
        battery_capacity_wh=400.0,
        energy_per_meter_wh=0.1,
    )
    return RobotState(
        spec=spec,
        node_id=5,
        battery_wh=100.0,
        available=current_order_id is None,
        current_order_id=current_order_id,
    )


def test_two_ports_charge_independently_and_waiters_are_fifo() -> None:
    runtime = ChargingStationRuntime(
        ChargingStation(id=0, node_id=5, charging_power_w=2_000.0, number_of_ports=2)
    )
    r1, r2, r3, r4 = (robot(i) for i in range(1, 5))

    e1 = runtime.request_charge(r1, energy_added_wh=100.0, now_min=0.0)
    e2 = runtime.request_charge(r2, energy_added_wh=200.0, now_min=0.0)
    e3 = runtime.request_charge(r3, energy_added_wh=150.0, now_min=1.0)
    e4 = runtime.request_charge(r4, energy_added_wh=100.0, now_min=2.0)

    assert e1 is not None and math.isclose(e1.time_min, 3.0)
    assert e2 is not None and math.isclose(e2.time_min, 6.0)
    assert e3 is None
    assert e4 is None
    assert runtime.free_ports == 0
    assert runtime.active_robot_ids == (1, 2)
    assert runtime.waiting_robot_ids == (3, 4)
    assert r1.activity is RobotActivity.CHARGING
    assert r2.activity is RobotActivity.CHARGING
    assert r3.activity is RobotActivity.WAITING
    assert r4.activity is RobotActivity.WAITING

    # Robot 1 finishing frees exactly one port. Robot 2 keeps its original
    # charging plan, while the oldest waiter (robot 3) starts immediately.
    first = runtime.complete_charge(e1)
    assert first.completed_robot_id == 1
    assert first.next_event is not None
    assert first.next_event.robot_id == 3
    assert math.isclose(first.next_event.time_min, 7.5)
    assert runtime.active_robot_ids == (2, 3)
    assert runtime.waiting_robot_ids == (4,)
    assert math.isclose(e2.time_min, 6.0)
    assert math.isclose(r1.battery_wh, 200.0)
    assert r1.activity is RobotActivity.IDLE

    # Robot 2 then frees a port for robot 4. Robot 3 continues untouched.
    second = runtime.complete_charge(e2)
    assert second.next_event is not None
    assert second.next_event.robot_id == 4
    assert math.isclose(second.next_event.time_min, 9.0)
    assert runtime.active_robot_ids == (3, 4)
    assert runtime.waiting_robot_ids == ()
    assert r3.activity is RobotActivity.CHARGING
    assert r4.activity is RobotActivity.CHARGING


def test_robot_on_delivery_stays_unavailable_after_charging() -> None:
    runtime = ChargingStationRuntime(
        ChargingStation(id=0, node_id=5, charging_power_w=2_000.0, number_of_ports=2)
    )
    serving_robot = robot(1, current_order_id=77)

    event = runtime.request_charge(
        serving_robot,
        energy_added_wh=100.0,
        now_min=10.0,
    )
    assert event is not None
    runtime.complete_charge(event)

    assert serving_robot.activity is RobotActivity.IDLE
    assert not serving_robot.available
    assert serving_robot.current_order_id == 77
    assert math.isclose(serving_robot.battery_wh, 200.0)
