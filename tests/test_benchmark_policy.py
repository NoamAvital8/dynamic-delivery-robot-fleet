import math

import networkx as nx
import pytest

from delivery_fleet.battery_routing import BatteryFeasibleRouter
from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.policies import NearestAvailableRobotPolicy
from delivery_fleet.robot import RobotSpec, RobotState
from delivery_fleet.scenario_creator import Item, Order


def line_graph() -> nx.Graph:
    graph = nx.path_graph(9)
    nx.set_edge_attributes(graph, 1000.0, "length")
    for node in graph.nodes:
        graph.nodes[node]["is_charging_station"] = node in {0, 4, 8}
    annotate_nearest_charging_stations(graph, [0, 4, 8])
    return graph


def robot_spec(robot_id: int, *, payload: float = 10.0) -> RobotSpec:
    # 400 Wh / 0.1 Wh/m = 4 km full-battery range.
    return RobotSpec(
        id=robot_id,
        speed_mps=1.0,
        max_payload_kg=payload,
        max_volume_l=20.0,
        battery_capacity_wh=400.0,
        energy_per_meter_wh=0.1,
    )


def test_robot_rejects_range_below_four_km() -> None:
    with pytest.raises(ValueError):
        RobotSpec(
            id=1,
            speed_mps=1.0,
            max_payload_kg=10.0,
            max_volume_l=20.0,
            battery_capacity_wh=399.0,
            energy_per_meter_wh=0.1,
        )


def test_router_uses_partial_charge_and_keeps_dropoff_reserve() -> None:
    graph = line_graph()
    robot = RobotState(
        spec=robot_spec(1),
        node_id=2,
        battery_wh=200.0,  # 2 km remaining range
    )
    router = BatteryFeasibleRouter(graph)

    route = router.plan(robot, pickup_node=3, dropoff_node=5)

    # Direct 2->3->5 plus a 1 km safety reserve would need 4 km, but the robot
    # starts with only 2 km. The optimal feasible route is 2->3->4(charge)->5.
    assert route.node_path == (2, 3, 4, 5)
    assert route.total_distance_m == 3000.0
    assert len(route.charging_events) == 1

    charge = route.charging_events[0]
    assert charge.node_id == 4
    assert math.isclose(charge.energy_added_wh, 200.0)
    assert charge.energy_added_wh < robot.spec.battery_capacity_wh
    # 2 kW = 33.333... Wh/min, so 200 Wh takes 6 minutes.
    assert math.isclose(charge.duration_min, 6.0)

    # Arrival at node 5 keeps exactly enough energy for the nearest charger at 4.
    assert math.isclose(route.required_dropoff_reserve_wh, 100.0)
    assert math.isclose(route.arrival_battery_wh, 100.0)
    assert math.isclose(route.travel_time_min, 50.0)
    assert math.isclose(route.total_time_min, 56.0)


def test_router_avoids_charging_when_current_battery_is_enough() -> None:
    graph = line_graph()
    robot = RobotState.fully_charged(robot_spec(1), node_id=2)
    router = BatteryFeasibleRouter(graph)

    route = router.plan(robot, pickup_node=3, dropoff_node=5)

    assert route.node_path == (2, 3, 4, 5)
    assert route.charging_events == ()
    assert math.isclose(route.arrival_battery_wh, 100.0)


def test_policy_chooses_nearest_available_capable_robot() -> None:
    graph = line_graph()
    policy = NearestAvailableRobotPolicy(graph)

    nearest_but_too_small = RobotState.fully_charged(
        robot_spec(1, payload=1.0), node_id=2
    )
    chosen = RobotState.fully_charged(robot_spec(2), node_id=1)
    farther = RobotState.fully_charged(robot_spec(3), node_id=0)

    order = Order(
        id=7,
        pickup_node=3,
        dropoff_node=5,
        request_time_min=10.0,
        item=Item(weight_kg=2.0, volume_l=3.0),
        importance=1.0,
    )

    assignment = policy.on_order(
        order,
        [nearest_but_too_small, chosen, farther],
    )

    assert assignment is not None
    assert assignment.robot_id == 2
    assert not chosen.available
    assert chosen.current_order_id == order.id
    assert nearest_but_too_small.available
    assert farther.available

    policy.complete_delivery(assignment, chosen)
    assert chosen.available
    assert chosen.current_order_id is None
    assert chosen.node_id == order.dropoff_node
    assert math.isclose(chosen.battery_wh, assignment.route.arrival_battery_wh)


def test_policy_returns_none_if_no_capable_available_robot() -> None:
    graph = line_graph()
    policy = NearestAvailableRobotPolicy(graph)
    robot = RobotState.fully_charged(robot_spec(1, payload=1.0), node_id=2)

    order = Order(
        id=8,
        pickup_node=3,
        dropoff_node=5,
        request_time_min=10.0,
        item=Item(weight_kg=2.0, volume_l=3.0),
        importance=1.0,
    )

    assert policy.on_order(order, [robot]) is None
    assert robot.available
