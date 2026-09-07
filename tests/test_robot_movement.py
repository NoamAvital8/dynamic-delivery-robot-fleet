import math

import networkx as nx

from delivery_fleet.robot import RobotActivity, RobotSpec, RobotState


def movement_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(0, 1, length=600.0)
    graph.add_edge(1, 2, length=600.0)
    graph.add_edge(2, 5, length=600.0)
    graph.add_edge(1, 3, length=900.0)
    graph.add_edge(3, 4, length=300.0)
    return graph


def spec() -> RobotSpec:
    # 400 Wh / 0.1 Wh/m = 4 km full-battery range.
    return RobotSpec(
        id=7,
        speed_mps=2.0,
        max_payload_kg=10.0,
        max_volume_l=20.0,
        battery_capacity_wh=400.0,
        energy_per_meter_wh=0.1,
    )


def test_movement_commits_only_next_edge_and_records_eta() -> None:
    graph = movement_graph()
    robot = RobotState.fully_charged(spec(), node_id=0)
    robot.set_planned_path([0, 1, 2, 5])

    event = robot.depart_next_edge(graph, now_min=10.0)

    assert event is not None
    assert robot.activity is RobotActivity.MOVING
    assert robot.node_id == 0  # last node actually reached
    assert robot.next_node == 1
    assert robot.remaining_route == [2, 5]
    # 600 m / 2 m/s = 300 s = 5 min.
    assert math.isclose(robot.next_node_arrival_time_min, 15.0)
    assert math.isclose(event.time_min, 15.0)
    assert robot.decision_node == 1
    assert math.isclose(robot.battery_at_decision_node_wh, 340.0)


def test_mid_edge_replan_keeps_committed_edge_and_changes_only_future_path() -> None:
    graph = movement_graph()
    robot = RobotState.fully_charged(spec(), node_id=0)
    robot.set_planned_path([0, 1, 2, 5])
    event = robot.depart_next_edge(graph, now_min=10.0)
    assert event is not None

    committed_eta = robot.next_node_arrival_time_min
    robot.replan_from_decision_node([1, 3, 4])

    # The robot cannot turn around inside 0->1.
    assert robot.node_id == 0
    assert robot.next_node == 1
    assert robot.next_node_arrival_time_min == committed_eta
    assert robot.remaining_route == [3, 4]

    # When the event fires, it reaches node 1 and battery is charged for exactly
    # the traversed 600 m edge. The new route may then take effect.
    robot.arrive_at_next_node(event)
    assert robot.node_id == 1
    assert robot.next_node is None
    assert robot.activity is RobotActivity.IDLE
    assert robot.remaining_route == [3, 4]
    assert math.isclose(robot.battery_wh, 340.0)

    next_event = robot.depart_next_edge(graph, now_min=event.time_min)
    assert next_event is not None
    assert robot.next_node == 3
    # 900 m / 2 m/s = 7.5 min.
    assert math.isclose(next_event.time_min, 22.5)


def test_projected_arrivals_distinguish_committed_eta_from_future_projections() -> None:
    graph = movement_graph()
    robot = RobotState.fully_charged(spec(), node_id=0)
    robot.set_planned_path([0, 1, 2, 5])
    robot.depart_next_edge(graph, now_min=10.0)

    arrivals = robot.projected_travel_arrivals(graph, now_min=12.0)

    # The first ETA remains the committed 15.0 even though the query happens at
    # time 12. Later values are travel-only projections and may be rerouted.
    assert arrivals == ((1, 15.0), (2, 20.0), (5, 25.0))
