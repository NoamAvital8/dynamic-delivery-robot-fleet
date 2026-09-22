from __future__ import annotations

import networkx as nx
import numpy as np

from delivery_fleet.anticipatory_policy import (
    OnlineAnticipatoryReservation,
    ReservationRobotSnapshot,
)
from delivery_fleet.fleet import Enduro, MiddleMan, Oomph, RobotType, SpeedyMcQueen
from delivery_fleet.reservation import (
    apportion_reservation_counts,
    assign_reservation_thresholds,
    reservation_eligibility,
)
from delivery_fleet.robot import RobotState
from delivery_fleet.reservation_nn import ReservationFCNN, ReservationFeatureSchema


LEVELS = (1.0, 2.0, 5.0)


def _robot(spec, node: int) -> RobotState:
    return RobotState.fully_charged(spec, node)


def test_largest_remainder_apportionment_preserves_each_type_total() -> None:
    fractions = {
        RobotType.MIDDLE_MAN: (0.34, 0.33, 0.33),
        RobotType.ENDURO: (0.1, 0.2, 0.7),
    }
    counts = apportion_reservation_counts(
        fractions,
        {RobotType.MIDDLE_MAN: 5, RobotType.ENDURO: 3},
        LEVELS,
    )
    assert counts[RobotType.MIDDLE_MAN] == {1.0: 2, 2.0: 2, 5.0: 1}
    assert sum(counts[RobotType.ENDURO].values()) == 3


def test_physical_robots_are_selected_high_priority_first_by_score() -> None:
    robots = [_robot(MiddleMan(index), index) for index in range(4)]
    counts = {RobotType.MIDDLE_MAN: {1.0: 1, 2.0: 1, 5.0: 2}}
    scores = {
        (0, 5.0): 8.0,
        (1, 5.0): 2.0,
        (2, 5.0): 1.0,
        (3, 5.0): 9.0,
        (0, 2.0): 3.0,
        (3, 2.0): 1.0,
    }
    assignment = assign_reservation_thresholds(
        robots, counts, scores, LEVELS, keep_largest_capacity_general=False
    )
    assert assignment.threshold_by_robot_id == {2: 5.0, 1: 5.0, 3: 2.0, 0: 1.0}
    assert reservation_eligibility(assignment, 2, 5.0)
    assert not reservation_eligibility(assignment, 2, 2.0)


def test_largest_capacity_robot_is_kept_for_general_service() -> None:
    robots = [
        _robot(SpeedyMcQueen(0), 0),
        _robot(Enduro(1), 1),
        _robot(Oomph(2), 2),
    ]
    counts = {
        RobotType.SPEEDY_MCQUEEN: {1.0: 0, 2.0: 0, 5.0: 1},
        RobotType.ENDURO: {1.0: 0, 2.0: 0, 5.0: 1},
        RobotType.OOMPH: {1.0: 0, 2.0: 0, 5.0: 1},
    }
    scores = {(robot.spec.id, level): 1.0 for robot in robots for level in LEVELS[1:]}
    assignment = assign_reservation_thresholds(robots, counts, scores, LEVELS)
    assert assignment.general_service_robot_id == 2
    assert assignment.threshold_by_robot_id[2] == 1.0
    assert assignment.counts_by_type[RobotType.OOMPH] == {1.0: 1, 2.0: 0, 5.0: 0}


def test_online_pipeline_updates_posterior_and_produces_assignment() -> None:
    graph = nx.Graph()
    graph.add_node(0, x=34.0, y=32.0, in_cluster=0, is_cluster_representative=True)
    graph.add_node(1, x=34.001, y=32.0, in_cluster=1, is_cluster_representative=True)
    graph.add_edge(0, 1, length=100.0)
    robots = [
        _robot(SpeedyMcQueen(0), 0),
        _robot(MiddleMan(1), 0),
        _robot(Enduro(2), 1),
        _robot(Oomph(3), 1),
    ]
    names = tuple(robot_type.value for robot_type in RobotType)
    schema = ReservationFeatureSchema(names, LEVELS)
    model = ReservationFCNN(len(schema.names), names, LEVELS, hidden_dim=4)
    model.w1.fill(0.0)
    model.w2.fill(0.0)
    model.b2[:] = np.tile(np.asarray((0.0, 0.0, 2.0)), len(names))
    policy = OnlineAnticipatoryReservation(
        graph,
        robots,
        model,
        {1.0: 60.0, 2.0: 15.0, 5.0: 5.0},
        horizon_min=60.0,
        charger_power_w=1_000.0,
    )
    policy.observe(0, 5.0, 10.0)
    snapshots = {
        robot.spec.id: ReservationRobotSnapshot(
            node_id=robot.node_id,
            available_in_min=0.0,
            battery_wh=robot.battery_wh,
            busy=False,
        )
        for robot in robots
    }
    assignment = policy.update(
        10.0,
        snapshots,
    )
    assert set(assignment.threshold_by_robot_id) == {0, 1, 2, 3}
    assert assignment.threshold_by_robot_id[3] == 1.0
    assert policy.demand.count(0, 5.0) == 1
