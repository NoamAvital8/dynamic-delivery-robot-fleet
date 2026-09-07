import math

import networkx as nx

from delivery_fleet.fleet import (
    Enduro,
    MiddleMan,
    Oomph,
    RobotType,
    SpeedyMcQueen,
    create_default_fleet,
    fleet_type_counts,
    fleet_type_summary,
    robot_count_for_graph,
)


def test_fixed_robot_models_have_expected_specs() -> None:
    speedy = SpeedyMcQueen(0)
    middle = MiddleMan(1)
    enduro = Enduro(2)
    oomph = Oomph(3)

    assert math.isclose(speedy.speed_mps, 5.5)
    assert math.isclose(middle.speed_mps, 4.0)
    assert math.isclose(enduro.speed_mps, 3.5)
    assert math.isclose(oomph.speed_mps, 2.8)

    assert speedy.max_payload_kg == 5.0
    assert middle.max_payload_kg == 12.0
    assert enduro.max_payload_kg == 10.0
    assert oomph.max_payload_kg == 30.0

    assert math.isclose(speedy.full_battery_range_m, 10_000.0)
    assert math.isclose(middle.full_battery_range_m, 16_000.0)
    assert math.isclose(enduro.full_battery_range_m, 1_800.0 / 0.065)
    assert math.isclose(oomph.full_battery_range_m, 24_000.0)


def test_robot_count_scales_with_graph_nodes() -> None:
    assert robot_count_for_graph(nx.path_graph(1)) == 1
    assert robot_count_for_graph(nx.path_graph(500)) == 1
    assert robot_count_for_graph(nx.path_graph(501)) == 2
    assert robot_count_for_graph(nx.path_graph(1_001)) == 3


def test_nyc_sized_fleet_composition() -> None:
    counts = fleet_type_counts(546)

    assert counts == {
        RobotType.SPEEDY_MCQUEEN: 137,
        RobotType.MIDDLE_MAN: 218,
        RobotType.ENDURO: 109,
        RobotType.OOMPH: 82,
    }
    assert sum(counts.values()) == 546


def test_default_fleet_is_reproducible_and_starts_full() -> None:
    graph = nx.path_graph(10_000)

    first = create_default_fleet(graph, seed=123)
    second = create_default_fleet(graph, seed=123)

    assert len(first) == len(second) == 20
    assert [r.node_id for r in first] == [r.node_id for r in second]
    assert [type(r.spec) for r in first] == [type(r.spec) for r in second]
    assert len({r.node_id for r in first}) == len(first)

    for robot in first:
        assert robot.available
        assert robot.battery_wh == robot.spec.battery_capacity_wh


def test_fleet_type_summary_matches_apportionment() -> None:
    graph = nx.path_graph(20_000)
    fleet = create_default_fleet(graph, seed=42)

    assert fleet_type_summary(fleet) == fleet_type_counts(len(fleet))
