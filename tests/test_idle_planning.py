from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.idle_planning import (
    ChargingCalendar,
    IdleFleetPlanner,
    IdlePlanningConfig,
    IdleRobotSnapshot,
)
from delivery_fleet.scenario_creator import Item, Order, Scenario
from delivery_fleet.spatial_demand import GammaPoissonDemandModel
import run_nyc_anticipatory_idle_policy as idle_runner


class _SmallIndex:
    def __init__(self, graph: nx.Graph, stations: tuple[int, ...]) -> None:
        self.graph = graph
        self.stations = stations
        self.oracle = self

    def distance(self, a: int, b: int) -> float:
        return float(nx.shortest_path_length(self.graph, a, b, weight="length"))

    def distances_to_stations(self, node: int) -> dict[int, float]:
        return {station: self.distance(node, station) for station in self.stations}


def _hotspot_planner(concentration: float, penalty: float) -> IdleFleetPlanner:
    graph = nx.path_graph(21)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.001, y=32.0,
            in_cluster=0 if node == 0 else 1,
            is_cluster_representative=(node in {0, 20}),
        )
    nx.set_edge_attributes(graph, 100.0, "length")
    return IdleFleetPlanner(
        graph, _SmallIndex(graph, (0,)),
        GammaPoissonDemandModel(graph, {5.0: 60.0}, prior_concentration=concentration),
        (0,), 2_000.0,
        IdlePlanningConfig(
            max_demand_clusters=2, candidate_clusters=2,
            relocation_uncertainty_penalty=penalty,
        ),
    )


def _hotspot_robot(battery_wh: float = 100.0) -> IdleRobotSnapshot:
    return IdleRobotSnapshot(
        robot_id=1, node_id=0, speed_mps=4.0,
        energy_per_meter_wh=0.01, battery_wh=battery_wh,
        battery_capacity_wh=100.0, max_payload_kg=12.0, max_volume_l=30.0,
    )


def test_uncertain_hotspot_stays_but_same_mean_with_strong_prior_moves() -> None:
    robot = _hotspot_robot()
    with (_hotspot_planner(0.01, 0.0) as original,
          _hotspot_planner(0.01, 1.0) as uncertain,
          _hotspot_planner(100.0, 1.0) as confident):
        assert uncertain.demand.prior(1, 5.0).mean_per_minute == pytest.approx(
            confident.demand.prior(1, 5.0).mean_per_minute
        )
        assert original.plan(0.0, [robot], [1], ChargingCalendar({0: 1}))[1].target_node == 20
        assert uncertain.plan(0.0, [robot], [1], ChargingCalendar({0: 1})) == {}
        assert confident.plan(0.0, [robot], [1], ChargingCalendar({0: 1}))[1].target_node == 20


def test_observed_hotspot_can_overcome_uncertainty_penalty() -> None:
    with _hotspot_planner(0.01, 1.0) as planner:
        for time_min in range(20):
            planner.demand.observe(20, 5.0, float(time_min))
        chosen = planner.plan(20.0, [_hotspot_robot()], [1], ChargingCalendar({0: 1}))
    assert chosen[1].kind == "reposition"
    assert chosen[1].target_node == 20


def test_uncertainty_gate_does_not_prevent_full_charging() -> None:
    robot = _hotspot_robot(battery_wh=1.0)
    with _hotspot_planner(0.01, 100.0) as planner:
        chosen = planner.plan(0.0, [robot], [1], ChargingCalendar({0: 1}))
    assert chosen[1].kind == "charge"
    assert chosen[1].battery_after_wh == robot.battery_capacity_wh


@pytest.mark.parametrize("penalty", [-1.0, float("nan"), float("inf")])
def test_invalid_uncertainty_penalty_is_rejected(penalty: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        IdlePlanningConfig(relocation_uncertainty_penalty=penalty)


def test_calendar_never_overlaps_on_one_port_and_uses_second_port() -> None:
    one_port = ChargingCalendar({10: 1})
    first = one_port.reserve(1, 10, 4.0, 8.0)
    second = one_port.reserve(2, 10, 5.0, 3.0)
    assert first.start_min == 4.0
    assert second.start_min == 12.0

    two_ports = ChargingCalendar({10: 2})
    assert two_ports.reserve(1, 10, 4.0, 8.0).port == 0
    other = two_ports.reserve(2, 10, 5.0, 3.0)
    assert other.port == 1
    assert other.start_min == 5.0


def test_charge_candidates_always_finish_with_full_battery() -> None:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.0001,
            y=32.0,
            in_cluster=node,
            is_cluster_representative=True,
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    index = _SmallIndex(graph, (0, 2))
    demand = GammaPoissonDemandModel(graph, {1.0: 60.0})
    planner = IdleFleetPlanner(
        graph, index, demand, (0, 2), 2_000.0,
        IdlePlanningConfig(candidate_clusters=2),
    )
    robot = IdleRobotSnapshot(
        robot_id=1, node_id=1, speed_mps=4.0,
        energy_per_meter_wh=0.1, battery_wh=10.0,
        battery_capacity_wh=100.0, max_payload_kg=12.0,
        max_volume_l=30.0,
    )
    calendar = ChargingCalendar({0: 1, 2: 1})
    actions = planner._candidate_actions(
        robot, (0, 1, 2), {0: 1.0, 1: 1.0, 2: 1.0}, 0.0, calendar
    )
    charges = [action for action in actions if action.kind == "charge"]
    assert charges
    assert all(action.battery_after_wh == robot.battery_capacity_wh for action in charges)
    assert all(action.slot is not None for action in charges)


def test_parallel_planning_caps_moves_and_keeps_charger_slots_separate() -> None:
    graph = nx.path_graph(15)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.0001,
            y=32.0,
            in_cluster=node // 3,
            is_cluster_representative=(node % 3 == 1),
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    stations = (0, 7, 14)
    index = _SmallIndex(graph, stations)
    demand = GammaPoissonDemandModel(graph, {1.0: 60.0, 5.0: 15.0})
    robots = [
        IdleRobotSnapshot(
            robot_id=robot_id, node_id=robot_id,
            speed_mps=4.0, energy_per_meter_wh=0.1,
            battery_wh=10.0, battery_capacity_wh=100.0,
            max_payload_kg=12.0, max_volume_l=30.0,
        )
        for robot_id in range(15)
    ]
    calendar = ChargingCalendar({station: 1 for station in stations})
    with IdleFleetPlanner(
        graph, index, demand, stations, 2_000.0,
        IdlePlanningConfig(
            candidate_clusters=4, candidate_chargers=3,
            max_actions_per_epoch=2, minimum_gain_fraction=0.0,
            processes=2,
        ),
    ) as planner:
        chosen = planner.plan(0.0, robots, range(15), calendar)
        assert planner._pool is not None  # The batch crossed the process threshold.
    assert len(chosen) <= 2
    assert any(action.kind == "charge" for action in chosen.values())
    for station in stations:
        slots = calendar.slots(station)
        for first in slots:
            for second in slots:
                if first.robot_id != second.robot_id:
                    assert (
                        first.finish_min <= second.start_min
                        or second.finish_min <= first.start_min
                    )


def test_marginal_value_sends_one_robot_to_demand_cluster() -> None:
    graph = nx.path_graph(11)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.001,
            y=32.0,
            in_cluster=0 if node < 5 else 1,
            is_cluster_representative=(node in {0, 10}),
        )
    nx.set_edge_attributes(graph, 100.0, "length")
    index = _SmallIndex(graph, (0,))
    demand = GammaPoissonDemandModel(graph, {5.0: 60.0})
    for _ in range(30):
        demand.observe(10, 5.0, 0.0)
    robots = [
        IdleRobotSnapshot(
            robot_id=robot_id, node_id=0, speed_mps=4.0,
            energy_per_meter_wh=0.01, battery_wh=100.0,
            battery_capacity_wh=100.0, max_payload_kg=12.0,
            max_volume_l=30.0,
        )
        for robot_id in range(2)
    ]
    with IdleFleetPlanner(
        graph, index, demand, (0,), 2_000.0,
        IdlePlanningConfig(
            max_demand_clusters=2, candidate_clusters=2,
            minimum_gain_fraction=0.075,
        ),
    ) as planner:
        chosen = planner.plan(
            0.0, robots, (0, 1), ChargingCalendar({0: 1})
        )
    assert len(chosen) == 1
    assert next(iter(chosen.values())).target_node == 10


@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_anticipatory_runner_loads_and_executes_small_scenario(tmp_path: Path, penalty: float) -> None:
    assert callable(idle_runner._load_namespace()["main"])
    graph = nx.path_graph(501)
    for node in graph:
        graph.nodes[node].update(
            x=34.0 + node * 0.00001,
            y=32.0,
            in_cluster=0 if node < 251 else 1,
            is_cluster_representative=(node in {125, 375}),
            is_charging_station=(node in {0, 250, 500}),
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    annotate_nearest_charging_stations(graph, [0, 250, 500])
    graph_path = tmp_path / "graph.graphml"
    scenario_path = tmp_path / "scenario.json"
    output = tmp_path / "output.json"
    nx.write_graphml(graph, graph_path)
    Scenario(
        graph_name="anticipatory_idle_test",
        duration_minutes=25.0,
        seed=42,
        orders=(Order(
            id=0, pickup_node=100, dropoff_node=300,
            request_time_min=1.0,
            item=Item(weight_kg=1.0, volume_l=1.0), importance=2.0,
        ),),
    ).save_json(scenario_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_nyc_anticipatory_idle_policy.py"),
         "--graph", str(graph_path), "--scenario", str(scenario_path),
         "--output", str(output), "--shortlist-k", "2",
         "--idle-replan-interval-min", "5",
         "--idle-minimum-gain-fraction", "0.001",
         "--idle-relocation-uncertainty-penalty", str(penalty)],
        cwd=ROOT, env=environment, check=True, capture_output=True,
        text=True, timeout=90,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["delivered"] == 1
    assert result["idle_policy"] == (
        "posterior_uncertainty_gated" if penalty > 0 else "posterior_marginal_utility"
    )
    assert result["idle_relocation_uncertainty_penalty"] == penalty
    assert result["idle_planning_epochs"] >= 1
    if penalty == 0:
        assert result["idle_reposition_actions"] >= 1
