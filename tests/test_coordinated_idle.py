from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.coordinated_idle import CoordinatedIdleFleetPlanner
from delivery_fleet.fleet import RobotType
from delivery_fleet.idle_planning import ChargingCalendar, IdleAction, IdlePlanningConfig, IdleRobotSnapshot
from delivery_fleet.robot import RobotActivity, RobotSpec, RobotState
from delivery_fleet.scenario_creator import Item, Order, Scenario
from delivery_fleet.spatial_demand import GammaPoissonDemandModel
import run_nyc_coordinated_idle_policy as runner


class Index:
    def __init__(self, graph, stations):
        self.graph, self.stations, self.oracle = graph, stations, self

    def distance(self, a, b):
        return float(nx.shortest_path_length(self.graph, a, b, weight="length"))

    def distances_to_stations(self, node):
        return {s: self.distance(node, s) for s in self.stations}


def hotspot():
    graph = nx.path_graph(11)
    for node in graph:
        graph.nodes[node].update(x=34. + node * .001, y=32.,
                                in_cluster=0 if node < 5 else 1,
                                is_cluster_representative=node in {0, 10})
    nx.set_edge_attributes(graph, 100., "length")
    demand = GammaPoissonDemandModel(graph, {5.: 60.})
    for _ in range(30):
        demand.observe(10, 5., 0.)
    robots = [IdleRobotSnapshot(i, 0, 4., .01, 100., 100., 12., 30.) for i in range(2)]
    planner = CoordinatedIdleFleetPlanner(
        graph, Index(graph, (0,)), demand, (0,), 2000.,
        IdlePlanningConfig(max_demand_clusters=2, candidate_clusters=2, minimum_gain_fraction=.075),
    )
    return planner, robots


def test_joint_allocation_does_not_send_every_robot_to_the_same_hotspot():
    planner, robots = hotspot()
    with planner:
        chosen = planner.plan(0., robots, [0, 1], ChargingCalendar({0: 1}))
    assert len(chosen) == 1
    assert next(iter(chosen.values())).target_node == 10


def test_several_robots_can_cover_a_busy_cluster_if_both_have_positive_marginal_value():
    planner, robots = hotspot()
    planner.config = replace(planner.config, minimum_gain_fraction=0.)
    with planner:
        chosen = planner.plan(0., robots, [0, 1], ChargingCalendar({0: 1}))
    assert len(chosen) == 2
    assert all(a.target_node == 10 for a in chosen.values())


def test_robot_already_heading_to_hotspot_counts_as_coverage_once():
    planner, robots = hotspot()
    baseline = IdleAction(0, "stay", 10, 0., 1., 1., 90.)
    with planner:
        without_intent = planner.plan(0., robots, [1], ChargingCalendar({0: 1}))
        with_intent = planner.plan(0., robots, [1], ChargingCalendar({0: 1}),
                                   baseline_actions={0: baseline})
    assert 1 in without_intent
    assert with_intent == {}


def test_switch_threshold_is_relative_to_continuing_not_original_position():
    planner, robots = hotspot()
    # The present plan already reaches the hotspot sooner and with more battery
    # than starting a fresh route there. Neither STAY nor restarting improves it.
    baseline = IdleAction(0, "stay", 10, 0., 0., 0., 100.)
    with planner:
        assert planner.plan(0., robots[:1], [0], ChargingCalendar({0: 1}),
                            baseline_actions={0: baseline}, switching_gain_fraction=0.) == {}
        # A weak old destination can be changed, unless the hysteresis gate is high.
        weak = IdleAction(0, "stay", 0, 0., 30., 30., 100.)
        assert planner.plan(0., robots[:1], [0], ChargingCalendar({0: 1}),
                            baseline_actions={0: weak}, switching_gain_fraction=0.)
        assert planner.plan(0., robots[:1], [0], ChargingCalendar({0: 1}),
                            baseline_actions={0: weak}, switching_gain_fraction=1.) == {}


def test_candidate_waits_for_committed_edge_without_spending_its_energy_twice():
    planner, robots = hotspot()
    snapshot = IdleRobotSnapshot(0, 0, 4., .01, 40., 100., 12., 30., available_in_min=3.)
    calendar = ChargingCalendar({0: 1})
    calendar.reserve(99, 0, 0., 5.)
    with planner:
        actions = planner._candidate_actions(snapshot, [0, 1], {0: 1., 1: 100.}, 0., calendar)
    move = next(a for a in actions if a.kind == "reposition")
    assert move.arrival_min == pytest.approx(3. + 1000. / 4. / 60.)
    assert move.ready_min == move.arrival_min
    assert move.battery_after_wh == 30.
    charge = next(a for a in actions if a.kind == "charge")
    assert charge.arrival_min == 3.
    assert charge.slot.start_min == 5.
    assert charge.ready_min == charge.slot.finish_min
    assert charge.battery_after_wh == 100.


def test_spatial_overlap_is_not_limited_to_identical_cluster_labels():
    planner, robots = hotspot()
    # Two differently labelled representatives have nearly identical response
    # geometry. Coverage vectors should not act as disjoint cluster buckets.
    planner._coordinates[10] = (planner._coordinates[0][0], planner._coordinates[0][1] + 1e-9)
    with planner:
        _, _, lat, lon, importance, _, _ = planner._demand_cells(0.)
        action = IdleAction(0, "stay", 0, 0., 0., 0., 100.)
        vector = planner._vectors([(robots[0], action, *planner._coordinates[0])], lat, lon, importance)[0]
    assert vector[0] == pytest.approx(vector[1], rel=1e-5)


def test_parallel_scoring_is_identical_to_serial_and_ports_never_overlap():
    graph = nx.path_graph(15)
    stations = (0, 7, 14)
    for node in graph:
        graph.nodes[node].update(x=34. + node * .0001, y=32., in_cluster=node // 3,
                                is_cluster_representative=node % 3 == 1)
    nx.set_edge_attributes(graph, 10., "length")
    demand = GammaPoissonDemandModel(graph, {1.: 60., 5.: 15.})
    robots = [IdleRobotSnapshot(i, i, 4., .1, 10., 100., 12., 30.) for i in range(15)]
    choices = []
    for processes in (1, 2):
        calendar = ChargingCalendar({s: 1 for s in stations})
        with CoordinatedIdleFleetPlanner(graph, Index(graph, stations), demand, stations, 2000.,
                                        IdlePlanningConfig(candidate_clusters=4, candidate_chargers=3,
                                                           processes=processes)) as planner:
            choices.append(planner.plan(0., robots, range(15), calendar))
            if processes == 2:
                assert planner._pool is not None
        for station in stations:
            slots = calendar.slots(station)
            for a in slots:
                for b in slots:
                    if a.robot_id != b.robot_id:
                        assert a.finish_min <= b.start_min or b.finish_min <= a.start_min
        assert all(a.battery_after_wh == 100. for a in choices[-1].values() if a.kind == "charge")
    assert choices[0] == choices[1]


class SmallRobot(RobotSpec):
    robot_type = RobotType.MIDDLE_MAN


def run_small(tmp_path, monkeypatch, *, burst=False):
    graph = nx.path_graph(21)
    stations = [0, 5, 10, 15, 20]
    for node in graph:
        graph.nodes[node].update(x=34. + node * .001, y=32.,
                                in_cluster=0 if node < 10 else 1,
                                is_cluster_representative=node in {0, 20},
                                is_charging_station=node in stations)
    nx.set_edge_attributes(graph, 100., "length")
    annotate_nearest_charging_stations(graph, stations)
    graph_file, scenario_file = tmp_path / "graph.graphml", tmp_path / "scenario.json"
    nx.write_graphml(graph, graph_file)
    Scenario("coordinated", 60., 42, tuple(
        Order(i, 0, 20, 1. if burst else float(i + 1), Item(1., 1.), 2.) for i in range(3)
    )).save_json(scenario_file)
    robots = [RobotState.fully_charged(SmallRobot(i, 4., 1., 1., 100., .01), 0) for i in range(4)]
    namespace = runner._load_namespace()
    monkeypatch.setitem(namespace, "create_default_fleet", lambda *a, **kw: robots)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["coordinated", "--graph", str(graph_file), "--scenario", str(scenario_file),
                                     "--output", str(output), "--idle-processes", "1",
                                     "--idle-replan-interval-min", "1000"])
    captured = {}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is namespace["main"].__code__:
            captured.update(frame.f_locals)

    sys.setprofile(profile)
    try:
        namespace["main"]()
    finally:
        sys.setprofile(previous)
    return namespace, captured, robots, json.loads(output.read_text())


@pytest.mark.parametrize("burst", [False, True])
def test_assignments_trigger_refresh_and_same_time_burst_is_coalesced(tmp_path, monkeypatch, burst):
    _, captured, _, result = run_small(tmp_path, monkeypatch, burst=burst)
    assert result["delivered"] == 3
    assert result["policy_version"] == "coordinated_idle_v1"
    assert result["idle_assignment_refresh_requests"] == 3
    assert result["idle_assignment_refresh_epochs"] == (1 if burst else 3)
    assert result["idle_readiness_missing_plan"] == result["idle_readiness_infeasible_plan"] == 0
    assert result["queue_forecast_missing_cached_plans"] == result["queue_forecast_infeasible_cached_plans"] == 0
    assert not captured["coordinated_pending_times"]
    assert Path(result["idle_decisions_file"]).is_file()
    records = [json.loads(line) for line in Path(result["idle_decisions_file"]).read_text().splitlines()]
    assert all(r["arrival_min"] >= r["decision_time_min"] >= r["time_min"] for r in records)


def test_runtime_retarget_preserves_committed_edge_and_never_interrupts_charging(tmp_path, monkeypatch):
    namespace, captured, robots, _ = run_small(tmp_path, monkeypatch)
    robot = robots[3]
    rid = robot.spec.id
    # Isolate a legally editable, idle route after the integration run.
    robot.clear_movement_plan()
    robot.activity = RobotActivity.IDLE
    robot.node_id, robot.battery_wh = 0, 100.
    robot.set_planned_path((0, 1, 2, 3))
    edge = robot.depart_next_edge(captured["graph"], 100.)
    captured["idle_intents"][rid] = {"kind": "reposition", "target_node": 3}
    event_count = len(captured["events"])
    action = IdleAction(rid, "reposition", 5, 400., 103., 3., 95.)
    captured["start_idle_action"](robot, action, 100.1)
    assert robot.next_node == edge.node_id
    assert robot.next_node_arrival_time_min == edge.time_min
    assert robot.remaining_route == [2, 3, 4, 5]
    assert len(captured["events"]) == event_count  # Not a second arrival event.
    assert captured["idle_intents"][rid]["battery_arrival_wh"] == 95.
    # Cancelling the remaining relocation still finishes the one committed edge.
    captured["start_idle_action"](robot, IdleAction(rid, "stay", 1, 0., edge.time_min,
                                                  edge.time_min - 100.2, 99.), 100.2)
    assert robot.next_node == 1 and robot.remaining_route == []
    assert robot.next_node_arrival_time_min == edge.time_min
    robot.arrive_at_next_node(edge)
    captured["idle_intents"].pop(rid)
    robot.activity = RobotActivity.CHARGING
    with pytest.raises(RuntimeError, match="cannot interrupt"):
        captured["start_idle_action"](robot, action, 101.)
    robot.activity = RobotActivity.WAITING
    with pytest.raises(RuntimeError, match="cannot interrupt"):
        captured["start_idle_action"](robot, action, 101.)
    robot.activity = RobotActivity.IDLE
    captured["idle_intents"][rid] = {"kind": "charge", "target_node": 5}
    with pytest.raises(RuntimeError, match="committed charging trip"):
        captured["start_idle_action"](robot, action, 101.)


def test_runtime_cooldown_keeps_current_target_then_compares_continuation(tmp_path, monkeypatch):
    _, captured, robots, _ = run_small(tmp_path, monkeypatch)
    robot = robots[3]
    rid = robot.spec.id
    robot.clear_movement_plan()
    robot.activity = RobotActivity.IDLE
    robot.node_id, robot.battery_wh = 0, 100.
    robot.set_planned_path((0, 1, 2, 3))
    edge = robot.depart_next_edge(captured["graph"], 100.)
    for state in captured["station_state"].values():
        state.active.clear()
        state.queue.clear()
    captured["idle_intents"].clear()
    captured["idle_intents"][rid] = {
        "robot_id": rid, "kind": "reposition", "target_node": 3,
        "arrival_min": 105., "battery_arrival_wh": 97., "slot": None,
    }
    captured["idle_last_target_change"][rid] = 100.
    calls = []

    def no_change(now, snapshots, ids, calendar, **kwargs):
        calls.append((now, snapshots, ids, kwargs))
        return {}

    monkeypatch.setattr(captured["idle_planner"], "plan", no_change)
    captured["plan_idle_fleet"](100.1, {rid})
    assert not calls
    assert captured["idle_stats"]["cooldown_protected"] > 0
    assert captured["idle_intents"][rid]["target_node"] == 3
    # After cooldown, the central planner sees the legal origin and the
    # separate continuation baseline. No selected change preserves movement.
    captured["plan_idle_fleet"](102.1, {rid})
    assert len(calls) == 1
    _, snapshots, ids, kwargs = calls[0]
    assert ids == {rid}
    assert next(r for r in snapshots if r.robot_id == rid).node_id == 1
    assert kwargs["baseline_actions"][rid].target_node == 3
    assert kwargs["baseline_actions"][rid].battery_after_wh == 97.
    assert robot.next_node_arrival_time_min == edge.time_min
    assert not robot.available
    captured["pending"].append(Order(99, 0, 20, 103., Item(1., 1.), 2.))
    captured["plan_idle_fleet"](103., {rid})
    assert len(calls) == 1  # Unassigned real orders block speculative moves.


def test_legacy_loaders_remain_opt_out():
    assert callable(runner._load_namespace()["main"])
    assert "CoordinatedIdleFleetPlanner" not in runner.queue._load_namespace()
    assert "coordinated_idle_replan" not in runner.queue.idle.heuristic.corrected.TARGET.read_text()


@pytest.mark.parametrize("fraction", [-1., float("nan"), float("inf")])
def test_invalid_switch_threshold_rejected(fraction):
    planner, robots = hotspot()
    with planner, pytest.raises(ValueError, match="finite and non-negative"):
        planner.plan(0., robots, [0], ChargingCalendar({0: 1}), switching_gain_fraction=fraction)
