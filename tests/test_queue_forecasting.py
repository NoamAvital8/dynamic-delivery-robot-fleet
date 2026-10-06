from __future__ import annotations

import json
from pathlib import Path
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from delivery_fleet.queue_forecasting import (
    ActiveChargingForecast as Active, KnownChargingRequest as Request,
    KnownChargingForecast,
)
from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.fleet import RobotType
from delivery_fleet.robot import RobotSpec, RobotState
from delivery_fleet.scenario_creator import Item, Order, Scenario
import run_nyc_queue_aware_policy as runner


def test_two_ports_observed_queue_and_predicted_arrivals() -> None:
    model = KnownChargingForecast(0, {10: 2}, [Active(1, 10, 18), Active(2, 10, 25)],
                                  [Request(3, 10, 0, 7), Request(4, 10, 8, 5)])
    result = model.project(9).serve(10, 10, 3)
    # Queued 3 starts at 18 and finishes at 25; arrival 4 uses the other
    # port from 25 to 30; our robot then starts on the first port at 25.
    assert result.start_min == 25
    assert result.wait_min == 15
    assert result.finish_min == 28


def test_later_arrival_is_not_a_protected_reservation() -> None:
    model = KnownChargingForecast(0, {1: 1}, requests=[Request(2, 1, 6, 30)])
    candidate = model.project(1)
    assert candidate.serve(1, 5, 10).wait_min == 0
    # The later request must wait for our previous visit, not vice versa.
    assert candidate.serve(1, 20, 2).start_min == 45
    # A separate route has no knowledge of that hypothetical first visit.
    assert model.project(1).serve(1, 20, 2).start_min == 36


def test_cancel_own_old_active_session_and_future_bookings() -> None:
    model = KnownChargingForecast(3, {1: 1}, [Active(1, 1, 100)],
                                  [Request(1, 1, 10, 100), Request(2, 1, 4, 5)])
    assert model.project(1).serve(1, 5, 2).wait_min == 4
    assert model.project(3).serve(1, 5, 2).wait_min == 100


def test_zero_wait_and_charging_duration_are_separate() -> None:
    model = KnownChargingForecast(10, {1: 2}, [Active(1, 1, 30)])
    result = model.project(9).serve(1, 15, 40)
    assert result.wait_min == 0
    assert result.finish_min == 55


def test_each_candidate_is_private_and_preview_does_not_reserve() -> None:
    model = KnownChargingForecast(0, {1: 1}, [Active(2, 1, 10)])
    projection = model.project(1)
    assert projection.wait_at(1, 5) == 5
    assert projection.wait_at(1, 5) == 5
    assert projection.serve(1, 5, 3).start_min == 10
    clone = projection.clone()
    assert clone.serve(1, 11, 9).start_min == 13
    assert projection.serve(1, 11, 1).finish_min == 14
    assert model.project(1).serve(1, 5, 3).start_min == 10


def test_invalid_forecasts_fail_loudly() -> None:
    with pytest.raises(ValueError):
        KnownChargingForecast(0, {1: 0})
    with pytest.raises(ValueError):
        KnownChargingForecast(0, {1: 1}, [Active(1, 1, 2), Active(2, 1, 3)])
    with pytest.raises(ValueError):
        KnownChargingForecast(0, {1: 1}, requests=[Request(2, 1, 3, -1)])


class _QueueRobot(RobotSpec):
    robot_type = RobotType.MIDDLE_MAN


def _congested_scenario(tmp_path):
    graph = nx.path_graph(21)
    stations = [0, 5, 10, 15, 20]
    for node in graph:
        graph.nodes[node].update(x=34 + node * 0.01, y=32., in_cluster=0,
                                 is_cluster_representative=(node == 10),
                                 is_charging_station=(node in stations))
    nx.set_edge_attributes(graph, 1000., "length")
    annotate_nearest_charging_stations(graph, stations)
    graph_file, scenario_file = tmp_path / "graph.graphml", tmp_path / "scenario.json"
    nx.write_graphml(graph, graph_file)
    Scenario("charging_congestion", 90., 42, tuple(
        Order(i, 0, 20, 0., Item(1., 1.), 2.) for i in range(4)
    )).save_json(scenario_file)
    robots = [RobotState.fully_charged(_QueueRobot(i, 10., 1., 1., 100., .01), 0)
              for i in range(4)]
    return graph_file, scenario_file, robots


@pytest.mark.parametrize("alternatives", [0, 3])
def test_full_runner_forecasts_congestion_and_logs_mission_waits(tmp_path, monkeypatch, alternatives) -> None:
    graph, scenario, robots = _congested_scenario(tmp_path)
    namespace = runner._load_namespace()
    monkeypatch.setitem(namespace, "create_default_fleet", lambda *args, **kwargs: robots)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["queue-policy", "--graph", str(graph), "--scenario", str(scenario),
                                     "--output", str(output), "--idle-processes", "1",
                                     "--queue-charger-alternatives", str(alternatives)])
    namespace["main"]()
    result = json.loads(output.read_text())
    records = [json.loads(line) for line in output.with_suffix(".queue.jsonl").read_text().splitlines()]
    mission = [r for r in records if r["purpose"] == "delivery"]
    assert result["delivered"] == 4
    assert result["assignment_queue_forecast"] == "known_traffic_fifo_v1"
    assert result["queue_forecast_missing_cached_plans"] == 0
    assert result["queue_forecast_infeasible_cached_plans"] == 0
    assert result["delivery_queue_visit_count"] == len(mission)
    assert result["delivery_queue_wait_total_min"] == pytest.approx(sum(r["wait_min"] for r in mission))
    assert result["delivery_queue_prediction_samples"] > 0
    assert result["queue_positive_wait_prefixes"] > 0
    if not alternatives:
        assert result["delivery_queue_wait_total_min"] > 0
        assert result["delivery_queue_prediction_mae_min"] == pytest.approx(0, abs=1e-6)
    else:
        assert result["queue_alternative_prefix_evaluations"] > 0


def test_queue_generated_source_loads_and_legacy_is_untouched() -> None:
    namespace = runner._load_namespace()
    assert callable(namespace["main"])
    legacy = runner.idle._load_namespace()
    assert "KnownChargingForecast" not in legacy
    assert "known_traffic_fifo_v1" not in runner.idle.heuristic.corrected.TARGET.read_text()


def test_no_queue_no_charge_case_matches_legacy_full(tmp_path, monkeypatch) -> None:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(x=34. + node * .00001, y=32., in_cluster=0,
                                 is_cluster_representative=(node == 1), is_charging_station=True)
    nx.set_edge_attributes(graph, 10., "length")
    annotate_nearest_charging_stations(graph, [0, 1, 2])
    graph_file, scenario_file = tmp_path / "graph.graphml", tmp_path / "scenario.json"
    nx.write_graphml(graph, graph_file)
    Scenario("no_queue", 10., 55, (Order(0, 0, 2, 0., Item(1., 1.), 2.),)).save_json(scenario_file)
    results = []
    for name, load in (("old", runner.idle._load_namespace), ("new", runner._load_namespace)):
        output = tmp_path / f"{name}.json"
        monkeypatch.setattr(sys, "argv", [name, "--graph", str(graph_file), "--scenario", str(scenario_file),
                                         "--output", str(output), "--idle-processes", "1"])
        load()["main"]()
        results.append(json.loads(output.read_text()))
    for key in ("delivered", "on_time", "loss_objective", "delivery_route_distance_km"):
        assert results[0][key] == pytest.approx(results[1][key])
    assert "delivery_queue_wait_mean_min" not in results[0]
    assert results[1]["delivery_queue_wait_mean_min"] == 0


def test_cancelled_wait_is_logged_without_charging_time(tmp_path, monkeypatch) -> None:
    graph, scenario, robots = _congested_scenario(tmp_path)
    namespace = runner._load_namespace()
    monkeypatch.setitem(namespace, "create_default_fleet", lambda *args, **kwargs: robots)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["queue-policy", "--graph", str(graph), "--scenario", str(scenario),
                                     "--output", str(output), "--idle-processes", "1"])
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
    # Directly exercise the same cancellation hook the runtime calls. Large
    # charge energy must not contaminate the elapsed queue-wait measurement.
    request = namespace["base"].ChargeRequest(0, 10000., 100., "delivery")
    station = captured["station_state"][10]
    station.queue.append(request)
    captured["cancel_stationary_wait"](robots[0], 107.)
    record = json.loads(output.with_suffix(".queue.jsonl").read_text().splitlines()[-1])
    assert record["purpose"] == "delivery"
    assert record["outcome"] == "cancelled"
    assert record["wait_min"] == 7.
    assert captured["queue_stats"]["delivery_queue_cancelled_total_min"] == 7.
