from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_multicity_suite import make_demand
from run_multicity_experiments import POLICIES, _execute, policy_command, summarize
from run_multicity_campaign import _effective_workers
from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.fleet import RobotType
from delivery_fleet.reservation_nn import ReservationFCNN
from delivery_fleet.scenario_creator import Item, Order, Scenario


def test_sixty_logical_cpu_ceiling_respects_memory_and_nested_workers() -> None:
    available = 320 * 1024 ** 3
    assert _effective_workers(
        requested=60, max_processes=60, logical_cpus=72,
        available_memory_bytes=available, memory_per_simulator_gib=8.0,
        memory_fraction=0.75,
    ) == 30
    assert _effective_workers(
        requested=60, max_processes=60, logical_cpus=72,
        available_memory_bytes=available, memory_per_simulator_gib=8.0,
        memory_fraction=0.75, nested_processes=4,
    ) == 12


def test_suite_demand_is_node_level_and_normalized() -> None:
    nodes = np.arange(8)
    latitude = np.linspace(32.0, 32.01, len(nodes))
    longitude = np.linspace(34.0, 34.01, len(nodes))
    profile, per_robot = make_demand(
        nodes, latitude, longitude,
        duration_hours=3, fleet_count=4, rng=np.random.default_rng(7),
    )
    assert profile.lambda_by_bucket.shape == (3, len(nodes))
    assert 0.2 <= per_robot <= 0.42
    assert np.all(profile.lambda_by_bucket > 0)
    assert np.all(profile.lambda_by_bucket.sum(axis=1) > 0)


def test_six_commands_and_summary_fields(tmp_path) -> None:
    assert len(POLICIES) == 6
    commands = {
        policy: policy_command(
            policy, python="python", graph=tmp_path / "graph.graphml",
            scenario=tmp_path / "scenario.json", output=tmp_path / "result.json",
            spatial_model=tmp_path / "spatial.npz", paper_model=tmp_path / "paper.npz",
            shortlist_k=10, idle_processes=2,
            prior_rates={"1": 10.0, "2": 2.0, "5": 1.0}, prior_concentration=4.0,
        )
        for policy in POLICIES
    }
    assert "run_nyc_benchmark5.py" in commands["myopic_ab"][1]
    assert "run_nyc_benchmark5_loss.py" in commands["reactive_insertion"][1]
    assert "--reservation-style" in commands["paper_sa_adapted"]
    assert "--reservation-model" not in commands["full_no_nn"]
    assert "--reservation-model" in commands["full_no_idle"]
    job = {"city": "haifa", "scenario_id": "test_000", "seed": 1,
           "policy": "full", "output": tmp_path / "result.json", "log": tmp_path / "run.log"}
    row = summarize({"orders": 3, "delivered": 3, "on_time": 2,
                     "loss_objective": 13.5, "wall_clock_seconds": 42.0,
                     "simulation_finish_min": 100.0}, job)
    assert row["late"] == 1
    assert row["loss_objective"] == 13.5
    assert row["wall_clock_seconds"] == 42.0


def test_all_six_policies_write_durable_city_results(tmp_path) -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node].update(
            x=34.0 + 0.00001 * node, y=32.0,
            in_cluster=0, is_cluster_representative=(node == 1),
            is_charging_station=True,
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    annotate_nearest_charging_stations(graph, [0, 1, 2])
    graph_path = tmp_path / "haifa.graphml"
    nx.write_graphml(graph, graph_path)
    scenario_path = tmp_path / "haifa_test_000.json"
    Scenario(
        graph_name="haifa", duration_minutes=10.0, seed=55,
        orders=(Order(0, 0, 2, 0.0, Item(1.0, 1.0), 2.0),),
    ).save_json(scenario_path)
    model = ReservationFCNN(4, [item.value for item in RobotType], (1.0, 2.0, 5.0))
    spatial = tmp_path / "spatial.npz"
    paper = tmp_path / "paper.npz"
    model.save(spatial)
    model.save(paper)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cities": {"haifa": {
        "graph": graph_path.name,
        "prior_rates_per_hour": {"1": 10.0, "2": 2.0, "5": 1.0},
        "test": [{"id": "haifa_test_000", "scenario": scenario_path.name,
                  "seed": 55, "orders": 1}],
    }}}), encoding="utf-8")
    results_dir = tmp_path / "results"
    command = [sys.executable, str(ROOT / "scripts/run_multicity_experiments.py"),
               str(suite), str(results_dir), "--spatial-model", str(spatial),
               "--paper-model", str(paper), "--processes", "2",
               "--idle-processes", "1"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, cwd=ROOT, env=environment, check=True,
                   capture_output=True, text=True, timeout=120)
    with (results_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["policy"] for row in rows} == set(POLICIES)
    assert all(row["status"] == "complete" for row in rows)
    assert all(int(row["delivered"]) == 1 for row in rows)
    assert all(int(row["on_time"]) + int(row["late"]) == 1 for row in rows)
    with (results_dir / "city_policy_summary.csv").open(encoding="utf-8", newline="") as stream:
        aggregate = list(csv.DictReader(stream))
    assert len(aggregate) == 6
    assert all(int(row["completed_scenarios"]) == 1 for row in aggregate)
    for policy in POLICIES:
        job_dir = results_dir / "haifa" / "haifa_test_000" / policy
        assert (job_dir / "result.json").is_file()
        assert (job_dir / "result.progress.json").is_file()
        assert (job_dir / "runner.log").is_file()
        assert (job_dir / "stamp.json").is_file()
    rerun = subprocess.run(command, cwd=ROOT, env=environment,
                           capture_output=True, text=True, timeout=30)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr


def test_zero_timeout_waits_without_a_deadline(tmp_path, monkeypatch) -> None:
    output = tmp_path / "result.json"
    log = tmp_path / "runner.log"
    status = tmp_path / "status.json"
    stamp = tmp_path / "stamp.json"
    waits = []

    class Process:
        def wait(self, timeout=None):
            waits.append(timeout)
            output.write_text(json.dumps({
                "orders": 1, "delivered": 1, "on_time": 1,
                "loss_objective": 1.0, "wall_clock_seconds": 1.0,
                "simulation_finish_min": 1.0,
            }), encoding="utf-8")
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    job = {
        "city": "test", "scenario_id": "test_000", "seed": 1,
        "policy": "reactive_insertion", "orders": 1, "output": output,
        "log": log, "status_path": status, "stamp_path": stamp,
        "command": ["simulator"], "signature": "new",
        "accepted_signatures": {"new"},
    }
    row = _execute(job, None)
    assert row["status"] == "complete"
    assert waits == [None]
