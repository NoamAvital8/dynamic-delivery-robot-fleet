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

from create_multicity_suite import CITIES, DEFAULT_CITIES, NEW_CITIES, make_demand
from run_multicity_experiments import DEFAULT_POLICIES, POLICIES, _atomic_json, _execute, order_jobs, policy_command, summarize
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


def test_new_city_suite_excludes_original_cities_when_selected(tmp_path) -> None:
    assert len(DEFAULT_CITIES)==5 and len(NEW_CITIES)==6
    assert len(CITIES)==11 and not set(DEFAULT_CITIES)&set(NEW_CITIES)
    graph_dir=tmp_path/'graphs';graph_dir.mkdir()
    for city in NEW_CITIES:
        graph=nx.path_graph(3)
        for node in graph:
            graph.nodes[node].update(x=20.+node*.001,y=-25.)
        nx.set_edge_attributes(graph,100.,'length')
        nx.write_graphml(graph,graph_dir/f'{city}.graphml')
    out=tmp_path/'suite'
    command=[sys.executable,str(ROOT/'scripts/create_multicity_suite.py'),
        '--graph-dir',str(graph_dir),'--output-dir',str(out),'--cities',*NEW_CITIES,
        '--train-per-city','2','--test-per-city','5','--duration-hours','12',
        '--base-seed','20261007']
    subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
    suite=json.loads((out/'suite.json').read_text())
    assert tuple(suite['cities'])==NEW_CITIES
    assert not set(suite['cities'])&set(DEFAULT_CITIES)
    for city,data in suite['cities'].items():
        assert len(data['test'])==5 and len(data['train'])==2
        assert {r['seed'] for r in data['test']}.isdisjoint(r['seed'] for r in data['train'])
        assert all(r['id'].startswith(city+'_test_') for r in data['test'])
    assert sum(len(d['test']) for d in suite['cities'].values())*len(POLICIES)==420
    before={p:p.read_bytes() for p in out.glob('test/*.json')}
    subprocess.run([*command,'--resume'],cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
    assert all(p.read_bytes()==data for p,data in before.items())


def test_round_robin_starts_all_cities_early_without_changing_jobs():
    jobs=[{'city_index':c,'scenario_index':s,'policy_index':p}
          for c in range(6) for s in range(5) for p in range(14)]
    assert order_jobs(jobs,'city-major') is jobs
    ordered=order_jobs(jobs,'round-robin')
    assert [j['city_index'] for j in ordered[:6]]==list(range(6))
    assert all(j['scenario_index']==0 and j['policy_index']==0 for j in ordered[:6])
    assert len(ordered)==420 and {id(j) for j in ordered}=={id(j) for j in jobs}


def test_policy_commands_preserve_baselines_and_enable_uncertainty_variants(tmp_path) -> None:
    assert len(DEFAULT_POLICIES) == 6
    commands = {
        policy: policy_command(
            policy, python="python", graph=tmp_path / "graph.graphml",
            scenario=tmp_path / "scenario.json", output=tmp_path / "result.json",
            spatial_model=tmp_path / "spatial.npz", paper_model=tmp_path / "paper.npz",
            shortlist_k=10, idle_processes=2,
            prior_rates={"1": 10.0, "2": 2.0, "5": 1.0}, prior_concentration=4.0,
            relocation_uncertainty_penalty=1.5,
        )
        for policy in POLICIES
    }
    assert "run_nyc_benchmark5.py" in commands["myopic_ab"][1]
    assert "run_nyc_benchmark5_loss.py" in commands["reactive_insertion"][1]
    assert "--reservation-style" in commands["paper_sa_adapted"]
    assert "--reservation-model" not in commands["full_no_nn"]
    assert "--reservation-model" in commands["full_no_idle"]
    assert "--idle-relocation-uncertainty-penalty" not in commands["full"]
    assert "--idle-relocation-uncertainty-penalty" not in commands["full_no_nn"]
    for policy in ("full_uncertainty_idle", "full_uncertainty_idle_no_nn"):
        command = commands[policy]
        assert command[command.index("--idle-relocation-uncertainty-penalty") + 1] == "1.5"
        assert "--idle-processes" in command
    assert "--reservation-model" in commands["full_uncertainty_idle"]
    assert "--reservation-model" not in commands["full_uncertainty_idle_no_nn"]
    assert "run_nyc_queue_aware_policy.py" in commands["full_queue_aware"][1]
    assert "--reservation-model" in commands["full_queue_aware"]
    assert "--reservation-model" not in commands["full_queue_aware_no_nn"]
    assert "--idle-processes" in commands["full_queue_aware"]
    assert "run_nyc_coordinated_idle_policy.py" in commands["full_coordinated_idle"][1]
    assert "--reservation-model" in commands["full_coordinated_idle"]
    assert "--reservation-model" not in commands["full_coordinated_idle_no_nn"]
    assert "--idle-switch-gain-fraction" in commands["full_coordinated_idle"]
    assert "--idle-retarget-cooldown-min" in commands["full_coordinated_idle"]
    job = {"city": "haifa", "scenario_id": "test_000", "seed": 1,
           "policy": "full", "output": tmp_path / "result.json", "log": tmp_path / "run.log"}
    row = summarize({"orders": 3, "delivered": 3, "on_time": 2,
                     "loss_objective": 13.5, "wall_clock_seconds": 42.0,
                     "simulation_finish_min": 100.0, "idle_reposition_actions": 4,
                     "idle_relocation_uncertainty_penalty": 1.5}, job)
    assert row["late"] == 1
    assert row["loss_objective"] == 13.5
    assert row["wall_clock_seconds"] == 42.0
    assert row["idle_reposition_actions"] == 4
    assert row["idle_relocation_uncertainty_penalty"] == 1.5
    assert row["idle_charge_actions"] == ""


def test_all_policies_write_durable_city_results(tmp_path) -> None:
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
               "--idle-processes", "1", "--policies", *POLICIES]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, cwd=ROOT, env=environment, check=True,
                   capture_output=True, text=True, timeout=120)
    with (results_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(POLICIES)
    assert {row["policy"] for row in rows} == set(POLICIES)
    assert all(row["status"] == "complete" for row in rows)
    assert all(int(row["delivered"]) == 1 for row in rows)
    assert all(int(row["on_time"]) + int(row["late"]) == 1 for row in rows)
    with (results_dir / "city_policy_summary.csv").open(encoding="utf-8", newline="") as stream:
        aggregate = list(csv.DictReader(stream))
    assert len(aggregate) == len(POLICIES)
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


def test_parallel_status_write_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []

    def briefly_locked(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("temporary reader lock")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    destination = tmp_path / "status.json"
    _atomic_json(destination, {"status": "complete"})
    assert json.loads(destination.read_text()) == {"status": "complete"}
    assert len(attempts) == 3
