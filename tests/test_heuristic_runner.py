from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import networkx as nx

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.scenario_creator import Item, Order, Scenario
import run_nyc_heuristic_policy as heuristic


def test_generated_heuristic_runner_contains_reservation_and_safe_fallback() -> None:
    source = heuristic.corrected._patch_source(
        heuristic.corrected.TARGET.read_text(encoding="utf-8")
    )
    generated = heuristic._patch_heuristic(source)
    compile(generated, "generated_heuristic_runner", "exec")
    assert "reservation_policy.observe" in generated
    assert "reservation_eligibility" in generated
    assert "shortlist_fallback_robots" in generated
    assert "minimum_exact_incremental_total_loss_with_safe_shortlist_expansion" in generated


def test_generated_heuristic_runner_namespace_loads() -> None:
    namespace = heuristic._load_namespace()
    assert callable(namespace["main"])
    assert namespace["__name__"] == "nyc_heuristic_policy_target"


def _write_small_benchmark(graph_path: Path, scenario_path: Path) -> None:
    # 501 nodes deliberately creates two robots under ceil(|V| / 500), while
    # short 10 m edges keep this exact end-to-end regression test inexpensive.
    graph = nx.path_graph(501)
    for node in graph.nodes:
        graph.nodes[node].update(
            x=34.0 + node * 0.00001,
            y=32.0,
            in_cluster=0,
            is_cluster_representative=(node == 250),
            is_charging_station=(node in {0, 250, 500}),
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    annotate_nearest_charging_stations(graph, [0, 250, 500])
    nx.write_graphml(graph, graph_path)
    Scenario(
        graph_name="top_k_equivalence_test",
        duration_minutes=30.0,
        seed=42,
        orders=(
            Order(
                id=0,
                pickup_node=100,
                dropoff_node=300,
                request_time_min=0.0,
                item=Item(weight_kg=1.0, volume_l=1.0),
                importance=2.0,
            ),
        ),
    ).save_json(scenario_path)


def _run_policy(script: str, graph: Path, scenario: Path, output: Path, *extra: str) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--graph",
            str(graph),
            "--scenario",
            str(scenario),
            "--output",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_full_k_matches_exhaustive_exact_policy_on_small_scenario(tmp_path) -> None:
    graph = tmp_path / "graph.graphml"
    scenario = tmp_path / "scenario.json"
    exhaustive_output = tmp_path / "exhaustive.json"
    heuristic_output = tmp_path / "heuristic.json"
    _write_small_benchmark(graph, scenario)

    exhaustive = _run_policy(
        "run_nyc_benchmark5_loss.py", graph, scenario, exhaustive_output
    )
    full_k = _run_policy(
        "run_nyc_heuristic_policy.py",
        graph,
        scenario,
        heuristic_output,
        "--shortlist-k",
        "2",
    )

    assert exhaustive["delivered"] == full_k["delivered"] == 1
    assert math.isclose(
        exhaustive["loss_objective"],
        full_k["loss_objective"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        exhaustive["delivery_route_distance_km"],
        full_k["delivery_route_distance_km"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )
