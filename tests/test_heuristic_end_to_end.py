"""End-to-end smoke test for the actual Haversine top-K simulation runner.

This test deliberately executes the CLI in fresh Python processes, rather
than only testing the spatial-demand utility functions.  With one eligible
robot, top-K (K=1) must produce the same exact cumulative loss as the
unshortlisted exact-incremental-loss policy on the same graph and requests.
Orders overlap to exercise busy-robot insertion.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.scenario_creator import Item, Order, Scenario


ROOT = Path(__file__).resolve().parents[1]


def _smoke_inputs(tmp_path: Path) -> tuple[Path, Path]:
    # A short undirected road graph with plausible NYC geographic coordinates.
    # The default fleet contains one robot (ceil(18 / 500)).
    graph = nx.path_graph(18)
    for node in graph.nodes:
        graph.nodes[node]["y"] = 40.75
        graph.nodes[node]["x"] = -73.98 + 0.00040 * node

    nx.set_edge_attributes(graph, 40.0, "length")

    station_nodes = (0, 6, 12, 17)
    for node in graph.nodes:
        graph.nodes[node]["is_charging_station"] = node in station_nodes
    for station_id, node in enumerate(station_nodes):
        graph.nodes[node]["charging_station_id"] = station_id
        graph.nodes[node]["charging_power_w"] = 2_000.0
        graph.nodes[node]["charging_ports"] = 2

    annotate_nearest_charging_stations(graph, station_nodes)

    graph_path = tmp_path / "smoke.graphml"
    nx.write_graphml(graph, graph_path)

    scenario = Scenario(
        graph_name="heuristic_smoke",
        duration_minutes=15.0,
        seed=42,
        orders=(
            Order(
                id=1,
                pickup_node=2,
                dropoff_node=15,
                request_time_min=0.0,
                item=Item(weight_kg=0.5, volume_l=1.0),
                importance=2.0,
            ),
            Order(
                id=2,
                pickup_node=4,
                dropoff_node=13,
                request_time_min=0.05,
                item=Item(weight_kg=0.5, volume_l=1.0),
                importance=5.0,
            ),
            Order(
                id=3,
                pickup_node=6,
                dropoff_node=11,
                request_time_min=0.10,
                item=Item(weight_kg=0.5, volume_l=1.0),
                importance=1.0,
            ),
        ),
    )
    scenario_path = tmp_path / "smoke_scenario.json"
    scenario.save_json(scenario_path)
    return graph_path, scenario_path


def _run_policy(
    script_name: str, graph_path: Path, scenario_path: Path, output: Path,
    *extra_args: str,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--graph", str(graph_path),
        "--scenario", str(scenario_path),
        "--output", str(output),
        *extra_args,
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, (
        f"{script_name} exited with {process.returncode}:\n{process.stdout}"
    )
    assert output.exists(), f"{script_name} did not create {output}"
    return json.loads(output.read_text(encoding="utf-8"))


def test_haversine_runner_completes_and_matches_full_search_when_k_is_all(
    tmp_path: Path,
) -> None:
    graph_path, scenario_path = _smoke_inputs(tmp_path)

    exact = _run_policy(
        "run_nyc_benchmark5_loss.py",
        graph_path, scenario_path, tmp_path / "exact.json",
    )
    heuristic = _run_policy(
        "run_nyc_heuristic_policy.py",
        graph_path, scenario_path, tmp_path / "heuristic.json",
        "--shortlist-k", "1",
    )

    assert exact["orders"] == heuristic["orders"] == 3
    assert exact["robots"] == heuristic["robots"] == 1
    assert exact["delivered"] == heuristic["delivered"] == 3
    assert heuristic["policy"] == "haversine_top_k_exact_incremental_loss"
    assert math.isclose(
        heuristic["loss_objective"],
        exact["loss_objective"],
        rel_tol=0,
        abs_tol=1e-7,
    ), "Top-K with K=all eligible robots must match the exact policy"
    assert heuristic["candidate_selection_calls"] >= 3
    assert heuristic["exact_sequence_evaluations"] > 0
    assert heuristic["assignments_to_busy_robots"] >= 1, (
        "The smoke scenario must exercise insertion on a busy robot"
    )
