from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.deadlines import delivery_loss, delivery_time_allowance_min
from delivery_fleet.exact_small_oracle import solve_small_hindsight
from delivery_fleet.fleet import create_default_fleet
from delivery_fleet.scenario_creator import Item, Order, Scenario


def _graph() -> nx.Graph:
    graph = nx.path_graph(501)
    nx.set_edge_attributes(graph, 1.0, "length")
    return graph


def test_single_order_matches_direct_best_robot() -> None:
    graph = _graph()
    order = Order(0, 40, 300, 2.0, Item(1.0, 1.0), 5.0)
    scenario = Scenario("tiny", 15.0, 31, (order,))
    result = solve_small_hindsight(graph, scenario)
    allowance = delivery_time_allowance_min(260.0, 5.0)
    expected = min(
        delivery_loss(
            max(2.0, abs(int(robot.node_id) - 40) / robot.spec.speed_mps / 60.0)
            + 1.0 + 260.0 / robot.spec.speed_mps / 60.0 + 1.0 - 2.0,
            allowance, 5.0,
        )
        for robot in create_default_fleet(graph, seed=31)
    )
    assert math.isclose(result["optimal_loss"], expected, abs_tol=1e-9)
    assert result["proven_optimal"] is True
    assert sum(len(route["assigned_order_ids"]) for route in result["routes"]) == 1


def test_small_oracle_comparison_runs_and_resumes(tmp_path: Path) -> None:
    command = [
        sys.executable, str(ROOT / "scripts/run_small_optimal_comparison.py"),
        str(tmp_path / "tiny_run"), "--scenarios", "1", "--processes", "2",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, cwd=ROOT, env=environment, check=True,
                   capture_output=True, text=True, timeout=180)
    comparison = tmp_path / "tiny_run/oracle_comparison.csv"
    with comparison.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["policy"] for row in rows} == {"myopic_ab", "reactive_insertion"}
    assert all(row["status"] == "complete" for row in rows)
    assert all(float(row["gap_percent"]) >= -1e-8 for row in rows)
    subprocess.run(command, cwd=ROOT, env=environment, check=True,
                   capture_output=True, text=True, timeout=30)
