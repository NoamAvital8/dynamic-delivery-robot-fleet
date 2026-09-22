from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import numpy as np

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.fleet import RobotType
from delivery_fleet.reservation_nn import ReservationFCNN
from delivery_fleet.scenario_creator import Item, Order, Scenario


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def test_parallel_perfect_information_target_generation(tmp_path) -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node].update(
            x=34.0 + node * 0.00001,
            y=32.0,
            in_cluster=0,
            is_cluster_representative=(node == 1),
            is_charging_station=True,
        )
    nx.set_edge_attributes(graph, 10.0, "length")
    annotate_nearest_charging_stations(graph, [0, 1, 2])
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    scenario_path = tmp_path / "scenario.json"
    Scenario(
        graph_name="parallel_training_test",
        duration_minutes=10.0,
        seed=42,
        orders=(
            Order(
                id=0,
                pickup_node=0,
                dropoff_node=2,
                request_time_min=0.0,
                item=Item(weight_kg=1.0, volume_l=1.0),
                importance=2.0,
            ),
        ),
    ).save_json(scenario_path)
    robot_types = [item.value for item in RobotType]
    general = {robot_type: [1.0, 0.0, 0.0] for robot_type in robot_types}
    high = {robot_type: [0.0, 0.0, 1.0] for robot_type in robot_types}
    manifest = {
        "robot_types": robot_types,
        "importance_levels": [1.0, 2.0, 5.0],
        "importance_prior_rates_per_hour": {"1": 60.0, "2": 10.0, "5": 5.0},
        "shortlist_k": 1,
        "instances": [
            {"id": "tiny", "graph": str(graph_path), "scenario": str(scenario_path)}
        ],
        "candidates": [
            {"id": "general", "fractions_by_type": general},
            {"id": "high", "fractions_by_type": high},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset_path = tmp_path / "targets.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_reservation_training_data.py"),
            str(manifest_path),
            str(dataset_path),
            "--processes",
            "2",
        ],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        assert data["features"].shape == (1, 4)
        assert data["targets"].shape == (1, 4, 3)
        assert str(data["best_candidate_id"][0]) == "general"


def test_parallel_training_restarts_select_and_save_model(tmp_path) -> None:
    rng = np.random.default_rng(12)
    features = rng.uniform(size=(30, 4))
    targets = np.zeros((30, 4, 3))
    targets[:, :, 0] = 1.0
    targets[features[:, 0] > 0.5, :, 0] = 0.0
    targets[features[:, 0] > 0.5, :, 2] = 1.0
    metadata = json.dumps(
        {
            "robot_types": [item.value for item in RobotType],
            "importance_levels": [1.0, 2.0, 5.0],
        }
    )
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        features=features,
        targets=targets,
        metadata=np.asarray(metadata),
    )
    model_path = tmp_path / "model.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_reservation_fcnn.py"),
            str(dataset),
            str(model_path),
            "--processes",
            "2",
            "--restarts",
            "3",
            "--epochs",
            "100",
        ],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    model = ReservationFCNN.load(model_path)
    assert model.input_dim == 4
    assert model.hidden_dim == 3
    assert model_path.with_suffix(".training.json").exists()
