from __future__ import annotations

"""Compare online policies against a proven tiny hindsight optimum.

The generated two-robot path cases certify that payload and battery cannot
bind, so exhaustive assignment/route enumeration is globally exact. This is
an oracle lower bound, not an online policy with access to future requests.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.exact_small_oracle import solve_small_hindsight
from delivery_fleet.scenario_creator import Item, Order, Scenario
from run_multicity_experiments import POLICIES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prepare_suite(root: Path, count: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / "tiny_path.graphml"
    if not graph_path.exists():
        graph = nx.path_graph(501)  # ceil(501/500) = two default robots.
        for node in graph.nodes:
            graph.nodes[node].update(
                x=34.0 + 0.00001 * node,
                y=32.0,
                in_cluster=0,
                is_cluster_representative=(node == 250),
                is_charging_station=(node in {0, 250, 500}),
            )
        nx.set_edge_attributes(graph, 1.0, "length")
        annotate_nearest_charging_stations(graph, [0, 250, 500])
        nx.write_graphml(graph, graph_path)

    records = []
    choices = np.asarray([25, 90, 160, 250, 340, 415, 475])
    for index in range(count):
        seed = 31000 + index
        scenario_path = root / f"tiny_test_{index:03d}.json"
        if not scenario_path.exists():
            rng = np.random.default_rng(seed)
            selected = rng.choice(choices, size=8, replace=True)
            priorities = rng.permutation([1.0, 2.0, 5.0, 5.0])
            orders = []
            for order_id in range(4):
                pickup = int(selected[2 * order_id])
                dropoff = int(selected[2 * order_id + 1])
                if pickup == dropoff:
                    dropoff = int(choices[(order_id + index + 3) % len(choices)])
                    if pickup == dropoff:
                        dropoff = int(choices[(order_id + index + 4) % len(choices)])
                orders.append(Order(
                    id=order_id, pickup_node=pickup, dropoff_node=dropoff,
                    request_time_min=(0.0, 0.5, 1.5, 3.0)[order_id],
                    item=Item(1.0, 1.0), importance=float(priorities[order_id]),
                ))
            Scenario("tiny_path", 15.0, seed, tuple(orders)).save_json(scenario_path)
        scenario = Scenario.load_json(scenario_path)
        records.append({
            "id": f"tiny_test_{index:03d}", "scenario": scenario_path.name,
            "seed": scenario.seed, "orders": len(scenario.orders),
        })
    suite_path = root / "suite.json"
    _save_json(suite_path, {"cities": {"tiny_path": {
        "graph": graph_path.name,
        "prior_rates_per_hour": {"1": 6.0, "2": 3.0, "5": 3.0},
        "test": records,
    }}})
    return suite_path


def _write_comparison(run_dir: Path, suite: dict, oracle_dir: Path) -> None:
    benchmark_path = run_dir / "benchmarks" / "summary.csv"
    if not benchmark_path.is_file():
        return
    rows = []
    with benchmark_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            oracle_path = oracle_dir / f"{row['scenario_id']}.json"
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            optimum = float(oracle["optimal_loss"])
            loss = float(row["loss_objective"]) if row["status"] == "complete" else None
            if loss is not None and loss < optimum - 1e-6:
                raise AssertionError(
                    f"{row['policy']} beat the certified optimum on {row['scenario_id']}: "
                    f"{loss} < {optimum}"
                )
            rows.append({
                "scenario_id": row["scenario_id"], "policy": row["policy"],
                "status": row["status"], "optimal_loss": optimum,
                "policy_loss": "" if loss is None else loss,
                "gap_percent": "" if loss is None else 100.0 * (loss / optimum - 1.0),
                "optimal_on_time": oracle["on_time"],
                "policy_on_time": row["on_time"], "policy_late": row["late"],
                "wall_clock_seconds": row["wall_clock_seconds"],
            })
    path = run_dir / "oracle_comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--idle-processes", type=int, default=2)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=["myopic_ab", "reactive_insertion"])
    parser.add_argument("--spatial-model", type=Path, default=Path("missing_spatial.npz"))
    parser.add_argument("--paper-model", type=Path, default=Path("missing_paper.npz"))
    args = parser.parse_args()
    if not 1 <= args.scenarios <= 20 or args.processes <= 0 or args.idle_processes <= 0:
        parser.error("scenarios must be 1..20 and process counts must be positive")

    run_dir = args.run_dir.resolve()
    suite_path = _prepare_suite(run_dir / "suite", args.scenarios)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    graph_path = suite_path.parent / "tiny_path.graphml"
    graph = nx.read_graphml(graph_path, node_type=int)
    oracle_dir = run_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = suite["cities"]["tiny_path"]["test"]
    for index, record in enumerate(records, start=1):
        scenario_path = suite_path.parent / record["scenario"]
        signature = hashlib.sha256(json.dumps({
            "graph": _sha256(graph_path), "scenario": _sha256(scenario_path),
            "oracle_source": _sha256(ROOT / "src/delivery_fleet/exact_small_oracle.py"),
        }, sort_keys=True).encode()).hexdigest()
        output = oracle_dir / f"{record['id']}.json"
        if output.is_file():
            cached = json.loads(output.read_text(encoding="utf-8"))
            if cached.get("input_signature") == signature:
                print(f"oracle [{index}/{len(records)}] {record['id']} cached", flush=True)
                continue
        result = solve_small_hindsight(
            graph, Scenario.load_json(scenario_path), max_orders=5, max_robots=3
        )
        result["input_signature"] = signature
        _save_json(output, result)
        elapsed = time.monotonic() - started
        print(f"oracle [{index}/{len(records)}] {record['id']} loss={result['optimal_loss']:.6f} "
              f"elapsed={elapsed:.1f}s eta={elapsed/index*(len(records)-index):.1f}s",
              flush=True)

    command = [
        sys.executable, str(ROOT / "scripts/run_multicity_experiments.py"),
        str(suite_path), str(run_dir / "benchmarks"),
        "--spatial-model", str(args.spatial_model.resolve()),
        "--paper-model", str(args.paper_model.resolve()),
        "--processes", str(args.processes),
        "--idle-processes", str(args.idle_processes),
        "--policies", *args.policies,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    finally:
        _write_comparison(run_dir, suite, oracle_dir)
    print(f"comparison={run_dir / 'oracle_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
