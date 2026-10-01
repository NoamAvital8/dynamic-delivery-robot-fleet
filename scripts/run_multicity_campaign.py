from __future__ import annotations

"""Resumable five-city training and six-policy experiment campaign.

Launch with nohup on the VM. Completed target evaluations, models, and policy
results are durable; restarting this command skips verified completed work.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CITIES = ("tel_aviv", "haifa", "manhattan", "new_york_city", "barcelona")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_stage(name: str, command: list[str], run_dir: Path, state: dict[str, Any]) -> None:
    stage_dir = run_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{name}.log"
    state["current_stage"] = name
    state["stages"][name] = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "log": str(log_path),
    }
    _save_json(run_dir / "campaign_status.json", state)
    print(f"START {name}: {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- {datetime.now(timezone.utc).isoformat()} ---\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    stage = state["stages"][name]
    stage["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stage["exit_code"] = completed.returncode
    stage["status"] = "complete" if completed.returncode == 0 else "failed"
    _save_json(run_dir / "campaign_status.json", state)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit {completed.returncode}; see {log_path}")
    print(f"DONE {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/graphs")
    parser.add_argument("--train-per-city", type=int, default=20)
    parser.add_argument("--test-per-city", type=int, default=5)
    parser.add_argument("--duration-hours", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261001)
    parser.add_argument("--target-processes", type=int, default=12)
    parser.add_argument("--nn-processes", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--benchmark-processes", type=int, default=4)
    parser.add_argument("--idle-processes", type=int, default=4)
    parser.add_argument("--max-cluster-fraction", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("train_per_city", "test_per_city", "duration_hours", "target_processes",
                 "nn_processes", "restarts", "epochs", "benchmark_processes", "idle_processes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not (0.0 < args.max_cluster_fraction < 1.0):
        parser.error("--max-cluster-fraction must be between zero and one")

    run_dir = args.run_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "campaign_status.json"
    configuration = {
        "graph_dir": str(graph_dir), "train_per_city": args.train_per_city,
        "test_per_city": args.test_per_city, "duration_hours": args.duration_hours,
        "base_seed": args.base_seed, "target_processes": args.target_processes,
        "nn_processes": args.nn_processes, "restarts": args.restarts,
        "epochs": args.epochs, "benchmark_processes": args.benchmark_processes,
        "idle_processes": args.idle_processes,
        "max_cluster_fraction": args.max_cluster_fraction,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration") != configuration:
            raise ValueError("existing campaign has different parameters; use a new run directory")
    else:
        state = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                 "configuration": configuration, "stages": {}}
    state["status"] = "running"
    _save_json(state_path, state)

    suite_dir = run_dir / "suite"
    suite_path = suite_dir / "suite.json"
    try:
        for city in CITIES:
            graph = graph_dir / f"{city}.graphml"
            metadata = graph_dir / f"{city}_clusters.json"
            if not graph.is_file():
                raise FileNotFoundError(graph)
            needs_rebuild = True
            if metadata.is_file():
                old = json.loads(metadata.read_text(encoding="utf-8"))
                needs_rebuild = old.get("cluster_selection_method") != "leaf"
            if needs_rebuild:
                _run_stage(f"clusters_{city}", [sys.executable,
                    str(ROOT / "scripts/initialize_demand_clusters.py"), str(graph),
                    "--cluster-selection-method", "leaf"], run_dir, state)
            info = json.loads(metadata.read_text(encoding="utf-8"))
            fraction = float(info["largest_cluster_fraction"])
            if fraction > args.max_cluster_fraction:
                raise RuntimeError(
                    f"{city}: largest cluster covers {fraction:.1%} of nodes; "
                    "do not train on this partition"
                )
            print(f"CLUSTERS {city}: {info['clusters']} regions, largest={fraction:.1%}", flush=True)

        if not suite_path.is_file():
            _run_stage("generate_suite", [
                sys.executable, str(ROOT / "scripts/create_multicity_suite.py"),
                "--graph-dir", str(graph_dir), "--output-dir", str(suite_dir),
                "--train-per-city", str(args.train_per_city),
                "--test-per-city", str(args.test_per_city),
                "--duration-hours", str(args.duration_hours),
                "--base-seed", str(args.base_seed),
                "--overwrite",
            ], run_dir, state)

        models: dict[str, Path] = {}
        for style, label in (("spatial_posterior", "spatial"),
                             ("paper_moving_average", "paper")):
            training_dir = run_dir / "training" / label
            dataset = training_dir / "targets.npz"
            model = training_dir / "reservation_fcnn.npz"
            models[label] = model
            if not dataset.is_file():
                _run_stage(f"targets_{label}", [
                    sys.executable, str(ROOT / "scripts/generate_reservation_training_data.py"),
                    str(suite_dir / f"training_{style}.json"), str(dataset),
                    "--processes", str(args.target_processes),
                    "--work-dir", str(training_dir / "jobs"),
                ], run_dir, state)
            if not model.is_file():
                _run_stage(f"train_{label}", [
                    sys.executable, str(ROOT / "scripts/train_reservation_fcnn.py"),
                    str(dataset), str(model), "--processes", str(args.nn_processes),
                    "--restarts", str(args.restarts), "--epochs", str(args.epochs),
                    "--learning-rate", "0.01", "--validation-fraction", "0.2",
                    "--threads-per-process", "1",
                ], run_dir, state)

        _run_stage("benchmarks", [
            sys.executable, str(ROOT / "scripts/run_multicity_experiments.py"),
            str(suite_path), str(run_dir / "benchmarks"),
            "--spatial-model", str(models["spatial"]),
            "--paper-model", str(models["paper"]),
            "--processes", str(args.benchmark_processes),
            "--idle-processes", str(args.idle_processes),
        ], run_dir, state)
        state["status"] = "complete"
        state["current_stage"] = None
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _save_json(state_path, state)


if __name__ == "__main__":
    main()
