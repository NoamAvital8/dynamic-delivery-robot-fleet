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
RESOURCE_KEYS = {
    "target_processes", "nn_processes", "benchmark_processes", "idle_processes",
    "max_processes", "memory_per_simulator_gib", "memory_fraction",
    "compatible_source_fingerprints",
}


def _available_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _effective_workers(
    *, requested: int, max_processes: int, logical_cpus: int,
    available_memory_bytes: int | None, memory_per_simulator_gib: float,
    memory_fraction: float, nested_processes: int = 0,
) -> int:
    cpu_limit = max(1, min(requested, max_processes, logical_cpus))
    if nested_processes:
        cpu_limit = max(1, min(cpu_limit, max_processes // (1 + nested_processes)))
    if available_memory_bytes is None:
        return cpu_limit
    budget = int(
        available_memory_bytes * memory_fraction
        / (memory_per_simulator_gib * (1024 ** 3))
    )
    return max(1, min(cpu_limit, budget))


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
    parser.add_argument("--max-processes", type=int, default=60,
                        help="Overall logical-CPU ceiling for this campaign.")
    parser.add_argument("--target-processes", type=int, default=60)
    parser.add_argument("--nn-processes", type=int, default=60)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--benchmark-processes", type=int, default=12)
    parser.add_argument("--idle-processes", type=int, default=4)
    parser.add_argument("--memory-per-simulator-gib", type=float, default=8.0)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--compatible-source-fingerprint", action="append", default=[])
    parser.add_argument("--max-cluster-fraction", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("train_per_city", "test_per_city", "duration_hours", "max_processes", "target_processes",
                 "nn_processes", "restarts", "epochs", "benchmark_processes", "idle_processes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not (0.0 < args.max_cluster_fraction < 1.0):
        parser.error("--max-cluster-fraction must be between zero and one")
    if args.memory_per_simulator_gib <= 0 or not (0.0 < args.memory_fraction <= 1.0):
        parser.error("memory-per-simulator-gib must be positive and memory-fraction in (0,1]")
    if args.max_processes < args.idle_processes + 1:
        parser.error("max-processes must fit a benchmark plus its idle workers")

    logical_cpus = os.cpu_count() or 1
    available_memory = _available_memory_bytes()
    target_workers = _effective_workers(
        requested=args.target_processes, max_processes=args.max_processes,
        logical_cpus=logical_cpus, available_memory_bytes=available_memory,
        memory_per_simulator_gib=args.memory_per_simulator_gib,
        memory_fraction=args.memory_fraction,
    )
    nn_workers = max(1, min(args.nn_processes, args.max_processes, logical_cpus))
    benchmark_workers = _effective_workers(
        requested=args.benchmark_processes, max_processes=args.max_processes,
        logical_cpus=logical_cpus, available_memory_bytes=available_memory,
        memory_per_simulator_gib=args.memory_per_simulator_gib,
        memory_fraction=args.memory_fraction,
        nested_processes=args.idle_processes,
    )
    available_gib = (
        f"{available_memory/(1024**3):.1f}" if available_memory is not None
        else "unknown"
    )
    print(f"RESOURCE_BUDGET logical_cpus={logical_cpus} "
          f"max_processes={args.max_processes} mem_available_gib={available_gib} "
          f"target_workers={target_workers} nn_workers={nn_workers} "
          f"benchmark_workers={benchmark_workers} idle_workers_per_benchmark={args.idle_processes}",
          flush=True)

    run_dir = args.run_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "campaign_status.json"
    configuration = {
        "graph_dir": str(graph_dir), "train_per_city": args.train_per_city,
        "test_per_city": args.test_per_city, "duration_hours": args.duration_hours,
        "base_seed": args.base_seed, "target_processes": target_workers,
        "nn_processes": nn_workers, "restarts": args.restarts,
        "epochs": args.epochs, "benchmark_processes": args.benchmark_processes,
        "idle_processes": args.idle_processes,
        "max_cluster_fraction": args.max_cluster_fraction,
        "max_processes": args.max_processes,
        "memory_per_simulator_gib": args.memory_per_simulator_gib,
        "memory_fraction": args.memory_fraction,
        "compatible_source_fingerprints": args.compatible_source_fingerprint,
    }
    configuration["benchmark_processes"] = benchmark_workers
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous = state.get("configuration", {})
        semantic = lambda value: {key: item for key, item in value.items()
                                  if key not in RESOURCE_KEYS}
        if semantic(previous) != semantic(configuration):
            raise ValueError("existing campaign has different scientific parameters; use a new run directory")
        if previous != configuration:
            state.setdefault("resource_history", []).append({
                "changed_utc": datetime.now(timezone.utc).isoformat(),
                "previous": previous, "updated": configuration,
            })
            state["configuration"] = configuration
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
                    "--processes", str(target_workers),
                    "--work-dir", str(training_dir / "jobs"),
                    *[value for fingerprint in args.compatible_source_fingerprint
                      for value in ("--compatible-source-fingerprint", fingerprint)],
                ], run_dir, state)
            if not model.is_file():
                _run_stage(f"train_{label}", [
                    sys.executable, str(ROOT / "scripts/train_reservation_fcnn.py"),
                    str(dataset), str(model), "--processes", str(nn_workers),
                    "--restarts", str(args.restarts), "--epochs", str(args.epochs),
                    "--learning-rate", "0.01", "--validation-fraction", "0.2",
                    "--threads-per-process", "1",
                ], run_dir, state)

        _run_stage("benchmarks", [
            sys.executable, str(ROOT / "scripts/run_multicity_experiments.py"),
            str(suite_path), str(run_dir / "benchmarks"),
            "--spatial-model", str(models["spatial"]),
            "--paper-model", str(models["paper"]),
            "--processes", str(benchmark_workers),
            "--idle-processes", str(args.idle_processes),
            "--timeout-seconds", "0",
            *[value for fingerprint in args.compatible_source_fingerprint
              for value in ("--compatible-source-fingerprint", fingerprint)],
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
