from __future__ import annotations

"""Generate perfect-information FCNN targets in parallel.

For every training scenario and fixed heterogeneous alpha candidate, this
script runs the real heuristic simulator. The candidate producing the lowest
final loss becomes that scenario's supervised target, following the paper's
offline perfect-information procedure. Only training scenarios may be listed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.reservation_nn import (
    FixedReservationModel,
    ReservationFeatureSchema,
    build_reservation_features,
)
from delivery_fleet.scenario_creator import Scenario


def _safe_id(value: object) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"unsafe id {text!r}; use letters, numbers, dot, dash or underscore")
    return text


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _evaluate_job(job: dict[str, Any]) -> tuple[str, str, float, int, str]:
    output = Path(job["output"])
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        return (
            job["instance_id"],
            job["candidate_id"],
            float(result["loss_objective"]),
            int(result["robots"]),
            "cached",
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    command = [
        job["python"],
        str(ROOT / "scripts" / "run_nyc_heuristic_policy.py"),
        "--graph",
        job["graph"],
        "--scenario",
        job["scenario"],
        "--output",
        str(output),
        "--shortlist-k",
        str(job["shortlist_k"]),
        "--fixed-reservation-fractions",
        job["fixed_config"],
        "--importance-prior-rates-per-hour",
        job["prior_rates_json"],
        "--prior-concentration",
        str(job["prior_concentration"]),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=job["timeout_seconds"],
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"simulation failed for {job['instance_id']}/{job['candidate_id']}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    return (
        job["instance_id"],
        job["candidate_id"],
        float(result["loss_objective"]),
        int(result["robots"]),
        "ran",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=86_400)
    args = parser.parse_args()
    if args.processes <= 0:
        parser.error("--processes must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    robot_types = tuple(str(value) for value in payload["robot_types"])
    levels = tuple(float(value) for value in payload["importance_levels"])
    schema = ReservationFeatureSchema(robot_types, levels)
    candidates = payload["candidates"]
    instances = payload["instances"]
    if not candidates or not instances:
        raise ValueError("manifest must contain candidates and training instances")

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else (args.output.parent / f"{args.output.stem}_jobs").resolve()
    )
    config_dir = work_dir / "fixed_alpha"
    result_dir = work_dir / "results"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    candidate_by_id: dict[str, dict[str, Any]] = {}
    config_by_id: dict[str, Path] = {}
    for raw in candidates:
        candidate_id = _safe_id(raw["id"])
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate candidate id {candidate_id!r}")
        config = {
            "robot_types": robot_types,
            "importance_levels": levels,
            "fractions_by_type": raw["fractions_by_type"],
        }
        FixedReservationModel(
            robot_types, levels, raw["fractions_by_type"]
        )
        config_path = config_dir / f"{candidate_id}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        candidate_by_id[candidate_id] = raw
        config_by_id[candidate_id] = config_path

    instance_by_id: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    prior_rates_json = json.dumps(payload["importance_prior_rates_per_hour"])
    for raw in instances:
        instance_id = _safe_id(raw["id"])
        if instance_id in instance_by_id:
            raise ValueError(f"duplicate instance id {instance_id!r}")
        instance_by_id[instance_id] = raw
        graph = _resolve(manifest_dir, raw["graph"])
        scenario = _resolve(manifest_dir, raw["scenario"])
        if not graph.is_file() or not scenario.is_file():
            raise FileNotFoundError(
                f"missing graph or scenario for training instance {instance_id!r}"
            )
        for candidate_id in candidate_by_id:
            jobs.append(
                {
                    "instance_id": instance_id,
                    "candidate_id": candidate_id,
                    "graph": str(graph),
                    "scenario": str(scenario),
                    "output": str(result_dir / f"{instance_id}__{candidate_id}.json"),
                    "fixed_config": str(config_by_id[candidate_id]),
                    "shortlist_k": int(payload.get("shortlist_k", 10)),
                    "prior_rates_json": prior_rates_json,
                    "prior_concentration": float(payload.get("prior_concentration", 4.0)),
                    "python": str(args.python),
                    "timeout_seconds": args.timeout_seconds,
                }
            )

    losses: dict[str, dict[str, float]] = defaultdict(dict)
    fleet_counts: dict[str, int] = {}
    worker_count = min(args.processes, len(jobs))
    print(f"evaluations={len(jobs)} processes={worker_count} work_dir={work_dir}", flush=True)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [executor.submit(_evaluate_job, job) for job in jobs]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            instance_id, candidate_id, loss, fleet_count, source = future.result()
            losses[instance_id][candidate_id] = loss
            previous_count = fleet_counts.setdefault(instance_id, fleet_count)
            if previous_count != fleet_count:
                raise RuntimeError("fleet count changed between candidate evaluations")
            print(
                f"[{completed_count}/{len(jobs)}] {instance_id}/{candidate_id} "
                f"loss={loss:.6f} ({source})",
                flush=True,
            )

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    best_losses: list[float] = []
    best_candidates: list[str] = []
    candidate_order = list(candidate_by_id)
    for instance_id, raw in instance_by_id.items():
        scenario = Scenario.load_json(_resolve(manifest_dir, raw["scenario"]))
        counts = Counter(float(order.importance) for order in scenario.orders)
        features.append(
            build_reservation_features(
                schema,
                predicted_requests_by_importance=counts,
                fleet_count=fleet_counts[instance_id],
            )
        )
        candidate_id = min(
            candidate_order,
            key=lambda value: (losses[instance_id][value], candidate_order.index(value)),
        )
        best = candidate_by_id[candidate_id]["fractions_by_type"]
        targets.append(
            np.asarray([best[robot_type] for robot_type in robot_types], dtype=float)
        )
        best_candidates.append(candidate_id)
        best_losses.append(losses[instance_id][candidate_id])

    metadata = json.dumps(
        {
            "source_manifest": str(manifest_path),
            "robot_types": robot_types,
            "importance_levels": levels,
            "feature_names": schema.names,
            "target_method": "minimum final simulation loss under fixed alpha",
            "paper_alignment": "C class shares plus total requests per robot",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        targets=np.stack(targets),
        perfect_information_loss=np.asarray(best_losses),
        best_candidate_id=np.asarray(best_candidates),
        metadata=np.asarray(metadata),
    )
    print(f"saved={args.output} samples={len(features)}", flush=True)


if __name__ == "__main__":
    main()
