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
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for directory in (ROOT / "scripts", ROOT / "src" / "delivery_fleet"):
        for path in sorted(directory.glob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _evaluate_job(job: dict[str, Any]) -> tuple[str, str, float, int, str]:
    output = Path(job["output"])
    stamp_path = output.with_suffix(".stamp.json")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.exists() else {}
    prior_signature = stamp.get("signature")
    compatible = prior_signature in job.get("compatible_signatures", ())
    if output.exists() and (prior_signature == job["signature"] or compatible):
        result = json.loads(output.read_text(encoding="utf-8"))
        loss = float(result["loss_objective"])
        fleet_count = int(result["robots"])
        if not np.isfinite(loss) or fleet_count <= 0:
            raise ValueError(f"invalid cached simulator result: {output}")
        if compatible:
            temporary = stamp_path.with_suffix(".stamp.tmp")
            temporary.write_text(json.dumps({
                "signature": job["signature"],
                "migrated_from_signature": prior_signature,
                "migration_reason": "completed before infeasible-baseline recovery fix",
            }), encoding="utf-8")
            temporary.replace(stamp_path)
        return (
            job["instance_id"],
            job["candidate_id"],
            loss,
            fleet_count,
            "cached-compatible" if compatible else "cached",
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
        "--reservation-style",
        job["reservation_style"],
        "--reservation-lookback-min",
        str(job["reservation_lookback_min"]),
    ]
    log_path = output.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=job["timeout_seconds"],
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"simulation failed for {job['instance_id']}/{job['candidate_id']}\n"
            f"log: {log_path}\n"
            f"tail:\n{log_path.read_text(encoding='utf-8')[-4000:]}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    temporary = stamp_path.with_suffix(".stamp.tmp")
    temporary.write_text(json.dumps({"signature": job["signature"]}), encoding="utf-8")
    temporary.replace(stamp_path)
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
    parser.add_argument(
        "--compatible-source-fingerprint", action="append", default=[],
        help="Explicitly trust completed jobs made with this older source hash; "
             "the existing result is validated and its stamp records the migration.",
    )
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
    default_prior_rates = payload["importance_prior_rates_per_hour"]
    reservation_style = str(payload.get("reservation_style", "spatial_posterior"))
    if reservation_style not in {"spatial_posterior", "paper_moving_average"}:
        raise ValueError("unknown reservation_style")
    reservation_lookback_min = float(payload.get("reservation_lookback_min", 100.0))
    if reservation_lookback_min <= 0:
        raise ValueError("reservation_lookback_min must be positive")
    source_fingerprint = _source_fingerprint()
    for fingerprint in args.compatible_source_fingerprint:
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            parser.error("--compatible-source-fingerprint must be a SHA-256 hex digest")
    graph_hashes: dict[Path, str] = {}
    for raw in instances:
        instance_id = _safe_id(raw["id"])
        if instance_id in instance_by_id:
            raise ValueError(f"duplicate instance id {instance_id!r}")
        instance_by_id[instance_id] = raw
        graph = _resolve(manifest_dir, raw["graph"])
        scenario = _resolve(manifest_dir, raw["scenario"])
        prior_rates_json = json.dumps(
            raw.get("importance_prior_rates_per_hour", default_prior_rates),
            sort_keys=True,
        )
        if not graph.is_file() or not scenario.is_file():
            raise FileNotFoundError(
                f"missing graph or scenario for training instance {instance_id!r}"
            )
        if graph not in graph_hashes:
            graph_hashes[graph] = _sha256(graph)
        graph_hash = graph_hashes[graph]
        scenario_hash = _sha256(scenario)
        for candidate_id in candidate_by_id:
            signature_payload = {
                "graph": graph_hash,
                "scenario": scenario_hash,
                "fixed_config": _sha256(config_by_id[candidate_id]),
                "shortlist_k": int(payload.get("shortlist_k", 10)),
                "prior_rates_json": prior_rates_json,
                "prior_concentration": float(payload.get("prior_concentration", 4.0)),
                "reservation_style": reservation_style,
                "reservation_lookback_min": reservation_lookback_min,
                "source": source_fingerprint,
            }
            signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
            compatible_signatures = []
            for old_source in args.compatible_source_fingerprint:
                signature_payload["source"] = old_source
                compatible_signatures.append(hashlib.sha256(
                    json.dumps(signature_payload, sort_keys=True).encode()
                ).hexdigest())
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
                    "reservation_style": reservation_style,
                    "reservation_lookback_min": reservation_lookback_min,
                    "signature": signature,
                    "compatible_signatures": compatible_signatures,
                    "python": str(args.python),
                    "timeout_seconds": args.timeout_seconds,
                }
            )

    losses: dict[str, dict[str, float]] = defaultdict(dict)
    fleet_counts: dict[str, int] = {}
    worker_count = min(args.processes, len(jobs))
    for job in jobs:
        output = Path(job["output"])
        try:
            signature = json.loads(output.with_suffix(".stamp.json").read_text(
                encoding="utf-8"
            )).get("signature")
            job["already_complete"] = output.is_file() and (
                signature == job["signature"]
                or signature in job["compatible_signatures"]
            )
        except (FileNotFoundError, json.JSONDecodeError):
            job["already_complete"] = False
    pending_count = sum(not job["already_complete"] for job in jobs)
    print(f"evaluations={len(jobs)} cached={len(jobs)-pending_count} "
          f"pending={pending_count} processes={worker_count} work_dir={work_dir}", flush=True)
    started = time.monotonic()
    newly_completed = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_evaluate_job, job): job for job in jobs}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            instance_id, candidate_id, loss, fleet_count, source = future.result()
            if not futures[future]["already_complete"]:
                newly_completed += 1
            losses[instance_id][candidate_id] = loss
            previous_count = fleet_counts.setdefault(instance_id, fleet_count)
            if previous_count != fleet_count:
                raise RuntimeError("fleet count changed between candidate evaluations")
            elapsed = time.monotonic() - started
            eta_text = (
                f"{elapsed/newly_completed*(pending_count-newly_completed):.0f}s"
                if newly_completed else "unknown"
            )
            print(
                f"[{completed_count}/{len(jobs)}] {instance_id}/{candidate_id} "
                f"loss={loss:.6f} ({source}) elapsed={elapsed:.0f}s "
                f"eta={eta_text}",
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
