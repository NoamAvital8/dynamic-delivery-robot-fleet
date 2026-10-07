from __future__ import annotations

"""Run paired policies across held-out city scenarios with durable results."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_fleet.mle_reservation import MLEConfig, add_mle_arguments, mle_config_from_args

MLE_POLICIES = ("queue_mle_reservation", "full_mle_reservation")
DEFAULT_POLICIES = (
    "full",
    "myopic_ab",
    "reactive_insertion",
    "paper_sa_adapted",
    "full_no_nn",
    "full_no_idle",
)
POLICIES = (*DEFAULT_POLICIES, "full_uncertainty_idle", "full_uncertainty_idle_no_nn",
            "full_queue_aware", "full_queue_aware_no_nn",
            "full_coordinated_idle", "full_coordinated_idle_no_nn", *MLE_POLICIES)
MLE_FIELDS = (
    "mle_idle_mode", "mle_processes", "mle_reservation_confidence",
    "mle_reservation_observed_arrivals", "mle_reservation_updates",
    "mle_reservation_activations", "mle_reservation_deactivations",
    "mle_reservation_active_minutes", "mle_reservation_planning_seconds",
    "mle_reservation_candidates_scored", "mle_reservation_budget_fallbacks",
    "mle_reservation_final_reason", "mle_reservation_last_decision_reason",
    "mle_reservation_first_activation_min", "mle_reservation_decisions_file",
)
QUEUE_FIELDS = (
    "policy_version", "assignment_queue_forecast", "queue_wait_records_file",
    "delivery_queue_wait_total_min", "delivery_queue_visit_count",
    "delivery_queue_waited_count", "delivery_queue_wait_mean_min",
    "delivery_queue_wait_mean_if_waited_min", "delivery_queue_wait_p95_min",
    "delivery_queue_wait_max_min", "delivery_queue_cancelled_count",
    "delivery_queue_cancelled_wait_total_min", "background_queue_wait_total_min",
    "delivery_queue_prediction_samples", "delivery_queue_prediction_mae_min",
    "delivery_queue_prediction_bias_min", "queue_forecast_missing_cached_plans",
    "queue_forecast_infeasible_cached_plans", "queue_alternative_prefix_selected",
)
FIELDS = (
    "city", "scenario_id", "seed", "policy", "status", "orders",
    "delivered", "on_time", "late", "loss_objective",
    "wall_clock_seconds", "simulation_finish_min", "result_file", "log_file", "error",
    "idle_reposition_actions", "idle_charge_actions", "idle_stay_decisions",
    "idle_planning_seconds", "idle_relocation_uncertainty_penalty",
    "idle_readiness_estimator", "idle_readiness_projections",
    "idle_readiness_missing_plan", "idle_readiness_infeasible_plan",
    "idle_readiness_queue_min",
    "idle_coordination", "idle_assignment_refresh_requests", "idle_assignment_refresh_epochs",
    "idle_retarget_actions", "idle_retarget_preserved_edges", "idle_cooldown_protected",
    "idle_commit_rejected", "idle_pending_blocked_epochs", "idle_switch_gain_fraction", "idle_retarget_cooldown_min",
    "idle_decisions_file",
    *QUEUE_FIELDS,
    *MLE_FIELDS,
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    for directory in (ROOT / "scripts", ROOT / "src" / "delivery_fleet"):
        for path in sorted(directory.glob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # A summary reader or Windows scanner may briefly hold the old
            # status file. Keep the complete temporary payload and retry.
            if attempt == 4:
                raise
            time.sleep(0.05)


def policy_command(
    policy: str,
    *,
    python: str,
    graph: Path,
    scenario: Path,
    output: Path,
    spatial_model: Path,
    paper_model: Path,
    shortlist_k: int,
    idle_processes: int,
    prior_rates: dict[str, float],
    prior_concentration: float,
    relocation_uncertainty_penalty: float = 1.0,
    switching_gain_fraction: float = 0.005,
    retarget_cooldown_min: float = 2.0,
    mle_config: MLEConfig | None = None,
) -> list[str]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    script = {
        "full": "run_nyc_anticipatory_idle_policy.py",
        "myopic_ab": "run_nyc_benchmark5.py",
        "reactive_insertion": "run_nyc_benchmark5_loss.py",
        "paper_sa_adapted": "run_nyc_heuristic_policy.py",
        "full_no_nn": "run_nyc_anticipatory_idle_policy.py",
        "full_no_idle": "run_nyc_heuristic_policy.py",
        "full_uncertainty_idle": "run_nyc_anticipatory_idle_policy.py",
        "full_uncertainty_idle_no_nn": "run_nyc_anticipatory_idle_policy.py",
        "full_queue_aware": "run_nyc_queue_aware_policy.py",
        "full_queue_aware_no_nn": "run_nyc_queue_aware_policy.py",
        "full_coordinated_idle": "run_nyc_coordinated_idle_policy.py",
        "full_coordinated_idle_no_nn": "run_nyc_coordinated_idle_policy.py",
        "queue_mle_reservation": "run_nyc_mle_reservation_policy.py",
        "full_mle_reservation": "run_nyc_mle_reservation_policy.py",
    }[policy]
    command = [python, str(ROOT / "scripts" / script), "--graph", str(graph),
               "--scenario", str(scenario), "--output", str(output)]
    if policy not in {"myopic_ab", "reactive_insertion"}:
        command += [
            "--shortlist-k", str(shortlist_k),
            "--importance-prior-rates-per-hour", json.dumps(prior_rates, sort_keys=True),
            "--prior-concentration", str(prior_concentration),
        ]
    if policy in {"full", "full_no_nn", "full_uncertainty_idle", "full_uncertainty_idle_no_nn",
                  "full_queue_aware", "full_queue_aware_no_nn",
                  "full_coordinated_idle", "full_coordinated_idle_no_nn", *MLE_POLICIES}:
        command += ["--idle-processes", str(idle_processes)]
    if policy in {"full", "full_no_idle", "full_uncertainty_idle", "full_queue_aware", "full_coordinated_idle"}:
        command += ["--reservation-model", str(spatial_model)]
    if policy in {"full_coordinated_idle", "full_coordinated_idle_no_nn", "full_mle_reservation"}:
        if any(not math.isfinite(v) or v < 0 for v in (switching_gain_fraction, retarget_cooldown_min)):
            raise ValueError("switch gain fraction and cooldown must be finite and non-negative")
        command += ["--idle-switch-gain-fraction", str(switching_gain_fraction),
                    "--idle-retarget-cooldown-min", str(retarget_cooldown_min)]
    if policy in {"full_uncertainty_idle", "full_uncertainty_idle_no_nn"}:
        if not math.isfinite(relocation_uncertainty_penalty) or relocation_uncertainty_penalty <= 0:
            raise ValueError("relocation uncertainty penalty must be finite and positive")
        command += ["--idle-relocation-uncertainty-penalty", str(relocation_uncertainty_penalty)]
    if policy == "paper_sa_adapted":
        command += ["--reservation-model", str(paper_model),
                    "--reservation-style", "paper_moving_average",
                    "--reservation-lookback-min", "100"]
    if policy in MLE_POLICIES:
        mode = "coordinated" if policy == "full_mle_reservation" else "legacy"
        command += ["--mle-idle-mode", mode]
        for name, value in asdict(mle_config or MLEConfig()).items():
            command += ["--mle-"+name.replace("_", "-"), str(value)]
    return command


def mle_worker_budget(simulators: int, idle_processes: int, mle_processes: int) -> int:
    """Inline scoring needs no child pool. Controller threads are not compute workers."""
    return simulators * (1 + (idle_processes if idle_processes > 1 else 0)
                         + (mle_processes if mle_processes > 1 else 0))


def order_jobs(jobs: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "city-major":
        return jobs
    if mode != "round-robin":
        raise ValueError(f"unknown job order {mode!r}")
    # Launch each policy/seed across cities before moving to the next policy.
    # Large-city jobs start early, while small maps provide early results.
    return sorted(jobs, key=lambda j: (j["scenario_index"], j["policy_index"], j["city_index"]))


def summarize(result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    delivered = int(result["delivered"])
    on_time = int(result["on_time"])
    orders = int(result["orders"])
    if not (0 <= on_time <= delivered <= orders):
        raise ValueError("invalid delivery/on-time counts in simulator result")
    return {
        "city": job["city"], "scenario_id": job["scenario_id"],
        "seed": job["seed"], "policy": job["policy"], "status": "complete",
        "orders": orders, "delivered": delivered, "on_time": on_time,
        "late": delivered - on_time,
        "loss_objective": float(result["loss_objective"]),
        "wall_clock_seconds": float(result["wall_clock_seconds"]),
        "simulation_finish_min": float(result["simulation_finish_min"]),
        "result_file": str(job["output"]), "log_file": str(job["log"]), "error": "",
        **{name: result.get(name, "") for name in FIELDS
           if name.startswith("idle_") or name in QUEUE_FIELDS or name in MLE_FIELDS},
    }


def _write_summary(results_dir: Path, jobs: list[dict[str, Any]]) -> None:
    rows = []
    for job in jobs:
        status_path = job["status_path"]
        row = None
        for attempt in range(5):
            try:
                row = json.loads(status_path.read_text(encoding="utf-8"))
                break
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                # A worker may be atomically replacing this file on Windows.
                if attempt < 4:
                    time.sleep(0.05)
        if row is None:
            row = {
                "city": job["city"], "scenario_id": job["scenario_id"],
                "seed": job["seed"], "policy": job["policy"], "status": "pending",
                "orders": job["orders"], "result_file": str(job["output"]),
                "log_file": str(job["log"]),
            }
        rows.append({field: row.get(field, "") for field in FIELDS})
    destination = results_dir / "summary.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["city"]), str(row["policy"]))
        bucket = grouped.setdefault(key, {
            "city": key[0], "policy": key[1], "planned_scenarios": 0,
            "completed_scenarios": 0, "total_loss": 0.0,
            "total_wall_clock_seconds": 0.0, "on_time": 0, "late": 0,
            "delivered": 0, "orders": 0,
            "mission_queue_measured_scenarios": 0, "delivery_queue_wait_total_min": 0.0,
            "delivery_queue_visit_count": 0, "delivery_queue_waited_count": 0,
        })
        bucket["planned_scenarios"] += 1
        if row["status"] == "complete":
            bucket["completed_scenarios"] += 1
            bucket["total_loss"] += float(row["loss_objective"])
            bucket["total_wall_clock_seconds"] += float(row["wall_clock_seconds"])
            for field in ("on_time", "late", "delivered", "orders"):
                bucket[field] += int(row[field])
            if row.get("delivery_queue_visit_count", "") != "":
                bucket["mission_queue_measured_scenarios"] += 1
                bucket["delivery_queue_wait_total_min"] += float(row["delivery_queue_wait_total_min"])
                for field in ("delivery_queue_visit_count", "delivery_queue_waited_count"):
                    bucket[field] += int(row[field])
    aggregate_fields = (
        "city", "policy", "completed_scenarios", "planned_scenarios",
        "total_loss", "mean_loss", "total_wall_clock_seconds",
        "mean_wall_clock_seconds", "on_time", "late", "delivered", "orders",
        "mission_queue_measured_scenarios", "delivery_queue_wait_total_min",
        "delivery_queue_visit_count", "delivery_queue_waited_count",
        "delivery_queue_wait_mean_min", "delivery_queue_wait_mean_if_waited_min",
    )
    aggregate_rows = []
    for bucket in grouped.values():
        completed = bucket["completed_scenarios"]
        bucket["mean_loss"] = bucket["total_loss"] / completed if completed else ""
        bucket["mean_wall_clock_seconds"] = (
            bucket["total_wall_clock_seconds"] / completed if completed else ""
        )
        measured = bucket["mission_queue_measured_scenarios"]
        visits = bucket["delivery_queue_visit_count"]
        waited = bucket["delivery_queue_waited_count"]
        total_wait = bucket["delivery_queue_wait_total_min"]
        bucket["delivery_queue_wait_mean_min"] = total_wait / visits if visits else (0.0 if measured else "")
        bucket["delivery_queue_wait_mean_if_waited_min"] = total_wait / waited if waited else (0.0 if measured else "")
        if not measured:
            for field in ("delivery_queue_wait_total_min", "delivery_queue_visit_count", "delivery_queue_waited_count"):
                bucket[field] = ""
        aggregate_rows.append(bucket)
    aggregate_path = results_dir / "city_policy_summary.csv"
    aggregate_tmp = aggregate_path.with_suffix(".csv.tmp")
    with aggregate_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    aggregate_tmp.replace(aggregate_path)


def _execute(job: dict[str, Any], timeout_seconds: int | None,
             adopt_existing_results: bool = False) -> dict[str, Any]:
    output: Path = job["output"]
    status_path: Path = job["status_path"]
    stamp_path: Path = job["stamp_path"]
    if output.is_file() and stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("signature") in job["accepted_signatures"]:
            row = summarize(json.loads(output.read_text(encoding="utf-8")), job)
            _atomic_json(status_path, row)
            return row
        raise RuntimeError(f"existing result has a different input signature: {output}")

    if output.is_file() and adopt_existing_results:
        row = summarize(json.loads(output.read_text(encoding="utf-8")), job)
        _atomic_json(stamp_path, {"signature": job["signature"]})
        _atomic_json(status_path, row)
        return row

    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(status_path, {
        "city": job["city"], "scenario_id": job["scenario_id"],
        "seed": job["seed"], "policy": job["policy"], "status": "running",
        "orders": job["orders"], "result_file": str(output),
        "log_file": str(job["log"]),
    })
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    try:
        with job["log"].open("w", encoding="utf-8") as log:
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError(f"simulator exceeded {timeout_seconds} seconds")
        if returncode != 0:
            raise RuntimeError(f"simulator exited {returncode}")
        result = json.loads(output.read_text(encoding="utf-8"))
        row = summarize(result, job)
        _atomic_json(stamp_path, {"signature": job["signature"]})
        _atomic_json(status_path, row)
        return row
    except Exception as exc:
        row = {
            "city": job["city"], "scenario_id": job["scenario_id"],
            "seed": job["seed"], "policy": job["policy"], "status": "failed",
            "orders": job["orders"], "result_file": str(output),
            "log_file": str(job["log"]), "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(status_path, row)
        return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--spatial-model", type=Path, required=True)
    parser.add_argument("--paper-model", type=Path, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--idle-processes", type=int, default=4)
    parser.add_argument("--shortlist-k", type=int, default=10)
    parser.add_argument("--prior-concentration", type=float, default=4.0)
    parser.add_argument("--idle-relocation-uncertainty-penalty", type=float, default=1.0,
                        help="Posterior gain standard-deviation penalty for uncertainty idle variants.")
    parser.add_argument("--idle-switch-gain-fraction", type=float, default=0.005)
    parser.add_argument("--idle-retarget-cooldown-min", type=float, default=2.0)
    parser.add_argument(
        "--timeout-seconds", type=int, default=0,
        help="Per-simulation wall-clock limit; zero disables the timeout (default).",
    )
    parser.add_argument("--compatible-source-fingerprint", action="append", default=[])
    parser.add_argument("--scenario-ids", nargs="+", default=None)
    parser.add_argument("--adopt-existing-results", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(DEFAULT_POLICIES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job-order", choices=["city-major", "round-robin"], default="city-major")
    add_mle_arguments(parser)
    args = parser.parse_args()
    try:
        mle_config = mle_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.processes, args.idle_processes, args.shortlist_k) <= 0:
        parser.error("process counts and shortlist K must be positive")
    if set(args.policies) & set(MLE_POLICIES):
        budget = mle_worker_budget(args.processes, args.idle_processes, mle_config.processes)
        if budget > 60:
            parser.error(f"MLE comparison needs up to {budget} compute workers; cap is 60")
    if args.timeout_seconds < 0:
        parser.error("timeout must be non-negative")
    if any(not math.isfinite(v) or v < 0 for v in (args.idle_switch_gain_fraction, args.idle_retarget_cooldown_min)):
        parser.error("switch gain fraction and cooldown must be finite and non-negative")
    if (not math.isfinite(args.idle_relocation_uncertainty_penalty)
            or args.idle_relocation_uncertainty_penalty <= 0):
        parser.error("idle relocation uncertainty penalty must be finite and positive")
    timeout_seconds = args.timeout_seconds or None

    suite_path = args.suite.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    base = suite_path.parent
    results_dir = args.results_dir.resolve()
    spatial_model = args.spatial_model.resolve()
    paper_model = args.paper_model.resolve()
    required_models = set(args.policies) & {"full", "full_no_idle", "full_uncertainty_idle", "paper_sa_adapted", "full_queue_aware", "full_coordinated_idle"}
    if required_models & {"full", "full_no_idle", "full_uncertainty_idle", "full_queue_aware", "full_coordinated_idle"} and not spatial_model.is_file():
        raise FileNotFoundError(spatial_model)
    if "paper_sa_adapted" in required_models and not paper_model.is_file():
        raise FileNotFoundError(paper_model)
    model_hashes = {
        "spatial": _sha256(spatial_model) if spatial_model.is_file() else None,
        "paper": _sha256(paper_model) if paper_model.is_file() else None,
    }
    source_hash = _source_digest()
    jobs: list[dict[str, Any]] = []
    graph_hashes: dict[Path, str] = {}
    for city_index, (city, city_data) in enumerate(suite["cities"].items()):
        graph = _resolve(base, city_data["graph"])
        if graph not in graph_hashes:
            graph_hashes[graph] = _sha256(graph)
        for scenario_index, record in enumerate(city_data["test"]):
            if args.scenario_ids is not None and record["id"] not in args.scenario_ids:
                continue
            scenario = _resolve(base, record["scenario"])
            if not scenario.is_file():
                raise FileNotFoundError(scenario)
            scenario_hash = _sha256(scenario)
            for policy_index, policy in enumerate(args.policies):
                job_dir = results_dir / city / record["id"] / policy
                output = job_dir / "result.json"
                command = policy_command(
                    policy, python=sys.executable, graph=graph, scenario=scenario,
                    output=output, spatial_model=spatial_model, paper_model=paper_model,
                    shortlist_k=args.shortlist_k, idle_processes=args.idle_processes,
                    prior_rates=city_data["prior_rates_per_hour"],
                    prior_concentration=args.prior_concentration,
                    relocation_uncertainty_penalty=args.idle_relocation_uncertainty_penalty,
                    switching_gain_fraction=args.idle_switch_gain_fraction,
                    retarget_cooldown_min=args.idle_retarget_cooldown_min,
                    mle_config=mle_config,
                )
                def signature_for(source: str) -> str:
                    return hashlib.sha256(json.dumps({
                        "source": source, "graph": graph_hashes[graph],
                        "scenario": scenario_hash, "models": model_hashes,
                        "command": command,
                    }, sort_keys=True).encode()).hexdigest()

                signature = signature_for(source_hash)
                accepted_signatures = {
                    signature,
                    *(signature_for(value) for value in args.compatible_source_fingerprint),
                }
                jobs.append({
                    "city_index": city_index, "scenario_index": scenario_index, "policy_index": policy_index,
                    "city": city, "scenario_id": record["id"], "seed": record["seed"],
                    "orders": record["orders"], "policy": policy,
                    "output": output, "log": job_dir / "runner.log",
                    "status_path": job_dir / "status.json",
                    "stamp_path": job_dir / "stamp.json",
                    "command": command, "signature": signature,
                    "accepted_signatures": accepted_signatures,
                })
    jobs = order_jobs(jobs, args.job_order)
    for job in jobs:
        stamp_path = job["stamp_path"]
        try:
            job["already_complete"] = (
                job["output"].is_file()
                and json.loads(stamp_path.read_text(encoding="utf-8")).get("signature")
                in job["accepted_signatures"]
            )
        except (FileNotFoundError, json.JSONDecodeError):
            job["already_complete"] = False
    pending_count = sum(not job["already_complete"] for job in jobs)
    print(f"jobs={len(jobs)} cached={len(jobs)-pending_count} pending={pending_count} "
          f"cities={len(suite['cities'])} processes={args.processes}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["city"], job["scenario_id"], job["policy"], flush=True)
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(results_dir / "run_manifest.json", {
        "suite": str(suite_path), "policies": list(args.policies),
        "job_order": args.job_order,
        "spatial_model": str(spatial_model), "paper_model": str(paper_model),
        "model_sha256": model_hashes, "source_sha256": source_hash,
        "graph_sha256": {str(path): digest for path, digest in graph_hashes.items()},
        "processes": args.processes, "idle_processes": args.idle_processes,
        "shortlist_k": args.shortlist_k,
        "idle_relocation_uncertainty_penalty": args.idle_relocation_uncertainty_penalty,
        "idle_switch_gain_fraction": args.idle_switch_gain_fraction,
        "idle_retarget_cooldown_min": args.idle_retarget_cooldown_min,
        "mle_reservation_config": asdict(mle_config) if set(args.policies) & set(MLE_POLICIES) else None,
        "timeout_seconds": args.timeout_seconds,
        "compatible_source_fingerprints": args.compatible_source_fingerprint,
    })
    _write_summary(results_dir, jobs)
    failed = False
    started = time.monotonic()
    newly_completed = 0
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        futures = {
            executor.submit(
                _execute, job, timeout_seconds, args.adopt_existing_results
            ): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            _write_summary(results_dir, jobs)
            elapsed = time.monotonic() - started
            if not futures[future]["already_complete"]:
                newly_completed += 1
            eta_text = (
                f"{elapsed / newly_completed * (pending_count-newly_completed):.0f}s"
                if newly_completed else "unknown"
            )
            print(f"[{index}/{len(jobs)}] {row['city']} {row['scenario_id']} "
                  f"{row['policy']} {row['status']} loss={row.get('loss_objective', '')} "
                  f"elapsed={elapsed:.0f}s eta={eta_text}", flush=True)
            if row["status"] == "failed":
                failed = True
                for other in futures:
                    other.cancel()
                break
    if failed:
        raise RuntimeError(f"benchmark failed; inspect {results_dir / 'summary.csv'}")
    print(f"complete summary={results_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
