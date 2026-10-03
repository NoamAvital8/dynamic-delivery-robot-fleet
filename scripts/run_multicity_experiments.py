from __future__ import annotations

"""Run six paired policies across held-out city scenarios with durable results."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "full",
    "myopic_ab",
    "reactive_insertion",
    "paper_sa_adapted",
    "full_no_nn",
    "full_no_idle",
)
FIELDS = (
    "city", "scenario_id", "seed", "policy", "status", "orders",
    "delivered", "on_time", "late", "loss_objective",
    "wall_clock_seconds", "simulation_finish_min", "result_file", "log_file", "error",
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
    temporary.replace(path)


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
    }[policy]
    command = [python, str(ROOT / "scripts" / script), "--graph", str(graph),
               "--scenario", str(scenario), "--output", str(output)]
    if policy not in {"myopic_ab", "reactive_insertion"}:
        command += [
            "--shortlist-k", str(shortlist_k),
            "--importance-prior-rates-per-hour", json.dumps(prior_rates, sort_keys=True),
            "--prior-concentration", str(prior_concentration),
        ]
    if policy in {"full", "full_no_nn"}:
        command += ["--idle-processes", str(idle_processes)]
    if policy in {"full", "full_no_idle"}:
        command += ["--reservation-model", str(spatial_model)]
    if policy == "paper_sa_adapted":
        command += ["--reservation-model", str(paper_model),
                    "--reservation-style", "paper_moving_average",
                    "--reservation-lookback-min", "100"]
    return command


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
        })
        bucket["planned_scenarios"] += 1
        if row["status"] == "complete":
            bucket["completed_scenarios"] += 1
            bucket["total_loss"] += float(row["loss_objective"])
            bucket["total_wall_clock_seconds"] += float(row["wall_clock_seconds"])
            for field in ("on_time", "late", "delivered", "orders"):
                bucket[field] += int(row[field])
    aggregate_fields = (
        "city", "policy", "completed_scenarios", "planned_scenarios",
        "total_loss", "mean_loss", "total_wall_clock_seconds",
        "mean_wall_clock_seconds", "on_time", "late", "delivered", "orders",
    )
    aggregate_rows = []
    for bucket in grouped.values():
        completed = bucket["completed_scenarios"]
        bucket["mean_loss"] = bucket["total_loss"] / completed if completed else ""
        bucket["mean_wall_clock_seconds"] = (
            bucket["total_wall_clock_seconds"] / completed if completed else ""
        )
        aggregate_rows.append(bucket)
    aggregate_path = results_dir / "city_policy_summary.csv"
    aggregate_tmp = aggregate_path.with_suffix(".csv.tmp")
    with aggregate_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    aggregate_tmp.replace(aggregate_path)


def _execute(job: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    output: Path = job["output"]
    status_path: Path = job["status_path"]
    stamp_path: Path = job["stamp_path"]
    if output.is_file() and stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("signature") == job["signature"]:
            row = summarize(json.loads(output.read_text(encoding="utf-8")), job)
            _atomic_json(status_path, row)
            return row
        raise RuntimeError(f"existing result has a different input signature: {output}")

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
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.processes, args.idle_processes, args.shortlist_k, args.timeout_seconds) <= 0:
        parser.error("process counts, shortlist K and timeout must be positive")

    suite_path = args.suite.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    base = suite_path.parent
    results_dir = args.results_dir.resolve()
    spatial_model = args.spatial_model.resolve()
    paper_model = args.paper_model.resolve()
    required_models = set(args.policies) & {"full", "full_no_idle", "paper_sa_adapted"}
    if required_models & {"full", "full_no_idle"} and not spatial_model.is_file():
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
    for city, city_data in suite["cities"].items():
        graph = _resolve(base, city_data["graph"])
        if graph not in graph_hashes:
            graph_hashes[graph] = _sha256(graph)
        for record in city_data["test"]:
            scenario = _resolve(base, record["scenario"])
            if not scenario.is_file():
                raise FileNotFoundError(scenario)
            scenario_hash = _sha256(scenario)
            for policy in args.policies:
                job_dir = results_dir / city / record["id"] / policy
                output = job_dir / "result.json"
                command = policy_command(
                    policy, python=sys.executable, graph=graph, scenario=scenario,
                    output=output, spatial_model=spatial_model, paper_model=paper_model,
                    shortlist_k=args.shortlist_k, idle_processes=args.idle_processes,
                    prior_rates=city_data["prior_rates_per_hour"],
                    prior_concentration=args.prior_concentration,
                )
                signature = hashlib.sha256(json.dumps({
                    "source": source_hash, "graph": graph_hashes[graph],
                    "scenario": scenario_hash, "models": model_hashes,
                    "command": command,
                }, sort_keys=True).encode()).hexdigest()
                jobs.append({
                    "city": city, "scenario_id": record["id"], "seed": record["seed"],
                    "orders": record["orders"], "policy": policy,
                    "output": output, "log": job_dir / "runner.log",
                    "status_path": job_dir / "status.json",
                    "stamp_path": job_dir / "stamp.json",
                    "command": command, "signature": signature,
                })
    for job in jobs:
        stamp_path = job["stamp_path"]
        try:
            job["already_complete"] = (
                job["output"].is_file()
                and json.loads(stamp_path.read_text(encoding="utf-8")).get("signature")
                == job["signature"]
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
        "spatial_model": str(spatial_model), "paper_model": str(paper_model),
        "model_sha256": model_hashes, "source_sha256": source_hash,
        "graph_sha256": {str(path): digest for path, digest in graph_hashes.items()},
        "processes": args.processes, "idle_processes": args.idle_processes,
        "shortlist_k": args.shortlist_k,
    })
    _write_summary(results_dir, jobs)
    failed = False
    started = time.monotonic()
    newly_completed = 0
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        futures = {executor.submit(_execute, job, args.timeout_seconds): job for job in jobs}
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
