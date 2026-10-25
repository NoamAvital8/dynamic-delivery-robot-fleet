"""Explicitly launched parallel collection, periodic refits and frozen validation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import sys
import time

from . import LABEL_VERSION
from .collection import collect_episode, init_worker
from .config import DEVELOPMENT, HELD_OUT, POLICIES, atomic, available_gib, identity, locked, read, sha, workers
from .data import load_dataset, science_id
from .models import fit

FAMILIES = ("xgboost", "nn")


def estimate(config, cpus=72, memory_gib=171.):
    """Planning arithmetic only; no maps, simulator or models are loaded."""
    slots = workers(config, cpus, memory_gib)
    train = len(DEVELOPMENT) * config.train_per_city * len(FAMILIES)
    validation = len(DEVELOPMENT) * config.validation_per_city * len(FAMILIES)
    # Historical means for actual 15-minute branch windows, without scoring tails.
    historical = {"new_delhi":.0863,"haifa":.2774,"tel_aviv":.4054,"manhattan":2.0260}
    # New maps and heavier workloads have not been timed. Explicit planning band.
    decisions = 50  # 48 arrival-horizon decisions plus a modest delivery drain.
    branches = train * decisions * len(POLICIES)
    worker_hours = [branches * seconds / 3600 for seconds in (.7, 2.5)]
    effective_slots = max(1., min(36., slots * .65))
    elapsed = [value / effective_slots + overhead for value,overhead in zip(worker_hours,(2.,6.))]
    return {"label_version":LABEL_VERSION,"development_cities":sorted(DEVELOPMENT),
        "reserved_test_cities":sorted(HELD_OUT),"families":list(FAMILIES),
        "training_episodes_total":train,"validation_episodes_total":validation,
        "future_test_scenarios":len(HELD_OUT)*config.test_per_city,
        "decision_minutes":15,"scenario_hours":12,"benchmarks_per_training_state":len(POLICIES),
        "training_branches_approx":branches,"admitted_worker_slots":slots,
        "cpu_budget":cpus,"fit_threads":config.fit_threads,"memory_reserve_gib":config.memory_reserve_gib,
        "historical_window_seconds_by_city":historical,
        "assumed_seconds_per_branch_including_projection_range":[.7,2.5],
        "estimated_collection_worker_hours_range":worker_hours,
        "estimated_training_plus_validation_elapsed_hours_range":elapsed,
        "estimate_status":"unbenchmarked new projected scorer; not a measured completion promise",
        "old_label_rows_reused":0,
        "scope_note":"Both model-driven collection streams, periodic fits and validation; final held-out benchmark campaign excluded",
        "launches_training":False,"launches_simulations":False}


def _rows(root, family, science):
    result = []
    for p in sorted((root / "actors" / family / "train").glob("*/rows/*.json")):
        row = read(p)
        if row["science_id"] != science or row["family"] != family:
            raise ValueError("Mixed experiment identities in replay")
        result.append(row)
    return result


def _publish(config, family, rows, science):
    from threadpoolctl import threadpool_limits
    if config.dependency_dir and config.dependency_dir not in sys.path:
        sys.path.insert(0, config.dependency_dir)
    with threadpool_limits(limits=config.fit_threads):
        payload = fit(rows, family, config.fit_threads)
    payload["science_id"] = science
    payload["version_id"] = identity([science,family,payload["training_records_sha256"]])
    path = Path(config.output) / "models" / family / (payload["version_id"] + ".json")
    if path.exists():
        if read(path) != payload:
            raise ValueError("A model version changed; preserve existing artifacts")
    else:
        atomic(path, payload)
    published = {"path":str(path),"sha256":sha(path),"training_states":len(rows),"science_id":science}
    atomic(path.parent / "current.json", published)
    return published


def _pool(config, jobs, science, progress=None):
    """One process budget across episodes; benchmark branches stay sequential."""
    slots = workers(config)
    if slots < 1:
        raise RuntimeError("Insufficient memory/CPU allowance; no workers were launched")
    results, active = [], {}
    iterator = iter(jobs)
    next_job = next(iterator, None)
    with ProcessPoolExecutor(max_workers=slots, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker) as pool:
        while next_job is not None or active:
            # Reserve estimated initialization RAM for submitted jobs as well.
            launch_limit = min(slots, max(1, len(active)) + 8)
            while next_job is not None and len(active) < launch_limit:
                startup_allowance = config.worker_memory_gib * max(1, launch_limit - len(active))
                if available_gib() < config.memory_reserve_gib + startup_allowance:
                    break
                e, family, frozen = next_job
                future = pool.submit(collect_episode, config, e, family, science, frozen)
                active[future] = (e["id"], family)
                next_job = next(iterator, None)
            if not active:
                raise RuntimeError("Memory reserve prevents further admission; existing results remain resumable")
            finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                identity_ = active.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    for queued in active:
                        queued.cancel()
                    raise RuntimeError(f"Episode failed {identity_}; resume preserved checkpoints after repair") from exc
            if progress:
                progress(results, len(active))
    return results


def train(config):
    root = Path(config.output)
    with locked(root / ".training.lock"):
        data = load_dataset(config)
        science = data["science_id"]
        atomic(root / "runtime_plan.json", estimate(config, len(os.sched_getaffinity(0)), available_gib()))
        for family in FAMILIES:
            path = root / "models" / family / "bootstrap.json"
            registry = path.parent / "current.json"
            if not registry.exists():
                atomic(path, {"kind":"bootstrap","family":family,"policy":config.bootstrap_policy,
                    "policies":list(POLICIES),"label_version":LABEL_VERSION,"science_id":science})
                atomic(registry, {"path":str(path),"sha256":sha(path),"training_states":0,"science_id":science})
            elif read(registry)["science_id"] != science:
                raise ValueError("Published model belongs to incompatible scoring/backend code")
        training = [e for e in data["episodes"] if e["split"] == "train"]
        # Interleave model families/cities, so both learners acquire early data.
        training.sort(key=lambda e:(int(e["id"].rsplit('_',1)[-1]), e["city"]))
        jobs = [(e,family,None) for e in training for family in FAMILIES]
        fitted_counts = {f:read(root / "models" / f / "current.json")["training_states"] for f in FAMILIES}
        last_scan = 0.
        def progress(results, running):
            nonlocal last_scan
            atomic(root / "training_status.json", {"stage":"model_driven_collection",
                "complete_episodes":len(results),"running":running,"expected":len(jobs),
                "available_gib":available_gib(),"fitted_states":fitted_counts})
            if time.monotonic() - last_scan < 15:
                return
            last_scan = time.monotonic()
            for family in FAMILIES:
                rows = _rows(root, family, science)
                if len(rows) >= fitted_counts[family] + config.update_every_states:
                    _publish(config, family, rows, science)
                    fitted_counts[family] = len(rows)
        training_results = _pool(config, jobs, science, progress)
        final = {}
        for family in FAMILIES:
            rows = _rows(root, family, science)
            final[family] = _publish(config, family, rows, science)
        # Frozen, complete-episode validation: never fits or collects teacher labels.
        validation = [e for e in data["episodes"] if e["split"] == "validation"]
        results = _pool(config, [(e,f,final[f]) for e in validation for f in FAMILIES], science)
        summary = {f:{"episodes":sum(r["family"]==f for r in results),
                      "total_loss":sum(r["loss"] for r in results if r["family"]==f)} for f in FAMILIES}
        for family in FAMILIES:
            per_city = {}
            for city in sorted(DEVELOPMENT):
                matched = [r for r in results if r["family"]==family and r["city"]==city]
                per_city[city] = {"episodes":len(matched),
                    "mean_loss":sum(r["loss"] for r in matched)/len(matched),
                    "mean_loss_per_robot":sum(r["loss"]/r["robots"] for r in matched)/len(matched)}
            summary[family]["per_city"] = per_city
            summary[family]["equal_city_mean_loss_per_robot"] = sum(r["mean_loss_per_robot"] for r in per_city.values())/len(per_city)
        atomic(root / "validation_summary.json", {"models":final,"scores":summary,"results":results})
        atomic(root / "training_summary.json", {"science_id":science,"results":training_results,"models":final})
        atomic(root / "frozen_models.json", {"models":final,"science_id":science,
            "validation_only_comparison":summary,"held_out_cities":sorted(HELD_OUT)})
        atomic(root / "training_status.json", {"stage":"complete","models":final,
            "held_out_test_simulations_started":0})
        return summary


def evaluate(config):
    root = Path(config.output)
    with locked(root / ".test.lock"):
        frozen = read(root / "frozen_models.json")
        data = load_dataset(config, test=True)
        if frozen["science_id"] != data["science_id"]:
            raise ValueError("Test scorer/backend differs from the frozen development experiment")
        # No model updates in this function. No test outcomes enter model fitting.
        jobs = [(e,f,frozen["models"][f]) for e in data["episodes"] for f in FAMILIES]
        result = _pool(config, jobs, data["science_id"])
        atomic(root / "test_summary.json", {"models":frozen["models"],"results":result})
        return {"complete":len(result)}
