"""Explicit collection, bounded asynchronous fitting and complete matched evaluation."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
import time

from . import LABEL_VERSION
from .config import DEVELOPMENT, HELD_OUT, POLICIES, atomic, available_gib, identity, immutable, locked, read, sha, workers
from .data import load_dataset
from .execution import FitTask, numeric_environment, run_episodes
from .models import fit
from .provenance import ensure_campaign, freeze_receipt, learner_id, require_development_open
from . import replay
from .reporting import FIXED_PREFIX, compare

FAMILIES = ("xgboost", "nn")


def estimate(config, cpus=72, memory_gib=171.):
    slots = workers(config, cpus, memory_gib)
    train = len(DEVELOPMENT) * config.train_per_city * len(FAMILIES)
    validation = len(DEVELOPMENT) * config.validation_per_city * len(FAMILIES)
    fixed_validation = len(DEVELOPMENT) * config.validation_per_city * len(POLICIES)
    decisions = 50
    branches = train * decisions * len(POLICIES)
    validation_branches = (validation + fixed_validation) * decisions
    worker_hours = [(branches + validation_branches) * seconds / 3600 for seconds in (.7, 2.5)]
    effective = max(1., min(36., slots * .65))
    elapsed = [value / effective + extra for value, extra in zip(worker_hours, (2., 6.))]
    return {"label_version":LABEL_VERSION,"development_cities":sorted(DEVELOPMENT),
        "reserved_test_cities":sorted(HELD_OUT),"families":list(FAMILIES),
        "training_episodes_total":train,"validation_episodes_total":validation,
        "validation_fixed_benchmark_episodes_total":fixed_validation,
        "future_test_scenarios":len(HELD_OUT)*config.test_per_city,
        "future_test_selector_and_fixed_episodes":len(HELD_OUT)*config.test_per_city*(len(FAMILIES)+len(POLICIES)),
        "decision_minutes":15,"scenario_hours":12,"benchmarks_per_training_state":len(POLICIES),
        "training_branches_approx":branches,"validation_branches_approx":validation_branches,
        "admitted_worker_slots":slots,"cpu_budget":cpus,"fit_threads":config.fit_threads,
        "memory_reserve_gib":config.memory_reserve_gib,"trainer_memory_gib":config.trainer_memory_gib,
        "episode_process_lifetime":"one episode", "old_label_rows_reused":0,
        "historical_window_seconds_by_city":{"new_delhi":.0863,"haifa":.2774,"tel_aviv":.4054,"manhattan":2.026},
        "assumed_seconds_per_branch_including_projection_range":[.7,2.5],
        "estimated_collection_and_validation_worker_hours_range":worker_hours,
        "legacy_unmeasured_training_plus_validation_elapsed_hours_range":elapsed,
        "legacy_first_campaign_budget_hours_range":[24,48],
        "estimate_status":"Legacy unmeasured arithmetic; consult the measured development-pilot report for the default full-campaign estimate",
        "measured_pilot_report":"docs/SWITCHING_PILOT_RESULTS_2026-10-25.md",
        "measured_pilot_estimate":"review_bundle/evidence/switching_pilot_20261025/TRAINING_TIME_ESTIMATE.json",
        "scope_note":"Both learners plus selectors and all ten fixed policies on development validation; final test excluded",
        "launches_training":False,"launches_simulations":False}


def _publish(config, family, rows, science):
    numeric_environment(config.fit_threads)
    if config.dependency_dir and config.dependency_dir not in sys.path:
        sys.path.insert(0, config.dependency_dir)
    # Load native libraries before constructing the dynamic limiter.
    import numpy
    if family == "xgboost":
        import xgboost
    from threadpoolctl import threadpool_info, threadpool_limits
    with threadpool_limits(limits=config.fit_threads):
        payload = fit(rows, family, config.fit_threads)
        effective_threads = threadpool_info()
    payload["science_id"] = science
    payload["learner_id"] = learner_id(config, family)
    payload["execution_fit_threads"] = config.fit_threads
    payload["version_id"] = identity(payload)
    path = Path(config.output) / "models" / family / (payload["version_id"] + ".json")
    immutable(path, payload)
    pointer = {"path":str(path),"sha256":sha(path),"training_states":len(rows),"science_id":science,
               "learner_id":payload["learner_id"]}
    atomic(path.parent / "current.json", pointer)
    atomic(path.parent / "last_fit_runtime.json", {"native_libraries_inside_limit":effective_threads})
    return pointer


def _pool(config, jobs, science, progress=None):
    return run_episodes(config, jobs, science, progress)


def _fixed_models(config, science):
    result = {}
    for policy in POLICIES:
        path = Path(config.output) / "models/fixed" / (policy + ".json")
        immutable(path, {"kind":"bootstrap","policy":policy,"policies":list(POLICIES),
            "label_version":LABEL_VERSION,"science_id":science})
        result[policy] = {"path":str(path),"sha256":sha(path),"science_id":science}
    return result


def _wait_fit(task, config):
    try:
        while True:
            if available_gib() < config.memory_reserve_gib:
                raise MemoryError("Fit crossed the available-memory reserve")
            result = task.poll()
            if result is not None:
                return result
            time.sleep(.1)
    except BaseException:
        task.stop()
        raise


def train(config):
    root = Path(config.output)
    with locked(root / ".campaign.lock"):
        require_development_open(config)
        campaign = ensure_campaign(config)
        data = load_dataset(config)
        science = data["science_id"]
        episodes = {e["id"]:e for e in data["episodes"] if e["split"] == "train"}
        atomic(root / "execution_settings.json", asdict(config))
        atomic(root / "runtime_plan.json", estimate(config, len(os.sched_getaffinity(0)), available_gib()))
        fitted = {}
        for family in FAMILIES:
            path = root / "models" / family / "bootstrap.json"
            registry = path.parent / "current.json"
            if not registry.exists():
                immutable(path, {"kind":"bootstrap","family":family,"policy":config.bootstrap_policy,
                    "policies":list(POLICIES),"label_version":LABEL_VERSION,"science_id":science})
                atomic(registry, {"path":str(path),"sha256":sha(path),"training_states":0,"science_id":science,
                                  "learner_id":learner_id(config, family)})
            pointer = read(registry)
            if pointer["science_id"] != science or pointer.get("learner_id") != learner_id(config, family):
                raise ValueError("Published actor belongs to incompatible experiment/learner code")
            fitted[family] = pointer["training_states"]
        training = sorted(episodes.values(), key=lambda e:(int(e["id"].rsplit('_',1)[-1]), e["city"]))
        jobs = [(e, family, None) for e in training for family in FAMILIES]
        task, last_scan = None, 0.

        def progress(results, running):
            nonlocal task, last_scan
            if task is not None:
                result = task.poll()
                if result is not None:
                    fitted[task.family] = result["training_states"]
                    task = None
            if time.monotonic() - last_scan < 5:
                return
            last_scan = time.monotonic()
            stats = replay.statistics(root, science)
            atomic(root / "training_status.json", {"stage":"model_driven_collection",
                "complete_episodes":len(results),"running":running,"expected":len(jobs),
                "available_gib":available_gib(),"fitted_states":fitted,
                "fit_family":task.family if task else None,"replay":stats})
            if task is None:
                eligible = [f for f in FAMILIES if stats.get(f, {}).get("count", 0) >= fitted[f] + config.update_every_states]
                if eligible:
                    family = max(eligible, key=lambda f:stats[f]["count"] - fitted[f])
                    task = FitTask(config, family, science, episodes, stats[family]["watermark"])
        try:
            training_results = _pool(config, jobs, science, progress)
            if task is not None:
                _wait_fit(task, config)
                task = None
            stats = replay.statistics(root, science)
            final = {}
            for family in FAMILIES:
                task = FitTask(config, family, science, episodes, stats[family]["watermark"])
                final[family] = _wait_fit(task, config)
                task = None
        finally:
            if task is not None:
                task.stop()
        baselines = _fixed_models(config, science)
        validation = [e for e in data["episodes"] if e["split"] == "validation"]
        jobs = [(e,f,final[f]) for e in validation for f in FAMILIES]
        jobs += [(e,FIXED_PREFIX+p,baselines[p]) for e in validation for p in POLICIES]
        results = _pool(config, jobs, science)
        report = compare(results, validation, expected_science=science)
        atomic(root / "validation_summary.json", {"models":final, **report})
        atomic(root / "training_summary.json", {"science_id":science,"results":training_results,"models":final})
        frozen = {"models":final,"baselines":baselines,"science_id":science,"campaign_id":campaign["campaign_id"],
            "development_manifest_sha256":sha(root / "datasets/development_manifest.json"),
            "validation_report_sha256":sha(root / "validation_summary.json"),
            "development_selected_fixed_policy":report["development_selected_fixed_policy"],
            "selection_metric":report["selection_metric"],"held_out_cities":sorted(HELD_OUT)}
        frozen["freeze_id"] = identity(frozen)
        immutable(root / "frozen_models.json", frozen)
        atomic(root / "training_status.json", {"stage":"complete","models":final,"held_out_test_simulations_started":0})
        return report["scores"]


def evaluate(config):
    root = Path(config.output)
    with locked(root / ".campaign.lock"):
        frozen = freeze_receipt(config)
        data = load_dataset(config, test=True)
        opened = read(root / "test_opened.json")
        if opened["freeze_id"] != frozen["freeze_id"]:
            raise ValueError("Final-test opening receipt changed")
        jobs = [(e,f,frozen["models"][f]) for e in data["episodes"] for f in FAMILIES]
        jobs += [(e,FIXED_PREFIX+p,frozen["baselines"][p]) for e in data["episodes"] for p in POLICIES]
        results = _pool(config, jobs, data["science_id"])
        report = compare(results, data["episodes"], frozen["development_selected_fixed_policy"], expected_science=data["science_id"])
        atomic(root / "test_summary.json", {"freeze_id":frozen["freeze_id"], **report})
        return {"complete":len(results),"missing":0,"summary":str(root / "test_summary.json")}
