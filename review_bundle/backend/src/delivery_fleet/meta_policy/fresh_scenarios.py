"""Fresh selector datasets; old city suites supply maps and reserved tests only."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys

from .artifacts import RunLock, atomic_json, digest, identity, runtime_versions, worker_count
from .separation import check_separation, discover_suites, reserve_suites, workload_identity
from . import DEFAULT_EXCLUDED_CITIES, DEFAULT_PORTFOLIO, POLICIES

ROOT = Path(__file__).resolve().parents[3]


def _city_scenarios(job):
    # One graph-node read per city. This never constructs routes or runs policies.
    from .pipeline import configure_threads
    configure_threads()
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from create_multicity_suite import make_demand
    from create_nyc_reference_scenario import read_graphml_nodes
    from delivery_fleet.scenario_creator import ImportanceDistribution, ScenarioCreator, Scenario
    nodes, lat, lon = read_graphml_nodes(Path(job["graph"]))
    fleet_count = max(1, math.ceil(len(nodes) / 500))
    used_seeds, used_workloads = set(job["blocked_seeds"]), set(job["blocked_workloads"])
    result = []
    for split, count in (("train", job["train_count"]), ("validation", job["validation_count"])):
        for index in range(count):
            for attempt in range(128):
                seed = int(identity(["fresh-meta-v1", job["base_seed"], job["city"],
                                     split, index, attempt])[:16], 16)
                if seed in used_seeds:
                    continue
                rng = np.random.default_rng(seed)
                demand, rate = make_demand(nodes, lat, lon, duration_hours=job["duration_hours"],
                                           fleet_count=fleet_count, rng=rng)
                probabilities = rng.dirichlet([16., 3., 1.])
                scenario = ScenarioCreator(importance_distribution=ImportanceDistribution(
                    (1., 2., 5.), tuple(map(float, probabilities)))).create(job["city"], demand, seed)
                workload = workload_identity(scenario)
                if scenario.orders and workload not in used_workloads:
                    break
            else:
                raise RuntimeError("could not generate a distinct nonempty workload in 128 attempts")
            used_seeds.add(seed)
            used_workloads.add(workload)
            name = f"{job['city']}_meta_{split}_{index:03d}"
            path = Path(job["directory"]) / split / f"{name}.json"
            payload = asdict(scenario)
            if path.exists():
                if not job["resume"] or Scenario.load_json(path) != scenario:
                    raise ValueError(f"refusing incompatible existing generated scenario: {path}")
            else:
                atomic_json(path, payload)
            counts = Counter(o.importance for o in scenario.orders)
            result.append({"id": name, "city": job["city"], "split": split,
                "graph": job["graph"], "scenario": str(path), "seed": seed,
                "sha256": digest(path), "workload_sha256": workload,
                "orders": len(scenario.orders), "orders_per_robot_hour": rate,
                "importance_counts": {str(k): counts[float(k)] for k in (1, 2, 5)}})
            print(f"fresh scenario {name}: orders={len(scenario.orders)}", flush=True)
    return result


def prepare(reference_suite, spatial_model, paper_model, output, *, train_per_city=20,
            validation_per_city=5, base_seed=20261015, holding_minutes=(15.,), cities=None,
            reserved_suites=(), scenario_dir=None, resume=False, processes=60,
            memory_per_worker_gib=4., reserve_gib=16., policies=DEFAULT_PORTFOLIO,
            continuation_policy="full_no_idle", initial_policy="full_no_idle",
            exclude_cities=DEFAULT_EXCLUDED_CITIES,
            exploration_stickiness=.75, fixed_behavior_fraction=.2, label_step_probability=.25,
            decide_until="all_delivered", scoring_horizon_minutes=60., controller_settings=None):
    """Generate NEW train/validation scenarios for one or more reference suites.

    Several suites (e.g. the 5 original and the 6 new cities) can be combined so
    one selector is trained across all cities and leave-one-city-out is possible.
    """
    from .pipeline import DECIDE_UNTIL
    from .runtime import controller_settings as normalize_settings
    suites = [reference_suite] if isinstance(reference_suite, (str, Path)) else list(reference_suite)
    suites = [Path(p).resolve() for p in suites]
    output = Path(output).resolve()
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("give one or more distinct reference suites")
    references = {path: json.loads(path.read_text(encoding="utf-8")) for path in suites}
    if train_per_city < 1 or validation_per_city < 1:
        raise ValueError("create positive, separate training and validation counts")
    windows = list(map(float, holding_minutes))
    if (15. not in windows or len(windows) != len(set(windows))
            or any(not math.isfinite(v) or v <= 0 or v > 60 for v in windows)):
        raise ValueError("holding windows must include 15 and lie in (0,60]")
    durations = {references[path]["duration_hours"] for path in suites}
    if len(durations) != 1:
        raise ValueError("reference suites must share one scenario duration")
    duration = durations.pop()
    if isinstance(duration, bool) or int(duration) != duration or duration <= 0:
        raise ValueError("reference duration must be a positive integer number of hours")
    city_suite = {}
    for path in suites:
        for city in references[path]["cities"]:
            if city in city_suite:
                raise ValueError(f"city {city} appears in more than one reference suite")
            city_suite[city] = path
    # An explicit city list wins; otherwise every suite city except the excluded ones.
    chosen = list(cities) if cities else [c for c in city_suite if c not in set(exclude_cities)]
    if (not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(city_suite)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", city) for city in chosen)):
        raise ValueError("select distinct, safe city names present in the reference suites")
    policies = tuple(policies)
    if (not policies or len(set(policies)) != len(policies) or set(policies) - set(POLICIES)
            or continuation_policy not in POLICIES or initial_policy not in policies):
        raise ValueError("invalid candidate portfolio, continuation or initial policy")
    if (not 0 <= exploration_stickiness < 1 or not 0 <= fixed_behavior_fraction <= 1
            or not 0 < label_step_probability <= 1 or decide_until not in DECIDE_UNTIL
            or not math.isfinite(scoring_horizon_minutes) or scoring_horizon_minutes < max(windows)):
        raise ValueError("invalid exploration, labelling, decision or scoring settings")
    settings = normalize_settings(controller_settings)
    directory = Path(scenario_dir).resolve() if scenario_dir else output.parent / (output.stem + "_scenarios")
    comparison_path = output.with_name(output.stem + "_comparison.json")
    discovered = set()
    for path in suites:
        discovered.update(discover_suites(path, reserved_suites))
    separation = reserve_suites(sorted(discovered))
    models = {key: str(Path(path).resolve()) for key, path in
              (("spatial_model", spatial_model), ("paper_model", paper_model))}
    graphs = {}
    for city in chosen:
        base = city_suite[city]
        graph = (base.parent / references[base]["cities"][city]["graph"]).resolve()
        graphs[city] = {"path": str(graph), "sha256": digest(graph)}
    generation = {"version": 1, "generator": "fresh-meta-v1", "base_seed": int(base_seed),
        "cities": chosen, "graphs": graphs, "duration_hours": int(duration),
        "train_per_city": train_per_city, "validation_per_city": validation_per_city,
        "holding_minutes": windows, "scenario_separation": separation,
        "model_files": {k: {"path": v, "sha256": digest(v)} for k, v in models.items()},
        "implementation": {str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), ROOT / "scripts/create_multicity_suite.py",
             ROOT / "src/delivery_fleet/scenario_creator.py",
             ROOT / "src/delivery_fleet/meta_policy/separation.py")},
        "runtime_versions": runtime_versions(), "directory": str(directory)}
    signature = identity(generation)
    generation_path = directory / "generation_manifest.json"
    if (output.exists() or comparison_path.exists()) and not resume:
        raise FileExistsError("output exists; use --resume or choose a new dataset directory")
    if generation_path.exists():
        old = json.loads(generation_path.read_text())
        if not resume or old.get("signature") != signature:
            raise ValueError("incompatible generation manifest; use a new scenario directory")
    jobs = [{"city": city, "graph": graphs[city]["path"], "directory": str(directory),
        "duration_hours": int(duration), "base_seed": int(base_seed),
        "train_count": train_per_city, "validation_count": validation_per_city,
        "blocked_seeds": [r["seed"] for r in separation["reserved"]],
        "blocked_workloads": [r["workload_sha256"] for r in separation["reserved"]],
        "resume": resume} for city in chosen]
    workers = min(len(jobs), worker_count(processes, memory_per_worker_gib=memory_per_worker_gib,
                                          reserve_gib=reserve_gib))
    with RunLock(directory / ".generation.lock"):
        if not generation_path.exists():
            atomic_json(generation_path, {**generation, "signature": signature})
        if workers:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                records = [row for rows in pool.map(_city_scenarios, jobs) for row in rows]
        else:
            records = [row for job in jobs for row in _city_scenarios(job)]
        # Recommended defaults are written explicitly so every run is self-describing.
        config = {"version": 1, **models, "policies": list(policies),
            "holding_minutes": windows, "default_holding_minutes": 15,
            "scoring_horizon_minutes": float(scoring_horizon_minutes),
            "continuation_policy": continuation_policy, "initial_policy": initial_policy,
            "exploration": {"stickiness": float(exploration_stickiness),
                            "fixed_episode_fraction": float(fixed_behavior_fraction),
                            "policies": list(policies)},
            "label_step_probability": float(label_step_probability), "decide_until": decide_until,
            "shortlist_k": 10, "controller_settings": settings, "scenario_separation": separation,
            "generation_signature": signature, "episodes": []}
        for city in chosen:
            training = [r for r in records if r["city"] == city and r["split"] == "train"]
            # Positive Gamma-Poisson smoothing for rare importance classes.
            # Only NEW training arrivals contribute, never validation or tests.
            rates = {str(level): (sum(r["importance_counts"][str(level)] for r in training) + .5)
                     / (len(training) * duration) for level in (1, 2, 5)}
            for record in (r for r in records if r["city"] == city):
                config["episodes"].append({**record, "prior_rates_per_hour": rates,
                    "prior_source_scenario_ids": [r["id"] for r in training],
                    "prior_pseudocount": .5})
        check_separation(config, config["episodes"], mode="collect")
        comparison = {**config, "episodes": [],
            "comparison_reference": [{"suite": str(path), "sha256": digest(path),
                "note": "Identical old scenarios/maps/priors; use only completed matched baselines."}
                for path in suites]}
        for city in chosen:
            base = city_suite[city]
            data = references[base]["cities"][city]
            for record in data.get("test", []):
                comparison["episodes"].append({"id": record["id"], "city": city, "split": "test",
                    "graph": str((base.parent / record["graph"]).resolve()),
                    "scenario": str((base.parent / record["scenario"]).resolve()),
                    "prior_rates_per_hour": data["prior_rates_per_hour"]})
        for path, value in ((output, config), (comparison_path, comparison)):
            if path.exists():
                if not resume or json.loads(path.read_text()) != value:
                    raise ValueError(f"refusing to replace different manifest: {path}")
            else:
                atomic_json(path, value)
        atomic_json(directory / "complete.json", {"signature": signature,
            "training_manifest": str(output), "comparison_manifest": str(comparison_path),
            "records": records})
    return config
