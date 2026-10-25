"""Explicit data-only preparation with fresh seeds and immutable split manifests."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path

from . import LABEL_VERSION
from .backend import activate
from .config import DEVELOPMENT, HELD_OUT, POLICIES, atomic, guard_city, identity, read, sha

PROFILES = ("ordinary", "surge", "spatial_shift", "priority_shift", "charging_pressure")


def science_id(config):
    backend = Path(config.backend_root)
    paths = sorted((backend / "src/delivery_fleet").rglob("*.py")) + sorted((backend / "scripts").glob("*.py"))
    if not paths:
        raise ValueError("The preserved simulator backend is missing")
    own = Path(__file__).parent
    return identity({"backend": {str(p.relative_to(backend)): sha(p) for p in paths},
        "adapter": sha(own / "backend.py"), "projection": sha(own / "projection.py"),
        "label": LABEL_VERSION, "decision_minutes": 15, "policies": POLICIES})


def prepare(config, test=False):
    """Only an explicit CLI command calls this; never executes a delivery policy."""
    import numpy as np
    activate(config)
    from create_multicity_suite import make_demand
    from create_nyc_reference_scenario import read_graphml_nodes
    from delivery_fleet.scenario_creator import ImportanceDistribution, NodeDemandProfile, ScenarioCreator
    from delivery_fleet.meta_policy.separation import workload_identity
    root = Path(config.output)
    if test and not (root / "frozen_models.json").is_file():
        raise ValueError("Freeze development models before preparing the held-out test set")
    if test:
        frozen = read(root / "frozen_models.json")
        if any(sha(v["path"]) != v["sha256"] for v in frozen["models"].values()):
            raise ValueError("A frozen development model changed")
    splits = ("test",) if test else ("train", "validation")
    manifest_path = root / "datasets" / ("test_manifest.json" if test else "development_manifest.json")
    if manifest_path.exists():
        raise ValueError("Dataset manifest already exists; use it rather than regenerate scenarios")
    episodes, seen_workloads, seen_seeds = [], set(), set()
    for row in config.maps("test" if test else "train"):
        graph = Path(row["path"])
        if sha(graph) != row["sha256"]:
            raise ValueError("Prepared map hash changed")
        nodes, lat, lon = read_graphml_nodes(graph)
        nominal_fleet = math.ceil(len(nodes) / 500)
        for split in splits:
            guard_city(row["key"], split)
            count = getattr(config, {"train":"train_per_city", "validation":"validation_per_city", "test":"test_per_city"}[split])
            for index in range(count):
                seed = int(identity(["small-switch-v2", config.base_seed, row["key"], split, index])[:15], 16)
                if seed in seen_seeds:
                    raise ValueError("Scenario-seed collision")
                seen_seeds.add(seed)
                rng = np.random.default_rng(seed)
                fleet = min(80, max(5, round(nominal_fleet * rng.uniform(.75, 1.25))))
                profile = PROFILES[index % len(PROFILES)]
                demand, _ = make_demand(nodes, lat, lon, duration_hours=config.duration_hours, fleet_count=fleet, rng=rng)
                rates = demand.lambda_by_bucket.copy() * rng.uniform(.8, 1.3)
                if profile == "surge":
                    rates[4:7] *= 2.
                elif profile == "spatial_shift":
                    x = np.asarray(lon, dtype=float)
                    zone = (x - x.min()) / max(1e-9, float(np.ptp(x)))
                    for hour in range(len(rates)):
                        total = rates[hour].sum()
                        rates[hour] *= .3 + 2 * (zone if hour < 6 else 1 - zone)
                        rates[hour] *= total / rates[hour].sum()
                elif profile == "charging_pressure":
                    rates *= 1.5
                scenario = ScenarioCreator(importance_distribution=ImportanceDistribution()).create(
                    row["key"], NodeDemandProfile(nodes, rates, bucket_minutes=60.), seed)
                if profile == "priority_shift":
                    scenario = replace(scenario, orders=tuple(replace(o, importance=float(rng.choice(
                        [1., 2., 5.], p=[.45,.2,.35] if 240 <= o.request_time_min < 480 else [.8,.15,.05]))) for o in scenario.orders))
                workload = workload_identity(scenario)
                if workload in seen_workloads or not scenario.orders:
                    raise ValueError("Empty or duplicated generated workload")
                seen_workloads.add(workload)
                key = f"{row['key']}_switch_{split}_{index:03d}"
                path = root / "datasets" / split / (key + ".json")
                payload = asdict(scenario)
                if path.exists():
                    # Recover interrupted data-only preparation without rewriting it.
                    if identity(read(path)) != identity(payload):
                        raise ValueError("Refusing to overwrite an incompatible preserved scenario")
                else:
                    atomic(path, payload)
                episodes.append({"id":key,"city":row["key"],"split":split,"seed":seed,
                    "scenario":str(path),"scenario_sha256":sha(path),"workload_sha256":workload,
                    "graph":str(graph),"graph_sha256":row["sha256"],"orders":len(scenario.orders),
                    "fleet_count":fleet,"profile":profile,
                    "initial_battery_fraction":float(rng.uniform(.4,.75)) if profile == "charging_pressure" else 1.,
                    # Configured public historical expectation, never test realizations.
                    "prior_rates_per_hour":{str(k):fleet*.30*p for k,p in [(1,.8),(2,.15),(5,.05)]}})
    result = {"label_version":LABEL_VERSION,"science_id":science_id(config),
              "config":asdict(config),"episodes":episodes,"role":"test" if test else "development"}
    atomic(manifest_path, result)
    return result


def load_dataset(config, test=False):
    path = Path(config.output) / "datasets" / ("test_manifest.json" if test else "development_manifest.json")
    data = read(path)
    if data["label_version"] != LABEL_VERSION or data["science_id"] != science_id(config):
        raise ValueError("Simulator/projection identity changed; do not reuse incompatible labels")
    for key in ("base_seed","duration_hours","decision_minutes","train_per_city","validation_per_city","test_per_city"):
        if data["config"][key] != getattr(config,key):
            raise ValueError("Dataset preparation settings changed")
    maps = {r["key"]:r for r in config.maps("test" if test else "train")}
    seen_ids, seeds, workloads = set(), {}, {}
    for e in data["episodes"]:
        guard_city(e["city"], e["split"])
        if (test and e["split"] != "test") or (not test and e["split"] == "test"):
            raise ValueError("Dataset role mismatch")
        if e["id"] in seen_ids:
            raise ValueError("Duplicate episode identity")
        seen_ids.add(e["id"])
        if e["city"] not in maps or str(Path(e["graph"])) != str(Path(maps[e["city"]]["path"])) or e["graph_sha256"] != maps[e["city"]]["sha256"]:
            raise ValueError("Scenario graph/city role mismatch")
        if sha(e["scenario"]) != e["scenario_sha256"]:
            raise ValueError("Scenario bytes changed")
        for key, seen in (("seed",seeds),("workload_sha256",workloads)):
            previous = seen.setdefault(e[key], e["split"])
            if previous != e["split"]:
                raise ValueError("Training/validation overlap")
    return data
