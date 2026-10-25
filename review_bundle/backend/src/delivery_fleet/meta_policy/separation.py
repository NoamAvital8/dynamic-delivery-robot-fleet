"""Scenario separation by RNG seed and actual workload, not filename alone."""
from __future__ import annotations

import json
from pathlib import Path

from delivery_fleet.scenario_creator import Scenario
from .artifacts import digest, identity


def workload_identity(scenario):
    """Changing descriptive names, order ids or seed does not make new orders."""
    return identity({"duration_minutes": float(scenario.duration_minutes), "orders": [
        [float(o.request_time_min), int(o.pickup_node), int(o.dropoff_node),
         float(o.importance), float(o.item.weight_kg), float(o.item.volume_l)]
        for o in scenario.orders]})


def discover_suites(reference, additional=()):
    """Find archived city suites in the reference repository, plus explicit ones."""
    reference = Path(reference).resolve()
    found = {reference, *(Path(p).resolve() for p in additional)}
    root = next((p for p in reference.parents if (p / "src").is_dir()
                 and (p / "scripts").is_dir()), None)
    if root is not None:
        for directory in (root / "data/scenarios", root / "runs"):
            if directory.exists():
                for path in directory.rglob("suite.json"):
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value.get("cities"), dict):
                        found.add(path.resolve())
    return sorted(found)


def reserve_suites(paths):
    """Pin every archived train/validation/test scenario from these suites."""
    records, seen = [], set()
    for suite_path in paths:
        suite_path = Path(suite_path).resolve()
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if not isinstance(suite.get("cities"), dict):
            raise ValueError(f"not a city scenario suite: {suite_path}")
        for city, data in suite["cities"].items():
            for split in ("train", "validation", "test"):
                for record in data.get(split, []):
                    scenario_path = (suite_path.parent / record["scenario"]).resolve()
                    if scenario_path in seen:
                        continue
                    seen.add(scenario_path)
                    scenario = Scenario.load_json(scenario_path)
                    if "seed" in record and int(record["seed"]) != scenario.seed:
                        raise ValueError(f"suite/scenario seed mismatch: {scenario_path}")
                    records.append({"city": city, "id": record["id"], "split": split,
                        "path": str(scenario_path), "seed": int(scenario.seed),
                        "sha256": digest(scenario_path), "workload_sha256": workload_identity(scenario)})
    if not records:
        raise ValueError("at least one archived scenario is required for separation checks")
    return {"version": 1, "policy": "fresh_train_and_validation",
            "reference_suites": [{"path": str(p), "sha256": digest(p)} for p in paths],
            "reserved": records}


def check_separation(config, episodes, *, mode):
    """Fail before simulator initialization if reserved data could enter training."""
    separation = config.get("scenario_separation")
    if (not separation or separation.get("version") != 1
            or separation.get("policy") != "fresh_train_and_validation"
            or not separation.get("reserved")):
        raise ValueError("manifest needs a scenario_separation registry; generate fresh meta scenarios first")
    for suite in separation.get("reference_suites", []):
        if digest(suite["path"]) != suite["sha256"]:
            raise ValueError("archived reference suite changed")
    for record in separation["reserved"]:
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("reserved benchmark scenario changed")
    seeds = {int(r["seed"]) for r in separation["reserved"]}
    hashes = {r["sha256"] for r in separation["reserved"]}
    workloads = {r["workload_sha256"] for r in separation["reserved"]}
    provenance, used_seeds, used_workloads = {}, {}, {}
    for episode in episodes:
        scenario = Scenario.load_json(episode["scenario"])
        sha, workload = digest(episode["scenario"]), workload_identity(scenario)
        if mode == "collect":
            if scenario.seed in seeds or sha in hashes or workload in workloads:
                raise ValueError("training/validation overlaps reserved scenarios by seed or workload")
            if scenario.seed in used_seeds or workload in used_workloads:
                raise ValueError("fresh episodes duplicate a seed or workload")
            used_seeds[scenario.seed] = episode["id"]
            used_workloads[workload] = episode["id"]
        provenance[episode["id"]] = {"seed": scenario.seed, "workload_sha256": workload,
            "registry_identity": identity(separation), "role": episode["split"],
            "orders": len(scenario.orders)}
    return provenance
