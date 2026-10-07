from __future__ import annotations

"""Create reproducible, disjoint multi-city training and test scenarios.

Pickup intensity is defined over graph nodes, independently of the policy's
HDBSCAN regions. City priors are estimated from training realizations only.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from create_nyc_reference_scenario import read_graphml_nodes
from delivery_fleet.scenario_creator import (
    ImportanceDistribution,
    NodeDemandProfile,
    ScenarioCreator,
    Scenario,
)


DEFAULT_CITIES = ("tel_aviv", "haifa", "manhattan", "new_york_city", "barcelona")
NEW_CITIES = ("beijing", "sydney", "moscow", "johannesburg", "new_delhi", "paris")
CITIES = (*DEFAULT_CITIES, *NEW_CITIES)
LEVELS = (1.0, 2.0, 5.0)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def make_demand(
    node_ids: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    duration_hours: int,
    fleet_count: int,
    rng: np.random.Generator,
) -> tuple[NodeDemandProfile, float]:
    if len(node_ids) < 2 or duration_hours <= 0 or fleet_count <= 0:
        raise ValueError("invalid graph or experiment duration")
    lat0 = math.radians(float(np.mean(latitudes)))
    x = 111.32 * math.cos(lat0) * longitudes
    y = 111.32 * latitudes
    span_km = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    centers = rng.choice(len(node_ids), size=min(4, len(node_ids)), replace=False)
    fields = []
    for center in centers:
        sigma_km = span_km * float(rng.uniform(0.06, 0.16))
        squared = (x - x[center]) ** 2 + (y - y[center]) ** 2
        fields.append(np.exp(-0.5 * squared / (sigma_km * sigma_km)))
    amplitudes = rng.uniform(1.0, 3.0, size=len(fields))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(fields))
    orders_per_robot_hour = float(rng.uniform(0.20, 0.42))
    rates = np.empty((duration_hours, len(node_ids)), dtype=np.float64)
    for hour in range(duration_hours):
        time_fraction = (hour + 0.5) / duration_hours
        workload_scale = 0.8 + 0.4 * math.sin(math.pi * time_fraction)
        raw = np.full(len(node_ids), 0.15, dtype=np.float64)
        for field, amplitude, phase in zip(fields, amplitudes, phases, strict=True):
            raw += amplitude * (1.0 + 0.5 * math.sin(2.0 * math.pi * time_fraction + phase)) * field
        target = fleet_count * orders_per_robot_hour * workload_scale
        rates[hour] = raw * (target / float(raw.sum()))
    return NodeDemandProfile(node_ids, rates, bucket_minutes=60.0), orders_per_robot_hour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/scenarios/multicity_v1")
    parser.add_argument("--cities", nargs="+", choices=CITIES, default=list(DEFAULT_CITIES))
    parser.add_argument("--train-per-city", type=int, default=20)
    parser.add_argument("--test-per-city", type=int, default=5)
    parser.add_argument("--duration-hours", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse exactly matching generated scenario files.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("choose resume or overwrite, not both")
    if min(args.train_per_city, args.test_per_city, args.duration_hours) <= 0:
        parser.error("train/test counts and duration must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads((ROOT / "configs/reservation_training_manifest.example.json").read_text(encoding="utf-8"))
    suite: dict[str, object] = {
        "generator": "create_multicity_suite.py",
        "base_seed": args.base_seed,
        "duration_hours": args.duration_hours,
        "cities": {},
    }
    training_instances: list[dict[str, object]] = []

    for city_index, city in enumerate(args.cities):
        graph_path = (args.graph_dir / f"{city}.graphml").resolve()
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        node_ids, latitudes, longitudes = read_graphml_nodes(graph_path)
        fleet_count = max(1, math.ceil(len(node_ids) / 500))
        city_records: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
        train_counts: Counter[float] = Counter()
        for split, count in (("train", args.train_per_city), ("test", args.test_per_city)):
            for index in range(count):
                seed = args.base_seed + city_index * 10000 + index
                if split == "test":
                    seed += 500000
                rng = np.random.default_rng(seed)
                demand, rate = make_demand(
                    node_ids, latitudes, longitudes,
                    duration_hours=args.duration_hours,
                    fleet_count=fleet_count,
                    rng=rng,
                )
                probabilities = rng.dirichlet([16.0, 3.0, 1.0])
                creator = ScenarioCreator(
                    importance_distribution=ImportanceDistribution(
                        LEVELS, tuple(float(value) for value in probabilities)
                    )
                )
                scenario = creator.create(city, demand, seed)
                scenario_path = output_dir / split / f"{city}_{split}_{index:03d}.json"
                if scenario_path.exists() and args.resume:
                    if Scenario.load_json(scenario_path) != scenario:
                        raise RuntimeError(f"incompatible existing scenario: {scenario_path}")
                else:
                    if scenario_path.exists() and not args.overwrite:
                        raise FileExistsError(f"refusing to overwrite {scenario_path}")
                    scenario_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = scenario_path.with_suffix('.json.tmp')
                    scenario.save_json(temporary)
                    temporary.replace(scenario_path)
                counts = Counter(float(order.importance) for order in scenario.orders)
                if split == "train":
                    train_counts.update(counts)
                city_records[split].append({
                    "id": f"{city}_{split}_{index:03d}",
                    "graph": _relative(graph_path, output_dir),
                    "scenario": _relative(scenario_path, output_dir),
                    "seed": seed,
                    "orders": len(scenario.orders),
                    "importance_counts": {str(int(level)): counts[level] for level in LEVELS},
                    "orders_per_robot_hour": rate,
                })
        rates = {
            str(int(level)): train_counts[level] / (args.train_per_city * args.duration_hours)
            for level in LEVELS
        }
        suite["cities"][city] = {  # type: ignore[index]
            "graph": _relative(graph_path, output_dir),
            "nodes": len(node_ids),
            "robots": fleet_count,
            "prior_rates_per_hour": rates,
            **city_records,
        }
        for record in city_records["train"]:
            training_instances.append({
                "id": record["id"],
                "graph": record["graph"],
                "scenario": record["scenario"],
                "importance_prior_rates_per_hour": rates,
            })
        print(f"{city}: nodes={len(node_ids)} robots={fleet_count} "
              f"train={len(city_records['train'])} test={len(city_records['test'])}", flush=True)

    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    for style in ("spatial_posterior", "paper_moving_average"):
        manifest = {
            "robot_types": template["robot_types"],
            "importance_levels": template["importance_levels"],
            "importance_prior_rates_per_hour": template["importance_prior_rates_per_hour"],
            "prior_concentration": template["prior_concentration"],
            "shortlist_k": template["shortlist_k"],
            "reservation_style": style,
            "reservation_lookback_min": 100.0,
            "instances": training_instances,
            "candidates": template["candidates"],
        }
        (output_dir / f"training_{style}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"suite={suite_path}", flush=True)


if __name__ == "__main__":
    main()
