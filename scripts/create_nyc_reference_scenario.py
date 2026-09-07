from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.scenario_creator import NodeDemandProfile, ScenarioCreator


SCENARIO_SEED = 42
START_HOUR_LOCAL = 8
TARGET_ORDERS_PER_HOUR = np.asarray(
    [60, 75, 95, 120, 145, 165, 160, 150, 170, 190, 180, 140],
    dtype=np.float64,
)
BASELINE_WEIGHT = 0.18

TIME_PROFILES = {
    "business": [1.55, 1.70, 1.55, 1.35, 1.55, 1.50, 1.30, 1.20, 1.05, 0.90, 0.70, 0.55],
    "mixed": [1.00, 1.05, 1.05, 1.15, 1.25, 1.25, 1.20, 1.25, 1.35, 1.40, 1.30, 1.15],
    "residential": [0.85, 0.75, 0.70, 0.75, 0.90, 1.00, 1.10, 1.25, 1.45, 1.65, 1.80, 1.75],
}

HOTSPOTS = [
    {"name": "midtown_manhattan", "lat": 40.7549, "lon": -73.9840, "sigma_km": 2.5, "weight": 1.60, "profile": "business"},
    {"name": "lower_manhattan", "lat": 40.7075, "lon": -74.0113, "sigma_km": 2.5, "weight": 1.20, "profile": "business"},
    {"name": "downtown_brooklyn", "lat": 40.6920, "lon": -73.9860, "sigma_km": 2.8, "weight": 1.10, "profile": "mixed"},
    {"name": "williamsburg", "lat": 40.7180, "lon": -73.9580, "sigma_km": 2.5, "weight": 1.00, "profile": "mixed"},
    {"name": "long_island_city", "lat": 40.7447, "lon": -73.9485, "sigma_km": 2.8, "weight": 0.90, "profile": "mixed"},
    {"name": "flushing", "lat": 40.7590, "lon": -73.8300, "sigma_km": 3.5, "weight": 0.80, "profile": "residential"},
    {"name": "bronx_hub", "lat": 40.8170, "lon": -73.9190, "sigma_km": 3.5, "weight": 0.80, "profile": "residential"},
    {"name": "jamaica", "lat": 40.7030, "lon": -73.8020, "sigma_km": 3.5, "weight": 0.70, "profile": "residential"},
    {"name": "st_george", "lat": 40.6437, "lon": -74.0736, "sigma_km": 4.0, "weight": 0.45, "profile": "residential"},
]


def read_graphml_nodes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only node id/x/y fields without loading the large edge set."""
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    key_names: dict[str, str] = {}
    node_ids: list[int] = []
    latitudes: list[float] = []
    longitudes: list[float] = []

    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == ns + "key":
            if elem.attrib.get("for") == "node":
                key_names[elem.attrib["id"]] = elem.attrib.get("attr.name", "")
            elem.clear()
            continue

        if elem.tag != ns + "node":
            continue

        x = None
        y = None
        for data in elem.findall(ns + "data"):
            name = key_names.get(data.attrib.get("key", ""))
            if name == "x":
                x = float(data.text)
            elif name == "y":
                y = float(data.text)

        if x is not None and y is not None:
            node_ids.append(int(elem.attrib["id"]))
            longitudes.append(x)
            latitudes.append(y)
        elem.clear()

    return (
        np.asarray(node_ids, dtype=np.int64),
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
    )


def haversine_km(
    center_lat: float,
    center_lon: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    earth_radius_km = 6371.0088
    phi1 = np.radians(center_lat)
    phi2 = np.radians(latitudes)
    dphi = np.radians(latitudes - center_lat)
    dlambda = np.radians(longitudes - center_lon)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def build_reference_demand(
    node_ids: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> NodeDemandProfile:
    """Create node-level lambda[v,t], independently of any algorithm clusters."""
    hotspot_fields: list[np.ndarray] = []
    for hotspot in HOTSPOTS:
        distance_km = haversine_km(
            hotspot["lat"], hotspot["lon"], latitudes, longitudes
        )
        field = hotspot["weight"] * np.exp(
            -0.5 * (distance_km / hotspot["sigma_km"]) ** 2
        )
        hotspot_fields.append(field)

    rates = np.empty((12, len(node_ids)), dtype=np.float64)
    for hour_index, target_orders in enumerate(TARGET_ORDERS_PER_HOUR):
        raw = np.full(len(node_ids), BASELINE_WEIGHT, dtype=np.float64)
        for hotspot, field in zip(HOTSPOTS, hotspot_fields, strict=True):
            raw += field * TIME_PROFILES[hotspot["profile"]][hour_index]

        # Normalize so sum_v lambda[v,t] equals the desired city-wide rate.
        rates[hour_index] = raw * (target_orders / raw.sum())

    return NodeDemandProfile(
        node_ids=node_ids,
        lambda_by_bucket=rates,
        bucket_minutes=60.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data" / "graphs" / "new_york_city.graphml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "scenarios",
    )
    args = parser.parse_args()

    node_ids, latitudes, longitudes = read_graphml_nodes(args.graph)
    demand = build_reference_demand(node_ids, latitudes, longitudes)

    scenario = ScenarioCreator().create(
        graph_name="new_york_city",
        demand=demand,
        seed=SCENARIO_SEED,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = args.output_dir / "nyc_reference_12h_seed42.json"
    demand_path = args.output_dir / "nyc_reference_12h_seed42_demand.json"
    scenario.save_json(scenario_path)

    realized_per_hour = [0] * 12
    for order in scenario.orders:
        realized_per_hour[min(int(order.request_time_min // 60), 11)] += 1

    metadata = {
        "name": "nyc_reference_12h_demand_v1",
        "graph_name": "new_york_city",
        "graph_nodes": int(len(node_ids)),
        "duration_hours": 12,
        "start_hour_local": START_HOUR_LOCAL,
        "bucket_minutes": 60.0,
        "scenario_seed": SCENARIO_SEED,
        "model": "continuous_node_level_gaussian_hotspots",
        "formula": "lambda[v,t] = target[t] * raw[v,t] / sum_x raw[x,t]; raw[v,t] = baseline + sum_k weight_k * profile_k[t] * exp(-0.5*(haversine_km(v,center_k)/sigma_km_k)^2)",
        "baseline_weight": BASELINE_WEIGHT,
        "target_orders_per_hour": TARGET_ORDERS_PER_HOUR.tolist(),
        "time_profiles": TIME_PROFILES,
        "hotspots": HOTSPOTS,
        "realized_orders_per_hour": realized_per_hour,
        "realized_total_orders": len(scenario.orders),
        "notes": [
            "Ground-truth pickup demand is defined per OSM node, independently of any clustering used by the planning algorithm.",
            "For any later cluster C, its true lambda at hour t is sum(lambda[v,t] for v in C).",
            "Drop-off nodes are uniform over all graph nodes except the pickup node in this V1 reference scenario.",
        ],
    }
    demand_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"nodes: {len(node_ids):,}")
    print(f"orders: {len(scenario.orders):,}")
    print(f"expected/hour: {TARGET_ORDERS_PER_HOUR.astype(int).tolist()}")
    print(f"realized/hour: {realized_per_hour}")
    print(f"scenario: {scenario_path}")
    print(f"demand metadata: {demand_path}")


if __name__ == "__main__":
    main()
