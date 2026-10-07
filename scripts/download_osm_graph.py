from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import networkx as nx
import osmnx as ox
from requests.exceptions import RequestException


PLACES = {
    "tel_aviv": "Tel Aviv-Yafo, Israel",
    "haifa": "Haifa, Israel",
    "manhattan": "Manhattan, New York City, New York, USA",
    "new_york_city": "New York City, New York, USA",
    "barcelona": "Barcelona, Catalonia, Spain",
    "paris": "Paris, France",
    "new_delhi": "New Delhi, Delhi, India",
    "moscow": "Moscow, Russia",
    "johannesburg": "City of Johannesburg Metropolitan Municipality, Gauteng, South Africa",
    "beijing": "Beijing, China",
    "sydney": "Sydney, New South Wales, Australia",
}


def download_graph(
    city_key: str, output_dir: Path, *, attempts: int = 4,
    retry_delay_seconds: float = 15.0, resume: bool = False,
) -> dict:
    if attempts < 1 or retry_delay_seconds < 0:
        raise ValueError("attempts must be positive and retry delay non-negative")
    place = PLACES[city_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"{city_key}.graphml"
    metadata_path = output_dir / f"{city_key}.json"
    if resume and graph_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("key") == city_key and metadata.get("place") == place
                and metadata.get("network_type") == "walk"
                and metadata.get("simplified") is True
                and metadata.get("undirected") is True
                and metadata.get("largest_connected_component_only") is True
                and metadata.get("graphml") == graph_path.name):
            print(f"Reusing completed graph: {city_key}", flush=True)
            return metadata
    print(f"Downloading {city_key}: {place}", flush=True)

    # Delivery robots are modeled on a pedestrian-like street network.
    # OSMnx caches successful subrequests, so retries also retain progress
    # within a large city. Retry network errors only, not invalid graph data.
    ox.settings.use_cache = True
    for attempt in range(1, attempts + 1):
        try:
            G = ox.graph_from_place(
                place, network_type="walk", simplify=True, retain_all=False,
            )
            break
        except RequestException as exc:
            response = exc.response
            if response is not None and response.status_code not in {
                408, 429, 500, 502, 503, 504,
            }:
                raise
            if attempt == attempts:
                raise
            delay = retry_delay_seconds * 2 ** (attempt - 1)
            print(f"{city_key}: {type(exc).__name__} on attempt "
                  f"{attempt}/{attempts}; retrying in {delay:g}s", flush=True)
            time.sleep(delay)

    # Our project treats the road network as undirected.
    G = ox.convert.to_undirected(G)

    # Keep one connected routing component so every retained node is reachable.
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    temporary_graph = graph_path.with_suffix(".graphml.tmp")
    ox.save_graphml(G, temporary_graph)
    temporary_graph.replace(graph_path)

    metadata = {
        "key": city_key,
        "place": place,
        "network_type": "walk",
        "simplified": True,
        "undirected": True,
        "largest_connected_component_only": True,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "graphml": graph_path.name,
    }

    _save_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def _save_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("city", choices=[*PLACES, "all"])
    parser.add_argument("--output-dir", default="data/graphs")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed city graphs and metadata.")
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("attempts must be positive")

    output_dir = Path(args.output_dir)
    cities = PLACES.keys() if args.city == "all" else [args.city]
    summary = []
    for city in cities:
        summary.append(download_graph(
            city, output_dir, attempts=args.attempts, resume=args.resume,
        ))
        _save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
