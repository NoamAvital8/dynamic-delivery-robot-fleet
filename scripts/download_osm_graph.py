from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import osmnx as ox


PLACES = {
    "tel_aviv": "Tel Aviv-Yafo, Israel",
    "haifa": "Haifa, Israel",
    "manhattan": "Manhattan, New York City, New York, USA",
    "new_york_city": "New York City, New York, USA",
    "barcelona": "Barcelona, Catalonia, Spain",
}


def download_graph(city_key: str, output_dir: Path) -> dict:
    place = PLACES[city_key]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delivery robots are modeled on a pedestrian-like street network.
    G = ox.graph_from_place(
        place,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    # Our project treats the road network as undirected.
    G = ox.convert.to_undirected(G)

    # Keep one connected routing component so every retained node is reachable.
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    graph_path = output_dir / f"{city_key}.graphml"
    ox.save_graphml(G, graph_path)

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

    metadata_path = output_dir / f"{city_key}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("city", choices=[*PLACES, "all"])
    parser.add_argument("--output-dir", default="data/graphs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cities = PLACES.keys() if args.city == "all" else [args.city]
    summary = [download_graph(city, output_dir) for city in cities]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
