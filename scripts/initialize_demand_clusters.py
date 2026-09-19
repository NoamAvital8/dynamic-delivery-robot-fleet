from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_fleet.spatial_demand import annotate_hdbscan_clusters


def initialize_graph(
    graph_path: Path,
    *,
    output_path: Path | None = None,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_epsilon_m: float = 0.0,
) -> dict:
    """Annotate one GraphML file with complete offline demand clusters."""

    graph = nx.read_graphml(graph_path, node_type=int)

    summary = annotate_hdbscan_clusters(
        graph,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon_m=cluster_selection_epsilon_m,
    )

    if output_path is None:
        output_path = graph_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)

    payload = {
        "graph": str(output_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "clusters": summary.cluster_count,
        "hdbscan_noise_nodes": summary.hdbscan_noise_nodes,
        "cluster_sizes": summary.cluster_sizes,
        "representatives": {
            str(cluster_id): int(node)
            for cluster_id, node in summary.representatives.items()
        },
        "cluster_attribute": "in_cluster",
        "method": graph.graph.get("demand_cluster_method"),
        "min_cluster_size": graph.graph.get("demand_cluster_min_cluster_size"),
        "min_samples": graph.graph.get("demand_cluster_min_samples"),
        "cluster_selection_epsilon_m": graph.graph.get(
            "demand_cluster_selection_epsilon_m"
        ),
    }

    metadata_path = output_path.with_name(output_path.stem + "_clusters.json")
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", nargs="+", type=Path)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon-m", type=float, default=0.0)
    args = parser.parse_args()

    for graph_path in args.graphs:
        payload = initialize_graph(
            graph_path,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon_m=args.cluster_selection_epsilon_m,
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
