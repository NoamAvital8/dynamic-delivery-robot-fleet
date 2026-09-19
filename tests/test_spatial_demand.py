from __future__ import annotations

import math

import networkx as nx
import numpy as np

import delivery_fleet.spatial_demand as spatial
from delivery_fleet.spatial_demand import (
    GammaPoissonDemandModel,
    annotate_hdbscan_clusters,
    weighted_reservation_score,
)


def _clustered_graph() -> nx.Graph:
    graph = nx.Graph()
    for node, cluster_id in [(0, 0), (1, 0), (2, 0), (3, 1)]:
        graph.add_node(
            node,
            x=float(node) * 0.001,
            y=32.0,
            in_cluster=cluster_id,
            is_cluster_representative=(node in {0, 3}),
        )
    graph.add_edge(0, 1, length=100.0)
    graph.add_edge(1, 2, length=100.0)
    graph.add_edge(2, 3, length=100.0)
    return graph


def test_gamma_prior_mean_scales_with_cluster_size() -> None:
    graph = _clustered_graph()
    model = GammaPoissonDemandModel(
        graph,
        {1.0: 40.0, 2.0: 10.0},
        prior_concentration=4.0,
    )

    large = model.prior(0, 1.0).mean_per_hour
    small = model.prior(1, 1.0).mean_per_hour

    assert math.isclose(large, 30.0)
    assert math.isclose(small, 10.0)
    assert math.isclose(large / small, 3.0)


def test_observation_updates_only_matching_cluster_importance_count() -> None:
    graph = _clustered_graph()
    model = GammaPoissonDemandModel(
        graph,
        {1.0: 40.0, 2.0: 10.0},
        prior_concentration=4.0,
    )

    before_target = model.posterior(0, 2.0, at_time_min=15.0)
    before_other = model.posterior(1, 2.0, at_time_min=15.0)

    model.observe(1, 2.0, 15.0)

    after_target = model.posterior(0, 2.0, at_time_min=15.0)
    after_other = model.posterior(1, 2.0, at_time_min=15.0)

    assert model.count(0, 2.0) == 1
    assert model.count(1, 2.0) == 0
    assert math.isclose(after_target.shape, before_target.shape + 1.0)
    assert math.isclose(after_target.rate, before_target.rate)
    assert math.isclose(after_other.shape, before_other.shape)
    assert math.isclose(after_other.rate, before_other.rate)


def test_weighted_reservation_score_prefers_robot_close_to_busy_cluster() -> None:
    graph = _clustered_graph()
    model = GammaPoissonDemandModel(
        graph,
        {1.0: 20.0, 2.0: 20.0},
        prior_concentration=4.0,
    )

    # Cluster 0 contains 75% of graph nodes, so its prior high-priority demand
    # receives 75% of the weight.
    score_near_busy = weighted_reservation_score(
        {0: 2.0, 1: 10.0},
        model,
        min_importance=2.0,
        at_time_min=0.0,
    )
    score_near_quiet = weighted_reservation_score(
        {0: 7.0, 1: 1.0},
        model,
        min_importance=2.0,
        at_time_min=0.0,
    )

    assert score_near_busy < score_near_quiet
    assert math.isclose(score_near_busy, 4.0)
    assert math.isclose(score_near_quiet, 5.5)


def test_hdbscan_noise_nodes_receive_graph_nearest_cluster(monkeypatch) -> None:
    graph = nx.Graph()
    coordinates = {
        0: (0.0000, 32.0),
        1: (0.0005, 32.0),
        2: (0.0010, 32.0),
        3: (0.0100, 32.0),
        4: (0.0105, 32.0),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    graph.add_edge(0, 1, length=50.0)
    graph.add_edge(1, 2, length=50.0)
    graph.add_edge(2, 3, length=900.0)
    graph.add_edge(3, 4, length=50.0)

    class FakeHDBSCAN:
        def __init__(self, **_: object) -> None:
            pass

        def fit_predict(self, _: np.ndarray) -> np.ndarray:
            # Node 2 is noise between two discovered clusters.
            return np.asarray([0, 0, -1, 1, 1], dtype=np.int64)

    monkeypatch.setattr(spatial, "_load_hdbscan_class", lambda: FakeHDBSCAN)

    summary = annotate_hdbscan_clusters(
        graph,
        min_cluster_size=2,
        min_samples=1,
    )

    assert summary.cluster_count == 2
    assert summary.hdbscan_noise_nodes == 1
    assert all("in_cluster" in graph.nodes[node] for node in graph.nodes)
    assert graph.nodes[2]["in_cluster"] == graph.nodes[1]["in_cluster"]
