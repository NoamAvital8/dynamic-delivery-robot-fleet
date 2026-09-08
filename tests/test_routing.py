import math

import networkx as nx

from delivery_fleet.routing import ChargerDistanceIndex, DistanceOracle


def weighted_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(0, 1, length=1.0)
    graph.add_edge(1, 2, length=1.0)
    graph.add_edge(2, 3, length=1.0)
    graph.add_edge(3, 4, length=1.0)
    graph.add_edge(1, 5, length=2.5)
    return graph


def test_nearest_targets_are_yielded_from_one_outward_search() -> None:
    graph = weighted_graph()
    oracle = DistanceOracle(graph)

    found = list(oracle.iter_nearest_targets(0, {5, 4, 2}))

    assert [item.node_id for item in found] == [2, 5, 4]
    assert [item.distance_m for item in found] == [2.0, 3.5, 4.0]


def test_nearest_targets_respect_cutoff() -> None:
    graph = weighted_graph()
    oracle = DistanceOracle(graph)

    found = list(oracle.iter_nearest_targets(0, {2, 4}, cutoff_m=2.5))

    assert [(item.node_id, item.distance_m) for item in found] == [(2, 2.0)]


def test_dense_charger_index_returns_exact_distances_and_paths() -> None:
    graph = weighted_graph()
    oracle = DistanceOracle(graph)
    index = ChargerDistanceIndex(
        graph,
        station_nodes=[0, 4],
        oracle=oracle,
        build_dense=True,
    )

    assert index.is_dense
    assert math.isclose(index.distance(0, 3), 3.0, abs_tol=1e-5)
    assert index.distances_to_stations(2, cutoff_m=2.1) == {0: 2.0, 4: 2.0}
    assert index.station_neighbors(0, 3.9) == {}
    assert index.station_neighbors(0, 4.1) == {4: 4.0}
    assert index.path_from_station(0, 4) == (0, 1, 2, 3, 4)
    assert index.path(2, 4) == (2, 3, 4)


def test_dense_index_uses_minimum_parallel_edge_weight() -> None:
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, length=10.0)
    graph.add_edge(0, 1, length=2.0)
    graph.add_edge(1, 2, length=3.0)

    index = ChargerDistanceIndex(graph, station_nodes=[0, 2], build_dense=True)

    assert math.isclose(index.distance(0, 2), 5.0, abs_tol=1e-5)
    assert index.path_from_station(0, 2) == (0, 1, 2)
