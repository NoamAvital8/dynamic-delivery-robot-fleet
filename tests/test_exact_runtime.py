import json
import os
from pathlib import Path
import subprocess
import sys
import time

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'operations'))
sys.path.insert(0, str(ROOT/'scripts'))
import exact_runtime as runtime
from delivery_fleet.routing import ChargerDistanceIndex


def graph_fixture():
    graph = nx.MultiGraph()
    graph.add_edge(9, 3, length=698.6494471179697)
    graph.add_edge(9, 3, length=900.)
    graph.add_edge(3, 7, length=100.)
    graph.add_edge(7, 2, length=100.)
    graph.add_edge(3, 2, length=200.)  # Equal paths test predecessor ties.
    return graph


def install_restored(monkeypatch, tmp_path, **kwargs):
    for name in ('_build_dense_if_possible', 'distances_to_stations', 'path_from_station'):
        monkeypatch.setattr(ChargerDistanceIndex, name, getattr(ChargerDistanceIndex, name))
    runtime.install(tmp_path, **kwargs)


def test_query_cache_is_exact_and_mutation_safe(tmp_path, monkeypatch):
    graph = graph_fixture()
    baseline = ChargerDistanceIndex(graph, [9, 2])
    index = ChargerDistanceIndex(graph, [9, 2])
    install_restored(monkeypatch, tmp_path)
    for node in graph:
        for cutoff in (None, 0., 100., 698.6494471179697-1e-9, 698.6494471179697, 1000.):
            for include_self in (True, False):
                expected = baseline.distances_to_stations(node, cutoff_m=cutoff,
                                                          include_self=include_self)
                # Baseline instance now also uses cache but its first call has
                # identical stored dense rows; independently call original below.
                value = index.distances_to_stations(node, cutoff_m=cutoff,
                                                    include_self=include_self)
                assert list(value.items()) == list(expected.items())
                value.clear()
                assert list(index.distances_to_stations(node, cutoff_m=cutoff,
                    include_self=include_self).items()) == list(expected.items())
    assert index.path_from_station(9, 2) == baseline.path_from_station(9, 2)
    assert index.path_from_station(9, 2) == baseline.path_from_station(9, 2)


def test_bounded_caches_eviction_and_no_cutoff_rounding(tmp_path, monkeypatch):
    index = ChargerDistanceIndex(graph_fixture(), [9, 2])
    original = ChargerDistanceIndex.distances_to_stations
    install_restored(monkeypatch, tmp_path, query_cache_bytes=512)
    for i in range(100):
        cutoff = 698.6494471179697 + i*1e-10
        assert index.distances_to_stations(3, cutoff_m=cutoff) == original(index, 3, cutoff_m=cutoff)
        assert index._execution_query_bytes <= 512
    with pytest.raises(ValueError):
        runtime.install(tmp_path, workers=5)


@pytest.mark.skipif(sys.platform != 'linux', reason='production mmap locking is Linux')
@pytest.mark.parametrize('directed', [False, True])
def test_parallel_shared_index_bitwise_distances_and_predecessors(tmp_path, directed):
    graph = graph_fixture()
    if directed:
        graph = graph.to_directed()
    baseline = ChargerDistanceIndex(graph, [9, 2])
    index = ChargerDistanceIndex(graph, [9, 2], build_dense=False)
    key = runtime.shared_dense(index, tmp_path, workers=2)
    assert np.array_equal(index._distances, baseline._distances)
    assert np.array_equal(index._predecessors, baseline._predecessors)
    assert not index._distances.flags.writeable
    for a in index.station_nodes:
        for b in graph:
            assert index.path_from_station(a, b) == baseline.path_from_station(a, b)
    ready = tmp_path/key/'ready.json'
    before = {p.name: p.read_bytes() for p in ready.parent.glob('*.npy')}
    second = ChargerDistanceIndex(graph, [9, 2], build_dense=False)
    assert runtime.shared_dense(second, tmp_path, 2) == key
    assert all((ready.parent/name).read_bytes() == value for name, value in before.items())
    # Source order / edges / station order cannot alias another index.
    changed = graph.copy()
    changed[9][3][0]['length'] += .001
    other = ChargerDistanceIndex(changed, [9, 2], build_dense=False)
    assert runtime.index_key(other, other._build_csr()) != key
    (ready.parent/'distances.npy').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='checksum'):
        runtime.shared_dense(second, tmp_path, 2)


def semantic_result(value):
    if isinstance(value, dict):
        return {k: semantic_result(v) for k, v in value.items()
                if not any(x in k for x in ('seconds', 'wall_clock', 'index_build'))
                and not k.endswith(('_ms', '_file'))
                and k not in ('compiled_dijkstra_calls','compiled_dijkstra_cache_hits')}
    if isinstance(value, list):
        return [semantic_result(v) for v in value]
    return value


@pytest.mark.skipif(sys.platform != 'linux', reason='production wrapper uses Linux locking')
@pytest.mark.parametrize('policy', ['full', 'myopic_ab', 'reactive_insertion',
    'paper_sa_adapted', 'full_no_nn', 'full_no_idle', 'full_queue_aware',
    'full_coordinated_idle', 'queue_mle_reservation', 'full_mle_reservation'])
def test_all_ten_policies_identical_end_to_end(tmp_path, policy):
    from delivery_fleet.charging import annotate_nearest_charging_stations
    from delivery_fleet.fleet import RobotType
    from delivery_fleet.reservation_nn import ReservationFCNN
    from delivery_fleet.scenario_creator import Item, Order, Scenario
    from run_multicity_experiments import policy_command
    graph = nx.path_graph(1001)
    for node in graph:
        graph.nodes[node].update(x=34.+node*.00001, y=32., in_cluster=0,
            is_cluster_representative=(node==500), is_charging_station=(node in {0,500,1000}))
    nx.set_edge_attributes(graph, 10., 'length')
    annotate_nearest_charging_stations(graph, [0,500,1000])
    graphpath = tmp_path/'graph.graphml'; nx.write_graphml(graph, graphpath)
    scenario = tmp_path/'scenario.json'
    Scenario(graph_name='optimization_regression', duration_minutes=45., seed=42,
        orders=tuple(Order(id=i, pickup_node=100+(i%3)*100, dropoff_node=800-(i%3)*80,
            request_time_min=float(i*2), item=Item(weight_kg=1., volume_l=1.),
            importance=(1.,2.,5.)[i%3]) for i in range(8))).save_json(scenario)
    model = tmp_path/'model.npz'
    ReservationFCNN(4, [t.value for t in RobotType], (1.,2.,5.)).save(model)
    results = []
    queue_audits = []
    for optimized in (False, True):
        output = tmp_path/('optimized.json' if optimized else 'original.json')
        command = policy_command(policy, python=sys.executable, graph=graphpath,
            scenario=scenario, output=output, spatial_model=model, paper_model=model,
            shortlist_k=10, idle_processes=2, prior_rates={'1':10.,'2':2.,'5':1.},
            prior_concentration=4.)
        if optimized:
            command = [sys.executable, str(ROOT/'operations/run_exact_optimized_policy.py'),
                       '--cache-dir', str(tmp_path/'cache'), '--workers','2','--',*command[1:]]
        environment = {**os.environ, 'PYTHONPATH':str(ROOT/'src'), 'OMP_NUM_THREADS':'1',
                       'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
        subprocess.run(command, cwd=ROOT, env=environment, check=True,
                       capture_output=True, text=True, timeout=120)
        data = json.loads(output.read_text())
        audit = data.get('queue_wait_records_file')
        queue_audits.append(Path(audit).read_text() if audit else None)
        results.append(semantic_result(data))
    assert results[0] == results[1]
    assert queue_audits[0] == queue_audits[1]
