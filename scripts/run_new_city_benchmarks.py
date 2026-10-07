"""Prepare missing charger annotations once, then run only the six new cities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import networkx as nx
from create_multicity_suite import NEW_CITIES
from initialize_chargers import initialize_chargers, normalize_edge_lengths, truthy
from run_multicity_experiments import POLICIES, _atomic_json, _sha256, _source_digest, mle_worker_budget


def validate_graph(graph):
    if graph.is_directed() or len(graph) < 2 or not nx.is_connected(graph):
        raise ValueError('expected one connected undirected city graph')
    clusters, representatives = set(), set()
    for node, attrs in graph.nodes(data=True):
        x, y = float(attrs['x']), float(attrs['y'])
        if not (math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90):
            raise ValueError(f'invalid coordinates at {node}')
        cluster = int(attrs['in_cluster'])
        if cluster < 0:
            raise ValueError(f'unassigned cluster at {node}')
        clusters.add(cluster)
        if truthy(attrs.get('is_cluster_representative', False)):
            representatives.add(cluster)
    if clusters != representatives:
        raise ValueError('each cluster must have a saved representative')
    for *_, attrs in graph.edges(data=True):
        length = float(attrs['length'])
        if not math.isfinite(length) or length < 0:
            raise ValueError('invalid edge length')
    return len(clusters)


def prepare_city(job):
    city, graph_dir, prepared_dir, source_digest = job
    source = Path(graph_dir) / f'{city}.graphml'
    destination = Path(prepared_dir) / source.name
    stamp = destination.with_suffix('.prepared.json')
    source_hash = _sha256(source)
    signature = {'source_graph_sha256': source_hash, 'preparation_source_sha256': source_digest}
    if stamp.exists():
        record = json.loads(stamp.read_text())
        if (any(record.get(k) != v for k, v in signature.items()) or not destination.exists()
                or _sha256(destination) != record['prepared_graph_sha256']):
            raise RuntimeError(f'incompatible prepared graph {city}; use a new campaign folder')
        print(f'{city}: reused compatible prepared graph', flush=True)
        return record
    started = time.perf_counter()
    print(f'{city}: loading and validating {source}', flush=True)
    graph = nx.read_graphml(source, node_type=int)
    normalize_edge_lengths(graph)
    clusters = validate_graph(graph)
    stations = tuple(n for n, a in graph.nodes(data=True) if truthy(a.get('is_charging_station', False)))
    if not stations:
        print(f'{city}: placing deterministic chargers once', flush=True)
        stations = initialize_chargers(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.graphml.tmp')
    nx.write_graphml(graph, temporary)
    if destination.exists():
        if _sha256(destination) != _sha256(temporary):
            raise RuntimeError(f'unstamped prepared graph differs from expected output: {destination}')
        temporary.unlink()  # Exact temporary file just written and compared above.
    else:
        temporary.replace(destination)
    record = {'city': city, **signature, 'prepared_graph_sha256': _sha256(destination),
              'nodes': len(graph), 'edges': graph.number_of_edges(), 'clusters': clusters,
              'charging_stations': len(stations), 'prepared_graph': str(destination),
              'preparation_seconds': time.perf_counter() - started}
    _atomic_json(stamp, record)
    print(f'{city}: prepared nodes={len(graph)} chargers={len(stations)} '
          f'elapsed={record["preparation_seconds"]:.0f}s', flush=True)
    return record


def check_suite(path):
    suite = json.loads(path.read_text())
    if tuple(suite['cities']) != NEW_CITIES:
        raise ValueError('suite must contain ONLY the six new cities in the configured order')
    if any(len(data['test']) != 5 for data in suite['cities'].values()):
        raise ValueError('expected five held-out scenarios per new city')
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--spatial-model', type=Path, required=True)
    parser.add_argument('--paper-model', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=12)
    parser.add_argument('--idle-processes', type=int, default=4)
    parser.add_argument('--prepare-processes', type=int, default=2)
    parser.add_argument('--base-seed', type=int, default=20261007)
    args = parser.parse_args()
    if min(args.processes, args.idle_processes, args.prepare_processes) < 1:
        parser.error('process counts must be positive')
    if args.prepare_processes > 6 or mle_worker_budget(args.processes, args.idle_processes, 1) > 60:
        parser.error('preparation cap is six workers; benchmark compute-worker cap is 60')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = out / 'stages'; stages.mkdir(exist_ok=True)
    graph_dir = args.graph_dir.resolve()
    source_digest = _source_digest()
    config = {'cities': list(NEW_CITIES), 'policies': list(POLICIES), 'train_per_city': 20,
              'test_per_city': 5, 'duration_hours': 12, 'base_seed': args.base_seed,
              'source_sha256': source_digest, 'graph_dir': str(graph_dir),
              'graph_sha256': {c: _sha256(graph_dir/f'{c}.graphml') for c in NEW_CITIES},
              'spatial_model': str(args.spatial_model.resolve()), 'paper_model': str(args.paper_model.resolve()),
              'model_sha256': {'spatial': _sha256(args.spatial_model), 'paper': _sha256(args.paper_model)},
              'benchmark_processes': args.processes, 'idle_processes': args.idle_processes,
              'mle_processes': 1, 'prepare_processes': args.prepare_processes,
              'timeout_seconds': 0, 'expected_evaluations': 420}
    config_path = out / 'campaign.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('campaign inputs changed; use a new output folder')
    _atomic_json(config_path, config)
    state = out / 'campaign_status.json'
    _atomic_json(state, {'stage': 'preparing_maps', 'status': 'running'})
    try:
        jobs = [(city, str(graph_dir), str(out/'maps'), source_digest) for city in NEW_CITIES]
        with ProcessPoolExecutor(max_workers=args.prepare_processes,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            records = list(pool.map(prepare_city, jobs))
        _atomic_json(out/'prepared_maps.json', {'maps': records})
        suite_path = out / 'suite/suite.json'
        if not suite_path.exists():
            _atomic_json(state, {'stage': 'generating_scenarios_and_calibration_priors', 'status': 'running'})
            command = [sys.executable, '-u', str(ROOT/'scripts/create_multicity_suite.py'),
                       '--graph-dir', str(out/'maps'), '--output-dir', str(out/'suite'),
                       '--cities', *NEW_CITIES, '--train-per-city', '20', '--test-per-city', '5',
                       '--duration-hours', '12', '--base-seed', str(args.base_seed)]
            command.append('--resume')
            with (stages/'suite.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_suite(suite_path)
        _atomic_json(state, {'stage': 'benchmarks', 'status': 'running', 'expected_evaluations': 420})
        command = [sys.executable, '-u', str(ROOT/'scripts/run_multicity_experiments.py'),
                   str(suite_path), str(out/'benchmarks'), '--spatial-model', config['spatial_model'],
                   '--paper-model', config['paper_model'], '--policies', *POLICIES,
                   '--processes', str(args.processes), '--idle-processes', str(args.idle_processes),
                   '--mle-processes', '1', '--timeout-seconds', '0', '--job-order', 'round-robin']
        print(f'starting 420 benchmarks, ONLY {list(NEW_CITIES)}', flush=True)
        with (stages/'benchmarks.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        _atomic_json(state, {'stage': 'complete', 'status': 'complete', 'expected_evaluations': 420})
    except Exception as exc:
        _atomic_json(state, {'stage': 'failed', 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
