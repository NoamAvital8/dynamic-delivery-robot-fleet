"""Opt-in execution accelerators; no policy, candidate or numeric changes.

Kept outside scripts/src so the original scientific input signature remains
valid. Execution version/command is recorded separately by the controller.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def index_key(index, csr):
    """Hash all kernel inputs, including ordering and SciPy implementation."""
    import scipy
    h = hashlib.sha256()
    h.update(json.dumps({'version': 1, 'scipy': scipy.__version__,
                         'directed': index.graph.is_directed(),
                         'nodes': [(type(n).__name__, repr(n)) for n in index._node_order],
                         'stations': [(type(n).__name__, repr(n)) for n in index.station_nodes]},
                        separators=(',', ':')).encode())
    for a in (csr.indptr, csr.indices, csr.data):
        h.update(a.dtype.str.encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


_worker_inputs = None


def _initialize_worker(csr, directed, indices, distance_file, predecessor_file):
    global _worker_inputs
    _worker_inputs = (csr, directed, indices, distance_file, predecessor_file)


def _write_chunk(bounds):
    from scipy.sparse.csgraph import dijkstra
    csr, directed, indices, distance_file, predecessor_file = _worker_inputs
    start, end = bounds
    distances, predecessors = dijkstra(csr, directed=directed,
                                       indices=indices[start:end], return_predecessors=True)
    d = np.load(distance_file, mmap_mode='r+')
    p = np.load(predecessor_file, mmap_mode='r+')
    d[start:end] = distances
    p[start:end] = predecessors
    d.flush()
    p.flush()
    return end - start


def _load_valid(folder, key, shape):
    ready = folder/'ready.json'
    if not ready.exists():
        return None
    meta = json.loads(ready.read_text())
    if meta.get('key') != key or tuple(meta.get('shape', ())) != shape:
        raise RuntimeError('shared charger index identity mismatch')
    arrays = []
    for name, dtype in [('distances.npy', np.float64), ('predecessors.npy', np.int32)]:
        path = folder/name
        if file_hash(path) != meta['sha256'][name]:
            raise RuntimeError(f'shared charger index checksum mismatch: {path}')
        value = np.load(path, mmap_mode='r')
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError('shared charger index dtype/shape mismatch')
        arrays.append(value)
    return tuple(arrays)


def shared_dense(index, cache_dir, workers):
    """Identical SciPy per-source Dijkstra rows, parallel and read-only mmap."""
    import fcntl
    from numpy.lib.format import open_memmap
    csr = index._build_csr()  # Existing runner's CSR-retention hook still runs.
    key = index_key(index, csr)
    folder = Path(cache_dir)/key
    folder.mkdir(parents=True, exist_ok=True)
    shape = (len(index.station_nodes), len(index._node_order))
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arrays = _load_valid(folder, key, shape)
        if arrays is None:
            started = time.monotonic()
            # Incomplete builds never become ready. Unique temporary names
            # preserve interrupted artifacts without accidentally adopting them.
            with tempfile.TemporaryDirectory(prefix='building-', dir=folder) as temp:
                temporary = Path(temp)
                dpath, ppath = temporary/'distances.npy', temporary/'predecessors.npy'
                del_d = open_memmap(dpath, mode='w+', dtype=np.float64, shape=shape)
                del_p = open_memmap(ppath, mode='w+', dtype=np.int32, shape=shape)
                del del_d, del_p
                indices = np.asarray([index._node_index[n] for n in index.station_nodes],
                                     dtype=np.int64)
                # Bound each worker's temporary rows to roughly 128 MiB.
                chunk = max(1, min(16, (128 * 1024**2)//max(1, shape[1]*12)))
                bounds = [(i, min(i+chunk, shape[0])) for i in range(0, shape[0], chunk)]
                context = multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=_initialize_worker,
                        initargs=(csr, index.graph.is_directed(), indices, str(dpath), str(ppath))) as pool:
                    finished = 0
                    for count in pool.map(_write_chunk, bounds):
                        finished += count
                        if finished == shape[0] or finished % max(chunk, 256) == 0:
                            print(f'exact shared index {finished}/{shape[0]} rows '
                                  f'elapsed={time.monotonic()-started:.0f}s', flush=True)
                hashes = {name: file_hash(temporary/name)
                          for name in ('distances.npy', 'predecessors.npy')}
                for name in hashes:
                    (temporary/name).replace(folder/name)
                ready_temp = folder/'ready.json.tmp'
                ready_temp.write_text(json.dumps({'key': key, 'shape': shape,
                                                 'sha256': hashes, 'workers': workers}))
                ready_temp.replace(folder/'ready.json')
            arrays = _load_valid(folder, key, shape)
        index._distances, index._predecessors = arrays
    # Preserve original seeding order and float64 values (including LRU eviction).
    columns = [index._node_index[b] for b in index.station_nodes]
    for i, a in enumerate(index.station_nodes):
        row = index._distances[i, columns]
        for b, value in zip(index.station_nodes, row):
            if np.isfinite(value):
                index.oracle.remember_distance(a, b, float(value))
    return key


def install(cache_dir, *, workers=4, query_cache_bytes=64*1024**2, on_index_ready=None):
    """Install once in an otherwise unmodified original simulator process."""
    from delivery_fleet.routing import ChargerDistanceIndex
    if workers < 1 or workers > 4:
        raise ValueError('preprocessing workers must be in 1..4')
    original_dense = ChargerDistanceIndex._build_dense_if_possible
    original_query = ChargerDistanceIndex.distances_to_stations
    original_path = ChargerDistanceIndex.path_from_station

    def build(self):
        try:
            import scipy
        except ImportError:
            return original_dense(self)
        self._execution_shared_index_key = shared_dense(self, cache_dir, workers)
        # The original eight-row cache thrashes when pending requests are
        # revisited. Keep more identical float64 rows, within a 512 MiB budget.
        # This changes recomputation counts, not any distance/ordering value.
        import sys
        fast = sys.modules.get('run_nyc_benchmark2_fast')
        if fast is not None:
            fast.DISTANCE_ROW_CACHE_SIZE = max(8, min(256,
                (512*1024**2)//max(1,len(self._node_order)*8)))
        if on_index_ready is not None:
            on_index_ready(self._execution_shared_index_key)

    def query(self, node, *, cutoff_m=None, include_self=True):
        cache = getattr(self, '_execution_queries', None)
        if cache is None:
            cache = self._execution_queries = OrderedDict()
            self._execution_query_bytes = 0
        key = (node, cutoff_m, include_self)  # No rounding or cutoff bucketing.
        value = cache.pop(key, None)
        if value is None:
            value = original_query(self, node, cutoff_m=cutoff_m, include_self=include_self)
            cost = 256 + 100 * len(value)
            if cost > query_cache_bytes:
                return value
            self._execution_query_bytes += cost
        cache[key] = value
        while self._execution_query_bytes > query_cache_bytes:
            _, removed = cache.popitem(last=False)
            self._execution_query_bytes -= 256 + 100 * len(removed)
        # Preserve public mutable-dict behavior; callers cannot poison cache.
        return value.copy()

    def path(self, station, node):
        cache = getattr(self, '_execution_paths', None)
        if cache is None:
            cache = self._execution_paths = OrderedDict()
            self._execution_path_bytes = 0
        key = (station, node)
        value = cache.pop(key, None)
        if value is None:
            value = original_path(self, station, node)
            cost = 128 + 40 * len(value)
            if cost > query_cache_bytes:
                return value
            self._execution_path_bytes += cost
        cache[key] = value
        while self._execution_path_bytes > query_cache_bytes:
            _, removed = cache.popitem(last=False)
            self._execution_path_bytes -= 128 + 40 * len(removed)
        return value

    ChargerDistanceIndex._build_dense_if_possible = build
    ChargerDistanceIndex.distances_to_stations = query
    ChargerDistanceIndex.path_from_station = path

