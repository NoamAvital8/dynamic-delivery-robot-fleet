"""Shared routing primitives for policies and battery-aware planning.

The street graph is static during a simulation, so expensive routing work should
be shared by every policy rather than repeated inside each decision rule.

Two layers are provided:

* :class:`DistanceOracle` owns exact point-to-point routing, small LRU caches,
  and an exact outward Dijkstra iterator that stops as soon as requested target
  nodes are reached.  This is useful for nearest-k robot / transfer-partner
  queries without searching from every candidate independently.
* :class:`ChargerDistanceIndex` precomputes exact distances from every fixed
  charging station to every graph node.  With SciPy available this is done in
  compiled code on a CSR representation of the graph and also stores
  predecessors, allowing station-related paths to be reconstructed without
  another NetworkX shortest-path search.

The index is graph/policy agnostic and can therefore be reused by all dispatch,
insertion, look-ahead, repositioning, and transfer policies in one simulator.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Hashable, Iterable, Iterator

import networkx as nx
import numpy as np

NodeId = Hashable


@dataclass(frozen=True, slots=True)
class NearestTarget:
    node_id: NodeId
    distance_m: float


class DistanceOracle:
    """Exact shared shortest-distance/path service for a static graph."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        edge_weight: str = "length",
        distance_cache_size: int = 100_000,
        path_cache_size: int = 10_000,
    ) -> None:
        self.graph = graph
        self.edge_weight = edge_weight
        self.distance_cache_size = max(0, int(distance_cache_size))
        self.path_cache_size = max(0, int(path_cache_size))
        self._distance_cache: OrderedDict[object, float] = OrderedDict()
        self._path_cache: OrderedDict[tuple[NodeId, NodeId], tuple[NodeId, ...]] = OrderedDict()

    def _distance_key(self, source: NodeId, target: NodeId) -> object:
        if self.graph.is_directed():
            return (source, target)
        if source == target:
            return (source, source)
        # frozenset avoids requiring node IDs to be mutually orderable.
        return frozenset((source, target))

    @staticmethod
    def _lru_get(cache: OrderedDict, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    @staticmethod
    def _lru_put(cache: OrderedDict, key, value, max_size: int) -> None:
        if max_size <= 0:
            return
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def remember_distance(self, source: NodeId, target: NodeId, distance_m: float) -> None:
        self._lru_put(
            self._distance_cache,
            self._distance_key(source, target),
            float(distance_m),
            self.distance_cache_size,
        )

    def distance(self, source: NodeId, target: NodeId) -> float:
        if source == target:
            return 0.0
        key = self._distance_key(source, target)
        cached = self._lru_get(self._distance_cache, key)
        if cached is not None:
            return float(cached)

        # Bidirectional Dijkstra is exact and generally much cheaper than a
        # full single-source search for one arbitrary point-to-point query.
        distance, path = nx.bidirectional_dijkstra(
            self.graph,
            source,
            target,
            weight=self.edge_weight,
        )
        distance = float(distance)
        self.remember_distance(source, target, distance)
        self._lru_put(
            self._path_cache,
            (source, target),
            tuple(path),
            self.path_cache_size,
        )
        if not self.graph.is_directed():
            self._lru_put(
                self._path_cache,
                (target, source),
                tuple(reversed(path)),
                self.path_cache_size,
            )
        return distance

    def path(self, source: NodeId, target: NodeId) -> tuple[NodeId, ...]:
        if source == target:
            return (source,)
        key = (source, target)
        cached = self._lru_get(self._path_cache, key)
        if cached is not None:
            return tuple(cached)

        distance, path = nx.bidirectional_dijkstra(
            self.graph,
            source,
            target,
            weight=self.edge_weight,
        )
        path_tuple = tuple(path)
        self.remember_distance(source, target, float(distance))
        self._lru_put(self._path_cache, key, path_tuple, self.path_cache_size)
        if not self.graph.is_directed():
            self._lru_put(
                self._path_cache,
                (target, source),
                tuple(reversed(path_tuple)),
                self.path_cache_size,
            )
        return path_tuple

    def _edge_cost(self, u: NodeId, v: NodeId, data) -> float:
        if self.graph.is_multigraph():
            return min(
                float(attrs.get(self.edge_weight, 1.0))
                for attrs in data.values()
            )
        return float(data.get(self.edge_weight, 1.0))

    def iter_nearest_targets(
        self,
        source: NodeId,
        target_nodes: Iterable[NodeId],
        *,
        cutoff_m: float | None = None,
    ) -> Iterator[NearestTarget]:
        """Yield target nodes in nondecreasing exact graph distance from source.

        Only the portion of the graph needed to reach the requested targets is
        explored.  A caller looking for the nearest feasible robot can continue
        the *same* iterator when the first candidate is infeasible, rather than
        restarting Dijkstra for the second/third candidate.
        """

        targets = set(target_nodes)
        if not targets:
            return
        if source not in self.graph:
            raise nx.NodeNotFound(f"source {source!r} is not in the graph")

        remaining = set(targets)
        best: dict[NodeId, float] = {source: 0.0}
        settled: set[NodeId] = set()
        counter = itertools.count()
        heap: list[tuple[float, int, NodeId]] = [(0.0, next(counter), source)]

        while heap and remaining:
            distance, _, node = heapq.heappop(heap)
            if node in settled:
                continue
            if distance != best.get(node):
                continue
            if cutoff_m is not None and distance > cutoff_m:
                break
            settled.add(node)

            if node in remaining:
                remaining.remove(node)
                self.remember_distance(source, node, distance)
                yield NearestTarget(node, float(distance))

            for neighbor, data in self.graph[node].items():
                if neighbor in settled:
                    continue
                edge_cost = self._edge_cost(node, neighbor, data)
                if edge_cost < 0:
                    raise ValueError("Dijkstra routing requires nonnegative edge weights")
                candidate = distance + edge_cost
                if cutoff_m is not None and candidate > cutoff_m:
                    continue
                if candidate < best.get(neighbor, math.inf):
                    best[neighbor] = candidate
                    heapq.heappush(heap, (candidate, next(counter), neighbor))

    def distances_to_targets(
        self,
        source: NodeId,
        target_nodes: Iterable[NodeId],
        *,
        cutoff_m: float | None = None,
    ) -> dict[NodeId, float]:
        return {
            item.node_id: item.distance_m
            for item in self.iter_nearest_targets(
                source,
                target_nodes,
                cutoff_m=cutoff_m,
            )
        }


class ChargerDistanceIndex:
    """Exact reusable station-to-node distance/path index.

    The dense backend computes one all-destinations shortest-path row per fixed
    charging station in compiled SciPy code.  Distance rows are stored as
    float32 to keep the NYC index modest in memory; predecessor rows are int32.
    For a 262 x 272,526 NYC index the two matrices together are roughly 545 MiB.

    If SciPy is unavailable, methods transparently fall back to the shared
    :class:`DistanceOracle`; this preserves correctness, only not the speedup.
    """

    NO_PREDECESSOR = -9999

    def __init__(
        self,
        graph: nx.Graph,
        station_nodes: Iterable[NodeId],
        *,
        oracle: DistanceOracle | None = None,
        edge_weight: str = "length",
        build_dense: bool = True,
    ) -> None:
        self.graph = graph
        self.edge_weight = edge_weight
        self.oracle = oracle or DistanceOracle(graph, edge_weight=edge_weight)
        self.station_nodes = tuple(dict.fromkeys(station_nodes))
        if not self.station_nodes:
            raise ValueError("at least one charging station is required")
        self.station_set = set(self.station_nodes)
        self._station_row = {node: i for i, node in enumerate(self.station_nodes)}
        self._node_order = tuple(graph.nodes)
        self._node_index = {node: i for i, node in enumerate(self._node_order)}
        missing = [node for node in self.station_nodes if node not in self._node_index]
        if missing:
            raise ValueError(f"charging stations missing from graph: {missing[:3]!r}")

        self._distances: np.ndarray | None = None
        self._predecessors: np.ndarray | None = None
        if build_dense:
            self._build_dense_if_possible()

    @property
    def is_dense(self) -> bool:
        return self._distances is not None

    def _build_csr(self):
        from scipy.sparse import csr_matrix

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        multigraph = self.graph.is_multigraph()

        # Iterate adjacency rather than edges so undirected graphs naturally
        # receive both directions.  For MultiGraphs take the minimum parallel
        # edge, matching NetworkX weighted shortest-path semantics.
        for u in self._node_order:
            ui = self._node_index[u]
            for v, edge_data in self.graph[u].items():
                vi = self._node_index[v]
                if multigraph:
                    weight = min(
                        float(attrs.get(self.edge_weight, 1.0))
                        for attrs in edge_data.values()
                    )
                else:
                    weight = float(edge_data.get(self.edge_weight, 1.0))
                if weight < 0:
                    raise ValueError("Dijkstra routing requires nonnegative edge weights")
                rows.append(ui)
                cols.append(vi)
                values.append(weight)

        return csr_matrix(
            (np.asarray(values, dtype=np.float64), (rows, cols)),
            shape=(len(self._node_order), len(self._node_order)),
        )

    def _build_dense_if_possible(self) -> None:
        try:
            from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
        except ImportError:
            return

        csr = self._build_csr()
        station_indices = np.asarray(
            [self._node_index[node] for node in self.station_nodes],
            dtype=np.int64,
        )
        distances, predecessors = scipy_dijkstra(
            csr,
            directed=self.graph.is_directed(),
            indices=station_indices,
            return_predecessors=True,
        )
        # float32 precision is far below a meter at city-scale distances and
        # halves memory.  Predecessors returned by SciPy are already int32.
        self._distances = np.asarray(distances, dtype=np.float32)
        self._predecessors = np.asarray(predecessors, dtype=np.int32)

        # The CSR can be released after preprocessing.  Also seed the oracle's
        # station-pair distance cache because those queries are common.
        for i, a in enumerate(self.station_nodes):
            column_indices = [self._node_index[b] for b in self.station_nodes]
            row_values = self._distances[i, column_indices]
            for b, value in zip(self.station_nodes, row_values):
                if np.isfinite(value):
                    self.oracle.remember_distance(a, b, float(value))

    def distance(self, station_node: NodeId, node: NodeId) -> float:
        if station_node not in self.station_set:
            raise ValueError("first argument must be a charging station")
        if self._distances is None:
            return self.oracle.distance(station_node, node)
        value = float(
            self._distances[
                self._station_row[station_node],
                self._node_index[node],
            ]
        )
        if not math.isfinite(value):
            raise nx.NetworkXNoPath(f"no path between {station_node!r} and {node!r}")
        return value

    def distances_to_stations(
        self,
        node: NodeId,
        *,
        cutoff_m: float | None = None,
        include_self: bool = True,
    ) -> dict[NodeId, float]:
        if self._distances is None:
            result = self.oracle.distances_to_targets(
                node,
                self.station_nodes,
                cutoff_m=cutoff_m,
            )
        else:
            col = self._distances[:, self._node_index[node]]
            mask = np.isfinite(col)
            if cutoff_m is not None:
                mask &= col <= float(cutoff_m) + 1e-3
            indices = np.flatnonzero(mask)
            result = {
                self.station_nodes[int(i)]: float(col[int(i)])
                for i in indices
            }
        if not include_self:
            result.pop(node, None)
        return result

    def station_neighbors(
        self,
        station_node: NodeId,
        cutoff_m: float,
    ) -> dict[NodeId, float]:
        if station_node not in self.station_set:
            raise ValueError("source must be a charging station")
        return self.distances_to_stations(
            station_node,
            cutoff_m=cutoff_m,
            include_self=False,
        )

    def path_from_station(
        self,
        station_node: NodeId,
        node: NodeId,
    ) -> tuple[NodeId, ...]:
        """Return an exact shortest path station -> node."""

        if station_node not in self.station_set:
            raise ValueError("source must be a charging station")
        if station_node == node:
            return (station_node,)
        if self._predecessors is None:
            return self.oracle.path(station_node, node)

        row = self._station_row[station_node]
        source_idx = self._node_index[station_node]
        current = self._node_index[node]
        reversed_indices = [current]
        max_steps = len(self._node_order) + 1
        for _ in range(max_steps):
            if current == source_idx:
                break
            current = int(self._predecessors[row, current])
            if current == self.NO_PREDECESSOR:
                raise nx.NetworkXNoPath(f"no path between {station_node!r} and {node!r}")
            reversed_indices.append(current)
        else:
            raise RuntimeError("predecessor chain did not terminate")

        reversed_indices.reverse()
        return tuple(self._node_order[i] for i in reversed_indices)

    def path(self, source: NodeId, target: NodeId) -> tuple[NodeId, ...]:
        """Use station predecessors whenever either endpoint is a station."""

        if source in self.station_set:
            return self.path_from_station(source, target)
        if not self.graph.is_directed() and target in self.station_set:
            return tuple(reversed(self.path_from_station(target, source)))
        return self.oracle.path(source, target)
