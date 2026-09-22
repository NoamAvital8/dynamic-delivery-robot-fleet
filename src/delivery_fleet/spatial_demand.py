"""Offline spatial clustering and online Gamma-Poisson demand estimation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from typing import Hashable, Mapping

import networkx as nx
import numpy as np

NodeId = Hashable
CLUSTER_NODE_ATTR = "in_cluster"
CLUSTER_REPRESENTATIVE_ATTR = "is_cluster_representative"
EARTH_RADIUS_M = 6_371_008.8
_EPS = 1e-12


def haversine_distance_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the great-circle distance between two WGS84 coordinates.

    The result is a cheap geographic approximation in meters.  It is intended
    for heuristic ranking only; exact route evaluation must still use graph
    distance and the battery-feasible router.
    """

    values = (latitude_a, longitude_a, latitude_b, longitude_b)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("coordinates must be finite")
    if not -90.0 <= float(latitude_a) <= 90.0:
        raise ValueError("latitude_a must be between -90 and 90 degrees")
    if not -90.0 <= float(latitude_b) <= 90.0:
        raise ValueError("latitude_b must be between -90 and 90 degrees")

    phi_a = math.radians(float(latitude_a))
    phi_b = math.radians(float(latitude_b))
    delta_phi = phi_b - phi_a
    delta_lambda = math.radians(float(longitude_b) - float(longitude_a))
    haversine = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a)
        * math.cos(phi_b)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    # Floating-point roundoff can put a theoretically valid value just outside
    # [0, 1], especially for antipodal or identical points.
    haversine = min(1.0, max(0.0, haversine))
    return float(2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(haversine)))


def haversine_node_distance_m(
    graph: nx.Graph,
    source: NodeId,
    target: NodeId,
) -> float:
    """Return cheap straight-line distance between two OSM-style graph nodes."""

    try:
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        return haversine_distance_m(
            float(source_data["y"]),
            float(source_data["x"]),
            float(target_data["y"]),
            float(target_data["x"]),
        )
    except KeyError as exc:
        raise ValueError(
            "source and target nodes must exist and contain OSM-style x/y coordinates"
        ) from exc


def haversine_response_times_by_cluster(
    graph: nx.Graph,
    source_node: NodeId,
    speed_mps: float,
    representatives: Mapping[int, NodeId],
    *,
    available_in_min: float = 0.0,
    charging_delay_min_by_cluster: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Estimate response time from one robot to every cluster representative.

    Haversine supplies the cheap travel-distance term used by the reservation
    heuristic.  Robot availability and optional estimated charging delay are
    added in minutes.  No graph shortest-path query is performed here.
    """

    speed_mps = float(speed_mps)
    available_in_min = float(available_in_min)
    if speed_mps <= 0 or not math.isfinite(speed_mps):
        raise ValueError("speed_mps must be finite and positive")
    if available_in_min < 0 or not math.isfinite(available_in_min):
        raise ValueError("available_in_min must be finite and non-negative")

    charging_delays = charging_delay_min_by_cluster or {}
    unknown_delays = set(charging_delays) - {int(cid) for cid in representatives}
    if unknown_delays:
        raise ValueError(
            f"charging delays supplied for unknown clusters {sorted(unknown_delays)}"
        )

    response_times: dict[int, float] = {}
    for raw_cluster_id, representative in representatives.items():
        cluster_id = int(raw_cluster_id)
        charging_delay = float(charging_delays.get(cluster_id, 0.0))
        if charging_delay < 0 or not math.isfinite(charging_delay):
            raise ValueError("charging delays must be finite and non-negative")
        distance_m = haversine_node_distance_m(graph, source_node, representative)
        response_times[cluster_id] = (
            available_in_min + distance_m / speed_mps / 60.0 + charging_delay
        )
    return response_times


@dataclass(frozen=True, slots=True)
class SpatialClusterSummary:
    """Summary of a complete persisted graph partition."""

    cluster_sizes: dict[int, int]
    representatives: dict[int, NodeId]
    hdbscan_noise_nodes: int

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_sizes)

    @property
    def node_count(self) -> int:
        return sum(self.cluster_sizes.values())


@dataclass(frozen=True, slots=True)
class GammaPosterior:
    """Gamma(shape, rate) distribution parameters."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError("Gamma shape must be positive")
        if self.rate <= 0:
            raise ValueError("Gamma rate must be positive")

    @property
    def mean_per_minute(self) -> float:
        return self.shape / self.rate

    @property
    def mean_per_hour(self) -> float:
        return 60.0 * self.mean_per_minute


def _load_hdbscan_class():
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore
        return HDBSCAN
    except (ImportError, AttributeError):
        try:
            from hdbscan import HDBSCAN  # type: ignore
            return HDBSCAN
        except ImportError as exc:
            raise RuntimeError(
                "HDBSCAN requires scikit-learn>=1.3 or the hdbscan package"
            ) from exc


def _project_nodes(graph: nx.Graph) -> tuple[list[NodeId], np.ndarray]:
    """Project latitude/longitude to a local metric plane for clustering."""

    nodes = list(graph.nodes())
    if not nodes:
        raise ValueError("graph must contain at least one node")
    try:
        lat = np.radians(
            np.asarray([float(graph.nodes[n]["y"]) for n in nodes], dtype=float)
        )
        lon = np.radians(
            np.asarray([float(graph.nodes[n]["x"]) for n in nodes], dtype=float)
        )
    except KeyError as exc:
        raise ValueError("nodes must contain OSM-style x/y coordinates") from exc

    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("node coordinates must be finite")

    lat0 = float(np.mean(lat))
    xy = np.column_stack(
        (
            EARTH_RADIUS_M * np.cos(lat0) * lon,
            EARTH_RADIUS_M * lat,
        )
    )
    return nodes, xy


def _representative(
    nodes: list[NodeId],
    xy: np.ndarray,
    member_indices: np.ndarray,
) -> NodeId:
    """Return the cluster member nearest its geometric centroid."""

    member_xy = xy[member_indices]
    centroid = np.mean(member_xy, axis=0)
    squared = np.sum((member_xy - centroid) ** 2, axis=1)
    return nodes[int(member_indices[int(np.argmin(squared))])]


def _edge_length(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    edge_weight: str,
) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        raise RuntimeError(f"missing edge {u!r}->{v!r}")
    if graph.is_multigraph():
        value = min(float(attrs.get(edge_weight, 1.0)) for attrs in data.values())
    else:
        value = float(data.get(edge_weight, 1.0))
    if value < 0 or not math.isfinite(value):
        raise ValueError("edge weights must be finite and non-negative")
    return value


def _nearest_cluster_by_graph_distance(
    graph: nx.Graph,
    representatives: Mapping[int, NodeId],
    *,
    edge_weight: str,
) -> dict[NodeId, int]:
    """One multi-source Dijkstra storing only distance and winning cluster."""

    routing_graph = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    distance: dict[NodeId, float] = {}
    owner: dict[NodeId, int] = {}
    heap: list[tuple[float, int, str, NodeId]] = []

    for cluster_id, node in representatives.items():
        cid = int(cluster_id)
        distance[node] = 0.0
        owner[node] = cid
        heapq.heappush(heap, (0.0, cid, str(node), node))

    while heap:
        dist, cid, _, node = heapq.heappop(heap)
        if dist > distance.get(node, math.inf) + _EPS:
            continue
        if owner.get(node) != cid:
            continue

        for neighbor in routing_graph.neighbors(node):
            candidate = dist + _edge_length(
                routing_graph, node, neighbor, edge_weight
            )
            old = distance.get(neighbor, math.inf)
            old_owner = owner.get(neighbor, math.inf)
            if (
                candidate < old - _EPS
                or (
                    math.isclose(candidate, old, rel_tol=0.0, abs_tol=_EPS)
                    and cid < old_owner
                )
            ):
                distance[neighbor] = candidate
                owner[neighbor] = cid
                heapq.heappush(heap, (candidate, cid, str(neighbor), neighbor))
    return owner


def summarize_existing_clusters(
    graph: nx.Graph,
    *,
    cluster_attr: str = CLUSTER_NODE_ATTR,
) -> SpatialClusterSummary:
    """Read a complete partition already saved on graph node attributes."""

    sizes: Counter[int] = Counter()
    representatives: dict[int, NodeId] = {}

    for node, data in graph.nodes(data=True):
        if cluster_attr not in data:
            raise ValueError(f"node {node!r} has no {cluster_attr!r}")
        cid = int(data[cluster_attr])
        sizes[cid] += 1
        rep = str(data.get(CLUSTER_REPRESENTATIVE_ATTR, False)).strip().lower()
        if rep in {"1", "true", "yes"}:
            representatives[cid] = node

    if not sizes:
        raise ValueError("graph has no nodes")

    missing = set(sizes) - set(representatives)
    for cid in sorted(missing):
        representatives[cid] = next(
            node
            for node, data in graph.nodes(data=True)
            if int(data[cluster_attr]) == cid
        )

    return SpatialClusterSummary(
        cluster_sizes=dict(sorted(sizes.items())),
        representatives=representatives,
        hdbscan_noise_nodes=int(graph.graph.get("hdbscan_noise_nodes", 0)),
    )


def annotate_hdbscan_clusters(
    graph: nx.Graph,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_epsilon_m: float = 0.0,
    edge_weight: str = "length",
    cluster_attr: str = CLUSTER_NODE_ATTR,
) -> SpatialClusterSummary:
    """Run HDBSCAN once and guarantee that every graph node gets a cluster.

    HDBSCAN core assignments are retained. Nodes marked as HDBSCAN noise are
    attached to the nearest cluster representative by graph shortest-path
    distance. The result is saved in node[cluster_attr].

    This function is intended for graph-build/initialization time, not the
    online decision loop.
    """

    if cluster_selection_epsilon_m < 0:
        raise ValueError("cluster_selection_epsilon_m cannot be negative")

    nodes, xy = _project_nodes(graph)
    n = len(nodes)

    if min_cluster_size is None:
        min_cluster_size = max(20, int(round(math.sqrt(n))))
    if min_samples is None:
        min_samples = max(5, int(min_cluster_size) // 4)
    if int(min_cluster_size) < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be positive")

    HDBSCAN = _load_hdbscan_class()
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_epsilon=float(cluster_selection_epsilon_m),
    )
    labels = np.asarray(model.fit_predict(xy), dtype=np.int64)
    raw_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    noise_count = int(np.sum(labels < 0))

    if not raw_ids:
        all_indices = np.arange(n, dtype=np.int64)
        representative = _representative(nodes, xy, all_indices)
        for node in nodes:
            graph.nodes[node][cluster_attr] = 0
            graph.nodes[node][CLUSTER_REPRESENTATIVE_ATTR] = False
        graph.nodes[representative][CLUSTER_REPRESENTATIVE_ATTR] = True
        graph.graph["demand_cluster_method"] = "hdbscan_single_cluster_fallback"
        graph.graph["demand_cluster_count"] = 1
        graph.graph["hdbscan_noise_nodes"] = noise_count
        return SpatialClusterSummary(
            cluster_sizes={0: n},
            representatives={0: representative},
            hdbscan_noise_nodes=noise_count,
        )

    raw_reps = {
        raw_id: _representative(nodes, xy, np.flatnonzero(labels == raw_id))
        for raw_id in raw_ids
    }
    ordered_raw = sorted(raw_ids, key=lambda cid: str(raw_reps[cid]))
    normalize = {raw_id: cid for cid, raw_id in enumerate(ordered_raw)}
    representatives = {
        normalize[raw_id]: raw_reps[raw_id] for raw_id in ordered_raw
    }

    final_labels: dict[NodeId, int] = {}
    for i, node in enumerate(nodes):
        raw_id = int(labels[i])
        if raw_id >= 0:
            final_labels[node] = normalize[raw_id]

    nearest = _nearest_cluster_by_graph_distance(
        graph, representatives, edge_weight=edge_weight
    )

    node_index = {node: i for i, node in enumerate(nodes)}
    rep_ids = sorted(representatives)
    rep_xy = np.asarray(
        [xy[node_index[representatives[cid]]] for cid in rep_ids],
        dtype=float,
    )

    for node in nodes:
        if node in final_labels:
            continue
        cid = nearest.get(node)
        if cid is None:
            point = xy[node_index[node]]
            cid = rep_ids[int(np.argmin(np.sum((rep_xy - point) ** 2, axis=1)))]
        final_labels[node] = int(cid)

    for node in nodes:
        graph.nodes[node][cluster_attr] = int(final_labels[node])
        graph.nodes[node][CLUSTER_REPRESENTATIVE_ATTR] = False
    for representative in representatives.values():
        graph.nodes[representative][CLUSTER_REPRESENTATIVE_ATTR] = True

    sizes = Counter(final_labels.values())
    graph.graph["demand_cluster_method"] = "hdbscan_plus_graph_nearest_noise"
    graph.graph["demand_cluster_count"] = len(sizes)
    graph.graph["demand_cluster_attr"] = cluster_attr
    graph.graph["demand_cluster_min_cluster_size"] = int(min_cluster_size)
    graph.graph["demand_cluster_min_samples"] = int(min_samples)
    graph.graph["demand_cluster_selection_epsilon_m"] = float(
        cluster_selection_epsilon_m
    )
    graph.graph["hdbscan_noise_nodes"] = noise_count

    return SpatialClusterSummary(
        cluster_sizes={int(k): int(v) for k, v in sorted(sizes.items())},
        representatives=representatives,
        hdbscan_noise_nodes=noise_count,
    )


class GammaPoissonDemandModel:
    """Posterior rate for every (cluster, importance) pair.

    Let q_z = |V_z| / |V| and Lambda_c be the configured city-wide prior
    arrival rate for importance c. With concentration eta:

        alpha0[z,c] = eta * q_z
        beta0[c] = eta / Lambda_c

    using Gamma(shape, rate), with Lambda_c in orders/minute. Therefore:

        E[lambda[z,c]] = q_z * Lambda_c

    so larger clusters have proportionally larger prior mean demand.

    At elapsed time t, after n[z,c] observed arrivals:

        lambda[z,c] | data ~ Gamma(alpha0[z,c] + n[z,c],
                                   beta0[c] + t)

    An arriving order increments only its own (cluster, importance) count.
    Elapsed time is evidence for every pair and is included lazily when the
    posterior is queried.
    """

    def __init__(
        self,
        graph: nx.Graph,
        importance_rates_per_hour: Mapping[float, float],
        *,
        prior_concentration: float = 4.0,
        cluster_attr: str = CLUSTER_NODE_ATTR,
        start_time_min: float = 0.0,
    ) -> None:
        if prior_concentration <= 0:
            raise ValueError("prior_concentration must be positive")
        if start_time_min < 0:
            raise ValueError("start_time_min cannot be negative")
        if not importance_rates_per_hour:
            raise ValueError("importance_rates_per_hour cannot be empty")

        summary = summarize_existing_clusters(graph, cluster_attr=cluster_attr)
        self.cluster_sizes = summary.cluster_sizes
        self.cluster_by_node = {
            node: int(data[cluster_attr]) for node, data in graph.nodes(data=True)
        }
        self.total_nodes = sum(self.cluster_sizes.values())
        self.start_time_min = float(start_time_min)
        self.current_time_min = float(start_time_min)
        self.prior_concentration = float(prior_concentration)

        self.importance_rates_per_minute: dict[float, float] = {}
        for importance, rate_per_hour in importance_rates_per_hour.items():
            importance = float(importance)
            rate_per_hour = float(rate_per_hour)
            if importance <= 0:
                raise ValueError("importance values must be positive")
            if rate_per_hour <= 0 or not math.isfinite(rate_per_hour):
                raise ValueError("importance prior rates must be finite and positive")
            self.importance_rates_per_minute[importance] = rate_per_hour / 60.0

        self.importance_rates_per_minute = dict(
            sorted(self.importance_rates_per_minute.items())
        )
        self.importance_levels = tuple(self.importance_rates_per_minute)
        self._alpha0: dict[tuple[int, float], float] = {}
        self._beta0: dict[float, float] = {}

        for importance, total_rate in self.importance_rates_per_minute.items():
            self._beta0[importance] = self.prior_concentration / total_rate
            for cluster_id, size in self.cluster_sizes.items():
                q_z = size / self.total_nodes
                self._alpha0[(cluster_id, importance)] = (
                    self.prior_concentration * q_z
                )

        self._counts: Counter[tuple[int, float]] = Counter()

    def cluster_for_node(self, node: NodeId) -> int:
        try:
            return self.cluster_by_node[node]
        except KeyError as exc:
            raise KeyError(f"unknown pickup node {node!r}") from exc

    def _importance(self, importance: float) -> float:
        importance = float(importance)
        if importance not in self.importance_rates_per_minute:
            raise KeyError(f"importance {importance!r} has no prior rate")
        return importance

    def observe(
        self,
        pickup_node: NodeId,
        importance: float,
        time_min: float,
    ) -> None:
        time_min = float(time_min)
        if time_min + _EPS < self.current_time_min:
            raise ValueError("observations must arrive in nondecreasing time")
        importance = self._importance(importance)
        cluster_id = self.cluster_for_node(pickup_node)
        self.current_time_min = max(self.current_time_min, time_min)
        self._counts[(cluster_id, importance)] += 1

    def count(self, cluster_id: int, importance: float) -> int:
        return int(self._counts[(int(cluster_id), self._importance(importance))])

    def prior(self, cluster_id: int, importance: float) -> GammaPosterior:
        importance = self._importance(importance)
        cluster_id = int(cluster_id)
        try:
            shape = self._alpha0[(cluster_id, importance)]
        except KeyError as exc:
            raise KeyError(f"unknown cluster {cluster_id}") from exc
        return GammaPosterior(shape, self._beta0[importance])

    def posterior(
        self,
        cluster_id: int,
        importance: float,
        *,
        at_time_min: float | None = None,
    ) -> GammaPosterior:
        importance = self._importance(importance)
        cluster_id = int(cluster_id)
        prior = self.prior(cluster_id, importance)
        now = self.current_time_min if at_time_min is None else float(at_time_min)
        if now + _EPS < self.start_time_min:
            raise ValueError("posterior time cannot precede model start")
        elapsed = max(0.0, now - self.start_time_min)
        return GammaPosterior(
            shape=prior.shape + self._counts[(cluster_id, importance)],
            rate=prior.rate + elapsed,
        )

    def posterior_mean_per_hour(
        self,
        cluster_id: int,
        importance: float,
        *,
        at_time_min: float | None = None,
    ) -> float:
        return self.posterior(
            cluster_id, importance, at_time_min=at_time_min
        ).mean_per_hour

    def rate_at_least_importance_per_minute(
        self,
        cluster_id: int,
        min_importance: float,
        *,
        at_time_min: float | None = None,
    ) -> float:
        threshold = float(min_importance)
        return float(
            sum(
                self.posterior(
                    cluster_id, importance, at_time_min=at_time_min
                ).mean_per_minute
                for importance in self.importance_levels
                if importance >= threshold
            )
        )

    def cluster_weights_for_min_importance(
        self,
        min_importance: float,
        *,
        at_time_min: float | None = None,
    ) -> dict[int, float]:
        rates = {
            cluster_id: self.rate_at_least_importance_per_minute(
                cluster_id, min_importance, at_time_min=at_time_min
            )
            for cluster_id in self.cluster_sizes
        }
        total = sum(rates.values())
        if total <= 0:
            raise RuntimeError("posterior demand unexpectedly sums to zero")
        return {cluster_id: rate / total for cluster_id, rate in rates.items()}


def weighted_reservation_score(
    response_time_min_by_cluster: Mapping[int, float],
    demand_model: GammaPoissonDemandModel,
    min_importance: float,
    *,
    at_time_min: float | None = None,
) -> float:
    """Demand-weighted expected response time; lower is better.

    W[z,c] is the posterior expected request rate in cluster z for importance
    c or higher. Busy clusters therefore contribute more strongly. A robot is
    rewarded for having a small response time to the busy clusters, rather
    than merely for being inside one particular cluster.
    """

    weights = demand_model.cluster_weights_for_min_importance(
        min_importance, at_time_min=at_time_min
    )
    missing = set(weights) - set(response_time_min_by_cluster)
    if missing:
        raise ValueError(f"missing response times for clusters {sorted(missing)}")

    score = 0.0
    for cluster_id, weight in weights.items():
        response = float(response_time_min_by_cluster[cluster_id])
        if response < 0 or not math.isfinite(response):
            raise ValueError("response times must be finite and non-negative")
        score += weight * response
    return float(score)
