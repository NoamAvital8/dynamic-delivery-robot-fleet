from __future__ import annotations

"""Fast exact completion scoring for many battery-route candidates.

The street graph and charger-to-node distances are already precomputed.  This
module avoids running the Python charger-meta Dijkstra once per robot candidate.
Instead, for each distinct robot type in a request, it solves the tiny charger
problem in compiled SciPy code and then scores every robot of that type with
vectorized NumPy operations.  The full BatteryFeasibleRouter.evaluate() call is
still used for the finally selected robot so route/charge actions remain exactly
those of the canonical router.
"""

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from delivery_fleet.charging import (
    DEFAULT_CHARGING_POWER_W,
    DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR,
)

DISTANCE_EPS_M = 1e-3
_SINK_ZERO_EDGE_M = 1e-12


FAST_META_STATS: dict[str, float] = {
    "calls": 0.0,
    "seconds": 0.0,
    "type_solves": 0.0,
    "candidates": 0.0,
}


@dataclass(frozen=True, slots=True)
class FastCandidateScore:
    completion_time_min: float
    route_time_min: float
    route_distance_m: float
    robot_id: int
    candidate: object


@dataclass(frozen=True, slots=True)
class _TypeGraph:
    full_range_m: float
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray


class FastBatteryMetaScorer:
    def __init__(self, router) -> None:
        self.router = router
        index = router.charger_index
        distances = getattr(index, "_distances", None)
        if distances is None:
            raise RuntimeError("fast battery meta scorer requires dense charger index")

        self.index = index
        self.station_nodes = tuple(index.station_nodes)
        self.station_set = set(self.station_nodes)
        self.n = len(self.station_nodes)
        self.node_index = index._node_index
        self.station_node_indices = np.asarray(
            [self.node_index[node] for node in self.station_nodes],
            dtype=np.int64,
        )
        self.station_to_node = distances
        self.station_matrix = np.asarray(
            distances[:, self.station_node_indices],
            dtype=np.float64,
        )
        self._type_graphs: dict[tuple[float, float, float], _TypeGraph] = {}

    @staticmethod
    def _spec_key(spec) -> tuple[float, float, float]:
        return (
            float(spec.speed_mps),
            float(spec.battery_capacity_wh),
            float(spec.energy_per_meter_wh),
        )

    def _type_graph(self, spec) -> _TypeGraph:
        key = self._spec_key(spec)
        cached = self._type_graphs.get(key)
        if cached is not None:
            return cached

        full_range = float(spec.full_battery_range_m)
        feasible = (
            np.isfinite(self.station_matrix)
            & (self.station_matrix <= full_range + DISTANCE_EPS_M)
        )
        np.fill_diagonal(feasible, False)
        rows, cols = np.nonzero(feasible)
        data = self.station_matrix[rows, cols].astype(np.float64, copy=True)
        graph = _TypeGraph(
            full_range_m=full_range,
            rows=rows.astype(np.int32, copy=False),
            cols=cols.astype(np.int32, copy=False),
            data=data,
        )
        self._type_graphs[key] = graph
        return graph

    def _sink_distances(self, graph: _TypeGraph, terminal: np.ndarray) -> np.ndarray:
        """Shortest distance from every charger to a virtual terminal.

        The physical charger graph is undirected, so we run Dijkstra from a
        virtual sink with reversed terminal edges.  A 1e-12 replacement keeps a
        true zero-cost sink edge represented in SciPy sparse storage.
        """
        valid = np.flatnonzero(np.isfinite(terminal))
        if valid.size == 0:
            return np.full(self.n, np.inf, dtype=np.float64)

        sink = self.n
        rows = np.concatenate(
            (graph.rows, np.full(valid.size, sink, dtype=np.int32))
        )
        cols = np.concatenate((graph.cols, valid.astype(np.int32, copy=False)))
        sink_data = np.maximum(terminal[valid], _SINK_ZERO_EDGE_M)
        data = np.concatenate((graph.data, sink_data.astype(np.float64, copy=False)))
        csr = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n + 1, self.n + 1),
            dtype=np.float64,
        )
        result = np.asarray(
            scipy_dijkstra(
                csr,
                directed=True,
                indices=sink,
                return_predecessors=False,
            ),
            dtype=np.float64,
        )
        result = result[: self.n]
        result[result < 1e-10] = 0.0
        return result

    @staticmethod
    def _prefix_min_for_thresholds(
        station_distance: np.ndarray,
        continuation: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """min(distance[j] + continuation[j]) for distance[j] <= threshold."""
        finite = np.isfinite(station_distance) & np.isfinite(continuation)
        if not np.any(finite):
            return np.full(np.shape(thresholds), np.inf, dtype=np.float64)

        d = station_distance[finite]
        h = d + continuation[finite]
        order = np.argsort(d, kind="stable")
        d = d[order]
        prefix = np.minimum.accumulate(h[order])
        pos = np.searchsorted(d, thresholds + DISTANCE_EPS_M, side="right") - 1
        out = np.full(np.shape(thresholds), np.inf, dtype=np.float64)
        ok = pos >= 0
        out[ok] = prefix[pos[ok]]
        return out

    def _type_continuations(
        self,
        spec,
        pickup_node,
        dropoff_node,
        pickup_to_dropoff_m: float,
    ) -> tuple[_TypeGraph, np.ndarray, np.ndarray, np.ndarray, float]:
        graph = self._type_graph(spec)
        full_range = graph.full_range_m
        pickup_col = self.node_index[pickup_node]
        dropoff_col = self.node_index[dropoff_node]
        pickup_station = np.asarray(
            self.station_to_node[:, pickup_col],
            dtype=np.float64,
        )
        dropoff_station = np.asarray(
            self.station_to_node[:, dropoff_col],
            dtype=np.float64,
        )
        reserve_distance = float(
            self.router.graph.nodes[dropoff_node][
                DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR
            ]
        )

        # Post-pickup: from each charger, find shortest feasible charger chain
        # ending at dropoff while keeping enough reserve to reach a charger.
        finish_terminal = np.where(
            np.isfinite(dropoff_station)
            & (dropoff_station + reserve_distance <= full_range + DISTANCE_EPS_M),
            dropoff_station,
            np.inf,
        )
        post_cost = self._sink_distances(graph, finish_terminal)

        # Pre-pickup: at charger i, either go pickup->dropoff directly or go
        # pickup->some post-pickup charger j, then continue with post_cost[j].
        direct_terminal = np.where(
            np.isfinite(pickup_station)
            & (
                pickup_station
                + float(pickup_to_dropoff_m)
                + reserve_distance
                <= full_range + DISTANCE_EPS_M
            ),
            pickup_station + float(pickup_to_dropoff_m),
            np.inf,
        )
        thresholds = full_range - pickup_station
        via_post_tail = self._prefix_min_for_thresholds(
            pickup_station,
            post_cost,
            thresholds,
        )
        via_post_terminal = pickup_station + via_post_tail
        pickup_terminal = np.minimum(direct_terminal, via_post_terminal)
        pre_cost = self._sink_distances(graph, pickup_terminal)

        return graph, pickup_station, post_cost, pre_cost, reserve_distance

    def score(
        self,
        candidates: Iterable[object],
        *,
        pickup_node,
        dropoff_node,
        pickup_to_dropoff_m: float,
        now_min: float,
        handling_min: float,
    ) -> list[FastCandidateScore]:
        started = time.perf_counter()
        candidate_list = list(candidates)
        FAST_META_STATS["calls"] += 1.0
        FAST_META_STATS["candidates"] += float(len(candidate_list))
        if not candidate_list:
            return []

        grouped: dict[tuple[float, float, float], list[object]] = {}
        spec_by_key: dict[tuple[float, float, float], object] = {}
        for candidate in candidate_list:
            spec = candidate.robot.spec
            key = self._spec_key(spec)
            grouped.setdefault(key, []).append(candidate)
            spec_by_key[key] = spec

        scored: list[FastCandidateScore] = []
        for key, group in grouped.items():
            spec = spec_by_key[key]
            FAST_META_STATS["type_solves"] += 1.0
            (
                graph,
                pickup_station,
                post_cost,
                pre_cost,
                reserve_distance,
            ) = self._type_continuations(
                spec,
                pickup_node,
                dropoff_node,
                pickup_to_dropoff_m,
            )

            k = len(group)
            node_indices = np.asarray(
                [self.node_index[c.route_start_node] for c in group],
                dtype=np.int64,
            )
            start_station = np.asarray(
                self.station_to_node[:, node_indices],
                dtype=np.float64,
            )
            batteries = np.asarray(
                [float(c.route_start_battery_wh) for c in group],
                dtype=np.float64,
            )
            start_to_pickup = np.asarray(
                [float(c.decision_to_pickup_m) for c in group],
                dtype=np.float64,
            )
            start_delays = np.asarray(
                [max(0.0, float(c.route_start_time_min) - float(now_min)) for c in group],
                dtype=np.float64,
            )
            is_station = np.asarray(
                [c.route_start_node in self.station_set for c in group],
                dtype=bool,
            )
            current_range = batteries / float(spec.energy_per_meter_wh)
            available_range = np.where(is_station, graph.full_range_m, current_range)

            # Original router rejects a non-stationary start that is already
            # below the reserve required to reach its nearest charger.
            nearest = np.asarray(
                [
                    float(
                        self.router.graph.nodes[c.route_start_node][
                            DISTANCE_TO_NEAREST_CHARGING_STATION_M_ATTR
                        ]
                    )
                    for c in group
                ],
                dtype=np.float64,
            )
            safe_start = is_station | (
                current_range + DISTANCE_EPS_M >= nearest
            )

            # Option 1: reach a charger before pickup, then use pre_cost.
            pre_matrix = start_station + pre_cost[:, None]
            pre_matrix[
                start_station > available_range[None, :] + DISTANCE_EPS_M
            ] = np.inf
            via_pre = np.min(pre_matrix, axis=0)

            # Option 2: reach pickup first and then a post-pickup charger.
            thresholds = available_range - start_to_pickup
            pickup_tail = self._prefix_min_for_thresholds(
                pickup_station,
                post_cost,
                thresholds,
            )
            via_pickup_charger = start_to_pickup + pickup_tail

            # Option 3: pickup and dropoff in one segment.
            direct_distance = start_to_pickup + float(pickup_to_dropoff_m)
            via_direct = np.where(
                direct_distance + reserve_distance
                <= available_range + DISTANCE_EPS_M,
                direct_distance,
                np.inf,
            )

            best_distance = np.minimum(via_pre, np.minimum(via_pickup_charger, via_direct))
            best_distance[~safe_start] = np.inf

            reserve_wh = reserve_distance * float(spec.energy_per_meter_wh)
            energy_needed = (
                best_distance * float(spec.energy_per_meter_wh) + reserve_wh
            )
            added_wh = np.maximum(0.0, energy_needed - batteries)
            route_time = (
                best_distance / float(spec.speed_mps) / 60.0
                + added_wh / DEFAULT_CHARGING_POWER_W * 60.0
            )
            completion = start_delays + route_time + float(handling_min)

            for index in range(k):
                candidate = group[index]
                scored.append(
                    FastCandidateScore(
                        completion_time_min=float(completion[index]),
                        route_time_min=float(route_time[index]),
                        route_distance_m=float(best_distance[index]),
                        robot_id=int(candidate.robot.spec.id),
                        candidate=candidate,
                    )
                )

        scored.sort(key=lambda row: (row.completion_time_min, row.robot_id))
        FAST_META_STATS["seconds"] += time.perf_counter() - started
        return scored


_SCORERS: dict[int, FastBatteryMetaScorer] = {}


def rank_candidates(
    router,
    candidates: Iterable[object],
    *,
    pickup_node,
    dropoff_node,
    pickup_to_dropoff_m: float,
    now_min: float,
    handling_min: float,
) -> list[FastCandidateScore]:
    scorer = _SCORERS.get(id(router))
    if scorer is None:
        scorer = FastBatteryMetaScorer(router)
        _SCORERS[id(router)] = scorer
    return scorer.score(
        candidates,
        pickup_node=pickup_node,
        dropoff_node=dropoff_node,
        pickup_to_dropoff_m=pickup_to_dropoff_m,
        now_min=now_min,
        handling_min=handling_min,
    )
