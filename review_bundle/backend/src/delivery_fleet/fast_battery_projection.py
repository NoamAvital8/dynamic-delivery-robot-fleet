"""Fast numeric battery-feasible projections for insertion search.

The exact B5 insertion search evaluates many candidate stop sequences.  The
original battery router solves a small Python Dijkstra over the charging-station
meta graph for every stop of every candidate.  With hundreds of NYC charging
stations this dominates runtime.

This module exploits two facts that are specific to *candidate scoring*:

1. Charging stations and the street graph are fixed for the whole simulation.
2. The existing battery router minimizes physical route distance.  Given the
   shortest feasible distance and the next mandatory node, travel time,
   required charging energy, and battery at the first mandatory node are
   determined numerically; the actual street/station path is only needed for
   the finally selected candidate.

For each distinct robot full-battery range we therefore precompute all-pairs
shortest distances on the charging-station graph in compiled SciPy code.  A
start -> first mandatory stop -> second mandatory stop projection then reduces
to vectorized min-plus operations over a few hundred stations rather than a
fresh Python graph search.

The final winning sequence should still be materialized with the normal exact
battery router so execution uses the original path/tie-breaking semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import numpy as np

from .battery_routing import BatteryFeasibleRouter, NoFeasibleBatteryRoute
from .charging import DEFAULT_CHARGING_POWER_W
from .robot import RobotState

NodeId = Hashable
_DISTANCE_EPS_M = 1e-3


@dataclass(frozen=True, slots=True)
class NumericPrefix:
    """Numeric state on arrival at the first mandatory stop."""

    total_time_min: float
    total_distance_m: float
    arrival_battery_wh: float


class FastBatteryProjectionIndex:
    """Compiled/vectorized numeric projection backend for an undirected graph."""

    def __init__(self, router: BatteryFeasibleRouter) -> None:
        self.router = router
        index = router.charger_index
        if router.graph.is_directed() or not index.is_dense:
            raise ValueError(
                "FastBatteryProjectionIndex requires an undirected dense charger index"
            )

        distances = getattr(index, "_distances", None)
        node_index = getattr(index, "_node_index", None)
        if distances is None or node_index is None:
            raise ValueError("dense charger internals are unavailable")

        self.station_nodes = tuple(index.station_nodes)
        self.station_index = {
            node: position for position, node in enumerate(self.station_nodes)
        }
        self._node_index = node_index
        station_columns = np.asarray(
            [node_index[node] for node in self.station_nodes], dtype=np.int64
        )
        self._station_to_station = np.asarray(
            distances[:, station_columns], dtype=np.float64
        )
        self._distance_matrix = distances

        self._closure_by_range: dict[float, np.ndarray] = {}
        self._post_finish_cache: dict[tuple[float, NodeId], np.ndarray] = {}

    def _node_station_distances(self, node: NodeId) -> np.ndarray:
        """Exact (charger-index precision) distances from node to all stations."""
        column = int(self._node_index[node])
        # NYC/Tel-Aviv graphs are undirected, so the station->node dense column
        # is also node->station distance.
        return np.asarray(self._distance_matrix[:, column], dtype=np.float64)

    def _closure(self, full_range_m: float) -> np.ndarray:
        key = float(full_range_m)
        cached = self._closure_by_range.get(key)
        if cached is not None:
            return cached

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

        direct = self._station_to_station
        mask = (
            np.isfinite(direct)
            & (direct <= key + _DISTANCE_EPS_M)
            & (~np.eye(direct.shape[0], dtype=bool))
        )
        rows, cols = np.nonzero(mask)
        values = direct[rows, cols]
        graph = csr_matrix(
            (values, (rows, cols)),
            shape=direct.shape,
            dtype=np.float64,
        )
        closure = scipy_dijkstra(
            graph,
            directed=False,
            indices=np.arange(direct.shape[0], dtype=np.int64),
            return_predecessors=False,
        )
        closure = np.asarray(closure, dtype=np.float64)
        np.fill_diagonal(closure, 0.0)
        self._closure_by_range[key] = closure
        return closure

    @staticmethod
    def _sorted_prefix_min(
        station_distance: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sort by station_distance and build cumulative minima + argmins."""
        order = np.argsort(station_distance, kind="stable")
        d = station_distance[order]
        v = values[order]
        prefix_values = np.empty_like(v)
        prefix_args = np.empty(order.shape, dtype=np.int64)
        best_value = math.inf
        best_index = -1
        for position, value in enumerate(v):
            value = float(value)
            if value < best_value:
                best_value = value
                best_index = int(order[position])
            prefix_values[position] = best_value
            prefix_args[position] = best_index
        return d, prefix_values, prefix_args

    @staticmethod
    def _query_prefix_min(
        sorted_distance: np.ndarray,
        prefix_values: np.ndarray,
        prefix_args: np.ndarray,
        threshold: float,
    ) -> tuple[float, int]:
        pos = int(np.searchsorted(sorted_distance, threshold, side="right")) - 1
        if pos < 0:
            return math.inf, -1
        return float(prefix_values[pos]), int(prefix_args[pos])

    def _pre_costs(
        self,
        robot: RobotState,
        closure: np.ndarray,
    ) -> np.ndarray:
        """Shortest feasible distance from start to every charger state."""
        start = robot.node_id
        if start in self.station_index:
            return np.asarray(closure[self.station_index[start]], dtype=np.float64)

        to_stations = self._node_station_distances(start)
        current_range = float(robot.remaining_range_m)
        feasible = np.isfinite(to_stations) & (
            to_stations <= current_range + _DISTANCE_EPS_M
        )
        if not np.any(feasible):
            return np.full(len(self.station_nodes), np.inf, dtype=np.float64)
        # First reach one charger on current battery, then use the precomputed
        # full-battery charger-network closure.
        return np.min(
            to_stations[feasible, None] + closure[feasible, :],
            axis=0,
        )

    def _post_finish(
        self,
        full_range_m: float,
        target_node: NodeId,
        closure: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Distance from each charger state to target with final reserve."""
        cache_key = (float(full_range_m), target_node)
        cached = self._post_finish_cache.get(cache_key)
        target_to_stations = self._node_station_distances(target_node)
        reserve_distance = float(np.min(target_to_stations))
        if cached is not None:
            return cached, reserve_distance, target_to_stations

        final_feasible = np.isfinite(target_to_stations) & (
            target_to_stations + reserve_distance
            <= float(full_range_m) + _DISTANCE_EPS_M
        )
        if not np.any(final_feasible):
            finish = np.full(len(self.station_nodes), np.inf, dtype=np.float64)
        else:
            finish = np.min(
                closure[:, final_feasible]
                + target_to_stations[final_feasible][None, :],
                axis=1,
            )
        self._post_finish_cache[cache_key] = finish
        return finish, reserve_distance, target_to_stations

    @staticmethod
    def _numeric_prefix(
        robot: RobotState,
        *,
        prefix_distance_m: float,
        continuation_after_first_m: float,
        terminal_reserve_m: float,
    ) -> NumericPrefix:
        """Convert a selected feasible meta-route structure into first-stop state."""
        rate = float(robot.spec.energy_per_meter_wh)
        initial = float(robot.battery_wh)
        required_until_next_charge_or_finish_wh = (
            prefix_distance_m
            + continuation_after_first_m
            + terminal_reserve_m
        ) * rate
        energy_added_before_first = max(
            0.0,
            required_until_next_charge_or_finish_wh - initial,
        )
        battery_at_first = (
            initial
            + energy_added_before_first
            - prefix_distance_m * rate
        )
        battery_at_first = min(
            float(robot.spec.battery_capacity_wh),
            max(0.0, float(battery_at_first)),
        )
        travel_time = prefix_distance_m / float(robot.spec.speed_mps) / 60.0
        charging_time = (
            energy_added_before_first / DEFAULT_CHARGING_POWER_W * 60.0
        )
        return NumericPrefix(
            total_time_min=float(travel_time + charging_time),
            total_distance_m=float(prefix_distance_m),
            arrival_battery_wh=float(battery_at_first),
        )

    def project_pair_prefix(
        self,
        robot: RobotState,
        first_node: NodeId,
        second_node: NodeId,
        *,
        start_to_first_m: float,
        first_to_second_m: float,
    ) -> NumericPrefix:
        """Project arrival at first_node for the shortest feasible route via second_node.

        Numerically matches the distance-minimizing layered charger meta graph
        used by ``evaluate_pair_from_committed_state`` but does not reconstruct
        the actual charger/street path.
        """
        if first_node == second_node:
            raise ValueError("first and second mandatory nodes must differ")

        spec = robot.spec
        full_range = float(spec.full_battery_range_m)
        current_range = (
            full_range
            if robot.node_id in self.station_index
            else float(robot.remaining_range_m)
        )
        closure = self._closure(full_range)
        pre_cost = self._pre_costs(robot, closure)
        d_first = self._node_station_distances(first_node)
        post_finish, reserve, _ = self._post_finish(
            full_range, second_node, closure
        )

        best_total = math.inf
        best_prefix = math.inf
        best_tail = 0.0
        best_reserve = 0.0

        def consider(total: float, prefix: float, tail: float, terminal_reserve: float) -> None:
            nonlocal best_total, best_prefix, best_tail, best_reserve
            if total + 1e-9 < best_total:
                best_total = float(total)
                best_prefix = float(prefix)
                best_tail = float(tail)
                best_reserve = float(terminal_reserve)

        start_to_first_m = float(start_to_first_m)
        first_to_second_m = float(first_to_second_m)

        if first_node in self.station_index:
            first_index = self.station_index[first_node]
            distance_to_first = float(pre_cost[first_index])
            if math.isfinite(distance_to_first) and math.isfinite(post_finish[first_index]):
                consider(
                    distance_to_first + float(post_finish[first_index]),
                    distance_to_first,
                    0.0,
                    0.0,
                )
        else:
            # Direct start -> first -> second, with no charger after first.
            if (
                start_to_first_m + first_to_second_m + reserve
                <= current_range + _DISTANCE_EPS_M
            ):
                consider(
                    start_to_first_m + first_to_second_m,
                    start_to_first_m,
                    first_to_second_m,
                    reserve,
                )

            # Direct start -> first -> post charger j.
            direct_post_feasible = (
                np.isfinite(d_first)
                & np.isfinite(post_finish)
                & (
                    start_to_first_m + d_first
                    <= current_range + _DISTANCE_EPS_M
                )
            )
            if np.any(direct_post_feasible):
                totals = (
                    start_to_first_m
                    + d_first[direct_post_feasible]
                    + post_finish[direct_post_feasible]
                )
                local = int(np.argmin(totals))
                js = np.flatnonzero(direct_post_feasible)
                j = int(js[local])
                consider(
                    float(totals[local]),
                    start_to_first_m,
                    float(d_first[j]),
                    0.0,
                )

            # Paths reaching one or more chargers before first_node.  A_i is
            # shortest distance from start to charger i and then to first_node.
            a = pre_cost + d_first
            sorted_d, prefix_min, prefix_arg = self._sorted_prefix_min(d_first, a)

            # pre charger i -> first -> second directly.
            threshold_done = full_range - first_to_second_m - reserve
            best_a, _ = self._query_prefix_min(
                sorted_d, prefix_min, prefix_arg, threshold_done
            )
            if math.isfinite(best_a):
                consider(
                    best_a + first_to_second_m,
                    best_a,
                    first_to_second_m,
                    reserve,
                )

            # pre charger i -> first -> post charger j, then charger closure -> second.
            thresholds = full_range - d_first
            positions = np.searchsorted(sorted_d, thresholds, side="right") - 1
            valid_j = (
                (positions >= 0)
                & np.isfinite(d_first)
                & np.isfinite(post_finish)
            )
            if np.any(valid_j):
                js = np.flatnonzero(valid_j)
                positions_valid = positions[js].astype(np.int64)
                best_a_for_j = prefix_min[positions_valid]
                totals = best_a_for_j + d_first[js] + post_finish[js]
                local = int(np.argmin(totals))
                j = int(js[local])
                i = int(prefix_arg[int(positions[j])])
                # Prefix distance ends at first, while the charge at i must also
                # cover first -> j before another charge is possible.
                prefix_distance = float(pre_cost[i] + d_first[i])
                consider(
                    float(totals[local]),
                    prefix_distance,
                    float(d_first[j]),
                    0.0,
                )

        if not math.isfinite(best_total):
            raise NoFeasibleBatteryRoute(
                "no battery-feasible numeric pair projection exists"
            )
        return self._numeric_prefix(
            robot,
            prefix_distance_m=best_prefix,
            continuation_after_first_m=best_tail,
            terminal_reserve_m=best_reserve,
        )

    def project_final_prefix(
        self,
        robot: RobotState,
        target_node: NodeId,
        *,
        start_to_target_m: float,
    ) -> NumericPrefix:
        """Project shortest feasible start -> final target, including target reserve."""
        spec = robot.spec
        full_range = float(spec.full_battery_range_m)
        current_range = (
            full_range
            if robot.node_id in self.station_index
            else float(robot.remaining_range_m)
        )
        closure = self._closure(full_range)
        to_target_stations = self._node_station_distances(target_node)
        reserve = 0.0 if target_node in self.station_index else float(np.min(to_target_stations))

        start_to_target_m = float(start_to_target_m)
        best_distance = math.inf
        if start_to_target_m + reserve <= current_range + _DISTANCE_EPS_M:
            best_distance = start_to_target_m

        pre_cost = self._pre_costs(robot, closure)
        final_feasible = np.isfinite(to_target_stations) & (
            to_target_stations + reserve <= full_range + _DISTANCE_EPS_M
        )
        if np.any(final_feasible):
            via = pre_cost[final_feasible] + to_target_stations[final_feasible]
            candidate = float(np.min(via))
            if candidate < best_distance:
                best_distance = candidate

        if not math.isfinite(best_distance):
            raise NoFeasibleBatteryRoute(
                "no battery-feasible numeric route to final target"
            )

        rate = float(spec.energy_per_meter_wh)
        initial = float(robot.battery_wh)
        # Any route longer than the current non-station range necessarily reaches
        # a charger before the target; a station start can charge immediately.
        can_charge_before_target = (
            robot.node_id in self.station_index
            or best_distance > float(robot.remaining_range_m) + _DISTANCE_EPS_M
        )
        required = (best_distance + reserve) * rate
        if can_charge_before_target:
            added = max(0.0, required - initial)
        else:
            added = 0.0
            if required > initial + 1e-3:
                raise NoFeasibleBatteryRoute(
                    "numeric final route lacks a charging opportunity"
                )
        battery = initial + added - best_distance * rate
        travel_time = best_distance / float(spec.speed_mps) / 60.0
        charging_time = added / DEFAULT_CHARGING_POWER_W * 60.0
        return NumericPrefix(
            total_time_min=float(travel_time + charging_time),
            total_distance_m=float(best_distance),
            arrival_battery_wh=float(max(0.0, battery)),
        )
