"""Bounded, coordinated predictive control for robots without delivery work.

The dispatcher remains responsible for real orders.  This controller uses only
the online Gamma-Poisson posterior to estimate the marginal future value of
STAY, REPOSITION and CHARGE actions.  Its concave coverage value gives a
second robot in a well-covered region less credit than the first.  A short
candidate list and optional process-parallel vector evaluation bound runtime.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import math
import multiprocessing
import os
from typing import Hashable, Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np

from .spatial_demand import GammaPoissonDemandModel, summarize_existing_clusters


@dataclass(frozen=True, slots=True)
class IdlePlanningConfig:
    horizon_min: float = 45.0
    max_demand_clusters: int = 32
    candidate_clusters: int = 5
    candidate_chargers: int = 3
    response_scale_min: float = 18.0
    minimum_gain_fraction: float = 0.001
    relocation_uncertainty_penalty: float = 0.0
    max_actions_per_epoch: int = 32
    processes: int = 1

    def __post_init__(self) -> None:
        if self.horizon_min <= 0 or self.response_scale_min <= 0:
            raise ValueError("planning horizon and response scale must be positive")
        if min(self.max_demand_clusters, self.candidate_clusters,
               self.candidate_chargers, self.max_actions_per_epoch,
               self.processes) <= 0:
            raise ValueError("candidate limits and processes must be positive")
        if self.minimum_gain_fraction < 0:
            raise ValueError("minimum gain fraction must be non-negative")
        if (not math.isfinite(self.relocation_uncertainty_penalty)
                or self.relocation_uncertainty_penalty < 0):
            raise ValueError("relocation uncertainty penalty must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class IdleRobotSnapshot:
    robot_id: int
    node_id: Hashable
    speed_mps: float
    energy_per_meter_wh: float
    battery_wh: float
    battery_capacity_wh: float
    max_payload_kg: float
    max_volume_l: float
    available_in_min: float = 0.0
    min_importance: float = 0.0


@dataclass(frozen=True, slots=True)
class ChargeSlot:
    robot_id: int
    station_node: Hashable
    port: int
    start_min: float
    finish_min: float


class ChargingCalendar:
    """Non-overlapping, half-open charging intervals on physical ports."""

    def __init__(self, ports_by_station: Mapping[Hashable, int]) -> None:
        self._ports: dict[Hashable, list[list[ChargeSlot]]] = {}
        for station, count in ports_by_station.items():
            if int(count) <= 0:
                raise ValueError("every station needs at least one port")
            self._ports[station] = [[] for _ in range(int(count))]

    def earliest(
        self, robot_id: int, station_node: Hashable,
        arrival_min: float, duration_min: float,
    ) -> ChargeSlot:
        if duration_min <= 0 or arrival_min < 0:
            raise ValueError("charging duration must be positive and arrival non-negative")
        best: ChargeSlot | None = None
        for port, intervals in enumerate(self._ports[station_node]):
            start = float(arrival_min)
            for interval in intervals:
                if start + duration_min <= interval.start_min + 1e-9:
                    break
                start = max(start, interval.finish_min)
            proposed = ChargeSlot(
                int(robot_id), station_node, port, start, start + duration_min
            )
            if best is None or (proposed.finish_min, port) < (best.finish_min, best.port):
                best = proposed
        assert best is not None
        return best

    def reserve(
        self, robot_id: int, station_node: Hashable,
        arrival_min: float, duration_min: float,
    ) -> ChargeSlot:
        slot = self.earliest(robot_id, station_node, arrival_min, duration_min)
        intervals = self._ports[station_node][slot.port]
        intervals.append(slot)
        intervals.sort(key=lambda item: item.start_min)
        return slot

    def slots(self, station_node: Hashable) -> tuple[ChargeSlot, ...]:
        return tuple(
            slot for port in self._ports[station_node] for slot in port
        )


@dataclass(frozen=True, slots=True)
class IdleAction:
    robot_id: int
    kind: Literal["stay", "reposition", "charge"]
    target_node: Hashable
    travel_distance_m: float
    arrival_min: float
    ready_min: float
    battery_after_wh: float
    slot: ChargeSlot | None = None


def _coverage_vector(
    robot: IdleRobotSnapshot,
    action: IdleAction,
    cell_lat_rad: np.ndarray,
    cell_lon_rad: np.ndarray,
    cell_importance: np.ndarray,
    target_lat_rad: float,
    target_lon_rad: float,
    response_scale_min: float,
    horizon_min: float,
) -> np.ndarray:
    """Average response coverage for a uniform future arrival time in [0,H]."""

    dlat = cell_lat_rad - target_lat_rad
    dlon = cell_lon_rad - target_lon_rad
    h = np.sin(dlat / 2.0) ** 2 + (
        np.cos(target_lat_rad) * np.cos(cell_lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    distance_m = 2.0 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    response_scale = response_scale_min / np.sqrt(cell_importance)
    ready = max(0.0, action.ready_min)
    if ready <= horizon_min:
        availability = (
            response_scale * (-np.expm1(-ready / response_scale))
            + horizon_min - ready
        ) / horizon_min
    else:
        availability = (
            response_scale / horizon_min
            * np.exp(-(ready - horizon_min) / response_scale)
            * (-np.expm1(-horizon_min / response_scale))
        )
    battery_at_pickup = action.battery_after_wh - (
        distance_m * robot.energy_per_meter_wh
    )
    battery_factor = np.clip(
        battery_at_pickup / (0.65 * robot.battery_capacity_wh), 0.0, 1.0
    )
    capability_factor = min(
        1.0,
        0.5 * robot.max_payload_kg / 12.0 + 0.5 * robot.max_volume_l / 30.0,
    )
    travel_response = distance_m / robot.speed_mps / 60.0
    return (
        capability_factor * battery_factor * availability
        * np.exp(-travel_response / response_scale)
        * (cell_importance >= robot.min_importance)
    )


def _coverage_batch(args: tuple[Sequence[tuple], np.ndarray, np.ndarray, np.ndarray,
                                float, float]) -> list[np.ndarray]:
    jobs, lat, lon, importance, response_scale, horizon = args
    return [
        _coverage_vector(robot, action, lat, lon, importance,
                         target_lat, target_lon, response_scale, horizon)
        for robot, action, target_lat, target_lon in jobs
    ]


class IdleFleetPlanner:
    """Plan marginally useful idle moves using current posterior demand only."""

    def __init__(
        self,
        graph: nx.Graph,
        charger_index: object,
        demand: GammaPoissonDemandModel,
        station_nodes: Iterable[Hashable],
        charger_power_w: float,
        config: IdlePlanningConfig = IdlePlanningConfig(),
    ) -> None:
        if charger_power_w <= 0:
            raise ValueError("charger power must be positive")
        self.graph = graph
        self.charger_index = charger_index
        self.demand = demand
        self.station_nodes = tuple(station_nodes)
        self.charger_power_w = float(charger_power_w)
        self.config = config
        self.representatives = summarize_existing_clusters(graph).representatives
        self._reserve_distance_m = {
            node: min(charger_index.distances_to_stations(node).values())
            for node in self.representatives.values()
        }
        self._coordinates = {
            node: (math.radians(float(data["y"])), math.radians(float(data["x"])))
            for node, data in graph.nodes(data=True)
        }
        self._pool: ProcessPoolExecutor | None = None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self) -> "IdleFleetPlanner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _vectors(
        self, jobs: list[tuple], lat: np.ndarray, lon: np.ndarray,
        importance: np.ndarray,
    ) -> list[np.ndarray]:
        workers = min(self.config.processes, os.cpu_count() or self.config.processes)
        if workers == 1 or len(jobs) < 64:
            return _coverage_batch(
                (jobs, lat, lon, importance, self.config.response_scale_min,
                 self.config.horizon_min)
            )
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        chunk_size = max(1, math.ceil(len(jobs) / (workers * 2)))
        chunks = [
            (jobs[i:i + chunk_size], lat, lon, importance,
             self.config.response_scale_min, self.config.horizon_min)
            for i in range(0, len(jobs), chunk_size)
        ]
        return [vector for batch in self._pool.map(_coverage_batch, chunks)
                for vector in batch]

    def _demand_cells(self, now_min: float) -> tuple[list[int], dict[int, float],
                                                      np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray,
                                                      np.ndarray]:
        values: list[tuple[float, int, float, float]] = []
        cluster_value: dict[int, float] = {}
        for cluster in self.representatives:
            total = 0.0
            for importance in self.demand.importance_levels:
                posterior = self.demand.posterior(
                    cluster, importance, at_time_min=now_min
                )
                # The late-loss slope is the relevant urgency scale.
                scale = self.config.horizon_min * (importance + 1.0) ** 2
                value = posterior.mean_per_minute * scale
                values.append((value, cluster, importance, posterior.std_per_minute * scale))
                total += value
            cluster_value[cluster] = total
        ranked_clusters = sorted(
            self.representatives,
            key=lambda cluster: (-cluster_value[cluster], cluster),
        )[:self.config.max_demand_clusters]
        selected = [(v, z, c, s) for v, z, c, s in values if z in ranked_clusters]
        lat = np.asarray(
            [self._coordinates[self.representatives[z]][0] for _, z, _, _ in selected]
        )
        lon = np.asarray(
            [self._coordinates[self.representatives[z]][1] for _, z, _, _ in selected]
        )
        importance = np.asarray([c for _, _, c, _ in selected])
        weight = np.asarray([v for v, _, _, _ in selected])
        weight_std = np.asarray([s for _, _, _, s in selected])
        return ranked_clusters, cluster_value, lat, lon, importance, weight, weight_std

    def _candidate_actions(
        self,
        robot: IdleRobotSnapshot,
        ranked_clusters: Sequence[int],
        cluster_value: Mapping[int, float],
        now_min: float,
        calendar: ChargingCalendar,
    ) -> list[IdleAction]:
        node = robot.node_id
        actions: list[IdleAction] = []
        lat0, lon0 = self._coordinates[node]
        scored_clusters = []
        for cluster in ranked_clusters:
            lat, lon = self._coordinates[self.representatives[cluster]]
            h = math.sin((lat - lat0) / 2.0) ** 2 + (
                math.cos(lat0) * math.cos(lat) * math.sin((lon - lon0) / 2.0) ** 2
            )
            distance = 2.0 * 6_371_000.0 * math.asin(math.sqrt(min(1.0, h)))
            travel_min = distance / robot.speed_mps / 60.0
            score = cluster_value[cluster] / (
                1.0 + travel_min / self.config.response_scale_min
            )
            scored_clusters.append((-score, cluster))
        for _, cluster in sorted(scored_clusters)[:self.config.candidate_clusters]:
            target = self.representatives[cluster]
            if target == node:
                continue
            # ChargerDistanceIndex.distance accepts a station as its first
            # argument.  Cluster representatives need the general oracle.
            distance = float(self.charger_index.oracle.distance(node, target))
            remaining = robot.battery_wh - distance * robot.energy_per_meter_wh
            if remaining < 0:
                continue
            reserve_m = self._reserve_distance_m[target]
            if remaining + 1e-6 < reserve_m * robot.energy_per_meter_wh:
                continue
            arrival = now_min + distance / robot.speed_mps / 60.0
            actions.append(IdleAction(
                robot.robot_id, "reposition", target, distance, arrival,
                arrival - now_min, remaining,
            ))

        distances = self.charger_index.distances_to_stations(node)
        nearest = sorted(
            ((float(distance), station) for station, distance in distances.items()
             if station in self.station_nodes),
            key=lambda item: (item[0], str(item[1])),
        )[:max(12, self.config.candidate_chargers * 4)]
        charge_options: list[IdleAction] = []
        for distance, station in nearest:
            battery_arrival = robot.battery_wh - distance * robot.energy_per_meter_wh
            if battery_arrival < -1e-6:
                continue
            energy_needed = robot.battery_capacity_wh - max(0.0, battery_arrival)
            if energy_needed <= 1e-8:
                continue
            arrival = now_min + distance / robot.speed_mps / 60.0
            slot = calendar.earliest(
                robot.robot_id, station, arrival,
                energy_needed / self.charger_power_w * 60.0,
            )
            charge_options.append(IdleAction(
                robot.robot_id, "charge", station, distance, arrival,
                slot.finish_min - now_min, robot.battery_capacity_wh, slot,
            ))
        if charge_options:
            selected = [charge_options[0]]  # Always compare the closest station.
            for option in sorted(
                charge_options[1:],
                key=lambda action: (
                    action.slot.finish_min, action.travel_distance_m,
                    str(action.target_node),
                ),
            ):
                if len(selected) >= self.config.candidate_chargers:
                    break
                selected.append(option)
            actions.extend(selected)
        return actions

    def plan(
        self,
        now_min: float,
        robots: Sequence[IdleRobotSnapshot],
        candidate_ids: Iterable[int],
        calendar: ChargingCalendar,
    ) -> dict[int, IdleAction]:
        """Choose gains above the threshold, optionally penalizing uncertain moves."""

        if not robots:
            return {}
        candidates = set(candidate_ids)
        ranked, cluster_value, lat, lon, importance, weight, weight_std = self._demand_cells(now_min)
        by_id = {robot.robot_id: robot for robot in robots}
        jobs: list[tuple] = []
        options: dict[int, list[IdleAction]] = {}
        for robot in robots:
            stay = IdleAction(
                robot.robot_id, "stay", robot.node_id, 0.0, now_min,
                max(0.0, robot.available_in_min), robot.battery_wh,
            )
            target_lat, target_lon = self._coordinates[robot.node_id]
            jobs.append((robot, stay, target_lat, target_lon))
            if robot.robot_id in candidates:
                options[robot.robot_id] = self._candidate_actions(
                    robot, ranked, cluster_value, now_min, calendar
                )
                for action in options[robot.robot_id]:
                    target_lat, target_lon = self._coordinates[action.target_node]
                    jobs.append((robot, action, target_lat, target_lon))

        vectors = self._vectors(jobs, lat, lon, importance)
        vector_by_action = {
            (action.robot_id, action.kind, action.target_node): vector
            for (_, action, _, _), vector in zip(jobs, vectors, strict=True)
        }
        coverage = sum(
            (vector_by_action[(robot.robot_id, "stay", robot.node_id)]
             for robot in robots),
            np.zeros_like(weight),
        )
        flat_actions = [
            action for robot_id in sorted(options) for action in options[robot_id]
        ]
        if not flat_actions:
            return {}
        baseline_by_robot = {
            robot.robot_id: vector_by_action[(robot.robot_id, "stay", robot.node_id)]
            for robot in robots
        }
        deltas = np.stack([
            vector_by_action[(action.robot_id, action.kind, action.target_node)]
            - baseline_by_robot[action.robot_id]
            for action in flat_actions
        ])
        gain_factors = -np.expm1(-deltas)
        active = np.ones(len(flat_actions), dtype=bool)
        option_robot_ids = np.asarray(
            [action.robot_id for action in flat_actions], dtype=int
        )
        relocation_mask = np.asarray([action.kind == "reposition" for action in flat_actions])
        selected: dict[int, IdleAction] = {}
        minimum_gain = self.config.minimum_gain_fraction * float(sum(weight))
        calendar_changed = False

        while active.any() and len(selected) < self.config.max_actions_per_epoch:
            if calendar_changed:
                for index, original in enumerate(flat_actions):
                    if not active[index] or original.kind != "charge":
                        continue
                    assert original.slot is not None
                    refreshed = calendar.earliest(
                        original.robot_id, original.target_node,
                        original.arrival_min,
                        original.slot.finish_min - original.slot.start_min,
                    )
                    if refreshed == original.slot:
                        continue
                    action = replace(
                        original, slot=refreshed,
                        ready_min=refreshed.finish_min - now_min,
                    )
                    flat_actions[index] = action
                    robot = by_id[action.robot_id]
                    target_lat, target_lon = self._coordinates[action.target_node]
                    vector = _coverage_vector(
                        robot, action, lat, lon, importance,
                        target_lat, target_lon,
                        self.config.response_scale_min,
                        self.config.horizon_min,
                    )
                    deltas[index] = vector - baseline_by_robot[action.robot_id]
                    gain_factors[index] = -np.expm1(-deltas[index])
                calendar_changed = False

            gains = gain_factors @ (weight * np.exp(-coverage))
            if self.config.relocation_uncertainty_penalty > 0:
                # Conditional on fixed response vectors, the marginal utility
                # is linear in independent Gamma rates. Its variance includes
                # both uncertain destination benefit and coverage lost by leaving.
                gain_std = np.sqrt(
                    (gain_factors[relocation_mask] ** 2)
                    @ ((weight_std * np.exp(-coverage)) ** 2)
                )
                gains[relocation_mask] -= self.config.relocation_uncertainty_penalty * gain_std
            gains[~active] = -math.inf
            best_index = int(np.argmax(gains))
            if gains[best_index] <= minimum_gain + 1e-12:
                break
            best_action = flat_actions[best_index]
            robot_id = best_action.robot_id
            if best_action.kind == "charge":
                assert best_action.slot is not None
                slot = calendar.reserve(
                    robot_id, best_action.target_node,
                    best_action.arrival_min,
                    best_action.slot.finish_min - best_action.slot.start_min,
                )
                best_action = replace(best_action, slot=slot)
                calendar_changed = True
            selected[robot_id] = best_action
            coverage += deltas[best_index]
            active[option_robot_ids == robot_id] = False
        return selected
