"""Online Gamma-Poisson -> FCNN -> concrete robot reservation pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import networkx as nx

from .fleet import RobotType
from .reservation import (
    ReservationAssignment,
    apportion_reservation_counts,
    assign_reservation_thresholds,
)
from .reservation_nn import (
    ReservationFCNN,
    ReservationFeatureSchema,
    build_reservation_features,
)
from .robot import RobotState
from .spatial_demand import (
    GammaPoissonDemandModel,
    haversine_node_distance_m,
    summarize_existing_clusters,
    weighted_reservation_score,
)


@dataclass(frozen=True, slots=True)
class ReservationRobotSnapshot:
    """Cheap response state used when choosing concrete reserved robots."""

    node_id: Hashable
    available_in_min: float
    battery_wh: float
    busy: bool


class OnlineAnticipatoryReservation:
    """Maintain online demand and recompute FCNN-driven reservations."""

    def __init__(
        self,
        graph: nx.Graph,
        robots: Iterable[RobotState],
        model: ReservationFCNN,
        importance_rates_per_hour: Mapping[float, float],
        *,
        prior_concentration: float = 4.0,
        start_time_min: float = 0.0,
        horizon_min: float = 1.0,
        charger_power_w: float,
    ) -> None:
        self.graph = graph
        self.robots = tuple(robots)
        self.model = model
        self.importance_levels = tuple(float(value) for value in model.importance_levels)
        self.horizon_min = max(float(horizon_min), 1e-9)
        self.charger_power_w = float(charger_power_w)
        if self.charger_power_w <= 0:
            raise ValueError("charger_power_w must be positive")
        fleet_types = tuple(robot_type.value for robot_type in RobotType)
        if set(model.robot_types) != set(fleet_types):
            raise ValueError(
                "reservation model robot types must exactly match the configured fleet types"
            )
        if set(self.importance_levels) != {float(value) for value in importance_rates_per_hour}:
            raise ValueError("model importance levels must match configured demand-prior levels")
        self.schema = ReservationFeatureSchema(model.robot_types, self.importance_levels)
        if model.input_dim != len(self.schema.names):
            raise ValueError(
                f"reservation model expects {model.input_dim} features, "
                f"but runtime schema supplies {len(self.schema.names)}"
            )
        self.demand = GammaPoissonDemandModel(
            graph,
            importance_rates_per_hour,
            prior_concentration=prior_concentration,
            start_time_min=start_time_min,
        )
        self.cluster_summary = summarize_existing_clusters(graph)
        self.assignment: ReservationAssignment | None = None

    def observe(self, pickup_node: Hashable, importance: float, time_min: float) -> None:
        self.demand.observe(pickup_node, importance, time_min)

    def update(
        self,
        time_min: float,
        snapshots: Mapping[int, ReservationRobotSnapshot],
        *,
        backlog_by_importance: Mapping[float, int],
        charger_congestion: float = 0.0,
    ) -> ReservationAssignment:
        """Predict strata and choose physical robots using ``H_reserve``."""

        time_min = float(time_min)
        type_robots: dict[RobotType, list[RobotState]] = defaultdict(list)
        for robot in self.robots:
            type_robots[robot.spec.robot_type].append(robot)

        demand_by_importance: dict[float, float] = {}
        cluster_rates: dict[tuple[int, float], float] = {}
        for importance in self.importance_levels:
            total = 0.0
            for cluster_id in self.cluster_summary.cluster_sizes:
                rate = self.demand.posterior(
                    cluster_id, importance, at_time_min=time_min
                ).mean_per_minute
                cluster_rates[(cluster_id, importance)] = rate
                total += rate
            demand_by_importance[importance] = total

        busy_by_type: dict[str, float] = {}
        battery_by_type: dict[str, float] = {}
        for robot_type, members in type_robots.items():
            available = [snapshots[robot.spec.id] for robot in members]
            busy_by_type[robot_type.value] = sum(item.busy for item in available) / len(available)
            battery_by_type[robot_type.value] = sum(
                max(0.0, min(1.0, item.battery_wh / robot.spec.battery_capacity_wh))
                for robot, item in zip(members, available, strict=True)
            ) / len(available)

        features = build_reservation_features(
            self.schema,
            demand_rate_per_minute=demand_by_importance,
            fleet_count=len(self.robots),
            backlog_by_importance=backlog_by_importance,
            busy_fraction_by_type=busy_by_type,
            mean_battery_fraction_by_type=battery_by_type,
            cluster_rate_per_minute=cluster_rates,
            charger_congestion=float(charger_congestion),
            remaining_horizon_fraction=max(
                0.0, min(1.0, (self.horizon_min - time_min) / self.horizon_min)
            ),
        )
        fractions_by_name = self.model.predict(features)
        fractions = {
            robot_type: fractions_by_name[robot_type.value] for robot_type in type_robots
        }
        counts = apportion_reservation_counts(
            fractions,
            {robot_type: len(members) for robot_type, members in type_robots.items()},
            self.importance_levels,
        )

        scores: dict[tuple[int, float], float] = {}
        for robot in self.robots:
            snapshot = snapshots[robot.spec.id]
            response: dict[int, float] = {}
            for cluster_id, representative in self.cluster_summary.representatives.items():
                distance_m = haversine_node_distance_m(
                    self.graph, snapshot.node_id, representative
                )
                travel_min = distance_m / robot.spec.speed_mps / 60.0
                energy_wh = distance_m * robot.spec.energy_per_meter_wh
                charge_min = (
                    max(0.0, energy_wh - snapshot.battery_wh)
                    / self.charger_power_w
                    * 60.0
                )
                response[cluster_id] = snapshot.available_in_min + travel_min + charge_min
            for threshold in self.importance_levels[1:]:
                scores[(robot.spec.id, threshold)] = weighted_reservation_score(
                    response,
                    self.demand,
                    min_importance=threshold,
                    at_time_min=time_min,
                )

        self.assignment = assign_reservation_thresholds(
            self.robots, counts, scores, self.importance_levels
        )
        return self.assignment
