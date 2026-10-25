"""Scenario generation for dynamic delivery-robot experiments.

Ground-truth pickup demand is defined at the node level. For node v and
time bucket t, lambda_by_bucket[t, v] is the expected number of orders whose
pickup is v during that bucket.

Instead of drawing one Poisson random variable per node, generation uses the
Poisson-superposition identity:

    N_t ~ Poisson(sum_v lambda[v, t])
    P(pickup=v | an order occurred) = lambda[v, t] / sum_v lambda[v, t]

This is exactly equivalent to independent Poisson draws at every node, but it
is substantially cheaper on large OSM graphs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Item:
    """Physical properties relevant to robot capacity constraints."""

    weight_kg: float
    volume_l: float

    def __post_init__(self) -> None:
        if self.weight_kg <= 0:
            raise ValueError("weight_kg must be positive")
        if self.volume_l <= 0:
            raise ValueError("volume_l must be positive")


@dataclass(frozen=True, slots=True)
class Order:
    """One delivery request, revealed to a policy only at request_time_min."""

    id: int
    pickup_node: int
    dropoff_node: int
    request_time_min: float
    item: Item
    importance: float

    def __post_init__(self) -> None:
        if self.pickup_node == self.dropoff_node:
            raise ValueError("pickup_node and dropoff_node must be different")
        if self.request_time_min < 0:
            raise ValueError("request_time_min cannot be negative")
        if self.importance <= 0:
            raise ValueError("importance must be positive")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A fixed realization used unchanged across competing policies."""

    graph_name: str
    duration_minutes: float
    seed: int
    orders: tuple[Order, ...]

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        if any(
            self.orders[i].request_time_min > self.orders[i + 1].request_time_min
            for i in range(len(self.orders) - 1)
        ):
            raise ValueError("orders must be sorted by request_time_min")

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph_name": self.graph_name,
            "duration_minutes": self.duration_minutes,
            "seed": self.seed,
            "orders": [asdict(order) for order in self.orders],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Scenario":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        orders = tuple(
            Order(
                id=int(raw["id"]),
                pickup_node=int(raw["pickup_node"]),
                dropoff_node=int(raw["dropoff_node"]),
                request_time_min=float(raw["request_time_min"]),
                item=Item(**raw["item"]),
                importance=float(raw["importance"]),
            )
            for raw in payload["orders"]
        )
        return cls(
            graph_name=str(payload["graph_name"]),
            duration_minutes=float(payload["duration_minutes"]),
            seed=int(payload["seed"]),
            orders=orders,
        )


@dataclass(frozen=True)
class NodeDemandProfile:
    """Expected pickup-order counts for each node and time bucket.

    lambda_by_bucket has shape (num_buckets, num_nodes). Entry [t, i] is
    lambda_{v_i,t}: the expected number of pickups at node_ids[i] during
    bucket t.
    """

    node_ids: np.ndarray
    lambda_by_bucket: np.ndarray
    bucket_minutes: float = 60.0

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        rates = np.asarray(self.lambda_by_bucket, dtype=np.float64)

        if node_ids.ndim != 1:
            raise ValueError("node_ids must be one-dimensional")
        if len(node_ids) < 2:
            raise ValueError("at least two graph nodes are required")
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique")
        if rates.ndim != 2:
            raise ValueError("lambda_by_bucket must be a 2-D array")
        if rates.shape[1] != len(node_ids):
            raise ValueError(
                "lambda_by_bucket.shape[1] must equal len(node_ids)"
            )
        if rates.shape[0] == 0:
            raise ValueError("at least one time bucket is required")
        if not np.all(np.isfinite(rates)):
            raise ValueError("all lambda values must be finite")
        if np.any(rates < 0):
            raise ValueError("lambda values cannot be negative")
        if self.bucket_minutes <= 0:
            raise ValueError("bucket_minutes must be positive")

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "lambda_by_bucket", rates)

    @property
    def num_buckets(self) -> int:
        return int(self.lambda_by_bucket.shape[0])

    @property
    def duration_minutes(self) -> float:
        return self.num_buckets * self.bucket_minutes


@dataclass(frozen=True, slots=True)
class ItemDistribution:
    """Simple V1 item model; all values are sampled uniformly."""

    min_weight_kg: float = 0.5
    max_weight_kg: float = 10.0
    min_volume_l: float = 0.5
    max_volume_l: float = 25.0

    def __post_init__(self) -> None:
        if not (0 < self.min_weight_kg <= self.max_weight_kg):
            raise ValueError("invalid weight range")
        if not (0 < self.min_volume_l <= self.max_volume_l):
            raise ValueError("invalid volume range")

    def sample(self, rng: np.random.Generator) -> Item:
        return Item(
            weight_kg=float(
                rng.uniform(self.min_weight_kg, self.max_weight_kg)
            ),
            volume_l=float(
                rng.uniform(self.min_volume_l, self.max_volume_l)
            ),
        )


@dataclass(frozen=True)
class ImportanceDistribution:
    """Discrete importance levels used by the weighted waiting-time loss."""

    values: tuple[float, ...] = (1.0, 2.0, 5.0)
    probabilities: tuple[float, ...] = (0.80, 0.15, 0.05)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)

        if values.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("values and probabilities must be 1-D")
        if len(values) == 0 or len(values) != len(probabilities):
            raise ValueError("values and probabilities must have equal length")
        if np.any(values <= 0):
            raise ValueError("all importance values must be positive")
        if np.any(probabilities < 0):
            raise ValueError("probabilities cannot be negative")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("importance probabilities must sum to 1")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self.values, p=self.probabilities))


class ScenarioCreator:
    """Generate fixed stochastic order realizations from node-level demand."""

    def __init__(
        self,
        item_distribution: ItemDistribution | None = None,
        importance_distribution: ImportanceDistribution | None = None,
    ) -> None:
        self.item_distribution = item_distribution or ItemDistribution()
        self.importance_distribution = (
            importance_distribution or ImportanceDistribution()
        )

    def create(
        self,
        graph_name: str,
        demand: NodeDemandProfile,
        seed: int,
    ) -> Scenario:
        """Generate one scenario.

        Drop-off nodes are uniform over all graph nodes except the pickup node
        in V1. A richer origin/destination model can be swapped in later
        without changing Order or Scenario.
        """

        rng = np.random.default_rng(seed)
        pending: list[tuple[float, int, int, Item, float]] = []

        for bucket in range(demand.num_buckets):
            rates = demand.lambda_by_bucket[bucket]
            total_lambda = float(rates.sum())

            if total_lambda == 0:
                continue

            # Equivalent to independent:
            # N_{v,t} ~ Poisson(lambda_{v,t}) for every node v.
            order_count = int(rng.poisson(total_lambda))
            if order_count == 0:
                continue

            pickup_probabilities = rates / total_lambda
            pickup_indices = rng.choice(
                len(demand.node_ids),
                size=order_count,
                p=pickup_probabilities,
            )

            bucket_start = bucket * demand.bucket_minutes
            request_times = bucket_start + rng.uniform(
                0.0, demand.bucket_minutes, size=order_count
            )

            for pickup_index, request_time in zip(
                pickup_indices, request_times, strict=True
            ):
                pickup_index = int(pickup_index)

                # Uniformly choose any other node without rejection sampling.
                dropoff_index = int(rng.integers(0, len(demand.node_ids) - 1))
                if dropoff_index >= pickup_index:
                    dropoff_index += 1

                pending.append(
                    (
                        float(request_time),
                        int(demand.node_ids[pickup_index]),
                        int(demand.node_ids[dropoff_index]),
                        self.item_distribution.sample(rng),
                        self.importance_distribution.sample(rng),
                    )
                )

        pending.sort(key=lambda row: row[0])

        orders = tuple(
            Order(
                id=order_id,
                pickup_node=pickup,
                dropoff_node=dropoff,
                request_time_min=request_time,
                item=item,
                importance=importance,
            )
            for order_id, (
                request_time,
                pickup,
                dropoff,
                item,
                importance,
            ) in enumerate(pending)
        )

        return Scenario(
            graph_name=graph_name,
            duration_minutes=demand.duration_minutes,
            seed=seed,
            orders=orders,
        )


def constant_demand_profile(
    node_ids: Sequence[int],
    expected_orders_per_hour: float,
    duration_hours: int,
    *,
    bucket_minutes: float = 60.0,
) -> NodeDemandProfile:
    """Convenience profile for smoke tests.

    Total city-wide demand is constant. It is distributed uniformly across
    nodes, so each row still represents node-level lambda_{v,t}.
    """

    if expected_orders_per_hour < 0:
        raise ValueError("expected_orders_per_hour cannot be negative")
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")

    nodes = np.asarray(node_ids, dtype=np.int64)
    if len(nodes) < 2:
        raise ValueError("at least two nodes are required")

    duration_minutes = duration_hours * 60.0
    num_buckets_float = duration_minutes / bucket_minutes
    num_buckets = round(num_buckets_float)
    if not np.isclose(num_buckets_float, num_buckets):
        raise ValueError(
            "duration_hours * 60 must be divisible by bucket_minutes"
        )

    expected_per_bucket = expected_orders_per_hour * bucket_minutes / 60.0
    per_node = expected_per_bucket / len(nodes)
    rates = np.full((num_buckets, len(nodes)), per_node, dtype=np.float64)

    return NodeDemandProfile(
        node_ids=nodes,
        lambda_by_bucket=rates,
        bucket_minutes=bucket_minutes,
    )
