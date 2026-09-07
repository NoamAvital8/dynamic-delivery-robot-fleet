"""Fixed V1 robot models and reproducible default fleet generation.

The fleet size scales with graph size:

    number_of_robots = ceil(number_of_nodes / 500)

Four fixed robot subclasses provide controlled heterogeneity.  Robots of the
same subclass are intentionally identical; stochasticity comes from the
scenario and randomized initial positions, not arbitrary within-model noise.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
import math
from typing import TypeAlias

import networkx as nx
import numpy as np

from .robot import NodeId, RobotSpec, RobotState


NODES_PER_ROBOT = 500
DEFAULT_FLEET_SEED = 42


class RobotType(str, Enum):
    SPEEDY_MCQUEEN = "speedy_mcqueen"
    MIDDLE_MAN = "middle_man"
    ENDURO = "enduro"
    OOMPH = "oomph"


class SpeedyMcQueen(RobotSpec):
    """Very fast, small-capacity, short-range robot."""

    __slots__ = ()
    robot_type = RobotType.SPEEDY_MCQUEEN
    display_name = "Speedy McQueen"

    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            speed_mps=5.5,
            max_payload_kg=5.0,
            max_volume_l=12.0,
            battery_capacity_wh=700.0,
            energy_per_meter_wh=0.07,
        )


class MiddleMan(RobotSpec):
    """Balanced general-purpose robot."""

    __slots__ = ()
    robot_type = RobotType.MIDDLE_MAN
    display_name = "Middle Man"

    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            speed_mps=4.0,
            max_payload_kg=12.0,
            max_volume_l=30.0,
            battery_capacity_wh=1_200.0,
            energy_per_meter_wh=0.075,
        )


class Enduro(RobotSpec):
    """Efficient long-range robot with moderate carrying capacity."""

    __slots__ = ()
    robot_type = RobotType.ENDURO
    display_name = "Enduro"

    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            speed_mps=3.5,
            max_payload_kg=10.0,
            max_volume_l=25.0,
            battery_capacity_wh=1_800.0,
            energy_per_meter_wh=0.065,
        )


class Oomph(RobotSpec):
    """Slowest model, but with very large cargo capacity and battery."""

    __slots__ = ()
    robot_type = RobotType.OOMPH
    display_name = "Oomph"

    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            speed_mps=2.8,
            max_payload_kg=30.0,
            max_volume_l=80.0,
            battery_capacity_wh=2_400.0,
            energy_per_meter_wh=0.10,
        )


RobotModel: TypeAlias = type[SpeedyMcQueen] | type[MiddleMan] | type[Enduro] | type[Oomph]

# Exact target shares. Any integer-rounding remainder is assigned by largest
# fractional remainder, making fleet composition deterministic for every size.
DEFAULT_FLEET_PROPORTIONS: dict[RobotType, float] = {
    RobotType.SPEEDY_MCQUEEN: 0.25,
    RobotType.MIDDLE_MAN: 0.40,
    RobotType.ENDURO: 0.20,
    RobotType.OOMPH: 0.15,
}

_MODEL_BY_TYPE: dict[RobotType, type[RobotSpec]] = {
    RobotType.SPEEDY_MCQUEEN: SpeedyMcQueen,
    RobotType.MIDDLE_MAN: MiddleMan,
    RobotType.ENDURO: Enduro,
    RobotType.OOMPH: Oomph,
}


def robot_count_for_graph(graph: nx.Graph) -> int:
    """Return max(1, ceil(|V| / 500))."""

    number_of_nodes = graph.number_of_nodes()
    if number_of_nodes <= 0:
        raise ValueError("graph must contain at least one node")
    return max(1, math.ceil(number_of_nodes / NODES_PER_ROBOT))


def fleet_type_counts(robot_count: int) -> dict[RobotType, int]:
    """Convert target proportions to exact integer counts summing to robot_count."""

    if robot_count <= 0:
        raise ValueError("robot_count must be positive")

    exact = {
        robot_type: robot_count * proportion
        for robot_type, proportion in DEFAULT_FLEET_PROPORTIONS.items()
    }
    counts = {robot_type: math.floor(value) for robot_type, value in exact.items()}
    remaining = robot_count - sum(counts.values())

    # Deterministic largest-remainder apportionment. Enum declaration order is
    # the final tie-breaker.
    order = list(DEFAULT_FLEET_PROPORTIONS)
    ranked = sorted(
        order,
        key=lambda robot_type: (
            -(exact[robot_type] - counts[robot_type]),
            order.index(robot_type),
        ),
    )
    for robot_type in ranked[:remaining]:
        counts[robot_type] += 1

    return counts


def _make_spec(robot_type: RobotType, robot_id: int) -> RobotSpec:
    return _MODEL_BY_TYPE[robot_type](robot_id)


def create_default_fleet(
    graph: nx.Graph,
    *,
    seed: int = DEFAULT_FLEET_SEED,
    robot_count: int | None = None,
) -> list[RobotState]:
    """Create the default heterogeneous fleet at random graph nodes.

    - Fleet size defaults to ``ceil(number_of_nodes / 500)``.
    - Type counts use the fixed 25/40/20/15 percent composition.
    - Initial nodes are sampled uniformly without replacement when possible.
    - Every robot starts idle, available, and at 100% battery.
    - The same graph + seed + count produces the same fleet.
    """

    if graph.number_of_nodes() <= 0:
        raise ValueError("graph must contain at least one node")

    if robot_count is None:
        robot_count = robot_count_for_graph(graph)
    if robot_count <= 0:
        raise ValueError("robot_count must be positive")
    if robot_count > graph.number_of_nodes():
        raise ValueError(
            "robot_count cannot exceed node count when initial positions are unique"
        )

    rng = np.random.default_rng(seed)
    nodes: list[NodeId] = sorted(graph.nodes(), key=lambda node: str(node))
    chosen_indices = rng.choice(len(nodes), size=robot_count, replace=False)
    initial_nodes = [nodes[int(index)] for index in chosen_indices]

    counts = fleet_type_counts(robot_count)
    robot_types: list[RobotType] = []
    for robot_type in DEFAULT_FLEET_PROPORTIONS:
        robot_types.extend([robot_type] * counts[robot_type])
    rng.shuffle(robot_types)

    fleet: list[RobotState] = []
    for robot_id, (robot_type, node_id) in enumerate(
        zip(robot_types, initial_nodes, strict=True)
    ):
        spec = _make_spec(robot_type, robot_id)
        fleet.append(RobotState.fully_charged(spec, node_id))

    return fleet


def fleet_type_summary(fleet: list[RobotState]) -> dict[RobotType, int]:
    """Count robot subclasses in an already-created fleet."""

    counts: Counter[RobotType] = Counter()
    for robot in fleet:
        robot_type = getattr(robot.spec, "robot_type", None)
        if robot_type is None:
            raise ValueError("fleet contains a RobotSpec without a fixed RobotType")
        counts[robot_type] += 1
    return {robot_type: counts[robot_type] for robot_type in RobotType}
