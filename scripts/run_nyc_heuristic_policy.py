from __future__ import annotations

"""Run the Haversine-shortlisted, exact-loss NYC insertion policy.

The first stage is deliberately cheap.  For every payload-capable robot it:

1. enumerates precedence- and capacity-feasible pickup/drop-off insertions;
2. estimates every route leg with Haversine distance;
3. estimates completion times for all affected orders, including a simple
   charging-duration term; and
4. scores the robot by the smallest estimated incremental total delivery loss.

Only the best ``--shortlist-k`` robots reach the existing exact graph-distance
and battery-feasible insertion search.  The final robot/insertion is therefore
the exact minimum incremental loss inside the heuristic shortlist.  Haversine
is never used to execute robot movement or to replace exact final routing.
"""

import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_nyc_benchmark5_loss as corrected


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def _patch_heuristic(source: str) -> str:
    source = _replace_once(
        source,
        "from delivery_fleet.scenario_creator import Order, Scenario\n",
        "from delivery_fleet.scenario_creator import Order, Scenario\n"
        "from delivery_fleet.spatial_demand import haversine_node_distance_m\n",
        "Haversine import",
    )

    source = _replace_once(
        source,
        '    "max_active_orders": 0.0,\n',
        '    "max_active_orders": 0.0,\n'
        '    "haversine_robots_scored": 0.0,\n'
        '    "haversine_insertion_candidates": 0.0,\n'
        '    "haversine_capacity_pruned": 0.0,\n'
        '    "shortlist_robots_selected": 0.0,\n',
        "heuristic counters",
    )

    source = _replace_once(
        source,
        "    parser.add_argument(\n"
        "        \"--output\",\n"
        "        type=Path,\n"
        "        default=ROOT / \"benchmark5_results.json\",\n"
        "    )\n"
        "    args = parser.parse_args()\n",
        "    parser.add_argument(\n"
        "        \"--output\",\n"
        "        type=Path,\n"
        "        default=ROOT / \"heuristic_policy_results.json\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--shortlist-k\",\n"
        "        type=int,\n"
        "        default=10,\n"
        "        help=\"Number of Haversine-ranked robots sent to exact search.\",\n"
        "    )\n"
        "    args = parser.parse_args()\n"
        "    if args.shortlist_k <= 0:\n"
        "        parser.error(\"--shortlist-k must be positive\")\n",
        "shortlist CLI",
    )

    marker = '''    def sequence_completion_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
'''
    approximate_loss = '''    def haversine_sequence_loss(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        order_states: dict[int, ActiveOrder],
    ) -> float:
        """Cheap total loss estimate for all orders affected by one route.

        Route legs use straight-line Haversine distance.  Charging is estimated
        as the cumulative energy deficit divided by charger power; station
        detours and queues are intentionally left to the exact second stage.
        """
        node = int(snapshot.node_id)
        base_elapsed_min = 0.0
        cumulative_energy_wh = 0.0
        initial_battery_wh = max(0.0, float(snapshot.battery_wh))
        projected_loss = 0.0
        completed: set[int] = set()

        for stop in stops:
            distance_m = haversine_node_distance_m(
                graph, node, int(stop.node_id)
            )
            base_elapsed_min += distance_m / robot.spec.speed_mps / 60.0
            cumulative_energy_wh += distance_m * robot.spec.energy_per_meter_wh
            base_elapsed_min += (
                PICKUP_HANDLING_MIN
                if stop.kind == "pickup"
                else DROPOFF_HANDLING_MIN
            )
            node = int(stop.node_id)

            if stop.kind == "dropoff":
                state = order_states[stop.order_id]
                estimated_charge_min = (
                    max(0.0, cumulative_energy_wh - initial_battery_wh)
                    / DEFAULT_CHARGING_POWER_W
                    * 60.0
                )
                completion_time = (
                    float(snapshot.time_min)
                    + base_elapsed_min
                    + estimated_charge_min
                )
                allowance = state.deadline_min - state.order.request_time_min
                projected_loss += delivery_loss(
                    completion_time - state.order.request_time_min,
                    allowance,
                    state.order.importance,
                )
                completed.add(stop.order_id)

        missing = set(order_states) - completed
        if missing:
            raise RuntimeError(
                f"Haversine projection did not complete orders {sorted(missing)}"
            )
        return float(projected_loss)

'''
    source = _replace_once(
        source,
        marker,
        approximate_loss + marker,
        "Haversine sequence loss helper",
    )

    old_selection_start = '''        robot_rows.sort(key=lambda row: (row[0], row[1]))
        best: InsertionChoice | None = None
        best_delta_loss = math.inf
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf
        new_allowance = (
            delivery_deadline_min(order.request_time_min, direct, order.importance)
            - order.request_time_min
        )

        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):
'''
    new_selection_start = '''        # Stage 1: rank robots by Haversine-estimated incremental total loss.
        # Distance and delivery time are inputs to the loss estimate; neither is
        # the selection objective by itself.
        heuristic_rows: list[tuple[float, int]] = []
        for _, robot_id, snapshot in robot_rows:
            robot = robot_by_id[robot_id]
            schedule = schedules[robot_id]
            existing = tuple(schedule.stops)
            existing_states = dict(schedule.orders)
            baseline_loss = (
                haversine_sequence_loss(
                    robot, snapshot, existing, existing_states
                )
                if existing
                else 0.0
            )

            new_state = ActiveOrder(
                order=order,
                direct_distance_m=direct,
                deadline_min=delivery_deadline_min(
                    order.request_time_min, direct, order.importance
                ),
                assigned_at_min=float(now),
                picked_up=False,
            )
            candidate_states = dict(existing_states)
            candidate_states[order.id] = new_state
            best_estimated_delta = math.inf

            for _, _, candidate_stops in _enumerate_insertions(
                existing, pickup_stop, dropoff_stop
            ):
                B5_STATS["haversine_insertion_candidates"] += 1.0
                if not _capacity_feasible(robot, candidate_stops, candidate_states):
                    B5_STATS["haversine_capacity_pruned"] += 1.0
                    continue
                candidate_loss = haversine_sequence_loss(
                    robot, snapshot, candidate_stops, candidate_states
                )
                best_estimated_delta = min(
                    best_estimated_delta, candidate_loss - baseline_loss
                )

            if math.isfinite(best_estimated_delta):
                B5_STATS["haversine_robots_scored"] += 1.0
                heuristic_rows.append((best_estimated_delta, robot_id))

        heuristic_rows.sort(key=lambda row: (row[0], row[1]))
        shortlist_ids = {
            robot_id for _, robot_id in heuristic_rows[: args.shortlist_k]
        }
        B5_STATS["shortlist_robots_selected"] += float(len(shortlist_ids))
        robot_rows = [
            row for row in robot_rows if int(row[1]) in shortlist_ids
        ]
        robot_rows.sort(key=lambda row: (row[0], row[1]))

        # Stage 2: exact graph/battery search and exact incremental-loss choice
        # over only the shortlisted robots.
        best: InsertionChoice | None = None
        best_delta_loss = math.inf
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf
        new_allowance = (
            delivery_deadline_min(order.request_time_min, direct, order.importance)
            - order.request_time_min
        )

        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):
'''
    source = _replace_once(
        source,
        old_selection_start,
        new_selection_start,
        "Haversine shortlist insertion",
    )

    source = _replace_once(
        source,
        '        "policy": "reactive_insertion_interruptible_busy_routes",\n',
        '        "policy": "haversine_top_k_exact_incremental_loss",\n',
        "policy metadata",
    )
    source = _replace_once(
        source,
        '        "optimization": "all-order-loss-lower-bound-plus-baseline-state-cache",\n',
        '        "optimization": "haversine_all_order_loss_shortlist_then_exact_search",\n'
        '        "shortlist_k": int(args.shortlist_k),\n'
        '        "shortlist_distance": "haversine_great_circle_meters",\n'
        '        "shortlist_objective": "minimum_estimated_incremental_total_loss",\n'
        '        "final_objective": "minimum_exact_incremental_total_loss_within_shortlist",\n',
        "heuristic metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = corrected._patch_source(corrected.TARGET.read_text(encoding="utf-8"))
    source = _patch_heuristic(source)
    module_name = "nyc_heuristic_policy_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(corrected.TARGET)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(corrected.TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    namespace = _load_namespace()
    namespace["main"]()


if __name__ == "__main__":
    main()
