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
        "from delivery_fleet.anticipatory_policy import (\n"
        "    OnlineAnticipatoryReservation, ReservationRobotSnapshot,\n"
        ")\n"
        "from delivery_fleet.reservation import reservation_eligibility\n"
        "from delivery_fleet.reservation_nn import (\n"
        "    FixedReservationModel, ReservationFCNN,\n"
        ")\n"
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
        '    "shortlist_robots_selected": 0.0,\n'
        '    "shortlist_fallback_robots": 0.0,\n'
        '    "reservation_updates": 0.0,\n'
        '    "reservation_filtered_robots": 0.0,\n',
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
        "    parser.add_argument(\n"
        "        \"--reservation-model\", type=Path, default=None,\n"
        "        help=\"Trained ReservationFCNN .npz file; omit to disable reservation.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--fixed-reservation-fractions\", type=Path, default=None,\n"
        "        help=\"Fixed alpha JSON used only for offline target evaluation.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--importance-prior-rates-per-hour\",\n"
        "        default='{\"1\": 120.0, \"2\": 22.5, \"5\": 7.5}',\n"
        "        help=\"Configured JSON mapping of importance to city-wide hourly rate.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--prior-concentration\", type=float, default=4.0,\n"
        "    )\n"
        "    args = parser.parse_args()\n"
        "    if args.shortlist_k <= 0:\n"
        "        parser.error(\"--shortlist-k must be positive\")\n"
        "    if (\n"
        "        args.reservation_model is not None\n"
        "        and args.fixed_reservation_fractions is not None\n"
        "    ):\n"
        "        parser.error(\"choose either a trained model or fixed fractions, not both\")\n",
        "shortlist CLI",
    )

    source = _replace_once(
        source,
        '    print(f"robots={len(robots):,} types={fleet_type_summary(robots)}", flush=True)\n\n'
        '    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)\n',
        '    print(f"robots={len(robots):,} types={fleet_type_summary(robots)}", flush=True)\n\n'
        '    reservation_policy = None\n'
        '    reservation_assignment = None\n'
        '    if (\n'
        '        args.reservation_model is not None\n'
        '        or args.fixed_reservation_fractions is not None\n'
        '    ):\n'
        '        try:\n'
        '            prior_rates = {\n'
        '                float(key): float(value)\n'
        '                for key, value in json.loads(\n'
        '                    args.importance_prior_rates_per_hour\n'
        '                ).items()\n'
        '            }\n'
        '        except (TypeError, ValueError, json.JSONDecodeError) as exc:\n'
        '            parser.error(f"invalid importance prior rates JSON: {exc}")\n'
        '        reservation_model = (\n'
        '            ReservationFCNN.load(args.reservation_model)\n'
        '            if args.reservation_model is not None\n'
        '            else FixedReservationModel.load(args.fixed_reservation_fractions)\n'
        '        )\n'
        '        reservation_policy = OnlineAnticipatoryReservation(\n'
        '            graph, robots, reservation_model, prior_rates,\n'
        '            prior_concentration=args.prior_concentration,\n'
        '            start_time_min=0.0,\n'
        '            horizon_min=scenario.duration_minutes,\n'
        '            charger_power_w=DEFAULT_CHARGING_POWER_W,\n'
        '        )\n'
        '        print(\n'
        '            f"reservation source="\n'
        '            f"{args.reservation_model or args.fixed_reservation_fractions}",\n'
        '            flush=True,\n'
        '        )\n\n'
        '    oracle = DistanceOracle(graph, edge_weight=EDGE_WEIGHT)\n',
        "reservation initialization",
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

    reservation_helper_marker = '''    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:
'''
    reservation_helper = '''    def refresh_reservations(now: float) -> None:
        nonlocal reservation_assignment
        if reservation_policy is None:
            reservation_assignment = None
            return
        snapshots = {}
        for robot in robots:
            snapshot = decision_snapshot(robot, now)
            if snapshot is None:
                snapshots[robot.spec.id] = ReservationRobotSnapshot(
                    node_id=int(robot.node_id),
                    available_in_min=0.0,
                    battery_wh=float(robot.battery_wh),
                    busy=True,
                )
            else:
                snapshots[robot.spec.id] = ReservationRobotSnapshot(
                    node_id=int(snapshot.node_id),
                    available_in_min=max(0.0, float(snapshot.time_min) - float(now)),
                    battery_wh=max(0.0, float(snapshot.battery_wh)),
                    busy=bool(schedules[robot.spec.id].orders),
                )
        reservation_assignment = reservation_policy.update(
            now,
            snapshots,
        )
        B5_STATS["reservation_updates"] += 1.0

'''
    source = _replace_once(
        source,
        reservation_helper_marker,
        reservation_helper + reservation_helper_marker,
        "online reservation refresh helper",
    )

    source = _replace_once(
        source,
        '''        for robot in robots:
            if not robot.can_hold(order.item):
                continue
''',
        '''        for robot in robots:
            if not reservation_eligibility(
                reservation_assignment, robot.spec.id, order.importance
            ):
                B5_STATS["reservation_filtered_robots"] += 1.0
                continue
            if not robot.can_hold(order.item):
                continue
''',
        "reservation hard-feasibility filter",
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
        heuristic_rank = {
            robot_id: rank for rank, (_, robot_id) in enumerate(heuristic_rows)
        }
        B5_STATS["shortlist_robots_selected"] += float(
            min(args.shortlist_k, len(heuristic_rows))
        )
        robot_rows = [
            row for row in robot_rows if int(row[1]) in heuristic_rank
        ]
        robot_rows.sort(key=lambda row: (heuristic_rank[int(row[1])], row[1]))

        # Stage 2: exact graph/battery search and exact incremental-loss choice
        # over the shortlist.  If every initial top-K robot is battery-infeasible,
        # expand one ranked robot at a time until a feasible candidate is found.
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
            if robot_index >= args.shortlist_k and best is not None:
                break
            if robot_index >= args.shortlist_k:
                B5_STATS["shortlist_fallback_robots"] += 1.0
'''
    source = _replace_once(
        source,
        old_selection_start,
        new_selection_start,
        "Haversine shortlist insertion",
    )

    source = _replace_once(
        source,
        '''        if kind == "order_arrival":
            pending.append(payload)
            dispatch_pending(now)
''',
        '''        if kind == "order_arrival":
            order = payload
            pending.append(order)
            if reservation_policy is not None:
                reservation_policy.observe(
                    order.pickup_node, order.importance, now
                )
                refresh_reservations(now)
            dispatch_pending(now)
''',
        "online posterior update",
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
        '        "final_objective": "minimum_exact_incremental_total_loss_with_safe_shortlist_expansion",\n'
        '        "reservation_enabled": reservation_policy is not None,\n'
        '        "reservation_model": (\n'
        '            str(args.reservation_model) if args.reservation_model is not None else None\n'
        '        ),\n'
        '        "fixed_reservation_fractions": (\n'
        '            str(args.fixed_reservation_fractions)\n'
        '            if args.fixed_reservation_fractions is not None else None\n'
        '        ),\n'
        '        "reservation_updates": int(B5_STATS["reservation_updates"]),\n'
        '        "reservation_filtered_robots": int(B5_STATS["reservation_filtered_robots"]),\n'
        '        "shortlist_fallback_robots": int(B5_STATS["shortlist_fallback_robots"]),\n',
        "heuristic metadata",
    )
    source = _replace_once(
        source,
        '    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")\n',
        '    args.output.parent.mkdir(parents=True, exist_ok=True)\n'
        '    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")\n',
        "output directory creation",
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
