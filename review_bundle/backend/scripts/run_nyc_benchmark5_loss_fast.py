from __future__ import annotations

"""High-throughput corrected Benchmark 5 runner.

The policy objective is still exactly

    Delta L = projected_loss_after_insertion - projected_loss_before_insertion.

The expensive part of B5 was not the insertion enumeration itself: it was
running a Python charging-station meta-graph Dijkstra for every stop of every
candidate sequence.  On NYC there are 262 chargers, so that dominated runtime.

This runner scores candidates with ``FastBatteryProjectionIndex``.  It
precomputes the charger-to-charger feasible closure once per robot range in
compiled SciPy and then evaluates numeric battery/travel/charge state with
vectorized operations.  Only the final winning insertion is materialized with
the original exact battery router, so actual execution keeps the original path
and charger tie-breaking semantics.

Additional exact branch-and-bound improvements:
- solve empty-schedule robots first to obtain a strong feasible incumbent;
- include the existing-order sequence loss in the busy-robot lower bound;
- keep the corrected all-order insertion loss lower bound and baseline cache.
"""

import math
from pathlib import Path
import sys
import time
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


def _patch_fast(source: str) -> str:
    source = _replace_once(
        source,
        "from delivery_fleet.insertion_routing import evaluate_pair_from_committed_state\n",
        "from delivery_fleet.insertion_routing import evaluate_pair_from_committed_state\n"
        "from delivery_fleet.fast_battery_projection import FastBatteryProjectionIndex\n",
        "fast projection import",
    )

    source = _replace_once(
        source,
        "    router = BatteryFeasibleRouter(\n"
        "        graph,\n"
        "        station_nodes=station_nodes,\n"
        "        edge_weight=EDGE_WEIGHT,\n"
        "        distance_oracle=oracle,\n"
        "        charger_index=charger_index,\n"
        "    )\n\n"
        "    node_index = getattr(oracle, \"_fast_node_index\", None)\n",
        "    router = BatteryFeasibleRouter(\n"
        "        graph,\n"
        "        station_nodes=station_nodes,\n"
        "        edge_weight=EDGE_WEIGHT,\n"
        "        distance_oracle=oracle,\n"
        "        charger_index=charger_index,\n"
        "    )\n"
        "    fast_projection = FastBatteryProjectionIndex(router)\n\n"
        "    node_index = getattr(oracle, \"_fast_node_index\", None)\n",
        "fast projection initialization",
    )

    source = _replace_once(
        source,
        "    baseline_projection_cache: dict[int, tuple[object, float]] = {}\n\n"
        "    delivery_distance_m = 0.0\n",
        "    baseline_projection_cache: dict[int, tuple[object, float]] = {}\n"
        "    numeric_sequence_evaluations = 0\n"
        "    numeric_sequence_seconds = 0.0\n\n"
        "    delivery_distance_m = 0.0\n",
        "numeric scoring counters",
    )

    # Insert a numeric sequence evaluator immediately before the lower-bound
    # helper.  It mirrors evaluate_sequence's timing/loss semantics but asks the
    # vectorized charger projection only for numeric arrival state; phases are
    # deliberately empty because only the winner is later materialized exactly.
    marker = '''    def sequence_completion_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
'''
    numeric_eval = '''    def evaluate_sequence_score(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        order_states: dict[int, ActiveOrder],
        new_order_id: int | None,
        special_rows: dict[int, np.ndarray],
    ) -> SequenceEvaluation:
        nonlocal numeric_sequence_evaluations, numeric_sequence_seconds
        started = time.perf_counter()
        numeric_sequence_evaluations += 1
        try:
            node = int(snapshot.node_id)
            battery = float(snapshot.battery_wh)
            projected_time = float(snapshot.time_min)
            steps: list[StepPlan] = []
            completion: dict[int, float] = {}
            total_distance = 0.0

            for i, stop in enumerate(stops):
                if node == int(stop.node_id):
                    route_time = 0.0
                    route_distance = 0.0
                    arrival_battery = battery
                else:
                    lookahead = first_distinct_later_node(stops, i)
                    snapshot_robot = RobotState(
                        spec=robot.spec,
                        node_id=node,
                        battery_wh=battery,
                        available=False,
                    )
                    start_to_first = exact_distance(
                        node, int(stop.node_id), special_rows
                    )
                    if lookahead is None:
                        numeric = fast_projection.project_final_prefix(
                            snapshot_robot,
                            int(stop.node_id),
                            start_to_target_m=start_to_first,
                        )
                    else:
                        first_to_next = exact_distance(
                            int(stop.node_id), int(lookahead), special_rows
                        )
                        numeric = fast_projection.project_pair_prefix(
                            snapshot_robot,
                            int(stop.node_id),
                            int(lookahead),
                            start_to_first_m=start_to_first,
                            first_to_second_m=first_to_next,
                        )
                    route_time = float(numeric.total_time_min)
                    route_distance = float(numeric.total_distance_m)
                    arrival_battery = float(numeric.arrival_battery_wh)

                steps.append(
                    StepPlan(
                        stop=stop,
                        start_node=node,
                        start_battery_wh=battery,
                        phases=(),
                        route_time_min=route_time,
                        route_distance_m=route_distance,
                        arrival_battery_wh=arrival_battery,
                    )
                )
                projected_time += route_time
                total_distance += route_distance
                node = int(stop.node_id)
                battery = arrival_battery
                if stop.kind == "pickup":
                    projected_time += PICKUP_HANDLING_MIN
                else:
                    projected_time += DROPOFF_HANDLING_MIN
                    completion[stop.order_id] = float(projected_time)

            if new_order_id is not None and new_order_id not in completion:
                raise RuntimeError("numeric candidate never completed the new order")

            projected_loss = 0.0
            for order_id, state in order_states.items():
                done_at = completion.get(order_id)
                if done_at is None:
                    raise RuntimeError(
                        f"numeric candidate did not complete order {order_id}"
                    )
                allowance = state.deadline_min - state.order.request_time_min
                projected_loss += delivery_loss(
                    done_at - state.order.request_time_min,
                    allowance,
                    state.order.importance,
                )

            return SequenceEvaluation(
                steps=tuple(steps),
                completion_by_order=completion,
                total_distance_m=float(total_distance),
                projected_loss=float(projected_loss),
                new_completion_time_min=(
                    float(completion[new_order_id])
                    if new_order_id is not None
                    else math.inf
                ),
            )
        finally:
            numeric_sequence_seconds += time.perf_counter() - started

''' + marker
    source = _replace_once(source, marker, numeric_eval, "numeric sequence evaluator")

    # Baseline projections only need completion times/loss, not an executable
    # path, so use the vectorized scorer and retain the same state cache.
    source = _replace_once(
        source,
        "                    baseline_evaluation = evaluate_sequence(\n"
        "                        robot, snapshot, existing, order_states, None, {}\n"
        "                    )\n",
        "                    baseline_evaluation = evaluate_sequence_score(\n"
        "                        robot, snapshot, existing, order_states, None, {}\n"
        "                    )\n",
        "numeric baseline projection",
    )

    # Establish a strong exact-with-respect-to-score incumbent from robots that
    # have no active delivery schedule.  For them there is only one stop order.
    seed_marker = '''        best: InsertionChoice | None = None
        best_delta_loss = math.inf
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf
        new_allowance = (
            delivery_deadline_min(order.request_time_min, direct, order.importance)
            - order.request_time_min
        )

        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):
            B5_STATS["robots_considered"] += 1.0
'''
    seed_replacement = '''        best: InsertionChoice | None = None
        best_delta_loss = math.inf
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf
        new_allowance = (
            delivery_deadline_min(order.request_time_min, direct, order.importance)
            - order.request_time_min
        )

        for universal_lb, robot_id, snapshot in robot_rows:
            schedule = schedules[robot_id]
            if schedule.orders:
                continue
            loss_lb = delivery_loss(
                max(0.0, universal_lb - order.request_time_min),
                new_allowance,
                order.importance,
            )
            if loss_lb >= best_delta_loss - 1e-9:
                continue
            robot = robot_by_id[robot_id]
            new_state = ActiveOrder(
                order=order,
                direct_distance_m=direct,
                deadline_min=delivery_deadline_min(
                    order.request_time_min, direct, order.importance
                ),
                assigned_at_min=float(now),
                picked_up=False,
            )
            empty_stops = (pickup_stop, dropoff_stop)
            try:
                evaluation = evaluate_sequence_score(
                    robot,
                    snapshot,
                    empty_stops,
                    {order.id: new_state},
                    order.id,
                    special_rows,
                )
            except NoFeasibleBatteryRoute:
                B5_STATS["no_feasible_sequence"] += 1.0
                continue
            completion = evaluation.new_completion_time_min
            loss = evaluation.projected_loss
            distance = evaluation.total_distance_m
            better = (
                loss < best_delta_loss - 1e-6
                or (
                    math.isclose(loss, best_delta_loss, abs_tol=1e-6, rel_tol=0.0)
                    and (
                        completion < best_completion - 1e-9
                        or (
                            math.isclose(
                                completion,
                                best_completion,
                                abs_tol=1e-9,
                                rel_tol=0.0,
                            )
                            and (
                                distance < best_distance - 1e-6
                                or (
                                    math.isclose(
                                        distance,
                                        best_distance,
                                        abs_tol=1e-6,
                                        rel_tol=0.0,
                                    )
                                    and (best is None or robot_id < best.robot_id)
                                )
                            )
                        )
                    )
                )
            )
            if better:
                best_delta_loss = float(loss)
                best_completion = float(completion)
                best_loss = float(loss)
                best_distance = float(distance)
                best = InsertionChoice(
                    robot_id=robot_id,
                    pickup_pos=0,
                    dropoff_pos=1,
                    stops=empty_stops,
                    evaluation=evaluation,
                    snapshot=snapshot,
                )

        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):
            B5_STATS["robots_considered"] += 1.0
            # Empty schedules were already solved above.
            if not schedules[robot_id].orders:
                continue
'''
    source = _replace_once(
        source, seed_marker, seed_replacement, "empty-schedule incumbent"
    )

    # The corrected runner's robot-level bound considered only the new request.
    # Existing stops retain their relative order after insertion, so the
    # no-insertion sequence loss lower bound can safely be added too.
    source = _replace_once(
        source,
        "            robot_new_loss_lb = delivery_loss(\n"
        "                max(0.0, universal_lb - order.request_time_min),\n"
        "                new_allowance,\n"
        "                order.importance,\n"
        "            )\n"
        "            robot_delta_lb = robot_new_loss_lb - baseline_loss\n",
        "            robot_new_loss_lb = delivery_loss(\n"
        "                max(0.0, universal_lb - order.request_time_min),\n"
        "                new_allowance,\n"
        "                order.importance,\n"
        "            )\n"
        "            existing_loss_lb = sequence_loss_lower_bound(\n"
        "                robot, snapshot, existing, order_states, {}\n"
        "            )\n"
        "            robot_delta_lb = (\n"
        "                existing_loss_lb + robot_new_loss_lb - baseline_loss\n"
        "            )\n",
        "strong robot loss lower bound",
    )

    # Candidate score evaluation becomes numeric/vectorized.  This replacement
    # is unique after the baseline call above was changed separately.
    source = _replace_once(
        source,
        "                    evaluation = evaluate_sequence(\n"
        "                        robot,\n"
        "                        snapshot,\n"
        "                        stops,\n"
        "                        order_states,\n"
        "                        order.id,\n"
        "                        special_rows,\n"
        "                    )\n",
        "                    evaluation = evaluate_sequence_score(\n"
        "                        robot,\n"
        "                        snapshot,\n"
        "                        stops,\n"
        "                        order_states,\n"
        "                        order.id,\n"
        "                        special_rows,\n"
        "                    )\n",
        "numeric candidate evaluation",
    )

    # Before returning the selected candidate, materialize it once with the
    # original route planner.  This supplies executable RoutePhase objects and
    # preserves original charger/path tie-breaking during the actual simulation.
    source = _replace_once(
        source,
        "        return best\n\n"
        "    def cancel_editable_route_for_replan(\n",
        "        if best is not None:\n"
        "            winner_robot = robot_by_id[best.robot_id]\n"
        "            winner_schedule = schedules[best.robot_id]\n"
        "            winner_state = ActiveOrder(\n"
        "                order=order,\n"
        "                direct_distance_m=direct,\n"
        "                deadline_min=delivery_deadline_min(\n"
        "                    order.request_time_min, direct, order.importance\n"
        "                ),\n"
        "                assigned_at_min=float(now),\n"
        "                picked_up=False,\n"
        "            )\n"
        "            winner_states = dict(winner_schedule.orders)\n"
        "            winner_states[order.id] = winner_state\n"
        "            materialized = evaluate_sequence(\n"
        "                winner_robot,\n"
        "                best.snapshot,\n"
        "                best.stops,\n"
        "                winner_states,\n"
        "                order.id,\n"
        "                special_rows,\n"
        "            )\n"
        "            # Numeric scoring and exact materialization solve the same\n"
        "            # distance-minimizing battery meta problem. Fail loudly if a\n"
        "            # future routing change breaks that equivalence.\n"
        "            if not math.isclose(\n"
        "                materialized.projected_loss,\n"
        "                best.evaluation.projected_loss,\n"
        "                rel_tol=0.0,\n"
        "                abs_tol=2e-4,\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'fast B5 score/materialization loss mismatch: '\n"
        "                    f'{best.evaluation.projected_loss} vs '\n"
        "                    f'{materialized.projected_loss}'\n"
        "                )\n"
        "            best = InsertionChoice(\n"
        "                robot_id=best.robot_id,\n"
        "                pickup_pos=best.pickup_pos,\n"
        "                dropoff_pos=best.dropoff_pos,\n"
        "                stops=best.stops,\n"
        "                evaluation=materialized,\n"
        "                snapshot=best.snapshot,\n"
        "            )\n"
        "        return best\n\n"
        "    def cancel_editable_route_for_replan(\n",
        "winner exact materialization",
    )

    source = _replace_once(
        source,
        '        "optimization": "all-order-loss-lower-bound-plus-baseline-state-cache",\n',
        '        "optimization": "vectorized-charger-projection-plus-exact-winner-materialization",\n'
        '        "numeric_sequence_evaluations": int(numeric_sequence_evaluations),\n'
        '        "numeric_sequence_seconds": float(numeric_sequence_seconds),\n',
        "fast result metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = corrected._patch_source(corrected.TARGET.read_text(encoding="utf-8"))
    source = _patch_fast(source)
    module_name = "benchmark5_loss_fast_target"
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
