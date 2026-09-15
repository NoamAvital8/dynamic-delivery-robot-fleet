from __future__ import annotations

"""Fast exact runner for corrected Benchmark 5.

This keeps the corrected B5 objective exactly unchanged:

    Delta L = projected_loss_after_insertion - projected_loss_before_insertion

and adds exact branch-and-bound accelerations on top of
``run_nyc_benchmark5_loss.py``:

1. Find the exact best empty-schedule robot first.  This gives a strong feasible
   incumbent before considering busy robots.  Empty-schedule robots have only
   one possible pickup/dropoff sequence, and their loss lower bound is monotone
   in the new request completion-time lower bound, so this seeding is exact.
2. Strengthen the busy-robot lower bound with a lower bound on the loss of the
   robot's existing stop sequence.  New inserted stops cannot make the old
   relative-order sequence complete earlier than that no-insertion lower bound.
3. Cache battery-feasible per-stop PrefixPlan objects within each assignment
   decision.  Many insertion candidates share identical prefixes; reusing the
   exact prefix plan avoids repeating the expensive charger-aware route search.

No candidate is dropped unless an admissible lower bound proves it cannot beat
an already evaluated feasible solution.
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


def _patch_fast(source: str) -> str:
    # Per-assignment prefix cache.  It is intentionally cleared for each new
    # request so memory stays bounded; sharing is most valuable among the many
    # insertion candidates evaluated for the same request.
    source = _replace_once(
        source,
        "    baseline_projection_cache: dict[int, tuple[object, float]] = {}\n\n"
        "    delivery_distance_m = 0.0\n",
        "    baseline_projection_cache: dict[int, tuple[object, float]] = {}\n"
        "    step_prefix_cache: dict[tuple[object, ...], PrefixPlan] = {}\n"
        "    prefix_cache_hits = 0\n"
        "    prefix_cache_misses = 0\n\n"
        "    delivery_distance_m = 0.0\n",
        "step-prefix cache state",
    )

    old_plan = '''                if node == int(stop.node_id):
                    prefix = PrefixPlan((), 0.0, 0.0, battery)
                else:
                    lookahead = first_distinct_later_node(stops, i)
                    snapshot_robot = RobotState(
                        spec=robot.spec,
                        node_id=node,
                        battery_wh=battery,
                        available=False,
                    )
                    start_to_first = exact_distance(node, int(stop.node_id), special_rows)
                    if lookahead is None:
                        prefix = plan_final_target(
                            snapshot_robot,
                            int(stop.node_id),
                            start_to_target_m=start_to_first,
                        )
                    else:
                        first_to_next = exact_distance(
                            int(stop.node_id),
                            int(lookahead),
                            special_rows,
                        )
                        quote = evaluate_pair_from_committed_state(
                            router,
                            snapshot_robot,
                            int(stop.node_id),
                            int(lookahead),
                            start_to_pickup_m=start_to_first,
                            pickup_to_dropoff_m=first_to_next,
                        )
                        prefix = prefix_from_pair_quote(
                            snapshot_robot,
                            quote,
                            int(stop.node_id),
                        )
'''
    new_plan = '''                if node == int(stop.node_id):
                    prefix = PrefixPlan((), 0.0, 0.0, battery)
                else:
                    lookahead = first_distinct_later_node(stops, i)
                    # Prefix routing has no dependence on simulated clock time or
                    # order identity: only physical state, robot spec, target and
                    # one-stop lookahead matter.  Exact float battery values are
                    # used in the key, so cache hits are true identical states.
                    prefix_key = (
                        int(robot.spec.id),
                        int(node),
                        float(battery),
                        int(stop.node_id),
                        None if lookahead is None else int(lookahead),
                    )
                    prefix = step_prefix_cache.get(prefix_key)
                    if prefix is not None:
                        prefix_cache_hits += 1
                    else:
                        prefix_cache_misses += 1
                        snapshot_robot = RobotState(
                            spec=robot.spec,
                            node_id=node,
                            battery_wh=battery,
                            available=False,
                        )
                        start_to_first = exact_distance(node, int(stop.node_id), special_rows)
                        if lookahead is None:
                            prefix = plan_final_target(
                                snapshot_robot,
                                int(stop.node_id),
                                start_to_target_m=start_to_first,
                            )
                        else:
                            first_to_next = exact_distance(
                                int(stop.node_id),
                                int(lookahead),
                                special_rows,
                            )
                            quote = evaluate_pair_from_committed_state(
                                router,
                                snapshot_robot,
                                int(stop.node_id),
                                int(lookahead),
                                start_to_pickup_m=start_to_first,
                                pickup_to_dropoff_m=first_to_next,
                            )
                            prefix = prefix_from_pair_quote(
                                snapshot_robot,
                                quote,
                                int(stop.node_id),
                            )
                        step_prefix_cache[prefix_key] = prefix
'''
    source = _replace_once(source, old_plan, new_plan, "cached exact prefix planning")

    # evaluate_sequence mutates the two cache counters from the surrounding main
    # scope, so declare them nonlocal in that nested function.
    source = _replace_once(
        source,
        "    ) -> SequenceEvaluation:\n"
        "        started = time.perf_counter()\n"
        "        B5_STATS[\"exact_sequence_evaluations\"] += 1.0\n",
        "    ) -> SequenceEvaluation:\n"
        "        nonlocal prefix_cache_hits, prefix_cache_misses\n"
        "        started = time.perf_counter()\n"
        "        B5_STATS[\"exact_sequence_evaluations\"] += 1.0\n",
        "prefix cache counter nonlocal",
    )

    # Reset only the per-decision route-plan cache. Baseline projection cache is
    # deliberately retained across request arrivals when the robot state matches.
    source = _replace_once(
        source,
        "    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:\n"
        "        B5_STATS[\"selection_calls\"] += 1.0\n",
        "    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:\n"
        "        step_prefix_cache.clear()\n"
        "        B5_STATS[\"selection_calls\"] += 1.0\n",
        "per-decision prefix-cache reset",
    )

    # Seed the global incumbent using empty-schedule robots.  This is especially
    # effective in the NYC scenario: it gives a real feasible loss before the
    # expensive busy-route insertion search begins.
    marker = '''        best: InsertionChoice | None = None
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
    seeded = '''        best: InsertionChoice | None = None
        best_delta_loss = math.inf
        best_completion = math.inf
        best_loss = math.inf
        best_distance = math.inf
        new_allowance = (
            delivery_deadline_min(order.request_time_min, direct, order.importance)
            - order.request_time_min
        )

        # Exact best solution among robots with no active delivery schedule.
        # Their only precedence-feasible sequence is [new pickup, new dropoff].
        # Iterate in completion lower-bound order and stop when the request-loss
        # lower bound cannot beat the best empty-robot solution already found.
        for universal_lb, robot_id, snapshot in robot_rows:
            schedule = schedules[robot_id]
            if schedule.orders:
                continue
            empty_loss_lb = delivery_loss(
                max(0.0, universal_lb - order.request_time_min),
                new_allowance,
                order.importance,
            )
            if empty_loss_lb >= best_delta_loss - 1e-9:
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
                evaluation = evaluate_sequence(
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
            if (
                loss < best_delta_loss - 1e-6
                or (
                    math.isclose(loss, best_delta_loss, abs_tol=1e-6, rel_tol=0.0)
                    and (
                        completion < best_completion - 1e-9
                        or (
                            math.isclose(
                                completion, best_completion, abs_tol=1e-9, rel_tol=0.0
                            )
                            and (
                                distance < best_distance - 1e-6
                                or (
                                    math.isclose(
                                        distance, best_distance, abs_tol=1e-6, rel_tol=0.0
                                    )
                                    and (best is None or robot_id < best.robot_id)
                                )
                            )
                        )
                    )
                )
            ):
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
            # Empty-schedule robots were solved exactly in the seed pass.
            if not schedules[robot_id].orders:
                continue
'''
    source = _replace_once(source, marker, seeded, "empty-robot exact incumbent seed")

    # Strengthen the busy-robot bound.  Any insertion preserves the relative
    # order of old stops, so old requests cannot beat the no-insertion sequence
    # lower bound.  Adding the new request's universal loss lower bound is still
    # admissible for the full candidate loss.
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
        "strong busy-robot lower bound",
    )

    # Expose cache behavior in the result so large-run performance is auditable.
    source = _replace_once(
        source,
        '        "optimization": "all-order-loss-lower-bound-plus-baseline-state-cache",\n',
        '        "optimization": "exact-empty-incumbent-plus-loss-bounds-plus-prefix-cache",\n'
        '        "prefix_plan_cache_hits": int(prefix_cache_hits),\n'
        '        "prefix_plan_cache_misses": int(prefix_cache_misses),\n',
        "fast optimization metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    base_source = corrected.TARGET.read_text(encoding="utf-8")
    source = corrected._patch_source(base_source)
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
