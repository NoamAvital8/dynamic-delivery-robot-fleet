from __future__ import annotations

"""NYC Benchmark 4: full Myopic-A earliest completion with busy robots.

Benchmark 4 extends Benchmark 3 by allowing a new request to be assigned to a
robot that is already serving (or already scheduled to serve) other requests.
The request is appended to that robot's delivery schedule and starts after all
earlier committed deliveries finish.

Selection is still myopic/reactive: among all capable robots, choose the robot
with the smallest planned request completion time using only currently known
commitments. Planned delivery time includes battery-feasible charger detours and
charging duration but deliberately does not predict future charger queue waits.

The expensive candidate comparison is batched by robot type using the tiny
compiled charger meta graph. The canonical battery router is called only for the
predicted winner (and exact ties / numeric fallbacks).
"""

from collections import deque
import math
import sys
import time
import types

import fast_battery_meta as fast_meta
import run_nyc_benchmark2_fast as fast


B4_STATS: dict[str, float] = {
    "selection_calls": 0.0,
    "candidates_considered": 0.0,
    "route_evaluations": 0.0,
    "route_evaluate_seconds": 0.0,
    "lower_bound_pruned": 0.0,
    "no_feasible_route": 0.0,
    "assignments_to_busy_robots": 0.0,
    "max_waiting_orders_on_robot": 0.0,
}


def _load_benchmark4_namespace() -> dict[str, object]:
    source = fast.TARGET.read_text(encoding="utf-8")

    old_battery_check = "if robot.battery_wh < -1e-6:"
    new_battery_check = f"if robot.battery_wh < -{fast.BATTERY_EPS_WH}:"
    count = source.count(old_battery_check)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one Benchmark 2 delivery battery check, found {count}"
        )
    source = source.replace(old_battery_check, new_battery_check)

    source = source.replace(
        'default=ROOT / "benchmark2_results.json",',
        'default=ROOT / "benchmark4_results.json",',
        1,
    )
    source = source.replace('"benchmark": 2,', '"benchmark": 4,', 1)
    source = source.replace(
        '"policy": "nearest_available_robot_with_interruptible_return_to_charge",',
        '"policy": "myopic_a_earliest_completion_including_busy_robots",',
        1,
    )
    source = source.replace(
        'print("BENCHMARK2_RESULTS_JSON")',
        'print("BENCHMARK4_RESULTS_JSON")',
        1,
    )

    state_anchor = (
        "    charge_token: dict[int, int] = {robot.spec.id: 0 for robot in robots}\n"
    )
    if state_anchor not in source:
        raise RuntimeError("could not locate Benchmark 2 runtime state")
    source = source.replace(
        state_anchor,
        state_anchor
        + '''    waiting_plans: dict[int, deque[DeliveryPlan]] = {
        robot.spec.id: deque() for robot in robots
    }
    active_plan_started_at: dict[int, float] = {}
    active_delivery_queue_delay: dict[int, float] = {}

''',
        1,
    )

    charge_delay_anchor = (
        "        if request.arrival_time_min + EPS < now:\n"
        "            metrics.queue_waits_min.append(now - request.arrival_time_min)\n"
    )
    if charge_delay_anchor not in source:
        raise RuntimeError("could not locate charger queue-wait accounting")
    source = source.replace(
        charge_delay_anchor,
        charge_delay_anchor
        + '''            if request.purpose == "delivery":
                active_delivery_queue_delay[request.robot_id] = (
                    active_delivery_queue_delay.get(request.robot_id, 0.0)
                    + (now - request.arrival_time_min)
                )
''',
        1,
    )

    cand_start = source.find(
        "    def policy_candidates(order: Order, now: float) -> Iterator[Candidate]:\n"
    )
    selector_end_marker = "    def advance_delivery(robot_id: int, now: float) -> None:\n"
    cand_end = source.find(selector_end_marker, cand_start)
    if cand_start < 0 or cand_end < 0:
        raise RuntimeError("could not locate Benchmark 2 candidate/selector block")

    candidate_block = r'''    def plan_duration_min(plan: DeliveryPlan) -> float:
        return (
            plan.route.total_time_min
            + PICKUP_HANDLING_MIN
            + DROPOFF_HANDLING_MIN
        )

    def queued_delivery_wait_elapsed(robot_id: int, now: float) -> float:
        for state in station_state.values():
            for request in state.queue:
                if (
                    request.robot_id == robot_id
                    and request.purpose == "delivery"
                ):
                    return max(0.0, now - request.arrival_time_min)
        return 0.0

    def delivery_tail_snapshot(
        robot: RobotState,
        now: float,
    ) -> tuple[int, float, float, float] | None:
        robot_id = robot.spec.id

        if robot_id in deferred:
            current_plan = deferred[robot_id].plan
            tail_time = (
                deferred[robot_id].expected_start_time_min
                + plan_duration_min(current_plan)
            )
            tail_node = int(current_plan.route.dropoff_node)
            tail_battery = float(current_plan.route.arrival_battery_wh)
        elif robot_id in delivery_plans:
            current_plan = delivery_plans[robot_id]
            started = active_plan_started_at.get(
                robot_id,
                current_plan.route_start_time_min,
            )
            known_queue_delay = (
                active_delivery_queue_delay.get(robot_id, 0.0)
                + queued_delivery_wait_elapsed(robot_id, now)
            )
            tail_time = started + plan_duration_min(current_plan) + known_queue_delay
            tail_time = max(float(now), float(tail_time))
            tail_node = int(current_plan.route.dropoff_node)
            tail_battery = float(current_plan.route.arrival_battery_wh)
        else:
            return None

        for waiting_plan in waiting_plans[robot_id]:
            tail_time += plan_duration_min(waiting_plan)
            tail_node = int(waiting_plan.route.dropoff_node)
            tail_battery = float(waiting_plan.route.arrival_battery_wh)

        return tail_node, float(tail_time), tail_battery, 0.0

    def availability_snapshot(
        robot: RobotState,
        now: float,
    ) -> tuple[int, float, float, float]:
        delivery_tail = delivery_tail_snapshot(robot, now)
        if delivery_tail is not None:
            return delivery_tail
        return background_snapshot(robot, now)

    def policy_candidates(order: Order, now: float) -> Iterator[Candidate]:
        candidates = [robot for robot in robots if robot.can_hold(order.item)]
        if not candidates:
            return

        by_node: dict[int, list[tuple[RobotState, float, float, float]]] = {}
        for robot in candidates:
            node, start_time, battery, residual_m = availability_snapshot(robot, now)
            by_node.setdefault(node, []).append(
                (robot, start_time, battery, residual_m)
            )
        for values in by_node.values():
            values.sort(key=lambda item: item[0].spec.id)

        buffered: list[tuple[float, int, Candidate]] = []
        for nearest in oracle.iter_nearest_targets(order.pickup_node, by_node):
            base = float(nearest.distance_m)
            while buffered and buffered[0][0] < base - 1e-9:
                yield heapq.heappop(buffered)[2]

            for robot, start_time, battery, residual_m in by_node[int(nearest.node_id)]:
                candidate = Candidate(
                    robot=robot,
                    policy_distance_m=residual_m + base,
                    route_start_node=int(nearest.node_id),
                    route_start_time_min=float(start_time),
                    route_start_battery_wh=float(battery),
                    decision_to_pickup_m=base,
                )
                heapq.heappush(
                    buffered,
                    (candidate.policy_distance_m, robot.spec.id, candidate),
                )

        while buffered:
            yield heapq.heappop(buffered)[2]

    def nearest_feasible_assignment(
        order: Order,
        now: float,
    ) -> tuple[Candidate, BatteryRouteQuote, float] | None:
        direct = get_direct_distance(order)
        candidates = list(policy_candidates(order, now))
        B4_STATS["selection_calls"] += 1.0
        B4_STATS["candidates_considered"] += float(len(candidates))
        if not candidates:
            return None

        handling = PICKUP_HANDLING_MIN + DROPOFF_HANDLING_MIN
        ranked = fast_meta.rank_candidates(
            router,
            candidates,
            pickup_node=order.pickup_node,
            dropoff_node=order.dropoff_node,
            pickup_to_dropoff_m=direct,
            now_min=now,
            handling_min=handling,
        )
        finite_ranked = [row for row in ranked if math.isfinite(row.completion_time_min)]
        B4_STATS["no_feasible_route"] += float(len(ranked) - len(finite_ranked))
        if not finite_ranked:
            return None

        best: tuple[Candidate, BatteryRouteQuote, float] | None = None
        best_completion = math.inf
        best_robot_id = math.inf

        for index, scored in enumerate(finite_ranked):
            if scored.completion_time_min > best_completion + 1e-9:
                B4_STATS["lower_bound_pruned"] += float(len(finite_ranked) - index)
                break

            candidate = scored.candidate
            snapshot = RobotState(
                spec=candidate.robot.spec,
                node_id=candidate.route_start_node,
                battery_wh=candidate.route_start_battery_wh,
            )

            eval_started = time.perf_counter()
            B4_STATS["route_evaluations"] += 1.0
            try:
                quote = router.evaluate(
                    snapshot,
                    order.pickup_node,
                    order.dropoff_node,
                    start_to_pickup_m=candidate.decision_to_pickup_m,
                    pickup_to_dropoff_m=direct,
                )
            except NoFeasibleBatteryRoute:
                B4_STATS["no_feasible_route"] += 1.0
                continue
            finally:
                B4_STATS["route_evaluate_seconds"] += (
                    time.perf_counter() - eval_started
                )

            if not math.isclose(
                quote.total_time_min,
                scored.route_time_min,
                rel_tol=2e-6,
                abs_tol=1e-2,
            ):
                raise RuntimeError(
                    "fast battery meta score disagrees with canonical router: "
                    f"robot={scored.robot_id} fast={scored.route_time_min} "
                    f"canonical={quote.total_time_min}"
                )

            start_delay = max(0.0, candidate.route_start_time_min - now)
            exact_completion = start_delay + quote.total_time_min + handling
            robot_id = scored.robot_id
            if (
                exact_completion < best_completion - 1e-12
                or (
                    math.isclose(
                        exact_completion,
                        best_completion,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and robot_id < best_robot_id
                )
            ):
                best_completion = exact_completion
                best_robot_id = robot_id
                best = (candidate, quote, direct)

        return best

'''
    source = source[:cand_start] + candidate_block + source[cand_end:]

    activate_anchor = (
        "        delivery_plans[robot_id] = plan\n"
        "        deferred.pop(robot_id, None)\n"
        "        advance_delivery(robot_id, now)\n"
    )
    if activate_anchor not in source:
        raise RuntimeError("could not locate assignment activation block")
    source = source.replace(
        activate_anchor,
        '''        delivery_plans[robot_id] = plan
        deferred.pop(robot_id, None)
        active_plan_started_at[robot_id] = float(now)
        active_delivery_queue_delay[robot_id] = 0.0
        plan.route_start_time_min = float(now)
        advance_delivery(robot_id, now)
''',
        1,
    )

    assign_tail = '''        if robot.activity is RobotActivity.MOVING:
            deferred[robot_id] = DeferredAssignment(
                plan=plan,
                expected_start_node=candidate.route_start_node,
                expected_start_time_min=candidate.route_start_time_min,
            )
            robot.available = False
            metrics.interrupted_repositioning += 1
            return

        cancel_stationary_background(robot, now)
        if not math.isclose(
            robot.battery_wh,
            candidate.route_start_battery_wh,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("assignment battery snapshot changed during interruption")
        activate_assignment(robot, plan, now)
'''
    if assign_tail not in source:
        raise RuntimeError("could not locate Benchmark 2 assignment commit logic")
    source = source.replace(
        assign_tail,
        '''        already_delivery_committed = (
            robot_id in delivery_plans
            or robot_id in deferred
            or bool(waiting_plans[robot_id])
        )
        if already_delivery_committed:
            waiting_plans[robot_id].append(plan)
            B4_STATS["assignments_to_busy_robots"] += 1.0
            B4_STATS["max_waiting_orders_on_robot"] = max(
                B4_STATS["max_waiting_orders_on_robot"],
                float(len(waiting_plans[robot_id])),
            )
            return

        if robot.activity is RobotActivity.MOVING:
            deferred[robot_id] = DeferredAssignment(
                plan=plan,
                expected_start_node=candidate.route_start_node,
                expected_start_time_min=candidate.route_start_time_min,
            )
            robot.available = False
            metrics.interrupted_repositioning += 1
            return

        cancel_stationary_background(robot, now)
        if not math.isclose(
            robot.battery_wh,
            candidate.route_start_battery_wh,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("assignment battery snapshot changed during interruption")
        activate_assignment(robot, plan, now)
''',
        1,
    )

    complete_anchor = '''            robot.current_order_id = None
            robot.activity = RobotActivity.IDLE
            robot.available = True
            delivered += 1

            if pending:
                dispatch_pending(now)
            if robot_id not in delivery_plans and robot_id not in deferred:
                begin_background_return(robot, now)
'''
    if complete_anchor not in source:
        raise RuntimeError("could not locate delivery completion scheduling block")
    source = source.replace(
        complete_anchor,
        '''            robot.current_order_id = None
            robot.activity = RobotActivity.IDLE
            robot.available = True
            active_plan_started_at.pop(robot_id, None)
            active_delivery_queue_delay.pop(robot_id, None)
            delivered += 1

            if waiting_plans[robot_id]:
                next_plan = waiting_plans[robot_id].popleft()
                activate_assignment(robot, next_plan, now)
                if pending:
                    dispatch_pending(now)
            else:
                if pending:
                    dispatch_pending(now)
                if robot_id not in delivery_plans and robot_id not in deferred:
                    begin_background_return(robot, now)
''',
        1,
    )

    done_anchor = '''            and not deferred
        ):
            break
'''
    if done_anchor not in source:
        raise RuntimeError("could not locate completion criterion")
    source = source.replace(
        done_anchor,
        '''            and not deferred
            and not any(waiting_plans.values())
        ):
            break
''',
        1,
    )

    incomplete_anchor = '''    if pending or delivery_plans or deferred or delivered != len(scenario.orders):
        raise RuntimeError(
            "simulation ended incomplete: "
            f"delivered={delivered}, pending={len(pending)}, "
            f"active={len(delivery_plans)}, deferred={len(deferred)}"
        )
'''
    if incomplete_anchor not in source:
        raise RuntimeError("could not locate incomplete-simulation check")
    source = source.replace(
        incomplete_anchor,
        '''    if (
        pending
        or delivery_plans
        or deferred
        or any(waiting_plans.values())
        or delivered != len(scenario.orders)
    ):
        raise RuntimeError(
            "simulation ended incomplete: "
            f"delivered={delivered}, pending={len(pending)}, "
            f"active={len(delivery_plans)}, deferred={len(deferred)}, "
            f"waiting={sum(len(q) for q in waiting_plans.values())}"
        )
''',
        1,
    )

    busy_anchor = (
        '                f"pending={len(pending)} busy={len(delivery_plans)+len(deferred)} "\n'
    )
    if busy_anchor in source:
        source = source.replace(
            busy_anchor,
            '                f"pending={len(pending)} busy={len(delivery_plans)+len(deferred)} "\n'
            '                f"scheduled={sum(len(q) for q in waiting_plans.values())} "\n',
            1,
        )

    stats_anchor = '        "routing_index_build_seconds": index_seconds,\n'
    if stats_anchor not in source:
        raise RuntimeError("could not locate result routing stats")
    stats_fields = stats_anchor + '''        "candidate_selection_calls": int(B4_STATS["selection_calls"]),
        "candidate_routes_considered": int(B4_STATS["candidates_considered"]),
        "candidate_route_evaluations": int(B4_STATS["route_evaluations"]),
        "candidate_route_evaluate_seconds": float(B4_STATS["route_evaluate_seconds"]),
        "candidate_route_evaluate_mean_ms": (
            1000.0 * B4_STATS["route_evaluate_seconds"]
            / max(1.0, B4_STATS["route_evaluations"])
        ),
        "candidate_lower_bound_pruned": int(B4_STATS["lower_bound_pruned"]),
        "candidate_no_feasible_route": int(B4_STATS["no_feasible_route"]),
        "mean_candidates_per_selection": (
            B4_STATS["candidates_considered"]
            / max(1.0, B4_STATS["selection_calls"])
        ),
        "mean_route_evaluations_per_selection": (
            B4_STATS["route_evaluations"]
            / max(1.0, B4_STATS["selection_calls"])
        ),
        "battery_meta_batch_seconds": float(fast_meta.FAST_META_STATS["seconds"]),
        "battery_meta_batch_mean_ms": (
            1000.0 * fast_meta.FAST_META_STATS["seconds"]
            / max(1.0, fast_meta.FAST_META_STATS["calls"])
        ),
        "battery_meta_type_solves": int(fast_meta.FAST_META_STATS["type_solves"]),
        "assignments_to_busy_robots": int(B4_STATS["assignments_to_busy_robots"]),
        "max_waiting_orders_on_robot": int(B4_STATS["max_waiting_orders_on_robot"]),
'''
    source = source.replace(stats_anchor, stats_fields, 1)

    module_name = "benchmark4_fast_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(fast.TARGET)
    module.__package__ = None
    module.__dict__["B4_STATS"] = B4_STATS
    module.__dict__["deque"] = deque
    module.__dict__["fast_meta"] = fast_meta
    sys.modules[module_name] = module
    exec(compile(source, str(fast.TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    fast._install_in_memory_graph_initialization()
    distance_row = fast._install_compiled_distance_backend()
    namespace = _load_benchmark4_namespace()
    fast._install_ordered_charge_binding(namespace)
    fast._install_fast_direct_distance(namespace, distance_row)

    started = time.perf_counter()
    try:
        namespace["main"]()
    finally:
        elapsed = time.perf_counter() - started
        evals = int(B4_STATS["route_evaluations"])
        eval_seconds = float(B4_STATS["route_evaluate_seconds"])
        meta_calls = int(fast_meta.FAST_META_STATS["calls"])
        meta_seconds = float(fast_meta.FAST_META_STATS["seconds"])
        print(
            "BENCHMARK4_POLICY_STATS "
            f"selection_calls={int(B4_STATS['selection_calls'])} "
            f"candidates={int(B4_STATS['candidates_considered'])} "
            f"route_evaluations={evals} "
            f"route_eval_seconds={eval_seconds:.3f} "
            f"route_eval_mean_ms={1000.0 * eval_seconds / max(1, evals):.3f} "
            f"meta_batch_seconds={meta_seconds:.3f} "
            f"meta_batch_mean_ms={1000.0 * meta_seconds / max(1, meta_calls):.3f} "
            f"meta_type_solves={int(fast_meta.FAST_META_STATS['type_solves'])} "
            f"pruned={int(B4_STATS['lower_bound_pruned'])} "
            f"no_feasible={int(B4_STATS['no_feasible_route'])} "
            f"busy_assignments={int(B4_STATS['assignments_to_busy_robots'])} "
            f"max_waiting={int(B4_STATS['max_waiting_orders_on_robot'])}",
            flush=True,
        )
        print(
            "FAST_ROUTING_STATS "
            f"compiled_dijkstra_calls={fast._fast_dijkstra_calls} "
            f"cache_hits={fast._fast_dijkstra_cache_hits} "
            f"compiled_dijkstra_seconds={fast._fast_dijkstra_seconds:.3f} "
            f"runner_seconds={elapsed:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
