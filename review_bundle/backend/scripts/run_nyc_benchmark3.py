from __future__ import annotations

"""NYC Benchmark 3: fastest planned completion among non-busy robots.

Benchmark 3 preserves Benchmark 2's physical/simulator semantics, including
interruptible background return-to-charge behavior. The only policy change is
robot selection: among robots that are not doing delivery work, choose the
battery-feasible robot with the smallest planned time from now until the new
order is dropped off.

Performance strategy:
- reuse Benchmark 2's compiled SciPy/CSR pickup-rooted Dijkstra row;
- solve the charger meta problem in a tiny compiled batch once per robot type;
- score all candidates of that type with vectorized NumPy operations;
- call the canonical BatteryFeasibleRouter.evaluate() only for the predicted
  winner (and exact ties / numeric fallbacks).

Future charger queue waiting is deliberately not predicted; actual queues are
still simulated exactly after assignment, just as in Benchmark 2.
"""

import math
import sys
import time
import types

import fast_battery_meta as fast_meta
import run_nyc_benchmark2_fast as fast


B3_STATS: dict[str, float] = {
    "selection_calls": 0.0,
    "candidates_considered": 0.0,
    "route_evaluations": 0.0,
    "route_evaluate_seconds": 0.0,
    "lower_bound_pruned": 0.0,
    "no_feasible_route": 0.0,
}


def _load_benchmark3_namespace() -> dict[str, object]:
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
        'default=ROOT / "benchmark3_results.json",',
        1,
    )
    source = source.replace('"benchmark": 2,', '"benchmark": 3,', 1)
    source = source.replace(
        '"policy": "nearest_available_robot_with_interruptible_return_to_charge",',
        '"policy": "fastest_planned_completion_non_busy_with_interruptible_return_to_charge",',
        1,
    )
    source = source.replace(
        'print("BENCHMARK2_RESULTS_JSON")',
        'print("BENCHMARK3_RESULTS_JSON")',
        1,
    )

    start_marker = "    def nearest_feasible_assignment(\n"
    end_marker = "    def advance_delivery(robot_id: int, now: float) -> None:\n"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("could not locate Benchmark 2 assignment selector")

    replacement = '''    def nearest_feasible_assignment(\n        order: Order,\n        now: float,\n    ) -> tuple[Candidate, BatteryRouteQuote, float] | None:\n        direct = get_direct_distance(order)\n        candidates = list(policy_candidates(order, now))\n        B3_STATS["selection_calls"] += 1.0\n        B3_STATS["candidates_considered"] += float(len(candidates))\n        if not candidates:\n            return None\n\n        handling = PICKUP_HANDLING_MIN + DROPOFF_HANDLING_MIN\n        ranked = fast_meta.rank_candidates(\n            router,\n            candidates,\n            pickup_node=order.pickup_node,\n            dropoff_node=order.dropoff_node,\n            pickup_to_dropoff_m=direct,\n            now_min=now,\n            handling_min=handling,\n        )\n        finite_ranked = [row for row in ranked if math.isfinite(row.completion_time_min)]\n        B3_STATS["no_feasible_route"] += float(len(ranked) - len(finite_ranked))\n        if not finite_ranked:\n            return None\n\n        best: tuple[Candidate, BatteryRouteQuote, float] | None = None\n        best_completion = math.inf\n        best_robot_id = math.inf\n\n        for index, scored in enumerate(finite_ranked):\n            if scored.completion_time_min > best_completion + 1e-9:\n                B3_STATS["lower_bound_pruned"] += float(len(finite_ranked) - index)\n                break\n\n            candidate = scored.candidate\n            snapshot = RobotState(\n                spec=candidate.robot.spec,\n                node_id=candidate.route_start_node,\n                battery_wh=candidate.route_start_battery_wh,\n            )\n\n            eval_started = time.perf_counter()\n            B3_STATS["route_evaluations"] += 1.0\n            try:\n                quote = router.evaluate(\n                    snapshot,\n                    order.pickup_node,\n                    order.dropoff_node,\n                    start_to_pickup_m=candidate.decision_to_pickup_m,\n                    pickup_to_dropoff_m=direct,\n                )\n            except NoFeasibleBatteryRoute:\n                B3_STATS["no_feasible_route"] += 1.0\n                continue\n            finally:\n                B3_STATS["route_evaluate_seconds"] += (\n                    time.perf_counter() - eval_started\n                )\n\n            if not math.isclose(\n                quote.total_time_min,\n                scored.route_time_min,\n                rel_tol=2e-6,\n                abs_tol=1e-2,\n            ):\n                raise RuntimeError(\n                    "fast battery meta score disagrees with canonical router: "\n                    f"robot={scored.robot_id} fast={scored.route_time_min} "\n                    f"canonical={quote.total_time_min}"\n                )\n\n            start_delay = max(0.0, candidate.route_start_time_min - now)\n            exact_completion = start_delay + quote.total_time_min + handling\n            robot_id = scored.robot_id\n            if (\n                exact_completion < best_completion - 1e-12\n                or (\n                    math.isclose(\n                        exact_completion,\n                        best_completion,\n                        rel_tol=0.0,\n                        abs_tol=1e-12,\n                    )\n                    and robot_id < best_robot_id\n                )\n            ):\n                best_completion = exact_completion\n                best_robot_id = robot_id\n                best = (candidate, quote, direct)\n\n        return best\n\n'''
    source = source[:start] + replacement + source[end:]

    stats_anchor = '        "routing_index_build_seconds": index_seconds,\n'
    if stats_anchor not in source:
        raise RuntimeError("could not locate Benchmark 2 result routing stats")
    stats_fields = stats_anchor + '''        "candidate_selection_calls": int(B3_STATS["selection_calls"]),\n        "candidate_routes_considered": int(B3_STATS["candidates_considered"]),\n        "candidate_route_evaluations": int(B3_STATS["route_evaluations"]),\n        "candidate_route_evaluate_seconds": float(B3_STATS["route_evaluate_seconds"]),\n        "candidate_route_evaluate_mean_ms": (\n            1000.0 * B3_STATS["route_evaluate_seconds"]\n            / max(1.0, B3_STATS["route_evaluations"])\n        ),\n        "candidate_lower_bound_pruned": int(B3_STATS["lower_bound_pruned"]),\n        "candidate_no_feasible_route": int(B3_STATS["no_feasible_route"]),\n        "mean_candidates_per_selection": (\n            B3_STATS["candidates_considered"]\n            / max(1.0, B3_STATS["selection_calls"])\n        ),\n        "mean_route_evaluations_per_selection": (\n            B3_STATS["route_evaluations"]\n            / max(1.0, B3_STATS["selection_calls"])\n        ),\n        "battery_meta_batch_seconds": float(fast_meta.FAST_META_STATS["seconds"]),\n        "battery_meta_batch_mean_ms": (\n            1000.0 * fast_meta.FAST_META_STATS["seconds"]\n            / max(1.0, fast_meta.FAST_META_STATS["calls"])\n        ),\n        "battery_meta_type_solves": int(fast_meta.FAST_META_STATS["type_solves"]),\n'''
    source = source.replace(stats_anchor, stats_fields, 1)

    module_name = "benchmark3_fast_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(fast.TARGET)
    module.__package__ = None
    module.__dict__["B3_STATS"] = B3_STATS
    module.__dict__["fast_meta"] = fast_meta
    sys.modules[module_name] = module
    exec(compile(source, str(fast.TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    fast._install_in_memory_graph_initialization()
    distance_row = fast._install_compiled_distance_backend()
    namespace = _load_benchmark3_namespace()
    fast._install_ordered_charge_binding(namespace)
    fast._install_fast_direct_distance(namespace, distance_row)

    started = time.perf_counter()
    try:
        namespace["main"]()
    finally:
        elapsed = time.perf_counter() - started
        evals = int(B3_STATS["route_evaluations"])
        eval_seconds = float(B3_STATS["route_evaluate_seconds"])
        meta_calls = int(fast_meta.FAST_META_STATS["calls"])
        meta_seconds = float(fast_meta.FAST_META_STATS["seconds"])
        print(
            "BENCHMARK3_POLICY_STATS "
            f"selection_calls={int(B3_STATS['selection_calls'])} "
            f"candidates={int(B3_STATS['candidates_considered'])} "
            f"route_evaluations={evals} "
            f"route_eval_seconds={eval_seconds:.3f} "
            f"route_eval_mean_ms={1000.0 * eval_seconds / max(1, evals):.3f} "
            f"meta_batch_seconds={meta_seconds:.3f} "
            f"meta_batch_mean_ms={1000.0 * meta_seconds / max(1, meta_calls):.3f} "
            f"meta_type_solves={int(fast_meta.FAST_META_STATS['type_solves'])} "
            f"pruned={int(B3_STATS['lower_bound_pruned'])} "
            f"no_feasible={int(B3_STATS['no_feasible_route'])}",
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
