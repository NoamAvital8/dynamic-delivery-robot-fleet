from __future__ import annotations

"""Benchmark 5 with incremental deadline-aware loss as the insertion objective.

This runner reuses Benchmark 5's simulator and interruptible insertion machinery,
but corrects the decision objective. For each robot/insertion candidate it
minimizes

    Delta L = projected_loss_after_insertion - projected_loss_before_insertion

for the affected robot. Losses of every other robot are unchanged by that
candidate and therefore cancel from the global comparison.

The optimized version keeps the same exact decision objective while adding two
important accelerations for large scenarios:

1. Candidate pruning uses a lower bound on the *total loss of every affected
   order*, not only the new order. The bound ignores charging time and uses
   shortest-path/haversine distance lower bounds, so it is admissible.
2. Baseline route loss is cached per robot decision state. While a robot is
   traversing the same immutable edge, many request arrivals share the same
   decision node/time/battery and remaining stop sequence, so the expensive
   baseline projection can be reused exactly.
"""

import math
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

TARGET = SCRIPTS / "run_nyc_benchmark5.py"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def _patch_source(source: str) -> str:
    source = _replace_once(
        source,
        "Benchmark 5 keeps Benchmark 4's myopic objective -- minimize the newly arrived\n"
        "request's planned completion time -- but removes B4's append-only restriction.\n",
        "Benchmark 5 removes B4's append-only restriction and chooses the robot/insertion\n"
        "that minimizes incremental projected deadline-aware loss over all affected orders.\n",
        "docstring objective",
    )

    # Allow evaluate_sequence to project an existing route without designating a
    # newly inserted order. This is used to compute the baseline loss.
    source = _replace_once(
        source,
        "        new_order_id: int,\n"
        "        special_rows: dict[int, np.ndarray],\n"
        "    ) -> SequenceEvaluation:\n",
        "        new_order_id: int | None,\n"
        "        special_rows: dict[int, np.ndarray],\n"
        "    ) -> SequenceEvaluation:\n",
        "optional baseline evaluation id",
    )
    source = _replace_once(
        source,
        "            if new_order_id not in completion:\n"
        "                raise RuntimeError(\"candidate sequence never completed the new order\")\n",
        "            if new_order_id is not None and new_order_id not in completion:\n"
        "                raise RuntimeError(\"candidate sequence never completed the new order\")\n",
        "optional completion check",
    )
    source = _replace_once(
        source,
        "                new_completion_time_min=float(completion[new_order_id]),\n",
        "                new_completion_time_min=(\n"
        "                    float(completion[new_order_id])\n"
        "                    if new_order_id is not None\n"
        "                    else math.inf\n"
        "                ),\n",
        "optional completion value",
    )

    # One-entry-per-robot baseline cache. Keeping only the latest decision state
    # bounds memory while capturing the common case where several requests arrive
    # before the robot finishes its current immutable graph edge.
    source = _replace_once(
        source,
        "    direct_distance_cache: dict[int, float] = {}\n\n"
        "    delivery_distance_m = 0.0\n",
        "    direct_distance_cache: dict[int, float] = {}\n"
        "    baseline_projection_cache: dict[int, tuple[object, float]] = {}\n\n"
        "    delivery_distance_m = 0.0\n",
        "baseline projection cache",
    )

    # Strengthen the admissible candidate lower bound. The previous corrected
    # B5 bounded only the new order's loss; on NYC that was much too weak and
    # caused millions of unnecessary exact multi-stop battery evaluations.
    old_lb_function = '''    def sequence_completion_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
        now = float(snapshot.time_min)
        node = int(snapshot.node_id)
        for stop in stops:
            distance = lower_bound_distance(node, int(stop.node_id), special_rows)
            now += distance / robot.spec.speed_mps / 60.0
            now += PICKUP_HANDLING_MIN if stop.kind == "pickup" else DROPOFF_HANDLING_MIN
            node = int(stop.node_id)
            if stop.kind == "dropoff" and stop.order_id == new_order_id:
                return float(now)
        return math.inf

    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:
'''
    new_lb_function = '''    def sequence_completion_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        new_order_id: int,
        special_rows: dict[int, np.ndarray],
    ) -> float:
        now = float(snapshot.time_min)
        node = int(snapshot.node_id)
        for stop in stops:
            distance = lower_bound_distance(node, int(stop.node_id), special_rows)
            now += distance / robot.spec.speed_mps / 60.0
            now += PICKUP_HANDLING_MIN if stop.kind == "pickup" else DROPOFF_HANDLING_MIN
            node = int(stop.node_id)
            if stop.kind == "dropoff" and stop.order_id == new_order_id:
                return float(now)
        return math.inf

    def sequence_loss_lower_bound(
        robot: RobotState,
        snapshot: DecisionSnapshot,
        stops: tuple[ServiceStop, ...],
        order_states: dict[int, ActiveOrder],
        special_rows: dict[int, np.ndarray],
    ) -> float:
        """Admissible lower bound on total projected loss for a stop sequence.

        Travel uses exact distances when already available and otherwise a
        shortest-path row for the new request or haversine distance. Charging
        time is omitted. Therefore every projected drop-off is no later than in
        the exact battery-feasible evaluation, and delivery_loss is monotone in
        completion time.
        """
        projected_time = float(snapshot.time_min)
        node = int(snapshot.node_id)
        lower_loss = 0.0
        for stop in stops:
            distance = lower_bound_distance(node, int(stop.node_id), special_rows)
            projected_time += distance / robot.spec.speed_mps / 60.0
            projected_time += (
                PICKUP_HANDLING_MIN if stop.kind == "pickup" else DROPOFF_HANDLING_MIN
            )
            node = int(stop.node_id)
            if stop.kind == "dropoff":
                state = order_states[stop.order_id]
                allowance = state.deadline_min - state.order.request_time_min
                lower_loss += delivery_loss(
                    projected_time - state.order.request_time_min,
                    allowance,
                    state.order.importance,
                )
        return float(lower_loss)

    def choose_insertion(order: Order, now: float) -> InsertionChoice | None:
'''
    source = _replace_once(
        source,
        old_lb_function,
        new_lb_function,
        "total-loss lower bound helper",
    )

    source = _replace_once(
        source,
        "        best: InsertionChoice | None = None\n"
        "        best_completion = math.inf\n"
        "        best_loss = math.inf\n"
        "        best_distance = math.inf\n",
        "        best: InsertionChoice | None = None\n"
        "        best_delta_loss = math.inf\n"
        "        best_completion = math.inf\n"
        "        best_loss = math.inf\n"
        "        best_distance = math.inf\n"
        "        new_allowance = (\n"
        "            delivery_deadline_min(order.request_time_min, direct, order.importance)\n"
        "            - order.request_time_min\n"
        "        )\n",
        "best score state",
    )

    source = _replace_once(
        source,
        "        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):\n"
        "            if universal_lb >= best_completion - 1e-9:\n"
        "                B5_STATS[\"robots_lb_pruned\"] += float(len(robot_rows) - robot_index)\n"
        "                break\n"
        "            B5_STATS[\"robots_considered\"] += 1.0\n",
        "        for robot_index, (universal_lb, robot_id, snapshot) in enumerate(robot_rows):\n"
        "            B5_STATS[\"robots_considered\"] += 1.0\n",
        "old completion-based robot pruning",
    )

    # Compute baseline exactly, but reuse it when the robot's decision state and
    # remaining sequence are unchanged. Baseline does not depend on the new
    # request's two distance rows, so use an empty special-row map to maximize
    # cache reuse without changing the route semantics.
    source = _replace_once(
        source,
        "            order_states = dict(schedule.orders)\n"
        "            order_states[order.id] = new_state\n\n"
        "            candidates: list[tuple[float, int, int, tuple[ServiceStop, ...]]] = []\n",
        "            order_states = dict(schedule.orders)\n\n"
        "            # Compare candidates by their change to the global objective.\n"
        "            # Other robots are identical across alternatives, so they cancel.\n"
        "            baseline_loss = 0.0\n"
        "            if existing:\n"
        "                baseline_key = (\n"
        "                    existing,\n"
        "                    int(snapshot.node_id),\n"
        "                    float(snapshot.time_min),\n"
        "                    round(float(snapshot.battery_wh), 6),\n"
        "                )\n"
        "                cached_baseline = baseline_projection_cache.get(robot_id)\n"
        "                if cached_baseline is not None and cached_baseline[0] == baseline_key:\n"
        "                    baseline_loss = float(cached_baseline[1])\n"
        "                else:\n"
        "                    baseline_evaluation = evaluate_sequence(\n"
        "                        robot, snapshot, existing, order_states, None, {}\n"
        "                    )\n"
        "                    baseline_loss = float(baseline_evaluation.projected_loss)\n"
        "                    baseline_projection_cache[robot_id] = (baseline_key, baseline_loss)\n\n"
        "            robot_new_loss_lb = delivery_loss(\n"
        "                max(0.0, universal_lb - order.request_time_min),\n"
        "                new_allowance,\n"
        "                order.importance,\n"
        "            )\n"
        "            robot_delta_lb = robot_new_loss_lb - baseline_loss\n"
        "            if robot_delta_lb >= best_delta_loss - 1e-9:\n"
        "                B5_STATS[\"robots_lb_pruned\"] += 1.0\n"
        "                continue\n\n"
        "            order_states[order.id] = new_state\n\n"
        "            candidates: list[\n"
        "                tuple[float, float, int, int, tuple[ServiceStop, ...]]\n"
        "            ] = []\n",
        "cached baseline incremental loss",
    )

    old_candidates = '''                lb = sequence_completion_lower_bound(
                    robot,
                    snapshot,
                    stops,
                    order.id,
                    special_rows,
                )
                if lb >= best_completion - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += 1.0
                    continue
                candidates.append((lb, pickup_pos, dropoff_pos, stops))

            candidates.sort(key=lambda row: (row[0], row[1], row[2]))
            for candidate_index, (lb, pickup_pos, dropoff_pos, stops) in enumerate(candidates):
                if lb >= best_completion - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += float(
                        len(candidates) - candidate_index
                    )
                    break
'''
    new_candidates = '''                completion_lb = sequence_completion_lower_bound(
                    robot,
                    snapshot,
                    stops,
                    order.id,
                    special_rows,
                )
                projected_loss_lb = sequence_loss_lower_bound(
                    robot,
                    snapshot,
                    stops,
                    order_states,
                    special_rows,
                )
                delta_lb = projected_loss_lb - baseline_loss
                if delta_lb >= best_delta_loss - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += 1.0
                    continue
                candidates.append(
                    (delta_lb, completion_lb, pickup_pos, dropoff_pos, stops)
                )

            candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
            for candidate_index, (
                delta_lb,
                completion_lb,
                pickup_pos,
                dropoff_pos,
                stops,
            ) in enumerate(candidates):
                if delta_lb >= best_delta_loss - 1e-9:
                    B5_STATS["insertion_lb_pruned"] += float(
                        len(candidates) - candidate_index
                    )
                    break
'''
    source = _replace_once(
        source,
        old_candidates,
        new_candidates,
        "total-loss candidate pruning",
    )

    comparison_pattern = re.compile(
        r'''                completion = evaluation\.new_completion_time_min\n'''
        r'''                loss = evaluation\.projected_loss\n'''
        r'''                distance = evaluation\.total_distance_m\n'''
        r'''                better = \(.*?\n'''
        r'''                \)\n'''
        r'''                if better:\n''',
        re.S,
    )
    replacement = '''                completion = evaluation.new_completion_time_min
                loss = evaluation.projected_loss
                delta_loss = loss - baseline_loss
                distance = evaluation.total_distance_m
                better = (
                    delta_loss < best_delta_loss - 1e-6
                    or (
                        math.isclose(
                            delta_loss, best_delta_loss, abs_tol=1e-6, rel_tol=0.0
                        )
                        and (
                            loss < best_loss - 1e-6
                            or (
                                math.isclose(
                                    loss, best_loss, abs_tol=1e-6, rel_tol=0.0
                                )
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
                                                and (
                                                    best is None
                                                    or (robot_id, pickup_pos, dropoff_pos)
                                                    < (
                                                        best.robot_id,
                                                        best.pickup_pos,
                                                        best.dropoff_pos,
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                if better:
'''
    source, count = comparison_pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"candidate comparison: expected one match, found {count}")

    source = _replace_once(
        source,
        "                if better:\n"
        "                    best_completion = completion\n",
        "                if better:\n"
        "                    best_delta_loss = delta_loss\n"
        "                    best_completion = completion\n",
        "best delta update",
    )

    source = _replace_once(
        source,
        '        "selection_objective": "earliest_new_request_completion_time",\n',
        '        "selection_objective": "minimum_incremental_projected_deadline_aware_loss",\n'
        '        "candidate_score": "projected_loss_after_minus_projected_loss_before",\n'
        '        "optimization": "all-order-loss-lower-bound-plus-baseline-state-cache",\n',
        "result objective metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = _patch_source(TARGET.read_text(encoding="utf-8"))
    module_name = "benchmark5_loss_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(TARGET)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    namespace = _load_namespace()
    namespace["main"]()


if __name__ == "__main__":
    main()
