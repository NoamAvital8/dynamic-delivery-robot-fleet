from __future__ import annotations

"""Exact high-throughput corrected B5 using battery-routing fast paths.

This version keeps the corrected B5 search and objective unchanged.  It only
avoids expensive charger-meta-graph searches when the mandatory direct route is
already provably battery-feasible.  Because each direct leg is a shortest-path
distance, no charger detour can have lower route distance in that case.

It also strengthens the admissible sequence-loss lower bound with unavoidable
minimum charging time implied by cumulative energy consumption.  This affects
pruning only; candidates are still evaluated with the exact routing model.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_nyc_benchmark5_loss as corrected


def _once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return source.replace(old, new, 1)


def _patch(source: str) -> str:
    source = _once(
        source,
        "from delivery_fleet.insertion_routing import evaluate_pair_from_committed_state\n",
        "from delivery_fleet.fast_pair_routing import (\n"
        "    evaluate_pair_from_committed_state_fast as evaluate_pair_from_committed_state,\n"
        ")\n",
        "fast committed-pair import",
    )

    # Exact O(1) fast path for the final mandatory stop.  If direct travel plus
    # the normal final charger reserve fits the available range, it is physically
    # shortest and the generic charger Dijkstra cannot improve it.
    old = '''        start_is_station = start in router.station_set

        START = ("start",)
        DONE = ("done",)
'''
    new = '''        start_is_station = start in router.station_set

        available_range = full_range if start_is_station else current_range
        if start_to_target_m + reserve_distance <= available_range + DISTANCE_EPS_M:
            required_departure = (
                start_to_target_m * spec.energy_per_meter_wh + reserve_wh
            )
            battery = float(robot.battery_wh)
            phases: list[RoutePhase] = []
            charging_time = 0.0
            if battery + BATTERY_EPS_WH < required_departure:
                if not start_is_station:
                    raise RuntimeError("direct final fast path needs non-station charge")
                energy = required_departure - battery
                phases.append(RoutePhase("charge", start, energy_wh=float(energy)))
                battery += energy
                charging_time = energy / DEFAULT_CHARGING_POWER_W * 60.0
            phases.append(
                RoutePhase("travel", target_node, distance_m=float(start_to_target_m))
            )
            battery -= start_to_target_m * spec.energy_per_meter_wh
            travel_time = start_to_target_m / spec.speed_mps / 60.0
            return PrefixPlan(
                phases=tuple(phases),
                total_time_min=float(travel_time + charging_time),
                total_distance_m=float(start_to_target_m),
                arrival_battery_wh=float(max(0.0, battery)),
            )

        START = ("start",)
        DONE = ("done",)
'''
    source = _once(source, old, new, "final target direct fast path")

    # Add a safe lower bound on charging time.  After cumulative shortest-path
    # distance d, at least max(0, d*energy_rate - starting_battery) Wh must have
    # been charged somewhere before that point.  Actual routes can only travel
    # farther and/or charge more, so this remains admissible.
    old_lb = '''        projected_time = float(snapshot.time_min)
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
'''
    new_lb = '''        projected_time = float(snapshot.time_min)
        node = int(snapshot.node_id)
        lower_loss = 0.0
        cumulative_distance = 0.0
        charged_lb_wh = 0.0
        start_battery = float(snapshot.battery_wh)
        energy_rate = float(robot.spec.energy_per_meter_wh)
        for stop in stops:
            distance = lower_bound_distance(node, int(stop.node_id), special_rows)
            cumulative_distance += distance
            projected_time += distance / robot.spec.speed_mps / 60.0
            required_charge_wh = max(
                0.0, cumulative_distance * energy_rate - start_battery
            )
            if required_charge_wh > charged_lb_wh:
                projected_time += (
                    (required_charge_wh - charged_lb_wh)
                    / DEFAULT_CHARGING_POWER_W
                    * 60.0
                )
                charged_lb_wh = required_charge_wh
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
'''
    source = _once(source, old_lb, new_lb, "energy-aware loss lower bound")

    source = _once(
        source,
        '        "optimization": "all-order-loss-lower-bound-plus-baseline-state-cache",\n',
        '        "optimization": "exact-direct-battery-fast-path-plus-energy-aware-loss-bound",\n',
        "optimization metadata",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = corrected._patch_source(corrected.TARGET.read_text(encoding="utf-8"))
    source = _patch(source)
    module_name = "benchmark5_loss_direct_fast_target"
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
