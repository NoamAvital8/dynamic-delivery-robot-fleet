from __future__ import annotations

"""Opt-in uncertainty-gated MLE reservation, with legacy or coordinated idle."""
import argparse
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_nyc_coordinated_idle_policy as coordinated


def _patch_mle(source: str) -> str:
    replace = coordinated.queue.idle._replace_once
    source = replace(source, '    args = parser.parse_args()\n', '''    parser.add_argument('--mle-idle-mode', choices=['legacy', 'coordinated'], default='coordinated')
    from delivery_fleet.mle_reservation import add_mle_arguments, mle_config_from_args
    add_mle_arguments(parser)
    args = parser.parse_args()
    from delivery_fleet.mle_policy import OnlineMLEReservation
    if args.reservation_model is not None or args.fixed_reservation_fractions is not None:
        parser.error('MLE reservation does not load an NN or fixed fractions')
    try:
        mle_config = mle_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    worker_budget = 1 + (args.idle_processes if args.idle_processes > 1 else 0)
    worker_budget += args.mle_processes if args.mle_processes > 1 else 0
    if worker_budget > 60:
        parser.error(f'MLE simulator needs up to {worker_budget} compute workers; cap is 60')
''', 'MLE CLI')
    source = replace(source, '    if reservation_policy is None:\n        try:\n', '''    try:
        mle_prior_rates = {float(k): float(v) for k,v in json.loads(args.importance_prior_rates_per_hour).items()}
        reservation_policy = OnlineMLEReservation(
            graph, robots, mle_prior_rates, charger_power_w=DEFAULT_CHARGING_POWER_W,
            horizon_min=scenario.duration_minutes, config=mle_config,
            prior_concentration=args.prior_concentration,
            audit_path=args.output.with_suffix('.reservation.jsonl'),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if reservation_policy is None:
        try:
''', 'MLE initialization before idle demand')
    begin = source.index('    def refresh_reservations(now: float) -> None:\n')
    end = source.index('\n    def ', begin+5)
    helper = '''    def refresh_reservations(now: float, *, arrival: bool = False) -> None:
        nonlocal reservation_assignment
        backlog = max(0, len(pending)-int(arrival))
        # Observe every arrival, but do not rebuild projections/search on every one.
        if (now < reservation_policy.next_update and now < scenario.duration_minutes
                and int(now // args.mle_epoch_min) == reservation_policy.rates.epoch and not backlog):
            reservation_assignment = reservation_policy.assignment
            return
        if backlog or now >= scenario.duration_minutes:
            reservation_assignment = reservation_policy.update(now, {}, pending_count=backlog)
            return
        calendar = build_charging_calendar(now)
        snapshots = {}
        queue_waits = {}
        forecast = build_assignment_queue_forecast(now)
        for robot in robots:
            rid = robot.spec.id
            intent = idle_intents.get(rid)
            if intent is not None:
                node, battery = intent['target_node'], intent['battery_arrival_wh']
                ready = max(0.0, intent['arrival_min']-now)
                if intent['kind'] == 'charge':
                    battery = robot.spec.battery_capacity_wh
                    ready = max(0.0, intent['slot'].finish_min-now)
            elif robot.activity is RobotActivity.CHARGING:
                found = find_active_charge(rid)
                if found is None:
                    raise RuntimeError('charging robot has no active session')
                _, _, session = found
                node = int(robot.node_id)
                battery = min(robot.spec.battery_capacity_wh, session.start_battery_wh+session.energy_wh)
                ready = max(0.0, session.start_time_min+session.energy_wh/DEFAULT_CHARGING_POWER_W*60-now)
            elif robot.activity is RobotActivity.WAITING:
                node = int(robot.node_id)
                booked = next((s for s in calendar.slots(node) if s.robot_id == rid), None)
                request = next((q for q in station_state[node].queue if q.robot_id == rid), None)
                if booked is None or request is None:
                    raise RuntimeError('waiting robot has no projected charging request')
                battery = min(robot.spec.battery_capacity_wh, robot.battery_wh+request.energy_wh)
                ready = max(0.0, booked.finish_min-now)
            else:
                decision = decision_snapshot(robot, now)
                node = int(decision.node_id) if decision else int(robot.node_id)
                battery = float(decision.battery_wh) if decision else float(robot.battery_wh)
                ready = max(0.0, decision.time_min-now) if decision else 0.0
            if rid in service_lock:
                ready = max(ready, idle_service_finish_min[rid]-now)
            if schedules[rid].orders:
                initial = IdleRobotSnapshot(rid, node, robot.spec.speed_mps,
                    robot.spec.energy_per_meter_wh, battery, robot.spec.battery_capacity_wh,
                    robot.spec.max_payload_kg, robot.spec.max_volume_l, available_in_min=ready)
                steps = busy_readiness_steps(robot)
                if steps is None:
                    ready, battery = math.inf, 0.0
                else:
                    projected = project_remaining_readiness(initial, now, steps, calendar,
                        distance_m=idle_haversine_distance, charger_power_w=DEFAULT_CHARGING_POWER_W,
                        pickup_handling_min=PICKUP_HANDLING_MIN, dropoff_handling_min=DROPOFF_HANDLING_MIN)
                    node, battery, ready = int(projected.node_id), projected.battery_wh, max(0.0, projected.finish_min-now)
            snapshots[rid] = ReservationRobotSnapshot(node, ready, battery, bool(schedules[rid].orders))
            if math.isfinite(ready):
                nearest = min(charger_index.distances_to_stations(node).items(), key=lambda row: (row[1], row[0]))
                eta = now+ready+nearest[1]/robot.spec.speed_mps/60
                queue_waits[rid] = forecast.project(rid).wait_at(nearest[0], eta)
        reservation_assignment = reservation_policy.update(now, snapshots,
            queue_wait_by_robot=queue_waits, handling_min=PICKUP_HANDLING_MIN+DROPOFF_HANDLING_MIN)
        B5_STATS['reservation_updates'] += 1.0

'''
    source = replace(source, source[begin:end], helper, 'MLE readiness and periodic optimizer')
    source = replace(source, '''                refresh_reservations(now)
            else:
''', '''                reservation_policy.observe_order(order, now)
                refresh_reservations(now, arrival=True)
            else:
''', 'MLE revealed-arrival features')
    source = replace(source, '    tick = 0.0\n    while tick < scenario.duration_minutes:\n', '''    reservation_tick = 0.0
    while reservation_tick < scenario.duration_minutes:
        push_event(reservation_tick, 3, 'mle_reservation_refresh', None)
        reservation_tick += args.mle_update_interval_min
    push_event(scenario.duration_minutes, 3, 'mle_reservation_refresh', None)
    epoch_tick = args.mle_epoch_min
    while epoch_tick < scenario.duration_minutes:
        push_event(epoch_tick, 3, 'mle_reservation_refresh', None)
        epoch_tick += args.mle_epoch_min
    tick = 0.0
    while tick < scenario.duration_minutes:
''', 'MLE updates independent of order or idle event frequency')
    source = replace(source, '        elif kind == "idle_replan":\n', '''        elif kind == 'mle_reservation_refresh':
            refresh_reservations(now)
            dispatch_pending(now)

        elif kind == "idle_replan":
''', 'MLE inspection events')
    source = replace(source, '    idle_planner.close()\n', '''    reservation_policy.finalize(now)
    idle_planner.close()
''', 'close MLE workers and final audit')
    # Metadata makes configured controller vs actual gated reservations explicit.
    source = replace(source, '        "reservation_enabled": reservation_policy is not None,\n',
                     '''        "reservation_controller_configured": True,
        "reservation_enabled": reservation_policy.stats['activations'] > 0,
''', 'MLE configured vs activated')
    source = replace(source, '        "reservation_style": (args.reservation_style if reservation_policy is not None else None),\n',
                     '        "reservation_style": "epoch_poisson_mle_anytime_gated_surrogate",\n',
                     'MLE reservation style')
    version_line = ('        "policy_version": "coordinated_idle_v1",\n'
                    if '        "policy_version": "coordinated_idle_v1",\n' in source
                    else '        "policy_version": "queue_aware_v1",\n')
    source = replace(source, version_line, '''        "policy_version": "uncertainty_gated_mle_v1",
        "reservation_fallback": "queue_aware_haversine_top_k_greedy_no_priority_reservation",
        "reservation_statistical_guarantee": "arrival_rate_bounds_only_under_epoch_poisson_assumptions",
        "reservation_optimizer": "bounded_empirical_nonpreemptive_server_surrogate",
        "mle_idle_mode": args.mle_idle_mode,
        "mle_processes": args.mle_processes,
        **reservation_policy.diagnostics(),
''', 'MLE diagnostics and honest statistical scope')
    return source


def _load_namespace(coordinated_idle=True):
    queue = coordinated.queue
    source = queue.idle.heuristic.corrected._patch_source(queue.idle.heuristic.corrected.TARGET.read_text(encoding='utf-8'))
    source = queue.idle.heuristic._patch_heuristic(source)
    source = queue.idle._patch_idle(source)
    source = queue._patch_queue(source)
    if coordinated_idle:
        source = coordinated._patch_coordinated(source)
    source = _patch_mle(source)
    module = types.ModuleType('nyc_mle_reservation_' + ('coordinated' if coordinated_idle else 'legacy'))
    module.__file__ = str(queue.idle.heuristic.corrected.TARGET)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module.__dict__


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--mle-idle-mode', choices=['legacy', 'coordinated'], default='coordinated')
    options, _ = selector.parse_known_args()
    _load_namespace(options.mle_idle_mode == 'coordinated')['main']()


if __name__ == '__main__':
    main()
