"""Online MLE -> bounded alpha search -> concrete priority thresholds.

All templates are revealed arrivals. Idle demand keeps its old Gamma-Poisson
model to isolate the reservation replacement from a new relocation model.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .anticipatory_policy import ReservationRobotSnapshot
from .deadlines import delivery_time_allowance_min
from .mle_reservation import (CandidateScorer, EpochPoissonRates, MLEConfig,
                              SurrogateTrace, importance_stress_rates,
                              select_gated_candidate)
from .reservation import apportion_reservation_counts, assign_reservation_thresholds
from .spatial_demand import (GammaPoissonDemandModel, haversine_node_distance_m,
                             summarize_existing_clusters)


class OnlineMLEReservation:
    def __init__(self, graph, robots, importance_rates_per_hour, *,
                 charger_power_w, horizon_min, config=MLEConfig(),
                 prior_concentration=4., audit_path=None):
        self.graph, self.robots = graph, tuple(robots)
        if not self.robots or len({r.spec.id for r in self.robots}) != len(self.robots):
            raise ValueError('fleet must be nonempty with unique IDs')
        if not math.isfinite(charger_power_w) or charger_power_w <= 0:
            raise ValueError('charger power must be finite and positive')
        if not math.isfinite(horizon_min) or horizon_min <= 0:
            raise ValueError('simulation horizon must be finite and positive')
        self.config, self.charger_power_w, self.horizon_min = config, charger_power_w, horizon_min
        self.importance_levels = tuple(sorted(float(v) for v in importance_rates_per_hour))
        if len(self.importance_levels) > 4:
            raise ValueError('bounded stress search supports at most four importance levels')
        self.demand = GammaPoissonDemandModel(graph, importance_rates_per_hour,
                                            prior_concentration=prior_concentration)
        self.cluster_summary = summarize_existing_clusters(graph)
        self.cells = tuple((z, c) for z in sorted(self.cluster_summary.cluster_sizes)
                           for c in self.importance_levels)
        self.rates = EpochPoissonRates((*self.cells, *(('total', c) for c in self.importance_levels)),
                                      confidence=config.confidence, epoch_min=config.epoch_min)
        self.templates = defaultdict(lambda: deque(maxlen=128))
        self.by_type = defaultdict(list)
        for robot in self.robots:
            self.by_type[robot.spec.robot_type].append(robot)
        self.type_counts = {k: len(rs) for k, rs in self.by_type.items()}
        self.scorer = CandidateScorer(config.processes)
        self.assignment = None
        self.next_update = 0.
        self.stats = Counter()
        self.reason = 'cold_start'
        self.last_decision_reason = self.reason
        self.reason_counts = Counter()
        self.first_activation_min = None
        self.last_time = 0.
        self.active_since = None
        self.active_minutes = 0.
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.write_text('', encoding='utf-8')

    def _advance(self, now):
        if self.rates.advance(now):
            self.templates.clear()
            self.next_update = now
            # Previous-epoch assignments are never silently retained.
            self.assignment = None

    def observe(self, pickup_node, importance, time_min):
        self.demand.observe(pickup_node, importance, time_min)
        self._advance(float(time_min))
        cluster = int(self.graph.nodes[pickup_node]['in_cluster'])
        self.rates.observe((cluster, float(importance)), float(time_min))
        self.rates.observe(('total', float(importance)), float(time_min))
        self.stats['arrivals'] += 1

    def observe_order(self, order, now):
        """Record known pickup/dropoff, item and geometry, never future orders."""
        if order.request_time_min > now:
            raise ValueError('cannot learn from an unreleased order')
        cluster = int(self.graph.nodes[order.pickup_node]['in_cluster'])
        distance = haversine_node_distance_m(self.graph, order.pickup_node, order.dropoff_node)
        self.templates[(cluster, float(order.importance))].append((
            int(order.pickup_node), distance, float(order.item.weight_kg), float(order.item.volume_l)))

    def _finish(self, now, assignment, reason, details=None):
        was_active = self.active_since is not None
        if was_active:
            self.active_minutes += now-self.last_time
        if bool(assignment) != was_active:
            self.stats['activations' if assignment else 'deactivations'] += 1
        if assignment is not None and self.first_activation_min is None:
            self.first_activation_min = now
        self.assignment, self.reason = assignment, reason
        if reason != 'simulation_finished':
            self.last_decision_reason = reason
            self.reason_counts[reason] += 1
        self.active_since = now if assignment else None
        self.last_time = now
        record = {'time_min': now, 'epoch': self.rates.epoch,
                  'exposure_min': now-self.rates.epoch*self.config.epoch_min,
                  'observed_arrivals': self.stats['arrivals'],
                  'epoch_arrivals': sum(self.rates.counts[('total',c)] for c in self.importance_levels),
                  'enabled': assignment is not None, 'reason': reason, **(details or {})}
        if self.audit_path:
            with self.audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
        return assignment

    def _candidates(self, bounds, snapshots):
        rates = {cell: bounds[cell].mle for cell in self.cells}
        scores = {}
        service_by_type = {}
        for kind, robots in self.by_type.items():
            values = []
            for robot in robots:
                state = snapshots[robot.spec.id]
                response = {}
                for z, rep in self.cluster_summary.representatives.items():
                    dist = haversine_node_distance_m(self.graph, state.node_id, rep)
                    charge = max(0., dist*robot.spec.energy_per_meter_wh-state.battery_wh)/self.charger_power_w*60
                    response[z] = state.available_in_min+dist/robot.spec.speed_mps/60+charge
                values.append(np.mean(list(response.values())))
                for threshold in self.importance_levels[1:]:
                    weights = [(rates[(z,c)], response[z]) for z,c in self.cells if c >= threshold]
                    mass = sum(w for w, _ in weights)
                    scores[(robot.spec.id, threshold)] = (sum(w*v for w,v in weights)/mass
                                                           if mass else float(np.mean(list(response.values()))))
            service_by_type[kind] = max(1., float(np.mean(values)))
        finite = [v for v in service_by_type.values() if math.isfinite(v)]
        reference = float(np.mean(finite)) if finite else 1.
        high_weights = np.array([bounds[('total',c)].mle*(c+1)**2 for c in self.importance_levels[1:]])
        if high_weights.sum() == 0:
            high_weights = np.ones_like(high_weights)
        if len(high_weights):
            high_weights /= high_weights.sum()
        assignments, fractions, seen = [], [], set()
        for reserve in (0., .025, .05, .10, .20, .35, .50):
            by_type = {}
            for kind in self.by_type:
                efficiency = min(1.5, reference/service_by_type[kind])
                amount = min(.75, reserve*efficiency) if len(high_weights) else 0.
                by_type[kind] = np.r_[1-amount, amount*high_weights]
            counts = apportion_reservation_counts(by_type, self.type_counts, self.importance_levels)
            assignment = assign_reservation_thresholds(self.robots, counts, scores, self.importance_levels)
            signature = tuple(assignment.threshold_by_robot_id[r.spec.id] for r in self.robots)
            if signature in seen:
                continue
            seen.add(signature)
            assignments.append(assignment)
            # Stability compares REALIZED integer fractions, including protected general capacity.
            fractions.append(np.array([[assignment.counts_by_type[k][c]/self.type_counts[k]
                                        for c in self.importance_levels] for k in self.by_type]))
        return assignments, np.asarray(fractions)

    def _scenarios(self, bounds, snapshots, horizon, queue_wait_by_robot, handling_min, now):
        all_templates = [t for bucket in self.templates.values() for t in bucket]
        if not all_templates:
            raise ValueError('no revealed request templates')
        class_bounds = [bounds[('total',c)] for c in self.importance_levels]
        stress_rates = importance_stress_rates(class_bounds)
        # Extra spatial stress changes conditional cluster weights, without
        # pretending this finite grid exhausts the high-dimensional rate box.
        variants = [(rates, 'mle') for rates in stress_rates]
        variants += [(stress_rates[0], mode) for mode in ['lower', 'upper']]
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.rates.epoch,
                                                           int(now/self.config.update_interval_min)]))
        robot_states = [snapshots[r.spec.id] for r in self.robots]
        robot_lat = np.radians([float(self.graph.nodes[s.node_id]['y']) for s in robot_states])
        robot_lon = np.radians([float(self.graph.nodes[s.node_id]['x']) for s in robot_states])
        speed = np.array([r.spec.speed_mps for r in self.robots])
        energy = np.array([r.spec.energy_per_meter_wh for r in self.robots])
        battery = np.array([s.battery_wh for s in robot_states])
        ready = np.array([s.available_in_min for s in robot_states])
        payload = np.array([r.spec.max_payload_kg for r in self.robots])
        volume = np.array([r.spec.max_volume_l for r in self.robots])
        queue = np.array([queue_wait_by_robot.get(r.spec.id, 0.) for r in self.robots])
        result = []
        for rates, mode in variants:
            traces = []
            for _ in range(self.config.rollouts):
                sizes = rng.poisson(rates*horizon)
                if sizes.sum() > self.config.max_requests_per_trace:
                    raise OverflowError('stress workload exceeds bounded search budget')
                levels, templates, arrivals = [], [], []
                for i, count in enumerate(sizes):
                    c = self.importance_levels[i]
                    cells = [cell for cell in self.cells if cell[1] == c]
                    if cells:
                        weights = np.array([getattr(bounds[cell], mode) for cell in cells])
                        if weights.sum() == 0:
                            weights = np.ones(len(cells))
                        weights /= weights.sum()
                        chosen_cells = rng.choice(len(cells), size=count, p=weights)
                        for index in chosen_cells:
                            cell = cells[index]
                            pool = self.templates[cell]
                            if pool:
                                templates.append(pool[int(rng.integers(len(pool)))])
                            else:
                                known = all_templates[int(rng.integers(len(all_templates)))]
                                templates.append((self.cluster_summary.representatives[cell[0]], *known[1:]))
                    else:
                        templates.extend(all_templates[int(j)] for j in rng.integers(len(all_templates), size=count))
                    levels.extend([c]*int(count))
                    arrivals.extend(rng.uniform(0., horizon, size=count))
                if not levels:
                    traces.append(SurrogateTrace(np.array([]), np.array([]), np.array([]),
                                                 np.empty((0,len(self.robots))), ready))
                    continue
                order = np.argsort(arrivals, kind='stable')
                levels = np.asarray(levels)[order]; arrivals = np.asarray(arrivals)[order]
                templates = [templates[int(i)] for i in order]
                lat = np.radians([float(self.graph.nodes[t[0]]['y']) for t in templates])[:,None]
                lon = np.radians([float(self.graph.nodes[t[0]]['x']) for t in templates])[:,None]
                h = np.sin((lat-robot_lat)/2)**2+np.cos(lat)*np.cos(robot_lat)*np.sin((lon-robot_lon)/2)**2
                approach = 2*6_371_008.8*np.arcsin(np.sqrt(np.clip(h,0.,1.)))
                direct = np.array([t[1] for t in templates])
                distance = approach+direct[:,None]
                deficit = np.maximum(0., distance*energy-battery)
                service = distance/speed/60+handling_min+deficit/self.charger_power_w*60
                service += np.where(deficit > 0, queue, 0.)
                capable = ((np.array([t[2] for t in templates])[:,None] <= payload)
                           & (np.array([t[3] for t in templates])[:,None] <= volume))
                service = np.where(capable, service, np.inf)
                allowance = np.array([delivery_time_allowance_min(d,c) for d,c in zip(direct,levels)])
                traces.append(SurrogateTrace(arrivals, levels, allowance, service, ready))
            result.append(traces)
        return result

    def update(self, now, snapshots, *, queue_wait_by_robot=None, handling_min=2., pending_count=0):
        now = float(now)
        self._advance(now)
        if now >= self.horizon_min:
            return self._finish(now, None, 'arrival_horizon_ended')
        if pending_count:
            return self._finish(now, None, 'pending_real_orders')
        if now < self.next_update:
            return self.assignment
        self.next_update = now+self.config.update_interval_min
        started = time.perf_counter()
        self.stats['updates'] += 1
        try:
            return self._plan(now, snapshots, queue_wait_by_robot or {}, handling_min)
        finally:
            self.stats['planning_seconds'] += time.perf_counter()-started

    def _plan(self, now, snapshots, queue_wait_by_robot, handling_min):
        bounds = self.rates.bounds(now)
        if (now-self.rates.epoch*self.config.epoch_min == 0 or
                sum(bounds[('total',c)].count for c in self.importance_levels) < self.config.minimum_arrivals):
            return self._finish(now, None, 'insufficient_arrivals')
        class_bounds = [bounds[('total',c)] for c in self.importance_levels]
        width = sum(b.upper-b.lower for b in class_bounds)/max(sum(b.mle for b in class_bounds), 1e-12)
        rate_details = {'class_rates': {str(c): asdict(bounds[('total',c)]) for c in self.importance_levels},
                        'relative_rate_width': width}
        if width > self.config.maximum_relative_rate_width:
            return self._finish(now, None, 'rate_intervals_too_wide', rate_details)
        if any(not math.isfinite(s.available_in_min) for s in snapshots.values()):
            return self._finish(now, None, 'infeasible_readiness')
        if not all(math.isfinite(v) and v >= 0 for v in (queue_wait_by_robot or {}).values()):
            raise ValueError('queue waits must be finite and non-negative')
        assignments, fractions = self._candidates(bounds, snapshots)
        horizon = min(self.config.horizon_min, self.horizon_min-now,
                      (self.rates.epoch+1)*self.config.epoch_min-now)
        try:
            scenarios = self._scenarios(bounds, snapshots, horizon, queue_wait_by_robot or {}, handling_min, now)
        except OverflowError:
            self.stats['budget_fallbacks'] += 1
            return self._finish(now, None, 'stress_budget_exceeded')
        thresholds = [np.array([a.threshold_by_robot_id[r.spec.id] for r in self.robots]) for a in assignments]
        scores = self.scorer.score(thresholds, scenarios)
        gate = select_gated_candidate(scores, fractions, self.config)
        self.stats['candidates_scored'] += len(assignments)
        details = {'confidence': self.config.confidence,
                   **rate_details,
                   'candidate_count': len(assignments), 'stress_cases': len(scenarios),
                   'minimum_stress_gain_fraction': gate.minimum_gain_fraction,
                   'maximum_alpha_disagreement': (gate.maximum_alpha_disagreement
                                                 if math.isfinite(gate.maximum_alpha_disagreement) else None),
                   'alpha_by_type': {getattr(k,'value',str(k)): {str(c): assignments[gate.candidate_index].counts_by_type[k][c]/self.type_counts[k]
                                                              for c in self.importance_levels} for k in self.by_type}}
        return self._finish(now, assignments[gate.candidate_index] if gate.enabled else None,
                            gate.reason, details)

    def close(self):
        self.scorer.close()

    def finalize(self, now):
        self._finish(float(now), None, 'simulation_finished')
        self.close()

    def diagnostics(self):
        return {'mle_reservation_confidence': self.config.confidence,
                'mle_reservation_config': asdict(self.config),
                'mle_reservation_observed_arrivals': self.stats['arrivals'],
                'mle_reservation_updates': self.stats['updates'],
                'mle_reservation_activations': self.stats['activations'],
                'mle_reservation_deactivations': self.stats['deactivations'],
                'mle_reservation_active_minutes': self.active_minutes,
                'mle_reservation_planning_seconds': self.stats['planning_seconds'],
                'mle_reservation_candidates_scored': self.stats['candidates_scored'],
                'mle_reservation_budget_fallbacks': self.stats['budget_fallbacks'],
                'mle_reservation_final_reason': self.reason,
                'mle_reservation_last_decision_reason': self.last_decision_reason,
                'mle_reservation_reason_counts': dict(self.reason_counts),
                'mle_reservation_first_activation_min': self.first_activation_min,
                'mle_reservation_decisions_file': str(self.audit_path) if self.audit_path else None}
