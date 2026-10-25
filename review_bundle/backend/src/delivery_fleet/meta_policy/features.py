"""Observable, versioned features. Never inspect future scenario orders.

Feature extraction is read-only with respect to simulator state (no cache
writes), so it cannot change a snapshot or a later decision.
"""
from __future__ import annotations

from collections import Counter
import heapq
import math
import numpy as np

from delivery_fleet.fleet import RobotType
from delivery_fleet.deadlines import delivery_deadline_min
from . import POLICIES

VERSION = 2


def map_features(graph):
    """Static summaries computed once; graph nodes/edges stay out of checkpoints."""
    clusters = Counter(int(d.get("in_cluster", -1)) for _, d in graph.nodes(data=True))
    n, e = len(graph), graph.number_of_edges()
    lat = np.array([float(d["y"]) for _, d in graph.nodes(data=True)])
    lon = np.array([float(d["x"]) for _, d in graph.nodes(data=True)])
    lengths = np.array([float(d.get("length", 1.)) for *_, d in graph.edges(data=True)])
    xspan = float(np.ptp(lon)) * 111.32 * math.cos(math.radians(float(lat.mean())))
    yspan = float(np.ptp(lat)) * 111.32
    area = max(1e-6, xspan * yspan)
    positive = [v for k, v in clusters.items() if k >= 0]
    return {
        "map.nodes_log1p": math.log1p(n), "map.edges_log1p": math.log1p(e),
        "map.edges_per_node": e / max(1, n), "map.bbox_area_km2_log1p": math.log1p(area),
        "map.edge_length_mean_m": float(lengths.mean()) if len(lengths) else 0.,
        "map.edge_length_std_m": float(lengths.std()) if len(lengths) else 0.,
        "map.cluster_count": float(len(positive)),
        "map.largest_cluster_share": max(positive, default=0) / max(1, n),
        "map.chargers_per_km2": sum(str(d.get("is_charging_station", "")).lower()
            in {"true", "1", "yes"} for _, d in graph.nodes(data=True)) / area,
    }


def _direct_distance(ns, order):
    """Cached direct distance, or a read-only computation (no cache writes)."""
    cached = ns["direct_distance_cache"].get(order.id)
    if cached is not None:
        return cached
    return ns["row_distance"](ns["distance_row"](ns["oracle"], order.pickup_node), order.dropoff_node)


def _window_rate(count, start, end):
    """Arrivals per minute over the observed part of (start, end]."""
    observed = max(0., end - max(start, 0.))
    return count / observed if observed >= 1. else 0.


def extract(runtime, static, *, holding_min, policy_age_min=0., recent_loss_rate=0.):
    return extract_windows(runtime, static, (holding_min,), policy_age_min=policy_age_min,
                           recent_loss_rate=recent_loss_rate)[holding_min]


def extract_windows(runtime, static, windows, *, policy_age_min=0., recent_loss_rate=0.):
    """State features once per decision; only `decision.holding_min` differs per window."""
    ns, now = runtime.ns, runtime.now
    robots, observed = ns["robots"], ns["_meta_observed"]
    count = len(robots)
    duration = ns["scenario"].duration_minutes
    area_km2 = max(1e-6, math.expm1(static["map.bbox_area_km2_log1p"]))
    row = dict(static)
    row.update({"state.time_fraction": now / duration,
                "state.after_arrival_horizon": float(now >= duration),
                "state.policy_age_min": float(policy_age_min),
                "state.recent_loss_per_robot_min": recent_loss_rate / count,
                "state.fleet_log1p": math.log1p(count),
                "fleet.robots_per_km2_log1p": math.log1p(count / area_km2),
                "reservation.active": float(ns.get("reservation_assignment") is not None)})
    for p in POLICIES:
        row["state.policy." + p] = float(runtime.policy == p)
    intents = list(ns["idle_intents"].values())
    row["fleet.reposition_share"] = sum(i["kind"] == "reposition" for i in intents) / count
    row["fleet.charge_intent_share"] = sum(i["kind"] == "charge" for i in intents) / count
    for kind in RobotType:
        rs = [r for r in robots if r.spec.robot_type == kind]
        denominator = max(1, len(rs))
        values = [ns["projected_charge_battery"](r, now) / r.spec.battery_capacity_wh for r in rs]
        prefix = "fleet." + kind.value + "."
        row.update({prefix + "share": len(rs) / count,
                    prefix + "busy": sum(bool(ns["schedules"][r.spec.id].orders) for r in rs) / denominator,
                    prefix + "charging": sum(r.activity.value == "charging" for r in rs) / denominator,
                    prefix + "waiting": sum(r.activity.value == "waiting" for r in rs) / denominator,
                    prefix + "battery_mean": float(np.mean(values)) if values else 0.,
                    prefix + "battery_std": float(np.std(values)) if values else 0.,
                    prefix + "battery_min": min(values, default=0.),
                    prefix + "battery_below_20pct": sum(v < .2 for v in values) / denominator})
    active = [o for s in ns["schedules"].values() for o in s.orders.values()]
    outstanding = [(o.order, o.deadline_min) for o in active]
    outstanding += [(o, delivery_deadline_min(o.request_time_min,
                     _direct_distance(ns, o), o.importance)) for o in ns["pending"]]
    for importance in ns["_meta_demand"].importance_levels:
        level = float(importance)
        recent = sum(o.importance == level and now - 15. < o.request_time_min <= now for o in observed)
        previous = sum(o.importance == level and now - 30. < o.request_time_min <= now - 15.
                       for o in observed)
        recent_rate = _window_rate(recent, now - 15., now)
        previous_rate = _window_rate(previous, now - 30., now - 15.)
        rates = [ns["_meta_demand"].posterior(c, level, at_time_min=now)
                 for c in ns["_meta_demand"].cluster_sizes]
        prefix = f"orders.importance_{level:g}."
        row.update({prefix + "outstanding_per_robot": sum(o.importance == level for o, _ in outstanding) / count,
                    prefix + "recent_rate_per_robot_min": recent_rate / count,
                    # Rate change between the last two 15-minute windows; 0 until
                    # both windows contain at least one observed minute.
                    prefix + "rate_trend_per_robot_min": ((recent_rate - previous_rate) / count
                                                          if now - 15. >= 1. else 0.),
                    prefix + "posterior_rate_per_robot_min": sum(r.mean_per_minute for r in rates) / count,
                    prefix + "posterior_std_per_robot_min": math.sqrt(sum(r.std_per_minute ** 2 for r in rates)) / count})
    slack = [deadline - now for _, deadline in outstanding]
    locations = Counter(int(ns["graph"].nodes[o.pickup_node].get("in_cluster", -1))
                        for o in observed if now - 30. < o.request_time_min <= now)
    size = sum(locations.values())
    row.update({"orders.pending_per_robot": len(ns["pending"]) / count,
                "orders.slack_mean_min": float(np.mean(slack)) if slack else 0.,
                "orders.slack_std_min": float(np.std(slack)) if slack else 0.,
                "orders.overdue_share": sum(s < 0 for s in slack) / max(1, len(slack)),
                "demand.pickup_entropy": -sum((v / size) * math.log(v / size) for v in locations.values()) if size else 0.,
                "chargers.occupied_per_port": sum(len(s.active) for s in ns["station_state"].values())
                    / max(1, len(ns["station_nodes"]) * ns["DEFAULT_NUMBER_OF_PORTS"]),
                "chargers.waiting_per_robot": sum(len(s.queue) for s in ns["station_state"].values()) / count})
    # Known FIFO waiting estimate, not actual future waiting or charging duration.
    waits = []
    for state in ns["station_state"].values():
        free = [now] * (ns["DEFAULT_NUMBER_OF_PORTS"] - len(state.active))
        free += [max(now, s.start_time_min + s.energy_wh / ns["DEFAULT_CHARGING_POWER_W"] * 60)
                 for s in state.active.values()]
        heapq.heapify(free)
        for request in state.queue:
            start = max(now, heapq.heappop(free))
            waits.append(start - now)
            heapq.heappush(free, start + request.energy_wh / ns["DEFAULT_CHARGING_POWER_W"] * 60)
    row["chargers.predicted_queue_wait_mean_min"] = float(np.mean(waits)) if waits else 0.
    if not all(math.isfinite(v) for v in row.values()):
        raise ValueError("non-finite observable feature")
    return {float(hold): {**row, "decision.holding_min": float(hold)} for hold in windows}
