"""Additional observable route-work and spatial supply/demand summaries."""
from __future__ import annotations

from collections import Counter
import math


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.
    index = fraction * (len(values) - 1)
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values)-1)] - values[lower]) * (index - lower)


def enrich(features, ns):
    row = dict(features)
    robots, now = ns["robots"], ns["now"]
    work = [len(ns["schedules"][r.spec.id].stops) for r in robots]
    for fraction, name in ((.5, "median"), (.9, "p90"), (1., "max")):
        row["routes.remaining_stops_" + name] = quantile(work, fraction)
    row["routes.remaining_edges_per_robot"] = sum(len(r.remaining_route) + int(r.is_moving) for r in robots) / len(robots)
    row["routes.active_handling_share"] = len(ns["service_lock"]) / len(robots)
    row["chargers.max_waiting_count"] = float(max((len(s.queue) for s in ns["station_state"].values()), default=0))
    graph = ns["graph"]
    def cluster(node):
        return int(graph.nodes[node].get("in_cluster", -1))
    arrivals = Counter(cluster(o.pickup_node) for o in ns["_meta_observed"] if now - 30 < o.request_time_min <= now)
    free = Counter(cluster(r.node_id) for r in robots if r.available and not ns["schedules"][r.spec.id].orders)
    demand_total, free_total = sum(arrivals.values()), sum(free.values())
    row["spatial.observed_demand_free_capacity_imbalance"] = (
        sum(abs(arrivals[c] / demand_total - free[c] / max(1, free_total)) for c in arrivals.keys() | free.keys())
        if demand_total else 0.)
    row["spatial.recent_demand_without_free_capacity_share"] = (
        sum(count for c, count in arrivals.items() if not free[c]) / demand_total if demand_total else 0.)
    if not all(math.isfinite(v) for v in row.values()):
        raise ValueError("Non-finite enriched observable features")
    return row
