"""Small synthetic correctness fixtures, never campaign/training inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.fleet import RobotType
from delivery_fleet.meta_policy import POLICIES
from delivery_fleet.meta_policy.runtime import FleetRuntime, _EventCounter
from delivery_fleet.reservation_nn import ReservationFCNN
from delivery_fleet.scenario_creator import Item, Order, Scenario
from run_multicity_experiments import policy_command


@pytest.mark.parametrize("start", [0, 1042, 2**80])
def test_event_counter_pickle_preserves_exact_next_value(start):
    counter = _EventCounter(start)
    assert next(counter) == start
    restored = pickle.loads(pickle.dumps(counter, protocol=5))
    assert iter(restored) is restored
    assert [next(restored) for _ in range(4)] == list(range(start + 1, start + 5))
    assert next(counter) == start + 1


@pytest.fixture
def fixture(tmp_path, request):
    charging = getattr(request, "param", False)
    graph = nx.path_graph(1001)
    stations = list(range(0, 1001, 100)) if charging else [0, 500, 1000]
    for node in graph:
        graph.nodes[node].update(x=34. + node * .00001, y=32., in_cluster=0,
            is_cluster_representative=node == 500, is_charging_station=node in stations)
    nx.set_edge_attributes(graph, 40. if charging else 10., "length")
    annotate_nearest_charging_stations(graph, stations)
    gp, sp, mp = (tmp_path / name for name in ("map.graphml", "scenario.json", "nn.npz"))
    nx.write_graphml(graph, gp)
    Scenario(graph_name="independent_meta_unit_fixture", duration_minutes=35., seed=42,
        orders=tuple(Order(id=i, pickup_node=100 + (i % 3) * 100,
            dropoff_node=800 - (i % 3) * 80, request_time_min=float(i * 2),
            item=Item(1., 1.), importance=(1., 2., 5.)[i % 3]) for i in range(8))).save_json(sp)
    ReservationFCNN(4, [t.value for t in RobotType], (1., 2., 5.), seed=3).save(mp)
    return dict(graph=gp, scenario=sp, spatial_model=mp, paper_model=mp,
                prior_rates={"1": 10., "2": 2., "5": 1.}, scratch=tmp_path / "scratch")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fixture", [False, True], indirect=True, ids=["short_routes", "requires_charging"])
def test_fixed_adapter_matches_existing_policy(fixture, tmp_path, policy):
    command = policy_command(policy, python=sys.executable,
        output=tmp_path / "original.json", shortlist_k=10, idle_processes=1,
        prior_concentration=4., **{k: fixture[k] for k in
            ("graph", "scenario", "spatial_model", "paper_model", "prior_rates")})
    subprocess.run(command, check=True, capture_output=True, timeout=90,
                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    original = json.loads((tmp_path / "original.json").read_text())
    runtime = FleetRuntime(**fixture)
    try:
        runtime.set_policy(policy)
        runtime.drain()
        assert runtime.loss == pytest.approx(original["loss_objective"], rel=0, abs=1e-8)
        assert runtime.ns["delivered"] == original["delivered"]
        assert runtime.ns["metrics"].on_time == original["on_time"]
    finally:
        runtime.close()


def test_snapshot_aliasing_branch_isolation_and_switches(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.set_policy("full_coordinated_idle")
        runtime.advance(3.)
        snapshot = runtime.snapshot()
        before = [(r.spec.id, r.node_id, r.battery_wh, r.next_node,
                   r.next_node_arrival_time_min) for r in runtime.ns["robots"]]
        for policy in POLICIES:
            runtime.restore(snapshot)
            runtime.set_policy(policy)
            assert before == [(r.spec.id, r.node_id, r.battery_wh, r.next_node,
                               r.next_node_arrival_time_min) for r in runtime.ns["robots"]]
            assert all(runtime.ns["robot_by_id"][r.spec.id] is r for r in runtime.ns["robots"])
            assert all(c.robots[0] is runtime.ns["robots"][0]
                       for c in runtime.ns["_meta_controllers"].values())
            runtime.advance(8.)
            runtime.set_policy("reactive_insertion")
            runtime.drain()
            assert runtime.ns["delivered"] == 8
        runtime.restore(snapshot)
        assert runtime.now == 3.
        assert len(runtime.ns["_meta_observed"]) == 2
    finally:
        runtime.close()


def test_snapshot_restore_preserves_event_tie_breakers(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.advance(3.)
        before = [(e[0], e[1], e[2], e[3]) for e in runtime.ns["events"]]
        snapshot = runtime.snapshot()
        expected = next(runtime.ns["event_counter"])
        assert next(runtime.ns["event_counter"]) == expected + 1
        for _ in range(3):
            runtime.restore(snapshot)
            assert [(e[0], e[1], e[2], e[3]) for e in runtime.ns["events"]] == before
            assert next(runtime.ns["event_counter"]) == expected
        runtime.restore(snapshot)
        runtime.drain()
        assert runtime.ns["delivered"] == 8
    finally:
        runtime.close()
