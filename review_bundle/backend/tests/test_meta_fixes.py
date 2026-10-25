"""Regression tests for the meta-policy review fixes (independent tiny fixtures)."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_meta_runtime import fixture  # noqa: F401  (pytest fixture)
from test_meta_training import _reserved_fixture
from delivery_fleet.charging import annotate_nearest_charging_stations
from delivery_fleet.fleet import RobotType
from delivery_fleet.mle_reservation import MLEConfig
from delivery_fleet.reservation_nn import ReservationFCNN
from delivery_fleet.scenario_creator import Item, Order, Scenario
from delivery_fleet.meta_policy import DEFAULT_PORTFOLIO, POLICIES, labels, pipeline
from delivery_fleet.meta_policy.features import VERSION, extract_windows, map_features
from delivery_fleet.meta_policy.learning import (LossSelector, build_provenance, evaluate_selector,
                                                 fit_selector)
from delivery_fleet.meta_policy.pipeline import collect_examples, run, simulator_files
from delivery_fleet.meta_policy.runtime import FleetRuntime, controller_settings
from run_multicity_experiments import policy_command, validate_result

ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
RELAXED_MLE = dict(minimum_arrivals=3, maximum_relative_rate_width=100., minimum_gain_fraction=0.,
                   stability_tolerance=1., epoch_min=30., update_interval_min=5., horizon_min=20., rollouts=2)


def _world(tmp_path, *, nodes, count, gap, span=40, step=10):
    """Path-graph city with three clusters; robots = ceil(nodes / 500)."""
    graph = nx.path_graph(nodes)
    stations = list(range(0, nodes, 300))
    third = nodes // 3
    for node in graph:
        graph.nodes[node].update(x=34. + node * .00001, y=32., in_cluster=min(node // third, 2),
            is_cluster_representative=node in (third // 2, third + third // 2, 2 * third + third // 2),
            is_charging_station=node in stations)
    nx.set_edge_attributes(graph, 10., "length")
    annotate_nearest_charging_stations(graph, stations)
    gp, sp, mp = tmp_path / "map.graphml", tmp_path / "scenario.json", tmp_path / "nn.npz"
    nx.write_graphml(graph, gp)
    usable = nodes - 2 * span - 2 * 5 * step - 2
    orders = tuple(Order(id=i, pickup_node=(397 * i) % usable + span + 5 * step,
                         dropoff_node=(397 * i) % usable + span + 5 * step + (-1) ** i * (span + (i % 5) * step),
                         request_time_min=i * gap, item=Item(1., 1.), importance=(1., 1., 2., 5.)[i % 4])
                   for i in range(count))
    Scenario(graph_name="review_fixture", duration_minutes=count * gap + 1, seed=42, orders=orders).save_json(sp)
    ReservationFCNN(4, [t.value for t in RobotType], (1., 2., 5.), seed=3).save(mp)
    rate = 60 / gap
    return dict(graph=gp, scenario=sp, spatial_model=mp, paper_model=mp,
                prior_rates={"1": rate * .5, "2": rate * .25, "5": rate * .25}, scratch=tmp_path / "scratch")


def _original(world, policy, tmp_path, mle=None):
    output = tmp_path / f"original_{policy}.json"
    command = policy_command(policy, python=sys.executable, output=output, shortlist_k=10, idle_processes=1,
                             prior_concentration=4., mle_config=MLEConfig(**(mle or {})),
                             **{k: world[k] for k in ("graph", "scenario", "spatial_model", "paper_model", "prior_rates")})
    subprocess.run(command, check=True, capture_output=True, timeout=600, env=ENV)
    return json.loads(output.read_text())


def _meta(world, policy, *, step=None, restore=False, settings=None):
    runtime = FleetRuntime(**world, settings=settings)
    try:
        runtime.set_policy(policy)
        end = runtime.ns["scenario"].duration_minutes
        while step and runtime.now < end and not runtime.ns["_meta_finished"]:
            if restore:
                runtime.restore(runtime.snapshot())
            runtime.set_policy(policy)
            runtime.advance(min(runtime.now + step, end))
        runtime.drain()
        mle = runtime.ns["_meta_controllers"]["mle"]
        return runtime.loss, runtime.ns["delivered"], dict(mle.reason_counts)
    finally:
        runtime.close()


# --- shared simulator ---------------------------------------------------------

def test_mle_policies_keep_original_arrival_semantics(tmp_path):
    """P0-2: the shared engine must not treat a new arrival as an existing backlog."""
    world = _world(tmp_path, nodes=3001, count=40, gap=1.)
    for policy in ("full_mle_reservation", "queue_mle_reservation"):
        original = _original(world, policy, tmp_path, RELAXED_MLE)
        reasons = original["mle_reservation_reason_counts"]
        assert set(reasons) - {"insufficient_arrivals", "arrival_horizon_ended"}, "fixture must reach MLE planning"
        loss, delivered, meta_reasons = _meta(world, policy, settings={"mle": RELAXED_MLE})
        assert meta_reasons == reasons
        assert "pending_real_orders" not in meta_reasons
        assert loss == pytest.approx(original["loss_objective"], rel=0, abs=1e-8)
        assert delivered == original["delivered"]


def test_new_arrival_keeps_an_active_mle_assignment(fixture):
    runtime = FleetRuntime(**fixture, settings={"mle": RELAXED_MLE})
    try:
        runtime.set_policy("full_mle_reservation")
        runtime.advance(3.)
        ns, mle = runtime.ns, runtime.ns["_meta_controllers"]["mle"]
        active = object()
        mle.assignment, mle.next_update = active, runtime.now + 100.
        mle.rates.epoch = int(runtime.now // mle.config.epoch_min)
        new_order = runtime.ns["scenario"].orders[-1]
        ns["pending"][:] = [new_order]
        runtime._refresh_arrival(runtime.now)
        assert ns["reservation_assignment"] is active
        assert "pending_real_orders" not in mle.reason_counts
        ns["pending"][:] = [new_order, new_order]  # one genuine backlog order
        runtime._refresh_arrival(runtime.now)
        assert ns["reservation_assignment"] is None
        assert mle.reason_counts["pending_real_orders"] == 1
        ns["pending"][:] = []
    finally:
        runtime.close()


@pytest.mark.parametrize("policy", ["full", "paper_sa_adapted", "full_no_idle", "full_queue_aware",
                                    "full_coordinated_idle", "full_mle_reservation"])
def test_boundaries_and_restores_match_original_under_backlog(tmp_path, policy):
    """3 robots, 16 orders in 8 minutes: dispatch happens from a pending backlog."""
    world = _world(tmp_path, nodes=1001, count=16, gap=.5, span=20, step=5)
    original = _original(world, policy, tmp_path)
    for restore in (False, True):
        loss, delivered, _ = _meta(world, policy, step=2., restore=restore)
        assert loss == pytest.approx(original["loss_objective"], rel=0, abs=1e-8)
        assert delivered == original["delivered"]


def test_reselecting_the_active_policy_is_a_noop(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.set_policy("full")
        runtime.advance(3.)
        assignment = runtime.ns["reservation_assignment"]
        calls = []
        original = runtime.ns["refresh_reservations"]
        runtime.ns["refresh_reservations"] = lambda *a, **k: (calls.append(a), original(*a, **k))[1]
        runtime.set_policy("full")
        assert calls == [] and runtime.ns["reservation_assignment"] is assignment
    finally:
        runtime.close()


def test_mle_replans_only_when_reentered_after_a_switch(fixture):
    runtime = FleetRuntime(**fixture, settings={"mle": RELAXED_MLE})
    try:
        mle = runtime.ns["_meta_controllers"]["mle"]
        scheduled = mle.next_update
        runtime.set_policy("full_mle_reservation")
        assert mle.next_update == scheduled  # first entry keeps the original schedule
        runtime.advance(2.)
        runtime.set_policy("full")
        assert mle.meta_reentry_pending
        runtime.set_policy("queue_mle_reservation")
        assert not mle.meta_reentry_pending and mle.next_update >= runtime.now
    finally:
        runtime.close()


def test_snapshot_restores_reservation_and_queue_forecast(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.set_policy("full_queue_aware")
        assert runtime.ns["assignment_queue_forecast"] is not None  # rebuilt on switch
        runtime.advance(3.)
        assignment = runtime.ns["reservation_assignment"]
        assert assignment is not None
        blob = runtime.snapshot()
        runtime.ns["reservation_assignment"] = None
        runtime.ns["assignment_queue_forecast"] = None
        runtime.restore(blob)
        assert runtime.ns["reservation_assignment"].threshold_by_robot_id == assignment.threshold_by_robot_id
        assert runtime.ns["assignment_queue_forecast"] is not None
    finally:
        runtime.close()


def test_equal_tail_digests_have_equal_tails(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.set_policy("full")
        runtime.advance(3.)
        before, start = runtime.snapshot(), runtime.loss
        results = []
        for _ in range(2):
            _, tail, _ = labels.prepare_tail(runtime, before, candidate="paper_sa_adapted", boundary=8.,
                                             horizon=20., continuation="full_no_idle", keep_boundary=False)
            results.append((tail, labels.score_tail(runtime, horizon=20., start_loss=start)[0]))
        assert results[0] == results[1]
        _, other, _ = labels.prepare_tail(runtime, before, candidate="paper_sa_adapted", boundary=9.,
                                          horizon=20., continuation="full_no_idle", keep_boundary=False)
        assert other != results[0][0]
    finally:
        runtime.close()


def test_controller_settings_reach_the_engine_and_unknown_keys_fail(fixture):
    with pytest.raises(ValueError, match="unsupported"):
        controller_settings({"not_a_setting": 1})
    with pytest.raises(ValueError, match="unsupported MLE"):
        controller_settings({"mle": {"processes": 4}})
    defaults = FleetRuntime(**fixture)
    custom = FleetRuntime(**fixture, settings={"idle_horizon_min": 30., "mle": {"minimum_arrivals": 3}})
    try:
        assert defaults.planners["legacy"].config.horizon_min == 45.
        assert custom.planners["legacy"].config.horizon_min == 30.
        assert custom.ns["_meta_controllers"]["mle"].config.minimum_arrivals == 3
        assert custom.input_digest != defaults.input_digest
        # Shared, not copied, planner tables.
        assert custom.planners["legacy"]._coordinates is custom.planners["coordinated"]._coordinates
    finally:
        defaults.close()
        custom.close()


@pytest.mark.skipif(os.name == "nt", reason="shared exact index uses Linux flock/mmap")
def test_verified_index_is_not_rehashed_until_files_change(fixture, tmp_path):
    code = r"""
import json, os, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts", sys.argv[1] + "/operations"]
from delivery_fleet.meta_policy.runtime import FleetRuntime
world = json.loads(sys.argv[2]); cache = Path(sys.argv[3])
FleetRuntime(**world, shared_cache=cache).close()
import exact_runtime
calls = []
original = exact_runtime.file_hash
exact_runtime.file_hash = lambda path: (calls.append(path), original(path))[1]
FleetRuntime(**world, shared_cache=cache).close()
unchanged = len(calls)
table = next(cache.glob("*/predecessors.npy"))
os.utime(table, ns=(table.stat().st_atime_ns, table.stat().st_mtime_ns + 1000))
FleetRuntime(**world, shared_cache=cache).close()
print(json.dumps({"unchanged": unchanged, "after_change": len(calls) - unchanged}))
"""
    world = {k: str(v) for k, v in fixture.items() if k != "prior_rates"}
    world["prior_rates"] = fixture["prior_rates"]
    result = subprocess.run([sys.executable, "-c", code, str(ROOT), json.dumps(world), str(tmp_path / "cache")],
                            capture_output=True, text=True, timeout=300, env=ENV)
    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout.strip().splitlines()[-1])
    assert counts == {"unchanged": 0, "after_change": 2}


# --- pipeline -------------------------------------------------------------------

def _manifest(tmp_path, fixture, **settings):
    scenario = Scenario.load_json(fixture["scenario"])
    validation = tmp_path / "validation.json"
    replace(scenario, seed=43, orders=tuple(replace(o, request_time_min=o.request_time_min + .1)
                                            for o in scenario.orders)).save_json(validation)
    config = {"version": 1, "spatial_model": str(fixture["spatial_model"]),
              "paper_model": str(fixture["paper_model"]),
              "policies": ["reactive_insertion", "full_no_idle"], "holding_minutes": [15],
              "scenario_separation": _reserved_fixture(tmp_path, fixture),
              "episodes": [{"id": name, "city": "fixture", "split": split, "graph": str(fixture["graph"]),
                            "scenario": str(path), "prior_rates_per_hour": fixture["prior_rates"]}
                           for name, split, path in (("train", "train", fixture["scenario"]),
                                                     ("valid", "validation", validation))],
              **settings}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config))
    return path, config


def test_identical_tails_are_scored_once(fixture, tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path, fixture)
    monkeypatch.setattr(FleetRuntime, "state_digest", lambda self: "same-state")
    run(manifest, tmp_path / "labels", processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    by_step = {}
    for path in (tmp_path / "labels").glob("episodes/*/rollouts/*.json"):
        step = (path.parent.parent.name, path.name.split("_")[0])
        by_step.setdefault(step, []).append(json.loads(path.read_text()))
    assert by_step and all(sum(not r["tail_reused"] for r in group) == 1 for group in by_step.values())
    for row in collect_examples([tmp_path / "labels"]):
        assert len(set(row["losses"].values())) == 1


def test_sampled_labels_sticky_exploration_and_post_horizon_decisions(fixture, tmp_path):
    scenario = Scenario.load_json(fixture["scenario"])
    replace(scenario, duration_minutes=5., orders=scenario.orders[:3]).save_json(fixture["scenario"])
    manifest, _ = _manifest(tmp_path, fixture, label_step_probability=.5, decide_until="all_delivered",
                            exploration={"stickiness": .9})
    output = tmp_path / "labels"
    first = run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    for result in first:
        folder = output / "episodes" / result["episode"]
        traces = [json.loads(p.read_text()) for p in sorted((folder / "decisions").glob("*.json"))]
        assert any(t["time_min"] >= 5. for t in traces), "decisions must continue after the arrival horizon"
        assert result["decisions_after_arrival_horizon"] == sum(t["time_min"] >= 5. for t in traces)
        labelled = {round(t["time_min"], 6) for t in traces if t["labelled"]}
        examples = [json.loads(p.read_text()) for p in (folder / "examples").glob("*.json")]
        assert {round(e["time_min"], 6) for e in examples} == labelled
    stamps = {str(p): p.stat().st_mtime_ns for p in output.glob("episodes/*/*/*.json")}
    assert run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0) == first
    assert stamps == {str(p): p.stat().st_mtime_ns for p in output.glob("episodes/*/*/*.json")}


def test_pilot_stops_early_and_resumes(fixture, tmp_path):
    manifest, _ = _manifest(tmp_path, fixture)
    output = tmp_path / "pilot"
    pilot = run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0, max_steps=1)
    assert {r["status"] for r in pilot} == {"pilot_stopped"}
    assert json.loads((output / "summary.json").read_text())["status"] == "pilot_stopped"
    assert not list(output.glob("episodes/*/complete.json"))
    first_decision = (output / "episodes/train/decisions/00000.json").read_bytes()
    done = run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    assert {r["status"] for r in done} == {"complete"}
    assert (output / "episodes/train/decisions/00000.json").read_bytes() == first_decision
    states = list((output / "episodes/train/states").glob("*.pickle"))
    checkpoint = json.loads((output / "episodes/train/checkpoint.json").read_text())
    assert [p.name for p in states] == [Path(checkpoint["state"]).name], "only the committed state is kept"


def test_one_failed_episode_does_not_stop_the_others(fixture, tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path, fixture)
    original = pipeline._episode

    def failing(job):
        if job["episode"]["id"] == "train":
            raise RuntimeError("synthetic episode failure")
        return original(job)
    monkeypatch.setattr(pipeline, "_episode", failing)
    output = tmp_path / "labels"
    with pytest.raises(RuntimeError, match=r"1 episode\(s\) failed"):
        run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    assert (output / "episodes/valid/complete.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert [f["episode"] for f in failure["failures"]] == ["train"]
    assert json.loads((output / "progress.json").read_text())["status"] == "failed"


def test_actual_trajectory_follows_only_the_chosen_policy(fixture, tmp_path):
    """Alternatives are scored and discarded; the world continues from the chosen branch.

    Replaying only the recorded choices, without any counterfactual branches,
    must reproduce the collected episode exactly.
    """
    policies = ["reactive_insertion", "full_no_idle"]
    seed = next(s for s in range(200)
                if len({pipeline._pick(policies, s, "train", step) for step in range(3)}) > 1)
    manifest, _ = _manifest(tmp_path, fixture, exploration={"stickiness": 0.})
    output = tmp_path / "labels"
    run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0, seed=seed)
    folder = output / "episodes/train"
    traces = [json.loads(p.read_text()) for p in sorted((folder / "decisions").glob("*.json"))]
    assert len({t["policy"] for t in traces}) > 1, "the episode must switch policies"
    assert all(t["labelled"] for t in traces)
    rollouts = list((folder / "rollouts").glob("*.json"))
    assert len(rollouts) == len(traces) * len(policies), "every alternative was scored"
    complete = json.loads((folder / "complete.json").read_text())
    runtime = FleetRuntime(**fixture, shared_cache=output / "shared_exact_index" if os.name != "nt" else None)
    try:
        runtime.set_policy("reactive_insertion")  # manifest default initial policy
        for trace in traces:
            assert runtime.now == trace["time_min"]
            runtime.set_policy(trace["policy"])
            runtime.advance(trace["committed_until_min"])
        runtime.drain()
        assert runtime.loss == complete["loss"]
        assert runtime.ns["delivered"] == complete["delivered"]
        assert runtime.ns["metrics"].on_time == complete["on_time"]
    finally:
        runtime.close()


def test_validate_mode_runs_only_on_held_out_validation(fixture, tmp_path):
    manifest, config = _manifest(tmp_path, fixture)
    run(manifest, tmp_path / "labels", processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    records = collect_examples([tmp_path / "labels"])
    selector, report = fit_selector(records, max_iter=2)
    assert "rule_regret" in report["validation_decisions"]["per_episode_mean"]
    selector.provenance = build_provenance(records)
    path = tmp_path / "selector.pkl"
    selector.save(path)
    results = run(manifest, tmp_path / "validate", mode="validate", selector=path, processes=1,
                  memory_per_worker_gib=.01, reserve_gib=0)
    assert [r["episode"] for r in results] == ["valid"] and results[0]["status"] == "complete"
    as_test = dict(config, episodes=[dict(e, split="test") for e in config["episodes"] if e["id"] == "valid"])
    (tmp_path / "as_test.json").write_text(json.dumps(as_test))
    with pytest.raises(ValueError, match="used to fit or validate"):
        run(tmp_path / "as_test.json", tmp_path / "evaluate", mode="evaluate", selector=path, processes=1,
            memory_per_worker_gib=.01, reserve_gib=0)
    fit_as_validation = dict(config, episodes=[dict(e, split="validation") for e in config["episodes"]
                                               if e["id"] == "train"])
    (tmp_path / "fit_as_validation.json").write_text(json.dumps(fit_as_validation))
    with pytest.raises(ValueError, match="held-out validation"):
        run(tmp_path / "fit_as_validation.json", tmp_path / "bad", mode="validate", selector=path,
            processes=1, memory_per_worker_gib=.01, reserve_gib=0)


def test_label_identity_ignores_learning_and_unrelated_scripts():
    files = {str(p.relative_to(ROOT)).replace("\\", "/") for p in simulator_files()}
    for required in ("src/delivery_fleet/meta_policy/runtime.py", "src/delivery_fleet/meta_policy/labels.py",
                     "operations/exact_runtime.py", "scripts/run_nyc_benchmark5.py",
                     "scripts/run_nyc_mle_reservation_policy.py"):
        assert required in files
    for unrelated in ("src/delivery_fleet/meta_policy/learning.py", "src/delivery_fleet/meta_policy/pipeline.py",
                      "src/delivery_fleet/meta_policy/features.py", "scripts/train_meta_policy.py",
                      "scripts/run_nyc_benchmark3.py", "scripts/run_multicity_experiments.py"):
        assert unrelated not in files


def test_features_once_per_decision_and_trend_is_a_rate(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        static = map_features(runtime.ns["graph"])
        assert "map.cluster_noise_share" not in static and VERSION == 2
        rows = extract_windows(runtime, static, (5., 15.))
        assert {k for k in rows[5.] if rows[5.][k] != rows[15.][k]} == {"decision.holding_min"}
        assert all(v == 0. for k, v in rows[5.].items() if "rate_trend" in k)
        runtime.advance(3.)
        cache = dict(runtime.ns["direct_distance_cache"])
        extract_windows(runtime, static, (15.,))
        assert runtime.ns["direct_distance_cache"] == cache, "feature extraction must not write state"
    finally:
        runtime.close()


# --- learning -------------------------------------------------------------------

class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


class ByHold:
    def __init__(self, values):
        self.values = values

    def predict(self, x):
        return np.array([self.values[float(v)] for v in x[:, 0]])


def test_deployed_rule_regret_is_reported_separately_from_raw_argmin():
    """Second-review F2 example: raw argmin regret 0, conservative rule regret 50."""
    selector = LossSelector(("f",), ("reactive_insertion", "full_no_idle"), (15.,),
                            {"reactive_insertion": Constant(100.), "full_no_idle": Constant(80.)},
                            {"reactive_insertion": 15., "full_no_idle": 15.}, minimum_gain_fraction=0.)
    row = {"group": "g", "step": 0, "city": "c", "holding_min": 15., "robots": 1,
           "features": {"f": 0., "state.policy.reactive_insertion": 1., "state.policy.full_no_idle": 0.},
           "losses": {"reactive_insertion": 130., "full_no_idle": 80.}}
    report = evaluate_selector(selector, [row])["per_row_mean"]
    assert report["raw_argmin_regret"] == 0. and report["rule_regret"] == 50.
    assert report["vetoed"] == 1.


def test_long_hold_requires_agreement_at_every_shorter_horizon():
    """Second-review F4 example: the 5-minute winner differs, so no 30-minute hold."""
    selector = LossSelector(("decision.holding_min",), ("reactive_insertion", "full_no_idle"), (5., 15., 30.),
                            {"reactive_insertion": ByHold({5.: 50., 15.: 10., 30.: 10.}),
                             "full_no_idle": ByHold({5.: 10., 15.: 50., 30.: 50.})},
                            {"reactive_insertion": 0., "full_no_idle": 0.}, minimum_gain_fraction=0., adaptive=True)
    features = {w: {"decision.holding_min": w} for w in selector.windows}
    policy, hold, _ = selector.choose(features, "reactive_insertion")
    assert policy == "reactive_insertion" and hold != 30.


def test_first_decision_has_no_incumbent_hysteresis():
    selector = LossSelector(("f",), ("reactive_insertion", "full_no_idle"), (15.,),
                            {"reactive_insertion": Constant(100.), "full_no_idle": Constant(99.)},
                            {"reactive_insertion": 1e6, "full_no_idle": 1e6})
    assert selector.choose({15.: {"f": 0.}}, "reactive_insertion")[0] == "reactive_insertion"
    assert selector.choose({15.: {"f": 0.}}, None)[0] == "full_no_idle"


def test_advantage_target_margins_ignore_the_common_loss_level():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        level, signal = rng.uniform(0, 1000), rng.uniform(-1, 1)
        rows.append({"split": "train" if i < 240 else "validation", "group": f"e{i // 10}", "step": i % 10,
                     "city": "c", "scenario_sha256": f"s{i // 10}", "holding_min": 15., "robots": 1,
                     "features": {"level": level, "signal": signal},
                     "losses": {"reactive_insertion": level + 10 * signal + 20,
                                "full_no_idle": level - 10 * signal + 20}})
    selector, report = fit_selector(rows, max_iter=60)
    pair = selector.pair_error("reactive_insertion", "full_no_idle")
    absolute = report["validation_error_q90_per_robot"]
    assert pair < .5 * (absolute["reactive_insertion"] + absolute["full_no_idle"])
    assert report["validation_decisions"]["per_row_mean"]["top1"] > .9


# --- benchmark results and data preparation -----------------------------------------

def test_benchmark_results_are_validated_before_acceptance():
    job = {"seed": 123, "orders": 100, "output": "result.json"}
    good = {"scenario_seed": 123, "orders": 100, "delivered": 100, "on_time": 60,
            "loss_objective": 5., "wall_clock_seconds": 1., "simulation_finish_min": 700.}
    validate_result(good, job)
    for change in ({"scenario_seed": 999}, {"orders": 1, "delivered": 1}, {"delivered": 99},
                   {"loss_objective": math.nan}, {"wall_clock_seconds": -1.},
                   {"simulation_finish_min": math.inf}, {"on_time": 101}):
        with pytest.raises(ValueError):
            validate_result({**good, **change}, job)


def test_reused_result_must_match_its_stamp(tmp_path):
    from run_multicity_experiments import _execute, _sha256
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"scenario_seed": 1, "orders": 1, "delivered": 1, "on_time": 1,
                                  "loss_objective": 1., "wall_clock_seconds": 1., "simulation_finish_min": 1.}))
    stamp = tmp_path / "stamp.json"
    stamp.write_text(json.dumps({"signature": "s", "result_sha256": _sha256(output)}))
    job = {"city": "c", "scenario_id": "x", "seed": 1, "policy": "full", "orders": 1, "output": output,
           "log": tmp_path / "log", "status_path": tmp_path / "status.json", "stamp_path": stamp,
           "command": ["unused"], "signature": "s", "accepted_signatures": {"s"}}
    assert _execute(job, None)["status"] == "complete"
    output.write_text(output.read_text().replace('"loss_objective": 1.0', '"loss_objective": 0.5'))
    with pytest.raises(RuntimeError, match="changed after"):
        _execute(job, None)


def test_default_portfolio_and_city_exclusions():
    from delivery_fleet.meta_policy import DEFAULT_EXCLUDED_CITIES
    assert "reactive_insertion" not in DEFAULT_PORTFOLIO and "myopic_ab" not in DEFAULT_PORTFOLIO
    assert len(DEFAULT_PORTFOLIO) == 6 and set(DEFAULT_PORTFOLIO) <= set(POLICIES)
    # Both queue-aware idle styles, each with NN reservation and without (MLE).
    assert {"full_queue_aware", "queue_mle_reservation",
            "full_coordinated_idle", "full_mle_reservation"} <= set(DEFAULT_PORTFOLIO)
    # Order follows POLICIES, which fixes the selector's tie-breaking.
    assert list(DEFAULT_PORTFOLIO) == [p for p in POLICIES if p in DEFAULT_PORTFOLIO]
    assert set(DEFAULT_EXCLUDED_CITIES) == {"beijing", "sydney"}


def test_prepare_skips_excluded_cities_unless_listed(fixture, tmp_path):
    from delivery_fleet.meta_policy.fresh_scenarios import prepare
    archived = Scenario.load_json(fixture["scenario"])
    cities = {}
    for name, seed in (("beijing", 97), ("paris", 96)):
        test = tmp_path / f"{name}_test.json"
        replace(archived, seed=seed).save_json(test)
        cities[name] = {"graph": str(fixture["graph"]), "prior_rates_per_hour": fixture["prior_rates"],
                        "test": [{"id": f"{name}_test", "scenario": str(test), "graph": str(fixture["graph"]),
                                  "seed": seed}]}
    suite = tmp_path / "suite" / "suite.json"
    suite.parent.mkdir()
    suite.write_text(json.dumps({"duration_hours": 1, "cities": cities}))
    options = dict(train_per_city=1, validation_per_city=1, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    default = prepare(suite, fixture["spatial_model"], fixture["paper_model"], tmp_path / "d/manifest.json", **options)
    assert {e["city"] for e in default["episodes"]} == {"paris"}
    assert default["initial_policy"] in default["policies"]
    explicit = prepare(suite, fixture["spatial_model"], fixture["paper_model"], tmp_path / "e/manifest.json",
                       cities=["beijing"], **options)
    assert {e["city"] for e in explicit["episodes"]} == {"beijing"}


def test_prepare_combines_several_reference_suites(fixture, tmp_path):
    from delivery_fleet.meta_policy.fresh_scenarios import prepare
    archived = Scenario.load_json(fixture["scenario"])
    suites = []
    for name, seed in (("a", 99), ("b", 98)):
        test = tmp_path / f"{name}_test.json"
        replace(archived, seed=seed).save_json(test)
        suite = tmp_path / name / "suite.json"
        suite.parent.mkdir()
        suite.write_text(json.dumps({"duration_hours": 1, "cities": {name: {
            "graph": str(fixture["graph"]), "prior_rates_per_hour": fixture["prior_rates"],
            "test": [{"id": f"{name}_test", "scenario": str(test), "graph": str(fixture["graph"]), "seed": seed}]}}}))
        suites.append(suite)
    manifest = prepare(suites, fixture["spatial_model"], fixture["paper_model"], tmp_path / "out/manifest.json",
                       train_per_city=1, validation_per_city=1, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    assert {e["city"] for e in manifest["episodes"]} == {"a", "b"}
    assert tuple(manifest["policies"]) == DEFAULT_PORTFOLIO and "reactive_insertion" not in manifest["policies"]
    assert manifest["continuation_policy"] == "full_no_idle" and manifest["decide_until"] == "all_delivered"
    comparison = json.loads((tmp_path / "out/manifest_comparison.json").read_text())
    assert {e["id"] for e in comparison["episodes"]} == {"a_test", "b_test"}
    duplicate = tmp_path / "copy" / "suite.json"
    duplicate.parent.mkdir()
    duplicate.write_text(suites[0].read_text())
    with pytest.raises(ValueError, match="more than one reference suite"):
        prepare([suites[0], duplicate], fixture["spatial_model"],
                fixture["paper_model"], tmp_path / "dup/manifest.json")
