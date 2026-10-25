from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_meta_runtime import fixture
from delivery_fleet.meta_policy.artifacts import worker_count, digest
from delivery_fleet.meta_policy.features import extract, map_features
from delivery_fleet.meta_policy.learning import LossSelector, fit_selector
from delivery_fleet.meta_policy.pipeline import run, collect_examples
from delivery_fleet.meta_policy.runtime import FleetRuntime
from delivery_fleet.scenario_creator import Scenario


def _reserved_fixture(tmp_path, fixture):
    from delivery_fleet.meta_policy.separation import workload_identity
    old = Scenario.load_json(fixture["scenario"])
    old = replace(old, seed=987654,
                  orders=tuple(replace(o, request_time_min=o.request_time_min + .123) for o in old.orders))
    path = tmp_path / "archived_benchmark.json"
    old.save_json(path)
    return {"version": 1, "policy": "fresh_train_and_validation", "reference_suites": [],
            "reserved": [{"path": str(path), "seed": old.seed, "sha256": digest(path),
                          "workload_sha256": workload_identity(old)}]}


def test_process_budget_counts_coordinator_and_ram():
    assert worker_count(60, memory_per_worker_gib=8, reserve_gib=16, available=400) <= 48
    assert worker_count(60, memory_per_worker_gib=8, reserve_gib=16, available=32) == 2
    assert worker_count(1, memory_per_worker_gib=8, reserve_gib=16, available=32) == 0
    with pytest.raises(ValueError):
        worker_count(61, memory_per_worker_gib=1, reserve_gib=0, available=400)
    with pytest.raises(RuntimeError):
        worker_count(60, memory_per_worker_gib=8, reserve_gib=16, available=20)


def test_features_do_not_depend_on_unreleased_orders(fixture):
    runtime = FleetRuntime(**fixture)
    try:
        runtime.advance(3.)
        static = map_features(runtime.ns["graph"])
        before = extract(runtime, static, holding_min=15.)
        runtime.ns["scenario"] = replace(runtime.ns["scenario"], orders=())
        assert extract(runtime, static, holding_min=15.) == before
        assert all(np.isfinite(v) for v in before.values())
        assert len(runtime.ns["_meta_observed"]) == 2
        assert runtime.ns["_meta_controllers"]["mle"].stats["arrivals"] == 2
    finally:
        runtime.close()


def _records():
    return [{"split": "train" if i < 12 else "validation", "group": f"episode_{i//4}",
             "city": "fixture", "scenario_sha256": f"hash_{i//4}", "step": i % 4,
             "holding_min": 15., "robots": 2,
             "features": {"state": float(i), "decision.holding_min": 15.},
             "losses": {"reactive_insertion": float(30+i), "full_no_idle": float(60-i)}}
            for i in range(16)]


@pytest.mark.parametrize("kind", ["trees", "nn"])
def test_regression_model_and_serialization(kind, tmp_path):
    model, report = fit_selector(_records(), kind=kind, max_iter=3)
    prediction = model.predictions(_records()[0]["features"])
    assert set(prediction) == {"reactive_insertion", "full_no_idle"}
    assert all(np.isfinite(v) for v in prediction.values())
    model.save(tmp_path / "selector.pkl")
    assert LossSelector.load(tmp_path / "selector.pkl").predictions(_records()[0]["features"]) == prediction
    assert report["training_episodes"] == 3


def test_episode_and_scenario_leakage_rejected():
    rows = _records()
    rows[-1]["group"] = "episode_0"
    with pytest.raises(ValueError, match="leakage"):
        fit_selector(rows, max_iter=1)
    rows = _records()
    rows[-1]["scenario_sha256"] = "hash_0"
    with pytest.raises(ValueError, match="same scenario"):
        fit_selector(rows, max_iter=1)
    rows = _records()
    rows[-1]["split"] = "test"
    with pytest.raises(ValueError, match="test rows"):
        fit_selector(rows, max_iter=1)


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def test_adaptive_interval_uses_loss_gap_and_uncertainty_not_softmax():
    selector = LossSelector(("holding",), ("reactive_insertion", "full_no_idle"), (5., 15., 30.),
        {"reactive_insertion": Constant(100.), "full_no_idle": Constant(20.)},
        {"reactive_insertion": 1., "full_no_idle": 1.}, adaptive=True)
    features = {w: {"holding": w} for w in selector.windows}
    assert selector.choose(features, "reactive_insertion")[:2] == ("full_no_idle", 30.)
    selector.validation_error_q90 = {p: 100. for p in selector.policies}
    assert selector.choose(features, "reactive_insertion")[0] == "reactive_insertion"
    selector.predictors["full_no_idle"] = Constant(100.)
    assert selector.choose(features, "reactive_insertion")[:2] == ("reactive_insertion", 15.)


def test_collect_and_resume_with_spawn_workers(fixture, tmp_path):
    # Two tiny episodes: exercise actual CLI-level plumbing without using any
    # project campaign/scenario/model. This is a unit integration fixture only.
    sp = Path(fixture["scenario"])
    scenario = Scenario.load_json(sp)
    scenario = replace(scenario, duration_minutes=5., orders=scenario.orders[:2])
    scenario.save_json(sp)
    vp = tmp_path / "validation.json"
    replace(scenario, seed=43, orders=tuple(replace(o, request_time_min=o.request_time_min + .1)
                                          for o in scenario.orders)).save_json(vp)
    config = {"version": 1, "spatial_model": str(fixture["spatial_model"]),
        "paper_model": str(fixture["paper_model"]),
        "policies": ["reactive_insertion", "full_no_idle"], "holding_minutes": [15],
        "scenario_separation": _reserved_fixture(tmp_path, fixture),
        "episodes": [{"id": name, "city": "fixture", "split": split,
            "graph": str(fixture["graph"]), "scenario": str(path),
            "prior_rates_per_hour": fixture["prior_rates"]} for name, split, path in
            (("train", "train", sp), ("valid", "validation", vp))]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config))
    result = run(manifest, tmp_path / "labels", processes=3,
                 memory_per_worker_gib=.01, reserve_gib=0)
    assert len(result) == 2
    assert all(r["delivered"] == 2 for r in result)
    records = collect_examples([tmp_path / "labels"])
    assert len(records) == 2
    assert set(records[0]["losses"]) == set(config["policies"])
    paths = list((tmp_path / "labels").glob("episodes/*/rollouts/*.json"))
    before = {str(p): (digest(p), p.stat().st_mtime_ns) for p in paths}
    run(manifest, tmp_path / "labels", processes=2, memory_per_worker_gib=.01, reserve_gib=0)
    assert before == {str(p): (digest(p), p.stat().st_mtime_ns) for p in paths}


def test_interrupted_collection_resumes_committed_state(fixture, tmp_path, monkeypatch):
    config = {"version": 1, "spatial_model": str(fixture["spatial_model"]),
        "paper_model": str(fixture["paper_model"]),
        "policies": ["reactive_insertion", "full_no_idle"], "holding_minutes": [15],
        "scenario_separation": _reserved_fixture(tmp_path, fixture),
        "episodes": [{"id": "interruption", "city": "fixture", "split": "train",
            "graph": str(fixture["graph"]), "scenario": str(fixture["scenario"]),
            "prior_rates_per_hour": fixture["prior_rates"]}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config))
    original = FleetRuntime.drain
    calls = 0
    def interrupted(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic interruption")
        return original(self, **kwargs)
    monkeypatch.setattr(FleetRuntime, "drain", interrupted)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match=r"episode\(s\) failed"):
        run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    assert "synthetic interruption" in (output / "failure.json").read_text()
    assert json.loads((output / "progress.json").read_text())["status"] == "failed"
    folder = output / "episodes/interruption"
    assert json.loads((folder / "checkpoint.json").read_text())["step"] == 1
    saved = {str(p): (digest(p), p.stat().st_mtime_ns) for p in (folder / "rollouts").glob("*.json")}
    monkeypatch.setattr(FleetRuntime, "drain", original)
    results = run(manifest, output, processes=1, memory_per_worker_gib=.01, reserve_gib=0)
    assert results[0]["delivered"] == 8
    assert saved == {p: (digest(p), Path(p).stat().st_mtime_ns) for p in saved}


def test_fitting_resumes_completed_regressors(tmp_path, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    first, _ = fit_selector(_records(), max_iter=2, checkpoint_dir=tmp_path / "fit")
    def forbidden(*args, **kwargs):
        raise AssertionError("completed regressor must not be fitted again")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    second, _ = fit_selector(_records(), max_iter=2, checkpoint_dir=tmp_path / "fit")
    assert first.predictions(_records()[0]["features"]) == second.predictions(_records()[0]["features"])
