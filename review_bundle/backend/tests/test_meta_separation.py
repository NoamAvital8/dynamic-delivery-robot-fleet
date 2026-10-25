"""Fresh data and exact old-scenario comparison, using independent tiny inputs."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_meta_runtime import fixture
from test_meta_training import _reserved_fixture
from delivery_fleet.meta_policy.artifacts import digest
from delivery_fleet.meta_policy.fresh_scenarios import prepare
from delivery_fleet.meta_policy.separation import check_separation, workload_identity
from delivery_fleet.scenario_creator import Scenario


def test_renamed_reseeded_or_reformatted_old_workload_is_rejected(fixture, tmp_path):
    registry = _reserved_fixture(tmp_path, fixture)
    config = {"scenario_separation": registry}
    archived = Scenario.load_json(registry["reserved"][0]["path"])
    renamed = replace(archived, graph_name="another_name", seed=321,
                      orders=tuple(replace(o, id=o.id + 1000) for o in archived.orders))
    copied = tmp_path / "apparently_new.json"
    renamed.save_json(copied)
    assert workload_identity(renamed) == workload_identity(archived)
    with pytest.raises(ValueError, match="overlaps reserved"):
        check_separation(config, [{"id": "new", "scenario": str(copied), "split": "train"}], mode="collect")
    genuinely_different = replace(archived, orders=archived.orders[:-1])
    genuinely_different.save_json(copied)
    with pytest.raises(ValueError, match="overlaps reserved"):
        check_separation(config, [{"id": "new", "scenario": str(copied), "split": "validation"}], mode="collect")


def test_legacy_manifest_cannot_silently_reuse_existing_scenarios(fixture):
    with pytest.raises(ValueError, match="scenario_separation registry"):
        check_separation({}, [{"id": "old", "scenario": str(fixture["scenario"])}], mode="collect")


def test_distinct_files_with_same_workload_cannot_cross_new_splits(fixture, tmp_path):
    registry = _reserved_fixture(tmp_path, fixture)
    original = Scenario.load_json(fixture["scenario"])
    copy = tmp_path / "reseeded_validation.json"
    replace(original, seed=54321).save_json(copy)
    with pytest.raises(ValueError, match="duplicate a seed or workload"):
        check_separation({"scenario_separation": registry}, [
            {"id": "train", "split": "train", "scenario": str(fixture["scenario"])},
            {"id": "validation", "split": "validation", "scenario": str(copy)}], mode="collect")


def test_fresh_generation_preserves_old_tests_and_has_disjoint_seeds_and_workloads(fixture, tmp_path):
    archived = Scenario.load_json(fixture["scenario"])
    reference = tmp_path / "suite.json"
    test_path = tmp_path / "old_test.json"
    replace(archived, seed=99).save_json(test_path)
    reference.write_text(json.dumps({"duration_hours": 1, "cities": {"city": {
        "graph": str(fixture["graph"]), "prior_rates_per_hour": fixture["prior_rates"],
        "train": [{"id": "old_train", "scenario": str(fixture["scenario"]), "seed": archived.seed}],
        "test": [{"id": "old_test", "scenario": str(test_path), "graph": str(fixture["graph"]), "seed": 99}]}}}))
    old_hashes = {p: digest(p) for p in (reference, test_path, Path(fixture["scenario"]))}
    output = tmp_path / "fresh" / "manifest.json"
    options = dict(train_per_city=2, validation_per_city=1, processes=1,
                   memory_per_worker_gib=.01, reserve_gib=0)
    manifest = prepare(reference, fixture["spatial_model"], fixture["paper_model"], output, **options)
    assert [r["split"] for r in manifest["episodes"]] == ["train", "train", "validation"]
    fresh = [Scenario.load_json(r["scenario"]) for r in manifest["episodes"]]
    assert len({s.seed for s in fresh}) == 3
    assert not {s.seed for s in fresh} & {archived.seed, 99}
    assert len({workload_identity(s) for s in fresh}) == 3
    assert workload_identity(archived) not in {workload_identity(s) for s in fresh}
    training = [r for r in manifest["episodes"] if r["split"] == "train"]
    for level in ("1", "2", "5"):
        expected = (sum(r["importance_counts"][level] for r in training) + .5) / 2
        assert all(r["prior_rates_per_hour"][level] == expected for r in manifest["episodes"])
    comparison = json.loads(output.with_name("manifest_comparison.json").read_text())
    assert comparison["episodes"] == [{"id": "old_test", "city": "city", "split": "test",
        "graph": str(fixture["graph"]), "scenario": str(test_path),
        "prior_rates_per_hour": fixture["prior_rates"]}]
    assert old_hashes == {p: digest(p) for p in old_hashes}
    before = {r["scenario"]: (digest(r["scenario"]), Path(r["scenario"]).stat().st_mtime_ns)
              for r in manifest["episodes"]}
    again = prepare(reference, fixture["spatial_model"], fixture["paper_model"], output,
                    resume=True, **options)
    assert manifest == again
    assert before == {p: (digest(p), Path(p).stat().st_mtime_ns) for p in before}
    with pytest.raises(ValueError, match="incompatible generation"):
        prepare(reference, fixture["spatial_model"], fixture["paper_model"], output,
                base_seed=123, resume=True, **options)


def test_parallel_city_generation_has_globally_distinct_data(fixture, tmp_path):
    reference = tmp_path / "suite.json"
    record = {"graph": str(fixture["graph"]), "prior_rates_per_hour": fixture["prior_rates"],
              "test": [{"id": "reserved", "scenario": str(fixture["scenario"]),
                        "graph": str(fixture["graph"]), "seed": 42}]}
    reference.write_text(json.dumps({"duration_hours": 1, "cities": {"a": record, "b": record}}))
    result = prepare(reference, fixture["spatial_model"], fixture["paper_model"],
                     tmp_path / "parallel/manifest.json", train_per_city=1, validation_per_city=1,
                     processes=3, memory_per_worker_gib=.01, reserve_gib=0)
    assert len(result["episodes"]) == 4
    assert len({r["seed"] for r in result["episodes"]}) == 4
    assert len({r["workload_sha256"] for r in result["episodes"]}) == 4
