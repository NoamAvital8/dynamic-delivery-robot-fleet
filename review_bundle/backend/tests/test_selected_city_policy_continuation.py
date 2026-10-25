import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'operations'))
import continue_selected_city_policies as controller


def result_job(tmp_path):
    return {'city': 'paris', 'scenario_id': 'paris_test_000', 'seed': 7,
            'orders': 3, 'policy': 'full', 'output': tmp_path/'result.json',
            'log': tmp_path/'runner.log', 'status_path': tmp_path/'status.json',
            'stamp_path': tmp_path/'stamp.json', 'signature': 'same',
            'command': ['python', 'simulator.py']}


def write_result(job):
    job['output'].write_text(json.dumps({'orders': 3, 'delivered': 3, 'on_time': 2,
                                       'scenario_seed': 7, 'loss_objective': 42.,
                                       'wall_clock_seconds': 100., 'simulation_finish_min': 800.}))


def test_exact_ten_policies_and_no_removed_variants():
    assert len(controller.SELECTED_POLICIES) == 10
    assert not {'full_uncertainty_idle', 'full_uncertainty_idle_no_nn',
                'full_queue_aware_no_nn', 'full_coordinated_idle_no_nn'} & set(controller.SELECTED_POLICIES)


def test_live_adoption_never_launches_or_changes_result(tmp_path, monkeypatch):
    job = result_job(tmp_path)
    write_result(job)
    original = job['output'].read_bytes()
    sequence = iter([True, True, False])
    monkeypatch.setattr(controller, 'same_live_process', lambda record: next(sequence))
    monkeypatch.setattr(controller.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(controller, '_execute', lambda *a: pytest.fail('must not relaunch'))
    row = controller.adopt_simulator(job, {'command': job['command'], 'signature': 'same'})
    assert row['status'] == 'complete'
    assert job['output'].read_bytes() == original
    assert json.loads(job['stamp_path'].read_text()) == {'signature': 'same'}


def test_failed_preserved_simulator_is_not_restarted(tmp_path, monkeypatch):
    job = result_job(tmp_path)
    monkeypatch.setattr(controller, 'same_live_process', lambda record: False)
    monkeypatch.setattr(controller, '_execute', lambda *a: pytest.fail('must not restart'))
    with pytest.raises(FileNotFoundError):
        controller.adopt_simulator(job, {'command': job['command'], 'signature': 'same'})


def test_pid_reuse_is_not_a_live_original_process(monkeypatch):
    monkeypatch.setattr(controller, 'process_record', lambda pid: {'start_ticks': 20, 'state': 'R'})
    assert not controller.same_live_process({'pid': 123, 'start_ticks': 19})


def test_unclaimed_running_job_cannot_be_duplicated(tmp_path):
    job = result_job(tmp_path)
    job['status_path'].write_text(json.dumps({'status': 'running'}))
    with pytest.raises(RuntimeError, match='unclaimed'):
        controller.new_job_is_unclaimed(job, {})
    controller.new_job_is_unclaimed(job, {str(job['output']): {}})


def test_incompatible_stamped_output_rejected(tmp_path):
    job = result_job(tmp_path)
    write_result(job)
    job['stamp_path'].write_text(json.dumps({'signature': 'different'}))
    with pytest.raises(RuntimeError, match='incompatible'):
        controller.inspect_jobs([job])


def test_nonfinite_result_rejected(tmp_path):
    job = result_job(tmp_path)
    write_result(job)
    data = json.loads(job['output'].read_text()); data['loss_objective'] = float('nan')
    job['output'].write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='invalid'):
        controller.validated_row(job)


def test_300_jobs_keep_original_signatures(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, '_source_digest', lambda: 'original-source')
    spatial, paper = tmp_path/'spatial.npz', tmp_path/'paper.npz'
    spatial.write_bytes(b'spatial'); paper.write_bytes(b'paper')
    suite = {'cities': {}}
    hashes = {}
    for city in controller.EXPECTED_CITIES:
        graph = tmp_path/f'{city}.graphml'; graph.write_bytes(city.encode())
        hashes[str(graph)] = controller._sha256(graph)
        records = []
        for i in range(5):
            scenario = tmp_path/f'{city}_{i}.json'; scenario.write_bytes(str(i).encode())
            records.append({'id': f'{city}_test_{i:03}', 'scenario': str(scenario), 'seed': i, 'orders': 3})
        suite['cities'][city] = {'graph': str(graph), 'test': records, 'prior_rates_per_hour': {'1': 1.}}
    suitepath = tmp_path/'suite.json'; suitepath.write_text(json.dumps(suite))
    mle = controller.MLEConfig()
    from dataclasses import asdict
    manifest = {'source_sha256': 'original-source', 'spatial_model': str(spatial),
                'paper_model': str(paper), 'model_sha256': {'spatial': controller._sha256(spatial),
                'paper': controller._sha256(paper)}, 'suite': str(suitepath), 'graph_sha256': hashes,
                'timeout_seconds': 0, 'processes': 12, 'idle_processes': 4, 'shortlist_k': 10,
                'mle_reservation_config': asdict(mle), 'idle_relocation_uncertainty_penalty': 1.,
                'idle_switch_gain_fraction': .005, 'idle_retarget_cooldown_min': 2., 'job_order': 'round-robin'}
    jobs = controller.build_jobs(tmp_path, manifest)
    assert len(jobs) == 300
    assert len({j['output'] for j in jobs}) == 300
    assert {j['city'] for j in jobs[:6]} == set(controller.EXPECTED_CITIES)
    for job in jobs:
        original_signature = hashlib.sha256(json.dumps({
            'source': 'original-source', 'graph': hashes[job['command'][3]],
            'scenario': controller._sha256(Path(job['command'][5])),
            'models': manifest['model_sha256'], 'command': job['command'],
        }, sort_keys=True).encode()).hexdigest()
        assert job['signature'] == original_signature
