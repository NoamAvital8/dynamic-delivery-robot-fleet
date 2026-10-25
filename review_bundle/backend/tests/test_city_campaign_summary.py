import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'operations'))
from show_city_campaign_summary import collect, render


def campaign(tmp_path):
    folder = tmp_path/'benchmarks'; folder.mkdir()
    suite = tmp_path/'suite.json'
    suite.write_text(json.dumps({'cities': {'paris': {'test': [
        {'id': 'paris_test_000', 'seed': 1, 'orders': 3},
        {'id': 'paris_test_001', 'seed': 2, 'orders': 3}]}}}))
    (folder/'run_manifest.json').write_text(json.dumps({'suite': str(suite), 'policies': ['full', 'myopic_ab']}))
    return tmp_path


def completed(root, policy='full'):
    folder = root/'benchmarks/paris/paris_test_000'/policy
    folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'status': 'complete'}))
    (folder/'stamp.json').write_text(json.dumps({'signature': 'valid'}))
    (folder/'result.json').write_text(json.dumps({'orders': 3, 'scenario_seed': 1,
        'loss_objective': 1234.56, 'delivered': 3, 'on_time': 2}))
    return folder


def test_counts_pending_and_only_selected_policies(tmp_path):
    root = campaign(tmp_path); completed(root)
    completed(root, 'excluded_old_policy')
    data = collect(root)
    assert data['expected'] == 4
    assert data['counts'] == {'complete': 1, 'pending': 3}
    assert len(data['results']) == 2
    row = next(r for r in data['results'] if r['policy'] == 'full')
    assert row['loss'] == 1234.56 and row['late'] == 1
    assert '1,235 (1/2)' in render(data)
    assert 'Late' in render(data, deliveries=True)


def test_running_output_is_not_a_finished_evaluation(tmp_path):
    root = campaign(tmp_path); folder = completed(root)
    (folder/'status.json').write_text(json.dumps({'status': 'running'}))
    data = collect(root)
    assert data['counts'] == {'running': 1, 'pending': 3}
    assert all(r['completed'] == 0 for r in data['results'])


def test_invalid_loss_raises_instead_of_printing_it(tmp_path):
    root = campaign(tmp_path); folder = completed(root)
    path = folder/'result.json'; data = json.loads(path.read_text()); data['loss_objective'] = float('inf')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='invalid completed result'):
        collect(root)


def test_summary_is_read_only(tmp_path):
    root = campaign(tmp_path); completed(root)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    collect(root)
    after = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before == after
