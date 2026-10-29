"""Five-action selection, collector, model loading and matched validation."""
from dataclasses import replace
from pathlib import Path

import pytest

from fleet_switching import collection
from fleet_switching.config import TOP5_NO_FULL_MLE_POLICIES, NO_FULL_MLE_POLICIES, atomic, read, sha
from fleet_switching.models import Predictor, check_rows
from fleet_switching.provenance import ensure_campaign, science_id
from fleet_switching.reporting import compare
from fleet_switching.workflow import _fixed_models, estimate
from test_switching_collector import FakeBackend, FakeRuntime, fixture
from test_switching_hardening import campaign_fixture, comparison_fixture
from test_switching_portfolio import training_rows

POLICIES = TOP5_NO_FULL_MLE_POLICIES
BOOTSTRAP = 'full_coordinated_idle'


def test_selection_is_best_five_non_full_mle_from_previous_development_validation():
    path = Path(__file__).resolve().parents[1]/'review_bundle/evidence/final_test_launch_20261026/validation_summary.json'
    # Integration-only test: no test outcomes are read.
    report = read(path)
    assert {r['split'] for r in report['results']} == {'validation'}
    ranked = sorted((name.removeprefix('fixed__'),value['equal_city_mean_loss_per_robot'])
        for name,value in report['scores'].items()
        if name.startswith('fixed__') and name!='fixed__full_mle_reservation')
    ranked.sort(key=lambda row:(row[1],row[0]))
    assert {name for name,_ in ranked[:5]} == set(POLICIES)
    assert ranked[0][0] == BOOTSTRAP


class FiveRuntime(FakeRuntime):
    def advance(self, end):
        assert self.now == 0
        self.now = end
        self.loss = 1. if self.policy == 'full' else 10.
        self.ns['delivered'] = 1


class FiveBackend(FakeBackend):
    def __init__(self, *args):
        self.runtime = FiveRuntime()


def test_five_teacher_branches_preserve_actor_selected_trajectory(tmp_path):
    cfg, episode, model = fixture(tmp_path)
    cfg = replace(cfg, policies=POLICIES, bootstrap_policy=BOOTSTRAP)
    atomic(model, {'kind':'bootstrap','policy':BOOTSTRAP,'policies':list(POLICIES),
                   'label_version':read(model)['label_version']})
    atomic(model.parent/'current.json', {'path':str(model),'sha256':sha(model)})
    result = collection.collect_episode(cfg, episode, 'nn', 'science', backend_factory=FiveBackend)
    row = read(next((Path(cfg.output)/'actors/nn/train'/episode['id']/'rows').glob('*.json')))
    assert set(row['costs']) == set(row['diagnostics']) == set(POLICIES)
    assert len(row['costs']) == 5
    assert row['costs']['full'] == 1.
    assert result['loss'] == 10.
    assert result['policy_minutes'] == {BOOTSTRAP:15.}
    plan = estimate(cfg)
    assert plan['benchmarks_per_training_state'] == 5
    assert plan['training_branches_approx'] == 400000
    assert plan['validation_episodes_total'] + plan['validation_fixed_benchmark_episodes_total'] == 1400


def test_nine_action_replay_actor_and_campaign_are_incompatible_with_five(tmp_path):
    cfg, _ = campaign_fixture(tmp_path)
    nine = replace(cfg, policies=NO_FULL_MLE_POLICIES)
    ensure_campaign(nine, create=True)
    five = replace(cfg, policies=POLICIES, bootstrap_policy=BOOTSTRAP)
    assert science_id(nine) != science_id(five)
    with pytest.raises(ValueError, match='Campaign identity'):
        ensure_campaign(five)
    with pytest.raises(ValueError, match='schema'):
        check_rows(training_rows('nn', NO_FULL_MLE_POLICIES), POLICIES)
    (tmp_path/'fixture').mkdir()
    cfg, _, model = fixture(tmp_path/'fixture')
    payload = read(model)
    payload.update(policies=list(NO_FULL_MLE_POLICIES))
    with pytest.raises(ValueError, match='identity mismatch'):
        Predictor(payload, POLICIES)


def test_five_action_config_rejects_removed_bootstrap_and_malformed_model(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='Bootstrap'):
        replace(cfg, policies=POLICIES)
    payload = {'kind':'nn','label_version':'projected_completion_fifo_v2_event_order',
        'policies':list(POLICIES),'weights':[[[1.]],[[1.]],[[0.]*9]],
        'biases':[[0.],[0.],[0.]*9],'target_mean':[0.]*9}
    with pytest.raises(ValueError, match='output schema'):
        Predictor(payload, POLICIES)


def test_five_fixed_policies_and_two_selectors_are_required_for_validation(tmp_path):
    episodes, results = comparison_fixture()
    results = [r for r in results if not r['family'].startswith('fixed__') or r['family'][7:] in POLICIES]
    report = compare(results, episodes, policies=POLICIES)
    assert report['completed_results'] == 7
    assert set(report['policies']) == set(POLICIES)
    with pytest.raises(ValueError, match='All selector'):
        compare(results[:-1], episodes, policies=POLICIES)
    cfg, _, _ = fixture(tmp_path)
    cfg = replace(cfg, policies=POLICIES, bootstrap_policy=BOOTSTRAP)
    assert set(_fixed_models(cfg, 'science')) == set(POLICIES)
