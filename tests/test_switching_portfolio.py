"""Nine-action ablation: synthetic collector, real tiny fits and split guards."""
from dataclasses import replace
from pathlib import Path

import pytest

from fleet_switching import collection
from fleet_switching.config import NO_FULL_MLE_POLICIES, POLICIES, atomic, read, sha
from fleet_switching.models import Predictor, check_rows, fit
from fleet_switching.provenance import ensure_campaign, learner_id, row_digest, science_id
from fleet_switching.reporting import compare
from fleet_switching.workflow import _fixed_models, estimate
from test_switching_collector import FakeBackend, fixture
from test_switching_hardening import campaign_fixture, comparison_fixture, row_fixture


def test_only_full_mle_is_removed_and_workload_is_unchanged(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    cfg = replace(cfg, policies=NO_FULL_MLE_POLICIES)
    assert cfg.policies == POLICIES[:-1]
    assert 'queue_mle_reservation' in cfg.policies
    plan = estimate(cfg)
    assert plan['training_episodes_total'] == 1600
    assert plan['training_branches_approx'] == 720000
    assert plan['validation_fixed_benchmark_episodes_total'] == 1800
    assert plan['validation_episodes_total'] == 400
    assert plan['benchmarks_per_training_state'] == 9
    assert plan['decision_minutes'] == 15


def test_collector_scores_nine_branches_and_follows_actor(tmp_path):
    cfg, episode, model = fixture(tmp_path)
    cfg = replace(cfg, policies=NO_FULL_MLE_POLICIES)
    payload = read(model)
    payload['policies'] = list(cfg.policies)
    atomic(model, payload)
    atomic(model.parent/'current.json', {'path':str(model), 'sha256':sha(model)})
    result = collection.collect_episode(cfg, episode, 'nn', 'science', backend_factory=FakeBackend)
    row = read(next((Path(cfg.output)/'actors/nn/train'/episode['id']/'rows').glob('*.json')))
    assert tuple(row['costs']) != ()
    assert set(row['costs']) == set(cfg.policies) == set(row['diagnostics'])
    assert 'full_mle_reservation' not in row['costs']
    assert row['intent']['chosen_policy'] == 'reactive_insertion'
    assert result['loss'] == 10.  # Teacher's preferred action has loss 1.
    assert result['policy_minutes'] == {'reactive_insertion':15.}


def test_ten_action_actor_cannot_enter_nine_action_collector(tmp_path):
    cfg, episode, _ = fixture(tmp_path)
    cfg = replace(cfg, policies=NO_FULL_MLE_POLICIES)
    with pytest.raises(ValueError, match='identity mismatch'):
        collection.collect_episode(cfg, episode, 'nn', 'science', backend_factory=FakeBackend)


def test_portfolio_changes_all_training_identities_and_cannot_reuse_campaign(tmp_path):
    cfg, _ = campaign_fixture(tmp_path)
    original = ensure_campaign(cfg, create=True)
    assert ensure_campaign(cfg) == original
    ablation = replace(cfg, policies=NO_FULL_MLE_POLICIES)
    assert science_id(ablation) != science_id(cfg)
    for family in ('nn', 'xgboost'):
        assert learner_id(ablation, family) != learner_id(cfg, family)
    with pytest.raises(ValueError, match='Campaign identity'):
        ensure_campaign(ablation)
    ablation = replace(ablation, output=str(tmp_path/'ablation'))
    created = ensure_campaign(ablation, create=True)
    assert ensure_campaign(ablation) == created
    assert created['settings']['policies'] == list(NO_FULL_MLE_POLICIES)


def training_rows(family, policies):
    rows = []
    for i in range(32):
        row = row_fixture(i)
        row.update(family=family, features={'state.observed':float(i % 8)},
                   costs={p:float(1 + (i % 8 - j)**2) for j,p in enumerate(policies)})
        row['content_sha256'] = row_digest(row)
        rows.append(row)
    return rows


@pytest.mark.parametrize('family', ['nn', 'xgboost'])
@pytest.mark.parametrize('policies', [POLICIES, NO_FULL_MLE_POLICIES])
def test_small_real_fit_exports_exactly_configured_heads(tmp_path, family, policies):
    payload = fit(training_rows(family, policies), family, threads=1, policies=policies)
    assert payload['policies'] == list(policies)
    assert (len(payload['boosters']) if family == 'xgboost'
            else len(payload['weights'][-1][0])) == len(policies)
    path = tmp_path/'model.json'
    atomic(path, payload)
    predictor = Predictor.load(path, sha(path), policies)
    for value in range(8):
        assert predictor.choose({'state.observed':float(value)}) in policies
    other = NO_FULL_MLE_POLICIES if policies == POLICIES else POLICIES
    with pytest.raises(ValueError, match='identity mismatch'):
        Predictor.load(path, sha(path), other)


def test_fit_rejects_extra_or_missing_action_labels():
    with pytest.raises(ValueError, match='schema'):
        check_rows(training_rows('nn', POLICIES), NO_FULL_MLE_POLICIES)
    with pytest.raises(ValueError, match='schema'):
        check_rows(training_rows('nn', NO_FULL_MLE_POLICIES), POLICIES)


def test_ablation_validation_requires_two_selectors_and_nine_fixed_policies(tmp_path):
    episodes, results = comparison_fixture()
    results = [r for r in results if r['family'] != 'fixed__full_mle_reservation']
    report = compare(results, episodes, policies=NO_FULL_MLE_POLICIES)
    assert report['completed_results'] == 11
    assert 'full_mle_reservation' not in report['paired']['nn']
    with pytest.raises(ValueError, match='All selector'):
        compare(results[:-1], episodes, policies=NO_FULL_MLE_POLICIES)
    cfg, _, _ = fixture(tmp_path)
    baselines = _fixed_models(replace(cfg, policies=NO_FULL_MLE_POLICIES), 'science')
    assert set(baselines) == set(NO_FULL_MLE_POLICIES)


def test_removed_bootstrap_and_reordered_portfolio_are_rejected(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='Bootstrap'):
        replace(cfg, policies=NO_FULL_MLE_POLICIES, bootstrap_policy='full_mle_reservation')
    with pytest.raises(ValueError, match='ordering'):
        replace(cfg, policies=tuple(reversed(NO_FULL_MLE_POLICIES)))
