"""Split, launcher and predictor checks; deliberately never calls fit()."""
import json
from pathlib import Path

import pytest

from fleet_switching import LABEL_VERSION
from fleet_switching.cli import main
from fleet_switching.config import Config, DEVELOPMENT, HELD_OUT, POLICIES, guard_city, workers
from fleet_switching.models import Predictor, check_rows
from fleet_switching.workflow import estimate


def config(**kw):
    return Config(backend_root="unused",map_manifest="unused",output="unused",
                  spatial_model="unused",paper_model="unused",**kw)


@pytest.mark.parametrize('city', sorted(HELD_OUT))
@pytest.mark.parametrize('split', ['train','validation'])
def test_every_reserved_city_is_rejected_from_development(city,split):
    with pytest.raises(ValueError,match="forbidden"):
        guard_city(city,split)


def test_cpu_admission_has_no_legacy_sixty_process_cap():
    assert workers(config(),72,171)==64
    assert workers(config(worker_limit=80,reserved_cpus=5),72,200)==67
    assert workers(config(),72,63)==0


def test_full_plan_counts_both_independent_collectors_and_all_actions():
    result=estimate(config())
    assert result['training_episodes_total']==1600
    assert result['validation_episodes_total']==400
    assert result['training_branches_approx']==800000
    assert result['future_test_scenarios']==200
    assert result['old_label_rows_reused']==0
    assert result['launches_simulations'] is False


@pytest.mark.parametrize('command',['prepare','train','prepare-test','evaluate'])
def test_mutating_commands_do_nothing_without_execute(command,tmp_path):
    nonexistent=tmp_path/'missing.json'
    with pytest.raises(SystemExit) as exc:
        main([command,'--config',str(nonexistent)])
    assert exc.value.code==2
    assert list(tmp_path.iterdir())==[]


def test_model_fitting_rejects_test_rows_before_any_learning():
    row={'id':'one','city':'ghent','split':'train','label_version':LABEL_VERSION,
         'features':{'state.observed':0.},'costs':dict.fromkeys(POLICIES,1.),'robots':1}
    with pytest.raises(ValueError,match='forbidden'):
        check_rows([row])
    row.update(city='haifa',split='validation')
    with pytest.raises(ValueError,match='Only compatible training'):
        check_rows([row])


def test_mlp_export_forward_pass_without_optimization():
    payload={'kind':'nn','label_version':LABEL_VERSION,'policies':list(POLICIES),
             'feature_names':['x'],'feature_mean':[0.],'feature_scale':[1.],
             'weights':[[[1.]],[[1.]],[[0.]*len(POLICIES)]],
             'biases':[[0.],[0.],[float(i) for i in range(len(POLICIES))]],
             'target_scale':1.,'target_mean':[0.]*len(POLICIES)}
    assert Predictor(json.loads(json.dumps(payload))).choose({'x':2.})==POLICIES[0]
    with pytest.raises(ValueError,match='schema'):
        Predictor(payload).choose({'future':2.})


def test_manifest_cannot_reclassify_a_test_city(tmp_path):
    maps=[{'key':c,'role':'development'} for c in DEVELOPMENT]
    maps += [{'key':c,'role':'reserved_final_test'} for c in HELD_OUT]
    next(r for r in maps if r['key']=='ghent')['role']='development'
    path=tmp_path/'maps.json'
    path.write_text(json.dumps({'status':'complete','maps':maps}))
    cfg=Config(backend_root='unused',map_manifest=str(path),output='unused',spatial_model='unused',paper_model='unused')
    with pytest.raises(ValueError,match='allowlist'):
        cfg.maps('train')
