"""Deterministic test doubles only. Never constructs FleetRuntime or trains."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleet_switching import LABEL_VERSION
from fleet_switching import collection
from fleet_switching.config import Config, POLICIES, atomic, read, sha


class FakeRuntime:
    def __init__(self):
        self.now,self.loss,self.policy=0.,0.,'reactive_insertion'
        self.ns={'robots':[0],'delivered':0}
    def outstanding(self): return self.ns['delivered']==0
    def snapshot(self):
        return json.dumps([self.now,self.loss,self.policy,self.ns['delivered']]).encode()
    def restore(self,blob):
        self.now,self.loss,self.policy,self.ns['delivered']=json.loads(blob)
    def set_policy(self,policy): self.policy=policy
    def advance(self,end):
        assert self.now==0  # All candidates must start at the same boundary.
        self.now=end
        self.loss=1. if self.policy=='myopic_ab' else 10.
        self.ns['delivered']=1


class FakeBackend:
    def __init__(self,*args): self.runtime=FakeRuntime()
    def features(self,*args): return {'state.observed':0.}
    def project(self):
        # An accidental scoring mutation must never enter the next actual state.
        self.runtime.loss=99999.
        return SimpleNamespace(loss=0.),{'projected_queue_wait_min':0.}
    def close(self): pass


def fixture(tmp_path):
    graph=tmp_path/'graph.json'; graph.write_text('{}')
    scenario=tmp_path/'scenario.json'; scenario.write_text('{}')
    cfg=Config(backend_root='unused',map_manifest='unused',output=str(tmp_path/'out'),spatial_model=str(graph),paper_model=str(graph))
    episode={'id':'haifa_train_000','city':'haifa','split':'train','seed':1,'scenario':str(scenario),
             'scenario_sha256':sha(scenario),'graph':str(graph),'graph_sha256':sha(graph),
             'workload_sha256':'one','orders':1}
    model=Path(cfg.output)/'models/nn/bootstrap.json'
    atomic(model,{'kind':'bootstrap','policy':'reactive_insertion','policies':list(POLICIES),'label_version':LABEL_VERSION})
    atomic(model.parent/'current.json',{'path':str(model),'sha256':sha(model)})
    return cfg,episode,model


def test_trajectory_follows_actor_not_best_teacher_and_projection_is_isolated(tmp_path):
    cfg,e,_=fixture(tmp_path)
    result=collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    assert result['loss']==10.  # Oracle would choose myopic, actor chose reactive.
    row=read(next((Path(cfg.output)/'actors/nn/train'/e['id']/'rows').glob('*.json')))
    assert row['costs']['myopic_ab']==1.
    assert len(row['costs'])==10
    assert row['intent']['chosen_policy']=='reactive_insertion'


def test_resume_after_row_write_preserves_original_choice(tmp_path,monkeypatch):
    cfg,e,model=fixture(tmp_path)
    original=collection.atomic
    def fail_checkpoint(path,value):
        if Path(path).name=='checkpoint.json':
            raise RuntimeError('injected interruption')
        return original(path,value)
    monkeypatch.setattr(collection,'atomic',fail_checkpoint)
    with pytest.raises(RuntimeError,match='injected'):
        collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    payload=read(model); payload['policy']='myopic_ab'; atomic(model,payload)
    atomic(model.parent/'current.json',{'path':str(model),'sha256':sha(model)})
    monkeypatch.setattr(collection,'atomic',original)
    result=collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    assert result['loss']==10.


def test_validation_cannot_follow_live_model_registry(tmp_path):
    cfg,e,_=fixture(tmp_path); e['split']='validation'
    with pytest.raises(ValueError,match='explicitly frozen'):
        collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)


def test_empty_frozen_pointer_cannot_fall_back_to_live_registry(tmp_path):
    cfg,e,_=fixture(tmp_path); e['split']='validation'
    with pytest.raises(ValueError,match='explicitly frozen'):
        collection.collect_episode(cfg,e,'nn','test-science',frozen={},backend_factory=FakeBackend)


def test_changed_frozen_model_rejected_before_loading_backend(tmp_path):
    cfg,e,model=fixture(tmp_path); e['split']='validation'
    pointer={'path':str(model),'sha256':'wrong','science_id':'test-science'}
    with pytest.raises(ValueError,match='identity changed'):
        collection.collect_episode(cfg,e,'nn','test-science',frozen=pointer,backend_factory=FakeBackend)


class TwoStepRuntime(FakeRuntime):
    def advance(self,end):
        assert end==self.now+15
        self.now=end
        self.loss+=1. if self.policy=='myopic_ab' else 10.
        self.ns['delivered']=int(end>=30)


class TwoStepBackend(FakeBackend):
    def __init__(self,*args): self.runtime=TwoStepRuntime()


def test_corrupt_latest_state_recovers_previous_pair_and_original_actor(tmp_path):
    cfg,e,model=fixture(tmp_path)
    result=collection.collect_episode(cfg,e,'nn','test-science',backend_factory=TwoStepBackend)
    assert result['loss']==20.
    folder=Path(cfg.output)/'actors/nn/train'/e['id']
    (folder/'complete.json').unlink()
    cp=read(folder/'checkpoint.json')
    Path(cp['state_path']).write_bytes(b'corrupted newest state')
    payload=read(model); payload['policy']='myopic_ab'; atomic(model,payload)
    atomic(model.parent/'current.json',{'path':str(model),'sha256':sha(model)})
    recovered=collection.collect_episode(cfg,e,'nn','test-science',backend_factory=TwoStepBackend)
    assert recovered['loss']==20.
    assert recovered['policy_minutes']=={'reactive_insertion':30.}
    assert recovered['switches_after_initial_choice']==0
    previous=read(folder/'checkpoint.previous.json')
    assert previous['policy_minutes']=={'reactive_insertion':15.}


def test_changed_base_weights_invalidate_completed_episode(tmp_path):
    cfg,e,_=fixture(tmp_path)
    collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    Path(cfg.spatial_model).write_bytes(b'changed fixed model')
    with pytest.raises(ValueError,match='input bytes changed|different experiment'):
        collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)


def test_first_intent_restart_uses_persisted_initial_state(tmp_path,monkeypatch):
    cfg,e,_=fixture(tmp_path)
    original=collection.Predictor.choose
    calls=[]
    def choose_then_fail(self,features):
        calls.append('choose')
        return original(self,features)
    monkeypatch.setattr(collection.Predictor,'choose',choose_then_fail)
    original_advance=FakeRuntime.advance
    monkeypatch.setattr(FakeRuntime,'advance',lambda *args:(_ for _ in ()).throw(RuntimeError('after intent')))
    with pytest.raises(RuntimeError,match='after intent'):
        collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    folder=Path(cfg.output)/'actors/nn/train'/e['id']
    assert (folder/'states/initial.pickle').is_file()
    monkeypatch.setattr(FakeRuntime,'advance',original_advance)
    collection.collect_episode(cfg,e,'nn','test-science',backend_factory=FakeBackend)
    assert calls==['choose']
