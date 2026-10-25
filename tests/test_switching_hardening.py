"""Synthetic protocol, content-integrity and queue fixtures. No fitting/simulation."""
from dataclasses import replace
import gc
import json
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import pytest

from fleet_switching import LABEL_VERSION, backend_lifecycle, replay
from fleet_switching.config import Config, DEVELOPMENT, HELD_OUT, POLICIES, atomic, identity, read, sha
from fleet_switching.data import dataset_receipt, load_dataset, prepare
from fleet_switching.projection import OrderCost, Session, Step, project
from fleet_switching.provenance import ensure_campaign, row_digest, science_id, validate_row
from fleet_switching.reporting import FIXED_PREFIX, compare
from fleet_switching.workflow import train


def test_handling_completion_precedes_travel_at_equal_time():
    plans = {1:(Step('travel',1),Step('charge',10,7),Step('travel',1),Step('service',1,dropoff=1)),
             2:(Step('service',1),Step('charge',1,7),Step('travel',1),Step('service',1,dropoff=2))}
    result = project(0,plans,{7:1},{1:OrderCost(0,100,1),2:OrderCost(0,100,1)})
    assert result.completed == {2:4,1:14}
    assert result.loss == 18


@pytest.mark.parametrize('ids',[(1,2),(2,1)])
def test_saved_arrival_sequence_overrides_robot_id(ids):
    first, second = ids
    plans = {first:(Step('travel',1,event_order=20),Step('charge',10,7),Step('service',1,dropoff=1)),
             second:(Step('travel',1,event_order=10),Step('charge',1,7),Step('service',1,dropoff=2))}
    result = project(0,plans,{7:1},{1:OrderCost(0,100,1),2:OrderCost(0,100,5)},sequence_start=100)
    assert result.completed == {2:3,1:13}
    assert result.loss == 28


def test_duplicate_physical_event_ids_fail_closed():
    with pytest.raises(ValueError,match='Duplicate saved'):
        project(0,{1:(Step('travel',1,event_order=2),),2:()}, {7:1},{},active=[Session(2,7,1,2)])


def test_legacy_hook_installation_is_idempotent_and_cleanup_releases_graph():
    calls = []
    fast = NS(_install_in_memory_graph_initialization=lambda:calls.append('graph'),
        _install_compiled_distance_backend=lambda:(calls.append('distance') or object()),
        _oracle_by_graph_id={},_distance_rows={})
    backend_lifecycle.install(fast)
    graph_hook, distance_hook = fast._install_in_memory_graph_initialization, fast._install_compiled_distance_backend
    for _ in range(5):
        backend_lifecycle.install(fast)
        graph_hook()
        distance_hook()
    assert calls == ['graph','distance']
    assert fast._install_in_memory_graph_initialization is graph_hook
    class Graph: pass
    class Oracle: pass
    graph, oracle = Graph(), Oracle()
    oracle.graph = graph
    ref = weakref.ref(graph)
    fast._oracle_by_graph_id[id(graph)] = oracle
    fast._distance_rows[(id(oracle),1)] = 'own row'
    fast._distance_rows[(999,1)] = 'another runtime'
    backend_lifecycle.release(fast,graph,oracle)
    del graph,oracle
    gc.collect()
    assert ref() is None
    assert fast._distance_rows == {(999,1):'another runtime'}


def campaign_fixture(tmp_path):
    backend = tmp_path/'backend'
    for relative in ('src/delivery_fleet/dummy.py','scripts/dummy.py','operations/exact_runtime.py'):
        path = backend/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# inspection-only fixture\n')
    weights = tmp_path/'weights.npz'
    weights.write_bytes(b'fixed weight bytes; never loaded')
    rows=[]
    for city in sorted(DEVELOPMENT | HELD_OUT):
        graph = tmp_path/(city+'.graphml')
        graph.write_text('synthetic identity fixture '+city)
        rows.append({'key':city,'role':'development' if city in DEVELOPMENT else 'reserved_final_test',
                     'path':str(graph),'sha256':sha(graph)})
    manifest = tmp_path/'maps.json'
    atomic(manifest,{'status':'complete','maps':rows})
    config = Config(backend_root=str(backend),map_manifest=str(manifest),output=str(tmp_path/'run'),
        spatial_model=str(weights),paper_model=str(weights),train_per_city=1,validation_per_city=1,test_per_city=1)
    return config, rows


def test_campaign_rejects_changed_weights_but_accepts_worker_limit(tmp_path):
    config,_=campaign_fixture(tmp_path)
    campaign = ensure_campaign(config,create=True)
    assert ensure_campaign(replace(config,worker_limit=2)) == campaign
    Path(config.spatial_model).write_bytes(b'changed fixed dependency')
    with pytest.raises(ValueError,match='Campaign identity'):
        ensure_campaign(config)


def test_science_includes_exact_operations_dependency(tmp_path):
    config,_=campaign_fixture(tmp_path)
    before = science_id(config)
    (Path(config.backend_root)/'operations/exact_runtime.py').write_text('# changed routing behavior\n')
    assert science_id(config) != before


def test_campaign_rejects_changed_online_update_cadence(tmp_path):
    config,_=campaign_fixture(tmp_path)
    ensure_campaign(config,create=True)
    with pytest.raises(ValueError,match='Campaign identity'):
        ensure_campaign(replace(config,update_every_states=config.update_every_states//2))


def test_manifest_requires_every_city_and_exact_split_counts(tmp_path):
    config,maps=campaign_fixture(tmp_path)
    campaign=ensure_campaign(config,create=True)
    episodes=[]
    for graph in maps:
        if graph['key'] not in DEVELOPMENT:
            continue
        for split in ('train','validation'):
            path=tmp_path/(graph['key']+'_'+split+'.json')
            atomic(path,{'orders':[{'id':1}],'fixture':path.name})
            episodes.append({'id':path.stem,'city':graph['key'],'split':split,'seed':len(episodes),
                'scenario':str(path),'scenario_sha256':sha(path),'workload_sha256':path.stem,
                'graph':graph['path'],'graph_sha256':graph['sha256'],'orders':1,'fleet_count':1})
    data={'label_version':LABEL_VERSION,'science_id':campaign['science_id'],'campaign_id':campaign['campaign_id'],
        'config':{key:getattr(config,key) for key in ('base_seed','duration_hours','decision_minutes','train_per_city','validation_per_city','test_per_city')},
        'episodes':episodes}
    path=Path(config.output)/'datasets/development_manifest.json'
    atomic(path,data)
    atomic(path.with_name('development_receipt.json'),dataset_receipt(data,path))
    assert len(load_dataset(config)['episodes'])==16
    data['episodes']=episodes[1:]
    atomic(path,data)
    with pytest.raises(ValueError,match='cardinalities'):
        load_dataset(config)


def row_fixture(step=0):
    row={'id':str(step),'episode':'haifa_train_000','family':'nn','science_id':'science','label_version':LABEL_VERSION,
        'city':'haifa','split':'train','seed':1,'scenario_sha256':'scenario','graph_sha256':'graph','workload_sha256':'workload',
        'step':step,'robots':1,'features':{'state.observed':2.},'costs':dict.fromkeys(POLICIES,3.),
        'intent':{'chosen_policy':'reactive_insertion'},'next_state_sha256':'next'}
    row['content_sha256']=row_digest(row)
    return row


def test_replay_rejects_finite_cost_edits_and_changed_same_key(tmp_path):
    row=row_fixture()
    replay.record(tmp_path,row)
    replay.record(tmp_path,row)
    row['costs']['myopic_ab']=100.
    with pytest.raises(ValueError,match='content hash'):
        validate_row(row)
    row['content_sha256']=row_digest(row)
    with pytest.raises(ValueError,match='immutable replay row'):
        replay.record(tmp_path,row)


def test_fit_snapshot_is_fixed_and_rejects_orphan_episode(tmp_path):
    first=row_fixture()
    replay.record(tmp_path,first)
    watermark=replay.statistics(tmp_path,'science')['nn']['watermark']
    replay.record(tmp_path,row_fixture(1))
    episode={k:first[k] for k in ('city','split','seed','scenario_sha256','graph_sha256','workload_sha256')}
    assert len(replay.snapshot(tmp_path,'nn','science',{first['episode']:episode},watermark))==1
    with pytest.raises(ValueError,match='absent'):
        replay.snapshot(tmp_path,'nn','science',{},watermark)


@pytest.mark.parametrize('entry',[prepare,train])
def test_final_test_barrier_blocks_development_before_backend_import(tmp_path,entry):
    config,_=campaign_fixture(tmp_path)
    atomic(Path(config.output)/'test_opened.json',{'freeze_id':'opened'})
    with pytest.raises(ValueError,match='Final test has been opened'):
        entry(config)


def comparison_fixture():
    episodes=[{'id':'haifa_val_000','city':'haifa','split':'validation','scenario_sha256':'s','graph_sha256':'g','orders':1,'fleet_count':2}]
    results=[]
    for family in ('xgboost','nn',*(FIXED_PREFIX+p for p in POLICIES)):
        results.append({'episode':'haifa_val_000','city':'haifa','family':family,'status':'complete','scenario_sha256':'s',
            'graph_sha256':'g','orders':1,'delivered':1,'robots':2,'loss':2. if family=='nn' else 4.,
            'episode_identity':identity(episodes[0]),'science_id':'science'})
    return episodes,results


def test_comparison_requires_all_ten_matched_baselines():
    episodes,results=comparison_fixture()
    report=compare(results,episodes)
    assert report['completed_results']==12
    assert report['paired']['nn']['hindsight_best_fixed_per_scenario']['total_loss_difference']==-2.
    with pytest.raises(ValueError,match='All selector'):
        compare(results[:-1],episodes)
    results[-1]['scenario_sha256']='different'
    with pytest.raises(ValueError,match='Unmatched comparison input'):
        compare(results,episodes)


def test_weighted_nn_derivatives_without_optimization():
    import numpy as np
    from fleet_switching.models import nn_loss_gradients
    x=np.array([[.2,.7],[.8,.1]])
    y=np.array([[.1,.9],[.3,.6]])
    weights=[np.array([[.2,.3],[.4,.5]]),np.array([[.3,.2],[.6,.4]]),np.array([[.7,.1],[.2,.8]])]
    biases=[np.array([.1,.2]),np.array([.1,.2]),np.array([.1,.2])]
    sample_weights=np.array([.5,1.5])
    _,gradients=nn_loss_gradients(x,y,weights,biases,sample_weights)
    parameters=[item for pair in zip(weights,biases) for item in pair]
    for parameter,gradient in zip(parameters,gradients):
        for index in np.ndindex(parameter.shape):
            before=parameter[index]
            parameter[index]=before+1e-6
            plus,_=nn_loss_gradients(x,y,weights,biases,sample_weights)
            parameter[index]=before-1e-6
            minus,_=nn_loss_gradients(x,y,weights,biases,sample_weights)
            parameter[index]=before
            assert gradient[index]==pytest.approx((plus-minus)/2e-6,rel=1e-5,abs=1e-7)


def test_spatial_features_distinguish_supply_alignment_without_future_orders():
    from fleet_switching.features import enrich
    r=NS(spec=NS(id=1),node_id=1,available=True,remaining_route=[],is_moving=False)
    ns={'robots':[r],'now':20.,'schedules':{1:NS(stops=[],orders={})},'service_lock':{},'station_state':{},
        'graph':NS(nodes={1:{'in_cluster':1},2:{'in_cluster':2}}),
        '_meta_observed':[NS(pickup_node=1,request_time_min=10.),NS(pickup_node=2,request_time_min=999.)]}
    near=enrich({},ns)
    assert near['spatial.observed_demand_free_capacity_imbalance']==0.
    r.node_id=2
    far=enrich({},ns)
    assert far['spatial.observed_demand_free_capacity_imbalance']==2.
    assert far['spatial.recent_demand_without_free_capacity_share']==1.


def test_test_outcomes_cannot_select_deployable_baseline():
    episodes,results=comparison_fixture()
    episodes[0]['split']='test'
    with pytest.raises(ValueError,match='not test outcomes'):
        compare(results,episodes)
