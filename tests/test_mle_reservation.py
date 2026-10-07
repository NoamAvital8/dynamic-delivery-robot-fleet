from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

from delivery_fleet.anticipatory_policy import ReservationRobotSnapshot
from delivery_fleet.fleet import RobotType
from delivery_fleet.mle_policy import OnlineMLEReservation
from delivery_fleet.mle_reservation import (CandidateScorer, EpochPoissonRates, MLEConfig,
    RateBounds, SurrogateTrace, importance_stress_rates, poisson_confidence_sequence,
    select_gated_candidate, surrogate_loss)
from delivery_fleet.robot import RobotSpec, RobotState
from delivery_fleet.scenario_creator import Item, Order, Scenario
from delivery_fleet.charging import annotate_nearest_charging_stations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_nyc_mle_reservation_policy as runner


@pytest.mark.parametrize('n,t', [(0,10.), (1,10.), (100,120.), (1000,500.)])
def test_anytime_bounds_contain_mle_and_cross_mixture_boundary(n,t):
    error = .001
    b = poisson_confidence_sequence(n,t,error)
    assert b.lower <= b.mle <= b.upper and b.upper > 0
    const = .5*math.log(10)+math.lgamma(.5+n)-math.lgamma(.5)-(.5+n)*math.log(10+t)
    for endpoint in [b.lower,b.upper]:
        if endpoint > 0:
            assert const+endpoint*t-n*math.log(endpoint) == pytest.approx(-math.log(error), abs=1e-8)


def test_no_arrivals_does_not_mean_zero_future_demand():
    assert poisson_confidence_sequence(0,120.,.05).upper > 0
    assert poisson_confidence_sequence(0,0.,.05).upper == math.inf
    narrow = poisson_confidence_sequence(100,100.,.05)
    wide = poisson_confidence_sequence(100,100.,.001)
    assert wide.lower < narrow.lower < narrow.upper < wide.upper


@pytest.mark.parametrize('count,t,error', [(-1,10.,.05),(1.5,10.,.05),(1,0.,.05),
    (0,-1.,.05),(0,math.inf,.05),(0,1.,1.),(0,1.,0.)])
def test_invalid_rate_inputs(count,t,error):
    with pytest.raises(ValueError):poisson_confidence_sequence(count,t,error)


def test_epoch_reset_exposure_and_joint_error_budget():
    rates = EpochPoissonRates(['a','b'], confidence=.95, epoch_min=120.)
    rates.observe('a',5.);rates.observe('a',50.)
    assert rates.bounds(60.)['a'].mle == 2/60
    assert rates.bounds(120.)['a'].count == 0
    rates.observe('b',120.)
    assert rates.bounds(120.)['b'].upper == math.inf
    b = rates.bounds(121.)['b']
    error = .05*6/math.pi**2/4/2
    assert b.upper == pytest.approx(poisson_confidence_sequence(1,1.,error).upper)
    with pytest.raises(ValueError):rates.observe('a',119.)


@pytest.mark.parametrize('kwargs', [{'confidence':1.},{'epoch_min':0.},{'rollouts':0},
    {'processes':0},{'minimum_gain_fraction':-1.},{'stability_tolerance':math.nan},
    {'seed':-1},{'maximum_relative_rate_width':0.}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):MLEConfig(**kwargs)


def test_gate_requires_stable_alpha_and_gain_in_every_stress_case():
    fractions = np.array([[[1.,0.,0.]], [[.8,.1,.1]], [[.5,.2,.3]]])
    config = MLEConfig(stability_tolerance=.05,minimum_gain_fraction=.01)
    stable = select_gated_candidate(np.array([[100.,120.],[80.,90.],[95.,110.]]),fractions,config)
    assert stable.enabled and stable.candidate_index==1
    assert select_gated_candidate(np.array([[100.,100.],[80.,90.],[95.,70.]]),fractions,config).reason=='unstable_alpha'
    assert select_gated_candidate(np.array([[100.,100.],[80.,110.],[95.,120.]]),fractions,replace(config,stability_tolerance=1.)).reason=='insufficient_stress_gain'
    assert not select_gated_candidate(np.array([[100.,100.],[110.,105.],[120.,130.]]),fractions,config).enabled
    assert not select_gated_candidate(np.array([[100.,100.],[math.inf,1.],[1.,2.]]),fractions,config).enabled


def test_rate_stress_grid_includes_all_class_corners():
    rates = [RateBounds(1,10.,.1,.02,.3),RateBounds(1,10.,.1,.01,.4)]
    grid = importance_stress_rates(rates)
    assert len(grid)==5
    assert tuple(grid[0])==(.1,.1)
    assert {tuple(x) for x in grid[1:]}=={(.02,.01),(.02,.4),(.3,.01),(.3,.4)}


def test_reservation_can_protect_fast_robot_from_low_priority_future_work():
    trace = SurrogateTrace(np.array([0.,1.]),np.array([1.,5.]),np.array([100.,1.]),
                           np.array([[10.,30.],[1.,30.]]),np.zeros(2))
    assert surrogate_loss(trace,np.array([5.,1.])) < surrogate_loss(trace,np.array([1.,1.]))


def test_parallel_and_serial_candidate_scores_are_identical():
    trace = SurrogateTrace(np.array([0.,1.]),np.array([1.,5.]),np.array([100.,1.]),
                           np.array([[10.,30.],[1.,30.]]),np.zeros(2))
    choices=[np.array([1.,1.]),np.array([5.,1.])]
    a,b=CandidateScorer(1),CandidateScorer(2)
    try:
        assert np.array_equal(a.score(choices,[[trace],[trace]]),b.score(choices,[[trace],[trace]]))
    finally:a.close();b.close()


class SmallRobot(RobotSpec):
    robot_type=RobotType.MIDDLE_MAN


def fixture_policy(tmp_path,config=MLEConfig()):
    graph=nx.path_graph(21)
    for node in graph:
        graph.nodes[node].update(x=34.+node*.001,y=32.,in_cluster=0 if node<10 else 1,
            is_cluster_representative=node in {0,20},is_charging_station=node in {0,5,10,15,20})
    nx.set_edge_attributes(graph,100.,'length')
    robots=[RobotState.fully_charged(SmallRobot(i,4.,10.,10.,100.,.01),0) for i in range(8)]
    policy=OnlineMLEReservation(graph,robots,{1.:30.,2.:10.,5.:5.},charger_power_w=2000.,
                               horizon_min=60.,config=config,audit_path=tmp_path/'decisions.jsonl')
    snapshots={r.spec.id:ReservationRobotSnapshot(0,0.,100.,False) for r in robots}
    return policy,snapshots


def test_cold_start_arrival_only_learning_and_horizon_drain(tmp_path):
    policy, snapshots=fixture_policy(tmp_path)
    assert policy.update(0.,snapshots) is None
    order=Order(0,0,20,1.,Item(1.,1.),5.)
    with pytest.raises(ValueError,match='unreleased'):policy.observe_order(order,0.)
    policy.observe(0,5.,1.);policy.observe_order(order,1.)
    assert policy.rates.bounds(2.)[('total',5.)].count==1
    assert policy.update(20.,snapshots) is None
    assert policy.reason=='insufficient_arrivals'
    assert policy.update(60.,{}) is None and policy.reason=='arrival_horizon_ended'
    policy.close()


def test_real_optimizer_trace_budget_and_audit(tmp_path):
    policy,snapshots=fixture_policy(tmp_path,MLEConfig(minimum_arrivals=1,rollouts=1,
        maximum_relative_rate_width=100.,max_requests_per_trace=100))
    for i in range(6):
        order=Order(i,0,20,float(i+1),Item(1.,1.),1. if i<4 else 5.)
        policy.observe(order.pickup_node,order.importance,order.request_time_min)
        policy.observe_order(order,order.request_time_min)
    policy.update(30.,snapshots)
    assert policy.stats['candidates_scored']>0
    records=[json.loads(line) for line in policy.audit_path.read_text().splitlines()]
    assert records[-1]['stress_cases']==11 and records[-1]['class_rates']['5.0']['count']==2
    assert policy.update(31.,snapshots)==policy.assignment # Throttled
    policy.config=replace(policy.config,max_requests_per_trace=1)
    assert policy.update(45.,snapshots) is None and policy.reason=='stress_budget_exceeded'
    policy.finalize(60.)


def test_wide_rate_bounds_block_reservation_before_search(tmp_path):
    policy,snapshots=fixture_policy(tmp_path,MLEConfig(minimum_arrivals=1))
    policy.observe(0,1.,1.)
    policy.observe_order(Order(0,0,20,1.,Item(1.,1.),1.),1.)
    assert policy.update(15.,snapshots) is None
    assert policy.reason=='rate_intervals_too_wide'
    assert policy.stats['candidates_scored']==0
    assert policy.stats['planning_seconds']>0
    policy.close()


def test_activation_pending_fallback_epoch_reset_and_drain(tmp_path,monkeypatch):
    policy,snapshots=fixture_policy(tmp_path,MLEConfig(minimum_arrivals=1,rollouts=1,
        maximum_relative_rate_width=100.,epoch_min=20.,horizon_min=5.))
    def stable_scores(thresholds,scenarios):
        scores=np.full((len(thresholds),len(scenarios)),80.)
        scores[0]=100.
        return scores
    monkeypatch.setattr(policy.scorer,'score',stable_scores)
    policy.observe(0,5.,1.);policy.observe_order(Order(0,0,20,1.,Item(1.,1.),5.),1.)
    assert policy.update(10.,snapshots) is not None
    assert policy.update(11.,{},pending_count=1) is None
    assert policy.active_minutes==1.
    policy.observe(0,1.,20.)
    assert not policy.templates and policy.assignment is None
    policy.observe_order(Order(1,0,20,20.,Item(1.,1.),1.),20.)
    assert policy.update(20.,snapshots) is None # Zero epoch exposure, even with arrivals.
    assert policy.reason=='insufficient_arrivals'
    policy.finalize(60.)
    diagnostics=policy.diagnostics()
    assert diagnostics['mle_reservation_activations']==diagnostics['mle_reservation_deactivations']==1
    assert diagnostics['mle_reservation_first_activation_min']==10.
    assert diagnostics['mle_reservation_last_decision_reason']=='insufficient_arrivals'
    assert diagnostics['mle_reservation_reason_counts']['pending_real_orders']==1


@pytest.mark.parametrize('coordinated_idle',[False,True])
@pytest.mark.parametrize('force_gate',[False,True])
def test_full_runner_loads_and_executes_without_nn(tmp_path,monkeypatch,coordinated_idle,force_gate):
    policy,_=fixture_policy(tmp_path)
    graph=policy.graph;annotate_nearest_charging_stations(graph,[0,5,10,15,20])
    gp,sp=tmp_path/'g.graphml',tmp_path/'s.json';nx.write_graphml(graph,gp)
    orders=tuple(Order(i,0,20,float(i+1),Item(1.,1.),2.) for i in range(3))
    if force_gate:
        orders+=(Order(3,0,20,40.,Item(1.,1.),2.),)
    Scenario('mle',60.,42,orders).save_json(sp)
    namespace=runner._load_namespace(coordinated_idle)
    monkeypatch.setitem(namespace,'create_default_fleet',lambda *a,**kw:list(policy.robots))
    out=tmp_path/'result.json'
    command=['mle','--graph',str(gp),'--scenario',str(sp),'--output',str(out),
        '--idle-processes','1','--mle-idle-mode','coordinated' if coordinated_idle else 'legacy']
    if force_gate:
        def stable_scores(self,thresholds,scenarios):
            scores=np.full((len(thresholds),len(scenarios)),80.)
            scores[0]=100.
            return scores
        monkeypatch.setattr(CandidateScorer,'score',stable_scores)
        command+=['--mle-minimum-arrivals','1','--mle-maximum-relative-rate-width','100',
                  '--mle-update-interval-min','15','--mle-epoch-min','20',
                  '--mle-rollouts','1','--mle-horizon-min','5']
    monkeypatch.setattr(sys,'argv',command)
    namespace['main']()
    result=json.loads(out.read_text())
    assert result['delivered']==len(orders) and result['reservation_model'] is None
    assert result['policy_version']=='uncertainty_gated_mle_v1'
    assert bool(result['mle_reservation_activations'])==force_gate
    assert result['reservation_enabled']==force_gate
    assert result['reservation_style']=='epoch_poisson_mle_anytime_gated_surrogate'
    assert result['mle_reservation_decisions_file']==str(out.with_suffix('.reservation.jsonl'))
    assert result['idle_readiness_missing_plan']==result['queue_forecast_missing_cached_plans']==0
    if force_gate:
        audit=[json.loads(line) for line in out.with_suffix('.reservation.jsonl').read_text().splitlines()]
        assert any(r['time_min']==20. and r['epoch']==1 and not r['enabled'] for r in audit)
    policy.close()


def test_multicity_commands_and_summary_preserve_mle_controls(tmp_path):
    import run_multicity_experiments as multicity
    config=MLEConfig(processes=2,confidence=.99,epoch_min=120.,seed=17)
    for name,mode in [('queue_mle_reservation','legacy'),('full_mle_reservation','coordinated')]:
        command=multicity.policy_command(name,python=sys.executable,graph=tmp_path/'g',
            scenario=tmp_path/'s',output=tmp_path/'o',spatial_model=tmp_path/'unused_nn',
            paper_model=tmp_path/'unused_paper',shortlist_k=10,idle_processes=4,
            prior_rates={'1':10.,'2':5.,'5':2.},prior_concentration=4.,mle_config=config)
        assert command[command.index('--mle-idle-mode')+1]==mode
        assert command[command.index('--mle-processes')+1]=='2'
        assert command[command.index('--mle-confidence')+1]=='0.99'
        assert command[command.index('--mle-seed')+1]=='17'
        assert '--reservation-model' not in command
    assert len(multicity.DEFAULT_POLICIES)==6 and not set(multicity.DEFAULT_POLICIES)&set(multicity.MLE_POLICIES)
    assert multicity.mle_worker_budget(12,4,1)==60
    assert multicity.mle_worker_budget(10,1,2)==30
    result={'delivered':2,'on_time':1,'orders':2,'loss_objective':100.,'wall_clock_seconds':3.,
            'simulation_finish_min':10.,'mle_reservation_updates':7,'mle_reservation_activations':1}
    job={'city':'test','scenario_id':'001','seed':1,'policy':'full_mle_reservation',
         'output':tmp_path/'o','log':tmp_path/'l'}
    row=multicity.summarize(result,job)
    assert row['mle_reservation_updates']==7 and row['mle_reservation_activations']==1


def test_multicity_rejects_nested_worker_budget_over_sixty(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/run_multicity_experiments.py'),
        str(tmp_path/'not_read.json'),str(tmp_path/'not_created'),
        '--spatial-model','unused','--paper-model','unused','--policies','full_mle_reservation',
        '--processes','13','--idle-processes','4','--mle-processes','1'],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==2 and 'cap is 60' in result.stderr
    assert not (tmp_path/'not_created').exists()
