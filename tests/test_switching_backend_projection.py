"""Projection adapter fixtures, without constructing or advancing a simulator."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from fleet_switching.backend import completion_projection


@dataclass(frozen=True)
class Stop:
    order_id: int
    kind: str
    node_id: int


def robot(rid,node,battery=100,moving=False):
    spec=NS(id=rid,speed_mps=1.,energy_per_meter_wh=.01,battery_capacity_wh=100.,can_hold=lambda item:True)
    return NS(spec=spec,node_id=node,battery_wh=battery,is_moving=moving,remaining_route=[])


def order(oid,pickup,dropoff):
    return NS(id=oid,pickup_node=pickup,dropoff_node=dropoff,request_time_min=0.,importance=1.,item=object())


def namespace(robots):
    def plan(proxy,target,**kwargs):
        distance=abs(proxy.node_id-target)*60.
        phases=[]
        if proxy.battery_wh < distance*.01:
            phases.append(NS(kind='charge',target_node=proxy.node_id,energy_wh=5.))
        phases.append(NS(kind='travel',target_node=target,distance_m=distance))
        return NS(phases=phases)
    return {'now':0.,'DEFAULT_CHARGING_POWER_W':60.,'DEFAULT_NUMBER_OF_PORTS':1,
        'station_state':{},'robots':robots,'robot_by_id':{r.spec.id:r for r in robots},
        'schedules':{r.spec.id:NS(orders={},stops=[],prepared_steps=[],version=0) for r in robots},
        'service_lock':{},'idle_service_finish_min':{},'phase_version':{r.spec.id:-1 for r in robots},
        'active_phases':{r.spec.id:[] for r in robots},'idle_intents':{},'pending':[],
        'PICKUP_HANDLING_MIN':1.,'DROPOFF_HANDLING_MIN':1.,
        'RobotState':lambda **kwargs:NS(**kwargs),'plan_final_target':plan,
        'distance_row':lambda oracle,source:source,'row_distance':lambda source,target:abs(source-target)*60.,
        'oracle':None,'get_direct_distance':lambda o:abs(o.pickup_node-o.dropoff_node)*60.,
        'delivery_deadline_min':lambda *args:100.,'NoFeasibleBatteryRoute':RuntimeError}


def test_pending_order_has_real_completion_cost_without_mutating_assignments():
    ns=namespace([robot(1,0),robot(2,5)])
    ns['pending']=[order(9,1,2)]
    result,info=completion_projection(ns)
    assert result.completed[9]==4.
    assert result.loss==4.
    assert info['pending_orders']==1
    assert len(ns['pending'])==1
    assert all(not s.orders for s in ns['schedules'].values())
    assert [r.battery_wh for r in ns['robots']]==[100,100]


def test_pending_robot_can_delay_existing_mission_at_a_shared_station():
    ns=namespace([robot(1,2,90),robot(2,7,.1)])
    stop=Stop(1,'dropoff',7)
    phases=[NS(kind='travel',target_node=7,distance_m=300.),NS(kind='charge',target_node=7,energy_wh=5.)]
    ns['schedules'][1]=NS(orders={1:NS(order=order(1,2,7),deadline_min=100.)},
        stops=[stop],prepared_steps=[NS(stop=stop,phases=phases)],version=0)
    ns['station_state']={7:NS(active={},queue=[])}
    ns['pending']=[order(2,7,8)]
    result,info=completion_projection(ns)
    assert result.completed=={2:8.,1:12.}
    assert result.loss==20.
    assert result.queue_wait_minutes==1.


def test_partial_edge_and_remaining_phase_include_known_fifo_delay():
    r1,r2=robot(1,0,80,True),robot(2,3,50)
    r1.remaining_route=[3]
    ns=namespace([r1,r2])
    stop=Stop(1,'dropoff',3)
    ns['schedules'][1]=NS(orders={1:NS(order=order(1,0,3),deadline_min=100.)},
        stops=[stop],prepared_steps=[],version=0)
    ns['phase_version'][1]=0
    ns['active_phases'][1]=[NS(kind='charge',target_node=3,energy_wh=3.)]
    ns['decision_snapshot']=lambda r,now:NS(node_id=1,time_min=2.,battery_wh=80.)
    ns['path_distance']=lambda path:180.
    ns['station_state']={3:NS(active={2:NS(start_time_min=0.,energy_wh=7.,start_battery_wh=50.)},queue=[])}
    result,_=completion_projection(ns)
    assert result.completed[1]==11.
    assert result.queue_wait_minutes==2.


def test_service_lock_uses_only_remaining_handling_time():
    r=robot(1,3)
    ns=namespace([r]); ns['now']=10.
    stop=Stop(1,'dropoff',3)
    ns['schedules'][1]=NS(orders={1:NS(order=order(1,0,3),deadline_min=100.)},
        stops=[stop],prepared_steps=[],version=0)
    ns['service_lock']={1:(9,stop)}
    ns['idle_service_finish_min']={1:10.25}
    result,_=completion_projection(ns)
    assert result.completed[1]==10.25


def test_independent_released_order_conservation_detects_lost_work():
    ns=namespace([robot(1,0)])
    ns.update(_meta_observed=[order(9,1,2)],delivered=0)
    with pytest.raises(ValueError,match='conservation'):
        completion_projection(ns)


def test_stale_prepared_leg_is_rebuilt_from_current_node_and_battery():
    ns=namespace([robot(1,0)])
    stop=Stop(9,'dropoff',2)
    ns['schedules'][1]=NS(orders={9:NS(order=order(9,0,2),deadline_min=100.)},stops=[stop],version=0,
        prepared_steps=[NS(stop=stop,start_node=999,start_battery_wh=0.,phases=[])])
    result,info=completion_projection(ns)
    assert result.completed[9]==3.
    assert info['reconstructed_route_legs']==1


def test_adapter_preserves_saved_incoming_event_order():
    r1,r2=robot(1,0,90,True),robot(2,0,90,True)
    r1.next_node=r2.next_node=7
    r1.next_node_arrival_time_min=r2.next_node_arrival_time_min=1.
    ns=namespace([r1,r2])
    ns['station_state']={7:NS(active={},queue=[])}
    ns['events']=[(1.,1,20,'delivery_node_arrival',NS(robot_id=1,node_id=7,time_min=1.)),
                  (1.,1,10,'delivery_node_arrival',NS(robot_id=2,node_id=7,time_min=1.))]
    ns['event_counter']=NS(next_value=21)
    ns['decision_snapshot']=lambda r,now:NS(node_id=7,time_min=1.,battery_wh=90.)
    for rid,energy in ((1,10.),(2,1.)):
        stop=Stop(rid,'dropoff',7)
        ns['schedules'][rid]=NS(orders={rid:NS(order=order(rid,0,7),deadline_min=100.)},
            stops=[stop],prepared_steps=[],version=0)
        ns['phase_version'][rid]=0
        ns['active_phases'][rid]=[NS(kind='charge',target_node=7,energy_wh=energy)]
    result,_=completion_projection(ns)
    assert result.completed=={2:3.,1:13.}


def test_undirected_pending_shortlist_uses_one_pickup_row(tmp_path):
    ns=namespace([robot(rid,rid) for rid in range(80)])
    ns['graph']=NS(is_directed=lambda:False)
    ns['pending']=[order(9,2,3)]
    sources=[]
    ns['distance_row']=lambda oracle,source:(sources.append(source) or source)
    result,_=completion_projection(ns)
    assert result.completed[9]==3.
    assert len(sources)<20  # Full eligible-fleet ranking no longer needs 80 rows.


def test_finished_delivery_branch_ignores_background_events_before_boundary():
    r=robot(1,0,80,True)
    ns=namespace([r]);ns.update(now=735.,_meta_finished=True,delivered=1,scenario=NS(orders=[object()]))
    # This is a real engine condition: final dropoff stops the engine before
    # advance() sets the clock to the requested decision boundary.
    ns['events']=[(730.,1,10,'background_node_arrival',NS(robot_id=1,node_id=2,time_min=730.))]
    result,info=completion_projection(ns)
    assert result.loss==0. and result.completed=={}
    assert info['finished_delivery_branch'] is True
    ns['pending']=[order(2,1,2)]
    with pytest.raises(ValueError,match='unfinished delivery work'):
        completion_projection(ns)
