"""Projection adapter fixtures, without constructing or advancing a simulator."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

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
