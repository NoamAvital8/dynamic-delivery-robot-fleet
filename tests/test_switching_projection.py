"""Closed-plan arithmetic only: no production simulator or model fitting."""
import copy
import math

import pytest

from fleet_switching.projection import OrderCost, Session, Step, Waiting, project


def order(importance=1, deadline=100):
    return OrderCost(0, deadline, importance)


def test_active_ports_existing_fifo_and_future_arrivals():
    plans = {1:(),2:(),3:(Step("service",1,dropoff=30),),
             4:(Step("travel",1),Step("charge",2,7),Step("service",1,dropoff=40))}
    original = copy.deepcopy(plans)
    result = project(0,plans,{7:2},{30:order(),40:order()},
        active=[Session(1,7,5),Session(2,7,10)], waiting=[Waiting(3,7,4,0)])
    assert result.completed == {40:12,30:10}
    assert result.robot_ready[1]==5 and result.robot_ready[2]==10
    assert result.queue_wait_minutes==13  # 3 waits 5, then 4 waits 8.
    assert plans==original


def test_delay_at_first_charger_changes_fifo_order_at_second_charger():
    plans = {0:(),1:(Step("charge",2,10),Step("travel",1),Step("charge",1,20),Step("service",1,dropoff=1)),
             2:(Step("travel",4),Step("charge",10,20),Step("service",1,dropoff=2))}
    result=project(0,plans,{10:1,20:1},{1:order(),2:order()},active=[Session(0,10,8)])
    assert result.completed[2]==15
    assert result.completed[1]==16
    assert result.queue_wait_minutes==11  # First station 8, second station 3.


def test_partial_handling_and_deadline_penalty_are_counted_once():
    result=project(10,{1:(Step("service",.5,dropoff=5),)}, {}, {5:OrderCost(0,10,5)})
    assert result.completed[5]==10.5
    assert result.loss==68  # 10*5 + .5*36.


def test_late_pending_work_cannot_be_silently_omitted():
    with pytest.raises(ValueError,match="Incomplete"):
        project(0,{1:()}, {}, {99:order()})
    with pytest.raises(ValueError,match="Duplicate"):
        project(0,{1:(Step("service",1,dropoff=1),Step("service",1,dropoff=1))},{},{1:order()})


def test_invalid_energy_time_and_ports_fail_closed():
    with pytest.raises(ValueError):
        Step("charge",-1,7)
    with pytest.raises(ValueError):
        Session(1,7,float('nan'))
    with pytest.raises(ValueError):
        OrderCost(0,float('nan'),1)
    with pytest.raises(ValueError):
        project(0,{1:()}, {7:1},{},active=[Session(1,7,1),Session(1,7,2)])
