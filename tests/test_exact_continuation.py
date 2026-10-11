from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'operations'))
import continue_exact_optimized_campaign as controller


def test_worker_limits_include_nested_preprocessing_and_idle():
    assert controller.worker_cost({'policy':'reactive_insertion'}) == 1
    assert controller.worker_cost({'policy':'full'}) == 5
    assert controller.worker_cost({'policy':'myopic_ab'},'preprocessing') == 5
    assert controller.can_admit(live_count=23,workers_used=55,cost=5,
        available=100*controller.GIB,reserved=0,needed=8*controller.GIB)
    assert not controller.can_admit(live_count=23,workers_used=56,cost=5,
        available=100*controller.GIB,reserved=0,needed=8*controller.GIB)
    assert not controller.can_admit(live_count=24,workers_used=1,cost=1,
        available=100*controller.GIB,reserved=0,needed=controller.GIB)


def test_memory_guard_reserves_inflight_preprocessing():
    args=dict(live_count=12,workers_used=20,cost=5,available=56*controller.GIB,
              reserved=33*controller.GIB,needed=8*controller.GIB)
    assert not controller.can_admit(**args)
    args['needed']=controller.GIB
    assert controller.can_admit(**args)


def test_pending_jobs_never_include_completed_or_live(tmp_path):
    def job(name):
        folder=tmp_path/name;folder.mkdir()
        return {'output':folder/'result.json','stamp_path':folder/'stamp.json',
                'status_path':folder/'status.json'}
    complete, live, pending = [job(n) for n in ['complete','live','pending']]
    complete['output'].write_bytes(b'kept')
    complete['stamp_path'].write_bytes(b'kept-stamp')
    live['status_path'].write_text('{"status":"running"}')
    assert controller.pending_jobs([complete,live,pending],{str(live['output']):{}})==[pending]
    assert complete['output'].read_bytes()==b'kept'
    with pytest.raises(RuntimeError,match='unclaimed'):
        controller.pending_jobs([live],{})
    pending['output'].write_bytes(b'unstamped')
    with pytest.raises(RuntimeError,match='unstamped'):
        controller.pending_jobs([pending],{})
