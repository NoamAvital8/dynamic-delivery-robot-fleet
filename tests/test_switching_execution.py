"""Real child-process control with tiny stubs; never imports a simulator or fits."""
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from fleet_switching import execution, workflow
from fleet_switching.config import Config, read


def stub_episode(pipe, stop, config, episode, family, science, frozen):
    try:
        pipe.send(('ready',None))
        if episode['id']=='fail':
            time.sleep(.1)
            pipe.send(('error','injected worker failure'))
        elif episode['id']=='unresponsive':
            while True:
                time.sleep(.05)
        else:
            pipe.send(('result',{'episode':episode['id'],'pid':os.getpid()}))
    finally:
        pipe.close()


def config(tmp_path,**kw):
    return Config(backend_root='unused',map_manifest='unused',output=str(tmp_path),
                  spatial_model='unused',paper_model='unused',cancellation_grace_seconds=.2,**kw)


def test_episode_process_is_not_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'workers',lambda *args,**kw:1)
    jobs=[({'id':str(i)},'nn',None) for i in range(3)]
    results=execution.run_episodes(config(tmp_path),jobs,'stub',entry=stub_episode,memory=lambda:1000.)
    assert len({r['pid'] for r in results})==3
    assert [r['episode'] for r in results]==['0','1','2']


def test_failure_stops_even_unresponsive_owned_peer(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'workers',lambda *args,**kw:2)
    before={p.pid for p in multiprocessing.active_children()}
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='injected worker failure'):
        execution.run_episodes(config(tmp_path),[({'id':'fail'},'nn',None),({'id':'unresponsive'},'nn',None)],
                               'stub',entry=stub_episode,memory=lambda:1000.)
    assert time.monotonic()-started<10
    assert {p.pid for p in multiprocessing.active_children()} <= before
    assert read(tmp_path/'last_shutdown.json')['owned_processes_stopped'] is True


def test_memory_reserve_prevents_any_child_start(tmp_path):
    with pytest.raises(RuntimeError,match='Insufficient memory'):
        execution.run_episodes(config(tmp_path),[({'id':'one'},'nn',None)],'stub',entry=stub_episode,memory=lambda:63.)


def test_publisher_limits_loaded_libraries_with_fit_stub(tmp_path,monkeypatch):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key,os.environ.get(key,'1'))
    def no_optimization(rows,family,threads,policies):
        from threadpoolctl import threadpool_info
        assert all(item['num_threads']<=threads for item in threadpool_info())
        return {'kind':'test_only_fixed_payload','training_records_sha256':'synthetic'}
    monkeypatch.setattr(workflow,'fit',no_optimization)
    pointer=workflow._publish(config(tmp_path,fit_threads=2),'nn',[{}],'synthetic-science')
    assert Path(pointer['path']).is_file()
    assert pointer['training_states']==1
    assert read(Path(pointer['path']).parent/'last_fit_runtime.json')['native_libraries_inside_limit']
