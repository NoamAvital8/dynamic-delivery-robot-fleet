"""Real small child-process regressions; no simulator, scenarios, or fitting."""
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fleet_switching import execution as original

spec=importlib.util.spec_from_file_location('fleet_switching.execution_recovery_tests',Path(__file__).with_name('execution_recovery.py'))
execution=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=execution
spec.loader.exec_module(execution)

def fit_success_then_cleanup(pipe,*args):
    pipe.send(('result',{'saved':True}))
    time.sleep(1.4)
    pipe.close()

def fit_slow_before_result(pipe,*args):
    time.sleep(.25)
    pipe.send(('result',{'saved':True}))
    pipe.close()

def fit_error(pipe,*args):
    pipe.send(('error','deliberate fit exception'))
    pipe.close()

def fit_exit_bad_after_result(pipe,*args):
    pipe.send(('result',{'saved':True}))
    pipe.close()
    raise SystemExit(3)

def fit_no_result(pipe,*args):
    pipe.close()

def episode_success_then_cleanup(pipe,*args):
    pipe.send(('ready',None))
    pipe.send(('result',{'episode':'saved'}))
    time.sleep(1.4)
    pipe.close()

def episode_bad_after_result(pipe,*args):
    pipe.send(('ready',None))
    pipe.send(('result',{'episode':'saved'}))
    pipe.close()
    raise SystemExit(4)

def cfg(tmp_path):
    return SimpleNamespace(output=str(tmp_path),fit_threads=1,fit_timeout_seconds=.001,
        cancellation_grace_seconds=.2,memory_reserve_gib=64.,trainer_memory_gib=8.,worker_memory_gib=1.5,
        reserved_cpus=5,worker_limit=8)

def wait_for(task):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        result=task.poll()
        if result is not None:return result
        time.sleep(.02)
    raise AssertionError('Test child did not finish')

def test_fit_success_retained_during_slow_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'_fit_entry',fit_success_then_cleanup)
    task=execution.FitTask(cfg(tmp_path),'nn','stub',{},0)
    try:
        assert wait_for(task)=={'saved':True}
        assert task.process.exitcode==0
    finally:task.stop()

def test_healthy_fit_has_no_elapsed_time_abort(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'_fit_entry',fit_slow_before_result)
    task=execution.FitTask(cfg(tmp_path),'nn','stub',{},0)
    task.started-=100_000
    try:assert wait_for(task)=={'saved':True}
    finally:task.stop()

@pytest.mark.parametrize('entry,match',[(fit_error,'deliberate fit exception'),
    (fit_exit_bad_after_result,'exited with code 3'),(fit_no_result,'without a result')])
def test_genuine_fit_failures_remain_failures(tmp_path,monkeypatch,entry,match):
    monkeypatch.setattr(execution,'_fit_entry',entry)
    task=execution.FitTask(cfg(tmp_path),'nn','stub',{},0)
    try:
        with pytest.raises(RuntimeError,match=match):wait_for(task)
    finally:task.stop()

def test_episode_success_retained_during_slow_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'workers',lambda *a,**k:1)
    result=execution.run_episodes(cfg(tmp_path),[({'id':'one'},'nn',None)],'stub',
        entry=episode_success_then_cleanup,memory=lambda:1000.)
    assert result==[{'episode':'saved'}]

def test_episode_nonzero_exit_not_hidden_by_success_message(tmp_path,monkeypatch):
    monkeypatch.setattr(execution,'workers',lambda *a,**k:1)
    with pytest.raises(RuntimeError,match='did not exit successfully'):
        execution.run_episodes(cfg(tmp_path),[({'id':'one'},'nn',None)],'stub',
            entry=episode_bad_after_result,memory=lambda:1000.)

def test_memory_reserve_remains_enforced(tmp_path):
    with pytest.raises(RuntimeError,match='Insufficient memory'):
        execution.run_episodes(cfg(tmp_path),[({'id':'one'},'nn',None)],'stub',
            entry=episode_success_then_cleanup,memory=lambda:63.)

def test_original_scientific_execution_source_is_preserved():
    import hashlib
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()=='d0cbd9dd2bba0665cc01f46f592981877abfcc64a37960af49024fe38aec6f90'
