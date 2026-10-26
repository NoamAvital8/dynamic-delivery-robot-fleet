"""Explicit final-test supervisor: fixed models, four cities, no fitting path."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fleet_switching.config import Config, HELD_OUT, atomic, available_gib, immutable, locked, read, sha

OVERRIDE_SHA = '20cd1498fce964794fd293747c0773019be1f776b14d217ec7386e9ce5e2b956'


def utc():return datetime.now(timezone.utc).isoformat()


def config():
    base=Config.load(ROOT/'configs/small_city_switching_v2.json')
    if Path(base.output).resolve()!=ROOT or base.test_per_city!=50:
        raise ValueError('Unexpected test root or scenario count')
    return replace(base,worker_limit=min(67,len(os.sched_getaffinity(0))-5),reserved_cpus=5)


def verify(expected_freeze):
    if sha(ROOT/'frozen_models.json')!=expected_freeze:raise ValueError('Frozen receipt changed')
    from fleet_switching.provenance import freeze_receipt
    frozen=freeze_receipt(config())
    if set(frozen['held_out_cities'])!=HELD_OUT:raise ValueError('Test-city roster changed')
    for name,digest in read(ROOT/'review_bundle/CURRENT_BUILD_SOURCES.json')['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Sealed source changed: '+name)
    return frozen


def protected_files(frozen):
    paths=[ROOT/n for n in ['frozen_models.json','datasets/development_manifest.json',
        'validation_summary.json','training_summary.json','models/xgboost/current.json','models/nn/current.json']]
    paths += [Path(v['path']) for v in (*frozen['models'].values(),*frozen['baselines'].values())]
    paths += [p for p in [ROOT/'replay.sqlite',ROOT/'replay.sqlite-wal'] if p.exists()]
    return {str(p):sha(p) for p in paths}


def check_protected():
    for name,digest in read(ROOT/'final_test/PROTECTED_DEVELOPMENT_FILES.json')['files'].items():
        if sha(name)!=digest:raise ValueError('Development artifact changed during final test: '+name)


def worker(expected_freeze):
    frozen=verify(expected_freeze)
    override=ROOT/'recovery/20261026/execution_recovery.py'
    if sha(override)!=OVERRIDE_SHA:raise ValueError('Execution override changed')
    name='fleet_switching.execution'
    spec=importlib.util.spec_from_file_location(name,override)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.numeric_environment(1)
    from fleet_switching.data import prepare, load_dataset
    from fleet_switching.workflow import evaluate
    folder=ROOT/'final_test'
    atomic(folder/'worker_stage.json',{'stage':'preparing_test_data','updated_utc':utc(),'pid':os.getpid()})
    prepare(config(),test=True)
    data=load_dataset(config(),test=True)
    counts=Counter(e['city'] for e in data['episodes'])
    if counts!={c:50 for c in HELD_OUT} or any(e['split']!='test' for e in data['episodes']):
        raise ValueError('Unexpected final-test split')
    immutable(folder/'TEST_DATA_VERIFIED.json',{'freeze_id':frozen['freeze_id'],
        'test_manifest_sha256':sha(ROOT/'datasets/test_manifest.json'),
        'inputs':200,'city_counts':dict(counts),'seeds':len({e['seed'] for e in data['episodes']}),
        'workloads':len({e['workload_sha256'] for e in data['episodes']}),
        'development_seed_and_workload_overlap':False})
    atomic(folder/'worker_stage.json',{'stage':'final_test_evaluation','updated_utc':utc(),'pid':os.getpid()})
    result=evaluate(config())
    check_protected()
    atomic(folder/'WORKER_COMPLETE.json',{'completed_utc':utc(),'freeze_id':frozen['freeze_id'],'result':result,
        'development_artifacts_unchanged':True,'models_retrained':False})


def snapshot(child):
    folder=ROOT/'final_test'
    stage=read(folder/'worker_stage.json').get('stage','starting') if (folder/'worker_stage.json').exists() else 'starting'
    completed=len(list(ROOT.glob('actors/*/test/*/complete.json')))
    value={'updated_utc':utc(),'stage':stage,'supervisor_pid':os.getpid(),
        'child_pid':child.pid,'completed_outcomes':completed,'expected_outcomes':2400,
        'worker_limit':config().worker_limit,'available_gib':available_gib(),
        'load_average':os.getloadavg(),'test_cities':sorted(HELD_OUT),'models_retrained':False}
    atomic(ROOT/'test_run_status.json',value)
    return value


def stop_child(child):
    if child is None or child.poll() is not None:return
    child.send_signal(signal.SIGINT)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def supervisor(expected_freeze):
    folder=ROOT/'final_test';folder.mkdir(exist_ok=True)
    with locked(ROOT/'.final_test_supervisor.lock'):
        frozen=verify(expected_freeze)
        with locked(ROOT/'.campaign.lock'):pass
        protected=folder/'PROTECTED_DEVELOPMENT_FILES.json'
        if not protected.exists():
            immutable(protected,{'freeze_id':frozen['freeze_id'],'files':protected_files(frozen)})
        else:check_protected()
        if (folder/'COMPLETE.json').exists():
            print('Final test already complete; no new evaluations started',flush=True);return
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--execute','--worker',
            '--expected-freeze-sha256',expected_freeze]
        child=None
        try:
            with (folder/'evaluation.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic(folder/'CURRENT_PROCESS.json',{'started_utc':utc(),'pid':child.pid,'command':command})
                while child.poll() is None:
                    snapshot(child);time.sleep(10)
                if child.returncode:raise RuntimeError('Final-test worker exited '+str(child.returncode)+'; see final_test/evaluation.log')
            check_protected();verify(expected_freeze)
            summary=read(ROOT/'test_summary.json')
            if (summary['completed_results'],summary['missing_results'],summary['failed_results'])!=(2400,0,0):
                raise ValueError('Incomplete final-test comparison')
            value=snapshot(child);value.update(stage='complete',completed_utc=utc(),
                summary=str(ROOT/'test_summary.json'),development_artifacts_unchanged=True)
            atomic(folder/'COMPLETE.json',value);atomic(ROOT/'test_run_status.json',value)
        except BaseException:
            stop_child(child)
            atomic(ROOT/'test_run_status.json',{'updated_utc':utc(),'stage':'failed',
                'supervisor_pid':os.getpid(),'error':traceback.format_exc(),'models_retrained':False})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--expected-freeze-sha256',required=True);args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required to open final test')
    def interrupted(*_):raise KeyboardInterrupt('Explicit final-test stop')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    (worker if args.worker else supervisor)(args.expected_freeze_sha256)


if __name__=='__main__':main()
