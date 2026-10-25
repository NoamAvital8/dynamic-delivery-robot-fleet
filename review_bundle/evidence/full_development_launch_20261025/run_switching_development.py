"""Supervise an explicitly authorized full development run; never opens final test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fleet_switching.config import Config, DEVELOPMENT, HELD_OUT, atomic, available_gib, locked, read, sha, workers


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_holdout(root):
    for path in (root/'test_opened.json', root/'datasets/test_manifest.json'):
        if path.exists():
            raise ValueError('Final test must remain unopened: '+str(path))
    if any((root/'actors').glob('*/test/*')):
        raise ValueError('Final-test actor output exists')


def snapshot(root, config, stage, child=None):
    check_holdout(root)
    status = read(root/'training_status.json') if (root/'training_status.json').exists() else {}
    train_complete = len(list((root/'actors').glob('*/train/*/complete.json')))
    val_complete = len(list((root/'actors').glob('*/validation/*/complete.json')))
    phase = ('development_validation' if stage=='training_and_validation' and val_complete
             else stage)
    value = {'updated_utc':utc(), 'stage':phase, 'supervisor_pid':os.getpid(),
        'child_pid':child.pid if child else None, 'worker_limit':config.worker_limit,
        'fit_threads':config.fit_threads, 'available_gib':available_gib(),
        'load_average':os.getloadavg(), 'training_episodes_complete':train_complete,
        'training_episodes_expected':1600, 'validation_outcomes_complete':val_complete,
        'validation_outcomes_expected':2400, 'controller':status,
        'held_out_cities':sorted(HELD_OUT), 'final_test_opened':False}
    if stage=='preparing_development_data':
        value['scenario_files_written']=len(list((root/'datasets').glob('*/*.json')))
    atomic(root/'full_run_status.json',value)
    return value


def stop_child(child, grace):
    if child is None or child.poll() is not None:return
    child.send_signal(signal.SIGINT)
    try:child.wait(timeout=grace+10.)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()


def run_phase(root, config, name, command):
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    args=[sys.executable,'-B','-u',str(root/'operations/switching_meta.py'),command,
          '--config',str(root/'configs/small_city_switching_v2.json'),'--execute']
    child=None
    try:
        with (root/'logs'/(command+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
            atomic(root/(command+'_process.json'),{'started_utc':utc(),'pid':child.pid,'command':args})
            print(utc(),name,'started',child.pid,flush=True)
            while child.poll() is None:
                snapshot(root,config,name,child)
                time.sleep(10.)
            if child.returncode:
                raise RuntimeError(command+' exited with code '+str(child.returncode)+'; see logs/'+command+'.log')
    except BaseException:
        stop_child(child,config.cancellation_grace_seconds)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--expected-source-sha256',required=True)
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    root=ROOT
    config=Config.load(root/'configs/small_city_switching_v2.json')
    if Path(config.output).resolve()!=root:raise ValueError('Unexpected campaign output')
    if (config.train_per_city,config.validation_per_city,config.worker_limit,config.fit_threads)!=(100,25,64,4):
        raise ValueError('Unexpected full development scope or resource configuration')
    manifest=root/'review_bundle/CURRENT_BUILD_SOURCES.json'
    if sha(manifest)!=args.expected_source_sha256:raise ValueError('Build manifest changed')
    for name,digest in read(manifest)['files'].items():
        if sha(root/name)!=digest:raise ValueError('Verified source changed: '+name)
    if {r['key'] for r in config.maps('train')}!=DEVELOPMENT:raise ValueError('Development roster changed')
    check_holdout(root)
    with locked(root/'.full_run_supervisor.lock'):
        started=utc()
        (root/'logs').mkdir(exist_ok=True)
        try:
            snapshot(root,config,'preparing_development_data')
            run_phase(root,config,'preparing_development_data','prepare')
            from fleet_switching.data import load_dataset
            data=load_dataset(config)
            counts=Counter((e['city'],e['split']) for e in data['episodes'])
            expected={(c,s):n for c in DEVELOPMENT for s,n in [('train',100),('validation',25)]}
            if counts!=expected or len(data['episodes'])!=1000:raise ValueError('Unexpected dataset roster')
            atomic(root/'FULL_DEVELOPMENT_DATA_VERIFIED.json',{'verified_utc':utc(),
                'development_manifest_sha256':sha(root/'datasets/development_manifest.json'),
                'inputs':1000,'training_inputs':800,'validation_inputs':200,
                'cities':sorted(DEVELOPMENT),'held_out_city_overlap':[],
                'seed_count':len({e['seed'] for e in data['episodes']}),
                'workload_count':len({e['workload_sha256'] for e in data['episodes']})})
            run_phase(root,config,'training_and_validation','train')
            if not (root/'frozen_models.json').exists():raise ValueError('Missing final model freeze')
            final=snapshot(root,config,'complete')
            if (final['training_episodes_complete'],final['validation_outcomes_complete'])!=(1600,2400):
                raise ValueError('Incomplete development campaign')
            atomic(root/'FULL_DEVELOPMENT_COMPLETE.json',{'started_utc':started,'completed_utc':utc(),
                'training_episodes':1600,'development_validation_outcomes':2400,
                'final_test_opened':False,'models':read(root/'frozen_models.json')['models']})
        except BaseException:
            atomic(root/'full_run_status.json',{'updated_utc':utc(),'stage':'failed',
                'supervisor_pid':os.getpid(),'error':traceback.format_exc(),'final_test_opened':False})
            raise


if __name__=='__main__':main()
