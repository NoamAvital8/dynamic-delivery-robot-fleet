"""Resume saved development training with a recorded execution-only override.

The original sealed source files, campaign/science/learner identities, replay,
models, inputs and algorithm settings are unchanged. No final test is opened.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
assert ROOT.name=='icaps_small_city_switching_v4_pilot_fixes_20261025'
sys.path.insert(0,str(ROOT/'src'))
from fleet_switching.config import Config, atomic, available_gib, locked, read, sha

def utc():return datetime.now(timezone.utc).isoformat()

def effective_config():
    config=Config.load(ROOT/'configs/small_city_switching_v2.json')
    cpus=len(os.sched_getaffinity(0))
    # Four fit threads + one coordinator slot; collection consumes the rest
    # only while the unchanged RAM admission checks permit it.
    return replace(config,worker_limit=max(1,cpus-config.fit_threads-1),reserved_cpus=config.fit_threads+1)

def verify_sources():
    expected=read(ROOT/'review_bundle/CURRENT_BUILD_SOURCES.json')['files']
    for name,digest in expected.items():
        if sha(ROOT/name)!=digest:raise ValueError('Sealed source changed: '+name)
    if sha(ROOT/'src/fleet_switching/execution.py')!='d0cbd9dd2bba0665cc01f46f592981877abfcc64a37960af49024fe38aec6f90':
        raise ValueError('Unexpected execution source')
    for path in (ROOT/'test_opened.json',ROOT/'datasets/test_manifest.json'):
        if path.exists():raise ValueError('Final test must remain unopened')

def install_override():
    name='fleet_switching.execution'
    spec=importlib.util.spec_from_file_location(name,HERE/'execution_recovery.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def worker():
    verify_sources()
    config=effective_config()
    execution=install_override()
    execution.numeric_environment(1)
    from fleet_switching.provenance import ensure_campaign
    before=ensure_campaign(config)
    from fleet_switching.workflow import train
    atomic(HERE/'execution_override_active.json',{'updated_utc':utc(),'pid':os.getpid(),
        'campaign_id':before['campaign_id'],'science_id':before['science_id'],
        'original_execution_sha256':sha(ROOT/'src/fleet_switching/execution.py'),
        'execution_override_sha256':sha(HERE/'execution_recovery.py'),
        'worker_limit':config.worker_limit,'reserved_cpus':config.reserved_cpus,'fit_threads':config.fit_threads,
        'memory_reserve_gib':config.memory_reserve_gib,'fit_timeout_enabled':False,
        'labels_inputs_and_learners_unchanged':True})
    result=train(config)
    atomic(HERE/'recovered_development_complete.json',{'completed_utc':utc(),'report':result,
        'final_test_opened':False})

def status(child,stage):
    train_count=len(list(ROOT.glob('actors/*/train/*/complete.json')))
    val_count=len(list(ROOT.glob('actors/*/validation/*/complete.json')))
    cfg=effective_config()
    value={'updated_utc':utc(),'stage':stage,'supervisor_pid':os.getpid(),'child_pid':child.pid,
        'training_episodes_complete':train_count,'training_episodes_expected':1600,
        'validation_outcomes_complete':val_count,'validation_outcomes_expected':2400,
        'worker_limit':cfg.worker_limit,'fit_threads':cfg.fit_threads,'available_gib':available_gib(),
        'controller':read(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {},
        'final_test_opened':False,'recovery_execution_override':str(HERE/'execution_recovery.py')}
    if val_count and stage=='training_and_validation':value['stage']='development_validation'
    atomic(ROOT/'full_run_status.json',value)
    return value

def immutable_preservation_path(path):
    """Allow checkpoints/current pointers to advance; protect finished data."""
    p=Path(path)
    relative=p.relative_to(ROOT)
    if relative.parts[0]=='datasets':return True
    if relative.parts[0]=='actors':return p.name=='complete.json'
    if relative.parts[0]=='models':return p.name not in ('current.json','last_fit_runtime.json')
    return False

def supervisor():
    verify_sources()
    with locked(HERE/'.recovery_supervisor.lock'):
        # Fail closed if another controller holds the campaign lock.
        with locked(ROOT/'.campaign.lock'):pass
        archive=HERE/'before_resume'
        archive.mkdir(exist_ok=True)
        for name in ('full_run_status.json','training_status.json','last_shutdown.json','train_process.json',
                     'campaign.json','datasets/development_manifest.json','execution_settings.json'):
            source=ROOT/name
            target=archive/name.replace('/','__')
            if source.exists() and not target.exists():target.write_bytes(source.read_bytes())
        backup=archive/'replay.sqlite'
        if not backup.exists():
            with sqlite3.connect((ROOT/'replay.sqlite').as_uri()+'?mode=ro',uri=True) as source:
                with sqlite3.connect(backup) as dest:source.backup(dest)
        baseline=read(HERE/'comparison_paused.json')
        old_launch=HERE/'RECOVERY_LAUNCH.json'
        if old_launch.exists():
            history=HERE/'launch_history'
            history.mkdir(exist_ok=True)
            (history/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')).write_bytes(old_launch.read_bytes())
        # The first recovery verified ALL saved artifacts before any progress.
        # Later restarts must not mistake legitimate checkpoint/current-model
        # advancement or superseded-state retirement for data corruption.
        previous_launch=(HERE/'RECOVERY_LAUNCH.json').exists()
        protected={name:digest for name,digest in baseline['protected_sha256'].items()
            if not previous_launch or immutable_preservation_path(name)}
        mismatches=[name for name,digest in protected.items() if sha(name)!=digest]
        if mismatches:raise ValueError('Preserved artifact changed before recovery: '+str(mismatches[:5]))
        args=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--execute','--worker']
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        child=None
        try:
            with (HERE/'training_resume.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
                atomic(HERE/'RECOVERY_LAUNCH.json',{'launched_utc':utc(),'supervisor_pid':os.getpid(),
                    'child_pid':child.pid,'child_start_ticks':int(stat[19]),'command':args,
                    'completed_training_before_resume':len(list(ROOT.glob('actors/*/train/*/complete.json'))),
                    'protected_artifacts_verified':len(protected),
                    'supervisor_sha256':sha(Path(__file__).resolve()),
                    'no_final_test':True,'fit_timeout_enabled':False,'resource_priority':'training_only',
                    'worker_limit':effective_config().worker_limit,'fit_threads':effective_config().fit_threads,
                    'override_sha256':sha(HERE/'execution_recovery.py')})
                while child.poll() is None:
                    status(child,'training_and_validation')
                    time.sleep(10)
                if child.returncode:raise RuntimeError('Recovery controller exited '+str(child.returncode))
            final=status(child,'complete')
            if (final['training_episodes_complete'],final['validation_outcomes_complete'])!=(1600,2400):
                raise ValueError('Incomplete development campaign')
            atomic(ROOT/'FULL_DEVELOPMENT_COMPLETE.json',{'completed_utc':utc(),'recovered':True,
                'training_episodes':1600,'development_validation_outcomes':2400,
                'final_test_opened':False,'models':read(ROOT/'frozen_models.json')['models']})
        except BaseException:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=10)
            failure={'updated_utc':utc(),'stage':'failed','error':traceback.format_exc(),'final_test_opened':False}
            atomic(HERE/'recovery_failed.json',failure)
            atomic(ROOT/'full_run_status.json',failure)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    if args.check_only:
        verify_sources()
        from fleet_switching.provenance import ensure_campaign
        config=effective_config()
        campaign=ensure_campaign(config)
        print({'status':'compatible','campaign_id':campaign['campaign_id'],
            'science_id':campaign['science_id'],'worker_limit':config.worker_limit,
            'fit_threads':config.fit_threads,'available_gib':available_gib()})
        return
    def interrupted(*_):raise KeyboardInterrupt('Explicit supervisor stop')
    if not args.worker:
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
    (worker if args.worker else supervisor)()

if __name__=='__main__':main()
