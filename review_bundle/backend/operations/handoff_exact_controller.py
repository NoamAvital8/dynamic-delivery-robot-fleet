"""Replace only the verified scheduler; preserve all simulator identities/data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import continue_selected_city_policies as original
from exact_runtime import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve()
    launch=json.loads((campaign/'launch.json').read_text())
    old=original.process_record(launch['controller_pid'])
    if (old is None or old['start_ticks']!=launch['controller_start_ticks']
            or old['command']!=launch['argv'] or old['state'] in ('Z','X')):
        raise RuntimeError('controller identity mismatch; no process changed')
    if (campaign/'optimization_handoff.json').exists():
        raise RuntimeError('handoff already exists; inspect before any recovery')
    scope=json.loads((campaign/'scope_change.json').read_text())
    jobs=original.build_jobs(campaign,scope['original_manifest'])
    allowed={str(job['output']):job for job in jobs}
    os.kill(old['pid'],signal.SIGSTOP)  # All scheduler threads; children untouched.
    try:
        live={}
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            try:
                record=original.process_record(int(folder.name))
            except (PermissionError,ProcessLookupError):
                continue
            if not record or record['state'] in ('Z','X') or '--output' not in record['command']:
                continue
            actual=record['command']
            output=actual[actual.index('--output')+1]
            if output not in allowed:
                continue
            job=allowed[output]
            if output in live:
                raise RuntimeError('duplicate live simulation detected')
            if actual!=job['command']:
                raise RuntimeError('unexpected live command; refusing adoption')
            live[output]={**record,'actual_command':actual,'command':job['command'],
                          'signature':job['signature'],'output':output}
        # A child can finish during handoff; adopt its output instead of rerunning.
        for job in jobs:
            if job['status_path'].exists():
                status=json.loads(job['status_path'].read_text())
                if status.get('status')=='running' and str(job['output']) not in live:
                    original.validated_row(job)
                    live[str(job['output'])]={'pid':0,'start_ticks':-1,'state':'X',
                        'actual_command':job['command'],'command':job['command'],
                        'signature':job['signature'],'output':str(job['output'])}
        results={str(job['output']):file_hash(job['output']) for job in jobs
                 if job['output'].exists() and job['stamp_path'].exists()}
        payload={'captured_at':datetime.now(timezone.utc).isoformat(),
            'previous_controllers':[old], 'preserved_simulators':list(live.values()),
            'completed_result_sha256':results,'previous_launch':launch,
            'selected_policies':list(original.SELECTED_POLICIES),'expected':300}
        original._atomic_json(campaign/'optimization_handoff.json',payload)
        original._atomic_json(campaign/'launch.before_exact_optimization.json',launch)
    except BaseException:
        os.kill(old['pid'],signal.SIGCONT)
        raise
    # Identity checked again; kill only this scheduler PID, NEVER its PGID.
    current=original.process_record(old['pid'])
    if current is None or current['start_ticks']!=old['start_ticks']:
        raise RuntimeError('scheduler identity changed during handoff')
    os.kill(old['pid'],signal.SIGKILL)
    deadline=time.monotonic()+10
    while original.same_live_process(old) and time.monotonic()<deadline:
        time.sleep(.1)
    if original.same_live_process(old):
        raise RuntimeError('old scheduler did not exit')
    log=campaign/'stages/benchmarks_exact_optimized.log'
    command=[sys.executable,'-u',str(Path(__file__).with_name('continue_exact_optimized_campaign.py')),
             '--campaign',str(campaign),'--max-simulators','24','--max-workers','60',
             '--preprocessing-workers','4','--memory-reserve-gib','16']
    environment={**os.environ,'PYTHONPATH':str(original.ROOT/'src')}
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[name]='1'
    with log.open('a') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,env=environment,cwd=original.ROOT,start_new_session=True)
    original._atomic_json(campaign/'optimization_dispatch.json',
        {'pid':process.pid,'command':command,'log':str(log),'preserved':len(live),
         'completed':len(results)})
    print(json.dumps({'new_controller_pid':process.pid,'preserved_simulators':len(live),
                      'completed_results_preserved':len(results),'log':str(log)}),flush=True)


if __name__=='__main__':
    main()
