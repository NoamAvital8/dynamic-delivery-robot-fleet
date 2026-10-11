"""RAM/worker-aware continuation: adopt live jobs; optimize pending jobs only."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import argparse

import continue_selected_city_policies as original
from exact_runtime import file_hash

ROOT = Path(__file__).resolve().parents[1]
IDLE_POLICIES = {'full','full_no_nn','full_queue_aware','full_coordinated_idle',
                 'queue_mle_reservation','full_mle_reservation'}
GIB = 1024**3
# Conservative initial private-memory allowance, plus one unbuilt dense index.
PRIVATE_GIB = {'beijing':8, 'sydney':8, 'moscow':8, 'johannesburg':4,
               'paris':1, 'new_delhi':1}
INDEX_GIB = {'beijing':25, 'sydney':10, 'moscow':7, 'johannesburg':2,
             'paris':.1, 'new_delhi':.01}


def read_json(path):
    return json.loads(Path(path).read_text())


def available_memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return fields['MemAvailable']


def index_ready(job, manifest, cache):
    graph = job['command'][job['command'].index('--graph')+1]
    digest = manifest['graph_sha256'][graph]
    marker = cache/'graphs'/f'{digest}.json'
    if not marker.exists():
        return False
    data = read_json(marker)
    return (data.get('graph_sha256') == digest and
            (cache/data['index_key']/'ready.json').exists())


def worker_cost(job, phase='simulation', idle_workers=4, preprocessing_workers=4):
    if phase == 'preprocessing':
        return 1 + preprocessing_workers
    return 1 + idle_workers if job['policy'] in IDLE_POLICIES else 1


def memory_cost(job, ready):
    return int((PRIVATE_GIB[job['city']] + (0 if ready else INDEX_GIB[job['city']]))*GIB)


def can_admit(*, live_count, workers_used, cost, available, reserved, needed,
              max_simulators=24, max_workers=60, reserve_gib=16):
    return (live_count < max_simulators and workers_used+cost <= max_workers
            and available-reserved-needed >= reserve_gib*GIB)


def execute_optimized(job, cache, workers, code_hash):
    execution = job['output'].with_suffix('.execution.json')
    actual_command = [sys.executable, str(ROOT/'operations/run_exact_optimized_policy.py'),
        '--cache-dir',str(cache),'--workers',str(workers),'--execution-status',str(execution),
        '--',*job['command'][1:]]
    job['output'].parent.mkdir(parents=True, exist_ok=True)
    original._atomic_json(execution, {'phase':'preprocessing', 'execution_sha256':code_hash,
        'scientific_signature':job['signature'], 'original_command':job['command'],
        'actual_command':actual_command, 'preprocessing_workers':workers})
    # Input signature remains the *original* command and exact policy source.
    # The different execution command/version is explicit, never disguised.
    row = original._execute({**job,'command':actual_command}, None, False)
    if row['status']=='complete':
        original.validated_row(job)
        original._atomic_json(job['stamp_path'], {'signature':job['signature'],
                              'execution_sha256':code_hash, 'execution_command':actual_command})
        data = read_json(execution); data['phase']='complete'
        original._atomic_json(execution, data)
    return row


def pending_jobs(jobs, records):
    """No stamped result or live process is ever submitted for execution."""
    result = []
    for job in jobs:
        if job['output'].exists() and job['stamp_path'].exists():
            continue
        if str(job['output']) in records:
            continue
        original.new_job_is_unclaimed(job, records)
        if job['output'].exists():
            raise RuntimeError(f'unclaimed unstamped result: {job["output"]}')
        result.append(job)
    return result


def run(campaign, max_simulators=24, max_workers=60, preprocessing_workers=4, reserve_gib=16):
    if not (1 <= max_simulators <= 60 and 1 <= max_workers <= 60
            and 1 <= preprocessing_workers <= 4 and reserve_gib >= 16):
        raise ValueError('unsafe process/memory configuration')
    handoff = read_json(campaign/'optimization_handoff.json')
    scope = read_json(campaign/'scope_change.json')
    manifest = scope['original_manifest']
    jobs = original.build_jobs(campaign, manifest)  # Original source/models/maps verified.
    for previous in handoff['previous_controllers']:
        if original.same_live_process(previous):
            raise RuntimeError('previous controller still alive; refusing duplicate scheduling')
    for path, digest in handoff['completed_result_sha256'].items():
        if file_hash(path) != digest:
            raise RuntimeError(f'completed evaluation changed: {path}')
    records = {record['output']:record for record in handoff['preserved_simulators']}
    if len(records) != len(handoff['preserved_simulators']):
        raise RuntimeError('duplicate adoption records')
    allowed = {str(j['output']):j for j in jobs}
    if set(records)-set(allowed):
        raise RuntimeError('out-of-scope adopted evaluation')
    # Validate every identity/argv/signature *before* any new simulator launches.
    for path, record in records.items():
        job = allowed[path]
        if record['command'] != job['command'] or record['signature'] != job['signature']:
            raise RuntimeError('preserved simulator input identity mismatch')
        if original.same_live_process(record):
            if original.process_record(record['pid'])['command'] != record['actual_command']:
                raise RuntimeError('preserved simulator argv mismatch')
    pending = pending_jobs(jobs, records)
    cache = campaign/'shared_exact_index'
    code_hash = file_hash(ROOT/'operations/exact_runtime.py')+'-'+file_hash(
        ROOT/'operations/run_exact_optimized_policy.py')
    identity = original.process_record(os.getpid())
    log = campaign/'stages/benchmarks_exact_optimized.log'
    old_launch = read_json(campaign/'launch.json')
    launch = {**old_launch, 'launched_at':datetime.now(timezone.utc).isoformat(),
        'controller_pid':os.getpid(), 'controller_start_ticks':identity['start_ticks'],
        'controller_process_group':identity['pgid'], 'argv':identity['command'],
        'log':str(log), 'maximum_compute_workers':max_workers,
        'benchmark_processes':max_simulators, 'preprocessing_workers':preprocessing_workers,
        'memory_reserve_gib':reserve_gib, 'execution_optimization':code_hash,
        'optimization_handoff_file':str(campaign/'optimization_handoff.json')}
    original._atomic_json(campaign/'launch.json',launch)
    status_path = campaign/'campaign_status.json'
    original._atomic_json(campaign/'optimization_controller_ready.json',
                         {'pid':os.getpid(),'jobs':300,'pending':len(pending)})
    print(f'exact continuation jobs=300 completed={original.inspect_jobs(jobs)} '
          f'preserved={len(records)} pending={len(pending)} max_simulators={max_simulators} '
          f'max_workers={max_workers}',flush=True)
    running = {}
    failed = None
    last_status = 0
    with ThreadPoolExecutor(max_workers=max(max_simulators,len(records))) as executor:
        for path, record in records.items():
            job = allowed[path]
            future = executor.submit(original.adopt_simulator,job,record)
            running[future] = {'job':job,'record':record,'optimized':False,'reserved_bytes':0}
        while pending or running:
            for future in list(running):
                if not future.done():
                    continue
                entry = running.pop(future)
                try:
                    row = future.result()
                    if row['status'] != 'complete':
                        raise RuntimeError(row.get('error','simulation failed'))
                    original._write_summary(campaign/'benchmarks',jobs)
                    print(f'[{original.inspect_jobs(jobs)}/300] {row["city"]} '
                          f'{row["scenario_id"]} {row["policy"]} loss={row["loss_objective"]}',flush=True)
                except Exception as exc:
                    failed = str(exc)
                    print(f'FAILURE: {failed}; no new jobs will launch; preserving live children',flush=True)
            workers_used = 0
            reserved = 0
            for entry in running.values():
                phase = 'simulation'
                if entry['optimized']:
                    execution = entry['job']['output'].with_suffix('.execution.json')
                    if not execution.exists() or read_json(execution).get('phase')=='preprocessing':
                        phase = 'preprocessing'
                    # Reserve predicted allocations until preparation ends.
                    # This prevents a burst of workers racing MemAvailable.
                    if phase=='preprocessing':
                        reserved += entry['reserved_bytes']
                workers_used += worker_cost(entry['job'],phase,
                    preprocessing_workers=preprocessing_workers)
            if not failed:
                for job in list(pending):
                    ready = index_ready(job,manifest,cache)
                    cost = max(worker_cost(job),1+preprocessing_workers)
                    needed = memory_cost(job,ready)
                    if not can_admit(live_count=len(running),workers_used=workers_used,cost=cost,
                            available=available_memory(),reserved=reserved,needed=needed,
                            max_simulators=max_simulators,max_workers=max_workers,reserve_gib=reserve_gib):
                        continue
                    future = executor.submit(execute_optimized,job,cache,preprocessing_workers,code_hash)
                    running[future]={'job':job,'optimized':True,'reserved_bytes':needed}
                    pending.remove(job)
                    reserved += needed
                    workers_used += cost
                    print(f'launch {job["city"]} {job["scenario_id"]} {job["policy"]} '
                          f'reserved={needed/GIB:.2f}GiB workers_budget={workers_used}',flush=True)
            if time.monotonic()-last_status > 30:
                original._atomic_json(status_path,{'stage':'failed' if failed else 'benchmarks',
                    'status':'failed' if failed else 'running','error':failed,
                    'expected_evaluations':300,'completed':original.inspect_jobs(jobs),
                    'running':len(running),'pending':len(pending),'workers_budget':workers_used,
                    'available_memory_gib':available_memory()/GIB,'optimization':code_hash})
                last_status = time.monotonic()
            if failed and not running:
                raise RuntimeError(failed)
            time.sleep(5)
    if original.inspect_jobs(jobs) != 300:
        raise RuntimeError('completion verification failed')
    original._atomic_json(status_path,{'stage':'complete','status':'complete','expected_evaluations':300})
    print('complete: 300/300',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--max-simulators',type=int,default=24)
    parser.add_argument('--max-workers',type=int,default=60)
    parser.add_argument('--preprocessing-workers',type=int,default=4)
    parser.add_argument('--memory-reserve-gib',type=float,default=16)
    args=parser.parse_args()
    import fcntl
    with (args.campaign/'scope_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.campaign,args.max_simulators,args.max_workers,
            args.preprocessing_workers,args.memory_reserve_gib)


if __name__=='__main__':
    main()
