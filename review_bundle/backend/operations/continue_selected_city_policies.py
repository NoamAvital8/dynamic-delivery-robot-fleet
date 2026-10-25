"""Continue ten selected policies while adopting, never restarting, live simulators.

This operational controller is deliberately outside scripts/: simulator source
fingerprints remain unchanged. scope_change.json records the verified handoff.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from run_multicity_experiments import (
    MLEConfig, _atomic_json, _execute, _resolve, _sha256, _source_digest,
    _write_summary, mle_worker_budget, order_jobs, policy_command, summarize,
)

SELECTED_POLICIES = (
    'full', 'myopic_ab', 'reactive_insertion', 'paper_sa_adapted',
    'full_no_nn', 'full_no_idle', 'full_queue_aware', 'full_coordinated_idle',
    'queue_mle_reservation', 'full_mle_reservation',
)
EXPECTED_CITIES = ('beijing', 'sydney', 'moscow', 'johannesburg', 'new_delhi', 'paris')


def process_record(pid):
    """Linux identity includes start ticks, protecting against PID reuse."""
    folder = Path(f'/proc/{pid}')
    try:
        fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        command = (folder/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
            'ppid': int(fields[1]), 'pgid': int(fields[2]),
            'command': [x.decode() for x in command if x]}


def same_live_process(record):
    current = process_record(record['pid'])
    return (current is not None and current['start_ticks'] == record['start_ticks']
            and current['state'] not in ('Z', 'X'))


def build_jobs(campaign, manifest):
    if _source_digest() != manifest['source_sha256']:
        raise RuntimeError('simulator sources changed; cannot continue compatible work')
    for name, path in [('spatial', manifest['spatial_model']), ('paper', manifest['paper_model'])]:
        if _sha256(Path(path)) != manifest['model_sha256'][name]:
            raise RuntimeError(f'{name} model changed')
    suite_path = Path(manifest['suite'])
    suite = json.loads(suite_path.read_text())
    if tuple(suite['cities']) != EXPECTED_CITIES:
        raise RuntimeError('expected only the six authorized new cities')
    if manifest['timeout_seconds'] != 0 or mle_worker_budget(
            manifest['processes'], manifest['idle_processes'],
            manifest['mle_reservation_config']['processes']) > 60:
        raise RuntimeError('timeout or compute-worker configuration changed')
    jobs = []
    for ci, (city, data) in enumerate(suite['cities'].items()):
        graph = _resolve(suite_path.parent, data['graph'])
        graph_hash = _sha256(graph)
        if graph_hash != manifest['graph_sha256'][str(graph)]:
            raise RuntimeError(f'prepared graph changed: {city}')
        if len(data['test']) != 5:
            raise RuntimeError('expected five paired test scenarios per city')
        for si, record in enumerate(data['test']):
            scenario = _resolve(suite_path.parent, record['scenario'])
            scenario_hash = _sha256(scenario)
            for pi, policy in enumerate(SELECTED_POLICIES):
                folder = campaign/'benchmarks'/city/record['id']/policy
                command = policy_command(
                    policy, python=sys.executable, graph=graph, scenario=scenario,
                    output=folder/'result.json',
                    spatial_model=Path(manifest['spatial_model']),
                    paper_model=Path(manifest['paper_model']),
                    shortlist_k=manifest['shortlist_k'], idle_processes=manifest['idle_processes'],
                    prior_rates=data['prior_rates_per_hour'],
                    prior_concentration=manifest.get('prior_concentration', 4.0),
                    relocation_uncertainty_penalty=manifest['idle_relocation_uncertainty_penalty'],
                    switching_gain_fraction=manifest['idle_switch_gain_fraction'],
                    retarget_cooldown_min=manifest['idle_retarget_cooldown_min'],
                    mle_config=MLEConfig(**manifest['mle_reservation_config']),
                )
                signature = hashlib.sha256(json.dumps({
                    'source': manifest['source_sha256'], 'graph': graph_hash,
                    'scenario': scenario_hash, 'models': manifest['model_sha256'],
                    'command': command,
                }, sort_keys=True).encode()).hexdigest()
                jobs.append({'city_index': ci, 'scenario_index': si, 'policy_index': pi,
                             'city': city, 'scenario_id': record['id'], 'seed': record['seed'],
                             'orders': record['orders'], 'policy': policy,
                             'output': folder/'result.json', 'log': folder/'runner.log',
                             'status_path': folder/'status.json', 'stamp_path': folder/'stamp.json',
                             'command': command, 'signature': signature,
                             'accepted_signatures': {signature}})
    if len(jobs) != 300:
        raise RuntimeError('expected exactly 300 selected evaluations')
    return order_jobs(jobs, manifest['job_order'])


def validated_row(job):
    result = json.loads(job['output'].read_text())
    if (not math.isfinite(float(result['loss_objective']))
            or not math.isfinite(float(result['wall_clock_seconds']))
            or result['wall_clock_seconds'] < 0 or result['orders'] != job['orders']
            or result['scenario_seed'] != job['seed']):
        raise RuntimeError(f'invalid or mismatched result: {job["output"]}')
    return summarize(result, job)


def inspect_jobs(jobs):
    completed = 0
    for job in jobs:
        stamp = job['stamp_path']
        if job['output'].exists() and stamp.exists():
            if json.loads(stamp.read_text())['signature'] != job['signature']:
                raise RuntimeError(f'incompatible completed result: {job["output"]}')
            validated_row(job)
            completed += 1
    return completed


def adopt_simulator(job, record, poll_seconds=5):
    if record['command'] != job['command'] or record['signature'] != job['signature']:
        raise RuntimeError('live simulator command/signature mismatch')
    while same_live_process(record):
        time.sleep(poll_seconds)
    # Never relaunch a preserved simulator. Its output must be valid on exit.
    row = validated_row(job)
    _atomic_json(job['stamp_path'], {'signature': job['signature']})
    _atomic_json(job['status_path'], row)
    return row


def new_job_is_unclaimed(job, adopted):
    status = job['status_path']
    if status.exists() and json.loads(status.read_text()).get('status') == 'running':
        if str(job['output']) not in adopted:
            raise RuntimeError(f'unclaimed running simulator; will not duplicate: {job["output"]}')


def run(campaign):
    scope = json.loads((campaign/'scope_change.json').read_text())
    manifest = scope['original_manifest']
    if tuple(scope['selected_policies']) != SELECTED_POLICIES:
        raise RuntimeError('unexpected selected policies')
    jobs = build_jobs(campaign, manifest)
    completed = inspect_jobs(jobs)
    adopted = {r['output']: r for r in scope['preserved_simulators']}
    allowed = {str(j['output']): j for j in jobs}
    if len(adopted) != len(scope['preserved_simulators']) or set(adopted)-set(allowed):
        raise RuntimeError('duplicate or excluded live simulator in handoff')
    if len(adopted) > manifest['processes']:
        raise RuntimeError('preserved simulators exceed slot limit')
    for job in jobs:
        new_job_is_unclaimed(job, adopted)
    for old in scope['previous_controllers']:
        if same_live_process(old):
            raise RuntimeError('previous scheduler is still alive; refusing duplicate scheduling')
    updated = {**manifest, 'policies': list(SELECTED_POLICIES),
               'expected_evaluations': 300, 'scope_change_file': str(campaign/'scope_change.json')}
    _atomic_json(campaign/'benchmarks/run_manifest.json', updated)
    state = campaign/'campaign_status.json'
    _atomic_json(state, {'stage': 'benchmarks', 'status': 'running',
                         'expected_evaluations': 300, 'policies': list(SELECTED_POLICIES),
                         'preserved_simulators': len(adopted), 'completed_at_handoff': completed})
    _write_summary(campaign/'benchmarks', jobs)
    # Adopted processes occupy slots first; only free slots can start new jobs.
    jobs.sort(key=lambda j: str(j['output']) not in adopted)
    print(f'jobs=300 cached={completed} preserved-live={len(adopted)} processes={manifest["processes"]}', flush=True)
    _atomic_json(campaign/'scope_controller_ready.json', {'pid': os.getpid(), 'jobs': 300})
    count = 0
    with ThreadPoolExecutor(max_workers=manifest['processes']) as executor:
        futures = {}
        for job in jobs:
            record = adopted.get(str(job['output']))
            future = (executor.submit(adopt_simulator, job, record) if record
                      else executor.submit(_execute, job, None, False))
            futures[future] = job
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
                if row['status'] != 'complete':
                    raise RuntimeError(row.get('error', 'simulation failed'))
            except Exception as exc:
                for other in futures:
                    other.cancel()
                _atomic_json(state, {'stage': 'failed', 'status': 'failed',
                                     'expected_evaluations': 300, 'error': str(exc)})
                raise
            count += 1
            _write_summary(campaign/'benchmarks', jobs)
            print(f'[{count}/300] {job["city"]} {job["scenario_id"]} {job["policy"]} '
                  f'complete loss={row["loss_objective"]}', flush=True)
    if inspect_jobs(jobs) != 300:
        raise RuntimeError('completion verification failed')
    _atomic_json(state, {'stage': 'complete', 'status': 'complete', 'expected_evaluations': 300})
    print('complete: 300/300 selected evaluations', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.inspect:
        path = campaign/'scope_change.json'
        manifest = (json.loads(path.read_text())['original_manifest'] if path.exists()
                    else json.loads((campaign/'benchmarks/run_manifest.json').read_text()))
        jobs = build_jobs(campaign, manifest)
        print(json.dumps({'selected_policies': SELECTED_POLICIES, 'jobs': len(jobs),
                          'compatible_completed': inspect_jobs(jobs)}))
    else:
        import fcntl  # Linux VM only; prevents two operational controllers.
        with (campaign/'scope_controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(campaign)


if __name__ == '__main__':
    main()
