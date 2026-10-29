"""Verify the separate Linux build and start only its development service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fleet_switching.config import Config, NO_FULL_MLE_POLICIES, atomic, available_gib, read, sha, workers

UNIT = 'robot-switching-no-full-mle-20261029.service'
BACKEND_COMMIT = '5e76f5384aec98e9d5a538b27dae205df301fa28'


def preflight():
    if sys.platform != 'linux':
        raise RuntimeError('Launch requires the Linux VM; no local training is started')
    config = Config.load(ROOT/'configs/small_city_switching_no_full_mle_20261029.json')
    if Path(config.output).resolve() != ROOT:
        raise ValueError('Extract into the new campaign output directory before launching')
    if ROOT.name != 'icaps_small_city_switching_no_full_mle_v1_20261029':
        raise ValueError('Refuse to reuse an existing experiment directory')
    if tuple(config.policies) != NO_FULL_MLE_POLICIES:
        raise ValueError('Unexpected action portfolio')
    manifest = ROOT/'review_bundle/CURRENT_BUILD_SOURCES.json'
    sources = read(manifest)
    if sources['vm_stage_target'] != str(ROOT):
        raise ValueError('Source manifest targets another experiment')
    for name, digest in sources['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError('Source checksum mismatch: '+name)
    def git(*args):
        return subprocess.check_output(['git','-C',config.backend_root,*args],text=True).strip()
    if git('rev-parse','HEAD') != BACKEND_COMMIT or git('status','--porcelain'):
        raise ValueError('Pinned simulator backend must be clean and unchanged')
    for path in (config.spatial_model, config.paper_model):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    config.maps('train')
    from run_switching_no_full_mle import check_holdout
    check_holdout(ROOT)
    cpus, memory = len(os.sched_getaffinity(0)), available_gib()
    slots = workers(config, cpus, memory)
    if slots < 1:
        raise RuntimeError('Insufficient memory/CPU capacity; no service started')
    return config, {'source_manifest_sha256':sha(manifest), 'available_gib':memory,
        'logical_cpus':cpus, 'admitted_worker_slots':slots, 'fit_threads':config.fit_threads,
        'policies':list(config.policies), 'unit':UNIT, 'final_test_opened':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    config, receipt = preflight()
    print(json.dumps(receipt, indent=2), flush=True)
    if not args.execute:
        return
    state = subprocess.run(['systemctl','--user','is-active',UNIT],text=True,capture_output=True)
    if state.stdout.strip() in ('active','activating','deactivating','reloading'):
        raise RuntimeError('Campaign service already exists; inspect it rather than launching a duplicate')
    if (ROOT/'FULL_DEVELOPMENT_COMPLETE.json').exists():
        raise RuntimeError('This campaign is complete; refuse an unnecessary restart')
    atomic(ROOT/'LAUNCH_PREFLIGHT.json',receipt)
    command = ['systemd-run','--user','--unit='+UNIT,'--description=Nine-action switching XGBoost and NN',
        '--working-directory='+str(ROOT),'--property=Restart=on-failure','--property=RestartSec=30s',
        '--property=StartLimitIntervalSec=0','--property=RuntimeMaxSec=infinity',
        '--property=TimeoutStopSec=45s','--property=KillMode=control-group',
        '--property=CPUAccounting=yes','--property=MemoryAccounting=yes',
        '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=PYTHONUNBUFFERED=1',
        sys.executable,'-B','-u',str(ROOT/'operations/run_switching_no_full_mle.py'),
        '--execute','--expected-source-sha256',receipt['source_manifest_sha256']]
    subprocess.run(command,check=True)
    print('Development service submitted; inspect systemctl and full_run_status.json to confirm progress.',flush=True)


if __name__ == '__main__':
    main()
