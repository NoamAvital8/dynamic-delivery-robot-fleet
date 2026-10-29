"""Verify saved development state and install its reboot-persistent user service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

CAMPAIGN = Path('/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_top5_no_full_mle_v1_20261029')
UNIT = 'robot-switching-top5-no-full-mle-20261029.service'


def command(*args):
    return subprocess.check_output(list(args), text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        raise RuntimeError('Recovery runs only on the Linux VM')
    sys.path.insert(0, str(CAMPAIGN/'operations'))
    from launch_switching_top5_no_full_mle import preflight
    config, preflight_receipt = preflight()
    from fleet_switching.config import atomic, read, sha
    from fleet_switching.provenance import ensure_campaign, require_development_open
    require_development_open(config)
    if (CAMPAIGN/'FULL_DEVELOPMENT_COMPLETE.json').exists():
        raise ValueError('Development is already complete; do not restart it')
    campaign = ensure_campaign(config)
    models = {}
    for family in ('xgboost', 'nn'):
        pointer = read(CAMPAIGN/'models'/family/'current.json')
        payload = read(pointer['path'])
        if (sha(pointer['path']) != pointer['sha256'] or tuple(payload['policies']) != config.policies
                or pointer['science_id'] != campaign['science_id']):
            raise ValueError('Saved model identity changed: '+family)
        models[family] = pointer
    db = sqlite3.connect('file:'+str(CAMPAIGN/'replay.sqlite')+'?mode=ro', uri=True)
    try:
        check = db.execute('PRAGMA quick_check').fetchall()
        if check != [('ok',)]:
            raise ValueError('Replay integrity check failed: '+str(check))
        replay_rows = dict(db.execute('SELECT family,COUNT(*) FROM rows GROUP BY family'))
    finally:
        db.close()
    if command('loginctl','show-user',str(os.getuid()),'-p','Linger','--value') != 'yes':
        raise ValueError('User lingering must be enabled for unattended boot recovery')
    state = subprocess.run(['systemctl','--user','is-active',UNIT],text=True,capture_output=True).stdout.strip()
    if state in ('active','activating','deactivating','reloading'):
        raise ValueError('Campaign is already running or changing state; do not duplicate it')
    unit = f'''[Unit]
Description=Five-action switching XGBoost and NN (persistent recovery)
StartLimitIntervalSec=0
ConditionPathExists=!{CAMPAIGN}/FULL_DEVELOPMENT_COMPLETE.json
ConditionPathExists=!{CAMPAIGN}/test_opened.json

[Service]
Type=simple
WorkingDirectory={CAMPAIGN}
ExecStart={sys.executable} -B -u {CAMPAIGN}/operations/run_switching_top5_no_full_mle.py --execute --expected-source-sha256 {preflight_receipt['source_manifest_sha256']}
Restart=on-failure
RestartSec=30s
RuntimeMaxSec=infinity
TimeoutStopSec=45s
KillMode=control-group
MemoryAccounting=yes
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
'''
    target = Path.home()/'.config/systemd/user'/UNIT
    if target.exists() and target.read_text() != unit:
        raise ValueError('Preserve the existing different service definition: '+str(target))
    receipt = {'verified_utc':datetime.now(timezone.utc).isoformat(),
        'campaign_id':campaign['campaign_id'],'science_id':campaign['science_id'],
        'preflight':preflight_receipt,'models_before_resume':models,
        'replay_quick_check':'ok','replay_rows':replay_rows,'unit_path':str(target),
        'unit_text':unit,'existing_service_state':state,'test_opened':False,
        'scientific_sources_modified':False,'new_training_started':False}
    print(json.dumps(receipt,indent=2),flush=True)
    if not args.execute:
        return
    evidence = CAMPAIGN/'reboot_recovery_20261029'
    atomic(evidence/'PRE_RESUME_VERIFICATION.json',receipt)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with target.open('x') as stream:
            stream.write(unit)
            stream.flush()
            os.fsync(stream.fileno())
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable',UNIT],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    receipt['enabled_state'] = command('systemctl','--user','is-enabled',UNIT)
    receipt['active_state'] = command('systemctl','--user','is-active',UNIT)
    receipt['resumed_utc'] = datetime.now(timezone.utc).isoformat()
    receipt['resumed_existing_training'] = True
    atomic(evidence/'RESUMED.json',receipt)
    print('Resumed existing campaign with a persistent boot-enabled user service.',flush=True)


if __name__ == '__main__':
    main()
