"""Read-only experiment audit, excluding final-test files and simulations."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

BASE = Path('/data/workspace/robot_delivery/dynamic-delivery-robot-fleet')
ROOT = BASE/'runs/icaps_small_city_switching_top5_no_full_mle_v1_20261029'
OLD = BASE/'runs/icaps_small_city_switching_v4_pilot_fixes_20261025'
UNIT = 'robot-switching-top5-no-full-mle-20261029.service'
ALLOWED = {'full_no_nn','full','full_queue_aware','full_coordinated_idle','queue_mle_reservation'}
DEVELOPMENT = {'tel_aviv','haifa','manhattan','new_delhi','delft','leuven','oxford','palo_alto'}
HELD_OUT = {'cambridge_uk','ghent','salamanca','lucca'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service():
    result = subprocess.check_output(['systemctl','--user','show',UNIT,
        '-p','ActiveState','-p','SubState','-p','MainPID','-p','NRestarts','-p','MemoryCurrent',
        '-p','CPUUsageNSec','-p','ExecMainStartTimestamp','-p','RuntimeMaxUSec',
        '-p','Restart','-p','ControlGroup'],text=True)
    return dict(line.split('=',1) for line in result.splitlines() if '=' in line)


def main():
    before, started = service(), time.monotonic()
    time.sleep(10)
    after = service()
    cpu_cores = (int(after['CPUUsageNSec'])-int(before['CPUUsageNSec']))/1e9/(time.monotonic()-started)
    manifest = read(ROOT/'datasets/development_manifest.json')
    old_manifest = read(OLD/'datasets/development_manifest.json')
    original = {row['id']:row for row in old_manifest['episodes']}
    fields = ['seed','scenario_sha256','workload_sha256','graph_sha256','fleet_count']
    mismatches = [row['id'] for row in manifest['episodes']
        if row['id'] not in original or any(row[key]!=original[row['id']][key] for key in fields)]
    cities = {row['city'] for row in manifest['episodes']}
    assert cities == DEVELOPMENT and not cities & HELD_OUT
    assert not mismatches
    assert not (ROOT/'test_opened.json').exists()
    assert not (ROOT/'datasets/test_manifest.json').exists()
    assert not any((ROOT/'actors').glob('*/test/*'))
    db = sqlite3.connect('file:'+str(ROOT/'replay.sqlite')+'?mode=ro',uri=True)
    families, cost_counts, row_cities, actor_kinds, choices = Counter(), Counter(), set(), {}, {}
    for family, payload in db.execute('SELECT family,payload FROM rows'):
        row = json.loads(payload)
        assert row['split']=='train' and row['city'] in DEVELOPMENT
        assert set(row['costs']) == ALLOWED == set(row['diagnostics'])
        assert row['intent']['chosen_policy'] in ALLOWED
        assert row['science_id'] == manifest['science_id']
        assert row['time_min'] % 15 == 0
        families[family] += 1
        cost_counts[len(row['costs'])] += 1
        row_cities.add(row['city'])
        actor_kinds.setdefault(family,Counter())[row['intent']['model_kind']] += 1
        choices.setdefault(family,Counter())[row['intent']['chosen_policy']] += 1
    db.close()
    models = {}
    for family in ('xgboost','nn'):
        pointer = read(ROOT/'models'/family/'current.json')
        model = read(Path(pointer['path']))
        assert sha(Path(pointer['path'])) == pointer['sha256']
        assert set(model['policies']) == ALLOWED
        heads = (len(model['boosters']) if model['kind']=='xgboost' else
                 len(model['weights'][-1][0]) if model['kind']=='nn' else None)
        if heads is not None:
            assert heads == 5
        models[family] = {**pointer,'kind':model['kind'],'output_heads':heads,'policies':model['policies']}
    originals = {}
    for family, expected in [('xgboost','e6b8598d308206554428603db7e84a30164f6a13989ff2b0dd79f299076e0cdf'),
                              ('nn','b8751b3ba6f9299b679dae50633267d859349994890ba7ed4b5f618c303ce576')]:
        pointer = read(OLD/'models'/family/'current.json')
        actual = sha(Path(pointer['path']))
        assert actual == expected
        originals[family] = actual
    previous_service = subprocess.run(['systemctl','--user','is-active',
        'robot-switching-no-full-mle-20261029.service'],capture_output=True,text=True).stdout.strip()
    assert previous_service == 'inactive'
    print(json.dumps({'previous_nine_action_service_state':previous_service, 'audited_utc':datetime.now(timezone.utc).isoformat(),
        'source_git_commit':'dc2f75788210fda4ce9887bba1c68a7e97d9c822',
        'source_manifest_sha256':sha(ROOT/'review_bundle/CURRENT_BUILD_SOURCES.json'),
        'campaign_id':manifest['campaign_id'],'science_id':manifest['science_id'],
        'service':after,'average_cpu_cores_over_10_seconds':cpu_cores,
        'percent_of_72_logical_cpus':cpu_cores/72*100,
        'training_status':read(ROOT/'training_status.json'),
        'scenario_inputs':len(manifest['episodes']),
        'scenario_counts':{city:dict(Counter(e['split'] for e in manifest['episodes'] if e['city']==city)) for city in sorted(cities)},
        'original_workload_pairing':{'fields':fields,'mismatches':mismatches},
        'development_manifest_sha256':sha(ROOT/'datasets/development_manifest.json'),
        'replay_rows_by_family':dict(families),'replay_cost_key_counts':dict(cost_counts),
        'replay_actor_kinds':actor_kinds,'replay_choices':choices,'replay_cities':sorted(row_cities),
        'models':models,'original_model_sha256_unchanged':originals,
        'held_out_cities':sorted(HELD_OUT),'held_out_inputs':0,'test_opened':False,
        'old_training_labels_reused':0},indent=2))


if __name__ == '__main__':
    main()
