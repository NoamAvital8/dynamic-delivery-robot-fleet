"""Offline pilot analysis. Reads saved evidence; never imports or runs a simulator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fleet_switching.config import DEVELOPMENT, HELD_OUT, POLICIES, identity, sha
from fleet_switching.provenance import validate_row


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def analyze(root):
    data=read(root/'PILOT_DATASET.json')
    training=read(root/'PILOT_TRAINING_RESULTS.json')
    validation=read(root/'PILOT_VALIDATION_RESULTS.json')
    episodes={e['id']:e for e in data['episodes']}
    train={k:e for k,e in episodes.items() if e['split']=='train'}
    val={k:e for k,e in episodes.items() if e['split']=='validation'}
    assert len(train)==16 and len(val)==2
    assert {e['city'] for e in train.values()}==DEVELOPMENT
    assert not {e['city'] for e in episodes.values()}&HELD_OUT
    assert not {e['workload_sha256'] for e in train.values()}&{e['workload_sha256'] for e in val.values()}
    assert len({e['seed'] for e in episodes.values()})==18
    original_root=PurePosixPath(read(root/'PILOT_CONFIG.json')['output']).parent
    for episode in episodes.values():
        path=root/str(PurePosixPath(episode['scenario']).relative_to(original_root))
        assert sha(path)==episode['scenario_sha256']
    assert len(training['results'])==32 and validation['completed_results']==24
    grouped=defaultdict(list)
    for result in training['results']:
        episode=train[result['episode']]
        assert result['episode_identity']==identity(episode)
        assert result['delivered']==result['orders']==episode['orders']
        grouped[episode['city'],episode['profile']].append(result)
    assert all(len(v)==2 for v in grouped.values())
    city={}
    for c in sorted(DEVELOPMENT):
        city[c]={p:{'mean_seconds':sum(r['pilot_job_wall_seconds'] for r in grouped[c,p])/2,
                   'worker_seconds':[r['pilot_job_wall_seconds'] for r in grouped[c,p]],
                   'orders':grouped[c,p][0]['orders'],'robots':grouped[c,p][0]['robots']}
                 for p in ['ordinary','charging_pressure']}
    rows=[];verified_models=set()
    with gzip.open(root/'TRAINING_ROWS.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)['row']
            validate_row(row,train,data['science_id'],row['family'])
            assert set(row['costs'])==set(POLICIES)
            assert all(math.isfinite(v) and v>=0 for v in row['costs'].values())
            intent=row['intent'];key=(intent['model_path'],intent['model_sha256'])
            if key not in verified_models:
                model=root/str(PurePosixPath(intent['model_path']).relative_to(original_root))
                assert sha(model)==intent['model_sha256']
                payload=read(model)
                assert payload['science_id']==data['science_id'] and payload['kind']==intent['model_kind']
                verified_models.add(key)
            rows.append(row)
    assert len(rows)==sum(s['count'] for s in training['replay'].values())
    teacher={}
    for family in ['xgboost','nn']:
        subset=[r for r in rows if r['family']==family]
        regrets=[(r['costs'][r['intent']['chosen_policy']]-min(r['costs'].values()))/r['robots'] for r in subset]
        teacher[family]={'states':len(subset),'mean_projected_regret_per_robot':sum(regrets)/len(subset),
            'all_policy_costs_tied_states':sum(max(r['costs'].values())-min(r['costs'].values())<1e-7 for r in subset),
            'bootstrap_states':sum(r['intent']['model_kind']=='bootstrap' for r in subset),
            'chosen_policies':dict(Counter(r['intent']['chosen_policy'] for r in subset)),
            'distinct_actor_versions':len({r['intent']['model_sha256'] for r in subset})}
    raw=defaultdict(float);projection=defaultdict(float)
    for r in rows:
        for info in r['diagnostics'].values():
            raw[r['city']]+=info['window_restore_advance_snapshot_seconds']
            projection[r['city']]+=info['projection_seconds']
    # Four actor episodes per city, each scoring all ten policies. Use those
    # state-conditional raw windows as a proxy for ten whole fixed episodes.
    proxy_city={c:raw[c]/4 for c in sorted(DEVELOPMENT)}
    validation_proxy_seconds=sum(proxy_city[c] for c in sorted({e['city'] for e in val.values()}))*12/10
    actual_val_seconds=sum(r['pilot_job_wall_seconds'] for r in validation['results'])
    calibration=actual_val_seconds/validation_proxy_seconds
    full_val_worker_hours=sum(proxy_city.values())*25*12/10*calibration/3600
    full_train_worker_hours=sum(r['pilot_job_wall_seconds'] for r in training['results'])*50/3600
    ordinary_weighted_hours=sum(200*(.8*city[c]['ordinary']['mean_seconds']+.2*city[c]['charging_pressure']['mean_seconds']) for c in sorted(DEVELOPMENT))/3600
    fits={}
    for f in ['xgboost','nn']:
        last=next(u for u in reversed(training['fit_updates']) if u['family']==f and u.get('final'))
        final_count=last['states']*50
        sizes=list(range(2000,final_count,2000))+[final_count]
        exponents=(1.,1.3) if f=='xgboost' else (1.,1.1)
        fits[f]={'pilot_final_states':last['states'],'pilot_final_fit_seconds':last['wall_seconds'],
            'projected_final_states':final_count,'projected_number_of_refits':len(sizes),
            'assumed_size_exponents':exponents,
            'projected_sum_fit_hours':[sum(last['wall_seconds']*(n/last['states'])**p for n in sizes)/3600 for p in exponents],
            'projected_final_fit_seconds':[last['wall_seconds']*(final_count/last['states'])**p for p in exponents]}
    fit_high=sum(v['projected_sum_fit_hours'][1] for v in fits.values())
    final_fit_low=sum(v['projected_final_fit_seconds'][0] for v in fits.values())/3600
    low=(min(ordinary_weighted_hours,full_train_worker_hours)+.7*full_val_worker_hours)/36+final_fit_low+.25
    high=(2*max(ordinary_weighted_hours,full_train_worker_hours)+1.5*full_val_worker_hours)/24+fit_high+.5
    estimate={'scope':'1600 training episodes plus 2400 development validation episodes; final test excluded',
        'measured_pilot_episode_workers':8,'future_admitted_episode_workers':64,
        'future_effective_worker_equivalent_range':[24,36],
        'training_worker_hours_same_50_50_mix':full_train_worker_hours,
        'training_worker_hours_80_20_ordinary_proxy':ordinary_weighted_hours,
        'training_high_work_multiplier':2.,'validation_low_high_work_multipliers':[.7,1.5],
        'validation_window_proxy_seconds_by_city':dict(sorted(proxy_city.items())),
        'validation_calibration_observed_cities':sorted({e['city'] for e in val.values()}),
        'validation_observed_to_training_window_proxy_ratio':calibration,
        'validation_worker_hours_calibrated_proxy':full_val_worker_hours,
        'fit_extrapolation':fits,'training_plus_validation_elapsed_hours_range':[low,high],
        'rounded_planning_budget_hours_range':[max(1,math.floor(low)),math.ceil(high)],
        'central_expectation_hours_range':[4,6],
        'practical_first_campaign_budget_hours_range':[6,12],
        'budget_interpretation':'Judgmental planning bands, not confidence intervals or a runtime guarantee; the arithmetic range includes an optimistic lower bound',
        'method':'Summed worker costs, separately calibrated validation and size-extrapolated fits; not pilot elapsed multiplied by 50',
        'limitations':['Only two training inputs per city; ordinary and charging-pressure, not all five production profiles',
            'Only two fresh validation inputs; transferring the validation calibration to six other cities is unmeasured',
            '64-worker SMT, memory, disk and SQLite contention have not been benchmarked',
            'Online learning can change trajectory cost and drain length',
            'Fit scaling exponents and effective concurrency are assumptions, not measurements'],
        'full_training_started':False,'final_test_time':'Not measured; held-out maps and workloads were not simulated'}
    analysis={'inputs_verified':len(episodes),'training_episode_receipts_verified':32,'matched_validation_outcomes':24,
        'training_rows_content_hashes_verified':len(rows),'candidate_branches':len(rows)*10,
        'actor_model_versions_content_hashes_verified':len(verified_models),
        'held_out_cities_used':[],'per_city':city,'teacher':teacher,
        'training_collection_wall_seconds':training['wall_seconds'],
        'training_episode_worker_seconds':sum(r['pilot_job_wall_seconds'] for r in training['results']),
        'validation_wall_seconds':validation['wall_seconds'],'validation_episode_worker_seconds':actual_val_seconds,
        'training_raw_window_seconds':sum(raw.values()),'training_projection_seconds':sum(projection.values()),
        'minimum_available_gib':training['minimum_available_gib'],
        'maximum_episode_peak_rss_gib':max(r['pilot_process_peak_rss_gib'] for r in training['results']+validation['results'])}
    write(root/'PILOT_ANALYSIS.json',analysis);write(root/'TRAINING_TIME_ESTIMATE.json',estimate)
    return {'analysis':analysis,'estimate':estimate}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.evidence),indent=2))
