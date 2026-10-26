"""Analyze existing training labels and development MLE counters by time.

No runtime is constructed or advanced; no model is fitted and no test outcome is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fleet_switching.config import HELD_OUT, POLICIES, identity, read, sha
from fleet_switching.provenance import validate_row
from analyze_switching_policy_optimality import add, finish, new_group

TARGET='full_mle_reservation'
TOLERANCE=1e-7


class OpaqueEvent:
    def __new__(cls,*args,**kwargs):return object.__new__(cls)
    def __init__(self,*args,**kwargs):pass
    def __setstate__(self,state):self.raw_state=state


class MetadataReader(pickle.Unpickler):
    """Replace external graph/model references and dynamic events with inert values."""
    opaque={}
    def persistent_load(self,key):return None
    def find_class(self,module,name):
        if module=='delivery_fleet._adaptive_event_engine':
            return self.opaque.setdefault(name,type(name,(OpaqueEvent,),{}))
        return super().find_class(module,name)


def activation_audit(root):
    cfg=read(root/'configs/small_city_switching_v2.json');backend=Path(cfg['backend_root'])
    for p in [backend/'src',backend/'scripts',backend/'operations',Path(cfg['dependency_dir'])]:
        sys.path.insert(0,str(p))
    episodes={e['id']:e for e in read(root/'datasets/development_manifest.json')['episodes'] if e['split']=='validation'}
    records=[];reasons=Counter();stats=Counter();sampled=defaultdict(Counter)
    for folder in sorted((root/'actors/fixed__full_mle_reservation/validation').iterdir()):
        if not folder.is_dir():continue
        source=episodes[folder.name];receipt=read(folder/'complete.json')
        if receipt['delivered']!=source['orders']:raise ValueError('Incomplete development episode')
        cp=read(folder/'checkpoint.json');p=Path(cp['state_path'])
        if sha(p)!=cp['state_sha256']:raise ValueError('Checkpoint changed')
        with p.open('rb') as stream:state=MetadataReader(stream).load()['state']
        controller=state['_meta_controllers']['mle']
        if controller.stats['arrivals']!=source['orders']:raise ValueError('Arrival observation count mismatch')
        record={'episode':source['id'],'city':source['city'],'profile':source['profile'],
            'checkpoint_sha256':cp['state_sha256'],'epoch_minutes':controller.config.epoch_min,
            'stats':dict(controller.stats),'reason_counts':dict(controller.reason_counts),
            'first_activation_min':controller.first_activation_min}
        records.append(record);reasons.update(controller.reason_counts);stats.update(controller.stats)
        for p in sorted((folder/'rows').glob('*.json')):
            row=validate_row(read(p));t=row['time_min'];hour=str(int(t//60)) if t<720 else 'drain'
            sampled[hour]['states']+=1
            sampled[hour]['active_reservation_states']+=int(row['features']['reservation.active']>0)
    if len(records)!=200:raise ValueError('Expected all 200 fixed-MLE development episodes')
    return {'episodes':len(records),'episodes_ever_activated':sum(r['first_activation_min'] is not None for r in records),
        'total_activations':stats['activations'],'total_candidates_scored':stats['candidates_scored'],
        'epoch_lengths_minutes':sorted({r['epoch_minutes'] for r in records}),
        'reason_counts':dict(reasons),'sampled_states_by_hour':{k:dict(v) for k,v in sampled.items()},
        'records':records,'method':'Metadata-only decode of preserved terminal checkpoints; persistent references replaced with None and dynamic event objects made inert. No FleetRuntime construction, restore or advance.'}


def analyze(root):
    manifest=read(root/'datasets/development_manifest.json');episodes={e['id']:e for e in manifest['episodes'] if e['split']=='train'}
    frozen=read(root/'frozen_models.json');groups=defaultdict(new_group);ids=defaultdict(list)
    alternatives=defaultdict(lambda:defaultdict(float))
    with sqlite3.connect((root/'replay.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        sizes={(f,e):n for f,e,n in db.execute('SELECT family,episode,COUNT(*) FROM rows GROUP BY family,episode')}
        for index,(family,key,digest,payload) in enumerate(db.execute('SELECT family,id,digest,payload FROM rows ORDER BY seq'),1):
            row=validate_row(json.loads(payload),episodes,frozen['science_id'],family)
            if row['id']!=key or row['content_sha256']!=digest or row['city'] in HELD_OUT:raise ValueError('Replay identity mismatch')
            if set(row['costs'])!=set(POLICIES):raise ValueError('Incomplete policy costs')
            t=row['time_min'];hour=str(int(t//60)) if t<720 else 'drain'
            low=min(row['costs'].values());winners={p for p,c in row['costs'].items() if c-low<=TOLERANCE}
            weight=1/sizes[family,row['episode']]
            keys=[('hour','combined',hour),('hour',family,hour),('city',row['city'],hour),('profile',row['profile'],hour)]
            if t<720:keys += [('three_hour','combined',str(int(t//180))),('phase','combined',str(int(t%60)))]
            for k in keys:add(groups[k],row,TARGET,winners,weight)
            for p in POLICIES:alternatives[hour][p]+=(row['costs'][p]-row['costs'][TARGET])/row['robots']
            ids[family].append((key,digest))
            if index%20000==0:print('validated training rows',index,flush=True)
    checks={}
    for f,values in ids.items():
        pointer=frozen['models'][f];model=read(pointer['path'])
        if sha(pointer['path'])!=pointer['sha256'] or identity(sorted(values))!=model['training_records_sha256']:
            raise ValueError('Training rows differ from final fit')
        checks[f]={'states':len(values),'final_fit_records_hash_matches':True}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_script_sha256':sha(__file__),
        'frequency_helper_sha256':sha(Path(__file__).with_name('analyze_switching_policy_optimality.py')),
        'absolute_tie_tolerance':TOLERANCE,'model_checks':checks,
        'groups':{kind:{name:{bucket:finish(v) for (k,n,bucket),v in groups.items() if k==kind and n==name}
                 for name in sorted({k[1] for k in groups if k[0]==kind})}
                 for kind in ['hour','three_hour','phase','city','profile']},
        'mean_alternative_minus_full_mle_per_robot':{h:{p:value/groups['hour','combined',h]['states'] for p,value in costs.items()} for h,costs in alternatives.items()},
        'development_fixed_mle_activation':activation_audit(root),
        'test_outcomes_read':False,'simulations_started':0,'models_fitted':0,
        'caveats':['Saved labels are state-conditional 15-minute-plus-projection targets, not remaining-episode values',
            'Time-of-day, demand profile and fleet state are confounded; the curve alone does not establish an MLE-accuracy mechanism',
            'Evaluating a cutoff requires replaying full episodes under the prefix-then-MLE rule; saved actor rows cannot be spliced into a valid new trajectory']}
    return result


def markdown(v):
    lines=['# Full MLE: time-of-day diagnosis','','Read-only training-label and development-controller analysis. No final-test outcome was used and no new simulation or fitting was run.','','## Training target by simulation hour','','Rows use the start of each 15-minute decision window. Drain after hour 12 is separate. Best includes ties within 1e-7 raw loss units.','','| Simulation interval | Decisions | Best/tied % | Unique best % | All ten tied % | Best/tied %, excluding all-ten ties | Mean Full MLE regret / robot |','|---|---:|---:|---:|---:|---:|---:|']
    values=v['groups']['hour']['combined']
    for h in [str(i) for i in range(12)]+['drain']:
        if h not in values:continue
        g=values[h];label=f'{h}–{int(h)+1} h' if h!='drain' else 'After 12 h (drain)'
        lines.append(f"| {label} | {g['states']:,} | {g['policy_best_including_ties_percent']:.2f} | {g['policy_unique_best_percent']:.2f} | {g['all_policies_tied_percent']:.2f} | {g['policy_best_percent_excluding_all_policy_ties']:.2f} | {g['mean_policy_projected_regret_per_robot']:.5f} |")
    a=v['development_fixed_mle_activation'];lines+=['','## MLE activation audit','',f"Across all {a['episodes']} fixed-Full-MLE development validation episodes: **{a['episodes_ever_activated']} episodes ever activated MLE reservations**, {a['total_activations']} activations, and {a['total_candidates_scored']} candidates scored.",'','The exact cumulative controller counters were decoded from terminal checkpoints without constructing or advancing a runtime. Saved decision features give a separate sampled-state cross-check.','','Rate counts and request templates reset on fixed 60-minute epochs. Arrivals are observed under every active benchmark, including while MLE is inactive. The epoch confidence error allocation also becomes stricter with epoch number. Thus later clock time does not mean a larger cumulative MLE rate sample.','','Reason counters:',*['- '+reason+': '+str(n) for reason,n in sorted(a['reason_counts'].items())],'','## Interpretation and possible cutoff experiment','','A prefix policy can collect arrivals before switching because the controller is already observed in the background. However, choosing a cutoff must use development data and complete episode loss, including drain. The original final-test results must remain a report of the original frozen experiment.','','Saved counterfactual costs cannot simply be summed or spliced to evaluate a new schedule: switching changes routes, batteries, queues and later states. A genuine cutoff comparison needs fresh full-episode runs in a separate development output, preserving commitments at the switch.','','The JSON report includes per-family, city, profile, three-hour and within-hour phase breakdowns plus per-policy cost differences. Any time trend here is descriptive and cannot establish that MLE accuracy caused it.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    v=analyze(args.campaign_root);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'MLE_BY_TIME.json').write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
    (args.output/'MLE_BY_TIME.md').write_bytes(markdown(v).encode())
    print(json.dumps({'three_hour':v['groups']['three_hour']['combined'],'phase':v['groups']['phase']['combined'],
        'activation':{k:x for k,x in v['development_fixed_mle_activation'].items() if k not in ['records','sampled_states_by_hour']}},indent=2))
