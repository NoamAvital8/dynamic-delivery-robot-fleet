"""Read-only replay analysis of teacher-optimal actions; no simulation or fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fleet_switching.config import HELD_OUT, POLICIES, identity, read, sha
from fleet_switching.provenance import validate_row


def new_group():
    return Counter()


def add(group,row,policy,winners,weight):
    good=policy in winners;unique=good and len(winners)==1
    all_tied=len(winners)==len(POLICIES);chosen=row['intent']['chosen_policy']
    group['states']+=1;group['policy_best_including_ties']+=int(good)
    group['policy_unique_best']+=int(unique)
    group['policy_shared_best']+=int(good and not unique)
    group['all_policies_tied']+=int(all_tied)
    group['any_unique_winner']+=int(len(winners)==1)
    group['actor_chose_policy']+=int(chosen==policy)
    group['actor_chose_any_optimal_policy']+=int(chosen in winners)
    group['policy_best_but_actor_worse']+=int(good and chosen not in winners)
    group['episode_weight_sum']+=weight
    group['episode_weighted_policy_best_sum']+=weight*good
    group['episode_weighted_policy_unique_best_sum']+=weight*unique
    group['policy_projected_regret_per_robot_sum']+=(row['costs'][policy]-min(row['costs'].values()))/row['robots']
    group['actor_projected_regret_per_robot_sum']+=(row['costs'][chosen]-min(row['costs'].values()))/row['robots']


def finish(group):
    v=dict(group);n=v['states']
    for key in ['policy_best_including_ties','policy_unique_best','policy_shared_best',
                'all_policies_tied','actor_chose_policy','actor_chose_any_optimal_policy',
                'policy_best_but_actor_worse']:
        v[key+'_percent']=100*v.get(key,0)/n
    informative=n-v.get('all_policies_tied',0)
    v['states_with_different_policy_scores']=informative
    v['policy_best_excluding_all_policy_ties']=v.get('policy_best_including_ties',0)-v.get('all_policies_tied',0)
    v['policy_best_percent_excluding_all_policy_ties']=100*v['policy_best_excluding_all_policy_ties']/informative if informative else None
    v['episode_weighted_policy_best_percent']=100*v.get('episode_weighted_policy_best_sum',0)/v['episode_weight_sum']
    v['episode_weighted_policy_unique_best_percent']=100*v.get('episode_weighted_policy_unique_best_sum',0)/v['episode_weight_sum']
    v['mean_policy_projected_regret_per_robot']=v['policy_projected_regret_per_robot_sum']/n
    v['mean_actor_projected_regret_per_robot']=v['actor_projected_regret_per_robot_sum']/n
    return v


def analyze(root,policy,tolerance):
    data=read(root/'datasets/development_manifest.json')
    episodes={e['id']:e for e in data['episodes'] if e['split']=='train'}
    frozen=read(root/'frozen_models.json');science=frozen['science_id']
    groups=defaultdict(new_group);policy_best=defaultdict(Counter);unique_best=defaultdict(Counter)
    row_ids=defaultdict(list);sensitivities=defaultdict(Counter)
    path=root/'replay.sqlite'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        counts={(family,episode):n for family,episode,n in db.execute('SELECT family,episode,COUNT(*) FROM rows GROUP BY family,episode')}
        for index,(family,key,digest,payload) in enumerate(db.execute('SELECT family,id,digest,payload FROM rows ORDER BY seq'),1):
            row=validate_row(json.loads(payload),episodes,science,family)
            if row['id']!=key or row['content_sha256']!=digest:raise ValueError('Replay catalog mismatch')
            if row['city'] in HELD_OUT or row['split']!='train':raise ValueError('Non-training row')
            if set(row['costs'])!=set(POLICIES) or any(not math.isfinite(c) or c<0 for c in row['costs'].values()):
                raise ValueError('Invalid teacher costs')
            low=min(row['costs'].values());winners={p for p,c in row['costs'].items() if c-low<=tolerance}
            weight=1/counts[family,row['episode']]
            for group in [('overall','combined'),('family',family),('city',family,row['city']),
                          ('profile',family,row['profile']),('actor_kind',family,'bootstrap' if row['intent']['model_kind']=='bootstrap' else 'learned')]:
                add(groups[group],row,policy,winners,weight)
            for p in winners:policy_best[family][p]+=1
            if len(winners)==1:unique_best[family][next(iter(winners))]+=1
            gap=row['costs'][policy]-low
            for eps in [0.,1e-8,1e-7,1e-6]:sensitivities[family][str(eps)]+=int(gap<=eps)
            row_ids[family].append((key,digest))
            if index%20000==0:print('validated training states',index,flush=True)
    model_checks={}
    for family,values in row_ids.items():
        pointer=frozen['models'][family];model=read(pointer['path'])
        if sha(pointer['path'])!=pointer['sha256']:raise ValueError('Frozen model changed')
        if len(values)!=model['training_states'] or identity(sorted(values))!=model['training_records_sha256']:
            raise ValueError('Rows differ from those used by final model: '+family)
        model_checks[family]={'states':len(values),'frozen_model_sha256':pointer['sha256'],
            'training_records_hash_matches_final_fit':True,'training_episodes':len(model['training_episodes'])}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_script_sha256':sha(__file__),
        'policy':policy,'absolute_tie_tolerance':tolerance,'source_replay':str(path),'science_id':science,
        'target_definition':'Loss realized during the next 15-minute branch plus projected completion loss of released unfinished orders, including known charging queues',
        'interpretation':'Optimal for the saved training target on each actor-visited state, not proof of best full remaining-episode action',
        'models':model_checks,'overall':finish(groups['overall','combined']),
        'families':{f:finish(groups['family',f]) for f in sorted(row_ids)},
        'cities':{f:{c:finish(v) for (kind,g,c),v in ((k,v) for k,v in groups.items() if k[0]=='city') if g==f} for f in sorted(row_ids)},
        'profiles':{f:{c:finish(v) for (kind,g,c),v in ((k,v) for k,v in groups.items() if k[0]=='profile') if g==f} for f in sorted(row_ids)},
        'actor_kinds':{f:{c:finish(v) for (kind,g,c),v in ((k,v) for k,v in groups.items() if k[0]=='actor_kind') if g==f} for f in sorted(row_ids)},
        'all_policy_best_counts':{f:dict(policy_best[f]) for f in sorted(row_ids)},
        'all_policy_unique_best_counts':{f:dict(unique_best[f]) for f in sorted(row_ids)},
        'tie_tolerance_sensitivity_best_counts':{f:dict(sensitivities[f]) for f in sorted(row_ids)},
        'simulations_started':0,'models_fitted':0,'test_outcomes_used':False}
    return report


def markdown(report):
    lines=['# Full MLE Reservation: training decision optimality','','Analysis of immutable training labels only; no new simulations or fitting. Counts include bootstrap and learned-actor decisions.','','A policy is best when its saved projected cost is within `1e-7` absolute loss units of the minimum among all ten policies.','','| Trajectory family | States | Best including ties | Unique best | Shared best | All ten tied |','|---|---:|---:|---:|---:|---:|']
    for name,v in [*report['families'].items(),('combined',report['overall'])]:
        def cell(key):return f"{v.get(key,0):,} ({v[key+'_percent']:.2f}%)"
        lines.append(f"| {name} | {v['states']:,} | {cell('policy_best_including_ties')} | {cell('policy_unique_best')} | {cell('policy_shared_best')} | {cell('all_policies_tied')} |")
    lines+=['','## Excluding states where all policies tied','','| Family | States with different scores | Full MLE among best | Episode-weighted best rate, all states | Actor actually chose Full MLE |','|---|---:|---:|---:|---:|']
    for f,v in report['families'].items():lines.append(f"| {f} | {v['states_with_different_policy_scores']:,} | {v['policy_best_excluding_all_policy_ties']:,} ({v['policy_best_percent_excluding_all_policy_ties']:.2f}%) | {v['episode_weighted_policy_best_percent']:.2f}% | {v['actor_chose_policy']:,} ({v['actor_chose_policy_percent']:.2f}%) |")
    lines+=['','XGBoost and NN visited different states on their own trajectories. Pooled counts are state visits, not independent scenarios. Episode-weighted percentages give each trajectory equal total weight, matching the fitting weight scheme.','','Every row content hash and input provenance was checked. The complete row-ID/content-hash set for each family exactly matches the final frozen model\'s `training_records_sha256` and state count.','','**Meaning of right:** lowest 15-minute-plus-projected-completion training target. It omits future demand and terminal fleet value; this is not guaranteed full-episode optimality. Ties do not imply a unique correct action.','','The JSON report includes per-city/profile and bootstrap/learned breakdowns, all-policy frequencies, tolerance sensitivity and projected regret diagnostics. It reads training rows only and uses no validation/test outcomes.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--policy',choices=POLICIES,default='full_mle_reservation')
    parser.add_argument('--tie-tolerance',type=float,default=1e-7);args=parser.parse_args()
    if not math.isfinite(args.tie_tolerance) or args.tie_tolerance<0:parser.error('Invalid tolerance')
    result=analyze(args.campaign_root,args.policy,args.tie_tolerance);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'POLICY_OPTIMALITY.json').write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    (args.output/'POLICY_OPTIMALITY.md').write_bytes(markdown(result).encode())
    print(json.dumps({'families':result['families'],'overall':result['overall'],'model_checks':result['models']},indent=2))
