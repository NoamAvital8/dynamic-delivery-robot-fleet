"""Offline comparison of fixed benchmark test outcomes only; no simulations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fleet_switching.config import HELD_OUT, POLICIES, identity, read, sha

NAMES={'full_mle_reservation':'Full MLE Reservation','full_coordinated_idle':'Full Coordinated Idle',
    'queue_mle_reservation':'Queue MLE Reservation','full_no_nn':'Full without NN',
    'full_queue_aware':'Full Queue Aware','full':'Full','reactive_insertion':'Reactive Insertion',
    'full_no_idle':'Full without Idle','paper_sa_adapted':'Paper SA Adapted','myopic_ab':'Myopic AB'}


def mean(values):return sum(values)/len(values)


def quantile(values,p):
    a=sorted(values);position=p*(len(a)-1);lo=int(position);hi=min(lo+1,len(a)-1)
    return a[lo]+(position-lo)*(a[hi]-a[lo])


def analyze(report_path,manifest_path):
    source=read(report_path);manifest=read(manifest_path)
    assert source['freeze_id']==manifest['freeze_id']
    episodes={e['id']:e for e in manifest['episodes']}
    assert len(episodes)==200 and Counter(e['city'] for e in episodes.values())=={c:50 for c in HELD_OUT}
    families={'fixed__'+p for p in POLICIES}
    rows=[r for r in source['results'] if r['family'] in families]
    indexed={}
    for r in rows:
        e=episodes[r['episode']];key=(r['episode'],r['family'])
        assert key not in indexed and e['split']==r['split']=='test' and r['status']=='complete'
        assert r['episode_identity']==identity(e)
        assert r['science_id']==manifest['science_id']
        assert r['receipt_sha256']==identity({k:v for k,v in r.items() if k!='receipt_sha256'})
        for k in ['city','scenario_sha256','graph_sha256']:assert r[k]==e[k]
        assert r['orders']==r['delivered']==e['orders'] and r['robots']==e['fleet_count']
        assert math.isfinite(r['loss']) and r['loss']>=0
        indexed[key]=r
    assert set(indexed)=={(e,f) for e in episodes for f in families} and len(rows)==2000
    cities=sorted(HELD_OUT);by_city={c:sorted(e['id'] for e in episodes.values() if e['city']==c) for c in cities}
    reference=source['development_selected_fixed_policy'];ref_family='fixed__'+reference
    rng=random.Random(20261026)
    samples={c:[[rng.randrange(50) for _ in range(50)] for _ in range(5000)] for c in cities}
    results={}
    for policy in POLICIES:
        family='fixed__'+policy;subset=[indexed[e,family] for e in sorted(episodes)]
        per_city={c:mean([indexed[e,family]['loss']/indexed[e,family]['robots'] for e in by_city[c]]) for c in cities}
        score=mean(list(per_city.values()))
        assert abs(score-source['scores'][family]['equal_city_mean_loss_per_robot'])<1e-8
        differences={c:[(indexed[e,family]['loss']-indexed[e,ref_family]['loss'])/indexed[e,family]['robots'] for e in by_city[c]] for c in cities}
        draws=[mean([mean([differences[c][j] for j in samples[c][i]]) for c in cities]) for i in range(5000)]
        wins=ties=unique=best_including=0
        for e in episodes:
            loss=indexed[e,family]['loss'];baseline=indexed[e,ref_family]['loss'];low=min(indexed[e,f]['loss'] for f in families)
            winners={f for f in families if indexed[e,f]['loss']-low<=1e-7}
            best_including+=int(family in winners);unique+=int(winners=={family})
            wins+=int(loss<baseline-1e-7);ties+=int(abs(loss-baseline)<=1e-7)
        total_orders=sum(r['orders'] for r in subset)
        on_time=sum(r['on_time'] for r in subset) if all(r['on_time'] is not None for r in subset) else None
        results[policy]={'name':NAMES[policy],'episodes':200,'primary_loss':score,
            'per_city_loss_per_robot':per_city,'total_loss':sum(r['loss'] for r in subset),
            'delivered_orders':total_orders,'on_time_deliveries':on_time,
            'on_time_percent':100*on_time/total_orders if on_time is not None else None,
            'scenario_best_including_ties':best_including,'scenario_unique_best':unique,
            'wins_vs_development_selected_reference':wins,'ties_vs_development_selected_reference':ties,
            'losses_vs_development_selected_reference':200-wins-ties,
            'paired_primary_difference_vs_reference':score-source['scores'][ref_family]['equal_city_mean_loss_per_robot'],
            'conditional_stratified_episode_bootstrap_95pct_difference_interval':[quantile(draws,.025),quantile(draws,.975)]}
    ranking=sorted(POLICIES,key=lambda p:results[p]['primary_loss']);best=ranking[0];best_value=results[best]['primary_loss']
    for p in ranking:results[p]['percent_higher_loss_vs_best_test']=100*(results[p]['primary_loss']/best_value-1)
    winners={c:min(POLICIES,key=lambda p:results[p]['per_city_loss_per_robot'][c]) for c in cities}
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_source_sha256':sha(__file__),
        'input_report_sha256':sha(report_path),'input_manifest_sha256':sha(manifest_path),
        'scope':'Fixed benchmarks, final test only; no meta-model outcome used',
        'test_scenarios':200,'matched_benchmark_outcomes':2000,'scenarios_per_city':50,
        'metric':'Equal-city mean of scenario loss per robot; lower is better',
        'development_selected_reference_policy':reference,'best_observed_fixed_test_policy':best,
        'ranking':ranking,'per_city_winners':winners,'benchmarks':results,
        'tie_tolerance_raw_loss':1e-7,'bootstrap_draws':5000,
        'interval_scope':'Episode resampling within each of the four observed cities, conditional on those cities; unadjusted for multiple comparisons',
        'simulations_started':0,'models_fitted':0}


def markdown(v):
    lines=['# Fixed benchmarks: final-test comparison','','Only the ten fixed benchmarks are included. No meta-model outcomes, training scores or validation scores enter this comparison.','','The same **200 test scenarios** (50 each in Cambridge UK, Ghent, Salamanca and Lucca) give **2,000 complete matched benchmark outcomes**. Input identities, outcome receipt hashes, fleets and delivery completion were verified.','','Primary metric: **equal-city mean loss per robot**, lower is better. These are deadline/importance-weighted loss values.','','| Rank | Benchmark | Primary loss | Higher loss versus best | On-time deliveries | Best/tied scenarios |','|---|---|---:|---:|---:|---:|']
    for i,p in enumerate(v['ranking'],1):
        r=v['benchmarks'][p];ontime=f"{r['on_time_percent']:.2f}%" if r['on_time_percent'] is not None else 'unavailable'
        lines.append(f"| {i} | {r['name']} | {r['primary_loss']:.5f} | {r['percent_higher_loss_vs_best_test']:+.2f}% | {ontime} | {r['scenario_best_including_ties']}/200 |")
    lines+=['','Best/tied scenario counts overlap when several fixed policies tie. They do not sum to 200. All admitted orders completed; on-time percentage is a separate metric from weighted loss.','','## Per-city primary scores','','| Benchmark | Cambridge UK | Ghent | Salamanca | Lucca |','|---|---:|---:|---:|---:|']
    for p in v['ranking']:
        r=v['benchmarks'][p];lines.append('| '+r['name']+' | '+' | '.join(f"{r['per_city_loss_per_robot'][c]:.4f}" for c in ['cambridge_uk','ghent','salamanca','lucca'])+' |')
    lines+=['','## Per-city winners','']
    for c,p in v['per_city_winners'].items():lines.append(f"- {c}: **{v['benchmarks'][p]['name']}**, {v['benchmarks'][p]['per_city_loss_per_robot'][c]:.5f}.")
    lines+=['','## Paired uncertainty against the preselected reference','','Full MLE Reservation was selected on development before test. Positive differences favor that reference. The JSON contains 5,000-draw stratified paired bootstrap intervals, conditional on the four observed cities and unadjusted for multiple comparisons. The table reports observed rankings, not universal geographic superiority.','','This compares composite benchmark implementations. The separate reservation-activation audit is relevant when attributing any benefit to the MLE component itself. No new simulation or fitting was performed for this report.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.report,args.manifest);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'BENCHMARKS_TEST_ONLY.json').write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    (args.output/'BENCHMARKS_TEST_ONLY.md').write_bytes(markdown(result).encode())
    print(json.dumps({'ranking':[(result['benchmarks'][p]['name'],result['benchmarks'][p]['primary_loss'],result['benchmarks'][p]['percent_higher_loss_vs_best_test'],result['benchmarks'][p]['scenario_best_including_ties']) for p in result['ranking']],'city_winners':result['per_city_winners']},indent=2))
