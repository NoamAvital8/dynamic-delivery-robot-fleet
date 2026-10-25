"""Bounded development-only lookahead diagnostic on existing pilot inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))

from switching_pilot import guard_scope, init_worker
from fleet_switching.backend import Backend
from fleet_switching.config import Config, HELD_OUT, POLICIES, atomic, read, sha
from fleet_switching.execution import run_episodes


def entry(pipe,stop,config,episode,policy,science,frozen):
    import resource
    started=time.monotonic()
    backend=None
    try:
        init_worker()
        backend=Backend(config,episode,Path(config.output)/'continuation_scratch'/episode['id']/policy)
        pipe.send(('ready',None))
        runtime=backend.runtime
        runtime.set_policy('reactive_insertion');runtime.advance(360.)
        before=runtime.state_digest();paid=runtime.loss
        runtime.set_policy(policy);runtime.advance(375.)
        boundary=runtime.snapshot()
        projection,info=backend.project()
        projected=runtime.loss-paid+projection.loss
        runtime.restore(boundary);runtime.set_policy('reactive_insertion')
        while runtime.outstanding():
            if stop.is_set():raise InterruptedError('Pilot diagnostic stopped')
            if runtime.now>config.duration_hours*60+config.max_post_horizon_minutes:
                raise RuntimeError('Diagnostic continuation exceeded the completion bound')
            runtime.advance(runtime.now+15.)
        result={'city':episode['city'],'episode':episode['id'],'candidate_policy':policy,
            'before_physical_digest':before,'projected_cost':projected,
            'remaining_episode_reference_cost':runtime.loss-paid,
            'reference_continuation':'Reactive Insertion after the candidate 15-minute window; actual remaining arrivals retained',
            'projection_diagnostics':info,'wall_seconds':time.monotonic()-started,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
        backend.close();backend=None
        pipe.send(('result',result))
    except BaseException:pipe.send(('error',traceback.format_exc()))
    finally:
        if backend is not None:backend.close()
        pipe.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required; no final-test or full-training path exists')
    root=Path(args.root).resolve();guard_scope(root)
    assert read(root/'pilot_status.json')['stage']=='complete'
    data=read(root/'PILOT_DATASET.json');config=Config.load(root/'PILOT_CONFIG.json')
    episodes=[e for e in data['episodes'] if e['split']=='train' and e['profile']=='ordinary' and e['city'] in {'new_delhi','manhattan'}]
    assert len(episodes)==2 and not {e['city'] for e in episodes}&HELD_OUT
    jobs=[(e,p,None) for e in episodes for p in POLICIES]
    results=run_episodes(config,jobs,data['science_id'],lambda *_:guard_scope(root),entry=entry)
    summary={}
    for city in ['new_delhi','manhattan']:
        rows=[r for r in results if r['city']==city]
        assert len(rows)==10 and len({r['before_physical_digest'] for r in rows})==1
        projected={r['candidate_policy']:r['projected_cost'] for r in rows}
        reference={r['candidate_policy']:r['remaining_episode_reference_cost'] for r in rows}
        low=min(projected.values());best=[p for p in POLICIES if abs(projected[p]-low)<1e-7]
        selected='reactive_insertion' if 'reactive_insertion' in best else best[0]
        regret=reference[selected]-min(reference.values())
        summary[city]={'projected_best_set':best,'tie_break':'incumbent Reactive Insertion if tied, otherwise portfolio order',
            'projected_best_selected':selected,'reference_best_set':[p for p in POLICIES if abs(reference[p]-min(reference.values()))<1e-7],
            'reference_regret_of_projected_choice':regret,
            'reference_regret_percent':100*regret/max(1.,min(reference.values())),
            'projected_cost_range':max(projected.values())-min(projected.values()),
            'reference_cost_range':max(reference.values())-min(reference.values())}
    atomic(root/'CONTINUATION_RANK_DIAGNOSTICS.json',{'scope':'development diagnostic only; not training labels or final-test evaluation',
        'driver_sha256':sha(__file__),'summary':summary,'results':results,'held_out_cities_used':[]})
    print(summary,flush=True)


if __name__=='__main__':main()
