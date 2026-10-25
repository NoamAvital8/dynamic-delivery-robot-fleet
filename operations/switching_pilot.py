"""Explicit, capped development pilot. Never calls the full training workflow."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_switching import LABEL_VERSION
from fleet_switching.backend import Backend, activate
from fleet_switching.collection import collect_episode, init_worker
from fleet_switching.config import Config, DEVELOPMENT, HELD_OUT, POLICIES, atomic, available_gib, identity, immutable, read, sha
from fleet_switching.execution import FitTask, numeric_environment, run_episodes
from fleet_switching.models import Predictor, fit
from fleet_switching.provenance import row_digest, science_id
from fleet_switching.reporting import FIXED_PREFIX, compare
from fleet_switching import replay
from fleet_switching.workflow import _fixed_models

PROFILES = ("ordinary", "charging_pressure")
VALIDATION_CITIES = frozenset(("new_delhi", "manhattan"))


class MeasuredBackend(Backend):
    def __init__(self, config, episode, scratch):
        started = time.monotonic()
        super().__init__(config, episode, scratch)
        self.initialization_seconds = time.monotonic() - started


def pilot_entry(pipe, stop, config, episode, family, science, frozen):
    import resource
    started = time.monotonic()
    try:
        init_worker()
        measured = {}
        def factory(*args):
            backend = MeasuredBackend(*args)
            measured['initialization_seconds'] = backend.initialization_seconds
            return backend
        result = collect_episode(config, episode, family, science, frozen,
            backend_factory=factory, stop_signal=stop,
            on_ready=lambda:pipe.send(("ready", None)))
        profile_path=Path(config.output)/'actors'/family/episode['split']/episode['id']/'pilot_profile.json'
        if profile_path.exists():profile=read(profile_path)
        else:
            profile={"pilot_job_wall_seconds":time.monotonic()-started,
                "pilot_process_peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                "pilot_initialization_seconds":measured.get('initialization_seconds')}
            atomic(profile_path,profile)
        result.update(profile)
        pipe.send(("result", result))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


def guard_scope(root):
    scope = read(root / "PILOT_LIMITS.json")
    if (set(scope["development_cities"]) != DEVELOPMENT or set(scope["held_out_cities"]) != HELD_OUT
            or scope["training_inputs_total"] != 16 or scope["training_actor_episodes_total"] != 32
            or scope["worker_limit"] > 8 or scope["maximum_collection_wall_seconds"] > 7200
            or scope["full_training_authorized"] is not False):
        raise ValueError("Pilot scope changed or could enable full training")
    start = root / "COLLECTION_STARTED.json"
    if start.exists() and time.time() - read(start)["unix_time"] > scope["maximum_collection_wall_seconds"]:
        raise TimeoutError("Pilot wall-clock budget reached; full training is not authorized")
    if (root / "run/datasets/test_manifest.json").exists():
        raise ValueError("A pilot must never create a final-test dataset")
    return scope


def config_for(base, root):
    return replace(Config.load(base), output=str(root / "run"), worker_limit=8,
        train_per_city=2, validation_per_city=1, update_every_states=200,
        fit_timeout_seconds=300., max_post_horizon_minutes=1440)


def pilot_science(config, root):
    return identity({"label_science":science_id(config), "driver":sha(__file__),
        "scope":read(root / "PILOT_LIMITS.json"), "pilot_config":asdict(config)})


def prepare(config, root):
    guard_scope(root)
    numeric_environment(1)
    activate(config)
    import numpy as np
    from create_multicity_suite import make_demand
    from create_nyc_reference_scenario import read_graphml_nodes
    from delivery_fleet.scenario_creator import ImportanceDistribution, NodeDemandProfile, ScenarioCreator
    from delivery_fleet.meta_policy.separation import workload_identity
    path = root / "PILOT_DATASET.json"
    science = pilot_science(config, root)
    if path.exists():
        data = read(path)
        if data["science_id"] != science:
            raise ValueError("Pilot sources/config changed; use a separate attempt")
        return data
    maps = [r for r in read(config.map_manifest)["maps"] if r["key"] in DEVELOPMENT]
    assert {r["key"] for r in maps} == DEVELOPMENT
    episodes, seen = [], set()
    for row in maps:
        assert row["role"] == "development" and row["key"] not in HELD_OUT
        graph = Path(row["path"])
        assert sha(graph) == row["sha256"]
        nodes, lat, lon = read_graphml_nodes(graph)
        cases = [("train", i, profile) for i, profile in enumerate(PROFILES)]
        if row["key"] in VALIDATION_CITIES:
            cases.append(("validation", 2, "surge"))
        for split, index, profile in cases:
            seed = int(identity(["bounded-pilot-20261025", row["key"], split, index])[:15], 16)
            rng = np.random.default_rng(seed)
            fleet = min(80, max(5, round(math.ceil(len(nodes)/500) * rng.uniform(.75,1.25))))
            demand, _ = make_demand(nodes, lat, lon, duration_hours=12, fleet_count=fleet, rng=rng)
            rates = demand.lambda_by_bucket.copy() * rng.uniform(.8,1.3)
            if profile == "charging_pressure": rates *= 1.5
            if profile == "surge": rates[4:7] *= 2.
            scenario = ScenarioCreator(importance_distribution=ImportanceDistribution()).create(
                row["key"], NodeDemandProfile(nodes, rates, bucket_minutes=60.), seed)
            workload = workload_identity(scenario)
            assert workload not in seen and scenario.orders
            seen.add(workload)
            key = f"{row['key']}_pilot_{split}_{index:03d}"
            scenario_path = root / "inputs" / split / (key + ".json")
            immutable(scenario_path, asdict(scenario))
            episodes.append({"id":key,"city":row["key"],"split":split,"seed":seed,
                "scenario":str(scenario_path),"scenario_sha256":sha(scenario_path),
                "workload_sha256":workload,"graph":str(graph),"graph_sha256":row["sha256"],
                "orders":len(scenario.orders),"fleet_count":fleet,"profile":profile,
                "initial_battery_fraction":float(rng.uniform(.4,.75)) if profile=="charging_pressure" else 1.,
                "prior_rates_per_hour":{str(k):fleet*.30*p for k,p in ((1,.8),(2,.15),(5,.05))}})
    assert sum(e["split"]=="train" for e in episodes)==16
    assert sum(e["split"]=="validation" for e in episodes)==2
    data = {"science_id":science,"scope":"pilot_only","episodes":episodes,
            "held_out_cities_used":[],"config":asdict(config),"driver_sha256":sha(__file__)}
    immutable(path, data)
    atomic(root / "PILOT_CONFIG.json", asdict(config))
    return data


def costs(predictor, features):
    import numpy as np
    p = predictor.payload
    x = (np.array([[features[k] for k in p["feature_names"]]]) - p["feature_mean"]) / p["feature_scale"]
    if p["kind"] == "xgboost":
        return np.array([head.inplace_predict(x)[0] for head in predictor.heads])
    for index, (w,b) in enumerate(zip(predictor.weights,predictor.biases)):
        x = x @ w + b
        if index<2: x=np.maximum(0,x)
    return x[0]*p["target_scale"]+p["target_mean"]


def fitting_gate(config, root):
    numeric_environment(4)
    if config.dependency_dir: sys.path.insert(0,config.dependency_dir)
    import numpy as np
    import xgboost
    from threadpoolctl import threadpool_limits, threadpool_info
    report = {}
    for family in ("xgboost","nn"):
        rows=[]
        for i in range(32):
            row={"id":str(i),"episode":"tiny_"+str(i//16),"city":"new_delhi","split":"train",
                "family":family,"label_version":LABEL_VERSION,"robots":2,"seed":i,
                "scenario_sha256":"independent_fit_fixture", "features":{"x":i/31.,"z":(i%3)/2.},
                "costs":{p:1+10*(j/9-i/31)**2 for j,p in enumerate(POLICIES)}}
            row["content_sha256"]=row_digest(row);rows.append(row)
        started=time.monotonic()
        with threadpool_limits(limits=4):
            payload=fit(rows,family,4)
            native=threadpool_info()
        path=root/'gates/fits'/(family+'.json');atomic(path,payload)
        first,loaded=Predictor(payload),Predictor.load(path)
        error=max(float(np.max(np.abs(costs(first,r['features'])-costs(loaded,r['features'])))) for r in rows)
        assert error<1e-8 and all(first.choose(r['features'])==loaded.choose(r['features']) for r in rows)
        report[family]={"rows":32,"fit_and_roundtrip_seconds":time.monotonic()-started,
            "maximum_prediction_roundtrip_error":error,"native_libraries":native}
    atomic(root/'gates/fitting_gate.json',report)
    return report


def wait_fit(task, root):
    while True:
        guard_scope(root)
        result=task.poll()
        if result is not None:return result
        time.sleep(.1)


def collect(config,root,data):
    guard_scope(root)
    if not (root/'COLLECTION_STARTED.json').exists():immutable(root/'COLLECTION_STARTED.json',{'unix_time':time.time()})
    science=data['science_id'];run=Path(config.output)
    episodes={e['id']:e for e in data['episodes'] if e['split']=='train'}
    families=('xgboost','nn')
    fitted={}
    for family in families:
        path=run/'models'/family/'bootstrap.json'
        immutable(path,{'kind':'bootstrap','family':family,'policy':'reactive_insertion',
            'policies':list(POLICIES),'label_version':LABEL_VERSION,'science_id':science})
        pointer=path.parent/'current.json'
        if not pointer.exists():atomic(pointer,{'path':str(path),'sha256':sha(path),'science_id':science,'training_states':0})
        fitted[family]=read(pointer)['training_states']
    ordered=sorted(episodes.values(),key=lambda e:(PROFILES.index(e['profile']),e['city']))
    jobs=[(e,f,None) for e in ordered for f in families]
    assert len(jobs)==32 and not {e['city'] for e,_,_ in jobs}&HELD_OUT
    task,last_scan=None,0.
    updates=[];minimum_available=available_gib();started=time.monotonic()
    def progress(results,running):
        nonlocal task,last_scan,minimum_available
        guard_scope(root)
        minimum_available=min(minimum_available,available_gib())
        if task is not None:
            result=task.poll()
            if result is not None:
                updates.append({'family':task.family,'states':result['training_states'],
                    'wall_seconds':time.monotonic()-task.started,'sha256':result['sha256']})
                fitted[task.family]=result['training_states'];task=None
        if time.monotonic()-last_scan<2:return
        last_scan=time.monotonic();stats=replay.statistics(run,science)
        atomic(root/'pilot_status.json',{'stage':'collect','complete':len(results),'expected':32,'running':running,
            'wall_seconds':time.monotonic()-started,'available_gib':available_gib(),'replay':stats,'fits':updates})
        if task is None:
            eligible=[f for f in families if stats.get(f,{}).get('count',0)>=fitted[f]+200]
            if eligible:
                family=max(eligible,key=lambda f:stats[f]['count']-fitted[f])
                task=FitTask(config,family,science,episodes,stats[family]['watermark'])
    try:
        results=run_episodes(config,jobs,science,progress,entry=pilot_entry)
        if task is not None:
            result=wait_fit(task,root)
            updates.append({'family':task.family,'states':result['training_states'],'wall_seconds':time.monotonic()-task.started,'sha256':result['sha256']});task=None
        stats=replay.statistics(run,science);final={}
        for family in families:
            task=FitTask(config,family,science,episodes,stats[family]['watermark'])
            final[family]=wait_fit(task,root)
            updates.append({'family':family,'states':final[family]['training_states'],'wall_seconds':time.monotonic()-task.started,'final':True})
            task=None
        report={'results':results,'models':final,'fit_updates':updates,'wall_seconds':time.monotonic()-started,
            'minimum_available_gib':minimum_available,'replay':stats,'scope':'pilot_only'}
        atomic(root/'PILOT_TRAINING_RESULTS.json',report)
        return report
    finally:
        if task is not None:task.stop()


def validate(config,root,data,training):
    guard_scope(root)
    baselines=_fixed_models(config,data['science_id'])
    episodes=[e for e in data['episodes'] if e['split']=='validation']
    jobs=[(e,f,training['models'][f]) for e in episodes for f in ('xgboost','nn')]
    jobs += [(e,FIXED_PREFIX+p,baselines[p]) for e in episodes for p in POLICIES]
    assert len(jobs)==24 and {e['city'] for e,_,_ in jobs}==VALIDATION_CITIES
    started=time.monotonic()
    results=run_episodes(config,jobs,data['science_id'],lambda *_:guard_scope(root),entry=pilot_entry)
    report=compare(results,episodes,expected_science=data['science_id'])
    report['wall_seconds']=time.monotonic()-started
    atomic(root/'PILOT_VALIDATION_RESULTS.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('fit-check','prepare','collect','validate','all'))
    parser.add_argument('--base-config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required; this command is pilot-only')
    root=Path(args.root).resolve();guard_scope(root)
    config=config_for(args.base_config,root)
    if config.dependency_dir:sys.path.insert(0,config.dependency_dir)
    try:
        if args.command in ('all','fit-check'):print(json.dumps(fitting_gate(config,root)),flush=True)
        if args.command=='fit-check':return
        data=prepare(config,root)
        if args.command=='prepare':return
        if args.command in ('all','collect'):training=collect(config,root,data)
        else:training=read(root/'PILOT_TRAINING_RESULTS.json')
        if args.command in ('all','validate'):validate(config,root,data,training)
        atomic(root/'pilot_status.json',{'stage':'complete','full_training_started':False,'held_out_cities_used':[]})
    except BaseException:
        atomic(root/'pilot_status.json',{'stage':'failed','error':traceback.format_exc(),'full_training_started':False,'held_out_cities_used':[]})
        raise


if __name__=='__main__':main()
