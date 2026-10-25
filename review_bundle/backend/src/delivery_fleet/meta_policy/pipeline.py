"""Resumable offline data collection, validation and causal selector evaluation.

Parallelize independent episodes, not nested policy/idle/BLAS workers. Each
worker retains one map/index and simulator while evaluating its counterfactual
branches sequentially. This avoids serializing huge graphs per decision and
can use up to 59 episode workers plus one coordinator under the 60-process cap.

Modes:
  collect   train + validation episodes; counterfactual labels on sampled steps
  validate  a frozen selector on its own held-out validation episodes (online)
  evaluate  a frozen selector on test episodes never used for fitting/validation
"""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
import traceback

from . import POLICIES
from .artifacts import (RunLock, atomic_bytes, atomic_json, digest, identity, available_gib,
                        peak_rss_gib, process_identity, worker_count, runtime_versions)

ROOT = Path(__file__).resolve().parents[3]
MODES = ("collect", "validate", "evaluate")
SPLITS_BY_MODE = {"collect": {"train", "validation"}, "validate": {"validation"}, "evaluate": {"test"}}
DECIDE_UNTIL = ("arrival_horizon", "all_delivered")
MAX_POST_HORIZON_MIN = 30 * 24 * 60.  # safety bound on deciding after the last arrival


def load_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1:
        raise ValueError("expected meta training manifest version 1")
    policies = tuple(value.get("policies", POLICIES))
    if not policies or len(set(policies)) != len(policies) or set(policies) - set(POLICIES):
        raise ValueError("invalid or duplicated candidate policies")
    windows = tuple(float(v) for v in value.get("holding_minutes", [15]))
    horizon = float(value.get("scoring_horizon_minutes", 60))
    default = float(value.get("default_holding_minutes", 15))
    if (not windows or len(set(windows)) != len(windows) or default not in windows
            or not all(math.isfinite(v) and v > 0 for v in windows)
            or not math.isfinite(horizon) or horizon < max(windows)):
        raise ValueError("invalid holding/scoring horizons")
    for name in ("continuation_policy", "initial_policy"):
        if value.get(name, "reactive_insertion") not in POLICIES:
            raise ValueError(f"invalid {name}")
    exploration = dict(value.get("exploration", {}))
    exploration = {"stickiness": float(exploration.get("stickiness", 0.)),
                   "fixed_episode_fraction": float(exploration.get("fixed_episode_fraction", 0.)),
                   "policies": tuple(exploration.get("policies", policies))}
    if (not 0 <= exploration["stickiness"] < 1 or not 0 <= exploration["fixed_episode_fraction"] <= 1
            or not exploration["policies"] or set(exploration["policies"]) - set(policies)
            or len(set(exploration["policies"])) != len(exploration["policies"])):
        raise ValueError("invalid exploration settings")
    label_probability = float(value.get("label_step_probability", 1.))
    if not 0 < label_probability <= 1:
        raise ValueError("label_step_probability must lie in (0, 1]")
    if value.get("decide_until", "arrival_horizon") not in DECIDE_UNTIL:
        raise ValueError(f"decide_until must be one of {DECIDE_UNTIL}")
    from .runtime import controller_settings
    def resolve(p):
        return str((path.parent / p).resolve())
    for name in ("spatial_model", "paper_model"):
        value[name] = resolve(value[name])
    separation = value.get("scenario_separation", {})
    for record in [*separation.get("reserved", []), *separation.get("reference_suites", [])]:
        record["path"] = resolve(record["path"])
    seen = set()
    for episode in value["episodes"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", episode["id"]) or episode["id"] in seen:
            raise ValueError("unsafe or duplicate episode id")
        seen.add(episode["id"])
        if episode["split"] not in {"train", "validation", "test"}:
            raise ValueError("episodes require an explicit train/validation/test split")
        for name in ("graph", "scenario"):
            episode[name] = resolve(episode[name])
        rates = episode["prior_rates_per_hour"]
        if set(map(float, rates)) != {1., 2., 5.} or any(
                not math.isfinite(float(v)) or float(v) <= 0 for v in rates.values()):
            raise ValueError("supply positive training-only priors for importance 1, 2, 5")
    value.update(policies=policies, holding_minutes=windows,
                 default_holding_minutes=default, scoring_horizon_minutes=horizon,
                 continuation_policy=value.get("continuation_policy", "reactive_insertion"),
                 initial_policy=value.get("initial_policy", "reactive_insertion"),
                 exploration=exploration, label_step_probability=label_probability,
                 decide_until=value.get("decide_until", "arrival_horizon"),
                 shortlist_k=int(value.get("shortlist_k", 10)),
                 controller_settings=controller_settings(value.get("controller_settings")))
    if not value["episodes"]:
        raise ValueError("no episodes in manifest")
    return value


# --- identities -------------------------------------------------------------
# Labels depend on the simulator, the policy programs and the label code, not on
# orchestration, learning or unrelated scripts. Editing learning.py, this file or
# an unrelated benchmark script must not invalidate weeks of collected labels.

def _module_file(parts):
    if not parts:
        return None
    if len(parts) == 1 and (ROOT / "scripts" / f"{parts[0]}.py").is_file():
        return ROOT / "scripts" / f"{parts[0]}.py"
    if len(parts) == 1 and (ROOT / "operations" / f"{parts[0]}.py").is_file():
        return ROOT / "operations" / f"{parts[0]}.py"
    base = ROOT / "src" / Path(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(tree, path):
    found = []
    package = None
    if path is not None and (ROOT / "src") in path.parents:
        rel = list(path.relative_to(ROOT / "src").with_suffix("").parts)
        package = rel[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                found += [parts[:i] for i in range(1, len(parts) + 1)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                base = package[:len(package) - (node.level - 1)]
            else:
                base = []
            module = base + (node.module.split(".") if node.module else [])
            found += [module[:i] for i in range(1, len(module) + 1)]
            found += [module + [alias.name] for alias in node.names]
    return found


def _local_closure(paths, texts=()):
    files, pending = set(), [Path(p).resolve() for p in paths]
    trees = [(ast.parse(text), None) for text in texts]
    while pending or trees:
        if trees:
            tree, origin = trees.pop()
        else:
            origin = pending.pop()
            if origin in files:
                continue
            files.add(origin)
            tree = ast.parse(origin.read_text(encoding="utf-8"))
        for parts in _imports(tree, origin):
            found = _module_file(parts)
            if found is not None and found.resolve() not in files:
                pending.append(found.resolve())
    return sorted(files)


def simulator_files(program=None):
    """Local files the simulator, the ten policy programs and the labels import."""
    if program is None:
        from . import runtime
        program = runtime.sources()
    meta = ROOT / "src" / "delivery_fleet" / "meta_policy"
    seeds = [meta / "runtime.py", meta / "labels.py", meta / "__init__.py",
             ROOT / "operations" / "exact_runtime.py", ROOT / "scripts" / "run_nyc_benchmark5.py"]
    return _local_closure(seeds, [program[p] for p in POLICIES])


def simulator_identity():
    """Hash of the generated policy programs plus their local import closure."""
    from . import runtime
    program = runtime.sources()
    files = simulator_files(program)
    return identity({"generated_programs": identity([program[p] for p in POLICIES]),
                     "files": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in files}})


def feature_identity():
    from . import features
    return identity({"version": features.VERSION, "file": digest(features.__file__)})


def code_identity():
    """Backward-compatible alias used in reports: the simulator identity."""
    return simulator_identity()


# --- helpers ----------------------------------------------------------------

def configure_threads():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    # Also limit libraries loaded before initializer execution (not only env vars).
    from threadpoolctl import threadpool_limits
    global _thread_limiter
    _thread_limiter = threadpool_limits(limits=1)


def _read_receipt(path, signature):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("signature") != signature:
        raise ValueError(f"incompatible saved artifact; use a new output directory: {path}")
    if "loss" in value and (not math.isfinite(value["loss"]) or value["loss"] < 0):
        raise ValueError(f"invalid saved rollout loss: {path}")
    return value


def _uniform(*parts):
    """Deterministic U[0,1) from the given parts (no RNG state to checkpoint)."""
    return int(identity(list(parts))[:13], 16) / 16**13


def _pick(options, *parts):
    return options[int(identity(list(parts))[:12], 16) % len(options)]


def _keep_deciding(runtime, duration, config):
    if runtime.ns["_meta_finished"]:
        return False
    if runtime.now < duration:
        return True
    if runtime.now > duration + MAX_POST_HORIZON_MIN:
        raise RuntimeError("deliveries still outstanding long after the arrival horizon")
    return (config["decide_until"] == "all_delivered" and runtime.outstanding()
            and bool(runtime.ns["events"]))


def _episode(job):
    """One episode per process task; every stage has durable atomic receipts."""
    from .runtime import FleetRuntime
    from .features import extract_windows, map_features
    from .learning import LossSelector
    from . import labels
    configure_threads()
    config, episode, mode = job["config"], job["episode"], job["mode"]
    folder = Path(job["output"]) / "episodes" / episode["id"]
    folder.mkdir(parents=True, exist_ok=True)
    signature = job["signature"]
    with RunLock(folder / ".episode.lock"):
        receipt = _read_receipt(folder / "complete.json", signature)
        if receipt:
            return receipt
        started = time.monotonic()
        base_status = {"signature": signature, "episode": episode["id"]}
        atomic_json(folder / "status.json", {**base_status, "status": "running", **process_identity()})
        selector = LossSelector.load(job["selector"]) if job["selector"] else None
        runtime = None
        try:
            runtime = FleetRuntime(episode["graph"], episode["scenario"],
                config["spatial_model"], config["paper_model"], episode["prior_rates_per_hour"],
                scratch=folder / "scratch", shortlist_k=config["shortlist_k"],
                shared_cache=job["shared_cache"] if os.name != "nt" else None,
                settings=config["controller_settings"])
            static = map_features(runtime.ns["graph"])
            checkpoint = _read_receipt(folder / "checkpoint.json", signature)
            if checkpoint:
                state_path = folder / checkpoint["state"]
                if digest(state_path) != checkpoint["sha256"]:
                    raise ValueError("episode checkpoint checksum mismatch")
                runtime.restore(state_path.read_bytes())
                step = checkpoint["step"]
                age, recent = checkpoint["policy_age_min"], checkpoint["recent_loss_rate"]
            else:
                step, age, recent = 0, 0., 0.
                runtime.set_policy(config["initial_policy"])
            duration = runtime.ns["scenario"].duration_minutes
            portfolio, windows = config["policies"], config["holding_minutes"]
            explore = config["exploration"]
            fixed_behaviour = None
            if selector is None and _uniform(job["seed"], episode["id"], "fixed") < explore["fixed_episode_fraction"]:
                fixed_behaviour = _pick(explore["policies"], job["seed"], episode["id"], "fixed-policy")
            stopped = False
            while _keep_deciding(runtime, duration, config):
                if job["max_steps"] is not None and step >= job["max_steps"]:
                    stopped = True
                    break
                t, old_loss, old_policy = runtime.now, runtime.loss, runtime.policy
                features = extract_windows(runtime, static, windows,
                                           policy_age_min=age, recent_loss_rate=recent)
                if selector:
                    # No incumbent at the first decision: plain argmin, no hysteresis.
                    policy, holding, prediction = selector.choose(features, None if step == 0 else old_policy)
                else:
                    # Reproducible exploration, determined without looking at any
                    # rollout outcomes or unreleased orders. Changes across rounds.
                    if fixed_behaviour is not None:
                        policy, reason = fixed_behaviour, "fixed_behaviour_episode"
                    elif (step > 0 and old_policy in explore["policies"]
                          and _uniform(job["seed"], episode["id"], step, "stay") < explore["stickiness"]):
                        policy, reason = old_policy, "sticky"
                    else:
                        policy, reason = _pick(explore["policies"], job["seed"], episode["id"], step), "uniform"
                    holding = config["default_holding_minutes"]
                    prediction = {"bootstrap": True, "selected": policy, "holding_min": holding,
                                  "reason": reason}
                labelled = (mode == "collect" and _uniform(job["seed"], episode["id"], step, "label")
                            < config["label_step_probability"])
                decision_signature = identity([signature, step, t, policy, holding, labelled])
                trace_path = folder / "decisions" / f"{step:05d}.json"
                old_trace = _read_receipt(trace_path, decision_signature)
                if old_trace and (old_trace["policy"] != policy or old_trace["holding_min"] != holding):
                    raise ValueError("resume changed a previously chosen action")

                def boundary(hold):
                    return min(t + hold, duration) if t < duration else t + hold
                if labelled:
                    before = runtime.snapshot()
                    horizon = t + config["scoring_horizon_minutes"]
                    next_state, tails = None, {}
                    for hold in windows:
                        losses = {}
                        for candidate in portfolio:
                            branch = folder / "rollouts" / f"{step:05d}_{hold:.17g}_{candidate}.json"
                            branch_signature = identity([decision_signature, candidate, hold])
                            saved = _read_receipt(branch, branch_signature)
                            selected = candidate == policy and hold == holding
                            boundary_path = folder / "states" / f"{step + 1:05d}.pickle"
                            if saved:
                                if selected:
                                    if digest(boundary_path) != saved["boundary_sha256"]:
                                        raise ValueError("selected branch checkpoint checksum mismatch")
                                    next_state = boundary_path.read_bytes()
                                losses[candidate] = saved["loss"]
                                if saved.get("tail_digest"):
                                    tails.setdefault(saved["tail_digest"], saved["loss"])
                                continue
                            state, tail_digest, window_seconds = labels.prepare_tail(
                                runtime, before, candidate=candidate, boundary=boundary(hold),
                                horizon=horizon, continuation=config["continuation_policy"],
                                keep_boundary=selected)
                            state_sha = None
                            if selected:
                                atomic_bytes(boundary_path, state)
                                state_sha = digest(boundary_path)
                                next_state = state
                            # Identical tail-start states have identical tails.
                            reused = tail_digest in tails
                            if reused:
                                loss, tail_seconds = tails[tail_digest], 0.
                            else:
                                loss, tail_seconds = labels.score_tail(runtime, horizon=horizon,
                                                                       start_loss=old_loss)
                                tails[tail_digest] = loss
                            atomic_json(branch, {"signature": branch_signature, "loss": loss,
                                "policy": candidate, "holding_min": hold,
                                "boundary_sha256": state_sha, "horizon_min": horizon,
                                "tail_digest": tail_digest, "tail_reused": reused,
                                "window_wall_seconds": window_seconds,
                                "tail_wall_seconds": tail_seconds,
                                "wall_seconds": window_seconds + tail_seconds,
                                "peak_rss_gib": peak_rss_gib()})
                            losses[candidate] = loss
                        row = {"signature": identity([decision_signature, hold]),
                               "group": episode["id"] + ":" + job["scenario_sha256"], "city": episode["city"],
                               "split": episode["split"], "scenario_sha256": job["scenario_sha256"],
                               "source_identity": job["simulator_identity"], "run_signature": signature,
                               "label_definition": job["label_definition"],
                               "feature_identity": job["feature_identity"],
                               "scenario_provenance": job["scenario_provenance"],
                               "step": step, "time_min": t, "holding_min": hold,
                               "after_arrival_horizon": t >= duration,
                               "robots": len(runtime.ns["robots"]), "features": features[hold],
                               "losses": losses, "chosen_policy": policy, "chosen_holding_min": holding}
                        atomic_json(folder / "examples" / f"{step:05d}_{hold:.17g}.json", row)
                    if next_state is None:
                        raise RuntimeError("selected branch was not evaluated")
                    runtime.restore(next_state)
                else:
                    # Only the chosen policy runs (unlabelled step, validation or test).
                    runtime.set_policy(policy)
                    runtime.advance(boundary(holding))
                    next_state = runtime.snapshot()
                    atomic_bytes(folder / "states" / f"{step + 1:05d}.pickle", next_state)
                age = age + runtime.now - t if policy == old_policy else runtime.now - t
                recent = (runtime.loss - old_loss) / max(runtime.now - t, 1e-9)
                atomic_json(trace_path, {"signature": decision_signature, "time_min": t,
                    "policy": policy, "holding_min": holding, "prediction": prediction,
                    "labelled": labelled, "committed_until_min": runtime.now})
                step += 1
                state = f"states/{step:05d}.pickle"
                atomic_json(folder / "checkpoint.json", {"signature": signature, "step": step,
                    "state": state, "sha256": digest(folder / state),
                    "policy_age_min": age, "recent_loss_rate": recent})
                # Resume needs only the committed state; older snapshots (tens of
                # MB each on big maps) would otherwise fill the disk.
                (folder / "states" / f"{step - 1:05d}.pickle").unlink(missing_ok=True)
                atomic_json(folder / "status.json", {**base_status, "status": "running",
                    "step": step, "simulation_time_min": runtime.now,
                    "simulation_duration_min": duration, "wall_seconds": time.monotonic() - started,
                    "peak_rss_gib": peak_rss_gib(), **process_identity()})
                print(f"{episode['id']} step={step} sim={runtime.now:g}/{duration:g} "
                      f"policy={policy} hold={holding:g}m labelled={labelled}", flush=True)
            if stopped:
                # Pilot measurement: no drain and no completion receipt, so a later
                # run with the same output directory resumes this episode.
                receipt = {**base_status, "status": "pilot_stopped", "city": episode["city"],
                           "split": episode["split"], "steps": step, "simulation_time_min": runtime.now,
                           "wall_seconds_this_attempt": time.monotonic() - started,
                           "peak_rss_gib": peak_rss_gib(), **process_identity()}
                atomic_json(folder / "status.json", receipt)
                return receipt
            # Delivery drain is part of the actual trajectory result as well.
            runtime.drain()
            traces = [json.loads(p.read_text()) for p in sorted((folder / "decisions").glob("*.json"))]
            previous = config["initial_policy"]
            switches, minutes = 0, {p: 0. for p in portfolio}
            for trace in traces:
                switches += trace["policy"] != previous
                previous = trace["policy"]
                minutes[trace["policy"]] += trace["committed_until_min"] - trace["time_min"]
            receipt = {**base_status, "city": episode["city"],
                "split": episode["split"], "steps": step, "loss": runtime.loss,
                "delivered": runtime.ns["delivered"], "on_time": runtime.ns["metrics"].on_time,
                "late": runtime.ns["delivered"] - runtime.ns["metrics"].on_time,
                "finish_min": runtime.now, "wall_seconds_this_attempt": time.monotonic() - started,
                "labelled_steps": sum(bool(t.get("labelled")) for t in traces),
                "decisions_after_arrival_horizon": sum(t["time_min"] >= duration for t in traces),
                "switches": switches, "policy_minutes_before_drain": minutes,
                "peak_rss_gib": peak_rss_gib(), "mode": mode, "status": "complete"}
            atomic_json(folder / "complete.json", receipt)
            atomic_json(folder / "status.json", {**receipt, **process_identity()})
            return receipt
        except Exception:
            atomic_json(folder / "status.json", {**base_status, "status": "failed",
                "error": traceback.format_exc(), **process_identity()})
            raise
        finally:
            if runtime is not None:
                runtime.close()


def _check_selector(trained, config, mode, selected, hashes, provenance, label_definition, features):
    if (trained.policies != tuple(p for p in POLICIES if p in config["policies"])
            or trained.windows != tuple(sorted(config["holding_minutes"]))
            or trained.default_window != config["default_holding_minutes"]):
        raise ValueError("selector and manifest portfolio/windows disagree")
    p = trained.provenance or {}
    if p.get("label_definition") != label_definition:
        raise ValueError("selector was trained with different simulator code, base models or label settings")
    if p.get("feature_identity") != features:
        raise ValueError("selector was trained with a different feature extractor")
    seen_hashes = set(p.get("scenario_hashes", []))
    seen_workloads = set(p.get("scenario_workload_hashes", []))
    seen_seeds = set(p.get("scenario_seeds", []))
    if mode == "evaluate":
        for e in selected:
            if (hashes[e["scenario"]] in seen_hashes or provenance[e["id"]]["workload_sha256"] in seen_workloads
                    or provenance[e["id"]]["seed"] in seen_seeds):
                raise ValueError("evaluation scenario, seed or workload was used to fit or validate this selector")
    elif mode == "validate":
        validation = set(p.get("validation_scenario_hashes", []))
        fit_hashes = set(p.get("fit_scenario_hashes", []))
        fit_workloads = set(p.get("fit_scenario_workload_hashes", []))
        fit_seeds = set(p.get("fit_scenario_seeds", []))
        if not validation:
            raise ValueError("selector provenance has no validation scenarios; refit with this version")
        for e in selected:
            sha = hashes[e["scenario"]]
            if (sha not in validation or sha in fit_hashes
                    or provenance[e["id"]]["workload_sha256"] in fit_workloads
                    or provenance[e["id"]]["seed"] in fit_seeds):
                raise ValueError("validate mode accepts only this selector's held-out validation episodes")


def run(manifest, output, *, processes=60, memory_per_worker_gib=8., reserve_gib=16.,
        selector=None, seed=42, mode="collect", shared_cache=None, describe=False, max_steps=None):
    config = load_manifest(manifest)
    if mode not in MODES:
        raise ValueError("unknown execution mode")
    selected = [e for e in config["episodes"] if e["split"] in SPLITS_BY_MODE[mode]]
    if not selected or (mode != "collect" and selector is None):
        raise ValueError("need appropriate episodes, and a selector for validate/evaluate")
    if config["initial_policy"] not in config["policies"]:
        raise ValueError("initial policy must be in candidate portfolio")
    if max_steps is not None and (isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    hashes = {}
    for path in {config["spatial_model"], config["paper_model"],
                 *(e[n] for e in selected for n in ("graph", "scenario"))}:
        hashes[path] = digest(path)
    from .separation import check_separation
    provenance = check_separation(config, selected, mode=mode)
    simulator, features = simulator_identity(), feature_identity()
    from . import labels
    label_definition = identity({"simulator": simulator, "label_version": labels.LABEL_VERSION,
        "models": [hashes[config[n]] for n in ("spatial_model", "paper_model")],
        "shortlist_k": config["shortlist_k"], "controller_settings": config["controller_settings"],
        "scoring_horizon_minutes": config["scoring_horizon_minutes"],
        "continuation_policy": config["continuation_policy"], "runtime_versions": runtime_versions()})
    selector_path = None
    if selector:
        from .learning import LossSelector
        selector_path = Path(selector).resolve()
        hashes[str(selector_path)] = digest(selector_path)
        _check_selector(LossSelector.load(selector_path), config, mode, selected, hashes, provenance,
                        label_definition, features)
    signature = identity({"version": 2, "config": config, "files": hashes,
                          "label_definition": label_definition, "feature_identity": features,
                          "selector": str(selector_path) if selector_path else None,
                          "seed": seed, "mode": mode})
    scenario_splits = {}
    for episode in selected:
        sha = hashes[episode["scenario"]]
        split = scenario_splits.setdefault(sha, episode["split"])
        if split != episode["split"]:
            raise ValueError("a scenario payload occurs in multiple episode splits")
    workers = min(len(selected), worker_count(processes, memory_per_worker_gib=memory_per_worker_gib,
                                             reserve_gib=reserve_gib))
    report = {"signature": signature, "episodes": len(selected), "episode_workers": workers,
        "max_compute_processes": workers + 1 if workers else 1, "requested_process_budget": processes,
        "available_memory_gib": available_gib(), "memory_per_worker_gib": memory_per_worker_gib,
        "reserve_gib": reserve_gib, "mode": mode, "max_steps": max_steps,
        "simulator_identity": simulator, "feature_identity": features,
        "label_definition": label_definition, "runtime_versions": runtime_versions(),
        "portfolio": list(config["policies"]), "continuation_policy": config["continuation_policy"],
        "label_step_probability": config["label_step_probability"],
        "decide_until": config["decide_until"],
        "note": "RAM estimates are conservative admission settings, not kernel-enforced limits"}
    if describe:
        print(json.dumps(report, indent=2))
        return report
    configure_threads()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(shared_cache).resolve() if shared_cache else output / "shared_exact_index"
    jobs = [{"config": config, "episode": episode, "signature": identity([signature, episode["id"]]),
             "output": str(output), "selector": str(selector_path) if selector_path else None,
             "seed": seed, "mode": mode, "max_steps": max_steps,
             "simulator_identity": simulator, "feature_identity": features,
             "label_definition": label_definition,
             "scenario_provenance": provenance[episode["id"]],
             "scenario_sha256": hashes[episode["scenario"]], "shared_cache": str(cache)}
            for episode in selected]
    # Longest expected episodes first (largest map, then most orders), so big
    # maps don't start last and leave the rest of the VM idle at the end.
    jobs.sort(key=lambda j: (-os.path.getsize(j["episode"]["graph"]),
                             -j["scenario_provenance"]["orders"], j["episode"]["id"]))
    with RunLock(output / ".controller.lock"):
        _read_receipt(output / "run_manifest.json", signature)
        atomic_json(output / "run_manifest.json", {**report, "configuration": config, "file_hashes": hashes})
        print(json.dumps(report), flush=True)
        results, failures = [], []
        started = time.monotonic()

        def progress(status):
            atomic_json(output / "progress.json", {"complete": len(results), "failed": len(failures),
                "total": len(jobs), "elapsed_seconds": time.monotonic() - started,
                "status": status, **process_identity()})

        def record(result):
            results.append(result)
            progress("running")
            print(f"episodes {len(results)}/{len(jobs)} last={result['episode']} "
                  f"status={result['status']} elapsed={(time.monotonic()-started)/3600:.2f}h; "
                  "ETA requires representative completed large-map episodes", flush=True)

        def fail(job, error):
            # One failed episode must not discard or stall the others.
            failures.append({"episode": job["episode"]["id"], "error": error})
            progress("running")
            print(f"episode FAILED {job['episode']['id']}; continuing with the others", flush=True)

        progress("running")
        if workers:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=configure_threads, max_tasks_per_child=1) as pool:
                futures = {pool.submit(_episode, job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        record(future.result())
                    except Exception:
                        fail(futures[future], traceback.format_exc())
        else:
            for job in jobs:
                try:
                    record(_episode(job))
                except Exception:
                    fail(job, traceback.format_exc())
        final = ("failed" if failures else "pilot_stopped"
                 if any(r["status"] == "pilot_stopped" for r in results) else "complete")
        if failures:
            atomic_json(output / "failure.json", {"signature": signature, "failures": failures})
        atomic_json(output / "summary.json", {"signature": signature, "status": final,
                    "results": sorted(results, key=lambda r: r["episode"]), "failures": failures})
        progress(final)
        if failures:
            raise RuntimeError(f"{len(failures)} episode(s) failed; see {output / 'failure.json'}")
        return results


def collect_examples(roots):
    """Read complete portfolio rows, including compatible work from partial runs."""
    records = []
    seen = set()
    for root in roots:
        for path in sorted(Path(root).glob("episodes/*/examples/*.json")):
            row = json.loads(path.read_text())
            proof = row.get("scenario_provenance", {})
            if (proof.get("role") != row["split"] or not proof.get("registry_identity")
                    or not proof.get("workload_sha256")):
                raise ValueError("training example lacks fresh-scenario separation provenance")
            if row["signature"] in seen:
                continue
            seen.add(row["signature"])
            records.append(row)
    if not records:
        raise ValueError("no completed counterfactual training examples found")
    if len({r["label_definition"] for r in records}) != 1:
        raise ValueError("refusing to mix simulator versions, loss horizons, continuation policies "
                         "or base-model versions")
    if len({r.get("feature_identity") for r in records}) != 1:
        raise ValueError("refusing to mix feature extractor versions")
    return records
