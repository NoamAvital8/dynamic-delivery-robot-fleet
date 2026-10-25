"""Model-led 15-minute branches, transactional boundary checkpoints and replay.

Every candidate starts from the identical complete snapshot. The action intent
is recorded before any branch is scored. Projected completion never becomes the
next physical state. Model publishers may update between boundaries; validation
and test jobs instead receive one fixed artifact for the entire episode.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time

from . import LABEL_VERSION
from .backend import Backend, activate
from .config import POLICIES, atomic, guard_city, identity, locked, read, sha
from .models import Predictor


def init_worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    # Retain the controller object for this worker's entire process lifetime.
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def save_state(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(blob)
    temp.replace(path)
    return hashlib.sha256(blob).hexdigest()


def collect_episode(config, episode, family, science, frozen=None, backend_factory=Backend):
    guard_city(episode["city"], episode["split"])
    training = episode["split"] == "train"
    if not training and (not frozen or not all(k in frozen for k in ("path","sha256","science_id"))):
        raise ValueError("Validation/test requires an explicitly frozen model")
    if training and frozen is not None:
        raise ValueError("Training uses the versioned live registry")
    if not training and (frozen["science_id"] != science or sha(frozen["path"]) != frozen["sha256"]):
        raise ValueError("Frozen validation/test model identity changed")
    root = Path(config.output)
    folder = root / "actors" / family / episode["split"] / episode["id"]
    folder.mkdir(parents=True, exist_ok=True)
    job_signature = identity({"episode":episode,"family":family,"science_id":science,
                              "frozen":frozen,"label_version":LABEL_VERSION})
    done_path = folder / "complete.json"
    if done_path.exists():
        receipt = read(done_path)
        if receipt["job_signature"] != job_signature:
            raise ValueError("A completed episode belongs to a different experiment")
        return receipt
    if sha(episode["scenario"]) != episode["scenario_sha256"] or sha(episode["graph"]) != episode["graph_sha256"]:
        raise ValueError("Episode input bytes changed")
    # One outer worker owns an episode. OS lock also protects accidental retries.
    with locked(folder / ".episode.lock"):
        backend = backend_factory(config, episode, folder / "scratch")
        runtime = backend.runtime
        cp_path = folder / "checkpoint.json"
        step, age, recent = 0, 0., 0.
        started = time.monotonic()
        model, model_hash = None, None
        try:
            state_blob = runtime.snapshot()
            if cp_path.exists():
                cp = read(cp_path)
                if cp["job_signature"] != job_signature or sha(cp["state_path"]) != cp["state_sha256"]:
                    raise ValueError("Checkpoint/input identity changed")
                state_blob = Path(cp["state_path"]).read_bytes()
                runtime.restore(state_blob)
                step, age, recent = cp["step"], cp["age"], cp["recent"]
            while runtime.outstanding():
                if runtime.now > config.duration_hours * 60 + config.max_post_horizon_minutes:
                    raise RuntimeError("Post-horizon safety bound exceeded; episode is incomplete")
                before = state_blob
                before_hash = hashlib.sha256(before).hexdigest()
                t, old_loss, previous = runtime.now, runtime.loss, runtime.policy
                features = backend.features(age, recent)
                intent_path = folder / "intents" / f"{step:05d}.json"
                row_path = folder / "rows" / f"{step:05d}.json"
                boundary_path = folder / "states" / f"{step+1:05d}.pickle"
                if intent_path.exists():
                    intent = read(intent_path)
                    if intent["state_sha256"] != before_hash or intent["job_signature"] != job_signature:
                        raise ValueError("Uncommitted decision no longer matches its start state")
                else:
                    published = frozen or read(root / "models" / family / "current.json")
                    if model_hash != published["sha256"]:
                        model = Predictor.load(published["path"], published["sha256"])
                        model_hash = published["sha256"]
                    action = model.choose(features)  # Always before observing branch labels.
                    intent = {"chosen_policy":action,"model_sha256":model_hash,"model_path":published["path"],
                              "state_sha256":before_hash,"job_signature":job_signature,"time_min":t}
                    atomic(intent_path, intent)
                chosen = intent["chosen_policy"]
                if chosen not in POLICIES:
                    raise ValueError("Actor chose an unsupported benchmark")
                end = min(t + 15., config.duration_hours * 60.) if t < config.duration_hours * 60 else t + 15.
                if row_path.exists():
                    row = read(row_path)
                    if row["intent"] != intent or sha(boundary_path) != row["next_state_sha256"]:
                        raise ValueError("Committed row/boundary mismatch")
                    chosen_blob = boundary_path.read_bytes()
                else:
                    costs, diagnostics, chosen_blob = {}, {}, None
                    candidates = POLICIES if training else (chosen,)
                    for candidate in candidates:
                        runtime.restore(before)
                        runtime.set_policy(candidate)
                        runtime.advance(end)
                        boundary = runtime.snapshot()  # Before projection touches planner caches.
                        realized = runtime.loss - old_loss
                        if training:
                            terminal, info = backend.project()
                            cost = realized + terminal.loss
                            if not math.isfinite(cost) or cost < -1e-8:
                                raise ValueError("Invalid projected branch cost")
                            costs[candidate] = max(0., cost)
                            diagnostics[candidate] = {**info,"realized_loss":realized,"projected_loss":terminal.loss}
                        if candidate == chosen:
                            chosen_blob = boundary
                    if chosen_blob is None:
                        raise ValueError("The actor-selected branch was not evaluated")
                    next_hash = save_state(boundary_path, chosen_blob)
                    row = {"id":identity([job_signature,step,before_hash]),"episode":episode["id"],
                           "city":episode["city"],"split":episode["split"],"seed":episode["seed"],
                           "scenario_sha256":episode["scenario_sha256"],"graph_sha256":episode["graph_sha256"],
                           "workload_sha256":episode["workload_sha256"],"science_id":science,
                           "label_version":LABEL_VERSION,"family":family,"step":step,"time_min":t,
                           "robots":len(runtime.ns["robots"]),"features":features,"costs":costs,
                           "diagnostics":diagnostics,"intent":intent,"next_state_sha256":next_hash}
                    atomic(row_path, row)
                runtime.restore(chosen_blob)
                state_blob = chosen_blob
                age = age + (runtime.now - t) if chosen == previous else runtime.now - t
                recent = (runtime.loss - old_loss) / max(1e-9, runtime.now - t)
                step += 1
                atomic(cp_path, {"job_signature":job_signature,"step":step,"state_path":str(boundary_path),
                    "state_sha256":row["next_state_sha256"],"age":age,"recent":recent,
                    "model_sha256":intent["model_sha256"]})
                # Only this job's superseded state is disposable; receipts/rows persist.
                old_state = folder / "states" / f"{step-1:05d}.pickle"
                if old_state.is_file() and old_state.resolve().is_relative_to(folder.resolve()):
                    old_state.unlink()
            result = {"status":"complete","job_signature":job_signature,"episode":episode["id"],
                "city":episode["city"],"split":episode["split"],"family":family,"steps":step,
                "loss":runtime.loss,"delivered":runtime.ns["delivered"],"orders":episode["orders"],
                "robots":len(runtime.ns["robots"]),
                "wall_seconds_this_attempt":time.monotonic()-started,"frozen_model":frozen}
            if result["delivered"] != result["orders"]:
                raise ValueError("Incomplete episode cannot be reported as successful")
            atomic(done_path, result)
            return result
        finally:
            backend.close()
