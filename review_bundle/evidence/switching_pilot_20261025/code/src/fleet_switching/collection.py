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
from .config import POLICIES, atomic, durable_bytes, guard_city, identity, immutable, locked, read, sha, sync_directory
from .models import Predictor
from .provenance import base_weights, learner_id, row_digest, validate_row
from . import replay


class EpisodeCancelled(RuntimeError):
    pass


def check_stop(signal):
    if signal is not None and signal.is_set():
        raise EpisodeCancelled("Episode stopped at a resumable branch boundary")


def init_worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    # Retain the controller object for this worker's entire process lifetime.
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def save_state(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_bytes(path, blob)
    return hashlib.sha256(blob).hexdigest()


def recover_checkpoint(folder, signature):
    """Use the previous durable pair if the newest pair is missing/corrupt."""
    found = False
    for name in ("checkpoint.json", "checkpoint.previous.json"):
        path = folder / name
        if not path.exists():
            continue
        found = True
        try:
            value = read(path)
        except (ValueError, OSError):
            continue
        if value["job_signature"] != signature:
            raise ValueError("Checkpoint/input identity changed")
        state = Path(value["state_path"])
        if state.parent.resolve() != (folder / "states").resolve():
            raise ValueError("Checkpoint points outside its owned state directory")
        if state.is_file() and sha(state) == value["state_sha256"]:
            return value
    if found:
        raise ValueError("No valid checkpoint/state pair remains")
    return None


def collect_episode(config, episode, family, science, frozen=None, backend_factory=Backend,
                    stop_signal=None, on_ready=None):
    check_stop(stop_signal)
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
                              "frozen":frozen,"label_version":LABEL_VERSION,
                              "base_weights":base_weights(config), "learner_id":learner_id(config, family),
                              "bootstrap_policy":config.bootstrap_policy,
                              "max_post_horizon_minutes":config.max_post_horizon_minutes})
    if sha(episode["scenario"]) != episode["scenario_sha256"] or sha(episode["graph"]) != episode["graph_sha256"]:
        raise ValueError("Episode input bytes changed")
    done_path = folder / "complete.json"
    if done_path.exists():
        receipt = read(done_path)
        if (receipt["job_signature"] != job_signature
                or receipt.get("receipt_sha256") != identity({k:v for k,v in receipt.items() if k != "receipt_sha256"})):
            raise ValueError("A completed episode belongs to a different experiment")
        return receipt
    # One outer worker owns an episode. OS lock also protects accidental retries.
    with locked(folder / ".episode.lock"):
        backend = backend_factory(config, episode, folder / "scratch")
        runtime = backend.runtime
        cp_path = folder / "checkpoint.json"
        step, age, recent, switches, occupancy = 0, 0., 0., 0, {}
        started = time.monotonic()
        model, model_hash = None, None
        try:
            if on_ready:
                on_ready()
            state_blob = runtime.snapshot()
            initial = folder / "states/initial.pickle"
            initial_receipt = folder / "initial.json"
            if initial_receipt.exists():
                info = read(initial_receipt)
                if info["job_signature"] != job_signature or sha(initial) != info["sha256"]:
                    raise ValueError("Initial durable state changed")
                state_blob = initial.read_bytes()
                runtime.restore(state_blob)
            else:
                initial_hash = save_state(initial, state_blob)
                immutable(initial_receipt, {"job_signature":job_signature,"sha256":initial_hash})
            cp = recover_checkpoint(folder, job_signature)
            if cp is not None:
                state_blob = Path(cp["state_path"]).read_bytes()
                runtime.restore(state_blob)
                step, age, recent = cp["step"], cp["age"], cp["recent"]
                switches, occupancy = cp["switches"], dict(cp["policy_minutes"])
            while runtime.outstanding():
                check_stop(stop_signal)
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
                              "model_kind":model.payload["kind"],
                              "state_sha256":before_hash,"job_signature":job_signature,"time_min":t}
                    atomic(intent_path, intent)
                chosen = intent["chosen_policy"]
                if chosen not in POLICIES:
                    raise ValueError("Actor chose an unsupported benchmark")
                end = min(t + 15., config.duration_hours * 60.) if t < config.duration_hours * 60 else t + 15.
                if row_path.exists():
                    row = validate_row(read(row_path), science=science, family=family)
                    if row["intent"] != intent:
                        raise ValueError("Committed row/action intent mismatch")
                    if not boundary_path.is_file() or sha(boundary_path) != row["next_state_sha256"]:
                        # A retained previous checkpoint can repair a lost newest
                        # boundary. Preserve the original actor intent and row;
                        # accept only an exact reproduction of its saved bytes.
                        check_stop(stop_signal)
                        runtime.restore(before)
                        runtime.set_policy(chosen)
                        runtime.advance(end)
                        repaired = runtime.snapshot()
                        if hashlib.sha256(repaired).hexdigest() != row["next_state_sha256"]:
                            raise ValueError("Recovered physical boundary differs from committed row")
                        save_state(boundary_path, repaired)
                    chosen_blob = boundary_path.read_bytes()
                else:
                    costs, diagnostics, chosen_blob = {}, {}, None
                    candidates = POLICIES if training else (chosen,)
                    for candidate in candidates:
                        check_stop(stop_signal)
                        branch_started = time.monotonic()
                        runtime.restore(before)
                        runtime.set_policy(candidate)
                        runtime.advance(end)
                        boundary = runtime.snapshot()  # Before projection touches planner caches.
                        window_seconds = time.monotonic() - branch_started
                        realized = runtime.loss - old_loss
                        if training:
                            projection_started = time.monotonic()
                            terminal, info = backend.project()
                            cost = realized + terminal.loss
                            if not math.isfinite(cost) or cost < -1e-8:
                                raise ValueError("Invalid projected branch cost")
                            costs[candidate] = max(0., cost)
                            diagnostics[candidate] = {**info,"realized_loss":realized,"projected_loss":terminal.loss,
                                "window_restore_advance_snapshot_seconds":window_seconds,
                                "projection_seconds":time.monotonic()-projection_started}
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
                           "profile":episode.get("profile", "unknown"),
                           "robots":len(runtime.ns["robots"]),"features":features,"costs":costs,
                           "diagnostics":diagnostics,"intent":intent,"next_state_sha256":next_hash}
                    row["content_sha256"] = row_digest(row)
                    immutable(row_path, row)
                if training:
                    replay.record(root, row)
                runtime.restore(chosen_blob)
                state_blob = chosen_blob
                age = age + (runtime.now - t) if chosen == previous else runtime.now - t
                recent = (runtime.loss - old_loss) / max(1e-9, runtime.now - t)
                switches += int(step > 0 and chosen != previous)
                occupancy[chosen] = occupancy.get(chosen, 0.) + runtime.now - t
                step += 1
                if cp is not None:
                    atomic(folder / "checkpoint.previous.json", cp)
                next_cp = {"job_signature":job_signature,"step":step,"state_path":str(boundary_path),
                    "state_sha256":row["next_state_sha256"],"age":age,"recent":recent,
                    "switches":switches, "policy_minutes":dict(occupancy),
                    "model_sha256":intent["model_sha256"]}
                atomic(cp_path, next_cp)
                # Retain the initial state and both durable checkpoint pairs.
                retained = {initial.resolve(), boundary_path.resolve()}
                if cp is not None:
                    retained.add(Path(cp["state_path"]).resolve())
                cp = next_cp
                for old_state in (folder / "states").glob("[0-9][0-9][0-9][0-9][0-9].pickle"):
                    if old_state.resolve().parent == (folder / "states").resolve() and old_state.resolve() not in retained:
                        old_state.unlink()
                sync_directory(folder / "states")
            metrics = runtime.ns.get("metrics")
            on_time = getattr(metrics, "on_time", None)
            result = {"status":"complete","job_signature":job_signature,"episode":episode["id"],
                "city":episode["city"],"split":episode["split"],"family":family,"steps":step,
                "loss":runtime.loss,"delivered":runtime.ns["delivered"],"orders":episode["orders"],
                "robots":len(runtime.ns["robots"]),
                "wall_seconds_this_attempt":time.monotonic()-started,"frozen_model":frozen,
                "finish_decision_boundary_min":runtime.now, "switches_after_initial_choice":switches,
                "policy_minutes":occupancy, "on_time":on_time,
                "late":runtime.ns["delivered"] - on_time if on_time is not None else None,
                "charger_queue_wait_min":sum(getattr(metrics, "queue_waits_min", ())),
                "queue_wait_scope":"delivery_and_background", "science_id":science,
                "scenario_sha256":episode["scenario_sha256"], "graph_sha256":episode["graph_sha256"]}
            result["episode_identity"] = identity(episode)
            if result["delivered"] != result["orders"]:
                raise ValueError("Incomplete episode cannot be reported as successful")
            result["receipt_sha256"] = identity(result)
            atomic(done_path, result)
            return result
        finally:
            backend.close()
