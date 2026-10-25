"""Bounded, owned child processes. One fresh process per episode or model fit."""
from __future__ import annotations

import multiprocessing
from pathlib import Path
import time
import traceback

from .config import atomic, available_gib, workers


def numeric_environment(threads):
    import os
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def readable(pipe):
    try:
        return pipe.poll()
    except (EOFError, BrokenPipeError, OSError):
        return False


def _episode_entry(pipe, stop, config, episode, family, science, frozen):
    from .collection import EpisodeCancelled, collect_episode, init_worker
    try:
        init_worker()
        result = collect_episode(config, episode, family, science, frozen,
            stop_signal=stop, on_ready=lambda: pipe.send(("ready", None)))
        pipe.send(("result", result))
    except EpisodeCancelled as exc:
        pipe.send(("cancelled", str(exc)))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


def stop_owned(processes, signal, grace):
    """Only terminate children created by this controller; never other VM jobs."""
    signal.set()
    deadline = time.monotonic() + grace
    while any(p.is_alive() for p in processes) and time.monotonic() < deadline:
        for process in processes:
            process.join(timeout=0)
        time.sleep(.05)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2
    while any(p.is_alive() for p in processes) and time.monotonic() < deadline:
        time.sleep(.05)
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(timeout=.1)


def run_episodes(config, jobs, science, progress=None, *, entry=_episode_entry,
                 memory=available_gib, context=None):
    context = context or multiprocessing.get_context("spawn")
    slots = workers(config, memory_gib=memory())
    if slots < 1:
        raise RuntimeError("Insufficient memory/CPU allowance; no workers were launched")
    stop = context.Event()
    active, results = {}, []
    pending = iter(jobs)
    next_job = next(pending, None)
    try:
        while next_job is not None or active:
            available = memory()
            if available < config.memory_reserve_gib:
                raise MemoryError("Available RAM crossed the reserve; checkpointing and stopping owned workers")
            # Account for allocations not yet visible in MemAvailable and reserve
            # the fitter's RAM even before a fit starts. Ramp by at most 8 jobs.
            unready = sum(not job["ready"] for job in active.values())
            allowance = config.memory_reserve_gib + config.trainer_memory_gib + (unready + 1) * config.worker_memory_gib
            admitted = 0
            while next_job is not None and len(active) < slots and admitted < 8 and available >= allowance:
                episode, family, frozen = next_job
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=entry, args=(child, stop, config, episode, family, science, frozen))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                active[process.pid] = {"process":process, "pipe":parent, "ready":False,
                    "episode":episode["id"], "family":family}
                next_job = next(pending, None)
                admitted += 1
                allowance += config.worker_memory_gib
                available = memory()
            if not active:
                raise RuntimeError("Memory reserve prevents further admission; existing work remains resumable")
            for pid, job in list(active.items()):
                process, pipe = job["process"], job["pipe"]
                result = None
                while readable(pipe):
                    try:
                        kind, payload = pipe.recv()
                    except EOFError:
                        if result is None:
                            raise RuntimeError(f"Episode worker closed without a result: {job['episode']}")
                        break
                    if kind == "ready":
                        job["ready"] = True
                    elif kind == "result":
                        result = payload
                        break
                    else:
                        raise RuntimeError(f"Episode failed {job['episode']}/{job['family']}: {kind}\n{payload}")
                if result is not None:
                    process.join(timeout=1)
                    if process.is_alive() or process.exitcode != 0:
                        raise RuntimeError("Episode returned but its process did not exit successfully")
                    results.append(result)
                    pipe.close()
                    del active[pid]
                elif not process.is_alive():
                    # A message can arrive between poll and the exit check. Read
                    # it on the next pass before classifying an abrupt exit.
                    if not readable(pipe):
                        raise RuntimeError(f"Episode worker exited without a result: {job['episode']} ({process.exitcode})")
            if progress:
                progress(results, len(active))
            if active:
                time.sleep(.05)
        return results
    except BaseException as exc:
        stop_owned([job["process"] for job in active.values()], stop, config.cancellation_grace_seconds)
        atomic(Path(config.output) / "last_shutdown.json", {"reason":str(exc),
            "resumable_jobs":[{"episode":job["episode"],"family":job["family"]} for job in active.values()],
            "completed_jobs":len(results),"owned_processes_stopped":True})
        raise
    finally:
        for job in active.values():
            job["pipe"].close()


def _fit_entry(pipe, config, family, science, episodes, watermark):
    try:
        numeric_environment(config.fit_threads)
        from .workflow import _publish
        from .replay import snapshot
        rows = snapshot(config.output, family, science, episodes, watermark)
        pipe.send(("result", _publish(config, family, rows, science)))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


class FitTask:
    def __init__(self, config, family, science, episodes, watermark):
        self.config, self.family, self.started = config, family, time.monotonic()
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_fit_entry, args=(child, config, family, science, episodes, watermark))
        try:
            self.process.start()
        except BaseException:
            self.pipe.close()
            child.close()
            raise
        child.close()

    def poll(self):
        if readable(self.pipe):
            try:
                kind, payload = self.pipe.recv()
            except EOFError as exc:
                raise RuntimeError("Fit process exited without a result") from exc
            self.process.join(timeout=1)
            if kind != "result" or self.process.is_alive() or self.process.exitcode != 0:
                raise RuntimeError("Fit failed: " + str(payload))
            self.pipe.close()
            return payload
        if not self.process.is_alive():
            raise RuntimeError("Fit process exited without a result")
        if time.monotonic() - self.started > self.config.fit_timeout_seconds:
            raise RuntimeError("Fit exceeded its wall-clock limit")
        return None

    def stop(self):
        context = multiprocessing.get_context("spawn")
        stop_owned([self.process], context.Event(), .1)
        self.pipe.close()
