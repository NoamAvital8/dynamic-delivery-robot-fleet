from __future__ import annotations

"""Checkpointable runner for the corrected loss-based Benchmark 5.

GitHub-hosted jobs are limited to six hours.  The full NYC B5 search can exceed
that even after exact pruning, so this wrapper preserves the *same policy and
objective* while allowing the event-driven simulation to stop only between
events, serialize its dynamic state, and resume in a fresh job.

Environment variables:
- B5_CHECKPOINT_OUT: path to write a checkpoint before the segment wall limit.
- B5_RESUME_CHECKPOINT: checkpoint path to restore before processing events.
- B5_MAX_SEGMENT_WALL_SECONDS: segment runtime limit; 0 disables checkpoint stop.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_nyc_benchmark5_loss as loss

TARGET = SCRIPTS / "run_nyc_benchmark5.py"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def _patch_checkpointing(source: str) -> str:
    source = _replace_once(
        source,
        "import math\nfrom pathlib import Path\n",
        "import math\nimport os\nimport pickle\nfrom pathlib import Path\n",
        "checkpoint imports",
    )

    old_loop_start = '''    delivered = 0
    now = 0.0
    last_progress = time.perf_counter()

    while events:
        now, _, _, kind, payload = heapq.heappop(events)
'''
    new_loop_start = '''    delivered = 0
    now = 0.0
    wall_seconds_before_resume = 0.0

    checkpoint_out_env = os.environ.get("B5_CHECKPOINT_OUT", "").strip()
    resume_checkpoint_env = os.environ.get("B5_RESUME_CHECKPOINT", "").strip()
    max_segment_wall_seconds = float(
        os.environ.get("B5_MAX_SEGMENT_WALL_SECONDS", "0") or 0.0
    )
    checkpoint_out = Path(checkpoint_out_env) if checkpoint_out_env else None
    resume_checkpoint = (
        Path(resume_checkpoint_env) if resume_checkpoint_env else None
    )

    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
        with resume_checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("format") != "b5-loss-checkpoint-v1":
            raise RuntimeError("unsupported B5 checkpoint format")
        if checkpoint.get("scenario_seed") != scenario.seed:
            raise RuntimeError("checkpoint scenario seed does not match")
        if checkpoint.get("order_count") != len(scenario.orders):
            raise RuntimeError("checkpoint scenario order count does not match")

        robots = checkpoint["robots"]
        robot_by_id = {robot.spec.id: robot for robot in robots}
        schedules = checkpoint["schedules"]
        station_state = checkpoint["station_state"]
        charge_token = checkpoint["charge_token"]
        pending = checkpoint["pending"]
        metrics = checkpoint["metrics"]
        active_phases = checkpoint["active_phases"]
        phase_version = checkpoint["phase_version"]
        service_lock = checkpoint["service_lock"]
        service_token = checkpoint["service_token"]
        pair_distance_cache = checkpoint["pair_distance_cache"]
        coord_cache = checkpoint["coord_cache"]
        direct_distance_cache = checkpoint["direct_distance_cache"]
        baseline_projection_cache = checkpoint["baseline_projection_cache"]
        delivery_distance_m = float(checkpoint["delivery_distance_m"])
        background_distance_m = float(checkpoint["background_distance_m"])
        events = checkpoint["events"]
        heapq.heapify(events)
        event_counter = itertools.count(
            max((int(event[2]) for event in events), default=-1) + 1
        )
        delivered = int(checkpoint["delivered"])
        now = float(checkpoint["now"])
        wall_seconds_before_resume = float(
            checkpoint.get("wall_clock_seconds", 0.0)
        )
        B5_STATS.clear()
        B5_STATS.update(checkpoint["b5_stats"])
        fast._fast_dijkstra_calls = int(
            checkpoint.get("fast_dijkstra_calls", 0)
        )
        fast._fast_dijkstra_cache_hits = int(
            checkpoint.get("fast_dijkstra_cache_hits", 0)
        )
        fast._fast_dijkstra_seconds = float(
            checkpoint.get("fast_dijkstra_seconds", 0.0)
        )
        print(
            f"resumed checkpoint at sim_t={now:.1f} delivered={delivered}/"
            f"{len(scenario.orders)} events={len(events)} prior_wall="
            f"{wall_seconds_before_resume:.1f}s",
            flush=True,
        )

    segment_wall_start = time.perf_counter()
    last_progress = segment_wall_start

    def save_checkpoint() -> None:
        if checkpoint_out is None:
            raise RuntimeError(
                "segment wall limit reached but B5_CHECKPOINT_OUT is not set"
            )
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": "b5-loss-checkpoint-v1",
            "scenario_seed": scenario.seed,
            "order_count": len(scenario.orders),
            "robots": robots,
            "schedules": schedules,
            "station_state": station_state,
            "charge_token": charge_token,
            "pending": pending,
            "metrics": metrics,
            "active_phases": active_phases,
            "phase_version": phase_version,
            "service_lock": service_lock,
            "service_token": service_token,
            "pair_distance_cache": pair_distance_cache,
            "coord_cache": coord_cache,
            "direct_distance_cache": direct_distance_cache,
            "baseline_projection_cache": baseline_projection_cache,
            "delivery_distance_m": delivery_distance_m,
            "background_distance_m": background_distance_m,
            "events": events,
            "delivered": delivered,
            "now": now,
            "b5_stats": dict(B5_STATS),
            "fast_dijkstra_calls": int(fast._fast_dijkstra_calls),
            "fast_dijkstra_cache_hits": int(fast._fast_dijkstra_cache_hits),
            "fast_dijkstra_seconds": float(fast._fast_dijkstra_seconds),
            "wall_clock_seconds": (
                wall_seconds_before_resume + time.perf_counter() - wall_start
            ),
        }
        temporary = checkpoint_out.with_suffix(checkpoint_out.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint_out)
        print(
            f"CHECKPOINT_SAVED path={checkpoint_out} sim_t={now:.1f} "
            f"delivered={delivered}/{len(scenario.orders)} events={len(events)}",
            flush=True,
        )

    while events:
        if (
            max_segment_wall_seconds > 0.0
            and time.perf_counter() - segment_wall_start >= max_segment_wall_seconds
        ):
            save_checkpoint()
            return
        now, _, _, kind, payload = heapq.heappop(events)
'''
    source = _replace_once(
        source,
        old_loop_start,
        new_loop_start,
        "checkpoint loop setup",
    )

    source = _replace_once(
        source,
        '        "wall_clock_seconds": time.perf_counter() - wall_start,\n',
        '        "wall_clock_seconds": (\n'
        '            wall_seconds_before_resume + time.perf_counter() - wall_start\n'
        '        ),\n',
        "cumulative wall clock",
    )
    return source


def _load_namespace() -> dict[str, object]:
    source = loss._patch_source(TARGET.read_text(encoding="utf-8"))
    source = _patch_checkpointing(source)
    module_name = "benchmark5_loss_checkpoint_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(TARGET)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    namespace = _load_namespace()
    namespace["main"]()


if __name__ == "__main__":
    main()
