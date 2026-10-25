from __future__ import annotations

"""Fast runner for NYC Benchmark 1 without changing policy semantics.

Benchmark 1 still chooses the nearest currently available battery-feasible robot.
This wrapper only reuses the compiled SciPy/CSR distance backend and in-memory
charger initialization already used by later benchmarks.
"""

import sys
import time
import types

import run_nyc_benchmark2_fast as fast


TARGET = fast.SCRIPTS / "run_nyc_benchmark.py"


def _load_benchmark1_namespace() -> dict[str, object]:
    source = TARGET.read_text(encoding="utf-8")
    module_name = "benchmark1_fast_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(TARGET)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    fast._install_in_memory_graph_initialization()
    distance_row = fast._install_compiled_distance_backend()
    namespace = _load_benchmark1_namespace()
    fast._install_fast_direct_distance(namespace, distance_row)

    started = time.perf_counter()
    try:
        namespace["main"]()
    finally:
        elapsed = time.perf_counter() - started
        print(
            "FAST_ROUTING_STATS "
            f"compiled_dijkstra_calls={fast._fast_dijkstra_calls} "
            f"cache_hits={fast._fast_dijkstra_cache_hits} "
            f"compiled_dijkstra_seconds={fast._fast_dijkstra_seconds:.3f} "
            f"runner_seconds={elapsed:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
