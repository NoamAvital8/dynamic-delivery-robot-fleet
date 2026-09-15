from __future__ import annotations

"""Compatibility wrapper for the vectorized corrected B5 runner.

The first fast runner expects the following nested function declaration to be
split after the opening parenthesis when injecting exact winner
materialization.  The base B5 source keeps it on one line.  This wrapper makes
that formatting-only transformation before applying the fast patch.  It does
not alter simulation or policy semantics.
"""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_nyc_benchmark5_loss as corrected
import run_nyc_benchmark5_loss_fast as fast


def _load_namespace() -> dict[str, object]:
    source = corrected._patch_source(corrected.TARGET.read_text(encoding="utf-8"))
    source = source.replace(
        "    def cancel_editable_route_for_replan(robot: RobotState, now: float, was_busy: bool) -> None:\n",
        "    def cancel_editable_route_for_replan(\n"
        "        robot: RobotState, now: float, was_busy: bool\n"
        "    ) -> None:\n",
        1,
    )
    source = fast._patch_fast(source)
    module_name = "benchmark5_loss_fast_v2_target"
    module = types.ModuleType(module_name)
    module.__file__ = str(corrected.TARGET)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(corrected.TARGET), "exec"), module.__dict__)
    return module.__dict__


def main() -> None:
    namespace = _load_namespace()
    namespace["main"]()


if __name__ == "__main__":
    main()
