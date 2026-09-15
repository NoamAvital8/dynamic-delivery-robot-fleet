from __future__ import annotations

"""Vectorized B5 candidate scoring with exact baseline loss.

V2 also approximated the *baseline* projection numerically.  Delta-loss scores
from different robots subtract different baselines, so even tiny baseline
projection differences can change the selected robot.  V3 keeps the cached
baseline evaluation on the original exact battery router and only vectorizes
candidate route scoring.  The selected winner is still materialized exactly.
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
    source = source.replace(
        "                    baseline_evaluation = evaluate_sequence_score(\n"
        "                        robot, snapshot, existing, order_states, None, {}\n"
        "                    )\n",
        "                    baseline_evaluation = evaluate_sequence(\n"
        "                        robot, snapshot, existing, order_states, None, {}\n"
        "                    )\n",
        1,
    )
    module_name = "benchmark5_loss_fast_v3_target"
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
