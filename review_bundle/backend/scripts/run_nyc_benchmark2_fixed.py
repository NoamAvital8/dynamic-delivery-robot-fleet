from __future__ import annotations

"""Run Benchmark 2 with a numerical battery tolerance suitable for float32 routing.

The dense charger-distance index stores distances as float32.  Splitting a quoted
segment into semantic runtime legs can therefore differ by sub-millimetre amounts,
which previously caused false negative-battery failures on the order of micro-Wh.
This runner applies a 1e-3 Wh runtime tolerance and then executes the existing
diagnostic Benchmark 2 runner so any genuine failure still gets a full state dump.
"""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "run_nyc_benchmark2.py"
DIAGNOSTIC = ROOT / "scripts" / "run_nyc_benchmark2_diagnostic.py"

OLD = "if robot.battery_wh < -1e-6:"
NEW = "if robot.battery_wh < -1e-3:"

source = TARGET.read_text(encoding="utf-8")
count = source.count(OLD)
if count != 1:
    raise RuntimeError(
        f"expected exactly one Benchmark 2 battery tolerance check, found {count}"
    )
TARGET.write_text(source.replace(OLD, NEW), encoding="utf-8")

runpy.run_path(str(DIAGNOSTIC), run_name="__main__")
