from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_nyc_heuristic_policy as heuristic


def test_generated_heuristic_runner_contains_reservation_and_safe_fallback() -> None:
    source = heuristic.corrected._patch_source(
        heuristic.corrected.TARGET.read_text(encoding="utf-8")
    )
    generated = heuristic._patch_heuristic(source)
    compile(generated, "generated_heuristic_runner", "exec")
    assert "reservation_policy.observe" in generated
    assert "reservation_eligibility" in generated
    assert "shortlist_fallback_robots" in generated
    assert "minimum_exact_incremental_total_loss_with_safe_shortlist_expansion" in generated
