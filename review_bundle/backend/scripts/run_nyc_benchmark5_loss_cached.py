from __future__ import annotations

"""Corrected B5 with exact direct fast paths and charger-query memoization."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from delivery_fleet.routing_query_cache import install_charger_distance_query_cache
import run_nyc_benchmark5_loss_direct_fast as direct_fast


def main() -> None:
    install_charger_distance_query_cache()
    namespace = direct_fast._load_namespace()
    namespace["main"]()


if __name__ == "__main__":
    main()
