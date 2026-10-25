"""Readable launcher; no default action and no launch on import."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
source = root / "code/src" if (root / "code/src").is_dir() else root / "src"
sys.path.insert(0, str(source))
from fleet_switching.cli import main

if __name__ == "__main__":
    main()
