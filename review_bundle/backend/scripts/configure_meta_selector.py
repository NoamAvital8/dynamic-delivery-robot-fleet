"""Write a variant of a fitted selector with different switching settings.

Used to tune the switching rule ONLINE: create a few variants, run each in
`generate_meta_training_data.py --mode validate`, compare actual episode loss,
then freeze one variant for the final evaluation. The fitted regressors and
provenance are copied unchanged; the parent model's checksum is recorded.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-gain-fraction", type=float)
    parser.add_argument("--margin-scale", type=float)
    parser.add_argument("--adaptive-holding", choices=("on", "off"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from delivery_fleet.meta_policy.artifacts import atomic_json, digest
    from delivery_fleet.meta_policy.learning import LossSelector
    if args.output.exists():
        parser.error("output exists; choose a new versioned selector filename")
    selector = copy.deepcopy(LossSelector.load(args.selector))
    changes = {}
    for name, value in (("minimum_gain_fraction", args.minimum_gain_fraction),
                        ("margin_scale", args.margin_scale)):
        if value is not None:
            if not math.isfinite(value) or value < 0:
                parser.error(f"{name} must be finite and non-negative")
            setattr(selector, name, value)
            changes[name] = value
    if args.adaptive_holding is not None:
        if args.adaptive_holding == "on" and len(selector.windows) < 2:
            parser.error("adaptive holding needs a selector fitted on several holding windows")
        selector.adaptive = args.adaptive_holding == "on"
        changes["adaptive"] = selector.adaptive
    if not changes:
        parser.error("nothing to change")
    selector.provenance = {**(selector.provenance or {}),
                           "configured_from_sha256": digest(args.selector), "configuration": changes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selector.save(args.output)
    atomic_json(args.output.with_suffix(".configuration.json"),
                {"parent": str(args.selector.resolve()), "parent_sha256": digest(args.selector),
                 "changes": changes, "model_sha256": digest(args.output)})
    print(f"saved {args.output} with {changes}")


if __name__ == "__main__":
    main()
