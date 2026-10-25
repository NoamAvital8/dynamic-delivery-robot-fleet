"""Planning is read-only; every data-generation/training/evaluation action is opt-in."""
from __future__ import annotations

import argparse
import json

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan","prepare","train","prepare-test","evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true", help="Required for all commands except read-only plan")
    parser.add_argument("--cpus", type=int, default=72, help="Read-only planning assumption")
    parser.add_argument("--available-gib", type=float, default=171., help="Read-only planning assumption")
    args = parser.parse_args(argv)
    if args.command != "plan" and not args.execute:
        parser.error("No work started: this command requires explicit --execute")
    config = Config.load(args.config)
    if args.command == "plan":
        from .workflow import estimate
        config.maps("train")  # Inspect roles only; never open held-out graph files.
        result = estimate(config,args.cpus,args.available_gib)
    elif args.command in ("prepare","prepare-test"):
        from .data import prepare
        result = prepare(config,test=args.command=="prepare-test")
        result = {"episodes_prepared":len(result["episodes"]),"simulations_started":0}
    elif args.command == "train":
        from .workflow import train
        result = train(config)
    else:
        from .workflow import evaluate
        result = evaluate(config)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
