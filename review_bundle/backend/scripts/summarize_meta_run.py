"""Read-only cost/progress summary of a meta collection or pilot output folder.

Reports, per city: episodes, decision steps, labelled steps, roll-outs, how
often a scoring tail was reused, mean window/tail wall time, peak memory, and
a rough per-episode extrapolation. The extrapolation assumes later steps cost
like the measured ones; overloaded cities usually get MORE expensive later, so
treat it as a lower bound until whole representative episodes have finished.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps-per-episode", type=float,
                        help="Expected decisions per episode for the extrapolation (default: measured maximum).")
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "run_manifest.json").read_text())
    config = manifest["configuration"]
    cities = {e["id"]: e["city"] for e in config["episodes"]}
    probability = float(config.get("label_step_probability", 1.))
    stats = defaultdict(lambda: defaultdict(list))
    for folder in sorted((args.output_dir / "episodes").glob("*")):
        city = cities.get(folder.name, "?")
        s = stats[city]
        status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
        s["episodes"].append(folder.name)
        s["status"].append(status.get("status", "unknown"))
        if status.get("peak_rss_gib") is not None:
            s["rss"].append(float(status["peak_rss_gib"]))
        decisions = [json.loads(p.read_text()) for p in sorted((folder / "decisions").glob("*.json"))]
        s["steps"].append(len(decisions))
        s["labelled"].append(sum(bool(d.get("labelled")) for d in decisions))
        for p in (folder / "rollouts").glob("*.json"):
            r = json.loads(p.read_text())
            s["window"].append(float(r.get("window_wall_seconds", r.get("wall_seconds", 0.))))
            s["tail"].append(float(r.get("tail_wall_seconds", 0.)))
            s["reused"].append(bool(r.get("tail_reused")))
    portfolio = len(config["policies"]) * len(config["holding_minutes"])
    report = {}
    for city, s in sorted(stats.items()):
        branch = (statistics.mean(s["window"]) + statistics.mean(s["tail"])) if s["window"] else math.nan
        steps = args.steps_per_episode or (max(s["steps"]) if s["steps"] else math.nan)
        report[city] = {
            "episodes": len(s["episodes"]), "status": dict((k, s["status"].count(k)) for k in set(s["status"])),
            "decision_steps": sum(s["steps"]), "labelled_steps": sum(s["labelled"]),
            "rollouts": len(s["window"]),
            "tail_reuse_rate": sum(s["reused"]) / len(s["reused"]) if s["reused"] else None,
            "mean_window_wall_s": statistics.mean(s["window"]) if s["window"] else None,
            "mean_tail_wall_s": statistics.mean(s["tail"]) if s["tail"] else None,
            "max_peak_rss_gib": max(s["rss"]) if s["rss"] else None,
            "rough_label_hours_per_episode": (steps * probability * portfolio * branch / 3600
                                              if math.isfinite(branch) else None),
        }
    print(json.dumps({"output": str(args.output_dir.resolve()), "label_step_probability": probability,
                      "branches_per_labelled_step": portfolio, "cities": report,
                      "caveat": "lower-bound extrapolation from measured steps only"}, indent=2))


if __name__ == "__main__":
    main()
