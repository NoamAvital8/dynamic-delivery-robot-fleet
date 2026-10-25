"""Create NEW meta training/validation scenarios, retaining old tests for comparison.

Creates scenario files only; no simulator or model training is started.
Pass --reference-suite several times to combine cities from several suites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_fleet.meta_policy import DEFAULT_EXCLUDED_CITIES, DEFAULT_PORTFOLIO, POLICIES
from delivery_fleet.meta_policy.fresh_scenarios import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-suite", "--suite", required=True, type=Path, action="append",
                        help="Existing benchmark suite: prepared maps and reserved scenarios. Repeatable.")
    parser.add_argument("--reserved-suite", action="append", type=Path, default=[],
                        help="Additional archived suites outside the reference repository.")
    parser.add_argument("--spatial-model", required=True, type=Path)
    parser.add_argument("--paper-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument("--train-per-city", type=int, default=20)
    parser.add_argument("--validation-per-city", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20261015)
    parser.add_argument("--holding-minutes", type=float, nargs="+", default=[15.])
    parser.add_argument("--cities", nargs="+", help="Explicit city list (overrides --exclude-cities).")
    parser.add_argument("--exclude-cities", nargs="*", default=list(DEFAULT_EXCLUDED_CITIES),
                        help="Suite cities to leave out (default: beijing sydney, too expensive to label).")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(DEFAULT_PORTFOLIO),
                        help="Candidate portfolio (default: the six complementary policies in meta_policy/__init__.py).")
    parser.add_argument("--continuation-policy", choices=POLICIES, default="full_no_idle",
                        help="Common policy after each candidate window. Validate the ranking in a pilot.")
    parser.add_argument("--initial-policy", choices=POLICIES, default="full_no_idle")
    parser.add_argument("--exploration-stickiness", type=float, default=.75,
                        help="Probability of keeping the previous behaviour policy at a bootstrap decision.")
    parser.add_argument("--fixed-behavior-fraction", type=float, default=.2,
                        help="Share of bootstrap episodes that follow one fixed policy throughout.")
    parser.add_argument("--label-step-probability", type=float, default=.25,
                        help="Share of decisions that get the full counterfactual tournament.")
    parser.add_argument("--decide-until", choices=("arrival_horizon", "all_delivered"), default="all_delivered")
    parser.add_argument("--scoring-horizon-minutes", type=float, default=60.)
    parser.add_argument("--controller-settings", type=json.loads, default=None,
                        help="JSON of non-default scientific controller settings (unknown keys are rejected).")
    parser.add_argument("--processes", type=int, default=60)
    parser.add_argument("--memory-per-worker-gib", type=float, default=4.)
    parser.add_argument("--memory-reserve-gib", type=float, default=16.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    value = prepare(args.reference_suite, args.spatial_model, args.paper_model, args.output,
        train_per_city=args.train_per_city, validation_per_city=args.validation_per_city,
        base_seed=args.base_seed, holding_minutes=args.holding_minutes, cities=args.cities,
        exclude_cities=args.exclude_cities,
        reserved_suites=args.reserved_suite, scenario_dir=args.scenario_dir, resume=args.resume,
        processes=args.processes, memory_per_worker_gib=args.memory_per_worker_gib,
        reserve_gib=args.memory_reserve_gib, policies=args.policies,
        continuation_policy=args.continuation_policy, initial_policy=args.initial_policy,
        exploration_stickiness=args.exploration_stickiness,
        fixed_behavior_fraction=args.fixed_behavior_fraction,
        label_step_probability=args.label_step_probability, decide_until=args.decide_until,
        scoring_horizon_minutes=args.scoring_horizon_minutes,
        controller_settings=args.controller_settings)
    print(f"saved {len(value['episodes'])} NEW training/validation scenarios; "
          "the separate *_comparison.json manifest retains original benchmark tests")


if __name__ == "__main__":
    main()
