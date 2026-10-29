"""Complete matched comparisons; no simulator imports and no model fitting."""
from __future__ import annotations

from collections import defaultdict
import random
import math

from .config import POLICIES, identity, validate_policies
from .features import quantile

FIXED_PREFIX = "fixed__"


def mean(values):
    return sum(values) / len(values)


def summarize(results):
    by_family = defaultdict(list)
    for row in results:
        by_family[row["family"]].append(row)
    answer = {}
    for family, rows in sorted(by_family.items()):
        cities = defaultdict(list)
        for row in rows:
            cities[row["city"]].append(row)
        per_city = {city:{"episodes":len(group), "mean_loss":mean([r["loss"] for r in group]),
            "mean_loss_per_robot":mean([r["loss"] / r["robots"] for r in group])}
            for city, group in sorted(cities.items())}
        answer[family] = {"episodes":len(rows), "total_loss":sum(r["loss"] for r in rows),
            "equal_city_mean_loss_per_robot":mean([r["mean_loss_per_robot"] for r in per_city.values()]),
            "per_city":per_city}
    return answer


def _interval(rows, draws=500):
    """Stratified episode bootstrap, conditional on these cities, not all cities."""
    groups = defaultdict(list)
    for city, value in rows:
        groups[city].append(value)
    rng = random.Random(20261025)
    samples = [mean([mean(rng.choices(values, k=len(values))) for values in groups.values()]) for _ in range(draws)]
    return [quantile(samples, .025), quantile(samples, .975)]


def compare(results, episodes, best_fixed=None, *, expected_science=None, policies=POLICIES):
    policies = validate_policies(policies)
    expected_episodes = {row["id"]:row for row in episodes}
    if not episodes or len(expected_episodes) != len(episodes):
        raise ValueError("Comparison needs unique, nonempty input episodes")
    if best_fixed is None and any(row["split"] == "test" for row in episodes):
        raise ValueError("Select the deployable fixed baseline on development data, not test outcomes")
    families = ("xgboost", "nn", *(FIXED_PREFIX + policy for policy in policies))
    indexed = {}
    for row in results:
        key = row["episode"], row["family"]
        if key in indexed or row["episode"] not in expected_episodes or row["family"] not in families:
            raise ValueError("Duplicate, orphan or unknown comparison result")
        source = expected_episodes[row["episode"]]
        if (not math.isfinite(row["loss"]) or row["loss"] < 0 or row["robots"] < 1
                or row["episode_identity"] != identity(source)):
            raise ValueError("Invalid or scientifically unmatched comparison result")
        if expected_science is not None and row["science_id"] != expected_science:
            raise ValueError("Comparison scientific identity mismatch")
        if row["status"] != "complete" or row["delivered"] != source["orders"] or row["orders"] != source["orders"]:
            raise ValueError("Incomplete comparisons cannot be silently ranked")
        for field in ("city", "scenario_sha256", "graph_sha256"):
            if row[field] != source[field]:
                raise ValueError("Unmatched comparison input: " + field)
        if row["robots"] != source["fleet_count"]:
            raise ValueError("Unmatched comparison fleet")
        indexed[key] = row
    expected = {(episode, family) for episode in expected_episodes for family in families}
    if set(indexed) != expected:
        raise ValueError("All selector and configured fixed-policy outcomes are required for comparison")
    scores = summarize(results)
    if best_fixed is None:
        best_fixed = min(policies, key=lambda p:scores[FIXED_PREFIX+p]["equal_city_mean_loss_per_robot"])
    if best_fixed not in policies:
        raise ValueError("Unknown development-selected fixed baseline")
    paired = {}
    for family in ("xgboost", "nn"):
        paired[family] = {}
        for policy in (*policies, "hindsight_best_fixed_per_scenario"):
            differences, raw, wins, ties = [], [], 0, 0
            for episode, source in expected_episodes.items():
                actor = indexed[episode, family]
                baseline = (min(indexed[episode,FIXED_PREFIX+p]["loss"] for p in policies)
                            if policy == "hindsight_best_fixed_per_scenario" else indexed[episode,FIXED_PREFIX+policy]["loss"])
                delta = actor["loss"] - baseline
                raw.append(delta)
                differences.append((source["city"], delta / actor["robots"]))
                wins += int(delta < -1e-8)
                ties += int(abs(delta) <= 1e-8)
            cities = defaultdict(list)
            for city, value in differences:
                cities[city].append(value)
            paired[family][policy] = {"episodes":len(raw), "total_loss_difference":sum(raw),
                "equal_city_mean_loss_per_robot_difference":mean([mean(v) for v in cities.values()]),
                "stratified_episode_bootstrap_95pct_interval":_interval(differences),
                "wins":wins,"ties":ties,"losses":len(raw)-wins-ties}
    return {"scores":scores,"paired":paired,"policies":list(policies),"development_selected_fixed_policy":best_fixed,
        "selection_metric":"equal_city_mean_loss_per_robot", "completed_results":len(results),
        "missing_results":0,"failed_results":0,
        "interval_scope":"Episode resampling within the observed cities; not population-level geographic certainty",
        "difference_direction":"negative favors selector", "results":results}
