"""Immutable campaign, learner and replay content identities. No numeric imports."""
from __future__ import annotations

import importlib.metadata as metadata
import platform
from pathlib import Path

from . import LABEL_VERSION
from .config import DEVELOPMENT, HELD_OUT, identity, immutable, read, sha

SCIENTIFIC_SETTINGS = ("decision_minutes", "duration_hours", "base_seed", "train_per_city",
    "validation_per_city", "test_per_city", "bootstrap_policy", "max_post_horizon_minutes",
    "update_every_states", "policies")


def package_versions(config):
    names = {"numpy", "scipy", "networkx", "scikit-learn", "xgboost", "xgboost-cpu", "threadpoolctl"}
    result = {}
    for name in sorted(names):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    if config.dependency_dir:
        for package in metadata.distributions(path=[config.dependency_dir]):
            name = package.metadata["Name"].lower().replace("_", "-")
            if name in names:
                result[name] = package.version
    return result


def base_weights(config):
    return {"spatial": sha(config.spatial_model), "paper": sha(config.paper_model)}


def science_id(config):
    backend = Path(config.backend_root)
    paths = sorted(p for directory in ("src", "scripts", "operations")
                   for p in (backend / directory).rglob("*.py"))
    if not paths:
        raise ValueError("The preserved simulator backend is missing")
    own = Path(__file__).parent
    # Learner implementation has a separate identity; operational settings are
    # recorded, but changing a worker limit does not change label compatibility.
    sources = ("backend.py", "backend_lifecycle.py", "projection.py", "features.py",
               "collection.py", "data.py", "provenance.py", "config.py", "replay.py")
    return identity({"backend": {p.relative_to(backend).as_posix(): sha(p) for p in paths},
        "label_sources": {name: sha(own / name) for name in sources},
        "base_weights": base_weights(config), "versions": package_versions(config),
        "label": LABEL_VERSION, "decision_minutes": config.decision_minutes, "policies": config.policies})


def learner_id(config, family):
    own = Path(__file__).parent
    return identity({"family": family, "models": sha(own / "models.py"),
        "workflow": sha(own / "workflow.py"), "versions": package_versions(config),
        "label": LABEL_VERSION, "policies": config.policies})


def ensure_campaign(config, create=False):
    maps = read(config.map_manifest)["maps"]
    config.maps("train")
    value = {"science_id": science_id(config), "label_version": LABEL_VERSION,
        "settings": {key: list(config.policies) if key == "policies" else getattr(config, key)
                     for key in SCIENTIFIC_SETTINGS},
        "maps": sorted(({"city": row["key"], "role": row["role"], "sha256": row["sha256"]}
                         for row in maps), key=lambda row: row["city"]),
        "base_weights": base_weights(config), "development": sorted(DEVELOPMENT),
        "held_out": sorted(HELD_OUT),
        "actor_learners": {family:learner_id(config, family) for family in ("xgboost", "nn")},
        "implementation": {p.name:sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python":platform.python_version(), "platform":[platform.system(),platform.machine()]}
    value["campaign_id"] = identity(value)
    path = Path(config.output) / "campaign.json"
    if create:
        immutable(path, value)
    elif not path.exists() or read(path) != value:
        raise ValueError("Campaign identity changed or is missing; use a new output directory")
    return value


def require_development_open(config):
    if (Path(config.output) / "test_opened.json").exists():
        raise ValueError("Final test has been opened; development is immutable in this campaign")


def row_digest(row):
    return identity({key: value for key, value in row.items() if key != "content_sha256"})


def validate_row(row, episodes=None, science=None, family=None):
    if row.get("content_sha256") != row_digest(row):
        raise ValueError("Replay row content hash mismatch")
    if science is not None and row["science_id"] != science:
        raise ValueError("Replay scientific identity mismatch")
    if family is not None and row["family"] != family:
        raise ValueError("Replay learner family mismatch")
    if episodes is not None:
        episode = episodes.get(row["episode"])
        if episode is None or episode["split"] != "train":
            raise ValueError("Replay episode is absent from the active training manifest")
        for key in ("city", "split", "seed", "scenario_sha256", "graph_sha256", "workload_sha256"):
            if row[key] != episode[key]:
                raise ValueError("Replay input provenance mismatch: " + key)
    return row


def freeze_receipt(config):
    root = Path(config.output)
    value = read(root / "frozen_models.json")
    unsigned = {key: item for key, item in value.items() if key != "freeze_id"}
    if value.get("freeze_id") != identity(unsigned):
        raise ValueError("Frozen experiment receipt changed")
    campaign = ensure_campaign(config)
    if value["campaign_id"] != campaign["campaign_id"]:
        raise ValueError("Frozen campaign changed")
    if sha(root / "datasets/development_manifest.json") != value["development_manifest_sha256"]:
        raise ValueError("Frozen development dataset changed")
    if sha(root / "validation_summary.json") != value["validation_report_sha256"]:
        raise ValueError("Frozen development evidence changed")
    for pointer in (*value["models"].values(), *value["baselines"].values()):
        if sha(pointer["path"]) != pointer["sha256"] or pointer["science_id"] != campaign["science_id"]:
            raise ValueError("Frozen model or benchmark changed")
    return value
