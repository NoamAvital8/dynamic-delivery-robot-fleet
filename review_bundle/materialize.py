"""Verify the Git review bundle; optionally inflate maps and write a local config.

No simulator, scenario generator, model or trainer is imported. Without
--execute this only reads and verifies existing files. Paths outside the
requested output directory are never written.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "review_bundle"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify():
    provenance = read(BUNDLE / "PROVENANCE.json")
    for name, expected in provenance["source_files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(BUNDLE.resolve()) or sha(path) != expected:
            raise ValueError("Export checksum mismatch: " + name)
    current = BUNDLE / "CURRENT_BUILD_SOURCES.json"
    sources = read(current if current.is_file() else BUNDLE / "evidence/build/BUILD_SOURCES.json")
    overrides = {"config.json":"configs/small_city_switching_v2.json",
                 "README.md":"docs/small_city_switching_v2.md",
                 "requirements.txt":"requirements-switching.txt"}
    for name, expected in sources["files"].items():
        relative = (name if sources.get("format") == "repository_paths_v1" else
                    name.removeprefix("code/") if name.startswith("code/") else overrides.get(name,name))
        if sha(ROOT / relative) != expected:
            raise ValueError("Current implementation source checksum mismatch: " + relative)
    pilot_manifest = BUNDLE / "evidence/switching_pilot_20261025/PILOT_EVIDENCE_SOURCES.json"
    pilot_files = read(pilot_manifest)["files"] if pilot_manifest.exists() else {}
    for name, expected in pilot_files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(pilot_manifest.parent.resolve()) or sha(path) != expected:
            raise ValueError("Pilot evidence checksum mismatch: " + name)
    launch_manifest = BUNDLE / "evidence/full_development_launch_20261025/LAUNCH_EVIDENCE_SOURCES.json"
    launch_files = read(launch_manifest)["files"] if launch_manifest.exists() else {}
    for name, expected in launch_files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(launch_manifest.parent.resolve()) or sha(path) != expected:
            raise ValueError("Full-development launch checksum mismatch: " + name)
    metadata = read(BUNDLE / "data/map_manifest.json")
    return {"status":"verified","backend_commit":provenance["backend_commit"],
            "exported_files_verified":len(provenance["source_files"]),
            "new_build_files_verified":len(sources["files"]),"maps":len(metadata["maps"]),
            "pilot_evidence_files_verified":len(pilot_files),
            "full_development_launch_files_verified":len(launch_files),
            "compressed_map_bytes":sum(r["compressed_bytes"] for r in metadata["maps"]),
            "inflated_map_bytes":sum(r["uncompressed_bytes"] for r in metadata["maps"]),
            "training_started":False,"simulations_started":0}


def materialize(output):
    output = Path(output).resolve()
    metadata = read(BUNDLE / "data/map_manifest.json")
    maps = output / "maps"
    maps.mkdir(parents=True,exist_ok=True)
    for row in metadata["maps"]:
        target = (maps / (row["key"] + ".graphml")).resolve()
        if target.parent != maps.resolve():
            raise ValueError("Invalid map key")
        if target.exists():
            if sha(target) != row["sha256"]:
                raise ValueError("Preserve existing differing map: " + str(target))
        else:
            temp = target.with_suffix(".partial")
            with gzip.open(ROOT / row["compressed_path"],"rb") as source,temp.open("wb") as dest:
                for block in iter(lambda:source.read(1024*1024),b""):
                    dest.write(block)
            if sha(temp) != row["sha256"]:
                raise ValueError("Inflated map checksum mismatch")
            temp.replace(target)
        row["path"] = str(target)
        if row.get("boundary_path"):
            row["boundary_path"] = str(BUNDLE / "data/boundaries" / (row["key"] + ".geojson"))
    manifest_path = output / "map_manifest.json"
    manifest_path.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    config = read(ROOT / "configs/small_city_switching_v2.json")
    config.update(backend_root=str(BUNDLE / "backend"),map_manifest=str(manifest_path),
                  output=str(output / "run"),dependency_dir="",
                  spatial_model=str(BUNDLE / "data/base_models/spatial/reservation_fcnn.npz"),
                  paper_model=str(BUNDLE / "data/base_models/paper/reservation_fcnn.npz"))
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    return {"config":str(config_path),"map_manifest":str(manifest_path),
            "maps_materialized":len(metadata["maps"]),"training_started":False,"simulations_started":0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true",help="Only decompress data and write portable paths; never launches training")
    parser.add_argument("--output-dir",default=str(ROOT / ".review-work"))
    args = parser.parse_args(argv)
    result = verify()
    if args.execute:
        result["materialized"] = materialize(args.output_dir)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
