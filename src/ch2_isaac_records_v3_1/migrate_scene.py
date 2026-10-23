#!/usr/bin/env python3
"""Convert one existing scene into the frozen lightweight scene layout.

The source scene is never modified. A public legacy capture is converted when
present; a scene without one receives only an explicit unavailable manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from export_records import export_capture


def migrate_scene(source: Path, output: Path, receipt: Path | None = None) -> dict:
    source = source.resolve()
    output = output.resolve()
    if (source / "config.json").is_file() and (source / "records.jsonl").is_file():
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / "config.json", output / "config.json")
        shutil.copyfile(source / "records.jsonl", output / "records.jsonl")
        if (source / "manifest.json").is_file():
            shutil.copyfile(source / "manifest.json", output / "manifest.json")
        if (source / "labels").is_dir():
            shutil.copytree(source / "labels", output / "labels")
        return {"status": "already_frozen", "source": str(source), "output": str(output)}

    candidates = [source, source / "public_capture", source / "representative" / "public_capture"]
    legacy = next((path for path in candidates
                   if (path / "static_config.json").is_file()
                   and (path / "ticks.jsonl.gz").is_file()), None)
    if legacy is not None:
        if receipt is None:
            receipt = next((path for path in (source / "representative" / "run_receipt.json",
                                              source / "run_receipt.json") if path.is_file()), None)
        result = export_capture(legacy, output, receipt)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({"schema": "scene_manifest_v1", "status": "converted_legacy",
                    "source_scene": str(source), "legacy_public_capture": str(legacy),
                    "scene_files": {"config": "config.json", "records": "records.jsonl",
                                    "manifest": "manifest.json", "labels": "labels"}})
        # Preserve existing sampled evidence, without copying whole-scene logs.
        old_index = next((path for path in (
            source / "representative" / "state_sampling" / "state_index.json",
            source / "state_sampling" / "state_index.json") if path.is_file()), None)
        if old_index is not None:
            index = json.loads(old_index.read_text(encoding="utf-8"))
            snapshots = output / "labels" / "snapshots"
            for entry in index["states"]:
                for key in ("private_file", "public_file"):
                    original = old_index.parent / entry[key]
                    destination = snapshots / entry[key]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, destination)
                    entry[key] = destination.relative_to(output / "labels").as_posix()
            index["source_state_index_sha256"] = hashlib.sha256(old_index.read_bytes()).hexdigest()
            index_path = output / "labels" / "state_index.json"
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest["labels_state_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            manifest["labels"] = index
        if (source / "suffixes").is_dir():
            manifest["existing_pair_evidence"] = {"path": str(source / "suffixes"),
                                                  "storage": "read_only_reference"}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        return {"status": "converted_legacy", "source": str(source), "output": str(output),
                "records": result}

    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "scene_manifest_v1", "status": "conversion_unavailable",
                "source_scene": str(source),
                "reason": "no_public_config_and_records_or_legacy_public_capture",
                "scene_files": {"config": None, "records": None,
                                "manifest": "manifest.json", "labels": None}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return {"status": "conversion_unavailable", "source": str(source), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_scene(args.source, args.output, args.receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
