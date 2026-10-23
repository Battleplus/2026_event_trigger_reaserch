#!/usr/bin/env python3
"""Migrate an archived scene tree without changing the source tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from migrate_scene import migrate_scene


def migrate_batch(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    scene_root = source_root / "scenes" if (source_root / "scenes").is_dir() else source_root
    scene_dirs = sorted(path for path in scene_root.iterdir() if path.is_dir())
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for source in scene_dirs:
        target = output_root / source.name
        results.append(migrate_scene(source, target, source / "run_receipt.json"))
    manifest = {"schema": "scene_batch_manifest_v1", "status": "complete",
                "source_root": str(source_root), "scene_count": len(results),
                "results": results}
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_batch(args.source_root, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
