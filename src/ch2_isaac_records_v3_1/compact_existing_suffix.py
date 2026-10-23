#!/usr/bin/env python3
"""Copy one finished suffix into a delta-encoded directory without changing it."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from compact_trace import FORMAT, DeltaTrajectoryWriter, compact_file_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        first = stream.readline()
        if not first:
            raise ValueError("source trajectory is empty")
        header = json.loads(first)
        if header.get("format") == FORMAT:
            raise ValueError("source trajectory is already delta encoded")
        yield header
        for line in stream:
            if line.strip():
                yield json.loads(line)


def convert_suffix(source_dir: Path, output_dir: Path) -> dict:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    raw_path = next((path for path in (source_dir / "trajectory.jsonl.gz",
                                       source_dir / "trajectory.jsonl") if path.is_file()), None)
    if raw_path is None:
        raise FileNotFoundError("source suffix has no trajectory JSONL")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_trace = output_dir / "trajectory.jsonl.gz"
    writer = DeltaTrajectoryWriter(output_trace)
    try:
        for row in source_rows(raw_path):
            writer.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
    finally:
        writer.close()
    info = compact_file_info(output_trace, writer.row_count,
                             writer.reconstructed_sha256.hexdigest())
    for name in ("run_receipt.json", "manifest.json", "events.jsonl"):
        shutil.copy2(source_dir / name, output_dir / name)
    source_bytes = raw_path.stat().st_size
    receipt_path = output_dir / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["trajectory_encoding"] = FORMAT
    receipt["trajectory_storage"] = info
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trajectory_encoding"] = FORMAT
    manifest["trajectory_storage"] = info
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {"source_suffix": str(source_dir), "source_trajectory": raw_path.name,
                "source_trajectory_bytes": source_bytes,
                "source_trajectory_sha256": sha256(raw_path),
                "trajectory_storage": info,
                "byte_reduction_fraction": 1 - info["stored_bytes"] / source_bytes}
    (output_dir / "conversion_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = convert_suffix(args.suffix_dir, args.output)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
