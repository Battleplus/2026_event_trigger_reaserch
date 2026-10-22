#!/usr/bin/env python3
"""Run the frozen suffix evaluator over a losslessly delta-encoded trace."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterator

from compact_trace import FORMAT, iter_lossless_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_suffix_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_snapshot_backend(backend: dict) -> dict:
    """Expose Isaac MotionState.position at the frozen evaluator's read boundary."""
    for platform in backend["platforms"].values():
        motion_state = platform["motion_model_state"]
        if "position" not in motion_state:
            motion_state["position"] = copy.deepcopy(motion_state["state"].position)
    return backend


def _is_delta_trace(path: Path) -> bool:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
    return header.get("format") == FORMAT


def iter_trace_rows(path: Path, raw_jsonl):
    if _is_delta_trace(path):
        yield from iter_lossless_rows(path)
    else:
        yield from raw_jsonl(path)


def evaluate_suffix(suffix_dir: Path, source_index: Path, source_root: Path,
                    collector_root: Path, evaluator_path: Path) -> dict:
    evaluator = load_evaluator(evaluator_path)
    original_jsonl = evaluator._jsonl
    original_read_snapshot = evaluator._read_snapshot
    trace_path = suffix_dir / "trajectory.jsonl.gz"
    is_delta = _is_delta_trace(trace_path)
    storage = {"format": FORMAT if is_delta else "raw_jsonl_v1",
               "stored_sha256": sha256(trace_path)}

    def read_snapshot(*args, **kwargs):
        backend, common, digest = original_read_snapshot(*args, **kwargs)
        return adapt_snapshot_backend(backend), common, digest

    def read_jsonl(path: Path) -> Iterator[dict]:
        if Path(path).resolve() == trace_path.resolve():
            rows = iter_trace_rows(path, original_jsonl)
            count = 0
            digest = hashlib.sha256()
            for row in rows:
                digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n")
                count += 1
                yield row
            storage["decoded_rows"] = count
            storage["decoded_sha256"] = digest.hexdigest()
            return
        yield from original_jsonl(path)

    evaluator._jsonl = read_jsonl
    evaluator._read_snapshot = read_snapshot
    report = evaluator.evaluate_suffix(suffix_dir, source_index, source_root, collector_root)
    report["trajectory_storage"] = storage
    report["snapshot_io_adapter"] = "motion_model_state.state.position_to_position_v1"
    expected = {}
    for source in (suffix_dir / "run_receipt.json", suffix_dir / "manifest.json"):
        if source.is_file():
            metadata = json.loads(source.read_text(encoding="utf-8"))
            expected.update(metadata.get("trajectory_storage", {}))
    if is_delta and expected and "decoded_rows" in storage:
        if (int(expected["reconstructed_row_count"]) != storage["decoded_rows"]
                or expected["reconstructed_rows_sha256"] != storage["decoded_sha256"]):
            report.update({"status": "FAIL", "valid_mask": False})
            report.setdefault("reasons", []).append("delta_trajectory_reconstruction_hash_mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True,
                        help="unchanged formal_evidence_v1/evaluate_suffix.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_suffix(args.suffix_dir, args.source_index, args.source_root,
                             args.collector_root, args.evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report.get("status"),
                      "valid_mask": report.get("valid_mask"),
                      "trajectory_storage": report.get("trajectory_storage"),
                      "output": str(args.output)}, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
