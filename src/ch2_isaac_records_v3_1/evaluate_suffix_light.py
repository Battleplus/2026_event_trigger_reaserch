#!/usr/bin/env python3
"""Run the frozen suffix evaluator over a losslessly delta-encoded trace."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import shutil
import tempfile
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


def _copy_state_index(source_index: Path, destination: Path) -> Path:
    index = json.loads(source_index.read_text(encoding="utf-8"))
    target_dir = destination / "labels"
    snapshots_dir = target_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)
    for entry in index.get("states", ()):
        for key in ("private_file", "public_file"):
            relative = entry.get(key)
            if relative:
                source = source_index.parent / relative
                target = target_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    target_index = target_dir / "state_index.json"
    shutil.copy2(source_index, target_index)
    return target_index


def _legacy_evaluator_inputs(suffix_dir: Path, source_index: Path, temp_root: Path):
    suffix_receipt = json.loads((suffix_dir / "run_receipt.json").read_text(encoding="utf-8"))
    suffix_manifest = json.loads((suffix_dir / "manifest.json").read_text(encoding="utf-8"))
    suffix_meta = suffix_receipt.get("suffix", {})
    source_metadata = dict(suffix_meta.get("source_manifest_metadata") or {})
    config = suffix_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("suffix manifest has no source run config for evaluator adaptation")
    internal_config_hash = (suffix_meta.get("source_manifest_config_sha256")
                            or suffix_meta.get("source_config_sha256")
                            or suffix_manifest.get("config_sha256"))
    source_manifest = dict(source_metadata)
    source_manifest.update({"run_id": suffix_manifest["run_id"],
                            "config": config,
                            "config_sha256": internal_config_hash})
    scene_dir = temp_root / "scene"
    scene_dir.mkdir(parents=True)
    scene_index = _copy_state_index(source_index, scene_dir)
    truth_source = source_index.parent / "evaluator_truth.json"
    truth_path = scene_dir / "evaluator_truth.json"
    shutil.copy2(truth_source, truth_path)
    source_manifest_path = scene_dir / "manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    source_manifest_hash = sha256(source_manifest_path)
    source_receipt = {"run_id": source_manifest["run_id"], "status": "complete",
                      "actual_isaac_run": True, "config_sha256": internal_config_hash,
                      "evaluator_truth_sha256": sha256(truth_path),
                      "state_sampling": json.loads(scene_index.read_text(encoding="utf-8"))}
    (scene_dir / "run_receipt.json").write_text(
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    temp_suffix = temp_root / "suffix"
    temp_suffix.mkdir(parents=True)
    for name in ("trajectory.jsonl.gz", "trajectory.jsonl", "events.jsonl"):
        source = suffix_dir / name
        if source.is_file():
            shutil.copy2(source, temp_suffix / name)
    suffix_manifest["config_sha256"] = internal_config_hash
    (temp_suffix / "manifest.json").write_text(
        json.dumps(suffix_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suffix_meta.update({"source_manifest": str(source_manifest_path.resolve()),
                        "source_manifest_sha256": source_manifest_hash,
                        "source_manifest_metadata": source_manifest,
                        "source_config_sha256": internal_config_hash,
                        "source_manifest_config_sha256": internal_config_hash,
                        "state_index": str(scene_index.resolve()),
                        "state_index_sha256": sha256(scene_index)})
    state_id = str(suffix_meta.get("state_id", ""))
    entry = next((row for row in json.loads(scene_index.read_text(encoding="utf-8"))["states"]
                  if row.get("state_id") == state_id), None)
    if entry is None:
        raise ValueError(f"source state index does not contain suffix state {state_id!r}")
    suffix_meta["private_snapshot"] = str((scene_index.parent / entry["private_file"]).resolve())
    suffix_meta["private_snapshot_sha256"] = entry["private_sha256"]
    suffix_receipt_path = temp_suffix / "run_receipt.json"
    suffix_receipt_path.write_text(
        json.dumps(suffix_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return temp_suffix, scene_index


def evaluate_suffix(suffix_dir: Path, source_index: Path, source_root: Path,
                    collector_root: Path, evaluator_path: Path) -> dict:
    evaluator = load_evaluator(evaluator_path)
    original_jsonl = evaluator._jsonl
    original_read_snapshot = evaluator._read_snapshot
    trace_path = suffix_dir / "trajectory.jsonl.gz"
    is_delta = _is_delta_trace(trace_path)
    trace_paths = {trace_path.resolve()}
    storage = {"format": FORMAT if is_delta else "raw_jsonl_v1",
               "stored_sha256": sha256(trace_path)}

    def read_snapshot(*args, **kwargs):
        backend, common, digest = original_read_snapshot(*args, **kwargs)
        return adapt_snapshot_backend(backend), common, digest

    def read_jsonl(path: Path) -> Iterator[dict]:
        if Path(path).resolve() in trace_paths:
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
    with tempfile.TemporaryDirectory(prefix="suffix-evaluator-") as temp_name:
        temp_root = Path(temp_name)
        source_scene = source_index.resolve().parent.parent
        legacy_inputs = (source_scene / "manifest.json",
                         source_scene / "run_receipt.json",
                         source_scene / "evaluator_truth.json")
        if all(path.is_file() for path in legacy_inputs):
            evaluator_suffix, evaluator_index = suffix_dir, source_index
        else:
            evaluator_suffix, evaluator_index = _legacy_evaluator_inputs(
                suffix_dir, source_index, temp_root)
            trace_paths.add((evaluator_suffix / trace_path.name).resolve())
        report = evaluator.evaluate_suffix(evaluator_suffix, evaluator_index,
                                           source_root, collector_root)
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
