#!/usr/bin/env python3
"""Acquire only registry scenes that never started in the previous batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    return env


def run(command: list[str]) -> tuple[int, float, str | None]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, check=False, env=environment())
        return result.returncode, time.perf_counter() - started, None
    except Exception as exc:
        return 2, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def select_scenes(rows: list[dict], old_output_root: Path):
    remaining = []
    references = []
    for row in rows:
        previous_dir = old_output_root / row["output_path"]
        receipt_path = previous_dir / "scene_receipt.json"
        if previous_dir.exists():
            receipt = None
            if receipt_path.is_file():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            references.append({"run_id": row["run_id"], "output_path": row["output_path"],
                               "status": (receipt or {}).get("execution", "EXISTING_NO_FINAL_RECEIPT"),
                               "receipt_sha256": sha256(receipt_path) if receipt_path.is_file() else None})
        else:
            remaining.append(row)
    return remaining, references


def acquire_scene(row: dict, source_root: Path, output_root: Path,
                  collector_root: Path, evaluator_path: Path) -> dict:
    start_wall = time.perf_counter()
    scene_dir = output_root / row["output_path"]
    scene_dir.mkdir(parents=True, exist_ok=False)
    result = {key: row.get(key) for key in
              ("run_id", "scenario_id", "split", "condition", "source", "seed", "parent_family",
               "output_path", "config_sha256")}
    result.update({"execution": "FAILED", "states": [], "error": None})
    representative = scene_dir / "representative"
    code, elapsed, error = run([sys.executable, str(ROOT / "run_capture.py"),
                                "--config", str(row["config_file"]),
                                "--source-root", str(source_root),
                                "--output", str(representative), "--phase", "representative"])
    receipt_path = representative / "run_receipt.json"
    capture_receipt = (json.loads(receipt_path.read_text(encoding="utf-8"))
                       if receipt_path.is_file() else {})
    result["representative"] = {"return_code": code, "wall_duration_s": elapsed,
                                "error": error, "status": capture_receipt.get("status"),
                                "trajectory_encoding": capture_receipt.get("trajectory_storage", {}).get("format")}
    if code != 0 or capture_receipt.get("status") != "complete":
        result["error"] = "representative capture failed; suffix collection stopped"
    else:
        index_path = representative / "state_sampling" / "state_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        states = index["states"][:8]
        for position, entry in enumerate(states):
            state_id = Path(entry["private_file"]).stem
            pair_dir = scene_dir / "suffixes" / state_id
            code, elapsed, error = run([sys.executable, str(ROOT / "run_suffix_pairs.py"),
                                        "--config", str(row["config_file"]),
                                        "--source-root", str(source_root),
                                        "--state-index", str(index_path), "--state-id", state_id,
                                        "--output", str(pair_dir)])
            pair_receipt_path = pair_dir / "suffix_worker_receipt.json"
            pair_receipt = (json.loads(pair_receipt_path.read_text(encoding="utf-8"))
                            if pair_receipt_path.is_file() else {})
            state_result = {"state_id": state_id, "sim_time_s": int(entry["sim_time_s"]),
                            "return_code": code, "wall_duration_s": elapsed,
                            "worker_status": pair_receipt.get("status"), "suffixes": []}
            result["states"].append(state_result)
            if code != 0 or pair_receipt.get("status") != "complete":
                result["error"] = f"suffix workers failed at state {state_id}; scene stopped"
                result["not_started_state_ids"] = [Path(item["private_file"]).stem
                                                    for item in states[position + 1:]]
                break
            for suffix_seed in (730201, 730202):
                suffix_dir = pair_dir / f"suffix_{suffix_seed}"
                report_path = suffix_dir / "independent_gain_report.json"
                code, eval_elapsed, eval_error = run([
                    sys.executable, str(ROOT / "evaluate_suffix_light.py"),
                    "--suffix-dir", str(suffix_dir),
                    "--source-index", str(index_path), "--source-root", str(source_root),
                    "--collector-root", str(collector_root),
                    "--evaluator", str(evaluator_path),
                    "--output", str(report_path)])
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
                state_result["suffixes"].append({"seed": suffix_seed, "return_code": code,
                    "status": report.get("status"), "valid_mask": report.get("valid_mask"),
                    "wall_duration_s": eval_elapsed, "error": eval_error})
                if report.get("status") == "FAIL":
                    result["error"] = f"independent suffix evaluation failed at {state_id}/{suffix_seed}"
                    break
            if result["error"]:
                result["not_started_state_ids"] = [Path(item["private_file"]).stem
                                                    for item in states[position + 1:]]
                break
        if result["error"] is None and states:
            result["execution"] = "COMPLETE"
        elif result["error"] is None:
            result["error"] = "representative capture yielded no sampled states"
    result["wall_duration_s"] = time.perf_counter() - start_wall
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (scene_dir / "scene_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_remaining(registry_path: Path, source_root: Path, old_output_root: Path,
                  new_output_root: Path) -> int:
    registry_path = registry_path.resolve()
    collector_root = registry_path.parent
    sys.path.insert(0, str(collector_root))
    from run_formal_batch import validate_registry

    registry, rows = validate_registry(registry_path)
    evaluator_path = registry_path.parent.parent / "formal_evidence_v1" / "evaluate_suffix.py"
    old_output_root = old_output_root.resolve()
    new_output_root = new_output_root.resolve()
    selected, old_references = select_scenes(rows, old_output_root)
    new_output_root.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "lightweight_remaining_collection_v1", "status": "running",
               "registry_id": registry["registry_id"], "registry_file": str(registry_path.resolve()),
               "registry_sha256": sha256(registry_path.resolve()), "registered_scene_count": len(rows),
               "old_output_root": str(old_output_root), "old_scene_references": old_references,
               "selected_new_scene_ids": [row["run_id"] for row in selected],
               "selected_scene_count": len(selected), "completed_new_scenes": [],
               "end_reduction": {"status": "end_reduction_pending",
                                  "reason": "final union of old raw and new delta suffixes must be reduced together"},
               "retry_policy": "existing old scene directories are never rerun; failed new scene directories are never retried"}
    receipt_path = new_output_root / "batch_receipt.json"
    receipt["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in selected:
        try:
            scene = acquire_scene(row, source_root.resolve(), new_output_root,
                                  collector_root, evaluator_path)
        except Exception as exc:
            scene_dir = new_output_root / row["output_path"]
            scene = {"run_id": row["run_id"], "output_path": row["output_path"],
                     "execution": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            if scene_dir.is_dir():
                (scene_dir / "scene_receipt.json").write_text(
                    json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["completed_new_scenes"].append(scene)
        if scene["execution"] != "COMPLETE":
            receipt.setdefault("failed_new_scene_ids", []).append(scene["run_id"])
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt.get("failed_new_scene_ids"):
        receipt["status"] = "partial_failed_pending_union_reduction"
    else:
        receipt["status"] = "collection_complete_pending_union_reduction"
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "collection_complete_pending_union_reduction" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-output-root", type=Path, required=True)
    parser.add_argument("--new-output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_remaining(args.registry, args.source_root,
                                   args.old_output_root, args.new_output_root))


if __name__ == "__main__":
    main()
