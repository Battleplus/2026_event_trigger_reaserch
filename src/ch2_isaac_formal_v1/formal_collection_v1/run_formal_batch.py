#!/usr/bin/env python3
"""Run the fixed scene registry serially and retain every scene receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "light_delay", "task_activation", "platform_fault", "energy_anomaly",
    "communication_outage", "environment_speed_block", "composite_all_enabled",
}
SOURCES = {"Existing-ET", "Periodic-4min", "Random-Allow-Suppress"}
MECHANISMS = {
    "light_delay", "task_activation", "platform_fault", "energy_anomaly",
    "communication_outage", "environment_speed_block",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def max_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker_environment():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "1"
    return env


def run_command(command: list[str]) -> dict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, check=False, env=worker_environment())
        return {"return_code": completed.returncode,
                "wall_duration_s": time.perf_counter() - started,
                "error": None}
    except Exception as exc:
        return {"return_code": None, "wall_duration_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}"}


def validate_registry(registry_path: Path) -> tuple[dict, list[dict]]:
    registry_path = registry_path.resolve()
    root = registry_path.parent
    sidecar_path = root / "formal_registry_sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if sidecar.get("registry_file") != registry_path.name:
        raise ValueError("registry SHA manifest names a different registry file")
    if sha(registry_path) != sidecar.get("registry_sha256"):
        raise ValueError("registry SHA256 differs from formal_registry_sha256.json")
    if registry.get("run_count") != 168 or len(registry.get("runs", [])) != 168:
        raise ValueError("fixed registry must contain exactly 168 scenes")
    if set(registry.get("conditions", [])) != CONDITIONS:
        raise ValueError("registry conditions differ from the frozen seven-condition design")
    if set(registry.get("sources", [])) != SOURCES:
        raise ValueError("registry sources differ from the frozen three-source design")
    paths = set()
    identities = set()
    rows = registry["runs"]
    counts = Counter((row["split"], row["condition"]) for row in rows)
    source_counts = Counter((row["split"], row["condition"], row["source"]) for row in rows)
    prepared = []
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            raise ValueError(f"unknown split: {row['split']}")
        if row["condition"] not in CONDITIONS or row["source"] not in SOURCES:
            raise ValueError(f"unknown condition/source: {row['condition']} / {row['source']}")
        if row["run_id"] in identities or row["output_path"] in paths:
            raise ValueError("registry scene identity or output path is duplicated")
        identities.add(row["run_id"]); paths.add(row["output_path"])
        config_path = (root / row["config_path"]).resolve()
        try:
            config_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("config path escapes registry directory") from exc
        if not config_path.is_file() or sha(config_path) != row["config_sha256"]:
            raise ValueError(f"config is missing or changed: {row['config_path']}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        formal = config.get("formal", {})
        if (config.get("run_id") != row["run_id"] or config.get("seed") != row["seed"]
                or config.get("scenario_id") != row.get("scenario_id")
                or config.get("mode") != "search" or config.get("targets_per_zone") != 3
                or formal.get("split") != row["split"]
                or formal.get("scenario_id") != row.get("scenario_id")
                or formal.get("condition") != row["condition"]
                or formal.get("source") != row["source"]):
            raise ValueError(f"scene config identity differs from registry row {row['run_id']}")
        expected = (MECHANISMS if row["condition"] == "composite_all_enabled"
                    else {row["condition"]})
        actual = {name for name in MECHANISMS
                  if bool(config.get("mechanisms", {}).get(name, {}).get("enabled"))}
        if actual != expected:
            raise ValueError(f"mechanism switches differ from condition {row['condition']}")
        if config.get("runtime", {}).get("multi_gpu") is not False:
            raise ValueError("scene config must request one GPU only")
        prepared.append({**row, "config_file": config_path, "config": config})
    if any(count != 12 for count in counts.values()):
        raise ValueError("each condition must contain 12 train and 12 validation scenes")
    if any(source_counts[(split, condition, source)] != 4
           for split in ("train", "validation")
           for condition in CONDITIONS for source in SOURCES):
        raise ValueError("each source must contribute four scenes per condition and split")
    for rel_name, expected_hash in sidecar.get("files", {}).items():
        path = root / rel_name
        if not path.is_file() or sha(path) != expected_hash:
            raise ValueError(f"frozen file hash differs or is missing: {rel_name}")
    return registry, prepared


def resource_summary(start_wall: float, start_cpu: float, worker_resources: list[dict]) -> dict:
    cpu_values = [float(item["process_cpu_s"]) for item in worker_resources
                  if item.get("process_cpu_s") is not None]
    rss_values = [int(item["max_rss_bytes"]) for item in worker_resources
                  if item.get("max_rss_bytes") is not None]
    return {
        "wall_duration_s": time.perf_counter() - start_wall,
        "orchestrator_process_cpu_s": time.process_time() - start_cpu,
        "worker_process_cpu_s_total": sum(cpu_values),
        "max_worker_rss_bytes": max(rss_values) if rss_values else None,
        "orchestrator_max_rss_bytes": max_rss_bytes(),
        "gpu_worker_count": 1,
        "cpu_threads_per_worker": 1,
    }


def receipt_from(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_scene(row: dict, registry_root: Path, source_root: Path,
              output_root: Path) -> dict:
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    scene_dir = output_root / row["output_path"]
    result = {"run_id": row["run_id"], "split": row["split"],
              "seed": row["seed"], "scenario_id": row["scenario_id"],
              "parent_family": row["parent_family"],
              "condition": row["condition"], "source": row["source"],
              "config_path": str(row["config_file"]), "config_sha256": row["config_sha256"],
              "output_path": row["output_path"], "execution": "FAILED",
              "representative": None, "states": [], "error": None}
    worker_resources = []
    if scene_dir.exists():
        result["error"] = "scene output already exists; refusing to overwrite or retry"
        result["resource_usage"] = resource_summary(start_wall, start_cpu, worker_resources)
        return result
    scene_dir.mkdir(parents=True)
    representative_dir = scene_dir / "representative"
    command = [sys.executable, str(registry_root / "run_capture.py"),
               "--config", str(row["config_file"]), "--source-root", str(source_root),
               "--output", str(representative_dir), "--phase", "representative"]
    run_result = run_command(command)
    run_receipt = receipt_from(representative_dir / "run_receipt.json")
    if run_receipt and run_receipt.get("resource_usage"):
        worker_resources.append(run_receipt["resource_usage"])
    result["representative"] = {**run_result,
                                 "status": run_receipt.get("status") if run_receipt else None,
                                 "receipt": "representative/run_receipt.json",
                                 "trajectory": run_receipt.get("trajectory") if run_receipt else None}
    if (run_result["return_code"] != 0 or not run_receipt
            or run_receipt.get("status") != "complete"
            or run_receipt.get("actual_isaac_run") is not True):
        result["error"] = "90-minute representative acquisition failed; no suffix state was attempted"
        result["execution"] = "FAILED"
    else:
        index_path = representative_dir / "state_sampling" / "state_index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            states = index.get("states", [])[:8]
            if not states:
                result["execution"] = "FAILED"
                result["error"] = "representative acquisition produced no ordinary indexed states"
            else:
                result["execution"] = "COMPLETE"
            for state_position, entry in enumerate(states):
                state_id = Path(entry["private_file"]).stem
                state_output = scene_dir / "suffixes" / state_id
                suffix_command = [sys.executable, str(registry_root / "run_suffix_pairs.py"),
                                  "--config", str(row["config_file"]),
                                  "--source-root", str(source_root),
                                  "--state-index", str(index_path), "--state-id", state_id,
                                  "--output", str(state_output)]
                suffix_run = run_command(suffix_command)
                pair_receipt = receipt_from(state_output / "suffix_worker_receipt.json")
                suffix_status = (pair_receipt or {}).get("status")
                state_result = {"state_id": state_id, "sim_time_s": int(entry["sim_time_s"]),
                                "output_path": str(state_output.relative_to(scene_dir)).replace("\\", "/"),
                                "status": suffix_status or ("complete" if suffix_run["return_code"] == 0 else "failed"),
                                "return_code": suffix_run["return_code"],
                                "worker_receipt": str((state_output / "suffix_worker_receipt.json").relative_to(scene_dir)).replace("\\", "/"),
                                "workers": (pair_receipt or {}).get("workers", [])}
                result["states"].append(state_result)
                if suffix_run["return_code"] != 0 or suffix_status != "complete":
                    result["execution"] = "PARTIAL_FAILED"
                    result["error"] = "suffix failure stopped the rest of this scene; completed raw artifacts were retained"
                for seed in (730201, 730202):
                    suffix_receipt = receipt_from(state_output / f"suffix_{seed}" / "run_receipt.json")
                    if suffix_receipt and suffix_receipt.get("resource_usage"):
                        worker_resources.append(suffix_receipt["resource_usage"])
                if suffix_run["return_code"] != 0 or suffix_status != "complete":
                    result["not_started_state_ids"] = [
                        Path(item["private_file"]).stem for item in states[state_position + 1:]
                    ]
                    break
        except Exception as exc:
            result["execution"] = "FAILED"
            result["error"] = f"state index or suffix launch failed: {type(exc).__name__}: {exc}"
    result["state_count"] = len(result["states"])
    result["resource_usage"] = resource_summary(start_wall, start_cpu, worker_resources)
    try:
        write_json(scene_dir / "scene_receipt.json", result)
    except OSError as exc:
        result["receipt_write_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_end_reduction(registry_root: Path, output_root: Path, source_root: Path,
                      rows: list[dict]) -> dict:
    evidence_root = registry_root.parent / "formal_evidence_v1"
    evaluator = evidence_root / "evaluate_suffix.py"
    aggregator = evidence_root / "aggregate.py"
    if not evaluator.is_file() or not aggregator.is_file():
        return {"status": "unavailable", "reason": "formal_evidence_v1 evaluator/aggregator not found",
                "reduction_count": 0}
    reductions_root = output_root / "reductions"
    scene_rows = {row["output_path"].split("/")[-1]: row for row in rows}
    calls = []
    for scene_dir in sorted((output_root / "scenes").glob("*")):
        representative = scene_dir / "representative"
        source_index = representative / "state_sampling" / "state_index.json"
        if not source_index.is_file():
            continue
        for suffix_dir in sorted(scene_dir.glob("suffixes/*/suffix_*")):
            suffix_receipt = receipt_from(suffix_dir / "run_receipt.json")
            if not suffix_receipt or suffix_receipt.get("status") != "complete":
                continue
            state_parent = suffix_dir.parent.name
            seed_name = suffix_dir.name
            identity = scene_rows.get(scene_dir.name)
            if identity is None:
                continue
            reduction = reductions_root / scene_dir.name / state_parent / seed_name / "reduction.json"
            command = [sys.executable, str(evaluator), "--suffix-dir", str(suffix_dir),
                       "--source-index", str(source_index), "--source-root", str(source_root),
                       "--collector-root", str(registry_root),
                       "--output", str(reduction)]
            result = run_command(command)
            calls.append({"scenario": identity["run_id"], "scenario_id": identity["scenario_id"],
                          "split": identity["split"], "condition": identity["condition"],
                          "parent_family": identity["parent_family"], "state_id": state_parent,
                          "suffix_seed": seed_name, "return_code": result["return_code"],
                          "output": str(reduction), "wall_duration_s": result["wall_duration_s"],
                          "error": result["error"]})
    aggregate_output = output_root / "aggregate.json"
    aggregate_result = run_command([sys.executable, str(aggregator), "--root", str(reductions_root),
                                    "--output", str(aggregate_output)])
    return {"status": "complete" if aggregate_result["return_code"] == 0
            and all(call["return_code"] == 0 for call in calls) else "failed",
            "reduction_count": len(calls), "reductions": calls,
            "aggregate_output": str(aggregate_output), "aggregate": aggregate_result}


def run_batch(registry_file: Path, source_root: Path, output_root: Path) -> int:
    registry, rows = validate_registry(registry_file)
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_receipt_path = output_root / "batch_receipt.json"
    progress_path = output_root / "batch_events.jsonl"
    if batch_receipt_path.exists() or progress_path.exists():
        raise FileExistsError("batch output already contains a receipt; refusing to restart or overwrite")
    batch_start = datetime.now(timezone.utc).isoformat()
    results = []
    for index, row in enumerate(rows, start=1):
        try:
            scene = run_scene(row, registry_file.resolve().parent, source_root, output_root)
        except Exception as exc:
            scene = {"run_id": row["run_id"], "split": row["split"],
                     "seed": row["seed"], "scenario_id": row["scenario_id"],
                     "parent_family": row["parent_family"], "condition": row["condition"],
                     "source": row["source"], "output_path": row["output_path"],
                     "execution": "FAILED",
                     "error": f"{type(exc).__name__}: {exc}", "states": []}
        scene["ordinal"] = index
        results.append(scene)
        try:
            append_jsonl(progress_path, scene)
        except OSError as exc:
            print(f"progress receipt could not be appended: {type(exc).__name__}: {exc}", file=sys.stderr)
    reductions = run_end_reduction(registry_file.resolve().parent, output_root, source_root, rows)
    receipt = {
        "schema": "isaac_roi_formal_batch_receipt_v1",
        "registry_file": str(registry_file.resolve()),
        "registry_sha256": sha(registry_file.resolve()),
        "execution": "COMPLETE" if all(row["execution"] == "COMPLETE" for row in results) else "PARTIAL_OR_FAILED",
        "started_at_utc": batch_start,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(results),
        "status_counts": dict(Counter(row["execution"] for row in results)),
        "retry_policy": "none; no scene is retried or replaced",
        "representative_phase_is_acquisition_episode": True,
        "registry_design_status": registry.get("design_status"),
        "scenes": results,
        "end_reduction": reductions,
    }
    write_json(batch_receipt_path, receipt)
    return 0 if receipt["execution"] == "COMPLETE" and reductions.get("status") == "complete" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-file", type=Path, required=True,
                        help="path to the frozen formal_batch_registry.json")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="trusted Isaac source tree required by the scene backend")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new empty output directory for raw trajectories and receipts")
    args = parser.parse_args()
    raise SystemExit(run_batch(args.registry_file, args.source_root, args.output_root))


if __name__ == "__main__":
    main()
