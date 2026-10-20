"""Continue one interrupted V3.3 collection without changing frozen sources."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "civilian"))

import v3_runner as runner  # noqa: E402
import v3_shards as shards  # noqa: E402


SOURCE_RUN = "v33-collect-20261020T055710Z-8e3a1c"
WORKERS = 2


def _worker(task: tuple[dict[str, Any], int, str, dict[str, Any]]) -> dict[str, Any]:
    spec, ordinal, run_id, binding = task
    return runner._build_and_commit(
        spec, ordinal, output_dir=ROOT / "runs-v33", run_id=run_id,
        binding=binding, stage="collect",
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="ascii")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-job", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.one_job:
        task = json.loads(args.one_job.read_text(encoding="ascii"))
        result = _worker((task["spec"], task["ordinal"], task["run_id"], task["binding"]))
        print(json.dumps({"scenario_id": result["scenario_id"], "split": result["split"],
                          "condition": result["condition"]}, sort_keys=True), flush=True)
        return 0

    started = time.time()
    binding = shards.current_binding()
    registry = shards.validate_registry(shards.expected_registry(task_mode="search_rescue", targets_per_region=3))
    old = shards.verify_run(ROOT / "runs-v33", SOURCE_RUN, expected_binding=binding)
    factory = runner._scenario_factory()
    expected_ids: dict[str, tuple[int, dict[str, Any]]] = {}
    for ordinal, spec in enumerate(registry):
        scenario = factory(
            split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"],
            instance_index=spec["instance_index"], task_mode=spec["task_mode"],
            targets_per_region=spec["targets_per_region"],
        )
        expected_ids[scenario["parent_id"]] = (ordinal, spec)
    if any(row["scenario_id"] not in expected_ids for row in old["verified"]):
        raise ValueError("RESUME_SOURCE_SHARD_OUTSIDE_FROZEN_REGISTRY")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"v33-collect-resume-{stamp}-{uuid.uuid4().hex[:12]}"
    run_root = ROOT / "runs-v33" / run_id
    if run_root.exists():
        progress_path = run_root / "COLLECTION_PROGRESS.json"
        prior = json.loads(progress_path.read_text(encoding="ascii"))
        if prior.get("source_run_id") != SOURCE_RUN or prior.get("binding") != binding:
            raise ValueError("RESUME_EXISTING_RUN_IDENTITY_MISMATCH")
        reused = prior.get("reused_shards")
        if reused is None:
            reused = json.loads((run_root / "REUSED_SHARDS.json").read_text(encoding="ascii"))["shards"]
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        reused = []
        for marker in old["verified"]:
            source_marker = Path(marker["marker_path"])
            source_shard = source_marker.with_name(marker["shard_file"])
            target_dir = run_root / "shards" / marker["split"] / marker["condition"]
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_shard, target_dir / source_shard.name)
            shutil.copy2(source_marker, target_dir / source_marker.name)
            reused.append({
                "scenario_id": marker["scenario_id"], "split": marker["split"],
                "condition": marker["condition"], "source_run_id": SOURCE_RUN,
                "source_marker_sha256": marker["marker_sha256"],
                "compressed_sha256": marker["compressed_sha256"],
            })
        _write_json(run_root / "REUSED_SHARDS.json", {
            "schema": "gain-gate-v33-resume-lineage/1", "source_run_id": SOURCE_RUN,
            "source_verified_count": len(reused), "binding": binding, "shards": reused,
        })
    # Validate the byte-identical inherited files in their new run layout.
    inherited = shards.verify_run(ROOT / "runs-v33", run_id, expected_binding=binding)
    if inherited["committed_shards"] < len(reused):
        raise ValueError("RESUME_INHERITED_SHARD_COUNT_MISMATCH")

    completed_ids = {row["scenario_id"] for row in inherited["verified"]}
    pending = [(spec, ordinal, run_id, binding) for ordinal, spec in enumerate(registry)
               if expected_ids[next_id(ordinal, registry, factory)][1] == spec
               and next_id(ordinal, registry, factory) not in completed_ids]
    progress_path = run_root / "COLLECTION_PROGRESS.json"
    state: dict[str, Any] = json.loads(progress_path.read_text(encoding="ascii")) if progress_path.exists() else {
        "schema": "gain-gate-v33-collection-progress/1", "run_id": run_id,
        "source_run_id": SOURCE_RUN, "binding": binding, "total_scenarios": len(registry),
        "reused_committed": len(reused), "reused_shards": reused,
        "new_committed": 0, "failed_scenarios": [],
        "worker_count": WORKERS, "pending_at_start": len(pending), "started_unix": started,
    }
    state["worker_count"] = WORKERS
    state["new_committed"] = max(0, inherited["committed_shards"] - len(reused))
    state.setdefault("failed_scenarios", [])
    state["pending_at_start"] = len(pending)
    _write_json(progress_path, state)
    print(json.dumps({"event": "started", "run_id": run_id, "reused": len(reused),
                      "pending": len(pending), "workers": WORKERS}, sort_keys=True), flush=True)
    next_pending = 0
    active: dict[subprocess.Popen, tuple[dict[str, Any], Path, Path]] = {}
    while active or next_pending < len(pending):
        task = pending[next_pending] if next_pending < len(pending) and len(active) < WORKERS else None
        if task is not None:
            next_pending += 1
            spec, ordinal, _, _ = task
            job_path = run_root / "jobs" / f"registry-{ordinal:03d}.json"
            _write_json(job_path, {"spec": spec, "ordinal": ordinal, "run_id": run_id, "binding": binding})
            stdout_path = run_root / "jobs" / f"registry-{ordinal:03d}.stdout.log"
            stderr_path = run_root / "jobs" / f"registry-{ordinal:03d}.stderr.log"
            stdout_stream = stdout_path.open("ab")
            stderr_stream = stderr_path.open("ab")
            process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--one-job", str(job_path)],
                cwd=ROOT, stdout=stdout_stream, stderr=stderr_stream,
            )
            active[process] = (spec, stdout_path, stderr_path)
            process._resume_log_streams = (stdout_stream, stderr_stream)
            print(json.dumps({"event": "worker_started", "ordinal": ordinal, "pid": process.pid}, sort_keys=True), flush=True)
        finished = [process for process in active if process.poll() is not None]
        for process in finished:
            spec, stdout_path, stderr_path = active.pop(process)
            for stream in process._resume_log_streams:
                stream.close()
            scenario_id = next_id(next(i for i, row in enumerate(registry) if row == spec), registry, factory)
            if process.returncode == 0:
                state["new_committed"] += 1
                state.setdefault("completed_scenarios", []).append(scenario_id)
                event = {"event": "committed", "run_id": run_id,
                         "completed": len(reused) + state["new_committed"], "total": len(registry),
                         "scenario_id": scenario_id}
            else:
                failed = {"scenario_id": scenario_id, "split": spec["split"], "condition": spec["condition"],
                          "instance_index": spec["instance_index"], "exit_code": process.returncode,
                          "stderr_log": str(stderr_path.relative_to(ROOT))}
                state["failed_scenarios"].append(failed)
                event = {"event": "failed", "run_id": run_id, **failed}
            state["updated_unix"] = time.time()
            _write_json(progress_path, state)
            print(json.dumps(event, sort_keys=True), flush=True)
        if active:
            time.sleep(0.5)

    verified = shards.verify_run(ROOT / "runs-v33", run_id, expected_binding=binding)
    state["verified_committed"] = verified["committed_shards"]
    state["status"] = "complete" if verified["committed_shards"] == len(registry) and not state["failed_scenarios"] else "incomplete_with_failures"
    state["elapsed_seconds"] = time.time() - started
    state["updated_unix"] = time.time()
    _write_json(progress_path, state)
    print(json.dumps({"event": "finished", "run_id": run_id, "status": state["status"],
                      "verified_committed": state["verified_committed"],
                      "failed": len(state["failed_scenarios"]),
                      "elapsed_seconds": state["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0 if state["status"] == "complete" else 1


def next_id(ordinal: int, registry: list[dict[str, Any]], factory: Any) -> str:
    spec = registry[ordinal]
    scenario = factory(
        split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"],
        instance_index=spec["instance_index"], task_mode=spec["task_mode"],
        targets_per_region=spec["targets_per_region"],
    )
    return scenario["parent_id"]


if __name__ == "__main__":
    raise SystemExit(main())
