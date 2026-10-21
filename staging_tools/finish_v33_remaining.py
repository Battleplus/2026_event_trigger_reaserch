"""Finish uncommitted V3.3 scenarios one at a time, preserving the original run."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "v33-collect-resume-20261020T075751Z-cf8baae09a4a"
RUNS = ROOT / "runs-v33"
RUN_ROOT = RUNS / RUN_ID
PROGRESS = RUN_ROOT / "RETRY_PROGRESS.json"

sys.path.insert(0, str(ROOT / "package" / "civilian"))
import v3_runner as runner  # noqa: E402
import v3_shards as shards  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="ascii")
    tmp.replace(path)


def main() -> int:
    binding = shards.current_binding()
    registry = shards.validate_registry(shards.expected_registry(task_mode="search_rescue", targets_per_region=3))
    factory = runner._scenario_factory()
    expected = {}
    for ordinal, spec in enumerate(registry):
        scenario = factory(split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"],
                           instance_index=spec["instance_index"], task_mode=spec["task_mode"],
                           targets_per_region=spec["targets_per_region"])
        expected[scenario["parent_id"]] = (ordinal, spec)

    verified = shards.verify_run(RUNS, RUN_ID, expected_binding=binding)
    done = {row["scenario_id"] for row in verified["verified"]}
    retry = json.loads(PROGRESS.read_text(encoding="ascii")) if PROGRESS.exists() else {
        "schema": "gain-gate-v33-sequential-finish/1", "run_id": RUN_ID,
        "binding": binding, "started_unix": time.time(), "completed": [], "failures": []
    }
    done.update(retry.get("completed", []))
    pending = [(sid, *expected[sid]) for sid in sorted(expected) if sid not in done]
    retry["pending_at_start"] = len(pending)
    write_json(PROGRESS, retry)

    for sid, ordinal, spec in pending:
        stem = f"retry-{ordinal:03d}"
        job = RUN_ROOT / "jobs" / f"{stem}.json"
        stdout = RUN_ROOT / "jobs" / f"{stem}.stdout.log"
        stderr = RUN_ROOT / "jobs" / f"{stem}.stderr.log"
        write_json(job, {"spec": spec, "ordinal": ordinal, "run_id": RUN_ID, "binding": binding})
        with stdout.open("wb") as out, stderr.open("wb") as err:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "resume_v33_collect.py"),
                                     "--one-job", str(job)], cwd=ROOT, stdout=out, stderr=err)
        if result.returncode == 0:
            retry.setdefault("completed", []).append(sid)
            print(json.dumps({"event": "committed", "scenario_id": sid}, sort_keys=True), flush=True)
        else:
            retry.setdefault("failures", []).append({"scenario_id": sid, "ordinal": ordinal,
                                                       "returncode": result.returncode,
                                                       "stderr_log": str(stderr.relative_to(ROOT))})
            print(json.dumps({"event": "failed", "scenario_id": sid, "returncode": result.returncode}, sort_keys=True), flush=True)
        retry["updated_unix"] = time.time()
        write_json(PROGRESS, retry)

    final = shards.verify_run(RUNS, RUN_ID, expected_binding=binding)
    retry["verified_committed"] = final["committed_shards"]
    retry["remaining"] = len(registry) - final["committed_shards"]
    retry["finished_unix"] = time.time()
    write_json(PROGRESS, retry)
    print(json.dumps({"event": "finished", "verified_committed": final["committed_shards"],
                      "remaining": retry["remaining"], "failures": len(retry.get("failures", []))}, sort_keys=True), flush=True)
    return 0 if retry["remaining"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
