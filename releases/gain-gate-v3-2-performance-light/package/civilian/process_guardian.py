"""Linux parent-process guard for one complete driver stage invocation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from package_layout import require_package_root

from resource_supervisor import (
    DRIVER_STAGE_SCHEMA,
    EXPORT_DIR_NAME,
    EXPORT_STORAGE_LIMIT_BYTES,
    OUTER_GUARDIAN_SCHEMA,
    RESOURCE_LEDGER_SCHEMA,
    RSS_LIMIT_BYTES,
    STAGE_LIMITS,
    STAGE_ORDER,
    STAGE_PROTOCOL_ID,
    TOTAL_CPU_LIMIT,
    TOTAL_WALL_LIMIT,
    WORK_STORAGE_LIMIT_BYTES,
    _work_export_bytes,
    CARRY_FORWARD_RECEIPT_NAME,
    cumulative_budget_limits,
    _read_prior_resource_cost,
    _read_ledger,
    require_stage_budget,
)


POLL_SECONDS = 0.25
REAP_RACE_CONFIRM_ATTEMPTS = 10
REAP_RACE_CONFIRM_INTERVAL_SECONDS = 0.005


class _ProcRSSMissing(RuntimeError):
    def __init__(self, pid: int, state: str | None, cpu_seconds: float):
        super().__init__(f"RSS fields missing for pid {pid}")
        self.pid = pid
        self.state = state
        self.cpu_seconds = cpu_seconds


class _ProcZombie(_ProcRSSMissing):
    def __init__(self, pid: int, state: str | None, cpu_seconds: float):
        super().__init__(pid, state, cpu_seconds)


def _confirm_child_reaped(child_pid: int) -> tuple[int, int, Any]:
    for attempt in range(REAP_RACE_CONFIRM_ATTEMPTS + 1):
        result = os.wait4(child_pid, os.WNOHANG)
        if result[0] == child_pid:
            return result
        if attempt < REAP_RACE_CONFIRM_ATTEMPTS:
            time.sleep(REAP_RACE_CONFIRM_INTERVAL_SECONDS)
    return 0, 0, None


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _proc_metrics(pid: int) -> dict[str, int]:
    status_path = Path(f"/proc/{pid}/status")
    stat_path = Path(f"/proc/{pid}/stat")
    values: dict[str, int] = {}
    state = None
    for line in status_path.read_text(encoding="ascii").splitlines():
        if line.startswith("State:"):
            state = line.split()[1]
        elif line.startswith("VmRSS:"):
            values["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            values["rss_hwm_bytes"] = int(line.split()[1]) * 1024
    stat = stat_path.read_text(encoding="ascii")
    fields = stat[stat.rfind(")") + 2 :].split()
    stat_state = fields[0] if fields else None
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    cpu_seconds = (int(fields[11]) + int(fields[12])) / ticks_per_second
    values["cpu_seconds"] = cpu_seconds
    if "rss_bytes" not in values or "rss_hwm_bytes" not in values:
        sampled_state = state or stat_state
        if state in {"Z", "X"} or stat_state in {"Z", "X"}:
            raise _ProcZombie(pid, sampled_state, cpu_seconds)
        raise _ProcRSSMissing(pid, sampled_state, cpu_seconds)
    return values


def _process_started_at() -> tuple[float, float]:
    """Use proc start ticks so Python import and CLI setup are inside the wall bound."""
    uptime = float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])
    stat = Path("/proc/self/stat").read_text(encoding="ascii")
    fields = stat[stat.rfind(")") + 2 :].split()
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    elapsed_wall = uptime - int(fields[19]) / ticks_per_second
    started_wall = time.monotonic() - elapsed_wall
    started_cpu = max(0.0, time.process_time() - (int(fields[11]) + int(fields[12])) / ticks_per_second)
    return started_wall, started_cpu


def _rewrite_last_row(ledger_path: Path, row: dict[str, Any], *, replace_last: bool) -> None:
    rows = _read_ledger_tolerating_partial_tail(ledger_path)
    if replace_last and rows:
        rows[-1] = row
    else:
        rows.append(row)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, ledger_path)


def _read_ledger_tolerating_partial_tail(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.is_file():
        return []
    rows = []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if (
            row.get("schema") != RESOURCE_LEDGER_SCHEMA
            or row.get("stage_protocol") != STAGE_PROTOCOL_ID
            or row.get("stage") not in STAGE_ORDER
            or not isinstance(row.get("metadata"), dict)
            or row.get("runtime_identity_sha256") != row.get("metadata", {}).get("runtime_identity_sha256")
            or row.get("release_digest") != row.get("runtime_identity_sha256")
            or row.get("metadata", {}).get("release_digest") != row.get("release_digest")
        ):
            raise ValueError(f"unsupported resource ledger row {index + 1}")
        rows.append(row)
    return rows


def _checkpoint_inventory(work_root: Path) -> list[dict[str, Any]]:
    import hashlib

    checkpoints = []
    for path in sorted(work_root.rglob("*.pt")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoints.append({
            "path": path.relative_to(work_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return checkpoints


def _stage_receipt(work_root: Path, stage: str, ledger_rows: list[dict]) -> Path:
    count = len(ledger_rows) + 1
    return work_root / "receipts" / f"{count:02d}-{stage}.json"


def _proc_tree(root_pid: int) -> set[int]:
    """Return a Linux process tree, failing closed when a live tree is unreadable."""
    tree = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        task_root = Path(f"/proc/{pid}/task")
        try:
            tasks = list(task_root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect process tree for pid {pid}") from exc
        for task in tasks:
            try:
                children = (task / "children").read_text(encoding="ascii").split()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(f"cannot inspect process children for pid {pid}") from exc
            for value in children:
                child = int(value)
                if child not in tree:
                    tree.add(child)
                    pending.append(child)
    return tree


def _proc_group_pids(process_group: int) -> set[int]:
    pids = set()
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 2 and int(fields[2]) == process_group:
            pids.add(int(stat_path.parent.name))
    return pids


def _sample_group_rss(pids: set[int]) -> tuple[int, float]:
    total_rss = 0
    total_cpu = 0.0
    for pid in sorted(pids):
        try:
            metrics = _proc_metrics(pid)
        except FileNotFoundError:
            continue
        except _ProcRSSMissing as exc:
            if exc.state in {"Z", "X"}:
                continue
            raise RuntimeError(f"process {pid} has no measurable RSS while still live") from exc
        total_rss += metrics["rss_bytes"]
        total_cpu += metrics["cpu_seconds"]
    return total_rss, total_cpu


def _phase_protocol_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PermissionError("worker phase-accounting journal is missing")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PermissionError(f"invalid worker phase-accounting row {line_number}") from exc
    return rows


def _expected_boundaries(stage_order: tuple[str, ...], *, preflight_only: bool) -> list[tuple[str, str, str]]:
    if preflight_only:
        return [("stage_enter", stage_order[0], stage_order[0])]
    return [("stage_enter", stage_order[0], stage_order[0])] + [
        ("transition", stage, stage_order[index + 1])
        for index, stage in enumerate(stage_order[:-1])
    ]


def _validate_pipeline_handshakes(
    server: Any,
    worker_pid: int,
    work_root: Path,
    stage_order: tuple[str, ...],
    *,
    preflight_only: bool,
) -> list[dict[str, Any]]:
    expected = _expected_boundaries(stage_order, preflight_only=preflight_only)
    if (
        server.pending_request is not None
        or server.last_rejection is not None
        or server.last_sequence != len(expected)
        or set(server.committed_requests) != set(range(1, len(expected) + 1))
    ):
        raise PermissionError("StageServer did not commit the complete contiguous boundary sequence")
    for sequence, (event, stage, next_stage) in enumerate(expected, 1):
        committed = server.committed_requests[sequence]
        payload = committed.get("payload", {})
        if (
            payload.get("pid") != worker_pid
            or payload.get("protocol_sequence") != sequence
            or payload.get("event") != event
            or payload.get("stage") != stage
            or payload.get("next_stage") != next_stage
            or committed.get("row", {}).get("event") != event
        ):
            raise PermissionError(f"StageServer committed an invalid boundary at sequence {sequence}")
    phase_rows = _phase_protocol_rows(work_root / "worker-phase-accounting.jsonl")
    boundaries = [row for row in phase_rows if row.get("event") in {"stage_enter", "transition"}]
    if len(boundaries) != len(expected):
        raise PermissionError("worker accounting journal does not contain exactly the accepted boundary chain")
    for sequence, (row, (event, stage, next_stage)) in enumerate(zip(boundaries, expected, strict=True), 1):
        handshake = row.get("supervisor_handshake", {})
        if (
            row.get("pid") != worker_pid
            or row.get("protocol_sequence") != sequence
            or row.get("event") != event
            or row.get("stage") != stage
            or row.get("next_stage") != next_stage
            or handshake.get("status") != "acknowledged"
            or handshake.get("sequence") != sequence
            or handshake.get("server_accepted") is not True
            or handshake.get("worker_confirmed") is not True
            or handshake.get("protocol_committed") is not True
            or handshake.get("transition_committed") is not (event == "transition")
        ):
            raise PermissionError(f"worker accounting receipt chain is invalid at sequence {sequence}")
    return boundaries


def _stage_measurements(
    server: Any,
    final_worker_cpu_seconds: float,
    final_worker_stage_cpu_seconds: float,
    finished_wall: float,
    *,
    started_cpu: float,
    started_wall: float,
    preflight_only: bool,
) -> dict[str, dict[str, float]]:
    stage_order = tuple(server.stage_order)
    expected_count = 1 if preflight_only else len(stage_order)
    measurements: dict[str, dict[str, float]] = {}
    for sequence in range(1, expected_count + 1):
        stage = stage_order[sequence - 1]
        boundary_sequence = sequence + 1
        if sequence < expected_count and boundary_sequence in server.committed_requests:
            row = server.committed_requests[boundary_sequence]["row"]
            wall_seconds = float(row["stage_elapsed_wall_seconds"])
            cpu_seconds = float(row["stage_elapsed_cpu_seconds"])
            worker_cpu_seconds = float(row["worker_reported_cpu_seconds"])
            stage_finished_wall = float(row["sampled_monotonic"])
        elif server.stage == stage and server.last_sequence >= sequence:
            stage_wall_start = started_wall if sequence == 1 else float(server.wall_start)
            stage_cpu_start = started_cpu if sequence == 1 else float(server.cpu_start)
            wall_seconds = max(0.0, finished_wall - stage_wall_start)
            cpu_at_exit = time.process_time() + final_worker_cpu_seconds
            cpu_seconds = max(0.0, cpu_at_exit - stage_cpu_start)
            worker_cpu_seconds = final_worker_stage_cpu_seconds
            stage_finished_wall = finished_wall
        else:
            break
        if sequence == 1:
            stage_started_wall = started_wall
        else:
            stage_started_wall = float(server.committed_requests[sequence]["row"]["sampled_monotonic"])
        measurements[stage] = {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "worker_cpu_seconds": worker_cpu_seconds,
            "started_wall_monotonic": stage_started_wall,
            "finished_wall_monotonic": stage_finished_wall,
        }
    return measurements


def _write_pipeline_receipts(
    *,
    work_root: Path,
    package_root: Path,
    identity: str,
    stage_order: tuple[str, ...],
    measurements: dict[str, dict[str, float]],
    child_returncode: int,
    peak_group_rss: int,
    peak_work_disk: int,
    peak_export_disk: int,
    prior_cost: dict[str, Any],
    guardian_error: str | None,
    kill_reason: str | None,
    child_pid: int | None,
    invocation_measurement: dict[str, Any],
    preflight_only: bool,
    failed_stage: str | None,
) -> str | None:
    ledger_path = work_root / "resource-ledger.jsonl"
    rows = _read_ledger_tolerating_partial_tail(ledger_path)
    expected_stages = stage_order[:1] if preflight_only else stage_order
    if len(rows) > len(expected_stages) or [row.get("stage") for row in rows] != list(expected_stages[: len(rows)]):
        guardian_error = guardian_error or "worker resource ledger is not the ordered pipeline stage chain"
    first_error = guardian_error
    if child_returncode != 0 and first_error is None:
        first_error = f"guarded pipeline child exited with status {child_returncode}"
    if len(rows) != len(expected_stages) and first_error is None:
        first_error = "guarded pipeline did not emit one resource row for every required stage"
    receipt_stages = list(expected_stages[: len(rows)])
    if first_error and failed_stage in expected_stages and failed_stage not in receipt_stages:
        receipt_stages.append(failed_stage)
    if not receipt_stages and first_error:
        receipt_stages.append(expected_stages[0])
    for stage in receipt_stages:
        index = expected_stages.index(stage)
        receipt_path = work_root / "receipts" / f"{index + 1:02d}-{stage}.json"
        row = rows[index] if index < len(rows) and rows[index].get("stage") == stage else None
        stage_metrics = measurements.get(stage)
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
        else:
            receipt = {}
        stage_error = row.get("first_error") if row is not None else receipt.get("first_error")
        if stage == failed_stage and first_error:
            stage_error = stage_error or first_error
        status = "failed" if stage_error else (row.get("status", "failed") if row is not None else "failed")
        guardian_path = work_root / "guardians" / f"{index + 1:02d}-{stage}.json"
        expected_receipt_status = "accepted" if stage == stage_order[0] else "complete"
        stage_state = (
            "stage_entered"
            if row is not None or stage_metrics is not None
            else "stage_not_entered"
        )
        stage_measurement_status = "measured" if stage_metrics is not None else "unknown"
        stage_wall_seconds = stage_metrics.get("wall_seconds") if stage_metrics is not None else None
        stage_cpu_seconds = stage_metrics.get("cpu_seconds") if stage_metrics is not None else None
        stage_worker_cpu_seconds = stage_metrics.get("worker_cpu_seconds") if stage_metrics is not None else None
        guardian_receipt = {
            "schema": OUTER_GUARDIAN_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "runtime_identity_sha256": identity,
            "release_digest": identity,
            "stage": stage,
            "status": "failed" if status == "failed" else "complete",
            "first_error": stage_error,
            "stage_state": stage_state,
            "stage_measurement_status": stage_measurement_status,
            "outer_guardian_error": guardian_error,
            "outer_guardian_kill_reason": kill_reason,
            "child_returncode": child_returncode,
            "child_pid": child_pid,
            "prior_resource_cost_sha256": prior_cost["receipt_sha256"],
            "scope": "one guarded worker process across the complete pipeline; wait4 child CPU/RSS plus guardian CPU; phase transitions assign cumulative process use to each stage",
            "wall_seconds": stage_wall_seconds,
            "cpu_seconds": stage_cpu_seconds,
            "child_cpu_seconds": stage_worker_cpu_seconds,
            "guardian_cpu_seconds": (
                max(0.0, stage_cpu_seconds - stage_worker_cpu_seconds)
                if stage_cpu_seconds is not None and stage_worker_cpu_seconds is not None
                else None
            ),
            "guardian_invocation_measurement": invocation_measurement,
            "peak_process_group_rss_bytes": peak_group_rss,
            "peak_work_bytes": peak_work_disk,
            "peak_export_bytes": peak_export_disk,
            "work_limit_bytes": WORK_STORAGE_LIMIT_BYTES,
            "export_limit_bytes": EXPORT_STORAGE_LIMIT_BYTES,
            "resource_entry": row,
            "stage_observation": receipt.get("resource_entry") if row is None else None,
        }
        if row is not None:
            row.update({
                "schema": RESOURCE_LEDGER_SCHEMA,
                "stage_protocol": STAGE_PROTOCOL_ID,
                "runtime_identity_sha256": identity,
                "release_digest": identity,
                "status": status,
                "peak_rss_bytes": peak_group_rss,
                "peak_work_bytes": peak_work_disk,
                "peak_export_bytes": peak_export_disk,
                "outer_guardian_error": guardian_error,
                "outer_guardian_kill_reason": kill_reason,
                "first_error": stage_error,
            })
            metadata = row.setdefault("metadata", {})
            metadata.update({
                "runtime_identity_sha256": identity,
                "release_digest": identity,
                "receipt": receipt_path.as_posix(),
                "outer_guardian_receipt": guardian_path.as_posix(),
                "outer_guardian_sampling_interval_seconds": POLL_SECONDS,
                "prior_resource_cost_sha256": prior_cost["receipt_sha256"],
            })
            if stage_metrics is not None:
                row.update({
                    "started_wall_monotonic": stage_metrics.get("started_wall_monotonic", row.get("started_wall_monotonic")),
                    "finished_wall_monotonic": stage_metrics.get("finished_wall_monotonic", row.get("finished_wall_monotonic")),
                    "wall_seconds": stage_wall_seconds,
                    "cpu_seconds": stage_cpu_seconds,
                })
        guardian_receipt["resource_entry"] = row
        _json_write(guardian_path, guardian_receipt)
        receipt.update({
            "schema": DRIVER_STAGE_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "stage": stage,
            "status": "failed" if stage_error else receipt.get("status", expected_receipt_status),
            "runtime_identity_sha256": identity,
            "release_digest": identity,
            "first_error": stage_error,
            "resource_entry": row if row is not None else receipt.get("resource_entry"),
            "outer_guardian_receipt": guardian_path.as_posix(),
            "stage_state": stage_state,
            "stage_measurement_status": stage_measurement_status,
            "guardian_invocation_measurement": invocation_measurement,
        })
        if row is None:
            observation = receipt.get("resource_entry")
            if not isinstance(observation, dict) or observation.get("schema") != "pre-stage-failure-observation/1":
                observation = {
                    "schema": "pre-stage-failure-observation/1",
                    "stage_protocol": STAGE_PROTOCOL_ID,
                    "stage": stage,
                    "status": "failed",
                    "stage_selected": False,
                    "stage_state": stage_state,
                    "first_error": stage_error,
                    "resource_measurement": {
                        "status": stage_measurement_status,
                        "wall_seconds": stage_wall_seconds,
                        "cpu_seconds": stage_cpu_seconds,
                        "scope": "stage",
                        "reason": "no settled resource-ledger row is available for this stage",
                    },
                    "guardian_invocation_measurement": invocation_measurement,
                    "runtime_identity_sha256": identity,
                    "release_digest": identity,
                }
            else:
                observation = dict(observation)
                observation["guardian_invocation_measurement"] = invocation_measurement
            receipt["stage_observation"] = observation
        receipt.setdefault("partial_checkpoints", [])
        _json_write(receipt_path, receipt)
        prior_stage = prior_cost["stage_costs"].get(stage, {})
        rows_before = rows[:index]
        prior_stage_wall = float(prior_stage.get("wall_seconds", 0.0)) + sum(
            float(item.get("wall_seconds", 0.0)) for item in rows_before if item.get("stage") == stage
        )
        prior_stage_cpu = float(prior_stage.get("cpu_seconds", 0.0)) + sum(
            float(item.get("cpu_seconds", 0.0)) for item in rows_before if item.get("stage") == stage
        )
        prior_total_wall = prior_cost["total_wall_seconds"] + sum(
            float(item.get("wall_seconds", 0.0)) for item in rows_before
        )
        prior_total_cpu = prior_cost["total_cpu_seconds"] + sum(
            float(item.get("cpu_seconds", 0.0)) for item in rows_before
        )
        if row is not None and stage_metrics is not None:
            row.update({
                "stage_wall_used_including_this": prior_stage_wall + stage_wall_seconds,
                "stage_cpu_used_including_this": prior_stage_cpu + stage_cpu_seconds,
                "pilot_wall_used_including_this": prior_total_wall + stage_wall_seconds,
                "pilot_cpu_used_including_this": prior_total_cpu + stage_cpu_seconds,
            })
            guardian_receipt["resource_entry"] = row
            _json_write(guardian_path, guardian_receipt)
            receipt["resource_entry"] = row
            _json_write(receipt_path, receipt)
    if rows:
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ledger_path)
    return first_error


def _update_stage_receipt(
    receipt_path: Path,
    row: dict[str, Any],
    *,
    error: str | None,
    checkpoints: list[dict[str, Any]],
    guardian_path: Path,
) -> None:
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    else:
        receipt = {
            "schema": DRIVER_STAGE_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "stage": row["stage"],
            "result": None,
            "partial_checkpoints": checkpoints,
        }
    receipt["resource_entry"] = row
    receipt["runtime_identity_sha256"] = row.get("metadata", {}).get("runtime_identity_sha256")
    receipt["release_digest"] = row.get("metadata", {}).get("release_digest")
    receipt["outer_guardian_receipt"] = guardian_path.as_posix()
    if error is not None:
        receipt["status"] = "failed"
        receipt["first_error"] = row.get("first_error") or error
        receipt["partial_checkpoints"] = checkpoints
    _json_write(receipt_path, receipt)


def run_guarded_stage(
    *,
    argv: list[str],
    driver_path: Path,
    package_root: Path,
    work_root: Path,
    stage: str,
    runtime_identity_sha256: str,
    runtime_environment: dict[str, str],
) -> int:
    raise RuntimeError("single-stage process guarding is disabled; use the complete guarded pipeline")

    if sys.platform != "linux":
        raise RuntimeError("full-process resource guarding requires the accepted Linux target runtime")
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown guarded stage: {stage}")
    if (
        len(runtime_identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in runtime_identity_sha256)
    ):
        raise ValueError("runtime identity must be a lowercase SHA-256 digest")

    ledger_path = work_root / "resource-ledger.jsonl"
    work_root.mkdir(parents=True, exist_ok=True)
    previous_rows = _read_ledger(ledger_path)
    if any(
        row.get("runtime_identity_sha256") != runtime_identity_sha256
        or row.get("release_digest") != runtime_identity_sha256
        or row.get("metadata", {}).get("runtime_identity_sha256") != runtime_identity_sha256
        or row.get("metadata", {}).get("release_digest") != runtime_identity_sha256
        for row in previous_rows
    ):
        raise PermissionError("resource ledger belongs to a different package, data, runtime, or work-root identity")
    prior_cost = _read_prior_resource_cost(package_root / CARRY_FORWARD_RECEIPT_NAME, required=True)
    carried_stage = prior_cost["stage_costs"].get(stage, {})
    previous_stage_wall = float(carried_stage.get("wall_seconds", 0.0)) + sum(row["wall_seconds"] for row in previous_rows if row["stage"] == stage)
    previous_stage_cpu = float(carried_stage.get("cpu_seconds", 0.0)) + sum(row["cpu_seconds"] for row in previous_rows if row["stage"] == stage)
    previous_total_wall = prior_cost["total_wall_seconds"] + sum(row["wall_seconds"] for row in previous_rows)
    previous_total_cpu = prior_cost["total_cpu_seconds"] + sum(row["cpu_seconds"] for row in previous_rows)
    require_stage_budget(stage, previous_total_wall, previous_total_cpu)
    stage_wall_limit, stage_cpu_limit = STAGE_LIMITS[stage]
    total_wall_limit, total_cpu_limit = cumulative_budget_limits(stage)
    started_wall, started_cpu = _process_started_at()
    guardian_cpu_start = started_cpu
    export_root = work_root / EXPORT_DIR_NAME
    current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
    peak_work_disk = current_work_disk
    peak_export_disk = current_export_disk
    peak_group_rss = _proc_metrics(os.getpid())["rss_hwm_bytes"]
    first_error: str | None = None
    outer_error: str | None = None
    kill_reason: str | None = None
    outer_error_at: float | None = None
    status_code = 1
    child_rusage = None
    child_status = 0
    child_metrics: dict[str, int] = {"cpu_seconds": 0, "rss_bytes": 0, "rss_hwm_bytes": 0}
    last_disk_sample = time.monotonic()
    guardian_path = work_root / "guardians" / f"{len(previous_rows) + 1:02d}-{stage}.json"
    token = uuid.uuid4().hex
    env = os.environ.copy()
    env.update(runtime_environment)
    env["RELIEF_GRAPH_GUARDIAN_TOKEN"] = token
    child_argv = [*argv, "--_guardian-token", token]
    command = [sys.executable, "-B", str(driver_path), *child_argv]

    try:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {0})
            except OSError:
                pass
        process = subprocess.Popen(command, cwd=str(package_root), env=env, close_fds=True)
        child_pid = process.pid
        while True:
            waited_pid, child_status, child_rusage = os.wait4(child_pid, os.WNOHANG)
            if waited_pid == child_pid:
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
                break
            now = time.monotonic()
            elapsed_wall = now - started_wall
            try:
                child_metrics = _proc_metrics(child_pid)
            except _ProcRSSMissing as exc:
                # A missing RSS sample is accepted only after wait4 confirms and accounts for exit.
                waited_pid, child_status, child_rusage = _confirm_child_reaped(child_pid)
                if waited_pid == child_pid:
                    process.returncode = os.waitstatus_to_exitcode(child_status)
                    status_code = int(process.returncode)
                    break
                raise RuntimeError(f"{exc}; wait4 did not confirm child exit (state={exc.state!r})") from exc
            except FileNotFoundError as exc:
                waited_pid, child_status, child_rusage = _confirm_child_reaped(child_pid)
                if waited_pid == child_pid:
                    process.returncode = os.waitstatus_to_exitcode(child_status)
                    status_code = int(process.returncode)
                    break
                raise RuntimeError(f"/proc metrics disappeared while child {child_pid} remained live") from exc
            guardian_metrics = _proc_metrics(os.getpid())
            group_rss = child_metrics["rss_bytes"] + guardian_metrics["rss_bytes"]
            peak_group_rss = max(peak_group_rss, group_rss)
            child_cpu = child_metrics["cpu_seconds"]
            elapsed_cpu = child_cpu + max(0.0, time.process_time() - guardian_cpu_start)
            if now - last_disk_sample >= POLL_SECONDS:
                current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
                peak_work_disk = max(peak_work_disk, current_work_disk)
                peak_export_disk = max(peak_export_disk, current_export_disk)
                last_disk_sample = now
            violations = []
            if previous_stage_wall + elapsed_wall > stage_wall_limit:
                violations.append(f"{stage} wall {previous_stage_wall + elapsed_wall:.3f}>{stage_wall_limit:.0f}s")
            if previous_stage_cpu + elapsed_cpu > stage_cpu_limit:
                violations.append(f"{stage} CPU {previous_stage_cpu + elapsed_cpu:.3f}>{stage_cpu_limit:.0f}s")
            if previous_total_wall + elapsed_wall > total_wall_limit:
                violations.append(f"whole-pilot wall {previous_total_wall + elapsed_wall:.3f}>{total_wall_limit:.0f}s")
            if previous_total_cpu + elapsed_cpu > total_cpu_limit:
                violations.append(f"whole-pilot CPU {previous_total_cpu + elapsed_cpu:.3f}>{total_cpu_limit:.0f}s")
            if peak_group_rss > RSS_LIMIT_BYTES:
                violations.append(f"process-group RSS {peak_group_rss}>{RSS_LIMIT_BYTES} bytes")
            if current_work_disk > WORK_STORAGE_LIMIT_BYTES:
                violations.append(f"work storage {current_work_disk}>{WORK_STORAGE_LIMIT_BYTES} bytes")
            if current_export_disk > EXPORT_STORAGE_LIMIT_BYTES:
                violations.append(f"export storage {current_export_disk}>{EXPORT_STORAGE_LIMIT_BYTES} bytes")
            if violations:
                kill_reason = "; ".join(violations)
                outer_error = outer_error or f"outer guardian: {kill_reason}"
                outer_error_at = outer_error_at or now
                os.kill(child_pid, signal.SIGKILL)
                _waited_pid, child_status, child_rusage = os.wait4(child_pid, 0)
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
                break
            time.sleep(POLL_SECONDS)
    except BaseException as exc:
        outer_error = outer_error or f"{type(exc).__name__}: {exc}"
        outer_error_at = outer_error_at or time.monotonic()
        if "process" in locals():
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _waited_pid, child_status, child_rusage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
            except ChildProcessError:
                pass

    child_cpu_seconds = 0.0
    child_peak_rss = 0
    if child_rusage is not None:
        child_cpu_seconds = float(child_rusage.ru_utime + child_rusage.ru_stime)
        child_peak_rss = int(child_rusage.ru_maxrss) * 1024
    current_rows = _read_ledger_tolerating_partial_tail(ledger_path)
    has_child_entry = len(current_rows) > len(previous_rows)
    row = current_rows[-1] if has_child_entry else None
    if row is None or row.get("stage") != stage:
        outer_error = outer_error or "outer guardian found no settled child stage entry"
        outer_error_at = outer_error_at or time.monotonic()
        row = {
            "schema": RESOURCE_LEDGER_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "stage": stage,
            "runtime_identity_sha256": runtime_identity_sha256,
            "release_digest": runtime_identity_sha256,
            "status": "failed",
            "started_wall_monotonic": started_wall,
            "metadata": {
                "runtime_identity_sha256": runtime_identity_sha256,
                "release_digest": runtime_identity_sha256,
            },
        }
    if row.get("metadata", {}).get("runtime_identity_sha256") != runtime_identity_sha256:
        outer_error = outer_error or "child resource ledger identity does not match the guarded release"
        outer_error_at = outer_error_at or time.monotonic()
    if row.get("runtime_identity_sha256") != runtime_identity_sha256:
        outer_error = outer_error or "child resource ledger row identity does not match the guarded release"
        outer_error_at = outer_error_at or time.monotonic()
    if row.get("release_digest") != runtime_identity_sha256:
        outer_error = outer_error or "child resource ledger release digest does not match the guarded release"
        outer_error_at = outer_error_at or time.monotonic()
    child_started_wall = row.get("started_wall_monotonic")

    receipt_path = Path(row.get("metadata", {}).get("receipt", ""))
    if not receipt_path.is_absolute():
        receipt_path = work_root / receipt_path
    if not receipt_path.is_file():
        receipt_path = _stage_receipt(work_root, stage, previous_rows)
    child_receipt_error = None
    child_receipt_identity_valid = False
    if receipt_path.is_file():
        try:
            child_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            child_receipt_error = child_receipt.get("first_error")
            child_receipt_identity_valid = (
                child_receipt.get("schema") == DRIVER_STAGE_SCHEMA
                and child_receipt.get("stage_protocol") == STAGE_PROTOCOL_ID
                and child_receipt.get("stage") == stage
                and child_receipt.get("runtime_identity_sha256") == runtime_identity_sha256
                and child_receipt.get("release_digest") == runtime_identity_sha256
            )
            if not child_receipt_identity_valid and status_code == 0:
                outer_error = outer_error or "child stage receipt does not bind the guarded release identity"
                outer_error_at = outer_error_at or time.monotonic()
        except (json.JSONDecodeError, OSError):
            if status_code == 0:
                outer_error = outer_error or "successful child stage receipt is unreadable"
                outer_error_at = outer_error_at or time.monotonic()
    elif status_code == 0:
        outer_error = outer_error or "successful child stage receipt is missing"
        outer_error_at = outer_error_at or time.monotonic()
    child_error = row.get("first_error") or child_receipt_error
    child_error_preceded_outer = bool(
        child_error
        and (outer_error_at is None or row.get("finished_wall_monotonic", float("inf")) <= outer_error_at)
    )
    first_error = child_error if child_error_preceded_outer else outer_error
    if status_code != 0 and first_error is None:
        first_error = child_error or f"guarded child exited with status {status_code}"
    first_error = first_error or child_error

    # Include failed-checkpoint inventory hashing in the outer stage totals.
    checkpoints = _checkpoint_inventory(work_root) if first_error is not None else []
    finished_wall = time.monotonic()
    finished_cpu = time.process_time()
    wall_seconds = max(0.0, finished_wall - started_wall)
    guardian_cpu_seconds = max(0.0, finished_cpu - guardian_cpu_start)
    cpu_seconds = child_cpu_seconds + guardian_cpu_seconds
    guardian_final = _proc_metrics(os.getpid())
    peak_group_rss = max(peak_group_rss, child_peak_rss + guardian_final["rss_bytes"])
    current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
    peak_work_disk = max(peak_work_disk, current_work_disk)
    peak_export_disk = max(peak_export_disk, current_export_disk)
    final_violations = []
    if previous_stage_wall + wall_seconds > stage_wall_limit:
        final_violations.append(f"{stage} wall {previous_stage_wall + wall_seconds:.3f}>{stage_wall_limit:.0f}s")
    if previous_stage_cpu + cpu_seconds > stage_cpu_limit:
        final_violations.append(f"{stage} CPU {previous_stage_cpu + cpu_seconds:.3f}>{stage_cpu_limit:.0f}s")
    if previous_total_wall + wall_seconds > total_wall_limit:
        final_violations.append(f"whole-pilot wall {previous_total_wall + wall_seconds:.3f}>{total_wall_limit:.0f}s")
    if previous_total_cpu + cpu_seconds > total_cpu_limit:
        final_violations.append(f"whole-pilot CPU {previous_total_cpu + cpu_seconds:.3f}>{total_cpu_limit:.0f}s")
    if peak_group_rss > RSS_LIMIT_BYTES:
        final_violations.append(f"process-group RSS {peak_group_rss}>{RSS_LIMIT_BYTES} bytes")
    if peak_work_disk > WORK_STORAGE_LIMIT_BYTES:
        final_violations.append(f"work storage {peak_work_disk}>{WORK_STORAGE_LIMIT_BYTES} bytes")
    if peak_export_disk > EXPORT_STORAGE_LIMIT_BYTES:
        final_violations.append(f"export storage {peak_export_disk}>{EXPORT_STORAGE_LIMIT_BYTES} bytes")
    if final_violations:
        outer_error = outer_error or f"outer guardian: {'; '.join(final_violations)}"
        outer_error_at = outer_error_at or finished_wall
        first_error = first_error or outer_error
        if first_error is not None:
            row["status"] = "failed"
            row["first_error"] = first_error
        if not checkpoints:
            checkpoints = _checkpoint_inventory(work_root)
            finished_wall = time.monotonic()
            finished_cpu = time.process_time()
            wall_seconds = max(0.0, finished_wall - started_wall)
            guardian_cpu_seconds = max(0.0, finished_cpu - guardian_cpu_start)
            cpu_seconds = child_cpu_seconds + guardian_cpu_seconds
            current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
            peak_work_disk = max(peak_work_disk, current_work_disk)
            peak_export_disk = max(peak_export_disk, current_export_disk)

    if first_error is not None:
        row["status"] = "failed"
        row["first_error"] = first_error
    row.update({
        "schema": RESOURCE_LEDGER_SCHEMA,
        "stage_protocol": STAGE_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_identity_sha256,
        "release_digest": runtime_identity_sha256,
        "started_wall_monotonic": started_wall,
        "finished_wall_monotonic": finished_wall,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": peak_group_rss,
        "rss_high_water_source": "wait4.ru_maxrss plus /proc sampled process-group VmRSS",
        "outer_guardian_error": outer_error,
        "outer_guardian_kill_reason": kill_reason,
        "outer_guardian_error_at_wall_monotonic": outer_error_at,
        "child_error_preceded_outer_guardian": child_error_preceded_outer,
        "peak_work_bytes": peak_work_disk,
        "peak_export_bytes": peak_export_disk,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "work_limit_bytes": WORK_STORAGE_LIMIT_BYTES,
        "export_limit_bytes": EXPORT_STORAGE_LIMIT_BYTES,
        "stage_wall_used_including_this": previous_stage_wall + wall_seconds,
        "stage_cpu_used_including_this": previous_stage_cpu + cpu_seconds,
        "pilot_wall_used_including_this": previous_total_wall + wall_seconds,
        "pilot_cpu_used_including_this": previous_total_cpu + cpu_seconds,
    })
    row.setdefault("metadata", {})["outer_guardian_receipt"] = guardian_path.as_posix()
    row["metadata"]["outer_guardian_sampling_interval_seconds"] = POLL_SECONDS
    row["metadata"]["child_exit_race_confirmation"] = {
        "maximum_window_seconds": REAP_RACE_CONFIRM_ATTEMPTS * REAP_RACE_CONFIRM_INTERVAL_SECONDS,
        "interval_seconds": REAP_RACE_CONFIRM_INTERVAL_SECONDS,
        "accepted_only_after_wait4_reap": True,
    }
    row["metadata"]["disk_peak_sampling_method"] = "directory size scan every 0.25 seconds; short-lived growth between scans may be missed"
    if child_started_wall is not None:
        row["metadata"]["child_started_wall_monotonic"] = child_started_wall
    if prior_cost["receipt_sha256"] is not None:
        row["metadata"]["prior_resource_cost_sha256"] = prior_cost["receipt_sha256"]
    _rewrite_last_row(ledger_path, row, replace_last=has_child_entry)

    if first_error is not None and child_receipt_identity_valid:
        _update_stage_receipt(
            receipt_path,
            row,
            error=first_error,
            checkpoints=checkpoints,
            guardian_path=guardian_path,
        )
    elif first_error is None:
        receipt_path = Path(row.get("metadata", {}).get("receipt", ""))
        if receipt_path.is_file():
            _update_stage_receipt(receipt_path, row, error=None, checkpoints=[], guardian_path=guardian_path)

    guardian_receipt = {
        "schema": OUTER_GUARDIAN_SCHEMA,
        "stage_protocol": STAGE_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_identity_sha256,
        "release_digest": runtime_identity_sha256,
        "stage": stage,
        "status": "failed" if first_error is not None else "complete",
        "first_error": first_error,
        "outer_guardian_error": outer_error,
        "outer_guardian_kill_reason": kill_reason,
        "outer_guardian_error_at_wall_monotonic": outer_error_at,
        "child_returncode": status_code,
        "child_pid": child_pid if "child_pid" in locals() else None,
        "prior_resource_cost_sha256": prior_cost["receipt_sha256"],
        "scope": "driver process start through child exit and failed-checkpoint inventory hashing; wait4 child CPU/RSS plus guardian CPU; final ledger and receipt writes are excluded",
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "child_cpu_seconds": child_cpu_seconds,
        "guardian_cpu_seconds": guardian_cpu_seconds,
        "peak_process_group_rss_bytes": peak_group_rss,
        "rss_measurement": "Linux wait4.ru_maxrss plus 0.25-second /proc process-group samples",
        "peak_work_bytes": peak_work_disk,
        "peak_export_bytes": peak_export_disk,
        "work_limit_bytes": WORK_STORAGE_LIMIT_BYTES,
        "export_limit_bytes": EXPORT_STORAGE_LIMIT_BYTES,
        "disk_sampling_interval_seconds": POLL_SECONDS,
        "disk_sampling_limit": "short-lived growth between directory scans may be missed",
        "stage_wall_used_including_this": previous_stage_wall + wall_seconds,
        "stage_cpu_used_including_this": previous_stage_cpu + cpu_seconds,
        "pilot_wall_used_including_this": previous_total_wall + wall_seconds,
        "pilot_cpu_used_including_this": previous_total_cpu + cpu_seconds,
        "resource_entry": row,
    }
    _json_write(guardian_path, guardian_receipt)
    return status_code if first_error is None else (status_code if status_code != 0 else 1)


def _validate_work_call_database(path: Path) -> None:
    import sqlite3

    if not path.is_file():
        raise PermissionError("SQLite work-call ledger is missing")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise PermissionError("SQLite work-call ledger integrity check failed")
        pending = connection.execute("SELECT COUNT(*) FROM work_calls WHERE status='pending'").fetchone()[0]
        failed = connection.execute("SELECT COUNT(*) FROM work_calls WHERE status='failed'").fetchone()[0]
        if pending or failed:
            raise PermissionError("SQLite work-call ledger contains pending or failed calls")
        invalid = connection.execute(
            "SELECT COUNT(*) FROM work_calls WHERE actual_count IS NULL OR actual_count > reserved_count OR status!='complete'"
        ).fetchone()[0]
        if invalid:
            raise PermissionError("SQLite work-call ledger contains an unsettled or over-reserved call")
        pending_legacy = connection.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0]
        failed_legacy = connection.execute("SELECT COUNT(*) FROM calls WHERE status='failed'").fetchone()[0]
        if pending_legacy or failed_legacy:
            raise PermissionError("SQLite budget ledger contains pending or failed calls")
    finally:
        connection.close()


def run_guarded_pipeline(
    *,
    argv: list[str],
    driver_path: Path,
    package_root: Path,
    work_root: Path,
    runtime_identity_sha256: str,
    runtime_environment: dict[str, str],
    preflight_only: bool,
    systemd_context: dict | None = None,
) -> int:
    """Run one worker process while the guardian owns the continuous StageServer."""
    if sys.platform != "linux":
        raise RuntimeError("full-process resource guarding requires the accepted Linux target runtime")
    if (
        len(runtime_identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in runtime_identity_sha256)
    ):
        raise ValueError("runtime identity must be a lowercase SHA-256 digest")
    if type(preflight_only) is not bool:
        raise ValueError("preflight_only must be boolean")
    if not preflight_only and (
        not isinstance(systemd_context, dict)
        or systemd_context.get("schema") != "n02-v4-systemd-context/1"
        or not systemd_context.get("invocation_id")
        or not systemd_context.get("control_group")
    ):
        raise PermissionError("formal pipeline supervisor must carry a verified systemd InvocationID and cgroup")
    required_mode = "--preflight" if preflight_only else "--run"
    other_mode = "--run" if preflight_only else "--preflight"
    if argv.count(required_mode) != 1 or other_mode in argv:
        raise PermissionError("guarded invocation must select exactly one complete-run or controlled-preflight mode")
    if any(value in argv for value in ("--_guardian-token", "--_guarded-pipeline")):
        raise PermissionError("internal guardian arguments cannot be supplied by the public invocation")

    from phase_handshake import N2_STAGE_ORDER, StageServer
    from resource_supervisor import (
        build_work_call_request,
        cumulative_budget_limits,
        _read_prior_resource_cost,
        _read_ledger,
        require_stage_budget,
    )

    stage_order = tuple(N2_STAGE_ORDER)
    if stage_order != STAGE_ORDER:
        raise RuntimeError("driver and StageServer stage orders disagree")
    package_root = require_package_root(package_root)
    work_root = Path(work_root).resolve()
    prior_cost = _read_prior_resource_cost(package_root / CARRY_FORWARD_RECEIPT_NAME, required=True)
    if preflight_only and not prior_cost["preflight_allowed"]:
        raise PermissionError("N2 target-runtime preflight is not permitted by the prior resource-cost state")
    if not preflight_only and not prior_cost["execution_allowed"]:
        raise PermissionError("N2 pipeline execution requires a measured prior resource-cost receipt")
    driver_path = Path(driver_path).resolve(strict=True)
    work_root.mkdir(parents=True, exist_ok=True)
    ledger_path = work_root / "resource-ledger.jsonl"
    work_call_path = work_root / "work-calls.sqlite3"
    stale_paths = (
        ledger_path,
        work_call_path,
        work_root / "worker-phase-accounting.jsonl",
        work_root / "worker-phase-protocol.jsonl",
        work_root / "supervisor-phase-protocol.jsonl",
        work_root / "receipts",
        work_root / "guardians",
    )
    if any(path.exists() for path in stale_paths):
        raise PermissionError("guarded pipeline requires a fresh work-root accounting chain")
    if not preflight_only:
        context_path = work_root / "systemd_context.json"
        if context_path.exists():
            raise PermissionError("systemd invocation evidence already exists; preserving the attempt")
        context_path.write_text(json.dumps(systemd_context, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if _read_ledger(ledger_path):
        raise PermissionError("guarded pipeline resource ledger must start empty")
    require_stage_budget(stage_order[0], prior_cost["total_wall_seconds"], prior_cost["total_cpu_seconds"])

    started_wall, started_cpu = _process_started_at()
    guardian_cpu_start = started_cpu
    export_root = work_root / EXPORT_DIR_NAME
    current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
    peak_work_disk = current_work_disk
    peak_export_disk = current_export_disk
    peak_group_rss = _proc_metrics(os.getpid())["rss_hwm_bytes"]
    child_rusage = None
    child_status = 0
    status_code = 1
    child_pid: int | None = None
    process = None
    server = None
    guardian_error: str | None = None
    kill_reason: str | None = None
    outer_error_at: float | None = None
    expected_stages = stage_order[:1] if preflight_only else stage_order
    measurements: dict[str, dict[str, float]] = {}
    failed_stage = stage_order[0]

    try:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {0})
            except OSError:
                pass
        server = StageServer(
            work_root / "phase-socket-placeholder",
            build_work_call_request(),
            tree_reader=_proc_tree,
            cpu_origin_seconds=started_cpu,
            wall_origin_monotonic=started_wall,
            journal_path=work_root / "supervisor-phase-protocol.jsonl",
        )
        token = uuid.uuid4().hex
        env = os.environ.copy()
        env.update(runtime_environment)
        env["RELIEF_GRAPH_GUARDIAN_TOKEN"] = token
        env["W1_PHASE_ACCOUNTING_SOCKET"] = str(server.path)
        child_argv = [*argv, "--_guarded-pipeline", "--_guardian-token", token]
        command = [sys.executable, "-B", str(driver_path), *child_argv]
        process = subprocess.Popen(
            command,
            cwd=str(package_root),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        child_pid = process.pid
        active_group = {os.getpid()}
        last_disk_sample = time.monotonic()
        while True:
            server.process_pending(child_pid)
            waited_pid, child_status, child_rusage = os.wait4(child_pid, os.WNOHANG)
            if waited_pid == child_pid:
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
                break

            now = time.monotonic()
            failed_stage = server.stage
            active_group = _proc_group_pids(child_pid)
            group_rss, group_cpu = _sample_group_rss(active_group)
            guardian_metrics = _proc_metrics(os.getpid())
            group_rss += guardian_metrics["rss_bytes"]
            peak_group_rss = max(peak_group_rss, group_rss)
            if now - last_disk_sample >= POLL_SECONDS:
                current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
                peak_work_disk = max(peak_work_disk, current_work_disk)
                peak_export_disk = max(peak_export_disk, current_export_disk)
                last_disk_sample = now

            elapsed_wall = now - started_wall
            elapsed_cpu = max(0.0, time.process_time() - guardian_cpu_start) + group_cpu
            stage_wall_limit, stage_cpu_limit = STAGE_LIMITS[server.stage]
            total_wall_limit, total_cpu_limit = cumulative_budget_limits(server.stage)
            stage_cost = prior_cost["stage_costs"].get(server.stage, {})
            current_stage_wall = max(0.0, now - server.wall_start)
            current_stage_cpu = max(0.0, time.process_time() + group_cpu - server.cpu_start)
            violations = []
            if float(stage_cost.get("wall_seconds", 0.0)) + current_stage_wall > stage_wall_limit:
                violations.append(f"{server.stage} wall budget exceeded")
            if float(stage_cost.get("cpu_seconds", 0.0)) + current_stage_cpu > stage_cpu_limit:
                violations.append(f"{server.stage} CPU budget exceeded")
            if prior_cost["total_wall_seconds"] + elapsed_wall > total_wall_limit:
                violations.append("whole-pilot wall budget exceeded")
            if prior_cost["total_cpu_seconds"] + elapsed_cpu > total_cpu_limit:
                violations.append("whole-pilot CPU budget exceeded")
            if peak_group_rss > RSS_LIMIT_BYTES:
                violations.append(f"process-group RSS {peak_group_rss}>{RSS_LIMIT_BYTES} bytes")
            if current_work_disk > WORK_STORAGE_LIMIT_BYTES:
                violations.append(f"work storage {current_work_disk}>{WORK_STORAGE_LIMIT_BYTES} bytes")
            if current_export_disk > EXPORT_STORAGE_LIMIT_BYTES:
                violations.append(f"export storage {current_export_disk}>{EXPORT_STORAGE_LIMIT_BYTES} bytes")
            if violations:
                kill_reason = "; ".join(violations)
                guardian_error = "outer guardian: " + kill_reason
                outer_error_at = now
                os.killpg(child_pid, signal.SIGKILL)
                _waited_pid, child_status, child_rusage = os.wait4(child_pid, 0)
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
                break
            time.sleep(POLL_SECONDS)
    except BaseException as exc:
        guardian_error = guardian_error or f"{type(exc).__name__}: {exc}"
        outer_error_at = outer_error_at or time.monotonic()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _waited_pid, child_status, child_rusage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(child_status)
                status_code = int(process.returncode)
            except ChildProcessError:
                pass
        if process is not None and process.returncode is not None:
            status_code = int(process.returncode)

    finished_wall = time.monotonic()
    child_cpu_seconds = 0.0
    child_peak_rss = 0
    if child_rusage is not None:
        child_cpu_seconds = float(child_rusage.ru_utime + child_rusage.ru_stime)
        child_peak_rss = int(child_rusage.ru_maxrss) * 1024
    try:
        guardian_final = _proc_metrics(os.getpid())
        peak_group_rss = max(peak_group_rss, child_peak_rss + guardian_final["rss_bytes"])
        current_work_disk, current_export_disk = _work_export_bytes(work_root, export_root)
        peak_work_disk = max(peak_work_disk, current_work_disk)
        peak_export_disk = max(peak_export_disk, current_export_disk)
        snapshots = []
        if process is not None and server is not None and (work_root / "worker-phase-accounting.jsonl").is_file():
            try:
                phase_rows = _phase_protocol_rows(work_root / "worker-phase-accounting.jsonl")
                snapshots = [row for row in phase_rows if row.get("event") == "snapshot"]
            except (OSError, PermissionError):
                snapshots = []
        snapshot_valid = bool(
            snapshots
            and snapshots[-1].get("pid") == child_pid
            and snapshots[-1].get("stage") == expected_stages[-1]
            and snapshots[-1].get("reason") == "ledger_close"
        )
        final_worker_cpu = (
            float(snapshots[-1]["cpu_end_seconds"])
            if snapshot_valid
            else child_cpu_seconds
        )
        final_worker_stage_cpu = (
            float(snapshots[-1]["cpu_seconds"])
            if snapshot_valid
            else max(0.0, child_cpu_seconds - float(server.last_worker_cpu_end if server is not None else 0.0))
        )
        if server is not None and child_pid is not None:
            try:
                measurements = _stage_measurements(
                    server,
                    final_worker_cpu,
                    final_worker_stage_cpu,
                    finished_wall,
                    started_cpu=started_cpu,
                    started_wall=started_wall,
                    preflight_only=preflight_only,
                )
            except (KeyError, TypeError, ValueError) as exc:
                guardian_error = guardian_error or f"stage resource measurement failed: {type(exc).__name__}: {exc}"
        if process is not None and process.returncode == 0 and server is not None:
            try:
                _validate_pipeline_handshakes(
                    server,
                    child_pid,
                    work_root,
                    stage_order,
                    preflight_only=preflight_only,
                )
                if not snapshot_valid:
                    raise PermissionError("worker did not record the final stage-close accounting snapshot")
                _validate_work_call_database(work_call_path)
                ledger_rows = _read_ledger_tolerating_partial_tail(ledger_path)
                if len(ledger_rows) != len(expected_stages):
                    raise PermissionError("worker did not settle exactly one resource row per guarded stage")
                for index, (stage, row) in enumerate(zip(expected_stages, ledger_rows, strict=True), 1):
                    receipt_path = work_root / "receipts" / f"{index:02d}-{stage}.json"
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    expected_receipt_status = "accepted" if stage == stage_order[0] else "complete"
                    if (
                        row.get("stage") != stage
                        or row.get("status") != expected_receipt_status
                        or row.get("runtime_identity_sha256") != runtime_identity_sha256
                        or row.get("release_digest") != runtime_identity_sha256
                        or receipt.get("schema") != DRIVER_STAGE_SCHEMA
                        or receipt.get("stage_protocol") != STAGE_PROTOCOL_ID
                        or receipt.get("stage") != stage
                        or receipt.get("status") != expected_receipt_status
                        or receipt.get("runtime_identity_sha256") != runtime_identity_sha256
                        or receipt.get("release_digest") != runtime_identity_sha256
                        or receipt.get("resource_entry") != row
                    ):
                        raise PermissionError(f"worker stage receipt chain failed validation at {stage}")
            except BaseException as exc:
                guardian_error = guardian_error or f"{type(exc).__name__}: {exc}"
                failed_stage = server.stage
        elif process is None:
            guardian_error = guardian_error or "outer guardian did not start a worker process"
    except BaseException as exc:
        guardian_error = guardian_error or f"{type(exc).__name__}: {exc}"
        outer_error_at = outer_error_at or time.monotonic()
        if server is not None:
            rejection = server.last_rejection or {}
            pending = server.pending_request or {}
            target = rejection.get("next_stage") or pending.get("payload", {}).get("next_stage")
            failed_stage = target if target in stage_order else server.stage
        else:
            failed_stage = stage_order[0]

    if server is not None and failed_stage == stage_order[0] and server.stage in stage_order:
        if status_code != 0 or guardian_error is not None:
            failed_stage = server.stage

    if process is not None and child_pid is not None:
        remaining_group = _proc_group_pids(child_pid)
        if remaining_group:
            guardian_error = guardian_error or "worker process group remained after child exit"
            kill_reason = kill_reason or guardian_error
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if server is not None:
        try:
            server.close()
        except BaseException as exc:
            guardian_error = guardian_error or f"StageServer close failed: {type(exc).__name__}: {exc}"

    if status_code == 0 and guardian_error is None:
        first_error = None
    else:
        first_error = guardian_error or f"guarded pipeline child exited with status {status_code}"
    prior_total_wall = prior_cost["total_wall_seconds"]
    prior_total_cpu = prior_cost["total_cpu_seconds"]
    stage_wall_total = sum(item["wall_seconds"] for item in measurements.values())
    stage_cpu_total = sum(item["cpu_seconds"] for item in measurements.values())
    final_violations = []
    if prior_total_wall + stage_wall_total > TOTAL_WALL_LIMIT:
        final_violations.append("whole-pilot wall budget exceeded at exit")
    if prior_total_cpu + stage_cpu_total > TOTAL_CPU_LIMIT:
        final_violations.append("whole-pilot CPU budget exceeded at exit")
    if peak_group_rss > RSS_LIMIT_BYTES:
        final_violations.append(f"process-group RSS {peak_group_rss}>{RSS_LIMIT_BYTES} bytes")
    if peak_work_disk > WORK_STORAGE_LIMIT_BYTES:
        final_violations.append(f"work storage {peak_work_disk}>{WORK_STORAGE_LIMIT_BYTES} bytes")
    if peak_export_disk > EXPORT_STORAGE_LIMIT_BYTES:
        final_violations.append(f"export storage {peak_export_disk}>{EXPORT_STORAGE_LIMIT_BYTES} bytes")
    if final_violations:
        guardian_error = guardian_error or "outer guardian: " + "; ".join(final_violations)
        first_error = first_error or guardian_error

    guardian_invocation_cpu = max(0.0, time.process_time() - guardian_cpu_start)
    invocation_measurement = {
        "status": "measured",
        "scope": "guardian process start through worker exit, excluding final receipt writes",
        "wall_seconds": max(0.0, finished_wall - started_wall),
        "cpu_seconds": guardian_invocation_cpu + child_cpu_seconds,
        "child_cpu_seconds": child_cpu_seconds,
        "guardian_cpu_seconds": guardian_invocation_cpu,
        "source": "guardian time.monotonic/process_time plus wait4 child user and system CPU; stage allocation requires a committed phase boundary",
    }
    first_error = _write_pipeline_receipts(
        work_root=work_root,
        package_root=package_root,
        identity=runtime_identity_sha256,
        stage_order=stage_order,
        measurements=measurements,
        child_returncode=status_code,
        peak_group_rss=peak_group_rss,
        peak_work_disk=peak_work_disk,
        peak_export_disk=peak_export_disk,
        prior_cost=prior_cost,
        guardian_error=guardian_error,
        kill_reason=kill_reason,
        child_pid=child_pid,
        invocation_measurement=invocation_measurement,
        preflight_only=preflight_only,
        failed_stage=failed_stage,
    ) or first_error
    return 0 if first_error is None else (status_code if status_code != 0 else 1)
