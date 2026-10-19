"""Persistent stage and whole-pilot resource accounting."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from package_layout import require_prior_resource_cost_path
from protocol_contract import (
    N2_STAGE_ORDER,
    PACKAGE_RELEASE_ID,
    PROTOCOL_ID,
    STAGE_PROTOCOL_ID,
    validate_protocol_contract,
)

from budget_ledger import BudgetError, BudgetLedger, WORK_KIND_LIMITS


STAGE_ORDER = N2_STAGE_ORDER
RESOURCE_LEDGER_SCHEMA = "resource-ledger-entry/2"
DRIVER_STAGE_SCHEMA = "driver-stage-receipt/2"
OUTER_GUARDIAN_SCHEMA = "outer-process-guardian/2"
RESOURCE_COST_SCHEMA = "resource-cost-carry-forward/2"
PRIOR_RESOURCE_COST_STATUS_SCHEMA = "prior-resource-cost-status/1"
TOTAL_WALL_LIMIT = 1800.0
TOTAL_CPU_LIMIT = 900.0
SETTLEMENT_RESERVE_WALL_SECONDS = 120.0
SETTLEMENT_RESERVE_CPU_SECONDS = 90.0
WORK_STORAGE_LIMIT_BYTES = 1 * 1024**3
EXPORT_STORAGE_LIMIT_BYTES = 1 * 1024**3
EXPORT_DIR_NAME = "export"
# The task book defines cumulative limits, but no separate per-stage budgets.
# Each stage is bounded by the full cap while the ledger enforces the cumulative cap.
STAGE_LIMITS: dict[str, tuple[float, float]] = {
    stage: (TOTAL_WALL_LIMIT, TOTAL_CPU_LIMIT) for stage in STAGE_ORDER
}
RSS_LIMIT_BYTES = 4 * 1024**3
CARRY_FORWARD_RECEIPT_NAME = "PRIOR_RESOURCE_COST.json"
WORK_CALL_LEDGER_NAME = "work-calls.sqlite3"


class BudgetExceeded(RuntimeError):
    pass


def build_work_call_request() -> dict[str, Any]:
    """Return the frozen stage order and discrete work caps for this release."""
    return {
        "package_release": PACKAGE_RELEASE_ID,
        "protocol_id": PROTOCOL_ID,
        "stage_protocol": STAGE_PROTOCOL_ID,
        "initial_stage": STAGE_ORDER[0],
        "stage_order": list(STAGE_ORDER),
        "stages": {
            stage: {
                "wall_seconds": TOTAL_WALL_LIMIT,
                "complete_process_cpu_seconds": TOTAL_CPU_LIMIT,
            }
            for stage in STAGE_ORDER
        },
        "work_limits": dict(WORK_KIND_LIMITS),
    }


def _read_rss_metrics() -> tuple[int, int]:
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        current = None
        high_water = None
        for line in proc_status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                high_water = int(line.split()[1]) * 1024
        if current is None or high_water is None:
            raise RuntimeError("VmRSS or VmHWM missing from /proc/self/status")
        return current, high_water
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        get_process_memory_info.restype = wintypes.BOOL
        ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        return peak_bytes, peak_bytes
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("RSS measurement is unavailable on this platform") from exc


def _read_rss_bytes() -> int:
    """Return the kernel-measured high-water mark where the platform exposes it."""
    return _read_rss_metrics()[1]


def _disk_bytes(roots: tuple[Path, ...], *, excluded_roots: tuple[Path, ...] = ()) -> int:
    total = 0
    seen: set[Path] = set()
    excluded = tuple(root.resolve() for root in excluded_roots)
    for root in roots:
        if not root.exists():
            continue
        paths = (root,) if root.is_file() else root.rglob("*")
        for path in paths:
            try:
                resolved = path.resolve(strict=True)
                if (
                    resolved in seen
                    or path.is_symlink()
                    or not path.is_file()
                    or any(resolved == excluded_root or excluded_root in resolved.parents for excluded_root in excluded)
                ):
                    continue
                seen.add(resolved)
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _work_export_bytes(work_root: Path, export_root: Path) -> tuple[int, int]:
    export_bytes = _disk_bytes((export_root,))
    work_bytes = _disk_bytes((work_root,), excluded_roots=(export_root,))
    return work_bytes, export_bytes


def require_stage_budget(stage: str, prior_wall: float, prior_cpu: float) -> None:
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown budget stage: {stage}")
    remaining_wall = TOTAL_WALL_LIMIT - prior_wall
    remaining_cpu = TOTAL_CPU_LIMIT - prior_cpu
    if stage == STAGE_ORDER[-1]:
        if remaining_wall < SETTLEMENT_RESERVE_WALL_SECONDS or remaining_cpu < SETTLEMENT_RESERVE_CPU_SECONDS:
            raise BudgetExceeded("remaining cumulative budget cannot preserve the settlement/export reserve")
    elif (
        remaining_wall <= SETTLEMENT_RESERVE_WALL_SECONDS
        or remaining_cpu <= SETTLEMENT_RESERVE_CPU_SECONDS
    ):
        raise BudgetExceeded("remaining cumulative budget is at the settlement/export reserve; no work stage may start")


def cumulative_budget_limits(stage: str) -> tuple[float, float]:
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown budget stage: {stage}")
    if stage == STAGE_ORDER[-1]:
        return TOTAL_WALL_LIMIT, TOTAL_CPU_LIMIT
    return (
        TOTAL_WALL_LIMIT - SETTLEMENT_RESERVE_WALL_SECONDS,
        TOTAL_CPU_LIMIT - SETTLEMENT_RESERVE_CPU_SECONDS,
    )


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid resource ledger line {line_number}") from exc
            if (
                row.get("schema") != RESOURCE_LEDGER_SCHEMA
                or row.get("stage_protocol") != STAGE_PROTOCOL_ID
                or row.get("stage") not in STAGE_ORDER
                or not isinstance(row.get("metadata"), dict)
                or not isinstance(row.get("runtime_identity_sha256"), str)
                or row.get("release_digest") != row.get("runtime_identity_sha256")
                or row.get("metadata", {}).get("runtime_identity_sha256") != row.get("runtime_identity_sha256")
                or row.get("metadata", {}).get("release_digest") != row.get("release_digest")
            ):
                raise ValueError(f"unsupported resource ledger line {line_number}")
            rows.append(row)
    return rows


def _read_prior_resource_cost(path: str | Path | None, *, required: bool = False) -> dict[str, Any]:
    empty = {
        "stage_costs": {},
        "total_wall_seconds": 0.0,
        "total_cpu_seconds": 0.0,
        "receipt_sha256": None,
        "status_sha256": None,
        "measurement_status": "not_provided",
        "preflight_allowed": True,
        "execution_allowed": False,
    }
    if path is None:
        if required:
            raise FileNotFoundError("required prior resource-cost receipt path was not provided")
        return empty
    candidate = Path(path)
    if not candidate.is_file():
        if required:
            raise FileNotFoundError(f"required prior resource-cost receipt is missing: {candidate}")
        return empty
    receipt_path = require_prior_resource_cost_path(path)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if receipt.get("schema") == PRIOR_RESOURCE_COST_STATUS_SCHEMA:
        measurement = receipt.get("measurement")
        known = receipt.get("known_consumed")
        gates = receipt.get("gates")
        validate_protocol_contract(
            package_release=receipt.get("package_release"),
            protocol_id=receipt.get("protocol_id"),
            stage_protocol=receipt.get("stage_protocol"),
            stage_order=receipt.get("stage_order", ()),
        )
        if (
            receipt.get("stage_protocol") != STAGE_PROTOCOL_ID
            or receipt.get("status") != "no_current_engineering_runs"
            or known != {"wall_seconds": 0.0, "cpu_seconds": 0.0}
            or not isinstance(measurement, dict)
            or measurement.get("status") != "pending"
            or measurement.get("actual") is not None
            or not isinstance(gates, dict)
            or gates.get("target_runtime_preflight_allowed") is not True
            or gates.get("pipeline_execution_allowed") is not False
            or gates.get("request_freeze_allowed") is not False
            or receipt.get("validated_receipt") is not None
            or "evidence_files" in receipt
        ):
            raise ValueError("invalid pending prior resource-cost status")
        return {
            "stage_costs": {},
            "total_wall_seconds": 0.0,
            "total_cpu_seconds": 0.0,
            "receipt_sha256": None,
            "status_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "measurement_status": "pending",
            "preflight_allowed": True,
            "execution_allowed": False,
        }
    validate_protocol_contract(
        package_release=receipt.get("package_release"),
        protocol_id=receipt.get("protocol_id"),
        stage_protocol=receipt.get("stage_protocol"),
        stage_order=receipt.get("stage_order", ()),
    )
    if receipt.get("schema") != RESOURCE_COST_SCHEMA:
        raise ValueError("unsupported prior resource-cost receipt")

    evidence_files = receipt.get("evidence_files")
    if not isinstance(evidence_files, list) or not evidence_files:
        raise ValueError("prior resource-cost receipt has no evidence hashes")
    evidence_hashes: dict[str, str] = {}
    root = receipt_path.parent
    for item in evidence_files:
        if not isinstance(item, dict):
            raise ValueError("invalid prior resource-cost evidence row")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative or not isinstance(expected, str):
            raise ValueError("invalid prior resource-cost evidence path or hash")
        evidence_path = (root / relative).resolve(strict=True)
        if root not in evidence_path.parents or not evidence_path.is_file():
            raise ValueError("prior resource-cost evidence escapes its package or is not a file")
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"prior resource-cost evidence hash mismatch: {relative}")
        evidence_hashes[relative] = digest

    roles = receipt.get("evidence_roles")
    if not isinstance(roles, dict):
        raise ValueError("prior resource-cost receipt has no evidence roles")
    required_roles = {"resource_ledger", "systemd_show", "deployment_receipt"}
    if not required_roles.issubset(roles):
        raise ValueError("prior resource-cost receipt is missing accounting evidence roles")
    if any(roles[name] not in evidence_hashes for name in required_roles):
        raise ValueError("prior resource-cost accounting evidence is not hash-bound")

    deployment = json.loads((root / roles["deployment_receipt"]).read_text(encoding="utf-8"))
    if (
        deployment.get("source_identity_sha256") != receipt.get("source_identity_sha256")
        or deployment.get("archive_sha256") != receipt.get("deployed_package_archive_sha256")
    ):
        raise ValueError("prior resource-cost receipt does not match its deployment evidence")

    ledger_rows = _read_ledger(root / roles["resource_ledger"])
    ledger_stage_costs: dict[str, dict[str, float]] = {}
    for row in ledger_rows:
        stage = row.get("stage")
        if stage not in STAGE_LIMITS:
            raise ValueError("prior resource ledger contains an unknown stage")
        cost = ledger_stage_costs.setdefault(stage, {"wall_seconds": 0.0, "cpu_seconds": 0.0})
        cost["wall_seconds"] += float(row["wall_seconds"])
        cost["cpu_seconds"] += float(row["cpu_seconds"])
    if ledger_stage_costs != receipt.get("ledger_stage_costs"):
        for stage in set(ledger_stage_costs) | set(receipt.get("ledger_stage_costs", {})):
            measured = ledger_stage_costs.get(stage, {"wall_seconds": 0.0, "cpu_seconds": 0.0})
            declared = receipt.get("ledger_stage_costs", {}).get(stage, {"wall_seconds": 0.0, "cpu_seconds": 0.0})
            if any(
                not math.isclose(float(measured[key]), float(declared[key]), rel_tol=1e-12, abs_tol=1e-9)
                for key in ("wall_seconds", "cpu_seconds")
            ):
                raise ValueError("prior resource-cost ledger totals do not match the source ledger")

    systemd_text = (root / roles["systemd_show"]).read_text(encoding="utf-8")
    systemd_values = {
        key: value
        for line in systemd_text.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    invocation_id = systemd_values.get("InvocationID")
    if invocation_id != receipt.get("systemd_invocation_id"):
        raise ValueError("prior resource-cost receipt does not match the systemd invocation")
    try:
        systemd_wall = (
            int(systemd_values["ExecMainExitTimestampMonotonic"])
            - int(systemd_values["ExecMainStartTimestampMonotonic"])
        ) / 1_000_000
        systemd_cpu = int(systemd_values["CPUUsageNSec"]) / 1_000_000_000
    except (KeyError, ValueError) as exc:
        raise ValueError("prior systemd accounting fields are missing or invalid") from exc
    if not math.isclose(systemd_wall, float(receipt["systemd_total_wall_seconds"]), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("prior resource-cost wall total does not match systemd")
    if not math.isclose(systemd_cpu, float(receipt["systemd_total_cpu_seconds"]), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("prior resource-cost CPU total does not match systemd")

    ledger_wall = sum(cost["wall_seconds"] for cost in ledger_stage_costs.values())
    ledger_cpu = sum(cost["cpu_seconds"] for cost in ledger_stage_costs.values())
    wall_overhead = systemd_wall - ledger_wall
    cpu_overhead = systemd_cpu - ledger_cpu
    if wall_overhead < -1e-9 or cpu_overhead < -1e-9:
        raise ValueError("systemd totals are below the prior process-guardian ledger")
    declared_overhead = receipt.get("systemd_wrapper_overhead_assigned_to_input_admission", {})
    if any(
        not math.isclose(
            max(0.0, actual),
            float(declared_overhead.get(field, float("nan"))),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for actual, field in ((wall_overhead, "wall_seconds"), (cpu_overhead, "cpu_seconds"))
    ):
        raise ValueError("prior systemd wrapper overhead is not reconciled")
    expected_stage_costs = {
        stage: {"wall_seconds": cost["wall_seconds"], "cpu_seconds": cost["cpu_seconds"]}
        for stage, cost in ledger_stage_costs.items()
    }
    admission = expected_stage_costs.setdefault(STAGE_ORDER[0], {"wall_seconds": 0.0, "cpu_seconds": 0.0})
    admission["wall_seconds"] += max(0.0, wall_overhead)
    admission["cpu_seconds"] += max(0.0, cpu_overhead)
    declared_stage_costs = receipt.get("stage_costs")
    if not isinstance(declared_stage_costs, dict) or set(declared_stage_costs) != set(expected_stage_costs):
        raise ValueError("prior resource-cost stage totals are invalid")
    for stage, measured in expected_stage_costs.items():
        declared = declared_stage_costs[stage]
        if any(
            not math.isclose(float(measured[key]), float(declared[key]), rel_tol=1e-12, abs_tol=1e-9)
            for key in ("wall_seconds", "cpu_seconds")
        ):
            raise ValueError("prior resource-cost stage totals do not reconcile")

    total_wall = sum(cost["wall_seconds"] for cost in declared_stage_costs.values())
    total_cpu = sum(cost["cpu_seconds"] for cost in declared_stage_costs.values())
    if not math.isclose(total_wall, systemd_wall, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("prior resource-cost wall stages do not sum to systemd")
    if not math.isclose(total_cpu, systemd_cpu, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("prior resource-cost CPU stages do not sum to systemd")
    return {
        "stage_costs": declared_stage_costs,
        "total_wall_seconds": total_wall,
        "total_cpu_seconds": total_cpu,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "status_sha256": None,
        "measurement_status": "measured",
        "preflight_allowed": True,
        "execution_allowed": True,
    }


@dataclass(slots=True)
class ResourceSupervisor:
    stage: str
    ledger_path: Path
    work_root: Path
    export_root: Path
    runtime_identity_sha256: str
    rss_limit_bytes: int = RSS_LIMIT_BYTES
    work_limit_bytes: int = WORK_STORAGE_LIMIT_BYTES
    export_limit_bytes: int = EXPORT_STORAGE_LIMIT_BYTES
    monitor_interval_seconds: float = 0.25
    interrupt_on_violation: bool = True
    prior_cost_path: Path | None = None
    invocation_start_wall: float | None = None
    invocation_start_cpu: float | None = None
    work_ledger: BudgetLedger | None = None
    _ledger_before: list[dict[str, Any]] = field(init=False, repr=False)
    _prior_cost: dict[str, Any] = field(init=False, repr=False)
    _stage_wall_limit: float = field(init=False, repr=False)
    _stage_cpu_limit: float = field(init=False, repr=False)
    _prior_stage_wall: float = field(init=False, repr=False)
    _prior_stage_cpu: float = field(init=False, repr=False)
    _prior_total_wall: float = field(init=False, repr=False)
    _prior_total_cpu: float = field(init=False, repr=False)
    _start_wall: float = field(init=False, repr=False)
    _start_cpu: float = field(init=False, repr=False)
    _peak_rss: int = field(init=False, repr=False)
    _peak_work_disk: int = field(init=False, repr=False)
    _peak_export_disk: int = field(init=False, repr=False)
    _last_disk_check: float = field(init=False, repr=False)
    _current_work_disk: int = field(init=False, repr=False)
    _current_export_disk: int = field(init=False, repr=False)
    _first_error: str | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _finished_row: dict[str, Any] | None = field(init=False, default=None, repr=False)
    _monitor_thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _monitor_stop: threading.Event = field(init=False, repr=False)
    _monitor_violation: threading.Event = field(init=False, repr=False)
    _interrupt_sent: bool = field(init=False, default=False, repr=False)
    _previous_signal_handler: Any = field(init=False, default=None, repr=False)
    _signal_number: int | None = field(init=False, default=None, repr=False)
    _state_lock: threading.RLock = field(init=False, repr=False)
    _stage_rows: list[dict[str, Any]] = field(init=False, repr=False)
    _stage_selected: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.stage not in STAGE_LIMITS:
            raise ValueError(f"unknown budget stage: {self.stage}")
        self.ledger_path = Path(self.ledger_path)
        self.work_root = Path(self.work_root)
        self.export_root = Path(self.export_root)
        if (
            len(self.runtime_identity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.runtime_identity_sha256)
        ):
            raise ValueError("runtime identity must be a lowercase SHA-256 digest")
        self._ledger_before = _read_ledger(self.ledger_path)
        if any(
            row.get("runtime_identity_sha256") != self.runtime_identity_sha256
            or row.get("release_digest") != self.runtime_identity_sha256
            or row.get("metadata", {}).get("runtime_identity_sha256") != self.runtime_identity_sha256
            or row.get("metadata", {}).get("release_digest") != self.runtime_identity_sha256
            for row in self._ledger_before
        ):
            raise ValueError("resource ledger belongs to a different package, data, runtime, or work-root identity")
        self._prior_cost = _read_prior_resource_cost(self.prior_cost_path, required=self.prior_cost_path is not None)
        self._stage_wall_limit, self._stage_cpu_limit = STAGE_LIMITS[self.stage]
        carry_stage = self._prior_cost["stage_costs"].get(self.stage, {})
        self._prior_stage_wall = float(carry_stage.get("wall_seconds", 0.0)) + sum(row["wall_seconds"] for row in self._ledger_before if row["stage"] == self.stage)
        self._prior_stage_cpu = float(carry_stage.get("cpu_seconds", 0.0)) + sum(row["cpu_seconds"] for row in self._ledger_before if row["stage"] == self.stage)
        self._prior_total_wall = self._prior_cost["total_wall_seconds"] + sum(row["wall_seconds"] for row in self._ledger_before)
        self._prior_total_cpu = self._prior_cost["total_cpu_seconds"] + sum(row["cpu_seconds"] for row in self._ledger_before)
        require_stage_budget(self.stage, self._prior_total_wall, self._prior_total_cpu)
        self._start_wall = self.invocation_start_wall if self.invocation_start_wall is not None else time.monotonic()
        self._start_cpu = self.invocation_start_cpu if self.invocation_start_cpu is not None else time.process_time()
        self._peak_rss = _read_rss_bytes()
        self._current_work_disk, self._current_export_disk = _work_export_bytes(self.work_root, self.export_root)
        self._peak_work_disk = self._current_work_disk
        self._peak_export_disk = self._current_export_disk
        self._last_disk_check = time.monotonic()
        self._first_error: str | None = None
        self._closed = False
        self._finished_row = None
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self._monitor_violation = threading.Event()
        self._interrupt_sent = False
        self._previous_signal_handler = None
        self._signal_number = None
        self._state_lock = threading.RLock()
        self._stage_rows = []
        if self.work_ledger is None:
            self.work_ledger = BudgetLedger(
                self.work_root / WORK_CALL_LEDGER_NAME,
                build_work_call_request(),
            )

    def start_monitor(self) -> None:
        if self._monitor_thread is not None:
            raise RuntimeError("resource monitor already started")
        if self.interrupt_on_violation and os.name == "posix" and threading.current_thread() is threading.main_thread():
            signal_number = getattr(signal, "SIGUSR1", None)
            if signal_number is not None:
                self._signal_number = signal_number
                self._previous_signal_handler = signal.getsignal(signal_number)
                signal.signal(signal_number, self._interrupt_handler)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"resource-watch-{self.stage}",
            daemon=True,
        )
        self._monitor_thread.start()

    def select_stage(self, stage: str) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("resource supervisor is already closed")
        stage_order = tuple(self.work_ledger.request.get("stage_order", ()))
        if stage_order != STAGE_ORDER or stage not in STAGE_ORDER:
            raise BudgetError("work-call Request stage order is invalid")
        if not self._stage_selected:
            if stage != STAGE_ORDER[0] or stage != self.stage:
                raise BudgetError("first selected stage must be input_admission")
            self.work_ledger.select(stage)
            self._stage_selected = True
            return None
        if self.stage == stage:
            raise BudgetError("stage may not be selected twice")
        current_index = STAGE_ORDER.index(self.stage)
        if current_index + 1 >= len(STAGE_ORDER) or STAGE_ORDER[current_index + 1] != stage:
            raise BudgetError(f"out-of-order ResourceSupervisor stage: {self.stage} -> {stage}")
        expected_previous_status = "accepted" if self.stage == STAGE_ORDER[0] else "complete"
        if self._finished_row is None or self._finished_row.get("status") != expected_previous_status:
            raise BudgetError("cannot leave an unsettled or unsuccessful stage")
        self.check(force_disk=True)
        self.work_ledger.select(stage)
        row = self._finished_row
        self._ledger_before.append(dict(row))
        self._prior_stage_wall = float(self._prior_cost["stage_costs"].get(stage, {}).get("wall_seconds", 0.0)) + sum(
            item["wall_seconds"] for item in self._ledger_before if item["stage"] == stage
        )
        self._prior_stage_cpu = float(self._prior_cost["stage_costs"].get(stage, {}).get("cpu_seconds", 0.0)) + sum(
            item["cpu_seconds"] for item in self._ledger_before if item["stage"] == stage
        )
        self._prior_total_wall += float(row["wall_seconds"])
        self._prior_total_cpu += float(row["cpu_seconds"])
        self.stage = stage
        self._start_wall = time.monotonic()
        self._start_cpu = time.process_time()
        self._peak_rss = _read_rss_bytes()
        self._current_work_disk, self._current_export_disk = _work_export_bytes(self.work_root, self.export_root)
        self._peak_work_disk = self._current_work_disk
        self._peak_export_disk = self._current_export_disk
        self._first_error = None
        self._finished_row = None
        return dict(row)

    def _interrupt_handler(self, _signum, _frame) -> None:
        raise BudgetExceeded(self._first_error or f"{self.stage} resource limit exceeded")

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self.monitor_interval_seconds):
            try:
                self._monitor_sample()
            except BudgetExceeded:
                self._monitor_violation.set()
                if self._signal_number is not None and not self._interrupt_sent:
                    self._interrupt_sent = True
                    try:
                        os.kill(os.getpid(), self._signal_number)
                    except OSError:
                        pass
                return
            except BaseException as exc:
                with self._state_lock:
                    self._first_error = self._first_error or f"resource monitor failed: {type(exc).__name__}: {exc}"
                self._monitor_violation.set()
                if self._signal_number is not None and not self._interrupt_sent:
                    self._interrupt_sent = True
                    try:
                        os.kill(os.getpid(), self._signal_number)
                    except OSError:
                        pass
                return

    def _sample_and_check(self, *, force_disk: bool = False, allow_finished: bool = False) -> None:
        with self._state_lock:
            if self._closed or (self._finished_row is not None and not allow_finished):
                return
            wall = self.wall_seconds
            cpu = self.cpu_seconds
            _current_rss, rss_high_water = _read_rss_metrics()
            if force_disk or time.monotonic() - self._last_disk_check >= 1.0:
                self._current_work_disk, self._current_export_disk = _work_export_bytes(self.work_root, self.export_root)
                self._last_disk_check = time.monotonic()
            self._peak_rss = max(self._peak_rss, rss_high_water)
            self._peak_work_disk = max(self._peak_work_disk, self._current_work_disk)
            self._peak_export_disk = max(self._peak_export_disk, self._current_export_disk)
            violations = self._limit_violations(
                wall,
                cpu,
                self._peak_rss,
                self._peak_work_disk,
                self._peak_export_disk,
            )
            if violations:
                self._first_error = self._first_error or "; ".join(violations)
                raise BudgetExceeded(self._first_error)

    def _monitor_sample(self) -> None:
        self._sample_and_check(force_disk=time.monotonic() - self._last_disk_check >= 1.0, allow_finished=True)

    @property
    def wall_seconds(self) -> float:
        return time.monotonic() - self._start_wall

    @property
    def cpu_seconds(self) -> float:
        return time.process_time() - self._start_cpu

    def _limit_violations(self, wall: float, cpu: float, rss: int, work_disk: int, export_disk: int) -> list[str]:
        stage_wall = self._prior_stage_wall + wall
        stage_cpu = self._prior_stage_cpu + cpu
        total_wall = self._prior_total_wall + wall
        total_cpu = self._prior_total_cpu + cpu
        violations = []
        if stage_wall > self._stage_wall_limit:
            violations.append(f"{self.stage} wall {stage_wall:.3f}>{self._stage_wall_limit:.0f}s")
        if stage_cpu > self._stage_cpu_limit:
            violations.append(f"{self.stage} CPU {stage_cpu:.3f}>{self._stage_cpu_limit:.0f}s")
        total_wall_limit, total_cpu_limit = cumulative_budget_limits(self.stage)
        if total_wall > total_wall_limit:
            violations.append(f"whole-pilot wall {total_wall:.3f}>{total_wall_limit:.0f}s (settlement reserve)")
        if total_cpu > total_cpu_limit:
            violations.append(f"whole-pilot CPU {total_cpu:.3f}>{total_cpu_limit:.0f}s (settlement reserve)")
        if rss > self.rss_limit_bytes:
            violations.append(f"RSS {rss}>{self.rss_limit_bytes} bytes")
        if work_disk > self.work_limit_bytes:
            violations.append(f"work storage {work_disk}>{self.work_limit_bytes} bytes")
        if export_disk > self.export_limit_bytes:
            violations.append(f"export storage {export_disk}>{self.export_limit_bytes} bytes")
        return violations

    def check(self, *, force_disk: bool = False) -> None:
        if self._closed:
            raise RuntimeError("resource supervisor is already closed")
        self._sample_and_check(force_disk=force_disk)

    def reserve_export_bytes(self, additional_bytes: int) -> None:
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("export reservation must be a nonnegative integer byte count")
        self.check(force_disk=True)
        projected = self._current_export_disk + additional_bytes
        if projected > self.export_limit_bytes:
            raise BudgetExceeded(f"export storage projection {projected}>{self.export_limit_bytes} bytes")

    def reserve_work_call(self, kind: str, estimated_count: int, metadata: dict | None = None) -> int:
        self.check()
        if self.work_ledger is None:
            raise BudgetError("SQLite work-call ledger is not attached to ResourceSupervisor")
        bound = dict(metadata or {})
        bound.setdefault("runtime_identity_sha256", self.runtime_identity_sha256)
        bound.setdefault("release_digest", self.runtime_identity_sha256)
        if bound["runtime_identity_sha256"] != self.runtime_identity_sha256 or bound["release_digest"] != self.runtime_identity_sha256:
            raise BudgetError("work-call metadata does not bind the active release")
        return self.work_ledger.reserve_work_call(kind, estimated_count, bound)

    def complete_work_call(self, token: int, actual_count: int, metadata: dict | None = None) -> None:
        if self.work_ledger is None:
            raise BudgetError("SQLite work-call ledger is not attached to ResourceSupervisor")
        bound = dict(metadata or {})
        bound.setdefault("runtime_identity_sha256", self.runtime_identity_sha256)
        bound.setdefault("release_digest", self.runtime_identity_sha256)
        if bound["runtime_identity_sha256"] != self.runtime_identity_sha256 or bound["release_digest"] != self.runtime_identity_sha256:
            raise BudgetError("work-call metadata does not bind the active release")
        self.work_ledger.complete_work_call(token, actual_count, bound)

    def fail_work_call(self, token: int, error: BaseException | str) -> None:
        if self.work_ledger is None:
            raise BudgetError("SQLite work-call ledger is not attached to ResourceSupervisor")
        self.work_ledger.fail_work_call(token, error)

    def work_call_snapshot(self) -> dict[str, Any]:
        if self.work_ledger is None:
            raise BudgetError("SQLite work-call ledger is not attached to ResourceSupervisor")
        return self.work_ledger.work_snapshot()

    def assert_work_calls_settled(self) -> dict[str, Any]:
        if self.work_ledger is None:
            raise BudgetError("SQLite work-call ledger is not attached to ResourceSupervisor")
        snapshot = self.work_ledger.work_snapshot()
        if snapshot["pending_calls"]:
            raise BudgetError("SQLite work-call ledger has pending calls")
        if snapshot["failed_calls"]:
            raise BudgetError("SQLite work-call ledger has failed calls")
        return snapshot

    def record_error(self, error: BaseException | str) -> None:
        if self._first_error is None:
            if isinstance(error, BaseException):
                self._first_error = f"{type(error).__name__}: {error}"
            else:
                self._first_error = str(error)

    def finish(self, *, status: str, error: BaseException | str | None = None, metadata: dict | None = None) -> dict:
        if not self._stage_selected:
            raise BudgetError("a stage must be selected before finishing")
        if self._finished_row is not None:
            raise RuntimeError("resource supervisor has already written its stage entry")
        if error is None:
            self.check(force_disk=True)
            self.work_ledger.assert_stage_settled(self.stage)
        if error is not None:
            self.record_error(error)
        finished_wall = time.monotonic()
        finished_cpu = time.process_time()
        _current_rss, rss = _read_rss_metrics()
        self._current_work_disk, self._current_export_disk = _work_export_bytes(self.work_root, self.export_root)
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_work_disk = max(self._peak_work_disk, self._current_work_disk)
        self._peak_export_disk = max(self._peak_export_disk, self._current_export_disk)
        if error is None:
            violations = self._limit_violations(
                finished_wall - self._start_wall,
                finished_cpu - self._start_cpu,
                rss,
                self._peak_work_disk,
                self._peak_export_disk,
            )
            if violations:
                self._first_error = self._first_error or "; ".join(violations)
                raise BudgetExceeded(self._first_error)
        entry_metadata = dict(metadata or {})
        bound_identity = entry_metadata.setdefault("runtime_identity_sha256", self.runtime_identity_sha256)
        if bound_identity != self.runtime_identity_sha256:
            raise ValueError("resource entry metadata does not match the active runtime identity")
        bound_release = entry_metadata.setdefault("release_digest", self.runtime_identity_sha256)
        if bound_release != self.runtime_identity_sha256:
            raise ValueError("resource entry metadata does not match the active release digest")
        row = {
            "schema": RESOURCE_LEDGER_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "stage": self.stage,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "release_digest": self.runtime_identity_sha256,
            "status": status,
            "started_wall_monotonic": self._start_wall,
            "finished_wall_monotonic": finished_wall,
            "wall_seconds": finished_wall - self._start_wall,
            "cpu_seconds": finished_cpu - self._start_cpu,
            "peak_rss_bytes": self._peak_rss,
            "rss_high_water_source": "linux:/proc/self/status:VmHWM" if Path("/proc/self/status").is_file() else "platform process high-water API",
            "peak_work_bytes": self._peak_work_disk,
            "peak_export_bytes": self._peak_export_disk,
            "rss_limit_bytes": self.rss_limit_bytes,
            "work_limit_bytes": self.work_limit_bytes,
            "export_limit_bytes": self.export_limit_bytes,
            "stage_wall_used_including_this": self._prior_stage_wall + (finished_wall - self._start_wall),
            "stage_cpu_used_including_this": self._prior_stage_cpu + (finished_cpu - self._start_cpu),
            "pilot_wall_used_including_this": self._prior_total_wall + (finished_wall - self._start_wall),
            "pilot_cpu_used_including_this": self._prior_total_cpu + (finished_cpu - self._start_cpu),
            "first_error": self._first_error,
            "metadata": {
                **entry_metadata,
                **({"prior_resource_cost_sha256": self._prior_cost["receipt_sha256"]} if self._prior_cost["receipt_sha256"] else {}),
            },
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._finished_row = row
        self._stage_rows.append(dict(row))
        return dict(row)

    def close(self, *, error: BaseException | str | None = None) -> dict[str, Any]:
        if self._closed:
            if error is not None:
                return self._revise_finished_entry(error)
            return dict(self._finished_row or {})
        if error is not None:
            self.record_error(error)
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=max(1.0, self.monitor_interval_seconds * 4))
            if self._monitor_thread.is_alive():
                self.record_error("resource monitor did not stop")
        if self._signal_number is not None and threading.current_thread() is threading.main_thread():
            signal.signal(self._signal_number, self._previous_signal_handler)
        self._sample_final_metrics()
        if self._finished_row is None:
            if error is None:
                raise RuntimeError("resource supervisor cannot close before writing a stage entry")
            if not self._stage_selected:
                if self.work_ledger is not None:
                    self.work_ledger.close(snapshot=False)
                self._closed = True
                return {
                    "schema": "pre-stage-failure-observation/1",
                    "stage_protocol": STAGE_PROTOCOL_ID,
                    "stage": self.stage,
                    "runtime_identity_sha256": self.runtime_identity_sha256,
                    "release_digest": self.runtime_identity_sha256,
                    "status": "failed",
                    "first_error": self._first_error or str(error),
                    "stage_selected": False,
                    "stage_state": "stage_not_entered",
                    "resource_measurement": {
                        "status": "unknown",
                        "wall_seconds": None,
                        "cpu_seconds": None,
                        "scope": "stage",
                        "reason": "no stage_enter boundary was committed",
                    },
                    "work_call_ledger_close": {
                        "snapshot_written": False,
                        "call_statuses_mutated": False,
                    },
                }
            self.finish(status="failed", error=error)
        if self._first_error is not None:
            self._finished_row["status"] = "failed"
            self._finished_row["first_error"] = self._first_error
        self._write_revised_last_entry()
        if self._stage_rows:
            self._stage_rows[-1] = dict(self._finished_row)
        self.work_ledger.close(snapshot=True)
        self._closed = True
        return dict(self._finished_row)

    def _sample_final_metrics(self) -> None:
        with self._state_lock:
            finished_wall = time.monotonic()
            finished_cpu = time.process_time()
            _current_rss, rss = _read_rss_metrics()
            self._current_work_disk, self._current_export_disk = _work_export_bytes(self.work_root, self.export_root)
            self._peak_rss = max(self._peak_rss, rss)
            self._peak_work_disk = max(self._peak_work_disk, self._current_work_disk)
            self._peak_export_disk = max(self._peak_export_disk, self._current_export_disk)
            violations = self._limit_violations(
                finished_wall - self._start_wall,
                finished_cpu - self._start_cpu,
                self._peak_rss,
                self._peak_work_disk,
                self._peak_export_disk,
            )
            if violations:
                self._first_error = self._first_error or "; ".join(violations)
            if self._finished_row is not None:
                row = self._finished_row
                row.update({
                    "finished_wall_monotonic": finished_wall,
                    "wall_seconds": finished_wall - self._start_wall,
                    "cpu_seconds": finished_cpu - self._start_cpu,
                    "peak_rss_bytes": self._peak_rss,
                    "peak_work_bytes": self._peak_work_disk,
                    "peak_export_bytes": self._peak_export_disk,
                    "stage_wall_used_including_this": self._prior_stage_wall + (finished_wall - self._start_wall),
                    "stage_cpu_used_including_this": self._prior_stage_cpu + (finished_cpu - self._start_cpu),
                    "pilot_wall_used_including_this": self._prior_total_wall + (finished_wall - self._start_wall),
                    "pilot_cpu_used_including_this": self._prior_total_cpu + (finished_cpu - self._start_cpu),
                })

    def _revise_finished_entry(self, error: BaseException | str) -> dict[str, Any]:
        self.record_error(error)
        if self._finished_row is not None:
            self._finished_row["status"] = "failed"
            self._finished_row["first_error"] = self._first_error
            self._write_revised_last_entry()
        return dict(self._finished_row or {})

    def _write_revised_last_entry(self) -> None:
        rows = _read_ledger(self.ledger_path)
        if not rows or rows[-1].get("started_wall_monotonic") != self._start_wall:
            raise RuntimeError("resource ledger changed while the stage was active")
        rows[-1] = dict(self._finished_row or {})
        temporary = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.ledger_path)


def ledger_summary(path: str | Path, prior_cost_path: str | Path | None = None) -> dict[str, Any]:
    rows = _read_ledger(Path(path))
    prior = _read_prior_resource_cost(prior_cost_path, required=prior_cost_path is not None)
    stage_wall = {stage: sum(row["wall_seconds"] for row in rows if row["stage"] == stage) for stage in STAGE_LIMITS}
    stage_cpu = {stage: sum(row["cpu_seconds"] for row in rows if row["stage"] == stage) for stage in STAGE_LIMITS}
    for stage, cost in prior["stage_costs"].items():
        stage_wall[stage] += cost["wall_seconds"]
        stage_cpu[stage] += cost["cpu_seconds"]
    wall = prior["total_wall_seconds"] + sum(row["wall_seconds"] for row in rows)
    cpu = prior["total_cpu_seconds"] + sum(row["cpu_seconds"] for row in rows)
    return {
        "entries": rows,
        "stage_wall_seconds": stage_wall,
        "stage_cpu_seconds": stage_cpu,
        "total_wall_seconds": wall,
        "total_cpu_seconds": cpu,
        "prior_resource_cost_sha256": prior["receipt_sha256"],
        "total_wall_limit_seconds": TOTAL_WALL_LIMIT,
        "total_cpu_limit_seconds": TOTAL_CPU_LIMIT,
        "all_successful": all(row["status"] in {"complete", "accepted"} for row in rows),
    }
