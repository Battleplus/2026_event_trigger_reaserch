"""Staged, fail-preserving runner for the Chapter 2 gain-gate study."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIVILIAN_DIR = Path(__file__).resolve().parent
STAGES = (
    "audit", "smoke", "validate", "collect", "train-light", "evaluate",
    "train-event", "ppo-transfer", "analyze",
)
RUN_SCHEMA = "gain-gate-run/1"
SOURCE_SCHEMA = "gain-gate-source-manifest/1"
EVIDENCE_SCHEMA = "gain-gate-evidence-manifest/1"
REVIEW_SCHEMA = "gain-gate-p1-review/1"
V2_SMOKE_REVIEW_SCHEMA = "gain-gate-v2-smoke-review/1"
V2_SMOKE_REVIEW_VERIFICATION_SCHEMA = "gain-gate-v2-smoke-review-verification/1"
V2_SMOKE_RUN_ID = "20261017T113110Z-8035318d2d79"
V2_SMOKE_CONFIG_SHA256 = "c49d94c3158f89faf34b97b33bd1bc58a6816fd8d6b7282ebe4c928b34e0f943"
V2_SMOKE_SOURCE_MANIFEST_SHA256 = "2657ebefb1f9eba1a99333baf83c601c4b88e8928311fca5e5d7454c3277db84"
V2_SMOKE_EVIDENCE_MANIFEST_SHA256 = "25b77385e6f11d46f11e56205051fc4b8649572dd0420b51636cb5906e51e1da"
V2_SMOKE_EVIDENCE_FILE_COUNT = 79
REPLANNER_VERSION = "Public-C1-Catalog-Replanner-v2"
COLLECTION_SCHEMA = "gain-gate-collection/1"
COLLECTION_INPUTS_SCHEMA = "gain-gate-collection-inputs/1"
TRAIN_COLLECTION_IDENTITY_SCHEMA = "gain-gate-train-input-identity/1"
TRAIN_STAGE_SCHEMA = "gain-gate-train-light-stage/1"
EVALUATION_SCHEMA = "gain-gate-test-evaluation/1"


def _shared_replanner_version() -> str:
    """Return the implementation version from the frozen replanner module."""
    try:
        from .fixed_replanner import REPLANNER_VERSION as version
    except ImportError:
        from fixed_replanner import REPLANNER_VERSION as version
    if not isinstance(version, str) or not version:
        raise ValueError("GAIN_RUN_SHARED_REPLANNER_VERSION_INVALID")
    return version


def _bootstrap_project_dependencies() -> None:
    for candidate in (
        PROJECT_ROOT / ".deps",
        PROJECT_ROOT / ".deps" / "Lib" / "site-packages",
        PROJECT_ROOT / ".deps" / "site-packages",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    if str(CIVILIAN_DIR) not in sys.path:
        sys.path.insert(0, str(CIVILIAN_DIR))


_bootstrap_project_dependencies()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
_THREADPOOL_LIMITER: Any = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _suffix_seed(scenario_id: str, state_id: str, timepoint: int, suffix_index: int) -> int:
    key = {"namespace": "counterfactual-suffix-v1", "scenario_id": scenario_id,
           "state_id": state_id, "time": int(timepoint), "suffix_index": int(suffix_index)}
    return int.from_bytes(hashlib.sha256(_canonical(key)).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _scenario_content_sha256(scenario: Mapping[str, Any]) -> str:
    supplied = scenario.get("scenario_content_sha256")
    if isinstance(supplied, str) and len(supplied) == 64:
        return supplied
    identity_fields = {
        "parent_id", "scenario_id", "scenario_instance_id", "split", "condition",
        "parent_family", "seed", "seed_variant", "scenario_seed", "scenario_instance_seed",
        "instance_seed", "generation_seed", "event_tape_id", "dgp_profile", "dgp_profile_hash",
    }
    content = {str(key): value for key, value in scenario.items() if str(key).lower() not in identity_fields}
    return _sha256_bytes(_canonical(content))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_suffix_gains(label: Mapping[str, Any], oracle_receipt: Mapping[str, Any]) -> list[float | None]:
    pairs = label.get("pairs")
    if (
        not isinstance(pairs, list) or len(pairs) != oracle_receipt.get("pair_count")
        or label.get("valid_mask") is not oracle_receipt.get("label_valid_mask")
    ):
        raise ValueError("GAIN_RUN_ORACLE_SUFFIX_PAIR_COUNT_OR_MASK_MISMATCH")
    values: list[float | None] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("GAIN_RUN_ORACLE_SUFFIX_PAIR_INVALID")
        if pair.get("valid_mask") is True:
            value = pair.get("gain")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("GAIN_RUN_ORACLE_SUFFIX_GAIN_INVALID")
            values.append(float(value))
        else:
            if pair.get("gain") is not None:
                raise ValueError("GAIN_RUN_ORACLE_INVALID_SUFFIX_HAS_GAIN")
            values.append(None)
    if oracle_receipt.get("label_valid_mask") is True:
        gains = [value for value in values if value is not None]
        mean = statistics.mean(gains)
        variance = statistics.variance(gains) if len(gains) > 1 else None
        expected_mean = oracle_receipt.get("recomputed_gain_mean")
        expected_variance = oracle_receipt.get("recomputed_paired_variance")
        if (
            len(gains) != len(values) or expected_mean is None
            or not math.isclose(mean, float(expected_mean), rel_tol=1e-10, abs_tol=1e-10)
            or (variance is None) != (expected_variance is None)
            or variance is not None and not math.isclose(
                variance, float(expected_variance), rel_tol=1e-10, abs_tol=1e-10,
            )
        ):
            raise ValueError("GAIN_RUN_ORACLE_SUFFIX_SUMMARY_MISMATCH")
    return values


def _validate_collection_label_suffix_binding(
    label: Mapping[str, Any], suffix_seeds: list[Any], *, forced: bool,
) -> int:
    """Validate label pair cardinality without confusing audit suffixes with pairs.

    Collection writes suffix snapshots for every candidate so the evidence is
    replayable.  A forced safety state is intentionally short-circuited by the
    counterfactual producer and therefore has no paired branches even though
    those audit suffix files still exist.  All labels that do contain pairs
    must bind them one-for-one to the row suffix seeds.
    """
    pairs = label.get("pairs")
    suffix_count = label.get("suffix_count")
    if not isinstance(pairs, list) or type(suffix_count) is not int or suffix_count < 0:
        raise ValueError("GAIN_RUN_COLLECTION_LABEL_SUFFIX_BINDING_INVALID")
    if suffix_count != len(pairs):
        raise ValueError("GAIN_RUN_COLLECTION_LABEL_SUFFIX_COUNT_MISMATCH")
    if suffix_count == 0:
        if not (
            forced is True
            and label.get("forced") is True
            and label.get("valid_mask") is False
            and label.get("unknown_reason") == "forced_safety_state"
        ):
            raise ValueError("GAIN_RUN_COLLECTION_LABEL_ZERO_PAIR_REASON_INVALID")
        return 0
    if suffix_count != len(suffix_seeds):
        raise ValueError("GAIN_RUN_COLLECTION_LABEL_SUFFIX_COUNT_MISMATCH")
    pair_seeds = [pair.get("suffix_seed") for pair in pairs]
    if pair_seeds != list(suffix_seeds):
        raise ValueError("GAIN_RUN_COLLECTION_LABEL_SUFFIX_SEEDS_MISMATCH")
    return suffix_count


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("GAIN_RUN_NONFINITE_JSON")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"GAIN_RUN_UNSERIALIZABLE:{type(value).__name__}")


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return payload.encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_serialized_json(value).decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _GzipJsonlWriter:
    """Exclusive, deterministic, line-buffered gzip JSONL output."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("xb")
        self._gzip = gzip.GzipFile(fileobj=self._raw, mode="wb", mtime=0)
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="\n")
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._text.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._text.flush()
        self.rows += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.close()
        return False

    def close(self) -> None:
        if self._text.closed:
            return
        self._text.flush()
        self._text.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()


def _write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _serialized_json(value)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
        raw.flush()
        os.fsync(raw.fileno())


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"GAIN_RUN_GZIP_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"GAIN_RUN_GZIP_JSONL_OBJECT_REQUIRED:{path.name}:{line_number}")
            rows.append(value)
    return rows


def _iter_gzip_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"GAIN_RUN_GZIP_JSONL_OBJECT_REQUIRED:{path.name}:{line_number}")
            yield value


def _set_cpu_affinity_one() -> list[int]:
    """Restrict this process to one currently allowed logical CPU."""
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise RuntimeError("GAIN_RUN_CPU_AFFINITY_EMPTY")
        os.sched_setaffinity(0, {allowed[0]})
        actual = sorted(os.sched_getaffinity(0))
    else:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("GAIN_RUN_CPU_AFFINITY_ENFORCEMENT_UNAVAILABLE") from exc
        process = psutil.Process()
        allowed = sorted(process.cpu_affinity())
        if not allowed:
            raise RuntimeError("GAIN_RUN_CPU_AFFINITY_EMPTY")
        process.cpu_affinity([allowed[0]])
        actual = sorted(process.cpu_affinity())
    if len(actual) != 1:
        raise RuntimeError(f"GAIN_RUN_CPU_AFFINITY_NOT_SINGLE_CORE:{actual}")
    return actual


def _set_torch_threads_one(torch_module: Any) -> dict[str, int]:
    torch_module.set_num_threads(1)
    interop_before = int(torch_module.get_num_interop_threads())
    if interop_before != 1:
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("GAIN_RUN_TORCH_INTEROP_THREADS_NOT_ENFORCEABLE") from exc
    actual = {"intraop": int(torch_module.get_num_threads()), "interop": int(torch_module.get_num_interop_threads())}
    if actual != {"intraop": 1, "interop": 1}:
        raise RuntimeError(f"GAIN_RUN_TORCH_THREADS_NOT_SINGLE:{actual}")
    return actual


def _threadpool_receipt(*, enforce: bool) -> dict[str, Any]:
    global _THREADPOOL_LIMITER
    try:
        import threadpoolctl
    except ImportError as exc:
        if enforce:
            raise RuntimeError("GAIN_RUN_THREADPOOLCTL_REQUIRED_FOR_CPU_LIMIT") from exc
        return {"available": False, "reason": "threadpoolctl_unavailable", "pools": []}
    if enforce:
        _THREADPOOL_LIMITER = threadpoolctl.threadpool_limits(limits=1)
    pools = [
        {
            "user_api": pool.get("user_api"), "internal_api": pool.get("internal_api"),
            "prefix": pool.get("prefix"), "num_threads": int(pool["num_threads"]),
        }
        for pool in threadpoolctl.threadpool_info()
    ]
    if enforce and any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"GAIN_RUN_BLAS_THREADPOOL_LIMIT_FAILED:{pools}")
    return {
        "available": True, "limit_enforced": bool(enforce),
        "observed_pool_count": len(pools), "pools": pools,
    }


def _runtime_receipt(stage: str, *, enforce: bool) -> dict[str, Any]:
    affinity = _set_cpu_affinity_one() if enforce else _current_cpu_affinity()
    package_versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    torch_runtime: dict[str, Any] = {"available": False}
    if package_versions["torch"] is not None:
        import torch
        threads = _set_torch_threads_one(torch) if enforce else {
            "intraop": int(torch.get_num_threads()), "interop": int(torch.get_num_interop_threads()),
        }
        torch_runtime = {
            "available": True, "version": torch.__version__, "threads": threads,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_is_initialized": bool(torch.cuda.is_initialized()),
        }
        if enforce and (torch_runtime["cuda_available"] or torch_runtime["cuda_is_initialized"]):
            raise RuntimeError("GAIN_RUN_CUDA_MUST_BE_DISABLED")
    threadpools = _threadpool_receipt(enforce=enforce)
    cpu_model = platform.processor().strip()
    if not cpu_model and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    resource_module = __import__("resource_supervisor")
    return {
        "schema": "gain-runtime-receipt/1", "stage": stage,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cpu_model": cpu_model or None, "cpu_affinity": affinity,
        "cpu_affinity_enforced": bool(enforce), "package_versions": package_versions,
        "torch": torch_runtime,
        "blas_threadpools": threadpools,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_rss_bytes": int(resource_module._read_rss_bytes()),
    }


def _current_cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    try:
        import psutil
    except ImportError:
        return None
    return sorted(psutil.Process().cpu_affinity())


class _ProjectResourceAccount:
    """Serialize project stages and persist CPU, wall, and RSS settlement."""

    schema = "gain-project-resource-entry/1"

    def __init__(self, output: Path, run_id: str, stage: str, config: Mapping[str, Any]):
        self.output = output
        self.run_id = run_id
        self.stage = stage
        self.config = config
        self.ledger_path = PROJECT_ROOT / "outputs" / "PROJECT_CPU_LEDGER.jsonl"
        self.lock_path = PROJECT_ROOT / "outputs" / ".PROJECT_CPU_LEDGER.lock"
        self.lock_handle = None
        self.locked = False
        self.prior_cpu_seconds = 0.0
        self.v2_cpu_seconds_before = 0.0
        collection = config["collection"]
        self.max_v2_cpu_seconds = float(collection.get("max_v2_stage_cpu_seconds", collection["max_project_cpu_seconds"]))
        self.prior_review_path = PROJECT_ROOT / str(collection.get(
            "prior_resource_review_path", "reviews/V2_PRIOR_RESOURCE_REVIEW.json"
        ))
        self.prior_review_sha256 = str(collection.get("prior_resource_review_sha256", ""))
        self.prior_manifest_cpu_seconds = float(collection.get("prior_resource_cpu_seconds", 0.0))
        self.child_cpu_seconds = 0.0
        self.start_cpu = time.process_time()
        self.start_wall = time.perf_counter()
        self.peak_rss_bytes = int(__import__("resource_supervisor")._read_rss_bytes())
        self.max_wall_seconds = float(config["collection"]["max_wall_seconds_per_server_job"])
        self.max_cpu_seconds = float(collection["max_project_cpu_seconds"])
        self.max_rss_bytes = int(config["collection"]["max_rss_bytes"])
        self.runtime: dict[str, Any] | None = None
        self.finished_row: dict[str, Any] | None = None

    def __enter__(self):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.lock_path.open("a+b")
        self.lock_handle.seek(0, os.SEEK_END)
        if self.lock_handle.tell() == 0:
            self.lock_handle.write(b"0")
            self.lock_handle.flush()
        self.lock_handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as exc:
            self.lock_handle.close()
            self.lock_handle = None
            raise RuntimeError("GAIN_RUN_PROJECT_RESOURCE_LEDGER_BUSY") from exc
        self.locked = True
        entries = self._read_entries()
        self.v2_cpu_seconds_before = sum(float(row["cpu_seconds"]) for row in entries)
        if self.prior_review_path.is_file():
            if self.prior_review_sha256 and _sha256_file(self.prior_review_path) != self.prior_review_sha256:
                raise ValueError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_HASH_MISMATCH")
            review = _read_json(self.prior_review_path)
            if review.get("schema") != "gain-v2-prior-resource-review/1":
                raise ValueError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_SCHEMA_INVALID")
            if not math.isclose(float(review.get("cpu_seconds", -1.0)), self.prior_manifest_cpu_seconds, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_CPU_MISMATCH")
        elif self.prior_manifest_cpu_seconds > 0.0:
            raise FileNotFoundError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_MISSING")
        self.prior_cpu_seconds = self.prior_manifest_cpu_seconds + self.v2_cpu_seconds_before
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        if self.locked and self.lock_handle is not None:
            try:
                self.lock_handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_handle.close()
                self.locked = False
        return False

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.ledger_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.ledger_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"GAIN_RUN_PROJECT_RESOURCE_LEDGER_JSON:{line_number}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"GAIN_RUN_PROJECT_RESOURCE_LEDGER_ENTRY_INVALID:{line_number}")
                cpu = row.get("cpu_seconds")
                if (
                    row.get("schema") != self.schema or row.get("status") not in {"complete", "failed"}
                    or not isinstance(row.get("run_id"), str) or not isinstance(cpu, (int, float))
                    or isinstance(cpu, bool) or not math.isfinite(float(cpu)) or float(cpu) < 0.0
                ):
                    raise ValueError(f"GAIN_RUN_PROJECT_RESOURCE_LEDGER_ENTRY_INVALID:{line_number}")
                rows.append(dict(row))
        return rows

    def add_child_cpu_seconds(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("GAIN_RUN_CHILD_CPU_SECONDS_INVALID")
        self.child_cpu_seconds += float(value)
        self.check()

    def set_runtime(self, runtime: Mapping[str, Any]) -> None:
        self.runtime = copy.deepcopy(dict(runtime))

    def check(self) -> dict[str, Any]:
        metrics = self._metrics()
        if metrics["stage_wall_seconds"] > self.max_wall_seconds:
            raise TimeoutError("GAIN_RUN_STAGE_WALL_LIMIT_EXCEEDED")
        if metrics["project_cpu_seconds_including_stage"] > self.max_cpu_seconds:
            raise TimeoutError("GAIN_RUN_PROJECT_CPU_LIMIT_EXCEEDED")
        if metrics["v2_cpu_seconds_including_stage"] > self.max_v2_cpu_seconds:
            raise TimeoutError("GAIN_RUN_V2_STAGE_CPU_LIMIT_EXCEEDED")
        if metrics["peak_rss_bytes"] > self.max_rss_bytes:
            raise MemoryError("GAIN_RUN_STAGE_RSS_LIMIT_EXCEEDED")
        return metrics

    def _metrics(self) -> dict[str, Any]:
        elapsed_wall = max(0.0, time.perf_counter() - self.start_wall)
        process_cpu = max(0.0, time.process_time() - self.start_cpu)
        total_cpu = self.prior_cpu_seconds + process_cpu + self.child_cpu_seconds
        v2_total_cpu = self.v2_cpu_seconds_before + process_cpu + self.child_cpu_seconds
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(__import__("resource_supervisor")._read_rss_bytes()))
        return {
            "stage_wall_seconds": elapsed_wall, "stage_process_cpu_seconds": process_cpu,
            "verified_child_cpu_seconds": self.child_cpu_seconds,
            "project_cpu_seconds_including_stage": total_cpu,
            "v2_cpu_seconds_including_stage": v2_total_cpu,
            "peak_rss_bytes": self.peak_rss_bytes,
        }

    def finish(self, *, status: str, error: BaseException | str | None = None) -> dict[str, Any]:
        if not self.locked:
            raise RuntimeError("GAIN_RUN_PROJECT_RESOURCE_ACCOUNT_NOT_LOCKED")
        if status not in {"complete", "failed"}:
            raise ValueError("GAIN_RUN_PROJECT_RESOURCE_STATUS_INVALID")
        metrics = self._metrics()
        violations = []
        if metrics["stage_wall_seconds"] > self.max_wall_seconds:
            violations.append("stage_wall_limit")
        if metrics["project_cpu_seconds_including_stage"] > self.max_cpu_seconds:
            violations.append("project_cpu_limit")
        if metrics["v2_cpu_seconds_including_stage"] > self.max_v2_cpu_seconds:
            violations.append("v2_stage_cpu_limit")
        if metrics["peak_rss_bytes"] > self.max_rss_bytes:
            violations.append("rss_limit")
        if violations:
            status = "failed"
            if error is None:
                error = "resource limit exceeded: " + ",".join(violations)
        receipt = {
            "schema": "gain-stage-resource-receipt/1", "run_id": self.run_id,
            "stage": self.stage, "status": status,
            **metrics, "project_cpu_seconds_before": self.prior_cpu_seconds,
            "project_cpu_budget_seconds": self.max_cpu_seconds,
            "prior_manifest_cpu_seconds": self.prior_manifest_cpu_seconds,
            "prior_resource_review_path": str(self.prior_review_path.relative_to(PROJECT_ROOT)),
            "prior_resource_review_sha256": self.prior_review_sha256,
            "v2_cpu_seconds_before": self.v2_cpu_seconds_before,
            "v2_stage_cpu_budget_seconds": self.max_v2_cpu_seconds,
            "stage_wall_budget_seconds": self.max_wall_seconds,
            "rss_budget_bytes": self.max_rss_bytes,
            "resource_contract": {
                "cpu_affinity_enforced": bool((self.runtime or {}).get("cpu_affinity_enforced", False)),
                "cpu_affinity": (self.runtime or {}).get("cpu_affinity"),
                "torch": copy.deepcopy((self.runtime or {}).get("torch")),
                "package_versions": copy.deepcopy((self.runtime or {}).get("package_versions", {})),
                "cpu_model": (self.runtime or {}).get("cpu_model"),
                "cuda_visible_devices": (self.runtime or {}).get("cuda_visible_devices"),
            },
            "runtime_receipt": copy.deepcopy(self.runtime),
            "error": None if error is None else {"type": type(error).__name__ if isinstance(error, BaseException) else "recorded_error", "message": str(error)},
            "ledger_path": str(self.ledger_path.relative_to(PROJECT_ROOT)),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt_path = self.output / "RESOURCE_RECEIPT.json"
        _write_json(receipt_path, receipt)
        try:
            receipt_relative_path = receipt_path.relative_to(PROJECT_ROOT)
        except ValueError:
            receipt_relative_path = Path(receipt_path.name)
        row = {
            "schema": self.schema, "run_id": self.run_id, "stage": self.stage,
            "status": status, "cpu_seconds": metrics["stage_process_cpu_seconds"] + self.child_cpu_seconds,
            "wall_seconds": metrics["stage_wall_seconds"], "peak_rss_bytes": metrics["peak_rss_bytes"],
            "receipt_path": receipt_relative_path.as_posix(),
            "receipt_sha256": _sha256_file(receipt_path),
        }
        entries = self._read_entries()
        if entries and entries[-1].get("run_id") == self.run_id:
            entries[-1] = row
            temporary = self.ledger_path.with_name(f".PROJECT_CPU_LEDGER.{uuid.uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                for entry in entries:
                    stream.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ledger_path)
        else:
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            entries.append(row)
        self.finished_row = row
        return receipt


def _check_resource_account(args: argparse.Namespace) -> None:
    account = getattr(args, "resource_account", None)
    if account is not None:
        account.check()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"GAIN_RUN_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _resolve_input(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        project_relative = PROJECT_ROOT / path
        path = project_relative if project_relative.exists() else Path.cwd() / path
    return path.resolve(strict=True)


def _load_config(path_text: str) -> tuple[Path, dict[str, Any], str]:
    path = _resolve_input(path_text)
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "gain-gate-config/1":
        raise ValueError("GAIN_RUN_CONFIG_SCHEMA_INVALID")
    _validate_config(config)
    return path, config, _sha256_bytes(raw)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "gain-gate-v2-100x100":
        raise ValueError("GAIN_RUN_V2_CONFIG_IDENTITY_INVALID")
    map_config = config.get("map", {})
    if (map_config.get("rows"), map_config.get("cols"), map_config.get("grid_cells")) != (100, 100, 10000):
        raise ValueError("GAIN_RUN_MAP_CONTRACT_INVALID")
    scenario = config.get("scenario", {})
    if (scenario.get("uav"), scenario.get("helicopter"), scenario.get("tasks"), scenario.get("horizon_minutes")) != (4, 2, 12, 60):
        raise ValueError("GAIN_RUN_PILOT_SHAPE_INVALID")
    replanner = config.get("replanner", {})
    if (
        replanner.get("version") != REPLANNER_VERSION
        or _shared_replanner_version() != REPLANNER_VERSION
        or replanner.get("max_candidates") != 16
    ):
        raise ValueError("GAIN_RUN_REPLANNER_CONTRACT_INVALID")
    if replanner.get("rollout_oracle_allowed") is not False:
        raise ValueError("GAIN_RUN_ROLLOUT_ORACLE_MUST_BE_DISABLED")
    labels = config.get("labels", {})
    if labels.get("horizon_minutes") != 30 or labels.get("matched_future_suffixes") != 2:
        raise ValueError("GAIN_RUN_LABEL_CONTRACT_INVALID")
    if labels.get("continuation") != "keep_resulting_queues_with_shared_safety_emergency_rule":
        raise ValueError("GAIN_RUN_CONTINUATION_CONTRACT_INVALID")
    if config.get("collection", {}).get("device") != "cpu" or config.get("collection", {}).get("threads") != 1:
        raise ValueError("GAIN_RUN_RUNTIME_MUST_BE_CPU1")
    collection = config.get("collection", {})
    if float(collection.get("max_project_cpu_seconds", 0.0)) != 86400.0:
        raise ValueError("GAIN_RUN_PROJECT_CPU_BUDGET_INVALID")
    if float(collection.get("max_v2_stage_cpu_seconds", 0.0)) > 78774.140625:
        raise ValueError("GAIN_RUN_V2_CPU_BUDGET_EXCEEDED")
    if collection.get("prior_resource_review_path") != "reviews/V2_PRIOR_RESOURCE_REVIEW.json":
        raise ValueError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_PATH_INVALID")
    if collection.get("prior_resource_review_sha256") != "c03fa60f6e1b8363fd81632d02d4dfdacda06d46babfe7633ff21b4245d663cc":
        raise ValueError("GAIN_RUN_PRIOR_RESOURCE_REVIEW_HASH_INVALID")
    if float(collection.get("prior_resource_cpu_seconds", -1.0)) != 7625.859375:
        raise ValueError("GAIN_RUN_PRIOR_RESOURCE_CPU_INVALID")


def _source_file_paths() -> list[Path]:
    candidates = [
        *CIVILIAN_DIR.glob("*.py"),
        *(PROJECT_ROOT / "package" / "tests").glob("*.py"),
        *(PROJECT_ROOT / "configs").glob("*.json"),
        *(PROJECT_ROOT / "protocols").glob("*.md"),
    ]
    candidates.extend(
        path for path in PROJECT_ROOT.glob("*.md")
        if path.name in {
            "BASELINE_PROVENANCE.json", "MANIFEST.md", "REUSE_INVENTORY.json",
            "REPO_AUDIT.md", "EXPERIMENT_PLAN.md", "EXPERIMENT_PROTOCOL.md", "STATUS.md",
        }
    )
    candidates.extend(
        path for path in (PROJECT_ROOT / "BASELINE_PROVENANCE.json", PROJECT_ROOT / "REUSE_INVENTORY.json")
        if path.is_file()
    )
    static_roots = (CIVILIAN_DIR / "fixtures", CIVILIAN_DIR / "registered_data")
    for root in static_roots:
        if root.is_dir():
            candidates.extend(root.rglob("*.json"))
    paths = sorted({path.resolve() for path in candidates if path.is_file()})
    return paths


def _source_manifest() -> dict[str, Any]:
    files = []
    for path in _source_file_paths():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    if not files:
        raise ValueError("GAIN_RUN_SOURCE_MANIFEST_EMPTY")
    static_inputs = [
        row for row in files
        if row["path"].startswith("package/civilian/fixtures/")
        or row["path"].startswith("package/civilian/registered_data/")
    ]
    return {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "static_inputs": static_inputs,
        "static_inputs_sha256": _sha256_bytes(_canonical(static_inputs)),
    }


def _create_run_dir(output_text: str | None) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:12]}"
    if output_text:
        output = Path(output_text)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output = output.resolve()
    else:
        output = PROJECT_ROOT / "outputs" / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    return output, run_id


def _write_initial_manifests(
    output: Path, run_id: str, stage: str, config_path: Path,
    config: dict[str, Any], config_sha256: str,
) -> dict[str, Any]:
    source = _source_manifest()
    _write_json(output / "SOURCE_MANIFEST.json", source)
    source_sha256 = _sha256_file(output / "SOURCE_MANIFEST.json")
    try:
        config_rel = config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        config_rel = config_path.name
    run = {
        "schema": RUN_SCHEMA, "run_id": run_id, "stage": stage,
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_rel, "config_sha256": config_sha256,
        "source_manifest_path": "SOURCE_MANIFEST.json",
        "source_manifest_sha256": source_sha256,
        "project_git_metadata_available": (PROJECT_ROOT / ".git").exists(),
        "python_version": platform.python_version(),
        "platform": platform.platform(), "device": "cpu",
        "thread_settings_status": "pending_runtime_measurement",
    }
    _write_json(output / "RUN.json", run)
    _write_json(output / "RESOLVED_CONFIG.json", config)
    return run


def _write_evidence_manifest(output: Path, run: dict[str, Any], status: str) -> dict[str, Any]:
    run["status"] = status
    _write_json(output / "RUN.json", run)
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json" or path.name.endswith(".tmp"):
            continue
        files.append({
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    payload = {
        "schema": EVIDENCE_SCHEMA, "run_id": run["run_id"], "stage": run["stage"],
        "status": status, "config_sha256": run["config_sha256"],
        "source_manifest_sha256": run["source_manifest_sha256"], "files": files,
    }
    _write_json(output / "EVIDENCE_MANIFEST.json", payload)
    return {
        "path": "EVIDENCE_MANIFEST.json",
        "sha256": _sha256_file(output / "EVIDENCE_MANIFEST.json"),
        "file_count": len(files),
    }


def _audit_checks(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = [row["path"] for row in source["files"]]
    forbidden = [
        path for path in source_paths
        if any(part in {".deps", "reviews", "evidence", "__pycache__"} for part in Path(path).parts)
        or path.endswith((".pyc", ".pyo"))
    ]
    required = {
        "package/civilian/gain_scenarios.py", "package/civilian/counterfactual_pairs.py",
        "package/civilian/cost_accounting.py", "package/civilian/gain_features.py",
        "package/civilian/gain_gate.py", "package/civilian/gain_models.py",
        "package/civilian/gain_analysis.py", "package/civilian/gain_oracle.py",
        "package/civilian/gain_variance.py",
        "package/tests/test_gain_models_analysis.py",
    }
    missing = sorted(required - set(source_paths))
    if forbidden or missing:
        raise ValueError(f"GAIN_RUN_SOURCE_AUDIT_FAILED:forbidden={forbidden}:missing={missing}")
    return {
        "config_schema": config["schema"], "source_file_count": len(source_paths),
        "private_or_generated_inputs_excluded": True,
        "git_metadata_available": (PROJECT_ROOT / ".git").exists(),
        "required_modules_present": True,
        "resource_contract": {
            "device": "cpu", "cpu_affinity_count": 1, "torch_threads": 1,
            "blas_threadpools": "runtime_receipt_required", "cuda_visible_devices": "",
        },
    }


def _load_smoke_modules() -> dict[str, Any]:
    import counterfactual_pairs
    import event_scenarios
    import gain_features
    import gain_gate
    import gain_oracle
    import gain_scenarios
    from event_environment import EventReliefEnv
    from event_trigger import TriggerEngine, identify_affected_scope
    try:
        import fixed_replanner
        import gain_safety
    except ImportError as exc:
        raise RuntimeError(f"GAIN_RUN_P1_DEPENDENCY_MISSING:{exc}") from exc
    return {
        "pairs": counterfactual_pairs, "scenarios": gain_scenarios,
        "features": gain_features, "gate": gain_gate, "oracle": gain_oracle,
        "event_scenarios": event_scenarios, "env_class": EventReliefEnv,
        "trigger_class": TriggerEngine, "identify_scope": identify_affected_scope,
        "replanner": fixed_replanner, "safety": gain_safety,
    }


def load_evaluation_modules() -> dict[str, Any]:
    """Expose the frozen environment components to closed-loop evaluation."""
    return _load_smoke_modules()


def _legacy_environment_scenario(gain_scenario: Mapping[str, Any]) -> dict[str, Any]:
    split = str(gain_scenario["split"])
    legacy_split = "train" if split in {"pilot", "train"} else "development"
    return {
        "parent_id": str(gain_scenario["parent_id"]),
        "split": legacy_split,
        "condition": str(gain_scenario["condition"]),
        "seed_variant": int(gain_scenario["seed_variant"]),
        "map": copy.deepcopy(gain_scenario["map"]),
        "base_scenario": copy.deepcopy(gain_scenario["base_scenario"]),
        "queue_plan": copy.deepcopy(gain_scenario["queue_plan"]),
        "event_tape": copy.deepcopy(gain_scenario["event_tape"]),
        "event_tape_id": str(gain_scenario["event_tape_id"]),
    }


def legacy_environment_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    return _legacy_environment_scenario(scenario)


def _noop_step(env: Any, pairs: Any) -> dict[str, Any]:
    before = len(getattr(env, "_steps", []))
    _, _, terminated, truncated, info = env.step(copy.deepcopy(pairs.NOOP_ACTION))
    if len(getattr(env, "_steps", [])) != before + 1:
        raise ValueError("GAIN_RUN_SMOKE_NOOP_STEP_NOT_PERSISTED")
    if not isinstance(info, Mapping):
        raise ValueError("GAIN_RUN_SMOKE_STEP_INFO_INVALID")
    return {"terminated": bool(terminated), "truncated": bool(truncated), "info": copy.deepcopy(dict(info))}


def _advance_to_candidate(scenario: Mapping[str, Any], modules: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    pairs = modules["pairs"]
    env = modules["env_class"](_legacy_environment_scenario(scenario))
    observation = env.reset()
    event_times = [int(row["time"]) for row in scenario["event_tape"]]
    target_time = min(event_times, default=0)
    while env.time < target_time:
        outcome = _noop_step(env, pairs)
        if outcome["terminated"] or outcome["truncated"]:
            break
    current = pairs.model_public_observation(env)
    history = pairs._public_history(env)
    previous = history[-2] if len(history) >= 2 else None
    engine = modules["trigger_class"]({}, {"threshold": 0.5, "delta_min": 4})
    detection = engine.detect(previous, current, history=history[-4:])
    scope = modules["identify_scope"](current, detection)
    return env, detection, scope


def _safety_forced(safety_report: Mapping[str, Any]) -> bool:
    return bool(safety_report.get("needs_forced_handling", safety_report.get("forced", False)))


def _safety_scope(safety_report: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = copy.deepcopy(dict(fallback or {}))
    violations = safety_report.get("violations", [])
    if not isinstance(violations, list):
        violations = []
    task_ids = {str(row["task_id"]) for row in violations if isinstance(row, Mapping) and row.get("task_id") is not None}
    platform_ids = set(map(str, safety_report.get("affected_platform_ids", [])))
    current["affected_task_ids"] = sorted(set(map(str, current.get("affected_task_ids", []))) | task_ids)
    current["affected_platform_ids"] = sorted(set(map(str, current.get("affected_platform_ids", []))) | platform_ids)
    required_scope = str(safety_report.get("required_scope", current.get("scope", "LOCAL_REPAIR")))
    current["scope"] = required_scope
    current["local_eligible"] = required_scope == "LOCAL_REPAIR"
    current["global_eligible"] = required_scope == "GLOBAL_REPLAN"
    current["reasons"] = sorted(set(map(str, current.get("reasons", []))) | {
        str(row.get("code")) for row in violations if isinstance(row, Mapping) and row.get("code") is not None
    })
    return current


def safety_scope(safety_report: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _safety_scope(safety_report, fallback)


def _safety_receipt(modules: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    result = modules["safety"].assess_queue_safety(copy.deepcopy(dict(observation)))
    if not isinstance(result, Mapping):
        raise ValueError("GAIN_RUN_SAFETY_RECEIPT_INVALID")
    if type(result.get("safe_to_suppress")) is not bool:
        raise ValueError("GAIN_RUN_SAFETY_RECEIPT_MISSING_SAFE_TO_SUPPRESS")
    return copy.deepcopy(dict(result))


def safety_receipt(modules: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    return _safety_receipt(modules, observation)


def _verify_safety_disposition(
    modules: Mapping[str, Any], env: Any, observation: Mapping[str, Any],
    action: Mapping[str, Any], safety: Mapping[str, Any],
    *, apply_receipt: Mapping[str, Any] | None = None,
    post_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verifier = getattr(modules["safety"], "verify_live_safety_disposition", None)
    if apply_receipt is not None and post_observation is not None and callable(verifier):
        result = verifier(
            env, copy.deepcopy(dict(observation)), copy.deepcopy(dict(safety)),
            copy.deepcopy(dict(apply_receipt)), copy.deepcopy(dict(post_observation)),
        )
    else:
        result = modules["safety"].verify_safety_disposition(
            env, copy.deepcopy(dict(observation)), copy.deepcopy(dict(action)),
            initial_report=copy.deepcopy(dict(safety)),
        )
    if (
        not isinstance(result, Mapping)
        or result.get("schema") != "gain-safety-disposition/1"
        or type(result.get("resolved")) is not bool
    ):
        raise ValueError("GAIN_RUN_SAFETY_DISPOSITION_INVALID")
    return copy.deepcopy(dict(result))


def _apply_live_safety_disposition(
    modules: Mapping[str, Any], env: Any, observation: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply and independently verify a same-tick safety transition.

    The hook is deliberately completed before any candidate generation.  The
    returned observation is the post-mutation public state that downstream
    scope selection and replanning must use.
    """
    apply_fn = getattr(modules["safety"], "apply_safety_disposition", None)
    if not callable(apply_fn):
        raise ValueError("GAIN_RUN_LIVE_SAFETY_HOOK_UNAVAILABLE")
    pre = copy.deepcopy(dict(observation))
    receipt = apply_fn(env, pre, copy.deepcopy(dict(safety)))
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "gain-live-safety-disposition/1":
        raise ValueError("GAIN_RUN_LIVE_SAFETY_RECEIPT_INVALID")
    post = modules["pairs"].model_public_observation(env)
    # Keep the hook receipt in the trace while returning the refreshed public
    # state separately for feature extraction and planner scope selection.
    disposition = _verify_safety_disposition(
        modules, env, pre, {}, safety,
        apply_receipt=receipt, post_observation=post,
    )
    if disposition.get("schema") != "gain-safety-disposition/1":
        raise ValueError("GAIN_RUN_SAFETY_DISPOSITION_INVALID")
    return copy.deepcopy(post), copy.deepcopy(dict(disposition)), copy.deepcopy(dict(receipt))


def verify_safety_disposition(
    modules: Mapping[str, Any], env: Any, observation: Mapping[str, Any],
    action: Mapping[str, Any], safety: Mapping[str, Any],
) -> dict[str, Any]:
    return _verify_safety_disposition(modules, env, observation, action, safety)


def apply_live_safety_disposition(
    modules: Mapping[str, Any], env: Any, observation: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _apply_live_safety_disposition(modules, env, observation, safety)


def _invoke_replanner(
    modules: Mapping[str, Any], env: Any, scope: Mapping[str, Any] | None,
    *, max_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    before_cpu, before_wall = time.process_time(), time.perf_counter()
    receipt = modules["replanner"].replan(env, affected_scope=copy.deepcopy(scope), max_candidates=max_candidates)
    cpu_seconds = max(0.0, time.process_time() - before_cpu)
    wall_seconds = max(0.0, time.perf_counter() - before_wall)
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "gain-fixed-replanner-receipt/1":
        raise ValueError("GAIN_RUN_REPLANNER_RECEIPT_SCHEMA")
    if receipt.get("replanner_version") != REPLANNER_VERSION:
        raise ValueError("GAIN_RUN_REPLANNER_VERSION_MISMATCH")
    candidates = receipt.get("candidates")
    count = receipt.get("actual_candidate_count")
    if (
        not isinstance(candidates, list) or type(count) is not int
        or not 1 <= count <= max_candidates or len(candidates) > count
    ):
        raise ValueError("GAIN_RUN_REPLANNER_CANDIDATE_LIMIT_OR_RECEIPT")
    action = receipt.get("selected_action")
    if not isinstance(action, Mapping):
        raise ValueError("GAIN_RUN_REPLANNER_ACTION_MISSING")
    normalized, issues = env._check_action(copy.deepcopy(dict(action)))
    if issues or not isinstance(normalized, dict) or normalized != dict(action):
        raise ValueError(f"GAIN_RUN_REPLANNER_SELECTED_ACTION_INVALID:{sorted(set(map(str, issues)))}")
    record = copy.deepcopy(dict(receipt))
    record.update({"measured_cpu_seconds": cpu_seconds, "measured_wall_seconds": wall_seconds})
    return copy.deepcopy(dict(action)), record, cpu_seconds, wall_seconds


def invoke_replanner(
    modules: Mapping[str, Any], env: Any, scope: Mapping[str, Any] | None,
    *, max_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    return _invoke_replanner(modules, env, scope, max_candidates=max_candidates)


def _make_safety_continuation(modules: Mapping[str, Any], config: Mapping[str, Any]) -> Callable[[Any], Mapping[str, Any]]:
    pairs = modules["pairs"]
    engines: dict[int, Any] = {}

    def continuation(env: Any) -> Mapping[str, Any]:
        started = time.process_time()
        observation = pairs.model_public_observation(env)
        history = pairs._public_history(env)
        previous = history[-2] if len(history) >= 2 else None
        engine = engines.get(id(env))
        if engine is None:
            engine = modules["trigger_class"]({}, {"threshold": 0.5, "delta_min": 4})
            engines[id(env)] = engine
        detection = engine.detect(previous, observation, history=history[-4:])
        decision = engine.decide(detection, int(observation.get("time", env.time)), mode="event")
        safety = _safety_receipt(modules, observation)
        forced = bool(detection.get("strong_event")) or _safety_forced(safety) or not safety["safe_to_suppress"]
        observation, disposition, safety_apply_receipt = _apply_live_safety_disposition(
            modules, env, observation, safety,
        )
        # Safety mutation is same-tick.  Refresh history/scope before deciding
        # whether the fixed replanner is allowed to generate candidates.
        history = pairs._public_history(env)
        refreshed_scope = modules["identify_scope"](observation, detection)
        if not forced:
            elapsed = max(0.0, time.process_time() - started)
            return {
                "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
                "action": copy.deepcopy(pairs.NOOP_ACTION), "planner_invoked": False,
                "planner_cpu_seconds": 0.0, "safety_monitor_cpu_seconds": elapsed,
                "policy_version": "shared-safety-trigger-v1",
                "safety_report": safety,
                "safety_disposition": disposition,
                "safety_apply_receipt": safety_apply_receipt,
                "unknown_reason": None,
                "detection": detection, "public_history": history,
            }
        scope = _safety_scope(safety, refreshed_scope or decision.get("affected_scope"))
        action, receipt, planner_cpu, _ = _invoke_replanner(
            modules, env, scope, max_candidates=int(config["replanner"]["max_candidates"]),
        )
        elapsed = max(0.0, time.process_time() - started)
        return {
            "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
            "action": action, "planner_invoked": True,
            "planner_cpu_seconds": min(planner_cpu, elapsed),
            "safety_monitor_cpu_seconds": max(0.0, elapsed - planner_cpu),
            "policy_version": str(receipt.get("replanner_version", REPLANNER_VERSION)),
            "safety_report": safety, "safety_disposition": disposition,
            "safety_apply_receipt": safety_apply_receipt,
            "unknown_reason": disposition.get("unknown_reason"), "detection": detection,
            "public_history": history,
        }

    return continuation


def make_safety_continuation(modules: Mapping[str, Any], config: Mapping[str, Any]) -> Callable[[Any], Mapping[str, Any]]:
    return _make_safety_continuation(modules, config)


def _physical_trace_projection(trace: Mapping[str, Any]) -> dict[str, Any]:
    step_keys = (
        "time", "full_environment_action", "raw_events", "reward_primitives",
        "reward_terms", "operational_dispatches", "operation_receipts",
        "communication_receipts", "task_owner_change_count", "task_switch_count",
    )
    steps = [
        {key: copy.deepcopy(step[key]) for key in step_keys if key in step}
        for step in trace.get("steps", [])
    ]
    return {
        "arm": trace.get("arm"), "start_time": trace.get("start_time"),
        "end_time": trace.get("end_time"), "planned_action": trace.get("planned_action"),
        "steps": steps, "start_truth": trace.get("start_truth"), "raw_truth": trace.get("raw_truth"),
    }


def _advance_clone_to_time(scenario: Mapping[str, Any], timepoint: int, modules: Mapping[str, Any]) -> Any:
    env = modules["env_class"](_legacy_environment_scenario(scenario))
    env.reset()
    while env.time < timepoint:
        outcome = _noop_step(env, modules["pairs"])
        if outcome["terminated"] or outcome["truncated"]:
            break
    if env.time != timepoint:
        raise ValueError("GAIN_RUN_HIDDEN_INVARIANCE_CLONE_TIME_MISMATCH")
    return env


def _hidden_mutation(scenario: Mapping[str, Any], current_time: int) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(scenario))
    arrived = {
        str(row["task_id"]) for row in scenario["event_tape"]
        if row.get("kind") == "task_arrival" and int(row["time"]) <= current_time
    }
    task = next((row for row in mutated["base_scenario"]["tasks"]
                 if not row.get("known_initial") and str(row["id"]) not in arrived), None)
    if task is None:
        raise ValueError("GAIN_RUN_HIDDEN_INVARIANCE_NO_UNARRIVED_TASK")
    x, y = task["position"]
    if task.get("local"):
        task["position"] = [round(40.1 + (float(x) - 40.0) % 1.8, 4), round(40.1 + (float(y) - 40.0) % 1.8, 4)]
    else:
        task["position"] = [12.3 if float(x) > 20 else 83.7, 86.4 if float(y) < 80 else 13.2]
    return mutated


def _probe_online_gate(
    modules: Mapping[str, Any], env: Any, detection: Mapping[str, Any], scope: Mapping[str, Any],
    safety: Mapping[str, Any], predictor: Callable[[Mapping[str, float]], Mapping[str, Any]],
    replanner: Callable[[], Any], *, gate_mode: str = "calibrated_lower_bound",
) -> dict[str, Any]:
    observation = modules["pairs"].model_public_observation(env)
    started = time.perf_counter()
    features = modules["features"].extract_public_features(observation, dict(detection), dict(scope))
    forced = bool(detection.get("strong_event")) or _safety_forced(safety) or not bool(safety.get("safe_to_suppress"))
    if forced:
        prediction = None
        decision = modules["gate"].decide_gain(gate_mode=gate_mode, forced=True, safe_to_suppress=False)
    else:
        prediction = dict(predictor(features))
        decision = modules["gate"].decide_gain(
            prediction.get("gain_hat"), prediction.get("epsilon", 0.0),
            prediction.get("lower_bound"), gate_mode=gate_mode,
            forced=False, safe_to_suppress=bool(safety.get("safe_to_suppress")),
        )
    feature_gate_seconds = max(0.0, time.perf_counter() - started)
    plan_receipt = None
    planner_invoked = False
    if decision["allow_replan"]:
        planner_invoked = True
        plan_receipt = replanner()
    return {
        "features": features, "prediction": prediction, "decision": decision,
        "planner_invoked": planner_invoked, "planner_receipt": plan_receipt,
        "feature_and_gate_wall_seconds": feature_gate_seconds,
    }


def _state_and_suffix_rows(
    scenario: Mapping[str, Any], env: Any, modules: Mapping[str, Any], config: Mapping[str, Any],
    *, state_id: str, event_type: str, forced: bool,
) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    pairs = modules["pairs"]
    safety = _safety_receipt(modules, pairs.model_public_observation(env))
    history = pairs._public_history(env)
    current = history[-1] if history else pairs.model_public_observation(env)
    current = pairs.model_public_observation(env) if not isinstance(current, dict) else current
    snapshot = pairs.capture_snapshot(
        env, state_id=state_id, scenario_id=str(scenario["parent_id"]),
        parent_family=str(scenario["parent_family"]), split=str(scenario["split"]),
        event_type=str(event_type), forced=bool(forced), replanner_version=REPLANNER_VERSION,
    )
    suffix_rows = []
    suffix_jsons = []
    suffix_hashes = []
    for index in range(int(config["labels"]["matched_future_suffixes"])):
        suffix_seed = _suffix_seed(str(scenario["parent_id"]), state_id, int(env.time), index)
        future_events = modules["scenarios"].sample_future_event_suffix(
            copy.deepcopy(dict(scenario)), after_time=int(env.time),
            horizon_minutes=int(config["labels"]["horizon_minutes"]), suffix_seed=suffix_seed,
        )
        suffix_rows.append({"suffix_seed": suffix_seed, "future_events": future_events})
        suffix_json = pairs.prepare_suffix_snapshot(snapshot, future_events=future_events, suffix_seed=suffix_seed)
        suffix_jsons.append(suffix_json)
        suffix_hashes.append(_sha256_bytes(suffix_json.encode("utf-8")))
    return snapshot, suffix_rows, suffix_jsons, suffix_hashes


def _trajectory_source(split: str, condition: str, counts: dict[tuple[str, str], int]) -> str:
    sources = ("Existing-ET", "Periodic-4min", "Random-Allow-Suppress")
    key = (split, condition)
    index = counts.get(key, 0)
    counts[key] = index + 1
    return sources[index % len(sources)]


def _candidate_reservoir_add(
    reservoir: list[dict[str, Any]], candidate: dict[str, Any], seen_count: int,
    limit: int, rng: random.Random,
) -> None:
    if seen_count <= limit:
        reservoir.append(candidate)
        return
    replacement = rng.randrange(seen_count)
    if replacement < limit:
        reservoir[replacement] = candidate


def _scan_collection_trajectory(
    scenario: Mapping[str, Any], source: str, modules: Mapping[str, Any],
    config: Mapping[str, Any], max_states: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one source policy and retain a deterministic reservoir of candidate states."""
    pairs = modules["pairs"]
    env = modules["env_class"](_legacy_environment_scenario(scenario))
    previous = env.reset()
    engine = modules["trigger_class"]({}, {"threshold": 0.5, "delta_min": 4})
    seed = _suffix_seed(
        str(scenario["parent_id"]), f"trajectory:{source}", int(scenario["seed_variant"]), 0,
    )
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    candidate_count = 0
    forced_candidate_count = 0
    random_allow_count = 0
    random_suppress_count = 0
    pending_action = copy.deepcopy(pairs.NOOP_ACTION)
    pending_effective = False
    periodic_interval = int(config.get("collection", {}).get("periodic_interval_minutes", 4))
    horizon = int(scenario["base_scenario"]["horizon"])

    while int(env.time) < horizon:
        action = copy.deepcopy(pending_action)
        _, _, terminated, truncated, _ = env.step(action)
        current = pairs.model_public_observation(env)
        if pending_effective:
            engine.record_reconstruction(int(env.time), True)
        pending_action = copy.deepcopy(pairs.NOOP_ACTION)
        pending_effective = False

        history = pairs._public_history(env)
        detection = engine.detect(previous, current, history=history[-4:])
        event_decision = engine.decide(
            detection, int(env.time), mode="periodic" if source == "Periodic-4min" else "event",
        )
        safety = _safety_receipt(modules, current)
        forced = bool(detection.get("strong_event")) or _safety_forced(safety) or not bool(safety.get("safe_to_suppress"))
        current, disposition, safety_apply_receipt = _apply_live_safety_disposition(
            modules, env, current, safety,
        )
        periodic_due = source == "Periodic-4min" and int(env.time) % max(1, periodic_interval) == 0
        is_candidate = forced or bool(event_decision.get("triggered")) or periodic_due
        scope = modules["identify_scope"](current, detection)

        if is_candidate:
            candidate_count += 1
            forced_candidate_count += int(forced)
            features_started = time.perf_counter()
            features = modules["features"].extract_public_features(current, detection, scope)
            feature_seconds = max(0.0, time.perf_counter() - features_started)
            selected_scope = _safety_scope(safety, scope) if forced else scope
            planner_action, planner_receipt, planner_cpu, planner_wall = _invoke_replanner(
                modules, env, selected_scope,
                max_candidates=int(config["replanner"]["max_candidates"]),
            )
            state_id = f"{scenario['parent_id']}-{source.lower().replace('-', '_')}-t{int(env.time):03d}-c{candidate_count:04d}"
            snapshot = pairs.capture_snapshot(
                env, state_id=state_id, scenario_id=str(scenario["parent_id"]),
                parent_family=str(scenario["parent_family"]), split=str(scenario["split"]),
                event_type=str(detection.get("event_type") or "none"), forced=forced,
                replanner_version=REPLANNER_VERSION,
            )
            allowed = forced or source != "Random-Allow-Suppress" or rng.random() < 0.5
            if source == "Random-Allow-Suppress" and not forced:
                random_allow_count += int(allowed)
                random_suppress_count += int(not allowed)
            row = {
                "state_id": state_id, "time": int(env.time), "event_type": str(detection.get("event_type") or "none"),
                "forced": forced, "detection": copy.deepcopy(detection), "safety": copy.deepcopy(safety),
                "safety_disposition": disposition, "safety_apply_receipt": safety_apply_receipt,
                "features": features,
                "feature_gate_seconds": feature_seconds, "snapshot_json": snapshot,
                "replan_action": copy.deepcopy(planner_action), "planner_receipt": planner_receipt,
                "planner_cpu_seconds": planner_cpu, "planner_wall_seconds": planner_wall,
                "trajectory_source": source, "trajectory_id": f"{scenario['parent_id']}::{source}",
                "source_action_allowed": bool(allowed), "candidate_ordinal": candidate_count,
                "unknown_reason": disposition.get("unknown_reason"),
            }
            _candidate_reservoir_add(reservoir, row, candidate_count, max_states, rng)
            if allowed:
                pending_action = copy.deepcopy(planner_action)
                pending_effective = bool(planner_action.get("queue_operations") or planner_action.get("dispatches"))

        previous = current
        if terminated or truncated:
            break

    reservoir.sort(key=lambda row: (int(row["time"]), int(row["candidate_ordinal"])))
    return reservoir, {
        "trajectory_source": source, "trajectory_id": f"{scenario['parent_id']}::{source}",
        "candidate_event_count": candidate_count, "selected_candidate_count": len(reservoir),
        "forced_candidate_count": forced_candidate_count,
        "random_allow_count": random_allow_count, "random_suppress_count": random_suppress_count,
        "reservoir_limit": max_states, "reservoir_seed": seed,
        "episode_end_time": int(env.time),
    }


def _collection_scenarios(
    config: Mapping[str, Any], *, limit_per_split: int | None, conditions: list[str] | None,
    splits: list[str] | None,
) -> tuple[Any, dict[str, int], list[str], list[str], bool]:
    import gain_scenarios

    configured = config["scenario"]["formal_instances_per_condition"]
    counts = {split: int(configured[split]) for split in ("train", "validation", "test")}
    if limit_per_split is not None:
        if isinstance(limit_per_split, bool) or limit_per_split < 1:
            raise ValueError("GAIN_RUN_LIMIT_INSTANCES_PER_SPLIT_MUST_BE_POSITIVE")
        counts = {split: int(limit_per_split) for split in counts}
    selected_splits = list(splits or ("train", "validation"))
    if not selected_splits or len(selected_splits) != len(set(selected_splits)):
        raise ValueError("GAIN_RUN_COLLECTION_SPLITS_INVALID")
    if set(selected_splits) - {"train", "validation"}:
        raise ValueError("GAIN_RUN_TEST_LABELS_MUST_BE_GENERATED_AFTER_MODEL_FREEZE")
    counts = {split: (counts[split] if split in selected_splits else 0) for split in counts}
    selected_conditions = list(conditions or gain_scenarios.FORMAL_CONDITIONS)
    if not selected_conditions or len(selected_conditions) != len(set(selected_conditions)):
        raise ValueError("GAIN_RUN_FORMAL_CONDITIONS_INVALID")
    if set(selected_conditions) - set(gain_scenarios.FORMAL_CONDITIONS):
        raise ValueError("GAIN_RUN_FORMAL_CONDITION_UNKNOWN")
    full_scale = (
        limit_per_split is None
        and set(selected_splits) == {"train", "validation"}
        and selected_conditions == list(gain_scenarios.FORMAL_CONDITIONS)
    )
    iterator = gain_scenarios.iter_gain_scenarios(counts, seed=8201, conditions=tuple(selected_conditions))
    return iterator, counts, selected_conditions, selected_splits, full_scale


def _safe_artifact_path(root: Path, relative_text: Any, code: str) -> Path:
    if not isinstance(relative_text, str) or not relative_text:
        raise ValueError(f"{code}:PATH_REQUIRED")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{code}:PATH_UNSAFE")
    target = (root / relative).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        raise ValueError(f"{code}:FILE_MISSING_OR_OUTSIDE_ROOT")
    return target


def _resolve_collection_inputs(input_texts: list[str] | None) -> list[tuple[Path, str | None]]:
    if not input_texts:
        raise ValueError("GAIN_RUN_TRAIN_INPUT_REQUIRED")
    if len(input_texts) == 1:
        candidate = _resolve_input(input_texts[0])
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            manifest = _read_json(candidate)
            if manifest.get("schema") == COLLECTION_INPUTS_SCHEMA:
                entries = manifest.get("collections")
                if not isinstance(entries, list) or not entries:
                    raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_EMPTY")
                resolved: list[tuple[Path, str | None]] = []
                base = candidate.parent.resolve()
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_ENTRY_INVALID")
                    relative_text = entry.get("path")
                    if not isinstance(relative_text, str) or not relative_text:
                        raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_PATH_REQUIRED")
                    relative = Path(relative_text)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_PATH_UNSAFE")
                    directory = (base / relative).resolve(strict=True)
                    if not directory.is_dir() or base not in directory.parents:
                        raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_PATH_OUTSIDE_ROOT")
                    expected = entry.get("evidence_manifest_sha256")
                    if not isinstance(expected, str) or len(expected) != 64:
                        raise ValueError("GAIN_RUN_COLLECTION_INPUT_MANIFEST_EVIDENCE_HASH_REQUIRED")
                    resolved.append((directory, expected))
                if len({path for path, _ in resolved}) != len(resolved):
                    raise ValueError("GAIN_RUN_COLLECTION_INPUT_DUPLICATE_DIRECTORY")
                return resolved
    resolved_inputs = []
    for input_text in input_texts:
        path = _resolve_input(input_text)
        if not path.is_dir():
            raise ValueError("GAIN_RUN_COLLECTION_DIRECTORY_REQUIRED")
        resolved_inputs.append((path, None))
    if len({path for path, _ in resolved_inputs}) != len(resolved_inputs):
        raise ValueError("GAIN_RUN_COLLECTION_INPUT_DUPLICATE_DIRECTORY")
    return resolved_inputs


def _verify_collection_manifest(
    collection_dir: Path, expected_evidence_sha256: str | None, expected_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path], str]:
    run_path = collection_dir / "RUN.json"
    evidence_path = collection_dir / "EVIDENCE_MANIFEST.json"
    report_path = collection_dir / "COLLECTION_REPORT.json"
    source_path = collection_dir / "SOURCE_MANIFEST.json"
    for path in (run_path, evidence_path, report_path, source_path):
        if not path.is_file():
            raise ValueError(f"GAIN_RUN_COLLECTION_MANIFEST_FILE_MISSING:{path.name}")
    evidence_sha = _sha256_file(evidence_path)
    if expected_evidence_sha256 is not None and evidence_sha != expected_evidence_sha256:
        raise ValueError("GAIN_RUN_COLLECTION_INPUT_EVIDENCE_HASH_MISMATCH")
    run = _read_json(run_path)
    evidence = _read_json(evidence_path)
    report = _read_json(report_path)
    if (
        run.get("schema") != RUN_SCHEMA or run.get("stage") != "collect" or run.get("status") != "complete"
        or evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("stage") != "collect"
        or evidence.get("status") != "complete" or evidence.get("run_id") != run.get("run_id")
        or evidence.get("config_sha256") != run.get("config_sha256")
        or evidence.get("source_manifest_sha256") != run.get("source_manifest_sha256")
        or report.get("schema") != COLLECTION_SCHEMA or report.get("status") != "complete"
        or report.get("run_id") != run.get("run_id")
        or report.get("config_sha256") != run.get("config_sha256")
        or report.get("source_manifest_sha256") != run.get("source_manifest_sha256")
    ):
        raise ValueError("GAIN_RUN_COLLECTION_RUN_EVIDENCE_BINDING_INVALID")
    if run.get("config_sha256") != expected_config_sha256:
        raise ValueError("GAIN_RUN_COLLECTION_CONFIG_HASH_MISMATCH")
    if run.get("replanner_version", REPLANNER_VERSION) != REPLANNER_VERSION:
        raise ValueError("GAIN_RUN_COLLECTION_REPLANNER_VERSION_MISMATCH")
    if evidence.get("replanner_version", REPLANNER_VERSION) != REPLANNER_VERSION:
        raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_REPLANNER_MISMATCH")
    source_hash = _sha256_file(source_path)
    if run.get("source_manifest_path") != "SOURCE_MANIFEST.json" or run.get("source_manifest_sha256") != source_hash:
        raise ValueError("GAIN_RUN_COLLECTION_SOURCE_MANIFEST_HASH_MISMATCH")
    source = _read_json(source_path)
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("GAIN_RUN_COLLECTION_SOURCE_MANIFEST_SCHEMA")
    source_files = source.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("GAIN_RUN_COLLECTION_SOURCE_MANIFEST_EMPTY")
    for source_entry in source_files:
        if (
            not isinstance(source_entry, Mapping) or not isinstance(source_entry.get("path"), str)
            or not isinstance(source_entry.get("sha256"), str) or len(source_entry["sha256"]) != 64
            or type(source_entry.get("size_bytes")) is not int or source_entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_COLLECTION_SOURCE_ENTRY_INVALID")
    static_inputs = source.get("static_inputs")
    if not isinstance(static_inputs, list) or source.get("static_inputs_sha256") != _sha256_bytes(_canonical(static_inputs)):
        raise ValueError("GAIN_RUN_COLLECTION_STATIC_INPUT_MANIFEST_INVALID")
    source_file_lookup = {str(entry["path"]): entry for entry in source_files}
    for entry in static_inputs:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("GAIN_RUN_COLLECTION_STATIC_INPUT_ENTRY_INVALID")
        path = str(entry["path"])
        if not (path.startswith("package/civilian/fixtures/") or path.startswith("package/civilian/registered_data/")):
            raise ValueError("GAIN_RUN_COLLECTION_STATIC_INPUT_PATH_INVALID")
        if source_file_lookup.get(path) != dict(entry):
            raise ValueError("GAIN_RUN_COLLECTION_STATIC_INPUT_NOT_BOUND")
    file_entries = evidence.get("files")
    if not isinstance(file_entries, list):
        raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_FILES_INVALID")
    verified: dict[str, Path] = {}
    for entry in file_entries:
        if (
            not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_ENTRY_INVALID")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_PATH_UNSAFE")
        target = (collection_dir / relative).resolve()
        if collection_dir.resolve() not in target.parents or not target.is_file():
            raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_FILE_MISSING")
        relative_key = relative.as_posix()
        if relative_key in verified:
            raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_DUPLICATE_PATH")
        if target.stat().st_size != entry["size_bytes"] or _sha256_file(target) != entry["sha256"]:
            raise ValueError(f"GAIN_RUN_COLLECTION_EVIDENCE_FILE_HASH_MISMATCH:{relative_key}")
        verified[relative_key] = target
    for required in ("RUN.json", "STAGE_RESULT.json", "COLLECTION_REPORT.json", "SOURCE_MANIFEST.json"):
        if required not in verified:
            raise ValueError(f"GAIN_RUN_COLLECTION_EVIDENCE_REQUIRED_FILE_MISSING:{required}")
    if _sha256_file(verified["RUN.json"]) != _sha256_file(run_path):
        raise ValueError("GAIN_RUN_COLLECTION_RUN_HASH_BINDING_MISMATCH")
    if _read_json(collection_dir / "STAGE_RESULT.json") != report:
        raise ValueError("GAIN_RUN_COLLECTION_STAGE_REPORT_MISMATCH")
    return run, evidence, report, verified, evidence_sha


def _compact_verified_collection(
    collection_dir: Path, run: Mapping[str, Any], report: Mapping[str, Any],
    verified_files: Mapping[str, Path], modules: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("GAIN_RUN_COLLECTION_ARTIFACTS_REQUIRED")
    rows_artifact = artifacts.get("rows")
    outcomes_artifact = artifacts.get("scenario_outcomes")
    if not isinstance(rows_artifact, Mapping) or not isinstance(outcomes_artifact, Mapping):
        raise ValueError("GAIN_RUN_COLLECTION_ROW_ARTIFACTS_REQUIRED")
    rows_path = _safe_artifact_path(collection_dir, rows_artifact.get("path"), "COLLECTION_ROWS")
    outcomes_path = _safe_artifact_path(collection_dir, outcomes_artifact.get("path"), "COLLECTION_OUTCOMES")
    if rows_path != verified_files.get(Path(rows_artifact["path"]).as_posix()):
        raise ValueError("GAIN_RUN_COLLECTION_ROWS_NOT_IN_EVIDENCE")
    if outcomes_path != verified_files.get(Path(outcomes_artifact["path"]).as_posix()):
        raise ValueError("GAIN_RUN_COLLECTION_OUTCOMES_NOT_IN_EVIDENCE")
    if (
        _sha256_file(rows_path) != rows_artifact.get("sha256")
        or _sha256_file(outcomes_path) != outcomes_artifact.get("sha256")
        or rows_path.stat().st_size != rows_artifact.get("size_bytes")
        or outcomes_path.stat().st_size != outcomes_artifact.get("size_bytes")
    ):
        raise ValueError("GAIN_RUN_COLLECTION_ROW_ARTIFACT_HASH_MISMATCH")
    if report.get("test_labels_collected") is not False:
        raise ValueError("GAIN_RUN_COLLECTION_TEST_LABEL_ACCESS_FORBIDDEN")
    if report.get("replanner_version") != REPLANNER_VERSION:
        raise ValueError("GAIN_RUN_COLLECTION_REPORT_REPLANNER_MISMATCH")

    scenario_records: dict[str, dict[str, Any]] = {}
    scenario_count = 0
    for outcome in _iter_gzip_jsonl(outcomes_path):
        scenario_id = str(outcome.get("scenario_id", ""))
        if not scenario_id or scenario_id in scenario_records:
            raise ValueError("GAIN_RUN_COLLECTION_SCENARIO_OUTCOME_ID_INVALID")
        scenario_path = _safe_artifact_path(collection_dir, outcome.get("scenario_path"), "COLLECTION_SCENARIO")
        relative_path = Path(str(outcome["scenario_path"])).as_posix()
        if verified_files.get(relative_path) != scenario_path or _sha256_file(scenario_path) != outcome.get("scenario_file_sha256"):
            raise ValueError("GAIN_RUN_COLLECTION_SCENARIO_FILE_HASH_MISMATCH")
        scenario = _read_gzip_json(scenario_path)
        modules["scenarios"].validate_gain_scenario(scenario)
        if (
            scenario.get("parent_id") != scenario_id or scenario.get("split") != outcome.get("split")
            or scenario.get("condition") != outcome.get("condition")
            or scenario.get("parent_family") != outcome.get("parent_family")
            or scenario.get("scenario_content_sha256") != outcome.get("scenario_content_sha256")
            or _sha256_bytes(_canonical(scenario.get("event_tape"))) != outcome.get("event_tape_sha256")
        ):
            raise ValueError("GAIN_RUN_COLLECTION_SCENARIO_OUTCOME_BINDING_MISMATCH")
        scenario_records[scenario_id] = {
            "split": str(outcome["split"]), "condition": str(outcome["condition"]),
            "parent_family": str(outcome["parent_family"]),
            "scenario_content_sha256": str(outcome["scenario_content_sha256"]),
            "event_tape_sha256": str(outcome["event_tape_sha256"]),
            "event_tape_id": str(scenario["event_tape_id"]),
            "scenario_seed": int(scenario["seed_variant"]),
            "scenario_seed_token": f"{scenario['parent_family']}:{scenario['condition']}:{int(scenario['seed_variant'])}",
            "scenario_path": str(outcome["scenario_path"]),
            "scenario_file_sha256": str(outcome["scenario_file_sha256"]),
        }
        scenario_count += 1
    if scenario_count != int(outcomes_artifact.get("row_count", -1)):
        raise ValueError("GAIN_RUN_COLLECTION_SCENARIO_OUTCOME_COUNT_MISMATCH")

    compact_rows: list[dict[str, Any]] = []
    variance_sources: dict[str, dict[str, Any]] = {}
    seen_states: set[str] = set()
    for row in _iter_gzip_jsonl(rows_path):
        if row.get("schema") != "gain-gate-training-row/1":
            raise ValueError("GAIN_RUN_COLLECTION_TRAINING_ROW_SCHEMA")
        state_id = str(row.get("state_id", ""))
        scenario_id = str(row.get("scenario_id", ""))
        if not state_id or state_id in seen_states:
            raise ValueError("GAIN_RUN_COLLECTION_STATE_ID_DUPLICATE_OR_EMPTY")
        seen_states.add(state_id)
        scenario_info = scenario_records.get(scenario_id)
        if scenario_info is None:
            raise ValueError("GAIN_RUN_COLLECTION_ROW_SCENARIO_UNKNOWN")
        for key in ("split", "condition", "parent_family", "scenario_content_sha256", "event_tape_sha256", "event_tape_id", "scenario_seed"):
            if row.get(key) != scenario_info[key]:
                raise ValueError(f"GAIN_RUN_COLLECTION_ROW_SCENARIO_BINDING_MISMATCH:{key}")
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("GAIN_RUN_COLLECTION_NONTRAIN_SPLIT_REJECTED")
        if row.get("replanner_version") != REPLANNER_VERSION:
            raise ValueError("GAIN_RUN_COLLECTION_ROW_REPLANNER_MISMATCH")
        feature_map = row.get("public_features")
        if not isinstance(feature_map, Mapping) or not feature_map or row.get("features") != feature_map:
            raise ValueError("GAIN_RUN_COLLECTION_PUBLIC_FEATURES_REQUIRED")
        import gain_models
        feature_map = gain_models._feature_map(feature_map)
        for timing_key in ("feature_gate_seconds", "planner_cpu_seconds", "planner_wall_seconds"):
            value = row.get(timing_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"GAIN_RUN_COLLECTION_TIMING_INVALID:{timing_key}")
        if type(row.get("forced")) is not bool or type(row.get("valid_mask")) is not bool:
            raise ValueError("GAIN_RUN_COLLECTION_LABEL_FLAGS_INVALID")

        snapshot_path = _safe_artifact_path(collection_dir, row.get("snapshot_path"), "COLLECTION_SNAPSHOT")
        snapshot_rel = Path(str(row["snapshot_path"])).as_posix()
        if verified_files.get(snapshot_rel) != snapshot_path or _sha256_file(snapshot_path) != row.get("snapshot_file_sha256"):
            raise ValueError("GAIN_RUN_COLLECTION_SNAPSHOT_FILE_HASH_MISMATCH")
        snapshot_payload = _read_gzip_json(snapshot_path)
        snapshot_text = snapshot_payload.get("snapshot_json")
        if not isinstance(snapshot_text, str):
            raise ValueError("GAIN_RUN_COLLECTION_SNAPSHOT_CONTENT_MISSING")
        snapshot = json.loads(snapshot_text)
        if (
            snapshot.get("state_id") != state_id
            or snapshot.get("scenario_id") != scenario_id
            or snapshot.get("snapshot_sha256") != row.get("snapshot_sha256")
            or snapshot.get("private_snapshot_sha256") != row.get("private_snapshot_sha256")
            or snapshot.get("public_history_sha256") != row.get("public_history_sha256")
        ):
            raise ValueError("GAIN_RUN_COLLECTION_SNAPSHOT_BINDING_MISMATCH")
        suffix_paths = row.get("suffix_paths")
        suffix_file_hashes = row.get("suffix_file_hashes")
        suffix_hashes = row.get("suffix_hashes")
        suffix_seeds = row.get("suffix_seed")
        if not all(isinstance(value, list) for value in (suffix_paths, suffix_file_hashes, suffix_hashes, suffix_seeds)):
            raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_BINDING_INVALID")
        if not (len(suffix_paths) == len(suffix_file_hashes) == len(suffix_hashes) == len(suffix_seeds)):
            raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_COUNT_MISMATCH")
        for index, relative in enumerate(suffix_paths):
            suffix_path = _safe_artifact_path(collection_dir, relative, "COLLECTION_SUFFIX")
            if verified_files.get(Path(str(relative)).as_posix()) != suffix_path:
                raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_NOT_IN_EVIDENCE")
            if _sha256_file(suffix_path) != suffix_file_hashes[index]:
                raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_FILE_HASH_MISMATCH")
            suffix_payload = _read_gzip_json(suffix_path)
            suffix_text = suffix_payload.get("suffix_json")
            if not isinstance(suffix_text, str) or _sha256_bytes(suffix_text.encode("utf-8")) != suffix_hashes[index]:
                raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_CONTENT_HASH_MISMATCH")
            suffix_obj = json.loads(suffix_text)
            if int(suffix_obj.get("suffix_seed", -1)) != int(suffix_seeds[index]):
                raise ValueError("GAIN_RUN_COLLECTION_SUFFIX_SEED_MISMATCH")

        label_path = _safe_artifact_path(collection_dir, row.get("raw_label_path"), "COLLECTION_LABEL")
        label_rel = Path(str(row["raw_label_path"])).as_posix()
        if verified_files.get(label_rel) != label_path or _sha256_file(label_path) != row.get("raw_label_sha256"):
            raise ValueError("GAIN_RUN_COLLECTION_LABEL_FILE_HASH_MISMATCH")
        label_payload = _read_gzip_json(label_path)
        label = label_payload.get("label")
        if not isinstance(label, Mapping) or label.get("label_sha256") != row.get("label_sha256"):
            raise ValueError("GAIN_RUN_COLLECTION_LABEL_BINDING_MISMATCH")
        oracle_result = modules["oracle"].validate_gain_label(label)
        label_suffix_count = _validate_collection_label_suffix_binding(
            label, suffix_seeds, forced=bool(row["forced"]),
        )
        label_bindings = {
            "state_id": state_id, "scenario_id": scenario_id, "parent_family": row["parent_family"],
            "split": row["split"], "event_type": row["event_type"], "forced": row["forced"],
            "replanner_version": REPLANNER_VERSION, "source_snapshot_sha256": row["snapshot_sha256"],
            "private_snapshot_sha256": row["private_snapshot_sha256"],
            "public_history_sha256": row["public_history_sha256"], "suffix_count": label_suffix_count,
        }
        if any(label.get(key) != value for key, value in label_bindings.items()):
            raise ValueError("GAIN_RUN_COLLECTION_LABEL_ROW_BINDING_MISMATCH")
        label_valid = oracle_result.get("label_valid_mask") is True
        if label_valid != row["valid_mask"]:
            raise ValueError("GAIN_RUN_COLLECTION_ROW_ORACLE_MASK_MISMATCH")
        if label_valid:
            value = oracle_result.get("recomputed_gain_mean")
            variance = oracle_result.get("recomputed_paired_variance")
            if (
                value is None or not math.isclose(float(value), float(row.get("gain_mean")), rel_tol=1e-10, abs_tol=1e-10)
                or variance != row.get("paired_variance") and not (
                    variance is not None and row.get("paired_variance") is not None
                    and math.isclose(float(variance), float(row["paired_variance"]), rel_tol=1e-10, abs_tol=1e-10)
                )
            ):
                raise ValueError("GAIN_RUN_COLLECTION_ROW_ORACLE_VALUE_MISMATCH")
        elif row.get("gain_mean") is not None or row.get("paired_variance") is not None:
            raise ValueError("GAIN_RUN_COLLECTION_INVALID_ROW_HAS_GAIN")

        compact_rows.append({
            "schema": row["schema"], "state_id": state_id, "scenario_id": scenario_id,
            "scenario_instance_id": scenario_id, "parent_id": scenario_id,
            "parent_family": str(row["parent_family"]), "condition": str(row["condition"]),
            "split": str(row["split"]), "trajectory_id": str(row["trajectory_id"]),
            "trajectory_source": str(row["trajectory_source"]),
            "scenario_content_sha256": str(row["scenario_content_sha256"]),
            "snapshot_sha256": str(row["snapshot_sha256"]),
            "label_sha256": str(row["label_sha256"]),
            "private_snapshot_sha256": str(row["private_snapshot_sha256"]),
            "public_history_sha256": str(row["public_history_sha256"]),
            "scenario_seed": int(row["scenario_seed"]), "suffix_seed": list(map(int, suffix_seeds)),
            "time": int(row["time"]), "event_type": str(row["event_type"]),
            "forced": bool(row["forced"]), "public_features": dict(feature_map), "features": dict(feature_map),
            "gain_mean": float(oracle_result["recomputed_gain_mean"]) if label_valid else None,
            "paired_variance": oracle_result.get("recomputed_paired_variance") if label_valid else None,
            "suffix_gains": copy.deepcopy(row.get("suffix_gains", [])) if label_valid else [],
            "valid_mask": bool(label_valid), "unknown_reason": row.get("unknown_reason"),
            "feature_gate_seconds": float(row["feature_gate_seconds"]),
            "planner_cpu_seconds": float(row["planner_cpu_seconds"]),
            "planner_wall_seconds": float(row["planner_wall_seconds"]),
            "collection_run_id": str(run["run_id"]),
            "collection_evidence_manifest_sha256": str(report.get("evidence_manifest_sha256", "")),
        })
        variance_sources[state_id] = {
            "collection_directory": collection_dir,
            "scenario_path": scenario_info["scenario_path"],
            "scenario_file_sha256": scenario_info["scenario_file_sha256"],
            "snapshot_path": str(row["snapshot_path"]),
            "snapshot_file_sha256": str(row["snapshot_file_sha256"]),
            "suffix_paths": [str(value) for value in suffix_paths],
            "suffix_file_hashes": [str(value) for value in suffix_file_hashes],
            "label_path": str(row["raw_label_path"]),
            "label_file_sha256": str(row["raw_label_sha256"]),
        }
    if len(compact_rows) != int(rows_artifact.get("row_count", -1)) or len(compact_rows) != int(report.get("row_count", -2)):
        raise ValueError("GAIN_RUN_COLLECTION_ROW_COUNT_MISMATCH")
    return compact_rows, {
        "run_id": str(run["run_id"]), "directory": str(collection_dir),
        "evidence_manifest_sha256": str(_sha256_file(collection_dir / "EVIDENCE_MANIFEST.json")),
        "config_sha256": str(run["config_sha256"]),
        "source_manifest_sha256": str(run["source_manifest_sha256"]),
        "full_training_scale": bool(report.get("full_training_scale", False)),
        "scenario_count": scenario_count, "row_count": len(compact_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in compact_rows).items())),
        "scenario_ids": sorted(scenario_records),
        "scenario_content_hashes": sorted(row["scenario_content_sha256"] for row in scenario_records.values()),
        "scenario_seed_tokens": sorted(row["scenario_seed_token"] for row in scenario_records.values()),
        "event_tape_hashes": sorted(row["event_tape_sha256"] for row in scenario_records.values()),
        "private_snapshot_hashes": sorted(row["private_snapshot_sha256"] for row in compact_rows),
        "suffix_seeds": sorted({seed for row in compact_rows for seed in row["suffix_seed"]}),
    }, variance_sources


def _assert_collection_identity_disjoint(previous: Mapping[str, set[str]], current: Mapping[str, Any]) -> None:
    identity_keys = (
        "scenario_ids", "scenario_content_hashes", "scenario_seed_tokens",
        "private_snapshot_hashes", "suffix_seeds",
    )
    for key in identity_keys:
        values = current.get(key, [])
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f"GAIN_RUN_COLLECTION_IDENTITY_VALUES_INVALID:{key}")
        if previous.get(key, set()).intersection(map(str, values)):
            raise ValueError(f"GAIN_RUN_COLLECTION_INPUT_OVERLAP:{key}")


def _load_verified_collection_rows(
    input_texts: list[str] | None, config_sha256: str, modules: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    inputs = _resolve_collection_inputs(input_texts)
    all_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    variance_sources: dict[str, dict[str, Any]] = {}
    seen_states: set[str] = set()
    seen_identity: dict[str, set[str]] = {
        key: set() for key in (
            "scenario_ids", "scenario_content_hashes", "scenario_seed_tokens",
            "private_snapshot_hashes", "suffix_seeds",
        )
    }
    for directory, expected_evidence_sha in inputs:
        run, _, report, verified, evidence_sha = _verify_collection_manifest(
            directory, expected_evidence_sha, config_sha256,
        )
        rows, identity, collection_variance_sources = _compact_verified_collection(
            directory, run, report, verified, modules,
        )
        if evidence_sha != identity["evidence_manifest_sha256"]:
            raise ValueError("GAIN_RUN_COLLECTION_EVIDENCE_IDENTITY_MISMATCH")
        _assert_collection_identity_disjoint(seen_identity, identity)
        for key in seen_identity:
            seen_identity[key].update(map(str, identity[key]))
        for row in rows:
            if row["state_id"] in seen_states:
                raise ValueError("GAIN_RUN_COLLECTION_INPUT_STATE_OVERLAP")
            seen_states.add(row["state_id"])
            variance_sources[row["state_id"]] = collection_variance_sources[row["state_id"]]
        all_rows.extend(rows)
        identities.append(identity)
    import gain_models
    leakage = gain_models.check_group_leakage(all_rows)
    if leakage.get("split_leakage_count") != 0:
        raise ValueError("GAIN_RUN_COLLECTION_GROUP_LEAKAGE")
    if {row["split"] for row in all_rows} - {"train", "validation"}:
        raise ValueError("GAIN_RUN_COLLECTION_TEST_SPLIT_REJECTED")
    if not any(row["split"] == "train" for row in all_rows) or not any(row["split"] == "validation" for row in all_rows):
        raise ValueError("GAIN_RUN_COLLECTION_TRAIN_AND_VALIDATION_REQUIRED")
    return all_rows, identities, variance_sources


def _load_variance_record(
    row: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    state_id = str(row.get("state_id", ""))
    source = sources.get(state_id)
    if not state_id or not isinstance(source, Mapping):
        raise ValueError("GAIN_RUN_VARIANCE_SOURCE_MISSING")
    collection_dir = Path(source["collection_directory"])
    scenario_path = _safe_artifact_path(collection_dir, source["scenario_path"], "VARIANCE_SCENARIO")
    if _sha256_file(scenario_path) != source["scenario_file_sha256"]:
        raise ValueError("GAIN_RUN_VARIANCE_SCENARIO_HASH_MISMATCH")
    scenario = _read_gzip_json(scenario_path)
    snapshot_path = _safe_artifact_path(collection_dir, source["snapshot_path"], "VARIANCE_SNAPSHOT")
    if _sha256_file(snapshot_path) != source["snapshot_file_sha256"]:
        raise ValueError("GAIN_RUN_VARIANCE_SNAPSHOT_HASH_MISMATCH")
    snapshot_payload = _read_gzip_json(snapshot_path)
    snapshot_json = snapshot_payload.get("snapshot_json")
    if not isinstance(snapshot_json, str):
        raise ValueError("GAIN_RUN_VARIANCE_SNAPSHOT_CONTENT_MISSING")
    suffix_jsons: list[str] = []
    for relative, expected_sha in zip(
        source["suffix_paths"], source["suffix_file_hashes"], strict=True,
    ):
        suffix_path = _safe_artifact_path(collection_dir, relative, "VARIANCE_SUFFIX")
        if _sha256_file(suffix_path) != expected_sha:
            raise ValueError("GAIN_RUN_VARIANCE_SUFFIX_HASH_MISMATCH")
        suffix_payload = _read_gzip_json(suffix_path)
        suffix_json = suffix_payload.get("suffix_json")
        if not isinstance(suffix_json, str):
            raise ValueError("GAIN_RUN_VARIANCE_SUFFIX_CONTENT_MISSING")
        suffix_jsons.append(suffix_json)
    label_path = _safe_artifact_path(collection_dir, source["label_path"], "VARIANCE_LABEL")
    if _sha256_file(label_path) != source["label_file_sha256"]:
        raise ValueError("GAIN_RUN_VARIANCE_LABEL_HASH_MISMATCH")
    label_payload = _read_gzip_json(label_path)
    label = label_payload.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("GAIN_RUN_VARIANCE_LABEL_CONTENT_MISSING")
    if (
        row.get("snapshot_sha256") != json.loads(snapshot_json).get("snapshot_sha256")
        or row.get("label_sha256") != label.get("label_sha256")
    ):
        raise ValueError("GAIN_RUN_VARIANCE_ROW_ARTIFACT_BINDING_MISMATCH")
    return {
        "row": row, "scenario": scenario, "snapshot_json": snapshot_json,
        "label": label, "suffix_jsons": suffix_jsons,
    }


def _training_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = config["models"]
    return {
        "validation_partition_seed": 8201,
        "epsilon_grid": list(model_config.get("epsilon_grid", [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])),
        "calibration_coverage": float(model_config.get("calibration_coverage", 0.9)),
        "ridge_alpha": float(model_config.get("ridge_alpha", 1.0)),
        "gradient_boosting": {"depths": list(model_config.get("tree_depths", [3, 5])), "seed": 8201},
        "mlp_seeds": list(model_config.get("seeds", [8201, 8202, 8203])),
        "mlp": {
            "learning_rate": float(model_config.get("mlp_learning_rate", 0.001)),
            "batch_size": int(model_config.get("mlp_batch_size", 32)),
            "epochs": int(model_config.get("mlp_epochs", 100)),
            "patience": int(model_config.get("mlp_early_stop_patience", 10)),
        },
    }


def _train_light(
    output: Path, run: dict[str, Any], config: dict[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    import gain_diagnostics
    import gain_models

    modules = _load_smoke_modules()
    _check_resource_account(args)
    rows, collection_identities, variance_sources = _load_verified_collection_rows(
        args.input, run["config_sha256"], modules,
    )
    _check_resource_account(args)
    import gain_variance
    variance_started_cpu = time.process_time()
    variance_started_wall = time.perf_counter()
    variance_diagnostic = gain_variance.run_gain_variance_diagnostic(
        rows, output / "variance", modules=modules,
        continuation_policy=_make_safety_continuation(modules, config),
        record_loader=lambda row: _load_variance_record(row, variance_sources),
        resource_check=lambda: _check_resource_account(args),
    )
    variance_cpu = max(0.0, time.process_time() - variance_started_cpu)
    variance_wall = max(0.0, time.perf_counter() - variance_started_wall)
    _check_resource_account(args)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    model_config = _training_model_config(config)
    training_started_cpu = time.process_time()
    training_started_wall = time.perf_counter()
    resource_check = lambda: _check_resource_account(args)
    model_training = gain_models.train_light_models(rows, models_dir, model_config, resource_check=resource_check)
    _check_resource_account(args)
    training_cpu = max(0.0, time.process_time() - training_started_cpu)
    training_wall = max(0.0, time.perf_counter() - training_started_wall)
    feature_row = next((row for row in rows if row["valid_mask"] and not row["forced"]), None)
    if feature_row is None:
        raise ValueError("GAIN_RUN_TRAIN_NO_ORDINARY_VALID_FEATURE_ROW")
    model_records: list[dict[str, Any]] = model_training.get("models", [])
    restored: list[dict[str, Any]] = []
    per_model_timings: dict[str, dict[str, Any]] = {}
    timing_rows = [row for row in rows if row["valid_mask"] and not row["forced"]]
    for record in model_records:
        model_name = str(record["model_name"])
        bundle_path = models_dir / str(record["bundle_file"])
        bundle = gain_models.load_bundle(bundle_path)
        runtime = gain_models.load_runtime(bundle_path)
        probe_features = feature_row["public_features"]
        restore = gain_models.fresh_process_restore_check(bundle_path, probe_features)
        account = getattr(args, "resource_account", None)
        if account is not None:
            account.add_child_cpu_seconds(float(restore.get("fresh_process_cpu_seconds", 0.0)))
        _check_resource_account(args)
        if restore.get("fresh_process") is not True or restore.get("prediction_matches") is not True:
            raise ValueError(f"GAIN_RUN_TRAIN_FRESH_PROCESS_RESTORE_FAILED:{model_name}")
        restored.append({"model_name": model_name, **restore})
        for row in timing_rows[:5]:
            runtime.predict(row["public_features"])
        prediction_times: list[float] = []
        for row in timing_rows:
            runtime.predict(row["public_features"])
            prediction_times.append(float(runtime.forward_seconds[-1]))
        feature_gate = [float(row["feature_gate_seconds"]) + value for row, value in zip(timing_rows, prediction_times, strict=True)]
        planner = [float(row["planner_wall_seconds"]) for row in timing_rows]
        per_model_timings[model_name] = {
            "feature_gate_seconds": feature_gate,
            "planner_seconds": planner,
            "predictor_forward_seconds": prediction_times,
            "cold_load_seconds": float(runtime.cold_load_seconds),
            "optimizer_state_restored": bool(runtime.optimizer_state_restored),
        }

    primary_model = next((name for name in per_model_timings if name == "mlp_seed_8201"), next(iter(per_model_timings)))
    diagnostics_dir = output / "diagnostics"
    diagnostics = gain_diagnostics.run_learnability_diagnostics(
        rows, diagnostics_dir,
        bootstrap_replicates=int(config["statistics"].get("bootstrap_replicates", 10_000)),
        seed=8201,
        planner_timing={
            "feature_gate_seconds": per_model_timings[primary_model]["feature_gate_seconds"],
            "planner_seconds": per_model_timings[primary_model]["planner_seconds"],
        },
        resource_check=resource_check,
    )
    counts = diagnostics["counts"]
    cv_models = diagnostics.get("grouped_cross_validation", {}).get("models", {})
    grouped_signal = any(
        isinstance(item, Mapping) and item.get("grouped_cv_signal") == "IMPROVES_OVER_CONSTANT"
        for item in cv_models.values()
    )
    timing = diagnostics.get("timing", {})
    entry_checks = {
        "ordinary_positive_states_ge_30": int(counts.get("positive_ordinary_state_count", 0)) >= 30,
        "ordinary_nonpositive_states_ge_30": int(counts.get("nonpositive_ordinary_state_count", 0)) >= 30,
        "grouped_cv_beats_constant": grouped_signal,
        "oracle_headroom_positive": float(diagnostics.get("label_distribution", {}).get("oracle_positive_action_mean_gain_scenario_weighted") or 0.0) > 0.0,
        "feature_gate_within_20_percent_of_planner": timing.get("status") == "WITHIN_20_PERCENT",
    }
    entry_criteria_met = all(entry_checks.values())
    collection_manifest = {
        "schema": TRAIN_COLLECTION_IDENTITY_SCHEMA,
        "collections": [
            {"run_id": item["run_id"], "evidence_manifest_sha256": item["evidence_manifest_sha256"],
             "source_manifest_sha256": item["source_manifest_sha256"], "scenario_count": item["scenario_count"],
             "row_count": item["row_count"], "split_counts": item["split_counts"]}
            for item in collection_identities
        ],
        "combined_scenario_ids_sha256": _sha256_bytes(_canonical(sorted({row["scenario_id"] for row in rows}))),
        "combined_state_ids_sha256": _sha256_bytes(_canonical(sorted(row["state_id"] for row in rows))),
        "source_evidence_hashes_verified": True,
        "raw_pair_labels_independently_recomputed": True,
        "test_labels_loaded": False,
    }
    _write_json(output / "TRAIN_COLLECTION_IDENTITY.json", collection_manifest)
    model_artifacts = []
    for path in sorted(models_dir.iterdir()):
        if path.is_file():
            model_artifacts.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    model_manifest = {
        "schema": "gain-gate-model-artifacts/1", "run_id": run["run_id"],
        "config_sha256": run["config_sha256"], "training_schema": model_training["schema"],
        "replanner_version": REPLANNER_VERSION, "test_accessed": False, "files": model_artifacts,
    }
    _write_json(models_dir / "MODEL_ARTIFACT_MANIFEST.json", model_manifest)
    report = {
        "schema": TRAIN_STAGE_SCHEMA, "status": "complete", "run_id": run["run_id"],
        "config_sha256": run["config_sha256"], "source_manifest_sha256": run["source_manifest_sha256"],
        "replanner_version": REPLANNER_VERSION, "collection_inputs": collection_manifest,
        "data_scope": "formal_train_validation" if all(item.get("full_training_scale") for item in collection_identities) else "pilot_or_partial",
        "train_row_count": sum(row["split"] == "train" for row in rows),
        "validation_row_count": sum(row["split"] == "validation" for row in rows),
        "test_labels_loaded": False, "model_config": model_config,
        "training_cpu_seconds": training_cpu, "training_wall_seconds": training_wall,
        "model_training": model_training, "learnability_diagnostics": diagnostics,
        "label_variance_diagnostic": variance_diagnostic,
        "label_variance_cpu_seconds": variance_cpu,
        "label_variance_wall_seconds": variance_wall,
        "p2_entry_checks": entry_checks,
        "complex_model_entry_gate": "ENTRY_CRITERIA_MET" if entry_criteria_met else "INSUFFICIENT_EVIDENCE_OR_SIGNAL",
        "gate_is_diagnostic_only": True,
        "fresh_process_restore_checks": restored, "timing_by_model": per_model_timings,
        "model_artifact_manifest_sha256": _sha256_file(models_dir / "MODEL_ARTIFACT_MANIFEST.json"),
        "model_artifact_directory": "models",
    }
    _write_json(output / "TRAIN_LIGHT_REPORT.json", report)
    return report


def _verify_run_evidence(run_dir: Path, expected_stage: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    run_path = run_dir / "RUN.json"
    evidence_path = run_dir / "EVIDENCE_MANIFEST.json"
    if not run_path.is_file() or not evidence_path.is_file():
        raise ValueError("GAIN_RUN_EVIDENCE_BINDING_FILES_MISSING")
    run = _read_json(run_path)
    evidence = _read_json(evidence_path)
    evidence_hash = _sha256_file(evidence_path)
    if (
        run.get("schema") != RUN_SCHEMA or run.get("stage") != expected_stage or run.get("status") != "complete"
        or evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("stage") != expected_stage
        or evidence.get("status") != "complete" or evidence.get("run_id") != run.get("run_id")
        or evidence.get("config_sha256") != run.get("config_sha256")
        or evidence.get("source_manifest_sha256") != run.get("source_manifest_sha256")
    ):
        raise ValueError("GAIN_RUN_EVIDENCE_BINDING_INVALID")
    files = evidence.get("files")
    if not isinstance(files, list):
        raise ValueError("GAIN_RUN_EVIDENCE_FILE_LIST_INVALID")
    verified: dict[str, Path] = {}
    for entry in files:
        if (
            not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_EVIDENCE_FILE_ENTRY_INVALID")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GAIN_RUN_EVIDENCE_PATH_UNSAFE")
        target = (run_dir / relative).resolve()
        if run_dir.resolve() not in target.parents or not target.is_file():
            raise ValueError("GAIN_RUN_EVIDENCE_FILE_MISSING")
        key = relative.as_posix()
        if key in verified:
            raise ValueError("GAIN_RUN_EVIDENCE_DUPLICATE_PATH")
        if target.stat().st_size != entry["size_bytes"] or _sha256_file(target) != entry["sha256"]:
            raise ValueError(f"GAIN_RUN_EVIDENCE_HASH_MISMATCH:{key}")
        verified[key] = target
    for required in ("RUN.json", "STAGE_RESULT.json"):
        if required not in verified:
            raise ValueError(f"GAIN_RUN_EVIDENCE_REQUIRED_FILE_MISSING:{required}")
    return run, evidence, evidence_hash


def _verify_frozen_model_directory(model_dir_text: str | None) -> tuple[dict[str, Path], dict[str, Any]]:
    if not model_dir_text:
        raise ValueError("GAIN_RUN_EVALUATE_MODEL_DIR_REQUIRED")
    model_dir = _resolve_input(model_dir_text)
    if not model_dir.is_dir():
        raise ValueError("GAIN_RUN_EVALUATE_MODEL_DIRECTORY_REQUIRED")
    model_manifest_path = model_dir / "MODEL_ARTIFACT_MANIFEST.json"
    if not model_manifest_path.is_file():
        raise ValueError("GAIN_RUN_MODEL_ARTIFACT_MANIFEST_MISSING")
    model_manifest = _read_json(model_manifest_path)
    if (
        model_manifest.get("schema") != "gain-gate-model-artifacts/1"
        or model_manifest.get("test_accessed") is not False
        or model_manifest.get("replanner_version") != REPLANNER_VERSION
    ):
        raise ValueError("GAIN_RUN_MODEL_ARTIFACT_MANIFEST_INVALID")
    files = model_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("GAIN_RUN_MODEL_ARTIFACT_FILE_LIST_INVALID")
    verified: dict[str, Path] = {}
    for entry in files:
        if (
            not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_MODEL_ARTIFACT_ENTRY_INVALID")
        path = _safe_artifact_path(model_dir, entry["path"], "MODEL_ARTIFACT")
        if entry["path"] in verified:
            raise ValueError("GAIN_RUN_MODEL_ARTIFACT_DUPLICATE_PATH")
        if path.stat().st_size != entry["size_bytes"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"GAIN_RUN_MODEL_ARTIFACT_HASH_MISMATCH:{entry['path']}")
        verified[entry["path"]] = path
    model_run_dir = model_dir.parent
    model_run, _, model_evidence_sha = _verify_run_evidence(model_run_dir, "train-light")
    training_report_path = model_run_dir / "TRAIN_LIGHT_REPORT.json"
    stage_result_path = model_run_dir / "STAGE_RESULT.json"
    if not training_report_path.is_file() or not stage_result_path.is_file():
        raise ValueError("GAIN_RUN_MODEL_TRAINING_REPORT_MISSING")
    training_report = _read_json(training_report_path)
    if (
        training_report.get("schema") != TRAIN_STAGE_SCHEMA or training_report.get("status") != "complete"
        or training_report.get("test_labels_loaded") is not False
        or training_report.get("replanner_version") != REPLANNER_VERSION
        or training_report.get("model_artifact_manifest_sha256") != _sha256_file(model_manifest_path)
        or training_report.get("run_id") != model_run.get("run_id")
    ):
        raise ValueError("GAIN_RUN_MODEL_TRAINING_REPORT_INVALID")
    import gain_models
    mapping: dict[str, Path] = {}
    required_models = {"gain_tree": "gradient_boosting", **{f"gain_mlp_{seed}": f"mlp_seed_{seed}" for seed in (8201, 8202, 8203)}}
    for key, model_name in required_models.items():
        bundle_name = f"{model_name}.json"
        relative = f"models/{bundle_name}"
        if bundle_name not in verified:
            raise ValueError(f"GAIN_RUN_MODEL_REQUIRED_ARTIFACT_MISSING:{model_name}")
        bundle_path = verified[bundle_name]
        bundle = gain_models.load_bundle(bundle_path)
        data_identity = bundle.get("data_identity", {})
        if (
            bundle.get("model_name") != model_name
            or data_identity.get("test_accessed") is not False
            or bundle.get("threshold", {}).get("frozen_before_test") is not True
            or bundle.get("training", {}).get("fit_split") != "train"
        ):
            raise ValueError(f"GAIN_RUN_MODEL_NOT_FROZEN_BEFORE_TEST:{model_name}")
        mapping[key] = bundle_path
    return mapping, {
        "model_directory": str(model_dir), "model_manifest_sha256": _sha256_file(model_manifest_path),
        "training_run_id": str(model_run["run_id"]), "training_evidence_manifest_sha256": model_evidence_sha,
        "model_names": sorted(required_models.values()), "test_accessed_before_evaluation": False,
    }


def _evaluate_closed_loop(
    output: Path, run: dict[str, Any], config: dict[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    import gain_evaluation
    import gain_scenarios
    import gain_models

    _check_resource_account(args)
    model_paths, model_identity = _verify_frozen_model_directory(args.model_dir)
    conditions = list(args.conditions or gain_scenarios.FORMAL_CONDITIONS)
    if not conditions or len(conditions) != len(set(conditions)) or set(conditions) - set(gain_scenarios.FORMAL_CONDITIONS):
        raise ValueError("GAIN_RUN_EVALUATION_CONDITIONS_INVALID")
    per_condition = int(config["scenario"]["formal_instances_per_condition"]["test"])
    if args.limit_instances_per_split is not None:
        if isinstance(args.limit_instances_per_split, bool) or args.limit_instances_per_split < 1:
            raise ValueError("GAIN_RUN_EVALUATION_LIMIT_MUST_BE_POSITIVE")
        per_condition = int(args.limit_instances_per_split)
    scenarios = list(gain_scenarios.iter_gain_scenarios(
        {"train": 0, "validation": 0, "test": per_condition},
        seed=8201, conditions=tuple(conditions),
    ))
    if not scenarios or any(row.get("split") != "test" for row in scenarios):
        raise ValueError("GAIN_RUN_EVALUATION_TEST_SCENARIO_GENERATION_FAILED")
    scenario_path = output / "TEST_SCENARIOS.jsonl.gz"
    with _GzipJsonlWriter(scenario_path) as writer:
        for scenario in scenarios:
            writer.write(scenario)
    scenario_ids = [str(row["parent_id"]) for row in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("GAIN_RUN_EVALUATION_SCENARIO_ID_DUPLICATE")
    scenario_identity = {
        "schema": "gain-gate-frozen-test-scenarios/1", "split": "test",
        "scenario_count": len(scenarios), "scenario_ids_sha256": _sha256_bytes(_canonical(sorted(scenario_ids))),
        "scenario_content_sha256": _sha256_bytes(_canonical(sorted(str(row["scenario_content_sha256"]) for row in scenarios))),
        "event_tape_ids_sha256": _sha256_bytes(_canonical(sorted(str(row["event_tape_id"]) for row in scenarios))),
        "artifact_path": scenario_path.name, "artifact_sha256": _sha256_file(scenario_path),
        "artifact_size_bytes": scenario_path.stat().st_size,
    }
    _write_json(output / "TEST_SCENARIO_IDENTITY.json", scenario_identity)
    runtimes: dict[str, Any] = {}

    def runtime_loader(path: str | Path):
        runtime = gain_models.load_runtime(path)
        runtimes[str(Path(path).resolve())] = runtime
        return runtime

    evaluation_modules = load_evaluation_modules()
    runner_api = {
        "legacy_environment_scenario": legacy_environment_scenario,
        "make_safety_continuation": make_safety_continuation,
        "safety_receipt": safety_receipt, "safety_scope": safety_scope,
        "verify_safety_disposition": verify_safety_disposition,
        "apply_live_safety_disposition": apply_live_safety_disposition,
        "invoke_replanner": invoke_replanner,
        "resource_check": lambda: _check_resource_account(args),
    }
    evaluation = gain_evaluation.run_closed_loop_evaluation(
        scenarios, {key: str(path) for key, path in model_paths.items()}, output,
        config, runtime_loader=runtime_loader,
        evaluation_modules=evaluation_modules, runner_api=runner_api,
    )
    if not isinstance(evaluation, Mapping) or evaluation.get("schema") != "gain-closed-loop-evaluation/1":
        raise ValueError("GAIN_RUN_EVALUATION_REPORT_SCHEMA_INVALID")
    expected_methods = list(gain_evaluation.METHODS)
    if evaluation.get("method_names") != expected_methods:
        raise ValueError("GAIN_RUN_EVALUATION_METHOD_SET_MISMATCH")
    _check_resource_account(args)
    report = {
        "schema": EVALUATION_SCHEMA, "status": "complete", "run_id": run["run_id"],
        "config_sha256": run["config_sha256"], "source_manifest_sha256": run["source_manifest_sha256"],
        "replanner_version": REPLANNER_VERSION, "split": "test",
        "model_identity": model_identity, "test_scenarios": scenario_identity,
        "method_names": expected_methods,
        "evaluation": dict(evaluation), "test_labels_loaded": False,
        "per_condition_instance_count": per_condition, "condition_count": len(conditions),
        "closed_loop": True,
    }
    _write_json(output / "EVALUATION_REPORT.json", report)
    return report


def _collect_formal(
    output: Path, run: dict[str, Any], config: dict[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    modules = _load_smoke_modules()
    iterator, counts, conditions, splits, full_training_scale = _collection_scenarios(
        config, limit_per_split=args.limit_instances_per_split, conditions=args.conditions,
        splits=args.splits,
    )
    max_states = int(config["collection"]["max_states_per_trajectory"])
    if max_states < 1 or max_states > 8:
        raise ValueError("GAIN_RUN_COLLECTION_STATE_CAP_MUST_BE_1_TO_8")
    scenario_dir = output / "scenarios"
    snapshot_dir = output / "snapshots"
    suffix_dir = output / "suffixes"
    label_dir = output / "raw_labels"
    for directory in (scenario_dir, snapshot_dir, suffix_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows_path = output / "COLLECTION_ROWS.jsonl.gz"
    scenarios_path = output / "SCENARIO_OUTCOMES.jsonl.gz"
    rows_writer = _GzipJsonlWriter(rows_path)
    outcomes_writer = _GzipJsonlWriter(scenarios_path)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    group_source_index: dict[tuple[str, str], int] = {}
    suffix_count = 0
    valid_label_count = 0
    unresolved_safety_count = 0
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    max_wall = float(config["collection"]["max_wall_seconds_per_server_job"])
    max_cpu = float(config["collection"]["max_project_cpu_seconds"])
    max_rss = int(config["collection"]["max_rss_bytes"])
    resource_module = __import__("resource_supervisor")
    cost_module = __import__("cost_accounting")
    weights = config["score_weights"]
    cost_config = cost_module.GainCostConfig(
        completion=float(weights["completion"]), delay=float(weights["delay"]),
        energy=float(weights["energy"]), risk=float(weights["risk"]),
        communication=float(weights["communication"]), switch=float(weights["switch"]),
        reconstruction=float(weights["reconstruction_call"]),
        reconstruction_size=float(weights["reconstruction_size"]),
        cpu_second=float(config["labels"]["planning_cpu_cost_weight_per_second"]),
    )
    continuation = _make_safety_continuation(modules, config)
    total_scenarios = 0
    try:
        for scenario in iterator:
            _check_resource_account(args)
            if time.perf_counter() - wall_started > max_wall:
                raise TimeoutError("GAIN_RUN_COLLECT_WALL_LIMIT_EXCEEDED")
            if time.process_time() - cpu_started > max_cpu:
                raise TimeoutError("GAIN_RUN_COLLECT_CPU_LIMIT_EXCEEDED")
            rss = int(resource_module._read_rss_bytes())
            if rss > max_rss:
                raise MemoryError("GAIN_RUN_COLLECT_RSS_LIMIT_EXCEEDED")
            scenario_id = str(scenario["parent_id"])
            scenario_path = scenario_dir / f"{scenario_id}.json.gz"
            _write_gzip_json(scenario_path, scenario)
            split = str(scenario["split"])
            condition = str(scenario["condition"])
            source = _trajectory_source(split, condition, group_source_index)
            selected, trajectory_summary = _scan_collection_trajectory(
                scenario, source, modules, config, max_states,
            )
            total_scenarios += 1
            split_counts[split] += 1
            condition_counts[f"{split}:{condition}"] += 1
            source_counts[source] += 1
            candidate_counts["seen"] += int(trajectory_summary["candidate_event_count"])
            candidate_counts["selected"] += int(trajectory_summary["selected_candidate_count"])
            candidate_counts["forced"] += int(trajectory_summary["forced_candidate_count"])

            scenario_valid = 0
            for candidate in selected:
                state_id = str(candidate["state_id"])
                event_counts[candidate["event_type"]] += 1
                restored_env = modules["pairs"].restore_snapshot(
                    candidate["snapshot_json"], env_factory=modules["env_class"],
                )
                snapshot, suffix_rows, suffix_jsons, suffix_hashes = _state_and_suffix_rows(
                    scenario, restored_env, modules, config, state_id=state_id,
                    event_type=str(candidate["event_type"]), forced=bool(candidate["forced"]),
                )
                snapshot_payload = json.loads(snapshot)
                snapshot_path = snapshot_dir / f"{state_id}.snapshot.json.gz"
                _write_gzip_json(snapshot_path, {"snapshot_json": snapshot})
                suffix_paths = []
                suffix_file_hashes = []
                for suffix_index, suffix_json in enumerate(suffix_jsons):
                    suffix_path = suffix_dir / f"{state_id}.suffix-{suffix_index + 1:02d}.json.gz"
                    _write_gzip_json(suffix_path, {"suffix_json": suffix_json})
                    suffix_paths.append(suffix_path.relative_to(output).as_posix())
                    suffix_file_hashes.append(_sha256_file(suffix_path))
                suffix_count += len(suffix_rows)
                label = modules["pairs"].run_paired_counterfactual(
                    snapshot, replan_action=candidate["replan_action"], suffixes=suffix_jsons,
                    continuation_policy=continuation, replanner_version=REPLANNER_VERSION,
                    replanner_cpu_seconds=float(candidate["planner_cpu_seconds"]),
                    cost_config=cost_config,
                    horizon_minutes=int(config["labels"]["horizon_minutes"]), action_delay_ticks=0,
                )
                oracle = modules["oracle"].validate_gain_label(label)
                label_path = label_dir / f"{state_id}.label.json.gz"
                _write_gzip_json(label_path, {"label": label})
                label_sha = _sha256_file(label_path)
                valid = bool(label.get("valid_mask"))
                valid_label_count += int(valid)
                scenario_valid += int(valid)
                unresolved_safety_count += int(not bool(candidate["safety_disposition"].get("resolved", True)))
                suffix_gains = _validated_suffix_gains(label, oracle)
                row = {
                    "schema": "gain-gate-training-row/1", "state_id": state_id,
                    "scenario_id": scenario_id, "scenario_instance_id": scenario_id,
                    "parent_id": scenario_id, "parent_family": scenario["parent_family"],
                    "condition": condition, "split": split, "trajectory_id": candidate["trajectory_id"],
                    "trajectory_source": source, "scenario_content_sha256": scenario.get("scenario_content_sha256", _scenario_content_sha256(scenario)),
                    "scenario_seed": int(scenario["seed_variant"]),
                    "event_tape_id": scenario["event_tape_id"],
                    "event_tape_sha256": _sha256_bytes(_canonical(scenario["event_tape"])),
                    "time": candidate["time"], "event_type": candidate["event_type"],
                    "forced": candidate["forced"], "detection": candidate["detection"],
                    "safety": candidate["safety"], "safety_disposition": candidate["safety_disposition"],
                    "safety_apply_receipt": candidate.get("safety_apply_receipt"),
                    "public_features": candidate["features"], "features": candidate["features"],
                    "feature_gate_seconds": candidate["feature_gate_seconds"],
                    "planner_cpu_seconds": candidate["planner_cpu_seconds"],
                    "planner_wall_seconds": candidate["planner_wall_seconds"],
                    "candidate_count": candidate["planner_receipt"]["actual_candidate_count"],
                    "candidate_receipt": candidate["planner_receipt"],
                    "replanner_version": REPLANNER_VERSION,
                    "snapshot_sha256": snapshot_payload["snapshot_sha256"],
                    "private_snapshot_sha256": snapshot_payload["private_snapshot_sha256"],
                    "public_history_sha256": snapshot_payload["public_history_sha256"],
                    "snapshot_path": snapshot_path.relative_to(output).as_posix(),
                    "snapshot_file_sha256": _sha256_file(snapshot_path),
                    "suffix_seed": [item["suffix_seed"] for item in suffix_rows],
                    "suffix_hashes": suffix_hashes, "suffix_paths": suffix_paths,
                    "suffix_file_hashes": suffix_file_hashes,
                    "raw_label_path": label_path.relative_to(output).as_posix(),
                    "raw_label_sha256": label_sha, "label_sha256": label["label_sha256"],
                    "oracle": {key: oracle.get(key) for key in (
                        "schema", "valid", "label_valid_mask", "recomputed_gain_mean",
                        "recomputed_paired_variance", "label_sha256",
                    )},
                    "gain_mean": oracle.get("recomputed_gain_mean"),
                    "oracle_gain": oracle.get("recomputed_gain_mean"),
                    "suffix_gains": suffix_gains,
                    "paired_variance": oracle.get("recomputed_paired_variance"),
                    "valid_mask": valid,
                    "unknown_reason": label.get("unknown_reason"),
                    "source_action_allowed": candidate["source_action_allowed"],
                }
                rows_writer.write(row)
            outcomes_writer.write({
                "scenario_id": scenario_id, "split": split, "condition": condition,
                "parent_family": scenario["parent_family"],
                "scenario_content_sha256": scenario.get("scenario_content_sha256", _scenario_content_sha256(scenario)),
                "event_tape_sha256": _sha256_bytes(_canonical(scenario["event_tape"])),
                "scenario_path": scenario_path.relative_to(output).as_posix(),
                "scenario_file_sha256": _sha256_file(scenario_path),
                "candidate_event_count": trajectory_summary["candidate_event_count"],
                "selected_candidate_count": trajectory_summary["selected_candidate_count"],
                "valid_label_count": scenario_valid, **trajectory_summary,
            })
    finally:
        rows_writer.close()
        outcomes_writer.close()

    rows_hash = _sha256_file(rows_path)
    outcomes_hash = _sha256_file(scenarios_path)
    artifacts = {
        "rows": {"path": rows_path.name, "sha256": rows_hash, "size_bytes": rows_path.stat().st_size, "row_count": rows_writer.rows},
        "scenario_outcomes": {"path": scenarios_path.name, "sha256": outcomes_hash, "size_bytes": scenarios_path.stat().st_size, "row_count": outcomes_writer.rows},
    }
    dataset_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    report = {
        "schema": COLLECTION_SCHEMA, "status": "complete", "run_id": run["run_id"],
        "config_sha256": run["config_sha256"], "source_manifest_sha256": run["source_manifest_sha256"],
        "replanner_version": REPLANNER_VERSION, "full_training_scale": full_training_scale,
        "full_formal_scale": False, "test_labels_collected": False,
        "formal_conditions": conditions, "requested_splits": splits,
        "instances_per_condition_by_split": counts,
        "generated_scenario_count": total_scenarios,
        "scenario_count_by_split": dict(sorted(split_counts.items())),
        "scenario_count_by_split_condition": dict(sorted(condition_counts.items())),
        "trajectory_source_count": dict(sorted(source_counts.items())),
        "trajectory_source_policy": "deterministic_round_robin_within_split_and_condition",
        "candidate_count": dict(candidate_counts), "event_type_count": dict(sorted(event_counts.items())),
        "row_count": rows_writer.rows, "valid_label_count": valid_label_count,
        "suffix_measurement_count": suffix_count,
        "unresolved_safety_disposition_count": unresolved_safety_count,
        "max_states_per_trajectory": max_states,
        "reservoir_sampling": "deterministic_streaming_reservoir",
        "raw_labels": "one gzipped full paired label per row; includes raw branch trajectories",
        "artifacts": artifacts, "dataset_bytes": dataset_bytes,
        "cpu_seconds": max(0.0, time.process_time() - cpu_started),
        "wall_seconds": max(0.0, time.perf_counter() - wall_started),
        "peak_rss_bytes": int(resource_module._read_rss_bytes()),
        "resource_limits": {"max_wall_seconds": max_wall, "max_cpu_seconds": max_cpu, "max_rss_bytes": max_rss},
    }
    _write_json(output / "COLLECTION_REPORT.json", report)
    return report


def _smoke(
    output: Path, run: dict[str, Any], config: dict[str, Any], resource_account: _ProjectResourceAccount | None = None,
) -> dict[str, Any]:
    modules = _load_smoke_modules()
    scenarios = modules["scenarios"].build_pilot_scenarios(
        instances_per_parent=int(config["scenario"]["pilot_instances_per_parent"]),
        seed=8201,
    )
    if len(scenarios) != 18:
        raise ValueError(f"GAIN_RUN_PILOT_EXPECTED_18_INSTANCES:{len(scenarios)}")
    state_dir = output / "private_snapshots"
    raw_dir = output / "raw_labels"
    suffix_dir = output / "suffix_snapshots"
    state_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix_dir.mkdir(parents=True, exist_ok=True)
    label_rows_path = output / "SMOKE_ROWS.jsonl"
    suffix_hashes_all: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    event_type_counts: Counter[str] = Counter()
    valid_labels = 0
    unresolved_forced_dispositions = 0
    replay_check: dict[str, Any] | None = None
    tamper_check: dict[str, Any] | None = None
    max_wall = float(config["collection"]["max_wall_seconds_per_server_job"])
    max_rss_bytes = int(config["collection"]["max_rss_bytes"])
    max_cpu_seconds = float(config["collection"]["max_project_cpu_seconds"])
    smoke_started = time.perf_counter()
    cpu_started = time.process_time()
    resource_module = __import__("resource_supervisor")
    peak_rss_bytes = int(resource_module._read_rss_bytes())
    cost_module = __import__("cost_accounting")
    weights = config["score_weights"]
    cost_config = cost_module.GainCostConfig(
        completion=float(weights["completion"]), delay=float(weights["delay"]),
        energy=float(weights["energy"]), risk=float(weights["risk"]),
        communication=float(weights["communication"]), switch=float(weights["switch"]),
        reconstruction=float(weights["reconstruction_call"]),
        reconstruction_size=float(weights["reconstruction_size"]),
        cpu_second=float(config["labels"]["planning_cpu_cost_weight_per_second"]),
    )
    continuation = _make_safety_continuation(modules, config)
    ordinary_probe_done = False
    gate_probe_checks = {
        "ordinary_suppressed_planner_calls": 0, "ordinary_predictor_calls": 0,
        "forced_replanner_calls": 0, "forced_predictor_calls": 0,
    }

    for index, scenario in enumerate(scenarios):
        if resource_account is not None:
            resource_account.check()
        if time.perf_counter() - smoke_started > max_wall:
            raise TimeoutError("GAIN_RUN_SMOKE_WALL_LIMIT_EXCEEDED")
        if time.process_time() - cpu_started > max_cpu_seconds:
            raise TimeoutError("GAIN_RUN_SMOKE_CPU_LIMIT_EXCEEDED")
        peak_rss_bytes = max(peak_rss_bytes, int(resource_module._read_rss_bytes()))
        if peak_rss_bytes > max_rss_bytes:
            raise MemoryError("GAIN_RUN_SMOKE_RSS_LIMIT_EXCEEDED")
        scenario_id = str(scenario["parent_id"])
        state_id = f"{scenario_id}-state-000"
        env, detection, scope = _advance_to_candidate(scenario, modules)
        observation = modules["pairs"].model_public_observation(env)
        safety = _safety_receipt(modules, observation)
        forced = bool(detection.get("strong_event")) or _safety_forced(safety) or not safety["safe_to_suppress"]
        observation, safety_disposition, safety_apply_receipt = _apply_live_safety_disposition(
            modules, env, observation, safety,
        )
        scope = modules["identify_scope"](observation, detection)
        event_type = str(detection.get("event_type") or "none")
        event_type_counts[event_type] += 1
        features_started = time.perf_counter()
        features = modules["features"].extract_public_features(observation, detection, scope)
        feature_seconds = max(0.0, time.perf_counter() - features_started)

        # Check that a suppressed ordinary event never calls the fixed planner.
        if not ordinary_probe_done and not forced:
            calls = {"count": 0, "predictor_count": 0}
            def predictor(_features: Mapping[str, float]) -> dict[str, float]:
                calls["predictor_count"] += 1
                return {"gain_hat": -1.0, "lower_bound": -1.0, "epsilon": 0.0}
            gate_probe = _probe_online_gate(
                modules, env, detection, scope,
                {"needs_forced_handling": False, "safe_to_suppress": True}, predictor,
                lambda: calls.__setitem__("count", calls["count"] + 1),
            )
            if gate_probe["planner_invoked"] or calls["count"] != 0 or calls["predictor_count"] != 1:
                raise ValueError("GAIN_RUN_SUPPRESSED_EVENT_CALLED_REPLANNER")
            gate_probe_checks["ordinary_suppressed_planner_calls"] = calls["count"]
            gate_probe_checks["ordinary_predictor_calls"] = calls["predictor_count"]
            ordinary_probe_done = True

        if forced and gate_probe_checks["forced_replanner_calls"] == 0:
            calls = {"planner_count": 0, "predictor_count": 0}

            def forced_predictor(_features: Mapping[str, float]) -> dict[str, float]:
                calls["predictor_count"] += 1
                return {"gain_hat": -1.0, "lower_bound": -1.0, "epsilon": 0.0}

            forced_probe = _probe_online_gate(
                modules, env, detection, scope, safety, forced_predictor,
                lambda: calls.__setitem__("planner_count", calls["planner_count"] + 1),
            )
            if (
                not forced_probe["decision"]["allow_replan"]
                or not forced_probe["decision"]["forced"]
                or not forced_probe["planner_invoked"]
                or calls["planner_count"] != 1
                or calls["predictor_count"] != 0
            ):
                raise ValueError("GAIN_RUN_FORCED_EVENT_DID_NOT_BYPASS_GAIN_GATE")
            gate_probe_checks["forced_replanner_calls"] = calls["planner_count"]
            gate_probe_checks["forced_predictor_calls"] = calls["predictor_count"]

        # Public features must not change when an unarrived task's private position changes.
        clone_scenario = _hidden_mutation(scenario, int(env.time))
        clone = _advance_clone_to_time(clone_scenario, int(env.time), modules)
        clone_observation = modules["pairs"].model_public_observation(clone)
        clone_safety = _safety_receipt(modules, clone_observation)
        clone_observation, _, _ = _apply_live_safety_disposition(
            modules, clone, clone_observation, clone_safety,
        )
        clone_history = modules["pairs"]._public_history(clone)
        clone_detection = modules["trigger_class"]({}, {"threshold": 0.5, "delta_min": 4}).detect(
            clone_history[-2] if len(clone_history) >= 2 else None, clone_observation,
            history=clone_history[-4:],
        )
        clone_scope = modules["identify_scope"](clone_observation, clone_detection)
        clone_features = modules["features"].extract_public_features(clone_observation, clone_detection, clone_scope)
        if features != clone_features:
            raise ValueError("GAIN_RUN_HIDDEN_DATA_CHANGED_PUBLIC_FEATURES")
        score = sum(features.values()) * 1e-6
        decision_a = modules["gate"].decide_gain(score, 0.0, score, gate_mode="calibrated_lower_bound")
        decision_b = modules["gate"].decide_gain(sum(clone_features.values()) * 1e-6, 0.0, score, gate_mode="calibrated_lower_bound")
        if decision_a != decision_b:
            raise ValueError("GAIN_RUN_HIDDEN_DATA_CHANGED_GATE_OUTPUT")

        planner_action = copy.deepcopy(modules["pairs"].NOOP_ACTION)
        planner_receipt = None
        planner_cpu = 0.0
        planner_wall = 0.0
        unknown_reason = None
        if forced:
            planner_action, planner_receipt, planner_cpu, planner_wall = _invoke_replanner(
                modules, env, _safety_scope(safety, scope),
                max_candidates=int(config["replanner"]["max_candidates"]),
            )
            if not safety_disposition["resolved"]:
                unresolved_forced_dispositions += 1
                unknown_reason = str(safety_disposition.get("unknown_reason") or "forced_safety_disposition_unresolved")
        else:
            planner_action, planner_receipt, planner_cpu, planner_wall = _invoke_replanner(
                modules, env, scope, max_candidates=int(config["replanner"]["max_candidates"]),
            )

        snapshot, suffix_rows, suffix_jsons, suffix_hashes = _state_and_suffix_rows(
            scenario, env, modules, config, state_id=state_id,
            event_type=str(detection.get("event_type", scenario.get("condition", "unknown"))), forced=forced,
        )
        source_payload = json.loads(snapshot)
        snapshot_path = state_dir / f"{state_id}.snapshot.json"
        if snapshot_path.exists():
            raise FileExistsError(f"GAIN_RUN_REFUSE_SNAPSHOT_OVERWRITE:{snapshot_path.name}")
        snapshot_path.write_text(snapshot, encoding="utf-8", newline="\n")
        suffix_paths = []
        for suffix_index, suffix_json in enumerate(suffix_jsons):
            suffix_path = suffix_dir / f"{state_id}.suffix-{suffix_index + 1:02d}.json"
            if suffix_path.exists():
                raise FileExistsError(f"GAIN_RUN_REFUSE_SUFFIX_OVERWRITE:{suffix_path.name}")
            suffix_path.write_text(suffix_json, encoding="utf-8", newline="\n")
            suffix_paths.append(suffix_path.relative_to(output).as_posix())
        suffix_hashes_all.extend(_sha256_bytes(_canonical(row["future_events"])) for row in suffix_rows)

        label = modules["pairs"].run_paired_counterfactual(
            snapshot, replan_action=planner_action, suffixes=suffix_jsons,
            continuation_policy=continuation, replanner_version=REPLANNER_VERSION,
            replanner_cpu_seconds=planner_cpu, cost_config=cost_config,
            horizon_minutes=int(config["labels"]["horizon_minutes"]), action_delay_ticks=0,
        )
        oracle_receipt = modules["oracle"].validate_gain_label(label)
        suffix_gains = _validated_suffix_gains(label, oracle_receipt)
        label_path = raw_dir / f"{state_id}.label.json"
        modules["pairs"].persist_label(label_path, label)
        label_hash = _sha256_file(label_path)
        if label.get("valid_mask"):
            valid_labels += 1
            if tamper_check is None:
                tampered = copy.deepcopy(label)
                tampered["gain_mean"] = float(tampered["gain_mean"]) + 1.0
                tampered["label_sha256"] = _sha256_bytes(_canonical({key: value for key, value in tampered.items() if key != "label_sha256"}))
                tamper_rejected = False
                try:
                    modules["oracle"].validate_gain_label(tampered)
                except (ValueError, TypeError):
                    tamper_rejected = True
                if not tamper_rejected:
                    raise ValueError("GAIN_RUN_TAMPERED_LABEL_ACCEPTED_BY_ORACLE")
                tamper_check = {"tampered_label_rejected": True, "original_label_sha256": label.get("label_sha256")}
            if replay_check is None:
                first_projection = {
                    "keep": _physical_trace_projection(label["pairs"][0]["keep"]["trace"]),
                    "replan": _physical_trace_projection(label["pairs"][0]["replan"]["trace"]),
                }
                replay = modules["pairs"].run_paired_counterfactual(
                    snapshot, replan_action=planner_action, suffixes=[suffix_jsons[0]],
                    continuation_policy=continuation, replanner_version=REPLANNER_VERSION,
                    replanner_cpu_seconds=planner_cpu, cost_config=cost_config,
                    horizon_minutes=int(config["labels"]["horizon_minutes"]), action_delay_ticks=0,
                )
                replay_projection = {
                    "keep": _physical_trace_projection(replay["pairs"][0]["keep"]["trace"]),
                    "replan": _physical_trace_projection(replay["pairs"][0]["replan"]["trace"]),
                }
                first_hash = _sha256_bytes(_canonical(first_projection))
                replay_hash = _sha256_bytes(_canonical(replay_projection))
                if first_hash != replay_hash:
                    raise ValueError("GAIN_RUN_PHYSICAL_REPLAY_NOT_DETERMINISTIC")
                replay_check = {"deterministic": True, "physical_trace_sha256": first_hash, "replay_sha256": replay_hash}

        row = {
            "schema": "gain-gate-smoke-row/1", "state_id": state_id,
            "scenario_id": scenario_id, "parent_family": scenario["parent_family"],
            "split": scenario["split"], "condition": scenario["condition"],
            "scenario_content_sha256": _scenario_content_sha256(scenario),
            "scenario_seed": scenario.get("scenario_seed", scenario.get("instance_seed")),
            "event_tape_sha256": _sha256_bytes(_canonical(scenario.get("event_tape", []))),
            "time": int(env.time), "event_type": event_type,
            "forced": forced, "detection": detection, "safety": safety,
            "safety_disposition": safety_disposition, "safety_apply_receipt": safety_apply_receipt,
            "unknown_reason": unknown_reason,
            "features": features, "feature_wall_seconds": feature_seconds,
            "replanner_version": REPLANNER_VERSION,
            "candidate_count": planner_receipt.get("actual_candidate_count") if planner_receipt else 0,
            "candidate_receipt": planner_receipt, "replanner_cpu_seconds": planner_cpu,
            "replanner_wall_seconds": planner_wall,
            "snapshot_sha256": source_payload.get("snapshot_sha256"),
            "private_snapshot_sha256": source_payload.get("private_snapshot_sha256"),
            "public_history_sha256": source_payload.get("public_history_sha256"),
            "suffix_seed": [item["suffix_seed"] for item in suffix_rows],
            "suffix_hashes": suffix_hashes, "suffix_paths": suffix_paths,
            "label_path": label_path.relative_to(output).as_posix(),
            "label_file_sha256": label_hash, "label_sha256": label.get("label_sha256"),
            "gain_mean": oracle_receipt.get("recomputed_gain_mean"),
            "oracle_gain": oracle_receipt.get("recomputed_gain_mean"),
            "suffix_gains": suffix_gains,
            "paired_variance": oracle_receipt.get("recomputed_paired_variance"),
            "valid_mask": label.get("valid_mask"), "unknown_reason": label.get("unknown_reason"),
            "oracle": oracle_receipt,
        }
        _append_jsonl(label_rows_path, row)
        summary_rows.append({
            "state_id": state_id, "scenario_id": scenario_id,
            "parent_family": scenario["parent_family"], "condition": scenario["condition"],
            "forced": forced, "valid_mask": label.get("valid_mask"),
            "event_type": event_type,
            "safety_disposition_status": safety_disposition.get("status"),
            "unknown_reason": unknown_reason,
            "gain_mean": oracle_receipt.get("recomputed_gain_mean"),
            "paired_variance": oracle_receipt.get("recomputed_paired_variance"),
            "feature_wall_seconds": feature_seconds, "replanner_cpu_seconds": planner_cpu,
            "replanner_wall_seconds": planner_wall, "candidate_count": row["candidate_count"],
            "snapshot_sha256": row["snapshot_sha256"], "label_file_sha256": label_hash,
        })
        peak_rss_bytes = max(peak_rss_bytes, int(resource_module._read_rss_bytes()))
        if peak_rss_bytes > max_rss_bytes:
            raise MemoryError("GAIN_RUN_SMOKE_RSS_LIMIT_EXCEEDED")
        if time.process_time() - cpu_started > max_cpu_seconds:
            raise TimeoutError("GAIN_RUN_SMOKE_CPU_LIMIT_EXCEEDED")

    unique_suffixes = len(set(suffix_hashes_all))
    if unique_suffixes < 2:
        raise ValueError("GAIN_RUN_PILOT_FUTURE_SUFFIXES_DID_NOT_VARY")
    if valid_labels < 1 or replay_check is None or tamper_check is None:
        raise ValueError("GAIN_RUN_SMOKE_VALID_LABEL_OR_AUDIT_CHECK_MISSING")
    if gate_probe_checks["ordinary_predictor_calls"] != 1 or gate_probe_checks["forced_replanner_calls"] != 1:
        raise ValueError("GAIN_RUN_SMOKE_ORDINARY_AND_FORCED_GATE_CHECKS_REQUIRED")
    wall_seconds = max(0.0, time.perf_counter() - smoke_started)
    cpu_seconds = max(0.0, time.process_time() - cpu_started)
    report = {
        "schema": "gain-gate-p1-smoke-report/1", "status": "complete",
        "run_id": run["run_id"], "config_sha256": run["config_sha256"],
        "source_manifest_sha256": run["source_manifest_sha256"],
        "pilot_instance_count": len(scenarios), "label_count": len(summary_rows),
        "valid_label_count": valid_labels,
        "forced_label_count": sum(bool(row["forced"]) for row in summary_rows),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "zero_event_instance_count": event_type_counts.get("none", 0),
        "zero_event_instance_rate": event_type_counts.get("none", 0) / len(summary_rows),
        "unresolved_forced_disposition_count": unresolved_forced_dispositions,
        "engineering_issues": (
            ["forced_safety_disposition_unresolved"] if unresolved_forced_dispositions else []
        ),
        "suffix_count": len(suffix_hashes_all), "unique_suffix_count": unique_suffixes,
        "suffixes_varied_across_pilot": True,
        "replanner_version": REPLANNER_VERSION,
        "candidate_count_upper_bound": int(config["replanner"]["max_candidates"]),
        "action_delay_ticks": 0,
        "checks": {
            "public_features_invariant_to_hidden_future_data": True,
            "suppressed_ordinary_event_has_zero_planner_calls": gate_probe_checks["ordinary_suppressed_planner_calls"] == 0,
            "ordinary_event_predictor_called_once": gate_probe_checks["ordinary_predictor_calls"] == 1,
            "forced_event_bypasses_predictor_and_calls_replanner": (
                gate_probe_checks["forced_predictor_calls"] == 0
                and gate_probe_checks["forced_replanner_calls"] == 1
            ),
            "all_forced_dispositions_resolved": unresolved_forced_dispositions == 0,
            "physical_replay_deterministic": replay_check["deterministic"],
            "label_tamper_rejected_by_independent_oracle": tamper_check["tampered_label_rejected"],
            "all_labels_independently_recomputed": True,
            "safe_continuation_used_by_both_arms": True,
            "full_candidate_generation_only_inside_replanner": True,
        },
        "gate_probe_checks": gate_probe_checks, "replay_check": replay_check,
        "tamper_check": tamper_check,
        "feature_wall_seconds_mean": sum(row["feature_wall_seconds"] for row in summary_rows) / len(summary_rows),
        "planner_cpu_seconds_mean": sum(row["replanner_cpu_seconds"] for row in summary_rows) / len(summary_rows),
        "planner_wall_seconds_mean": sum(row["replanner_wall_seconds"] for row in summary_rows) / len(summary_rows),
        "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "resource_measurement": "process_high_water_mark_sampled_before_and_after_each_scenario",
        "resource_limits": {
            "max_wall_seconds_per_job": max_wall,
            "max_rss_bytes": max_rss_bytes,
            "max_project_cpu_seconds": max_cpu_seconds,
        },
        "scenario_rows": summary_rows,
    }
    _write_json(output / "P1_SMOKE_REPORT.json", report)
    return report


def _review_path(value: str | None) -> Path:
    if value:
        return _resolve_input(value)
    return (PROJECT_ROOT / "reviews" / "P1_REVIEW.json").resolve(strict=True)


def _verify_p1_review(review_path: Path, config_sha256: str) -> dict[str, Any]:
    review = _read_json(review_path)
    if review.get("schema") != REVIEW_SCHEMA or review.get("decision") != "PASS":
        raise ValueError("GAIN_RUN_P1_REVIEW_NOT_PASS")
    evidence_path_value = review.get("evidence_manifest_path")
    if not isinstance(evidence_path_value, str) or not evidence_path_value:
        raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_PATH_MISSING")
    evidence_path = _resolve_input(evidence_path_value)
    run_dir = evidence_path.parent
    run = _read_json(run_dir / "RUN.json")
    source_manifest_path = run_dir / str(run.get("source_manifest_path", "SOURCE_MANIFEST.json"))
    if not source_manifest_path.is_file() or not evidence_path.is_file():
        raise ValueError("GAIN_RUN_P1_REVIEW_BINDING_FILES_MISSING")
    current_source = _source_manifest()
    current_source_sha = _sha256_bytes(_serialized_json(current_source))
    source_sha = _sha256_file(source_manifest_path)
    evidence_sha = _sha256_file(evidence_path)
    expected_evidence_path = (run_dir / "EVIDENCE_MANIFEST.json").resolve()
    if evidence_path.resolve() != expected_evidence_path:
        raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_PATH_MISMATCH")
    if run.get("stage") != "smoke" or run.get("run_id") != review.get("smoke_run_id"):
        raise ValueError("GAIN_RUN_P1_REVIEW_SMOKE_RUN_MISMATCH")
    if run.get("config_sha256") != config_sha256 or review.get("config_sha256") != config_sha256:
        raise ValueError("GAIN_RUN_P1_REVIEW_CONFIG_HASH_MISMATCH")
    if (
        run.get("source_manifest_sha256") != source_sha
        or review.get("source_manifest_sha256") != source_sha
        or source_sha != current_source_sha
    ):
        raise ValueError("GAIN_RUN_P1_REVIEW_SOURCE_HASH_MISMATCH")
    if review.get("evidence_manifest_sha256") != evidence_sha:
        raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_HASH_MISMATCH")
    manifest = _read_json(evidence_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("run_id") != run["run_id"]
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("source_manifest_sha256") != source_sha
    ):
        raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_MANIFEST_INVALID")
    listed_files = manifest.get("files")
    if not isinstance(listed_files, list):
        raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_FILES_INVALID")
    for entry in listed_files:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_FILES_INVALID")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_PATH_UNSAFE")
        target = (run_dir / relative).resolve()
        if run_dir.resolve() not in target.parents or not target.is_file():
            raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_FILE_MISSING")
        if target.stat().st_size != entry.get("size_bytes") or _sha256_file(target) != entry.get("sha256"):
            raise ValueError("GAIN_RUN_P1_REVIEW_EVIDENCE_FILE_HASH_MISMATCH")
    if not isinstance(review.get("reviewer_model"), str) or not review["reviewer_model"].strip():
        raise ValueError("GAIN_RUN_P1_REVIEW_MODEL_NOT_RECORDED")
    scope = review.get("scope")
    if not isinstance(scope, list) or not {"source", "smoke", "pairing", "oracle", "leakage", "timing"} <= set(scope):
        raise ValueError("GAIN_RUN_P1_REVIEW_SCOPE_INCOMPLETE")
    return {
        "schema": "gain-gate-p1-review-verification/1", "decision": "PASS",
        "smoke_run_id": run["run_id"], "config_sha256": config_sha256,
        "source_manifest_sha256": source_sha, "evidence_manifest_sha256": evidence_sha,
        "reviewer_model": review["reviewer_model"],
    }


def _v2_smoke_review_path(value: str | None) -> Path:
    if value:
        return _resolve_input(value)
    return (PROJECT_ROOT / "reviews" / "V2_SMOKE_REVIEW_20261017_R3.json").resolve(strict=True)


def _positive_claim(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {
            "", "0", "false", "no", "none", "null", "never", "disabled",
            "not_run", "not_authorized", "not_determined", "not_accessed",
        }
    return bool(value)


def _reject_forbidden_smoke_claims(value: Any, location: str = "review") -> None:
    """Reject metadata that turns the pilot gate into a test/model/OOD claim."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = "".join(char for char in str(raw_key).lower() if char.isalnum())
            forbidden = (
                ("test" in key and "label" in key)
                or ("ood" in key and any(token in key for token in ("access", "scenario", "label", "used")))
                or ("model" in key and any(token in key for token in ("artifact", "train", "access", "load", "checkpoint")))
                or ("training" in key and any(token in key for token in ("access", "started", "completed", "used")))
            )
            if forbidden and _positive_claim(child):
                raise ValueError(f"GAIN_RUN_V2_SMOKE_FORBIDDEN_CLAIM:{location}.{raw_key}")
            _reject_forbidden_smoke_claims(child, f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_smoke_claims(child, f"{location}[{index}]")


def _verify_v2_smoke_review(review_path: Path, config_sha256: str) -> dict[str, Any]:
    """Verify the coordinator's immutable V2 pilot gate before formal collection."""
    review_path = review_path.resolve(strict=True)
    review = _read_json(review_path)
    if review.get("schema") != V2_SMOKE_REVIEW_SCHEMA:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_SCHEMA_INVALID")
    if review.get("decision") != "PASS_TO_FORMAL_COLLECTION_REVIEW_REQUIRED":
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_DECISION_INVALID")
    if review.get("engineering") != "PASS_TO_FORMAL_REVIEW":
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_ENGINEERING_INVALID")
    if review.get("config_sha256") != V2_SMOKE_CONFIG_SHA256 or config_sha256 != V2_SMOKE_CONFIG_SHA256:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_CONFIG_HASH_MISMATCH")
    if review.get("run_id") != V2_SMOKE_RUN_ID:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_RUN_ID_MISMATCH")
    if review.get("source_manifest_sha256") != V2_SMOKE_SOURCE_MANIFEST_SHA256:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_SOURCE_HASH_MISMATCH")
    if review.get("evidence_manifest_sha256") != V2_SMOKE_EVIDENCE_MANIFEST_SHA256:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_HASH_MISMATCH")
    if review.get("method") not in {"NOT_DETERMINED", "NOT_AUTHORIZED"}:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_METHOD_SCOPE_INVALID")
    if review.get("world_model") != "NOT_AUTHORIZED" or review.get("ppo") != "NOT_AUTHORIZED":
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_UNAUTHORIZED_SCOPE")
    actual_model = review.get("actual_reviewer_model")
    if not isinstance(actual_model, str) or not actual_model.strip():
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_MODEL_NOT_RECORDED")
    model_text = actual_model.lower()
    if "gpt-6.1 sol high" in model_text and "not" not in model_text:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_MODEL_MISREPRESENTED")
    _reject_forbidden_smoke_claims(review)

    checks = review.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_CHECKS_MISSING")
    required_checks = {
        "historical_unresolved_replay_rows": 221,
        "historical_replay_resolved": 221,
        "historical_replay_unresolved": 0,
        "historical_replay_unknown": 0,
        "evidence_files": V2_SMOKE_EVIDENCE_FILE_COUNT,
    }
    for key, expected in required_checks.items():
        if checks.get(key) != expected:
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_CHECK_FAILED:{key}")
    required_true_checks = {
        "all_forced_dispositions_resolved",
        "independent_oracle",
        "leakage_probe",
        "planner_isolation",
        "replay",
    }
    for key in required_true_checks:
        if checks.get(key) is not True:
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_CHECK_FAILED:{key}")

    run_dir = (PROJECT_ROOT / "outputs" / "v2-smoke-final-20261017").resolve(strict=True)
    evidence_path = run_dir / "EVIDENCE_MANIFEST.json"
    run_path = run_dir / "RUN.json"
    source_path = run_dir / "SOURCE_MANIFEST.json"
    for path in (run_path, source_path, evidence_path):
        if not path.is_file():
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_FILE_MISSING:{path.name}")
    run = _read_json(run_path)
    if (
        run.get("schema") != RUN_SCHEMA or run.get("stage") != "smoke" or run.get("status") != "complete"
        or run.get("run_id") != V2_SMOKE_RUN_ID
        or run.get("config_sha256") != V2_SMOKE_CONFIG_SHA256
        or run.get("source_manifest_path") != "SOURCE_MANIFEST.json"
        or run.get("source_manifest_sha256") != V2_SMOKE_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_RUN_BINDING_INVALID")
    source_sha = _sha256_file(source_path)
    if source_sha != V2_SMOKE_SOURCE_MANIFEST_SHA256:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_SOURCE_FILE_HASH_MISMATCH")
    source = _read_json(source_path)
    source_files = source.get("files")
    if source.get("schema") != SOURCE_SCHEMA or not isinstance(source_files, list) or not source_files:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_SOURCE_MANIFEST_INVALID")
    for entry in source_files:
        if (
            not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
            or Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_SOURCE_ENTRY_INVALID")
    static_inputs = source.get("static_inputs")
    if not isinstance(static_inputs, list) or source.get("static_inputs_sha256") != _sha256_bytes(_canonical(static_inputs)):
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_STATIC_INPUTS_INVALID")

    evidence_sha = _sha256_file(evidence_path)
    if evidence_sha != V2_SMOKE_EVIDENCE_MANIFEST_SHA256:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_FILE_HASH_MISMATCH")
    evidence = _read_json(evidence_path)
    entries = evidence.get("files")
    if (
        evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("run_id") != V2_SMOKE_RUN_ID
        or evidence.get("stage") != "smoke" or evidence.get("status") != "complete"
        or evidence.get("config_sha256") != V2_SMOKE_CONFIG_SHA256
        or evidence.get("source_manifest_sha256") != V2_SMOKE_SOURCE_MANIFEST_SHA256
        or not isinstance(entries, list) or len(entries) != V2_SMOKE_EVIDENCE_FILE_COUNT
    ):
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_MANIFEST_INVALID")

    replay_report_value = checks.get("historical_replay_report_path")
    replay_report_sha = checks.get("historical_replay_report_sha256")
    if not isinstance(replay_report_value, str) or not replay_report_value:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_REPLAY_REPORT_PATH_MISSING")
    replay_report = (PROJECT_ROOT / replay_report_value).resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in replay_report.parents or not replay_report.is_file():
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_REPLAY_REPORT_PATH_INVALID")
    if not isinstance(replay_report_sha, str) or len(replay_report_sha) != 64:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_REPLAY_REPORT_HASH_MISSING")
    if _sha256_file(replay_report) != replay_report_sha:
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_REPLAY_REPORT_HASH_MISMATCH")
    verified_paths: dict[str, Path] = {}
    for entry in entries:
        if (
            not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
        ):
            raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_ENTRY_INVALID")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in verified_paths:
            raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_PATH_INVALID")
        target = (run_dir / relative).resolve()
        if run_dir not in target.parents or not target.is_file():
            raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_FILE_MISSING")
        if target.stat().st_size != entry["size_bytes"] or _sha256_file(target) != entry["sha256"]:
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_FILE_HASH_MISMATCH:{relative.as_posix()}")
        verified_paths[relative.as_posix()] = target

    required_artifacts = {
        "RUN.json": run_path,
        "RESOLVED_CONFIG.json": run_dir / "RESOLVED_CONFIG.json",
        "SOURCE_MANIFEST.json": source_path,
        "EVIDENCE_MANIFEST.json": evidence_path,
        "STAGE_RESULT.json": run_dir / "STAGE_RESULT.json",
        "RESOURCE_RECEIPT.json": run_dir / "RESOURCE_RECEIPT.json",
    }
    review_hashes = review.get("hashes")
    if not isinstance(review_hashes, Mapping):
        raise ValueError("GAIN_RUN_V2_SMOKE_REVIEW_HASHES_MISSING")
    for name, target in required_artifacts.items():
        if not target.is_file() or (name != "EVIDENCE_MANIFEST.json" and name not in verified_paths):
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_REQUIRED_ARTIFACT_MISSING:{name}")
        actual = _sha256_file(target)
        if review_hashes.get(name) != actual or (name != "EVIDENCE_MANIFEST.json" and verified_paths[name] != target):
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_ARTIFACT_HASH_MISMATCH:{name}")

    # Read only JSON evidence metadata for scope claims; all listed files are still hash-verified above.
    for relative, target in verified_paths.items():
        if target.suffix.lower() != ".json":
            continue
        try:
            _reject_forbidden_smoke_claims(_read_json(target), f"evidence.{relative}")
        except json.JSONDecodeError:
            raise ValueError(f"GAIN_RUN_V2_SMOKE_REVIEW_EVIDENCE_JSON_INVALID:{relative}")

    try:
        review_ref = review_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        review_ref = review_path.name
    return {
        "schema": V2_SMOKE_REVIEW_VERIFICATION_SCHEMA,
        "decision": "PASS",
        "review_sha256": _sha256_file(review_path),
        "review_path": review_ref,
        "smoke_run_id": V2_SMOKE_RUN_ID,
        "config_sha256": V2_SMOKE_CONFIG_SHA256,
        "source_manifest_sha256": source_sha,
        "evidence_manifest_sha256": evidence_sha,
        "evidence_file_count": len(verified_paths),
        "historical_replay_resolved": checks["historical_replay_resolved"],
        "historical_replay_unresolved": checks["historical_replay_unresolved"],
    }


def _verify_smoke_review_auto(review_path: Path, config_sha256: str) -> dict[str, Any]:
    """Dispatch review verification by schema while retaining legacy P1 validation."""
    review = _read_json(review_path)
    if review.get("schema") == V2_SMOKE_REVIEW_SCHEMA:
        return _verify_v2_smoke_review(review_path, config_sha256)
    return _verify_p1_review(review_path, config_sha256)


def _dispatch(stage: str, output: Path, run: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if stage == "audit":
        result = {"schema": "gain-gate-audit-report/1", "status": "complete", "checks": _audit_checks(config, _source_manifest())}
        _write_json(output / "AUDIT_REPORT.json", result)
        return result
    if stage == "smoke":
        _audit_checks(config, _source_manifest())
        return _smoke(output, run, config, getattr(args, "resource_account", None))
    if stage == "validate":
        review_path = _v2_smoke_review_path(getattr(args, "v2_smoke_review", None) or getattr(args, "p1_review", None))
        report = _verify_smoke_review_auto(review_path, run["config_sha256"])
        result = {"schema": "gain-gate-validation-report/1", "status": "complete", "review": report}
        _write_json(output / "VALIDATION_REPORT.json", result)
        return result
    if stage in {"collect", "train-light", "evaluate", "train-event", "ppo-transfer", "analyze"}:
        review_path = _v2_smoke_review_path(getattr(args, "v2_smoke_review", None) or getattr(args, "p1_review", None))
        # Formal collection and every dependent stage require the V2 smoke gate;
        # a legacy P1 review must never bypass this entry check.
        _verify_v2_smoke_review(review_path, run["config_sha256"])
        if stage == "collect":
            _audit_checks(config, _source_manifest())
            return _collect_formal(output, run, config, args)
        if stage == "train-light":
            return _train_light(output, run, config, args)
        if stage == "evaluate":
            return _evaluate_closed_loop(output, run, config, args)
        raise RuntimeError(f"GAIN_RUN_STAGE_NOT_IMPLEMENTED:{stage}:P1_APPROVAL_VERIFIED_BUT_STAGE_NOT_WIRED")
    raise ValueError(f"GAIN_RUN_UNKNOWN_STAGE:{stage}")


def run(args: argparse.Namespace) -> int:
    config_path, config, config_sha256 = _load_config(args.config)
    output, run_id = _create_run_dir(args.output)
    resource_account = _ProjectResourceAccount(output, run_id, args.stage, config)
    args.resource_account = resource_account
    run_record: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        resource_account.__enter__()
        resource_account.check()
        run_record = _write_initial_manifests(output, run_id, args.stage, config_path, config, config_sha256)
        runtime = _runtime_receipt(args.stage, enforce=args.stage not in {"audit", "validate"})
        resource_account.set_runtime(runtime)
        run_record["runtime"] = runtime
        run_record["device"] = "cpu"
        run_record["cpu_affinity_count"] = len(runtime.get("cpu_affinity") or [])
        run_record["torch_threads"] = runtime.get("torch", {}).get("threads")
        run_record["blas_threadpools"] = runtime.get("blas_threadpools", {})
        run_record["thread_settings_status"] = "measured"
        _write_json(output / "RUN.json", run_record)
        result = _dispatch(args.stage, output, run_record, config, args)
        _write_json(output / "STAGE_RESULT.json", result)
        resource_receipt = resource_account.finish(status="complete")
        if resource_receipt.get("status") != "complete":
            raise TimeoutError("GAIN_RUN_STAGE_RESOURCE_LIMIT_REACHED")
        receipt = _write_evidence_manifest(output, run_record, "complete")
        print(json.dumps({"status": "complete", "run_id": run_id, "output": str(output), "evidence_manifest": receipt}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        if resource_account.locked:
            try:
                resource_account.finish(status="failed", error=exc)
            except BaseException as resource_exc:
                failure_receipt = {
                    "schema": "gain-stage-resource-receipt/1", "run_id": run_id,
                    "stage": args.stage, "status": "failed", "measurement_status": "partial",
                    "error": {"type": type(resource_exc).__name__, "message": str(resource_exc)},
                    "runtime_receipt": runtime,
                }
                _write_json(output / "RESOURCE_RECEIPT.json", failure_receipt)
        if run_record is None:
            run_record = {
                "schema": RUN_SCHEMA, "run_id": run_id, "stage": args.stage,
                "status": "failed", "config_path": config_path.name,
                "config_sha256": config_sha256, "source_manifest_path": "SOURCE_MANIFEST.json",
                "source_manifest_sha256": None, "device": "cpu",
                "thread_settings_status": "runtime_measurement_failed",
            }
        failure = {
            "schema": "gain-gate-run-failure/1", "run_id": run_id,
            "stage": args.stage, "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output / "FAILURE.json", failure)
        receipt = _write_evidence_manifest(output, run_record, "failed")
        print(json.dumps({"status": "failed", "run_id": run_id, "output": str(output), "error": str(exc), "evidence_manifest": receipt}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        resource_account.__exit__(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="New output directory; it must not already exist")
    parser.add_argument("--p1-review", help="Coordinator P1 review JSON; required for protected stages")
    parser.add_argument("--v2-smoke-review", help="Coordinator V2 smoke review JSON; required for formal collection and dependent stages")
    parser.add_argument("--input", nargs="+", help="Collection run directories, a collection-input manifest, or evaluation inputs")
    parser.add_argument("--model-dir", help="Frozen model artifact directory for evaluate")
    parser.add_argument("--limit-instances-per-split", type=int,
                        help="Explicit pilot cap per condition and split; omitted means frozen full counts")
    parser.add_argument("--conditions", nargs="+", help="Subset of frozen formal conditions for a bounded pilot")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"),
                        help="Collection splits; test labels are generated only after model freeze")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
