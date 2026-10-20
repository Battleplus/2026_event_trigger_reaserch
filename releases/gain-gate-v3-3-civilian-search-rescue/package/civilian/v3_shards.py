"""Atomic, verifiable scenario shards for the V3 gain-gate collection."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "GAIN_GATE_CONFIG.json"
BASELINE_PATH = PROJECT_ROOT / "BASELINE_PROVENANCE.json"
CONDITIONS = (
    "none_or_light", "task_update_or_arrival", "platform_failure_or_capability_loss",
    "energy_anomaly", "communication_degradation", "environment_or_obstacle_change", "compound",
)
SPLITS = ("pilot", "train", "validation")
TASK_MODES = ("search_only", "search_rescue")
TARGET_DENSITIES = (1, 3, 6)
SHARD_SCHEMA = "gain-gate-v33-scenario-shard/1"
MARKER_SCHEMA = "gain-gate-v33-scenario-commit/1"
RUN_RECEIPT_SCHEMA = "gain-gate-v33-run-receipt/1"


def summarize_gain_states(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize validated V3 labels once per independent state (stdlib only)."""
    states: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = raw.get("training_row") if isinstance(raw, Mapping) and isinstance(raw.get("training_row"), Mapping) else raw
        if not isinstance(row, Mapping):
            continue
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            continue
        label = row.get("label") if isinstance(row.get("label"), Mapping) else {}
        receipt = row.get("oracle_receipt") if isinstance(row.get("oracle_receipt"), Mapping) else {}
        gain = label.get("gain_mean", receipt.get("recomputed_gain_mean"))
        gain_valid = isinstance(gain, (int, float)) and not isinstance(gain, bool)
        unknown_reason = label.get("unknown_reason")
        state = {
            "state_id": state_id,
            "split": str(row.get("split", "unknown")),
            "scenario_id": str(row.get("scenario_id", row.get("parent_id", "unknown"))),
            "forced": row.get("forced") is True,
            "empty": unknown_reason == "empty_obligation_window",
            "valid": (
                label.get("valid_mask") is True
                and receipt.get("valid") is True
                and receipt.get("label_valid_mask") is True
                and gain_valid
            ),
            "unknown_reason": unknown_reason,
            "oracle_structurally_valid": receipt.get("valid") is True,
            "gain": gain,
        }
        if state["unknown_reason"] is None and state["forced"]:
            state["unknown_reason"] = "forced_state"
        elif state["unknown_reason"] is None and not state["valid"] and not state["empty"]:
            state["unknown_reason"] = "invalid_or_missing_gain"
        previous = states.get(state_id)
        if previous is not None and previous != state:
            raise ValueError(f"V3_SUMMARY_CONFLICTING_STATE:{state_id}")
        states[state_id] = state
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({state["split"] for state in states.values()}):
        subset = [state for state in states.values() if state["split"] == split]
        forced = [state for state in subset if state["forced"]]
        empty = [state for state in subset if state["empty"] and not state["forced"]]
        unknown = [state for state in subset if not state["forced"] and not state["empty"] and not state["valid"]]
        evaluable = [state for state in subset if state not in forced and state not in empty and state not in unknown]
        positive = [state for state in evaluable if isinstance(state["gain"], (int, float)) and not isinstance(state["gain"], bool) and state["gain"] > 0]
        non_positive = [state for state in evaluable if state not in positive]
        reasons = sorted({str(state["unknown_reason"]) for state in unknown})
        by_split[split] = {
            "states": len(subset), "forced_excluded": len(forced), "empty_excluded": len(empty),
            "unknown": len(unknown), "evaluable": len(evaluable), "positive": len(positive),
            "non_positive": len(non_positive),
            "oracle_structurally_valid": sum(state["oracle_structurally_valid"] for state in subset),
            "unknown_reasons": {reason: sum(str(state["unknown_reason"]) == reason for state in unknown) for reason in reasons},
            "positive_scenario_groups": sorted({state["scenario_id"] for state in positive}),
            "non_positive_scenario_groups": sorted({state["scenario_id"] for state in non_positive}),
            "state_ids": sorted(state["state_id"] for state in evaluable),
        }
    return {"schema": "gain-gate-v33-state-summary/1", "state_count": len(states), "splits": by_split}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    files = sorted((PACKAGE_ROOT / "civilian").glob("*.py"))
    rows = [{"path": str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/"), "sha256": sha256_file(path)} for path in files]
    return {"schema": "gain-gate-v33-source-manifest/1", "files": rows, "sha256": sha256_bytes(canonical_json(rows))}


def current_binding() -> dict[str, Any]:
    source = source_manifest()
    config_files = sorted((PROJECT_ROOT / "configs").glob("*.json"))
    config_rows = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in config_files
    ]
    config_hash = sha256_bytes(canonical_json(config_rows))
    baseline_hash = sha256_file(BASELINE_PATH)
    evidence_hash = sha256_bytes(canonical_json({"baseline_provenance_sha256": baseline_hash}))
    return {
        "source_manifest_sha256": source["sha256"],
        "config_sha256": config_hash,
        "config_files": config_rows,
        "evidence_sha256": evidence_hash,
        "baseline_provenance_sha256": baseline_hash,
    }


def expected_registry(*, task_mode: str = "search_rescue", targets_per_region: int = 3) -> list[dict[str, Any]]:
    """Return the fixed 7 x (12 train + 12 validation) registry."""
    if task_mode not in TASK_MODES or targets_per_region not in TARGET_DENSITIES:
        raise ValueError("V33_TASK_CONFIGURATION_INVALID")
    result = []
    for condition_index, condition in enumerate(CONDITIONS):
        for split in ("train", "validation"):
            for instance_index in range(12):
                result.append({
                    "split": split,
                    "condition": condition,
                    "instance_index": instance_index,
                    "seed_variant": 8201 + condition_index * 1000 + instance_index + (100_000 if split == "validation" else 0),
                    "task_mode": task_mode,
                    "targets_per_region": targets_per_region,
                })
    return result


def validate_registry(rows: Iterable[Mapping[str, Any]], *, task_mode: str = "search_rescue", targets_per_region: int = 3) -> list[dict[str, Any]]:
    actual = [dict(row) for row in rows]
    expected = expected_registry(task_mode=task_mode, targets_per_region=targets_per_region)
    formal_splits = ("train", "validation")
    if (
        len(expected) != 168
        or any(row.get("split") not in formal_splits for row in expected)
        or any(
            sum(row["condition"] == condition and row["split"] == split for row in expected) != 12
            for condition in CONDITIONS for split in formal_splits
        )
    ):
        raise ValueError("V33_FORMAL_REGISTRY_NOT_168_TRAIN_VALIDATION")
    if any(row.get("split") not in formal_splits for row in actual):
        raise ValueError("V3_TEST_OOD_SPLITS_ARE_SEALED")
    if actual != expected:
        raise ValueError("V33_REGISTRY_NOT_FROZEN_7X12_TRAIN_VALIDATION")
    return actual


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_gzip_jsonl(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"V3_SHARD_JSONL_LINE_NOT_TERMINATED:{line_number}")
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"V3_SHARD_ROW_OBJECT_REQUIRED:{line_number}")
            digest.update(line)
            rows += 1
    return rows, digest.hexdigest()


def commit_scenario_shard(
    *, output_dir: str | Path, run_id: str, scenario: Mapping[str, Any], rows: Iterable[Mapping[str, Any]],
    binding: Mapping[str, str], runtime_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit a gzip JSONL shard, then its marker; never replace existing evidence."""
    split, condition, parent_id = scenario.get("split"), scenario.get("condition"), scenario.get("parent_id")
    if split not in SPLITS or condition not in CONDITIONS or not isinstance(parent_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", parent_id) or parent_id in {".", ".."}:
        raise ValueError("V3_SHARD_SCENARIO_IDENTITY_INVALID")
    if not run_id or not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", run_id) or run_id in {".", ".."}:
        raise ValueError("V3_RUN_ID_INVALID")
    root = Path(output_dir) / run_id
    shard_dir = root / "shards" / str(split) / str(condition)
    shard_path = shard_dir / f"{parent_id}.jsonl.gz"
    marker_path = shard_dir / f"{parent_id}.commit.json"
    if shard_path.exists() or marker_path.exists():
        raise FileExistsError("V3_SHARD_ALREADY_EXISTS")
    shard_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{parent_id}.", suffix=".tmp", dir=shard_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    raw_digest = hashlib.sha256()
    row_count = 0
    started = time.perf_counter()
    try:
        with temp_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for value in rows:
                    row = dict(value)
                    row_bytes = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
                    compressed.write(row_bytes)
                    raw_digest.update(row_bytes)
                    row_count += 1
            raw.flush()
            os.fsync(raw.fileno())
        validated_rows, decompressed_hash = _validate_gzip_jsonl(temp_path)
        if validated_rows != row_count or decompressed_hash != raw_digest.hexdigest():
            raise ValueError("V3_SHARD_POSTWRITE_VALIDATION_MISMATCH")
        compressed_hash = sha256_file(temp_path)
        os.replace(temp_path, shard_path)
        marker = {
            "schema": MARKER_SCHEMA,
            "run_id": run_id,
            "scenario_id": parent_id,
            "split": split,
            "condition": condition,
            "scenario_content_sha256": scenario.get("scenario_content_sha256"),
            "row_count": row_count,
            "compressed_sha256": compressed_hash,
            "decompressed_jsonl_sha256": decompressed_hash,
            "binding": dict(binding),
            "runtime_receipt": dict(runtime_receipt),
            "wall_seconds": max(0.0, time.perf_counter() - started),
            "shard_file": shard_path.name,
        }
        marker["marker_sha256"] = sha256_bytes(canonical_json(marker))
        marker_bytes = json.dumps(marker, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
        marker_tmp = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
        _write_exclusive(marker_tmp, marker_bytes)
        os.replace(marker_tmp, marker_path)
        return {**marker, "marker_path": str(marker_path.resolve())}
    except BaseException as exc:
        failure = {
            "schema": "gain-gate-v33-shard-failure/1", "run_id": run_id,
            "scenario_id": parent_id, "split": split, "condition": condition,
            "error_type": type(exc).__name__, "error": str(exc),
            "temporary_data_path": str(temp_path) if temp_path.exists() else None,
            "data_committed_without_marker": shard_path.exists() and not marker_path.exists(),
            "completion_marker_written": marker_path.exists(),
        }
        try:
            failure_path = root / "failures" / f"{parent_id}.failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_bytes = json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii") + b"\n"
            _write_exclusive(failure_path, failure_bytes)
        except Exception:
            pass
        raise


def verify_scenario_shard(marker_path: str | Path, *, expected_binding: Mapping[str, str] | None = None) -> dict[str, Any]:
    marker_path = Path(marker_path)
    marker = json.loads(marker_path.read_text(encoding="ascii"))
    if not isinstance(marker, dict) or marker.get("schema") != MARKER_SCHEMA:
        raise ValueError("V3_COMMIT_MARKER_SCHEMA_INVALID")
    supplied_hash = marker.pop("marker_sha256", None)
    if supplied_hash != sha256_bytes(canonical_json(marker)):
        raise ValueError("V3_COMMIT_MARKER_HASH_MISMATCH")
    marker["marker_sha256"] = supplied_hash
    scenario_id = marker.get("scenario_id")
    if not isinstance(scenario_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", scenario_id) or scenario_id in {".", ".."}:
        raise ValueError("V3_COMMIT_MARKER_SCENARIO_ID_INVALID")
    if marker_path.name != f"{scenario_id}.commit.json" or marker.get("shard_file") != f"{scenario_id}.jsonl.gz":
        raise ValueError("V3_COMMIT_MARKER_FILENAME_MISMATCH")
    if expected_binding is not None and marker.get("binding") != dict(expected_binding):
        raise ValueError("V3_SHARD_BINDING_MISMATCH")
    path = marker_path.with_name(str(marker.get("shard_file", "")))
    if path.name != marker.get("shard_file") or not path.is_file():
        raise ValueError("V3_COMMITTED_SHARD_MISSING")
    if sha256_file(path) != marker.get("compressed_sha256"):
        raise ValueError("V3_SHARD_COMPRESSED_HASH_MISMATCH")
    rows, decompressed_hash = _validate_gzip_jsonl(path)
    if rows != marker.get("row_count") or decompressed_hash != marker.get("decompressed_jsonl_sha256"):
        raise ValueError("V3_SHARD_GZIP_CONTENT_MISMATCH")
    return marker


def verify_run(output_dir: str | Path, run_id: str, *, expected_binding: Mapping[str, str] | None = None) -> dict[str, Any]:
    run_root = Path(output_dir) / run_id
    markers = sorted(run_root.glob("shards/*/*/*.commit.json"))
    verified = []
    for path in markers:
        marker = verify_scenario_shard(path, expected_binding=expected_binding)
        if marker.get("split") not in SPLITS or marker.get("condition") not in CONDITIONS:
            raise ValueError("V3_SEALED_SPLIT_OR_UNKNOWN_CONDITION")
        marker["marker_path"] = str(path.resolve())
        verified.append(marker)
    identities = [(row["split"], row["condition"], row["scenario_id"]) for row in verified]
    if len(identities) != len(set(identities)):
        raise ValueError("V33_DUPLICATE_COMMITTED_SCENARIO")
    return {"run_id": run_id, "committed_shards": len(verified), "verified": verified}


def resource_receipt(*, stage: str, started_wall: float, started_cpu: float, current_rss_bytes: int | None) -> dict[str, Any]:
    wall = max(0.0, time.perf_counter() - started_wall)
    cpu = max(0.0, time.process_time() - started_cpu)
    return {
        "schema": "gain-gate-v33-resource-receipt/1", "stage": stage,
        "python": platform.python_version(), "platform": platform.platform(),
        "wall_seconds": wall, "process_cpu_seconds": cpu,
        "current_rss_bytes_at_receipt": current_rss_bytes,
        "execution": {
            "simulation_device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
    }
