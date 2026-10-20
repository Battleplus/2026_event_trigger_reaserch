"""Read-only gates for the frozen formal collection.

The gate deliberately reads the batch receipt before opening ``aggregate.json``.
It only reads aggregate coverage summaries.  Individual state rows and labels are
opened by :mod:`dataset` after this gate passes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SCENE_COUNT = 168
TERMINAL_EXECUTIONS = {"COMPLETE", "PARTIAL_OR_FAILED"}
SUMMARY_FIELDS = (
    "status",
    "state_count",
    "valid_state_count",
    "positive_count",
    "nonpositive_count",
    "count_30_30_met",
    "split_coverage_met",
    "met_30_30",
    "excluded_state_count",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _scene_attempt_is_recorded(scene: Any) -> bool:
    if not isinstance(scene, dict):
        return False
    required = ("run_id", "output_path", "execution")
    if any(not scene.get(key) for key in required):
        return False
    # run_formal_batch records representative/state attempts inline.  The other
    # names make the check forward-compatible with an explicit scene receipt.
    return any(key in scene for key in ("representative", "states", "scene_receipt", "receipt", "error"))


def _summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {key: aggregate.get(key) for key in SUMMARY_FIELDS}


def inspect_batch_output(batch_output: str | Path) -> dict[str, Any]:
    """Return a gate result without opening aggregate state rows.

    ``batch_output`` is the output root produced by ``run_formal_batch.py``.
    The canonical aggregate is always ``<root>/aggregate.json``.  A receipt
    pointing elsewhere is rejected so the reduction being gated is unambiguous.
    """
    root = Path(batch_output).resolve()
    receipt_path = root / "batch_receipt.json"
    receipt = _read_json(receipt_path)
    base: dict[str, Any] = {
        "status": "NOT_RUN",
        "ready": False,
        "batch_output": str(root),
        "batch_receipt": str(receipt_path),
        "aggregate": str(root / "aggregate.json"),
        "reason": None,
        "aggregate_summary": None,
    }
    if receipt is None:
        base["reason"] = "BATCH_RECEIPT_MISSING_OR_INVALID"
        return base

    execution = receipt.get("execution")
    if execution not in TERMINAL_EXECUTIONS or not receipt.get("completed_at_utc"):
        base["reason"] = "BATCH_NOT_TERMINAL"
        base["receipt_execution"] = execution
        return base

    scenes = receipt.get("scenes")
    scene_count = receipt.get("scene_count")
    recorded = isinstance(scenes, list) and len(scenes) == EXPECTED_SCENE_COUNT and scene_count == EXPECTED_SCENE_COUNT
    if recorded:
        recorded = all(_scene_attempt_is_recorded(scene) for scene in scenes)
        run_ids = [scene.get("run_id") for scene in scenes if isinstance(scene, dict)]
        recorded = recorded and len(set(run_ids)) == EXPECTED_SCENE_COUNT
    if not recorded:
        base["reason"] = "BATCH_SCENE_RECEIPTS_INCOMPLETE"
        base["receipt_execution"] = execution
        base["scene_count"] = scene_count
        base["recorded_scene_entries"] = len(scenes) if isinstance(scenes, list) else None
        return base

    recorded_aggregate = (receipt.get("end_reduction") or {}).get("aggregate_output")
    canonical_aggregate = root / "aggregate.json"
    if recorded_aggregate:
        try:
            recorded_path = Path(str(recorded_aggregate))
            if not recorded_path.is_absolute():
                recorded_path = root / recorded_path
            if recorded_path.resolve() != canonical_aggregate.resolve():
                base["reason"] = "AGGREGATE_PATH_MISMATCH"
                base["recorded_aggregate"] = str(recorded_path.resolve())
                return base
        except (OSError, RuntimeError, ValueError):
            base["reason"] = "AGGREGATE_PATH_INVALID"
            return base

    aggregate = _read_json(canonical_aggregate)
    if aggregate is None:
        base["reason"] = "AGGREGATE_MISSING_OR_INVALID"
        return base
    summary = _summary(aggregate)
    base["aggregate_summary"] = summary
    if any(summary[key] is None for key in SUMMARY_FIELDS):
        base["reason"] = "AGGREGATE_SUMMARY_INCOMPLETE"
        return base
    try:
        coverage_ok = (
            summary["met_30_30"] is True
            and summary["count_30_30_met"] is True
            and summary["split_coverage_met"] is True
            and int(summary["positive_count"]) >= 30
            and int(summary["nonpositive_count"]) >= 30
        )
    except (TypeError, ValueError):
        coverage_ok = False
    if not coverage_ok:
        base["reason"] = "AGGREGATE_COVERAGE_INSUFFICIENT"
        return base
    base.update({"status": "READY", "ready": True, "reason": None,
                 "receipt_execution": execution, "scene_count": EXPECTED_SCENE_COUNT})
    return base


def check_feature_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute 30/30 after feature extraction and exclusions.

    This is intentionally independent of aggregate summaries.  Rows excluded
    because their public projection or hash binding is unavailable cannot count
    toward the training gate.
    """
    usable = []
    excluded = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("excluded"):
            excluded += 1
            continue
        try:
            target = float(row["target"])
            if not math.isfinite(target) or row.get("split") not in {"train", "validation"}:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            excluded += 1
            continue
        usable.append(row)
    positive = [row for row in usable if float(row["target"]) > 0]
    nonpositive = [row for row in usable if float(row["target"]) <= 0]
    split_counts = {label: {split: 0 for split in ("train", "validation")}
                    for label in ("positive", "nonpositive")}
    for label, selected in (("positive", positive), ("nonpositive", nonpositive)):
        for row in selected:
            split_counts[label][row["split"]] += 1
    count_gate = len(positive) >= 30 and len(nonpositive) >= 30
    split_gate = all(split_counts[label][split] > 0
                     for label in ("positive", "nonpositive")
                     for split in ("train", "validation"))
    return {
        "usable_count": len(usable),
        "excluded_count": excluded,
        "positive_count": len(positive),
        "nonpositive_count": len(nonpositive),
        "split_counts": split_counts,
        "count_30_30_met": count_gate,
        "split_coverage_met": split_gate,
        "met_30_30": count_gate and split_gate,
    }
