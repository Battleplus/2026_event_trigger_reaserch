"""Bind aggregate labels to public projections without opening private snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from .data_gate import check_feature_coverage, inspect_batch_output
    from .features import FEATURE_SCHEMA, extract_features
except ImportError:  # direct invocation from this directory for local tests
    from data_gate import check_feature_coverage, inspect_batch_output
    from features import FEATURE_SCHEMA, extract_features


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _report_location(report_path: Any) -> tuple[str, str] | None:
    if not isinstance(report_path, str):
        return None
    normalized = report_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.name != "reduction.json":
        return None
    if len(path.parts) < 4 or not path.parts[-2].startswith("suffix_"):
        return None
    return path.parts[-4], path.parts[-3]


def _safe_public_path(state_root: Path, public_file: Any) -> Path | None:
    if not isinstance(public_file, str):
        return None
    candidate = (state_root / public_file).resolve()
    try:
        candidate.relative_to(state_root.resolve())
    except ValueError:
        return None
    return candidate


def _load_public_projection(batch_root: Path, scene_name: str, local_state_id: str) -> tuple[dict[str, Any] | None, str | None]:
    state_root = batch_root / "scenes" / scene_name / "representative" / "state_sampling"
    index_path = state_root / "state_index.json"
    index = _read(index_path)
    if index is None or not isinstance(index.get("states"), list):
        return None, "state_index_missing_or_invalid"
    matches = []
    for entry in index["states"]:
        if not isinstance(entry, dict):
            continue
        private_file = str(entry.get("private_file", ""))
        if Path(private_file).stem == local_state_id:
            matches.append(entry)
    if len(matches) != 1:
        return None, "state_index_local_state_ambiguous_or_missing"
    entry = matches[0]
    public_path = _safe_public_path(state_root, entry.get("public_file"))
    if public_path is None or not public_path.is_file():
        return None, "public_projection_missing"
    try:
        public_bytes = public_path.read_bytes()
    except OSError:
        return None, "public_projection_unreadable"
    expected_hash = entry.get("public_sha256")
    if not isinstance(expected_hash, str) or hashlib.sha256(public_bytes).hexdigest() != expected_hash:
        return None, "public_projection_hash_mismatch"
    try:
        projection = json.loads(public_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None, "public_projection_invalid_json"
    if not isinstance(projection, dict):
        return None, "public_projection_not_object"
    return projection, None


def load_feature_dataset(batch_output: str | Path) -> dict[str, Any]:
    """Load features only after the terminal batch and aggregate gates pass."""
    gate = inspect_batch_output(batch_output)
    result: dict[str, Any] = {
        "gate": gate,
        "feature_schema": list(FEATURE_SCHEMA),
        "rows": [],
        "excluded": [],
        "coverage": None,
    }
    if not gate["ready"]:
        return result
    root = Path(batch_output).resolve()
    receipt = _read(root / "batch_receipt.json")
    aggregate = _read(root / "aggregate.json")
    if receipt is None or aggregate is None:
        result["gate"] = {**gate, "status": "NOT_RUN", "ready": False,
                           "reason": "BATCH_OR_AGGREGATE_CHANGED_AFTER_GATE"}
        return result

    scene_map: dict[str, dict[str, Any]] = {}
    for scene in receipt.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        output_path = str(scene.get("output_path", "")).replace("\\", "/").strip("/")
        if output_path:
            scene_map[output_path.split("/")[-1]] = scene

    states = aggregate.get("states")
    if not isinstance(states, list):
        result["excluded"].append({"reason": "aggregate_states_missing"})
        result["coverage"] = check_feature_coverage(result["rows"])
        return result

    for state in states:
        if not isinstance(state, dict):
            result["excluded"].append({"reason": "aggregate_state_not_object"})
            continue
        if state.get("valid_mask") is not True:
            result["excluded"].append({"state_id": state.get("state_id"),
                                        "reason": "invalid_or_unknown_label",
                                        "unknown_reason": state.get("unknown_reason")})
            continue
        try:
            target = float(state["gain_mean"])
        except (KeyError, TypeError, ValueError):
            result["excluded"].append({"state_id": state.get("state_id"), "reason": "invalid_gain_mean"})
            continue
        scenario_id = state.get("scenario_id")
        local_state_id = state.get("local_state_id")
        split = state.get("split")
        reports = state.get("source_reports")
        locations = [_report_location(item) for item in reports] if isinstance(reports, list) else []
        if (not isinstance(scenario_id, str) or not isinstance(local_state_id, str)
                or split not in {"train", "validation"} or len(locations) != 2
                or any(location is None for location in locations)):
            result["excluded"].append({"state_id": state.get("state_id"), "reason": "aggregate_identity_or_source_reports_invalid"})
            continue
        scene_names = {location[0] for location in locations if location is not None}
        local_names = {location[1] for location in locations if location is not None}
        suffix_names = {PurePosixPath(str(item).replace("\\", "/")).parts[-2]
                        for item in reports if isinstance(item, str)}
        if (len(scene_names) != 1 or local_names != {local_state_id}
                or len(suffix_names) != 2
                or state.get("state_id") != f"{scenario_id}::{local_state_id}"):
            result["excluded"].append({"state_id": state.get("state_id"), "reason": "source_reports_do_not_bind_one_public_state"})
            continue
        scene_name = next(iter(scene_names))
        scene_record = scene_map.get(scene_name)
        if scene_record is None or scene_record.get("scenario_id") != scenario_id or scene_record.get("split") != split:
            result["excluded"].append({"state_id": state.get("state_id"), "reason": "source_report_scene_not_in_batch_receipt"})
            continue
        projection, error = _load_public_projection(root, scene_name, local_state_id)
        if error:
            result["excluded"].append({"state_id": state.get("state_id"), "reason": error})
            continue
        try:
            features = extract_features(projection or {})
        except (TypeError, ValueError, AssertionError) as exc:
            result["excluded"].append({"state_id": state.get("state_id"),
                                        "reason": "feature_extraction_failed",
                                        "detail": str(exc)})
            continue
        result["rows"].append({
            "state_id": state.get("state_id"), "local_state_id": local_state_id,
            "scenario_id": scenario_id, "split": split, "target": target,
            "features": features,
        })
    result["coverage"] = check_feature_coverage(result["rows"])
    result["exclusion_counts"] = {}
    for item in result["excluded"]:
        reason = item.get("reason", "unknown")
        result["exclusion_counts"][reason] = result["exclusion_counts"].get(reason, 0) + 1
    return result
