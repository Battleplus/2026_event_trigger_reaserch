#!/usr/bin/env python
"""Audit interrupted gain-gate collection outputs without modifying them."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _safe_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    result = (root / path).resolve()
    if root.resolve() not in result.parents:
        return None
    return result


def _read_gzip_jsonl(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    gzip_error = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("row is not an object")
                    rows.append(value)
                except (json.JSONDecodeError, ValueError) as error:
                    errors.append(f"line {len(rows) + len(errors) + 1}: {error}")
    except (EOFError, OSError, UnicodeDecodeError) as error:
        gzip_error = f"{type(error).__name__}: {error}"
    return {"rows": rows, "parse_errors": errors, "gzip_error": gzip_error, "gzip_complete": gzip_error is None}


def _read_gzip_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream), None
    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _check_linked_file(
    root: Path, relative: Any, expected_hash: Any, *, gz_json: bool = False,
) -> tuple[bool, str | None, Any | None]:
    path = _safe_path(root, relative)
    if path is None or not path.is_file():
        return False, "missing_or_unsafe_path", None
    actual_hash = _sha256(path)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        return False, "sha256_mismatch", None
    value = None
    if gz_json:
        value, error = _read_gzip_json(path)
        if error:
            return False, f"gzip_json_invalid:{error}", None
    return True, None, value


def _audit_run(
    run_dir: Path, *, config_source: Path, cpu_lower_bound: float | None, expected_total: int | None,
) -> dict[str, Any]:
    run = _load_json(run_dir / "RUN.json")
    config = _load_json(run_dir / "RESOLVED_CONFIG.json")
    source_config = _load_json(config_source)
    source_manifest = _load_json(run_dir / "SOURCE_MANIFEST.json")
    rows_result = _read_gzip_jsonl(run_dir / "COLLECTION_ROWS.jsonl.gz")
    outcomes_result = _read_gzip_jsonl(run_dir / "SCENARIO_OUTCOMES.jsonl.gz")
    rows = rows_result["rows"]
    outcomes = outcomes_result["rows"]

    outcome_ids: set[str] = set()
    duplicate_outcome_ids: list[str] = []
    outcome_errors: list[dict[str, str]] = []
    for outcome in outcomes:
        scenario_id = str(outcome.get("scenario_id", ""))
        if not scenario_id or scenario_id in outcome_ids:
            duplicate_outcome_ids.append(scenario_id)
        outcome_ids.add(scenario_id)
        scenario_path = _safe_path(run_dir, outcome.get("scenario_path"))
        if scenario_path is None or not scenario_path.is_file():
            outcome_errors.append({"scenario_id": scenario_id, "error": "scenario_file_missing_or_unsafe"})
            continue
        if _sha256(scenario_path) != outcome.get("scenario_file_sha256"):
            outcome_errors.append({"scenario_id": scenario_id, "error": "scenario_file_sha256_mismatch"})
            continue
        scenario, error = _read_gzip_json(scenario_path)
        if error:
            outcome_errors.append({"scenario_id": scenario_id, "error": f"scenario_gzip_invalid:{error}"})
        elif not isinstance(scenario, dict) or scenario.get("parent_id") != scenario_id:
            outcome_errors.append({"scenario_id": scenario_id, "error": "scenario_identity_mismatch"})

    scenario_files = sorted((run_dir / "scenarios").glob("*.json.gz"))
    scenario_file_ids = {path.name[:-len(".json.gz")] for path in scenario_files}
    incomplete_scenario_ids = sorted(scenario_file_ids - outcome_ids)
    orphan_outcome_ids = sorted(outcome_ids - scenario_file_ids)

    state_ids = [str(row.get("state_id", "")) for row in rows]
    duplicate_state_ids = sorted(
        state_id for state_id, count in collections.Counter(state_ids).items() if not state_id or count > 1
    )
    artifact_errors: list[dict[str, str]] = []
    sign_counts: collections.Counter[str] = collections.Counter()
    unknown_reasons: collections.Counter[str] = collections.Counter()
    forced_count = empty_count = variance_valid_count = 0
    rows_by_scenario: collections.Counter[str] = collections.Counter()
    complete_rows_by_scenario: collections.Counter[str] = collections.Counter()
    partial_rows_by_scenario: collections.Counter[str] = collections.Counter()
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        rows_by_scenario[scenario_id] += 1
        if scenario_id in outcome_ids:
            complete_rows_by_scenario[scenario_id] += 1
        else:
            partial_rows_by_scenario[scenario_id] += 1
        label_ok, label_error, label_obj = _check_linked_file(
            run_dir, row.get("raw_label_path"), row.get("raw_label_sha256"), gz_json=True,
        )
        if not label_ok:
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": f"raw_label:{label_error}"})
            continue
        label = label_obj.get("label") if isinstance(label_obj, dict) else None
        if not isinstance(label, dict):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "raw_label_shape_invalid"})
            continue
        if label.get("label_sha256") != row.get("label_sha256"):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "label_sha256_mismatch"})
        if label.get("state_id") != row.get("state_id") or label.get("scenario_id") != scenario_id:
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "label_identity_mismatch"})
        oracle = row.get("oracle") or {}
        if oracle.get("label_sha256") != label.get("label_sha256"):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "oracle_label_binding_mismatch"})
        # ``oracle.valid`` means the validator accepted the receipt.  The
        # trainability mask is carried separately as ``label_valid_mask``.
        if bool(oracle.get("label_valid_mask")) != bool(row.get("valid_mask")):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "oracle_label_validity_mismatch"})
        if label.get("valid_mask") != row.get("valid_mask") or label.get("forced") != row.get("forced"):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "row_label_mask_or_forced_mismatch"})
        snapshot_ok, snapshot_error, _ = _check_linked_file(
            run_dir, row.get("snapshot_path"), row.get("snapshot_file_sha256"), gz_json=True,
        )
        if not snapshot_ok:
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": f"snapshot:{snapshot_error}"})
        suffix_paths = row.get("suffix_paths") or []
        suffix_hashes = row.get("suffix_file_hashes") or []
        if not isinstance(suffix_paths, list) or not isinstance(suffix_hashes, list) or len(suffix_paths) != len(suffix_hashes):
            artifact_errors.append({"state_id": str(row.get("state_id")), "error": "suffix_path_hash_count_mismatch"})
        else:
            for relative, expected in zip(suffix_paths, suffix_hashes):
                suffix_ok, suffix_error, _ = _check_linked_file(run_dir, relative, expected, gz_json=True)
                if not suffix_ok:
                    artifact_errors.append({"state_id": str(row.get("state_id")), "error": f"suffix:{suffix_error}"})
                    break
        forced = bool(label.get("forced"))
        valid = bool(label.get("valid_mask"))
        reason = label.get("unknown_reason")
        forced_count += int(forced)
        empty_count += int(reason == "empty_obligation_window")
        variance_valid_count += int(bool(label.get("variance_valid_mask")))
        if not valid:
            unknown_reasons[str(reason or "unspecified")]+=1
            sign_counts["unknown"] += 1
        else:
            gain = label.get("gain_mean")
            if not isinstance(gain, (int, float)):
                artifact_errors.append({"state_id": str(row.get("state_id")), "error": "valid_label_missing_gain"})
                sign_counts["invalid_valid_label"] += 1
            elif gain > 0:
                sign_counts["positive"] += 1
                if not forced and reason is None:
                    sign_counts["ordinary_positive"] += 1
            else:
                sign_counts["nonpositive"] += 1
                if not forced and reason is None:
                    sign_counts["ordinary_nonpositive"] += 1

    complete_instance_signs: collections.Counter[str] = collections.Counter()
    partial_instance_signs: collections.Counter[str] = collections.Counter()
    for row in rows:
        label_path = _safe_path(run_dir, row.get("raw_label_path"))
        if label_path is None or not label_path.is_file():
            continue
        label_obj, error = _read_gzip_json(label_path)
        if error or not isinstance(label_obj, dict) or not isinstance(label_obj.get("label"), dict):
            continue
        label = label_obj["label"]
        bucket = complete_instance_signs if row.get("scenario_id") in outcome_ids else partial_instance_signs
        if not label.get("valid_mask"):
            bucket["unknown"] += 1
        elif label.get("gain_mean") is not None and label["gain_mean"] > 0:
            bucket["positive"] += 1
        else:
            bucket["nonpositive"] += 1

    config_counts = config.get("scenario", {}).get("formal_instances_per_condition", {})
    config_total = sum(int(value) for value in config_counts.values()) * len(config.get("scenario", {}).get("formal_conditions", []))
    return {
        "run_id": run.get("run_id"),
        "stage": run.get("stage"),
        "recorded_status": run.get("status"),
        "audit_disposition": "interrupted_incomplete",
        "source_manifest_sha256": run.get("source_manifest_sha256"),
        "source_manifest_self_sha256": _sha256(run_dir / "SOURCE_MANIFEST.json"),
        "config_sha256_recorded": run.get("config_sha256"),
        "config_self_sha256": _sha256(run_dir / "RESOLVED_CONFIG.json"),
        "config_source_sha256": _sha256(config_source),
        "source_identity_matches_run": run.get("source_manifest_sha256") == _sha256(run_dir / "SOURCE_MANIFEST.json"),
        "config_identity_matches_run": run.get("config_sha256") == _sha256(config_source),
        "scenario_file_count": len(scenario_files),
        "completed_scenario_outcome_count": len(outcome_ids),
        "scenario_ids_without_outcome": incomplete_scenario_ids,
        "outcome_ids_without_scenario_file": orphan_outcome_ids,
        "duplicate_outcome_ids": duplicate_outcome_ids,
        "duplicate_state_ids": duplicate_state_ids,
        "collection_rows": {
            "parseable_complete_json_rows": len(rows),
            "gzip_footer_valid": rows_result["gzip_complete"],
            "gzip_error": rows_result["gzip_error"],
            "parse_errors": rows_result["parse_errors"],
        },
        "scenario_outcomes": {
            "parseable_complete_json_rows": len(outcomes),
            "gzip_footer_valid": outcomes_result["gzip_complete"],
            "gzip_error": outcomes_result["gzip_error"],
            "parse_errors": outcomes_result["parse_errors"],
        },
        "labels_all_parseable_rows": {
            "positive": sign_counts["positive"],
            "nonpositive": sign_counts["nonpositive"],
            "ordinary_positive": sign_counts["ordinary_positive"],
            "ordinary_nonpositive": sign_counts["ordinary_nonpositive"],
            "unknown": sign_counts["unknown"],
            "unknown_reasons": dict(sorted(unknown_reasons.items())),
            "forced": forced_count,
            "empty_obligation_window": empty_count,
            "paired_variance_valid": variance_valid_count,
        },
        "labels_linked_to_completed_outcomes": dict(sorted(complete_instance_signs.items())),
        "labels_isolated_to_scenario_without_outcome": dict(sorted(partial_instance_signs.items())),
        "rows_grouped_by_scenario_id": dict(sorted(rows_by_scenario.items())),
        "integrity": {
            "outcome_scenario_errors": outcome_errors,
            "linked_artifact_errors": artifact_errors,
            "all_parseable_row_artifacts_verified": not artifact_errors,
            "all_outcome_scenarios_verified": not outcome_errors,
        },
        "resource_accounting": {
            "cpu_seconds_total": None,
            "cpu_seconds_total_status": "unknown_incomplete_run",
            "last_observation_cpu_seconds_lower_bound": cpu_lower_bound,
            "lower_bound_source": "coordinator_last_observation" if cpu_lower_bound is not None else None,
            "rss_peak_final": None,
        },
        "expected_instance_count_operator_supplied": expected_total,
        "config_declared_formal_instance_count_across_all_splits": config_total,
        "training_gate_met": sign_counts["ordinary_positive"] >= 30 and sign_counts["ordinary_nonpositive"] >= 30,
        "light_model_training_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="diagnostic JSON written outside outputs/")
    parser.add_argument("--expected-total", type=int, default=168)
    parser.add_argument("--final-last-cpu-lower-bound", type=float, default=1890.72)
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    output = args.output.resolve()
    if project == output or project in output.parents and output.parent.name == "outputs":
        parser.error("diagnostic output must not be written inside the source project or outputs tree")
    specs = (
        ("v2-collect-final-20261017", args.final_last_cpu_lower_bound),
        ("v2-collect-r2-20261017", None),
    )
    report = {
        "schema": "gain-interrupted-collection-audit/1",
        "audit_scope": "read_only_input; no collection, completion receipts, or resource totals synthesized",
        "expected_instance_count_basis": "operator-supplied registered collection target; run metadata does not persist CLI limits",
        "runs": [_audit_run(project / "outputs" / name, config_source=project / "configs" / "GAIN_GATE_CONFIG.json", cpu_lower_bound=cpu, expected_total=args.expected_total) for name, cpu in specs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": [
        {"run_id": item["run_id"], "completed_outcomes": item["completed_scenario_outcome_count"],
         "scenario_files": item["scenario_file_count"], "ordinary_positive": item["labels_all_parseable_rows"]["ordinary_positive"],
         "ordinary_nonpositive": item["labels_all_parseable_rows"]["ordinary_nonpositive"],
         "integrity_errors": len(item["integrity"]["linked_artifact_errors"]) + len(item["integrity"]["outcome_scenario_errors"])}
        for item in report["runs"]
    ]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
