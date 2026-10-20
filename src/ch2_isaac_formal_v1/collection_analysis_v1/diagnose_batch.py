#!/usr/bin/env python3
"""Read-only engineering diagnostics for one completed formal collection batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


EXPECTED_SUFFIX_SEEDS = (730201, 730202)
LABEL_TARGET_PER_CLASS = 30
REQUIRED_SPLITS = ("train", "validation")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _child(root: Path, relative) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()
    if not _inside(root, resolved):
        raise ValueError(f"receipt path escapes evidence root: {relative}")
    return resolved


def _optional_json(path: Path):
    try:
        return _read_json(path), None
    except FileNotFoundError:
        return None, "missing_file"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable_json:{type(exc).__name__}:{exc}"


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values):
    clean = [float(value) for value in values if _finite_number(value) is not None]
    return sum(clean) / len(clean) if clean else None


def _sum(values):
    clean = [float(value) for value in values if _finite_number(value) is not None]
    return sum(clean) if clean else None


def _numeric_summary(values):
    known = [float(value) for value in values if _finite_number(value) is not None]
    return {
        "known_scene_count": len(known),
        "unknown_scene_count": len(values) - len(known),
        "mean_across_scenes": _mean(known),
        "sum_across_scenes": _sum(known),
        "min_across_scenes": min(known) if known else None,
        "max_across_scenes": max(known) if known else None,
    }


def _valid_run_identity(row: dict, receipt: dict | None) -> bool:
    if not isinstance(receipt, dict):
        return False
    if not all(receipt.get(key) == row.get(key)
               for key in ("run_id", "scenario_id", "split", "condition", "source")):
        return False
    if receipt.get("config_sha256") != row.get("config_sha256"):
        return False
    if receipt.get("output_path") != row.get("output_path"):
        return False
    return True


def _collector_receipt_matches(row: dict, receipt: dict | None, phase: str) -> bool:
    if not isinstance(receipt, dict):
        return False
    formal_identity = receipt.get("formal_identity")
    return (
        receipt.get("run_id") == row.get("run_id")
        and receipt.get("config_sha256") == row.get("config_sha256")
        and receipt.get("phase") == phase
        and isinstance(formal_identity, dict)
        and all(formal_identity.get(key) == row.get(key)
                for key in ("scenario_id", "split", "condition", "source"))
    )


def _state_index(rep_dir: Path, rep_receipt: dict | None):
    if not isinstance(rep_receipt, dict):
        return None, ["representative_receipt_missing"]
    if rep_receipt.get("status") != "complete" or rep_receipt.get("actual_isaac_run") is not True:
        return None, ["representative_not_complete_actual_isaac"]

    embedded = rep_receipt.get("state_sampling")
    index_path = rep_dir / "state_sampling" / "state_index.json"
    index, error = _optional_json(index_path)
    if error:
        return None, [f"state_index_{error}"]
    if not isinstance(index, dict) or not isinstance(index.get("states"), list):
        return None, ["state_index_malformed"]
    if embedded != index:
        return None, ["state_index_differs_from_representative_receipt"]

    sidecar = index_path.with_suffix(".sha256")
    try:
        recorded_hash = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError):
        return None, ["state_index_sidecar_missing_or_unreadable"]
    if recorded_hash != _sha256(index_path):
        return None, ["state_index_sidecar_mismatch"]

    state_ids = []
    reasons = []
    sampling_dir = index_path.parent
    for entry in index["states"]:
        try:
            state_id = Path(str(entry["private_file"])).stem
            private_path = _child(sampling_dir, entry["private_file"])
            public_path = _child(sampling_dir, entry["public_file"])
            if not private_path.is_file() or not public_path.is_file():
                reasons.append(f"state_snapshot_or_projection_missing:{state_id}")
                continue
            if _sha256(private_path) != entry.get("private_sha256"):
                reasons.append(f"state_snapshot_hash_mismatch:{state_id}")
                continue
            if _sha256(public_path) != entry.get("public_sha256"):
                reasons.append(f"state_projection_hash_mismatch:{state_id}")
                continue
            state_ids.append(state_id)
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"malformed_state_index_entry:{type(exc).__name__}")
    if len(state_ids) != len(set(state_ids)):
        reasons.append("duplicate_state_ids_in_index")
        state_ids = sorted(set(state_ids))
    if index.get("selected_count") != len(index["states"]):
        reasons.append("selected_count_differs_from_index_entries")
    return {"state_ids": state_ids, "index": index, "path": index_path}, reasons


def _read_events_candidate_count(events_path: Path, decision_time_s):
    decision_id = f"replan-decision-{decision_time_s}"
    matches = []
    try:
        with events_path.open("rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return None, [f"invalid_events_jsonl_line:{line_number}"]
                if not isinstance(event, dict):
                    return None, [f"invalid_events_jsonl_record:{line_number}"]
                if (event.get("event") == "candidate_generation"
                        and event.get("decision_id") == decision_id
                        and event.get("branch") == "replan"):
                    count = event.get("candidate_count")
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        matches.append(count)
                    else:
                        return None, ["initial_candidate_count_missing_or_invalid"]
    except OSError as exc:
        return None, [f"events_unreadable:{type(exc).__name__}"]
    if len(matches) != 1:
        return None, [f"initial_candidate_generation_count:{len(matches)}"]
    return matches[0], []


def _load_reduction(path: Path):
    value, error = _optional_json(path)
    if error:
        return None, [f"reduction_{error}"]
    if not isinstance(value, dict):
        return None, ["reduction_malformed"]
    if value.get("status") != "PASS" or value.get("valid_mask") is not True:
        reasons = value.get("reasons") or value.get("unknown_reason")
        if not isinstance(reasons, list):
            reasons = [reasons] if reasons else [
                f"reduction_status_{value.get('status', 'missing')}"]
        return value, [f"reduction_{reason}" for reason in reasons]
    return value, []


def _suffix_evidence(batch_root: Path, row: dict, state_id: str, seed: int,
                     decision_time_s, suffix_summary: dict | None):
    reasons = []
    state_dir = _child(batch_root / row["output_path"], Path("suffixes") / state_id)
    suffix_dir = _child(state_dir, f"suffix_{seed}")
    receipt_path = suffix_dir / "run_receipt.json"
    receipt, receipt_error = _optional_json(receipt_path)
    receipt_complete = bool(receipt and receipt.get("status") == "complete"
                            and receipt.get("actual_isaac_run") is True)
    if not receipt_complete:
        reasons.append(f"suffix_receipt_{receipt_error or 'not_complete_actual_isaac'}")
    receipt_identity_matches = _collector_receipt_matches(row, receipt, "suffix_pair")
    if receipt_complete and not receipt_identity_matches:
        reasons.append("suffix_receipt_run_or_config_identity_mismatch")

    suffix_meta = receipt.get("suffix", {}) if isinstance(receipt, dict) else {}
    audit = suffix_meta.get("audit_interface", {}) if isinstance(suffix_meta, dict) else {}
    is_ordinary = (audit.get("state_kind") == "ordinary_sampled_decision_state"
                   and audit.get("forced") is False
                   and str(audit.get("state_id")) == state_id)
    if receipt_complete and not is_ordinary:
        reasons.append("suffix_state_not_confirmed_ordinary_or_not_forced")
    suffix_identity_matches = (
        isinstance(suffix_meta, dict)
        and suffix_meta.get("seed") == seed
        and suffix_meta.get("decision_time_s") == decision_time_s
    )
    if receipt_complete and not suffix_identity_matches:
        reasons.append("suffix_seed_or_decision_time_mismatch")

    trace_name = receipt.get("trajectory", "trajectory.jsonl.gz") if isinstance(receipt, dict) else "trajectory.jsonl.gz"
    events_name = receipt.get("events", "events.jsonl") if isinstance(receipt, dict) else "events.jsonl"
    try:
        trace_path = _child(suffix_dir, trace_name)
        events_path = _child(suffix_dir, events_name)
        trace_exists = trace_path.is_file()
        events_exists = events_path.is_file()
    except ValueError as exc:
        trace_path = events_path = suffix_dir
        trace_exists = events_exists = False
        reasons.append(str(exc))
    if receipt_complete and not trace_exists:
        reasons.append("trajectory_missing")
    if receipt_complete and not events_exists:
        reasons.append("events_missing")
    worker_complete = (isinstance(suffix_summary, dict)
                       and suffix_summary.get("return_code") == 0
                       and suffix_summary.get("suffix_seed") == seed)
    if receipt_complete and not worker_complete:
        reasons.append("suffix_worker_receipt_missing_or_failed")
    raw_complete = (receipt_complete and receipt_identity_matches and is_ordinary
                    and suffix_identity_matches and worker_complete
                    and trace_exists and events_exists)

    candidate_count = None
    if raw_complete:
        candidate_count, candidate_reasons = _read_events_candidate_count(events_path, decision_time_s)
        reasons.extend(candidate_reasons)

    reduction_path = _child(batch_root, Path("reductions") / row["run_id"] / state_id
                            / f"suffix_{seed}" / "reduction.json")
    reduction, reduction_reasons = _load_reduction(reduction_path)
    if reduction is not None:
        reduction_identity = {
            "scenario_id": row.get("scenario_id"),
            "local_state_id": state_id,
            "state_id": f"{row.get('scenario_id')}::{state_id}",
            "split": row.get("split"),
            "suffix_seed": seed,
            "config_sha256": row.get("config_sha256"),
        }
        if any(reduction.get(key) != expected
               for key, expected in reduction_identity.items()):
            reduction_reasons.append("reduction_run_state_or_config_identity_mismatch")
    if not raw_complete:
        reduction_reasons = [f"not_counted_without_complete_raw_suffix:{reason}"
                             for reason in reduction_reasons]
    reasons.extend(reduction_reasons)
    valid_reduction = raw_complete and reduction is not None and not reduction_reasons

    selected_action = suffix_meta.get("selected_action") if isinstance(suffix_meta, dict) else None
    action_kind = selected_action.get("kind") if isinstance(selected_action, dict) else None
    noop = (action_kind == "noop") if action_kind in {"noop", "uav_transfer"} else None
    if raw_complete and noop is None:
        reasons.append("selected_action_missing_or_unknown_kind")

    resource = receipt.get("resource_usage", {}) if isinstance(receipt, dict) else {}
    catalog_timing = suffix_meta.get("catalog_timing", {}) if isinstance(suffix_meta, dict) else {}
    try:
        trajectory_bytes = trace_path.stat().st_size if raw_complete else None
    except OSError:
        trajectory_bytes = None
        reasons.append("trajectory_stat_unavailable")

    worker_summary = suffix_summary or {}
    return {
        "run_id": row["run_id"], "state_id": state_id, "suffix_seed": seed,
        "raw_complete": raw_complete,
        "reduction_status": (reduction or {}).get("status"),
        "valid_reduction": valid_reduction,
        "selected_action_kind": action_kind,
        "selected_action_noop": noop,
        "initial_candidate_count": candidate_count,
        "gain": {key: reduction.get(key) for key in
                 ("J_exec_keep", "J_exec_replan", "C_increment", "G")}
                if valid_reduction else None,
        "planning_cpu_seconds_incremental": (
            (reduction.get("incremental") or {}).get("planning_cpu_seconds")
            if valid_reduction else None),
        "catalog_cpu_duration_ns": catalog_timing.get("cpu_duration_ns"),
        "catalog_wall_duration_ns": catalog_timing.get("wall_duration_ns"),
        "process_cpu_s": resource.get("process_cpu_s"),
        "wall_duration_s": resource.get("wall_duration_s"),
        "max_rss_bytes": resource.get("max_rss_bytes"),
        "trajectory_compressed_bytes": trajectory_bytes,
        "worker_return_code": worker_summary.get("return_code"),
        "worker_wall_duration_s": worker_summary.get("wall_duration_s"),
        "unknown_reasons": reasons,
        "reduction_path": str(reduction_path),
        "suffix_dir": str(suffix_dir),
    }


def _state_seed_summaries(batch_root: Path, row: dict, scene_data: dict,
                          expected_seeds: tuple[int, ...]):
    state_entries = scene_data.get("states")
    if not isinstance(state_entries, list):
        return [], ["scene_state_list_missing"]
    rep_state = scene_data.get("ordinary_state_ids")
    if rep_state is None:
        return [], ["ordinary_state_index_unknown"]
    by_state = {}
    for item in state_entries:
        if isinstance(item, dict) and item.get("state_id"):
            state_id = str(item["state_id"])
            by_state.setdefault(state_id, item)

    state_summaries = []
    for state_id in rep_state:
        state_receipt = by_state.get(state_id)
        if state_receipt is None:
            scene_data["not_started_state_ids"].append(state_id)
            scene_data["state_pair_counts"]["not_started"] += 1
            scene_data["unknown_reasons"].append(f"state_not_started:{state_id}")
            continue
        worker_rows = state_receipt.get("workers", [])
        worker_by_seed = {str(item.get("suffix_seed")): item for item in worker_rows
                          if isinstance(item, dict) and item.get("suffix_seed") is not None}
        suffix_rows = []
        complete_count = 0
        attempted_seed_count = 0
        failed_count = 0
        for seed in expected_seeds:
            summary = worker_by_seed.get(str(seed))
            if summary is None:
                scene_data["missing_suffixes"].append({"state_id": state_id, "suffix_seed": seed})
            elif summary.get("return_code") is not None:
                attempted_seed_count += 1
                if summary.get("return_code") != 0:
                    failed_count += 1
            row_evidence = _suffix_evidence(batch_root, row, state_id, seed,
                                            scene_data["decision_time_by_state"].get(state_id), summary)
            if row_evidence["raw_complete"]:
                complete_count += 1
            else:
                if not any(item.get("state_id") == state_id and item.get("suffix_seed") == seed
                           for item in scene_data["missing_suffixes"]):
                    scene_data["missing_suffixes"].append({"state_id": state_id, "suffix_seed": seed})
            scene_data["suffix_metrics"].append(row_evidence)
            scene_data["unknown_reasons"].extend(
                f"{state_id}/suffix_{seed}:{reason}" for reason in row_evidence["unknown_reasons"])
            suffix_rows.append(row_evidence)

        if complete_count == len(expected_seeds):
            pair_status = "COMPLETE"
            scene_data["state_pair_counts"]["complete"] += 1
        elif complete_count > 0:
            pair_status = "PARTIAL"
            scene_data["state_pair_counts"]["partial"] += 1
        elif failed_count:
            pair_status = "FAILED"
            scene_data["state_pair_counts"]["failed"] += 1
        else:
            pair_status = "UNKNOWN"
            scene_data["state_pair_counts"]["unknown"] += 1
        state_summaries.append({"state_id": state_id, "pair_status": pair_status,
                                "expected_suffix_seeds": list(expected_seeds),
                                "complete_suffix_count": complete_count,
                                "suffixes": suffix_rows})

    attempted_state_ids = set(by_state)
    unstarted = sorted(set(rep_state) - attempted_state_ids)
    for state_id in scene_data["not_started_state_ids"]:
        if state_id not in unstarted:
            unstarted.append(state_id)
    scene_data["not_started_state_ids"] = sorted(set(unstarted))
    return state_summaries, []


def _scene_metric_values(scene_data: dict):
    suffixes = scene_data["suffix_metrics"]
    raw = [item for item in suffixes if item["raw_complete"]]
    valid = [item for item in suffixes if item["valid_reduction"]]
    known_noops = [item["selected_action_noop"] for item in raw
                   if item["selected_action_noop"] is not None]
    known_candidates = [item["initial_candidate_count"] for item in raw
                        if item["initial_candidate_count"] is not None]
    gains = {name: [item["gain"][name] for item in valid if item["gain"].get(name) is not None]
             for name in ("J_exec_keep", "J_exec_replan", "C_increment", "G")}
    incremental_planning = [item["planning_cpu_seconds_incremental"] for item in valid
                            if item["planning_cpu_seconds_incremental"] is not None]
    resources = scene_data.get("resource_usage") or {}
    completed_bytes = None
    if scene_data["missing_suffixes"] == [] and scene_data.get("all_ordinary_pairs_complete"):
        sizes = [item["trajectory_compressed_bytes"] for item in raw]
        completed_bytes = sum(sizes) if len(sizes) == len(scene_data["ordinary_state_ids"]
                                                        * len(EXPECTED_SUFFIX_SEEDS)) else None
    return {
        "selected_action_noop_rate": _mean([1.0 if value else 0.0 for value in known_noops]),
        "selected_action_noop_count": sum(bool(value) for value in known_noops) if known_noops else None,
        "selected_action_observation_count": len(known_noops),
        "initial_candidate_count_mean_per_suffix": _mean(known_candidates),
        "initial_candidate_count_observation_count": len(known_candidates),
        "gain_suffix_observation_count": len(valid),
        "gain_components_mean_per_suffix": {name: _mean(values) for name, values in gains.items()},
        "planning_cpu_seconds_incremental_mean_per_suffix": _mean(incremental_planning),
        "catalog_cpu_duration_ns_mean_per_suffix": _mean(
            [item["catalog_cpu_duration_ns"] for item in raw]),
        "worker_cpu_seconds_total": resources.get("worker_process_cpu_s_total"),
        "scene_wall_duration_s": resources.get("wall_duration_s"),
        "scene_max_worker_rss_bytes": resources.get("max_worker_rss_bytes"),
        "trajectory_compressed_bytes_observed": _sum(
            [item["trajectory_compressed_bytes"] for item in raw]),
        "trajectory_compressed_bytes_complete_total": completed_bytes,
    }


def _group_metrics(scene_rows: list[dict]):
    metric_names = (
        "selected_action_noop_rate", "selected_action_noop_count",
        "initial_candidate_count_mean_per_suffix", "gain_components_mean_per_suffix",
        "planning_cpu_seconds_incremental_mean_per_suffix",
        "catalog_cpu_duration_ns_mean_per_suffix", "worker_cpu_seconds_total",
        "scene_wall_duration_s", "scene_max_worker_rss_bytes",
        "trajectory_compressed_bytes_complete_total",
    )
    result = {"scene_count": len(scene_rows), "metrics": {}}
    for name in metric_names:
        if name == "gain_components_mean_per_suffix":
            result["metrics"][name] = {
                component: _numeric_summary([
                    row["metrics"][name].get(component) for row in scene_rows
                ]) for component in ("J_exec_keep", "J_exec_replan", "C_increment", "G")
            }
        else:
            result["metrics"][name] = _numeric_summary([
                row["metrics"].get(name) for row in scene_rows
            ])
    return result


def _parse_source_report(value: str):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = path.parts
    if len(parts) != 4 or parts[-1] != "reduction.json":
        return None
    run_id, state_id, suffix_name, _ = parts
    if not suffix_name.startswith("suffix_"):
        return None
    return run_id, state_id, suffix_name.removeprefix("suffix_")


def analyze_batch(registry_path: Path, batch_receipt_path: Path,
                  aggregate_path: Path | None = None):
    registry_path = Path(registry_path).resolve()
    batch_receipt_path = Path(batch_receipt_path).resolve()
    batch_root = batch_receipt_path.parent
    registry = _read_json(registry_path)
    batch = _read_json(batch_receipt_path)
    registry_hash = _sha256(registry_path)
    input_issues = []
    if batch.get("registry_sha256") != registry_hash:
        input_issues.append("batch_receipt_registry_sha256_mismatch")
    registry_rows = registry.get("runs")
    if not isinstance(registry_rows, list):
        raise ValueError("registry has no runs list")
    registry_by_run = {}
    scenario_ids = set()
    for row in registry_rows:
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in registry_by_run:
            input_issues.append("registry_run_id_missing_or_duplicate")
        else:
            registry_by_run[run_id] = row
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id or scenario_id in scenario_ids:
            input_issues.append("registry_scenario_id_missing_or_duplicate")
        scenario_ids.add(scenario_id)
    if registry.get("run_count") != len(registry_rows):
        input_issues.append("registry_run_count_differs_from_rows")
    if len(registry_by_run) != len(registry_rows):
        input_issues.append("registry_run_ids_not_unique")

    if aggregate_path is None:
        aggregate_path = batch.get("end_reduction", {}).get("aggregate_output")
        if aggregate_path:
            aggregate_path = Path(str(aggregate_path))
            if not aggregate_path.is_absolute():
                aggregate_path = batch_root / aggregate_path
    aggregate_path = Path(aggregate_path).resolve() if aggregate_path else None
    aggregate = None
    aggregate_error = "aggregate_path_missing"
    if aggregate_path is not None:
        aggregate, aggregate_error = _optional_json(aggregate_path)

    batch_scenes = batch.get("scenes", [])
    batch_by_run = {}
    duplicate_batch_ids = set()
    for item in batch_scenes if isinstance(batch_scenes, list) else []:
        if not isinstance(item, dict) or not item.get("run_id"):
            input_issues.append("batch_scene_identity_missing")
            continue
        run_id = str(item["run_id"])
        if run_id in batch_by_run:
            duplicate_batch_ids.add(run_id)
        batch_by_run[run_id] = item
    if duplicate_batch_ids:
        input_issues.append("batch_scene_run_id_duplicate")

    scenes = []
    suffix_by_key = {}
    all_suffix_metrics = []
    for row in registry_rows:
        run_id = str(row["run_id"])
        embedded = batch_by_run.get(run_id)
        scene_dir = _child(batch_root, row.get("output_path", ""))
        scene_receipt_path = scene_dir / "scene_receipt.json"
        scene_receipt, scene_error = _optional_json(scene_receipt_path)
        if scene_receipt is None:
            scene_receipt = embedded
        identity_matches = _valid_run_identity(row, scene_receipt)
        batch_identity_matches = _valid_run_identity(row, embedded)
        if not identity_matches:
            input_issues.append(f"scene_identity_missing_or_mismatched:{run_id}")
        if not batch_identity_matches:
            input_issues.append(f"batch_scene_identity_missing_or_mismatched:{run_id}")
        rep_dir = scene_dir / "representative"
        rep_receipt, rep_error = _optional_json(rep_dir / "run_receipt.json")
        rep_identity_matches = _collector_receipt_matches(row, rep_receipt, "representative")
        attempted = bool(embedded and isinstance(embedded.get("representative"), dict)
                         and embedded["representative"].get("return_code") is not None)
        if not attempted and rep_receipt is not None:
            attempted = True
        index_data, index_reasons = _state_index(rep_dir, rep_receipt)
        ordinary_ids = index_data["state_ids"] if index_data else None
        explicit_status = (scene_receipt or {}).get("execution")
        scene_receipt_present = scene_receipt_path.is_file()
        explicit_status_match = (scene_receipt is not None and embedded is not None
                                 and scene_receipt.get("execution") == embedded.get("execution"))

        scene_data = {
            "run_id": run_id, "scenario_id": row.get("scenario_id"),
            "split": row.get("split"), "condition": row.get("condition"),
            "source": row.get("source"), "execution": explicit_status,
            "attempted": attempted, "scene_receipt_present": scene_receipt_present,
            "scene_receipt_identity_matches": identity_matches,
            "batch_scene_identity_matches": batch_identity_matches,
            "scene_receipt_matches_batch_entry": explicit_status_match,
            "representative_status": rep_receipt.get("status") if isinstance(rep_receipt, dict) else None,
            "representative_actual_isaac_run": rep_receipt.get("actual_isaac_run") if isinstance(rep_receipt, dict) else None,
            "representative_identity_matches": rep_identity_matches,
            "ordinary_state_ids": ordinary_ids,
            "ordinary_state_count": len(ordinary_ids) if ordinary_ids is not None else None,
            "state_index_unknown_reasons": index_reasons,
            "not_started_state_ids": [], "missing_suffixes": [],
            "state_pair_counts": {"complete": 0, "failed": 0, "partial": 0,
                                  "not_started": 0, "unknown": 0},
            "suffix_metrics": [], "decision_time_by_state": {},
            "resource_usage": (scene_receipt or {}).get("resource_usage"),
            "unknown_reasons": [],
        }
        if ordinary_ids is not None:
            for entry in index_data["index"]["states"]:
                scene_data["decision_time_by_state"][Path(entry["private_file"]).stem] = entry.get("decision_time_s")
        if scene_error:
            scene_data["unknown_reasons"].append(f"scene_receipt_{scene_error}")
        if rep_error:
            scene_data["unknown_reasons"].append(f"representative_receipt_{rep_error}")
        scene_data["unknown_reasons"].extend(index_reasons)
        scene_data["no_ordinary_states"] = ordinary_ids == [] if ordinary_ids is not None else None

        state_summaries = []
        if ordinary_ids is not None and scene_receipt is not None:
            state_summaries, state_errors = _state_seed_summaries(
                batch_root, row, scene_data, tuple(registry.get("suffix_seeds", EXPECTED_SUFFIX_SEEDS)))
            scene_data["unknown_reasons"].extend(state_errors)
        elif ordinary_ids is None:
            scene_data["unknown_reasons"].append("ordinary_state_inventory_unknown")
        scene_state_list = (scene_receipt or {}).get("states", [])
        attempted_ids = {str(item.get("state_id")) for item in scene_state_list
                         if isinstance(item, dict) and item.get("state_id")}
        if ordinary_ids is not None:
            scene_data["not_started_state_ids"] = sorted(set(ordinary_ids) - attempted_ids)
            if scene_data["not_started_state_ids"] and scene_data["state_pair_counts"]["not_started"] == 0:
                scene_data["state_pair_counts"]["not_started"] = len(scene_data["not_started_state_ids"])
                scene_data["unknown_reasons"].extend(
                    f"state_not_started:{state_id}" for state_id in scene_data["not_started_state_ids"])

        state_pair_total = len(ordinary_ids) if ordinary_ids is not None else None
        complete_pairs = scene_data["state_pair_counts"]["complete"]
        scene_data["all_ordinary_pairs_complete"] = (
            state_pair_total is not None and state_pair_total > 0
            and complete_pairs == state_pair_total and not scene_data["missing_suffixes"]
            and not scene_data["not_started_state_ids"])
        scene_data["state_summaries"] = state_summaries
        scene_data["actual_complete"] = bool(
            scene_receipt_present and embedded is not None and explicit_status_match
            and identity_matches and batch_identity_matches and explicit_status == "COMPLETE"
            and rep_receipt and rep_receipt.get("status") == "complete"
            and rep_receipt.get("actual_isaac_run") is True
            and rep_identity_matches
            and scene_data["all_ordinary_pairs_complete"])
        if explicit_status == "COMPLETE" and not scene_data["actual_complete"]:
            scene_data["unknown_reasons"].append("scene_not_fully_confirmed_by_raw_receipts")

        scene_data["metrics"] = _scene_metric_values(scene_data)
        scenes.append(scene_data)
        all_suffix_metrics.extend(scene_data["suffix_metrics"])
        for suffix in scene_data["suffix_metrics"]:
            suffix_by_key[(run_id, suffix["state_id"], str(suffix["suffix_seed"]))] = suffix

    status_counts = Counter(scene.get("execution") or "UNKNOWN" for scene in scenes)
    attempted_scenes = sum(bool(scene["attempted"]) for scene in scenes)
    complete_scenes = sum(bool(scene["actual_complete"]) for scene in scenes)
    failed_scenes = status_counts.get("FAILED", 0)
    partial_scenes = status_counts.get("PARTIAL_FAILED", 0)
    unconfirmed_scenes = len(scenes) - complete_scenes - failed_scenes - partial_scenes
    sampled_ordinary_states = sum(scene["ordinary_state_count"] or 0
                                  for scene in scenes if scene["ordinary_state_count"] is not None)
    no_ordinary_scenes = sum(scene["no_ordinary_states"] is True for scene in scenes)
    unknown_ordinary_scenes = sum(scene["no_ordinary_states"] is None for scene in scenes)
    unstarted_state_count = sum(len(scene["not_started_state_ids"]) for scene in scenes)
    missing_suffix_count = sum(len(scene["missing_suffixes"]) for scene in scenes)

    aggregate_rows = aggregate.get("states", []) if isinstance(aggregate, dict) else []
    matched_labels = []
    aggregate_unknown_reasons = Counter()
    invalid_aggregate_states = []
    seen_state_keys = set()
    for state in aggregate_rows if isinstance(aggregate_rows, list) else []:
        reasons = []
        source_reports = state.get("source_reports", [])
        parsed = [_parse_source_report(path) for path in source_reports]
        if not parsed or any(item is None for item in parsed):
            reasons.append("aggregate_source_report_path_malformed")
        run_ids = {item[0] for item in parsed if item is not None}
        local_ids = {item[1] for item in parsed if item is not None}
        seeds = {item[2] for item in parsed if item is not None}
        if len(run_ids) != 1 or len(local_ids) != 1 or len(seeds) != len(parsed):
            reasons.append("aggregate_source_reports_not_one_scene_and_distinct_suffixes")
        run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
        local_id = next(iter(local_ids)) if len(local_ids) == 1 else None
        row = registry_by_run.get(run_id) if run_id else None
        if row is None:
            reasons.append("aggregate_source_reports_unmapped_to_registry_run_id")
        elif (state.get("scenario_id") != row.get("scenario_id")
              or state.get("split") != row.get("split")
              or local_id != state.get("local_state_id")):
            reasons.append("aggregate_state_identity_differs_from_registry_or_paths")
        if len(parsed) != 2 or seeds != {str(seed) for seed in EXPECTED_SUFFIX_SEEDS}:
            reasons.append("aggregate_state_requires_expected_two_suffix_reports")
        if state.get("valid_mask") is not True:
            reasons.extend(str(reason) for reason in state.get("unknown_reason", []))
        for parsed_row in parsed:
            if parsed_row is None:
                continue
            suffix_evidence = suffix_by_key.get((parsed_row[0], parsed_row[1], parsed_row[2]))
            if suffix_evidence is None or not suffix_evidence["raw_complete"]:
                reasons.append("aggregate_state_suffix_lacks_complete_raw_receipt")
            elif not suffix_evidence["valid_reduction"]:
                reasons.append("aggregate_state_suffix_reduction_not_pass")
        state_key = (run_id, local_id) if run_id and local_id else None
        if state_key in seen_state_keys:
            reasons.append("duplicate_valid_state_for_run_id")
        if state_key:
            seen_state_keys.add(state_key)
        if not reasons and row is not None:
            state_gain = _finite_number(state.get("gain_mean"))
            if state_gain is None:
                reasons.append("aggregate_gain_mean_missing_or_invalid")
            else:
                raw_gains = [suffix_by_key[(run_id, local_id, str(seed))]["gain"]["G"]
                             for seed in EXPECTED_SUFFIX_SEEDS]
                if any(_finite_number(value) is None for value in raw_gains):
                    reasons.append("suffix_gain_missing_for_aggregate_state")
                elif abs(state_gain - sum(float(value) for value in raw_gains) / 2.0) > 1e-9:
                    reasons.append("aggregate_gain_mean_differs_from_raw_suffix_reductions")
        if reasons:
            for reason in reasons:
                aggregate_unknown_reasons[reason] += 1
            invalid_aggregate_states.append({"state_id": state.get("state_id"),
                                             "run_id": run_id, "local_state_id": local_id,
                                             "reasons": sorted(set(reasons))})
            continue
        matched_labels.append({"run_id": run_id, "scenario_id": row["scenario_id"],
                               "local_state_id": local_id, "split": row["split"],
                               "condition": row["condition"], "source": row["source"],
                               "gain_mean": state_gain,
                               "class": "positive" if state_gain > 0 else "nonpositive"})

    label_counts = Counter(item["class"] for item in matched_labels)
    class_split_counts = {label: {split: 0 for split in REQUIRED_SPLITS}
                          for label in ("positive", "nonpositive")}
    for item in matched_labels:
        if item["split"] in REQUIRED_SPLITS:
            class_split_counts[item["class"]][item["split"]] += 1
    count_gate = all(label_counts[label] >= LABEL_TARGET_PER_CLASS
                     for label in ("positive", "nonpositive"))
    split_gate = all(class_split_counts[label][split] > 0
                     for label in class_split_counts for split in REQUIRED_SPLITS)
    aggregate_state_counts = {
        "aggregate_state_row_count": len(aggregate_rows) if isinstance(aggregate_rows, list) else None,
        "verified_unique_ordinary_state_count": len(matched_labels),
        "positive_count": label_counts["positive"],
        "nonpositive_count": label_counts["nonpositive"],
        "positive_by_split": class_split_counts["positive"],
        "nonpositive_by_split": class_split_counts["nonpositive"],
        "at_least_30_per_class": count_gate,
        "train_validation_coverage_met": split_gate,
        "reported_aggregate_status": aggregate.get("status") if isinstance(aggregate, dict) else None,
        "reported_aggregate_met_30_30": aggregate.get("met_30_30") if isinstance(aggregate, dict) else None,
        "invalid_or_unknown_state_count": len(invalid_aggregate_states),
        "invalid_or_unknown_reasons": dict(sorted(aggregate_unknown_reasons.items())),
        "invalid_or_unknown_states": invalid_aggregate_states,
        "verified_states": matched_labels,
    }

    per_scene_metrics = [{
        "run_id": scene["run_id"], "scenario_id": scene["scenario_id"],
        "split": scene["split"], "condition": scene["condition"], "source": scene["source"],
        "execution": scene["execution"], "actual_complete": scene["actual_complete"],
        "metrics": scene["metrics"],
    } for scene in scenes]
    by_condition = {key: _group_metrics([item for item in per_scene_metrics
                                         if item["condition"] == key])
                    for key in sorted({str(row.get("condition")) for row in registry_rows})}
    by_source = {key: _group_metrics([item for item in per_scene_metrics if item["source"] == key])
                 for key in sorted({str(row.get("source")) for row in registry_rows})}
    unknown_reasons = Counter()
    for scene in scenes:
        unknown_reasons.update(scene["unknown_reasons"])
    if aggregate_error:
        unknown_reasons[f"aggregate_{aggregate_error}"] += 1
    unknown_reasons.update(aggregate_unknown_reasons)
    if not aggregate_rows and aggregate is not None:
        unknown_reasons["aggregate_has_no_state_rows"] += 1

    all_scenes_complete = len(scenes) == len(registry_rows) and complete_scenes == len(registry_rows)
    terminal_batch_status = batch.get("execution") in {"COMPLETE", "PARTIAL_OR_FAILED"}
    terminal_scene_statuses = {"COMPLETE", "FAILED", "PARTIAL_FAILED"}
    batch_scene_ids = set(batch_by_run)
    registry_scene_ids = set(registry_by_run)
    all_scene_attempts_recorded = (
        isinstance(batch_scenes, list)
        and len(batch_scenes) == len(registry_rows)
        and batch_scene_ids == registry_scene_ids
        and not duplicate_batch_ids
        and all(isinstance(scene, dict)
                and scene.get("execution") in terminal_scene_statuses
                and scene.get("run_id") in registry_scene_ids
                and _valid_run_identity(registry_by_run[scene["run_id"]], scene)
                for scene in batch_scenes)
    )
    aggregate_has_30_30 = (isinstance(aggregate, dict)
                           and aggregate_error is None
                           and aggregate.get("met_30_30") is True)
    data_target_met = (terminal_batch_status and all_scene_attempts_recorded
                       and count_gate and split_gate and aggregate_has_30_30)
    report = {
        "schema": "isaac_roi_collection_analysis_v1",
        "status": "PASS" if not input_issues else "INPUT_MISMATCH",
        "input_issues": input_issues,
        "inputs": {
            "registry_path": str(registry_path), "registry_sha256": registry_hash,
            "batch_receipt_path": str(batch_receipt_path),
            "aggregate_path": str(aggregate_path) if aggregate_path else None,
        },
        "evidence_scope": {
            "actual_batch_started_at_utc": batch.get("started_at_utc"),
            "actual_batch_completed_at_utc": batch.get("completed_at_utc"),
            "registry_design_status": batch.get("registry_design_status"),
            "batch_execution_status": batch.get("execution"),
            "analysis_is_descriptive_engineering_diagnostics_only": True,
            "inferential_tests": "NOT_RUN",
            "method_status": "INCONCLUSIVE",
            "method_run_status": "NOT_RUN",
            "world_model_status": "NOT_RUN",
            "actual_isaac_run_count": complete_scenes,
            "actual_scope_limitation": "Counts below require complete receipt evidence; no missing metric is imputed.",
        },
        "scene_coverage": {
            "planned_scene_count": len(registry_rows),
            "batch_receipt_scene_row_count": len(batch_scenes) if isinstance(batch_scenes, list) else None,
            "acquisition_attempted_scene_count": attempted_scenes,
            "confirmed_complete_scene_count": complete_scenes,
            "explicit_failed_scene_count": failed_scenes,
            "explicit_partial_scene_count": partial_scenes,
            "unconfirmed_or_unknown_scene_count": unconfirmed_scenes,
            "explicit_execution_status_counts": dict(sorted(status_counts.items())),
            "batch_terminal": terminal_batch_status,
            "all_scene_attempts_recorded": all_scene_attempts_recorded,
        },
        "state_coverage": {
            "sampled_ordinary_state_count": sampled_ordinary_states,
            "scenes_with_no_ordinary_states": no_ordinary_scenes,
            "scenes_with_unknown_ordinary_state_inventory": unknown_ordinary_scenes,
            "unstarted_ordinary_state_count": unstarted_state_count,
            "missing_suffix_count": missing_suffix_count,
            "complete_state_pair_count": sum(scene["state_pair_counts"]["complete"] for scene in scenes),
            "failed_state_pair_count": sum(scene["state_pair_counts"]["failed"] for scene in scenes),
            "partial_state_pair_count": sum(scene["state_pair_counts"]["partial"] for scene in scenes),
            "unknown_state_pair_count": sum(scene["state_pair_counts"]["unknown"] for scene in scenes),
            "state_label_coverage": aggregate_state_counts,
        },
        "data_status": "DATA_TARGET_MET" if data_target_met else "INCOMPLETE_OR_INSUFFICIENT",
        "remaining_experiment_items": [
            "Retain missing or invalid raw evidence as unknown; preserve the frozen no-retry and no-replacement design.",
            "Complete independent trajectory audit and verify the requested label coverage against actual suffix reductions.",
            "Run method training and frozen ID test only after acquisition and audit close; method remains INCONCLUSIVE / NOT_RUN here.",
            "Run the sealed OOD evaluation only after model and threshold freeze; world-model work remains NOT_RUN.",
        ],
        "unknown_reason_counts": dict(sorted(unknown_reasons.items())),
        "metrics": {
            "aggregation_unit": "registry run_id scene; within-scene suffix/state rows are descriptive repeats, not independent scenes",
            "by_condition": by_condition,
            "by_source": by_source,
            "per_scene": per_scene_metrics,
        },
        "scene_details": scenes,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-receipt", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path,
                        help="aggregate.json path; defaults to end_reduction.aggregate_output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze_batch(args.registry, args.batch_receipt, args.aggregate)
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        report = {"schema": "isaac_roi_collection_analysis_v1", "status": "INPUT_ERROR",
                  "error": f"{type(exc).__name__}:{exc}", "data_status": "UNKNOWN",
                  "evidence_scope": {"method_status": "INCONCLUSIVE",
                                     "method_run_status": "NOT_RUN",
                                     "world_model_status": "NOT_RUN"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "data_status": report.get("data_status"),
                      "planned_scene_count": report.get("scene_coverage", {}).get("planned_scene_count"),
                      "confirmed_complete_scene_count": report.get("scene_coverage", {}).get("confirmed_complete_scene_count"),
                      "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
