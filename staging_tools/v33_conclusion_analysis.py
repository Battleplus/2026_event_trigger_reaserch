#!/usr/bin/env python3
"""Independent, read-only analysis of committed V33 collection shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import gzip
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "civilian"))
import gain_oracle  # noqa: E402
import fixed_replanner  # noqa: E402


REGION = re.compile(r"(?:^|-)r(\d{2})(?:-|$)")
REGION_ORDINAL_TO_COORD = {
    "00": "r00", "01": "r01", "02": "r02",
    "03": "r10", "04": "r11", "05": "r12",
    "06": "r20", "07": "r21", "08": "r22",
}


def _read_shard(path: Path, *, expected_binding: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with gzip.open(path, "rt", encoding="ascii") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    headers = [row for row in rows if row.get("schema") == "gain-gate-v33-shard-header/1"]
    if len(headers) != 1:
        raise ValueError(f"{path}: expected exactly one header, got {len(headers)}")
    marker = path.with_suffix("").with_suffix(".commit.json")
    if not marker.is_file():
        raise ValueError(f"{path}: missing commit marker")
    import v3_shards
    commit = v3_shards.verify_scenario_shard(marker, expected_binding=expected_binding)
    scenario_id = str(headers[0].get("scenario", {}).get("parent_id"))
    if commit.get("scenario_id") != scenario_id:
        raise ValueError(f"{path}: commit/header scenario mismatch")
    return headers[0], [row["training_row"] for row in rows if row.get("schema") == "gain-gate-v33-shard-row/1"], commit


def _region(value: Any) -> str | None:
    match = REGION.search(str(value))
    return match.group(1) if match else None


def _row_evidence_sha256(row: Mapping[str, Any]) -> str:
    canonical = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _audit_live_safety_receipt(trajectory: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace = trajectory.get("trajectory_trace")
    actual_unresolved: int | None = None
    actual_reasons: list[Any] | None = None
    unknown_resolution_count: int | None = None
    if isinstance(trace, list):
        dispositions = [step["safety_disposition"] for step in trace
                       if isinstance(step, Mapping) and isinstance(step.get("safety_disposition"), Mapping)]
        unresolved = [item for item in dispositions if item.get("resolved") is False]
        actual_unresolved = len(unresolved)
        actual_reasons = [item.get("unknown_reason") for item in unresolved]
        unknown_resolution_count = sum("resolved" not in item or not isinstance(item.get("resolved"), bool)
                                       for item in dispositions)
    reported = trajectory.get("live_unresolved_safety_disposition_count")
    reported = reported if type(reported) is int and reported >= 0 else None
    forced_rows = [row for row in rows if row.get("forced") is True]
    forced_ids = {str(row.get("state_id")) for row in forced_rows if row.get("state_id")}
    forced_actual = len(forced_ids) if forced_ids else len(forced_rows)
    forced_reported = trajectory.get("forced_state_count")
    forced_reported = forced_reported if type(forced_reported) is int and forced_reported >= 0 else None
    return {
        "actual_unresolved_disposition_count": actual_unresolved,
        "reported_unresolved_disposition_count": reported,
        "unresolved_count_matches_summary": (actual_unresolved == reported) if actual_unresolved is not None and reported is not None else None,
        "actual_unresolved_unknown_reasons": actual_reasons,
        "unknown_resolution_disposition_count": unknown_resolution_count,
        "trajectory_trace_available": isinstance(trace, list),
        "actual_forced_state_count": forced_actual,
        "reported_forced_state_count": forced_reported,
        "forced_count_matches_summary": forced_actual == forced_reported if forced_reported is not None else None,
        "receipt_interpretation": "live safety-disposition receipt audit only; not proof of all hard constraints or method safety",
    }


def _cross_region_routes(action: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = []
    ops = action.get("queue_operations", [])
    for op in ops:
        task = str(op.get("task_id", ""))
        to_platform = op.get("to_platform")
        from_platform = op.get("from_platform")
        receiver = _region(to_platform)
        task_region = _region(task)
        donor = _region(from_platform) if from_platform else None
        cross_home = receiver is not None and task_region is not None and receiver != task_region
        cross_transfer = donor is not None and receiver is not None and donor != receiver
        if cross_home or cross_transfer:
            kind = str(op.get("op", "unknown_operation"))
            route_class = ("home_region_insert" if kind == "insert" and cross_home and not cross_transfer
                           else "donor_transfer" if kind == "transfer" and cross_transfer
                           else "other_cross_region_operation")
            routes.append({"kind": kind, "channel": "queue_operation", "task_id": task,
                           "task_region": task_region, "receiver_region": receiver, "donor_region": donor,
                           "route_class": route_class,
                           "receiver_platform_id": to_platform, "donor_platform_id": from_platform})
    for dispatch in action.get("dispatches", []):
        task = str(dispatch.get("task_id", ""))
        platform = dispatch.get("platform_id")
        task_region = _region(task)
        receiver = _region(platform)
        if task_region is not None and receiver is not None and receiver != task_region:
            routes.append({"kind": "direct_dispatch", "channel": "dispatch", "task_id": task,
                           "task_region": task_region, "receiver_region": receiver, "donor_region": None,
                           "route_class": "direct_cross_region_dispatch",
                           "receiver_platform_id": platform, "donor_platform_id": None})
    return routes


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    return {"noop": action.get("noop") is True, "scope": action.get("scope"),
            "queue_operations": action.get("queue_operations", []), "dispatches": action.get("dispatches", []),
            "cross_region_routes": _cross_region_routes(action)}


def _task_region_changes(pair: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for arm in ("keep", "replan"):
        trace = pair[arm]["trace"]
        start = {str(x["id"]): x for x in trace["start_truth"]["tasks"]}
        end = {str(x["id"]): x for x in trace["raw_truth"]["tasks"]}
        for task_id in sorted(set(start) & set(end)):
            reg = _region(task_id)
            if reg is None:
                continue
            row = rows.setdefault(task_id, {"region": reg})
            task = end[task_id]
            row[f"{arm}_status"] = task.get("status")
            row[f"{arm}_progress"] = task.get("progress")
            row[f"{arm}_completed"] = task.get("status") == "completed"
            row[f"{arm}_remaining_fraction"] = 0.0 if task.get("status") == "completed" else max(0.0, min(1.0, 1.0 - float(task.get("progress", 0.0))))
    regions: dict[str, dict[str, float]] = defaultdict(lambda: Counter())
    for row in rows.values():
        reg = row["region"]
        for arm in ("keep", "replan"):
            regions[reg][f"{arm}_completed"] += int(row.get(f"{arm}_completed", False))
            regions[reg][f"{arm}_remaining_fraction_sum"] += float(row.get(f"{arm}_remaining_fraction", 0.0))
        regions[reg]["task_count"] += 1
    result = {}
    for reg, vals in sorted(regions.items()):
        result[reg] = {key: float(value) for key, value in vals.items()}
        result[reg]["completed_delta_replan_minus_keep"] = vals["replan_completed"] - vals["keep_completed"]
        result[reg]["remaining_fraction_delta_keep_minus_replan"] = vals["keep_remaining_fraction_sum"] - vals["replan_remaining_fraction_sum"]
    return {"by_task": rows, "by_region": result}


def _branch_components(pair: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for arm in ("keep", "replan"):
        cost = pair[arm]["cost"]
        out[arm] = {"J_total": cost["J_total"], "J_exec": cost["J_exec"], "C_replan": cost["C_replan"],
                    "C_safety": cost["C_safety"], "execution_components": cost["execution_components"],
                    "replan_components": cost["replan_components"], "safety_components": cost["safety_components"]}
    return out


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _scenario_gain_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted({state["split"] for state in states}):
        by_scenario: dict[str, list[float]] = defaultdict(list)
        for state in states:
            if state["split"] == split and state["valid"] and not state["forced"] and not state["empty"]:
                by_scenario[state["scenario_id"]].append(float(state["gain"]))
        means = {scenario: sum(values) / len(values) for scenario, values in sorted(by_scenario.items())}
        scenario_values = list(means.values())
        if len(scenario_values) < 2:
            result[split] = {"scenario_count": len(scenario_values), "scenario_means": means,
                             "mean_of_scenario_means": (scenario_values[0] if scenario_values else None),
                             "bootstrap_95_ci": None, "ci_status": "not_estimable_n_lt_2"}
            continue
        rng = random.Random(33303 + (0 if split == "train" else 1))
        boot = [sum(rng.choice(scenario_values) for _ in scenario_values) / len(scenario_values) for _ in range(10000)]
        result[split] = {"scenario_count": len(means), "scenario_means": means,
                         "mean_of_scenario_means": sum(scenario_values) / len(scenario_values),
                         "bootstrap_95_ci": [_quantile(boot, 0.025), _quantile(boot, 0.975)],
                         "method": "percentile bootstrap of scenario means; states averaged within scenario; 10000 replicates; seed=33303",
                         "interpretation": "descriptive label-gain summary, not a strategy or cross-region treatment-effect interval",
                         "ci_status": "estimable"}
    return result


def _outcome_decomposition(states: list[dict[str, Any]]) -> dict[str, Any]:
    cost_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    region_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pairs = 0
    completed_tasks = 0
    completed_tasks_below_full_progress = 0
    for state in states:
        if not state["valid"] or state["forced"] or state["empty"]:
            continue
        for pair_i, components in enumerate(state["branch_components"]):
            pairs += 1
            for arm in ("keep", "replan"):
                cost = components[arm]
                for key in ("J_total", "J_exec", "C_replan", "C_safety"):
                    cost_values[key][arm].append(float(cost[key]))
                for group in ("execution_components", "replan_components", "safety_components"):
                    for key, value in cost[group].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            cost_values[f"{group}.{key}"][arm].append(float(value))
            effects = state["region_effects"][pair_i]
            for region, values in effects["by_region"].items():
                for key, value in values.items():
                    region_values[region][key].append(float(value))
            for task in effects["by_task"].values():
                if task.get("replan_status") == "completed":
                    completed_tasks += 1
                    completed_tasks_below_full_progress += int(float(task.get("replan_progress") or 0.0) < 1.0)
                if task.get("keep_status") == "completed":
                    completed_tasks += 1
                    completed_tasks_below_full_progress += int(float(task.get("keep_progress") or 0.0) < 1.0)
    costs = {}
    for key, arms in sorted(cost_values.items()):
        means = {arm: (sum(vals) / len(vals) if vals else None) for arm, vals in arms.items()}
        if key in {"J_total", "J_exec", "C_replan", "C_safety"}:
            means["delta_replan_minus_keep"] = None if means.get("replan") is None or means.get("keep") is None else means["replan"] - means["keep"]
        costs[key] = {"pair_count": len(arms.get("keep", [])), "mean": means}
    regions = {}
    for region, values in sorted(region_values.items()):
        regions[region] = {key: {"count": len(vals), "mean": sum(vals) / len(vals) if vals else None}
                           for key, vals in sorted(values.items())}
    return {"ordinary_valid_suffix_pairs": pairs,
            "completed_endpoint_tasks": completed_tasks,
            "completed_endpoint_tasks_below_full_progress": completed_tasks_below_full_progress,
            "regional_endpoint_effect_means": regions,
            "global_cost_component_means": costs}


def _cross_selected_decomposition(states: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [s for s in states if s["valid"] and not s["forced"] and not s["empty"]
                and s["selected_action"]["cross_region_routes"]]
    base = _outcome_decomposition(selected)
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for state in selected:
        by_scenario[state["scenario_id"]].append(float(state["gain"]))
    scenario_means = {sid: sum(vals) / len(vals) for sid, vals in sorted(by_scenario.items())}
    vals = list(scenario_means.values())
    ci = None
    if len(vals) >= 2:
        rng = random.Random(33333)
        boot = [sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(10000)]
        ci = [_quantile(boot, .025), _quantile(boot, .975)]
    route_counts = Counter(route.get("route_class", "other_cross_region_operation")
                           for state in selected for route in state["selected_action"]["cross_region_routes"])
    role_regions: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overlap_regions: Counter[str] = Counter()
    for state in selected:
        routes = state["selected_action"]["cross_region_routes"]
        roles: dict[str, set[str]] = defaultdict(set)
        for route in routes:
            if route.get("task_region") is not None:
                roles["recipient_task_region"].add(str(route["task_region"]))
            if route.get("receiver_region") is not None:
                roles["platform_home_region"].add(str(route["receiver_region"]))
            if route.get("kind") == "transfer" and route.get("donor_region") is not None:
                roles["donor_home_region"].add(str(route["donor_region"]))
        involved = set().union(*roles.values()) if roles else set()
        memberships: dict[str, set[str]] = defaultdict(set)
        for role, regions in roles.items():
            for region in regions:
                memberships[region].add(role)
        for region, region_roles in memberships.items():
            if len(region_roles) > 1:
                overlap_regions[region] += 1
        for effects in state["region_effects"]:
            region_rows = effects["by_region"]
            pair_roles = {role: regions & set(region_rows) for role, regions in roles.items()}
            pair_roles["other_region"] = set(region_rows) - involved
            for role, regions in pair_roles.items():
                for region in regions:
                    values = region_rows[region]
                    for metric, value in values.items():
                        role_regions[role][metric].setdefault(region, []).append(float(value))
    role_region_summary = {}
    for role, metrics in sorted(role_regions.items()):
        role_region_summary[role] = {
            metric: {region: {"count": len(values), "mean": sum(values) / len(values) if values else None}
                     for region, values in sorted(regions.items())}
            for metric, regions in sorted(metrics.items())}
    state_rows = [{"state_id": s["state_id"], "scenario_id": s["scenario_id"], "split": s["split"],
                   "gain": s["gain"], "selected_action": s["selected_action"],
                   "public_cost_estimate": s["public_cost_estimate"], "pair_count": s["pair_count"],
                   "branch_components": s["branch_components"], "region_effects": s["region_effects"]}
                  for s in selected]
    return {"unique_state_count": len(selected), "scenario_count": len(vals), "scenario_means": scenario_means,
            "mean_of_scenario_means": sum(vals) / len(vals) if vals else None,
            "bootstrap_95_ci": ci, "ci_status": "descriptive_estimable" if ci else "not_estimable_n_lt_2",
            "interpretation": "descriptive selected-cross-region subset; not a pure cross-region causal effect",
            "selected_route_class_counts": dict(route_counts),
            "regional_role_effects": role_region_summary,
            "regions_with_overlapping_roles": dict(overlap_regions),
            "outcomes": base, "state_details": state_rows}


def analyze(run_dirs: list[Path]) -> dict[str, Any]:
    import v3_shards
    expected_binding = v3_shards.current_binding()
    for run_dir in run_dirs:
        progress_path = run_dir / "COLLECTION_PROGRESS.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("binding") is not None and progress["binding"] != expected_binding:
                raise ValueError(f"{progress_path}: collection progress binding differs from frozen package/config binding")
    unique_shards: dict[str, tuple[Path, dict[str, Any]]] = {}
    duplicate_shards = 0
    for run_dir in run_dirs:
        marker_paths = sorted(run_dir.rglob("*.commit.json"))
        if not marker_paths:
            raise ValueError(f"no committed shard markers under {run_dir}")
        for marker_path in marker_paths:
            commit = json.loads(marker_path.read_text(encoding="utf-8"))
            if commit.get("binding") != dict(expected_binding):
                raise ValueError(f"{marker_path}: committed shard binding differs from frozen package/config binding")
            path = marker_path.with_name(str(commit.get("shard_file", "")))
            scenario_id = str(commit.get("scenario_id", ""))
            if not scenario_id:
                raise ValueError(f"{marker_path}: commit marker missing scenario_id")
            previous = unique_shards.get(scenario_id)
            if previous is not None:
                if previous[1].get("compressed_sha256") != commit.get("compressed_sha256"):
                    raise ValueError(f"conflicting shard content for scenario {scenario_id}")
                duplicate_shards += 1
                continue
            unique_shards[scenario_id] = (path, commit)
    shard_paths = [item[0] for _, item in sorted(unique_shards.items())]
    if not shard_paths:
        raise ValueError("no unique committed shards found")
    states: dict[str, dict[str, Any]] = {}
    state_row_digests: dict[str, str] = {}
    selected_rows: list[dict[str, Any]] = []
    candidate_counter: Counter[str] = Counter()
    candidate_state_coverage: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"ordinary": set(), "forced": set()})
    selected_state_coverage: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"ordinary": set(), "forced": set()})
    noop_reasons: Counter[str] = Counter()
    nonnoop_reasons: Counter[str] = Counter()
    selection_checks: Counter[str] = Counter()
    class_scenarios: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    scenario_headers: dict[str, dict[str, Any]] = {}
    shard_records = []
    resource_receipts = []
    cost_semantics = {"completed_residual_penalty": "zero (completed endpoint tasks are scored as complete and skipped in residual accumulation)",
                      "unfinished_residual_penalty": "weighted clipped 1-progress fraction", "source": "package/civilian/cost_accounting.py and package/civilian/gain_oracle.py"}

    lineage_records = []
    for run_dir in run_dirs:
        lineage_path = run_dir / "REUSED_SHARDS.json"
        if lineage_path.is_file():
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            refs = lineage.get("shards", [])
            if len(refs) != int(lineage.get("source_verified_count", -1)):
                raise ValueError(f"{lineage_path}: reused shard count mismatch")
            for ref in refs:
                scenario_id = str(ref.get("scenario_id", ""))
                pair = unique_shards.get(scenario_id)
                if pair is None or pair[1].get("compressed_sha256") != ref.get("compressed_sha256"):
                    raise ValueError(f"{lineage_path}: imported shard not present/matching: {scenario_id}")
            lineage_records.append({"run_id": run_dir.name, "imported_from": lineage.get("source_run_id"),
                                    "source_verified_count": len(refs), "verified_against_run_shards": True})

    for path in shard_paths:
        header, rows, commit = _read_shard(path, expected_binding=expected_binding)
        if isinstance(commit.get("runtime_receipt"), Mapping):
            resource_receipts.append(commit["runtime_receipt"])
        scenario = header["scenario"]
        scenario_id = str(scenario.get("parent_id"))
        trajectory = header.get("trajectory", {})
        scenario_headers[scenario_id] = {"split": scenario.get("split"), "condition": scenario.get("condition"),
                                         "target_density": scenario.get("targets_per_region"), "task_mode": scenario.get("task_mode"),
                                         "seed_variant": scenario.get("seed_variant"),
                                         "trajectory_source": trajectory.get("trajectory_source"),
                                         "ordinary_candidate_count": trajectory.get("ordinary_candidate_count"),
                                         "selected_state_count": trajectory.get("selected_state_count"),
                                         "elapsed_ticks": trajectory.get("elapsed_ticks"),
                                         "trajectory_terminated": trajectory.get("terminated"),
                                         "trajectory_truncated": trajectory.get("truncated"),
                                         "platform_usage": trajectory.get("platform_usage"),
                                         "live_safety_receipt_audit": _audit_live_safety_receipt(trajectory, rows)}
        shard_records.append({"file_name": path.name, "source_run_id": commit.get("run_id"), "scenario_id": scenario_id,
                              "split": scenario.get("split"), "rows": len(rows), "compressed_sha256": commit.get("compressed_sha256")})
        for row in rows:
            sid = str(row.get("state_id", ""))
            if not sid:
                continue
            row_digest = _row_evidence_sha256(row)
            if sid in state_row_digests:
                if state_row_digests[sid] != row_digest:
                    raise ValueError(f"conflicting duplicate state_id evidence {sid}")
                continue
            state_row_digests[sid] = row_digest
            label = row.get("label", {})
            validation = gain_oracle.validate_gain_label(label)
            recomputed = gain_oracle.recompute_gain_label(label)
            if validation.get("valid") is not True:
                raise ValueError(f"oracle failed structurally valid or masked label {sid}")
            if (validation.get("label_valid_mask") != recomputed.get("valid_mask")
                    or validation.get("recomputed_gain_mean") != recomputed.get("gain_mean")):
                raise ValueError(f"oracle validation/recomputation mismatch {sid}")
            if row.get("oracle_receipt", {}).get("label_sha256") != label.get("label_sha256"):
                raise ValueError(f"stored oracle receipt does not bind label {sid}")
            valid = recomputed["valid_mask"] is True
            forced = row.get("forced") is True or label.get("forced") is True
            empty = label.get("unknown_reason") == "empty_obligation_window"
            unknown = not valid and not forced and not empty
            gain = recomputed["gain_mean"] if valid else None
            state = {"state_id": sid, "scenario_id": str(row.get("scenario_id")), "split": row.get("split"),
                     "event_type": row.get("event_type"), "time": row.get("time"), "forced": forced,
                     "valid": valid, "empty": empty, "unknown": unknown,
                     "unknown_reason": label.get("unknown_reason"), "gain": gain,
                     "gain_class": ("positive" if gain > 0 else "nonpositive") if valid else None,
                     "selected_action": _action_summary(row.get("replan_action", {})),
                     "public_cost_estimate": (row.get("candidate_receipt") or {}).get("selected_public_cost_estimate"),
                     "selection_key": (row.get("candidate_receipt") or {}).get("selection_key"),
                     "selected_candidate_key": (row.get("candidate_receipt") or {}).get("selected_candidate_key"),
                     "planner_cpu_seconds": row.get("planner_cpu_seconds"),
                     "offline_label_planner_cpu_seconds": row.get("offline_label_planner_cpu_seconds"),
                     "pair_count": recomputed["pair_count"], "pair_gains": [], "branch_components": [], "region_effects": []}
            for pi, pair in enumerate(recomputed["pairs"]):
                if pair.get("valid_mask") is True:
                    state["pair_gains"].append(pair["gain"])
                    state["branch_components"].append(_branch_components(label["pairs"][pi]))
                    state["region_effects"].append(_task_region_changes(label["pairs"][pi]))
            states[sid] = state

            receipt = row.get("candidate_receipt", {})
            candidates = receipt.get("candidates", []) if isinstance(receipt, Mapping) else []
            selected = row.get("replan_action", {})
            ranked_actions = [
                (float(candidate["public_cost_estimate"]), fixed_replanner.canonical_action_key(candidate["full_environment_action"]), candidate["full_environment_action"])
                for candidate in candidates
                if isinstance(candidate.get("public_cost_estimate"), (int, float)) and isinstance(candidate.get("full_environment_action"), Mapping)
            ]
            if ranked_actions:
                expected_action = min(ranked_actions, key=lambda item: (item[0], item[1]))[2]
                selection_checks["verified"] += int(expected_action == selected)
                selection_checks["mismatch"] += int(expected_action != selected)
            else:
                selection_checks["missing_candidates"] += 1
            state_kind = "forced" if forced else "ordinary"
            for candidate in candidates:
                routes = _cross_region_routes(candidate.get("full_environment_action", {}))
                if not routes:
                    continue
                candidate_counter["candidate_actions_any_cross_region"] += 1
                kinds = {route["kind"] for route in routes}
                candidate_counter[f"candidate_actions_any_cross_region_{state_kind}"] += 1
                candidate_state_coverage["any"][state_kind].add(sid)
                for route in routes:
                    candidate_counter[f"candidate_operations_{route['kind']}"] += 1
                    if route["channel"] == "queue_operation":
                        candidate_counter[f"candidate_queue_operations_{route['kind']}"] += 1
                    else:
                        candidate_counter[f"candidate_direct_dispatch_routes_{state_kind}"] += 1
                for kind in kinds:
                    candidate_counter[f"candidate_actions_{kind}"] += 1
                    candidate_counter[f"candidate_actions_{state_kind}_{kind}"] += 1
                    candidate_state_coverage[kind][state_kind].add(sid)
            selected_routes = _cross_region_routes(selected)
            if selected_routes:
                selected_kinds = {route["kind"] for route in selected_routes}
                candidate_counter["selected_actions_any_cross_region"] += 1
                candidate_counter[f"selected_actions_any_cross_region_{state_kind}"] += 1
                selected_state_coverage["any"][state_kind].add(sid)
                for route in selected_routes:
                    candidate_counter[f"selected_operations_{route['kind']}"] += 1
                    if route["channel"] == "queue_operation":
                        candidate_counter[f"selected_queue_operations_{route['kind']}"] += 1
                    else:
                        candidate_counter[f"selected_direct_dispatch_routes_{state_kind}"] += 1
                for kind in selected_kinds:
                    candidate_counter[f"selected_actions_{kind}"] += 1
                    candidate_counter[f"selected_actions_{state_kind}_{kind}"] += 1
                    selected_state_coverage[kind][state_kind].add(sid)
                selected_rows.append({"state_id": sid, "scenario_id": state["scenario_id"], "split": state["split"],
                                      "forced": forced, "label_valid": valid, "gain": gain,
                                      "routes": selected_routes, "selected_public_cost_estimate": receipt.get("selected_public_cost_estimate"),
                                      "selection_key": receipt.get("selection_key"), "candidate_count": receipt.get("actual_candidate_count"),
                                      "infeasible_count": receipt.get("infeasible_count"), "physical_duplicate_count": receipt.get("physical_duplicate_count")})

            if forced or not valid:
                continue
            if gain is not None:
                class_scenarios[state["split"]][state["gain_class"]].add(state["scenario_id"])
            if selected.get("noop") is True:
                noop_candidates = [c for c in candidates if c.get("full_environment_action", {}).get("noop") is True]
                alternatives = [c for c in candidates if c.get("full_environment_action", {}).get("noop") is not True]
                state["actual_candidate_count"] = receipt.get("actual_candidate_count")
                state["legal_candidate_count"] = len(candidates)
                state["infeasible_candidate_count"] = receipt.get("infeasible_count")
                if not noop_candidates:
                    state["noop_selection_reason"] = "selected_NOOP_missing_from_receipt"
                    noop_reasons["selected_NOOP_missing_from_receipt"] += 1
                elif alternatives:
                    noop_cost = float(noop_candidates[0]["public_cost_estimate"])
                    alternative_min = min(float(c["public_cost_estimate"]) for c in alternatives)
                    reason = "NOOP_has_lower_public_cost_than_all_alternatives" if noop_cost < alternative_min else "NOOP_tie_broken_by_canonical_action_key"
                    state["noop_selection_reason"] = reason
                    noop_reasons[reason] += 1
                    state["noop_public_cost_estimate"] = noop_cost
                    state["lowest_nonnoop_public_cost_estimate"] = alternative_min
                    state["lowest_nonnoop_minus_noop_public_cost"] = alternative_min - noop_cost
                else:
                    state["noop_selection_reason"] = "NOOP_only_legal_candidate"
                    state["noop_public_cost_estimate"] = float(noop_candidates[0]["public_cost_estimate"]) if noop_candidates else None
                    noop_reasons["NOOP_only_legal_candidate"] += 1
            else:
                noop_rows = [c for c in candidates if c.get("full_environment_action", {}).get("noop") is True]
                if not noop_rows:
                    nonnoop_reasons["NOOP_candidate_missing"] += 1
                else:
                    noop_cost = float(noop_rows[0].get("public_cost_estimate"))
                    selected_cost = float(receipt.get("selected_public_cost_estimate"))
                    nonnoop_reasons["NONNOOP_selected_by_public_cost"] += 1
                    state["noop_public_cost_estimate"] = noop_cost
                    state["selected_minus_noop_public_cost"] = selected_cost - noop_cost

    safety_audits = [row["live_safety_receipt_audit"] for row in scenario_headers.values()]
    actual_safety_counts = [row["actual_unresolved_disposition_count"] for row in safety_audits
                            if row["actual_unresolved_disposition_count"] is not None]
    reported_safety_counts = [row["reported_unresolved_disposition_count"] for row in safety_audits
                              if row["reported_unresolved_disposition_count"] is not None]
    safety_mismatches = [row for row in scenario_headers.values()
                         if row["live_safety_receipt_audit"]["unresolved_count_matches_summary"] is False]
    actual_forced_counts = [row["actual_forced_state_count"] for row in safety_audits]
    reported_forced_counts = [row["reported_forced_state_count"] for row in safety_audits
                              if row["reported_forced_state_count"] is not None]
    evaluable = [s for s in states.values() if s["valid"] and not s["forced"] and not s["empty"]]
    positives = [s for s in evaluable if s["gain"] > 0]
    nonpositives = [s for s in evaluable if s["gain"] <= 0]
    by_split = {}
    for split in sorted({s["split"] for s in states.values()}):
        subset = [s for s in states.values() if s["split"] == split]
        ev = [s for s in subset if s["valid"] and not s["forced"] and not s["empty"]]
        by_split[split] = {"states_unique": len(subset), "forced_excluded": sum(s["forced"] for s in subset),
                           "empty_excluded": sum(s["empty"] and not s["forced"] for s in subset),
                           "unknown_excluded": sum(s["unknown"] for s in subset), "evaluable": len(ev),
                           "positive": sum(s["gain"] > 0 for s in ev), "nonpositive": sum(s["gain"] <= 0 for s in ev),
                           "positive_scenarios": len({s["scenario_id"] for s in ev if s["gain"] > 0}),
                           "nonpositive_scenarios": len({s["scenario_id"] for s in ev if s["gain"] <= 0}),
                           "unknown_reasons": dict(Counter(s["unknown_reason"] or "unspecified" for s in subset if s["unknown"]))}
    expected_registry = v3_shards.validate_registry(
        v3_shards.expected_registry(task_mode="search_rescue", targets_per_region=3),
        task_mode="search_rescue", targets_per_region=3)
    expected_by_id = {
        f"v33-{row['split']}-{row['condition']}-{row['task_mode']}-n{row['targets_per_region']}-{row['instance_index']:03d}-{row['seed_variant']:08d}": row
        for row in expected_registry
    }
    expected_scenarios = len(expected_by_id)
    if expected_scenarios != 168 or expected_scenarios != len(expected_registry):
        raise ValueError("frozen V33 expected registry does not define 168 unique scenario identities")
    committed_ids = set(scenario_headers)
    missing_scenario_ids = sorted(set(expected_by_id) - committed_ids)
    unexpected_scenario_ids = sorted(committed_ids - set(expected_by_id))
    identity_mismatches = {}
    for scenario_id in sorted(committed_ids & set(expected_by_id)):
        expected = expected_by_id[scenario_id]
        observed = scenario_headers[scenario_id]
        mismatch = {key: {"expected": value, "observed": observed.get(key)}
                    for key, value in (("split", expected["split"]), ("condition", expected["condition"]),
                                       ("task_mode", expected["task_mode"]),
                                       ("target_density", expected["targets_per_region"]),
                                       ("seed_variant", expected["seed_variant"]))
                    if observed.get(key) != value}
        if mismatch:
            identity_mismatches[scenario_id] = mismatch
    progress_records = []
    for run_dir in run_dirs:
        progress_path = run_dir / "COLLECTION_PROGRESS.json"
        if progress_path.is_file():
            progress_records.append(json.loads(progress_path.read_text(encoding="utf-8")))
    historical_failures = []
    for record in progress_records:
        for failure in record.get("failed_scenarios", []):
            failure_row = failure if isinstance(failure, Mapping) else {"scenario_id": str(failure)}
            scenario_id = str(failure_row.get("scenario_id", ""))
            historical_failures.append({"run_id": record.get("run_id"), **dict(failure_row),
                                        "resolved_by_committed_header": bool(scenario_id) and scenario_id in committed_ids})
    unresolved_failures = sorted({str(row.get("scenario_id") or "<missing_scenario_id>") for row in historical_failures
                                  if not row.get("scenario_id") or not row["resolved_by_committed_header"]})
    complete = (not missing_scenario_ids and not unexpected_scenario_ids and not identity_mismatches
                and not unresolved_failures)
    collection_state = ("complete_168" if complete else "failed_shards" if unresolved_failures
                        else "registry_identity_mismatch" if unexpected_scenario_ids or identity_mismatches
                        else "partial")
    total_positive = sum(x["positive"] for x in by_split.values())
    total_nonpositive = sum(x["nonpositive"] for x in by_split.values())
    split_coverage = all(
        by_split.get(split, {}).get("positive", 0) > 0
        and by_split.get(split, {}).get("nonpositive", 0) > 0
        and by_split.get(split, {}).get("positive_scenarios", 0) > 1
        and by_split.get(split, {}).get("nonpositive_scenarios", 0) > 1
        for split in ("train", "validation")
    )
    runner_gate_met = total_positive >= 30 and total_nonpositive >= 30 and split_coverage
    marker_manifest = [{"scenario_id": record["scenario_id"], "marker_sha256": record["marker_sha256"]}
                       for record in sorted((commit for _, commit in unique_shards.values()), key=lambda item: item["scenario_id"])]
    marker_manifest_sha256 = hashlib.sha256(json.dumps(marker_manifest, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    expected_cells = Counter(f"{row['split']}:{row['condition']}" for row in expected_registry)
    observed_cells = Counter(f"{v['split']}:{v['condition']}" for v in scenario_headers.values())
    missing_cells = {cell: count - observed_cells.get(cell, 0) for cell, count in expected_cells.items()
                     if observed_cells.get(cell, 0) < count}
    ordinary_valid = [state for state in states.values() if state["valid"] and not state["forced"] and not state["empty"]]
    action_state_counts = {
        "candidate_actions": candidate_counter.get("candidate_actions_any_cross_region", 0),
        "selected_actions": candidate_counter.get("selected_actions_any_cross_region", 0),
        "candidate_states": {split: len(ids) for split, ids in candidate_state_coverage["any"].items()},
        "selected_states": {split: len(ids) for split, ids in selected_state_coverage["any"].items()},
        "candidate_actions_by_kind": {key.removeprefix("candidate_actions_"): value for key, value in candidate_counter.items() if key.startswith("candidate_actions_") and not key.startswith("candidate_actions_any_") and not key.startswith("candidate_actions_ordinary_") and not key.startswith("candidate_actions_forced_")},
        "selected_actions_by_kind": {key.removeprefix("selected_actions_"): value for key, value in candidate_counter.items() if key.startswith("selected_actions_") and not key.startswith("selected_actions_any_") and not key.startswith("selected_actions_ordinary_") and not key.startswith("selected_actions_forced_")},
    }
    noop_selected = sum(state["selected_action"]["noop"] for state in ordinary_valid)
    noop_summary = {"ordinary_valid_states": len(ordinary_valid), "selected_noop": noop_selected,
                    "selected_nonnoop": len(ordinary_valid) - noop_selected,
                    "selection_reasons_for_noop": dict(noop_reasons),
                    "selection_reasons_for_nonnoop": dict(nonnoop_reasons),
                    "outcome_decomposition_by_selected_noop": {
                        "NOOP": _outcome_decomposition([s for s in ordinary_valid if s["selected_action"]["noop"]]),
                        "non_NOOP": _outcome_decomposition([s for s in ordinary_valid if not s["selected_action"]["noop"]])}}
    selected_cross = _cross_selected_decomposition(list(states.values()))
    status = ("COMPLETE_168_GATE_MET" if runner_gate_met else "COMPLETE_168_GATE_NOT_MET") if complete else ("FAILED_SHARDS" if unresolved_failures else "PARTIAL")
    return {"status": status,
            "run_dirs": [x.name for x in run_dirs], "committed_shards": len(shard_paths), "duplicate_shard_references": duplicate_shards, "scenario_count": len(scenario_headers),
            "expected_scenarios": expected_scenarios, "complete_168": complete,
            "collection_status": {"state": collection_state, "failed_scenario_count": len(unresolved_failures),
                                  "failed_scenarios": unresolved_failures,
                                  "historical_failures": historical_failures,
                                  "missing_scenario_ids": missing_scenario_ids,
                                  "unexpected_scenario_ids": unexpected_scenario_ids,
                                  "identity_mismatches": identity_mismatches,
                                  "progress_records": [{"run_id": r.get("run_id"),
                                                        "completed": int(r.get("reused_committed", 0)) + int(r.get("new_committed", 0)),
                                                        "new_completed_scenarios_list_count": len(r.get("completed_scenarios", [])),
                                                        "reused_committed": int(r.get("reused_committed", 0)),
                                                        "new_committed": int(r.get("new_committed", 0)),
                                                        "failed": len(r.get("failed_scenarios", [])),
                                                        "total": r.get("total_scenarios")} for r in progress_records]},
            "snapshot": {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "committed_marker_count": len(marker_manifest),
                         "committed_marker_manifest_sha256": marker_manifest_sha256},
            "region_id_mapping": {"ordinal_to_contract_region": REGION_ORDINAL_TO_COORD,
                                  "note": "Task/platform IDs encode a two-digit row-major region ordinal; 00..08 map to contract region coordinates r00,r01,r02,r10,r11,r12,r20,r21,r22."},
            "live_safety_receipt_audit": {
                "scenario_count": len(safety_audits),
                "scenarios_with_trace": sum(row["trajectory_trace_available"] for row in safety_audits),
                "scenarios_missing_trace": sum(not row["trajectory_trace_available"] for row in safety_audits),
                "actual_unresolved_disposition_total_known_scenarios": sum(actual_safety_counts),
                "actual_unresolved_disposition_unknown_scenarios": len(safety_audits) - len(actual_safety_counts),
                "reported_unresolved_disposition_total_known_scenarios": sum(reported_safety_counts),
                "reported_unresolved_disposition_unknown_scenarios": len(safety_audits) - len(reported_safety_counts),
                "unresolved_count_summary_mismatch_scenarios": [row["scenario_id"] for row in safety_mismatches],
                "actual_forced_state_total": sum(actual_forced_counts),
                "reported_forced_state_total_known_scenarios": sum(reported_forced_counts),
                "reported_forced_state_unknown_scenarios": len(safety_audits) - len(reported_forced_counts),
                "forced_count_summary_mismatch_scenarios": [row["scenario_id"] for row in scenario_headers.values()
                    if row["live_safety_receipt_audit"]["forced_count_matches_summary"] is False],
                "interpretation": "trajectory disposition receipt cross-check and forced-state counts only; does not establish all hard constraints or method safety",
            },
            "run_coverage": {"split_condition": dict(observed_cells), "expected_split_condition": expected_cells,
                             "missing_or_short_cells": missing_cells, "coverage_complete": not missing_cells},
            "scenario_level_gain_summary": _scenario_gain_summary(list(states.values())),
            "state_counts": {"unique_state_id": len(states), "evaluable": len(evaluable), "positive": len(positives),
                             "zero": sum(s["gain"] == 0 for s in evaluable), "negative": sum(s["gain"] < 0 for s in evaluable),
                             "forced_excluded": sum(s["forced"] for s in states.values()),
                             "empty_excluded": sum(s["empty"] and not s["forced"] for s in states.values()),
                             "unknown_excluded": sum(s["unknown"] for s in states.values()),
                             "total_30_positive_30_nonpositive_met": total_positive >= 30 and total_nonpositive >= 30,
                             "split_coverage_met": split_coverage,
                             "runner_thirty_thirty_met": runner_gate_met,
                             "by_split": by_split},
            "lineage": lineage_records,
            "candidate_coverage": {**dict(candidate_counter), **action_state_counts,
                                   "candidate_states_by_kind": {kind: {split: len(ids) for split, ids in coverage.items()} for kind, coverage in candidate_state_coverage.items()},
                                   "selected_states_by_kind": {kind: {split: len(ids) for split, ids in coverage.items()} for kind, coverage in selected_state_coverage.items()}}, "selected_cross_region_actions": selected_rows,
            "noop_summary": noop_summary, "noop_failure_reasons": dict(noop_reasons), "selection_rank_verification": dict(selection_checks), "cost_semantics": cost_semantics,
            "resource_evidence": {"selected_planner_cpu_seconds_sum": sum(float(s["planner_cpu_seconds"]) for s in states.values() if isinstance(s["planner_cpu_seconds"], (int,float))),
                                  "selected_planner_cpu_missing_rows": sum(s["planner_cpu_seconds"] is None for s in states.values()),
                                  "offline_label_planner_cpu_seconds_sum": sum(float(s["offline_label_planner_cpu_seconds"]) for s in states.values() if isinstance(s["offline_label_planner_cpu_seconds"], (int,float))),
                                  "committed_shard_runtime_receipts": len(resource_receipts),
                                  "sum_shard_process_cpu_seconds": sum(float(x["process_cpu_seconds"]) for x in resource_receipts if isinstance(x.get("process_cpu_seconds"), (int,float))),
                                  "sum_shard_wall_seconds": sum(float(x["wall_seconds"]) for x in resource_receipts if isinstance(x.get("wall_seconds"), (int,float))),
                                  "maximum_recorded_rss_bytes": max((int(x["current_rss_bytes_at_receipt"]) for x in resource_receipts if isinstance(x.get("current_rss_bytes_at_receipt"), (int,float))), default=None),
                                  "memory_peak": "UNKNOWN: commit records contain RSS at receipt, not peak RSS",
                                  "platform_usage": "present per scenario header; not a host CPU resource measure"},
            "outcome_decomposition": _outcome_decomposition(list(states.values())),
            "selected_cross_region_decomposition": selected_cross,
            "scenarios": scenario_headers, "shards": shard_records,
            "states": sorted(states.values(), key=lambda x: (x["split"], x["scenario_id"], x["state_id"]))}


def render_markdown(data: Mapping[str, Any]) -> str:
    c = data["state_counts"]
    splits = c["by_split"]
    cross = data["candidate_coverage"]
    snapshot = data["snapshot"]
    decomposition = data["outcome_decomposition"]
    runner_gate = "已达" if c["runner_thirty_thirty_met"] else "未达"
    lines = [f"# V33 结论分析（{data['status']}）", "",
             f"快照：{snapshot['generated_at_utc']}；{data['scenario_count']}/{data['expected_scenarios']} 场景、{data['committed_shards']} 个已提交分片；采集状态 `{data['collection_status']['state']}`。marker 清单 SHA-256：`{snapshot['committed_marker_manifest_sha256']}`。非 complete_168 时，结论仅为诊断。", "",
             "## 前页结论", "",
             f"- 跨区有效否：当前证据不足以作正式结论。发现跨区候选 action {cross.get('candidate_actions', 0)} 个、选派 action {cross.get('selected_actions', 0)} 个；候选/选派唯一状态数按 split 见下表。这些不是跨区相对本地策略的因果效应。",
             f"- 门控有效否：{c['positive']} 个正增益、{c['zero']} 个零增益、{c['negative']} 个负增益唯一普通状态。冻结门槛（总正≥30、总非正≥30，且 train 与 validation 各自两类均覆盖多个场景）{runner_gate}。",
             f"- 推荐方法：{('当前 168 场景均已提交且无失败分片，可进入门槛评估；' if data['complete_168'] else '继续完成预注册采集并按同一脚本重算；')}本快照不据此选定模型或部署门控。",
             "- WM 值不值得：当前数据没有 WM 与轻量方法/无模型的闭环对照，不能回答；不应据此训练或推荐 WM。",
             "- 适用范围：合成 100×100、3×3 九区、36 平台、search-rescue 场景与已覆盖扰动；不能外推真实灾害或未覆盖任务。", "",
             "## 状态与覆盖", "",
             "| split | 唯一状态 | forced | empty | unknown | 可评估 | 正 | 非正 | 正类场景 | 非正场景 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, x in sorted(splits.items()):
        lines.append(f"| {split} | {x['states_unique']} | {x['forced_excluded']} | {x['empty_excluded']} | {x['unknown_excluded']} | {x['evaluable']} | {x['positive']} | {x['nonpositive']} | {x['positive_scenarios']} | {x['nonpositive_scenarios']} |")
    lines += ["", "类按每个唯一 `state_id` 的 oracle 重算均值划分；同一状态的匹配后缀只贡献一个状态。forced、empty obligation 和 unknown 均排除。30/30 为 train+validation 合计门槛，并要求两个 split 中正类与非正类分别覆盖多个场景；没有单独的 train≥30 要求。", "",
              "区间按独立单位 `scenario_instance` 计算：先在场景内平均普通有效 state gain，再 bootstrap 场景 10,000 次。区间描述本批采集标签，不是策略或跨区处理效应 CI。", "",
              "| split | 场景数 | 场景均值的等权平均 gain | 场景 bootstrap 95% CI |", "|---|---:|---:|---:|"]
    for split, summary in sorted(data["scenario_level_gain_summary"].items()):
        ci = summary.get("bootstrap_95_ci")
        rendered_ci = "NA (n<2)" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"
        mean = summary.get("mean_of_scenario_means")
        lines.append(f"| {split} | {summary['scenario_count']} | {'UNKNOWN' if mean is None else f'{mean:.6f}'} | {rendered_ci} |")
    safety = data["live_safety_receipt_audit"]
    lines += ["", "## Live Safety Receipt", "",
              f"从 header `trajectory_trace` 独立计数 `safety_disposition.resolved == false`：已知场景 {safety['scenarios_with_trace']}/{safety['scenario_count']}，实际未解决处置数合计 {safety['actual_unresolved_disposition_total_known_scenarios']}；摘要计数合计 {safety['reported_unresolved_disposition_total_known_scenarios']}。缺失 trajectory trace 的场景 {safety['scenarios_missing_trace']} 个，不按零处理；逐场景计数、原因和摘要匹配结果在 `analysis.json.scenarios.*.live_safety_receipt_audit`。摘要不一致场景：`{json.dumps(safety['unresolved_count_summary_mismatch_scenarios'], ensure_ascii=False)}`。",
              f"强制状态实际计数 {safety['actual_forced_state_total']}；header 摘要已知计数合计 {safety['reported_forced_state_total_known_scenarios']}，缺摘要 {safety['reported_forced_state_unknown_scenarios']} 个场景；不一致场景：`{json.dumps(safety['forced_count_summary_mismatch_scenarios'], ensure_ascii=False)}`。这是运行时安全处置回执核验，不证明全部硬约束已满足或方法安全；branch safety 仍由 gain oracle 的配对标签单独复算，最终安全结论限于协议和实际核验范围。", ""]
    lines += ["",
              "## 跨区动作与后果", "",
              ("候选回执排序复核 {} 条、排序不匹配 {} 条。跨区 action 以每个候选/选中动作计数，唯一状态按 `state_id` 去重：候选 {} actions，选派 {} actions。按 queue operation 的原始语义分别计数，insert={}、transfer={}、split={}、cooperate={}；split/cooperate 不计作 donor transfer。含跨区 direct dispatch 的候选 action={}、选派 action={}，即使没有 queue operation 也纳入 action 总数.".format(
                  data["selection_rank_verification"].get("verified", 0), data["selection_rank_verification"].get("mismatch", 0),
                  cross.get("candidate_actions", 0), cross.get("selected_actions", 0),
                  cross.get("candidate_queue_operations_insert", 0), cross.get("candidate_queue_operations_transfer", 0),
                  cross.get("candidate_queue_operations_split", 0), cross.get("candidate_queue_operations_cooperate", 0),
                  cross.get("candidate_actions_direct_dispatch", 0), cross.get("selected_actions_direct_dispatch", 0))).replace("action 总数.", "action 总数。"),
              "| action / state | 候选 action 数 | 候选唯一状态数 | 选派 action 数 | 选派唯一状态数 |", "|---|---:|---:|---:|---:|"]
    for kind in ("any", "insert", "transfer", "split", "cooperate", "direct_dispatch"):
        candidate_actions = cross.get("candidate_actions", 0) if kind == "any" else cross.get("candidate_actions_by_kind", {}).get(kind, 0)
        selected_actions = cross.get("selected_actions", 0) if kind == "any" else cross.get("selected_actions_by_kind", {}).get(kind, 0)
        candidate_states = cross.get("candidate_states", {}) if kind == "any" else cross.get("candidate_states_by_kind", {}).get(kind, {})
        selected_states = cross.get("selected_states", {}) if kind == "any" else cross.get("selected_states_by_kind", {}).get(kind, {})
        lines.append(f"| {kind} | {candidate_actions} | {sum(candidate_states.values())} | {selected_actions} | {sum(selected_states.values())} |")
    lines += ["", "上述 action/state 统计另在 JSON 按 ordinary/forced 拆分；forced 动作不带有效 gain 标签。任务/平台 ID 中 `00`..`08` 是 row-major 区域 ordinal，映射契约区域 `r00,r01,r02,r10,r11,r12,r20,r21,r22`；完整映射见 JSON `region_id_mapping`。区域后果与成本分解只汇总普通、有效标签的匹配 suffix；每个 suffix 的明细见 `state_detail.csv`。区域进度变化是 endpoint 结果，不能单独识别跨区动作的因果效果。", "",
              "| 区域 | paired suffix 数 | keep 完成任务数均值 | replan 完成任务数均值 | 完成数差均值 | keep 剩余比例和均值 | replan 剩余比例和均值 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for region, values in decomposition["regional_endpoint_effect_means"].items():
        lines.append(f"| {region} | {values['task_count']['count']} | {values['keep_completed']['mean']:.4f} | {values['replan_completed']['mean']:.4f} | {values['completed_delta_replan_minus_keep']['mean']:.4f} | {values['keep_remaining_fraction_sum']['mean']:.4f} | {values['replan_remaining_fraction_sum']['mean']:.4f} |")
    lines += ["", "### 普通有效且选派跨区状态", "",
              f"唯一状态 {data['selected_cross_region_decomposition']['unique_state_count']} 个，覆盖 {data['selected_cross_region_decomposition']['scenario_count']} 个场景；场景均值的等权平均 gain 为 {data['selected_cross_region_decomposition']['mean_of_scenario_means']}，描述性 95% CI={data['selected_cross_region_decomposition']['bootstrap_95_ci']}（{data['selected_cross_region_decomposition']['ci_status']}）。路线按平台 home 区与任务区区分：home 区跨区 insert、显式 donor transfer、其他跨区操作及 direct dispatch 分开计数；insert 路由本身不证明平台空闲。这里只是被选中跨区状态的配对结果，不是跨区动作相对本地动作的因果效应。", "",
              "| 路线类别 | 被选路由数 |", "|---|---:|"]
    for key, value in sorted(data["selected_cross_region_decomposition"]["selected_route_class_counts"].items()):
        lines.append(f"| {key} | {value} |")
    cross_regional = data["selected_cross_region_decomposition"]["outcomes"]["regional_endpoint_effect_means"]
    lines += ["", "| 区域（all-region endpoint） | paired suffix 数 | keep 完成数 | replan 完成数 | 完成数差 | keep 剩余比例和 | replan 剩余比例和 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for region, values in cross_regional.items():
        lines.append(f"| {region} | {values['task_count']['count']} | {values['keep_completed']['mean']:.4f} | {values['replan_completed']['mean']:.4f} | {values['completed_delta_replan_minus_keep']['mean']:.4f} | {values['keep_remaining_fraction_sum']['mean']:.4f} | {values['replan_remaining_fraction_sum']['mean']:.4f} |")
    role_effects = data["selected_cross_region_decomposition"]["regional_role_effects"]
    lines += ["", "角色分组按每个唯一 state×suffix 中的路线区域去重；同一区域可同时属于多个角色，重叠 state 数保存在 JSON。`other_region` 是当前状态路线未涉及的区域。", "",
              "| 路由角色 | 区域 | paired suffix 数 | 完成数差均值 | 剩余比例差均值（keep−replan） |", "|---|---|---:|---:|---:|"]
    for role, metrics in sorted(role_effects.items()):
        for region, values in sorted(metrics.get("completed_delta_replan_minus_keep", {}).items()):
            rem = metrics.get("remaining_fraction_delta_keep_minus_replan", {}).get(region, {})
            lines.append(f"| {role} | {region} | {values['count']} | {values['mean']:.4f} | {rem.get('mean', 0.0):.4f} |")
    for group_name, group_data in (("selected cross-region", data["selected_cross_region_decomposition"]["outcomes"]),
                                   ("selected NOOP", data["noop_summary"]["outcome_decomposition_by_selected_noop"]["NOOP"]),
                                   ("selected non-NOOP", data["noop_summary"]["outcome_decomposition_by_selected_noop"]["non_NOOP"])):
        lines += ["", f"| {group_name} 全局成本分项均值 | keep | replan | replan − keep |", "|---|---:|---:|---:|"]
        cost_rows = group_data["global_cost_component_means"]
        for key in sorted(cost_rows):
            item = group_data["global_cost_component_means"].get(key, {}).get("mean", {})
            fmt = lambda value: "NA" if value is None else f"{value:.6f}"
            delta = item.get("delta_replan_minus_keep")
            if delta is None and item.get("keep") is not None and item.get("replan") is not None:
                delta = item["replan"] - item["keep"]
            lines.append(f"| {key} | {fmt(item.get('keep'))} | {fmt(item.get('replan'))} | {fmt(delta)} |")
    lines += ["", "| 全局成本分项（均值） | keep | replan | replan − keep |", "|---|---:|---:|---:|"]
    for key in ("J_total", "J_exec", "C_replan", "C_safety"):
        item = decomposition["global_cost_component_means"].get(key, {}).get("mean", {})
        fmt = lambda value: "NA" if value is None else f"{value:.6f}"
        lines.append(f"| {key} | {fmt(item.get('keep'))} | {fmt(item.get('replan'))} | {fmt(item.get('delta_replan_minus_keep'))} |")
    lines += ["", "execution / replan / safety 子项的全量数值与上述成本汇总一并保存在 `analysis.json` 的 `outcome_decomposition.global_cost_component_means`。这些值比较 paired keep 与 replan 的整体结果，不能单独解释为跨区的效果。", "",
              "## 可识别性与限制", "",
              f"普通有效状态共 {data['noop_summary']['ordinary_valid_states']} 个，选派 NOOP {data['noop_summary']['selected_noop']} 个、非 NOOP {data['noop_summary']['selected_nonnoop']} 个；NOOP 选择原因（仅 selected NOOP）：`{json.dumps(data['noop_summary']['selection_reasons_for_noop'], ensure_ascii=False)}`；selected non-NOOP 原因：`{json.dumps(data['noop_summary']['selection_reasons_for_nonnoop'], ensure_ascii=False)}`。候选与不可行数明细在 `state_detail.csv`。forced 的跨区选派没有可用 gain 标签，不能用来估计动作收益。", "",
              "当前可从轨迹识别：候选是否合法且跨区、所选动作与公开排序数值、匹配 suffix 下 keep/replan 的整体成本和区域任务后果、全局 gain 正零负及 mask 原因。live safety receipt 只核对运行时处置回执及摘要，不代表所有硬约束或方法安全证明。不能从该采集识别：将同一时点同一局面随机分配跨区与本地动作的纯跨区因果效应；跨区具体机制相对能力/距离/供需排序的独立贡献；训练后模型门控的闭环收益；相对 WM 的性能/成本；真实救灾的外部有效性。", "",
              f"资源：选派 planner CPU 合计 {data['resource_evidence']['selected_planner_cpu_seconds_sum']:.3f} 秒；已提交分片 runtime receipt 报告进程 CPU {data['resource_evidence']['sum_shard_process_cpu_seconds']:.3f} 秒、墙钟 {data['resource_evidence']['sum_shard_wall_seconds']:.3f} 秒，receipt 时 RSS 最大值 {data['resource_evidence']['maximum_recorded_rss_bytes']} bytes。峰值内存为 UNKNOWN，不能由 receipt 时 RSS 替代。",
              f"成本复核：独立复算覆盖 {decomposition['ordinary_valid_suffix_pairs']} 个普通有效 matched suffix；其中 keep/replan endpoint 共 {decomposition['completed_endpoint_tasks']} 个完成任务记录，完成但 progress<1 的记录为 {decomposition['completed_endpoint_tasks_below_full_progress']}。oracle 的 completed 任务分支直接跳过 residual 累加，因此完成任务残余惩罚为零；未完成任务按加权 clipped `(1-progress)` 计入。", "",
              "## 下一步所需证据", "",
              (f"采集状态 `{data['collection_status']['state']}`：已提交 {data['scenario_count']}/{data['expected_scenarios']} 场景，失败 {data['collection_status']['failed_scenario_count']} 个；相对冻结计划，split-condition 缺失或不足单元：`{json.dumps(data['run_coverage']['missing_or_short_cells'], ensure_ascii=False)}`。" if not data["complete_168"] else "168 个 train/validation 场景及 commit marker 均已校验；仍需检查 30/30 标签门槛及 train/validation 覆盖结果。"),
              "若要回答跨区纯因果问题，需预先定义并随机/成对构造具有共同可行性的 local 与 cross-region 动作对，并固定外生 suffix、安全继续策略和任务宇宙。若要回答门控与 WM，须在训练门槛通过后用闭环策略、相同 replanner/safety、独立测试场景比较，报告场景单位配对 CI。", "",
              "原 45 条 smoke 记录属于工程 smoke，不能拼入正式类门槛或方法效应。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "v33-conclusion")
    args = parser.parse_args()
    data = analyze([args.run_dir.resolve()])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "conclusion.md").write_text(render_markdown(data), encoding="utf-8")
    summary = {key: data[key] for key in ("status", "snapshot", "scenario_count", "expected_scenarios", "complete_168", "collection_status", "run_coverage", "live_safety_receipt_audit", "state_counts", "candidate_coverage", "noop_summary", "selection_rank_verification", "outcome_decomposition", "selected_cross_region_decomposition")}
    summary["run_ids"] = data["run_dirs"]
    summary["limitations"] = ["simulation_only", "cross_region_counts_are_descriptive_not_causal"]
    if not data["complete_168"]:
        summary["limitations"].append("partial_snapshot_not_formal_conclusion")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["state_id", "scenario_id", "split", "event_type", "time", "forced", "valid", "empty", "unknown", "unknown_reason", "gain", "gain_class", "selected_action", "public_cost_estimate", "selection_key", "planner_cpu_seconds", "offline_label_planner_cpu_seconds", "pair_count", "pair_gains", "branch_components", "region_effects", "noop_selection_reason", "noop_public_cost_estimate", "lowest_nonnoop_public_cost_estimate", "lowest_nonnoop_minus_noop_public_cost", "actual_candidate_count", "legal_candidate_count", "infeasible_candidate_count"]
    with (args.output_dir / "state_detail.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data["states"]:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields})
    print(json.dumps({"status": data["status"], "scenario_count": data["scenario_count"], "committed_shards": data["committed_shards"], "state_counts": data["state_counts"], "candidate_coverage": data["candidate_coverage"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
