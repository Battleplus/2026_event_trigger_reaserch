"""Scene-paired statistical comparison of independently reduced episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


BOOTSTRAP_REPLICATES = 10_000
METRICS = ("completion_rate", "discovery_rate", "mean_effect_delay_s",
           "safety_violations", "planner_calls", "online_method_cost")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _episode_metrics(reduction: dict[str, Any]) -> dict[str, Any]:
    tasks = reduction.get("tasks", {})
    discovery = reduction.get("discovery", {})
    safety = reduction.get("safety", {})
    planning = reduction.get("planning", {})
    objective = reduction.get("registered_objective", {})
    absolute = objective.get("absolute_online_episode_cost", {})
    cost = absolute.get("measured_online_method_cost", {})
    delays = reduction.get("decision_effect_delays", {})
    delay_values = delays.get("effect_delays") if delays.get("status") == "complete" else None
    delay_mean = None
    if isinstance(delay_values, list) and delay_values:
        values = [_finite(row.get("effect_delay_s")) for row in delay_values if isinstance(row, dict)]
        if values and all(value is not None for value in values):
            delay_mean = sum(values) / len(values)
    catalog_count = _finite(tasks.get("catalog_count"))
    completed_count = _finite(tasks.get("completed_count"))
    target_count = _finite(discovery.get("target_count"))
    discovered_count = _finite(discovery.get("discovered_count"))
    checks = reduction.get("checks", {})
    safety_ready = (reduction.get("status") == "PASS" and reduction.get("valid_mask") is True
                    and checks.get("stationary_hold_positions") is True
                    and safety.get("hold_motion_violation_check") is True
                    and safety.get("all_forced_safety_actions_applied") is True
                    and _finite(safety.get("controllable_violation_events")) is not None)
    return {
        "reduction_pass": reduction.get("status") == "PASS" and reduction.get("valid_mask") is True,
        "completion_rate": completed_count / catalog_count if catalog_count and catalog_count > 0 else None,
        "discovery_rate": discovered_count / target_count if target_count and target_count > 0 else None,
        "mean_effect_delay_s": delay_mean,
        "safety_violations": _finite(safety.get("controllable_violation_events")),
        "forced_safety_action_evidence": "complete" if safety_ready else "unknown",
        "planner_calls": _finite(planning.get("candidate_generation_calls")),
        "online_method_cost": (_finite(cost.get("value"))
            if absolute.get("status") == "measured" and cost.get("status") == "measured" else None),
        "online_cost_unit": "registered_objective_units",
    }


def _resolve_reduction(root: Path, row: dict[str, Any]) -> Path:
    location = row.get("reduction_path") or row.get("episode_reduction")
    if location:
        path = Path(str(location))
        return path if path.is_absolute() else root / path
    output = row.get("output_path")
    if output:
        return root / str(output) / "episode_reduction.json"
    run_id = row.get("run_id")
    if run_id:
        return root / str(run_id) / "episode_reduction.json"
    raise ValueError("registry row needs reduction_path, episode_reduction, output_path, or run_id")


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_statistics(values: list[tuple[str, float]], *, seed: int,
                       bootstrap_replicates: int) -> dict[str, Any]:
    if not values:
        return {"n_scenes": 0, "mean_difference": None, "ci95": None,
                "ci95_label": "uncorrected scene bootstrap interval", "sample_sd": None,
                "permutation_p_two_sided": None, "status": "INCONCLUSIVE_NO_COMPLETE_PAIRS"}
    differences = [value for _, value in values]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(bootstrap_replicates):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    exceed = 0
    for _ in range(bootstrap_replicates):
        permuted = [value if rng.getrandbits(1) else -value for value in differences]
        if abs(sum(permuted) / len(permuted)) >= abs(observed) - 1e-15:
            exceed += 1
    return {"n_scenes": len(values), "mean_difference": observed,
            "ci95": [_percentile(boot_means, 0.025), _percentile(boot_means, 0.975)],
            "ci95_label": "uncorrected scene bootstrap interval",
            "sample_sd": _sample_sd(differences),
            "permutation_p_two_sided": (1 + exceed) / (bootstrap_replicates + 1),
            "bootstrap_replicates": bootstrap_replicates,
            "scene_differences": {scene: value for scene, value in values}, "status": "COMPLETE"}


def _holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((name, p) for name, p in p_values.items() if p is not None), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    count = len(valid)
    for index, (name, p_value) in enumerate(valid):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[name] = running
    return adjusted


def _stage_label(split: str) -> str:
    return {"train": "training_diagnostic", "validation": "validation_only",
            "test": "independent_test", "ood": "ood_confirmation"}.get(split, "unregistered_stage")


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _scope_signature(items: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for item in items:
        row = item["registry"]
        reduction = item.get("reduction", {})
        scope = reduction.get("scope", {}) if isinstance(reduction, dict) else {}
        signature = {key: row.get(key) for key in ("condition", "parent_family", "cohort", "mode", "public_scope")
                     if key in row}
        signature["duration_s"] = scope.get("duration_s")
        signature["branch"] = scope.get("branch")
        signature["actor_seconds"] = scope.get("actor_seconds")
        signature["task_catalog_count"] = (reduction.get("tasks") or {}).get("catalog_count")
        signature["target_count"] = (reduction.get("discovery") or {}).get("target_count")
        signatures.add(json.dumps(signature, sort_keys=True, separators=(",", ":")))
    return signatures


def compare(registry_path: str | Path, reductions_root: str | Path, primary_method: str,
            *, bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
            seed: int = 20261021) -> dict[str, Any]:
    """Compare registry-registered methods by independent scene IDs."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    registry_path = Path(registry_path).resolve()
    root = Path(reductions_root).resolve()
    registry = _load_json(registry_path)
    rows = registry.get("runs", registry.get("scenes"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("registration manifest must contain a non-empty runs or scenes list")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_registration_rows = []
    methods = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            excluded_registration_rows.append({"index": row_index, "reason": "row_not_object"})
            continue
        scene = row.get("scene_id") or row.get("scenario_id")
        method = row.get("method") or row.get("source")
        split = row.get("split")
        if not scene or not method or not split:
            excluded_registration_rows.append({"index": row_index,
                "reason": "missing_scene_id_scenario_id_method_or_split"})
            continue
        method = str(method)
        methods.add(method)
        training_seed = row.get("training_seed", row.get("model_seed"))
        try:
            reduction_path = _resolve_reduction(root, row)
        except ValueError as exc:
            excluded_registration_rows.append({"index": row_index, "reason": str(exc)})
            continue
        item = {"scenario_id": str(scene), "method": method, "split": str(split),
                "training_seed": training_seed, "source_scenario_id": row.get("scenario_id"),
                "registry": row, "reduction_path": str(reduction_path)}
        if reduction_path.is_file():
            try:
                reduction = _load_json(reduction_path)
                actual_run_id = (reduction.get("scope") or {}).get("run_id")
                expected_run_id = row.get("run_id")
                if not expected_run_id or actual_run_id != expected_run_id:
                    raise ValueError("reduction run_id does not match registration manifest")
                if reduction.get("schema") != "closedloop_episode_reduction_v1":
                    raise ValueError("unsupported episode reduction schema")
                item["reduction"] = reduction
                item["metrics"] = _episode_metrics(reduction)
                item["reduction_schema"] = reduction.get("schema")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                item["load_error"] = f"{type(exc).__name__}:{exc}"
        else:
            item["load_error"] = "reduction_missing"
        records[str(split)].append(item)
    if primary_method not in methods:
        raise ValueError(f"primary method {primary_method!r} is absent from registration manifest")

    result_splits = {}
    expected_seed_map = registry.get("expected_training_seeds", registry.get("training_seeds", {}))
    for split, split_rows in sorted(records.items()):
        by_method_scene: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for item in split_rows:
            by_method_scene[item["method"]][item["scenario_id"]].append(item)
        method_summaries = {}
        all_tests: dict[str, float | None] = {}
        for method in sorted(methods - {primary_method}):
            method_scene = by_method_scene.get(method, {})
            control_scene = by_method_scene.get(primary_method, {})
            candidate_scenes = sorted(set(method_scene) & set(control_scene))
            paired_scenes = []
            scope_mismatches = []
            incomplete_groups = []
            invalid_group_scenes = set()
            for scene in candidate_scenes:
                for group_method, group_items in ((method, method_scene[scene]),
                                                  (primary_method, control_scene[scene])):
                    observed = [str(item["training_seed"]) for item in group_items
                                if item.get("training_seed") is not None]
                    expected = expected_seed_map.get(group_method, []) if isinstance(expected_seed_map, dict) else []
                    expected = set(map(str, expected)) if isinstance(expected, list) else set()
                    reasons = []
                    if len(observed) != len(set(observed)):
                        reasons.append("duplicate_training_seed")
                    if len(observed) != len(group_items) and len(group_items) > 1:
                        reasons.append("missing_training_seed_for_replicate")
                    if expected and set(observed) != expected:
                        reasons.append("pre_registered_training_seed_set_mismatch")
                    if any(item.get("load_error") for item in group_items):
                        reasons.append("reduction_missing_or_invalid")
                    for reason in reasons:
                        entry = {"scene": scene, "method": group_method, "reason": reason}
                        if reason == "pre_registered_training_seed_set_mismatch":
                            entry.update({"missing_seeds": sorted(expected - set(observed)),
                                          "unexpected_seeds": sorted(set(observed) - expected)})
                        incomplete_groups.append(entry)
                        invalid_group_scenes.add(scene)
            for scene in candidate_scenes:
                method_scope = _scope_signature(method_scene[scene])
                control_scope = _scope_signature(control_scene[scene])
                if method_scope == control_scope and len(method_scope) == 1:
                    paired_scenes.append(scene)
                else:
                    scope_mismatches.append(scene)
            method_values: dict[str, dict[str, float]] = {}
            control_values: dict[str, dict[str, float]] = {}
            missing_metrics: dict[str, list[str]] = defaultdict(list)
            for scene in paired_scenes:
                method_items = method_scene[scene]
                control_items = control_scene[scene]
                if scene in invalid_group_scenes:
                    continue
                # Replicate fits are averaged within scene and method before inference.
                method_agg = {}
                control_agg = {}
                for metric in METRICS:
                    mv = [_finite(item.get("metrics", {}).get(metric)) for item in method_items]
                    cv = [_finite(item.get("metrics", {}).get(metric)) for item in control_items]
                    mv = [value for value in mv if value is not None]
                    cv = [value for value in cv if value is not None]
                    if mv:
                        method_agg[metric] = sum(mv) / len(mv)
                    else:
                        missing_metrics[metric].append(scene)
                    if cv:
                        control_agg[metric] = sum(cv) / len(cv)
                    if metric in {"completion_rate", "discovery_rate", "safety_violations",
                                  "planner_calls", "online_method_cost"} and not (mv and cv):
                        missing_metrics[metric].append(scene)
                method_values[scene] = method_agg
                control_values[scene] = control_agg
            usable_pair_ids = {scene for scene in paired_scenes
                               if scene in method_values and scene in control_values}
            safety_complete = bool(usable_pair_ids) and all(
                item.get("metrics", {}).get("forced_safety_action_evidence") == "complete"
                for scene in usable_pair_ids
                for item in (*method_scene[scene], *control_scene[scene]))
            stats = {}
            metric_tests = {}
            for metric in METRICS:
                diffs = []
                for scene in sorted(usable_pair_ids):
                    if metric in method_values[scene] and metric in control_values[scene]:
                        # Positive effects are favorable for completion/discovery;
                        # costs, delays, planner calls, and violations favor lower values.
                        direction = -1.0 if metric in {"mean_effect_delay_s", "safety_violations",
                                                       "planner_calls", "online_method_cost"} else 1.0
                        diffs.append((scene, direction * (method_values[scene][metric] - control_values[scene][metric])))
                digest = hashlib.sha256(f"{seed}:{split}:{method}:{metric}".encode()).digest()
                metric_stat = _paired_statistics(diffs, seed=int.from_bytes(digest[:8], "big"),
                                                 bootstrap_replicates=bootstrap_replicates)
                stats[metric] = metric_stat
                test_id = f"{method}:{metric}"
                metric_tests[test_id] = metric_stat["permutation_p_two_sided"]
            all_tests.update(metric_tests)
            adjusted = {}
            violation_diffs = [(scene, method_values[scene]["safety_violations"] -
                                control_values[scene]["safety_violations"])
                               for scene in sorted(usable_pair_ids) if "safety_violations" in method_values[scene]
                               and "safety_violations" in control_values[scene]]
            no_new_violation = bool(violation_diffs) and all(value <= 0 for _, value in violation_diffs)
            completion_ci = stats["completion_rate"]["ci95"]
            cost_ci = stats["online_method_cost"]["ci95"]
            required_complete = (len(paired_scenes) > 0 and safety_complete and no_new_violation
                and not scope_mismatches and not incomplete_groups
                and usable_pair_ids == (set(method_scene) | set(control_scene))
                and not (set(method_scene) - set(control_scene))
                and not (set(control_scene) - set(method_scene))
                and stats["completion_rate"]["n_scenes"] == len(paired_scenes)
                and stats["online_method_cost"]["n_scenes"] == len(paired_scenes)
                and all(item.get("metrics", {}).get("reduction_pass") is True
                        for scene in paired_scenes for item in (*method_scene[scene], *control_scene[scene])))
            meets = (completion_ci is not None and cost_ci is not None
                     and completion_ci[0] > -0.02 and cost_ci[0] > 0.0)
            if not required_complete:
                status = "INCONCLUSIVE"
            elif split == "validation":
                status = "VALIDATION_ONLY_NO_TEST_OR_OOD_SUPPORT"
            elif split == "train":
                status = "TRAINING_DIAGNOSTIC_ONLY"
            else:
                status = "COMPLETE_NUMERICAL_ASSESSMENT_REQUIRES_PRE_REGISTERED_DECISION_REVIEW"
            method_summaries[method] = {
                "status": status, "dataset_stage": _stage_label(split),
                "primary_method": primary_method, "paired_scene_count": len(usable_pair_ids),
                "unpaired_method_scenes": sorted(set(method_scene) - set(control_scene)),
                "unpaired_primary_scenes": sorted(set(control_scene) - set(method_scene)),
                "noncomparable_scope_scenes": scope_mismatches,
                "incomplete_scene_method_groups": incomplete_groups,
                "training_seed_aggregation": "mean within scene and method before pairing",
                "per_method_episode_metrics": {
                    metric: {"mean": (sum(values[metric] for values in method_values.values()
                                         if metric in values) /
                                      sum(metric in values for values in method_values.values())
                                      if any(metric in values for values in method_values.values()) else None),
                             "sample_sd": _sample_sd([values[metric] for values in method_values.values()
                                                       if metric in values]),
                             "n_scenes": sum(metric in values for values in method_values.values())}
                    for metric in METRICS},
                "primary_method_episode_metrics": {
                    metric: {"mean": (sum(values[metric] for values in control_values.values()
                                         if metric in values) /
                                      sum(metric in values for values in control_values.values())
                                      if any(metric in values for values in control_values.values()) else None),
                             "sample_sd": _sample_sd([values[metric] for values in control_values.values()
                                                       if metric in values]),
                             "n_scenes": sum(metric in values for values in control_values.values())}
                    for metric in METRICS},
                "paired_favorable_differences": stats,
                "holm_adjusted_p_values": adjusted,
                "missing_required_metric_scenes": {key: sorted(set(value)) for key, value in missing_metrics.items()},
                "forced_safety_action_evidence_complete": safety_complete,
                "forced_safety_action_evidence_status": "complete" if safety_complete else "unknown",
                "no_new_controllable_violation": no_new_violation,
                "decision_rule": {"completion_ci95_lower_gt_minus_0_02":
                                  completion_ci[0] > -0.02 if completion_ci else None,
                                  "online_cost_improvement_ci95_lower_gt_zero":
                                  cost_ci[0] > 0.0 if cost_ci else None,
                                  "required_all_safety_handling_complete": safety_complete,
                                  "required_no_new_controllable_violation": no_new_violation,
                                  "validation_never_supports_test_or_ood": True,
                                  "thresholds_met_numerically": meets,
                                  "numerical_thresholds_are_not_support_decisions": True},
            }
        adjusted_global = _holm(all_tests)
        for method, summary in method_summaries.items():
            summary["holm_adjusted_p_values"] = {
                metric: adjusted_global.get(f"{method}:{metric}") for metric in METRICS}
        result_splits[split] = {"dataset_stage": _stage_label(split),
                                "scene_count_in_registration": len({item["scenario_id"] for item in split_rows}),
                                "method_comparisons": method_summaries,
                                "holm_family": "all non-primary methods x all metrics with available permutation p-values within this split",
                                "holm_adjusted_p_values": adjusted_global}
    return {"schema": "closedloop_method_comparison_v1", "status": "COMPLETE",
            "primary_method": primary_method, "registry_path": str(registry_path),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "reductions_root": str(root), "bootstrap_replicates": bootstrap_replicates,
            "random_seed": seed, "scene_is_inference_unit": True,
            "input_scope": "independent episode_reduction.json only; no private state, raw trajectory, model, or training reads",
            "episode_reducer_pass_does_not_imply_method_support": True,
            "excluded_registration_rows": excluded_registration_rows,
            "splits": result_splits}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare independently reduced episodes by registered scene.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reductions-root", type=Path, required=True)
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.registry, args.reductions_root, args.primary_method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "splits": sorted(result["splits"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
