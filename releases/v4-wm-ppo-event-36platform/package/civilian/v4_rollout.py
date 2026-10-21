"""Full-episode V4 PPO training and evaluation orchestration."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from .event_environment import EventReliefEnv, _new_action, generate_reconstruction_candidates, repair_action
    from .event_policy import (
        _is_noop, _mask_from_trigger,
    )
    from .event_trigger import TriggerEngine, identify_affected_scope, high_level_action_mask
    from .v4_models import predict_v4_ensemble
    from .v4_policy import (
        POLICY_SEEDS, VARIANTS, attach_smdp_gae, build_v4_policy,
        load_v4_checkpoint, save_v4_checkpoint, smdp_ppo_defaults, update_v4_policy,
        variant_policy_options,
    )
except ImportError:
    from event_environment import EventReliefEnv, _new_action, generate_reconstruction_candidates, repair_action
    from event_policy import _is_noop, _mask_from_trigger
    from event_trigger import TriggerEngine, identify_affected_scope, high_level_action_mask
    from v4_models import predict_v4_ensemble
    from v4_policy import POLICY_SEEDS, VARIANTS, attach_smdp_gae, build_v4_policy, load_v4_checkpoint, save_v4_checkpoint, smdp_ppo_defaults, update_v4_policy, variant_policy_options


def _finite_scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _wm_trigger_summary(prediction: dict[str, Any], candidates: Sequence[dict], *, no_uncertainty: bool = False) -> dict[str, float]:
    """Derive absolute predicted deterioration from the NOOP consequence."""
    noop_index = next((i for i, row in enumerate(candidates) if _is_noop(row)), None)
    if noop_index is None:
        raise ValueError("V4_TRIGGER_NOOP_REQUIRED")
    values = {}
    for name in ("progress", "delay", "energy", "failure", "communication_risk"):
        head = prediction.get(name)
        if not isinstance(head, torch.Tensor) or head.shape[0] != len(candidates):
            raise ValueError(f"V4_TRIGGER_HEAD_ALIGNMENT:{name}")
        values[name] = _finite_scalar(head[noop_index])
    disagreement = prediction.get("uncertainty", prediction.get("disagreement"))
    if disagreement is None:
        stds = [prediction.get(f"{name}_std") for name in
                ("progress", "delay", "energy", "failure", "communication_risk")]
        node_std = prediction.get("next_minute_std")
        node_mask = prediction.get("entity_mask")
        if isinstance(node_std, torch.Tensor):
            if isinstance(node_mask, torch.Tensor) and node_mask.shape == node_std.shape[:2]:
                expanded = node_mask.to(node_std.device, dtype=torch.bool).unsqueeze(-1).expand_as(node_std)
                stds.append(node_std[expanded])
            else:
                stds.append(node_std)
        disagreement = torch.cat([row.float().reshape(-1) for row in stds if isinstance(row, torch.Tensor)]).mean() if any(isinstance(row, torch.Tensor) for row in stds) else 0.0
    uncertainty = 0.0 if no_uncertainty else _finite_scalar(disagreement)
    degradation = (0.2 * max(0.0, values["delay"]) + 0.01 * max(0.0, values["energy"])
                   + 0.25 * max(0.0, values["failure"]) + 0.01 * max(0.0, values["communication_risk"])
                   - max(0.0, values["progress"]))
    return {"d_degradation": max(0.0, degradation),
            "d_uncertainty": max(0.0, uncertainty), "utility_noop": -degradation}


def _noop_candidate() -> dict:
    return {"candidate_key": "NOOP", "candidate_scope": "NOOP", "legal": True,
            "full_environment_action": _new_action(noop=True)}


def _predict_candidate_catalog(world_models: Sequence[Any], candidates: Sequence[dict],
                               history: Sequence[dict], noop_prediction: dict | None = None) -> dict:
    """Predict each physical action, reusing only an already-computed NOOP row."""
    rows = []
    for candidate in candidates:
        if _is_noop(candidate) and noop_prediction is not None:
            prediction = noop_prediction
        else:
            prediction = predict_v4_ensemble(world_models, [{
                "history": list(history), "action": candidate["full_environment_action"],
            }], device="cpu")
        rows.append(prediction)
    if not rows:
        raise ValueError("V4_WM_EMPTY_CANDIDATE_CATALOG")
    combined = {}
    for key, value in rows[0].items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1:
            if any(not isinstance(row.get(key), torch.Tensor) or row[key].shape != value.shape for row in rows):
                raise ValueError(f"V4_WM_CANDIDATE_PREDICTION_ALIGNMENT:{key}")
            combined[key] = torch.cat([row[key] for row in rows], dim=0)
        elif key == "entity_ids" and isinstance(value, (list, tuple)) and len(value) == 1:
            combined[key] = [copy.deepcopy(row[key][0]) for row in rows]
        else:
            combined[key] = copy.deepcopy(value)
    return combined


def _episode_catalog(env: Any, detection: dict, max_candidates: int, *, force_global: bool) -> tuple[list[dict], dict, dict]:
    """Prefer a feasible local catalog; widen only when scope or feasibility requires it."""
    affected = identify_affected_scope(env.public_observation(), detection)
    if force_global:
        affected["local_eligible"] = False
        affected["global_eligible"] = True
        affected["scope"] = "GLOBAL_REPLAN"
        candidates = generate_reconstruction_candidates(env, scope="GLOBAL_REPLAN", max_candidates=max_candidates)
        return candidates, affected, {"status": "global_ablation", "global_non_noop_candidate_count": sum(not _is_noop(row) for row in candidates)}
    must_global = (bool(detection.get("strong_event")) or not bool(affected.get("local_eligible"))
                   or bool(affected.get("global_eligible")))
    if not must_global:
        local = generate_reconstruction_candidates(env, scope=affected, max_candidates=max_candidates)
        feasible = [row for row in local if not _is_noop(row) and bool(row.get("legal", True))]
        if feasible:
            return local, affected, {"status": "local_first", "local_non_noop_candidate_count": len(feasible),
                                     "global_non_noop_candidate_count": 0}
        affected["local_eligible"] = False
        affected["global_eligible"] = True
        affected["scope"] = "GLOBAL_REPLAN"
        affected["reasons"] = list(affected.get("reasons", [])) + ["local_repair_no_feasible_solution"]
    candidates = generate_reconstruction_candidates(env, scope="GLOBAL_REPLAN", max_candidates=max_candidates)
    return candidates, affected, {"status": "global_fallback", "local_non_noop_candidate_count": 0,
                                 "global_non_noop_candidate_count": sum(not _is_noop(row) for row in candidates)}


def _post_step_new_queue_risks(current_report: dict, report: dict) -> set[str]:
    """Return canonical keys for queue risks newly present after one step."""
    existing = {json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in current_report.get("violations", [])}
    return {json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in report.get("violations", [])} - existing


_EXTERNAL_QUEUE_EVENT_TYPES = {
    "platform_failed", "platform_failure", "energy_fraction_drop", "energy_changed",
    "capability_loss", "load_change", "communication_changed", "environment_changed",
    "task_arrived", "task_cancelled", "task_cancel", "priority_change", "deadline_change",
}


def _queue_risk_entities(row: dict) -> set[tuple[str, str]]:
    entities = set()
    for field in ("platform_id", "affected_platform_ids"):
        value = row.get(field)
        values = value if isinstance(value, (list, tuple, set)) else [value]
        entities.update(("platform", str(item)) for item in values if item is not None)
    for field in ("task_id", "affected_task_ids"):
        value = row.get(field)
        values = value if isinstance(value, (list, tuple, set)) else [value]
        entities.update(("task", str(item)) for item in values if item is not None)
    return entities


def _compact_queue_safety_report(report: dict) -> dict:
    return {
        "forced": bool(report.get("forced")),
        "required_scope": report.get("required_scope"),
        "checked_task_count": report.get("checked_task_count"),
        "checked_platform_count": report.get("checked_platform_count"),
        "violations": [copy.deepcopy(row) for row in report.get("violations", []) if isinstance(row, dict)],
        "soft_deadline_risk_count": len(report.get("soft_deadline_risks", [])),
    }


def _external_queue_event_evidence(raw_events: Sequence[dict]) -> list[dict]:
    evidence = []
    for event in raw_events:
        if not isinstance(event, dict) or event.get("type") not in _EXTERNAL_QUEUE_EVENT_TYPES:
            continue
        entities: set[tuple[str, str]] = set()
        for field in ("platform_id", "source", "target", "task_id"):
            value = event.get(field)
            if value is not None:
                entity_type = "task" if field == "task_id" else "platform"
                entities.add((entity_type, str(value)))
        for field, entity_type in (("affected_platform_ids", "platform"), ("affected_task_ids", "task")):
            values = event.get(field, [])
            if isinstance(values, (list, tuple, set)):
                entities.update((entity_type, str(value)) for value in values if value is not None)
        for link in event.get("links", []):
            if isinstance(link, dict):
                entities.update(("platform", str(link[key])) for key in ("source", "target") if link.get(key) is not None)
        for cell in event.get("changed_cells", []):
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                entities.add(("cell", f"{cell[0]},{cell[1]}"))
        evidence.append({
            "time": event.get("time"), "type": event.get("type"), "event_id": event.get("event_id"),
            "entities": [{"type": kind, "id": identity} for kind, identity in sorted(entities)],
        })
    return evidence


def _classify_new_queue_risks(before_report: dict, after_report: dict,
                              action_entities: set[tuple[str, str]],
                              external_events: Sequence[dict],
                              unconfirmed_action_entities: set[tuple[str, str]] | None = None) -> list[dict]:
    before = {json.dumps(row, sort_keys=True, separators=(",", ":"))
              for row in before_report.get("violations", []) if isinstance(row, dict)}
    external_entities = {
        (str(entity.get("type")), str(entity.get("id")))
        for event in external_events for entity in event.get("entities", []) if isinstance(entity, dict)
    }
    results = []
    unconfirmed_action_entities = unconfirmed_action_entities or set()
    for violation in after_report.get("violations", []):
        if not isinstance(violation, dict):
            continue
        key = json.dumps(violation, sort_keys=True, separators=(",", ":"))
        if key in before:
            continue
        affected = _queue_risk_entities(violation)
        action_matches = sorted(action_entities & affected)
        unconfirmed_matches = sorted(unconfirmed_action_entities & affected)
        external_matches = sorted(external_entities & affected)
        if unconfirmed_matches:
            attribution = "unknown"
        elif action_matches and not external_matches:
            attribution = "controllable"
        elif external_matches and not action_matches:
            attribution = "external"
        else:
            attribution = "unknown"
        results.append({
            "violation": copy.deepcopy(violation), "attribution": attribution,
            "action_entity_matches": [{"type": kind, "id": identity} for kind, identity in action_matches],
            "unconfirmed_action_entity_matches": [{"type": kind, "id": identity} for kind, identity in unconfirmed_matches],
            "external_entity_matches": [{"type": kind, "id": identity} for kind, identity in external_matches],
        })
    return results


def _step_execution_evidence(observation: dict, repaired: dict, info: dict,
                             rejections: Sequence[dict], before_report: dict,
                             after_report: dict, post_observation: dict) -> dict:
    rejected_pairs = {(str(row.get("platform_id", "")), str(row.get("task_id", "")))
                      for row in rejections}
    operational = [row for row in info.get("operational_dispatches", []) if isinstance(row, dict)]
    submitted_dispatches = []
    before_tasks = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
    after_tasks = {str(row.get("id")): row for row in post_observation.get("tasks", []) if isinstance(row, dict)}
    before_platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    after_platforms = {str(row.get("id")): row for row in post_observation.get("platforms", []) if isinstance(row, dict)}
    action_entities: set[tuple[str, str]] = set()
    unconfirmed_action_entities: set[tuple[str, str]] = set()
    owner_changes = [copy.deepcopy(row) for row in info.get("raw_events", [])
                     if isinstance(row, dict) and row.get("type") == "task_owner_change"]
    owner_pairs = {(str(platform_id), str(row.get("task_id", "")))
                   for row in owner_changes for platform_id in row.get("new_owners", [])}
    for row in operational:
        pair = (str(row.get("platform_id", "")), str(row.get("task_id", "")))
        platform_id, task_id = pair
        previous_owners = set(map(str, before_tasks.get(task_id, {}).get("assignees", [])))
        if before_tasks.get(task_id, {}).get("assignee") is not None:
            previous_owners.add(str(before_tasks[task_id]["assignee"]))
        current_owners = set(map(str, after_tasks.get(task_id, {}).get("assignees", [])))
        if after_tasks.get(task_id, {}).get("assignee") is not None:
            current_owners.add(str(after_tasks[task_id]["assignee"]))
        state_confirmed = (
            (platform_id in current_owners and platform_id not in previous_owners)
            or (after_platforms.get(platform_id, {}).get("active_task_id") == task_id
                and before_platforms.get(platform_id, {}).get("active_task_id") != task_id)
            or (after_tasks.get(task_id, {}).get("started_at") is not None
                and before_tasks.get(task_id, {}).get("started_at") is None
                and platform_id in current_owners)
        )
        confirmed = pair in owner_pairs or state_confirmed
        submitted_dispatches.append({**copy.deepcopy(row), "execution_status":
                                     "rejected" if pair in rejected_pairs else "confirmed" if confirmed else "unconfirmed"})
        entities = {("platform", platform_id), ("task", task_id)}
        (action_entities if confirmed else unconfirmed_action_entities).update(entities)
    queue_changes = []
    for receipt in info.get("operation_receipts", []):
        if not isinstance(receipt, dict):
            continue
        before_queues = receipt.get("before_queues", {})
        after_queues = receipt.get("after_queues", {})
        changed = []
        for platform_id in sorted(set(before_queues) | set(after_queues)):
            old = list(before_queues.get(platform_id, []))
            new = list(after_queues.get(platform_id, []))
            if old == new:
                continue
            changed.append({"platform_id": str(platform_id), "before": old, "after": new})
            action_entities.add(("platform", str(platform_id)))
            action_entities.update(("task", str(task_id)) for task_id in old + new)
        if changed:
            operation = receipt.get("operation", {})
            queue_changes.append({"time": receipt.get("time"), "op": operation.get("op"),
                                  "task_id": operation.get("task_id"),
                                  "other_task_id": operation.get("other_task_id"), "queues": changed})
    for change in owner_changes:
        action_entities.add(("task", str(change.get("task_id", ""))))
        action_entities.update(("platform", str(value)) for value in
                               change.get("previous_owners", []) + change.get("new_owners", []))
    external_events = _external_queue_event_evidence(info.get("raw_events", []))
    new_risks = _classify_new_queue_risks(before_report, after_report, action_entities,
                                          external_events, unconfirmed_action_entities)
    return {
        "time": observation.get("time"), "scope": "observed_public_task_queue_feasibility",
        "pre_action_safety": _compact_queue_safety_report(before_report),
        "post_step_safety": _compact_queue_safety_report(after_report),
        "submitted_to_execution_dispatches": submitted_dispatches,
        "rejected_dispatches": [copy.deepcopy(row) for row in rejections],
        "selected_dispatches": copy.deepcopy(repaired.get("dispatches", [])),
        "queue_changes": queue_changes, "owner_changes": owner_changes,
        "external_events": external_events, "new_risks": new_risks,
        "attribution_basis": "entity overlap is associative evidence, not proof of causation",
    }


def _region_for_position(observation: dict, position: Any) -> str | None:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    boundaries = observation.get("map", {}).get("region_boundaries")
    if boundaries != [0, 33, 66, 100]:
        return None
    x, y = float(position[0]), float(position[1])
    col = next((i for i in range(3) if boundaries[i] <= x < boundaries[i + 1]), None)
    row = next((i for i in range(3) if boundaries[i] <= y < boundaries[i + 1]), None)
    return f"r{row}{col}" if row is not None and col is not None else None


def _related_action_entities(observation: dict, action: dict) -> dict:
    platform_ids, task_ids = set(), set()
    for row in action.get("dispatches", []):
        if isinstance(row, dict):
            platform_ids.add(str(row.get("platform_id", "")))
            task_ids.add(str(row.get("task_id", "")))
    for row in action.get("queue_operations", []):
        if not isinstance(row, dict):
            continue
        for key in ("from_platform", "to_platform"):
            if row.get(key) is not None:
                platform_ids.add(str(row[key]))
        for key in ("task_id", "other_task_id", "before_task_id"):
            if row.get(key) is not None:
                task_ids.add(str(row[key]))
    platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    tasks = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
    return {
        "platforms": [{"id": key, "home_region": platforms[key].get("home_region"),
                       "position": copy.deepcopy(platforms[key].get("position")),
                       "position_region": _region_for_position(observation, platforms[key].get("position"))}
                      for key in sorted(platform_ids) if key in platforms],
        "tasks": [{"id": key, "region_id": tasks[key].get("region_id"),
                   "position": copy.deepcopy(tasks[key].get("position"))}
                  for key in sorted(task_ids) if key in tasks],
    }


def _live_safety_force(observation: dict, detection: dict) -> tuple[bool, list[str]]:
    """Revalidate strong-event evidence against the current public snapshot."""
    if not bool(detection.get("strong_event")):
        return False, []
    details = detection.get("event_details", {})
    platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    tasks = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
    reasons = []
    failed_ids = set(map(str, details.get("failed_platform_ids", [])))
    if any(pid in platforms and bool(platforms[pid].get("failed")) for pid in failed_ids):
        reasons.append("platform_failure_live")
    critical_ids = set(map(str, details.get("new_task_ids", []))) | set(map(str, details.get("priority_change_ids", []))) | set(map(str, details.get("deadline_change_ids", [])))
    now = int(observation.get("time", 0))
    if any(tid in tasks and (float(tasks[tid].get("priority", 0.0)) >= 5.0 or int(tasks[tid].get("deadline", 10**9)) - now <= 4) for tid in critical_ids):
        reasons.append("critical_task_live")
    if bool(details.get("network_split")) and len(_public_components(observation)) > 1:
        reasons.append("network_split_live")
    if not reasons:
        raise ValueError("V4_STRONG_EVENT_FAILED_LIVE_SAFETY_REVALIDATION")
    return True, reasons


def _public_components(observation: dict) -> list[set[str]]:
    nodes = {str(row.get("id")) for row in observation.get("platforms", []) if isinstance(row, dict)}
    adjacency = {node: set() for node in nodes}
    for link in observation.get("communication", []):
        if not isinstance(link, dict) or not bool(link.get("connected", float(link.get("quality", 0.0)) >= 0.3)):
            continue
        left, right = str(link.get("source")), str(link.get("target"))
        if left in nodes and right in nodes:
            adjacency[left].add(right); adjacency[right].add(left)
    groups, seen = [], set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack, group = [start], set()
        seen.add(start)
        while stack:
            node = stack.pop(); group.add(node)
            for other in adjacency[node] - seen:
                seen.add(other); stack.append(other)
        groups.append(group)
    return groups


def _episode_cost(metrics: dict[str, Any], online_cpu_seconds: float, cpu_weight: float) -> dict[str, float]:
    utility = metrics.get("global_utility", {})
    value = utility.get("value") if isinstance(utility, dict) else utility
    if value is None and isinstance(metrics.get("metrics"), dict):
        inner = metrics["metrics"].get("global_utility", {})
        value = inner.get("value") if isinstance(inner, dict) else inner
    if value is None or not math.isfinite(float(value)):
        raise ValueError("V4_VALIDATION_UTILITY_REQUIRED")
    physical_cost = -float(value)
    compute_cost = max(0.0, float(online_cpu_seconds)) * max(0.0, float(cpu_weight))
    return {"physical_cost": physical_cost, "online_cpu_cost": compute_cost,
            "total_cost": physical_cost + compute_cost}


def _threshold_key(variant: str) -> str:
    if variant.endswith("-NoU"):
        return "world_model_no_uncertainty"
    if variant in {"Event-G-WM", "Event-G-WM-Global"}:
        return "world_model"
    return "public"


def evaluate_v4_episode(scenario: dict, policy, *, variant: str, seed: int,
                        world_models: Sequence[Any] | None = None, config: dict | None = None,
                        deterministic: bool = True) -> tuple[list[dict], dict]:
    """Run one complete minute-level environment episode with public inputs."""
    config = dict(config or {})
    use_wm = variant in VARIANTS and "-WM" in variant
    if policy.use_wm_actor != use_wm:
        raise ValueError("V4_VARIANT_POLICY_BINDING")
    if use_wm and (world_models is None or len(world_models) != 3):
        raise ValueError("V4_WM_ENSEMBLE_REQUIRED")
    threshold = config.get("threshold_manifest", {})
    if not isinstance(threshold, dict):
        raise ValueError("V4_THRESHOLD_MANIFEST")
    trigger_config = config.get("trigger", {})
    env = EventReliefEnv(scenario)
    observation = env.reset()
    try:
        from . import gain_safety
    except ImportError:
        import gain_safety
    previous = copy.deepcopy(observation)
    history = [copy.deepcopy(observation)]
    engine = TriggerEngine(trigger_config, threshold)
    max_candidates = int(config.get("max_candidates", 16))
    horizon = int(scenario.get("horizon", scenario.get("horizon_minutes", 90)))
    if horizon < 1:
        raise ValueError("V4_EPISODE_HORIZON")
    records = []
    started = time.perf_counter()
    phase_seconds = {key: 0.0 for key in ("safety", "detection", "features", "noop_prediction",
                                          "candidate_prediction", "planning", "policy", "repair", "environment")}
    online_cpu_seconds = 0.0
    terminated = truncated = False
    forced_safety_reports = resolved_safety_reports = 0
    repair_check_receipts = []
    submitted_action_rejections = []
    minute_execution_evidence = []
    post_step_new_queue_risk_keys = set()
    pending = None
    decision_trace = []
    cross_region_support_task_ids = set()
    idle_rewards: list[float] = []
    idle_audit: list[dict] = []
    idle_observation = None
    initial_critic_value = None
    for step_index in range(horizon):
        if terminated or truncated:
            break
        before_safety = copy.deepcopy(observation)
        phase_wall = time.perf_counter(); phase_cpu = time.process_time()
        safety_report = gain_safety.assess_queue_safety(before_safety)
        safety_apply = gain_safety.apply_safety_disposition(env, before_safety, safety_report)
        observation = env.public_observation()
        safety_verify = gain_safety.verify_live_safety_disposition(
            env, before_safety, safety_report, safety_apply, observation)
        phase_seconds["safety"] += time.perf_counter() - phase_wall
        elapsed_cpu = time.process_time() - phase_cpu
        online_cpu_seconds += elapsed_cpu
        if safety_verify.get("resolved") is not True:
            raise ValueError("V4_POLICY_SAFETY_DISPOSITION_UNRESOLVED")
        forced_safety_reports += int(bool(safety_report.get("forced")))
        resolved_safety_reports += int(bool(safety_report.get("forced")) and bool(safety_verify.get("resolved")))
        # Safety may mutate this frame without advancing simulator time. Keep model and
        # trigger history on the post-disposition public state.
        history[-1] = copy.deepcopy(observation)
        forced_by_safety = bool(safety_report.get("forced"))
        phase_wall = time.perf_counter(); phase_cpu = time.process_time()
        detection = engine.detect(previous, observation, history=history[-4:])
        forced, safety_reasons = _live_safety_force(observation, detection)
        forced = forced or forced_by_safety
        if forced_by_safety:
            safety_reasons = list(safety_reasons) + ["resolved_queue_safety_disposition"]
        if forced:
            detection["strong_event"] = True
            detection["event_severity"] = "strong"
        phase_seconds["detection"] += time.perf_counter() - phase_wall
        elapsed_cpu = time.process_time() - phase_cpu
        online_cpu_seconds += elapsed_cpu
        trigger_predictions = None
        wm_summary = None
        model_trigger = use_wm and not variant.startswith("Periodic") and variant != "Event-G-WM-Decision"
        if model_trigger and not forced:
            noop = _noop_candidate()
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            trigger_predictions = predict_v4_ensemble(world_models,
                [{"history": history[-4:], "action": noop["full_environment_action"]}], device="cpu")
            phase_seconds["noop_prediction"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            wm_summary = _wm_trigger_summary(trigger_predictions, [noop], no_uncertainty=variant.endswith("-NoU"))
            phase_seconds["features"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            detection = engine.detect(previous, observation, history=history[-4:], wm_summary=wm_summary)
            forced, safety_reasons = _live_safety_force(observation, detection)
            if forced:
                detection["strong_event"] = True
                detection["event_severity"] = "strong"
            phase_seconds["detection"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
        mode = "periodic" if variant.startswith("Periodic") else "event"
        phase_wall = time.perf_counter(); phase_cpu = time.process_time()
        preliminary = engine.decide(detection, int(observation.get("time", step_index)), mode=mode)
        if forced:
            preliminary["triggered"] = True
            preliminary["cooldown_suppressed"] = False
            detection["triggered"] = True
            detection["cooldown_suppressed"] = False
        phase_seconds["features"] += time.perf_counter() - phase_wall
        elapsed_cpu = time.process_time() - phase_cpu
        online_cpu_seconds += elapsed_cpu
        predictions = None
        repair_receipt = None
        if not preliminary.get("triggered"):
            candidates = [_noop_candidate()]
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            affected, receipt = identify_affected_scope(observation, detection), {"status": "not_generated_nontrigger"}
            phase_seconds["features"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            high_mask, low_mask = [True, False, False], [True]
            if initial_critic_value is None:
                phase_wall = time.perf_counter(); phase_cpu = time.process_time()
                with torch.no_grad():
                    value = policy.value(observation)
                phase_seconds["policy"] += time.perf_counter() - phase_wall
                elapsed_cpu = time.process_time() - phase_cpu
                online_cpu_seconds += elapsed_cpu
                initial_critic_value = float(value.detach().cpu())
            output = {"scope_index": 0, "candidate_index": 0, "high_level_mask": high_mask,
                      "candidate_mask": low_mask, "log_probability": 0.0,
                      "value": float(initial_critic_value), "full_environment_action": candidates[0]["full_environment_action"]}
            repaired, repair_receipt = candidates[0]["full_environment_action"], {"status": "noop_not_repaired"}
        else:
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            candidates, affected, receipt = _episode_catalog(
                env, detection, max_candidates, force_global=variant.endswith("-Global"))
            if not candidates:
                raise ValueError("V4_EMPTY_CATALOG")
            phase_seconds["planning"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            if use_wm:
                phase_wall = time.perf_counter(); phase_cpu = time.process_time()
                predictions = _predict_candidate_catalog(world_models, candidates, history[-4:], trigger_predictions)
                phase_seconds["candidate_prediction"] += time.perf_counter() - phase_wall
                elapsed_cpu = time.process_time() - phase_cpu
                online_cpu_seconds += elapsed_cpu
            if variant.endswith("-Global"):
                affected["global_eligible"] = True
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            high_mask = _mask_from_trigger(preliminary, detection, affected, high_level_action_mask)
            if variant.endswith("-Global"):
                high_mask[1] = False
                high_mask[2] = any(str(row.get("candidate_scope", "")).upper() in {"GLOBAL", "GLOBAL_REPLAN"} for row in candidates)
            phase_seconds["features"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            with torch.no_grad():
                output = policy.act(observation, candidates, high_mask, None, predictions if use_wm else None,
                                    deterministic=deterministic)
            phase_seconds["policy"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            phase_wall = time.perf_counter(); phase_cpu = time.process_time()
            repaired, repair_receipt = repair_action(env, output["full_environment_action"])
            phase_seconds["repair"] += time.perf_counter() - phase_wall
            elapsed_cpu = time.process_time() - phase_cpu
            online_cpu_seconds += elapsed_cpu
            repair_check_receipts.append({
                "time": observation.get("time"),
                "legal_before": repair_receipt.get("legal_before"),
                "issues_before": copy.deepcopy(repair_receipt.get("issues_before")),
                "postrepair_legal": repair_receipt.get("postrepair_legal"),
                "postrepair_issues": copy.deepcopy(repair_receipt.get("postrepair_issues")),
                "repaired": repair_receipt.get("repaired"),
            })
        decision = copy.deepcopy(preliminary)
        decision.update({"affected_scope": copy.deepcopy(affected), "high_level_mask": list(output["high_level_mask"]),
                         "scope_catalog_receipt": copy.deepcopy(receipt), "strong_event": bool(forced),
                         "safety_revalidation": {"passed": bool(forced), "reasons": safety_reasons},
                         "world_model_trigger_summary": copy.deepcopy(wm_summary)})
        env_start = time.perf_counter()
        next_observation, reward, terminated, truncated, info = env.step(repaired)
        phase_seconds["environment"] += time.perf_counter() - env_start
        rejection_start = len(submitted_action_rejections)
        current_dispatches = {(str(row.get("platform_id", "")), str(row.get("task_id", "")))
                              for row in repaired.get("dispatches", []) if isinstance(row, dict)}
        operational_sources = {(str(row.get("platform_id", "")), str(row.get("task_id", ""))):
                               str(row.get("source", "unknown"))
                               for row in info.get("operational_dispatches", []) if isinstance(row, dict)}
        for rejection in info.get("raw_events", []):
            if not isinstance(rejection, dict) or rejection.get("type") not in {"dispatch_rejected", "action_rejected"}:
                continue
            pair = (str(rejection.get("platform_id", "")), str(rejection.get("task_id", "")))
            source = operational_sources.get(pair)
            submitted_action_rejections.append({
                "time": observation.get("time"), "type": rejection.get("type"),
                "candidate_id": rejection.get("candidate_id"), "platform_id": rejection.get("platform_id"),
                "task_id": rejection.get("task_id"), "reason": rejection.get("reason"),
                "source": source or "unknown", "submitted_in_current_step": pair in current_dispatches,
            })
        safety_wall = time.perf_counter(); safety_cpu = time.process_time()
        next_safety = gain_safety.assess_queue_safety(next_observation)
        phase_seconds["safety"] += time.perf_counter() - safety_wall
        online_cpu_seconds += time.process_time() - safety_cpu
        # Disturbances applied by the simulator can create a new risk after a legal
        # action; keep that separate from policy/action feasibility accounting.
        current_safety = safety_verify.get("remaining_report", safety_report)
        minute_execution_evidence.append(_step_execution_evidence(
            observation, repaired, info, submitted_action_rejections[rejection_start:],
            current_safety, next_safety, next_observation))
        post_step_new_queue_risk_keys.update(_post_step_new_queue_risks(current_safety, next_safety))
        if preliminary.get("triggered"):
            selected_action = output.get("full_environment_action", {})
            requested_pairs = {(str(row.get("platform_id", "")), str(row.get("task_id", "")))
                               for row in selected_action.get("dispatches", []) if isinstance(row, dict)}
            for operation in selected_action.get("queue_operations", []):
                if not isinstance(operation, dict):
                    continue
                task_id = str(operation.get("task_id", ""))
                target = operation.get("to_platform")
                if target is not None:
                    requested_pairs.add((str(target), task_id))
                if operation.get("op") == "swap":
                    other_task = str(operation.get("other_task_id", ""))
                    source = operation.get("from_platform")
                    if source is not None:
                        requested_pairs.add((str(source), other_task))
            task_rows = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
            platform_rows = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
            for change in info.get("raw_events", []):
                if not isinstance(change, dict) or change.get("type") != "task_owner_change":
                    continue
                task_id = str(change.get("task_id", ""))
                task_region = task_rows.get(task_id, {}).get("region_id")
                for platform_id in change.get("new_owners", []):
                    platform_id = str(platform_id)
                    if ((platform_id, task_id) in requested_pairs
                            and task_region is not None
                            and platform_rows.get(platform_id, {}).get("home_region") != task_region):
                        cross_region_support_task_ids.add(task_id)
            selected_wm_means = {}
            if isinstance(predictions, dict):
                index = int(output.get("candidate_index", 0))
                for name in ("progress", "delay", "energy", "failure", "communication_risk"):
                    head = predictions.get(name)
                    if isinstance(head, torch.Tensor) and head.ndim and index < head.shape[0]:
                        selected_wm_means[name] = _finite_scalar(head[index])
            decision_trace.append({
                "time": observation.get("time"), "candidate_key": output.get("candidate_key"),
                "scope": selected_action.get("scope"),
                "selected_action": copy.deepcopy(selected_action), "executed_action": copy.deepcopy(repaired),
                "effective_reconstruction_count": int(info.get("effective_reconstruction_count", 0)),
                "effective_change": {"operation_receipts": copy.deepcopy(info.get("operation_receipts", [])),
                                     "operational_dispatches": copy.deepcopy(info.get("operational_dispatches", [])),
                                     "owner_changes": [copy.deepcopy(row) for row in info.get("raw_events", [])
                                                       if isinstance(row, dict) and row.get("type") == "task_owner_change"],
                                     "execution_rejections": [copy.deepcopy(row) for row in info.get("raw_events", [])
                                                               if isinstance(row, dict) and row.get("type") in {"dispatch_rejected", "action_rejected"}]},
                "repair_check": copy.deepcopy(repair_receipt),
                "related_entities_before": _related_action_entities(observation, selected_action),
                "selected_world_model_means": selected_wm_means or None,
            })
        if "effective_reconstruction_count" not in info:
            raise ValueError("V4_EFFECTIVE_RECONSTRUCTION_REQUIRED")
        engine.record_reconstruction(int(observation.get("time", step_index)),
                                     bool(int(info["effective_reconstruction_count"]) > 0))
        audit = {"step": step_index, "time": observation.get("time"), "reward": float(reward),
                 "triggered": bool(preliminary.get("triggered")),
                 "safety_forced": forced_by_safety, "external_safety_risk_count": len(next_safety.get("violations", []))}
        if not preliminary.get("triggered"):
            if pending is not None:
                pending["minute_rewards"].append(float(reward))
                pending["minute_audit"].append(audit)
                pending["dt"] += 1
                pending["reward"] += float(reward)
            else:
                if idle_observation is None:
                    idle_observation = copy.deepcopy(observation)
                idle_rewards.append(float(reward))
                idle_audit.append(audit)
        else:
            if pending is not None:
                pending["bootstrap_value"] = float(output["value"])
                records.append(pending)
                pending = None
            elif idle_rewards:
                records.append({
                    "variant": variant, "seed": int(seed), "parent_id": scenario.get("parent_id", scenario.get("scenario_id")),
                    "split": scenario.get("split"), "step": idle_audit[0]["step"], "time": idle_audit[0]["time"],
                    "observation": copy.deepcopy(idle_observation), "candidates": [_noop_candidate()],
                    "scope_index": 0, "candidate_index": 0, "high_level_mask": [True, False, False],
                    "candidate_mask": [True], "old_log_probability": 0.0, "old_value": float(initial_critic_value),
                    "wm_predictions": None, "decision": {"triggered": False, "critic_only_initial_interval": True},
                    "detection": {}, "selected_action": _noop_candidate()["full_environment_action"],
                    "executed_action": _noop_candidate()["full_environment_action"],
                    "minute_rewards": list(idle_rewards), "minute_audit": copy.deepcopy(idle_audit),
                    "dt": len(idle_rewards), "reward": sum(idle_rewards),
                    "terminated": False, "truncated": False,
                    "bootstrap_value": float(output["value"]),
                })
                idle_rewards.clear(); idle_audit.clear(); idle_observation = None
            pending = {
            "variant": variant, "seed": int(seed), "parent_id": scenario.get("parent_id", scenario.get("scenario_id")),
            "split": scenario.get("split"), "step": step_index, "time": observation.get("time"),
            "observation": copy.deepcopy(observation), "candidates": copy.deepcopy(candidates),
            "scope_index": int(output.get("scope_index", 0)), "candidate_index": int(output.get("candidate_index", 0)),
            "candidate_key": output.get("candidate_key"),
            "high_level_mask": list(output["high_level_mask"]), "candidate_mask": list(output["candidate_mask"]),
            "old_log_probability": float(output["log_probability"]), "old_value": float(output["value"]),
            "wm_predictions": copy.deepcopy(predictions) if use_wm else None,
            "decision": decision, "detection": copy.deepcopy(detection), "selected_action": copy.deepcopy(output["full_environment_action"]),
            "safety_report": safety_report, "safety_apply_receipt": safety_apply,
            "safety_verification": safety_verify,
            "executed_action": copy.deepcopy(repaired), "repair_receipt": copy.deepcopy(repair_receipt),
            "effective_reconstruction_count": int(info["effective_reconstruction_count"]),
            "minute_rewards": [float(reward)], "minute_audit": [audit], "dt": 1, "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated),
            }
        if pending is not None:
            pending["terminated"] = bool(terminated)
            pending["truncated"] = bool(truncated)
        previous, observation = copy.deepcopy(observation), next_observation
        history.append(copy.deepcopy(observation))
        history = history[-4:]
    if pending is not None:
        records.append(pending)
    elif idle_rewards:
        records.append({
            "variant": variant, "seed": int(seed), "parent_id": scenario.get("parent_id", scenario.get("scenario_id")),
            "split": scenario.get("split"), "step": idle_audit[0]["step"], "time": idle_audit[0]["time"],
            "observation": copy.deepcopy(idle_observation), "candidates": [_noop_candidate()],
            "scope_index": 0, "candidate_index": 0, "high_level_mask": [True, False, False],
            "candidate_mask": [True], "old_log_probability": 0.0, "old_value": float(initial_critic_value),
            "wm_predictions": None, "decision": {"triggered": False, "critic_only_initial_interval": True},
            "detection": {}, "selected_action": _noop_candidate()["full_environment_action"],
            "executed_action": _noop_candidate()["full_environment_action"],
            "minute_rewards": list(idle_rewards), "minute_audit": copy.deepcopy(idle_audit),
            "dt": len(idle_rewards), "reward": sum(idle_rewards),
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
    if not records:
        raise ValueError("V4_EMPTY_EPISODE")
    if truncated and not terminated:
        phase_wall = time.perf_counter(); phase_cpu = time.process_time()
        with torch.no_grad():
            records[-1]["bootstrap_value"] = float(policy.value(observation).detach().cpu())
        phase_seconds["policy"] += time.perf_counter() - phase_wall
        online_cpu_seconds += time.process_time() - phase_cpu
    attach_smdp_gae(records, gamma=float(config.get("gamma", 0.99)), gae_lambda=float(config.get("gae_lambda", 0.95)))
    metrics = env.episode_metrics()
    outcomes = metrics.get("task_outcomes", [])
    priority_total = sum(max(0.0, float(row.get("priority", 0.0))) for row in outcomes)
    weighted_completion = (sum(max(0.0, float(row.get("priority", 0.0))) * float(bool(row.get("completed")))
                               for row in outcomes) / priority_total if priority_total else None)
    cost = _episode_cost(metrics, online_cpu_seconds, float(config.get("planning_cpu_cost_weight_per_second", 0.01)))
    postrepair_failures = [row for row in repair_check_receipts if row.get("postrepair_legal") is False]
    repair_input_failures = [row for row in repair_check_receipts if row.get("legal_before") is False]
    rejection_reason_counts = {}
    rejection_source_counts = {}
    for row in submitted_action_rejections:
        reason = str(row.get("reason", "unknown"))
        source = str(row.get("source", "unknown"))
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
        rejection_source_counts[source] = rejection_source_counts.get(source, 0) + 1
    attribution_counts = {name: 0 for name in ("controllable", "external", "unknown")}
    for minute in minute_execution_evidence:
        for risk in minute.get("new_risks", []):
            attribution_counts[risk.get("attribution", "unknown")] += 1
    queue_evidence_complete = (len(minute_execution_evidence) == sum(
        len(x.get("minute_audit", [])) for x in records)
        and all(row.get("scope") == "observed_public_task_queue_feasibility"
                and isinstance(row.get("pre_action_safety"), dict)
                and isinstance(row.get("post_step_safety"), dict)
                for row in minute_execution_evidence))
    scoped_queue_violations = attribution_counts["controllable"] if queue_evidence_complete else None
    summary = {"parent_id": scenario.get("parent_id", scenario.get("scenario_id")), "split": scenario.get("split"),
               "ticks": sum(len(x.get("minute_audit", [])) for x in records),
               "decision_interval_count": len(records), "total_reward": sum(x["reward"] for x in records),
               "validation_cost": cost["total_cost"], **cost, "online_cpu_seconds": online_cpu_seconds,
               "trigger_count": sum(bool(x.get("decision", {}).get("triggered")) for x in records),
               "decision_trace": decision_trace,
               "cross_region_support_count": len(cross_region_support_task_ids),
               "effective_reconstruction_count": sum(int(x.get("effective_reconstruction_count", 0)) > 0 for x in records),
               "weighted_completion_rate": weighted_completion,
               "forced_safety_report_count": forced_safety_reports,
               "resolved_forced_safety_report_count": resolved_safety_reports,
               "forced_safety_handling_rate": (resolved_safety_reports / forced_safety_reports
                                                if forced_safety_reports else None),
               "repair_action_check": {
                   "checked_trigger_action_count": len(repair_check_receipts),
                   "input_illegal_count": len(repair_input_failures),
                   "postrepair_illegal_count": len(postrepair_failures),
                   "receipts": repair_check_receipts,
               },
               "submitted_action_rejections": {
                   "count": len(submitted_action_rejections),
                   "reason_counts": rejection_reason_counts,
                   "source_counts": rejection_source_counts,
                   "records": submitted_action_rejections,
               },
               "minute_execution_evidence": minute_execution_evidence,
               "actual_executed_controllable_hard_constraint_violations": scoped_queue_violations,
               "actual_executed_controllable_hard_constraint_assessment": {
                   "status": "complete" if queue_evidence_complete else "unassessed",
                   "scope": "observed_public_task_queue_feasibility",
                   "attribution_counts": attribution_counts,
                   "unknown_risk_count": attribution_counts["unknown"],
                   "reason": ("Compares public queue-safety violations before action and after execution; entity overlap is associative evidence, not proof of causation. This does not assess complete physical or flight safety."
                              if queue_evidence_complete else "Per-minute public queue-safety execution evidence is incomplete."),
               },
               "newly_introduced_controllable_hard_constraint_violations": scoped_queue_violations,
               "executed_action_feasibility_violations": scoped_queue_violations,
               "controllable_violation_scope": "observed_public_task_queue_feasibility; not complete physical or flight safety",
               "new_queue_risk_attribution_counts": attribution_counts,
               "post_step_new_queue_risk_count": len(post_step_new_queue_risk_keys),
               "environment_metrics": metrics,
               "phase_seconds": {**phase_seconds, "total": time.perf_counter() - started,
                                 "online_cpu_accounted": online_cpu_seconds}}
    return records, summary


def _calibrate_thresholds(train_scenarios: Sequence[dict], world_models: Sequence[Any] | None,
                          config: dict) -> dict:
    """Calibrate thresholds from training scenarios and public observations only."""
    try:
        from .event_trigger import calibrate_threshold
        from . import v4_data
    except ImportError:
        from event_trigger import calibrate_threshold
        import v4_data
    trigger_config = copy.deepcopy(config.get("trigger", {}))
    weights = trigger_config.get("weights", {"d_state": 0.35, "d_degradation": 0.30, "d_margin": 0.25, "d_uncertainty": 0.10})
    if isinstance(weights, (list, tuple)):
        weights = dict(zip(("d_state", "d_degradation", "d_margin", "d_uncertainty"), map(float, weights)))
    trigger_config["weights"] = weights
    trigger_config.setdefault("quantile", 0.75)
    trigger_config.setdefault("delta_min", 4)
    sequences = []
    for scenario in train_scenarios:
        env = EventReliefEnv(scenario)
        observation = env.reset()
        observations = []
        horizon = int(scenario.get("horizon", scenario.get("horizon_minutes", 90)))
        while env.time < horizon:
            observation, _ = v4_data._safety_disposition(env)
            observations.append(copy.deepcopy(observation))
            _, _, terminated, truncated, _ = env.step(copy.deepcopy(_noop_candidate()["full_environment_action"]))
            if terminated or truncated:
                break
        sequences.append({"split": "train", "parent_id": str(scenario.get("parent_id", scenario.get("scenario_id"))),
                          "observations": observations, "wm_summaries": [None] * len(observations)})
    public_manifest = calibrate_threshold(sequences, trigger_config)
    wm_sequences = copy.deepcopy(sequences)
    wm_no_u_manifest = None
    if world_models:
        for sequence in wm_sequences:
            observations = sequence["observations"]
            summaries = [None] * len(observations)
            for index in range(1, len(observations)):
                noop = _noop_candidate()
                samples = [{"history": observations[max(0, index - 3):index + 1],
                            "action": noop["full_environment_action"]}]
                predictions = predict_v4_ensemble(world_models, samples, device="cpu")
                summaries[index] = _wm_trigger_summary(predictions, [noop])
            sequence["wm_summaries"] = summaries
        wm_manifest = calibrate_threshold(wm_sequences, trigger_config)
        no_u_sequences = copy.deepcopy(wm_sequences)
        for sequence in no_u_sequences:
            sequence["wm_summaries"] = [
                None if summary is None else {**summary, "d_uncertainty": 0.0}
                for summary in sequence.get("wm_summaries", [])
            ]
        wm_no_u_manifest = calibrate_threshold(no_u_sequences, trigger_config)
    else:
        wm_manifest = public_manifest
        wm_no_u_manifest = public_manifest
    for manifest in (public_manifest, wm_manifest):
        manifest.update({"source_split": "train", "source_parent_ids": sorted(str(s.get("parent_id", s.get("scenario_id"))) for s in train_scenarios)})
    wm_no_u_manifest.update({"source_split": "train", "source_parent_ids": sorted(str(s.get("parent_id", s.get("scenario_id"))) for s in train_scenarios)})
    return {"public": public_manifest, "world_model": wm_manifest, "world_model_no_uncertainty": wm_no_u_manifest}


def train_v4_methods(train_scenarios: Sequence[dict], validation_scenarios: Sequence[dict],
                     world_models: Sequence[Any] | None, output_dir: str | Path,
                     config: dict | None = None, source_identity: str = "") -> dict[str, Any]:
    """Train all seven methods on complete episodes and retain best/last states."""
    settings = smdp_ppo_defaults()
    settings.update(config or {})
    seeds = tuple(int(x) for x in settings.get("seeds", POLICY_SEEDS))
    if seeds != POLICY_SEEDS or not train_scenarios or not validation_scenarios or not source_identity:
        raise ValueError("V4_TRAIN_VALIDATION_OR_SEED_CONTRACT")
    if any(row.get("split") not in (None, "train") for row in train_scenarios):
        raise ValueError("V4_TRAIN_SCENARIO_SPLIT")
    if any(row.get("split") not in (None, "validation", "development") for row in validation_scenarios):
        raise ValueError("V4_VALIDATION_SCENARIO_SPLIT")
    device = torch.device(str(settings.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V4_POLICY_CUDA_REQUIRED")
    if len(world_models or []) != 3:
        raise ValueError("V4_POLICY_WORLD_MODEL_ENSEMBLE_REQUIRED")
    world_model_state_before = [{key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                                for model in world_models]
    for model in world_models:
        model.eval().requires_grad_(False)
    calibration = settings.get("calibration") or _calibrate_thresholds(train_scenarios, world_models, settings)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "V4_PPO_TRAINING_PROGRESS.jsonl"
    rounds = int(settings.get("rounds", 128)); episodes_per_round = int(settings.get("episodes_per_round", 4))
    if rounds != 128 or episodes_per_round != 4:
        raise ValueError("V4_FULL_PPO_BUDGET_REQUIRED")
    result: dict[str, Any] = {"schema": "n02-v4-ppo-training/1", "status": "running", "variants": {},
                              "calibration": calibration, "config": {k: settings[k] for k in
                              ("seeds", "rounds", "episodes_per_round", "learning_rate", "epochs", "minibatch_size", "clip_ratio", "entropy_coefficient", "gamma", "gae_lambda")},
                              "selection_interval_rounds": 32, "source_identity": source_identity}
    for variant in VARIANTS:
        result["variants"][variant] = {}
        for seed in seeds:
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            policy = build_v4_policy(seed, **variant_policy_options(variant)).to(device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]))
            initial_policy_parameters = {name: value.detach().cpu().clone()
                                         for name, value in policy.named_parameters()}
            run_dir = output / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            best_cost = math.inf; best_path = run_dir / "best.pt"; last_path = run_dir / "last.pt"
            validation_curve = []; training_curve = []; timings = {"rollout": 0.0, "ppo": 0.0, "validation": 0.0}
            actor_updates = critic_updates = 0
            schedule_rng = random.Random(int(settings.get("episode_schedule_seed", 842701)))
            for round_index in range(rounds):
                train_rows = list(train_scenarios)
                schedule_rng.shuffle(train_rows)
                selected = [train_rows[(round_index * episodes_per_round + i) % len(train_rows)] for i in range(episodes_per_round)]
                batch_records = []; episode_rows = []
                for scenario in selected:
                    episode_config = {**settings, "threshold_manifest": calibration[_threshold_key(variant)]}
                    start = time.perf_counter()
                    records, summary = evaluate_v4_episode(scenario, policy, variant=variant, seed=seed,
                                                           world_models=world_models, config=episode_config,
                                                           deterministic=False)
                    timings["rollout"] += time.perf_counter() - start
                    batch_records.extend(records); episode_rows.append(summary)
                start = time.perf_counter()
                update_log = update_v4_policy(policy, optimizer, batch_records,
                    seed=seed + round_index * 100_003, epochs=int(settings["epochs"]),
                    minibatch_size=int(settings["minibatch_size"]), clip_ratio=float(settings["clip_ratio"]),
                    entropy_coefficient=float(settings["entropy_coefficient"]))
                timings["ppo"] += time.perf_counter() - start
                round_actor_updates = sum(int(row.get("actor_sample_count", 0) > 0) for row in update_log)
                round_critic_updates = len(update_log)
                actor_updates += round_actor_updates
                critic_updates += round_critic_updates
                training_curve.append({"round": round_index + 1, "mean_reward": sum(x["total_reward"] for x in episode_rows) / len(episode_rows),
                                       "transitions": len(batch_records), "optimizer_updates": len(update_log),
                                       "actor_updates": round_actor_updates, "critic_updates": round_critic_updates,
                                       "mean_actor_loss": sum(row["actor_loss"] for row in update_log) / len(update_log),
                                       "mean_critic_loss": sum(row["value_loss"] for row in update_log) / len(update_log)})
                progress_row = {"variant": variant, "seed": seed, "round": round_index + 1,
                                "mean_reward": training_curve[-1]["mean_reward"],
                                "transitions": len(batch_records), "actor_updates": round_actor_updates,
                                "critic_updates": round_critic_updates}
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress_row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                if (round_index + 1) % 32 == 0:
                    policy.to("cpu").eval()
                    start = time.perf_counter()
                    validation_rows = []
                    val_config = {**settings, "threshold_manifest": calibration[_threshold_key(variant)]}
                    for scenario in validation_scenarios:
                        _, summary = evaluate_v4_episode(scenario, policy, variant=variant, seed=seed,
                                                         world_models=world_models, config=val_config,
                                                         deterministic=True)
                        validation_rows.append(summary)
                    timings["validation"] += time.perf_counter() - start
                    mean_cost = sum(row["validation_cost"] for row in validation_rows) / len(validation_rows)
                    validation_curve.append({"round": round_index + 1, "mean_cost": mean_cost, "episodes": validation_rows})
                    policy.to(device).train()
                    if mean_cost < best_cost:
                        best_cost = mean_cost
                        save_v4_checkpoint(best_path, policy, optimizer, variant=variant, seed=seed,
                                           config={**settings, "learning_rate": float(settings["learning_rate"])},
                                           source_identity=source_identity)
            save_v4_checkpoint(last_path, policy, optimizer, variant=variant, seed=seed,
                               config={**settings, "learning_rate": float(settings["learning_rate"])},
                               source_identity=source_identity)
            if not best_path.is_file():
                raise RuntimeError("V4_VALIDATION_BEST_CHECKPOINT_MISSING")
            parameter_deltas = {"actor_l2": 0.0, "critic_l2": 0.0}
            for name, value in policy.named_parameters():
                delta = value.detach().cpu() - initial_policy_parameters[name]
                group = "critic_l2" if name.startswith(("critic_state.", "value_head.")) else "actor_l2"
                parameter_deltas[group] += float(delta.double().square().sum())
            parameter_deltas = {key: math.sqrt(value) for key, value in parameter_deltas.items()}
            result["variants"][variant][str(seed)] = {
                "best_checkpoint": str(best_path.resolve()), "last_checkpoint": str(last_path.resolve()),
                "best_validation_cost": best_cost, "validation_cost_curve": validation_curve,
                "training_curve": training_curve, "phase_seconds": timings,
                "training_episode_count": rounds * episodes_per_round,
                "training_transition_count": sum(row["transitions"] for row in training_curve),
                "actor_update_count": actor_updates, "critic_update_count": critic_updates,
                "actor_parameter_delta_l2": parameter_deltas["actor_l2"],
                "critic_parameter_delta_l2": parameter_deltas["critic_l2"],
                "world_model_frozen": all(not parameter.requires_grad for model in world_models for parameter in model.parameters()),
            }
    world_model_unchanged = all(
        set(before) == set(model.state_dict()) and all(torch.equal(before[key], value.detach().cpu())
                                                       for key, value in model.state_dict().items())
        for before, model in zip(world_model_state_before, world_models))
    if not world_model_unchanged:
        raise RuntimeError("V4_POLICY_WORLD_MODEL_STATE_CHANGED")
    for variant in VARIANTS:
        for seed in seeds:
            result["variants"][variant][str(seed)]["world_model_state_unchanged"] = world_model_unchanged
    result["status"] = "complete"
    (output / "V4_POLICY_TRAINING.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    return result


def evaluate_v4_methods(test_scenarios: Sequence[dict], policy_root: str | Path,
                        world_models: Sequence[Any], config: dict | None = None,
                        output_dir: str | Path | None = None) -> dict[str, Any]:
    """Evaluate each best checkpoint on the same frozen test episodes on CPU."""
    if not test_scenarios or len(world_models) != 3:
        raise ValueError("V4_TEST_SCENARIOS_OR_WORLD_MODELS_REQUIRED")
    root = Path(policy_root)
    training = json.loads((root / "V4_POLICY_TRAINING.json").read_text(encoding="utf-8"))
    if training.get("status") != "complete" or set(training.get("variants", {})) != set(VARIANTS):
        raise ValueError("V4_TRAINING_RECEIPT_REQUIRED")
    for model in world_models:
        model.eval().requires_grad_(False)
    base = dict(config or {})
    calibration = training.get("calibration", {})
    outputs: dict[str, Any] = {"schema": "n02-v4-frozen-test-evaluation/1", "status": "complete",
                               "split": "test", "scenario_count": len(test_scenarios), "variants": {}}
    for variant in VARIANTS:
        outputs["variants"][variant] = {}
        for seed in POLICY_SEEDS:
            checkpoint = training["variants"][variant][str(seed)]["best_checkpoint"]
            policy, _, payload = load_v4_checkpoint(checkpoint, device="cpu",
                                                     expected_source_identity=training.get("source_identity"))
            episode_config = {**base, "threshold_manifest": calibration[_threshold_key(variant)]}
            rows = []
            for scenario in test_scenarios:
                _, summary = evaluate_v4_episode(scenario, policy, variant=variant, seed=seed,
                                                 world_models=world_models, config=episode_config,
                                                 deterministic=True)
                rows.append(summary)
            costs = [float(row["validation_cost"]) for row in rows]
            outputs["variants"][variant][str(seed)] = {
                "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_source_identity": payload.get("source_identity"),
                "mean_total_cost": sum(costs) / len(costs),
                "mean_physical_cost": sum(float(row["physical_cost"]) for row in rows) / len(rows),
                "mean_online_cpu_cost": sum(float(row["online_cpu_cost"]) for row in rows) / len(rows),
                "episodes": rows,
            }
    if output_dir is not None:
        destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
        (destination / "V4_TEST_EVALUATION.json").write_text(
            json.dumps(outputs, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    return outputs
