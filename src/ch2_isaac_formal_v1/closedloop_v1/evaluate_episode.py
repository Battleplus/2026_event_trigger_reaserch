"""Independent full-episode reduction from closedloop_v1 raw artifacts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


VIOLATION_EVENTS = {"safety_violation", "boundary_violation", "constraint_violation"}
PHASE_NAMES = ("event_state_safety", "projection_policy_planning", "command_ready",
               "post_step_online_state")
SEPARATE_PHASE_NAMES = ("simulator_step", "post_step_evaluator_telemetry", "raw_artifact_logging")


def _jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unknown(reason: str, hashes: dict[str, str] | None = None) -> dict[str, Any]:
    return {"schema": "closedloop_episode_reduction_v1", "status": "UNKNOWN",
            "valid_mask": False, "reasons": [reason], "input_sha256": hashes or {}}


def _fail(reason: str, hashes: dict[str, str] | None = None) -> dict[str, Any]:
    return {"schema": "closedloop_episode_reduction_v1", "status": "FAIL",
            "valid_mask": False, "reasons": [reason], "input_sha256": hashes or {}}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _forced_safety_evidence(cfg: dict[str, Any], events: list[dict[str, Any]],
                            states: dict[tuple[int, str], dict[str, Any]],
                            duration: int, actor_by_id: dict[str, dict[str, Any]],
                            energy_hold_expectations: list[tuple[int, str]],
                            task_by_id: dict[str, dict[str, Any]] | None = None,
                            runtime_semantics_id: str | None = None) -> dict[str, Any]:
    """Cross-check forced-event records against persisted actor state and command effects."""
    mechanisms = cfg.get("mechanisms")
    schedule = cfg.get("evaluator_schedule", {}).get("disturbances", {})
    specs: list[tuple[str, str, int, str]] = []
    unknown_expectations = []

    def enabled(name: str) -> bool | None:
        item = mechanisms.get(name) if isinstance(mechanisms, dict) else None
        value = item.get("enabled") if isinstance(item, dict) else None
        return value if isinstance(value, bool) else None

    platform_item = mechanisms.get("platform_fault") if isinstance(mechanisms, dict) else None
    platform_enabled = (platform_item.get("enabled", True) if isinstance(platform_item, dict) else True)
    if platform_enabled:
        t_value = _finite_number(schedule.get("platform_fault_time_s", 900))
        if t_value is None:
            unknown_expectations.append("platform_fault_time")
            t_value = None
        t = int(t_value) if t_value is not None else 0
        if 1 <= t <= duration:
            specs.append(("platform_fault", "uav-0-0", t, "hold"))

    comm_item = mechanisms.get("communication_outage") if isinstance(mechanisms, dict) else None
    comm_enabled = (comm_item.get("enabled", True) if isinstance(comm_item, dict) else True)
    if comm_enabled:
        start_value = _finite_number(schedule.get("communication_outage_time_s", 1800))
        duration_value = _finite_number(schedule.get("communication_outage_duration_s", 120))
        if start_value is None or duration_value is None:
            unknown_expectations.append("communication_outage_timing")
            start_value = duration_value = None
        start = int(start_value) if start_value is not None else 0
        end = start + int(duration_value) if duration_value is not None else 0
        actor = "uav-1-0"
        if 1 <= start <= duration:
            specs.append(("communication_safety_hold", actor, start, "hold"))
        if 1 <= end <= duration:
            specs.append(("communication_safety_resume", actor, end, "resume"))

    capacities = cfg.get("energy_capacity_wh")
    reserve = _finite_number(cfg.get("energy_reserve_fraction"))
    if not isinstance(capacities, dict) or reserve is None:
        unknown_expectations.append("energy_safety_threshold")
    else:
        for actor_id, actor in actor_by_id.items():
            if _finite_number(capacities.get(str(actor.get("aircraft_type")))) is None:
                unknown_expectations.append(f"energy_capacity:{actor_id}")

    for t, actor in energy_hold_expectations:
        specs.append(("energy_safety_hold", actor, t, "hold"))

    observed_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for event in events:
        event_time = _finite_number(event.get("sim_time_s"))
        if event_time is None:
            if event.get("event") in {"platform_fault", "communication_safety_hold", "communication_safety_resume",
                                       "energy_safety_hold"}:
                unknown_expectations.append(f"event_time:{event.get('event_id', 'unknown_id')}")
            continue
        key = (str(event.get("event", "")), str(event.get("aircraft_id", "")), int(event_time))
        if key[0] in {"platform_fault", "communication_safety_hold", "communication_safety_resume",
                      "energy_safety_hold"}:
            observed_by_key.setdefault(key, []).append(event)

    evidence_rows = []
    fault_release_rows = []
    unrecovered_fault_task_ids: set[str] = set()
    applied_count = unknown_count = mismatch_count = 0
    spec_keys = {(kind, actor, time_s) for kind, actor, time_s, _ in specs}
    for kind, actor, time_s, action in specs:
        event_key = (kind, actor, time_s)
        matches = observed_by_key.get(event_key, [])
        expected_id = f"representative:{time_s}:{kind}:{actor}:"
        event_id = str(matches[0].get("event_id", expected_id)) if matches else expected_id
        reasons = []
        if len(matches) != 1:
            reasons.append("event_missing" if not matches else "duplicate_event")
        current = states.get((time_s, actor))
        if current is None:
            reasons.append("actor_state_missing")
        elif action == "hold":
            flag = {"platform_fault": "faulted", "communication_safety_hold": "communication_held",
                    "energy_safety_hold": "energy_paused"}[kind]
            if current.get(flag) is None:
                reasons.append("hold_state_field_unknown")
            elif current.get(flag) is not True:
                reasons.append("hold_state_not_observed")
            end = (int(schedule.get("communication_outage_time_s", 1800)) +
                   int(schedule.get("communication_outage_duration_s", 120))
                   if kind == "communication_safety_hold" else duration + 1)
            for check_t in range(time_s, min(duration, end - 1) + 1):
                state = states.get((check_t, actor))
                if state is None:
                    reasons.append(f"followup_state_missing@{check_t}")
                    break
                if state.get(flag) is None:
                    reasons.append(f"followup_state_unknown@{check_t}")
                    break
                if state.get(flag) is not True:
                    reasons.append(f"hold_state_released_early@{check_t}")
                    break
                if state.get("displacement_m", math.inf) > 0.02:
                    reasons.append(f"actor_moved_while_held@{check_t}")
                    break
        else:
            previous = states.get((time_s - 1, actor))
            if previous is None or previous.get("communication_held") is None:
                reasons.append("prior_hold_state_not_observed")
            elif previous.get("communication_held") is not True:
                reasons.append("prior_hold_state_mismatch")
            if current is None or current.get("communication_held") is None:
                reasons.append("resume_state_unknown")
            elif current.get("communication_held") is not False:
                reasons.append("resume_state_not_observed")

        # Detect a previously sent command taking effect while the forced hold is active.
        if action == "hold":
            release_time = (int(schedule.get("communication_outage_time_s", 1800)) +
                            int(schedule.get("communication_outage_duration_s", 120))
                            if kind == "communication_safety_hold" else duration + 1)
            pending_ids = set()
            command_time_unknown = False
            for sent in events:
                if (sent.get("event") != "transmission_sent"
                        or str(sent.get("aircraft_id", "")) != actor):
                    continue
                sent_time = _finite_number(sent.get("sim_time_s"))
                effect_time = _finite_number(sent.get("command_effect_sim_time_s"))
                if sent_time is None or effect_time is None:
                    command_time_unknown = True
                    continue
                if sent_time < time_s and effect_time >= time_s:
                    if sent.get("decision_id") is None:
                        command_time_unknown = True
                    else:
                        pending_ids.add(str(sent["decision_id"]))
            if command_time_unknown:
                reasons.append("pending_command_identity_unknown")
            else:
                for event in events:
                    applied_time = _finite_number(event.get("sim_time_s"))
                    if (event.get("event") == "command_applied"
                            and str(event.get("aircraft_id", "")) == actor
                            and (applied_time is None or time_s <= int(applied_time) < release_time)):
                        if applied_time is None or event.get("decision_id") is None:
                            reasons.append("applied_command_identity_or_time_unknown")
                            break
                        if str(event["decision_id"]) in pending_ids:
                            reasons.append("pre_hold_command_applied_during_hold")
                            break

        action_reasons = list(reasons)
        unfinished: list[str] = []
        expected_unfinished: list[str] = []
        if kind == "platform_fault":
            expected_lost: set[str] = set()
            release_unknown = False
            audit = (matches[0].get("details", {}).get("queue_release_state")
                     if len(matches) == 1 and isinstance(matches[0].get("details"), dict) else None)
            release_reasons = []
            if (not isinstance(audit, dict)
                    or audit.get("schema") != "fault_queue_release_state_v1"
                    or audit.get("source") != "executor_queues_cursor_and_mission_assignments_after_release"):
                release_reasons.append("executor_queue_release_evidence_missing")
            else:
                before = audit.get("pre_fault", {})
                after = audit.get("post_fault", {})
                old_queue = before.get("queue_task_ids") if isinstance(before, dict) else None
                old_cursor = before.get("cursor") if isinstance(before, dict) else None
                unfinished = before.get("unfinished_task_ids") if isinstance(before, dict) else None
                queues_after = after.get("queues_by_aircraft") if isinstance(after, dict) else None
                cursors_after = after.get("cursors_by_aircraft") if isinstance(after, dict) else None
                assignments_after = after.get("assignments_by_task") if isinstance(after, dict) else None
                completed_after = set(map(str, after.get("completed_task_ids", ()))) if isinstance(after, dict) else set()
                actor_state = after.get("actor_state_by_id") if isinstance(after, dict) else None
                if (isinstance(old_queue, list) and isinstance(old_cursor, list) and old_cursor
                        and isinstance(old_cursor[0], int) and 0 <= old_cursor[0] <= len(old_queue)):
                    expected_unfinished = list(map(str, old_queue[old_cursor[0]:]))
                else:
                    expected_unfinished = None
                if expected_unfinished is None or list(map(str, unfinished or [])) != expected_unfinished:
                    release_reasons.append("pre_fault_queue_cursor_disagreement")
                if list(map(str, matches[0].get("details", {}).get("unfinished_tasks_released", []))) != list(map(str, unfinished or [])):
                    release_reasons.append("released_task_event_disagreement")
                required_maps = (queues_after, cursors_after, assignments_after, actor_state)
                if any(not isinstance(value, dict) for value in required_maps):
                    release_reasons.append("post_fault_executor_state_missing")
                else:
                    fault_queue = queues_after.get(actor, [])
                    if not isinstance(fault_queue, list):
                        release_reasons.append("fault_actor_queue_unknown")
                        fault_queue = []
                    if set(map(str, fault_queue)).intersection(expected_unfinished or []):
                        release_reasons.append("released_task_retained_by_fault_actor")
                    for task_id in expected_unfinished or []:
                        owner_occurrences = [(owner, queued_task) for owner, queue in queues_after.items()
                                             if isinstance(queue, list) for queued_task in map(str, queue)
                                             if queued_task == task_id]
                        owners = [owner for owner, _ in owner_occurrences]
                        if task_id in completed_after:
                            if task_id in assignments_after:
                                release_reasons.append(f"completed_task_still_assigned:{task_id}")
                            continue
                        if len(owner_occurrences) > 1:
                            release_reasons.append(f"released_task_has_duplicate_owners:{task_id}")
                            continue
                        if not owners:
                            unrecovered_fault_task_ids.add(task_id)
                            if task_id in assignments_after:
                                release_reasons.append(f"unrecovered_task_has_stale_assignment:{task_id}")
                            continue
                        owner = owners[0]
                        if str(assignments_after.get(task_id, "")) != str(owner):
                            release_reasons.append(f"queue_assignment_disagreement:{task_id}")
                        owner_state = actor_state.get(owner, {})
                        if (not isinstance(owner_state, dict) or owner_state.get("available") is not True
                                or owner_state.get("faulted") is True
                                or owner_state.get("safety_paused") is True
                                or owner_state.get("communication_held") is True):
                            release_reasons.append(f"released_task_owner_not_available:{task_id}")
                        spec = (task_by_id or {}).get(task_id)
                        if not isinstance(spec, dict) or spec.get("allowed_kind") is None:
                            release_reasons.append(f"reassigned_task_legality_unknown:{task_id}")
                            release_unknown = True
                        elif isinstance(owner_state, dict):
                            if str(spec["allowed_kind"]) != str(owner_state.get("kind")):
                                release_reasons.append(f"released_task_owner_kind_mismatch:{task_id}")
                            if str(owner_state.get("kind")) == "helicopter":
                                task_zone = _finite_number(spec.get("zone"))
                                owner_zone = _finite_number(owner_state.get("home_zone"))
                                if task_zone is None or owner_zone is None or int(task_zone) != int(owner_zone):
                                    if task_zone is None or owner_zone is None:
                                        release_reasons.append(f"reassigned_helicopter_task_legality_unknown:{task_id}")
                                        release_unknown = True
                                    else:
                                        release_reasons.append(f"helicopter_released_task_outside_home_zone:{task_id}")
                        queue = queues_after.get(owner, [])
                        cursor = cursors_after.get(owner)
                        if (not isinstance(queue, list) or not isinstance(cursor, list) or not cursor
                                or not isinstance(cursor[0], int) or queue.index(task_id) < cursor[0]):
                            release_reasons.append(f"released_task_cursor_inconsistent:{task_id}")
                failure_events = [event for event in events
                                  if event.get("event") == "fault_task_unrecovered"
                                  and str(event.get("aircraft_id", "")) == actor
                                  and _finite_number(event.get("sim_time_s")) == time_s]
                expected_lost = {task_id for task_id in expected_unfinished or []
                                 if isinstance(queues_after, dict)
                                 and not any(isinstance(queue, list) and task_id in set(map(str, queue))
                                             for queue in queues_after.values())
                                 and task_id not in completed_after}
                observed_lost = [str(event.get("task_id")) for event in failure_events]
                if len(observed_lost) != len(set(observed_lost)) or set(observed_lost) != expected_lost:
                    release_reasons.append("unrecovered_task_failure_event_disagreement")
                if any(not isinstance(event.get("details"), dict)
                       or event["details"].get("failure_cost_unit") != "one_unrecovered_public_task"
                       for event in failure_events):
                    release_reasons.append("unrecovered_task_failure_cost_missing")
            release_status = "mismatch" if release_reasons else "applied"
            if (release_unknown or "executor_queue_release_evidence_missing" in release_reasons
                    or "post_fault_executor_state_missing" in release_reasons):
                release_status = "unknown"
            fault_release_rows.append({"event_id": event_id, "aircraft_id": actor,
                "sim_time_s": time_s, "status": release_status,
                "released_task_ids": (sorted(map(str, unfinished or []))
                                      if isinstance(audit, dict) else None),
                "unrecovered_task_ids": (sorted(expected_lost)
                                         if release_status != "unknown" else None),
                "reasons": release_reasons})
            reasons.extend(release_reasons)

        if action_reasons:
            status = "mismatch" if any(reason in {"event_missing", "duplicate_event", "hold_state_not_observed",
                "resume_state_not_observed", "prior_hold_state_mismatch"} or reason.startswith("hold_state_released_early") or reason.startswith("actor_moved")
                or reason.startswith("pre_hold_command") for reason in action_reasons) else "unknown"
        else:
            status = "applied"
        applied_count += status == "applied"
        unknown_count += status == "unknown"
        mismatch_count += status == "mismatch"
        evidence_rows.append({"event_id": event_id, "event": kind, "aircraft_id": actor,
                              "sim_time_s": time_s, "status": status, "reasons": action_reasons,
                              "state_evidence": "trajectory flags, position displacement, and command_applied events"})

    unexpected = []
    for key, matches in observed_by_key.items():
        if key in spec_keys:
            continue
        for event in matches:
            unexpected.append(str(event.get("event_id", ":".join(map(str, key)))))
            mismatch_count += 1
    unknown_count += len(unknown_expectations)
    pause_resume_actions_applied = (not unknown_expectations and not unexpected
                   and all(row["status"] == "applied" for row in evidence_rows))
    runtime_identity_valid = runtime_semantics_id == "closedloop_v1_fault_release_accounting_v2"
    all_statuses = [row["status"] for row in evidence_rows]
    release_statuses = [row["status"] for row in fault_release_rows]
    if not runtime_identity_valid or unknown_expectations or "unknown" in all_statuses or "unknown" in release_statuses:
        all_forced_applied = None
    else:
        all_forced_applied = (not unexpected and all(status == "applied" for status in all_statuses)
                              and all(status == "applied" for status in release_statuses))
    return {"all_forced_safety_actions_applied": all_forced_applied,
            "runtime_semantics_id": runtime_semantics_id,
            "pause_resume_actions_applied": pause_resume_actions_applied,
            "fault_task_release": {"status": ("not_applicable" if not specs or not any(
                spec[0] == "platform_fault" for spec in specs) else
                "unknown" if "unknown" in release_statuses else
                "mismatch" if "mismatch" in release_statuses else "applied"),
                "events": fault_release_rows,
                "released_task_ids": (sorted({task_id for row in fault_release_rows
                    for task_id in row.get("released_task_ids", []) or []})
                    if "unknown" not in release_statuses else None),
                "unrecovered_task_ids": (sorted(unrecovered_fault_task_ids)
                                         if "unknown" not in release_statuses else None),
                "unrecovered_task_failure_cost": (len(unrecovered_fault_task_ids)
                                                  if "unknown" not in release_statuses else None),
                "failure_cost_unit": "one_unrecovered_public_task; reported separately from frozen J_exec"},
            "forced_safety_action_counts": {"expected": len(specs), "applied": applied_count,
                "unknown": unknown_count, "mismatch": mismatch_count},
            "forced_safety_action_events": evidence_rows,
            "unknown_expected_event_families": unknown_expectations,
            "unexpected_event_ids": unexpected,
            "evidence_scope": "observable actor pause/resume, stationary follow-up, pre-hold command effects, and executor queue/cursor/assignment release audit",
            "runtime_default_sources": {"platform_fault_time_s": "runtime.py default 900 s when absent",
                "communication_outage_time_s": "runtime.py default 1800 s when absent",
                "communication_outage_duration_s": "runtime.py default 120 s when absent",
                "mechanism_enabled": "runtime.py mechanism_enabled defaults enabled=true when mechanism configuration is absent"}}


def reduce_episode(manifest: dict[str, Any], truth_doc: dict[str, Any],
                   trajectory_rows: Iterable[dict[str, Any]],
                   event_rows: Iterable[dict[str, Any]],
                   runtime_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reduce one representative episode without consulting runtime summary metrics."""
    cfg = manifest.get("evaluation_config")
    actors = manifest.get("actor_manifest")
    tasks = manifest.get("public_task_manifest")
    initial_positions = manifest.get("initial_positions_m")
    zone_bounds = manifest.get("zone_bounds_km")
    targets_doc = truth_doc.get("targets")
    if not all((isinstance(cfg, dict), isinstance(actors, list), isinstance(tasks, list),
                isinstance(initial_positions, dict), isinstance(targets_doc, list) and targets_doc)):
        return _unknown("manifest lacks frozen config, public task/actor geometry, initial positions, or evaluator truth")
    actor_by_id = {str(item["aircraft_id"]): item for item in actors}
    task_by_id = {str(item["task_id"]): item for item in tasks}
    truth = {str(item["target_id"]): item for item in targets_doc}
    if len(actor_by_id) != len(actors) or len(task_by_id) != len(tasks) or len(truth) != len(targets_doc):
        return _unknown("duplicate actor, task, or evaluator target IDs")
    if not actor_by_id or not task_by_id or set(initial_positions) != set(actor_by_id):
        return _unknown("actor/task catalog or initial-position coverage is incomplete")

    duration = int(cfg.get("representative_duration_s", 0))
    if duration <= 0:
        return _unknown("representative duration is absent")
    sensor_radius = float(cfg["sensor_radius_km"])
    tolerance = float(cfg["waypoint_tolerance_km"])
    rates = cfg.get("energy_wh_per_km")
    if not isinstance(rates, dict) or not rates:
        return _unknown("frozen analytical energy rates are absent")
    anomaly = cfg.get("energy_anomaly")
    targets = {target_id: (int(item["zone"]), tuple(map(float, item["target_xy_km"])))
               for target_id, item in truth.items()}
    events_list = list(event_rows)
    event_by_time: dict[int, list[dict[str, Any]]] = {}
    event_counts: dict[str, int] = {}
    for event in events_list:
        kind = str(event.get("event", ""))
        event_counts[kind] = event_counts.get(kind, 0) + 1
        time_s = event.get("sim_time_s")
        if isinstance(time_s, (int, float)) and math.isfinite(float(time_s)):
            event_by_time.setdefault(int(time_s), []).append(event)

    previous = {aid: list(map(float, initial_positions[aid])) for aid in actor_by_id}
    energy_wh = 0.0
    energy_by_actor = {aid: 0.0 for aid in actor_by_id}
    discovered: set[str] = set()
    unresolved_sum = 0.0
    task_activity: set[str] = set()
    completed_tasks: set[str] = set()
    waypoint_visits = 0
    actor_seconds = 0
    faulted_actor_seconds = 0
    communication_held_actor_seconds = 0
    energy_paused_actor_seconds = 0
    geometry_checks = 0
    energy_checks = 0
    safety_states: dict[tuple[int, str], dict[str, Any]] = {}
    energy_hold_expectations: list[tuple[int, str]] = []
    energy_hold_expected_actors: set[str] = set()
    safety_position_checks = 0
    aircraft_constraint_checks = 0
    previous_energy_pause: dict[str, bool] = {}
    row_count = 0
    timing_totals = {name: {"wall_duration_ns": 0, "cpu_duration_ns": 0}
                     for name in (*PHASE_NAMES, *SEPARATE_PHASE_NAMES)}
    timing_ticks = 0
    timing_invalid = False
    task_complete_events: dict[tuple[int, str, str], int] = {}
    task_completion_checks = 0
    waypoint_cursor: dict[tuple[str, str], int] = {}
    for event in events_list:
        if event.get("event") == "search_cell_complete":
            key = (int(event.get("sim_time_s", -1)), str(event.get("aircraft_id", "")),
                   str(event.get("task_id", "")))
            task_complete_events[key] = task_complete_events.get(key, 0) + 1

    expected_times = range(1, duration + 1)
    grouped = itertools.groupby(trajectory_rows, key=lambda row: int(row["sim_time_s"]))
    group_iter = iter(grouped)
    current_group = next(group_iter, None)
    finalizations: dict[str, dict[str, Any]] = {}
    created_decisions: dict[str, dict[str, Any]] = {}
    for t in expected_times:
        if current_group is None or current_group[0] != t:
            return _unknown(f"trajectory missing complete actor frame at t={t}")
        _, frame_iter = current_group
        frame = list(frame_iter)
        current_group = next(group_iter, None)
        if len(frame) != len(actor_by_id):
            return _unknown(f"trajectory actor coverage mismatch at t={t}")
        rows = {str(row.get("aircraft_id", "")): row for row in frame}
        if set(rows) != set(actor_by_id):
            return _fail(f"trajectory actor IDs mismatch at t={t}")
        events = event_by_time.get(t, [])
        by_actor_discoveries: set[str] = set()
        expected_discoveries: set[str] = set()
        for aid, row in rows.items():
            row_count += 1
            row_actor = actor_by_id[aid]
            capacities = cfg.get("energy_capacity_wh", {})
            reserve_fraction = _finite_number(cfg.get("energy_reserve_fraction"))
            capacity = (_finite_number(capacities.get(str(row_actor.get("aircraft_type"))))
                        if isinstance(capacities, dict) else None)
            # Runtime checks reserve before this tick's backend step. Energy spent
            # during the preceding tick can therefore trigger hold at this tick.
            if (aid not in energy_hold_expected_actors and capacity is not None
                    and reserve_fraction is not None
                    and capacity - energy_by_actor[aid] <= capacity * reserve_fraction):
                energy_hold_expectations.append((t, aid))
                energy_hold_expected_actors.add(aid)
            if row.get("branch") != "representative":
                return _fail("trajectory contains a non-representative branch")
            if row.get("aircraft_type") != row_actor.get("aircraft_type"):
                return _fail(f"trajectory aircraft type differs from manifest at {aid}@{t}")
            if int(row.get("zone", -1)) != int(row_actor.get("home_zone", -2)):
                return _fail(f"trajectory home zone differs from manifest at {aid}@{t}")
            point_m = list(map(float, row.get("usd_position_m", [])))
            prior_m = list(map(float, row.get("previous_usd_position_m", [])))
            xy_km = row.get("mission_xy_km")
            if len(point_m) != 3 or len(prior_m) != 3 or not isinstance(xy_km, list) or len(xy_km) < 2:
                return _unknown(f"trajectory position fields are incomplete at {aid}@{t}")
            if math.dist(prior_m, previous[aid]) > 0.02:
                return _fail(f"trajectory position continuity mismatch at {aid}@{t}")
            if math.dist([float(xy_km[0]), float(xy_km[1])],
                         [point_m[0] / 1000.0, point_m[1] / 1000.0]) > 2e-5:
                return _fail(f"mission XY disagrees with persisted position at {aid}@{t}")
            displacement_km = math.dist(prior_m, point_m) / 1000.0
            displacement_m = math.dist(prior_m, point_m)
            comm_row = row.get("communication") or {}
            energy_row = row.get("energy") or {}
            safety_states[(t, aid)] = {
                "position_m": point_m,
                "displacement_m": displacement_m,
                "faulted": row.get("faulted") if isinstance(row.get("faulted"), bool) else None,
                "communication_held": (comm_row.get("safety_hold")
                    if isinstance(comm_row.get("safety_hold"), bool) else None),
                "energy_paused": (energy_row.get("safety_pause")
                    if isinstance(energy_row.get("safety_pause"), bool) else None),
                "task_id": str(row.get("task_id")) if row.get("task_id") is not None else None,
            }
            if (bool(row.get("faulted", False)) or bool(comm_row.get("safety_hold", False))) and displacement_m > 0.02:
                return _fail(f"fault/communication-held actor moved at {aid}@{t}")
            if previous_energy_pause.get(aid, False) and displacement_m > 0.02:
                return _fail(f"energy-paused actor moved after hold at {aid}@{t}")
            safety_position_checks += 1
            if row_actor.get("aircraft_type") == "helicopter":
                bounds = zone_bounds.get(str(row_actor.get("home_zone"))) if isinstance(zone_bounds, dict) else None
                if not isinstance(bounds, list) or len(bounds) != 4:
                    aircraft_constraint_checks = -1
                else:
                    x, y = float(xy_km[0]), float(xy_km[1])
                    xmin, ymin, xmax, ymax = map(float, bounds)
                    if x < xmin - 1e-6 or x > xmax + 1e-6 or y < ymin - 1e-6 or y > ymax + 1e-6:
                        return _fail(f"helicopter {aid} left home-zone bounds at t={t}")
                    if aircraft_constraint_checks >= 0:
                        aircraft_constraint_checks += 1
            kind = str(row_actor["aircraft_type"])
            if kind not in rates:
                return _unknown(f"energy rate unavailable for actor type {kind}")
            rate = float(rates[kind])
            if anomaly and aid == anomaly.get("aircraft_id") and t >= float(anomaly["start_s"]):
                rate *= float(anomaly["rate_multiplier"])
            use_wh = displacement_km * rate
            energy_wh += use_wh
            energy_by_actor[aid] += use_wh
            energy_checks += 1
            reported_energy = row.get("energy", {}).get("incremental_wh")
            if reported_energy is not None and not math.isclose(float(reported_energy), use_wh,
                                                                 rel_tol=1e-8, abs_tol=1e-8):
                return _fail(f"reported energy differs from position-derived energy at {aid}@{t}")
            previous[aid] = point_m
            actor_seconds += 1
            faulted_actor_seconds += int(bool(row.get("faulted", False)))
            communication_held_actor_seconds += int(bool(row.get("communication", {}).get("safety_hold", False)))
            energy_paused_actor_seconds += int(bool(row.get("energy", {}).get("safety_pause", False)))
            previous_energy_pause[aid] = bool(row.get("energy", {}).get("safety_pause", False))
            task_id = row.get("task_id")
            if task_id is not None:
                task_id = str(task_id)
                if task_id not in task_by_id:
                    return _fail(f"trajectory references unknown public task {task_id}")
                task_activity.add(task_id)
                task_spec = task_by_id[task_id]
                if str(task_spec.get("allowed_kind")) != str(row_actor.get("aircraft_type")):
                    return _fail(f"actor type is not eligible for task {task_id} at {aid}@{t}")
                if (row_actor.get("aircraft_type") == "helicopter"
                        and int(task_spec.get("zone", -1)) != int(row_actor.get("home_zone", -2))):
                    return _fail(f"helicopter {aid} is assigned outside its home-zone task area at t={t}")
                if not bool(row.get("faulted", False)):
                    zone = int(task_spec["zone"])
                    point = (float(xy_km[0]), float(xy_km[1]))
                    expected_discoveries.update(target_id for target_id, (target_zone, target_xy) in targets.items()
                        if target_id not in discovered and target_zone == zone
                        and math.dist(point, target_xy) <= sensor_radius)
                    geometry_checks += 1
                if bool(row.get("search_cell_visited", False)):
                    waypoint_visits += 1
                waypoints = task_by_id[task_id].get("sweep_waypoints_km")
                if not isinstance(waypoints, list) or not waypoints:
                    return _unknown(f"public sweep geometry missing for task {task_id}")
                cursor_key = (aid, task_id)
                cursor = waypoint_cursor.get(cursor_key, 0)
                point = (float(xy_km[0]), float(xy_km[1]))
                waypoint_hit = (not bool(row.get("faulted", False)) and cursor < len(waypoints)
                                and math.dist(point, tuple(map(float, waypoints[cursor]))) <= tolerance)
                if waypoint_hit:
                    cursor += 1
                    waypoint_cursor[cursor_key] = cursor
                task_done = bool(row.get("task_completed", False))
                completion_event = task_complete_events.get((t, aid, task_id), 0)
                if task_done != (completion_event == 1):
                    return _fail(f"task completion row/event disagreement at {aid}@{t}")
                if bool(row.get("search_cell_visited", False)) != waypoint_hit:
                    return _fail(f"waypoint visit row disagrees with ordered sweep geometry at {aid}@{t}")
                replay_completed = cursor == len(waypoints)
                if task_done != replay_completed:
                    return _fail(f"task completion disagrees with ordered sweep replay at {aid}@{t}")
                if task_done:
                    task_completion_checks += 1
                    completed_tasks.add(task_id)
        row_discoveries = {str(target_id) for row in frame
                           for target_id in row.get("target_discovered_ids", [])}
        event_discoveries = {str(event.get("target_id")) for event in events
                             if event.get("event") == "target_discovered"}
        new_discoveries = expected_discoveries - discovered
        if row_discoveries != new_discoveries or event_discoveries != new_discoveries:
            return _fail(f"target discovery geometry differs from row/event evidence at t={t}")
        discovered.update(new_discoveries)
        unresolved_sum += (1.0 - len(discovered) / len(targets))
        tick_events = [event for event in events if event.get("event") == "online_timing_tick"]
        if len(tick_events) != 1:
            timing_invalid = True
        else:
            phases = tick_events[0].get("timing", {}).get("phases", {})
            try:
                for name in (*PHASE_NAMES, *SEPARATE_PHASE_NAMES):
                    item = phases[name]
                    wall = int(item["wall_duration_ns"])
                    cpu = int(item["cpu_duration_ns"])
                    if wall < 0 or cpu < 0:
                        raise ValueError("negative duration")
                    timing_totals[name]["wall_duration_ns"] += wall
                    timing_totals[name]["cpu_duration_ns"] += cpu
                timing_ticks += 1
            except (KeyError, TypeError, ValueError):
                timing_invalid = True
        for event in events:
            if event.get("event") == "decision_created":
                decision_id = event.get("decision_id")
                if decision_id:
                    created_decisions[str(decision_id)] = event
            elif event.get("event") == "decision_timing_finalized":
                decision_id = event.get("decision_id")
                if decision_id:
                    finalizations[str(decision_id)] = event
    if current_group is not None:
        return _unknown("trajectory contains rows beyond configured representative duration")

    forced_safety_evidence = _forced_safety_evidence(
        cfg, events_list, safety_states, duration, actor_by_id, energy_hold_expectations,
        task_by_id=task_by_id,
        runtime_semantics_id=((runtime_receipt or {}).get("closedloop", {}).get("runtime_semantics_id")
                              if isinstance((runtime_receipt or {}).get("closedloop"), dict) else None))

    if cfg.get("mode") == "search_strike":
        q_terminal = "unknown"
        unresolved_mean = "unknown"
        j_exec: float | str = "unknown"
        j_reason = "search_strike service dwell is not independently replayed by this evaluator"
    elif cfg.get("mode") == "search":
        q_terminal = 1.0 - len(discovered) / len(targets)
        unresolved_mean = unresolved_sum / duration
        violation_count = sum(event_counts.get(name, 0) for name in VIOLATION_EVENTS)
        j_exec = q_terminal + 0.2 * unresolved_mean + 0.01 * (energy_wh / 1_000_000.0) + 0.25 * violation_count
        j_reason = None
    else:
        q_terminal = unresolved_mean = j_exec = "unknown"
        j_reason = "unsupported mission mode"

    if timing_invalid or timing_ticks != duration:
        online_timing: dict[str, Any] | str = "unknown"
        timing_reason = "per-tick timing event missing or malformed"
    else:
        online_total = {
            measure: sum(timing_totals[name][f"{measure}_duration_ns"] for name in PHASE_NAMES)
            for measure in ("wall", "cpu")
        }
        online_timing = {"phase_totals": timing_totals, "online_algorithm_total": online_total,
                         "post_step_evaluator_and_raw_logging_are_separate": True,
                         "simulator_step_excluded_from_online_total": True,
                         "phase_count": timing_ticks}
        timing_reason = None
    created_decisions = {}
    finalizations = {}
    duplicate_created_ids = set()
    duplicate_finalized_ids = set()
    for event in events_list:
        decision_id = event.get("decision_id")
        if not decision_id:
            continue
        decision_id = str(decision_id)
        if event.get("event") == "decision_created":
            if decision_id in created_decisions:
                duplicate_created_ids.add(decision_id)
            created_decisions[decision_id] = event
        elif event.get("event") == "decision_timing_finalized":
            if decision_id in finalizations:
                duplicate_finalized_ids.add(decision_id)
            finalizations[decision_id] = event
    effect_delays = []
    missing_finalization_ids = sorted(set(created_decisions) - set(finalizations))
    orphan_finalization_ids = sorted(set(finalizations) - set(created_decisions))
    effect_delay_invalid_ids = []
    for decision_id in sorted(set(created_decisions) & set(finalizations)):
        event = created_decisions[decision_id]
        finalized = finalizations[decision_id]
        created = event.get("decision_created_sim_time_s")
        complete = finalized.get("decision_complete_sim_time_s")
        if isinstance(created, (int, float)) and isinstance(complete, (int, float)):
            delay = float(complete) - float(created)
            if math.isfinite(delay) and delay >= 0:
                effect_delays.append({"decision_id": decision_id, "effect_delay_s": delay,
                                      "source": "decision_timing_finalized_overrides_decision_created"})
            else:
                effect_delay_invalid_ids.append(decision_id)
        else:
            effect_delay_invalid_ids.append(decision_id)
    effect_delay_coverage = {
        "status": "complete" if not missing_finalization_ids and not orphan_finalization_ids
                  and not effect_delay_invalid_ids and not duplicate_created_ids
                  and not duplicate_finalized_ids else "incomplete",
        "created_count": len(created_decisions),
        "finalized_count": len(set(created_decisions) & set(finalizations)),
        "finalization_event_count": len(finalizations),
        "coverage_fraction": (len(set(created_decisions) & set(finalizations)) / len(created_decisions)
                              if created_decisions else None),
        "missing_finalization_decision_ids": missing_finalization_ids,
        "orphan_finalization_decision_ids": orphan_finalization_ids,
        "invalid_delay_decision_ids": sorted(effect_delay_invalid_ids),
        "duplicate_created_decision_ids": sorted(duplicate_created_ids),
        "duplicate_finalized_decision_ids": sorted(duplicate_finalized_ids),
        "effect_delays": effect_delays if not (missing_finalization_ids or orphan_finalization_ids
                                                 or effect_delay_invalid_ids or duplicate_created_ids
                                                 or duplicate_finalized_ids) else "unknown",
    }
    switches = event_counts.get("controller_switch", 0)
    messages_sent = event_counts.get("transmission_sent", 0)
    messages_dropped = event_counts.get("command_dropped", 0)
    planner_calls = event_counts.get("candidate_generation", 0)
    planner_cpu_ns = 0
    planner_timing_valid = True
    for event in events_list:
        if event.get("event") != "candidate_generation":
            continue
        timing = event.get("timing", {})
        cpu_ns = timing.get("cpu_duration_ns") if isinstance(timing, dict) else None
        try:
            cpu_value = int(cpu_ns)
            if cpu_value < 0:
                raise ValueError("negative duration")
            planner_cpu_ns += cpu_value
        except (TypeError, ValueError):
            planner_timing_valid = False
    planning_cpu_seconds: float | str = planner_cpu_ns / 1e9 if planner_timing_valid else "unknown"
    changed_actors = {str(actor_id) for event in events_list if event.get("event") == "donor_support_transfer"
                      for actor_id in (event.get("from_aircraft_id"), event.get("to_aircraft_id")) if actor_id}
    control_cost = "unknown"
    total_with_control = "unknown"
    absolute_cost_terms = {
        "messages": {"raw_count": messages_sent + messages_dropped, "unit": "messages_per_1000",
                     "weight": 0.01, "weighted_value": 0.01 * (messages_sent + messages_dropped) / 1000.0},
        "switches": {"raw_count": switches, "unit": "switches_per_100",
                     "weight": 0.05, "weighted_value": 0.05 * switches / 100.0},
        "planner_calls": {"raw_count": planner_calls, "unit": "planner_calls_per_100",
                          "weight": 0.05, "weighted_value": 0.05 * planner_calls / 100.0},
        "changed_actors": {"raw_count": len(changed_actors), "unit": "changed_actors_per_36",
                           "weight": 0.01, "weighted_value": 0.01 * len(changed_actors) / 36.0},
    }
    non_cpu_components_complete = all(isinstance(item["weighted_value"], (int, float))
                                      for item in absolute_cost_terms.values())
    absolute_control_non_cpu = (sum(item["weighted_value"] for item in absolute_cost_terms.values())
                                if non_cpu_components_complete else "unknown")
    planner_cpu_weighted = (0.01 * planning_cpu_seconds
                            if isinstance(planning_cpu_seconds, (int, float)) else "unknown")
    online_cpu_seconds = "unknown"
    if isinstance(online_timing, dict):
        online_cpu_ns = online_timing.get("online_algorithm_total", {}).get("cpu")
        if isinstance(online_cpu_ns, (int, float)) and math.isfinite(float(online_cpu_ns)) and online_cpu_ns >= 0:
            online_cpu_seconds = float(online_cpu_ns) / 1e9
    online_cpu_weighted = 0.01 * online_cpu_seconds if isinstance(online_cpu_seconds, (int, float)) else "unknown"
    planner_only_subtotal = (j_exec + absolute_control_non_cpu + planner_cpu_weighted
                             if j_exec != "unknown" and non_cpu_components_complete
                             and isinstance(planner_cpu_weighted, (int, float)) else "unknown")
    online_cpu_subtotal = (j_exec + absolute_control_non_cpu + online_cpu_weighted
                           if j_exec != "unknown" and non_cpu_components_complete
                           and isinstance(online_cpu_weighted, (int, float)) else "unknown")
    runtime_closedloop = (runtime_receipt or {}).get("closedloop")
    runtime_closedloop = runtime_closedloop if isinstance(runtime_closedloop, dict) else {}
    device_evidence_present = "predictor_device" in runtime_closedloop
    device = runtime_closedloop.get("predictor_device")
    if device_evidence_present and device in (None, "cpu"):
        gpu_status = "not_applicable"
        gpu_transfer_status = "not_applicable"
    elif device_evidence_present and (device == "cuda" or str(device).startswith("cuda:")):
        gpu_status = "unknown"
        gpu_transfer_status = "unknown"
    else:
        gpu_status = gpu_transfer_status = "unknown"
    runtime_timing = runtime_closedloop.get("timing") if isinstance(runtime_closedloop.get("timing"), dict) else {}
    startup_load = runtime_timing.get("startup_load")
    startup_load_valid = (isinstance(startup_load, dict)
                          and all(isinstance(startup_load.get(key), (int, float))
                                  and math.isfinite(float(startup_load[key])) and startup_load[key] >= 0
                                  for key in ("wall_duration_ns", "cpu_duration_ns")))
    backend_close = runtime_timing.get("backend_close")
    backend_close_valid = (isinstance(backend_close, dict)
                           and all(isinstance(backend_close.get(key), (int, float))
                                   and math.isfinite(float(backend_close[key])) and backend_close[key] >= 0
                                   for key in ("wall_duration_ns", "cpu_duration_ns")))
    receipt_timing = runtime_timing.get("online_by_branch", {}).get("representative", {})
    if not isinstance(receipt_timing, dict):
        receipt_timing = {}
    receipt_measured = {
        "timing_record_write": receipt_timing.get("timing_event_write_cpu_ns"),
        "timing_record_write_wall": receipt_timing.get("timing_event_write_wall_ns"),
        "artifact_flush": receipt_timing.get("artifact_flush_cpu_ns"),
        "artifact_flush_wall": receipt_timing.get("artifact_flush_wall_ns"),
    }
    receipt_measured_valid = all(isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
                                 for value in receipt_measured.values())
    phase_cpu_seconds = {
        name: timing_totals[name]["cpu_duration_ns"] / 1e9
        for name in (*PHASE_NAMES, *SEPARATE_PHASE_NAMES)
    } if timing_ticks == duration else {}
    if timing_ticks != duration:
        phase_cpu_seconds["online_tick_phases"] = "unknown"
    measured_run_cpu_components = {
        "startup_load": (float(startup_load["cpu_duration_ns"]) / 1e9 if startup_load_valid else "unknown"),
        **phase_cpu_seconds,
        "backend_close": (float(backend_close["cpu_duration_ns"]) / 1e9 if backend_close_valid else "unknown"),
        "timing_record_write": (float(receipt_measured["timing_record_write"]) / 1e9
                                if receipt_measured_valid else "unknown"),
        "artifact_flush": (float(receipt_measured["artifact_flush"]) / 1e9
                           if receipt_measured_valid else "unknown"),
    }
    measured_resource_cpu_complete = all(isinstance(value, (int, float))
                                         for value in measured_run_cpu_components.values())
    measured_resource_cpu_seconds = (sum(measured_run_cpu_components.values())
                                     if measured_resource_cpu_complete else "unknown")
    measured_run_wall_components = {
        "startup_load": (float(startup_load["wall_duration_ns"]) / 1e9 if startup_load_valid else "unknown"),
        **({name: timing_totals[name]["wall_duration_ns"] / 1e9
            for name in (*PHASE_NAMES, *SEPARATE_PHASE_NAMES)} if timing_ticks == duration else {}),
        "backend_close": (float(backend_close["wall_duration_ns"]) / 1e9 if backend_close_valid else "unknown"),
        "timing_record_write": (float(receipt_measured["timing_record_write_wall"]) / 1e9
                                if receipt_measured_valid else "unknown"),
        "artifact_flush": (float(receipt_measured["artifact_flush_wall"]) / 1e9
                           if receipt_measured_valid else "unknown"),
    }
    if timing_ticks != duration:
        measured_run_wall_components["online_tick_phases"] = "unknown"
    online_method_cost_status = ("measured" if isinstance(online_cpu_subtotal, (int, float))
                                 else "incomplete")
    deployment_unknowns = []
    if gpu_status == "unknown":
        deployment_unknowns.extend(["gpu_inference_cost", "host_device_transfer_cost"])
    if not device_evidence_present:
        deployment_unknowns.append("predictor_device_identity")
    offline_unmeasured = ["python_process_launch_and_import_before_run_entry",
                          "post_episode_metric_manifest_and_final_receipt_persistence",
                          "compressed_stream_close_overhead"]
    reasons = []
    if j_reason:
        reasons.append(j_reason)
    if timing_reason:
        reasons.append(timing_reason)
    if effect_delay_coverage["status"] != "complete":
        reasons.append("decision effect-delay finalization coverage is incomplete; local objective terms remain available")
    if not forced_safety_evidence["pause_resume_actions_applied"]:
        reasons.append("pause/resume action evidence is incomplete, unknown, or mismatched")
    if forced_safety_evidence["all_forced_safety_actions_applied"] is not True:
        reasons.append("full forced-safety evidence is incomplete, mismatched, or from an incompatible runtime identity")
    if aircraft_constraint_checks < 0:
        reasons.append("helicopter home-zone bounds are absent; aircraft-zone constraint check is incomplete")
    reduction_valid = (j_exec != "unknown" and online_timing != "unknown"
                       and effect_delay_coverage["status"] == "complete"
                       and safety_position_checks == row_count and aircraft_constraint_checks >= 0
                       and forced_safety_evidence["pause_resume_actions_applied"] is True
                       and forced_safety_evidence["all_forced_safety_actions_applied"] is True)
    resource_usage = (runtime_receipt or {}).get("resource_usage", {})
    process_observation = ({
        "wall_duration_s": resource_usage.get("wall_duration_s", "unknown"),
        "process_cpu_s": resource_usage.get("process_cpu_s", "unknown"),
        "source": "run_receipt; runner start through final receipt persistence, including simulator work",
        "independently_recomputed": False,
    } if isinstance(resource_usage, dict) and resource_usage else "unknown")
    return {
        "schema": "closedloop_episode_reduction_v1", "status": "PASS" if reduction_valid else "UNKNOWN",
        "valid_mask": reduction_valid, "reasons": reasons,
        "input_sha256": manifest.get("_input_sha256", {}),
        "scope": {"run_id": manifest.get("run_id"), "branch": "representative", "duration_s": duration,
                  "trajectory_rows": row_count, "actor_seconds": actor_seconds},
        "tasks": {"catalog_count": len(task_by_id), "active_task_ids": sorted(task_activity),
                  "completed_task_ids": sorted(completed_tasks), "completed_count": len(completed_tasks),
                  "failed_due_to_forced_fault": forced_safety_evidence["fault_task_release"]["unrecovered_task_ids"],
                  "failed_due_to_forced_fault_count": forced_safety_evidence["fault_task_release"]["unrecovered_task_failure_cost"],
                  "waypoint_visits": waypoint_visits},
        "discovery": {"target_count": len(targets), "discovered_target_ids": sorted(discovered),
                      "discovered_count": len(discovered), "independent_geometry_checks": geometry_checks},
        "energy": {"status": "position_delta_recomputed_from_config_rates" if energy_checks == row_count else "unknown",
                   "energy_wh": energy_wh if energy_checks == row_count else "unknown",
                   "energy_wh_by_actor": energy_by_actor if energy_checks == row_count else "unknown",
                   "position_rate_checks": energy_checks},
        "safety": {"controllable_violation_events": sum(event_counts.get(name, 0) for name in VIOLATION_EVENTS),
                   **forced_safety_evidence,
                   "violation_event_counts": {name: event_counts.get(name, 0) for name in sorted(VIOLATION_EVENTS)},
                   "safety_event_counts": {name: event_counts.get(name, 0) for name in
                       ("platform_fault", "energy_safety_hold", "communication_safety_hold",
                        "energy_safety_resume", "communication_safety_resume")},
                   "faulted_actor_seconds": faulted_actor_seconds,
                   "communication_held_actor_seconds": communication_held_actor_seconds,
                   "energy_paused_actor_seconds": energy_paused_actor_seconds,
                   "stationary_hold_position_checks": safety_position_checks,
                   "helicopter_home_zone_position_checks": aircraft_constraint_checks if aircraft_constraint_checks >= 0 else "unknown",
                   "hold_motion_violation_check": safety_position_checks == row_count,
                   "overlapping_actor_seconds_are_not_combined": True},
        "communication": {"transmissions_sent": messages_sent, "commands_dropped": messages_dropped,
                          "commands_delivered": event_counts.get("command_delivered", 0),
                          "command_applied": event_counts.get("command_applied", 0),
                          "message_attempts": messages_sent + messages_dropped},
        "switching": {"controller_switch_events": switches,
                      "transferred_tasks": event_counts.get("donor_support_transfer", 0),
                      "unique_transfer_actors": sorted(changed_actors)},
        "planning": {"candidate_generation_calls": planner_calls,
                      "planning_cpu_seconds": planning_cpu_seconds,
                      "cpu_timing_is_nested_in_projection_policy_planning_phase": True},
        "decision_effect_delays": effect_delay_coverage,
        "timing": {"online_phases": online_timing,
                   "simulator_step_separate": timing_totals.get("simulator_step") if timing_ticks == duration else "unknown",
                   "post_step_evaluator_telemetry_separate": timing_totals.get("post_step_evaluator_telemetry") if timing_ticks == duration else "unknown",
                   "raw_artifact_logging_separate": timing_totals.get("raw_artifact_logging") if timing_ticks == duration else "unknown",
                   "startup_load_measured": startup_load if startup_load_valid else "unknown",
                   "backend_close_measured": backend_close if backend_close_valid else "unknown",
                   "timing_record_and_flush_write_costs": receipt_timing if receipt_measured_valid else "unknown",
                   "episode_process_observation": process_observation,
                   "episode_process_observation_scope": "runner resource observation begins at run() entry and is measured after backend close but before the final receipt self-write",
                   "effect_time_quantum_s": 1,
                   "millisecond_responsiveness_claim": False,
                   "unmeasured_execution_boundaries": ["python process launch/import before run() entry",
                       "final receipt self-persistence", "compressed trajectory stream close"]},
        "checks": {"full_actor_frame_coverage": row_count == duration * len(actor_by_id),
                   "target_discovery_geometry": geometry_checks > 0,
                   "position_derived_energy": energy_checks == row_count,
                   "ordered_sweep_completion_geometry": task_completion_checks == len(completed_tasks),
                   "stationary_hold_positions": safety_position_checks == row_count,
                   "helicopter_home_zone_bounds": aircraft_constraint_checks >= 0,
                   "runtime_summary_used_as_reference": False},
        "registered_objective": {
            "weights_source": "formal_evidence_v1/evaluate_suffix.py frozen J_exec expression",
            "J_exec_formula": "Q_terminal + 0.2*mean_unresolved_L + 0.01*(energy_wh/1_000_000) + 0.25*controllable_violation_events",
            "weights": {"Q_terminal": 1.0, "mean_unresolved_L": 0.2,
                        "energy_wh_div_1e6": 0.01, "controllable_violation": 0.25},
            "Q_terminal": q_terminal, "mean_unresolved_L": unresolved_mean,
                   "J_exec": j_exec, "J_exec_reason": j_reason,
            "absolute_online_episode_cost": {
                "status": online_method_cost_status,
                "registered_planner_only_subtotal": {
                    "value": planner_only_subtotal,
                    "definition": "J_exec plus frozen absolute message/switch/planner-call/changed-actor terms and the registered 0.01 per planner CPU second term",
                    "cpu_component": {"raw_seconds": planning_cpu_seconds, "weighted_value": planner_cpu_weighted,
                                      "scope": "candidate_generation CPU only; nested within projection_policy_planning"},
                },
                "online_cpu_costed_subtotal": {
                    "value": online_cpu_subtotal,
                    "definition": "J_exec plus the same frozen absolute non-CPU terms and 0.01 per mutually exclusive online algorithm CPU second; online CPU replaces planner CPU and is not added to it",
                    "cpu_component": {"raw_seconds": online_cpu_seconds, "weighted_value": online_cpu_weighted,
                                      "unit_cost": 0.01,
                                      "unit_cost_source": "user-provided full-plan instruction; not a frozen J_exec weight",
                                      "phase_source": "event_state_safety + projection_policy_planning + command_ready + post_step_online_state",
                                      "planner_cpu_is_nested_and_not_added_again": True},
                },
                "non_cpu_absolute_cost_components": absolute_cost_terms,
                "measured_online_method_cost": {"value": online_cpu_subtotal,
                    "cpu_seconds": online_cpu_seconds, "cpu_unit_cost": 0.01,
                    "cpu_unit_cost_source": "user-provided full-plan instruction; separate from frozen J_exec yield label",
                    "includes_phases": list(PHASE_NAMES),
                    "excludes_simulator_evaluator_and_evidence_logging": True,
                    "status": online_method_cost_status},
                "startup_load_separate": {"wall_seconds": measured_run_wall_components["startup_load"],
                    "cpu_seconds": measured_run_cpu_components["startup_load"],
                    "included_in_online_method_cost": False,
                    "status": "measured" if startup_load_valid else "unknown"},
                "deployment_device_costs": {"gpu_inference": gpu_status,
                    "host_device_transfer": gpu_transfer_status,
                    "unknown_components": deployment_unknowns},
                "resource_accounting_only": {
                    "scope": "includes simulator and offline evaluator/evidence logging; not charged to method cost",
                    "cpu_seconds_by_component": measured_run_cpu_components,
                    "cpu_seconds_observed_sum": measured_resource_cpu_seconds,
                    "wall_seconds_by_component": measured_run_wall_components,
                },
                "unmeasured_offline_boundaries": offline_unmeasured,
                "device_costs": {"predictor_device": (device if device is not None else "no_predictor")
                                 if device_evidence_present else "unknown",
                                 "gpu_inference": gpu_status, "host_device_transfer": gpu_transfer_status},
                "component_coverage": {
                    "feature_extraction": ("not_applicable" if device_evidence_present and device is None else
                                            "included_in_projection_policy_planning_not_separately_timed"
                                            if device_evidence_present else "unknown"),
                    "predictor_inference": ("not_applicable" if device_evidence_present and device is None else
                                             "included_in_projection_policy_planning_not_separately_timed"
                                             if device_evidence_present else "unknown"),
                    "target_detection": "included_in_post_step_online_state",
                    "simulated_command_transmission": "message_count_is_weighted; physical_network_cost_not_applicable_to_simulated_queue",
                },
                "scope_note": "PASS status validates episode physical evidence only; it does not certify method cost acceptance or end-to-end cost completeness",
            },
            "paired_control_increment_C": control_cost,
            "J_with_paired_control_increment": total_with_control,
            "paired_control_increment_weights": {"messages_per_1000": 0.01,
                "switches_per_100": 0.05, "planner_calls_per_100": 0.05,
                "changed_actors_per_36": 0.01, "planning_cpu_seconds": 0.01}},
    }


def evaluate_run(run_dir: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        result = _unknown("manifest.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logs = manifest.get("logs", {})
        trajectory_path = root / str(logs.get("trajectory", "trajectory.jsonl.gz"))
        event_path = root / str(logs.get("events", "events.jsonl"))
        truth_name = manifest.get("evaluation_truth_file") or manifest.get("evaluator_truth", {}).get("file")
        truth_path = root / str(truth_name or "evaluator_truth.json")
        receipt_path = root / "run_receipt.json"
        required = (trajectory_path, event_path, truth_path)
        missing = [path.name for path in required if not path.is_file()]
        hashes = {path.name: _sha256(path) for path in (manifest_path, *required, receipt_path) if path.is_file()}
        if missing:
            result = _unknown(f"required raw artifacts missing: {', '.join(missing)}", hashes)
        else:
            expected_truth_hash = manifest.get("evaluator_truth", {}).get("sha256")
            if expected_truth_hash and expected_truth_hash != hashes[truth_path.name]:
                result = _unknown("evaluator truth hash differs from manifest", hashes)
            else:
                manifest["_input_sha256"] = hashes
                truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
                runtime_receipt = (json.loads(receipt_path.read_text(encoding="utf-8"))
                                   if receipt_path.is_file() else None)
                try:
                    result = reduce_episode(manifest, truth_doc, _jsonl(trajectory_path),
                                            list(_jsonl(event_path)), runtime_receipt=runtime_receipt)
                except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
                        OverflowError, json.JSONDecodeError) as exc:
                    result = _unknown(f"reducer_error:{type(exc).__name__}:{exc}", hashes)
                result["input_sha256"] = hashes
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_compact_collection(collection_dir: str | Path) -> dict[str, Any]:
    """Rebuild reducer inputs from public compact ticks and separate evaluator files."""
    root = Path(collection_dir)
    required = [root / name for name in ("evaluation_context.json", "evaluation_truth.jsonl",
                                         "evaluation_audit.jsonl", "ticks.jsonl")]
    missing = [path.name for path in required if not path.is_file()]
    hashes = {path.name: _sha256(path) for path in required if path.is_file()}
    if missing:
        return _unknown(f"compact evaluation artifacts missing: {', '.join(missing)}", hashes)
    try:
        try:
            from compact_collection_v1 import CompactPublicReader
        except ModuleNotFoundError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from compact_collection_v1 import CompactPublicReader

        context = json.loads(required[0].read_text(encoding="utf-8"))
        truth_records = list(_jsonl(required[1]))
        audit_records = list(_jsonl(required[2]))
        ticks = list(CompactPublicReader(root).iter_records())
        if (not ticks or len(audit_records) != len(ticks)
                or any(int(tick["projection"]["sim_time_s"]) != int(audit["sim_time_s"])
                       for tick, audit in zip(ticks, audit_records))):
            return _unknown("compact public ticks and evaluator audit are incomplete or misaligned", hashes)
        if not truth_records or not isinstance(truth_records[0].get("truth"), dict):
            return _unknown("separate compact evaluator truth is missing", hashes)
        tasks_by_id: dict[str, dict[str, Any]] = {}
        trajectory_rows = []
        event_rows = []
        for tick, audit in zip(ticks, audit_records):
            projection = tick["projection"]
            now = int(projection["sim_time_s"])
            targets_by_actor = audit.get("target_discovered_ids_by_aircraft")
            if not isinstance(targets_by_actor, dict):
                return _unknown(f"compact evaluator discovery labels missing at t={now}", hashes)
            for task_id, spec in projection.get("public_tasks", {}).items():
                task = tasks_by_id.setdefault(str(task_id), {
                    key: spec[key] for key in ("zone", "cell", "allowed_kind",
                        "public_bounds_km", "sweep_waypoints_km") if key in spec})
                task["task_id"] = str(task_id)
                if not task.get("sweep_waypoints_km"):
                    return _unknown(f"compact public sweep geometry missing for task {task_id}", hashes)
            for aircraft_id, aircraft in projection.get("aircraft", {}).items():
                transition = aircraft.get("transition")
                transition_cost = aircraft.get("transition_cost")
                if not isinstance(transition, dict) or not isinstance(transition_cost, dict):
                    return _unknown(f"compact transition evidence missing at {aircraft_id}@{now}", hashes)
                if str(aircraft_id) not in targets_by_actor:
                    return _unknown(f"compact discovery labels missing for {aircraft_id}@{now}", hashes)
                trajectory_rows.append({
                    "sim_time_s": now, "branch": "representative", "aircraft_id": str(aircraft_id),
                    "aircraft_type": aircraft.get("kind"), "zone": aircraft.get("home_zone"),
                    "usd_position_m": aircraft.get("position_m"),
                    "previous_usd_position_m": transition.get("previous_position_m"),
                    "mission_xy_km": transition.get("mission_xy_km"),
                    "faulted": aircraft.get("faulted"), "task_id": transition.get("task_id"),
                    "search_cell_visited": transition.get("search_cell_visited"),
                    "task_completed": transition.get("task_completed"),
                    "target_discovered_ids": targets_by_actor[str(aircraft_id)],
                    "energy": {"incremental_wh": transition_cost.get("energy_incremental_wh"),
                               "safety_pause": transition_cost.get("energy_safety_pause")},
                    "communication": {"safety_hold": aircraft.get("communication_held"),
                        "command_attempted": transition_cost.get("command_attempted"),
                        "command_attempt_accepted": transition_cost.get("command_attempt_accepted"),
                        "command_delivered": transition_cost.get("command_delivered"),
                        "command_dropped": transition_cost.get("command_dropped"),
                        "forced_outage": transition_cost.get("forced_outage")},
                })
            events = audit.get("events")
            if not isinstance(events, list):
                return _unknown(f"compact evaluator events missing at t={now}", hashes)
            event_rows.extend(events)
        context_required = ("evaluation_config", "actor_manifest", "initial_positions_m", "zone_bounds_km")
        if not all(key in context for key in context_required):
            return _unknown("compact evaluation context is incomplete", hashes)
        manifest = {"run_id": context.get("run_id"),
                    "evaluation_config": context["evaluation_config"],
                    "actor_manifest": context["actor_manifest"],
                    "initial_positions_m": context["initial_positions_m"],
                    "zone_bounds_km": context["zone_bounds_km"],
                    "public_task_manifest": list(tasks_by_id.values()),
                    "_input_sha256": hashes}
        runtime_receipt = context.get("runtime_receipt")
        if not isinstance(runtime_receipt, dict):
            runtime_receipt = {"closedloop": {"runtime_semantics_id": context.get("runtime_semantics_id")}}
        return reduce_episode(manifest, truth_records[0]["truth"], trajectory_rows, event_rows,
                             runtime_receipt=runtime_receipt)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            OverflowError, json.JSONDecodeError) as exc:
        return _unknown(f"compact_reducer_error:{type(exc).__name__}:{exc}", hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently reduce one closedloop_v1 episode.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--compact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (evaluate_run(args.run_dir, args.output) if args.run_dir is not None
              else evaluate_compact_collection(args.compact_dir))
    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "reasons": result.get("reasons", [])}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
