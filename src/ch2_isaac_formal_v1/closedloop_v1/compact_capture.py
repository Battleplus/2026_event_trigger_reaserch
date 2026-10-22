"""Public compact capture adapters for the closed-loop runtime."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


PUBLIC_EVENTS = {
    "task_activated", "public_observation", "search_cell_complete", "controller_switch",
    "platform_fault", "communication_safety_hold", "energy_safety_hold", "energy_anomaly",
    "fault_task_unrecovered", "communication_safety_resume", "command_dropped",
    "decision_gate", "candidate_generation", "decision_created", "decision_timing_finalized",
    "command_delivered", "command_applied", "transmission_sent", "donor_support_transfer",
    "replan_action_rejected", "replan_catalog_choice", "abstract_strike_complete",
}
DECISION_EVENTS = {"decision_gate", "candidate_generation", "decision_created",
                   "decision_timing_finalized", "replan_action_rejected", "replan_catalog_choice"}


def compact_tick(public_rows: Sequence[Mapping[str, Any]], tasks: Mapping[str, Any],
                 queues: Mapping[str, Sequence[str]], cursor: Mapping[str, Sequence[int]],
                 activation: Mapping[str, float], routes: Mapping[str, Mapping[str, Any]],
                 completed: set[str], sim_time_s: int, event_rows: Sequence[Mapping[str, Any]],
                 actor_status: Mapping[str, Mapping[str, bool]], assignments: Mapping[str, str]):
    """Build a public tick without consulting evaluator truth or future schedules."""
    now = int(sim_time_s)
    aircraft = {}
    for row in public_rows:
        aid = str(row["aircraft_id"])
        visible_queue = [str(task_id) for task_id in queues[aid]
                         if float(activation.get(task_id, float("inf"))) <= now
                         and task_id not in completed]
        task_index = int(cursor[aid][0])
        active_task = (queues[aid][task_index] if task_index < len(queues[aid])
                       and float(activation.get(queues[aid][task_index], float("inf"))) <= now
                       and queues[aid][task_index] not in completed else None)
        aircraft[aid] = {
            "kind": row["aircraft_type"], "home_zone": int(row["zone"]),
            "available": bool(actor_status[aid]["available"]), "faulted": bool(row["faulted"]),
            "safety_paused": bool(actor_status[aid]["safety_paused"]),
            "communication_held": bool(row["communication"]["safety_hold"]),
            "position_m": [float(value) for value in row["usd_position_m"]],
            "velocity_mps": [float(value) for value in row["velocity_mps"]],
            "available_energy_wh": float(row["energy"]["remaining_wh"]),
            "active_task_ids": [str(active_task)] if active_task is not None else [],
            "queue_task_ids": visible_queue,
        }
    public_tasks = {}
    for task_id, task in tasks.items():
        if float(activation.get(task_id, float("inf"))) > now:
            continue
        aid = assignments.get(task_id)
        task_queue = queues.get(aid, ()) if aid is not None else ()
        task_index = task_queue.index(task_id) if task_id in task_queue else None
        waypoint_cursor = (int(cursor[aid][1]) if task_index is not None
                           and task_index == int(cursor[aid][0]) else 0)
        public_tasks[str(task_id)] = {
            "zone": int(task.zone), "cell": int(task.cell), "kind": str(task.kind),
            "allowed_kind": str(task.allowed_kind),
            "public_xy_km": [float(v) for v in task.public_xy_km],
            "public_bounds_km": [float(v) for v in task.public_bounds_km],
            "assignment": str(aid) if aid is not None else None,
            "queue_position": task_index,
            "active": aid is not None and task_index == int(cursor[aid][0]) and task_id not in completed,
            "completed": task_id in completed,
            "route_waypoint_cursor": waypoint_cursor,
            "route_waypoint_count": len(routes.get(aid, {}).get(task_id, ())) if aid else 0,
            "priority": None, "deadline_sim_time_s": None,
        }
    observations = {}
    for row in public_rows:
        for item in row.get("public_target_observations", ()):
            if float(item.get("available_at_sim_time_s", float("inf"))) <= now:
                observations[str(item.get("observation_id", item.get("target_id")))] = item
    events = []
    decisions = []
    for event in event_rows:
        name = event.get("event")
        event_time = event.get("sim_time_s")
        if name not in PUBLIC_EVENTS or not isinstance(event_time, (int, float)) or event_time > now:
            continue
        public_event = {key: value for key, value in event.items()
                       if key in {"event", "event_id", "sim_time_s", "aircraft_id", "task_id",
                                  "decision_id", "target_id", "zone", "observation_id",
                                  "detected_at_sim_time_s", "available_at_sim_time_s",
                                  "decision_complete_sim_time_s", "decision_effect_sim_time_s",
                                  "selected_action", "reason", "timing", "from_aircraft_id",
                                  "to_aircraft_id", "chosen", "mode", "trigger_reasons",
                                  "gate_suppressed", "predicted_gain", "candidate_count"}}
        if name == "replan_catalog_choice" and isinstance(event.get("details"), Mapping):
            public_event["chosen"] = event["details"].get("chosen")
        events.append(public_event)
        if name in DECISION_EVENTS:
            decisions.append(public_event)
    costs = [{"aircraft_id": str(row["aircraft_id"]),
              "energy_incremental_wh": float(row["energy"]["incremental_wh"]),
              "command_attempted": bool(row["communication"]["command_attempted"]),
              "command_delivered": bool(row["communication"]["command_delivered"])}
             for row in public_rows]
    planner_cpu_ns = sum(int(event.get("timing", {}).get("cpu_duration_ns", 0))
                         for event in events if event.get("event") == "candidate_generation")
    decision_cpu_ns = sum(int(event.get("timing", {}).get("decision_cpu_duration_ns", 0))
                          for event in events if event.get("event") == "decision_created")
    costs.append({"sim_time_s": now, "planner_cpu_ns": planner_cpu_ns,
                  "decision_cpu_ns": decision_cpu_ns,
                  "task_switch_count": sum(event.get("event") == "controller_switch"
                                            for event in events),
                  "unrecovered_task_count": sum(event.get("event") == "fault_task_unrecovered"
                                                 for event in events)})
    complete = all(event.get("event") in PUBLIC_EVENTS
                   for event in event_rows
                   if isinstance(event.get("sim_time_s"), (int, float))
                   and event["sim_time_s"] <= now)
    return ({"sim_time_s": now, "aircraft": aircraft, "public_tasks": public_tasks,
             "public_observations": list(observations.values())}, events, decisions, costs, complete)
