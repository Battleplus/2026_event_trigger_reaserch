"""Build masked one-second WM examples from public projection snapshots only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import OneStepTarget, PublicHistoryFrame


TARGET_NAMES = ("task_progress", "energy_delta", "delay", "risk")
PUBLIC_FIELD_SOURCES = {
    "platform.position": "projection.aircraft[*].position_m",
    "platform.velocity": "projection.aircraft[*].velocity_mps",
    "platform.energy": "projection.aircraft[*].available_energy_wh",
    "platform.status": "projection.aircraft[*].kind/home_zone/available/faulted/safety_paused/communication_held",
    "task.assignment_and_queue": "projection.public_tasks activated-task mapping; aircraft queues used for joins/order",
    "observation": "projection.public_observations[*], filtered by available_at_sim_time_s <= frame time",
    "tasks": "projection.public_tasks[*], only tasks already activated; IDs used only for joins",
    "events": "public events with sim_time_s <= frame time",
    "action": "projection.applied_action when explicitly public; absent action is masked",
    "task_progress_target": "public search_cell_complete events in (t, t+1] when event logging is complete",
    "energy_delta_target": "difference of mean available_energy_wh across matching public aircraft keys",
    "delay_target": "public transmission_sent to command_applied elapsed time, joined by decision_id",
    "risk_target": "public safety/fault event labels in (t, t+1] when event logging is complete",
}
UNSUPPORTED = (
    "per-task geometric sweep progress: absent from current public projection",
    "task priority/deadline/route and task-specific capability beyond public platform kind: absent from current projection",
    "fault task queue release and private task completion truth",
    "future event tape, hidden target coordinates before public observation, scenario identity/hash",
    "unlogged or right-censored command delay",
    "30-minute endpoint as a one-step target",
)


@dataclass(frozen=True)
class PublicOneStepExample:
    history: tuple[PublicHistoryFrame, ...]
    target: OneStepTarget
    input_time_s: int
    target_time_s: int
    source: str = "public_projection_adjacent_snapshots_v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _projection_time(projection: Mapping[str, Any]) -> int:
    value = _number(projection.get("sim_time_s"))
    if value is None or not value.is_integer():
        raise ValueError("public projection sim_time_s must be a finite integer second")
    return int(value)


def public_projection_to_frame(projection: Mapping[str, Any]) -> PublicHistoryFrame:
    """Map the runtime's explicit public projection whitelist to anonymous nodes."""
    time_s = _projection_time(projection)
    aircraft = projection.get("aircraft")
    observations = projection.get("public_observations")
    if not isinstance(aircraft, Mapping) or not isinstance(observations, Sequence):
        raise ValueError("projection must contain public aircraft and public_observations")
    platforms = []
    task_nodes = []
    task_owners = []
    for _aircraft_id, row in sorted(aircraft.items(), key=lambda item: str(item[0])):
        if not isinstance(row, Mapping):
            raise ValueError("public aircraft entry must be an object")
        required_status = ("kind", "available", "faulted", "safety_paused", "communication_held",
                           "home_zone")
        if any(key not in row for key in required_status):
            raise ValueError("public aircraft status and queue fields must be explicit; missing values are not inferred")
        position = row.get("position_m")
        velocity = row.get("velocity_mps")
        energy = _number(row.get("available_energy_wh"))
        if (not isinstance(position, Sequence) or len(position) < 3
                or not isinstance(velocity, Sequence) or len(velocity) < 3 or energy is None):
            raise ValueError("public aircraft position, velocity, and energy are required")
        home_zone = _number(row.get("home_zone"))
        if home_zone is None:
            raise ValueError("public aircraft home_zone must be finite")
        xyz = [_number(value) for value in position[:3]]
        vxyz = [_number(value) for value in velocity[:3]]
        if any(value is None for value in (*xyz, *vxyz)):
            raise ValueError("public aircraft position and velocity must be finite")
        queue_known = isinstance(row.get("active_task_ids"), Sequence) and isinstance(row.get("queue_task_ids"), Sequence)
        active_ids = {str(value) for value in row.get("active_task_ids", ())} if queue_known else set()
        queue_ids = [str(value) for value in row.get("queue_task_ids", ())] if queue_known else []
        platforms.append({
            "x_norm": xyz[0] / 300_000.0, "y_norm": xyz[1] / 300_000.0,
            "z_norm": xyz[2] / 100_000.0,
            "speed_norm": math.sqrt(sum(value * value for value in vxyz)) / 100.0,
            "energy_log_norm": math.log1p(max(0.0, energy)) / 20.0,
            "platform_kind": row.get("kind"),
            "home_zone_norm": home_zone / 8.0,
            "available": row.get("available", True),
            "faulted": row.get("faulted", False),
            "communication_held": bool(row.get("communication_held", False)
                                        or row.get("safety_paused", False)),
            "active": bool(row.get("available", True)) and not bool(row.get("faulted", False)),
            "queue_length_norm": min(1.0, len(queue_ids) / 36.0),
            "active_task_count_norm": min(1.0, len(active_ids) / 4.0),
            "queue_data_known": queue_known,
            "_public_zone": home_zone,
        })
    observation_nodes = []
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        available_at = _number(item.get("available_at_sim_time_s"))
        if available_at is None or available_at > time_s:
            continue
        observed_at = _number(item.get("observed_at_sim_time_s"))
        xy = item.get("target_xy_km")
        node: dict[str, Any] = {"active": 1.0,
                                "age_norm": max(0.0, (time_s - available_at) / 5400.0),
                                "_public_zone": _number(item.get("zone"))}
        if observed_at is not None:
            node["availability_delay_norm"] = max(0.0, (available_at - observed_at) / 5400.0)
        if isinstance(xy, Sequence) and len(xy) >= 2:
            x, y = _number(xy[0]), _number(xy[1])
            if x is not None and y is not None:
                node["x_norm"] = x / 300.0
                node["y_norm"] = y / 300.0
        observation_nodes.append(node)
    public_tasks = projection.get("public_tasks")
    if isinstance(public_tasks, Mapping):
        queue_positions = {}
        for aid, row in aircraft.items():
            for index, task_id in enumerate(row.get("queue_task_ids", ())):
                queue_positions[str(task_id)] = (index, len(row.get("queue_task_ids", ())))
        for task_id, record in sorted(public_tasks.items(), key=lambda item: str(item[0])):
            if not isinstance(record, Mapping):
                continue
            task_id = str(task_id)
            assignment = record.get("assignment")
            owner = str(assignment) if assignment is not None else None
            zone = _number(record.get("zone"))
            node: dict[str, Any] = {"_public_id": task_id, "_owner_id": owner,
                "_public_zone": zone, "task_public_data_known": True,
                "active": bool(record.get("active", False)),
                "completed": bool(record.get("completed", False))}
            xy = record.get("public_xy_km")
            if zone is not None:
                node["home_zone_norm"] = zone / 8.0
            if isinstance(xy, Sequence) and not isinstance(xy, (str, bytes)) and len(xy) >= 2:
                x, y = _number(xy[0]), _number(xy[1])
                if x is not None and y is not None:
                    node["x_norm"] = x / 300.0
                    node["y_norm"] = y / 300.0
            queue_info = queue_positions.get(task_id)
            position = record.get("queue_position")
            if position is None and queue_info is not None:
                position, queue_len = queue_info
            else:
                queue_len = max(1, len(aircraft.get(owner, {}).get("queue_task_ids", ())))
            if position is not None:
                node["queue_position_norm"] = float(position) / max(1, queue_len - 1)
                node["queue_length_norm"] = min(1.0, queue_len / 36.0)
            task_nodes.append(node)
            task_owners.append(owner)
    else:
        # Legacy sparse projections retain queue nodes while marking task metadata unknown.
        for aid, row in aircraft.items():
            queue_ids = row.get("queue_task_ids", ())
            active_ids = set(row.get("active_task_ids", ()))
            for index, task_id in enumerate(queue_ids):
                task_nodes.append({"_public_id": str(task_id), "_owner_id": str(aid),
                    "task_public_data_known": False, "active": str(task_id) in active_ids,
                    "queue_position_norm": index / max(1, len(queue_ids) - 1),
                    "queue_length_norm": min(1.0, len(queue_ids) / 36.0)})
                task_owners.append(str(aid))
    node_keys = ([str(key) for key in sorted(aircraft, key=str)]
                 + [str(node["_public_id"]) for node in task_nodes]
                 + [None] * len(observation_nodes))
    action = projection.get("applied_action")
    if not isinstance(action, Mapping):
        action = None
    return PublicHistoryFrame(time_s=float(time_s), platforms=tuple(platforms),
                              tasks=tuple(task_nodes), observations=tuple(observation_nodes),
                              event={}, action=action, node_keys=tuple(node_keys),
                              task_owners=tuple([None] * len(platforms) + task_owners + [None] * len(observation_nodes)))


def _event_time(event: Mapping[str, Any]) -> int | None:
    value = _number(event.get("sim_time_s"))
    return int(value) if value is not None and value.is_integer() else None


def _event_available_time(event: Mapping[str, Any]) -> int | None:
    value = _number(event.get("available_at_sim_time_s"))
    return _event_time(event) if value is None else (int(value) if value.is_integer() else None)


def build_public_one_step_example(
    projections: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] = (),
    *,
    events_complete: bool = False,
    gain_label: float | None = None,
    gain_label_valid: bool = False,
) -> PublicOneStepExample:
    """Use all but the final snapshot as features and the adjacent final tick as labels."""
    if len(projections) < 2:
        raise ValueError("at least two adjacent public projections are required")
    times = [_projection_time(item) for item in projections]
    if any(right - left != 1 for left, right in zip(times, times[1:])):
        raise ValueError("public one-step history must contain adjacent 1-second snapshots")
    start_time, target_time = times[-2], times[-1]
    frames = []
    for projection, tick in zip(projections, times):
        frame = public_projection_to_frame(projection)
        visible_events = tuple(event for event in events if isinstance(event, Mapping)
                               and _event_time(event) is not None and _event_time(event) <= tick
                               and _event_available_time(event) is not None
                               and _event_available_time(event) <= tick)
        frames.append(PublicHistoryFrame(time_s=frame.time_s, platforms=frame.platforms,
            tasks=frame.tasks, observations=frame.observations, event=frame.event,
            public_events=visible_events, action=frame.action, node_keys=frame.node_keys,
            task_owners=frame.task_owners))
    if gain_label is not None and (not gain_label_valid or _number(gain_label) is None):
        raise ValueError("gain label requires an independently validated finite paired label")
    if gain_label_valid and gain_label is None:
        raise ValueError("gain_label_valid requires a finite paired gain label")
    allowed_events = []
    event_timestamps_complete = True
    for event in events:
        if not isinstance(event, Mapping):
            event_timestamps_complete = False
            continue
        event_time = _event_time(event)
        if event_time is None:
            event_timestamps_complete = False
        elif event_time <= target_time:
            allowed_events.append(event)
    interval_events = [event for event in allowed_events
                       if start_time < _event_time(event) <= target_time]

    previous_aircraft = projections[-2].get("aircraft", {})
    current_aircraft = projections[-1].get("aircraft", {})
    energy_values = []
    energy_valid = (isinstance(previous_aircraft, Mapping) and isinstance(current_aircraft, Mapping)
                    and set(previous_aircraft) == set(current_aircraft) and bool(previous_aircraft))
    if energy_valid:
        for aircraft_id in previous_aircraft:
            before = _number(previous_aircraft[aircraft_id].get("available_energy_wh"))
            after = _number(current_aircraft[aircraft_id].get("available_energy_wh"))
            if before is None or after is None:
                energy_valid = False
                break
            energy_values.append(after - before)

    applied_delays = []
    transmissions: dict[tuple[str, str], list[int]] = {}
    applications: dict[tuple[str, str], list[int]] = {}
    for event in allowed_events:
        event_type = event.get("event")
        if event_type not in {"transmission_sent", "command_applied"}:
            continue
        event_time = _event_time(event)
        if event_time is None or event.get("decision_id") is None or event.get("aircraft_id") is None:
            continue
        key = (str(event["aircraft_id"]), str(event["decision_id"]))
        destination = transmissions if event_type == "transmission_sent" else applications
        destination.setdefault(key, []).append(event_time)
    for key, sent_times in transmissions.items():
        applied_times = applications.get(key, [])
        if len(sent_times) != 1 or len(applied_times) != 1:
            continue
        created, applied_at = sent_times[0], applied_times[0]
        if start_time < applied_at <= target_time and applied_at >= created:
            applied_delays.append(float(applied_at - created))

    progress_events = sum(event.get("event") == "search_cell_complete" for event in interval_events)
    risk_events = {"platform_fault", "communication_safety_hold", "energy_safety_hold",
                   "safety_violation", "boundary_violation", "constraint_violation"}
    risk_observed = any(event.get("event") in risk_events for event in interval_events)
    masks = {
        "task_progress": bool(events_complete and event_timestamps_complete),
        "energy_delta": bool(energy_valid),
        "delay": bool(applied_delays),
        "risk": bool(events_complete and event_timestamps_complete),
    }
    targets = {
        "task_progress": float(progress_events),
        "energy_delta": (sum(energy_values) / len(energy_values) if energy_valid and energy_values else 0.0),
        "delay": (sum(applied_delays) / len(applied_delays) if applied_delays else 0.0),
        "risk": float(risk_observed),
    }
    target = OneStepTarget(dynamic=targets, gain=gain_label,
        dynamic_mask=masks,
        reason=None if any(masks.values()) or gain_label_valid else "no observable one-step targets")
    return PublicOneStepExample(
        history=tuple(frames[:-1]),
        target=target, input_time_s=start_time, target_time_s=target_time)
