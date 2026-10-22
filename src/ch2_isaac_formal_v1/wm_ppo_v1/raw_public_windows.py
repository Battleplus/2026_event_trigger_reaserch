"""Stream public WM windows from formal_collection_v1 raw capture outputs.

Trajectory rows are read once from gzip and retained only for one tick plus a
fixed-length rolling history. The event file is much smaller and is indexed in
memory so branch-local visibility and labels remain correct even if rows are
not time-sorted.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contracts import OneStepTarget, PublicHistoryFrame


PUBLIC_EVENT_TYPES = frozenset({
    "task_activated", "public_observation", "search_cell_complete", "energy_anomaly",
    "platform_fault", "communication_safety_hold", "energy_safety_hold",
    "communication_safety_resume", "command_dropped", "command_delivered",
    "command_applied", "transmission_sent", "controller_switch",
    "safety_violation", "boundary_violation", "constraint_violation",
})
RISK_EVENTS = frozenset({"platform_fault", "communication_safety_hold", "energy_safety_hold",
                         "safety_violation", "boundary_violation", "constraint_violation"})


@dataclass(frozen=True)
class RawPublicWindow:
    history: tuple[PublicHistoryFrame, ...]
    target: OneStepTarget
    input_time_s: int
    target_time_s: int
    branch: str
    source_run_id: str
    source_phase: str
    state_id: str | None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _events_by_branch(path: Path) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    complete = True
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event JSON at line {line_number}") from exc
            if not isinstance(row, Mapping):
                complete = False
                continue
            event_name = row.get("event")
            if event_name not in PUBLIC_EVENT_TYPES:
                continue
            event_time = _finite_number(row.get("sim_time_s"))
            if event_time is None:
                complete = False
                continue
            raw_available = row.get("available_at_sim_time_s", event_time)
            available = _finite_number(raw_available)
            if available is None:
                complete = False
                continue
            # Keep only public join/clock fields; payload blobs can contain private data.
            item = {key: row[key] for key in (
                "event", "sim_time_s", "available_at_sim_time_s", "branch", "aircraft_id",
                "task_id", "decision_id", "activation_sim_time_s", "command_applied_sim_time_s",
                "transmission_sent_sim_time_s") if key in row}
            item["_visible_at_s"] = max(event_time, available)
            indexed.setdefault(str(row.get("branch", "")), []).append(item)
    for rows in indexed.values():
        rows.sort(key=lambda item: (item["_visible_at_s"], item["sim_time_s"], item["event"]))
    return indexed, complete


def _manifest_tasks(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = receipt.get("public_task_manifest", ())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return {}
    return {str(item["task_id"]): item for item in records
            if isinstance(item, Mapping) and item.get("task_id") is not None}


def _tick_frame(
    tick: int,
    rows: Sequence[Mapping[str, Any]],
    task_manifest: Mapping[str, Mapping[str, Any]],
    activated: set[str],
    completed: set[str],
    visible_events: Sequence[Mapping[str, Any]],
) -> PublicHistoryFrame:
    platforms: list[dict[str, Any]] = []
    aircraft_keys: list[str] = []
    owner_by_task: dict[str, str] = {}
    observations: dict[str, Mapping[str, Any]] = {}
    for raw in sorted(rows, key=lambda row: str(row.get("aircraft_id", ""))):
        aid = str(raw["aircraft_id"])
        position = raw.get("usd_position_m")
        velocity = raw.get("velocity_mps")
        energy = raw.get("energy")
        communication = raw.get("communication")
        if (not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 3
                or not isinstance(velocity, Sequence) or isinstance(velocity, (str, bytes)) or len(velocity) < 3
                or not isinstance(energy, Mapping) or not isinstance(communication, Mapping)):
            raise ValueError(f"trajectory actor row at t={tick} lacks public physical/status fields")
        xyz = [_finite_number(value) for value in position[:3]]
        vxyz = [_finite_number(value) for value in velocity[:3]]
        remaining = _finite_number(energy.get("remaining_wh"))
        if any(value is None for value in (*xyz, *vxyz)) or remaining is None:
            raise ValueError(f"trajectory actor row at t={tick} has non-finite public state")
        faulted = raw.get("faulted")
        comm_hold = communication.get("safety_hold")
        energy_pause = energy.get("safety_pause")
        platform: dict[str, Any] = {
            "x_norm": xyz[0] / 300_000.0, "y_norm": xyz[1] / 300_000.0,
            "z_norm": xyz[2] / 100_000.0,
            "speed_norm": math.sqrt(sum(value * value for value in vxyz)) / 100.0,
            "energy_log_norm": math.log1p(max(0.0, remaining)) / 20.0,
            "platform_kind": raw.get("aircraft_type"),
            "queue_data_known": False,
            "active": False,
            "availability_known": isinstance(raw.get("available"), bool),
        }
        zone = _finite_number(raw.get("zone"))
        if zone is not None:
            platform["home_zone_norm"] = zone / 8.0
            platform["_public_zone"] = zone
        if isinstance(faulted, bool):
            platform["faulted"] = faulted
        if isinstance(comm_hold, bool):
            platform["communication_held"] = comm_hold
        if isinstance(raw.get("available"), bool):
            platform["available"] = raw["available"]
        if isinstance(energy_pause, bool) and isinstance(faulted, bool) and isinstance(comm_hold, bool):
            platform["active"] = not (faulted or comm_hold or energy_pause)
            platform["safety_paused"] = energy_pause
        if isinstance(raw.get("task_id"), str) and raw["task_id"] in activated:
            owner_by_task[raw["task_id"]] = aid
        public_observations = raw.get("public_target_observations", ())
        if isinstance(public_observations, Sequence) and not isinstance(public_observations, (str, bytes)):
            for observation in public_observations:
                if not isinstance(observation, Mapping):
                    continue
                available = _finite_number(observation.get("available_at_sim_time_s"))
                xy = observation.get("target_xy_km")
                observed = _finite_number(observation.get("observed_at_sim_time_s"))
                if (available is None or available > tick or not isinstance(xy, Sequence)
                        or isinstance(xy, (str, bytes)) or len(xy) < 2):
                    continue
                x, y = _finite_number(xy[0]), _finite_number(xy[1])
                if x is None or y is None:
                    continue
                # The raw observation identifier is a deduplication key only.
                key = str(observation.get("observation_id", observation.get("target_id", len(observations))))
                obs: dict[str, Any] = {"active": True, "x_norm": x / 300.0, "y_norm": y / 300.0,
                                       "_public_zone": _finite_number(observation.get("zone"))}
                if observed is not None:
                    obs["age_norm"] = max(0.0, (tick - observed) / 5400.0)
                observations[key] = obs
        platforms.append(platform)
        aircraft_keys.append(aid)

    tasks: list[dict[str, Any]] = []
    task_owners: list[str | None] = []
    task_keys: list[str] = []
    for task_id in sorted(activated):
        record = task_manifest.get(task_id)
        node: dict[str, Any] = {"_public_id": task_id, "_public_zone": None,
                                "task_public_data_known": record is not None,
                                "active": task_id not in completed, "completed": task_id in completed}
        owner = owner_by_task.get(task_id)
        if record is not None:
            zone = _finite_number(record.get("zone"))
            bounds = record.get("public_bounds_km")
            if zone is not None:
                node["_public_zone"] = zone
                node["home_zone_norm"] = zone / 8.0
            if isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)) and len(bounds) >= 4:
                values = [_finite_number(value) for value in bounds[:4]]
                if all(value is not None for value in values):
                    node["x_norm"] = ((values[0] + values[2]) / 2.0) / 300.0
                    node["y_norm"] = ((values[1] + values[3]) / 2.0) / 300.0
        tasks.append(node)
        task_owners.append(owner)
        task_keys.append(task_id)
    return PublicHistoryFrame(
        time_s=float(tick), platforms=tuple(platforms), tasks=tuple(tasks),
        observations=tuple(observations.values()), public_events=tuple(visible_events), action=None,
        node_keys=tuple(aircraft_keys + task_keys + [None] * len(observations)),
        task_owners=tuple([None] * len(platforms) + task_owners + [None] * len(observations)))


def _target(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    branch_events: Sequence[Mapping[str, Any]],
    t: int,
    events_complete: bool,
    gain: float | None,
) -> OneStepTarget:
    interval = [event for event in branch_events
                if t < float(event["sim_time_s"]) <= t + 1
                and float(event["_visible_at_s"]) <= t + 1]
    before_by_id = {str(row.get("aircraft_id")): row for row in before}
    after_by_id = {str(row.get("aircraft_id")): row for row in after}
    energy_deltas: list[float] = []
    energy_valid = bool(before_by_id) and before_by_id.keys() == after_by_id.keys()
    if energy_valid:
        for aid, old in before_by_id.items():
            old_energy = old.get("energy")
            new_energy = after_by_id[aid].get("energy")
            old_value = _finite_number(old_energy.get("remaining_wh")) if isinstance(old_energy, Mapping) else None
            new_value = _finite_number(new_energy.get("remaining_wh")) if isinstance(new_energy, Mapping) else None
            if old_value is None or new_value is None:
                energy_valid = False
                break
            energy_deltas.append(new_value - old_value)
    sent: dict[tuple[str, str], list[int]] = {}
    applied: dict[tuple[str, str], list[int]] = {}
    for event in branch_events:
        event_name = event.get("event")
        if event_name not in {"transmission_sent", "command_applied"}:
            continue
        if (float(event["sim_time_s"]) > t + 1
                or float(event["_visible_at_s"]) > t + 1):
            continue
        event_time = int(float(event["sim_time_s"]))
        aid, decision_id = event.get("aircraft_id"), event.get("decision_id")
        if aid is None or decision_id is None:
            continue
        bucket = sent if event_name == "transmission_sent" else applied
        bucket.setdefault((str(aid), str(decision_id)), []).append(event_time)
    delays = []
    for key, sent_times in sent.items():
        applied_times = applied.get(key, ())
        if len(sent_times) == len(applied_times) == 1:
            a, b = sent_times[0], applied_times[0]
            if t < b <= t + 1 and b >= a:
                delays.append(float(b - a))
    progress = sum(event.get("event") == "search_cell_complete" for event in interval)
    risk = any(event.get("event") in RISK_EVENTS for event in interval)
    masks = {"task_progress": events_complete, "energy_delta": energy_valid,
             "delay": bool(delays), "risk": events_complete}
    dynamic = {"task_progress": float(progress),
               "energy_delta": sum(energy_deltas) / len(energy_deltas) if energy_valid else 0.0,
               "delay": sum(delays) / len(delays) if delays else 0.0,
               "risk": float(risk)}
    return OneStepTarget(dynamic=dynamic, gain=gain, valid=any(masks.values()),
                         reason=None if any(masks.values()) else "no observable one-step targets",
                         dynamic_mask=masks)


def _validated_gain(
    labels: Iterable[Mapping[str, Any]], *, state_id: str | None, run_id: str,
    phase: str, branch: str, input_time: int, target_time: int,
) -> float | None:
    if state_id is None:
        return None
    for item in labels:
        if (item.get("validated") is True and item.get("state_id") == state_id
                and item.get("source_run_id") == run_id and item.get("source_phase") == phase
                and item.get("branch") == branch and item.get("input_time_s") == input_time
                and item.get("target_time_s") == target_time):
            return _finite_number(item.get("gain"))
    return None


def _ticks(stream) -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    key = None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trajectory JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"trajectory line {line_number} must be a JSON object")
        branch = str(row.get("branch", ""))
        sim_time = _finite_number(row.get("sim_time_s"))
        if not branch or sim_time is None or not sim_time.is_integer():
            raise ValueError(f"trajectory line {line_number} lacks a branch or integer sim_time_s")
        new_key = (branch, int(sim_time))
        if key is not None and new_key != key:
            yield key[0], key[1], rows
            rows = []
        key = new_key
        rows.append(row)
    if key is not None:
        yield key[0], key[1], rows


def iter_raw_public_windows(
    trajectory_path: str | Path,
    events_path: str | Path,
    receipt_path: str | Path,
    *,
    history_steps: int = 8,
    manifest_path: str | Path | None = None,
    validated_gain_labels: Iterable[Mapping[str, Any]] = (),
) -> Iterator[RawPublicWindow]:
    """Yield branch-local, adjacent one-second public windows from captured files."""
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    trajectory_path, events_path, receipt_path = map(Path, (trajectory_path, events_path, receipt_path))
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("actual_isaac_run") is not True:
        raise ValueError("raw public windows require a complete actual-ISAAC receipt")
    manifest_file = Path(manifest_path) if manifest_path else receipt_path.with_name("manifest.json")
    manifest = _read_json(manifest_file) if manifest_file.is_file() else {}
    log_names = manifest.get("logs", {}) if isinstance(manifest.get("logs"), Mapping) else {}
    for provided, key in ((trajectory_path, "trajectory"), (events_path, "events")):
        receipt_name = receipt.get(key)
        manifest_name = log_names.get(key)
        for expected in (receipt_name, manifest_name):
            if expected is not None and provided.name != expected:
                raise ValueError(f"{key} path does not match receipt/manifest log name")
    task_manifest = _manifest_tasks(receipt)
    event_map, event_rows_complete = _events_by_branch(events_path)
    event_complete = bool(event_rows_complete and receipt.get("events_complete", True))
    actor_manifest = receipt.get("actor_manifest", ())
    expected_actors = {str(item["aircraft_id"]) for item in actor_manifest
                       if isinstance(item, Mapping) and item.get("aircraft_id") is not None}
    run_id, phase = str(receipt.get("run_id", "")), str(receipt.get("phase", ""))
    suffix = receipt.get("suffix", {}) if isinstance(receipt.get("suffix"), Mapping) else {}
    state_id_raw = suffix.get("state_id", receipt.get("state_id"))
    state_id = str(state_id_raw) if state_id_raw is not None else None
    seed_remaining = set(map(str, suffix.get("audit_interface", {}).get("initial_remaining_public_task_ids", ())))
    seed_completed = set(map(str, suffix.get("audit_interface", {}).get("initial_completed_public_task_ids", ())))
    has_suffix_task_seed = isinstance(suffix.get("audit_interface"), Mapping)
    suffix_state_time = _finite_number(suffix.get("decision_time_s"))
    pair_snapshot = receipt.get("pair_snapshot", {})
    pair_snapshot_time = (_finite_number(pair_snapshot.get("sim_time_s"))
                          if isinstance(pair_snapshot, Mapping) else None)
    shared_pair_activated: set[str] = set()
    shared_pair_completed: set[str] = set()
    if pair_snapshot_time is not None:
        for event in event_map.get("pair_prefix", ()):
            if event["_visible_at_s"] > pair_snapshot_time:
                continue
            task_id = event.get("task_id")
            if task_id is None:
                continue
            if event.get("event") == "task_activated":
                shared_pair_activated.add(str(task_id))
            elif event.get("event") == "search_cell_complete":
                shared_pair_completed.add(str(task_id))
        shared_pair_activated.update(shared_pair_completed)
    labels = tuple(validated_gain_labels)

    active_branch: str | None = None
    previous_tick: int | None = None
    previous_rows: list[dict[str, Any]] | None = None
    history: deque[PublicHistoryFrame] = deque(maxlen=history_steps)
    event_pointer = 0
    ordered_events: list[dict[str, Any]] = []
    activated: set[str] = set()
    completed: set[str] = set()
    with gzip.open(trajectory_path, "rt", encoding="utf-8") as stream:
        for branch, tick, rows in _ticks(stream):
            row_ids = [str(row.get("aircraft_id", "")) for row in rows]
            if len(set(row_ids)) != len(row_ids):
                raise ValueError(f"duplicate actor row for branch={branch} t={tick}")
            if expected_actors and set(row_ids) != expected_actors:
                # Incomplete ticks break both the prior window and future adjacency.
                active_branch, previous_tick, previous_rows = branch, tick, None
                history.clear()
                activated.clear()
                completed.clear()
                event_pointer = 0
                ordered_events = event_map.get(branch, [])
                continue
            if branch != active_branch:
                active_branch, previous_tick, previous_rows = branch, None, None
                history.clear()
                activated, completed = set(), set()
                if branch in {"keep", "replan"}:
                    if has_suffix_task_seed:
                        if suffix_state_time is not None and tick >= suffix_state_time:
                            activated.update(seed_remaining)
                            completed.update(seed_completed)
                            activated.update(seed_completed)
                    elif pair_snapshot_time is not None and tick >= pair_snapshot_time:
                        # The shared prefix contributes public task state only, never history frames.
                        activated.update(shared_pair_activated)
                        completed.update(shared_pair_completed)
                event_pointer = 0
                ordered_events = event_map.get(branch, [])
            if previous_tick is not None and tick != previous_tick + 1:
                history.clear()
                previous_rows = None
            newly_visible = []
            while event_pointer < len(ordered_events) and ordered_events[event_pointer]["_visible_at_s"] <= tick:
                event = ordered_events[event_pointer]
                newly_visible.append(event)
                if event.get("event") == "task_activated" and event.get("task_id") is not None:
                    activated.add(str(event["task_id"]))
                elif event.get("event") == "search_cell_complete" and event.get("task_id") is not None:
                    completed.add(str(event["task_id"]))
                event_pointer += 1
            frame = _tick_frame(tick, rows, task_manifest, activated, completed, newly_visible)
            if previous_rows is not None and previous_tick == tick - 1:
                gain = _validated_gain(labels, state_id=state_id, run_id=run_id, phase=phase,
                    branch=branch, input_time=previous_tick, target_time=tick)
                target = _target(previous_rows, rows, ordered_events, previous_tick, event_complete, gain)
                yield RawPublicWindow(history=tuple(history), target=target,
                    input_time_s=previous_tick, target_time_s=tick, branch=branch,
                    source_run_id=run_id, source_phase=phase, state_id=state_id)
            history.append(frame)
            previous_tick, previous_rows = tick, rows
