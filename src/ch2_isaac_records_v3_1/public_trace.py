"""Minimal public scene records: one config and one records stream."""
from __future__ import annotations

import hashlib
import json
import gzip
from pathlib import Path
from typing import Any, Mapping


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _without_geometry(task: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items()
            if key not in {"public_xy_km", "public_bounds_km", "sweep_waypoints_km"}}


EVENTS = {
    "task_activated", "public_observation", "search_cell_complete", "controller_switch",
    "platform_fault", "communication_safety_hold", "energy_safety_hold", "energy_anomaly",
    "fault_task_unrecovered", "communication_safety_resume", "command_dropped",
    "command_delivered", "command_applied", "transmission_sent", "donor_support_transfer",
    "replan_action_rejected", "abstract_strike_complete", "target_discovered",
}
COMMAND_EVENTS = {"transmission_sent", "command_delivered", "command_applied", "command_dropped"}


def _rows(mapping: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": str(key), **dict(value)} for key, value in sorted(mapping.items())]


def _dynamic_aircraft(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    if "pos" not in row and "position_m" in row:
        row["pos"] = row.pop("position_m")
    if "energy" not in row and "available_energy_wh" in row:
        row["energy"] = row.pop("available_energy_wh")
    if "task_id" not in row:
        active = row.pop("active_task_ids", [])
        row["task_id"] = active[0] if active else None
    if "status" not in row:
        row["status"] = ("faulted" if row.get("faulted") else
                         "communication_lost" if row.get("communication_held") else
                         "paused" if row.get("safety_paused") else
                         "unavailable" if row.get("available") is False else
                         "normal" if row.get("available") is True else None)
    return {key: row[key] for key in ("pos", "energy", "status", "task_id", "velocity_mps")
            if key in row}


def _dynamic_task(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    if "status" not in row:
        row["status"] = ("completed" if row.get("completed") else
                         "running" if row.get("active") else
                         "pending" if "active" in row or "completed" in row else None)
    if "progress" not in row:
        count = row.get("route_waypoint_count", 0)
        cursor = row.get("route_waypoint_cursor", 0)
        row["progress"] = (1.0 if row.get("completed") else float(cursor) / count if count else None)
    if "executor" not in row:
        row["executor"] = row.get("assignment")
    return {key: row[key] for key in ("status", "progress", "executor")}


def _actual_actions(decisions: list[Mapping[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    rows = [row for row in decisions if row.get("event") in {
        "decision_created", "replan_catalog_choice"}]
    by_id: dict[str, Mapping[str, Any]] = {}
    anonymous: list[Mapping[str, Any]] = []
    for row in rows:
        decision_id = row.get("decision_id")
        if decision_id is None:
            anonymous.append(row)
        else:
            by_id[str(decision_id)] = row
    ordered = list(by_id.values()) + anonymous
    actions: list[tuple[int, dict[str, Any]]] = []
    for seq, row in enumerate(ordered, 1):
        selected = row.get("selected_action") or row.get("chosen")
        if selected is None:
            continue
        if isinstance(selected, Mapping):
            kind = selected.get("kind")
            if kind is None:
                continue
            action = {"type": "reassign" if kind == "uav_transfer" else "hold" if kind == "noop" else kind}
            if kind == "uav_transfer":
                action.update({"task_id": selected.get("task_id"),
                               "from": selected.get("donor"),
                               "to": selected.get("recipient")})
        elif isinstance(selected, str):
            action = {"type": "hold" if selected == "noop" else selected}
        else:
            continue
        for key in ("decision_id", "decision_complete_sim_time_s", "decision_effect_sim_time_s", "reason"):
            if key in row:
                action[key] = row[key]
        if "decision_effect_sim_time_s" in action:
            action["effect_time"] = action.pop("decision_effect_sim_time_s")
        actions.append((int(row.get("decision_sequence", seq)), action))
    return actions


def _actual_action(decisions: list[Mapping[str, Any]]):
    actions = _actual_actions(decisions)
    if not actions:
        return None
    return actions[-1][1]


def _event(event: Mapping[str, Any], default_t: int):
    name = event.get("event")
    if name not in EVENTS:
        return None
    object_id = event.get("aircraft_id", event.get("task_id", event.get("target_id")))
    change = {key: value for key, value in event.items()
              if key not in {"event", "event_id", "sim_time_s", "branch"}}
    return {"kind": "event", "t": event.get("sim_time_s", default_t),
            "event_id": event.get("event_id"), "event_type": name,
            "object": object_id, "change": change}


class PublicTraceWriter:
    def __init__(self, output_dir: str | Path, static_config: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "config.json").exists() or (self.output_dir / "records.jsonl").exists():
            raise FileExistsError(f"public capture already exists: {self.output_dir}")
        self.static_config = dict(static_config)
        self.static_hash = hashlib.sha256(_canonical(self.static_config)).hexdigest()
        (self.output_dir / "config.json").write_bytes(_canonical(self.static_config) + b"\n")
        self._records = (self.output_dir / "records.jsonl").open("w", encoding="utf-8", newline="\n")
        self.sample_count = 0
        self.event_count = 0
        self.decision_count = 0
        self._summary: dict[str, Any] | None = None
        self._written_decisions: set[tuple[Any, Any]] = set()
        self._command_events: set[tuple[Any, Any, Any]] = set()

    @staticmethod
    def _state(tick: Mapping[str, Any]) -> dict[str, Any]:
        return {"platforms": _rows({str(key): _dynamic_aircraft(value)
                                    for key, value in tick["aircraft"].items()}),
                "tasks": _rows({str(key): _dynamic_task(value)
                                for key, value in tick["public_tasks"].items()}),
                "observations": list(tick.get("public_observations") or [])}

    def append(self, tick: Mapping[str, Any], *, hidden_audit: Mapping[str, Any]) -> None:
        t = tick["sim_time_s"]
        decisions = list(tick.get("decisions") or [])
        actions = []
        for seq, action in _actual_actions(decisions):
            key = (t, action.get("decision_id", seq))
            if key not in self._written_decisions:
                self._written_decisions.add(key)
                actions.append((seq, action))
        periodic = abs(float(t) / 5.0 - round(float(t) / 5.0)) < 1e-9
        rows = []
        if periodic or actions:
            state = self._state(tick.get("decision_state", tick) if actions else tick)
            sample_rows = actions or [(None, None)]
            for decision_seq, action in sample_rows:
                row = {"kind": "sample", "t": t,
                       "reason": "decision" if action is not None else "periodic",
                       "state": state, "action": action}
                if decision_seq is not None:
                    row["decision_seq"] = int(decision_seq)
                rows.append(row)
                self.sample_count += 1
            self.decision_count += len(actions)
        for event in tick.get("events") or []:
            if event.get("event") in COMMAND_EVENTS:
                decision_id = event.get("decision_id")
                if decision_id is None:
                    continue
                key = (event["event"], decision_id, event.get("aircraft_id"))
                if key in self._command_events:
                    continue
                self._command_events.add(key)
            row = _event(event, t)
            if row is not None:
                rows.append(row)
                self.event_count += 1
        for row in sorted(rows, key=lambda item: item["t"]):
            self._records.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        self._summary = dict(summary)

    def flush(self) -> None:
        self._records.flush()

    def close(self) -> None:
        if self._records.closed:
            return
        if self._summary is not None:
            summary = dict(self._summary)
            summary.setdefault("kind", "summary")
            summary.setdefault("sample_count", self.sample_count)
            summary.setdefault("event_count", self.event_count)
            summary.setdefault("decision_count", self.decision_count)
            self._records.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
        self._records.flush()
        self._records.close()

    def file_info(self) -> dict[str, Any]:
        self.close()
        result = {"schema": "scene_records_v1", "static_config_sha256": self.static_hash,
                  "sample_count": self.sample_count, "event_count": self.event_count,
                  "decision_count": self.decision_count}
        for name in ("config.json", "records.jsonl"):
            path = self.output_dir / name
            result[name] = {"bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return result


class PublicTraceReader:
    """Read new records, with a read-only adapter for the old tick format."""

    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        config_path = self.trace_dir / "config.json"
        if config_path.is_file():
            self.format = "records"
            self.static_config = json.loads(config_path.read_text("utf-8"))
        else:
            self.format = "ticks"
            self.static_config = json.loads(
                (self.trace_dir / "static_config.json").read_text("utf-8"))

    def iter_records(self):
        if self.format == "ticks":
            aircraft: dict[str, Any] = {}
            tasks: dict[str, Any] = {}
            observations: dict[str, Any] = {}
            with gzip.open(self.trace_dir / "ticks.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    delta = record.get("deltas", {})
                    for name, target in (("aircraft", aircraft), ("public_tasks", tasks),
                                         ("observations", observations)):
                        change = delta.get(name, {})
                        target.update(change.get("upsert", {}))
                        for key in change.get("delete", ()):
                            target.pop(key, None)
                    yield {"projection": {"sim_time_s": int(record["sim_time_s"]),
                                           "aircraft": dict(aircraft),
                                           "public_tasks": dict(tasks),
                                           "public_observations": list(observations.values())},
                           "events": list(record.get("events") or []),
                           "decisions": list(record.get("decisions") or []),
                           "costs": list(record.get("costs") or [])}
            return

        with (self.trace_dir / "records.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("kind") == "sample":
                    state = record["state"]
                    aircraft = {item["id"]: {k: v for k, v in item.items() if k != "id"}
                                for item in state.get("platforms", [])}
                    tasks = {item["id"]: {k: v for k, v in item.items() if k != "id"}
                             for item in state.get("tasks", [])}
                    action = record.get("action")
                    yield {"projection": {"sim_time_s": record["t"],
                                           "aircraft": aircraft, "public_tasks": tasks,
                                           "public_observations": state.get("observations", [])},
                           "events": [],
                           "decisions": ([action] if action is not None else []),
                           "costs": []}
                elif record.get("kind") == "event":
                    yield {"projection": {"sim_time_s": record["t"],
                                           "aircraft": {}, "public_tasks": {},
                                           "public_observations": []},
                           "events": [{"event": record.get("event_type"),
                                       "sim_time_s": record.get("t"),
                                       "aircraft_id": record.get("object"),
                                       **dict(record.get("change") or {})}],
                           "decisions": [], "costs": []}

    def iter_public_states(self):
        for record in self.iter_records():
            projection = record["projection"]
            if projection.get("aircraft") or projection.get("public_tasks"):
                yield projection
