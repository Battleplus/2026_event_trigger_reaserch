"""Streaming compact collection of public per-second state.

Callers must pass an already-public projection. Known hidden-field names are
filtered as defense in depth; this is not a classifier for arbitrary private
fields. This is a local storage interface, not wired into an Isaac runner.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "compact_collection_v1"
_OBSERVATION_KEYS = ("observation_id", "id")
_TASK_KEYS = ("task_id", "id")
_PRIVATE_KEY_NAMES = {
    "scenario_id", "scenario_hash", "hidden_target_coordinates", "hidden_targets",
    "target_truth", "ground_truth", "future_event_tape",
}
_DELTA_TABLES = ("aircraft", "tasks", "public_tasks")


def _json_value(value: Any) -> Any:
    """Copy JSON-compatible values and reject NaN/Infinity and custom objects."""
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    return json.loads(encoded)


def _public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_KEY_NAMES
            and not str(key).lower().startswith(("hidden_", "future_private_"))
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return _json_value(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _time_s(state: Mapping[str, Any]) -> int:
    value = state.get("sim_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("public state sim_time_s must be an integer second")
    if not math.isfinite(float(value)) or not float(value).is_integer():
        raise ValueError("public state sim_time_s must be an integer second")
    return int(value)


def _record_map(value: Any, id_keys: Sequence[str], field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(f"{field_name} list entries must be objects")
            identifier = next((item[key] for key in id_keys if item.get(key) is not None), None)
            if identifier is None:
                raise ValueError(f"{field_name} list entries require a public identifier")
            key = str(identifier)
            if key in result:
                raise ValueError(f"{field_name} contains duplicate public identifiers")
            result[key] = _public_value(item)
        return result
    if value is None:
        return {}
    raise ValueError(f"{field_name} must be an object or identified object list")


def _keyed_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "upsert": {key: value for key, value in current.items() if previous.get(key) != value},
        "delete": sorted(set(previous) - set(current)),
    }


def _apply_keyed_delta(previous: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(previous)
    for key in delta.get("delete", ()):
        result.pop(str(key), None)
    result.update({str(key): value for key, value in delta.get("upsert", {}).items()})
    return result


class CompactCollectionWriter:
    """Append public deltas and separate evaluation-only truth into a new directory."""

    def __init__(self, output_dir: str | Path, static_config: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        config = _public_value(static_config)
        self.static_hash = hashlib.sha256(_canonical_bytes(config)).hexdigest()
        (self.output_dir / "static_config.json").write_bytes(_canonical_bytes(config) + b"\n")
        manifest = {"schema": SCHEMA_VERSION, "static_config_sha256": self.static_hash}
        (self.output_dir / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        self._previous: dict[str, Any] | None = None
        self._previous_time: int | None = None

    def append_tick(
        self,
        public_state: Mapping[str, Any],
        *,
        events: Sequence[Mapping[str, Any]] = (),
        decisions: Sequence[Mapping[str, Any]] = (),
        costs: Sequence[Mapping[str, Any]] = (),
        event_log_complete: bool = False,
        hidden_truth: Mapping[str, Any] | None = None,
    ) -> None:
        state = _public_value(public_state)
        if not isinstance(state, dict) or "aircraft" not in state:
            raise ValueError("public state must be an object containing aircraft")
        now = _time_s(state)
        if self._previous_time is not None and now != self._previous_time + 1:
            raise ValueError("compact stream requires contiguous one-second public states")
        if not isinstance(state["aircraft"], (Mapping, list)):
            raise ValueError("aircraft must be an object or identified object list")

        tables = {name: _record_map(state.get(name), _TASK_KEYS, name)
                  for name in _DELTA_TABLES if name in state}
        tables["aircraft"] = _record_map(state["aircraft"], ("aircraft_id", "id"), "aircraft")
        observations_value = state.get("public_observations", [])
        if not isinstance(observations_value, list):
            raise ValueError("public_observations must be a list")
        observations = _record_map(observations_value, _OBSERVATION_KEYS, "public_observations")

        previous = self._previous or {}
        previous_tables = previous.get("_tables", {})
        table_deltas = {}
        for name, values in tables.items():
            table_deltas[name] = _keyed_delta(previous_tables.get(name, {}), values)
        for name in set(previous_tables) - set(tables):
            table_deltas[name] = {"upsert": {}, "delete": sorted(previous_tables[name])}
        previous_observations = previous.get("_observations", {})
        observation_delta = _keyed_delta(previous_observations, observations)

        meta = {key: value for key, value in state.items()
                if key not in {*_DELTA_TABLES, "public_observations", "sim_time_s"}}
        previous_meta = previous.get("_meta", {})
        meta_update = {key: value for key, value in meta.items() if previous_meta.get(key) != value}
        meta_delete = sorted(set(previous_meta) - set(meta))
        record = {
            "sim_time_s": now,
            "static_config_sha256": self.static_hash,
            "tables": table_deltas,
            "observation_delta": observation_delta,
            "meta_update": meta_update,
            "meta_delete": meta_delete,
            "events": _public_value(list(events)),
            "decisions": _public_value(list(decisions)),
            "costs": _public_value(list(costs)),
            "event_log_complete": bool(event_log_complete),
        }
        with (self.output_dir / "ticks.jsonl").open("ab") as stream:
            stream.write(_canonical_bytes(record) + b"\n")
            stream.flush()
        if hidden_truth is not None:
            truth_record = {"sim_time_s": now, "truth": _json_value(hidden_truth)}
            with (self.output_dir / "evaluation_truth.jsonl").open("ab") as stream:
                stream.write(_canonical_bytes(truth_record) + b"\n")
                stream.flush()
        self._previous = {"_tables": tables, "_observations": observations, "_meta": meta}
        self._previous_time = now

    def append_pair_snapshot(
        self,
        pair_id: str,
        role: str,
        public_state: Mapping[str, Any],
        *,
        pair_engine_snapshot_ref: str,
        pair_engine_snapshot_sha256: str,
    ) -> None:
        """Store a complete public snapshot and reference external recovery state."""
        digest = pair_engine_snapshot_sha256.lower()
        if (not pair_id or role not in {"keep", "replan", "before", "after"}
                or not pair_engine_snapshot_ref or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("pair snapshot requires pair identity, role, reference, and SHA-256")
        snapshot = _public_value(public_state)
        _time_s(snapshot)
        record = {"pair_id": str(pair_id), "role": role, "snapshot": snapshot,
                  "static_config_sha256": self.static_hash,
                  "pair_engine_snapshot_ref": str(pair_engine_snapshot_ref),
                  "pair_engine_snapshot_sha256": digest}
        with (self.output_dir / "paired_snapshots.jsonl").open("ab") as stream:
            stream.write(_canonical_bytes(record) + b"\n")
            stream.flush()


class CompactPublicReader:
    """Read only public static config and tick records; never opens evaluation truth."""

    def __init__(self, collection_dir: str | Path):
        self.collection_dir = Path(collection_dir)
        self.manifest = json.loads((self.collection_dir / "manifest.json").read_text("utf-8"))
        self.static_config = json.loads((self.collection_dir / "static_config.json").read_text("utf-8"))
        actual_hash = hashlib.sha256(_canonical_bytes(self.static_config)).hexdigest()
        if self.manifest.get("schema") != SCHEMA_VERSION or actual_hash != self.manifest.get("static_config_sha256"):
            raise ValueError("compact collection manifest or static config hash mismatch")

    def iter_records(self) -> Iterator[dict[str, Any]]:
        tables: dict[str, dict[str, Any]] = {}
        observations: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        path = self.collection_dir / "ticks.jsonl"
        previous_time: int | None = None
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                now = record["sim_time_s"]
                if record.get("static_config_sha256") != self.manifest["static_config_sha256"]:
                    raise ValueError("tick static config reference mismatch")
                if previous_time is not None and now != previous_time + 1:
                    raise ValueError("stored public ticks are not contiguous")
                for name, delta in record["tables"].items():
                    tables[name] = _apply_keyed_delta(tables.get(name, {}), delta)
                observations = _apply_keyed_delta(observations, record["observation_delta"])
                for key in record["meta_delete"]:
                    meta.pop(key, None)
                meta.update(record["meta_update"])
                visible_observations = [item for item in observations.values()
                    if _available_at(item) is not None and _available_at(item) <= now]
                projection = {**meta, "sim_time_s": now,
                              "aircraft": tables.get("aircraft", {}),
                              "public_observations": visible_observations}
                for name in ("tasks", "public_tasks"):
                    if name in tables:
                        projection[name] = tables[name]
                yield {"projection": projection, "events": record["events"],
                       "decisions": record["decisions"], "costs": record["costs"],
                       "event_log_complete": record["event_log_complete"]}
                previous_time = now

    def iter_public_states(self) -> Iterator[dict[str, Any]]:
        for record in self.iter_records():
            yield record["projection"]


def _available_at(observation: Mapping[str, Any]) -> float | None:
    value = observation.get("available_at_sim_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def iter_public_one_step_examples(
    reader: CompactPublicReader,
    *,
    history_steps: int,
) -> Iterator[Any]:
    """Adapt consecutive compact public records to wm_ppo_v1 one-step examples."""
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    from wm_ppo_v1.public_dynamics import build_public_one_step_example

    window: list[dict[str, Any]] = []
    events: list[Mapping[str, Any]] = []
    complete_by_time: dict[int, bool] = {}
    for record in reader.iter_records():
        projection = record["projection"]
        now = _time_s(projection)
        window.append(projection)
        events.extend(record["events"])
        complete_by_time[now] = record["event_log_complete"]
        if len(window) < history_steps + 1:
            continue
        window = window[-(history_steps + 1):]
        target_time = now
        start_time = int(window[-2]["sim_time_s"])
        interval_events = []
        interval_command_keys = {
            (str(event.get("aircraft_id")), str(event.get("decision_id")))
            for event in events
            if event.get("event") == "command_applied"
            and isinstance(event.get("sim_time_s"), (int, float))
            and start_time < event["sim_time_s"] <= target_time
        }
        for event in events:
            event_time = event.get("sim_time_s")
            if not isinstance(event_time, (int, float)):
                interval_events.append(event)
            elif start_time < event_time <= target_time:
                interval_events.append(event)
            elif (event.get("event") == "transmission_sent"
                  and (str(event.get("aircraft_id")), str(event.get("decision_id")))
                  in interval_command_keys):
                interval_events.append(event)
        example = build_public_one_step_example(
            window, interval_events,
            events_complete=all(complete_by_time.get(int(t), False)
                                for t in range(int(window[-2]["sim_time_s"]), target_time + 1)),
        )
        yield example
