"""Portable, public-prefix-bound snapshots for paired N2 branches.

Snapshots are explicit JSON envelopes. Python pickle and implicit reset-from-
scenario are intentionally rejected. The adapter is designed for the existing
ReliefEnv state shape, but it requires the caller to provide a fresh environment
factory so the server can prove sibling isolation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


SCHEMA = "n2-public-bound-snapshot/1"
STATE_FIELDS = (
    "scenario", "initial_handoff", "_platform_spec", "_depot_spec", "_task_spec",
    "_original_owner", "_step_api_active", "_terminal", "_trajectory_cache", "time",
    "events", "transitions", "_event_seq", "_policy", "_platforms", "_depots",
    "_tasks", "_cargo", "_queues", "_scheduled_reports", "_public_platforms",
    "_public_depots", "_public_tasks", "_rewards_paid",
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _encode(value: Any) -> Any:
    if isinstance(value, defaultdict):
        return {"__type__": "defaultdict", "items": _encode(dict(value))}
    if isinstance(value, dict):
        return {"__type__": "dict", "items": [[_encode(key), _encode(item)] for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, set):
        return {"__type__": "set", "items": [_encode(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"SNAPSHOT_UNSUPPORTED_TYPE:{type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__type__")
    if marker == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if marker == "defaultdict":
        return defaultdict(float, _decode(value["items"]))
    if marker == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if marker == "set":
        return set(_decode(item) for item in value["items"])
    raise ValueError("SNAPSHOT_UNKNOWN_MARKER")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _public_prefix(env: Any) -> dict[str, Any]:
    observation = env.public_observation()
    if not isinstance(observation, dict):
        raise ValueError("SNAPSHOT_PUBLIC_OBSERVATION")
    return copy.deepcopy(observation)


def _catalog(env: Any) -> list[dict[str, Any]]:
    candidates = env.candidates()
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("SNAPSHOT_EMPTY_CATALOG")
    ids = [candidate.get("id") for candidate in candidates]
    if ids.count("NOOP") != 1 or any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("SNAPSHOT_CATALOG_IDENTITY")
    return copy.deepcopy(candidates)


@dataclass(frozen=True)
class SnapshotEnvelope:
    payload: dict[str, Any]

    def dumps(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"

    @classmethod
    def loads(cls, value: str) -> "SnapshotEnvelope":
        if not isinstance(value, str) or "pickle" in value.lower():
            raise ValueError("SNAPSHOT_SERIALIZATION")
        payload = json.loads(value)
        if payload.get("schema") != SCHEMA:
            raise ValueError("SNAPSHOT_SCHEMA")
        return cls(payload)


def capture(env: Any, *, parent_id: str, repeat: int, window_id: str, event_tape_id: str) -> SnapshotEnvelope:
    if not isinstance(parent_id, str) or not parent_id or type(repeat) is not int or repeat < 0 or not isinstance(window_id, str) or not window_id:
        raise ValueError("SNAPSHOT_IDENTITY")
    public_prefix = _public_prefix(env)
    catalog = _catalog(env)
    state = {}
    for field in STATE_FIELDS:
        if not hasattr(env, field):
            raise ValueError(f"SNAPSHOT_STATE_FIELD_MISSING:{field}")
        state[field] = _encode(copy.deepcopy(getattr(env, field)))
    payload = {
        "schema": SCHEMA,
        "parent_id": parent_id,
        "repeat": repeat,
        "window_id": window_id,
        "event_tape_id": event_tape_id,
        "scenario_sha256": _digest(env.scenario),
        "event_tape_sha256": _digest(env.scenario.get("disturbances", [])),
        "time": int(env.time),
        "public_prefix": public_prefix,
        "public_prefix_sha256": _digest(public_prefix),
        "candidate_catalog": catalog,
        "candidate_catalog_sha256": _digest(catalog),
        "state": state,
    }
    payload["state_sha256"] = _digest(state)
    payload["envelope_sha256"] = _digest({key: value for key, value in payload.items() if key != "envelope_sha256"})
    return SnapshotEnvelope(payload)


def restore(envelope: SnapshotEnvelope | str, env_factory: Callable[[dict[str, Any]], Any]) -> Any:
    if isinstance(envelope, str):
        envelope = SnapshotEnvelope.loads(envelope)
    payload = envelope.payload
    if payload.get("schema") != SCHEMA:
        raise ValueError("SNAPSHOT_SCHEMA")
    if payload.get("state_sha256") != _digest(payload.get("state")):
        raise ValueError("SNAPSHOT_STATE_DIGEST")
    if payload.get("envelope_sha256") != _digest({key: value for key, value in payload.items() if key != "envelope_sha256"}):
        raise ValueError("SNAPSHOT_ENVELOPE_DIGEST")
    state = {field: _decode(value) for field, value in payload["state"].items()}
    if set(state) != set(STATE_FIELDS):
        raise ValueError("SNAPSHOT_STATE_FIELDS")
    if payload.get("scenario_sha256") != _digest(state["scenario"]):
        raise ValueError("SNAPSHOT_SCENARIO_IDENTITY")
    if payload.get("event_tape_sha256") != _digest(state["scenario"].get("disturbances", [])):
        raise ValueError("SNAPSHOT_EVENT_TAPE_IDENTITY")
    env = env_factory(copy.deepcopy(state["scenario"]))
    for field in STATE_FIELDS:
        setattr(env, field, copy.deepcopy(state[field]))
    public_prefix = _public_prefix(env)
    catalog = _catalog(env)
    if _digest(public_prefix) != payload["public_prefix_sha256"] or public_prefix != payload["public_prefix"]:
        raise ValueError("SNAPSHOT_PUBLIC_PREFIX_MISMATCH")
    if _digest(catalog) != payload["candidate_catalog_sha256"] or catalog != payload["candidate_catalog"]:
        raise ValueError("SNAPSHOT_CATALOG_MISMATCH")
    return env


def assert_sibling_isolation(first: Any, second: Any) -> None:
    if first is second:
        raise ValueError("SNAPSHOT_SIBLING_ALIAS")
    if getattr(first, "events", None) is getattr(second, "events", None) or getattr(first, "transitions", None) is getattr(second, "transitions", None):
        raise ValueError("SNAPSHOT_MUTABLE_ALIAS")
