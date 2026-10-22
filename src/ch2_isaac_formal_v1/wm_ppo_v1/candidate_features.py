"""Public candidate descriptors for actor choice; never a WM action encoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MAX_CANDIDATES = 16
KIND_NAMES = ("noop", "uav_transfer")
NUMERIC_FIELDS = (
    "donor_x_m", "donor_y_m", "donor_z_m", "donor_home_zone_norm", "donor_energy_wh", "donor_queue_length",
    "donor_available", "donor_faulted", "recipient_x_m", "recipient_y_m", "recipient_z_m",
    "recipient_home_zone_norm", "recipient_energy_wh", "recipient_queue_length", "recipient_available", "recipient_faulted",
    "task_queue_position", "estimated_task_eta_s", "estimated_route_distance_km",
    "estimated_energy_wh", "estimated_gain_s", "switch_cost_s",
)
CATALOG_NUMERIC_FIELDS = (
    "estimated_task_eta_s", "estimated_route_distance_km", "estimated_energy_wh",
    "estimated_gain_s", "switch_cost_s",
)
FEATURE_NAMES = (
    "is_noop", "is_uav_transfer", "donor_kind_uav", "donor_kind_helicopter",
    "donor_kind_known", "recipient_kind_uav", "recipient_kind_helicopter", "recipient_kind_known",
    *NUMERIC_FIELDS,
    *(f"{name}_known" for name in NUMERIC_FIELDS),
)


@dataclass(frozen=True)
class CandidateFeatures:
    """Actor input and exact mapping back to the supplied legal catalog."""

    features: np.ndarray
    candidate_mask: np.ndarray
    noop_mask: np.ndarray
    action_keys: tuple[tuple[str, ...], ...]
    actions: tuple[Mapping[str, Any], ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    wm_action_summary: None = None
    wm_action_known: bool = False


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _action_key(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    kind = candidate.get("kind")
    if kind == "noop":
        return ("noop",)
    if kind == "uav_transfer":
        donor, recipient, task_id = (candidate.get(key) for key in ("donor", "recipient", "task_id"))
        if not all(isinstance(value, (str, int)) and not isinstance(value, bool)
                   for value in (donor, recipient, task_id)):
            raise ValueError("uav_transfer candidate requires donor, recipient, and task_id")
        return ("uav_transfer", str(donor), str(recipient), str(task_id))
    raise ValueError(f"unsupported public candidate kind: {kind!r}")


def _actor_values(projection: Mapping[str, Any], actor_id: str, prefix: str,
                  values: dict[str, float], known: dict[str, float]) -> None:
    aircraft = projection.get("aircraft")
    row = aircraft.get(actor_id) if isinstance(aircraft, Mapping) else None
    if not isinstance(row, Mapping):
        return
    kind = row.get("kind")
    if isinstance(kind, str):
        values[f"{prefix}_kind_known"] = 1.0
    if kind == "uav":
        values[f"{prefix}_kind_uav"] = 1.0
    elif kind == "helicopter":
        values[f"{prefix}_kind_helicopter"] = 1.0
    for name, source in (("x_m", "position_m"), ("y_m", "position_m"), ("z_m", "position_m")):
        axis = ("x_m", "y_m", "z_m").index(name)
        position = row.get(source)
        value = _finite(position[axis]) if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) > axis else None
        field = f"{prefix}_{name}"
        if value is not None:
            values[field], known[field] = value, 1.0
    for field, source in (("home_zone_norm", "home_zone"), ("energy_wh", "available_energy_wh"),
                          ("available", "available"), ("faulted", "faulted")):
        raw = row.get(source)
        value = float(raw) if isinstance(raw, bool) and field in ("available", "faulted") else _finite(raw)
        name = f"{prefix}_{field}"
        if value is not None:
            values[name], known[name] = (value / 8.0 if field == "home_zone_norm" else value), 1.0
    queue = row.get("queue_task_ids")
    if isinstance(queue, Sequence) and not isinstance(queue, (str, bytes)):
        field = f"{prefix}_queue_length"
        values[field], known[field] = float(len(queue)), 1.0


def build_candidate_features(
    projection: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]],
) -> CandidateFeatures:
    """Describe supplied legal candidates using only current public projection.

    The catalog is consumed as-is. This function does not enumerate legality,
    invoke a planner, inspect outcomes, or encode identifiers as numbers.
    """
    if not isinstance(projection, Mapping):
        raise ValueError("public projection must be a mapping")
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)) or not catalog:
        raise ValueError("legal candidate catalog must be a non-empty sequence")
    if len(catalog) > MAX_CANDIDATES:
        raise ValueError(f"candidate catalog exceeds the {MAX_CANDIDATES}-candidate limit")
    if any(not isinstance(candidate, Mapping) for candidate in catalog):
        raise ValueError("each legal candidate must be a mapping")
    actions = tuple(dict(candidate) for candidate in catalog)
    keys = tuple(_action_key(candidate) for candidate in actions)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate catalog contains duplicate stable action keys")
    noop_mask = np.asarray([key[0] == "noop" for key in keys], dtype=np.bool_)
    if int(noop_mask.sum()) != 1:
        raise ValueError("legal candidate catalog must contain exactly one NOOP")

    rows = []
    for candidate, key in zip(actions, keys):
        values = {name: 0.0 for name in FEATURE_NAMES}
        known = {name: 0.0 for name in NUMERIC_FIELDS}
        kind = key[0]
        values["is_noop"] = float(kind == "noop")
        values["is_uav_transfer"] = float(kind == "uav_transfer")
        if kind == "uav_transfer":
            donor, recipient, task_id = key[1:]
            _actor_values(projection, donor, "donor", values, known)
            _actor_values(projection, recipient, "recipient", values, known)
            aircraft = projection.get("aircraft")
            donor_row = aircraft.get(donor) if isinstance(aircraft, Mapping) else None
            queue = donor_row.get("queue_task_ids") if isinstance(donor_row, Mapping) else None
            if isinstance(queue, Sequence) and not isinstance(queue, (str, bytes)) and task_id in map(str, queue):
                name = "task_queue_position"
                values[name], known[name] = float(list(map(str, queue)).index(task_id)), 1.0
            for name in CATALOG_NUMERIC_FIELDS:
                value = _finite(candidate.get(name))
                if value is not None:
                    values[name], known[name] = value, 1.0
        row = [values[name] for name in FEATURE_NAMES[:8]]
        row.extend(values[name] for name in NUMERIC_FIELDS)
        row.extend(known[name] for name in NUMERIC_FIELDS)
        rows.append(row)
    return CandidateFeatures(
        features=np.asarray(rows, dtype=np.float32),
        candidate_mask=np.ones(len(rows), dtype=np.bool_),
        noop_mask=noop_mask,
        action_keys=keys,
        actions=actions,
    )
