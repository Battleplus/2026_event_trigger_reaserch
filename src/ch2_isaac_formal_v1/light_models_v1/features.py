"""Fixed, outcome-independent features from a public state projection."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping


KINDS = ("uav", "helicopter")
ZONES = tuple(range(9))
OBSERVATION_KINDS = ("uav", "helicopter", "other")
DEFAULT_PLATFORM_PARAMS = {
    "uav": {"energy_capacity_wh": 180000.0, "energy_wh_per_km": 240.0, "max_speed_mps": 65.0},
    "helicopter": {"energy_capacity_wh": 300000.0, "energy_wh_per_km": 900.0, "max_speed_mps": 100.0},
}


_BASE_NAMES = (
    "sim_time_s", "aircraft_count", "active_task_count", "queue_task_count",
    "available_fraction", "fault_fraction", "safety_paused_fraction",
    "communication_held_fraction", "energy_norm_mean", "energy_norm_min",
    "energy_norm_std", "position_rms_xy_m", "velocity_mean_mps", "velocity_rms_mps",
    "observation_count", "observation_age_mean_s", "observation_age_max_s",
    "observation_available_lag_mean_s", "observation_xy_rms_km",
)
_GROUP_NAMES = (
    "count", "available_fraction", "fault_fraction", "safety_paused_fraction",
    "communication_held_fraction", "energy_norm_mean", "energy_norm_min",
    "energy_norm_std", "energy_rate_wh_per_km", "max_speed_mps", "position_rms_xy_m",
)
_ZONE_NAMES = ("count", "available_fraction", "fault_fraction", "energy_norm_mean")
_OBS_NAMES = ("count", "age_mean_s", "age_max_s")

FEATURE_SCHEMA = list(_BASE_NAMES)
for _kind in KINDS:
    FEATURE_SCHEMA.extend(f"kind_{_kind}_{name}" for name in _GROUP_NAMES)
for _zone in ZONES:
    FEATURE_SCHEMA.extend(f"zone_{_zone}_{name}" for name in _ZONE_NAMES)
for _kind in OBSERVATION_KINDS:
    FEATURE_SCHEMA.extend(f"observation_kind_{_kind}_{name}" for name in _OBS_NAMES)
for _zone in ZONES:
    FEATURE_SCHEMA.extend(f"observation_zone_{_zone}_{name}" for name in _OBS_NAMES)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _fraction(values: list[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rms_about_centroid(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    cx = _mean([point[0] for point in points])
    cy = _mean([point[1] for point in points])
    return math.sqrt(_mean([(x - cx) ** 2 + (y - cy) ** 2 for x, y in points]))


def _platform_params(platform_params: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, float]]:
    result = {kind: dict(values) for kind, values in DEFAULT_PLATFORM_PARAMS.items()}
    for kind, values in (platform_params or {}).items():
        if kind in result and isinstance(values, Mapping):
            for key in result[kind]:
                result[kind][key] = _finite(values.get(key), result[kind][key])
    return result


def _platform_group(rows: list[dict[str, Any]], params: dict[str, float]) -> list[float]:
    energies = [_finite(row.get("available_energy_wh")) / max(params["energy_capacity_wh"], 1e-12)
                for row in rows]
    positions = []
    for row in rows:
        point = row.get("position_m", [])
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            positions.append((_finite(point[0]), _finite(point[1])))
    return [
        float(len(rows)), _fraction([bool(row.get("available", False)) for row in rows]),
        _fraction([bool(row.get("faulted", False)) for row in rows]),
        _fraction([bool(row.get("safety_paused", False)) for row in rows]),
        _fraction([bool(row.get("communication_held", False)) for row in rows]),
        _mean(energies), min(energies) if energies else 0.0, _std(energies),
        params["energy_wh_per_km"], params["max_speed_mps"], _rms_about_centroid(positions),
    ]


def _zone_group(rows: list[dict[str, Any]], params: dict[str, dict[str, float]]) -> list[float]:
    energies = []
    for row in rows:
        kind = row.get("kind") if row.get("kind") in params else "uav"
        capacity = max(params[kind]["energy_capacity_wh"], 1e-12)
        energies.append(_finite(row.get("available_energy_wh")) / capacity)
    return [
        float(len(rows)), _fraction([bool(row.get("available", False)) for row in rows]),
        _fraction([bool(row.get("faulted", False)) for row in rows]), _mean(energies),
    ]


def extract_features(projection: Mapping[str, Any],
                     platform_params: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, float]:
    """Extract the fixed schema from one public projection in O(number of rows)."""
    if not isinstance(projection, Mapping):
        raise TypeError("projection must be a mapping")
    params = _platform_params(platform_params)
    aircraft = projection.get("aircraft", {})
    if not isinstance(aircraft, Mapping):
        raise ValueError("projection.aircraft must be a mapping")
    platforms = [dict(value) for value in aircraft.values() if isinstance(value, Mapping)]
    observations = [dict(value) for value in projection.get("public_observations", [])
                    if isinstance(value, Mapping)]
    sim_time = _finite(projection.get("sim_time_s"))
    active_count = sum(len(row.get("active_task_ids", ())) for row in platforms)
    queue_count = sum(len(row.get("queue_task_ids", ())) for row in platforms)
    energies = []
    positions = []
    velocities = []
    for row in platforms:
        kind = row.get("kind") if row.get("kind") in params else "uav"
        cap = max(params[kind]["energy_capacity_wh"], 1e-12)
        energies.append(_finite(row.get("available_energy_wh")) / cap)
        point = row.get("position_m", [])
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            positions.append((_finite(point[0]), _finite(point[1])))
        velocity = row.get("velocity_mps", [])
        if isinstance(velocity, (list, tuple)):
            speed = math.sqrt(sum(_finite(value) ** 2 for value in velocity[:3]))
            velocities.append(speed)
    ages = [max(0.0, sim_time - _finite(row.get("observed_at_sim_time_s")))
            for row in observations]
    available_lags = [max(0.0, _finite(row.get("available_at_sim_time_s")) -
                          _finite(row.get("observed_at_sim_time_s"))) for row in observations]
    observed_points = []
    for row in observations:
        point = row.get("target_xy_km", [])
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            observed_points.append((_finite(point[0]), _finite(point[1])))

    values: list[float] = [
        sim_time, float(len(platforms)), float(active_count), float(queue_count),
        _fraction([bool(row.get("available", False)) for row in platforms]),
        _fraction([bool(row.get("faulted", False)) for row in platforms]),
        _fraction([bool(row.get("safety_paused", False)) for row in platforms]),
        _fraction([bool(row.get("communication_held", False)) for row in platforms]),
        _mean(energies), min(energies) if energies else 0.0, _std(energies),
        _rms_about_centroid(positions), _mean(velocities),
        math.sqrt(_mean([speed * speed for speed in velocities])) if velocities else 0.0,
        float(len(observations)), _mean(ages), max(ages) if ages else 0.0,
        _mean(available_lags), _rms_about_centroid(observed_points),
    ]
    for kind in KINDS:
        values.extend(_platform_group([row for row in platforms if row.get("kind") == kind], params[kind]))
    for zone in ZONES:
        group = [row for row in platforms if int(_finite(row.get("home_zone"), -1)) == zone]
        values.extend(_zone_group(group, params))
    for kind in OBSERVATION_KINDS:
        group = [row for row in observations
                 if (row.get("kind") if row.get("kind") in KINDS else "other") == kind]
        group_ages = [max(0.0, sim_time - _finite(row.get("observed_at_sim_time_s"))) for row in group]
        values.extend([float(len(group)), _mean(group_ages), max(group_ages) if group_ages else 0.0])
    for zone in ZONES:
        group = [row for row in observations if int(_finite(row.get("zone"), -1)) == zone]
        group_ages = [max(0.0, sim_time - _finite(row.get("observed_at_sim_time_s"))) for row in group]
        values.extend([float(len(group)), _mean(group_ages), max(group_ages) if group_ages else 0.0])
    if len(values) != len(FEATURE_SCHEMA):
        raise AssertionError(f"feature schema/value mismatch: {len(values)} != {len(FEATURE_SCHEMA)}")
    return {name: float(value) for name, value in zip(FEATURE_SCHEMA, values)}
