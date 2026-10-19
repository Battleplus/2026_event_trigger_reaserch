#!/usr/bin/env python3
"""Small headless civil waypoint pilot on the archived 20w Isaac scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_SOURCE_FILES = (
    "scenes/air_combat_scene.py",
    "scenes/air_combat_specs.py",
    "scenes/aircraft_motion.py",
    "sensors/air_combat_specs.py",
    "sensors/air_combat_obstacles.py",
    "weapons/air_combat_specs.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return value


def _make_reference(start: np.ndarray, endpoints: list[np.ndarray], dwell_s: float):
    def reference(tactical_time_s: float) -> np.ndarray:
        index = min(int(max(0.0, tactical_time_s) // dwell_s), len(endpoints) - 1)
        return endpoints[index].copy()

    return reference


def _event_counter():
    import omni.kit.app

    state = {"count": 0}

    def on_update(_event):
        state["count"] += 1

    subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        on_update, name="civil_isaac_pilot_frame_counter"
    )
    return state, subscription


def _usd_world_position(prim, time_code) -> np.ndarray:
    from pxr import UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
    translation = matrix.ExtractTranslation()
    return np.asarray([translation[0], translation[1], translation[2]], dtype=float)


def _restore_initial_state(scene, participants, initial, py_rng_state, np_rng_state):
    from scenes.air_combat_scene import _set_root_pose

    scene.tactical_time_s = 0.0
    scene.contacts = []
    scene.obstacle_contacts = []
    scene.assignments = []
    random.setstate(py_rng_state)
    np.random.set_state(np_rng_state)
    for participant in participants:
        platform = participant["platform"]
        state = platform.motion_model.state
        saved = initial[participant["id"]]
        state.position = saved["position"].copy()
        state.velocity = saved["velocity"].copy()
        state.yaw_deg = saved["yaw_deg"]
        state.pitch_deg = saved["pitch_deg"]
        state.roll_deg = saved["roll_deg"]
        platform.reference_fn = participant["reference"]
        _set_root_pose(platform.root_prim, state.position, state)


def _snapshot_state(participants):
    return {
        item["id"]: {
            "position": item["platform"].motion_model.state.position.copy(),
            "velocity": item["platform"].motion_model.state.velocity.copy(),
            "yaw_deg": float(item["platform"].motion_model.state.yaw_deg),
            "pitch_deg": float(item["platform"].motion_model.state.pitch_deg),
            "roll_deg": float(item["platform"].motion_model.state.roll_deg),
        }
        for item in participants
    }


def _build_waypoints(config, participants, scene):
    units_per_km = 1000.0 / scene.meters_per_unit
    plans = {}
    for item in participants:
        start = item["platform"].position
        endpoints_by_action = {}
        for action_name in ("keep", "replan"):
            offsets = item["waypoints_km"][action_name]
            points = []
            for offset in offsets:
                points.append(np.asarray([
                    start[0] + float(offset[0]) * units_per_km,
                    start[1] + float(offset[1]) * units_per_km,
                    start[2] + float(offset[2]) * units_per_km,
                ], dtype=float))
            endpoints_by_action[action_name] = points
        plans[item["id"]] = endpoints_by_action
    return plans


def _append_branch(
    *, app, event_state, stage, scene, participants, endpoints, action_name,
    branch_name, config, config_sha256, source_hashes, output_handle,
    app_update_calls, start_time_s,
):
    from pxr import Usd, UsdGeom

    dwell_s = float(config["pilot"]["seconds_per_waypoint"])
    dt = float(config["pilot"]["dt_s"])
    steps = int(config["pilot"]["steps_per_waypoint"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    frames_before = int(event_state["count"])
    branch_start = scene.tactical_time_s
    branch_initial = _snapshot_state(participants)
    branch_trace = {item["id"]: [] for item in participants}
    for item in participants:
        item["platform"].reference_fn = _make_reference(
            branch_initial[item["id"]]["position"],
            endpoints[item["id"]][action_name],
            dwell_s,
        )

    for step_index in range(steps * 2):
        next_time = scene.tactical_time_s + dt
        scene.update(tactical_dt=dt, tactical_time_s=next_time)
        before = int(event_state["count"])
        app.update()
        app_update_calls += 1
        after = int(event_state["count"])
        if after <= before:
            raise RuntimeError(
                "SimulationApp.update returned without a Kit update event; "
                "actual app-frame advancement was not verified"
            )

        for item in participants:
            platform = item["platform"]
            state = platform.motion_model.state
            model_position = np.asarray(state.position, dtype=float)
            usd_position = _usd_world_position(
                platform.root_prim, Usd.TimeCode.Default()
            )
            error_units = float(np.linalg.norm(model_position - usd_position))
            if error_units > float(config["validation"]["usd_pose_tolerance_units"]):
                raise RuntimeError(
                    f"USD transform diverged from motion state for {item['id']}: "
                    f"{error_units} scene units"
                )

            local_elapsed = scene.tactical_time_s - branch_start
            waypoint_index = min(
                int(local_elapsed // dwell_s),
                len(endpoints[item["id"]][action_name]) - 1,
            )
            target = endpoints[item["id"]][action_name][waypoint_index]
            row = {
                "schema_version": 1,
                "record_type": "step",
                "run_id": config["run_id"],
                "seed": int(config["seed"]),
                "branch": branch_name,
                "action_name": action_name,
                "platform_id": item["id"],
                "task": {
                    "task_id": "civil_inspection_waypoint_pilot",
                    "task_type": "waypoint_following",
                    "task_status": "pilot",
                },
                "action": {
                    "action_id": f"{action_name}_wp_{waypoint_index + 1}",
                    "waypoint_index": waypoint_index,
                    "target_position_scene_units": target.tolist(),
                },
                "frame": {
                    "app_update_call": app_update_calls,
                    "kit_update_event_count": after,
                    "kit_events_this_call": after - before,
                },
                "scene_time_s": float(scene.tactical_time_s),
                "dt_s": dt,
                "units": {
                    "position": "USD scene units",
                    "velocity": "USD scene units/s",
                    "meters_per_unit": meters_per_unit,
                },
                "pose": {
                    "motion_state_position": model_position.tolist(),
                    "usd_world_position": usd_position.tolist(),
                    "yaw_deg": float(state.yaw_deg),
                    "pitch_deg": float(state.pitch_deg),
                    "roll_deg": float(state.roll_deg),
                    "usd_position_error_units": error_units,
                },
                "velocity_scene_units_s": np.asarray(state.velocity, dtype=float).tolist(),
                "energy": {"status": "not_modeled", "value": None},
                "communication": {"status": "not_modeled", "value": None},
                "config_sha256": config_sha256,
                "source_archive_sha256": str(config["source_archive_sha256"]),
                "source_hashes_sha256": source_hashes,
            }
            output_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            branch_trace[item["id"]].append({
                "motion_position": model_position.copy(),
                "usd_position": usd_position.copy(),
                "velocity": np.asarray(state.velocity, dtype=float).copy(),
            })

    final_state = _snapshot_state(participants)
    branch_frames = int(event_state["count"]) - frames_before
    return final_state, branch_trace, app_update_calls, branch_frames


def run(config_path: Path, output_path: Path, source_root: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    config_sha256 = _sha256(config_path)
    source_root = source_root.resolve()
    for relative in REQUIRED_SOURCE_FILES:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(f"required scene source missing: {source_root / relative}")
    source_hashes = {
        relative: _sha256(source_root / relative)
        for relative in REQUIRED_SOURCE_FILES
    }
    archive_sha256 = str(config["source_archive_sha256"])
    if len(archive_sha256) != 64:
        raise ValueError("source_archive_sha256 must be a 64-character SHA-256")

    sys.path.insert(0, str(source_root))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    gpu_index = int(config["runtime"]["gpu_index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "runtime_initialization",
        "actual_isaac_run": False,
        "started_unix_s": time.time(),
        "requested_gpu_index": gpu_index,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source_archive_sha256": archive_sha256,
        "scene_source_hashes_sha256": source_hashes,
        "energy": "not_modeled",
        "communication": "not_modeled",
        "task_scale_claim": {"participants": len(config["participants"]), "grid_cells": 0, "regions": 0, "tasks": 0},
        "validation": {},
    }
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True,
            "renderer": "HydraStorm",
            "width": 640,
            "height": 480,
            "active_gpu": gpu_index,
            "physics_gpu": gpu_index,
            "multi_gpu": False,
            "max_gpu_count": 1,
        })
        manifest["status"] = "running"
    except Exception as exc:
        manifest.update({
            "status": "runtime_initialization_failed",
            "ended_unix_s": time.time(),
            "first_error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        from scenes import air_combat_scene

        stage = omni.usd.get_context().get_stage()
        if not stage:
            stage = Usd.Stage.CreateInMemory()
            omni.usd.get_context().set_stage(stage)

        scene_config = copy.deepcopy(config["scene_config"])
        scene_config.setdefault("simulation", {})["time_scale"] = 1.0
        scene_config["simulation"]["aircraft_model"] = "kinematic"
        scene = air_combat_scene.create_scene(stage, scene_config)
        actual_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        expected_meters_per_unit = float(config["scene_config"]["map"]["meters_per_unit"])
        if not math.isclose(actual_meters_per_unit, expected_meters_per_unit, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"stage metersPerUnit {actual_meters_per_unit} != configured "
                f"{expected_meters_per_unit}"
            )

        requested = config["participants"]
        entity_ids = {str(item["scene_entity_id"]) for item in requested}
        known = {platform.entity_id: platform for platform in scene.platforms}
        missing = entity_ids - set(known)
        if missing:
            raise KeyError(f"scene platform ids missing: {sorted(missing)}")
        participants = []
        for item in requested:
            participants.append({
                "id": str(item["id"]),
                "platform": known[str(item["scene_entity_id"])],
                "waypoints_km": item["waypoints_km"],
                "reference": known[str(item["scene_entity_id"])].reference_fn,
            })

        for platform in scene.platforms:
            if platform.entity_id not in entity_ids:
                stage.RemovePrim(platform.root_prim.GetPath())
        scene.platforms = [item["platform"] for item in participants]
        for target in scene.targets:
            stage.RemovePrim(target.root_prim.GetPath())
        scene.targets = []
        scene.contacts = []
        scene.obstacle_contacts = []
        scene.assignments = []

        endpoints = _build_waypoints(config, participants, scene)
        py_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        initial_state = _snapshot_state(participants)
        event_state, frame_subscription = _event_counter()
        tolerance = float(config["validation"]["usd_pose_tolerance_units"])
        for item in participants:
            usd_position = _usd_world_position(item["platform"].root_prim, Usd.TimeCode.Default())
            error = float(np.linalg.norm(usd_position - initial_state[item["id"]]["position"]))
            if error > tolerance:
                raise RuntimeError(f"initial USD pose mismatch for {item['id']}: {error} units")

        manifest.update({
            "actual_isaac_run": True,
            "isaac_scene_version": air_combat_scene.AIR_COMBAT_SCENE_VERSION,
            "map_size_km": float(scene.map_size_km),
            "map_size_units": float(scene.map_size_units),
            "meters_per_unit_configured": expected_meters_per_unit,
            "meters_per_unit_stage": actual_meters_per_unit,
            "simulation_time_scale": float(scene.config["simulation"]["time_scale"]),
            "motion_backend": "repository kinematic model advanced by scene.update; USD pose verified against motion state",
            "participant_count": len(participants),
            "participating_scene_entity_ids": [item["platform"].entity_id for item in participants],
            "scene_target_count_after_civil_filter": len(scene.targets),
            "platforms_spawned_in_source_scene": len(known),
            "kit_update_event_count_initial": int(event_state["count"]),
        })

        app_update_calls = 0
        branch_results = {}
        with output_path.open("w", encoding="utf-8", newline="\n") as output_handle:
            for branch_name, action_name in (("keep", "keep"), ("replay_keep", "keep"), ("replan", "replan")):
                _restore_initial_state(scene, participants, initial_state, py_rng_state, np_rng_state)
                before_initial = _snapshot_state(participants)
                restore_error = max(
                    float(np.linalg.norm(before_initial[item["id"]]["position"] - initial_state[item["id"]]["position"]))
                    for item in participants
                )
                if restore_error > tolerance:
                    raise RuntimeError(f"state restore mismatch: {restore_error} units")
                result, trace, app_update_calls, branch_frames = _append_branch(
                    app=app,
                    event_state=event_state,
                    stage=stage,
                    scene=scene,
                    participants=participants,
                    endpoints=endpoints,
                    action_name=action_name,
                    branch_name=branch_name,
                    config=config,
                    config_sha256=config_sha256,
                    source_hashes=source_hashes,
                    output_handle=output_handle,
                    app_update_calls=app_update_calls,
                    start_time_s=0.0,
                )
                branch_results[branch_name] = {"final": result, "trace": trace}
                manifest.setdefault("branch_frames", {})[branch_name] = branch_frames

        displacements = {
            item["id"]: float(np.linalg.norm(
                branch_results["keep"]["final"][item["id"]]["position"] - initial_state[item["id"]]["position"]
            ))
            for item in participants
        }
        replay_errors = {
            item["id"]: {
                "max_step_position_units": max(
                    float(np.linalg.norm(a["motion_position"] - b["motion_position"]))
                    for a, b in zip(
                        branch_results["keep"]["trace"][item["id"]],
                        branch_results["replay_keep"]["trace"][item["id"]],
                    )
                ),
                "max_step_usd_position_units": max(
                    float(np.linalg.norm(a["usd_position"] - b["usd_position"]))
                    for a, b in zip(
                        branch_results["keep"]["trace"][item["id"]],
                        branch_results["replay_keep"]["trace"][item["id"]],
                    )
                ),
                "max_step_velocity_units_s": max(
                    float(np.linalg.norm(a["velocity"] - b["velocity"]))
                    for a, b in zip(
                        branch_results["keep"]["trace"][item["id"]],
                        branch_results["replay_keep"]["trace"][item["id"]],
                    )
                ),
                "compared_steps": min(
                    len(branch_results["keep"]["trace"][item["id"]]),
                    len(branch_results["replay_keep"]["trace"][item["id"]]),
                ),
            }
            for item in participants
        }
        action_differences = {
            item["id"]: float(np.linalg.norm(
                branch_results["keep"]["final"][item["id"]]["position"] - branch_results["replan"]["final"][item["id"]]["position"]
            ))
            for item in participants
        }
        min_displacement = float(config["validation"]["min_displacement_m"])
        min_action_delta = float(config["validation"]["min_action_delta_m"])
        max_replay_error_units = float(config["validation"]["max_replay_error_units"])
        max_replay_error = max(
            max(v[key] for key in (
                "max_step_position_units",
                "max_step_usd_position_units",
                "max_step_velocity_units_s",
            ))
            for v in replay_errors.values()
        )
        meter_scale = actual_meters_per_unit
        checks = {
            "app_frames_advanced": int(event_state["count"]) > 0 and app_update_calls > 0,
            "usd_pose_matches_motion_state_each_step": True,
            "nonzero_movement": all(v * meter_scale >= min_displacement for v in displacements.values()),
            "same_input_replay_consistent": (
                all(v["compared_steps"] == int(config["pilot"]["steps_per_waypoint"]) * 2 for v in replay_errors.values())
                and max_replay_error <= max_replay_error_units
            ),
            "action_outcomes_differ": all(v * meter_scale >= min_action_delta for v in action_differences.values()),
            "stage_units_match_frozen_config": math.isclose(actual_meters_per_unit, expected_meters_per_unit),
        }
        manifest.update({
            "status": "complete" if all(checks.values()) else "validation_failed",
            "ended_unix_s": time.time(),
            "app_update_calls": app_update_calls,
            "kit_update_event_count_final": int(event_state["count"]),
            "validation": {
                "checks": checks,
                "displacement_m": {key: value * meter_scale for key, value in displacements.items()},
                "keep_vs_replay_error": replay_errors,
                "keep_vs_replan_delta_m": {key: value * meter_scale for key, value in action_differences.items()},
                "thresholds": {
                    "min_displacement_m": min_displacement,
                    "max_replay_error_units": max_replay_error_units,
                    "min_action_delta_m": min_action_delta,
                    "usd_pose_tolerance_units": tolerance,
                },
            },
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if manifest["status"] != "complete":
            raise RuntimeError(f"pilot validation failed; see {manifest_path}")
        return manifest
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "ended_unix_s": time.time(),
            "first_error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    finally:
        subscription = locals().get("frame_subscription")
        if subscription is not None:
            subscription.unsubscribe()
        app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_root = args.source_root or project_root / "source" / "54_20-master"
    run(args.config.resolve(), args.output.resolve(), source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
