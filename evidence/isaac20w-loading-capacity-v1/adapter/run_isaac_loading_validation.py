#!/usr/bin/env python3
"""Headless Isaac 4.5 loading/capacity validation; no combat demo or PhysX claim."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


def read_gpu_rows():
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
        rows = []
        for line in output.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 6:
                rows.append({
                    "index": int(fields[0]), "uuid": fields[1], "pci_bus_id": fields[2],
                    "name": fields[3], "memory_used_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                })
        return rows
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def process_rss_mib():
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def event_counter():
    import omni.kit.app

    state = {"count": 0}
    sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        lambda _event: state.__setitem__("count", state["count"] + 1),
        name="isaac_loading_validation_frame_counter",
    )
    return state, sub


def persist_report(path: Path, report: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = {
        "run_id": config["run_id"], "status": "initializing",
        "started_unix_s": time.time(), "actual_isaac_run": False,
        "requested_devices": config["runtime"], "stages": [], "first_error": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = None
    subscription = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp

        runtime = config["runtime"]
        app = SimulationApp({
            "headless": True,
            "renderer": runtime["renderer"],
            "width": runtime["width"], "height": runtime["height"],
            "active_gpu": runtime["renderer_active_gpu"],
            "physics_gpu": runtime["physics_cuda_device"],
            "multi_gpu": False, "max_gpu_count": 1,
        })
        report["status"] = "empty_startup_ok"
        report["actual_isaac_run"] = True
        report["gpu_at_startup"] = read_gpu_rows()
        report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            import carb
            settings = carb.settings.get_settings()
            report["kit_gpu_settings"] = {
                "renderer_active_gpu": settings.get("/renderer/activeGpu"),
                "physics_cuda_device": settings.get("/physics/cudaDevice"),
            }
        except Exception as exc:
            report["kit_gpu_settings_error"] = f"{type(exc).__name__}: {exc}"
        expected_uuid = runtime["expected_gpu_uuid"]
        startup_rows = report["gpu_at_startup"]
        if isinstance(startup_rows, list):
            selected = [row for row in startup_rows if row["uuid"] == expected_uuid]
            report["selected_gpu_startup"] = selected[0] if selected else None
            if not selected:
                raise RuntimeError(f"expected GPU UUID not found in nvidia-smi inventory: {expected_uuid}")
        event_state, subscription = event_counter()
        t0 = time.perf_counter()
        app.update()
        startup_wall = time.perf_counter() - t0
        if event_state["count"] < 1:
            raise RuntimeError("empty startup update did not produce a Kit update event")
        report["stages"].append({
            "name": "empty_headless_startup", "app_update_calls": 1,
            "kit_update_events": event_state["count"], "app_update_wall_s": startup_wall,
            "process_rss_mib": process_rss_mib(), "gpu_snapshot": read_gpu_rows(),
            "gpu_frame_time_ms": None,
            "gpu_frame_time_status": "not directly exposed by the HydraStorm/Kit update API",
        })
        report["status"] = "complete" if args.startup_only else "terrain_initialization"
        persist_report(args.output, report)
        if args.startup_only:
            return 0

        import omni.usd
        from pxr import Usd, UsdGeom

        import sys
        sys.path.insert(0, str(args.source_root.resolve()))
        from scenes import air_combat_scene as scene_api
        from usd_compat import patch_scene_transforms

        patch_scene_transforms(scene_api)

        stage = omni.usd.get_context().get_stage()
        if not stage:
            stage = Usd.Stage.CreateInMemory()
            omni.usd.get_context().set_stage(stage)
        scene_cfg = {
            "map": {
                "real_size_km": config["scene"]["map_size_km"],
                "meters_per_unit": config["scene"]["meters_per_unit"],
                "mountain_height_m": config["scene"]["mountain_height_m"],
                "terrain_vertical_exaggeration": config["scene"]["terrain_vertical_exaggeration"],
                "terrain_source": "procedural_grid", "terrain_usd": None,
                "grid": config["scene"]["terrain_grid"],
            },
            "visual": {
                "aircraft_scale": config["scene"]["aircraft_visual_scale"],
                "show_sensor_rings": False, "show_sensor_cones": False,
                "show_route_lines": False, "show_terrain_grid": False,
                "show_terrain_wireframe": False, "show_contours": False,
                "show_collision_body": False,
            },
            "weather": {"clouds": False, "local_zones": False, "rain": False},
            "bases": {"enabled": False},
            "simulation": {"time_scale": 1.0, "aircraft_model": "kinematic"},
            "targets": {"radar_sites": 0, "command_posts": 0, "armored_vehicles": 0, "forward_bases": False},
            "obstacles": {
                "enabled": True, "mountains": True, "forests": True, "rock_fields": True,
                "forest_tree_count": config["scene"]["forest_tree_count"],
                "forest_tree_visual_scale": config["scene"]["forest_tree_visual_scale"],
            },
        }
        terrain_stage_started = time.perf_counter()
        scene = scene_api.create_scene(stage, scene_cfg)
        map_units = scene.map_size_units
        factory_platform_count = len(scene.platforms)
        original_quad_positions = {
            platform.entity_id: platform.motion_model.state.position.copy()
            for platform in scene.platforms
            if platform.entity_id in {"Blue_Quad_Recon_1", "Blue_Quad_Recon_2"}
        }
        for platform in list(scene.platforms):
            stage.RemovePrim(platform.root_prim.GetPath())
        scene.platforms = []
        tree_trunks = [
            p for p in stage.Traverse()
            if "/ObstacleOverlays/" in str(p.GetPath())
            and "_Tree_" in p.GetName() and p.GetName().endswith("_Trunk")
        ]
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        terrain = stage.GetPrimAtPath("/World/AirCombat/Terrain")
        terrain_count = int(bool(terrain and terrain.IsValid() and terrain.IsA(UsdGeom.Mesh)))
        report["stages"].append({
            "name": "20w_terrain_140_trees_no_combat", "map_size_km": config["scene"]["map_size_km"],
            "meters_per_unit": meters_per_unit, "terrain_mesh_count": terrain_count,
            "tree_count_from_usd_trunk_prims": len(tree_trunks),
            "combat_platform_count": len(scene.platforms), "factory_platforms_pruned": factory_platform_count,
            "combat_target_count": len(scene.targets),
            "stage_prim_count": sum(1 for _ in stage.Traverse()),
            "stage_creation_wall_s": time.perf_counter() - terrain_stage_started,
        })
        report["status"] = "terrain_stage_complete"
        persist_report(args.output, report)
        if len(tree_trunks) != config["scene"]["forest_tree_count"]:
            raise RuntimeError(f"expected {config['scene']['forest_tree_count']} tree USD trunks, found {len(tree_trunks)}")
        if terrain_count < 1:
            raise RuntimeError("procedural terrain mesh was not found in USD stage")

        from scenes.air_combat_specs import QUAD_RECON_UAV
        from scenes.aircraft_motion import build_motion_model
        import numpy as np

        if len(original_quad_positions) != 2:
            raise RuntimeError(f"expected original positions for two quadrotor types, got {len(original_quad_positions)}")

        def make_platform(entity_id, idx, position):
            root, rotors = scene_api._spawn_air_platform(
                stage, f"/World/AirCombat/Platforms/{entity_id}", QUAD_RECON_UAV, position,
                meters_per_unit, float(config["scene"]["aircraft_visual_scale"]), idx, False,
            )
            model = build_motion_model(
                scene_api._aircraft_motion_config(QUAD_RECON_UAV, "kinematic", meters_per_unit),
                position, domain="air",
            )
            return scene_api.AirPlatform(
                entity_id=entity_id, spec=QUAD_RECON_UAV, root_prim=root,
                motion_model=model,
                reference_fn=lambda _t, p=position.copy(): p + np.asarray([10.0, 0.0, 0.0]),
                rotor_prims=rotors,
            )

        for count in config["platform_stages"]["counts"]:
            stage_started = time.perf_counter()
            if count == 2:
                platforms = [
                    make_platform(entity_id, idx, original_quad_positions[entity_id])
                    for idx, entity_id in enumerate(("Blue_Quad_Recon_1", "Blue_Quad_Recon_2"))
                ]
                scene.platforms = platforms
            elif count == 30:
                for platform in list(scene.platforms):
                    stage.RemovePrim(platform.root_prim.GetPath())
                platforms = []
                for idx in range(count):
                    x = -map_units * 0.30 + idx * map_units * 0.02
                    y = -map_units * 0.28 + (idx % 5) * map_units * 0.015
                    position = np.asarray([x, y, scene.surface_height(x, y) + 12.0], dtype=float)
                    platforms.append(make_platform(f"Capacity_Quad_{idx:02d}", idx, position))
                scene.platforms = platforms
            else:
                raise ValueError(f"unsupported platform validation count: {count}")
            start_events = event_state["count"]
            wall_times = []
            scene_update_times = []
            gpu_peak_by_uuid = {}
            rss_peak = process_rss_mib() or 0.0
            frames = int(config["platform_stages"]["frames_each"])
            dt = float(config["platform_stages"]["dt_s"])
            initial_positions = {
                platform.entity_id: [float(value) for value in platform.motion_model.state.position]
                for platform in platforms
            }
            max_pose_error = 0.0
            for frame in range(frames):
                scene_t0 = time.perf_counter()
                scene.update(tactical_dt=dt, tactical_time_s=scene.tactical_time_s + dt)
                scene_update_times.append(time.perf_counter() - scene_t0)
                before = event_state["count"]
                t0 = time.perf_counter()
                app.update()
                wall_times.append(time.perf_counter() - t0)
                if event_state["count"] <= before:
                    raise RuntimeError(f"Kit frame did not advance at {count}-platform frame {frame + 1}")
                for platform in platforms:
                    from pxr import Usd
                    transform = UsdGeom.Xformable(platform.root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    actual_pos = transform.ExtractTranslation()
                    model_pos = platform.motion_model.state.position
                    error = math.dist(
                        [float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])],
                        [float(value) for value in model_pos],
                    )
                    max_pose_error = max(max_pose_error, error)
                    if error > 1e-6:
                        raise RuntimeError(f"USD pose mismatch during {count}-platform frame {frame + 1}: {platform.entity_id} {error}")
                rss_peak = max(rss_peak, process_rss_mib() or 0.0)
                if frame % 10 == 0 or frame == frames - 1:
                    snapshot = read_gpu_rows()
                    if isinstance(snapshot, list):
                        for row in snapshot:
                            key = row["uuid"]
                            peak = gpu_peak_by_uuid.setdefault(key, {"memory_used_mib": 0, "utilization_percent": 0})
                            peak["memory_used_mib"] = max(peak["memory_used_mib"], row["memory_used_mib"])
                            peak["utilization_percent"] = max(peak["utilization_percent"], row["utilization_percent"])
            positions = []
            if event_state["count"] - start_events != frames:
                raise RuntimeError(
                    f"expected {frames} Kit events for {count} app.update calls, "
                    f"observed {event_state['count'] - start_events}"
                )
            for platform in platforms:
                xform = UsdGeom.Xformable(platform.root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                pos = xform.ExtractTranslation()
                actual = [float(pos[0]), float(pos[1]), float(pos[2])]
                expected = [float(x) for x in platform.motion_model.state.position]
                err = math.dist(actual, expected)
                max_pose_error = max(max_pose_error, err)
                if err > 1e-6:
                    raise RuntimeError(f"platform USD pose mismatch for {platform.entity_id}: {err}")
                delta = [actual[i] - initial_positions[platform.entity_id][i] for i in range(3)]
                positions.append({
                    "entity_id": platform.entity_id,
                    "prim_path": str(platform.root_prim.GetPath()),
                    "initial_position_scene_units": initial_positions[platform.entity_id],
                    "final_usd_world_position_scene_units": actual,
                    "displacement_scene_units": math.dist(actual, initial_positions[platform.entity_id]),
                    "displacement_m": math.dist(actual, initial_positions[platform.entity_id]) * meters_per_unit,
                    "motion_model_position_scene_units": expected,
                    "final_pose_error_units": err,
                })
            stage_wall_s = time.perf_counter() - stage_started
            report["stages"].append({
                "name": f"{count}_platforms", "platform_count": count,
                "independent_kinematic_states": len(platforms), "actual_platform_prims": len(platforms),
                "frames_requested": frames, "kit_update_events": event_state["count"] - start_events,
                "app_update_calls": frames, "frame_time_measurement": "app.update wall time only; excludes scene.update and GPU sampling",
                "app_update_wall_ms_mean": sum(wall_times) * 1000 / len(wall_times),
                "app_update_wall_ms_max": max(wall_times) * 1000,
                "scene_update_wall_ms_mean": sum(scene_update_times) * 1000 / len(scene_update_times),
                "scene_update_wall_ms_max": max(scene_update_times) * 1000,
                "stage_wall_s_including_scene_app_updates_and_sampling": stage_wall_s,
                "max_usd_pose_error_units_across_frames": max_pose_error,
                "gpu_frame_time_ms": None, "gpu_frame_time_status": "not exposed by this HydraStorm/Kit instrumentation",
                "process_rss_peak_mib": rss_peak, "gpu_peaks_sampled": gpu_peak_by_uuid,
                "platform_positions_usd_world": positions,
            })
            report["status"] = f"{count}_platform_stage_complete"
            persist_report(args.output, report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["first_error"] = f"{type(exc).__name__}: {exc}"
        report["failed_unix_s"] = time.time()
        persist_report(args.output, report)
        raise
    finally:
        report["ended_unix_s"] = time.time()
        persist_report(args.output, report)
        if subscription is not None:
            subscription.unsubscribe()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
