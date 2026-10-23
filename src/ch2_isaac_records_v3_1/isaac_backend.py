"""Small Isaac 4.5 backend for the civil ROI mission runner."""
from __future__ import annotations

import copy
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _patch_source_scene_transforms(scene_module):
    from pxr import Gf, UsdGeom

    def get_or_add(xform, op_type, add):
        for operation in xform.GetOrderedXformOps():
            if operation.GetOpType() == op_type:
                return operation
        return add()

    def set_xform(prim, translate=None, rotate_xyz=None, scale=None):
        xform = UsdGeom.Xformable(prim)
        if translate is not None:
            get_or_add(xform, UsdGeom.XformOp.TypeTranslate, xform.AddTranslateOp).Set(
                Gf.Vec3d(*(float(value) for value in translate))
            )
        if rotate_xyz is not None:
            get_or_add(xform, UsdGeom.XformOp.TypeRotateXYZ, xform.AddRotateXYZOp).Set(
                Gf.Vec3f(*(float(value) for value in rotate_xyz))
            )
        if scale is not None:
            get_or_add(xform, UsdGeom.XformOp.TypeScale, xform.AddScaleOp).Set(
                Gf.Vec3f(*(float(value) for value in scale))
            )

    def set_root_pose(root_prim, position, state):
        set_xform(root_prim, translate=position,
                  rotate_xyz=(state.roll_deg, state.pitch_deg, state.yaw_deg))

    def set_root_scale(root_prim, scale):
        set_xform(root_prim, scale=(scale, scale, scale))

    def spin_rotors(rotor_prims, time_s):
        angle = (float(time_s) * 900.0) % 360.0
        for index, prim in enumerate(rotor_prims):
            name = prim.GetName()
            if "QuadRotor" in name:
                direction = 1.0 if index % 2 == 0 else -1.0
                rotation = (0.0, 0.0, angle * direction + index * 45.0)
            elif "TailRotor" in name or index == 2:
                rotation = (0.0, 90.0, angle)
            else:
                rotation = (0.0, 0.0, angle + index * 90.0)
            set_xform(prim, rotate_xyz=rotation)

    scene_module._set_xform = set_xform
    scene_module._set_root_pose = set_root_pose
    scene_module._set_root_scale = set_root_scale
    scene_module._spin_rotors = spin_rotors


class IsaacBackend:
    """Advance source kinematic models and return transforms read from USD."""

    def __init__(self, source_root, platform_records, runtime):
        self.source_root = Path(source_root).resolve()
        self.platform_records = [dict(record) for record in platform_records]
        self.runtime = dict(runtime)
        self.app = None
        self.scene = None
        self.platforms = {}
        self._references = {}
        self.kit_frame = 0
        self._event_state = {"count": 0}
        self._subscription = None
        self._base_speed_scene = {}

        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source root not found: {self.source_root}")
        for relative in (
            "scenes/air_combat_scene.py",
            "scenes/air_combat_specs.py",
            "scenes/aircraft_motion.py",
            "sensors/air_combat_specs.py",
            "sensors/air_combat_obstacles.py",
            "weapons/air_combat_specs.py",
        ):
            if not (self.source_root / relative).is_file():
                raise FileNotFoundError(self.source_root / relative)

        expected_uuid = self.runtime.get("expected_gpu_uuid")
        if expected_uuid:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(expected_uuid)
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))

        try:
            from isaacsim import SimulationApp

            self.app = SimulationApp({
                "headless": bool(self.runtime.get("headless", True)),
                "renderer": self.runtime.get("renderer", "HydraStorm"),
                "width": int(self.runtime.get("width", 1280)),
                "height": int(self.runtime.get("height", 720)),
                "active_gpu": int(self.runtime.get("vulkan_ordinal", 1)),
                "physics_gpu": int(self.runtime.get("cuda_physics_ordinal", 0)),
                "multi_gpu": False,
                "max_gpu_count": 1,
            })

            import omni.kit.app
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom
            from scenes import air_combat_scene as scene_api
            from scenes.air_combat_specs import QUAD_RECON_UAV, WZ21_LEADER
            from scenes.aircraft_motion import build_motion_model

            self.scene_api = scene_api
            self.Usd = Usd
            self.UsdGeom = UsdGeom
            self.Gf = Gf
            self._spec_by_kind = {"uav": QUAD_RECON_UAV, "helicopter": WZ21_LEADER}
            _patch_source_scene_transforms(scene_api)
            scene_api.compute_sensor_contacts = lambda _scene: []
            scene_api.compute_obstacle_contacts = lambda _scene: []
            scene_api.assign_targets = lambda _scene: []

            context = omni.usd.get_context()
            context.new_stage()
            self.stage = context.get_stage()
            scene_cfg = {
                "map": {
                    "real_size_km": 300.0,
                    "meters_per_unit": 100.0,
                    "mountain_height_m": 1500.0,
                    "terrain_vertical_exaggeration": 10.0,
                    "terrain_source": "procedural_grid",
                    "terrain_usd": None,
                    "grid": int(self.runtime.get("terrain_grid", 128)),
                },
                "visual": {
                    "aircraft_scale": float(self.runtime.get("aircraft_visual_scale", 180.0)),
                    "show_sensor_rings": False,
                    "show_sensor_cones": False,
                    "show_route_lines": False,
                    "show_terrain_grid": False,
                    "show_terrain_wireframe": False,
                    "show_contours": False,
                    "show_collision_body": False,
                },
                "weather": {"clouds": False, "local_zones": False, "rain": False},
                "bases": {"enabled": False},
                "simulation": {"time_scale": 1.0, "aircraft_model": "kinematic"},
                "targets": {
                    "radar_sites": 0,
                    "command_posts": 0,
                    "armored_vehicles": 0,
                    "forward_bases": False,
                },
                "obstacles": {
                    "enabled": True,
                    "mountains": True,
                    "forests": True,
                    "rock_fields": True,
                    "forest_tree_count": int(self.runtime.get("forest_tree_count", 140)),
                    "forest_tree_visual_scale": 220.0,
                },
            }
            self.scene = scene_api.create_scene(self.stage, scene_cfg)
            for old in list(self.scene.platforms):
                self.stage.RemovePrim(old.root_prim.GetPath())
            for old in list(self.scene.targets):
                self.stage.RemovePrim(old.root_prim.GetPath())
            self.scene.platforms = []
            self.scene.targets = []
            self.scene.contacts = []
            self.scene.obstacle_contacts = []
            self.scene.assignments = []
            self.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(self.stage))
            if not math.isclose(self.meters_per_unit, 100.0, abs_tol=1e-9):
                raise RuntimeError(f"unexpected stage metersPerUnit={self.meters_per_unit}")

            for idx, record in enumerate(self.platform_records):
                aircraft_id = str(record["aircraft_id"])
                if aircraft_id in self.platforms:
                    raise ValueError(f"duplicate aircraft_id: {aircraft_id}")
                kind = str(record["kind"])
                if kind not in self._spec_by_kind:
                    raise ValueError(f"unsupported platform kind {kind!r} for {aircraft_id}")
                initial_km = np.asarray(record["initial_position_km"], dtype=float)
                if initial_km.shape != (3,) or not np.isfinite(initial_km).all():
                    raise ValueError(f"initial_position_km must contain three finite values for {aircraft_id}")
                position = initial_km * (1000.0 / self.meters_per_unit)
                clearance = float(record.get("terrain_clearance_m", 120.0)) / self.meters_per_unit
                ground = self.scene.surface_height(float(position[0]), float(position[1]))
                position[2] = max(float(position[2]), ground + clearance)

                spec = self._spec_by_kind[kind]
                root_path = f"/World/AirCombat/Platforms/Civil_{aircraft_id.replace('-', '_')}"
                root = UsdGeom.Xform.Define(self.stage, Sdf.Path(root_path)).GetPrim()
                UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
                body = UsdGeom.Cube.Define(self.stage, Sdf.Path(root_path + "/Body"))
                body.CreateSizeAttr(1.0)
                UsdGeom.Xformable(body.GetPrim()).AddScaleOp().Set(
                    Gf.Vec3f(0.03, 0.02, 0.008) if kind == "uav"
                    else Gf.Vec3f(0.16, 0.03, 0.04)
                )
                UsdGeom.Gprim(body.GetPrim()).CreateDisplayColorAttr().Set(
                    [Gf.Vec3f(0.08, 0.38, 0.78) if kind == "uav" else Gf.Vec3f(0.72, 0.36, 0.08)]
                )

                model_cfg = {
                    "type": "kinematic",
                    "max_speed": float(record["max_speed_mps"]) / self.meters_per_unit,
                }
                model = build_motion_model(model_cfg, position, domain="air")
                self._base_speed_scene[aircraft_id] = float(record["max_speed_mps"]) / self.meters_per_unit
                platform = scene_api.AirPlatform(
                    entity_id=aircraft_id,
                    spec=spec,
                    root_prim=root,
                    motion_model=model,
                    reference_fn=lambda _time, p=position.copy(): p.copy(),
                    rotor_prims=[],
                )
                self.platforms[aircraft_id] = platform
                self._references[aircraft_id] = position.copy()
                self.scene.platforms.append(platform)

            self._subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                lambda _event: self._event_state.__setitem__("count", self._event_state["count"] + 1),
                name="civil_roi_backend_frame_counter",
            )
        except Exception:
            # The caller must persist the failure receipt before closing Kit.
            raise

    def step(self, commands, dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number of seconds")
        unknown = set(commands) - set(self.platforms)
        if unknown:
            raise KeyError(f"commands reference unknown aircraft: {sorted(unknown)}")
        scale = 1000.0 / self.meters_per_unit
        for aircraft_id, waypoint_km in commands.items():
            point = np.asarray(waypoint_km, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"waypoint for {aircraft_id} must contain three finite km values")
            self._references[aircraft_id] = point * scale
        step_time = float(self.scene.tactical_time_s)
        for aircraft_id, platform in self.platforms.items():
            target = self._references[aircraft_id]
            platform.reference_fn = lambda _time, p=target.copy(): p.copy()
            position_km = np.asarray(platform.motion_model.state.position, dtype=float)[:2] * self.meters_per_unit / 1000.0
            cap = self._base_speed_scene[aircraft_id]
            for block in self.runtime.get("spatial_speed_blocks", []):
                xmin, ymin, xmax, ymax = map(float, block["rect_km"])
                start_s, end_s = map(float, block["interval_s"])
                if (start_s <= step_time < end_s and xmin <= position_km[0] <= xmax
                        and ymin <= position_km[1] <= ymax):
                    cap = min(cap, float(block["max_speed_mps"]) / self.meters_per_unit)
            platform.motion_model.max_speed = cap
        next_time = self.scene.tactical_time_s + dt
        self.scene.update(tactical_dt=dt, tactical_time_s=next_time)
        before = self._event_state["count"]
        self.app.update()
        if self._event_state["count"] <= before:
            raise RuntimeError("SimulationApp.update returned without a Kit update event")
        self.kit_frame += self._event_state["count"] - before
        return self.positions()

    def positions(self):
        from pxr import Usd

        result = {}
        for aircraft_id, platform in self.platforms.items():
            transform = self.UsdGeom.Xformable(platform.root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = transform.ExtractTranslation()
            usd_scene = np.asarray([float(translation[i]) for i in range(3)], dtype=float)
            state = platform.motion_model.state
            model_scene = np.asarray(state.position, dtype=float)
            velocity_scene = np.asarray(state.velocity, dtype=float)
            result[aircraft_id] = {
                "position_m": (usd_scene * self.meters_per_unit).tolist(),
                "velocity_mps": (velocity_scene * self.meters_per_unit).tolist(),
                "pose_error_m": float(np.linalg.norm(usd_scene - model_scene) * self.meters_per_unit),
                "kit_frame": int(self.kit_frame),
                "scene_time_s": float(self.scene.tactical_time_s),
            }
        return result

    def snapshot(self):
        return {
            "scene_time_s": float(self.scene.tactical_time_s),
            "kit_frame": int(self.kit_frame),
            "platforms": {
                aircraft_id: {
                    "motion_model_state": copy.deepcopy(platform.motion_model.__dict__),
                    "reference_position_scene_units": self._references[aircraft_id].copy(),
                }
                for aircraft_id, platform in self.platforms.items()
            },
            "python_rng_state": random.getstate(),
            "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        }

    def restore(self, snapshot):
        from pxr import Usd

        if set(snapshot["platforms"]) != set(self.platforms):
            raise ValueError("snapshot aircraft IDs do not match this backend")
        self.scene.tactical_time_s = float(snapshot["scene_time_s"])
        self.kit_frame = int(snapshot["kit_frame"])
        random.setstate(snapshot["python_rng_state"])
        np.random.set_state(copy.deepcopy(snapshot["numpy_rng_state"]))
        for aircraft_id, saved in snapshot["platforms"].items():
            platform = self.platforms[aircraft_id]
            platform.motion_model.__dict__.clear()
            platform.motion_model.__dict__.update(copy.deepcopy(saved["motion_model_state"]))
            reference = np.asarray(saved["reference_position_scene_units"], dtype=float).copy()
            self._references[aircraft_id] = reference
            platform.reference_fn = lambda _time, p=reference.copy(): p.copy()
            state = platform.motion_model.state
            self.scene_api._set_root_pose(platform.root_prim, state.position, state)
        self.scene.contacts = []
        self.scene.obstacle_contacts = []
        self.scene.assignments = []

    def close(self):
        subscription = getattr(self, "_subscription", None)
        if subscription is not None:
            subscription.unsubscribe()
            self._subscription = None
        app = getattr(self, "app", None)
        if app is not None:
            self.app = None
            app.close()
