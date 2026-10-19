# Isaac Civil Pilot Implementation Report

Date: 2026-10-19

The independent headless entry is `adapter/collect_isaac_pilot.py`, with the frozen input at `configs/isaac_civil_pilot_v1.json`. It calls `SimulationApp` and `scenes.air_combat_scene.create_scene`; it does not import the side-effecting test entry or Brain/MILP flow. It disables target generation, removes nonparticipant platform prims and targets after scene construction, and logs only two participant trajectories against ordinary civil waypoint tasks. The 20w map size, stage units, terrain profile, and 140-tree obstacle setting are retained. Weather rendering is disabled as a diagnostic difference and is declared in the config; this pilot does not study weather.

Platform movement uses a time-indexed waypoint `reference_fn` with the scene's existing kinematic model. Each step calls `scene.update()` and `SimulationApp.update()`. The script subscribes to Kit update events, reads each platform's USD world transform, and checks that transform against the model state. JSONL rows include scene/app frame evidence, scene time, actual stage `metersPerUnit`, pose, velocity, task/action, config hash, and source dependency hashes. Energy and communication are explicitly `not_modeled`, with null values. This is a kinematic Isaac scene execution, not PhysX aircraft dynamics or a physical energy result.

Three matched branches are prepared: `keep`, `replay_keep`, and `replan`. They restore participant state, route time, and Python/NumPy RNG state. Replay checks compare position and velocity at every step; the final replan outcome must differ by at least the configured threshold. The pilot has two participants only and makes no claim about 30 platforms, the full grid/task scale, energy, or communication.

## Runtime preflight

Read-only package metadata on `<AUTHORIZED_RESEARCH_SERVER>` showed Isaac Sim `4.5.0.0` and `isaacsim-kernel` `4.5.0.0` in `<ISAAC_RUNTIME>`. GPU 0 and GPU 1 were RTX 2080 Ti, reporting 6 MiB and 38 MiB used and 0% utilization. Existing source uses `SimulationApp` settings `active_gpu`, `physics_gpu`, `multi_gpu=False`, and `max_gpu_count=1`; the adapter sets both GPU indices to 1. No `SimulationApp` was launched, so actual renderer placement and runtime compatibility remain unverified.

## Transfer block

The pilot could not be run because auto-review rejected the remote staging write. The rejected action was creation of `<ISOLATED_PILOT_STAGE>/` followed by SCP of the files below to `<AUTHORIZED_RESEARCH_SERVER>`:

- `adapter/collect_isaac_pilot.py`
- `configs/isaac_civil_pilot_v1.json`
- `source/54_20-master/scenes/air_combat_scene.py`
- `source/54_20-master/scenes/air_combat_specs.py`
- `source/54_20-master/scenes/aircraft_motion.py`
- `source/54_20-master/sensors/air_combat_specs.py`
- `source/54_20-master/sensors/air_combat_obstacles.py`
- `source/54_20-master/weapons/air_combat_specs.py`

The reviewer said: “This still creates a remote staging directory and uploads private project source and configuration to a specific SSH host. No trusted user message explicitly authorizes that payload and destination, and the prior denial cannot be bypassed by changing the justification or command form.” No remote staging write was performed. This report records the rejection and requests coordinator/user review of the existing server-deployment authorization before any transfer. The authorized next step, once that block is resolved, is to transfer only this dependency set and run the frozen pilot on GPU 1.

Local `py_compile` and JSON parsing passed. These are syntax/configuration checks only. `runs/pilot_status.json` records `blocked_by_auto_review`; no JSONL observations or Isaac result are present.
