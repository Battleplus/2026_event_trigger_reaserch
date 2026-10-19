# Isaac Scene Reuse Audit

Audit date: 2026-10-19 (Asia/Shanghai)

## Source and handling

- Source archive: `54_20-master.zip`
- SHA-256: `99ec901f5ef9ff9ceaccc84c932507f207ae5b06af824da3fc04724a60d0eca5`
- Archive entries: 659; extracted regular files: 558.
- Isolated copy: `source/54_20-master`. Extraction checked every normalized path against the destination and rejected symlink entries. No archive script was run.
- No standalone LICENSE/COPYING file or explicit license statement was found in the root README. Keep this source snapshot private; redistribution rights are unverified. Historical combat, weapons, MILP, and experiment artifacts remain quarantined in this source snapshot and are not accepted as civil-task results.

## Scene identity and defaults

The requested 20w route is `test_air_combat_scene.py` -> `scenes/air_combat_scene.py`. The scene defaults to a 300 km square map, 100 meters per scene unit, and simulation time scale 45. `_setup_stage` writes that meters-per-unit value into USD stage metadata. The entry script's `build_config()` can override map size, meters-per-unit, time scale, terrain, weather, obstacles, and target counts through `QL_AIR_COMBAT_*` environment variables; the effective run configuration must be recorded at runtime. `configs/mountain_forest_aircraft.yaml` is not this scene's configuration.

Default platform creation is eight hard-coded aircraft with fixed route definitions. Default movement is the repository's kinematic motion model: `scene.update()` advances the model and writes the USD root pose. This is a USD scene advanced inside Isaac, but the default aircraft movement is not a PhysX rigid-body controller. A 30-platform workload needs configurable civil platform spawning and cannot be inferred from the current eight entities. No fixed formation is required by the new task; initial positions and routes can be sampled per episode.

The entry script creates a visible `SimulationApp` at module import and enables an opening missile demo by default. It is an unsafe acquisition entry point for the civil study. Reuse the scene construction and USD terrain/unit setup through a new headless civil adapter; do not launch this script or carry its combat logic into the task layer.

## Reusable interfaces and gaps

- Reusable scene foundations: USD stage setup, procedural terrain/obstacles, platform motion models, sensor/contact calculations, scene stepping, and SI-to-scene conversion. Verify all emitted samples carry the actual stage `metersPerUnit` and effective configuration.
- `IsaacSessionController` forces `simulation.time_scale = 1.0`, accepts per-platform `position_km` and `velocity_km_s` values, advances the scene, and returns snapshots with agents, targets, sensor contacts, and static obstacle data. The controller only accepts IDs already spawned by the scene. Its `_ExternalMotionModel` writes commanded state directly; treat that as state injection/telemetry plumbing, not executable movement control. For the first civil pilot, replace a spawned platform's `reference_fn` with a waypoint reference and let the existing kinematic model enforce its speed limits while `scene.update()` advances and writes the USD pose.
- Existing persistence is visual recording (MP4 plus a manifest), not a training trajectory export. Add deterministic per-step JSONL/Parquet records for observations, action/waypoint, task state, physical state, energy, communication state, seed, and episode/config IDs.
- `IsaacAirCombatEnvironment.reset()` only resets counters and performs a tiny scene update. The session controller has a stronger reset: it captures/restores platform pose/velocity/attitude, target routes/phase, destroyed flags/visuals, and world state. It does not capture RNG generator state, and retaining mutable motion-model objects is not a portable serialized snapshot. For paired keep/replan runs, implement and verify full state+RNG restoration or recreate each arm from the same recorded seed and initial state.
- The old `isaac_scene_bridge.py` translates combat entities and weapon semantics into the Brain/MILP schema. Do not use it as the public civil-task observation schema.
- No reusable civil inspection task, energy budget/depletion model, communication delay/dropout model, or paired keep/replan data logger was identified. Add these as a separate civil adapter and schema. The 100x100 grid, nine regions, 30 platforms, and 54 tasks are task-layer targets; map them into measured native scene coordinates instead of treating them as scene defaults.

## Runtime preflight

This round did not find an immediately callable Isaac runtime on the local workstation: the project venv has no `isaacsim` module, Isaac-related environment variables are unset, and the checked common install paths are absent. The authorized server `authorized-linux-server` is `authorized-linux-server` (Ubuntu 22.04, Linux 5.19). A candidate Python 3.10 venv exists at `<ISAAC_RUNTIME>`, but `site-packages/isaacsim` was absent at inspection time. An already-running external process (PID 2065131) was installing `isaacsim[all,extscache]==4.5.0.0`; it was not started or altered by this audit. GPU 0 and GPU 1 are RTX 2080 Ti with 6 MiB and 38 MiB used respectively and 0% utilization at inspection. No Isaac process was found. Docker inventory was unavailable to this user due to daemon permission denial. Do not assume the installation completes or use GPU 0 while other work may claim it.

## Shortest path to a real Isaac pilot

1. Confirm the external install completed, then read the venv's installed Isaac version and locate its supported headless launcher. Do not count Python-only historical runs as Isaac data.
2. Add a small headless civil adapter that constructs the 20w terrain/stage with an explicit, logged configuration and spawns one or two civil inspection platforms plus one or two inspection sites. Disable combat targets, weapons, and the old mission demo. Preserve free-form sampled platform layouts.
3. Replace each pilot platform's `reference_fn` with a civil waypoint reference and advance through the existing kinematic `scene.update()` at `time_scale=1`. Record per-step simulator state and civil energy/communication fields with a fixed seed/config. This is the first true Isaac sample; validate frame progression, nonzero movement, and state restoration before scaling.
4. Add paired keep/replan branches from a verified identical full state and RNG state. Only then expand toward 100x100 cells, nine regions, 30 platforms, and 54 tasks, with measured native coordinates and a trajectory table.

Current blockers are runtime availability being unconfirmed while the external install is still active, the absence of a civil mission/energy/communication adapter and trajectory exporter, and incomplete RNG restoration for paired runs. No Isaac scene was launched, no training was run, and no result in this audit is represented as an actual Isaac observation.
