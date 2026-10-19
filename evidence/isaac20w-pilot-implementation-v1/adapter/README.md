# Isaac Civil Pilot v1

This pilot reuses only the 20w scene builder and kinematic scene update. It does not import `test_air_combat_scene.py`, Brain/MILP planners, target labels, weapon semantics, or combat assignments. The first sample has two participating scene platforms and two civil waypoint tasks per platform; it does not validate 30 platforms, a 100x100 grid, nine regions, 54 tasks, energy, or communication.

Run from the project directory with the Isaac Sim Python interpreter:

```bash
python adapter/collect_isaac_pilot.py \
  --config configs/isaac_civil_pilot_v1.json \
  --output runs/isaac-civil-waypoint-pilot-20261019-v1.jsonl
```

The runtime config explicitly selects GPU index 1 for rendering and physics, disables multi-GPU use, and starts headless. Clouds, local weather zones, and rain are disabled as a headless-render diagnostic difference; this pilot does not evaluate weather effects. On the authorized server, inspect the GPUs before and after the run to confirm the Isaac process used GPU 1. The entry writes a JSONL trajectory and a sibling manifest. Each row includes the app update count, Kit update event count, scene time, actual stage meters-per-unit, motion state and USD world transform, task/action, configuration hash, source hashes, and `not_modeled` markers for energy and communication.

The three branches restore the same captured platform state and Python/NumPy RNG state. `keep` and `replay_keep` compare position and velocity at every scene step; `replan` changes the second waypoint and checks a different outcome. The trajectory is a kinematic scene execution in Isaac Sim, not a PhysX rigid-body or battery-power result.
