# Isaac ROI Mission v2

This package defines a civil search task layer over the archived 20w Isaac scene. The scene is 300 x 300 km at 100 m per USD unit. The episode samples a 50 x 50 km ROI and divides it into nine zones. It creates two UAVs and two helicopters per zone. Native scene `scene.update()` advances the source kinematic motion models; the runner records USD world-transform readback after each Isaac app update. This is kinematic scene motion, not PhysX aircraft dynamics.

The public task catalog is fixed at nine search cells per zone (81 total). Each cell is UAV-only or helicopter-only. Hidden target count is configured independently as 1, 3, or 5 per zone; truth coordinates and type do not affect the public routes. Search uses fixed lawnmower waypoints generated from the public cell bounds. After simulated sensor-radius discovery, target coordinates and kind become public observations used for abstract service routing. The separate `evaluator_truth.json` is evaluator-only and must not be supplied to planner code. Service only changes task state after arrival and dwell time; this package has no weapon controls, firing, or ballistics. Energy is a configured linear Wh/km estimate. Communication delay and dropout affect command delivery in the simulation task layer and do not model radio propagation. During the configured deterministic command outage, the affected UAV clears pending commands and locally holds at its current position; this is a simulated safety response, not radio propagation or an avionics guarantee. The commanded flight Z is 20 km and terrain clearance is a source-scene visualization/obstacle-avoidance setting; sensing is an analytic 2D distance check and does not represent real AGL or imaging.

The initial public assignment is `c1_greedy_proxy`: same-zone, same-type candidate filtering, then minimum assigned load, distance to public cell, and aircraft ID. The historical C1 solver is not reused. UAVs may provide cross-zone task support after a failure or replan. Helicopters remain restricted to their home zone.

## Representative Run

Set `ISAAC_PYTHON`, `SOURCE_ROOT`, and `OUTPUT` to the Isaac Sim 4.5 Python executable, read-only archived source tree, and desired output directory:

```powershell
$env:CUDA_VISIBLE_DEVICES = "<GPU UUID>"
& $env:ISAAC_PYTHON run_a1.py `
  --config configs/a1_representative.json `
  --source-root $env:SOURCE_ROOT `
  --output $env:OUTPUT `
  --phase representative
```

The runtime config requests Vulkan ordinal 1, CUDA physics ordinal 0, and `multi_gpu=false`. The representative episode advances 5,400 mission steps at one second per step. A UAV fault at 900 seconds holds the failed platform and releases unfinished UAV cells to eligible cross-zone UAVs. A deterministic 120-second command outage starts at 1,800 seconds for `uav-1-0`; the configured random dropout is addressed by seed, aircraft ID, and simulation second so paired branches receive the same exogenous loss schedule.

For the 30-minute paired diagnostic, use `--phase pair`. The runner records a 1,200-second common prefix, snapshots the live Isaac and mission state, then advances `keep` and `replan` for 1,800 seconds each. The pair's catalog policy is preliminary and needs the independent audit before it is used as a study result.

## Outputs

- `manifest.json`: frozen config, hashes of the source dependencies, and run settings.
- `trajectory.jsonl`: one row per aircraft per simulated second, including USD position, velocity, energy estimate, communication state, and task events.
- `events.jsonl`: task completion/discovery, communication delivery/drop, fault, local safety hold, and donor transfer events.
- `run_receipt.json`: durable stage progress, public task bounds and sweep waypoints, discovery thresholds, zone bounds, actor catalog, initial USD positions, and completion state. It records the sibling `evaluator_truth.json` path and hash for audit-only use. The receipt is flushed before `SimulationApp.close()`.
- `audit_report.json`: produced by the independent trajectory auditor as a separate post-run command. A complete run receipt alone does not certify the audit or represent a successful Isaac execution.

Run the task-layer checks locally:

```powershell
python -B -m unittest test_mission -v
python -B -m py_compile mission.py run_a1.py isaac_backend.py
```

## Validation Status

The local task-layer suite covers hidden-load separation, fixed public search cells, platform specialization, helicopter zone lock, sweep geometry, snapshot restoration, and abstract service idempotence. Python syntax and JSON parsing should be checked before execution. A successful Isaac run and independent trajectory audit must be recorded in the run receipt and audit report; local tests do not supply that evidence. Formal acquisition is staged after the 36-run pilot and review of pilot diagnostics, followed by the 168-run matrix. Model training is not part of this A1 package.
