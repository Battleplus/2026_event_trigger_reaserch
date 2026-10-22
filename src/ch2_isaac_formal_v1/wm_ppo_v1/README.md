# Conditional world model and PPO interface v1

This directory contains the post-collection model contract and a gated
world-model training entry point. The entry reads only a terminal
`light_models_v1` batch after its formal 168-scene receipt and aggregate gates
pass, then recomputes feature coverage after extraction. No training was run
for this implementation task. G, G-WM and PPO remain `NOT_RUN`.

The world model consumes public history only. Each input is a sequence of
variable-size platform/task/observation graphs. A masked node encoder pools each
graph and a GRU predicts **one-step** dynamic targets (`task_progress`,
`energy_delta`, `delay`, and `risk`). The optional gain head consumes an
independently reduced paired gain label. A 30-minute endpoint is never renamed
as a one-step target.

`public_dynamics.py` exposes `public_projection_to_frame` and
`build_public_one_step_example` for public runtime projection snapshots. The
input allowlist is limited to current aircraft position, velocity, type, home
zone, availability/fault/hold flags, available energy, active task IDs and queue
IDs, plus observations whose `available_at_sim_time_s` is no later than the
snapshot time. Identifiers are used only to align public state or count/order
queue nodes and are never encoded as feature values. Future/private keys are
ignored. Home zones are normalized over the public 0-to-8 zone range. One-step
examples require adjacent integer-second snapshots; only the
history through `t` is encoded, while the `t+1` snapshot and events are used for
targets. Per-target masks are carried into `DynamicGraphBatch`. Energy delta is
the mean available-energy change across matching public platforms. Task progress
is the count of public `search_cell_complete` events in the interval, not hidden
geometric sweep progress. Delay is supervised only when an `(aircraft_id,
decision_id)` has exactly one public `transmission_sent` and one
`command_applied` event with a valid time order; repeated or otherwise ambiguous
joins are masked. Right-censored or unjoined delay is also masked. Risk is a
public safety/fault event label and is zero only when the
caller attests complete interval event logging. Queue membership/order and
observed public locations are features; queue task IDs are not. The current
projection does not supply task route, priority, deadline, or geometric
within-task progress, so these are not inferred. Optional gain labels require
the caller to mark an independently validated paired label. No real one-step
dataset is claimed by this interface.

`wm_entry.py --stage world-model --batch-output <terminal-batch> --output <receipt.json>`
is the conditional training command. It additionally checks decision-gain
coverage and disjoint scenario splits, fits normalization on train scenes only,
and requires the configured GPU1 UUID to be the sole visible CUDA device mapped
to `cuda:0`. There is no CPU training fallback. Dynamic one-step and paired
gain losses run as separate phases. Seeds are 8201, 8202 and 8203. Checkpoints
record model and optimizer state, normalization, dataset and schema identity,
RNG state, architecture, device mapping and fixed-input predictions; a fresh
process reloads each checkpoint on the same device and checks predictions and
optimizer state.

The `g` and `g-wm` entry stages emit `NOT_RUN`; `ppo_entry.py` also remains
`NOT_RUN` until the replanner and real environment adapter are implemented.
The scope review in `collection/progress/WM_SUPERVISION_SCOPE_REVIEW_20261022.md`
defines the limits of the available public targets.

`ppo_core.py` now provides the masked-candidate graph actor/critic, variable-
interval GAE and clipped PPO optimizer for a future rollout caller. It is not
wired to Isaac. The G-WM critic-only auxiliary path distills frozen WM
one-step outputs for an actually executed action with explicit per-dimension
joint-support masks. Public candidate descriptors are separate from five-value
WM action summaries. Current collected action support and train-split
dynamic-target scale evidence are missing, so real G-WM auxiliary training
remains unavailable until those inputs are recorded and validated. An update
with no valid auxiliary elements reports `UNAVAILABLE_G_EQUIVALENT` and cannot
be counted as G-WM. The WM gain adapter is wired to the local
`closedloop_v1` catalogue gate, while real Isaac/PPO execution remains
`NOT_RUN`. The focused CPU test is synthetic and does not change that status.

Each completed world-model seed writes `gain_deployment.json` with validation
selection/calibration roles, selection-only threshold, calibration-only lower
residual bound, validation predictions and the checkpoint SHA-256. Validation
scenarios are assigned by `light_models_v1.train_models._scenario_validation_split`;
test/OOD rows are not accepted. At alpha 0.1, fewer than nine calibration
scenarios leave lower-bound mode `NOT_RUN` while mean mode remains available.
`GainDeploymentAdapter` exposes public-history inference with explicit phase,
checkpoint/schema checks, gain mean/lower bound/threshold/version/device, and
2-8 contiguous one-second frames ending at the pre-candidate decision phase.
Timing reports feature construction, CPU-to-device transfer, synchronized
forward, device-to-CPU return, calling-thread CPU time and end-to-end wall time. It reports
`NOT_INTEGRATED`; no method-E closed-loop run is claimed. CPU inference is
available only through an explicitly marked synthetic fixture path.

Synthetic interface checks only:

```powershell
Push-Location development/ch2_isaac_roi_mission_v3_pilot
python -B -m unittest discover -s wm_ppo_v1/tests -v
$files = (Get-ChildItem wm_ppo_v1 -Recurse -Filter *.py).FullName
python -B -m py_compile $files
Pop-Location
```

These checks do not establish a scientific result. GPU training and PPO are
conditional on the formal batch passing the 30/30 gate and the main coordinator
reviewing its receipts.
