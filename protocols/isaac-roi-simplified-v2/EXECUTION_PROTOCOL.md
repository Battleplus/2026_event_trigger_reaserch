# Isaac ROI Protocol v2: Execution Protocol

Status: protocol specification; execution has not started. Dates and observations below are based on the v1 audit artifacts dated 2026-10-19. This file does not certify an experiment as passed.

## Purpose and scope

Stage A is a bounded research experiment in Isaac. Stage B ports the frozen task, data contract, and selected methods to the formal platform. Stage A may compare decision methods; it must not claim a completed Stage B integration. The task is intentionally limited to search and search plus abstract strike selection. It has no weapon control, projectile simulation, ballistics, or physical damage model. A strike action means only selecting/assigning an abstract target for an eligible platform.

## Scenario contract

- The map is a 300 km by 300 km background. Sample a 50 km by 50 km ROI per episode and randomly place it within the background subject to valid terrain bounds.
- Partition the ROI into nine regions. Each region starts with two UAVs and two helicopters, for 36 platforms total. UAVs may cross region boundaries. Helicopters are confined to their assigned region. Tasks are platform-type-specific; a helicopter cannot receive a UAV-only task and vice versa.
- Target load is one of 1, 3, or 5 targets per region, yielding 9, 27, or 45 targets per episode. Keep target count and type quotas explicit in the sampled scenario record.
- Include search-only and search-plus-abstract-strike task modes. For the strike mode, log target selection and assignment only; do not introduce weapon release or physical trajectory behavior.
- Scenario seed determines ROI placement, terrain/obstacles, targets, platform initial state, task mix, and exogenous communication/weather draws. Save the generated scenario before any method sees it.
- Record the actual Isaac stage units and effective scene configuration. Do not interpret the historical v1 scene defaults as this task's map or platform configuration.

## Decision and observation labels

`decision_class` has two values: `forced` means a safety-critical decision that must bypass the learned gate; `ordinary` means a normal decision subject to the gate. Scripted perturbations and task/target state changes are not automatically forced decisions. Record them as event causes in separate fields.

Record label certainty and observation-window state independently: `label_status` is `known` or `unknown`, and `window_status` is `nonempty` or `empty`. Do not collapse an unknown label or empty window into a third decision class. Keep event cause, decision class, label status, and window status as separate fields. Unknown labels and empty windows are excluded from supervised label counts, reported separately, and never treated as negative examples. The `forced` path must bypass the gate at runtime; a forced decision is not a policy-discovered outcome.

## Stage A sequence

### A1: full mission and fair paired replay

Run one complete 90-minute simulated mission in the actual Isaac application using the 36-platform ROI task contract. Then run a 30-minute fair paired comparison from identical saved mission state, exogenous draws, and remaining time budget. The comparison must differ only in the decision being evaluated. Record simulator/app frame progress, simulation time, state hash at branch, RNG state or immutable exogenous stream, and per-step outputs. A Python-only simulation or a loading test is not an A1 run.

### A2: pilot design and decision signal

Run the 2 x 3 x 6 pilot grid: two task modes, three target loads (1/3/5 per region), and six independently seeded scenarios, for 36 pilot episodes. Use the pilot to establish that observations contain a decision signal and that the task produces a feasible, nontrivial workload. Do not select scenarios or hyperparameters because they favor a method. Record feasibility, completion, unresolved/unknown fraction, all-ROI consequences, and runtime. Freeze the scenario-generation and inclusion rules after this pilot.

### A3: model-data generation

Create seven model task classes, with 12 training and 12 validation trajectories per class, for a hard cap of 168 trajectories. Each trajectory contains at most eight ordinary decision states. For each state, collect two future suffixes; each suffix has paired `keep` and `replan` branches from the same state and exogenous inputs. Suffixes are used to compute the predeclared gain labels; they are not extra decision states.

Count positive and non-positive examples by unique `state_id` after deduplication, with coverage represented across the seven task classes and scenario groups. Before training any learned predictor, require at least 30 unique positive and 30 unique non-positive training state IDs under the frozen label definition. If either threshold is unmet within the 168-trajectory cap, stop the learned-model comparison and report it unavailable; do not collect beyond the cap or move validation data into training. Unknown labels and empty windows do not count toward either threshold.

### A4: model fitting and policy evaluation

- CPU baselines: constant predictor, Ridge regression, and a shallow tree model. Fix their feature schema and tuning procedure before looking at validation/test results.
- GPU learned models: MLP with 64 x 64 hidden units and graph GRU. The graph GRU implements a genuine one-step world model predicting next-state features from current graph/state and action. `G` is the graph PPO policy with its ordinary critic. `GWM` is the same policy with a frozen WM and an auxiliary critic; it is not a gate condition. The WM output is stop-gradient input to the auxiliary critic. Auxiliary loss gradients update the auxiliary critic, do not directly update the actor or frozen WM, and do not change the trigger rule. Train the WM on paired suffix gain labels, then freeze it before policy evaluation. The six gate/trigger conditions (ET, light, WM-gate, and 4-minute as specified in the matrix) are separate from the `G`/`GWM` critic choice. Safety-critical `forced` decisions bypass any gate.
- Training seeds: 8201, 8202, and 8203 for GPU models. Persist seed, code/config hashes, optimizer state, checkpoint hash, and validation output for each replicate.
- Train and freeze `G` and `GWM` before final evaluation. After freeze, recollect/resample only the relevant reconstructor's train/validation gain labels and fit that reconstructor using the already fixed episode splits. Gain labels are computed from each paired suffix under the predeclared score and sign/tie rule, frozen before model fitting. Do not construct or resample test labels; the test manifest and outcomes remain sealed until every method and checkpoint is frozen.
- The test set is 42 independent scenes: seven task classes x six scenes per class. Every scene is evaluated by all six methods and all three seeds (8201/8202/8203) as paired runs, yielding 42 scenes and 756 method-seed runs. The six methods are (A) ET + G; (B) light + G; (C) ET + GWM; (D) light + GWM; (E) WM-gate + GWM; (F) 4-minute + GWM. Preserve the exact group names and definitions in the run manifest. ET, light, WM-gate, and 4-minute are comparison conditions, not post-hoc labels.

## Measurement and fairness

Report all-ROI consequences, including every region, all targets, and all 36 platforms, even when the active decision concerns one region. Minimum outcomes: ordinary task completion, target coverage/selection, missed or unresolved tasks, safety-critical forced-decision count, unknown-label count, empty-window count, platform availability, and failures. Keep search and abstract-strike results separately attributable.

Measure online decision latency from the common trigger until the validated command becomes effective. Include observation assembly, feature extraction, model inference, graph construction, optimization/planning, communications, and action validation/repair. A slow model/planner delays command effectiveness and that delay remains part of the decision latency and resulting mission outcome. Report wall-clock and simulated-time decision latency, p50/p95, failures, and timeouts. Measure simulator/application stepping wall-clock separately as experiment infrastructure/runtime cost; do not add it to online decision latency or task/mission cost. Report task/mission cost separately, and include asynchronous method work in the appropriate online decision interval.

For paired comparisons, branch from the same complete simulator/task state and exogenous stream. Preserve ROI, targets, initial positions/velocities, task queue, energy/communication state, and RNG or equivalent sampled inputs. A pair is invalid if any of these differ. Log the pair key and verify equality before policy execution. Each A3 `keep`/`replan` suffix pair and each A4 scene/method/seed block follows this rule.

Prevent leakage by assigning whole scenario/trajectory groups to train, validation, or test before fitting. No frame from an episode may cross a split. Episode, scene, actor, task, and target IDs are metadata for joining and pairing only; exclude them from neural feature tensors. Keep public/observable inputs and hidden state in physically separate schemas/objects and pass only the public object to policy and model code; do not mix hidden fields into a feature bundle and rely on labels to suppress them. Fit normalization and graph vocabularies on training data only. Validation is for model selection; test is opened once after all policy and metric definitions are frozen. Archive split manifests and hashes.

## Resource policy and stop conditions

No CPU-time budget is imposed. No paid compute, cloud GPU, or paid service is authorized by this protocol. Use available local/authorized Isaac resources and record actual allocation. Stop and mark an episode invalid on missing state restoration, simulator/app frame mismatch, invalid task-type assignment, cross-region helicopter movement, missing outcome provenance, or incomplete cost instrumentation. Keep the raw receipt; never label an invalid run as a success.

## Evidence and reporting

Every run has an immutable manifest with scenario, seed, mode, target load, method/group, code/config/dependency hashes, runtime identity, unit scale, start/end timestamps, status, and output hashes. Results are reported as `planned`, `running`, `complete_unreviewed`, `invalid`, or `reviewed`; this protocol alone cannot assign `reviewed`. A successful short load test is capacity evidence only and cannot satisfy A1-A4.
