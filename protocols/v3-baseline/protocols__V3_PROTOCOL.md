# V3 Protocol: Nine Regions, Thirty Platforms, Fifty-Four Tasks

## Objective and stage boundary

V3 tests whether multi-platform regional execution and cross-region support work in the simulator, whether the frozen replanner creates legal and consequential alternatives, and whether the resulting ordinary-event gain labels have enough signal for light-model fitting. Implementation and 18-instance engineering smoke come first. Source/config/evidence are then frozen for coordinator review. Formal collection is prohibited until that review is recorded. This milestone does not train a world model or PPO policy.

## Frozen scenario contract

- Map: 100 x 100 global grid; valid cell coordinates are integers in [0, 99].
- Region boundaries on each axis: 0, 33, 66, 100. Cells 0-32, 33-65, and 66-99 map to region indices 0, 1, and 2. A coordinate exactly 33 or 66 belongs to the higher-index region.
- Nine rectangular responsibility regions are geographical metadata. They do not grant or deny platform capability.
- Fleet: 24 UAVs and six helicopters. Six regions home three UAVs and three home two; a seeded, deterministic rotation selects which regions have two. Helicopter home regions rotate deterministically across the nine regions.
- Task set: six logical tasks in each region, three delivery and three inspection. Per region four are initially public (two of each kind); two are revealed later (one of each). Total: 54 logical tasks.
- Episode horizon: 90 simulated minutes. Counterfactual evaluation horizon: 30 simulated minutes.
- Candidate limit: at most 16 physical actions including NOOP; never pad a short catalog.
- Local disturbance: a 2 x 2 patch whose location is independent of the 33 x 33 responsibility region geometry.
- Initial assignment: generated from the public C1 algorithm; record algorithm/version, public input hash, assignment, unassigned reason, and a recomputable receipt. Queue feasibility includes cumulative time, energy, inventory, and capability.

## Mobility, support, and communication

- UAVs may execute home-region tasks by default and may support four-neighbor regions in a replan, subject to the same capability, payload, energy, deadline, stock, and communication checks as all actions. Region membership never replaces those checks.
- Helicopters may support any region, subject to their explicit capability and resource limits.
- Replanner search proceeds through home region, four-neighbor regions, then global fallback when an event is severe or no legal local alternative exists. Search scope, generated count, rejection reasons, deduplication, and chosen/executed action are recorded.
- Cooperative work is represented by one logical task and atomic multi-platform commitment. Partial commits and duplicate completion reward are invalid.
- Communication graph is generated from intra-region and adjacent-region links. Link health affects command delivery and safety state transitions; the graph is not assumed fully connected.
- Platform/task graph inputs use dynamic node sets. Any model path with a fixed capacity must reject overflow explicitly; it may not truncate silently.

## Event and safety rules

The seven frozen conditions are none/light, task update/arrival, platform failure/capability loss, energy anomaly, communication degradation, environment/obstacle change, and compound. Event rates are specified per exposed platform, link, task, or region and frozen before smoke. Forced safety actions bypass cooldown and gain gating. Keep and replan branches share the same snapshot, counter-addressed external suffix, and safety continuation. Actual environment transitions are checked after safety pause, release, reassignment, or termination.

## Counterfactual label contract

For each eligible ordinary state, pair keep and fixed-replanner branches from an identical serialized snapshot and identical exogenous suffix. Continue each selected queue with the same future policy and shared safety emergency handling. Default horizon is 30 minutes, truncated only at episode end. Record component outcomes and use the frozen V2 score weights carried into the V3 config; do not tune weights to increase positive labels. Independent replay code recomputes NOOP, target-task, and non-target-task outcomes from raw traces. Invalid/empty windows stay masked with an explicit reason.

## Work stages and gates

### Stage 1: engineering smoke

Run deterministic construction cases for multiple UAVs in one region, cooperative task atomicity, neighboring UAV takeover, helicopter global support, simultaneous independent regional disturbances, link partition, safe pause/termination, legal substitute, and genuine no-substitute. Then run six existing condition families with three independently generated V3 instances each (18 total). Verify the 30/54/9 scenario contract, public C1 receipt, execution legality, paired replay, independent label recomputation, and hidden-future invariance.

Freeze source/config/evidence hashes and submit them for read-only coordinator review. Any post-review change to execution semantics requires a new version identity and rerun of affected checks.

### Stage 2: bounded formal collection (review required)

Only after recorded approval: seven conditions x 12 train + 12 validation instances = 168 maximum. Test and OOD sets remain sealed. Preserve the fixed three trajectory-source proportions; sample at most eight ordinary states per trajectory and two paired suffixes per state. Each scenario writes to its own temporary shard, closes and validates the file, then atomically commits a completion marker. Aggregate indexes are rebuilt only from validated shards. Restart uses a new run ID and can reference validated shards; it never appends to a damaged gzip stream. Per-shard CPU, wall time, peak RSS, exit state, and hashes are recorded.

Formal-data review checks only new evidence and key invariants: identity/split isolation, pairing, legality, safety, independent replay, file completeness, and resource receipts. It does not redo the full source audit.

### Stage 3: 30/30 stop gate

Count independent ordinary decision states, not branches, suffixes, or cooperative sub-actions. Exclude forced safety and empty-obligation windows. Require at least 30 valid positive and 30 valid non-positive states with distinct scenario-family coverage and both train and validation coverage. The 168-instance cap cannot be exceeded to reach this gate. If it fails, stop before fitting models and report candidate coverage, NOOP-only causes, gain components, timing, and resource limits.

### Stage 4: light models, conditional

Only if Stage 3 passes: fit constant, Ridge, shallow tree, and small MLP baselines; use grouped train/validation, select the threshold on validation, and calibrate separately. Compare Existing-ET, 4-minute periodic, public risk rule, heuristic gain, Gain-Tree, and Gain-MLP using the same V3 replanner and safety rules. Count feature, inference, planning, repair, communication, and switching costs end-to-end. Independent test may be opened only after safety, completion non-inferiority, and total-cost criteria from the parent protocol are satisfied.

## Interrupted-run recovery and resource accounting

An incomplete shard is evidence of failure, not a completed scenario. Do not synthesize gzip footers or receipts. Preserve the first error and last valid checkpoint/marker; issue a new run identity on retry. V3 resource totals are reported independently from the incomplete V2 total, with the V2 historical uncertainty stated alongside them. No paid compute is authorized.

## Claims and archive

Smoke PASS means only the engineering contracts held in the frozen smoke scope. A 30/30 label count authorizes light-model fitting but does not establish method efficacy. Report engineering status, data-gate status, method support, and later world-model eligibility separately. Archive source/config/evidence hashes and sanitized summaries in `Battleplus/2026_event_trigger_reaserch`; do not publish secrets, private credentials, private databases, or raw large private traces.
