# Stage A to Stage B Migration Contract

Status: interface and evidence contract only. The formal platform/model has not been supplied or inspected, so this document specifies acceptance obligations without claiming compatibility or implementation.

## Boundary

Stage A is the simplified Isaac experiment in `EXECUTION_PROTOCOL.md`. Stage B is the formal target platform. Stage A must export portable task definitions, scenario manifests, outcome provenance, model artifacts, and evaluation records. Stage B must consume the same task semantics and expose enough evidence to compare behavior and end-to-end cost. No Stage B behavior is inferred from the v1 Isaac scene or from the names `G` and `GWM`.

## Required portable interfaces

1. **Scenario manifest:** schema version, episode/pair IDs, seed, background and ROI geometry, region boundaries, target counts/types, platform IDs/types/home regions, task eligibility, initial task queue, and immutable exogenous input references. Units must be explicit and convertible.
2. **Observation:** keep public observations and hidden simulator state in physically separate schemas/objects. Policy/model entry points receive only the public observation object; hidden state is available only to evaluator/logging code. Episode/pair, entity, task, and target IDs are metadata for joins and pairing and must never enter neural feature tensors. Preserve timestamp, type/region, observed task/target state, age/confidence, communication provenance, and label certainty/window status. Missing data retains separate `label_status=unknown` and/or `window_status=empty` fields.
3. **Action:** actor ID/type, action time, task/target reference, action kind (`search`, `abstract_strike_select`, or platform-supported equivalent), and validation result. Stage B adapters may translate the action representation but must preserve semantics and eligibility rules.
4. **Transition/event:** before/after state references, simulator time and wall time, event cause, separate `decision_class` when a decision occurred, `label_status`, `window_status`, and evidence pointer. `forced` denotes only a safety-critical decision that bypasses the gate; it is not an event-cause label. Do not convert forced decisions into policy-discovered outcomes.
5. **Cost trace:** trigger and commit boundaries plus durations for observation, feature extraction, inference, graph construction, planner/optimizer, communications, action validation/repair, and platform execution. Both wall-clock and simulated-time totals are required.
6. **Model artifact:** model family and version, feature/graph schema, training and validation split hashes, training seeds, code/config hashes, checkpoint hash, normalization/vocabulary parameters, and inference signature. Identify the one-step WM target and predicted next-state fields. For `GWM`, record the frozen WM checkpoint and auxiliary critic: WM outputs are stop-gradient inputs; auxiliary gradients update the critic, not the actor directly or frozen WM, and do not alter trigger rules. Gate/trigger condition is an independent evaluation factor.

## Invariants across migration

- The task remains search or search plus abstract strike selection. Any Stage B implementation that invokes weapon, projectile, ballistic, or damage control is outside this contract and requires a new protocol.
- There are nine regions and 36 platforms in the specified study: two UAVs and two helicopters per region, totaling 18 UAVs and 18 helicopters. UAV cross-region travel is legal; helicopter movement and assignments remain in the home region. Task assignment is type-specific.
- ROI and target generation, task labels, split membership, and evaluation metrics remain frozen when comparing platforms. Any platform-caused limitation is logged as a limitation/failure, not silently fixed in the data.
- All ROI consequences and complete end-to-end decision cost remain in reports. Stage B may add platform-specific cost components, but may not remove Stage A components.
- Paired comparisons use equivalent saved states and exogenous inputs. Where exact simulator state cannot be transferred, document a predeclared state-matching tolerance and report the mismatch; do not call the pair identical.
- Train/validation/test boundaries are episode-level and immutable. Stage B adaptation or fine-tuning creates a new model lineage and cannot use the locked test set. Identifiers remain metadata, and hidden state is physically excluded from model inputs rather than mixed into features with a label-based restriction.

## Stage B acceptance evidence

Before making a migration claim, provide a versioned adapter and schema mapping; one end-to-end smoke episode with raw receipts; identity and hash of the actual formal platform/model; unit and frame/time mapping; task-eligibility and movement-invariant checks; explicit handling of missing observations and all three outcome labels; measured complete cost trace; paired-state reproducibility evidence; and a coordinator review tied to those artifacts. An adapter that only accepts requests or emits a plausible response is not evidence of operational execution.

## Current migration blockers

The v2 protocol has not yet generated task data, trained `G`/`GWM`, or frozen checkpoints. The formal platform/model and its execution interface are unspecified in the available v1 audit. Consequently no specific API, coordinate frame, action encoding, runtime, or Stage B result can be asserted here. Resolve those facts from the supplied formal platform artifacts before implementing or claiming Stage B; preserve this contract and record any necessary protocol amendment with rationale and date.
