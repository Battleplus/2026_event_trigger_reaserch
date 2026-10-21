# Read-only review of the new closedloop_v1 adapter

Role: main coordinator; actual runtime model identifier is not exposed here.
Implementation agent: gpt-6-luna. This is role separation, not a claim of
cross-family independent review or GPT-6.1 Sol High review.

Decision: REPAIR. This review concerns new local code only, not the frozen
formal collection. No Isaac evaluation or trained-model performance is proved.

Inspected: closedloop_v1/decision.py, checkpoint.py, README.md; the actual
MLP save mapping in light_models_v1/train_models.py; frozen EventTriggerGate,
ordinary-event recording, public projection and backend positions interfaces.

Findings returned to Luna:

1. All trigger types are currently OR-combined. Explicit method modes are
   required to compare Existing-ET, periodic and risk rules independently.
2. Forced events still call the predictor before the planner. They must bypass
   the predictor entirely; local safety execution remains the caller's duty.
3. A set of event names removes same-type event multiplicity, unlike the frozen
   event counter. Safety events must not contribute to ordinary pending count.
4. Predicted gain is not seconds. Current timer excludes readback/features and
   must be labelled partial timing, not full online cost.

The MLP checkpoint keys/model dimensions match the existing training save code,
but no real checkpoint has yet been trained or validated. Tree serialization,
end-to-end episode execution, command-effective delay integration and scientific
comparisons remain incomplete. The adapter is not a mission-level evaluator.

Requested verification is limited to mode separation at 240 seconds, predictor
bypass on forced events, and threshold counting of two same-type events. No
historical test suite, real-data training or holdout execution was requested.

## Repair readback

Luna returned a revised frozen adapter. Main coordinator reread the source and
verified these SHA-256 identities:

- decision.py: `668d09d14674d849a2bfbefa519a3764c494886f904bf5527079abf0345d9ed9`.
- README.md: `f5201ab069bea249769a92cbd1b8b6e751993f48f2887ceeaf1b43d1756b305a`.

All four findings above are addressed in the inspected source: explicit method
modes, forced-event predictor bypass, ordinary-event multiplicity, and partial
timing with an unqualified predicted_gain field. Luna reports the three limited
interface checks passed. No independent real Isaac execution was performed.

Revised disposition: STATIC_ADAPTER_REVIEW_ACCEPTED, limited to this decision
boundary. End-to-end evaluation remains INCOMPLETE, scientific status unchanged.
This is not ENGINEERING_PASS for a full closed-loop runner or method support.

## Episode runner review

Luna subsequently delivered runtime.py, derived from the frozen run_capture.py
(source SHA-256 `9329e4eb69652ec1f8278e8a7de71953826261ae6c66eeb0601150fd52ea7693`).
Main coordinator inspected its diff and the affected execution/decision blocks.
Disposition: REPAIR; not approved for scientific evaluation.

Required repairs returned to Luna:

- Apply immediate fault/communication/energy safety state before planning;
  the delivered version plans first and applies local safety afterwards.
- Preserve isolated evaluator truth and its manifest identity so independent
  discovery geometry can be recomputed; prevent policy access via public inputs.
- Remove evaluation-runner rejection of heldout configurations. The runner does
  not train; forbidden_for_training does not mean forbidden_for_evaluation.
  Actual heldout execution still waits for frozen model/threshold selection.
- Time the complete decision boundary including public projection and record
  suppressed/nontriggered costs; use that total once for action delay and cost,
  with nested planner/feature timing treated as breakdown, not extra charges.

No actual Isaac episode or scientific comparison has been run with this new
runtime. The live formal acquisition remains unchanged and separate.

## Episode repair readback

Main coordinator reread the affected runtime blocks and verified:

- runtime.py SHA-256: `31a552efcdb974395590c1058797e92e32c326eebcf8cf9198c9f80d593266d4`.
- README.md SHA-256: `ceda1917f0e95d82e512cde833fc74ef9ccef57b66e3d533c22fcfbb096ddca3`.
- PROVENANCE.json SHA-256: `72210b9ebe7f8ddf7ed3c3c8d2f56373b5c66d919d4518e48514b88c6e51b85f`.

Immediate safety-state updates now precede projection/planner calls. Evaluator
truth is persisted and hashed separately. Heldout configuration rejection was
removed. One measured decision interval covers public projection through
adapter return and is accumulated for every tick, including suppressed gates;
the action delay uses this interval once, with one-second clock quantization.
The receipt explicitly excludes earlier per-tick event/state-update work from
this decision interval. Those costs still require full online accounting before
an end-to-end method claim. CPU-only checkpoint inference remains unverified
against a real trained artifact. No real simulator episode was run here.

Disposition: STATIC_EPISODE_REPAIR_ACCEPTED within the source-review scope.
Runtime engineering validation, complete cost accounting, scalar evaluation,
trained tree/MLP comparison, and independent test/OOD remain unproved. This
does not pass the scientific or full closed-loop engineering milestone.
