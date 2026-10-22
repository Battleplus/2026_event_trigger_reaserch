# Independent episode reducer source review

Reviewer: main coordinator, read-only. Implementer: GPT-6 Luna. This is role
separation; no claim of a named external model review or real Isaac execution.

Reviewed evaluate_episode.py SHA-256:
755fafe5e272211ba96d43b9e7c8347dee9cdb56e5853166d6c3c5eddd400682.

Ordered per-actor/task waypoint replay and explicit decision-finalization
coverage address the two earlier local findings. Discovery and energy are
recomputed from raw position geometry and configuration rates. Online phase
timing excludes simulator stepping and nested planner diagnostics.

Disposition: REPAIR / FULL EPISODE EVALUATION INCOMPLETE.

- Missing finalization marks the delay series unknown but does not invalidate
  the overall PASS field. Overall completeness must reflect this while keeping
  valid partial execution metrics.
- Absolute per-method online control costs and a comparable total remain
  absent. A single-method episode can expose absolute weighted components;
  pairwise subtraction belongs in method comparisons. Do not invent weights
  where the registered protocol is silent or confuse net increments with totals.
- Safety currently reports flags/event counts without independent stopped-
  position or task-type/home-zone checks. Zero logged violations cannot prove
  actual safety execution. Required checks must use the frozen step semantics.

Returned to Luna for implementation, without changing collection or data.
No real training, simulator episode, heldout evaluation or scientific verdict
is established by this review.

## Second repair readback

Verified evaluate_episode.py SHA-256
e8dcbcc86a02cee29ae5f910cd91aad496d7ab70d6080b01b3740253d7337422 and
runtime.py 02c5a7364ee857674c90655fc1e0958248eb655ba2410ae0d8e89c7a05c05513.
Finalization coverage now participates in overall validity. The reducer checks
position stationarity for fault/communication holds and the prior-frame energy
pause, plus task/type and helicopter home-zone constraints. Zone bounds are
persisted in the new runtime manifest. Absolute registered control components
are separated from paired increments.

Remaining cost finding: the absolute scalar charges planner CPU only although
online detection, projection and inference CPU are recorded. It therefore does
not establish complete end-to-end cost. Requested a distinct measured-online
subtotal using the user's CPU coefficient once across the disjoint phases,
retaining the frozen label score and explicitly listing unmeasured components.
No additional real run or repeated test suite was requested.

Disposition remains REPAIR for full cost evaluation; static safety and coverage
repairs are accepted only in their inspected scope.

## CPU scope repair readback

Verified evaluate_episode.py SHA-256
df1ee58d2da79b8e302f78afaaf5e60413cc390d8cfae4b046f6e8f9e09466cc.
The frozen planner-only subtotal is now clearly separated from a measured-online
CPU subtotal. The latter replaces the nested planner charge with the sum of
disjoint measured online phases at the user's 0.01/CPU-second coefficient.
Full end-to-end cost remains explicitly incomplete because operational sensing
and task updates still share a post-step interval with evaluation/output.

Disposition: STATIC_CPU_SCOPE_REPAIR_ACCEPTED; full method-cost acceptance is
not established. Luna is implementing actual separation of those operations
in the new episode runner, without touching the ongoing frozen collection.
Missing device identity also needs to remain unknown rather than implying
absence of GPU use. No real simulator result has been produced by this code.

## Operational timing split readback

Inspected runtime.py ef4a939e324351ed8e11ffba15ce9217b9647e4b02288c49e5abd029f78c63cf
and evaluate_episode.py ddf9a26d4ecb0c01343c720251c4fc38ead3004fbfb7ee60375cf811e57b147d.
The new runtime measures post-step operational state separately from row/counter
bookkeeping and raw artifact logging. Startup and backend-close are measured;
missing device evidence now stays unknown. These are source-level observations,
not actual timing measurements from a real run.

Returned a material cost correction: the measured execution resource subtotal
includes simulator/evaluator/logging CPU and cannot be used as method J. Keep
that resource ledger separate from four-phase online method CPU. Experiment
process import/output boundaries must not be confused with online deployment
cost. Also update receipt resource-scope text after backend-close measurement.
No additional framework, simulator episode or full test suite was requested.

## Final static cost-scope readback

Verified evaluate_episode.py
44e319a56c650843c60670da2465fe9283994371e60fbc92bdf4624cd8c9f95f;
runtime remains ef4a939e324351ed8e11ffba15ce9217b9647e4b02288c49e5abd029f78c63cf.
The four online phases alone enter the method CPU subtotal, replacing nested
planner CPU. Simulator/evaluator/evidence logging are retained only in a separate
resource ledger. Startup/load is reported separately. Missing device identity
and deployment GPU/transfer costs remain explicit unknowns. Final resource
observation scope now correctly includes backend close.

Disposition: STATIC_EPISODE_COST_SCOPE_ACCEPTED for these frozen sources.
No real Isaac episode or actual model artifact has exercised this new runtime
and reducer. This is not ENGINEERING_PASS, a complete method-cost acceptance,
or scientific support. Real validation/test/OOD and grouped statistics remain
pending, as do model selection and calibration.
