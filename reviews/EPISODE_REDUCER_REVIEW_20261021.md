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
