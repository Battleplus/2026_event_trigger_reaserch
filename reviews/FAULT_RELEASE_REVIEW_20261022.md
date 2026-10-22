# Fault release repair: scoped source review

Decision: ACCEPT_SOURCE_REPAIR. No Isaac safety, dataset, training or method PASS is established.

Implementation: GPT-6 Luna. Reviewer: main Codex coordinator, read-only source review; no specific GPT-6.1 Sol High or cross-family independence claim.

Frozen files and SHA-256:

- runtime.py: aefbb30ab96e33e61a08b10b84ec3ffdf8c1234502c87a7da100bb344951101a
- evaluate_episode.py: 86c1b107c87c18d8cdb2a40bbceac8bcedd70cf9a26fdd83970de4418305d9e6
- test_forced_safety_evidence.py: ec9b433f31ecabae13961435cbf3a4ab38b6a961f2da78a2dc1d7b35104fdec3

Local file hashes were independently read and matched delivery. Luna reports eight focused synthetic tests and relevant compilation passed. The coordinator inspected the added queue mutation, ownership and missing-legality cases rather than rerunning unrelated tests.

## Accepted change

Fault processing excludes failed, paused and disconnected recipients. Unrecovered tasks lose stale mission assignment and produce an explicit failure event. Fault events retain execution queues/cursors and ownership before/after release, separately from planning projections. The reducer reconstructs unfinished tasks from queue/cursor, checks remaining owners, availability, assignment consistency and task legality, and reconciles failure events. Missing legality or queue evidence remains UNKNOWN. A stationary pause alone cannot establish complete forced processing. A synthetic retained-task/stale-owner corruption is rejected.

The new runtime identity is `closedloop_v1_fault_release_accounting_v2`. This changes execution semantics; old paired labels are not automatically valid for this runtime. Existing live collection, raw evidence and labels were not modified. No new Isaac run or neural training occurred.

## Limits

The current fault target is a UAV. This repair does not prove arbitrary helicopter fault recovery or all fault families. Reported safety fields cover the configured mechanisms and observed window only. Source and synthetic evidence do not prove physical execution of the repair.

Unrecovered task counts are reported independently in `one_unrecovered_public_task` units. They are not an added weighted term in frozen J_exec/J_total. Existing unresolved-work consequences remain in that frozen objective. Do not claim a new complete failure-cost score or add the same consequence again without a separately frozen accounting protocol.

Next real use requires a new run identity and corresponding paired-label compatibility. The ongoing 168-scene batch remains on its frozen collector; its final independent reduction and 30/30 decision are still pending.
