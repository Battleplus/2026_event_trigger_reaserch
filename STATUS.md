# Status

Current snapshot: **V2 repair smoke passed; formal collection interrupted** for the Chapter 2 gain-gate study.

Engineering disposition: **INCOMPLETE_COLLECTION; STOP**. The repaired V2 smoke run passed its immutable gate: 221/221 historical safety cases resolved, 0 unresolved, 0 unknown, candidate execution and paired replay checks passed, and all 79 evidence files matched their recorded hashes. The subsequent formal collection was interrupted: final run `20261017T113558Z-e8a3fc53b999` has 52 outcome records, 53 scenario files, one scenario without an outcome, and truncated aggregate gzip footers. The R2 interruption has 7 outcome records and 8 scenario files. No completion or final resource receipt exists. The last observed final-run CPU value, 1890.72 seconds, is a lower bound only and is not a total.

Method disposition: **INCONCLUSIVE; TRAINING GATE FAILED**. The final run has 0 ordinary positive and 78 ordinary non-positive labels among 219 parseable rows; 101 forced-safety and 40 empty-window labels remain unknown. The registered 30-positive/30-non-positive ordinary-state gate is not met. No light model, test/OOD evaluation, world-model experiment, or PPO transfer is authorized.

The earlier R1 diagnostic remains historical evidence of the initial failure and is not overwritten. See [the final V2 smoke review](reviews/V2_SMOKE_REVIEW_20261017_R3.json), [the safety classification](reports/V2_SAFETY_CLASSIFICATION.md), and [the V1/R1 diagnostic](reports/P2_R1_COLLECTION_DIAGNOSTIC_20261017.md).

The interrupted outputs and [read-only audit](reports/V2_INTERRUPTED_COLLECTION_AUDIT_20261017.json) are retained as diagnostic evidence. Do not count the partial scenario as a completed instance or synthesize completion/resource receipts. Test and OOD data remain sealed.
