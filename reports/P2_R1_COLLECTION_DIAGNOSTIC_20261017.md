# P2 R1 Collection Diagnostic

Status: the R1 collector completed, but the expanded run is **REPAIR_REQUIRED** because it has 221 unresolved safety dispositions and the actual simulator safety-control-link fallback is unmet. The gain signal is also too sparse for the registered light-model entry criterion, so light-model training was not started. The method result remains **INCONCLUSIVE**; this run does not disprove the method.

## Revision and identity

R1 changed only the pilot cap to 12 instances per condition in each of the train and validation splits. Scenario generation, event tapes, scoring weights, costs, replanner, label horizon, and collection code were unchanged. The full configured counts and test split were not run.

| Item | Value |
|---|---|
| Run | `20261017T074054Z-0c1d79d1d75a` |
| Configuration SHA-256 | `6586ed2c5fcaf2f839a5d10d09aa2b6b34a2975141488218e56b23dc1e1a695b` |
| Source manifest SHA-256 | `3d89082975ea5c1d9c6b3ac9de7657608a5a2e76febf332c29800b6c4e026382` |
| Collection report SHA-256 | `c26633752c4ce3d3aa0d8c0c67242410ad0a07da1c153d1f73ddeed35dfcf8ab` |
| Rows SHA-256 | `2ea96129e28e0daa4784c306ac71c37543657456bb32b207e9cd96347a1f4599` |
| Scenario outcomes SHA-256 | `31e75f76a78a94df23a9f81e99ca66a1881b4ce923264825931aab1f20034878` |
| Evidence manifest SHA-256 | `fdc92c9519952b76940f5929fe0d2f1162056669ed19f0852b76ee27030b19d7` |

The coordinator independently verified all 3,100 entries in the evidence manifest with no hash mismatches. The controlling read-only review is `reviews/P2_R1_REVIEW_20261017.json` (SHA-256 `ffebbf70050c49f7bd2b4ce125e2a74223dfb5b69631bb3c9a2752303175dcc6`). Raw run artifacts remain in the local execution workspace; this archive projection publishes their hashes and summary only.

The preceding 3-instance pilot had 67 valid ordinary labels and one positive. Its separate light-model attempt stopped before fitting with `GAIN_MODEL_NEED_TWO_VALIDATION_GROUPS_PER_FAMILY:energy_anomaly`. In R1, each condition has 2–4 scenario groups with valid ordinary rows in both train and validation; validation energy anomaly has four. This is a row-level coverage observation only. The light-model entry criterion still fails on positive-label count, so no model was fit and no model result is reported.

## Label distribution

| Measure | Count or value |
|---|---:|
| Scenario instances | 168 (84 train, 84 validation) |
| Rows | 731 |
| Valid ordinary paired labels | 243 |
| Positive ordinary gains | 2 (one train, one validation) |
| Non-positive ordinary gains | 241 |
| Forced safety rows, intentionally non-paired | 381 |
| Ordinary paired-collection failures, masked | 107 |
| Unresolved safety dispositions recorded | 221 |
| Valid ordinary labels with only one candidate, the NOOP | 229 / 243 |
| Valid ordinary labels whose selected action was NOOP | 238 / 243 |

Both positive labels came from `Periodic-4min`: one in `compound` and one in `task_update_or_arrival`. All 241 non-positive labels were also from that source. The selected rows from `Existing-ET` (139) and `Random-Allow-Suppress` (144) were all forced states; neither source contributed a valid ordinary label. The 107 ordinary pair records that failed remain unknown and masked as `paired_collection_failed`. Across suffix executions, 204 error occurrences had the exact message `counterfactual horizon contains no active logical obligations`. These are per-suffix errors, not 204 pair records; other suffixes may also have invalid masks, so the per-suffix error count does not classify every one of the 107 unknown pairs as a simulator exception.

Valid ordinary gain had mean `-0.03608`, median `-0.05031`, and range `[-0.18912, 2.29394]`. The two positives do not establish a learnable signal. In 229 states the fixed replanner had only a NOOP candidate. Since NOOP leaves execution unchanged, the two branches have nearly equal operational score; the replan branch then pays the registered reconstruction-call cost of `0.05` plus small measured costs, producing gains near `-0.05`. The positive labels came from two selected non-NOOP actions and are too few to meet the registered threshold.

The event-driven collection sources were dominated by forced safety/strong-event states in this sample. Periodic collection also exposed ordinary decision times, but increasing scenario instances alone did not provide enough positive labels or ordinary-label coverage across all three sources. This is a sampling and signal limitation, not evidence that the label validator should be relaxed.

## Resources and stage decision

The stage used `4,776.94` CPU seconds and `5,738.37` wall seconds, with peak RSS `419,475,456` bytes. The project ledger totals `7,553.94 / 86,400` CPU seconds after this run. The local collection payload occupies `221,533,346` bytes. CPU and CUDA settings were single-thread CPU with CUDA disabled. Resource limits were not reached.

The registered entry requirement is at least 30 positive and 30 non-positive independent ordinary states. R1 has only 2 positive states. Therefore:

- Collection execution: complete for this bounded R1 pilot. Engineering: **REPAIR_REQUIRED** because of the unresolved safety dispositions and unmet simulator safety-control-link fallback; this is not a P2 engineering pass.
- Light model: **not run**; do not relax sign-count or validation-group guards.
- Method: **INCONCLUSIVE**; no closed-loop method comparison was run.
- Event-aware world model: **STOPPED / NOT RUN under v1**.
- PPO transfer: **STOPPED / NOT RUN under v1**.
- Test and OOD: **not run**; no generalization claim is made.

The collection preserves zero, negative, masked, and forced labels under the existing contracts. It does not alter the gain equation, cost accounting, paired-suffix procedure, split grouping, or public-feature boundary.

R1 is the single authorized data revision in this protocol. No further data revision or dependent experiment is authorized under v1. The world-model and PPO branches remain stopped; restarting them requires a separately authorized protocol version after the engineering findings are repaired and reviewed.

## Archive projection

This report and `P2_R1_PUBLIC_EVIDENCE_RECEIPT_20261017.json` are safe summary artifacts for the public archive. They contain no absolute paths, credentials, row-level state identifiers, raw trajectories, or checkpoints. The 221 MB local run payload and 3,100-file evidence set are retained locally and are not embedded in the archive projection. The archive must retain the `REPAIR_REQUIRED` engineering and `INCONCLUSIVE` method status; it must not create a light-model success tag. No follow-up v1 collection is authorized.
