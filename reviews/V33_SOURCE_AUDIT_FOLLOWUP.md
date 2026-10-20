# V33 source audit follow-up (R1/R2 impact review)

**Verdict: REPAIR.** This is a source-readiness follow-up to the initial `REPAIR` report. The initial report and its input hashes are preserved unchanged. This review is limited to the changed R1/R2 paths; R3 remains open because the new primary raw pair evidence has not yet been supplied. No simulation or test was run by this reviewer.

Metadata: auditor **GPT-6 Astra ultra**; reviewer model `gpt-6-astra`; reasoning `ultra`; `review_independence: same-family`; `acceptance_status: provisional`; `review_type: source_readiness_followup`; `source_frozen: false`; `simulations_run_by_reviewer: 0`; `tests_run_by_reviewer: 0`.

## R1 — unresolved safety masking

**Production path: partial PASS.** The runner now applies the live safety hook, verifies the refreshed public state, records `safety_disposition` and `unknown_reason`, and forces NOOP when the disposition is unresolved (`package/civilian/v3_runner.py:200-215`, `:240-262`, `:297-305`). The paired collector derives unresolved reasons from both the disposition and live receipt, stores raw receipts and raw truth, and sets branch `valid_mask` false (`package/civilian/counterfactual_pairs.py:641-692`). Pair aggregation masks gain/cost when either branch is invalid (`package/civilian/counterfactual_pairs.py:1034-1087`). The shard summary requires label and oracle validity before counting a state (`package/civilian/v3_shards.py:44-68`). The focused contract test checks masked label/pair/branches, retained raw truth/receipts, and null gain (`package/tests/test_v33_scenario_contract.py:140-208`).

**Independent-oracle closure: REPAIR.** The oracle validates receipt schemas and derives unresolved reasons from persisted raw receipts (`package/civilian/gain_oracle.py:191-233`, `:559-608`). For a pair already marked invalid it independently checks that traces, pair validity, gain, count, and reason are masked (`package/civilian/gain_oracle.py:236-276`, `:793-804`). However, when a malicious or stale producer marks the same unresolved traces and pair as valid, `recompute_gain_label` proceeds through `_validate_trace` and cost recomputation (`package/civilian/gain_oracle.py:805-868`); its unresolved summary check at `:831-846` only compares recorded fields and does not reject a nonempty raw-derived unresolved reason. Thus the independent oracle can still return a valid gain if all validity flags and hashes are coherently forged around an unresolved receipt. Add an explicit rejection in the valid path whenever either trace yields unresolved safety, plus a test that mutates a masked label/pair into a hash-consistent “valid” shape and proves the oracle rejects it. Also test a live-receipt-only `resolved:false` case and verify it cannot enter positive/non-positive counts.

## R2 — split/cooperative dependent rescue discovery

**Core source semantics: provisional PASS.** `_logical_completed` now treats a split/cooperative parent as complete only when every manifest child is completed, and uses the last child completion time (`package/civilian/event_environment.py:1978-1986`). `_publish_completed_logical_discoveries` runs only on a newly completed logical inspection, schedules the dependent rescue with the original logical parent as `inspection_id`, applies `report_delay`, and deduplicates already scheduled/discovered work (`package/civilian/event_environment.py:1988-2009`). The helper is invoked in the step path before raw events are captured (`package/civilian/event_environment.py:2315-2339`). Report generation/receipt preserve the parent identity and due time (`package/civilian/relief_env.py:480-504`, `:578-584`, `:680-684`).

**Evidence/test closure: REPAIR.** The new contract test covers cooperation: one completed child does not reveal the rescue, all children complete, the rescue appears after one-minute delay, and the raw logical-completion/report causal IDs agree (`package/tests/test_v33_scenario_contract.py:211-298`). It does not cover a split operation, assert generation and receipt each occur exactly once after subsequent NOOP steps, or independently recompute a paired gain/oracle from the resulting raw traces. Add split and cooperate cases with partial completion, full completion, delay, exactly-once generation/receipt, and an oracle-valid paired label whose obligation set includes the revealed dependent rescue.

## R3 — primary raw pair evidence

**OPEN / not reviewed to closure.** The requested production raw pair/label evidence path was not supplied in this follow-up. The initial focused artifact remains invalid and non-replayable; no new evidence hash is asserted here.

## Audited changed-input hashes

SHA-256 values were computed from the files actually read during this follow-up; these are review inputs, not a release freeze:

```text
package/civilian/v3_runner.py                 sha256:c4542a7f0a7f47e4a1108fa5142ffa004e4ba6f1d087401925b19eb08b957565
package/civilian/counterfactual_pairs.py      sha256:139c07d332bc35d64af1c105a9044bcd7c122ff5f8e42a22c9666d1ffb7a4397
package/civilian/gain_oracle.py               sha256:51dec7d7cab71055b0ae45a11575f88500fc6bb95517fdf0d68a031b5a06c2df
package/civilian/event_environment.py         sha256:385b755b0ff5bde29ffb34a3dfdbc81862153e41d07cbf74bf11f5907caea407
package/civilian/relief_env.py                sha256:e7a7e58ebc818c2ba482e7fe4c74ec09fc35b3e9382a20e65c95278c3d527c80
package/civilian/gain_safety.py               sha256:fbd937b2b3d3bc4400fb25bab3dbd5c3f27ed619488468687e0c6e0abeed139c
package/civilian/v3_shards.py                 sha256:2dddcafdb79470251c82e9804f50e4dba46e34ab67c7d79fcb103fb9014d33c5
package/tests/test_v33_scenario_contract.py   sha256:b998957aed2d1a679c0a8ef19a125c62465c61f324251b3c1ac35fc1eb67e7dc
```

This follow-up does not accept smoke, collection, training, performance, or cancellation claims.
