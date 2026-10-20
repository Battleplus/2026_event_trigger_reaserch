# V3.3 focused source audit (final)

**Date:** 2026-10-20  
**Auditor:** GPT-6 Astra ultra (fresh same-family reviewer; provisional)  
**Review type:** read-only source-focused final audit  
**Project:** `ch2_disaster_relief_n02_gain_gate_v33_civilian_search_rescue_36platform`

## Overall verdict: WARN — source checks pass with an R2 evidence limitation

The current frozen source contains the required unresolved-safety rejection/masking and logical-parent discovery wiring. The cooperative contract test demonstrates delayed dependent-rescue publication. The source also contains generic split handling, but `test_v33_scenario_contract.py` has no split case and does not assert repeated-step exactly-once behavior or an oracle-recomputed pair whose obligation set includes the revealed rescue. Therefore R2 is **source-supported but evidence-limited**. No scientific result, performance, safety-rate, smoke-completion, collection, training, or generalization claim is accepted.

The prior wrapper failure is preserved as diagnostic metadata only and is excluded from experiment data: `registry-000.failure.json:2-6` records `ModuleNotFoundError: No module named 'package.civilian.'`; `:1` says `completion_marker_written: false`; `:29-35` records a failed resource receipt; `:37-39` identifies the failed run/schema.

## Focused checks

### 1. Independent oracle rejects raw unresolved safety after re-signing

**Status: PASS (source and focused contract evidence).**

- `package/civilian/gain_oracle.py:559-586` derives unresolved reasons from every persisted continuation receipt; when a trace is on the valid cost path, a nonempty raw-derived unresolved receipt raises `GAIN_ORACLE_VALID_TRACE_HAS_UNRESOLVED_SAFETY` at `:579-586`.
- The valid path verifies pair hashes before branching (`:798-809`), validates both traces (`:821-835`), and `validate_gain_label` verifies the outer label hash before recomputation (`:894-910`). A producer cannot make the unresolved receipt acceptable merely by re-signing trace, pair, and label hashes.
- `package/tests/test_v33_scenario_contract.py:211-278` constructs a valid label, injects a live unresolved receipt at `:246-253`, updates the safety evidence hash at `:254-260`, updates trace/pair/label summaries and hashes at `:261-275`, and expects the oracle rejection at `:277-278`.

### 2. Producer masks unresolved live safety and preserves raw reason

**Status: PASS (source and focused contract evidence).**

- `package/civilian/v3_runner.py:189-215` applies and verifies the live safety disposition, records `safety_apply_receipt`, `safety_disposition`, and `unknown_reason`, and returns a NOOP continuation when unresolved (`:207-215`). The main trajectory path repeats the live application/verification (`:230-248`) and forces NOOP when unresolved (`:297-305`).
- `package/civilian/counterfactual_pairs.py:641-699` derives unresolved reasons from persisted receipts, stores the reason in `unknown_reason`, and sets branch `valid_mask` false (`:691-692`). Pair collection rejects an invalid branch and emits masked pair/cost/gain fields (`:1034-1075`), then carries the raw-derived reason list into the pair (`:1076-1084`) and label (`:1086-1113`).
- `package/tests/test_v33_scenario_contract.py:143-208` checks label/pair/branch masks, null gains/costs, retained `unknown_reason`, raw truth, and the unresolved live receipt. This test is source evidence; this audit did not execute tests.

### 3. Cooperative logical completion reveals dependent rescue once after delay

**Status: PASS for source semantics; exact-once test evidence is partial.**

- `package/civilian/event_environment.py:1978-1986` treats a logical split/cooperation parent as complete only when all manifest children are completed and uses the last child completion time.
- `:1988-2009` publishes each dependent task only on a newly completed logical inspection, schedules it with the original parent `inspection_id` and `report_delay`, tracks already scheduled task IDs, skips already discovered tasks, and receives due reports.
- The step path invokes publication before raw-event capture (`:2315-2339`), preserving causal events. Report generation stores delay and parent identity (`package/civilian/relief_env.py:457-504`); due reports are consumed once (`:680-684`), and task discovery sets `known`, `discovered`, discovery time, and report receipt (`:540-550`).
- `package/tests/test_v33_scenario_contract.py:281-367` performs a cooperative split, verifies one child is complete while the rescue remains hidden (`:337-351`), advances after logical completion, verifies the delayed rescue becomes public (`:352-355`), and checks parent identity and causal report timing (`:356-367`). It does not count generated/received events after repeated NOOP steps.

### 4. Split support and evidence limits

**Status: WARN / EVIDENCE LIMITATION.**

- The source supports both operations. `package/civilian/event_environment.py:1596-1655` creates two child tasks for `split` or `cooperate`, records `parent_task_id`, `mode`, `child_ids`, and shares in `_split_manifests`, and marks the parent `split_parent` (`:1646-1652`). The same generic logical-completion/publication helpers cover either manifest mode (`:1978-2009`).
- The current focused test file contains one cooperative test (`test_v33_scenario_contract.py:281-367`) and no split operation/test (`rg` over the file finds no `op: "split"` or split test). It also has no paired raw trace/label/oracle recomputation tied to the revealed rescue’s obligation set.
- Consequently, split support is a **source-level provisional PASS**, while split/exactly-once/oracle closure is **not evidenced**. This limitation must remain visible even if source inspection suggests the path is correct.

### 5. No custom runtime cutoff in current source

**Status: PASS.**

- `package/civilian/v3_runner.py:183-230` drives the trajectory until the environment reports `terminated` or `truncated`; `:299-310` records each environment step. The only protocol time bounds are the simulator/label horizons (`:339-356`) and the scenario scale check requiring a 90-minute horizon (`:463-469`).
- Resource handling records wall time, process CPU, and RSS at receipts/failures (`:398-424`, `:757-783`); it does not terminate on those measurements. The config explicitly states `no automatic time or memory termination` (`configs/GAIN_GATE_CONFIG.json:72-80`).
- No custom `cutoff`, `timeout`, wall-time, CPU-time, or RSS termination branch was found in the audited runner/environment source. Deadline/horizon checks are simulation semantics, not a host-runtime cutoff.

## Evidence and claim boundaries

- The failed wrapper receipt at `evidence/v33-focused-checks/raw-smoke-first/raw-first-smoke-20261020-01/failures/registry-000.failure.json` is a failed execution record only; it is not experiment data and does not establish a smoke result.
- Existing `reviews/V33_SOURCE_AUDIT.md/.json` and `reviews/V33_SOURCE_AUDIT_FOLLOWUP.md/.json` were read as prior context. Their previous R1/R2 findings are superseded for the repaired oracle/source paths but remain relevant for the explicit split-test and primary raw-evidence limitations.
- This audit is simulation-only source review. It does not certify a scientific result or any 30/30 gate, model training, safety percentage, performance comparison, or generalization statement.

## Audited input hashes (SHA-256)

- `package/civilian/gain_oracle.py`: `sha256:604bc620685a971d71bf9e922b805e7cd48b820196e90aeabff7554af59a284b`
- `package/civilian/counterfactual_pairs.py`: `sha256:139c07d332bc35d64af1c105a9044bcd7c122ff5f8e42a22c9666d1ffb7a4397`
- `package/civilian/event_environment.py`: `sha256:385b755b0ff5bde29ffb34a3dfdbc81862153e41d07cbf74bf11f5907caea407`
- `package/civilian/v3_runner.py`: `sha256:c4542a7f0a7f47e4a1108fa5142ffa004e4ba6f1d087401925b19eb08b957565`
- `package/tests/test_v33_scenario_contract.py`: `sha256:4be2108dcb998fb8d212ffaef6d10794194ba4e6b8982559cdb737f6feee5ad1`
- `configs/GAIN_GATE_CONFIG.json`: `sha256:08c9c523f3301c1235cc35c7d87bcfb04b9afdda4d0578a6a06061960ed85859`
- `reviews/V33_SOURCE_AUDIT.md`: `sha256:980d7fff23c3469a5625345c3d48ebcd9d5fb1405a6542f63156d0ab5d3ac4d1`
- `reviews/V33_SOURCE_AUDIT.json`: `sha256:9d41d4ff4bc36fb49688ff09c379e9c2017109bbfd5df84637d0af20f01de4e2`
- `reviews/V33_SOURCE_AUDIT_FOLLOWUP.md`: `sha256:511a7427c46f0fb9127ae7c33062d4aa9ced33152be466cdfc7a77d52df418bf`
- `reviews/V33_SOURCE_AUDIT_FOLLOWUP.json`: `sha256:83b2a36c94eaec853592d2462b638c131a68fd0f036e16e80fdc3e383d1f8b81`
- `evidence/v33-focused-checks/raw-smoke-first/raw-first-smoke-20261020-01/failures/registry-000.failure.json`: `sha256:0963e50fe2fb7e576ce537277f48a61331ee41c449d8019cbf12686020d60ec3`

**Reviewer metadata:** `auditor=gpt-6-astra-ultra`; `reviewer_family=openai`; `review_independence=same-family`; `acceptance_status=provisional`; `simulations_run_by_reviewer=0`; `tests_run_by_reviewer=0`.
